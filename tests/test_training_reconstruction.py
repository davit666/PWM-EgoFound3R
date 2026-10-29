import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch
from egohandmetric_prompt.training.data import PreparedClips
from egohandmetric_prompt.training.geometry import point_surface_distance, contact_from_distance
from egohandmetric_prompt.training.objectives import masked_loss, geometry_loss, objective
from egohandmetric_prompt.training.evaluation import classification, distance_metrics
from egohandmetric_prompt.training.evaluation import fit_similarity, apply_similarity, world_metrics, temporal_metrics
from egohandmetric_prompt.training.scene import rotation_angle, gather_scene_targets
from egohandmetric_prompt.multirate_contracts import build_multirate_frame_map
from egohandmetric_prompt.training.labels import localization_labels, interaction_labels
from egohandmetric_prompt.models.hand_keypoint_localization_head import decode_local_keypoint_cells


class ReconstructionTests(unittest.TestCase):
    def test_zero2_effective_batch_and_export_contract(self):
        from scripts.pwm.train_zero2 import zero_config, export_model
        from egohandmetric_prompt.configs import load_project_config
        from egohandmetric_prompt.checkpoints import load_marker_model_weights
        recipe=json.loads(Path('configs/training_reconstruction_example.json').read_text())
        settings=zero_config(recipe,8)
        self.assertEqual(settings['train_batch_size'],64)
        self.assertEqual(settings['train_micro_batch_size_per_gpu']*settings['gradient_accumulation_steps']*8,64)
        config=load_project_config(Path('configs/pwm_baseline.toml'))
        model=torch.nn.Linear(2,1).bfloat16()
        with tempfile.TemporaryDirectory() as d:
            checkpoint=Path(d)/'export.pt'
            export_model(model,config,checkpoint,1)
            loaded=torch.nn.Linear(2,1)
            load_marker_model_weights(loaded,checkpoint,model_config=config.marker_model,align_model_floating_dtype=True)
            torch.testing.assert_close(loaded.weight,model.weight)

    def test_localization_cell_convention_roundtrip(self):
        uv = torch.full((1,2,21,2), 63.5)
        labels = localization_labels(uv, torch.ones(1,2,21,dtype=torch.bool), (256,256),(32,32))
        heat, _ = labels['heatmap']
        offsets, mask = labels['offset2d']
        raw = torch.atanh((offsets / 1.5).clamp(-.999,.999))
        xy, *_ = decode_local_keypoint_cells((heat*40)[None],raw[None])
        recovered = (xy+.5)*8-.5
        torch.testing.assert_close(recovered[0],uv,atol=1e-4,rtol=0)

    def test_missing_partner_does_not_generate_negative_contact(self):
        result = interaction_labels(torch.zeros(2,3),torch.ones(2,dtype=torch.bool),object_triangles=None,
            opposite_hand_triangles=torch.zeros(1,3,3),partner_annotations_complete=False,point_kind='joint')
        self.assertIsNone(result)

    def test_training_checkpoint_loads_with_released_loader(self):
        from egohandmetric_prompt.configs import load_project_config
        from egohandmetric_prompt.checkpoints import load_marker_model_weights
        from scripts.pwm.train_prepared import save_checkpoint
        model = torch.nn.Module()
        model.backbone = torch.nn.Linear(2,2)
        model.head = torch.nn.Linear(2,1)
        optimizer = torch.optim.AdamW(model.head.parameters())
        model.head(torch.ones(1,2)).sum().backward()
        optimizer.step()
        expected = model.head.weight.detach().clone()
        config = load_project_config(Path('configs/pwm_baseline.toml'))
        with tempfile.TemporaryDirectory() as d:
            path = Path(d)/'latest.pt'
            save_checkpoint(path,model,optimizer,1,config,{},'test')
            with torch.no_grad():
                model.head.weight.zero_()
            loaded = load_marker_model_weights(model,path,model_config=config.marker_model)
            self.assertEqual(loaded,2)
            torch.testing.assert_close(model.head.weight,expected)
            checkpoint = torch.load(path,weights_only=False)
            self.assertEqual(checkpoint['step'],1)
            self.assertTrue(checkpoint['optimizer']['state'])

    def test_rotation_identity_gradient_and_pi(self):
        rotation = torch.eye(3).requires_grad_()
        loss = rotation_angle(rotation, torch.eye(3))
        loss.backward()
        self.assertEqual(float(loss.detach()), 0.)
        self.assertTrue(torch.isfinite(rotation.grad).all())
        self.assertAlmostEqual(float(rotation_angle(torch.diag(torch.tensor([1., -1., -1.])), torch.eye(3))), np.pi, places=6)

    def test_similarity_and_world_gauge(self):
        torch.manual_seed(3)
        gt = torch.randn(4,2,5,3)
        rotation = torch.tensor([[0.,-1.,0.],[1.,0.,0.],[0.,0.,1.]])
        p = 2 * gt @ rotation + 10
        valid = torch.ones(4,2,5,dtype=torch.bool)
        transform = fit_similarity(p, gt, valid)
        torch.testing.assert_close(apply_similarity(p, transform), gt, atol=1e-6, rtol=1e-5)
        pose = torch.eye(4).repeat(4,1,1)
        scores = world_metrics(p, gt, valid, pose, pose, torch.ones(4,dtype=torch.bool))
        self.assertLess(scores['w_mm'], .001)
        self.assertLess(scores['wa_mm'], .001)

    def test_temporal_uses_seconds_and_pair_mask(self):
        t = torch.tensor([0., .1, .2])
        p = t[:,None,None,None].expand(3,2,3,3)
        gt = torch.zeros_like(p)
        result = temporal_metrics(p, gt, torch.ones(3,2,3,dtype=torch.bool), t)
        self.assertAlmostEqual(result['velocity_mm_s'], 1000*3**.5, places=3)
        self.assertAlmostEqual(result['acceleration_m_s2'], 0.)

    def test_scene_labels_gather_same_anchors(self):
        present = torch.ones(1,9,dtype=torch.bool)
        mapping = build_multirate_frame_map(high_frame_present=present, high_temporal_pair_valid=present[:,:-1], global_stride=5, global_anchor_phase=2)
        target = torch.arange(9.).reshape(1,9,1)
        gathered = gather_scene_targets({'scene_depth': (target, torch.ones_like(target,dtype=torch.bool))}, mapping)
        torch.testing.assert_close(gathered['scene_depth'][0].squeeze(-1), mapping.global_anchor_indices.float())

    def test_scene_scale_invariance_backward(self):
        depth = torch.ones(1,2,3,3, requires_grad=True)
        K = torch.eye(3).repeat(1,2,1,1)
        pose = torch.eye(4).repeat(1,2,1,1)
        targets = {k: (v,torch.ones_like(v,dtype=torch.bool)) for k,v in
                   {'scene_depth': depth.detach()*2, 'scene_K': K, 'scene_pose': pose}.items()}
        outputs = {'depth_global': depth, 'depth_conf_global': torch.ones_like(depth),
                   'intrinsics_global': K, 'camera_pose_global': pose}
        total, terms = objective(outputs, targets, {k:1 for k in ['scene_depth','scene_K','scene_translation','scene_rotation']},confidence_alpha=.1)
        total.backward()
        self.assertLess(abs(float(total.detach())), 1e-6)
        self.assertTrue(torch.isfinite(depth.grad).all())

    def test_masked_nan_does_not_poison_gradient(self):
        p = torch.tensor([0., 2.], requires_grad=True)
        loss = masked_loss(p, torch.tensor([1., float('nan')]), torch.tensor([True, False]))
        loss.backward()
        torch.testing.assert_close(p.grad, torch.tensor([-1., 0.]))

    def test_bad_active_prediction_fails(self):
        with self.assertRaises(FloatingPointError):
            masked_loss(torch.tensor([float('nan')]), torch.ones(1), torch.ones(1, dtype=torch.bool))

    def test_empty_mask_is_not_zero_supervision(self):
        self.assertIsNone(masked_loss(torch.ones(2), torch.ones(2), torch.zeros(2, dtype=torch.bool)))
        with self.assertRaises(ValueError):
            objective({}, {}, {}, confidence_alpha=.1)

    def test_triangle_interior_and_edges(self):
        tri = torch.tensor([[[0., 0., 0.], [1., 0., 0.], [0., 1., 0.]]])
        points = torch.tensor([[.25, .25, .1], [1., 1., 0.], [-1., 0., 0.]])
        torch.testing.assert_close(point_surface_distance(points, tri), torch.tensor([.1, 2**-.5, 1.]))

    def test_degenerate_triangle_and_empty_surface(self):
        tri = torch.zeros(1, 3, 3)
        torch.testing.assert_close(point_surface_distance(torch.tensor([[1., 0., 0.]]), tri), torch.ones(1))
        with self.assertRaises(ValueError):
            point_surface_distance(torch.zeros(1, 3), torch.zeros(0, 3, 3))

    def test_paper_contact_thresholds(self):
        d = torch.tensor([.013, .014, .017, .018, float('nan')])
        self.assertEqual(contact_from_distance(d, 'vertex')[0].tolist(), [True, False, False, False, False])
        self.assertEqual(contact_from_distance(d, 'joint')[0].tolist(), [True, True, True, False, False])

    def test_geometry_scale_normalization(self):
        target = torch.tensor([[[.1, 0., 0.], [0., .2, 0.]]])
        p = (2 * target).requires_grad_()
        value = geometry_loss(p, target, torch.ones(1, 2, dtype=torch.bool), torch.zeros(1, 2), .1)
        self.assertAlmostEqual(float(value.detach()), 0.)
        value.backward()
        self.assertTrue(torch.isfinite(p.grad).all())

    def test_distance_log_mm_and_nan_mask(self):
        p = torch.tensor([0., 0.], requires_grad=True)
        total, _ = objective({'dense_vertex_contact_log_distance': p},
            {'vertex_distance': (torch.tensor([.001, float('nan')]), torch.tensor([True, True]))},
            {'vertex_distance': 1.}, confidence_alpha=.1)
        self.assertAlmostEqual(float(total.detach()), .5 * np.log(2)**2, places=6)
        total.backward()
        self.assertEqual(float(p.grad[1]), 0.)

    def test_metrics_empty_subsets(self):
        m = torch.ones(2, dtype=torch.bool)
        scores = classification(torch.tensor([1., -1.]), torch.tensor([1., 1.]), m)
        self.assertAlmostEqual(scores['f1'], 2/3)
        values = distance_metrics(torch.zeros(2), torch.ones(2)*.002, -torch.ones(2), torch.zeros(2), m)
        self.assertAlmostEqual(values['all'], 2.)
        self.assertIsNone(values['pred_contact'])
        self.assertIsNone(values['gt_contact'])

    def test_split_leak_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / 'index.jsonl'
            path.write_text('\n'.join(json.dumps(dict(dataset='test', source_episode_id='one', split=s, path='a.npz')) for s in ('train', 'val')))
            with self.assertRaises(ValueError):
                PreparedClips(path, 'train')

    def test_real_reader_contract(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            np.savez(root / 'a.npz', rgb=np.zeros((2,16,16,3), dtype=np.uint8),
                     timestamp=np.array([0., .1]), frame_present=np.ones(2,dtype=bool),
                     target__joint_offset=np.zeros((2,2,21,3),dtype=np.float32),
                     valid__joint_offset=np.ones((2,2,21),dtype=bool))
            (root / 'index.jsonl').write_text(json.dumps(dict(dataset='test',source_episode_id='one',split='train',path='a.npz')))
            sample = PreparedClips(root / 'index.jsonl', 'train')[0]
            self.assertEqual(tuple(sample['images'].shape), (2,3,16,16))


if __name__ == '__main__':
    unittest.main()
