#!/usr/bin/env python3
"""Native-H validation for prepared clips, with window-macro attribute metrics.

This does not claim the authors' 2,378-window benchmark: use released inference
exports and matching ground-truth geometry for world/full-MANO benchmark scores.
"""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import torch
from egohandmetric_prompt.builders import build_runtime_marker_model
from egohandmetric_prompt.checkpoints import load_marker_model_weights
from egohandmetric_prompt.configs import load_project_config
from egohandmetric_prompt.multirate_contracts import build_multirate_frame_map
from egohandmetric_prompt.training.data import PreparedClips
from egohandmetric_prompt.training.evaluation import classification, distance_metrics, average_windows
from egohandmetric_prompt.training.objectives import objective
from egohandmetric_prompt.training.scene import gather_scene_targets
from scripts.pwm.train_prepared import native_training_outputs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('config','recipe','index','checkpoint','output'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--split', default='val')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--stride', type=int, default=5)
    args = parser.parse_args()
    config = load_project_config(args.config)
    recipe = json.loads(args.recipe.read_text())
    model = build_runtime_marker_model(config)
    loaded = load_marker_model_weights(model,args.checkpoint, model_config=config.marker_model, align_model_floating_dtype=True)
    model = model.float().to(args.device).eval()
    dataset = PreparedClips(args.index, args.split)
    rows = []
    with torch.inference_mode():
        for sample in dataset:
            images = sample['images'][None].to(args.device)
            targets = {k:(v[None].to(args.device),m[None].to(args.device)) for k,(v,m) in sample['targets'].items()}
            present = torch.ones(images.shape[:2],dtype=torch.bool,device=args.device)
            mapping = build_multirate_frame_map(high_frame_present=present,
                high_temporal_pair_valid=present[:,:-1], global_stride=args.stride, global_anchor_phase=min(2,args.stride-1))
            with torch.autocast(device_type=torch.device(args.device).type, dtype=torch.bfloat16, enabled=recipe['precision']=='bfloat16'):
                output = native_training_outputs(model(images,multirate_frame_map=mapping))
                total, terms = objective(output,gather_scene_targets(targets,mapping),recipe['loss_weights'],
                    confidence_alpha=recipe['confidence_alpha'],heatmap_positive_weight=recipe.get('heatmap_positive_weight',1.))
            metrics = {'loss':float(total), **{'loss/'+k:float(v) for k,v in terms.items()}}
            for side in ('joint','vertex'):
                for attr in ('contact','visibility'):
                    key = f'{side}_{attr}'
                    if key in targets:
                        t,m = targets[key]
                        scores = classification(output[f'dense_{side}_{attr}_logits'],t,m)
                        metrics.update({key+'/'+k:v for k,v in scores.items() if k != 'count'})
                if side+'_distance' in targets and side+'_contact' in targets:
                    d,dm = targets[side+'_distance']
                    c,cm = targets[side+'_contact']
                    prediction = torch.expm1(output[f'dense_{side}_contact_log_distance'].float()).clamp_min(0)/1000
                    scores = distance_metrics(prediction,d,output[f'dense_{side}_contact_logits'],c,dm&cm)
                    metrics.update({side+'/distance_mae_mm/'+k:v for k,v in scores.items()})
            rows.append({'identity':sample['identity'],'metrics':metrics})
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps({'loaded_state_keys':loaded,'split':args.split,
        'windows':rows,'macro':average_windows([r['metrics'] for r in rows]),
        'protocol':'prepared-native-H-validation; not paper benchmark'},indent=2))

if __name__ == '__main__':
    main()
