"""Reconstructed scene objective; scale estimator follows released inference.

All inputs here are on G. Adapters store H targets; gather_scene_targets performs
the sole H->G selection. Camera matrices are world-to-camera in metres.
"""
import torch
from egohandmetric_prompt.metric_geometry import scene_average_distance_from_depth
from egohandmetric_prompt.multirate_contracts import gather_time_axis, gather_time_mask


def gather_scene_targets(targets, mapping):
    return {k: (gather_time_axis(v, mapping.global_anchor_indices, mapping.global_frame_present),
                gather_time_mask(m, mapping.global_anchor_indices, mapping.global_frame_present))
            if k.startswith('scene_') else (v, m) for k, (v, m) in targets.items()}


def rotation_angle(pred, target):
    """atan2 SO(3) angle: finite gradients at identical rotations (unlike acos)."""
    relative = pred @ target.transpose(-1, -2)
    skew = torch.stack((relative[..., 2, 1] - relative[..., 1, 2],
                        relative[..., 0, 2] - relative[..., 2, 0],
                        relative[..., 1, 0] - relative[..., 0, 1]), -1) / 2
    cosine = (relative.diagonal(dim1=-2, dim2=-1).sum(-1) - 1) / 2
    return torch.atan2(skew.norm(dim=-1), cosine.clamp(-1, 1))


def scene_losses(outputs, targets, alpha):
    from .objectives import masked_loss
    result = {}
    if 'scene_K' in targets:
        K, kmask = targets['scene_K']
        result['scene_K'] = masked_loss(outputs['intrinsics_global'].float(), K, kmask)
    if not ({'scene_depth', 'scene_pose'} & targets.keys()):
        return result
    required = {'scene_depth', 'scene_K', 'scene_pose'}
    if required - targets.keys():
        raise ValueError('scale-normalized scene supervision requires depth, K and W2C together')
    depth, mask = targets['scene_depth']
    K, kmask = targets['scene_K']
    pose, pmask = targets['scene_pose']
    frame_valid = (kmask & torch.isfinite(K)).all(dim=(-2,-1)) & (pmask & torch.isfinite(pose)).all(dim=(-2,-1))
    # Gauge must use a real first anchor. Do not silently use an invalid reference.
    if not frame_valid[:, 0].all():
        raise ValueError('scene supervision requires valid first G camera')
    mask = mask & torch.isfinite(depth) & (depth > 0) & frame_valid[...,None,None]
    if not mask.flatten(1).any(-1).all():
        raise ValueError('each scene needs valid depth support for scale normalization')
    gt_depth = torch.where(mask, depth, 0).float()
    gt_K = torch.where(frame_valid[...,None,None], K, torch.eye(3, device=K.device)).float()
    gt_pose = torch.where(frame_valid[...,None,None], pose, torch.eye(4, device=pose.device)).float()
    pred_depth = outputs['depth_global'].float()
    if pred_depth.ndim == 5:
        pred_depth = pred_depth.squeeze(-1)
    pred_K, pred_pose = outputs['intrinsics_global'].float(), outputs['camera_pose_global'].float()
    if not all(torch.isfinite(x).all() for x in (pred_depth, pred_K, pred_pose)):
        raise FloatingPointError('nonfinite scene prediction')
    a = scene_average_distance_from_depth(pred_depth, pred_K, pred_pose, mask)
    b = scene_average_distance_from_depth(gt_depth, gt_K, gt_pose, mask)
    residual = (pred_depth / a[:,None,None,None] - gt_depth / b[:,None,None,None]).abs()[mask]
    confidence = outputs['depth_conf_global'].float()[mask]
    if not torch.isfinite(confidence).all() or (confidence <= 0).any():
        raise FloatingPointError('depth confidence must be finite and positive')
    result['scene_depth'] = (confidence * residual - alpha * confidence.log()).mean()
    # Express both trajectories in their first camera to remove world gauge.
    pred_local = pred_pose @ torch.linalg.inv(pred_pose[:, :1])
    gt_local = gt_pose @ torch.linalg.inv(gt_pose[:, :1])
    result['scene_translation'] = masked_loss(pred_local[...,:3,3] / a[:,None,None],
        gt_local[...,:3,3] / b[:,None,None], frame_valid[...,None].expand(-1,-1,3))
    result['scene_rotation'] = rotation_angle(pred_local[...,:3,:3][frame_valid],
                                             gt_local[...,:3,:3][frame_valid]).mean()
    return result
