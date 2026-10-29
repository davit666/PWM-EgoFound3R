"""Per-window metrics; undefined subsets return None and are not counted as zero."""
import torch


def classification(logits, target, valid):
    valid = valid & torch.isfinite(target)
    if not valid.any():
        return {"precision": None, "recall": None, "f1": None, "count": 0}
    if not torch.isfinite(logits[valid]).all():
        raise FloatingPointError("nonfinite prediction")
    p, t = logits[valid] >= 0, target[valid].bool()
    tp = int((p & t).sum())
    fp, fn = int((p & ~t).sum()), int((~p & t).sum())
    return {"precision": tp / (tp + fp) if tp + fp else None,
            "recall": tp / (tp + fn) if tp + fn else None,
            "f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else None,
            "count": int(valid.sum())}


def distance_metrics(pred_m, target_m, contact_logits, target_contact, valid):
    valid = valid & torch.isfinite(target_m) & torch.isfinite(target_contact)
    if not torch.isfinite(pred_m[valid]).all() or not torch.isfinite(contact_logits[valid]).all():
        raise FloatingPointError("nonfinite distance/contact prediction")
    result = {}
    for name, subset in (("all", valid), ("pred_contact", valid & (contact_logits >= 0)),
                         ("gt_contact", valid & target_contact.bool())):
        result[name] = float((pred_m[subset] - target_m[subset]).abs().mean() * 1000) if subset.any() else None
    return result


def point_error_mm(pred, target, valid):
    valid = valid & torch.isfinite(target).all(-1)
    if not valid.any():
        return None
    if not torch.isfinite(pred[valid]).all():
        raise FloatingPointError("nonfinite geometry")
    return float((pred[valid] - target[valid]).norm(dim=-1).mean() * 1000)


def fit_similarity(pred, target, valid):
    """Least-squares proper Sim(3), row-vector convention; no reflection."""
    valid = valid & torch.isfinite(target).all(-1)
    x, y = pred[valid].double(), target[valid].double()
    if len(x) < 3:
        return None
    if not torch.isfinite(x).all():
        raise FloatingPointError('nonfinite alignment prediction')
    mx, my = x.mean(0), y.mean(0)
    x, y = x - mx, y - my
    # At least two independent directions are needed to determine rotation.
    if torch.linalg.matrix_rank(x) < 2 or torch.linalg.matrix_rank(y) < 2:
        return None
    u, singular, vh = torch.linalg.svd(x.T @ y)
    sign = torch.ones(3, device=x.device, dtype=x.dtype)
    sign[-1] = torch.linalg.det(u @ vh).sign()
    rotation = u @ torch.diag(sign) @ vh
    scale = (singular * sign).sum() / x.square().sum()
    shift = my - scale * mx @ rotation
    return scale, rotation, shift


def apply_similarity(points, transform):
    scale, rotation, shift = transform
    return (scale * points.double() @ rotation + shift).to(points.dtype)


def geometry_metrics(pred, target, valid, pred_root, target_root, root_valid):
    """One complete window [T,2,N,3]; PA fits both hands jointly per frame."""
    result = {'raw_mm': point_error_mm(pred, target, valid)}
    rr_valid = valid & root_valid[...,None]
    result['rr_mm'] = point_error_mm(pred - pred_root[...,None,:],
                                    target - target_root[...,None,:], rr_valid)
    aligned, pa_valid = pred.clone(), valid.clone()
    for frame in range(len(pred)):
        transform = fit_similarity(pred[frame], target[frame], valid[frame])
        if transform is None:
            pa_valid[frame] = False
        else:
            aligned[frame] = apply_similarity(pred[frame], transform)
    result['pa_mm'] = point_error_mm(aligned, target, pa_valid)
    transform = fit_similarity(pred, target, valid)
    result['sim3_mm'] = point_error_mm(apply_similarity(pred, transform), target, valid) if transform else None
    return result


def camera_to_world(points, w2c):
    rotation, translation = w2c[...,:3,:3], w2c[...,:3,3]
    return torch.einsum('tji,tsnj->tsni', rotation, points - translation[:,None,None,:])


def world_metrics(pred, target, valid, pred_w2c, target_w2c, camera_valid):
    """W fits first two valid frames; WA fits the entire window."""
    mask = valid & camera_valid[:,None,None]
    # Validate before transforms; invalid camera slots use identity placeholders.
    for camera in (pred_w2c, target_w2c):
        if not torch.isfinite(camera[camera_valid]).all():
            raise FloatingPointError('invalid active camera')
    eye = torch.eye(4, device=pred.device, dtype=pred.dtype)
    pred_world = camera_to_world(pred, torch.where(camera_valid[:,None,None], pred_w2c, eye))
    gt_world = camera_to_world(target, torch.where(camera_valid[:,None,None], target_w2c, eye))
    frames = mask.flatten(1).any(-1).nonzero().flatten()
    first = torch.zeros_like(mask)
    first[frames[:2]] = mask[frames[:2]]
    result = {}
    for name, fit_mask in (('w_mm', first), ('wa_mm', mask)):
        transform = fit_similarity(pred_world, gt_world, fit_mask) if len(frames) >= 2 else None
        result[name] = point_error_mm(apply_similarity(pred_world, transform), gt_world, mask) if transform else None
    return result


def temporal_metrics(pred, target, valid, timestamp, pair_valid=None):
    """Forward differences in seconds; acceleration at interval midpoints."""
    if timestamp.ndim != 1 or len(timestamp) != len(pred) or not torch.isfinite(timestamp).all():
        raise ValueError('timestamp must be finite [T]')
    dt = timestamp.diff()
    if (dt <= 0).any():
        raise ValueError('timestamps must increase')
    result = {'velocity_mm_s': None, 'acceleration_m_s2': None}
    if len(pred) < 2:
        return result
    valid = valid & torch.isfinite(target).all(-1)
    pairs = valid[1:] & valid[:-1]
    if pair_valid is not None:
        pairs &= pair_valid[:,None,None]
    error = torch.where(valid[...,None], pred - target, 0)
    velocity_error = error.diff(dim=0) / dt[:,None,None,None]
    if pairs.any():
        result['velocity_mm_s'] = float(velocity_error[pairs].norm(dim=-1).mean() * 1000)
    triplets = pairs[1:] & pairs[:-1]
    if triplets.any():
        acc = velocity_error.diff(dim=0) / ((dt[1:] + dt[:-1]) / 2)[:,None,None,None]
        result['acceleration_m_s2'] = float(acc[triplets].norm(dim=-1).mean())
    return result


def average_windows(records):
    """Macro average, with an explicit denominator for every scalar metric."""
    keys = set().union(*(r.keys() for r in records)) if records else set()
    result = {}
    for key in sorted(keys):
        values = [r[key] for r in records if r.get(key) is not None]
        result[key] = {'mean': sum(values)/len(values) if values else None, 'windows': len(values)}
    return result


def depth_metrics(pred, target, valid, *, fit_scale=False):
    valid = valid & torch.isfinite(target) & (target > 0)
    if not valid.any():
        return {k: None for k in ('abs_rel','rmse_m','delta1','scale_ratio')}
    p, t = pred[valid].double(), target[valid].double()
    if not torch.isfinite(p).all() or (p <= 0).any():
        raise FloatingPointError('depth prediction must be finite positive')
    ratio = torch.quantile(t/p, .5)
    if fit_scale:
        p = p * ratio
    return {'abs_rel': float(((p-t).abs()/t).mean()),
            'rmse_m': float((p-t).square().mean().sqrt()),
            'delta1': float((torch.maximum(p/t,t/p) < 1.25).double().mean()),
            'scale_ratio': float(ratio)}


def camera_metrics(pred_w2c, target_w2c, valid):
    from .scene import rotation_angle
    if not valid.any():
        return {'ate_m': None, 'rotation_deg': None}
    p, t = pred_w2c[valid].double(), target_w2c[valid].double()
    if not torch.isfinite(p).all() or not torch.isfinite(t).all():
        raise FloatingPointError('nonfinite camera')
    pc, tc = torch.linalg.inv(p), torch.linalg.inv(t)
    centers_p, centers_t = pc[:,:3,3], tc[:,:3,3]
    transform = fit_similarity(centers_p, centers_t, torch.ones(len(p),dtype=torch.bool,device=p.device))
    ate = float((apply_similarity(centers_p,transform)-centers_t).square().sum(-1).mean().sqrt()) if transform else None
    relative_p = p @ pc[:1]
    relative_t = t @ tc[:1]
    rot = float(torch.rad2deg(rotation_angle(relative_p[:,:3,:3],relative_t[:,:3,:3])).mean())
    return {'ate_m': ate, 'rotation_deg': rot}
