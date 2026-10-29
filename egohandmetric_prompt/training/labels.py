"""Versioned adapter primitives. Inputs must already be synchronized/calibrated.

The caller establishes coordinate frames and annotation completeness. Unknown
contact partners or visibility must remain masked, never become negative labels.
"""
import torch
from .geometry import point_surface_distance, contact_from_distance
from egohandmetric_prompt.data.marker_vertices import MANO_MESHGRAPHORMER_LEVEL1_MARKER_VERTEX_IDS_195


def hand_geometry(joints_m, vertices_m, joint_valid, vertex_valid):
    if joints_m.shape[-2:] != (21,3) or vertices_m.shape[-2:] != (778,3):
        raise ValueError('adapter must convert to released MANO 21/778 order')
    ids = list(MANO_MESHGRAPHORMER_LEVEL1_MARKER_VERTEX_IDS_195)
    root = joints_m[..., :1, :]
    root_valid = joint_valid[..., :1] & torch.isfinite(root).all(-1)
    jmask = joint_valid & root_valid & torch.isfinite(joints_m).all(-1)
    vmask = vertex_valid[...,ids] & root_valid & torch.isfinite(vertices_m[...,ids,:]).all(-1)
    return {'joint_offset': (torch.where(jmask[...,None], joints_m - root, 0), jmask),
            'vertex_offset': (torch.where(vmask[...,None], vertices_m[...,ids,:] - root, 0), vmask)}


def interaction_labels(points_m, valid, *, object_triangles, opposite_hand_triangles,
                       partner_annotations_complete, point_kind):
    """A single hand/frame; union may contain multiple objects and opposite hand."""
    if not partner_annotations_complete:
        return None
    surfaces = [x for x in (object_triangles, opposite_hand_triangles) if x is not None and len(x)]
    if not surfaces:
        return None
    valid = valid & torch.isfinite(points_m).all(-1)
    distance = torch.zeros(points_m.shape[:-1], device=points_m.device)
    distance[valid] = point_surface_distance(points_m[valid], torch.cat(surfaces))
    contact, _ = contact_from_distance(distance, point_kind)
    return {f'{point_kind}_distance': (distance, valid),
            f'{point_kind}_contact': (contact.float(), valid)}


def localization_labels(uv_pixels, valid, image_hw, heatmap_hw, sigma=1.):
    """Gaussian heatmaps and local cell offsets, an explicit reconstruction choice.

    The released decoder uses integer cell coordinates and align_corners=False
    image scaling. uv_pixels is [T,2,21,2]; valid is [T,2,21].
    """
    ih, iw = image_hw
    hh, hw = heatmap_hw
    if sigma <= 0:
        raise ValueError('sigma must be positive')
    valid = valid & torch.isfinite(uv_pixels).all(-1)
    valid &= (uv_pixels[...,0] >= 0) & (uv_pixels[...,0] < iw) & (uv_pixels[...,1] >= 0) & (uv_pixels[...,1] < ih)
    uv = torch.where(valid[...,None], uv_pixels, 0)
    # See MarkerModel._decode_keypoints_from_localization: cell center -> pixels.
    xy = (uv + .5) * uv.new_tensor([hw/iw, hh/ih]) - .5
    yy, xx = torch.meshgrid(torch.arange(hh, device=uv.device),torch.arange(hw, device=uv.device), indexing='ij')
    delta = xy[...,None,None] - torch.stack((xx,yy)).to(uv)
    heatmap = torch.exp(-delta.square().sum(-3) / (2*sigma*sigma))
    heatmap_mask = valid[...,None,None].expand_as(heatmap)
    offset_mask = (delta.abs() < 1.5).all(-3) & heatmap_mask
    offset_mask = offset_mask.unsqueeze(-3).expand_as(delta)
    return {'heatmap': (heatmap, heatmap_mask),
            'offset2d': (torch.where(offset_mask, delta, 0), offset_mask),
            'reprojection': (uv, valid[...,None].expand_as(uv))}
