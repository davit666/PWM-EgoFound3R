"""Offline label generation in metres. Full triangles, not nearest vertices."""
import torch


def point_surface_distance(points, triangles, chunk_size=256):
    if points.ndim != 2 or points.shape[-1] != 3 or triangles.ndim != 3 or triangles.shape[1:] != (3, 3):
        raise ValueError("expected [N,3] points and [F,3,3] triangles")
    if not torch.isfinite(points).all() or not torch.isfinite(triangles).all():
        raise ValueError("invalid geometry")
    if len(triangles) == 0:
        raise ValueError("missing surface is not a negative-contact label")
    if chunk_size < 1:
        raise ValueError("chunk size must be positive")
    points, triangles = points.double(), triangles.double()
    result = []
    for block in points.split(chunk_size):
        best = torch.full((len(block),), float("inf"), device=points.device, dtype=points.dtype)
        for faces in triangles.split(chunk_size):
            a, b, c = faces.unbind(1)
            ab, ac = b - a, c - a
            ap = block[:, None] - a
            n = torch.linalg.cross(ab, ac)
            n2 = n.square().sum(-1)
            projection = ap - ((ap * n).sum(-1) / n2.clamp_min(1e-30))[..., None] * n
            d00, d01, d11 = (ab * ab).sum(-1), (ab * ac).sum(-1), (ac * ac).sum(-1)
            d20, d21 = (projection * ab).sum(-1), (projection * ac).sum(-1)
            denominator = d00 * d11 - d01.square()
            u = (d11 * d20 - d01 * d21) / denominator.clamp_min(1e-30)
            v = (d00 * d21 - d01 * d20) / denominator.clamp_min(1e-30)
            inside = (u >= 0) & (v >= 0) & (u + v <= 1) & (denominator > 1e-30)
            distance2 = torch.where(inside, (ap * n).sum(-1).square() / n2.clamp_min(1e-30), float("inf"))
            for x, y in ((a, b), (b, c), (c, a)):
                edge = y - x
                offset = block[:, None] - x
                t = ((offset * edge).sum(-1) / edge.square().sum(-1).clamp_min(1e-30)).clamp(0, 1)
                edge_distance = (offset - t[..., None] * edge).square().sum(-1)
                distance2 = torch.minimum(distance2, edge_distance)
            best = torch.minimum(best, distance2.min(-1).values)
        result.append(best.clamp_min(0).sqrt())
    return torch.cat(result).float() if result else points.new_empty(0).float()


def contact_from_distance(distance_m, point_kind):
    if point_kind not in ("joint", "vertex"):
        raise ValueError(point_kind)
    valid = torch.isfinite(distance_m) & (distance_m >= 0)
    threshold = 0.018 if point_kind == "joint" else 0.014
    return (distance_m < threshold) & valid, valid
