"""Strict prepared-clip reader. Raw dataset conversion is a separate audited step."""
from __future__ import annotations

import json
import hashlib
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset


class PreparedClips(Dataset):
    """JSONL rows: dataset, source_episode_id, split, path (NPZ relative to index).

    NPZ: rgb uint8[T,H,W,3], timestamp float64[T], frame_present bool[T],
    and target__NAME / valid__NAME pairs. Clips already carry the H time grid.
    No pickle, implicit resizing, synchronization, or zero-filled supervision.
    """
    def __init__(self, index, split):
        self.index = Path(index).resolve()
        records = [json.loads(line) for line in self.index.read_text().splitlines() if line.strip()]
        seen = {}
        for row in records:
            identity = (row["dataset"], row["source_episode_id"])
            if identity in seen and seen[identity] != row["split"]:
                raise ValueError(f"source episode leaks across splits: {identity}")
            seen[identity] = row["split"]
        self.rows = [r for r in records if r["split"] == split]
        self.verified = set()
        if not self.rows:
            raise ValueError(f"empty {split} split")

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        path = self.index.parent / row["path"]
        if row.get('sha256') and str(path) not in self.verified:
            digest = hashlib.sha256()
            with path.open('rb') as file:
                for block in iter(lambda: file.read(1024*1024), b''):
                    digest.update(block)
            if digest.hexdigest() != row['sha256']:
                raise ValueError(f'prepared clip checksum mismatch: {path}')
            self.verified.add(str(path))
        with np.load(path, allow_pickle=False) as src:
            rgb = src["rgb"]
            time = src["timestamp"]
            present = src["frame_present"]
            if rgb.dtype != np.uint8 or rgb.ndim != 4 or rgb.shape[-1] != 3:
                raise ValueError("rgb must be uint8[T,H,W,3]")
            if min(rgb.shape[1:3]) < 16 or any(n % 16 for n in rgb.shape[1:3]):
                raise ValueError("image dimensions must be positive multiples of 16")
            if present.dtype != np.bool_ or present.shape != (len(rgb),) or not present.all():
                raise ValueError("v1 reader requires complete unpadded clips")
            if time.shape != present.shape or not np.isfinite(time).all() or not (np.diff(time) > 0).all():
                raise ValueError("timestamps must be finite and strictly increasing")
            targets = {}
            for name in src.files:
                if name.startswith("target__"):
                    key = name.removeprefix("target__")
                    value, mask = src[name], src[f"valid__{key}"]
                    if mask.dtype != np.bool_:
                        raise TypeError("target masks must be bool")
                    expected = value.shape[:-1] if key.endswith("_offset") else value.shape
                    if mask.shape != expected:
                        raise ValueError(f"mask shape mismatch: {key}")
                    if not np.isfinite(value[mask]).all():
                        raise ValueError(f"nonfinite active target: {key}")
                    targets[key] = (torch.from_numpy(value.copy()).float(), torch.from_numpy(mask.copy()))
        return {"images": torch.from_numpy(rgb.copy()).permute(0, 3, 1, 2).float() / 255,
                "timestamp": torch.from_numpy(time.copy()).double(),
                "targets": targets, "identity": row}
