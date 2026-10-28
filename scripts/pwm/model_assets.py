from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ModelAsset:
    repo_id: str
    filename: str
    size_bytes: int
    sha256: str
    terms_url: str


ASSETS = (
    ModelAsset(
        repo_id="Frank-F2021/EgoFound3R",
        filename="EgoFound3R.pt",
        size_bytes=423_526_586,
        sha256="02ba3135b030a23f88b91a986c173f6a3c600f6da45637605fd09914e9d78e66",
        terms_url="https://huggingface.co/Frank-F2021/EgoFound3R",
    ),
    ModelAsset(
        repo_id="facebook/VGGT-Omega",
        filename="vggt_omega_1b_512.pt",
        size_bytes=4_576_706_117,
        sha256="c02da418b18bb01d0392598d3f6147366bcde1bb70fd08a5e3bf7925b0667934",
        terms_url="https://huggingface.co/facebook/VGGT-Omega",
    ),
)
