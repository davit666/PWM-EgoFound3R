#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys

import torch


REQUIRED_TENSORS = (
    "camera_pose",
    "intrinsics",
    "joint_xyz",
    "vertex_xyz",
    "joint_contact_logits",
    "vertex_contact_logits",
)


def assert_finite_json(value: object, path: str = "summary") -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError(f"non-finite JSON value at {path}: {value}")
    if isinstance(value, dict):
        for key, child in value.items():
            assert_finite_json(child, f"{path}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            assert_finite_json(child, f"{path}[{index}]")


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate a PWM baseline output directory.")
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args()
    output_dir = args.output_dir.resolve()
    summary_path = output_dir / "summary.json"
    tensor_path = output_dir / "inference_output.pt"
    for path in (summary_path, tensor_path):
        if not path.is_file() or path.stat().st_size == 0:
            raise FileNotFoundError(f"missing or empty required output: {path}")

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert_finite_json(summary)
    for key in ("T_R", "clip_count", "inference_schema_version", "provenance"):
        if key not in summary:
            raise KeyError(f"summary is missing {key!r}")
    if int(summary["T_R"]) <= 0 or int(summary["clip_count"]) <= 0:
        raise ValueError("summary contains no inferred frames or clips")

    payload = torch.load(tensor_path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise TypeError("inference_output.pt must contain a dictionary")
    if payload.get("inference_schema_version") != summary["inference_schema_version"]:
        raise ValueError("schema version differs between tensor payload and summary")
    if int(payload.get("T_R", -1)) != int(summary["T_R"]):
        raise ValueError("T_R differs between tensor payload and summary")

    tensor_report: dict[str, object] = {}
    for key in REQUIRED_TENSORS:
        value = payload.get(key)
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"payload field {key!r} is missing or is not a tensor")
        if value.numel() == 0:
            raise ValueError(f"payload tensor {key!r} is empty")
        if value.is_floating_point() and not bool(torch.isfinite(value).all().item()):
            raise ValueError(f"payload tensor {key!r} contains NaN or infinity")
        tensor_report[key] = {"shape": list(value.shape), "dtype": str(value.dtype)}

    result = {
        "schema_version": "pwm.egofound3r.validation.v1",
        "ok": True,
        "output_dir": str(output_dir),
        "frames": int(summary["T_R"]),
        "clips": int(summary["clip_count"]),
        "inference_schema_version": summary["inference_schema_version"],
        "tensors": tensor_report,
    }
    print(json.dumps(result, indent=2, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(f"validation failed: {type(error).__name__}: {error}", file=sys.stderr)
        raise
