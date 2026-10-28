#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import importlib
import json
from pathlib import Path
import platform
import sys

from model_assets import ASSETS


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate PWM-EgoFound3R runtime inputs.")
    parser.add_argument("--video", type=Path)
    parser.add_argument(
        "--config", type=Path, default=REPO_ROOT / "configs/pwm_baseline.toml"
    )
    parser.add_argument("--allow-missing-weights", action="store_true")
    parser.add_argument("--require-cuda", action="store_true")
    parser.add_argument("--skip-weight-hash", action="store_true")
    args = parser.parse_args()

    report: dict[str, object] = {
        "schema_version": "pwm.egofound3r.preflight.v1",
        "ok": True,
        "ready_for_inference": True,
        "python": platform.python_version(),
        "config": str(args.config.resolve()),
        "dependencies": {},
        "weights": [],
        "video": None,
        "errors": [],
    }
    errors: list[str] = report["errors"]  # type: ignore[assignment]
    dependencies: dict[str, str] = report["dependencies"]  # type: ignore[assignment]
    modules = {
        "torch": "torch",
        "torchvision": "torchvision",
        "numpy": "numpy",
        "opencv": "cv2",
        "scipy": "scipy",
        "Pillow": "PIL",
        "einops": "einops",
        "safetensors": "safetensors",
        "huggingface_hub": "huggingface_hub",
    }
    imported: dict[str, object] = {}
    for label, module_name in modules.items():
        try:
            module = importlib.import_module(module_name)
            imported[label] = module
            dependencies[label] = str(getattr(module, "__version__", "unknown"))
        except Exception as error:
            errors.append(f"cannot import {module_name}: {type(error).__name__}: {error}")

    if sys.version_info < (3, 10):
        errors.append("Python 3.10 or newer is required")
    numpy_module = imported.get("numpy")
    if numpy_module is not None:
        major = int(str(getattr(numpy_module, "__version__", "0")).split(".", 1)[0])
        if major >= 2:
            errors.append("NumPy must be <2 for the vendored VGGT-Omega runtime")

    torch_module = imported.get("torch")
    cuda: dict[str, object] = {"available": False, "devices": []}
    if torch_module is not None:
        torch = torch_module
        cuda["available"] = bool(torch.cuda.is_available())  # type: ignore[attr-defined]
        if cuda["available"]:
            cuda["runtime"] = str(torch.version.cuda)  # type: ignore[attr-defined]
            cuda["devices"] = [
                {
                    "index": index,
                    "name": torch.cuda.get_device_name(index),  # type: ignore[attr-defined]
                    "memory_bytes": torch.cuda.get_device_properties(index).total_memory,  # type: ignore[attr-defined]
                }
                for index in range(torch.cuda.device_count())  # type: ignore[attr-defined]
            ]
    report["cuda"] = cuda
    if args.require_cuda and not cuda["available"]:
        errors.append("CUDA was required but is unavailable")

    try:
        from egohandmetric_prompt.configs import load_project_config

        config = load_project_config(args.config)
        report["project_name"] = config.project.default_name
        report["vggt_checkpoint_path"] = config.marker_runtime.vggt_checkpoint_path
    except Exception as error:
        errors.append(f"invalid project config: {type(error).__name__}: {error}")

    weights: list[dict[str, object]] = report["weights"]  # type: ignore[assignment]
    missing_weights = False
    for asset in ASSETS:
        path = REPO_ROOT / "checkpoints" / asset.filename
        row: dict[str, object] = {
            "path": str(path),
            "exists": path.is_file(),
            "expected_size_bytes": asset.size_bytes,
            "expected_sha256": asset.sha256,
        }
        if not path.is_file():
            missing_weights = True
            row["valid"] = False
        else:
            row["size_bytes"] = path.stat().st_size
            size_ok = path.stat().st_size == asset.size_bytes
            hash_ok = True
            if not args.skip_weight_hash:
                row["sha256"] = sha256_file(path)
                hash_ok = row["sha256"] == asset.sha256
            row["valid"] = size_ok and hash_ok
            if not row["valid"]:
                errors.append(f"checkpoint failed integrity validation: {path}")
        weights.append(row)
    if missing_weights:
        report["ready_for_inference"] = False
        if not args.allow_missing_weights:
            errors.append("one or more checkpoints are missing")

    if args.video is not None:
        video = args.video.resolve()
        video_report: dict[str, object] = {"path": str(video), "exists": video.is_file()}
        if not video.is_file():
            errors.append(f"video does not exist: {video}")
        elif "opencv" in imported:
            cv2 = imported["opencv"]
            capture = cv2.VideoCapture(str(video))  # type: ignore[attr-defined]
            readable = bool(capture.isOpened())
            frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))  # type: ignore[attr-defined]
            fps = float(capture.get(cv2.CAP_PROP_FPS))  # type: ignore[attr-defined]
            ok, frame = capture.read()
            capture.release()
            video_report.update(
                {
                    "readable": readable and bool(ok),
                    "frame_count": frame_count,
                    "fps": fps,
                    "first_frame_shape": list(frame.shape) if ok else None,
                }
            )
            if not readable or not ok:
                errors.append(f"OpenCV cannot decode the first frame: {video}")
        report["video"] = video_report

    report["ok"] = not errors
    print(json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True))
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
