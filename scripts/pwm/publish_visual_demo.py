#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path


EXAMPLE_TITLES = {
    "arctic_scissors": "ARCTIC · scissors use",
    "taco_dust_roller": "TACO · dust / roller / pan",
    "stera_workbench": "STERA-10M · continuous workbench capture",
}


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Publish compact metadata and a README for PWM visual examples."
    )
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--publish-dir", type=Path, required=True)
    parser.add_argument("examples", nargs="+")
    args = parser.parse_args()

    output_root = args.output_root.resolve()
    publish_dir = args.publish_dir.resolve()
    publish_dir.mkdir(parents=True, exist_ok=True)

    examples: list[dict[str, object]] = []
    for name in args.examples:
        result_dir = output_root / name
        summary_path = result_dir / "summary.json"
        validation_path = result_dir / "validation.json"
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        validation = json.loads(validation_path.read_text(encoding="utf-8"))
        if not validation.get("ok"):
            raise ValueError(f"validation failed for {name}")
        examples.append(
            {
                "name": name,
                "title": EXAMPLE_TITLES.get(name, name),
                "frames": int(validation["frames"]),
                "clips": int(validation["clips"]),
                "source_video": summary["provenance"]["video_path"],
                "remote_result_dir": str(result_dir),
                "forward_seconds": summary["performance"]["forward_seconds_total"],
                "peak_reserved_gib": summary["performance"]["forward_peak_reserved_gib"],
                "validation": validation,
            }
        )

    manifest = {
        "schema_version": "pwm.egofound3r.visual_demo.v1",
        "paper_protocol": {
            "clip_seconds": 2.0,
            "global_stride": 5,
            "native_input_fps": True,
            "overlay_marker_count": 778,
            "modes": ["Geometry", "Visibility", "Contact", "Distance 0-50 mm"],
        },
        "examples": examples,
        "artifacts": {
            "gallery": "gallery.jpg",
            "montage": "montage.mp4",
        },
    }
    (publish_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    rows = [
        "# EgoFound3R paper-style visual examples",
        "",
        "These examples use the paper's default 2-second window and global stride `s=5`.",
        "The four panels are Geometry, Visibility, Contact, and metric contact Distance (0–50 mm).",
        "All examples use the official EgoFound3R and VGGT-Omega checkpoints with 778-vertex overlays.",
        "",
        "![Three-example gallery](gallery.jpg)",
        "",
        "[Play the concatenated video montage](montage.mp4)",
        "",
        "| Example | Frames | Forward time | Peak reserved VRAM |",
        "|---|---:|---:|---:|",
    ]
    for example in examples:
        rows.append(
            f"| {example['title']} | {example['frames']} | "
            f"{float(example['forward_seconds']):.3f} s | "
            f"{float(example['peak_reserved_gib']):.2f} GiB |"
        )
    rows.extend(
        [
            "",
            "Full tensors, summaries, validations, and source clips remain in each remote result directory recorded in `manifest.json`.",
            "",
        ]
    )
    (publish_dir / "README.md").write_text("\n".join(rows), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
