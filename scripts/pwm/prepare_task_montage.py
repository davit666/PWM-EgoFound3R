#!/usr/bin/env python3
"""Select eight duration-matched task videos and build one review montage."""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import shutil
import statistics
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class TaskVideo:
    path: str
    label: str
    group: str
    duration_seconds: float
    width: int
    height: int
    source_fps: str


def run(command: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(command, check=True, text=True, capture_output=True)


def probe(path: Path, label: str, group: str) -> TaskVideo | None:
    try:
        payload = json.loads(
            run(
                [
                    "ffprobe",
                    "-v",
                    "error",
                    "-select_streams",
                    "v:0",
                    "-show_entries",
                    "format=duration:stream=width,height,avg_frame_rate",
                    "-of",
                    "json",
                    str(path),
                ]
            ).stdout
        )
        stream = payload["streams"][0]
        duration = float(payload["format"]["duration"])
        if not math.isfinite(duration) or duration <= 0:
            return None
        return TaskVideo(
            path=str(path.resolve()),
            label=label,
            group=group,
            duration_seconds=duration,
            width=int(stream["width"]),
            height=int(stream["height"]),
            source_fps=str(stream.get("avg_frame_rate", "unknown")),
        )
    except (KeyError, ValueError, subprocess.CalledProcessError, json.JSONDecodeError):
        return None


def guocai_candidates(root: Path) -> list[tuple[Path, str, str]]:
    result = []
    for path in sorted(root.glob("*/*/videos/left.mkv")):
        session = path.parent.parent
        configs = sorted(session.glob("*_config.yaml"))
        label = configs[0].name.removesuffix("_config.yaml") if configs else session.name
        result.append((path, label, session.parent.name))
    return result


def stera_candidates(root: Path) -> list[tuple[Path, str, str]]:
    return [
        (path, path.parent.name, path.parent.name)
        for path in sorted(root.glob("session_data_*/rgb.mp4"))
    ]


def jianzhi_candidates(root: Path) -> list[tuple[Path, str, str]]:
    return [(path, path.stem, path.stem) for path in sorted(root.glob("*.mp4"))]


def choose_duration_matched(records: list[TaskVideo], count: int) -> list[TaskVideo]:
    """Choose a tight duration cluster while retaining label/group diversity."""
    if len(records) < count:
        raise ValueError(f"need {count} usable tasks, found {len(records)}")
    unique_groups = {record.group for record in records}
    initial_cap = max(1, math.ceil(count / max(1, len(unique_groups))))
    best: tuple[tuple[float, float, int, float], list[TaskVideo]] | None = None
    for target in (record.duration_seconds for record in records):
        for cap in range(initial_cap, count + 1):
            selected: list[TaskVideo] = []
            group_counts: dict[str, int] = {}
            for record in sorted(
                records,
                key=lambda item: (
                    abs(item.duration_seconds - target),
                    item.duration_seconds,
                    item.path,
                ),
            ):
                if group_counts.get(record.group, 0) >= cap:
                    continue
                selected.append(record)
                group_counts[record.group] = group_counts.get(record.group, 0) + 1
                if len(selected) == count:
                    break
            if len(selected) != count:
                continue
            durations = [record.duration_seconds for record in selected]
            spread = max(durations) - min(durations)
            center_error = statistics.mean(abs(value - target) for value in durations)
            diversity = len({record.group for record in selected})
            score = (spread, center_error, -diversity, statistics.median(durations))
            if best is None or score < best[0]:
                best = score, selected
            break
    if best is None:
        raise RuntimeError("unable to select duration-matched tasks")
    return sorted(best[1], key=lambda item: (item.group, item.label, item.path))


def safe_overlay_text(value: str) -> str:
    value = re.sub(r"[^A-Za-z0-9_. -]+", "_", value)
    return value.replace("\\", "_").replace(":", "_").replace("'", "_")[:90]


def atomic_json(path: Path, data: object) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def build(args: argparse.Namespace) -> dict[str, object]:
    root = args.root.resolve()
    output = args.output.resolve()
    if not root.is_dir():
        raise FileNotFoundError(root)
    if output == root or output in root.parents:
        raise ValueError("output must not contain or equal the source root")
    if output.exists() and not args.overwrite:
        raise FileExistsError(f"output exists; pass --overwrite: {output}")
    if output.exists():
        shutil.rmtree(output)
    clips = output / "clips"
    clips.mkdir(parents=True)

    discover = {
        "guocai": guocai_candidates,
        "stera": stera_candidates,
        "jianzhi": jianzhi_candidates,
    }[args.dataset]
    candidates = discover(root)
    records = []
    for index, (path, label, group) in enumerate(candidates, 1):
        record = probe(path, label, group)
        if record is not None and record.duration_seconds >= args.min_source_seconds:
            records.append(record)
        if index % 50 == 0:
            print(f"[montage] probed {index}/{len(candidates)}", flush=True)
    selected = choose_duration_matched(records, args.task_count)
    clip_seconds = min(args.max_clip_seconds, min(item.duration_seconds for item in selected))
    if clip_seconds < args.min_clip_seconds:
        raise ValueError(f"matched clip duration is too short: {clip_seconds:.3f}s")

    clip_records = []
    for index, item in enumerate(selected, 1):
        start_seconds = max(0.0, (item.duration_seconds - clip_seconds) / 2.0)
        clip_path = clips / f"task_{index:02d}.mp4"
        overlay = safe_overlay_text(
            f"{args.dataset} | task {index:02d} | {item.group} | {item.label}"
        )
        video_filter = (
            f"fps={args.target_fps},"
            f"scale={args.width}:{args.height}:force_original_aspect_ratio=decrease,"
            f"pad={args.width}:{args.height}:(ow-iw)/2:(oh-ih)/2:color=black,"
            "setsar=1,"
            f"drawtext=text='{overlay}':x=18:y=16:fontsize=25:"
            "fontcolor=white:borderw=2:bordercolor=black"
        )
        subprocess.run(
            [
                "ffmpeg",
                "-hide_banner",
                "-loglevel",
                "error",
                "-y",
                "-ss",
                f"{start_seconds:.6f}",
                "-i",
                item.path,
                "-t",
                f"{clip_seconds:.6f}",
                "-an",
                "-vf",
                video_filter,
                "-c:v",
                "libx264",
                "-preset",
                "fast",
                "-crf",
                "20",
                "-pix_fmt",
                "yuv420p",
                "-movflags",
                "+faststart",
                str(clip_path),
            ],
            check=True,
        )
        clip_records.append(
            {
                **asdict(item),
                "selection_index": index,
                "clip_start_seconds": start_seconds,
                "clip_duration_seconds": clip_seconds,
                "prepared_path": str(clip_path),
            }
        )
        print(f"[montage] prepared {index}/{len(selected)} {item.path}", flush=True)

    concat_file = output / "concat.txt"
    concat_file.write_text(
        "".join(f"file '{(clips / f'task_{index:02d}.mp4').as_posix()}'\n" for index in range(1, len(selected) + 1)),
        encoding="utf-8",
    )
    montage = output / f"{args.dataset}_8tasks_source.mp4"
    subprocess.run(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            str(concat_file),
            "-an",
            "-c",
            "copy",
            "-movflags",
            "+faststart",
            str(montage),
        ],
        check=True,
    )
    manifest: dict[str, object] = {
        "schema_version": "pwm.task_montage.v1",
        "dataset": args.dataset,
        "source_root": str(root),
        "candidate_count": len(candidates),
        "usable_candidate_count": len(records),
        "task_count": len(selected),
        "target_fps": args.target_fps,
        "output_size_wh": [args.width, args.height],
        "common_clip_duration_seconds": clip_seconds,
        "montage_duration_seconds": clip_seconds * len(selected),
        "montage_path": str(montage),
        "tasks": clip_records,
    }
    atomic_json(output / "manifest.json", manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2), flush=True)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("guocai", "jianzhi", "stera"), required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--task-count", type=int, default=8)
    parser.add_argument("--min-source-seconds", type=float, default=8.0)
    parser.add_argument("--min-clip-seconds", type=float, default=4.0)
    parser.add_argument("--max-clip-seconds", type=float, default=8.0)
    parser.add_argument("--target-fps", type=int, default=10)
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--height", type=int, default=540)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.task_count < 1 or args.target_fps < 1 or args.width < 2 or args.height < 2:
        parser.error("invalid task-count/fps/size")
    if not 0 < args.min_clip_seconds <= args.max_clip_seconds:
        parser.error("clip duration bounds are invalid")
    build(args)


if __name__ == "__main__":
    main()
