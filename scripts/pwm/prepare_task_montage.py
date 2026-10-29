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
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from egohandmetric_prompt.camera_undistortion import (
    Rectification,
    guocai_equidistant_rectification,
    stera_rgb_rectification,
)


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


def letterbox(frame: np.ndarray, width: int, height: int) -> np.ndarray:
    scale = min(width / frame.shape[1], height / frame.shape[0])
    resized_width = max(1, round(frame.shape[1] * scale))
    resized_height = max(1, round(frame.shape[0] * scale))
    resized = cv2.resize(frame, (resized_width, resized_height), interpolation=cv2.INTER_AREA)
    canvas = np.zeros((height, width, 3), dtype=np.uint8)
    x = (width - resized_width) // 2
    y = (height - resized_height) // 2
    canvas[y : y + resized_height, x : x + resized_width] = resized
    return canvas


def task_rectification(dataset: str, item: TaskVideo) -> Rectification:
    source = Path(item.path)
    source_size = (item.width, item.height)
    if dataset == "guocai":
        session = source.parent.parent
        return guocai_equidistant_rectification(
            session / "calibrations" / "camera-camchain.yaml",
            source_size,
        )
    if dataset == "stera":
        return stera_rgb_rectification(source.parent, source_size)
    raise ValueError(
        f"{dataset} videos do not carry an auditable calibration here; "
        "use prepare_jianzhi_montage.py for Jianzhi MCAP"
    )


def read_exact(stream: object, byte_count: int) -> bytes:
    chunks = []
    remaining = byte_count
    while remaining:
        chunk = stream.read(remaining)
        if not chunk:
            break
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def prepare_rectified_clip(
    item: TaskVideo,
    destination: Path,
    *,
    start_seconds: float,
    clip_seconds: float,
    target_fps: int,
    width: int,
    height: int,
    overlay: str,
    rectification: Rectification,
) -> dict[str, int]:
    target_frames = round(clip_seconds * target_fps)
    frame_bytes = item.width * item.height * 3
    decoder_command = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-ss",
        f"{start_seconds:.6f}",
        "-i",
        item.path,
        "-t",
        f"{clip_seconds:.6f}",
        "-an",
        "-vf",
        f"fps={target_fps}",
        "-frames:v",
        str(target_frames),
        "-f",
        "rawvideo",
        "-pix_fmt",
        "bgr24",
        "-",
    ]
    temporary = destination.with_name(f".{destination.stem}.{os.getpid()}.part.mp4")
    encoder_command = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "bgr24",
        "-s",
        f"{width}x{height}",
        "-r",
        str(target_fps),
        "-i",
        "-",
        "-an",
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
        str(temporary),
    ]
    decoder = subprocess.Popen(decoder_command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    encoder = subprocess.Popen(encoder_command, stdin=subprocess.PIPE)
    if decoder.stdout is None or decoder.stderr is None or encoder.stdin is None:
        raise RuntimeError("ffmpeg pipe unavailable")
    written = 0
    duplicated = 0
    last_canvas = None
    try:
        while written < target_frames:
            raw = read_exact(decoder.stdout, frame_bytes)
            if len(raw) != frame_bytes:
                break
            frame = np.frombuffer(raw, dtype=np.uint8).reshape(item.height, item.width, 3)
            frame = rectification.apply(frame)
            canvas = letterbox(frame, width, height)
            cv2.rectangle(canvas, (0, 0), (width, 52), (0, 0, 0), -1)
            cv2.putText(
                canvas,
                overlay,
                (16, 34),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.72,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
            encoder.stdin.write(canvas.tobytes())
            last_canvas = canvas
            written += 1
        decoder_error = decoder.stderr.read().decode("utf-8", errors="replace")
        decoder_return_code = decoder.wait()
        if decoder_return_code != 0:
            raise RuntimeError(f"ffmpeg decode failed ({decoder_return_code}): {decoder_error}")
        if last_canvas is None:
            raise ValueError(f"no frames decoded from {item.path}")
        while written < target_frames:
            encoder.stdin.write(last_canvas.tobytes())
            written += 1
            duplicated += 1
    except Exception:
        decoder.kill()
        encoder.stdin.close()
        encoder.wait()
        temporary.unlink(missing_ok=True)
        raise
    encoder.stdin.close()
    encoder_return_code = encoder.wait()
    if encoder_return_code != 0:
        temporary.unlink(missing_ok=True)
        raise subprocess.CalledProcessError(encoder_return_code, encoder_command)
    temporary.replace(destination)
    return {"written_frames": written, "duplicated_tail_frames": duplicated}


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
        preparation: dict[str, object] = {}
        if args.require_undistortion:
            rectification = task_rectification(args.dataset, item)
            preparation.update(
                prepare_rectified_clip(
                    item,
                    clip_path,
                    start_seconds=start_seconds,
                    clip_seconds=clip_seconds,
                    target_fps=args.target_fps,
                    width=args.width,
                    height=args.height,
                    overlay=overlay,
                    rectification=rectification,
                )
            )
            preparation["rectification"] = rectification.metadata
        else:
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
            preparation["rectification"] = {
                "output_is_undistorted": False,
                "undistortion_applied": False,
                "source_already_rectified": False,
                "method": "none (legacy/raw preparation)",
                "source_size_wh": [item.width, item.height],
            }
        clip_records.append(
            {
                **asdict(item),
                "selection_index": index,
                "clip_start_seconds": start_seconds,
                "clip_duration_seconds": clip_seconds,
                "prepared_path": str(clip_path),
                **preparation,
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
        "undistortion_required": args.require_undistortion,
        "all_inputs_undistorted": all(
            bool(record["rectification"]["output_is_undistorted"])
            for record in clip_records
        ),
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
    parser.add_argument("--require-undistortion", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.task_count < 1 or args.target_fps < 1 or args.width < 2 or args.height < 2:
        parser.error("invalid task-count/fps/size")
    if not 0 < args.min_clip_seconds <= args.max_clip_seconds:
        parser.error("clip duration bounds are invalid")
    build(args)


if __name__ == "__main__":
    main()
