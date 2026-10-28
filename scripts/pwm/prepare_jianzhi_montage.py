#!/usr/bin/env python3
"""Select duration-matched Jianzhi MCAP tasks and build an equal-length montage."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
from dataclasses import asdict, dataclass
from fractions import Fraction
from pathlib import Path

import av
import cv2
import numpy as np
from mcap.reader import make_reader
from mcap_protobuf.decoder import DecoderFactory


NANOSECONDS = Fraction(1, 1_000_000_000)


@dataclass(frozen=True)
class McapTask:
    path: str
    duration_seconds: float
    start_time_ns: int
    end_time_ns: int
    size_bytes: int


def atomic_json(path: Path, value: object) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def h264_nal_types(data: bytes) -> list[int]:
    result = []
    cursor = 0
    while True:
        position = data.find(b"\x00\x00\x01", cursor)
        if position < 0:
            return result
        if position + 3 < len(data):
            result.append(data[position + 3] & 31)
        cursor = position + 3


def inventory(root: Path) -> list[McapTask]:
    result = []
    paths = sorted(root.glob("*.mcap"))
    for index, path in enumerate(paths, 1):
        with path.open("rb") as stream:
            summary = make_reader(stream).get_summary()
        if summary is None or summary.statistics is None:
            raise ValueError(f"MCAP summary/statistics absent: {path}")
        statistics = summary.statistics
        duration = (statistics.message_end_time - statistics.message_start_time) / 1e9
        result.append(
            McapTask(
                path=str(path.resolve()),
                duration_seconds=duration,
                start_time_ns=statistics.message_start_time,
                end_time_ns=statistics.message_end_time,
                size_bytes=path.stat().st_size,
            )
        )
        print(f"[jianzhi] inventory {index}/{len(paths)}", flush=True)
    return result


def select_duration_matched(tasks: list[McapTask], count: int) -> list[McapTask]:
    if len(tasks) < count:
        raise ValueError(f"need {count} tasks, found {len(tasks)}")
    ordered = sorted(tasks, key=lambda item: (item.duration_seconds, item.path))
    candidates = []
    for index in range(len(ordered) - count + 1):
        group = ordered[index : index + count]
        durations = [item.duration_seconds for item in group]
        candidates.append(
            (
                max(durations) - min(durations),
                sum(durations) / len(durations),
                tuple(item.path for item in group),
                group,
            )
        )
    return sorted(candidates, key=lambda item: item[:3])[0][3]


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


def extract_clip(
    task: McapTask,
    output: Path,
    *,
    topic: str,
    clip_seconds: float,
    target_fps: int,
    width: int,
    height: int,
    label: str,
) -> dict[str, object]:
    center_ns = (task.start_time_ns + task.end_time_ns) // 2
    start_ns = center_ns - round(clip_seconds * 1e9 / 2)
    end_ns = start_ns + round(clip_seconds * 1e9)
    frame_interval_ns = round(1e9 / target_fps)
    target_frames = round(clip_seconds * target_fps)
    temporary = output.with_name(f".{output.stem}.{os.getpid()}.part.mp4")
    command = [
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
    process = subprocess.Popen(command, stdin=subprocess.PIPE)
    if process.stdin is None:
        raise RuntimeError("ffmpeg stdin unavailable")
    decoder: av.CodecContext | None = None
    next_sample_ns = start_ns
    written = 0
    source_decoded = 0
    duplicated = 0
    first_written_ns = None
    last_written_ns = None
    last_canvas: np.ndarray | None = None

    def write_canvas(canvas: np.ndarray, timestamp_ns: int | None) -> None:
        nonlocal written, last_canvas, first_written_ns, last_written_ns
        process.stdin.write(canvas.tobytes())
        written += 1
        last_canvas = canvas
        if timestamp_ns is not None:
            if first_written_ns is None:
                first_written_ns = timestamp_ns
            last_written_ns = timestamp_ns

    try:
        with Path(task.path).open("rb") as stream:
            reader = make_reader(stream, decoder_factories=[DecoderFactory()])
            for _, channel, message, decoded in reader.iter_decoded_messages(topics=[topic]):
                if written >= target_frames:
                    break
                if str(decoded.format).lower() != "h264":
                    raise ValueError(f"unsupported format on {topic}: {decoded.format}")
                payload = bytes(decoded.data)
                if decoder is None:
                    if not ({5, 7, 8} <= set(h264_nal_types(payload))):
                        continue
                    decoder = av.CodecContext.create("h264", "r")
                packet = av.Packet(payload)
                packet.pts = packet.dts = message.log_time
                packet.time_base = NANOSECONDS
                try:
                    frames = decoder.decode(packet)
                except av.error.InvalidDataError:
                    decoder = None
                    continue
                for frame in frames:
                    source_decoded += 1
                    timestamp_ns = int(Fraction(frame.pts) * frame.time_base * 1_000_000_000)
                    if timestamp_ns < next_sample_ns:
                        continue
                    if timestamp_ns >= end_ns or written >= target_frames:
                        break
                    canvas = letterbox(frame.to_ndarray(format="bgr24"), width, height)
                    cv2.rectangle(canvas, (0, 0), (width, 52), (0, 0, 0), -1)
                    cv2.putText(
                        canvas,
                        label,
                        (16, 34),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.72,
                        (255, 255, 255),
                        2,
                        cv2.LINE_AA,
                    )
                    write_canvas(canvas, timestamp_ns)
                    while next_sample_ns <= timestamp_ns:
                        next_sample_ns += frame_interval_ns
                if message.log_time > end_ns + 1_000_000_000:
                    break
        if last_canvas is None:
            raise ValueError(f"no frames decoded inside requested interval: {task.path}")
        while written < target_frames:
            write_canvas(last_canvas, None)
            duplicated += 1
    except Exception:
        process.stdin.close()
        process.wait()
        temporary.unlink(missing_ok=True)
        raise
    process.stdin.close()
    return_code = process.wait()
    if return_code != 0:
        temporary.unlink(missing_ok=True)
        raise subprocess.CalledProcessError(return_code, command)
    temporary.replace(output)
    return {
        **asdict(task),
        "topic": topic,
        "clip_start_ns": start_ns,
        "clip_end_ns": end_ns,
        "clip_duration_seconds": clip_seconds,
        "target_fps": target_fps,
        "written_frames": written,
        "duplicated_tail_frames": duplicated,
        "source_decoded_frames": source_decoded,
        "first_written_source_time_ns": first_written_ns,
        "last_written_source_time_ns": last_written_ns,
        "prepared_path": str(output),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--topic", default="/robot0/sensor/camera2/compressed")
    parser.add_argument("--task-count", type=int, default=8)
    parser.add_argument("--clip-seconds", type=float, default=8.0)
    parser.add_argument("--target-fps", type=int, default=10)
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--height", type=int, default=540)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.task_count < 1 or args.clip_seconds <= 0 or args.target_fps < 1:
        parser.error("invalid task count, clip duration, or target FPS")
    root = args.root.resolve()
    output = args.output.resolve()
    if not root.is_dir():
        raise FileNotFoundError(root)
    if output.exists() and not args.overwrite:
        raise FileExistsError(f"output exists; pass --overwrite: {output}")
    if output.exists():
        shutil.rmtree(output)
    clips = output / "clips"
    clips.mkdir(parents=True)

    tasks = inventory(root)
    selected = select_duration_matched(tasks, args.task_count)
    clip_records = []
    for index, task in enumerate(selected, 1):
        destination = clips / f"task_{index:02d}.mp4"
        record = extract_clip(
            task,
            destination,
            topic=args.topic,
            clip_seconds=args.clip_seconds,
            target_fps=args.target_fps,
            width=args.width,
            height=args.height,
            label=f"jianzhi | task {index:02d} | {Path(task.path).stem}",
        )
        record["selection_index"] = index
        clip_records.append(record)
        print(f"[jianzhi] prepared {index}/{len(selected)} {task.path}", flush=True)

    concat_file = output / "concat.txt"
    concat_file.write_text(
        "".join(f"file '{(clips / f'task_{index:02d}.mp4').as_posix()}'\n" for index in range(1, len(selected) + 1)),
        encoding="utf-8",
    )
    montage = output / "jianzhi_8tasks_source.mp4"
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
    durations = [task.duration_seconds for task in selected]
    manifest = {
        "schema_version": "pwm.task_montage.v1",
        "dataset": "jianzhi",
        "source_root": str(root),
        "candidate_count": len(tasks),
        "usable_candidate_count": len(tasks),
        "task_count": len(selected),
        "topic": args.topic,
        "source_duration_spread_seconds": max(durations) - min(durations),
        "target_fps": args.target_fps,
        "output_size_wh": [args.width, args.height],
        "common_clip_duration_seconds": args.clip_seconds,
        "montage_duration_seconds": args.clip_seconds * len(selected),
        "montage_path": str(montage),
        "tasks": clip_records,
    }
    atomic_json(output / "manifest.json", manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
