#!/usr/bin/env python3
"""Read MCAP summaries and sample camera-message metadata without altering data."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from collections import Counter
from pathlib import Path
from typing import Any

from mcap.reader import make_reader
from mcap_protobuf.decoder import DecoderFactory


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def message_metadata(message: Any) -> dict[str, Any]:
    result: dict[str, Any] = {"python_type": f"{type(message).__module__}.{type(message).__name__}"}
    descriptor = getattr(message, "DESCRIPTOR", None)
    if descriptor is not None:
        result["protobuf_type"] = descriptor.full_name
    for name in (
        "format",
        "encoding",
        "width",
        "height",
        "step",
        "frame_id",
        "is_bigendian",
    ):
        if hasattr(message, name):
            value = getattr(message, name)
            if isinstance(value, (str, int, float, bool)):
                result[name] = value
    if hasattr(message, "data"):
        try:
            payload = bytes(message.data)
            result["data_bytes"] = len(payload)
            result["data_prefix_hex"] = payload[:24].hex()
        except (TypeError, ValueError):
            result["data_bytes"] = None
    header = getattr(message, "header", None)
    if header is not None:
        for name in ("timestamp", "frame_id"):
            if hasattr(header, name):
                value = getattr(header, name)
                if isinstance(value, (str, int, float, bool)):
                    result[f"header_{name}"] = value
    return result


def inspect_file(path: Path, sample_messages: bool) -> dict[str, Any]:
    stat = path.stat()
    record: dict[str, Any] = {
        "path": str(path.resolve()),
        "size_bytes": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }
    with path.open("rb") as stream:
        reader = make_reader(
            stream,
            decoder_factories=[DecoderFactory()] if sample_messages else [],
        )
        summary = reader.get_summary()
        if summary is None or summary.statistics is None:
            raise ValueError(f"MCAP summary/statistics absent: {path}")
        statistics = summary.statistics
        channels = []
        for channel in sorted(summary.channels.values(), key=lambda item: (item.topic, item.id)):
            schema = summary.schemas.get(channel.schema_id)
            channels.append(
                {
                    "id": channel.id,
                    "topic": channel.topic,
                    "message_encoding": channel.message_encoding,
                    "metadata": channel.metadata,
                    "schema_name": schema.name if schema else None,
                    "schema_encoding": schema.encoding if schema else None,
                    "schema_sha256": hashlib.sha256(schema.data).hexdigest() if schema else None,
                    "message_count": statistics.channel_message_counts.get(channel.id, 0),
                }
            )
        duration = (statistics.message_end_time - statistics.message_start_time) / 1e9
        record.update(
            message_count=statistics.message_count,
            start_time_ns=statistics.message_start_time,
            end_time_ns=statistics.message_end_time,
            duration_seconds=duration,
            channels=channels,
            metadata_records=len(summary.metadata_indexes),
            attachments=len(summary.attachment_indexes),
        )
        if sample_messages:
            video_topics = [
                item["topic"]
                for item in channels
                if any(token in item["topic"].lower() for token in ("camera", "image", "video", "rgb"))
            ]
            sampled: dict[str, dict[str, Any]] = {}
            try:
                for _, channel, raw_message, decoded in reader.iter_decoded_messages(topics=video_topics):
                    if channel.topic in sampled:
                        continue
                    sampled[channel.topic] = {
                        "log_time_ns": raw_message.log_time,
                        "publish_time_ns": raw_message.publish_time,
                        **message_metadata(decoded),
                    }
                    if len(sampled) == len(video_topics):
                        break
            except Exception as error:  # summary remains useful if one schema cannot decode
                record["sample_decode_error"] = f"{type(error).__name__}: {error}"
            record["sampled_video_messages"] = sampled
    return record


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sample-files", type=int, default=1)
    args = parser.parse_args()
    if args.sample_files < 0:
        parser.error("--sample-files must be non-negative")
    root = args.root.resolve()
    paths = sorted(root.rglob("*.mcap"))
    if not paths:
        raise FileNotFoundError(f"no MCAP files below {root}")
    records = []
    errors = []
    for index, path in enumerate(paths, 1):
        try:
            record = inspect_file(path, index <= args.sample_files)
            record["relative_path"] = str(path.relative_to(root))
            records.append(record)
        except Exception as error:
            errors.append(
                {
                    "relative_path": str(path.relative_to(root)),
                    "error": f"{type(error).__name__}: {error}",
                }
            )
        print(f"[mcap-inspect] {index}/{len(paths)}", flush=True)
    duration_values = [item["duration_seconds"] for item in records]
    topic_counts: Counter[str] = Counter()
    for record in records:
        topic_counts.update(item["topic"] for item in record["channels"])
    result = {
        "schema_version": "pwm.mcap_inventory.v1",
        "root": str(root),
        "file_count": len(paths),
        "ok_count": len(records),
        "error_count": len(errors),
        "total_bytes": sum(path.stat().st_size for path in paths),
        "duration_seconds": {
            "min": min(duration_values) if duration_values else None,
            "max": max(duration_values) if duration_values else None,
            "mean": sum(duration_values) / len(duration_values) if duration_values else None,
        },
        "topic_file_counts": dict(sorted(topic_counts.items())),
        "files": records,
        "errors": errors,
    }
    for value in result["duration_seconds"].values():
        if value is not None and not math.isfinite(value):
            raise ValueError("non-finite duration in inventory")
    atomic_json(args.output.resolve(), result)
    print(json.dumps({key: result[key] for key in ("file_count", "ok_count", "error_count", "duration_seconds", "topic_file_counts")}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
