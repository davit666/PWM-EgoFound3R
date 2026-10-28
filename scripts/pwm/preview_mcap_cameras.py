#!/usr/bin/env python3
"""Decode one representative frame per H264 camera and build a contact sheet."""

from __future__ import annotations

import argparse
import json
import os
from fractions import Fraction
from pathlib import Path

import av
import cv2
import numpy as np
from mcap.reader import make_reader
from mcap_protobuf.decoder import DecoderFactory


NANOSECONDS = Fraction(1, 1_000_000_000)


def h264_nal_types(payload: bytes) -> list[int]:
    result = []
    cursor = 0
    while True:
        start = payload.find(b"\x00\x00\x01", cursor)
        if start < 0:
            return result
        if start + 3 < len(payload):
            result.append(payload[start + 3] & 31)
        cursor = start + 3


def atomic_json(path: Path, value: object) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tile-width", type=int, default=640)
    args = parser.parse_args()
    source = args.input.resolve()
    output = args.output.resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    if args.tile_width < 160:
        parser.error("--tile-width must be at least 160")
    output.mkdir(parents=True, exist_ok=True)

    with source.open("rb") as stream:
        reader = make_reader(stream, decoder_factories=[DecoderFactory()])
        summary = reader.get_summary()
        if summary is None:
            raise ValueError("MCAP summary absent")
        topics = sorted(
            channel.topic
            for channel in summary.channels.values()
            if channel.topic.endswith("/compressed")
        )
        decoders: dict[str, av.CodecContext | None] = {topic: None for topic in topics}
        frames: dict[str, np.ndarray] = {}
        timestamps: dict[str, int] = {}
        for _, channel, message, decoded in reader.iter_decoded_messages(topics=topics):
            topic = channel.topic
            if topic in frames:
                continue
            if str(decoded.format).lower() != "h264":
                raise ValueError(f"unsupported camera format on {topic}: {decoded.format}")
            payload = bytes(decoded.data)
            if decoders[topic] is None:
                if not ({5, 7, 8} <= set(h264_nal_types(payload))):
                    continue
                decoders[topic] = av.CodecContext.create("h264", "r")
            packet = av.Packet(payload)
            packet.pts = packet.dts = message.log_time
            packet.time_base = NANOSECONDS
            try:
                produced = decoders[topic].decode(packet)
            except av.error.InvalidDataError:
                decoders[topic] = None
                continue
            if not produced:
                continue
            frames[topic] = produced[0].to_ndarray(format="bgr24")
            timestamps[topic] = message.log_time
            if len(frames) == len(topics):
                break
    missing = sorted(set(topics) - set(frames))
    if missing:
        raise ValueError(f"could not decode representative frames for: {missing}")

    tiles = []
    records = []
    for topic in topics:
        frame = frames[topic]
        height = max(1, round(frame.shape[0] * args.tile_width / frame.shape[1]))
        tile = cv2.resize(frame, (args.tile_width, height), interpolation=cv2.INTER_AREA)
        cv2.rectangle(tile, (0, 0), (tile.shape[1], 54), (0, 0, 0), -1)
        cv2.putText(
            tile,
            topic,
            (12, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.72,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        image_path = output / f"camera_{len(tiles):02d}.jpg"
        if not cv2.imwrite(str(image_path), frame, [cv2.IMWRITE_JPEG_QUALITY, 92]):
            raise RuntimeError(f"failed to write {image_path}")
        tiles.append(tile)
        records.append(
            {
                "topic": topic,
                "timestamp_ns": timestamps[topic],
                "size_wh": [int(frame.shape[1]), int(frame.shape[0])],
                "image": str(image_path),
            }
        )
    row_height = max(tile.shape[0] for tile in tiles)
    padded = [
        cv2.copyMakeBorder(tile, 0, row_height - tile.shape[0], 0, 0, cv2.BORDER_CONSTANT)
        for tile in tiles
    ]
    if len(padded) % 2:
        padded.append(np.zeros_like(padded[0]))
    rows = [np.hstack(padded[index : index + 2]) for index in range(0, len(padded), 2)]
    sheet = np.vstack(rows)
    sheet_path = output / "camera_contact_sheet.jpg"
    if not cv2.imwrite(str(sheet_path), sheet, [cv2.IMWRITE_JPEG_QUALITY, 92]):
        raise RuntimeError(f"failed to write {sheet_path}")
    manifest = {
        "schema_version": "pwm.mcap_camera_preview.v1",
        "source": str(source),
        "contact_sheet": str(sheet_path),
        "cameras": records,
    }
    atomic_json(output / "manifest.json", manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
