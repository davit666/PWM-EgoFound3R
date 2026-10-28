#!/usr/bin/env python3
"""Compose the 2x2 semantic overlay with canonical MANO and original RGB panels."""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
from pathlib import Path

import cv2
import numpy as np
import torch

from egohandmetric_prompt.hand_upsampling import build_mano778_hand
from egohandmetric_prompt.inference_visualization import (
    semantic_point_colors,
    signal_face_colors,
)


DEFAULT_LEFT_SIZE = (960, 540)
DEFAULT_MANO_SIZE = (360, 540)
DEFAULT_RGB_SIZE = (960, 540)
HEADER_HEIGHT = 42
BACKGROUND_BGR = (20, 22, 26)
BORDER_BGR = (72, 76, 84)
TEXT_BGR = (245, 245, 245)


def atomic_json(path: Path, value: object) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def normalize(vector: np.ndarray, fallback: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if not math.isfinite(norm) or norm < 1e-8:
        return fallback.astype(np.float32, copy=True)
    return (vector / norm).astype(np.float32)


def derive_basis(
    joints: np.ndarray,
    *,
    side_index: int,
    previous: tuple[np.ndarray, np.ndarray] | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    wrist = joints[0]
    fingertips = joints[[8, 12, 16, 20]].mean(axis=0)
    y_axis = normalize(fingertips - wrist, np.array((0.0, -1.0, 0.0), dtype=np.float32))
    across = joints[5] - joints[17]
    across = across - y_axis * float(np.dot(across, y_axis))
    x_axis = normalize(across, np.array((1.0, 0.0, 0.0), dtype=np.float32))

    palm_center = joints[[0, 5, 9, 13, 17]].mean(axis=0)
    thumb_direction = joints[4] - palm_center
    desired_thumb_sign = 1.0 if side_index == 0 else -1.0
    if float(np.dot(thumb_direction, x_axis)) * desired_thumb_sign < 0.0:
        x_axis = -x_axis

    if previous is not None:
        previous_x, previous_y = previous
        if float(np.dot(y_axis, previous_y)) < 0.0:
            y_axis = -y_axis
        y_axis = normalize(0.30 * y_axis + 0.70 * previous_y, previous_y)
        x_axis = x_axis - y_axis * float(np.dot(x_axis, y_axis))
        if float(np.dot(x_axis, previous_x)) < 0.0:
            x_axis = -x_axis
        x_axis = normalize(0.30 * x_axis + 0.70 * previous_x, previous_x)
        x_axis = normalize(x_axis - y_axis * float(np.dot(x_axis, y_axis)), previous_x)
    z_axis = normalize(np.cross(x_axis, y_axis), np.array((0.0, 0.0, 1.0), dtype=np.float32))
    return x_axis, y_axis, z_axis


def canonicalize_sequence(
    vertices: np.ndarray,
    joints: np.ndarray,
    hand_valid: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    if vertices.ndim != 4 or vertices.shape[1:] != (2, 778, 3):
        raise ValueError(f"expected MANO vertices [T,2,778,3], got {vertices.shape}")
    if joints.shape[:2] != vertices.shape[:2] or joints.shape[-1] != 3:
        raise ValueError("joint and vertex axes do not align")
    canonical = np.full_like(vertices, np.nan, dtype=np.float32)
    valid = hand_valid.astype(bool, copy=True)
    previous: list[tuple[np.ndarray, np.ndarray] | None] = [None, None]
    for frame_index in range(vertices.shape[0]):
        for side_index in range(2):
            if not valid[frame_index, side_index]:
                continue
            current_vertices = vertices[frame_index, side_index]
            current_joints = joints[frame_index, side_index]
            if not (
                np.isfinite(current_vertices).all()
                and np.isfinite(current_joints).all()
                and current_joints.shape[0] >= 21
            ):
                valid[frame_index, side_index] = False
                continue
            x_axis, y_axis, z_axis = derive_basis(
                current_joints,
                side_index=side_index,
                previous=previous[side_index],
            )
            previous[side_index] = (x_axis, y_axis)
            palm_center = current_joints[[0, 5, 9, 13, 17]].mean(axis=0)
            local = current_vertices - palm_center
            canonical[frame_index, side_index] = np.stack(
                (
                    local @ x_axis,
                    local @ y_axis,
                    local @ z_axis,
                ),
                axis=-1,
            )
    return canonical, valid


def canonical_scale(
    canonical: np.ndarray,
    valid: np.ndarray,
    *,
    panel_width: int,
    panel_height: int,
) -> float:
    spans_x = []
    spans_y = []
    for frame_index in range(canonical.shape[0]):
        for side_index in range(2):
            if not valid[frame_index, side_index]:
                continue
            points = canonical[frame_index, side_index]
            spans_x.append(float(np.ptp(points[:, 0])))
            spans_y.append(float(np.ptp(points[:, 1])))
    if not spans_x or not spans_y:
        raise ValueError("no valid canonical hands")
    slot_width = panel_width / 2.0
    body_height = panel_height - HEADER_HEIGHT
    robust_x = max(float(np.quantile(spans_x, 0.98)), 1e-5)
    robust_y = max(float(np.quantile(spans_y, 0.98)), 1e-5)
    return min(slot_width * 0.88 / robust_x, body_height * 0.92 / robust_y)


def face_colors(
    mode: str,
    values: np.ndarray,
    faces: np.ndarray,
    canonical_vertices: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    if mode == "contact":
        points = semantic_point_colors(
            "Contact Class",
            side_index=0,
            point_kind="marker",
            visibility=np.ones(values.shape, dtype=np.float32),
            contact_probability=values,
            contact_distance_m=None,
        )
    elif mode == "distance":
        points = semantic_point_colors(
            "Contact Distance",
            side_index=0,
            point_kind="marker",
            visibility=np.ones(values.shape, dtype=np.float32),
            contact_probability=None,
            contact_distance_m=values,
        )
    else:
        raise ValueError(mode)
    colors = signal_face_colors(points, faces).astype(np.float32)
    triangle = canonical_vertices[faces]
    normals = np.cross(triangle[:, 1] - triangle[:, 0], triangle[:, 2] - triangle[:, 0])
    normal_norm = np.linalg.norm(normals, axis=-1).clip(min=1e-8)
    facing = np.abs(normals[:, 2] / normal_norm)
    shading = 0.55 + 0.45 * facing
    colors = np.rint(colors * shading[:, None]).clip(0, 255).astype(np.uint8)
    depth = triangle[..., 2].mean(axis=1)
    return colors, depth


def draw_mesh(
    panel: np.ndarray,
    canonical_vertices: np.ndarray,
    values: np.ndarray,
    faces: np.ndarray,
    *,
    mode: str,
    center_x: float,
    scale: float,
) -> None:
    finite = np.isfinite(canonical_vertices).all(axis=-1) & np.isfinite(values)
    valid_faces = finite[faces].all(axis=-1)
    current_faces = faces[valid_faces]
    if current_faces.size == 0:
        return
    current_vertices = canonical_vertices
    center = 0.5 * (
        np.nanmin(current_vertices[:, :2], axis=0)
        + np.nanmax(current_vertices[:, :2], axis=0)
    )
    uv = np.empty((current_vertices.shape[0], 2), dtype=np.float32)
    uv[:, 0] = center_x + (current_vertices[:, 0] - center[0]) * scale
    body_center_y = HEADER_HEIGHT + (panel.shape[0] - HEADER_HEIGHT) / 2.0
    uv[:, 1] = body_center_y - (current_vertices[:, 1] - center[1]) * scale
    colors, depth = face_colors(mode, values, current_faces, current_vertices)
    polygons = np.rint(uv[current_faces]).astype(np.int32)
    for face_index in np.argsort(depth):
        cv2.fillConvexPoly(
            panel,
            polygons[face_index],
            tuple(int(channel) for channel in colors[face_index, ::-1]),
            lineType=cv2.LINE_AA,
        )


def label(
    image: np.ndarray,
    text: str,
    origin: tuple[int, int],
    *,
    align_right: bool = False,
    scale: float = 0.55,
) -> None:
    thickness = 1
    size, _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)
    x, y = origin
    if align_right:
        x -= size[0]
    cv2.putText(
        image,
        text,
        (x, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        (28, 28, 28),
        3,
        cv2.LINE_AA,
    )
    cv2.putText(
        image,
        text,
        (x, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        TEXT_BGR,
        thickness,
        cv2.LINE_AA,
    )


def mano_panel(
    canonical_frame: np.ndarray,
    valid_frame: np.ndarray,
    values_frame: np.ndarray,
    faces: np.ndarray,
    *,
    mode: str,
    width: int,
    height: int,
    scale: float,
) -> np.ndarray:
    panel = np.full((height, width, 3), BACKGROUND_BGR, dtype=np.uint8)
    title = "MANO Contact" if mode == "contact" else "MANO Distance 0-50 mm"
    label(panel, title, (9, 18), scale=0.48)
    label(panel, "L", (width // 4 - 4, 37), scale=0.42)
    label(panel, "R", (3 * width // 4 - 4, 37), scale=0.42)
    centers = (width * 0.25, width * 0.75)
    for side_index in range(2):
        if valid_frame[side_index]:
            draw_mesh(
                panel,
                canonical_frame[side_index],
                values_frame[side_index],
                faces,
                mode=mode,
                center_x=centers[side_index],
                scale=scale,
            )
        else:
            label(
                panel,
                f"{'L' if side_index == 0 else 'R'} unavailable",
                (int(centers[side_index] - 48), height // 2),
                scale=0.36,
            )
    cv2.rectangle(panel, (0, 0), (width - 1, height - 1), BORDER_BGR, 1)
    return panel


def load_predictions(path: Path) -> dict[str, np.ndarray]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    predictions = payload.get("metric_predictions")
    hand = predictions.get("hand") if isinstance(predictions, dict) else None
    if not isinstance(hand, dict):
        raise ValueError("inference payload is missing metric_predictions.hand")
    hand_778 = payload.get("hand_778")
    if not isinstance(hand_778, dict):
        hand_778 = build_mano778_hand(hand)
    required = {
        "vertices": hand_778.get("vertex_xyz_metric"),
        "joints": hand.get("joint_xyz_metric"),
        "valid": hand_778.get("valid"),
        "contact": hand_778.get("vertex_contact_probability"),
        "distance": hand_778.get("vertex_contact_distance_metric"),
        "faces": hand_778.get("marker_faces"),
    }
    missing = [name for name, value in required.items() if not isinstance(value, torch.Tensor)]
    if missing:
        raise ValueError(f"inference payload is missing MANO fields: {missing}")
    return {
        name: value.detach().cpu().float().numpy()
        if name not in {"valid", "faces"}
        else (
            value.detach().cpu().bool().numpy()
            if name == "valid"
            else value.detach().cpu().long().numpy()
        )
        for name, value in required.items()
    }


def video_info(capture: cv2.VideoCapture, path: Path) -> dict[str, object]:
    if not capture.isOpened():
        raise ValueError(f"cannot open video: {path}")
    return {
        "path": str(path.resolve()),
        "frames": int(capture.get(cv2.CAP_PROP_FRAME_COUNT)),
        "fps": float(capture.get(cv2.CAP_PROP_FPS)),
        "width": int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--multimode-video", type=Path, required=True)
    parser.add_argument("--rgb-video", type=Path, required=True)
    parser.add_argument("--inference-output", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--fps", type=float, default=10.0)
    parser.add_argument("--max-frames", type=int)
    parser.add_argument("--crf", type=int, default=25)
    args = parser.parse_args()
    if args.fps <= 0 or (args.max_frames is not None and args.max_frames < 1):
        parser.error("invalid FPS or max frame count")
    predictions = load_predictions(args.inference_output.resolve())
    canonical, canonical_valid = canonicalize_sequence(
        predictions["vertices"], predictions["joints"], predictions["valid"]
    )
    panel_width, full_height = DEFAULT_MANO_SIZE
    panel_height = full_height // 2
    scale = canonical_scale(
        canonical,
        canonical_valid,
        panel_width=panel_width,
        panel_height=panel_height,
    )

    multimode_path = args.multimode_video.resolve()
    rgb_path = args.rgb_video.resolve()
    multimode_capture = cv2.VideoCapture(str(multimode_path))
    rgb_capture = cv2.VideoCapture(str(rgb_path))
    multimode_info = video_info(multimode_capture, multimode_path)
    rgb_info = video_info(rgb_capture, rgb_path)
    prediction_frames = int(canonical.shape[0])
    available_frames = min(
        int(multimode_info["frames"]), int(rgb_info["frames"]), prediction_frames
    )
    frame_count = available_frames if args.max_frames is None else min(available_frames, args.max_frames)
    if args.max_frames is None and not (
        int(multimode_info["frames"]) == int(rgb_info["frames"]) == prediction_frames
    ):
        raise ValueError(
            "full composition requires equal multimode, RGB, and prediction frame counts"
        )

    left_width, output_height = DEFAULT_LEFT_SIZE
    rgb_width, rgb_height = DEFAULT_RGB_SIZE
    if output_height != full_height or rgb_height != full_height:
        raise ValueError("layout components must share one height")
    output_width = left_width + panel_width + rgb_width
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
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
        f"{output_width}x{output_height}",
        "-r",
        str(args.fps),
        "-i",
        "-",
        "-an",
        "-c:v",
        "libx264",
        "-preset",
        "medium",
        "-crf",
        str(args.crf),
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(temporary),
    ]
    encoder = subprocess.Popen(command, stdin=subprocess.PIPE)
    if encoder.stdin is None:
        raise RuntimeError("ffmpeg stdin unavailable")
    try:
        for frame_index in range(frame_count):
            multimode_ok, multimode = multimode_capture.read()
            rgb_ok, rgb = rgb_capture.read()
            if not multimode_ok or not rgb_ok:
                raise RuntimeError(f"video decode ended at frame {frame_index}")
            left = cv2.resize(multimode, (left_width, output_height), interpolation=cv2.INTER_AREA)
            original = cv2.resize(rgb, (rgb_width, rgb_height), interpolation=cv2.INTER_AREA)
            contact = mano_panel(
                canonical[frame_index],
                canonical_valid[frame_index],
                predictions["contact"][frame_index],
                predictions["faces"],
                mode="contact",
                width=panel_width,
                height=panel_height,
                scale=scale,
            )
            distance = mano_panel(
                canonical[frame_index],
                canonical_valid[frame_index],
                predictions["distance"][frame_index],
                predictions["faces"],
                mode="distance",
                width=panel_width,
                height=panel_height,
                scale=scale,
            )
            mano = np.vstack((contact, distance))
            final = np.hstack((left, mano, original))
            cv2.line(final, (left_width, 0), (left_width, output_height - 1), BORDER_BGR, 2)
            cv2.line(
                final,
                (left_width + panel_width, 0),
                (left_width + panel_width, output_height - 1),
                BORDER_BGR,
                2,
            )
            label(
                final,
                "Original RGB",
                (output_width - 10, 22),
                align_right=True,
                scale=0.58,
            )
            encoder.stdin.write(final.tobytes())
            if (frame_index + 1) % 50 == 0 or frame_index + 1 == frame_count:
                print(f"[mano-layout] {frame_index + 1}/{frame_count}", flush=True)
    except Exception:
        encoder.stdin.close()
        encoder.wait()
        temporary.unlink(missing_ok=True)
        raise
    finally:
        multimode_capture.release()
        rgb_capture.release()
    encoder.stdin.close()
    return_code = encoder.wait()
    if return_code != 0:
        temporary.unlink(missing_ok=True)
        raise subprocess.CalledProcessError(return_code, command)
    temporary.replace(output)
    manifest = {
        "schema_version": "pwm.extended_mano_layout.v1",
        "multimode_video": multimode_info,
        "rgb_video": rgb_info,
        "inference_output": str(args.inference_output.resolve()),
        "output": str(output),
        "frame_count": frame_count,
        "fps": args.fps,
        "duration_seconds": frame_count / args.fps,
        "output_size_wh": [output_width, output_height],
        "layout": {
            "left": "2x2 Geometry / Visibility / Contact / Distance",
            "middle_top": "canonical palm-facing MANO Contact; left hand left, right hand right",
            "middle_bottom": "canonical palm-facing MANO Distance 0-50 mm; left hand left, right hand right",
            "right": "original RGB",
        },
        "mano": {
            "vertices_per_hand": 778,
            "faces": int(predictions["faces"].shape[0]),
            "projection": "orthographic canonical palm plane",
            "orientation": "mean four fingertips up; thumbs toward layout center",
            "fixed_scale_pixels_per_meter": scale,
        },
    }
    atomic_json(args.manifest.resolve(), manifest)
    print(json.dumps(manifest, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
