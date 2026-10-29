"""Auditable camera rectification helpers for PWM evaluation videos."""

from __future__ import annotations

import ast
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np


@dataclass(frozen=True)
class Rectification:
    """Destination-to-source maps plus JSON-serializable provenance."""

    map_x: np.ndarray | None
    map_y: np.ndarray | None
    metadata: dict[str, Any]

    def apply(self, frame: np.ndarray) -> np.ndarray:
        expected = tuple(self.metadata["source_size_wh"])
        actual = (int(frame.shape[1]), int(frame.shape[0]))
        if actual != expected:
            raise ValueError(f"rectification size mismatch: expected {expected}, got {actual}")
        if self.map_x is None or self.map_y is None:
            return frame
        return cv2.remap(
            frame,
            self.map_x,
            self.map_y,
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
        )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _map_valid_fraction(
    map_x: np.ndarray, map_y: np.ndarray, size_wh: tuple[int, int]
) -> float:
    width, height = size_wh
    valid = (
        np.isfinite(map_x)
        & np.isfinite(map_y)
        & (map_x >= 0.0)
        & (map_x <= width - 1)
        & (map_y >= 0.0)
        & (map_y <= height - 1)
    )
    return float(valid.mean())


def _pinhole_output_matrix(size_wh: tuple[int, int], focal: float) -> np.ndarray:
    width, height = size_wh
    return np.asarray(
        (
            (focal, 0.0, (width - 1.0) / 2.0),
            (0.0, focal, (height - 1.0) / 2.0),
            (0.0, 0.0, 1.0),
        ),
        dtype=np.float64,
    )


def _kalibr_cam0(path: Path) -> dict[str, Any]:
    """Read the small cam0 subset without adding a YAML runtime dependency."""
    values: dict[str, Any] = {}
    inside = False
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        if raw_line == "cam0:":
            inside = True
            continue
        if inside and raw_line and not raw_line.startswith(" "):
            break
        if not inside or not raw_line.startswith("  ") or ":" not in raw_line:
            continue
        key, raw_value = raw_line.strip().split(":", 1)
        raw_value = raw_value.strip()
        if key in {"intrinsics", "distortion_coeffs", "resolution"}:
            values[key] = ast.literal_eval(raw_value)
        elif key in {"camera_model", "distortion_model", "rostopic"}:
            values[key] = raw_value
    required = {
        "intrinsics",
        "distortion_coeffs",
        "resolution",
        "camera_model",
        "distortion_model",
        "rostopic",
    }
    missing = sorted(required - values.keys())
    if missing:
        raise ValueError(f"Kalibr cam0 fields missing from {path}: {missing}")
    return values


def guocai_equidistant_rectification(
    calibration_path: Path,
    source_size_wh: tuple[int, int],
) -> Rectification:
    """Rectify Guocai's left fisheye using its per-session Kalibr file."""
    calibration_path = calibration_path.resolve()
    camera = _kalibr_cam0(calibration_path)
    if camera["camera_model"] != "pinhole" or camera["distortion_model"] != "equidistant":
        raise ValueError(f"unsupported Guocai calibration model: {camera}")
    if camera["rostopic"] != "/fisheye/left/image_raw/compressed":
        raise ValueError(f"cam0 is not the left source camera: {camera['rostopic']}")

    width, height = source_size_wh
    calibration_width, calibration_height = (int(value) for value in camera["resolution"])
    fx, fy, cx, cy = (float(value) for value in camera["intrinsics"])
    if width != calibration_width or height < calibration_height:
        raise ValueError(
            "Guocai source/calibration resolution mismatch is not a centered vertical extension: "
            f"source={source_size_wh}, calibration={(calibration_width, calibration_height)}"
        )
    height_delta = height - calibration_height
    if height_delta % 2:
        raise ValueError(f"Guocai vertical extension must be even, got {height_delta}")
    vertical_offset = height_delta / 2.0
    source_k = np.asarray(
        ((fx, 0.0, cx), (0.0, fy, cy + vertical_offset), (0.0, 0.0, 1.0)),
        dtype=np.float64,
    )
    distortion = np.asarray(camera["distortion_coeffs"], dtype=np.float64).reshape(4, 1)
    output_k = _pinhole_output_matrix(source_size_wh, max(fx, fy))
    map_x, map_y = cv2.fisheye.initUndistortRectifyMap(
        source_k,
        distortion,
        np.eye(3, dtype=np.float64),
        output_k,
        source_size_wh,
        cv2.CV_32FC1,
    )
    valid_fraction = _map_valid_fraction(map_x, map_y, source_size_wh)
    if valid_fraction < 0.98:
        raise ValueError(f"Guocai rectification valid fraction is too low: {valid_fraction:.6f}")
    return Rectification(
        map_x=map_x,
        map_y=map_y,
        metadata={
            "output_is_undistorted": True,
            "undistortion_applied": True,
            "source_already_rectified": False,
            "model": "pinhole-equidistant",
            "method": "opencv.fisheye.initUndistortRectifyMap",
            "calibration_path": str(calibration_path),
            "calibration_sha256": _sha256(calibration_path),
            "source_size_wh": [width, height],
            "calibration_size_wh": [calibration_width, calibration_height],
            "centered_vertical_extension_pixels": height_delta,
            "source_k_adjusted": source_k.tolist(),
            "distortion_coefficients": distortion.reshape(-1).tolist(),
            "output_k": output_k.tolist(),
            "valid_pixel_fraction": valid_fraction,
        },
    )


def double_sphere_rectification(camera_info: dict[str, Any]) -> Rectification:
    """Rectify a Double-Sphere camera to a centered pinhole image.

    The destination-to-source projection follows Usenko, Demmel, and Cremers,
    3DV 2018, equation (40).  The output focal length is the larger calibrated
    focal length, retaining a wide view while keeping this dataset's full
    destination frame valid.
    """
    model = str(camera_info.get("distortion_model", "")).lower()
    if model != "ds":
        raise ValueError(f"expected Double-Sphere model 'ds', got {model!r}")
    width = int(camera_info["width"])
    height = int(camera_info["height"])
    source_size_wh = (width, height)
    intrinsic = np.asarray(camera_info["K"], dtype=np.float64).reshape(3, 3)
    parameters = np.asarray(camera_info["D"], dtype=np.float64).reshape(-1)
    if parameters.size != 6:
        raise ValueError(f"Double-Sphere D must contain [fx,fy,cx,cy,xi,alpha], got {parameters}")
    fx, fy, cx, cy, xi, alpha = (float(value) for value in parameters)
    expected_intrinsics = (
        intrinsic[0, 0],
        intrinsic[1, 1],
        intrinsic[0, 2],
        intrinsic[1, 2],
    )
    if not np.allclose(
        parameters[:4], expected_intrinsics, rtol=0.0, atol=1e-5
    ):
        raise ValueError("Double-Sphere D intrinsics do not match camera_info.K")
    if not 0.0 <= alpha <= 1.0:
        raise ValueError(f"invalid Double-Sphere alpha: {alpha}")

    output_k = _pinhole_output_matrix(source_size_wh, max(fx, fy))
    grid_x, grid_y = np.meshgrid(
        np.arange(width, dtype=np.float64),
        np.arange(height, dtype=np.float64),
    )
    x = (grid_x - output_k[0, 2]) / output_k[0, 0]
    y = (grid_y - output_k[1, 2]) / output_k[1, 1]
    z = np.ones_like(x)
    d1 = np.sqrt(x * x + y * y + z * z)
    z_shifted = xi * d1 + z
    d2 = np.sqrt(x * x + y * y + z_shifted * z_shifted)
    denominator = alpha * d2 + (1.0 - alpha) * z_shifted
    map_x = (fx * x / denominator + cx).astype(np.float32)
    map_y = (fy * y / denominator + cy).astype(np.float32)
    valid_fraction = _map_valid_fraction(map_x, map_y, source_size_wh)
    if valid_fraction < 0.98:
        raise ValueError(f"Double-Sphere rectification valid fraction is too low: {valid_fraction:.6f}")
    canonical = json.dumps(camera_info, sort_keys=True, separators=(",", ":")).encode()
    return Rectification(
        map_x=map_x,
        map_y=map_y,
        metadata={
            "output_is_undistorted": True,
            "undistortion_applied": True,
            "source_already_rectified": False,
            "model": "double-sphere",
            "method": "double-sphere destination-to-source projection (3DV 2018 eq. 40)",
            "calibration_sha256": hashlib.sha256(canonical).hexdigest(),
            "source_size_wh": [width, height],
            "source_k": intrinsic.tolist(),
            "distortion_parameters_fx_fy_cx_cy_xi_alpha": parameters.tolist(),
            "output_k": output_k.tolist(),
            "valid_pixel_fraction": valid_fraction,
        },
    )


def stera_rgb_rectification(session_path: Path, source_size_wh: tuple[int, int]) -> Rectification:
    """Load the distributed Stera RGB calibration and rectify only if needed."""
    calibration_dir = session_path.resolve() / "calibrations"
    meta_path = calibration_dir / "meta.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    rgb = meta["rgb"]
    expected_size = (int(rgb["width"]), int(rgb["height"]))
    if expected_size != source_size_wh:
        raise ValueError(f"Stera RGB calibration size mismatch: {expected_size} != {source_size_wh}")
    k_path = calibration_dir / str(rgb["files"]["K"])
    d_path = calibration_dir / str(rgb["files"]["D"])
    intrinsic = np.asarray(np.load(k_path, allow_pickle=False), dtype=np.float64).reshape(3, 3)
    distortion = np.asarray(np.load(d_path, allow_pickle=False), dtype=np.float64).reshape(-1)
    base_metadata: dict[str, Any] = {
        "output_is_undistorted": True,
        "model": str(rgb["distortion_model"]),
        "calibration_path": str(calibration_dir),
        "calibration_sha256": hashlib.sha256(
            (_sha256(meta_path) + _sha256(k_path) + _sha256(d_path)).encode()
        ).hexdigest(),
        "source_size_wh": list(source_size_wh),
        "source_k": intrinsic.tolist(),
        "distortion_coefficients": distortion.tolist(),
        "output_k": intrinsic.tolist(),
    }
    if distortion.size == 0 or np.allclose(distortion, 0.0):
        return Rectification(
            map_x=None,
            map_y=None,
            metadata={
                **base_metadata,
                "undistortion_applied": False,
                "source_already_rectified": True,
                "method": "identity (distributed RGB has no distortion coefficients)",
                "valid_pixel_fraction": 1.0,
            },
        )
    if str(rgb["distortion_model"]) != "plumb_bob":
        raise ValueError(f"unsupported Stera RGB distortion model: {rgb['distortion_model']}")
    map_x, map_y = cv2.initUndistortRectifyMap(
        intrinsic,
        distortion,
        np.eye(3, dtype=np.float64),
        intrinsic,
        source_size_wh,
        cv2.CV_32FC1,
    )
    return Rectification(
        map_x=map_x,
        map_y=map_y,
        metadata={
            **base_metadata,
            "undistortion_applied": True,
            "source_already_rectified": False,
            "method": "opencv.initUndistortRectifyMap",
            "valid_pixel_fraction": _map_valid_fraction(map_x, map_y, source_size_wh),
        },
    )
