from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from egohandmetric_prompt.camera_undistortion import (
    double_sphere_rectification,
    guocai_equidistant_rectification,
    stera_rgb_rectification,
)
from egohandmetric_prompt.inference_visualization import _contact_distance_colors


class CameraUndistortionTest(unittest.TestCase):
    def test_double_sphere_pinhole_limit_is_identity(self) -> None:
        width, height = 9, 7
        focal = 5.0
        cx, cy = (width - 1) / 2, (height - 1) / 2
        rectification = double_sphere_rectification(
            {
                "width": width,
                "height": height,
                "distortion_model": "ds",
                "K": [focal, 0, cx, 0, focal, cy, 0, 0, 1],
                "D": [focal, focal, cx, cy, 0.0, 0.0],
            }
        )
        grid_x, grid_y = np.meshgrid(np.arange(width), np.arange(height))
        np.testing.assert_allclose(rectification.map_x, grid_x, atol=1e-6)
        np.testing.assert_allclose(rectification.map_y, grid_y, atol=1e-6)

    def test_guocai_centered_vertical_extension_adjusts_principal_point(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "camera-camchain.yaml"
            path.write_text(
                """cam0:
  camera_model: pinhole
  distortion_coeffs: [0.0, 0.0, 0.0, 0.0]
  distortion_model: equidistant
  intrinsics: [5.0, 5.0, 4.0, 3.5]
  resolution: [9, 8]
  rostopic: /fisheye/left/image_raw/compressed
cam1:
  camera_model: pinhole
""",
                encoding="utf-8",
            )
            rectification = guocai_equidistant_rectification(path, (9, 10))
            self.assertEqual(
                rectification.metadata["centered_vertical_extension_pixels"], 2
            )
            self.assertEqual(rectification.metadata["source_k_adjusted"][1][2], 4.5)

    def test_stera_empty_distortion_is_already_rectified(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            session = Path(temporary)
            calibration = session / "calibrations"
            calibration.mkdir()
            (calibration / "meta.json").write_text(
                json.dumps(
                    {
                        "rgb": {
                            "width": 8,
                            "height": 6,
                            "distortion_model": "plumb_bob",
                            "files": {"K": "rgb_K.npy", "D": "rgb_D.npy"},
                        }
                    }
                ),
                encoding="utf-8",
            )
            np.save(calibration / "rgb_K.npy", np.eye(3))
            np.save(calibration / "rgb_D.npy", np.asarray([], dtype=np.float32))
            rectification = stera_rgb_rectification(session, (8, 6))
            self.assertIsNone(rectification.map_x)
            self.assertTrue(rectification.metadata["source_already_rectified"])
            self.assertTrue(rectification.metadata["output_is_undistorted"])

    def test_distance_colormap_is_hot_at_zero_and_cold_at_max(self) -> None:
        colors = _contact_distance_colors(np.asarray([0.0, 0.05], dtype=np.float32))
        self.assertGreater(int(colors[0, 0]), int(colors[0, 2]))
        self.assertGreater(int(colors[1, 2]), int(colors[1, 0]))


if __name__ == "__main__":
    unittest.main()
