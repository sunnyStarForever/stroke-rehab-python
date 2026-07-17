import unittest
from pathlib import Path

import numpy as np

from rehab_engine.alignment import (
    SoftwareRegistrationAligner,
    calibration_from_dict,
    load_calibration,
)


def _calibration(depth_fx=1.0, rgb_fx=1.0, translation=(0, 0, 0), unit="m"):
    return calibration_from_dict(
        {
            "depth_intrinsics": {"fx": depth_fx, "fy": 1, "cx": 0, "cy": 0},
            "rgb_intrinsics": {"fx": rgb_fx, "fy": 1, "cx": 0, "cy": 0},
            "depth_to_rgb_extrinsics": {
                "R": [1, 0, 0, 0, 1, 0, 0, 0, 1],
                "T": list(translation),
                "translation_unit": unit,
            },
        }
    )


class SoftwareRegistrationTests(unittest.TestCase):
    def test_identity_calibration_preserves_depth_pixels(self):
        depth = np.array([[1000, 0], [0, 1500]], dtype=np.uint16)
        aligned = SoftwareRegistrationAligner(_calibration()).align(depth, (2, 2))
        np.testing.assert_array_equal(aligned, depth)

    def test_full_frame_sparse_indices_remain_two_dimensional(self):
        depth = np.zeros((480, 640), dtype=np.uint16)
        depth[327, 615] = 1250
        aligned = SoftwareRegistrationAligner(_calibration()).align(depth, (640, 480))
        self.assertEqual(aligned[327, 615], 1250)

    def test_millimeter_translation_is_converted_to_meters(self):
        calibration = _calibration(
            depth_fx=100, rgb_fx=100, translation=(10, 0, 0), unit="mm"
        )
        self.assertAlmostEqual(calibration.translation_m[0], 0.01)
        depth = np.zeros((3, 3), dtype=np.uint16)
        depth[1, 0] = 1000
        aligned = SoftwareRegistrationAligner(calibration).align(depth, (3, 3))
        self.assertEqual(aligned[1, 1], 1000)

    def test_z_buffer_keeps_nearest_depth_when_points_collide(self):
        depth = np.zeros((1, 2), dtype=np.uint16)
        depth[0, 0] = 2000
        depth[0, 1] = 1000
        aligned = SoftwareRegistrationAligner(
            _calibration(depth_fx=1, rgb_fx=0.1)
        ).align(depth, (1, 1))
        self.assertEqual(aligned[0, 0], 1000)

    def test_missing_calibration_uses_nearest_resize(self):
        depth = np.array([[500]], dtype=np.uint16)
        aligned = SoftwareRegistrationAligner(None).align(depth, (2, 2))
        np.testing.assert_array_equal(aligned, np.full((2, 2), 500, dtype=np.uint16))

    def test_shipped_calibration_loads_uppercase_keys_and_explicit_mm_unit(self):
        path = Path(__file__).resolve().parent / "configs" / "calibration.yaml"
        calibration = load_calibration(str(path))
        self.assertIsNotNone(calibration)
        self.assertTrue(calibration.valid)
        self.assertAlmostEqual(calibration.translation_m[0], 0.0264757200251)


if __name__ == "__main__":
    unittest.main(verbosity=2)
