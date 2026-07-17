import unittest
from pathlib import Path

import numpy as np

from rehab_engine import PoseConfig
from rehab_engine.inference import (
    AdaptiveRoiTracker,
    BoundingBox2D,
    Keypoint2D,
    PersonDetector,
    RtmposeEstimator,
    map_halpe26_to_rehab22,
)


class _Node:
    def __init__(self, name):
        self.name = name


class _DetectorSession:
    def get_inputs(self):
        return [_Node("images")]

    def get_outputs(self):
        return [_Node("output0")]

    def run(self, outputs, feeds):
        # Keep the same shape discriminator used by the C++ implementation:
        # attributes <= 256, candidate boxes > 256.
        result = np.zeros((1, 84, 300), dtype=np.float32)
        result[0, :4, 0] = [160, 160, 100, 200]
        result[0, 4, 0] = 0.9
        result[0, :4, 1] = [162, 162, 96, 196]
        result[0, 4, 1] = 0.8
        return [result]


class _PoseSession:
    def get_inputs(self):
        return [_Node("input")]

    def get_outputs(self):
        return [_Node("simcc_x"), _Node("simcc_y")]

    def run(self, outputs, feeds):
        x = np.zeros((1, 26, 384), dtype=np.float32)
        y = np.zeros((1, 26, 512), dtype=np.float32)
        for joint in range(26):
            x[0, joint, 100 + joint] = 0.9
            y[0, joint, 200 + joint] = 0.8
        return [x, y]


class PersonDetectorTests(unittest.TestCase):
    def test_letterbox_decode_nms_and_largest_person_match_cpp_flow(self):
        config = PoseConfig(detector_input_size=320)
        detector = PersonDetector(config, _DetectorSession())
        self.assertTrue(detector.initialize("injected.onnx"))
        image = np.zeros((480, 640, 3), dtype=np.uint8)
        tensor, info = detector.preprocess(image)
        self.assertEqual(tensor.shape, (1, 3, 320, 320))
        self.assertEqual(info.pad_y, 40)
        boxes = detector.detect(image)
        self.assertEqual(len(boxes), 1)
        self.assertAlmostEqual(boxes[0].x, 220.0, places=3)
        self.assertAlmostEqual(boxes[0].y, 40.0, places=3)
        self.assertAlmostEqual(boxes[0].w, 200.0, places=3)

    def test_adaptive_roi_uses_periodic_detector_and_pose_feedback(self):
        class Detector:
            initialized = True

            def __init__(self):
                self.calls = 0

            def detect_largest(self, _image):
                self.calls += 1
                return BoundingBox2D(100, 50, 200, 300, 0.9, True)

        config = PoseConfig(
            detector_interval=3,
            roi_margin_ratio=0.2,
            min_track_mean_score=0.25,
            min_track_valid_points=6,
            max_consecutive_misses=2,
        )
        detector = Detector()
        tracker = AdaptiveRoiTracker(detector, config)
        image = np.zeros((480, 640, 3), dtype=np.uint8)
        first = tracker.get_primary_box(image)
        self.assertEqual(detector.calls, 1)
        self.assertEqual(tracker.debug_state, "detect")
        points = [Keypoint2D(140 + i * 5, 100 + i * 8, 0.9, 0.9, True) for i in range(8)]
        tracker.update_from_pose(first, points)
        tracked = tracker.get_primary_box(image)
        self.assertEqual(detector.calls, 1)
        self.assertEqual(tracker.debug_state, "track")
        tracker.get_primary_box(image)
        tracker.get_primary_box(image)
        self.assertEqual(detector.calls, 2)

        weak = [Keypoint2D() for _ in range(26)]
        tracker.update_from_pose(tracked, weak)
        tracker.update_from_pose(tracked, weak)
        tracker.get_primary_box(image)
        self.assertEqual(detector.calls, 3)


class RtmposeTests(unittest.TestCase):
    def test_runtime_json_affine_and_simcc_decode(self):
        root = Path(__file__).resolve().parent.parent / "stroke-rehab" / "including" / "rtmpose-t"
        estimator = RtmposeEstimator(PoseConfig(min_score=0.15), _PoseSession())
        self.assertTrue(
            estimator.initialize(
                "injected.onnx", str(root / "pipeline.json"), str(root / "detail.json")
            )
        )
        self.assertEqual((estimator.params.input_width, estimator.params.input_height), (192, 256))
        image = np.zeros((480, 640, 3), dtype=np.uint8)
        result = estimator.infer(image, BoundingBox2D(100, 50, 300, 400, 0.9, True))
        self.assertTrue(result.model_loaded)
        self.assertEqual(result.valid_count, 26)
        self.assertEqual(len(result.keypoints), 26)
        self.assertGreater(result.keypoints[0].score, 0.8)

    def test_halpe26_mapping_preserves_original_rehab22_indices(self):
        points = [Keypoint2D(float(i), float(i * 2), 1.0, 1.0, True) for i in range(26)]
        rehab = map_halpe26_to_rehab22(points)
        self.assertEqual(len(rehab), 22)
        self.assertEqual(rehab[0].x, 19.0)  # Halpe hip
        self.assertEqual(rehab[7].x, 5.0)   # left shoulder
        self.assertEqual(rehab[13].x, 10.0) # right wrist
        self.assertEqual(rehab[17].x, 21.0) # left big/small toe midpoint


if __name__ == "__main__":
    unittest.main(verbosity=2)
