"""Load the shipped ONNX models and execute one hardware-free inference."""

import json
import sys
from pathlib import Path

import numpy as np

from rehab_engine import PoseConfig
from rehab_engine.inference import BoundingBox2D, PersonDetector, RtmposeEstimator


def main() -> int:
    repo_root = Path(__file__).resolve().parent.parent
    model_root = repo_root / "stroke-rehab" / "including"
    detector_path = model_root / "yolov8n" / "yolov8n.onnx"
    pose_root = model_root / "rtmpose-t"
    pose_path = pose_root / "end2end.onnx"
    config = PoseConfig(inference_backend="python")

    detector = PersonDetector(config)
    pose = RtmposeEstimator(config)
    detector_ok = detector.initialize(str(detector_path))
    pose_ok = pose.initialize(
        str(pose_path), str(pose_root / "pipeline.json"), str(pose_root / "detail.json")
    )
    if not detector_ok or not pose_ok:
        print(
            json.dumps(
                {
                    "ok": False,
                    "detector_loaded": detector_ok,
                    "pose_loaded": pose_ok,
                    "detector_path": str(detector_path),
                    "pose_path": str(pose_path),
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 1

    image = np.zeros((480, 640, 3), dtype=np.uint8)
    image[80:440, 230:410] = 180
    detected = detector.detect_largest(image)
    roi = detected if detected.valid else BoundingBox2D(0, 0, 640, 480, 1.0, True)
    result = pose.infer(image, roi)
    report = {
        "ok": result.model_loaded and len(result.keypoints) == 26,
        "detector_loaded": detector_ok,
        "pose_loaded": pose_ok,
        "detector_input": detector.input_name,
        "detector_output": detector.output_name,
        "pose_input": pose.input_name,
        "pose_outputs": pose.output_names,
        "detector_providers": detector.session.get_providers(),
        "pose_providers": pose.session.get_providers(),
        "person_detected_on_synthetic_frame": detected.valid,
        "pose_valid_keypoints": result.valid_count,
        "pose_mean_score": result.mean_score,
        "pose_ms": result.pose_ms,
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
