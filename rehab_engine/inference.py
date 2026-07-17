"""Python-owned ONNX deployment for person detection and RTMPose.

The preprocessing and postprocessing mirror the original C++ implementations
in ``core/pose/PersonDetectorOrt`` and ``PoseEstimatorRTMPoseOrt``.  The native
extension remains usable for camera/depth acceleration, but model lifecycle and
inference policy are owned by the Python application.
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, List, Optional, Sequence, Tuple

import cv2
import numpy as np


@dataclass(frozen=True)
class BoundingBox2D:
    x: float = 0.0
    y: float = 0.0
    w: float = 0.0
    h: float = 0.0
    score: float = 0.0
    valid: bool = False


@dataclass(frozen=True)
class Keypoint2D:
    x: float = 0.0
    y: float = 0.0
    score: float = 0.0
    raw_score: float = 0.0
    valid: bool = False


@dataclass
class PoseInferenceResult:
    keypoints: List[Keypoint2D] = field(default_factory=lambda: [Keypoint2D() for _ in range(26)])
    used_box: BoundingBox2D = field(default_factory=BoundingBox2D)
    model_loaded: bool = False
    valid_count: int = 0
    mean_score: float = 0.0
    bbox_ms: float = 0.0
    pose_ms: float = 0.0


@dataclass(frozen=True)
class _LetterboxInfo:
    scale: float
    pad_x: float
    pad_y: float
    input_size: int


class OrtSessionFactory:
    """Create ORT sessions with deterministic thread/provider settings."""

    @staticmethod
    def create(model_path: str, config: Any):
        import onnxruntime as ort

        options = ort.SessionOptions()
        options.intra_op_num_threads = max(1, int(getattr(config, "onnx_intra_op_threads", 1)))
        options.inter_op_num_threads = max(1, int(getattr(config, "onnx_inter_op_threads", 1)))
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED
        requested = str(getattr(config, "onnx_execution_provider", "auto")).strip()
        available = ort.get_available_providers()
        if requested and requested.lower() != "auto":
            providers = [requested] if requested in available else ["CPUExecutionProvider"]
        else:
            preference = [
                "CUDAExecutionProvider",
                "OpenVINOExecutionProvider",
                "CoreMLExecutionProvider",
                "CPUExecutionProvider",
            ]
            providers = [name for name in preference if name in available]
        return ort.InferenceSession(model_path, sess_options=options, providers=providers)


class PersonDetector:
    COCO_PERSON_CLASS = 0
    COCO_CLASS_COUNT = 80

    def __init__(self, config: Any, session=None):
        self.config = config
        self.session = session
        self.input_name = ""
        self.output_name = ""
        self.initialized = False

    def initialize(self, model_path: str) -> bool:
        path = Path(model_path)
        if self.session is None:
            if not path.is_file():
                return False
            try:
                self.session = OrtSessionFactory.create(str(path), self.config)
            except (ImportError, OSError, RuntimeError):
                return False
        try:
            inputs = self.session.get_inputs()
            outputs = self.session.get_outputs()
            if not inputs or not outputs:
                return False
            self.input_name = inputs[0].name
            self.output_name = outputs[0].name
            self.initialized = bool(self.input_name and self.output_name)
            return self.initialized
        except Exception:
            return False

    def preprocess(self, bgr: np.ndarray) -> Tuple[np.ndarray, _LetterboxInfo]:
        if bgr is None or bgr.size == 0:
            raise ValueError("empty detector image")
        input_size = max(1, int(getattr(self.config, "detector_input_size", 320)))
        height, width = bgr.shape[:2]
        scale = min(input_size / width, input_size / height)
        resized_w = max(1, int(round(width * scale)))
        resized_h = max(1, int(round(height * scale)))
        resized = cv2.resize(bgr, (resized_w, resized_h), interpolation=cv2.INTER_LINEAR)
        canvas = np.full((input_size, input_size, 3), 114, dtype=np.uint8)
        pad_x = max(0, (input_size - resized_w) // 2)
        pad_y = max(0, (input_size - resized_h) // 2)
        copy_w = min(resized_w, input_size - pad_x)
        copy_h = min(resized_h, input_size - pad_y)
        canvas[pad_y:pad_y + copy_h, pad_x:pad_x + copy_w] = resized[:copy_h, :copy_w]
        rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
        tensor = rgb.astype(np.float32).transpose(2, 0, 1)[None] / 255.0
        return tensor, _LetterboxInfo(scale, float(pad_x), float(pad_y), input_size)

    @staticmethod
    def _layout(output: np.ndarray) -> Tuple[np.ndarray, int, int]:
        array = np.asarray(output)
        while array.ndim > 2 and array.shape[0] == 1:
            array = array[0]
        if array.ndim != 2:
            raise ValueError(f"unsupported YOLO output shape {array.shape}")
        dim_a, dim_b = array.shape
        attr_first = (dim_a <= 256 and dim_b > 256) or (
            not (dim_b <= 256 and dim_a > 256) and dim_a <= dim_b
        )
        rows = array.T if attr_first else array
        return rows, rows.shape[0], rows.shape[1]

    def decode(
        self, output: np.ndarray, info: _LetterboxInfo, image_shape: Sequence[int]
    ) -> List[BoundingBox2D]:
        rows, _box_count, attr_count = self._layout(output)
        if attr_count < 5:
            return []
        has_objectness = attr_count >= 5 + self.COCO_CLASS_COUNT
        class_start = 5 if has_objectness else 4
        confidence_threshold = float(getattr(self.config, "detector_conf_threshold", 0.35))
        image_h, image_w = int(image_shape[0]), int(image_shape[1])
        boxes: List[BoundingBox2D] = []
        for row in rows:
            cx, cy, width, height = map(float, row[:4])
            if width <= 1e-6 or height <= 1e-6:
                continue
            class_scores = row[class_start:]
            if not class_scores.size:
                continue
            best_class = int(np.argmax(class_scores))
            if best_class != self.COCO_PERSON_CLASS:
                continue
            objectness = float(row[4]) if has_objectness else 1.0
            score = objectness * float(class_scores[best_class])
            if score < confidence_threshold:
                continue
            x1 = (cx - width * 0.5 - info.pad_x) / info.scale
            y1 = (cy - height * 0.5 - info.pad_y) / info.scale
            x2 = (cx + width * 0.5 - info.pad_x) / info.scale
            y2 = (cy + height * 0.5 - info.pad_y) / info.scale
            x1 = float(np.clip(x1, 0, image_w - 1))
            y1 = float(np.clip(y1, 0, image_h - 1))
            x2 = float(np.clip(x2, 0, image_w))
            y2 = float(np.clip(y2, 0, image_h))
            if x2 - x1 > 1 and y2 - y1 > 1:
                boxes.append(BoundingBox2D(x1, y1, x2 - x1, y2 - y1, score, True))
        return self._nms(boxes)

    def _nms(self, boxes: List[BoundingBox2D]) -> List[BoundingBox2D]:
        threshold = float(getattr(self.config, "detector_nms_threshold", 0.45))
        picked: List[BoundingBox2D] = []
        for candidate in sorted(boxes, key=lambda item: item.score, reverse=True):
            if all(self._iou(candidate, item) <= threshold for item in picked):
                picked.append(candidate)
        return picked

    @staticmethod
    def _iou(a: BoundingBox2D, b: BoundingBox2D) -> float:
        x1, y1 = max(a.x, b.x), max(a.y, b.y)
        x2, y2 = min(a.x + a.w, b.x + b.w), min(a.y + a.h, b.y + b.h)
        intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
        denominator = a.w * a.h + b.w * b.h - intersection
        return intersection / denominator if denominator > 1e-6 else 0.0

    def detect(self, bgr: np.ndarray) -> List[BoundingBox2D]:
        if not self.initialized:
            return []
        tensor, info = self.preprocess(bgr)
        output = self.session.run([self.output_name], {self.input_name: tensor})[0]
        return self.decode(output, info, bgr.shape)

    def detect_largest(self, bgr: np.ndarray) -> BoundingBox2D:
        boxes = self.detect(bgr)
        return max(boxes, key=lambda item: item.w * item.h) if boxes else BoundingBox2D()


class AdaptiveRoiTracker:
    """Low-frequency YOLO + pose-feedback ROI ported from the C++ provider."""

    def __init__(self, detector: PersonDetector, config: Any):
        self.detector = detector
        self.config = config
        self.reset()

    def reset(self) -> None:
        self.tracked_box = BoundingBox2D()
        self.frame_counter = 0
        self.miss_counter = 0
        self.force_redetect = True
        self.last_image_size = (0, 0)
        self.last_state = "detect"

    @property
    def debug_state(self) -> str:
        return self.last_state

    def _expand_and_clip(
        self, box: BoundingBox2D, image_size: Tuple[int, int]
    ) -> BoundingBox2D:
        width, height = image_size
        if not box.valid or width <= 0 or height <= 0:
            return BoundingBox2D()
        margin = max(0.0, float(getattr(self.config, "roi_margin_ratio", 0.20)))
        x1 = float(np.clip(box.x - box.w * margin, 0.0, width - 1.0))
        y1 = float(np.clip(box.y - box.h * margin, 0.0, height - 1.0))
        x2 = float(np.clip(box.x + box.w * (1.0 + margin), 1.0, width))
        y2 = float(np.clip(box.y + box.h * (1.0 + margin), 1.0, height))
        return BoundingBox2D(x1, y1, x2 - x1, y2 - y1, box.score, x2 - x1 > 1 and y2 - y1 > 1)

    @staticmethod
    def _pose_box(keypoints: Sequence[Keypoint2D]) -> Tuple[BoundingBox2D, float, int]:
        valid = [point for point in keypoints if point.valid]
        if not valid:
            return BoundingBox2D(), 0.0, 0
        min_x, max_x = min(point.x for point in valid), max(point.x for point in valid)
        min_y, max_y = min(point.y for point in valid), max(point.y for point in valid)
        mean = sum(point.score for point in valid) / len(valid)
        box = BoundingBox2D(min_x, min_y, max_x - min_x, max_y - min_y, mean,
                            max_x - min_x > 1 and max_y - min_y > 1)
        return box, mean, len(valid)

    def get_primary_box(self, bgr: np.ndarray) -> BoundingBox2D:
        if bgr is None or bgr.size == 0:
            self.last_state = "full_fallback"
            return BoundingBox2D()
        height, width = bgr.shape[:2]
        self.last_image_size = (width, height)
        self.frame_counter += 1
        interval = max(1, int(getattr(self.config, "detector_interval", 30)))
        periodic = ((self.frame_counter - 1) % interval) == 0
        should_detect = (
            self.force_redetect
            or not self.tracked_box.valid
            or self.frame_counter <= 1
            or periodic
        )
        if should_detect and self.detector is not None and self.detector.initialized:
            detected = self.detector.detect_largest(bgr)
            if detected.valid:
                self.tracked_box = self._expand_and_clip(detected, self.last_image_size)
                self.force_redetect = False
                self.last_state = "detect"
                return self.tracked_box
            if self.tracked_box.valid:
                self.last_state = "track_fallback"
                return self.tracked_box
            self.last_state = "no_person"
            return BoundingBox2D()
        if self.tracked_box.valid:
            self.last_state = "track_fallback" if should_detect else "track"
            return self.tracked_box
        self.last_state = "no_person"
        return BoundingBox2D()

    def update_from_pose(
        self, used_box: BoundingBox2D, keypoints: Sequence[Keypoint2D]
    ) -> None:
        pose_box, mean_score, valid_count = self._pose_box(keypoints)
        weak = (
            mean_score < float(getattr(self.config, "min_track_mean_score", 0.25))
            or valid_count < int(getattr(self.config, "min_track_valid_points", 6))
        )
        if weak:
            self.miss_counter += 1
        if self.miss_counter >= int(getattr(self.config, "max_consecutive_misses", 3)):
            self.force_redetect = True
        if not pose_box.valid:
            self.force_redetect = True
            return
        dx = (pose_box.x + pose_box.w * 0.5) - (used_box.x + used_box.w * 0.5)
        dy = (pose_box.y + pose_box.h * 0.5) - (used_box.y + used_box.h * 0.5)
        threshold = float(getattr(self.config, "motion_trigger_ratio", 0.35)) * max(
            used_box.w, used_box.h, 1.0
        )
        if math.sqrt(dx * dx + dy * dy) > threshold:
            self.force_redetect = True
        if not weak:
            self.miss_counter = 0
            self.tracked_box = self._expand_and_clip(pose_box, self.last_image_size)


@dataclass
class RtmposeRuntimeParams:
    input_width: int = 192
    input_height: int = 256
    padding: float = 1.25
    simcc_split_ratio: float = 2.0
    mean: Tuple[float, float, float] = (123.675, 116.28, 103.53)
    std: Tuple[float, float, float] = (58.395, 57.12, 57.375)
    to_rgb: bool = True


class RtmposeEstimator:
    def __init__(self, config: Any, session=None):
        self.config = config
        self.session = session
        self.params = RtmposeRuntimeParams()
        self.input_name = "input"
        self.output_names = ["simcc_x", "simcc_y"]
        self.initialized = False

    def initialize(
        self,
        model_path: str,
        pipeline_json_path: str = "",
        detail_json_path: str = "",
    ) -> bool:
        self._load_runtime_params(pipeline_json_path, detail_json_path)
        if self.session is None:
            if not Path(model_path).is_file():
                return False
            try:
                self.session = OrtSessionFactory.create(model_path, self.config)
            except (ImportError, OSError, RuntimeError):
                return False
        try:
            inputs = [item.name for item in self.session.get_inputs()]
            outputs = [item.name for item in self.session.get_outputs()]
            if "input" not in inputs or "simcc_x" not in outputs or "simcc_y" not in outputs:
                return False
            self.initialized = True
            return True
        except Exception:
            return False

    def _load_runtime_params(self, pipeline_path: str, detail_path: str) -> None:
        for path_string in (pipeline_path, detail_path):
            if not path_string or not Path(path_string).is_file():
                continue
            try:
                data = json.loads(Path(path_string).read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            self._walk_runtime_config(data)

    def _walk_runtime_config(self, value: Any) -> None:
        if isinstance(value, dict):
            if "image_size" in value and len(value["image_size"]) >= 2:
                self.params.input_width, self.params.input_height = map(int, value["image_size"][:2])
            if "input_shape" in value and len(value["input_shape"]) >= 2:
                self.params.input_width, self.params.input_height = map(int, value["input_shape"][:2])
            if "padding" in value:
                self.params.padding = float(value["padding"])
            if "simcc_split_ratio" in value:
                self.params.simcc_split_ratio = float(value["simcc_split_ratio"])
            if "mean" in value and len(value["mean"]) >= 3:
                self.params.mean = tuple(map(float, value["mean"][:3]))
            if "std" in value and len(value["std"]) >= 3:
                self.params.std = tuple(map(float, value["std"][:3]))
            if "to_rgb" in value:
                self.params.to_rgb = bool(value["to_rgb"])
            for child in value.values():
                self._walk_runtime_config(child)
        elif isinstance(value, list):
            for child in value:
                self._walk_runtime_config(child)

    @staticmethod
    def _third_point(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        direction = a - b
        return b + np.array([-direction[1], direction[0]], dtype=np.float32)

    @staticmethod
    def sanitize_box(box: BoundingBox2D, image_shape: Sequence[int]) -> BoundingBox2D:
        image_h, image_w = int(image_shape[0]), int(image_shape[1])
        if not box.valid or box.w <= 1 or box.h <= 1:
            return BoundingBox2D(0, 0, float(image_w), float(image_h), 1.0, True)
        x = float(np.clip(box.x, 0, image_w - 1))
        y = float(np.clip(box.y, 0, image_h - 1))
        width = min(max(1.0, box.w), image_w - x)
        height = min(max(1.0, box.h), image_h - y)
        return BoundingBox2D(x, y, width, height, box.score, width > 1 and height > 1)

    def preprocess(self, bgr: np.ndarray, box: BoundingBox2D) -> Tuple[np.ndarray, np.ndarray]:
        box = self.sanitize_box(box, bgr.shape)
        aspect = self.params.input_width / self.params.input_height
        center = np.array([box.x + box.w * 0.5, box.y + box.h * 0.5], dtype=np.float32)
        width, height = box.w, box.h
        if width > aspect * height:
            height = width / aspect
        else:
            width = height * aspect
        width *= self.params.padding
        height *= self.params.padding
        source_direction = np.array([0.0, -0.5 * width], dtype=np.float32)
        target_center = np.array(
            [self.params.input_width * 0.5, self.params.input_height * 0.5], dtype=np.float32
        )
        target_direction = np.array([0.0, -0.5 * self.params.input_width], dtype=np.float32)
        source = np.stack(
            [center, center + source_direction, self._third_point(center, center + source_direction)]
        )
        target = np.stack(
            [
                target_center,
                target_center + target_direction,
                self._third_point(target_center, target_center + target_direction),
            ]
        )
        affine = cv2.getAffineTransform(source.astype(np.float32), target.astype(np.float32))
        warped = cv2.warpAffine(
            bgr,
            affine,
            (self.params.input_width, self.params.input_height),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0),
        ).astype(np.float32)
        inverse = cv2.invertAffineTransform(affine).astype(np.float32)
        if self.params.to_rgb:
            warped = cv2.cvtColor(warped, cv2.COLOR_BGR2RGB)
        normalized = (warped - np.asarray(self.params.mean, dtype=np.float32)) / np.asarray(
            self.params.std, dtype=np.float32
        )
        return normalized.transpose(2, 0, 1)[None].astype(np.float32), inverse

    def decode_simcc(
        self, simcc_x: np.ndarray, simcc_y: np.ndarray, inverse: np.ndarray
    ) -> List[Keypoint2D]:
        x = np.asarray(simcc_x)
        y = np.asarray(simcc_y)
        while x.ndim > 2 and x.shape[0] == 1:
            x = x[0]
        while y.ndim > 2 and y.shape[0] == 1:
            y = y[0]
        if x.ndim != 2 or y.ndim != 2:
            raise ValueError("invalid SimCC output shape")
        joint_count = min(26, x.shape[0], y.shape[0])
        minimum_score = float(getattr(self.config, "min_score", 0.15))
        points = [Keypoint2D() for _ in range(26)]
        for joint in range(joint_count):
            x_index, y_index = int(np.argmax(x[joint])), int(np.argmax(y[joint]))
            x_score, y_score = float(x[joint, x_index]), float(y[joint, y_index])
            px = x_index / self.params.simcc_split_ratio
            py = y_index / self.params.simcc_split_ratio
            image_x = float(inverse[0, 0] * px + inverse[0, 1] * py + inverse[0, 2])
            image_y = float(inverse[1, 0] * px + inverse[1, 1] * py + inverse[1, 2])
            score = 0.5 * (x_score + y_score)
            points[joint] = Keypoint2D(image_x, image_y, score, score, score >= minimum_score)
        return points

    def infer(self, bgr: np.ndarray, requested_box: BoundingBox2D) -> PoseInferenceResult:
        result = PoseInferenceResult(model_loaded=self.initialized)
        if not self.initialized or bgr is None or bgr.size == 0:
            return result
        box = self.sanitize_box(requested_box, bgr.shape)
        result.used_box = box
        started = time.monotonic()
        tensor, inverse = self.preprocess(bgr, box)
        outputs = self.session.run(self.output_names, {self.input_name: tensor})
        result.keypoints = self.decode_simcc(outputs[0], outputs[1], inverse)
        result.valid_count = sum(point.valid for point in result.keypoints)
        result.mean_score = (
            sum(point.score for point in result.keypoints if point.valid) / result.valid_count
            if result.valid_count
            else 0.0
        )
        result.pose_ms = (time.monotonic() - started) * 1000.0
        return result


def _interpolate(a: Keypoint2D, b: Keypoint2D, fraction: float) -> Keypoint2D:
    if not a.valid or not b.valid:
        return Keypoint2D()
    return Keypoint2D(
        a.x * (1 - fraction) + b.x * fraction,
        a.y * (1 - fraction) + b.y * fraction,
        a.score * (1 - fraction) + b.score * fraction,
        a.raw_score * (1 - fraction) + b.raw_score * fraction,
        True,
    )


def map_halpe26_to_rehab22(points: Sequence[Keypoint2D]) -> List[Keypoint2D]:
    """Map Halpe26 indices to the exact Rehab22 layout used by the C++ app."""
    source = list(points) + [Keypoint2D()] * max(0, 26 - len(points))
    midpoint = lambda a, b: _interpolate(a, b, 0.5)
    fallback = lambda primary, secondary: primary if primary.valid else secondary
    nose, left_shoulder, right_shoulder = source[0], source[5], source[6]
    left_hip, right_hip = source[11], source[12]
    head, neck, hip = source[17], source[18], source[19]
    chest = midpoint(left_shoulder, right_shoulder)
    waist = fallback(hip, midpoint(left_hip, right_hip))
    output = [Keypoint2D() for _ in range(22)]
    output[0], output[1], output[2] = waist, _interpolate(waist, chest, 0.5), chest
    output[3], output[4], output[5] = neck, nose, head
    output[6], output[10] = _interpolate(neck, left_shoulder, 0.35), _interpolate(neck, right_shoulder, 0.35)
    output[7], output[8], output[9] = left_shoulder, source[7], source[9]
    output[11], output[12], output[13] = right_shoulder, source[8], source[10]
    output[14], output[15], output[18], output[19] = left_hip, source[13], right_hip, source[14]
    output[16] = fallback(midpoint(source[15], source[24]), source[15])
    output[20] = fallback(midpoint(source[16], source[25]), source[16])
    output[17], output[21] = midpoint(source[20], source[22]), midpoint(source[21], source[23])
    return output


__all__ = [
    "BoundingBox2D",
    "AdaptiveRoiTracker",
    "Keypoint2D",
    "OrtSessionFactory",
    "PersonDetector",
    "PoseInferenceResult",
    "RtmposeEstimator",
    "RtmposeRuntimeParams",
    "map_halpe26_to_rehab22",
]
