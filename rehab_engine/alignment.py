"""RGB-D software registration owned by the Python pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Sequence, Tuple

import cv2
import numpy as np

try:
    import yaml
except ImportError:
    yaml = None


@dataclass(frozen=True)
class CameraIntrinsics:
    fx: float
    fy: float
    cx: float
    cy: float
    width: int = 0
    height: int = 0

    @property
    def valid(self) -> bool:
        return self.fx > 0 and self.fy > 0


@dataclass(frozen=True)
class RegistrationCalibration:
    depth: CameraIntrinsics
    rgb: CameraIntrinsics
    rotation: np.ndarray
    translation_m: np.ndarray
    source_path: str = ""

    @property
    def valid(self) -> bool:
        return (
            self.depth.valid
            and self.rgb.valid
            and self.rotation.shape == (3, 3)
            and self.translation_m.shape == (3,)
            and np.isfinite(self.rotation).all()
            and np.isfinite(self.translation_m).all()
        )


def _intrinsics(values: dict) -> CameraIntrinsics:
    return CameraIntrinsics(
        float(values.get("fx", 0)),
        float(values.get("fy", 0)),
        float(values.get("cx", 0)),
        float(values.get("cy", 0)),
        int(values.get("width", 0)),
        int(values.get("height", 0)),
    )


def calibration_from_dict(data: dict, source_path: str = "") -> RegistrationCalibration:
    extrinsics = data.get("depth_to_rgb_extrinsics", {})
    rotation_values = extrinsics.get("R", extrinsics.get("r", []))
    translation_values = extrinsics.get("T", extrinsics.get("t", []))
    rotation = np.asarray(rotation_values, dtype=np.float64)
    translation = np.asarray(translation_values, dtype=np.float64)
    if rotation.size == 9:
        rotation = rotation.reshape(3, 3)
    if translation.size == 3:
        translation = translation.reshape(3)
    unit = str(extrinsics.get("translation_unit", "auto")).lower()
    if translation.size == 3:
        if unit in ("mm", "millimeter", "millimeters"):
            translation = translation * 0.001
        elif unit == "auto" and float(np.max(np.abs(translation))) > 1.0:
            # Camera baselines are normally centimetres; values such as 26.47
            # in the shipped calibration are millimetres, not metres.
            translation = translation * 0.001
    return RegistrationCalibration(
        _intrinsics(data.get("depth_intrinsics", {})),
        _intrinsics(data.get("rgb_intrinsics", {})),
        rotation,
        translation,
        source_path,
    )


def load_calibration(path: str) -> Optional[RegistrationCalibration]:
    calibration_path = Path(path)
    if yaml is None or not calibration_path.is_file():
        return None
    try:
        text = calibration_path.read_text(encoding="utf-8")
        if text.startswith("%YAML:1.0"):
            text = "\n".join(text.splitlines()[1:])
        data = yaml.safe_load(text) or {}
        calibration = calibration_from_dict(data, str(calibration_path.resolve()))
        return calibration if calibration.valid else None
    except (OSError, ValueError, TypeError):
        return None


class SoftwareRegistrationAligner:
    """Project valid depth pixels into the RGB camera with a nearest z-buffer."""

    def __init__(self, calibration: Optional[RegistrationCalibration]):
        self.calibration = calibration

    @property
    def valid(self) -> bool:
        return bool(self.calibration and self.calibration.valid)

    def align(
        self,
        depth_image: np.ndarray,
        rgb_size: Tuple[int, int],
        depth_unit_to_meter: float = 0.001,
    ) -> np.ndarray:
        rgb_width, rgb_height = map(int, rgb_size)
        if depth_image is None or depth_image.size == 0 or rgb_width <= 0 or rgb_height <= 0:
            return np.empty((0, 0), dtype=np.uint16)
        depth = np.asarray(depth_image, dtype=np.uint16)
        if not self.valid:
            return cv2.resize(depth, (rgb_width, rgb_height), interpolation=cv2.INTER_NEAREST)
        calibration = self.calibration
        flat_indices = np.flatnonzero(depth)
        if flat_indices.size == 0:
            return np.zeros((rgb_height, rgb_width), dtype=np.uint16)
        ys, xs = np.divmod(flat_indices, depth.shape[1])
        raw_depth = depth.reshape(-1)[flat_indices]
        z = raw_depth.astype(np.float64) * (
            depth_unit_to_meter if depth_unit_to_meter > 0 else 0.001
        )
        points = np.stack(
            [
                (xs - calibration.depth.cx) * z / calibration.depth.fx,
                (ys - calibration.depth.cy) * z / calibration.depth.fy,
                z,
            ],
            axis=0,
        )
        rgb_points = calibration.rotation @ points + calibration.translation_m[:, None]
        positive = rgb_points[2] > 0
        if not np.any(positive):
            return cv2.resize(depth, (rgb_width, rgb_height), interpolation=cv2.INTER_NEAREST)
        rgb_points = rgb_points[:, positive]
        raw_depth = raw_depth[positive]
        u = np.rint(rgb_points[0] * calibration.rgb.fx / rgb_points[2] + calibration.rgb.cx).astype(np.int64)
        v = np.rint(rgb_points[1] * calibration.rgb.fy / rgb_points[2] + calibration.rgb.cy).astype(np.int64)
        inside = (u >= 0) & (u < rgb_width) & (v >= 0) & (v < rgb_height)
        if not np.any(inside):
            return cv2.resize(depth, (rgb_width, rgb_height), interpolation=cv2.INTER_NEAREST)
        u, v = u[inside], v[inside]
        z_rgb = rgb_points[2, inside]
        raw_depth = raw_depth[inside]
        flat_index = v * rgb_width + u
        # Sort nearest first, then keep the first observation for each RGB pixel.
        order = np.argsort(z_rgb, kind="stable")
        sorted_index = flat_index[order]
        _unique, first = np.unique(sorted_index, return_index=True)
        selected = order[first]
        aligned = np.zeros(rgb_width * rgb_height, dtype=np.uint16)
        aligned[flat_index[selected]] = raw_depth[selected]
        return aligned.reshape(rgb_height, rgb_width)


__all__ = [
    "CameraIntrinsics",
    "RegistrationCalibration",
    "SoftwareRegistrationAligner",
    "calibration_from_dict",
    "load_calibration",
]
