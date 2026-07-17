"""Python Rehab22 depth sampling, projection and temporal filtering.

This module ports the algorithms from core/pose so the native extension is only
needed for physical camera access, not for 3D skeleton ownership.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import Enum
import math
from typing import Any, List, Optional, Sequence, Tuple

import numpy as np


REHAB22_NAMES = (
    "waist", "spine", "chest", "neck", "head", "head_tip",
    "left_collar", "left_shoulder", "left_elbow", "left_wrist",
    "right_collar", "right_shoulder", "right_elbow", "right_wrist",
    "left_hip", "left_knee", "left_ankle", "left_toe",
    "right_hip", "right_knee", "right_ankle", "right_toe",
)
PARENT_INDEX = {15: 14, 16: 15, 17: 16, 9: 8, 19: 18, 20: 19, 21: 20, 13: 12}
TORSO_INDICES = (0, 1, 2, 3, 14, 18)


@dataclass
class Joint2D:
    name: str = ""
    x: float = 0.0
    y: float = 0.0
    score: float = 0.0
    raw_score: float = 0.0
    valid: bool = False


@dataclass
class Joint3D:
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0
    score: float = 0.0
    valid: bool = False
    u: float = 0.0
    v: float = 0.0
    raw_pose_score: float = 0.0
    sampled_depth_mm: int = 0
    sample_method: str = ""
    sample_reason: str = ""
    rejected_as_background: bool = False
    edge_ambiguous: bool = False
    foreground_recovered: bool = False


@dataclass
class EmaDebugInfo:
    alpha: float = 0.0
    reason: str = ""
    invalid_hold_count: int = 0


class DepthSampleMethod(str, Enum):
    FOREGROUND_WINDOW = "ForegroundWindow"
    FOREGROUND_PERCENTILE = "ForegroundPercentile"
    LIMB_INWARD_SEARCH = "LimbInwardSearch"
    FAILED_OUT_OF_BOUNDS = "FailedOutOfBounds"
    FAILED_NO_VALID_DEPTH = "FailedNoValidDepth"
    FAILED_BACKGROUND_HIT = "FailedBackgroundHit"


@dataclass
class DepthSampleContext:
    body_depth_ref_mm: float = 0.0
    background_depth_ref_mm: float = 0.0
    has_body_depth_ref: bool = False
    has_background_depth_ref: bool = False


@dataclass
class DepthSampleResult:
    valid: bool = False
    depth_raw_mm: int = 0
    depth_meters: float = 0.0
    u: int = 0
    v: int = 0
    used_radius: int = 0
    valid_pixel_count: int = 0
    foreground_pixel_count: int = 0
    rejected_background_count: int = 0
    body_depth_ref_mm: float = 0.0
    background_depth_ref_mm: float = 0.0
    depth_before_reject_mm: float = 0.0
    rejected_as_background: bool = False
    edge_ambiguous: bool = False
    foreground_recovered: bool = False
    method: DepthSampleMethod = DepthSampleMethod.FAILED_NO_VALID_DEPTH
    reason: str = ""


def _upper_median(values: Sequence[int]) -> int:
    array = np.asarray(values, dtype=np.uint16)
    index = len(array) // 2
    return int(np.partition(array, index)[index])


class DepthSampler:
    def __init__(self, options: Any) -> None:
        self.options = options

    def _raw_to_mm(self, raw: int, unit_to_meter: float) -> int:
        if raw <= 0 or unit_to_meter <= 0:
            return 0
        return int(math.floor(raw * unit_to_meter * 1000.0 + 0.5))

    def sample_single(
        self, depth: np.ndarray, x: float, y: float, unit_to_meter: float,
        window_size: int = 5,
    ) -> Tuple[float, int]:
        if not self._valid_depth(depth) or unit_to_meter <= 0:
            return 0.0, 0
        u, v = int(math.floor(x + 0.5)), int(math.floor(y + 0.5))
        height, width = depth.shape
        if not (0 <= u < width and 0 <= v < height):
            return 0.0, 0
        window = max(3, int(window_size))
        window += int(window % 2 == 0)
        radius = window // 2
        patch = depth[max(0, v-radius):min(height, v+radius+1),
                      max(0, u-radius):min(width, u+radius+1)]
        values = np.rint(patch.astype(np.float64) * unit_to_meter * 1000.0).astype(np.int64)
        valid = values[(values >= self.options.min_depth_mm) &
                       (values <= self.options.max_depth_mm)]
        if not valid.size:
            return 0.0, 0
        return _upper_median(valid.tolist()) / 1000.0, int(valid.size)

    def estimate_context(
        self, depth: np.ndarray, joints: Sequence[Joint2D], unit_to_meter: float,
        person_roi: Optional[Tuple[int, int, int, int]] = None,
    ) -> DepthSampleContext:
        context = DepthSampleContext()
        if not self._valid_depth(depth):
            return context
        torso = []
        for index in TORSO_INDICES:
            if index >= len(joints) or not joints[index].valid:
                continue
            value, count = self.sample_single(
                depth, joints[index].x, joints[index].y, unit_to_meter, 5)
            if value > 0 and count > 0:
                torso.append(int(math.floor(value * 1000.0 + 0.5)))
        if len(torso) >= 3:
            context.body_depth_ref_mm = float(_upper_median(torso))
            context.has_body_depth_ref = True
        background = self._estimate_background(depth, unit_to_meter, person_roi)
        if background > 0:
            context.background_depth_ref_mm = background
            context.has_background_depth_ref = True
        return context

    def sample_skeleton(
        self, depth: np.ndarray, joints: Sequence[Joint2D], unit_to_meter: float,
        person_roi: Optional[Tuple[int, int, int, int]] = None,
    ) -> List[DepthSampleResult]:
        context = self.estimate_context(depth, joints, unit_to_meter, person_roi)
        output = []
        for index, joint in enumerate(joints[:22]):
            parent = joints[PARENT_INDEX[index]] if index in PARENT_INDEX else None
            output.append(self.sample_joint(depth, joint, parent, context, unit_to_meter))
        output.extend(DepthSampleResult() for _ in range(22 - len(output)))
        return output

    def sample_joint(
        self, depth: np.ndarray, joint: Joint2D, parent: Optional[Joint2D],
        context: DepthSampleContext, unit_to_meter: float,
    ) -> DepthSampleResult:
        if not joint.valid:
            return DepthSampleResult(
                u=int(math.floor(joint.x + 0.5)), v=int(math.floor(joint.y + 0.5)),
                reason="joint_invalid")
        result = self._sample_at(depth, joint.x, joint.y, joint.name, context, unit_to_meter)
        needs_inward = any(token in joint.name for token in ("knee", "ankle", "toe", "wrist", "hand"))
        if result.valid or not needs_inward or parent is None or not parent.valid \
                or not self.options.limb_inward_search_enabled:
            return result
        dx, dy = parent.x - joint.x, parent.y - joint.y
        length = math.hypot(dx, dy)
        if length <= 1.0:
            result.reason = "limb_inward_no_parent_direction"
            return result
        for step in range(1, self.options.limb_inward_steps + 1):
            distance = step * self.options.limb_inward_step_px
            candidate = self._sample_at(
                depth, joint.x + dx / length * distance,
                joint.y + dy / length * distance, joint.name, context,
                unit_to_meter, self.options.limb_inward_radius)
            if candidate.valid:
                candidate.method = DepthSampleMethod.LIMB_INWARD_SEARCH
                candidate.foreground_recovered = True
                candidate.reason = "limb_inward_recovered"
                return candidate
        result.reason = "limb_inward_failed"
        return result

    def _sample_at(
        self, depth: np.ndarray, x: float, y: float, name: str,
        context: DepthSampleContext, unit_to_meter: float,
        radius_override: Optional[int] = None,
    ) -> DepthSampleResult:
        u, v = int(math.floor(x + 0.5)), int(math.floor(y + 0.5))
        result = DepthSampleResult(
            u=u, v=v, body_depth_ref_mm=context.body_depth_ref_mm,
            background_depth_ref_mm=context.background_depth_ref_mm)
        if not self._valid_depth(depth) or unit_to_meter <= 0:
            result.reason = "depth_image_invalid"
            return result
        height, width = depth.shape
        if not (0 <= u < width and 0 <= v < height):
            result.method = DepthSampleMethod.FAILED_OUT_OF_BOUNDS
            result.reason = "joint_out_of_bounds"
            return result
        radius = self._radius(name) if radius_override is None else radius_override
        result.used_radius = radius
        result.depth_before_reject_mm = self._raw_to_mm(int(depth[v, u]), unit_to_meter)
        patch = depth[max(0, v-radius):min(height, v+radius+1),
                      max(0, u-radius):min(width, u+radius+1)]
        mm = np.rint(patch.astype(np.float64) * unit_to_meter * 1000.0).astype(np.int64)
        valid = (mm >= self.options.min_depth_mm) & (mm <= self.options.max_depth_mm)
        result.valid_pixel_count = int(np.count_nonzero(valid))
        keep = valid.copy()
        if context.has_background_depth_ref and context.has_body_depth_ref:
            background = (
                np.abs(mm - context.background_depth_ref_mm) < self.options.background_match_band_mm
            ) & (mm > context.body_depth_ref_mm + self.options.background_reject_margin_mm)
            result.rejected_background_count = int(np.count_nonzero(valid & background))
            result.rejected_as_background = result.rejected_background_count > 0
            keep &= ~background
        if context.has_body_depth_ref:
            band = self.options.edge_body_depth_band_mm if self._is_edge(name) \
                else self.options.body_depth_band_mm
            keep &= np.abs(mm - context.body_depth_ref_mm) <= band
        foreground = mm[keep]
        result.foreground_pixel_count = int(foreground.size)
        if foreground.size < self.options.min_foreground_pixels:
            result.edge_ambiguous = self._is_edge(name)
            result.method = (DepthSampleMethod.FAILED_BACKGROUND_HIT
                             if result.rejected_background_count else
                             DepthSampleMethod.FAILED_NO_VALID_DEPTH)
            result.reason = ("background_rejected" if result.rejected_background_count
                             else "not_enough_foreground_pixels")
            return result
        if self._uses_percentile(name):
            values = np.sort(foreground)
            position = min(len(values) - 1, int(math.floor(
                min(1.0, max(0.0, self.options.foreground_percentile)) *
                (len(values) - 1) + 0.5)))
            value = int(values[position])
            result.method = DepthSampleMethod.FOREGROUND_PERCENTILE
        else:
            value = _upper_median(foreground.tolist())
            result.method = DepthSampleMethod.FOREGROUND_WINDOW
        result.valid = value > 0
        result.depth_raw_mm = value
        result.depth_meters = value / 1000.0
        result.reason = "foreground_window"
        return result

    def _estimate_background(
        self, depth: np.ndarray, unit_to_meter: float,
        roi: Optional[Tuple[int, int, int, int]],
    ) -> float:
        height, width = depth.shape
        values = []
        rx, ry, rw, rh = roi or (0, 0, 0, 0)
        for y in range(0, height, 16):
            for x in range(0, width, 16):
                if rw > 0 and rh > 0:
                    if rx <= x < rx + rw and ry <= y < ry + rh:
                        continue
                elif not (x < width // 6 or x > width * 5 // 6 or y < height // 5):
                    continue
                value = self._raw_to_mm(int(depth[y, x]), unit_to_meter)
                if self.options.min_depth_mm <= value <= self.options.max_depth_mm:
                    values.append(value)
        return float(_upper_median(values)) if len(values) >= 10 else 0.0

    def _radius(self, name: str) -> int:
        if name in ("waist", "chest", "spine", "neck") or "hip" in name:
            return self.options.hip_radius
        if "knee" in name:
            return self.options.knee_radius
        if "ankle" in name:
            return self.options.ankle_radius
        if "toe" in name:
            return self.options.toe_radius
        if "wrist" in name or "hand" in name:
            return self.options.wrist_radius
        return self.options.default_radius

    @staticmethod
    def _is_edge(name: str) -> bool:
        return any(token in name for token in ("knee", "ankle", "toe", "wrist", "hand"))

    @staticmethod
    def _uses_percentile(name: str) -> bool:
        return any(token in name for token in ("ankle", "toe", "wrist", "hand"))

    @staticmethod
    def _valid_depth(depth: np.ndarray) -> bool:
        return isinstance(depth, np.ndarray) and depth.ndim == 2 and depth.dtype == np.uint16


class JointProjector3D:
    def __init__(self) -> None:
        self.fx = self.fy = self.cx = self.cy = 0.0

    def set_intrinsics(self, fx: float, fy: float, cx: float, cy: float) -> None:
        self.fx, self.fy, self.cx, self.cy = map(float, (fx, fy, cx, cy))

    def intrinsics_valid(self) -> bool:
        return self.fx > 0 and self.fy > 0 and self.cx >= 0 and self.cy >= 0

    def project(
        self, joints: Sequence[Joint2D], samples: Sequence[DepthSampleResult]
    ) -> List[Joint3D]:
        output = []
        valid_intrinsics = self.intrinsics_valid()
        for joint, sample in zip(joints[:22], samples[:22]):
            point = Joint3D(
                score=joint.score, u=joint.x, v=joint.y,
                raw_pose_score=joint.raw_score,
                sampled_depth_mm=sample.depth_raw_mm,
                sample_method=sample.method.value,
                sample_reason=sample.reason,
                rejected_as_background=sample.rejected_as_background,
                edge_ambiguous=sample.edge_ambiguous,
                foreground_recovered=sample.foreground_recovered,
            )
            point.valid = joint.valid and sample.valid and sample.depth_meters > 0 and valid_intrinsics
            if point.valid:
                point.x = (joint.x - self.cx) * sample.depth_meters / self.fx
                point.y = (joint.y - self.cy) * sample.depth_meters / self.fy
                point.z = sample.depth_meters
            output.append(point)
        output.extend(Joint3D() for _ in range(22 - len(output)))
        return output


class EmaSkeletonFilter:
    def __init__(self, options: Any) -> None:
        self.options = options
        self.reset()

    def reset(self) -> None:
        self.last = [Joint3D() for _ in range(22)]
        self.has_last = False
        self.holds = [0] * 22
        self.last_debug = [EmaDebugInfo(reason="reset") for _ in range(22)]

    def filter(self, raw: Sequence[Joint3D], dt_seconds: float) -> List[Joint3D]:
        if not self.has_last:
            output = [replace(point) for point in raw[:22]]
            output.extend(Joint3D() for _ in range(22 - len(output)))
            self.last, self.has_last = output, True
            self.last_debug = [EmaDebugInfo(1.0, "init", 0) for _ in range(22)]
            return [replace(point) for point in output]
        safe_dt = dt_seconds if dt_seconds > 0 else 1.0 / 30.0
        output = []
        debug = []
        for index, current in enumerate(raw[:22]):
            previous = self.last[index]
            if not current.valid:
                if (self.options.hold_last_when_invalid and previous.valid and
                        self.holds[index] < self.options.max_hold_frames):
                    self.holds[index] += 1
                    output.append(replace(previous, valid=True))
                    debug.append(EmaDebugInfo(0.0, "invalid_hold", self.holds[index]))
                else:
                    output.append(replace(current, valid=False))
                    debug.append(EmaDebugInfo(0.0, "invalid", self.holds[index]))
                continue
            self.holds[index] = 0
            alpha = self.options.alpha_good
            reason = "good"
            if current.score < 0.30:
                alpha = self.options.alpha_low_confidence
                reason = "low_confidence"
            if current.foreground_recovered or current.sample_method == "LimbInwardSearch":
                alpha = self.options.alpha_recovered
                reason = "foreground_recovered"
            if previous.valid:
                distance = math.sqrt(
                    (current.x-previous.x)**2 + (current.y-previous.y)**2 +
                    (current.z-previous.z)**2)
                if abs(current.z - previous.z) > self.options.max_z_jump_m \
                        or distance / safe_dt > self.options.max_joint_speed_mps:
                    alpha = min(alpha, 0.20)
                    reason = "jump_limited"
                point = replace(current)
                point.x = alpha * current.x + (1-alpha) * previous.x
                point.y = alpha * current.y + (1-alpha) * previous.y
                point.z = alpha * current.z + (1-alpha) * previous.z
                point.score = alpha * current.score + (1-alpha) * previous.score
                point.valid = True
                output.append(point)
            else:
                output.append(replace(current))
            debug.append(EmaDebugInfo(float(alpha), reason, self.holds[index]))
        output.extend(Joint3D() for _ in range(22 - len(output)))
        debug.extend(EmaDebugInfo(reason="missing") for _ in range(22 - len(debug)))
        self.last = [replace(point) for point in output]
        self.last_debug = debug
        return output


@dataclass
class SmoothingStats:
    valid_input: int = 0
    valid_output: int = 0
    jump_rejected: int = 0


class SkeletonSmoother:
    def __init__(self, alpha: float = 0.35) -> None:
        self.alpha = min(1.0, max(0.0, float(alpha)))
        self.reset()

    def reset(self) -> None:
        self.has_state = [False] * 22
        self.last_timestamp_ns = [0] * 22
        self.invalid_streak = [0] * 22
        self.state = [Joint3D() for _ in range(22)]

    def smooth(
        self, joints: Sequence[Joint3D], timestamp_ns: int = 0
    ) -> Tuple[List[Joint3D], SmoothingStats]:
        output = [Joint3D() for _ in range(22)]
        stats = SmoothingStats()
        for index, point in enumerate(joints[:22]):
            if not point.valid:
                self.invalid_streak[index] += 1
                if self.invalid_streak[index] > 10:
                    self.has_state[index] = False
                    self.state[index] = Joint3D()
                    self.last_timestamp_ns[index] = 0
                continue
            stats.valid_input += 1
            self.invalid_streak[index] = 0
            dt = ((timestamp_ns - self.last_timestamp_ns[index]) / 1e9
                  if timestamp_ns > self.last_timestamp_ns[index] > 0 else 0.0)
            if (self.has_state[index] and 0 < dt < 0.2 and
                    abs(point.z - self.state[index].z) > 0.35):
                stats.jump_rejected += 1
                self.invalid_streak[index] = 1
                continue
            if not self.has_state[index]:
                self.state[index] = replace(point)
                self.has_state[index] = True
            else:
                state = self.state[index]
                state.x = self.alpha * point.x + (1-self.alpha) * state.x
                state.y = self.alpha * point.y + (1-self.alpha) * state.y
                state.z = self.alpha * point.z + (1-self.alpha) * state.z
                state.score = self.alpha * point.score + (1-self.alpha) * state.score
                state.valid = True
            self.last_timestamp_ns[index] = timestamp_ns
            output[index] = replace(self.state[index])
            stats.valid_output += 1
        return output, stats


def make_rehab22_joints(points: Sequence[Any], min_score: float) -> List[Joint2D]:
    output = []
    for index in range(22):
        point = points[index] if index < len(points) else None
        if point is None:
            x = y = score = 0.0
            source_valid = False
        elif hasattr(point, "x"):
            x, y = float(point.x), float(point.y)
            score = float(getattr(point, "score", 0.0))
            source_valid = bool(getattr(point, "valid", False))
        else:
            values = list(point)
            x = float(values[0]) if len(values) > 0 else 0.0
            y = float(values[1]) if len(values) > 1 else 0.0
            score = float(values[2]) if len(values) > 2 else 0.0
            source_valid = bool(values[3]) if len(values) > 3 else False
        output.append(Joint2D(
            name=REHAB22_NAMES[index], x=x, y=y, score=min(1.0, max(0.0, score)),
            raw_score=score, valid=source_valid and score >= min_score))
    return output


__all__ = [
    "DepthSampleContext", "DepthSampleMethod", "DepthSampleResult", "DepthSampler",
    "EmaSkeletonFilter", "Joint2D", "Joint3D", "JointProjector3D",
    "REHAB22_NAMES", "SkeletonSmoother", "SmoothingStats", "make_rehab22_joints",
]
