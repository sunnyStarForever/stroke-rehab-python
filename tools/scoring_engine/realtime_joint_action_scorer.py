"""
realtime_joint_action_scorer.py

实时关节输入 -> 动作分割计数 -> 返回最后一个完整周期的五维评分。

输入支持两种格式：

1）推荐格式：dict
frame = {
    "waist": [x, y, z],
    "spine": [x, y, z],
    ...
    "r_toe": [x, y, z],
}

2）数组格式：
frame = np.ndarray, shape = (22, 3)

每次调用：
result = analyzer.update(frame)

输出：
- count: 当前识别到的动作中心数量，即实时计数
- completed_count: 已完成并评分的周期数量
- last_cycle: 如果本次识别到新完整周期，则返回该周期评分，否则为 None
"""

from collections import deque
import re
import sys
import numpy as np
import pandas as pd
from scipy.signal import savgol_filter, find_peaks

from action_segmentation_template_eval import detect_peaks_after_detrend, detrend_signal_poly, smooth_signal, fit_polynomial_baseline
from feature_extractor import *
from apis import normalize_action_type
from rom_quality_report import compute_rom_angles

from action_config import (
    normalize_action_type,
    get_realtime_action_config,
    get_primary_angle,
)

# =========================================================
# 3. 输入帧解析
# =========================================================
def frame_to_array(frame):
    """
    将单帧关节数据转换为 shape=(22, 3) 的 numpy 数组。

    支持三种输入：

    1. np.ndarray:
        shape = (22, 3)

    2. dict，关节名作为 key:
        {
            "waist": [x, y, z],
            "spine": [x, y, z],
            ...
        }

    3. dict，关节编号作为 key:
        {
            0: [x, y, z],
            1: [x, y, z],
            ...
        }
        或：
        {
            "0": [x, y, z],
            "1": [x, y, z],
            ...
        }
    """

    # -----------------------------------------------------
    # 情况 1：已经是 numpy 数组
    # -----------------------------------------------------
    if isinstance(frame, np.ndarray):
        arr = np.asarray(frame, dtype=float)

        if arr.shape != (22, 3):
            raise ValueError(
                f"数组输入必须是 shape=(22, 3)，当前为 {arr.shape}"
            )

        return arr

    # -----------------------------------------------------
    # 情况 2：dict 输入
    # -----------------------------------------------------
    if isinstance(frame, dict):
        arr = np.zeros((22, 3), dtype=float)
        missing = []

        # 兼容 JOINTS = [(0, "waist"), ...] 或 JOINTS = ["waist", ...]
        if len(JOINTS) > 0 and isinstance(JOINTS[0], tuple):
            joint_pairs = JOINTS
        else:
            joint_pairs = list(enumerate(JOINTS))

        for joint_id, joint_name in joint_pairs:
            value = None

            # 优先支持关节名 key，例如 "waist"
            if joint_name in frame:
                value = frame[joint_name]

            # 兼容整数编号 key，例如 0
            elif joint_id in frame:
                value = frame[joint_id]

            # 兼容字符串编号 key，例如 "0"
            elif str(joint_id) in frame:
                value = frame[str(joint_id)]

            else:
                missing.append(joint_name)
                continue

            value = np.asarray(value, dtype=float)

            if value.shape != (3,):
                raise ValueError(
                    f"关节 {joint_name} 的坐标必须是长度为 3 的序列，"
                    f"当前 shape={value.shape}"
                )

            arr[joint_id] = value

        if missing:
            raise ValueError(f"输入 frame 缺少关节: {missing}")

        return arr

    raise TypeError(
        "frame 必须是 dict 或 np.ndarray(shape=(22, 3))，"
        f"当前类型为 {type(frame)}"
    )



# =========================================================
# 7. 单周期质量评分
# =========================================================
def rms(x):
    x = np.asarray(x, dtype=float)

    if len(x) == 0:
        return np.nan

    return float(np.sqrt(np.nanmean(x ** 2)))


def compute_smoothness_metric(signal):
    signal = np.asarray(signal, dtype=float)

    if len(signal) < 5:
        return np.nan

    vel = np.diff(signal, n=1)
    jerk = np.diff(signal, n=3)

    return rms(jerk) / (rms(vel) + 1e-8)


def score_higher_better(value, history_values):
    history_values = np.asarray(history_values, dtype=float)
    history_values = history_values[np.isfinite(history_values)]

    if len(history_values) <= 1:
        return 100.0

    ref = np.nanpercentile(history_values, 90)

    if not np.isfinite(ref) or abs(ref) < 1e-8:
        return 100.0

    return float(np.clip(value / ref * 100.0, 0.0, 100.0))


def score_lower_better(value, history_values):
    history_values = np.asarray(history_values, dtype=float)
    history_values = history_values[np.isfinite(history_values)]

    if len(history_values) <= 1:
        return 100.0

    low_ref = np.nanpercentile(history_values, 10)
    high_ref = np.nanpercentile(history_values, 90)

    if not np.isfinite(low_ref) or not np.isfinite(high_ref):
        return 100.0

    if abs(high_ref - low_ref) < 1e-8:
        return 100.0

    return float(np.clip((high_ref - value) / (high_ref - low_ref) * 100.0, 0.0, 100.0))


def compute_cycle_raw_metrics(
    angles_df,
    segment,
    action_id,
    previous_durations=None,
):
    primary_angle = get_primary_angle(action_id)

    start = int(segment["start"])
    end = int(segment["end"])

    seg_angles = angles_df.iloc[start:end + 1].copy()

    if len(seg_angles) < 5:
        return None

    if primary_angle not in seg_angles.columns:
        primary_angle = "knee_flexion_mean"

    primary_signal = seg_angles[primary_angle].to_numpy(dtype=float)

    amplitude_metric = float(np.nanmax(primary_signal) - np.nanmin(primary_signal))
    smoothness_metric = compute_smoothness_metric(primary_signal)

    trunk_metric = float(
        np.nanstd(seg_angles["trunk_sagittal_lean"].to_numpy(dtype=float))
        + np.nanstd(seg_angles["trunk_frontal_lean"].to_numpy(dtype=float))
    )

    is_upper = action_id in ["M7", "M8", "M9", "M10"]

    if is_upper:
        symmetry_cols = [
            "shoulder_abduction_asymmetry",
            "shoulder_flexion_asymmetry",
            "elbow_flexion_asymmetry",
        ]
    else:
        symmetry_cols = [
            "knee_flexion_asymmetry",
            "hip_flexion_asymmetry",
        ]

    symmetry_values = []

    for col in symmetry_cols:
        if col in seg_angles.columns:
            symmetry_values.append(np.nanmean(seg_angles[col].to_numpy(dtype=float)))

    symmetry_metric = float(np.nanmean(symmetry_values)) if symmetry_values else np.nan

    duration_frames = int(end - start + 1)

    if previous_durations is None or len(previous_durations) == 0:
        rhythm_metric = 0.0
    else:
        median_duration = np.nanmedian(previous_durations)

        if median_duration <= 1e-8:
            rhythm_metric = 0.0
        else:
            rhythm_metric = abs(duration_frames - median_duration) / median_duration

    return {
        "primary_angle": primary_angle,
        "amplitude_metric_rom_deg": amplitude_metric,
        "smoothness_metric": smoothness_metric,
        "trunk_stability_metric": trunk_metric,
        "symmetry_metric_deg": symmetry_metric,
        "duration_frames": duration_frames,
        "rhythm_metric_duration_dev": rhythm_metric,
    }


def score_cycle_metrics(current_metrics, metric_history):
    """
    metric_history 是已完成周期的原始指标历史。
    这里会把当前周期也临时加入历史后做相对评分。
    """

    history_with_current = {}

    for key, values in metric_history.items():
        history_with_current[key] = list(values)

    for key, value in current_metrics.items():
        if isinstance(value, (int, float, np.floating)) and np.isfinite(value):
            history_with_current.setdefault(key, []).append(float(value))

    amplitude_score = score_higher_better(
        current_metrics["amplitude_metric_rom_deg"],
        history_with_current["amplitude_metric_rom_deg"],
    )

    smoothness_score = score_lower_better(
        current_metrics["smoothness_metric"],
        history_with_current["smoothness_metric"],
    )

    trunk_score = score_lower_better(
        current_metrics["trunk_stability_metric"],
        history_with_current["trunk_stability_metric"],
    )

    symmetry_score = score_lower_better(
        current_metrics["symmetry_metric_deg"],
        history_with_current["symmetry_metric_deg"],
    )

    rhythm_metric = current_metrics["rhythm_metric_duration_dev"]

    rhythm_score = float(np.clip((1.0 - rhythm_metric / 0.30) * 100.0, 0.0, 100.0))

    overall_score = (
        0.25 * amplitude_score
        + 0.20 * smoothness_score
        + 0.20 * trunk_score
        + 0.20 * symmetry_score
        + 0.15 * rhythm_score
    )

    return {
        "amplitude_score": float(amplitude_score),
        "smoothness_score": float(smoothness_score),
        "trunk_score": float(trunk_score),
        "symmetry_score": float(symmetry_score),
        "rhythm_score": float(rhythm_score),
        "overall_score": float(overall_score),
    }


# 数据类型确保
def ensure_feature_dataframe(features, n_frames=None):
    """
    将特征结果统一转换为 pd.DataFrame。

    支持：
        1. pd.DataFrame
        2. dict[str, np.ndarray]
        3. dict 中含 frame_idx 或不含 frame_idx

    返回：
        features_df: pd.DataFrame
    """

    if isinstance(features, pd.DataFrame):
        features_df = features.copy()

    elif isinstance(features, dict):
        if len(features) == 0:
            raise ValueError("features 是空 dict，无法转换为 DataFrame")

        clean_features = {}

        for name, value in features.items():
            arr = np.asarray(value)

            # 只接受一维特征曲线
            if arr.ndim == 0:
                continue

            if arr.ndim > 1:
                arr = arr.reshape(arr.shape[0], -1)

                if arr.shape[1] != 1:
                    raise ValueError(
                        f"特征 {name} 不是一维序列，shape={arr.shape}"
                    )

                arr = arr[:, 0]

            clean_features[name] = arr

        lengths = [len(v) for v in clean_features.values()]

        if len(set(lengths)) != 1:
            raise ValueError(
                f"features 中各特征长度不一致: "
                f"{dict((k, len(v)) for k, v in clean_features.items())}"
            )

        features_df = pd.DataFrame(clean_features)

    else:
        raise TypeError(
            f"features 必须是 pd.DataFrame 或 dict，当前类型为 {type(features)}"
        )

    if n_frames is not None and len(features_df) != n_frames:
        raise ValueError(
            f"features_df 长度与当前帧数不一致: "
            f"len(features_df)={len(features_df)}, n_frames={n_frames}"
        )

    if "frame_idx" not in features_df.columns:
        features_df.insert(0, "frame_idx", np.arange(len(features_df)))

    return features_df



# =========================================================
# 8. 实时 API 类
# =========================================================
class RealtimeJointActionScorer:
    """
    实时动作评分 API。

    用法：
        analyzer = RealtimeJointActionScorer(action_type="M1", fs=30)

        result = analyzer.update(frame)

        if result["status"] == "new_completed_cycle":
            print(result["last_cycle"], file=sys.stderr, flush=True)
    """

    def __init__(
        self,
        action_type,
        fs=10,
        min_interval_sec=2.0,
        min_frames_before_detection=None,
        peak_confirm_sec=0.0,
        poly_degree=3,
        smooth_win=5,
        smooth_poly=3,
        prominence_ratio=0.30,
        normalize=True,
        smooth_features=True,
        max_buffer_frames=None,
    ):
        
        self.fs = fs
        
        # TODO: 这里的配置项应该从配置文件中读取，第一版手动赋值 此处做修改
        # self.action_id = normalize_action_type(action_type)
        # cfg = ACTION_FEATURE_CONFIG[self.action_id]
        # self.segment_feature_name = cfg["segment_feature"]
        # self.count_divisor = int(cfg["count_divisor"])
        
        cfg = get_realtime_action_config(
            action_type=action_type,
            segment_feature_name=None,
        )

        self.action_id = cfg["action_id"]
        self.segment_feature_name = cfg["segment_feature"]
        self.count_divisor = int(cfg["count_divisor"])
        self.primary_angle = cfg["primary_angle"]
        self.feature_priority = cfg["priority"]

        self.min_distance = int(round(fs * min_interval_sec))
        self.peak_confirm_frames = max(0, int(round(fs * peak_confirm_sec)))

        warmup_sec = 2.0
        self.min_frames_before_detection = (
            min_frames_before_detection
            if min_frames_before_detection is not None
            else max(
                smooth_win + self.peak_confirm_frames + 2,
                int(round(warmup_sec * self.fs)),
            )
        )

        self.poly_degree = poly_degree
        self.smooth_win = smooth_win
        self.smooth_poly = smooth_poly
        self.prominence_ratio = prominence_ratio

        self.normalize = normalize
        self.smooth_features = smooth_features

        self.max_buffer_frames = max_buffer_frames

        # ── Movement onset detection ──
        self._onset_window_sec = 1.0
        self._onset_sustain_sec = 0.5
        self._onset_window_frames = max(10, int(round(fs * self._onset_window_sec)))
        self._onset_sustain_frames = max(5,  int(round(fs * self._onset_sustain_sec)))
        self._onset_energy_ratio = 3.0
        self._movement_started = False

        self.frames = []
        self.accepted_peaks = []
        self.accepted_centers = []
        self.completed_segments = []
        self.last_confirmed_limit = -1

        self.metric_history = {
            "amplitude_metric_rom_deg": [],
            "smoothness_metric": [],
            "trunk_stability_metric": [],
            "symmetry_metric_deg": [],
            "duration_frames": [],
            "rhythm_metric_duration_dev": [],
        }

        # ---- Debug buffer (non-pickled, updated each frame) ----
        self._debug_features = {}          # dict[str, np.ndarray] — latest compute_features result
        self._debug_segment_signal = None  # segment_feature raw signal after smooth
        self._debug_baseline = None        # detrend baseline
        self._debug_detrended = None       # detrended signal (raw)
        self._debug_detrended_smooth = None
        self._debug_peaks = []             # peaks from detect_peaks_after_detrend
        self._debug_prominence = 0.0

    def reset(self):
        self.frames = []
        self.accepted_peaks = []
        self.accepted_centers = []
        self.completed_segments = []
        self.last_confirmed_limit = -1
        self._movement_started = False

        for key in self.metric_history:
            self.metric_history[key] = []

        self._debug_features = {}
        self._debug_segment_signal = None
        self._debug_baseline = None
        self._debug_detrended = None
        self._debug_detrended_smooth = None
        self._debug_peaks = []
        self._debug_prominence = 0.0

    def get_sequence(self):
        if len(self.frames) == 0:
            return np.empty((0, 22, 3), dtype=float)

        return np.stack(self.frames, axis=0)

    def _accept_new_peaks(self, peaks, n_frames):
        """
        只接受已经过确认延迟的峰，避免刚到来的峰位置不稳定。
        """

        confirmed_limit = n_frames - self.peak_confirm_frames
        confirmed_limit = max(0, confirmed_limit)
        if n_frames < self.min_frames_before_detection:
            return False

        new_candidates = [
            int(p) for p in peaks
            if self.last_confirmed_limit < int(p) <= confirmed_limit
        ]

        self.last_confirmed_limit = max(
            self.last_confirmed_limit,
            confirmed_limit,
        )

        if len(new_candidates) == 0:
            return False

        if len(self.accepted_peaks) == 0 and len(new_candidates) > 1:
            new_candidates = [new_candidates[-1]]

        changed = False
        min_gap = max(1, self.min_distance // 2)

        for p in new_candidates:
            if len(self.accepted_peaks) == 0:
                self.accepted_peaks.append(p)
                changed = True
            else:
                last_p = self.accepted_peaks[-1]

                if p > last_p + min_gap:
                    self.accepted_peaks.append(p)
                    changed = True

        return changed

    def _update_centers_from_peaks(self):
        """
        Bind action centers directly to accepted peaks.

        The realtime counter now uses one detected peak as one repetition.
        Therefore:

            detected_peaks == accepted_peaks == accepted_centers

        ``count_divisor`` is intentionally ignored in the realtime path. It is
        kept in the configuration only for compatibility with older/offline
        tooling that may still expose the parameter.
        """

        old_len = len(self.accepted_centers)
        self.accepted_centers = [int(p) for p in self.accepted_peaks]

        # A completed segment needs the following center as its right boundary.
        # If realtime peak detection drops centers, discard impossible/stale
        # completed segments so completed_count never belongs to an older peak
        # layout.
        max_completed = max(0, len(self.accepted_centers) - 1)
        if len(self.completed_segments) > max_completed:
            del self.completed_segments[max_completed:]

        return len(self.accepted_centers) > old_len

    def _detect_movement_onset(self, signal):
        """Find the frame index where sustained movement begins.

        Returns onset_frame (int) or 0 if movement already active or
        insufficient data.  The returned index is relative to the
        *current* signal, i.e. frames[onset_frame:] should be kept.
        """
        n = len(signal)
        if n < self._onset_window_frames + self._onset_sustain_frames:
            return 0

        # ── Rolling energy (std over window) ──
        import pandas as pd
        energy = pd.Series(signal).rolling(
            window=self._onset_window_frames, center=False, min_periods=5
        ).std().fillna(0.0).to_numpy(dtype=float)

        # Baseline energy = median of the first window frames
        first = energy[:self._onset_window_frames]
        baseline = float(np.nanmedian(first[first > 0])) if np.any(first > 0) else \
                   float(np.nanstd(signal[:self._onset_window_frames]))
        if baseline < 1e-6:
            baseline = 1.0

        threshold = baseline * self._onset_energy_ratio

        # Find the first stretch of `_onset_sustain_frames` consecutive
        # frames where energy exceeds the threshold.
        run = 0; onset_candidate = -1
        for i, val in enumerate(energy):
            if np.isfinite(val) and val >= threshold:
                run += 1
                if run == 1:
                    onset_candidate = i
                if run >= self._onset_sustain_frames:
                    # onset_candidate marks the beginning of the active run;
                    # trim from one window before that to keep a smooth edge.
                    return max(0, onset_candidate - self._onset_window_frames // 2)
            else:
                run = 0; onset_candidate = -1
        return 0

    def _trim_to_onset(self, onset: int):
        """Discard frames before `onset` and reset peak-detection state."""
        self.frames = self.frames[onset:]
        self.accepted_peaks.clear()
        self.accepted_centers.clear()
        self.completed_segments.clear()
        self.last_confirmed_limit = -1
        for key in self.metric_history:
            self.metric_history[key] = []
        self._movement_started = True

    def _build_new_completed_segment(self):
        """
        当出现新的 center 后，前一个 center 对应的周期才算完整。

        centers:
            c0, c1, c2 ...

        当 c1 出现时，可以完成 c0 对应周期：
            start = 0
            end = midpoint(c0, c1) - 1

        当 c2 出现时，可以完成 c1 对应周期：
            start = midpoint(c0, c1)
            end = midpoint(c1, c2) - 1
        """

        centers = self.accepted_centers

        if len(centers) < 2:
            return None

        completed_rep_id = len(self.completed_segments) + 1
        center_index = completed_rep_id - 1

        if center_index + 1 >= len(centers):
            return None

        c = centers[center_index]

        if center_index == 0:
            start = 0
        else:
            start = int((centers[center_index - 1] + centers[center_index]) / 2)

        end = int((centers[center_index] + centers[center_index + 1]) / 2) - 1

        if end <= start:
            return None

        segment = {
            "rep_id": completed_rep_id,
            "start": int(start),
            "end": int(end),
            "center": int(c),
        }

        # 避免重复添加
        if len(self.completed_segments) >= completed_rep_id:
            return None

        self.completed_segments.append(segment)

        return segment

    def _score_completed_segment(self, segment, seq, features_df, angles_df):
        previous_durations = self.metric_history["duration_frames"]

        raw_metrics = compute_cycle_raw_metrics(
            angles_df=angles_df,
            segment=segment,
            action_id=self.action_id,
            previous_durations=previous_durations,
        )

        if raw_metrics is None:
            return None

        scores = score_cycle_metrics(
            current_metrics=raw_metrics,
            metric_history=self.metric_history,
        )

        for key in self.metric_history:
            if key in raw_metrics and np.isfinite(raw_metrics[key]):
                self.metric_history[key].append(float(raw_metrics[key]))

        output = {
            "rep_id": int(segment["rep_id"]),
            "start": int(segment["start"]),
            "end": int(segment["end"]),
            "center": int(segment["center"]),
            "duration_frames": int(raw_metrics["duration_frames"]),

            "overall_score": scores["overall_score"],

            "dimension_scores": {
                "amplitude_score": scores["amplitude_score"],
                "smoothness_score": scores["smoothness_score"],
                "trunk_score": scores["trunk_score"],
                "symmetry_score": scores["symmetry_score"],
                "rhythm_score": scores["rhythm_score"],
            },

            "raw_metrics": raw_metrics,
        }

        return output

    def update(self, frame):
        """
        输入一帧关节数据。

        返回：
            如果没有新完整周期：
                status = "waiting" 或 "no_new_cycle"
                last_cycle = None

            如果有新完整周期：
                status = "new_completed_cycle"
                last_cycle = 最近一个完整周期评分
        """

        arr = frame_to_array(frame)
        self.frames.append(arr)

        if self.max_buffer_frames is not None and len(self.frames) > self.max_buffer_frames:
            raise NotImplementedError(
                "当前版本为累计式实时分析。若启用 max_buffer_frames，需要同步处理索引偏移。"
            )

        n_frames = len(self.frames)

        base_result = {
            "status": "waiting" if n_frames < self.min_frames_before_detection else "no_new_cycle",
            "action_id": self.action_id,
            "frame_index": n_frames - 1,
            "count": int(len(self.accepted_centers)),
            "completed_count": int(len(self.completed_segments)),
            "last_cycle": None,
        }

        if n_frames < self.min_frames_before_detection:
            return base_result

        seq = self.get_sequence()

        # -----------------------------------------------------
        # 1. 计算特征，并统一转换成 DataFrame
        # -----------------------------------------------------
        features = compute_features_from_sequence(
            seq,
            normalize=self.normalize,
            smooth=self.smooth_features,
            smooth_win=self.smooth_win,
            smooth_poly=self.smooth_poly,
        )

        features_df = ensure_feature_dataframe(
            features,
            n_frames=n_frames,
        )

        # ── Movement-onset detection (trim flat baseline before first peak search) ──
        if not self._movement_started and self.segment_feature_name in features_df.columns:
            signal_raw = features_df[self.segment_feature_name].to_numpy(dtype=float)
            onset = self._detect_movement_onset(signal_raw)
            if onset > 0:
                self._trim_to_onset(onset)
                n_frames = len(self.frames)
                if n_frames < self.min_frames_before_detection:
                    base_result["frame_index"] = n_frames - 1
                    return base_result
                # Recompute features from trimmed frames
                seq = self.get_sequence()
                features = compute_features_from_sequence(
                    seq, normalize=self.normalize, smooth=self.smooth_features,
                    smooth_win=self.smooth_win, smooth_poly=self.smooth_poly,
                )
                features_df = ensure_feature_dataframe(features, n_frames=n_frames)
                base_result["frame_index"] = n_frames - 1

        if self.segment_feature_name not in features_df.columns:
            raise ValueError(
                f"分割特征不存在: {self.segment_feature_name}\n"
                f"当前可用特征包括: {list(features_df.columns)}"
            )

        signal = features_df[self.segment_feature_name].to_numpy(dtype=float)

        # -----------------------------------------------------
        # 2. 去趋势寻峰
        # -----------------------------------------------------
        peaks, debug = detect_peaks_after_detrend(
            signal,
            min_distance=self.min_distance,
            poly_degree=self.poly_degree,
            smooth_win=self.smooth_win,
            smooth_poly=self.smooth_poly,
            prominence_ratio=self.prominence_ratio,
        )

        # ---- Save debug data (kept within buffer window for display) ----
        self._debug_features = features
        self._debug_segment_signal = signal
        self._debug_baseline = debug.get("baseline")
        self._debug_detrended = debug.get("detrended")
        self._debug_detrended_smooth = debug.get("detrended_smooth")
        self._debug_peaks = list(peaks) if len(peaks) else []
        self._debug_prominence = float(debug.get("prominence", 0.0))

        # -----------------------------------------------------
        # 3. Strict peak-based counting: every detected peak is accepted and
        #    every accepted peak is also the action center.
        # -----------------------------------------------------
        old_peak_count = len(self.accepted_peaks)
        self.accepted_peaks = [int(p) for p in peaks]
        new_peak_added = len(self.accepted_peaks) > old_peak_count

        new_center_added = self._update_centers_from_peaks()

        base_result["count"] = int(len(self.accepted_centers))
        base_result["completed_count"] = int(len(self.completed_segments))
        base_result["debug"] = {
            "segment_feature_name": self.segment_feature_name,
            "all_detected_peaks": peaks,
            "accepted_peaks": np.asarray(self.accepted_peaks, dtype=int),
            "accepted_centers": np.asarray(self.accepted_centers, dtype=int),
        }

        if not new_peak_added and not new_center_added:
            return base_result

        # -----------------------------------------------------
        # 4. 如果可以确认一个完整周期，则构建周期
        # -----------------------------------------------------
        new_segment = self._build_new_completed_segment()

        if new_segment is None:
            base_result["status"] = "no_new_completed_cycle"
            return base_result

        # -----------------------------------------------------
        # 5. 计算 ROM 角度，并对最后完整周期评分
        # -----------------------------------------------------
        angles_df = compute_rom_angles(seq)

        last_cycle = self._score_completed_segment(
            segment=new_segment,
            seq=seq,
            features_df=features_df,
            angles_df=angles_df,
        )

        if last_cycle is None:
            base_result["status"] = "no_new_completed_cycle"
            return base_result

        base_result["status"] = "new_completed_cycle"
        base_result["count"] = int(len(self.accepted_centers))
        base_result["completed_count"] = int(len(self.completed_segments))
        base_result["last_cycle"] = last_cycle

        return base_result

    def update_many(self, frames):
        """
        批量输入多帧，模拟实时调用。

        返回所有触发了新完整周期的结果。
        """

        outputs = []

        for frame in frames:
            result = self.update(frame)

            if result["status"] == "new_completed_cycle":
                outputs.append(result)

        return outputs

    def get_debug_state(self):
        """
        返回当前调试状态，包含特征波形、去趋势寻峰过程及帧数信息。

        返回内容（所有 numpy 数组均已转为列表以便 JSON 序列化）：
        - n_frames: int
        - action_id: str
        - segment_feature_name: str
        - segment_signal: list[float] — 分割特征原始信号
        - baseline: list[float] — 多项式基线
        - detrended: list[float] — 去趋势后的信号
        - detrended_smooth: list[float] — 平滑后的去趋势信号
        - detected_peaks: list[int] — 最近一次寻峰的位置
        - accepted_peaks: list[int] — accepted peaks; realtime mode keeps this
          identical to detected_peaks
        - accepted_centers: list[int] — action centers; realtime mode binds one
          center to each accepted peak
        - count: int — 当前中心计数
        - completed_count: int — 已完成周期数
        - prominence: float — 实际使用的 prominence 阈值
        - metric_history: dict — 各指标历史
        - params: dict — 当前参数快照
        - all_features: dict[str, list[float]] — 所有特征曲线（最近缓冲窗口内）
        """
        import numpy as np
        features = {}
        for name, arr in self._debug_features.items():
            if isinstance(arr, np.ndarray):
                features[name] = arr.tolist()
            else:
                features[name] = arr

        def safe_list(arr):
            if arr is None:
                return []
            if isinstance(arr, np.ndarray):
                return arr.tolist()
            return list(arr)

        return {
            "action_id": self.action_id,
            "n_frames": len(self.frames),
            "count": int(len(self.accepted_centers)),
            "completed_count": int(len(self.completed_segments)),
            "segment_feature_name": self.segment_feature_name,
            "segment_signal": safe_list(self._debug_segment_signal),
            "baseline": safe_list(self._debug_baseline),
            "detrended": safe_list(self._debug_detrended),
            "detrended_smooth": safe_list(self._debug_detrended_smooth),
            "detected_peaks": list(self._debug_peaks) if self._debug_peaks else [],
            "accepted_peaks": safe_list(np.asarray(self.accepted_peaks, dtype=int)),
            "accepted_centers": safe_list(np.asarray(self.accepted_centers, dtype=int)),
            "prominence": float(self._debug_prominence),
            "metric_history": {
                k: safe_list(np.asarray(v, dtype=float))
                for k, v in self.metric_history.items()
            },
            "params": {
                "action_id": self.action_id,
                "fs": self.fs,
                "count_divisor": self.count_divisor,
                "min_distance": self.min_distance,
                "peak_confirm_frames": self.peak_confirm_frames,
                "poly_degree": self.poly_degree,
                "smooth_win": self.smooth_win,
                "smooth_poly": self.smooth_poly,
                "prominence_ratio": self.prominence_ratio,
                "min_frames_before_detection": self.min_frames_before_detection,
                "strict_peak_count": True,
                "onset_window_sec": self._onset_window_sec,
                "onset_sustain_sec": self._onset_sustain_sec,
                "onset_energy_ratio": self._onset_energy_ratio,
                "movement_started": self._movement_started,
            },
            "all_features": features,
        }

