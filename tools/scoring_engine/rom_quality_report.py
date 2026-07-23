"""
rom_quality_report.py

功能：
1. 读取宽表骨架 CSV；
2. 计算关键关节 ROM；
3. 基于分割结果生成每一轮动作的 ROM 报告；
4. 基于特征和角度计算多维质量评分；
5. 评估关节偏移、身体前后倾斜、躯干稳定性；
6. 绘制 ROM 曲线、身体倾斜曲线和空间包络图。

典型输入：
    csv_path
    segments_df 或 analyze_action_csv() 返回结果中的 result["segments_df"]

典型输出：
    rom_detail_df
    rom_summary_df
    quality_df
    quality_summary
    offset_detail_df
    offset_summary_df
"""

import os
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
try:
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
except Exception as e:
    Axes3D = None


from feature_extractor import compute_features_from_csv, read_skeleton_csv
from feature_extractor import J, JOINTS, LAT, UP, DEP, BONES
from action_segmentation_template_eval import ACTION_FEATURE_CONFIG
from apis import analyze_action_csv


# =========================================================
# 1. 关节定义
# =========================================================
JOINT_NAME_TO_ID = {name: idx for idx, name in JOINTS}


# 不同动作默认用于 ROM 评分的主角度
ACTION_PRIMARY_ROM = {
    "M1": "knee_flex_mean",
    "M2": "knee_flex_mean",
    "M3": "knee_flex_mean",
    "M4": "knee_flex_mean",
    "M5": "knee_flex_mean",
    "M6": "hip_flex_mean",
    "M7": "hand_up_max",
    "M8": "hand_up_max",
    "M9": "hand_up_max",
    "M10": "hand_up_max",
}


# =========================================================
# 2. CSV 读取与基础数学函数
# =========================================================
def safe_norm(v, axis=-1, keepdims=False):
    return np.linalg.norm(v, axis=axis, keepdims=keepdims) + 1e-8


def unit_vector(v):
    return v / safe_norm(v, axis=1, keepdims=True)


def angle_between_vectors_deg(a, b):
    """
    a, b: shape = (T, 3)
    返回夹角，单位 degree。
    """

    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)

    cosv = np.sum(a * b, axis=1) / (
        safe_norm(a, axis=1) * safe_norm(b, axis=1)
    )
    cosv = np.clip(cosv, -1.0, 1.0)

    return np.degrees(np.arccos(cosv))


def angle3(A, B, C):
    """
    计算角 ABC。

    A/B/C: shape = (T, 3)
    """

    BA = A - B
    BC = C - B

    return angle_between_vectors_deg(BA, BC)


def project_to_two_axes(v, axis1, axis2):
    """
    将向量 v 投影到由 axis1 和 axis2 张成的平面中。

    v, axis1, axis2: shape = (T, 3)
    """

    a1 = unit_vector(axis1)
    a2 = unit_vector(axis2)

    return (
        np.sum(v * a1, axis=1, keepdims=True) * a1
        + np.sum(v * a2, axis=1, keepdims=True) * a2
    )


def body_scale(seq):
    """
    计算平均骨长，供偏移归一化使用。
    """

    lens = []

    for a, b in BONES:
        lens.append(np.linalg.norm(seq[:, a, :] - seq[:, b, :], axis=1))

    lens = np.stack(lens, axis=1)

    return float(np.mean(lens) + 1e-8)


def ensure_segments_df(segments):
    """
    将 segments 或 segments_df 统一转成 DataFrame。

    必须包含：
        rep_id, start, end
    """

    if isinstance(segments, pd.DataFrame):
        df = segments.copy()
    else:
        df = pd.DataFrame(segments)

    required = ["rep_id", "start", "end"]
    missing = [c for c in required if c not in df.columns]

    if missing:
        raise ValueError(f"segments 缺少必要列: {missing}")

    df["rep_id"] = df["rep_id"].astype(int)
    df["start"] = df["start"].astype(int)
    df["end"] = df["end"].astype(int)

    if "center" not in df.columns:
        df["center"] = ((df["start"] + df["end"]) / 2).round().astype(int)

    return df


# =========================================================
# 3. 关键关节角度计算
# =========================================================
def compute_rom_angles(seq):
    """
    计算关键关节活动角度。

    输入：
        seq: shape = (T, 22, 3)

    返回：
        angles_df: 每一行对应一帧，每一列为一个角度或姿态指标。

    说明：
        膝屈曲角：
            0 度约等于伸直；
            数值越大表示屈曲越明显。

        肩外展角 / 肩前屈角：
            这里是基于骨架点的近似角度。
            肩外展：上臂在冠状面内相对于躯干向下方向的夹角；
            肩前屈：上臂在矢状面内相对于躯干向下方向的夹角。
    """

    seq = np.asarray(seq, dtype=float)

    if seq.ndim != 3 or seq.shape[1:] != (22, 3):
        raise ValueError(f"seq shape 应为 (T, 22, 3)，当前为 {seq.shape}")

    waist = seq[:, J["waist"], :]
    neck = seq[:, J["neck"], :]

    l_shoulder = seq[:, J["l_shoulder"], :]
    r_shoulder = seq[:, J["r_shoulder"], :]
    l_elbow = seq[:, J["l_elbow"], :]
    r_elbow = seq[:, J["r_elbow"], :]
    l_hand = seq[:, J["l_hand"], :]
    r_hand = seq[:, J["r_hand"], :]

    l_hip = seq[:, J["l_hip"], :]
    r_hip = seq[:, J["r_hip"], :]
    l_knee = seq[:, J["l_knee"], :]
    r_knee = seq[:, J["r_knee"], :]
    l_foot = seq[:, J["l_foot"], :]
    r_foot = seq[:, J["r_foot"], :]

    # -----------------------------------------------------
    # 1. 下肢角度
    # -----------------------------------------------------
    l_knee_joint_angle = angle3(l_hip, l_knee, l_foot)
    r_knee_joint_angle = angle3(r_hip, r_knee, r_foot)

    l_knee_flexion = 180.0 - l_knee_joint_angle
    r_knee_flexion = 180.0 - r_knee_joint_angle

    l_hip_flexion = 180.0 - angle3(waist, l_hip, l_knee)
    r_hip_flexion = 180.0 - angle3(waist, r_hip, r_knee)

    # -----------------------------------------------------
    # 2. 上肢角度
    # -----------------------------------------------------
    l_elbow_flexion = 180.0 - angle3(l_shoulder, l_elbow, l_hand)
    r_elbow_flexion = 180.0 - angle3(r_shoulder, r_elbow, r_hand)

    trunk_axis = neck - waist
    vertical_axis = unit_vector(trunk_axis)
    downward_axis = -vertical_axis

    shoulder_lateral_axis = r_shoulder - l_shoulder
    shoulder_lateral_axis = unit_vector(shoulder_lateral_axis)

    depth_axis = np.cross(shoulder_lateral_axis, vertical_axis)
    depth_axis = unit_vector(depth_axis)

    l_upper_arm = l_elbow - l_shoulder
    r_upper_arm = r_elbow - r_shoulder

    # 肩外展：投影到冠状面 lateral-vertical
    l_upper_frontal = project_to_two_axes(
        l_upper_arm,
        shoulder_lateral_axis,
        vertical_axis
    )
    r_upper_frontal = project_to_two_axes(
        r_upper_arm,
        shoulder_lateral_axis,
        vertical_axis
    )

    l_shoulder_abduction = angle_between_vectors_deg(
        l_upper_frontal,
        downward_axis
    )
    r_shoulder_abduction = angle_between_vectors_deg(
        r_upper_frontal,
        downward_axis
    )

    # 肩前屈：投影到矢状面 depth-vertical
    l_upper_sagittal = project_to_two_axes(
        l_upper_arm,
        depth_axis,
        vertical_axis
    )
    r_upper_sagittal = project_to_two_axes(
        r_upper_arm,
        depth_axis,
        vertical_axis
    )

    l_shoulder_flexion = angle_between_vectors_deg(
        l_upper_sagittal,
        downward_axis
    )
    r_shoulder_flexion = angle_between_vectors_deg(
        r_upper_sagittal,
        downward_axis
    )

    # -----------------------------------------------------
    # 3. 躯干与骨盆姿态
    # -----------------------------------------------------
    trunk_vec = neck - waist

    trunk_sagittal_lean = np.degrees(
        np.arctan2(
            np.abs(trunk_vec[:, DEP]),
            np.abs(trunk_vec[:, UP]) + 1e-8
        )
    )

    trunk_frontal_lean = np.degrees(
        np.arctan2(
            np.abs(trunk_vec[:, LAT]),
            np.abs(trunk_vec[:, UP]) + 1e-8
        )
    )

    pelvis_height_diff = np.abs(l_hip[:, UP] - r_hip[:, UP])
    shoulder_height_diff = np.abs(l_shoulder[:, UP] - r_shoulder[:, UP])

    angles_df = pd.DataFrame({
        "frame_idx": np.arange(seq.shape[0]),

        "l_knee_flexion": l_knee_flexion,
        "r_knee_flexion": r_knee_flexion,
        "knee_flexion_mean": 0.5 * (l_knee_flexion + r_knee_flexion),
        "knee_flexion_asymmetry": np.abs(l_knee_flexion - r_knee_flexion),

        "l_hip_flexion": l_hip_flexion,
        "r_hip_flexion": r_hip_flexion,
        "hip_flexion_mean": 0.5 * (l_hip_flexion + r_hip_flexion),
        "hip_flexion_asymmetry": np.abs(l_hip_flexion - r_hip_flexion),

        "l_elbow_flexion": l_elbow_flexion,
        "r_elbow_flexion": r_elbow_flexion,
        "elbow_flexion_mean": 0.5 * (l_elbow_flexion + r_elbow_flexion),
        "elbow_flexion_asymmetry": np.abs(l_elbow_flexion - r_elbow_flexion),

        "l_shoulder_abduction": l_shoulder_abduction,
        "r_shoulder_abduction": r_shoulder_abduction,
        "shoulder_abduction_mean": 0.5 * (
            l_shoulder_abduction + r_shoulder_abduction
        ),
        "shoulder_abduction_asymmetry": np.abs(
            l_shoulder_abduction - r_shoulder_abduction
        ),

        "l_shoulder_flexion": l_shoulder_flexion,
        "r_shoulder_flexion": r_shoulder_flexion,
        "shoulder_flexion_mean": 0.5 * (
            l_shoulder_flexion + r_shoulder_flexion
        ),
        "shoulder_flexion_asymmetry": np.abs(
            l_shoulder_flexion - r_shoulder_flexion
        ),

        "trunk_sagittal_lean": trunk_sagittal_lean,
        "trunk_frontal_lean": trunk_frontal_lean,

        "pelvis_height_diff": pelvis_height_diff,
        "shoulder_height_diff": shoulder_height_diff,
    })

    return angles_df


# =========================================================
# 4. ROM 报告
# =========================================================
def compute_rom_report(
    angles_df,
    segments,
    angle_names=None,
):
    """
    统计每一轮动作的角度最大值、最小值、ROM、均值和标准差。

    返回：
        rom_detail_df:
            每一行 = 一个动作周期 × 一个角度。

        rom_summary_df:
            每一行 = 一个角度的跨周期统计。
    """

    segments_df = ensure_segments_df(segments)

    if angle_names is None:
        angle_names = [
            c for c in angles_df.columns
            if c != "frame_idx"
        ]

    detail_rows = []

    for _, seg in segments_df.iterrows():
        rep_id = int(seg["rep_id"])
        start = int(seg["start"])
        end = int(seg["end"])

        seg_df = angles_df.iloc[start:end + 1]

        for angle_name in angle_names:
            if angle_name not in angles_df.columns:
                continue

            values = seg_df[angle_name].to_numpy(dtype=float)

            if len(values) == 0:
                continue

            min_val = float(np.nanmin(values))
            max_val = float(np.nanmax(values))
            rom = max_val - min_val

            detail_rows.append({
                "rep_id": rep_id,
                "start": start,
                "end": end,
                "duration_frames": end - start + 1,
                "angle_name": angle_name,
                "min_angle_deg": min_val,
                "max_angle_deg": max_val,
                "rom_deg": float(rom),
                "mean_angle_deg": float(np.nanmean(values)),
                "std_angle_deg": float(np.nanstd(values)),
            })

    rom_detail_df = pd.DataFrame(detail_rows)

    if rom_detail_df.empty:
        rom_summary_df = pd.DataFrame()
    else:
        rom_summary_df = (
            rom_detail_df
            .groupby("angle_name")
            .agg(
                mean_rom_deg=("rom_deg", "mean"),
                std_rom_deg=("rom_deg", "std"),
                max_rom_deg=("rom_deg", "max"),
                min_rom_deg=("rom_deg", "min"),
                mean_min_angle_deg=("min_angle_deg", "mean"),
                mean_max_angle_deg=("max_angle_deg", "mean"),
                mean_angle_deg=("mean_angle_deg", "mean"),
                num_reps=("rep_id", "count"),
            )
            .reset_index()
        )

    return rom_detail_df, rom_summary_df


# =========================================================
# 5. 多维质量评分
# =========================================================
def score_higher_better(values, ref=None):
    """
    越大越好的指标评分，输出 0-100。

    若 ref=None，则用当前序列 90 分位数作为参考上限。
    """

    values = np.asarray(values, dtype=float)

    if len(values) == 0:
        return np.array([], dtype=float)

    if ref is None:
        ref = np.nanpercentile(values, 90)

    if not np.isfinite(ref) or abs(ref) < 1e-8:
        return np.full_like(values, 100.0, dtype=float)

    return np.clip(values / ref * 100.0, 0.0, 100.0)


def score_lower_better(values, low_ref=None, high_ref=None):
    """
    越小越好的指标评分，输出 0-100。

    默认使用当前序列 10 分位数和 90 分位数作为评分区间。
    """

    values = np.asarray(values, dtype=float)

    if len(values) == 0:
        return np.array([], dtype=float)

    if low_ref is None:
        low_ref = np.nanpercentile(values, 10)

    if high_ref is None:
        high_ref = np.nanpercentile(values, 90)

    if not np.isfinite(low_ref) or not np.isfinite(high_ref):
        return np.full_like(values, np.nan, dtype=float)

    if abs(high_ref - low_ref) < 1e-8:
        return np.full_like(values, 100.0, dtype=float)

    return np.clip(
        (high_ref - values) / (high_ref - low_ref) * 100.0,
        0.0,
        100.0
    )


def rms(x):
    x = np.asarray(x, dtype=float)

    if len(x) == 0:
        return np.nan

    return float(np.sqrt(np.nanmean(x ** 2)))


def compute_segment_smoothness_metric(signal):
    """
    速度平滑性指标。

    使用三阶差分近似 jerk，计算：
        jerk_rms / velocity_rms

    越小表示越平滑。
    """

    signal = np.asarray(signal, dtype=float)

    if len(signal) < 5:
        return np.nan

    vel = np.diff(signal, n=1)
    jerk = np.diff(signal, n=3)

    return rms(jerk) / (rms(vel) + 1e-8)


def get_primary_angle_for_action(action_id):
    if action_id is None:
        return "knee_flexion_mean"

    action_id = str(action_id).upper()

    return ACTION_PRIMARY_ROM.get(action_id, "knee_flexion_mean")


def compute_quality_scores(
    features_df,
    angles_df,
    segments,
    action_id=None,
    similarity_scores=None,
    weights=None,
):
    """
    从多个维度计算动作质量评分。

    评分维度：
        1. amplitude_score      动作幅度
        2. smoothness_score     速度平滑性
        3. trunk_score          躯干稳定性
        4. symmetry_score       左右对称性
        5. rhythm_score         节奏性
        6. similarity_score_100 模板相似度，可选

    返回：
        quality_df:
            每一行 = 一轮动作。

        quality_summary:
            整体评分摘要。
    """

    segments_df = ensure_segments_df(segments)

    if weights is None:
        weights = {
            "amplitude_score": 0.25,
            "smoothness_score": 0.20,
            "trunk_score": 0.20,
            "symmetry_score": 0.20,
            "rhythm_score": 0.15,
        }

    primary_angle = get_primary_angle_for_action(action_id)

    if primary_angle not in angles_df.columns:
        primary_angle = "knee_flexion_mean"

    # -----------------------------------------------------
    # 1. 基础指标
    # -----------------------------------------------------
    rows = []

    durations = (
        segments_df["end"].to_numpy(dtype=float)
        - segments_df["start"].to_numpy(dtype=float)
        + 1.0
    )

    median_duration = np.nanmedian(durations) if len(durations) else np.nan

    for i, seg in segments_df.iterrows():
        rep_id = int(seg["rep_id"])
        start = int(seg["start"])
        end = int(seg["end"])

        angle_seg = angles_df.iloc[start:end + 1]

        primary_values = angle_seg[primary_angle].to_numpy(dtype=float)

        amplitude_metric = float(
            np.nanmax(primary_values) - np.nanmin(primary_values)
        )

        smoothness_metric = compute_segment_smoothness_metric(primary_values)

        trunk_metric = float(
            np.nanstd(angle_seg["trunk_sagittal_lean"].to_numpy(dtype=float))
            + np.nanstd(angle_seg["trunk_frontal_lean"].to_numpy(dtype=float))
        )

        # 对称性：下肢动作用膝/髋，上肢动作用肩
        action_upper = str(action_id).upper() in ["M7", "M8", "M9", "M10"]

        if action_upper:
            symmetry_col_candidates = [
                "shoulder_abduction_asymmetry",
                "shoulder_flexion_asymmetry",
                "elbow_flexion_asymmetry",
            ]
        else:
            symmetry_col_candidates = [
                "knee_flexion_asymmetry",
                "hip_flexion_asymmetry",
            ]

        symmetry_values = []

        for col in symmetry_col_candidates:
            if col in angle_seg.columns:
                symmetry_values.append(
                    np.nanmean(angle_seg[col].to_numpy(dtype=float))
                )

        symmetry_metric = float(np.nanmean(symmetry_values))

        duration = float(end - start + 1)

        if np.isfinite(median_duration) and median_duration > 1e-8:
            rhythm_metric = abs(duration - median_duration) / median_duration
        else:
            rhythm_metric = np.nan

        rows.append({
            "rep_id": rep_id,
            "start": start,
            "end": end,
            "duration_frames": duration,
            "primary_angle": primary_angle,
            "amplitude_metric_rom_deg": amplitude_metric,
            "smoothness_metric": smoothness_metric,
            "trunk_stability_metric": trunk_metric,
            "symmetry_metric_deg": symmetry_metric,
            "rhythm_metric_duration_dev": rhythm_metric,
        })

    quality_df = pd.DataFrame(rows)

    if quality_df.empty:
        return quality_df, {}

    # -----------------------------------------------------
    # 2. 指标转评分
    # -----------------------------------------------------
    quality_df["amplitude_score"] = score_higher_better(
        quality_df["amplitude_metric_rom_deg"].to_numpy(dtype=float)
    )

    quality_df["smoothness_score"] = score_lower_better(
        quality_df["smoothness_metric"].to_numpy(dtype=float)
    )

    quality_df["trunk_score"] = score_lower_better(
        quality_df["trunk_stability_metric"].to_numpy(dtype=float)
    )

    quality_df["symmetry_score"] = score_lower_better(
        quality_df["symmetry_metric_deg"].to_numpy(dtype=float)
    )

    # 节奏性用绝对阈值更稳定：偏离中位周期 30% 以上记为低分
    rhythm_values = quality_df["rhythm_metric_duration_dev"].to_numpy(dtype=float)
    quality_df["rhythm_score"] = np.clip(
        (1.0 - rhythm_values / 0.30) * 100.0,
        0.0,
        100.0
    )

    # -----------------------------------------------------
    # 3. 模板相似度评分，可选
    # -----------------------------------------------------
    if similarity_scores is not None:
        similarity_scores = np.asarray(similarity_scores, dtype=float)

        if len(similarity_scores) == len(quality_df):
            if np.nanmin(similarity_scores) < 0:
                sim_score_100 = (similarity_scores + 1.0) / 2.0 * 100.0
            else:
                sim_score_100 = similarity_scores * 100.0

            quality_df["similarity_score_100"] = np.clip(
                sim_score_100,
                0.0,
                100.0
            )

    # -----------------------------------------------------
    # 4. 综合评分
    # -----------------------------------------------------
    total = np.zeros(len(quality_df), dtype=float)
    valid_weight_sum = 0.0

    for col, w in weights.items():
        if col in quality_df.columns:
            total += quality_df[col].to_numpy(dtype=float) * w
            valid_weight_sum += w

    if valid_weight_sum > 1e-8:
        quality_df["overall_quality_score"] = total / valid_weight_sum
    else:
        quality_df["overall_quality_score"] = np.nan

    if "similarity_score_100" in quality_df.columns:
        quality_df["overall_with_similarity_score"] = (
            0.70 * quality_df["overall_quality_score"]
            + 0.30 * quality_df["similarity_score_100"]
        )

    # -----------------------------------------------------
    # 5. 摘要
    # -----------------------------------------------------
    score_cols = [
        "amplitude_score",
        "smoothness_score",
        "trunk_score",
        "symmetry_score",
        "rhythm_score",
        "overall_quality_score",
    ]

    if "similarity_score_100" in quality_df.columns:
        score_cols.append("similarity_score_100")
        score_cols.append("overall_with_similarity_score")

    quality_summary = {}

    for col in score_cols:
        quality_summary[f"{col}_mean"] = float(np.nanmean(quality_df[col]))
        quality_summary[f"{col}_std"] = float(np.nanstd(quality_df[col]))

    quality_summary["num_reps"] = int(len(quality_df))
    quality_summary["primary_angle"] = primary_angle

    return quality_df, quality_summary


# =========================================================
# 6. 关节偏移与身体倾斜评估
# =========================================================
def compute_posture_offset_report(
    seq,
    angles_df,
    segments,
):
    """
    评估每一轮动作的身体偏移和躯干倾斜。

    输出指标包括：
        腰部左右偏移范围
        腰部前后偏移范围
        身体质心左右/前后偏移范围
        躯干前后倾最大值/均值/标准差
        躯干左右倾最大值/均值/标准差
        骨盆高度差
        肩部高度差

    坐标偏移同时给出归一化尺度：
        normalized value = raw displacement / mean bone length
    """

    seq = np.asarray(seq, dtype=float)
    segments_df = ensure_segments_df(segments)

    scale = body_scale(seq)

    origin = seq[0, J["waist"], :].copy()
    seq_norm = (seq - origin[None, None, :]) / scale

    waist = seq_norm[:, J["waist"], :]
    com = np.mean(seq_norm, axis=1)

    l_hip = seq_norm[:, J["l_hip"], :]
    r_hip = seq_norm[:, J["r_hip"], :]

    l_shoulder = seq_norm[:, J["l_shoulder"], :]
    r_shoulder = seq_norm[:, J["r_shoulder"], :]

    pelvis_height_diff_norm = np.abs(l_hip[:, UP] - r_hip[:, UP])
    shoulder_height_diff_norm = np.abs(l_shoulder[:, UP] - r_shoulder[:, UP])

    rows = []

    for _, seg in segments_df.iterrows():
        rep_id = int(seg["rep_id"])
        start = int(seg["start"])
        end = int(seg["end"])

        idx = slice(start, end + 1)

        angle_seg = angles_df.iloc[start:end + 1]

        rows.append({
            "rep_id": rep_id,
            "start": start,
            "end": end,

            "waist_lateral_range_norm": float(
                np.nanmax(waist[idx, LAT]) - np.nanmin(waist[idx, LAT])
            ),
            "waist_forward_range_norm": float(
                np.nanmax(waist[idx, DEP]) - np.nanmin(waist[idx, DEP])
            ),
            "waist_vertical_range_norm": float(
                np.nanmax(waist[idx, UP]) - np.nanmin(waist[idx, UP])
            ),

            "com_lateral_range_norm": float(
                np.nanmax(com[idx, LAT]) - np.nanmin(com[idx, LAT])
            ),
            "com_forward_range_norm": float(
                np.nanmax(com[idx, DEP]) - np.nanmin(com[idx, DEP])
            ),
            "com_vertical_range_norm": float(
                np.nanmax(com[idx, UP]) - np.nanmin(com[idx, UP])
            ),

            "trunk_sagittal_lean_mean_deg": float(
                np.nanmean(angle_seg["trunk_sagittal_lean"])
            ),
            "trunk_sagittal_lean_max_deg": float(
                np.nanmax(angle_seg["trunk_sagittal_lean"])
            ),
            "trunk_sagittal_lean_std_deg": float(
                np.nanstd(angle_seg["trunk_sagittal_lean"])
            ),

            "trunk_frontal_lean_mean_deg": float(
                np.nanmean(angle_seg["trunk_frontal_lean"])
            ),
            "trunk_frontal_lean_max_deg": float(
                np.nanmax(angle_seg["trunk_frontal_lean"])
            ),
            "trunk_frontal_lean_std_deg": float(
                np.nanstd(angle_seg["trunk_frontal_lean"])
            ),

            "pelvis_height_diff_mean_norm": float(
                np.nanmean(pelvis_height_diff_norm[idx])
            ),
            "pelvis_height_diff_max_norm": float(
                np.nanmax(pelvis_height_diff_norm[idx])
            ),
            "shoulder_height_diff_mean_norm": float(
                np.nanmean(shoulder_height_diff_norm[idx])
            ),
            "shoulder_height_diff_max_norm": float(
                np.nanmax(shoulder_height_diff_norm[idx])
            ),
        })

    offset_detail_df = pd.DataFrame(rows)

    if offset_detail_df.empty:
        offset_summary_df = pd.DataFrame()
    else:
        metric_cols = [
            c for c in offset_detail_df.columns
            if c not in ["rep_id", "start", "end"]
        ]

        offset_summary_df = (
            offset_detail_df[metric_cols]
            .agg(["mean", "std", "min", "max"])
            .T
            .reset_index()
            .rename(columns={"index": "metric"})
        )

    return offset_detail_df, offset_summary_df


# =========================================================
# 7. 绘图函数：ROM 曲线与分割区间
# =========================================================
def plot_rom_curves_with_segments(
    angles_df,
    segments,
    angle_names=None,
    title="ROM Curves with Segments",
    save_dir=None,
    show=True,
):
    """
    分别绘制多个 ROM 角度曲线，并叠加动作周期分割区间。

    每个角度单独生成一张图。
    """

    segments_df = ensure_segments_df(segments)

    if angle_names is None:
        angle_names = [
            "knee_flexion_mean",
            "hip_flexion_mean",
            "shoulder_abduction_mean",
            "shoulder_flexion_mean",
            "elbow_flexion_mean",
            "trunk_sagittal_lean",
            "trunk_frontal_lean",
        ]

    if save_dir is not None:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

    saved_paths = []

    x = np.arange(len(angles_df))

    for angle_name in angle_names:
        if angle_name not in angles_df.columns:
            continue

        plt.figure(figsize=(14, 5))
        y = angles_df[angle_name].to_numpy(dtype=float)

        plt.plot(x, y, linewidth=2, label=angle_name)

        for _, seg in segments_df.iterrows():
            plt.axvspan(
                int(seg["start"]),
                int(seg["end"]),
                alpha=0.08
            )
            plt.axvline(
                int(seg["start"]),
                linestyle=":",
                linewidth=1
            )
            plt.text(
                int(seg["center"]),
                np.nanmax(y),
                str(int(seg["rep_id"])),
                ha="center",
                va="top",
                fontsize=9
            )

        plt.title(f"{title} - {angle_name}")
        plt.xlabel("Frame")
        plt.ylabel("Angle (deg)")
        plt.grid(alpha=0.3)
        plt.legend()
        plt.tight_layout()

        if save_dir is not None:
            save_path = save_dir / f"rom_{angle_name}.png"
            plt.savefig(save_path, dpi=140, bbox_inches="tight")
            saved_paths.append(str(save_path))

        if show:
            plt.show()
        else:
            plt.close()

    return saved_paths


# =========================================================
# 8. 绘图函数：身体倾斜与偏移
# =========================================================
def plot_posture_lean_with_segments(
    angles_df,
    segments,
    title="Trunk Lean Assessment",
    save_path=None,
    show=True,
):
    """
    绘制躯干前后倾和左右倾曲线，并叠加分割区间。
    """

    segments_df = ensure_segments_df(segments)
    x = np.arange(len(angles_df))

    plt.figure(figsize=(14, 5))

    plt.plot(
        x,
        angles_df["trunk_sagittal_lean"],
        linewidth=2,
        label="trunk_sagittal_lean"
    )

    plt.plot(
        x,
        angles_df["trunk_frontal_lean"],
        linewidth=2,
        label="trunk_frontal_lean"
    )

    for _, seg in segments_df.iterrows():
        plt.axvspan(
            int(seg["start"]),
            int(seg["end"]),
            alpha=0.08
        )
        plt.axvline(
            int(seg["start"]),
            linestyle=":",
            linewidth=1
        )

    plt.title(title)
    plt.xlabel("Frame")
    plt.ylabel("Angle (deg)")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=140, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close()


# =========================================================
# 9. 绘图函数：运动轨迹空间包络图
# =========================================================
def plot_spatial_envelope_2d(
    seq,
    segments=None,
    joint_names=None,
    projection="xz",
    title=None,
    save_path=None,
    show=True,
):
    """
    绘制运动轨迹空间包络图。

    参数：
        seq:
            shape = (T, 22, 3)

        segments:
            可选，分割结果。
            如果提供，则会绘制每一轮动作的空间包络矩形。

        joint_names:
            需要绘制的关节名称。
            None 表示绘制所有关节。
            常用：
                ["waist", "neck", "l_hand", "r_hand", "l_foot", "r_foot"]

        projection:
            "xy", "xz", "yz"
    """

    seq = np.asarray(seq, dtype=float)

    if joint_names is None:
        joint_ids = list(range(22))
        joint_names = [name for _, name in JOINTS]
    else:
        joint_ids = [JOINT_NAME_TO_ID[name] for name in joint_names]

    projection = projection.lower()

    if projection == "xy":
        a, b = 0, 1
        xlabel, ylabel = "X", "Y"
    elif projection == "xz":
        a, b = 0, 2
        xlabel, ylabel = "X", "Z"
    elif projection == "yz":
        a, b = 1, 2
        xlabel, ylabel = "Y", "Z"
    else:
        raise ValueError("projection 必须是 'xy', 'xz', 或 'yz'")

    pts = seq[:, joint_ids, :].reshape(-1, 3)

    x = pts[:, a]
    y = pts[:, b]

    plt.figure(figsize=(8, 8))

    plt.scatter(
        x,
        y,
        s=6,
        alpha=0.25,
        label="all selected joint points"
    )

    # 总空间包络
    xmin, xmax = np.nanmin(x), np.nanmax(x)
    ymin, ymax = np.nanmin(y), np.nanmax(y)

    plt.plot(
        [xmin, xmax, xmax, xmin, xmin],
        [ymin, ymin, ymax, ymax, ymin],
        linewidth=2,
        label="global envelope"
    )

    # 关键关节轨迹
    for jid, jname in zip(joint_ids, joint_names):
        traj = seq[:, jid, :]
        plt.plot(
            traj[:, a],
            traj[:, b],
            linewidth=1.3,
            alpha=0.8,
            label=jname
        )

    # 每个动作周期的空间包络
    if segments is not None:
        segments_df = ensure_segments_df(segments)

        for _, seg in segments_df.iterrows():
            start = int(seg["start"])
            end = int(seg["end"])

            seg_pts = seq[start:end + 1, joint_ids, :].reshape(-1, 3)

            sx = seg_pts[:, a]
            sy = seg_pts[:, b]

            sxmin, sxmax = np.nanmin(sx), np.nanmax(sx)
            symin, symax = np.nanmin(sy), np.nanmax(sy)

            plt.plot(
                [sxmin, sxmax, sxmax, sxmin, sxmin],
                [symin, symin, symax, symax, symin],
                linestyle="--",
                linewidth=1,
                alpha=0.7
            )

            plt.text(
                0.5 * (sxmin + sxmax),
                0.5 * (symin + symax),
                str(int(seg["rep_id"])),
                ha="center",
                va="center",
                fontsize=9
            )

    if title is None:
        title = f"Spatial Envelope - {projection.upper()} Projection"

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.axis("equal")
    plt.grid(alpha=0.3)
    plt.legend(fontsize=8, loc="best")
    plt.tight_layout()

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=140, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close()


def plot_joint_trajectory_3d(
    seq,
    joint_names=None,
    title="3D Joint Trajectory",
    save_path=None,
    show=True,
):
    """
    绘制关键关节的 3D 运动轨迹。
    """

    seq = np.asarray(seq, dtype=float)

    if joint_names is None:
        joint_names = ["waist", "neck", "l_hand", "r_hand", "l_foot", "r_foot"]

    joint_ids = [JOINT_NAME_TO_ID[name] for name in joint_names]

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection="3d")

    for jid, jname in zip(joint_ids, joint_names):
        traj = seq[:, jid, :]

        ax.plot(
            traj[:, 0],
            traj[:, 2],
            traj[:, 1],
            linewidth=1.5,
            label=jname
        )

    ax.set_xlabel("X")
    ax.set_ylabel("Z")
    ax.set_zlabel("Y")
    ax.set_title(title)
    ax.legend()
    ax.grid(alpha=0.3)

    plt.tight_layout()

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=140, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close()


# =========================================================
# 10. 总入口：生成完整报告
# =========================================================
def generate_motion_rom_quality_report(
    csv_path,
    segments,
    action_id=None,
    features_df=None,
    similarity_scores=None,
    output_dir=None,
    plot=True,
    show_plots=True,
):
    """
    总入口函数。

    参数：
        csv_path:
            宽表骨架 CSV 路径。

        segments:
            分割结果，可以是：
                result["segments_df"]
                result["segments"]
                DataFrame

        action_id:
            动作编号，例如 "M1"。

        features_df:
            已提取特征。
            如果为 None，并且存在 compute_features_from_csv，则自动计算。

        similarity_scores:
            每一轮动作的模板相似度分数。
            可以传入 analyze_action_csv() 返回的 result["per_segment_scores"]。

        output_dir:
            如果不为 None，则保存报告 CSV 和图像。

        plot:
            是否绘图。

        show_plots:
            是否显示图像。
            如果只想保存图像，可设置 show_plots=False。

    返回：
        report: dict
    """

    csv_path = Path(csv_path)
    segments_df = ensure_segments_df(segments)

    seq = read_skeleton_csv(csv_path)
    angles_df = compute_rom_angles(seq)

    if features_df is None and compute_features_from_csv is not None:
        features_df = compute_features_from_csv(
            csv_path=csv_path,
            normalize=True,
            smooth=True,
            return_dataframe=True,
        )

    # -----------------------------------------------------
    # 1. ROM 报告
    # -----------------------------------------------------
    rom_detail_df, rom_summary_df = compute_rom_report(
        angles_df=angles_df,
        segments=segments_df,
        angle_names=None,
    )

    # -----------------------------------------------------
    # 2. 多维评分
    # -----------------------------------------------------
    quality_df, quality_summary = compute_quality_scores(
        features_df=features_df,
        angles_df=angles_df,
        segments=segments_df,
        action_id=action_id,
        similarity_scores=similarity_scores,
    )

    # -----------------------------------------------------
    # 3. 偏移与倾斜报告
    # -----------------------------------------------------
    offset_detail_df, offset_summary_df = compute_posture_offset_report(
        seq=seq,
        angles_df=angles_df,
        segments=segments_df,
    )

    # -----------------------------------------------------
    # 4. 保存
    # -----------------------------------------------------
    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        angles_df.to_csv(
            output_dir / "framewise_angles.csv",
            index=False,
            encoding="utf-8-sig"
        )

        rom_detail_df.to_csv(
            output_dir / "rom_detail_by_rep.csv",
            index=False,
            encoding="utf-8-sig"
        )

        rom_summary_df.to_csv(
            output_dir / "rom_summary.csv",
            index=False,
            encoding="utf-8-sig"
        )

        quality_df.to_csv(
            output_dir / "quality_scores_by_rep.csv",
            index=False,
            encoding="utf-8-sig"
        )

        pd.Series(quality_summary).to_csv(
            output_dir / "quality_summary.csv",
            encoding="utf-8-sig"
        )

        offset_detail_df.to_csv(
            output_dir / "posture_offset_detail_by_rep.csv",
            index=False,
            encoding="utf-8-sig"
        )

        offset_summary_df.to_csv(
            output_dir / "posture_offset_summary.csv",
            index=False,
            encoding="utf-8-sig"
        )

    # -----------------------------------------------------
    # 5. 绘图
    # -----------------------------------------------------
    if plot:
        fig_dir = None

        if output_dir is not None:
            fig_dir = output_dir / "figures"
            fig_dir.mkdir(parents=True, exist_ok=True)

        plot_rom_curves_with_segments(
            angles_df=angles_df,
            segments=segments_df,
            save_dir=fig_dir / "rom_curves" if fig_dir else None,
            show=show_plots,
        )

        plot_posture_lean_with_segments(
            angles_df=angles_df,
            segments=segments_df,
            save_path=fig_dir / "trunk_lean.png" if fig_dir else None,
            show=show_plots,
        )

        plot_spatial_envelope_2d(
            seq=seq,
            segments=segments_df,
            joint_names=["waist", "neck", "l_hand", "r_hand", "l_foot", "r_foot"],
            projection="xz",
            save_path=fig_dir / "spatial_envelope_xz.png" if fig_dir else None,
            show=show_plots,
        )

        plot_spatial_envelope_2d(
            seq=seq,
            segments=segments_df,
            joint_names=["waist", "neck", "l_hand", "r_hand", "l_foot", "r_foot"],
            projection="xy",
            save_path=fig_dir / "spatial_envelope_xy.png" if fig_dir else None,
            show=show_plots,
        )

        plot_joint_trajectory_3d(
            seq=seq,
            joint_names=["waist", "neck", "l_hand", "r_hand", "l_foot", "r_foot"],
            save_path=fig_dir / "joint_trajectory_3d.png" if fig_dir else None,
            show=show_plots,
        )

    report = {
        "seq": seq,
        "angles_df": angles_df,
        "rom_detail_df": rom_detail_df,
        "rom_summary_df": rom_summary_df,
        "quality_df": quality_df,
        "quality_summary": quality_summary,
        "offset_detail_df": offset_detail_df,
        "offset_summary_df": offset_summary_df,
    }

    return report


# =========================================================
# 11. 从 analyze_action_csv() 的结果直接生成报告
# =========================================================
def generate_report_from_action_api_result(
    csv_path,
    action_id,
    api_result,
    output_dir=None,
    plot=True,
    show_plots=True,
):
    """
    与你前面 analyze_action_csv() API 对接的快捷函数。

    api_result 需要至少包含：
        api_result["segments_df"] 或 api_result["segments"]
        api_result["per_segment_scores"] 可选
        api_result["features_df"] 可选
    """

    if "segments_df" in api_result:
        segments = api_result["segments_df"]
    elif "segments" in api_result:
        segments = api_result["segments"]
    else:
        raise ValueError("api_result 中必须包含 segments_df 或 segments")

    similarity_scores = api_result.get("per_segment_scores", None)
    features_df = api_result.get("features_df", None)

    return generate_motion_rom_quality_report(
        csv_path=csv_path,
        segments=segments,
        action_id=action_id,
        features_df=features_df,
        similarity_scores=similarity_scores,
        output_dir=output_dir,
        plot=plot,
        show_plots=show_plots,
    )
