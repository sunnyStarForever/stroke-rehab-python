from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter


# =========================================================
# 1. 关节索引，0-based
# =========================================================
J = {
    "waist": 0,
    "spine": 1,
    "chest": 2,
    "neck": 3,
    "head": 4,
    "head_tip": 5,
    "l_collar": 6,
    "l_shoulder": 7,
    "l_elbow": 8,
    "l_hand": 9,
    "r_collar": 10,
    "r_shoulder": 11,
    "r_elbow": 12,
    "r_hand": 13,
    "l_hip": 14,
    "l_knee": 15,
    "l_foot": 16,
    "l_toe": 17,
    "r_hip": 18,
    "r_knee": 19,
    "r_foot": 20,
    "r_toe": 21,
}

JOINTS = [
    (0,  "waist"),
    (1,  "spine"),
    (2,  "chest"),
    (3,  "neck"),
    (4,  "head"),
    (5,  "head_tip"),
    (6,  "l_collar"),
    (7,  "l_shoulder"),
    (8,  "l_elbow"),
    (9,  "l_hand"),
    (10, "r_collar"),
    (11, "r_shoulder"),
    (12, "r_elbow"),
    (13, "r_hand"),
    (14, "l_hip"),
    (15, "l_knee"),
    (16, "l_foot"),
    (17, "l_toe"),
    (18, "r_hip"),
    (19, "r_knee"),
    (20, "r_foot"),
    (21, "r_toe"),
]

# 默认坐标轴：x=左右，y=上下，z=前后
LAT = 0
UP = 1
DEP = 2

N_JOINTS = 22
N_DIMS = 3
VALUES_PER_FRAME = N_JOINTS * N_DIMS


# =========================================================
# 2. 骨骼连接关系
#    用于计算平均骨长
# =========================================================
BONES = [
    (0, 1), (1, 2), (2, 3), (3, 4), (4, 5),
    (3, 6), (6, 7), (7, 8), (8, 9),
    (3, 10), (10, 11), (11, 12), (12, 13),
    (0, 14), (14, 15), (15, 16), (16, 17),
    (0, 18), (18, 19), (19, 20), (20, 21),
]


# =========================================================
# 3. CSV 读取函数
# =========================================================
def read_skeleton_csv(csv_path):
    """
    直接读取骨架 CSV 文件。

    支持两种格式：
    1. frame_idx + 66 个坐标列；
    2. 只有 66 个坐标列。

    如果 CSV 具有标准列名：
        00_waist_x, 00_waist_y, 00_waist_z, ...

    则会优先按标准列名读取。

    返回：
        seq: np.ndarray, shape = (T, 22, 3)
    """

    csv_path = Path(csv_path)

    if not csv_path.exists():
        raise FileNotFoundError(f"CSV 文件不存在: {csv_path}")

    df = pd.read_csv(csv_path)

    if df.empty:
        raise ValueError(f"CSV 文件为空: {csv_path}")

    # -----------------------------------------------------
    # 情况 1：标准列名
    # -----------------------------------------------------
    expected_cols = []

    for joint_id, joint_name in JOINTS:
        for axis in ["x", "y", "z"]:
            expected_cols.append(f"{joint_id:02d}_{joint_name}_{axis}")

    if all(col in df.columns for col in expected_cols):
        data = df[expected_cols].to_numpy(dtype=float)

    else:
        # -------------------------------------------------
        # 情况 2：没有标准列名，按数值列读取
        # -------------------------------------------------
        if "frame_idx" in df.columns:
            df = df.drop(columns=["frame_idx"])

        numeric_df = df.select_dtypes(include=[np.number])

        if numeric_df.shape[1] == VALUES_PER_FRAME + 1:
            # 第一列可能是帧编号
            data = numeric_df.iloc[:, 1:].to_numpy(dtype=float)

        elif numeric_df.shape[1] == VALUES_PER_FRAME:
            data = numeric_df.to_numpy(dtype=float)

        else:
            raise ValueError(
                f"CSV 坐标列数不符合要求。期望 66 个坐标列，"
                f"或者 frame_idx + 66 个坐标列；当前数值列数为 {numeric_df.shape[1]}"
            )

    if data.shape[1] != VALUES_PER_FRAME:
        raise ValueError(
            f"读取后的坐标列数错误，期望 {VALUES_PER_FRAME}，当前为 {data.shape[1]}"
        )

    seq = data.reshape(-1, N_JOINTS, N_DIMS)

    return seq


# =========================================================
# 4. 基础工具函数
# =========================================================
def smooth_signal(x, win=11, poly=3):
    """
    对一维特征曲线进行 Savitzky-Golay 平滑。
    """

    x = np.asarray(x, dtype=float)

    if len(x) < 5:
        return x.copy()

    if win % 2 == 0:
        win += 1

    if win >= len(x):
        win = len(x) - 1 if len(x) % 2 == 0 else len(x)

    if win < 5:
        return x.copy()

    return savgol_filter(
        x,
        window_length=win,
        polyorder=min(poly, win - 1)
    )


def angle3(A, B, C):
    """
    计算角 ABC。

    参数：
        A, B, C: shape = (T, 3)

    返回：
        angle: shape = (T,)
        单位：degree
    """

    A = np.asarray(A, dtype=float)
    B = np.asarray(B, dtype=float)
    C = np.asarray(C, dtype=float)

    BA = A - B
    BC = C - B

    numerator = np.sum(BA * BC, axis=1)
    denominator = (
        np.linalg.norm(BA, axis=1)
        * np.linalg.norm(BC, axis=1)
        + 1e-8
    )

    cos_value = np.clip(numerator / denominator, -1.0, 1.0)

    return np.degrees(np.arccos(cos_value))


def body_scale(seq):
    """
    用平均骨长计算人体尺度。
    """

    seq = np.asarray(seq, dtype=float)

    lengths = []

    for a, b in BONES:
        length = np.linalg.norm(seq[:, a, :] - seq[:, b, :], axis=1)
        lengths.append(length)

    lengths = np.stack(lengths, axis=1)

    return float(np.mean(lengths) + 1e-8)


def normalize_sequence(seq):
    """
    骨架归一化。

    处理逻辑：
    1. 减去第一帧 waist 位置；
    2. 除以平均骨长。

    这样可以减少不同被试体型尺度差异，同时保留整体运动趋势。
    """

    seq = np.asarray(seq, dtype=float)

    if seq.ndim != 3 or seq.shape[1:] != (22, 3):
        raise ValueError(f"seq shape 应为 (T, 22, 3)，当前为 {seq.shape}")

    scale = body_scale(seq)
    origin = seq[0, J["waist"], :].copy()

    seq_norm = (seq - origin[None, None, :]) / scale

    return seq_norm


# =========================================================
# 5. 核心函数：从序列计算特征
# =========================================================
def compute_features_from_sequence(
    seq,
    normalize=True,
    smooth=True,
    smooth_win=11,
    smooth_poly=3,
):
    """
    从骨架序列中计算运动学特征。

    参数：
        seq:
            np.ndarray, shape = (T, 22, 3)

        normalize:
            是否进行尺度归一化。

        smooth:
            是否对每个特征曲线进行平滑。

        smooth_win:
            平滑窗口长度。

        smooth_poly:
            Savitzky-Golay 多项式阶数。

    返回：
        features: dict[str, np.ndarray]
            每个 key 是特征名；
            每个 value 是 shape = (T,) 的一维特征曲线。
    """

    seq = np.asarray(seq, dtype=float)

    if seq.ndim != 3 or seq.shape[1:] != (22, 3):
        raise ValueError(f"seq shape 应为 (T, 22, 3)，当前为 {seq.shape}")

    if normalize:
        seq = normalize_sequence(seq)

    # -----------------------------------------------------
    # 关节坐标
    # -----------------------------------------------------
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
    # 1）全局 / 基础运动学特征
    # -----------------------------------------------------
    vel = np.gradient(seq, axis=0)
    motion_energy = np.mean(np.linalg.norm(vel, axis=2), axis=1)

    body_com_height = np.mean(seq[:, :, UP], axis=1)
    waist_height = waist[:, UP]
    waist_lateral_shift = waist[:, LAT]
    waist_forward_shift = waist[:, DEP]

    # -----------------------------------------------------
    # 2）躯干相关特征
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

    pelvis_drop = np.abs(l_hip[:, UP] - r_hip[:, UP])

    # -----------------------------------------------------
    # 3）下肢相关特征
    # -----------------------------------------------------
    l_knee_flex = 180.0 - angle3(l_hip, l_knee, l_foot)
    r_knee_flex = 180.0 - angle3(r_hip, r_knee, r_foot)
    knee_flex_mean = 0.5 * (l_knee_flex + r_knee_flex)

    l_hip_flex = 180.0 - angle3(waist, l_hip, l_knee)
    r_hip_flex = 180.0 - angle3(waist, r_hip, r_knee)
    hip_flex_mean = 0.5 * (l_hip_flex + r_hip_flex)

    foot_up_max = np.maximum(l_foot[:, UP], r_foot[:, UP])

    l_foot_forward = np.abs(l_foot[:, DEP] - l_hip[:, DEP])
    r_foot_forward = np.abs(r_foot[:, DEP] - r_hip[:, DEP])
    foot_forward_max = np.maximum(l_foot_forward, r_foot_forward)

    foot_lateral_spread = np.abs(l_foot[:, LAT] - r_foot[:, LAT])

    # -----------------------------------------------------
    # 4）上肢相关特征
    # -----------------------------------------------------
    hand_up_max = np.maximum(l_hand[:, UP], r_hand[:, UP])

    l_hand_lateral = np.abs(l_hand[:, LAT] - l_shoulder[:, LAT])
    r_hand_lateral = np.abs(r_hand[:, LAT] - r_shoulder[:, LAT])
    hand_lateral_max = np.maximum(l_hand_lateral, r_hand_lateral)

    l_hand_forward = np.abs(l_hand[:, DEP] - l_shoulder[:, DEP])
    r_hand_forward = np.abs(r_hand[:, DEP] - r_shoulder[:, DEP])
    hand_forward_max = np.maximum(l_hand_forward, r_hand_forward)

    hands_dist = np.linalg.norm(l_hand - r_hand, axis=1)

    l_elbow_flex = 180.0 - angle3(l_shoulder, l_elbow, l_hand)
    r_elbow_flex = 180.0 - angle3(r_shoulder, r_elbow, r_hand)
    elbow_flex_mean = 0.5 * (l_elbow_flex + r_elbow_flex)

    trunk_unit = trunk_vec / (
        np.linalg.norm(trunk_vec, axis=1, keepdims=True) + 1e-8
    )

    l_upperarm = l_elbow - l_shoulder
    r_upperarm = r_elbow - r_shoulder

    l_upperarm_unit = l_upperarm / (
        np.linalg.norm(l_upperarm, axis=1, keepdims=True) + 1e-8
    )
    r_upperarm_unit = r_upperarm / (
        np.linalg.norm(r_upperarm, axis=1, keepdims=True) + 1e-8
    )

    l_arm_elev = np.degrees(
        np.arccos(
            np.clip(
                np.sum(l_upperarm_unit * trunk_unit, axis=1),
                -1.0,
                1.0
            )
        )
    )

    r_arm_elev = np.degrees(
        np.arccos(
            np.clip(
                np.sum(r_upperarm_unit * trunk_unit, axis=1),
                -1.0,
                1.0
            )
        )
    )

    arm_elevation_max = np.maximum(l_arm_elev, r_arm_elev)

    l_forearm = l_hand - l_elbow
    r_forearm = r_hand - r_elbow

    l_forearm_az = np.unwrap(
        np.arctan2(l_forearm[:, DEP], l_forearm[:, LAT])
    )
    r_forearm_az = np.unwrap(
        np.arctan2(r_forearm[:, DEP], r_forearm[:, LAT])
    )

    l_forearm_rot = np.abs(l_forearm_az - l_forearm_az[0])
    r_forearm_rot = np.abs(r_forearm_az - r_forearm_az[0])
    forearm_rotation_proxy = np.maximum(l_forearm_rot, r_forearm_rot)

    # -----------------------------------------------------
    # 汇总特征
    # -----------------------------------------------------
    features = {
        # Global / trunk
        "motion_energy": motion_energy,
        "body_com_height": body_com_height,
        "waist_height": waist_height,
        "waist_lateral_shift": waist_lateral_shift,
        "waist_forward_shift": waist_forward_shift,
        "trunk_sagittal_lean": trunk_sagittal_lean,
        "trunk_frontal_lean": trunk_frontal_lean,
        "pelvis_drop": pelvis_drop,

        # Lower limb
        "knee_flex_mean": knee_flex_mean,
        "hip_flex_mean": hip_flex_mean,
        "foot_up_max": foot_up_max,
        "foot_forward_max": foot_forward_max,
        "foot_lateral_spread": foot_lateral_spread,

        # Upper limb
        "hand_up_max": hand_up_max,
        "hand_lateral_max": hand_lateral_max,
        "hand_forward_max": hand_forward_max,
        "hands_dist": hands_dist,
        "elbow_flex_mean": elbow_flex_mean,
        "arm_elevation_max": arm_elevation_max,
        "forearm_rotation_proxy": forearm_rotation_proxy,
    }

    if smooth:
        features = {
            name: smooth_signal(
                signal,
                win=smooth_win,
                poly=smooth_poly
            )
            for name, signal in features.items()
        }

    return features


# =========================================================
# 6. 核心函数：直接从 CSV 计算特征
# =========================================================
def compute_features_from_csv(
    csv_path,
    normalize=True,
    smooth=True,
    smooth_win=11,
    smooth_poly=3,
    return_dataframe=True,
):
    """
    直接从 CSV 文件读取骨架数据并计算特征。

    参数：
        csv_path:
            骨架 CSV 文件路径。

        normalize:
            是否进行骨架归一化。

        smooth:
            是否进行特征曲线平滑。

        smooth_win:
            平滑窗口长度。

        smooth_poly:
            平滑多项式阶数。

        return_dataframe:
            True  : 返回 pd.DataFrame；
            False : 返回 dict[str, np.ndarray]。

    返回：
        features_df 或 features_dict
    """

    seq = read_skeleton_csv(csv_path)

    features = compute_features_from_sequence(
        seq,
        normalize=normalize,
        smooth=smooth,
        smooth_win=smooth_win,
        smooth_poly=smooth_poly,
    )

    if return_dataframe:
        df_features = pd.DataFrame(features)
        df_features.insert(0, "frame_idx", np.arange(len(df_features)))
        return df_features

    return features