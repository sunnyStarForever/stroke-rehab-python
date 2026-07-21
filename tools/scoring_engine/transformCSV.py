import sys
from pathlib import Path

import numpy as np
import pandas as pd


# =========================================================
# 1. 目标关节顺序：必须和之前特征提取代码一致
# =========================================================
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

N_JOINTS = 22
AXES = ["x", "y", "z"]


# =========================================================
# 2. 构建目标列名
# =========================================================
def build_target_columns():
    cols = []

    for joint_id, joint_name in JOINTS:
        for axis in AXES:
            cols.append(f"{joint_id:02d}_{joint_name}_{axis}")

    return cols


# =========================================================
# 3. 解析 flip_axes 参数
# =========================================================
def normalize_flip_axes(flip_axes):
    """
    将 flip_axes 统一解析成 set。

    支持输入：
        None
        "x"
        "y"
        "z"
        "xy"
        "xz"
        "yz"
        "xyz"
        "x,y,z"
        ["x", "z"]
        ("x", "y")

    返回：
        set，例如 {"x", "z"}
    """

    if flip_axes is None:
        return set()

    if isinstance(flip_axes, str):
        s = flip_axes.lower().replace(" ", "")

        if "," in s:
            axes = set(s.split(","))
        else:
            axes = set(list(s))

    elif isinstance(flip_axes, (list, tuple, set)):
        axes = set(str(a).lower().strip() for a in flip_axes)

    else:
        raise TypeError(
            "flip_axes 必须是 None、字符串、list、tuple 或 set，"
            f"当前类型为 {type(flip_axes)}"
        )

    valid_axes = {"x", "y", "z"}
    invalid_axes = axes - valid_axes

    if invalid_axes:
        raise ValueError(
            f"flip_axes 中包含非法轴: {invalid_axes}，"
            "只能包含 'x', 'y', 'z'"
        )

    return axes


# =========================================================
# 4. 单个长表 CSV 转换为宽表 CSV
# =========================================================
def convert_long_skeleton_csv_to_wide_csv(
    input_csv,
    output_csv,
    frame_col="frame_id",
    joint_col="joint_index",
    x_col="x_m",
    y_col="y_m",
    z_col="z_m",
    valid_col="valid",
    only_valid=True,
    drop_incomplete_frames=True,
    flip_axes=("y",),
):
    """
    将长表骨架 CSV 转换为宽表骨架 CSV。

    输入格式：
        每一行是一个关节：
            frame_id, joint_index, joint_name, x_m, y_m, z_m, ...

    输出格式：
        每一行是一帧：
            frame_idx,
            00_waist_x, 00_waist_y, 00_waist_z,
            01_spine_x, 01_spine_y, 01_spine_z,
            ...
            21_r_toe_x, 21_r_toe_y, 21_r_toe_z

    坐标翻转：
        flip_axes=("y",)：
            x_out = x_m
            y_out = -y_m
            z_out = z_m

        flip_axes=("x", "z")：
            x_out = -x_m
            y_out = y_m
            z_out = -z_m

        flip_axes=None：
            不翻转任何轴。

    参数：
        input_csv:
            原始长表 CSV 文件路径。

        output_csv:
            输出宽表 CSV 文件路径。

        only_valid:
            如果为 True，并且存在 valid 列，则只保留 valid == True 的关节。

        drop_incomplete_frames:
            如果为 True，则删除缺少 22 个关节的帧。
            如果为 False，则缺失值保留为 NaN。

        flip_axes:
            控制需要翻转的坐标轴。
            可以是 None、"y"、"xz"、"x,y,z"、["x", "z"] 等。
    """

    input_csv = Path(input_csv)
    output_csv = Path(output_csv)

    if not input_csv.exists():
        raise FileNotFoundError(f"输入文件不存在: {input_csv}")

    flip_axes = normalize_flip_axes(flip_axes)

    df = pd.read_csv(input_csv)

    required_cols = [frame_col, joint_col, x_col, y_col, z_col]
    missing_cols = [c for c in required_cols if c not in df.columns]

    if missing_cols:
        raise ValueError(
            f"输入 CSV 缺少必要列: {missing_cols}\n"
            f"当前列名为: {list(df.columns)}"
        )

    # -----------------------------------------------------
    # 1. 可选：只保留 valid == True 的点
    # -----------------------------------------------------
    if only_valid and valid_col in df.columns:
        df = df[
            df[valid_col].astype(str).str.strip().str.upper().isin(
                ["TRUE", "1", "YES", "Y"]
            )
        ].copy()

    # -----------------------------------------------------
    # 2. 只保留 0-21 号关节
    # -----------------------------------------------------
    df = df[df[joint_col].between(0, 21)].copy()

    if df.empty:
        raise ValueError("过滤后没有剩余有效关节，请检查 valid 或 joint_index 数据。")

    # -----------------------------------------------------
    # 3. 坐标列转为数值
    # -----------------------------------------------------
    for col in [x_col, y_col, z_col]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=[x_col, y_col, z_col]).copy()

    if df.empty:
        raise ValueError("坐标列转为数值后没有剩余有效数据。")

    # -----------------------------------------------------
    # 4. 根据 flip_axes 翻转坐标轴
    # -----------------------------------------------------
    axis_to_col = {
        "x": x_col,
        "y": y_col,
        "z": z_col,
    }

    for axis in flip_axes:
        col = axis_to_col[axis]
        df[col] = -df[col]

    # -----------------------------------------------------
    # 5. 同一帧同一关节如果有重复，取均值
    # -----------------------------------------------------
    df = (
        df.groupby([frame_col, joint_col], as_index=False)[[x_col, y_col, z_col]]
        .mean()
    )

    # -----------------------------------------------------
    # 6. 检查每帧关节数
    # -----------------------------------------------------
    joint_count_per_frame = df.groupby(frame_col)[joint_col].nunique()

    incomplete_frames = joint_count_per_frame[
        joint_count_per_frame < N_JOINTS
    ].index

    if len(incomplete_frames) > 0:
        print(
            f"[WARN] 检测到 {len(incomplete_frames)} 帧关节数量不足 22。",
            file=sys.stderr,
            flush=True,
        )

        if drop_incomplete_frames:
            print("[INFO] 已删除关节不完整的帧。", file=sys.stderr, flush=True)
            df = df[~df[frame_col].isin(incomplete_frames)].copy()

    if df.empty:
        raise ValueError("过滤后没有剩余有效帧，请检查 valid 或 joint_index 数据。")

    # -----------------------------------------------------
    # 7. 按 frame_id 和 joint_index 排序
    # -----------------------------------------------------
    frame_ids = sorted(df[frame_col].unique())

    rows = []
    target_cols = build_target_columns()

    for new_frame_idx, frame_id in enumerate(frame_ids):
        frame_df = df[df[frame_col] == frame_id].copy()
        frame_df = frame_df.set_index(joint_col)

        row = {
            "frame_idx": new_frame_idx,
        }

        for joint_id, joint_name in JOINTS:
            for axis, src_col in zip(AXES, [x_col, y_col, z_col]):
                target_col = f"{joint_id:02d}_{joint_name}_{axis}"

                if joint_id in frame_df.index:
                    row[target_col] = frame_df.loc[joint_id, src_col]
                else:
                    row[target_col] = np.nan

        rows.append(row)

    wide_df = pd.DataFrame(rows)

    # 保证列顺序严格一致
    wide_df = wide_df[["frame_idx"] + target_cols]

    # -----------------------------------------------------
    # 8. 保存
    # -----------------------------------------------------
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    wide_df.to_csv(output_csv, index=False, encoding="utf-8-sig")

    print("转换完成。", file=sys.stderr, flush=True)
    print(f"输入文件: {input_csv}", file=sys.stderr, flush=True)
    print(f"输出文件: {output_csv}", file=sys.stderr, flush=True)
    print(f"输出帧数: {len(wide_df)}", file=sys.stderr, flush=True)
    print(f"输出列数: {wide_df.shape[1]}", file=sys.stderr, flush=True)
    print(
        f"翻转坐标轴: {sorted(flip_axes) if flip_axes else 'None'}",
        file=sys.stderr,
        flush=True,
    )

    return wide_df


# =========================================================
# 5. 批量转换一个文件夹下所有 CSV
# =========================================================
def batch_convert_long_csv_to_wide_csv(
    input_dir,
    output_dir,
    pattern="*.csv",
    **kwargs
):
    """
    批量转换文件夹下的所有长表 CSV。
    """

    input_dir = Path(input_dir)
    output_dir = Path(output_dir)

    csv_files = sorted(input_dir.rglob(pattern))

    if len(csv_files) == 0:
        raise ValueError(f"没有找到 CSV 文件: {input_dir}")

    results = []

    for i, csv_path in enumerate(csv_files, start=1):
        rel_path = csv_path.relative_to(input_dir)
        output_csv = output_dir / rel_path

        print(f"\n[{i}/{len(csv_files)}] converting: {csv_path}", file=sys.stderr, flush=True)

        try:
            wide_df = convert_long_skeleton_csv_to_wide_csv(
                input_csv=csv_path,
                output_csv=output_csv,
                **kwargs
            )

            results.append({
                "file_name": csv_path.name,
                "status": "success",
                "input_path": str(csv_path),
                "output_path": str(output_csv),
                "num_frames": len(wide_df),
                "flip_axes": str(kwargs.get("flip_axes", ("y",))),
                "error": "",
            })

        except Exception as e:
            print(f"[ERROR] {csv_path}: {e}", file=sys.stderr, flush=True)

            results.append({
                "file_name": csv_path.name,
                "status": "failed",
                "input_path": str(csv_path),
                "output_path": str(output_csv),
                "num_frames": None,
                "flip_axes": str(kwargs.get("flip_axes", ("y",))),
                "error": str(e),
            })

    log_df = pd.DataFrame(results)
    log_path = output_dir / "long_to_wide_conversion_log.csv"
    output_dir.mkdir(parents=True, exist_ok=True)
    log_df.to_csv(log_path, index=False, encoding="utf-8-sig")

    print("\n批量转换完成。", file=sys.stderr, flush=True)
    print(f"日志文件: {log_path}", file=sys.stderr, flush=True)

    return log_df


# =========================================================
# 6. 使用示例
# =========================================================
if __name__ == "__main__":

    # -----------------------------------------------------
    # 示例 1：转换单个文件
    # -----------------------------------------------------
    # input_csv = r"examples\skeleton3d.csv"
    # output_csv = r"examples\skeleton_wide.csv"

    # convert_long_skeleton_csv_to_wide_csv(
    #     input_csv=input_csv,
    #     output_csv=output_csv,
    #     frame_col="frame_id",
    #     joint_col="joint_index",
    #     x_col="x_m",
    #     y_col="y_m",
    #     z_col="z_m",
    #     valid_col="valid",
    #     only_valid=True,
    #     drop_incomplete_frames=True,

    #     # 这里控制翻转方向
    #     # 只翻转 y 轴：
    #     # flip_axes=("y",),

    #     # 不翻转任何轴：
    #     # flip_axes=None,

    #     # 同时翻转 x 和 z：
    #     # flip_axes=("x", "z"),

    #     # 三个轴都翻转：
    #     flip_axes=("x", "y", "z"),
    # )

    # -----------------------------------------------------
    # 示例 2：批量转换
    # -----------------------------------------------------
    input_dir = r"data\raw"
    output_dir = r"data\processed"
    
    batch_convert_long_csv_to_wide_csv(
        input_dir=input_dir,
        output_dir=output_dir,
        pattern="*.csv",
        frame_col="frame_id",
        joint_col="joint_index",
        x_col="x_m",
        y_col="y_m",
        z_col="z_m",
        valid_col="valid",
        only_valid=True,
        drop_incomplete_frames=True,
    
        # 批量转换时也可以控制翻转方向
        flip_axes=("x", "y", "z"),
    )
