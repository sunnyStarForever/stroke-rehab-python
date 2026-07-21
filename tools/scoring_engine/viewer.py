from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


# =========================================================
# 1. 关节名称与编号
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
N_DIMS = 3
VALUES_PER_FRAME = N_JOINTS * N_DIMS


# =========================================================
# 2. 骨骼连接关系
#    与你原始 Python 代码中的 J 保持一致
# =========================================================
BONES = np.array([
    [1, 2], [2, 3], [3, 4], [4, 5], [5, 6],
    [4, 7], [7, 8], [8, 9], [9, 10],
    [4, 11], [11, 12], [12, 13], [13, 14],
    [1, 19], [19, 20], [20, 21], [21, 22],
    [1, 15], [15, 16], [16, 17], [17, 18],
], dtype=int) - 1


# =========================================================
# 3. 读取 CSV 为 shape = (T, 22, 3)
# =========================================================
def read_skeleton_csv(csv_path):
    """
    读取骨架 CSV 文件。

    支持两种格式：
    1. frame_idx + 66 个坐标列，共 67 列；
    2. 只有 66 个坐标列，无 frame_idx。

    返回：
        seq: np.ndarray, shape = (T, 22, 3)
             T 表示帧数。
    """

    csv_path = Path(csv_path)

    if not csv_path.exists():
        raise FileNotFoundError(f"CSV 文件不存在: {csv_path}")

    df = pd.read_csv(csv_path)

    if df.empty:
        raise ValueError(f"CSV 文件为空: {csv_path}")

    # -----------------------------------------------------
    # 情况 1：优先按照标准列名读取
    # 例如：
    # 00_waist_x, 00_waist_y, 00_waist_z, ...
    # -----------------------------------------------------
    expected_cols = []

    for joint_id, joint_name in JOINTS:
        for axis in ["x", "y", "z"]:
            expected_cols.append(f"{joint_id:02d}_{joint_name}_{axis}")

    if all(col in df.columns for col in expected_cols):
        data = df[expected_cols].to_numpy(dtype=float)

    else:
        # -------------------------------------------------
        # 情况 2：没有标准列名，则读取数值列
        # 如果包含 frame_idx，则去掉 frame_idx
        # -------------------------------------------------
        if "frame_idx" in df.columns:
            df_data = df.drop(columns=["frame_idx"])
        else:
            df_data = df.copy()

        # 只保留数值列
        df_data = df_data.select_dtypes(include=[np.number])

        if df_data.shape[1] == VALUES_PER_FRAME + 1:
            # 第一列可能是帧编号
            data = df_data.iloc[:, 1:].to_numpy(dtype=float)

        elif df_data.shape[1] == VALUES_PER_FRAME:
            data = df_data.to_numpy(dtype=float)

        else:
            raise ValueError(
                f"CSV 列数不符合要求。期望 66 个坐标列，"
                f"或者 frame_idx + 66 个坐标列；当前数值列数为 {df_data.shape[1]}"
            )

    if data.shape[1] != VALUES_PER_FRAME:
        raise ValueError(
            f"坐标列数错误，期望 {VALUES_PER_FRAME}，当前为 {data.shape[1]}"
        )

    seq = data.reshape(-1, N_JOINTS, N_DIMS)

    print(f"成功读取 CSV: {csv_path}")
    print(f"骨架数据 shape: {seq.shape}，即 T={seq.shape[0]} 帧, 22 个关节, 3 个坐标轴")

    return seq


# =========================================================
# 4. 计算显示范围
# =========================================================
def compute_axis_limits(seq, pad_ratio=0.05):
    """
    seq shape = (T, 22, 3)

    原始坐标：
        seq[:, :, 0] = x
        seq[:, :, 1] = y
        seq[:, :, 2] = z

    显示坐标：
        matplotlib X 轴 = 原始 x
        matplotlib Y 轴 = 原始 z
        matplotlib Z 轴 = 原始 y
    """

    x_all = seq[:, :, 0]
    y_all = seq[:, :, 1]
    z_all = seq[:, :, 2]

    xmin, xmax = np.nanmin(x_all), np.nanmax(x_all)
    ymin, ymax = np.nanmin(z_all), np.nanmax(z_all)
    zmin, zmax = np.nanmin(y_all), np.nanmax(y_all)

    pad_x = (xmax - xmin) * pad_ratio if xmax > xmin else 1.0
    pad_y = (ymax - ymin) * pad_ratio if ymax > ymin else 1.0
    pad_z = (zmax - zmin) * pad_ratio if zmax > zmin else 1.0

    return (
        xmin - pad_x, xmax + pad_x,
        ymin - pad_y, ymax + pad_y,
        zmin - pad_z, zmax + pad_z,
    )


# =========================================================
# 5. 可视化单个 CSV 文件
# =========================================================
def visualize_skeleton_csv(
    csv_path,
    interval=30,
    stride=1,
    show_joint_id=True,
    view_elev=30,
    view_azim=60,
    save_gif_path=None,
):
    """
    参数：
        csv_path:
            单个骨架 CSV 文件路径。

        interval:
            动画刷新间隔，单位 ms。
            30 ms 大约对应 33 FPS。

        stride:
            帧采样间隔。
            stride=1 表示每帧都显示；
            stride=2 表示每隔 1 帧显示一次。

        show_joint_id:
            是否显示关节编号。

        view_elev, view_azim:
            3D 视角参数，对应 matplotlib 的 ax.view_init。

        save_gif_path:
            如果不为 None，则保存为 GIF。
            例如 r"output.gif"。
    """

    seq = read_skeleton_csv(csv_path)

    if stride < 1:
        raise ValueError("stride 必须 >= 1")

    seq = seq[::stride]
    frames = seq.shape[0]

    (
        xmin, xmax,
        ymin, ymax,
        zmin, zmax,
    ) = compute_axis_limits(seq)

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection="3d")

    # 尽量保持三维比例一致
    ax.set_box_aspect((xmax - xmin, ymax - ymin, zmax - zmin))

    def update(i):
        ax.cla()

        # 原始坐标
        x = seq[i, :, 0]
        y = seq[i, :, 1]
        z = seq[i, :, 2]

        # 显示时保持你原始代码的 x, z, y 顺序
        ax.scatter(x, z, y, c="b", s=30)

        for p1, p2 in BONES:
            ax.plot(
                [x[p1], x[p2]],
                [z[p1], z[p2]],
                [y[p1], y[p2]],
                c="b",
                linewidth=2
            )

        if show_joint_id:
            for j in range(N_JOINTS):
                ax.text(
                    x[j],
                    z[j],
                    y[j] + 0.02,
                    str(j + 1),
                    fontsize=8
                )

        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
        ax.set_zlim(zmin, zmax)

        ax.set_xlabel("X")
        ax.set_ylabel("Z")
        ax.set_zlabel("Y")

        ax.set_title(
            f"{Path(csv_path).name} | Frame {i + 1}/{frames}"
        )

        ax.view_init(elev=view_elev, azim=view_azim)

        return []

    ani = FuncAnimation(
        fig,
        update,
        frames=frames,
        interval=interval,
        repeat=True
    )

    if save_gif_path is not None:
        save_gif_path = Path(save_gif_path)
        save_gif_path.parent.mkdir(parents=True, exist_ok=True)

        print(f"正在保存 GIF: {save_gif_path}")
        ani.save(save_gif_path, writer="pillow", fps=max(1, int(1000 / interval)))
        print("GIF 保存完成。")

    plt.show()


# =========================================================
# 6. 批量选择目录中的 CSV 文件进行可视化
# =========================================================
def list_csv_files(input_dir):
    input_dir = Path(input_dir)

    csv_files = sorted(input_dir.rglob("*.csv"))

    if len(csv_files) == 0:
        raise ValueError(f"在目录下没有找到 CSV 文件: {input_dir}")

    print("\n找到以下 CSV 文件：")
    for i, path in enumerate(csv_files):
        print(f"[{i}] {path}")

    return csv_files


def visualize_csv_by_index(input_dir, index=0, **kwargs):
    csv_files = list_csv_files(input_dir)

    if index < 0 or index >= len(csv_files):
        raise IndexError(f"index 超出范围，应在 0 到 {len(csv_files) - 1} 之间")

    visualize_skeleton_csv(csv_files[index], **kwargs)


# =========================================================
# 7. 使用示例
# =========================================================
if __name__ == "__main__":
    csv_path = r"data\processed_clean\M10\skeleton3d.csv"

    visualize_skeleton_csv(
        csv_path=csv_path,
        interval=30,
        stride=1,
        show_joint_id=True,
        view_elev=30,
        view_azim=60,
        save_gif_path=None,
    )
