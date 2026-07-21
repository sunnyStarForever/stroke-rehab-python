from pathlib import Path
import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from realtime_joint_action_scorer import RealtimeJointActionScorer


# =========================================================
# 1. 关节顺序：必须和实时 API 中一致
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


# =========================================================
# 2. 读取宽表 CSV -> frames, shape=(T, 22, 3)
# =========================================================
def read_wide_skeleton_csv_as_frames(csv_path):
    """
    读取你之前转换好的宽表 CSV。

    支持格式：
        frame_idx,
        00_waist_x,00_waist_y,00_waist_z,
        ...
        21_r_toe_x,21_r_toe_y,21_r_toe_z

    返回：
        frames: np.ndarray, shape=(T, 22, 3)
    """

    csv_path = Path(csv_path)

    if not csv_path.exists():
        raise FileNotFoundError(f"CSV 文件不存在: {csv_path}")

    df = pd.read_csv(csv_path)

    expected_cols = []

    for joint_id, joint_name in JOINTS:
        for axis in ["x", "y", "z"]:
            expected_cols.append(f"{joint_id:02d}_{joint_name}_{axis}")

    if all(col in df.columns for col in expected_cols):
        data = df[expected_cols].to_numpy(dtype=float)

    else:
        # 兼容没有标准列名、但数值列刚好是 66 或 frame_idx+66 的情况
        numeric_df = df.select_dtypes(include=[np.number]).copy()

        if "frame_idx" in numeric_df.columns:
            numeric_df = numeric_df.drop(columns=["frame_idx"])

        if numeric_df.shape[1] != 66:
            raise ValueError(
                f"无法解析 CSV。期望 66 个坐标列，当前数值列数为 {numeric_df.shape[1]}"
            )

        data = numeric_df.to_numpy(dtype=float)

    frames = data.reshape(-1, 22, 3)

    return frames


# =========================================================
# 3. 可选：array frame -> dict frame
#    实际 update() 可以直接接收 np.ndarray，不一定要转 dict。
# =========================================================
def frame_array_to_dict(frame_array):
    """
    将 shape=(22,3) 的单帧数组转换成字典格式。
    """

    frame_dict = {}

    for joint_id, joint_name in JOINTS:
        frame_dict[joint_name] = frame_array[joint_id].tolist()

    return frame_dict


# =========================================================
# 4. 基于 CSV 模拟实时逐帧输入
# =========================================================
def simulate_realtime_from_csv(
    csv_path,
    action_type,
    fs=30,
    use_dict_input=False,
    realtime_sleep=False,
    print_each_frame=False,
    save_event_log_path=None,
):
    """
    用已有 CSV 模拟实时逐帧输入。

    参数：
        csv_path:
            你的宽表骨架 CSV 文件路径。

        action_type:
            动作类型，例如 "M1", "M2", ..., "M10"。

        fs:
            帧率。你的数据如果是 30 Hz，则 fs=30。

        use_dict_input:
            False:
                每帧以 np.ndarray, shape=(22,3) 输入。
            True:
                每帧转换成 dict 输入。

        realtime_sleep:
            True:
                每帧 sleep 1/fs 秒，模拟真实实时速度。
            False:
                不 sleep，快速跑完整个 CSV。

        print_each_frame:
            是否打印每一帧状态。
            一般建议 False，否则输出很多。

        save_event_log_path:
            如果不为 None，保存每次识别到完整周期时的事件日志。
    """

    frames = read_wide_skeleton_csv_as_frames(csv_path)

    analyzer = RealtimeJointActionScorer(
        action_type=action_type,
        fs=fs,

        # 两个峰之间至少间隔 1 秒
        min_interval_sec=1.0,

        # 峰出现后等待 0.5 秒再确认，避免峰位置不稳定
        peak_confirm_sec=0.5,

        # 至少积累 3 秒数据再开始检测
        min_frames_before_detection=3 * fs,

        poly_degree=3,
        smooth_win=11,
        smooth_poly=3,
        prominence_ratio=0.30,

        normalize=True,
        smooth_features=True,
    )

    event_rows = []
    count_trace = []
    completed_count_trace = []

    print("========== Start realtime simulation ==========")
    print(f"CSV path     : {csv_path}")
    print(f"Action type  : {action_type}")
    print(f"Frames       : {len(frames)}")
    print(f"FS           : {fs} Hz")
    print(f"Input format : {'dict' if use_dict_input else 'np.ndarray'}")
    print("================================================\n")

    for i, frame_array in enumerate(frames):
        if use_dict_input:
            frame_input = frame_array_to_dict(frame_array)
        else:
            frame_input = frame_array

        result = analyzer.update(frame_input)

        count_trace.append(result["count"])
        completed_count_trace.append(result["completed_count"])

        if print_each_frame:
            print(
                f"frame={i:04d} | "
                f"status={result['status']} | "
                f"count={result['count']} | "
                f"completed={result['completed_count']}"
            )

        # 只有新完整周期出现时，才会返回最后一个周期评分
        if result["status"] == "new_completed_cycle":
            cycle = result["last_cycle"]

            print("\n[NEW COMPLETED CYCLE]")
            print(f"frame_index      : {result['frame_index']}")
            print(f"rep_id           : {cycle['rep_id']}")
            print(f"segment          : {cycle['start']} -> {cycle['end']}")
            print(f"center           : {cycle['center']}")
            print(f"total_count      : {result['count']}")
            print(f"completed_count  : {result['completed_count']}")
            print(f"overall_score    : {cycle['overall_score']:.2f}")
            print("dimension_scores :")
            for k, v in cycle["dimension_scores"].items():
                print(f"  {k}: {v:.2f}")

            row = {
                "trigger_frame": result["frame_index"],
                "rep_id": cycle["rep_id"],
                "start": cycle["start"],
                "end": cycle["end"],
                "center": cycle["center"],
                "duration_frames": cycle["duration_frames"],
                "count": result["count"],
                "completed_count": result["completed_count"],
                "overall_score": cycle["overall_score"],
                "amplitude_score": cycle["dimension_scores"]["amplitude_score"],
                "smoothness_score": cycle["dimension_scores"]["smoothness_score"],
                "trunk_score": cycle["dimension_scores"]["trunk_score"],
                "symmetry_score": cycle["dimension_scores"]["symmetry_score"],
                "rhythm_score": cycle["dimension_scores"]["rhythm_score"],
            }

            # 原始指标也保存，便于调试
            for k, v in cycle["raw_metrics"].items():
                if isinstance(v, (int, float, np.floating)):
                    row[k] = float(v)
                else:
                    row[k] = str(v)

            event_rows.append(row)

        if realtime_sleep:
            time.sleep(1.0 / fs)

    summary = analyzer.get_summary()

    print("\n========== Simulation finished ==========")
    print(f"Total frames       : {len(frames)}")
    print(f"Final count        : {summary['count']}")
    print(f"Completed scored   : {summary['completed_count']}")
    print(f"Accepted peaks     : {summary['accepted_peaks']}")
    print(f"Accepted centers   : {summary['accepted_centers']}")
    print("=========================================\n")

    event_df = pd.DataFrame(event_rows)

    if save_event_log_path is not None:
        save_event_log_path = Path(save_event_log_path)
        save_event_log_path.parent.mkdir(parents=True, exist_ok=True)
        event_df.to_csv(save_event_log_path, index=False, encoding="utf-8-sig")
        print(f"事件日志已保存: {save_event_log_path}")

    output = {
        "event_df": event_df,
        "summary": summary,
        "count_trace": np.asarray(count_trace, dtype=int),
        "completed_count_trace": np.asarray(completed_count_trace, dtype=int),
        "analyzer": analyzer,
        "frames": frames,
    }

    return output


# =========================================================
# 5. 可视化模拟过程中的计数变化
# =========================================================
def plot_realtime_simulation_trace(sim_result):
    count_trace = sim_result["count_trace"]
    completed_count_trace = sim_result["completed_count_trace"]
    event_df = sim_result["event_df"]

    x = np.arange(len(count_trace))

    plt.figure(figsize=(14, 5))

    plt.plot(x, count_trace, linewidth=2, label="real-time count")
    plt.plot(x, completed_count_trace, linewidth=2, label="completed scored count")

    if not event_df.empty:
        for _, row in event_df.iterrows():
            plt.axvline(
                int(row["trigger_frame"]),
                linestyle="--",
                alpha=0.5
            )
            plt.text(
                int(row["trigger_frame"]),
                row["completed_count"],
                f"rep {int(row['rep_id'])}",
                ha="center",
                va="bottom",
                fontsize=8
            )

    plt.xlabel("Frame")
    plt.ylabel("Count")
    plt.title("Realtime Simulation Count Trace")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()


# =========================================================
# 6. 使用示例
# =========================================================
if __name__ == "__main__":

    csv_path = r"data\examples\skeleton_wide.csv"
    action_type = "M1"

    sim_result = simulate_realtime_from_csv(
        csv_path=csv_path,
        action_type=action_type,
        fs=30,

        # False 表示直接传入 shape=(22,3) 的数组
        # True 表示转换成 dict 再传入
        use_dict_input=False,

        # False 表示快速模拟
        # True 表示按 30Hz 真实速度模拟
        realtime_sleep=False,

        print_each_frame=False,

        save_event_log_path=r"outputs\realtime_simulation\event_log.csv",
    )

    print(sim_result["event_df"])

    plot_realtime_simulation_trace(sim_result)