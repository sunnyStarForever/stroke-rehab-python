import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter, find_peaks

from feature_extractor import *
from action_segmentation_template_eval import *

def cosine_similarity(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)

    return float(
        np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8)
    )


# =========================================================
# 3. 动作类型与特征规则
# =========================================================
def normalize_action_type(action_type):
    """
    输入支持：
        "M1", "m1", "m01", 1, "1"
    输出：
        "M1"
    """

    s = str(action_type).strip().upper()

    match = re.search(r"M?0?(\d+)", s)

    if match is None:
        raise ValueError(f"无法解析动作类型: {action_type}")

    action_id = f"M{int(match.group(1))}"

    if action_id not in ACTION_FEATURE_CONFIG:
        raise ValueError(f"未知动作类型: {action_id}")

    return action_id


def get_segment_feature(action_id, feature_name=None):
    action_id = normalize_action_type(action_id)

    if feature_name is None:
        feature_name = ACTION_FEATURE_CONFIG[action_id]["default"]

    if feature_name in ACTION_FEATURE_CONFIG[action_id].get("half_count", []):
        count_divisor = 2
    else:
        count_divisor = 1

    return feature_name, count_divisor


# =========================================================
# 4. 峰值转分割区间
# =========================================================

def segments_to_label_sequence(segments, n_frames):
    """
    返回等长分割序列。

    规则：
        第一个周期标记为 1
        第二个周期标记为 2
        第三个周期标记为 3
        ...

    如果 use_full_range=True，则通常所有帧都会被标记。
    如果某些帧没有被任何 segment 覆盖，则为 0。
    """

    labels = np.zeros(n_frames, dtype=int)

    for seg in segments:
        rep_id = int(seg["rep_id"])
        start = int(seg["start"])
        end = int(seg["end"])

        start = max(0, start)
        end = min(n_frames - 1, end)

        labels[start:end + 1] = rep_id

    return labels


# =========================================================
# 6. 模板库加载与相似度计算
# =========================================================
def load_template_bank(template_npz, template_meta_csv=None):
    """
    加载模板库。

    推荐模板库格式：
        action_templates.npz:
            templates: shape = (N, D)
            action_id: shape = (N,)

        action_templates_meta.csv:
            至少包含 action_id；
            最好包含 template_features 字段。
    """

    template_npz = Path(template_npz)

    if not template_npz.exists():
        raise FileNotFoundError(f"模板文件不存在: {template_npz}")

    data = np.load(template_npz, allow_pickle=True)

    if "templates" not in data:
        raise KeyError("template_npz 中必须包含 templates")

    templates = data["templates"]

    if "action_id" in data:
        action_ids = np.asarray(data["action_id"]).astype(str)
    elif template_meta_csv is not None:
        meta_df = pd.read_csv(template_meta_csv)
        action_ids = meta_df["action_id"].astype(str).to_numpy()
    else:
        raise KeyError("模板库中没有 action_id，请提供 template_meta_csv")

    if template_meta_csv is not None:
        meta_df = pd.read_csv(template_meta_csv)
    else:
        meta_df = pd.DataFrame({
            "action_id": action_ids
        })

    return {
        "templates": np.asarray(templates, dtype=float),
        "action_ids": np.asarray(action_ids),
        "meta_df": meta_df,
    }


def get_action_template_bank(template_bank, action_id):
    action_id = normalize_action_type(action_id)

    templates = template_bank["templates"]
    action_ids = np.asarray(template_bank["action_ids"]).astype(str)

    idx = np.where(action_ids == action_id)[0]

    if len(idx) == 0:
        raise ValueError(f"模板库中没有动作 {action_id} 的模板")

    return templates[idx]


def compute_segment_similarity_scores(
    segment_templates,
    reference_templates,
    method="centroid",
):
    """
    计算每个分割片段与模板库的相似度。

    method:
        "centroid":
            先计算该动作所有模板的中心，然后每个片段和中心算 cosine。

        "max":
            每个片段和该动作所有模板分别算 cosine，取最大值。

        "mean":
            每个片段和该动作所有模板分别算 cosine，取平均值。
    """

    segment_templates = np.asarray(segment_templates, dtype=float)
    reference_templates = np.asarray(reference_templates, dtype=float)

    if len(segment_templates) == 0:
        return np.array([], dtype=float)

    if method == "centroid":
        centroid = np.mean(reference_templates, axis=0)
        scores = [
            cosine_similarity(x, centroid)
            for x in segment_templates
        ]

    elif method == "max":
        scores = []

        for x in segment_templates:
            sims = [
                cosine_similarity(x, ref)
                for ref in reference_templates
            ]
            scores.append(float(np.max(sims)))

    elif method == "mean":
        scores = []

        for x in segment_templates:
            sims = [
                cosine_similarity(x, ref)
                for ref in reference_templates
            ]
            scores.append(float(np.mean(sims)))

    else:
        raise ValueError(f"未知相似度计算方法: {method}")

    return np.asarray(scores, dtype=float)


# =========================================================
# 7. 主 API：CSV + 动作类型 -> 分割序列、计数、相似度
# =========================================================
def analyze_action_csv(
    csv_path,
    action_type,
    template_npz,
    template_meta_csv=None,
    segment_feature_name=None,
    template_feature_names=None,
    fs=30,
    min_interval_sec=1.0,
    poly_degree=3,
    smooth_win=11,
    smooth_poly=3,
    prominence_ratio=0.30,
    target_len=100,
    normalize=True,
    smooth_features=True,
    use_full_range=True,
    similarity_method="centroid",
    return_features=False,
):
    """
    统一 API。

    输入：
        csv_path:
            宽表骨架 CSV 路径。

        action_type:
            动作类型，例如 "M1", "M2", ..., "M10"。

        template_npz:
            已经计算好的模板库 npz 文件路径。

        template_meta_csv:
            模板元信息 CSV，可选。

    输出：
        result: dict
            result["segment_labels"]:
                等长分割序列。
                长度 = 原始 CSV 帧数。
                第一个周期标记为 1，第二个周期标记为 2，以此类推。

            result["count"]:
                动作计数。

            result["similarity_score"]:
                整体相似度评分，默认为每个片段评分的均值。

            result["per_segment_scores"]:
                每个动作周期的相似度评分。

            result["segments_df"]:
                每个周期的 start / end / center。
    """

    csv_path = Path(csv_path)
    action_id = normalize_action_type(action_type)

    # -----------------------------------------------------
    # 1. 读取 CSV 并计算特征
    # -----------------------------------------------------
    features_df = compute_features_from_csv(
        csv_path=csv_path,
        normalize=normalize,
        smooth=smooth_features,
        smooth_win=smooth_win,
        smooth_poly=smooth_poly,
        return_dataframe=True,
    )

    n_frames = len(features_df)

    # -----------------------------------------------------
    # 2. 根据动作类型选择分割特征
    # -----------------------------------------------------
    segment_feature_name, count_divisor = get_segment_feature(
        action_id,
        feature_name=segment_feature_name,
    )

    if segment_feature_name not in features_df.columns:
        raise ValueError(
            f"分割特征不存在: {segment_feature_name}，"
            f"当前可用特征为: {list(features_df.columns)}"
        )

    signal = features_df[segment_feature_name].to_numpy(dtype=float)

    min_distance = int(round(fs * min_interval_sec))

    # -----------------------------------------------------
    # 3. 去趋势寻峰
    # -----------------------------------------------------
    peaks, debug = detect_peaks_after_detrend(
        signal,
        min_distance=min_distance,
        poly_degree=poly_degree,
        smooth_win=smooth_win,
        smooth_poly=smooth_poly,
        prominence_ratio=prominence_ratio,
    )

    centers = convert_peaks_to_centers(
        peaks,
        count_divisor=count_divisor,
    )

    segments = build_segments_from_centers(
        centers,
        n_frames=n_frames,
        use_full_range=use_full_range,
        min_distance=min_distance,
    )

    count = len(segments)

    segment_labels = segments_to_label_sequence(
        segments=segments,
        n_frames=n_frames,
    )

    segments_df = pd.DataFrame(segments)

    # -----------------------------------------------------
    # 4. 构建当前输入文件的片段模板
    # -----------------------------------------------------
    if template_feature_names is None:
        template_feature_names = [segment_feature_name]

    segment_templates = []

    for seg in segments:
        temp = segment_to_template(
            features_df=features_df,
            segment=seg,
            feature_names=template_feature_names,
            target_len=target_len,
            normalize_each_feature=True,
        )

        if temp is not None:
            segment_templates.append(temp)

    if len(segment_templates) > 0:
        segment_templates = np.vstack(segment_templates)
    else:
        segment_templates = np.empty((0, target_len * len(template_feature_names)))

    # -----------------------------------------------------
    # 5. 加载模板库并计算相似度
    # -----------------------------------------------------
    template_bank = load_template_bank(
        template_npz=template_npz,
        template_meta_csv=template_meta_csv,
    )

    reference_templates = get_action_template_bank(
        template_bank=template_bank,
        action_id=action_id,
    )

    if segment_templates.shape[1] != reference_templates.shape[1]:
        raise ValueError(
            "当前片段模板维度与模板库维度不一致。\n"
            f"当前片段模板维度: {segment_templates.shape[1]}\n"
            f"模板库维度: {reference_templates.shape[1]}\n"
            "请确保 template_feature_names、target_len 与你计算模板时完全一致。"
        )

    per_segment_scores = compute_segment_similarity_scores(
        segment_templates=segment_templates,
        reference_templates=reference_templates,
        method=similarity_method,
    )

    if len(per_segment_scores) > 0:
        similarity_score = float(np.mean(per_segment_scores))
    else:
        similarity_score = np.nan

    # -----------------------------------------------------
    # 6. 汇总返回
    # -----------------------------------------------------
    result = {
        "csv_path": str(csv_path),
        "action_id": action_id,

        "segment_feature_name": segment_feature_name,
        "template_feature_names": template_feature_names,

        "count": int(count),
        "raw_peak_count": int(len(peaks)),
        "count_divisor": int(count_divisor),

        "segment_labels": segment_labels,
        "segments": segments,
        "segments_df": segments_df,

        "peaks": peaks,
        "centers": centers,

        "similarity_score": similarity_score,
        "per_segment_scores": per_segment_scores,
        "similarity_method": similarity_method,

        "fs": fs,
        "min_distance": min_distance,
        "debug": debug,
    }

    if return_features:
        result["features_df"] = features_df
        result["segment_templates"] = segment_templates

    return result

if __name__ == '__main__':
    result = analyze_action_csv(
        csv_path=r"data\processed\M1\skeleton3d.csv",
        action_type="M1",

        template_npz=r"data\outputs\templates\action_templates.npz",
        template_meta_csv=r"data\outputs\templates\action_templates_meta.csv",

        fs=20,
        min_interval_sec=1.0,

        template_feature_names=ACTION_FEATURE_CONFIG["M1"]["default"],
        target_len=100,

        similarity_method="centroid",
        return_features=False,
    )

    print("动作类型:", result["action_id"])
    print("计数:", result["count"])
    print("整体相似度:", result["similarity_score"])
    print("每个周期相似度:", result["per_segment_scores"])
    print("分割序列长度:", len(result["segment_labels"]))
    print("分割序列:", result["segment_labels"])
    print(result["segments_df"])