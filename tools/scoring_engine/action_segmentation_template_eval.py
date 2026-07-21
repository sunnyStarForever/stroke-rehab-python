import os
import re
import sys
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scipy.signal import find_peaks

from feature_extractor import *
from action_config import (
    ACTION_FEATURE_CONFIG,
    normalize_action_type,
    get_feature_rule,
    get_segment_feature,
    get_template_features_for_action,
)


# =========================================================
# 2. 基础信号处理
# =========================================================
def fit_polynomial_baseline(signal, degree=3):
    signal = np.asarray(signal, dtype=float)
    n = len(signal)

    if n <= degree + 1:
        return np.zeros_like(signal)

    x = np.arange(n, dtype=float)
    coeffs = np.polyfit(x, signal, deg=degree)
    baseline = np.polyval(coeffs, x)

    return baseline


def detrend_signal_poly(signal, degree=3):
    signal = np.asarray(signal, dtype=float)
    baseline = fit_polynomial_baseline(signal, degree=degree)
    detrended = signal - baseline

    return detrended, baseline


def detect_peaks_after_detrend(
    signal,
    min_distance=20,
    poly_degree=3,
    smooth_win=11,
    smooth_poly=3,
    prominence_ratio=0.30,
):
    signal = np.asarray(signal, dtype=float)

    detrended, baseline = detrend_signal_poly(signal, degree=poly_degree)
    detrended_smooth = smooth_signal(detrended, win=smooth_win, poly=smooth_poly)

    dynamic_range = np.max(detrended_smooth) - np.min(detrended_smooth)
    prominence = max(prominence_ratio * dynamic_range, 1e-6)

    peaks, properties = find_peaks(
        detrended_smooth,
        distance=min_distance,
        prominence=prominence
    )

    debug = {
        "baseline": baseline,
        "detrended": detrended,
        "detrended_smooth": detrended_smooth,
        "prominence": prominence,
        "properties": properties,
    }

    return peaks, debug


# =========================================================
# 3. 动作编号、被试编号解析
# =========================================================
def parse_action_id(file_path):
    stem = Path(file_path).stem
    m = re.search(r"[mM]0?(\d+)", stem)

    if m is None:
        raise ValueError(f"无法从文件名中解析动作编号: {stem}")

    return f"M{int(m.group(1))}"


def parse_subject_id(file_path):
    stem = Path(file_path).stem
    m = re.search(r"[sS]0?(\d+)", stem)

    if m is None:
        return None

    return f"S{int(m.group(1))}"


# =========================================================
# 4. 峰值 -> 动作中心 -> 分割区间
# =========================================================
def convert_peaks_to_centers(peaks, count_divisor=1):
    peaks = np.asarray(peaks, dtype=int)

    if count_divisor <= 1:
        return peaks

    usable_num = (len(peaks) // count_divisor) * count_divisor

    if usable_num == 0:
        return np.array([], dtype=int)

    peaks_used = peaks[:usable_num]
    grouped = peaks_used.reshape(-1, count_divisor)

    centers = np.round(np.mean(grouped, axis=1)).astype(int)

    return centers


def build_segments_from_centers(
    centers,
    n_frames,
    use_full_range=True,
    min_distance=30,
):
    centers = np.asarray(centers, dtype=int)
    centers = np.sort(centers)

    if len(centers) == 0:
        return []

    if len(centers) == 1:
        c = int(centers[0])

        if use_full_range:
            return [{
                "rep_id": 1,
                "start": 0,
                "end": n_frames - 1,
                "center": c,
            }]

        half_len = max(1, min_distance // 2)

        return [{
            "rep_id": 1,
            "start": max(0, c - half_len),
            "end": min(n_frames - 1, c + half_len),
            "center": c,
        }]

    midpoints = ((centers[:-1] + centers[1:]) / 2.0).astype(int)

    if use_full_range:
        boundaries = np.concatenate([
            np.array([0]),
            midpoints,
            np.array([n_frames - 1])
        ])
    else:
        first_gap = centers[1] - centers[0]
        last_gap = centers[-1] - centers[-2]

        start0 = max(0, centers[0] - first_gap // 2)
        end_last = min(n_frames - 1, centers[-1] + last_gap // 2)

        boundaries = np.concatenate([
            np.array([start0]),
            midpoints,
            np.array([end_last])
        ])

    segments = []

    for i, c in enumerate(centers):
        segments.append({
            "rep_id": i + 1,
            "start": int(boundaries[i]),
            "end": int(boundaries[i + 1]),
            "center": int(c),
        })

    return segments


# =========================================================
# 5. 单文件分割计数封装
# =========================================================
def segment_count_from_features(
    features_df,
    action_id,
    feature_name=None,
    fs=30,
    min_interval_sec=1.0,
    poly_degree=3,
    smooth_win=11,
    smooth_poly=3,
    prominence_ratio=0.30,
    use_full_range=True,
):
    action_id = action_id.upper()

    rule = get_feature_rule(
        action_id=action_id,
        feature_name=feature_name
    )

    feature_name = rule["feature_name"]
    count_divisor = rule["count_divisor"]

    if feature_name not in features_df.columns:
        raise ValueError(
            f"特征 {feature_name} 不存在。当前可用特征为: {list(features_df.columns)}"
        )

    signal = features_df[feature_name].to_numpy(dtype=float)
    n_frames = len(signal)

    min_distance = int(round(fs * min_interval_sec))

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
        count_divisor=count_divisor
    )

    segments = build_segments_from_centers(
        centers,
        n_frames=n_frames,
        use_full_range=use_full_range,
        min_distance=min_distance,
    )

    result = {
        "action_id": action_id,
        "feature_name": feature_name,
        "feature_priority": rule["priority"],
        "count_divisor": count_divisor,
        "fs": fs,
        "min_distance": min_distance,
        "raw_peak_count": int(len(peaks)),
        "count": int(len(centers)),
        "peaks": peaks,
        "centers": centers,
        "segments": segments,
        "segments_df": pd.DataFrame(segments),
        "debug": debug,
    }

    return result


def segment_count_from_csv(
    csv_path,
    action_id,
    feature_name=None,
    fs=30,
    min_interval_sec=1.0,
    poly_degree=3,
    smooth_win=11,
    smooth_poly=3,
    prominence_ratio=0.30,
    normalize=True,
    smooth_features=True,
    use_full_range=True,
    return_features=True,
):
    csv_path = Path(csv_path)

    features_df = compute_features_from_csv(
        csv_path=csv_path,
        normalize=normalize,
        smooth=smooth_features,
        smooth_win=smooth_win,
        smooth_poly=smooth_poly,
        return_dataframe=True,
    )

    seg_result = segment_count_from_features(
        features_df=features_df,
        action_id=action_id,
        feature_name=feature_name,
        fs=fs,
        min_interval_sec=min_interval_sec,
        poly_degree=poly_degree,
        smooth_win=smooth_win,
        smooth_poly=smooth_poly,
        prominence_ratio=prominence_ratio,
        use_full_range=use_full_range,
    )

    seg_result["csv_path"] = str(csv_path)
    seg_result["file_stem"] = csv_path.stem

    if return_features:
        seg_result["features_df"] = features_df

    return seg_result


# =========================================================
# 6. 分割可视化
# =========================================================
def visualize_segmentation_result(
    seg_result,
    save_path=None,
    show=True,
):
    features_df = seg_result.get("features_df", None)

    if features_df is None:
        raise ValueError("seg_result 中没有 features_df，请设置 return_features=True")

    feature_name = seg_result["feature_name"]
    action_id = seg_result["action_id"]

    signal = features_df[feature_name].to_numpy(dtype=float)
    x = np.arange(len(signal))

    peaks = seg_result["peaks"]
    centers = seg_result["centers"]
    segments = seg_result["segments"]
    debug = seg_result["debug"]

    baseline = debug["baseline"]
    detrended_smooth = debug["detrended_smooth"]

    fig, axes = plt.subplots(
        2,
        1,
        figsize=(16, 8),
        sharex=True,
        gridspec_kw={"height_ratios": [1.1, 1.0]}
    )

    ax1, ax2 = axes

    ax1.plot(x, signal, linewidth=2, label=f"feature: {feature_name}")
    ax1.plot(x, baseline, "--", linewidth=1.5, label="baseline")

    if len(peaks) > 0:
        ax1.scatter(
            peaks,
            signal[peaks],
            s=45,
            color="red",
            zorder=3,
            label=f"peaks={len(peaks)}"
        )

    for seg in segments:
        ax1.axvspan(seg["start"], seg["end"], alpha=0.08)
        ax1.axvline(seg["start"], color="gray", linestyle=":", linewidth=1)
        ax1.text(
            seg["center"],
            np.nanmax(signal),
            str(seg["rep_id"]),
            ha="center",
            va="top",
            fontsize=9
        )

    ax1.set_title(
        f"{seg_result.get('file_stem', '')} | {action_id} | "
        f"{feature_name} | count={seg_result['count']}"
    )
    ax1.set_ylabel("Feature value")
    ax1.grid(alpha=0.3)
    ax1.legend(loc="upper right")

    ax2.plot(x, detrended_smooth, linewidth=2, label="detrended + smoothed")

    if len(peaks) > 0:
        ax2.scatter(
            peaks,
            detrended_smooth[peaks],
            s=45,
            color="red",
            zorder=3,
            label="peaks"
        )

    for c in centers:
        ax2.axvline(c, color="purple", linestyle="--", linewidth=1.2)

    for seg in segments:
        ax2.axvspan(seg["start"], seg["end"], alpha=0.08)

    ax2.axhline(0, color="black", linewidth=1, alpha=0.5)
    ax2.set_xlabel("Frame")
    ax2.set_ylabel("Detrended value")
    ax2.grid(alpha=0.3)
    ax2.legend(loc="upper right")

    plt.tight_layout()

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"[INFO] saved figure: {save_path}", file=sys.stderr, flush=True)

    if show:
        plt.show()
    else:
        plt.close(fig)


# =========================================================
# 7. 模板构建：每个分割片段 -> 一个固定长度模板
# =========================================================
def zscore_1d(x):
    x = np.asarray(x, dtype=float)
    std = np.std(x)

    if std < 1e-8:
        return np.zeros_like(x)

    return (x - np.mean(x)) / std


def resample_1d(x, target_len=100):
    x = np.asarray(x, dtype=float)

    if len(x) <= 1:
        return np.zeros(target_len, dtype=float)

    old_idx = np.linspace(0, 1, len(x))
    new_idx = np.linspace(0, 1, target_len)

    return np.interp(new_idx, old_idx, x)


def segment_to_template(
    features_df,
    segment,
    feature_names,
    target_len=100,
    normalize_each_feature=True,
):
    """
    将一个分割片段转换成模板向量。

    输出：
        template_vector.shape = (target_len * num_features,)
    """

    if isinstance(feature_names, str):
        feature_names = [feature_names]

    start = int(segment["start"])
    end = int(segment["end"])

    if end <= start:
        raise ValueError(f"非法分割区间: start={start}, end={end}")

    pieces = []

    for feat in feature_names:
        # print(features_df.columns)
        if feat not in features_df.columns:
            raise ValueError(f"特征不存在: {feat}")

        sig = features_df[feat].to_numpy(dtype=float)
        seg_sig = sig[start:end + 1]

        if normalize_each_feature:
            seg_sig = zscore_1d(seg_sig)

        seg_resampled = resample_1d(seg_sig, target_len=target_len)
        pieces.append(seg_resampled)

    template = np.concatenate(pieces, axis=0)

    return template


def compute_templates_from_csv(
    csv_path,
    action_id,
    segment_feature_name=None,
    template_feature_names=None,
    template_feature_mode="default",
    target_len=100,
    fs=30,
    min_interval_sec=1.0,
    poly_degree=3,
    smooth_win=11,
    smooth_poly=3,
    prominence_ratio=0.30,
    normalize=True,
    smooth_features=True,
    use_full_range=True,
):
    """
    单个 CSV 文件：
        1. 提取特征；
        2. 分割计数；
        3. 将每个分割片段转换为模板向量。
    """

    csv_path = Path(csv_path)

    seg_result = segment_count_from_csv(
        csv_path=csv_path,
        action_id=action_id,
        feature_name=segment_feature_name,
        fs=fs,
        min_interval_sec=min_interval_sec,
        poly_degree=poly_degree,
        smooth_win=smooth_win,
        smooth_poly=smooth_poly,
        prominence_ratio=prominence_ratio,
        normalize=normalize,
        smooth_features=smooth_features,
        use_full_range=use_full_range,
        return_features=True,
    )

    features_df = seg_result["features_df"]

    if template_feature_names is None:
        template_feature_names = get_template_features_for_action(
            action_id,
            mode=template_feature_mode
        )

    templates = []
    meta_rows = []

    subject_id = parse_subject_id(csv_path)

    for seg in seg_result["segments"]:
        template = segment_to_template(
            features_df=features_df,
            segment=seg,
            feature_names=template_feature_names,
            target_len=target_len,
            normalize_each_feature=True,
        )

        templates.append(template)

        meta_rows.append({
            "csv_path": str(csv_path),
            "file_stem": csv_path.stem,
            "action_id": action_id.upper(),
            "subject_id": subject_id,
            "rep_id": seg["rep_id"],
            "start": seg["start"],
            "end": seg["end"],
            "center": seg["center"],
            "segment_feature": seg_result["feature_name"],
            "template_features": ",".join(template_feature_names),
            "template_len": len(template),
        })

    if len(templates) == 0:
        template_array = np.empty((0, target_len * len(template_feature_names)))
    else:
        template_array = np.vstack(templates)

    meta_df = pd.DataFrame(meta_rows)

    output = {
        "templates": template_array,
        "meta_df": meta_df,
        "seg_result": seg_result,
        "template_feature_names": template_feature_names,
    }

    return output


# =========================================================
# 8. 批量计算 global_positions_csv 下所有动作模板
# =========================================================
def compute_templates_for_directory(
    input_dir,
    pattern="*.csv",
    action_id_mode="filename",
    manual_action_id=None,
    output_npz=None,
    output_meta_csv=None,
    **kwargs
):
    """
    针对 global_positions_csv 下所有 CSV 计算模板。

    action_id_mode:
        "filename":
            从文件名 m01/m02/... 自动解析动作编号。

        "manual":
            所有文件都使用 manual_action_id。
    """

    input_dir = Path(input_dir)
    csv_files = sorted(input_dir.rglob(pattern))

    if len(csv_files) == 0:
        raise ValueError(f"没有找到 CSV 文件: {input_dir}")

    all_templates = []
    all_meta = []

    for i, csv_path in enumerate(csv_files, start=1):
        print(f"\n[{i}/{len(csv_files)}] processing: {csv_path}", file=sys.stderr, flush=True)

        try:
            if action_id_mode == "filename":
                action_id = parse_action_id(csv_path)
            elif action_id_mode == "manual":
                if manual_action_id is None:
                    raise ValueError("action_id_mode='manual' 时必须传入 manual_action_id")
                action_id = manual_action_id.upper()
            else:
                raise ValueError(f"未知 action_id_mode: {action_id_mode}")

            out = compute_templates_from_csv(
                csv_path=csv_path,
                action_id=action_id,
                **kwargs
            )

            templates = out["templates"]
            meta_df = out["meta_df"]

            if len(templates) > 0:
                all_templates.append(templates)
                all_meta.append(meta_df)

            print(
                f"[OK] {csv_path.name} | {action_id} | "
                f"segments={len(meta_df)}",
                file=sys.stderr,
                flush=True,
            )

        except Exception as e:
            print(f"[ERROR] {csv_path}: {e}", file=sys.stderr, flush=True)

    if len(all_templates) == 0:
        raise RuntimeError("没有成功生成任何模板。")

    templates_all = np.vstack(all_templates)
    meta_all = pd.concat(all_meta, ignore_index=True)

    if output_npz is not None:
        output_npz = Path(output_npz)
        output_npz.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            output_npz,
            templates=templates_all,
            action_id=meta_all["action_id"].to_numpy(),
            subject_id=meta_all["subject_id"].fillna("").to_numpy(),
            file_stem=meta_all["file_stem"].to_numpy(),
            rep_id=meta_all["rep_id"].to_numpy(),
        )
        print(f"[INFO] saved templates: {output_npz}", file=sys.stderr, flush=True)

    if output_meta_csv is not None:
        output_meta_csv = Path(output_meta_csv)
        output_meta_csv.parent.mkdir(parents=True, exist_ok=True)
        meta_all.to_csv(output_meta_csv, index=False, encoding="utf-8-sig")
        print(f"[INFO] saved metadata: {output_meta_csv}", file=sys.stderr, flush=True)

    return templates_all, meta_all


# =========================================================
# 9. 模板相似度与效果评估
# =========================================================
def cosine_similarity_matrix(X):
    X = np.asarray(X, dtype=float)

    norm = np.linalg.norm(X, axis=1, keepdims=True) + 1e-8
    Xn = X / norm

    return Xn @ Xn.T


def pearson_similarity_matrix(X):
    X = np.asarray(X, dtype=float)
    Xc = X - np.mean(X, axis=1, keepdims=True)

    norm = np.linalg.norm(Xc, axis=1, keepdims=True) + 1e-8
    Xn = Xc / norm

    return Xn @ Xn.T


def compute_action_centroids(templates, labels):
    templates = np.asarray(templates, dtype=float)
    labels = np.asarray(labels)

    centroids = {}

    for lab in sorted(np.unique(labels)):
        idx = np.where(labels == lab)[0]
        centroids[lab] = np.mean(templates[idx], axis=0)

    return centroids


def nearest_centroid_predict(
    templates,
    labels,
    metric="cosine",
    leave_one_out=True,
):
    templates = np.asarray(templates, dtype=float)
    labels = np.asarray(labels)

    preds = []

    for i in range(len(templates)):
        if leave_one_out:
            train_idx = np.arange(len(templates)) != i
        else:
            train_idx = np.ones(len(templates), dtype=bool)

        train_X = templates[train_idx]
        train_y = labels[train_idx]

        centroids = compute_action_centroids(train_X, train_y)

        best_label = None
        best_score = -np.inf

        for lab, cen in centroids.items():
            a = templates[i]
            b = cen

            if metric == "cosine":
                score = np.dot(a, b) / (
                    np.linalg.norm(a) * np.linalg.norm(b) + 1e-8
                )
            elif metric == "pearson":
                aa = a - np.mean(a)
                bb = b - np.mean(b)
                score = np.dot(aa, bb) / (
                    np.linalg.norm(aa) * np.linalg.norm(bb) + 1e-8
                )
            else:
                raise ValueError(f"未知 metric: {metric}")

            if score > best_score:
                best_score = score
                best_label = lab

        preds.append(best_label)

    return np.asarray(preds)


def evaluate_template_similarity(
    templates,
    meta_df,
    label_col="action_id",
    metric="cosine",
):
    """
    对模板效果进行评估。

    输出：
        1. 相似度矩阵；
        2. 同类平均相似度；
        3. 异类平均相似度；
        4. 同异类相似度间隔；
        5. 最近中心分类准确率。
    """

    labels = meta_df[label_col].to_numpy()

    if metric == "cosine":
        sim_mat = cosine_similarity_matrix(templates)
    elif metric == "pearson":
        sim_mat = pearson_similarity_matrix(templates)
    else:
        raise ValueError(f"未知 metric: {metric}")

    n = len(labels)

    same_vals = []
    diff_vals = []

    for i in range(n):
        for j in range(i + 1, n):
            if labels[i] == labels[j]:
                same_vals.append(sim_mat[i, j])
            else:
                diff_vals.append(sim_mat[i, j])

    same_vals = np.asarray(same_vals, dtype=float)
    diff_vals = np.asarray(diff_vals, dtype=float)

    preds = nearest_centroid_predict(
        templates=templates,
        labels=labels,
        metric=metric,
        leave_one_out=True,
    )

    acc = float(np.mean(preds == labels))

    summary = {
        "metric": metric,
        "num_templates": int(n),
        "num_classes": int(len(np.unique(labels))),
        "same_mean": float(np.mean(same_vals)) if len(same_vals) else np.nan,
        "same_std": float(np.std(same_vals)) if len(same_vals) else np.nan,
        "diff_mean": float(np.mean(diff_vals)) if len(diff_vals) else np.nan,
        "diff_std": float(np.std(diff_vals)) if len(diff_vals) else np.nan,
        "margin": (
            float(np.mean(same_vals) - np.mean(diff_vals))
            if len(same_vals) and len(diff_vals)
            else np.nan
        ),
        "nearest_centroid_acc": acc,
    }

    pred_df = meta_df.copy()
    pred_df["pred_label"] = preds
    pred_df["correct"] = pred_df[label_col] == pred_df["pred_label"]

    return {
        "similarity_matrix": sim_mat,
        "summary": summary,
        "prediction_df": pred_df,
    }


# =========================================================
# 10. 使用示例
# =========================================================
if __name__ == "__main__":

    input_dir = r"data\global_positions_csv"

    templates, meta_df = compute_templates_for_directory(
        input_dir=input_dir,
        pattern="*.csv",
        action_id_mode="filename",

        # 分割参数
        fs=30,
        min_interval_sec=1.0,
        poly_degree=3,
        smooth_win=11,
        smooth_poly=3,
        prominence_ratio=0.30,
        use_full_range=True,

        # 模板参数
        template_feature_mode="default",
        target_len=100,

        # 特征提取参数
        normalize=True,
        smooth_features=True,

        # 可选保存
        output_npz=r"outputs\templates\action_templates.npz",
        output_meta_csv=r"outputs\templates\action_templates_meta.csv",
    )

    eval_result = evaluate_template_similarity(
        templates=templates,
        meta_df=meta_df,
        label_col="action_id",
        metric="cosine",
    )

    print("\n========== Similarity Evaluation ==========", file=sys.stderr, flush=True)
    print(pd.Series(eval_result["summary"]), file=sys.stderr, flush=True)

    eval_result["prediction_df"].to_csv(
        r"outputs\templates\template_prediction_result.csv",
        index=False,
        encoding="utf-8-sig"
    )

    sim_df = pd.DataFrame(eval_result["similarity_matrix"])
    sim_df.to_csv(
        r"outputs\templates\template_similarity_matrix.csv",
        index=False,
        encoding="utf-8-sig"
    )
