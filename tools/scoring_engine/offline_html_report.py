# offline_html_report.py

import os
import sys
from pathlib import Path
import base64
import html
import json
from datetime import datetime

os.environ.setdefault("MPLBACKEND", "Agg")

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm


def setup_chinese_font(verbose=True):
    """
    配置 Matplotlib 中文字体，解决图像中中文乱码或方块问题。

    优先使用 Windows 常见中文字体：
        Microsoft YaHei
        SimHei
        SimSun

    也兼容 Linux / macOS 常见中文字体。
    """

    candidate_font_files = [
        r"C:\Windows\Fonts\msyh.ttc",
        r"C:\Windows\Fonts\msyhbd.ttc",
        r"C:\Windows\Fonts\simhei.ttf",
        r"C:\Windows\Fonts\simsun.ttc",
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
        "/System/Library/Fonts/PingFang.ttc",
    ]

    candidate_fonts = []
    for font_file in candidate_font_files:
        path = Path(font_file)
        if not path.exists():
            continue
        try:
            fm.fontManager.addfont(str(path))
            candidate_fonts.append(fm.FontProperties(fname=str(path)).get_name())
        except Exception:
            pass

    candidate_fonts.extend([
        "Microsoft YaHei",
        "SimHei",
        "SimSun",
        "KaiTi",
        "FangSong",
        "Noto Sans CJK SC",
        "Noto Sans CJK JP",
        "Source Han Sans SC",
        "Source Han Serif SC",
        "WenQuanYi Micro Hei",
        "Arial Unicode MS",
        "PingFang SC",
        "Heiti SC",
    ])

    available_font_names = set()

    for font in fm.fontManager.ttflist:
        available_font_names.add(font.name)

    selected_font = None

    for font_name in candidate_fonts:
        if font_name in available_font_names:
            selected_font = font_name
            break

    if selected_font is None:
        if verbose:
            print(
                "[WARN] 没有找到常见中文字体。"
                "如果图像中文仍显示为方块，请在系统中安装 Microsoft YaHei、SimHei 或 Noto Sans CJK SC。",
                file=sys.stderr,
                flush=True,
            )
        selected_font = "DejaVu Sans"
    else:
        if verbose:
            print(
                f"[INFO] Matplotlib 中文字体使用: {selected_font}",
                file=sys.stderr,
                flush=True,
            )

    matplotlib.rcParams["font.sans-serif"] = [
        selected_font,
        "Microsoft YaHei",
        "SimHei",
        "SimSun",
        "Noto Sans CJK SC",
        "WenQuanYi Micro Hei",
        "Arial Unicode MS",
        "DejaVu Sans",
    ]

    matplotlib.rcParams["font.family"] = "sans-serif"

    # 解决负号显示为方块的问题
    matplotlib.rcParams["axes.unicode_minus"] = False

    # 保存矢量图时尽量保留文字兼容性
    matplotlib.rcParams["pdf.fonttype"] = 42
    matplotlib.rcParams["ps.fonttype"] = 42

    return selected_font


# =========================================================
# 1. 导入你已有的离线 API
# =========================================================
from apis import analyze_action_csv
from rom_quality_report import generate_report_from_action_api_result

try:
    from action_config import ACTION_FEATURE_CONFIG
except Exception:
    ACTION_FEATURE_CONFIG = {}


# =========================================================
# 2. 中文显示名称
# =========================================================
DIMENSION_LABELS = {
    "amplitude_score": "动作幅度",
    "smoothness_score": "速度平滑性",
    "trunk_score": "躯干稳定",
    "symmetry_score": "左右对称性",
    "rhythm_score": "节奏性",
    "similarity_score_100": "模板相似度",
    "overall_quality_score": "综合质量评分",
    "overall_with_similarity_score": "融合综合评分",
}

ROM_LABELS = {
    "knee_flexion_mean": "膝屈伸角均值",
    "hip_flexion_mean": "髋屈伸角均值",
    "shoulder_abduction_mean": "肩外展角均值",
    "shoulder_flexion_mean": "肩前屈角均值",
    "elbow_flexion_mean": "肘屈伸角均值",
    "trunk_sagittal_lean": "躯干前后倾",
    "trunk_frontal_lean": "躯干左右倾",
}

EMG_STATE_LABELS = {
    "REST": "放松/静息",
    "SMOOTH_FLEX": "平稳发力",
    "TREMOR": "震颤/不稳定",
    "FATIGUE": "疲劳倾向",
}

EMG_FEATURE_LABELS = {
    "rms": "肌电强度（RMS）",
    "zcr": "过零率（收缩切换活跃度，ZCR）",
    "cv": "波动系数（稳定性，CV）",
    "fatigue_index": "疲劳指数",
}

EMG_RAW_FEATURE_LABELS = {
    "mav": "平均绝对肌电值（MAV）",
    "iemg": "积分肌电值（IEMG）",
    "wl": "波形长度（WL）",
    "zc": "过零次数（ZC）",
}


# =========================================================
# 3. 通用工具函数
# =========================================================
def _ensure_dir(path):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def _safe_float(x, default=np.nan):
    try:
        if x is None:
            return default
        val = float(x)
        if not np.isfinite(val):
            return default
        return val
    except Exception:
        return default


def _score_text(score):
    score = _safe_float(score)

    if not np.isfinite(score):
        return "无有效评分"

    if score >= 85:
        return "优秀"
    if score >= 70:
        return "良好"
    if score >= 55:
        return "一般"
    return "需关注"


def _score_class(score):
    score = _safe_float(score)

    if not np.isfinite(score):
        return "score-na"

    if score >= 85:
        return "score-good"
    if score >= 70:
        return "score-ok"
    if score >= 55:
        return "score-mid"
    return "score-low"


def _format_num(x, ndigits=2, default="-"):
    x = _safe_float(x)

    if not np.isfinite(x):
        return default

    return f"{x:.{ndigits}f}"


def _to_html_table(df, max_rows=20, index=False, float_format="{:.2f}"):
    if df is None or len(df) == 0:
        return '<div class="empty-box">暂无数据</div>'

    show_df = df.copy()

    if len(show_df) > max_rows:
        show_df = show_df.head(max_rows)

    return show_df.to_html(
        index=index,
        border=0,
        classes="data-table",
        escape=False,
        float_format=lambda x: float_format.format(x) if np.isfinite(x) else "-",
    )


def _image_to_base64(path):
    path = Path(path)

    if not path.exists():
        return None

    suffix = path.suffix.lower().replace(".", "")

    if suffix == "jpg":
        suffix = "jpeg"

    with path.open("rb") as f:
        data = base64.b64encode(f.read()).decode("utf-8")

    return f"data:image/{suffix};base64,{data}"


def _img_tag(path, title=None, embed=True):
    path = Path(path)

    if not path.exists():
        return ""

    if embed:
        src = _image_to_base64(path)
    else:
        src = path.resolve().as_uri()

    if src is None:
        return ""

    title_html = f"<div class='fig-title'>{html.escape(title)}</div>" if title else ""

    return f"""
    <div class="figure-card">
        {title_html}
        <img src="{src}" alt="{html.escape(title or path.name)}" width="980"/>
    </div>
    """


def _normalize_template_feature_names(template_feature_names, action_type):
    if template_feature_names is None:
        if ACTION_FEATURE_CONFIG and action_type in ACTION_FEATURE_CONFIG:
            template_feature_names = ACTION_FEATURE_CONFIG[action_type]["default"]
        else:
            template_feature_names = None

    if template_feature_names is None:
        return None

    if isinstance(template_feature_names, str):
        return [template_feature_names]

    return list(template_feature_names)


def _get_score_column(quality_df):
    if quality_df is None or quality_df.empty:
        return None

    candidates = [
        "overall_with_similarity_score",
        "overall_quality_score",
        "overall_score",
    ]

    for col in candidates:
        if col in quality_df.columns:
            return col

    return None


def _get_dimension_columns(quality_df, include_similarity=True):
    if quality_df is None or quality_df.empty:
        return []

    base = [
        "amplitude_score",
        "smoothness_score",
        "trunk_score",
        "symmetry_score",
        "rhythm_score",
    ]

    if include_similarity:
        base.append("similarity_score_100")

    return [c for c in base if c in quality_df.columns]


def _attach_similarity_to_quality_df(quality_df, per_segment_scores):
    """
    如果 quality_df 中还没有 similarity_score_100，则根据 per_segment_scores 补进去。
    """

    if quality_df is None or quality_df.empty:
        return quality_df

    qdf = quality_df.copy()

    if per_segment_scores is None:
        return qdf

    scores = np.asarray(per_segment_scores, dtype=float)

    if len(scores) != len(qdf):
        return qdf

    if "similarity_score_100" not in qdf.columns:
        if np.nanmin(scores) < 0:
            qdf["similarity_score_100"] = np.clip((scores + 1.0) / 2.0 * 100.0, 0, 100)
        else:
            qdf["similarity_score_100"] = np.clip(scores * 100.0, 0, 100)

    if "overall_with_similarity_score" not in qdf.columns and "overall_quality_score" in qdf.columns:
        qdf["overall_with_similarity_score"] = (
            0.70 * qdf["overall_quality_score"]
            + 0.30 * qdf["similarity_score_100"]
        )

    return qdf


# =========================================================
# 4. 额外可视化：雷达图、趋势图、分割时间轴
# =========================================================
def _plot_radar(values, labels, title, save_path):
    values = np.asarray(values, dtype=float)

    if len(values) == 0:
        return None

    values = np.nan_to_num(values, nan=0.0)
    values = np.clip(values, 0.0, 100.0)

    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False)
    values_closed = np.r_[values, values[0]]
    angles_closed = np.r_[angles, angles[0]]

    fig = plt.figure(figsize=(7, 7))
    ax = fig.add_subplot(111, polar=True)

    ax.plot(angles_closed, values_closed, linewidth=2)
    ax.fill(angles_closed, values_closed, alpha=0.18)

    ax.set_xticks(angles)
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylim(0, 100)
    ax.set_yticks([20, 40, 60, 80, 100])
    ax.set_title(title, pad=20)

    plt.tight_layout()
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=130, bbox_inches="tight")
    plt.close(fig)

    return save_path


def _plot_high_low_radar(high_row, low_row, dim_cols, save_path):
    if high_row is None or low_row is None or not dim_cols:
        return None

    labels = [DIMENSION_LABELS.get(c, c) for c in dim_cols]
    high_values = np.array([_safe_float(high_row.get(c), 0.0) for c in dim_cols])
    low_values = np.array([_safe_float(low_row.get(c), 0.0) for c in dim_cols])

    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False)

    high_closed = np.r_[high_values, high_values[0]]
    low_closed = np.r_[low_values, low_values[0]]
    angles_closed = np.r_[angles, angles[0]]

    fig = plt.figure(figsize=(7, 7))
    ax = fig.add_subplot(111, polar=True)

    ax.plot(angles_closed, high_closed, linewidth=2, label=f"高分周期 Rep {int(high_row['rep_id'])}")
    ax.fill(angles_closed, high_closed, alpha=0.15)

    ax.plot(angles_closed, low_closed, linewidth=2, linestyle="--", label=f"低分周期 Rep {int(low_row['rep_id'])}")
    ax.fill(angles_closed, low_closed, alpha=0.08)

    ax.set_xticks(angles)
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylim(0, 100)
    ax.set_yticks([20, 40, 60, 80, 100])
    ax.set_title("典型高分周期 vs 典型低分周期", pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.28, 1.12))

    plt.tight_layout()
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=130, bbox_inches="tight")
    plt.close(fig)

    return save_path


def _plot_score_trend(quality_df, save_path):
    if quality_df is None or quality_df.empty:
        return None

    dim_cols = _get_dimension_columns(quality_df, include_similarity=True)
    score_col = _get_score_column(quality_df)

    cols = dim_cols.copy()

    if score_col and score_col not in cols:
        cols.append(score_col)

    cols = [c for c in cols if c in quality_df.columns]

    if not cols:
        return None

    x = quality_df["rep_id"].to_numpy() if "rep_id" in quality_df.columns else np.arange(1, len(quality_df) + 1)

    fig = plt.figure(figsize=(12, 5))
    ax = fig.add_subplot(111)

    for col in cols:
        label = DIMENSION_LABELS.get(col, col)
        ax.plot(x, quality_df[col].to_numpy(dtype=float), marker="o", linewidth=2, label=label)

    ax.set_xlabel("动作周期编号")
    ax.set_ylabel("评分")
    ax.set_ylim(0, 105)
    ax.set_title("各周期评分趋势")
    ax.grid(alpha=0.3)
    ax.legend(ncol=3, fontsize=9)

    plt.tight_layout()
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=130, bbox_inches="tight")
    plt.close(fig)

    return save_path


def _plot_segment_timeline(segments_df, n_frames, save_path):
    if segments_df is None or segments_df.empty:
        return None

    fig = plt.figure(figsize=(12, 2.8))
    ax = fig.add_subplot(111)

    for _, row in segments_df.iterrows():
        rep_id = int(row["rep_id"])
        start = int(row["start"])
        end = int(row["end"])
        center = int(row["center"]) if "center" in row else (start + end) // 2

        ax.broken_barh([(start, end - start + 1)], (0, 5), alpha=0.55)
        ax.axvline(center, linestyle="--", linewidth=1, alpha=0.8)
        ax.text(center, 2.5, str(rep_id), ha="center", va="center", fontsize=9)

    ax.set_xlim(0, max(n_frames, int(segments_df["end"].max()) + 1))
    ax.set_ylim(0, 5)
    ax.set_yticks([])
    ax.set_xlabel("Frame")
    ax.set_title("动作周期分割时间轴")
    ax.grid(axis="x", alpha=0.3)

    plt.tight_layout()
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=130, bbox_inches="tight")
    plt.close(fig)

    return save_path


def _plot_rom_summary_bar(rom_summary_df, save_path, top_n=10):
    if rom_summary_df is None or rom_summary_df.empty:
        return None

    if "angle_name" not in rom_summary_df.columns or "mean_rom_deg" not in rom_summary_df.columns:
        return None

    df = rom_summary_df.copy()
    df = df.sort_values("mean_rom_deg", ascending=False).head(top_n)

    labels = [ROM_LABELS.get(x, x) for x in df["angle_name"].astype(str).tolist()]
    values = df["mean_rom_deg"].to_numpy(dtype=float)

    fig = plt.figure(figsize=(12, 5))
    ax = fig.add_subplot(111)

    ax.bar(np.arange(len(values)), values)
    ax.set_xticks(np.arange(len(values)))
    ax.set_xticklabels(labels, rotation=35, ha="right")
    ax.set_ylabel("平均 ROM (deg)")
    ax.set_title("主要关节平均活动范围 ROM")
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=130, bbox_inches="tight")
    plt.close(fig)

    return save_path


def _plot_similarity_by_cycle(result, save_path):
    scores = result.get("per_segment_scores", None)

    if scores is None:
        return None

    scores = np.asarray(scores, dtype=float)

    if len(scores) == 0:
        return None

    if np.nanmin(scores) < 0:
        y = np.clip((scores + 1.0) / 2.0 * 100.0, 0, 100)
    else:
        y = np.clip(scores * 100.0, 0, 100)

    x = np.arange(1, len(y) + 1)

    fig = plt.figure(figsize=(10, 4))
    ax = fig.add_subplot(111)

    ax.bar(x, y)
    ax.axhline(np.nanmean(y), linestyle="--", linewidth=1.5, label=f"平均: {np.nanmean(y):.1f}")
    ax.set_xlabel("动作周期编号")
    ax.set_ylabel("模板相似度评分")
    ax.set_ylim(0, 105)
    ax.set_title("各周期模板相似度")
    ax.grid(axis="y", alpha=0.3)
    ax.legend()

    plt.tight_layout()
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=130, bbox_inches="tight")
    plt.close(fig)

    return save_path


def _generate_extra_figures(result, report, output_dir):
    output_dir = Path(output_dir)
    html_fig_dir = _ensure_dir(output_dir / "figures" / "html_extra")

    quality_df = report.get("quality_df", pd.DataFrame()).copy()
    segments_df = result.get("segments_df", pd.DataFrame()).copy()

    n_frames = 0
    if "features_df" in result:
        n_frames = len(result["features_df"])
    elif "seq" in report:
        n_frames = len(report["seq"])
    elif not segments_df.empty:
        n_frames = int(segments_df["end"].max()) + 1

    quality_df = _attach_similarity_to_quality_df(
        quality_df,
        result.get("per_segment_scores", None),
    )

    report["quality_df"] = quality_df

    dim_cols = _get_dimension_columns(quality_df, include_similarity=False)

    paths = {}

    if not quality_df.empty and dim_cols:
        mean_values = [quality_df[c].mean() for c in dim_cols]
        labels = [DIMENSION_LABELS.get(c, c) for c in dim_cols]

        paths["radar_summary"] = _plot_radar(
            mean_values,
            labels,
            "五维动作质量平均评分",
            html_fig_dir / "radar_summary.png",
        )

        score_col = _get_score_column(quality_df)

        if score_col is not None:
            qdf_valid = quality_df.dropna(subset=[score_col]).copy()

            if not qdf_valid.empty:
                high_row = qdf_valid.loc[qdf_valid[score_col].idxmax()]
                low_row = qdf_valid.loc[qdf_valid[score_col].idxmin()]

                paths["high_low_radar"] = _plot_high_low_radar(
                    high_row,
                    low_row,
                    dim_cols,
                    html_fig_dir / "high_low_radar.png",
                )

        paths["score_trend"] = _plot_score_trend(
            quality_df,
            html_fig_dir / "score_trend.png",
        )

    if not segments_df.empty:
        paths["segment_timeline"] = _plot_segment_timeline(
            segments_df,
            n_frames=n_frames,
            save_path=html_fig_dir / "segment_timeline.png",
        )

    paths["rom_summary_bar"] = _plot_rom_summary_bar(
        report.get("rom_summary_df", pd.DataFrame()),
        html_fig_dir / "rom_summary_bar.png",
    )

    paths["similarity_bar"] = _plot_similarity_by_cycle(
        result,
        html_fig_dir / "similarity_by_cycle.png",
    )

    return {k: v for k, v in paths.items() if v is not None}


# =========================================================
# 5. 文本分析：高分/低分周期
# =========================================================
def _get_high_low_cycle(quality_df):
    if quality_df is None or quality_df.empty:
        return None, None, None

    score_col = _get_score_column(quality_df)

    if score_col is None:
        return None, None, None

    qdf = quality_df.dropna(subset=[score_col]).copy()

    if qdf.empty:
        return None, None, None

    high_row = qdf.loc[qdf[score_col].idxmax()].copy()
    low_row = qdf.loc[qdf[score_col].idxmin()].copy()

    return score_col, high_row, low_row


def _make_cycle_analysis(row, score_col, label="典型周期"):
    if row is None:
        return f"{label}：暂无有效评分。"

    rep_id = int(row["rep_id"]) if "rep_id" in row else -1
    total_score = _safe_float(row.get(score_col, np.nan))
    dim_cols = _get_dimension_columns(pd.DataFrame([row]), include_similarity=False)

    dim_values = {
        DIMENSION_LABELS.get(c, c): _safe_float(row.get(c, np.nan))
        for c in dim_cols
    }

    valid_items = [(k, v) for k, v in dim_values.items() if np.isfinite(v)]

    if valid_items:
        best_dim = max(valid_items, key=lambda x: x[1])
        weak_dim = min(valid_items, key=lambda x: x[1])
        detail = (
            f"其中表现最好的维度为「{best_dim[0]}」({_format_num(best_dim[1])} 分)，"
            f"相对薄弱的维度为「{weak_dim[0]}」({_format_num(weak_dim[1])} 分)。"
        )
    else:
        detail = "暂无可解释的五维评分。"

    return (
        f"{label}为第 {rep_id} 个周期，"
        f"{DIMENSION_LABELS.get(score_col, score_col)}为 {_format_num(total_score)} 分，"
        f"等级为「{_score_text(total_score)}」。{detail}"
    )


def _make_overall_interpretation(quality_df, result):
    if quality_df is None or quality_df.empty:
        return "暂无有效周期评分，无法生成整体表现分析。"

    score_col = _get_score_column(quality_df)
    dim_cols = _get_dimension_columns(quality_df, include_similarity=False)

    parts = []

    count = int(result.get("count", len(quality_df)))
    similarity = _safe_float(result.get("similarity_score", np.nan))

    parts.append(f"本次动作共识别到 {count} 个动作周期。")

    if score_col is not None:
        mean_score = quality_df[score_col].mean()
        parts.append(
            f"整体{DIMENSION_LABELS.get(score_col, score_col)}均值为 {_format_num(mean_score)} 分，"
            f"总体等级为「{_score_text(mean_score)}」。"
        )

    if np.isfinite(similarity):
        sim_show = similarity * 100 if similarity <= 1.5 else similarity
        parts.append(f"模板相似度平均水平约为 {_format_num(sim_show)} 分。")

    if dim_cols:
        dim_mean = quality_df[dim_cols].mean().sort_values(ascending=False)
        best = dim_mean.index[0]
        weak = dim_mean.index[-1]
        parts.append(
            f"五维评分中，「{DIMENSION_LABELS.get(best, best)}」整体表现相对最好，"
            f"均值为 {_format_num(dim_mean.loc[best])} 分；"
            f"「{DIMENSION_LABELS.get(weak, weak)}」相对薄弱，"
            f"均值为 {_format_num(dim_mean.loc[weak])} 分。"
        )

    return "".join(parts)


# =========================================================
# 6. HTML 渲染
# =========================================================
def _collect_existing_figures(output_dir):
    output_dir = Path(output_dir)
    fig_dir = output_dir / "figures"

    figures = {
        "trunk_lean": fig_dir / "trunk_lean.png",
        "spatial_envelope_xz": fig_dir / "spatial_envelope_xz.png",
        "spatial_envelope_xy": fig_dir / "spatial_envelope_xy.png",
        "joint_trajectory_3d": fig_dir / "joint_trajectory_3d.png",
    }

    rom_dir = fig_dir / "rom_curves"
    rom_figs = []

    if rom_dir.exists():
        rom_figs = sorted(rom_dir.glob("*.png"))

    return figures, rom_figs


def _render_cards(result, report):
    quality_df = report.get("quality_df", pd.DataFrame())
    score_col = _get_score_column(quality_df)

    count = int(result.get("count", 0))
    raw_peak_count = int(result.get("raw_peak_count", 0))
    sim = _safe_float(result.get("similarity_score", np.nan))
    sim_show = sim * 100 if np.isfinite(sim) and sim <= 1.5 else sim

    overall_score = np.nan

    if score_col and not quality_df.empty:
        overall_score = quality_df[score_col].mean()

    cards = [
        ("动作类型", html.escape(str(result.get("action_id", "-")))),
        ("识别周期数", str(count)),
        ("原始峰值数", str(raw_peak_count)),
        ("模板相似度", _format_num(sim_show)),
        ("整体评分", _format_num(overall_score)),
        ("整体等级", _score_text(overall_score)),
    ]

    html_cards = ""

    for title, value in cards:
        cls = _score_class(overall_score) if title in ["整体评分", "整体等级"] else ""
        html_cards += f"""
        <div class="metric-card {cls}">
            <div class="metric-title">{title}</div>
            <div class="metric-value">{value}</div>
        </div>
        """

    return html_cards


def _render_dimension_summary(quality_df):
    if quality_df is None or quality_df.empty:
        return '<div class="empty-box">暂无五维评分数据</div>'

    dim_cols = _get_dimension_columns(quality_df, include_similarity=True)

    if not dim_cols:
        return '<div class="empty-box">暂无五维评分数据</div>'

    items = ""

    for col in dim_cols:
        val = quality_df[col].mean()
        items += f"""
        <div class="dimension-row">
            <div class="dimension-name">{DIMENSION_LABELS.get(col, col)}</div>
            <div class="dimension-bar">
                <span style="width:{max(0, min(100, _safe_float(val, 0)))}%"></span>
            </div>
            <div class="dimension-score">{_format_num(val)}</div>
        </div>
        """

    return f'<div class="dimension-panel">{items}</div>'


def _load_emg_summary(output_dir):
    """读取动作目录下的肌电摘要；缺失时尝试从 emg_features.csv 现场统计。"""
    output_dir = Path(output_dir)
    candidates = [
        output_dir / "emg_summary.json",
        output_dir.parent / "emg_summary.json",
    ]
    for path in candidates:
        if path.exists():
            try:
                with path.open("r", encoding="utf-8-sig") as f:
                    data = json.load(f)
                data["_source"] = str(path)
                return data
            except Exception:
                pass

    feature_candidates = [
        output_dir / "emg_features.csv",
        output_dir.parent / "emg_features.csv",
    ]
    for path in feature_candidates:
        if not path.exists():
            continue
        try:
            df = pd.read_csv(path)
            if df.empty:
                continue
            states = df.get("state", pd.Series(dtype=str)).astype(str).str.upper()
            total = max(1, len(df))
            active = states.isin(["SMOOTH_FLEX", "TREMOR", "FATIGUE"]).sum()
            fatigue = (states == "FATIGUE").sum()
            tremor = (states == "TREMOR").sum()
            if "ch" in df and "rms" in df:
                ch0 = df[df["ch"] == 0]["rms"]
                ch1 = df[df["ch"] == 1]["rms"]
            else:
                ch0 = pd.Series(dtype=float)
                ch1 = pd.Series(dtype=float)
            return {
                "_source": str(path),
                "feature_rows": int(len(df)),
                "active_ratio": float(active) / total,
                "fatigue_ratio": float(fatigue) / total,
                "tremor_ratio": float(tremor) / total,
                "avg_rms": float(df["rms"].mean()) if "rms" in df else 0.0,
                "max_rms": float(df["rms"].max()) if "rms" in df else 0.0,
                "avg_fatigue_index": float(df["fatigue_index"].mean()) if "fatigue_index" in df else 0.0,
                "active_muscle_avg_rms": float(ch0.mean()) if not ch0.empty else 0.0,
                "antagonist_muscle_avg_rms": float(ch1.mean()) if not ch1.empty else 0.0,
                "dominant_state": states.mode().iloc[0] if not states.empty else "REST",
            }
        except Exception:
            continue
    return None


def _emg_feature_path(output_dir):
    output_dir = Path(output_dir)
    for path in (output_dir / "emg_features.csv", output_dir.parent / "emg_features.csv"):
        if path.exists():
            return path
    return None


def _emg_raw_path(output_dir):
    output_dir = Path(output_dir)
    for path in (output_dir / "emg_raw.csv", output_dir.parent / "emg_raw.csv"):
        if path.exists():
            return path
    return None


def _load_emg_feature_df(output_dir):
    path = _emg_feature_path(output_dir)
    if path is None:
        return None, None
    try:
        df = pd.read_csv(path)
    except Exception:
        return None, path
    if df.empty:
        return None, path
    for col in ("timestamp_ns", "seq", "ch", "rms", "zcr", "cv", "fatigue_index"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    valid_ts = df["timestamp_ns"].dropna() if "timestamp_ns" in df.columns else pd.Series(dtype=float)
    if not valid_ts.empty:
        t0 = valid_ts.min()
        df["time_s"] = (df["timestamp_ns"] - t0) / 1_000_000_000.0
    else:
        df["time_s"] = np.arange(len(df), dtype=float)
    return df.dropna(subset=["time_s"]), path


def _load_emg_raw_df(output_dir):
    path = _emg_raw_path(output_dir)
    if path is None:
        return None, None
    try:
        df = pd.read_csv(path)
    except Exception:
        return None, path
    if df.empty:
        return None, path
    for col in ("timestamp_ns", "packet_seq", "sample_index", "ch0", "ch1"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    valid_ts = df["timestamp_ns"].dropna() if "timestamp_ns" in df.columns else pd.Series(dtype=float)
    if not valid_ts.empty:
        t0 = valid_ts.min()
        df["time_s"] = (df["timestamp_ns"] - t0) / 1_000_000_000.0
    else:
        df["time_s"] = np.arange(len(df), dtype=float)
    return df.dropna(subset=["time_s"]), path


def _estimate_rate_from_time(df):
    if df is None or df.empty or "time_s" not in df.columns:
        return 0.0
    times = df["time_s"].dropna().to_numpy(dtype=float)
    if len(times) < 2:
        return 0.0
    duration = float(times[-1] - times[0])
    if duration <= 0:
        return 0.0
    if "seq" in df.columns:
        seq_count = int(df["seq"].dropna().nunique())
        if seq_count > 1:
            return (seq_count - 1) / duration
    return (len(times) - 1) / duration


def _thin_df(df, max_points=900):
    if df is None or len(df) <= max_points:
        return df
    step = max(1, int(np.ceil(len(df) / max_points)))
    return df.iloc[::step].copy()


def _plot_emg_feature_trends(feature_df, save_path):
    if feature_df is None or feature_df.empty:
        return None
    required = [col for col in EMG_FEATURE_LABELS if col in feature_df.columns]
    if not required or "ch" not in feature_df.columns:
        return None

    fig, axes = plt.subplots(len(required), 1, figsize=(9.2, 2.15 * len(required)), sharex=True)
    if len(required) == 1:
        axes = [axes]
    colors = {0: "#2563eb", 1: "#16a34a"}

    for ax, col in zip(axes, required):
        plotted = False
        for channel in (0, 1):
            ch_df = feature_df[feature_df["ch"] == channel]
            if ch_df.empty:
                continue
            ch_df = _thin_df(ch_df.sort_values("time_s"))
            ax.plot(
                ch_df["time_s"].to_numpy(dtype=float),
                ch_df[col].to_numpy(dtype=float),
                linewidth=1.8,
                color=colors[channel],
                label=f"CH{channel + 1}",
            )
            plotted = True
        ax.set_ylabel(EMG_FEATURE_LABELS.get(col, col), fontsize=9)
        ax.grid(alpha=0.25)
        if plotted:
            ax.legend(loc="upper right", fontsize=8)
    axes[-1].set_xlabel("训练时间（秒）")
    fig.suptitle("肌电特征随时间变化", fontsize=13, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return save_path


def _windowed_raw_features(raw_df, window_size=160, hop_size=80):
    if raw_df is None or raw_df.empty:
        return pd.DataFrame()
    channels = [col for col in ("ch0", "ch1") if col in raw_df.columns]
    if not channels:
        return pd.DataFrame()
    rows = []
    ordered = raw_df.sort_values("time_s").reset_index(drop=True)
    for start in range(0, max(0, len(ordered) - window_size + 1), hop_size):
        window = ordered.iloc[start:start + window_size]
        if window.empty:
            continue
        time_s = float(window["time_s"].iloc[len(window) // 2])
        for ch_idx, col in enumerate(channels):
            values = pd.to_numeric(window[col], errors="coerce").dropna().to_numpy(dtype=float)
            if len(values) < 2:
                continue
            centered = values - np.mean(values)
            diff = np.diff(centered)
            rows.append({
                "time_s": time_s,
                "ch": ch_idx,
                "mav": float(np.mean(np.abs(centered))),
                "iemg": float(np.sum(np.abs(centered))),
                "wl": float(np.sum(np.abs(diff))),
                "zc": float(np.sum(centered[:-1] * centered[1:] < 0)),
            })
    return pd.DataFrame(rows)


def _plot_emg_raw_time_features(raw_df, save_path):
    features = _windowed_raw_features(raw_df)
    if features.empty:
        return None
    cols = [col for col in ("mav", "iemg", "wl", "zc") if col in features.columns]
    fig, axes = plt.subplots(len(cols), 1, figsize=(9.2, 2.05 * len(cols)), sharex=True)
    if len(cols) == 1:
        axes = [axes]
    colors = {0: "#7c3aed", 1: "#0891b2"}
    for ax, col in zip(axes, cols):
        for channel in (0, 1):
            ch_df = features[features["ch"] == channel]
            if ch_df.empty:
                continue
            ch_df = _thin_df(ch_df.sort_values("time_s"))
            ax.plot(
                ch_df["time_s"].to_numpy(dtype=float),
                ch_df[col].to_numpy(dtype=float),
                linewidth=1.7,
                color=colors[channel],
                label=f"CH{channel + 1}",
            )
        ax.set_ylabel(EMG_RAW_FEATURE_LABELS.get(col, col), fontsize=9)
        ax.grid(alpha=0.25)
        ax.legend(loc="upper right", fontsize=8)
    axes[-1].set_xlabel("训练时间（秒）")
    fig.suptitle("原始肌电滑动窗口分析", fontsize=13, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return save_path


def _generate_emg_figures(output_dir):
    output_dir = Path(output_dir)
    try:
        setup_chinese_font(verbose=False)
    except Exception:
        pass
    fig_dir = _ensure_dir(output_dir / "figures" / "html_extra")
    figures = {}
    feature_df, _ = _load_emg_feature_df(output_dir)
    raw_df, _ = _load_emg_raw_df(output_dir)
    feature_plot = _plot_emg_feature_trends(feature_df, fig_dir / "emg_feature_trends.png")
    raw_plot = _plot_emg_raw_time_features(raw_df, fig_dir / "emg_raw_time_features.png")
    if feature_plot:
        figures["feature_trends"] = feature_plot
    if raw_plot:
        figures["raw_time_features"] = raw_plot
    return figures, feature_df, raw_df


def _render_emg_section(output_dir, embed_images=True):
    """生成报告中的肌电状态分析区域，保证无肌电文件时也不报错。"""
    summary = _load_emg_summary(output_dir)
    emg_figures, feature_df, raw_df = _generate_emg_figures(output_dir)
    if not summary:
        return """
        <div class="section">
            <h2>8. 肌电状态分析</h2>
            <div class="empty-box">本次未接入肌电数据。</div>
        </div>
        """

    active_rms = float(summary.get("active_muscle_avg_rms", summary.get("ch0_avg_rms", 0.0)) or 0.0)
    antagonist_rms = float(summary.get("antagonist_muscle_avg_rms", summary.get("ch1_avg_rms", 0.0)) or 0.0)
    active_ratio = float(summary.get("active_ratio", 0.0) or 0.0)
    fatigue_ratio = float(summary.get("fatigue_ratio", 0.0) or 0.0)
    tremor_ratio = float(summary.get("tremor_ratio", 0.0) or 0.0)
    avg_fatigue = float(summary.get("avg_fatigue_index", 0.0) or 0.0)
    dominant_state = str(summary.get("dominant_state", "REST"))
    feature_rate = _estimate_rate_from_time(feature_df)
    raw_rate = _estimate_rate_from_time(raw_df)

    suggestions = []
    if fatigue_ratio >= 0.25:
        suggestions.append("疲劳占比较高，提示存在肌肉疲劳，建议降低训练强度或增加休息。")
    if tremor_ratio >= 0.15:
        suggestions.append("震颤/异常占比较高，提示发力控制不稳定，建议关注动作稳定性。")
    if active_ratio < 0.20:
        suggestions.append("发力占比较低，提示训练参与度不足，可适当提高主动发力引导。")
    if not suggestions:
        suggestions.append("肌电参与度和疲劳指标处于可接受范围，可结合动作评分继续观察。")

    cards = [
        ("主动肌平均强度（RMS）", f"{active_rms:.1f}"),
        ("拮抗肌平均强度（RMS）", f"{antagonist_rms:.1f}"),
        ("主动发力占比", f"{active_ratio:.1%}"),
        ("疲劳倾向占比", f"{fatigue_ratio:.1%}"),
        ("震颤/不稳定占比", f"{tremor_ratio:.1%}"),
        ("平均疲劳指数", f"{avg_fatigue:.3f}"),
        ("主导状态", html.escape(EMG_STATE_LABELS.get(dominant_state, dominant_state))),
    ]
    if feature_rate > 0:
        cards.append(("肌电特征帧率", f"{feature_rate:.1f} Hz"))
    if raw_rate > 0:
        cards.append(("原始肌电采样率", f"{raw_rate:.1f} Hz"))
    cards_html = "".join(
        f"""
        <div class="metric-card">
            <div class="metric-title">{html.escape(title)}</div>
            <div class="metric-value">{value}</div>
        </div>
        """
        for title, value in cards
    )
    suggestion_html = "<br/>".join(html.escape(item) for item in suggestions)
    figure_html = ""
    if emg_figures.get("feature_trends"):
        figure_html += _img_tag(
            emg_figures["feature_trends"],
            title="肌电特征曲线：强度、过零率、稳定性与疲劳指数",
            embed=embed_images,
        )
    if emg_figures.get("raw_time_features"):
        figure_html += _img_tag(
            emg_figures["raw_time_features"],
            title="原始肌电窗口特征：平均绝对值、积分值、波形长度与过零次数",
            embed=embed_images,
        )
    if not figure_html:
        figure_html = '<div class="empty-box">未找到可绘制的肌电曲线数据；报告仅展示摘要指标。</div>'
    return f"""
    <div class="section">
        <h2>8. 肌电状态分析</h2>
        <div class="cards">{cards_html}</div>
        <div class="analysis-box">{suggestion_html}</div>
        <div class="figure-grid">{figure_html}</div>
        <div class="analysis-box">
            指标说明：肌电强度（RMS）反映肌肉收缩强弱；平均绝对肌电值用于观察整体发力水平；
            积分肌电值反映一个时间窗内的累计肌肉活动量；波形长度用于观察信号复杂度和发力变化；
            过零次数/过零率可辅助判断肌肉激活切换是否频繁。以上结果用于康复训练观察，不替代临床诊断。
        </div>
        <p class="empty-box">数据来源：{html.escape(str(summary.get("_source", "-")))}</p>
    </div>
    """


def _render_html(
    result,
    report,
    output_dir,
    extra_figures,
    html_title="离线动作分析报告",
    embed_images=True,
):
    output_dir = Path(output_dir)
    existing_figs, rom_figs = _collect_existing_figures(output_dir)

    quality_df = report.get("quality_df", pd.DataFrame())
    rom_summary_df = report.get("rom_summary_df", pd.DataFrame())
    rom_detail_df = report.get("rom_detail_df", pd.DataFrame())
    offset_summary_df = report.get("offset_summary_df", pd.DataFrame())
    offset_detail_df = report.get("offset_detail_df", pd.DataFrame())
    segments_df = result.get("segments_df", pd.DataFrame())

    score_col, high_row, low_row = _get_high_low_cycle(quality_df)

    overall_interpretation = _make_overall_interpretation(quality_df, result)
    high_analysis = _make_cycle_analysis(high_row, score_col, label="典型高分周期") if score_col else "暂无典型高分周期分析。"
    low_analysis = _make_cycle_analysis(low_row, score_col, label="典型低分周期") if score_col else "暂无典型低分周期分析。"

    cards_html = _render_cards(result, report)
    dimension_html = _render_dimension_summary(quality_df)

    extra_imgs_html = ""
    extra_order = [
        ("segment_timeline", "动作分割时间轴"),
        ("radar_summary", "五维动作质量雷达图"),
        ("score_trend", "各周期评分趋势"),
        ("high_low_radar", "典型高分/低分周期对比雷达图"),
        ("similarity_bar", "各周期模板相似度"),
        ("rom_summary_bar", "主要 ROM 指标汇总"),
    ]

    for key, title in extra_order:
        if key in extra_figures:
            extra_imgs_html += _img_tag(extra_figures[key], title=title, embed=embed_images)

    existing_imgs_html = ""
    existing_titles = {
        "trunk_lean": "躯干前后倾/左右倾评估",
        "spatial_envelope_xz": "运动轨迹空间包络图 XZ 投影",
        "spatial_envelope_xy": "运动轨迹空间包络图 XY 投影",
        "joint_trajectory_3d": "关键关节 3D 轨迹",
    }

    for key, path in existing_figs.items():
        existing_imgs_html += _img_tag(path, title=existing_titles.get(key, key), embed=embed_images)

    rom_imgs_html = ""

    for path in rom_figs[:12]:
        rom_imgs_html += _img_tag(path, title=f"ROM 曲线：{path.stem}", embed=embed_images)

    generated_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    emg_html = _render_emg_section(output_dir, embed_images=embed_images)

    css = """
    <style>
        body {
            font-family:
                "Microsoft YaHei",
                "微软雅黑",
                "SimHei",
                "黑体",
                "PingFang SC",
                "Hiragino Sans GB",
                "Noto Sans CJK SC",
                "Source Han Sans SC",
                "WenQuanYi Micro Hei",
                -apple-system,
                BlinkMacSystemFont,
                "Segoe UI",
                Arial,
                sans-serif;
            margin: 0;
            background: #f5f7fb;
            color: #1f2937;
        }
        .container {
            max-width: 1280px;
            margin: 0 auto;
            padding: 28px;
        }
        .header {
            background: linear-gradient(135deg, #111827, #374151);
            color: white;
            border-radius: 18px;
            padding: 28px 32px;
            margin-bottom: 22px;
        }
        .header h1 {
            margin: 0 0 10px 0;
            font-size: 30px;
        }
        .header .subtitle {
            opacity: 0.85;
            font-size: 14px;
        }
        .section {
            background: white;
            border-radius: 18px;
            padding: 24px;
            margin: 20px 0;
            box-shadow: 0 8px 28px rgba(15, 23, 42, 0.06);
        }
        .section h2 {
            margin-top: 0;
            padding-bottom: 10px;
            border-bottom: 1px solid #e5e7eb;
            font-size: 22px;
        }
        .section h3 {
            margin-top: 22px;
            font-size: 18px;
        }
        .cards {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
            gap: 14px;
            margin: 18px 0;
        }
        .metric-card {
            background: #f9fafb;
            border: 1px solid #e5e7eb;
            border-radius: 14px;
            padding: 16px;
        }
        .metric-title {
            color: #6b7280;
            font-size: 13px;
            margin-bottom: 10px;
        }
        .metric-value {
            font-size: 24px;
            font-weight: 700;
        }
        .score-good { border-left: 6px solid #16a34a; }
        .score-ok { border-left: 6px solid #2563eb; }
        .score-mid { border-left: 6px solid #f59e0b; }
        .score-low { border-left: 6px solid #dc2626; }
        .score-na { border-left: 6px solid #9ca3af; }
        .analysis-box {
            background: #f8fafc;
            border-left: 5px solid #2563eb;
            padding: 16px 18px;
            border-radius: 12px;
            line-height: 1.8;
            margin: 14px 0;
        }
        .warning-box {
            background: #fff7ed;
            border-left: 5px solid #f97316;
            padding: 16px 18px;
            border-radius: 12px;
            line-height: 1.8;
            margin: 14px 0;
        }
        .dimension-panel {
            display: grid;
            gap: 12px;
            margin-top: 12px;
        }
        .dimension-row {
            display: grid;
            grid-template-columns: 120px 1fr 60px;
            align-items: center;
            gap: 12px;
        }
        .dimension-name {
            font-size: 14px;
            color: #374151;
        }
        .dimension-bar {
            height: 12px;
            background: #e5e7eb;
            border-radius: 999px;
            overflow: hidden;
        }
        .dimension-bar span {
            display: block;
            height: 100%;
            background: linear-gradient(90deg, #60a5fa, #2563eb);
            border-radius: 999px;
        }
        .dimension-score {
            text-align: right;
            font-weight: 600;
        }
        .figure-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(420px, 1fr));
            gap: 18px;
            margin-top: 18px;
        }
        .figure-card {
            background: #f9fafb;
            border: 1px solid #e5e7eb;
            border-radius: 14px;
            padding: 12px;
            text-align: center;
            overflow: hidden;
        }
        .figure-card img {
            max-width: 100%;
            height: auto;
            border-radius: 10px;
        }
        .fig-title {
            font-weight: 600;
            margin: 6px 0 12px 0;
            color: #111827;
        }
        .data-table {
            width: 100%;
            border-collapse: collapse;
            font-size: 13px;
            margin-top: 12px;
            font-family:
                "Microsoft YaHei",
                "微软雅黑",
                "SimHei",
                "黑体",
                "Noto Sans CJK SC",
                Arial,
                sans-serif;
        }
        .data-table th {
            background: #f3f4f6;
            color: #374151;
            padding: 8px;
            text-align: left;
            border: 1px solid #e5e7eb;
        }
        .data-table td {
            padding: 8px;
            border: 1px solid #e5e7eb;
        }
        .data-table tr:nth-child(even) {
            background: #f9fafb;
        }
        .empty-box {
            color: #6b7280;
            background: #f9fafb;
            border: 1px dashed #d1d5db;
            border-radius: 12px;
            padding: 16px;
        }
        .footer {
            color: #6b7280;
            text-align: center;
            margin: 26px 0 8px 0;
            font-size: 12px;
        }
        code {
            background: #f3f4f6;
            padding: 2px 5px;
            border-radius: 4px;
            font-family:
                Consolas,
                "Microsoft YaHei",
                "微软雅黑",
                monospace;
        }
    </style>
    """

    html_content = f"""
    <!DOCTYPE html>
    <html lang="zh-CN">
    <head>
        <meta charset="utf-8"/>
        <title>{html.escape(html_title)}</title>
        {css}
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>{html.escape(html_title)}</h1>
                <div class="subtitle">
                    文件：{html.escape(str(result.get("csv_path", "-")))} |
                    动作：{html.escape(str(result.get("action_id", "-")))} |
                    生成时间：{generated_time}
                </div>
            </div>

            <div class="section">
                <h2>1. 总览</h2>
                <div class="cards">
                    {cards_html}
                </div>
                <div class="analysis-box">
                    {html.escape(overall_interpretation)}
                </div>
                <h3>五维评分概览</h3>
                {dimension_html}
            </div>

            <div class="section">
                <h2>2. 分割与计数结果</h2>
                <p>
                    本节展示动作周期分割结果。<code>rep_id</code> 表示第几个周期，
                    <code>start/end</code> 表示该周期的起止帧，
                    <code>center</code> 表示周期中心峰位置。
                </p>
                <div class="figure-grid">
                    {_img_tag(extra_figures.get("segment_timeline", ""), title="动作分割时间轴", embed=embed_images)}
                </div>
                <h3>分割明细</h3>
                {_to_html_table(segments_df, max_rows=50)}
            </div>

            <div class="section">
                <h2>3. 各周期评分表现</h2>
                <div class="figure-grid">
                    {_img_tag(extra_figures.get("radar_summary", ""), title="五维动作质量平均评分", embed=embed_images)}
                    {_img_tag(extra_figures.get("score_trend", ""), title="各周期评分趋势", embed=embed_images)}
                    {_img_tag(extra_figures.get("similarity_bar", ""), title="各周期模板相似度", embed=embed_images)}
                </div>
                <h3>周期评分表</h3>
                {_to_html_table(quality_df, max_rows=50)}
            </div>

            <div class="section">
                <h2>4. 典型高分周期与典型低分周期分析</h2>
                <div class="figure-grid">
                    {_img_tag(extra_figures.get("high_low_radar", ""), title="典型高分/低分周期对比雷达图", embed=embed_images)}
                </div>
                <div class="analysis-box">
                    {html.escape(high_analysis)}
                </div>
                <div class="warning-box">
                    {html.escape(low_analysis)}
                </div>
                <p>
                    解释：高分周期通常代表本次动作中幅度、平滑性、躯干稳定、对称性与节奏性综合表现较好的周期；
                    低分周期用于定位当前动作中最需要关注的薄弱环节。
                </p>
            </div>

            <div class="section">
                <h2>5. ROM 关节活动度报告</h2>
                <div class="figure-grid">
                    {_img_tag(extra_figures.get("rom_summary_bar", ""), title="主要关节平均 ROM", embed=embed_images)}
                </div>
                <h3>ROM 汇总</h3>
                {_to_html_table(rom_summary_df, max_rows=30)}
                <h3>各周期 ROM 明细</h3>
                {_to_html_table(rom_detail_df, max_rows=80)}
            </div>

            <div class="section">
                <h2>6. 姿态偏移与躯干稳定评估</h2>
                <div class="figure-grid">
                    {_img_tag(existing_figs.get("trunk_lean", ""), title="躯干前后倾/左右倾评估", embed=embed_images)}
                    {_img_tag(existing_figs.get("spatial_envelope_xz", ""), title="运动轨迹空间包络图 XZ 投影", embed=embed_images)}
                    {_img_tag(existing_figs.get("spatial_envelope_xy", ""), title="运动轨迹空间包络图 XY 投影", embed=embed_images)}
                    {_img_tag(existing_figs.get("joint_trajectory_3d", ""), title="关键关节 3D 轨迹", embed=embed_images)}
                </div>
                <h3>偏移与倾斜汇总</h3>
                {_to_html_table(offset_summary_df, max_rows=50)}
                <h3>各周期偏移与倾斜明细</h3>
                {_to_html_table(offset_detail_df, max_rows=80)}
            </div>

            <div class="section">
                <h2>7. ROM 曲线图</h2>
                <p>以下图像来自已有绘图函数 <code>plot_rom_curves_with_segments()</code> 的输出。</p>
                <div class="figure-grid">
                    {rom_imgs_html}
                </div>
            </div>

            {emg_html}

            <div class="section">
                <h2>9. 报告说明</h2>
                <p>
                    模板相似度用于描述当前动作与历史模板的形态一致性；
                    五维动作质量评分用于描述动作执行质量，包括动作幅度、速度平滑性、躯干稳定、左右对称性和节奏性。
                    因此，模板相似度回答“像不像模板”，五维评分回答“做得好不好”。
                </p>
                <p>
                    本报告中的评分主要用于运动表现分析和训练反馈，不应单独作为医学诊断结论。
                </p>
            </div>

            <div class="footer">
                Offline Motion Analysis Report | generated by offline_html_report.py
            </div>
        </div>
    </body>
    </html>
    """

    return html_content


# =========================================================
# 7. 主 API：输入离线配置，输出 HTML 报告
# =========================================================
def generate_offline_action_html_report(
    csv_path,
    action_type,
    output_dir,
    template_npz,
    template_meta_csv=None,
    template_feature_names=None,
    fs=20,
    min_interval_sec=1.0,
    target_len=100,
    poly_degree=3,
    smooth_win=11,
    smooth_poly=3,
    prominence_ratio=0.30,
    similarity_method="centroid",
    normalize=True,
    smooth_features=True,
    use_full_range=True,
    plot=True,
    show_plots=False,
    html_filename="offline_action_report.html",
    html_title="离线动作分析报告",
    embed_images=True,
):
    """
    一键生成离线动作分析 HTML 报告。

    参数：
        csv_path:
            宽表骨架 CSV 文件。

        action_type:
            动作类型，例如 "M1"。

        output_dir:
            输出目录。报告、CSV 和图像都会放到这个目录下。

        template_npz:
            模板库 npz。

        template_meta_csv:
            模板元信息 CSV。

        template_feature_names:
            模板相似度使用的特征。
            如果为 None，会尝试从 ACTION_FEATURE_CONFIG[action_type]["default"] 读取。

        fs:
            帧率。

        html_filename:
            输出 HTML 文件名。

        embed_images:
            True 表示把图像编码进 HTML，形成单文件报告；
            False 表示 HTML 以相对路径引用图像。
    """
    
    # -----------------------------------------------------
    # 0. 配置中文字体
    # -----------------------------------------------------
    selected_font = setup_chinese_font(verbose=True)

    csv_path = Path(csv_path)
    output_dir = _ensure_dir(output_dir)

    action_type = str(action_type).upper()

    template_feature_names = _normalize_template_feature_names(
        template_feature_names,
        action_type=action_type,
    )

    # -----------------------------------------------------
    # 1. 离线分割、计数、模板相似度
    # -----------------------------------------------------
    result = analyze_action_csv(
        csv_path=csv_path,
        action_type=action_type,
        template_npz=template_npz,
        template_meta_csv=template_meta_csv,
        template_feature_names=template_feature_names,
        fs=fs,
        min_interval_sec=min_interval_sec,
        poly_degree=poly_degree,
        smooth_win=smooth_win,
        smooth_poly=smooth_poly,
        prominence_ratio=prominence_ratio,
        target_len=target_len,
        normalize=normalize,
        smooth_features=smooth_features,
        use_full_range=use_full_range,
        similarity_method=similarity_method,
        return_features=True,
    )

    # -----------------------------------------------------
    # 2. ROM、质量评分、偏移评估、基础图像
    # -----------------------------------------------------
    report = generate_report_from_action_api_result(
        csv_path=csv_path,
        action_id=action_type,
        api_result=result,
        output_dir=output_dir,
        plot=plot,
        show_plots=show_plots,
    )

    # -----------------------------------------------------
    # 3. 补充 HTML 专用图像
    # -----------------------------------------------------
    extra_figures = _generate_extra_figures(
        result=result,
        report=report,
        output_dir=output_dir,
    )

    # -----------------------------------------------------
    # 4. 生成 HTML
    # -----------------------------------------------------
    html_content = _render_html(
        result=result,
        report=report,
        output_dir=output_dir,
        extra_figures=extra_figures,
        html_title=html_title,
        embed_images=embed_images,
    )

    html_path = output_dir / html_filename

    with html_path.open("w", encoding="utf-8") as f:
        f.write(html_content)

    return {
        "html_path": str(html_path),
        "result": result,
        "report": report,
        "extra_figures": extra_figures,
    }


# =========================================================
# 8. 使用示例
# =========================================================
if __name__ == "__main__":

    CSV_PATH = r"data\processed\M1\skeleton3d.csv"
    ACTION_TYPE = "M1"
    OUTPUT_DIR = r"data\outputs\rom_report\M1_example"

    out = generate_offline_action_html_report(
        csv_path=CSV_PATH,
        action_type=ACTION_TYPE,
        output_dir=OUTPUT_DIR,
        template_npz=r"data\outputs\templates\action_templates.npz",
        template_meta_csv=r"data\outputs\templates\action_templates_meta.csv",

        # 如果你已经统一了 action_config.py，建议保持 None 自动读取 default
        # 也可以显式写：
        # template_feature_names=ACTION_FEATURE_CONFIG[ACTION_TYPE]["default"],
        template_feature_names=None,

        fs=20,
        min_interval_sec=1.0,
        target_len=100,

        plot=True,
        show_plots=False,

        html_filename="offline_action_report.html",
        html_title="离线动作分析报告 - M1",
        embed_images=True,
    )

    print("HTML 报告已生成：", out["html_path"], file=sys.stderr, flush=True)
