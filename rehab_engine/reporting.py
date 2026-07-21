"""Reliable session-level HTML report generation for recorded skeleton CSV files."""

from __future__ import annotations

import csv
import html
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional


def _read_session_meta(session_dir: Path) -> Dict:
    for name in ("session_ui_meta.json", "session_meta.json", "meta.json"):
        path = session_dir / name
        if path.exists():
            try:
                return json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
    return {}


def _read_course_summary(session_dir: Path) -> Dict:
    path = session_dir / "course_summary.json"
    if not path.exists():
        return {}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        return value if isinstance(value, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


def analyze_skeleton_csv(csv_path: str) -> Dict[str, float]:
    """Compute lightweight, model-independent recording quality statistics."""
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"骨骼数据不存在：{path}")

    frames = 0
    duration = 0.0
    valid_sum = 0
    valid_total = 0
    confidence_sum = 0.0
    confidence_total = 0
    first_timestamp: Optional[int] = None
    last_timestamp: Optional[int] = None

    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        valid_columns = [name for name in (reader.fieldnames or []) if name.endswith("_valid")]
        score_columns = [name for name in (reader.fieldnames or []) if name.endswith("_score")]
        wide_x_columns = [
            name for name in (reader.fieldnames or [])
            if name.endswith("_x") and name[:2].isdigit()
        ]
        for row in reader:
            frames += 1
            try:
                duration += max(0.0, float(row.get("dt_seconds", 0) or 0))
            except ValueError:
                pass
            try:
                timestamp = int(row.get("timestamp_ns", 0) or 0)
                first_timestamp = timestamp if first_timestamp is None else first_timestamp
                last_timestamp = timestamp
            except ValueError:
                pass
            for column in valid_columns:
                valid_total += 1
                valid_sum += 1 if str(row.get(column, "0")).lower() in ("1", "true") else 0
            for column in score_columns:
                try:
                    confidence_sum += float(row.get(column, 0) or 0)
                    confidence_total += 1
                except ValueError:
                    pass
            if not valid_columns:
                for x_column in wide_x_columns:
                    prefix = x_column[:-2]
                    valid_total += 1
                    if all(str(row.get(f"{prefix}_{axis}", "")).strip() for axis in "xyz"):
                        valid_sum += 1

    if duration <= 0 and first_timestamp is not None and last_timestamp is not None:
        duration = max(0.0, (last_timestamp - first_timestamp) / 1_000_000_000)
    return {
        "frames": frames,
        "duration_seconds": duration,
        "valid_ratio": valid_sum / valid_total if valid_total else 0.0,
        "mean_confidence": confidence_sum / confidence_total if confidence_total else 0.0,
        "estimated_fps": frames / duration if duration > 0 else 0.0,
    }


def generate_session_report(session_dir: str, csv_path: str) -> str:
    """Generate an always-available HTML summary and return its absolute path."""
    session = Path(session_dir).resolve()
    session.mkdir(parents=True, exist_ok=True)
    analysis_path = Path(csv_path)
    detailed = analysis_path.with_name("skeleton_3d_detailed.csv")
    if analysis_path.name == "skeleton_3d.csv" and detailed.is_file():
        analysis_path = detailed
    try:
        stats = analyze_skeleton_csv(str(analysis_path))
    except FileNotFoundError:
        stats = {
            "frames": 0,
            "duration_seconds": 0.0,
            "valid_ratio": 0.0,
            "mean_confidence": 0.0,
            "estimated_fps": 0.0,
        }
    meta = _read_session_meta(session)
    course_summary = _read_course_summary(session)

    patient = html.escape(meta.get("patient_name") or "未命名训练对象")
    course = html.escape(
        meta.get("course_name") or course_summary.get("course_name") or "康复训练")
    engine_mode = "模拟模式" if meta.get("engine_mode") == "stub" else "真实引擎"
    minutes, seconds = divmod(int(meta.get("elapsed_seconds") or stats["duration_seconds"]), 60)
    validity = stats["valid_ratio"] * 100
    confidence = stats["mean_confidence"] * 100
    quality_text = "良好" if validity >= 90 else "一般" if validity >= 70 else "需要检查"
    generated_at = datetime.now().strftime("%Y-%m-%d %H:%M")
    actions = course_summary.get("actions", [])
    actions = actions if isinstance(actions, list) else []
    completed_actions = sum(
        1 for action in actions
        if isinstance(action, dict) and int(action.get("actual_reps", 0) or 0) > 0
    )
    scored_actions = [
        float(action.get("average_score", 0) or 0)
        for action in actions if isinstance(action, dict)
        and float(action.get("average_score", 0) or 0) > 0
    ]
    course_average = sum(scored_actions) / len(scored_actions) if scored_actions else 0.0
    action_rows = []
    for action in actions:
        if not isinstance(action, dict):
            continue
        name = html.escape(str(action.get("name_cn") or action.get("action_id") or "动作"))
        action_id = html.escape(str(action.get("action_id") or ""))
        actual = int(action.get("actual_reps", 0) or 0)
        target = int(action.get("target_reps", 0) or 0)
        average = float(action.get("average_score", 0) or 0)
        report_path = Path(str(action.get("report_path") or ""))
        report_link = "—"
        if report_path.is_file():
            report_link = (
                f'<a href="{html.escape(report_path.resolve().as_uri())}">查看动作报告</a>')
        action_rows.append(
            f"<tr><td>{name}<br><span class=\"muted\">{action_id}</span></td>"
            f"<td>{actual} / {target or '—'}</td>"
            f"<td>{average:.1f}</td><td>{report_link}</td></tr>"
        )
    action_section = ""
    if action_rows:
        action_section = (
            '<div class="card"><h2>课程动作完成情况</h2><table>'
            '<tr><th>动作</th><th>完成次数</th><th>平均评分</th><th>详细报告</th></tr>'
            + "".join(action_rows) + "</table></div>"
        )

    document = f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><style>
body{{font-family:'Segoe UI','Microsoft YaHei',sans-serif;background:#F4F7FB;color:#344054;margin:0;padding:34px;}}
.wrap{{max-width:980px;margin:auto;}} .eyebrow{{color:#2563EB;font-weight:700;font-size:12px;letter-spacing:1px;}}
h1{{color:#172033;margin:8px 0 4px;font-size:30px;}} .muted{{color:#667085;}}
.card{{background:#fff;border:1px solid #E4EAF2;border-radius:16px;padding:24px;margin-top:18px;}}
.grid{{display:grid;grid-template-columns:repeat(4,1fr);gap:14px;margin-top:18px;}}
.metric{{background:#F8FAFD;border-radius:12px;padding:18px;}} .label{{color:#667085;font-size:12px;}}
.value{{color:#172033;font-size:24px;font-weight:700;margin-top:6px;}} .good{{color:#059669;}}
.bar{{height:9px;background:#E9EEF5;border-radius:6px;overflow:hidden;margin-top:10px;}}
.fill{{height:100%;background:#2563EB;border-radius:6px;}} table{{width:100%;border-collapse:collapse;}}
th,td{{padding:10px;text-align:left;border-bottom:1px solid #EEF2F6;}} th{{color:#475467;}} td:first-child{{color:#667085;}}
a{{color:#2563EB;text-decoration:none;font-weight:600;}}
img{{max-width:100%;height:auto;}}
.note{{background:#EAF2FF;color:#1D4ED8;border-radius:10px;padding:14px 16px;line-height:1.7;}}
@media(max-width:760px){{.grid{{grid-template-columns:repeat(2,1fr);}}}}
</style></head><body><div class="wrap">
<div class="eyebrow">SESSION SUMMARY</div><h1>{course}</h1>
<div class="muted">{patient} · {html.escape(generated_at)}</div>
<div class="grid">
 <div class="metric"><div class="label">训练时长</div><div class="value">{minutes:02d}:{seconds:02d}</div></div>
 <div class="metric"><div class="label">骨骼帧数</div><div class="value">{stats['frames']}</div></div>
 <div class="metric"><div class="label">有效关节比例</div><div class="value good">{validity:.1f}%</div></div>
 <div class="metric"><div class="label">数据质量</div><div class="value">{quality_text}</div></div>
</div>
<div class="card"><h2>采集质量</h2><p class="muted">该部分描述骨骼数据完整度，不代表临床诊断或最终康复评分。</p>
<div>有效关节比例 <b>{validity:.1f}%</b><div class="bar"><div class="fill" style="width:{min(100, validity):.1f}%"></div></div></div>
<div style="margin-top:18px">平均关节置信度 <b>{confidence:.1f}%</b><div class="bar"><div class="fill" style="width:{min(100, confidence):.1f}%"></div></div></div></div>
<div class="card"><h2>会话信息</h2><table>
<tr><td>训练对象</td><td>{patient}</td></tr><tr><td>训练课程</td><td>{course}</td></tr>
<tr><td>运行模式</td><td>{engine_mode}</td></tr><tr><td>估算骨骼帧率</td><td>{stats['estimated_fps']:.1f} fps</td></tr>
<tr><td>是否完整结束</td><td>{'是' if meta.get('finished') else '否'}</td></tr></table></div>
{action_section}
<div class="card"><h2>课程汇总</h2><table>
<tr><td>已产生数据的动作</td><td>{completed_actions} / {len(actions)}</td></tr>
<tr><td>已评分动作平均分</td><td>{course_average:.1f}</td></tr></table></div>
<div class="card"><div class="note">建议结合实时五维动作评分和治疗师观察共同评估训练效果。若有效关节比例偏低，请检查站位、遮挡和相机距离后重新训练。</div></div>
</div></body></html>"""

    output = session / "session_report.html"
    temporary = output.with_suffix(".html.tmp")
    temporary.write_text(document, encoding="utf-8")
    temporary.replace(output)
    return str(output)
