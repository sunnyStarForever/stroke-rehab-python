"""Reliable session-level HTML report generation for recorded skeleton CSV files."""

import csv
import html
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional


def _read_session_meta(session_dir: Path) -> Dict:
    for name in ("session_ui_meta.json", "meta.json"):
        path = session_dir / name
        if path.exists():
            try:
                return json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
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
    stats = analyze_skeleton_csv(csv_path)
    meta = _read_session_meta(session)

    patient = html.escape(meta.get("patient_name") or "未命名训练对象")
    course = html.escape(meta.get("course_name") or "康复训练")
    engine_mode = "模拟模式" if meta.get("engine_mode") == "stub" else "真实引擎"
    minutes, seconds = divmod(int(meta.get("elapsed_seconds") or stats["duration_seconds"]), 60)
    validity = stats["valid_ratio"] * 100
    confidence = stats["mean_confidence"] * 100
    quality_text = "良好" if validity >= 90 else "一般" if validity >= 70 else "需要检查"
    generated_at = datetime.now().strftime("%Y-%m-%d %H:%M")

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
td{{padding:10px;border-bottom:1px solid #EEF2F6;}} td:first-child{{color:#667085;width:34%;}}
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
<div class="card"><div class="note">建议结合实时五维动作评分和治疗师观察共同评估训练效果。若有效关节比例偏低，请检查站位、遮挡和相机距离后重新训练。</div></div>
</div></body></html>"""

    output = session / "session_report.html"
    temporary = output.with_suffix(".html.tmp")
    temporary.write_text(document, encoding="utf-8")
    temporary.replace(output)
    return str(output)
