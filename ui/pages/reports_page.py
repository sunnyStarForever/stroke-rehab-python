"""
Reports page — displays training reports and history.
Replaces app/pages/ReportPage.cpp.
"""

import json
import threading
from pathlib import Path
from datetime import datetime
from typing import Optional

from PyQt5.QtCore import Qt, pyqtSignal, QEvent, QTimer
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QFileDialog, QSplitter, QLabel,
)
from PyQt5.QtGui import QDesktopServices
from PyQt5.QtCore import QUrl

from qfluentwidgets import (
    CardWidget, SimpleCardWidget,
    PrimaryPushButton, PushButton,
    TitleLabel, SubtitleLabel, BodyLabel, StrongBodyLabel, CaptionLabel,
    TextBrowser, ListWidget, ScrollArea, InfoBar, InfoBarPosition,
)
from ..theme import COLORS, PAGE_STYLE, pill_style
from rehab_engine.reporting import generate_session_report
from ..report_images import adapt_report_images


class ReportsPage(QWidget):
    """Training reports and history browser."""

    report_loaded = pyqtSignal(int, str, str, str, str)
    history_loaded = pyqtSignal(int, object)

    def __init__(self, config=None, parent=None):
        super().__init__(parent)
        self._config = config
        self._session_dir = ""
        self._csv_path = ""
        self._history_entries = []
        self._report_generation = 0
        self._history_generation = 0
        self._closing = False
        self._raw_html = ""
        self._report_base_dir = Path()
        self._last_report_width = -1
        self._resize_timer = QTimer(self)
        self._resize_timer.setSingleShot(True)
        self._resize_timer.setInterval(80)
        self._resize_timer.timeout.connect(self._apply_responsive_html)

        self._init_ui()
        self.report_loaded.connect(self._on_report_loaded)
        self.history_loaded.connect(self._on_history_loaded)
        self._set_report_html(self._default_html())
        self._load_history()

    def _init_ui(self):
        self.setStyleSheet(PAGE_STYLE)
        root = QVBoxLayout(self)
        root.setContentsMargins(20, 16, 20, 16)
        root.setSpacing(14)

        page_header = QHBoxLayout()
        heading = QVBoxLayout()
        eyebrow = QLabel("TRAINING INSIGHTS")
        eyebrow.setObjectName("pageEyebrow")
        page_title = QLabel("训练报告")
        page_title.setObjectName("pageTitle")
        page_hint = QLabel("回顾每次训练结果，持续观察动作质量与康复趋势")
        page_hint.setObjectName("pageSubtitle")
        heading.addWidget(eyebrow)
        heading.addWidget(page_title)
        heading.addWidget(page_hint)
        page_header.addLayout(heading)
        page_header.addStretch()
        root.addLayout(page_header)

        content = QSplitter(Qt.Horizontal, self)
        content.setHandleWidth(6)

        # --- Left: history list ---
        left_card = SimpleCardWidget(self)
        left_card.setMinimumWidth(260)
        left_card.setMaximumWidth(340)
        left_layout = QVBoxLayout(left_card)
        left_layout.setContentsMargins(14, 14, 14, 14)
        left_layout.setSpacing(10)

        history_header = QHBoxLayout()
        history_title = QLabel("训练记录")
        history_title.setObjectName("sectionTitle")
        self._history_count = QLabel("0 次")
        self._history_count.setStyleSheet(pill_style("primary"))
        history_header.addWidget(history_title)
        history_header.addStretch()
        history_header.addWidget(self._history_count)
        left_layout.addLayout(history_header)
        history_hint = QLabel("选择一条记录查看详细分析")
        history_hint.setObjectName("sectionHint")
        left_layout.addWidget(history_hint)
        self._history_list = ListWidget(self)
        self._history_list.setAlternatingRowColors(False)
        self._history_list.setSpacing(4)
        self._history_list.itemClicked.connect(self._on_history_selected)
        left_layout.addWidget(self._history_list, 1)

        # --- Right: report browser ---
        right_card = CardWidget(self)
        right_layout = QVBoxLayout(right_card)
        right_layout.setContentsMargins(16, 14, 16, 14)
        right_layout.setSpacing(10)

        header = QHBoxLayout()
        report_heading = QVBoxLayout()
        self._report_title = TitleLabel("报告概览")
        self._report_subtitle = CaptionLabel("尚未选择训练记录")
        self._report_subtitle.setStyleSheet(f"color:{COLORS['muted']};")
        report_heading.addWidget(self._report_title)
        report_heading.addWidget(self._report_subtitle)
        header.addLayout(report_heading, 1)
        self._btn_trend = PushButton("康复趋势")
        self._btn_trend.clicked.connect(self._show_trends)
        header.addWidget(self._btn_trend)

        self._btn_save = PushButton("保存报告")
        self._btn_save.setEnabled(False)
        self._btn_save.clicked.connect(self._save_report)
        self._btn_folder = PushButton("打开文件夹")
        self._btn_folder.setEnabled(False)
        self._btn_folder.clicked.connect(self._open_folder)
        header.addWidget(self._btn_save)
        header.addWidget(self._btn_folder)
        right_layout.addLayout(header)

        self._browser = TextBrowser(self)
        self._browser.setOpenExternalLinks(True)
        self._browser.viewport().installEventFilter(self)
        right_layout.addWidget(self._browser, 1)

        content.addWidget(left_card)
        content.addWidget(right_card)
        content.setStretchFactor(0, 0)
        content.setStretchFactor(1, 1)
        content.setSizes([290, 900])
        root.addWidget(content, 1)

    def load_session(self, session_dir: str, csv_path: str):
        """Load a session report — heavy I/O runs in background thread."""
        self._session_dir = session_dir
        self._csv_path = csv_path
        self._report_title.setText(f"训练报告 — {Path(session_dir).name}")
        self._btn_save.setEnabled(True)
        self._btn_folder.setEnabled(Path(session_dir).exists())
        self._report_subtitle.setText("正在后台生成/载入报告…")
        self._report_generation += 1
        generation = self._report_generation

        # ── Report generation in background (avoids UI freeze) ──
        sd = Path(session_dir)
        rp = sd / "session_report.html"
        need_gen = (not rp.exists() and csv_path and Path(csv_path).exists())

        def _do_load():
            subtitle = ""
            error = ""
            report_dir = str(sd)
            if need_gen:
                try:
                    path = Path(generate_session_report(session_dir, csv_path))
                    subtitle = f"报告文件：{path.name}"
                    html = path.read_text(encoding="utf-8", errors="ignore")
                    report_dir = str(path.parent)
                except Exception as exc:
                    html = self._default_html()
                    subtitle = "报告生成失败"
                    error = str(exc)
            else:
                html = self._default_html()
                for hf in ([rp] if rp.exists() else list(sd.rglob("*.html"))):
                    if hf.is_file():
                        html = hf.read_text(encoding="utf-8", errors="ignore")
                        subtitle = f"报告文件：{hf.name}"
                        report_dir = str(hf.parent)
                        break
                else:
                    subtitle = "骨骼数据：" + (
                        "已就绪" if csv_path and Path(csv_path).exists() else "未找到")
            try:
                self.report_loaded.emit(generation, html, subtitle, error, report_dir)
            except RuntimeError:
                pass

        threading.Thread(target=_do_load, name="report-load", daemon=True).start()

    def _on_report_loaded(self, generation: int, html: str,
                          subtitle: str, error: str, report_dir: str):
        """Runs on the Qt thread through a queued signal."""
        if self._closing or generation != self._report_generation:
            return
        self._report_subtitle.setText(subtitle)
        self._set_report_html(html, Path(report_dir))
        if error:
            InfoBar.error("报告生成失败", error, duration=6000,
                          position=InfoBarPosition.BOTTOM_RIGHT, parent=self)
        self._load_history()

    def _load_history(self):
        """Scan session directories off the Qt thread."""
        self._history_generation += 1
        generation = self._history_generation

        def _scan():
            entries = self._scan_history_entries()
            try:
                self.history_loaded.emit(generation, entries)
            except RuntimeError:
                pass

        threading.Thread(target=_scan, name="history-scan", daemon=True).start()

    def _scan_history_entries(self):
        entries = []
        project_root = Path(__file__).resolve().parents[2]
        configured_root = Path(
            str(getattr(self._config, "record_path", "recordings")))
        if not configured_root.is_absolute():
            configured_root = project_root / configured_root
        roots = [
            configured_root,
            configured_root / "output",
            configured_root / "sessions",
            project_root / "recordings/sessions",
            project_root / "records/sessions",
            project_root / "records/output",
            Path("recordings/sessions"),
            Path("records/sessions"),
            Path("records/output"),
        ]
        seen = set()
        for records_dir in roots:
            if not records_dir.exists():
                continue
            session_dirs = {p.parent for p in records_dir.rglob("skeleton_3d.csv")}
            session_dirs.update(p.parent for p in records_dir.rglob("meta.json"))
            session_dirs.update(p.parent for p in records_dir.rglob("session_ui_meta.json"))
            for session in session_dirs:
                session = session.resolve()
                if session in seen:
                    continue
                seen.add(session)
                meta_file = session / "session_ui_meta.json"
                if not meta_file.exists():
                    meta_file = session / "meta.json"
                start_time = ""
                session_label = ""
                if meta_file.exists():
                    try:
                        meta = json.loads(meta_file.read_text(encoding="utf-8"))
                        start_time = meta.get("start_time", meta.get("end_time", ""))
                        session_label = " · ".join(filter(None, [
                            meta.get("patient_name", ""), meta.get("course_name", "")]))
                    except Exception:
                        pass
                if not start_time:
                    start_time = datetime.fromtimestamp(
                        session.stat().st_mtime).isoformat(timespec="minutes")
                entries.append((str(session), start_time, session_label))
        entries.sort(key=lambda entry: entry[1] or entry[0], reverse=True)
        entries = entries[:50]  # Keep last 50
        return entries

    def _on_history_loaded(self, generation: int, entries):
        if self._closing or generation != self._history_generation:
            return
        self._history_entries = entries
        self._history_list.clear()
        refresh_trends = not self._session_dir

        for path, start_time, session_label in entries:
            name = session_label or Path(path).name
            if start_time:
                name = f"{start_time[:16].replace('T', ' ')}\n{name}"
            self._history_list.addItem(name)
        self._history_count.setText(f"{len(entries)} 次")

        if refresh_trends:
            self._show_trends()

    def _show_trends(self):
        self._session_dir = ""
        self._csv_path = ""
        self._report_title.setText("康复趋势")
        self._report_subtitle.setText("基于最近训练记录的连续康复观察")
        self._btn_save.setEnabled(True)
        self._btn_folder.setEnabled(False)
        self._browser.setHtml(self._trend_html())

    def _session_snapshot(self, session_dir: str, start_time: str, label: str) -> dict:
        session = Path(session_dir)
        meta = {}
        summary = {}
        for name in ("session_ui_meta.json", "meta.json"):
            path = session / name
            if path.exists():
                try:
                    meta = json.loads(path.read_text(encoding="utf-8"))
                    break
                except Exception:
                    pass
        summary_path = session / "course_summary.json"
        if summary_path.exists():
            try:
                summary = json.loads(summary_path.read_text(encoding="utf-8"))
            except Exception:
                summary = {}
        actions = summary.get("actions", []) if isinstance(summary, dict) else []
        scores = [
            float(item.get("average_score", 0.0) or 0.0)
            for item in actions
            if float(item.get("average_score", 0.0) or 0.0) > 0.0
        ]
        actual = sum(int(item.get("actual_reps", 0) or 0) for item in actions)
        target = sum(int(item.get("target_reps", 0) or 0) for item in actions)
        completion = (actual / target * 100.0) if target > 0 else 0.0
        elapsed = int(meta.get("elapsed_seconds", 0) or 0)
        return {
            "session": session,
            "start": start_time or meta.get("start_time", ""),
            "label": label or meta.get("course_name", "") or session.name,
            "score": sum(scores) / len(scores) if scores else 0.0,
            "completion": completion,
            "actual": actual,
            "target": target,
            "elapsed": elapsed,
            "finished": bool(meta.get("finished", False)),
        }

    def _trend_html(self) -> str:
        snapshots = [
            self._session_snapshot(path, start, label)
            for path, start, label in self._history_entries[:12]
        ]
        if not snapshots:
            return self._default_html()
        recent = snapshots[:6]
        avg_score = sum(item["score"] for item in recent) / len(recent)
        avg_completion = sum(item["completion"] for item in recent) / len(recent)
        total_minutes = sum(item["elapsed"] for item in recent) // 60
        rows = []
        cards = []
        for item in snapshots:
            score = max(0, min(100, item["score"]))
            completion = max(0, min(100, item["completion"]))
            date = item["start"][:16].replace("T", " ")
            rows.append(
                f"<tr><td>{date}</td><td>{item['label']}</td>"
                f"<td>{item['elapsed']//60} 分钟</td><td>{item['actual']}/{item['target']}</td>"
                f"<td>{score:.1f}</td><td>{completion:.0f}%</td></tr>"
            )
            cards.append(
                f"<div class='session'><div><b>{date}</b><span>{item['label']}</span></div>"
                f"<div class='bar'><i style='width:{score:.0f}%'></i></div>"
                f"<em>{score:.1f}</em></div>"
            )
        return f"""<html><head><meta charset="utf-8"><style>
body{{font-family:'Segoe UI','Microsoft YaHei',sans-serif;background:#F8FAFD;color:#344054;padding:28px;}}
h1{{color:#172033;font-size:28px;margin:0 0 8px;}} h2{{color:#172033;font-size:18px;margin-top:28px;}}
.hint{{color:#667085;line-height:1.7;}} .grid{{display:grid;grid-template-columns:repeat(3,1fr);gap:14px;margin:20px 0;}}
.metric{{background:white;border:1px solid #E4EAF2;border-radius:14px;padding:18px;}}
.metric span{{display:block;color:#667085;font-size:13px;}} .metric b{{font-size:30px;color:#2563EB;}}
.panel{{background:white;border:1px solid #E4EAF2;border-radius:14px;padding:18px;margin-top:14px;}}
.session{{display:grid;grid-template-columns:220px 1fr 70px;gap:12px;align-items:center;margin:12px 0;}}
.session span{{display:block;color:#667085;font-size:12px;margin-top:3px;}}
.bar{{height:10px;background:#E9EEF5;border-radius:999px;overflow:hidden;}} .bar i{{display:block;height:100%;background:#2563EB;}}
table{{width:100%;border-collapse:collapse;background:white;border-radius:12px;overflow:hidden;}}
th,td{{text-align:left;border-bottom:1px solid #E4EAF2;padding:10px;font-size:13px;}} th{{color:#667085;background:#F4F7FB;}}
</style></head><body>
<h1>康复趋势总览</h1>
<p class="hint">这里汇总最近训练记录，用来观察训练坚持度、动作完成率和综合评分的变化。单次医学判断仍应结合治疗师评估。</p>
<div class="grid">
<div class="metric"><span>近 6 次平均评分</span><b>{avg_score:.1f}</b></div>
<div class="metric"><span>近 6 次平均完成率</span><b>{avg_completion:.0f}%</b></div>
<div class="metric"><span>近 6 次训练总时长</span><b>{total_minutes}</b><span>分钟</span></div>
</div>
<div class="panel"><h2>评分走势</h2>{''.join(cards)}</div>
<div class="panel"><h2>训练明细</h2><table><tr><th>时间</th><th>训练</th><th>时长</th><th>完成</th><th>评分</th><th>完成率</th></tr>{''.join(rows)}</table></div>
</body></html>"""

    def shutdown(self):
        """Invalidate outstanding worker results before the page is destroyed."""
        self._closing = True
        self._report_generation += 1
        self._history_generation += 1
        self._resize_timer.stop()

    def eventFilter(self, watched, event):
        if watched is self._browser.viewport() and event.type() == QEvent.Resize:
            self._resize_timer.start()
        return super().eventFilter(watched, event)

    def _set_report_html(self, html: str, base_dir: Optional[Path] = None):
        self._raw_html = html
        self._report_base_dir = base_dir or Path()
        self._last_report_width = -1
        self._apply_responsive_html()

    def _apply_responsive_html(self):
        if not self._raw_html or self._closing:
            return
        width = max(1, self._browser.viewport().width() - 48)
        if width == self._last_report_width:
            return
        self._last_report_width = width
        adapted = adapt_report_images(
            self._raw_html, width,
            self._report_base_dir if str(self._report_base_dir) else None)
        scrollbar = self._browser.verticalScrollBar()
        position = scrollbar.value()
        self._browser.setHtml(adapted)
        scrollbar.setValue(position)

    def _on_history_selected(self, item):
        row = self._history_list.row(item) if item else -1
        if 0 <= row < len(self._history_entries):
            path = Path(self._history_entries[row][0])
            csv = path / "skeleton_3d.csv"
            self.load_session(str(path), str(csv) if csv.exists() else "")

    def _save_report(self):
        path, _ = QFileDialog.getSaveFileName(
            self, "保存报告", "training_report.html", "HTML (*.html)")
        if path:
            Path(path).write_text(self._raw_html or self._browser.toHtml(), encoding="utf-8")
            InfoBar.success("已保存", f"报告已保存到 {path}",
                            position=InfoBarPosition.BOTTOM_RIGHT, parent=self)

    def _open_folder(self):
        if self._session_dir and Path(self._session_dir).exists():
            QDesktopServices.openUrl(QUrl.fromLocalFile(self._session_dir))

    def _default_html(self) -> str:
        return """<html><head><meta charset="utf-8"><style>
body{font-family:'Segoe UI','Microsoft YaHei',sans-serif;background:#F8FAFD;color:#344054;padding:42px;}
section{background:white;border:1px solid #E4EAF2;border-radius:16px;padding:36px;margin:40px auto;max-width:720px;}
h1{color:#172033;font-size:28px;}h2{color:#17324D;}p{font-size:15px;line-height:1.8;}
.eyebrow{color:#2563EB;font-size:12px;font-weight:700;letter-spacing:1px;}
.hint{color:#667085;}.steps{background:#F4F7FB;border-radius:10px;padding:16px 20px;}
.score{color:#27AE60;font-weight:700;font-size:24px;}
</style></head><body><section>
<div class="eyebrow">REPORT CENTER</div>
<h1>从一次训练记录开始</h1>
<p class="hint">选择左侧历史训练，即可在这里查看动作完成度、综合评分和详细分析。</p>
<div class="steps"><p>① 完成一次康复训练</p><p>② 从左侧选择训练记录</p><p>③ 查看或保存完整报告</p></div>
</section></body></html>"""
