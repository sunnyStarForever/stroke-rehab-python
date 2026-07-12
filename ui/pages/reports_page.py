"""
Reports page — displays training reports and history.
Replaces app/pages/ReportPage.cpp.
"""

import json
import threading
from pathlib import Path
from datetime import datetime
from typing import Optional

from PyQt5.QtCore import Qt, pyqtSignal
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


class ReportsPage(QWidget):
    """Training reports and history browser."""

    report_loaded = pyqtSignal(int, str, str, str)
    history_loaded = pyqtSignal(int, object)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._session_dir = ""
        self._csv_path = ""
        self._history_entries = []
        self._report_generation = 0
        self._history_generation = 0
        self._closing = False

        self._init_ui()
        self.report_loaded.connect(self._on_report_loaded)
        self.history_loaded.connect(self._on_history_loaded)
        self._browser.setHtml(self._default_html())
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
            if need_gen:
                try:
                    path = Path(generate_session_report(session_dir, csv_path))
                    subtitle = f"报告文件：{path.name}"
                    html = path.read_text(encoding="utf-8", errors="ignore")
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
                        break
                else:
                    subtitle = "骨骼数据：" + (
                        "已就绪" if csv_path and Path(csv_path).exists() else "未找到")
            try:
                self.report_loaded.emit(generation, html, subtitle, error)
            except RuntimeError:
                pass

        threading.Thread(target=_do_load, name="report-load", daemon=True).start()

    def _on_report_loaded(self, generation: int, html: str,
                          subtitle: str, error: str):
        """Runs on the Qt thread through a queued signal."""
        if self._closing or generation != self._report_generation:
            return
        self._report_subtitle.setText(subtitle)
        self._browser.setHtml(html)
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
        roots = [
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

        for path, start_time, session_label in entries:
            name = session_label or Path(path).name
            if start_time:
                name = f"{start_time[:16].replace('T', ' ')}\n{name}"
            self._history_list.addItem(name)
        self._history_count.setText(f"{len(entries)} 次")

    def shutdown(self):
        """Invalidate outstanding worker results before the page is destroyed."""
        self._closing = True
        self._report_generation += 1
        self._history_generation += 1

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
            Path(path).write_text(self._browser.toHtml(), encoding="utf-8")
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
