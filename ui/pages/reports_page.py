"""
Reports page — displays training reports and history.
Replaces app/pages/ReportPage.cpp.
"""

import json
from pathlib import Path
from datetime import datetime
from typing import Optional

from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QFileDialog, QSplitter,
)
from PyQt5.QtGui import QDesktopServices
from PyQt5.QtCore import QUrl

from qfluentwidgets import (
    CardWidget, SimpleCardWidget,
    PrimaryPushButton, PushButton,
    TitleLabel, SubtitleLabel, BodyLabel, StrongBodyLabel, CaptionLabel,
    TextBrowser, ListWidget, ScrollArea, InfoBar, InfoBarPosition,
)


class ReportsPage(QWidget):
    """Training reports and history browser."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._session_dir = ""
        self._csv_path = ""
        self._history_entries = []

        self._init_ui()

    def _init_ui(self):
        root = QHBoxLayout(self)
        root.setContentsMargins(16, 12, 16, 12)
        root.setSpacing(12)

        # --- Left: history list ---
        left_card = SimpleCardWidget(self)
        left_card.setFixedWidth(280)
        left_layout = QVBoxLayout(left_card)
        left_layout.setContentsMargins(14, 14, 14, 14)
        left_layout.setSpacing(10)

        left_layout.addWidget(StrongBodyLabel("训练记录"))
        self._history_list = ListWidget(self)
        self._history_list.setAlternatingRowColors(True)
        self._history_list.itemClicked.connect(self._on_history_selected)
        left_layout.addWidget(self._history_list, 1)

        # --- Right: report browser ---
        right_card = CardWidget(self)
        right_layout = QVBoxLayout(right_card)
        right_layout.setContentsMargins(16, 14, 16, 14)
        right_layout.setSpacing(10)

        header = QHBoxLayout()
        self._report_title = TitleLabel("训练报告")
        header.addWidget(self._report_title, 1)

        self._btn_save = PushButton("保存报告")
        self._btn_save.clicked.connect(self._save_report)
        self._btn_folder = PushButton("打开文件夹")
        self._btn_folder.clicked.connect(self._open_folder)
        header.addWidget(self._btn_save)
        header.addWidget(self._btn_folder)
        right_layout.addLayout(header)

        self._browser = TextBrowser(self)
        self._browser.setOpenExternalLinks(True)
        right_layout.addWidget(self._browser, 1)

        root.addWidget(left_card)
        root.addWidget(right_card, 1)

    def load_session(self, session_dir: str, csv_path: str):
        """Load a session report from disk."""
        self._session_dir = session_dir
        self._csv_path = csv_path
        self._report_title.setText(
            f"训练报告 — {Path(session_dir).name}")

        # Try to find report HTML files
        sd = Path(session_dir)
        html_content = self._default_html()

        # Look for offline_action_report.html in actions subdir
        for html_file in list(sd.rglob("*.html")):
            if html_file.is_file():
                html_content = html_file.read_text(encoding="utf-8", errors="ignore")
                break

        self._browser.setHtml(html_content)

        # Refresh history
        self._load_history()

    def _load_history(self):
        """Scan records directory for past sessions."""
        self._history_list.clear()
        records_dir = Path("records/sessions")
        if not records_dir.exists():
            records_dir = Path("records/output")
        if not records_dir.exists():
            return

        entries = []
        for session in sorted(records_dir.rglob("session_*"), reverse=True):
            if session.is_dir():
                meta_file = session / "meta.json"
                if meta_file.exists():
                    try:
                        meta = json.loads(meta_file.read_text())
                        entries.append((str(session), meta.get("start_time", "")))
                    except Exception:
                        entries.append((str(session), ""))
        entries = entries[:50]  # Keep last 50

        for path, start_time in entries:
            name = f"{Path(path).name}"
            if start_time:
                name = f"{start_time[:10]} {Path(path).name}"
            self._history_list.addItem(name)

    def _on_history_selected(self, item):
        if item:
            text = item.text()
            # Search for matching session
            for p in Path("records").rglob("session_*"):
                if p.name in text and p.is_dir():
                    csv = p / "skeleton_3d.csv"
                    self.load_session(str(p), str(csv) if csv.exists() else "")
                    break

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
body{font-family:'Segoe UI','Microsoft YaHei',sans-serif;background:#F4F8FB;color:#1F2933;padding:28px;}
section{background:white;border-radius:16px;padding:24px;margin-bottom:16px;}
h1{color:#2F80ED;}h2{color:#17324D;}p{font-size:16px;line-height:1.7;}
.score{color:#27AE60;font-weight:700;font-size:24px;}
</style></head><body><section>
<h1>训练报告</h1>
<p>当前课程训练数据将在此显示。</p>
<p>完成训练后，系统会自动生成详细的动作分析和评分报告。</p>
<p>您也可以从左侧训练记录中选择历史训练进行查看。</p>
</section></body></html>"""