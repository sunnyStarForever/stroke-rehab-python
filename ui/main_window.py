"""
Main application window — FluentWindow with left navigation.
Replaces app/MainWindow.cpp.

Startup status panel shows engine mode and hardware availability.
"""

import sys
from datetime import datetime
from typing import Optional

from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtWidgets import QApplication

from qfluentwidgets import (
    FluentWindow,
    FluentIcon as FIF,
    NavigationItemPosition,
    InfoBar,
    InfoBarPosition,
    Theme,
    setTheme,
    setThemeColor,
)

from rehab_engine.config_loader import load_pipeline_config
from rehab_engine._stub import logger
from rehab_engine.diagnostics import Diagnostics, run_diagnostics

from .pages.training_page import TrainingPage
from .pages.reports_page import ReportsPage
from .pages.settings_page import SettingsPage


class StrokeRehabWindow(FluentWindow):
    def __init__(self, diagnostics: Optional[Diagnostics] = None, parent=None):
        setTheme(Theme.LIGHT)
        setThemeColor("#2563EB")
        super().__init__(parent)
        self.setWindowTitle("Stroke Rehab 康复训练系统")
        self.resize(1400, 880)
        self.setMinimumSize(1180, 760)

        # Store diagnostics for display
        self._diag = diagnostics

        self._config = load_pipeline_config()
        logger.set_callback(self._on_engine_log)

        self._init_navigation()

        # ---- Status refresh timer (1s) ----
        self._status_timer = QTimer(self)
        self._status_timer.timeout.connect(self._on_status_tick)
        self._status_timer.start(1000)

        # ---- Show startup status bar ----
        self._show_startup_status()

    # ================================================================
    # Startup status
    # ================================================================

    def _show_startup_status(self):
        """Display engine mode and hardware status in the status bar on startup."""
        if not self._diag:
            self._diag = run_diagnostics(self._config)

        n_ok = sum(1 for i in self._diag.items if i.status == "OK")
        n_warn = sum(1 for i in self._diag.items if i.status == "WARN")
        n_err = sum(1 for i in self._diag.items if i.status == "ERROR")
        total = len(self._diag.items)

        status_text = f"系统诊断: {n_ok}项正常"
        if n_warn:
            status_text += f" / {n_warn}项警告"
        if n_err:
            status_text += f" / {n_err}项错误"
        status_text += f" (共{total}项)"

        self.setStatusTip(status_text)
        print(f"\n[UI] 状态栏: {status_text}", flush=True)

        # Pop up warnings for errors
        for err_item in self._diag.errors():
            InfoBar.error(
                title=err_item.name,
                content=err_item.detail + (
                    f"\n{err_item.hint}" if err_item.hint else ""),
                duration=8000,
                position=InfoBarPosition.TOP_RIGHT,
                parent=self,
            )

    def refresh_diagnostics(self):
        """Re-run diagnostics (called when settings are applied)."""
        self._diag = run_diagnostics(self._config)
        self._show_startup_status()

    # ================================================================
    # Navigation
    # ================================================================

    def _init_navigation(self):
        training = TrainingPage(self._config, self)
        training.setObjectName("trainingPage")
        self.addSubInterface(
            training,
            FIF.PLAY, "训练",
            position=NavigationItemPosition.TOP,
        )

        reports = ReportsPage(self)
        reports.setObjectName("reportsPage")
        self.addSubInterface(
            reports,
            FIF.DOCUMENT, "报告",
            position=NavigationItemPosition.TOP,
        )

        settings = SettingsPage(self._config, self)
        settings.setObjectName("settingsPage")
        self.addSubInterface(
            settings,
            FIF.SETTING, "设置",
            position=NavigationItemPosition.BOTTOM,
        )

        training.report_requested.connect(self.navigate_to_reports)

    def navigate_to_reports(self, session_dir: str, csv_path: str):
        reports = self.findChild(ReportsPage)
        if reports:
            reports.load_session(session_dir, csv_path)
            self.switchTo(reports)

    # ================================================================
    # Engine log callback
    # ================================================================

    def _on_engine_log(self, level: str, message: str):
        tag = f"[{level}] {message[:150]}"
        print(tag, flush=True)

        if level in ("WARN", "ERROR"):
            InfoBar.warning(
                level, message[:120],
                duration=3000,
                position=InfoBarPosition.BOTTOM_RIGHT,
                parent=self,
            )

    # ================================================================
    # Status timer
    # ================================================================

    def _on_status_tick(self):
        training = self.findChild(TrainingPage)
        if not training:
            return

        parts = []

        # Engine mode badge
        from rehab_engine import _STUB_MODE
        if _STUB_MODE:
            parts.append("🔶 STUB模式")
        else:
            parts.append("🔷 引擎模式")

        # Training status
        if training.is_training:
            s = training.pipeline_stats()
            if s:
                parts.append(
                    f"Pair: {s['pair_fps']:.1f}fps | "
                    f"已处理: {s['processed']} | "
                    f"丢弃: {s['dropped_pairs']}"
                )

        # Camera status (from training page)
        cam_status = training.camera_status()
        if cam_status:
            parts.append(cam_status)

        if parts:
            self.setStatusTip("  |  ".join(parts))

    # ================================================================
    # Shutdown
    # ================================================================

    def closeEvent(self, event):
        print("\n[UI] 正在关闭应用...", flush=True)
        training = self.findChild(TrainingPage)
        if training:
            training.shutdown()
        print("[UI] 应用已关闭", flush=True)
        super().closeEvent(event)


def main():
    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling)
    QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps)
    app = QApplication(sys.argv)

    # Diagnostics run in main.py before QApplication,
    # but as a fallback we also run here if called directly
    print("\n[UI] 启动 Stroke Rehab 界面...", flush=True)
    window = StrokeRehabWindow()
    window.show()

    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
