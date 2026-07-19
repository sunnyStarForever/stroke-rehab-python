"""
Main application window — FluentWindow with left navigation.
Replaces app/MainWindow.cpp.

Startup status panel shows engine mode and hardware availability.
"""

import sys
from datetime import datetime
from typing import Optional

from PyQt5.QtCore import Qt, QTimer, pyqtSignal
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

from rehab_engine.config_loader import load_pipeline_config, save_pipeline_config
from rehab_engine import logger
from rehab_engine.diagnostics import Diagnostics, run_diagnostics
from rehab_engine.event_routing import UserNotificationGate

from .pages.training_page import TrainingPage
from .pages.reports_page import ReportsPage
from .pages.settings_page import SettingsPage
from .pages.patient_home_page import PatientHomePage
from .dialogs import LogDialog, LogEntry, PerformanceDialog


class StrokeRehabWindow(FluentWindow):
    engine_log_received = pyqtSignal(str, str)

    def __init__(self, diagnostics: Optional[Diagnostics] = None, parent=None):
        setTheme(Theme.LIGHT)
        setThemeColor("#2563EB")
        super().__init__(parent)
        self.setWindowTitle("Stroke Rehab 康复训练系统")
        self.resize(1400, 880)
        self.setMinimumSize(1180, 760)

        # Store diagnostics for display
        self._diag = diagnostics
        self._closing = False
        self._allow_close = False
        self._log_entries = []
        self._log_dialog = None
        self._performance_dialog = None
        self._notification_gate = UserNotificationGate(cooldown_seconds=30.0)

        self._config = load_pipeline_config()
        self.engine_log_received.connect(self._on_engine_log)
        logger.set_callback(self.engine_log_received.emit)

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
        home = PatientHomePage(self._config, self)
        home.setObjectName("patientHomePage")
        self.addSubInterface(
            home,
            FIF.HOME, "首页",
            position=NavigationItemPosition.TOP,
        )

        training = TrainingPage(self._config, self)
        training.setObjectName("trainingPage")
        self.addSubInterface(
            training,
            FIF.PLAY, "训练",
            position=NavigationItemPosition.TOP,
        )

        reports = ReportsPage(self._config, self)
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
        training.shutdown_ready.connect(self._on_training_shutdown_ready)
        settings.course_changed.connect(training.set_course)
        settings.course_changed.connect(lambda _course_id: home.refresh_from_config())
        settings.debug_changed.connect(training.set_debug_enabled)
        settings.settings_applied.connect(lambda _path: home.refresh_from_config())
        settings.log_requested.connect(self.show_log_dialog)
        settings.performance_requested.connect(self.show_performance_dialog)
        home.course_selected.connect(self._open_course)
        home.report_requested.connect(self.navigate_to_reports)
        training.set_debug_enabled(self._config.ui_debug_enabled)

    def _open_course(self, course_id: str):
        training = self.findChild(TrainingPage)
        if training is None or not training.set_course(course_id):
            InfoBar.warning(
                "无法切换课程", "当前训练尚未结束，课程将在结束后生效。",
                duration=3500,
                position=InfoBarPosition.TOP_RIGHT,
                parent=self,
            )
            return
        self._config.selected_course_id = course_id
        try:
            save_pipeline_config(self._config)
        except OSError as exc:
            self._on_engine_log("WARN", f"课程偏好保存失败: {exc}")
        home = self.findChild(PatientHomePage)
        if home:
            home.refresh_from_config()
        self.switchTo(training)

    def navigate_to_reports(self, session_dir: str, csv_path: str):
        if self._closing:
            return
        """Async report loading — defer to next event loop tick to avoid blocking UI."""
        from PyQt5.QtCore import QTimer
        reports = self.findChild(ReportsPage)
        if reports:
            QTimer.singleShot(0, lambda: self._load_report_async(reports, session_dir, csv_path))

    def _load_report_async(self, reports, session_dir: str, csv_path: str):
        reports.load_session(session_dir, csv_path)
        self.switchTo(reports)

    # ================================================================
    # Engine log callback
    # ================================================================

    def _on_engine_log(self, level: str, message: str):
        tag = f"[{level}] {message[:150]}"
        print(tag, flush=True)
        entry = LogEntry(datetime.now().strftime("%H:%M:%S"), level, message)
        self._log_entries.append(entry)
        if len(self._log_entries) > 2000:
            del self._log_entries[:len(self._log_entries) - 2000]
        if self._log_dialog is not None:
            self._log_dialog.append_entry(entry)

        if level == "INFO" and (
                "state=NORMAL" in message or "RECOVERED" in message.upper()):
            self._notification_gate.recover(level, message)
        if self._notification_gate.should_notify(level, message):
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
                    f"同步: {s.get('sync_fps', 0.0):.1f}fps | "
                    f"Worker: {s.get('worker_fps', s.get('pair_fps', 0.0)):.1f}fps | "
                    f"已处理: {s['processed']} | "
                    f"丢弃: {s['dropped_pairs']}"
                )

        # Camera status (from training page)
        cam_status = training.camera_status()
        if cam_status:
            parts.append(cam_status)

        if parts:
            self.setStatusTip("  |  ".join(parts))
        if self._performance_dialog is not None and self._performance_dialog.isVisible():
            self._performance_dialog.set_snapshot(training.pipeline_stats())

    def show_log_dialog(self):
        if self._log_dialog is None:
            self._log_dialog = LogDialog(self)
        self._log_dialog.set_entries(self._log_entries)
        self._log_dialog.show()
        self._log_dialog.raise_()
        self._log_dialog.activateWindow()

    def show_performance_dialog(self):
        if self._performance_dialog is None:
            self._performance_dialog = PerformanceDialog(self)
        training = self.findChild(TrainingPage)
        if training:
            self._performance_dialog.set_snapshot(training.pipeline_stats())
        self._performance_dialog.show()
        self._performance_dialog.raise_()
        self._performance_dialog.activateWindow()

    # ================================================================
    # Shutdown
    # ================================================================

    def closeEvent(self, event):
        if self._allow_close:
            event.accept()
            return
        event.ignore()
        if self._closing:
            return
        self._closing = True
        print("\n[UI] Closing application safely...", flush=True)
        # The logger is process-global.  Disconnect the bound Qt signal before
        # asynchronous pipeline shutdown can outlive this window instance.
        logger.set_callback(None)
        self.setEnabled(False)
        reports = self.findChild(ReportsPage)
        if reports:
            reports.shutdown()
        training = self.findChild(TrainingPage)
        if training:
            try:
                training.shutdown()
                return
            except Exception as e:
                print(f"[UI] Shutdown error: {e}", flush=True)
        self._on_training_shutdown_ready()

    def _on_training_shutdown_ready(self):
        if not self._closing:
            return
        print("[UI] Application resources released", flush=True)
        self._allow_close = True
        QTimer.singleShot(0, self.close)


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
