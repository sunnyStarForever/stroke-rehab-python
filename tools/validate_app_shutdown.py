"""Run the real Qt event loop and request a normal application shutdown."""

from __future__ import annotations

import argparse
import os
import sys
import gc
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("STROKE_VOICE_ENABLED", "false")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from PyQt5.QtCore import QCoreApplication, QEvent, QTimer
from PyQt5.QtWidgets import QApplication
from PyQt5 import sip

def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--target", choices=("training", "reports", "settings", "home", "main"),
        default="main")
    args = parser.parse_args()
    app = QApplication.instance() or QApplication([])
    if args.target == "main":
        from ui.main_window import StrokeRehabWindow
        window = StrokeRehabWindow()
        from rehab_engine import logger
        logger.set_callback(None)
        window._status_timer.stop()
    else:
        from rehab_engine._stub import PipelineConfig
        config = PipelineConfig()
        config.voice.enabled = False
        if args.target == "training":
            from ui.pages.training_page import TrainingPage
            window = TrainingPage(config)
        elif args.target == "reports":
            from ui.pages.reports_page import ReportsPage
            window = ReportsPage()
        elif args.target == "settings":
            from ui.pages.settings_page import SettingsPage
            window = SettingsPage(config)
        else:
            from ui.pages.patient_home_page import PatientHomePage
            window = PatientHomePage(config)
    window.show()
    QTimer.singleShot(1000, window.close)
    QTimer.singleShot(20000, lambda: app.exit(2))
    exit_code = int(app.exec_())
    window.deleteLater()
    QCoreApplication.sendPostedEvents(None, QEvent.DeferredDelete)
    app.processEvents()
    if not sip.isdeleted(window):
        sip.delete(window)
    del window
    gc.collect()
    QCoreApplication.sendPostedEvents(None, QEvent.DeferredDelete)
    app.processEvents()
    if not sip.isdeleted(app):
        sip.delete(app)
    print(f"APP_SHUTDOWN_TARGET={args.target} EXIT={exit_code}", flush=True)
    from main import _exit_after_qt
    _exit_after_qt(exit_code)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
