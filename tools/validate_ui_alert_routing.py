#!/usr/bin/env python3
"""Offscreen verification that performance logs never create InfoBars."""

from __future__ import annotations

import os
from unittest import mock

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("STROKE_REHAB_FORCE_STUB", "1")

from PyQt5.QtWidgets import QApplication

from ui.main_window import StrokeRehabWindow


def main() -> int:
    app = QApplication.instance() or QApplication([])
    window = StrokeRehabWindow()
    window._status_timer.stop()
    with mock.patch("ui.main_window.InfoBar.warning") as warning:
        performance = "[PERF] code=CALLBACK_P95_WARN source=depth state=WARN"
        window._on_engine_log("PERF", performance)
        window._on_engine_log("WARN", performance)
        if warning.call_count != 0:
            raise AssertionError("performance event created an InfoBar")
        fault = "[Camera] code=DEVICE_DISCONNECTED device stopped"
        window._on_engine_log("ERROR", fault)
        window._on_engine_log("ERROR", fault)
        if warning.call_count != 1:
            raise AssertionError("device event was not deduplicated")
        window._notification_gate.recover("ERROR", fault)
        window._on_engine_log("ERROR", fault)
        if warning.call_count != 2:
            raise AssertionError("recovered device fault did not notify again")
    window.close()
    app.processEvents()
    print("UI_ALERT_ROUTING=PASS performance=0 device_initial=1 device_after_recovery=1")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
