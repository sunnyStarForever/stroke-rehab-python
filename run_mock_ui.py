"""Open the desktop UI with demo data and no camera capture.

This launcher is intended for layout review: it prepares a small set of mock
session summaries so the reports page can render rehabilitation trends.
"""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QApplication


ROOT = Path(__file__).resolve().parent
MOCK_ROOT = ROOT / "mock_recordings" / "sessions"
MOCK_CONFIG = ROOT / "mock_config.user.json"


def _write_demo_sessions() -> None:
    MOCK_ROOT.mkdir(parents=True, exist_ok=True)
    now = datetime.now()
    courses = ["上肢主动活动", "平衡转移训练", "步态辅助训练"]
    for index in range(8):
        start = now - timedelta(days=index * 2)
        session = MOCK_ROOT / f"mock_session_{index + 1:02d}"
        session.mkdir(parents=True, exist_ok=True)
        elapsed = 9 * 60 + index * 35
        base_score = 68 + index * 2.8
        actions = []
        for order, action_id in enumerate(("M2", "M5", "M7"), start=1):
            target = 10
            actual = min(target, 6 + index // 2 + order)
            actions.append({
                "action_id": action_id,
                "movement_id": f"movement_{action_id}",
                "name_cn": f"模拟动作 {action_id}",
                "target_reps": target,
                "actual_reps": actual,
                "average_score": min(96.0, base_score + order * 1.7),
                "action_dir": "",
                "csv_path": "",
                "report_path": "",
            })
        meta = {
            "patient_id": "DEMO",
            "patient_name": "模拟患者",
            "course_name": courses[index % len(courses)],
            "start_time": start.isoformat(timespec="seconds"),
            "elapsed_seconds": elapsed,
            "finished": index % 3 != 0,
            "end_time": (start + timedelta(seconds=elapsed)).isoformat(timespec="seconds"),
            "engine_mode": "stub",
        }
        summary = {
            "course_name": meta["course_name"],
            "start_time": meta["start_time"],
            "end_time": meta["end_time"],
            "status": "finished" if meta["finished"] else "stopped",
            "actions": actions,
        }
        (session / "session_ui_meta.json").write_text(
            json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
        (session / "course_summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        (session / "skeleton_3d.csv").write_text(
            "frame,waist_x,waist_y,waist_z\n1,0,0,1\n", encoding="utf-8")


def _write_mock_config() -> None:
    payload = {
        "record_path": str((ROOT / "mock_recordings").resolve()),
        "selected_course_id": "",
        "patient_name": "模拟患者",
        "patient_id": "DEMO",
        "ui_debug_enabled": True,
        "ui_theme": "light",
        "emg": {
            "enabled": True,
            "capture_backend": "serial",
            "serial_device": "MOCK",
        },
    }
    MOCK_CONFIG.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    os.environ["STROKE_USER_CONFIG"] = str(MOCK_CONFIG)


def main() -> int:
    _write_demo_sessions()
    _write_mock_config()

    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling)
    QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps)
    app = QApplication(sys.argv)

    from ui.main_window import StrokeRehabWindow

    window = StrokeRehabWindow()
    window.setWindowTitle("Stroke Rehab UI Mock Preview")
    window.show()
    return app.exec_()


if __name__ == "__main__":
    raise SystemExit(main())
