"""
Headless UI import test — verifies all modules load without errors.
Run: /d/miniforge3/envs/stroke-rehab/python.exe test_ui.py
"""
import os
import sys
from pathlib import Path
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

if os.name != "nt" and not os.environ.get("DISPLAY"):
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt5.QtWidgets import QApplication

app = QApplication.instance() or QApplication(sys.argv)

print("=" * 50)
print("Stage 3 UI Import Test")
print("=" * 50)

# 1. Widget imports
print("\n[1] Widgets")
from ui.widgets.preview_widget import PreviewWidget
print("  PreviewWidget: OK")

from ui.widgets.score_panel import ScorePanel
from rehab_engine.scoring import ScoreResult
sp = ScorePanel()
sp.set_score(ScoreResult(overall_score=85.5, amplitude_score=90.0,
                       smoothness_score=80.0, trunk_score=88.0,
                       symmetry_score=75.0, rhythm_score=82.0))
print("  ScorePanel: OK")

from ui.widgets.emg_panel import EmgPanel
print("  EmgPanel: OK")

# 2. Page imports
print("\n[2] Pages")
from ui.pages.training_page import TrainingPage
from rehab_engine._stub import PipelineConfig
# TrainingPage requires a parent - can't instantiate without QApplication
print("  TrainingPage import: OK")

from ui.pages.reports_page import ReportsPage
print("  ReportsPage import: OK")

from ui.pages.settings_page import SettingsPage
print("  SettingsPage import: OK")

from ui.pages.patient_home_page import CourseCard, PatientHomePage
print("  PatientHomePage import: OK")

from ui.dialogs import LogDialog, LogEntry, PerformanceDialog
print("  Runtime dialogs import: OK")

# 3. Main window import
print("\n[3] MainWindow")
from ui.main_window import StrokeRehabWindow
print("  StrokeRehabWindow import: OK")

# 4. Full QApplication test
print("\n[4] QApplication launch test")

# Test pages instantiate
cfg = PipelineConfig()
tp = TrainingPage(cfg)  # No parent — standalone test
print("  TrainingPage instantiated: OK")

rp = ReportsPage()
print("  ReportsPage instantiated: OK")

settings_page = SettingsPage(cfg)
print("  SettingsPage instantiated: OK")

home_page = PatientHomePage(cfg)
assert home_page.findChildren(CourseCard)
print("  PatientHomePage instantiated: OK")

log_dialog = LogDialog()
log_dialog.append_entry(LogEntry("12:00:00", "INFO", "pipeline ready"))
log_dialog.append_entry(LogEntry("12:00:01", "ERROR", "camera failed"))
assert len(log_dialog.entries) == 2
log_dialog._level_combo.setCurrentText("错误")
assert "camera failed" in log_dialog._log_edit.toPlainText()
assert "pipeline ready" not in log_dialog._log_edit.toPlainText()
performance_dialog = PerformanceDialog()
performance_dialog.set_snapshot({"rgb_fps": 30.0, "pose_ms": 12.5})
assert performance_dialog._labels["rgb_fps"].text() == "30.0"
assert performance_dialog._labels["pose_ms"].text() == "12.5 ms"
print("  Runtime dialogs instantiated: OK")

# Test main window instantiation (without show)
mw = StrokeRehabWindow()
print("  StrokeRehabWindow instantiated: OK")
assert mw.findChild(TrainingPage) is not None
assert mw.findChild(ReportsPage) is not None
assert mw.findChild(SettingsPage) is not None
assert mw.findChild(PatientHomePage) is not None
course_card = mw.findChildren(CourseCard)[0]
with patch("ui.main_window.save_pipeline_config"):
    mw._open_course(course_card.course.course_id)
assert mw._config.selected_course_id == course_card.course.course_id
assert mw.stackedWidget.currentWidget() is mw.findChild(TrainingPage)
print("  Navigation pages: 4")

print()
print("=" * 50)
print("Stage 3: ALL IMPORT TESTS PASSED")
print("=" * 50)
tp.shutdown()
mw.close()
app.quit()
