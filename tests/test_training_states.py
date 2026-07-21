import os
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

os.environ.setdefault("STROKE_VOICE_ENABLED", "false")
if os.name != "nt" and not os.environ.get("DISPLAY"):
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt5.QtWidgets import QApplication

from rehab_engine._stub import PipelineConfig
from ui.pages.training_page import TrainingPage, TrainingState


APP = QApplication.instance() or QApplication(sys.argv)


class _FakePipeline:
    def __init__(self, session_dir: str):
        self.is_running = False
        self.is_stopping = False
        self.is_recording = False
        self.start_calls = 0
        self.record_calls = 0
        self.session_dir = session_dir
        self.camera_status = {"error": ""}
        self.preview = mock.Mock()
        self.preview.latest_frame.return_value = None

    def start(self):
        self.start_calls += 1
        self.is_running = True
        return True

    def start_recording(self, _root):
        self.record_calls += 1
        self.is_recording = True
        Path(self.session_dir).mkdir(parents=True, exist_ok=True)
        return self.session_dir

    def stop(self, on_complete=None):
        self.is_running = False
        self.is_recording = False
        if on_complete:
            on_complete(True, "stopped")


class StagedTrainingStateTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        config = PipelineConfig()
        config.voice.enabled = False
        self.page = TrainingPage(config)
        self.page._preview_timer.stop()
        self.fake = _FakePipeline(str(Path(self.tmp.name) / "session"))
        self.page._pipeline = self.fake
        self.page._capture_preflight_checks = lambda: ([], [])
        self.page._training_preflight_checks = lambda: []

    def tearDown(self):
        self.page._preview_timer.stop()
        self.page._training_timer.stop()
        self.page._voice.stop()
        self.page.deleteLater()
        APP.processEvents()
        self.tmp.cleanup()

    def _wait_for(self, predicate, timeout=2.0):
        deadline = time.monotonic() + timeout
        while not predicate() and time.monotonic() < deadline:
            APP.processEvents()
            time.sleep(0.01)
        self.assertTrue(predicate())

    def test_capture_precedes_training_and_pipeline_starts_once(self):
        self.assertEqual(self.page._state, TrainingState.IDLE)
        self.assertTrue(self.page._btn_capture.isEnabled())
        self.assertFalse(self.page._btn_start.isEnabled())

        self.page._on_start_capture()
        self._wait_for(lambda: self.page._state == TrainingState.CAPTURING)
        self.assertEqual(self.fake.start_calls, 1)
        self.assertEqual(self.fake.record_calls, 0)
        self.assertFalse(self.page._session_dir)
        self.assertTrue(self.page._btn_start.isEnabled())

        with mock.patch.object(self.page._course_runner, "start_course", return_value=True):
            self.page._on_start()
        self.assertEqual(self.fake.start_calls, 1)
        self.assertEqual(self.fake.record_calls, 1)
        self.assertEqual(self.page._state, TrainingState.TRAINING)

    def test_capture_frames_do_not_reach_scoring_or_csv(self):
        self.page._update_state(TrainingState.CAPTURING)
        bridge = mock.Mock()
        recorder = mock.Mock()
        self.page._score_bridge = bridge
        self.page._scoring_recorder = recorder
        self.page._on_pipeline_frame(mock.Mock())
        bridge.submit_skeleton.assert_not_called()
        recorder.append.assert_not_called()

    def test_failed_or_stale_capture_start_never_enables_training(self):
        self.page._start_generation = 4
        self.page._update_state(TrainingState.STARTING_CAPTURE)
        self.page._on_pipeline_started(3, True, "")
        self.assertFalse(self.fake.is_running)
        self.assertEqual(self.page._state, TrainingState.STARTING_CAPTURE)

        self.page._on_pipeline_started(4, False, "camera unavailable")
        self.assertEqual(self.page._state, TrainingState.IDLE)
        self.assertFalse(self.page._btn_start.isEnabled())

    def test_button_matrix_covers_capture_training_pause_finish_and_restart(self):
        self.fake.is_running = True
        cases = (
            (TrainingState.STARTING_CAPTURE, (False, False, False, False)),
            (TrainingState.CAPTURING, (False, True, False, True)),
            (TrainingState.TRAINING, (False, False, True, True)),
            (TrainingState.RESTING, (False, False, True, True)),
            (TrainingState.PAUSED, (False, False, True, True)),
            (TrainingState.STOPPING, (False, False, False, False)),
            (TrainingState.FINISHED, (True, False, False, False)),
        )
        for state, expected in cases:
            self.page._update_state(state)
            actual = (
                self.page._btn_capture.isEnabled(),
                self.page._btn_start.isEnabled(),
                self.page._btn_pause.isEnabled(),
                self.page._btn_stop.isEnabled(),
            )
            self.assertEqual(actual, expected, state.name)

    def test_unexpected_capture_stop_restores_idle_without_session(self):
        self.fake.is_running = False
        self.page._update_state(TrainingState.CAPTURING)
        self.page._refresh_preview()
        self.assertEqual(self.page._state, TrainingState.IDLE)
        self.assertFalse(self.page._session_dir)


if __name__ == "__main__":
    unittest.main(verbosity=2)
