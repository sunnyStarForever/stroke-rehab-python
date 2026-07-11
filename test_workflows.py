"""Integration checks for the training workflow added around the UI."""

import json
import tempfile
import time
import unittest
from pathlib import Path

from rehab_engine._stub import PipelineConfig
from rehab_engine.config_loader import _load_yaml, save_pipeline_config
from rehab_engine.course import Course, CourseAction, CourseRunner, RunnerState
from rehab_engine.reporting import generate_session_report
from rehab_engine.scoring import ScoreResult
from rehab_engine.sensor_pipeline import SensorPipeline


class CourseRunnerWorkflowTests(unittest.TestCase):
    def test_pause_and_resume_preserve_rest_countdown(self):
        runner = CourseRunner()
        course = Course(
            course_id="test",
            course_name="test",
            actions=[
                CourseAction(action_id="M1", name_cn="A", target_reps=1, rest_sec_after=2),
                CourseAction(action_id="M2", name_cn="B", target_reps=1),
            ],
        )
        self.assertTrue(runner.start_course(course))
        runner.on_score_updated(ScoreResult(count=1, completed_count=1, overall_score=80))
        self.assertEqual(runner.state, RunnerState.RESTING)
        self.assertTrue(runner.pause_course())
        remaining = runner.rest_remaining_sec
        time.sleep(1.1)
        self.assertEqual(runner.state, RunnerState.PAUSED)
        self.assertEqual(runner.rest_remaining_sec, remaining)
        self.assertTrue(runner.resume_course())
        deadline = time.monotonic() + 2.5
        while runner.current_action_index != 1 and time.monotonic() < deadline:
            time.sleep(0.05)
        self.assertEqual(runner.current_action_index, 1)
        self.assertEqual(runner.state, RunnerState.TRAINING)
        runner.stop_course()


class PipelineWorkflowTests(unittest.TestCase):
    def test_recording_pause_skips_frames_and_resume_continues(self):
        config = PipelineConfig()
        config.device.rgb_fps = 30
        pipeline = SensorPipeline(config)
        with tempfile.TemporaryDirectory() as tmp:
            self.assertTrue(pipeline.start())
            session = pipeline.start_recording(tmp)
            time.sleep(0.25)
            self.assertTrue(pipeline.pause_recording())
            paused_frames = pipeline.recording_stats()["frames"]
            time.sleep(0.20)
            self.assertEqual(pipeline.recording_stats()["frames"], paused_frames)
            self.assertTrue(pipeline.resume_recording())
            time.sleep(0.20)
            self.assertGreater(pipeline.recording_stats()["frames"], paused_frames)
            pipeline.stop_recording()
            pipeline.stop()
            self.assertTrue((Path(session) / "skeleton_3d.csv").exists())

    def test_session_report_is_generated_from_recording(self):
        config = PipelineConfig()
        pipeline = SensorPipeline(config)
        with tempfile.TemporaryDirectory() as tmp:
            pipeline.start()
            session = Path(pipeline.start_recording(tmp))
            time.sleep(0.20)
            pipeline.stop_recording()
            pipeline.stop()
            (session / "session_ui_meta.json").write_text(
                json.dumps({
                    "patient_name": "测试对象",
                    "course_name": "测试课程",
                    "elapsed_seconds": 12,
                    "finished": True,
                    "engine_mode": "stub",
                }, ensure_ascii=False),
                encoding="utf-8",
            )
            report = Path(generate_session_report(
                str(session), str(session / "skeleton_3d.csv")))
            self.assertTrue(report.exists())
            content = report.read_text(encoding="utf-8")
            self.assertIn("测试对象", content)
            self.assertIn("有效关节比例", content)


class ConfigWorkflowTests(unittest.TestCase):
    def test_fallback_yaml_parser_handles_nesting_and_empty_strings(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "device.yaml"
            path.write_text(
                'device:\n  uri: ""\n  enabled: true\n  size:\n    width: 640\n',
                encoding="utf-8",
            )
            data = _load_yaml(path)
            self.assertEqual(data["device"]["uri"], "")
            self.assertTrue(data["device"]["enabled"])
            self.assertEqual(data["device"]["size"]["width"], 640)

    def test_user_config_can_be_persisted(self):
        config = PipelineConfig()
        config.selected_course_id = "lower_limb_balance_transfer_basic"
        config.patient_name = "匿名测试"
        with tempfile.TemporaryDirectory() as tmp:
            output = save_pipeline_config(config, Path(tmp) / "config.json")
            data = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(data["patient_name"], "匿名测试")
            self.assertEqual(data["selected_course_id"], config.selected_course_id)


if __name__ == "__main__":
    unittest.main(verbosity=2)
