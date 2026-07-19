"""Integration checks for the training workflow added around the UI."""

import json
import threading
import tempfile
import time
import unittest
from pathlib import Path

import cv2
import numpy as np

from rehab_engine._stub import PipelineConfig
from rehab_engine.config_loader import _load_yaml, save_pipeline_config
from rehab_engine.course import Course, CourseAction, CourseRepository, CourseRunner, RunnerState
from rehab_engine.reporting import generate_session_report
from rehab_engine.inference import BoundingBox2D, Keypoint2D, PoseInferenceResult
from rehab_engine.scoring import ScoreResult
from rehab_engine.sensor_pipeline import RecordingOptions, SensorPipeline


class CourseRunnerWorkflowTests(unittest.TestCase):
    def test_stop_cancels_pending_rest_timer_without_advancing(self):
        runner = CourseRunner()
        course = Course(
            course_id="stop-race",
            actions=[
                CourseAction(action_id="M1", target_reps=1, rest_sec_after=1),
                CourseAction(action_id="M2", target_reps=1),
            ],
        )
        runner.start_course(course)
        runner.on_score_updated(ScoreResult(count=1, completed_count=1, overall_score=80))
        runner.stop_course()
        time.sleep(1.2)
        self.assertEqual(runner.state, RunnerState.IDLE)
        self.assertEqual(runner.current_action_index, -1)

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
    def test_pair_queue_discards_oldest_and_pose_runs_on_first_frame(self):
        config = PipelineConfig()
        config.pose.max_pair_queue = 2
        pipeline = SensorPipeline(config)
        pipeline._enqueue_pair({"id": 1})
        pipeline._enqueue_pair({"id": 2})
        pipeline._enqueue_pair({"id": 3})
        self.assertEqual(pipeline._pair_queue.get_nowait()["id"], 2)
        self.assertEqual(pipeline._pair_queue.get_nowait()["id"], 3)
        self.assertEqual(pipeline.performance_stats()["dropped_pairs"], 1)

        class Estimator:
            def __init__(self):
                self.calls = 0

            def infer(self, _image, box):
                self.calls += 1
                points = [Keypoint2D(100 + i, 100 + i, 0.9, 0.9, True) for i in range(26)]
                return PoseInferenceResult(points, box, True, 26, 0.9, 0.0, 1.0)

        estimator = Estimator()
        pipeline._pose_models_ready = True
        pipeline._python_inference = True
        pipeline._pose_estimator = estimator
        ok, encoded = cv2.imencode(".jpg", np.zeros((240, 320, 3), dtype=np.uint8))
        self.assertTrue(ok)
        pipeline._infer_pose_full(encoded.tobytes(), None, 6)
        self.assertEqual(estimator.calls, 1)
        pipeline._infer_pose_full(encoded.tobytes(), None, 6)
        self.assertEqual(estimator.calls, 1)
        config.pose.enable_pose_reuse = False
        pipeline._infer_pose_full(encoded.tobytes(), None, 6)
        self.assertEqual(estimator.calls, 2)

    def test_record_pairs_debug_rejects_untrusted_stub_depth(self):
        config = PipelineConfig()
        config.record_pairs = True
        config.device.rgb_width = 64
        config.device.rgb_height = 48
        config.device.depth_width = 64
        config.device.depth_height = 48
        with tempfile.TemporaryDirectory() as tmp:
            config.record_path = tmp
            pipeline = SensorPipeline(config)
            self.assertTrue(pipeline.start())
            time.sleep(0.12)
            stopped = threading.Event()
            pipeline.stop(lambda *_: stopped.set())
            self.assertTrue(stopped.wait(3.0))
            sessions = [path for path in Path(tmp).iterdir() if path.is_dir()]
            self.assertEqual(len(sessions), 1)
            rows = (sessions[0] / "pairs.csv").read_text(encoding="utf-8").splitlines()
            self.assertEqual(len(rows), 1)
            self.assertFalse(list(sessions[0].glob("pair_*_rgb.png")))
            self.assertFalse(list(sessions[0].glob("pair_*_depth_raw_u16.png")))
            self.assertFalse(list(sessions[0].glob("pair_*_depth_aligned_u16.png")))

    def test_async_stop_returns_immediately_and_waits_for_cleanup(self):
        pipeline = SensorPipeline(PipelineConfig())
        completed = threading.Event()
        result = []
        with tempfile.TemporaryDirectory() as tmp:
            self.assertTrue(pipeline.start())
            pipeline.start_recording(tmp)
            time.sleep(0.15)
            started = time.monotonic()
            pipeline.stop(lambda ok, message: (result.append((ok, message)), completed.set()))
            self.assertLess(time.monotonic() - started, 0.1)
            self.assertTrue(completed.wait(3.0))
            self.assertTrue(result[0][0], result[0][1])
            self.assertFalse(pipeline.is_running)
            self.assertFalse(pipeline.is_recording)
            self.assertFalse(pipeline.is_stopping)
            self.assertTrue(pipeline.start())
            restarted = threading.Event()
            pipeline.stop(lambda *_: restarted.set())
            self.assertTrue(restarted.wait(3.0))

    def test_recording_pause_skips_frames_and_resume_continues(self):
        config = PipelineConfig()
        config.device.rgb_fps = 30
        pipeline = SensorPipeline(config)
        with tempfile.TemporaryDirectory() as tmp:
            self.assertTrue(pipeline.start())
            session = pipeline.start_recording(tmp)
            time.sleep(0.25)
            self.assertTrue(pipeline.pause_recording())
            time.sleep(0.10)
            paused_frames = pipeline.recording_stats()["rgb_frames"]
            time.sleep(0.20)
            self.assertEqual(pipeline.recording_stats()["rgb_frames"], paused_frames)
            self.assertTrue(pipeline.resume_recording())
            time.sleep(0.20)
            self.assertGreater(pipeline.recording_stats()["rgb_frames"], paused_frames)
            self.assertEqual(pipeline.recording_stats()["frames"], 0)
            self.assertEqual(pipeline.recording_stats()["depth_frames"], 0)
            pipeline.stop_recording()
            pipeline.stop()
            self.assertTrue((Path(session) / "skeleton_3d.csv").exists())
            self.assertGreater((Path(session) / "rgb.mp4").stat().st_size, 0)
            meta = json.loads((Path(session) / "meta.json").read_text(encoding="utf-8"))
            self.assertTrue(meta["record_rgb"])
            self.assertFalse(meta["record_depth"])
            self.assertGreater(meta["saved_rgb_frames"], 0)
            self.assertEqual(Path(session).parent.parent.name, "output")

    def test_optional_depth_video_rejects_untrusted_stub_depth(self):
        config = PipelineConfig()
        config.device.rgb_width = 320
        config.device.rgb_height = 240
        pipeline = SensorPipeline(config)
        with tempfile.TemporaryDirectory() as tmp:
            self.assertTrue(pipeline.start())
            session = Path(
                pipeline.start_recording(
                    RecordingOptions(tmp, True, False, True)
                )
            )
            time.sleep(0.20)
            pipeline.stop_recording()
            pipeline.stop()
            self.assertTrue((session / "depth.avi").exists())
            meta = json.loads((session / "meta.json").read_text(encoding="utf-8"))
            self.assertEqual(meta["saved_depth_frames"], 0)
            self.assertEqual(meta["saved_skeleton_frames"], 0)

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
            action_report = session / "actions" / "01_M1_m01" / "report" / "offline_action_report.html"
            action_report.parent.mkdir(parents=True)
            action_report.write_text("<html>M1</html>", encoding="utf-8")
            (session / "course_summary.json").write_text(
                json.dumps({
                    "course_name": "测试课程",
                    "actions": [{
                        "action_id": "M1",
                        "name_cn": "测试动作",
                        "target_reps": 8,
                        "actual_reps": 8,
                        "average_score": 86.5,
                        "report_path": str(action_report),
                    }],
                }, ensure_ascii=False),
                encoding="utf-8",
            )
            report = Path(generate_session_report(
                str(session), str(session / "skeleton_3d.csv")))
            self.assertTrue(report.exists())
            content = report.read_text(encoding="utf-8")
            self.assertIn("测试对象", content)
            self.assertIn("有效关节比例", content)
            self.assertIn("测试动作", content)
            self.assertIn("86.5", content)
            self.assertIn("查看动作报告", content)


class ConfigWorkflowTests(unittest.TestCase):
    def test_course_repository_preserves_original_validation_sort_and_aliases(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "courses.json"
            path.write_text(json.dumps({"courses": [{
                "course_id": "upper_limb_shoulder_rom_basic",
                "course_name": "肩部",
                "actions": [
                    {"order": 2, "action_id": "M8", "target_reps": 3},
                    {"order": 1, "action_id": "M7", "target_reps": 4},
                ],
            }]}), encoding="utf-8")
            repo = CourseRepository()
            self.assertTrue(repo.load(str(path)), repo.last_error)
            self.assertEqual(
                [action.action_id for action in repo.courses[0].actions], ["M7", "M8"])
            self.assertIsNotNone(repo.find_by_id("shoulder_basic"))

            path.write_text(json.dumps({"courses": [{
                "course_id": "bad", "course_name": "bad",
                "actions": [{"order": 1, "action_id": "M11", "target_reps": 1}],
            }]}), encoding="utf-8")
            self.assertFalse(repo.load(str(path)))
            self.assertEqual(repo.courses, [])
            self.assertIn("M1-M10", repo.last_error)

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
        config.patient_id = "P-TEST"
        config.patient_gender = "女"
        config.patient_age = 61
        config.patient_diagnosis = "上肢康复测试"
        with tempfile.TemporaryDirectory() as tmp:
            output = save_pipeline_config(config, Path(tmp) / "config.json")
            data = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(data["patient_name"], "匿名测试")
            self.assertEqual(data["patient_id"], "P-TEST")
            self.assertEqual(data["patient_gender"], "女")
            self.assertEqual(data["patient_age"], 61)
            self.assertEqual(data["patient_diagnosis"], "上肢康复测试")
            self.assertEqual(data["selected_course_id"], config.selected_course_id)


if __name__ == "__main__":
    unittest.main(verbosity=2)
