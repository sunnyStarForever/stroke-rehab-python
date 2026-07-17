import threading
import time
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from rehab_engine.course import Course, CourseAction, CourseRunner, RunnerState
from rehab_engine.scoring import (
    ScoreBridge,
    ScoreResult,
    OfflineReportRunner,
    ScoringCsvRecorder,
    ScoringSkeletonAdapter,
    _find_scoring_engine,
    _parse_score_result,
)


class ScoringSkeletonAdapterTests(unittest.TestCase):
    def test_coordinate_transform_and_original_joint_fallbacks(self):
        joints = [
            SimpleNamespace(x=float(i), y=float(i + 10), z=float(i + 20), valid=True)
            for i in range(22)
        ]
        for index in (5, 6, 10, 17, 21):
            joints[index].valid = False

        converted = ScoringSkeletonAdapter.convert(joints)
        self.assertEqual(converted[0], [0.0, -10.0, -20.0])
        self.assertEqual(converted[5], [4.5, -14.5, -24.5])
        self.assertEqual(converted[6], [5.0, -15.0, -25.0])
        self.assertEqual(converted[10], [7.0, -17.0, -27.0])
        self.assertEqual(converted[17], converted[16])
        self.assertEqual(converted[21], converted[20])
        self.assertEqual(ScoringSkeletonAdapter.valid_joint_count(joints), 17)

    def test_non_finite_and_explicit_invalid_joints_are_rejected(self):
        joints = [[1.0, 2.0, 3.0, True] for _ in range(22)]
        joints[0][0] = float("nan")
        joints[1][3] = False
        self.assertEqual(ScoringSkeletonAdapter.valid_joint_count(joints), 20)


class ScoreProtocolTests(unittest.TestCase):
    def test_nested_last_cycle_matches_score_server_contract(self):
        result = _parse_score_result({
            "status": "new_completed_cycle",
            "count": 4,
            "completed_count": 3,
            "last_cycle": {
                "overall_score": 88.5,
                "dimension_scores": {
                    "amplitude_score": 81,
                    "smoothness_score": 82,
                    "trunk_score": 83,
                    "symmetry_score": 84,
                    "rhythm_score": 85,
                },
            },
        })
        self.assertEqual(result.count, 4)
        self.assertEqual(result.completed_count, 3)
        self.assertEqual(result.overall_score, 88.5)
        self.assertEqual(
            [result.amplitude_score, result.smoothness_score, result.trunk_score,
             result.symmetry_score, result.rhythm_score],
            [81, 82, 83, 84, 85],
        )

    def test_runner_limits_count_jump_and_averages_only_completed_cycles(self):
        runner = CourseRunner()
        completed = []
        runner.on_action_completed = lambda action, reps, score: completed.append((reps, score))
        runner.start_course(Course(actions=[CourseAction(action_id="M1", target_reps=3)]))

        runner.on_score_updated(ScoreResult(status="waiting", count=9, overall_score=1))
        self.assertEqual(runner.state, RunnerState.TRAINING)
        runner.on_score_updated(ScoreResult(
            status="new_completed_cycle", count=9, overall_score=80))
        self.assertEqual(runner.state, RunnerState.TRAINING)
        runner.on_score_updated(ScoreResult(
            status="new_completed_cycle", count=9, overall_score=100))

        self.assertEqual(runner.state, RunnerState.FINISHED)
        self.assertEqual(completed, [(3, 90.0)])


class ActionArtifactTests(unittest.TestCase):
    def test_scoring_csv_matches_original_wide_contract(self):
        with tempfile.TemporaryDirectory() as tmp:
            recorder = ScoringCsvRecorder()
            self.assertTrue(recorder.start(tmp))
            joints = [[i + 0.1, i + 0.2, i + 0.3] for i in range(22)]
            self.assertTrue(recorder.append(7, joints))
            recorder.stop()
            path = Path(tmp) / "skeleton3d.csv"
            rows = path.read_text(encoding="utf-8").splitlines()
            header = rows[0].split(",")
            self.assertEqual(header[:4], [
                "frame_idx", "00_waist_x", "00_waist_y", "00_waist_z"])
            self.assertEqual(header[-3:], ["21_r_toe_x", "21_r_toe_y", "21_r_toe_z"])
            self.assertEqual(len(header), 67)
            self.assertEqual(rows[1].split(",")[:4], [
                "7", "0.100000", "0.200000", "0.300000"])

    def test_offline_runner_generates_real_report_from_shipped_m1_data(self):
        engine = _find_scoring_engine()
        self.assertIsNotNone(engine)
        source = engine / "data" / "processed" / "M1" / "skeleton3d.csv"
        self.assertTrue(source.exists())
        with tempfile.TemporaryDirectory() as tmp:
            runner = OfflineReportRunner()
            done = threading.Event()
            ready = []
            errors = []
            runner.on_ready = lambda path: (ready.append(path), done.set())
            runner.on_error = lambda message: (errors.append(message), done.set())
            self.assertTrue(runner.run(str(source), "M1", tmp, 20.0))
            self.assertTrue(done.wait(60.0), "offline report timed out in test")
            self.assertFalse(errors, errors)
            self.assertTrue(ready and Path(ready[0]).exists(), ready)


class ScoreBridgeIntegrationTests(unittest.TestCase):
    def test_shipped_server_is_found_and_async_frame_response_arrives(self):
        self.assertIsNotNone(_find_scoring_engine())
        bridge = ScoreBridge()
        received = []
        errors = []
        done = threading.Event()
        bridge.on_score_updated = lambda result: (received.append(result), done.set())
        bridge.on_error = errors.append
        try:
            self.assertTrue(bridge.start("M1", 20.0), errors)
            joints = [[float(i + 1), float(i + 2), float(i + 3), True] for i in range(22)]
            started = time.monotonic()
            self.assertTrue(bridge.submit_skeleton(1, 123456789, joints))
            self.assertLess(time.monotonic() - started, 0.5)
            self.assertTrue(done.wait(5.0), errors)
            self.assertTrue(received[0].status)
        finally:
            bridge.stop()

    def test_invalid_action_is_rejected_before_process_start(self):
        bridge = ScoreBridge()
        errors = []
        bridge.on_error = errors.append
        self.assertFalse(bridge.start("shoulder_basic"))
        self.assertIn("课程ID", errors[-1])


if __name__ == "__main__":
    unittest.main(verbosity=2)
