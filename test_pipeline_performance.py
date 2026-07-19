import gc
import tempfile
import time
import unittest
import weakref
from pathlib import Path

import numpy as np

from rehab_engine.event_routing import UserNotificationGate, event_category
from rehab_engine.performance import CallbackPerformanceMonitor, PerformanceState
from rehab_engine.recorder import RecordingFrame, RgbDepthVideoRecorder
from rehab_engine._stub import PipelineConfig
from rehab_engine.sensor_pipeline import SensorPipeline


class CallbackPerformanceMonitorTests(unittest.TestCase):
    def monitor(self, **overrides):
        options = dict(
            window_seconds=5.0, min_samples=30, normal_p95_ms=8.0,
            warn_p95_ms=10.0, critical_p95_ms=25.0,
            warn_sustain_seconds=5.0, low_fps_sustain_seconds=3.0,
            recovery_seconds=5.0, target_fps={"rgb": 30.0},
        )
        options.update(overrides)
        return CallbackPerformanceMonitor(**options)

    @staticmethod
    def feed(monitor, start, stop, callback_ms, step=0.02):
        events = []
        now = start
        while now <= stop:
            monitor.observe_arrival("rgb", now)
            event = monitor.observe_callback("rgb", callback_ms, now)
            if event:
                events.append(event)
            now += step
        return events

    def test_warmup_normal_and_single_spike_do_not_warn(self):
        monitor = self.monitor()
        self.feed(monitor, 0.0, 0.62, 2.0)
        self.assertEqual(monitor.snapshot("rgb", 0.62).state, "NORMAL")
        event = monitor.observe_callback("rgb", 40.0, 0.64)
        self.assertIsNone(event)
        self.assertNotEqual(monitor.snapshot("rgb", 0.64).state, "CRITICAL")

    def test_sustained_warn_is_transition_only(self):
        monitor = self.monitor(target_fps={"rgb": 0.0})
        events = self.feed(monitor, 0.0, 9.0, 12.0, 0.05)
        warnings = [event for event in events if event.code == "CALLBACK_P95_WARN"]
        self.assertEqual(len(warnings), 1)
        self.assertEqual(warnings[0].state, PerformanceState.WARN.value)

    def test_critical_is_immediate_after_warmup_and_recovers_with_hysteresis(self):
        monitor = self.monitor(target_fps={"rgb": 0.0})
        self.feed(monitor, 0.0, 0.62, 30.0)
        critical = monitor.evaluate("rgb", 0.63)
        # The transition may have been emitted by the final observe call.
        self.assertEqual(monitor.snapshot("rgb", 0.63).state, "CRITICAL")
        self.assertIsNone(critical)
        events = self.feed(monitor, 1.0, 12.0, 2.0, 0.02)
        self.assertIn("PERFORMANCE_RECOVERING", [event.code for event in events])
        self.assertIn("PERFORMANCE_RECOVERED", [event.code for event in events])
        self.assertEqual(monitor.snapshot("rgb", 12.0).state, "NORMAL")

    def test_low_raw_fps_becomes_critical(self):
        monitor = self.monitor()
        events = self.feed(monitor, 0.0, 8.0, 2.0, 0.10)
        self.assertIn("RAW_FPS_CRITICAL", [event.code for event in events])


class LogRoutingTests(unittest.TestCase):
    def test_performance_never_notifies_but_device_error_is_deduplicated(self):
        gate = UserNotificationGate(cooldown_seconds=30.0)
        perf = "[PERF] code=CALLBACK_P95_WARN source=depth"
        self.assertEqual(event_category("WARN", perf), "performance")
        self.assertFalse(gate.should_notify("WARN", perf, 0.0))
        error = "[Camera] code=DEVICE_DISCONNECTED device stopped"
        self.assertTrue(gate.should_notify("ERROR", error, 1.0))
        self.assertFalse(gate.should_notify("ERROR", error, 2.0))
        self.assertTrue(gate.should_notify("ERROR", error, 32.0))

        gate.recover("ERROR", error)
        self.assertTrue(gate.should_notify("ERROR", error, 33.0))

    def test_recording_open_failure_still_notifies(self):
        gate = UserNotificationGate()
        self.assertTrue(gate.should_notify(
            "ERROR", "code=RECORDING_OPEN_FAILED cannot open video", 0.0))


class AsyncRecorderTests(unittest.TestCase):
    def test_bounded_queue_owns_array_references_until_drain(self):
        with tempfile.TemporaryDirectory() as directory:
            recorder = RgbDepthVideoRecorder()
            self.assertTrue(recorder.start(
                directory, 30, 16, 12, record_rgb=True,
                record_depth=False, queue_capacity=2))
            original_write = recorder._write_frame

            def slow_write(item, cv2):
                time.sleep(0.02)
                return original_write(item, cv2)

            recorder._write_frame = slow_write
            rgb = np.zeros((12, 16, 3), dtype=np.uint8)
            reference = weakref.ref(rgb)
            for index in range(12):
                recorder.submit(RecordingFrame(rgb, None, index, index, index))
            del rgb
            gc.collect()
            self.assertIsNotNone(reference())
            recorder.stop(drain_timeout_sec=2.0)
            stats = recorder.stats()
            self.assertEqual(stats.received, 12)
            self.assertGreater(stats.dropped, 0)
            self.assertEqual(stats.received, stats.written + stats.dropped + stats.failed)
            self.assertTrue(Path(stats.rgb_path).exists())
            self.assertTrue(Path(stats.metadata_path).exists())
            gc.collect()
            self.assertIsNone(reference())


class PipelineMetricBoundaryTests(unittest.TestCase):
    def test_raw_sync_worker_and_pose_use_independent_counters(self):
        pipeline = SensorPipeline(PipelineConfig())
        pipeline._last_perf_time = time.monotonic() - 2.0
        pipeline._rgb_since_last = 60
        pipeline._depth_since_last = 58
        pipeline._sync_since_last = 54
        pipeline._worker_since_last = 30
        pipeline._pose_since_last = 5
        pipeline._update_performance()
        stats = pipeline.performance_stats()
        self.assertGreater(stats["raw_rgb_fps"], stats["raw_depth_fps"])
        self.assertGreater(stats["raw_depth_fps"], stats["sync_fps"])
        self.assertGreater(stats["sync_fps"], stats["worker_fps"])
        self.assertGreater(stats["worker_fps"], stats["pose_fps"])
        self.assertEqual(stats["pair_fps"], stats["worker_fps"])


if __name__ == "__main__":
    unittest.main()
