import unittest

from rehab_engine import DeviceConfig, SyncConfig
from rehab_engine.capture import (
    FrameEnvelope,
    FrameSource,
    FrameSynchronizer,
    LatestFrameQueue,
    NativeRgbDepthBackend,
    TimestampNormalizer,
)


def _frame(source, timestamp, frame_id=0):
    return FrameEnvelope(source, b"data", 640, 480, timestamp, frame_id=frame_id)


class FrameSynchronizerTests(unittest.TestCase):
    def test_nearest_match_removes_each_frame_once_and_preserves_signed_delta(self):
        sync = FrameSynchronizer(SyncConfig(match_threshold_ns=20, queue_size=5))
        pairs = []
        sync.set_on_pair_ready(pairs.append)
        sync.push_frame(_frame(FrameSource.DEPTH, 100, 1))
        sync.push_frame(_frame(FrameSource.DEPTH, 130, 2))
        sync.push_frame(_frame(FrameSource.RGB, 125, 3))
        sync.push_frame(_frame(FrameSource.RGB, 102, 4))
        self.assertEqual([pair.depth.frame_id for pair in pairs], [2, 1])
        self.assertEqual([pair.delta_ns for pair in pairs], [-5, 2])
        self.assertEqual(sync.stats().matched, 2)
        self.assertEqual(sync.stats().rgb_queued, 0)
        self.assertEqual(sync.stats().depth_queued, 0)

    def test_threshold_miss_waits_and_bounded_queue_trims_oldest(self):
        sync = FrameSynchronizer(SyncConfig(match_threshold_ns=5, queue_size=2))
        sync.push_frame(_frame(FrameSource.DEPTH, 10))
        sync.push_frame(_frame(FrameSource.RGB, 30))
        self.assertEqual(sync.stats().threshold_misses, 1)
        sync.push_frame(_frame(FrameSource.RGB, 31))
        sync.push_frame(_frame(FrameSource.RGB, 32))
        stats = sync.stats()
        self.assertEqual(stats.rgb_trimmed, 1)
        self.assertEqual(stats.rgb_queued, 2)

    def test_callback_runs_outside_lock_and_can_reenter(self):
        sync = FrameSynchronizer(SyncConfig(match_threshold_ns=10, queue_size=3))
        observed = []

        def callback(pair):
            observed.append(sync.stats().matched)
            sync.clear()

        sync.set_on_pair_ready(callback)
        sync.push_frame(_frame(FrameSource.DEPTH, 100))
        sync.push_frame(_frame(FrameSource.RGB, 100))
        self.assertEqual(observed, [1])


class QueueAndTimestampTests(unittest.TestCase):
    def test_latest_queue_replaces_backlog_and_counts_drop(self):
        queue = LatestFrameQueue()
        queue.push(1)
        queue.push(2)
        self.assertEqual(queue.pop_latest(0), 2)
        self.assertEqual((queue.pushed, queue.popped, queue.dropped), (2, 1, 1))

    def test_timestamp_normalizer_preserves_payload_and_device_time(self):
        frame = _frame(FrameSource.RGB, 1, 7)
        stamped = TimestampNormalizer.stamp(frame, 99, 123)
        self.assertEqual((stamped.host_ts_ns, stamped.device_ts_us), (99, 123))
        self.assertEqual((stamped.payload, stamped.frame_id), (b"data", 7))


class _NativeConfig:
    pass


class _Rgb:
    instances = []
    start_ok = True

    def __init__(self):
        self.callback = None
        self.stopped = False
        self.__class__.instances.append(self)

    def set_on_status(self, callback):
        self.status_callback = callback

    def start(self, config, callback):
        self.config = config
        self.callback = callback
        return self.start_ok

    def stop(self):
        self.stopped = True

    def is_running(self):
        return self.start_ok and not self.stopped


class _Depth(_Rgb):
    instances = []
    attested = True
    driver_running = True

    def real_depth_active(self):
        return self.attested

    def is_running(self):
        return self.driver_running and not self.stopped

    def hardware_d2c_active(self):
        return True


class _Core:
    DeviceConfig = _NativeConfig
    RgbCaptureV4L2 = _Rgb
    DepthCaptureOpenNI = _Depth


class _LegacyDepth(_Rgb):
    instances = []

    def hardware_d2c_active(self):
        return False


class _LegacyCore(_Core):
    DepthCaptureOpenNI = _LegacyDepth


class NativeBackendTests(unittest.TestCase):
    def setUp(self):
        _Rgb.instances.clear()
        _Depth.instances.clear()
        _LegacyDepth.instances.clear()
        _Rgb.start_ok = _Depth.start_ok = True
        _LegacyDepth.start_ok = True
        _Depth.attested = True
        _Depth.driver_running = True

    def test_separate_native_drivers_feed_python_synchronizer(self):
        backend = NativeRgbDepthBackend(
            _Core, DeviceConfig(rgb_device_index=3), SyncConfig(match_threshold_ns=20)
        )
        pairs = []
        self.assertTrue(backend.start(pairs.append))
        depth = _Depth.instances[0]
        rgb = _Rgb.instances[0]
        depth.callback(
            b"png", 640, 480, 100, 8,
            1234, 0.0001, "DEPTH_100_UM", "depth",
        )
        rgb.callback(b"jpg", 640, 480, 105, 9, 0, 1.0, "MJPG", "rgb")
        self.assertEqual(len(pairs), 1)
        self.assertEqual(pairs[0].delta_ns, 5)
        self.assertEqual(pairs[0].rgb.payload, b"jpg")
        self.assertEqual(pairs[0].depth.device_ts_us, 1234)
        self.assertEqual(pairs[0].depth.depth_unit_to_meter, 0.0001)
        self.assertEqual(pairs[0].depth.pixel_format_name, "DEPTH_100_UM")
        self.assertEqual(pairs[0].rgb.pixel_format_name, "MJPG")
        self.assertTrue(backend.hardware_d2c_active())
        self.assertEqual(rgb.config.rgb_device_path, "/dev/video3")
        backend.stop()
        self.assertTrue(rgb.stopped and depth.stopped)

    def test_partial_driver_start_is_rolled_back(self):
        _Depth.start_ok = False
        backend = NativeRgbDepthBackend(_Core, DeviceConfig(), SyncConfig())
        self.assertFalse(backend.start(lambda pair: None))
        self.assertTrue(_Rgb.instances[0].stopped)
        self.assertTrue(_Depth.instances[0].stopped)

    def test_false_hardware_attestation_is_rejected(self):
        _Depth.attested = False
        _Depth.driver_running = False
        backend = NativeRgbDepthBackend(_Core, DeviceConfig(), SyncConfig())
        self.assertFalse(backend.start(lambda pair: None))
        self.assertTrue(_Rgb.instances[0].stopped)
        self.assertTrue(_Depth.instances[0].stopped)

    def test_legacy_depth_driver_without_attestation_is_rejected(self):
        backend = NativeRgbDepthBackend(_LegacyCore, DeviceConfig(), SyncConfig())
        self.assertFalse(backend.start(lambda pair: None))
        self.assertTrue(_Rgb.instances[0].stopped)
        self.assertTrue(_LegacyDepth.instances[0].stopped)


if __name__ == "__main__":
    unittest.main(verbosity=2)
