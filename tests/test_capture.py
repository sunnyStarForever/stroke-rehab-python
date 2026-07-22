import gc
import unittest
import weakref

import numpy as np

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
    shape = (2, 3, 3) if source is FrameSource.RGB else (2, 3)
    dtype = np.uint8 if source is FrameSource.RGB else np.uint16
    return FrameEnvelope(source, np.ones(shape, dtype=dtype), 3, 2, timestamp,
                         arrival_ts_ns=timestamp + 10, frame_id=frame_id)


class DeviceConfigTests(unittest.TestCase):
    def test_openni_hardware_d2c_is_opt_in(self):
        self.assertFalse(DeviceConfig().enable_hardware_d2c)


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

    def test_timestamp_normalizer_preserves_image_and_device_time(self):
        frame = _frame(FrameSource.RGB, 1, 7)
        stamped = TimestampNormalizer.stamp(frame, 99, 123)
        self.assertEqual((stamped.host_ts_ns, stamped.device_ts_us), (99, 123))
        self.assertIs(stamped.image, frame.image)
        self.assertEqual(stamped.frame_id, 7)

    def test_latest_queue_releases_replaced_array(self):
        queue = LatestFrameQueue()
        image = np.ones((2, 2, 3), dtype=np.uint8)
        reference = weakref.ref(image)
        queue.push(image)
        del image
        queue.push(np.zeros((2, 2, 3), dtype=np.uint8))
        gc.collect()
        self.assertIsNone(reference())


class _NativeConfig:
    pass


class _Rgb:
    instances = []
    start_ok = True
    start_order = []

    def __init__(self):
        self.callback = None
        self.stopped = False
        self.__class__.instances.append(self)

    def set_on_status(self, callback):
        self.status_callback = callback

    def start(self, config, callback):
        self.start_order.append(type(self).__name__)
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
        _Rgb.start_order.clear()
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
        depth_image = np.array([[0, 10000, 65535]], dtype=np.uint16)
        rgb_image = np.zeros((1, 3, 3), dtype=np.uint8)
        depth.callback(
            depth_image, 3, 1, 100, 8, 1234, 0.0001,
            "DEPTH_100_UM", "depth", 130, "us", "normalized_device", "", 0,
        )
        rgb.callback(
            rgb_image, 3, 1, 105, 9, 0, 1.0, "MJPG", "rgb", 135,
            "ns", "native_monotonic", "", 0,
        )
        self.assertEqual(len(pairs), 1)
        self.assertEqual(pairs[0].delta_ns, 5)
        self.assertIs(pairs[0].rgb.image, rgb_image)
        self.assertEqual(pairs[0].rgb.image.dtype, np.uint8)
        self.assertTrue(pairs[0].rgb.image.flags.c_contiguous)
        self.assertFalse(pairs[0].rgb.image.flags.writeable)
        self.assertEqual(pairs[0].depth.image.dtype, np.uint16)
        self.assertEqual(pairs[0].depth.image.tolist(), [[0, 10000, 65535]])
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

    def test_invalid_native_array_contract_is_rejected(self):
        backend = NativeRgbDepthBackend(_Core, DeviceConfig(), SyncConfig())
        pairs = []
        self.assertTrue(backend.start(pairs.append))
        self.assertEqual(_Rgb.start_order, ["_Depth", "_Rgb"])
        depth = _Depth.instances[0]
        rgb = _Rgb.instances[0]
        depth.callback(
            np.ones((2, 2), dtype=np.uint16), 2, 2, 100, 1,
            1000, 0.001, "DEPTH_1_MM", "depth", 100,
            "us", "normalized_device", "", 0)
        rgb.callback(
            np.ones((2, 2, 3), dtype=np.uint16), 2, 2, 100, 1,
            0, 1.0, "MJPG", "rgb", 100,
            "us", "native_monotonic", "", 0)
        self.assertEqual(pairs, [])
        self.assertEqual(backend.sync_stats().rgb_pushed, 0)
        backend.stop()

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
