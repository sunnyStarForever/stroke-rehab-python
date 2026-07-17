import struct
import time
import unittest
from dataclasses import dataclass

from rehab_engine import EmgConfig
from rehab_engine.emg import (
    BleGattCapture,
    EmgBleNotifyParser,
    EmgChannelFeature,
    EmgFeatureFrame,
    EmgFeatureProcessor,
    EmgFusionBuffer,
    EmgBluetoothDevice,
    EmgBluetoothScanResult,
    EmgManager,
    EmgMuscleState,
    EmgRawChunk,
    EmgRpmsgProtocol,
    SerialEmgCapture,
)


def _payload(sequence=7):
    values = []
    for index in range(25):
        values.extend((index * 10, -index * 10))
    return struct.pack("<I50i", sequence, *values)


class EmgBleNotifyParserTests(unittest.TestCase):
    def test_parses_original_204_byte_layout_and_timestamps(self):
        parser = EmgBleNotifyParser()
        parsed = parser.parse(_payload(42), 1_000_000_000, 1000)
        self.assertIsNotNone(parsed)
        samples, sequence = parsed
        self.assertEqual(sequence, 42)
        self.assertEqual(len(samples), 25)
        self.assertEqual(samples[0].host_ts_ns, 976_000_000)
        self.assertEqual(samples[-1].host_ts_ns, 1_000_000_000)
        self.assertEqual(samples[9].channels, (90, -90))

    def test_rejects_wrong_size_and_tracks_sequence_semantics(self):
        parser = EmgBleNotifyParser()
        self.assertIsNone(parser.parse(b"bad", 1, 1000))
        self.assertTrue(parser.observe_sequence(0xFFFFFFFE).accepted)
        wrapped = parser.observe_sequence(1)
        self.assertTrue(wrapped.accepted)
        self.assertEqual(wrapped.dropped, 2)
        self.assertTrue(parser.observe_sequence(1).duplicate)
        self.assertTrue(parser.observe_sequence(0).out_of_order)


class EmgRpmsgProtocolTests(unittest.TestCase):
    def test_config_and_raw_chunk_match_v2_header_contract(self):
        config = EmgConfig(raw_chunk_samples=25, channel_count=2)
        config_packet = EmgRpmsgProtocol.pack_config(config)
        self.assertEqual(EmgRpmsgProtocol.HEADER.size, 28)
        self.assertEqual(len(config_packet), 40)
        self.assertEqual(config_packet[:4], b"EMG1")

        chunk = EmgRawChunk(123456, 8, 1000, 2, tuple(range(50)))
        raw_packet = EmgRpmsgProtocol.pack_raw_chunk(chunk)
        self.assertIsNotNone(raw_packet)
        self.assertEqual(len(raw_packet), 228)
        header = EmgRpmsgProtocol.HEADER.unpack_from(raw_packet)
        self.assertEqual(header[2], EmgRpmsgProtocol.RAW_CHUNK)
        self.assertEqual(header[8], 25)
        self.assertEqual(header[9], 200)

    def test_feature_packet_validation_preserves_fields(self):
        payload = b"".join(
            (
                EmgRpmsgProtocol.CHANNEL_FEATURE.pack(10.0, 0.1, 0.2, 0.3, 1),
                EmgRpmsgProtocol.CHANNEL_FEATURE.pack(20.0, 0.2, 0.3, 0.4, 3),
            )
        )
        packet = EmgRpmsgProtocol._header(
            EmgRpmsgProtocol.FEATURE,
            seq=77,
            host_ts_ns=123456,
            sample_rate_hz=1000,
            channel_count=2,
            payload_bytes=len(payload),
        ) + payload
        frame = EmgRpmsgProtocol.parse_feature(packet)
        self.assertIsNotNone(frame)
        self.assertEqual(frame.seq, 77)
        self.assertEqual(frame.channels[1].state, EmgMuscleState.FATIGUE)
        self.assertIsNone(EmgRpmsgProtocol.parse_feature(packet[:-1]))


@dataclass
class _Device:
    name: str = "ESP32_EMG_TEST"


class _Scanner:
    @staticmethod
    async def discover(timeout=4.0):
        return [_Device()]


class _Client:
    instances = []

    def __init__(self, target):
        self.target = target
        self.callbacks = {}
        self.commands = []
        self.__class__.instances.append(self)

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_args):
        return False

    async def start_notify(self, uuid, callback):
        self.callbacks[uuid] = callback

    async def stop_notify(self, uuid):
        self.callbacks.pop(uuid, None)

    async def write_gatt_char(self, uuid, value, response=True):
        self.commands.append(bytes(value))
        config = self.config
        status = self.callbacks[config.ble_status_tx_uuid]
        if value == b"START_EMG":
            status(1, bytearray(b"EMG_START_OK"))
            self.callbacks[config.ble_notify_char_uuid](2, bytearray(_payload(9)))
        elif value == b"STOP_EMG":
            status(1, bytearray(b"EMG_STOP_OK"))


class BleGattCaptureTests(unittest.TestCase):
    def setUp(self):
        _Client.instances.clear()
        self.config = EmgConfig(
            enabled=True,
            mode="real",
            capture_backend="bluez",
            ble_service_uuid="service",
            ble_command_rx_uuid="command",
            ble_status_tx_uuid="status",
            ble_notify_char_uuid="data",
            ble_command_timeout_ms=1000,
        )
        _Client.config = self.config

    def test_start_stop_handshake_gates_and_publishes_data(self):
        samples = []
        statuses = []
        capture = BleGattCapture(self.config, _Scanner, _Client)
        self.assertTrue(capture.start(samples.append, statuses.append))
        deadline = time.monotonic() + 2.0
        while len(samples) < 25 and time.monotonic() < deadline:
            time.sleep(0.01)
        self.assertEqual(len(samples), 25)
        self.assertTrue(capture.is_connected)
        capture.stop()
        self.assertEqual(_Client.instances[0].commands, [b"START_EMG", b"STOP_EMG"])
        self.assertTrue(any("EMG_START_OK" in item for item in statuses))
        self.assertTrue(any("EMG_STOP_OK" in item for item in statuses))


class EmgManagerTests(unittest.TestCase):
    def test_mock_mode_produces_features_and_explicit_mock_status(self):
        manager = EmgManager(EmgConfig(enabled=True, mode="mock"))
        self.assertTrue(manager.start())
        deadline = time.monotonic() + 1.0
        while manager.latest_feature() is None and time.monotonic() < deadline:
            time.sleep(0.01)
        status = manager.runtime_status()
        manager.stop()
        self.assertEqual(status.link_state, "mock")
        self.assertTrue(status.mock_mode)
        self.assertGreater(status.raw_samples, 0)
        self.assertGreater(status.feature_frames, 0)


class _FakeSerialPort:
    def __init__(self, payload):
        self.payload = payload
        self.closed = False

    def read(self, _size):
        if self.payload:
            payload, self.payload = self.payload, b""
            return payload
        time.sleep(0.005)
        return b""

    def cancel_read(self):
        pass

    def close(self):
        self.closed = True


class SerialCaptureTests(unittest.TestCase):
    def test_text_parser_validates_sequence_channels_and_int32_bounds(self):
        sample = SerialEmgCapture.parse_text_line(
            " EMG,4294967295,999999999999,-999999999999 ", 2, 123
        )
        self.assertIsNotNone(sample)
        self.assertEqual(sample.host_ts_ns, 123)
        self.assertEqual(sample.packet_seq, 0xFFFFFFFF)
        self.assertEqual(sample.channels, (2**31 - 1, -(2**31)))
        self.assertIsNone(SerialEmgCapture.parse_text_line("EMG,-1,1,2", 2))
        self.assertIsNone(SerialEmgCapture.parse_text_line("bad", 2))

    def test_serial_worker_accepts_valid_lines_and_counts_invalid_lines(self):
        port = _FakeSerialPort(b"EMG,1,10,-20\r\ninvalid\n")
        capture = SerialEmgCapture(
            EmgConfig(channel_count=2),
            serial_factory=lambda **kwargs: port,
        )
        samples, statuses = [], []
        self.assertTrue(capture.start(samples.append, statuses.append))
        deadline = time.monotonic() + 1.0
        while (not samples or capture.parse_errors == 0) and time.monotonic() < deadline:
            time.sleep(0.01)
        capture.stop()
        self.assertEqual(samples[0].channels, (10, -20))
        self.assertEqual(capture.parse_errors, 1)
        self.assertTrue(port.closed)


class _Rpmsg:
    def __init__(self):
        self.is_connected = False
        self.invalid_feature_packets = 0
        self.chunks = []
        self.on_feature = None

    def set_on_feature(self, callback):
        self.on_feature = callback

    def set_on_status(self, callback):
        self.on_status = callback

    def connect(self):
        self.is_connected = True
        return True

    def close(self):
        self.is_connected = False

    def send_raw_chunk(self, chunk):
        self.chunks.append(chunk)
        channels = tuple(
            EmgChannelFeature(i, rms=100.0 + i, state=EmgMuscleState.SMOOTH_FLEX)
            for i in range(chunk.channel_count)
        )
        self.on_feature(
            EmgFeatureFrame(chunk.host_ts_ns, chunk.seq, chunk.sample_rate_hz, channels)
        )
        return True


class EmgRealPipelineTests(unittest.TestCase):
    def test_ble_samples_are_chunked_then_cpu1_features_are_published(self):
        config = EmgConfig(
            enabled=True,
            mode="real",
            capture_backend="bluez",
            ble_command_rx_uuid="command",
            ble_status_tx_uuid="status",
            ble_notify_char_uuid="data",
            ble_command_timeout_ms=1000,
            raw_chunk_samples=25,
        )
        _Client.config = config
        capture = BleGattCapture(config, _Scanner, _Client)
        rpmsg = _Rpmsg()
        manager = EmgManager(config, capture, rpmsg)
        self.assertTrue(manager.start())
        deadline = time.monotonic() + 2.0
        while not rpmsg.chunks and time.monotonic() < deadline:
            time.sleep(0.01)
        status = manager.runtime_status()
        manager.stop()
        self.assertEqual(len(rpmsg.chunks), 1)
        self.assertEqual(rpmsg.chunks[0].sample_count, 25)
        self.assertEqual(status.raw_samples, 25)
        self.assertEqual(status.raw_chunks, 1)
        self.assertEqual(status.feature_frames, 1)
        self.assertEqual(status.link_state, "real-ok")


class EmgFeatureAndFusionTests(unittest.TestCase):
    def test_feature_processor_matches_original_threshold_order(self):
        chunk = EmgRawChunk(
            1_000_000_000,
            9,
            1000,
            2,
            (-100, 1, 300, 1, -100, 1, 300, 1),
        )
        frame = EmgFeatureProcessor.process(chunk, 100.0, 10.0)
        self.assertTrue(frame.valid)
        self.assertEqual(frame.channels[0].state, EmgMuscleState.FATIGUE)
        self.assertEqual(frame.channels[1].state, EmgMuscleState.REST)
        self.assertAlmostEqual(frame.channels[0].rms, (50_000.0) ** 0.5)
        self.assertAlmostEqual(frame.channels[0].fatigue_index, 2.0 / 3.0)

    def test_fusion_buffer_nearest_pruning_and_interval_summary(self):
        buffer = EmgFusionBuffer(keep_ns=100)

        def feature(timestamp, seq, state, rms):
            return EmgFeatureFrame(
                timestamp,
                seq,
                1000,
                (EmgChannelFeature(0, rms=rms, fatigue_index=0.5, state=state),),
            )

        buffer.push_feature(feature(100, 1, EmgMuscleState.REST, 5.0))
        buffer.push_feature(feature(150, 2, EmgMuscleState.TREMOR, 20.0))
        buffer.push_feature(feature(250, 3, EmgMuscleState.FATIGUE, 30.0))
        self.assertEqual(buffer.nearest(160, 20).seq, 2)
        self.assertIsNone(buffer.nearest(100, 10))
        summary = buffer.summary_for_interval(140, 260)
        self.assertEqual(summary.frame_count, 2)
        self.assertEqual(summary.channel_observations, 2)
        self.assertAlmostEqual(summary.active_ratio, 1.0)
        self.assertAlmostEqual(summary.tremor_ratio, 0.5)
        self.assertAlmostEqual(summary.fatigue_ratio, 0.5)
        self.assertEqual(summary.dominant_state, EmgMuscleState.TREMOR)

    def test_manager_device_selection_and_scanner_contract(self):
        class Scanner:
            def scan(self, seconds):
                return EmgBluetoothScanResult(
                    True,
                    "scan finished",
                    (EmgBluetoothDevice("AA:BB", "EMG", -42),),
                )

        config = EmgConfig()
        manager = EmgManager(config, bluetooth_scanner=Scanner())
        self.assertEqual(manager.scan_devices(1).devices[0].name, "EMG")
        self.assertTrue(manager.connect_ble("AA:BB"))
        self.assertEqual(config.ble_address, "AA:BB")
        self.assertTrue(manager.connect_ble("/dev/rfcomm2"))
        self.assertEqual(config.serial_device, "/dev/rfcomm2")

    def test_serial_samples_use_same_raw_chunk_and_cpu1_feature_path(self):
        lines = b"".join(f"EMG,{index},{index},{-index}\n".encode() for index in range(25))
        port = _FakeSerialPort(lines)
        config = EmgConfig(
            enabled=True,
            mode="real",
            capture_backend="serial",
            raw_chunk_samples=25,
        )
        serial_capture = SerialEmgCapture(
            config, serial_factory=lambda **kwargs: port
        )
        rpmsg = _Rpmsg()
        manager = EmgManager(config, rpmsg_client=rpmsg, serial_capture=serial_capture)
        self.assertTrue(manager.start())
        deadline = time.monotonic() + 2.0
        while not rpmsg.chunks and time.monotonic() < deadline:
            time.sleep(0.01)
        status = manager.runtime_status()
        manager.stop()
        self.assertEqual(len(rpmsg.chunks), 1)
        self.assertEqual(status.raw_samples, 25)
        self.assertEqual(status.feature_frames, 1)
        self.assertEqual(status.link_state, "real-ok")


if __name__ == "__main__":
    unittest.main(verbosity=2)
