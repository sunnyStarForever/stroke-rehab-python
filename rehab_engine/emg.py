"""Pure-Python EMG acquisition primitives and manager.

This module is the Python counterpart of ``core/emg``.  Hardware transports
are kept behind a small callback API so the application layer owns lifecycle,
status and recording while an optional BLE library performs the I/O.
"""

from __future__ import annotations

import asyncio
import math
import os
import select
import struct
import threading
import time
from collections import deque
from dataclasses import dataclass, field, replace
from enum import IntEnum
from typing import Any, Callable, List, Optional, Sequence, Tuple

from .recorder import EmgRecorder


class EmgMuscleState(IntEnum):
    REST = 0
    SMOOTH_FLEX = 1
    TREMOR = 2
    FATIGUE = 3


@dataclass(frozen=True)
class EmgRawSample:
    host_ts_ns: int
    packet_seq: int
    sample_index: int
    channels: Tuple[int, ...]


@dataclass(frozen=True)
class EmgRawChunk:
    host_ts_ns: int
    seq: int
    sample_rate_hz: int
    channel_count: int
    interleaved_samples: Tuple[int, ...]

    @property
    def sample_count(self) -> int:
        if self.channel_count <= 0:
            return 0
        return len(self.interleaved_samples) // self.channel_count


@dataclass(frozen=True)
class EmgChannelFeature:
    channel: int
    rms: float = 0.0
    zcr: float = 0.0
    cv: float = 0.0
    fatigue_index: float = 0.0
    state: EmgMuscleState = EmgMuscleState.REST


@dataclass(frozen=True)
class EmgFeatureFrame:
    host_ts_ns: int
    seq: int
    sample_rate_hz: int
    channels: Tuple[EmgChannelFeature, ...]

    @property
    def valid(self) -> bool:
        return self.host_ts_ns > 0 and bool(self.channels)


@dataclass(frozen=True)
class EmgIntervalSummary:
    frame_count: int = 0
    channel_observations: int = 0
    active_ratio: float = 0.0
    fatigue_ratio: float = 0.0
    tremor_ratio: float = 0.0
    avg_rms: float = 0.0
    max_rms: float = 0.0
    avg_fatigue_index: float = 0.0
    dominant_state: EmgMuscleState = EmgMuscleState.REST


@dataclass(frozen=True)
class EmgBluetoothDevice:
    address: str
    name: str = ""
    rssi: int = 0
    connected: bool = False


@dataclass(frozen=True)
class EmgBluetoothScanResult:
    ok: bool
    message: str
    devices: Tuple[EmgBluetoothDevice, ...] = ()


class EmgFeatureProcessor:
    """Exact Python port of ``core/emg/EmgFeatureProcessor.cpp``."""

    @staticmethod
    def process(
        chunk: EmgRawChunk, active_threshold: float, noise_threshold: float
    ) -> EmgFeatureFrame:
        channels = max(0, min(2, int(chunk.channel_count)))
        samples = chunk.sample_count
        if (
            channels == 0
            or samples == 0
            or len(chunk.interleaved_samples) != channels * samples
        ):
            return EmgFeatureFrame(chunk.host_ts_ns, chunk.seq, chunk.sample_rate_hz, ())

        features = []
        for channel in range(channels):
            values = [
                int(chunk.interleaved_samples[index * channels + channel])
                for index in range(samples)
            ]
            total = float(sum(values))
            square_sum = float(sum(value * value for value in values))
            mean = total / samples
            mean_square = square_sum / samples
            variance = max(0.0, mean_square - mean * mean)
            rms = math.sqrt(mean_square)
            standard_deviation = math.sqrt(variance)
            mean_magnitude = abs(mean)
            cv = standard_deviation / mean_magnitude if mean_magnitude > 1.0e-12 else 0.0
            zero_crossings = sum(
                1
                for previous, value in zip(values, values[1:])
                if (previous < 0 <= value) or (previous >= 0 > value)
            )
            zcr = zero_crossings / (samples - 1) if samples > 1 else 0.0
            fatigue = min(1.0, max(0.0, cv / (1.0 + cv)))
            if rms < max(0.0, noise_threshold):
                state = EmgMuscleState.REST
            elif rms >= max(noise_threshold, active_threshold) and fatigue >= 0.5:
                state = EmgMuscleState.FATIGUE
            elif zcr >= 0.25:
                state = EmgMuscleState.TREMOR
            else:
                state = EmgMuscleState.SMOOTH_FLEX
            features.append(
                EmgChannelFeature(channel, rms, zcr, cv, fatigue, state)
            )
        return EmgFeatureFrame(
            chunk.host_ts_ns, chunk.seq, chunk.sample_rate_hz, tuple(features)
        )


class EmgFusionBuffer:
    """Thread-safe timestamp buffer matching the original EMG/vision fusion."""

    def __init__(self, keep_ns: int = 10_000_000_000) -> None:
        self.keep_ns = max(0, int(keep_ns))
        self._frames = deque()
        self._lock = threading.Lock()

    def clear(self) -> None:
        with self._lock:
            self._frames.clear()

    def push_feature(self, frame: EmgFeatureFrame) -> None:
        if not frame.valid:
            return
        with self._lock:
            self._frames.append(frame)
            newest = frame.host_ts_ns
            while (
                self._frames
                and self.keep_ns > 0
                and newest > self._frames[0].host_ts_ns
                and newest - self._frames[0].host_ts_ns > self.keep_ns
            ):
                self._frames.popleft()

    def latest(self) -> Optional[EmgFeatureFrame]:
        with self._lock:
            return self._frames[-1] if self._frames else None

    def nearest(self, host_ts_ns: int, max_delta_ns: int) -> Optional[EmgFeatureFrame]:
        if max_delta_ns < 0:
            return None
        with self._lock:
            if not self._frames:
                return None
            best = min(self._frames, key=lambda frame: abs(frame.host_ts_ns - host_ts_ns))
            return best if abs(best.host_ts_ns - host_ts_ns) <= max_delta_ns else None

    def summary_for_interval(self, start_ns: int, end_ns: int) -> EmgIntervalSummary:
        with self._lock:
            frames = tuple(
                frame
                for frame in self._frames
                if frame.host_ts_ns >= start_ns
                and (end_ns <= 0 or frame.host_ts_ns <= end_ns)
            )
        channels = [channel for frame in frames for channel in frame.channels]
        if not channels:
            return EmgIntervalSummary(frame_count=len(frames))
        counts = [0, 0, 0, 0]
        for channel in channels:
            counts[int(channel.state)] += 1
        observations = len(channels)
        dominant = EmgMuscleState(max(range(4), key=lambda index: counts[index]))
        return EmgIntervalSummary(
            frame_count=len(frames),
            channel_observations=observations,
            active_ratio=sum(counts[1:]) / observations,
            fatigue_ratio=counts[3] / observations,
            tremor_ratio=counts[2] / observations,
            avg_rms=sum(channel.rms for channel in channels) / observations,
            max_rms=max(channel.rms for channel in channels),
            avg_fatigue_index=(
                sum(channel.fatigue_index for channel in channels) / observations
            ),
            dominant_state=dominant,
        )


class EmgBluetoothScanner:
    """Discover Bluetooth devices through Bleak without blocking an event loop."""

    def scan(self, seconds: int = 4) -> EmgBluetoothScanResult:
        timeout = float(max(1, min(int(seconds), 20)))
        result = []
        error = []

        def worker() -> None:
            try:
                result.append(asyncio.run(self._discover(timeout)))
            except Exception as exc:
                error.append(exc)

        thread = threading.Thread(target=worker, name="emg-ble-scan", daemon=True)
        thread.start()
        thread.join(timeout + 5.0)
        if thread.is_alive():
            return EmgBluetoothScanResult(False, "Bluetooth scan timed out")
        if error:
            return EmgBluetoothScanResult(False, str(error[0]))
        devices = tuple(result[0] if result else ())
        return EmgBluetoothScanResult(
            True, "scan finished" if devices else "no bluetooth devices found", devices
        )

    async def _discover(self, timeout: float) -> List[EmgBluetoothDevice]:
        try:
            from bleak import BleakScanner
        except ImportError as exc:
            raise RuntimeError("Bleak is not installed") from exc
        try:
            discovered = await BleakScanner.discover(timeout=timeout, return_adv=True)
        except TypeError:
            discovered = await BleakScanner.discover(timeout=timeout)
        devices = []
        values = discovered.values() if isinstance(discovered, dict) else discovered
        for item in values:
            device, advertisement = item if isinstance(item, tuple) else (item, None)
            rssi = int(getattr(advertisement, "rssi", getattr(device, "rssi", 0)) or 0)
            devices.append(
                EmgBluetoothDevice(
                    address=str(getattr(device, "address", "")),
                    name=str(getattr(device, "name", "") or ""),
                    rssi=rssi,
                )
            )
        devices.sort(key=lambda device: (device.name.lower(), device.address.lower()))
        return devices


@dataclass(frozen=True)
class BleSequenceResult:
    accepted: bool = True
    dropped: int = 0
    duplicate: bool = False
    out_of_order: bool = False


@dataclass
class EmgRuntimeStatus:
    enabled: bool = False
    running: bool = False
    mock_mode: bool = False
    ble_connected: bool = False
    rpmsg_connected: bool = False
    recording: bool = False
    mode: str = "disabled"
    link_state: str = "disabled"
    capture_backend: str = "serial"
    message: str = "EMG disabled"
    raw_samples: int = 0
    raw_chunks: int = 0
    feature_frames: int = 0
    parse_errors: int = 0
    invalid_payloads: int = 0
    dropped_packets: int = 0
    duplicate_packets: int = 0
    out_of_order_packets: int = 0
    rpmsg_errors: int = 0
    invalid_feature_packets: int = 0
    ble_command_errors: int = 0
    ble_status_timeouts: int = 0
    estimated_sample_rate_hz: float = 0.0
    latest_feature: Optional[EmgFeatureFrame] = None


class EmgBleNotifyParser:
    """Parse the ESP32 204-byte GATT Data TX payload.

    Layout is identical to ``EmgBleNotifyParser`` in the original C++ code:
    little-endian uint32 packet sequence followed by 25 samples x 2 int32
    channels.  Timestamps are reconstructed backwards from notification time.
    """

    PAYLOAD_BYTES = 204
    CHANNEL_COUNT = 2
    SAMPLES_PER_PACKET = 25
    _PACKET = struct.Struct("<I50i")

    def __init__(self) -> None:
        self._last_sequence: Optional[int] = None

    def parse(
        self, payload: bytes, arrival_ts_ns: int, sample_rate_hz: int
    ) -> Optional[Tuple[List[EmgRawSample], int]]:
        if len(payload) != self.PAYLOAD_BYTES or sample_rate_hz <= 0:
            return None
        unpacked = self._PACKET.unpack(payload)
        sequence = unpacked[0]
        values = unpacked[1:]
        interval_ns = 1_000_000_000 // sample_rate_hz
        history_ns = interval_ns * (self.SAMPLES_PER_PACKET - 1)
        first_ts_ns = max(0, arrival_ts_ns - history_ns)
        samples = [
            EmgRawSample(
                host_ts_ns=first_ts_ns + interval_ns * index,
                packet_seq=sequence,
                sample_index=index,
                channels=(values[index * 2], values[index * 2 + 1]),
            )
            for index in range(self.SAMPLES_PER_PACKET)
        ]
        return samples, sequence

    def observe_sequence(self, packet_seq: int) -> BleSequenceResult:
        packet_seq &= 0xFFFFFFFF
        if self._last_sequence is None:
            self._last_sequence = packet_seq
            return BleSequenceResult()
        forward = (packet_seq - self._last_sequence) & 0xFFFFFFFF
        if forward == 0:
            return BleSequenceResult(accepted=False, duplicate=True)
        if forward < 0x80000000:
            self._last_sequence = packet_seq
            return BleSequenceResult(dropped=forward - 1)
        return BleSequenceResult(accepted=False, out_of_order=True)

    def reset_sequence(self) -> None:
        self._last_sequence = None


class EmgRpmsgProtocol:
    """Binary codec matching ``core/emg/EmgProtocol.h`` version 2."""

    MAGIC = 0x31474D45
    VERSION = 2
    MAX_PACKET_BYTES = 256
    MAX_CHANNELS = 2
    MAX_RAW_SAMPLES = 25
    HEADER = struct.Struct("<IBBHIQHBBHH")
    CONFIG = struct.Struct("<HBBff")
    CHANNEL_FEATURE = struct.Struct("<ffffB3x")
    RAW_CHUNK = 1
    FEATURE = 2
    CONFIG_MESSAGE = 3

    @classmethod
    def _header(
        cls,
        message_type: int,
        seq: int = 0,
        host_ts_ns: int = 0,
        sample_rate_hz: int = 1000,
        channel_count: int = 2,
        sample_count: int = 0,
        payload_bytes: int = 0,
    ) -> bytes:
        return cls.HEADER.pack(
            cls.MAGIC,
            cls.VERSION,
            message_type,
            cls.HEADER.size,
            seq & 0xFFFFFFFF,
            host_ts_ns,
            max(1, min(65535, int(sample_rate_hz))),
            channel_count,
            sample_count,
            payload_bytes,
            0,
        )

    @classmethod
    def pack_config(cls, config: Any) -> bytes:
        channel_count = max(1, min(cls.MAX_CHANNELS, int(getattr(config, "channel_count", 2))))
        raw_samples = max(1, min(cls.MAX_RAW_SAMPLES, int(getattr(config, "raw_chunk_samples", 25))))
        payload = cls.CONFIG.pack(
            raw_samples,
            4,
            0,
            float(getattr(config, "active_threshold", 800.0)),
            float(getattr(config, "noise_threshold", 15.0)),
        )
        return cls._header(
            cls.CONFIG_MESSAGE,
            sample_rate_hz=int(getattr(config, "sample_rate_hz", 1000)),
            channel_count=channel_count,
            payload_bytes=len(payload),
        ) + payload

    @classmethod
    def pack_raw_chunk(cls, chunk: EmgRawChunk) -> Optional[bytes]:
        count = chunk.sample_count
        expected = chunk.channel_count * count
        if (
            chunk.host_ts_ns <= 0
            or chunk.channel_count <= 0
            or chunk.channel_count > cls.MAX_CHANNELS
            or count <= 0
            or count > cls.MAX_RAW_SAMPLES
            or len(chunk.interleaved_samples) != expected
        ):
            return None
        payload = struct.pack(f"<{expected}i", *chunk.interleaved_samples)
        return cls._header(
            cls.RAW_CHUNK,
            seq=chunk.seq,
            host_ts_ns=chunk.host_ts_ns,
            sample_rate_hz=chunk.sample_rate_hz,
            channel_count=chunk.channel_count,
            sample_count=count,
            payload_bytes=len(payload),
        ) + payload

    @classmethod
    def parse_feature(cls, packet: bytes) -> Optional[EmgFeatureFrame]:
        if len(packet) < cls.HEADER.size:
            return None
        (
            magic,
            version,
            message_type,
            header_size,
            seq,
            host_ts_ns,
            sample_rate_hz,
            channel_count,
            sample_count,
            payload_bytes,
            _reserved,
        ) = cls.HEADER.unpack_from(packet)
        if (
            magic != cls.MAGIC
            or version != cls.VERSION
            or message_type != cls.FEATURE
            or header_size != cls.HEADER.size
            or not 1 <= channel_count <= cls.MAX_CHANNELS
            or sample_count != 0
            or payload_bytes != channel_count * cls.CHANNEL_FEATURE.size
            or len(packet) != cls.HEADER.size + payload_bytes
        ):
            return None
        channels = []
        for index in range(channel_count):
            offset = cls.HEADER.size + index * cls.CHANNEL_FEATURE.size
            rms, zcr, cv, fatigue, state_value = cls.CHANNEL_FEATURE.unpack_from(packet, offset)
            if (
                not all(math.isfinite(value) for value in (rms, zcr, cv, fatigue))
                or state_value > int(EmgMuscleState.FATIGUE)
            ):
                return None
            channels.append(
                EmgChannelFeature(
                    channel=index,
                    rms=rms,
                    zcr=zcr,
                    cv=cv,
                    fatigue_index=fatigue,
                    state=EmgMuscleState(state_value),
                )
            )
        frame = EmgFeatureFrame(host_ts_ns, seq, sample_rate_hz, tuple(channels))
        return frame if frame.valid else None


class EmgRpmsgClient:
    """Linux rpmsg-char client owned by the Python application layer."""

    # _IOW(0xb5, 0x1, struct rpmsg_endpoint_info[40]) on Linux.
    RPMSG_CREATE_EPT_IOCTL = 0x4028B501
    RPMSG_ADDR_ANY = 0xFFFFFFFF

    def __init__(self, config: Any) -> None:
        self._config = config
        self._ctrl_fd = -1
        self._data_fd = -1
        self._connected = threading.Event()
        self._reader_running = threading.Event()
        self._reader: Optional[threading.Thread] = None
        self._io_lock = threading.Lock()
        self._feature_callback: Optional[Callable[[EmgFeatureFrame], None]] = None
        self._status_callback: Optional[Callable[[str], None]] = None
        self.invalid_feature_packets = 0

    @property
    def is_connected(self) -> bool:
        return self._connected.is_set()

    def set_on_feature(self, callback) -> None:
        self._feature_callback = callback

    def set_on_status(self, callback) -> None:
        self._status_callback = callback

    def connect(self) -> bool:
        if self.is_connected:
            return True
        if not bool(getattr(self._config, "rpmsg_enabled", True)):
            self._emit("EMG RPMsg disabled by config")
            return False
        if os.name != "posix":
            self._emit("EMG RPMsg is only available on Linux")
            return False
        ctrl_path = str(getattr(self._config, "rpmsg_ctrl_device", "/dev/rpmsg_ctrl0"))
        data_path = str(getattr(self._config, "rpmsg_data_device", "/dev/rpmsg0"))
        endpoint = str(getattr(self._config, "rpmsg_endpoint_name", "emg_rpmsg"))
        if not os.path.exists(ctrl_path):
            self._emit("RPMsg device not found; start remoteproc0 and load rpmsg_char")
            return False
        try:
            import fcntl

            self._ctrl_fd = os.open(ctrl_path, os.O_RDWR | getattr(os, "O_CLOEXEC", 0))
            name = endpoint.encode("ascii", "strict")[:31].ljust(32, b"\x00")
            endpoint_info = struct.pack("<32sII", name, self.RPMSG_ADDR_ANY, self.RPMSG_ADDR_ANY)
            fcntl.ioctl(self._ctrl_fd, self.RPMSG_CREATE_EPT_IOCTL, endpoint_info)
            self._data_fd = os.open(
                data_path,
                os.O_RDWR | os.O_NONBLOCK | getattr(os, "O_CLOEXEC", 0),
            )
        except (OSError, ValueError, UnicodeError) as exc:
            self._emit(f"EMG RPMsg connect failed: {exc}")
            self.close()
            return False
        self.invalid_feature_packets = 0
        self._connected.set()
        self._reader_running.set()
        self._reader = threading.Thread(target=self._read_loop, name="emg-rpmsg", daemon=True)
        self._reader.start()
        self._emit(f"EMG RPMsg connected endpoint={endpoint}")
        if not self.send_config():
            self.close()
            return False
        return True

    def close(self) -> None:
        self._reader_running.clear()
        if self._reader and self._reader is not threading.current_thread():
            self._reader.join(timeout=1.0)
        self._reader = None
        with self._io_lock:
            for name in ("_data_fd", "_ctrl_fd"):
                descriptor = getattr(self, name)
                if descriptor >= 0:
                    try:
                        os.close(descriptor)
                    except OSError:
                        pass
                    setattr(self, name, -1)
        if self._connected.is_set():
            self._connected.clear()
            self._emit("EMG RPMsg disconnected")

    def send_config(self) -> bool:
        return self._write_packet(EmgRpmsgProtocol.pack_config(self._config))

    def send_raw_chunk(self, chunk: EmgRawChunk) -> bool:
        packet = EmgRpmsgProtocol.pack_raw_chunk(chunk)
        if packet is None:
            self._emit("EMG RPMsg raw chunk rejected by protocol bounds")
            return False
        return self._write_packet(packet)

    def _write_packet(self, packet: bytes) -> bool:
        if not self.is_connected or not packet or len(packet) > 256:
            return False
        with self._io_lock:
            if self._data_fd < 0:
                return False
            offset = 0
            try:
                while offset < len(packet):
                    written = os.write(self._data_fd, packet[offset:])
                    if written <= 0:
                        raise OSError("zero-byte RPMsg write")
                    offset += written
                return True
            except OSError as exc:
                self._connected.clear()
                self._emit(f"EMG RPMsg write failed: {exc}")
                return False

    def _read_loop(self) -> None:
        poller = select.poll()
        poller.register(self._data_fd, select.POLLIN)
        timeout = max(1, int(getattr(self._config, "rpmsg_poll_timeout_ms", 5)))
        while self._reader_running.is_set():
            try:
                ready = poller.poll(timeout)
                if not ready:
                    continue
                packet = os.read(self._data_fd, EmgRpmsgProtocol.MAX_PACKET_BYTES)
            except BlockingIOError:
                continue
            except OSError as exc:
                self._emit(f"EMG RPMsg read/poll failed: {exc}")
                break
            frame = EmgRpmsgProtocol.parse_feature(packet)
            if frame is None:
                self.invalid_feature_packets += 1
                self._emit("EMG RPMsg rejected invalid feature packet")
            elif self._feature_callback:
                self._feature_callback(frame)
        self._connected.clear()

    def _emit(self, message: str) -> None:
        if self._status_callback:
            self._status_callback(message)


class BleGattCapture:
    """Optional Bleak-based implementation of the firmware GATT handshake."""

    def __init__(self, config: Any, scanner_cls=None, client_cls=None) -> None:
        self._config = config
        self._scanner_cls = scanner_cls
        self._client_cls = client_cls
        self._parser = EmgBleNotifyParser()
        self._sample_callback: Optional[Callable[[EmgRawSample], None]] = None
        self._status_callback: Optional[Callable[[str], None]] = None
        self._thread: Optional[threading.Thread] = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._stop_async: Optional[asyncio.Event] = None
        self._running = threading.Event()
        self._connected = threading.Event()
        self._streaming = threading.Event()
        self._lock = threading.Lock()
        self.invalid_payloads = 0
        self.dropped_packets = 0
        self.duplicate_packets = 0
        self.out_of_order_packets = 0
        self.command_errors = 0
        self.status_timeouts = 0

    @property
    def is_connected(self) -> bool:
        return self._connected.is_set()

    def start(self, on_sample, on_status=None) -> bool:
        if self._running.is_set():
            return True
        self._sample_callback = on_sample
        self._status_callback = on_status
        if self._scanner_cls is None or self._client_cls is None:
            try:
                from bleak import BleakClient, BleakScanner
            except ImportError:
                self._emit("EMG BLE backend unavailable: install bleak")
                return False
            self._scanner_cls = BleakScanner
            self._client_cls = BleakClient
        required = (
            "ble_command_rx_uuid",
            "ble_status_tx_uuid",
            "ble_notify_char_uuid",
        )
        if any(not str(getattr(self._config, name, "")).strip() for name in required):
            self._emit("EMG BLE configuration is missing command/status/data UUIDs")
            return False
        self._parser.reset_sequence()
        self._reset_diagnostics()
        self._running.set()
        self._thread = threading.Thread(
            target=self._thread_main, name="emg-ble-gatt", daemon=True
        )
        self._thread.start()
        return True

    def stop(self) -> None:
        self._running.clear()
        loop, stop_event = self._loop, self._stop_async
        if loop is not None and stop_event is not None and loop.is_running():
            loop.call_soon_threadsafe(stop_event.set)
        if self._thread and self._thread is not threading.current_thread():
            self._thread.join(timeout=self._command_timeout_seconds() + 2.0)
        self._thread = None
        self._connected.clear()
        self._streaming.clear()

    def _thread_main(self) -> None:
        try:
            asyncio.run(self._run())
        except Exception as exc:
            self.command_errors += 1
            self._emit(f"EMG BLE failed: {exc}")
        finally:
            self._running.clear()
            self._connected.clear()
            self._streaming.clear()
            self._loop = None
            self._stop_async = None

    async def _run(self) -> None:
        self._loop = asyncio.get_running_loop()
        self._stop_async = asyncio.Event()
        target = await self._resolve_target()
        if target is None:
            self._emit("EMG BLE device not found")
            return

        start_ack = asyncio.Event()
        stop_ack = asyncio.Event()

        def on_status(_sender, payload: bytearray) -> None:
            response = bytes(payload).rstrip(b"\x00\r\n").decode("ascii", "replace")
            if response == "EMG_START_OK":
                self._streaming.set()
                start_ack.set()
                self._emit("EMG BLE streaming started: EMG_START_OK")
            elif response == "EMG_STOP_OK":
                stop_ack.set()
            elif response == "UNKNOWN_COMMAND":
                self.command_errors += 1
                self._emit("EMG BLE device rejected command: UNKNOWN_COMMAND")
            elif response:
                self._emit(f"EMG BLE status: {response}")

        def on_data(_sender, payload: bytearray) -> None:
            if not self._streaming.is_set():
                return
            parsed = self._parser.parse(
                bytes(payload), time.monotonic_ns(), self._sample_rate_hz()
            )
            if parsed is None:
                self.invalid_payloads += 1
                return
            samples, packet_seq = parsed
            sequence = self._parser.observe_sequence(packet_seq)
            self.dropped_packets += sequence.dropped
            self.duplicate_packets += int(sequence.duplicate)
            self.out_of_order_packets += int(sequence.out_of_order)
            if sequence.accepted and self._sample_callback:
                for sample in samples:
                    self._sample_callback(sample)

        async with self._client_cls(target) as client:
            self._connected.set()
            status_uuid = str(self._config.ble_status_tx_uuid)
            data_uuid = str(self._config.ble_notify_char_uuid)
            command_uuid = str(self._config.ble_command_rx_uuid)
            await client.start_notify(status_uuid, on_status)
            await client.start_notify(data_uuid, on_data)
            try:
                await client.write_gatt_char(command_uuid, b"START_EMG", response=True)
                self._emit("EMG BLE START_EMG sent; waiting for EMG_START_OK")
                try:
                    await asyncio.wait_for(start_ack.wait(), self._command_timeout_seconds())
                except asyncio.TimeoutError:
                    self.status_timeouts += 1
                    self._emit("EMG BLE timeout waiting for EMG_START_OK")
                    return
                await self._stop_async.wait()
                self._streaming.clear()
                try:
                    await client.write_gatt_char(command_uuid, b"STOP_EMG", response=True)
                    self._emit("EMG BLE STOP_EMG sent; waiting for EMG_STOP_OK")
                    await asyncio.wait_for(stop_ack.wait(), self._command_timeout_seconds())
                    self._emit("EMG BLE stopped: EMG_STOP_OK")
                except asyncio.TimeoutError:
                    self.status_timeouts += 1
                    self._emit("EMG BLE stopped without EMG_STOP_OK")
                except Exception as exc:
                    self.command_errors += 1
                    self._emit(f"EMG BLE STOP_EMG failed: {exc}")
            finally:
                await client.stop_notify(data_uuid)
                await client.stop_notify(status_uuid)

    async def _resolve_target(self):
        address = str(getattr(self._config, "ble_address", "")).strip()
        if address:
            return address
        prefix = str(getattr(self._config, "ble_name_prefix", "ESP32_EMG"))
        devices = await self._scanner_cls.discover(timeout=4.0)
        for device in devices:
            if str(getattr(device, "name", "") or "").startswith(prefix):
                return device
        return None

    def _sample_rate_hz(self) -> int:
        return max(1, int(getattr(self._config, "sample_rate_hz", 1000)))

    def _command_timeout_seconds(self) -> float:
        timeout_ms = max(1000, int(getattr(self._config, "ble_command_timeout_ms", 5000)))
        return timeout_ms / 1000.0

    def _emit(self, message: str) -> None:
        callback = self._status_callback
        if callback:
            callback(message)

    def _reset_diagnostics(self) -> None:
        self.invalid_payloads = 0
        self.dropped_packets = 0
        self.duplicate_packets = 0
        self.out_of_order_packets = 0
        self.command_errors = 0
        self.status_timeouts = 0


class SerialEmgCapture:
    """Text-protocol serial/rfcomm capture matching EmgBleSerialCapture.cpp."""

    def __init__(self, config: Any, serial_factory=None, clock_ns=time.monotonic_ns) -> None:
        self._config = config
        self._serial_factory = serial_factory
        self._clock_ns = clock_ns
        self._serial = None
        self._sample_callback: Optional[Callable[[EmgRawSample], None]] = None
        self._status_callback: Optional[Callable[[str], None]] = None
        self._running = threading.Event()
        self._connected = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self.parse_errors = 0
        self.estimated_sample_rate_hz = 0.0

    @property
    def is_connected(self) -> bool:
        return self._connected.is_set()

    @staticmethod
    def parse_text_line(
        line: str,
        expected_channels: int,
        host_ts_ns: Optional[int] = None,
    ) -> Optional[EmgRawSample]:
        clean = line.strip()
        if not clean.startswith("EMG,"):
            return None
        parts = [part.strip() for part in clean.split(",")]
        channel_count = max(1, int(expected_channels))
        if len(parts) < 2 + channel_count:
            return None
        try:
            sequence = int(parts[1], 10)
            if sequence < 0 or sequence > 0xFFFFFFFF:
                return None
            channels = []
            for index in range(channel_count):
                value = int(parts[2 + index], 10)
                channels.append(max(-(2**31), min(2**31 - 1, value)))
        except ValueError:
            return None
        return EmgRawSample(
            host_ts_ns if host_ts_ns is not None else time.monotonic_ns(),
            sequence,
            0,
            tuple(channels),
        )

    def start(self, on_sample, on_status=None) -> bool:
        if self._running.is_set():
            return True
        self._sample_callback = on_sample
        self._status_callback = on_status
        factory = self._serial_factory
        if factory is None:
            try:
                import serial
            except ImportError:
                self._emit("EMG serial backend unavailable: install pyserial")
                return False
            factory = serial.Serial
        device = str(getattr(self._config, "serial_device", "/dev/rfcomm0"))
        baud = int(getattr(self._config, "serial_baud_rate", 115200))
        try:
            self._serial = factory(port=device, baudrate=baud, timeout=0.1)
        except Exception as exc:
            self._emit(f"EMG serial open failed: {device} ({exc})")
            return False
        self.parse_errors = 0
        self.estimated_sample_rate_hz = 0.0
        self._running.set()
        self._connected.set()
        self._thread = threading.Thread(target=self._read_loop, name="emg-serial", daemon=True)
        self._thread.start()
        self._emit(f"EMG serial connected: {device}")
        return True

    def stop(self) -> None:
        self._running.clear()
        serial_port = self._serial
        if serial_port is not None:
            try:
                serial_port.cancel_read()
            except (AttributeError, OSError):
                pass
        if self._thread and self._thread is not threading.current_thread():
            self._thread.join(timeout=1.0)
        self._thread = None
        if serial_port is not None:
            try:
                serial_port.close()
            except Exception:
                pass
        self._serial = None
        was_connected = self._connected.is_set()
        self._connected.clear()
        if was_connected:
            self._emit("EMG serial disconnected")

    def _read_loop(self) -> None:
        line_buffer = bytearray()
        received = 0
        window_start = self._clock_ns()
        try:
            while self._running.is_set():
                chunk = self._serial.read(256)
                if not chunk:
                    continue
                for value in chunk:
                    if value == 10:
                        line = line_buffer.decode("ascii", "replace")
                        sample = self.parse_text_line(
                            line,
                            int(getattr(self._config, "channel_count", 2)),
                            self._clock_ns(),
                        )
                        if sample is not None:
                            if self._sample_callback:
                                self._sample_callback(sample)
                            received += 1
                            now = self._clock_ns()
                            elapsed = now - window_start
                            if elapsed >= 1_000_000_000:
                                self.estimated_sample_rate_hz = received * 1e9 / elapsed
                                received = 0
                                window_start = now
                        elif line_buffer:
                            self.parse_errors += 1
                            if self.parse_errors % 50 == 1:
                                self._emit(f"EMG serial parse failed: {line}")
                        line_buffer.clear()
                    elif value != 13:
                        line_buffer.append(value)
                        if len(line_buffer) > 512:
                            line_buffer.clear()
                            self.parse_errors += 1
        except Exception as exc:
            if self._running.is_set():
                self._emit(f"EMG serial read failed: {exc}")
        finally:
            self._connected.clear()
            self._running.clear()

    def _emit(self, message: str) -> None:
        if self._status_callback:
            self._status_callback(message)


class EmgManager:
    """Own EMG mode lifecycle and expose one status/feature API to Python UI."""

    def __init__(
        self,
        config: Any,
        ble_capture: Optional[BleGattCapture] = None,
        rpmsg_client: Optional[EmgRpmsgClient] = None,
        recorder: Optional[EmgRecorder] = None,
        serial_capture: Optional[SerialEmgCapture] = None,
        bluetooth_scanner: Optional[EmgBluetoothScanner] = None,
    ) -> None:
        self._config = config
        self._ble = ble_capture or BleGattCapture(config)
        self._rpmsg = rpmsg_client or EmgRpmsgClient(config)
        self._rpmsg.set_on_feature(self._publish_feature)
        self._rpmsg.set_on_status(self._handle_transport_status)
        self._recorder = recorder or EmgRecorder()
        self._serial = serial_capture or SerialEmgCapture(config)
        self._scanner = bluetooth_scanner or EmgBluetoothScanner()
        self._fusion = EmgFusionBuffer()
        self._status = EmgRuntimeStatus(
            enabled=bool(getattr(config, "enabled", False)),
            mode=str(getattr(config, "mode", "disabled")),
            capture_backend=str(getattr(config, "capture_backend", "serial")),
        )
        self._lock = threading.Lock()
        self._running = threading.Event()
        self._mock_thread: Optional[threading.Thread] = None
        self._feature_callback: Optional[Callable[[EmgFeatureFrame], None]] = None
        self._status_callback: Optional[Callable[[EmgRuntimeStatus], None]] = None
        self._mock_seq = 0
        self._pending_values: List[int] = []
        self._pending_ts_ns = 0
        self._pending_seq = 0

    def set_on_feature(self, callback) -> None:
        self._feature_callback = callback

    def set_on_status(self, callback) -> None:
        self._status_callback = callback

    def start(self) -> bool:
        enabled = bool(getattr(self._config, "enabled", False))
        mode = str(getattr(self._config, "mode", "disabled")).lower()
        with self._lock:
            self._status = EmgRuntimeStatus(
                enabled=enabled,
                running=False,
                mode=mode if enabled else "disabled",
                link_state="disabled",
                capture_backend=str(getattr(self._config, "capture_backend", "serial")),
            )
        if not enabled or mode == "disabled":
            self._emit_status("EMG disabled")
            return True
        self._running.set()
        self._fusion.clear()
        if mode == "mock":
            with self._lock:
                self._status.running = True
                self._status.mock_mode = True
                self._status.ble_connected = True
                self._status.link_state = "mock"
            self._mock_thread = threading.Thread(
                target=self._mock_loop, name="emg-mock", daemon=True
            )
            self._mock_thread.start()
            self._emit_status("EMG mock backend started")
            return True

        backend = str(getattr(self._config, "capture_backend", "serial")).lower()
        if not self._rpmsg.connect():
            self._running.clear()
            self._set_failed("EMG real mode failed because RPMsg is unavailable")
            return False
        with self._lock:
            self._status.running = True
            self._status.rpmsg_connected = True
            self._status.link_state = "real-connecting"
        capture_started = (
            self._ble.start(self._handle_raw_sample, self._handle_transport_status)
            if backend == "bluez"
            else self._serial.start(self._handle_raw_sample, self._handle_transport_status)
        )
        if not capture_started:
            self._rpmsg.close()
            self._running.clear()
            self._set_failed(f"EMG {backend} backend failed to start")
            return False
        self._emit_status(f"EMG real {backend} pipeline started")
        return True

    def stop(self) -> None:
        self._running.clear()
        self._ble.stop()
        self._serial.stop()
        self._rpmsg.close()
        if self._mock_thread and self._mock_thread is not threading.current_thread():
            self._mock_thread.join(timeout=1.0)
        self._mock_thread = None
        with self._lock:
            self._status.running = False
            self._status.ble_connected = False
            self._status.rpmsg_connected = False
            self._pending_values.clear()
        self._emit_status("EMG stopped")

    def start_recording(self, action_dir: str) -> bool:
        self._recorder.set_link_context(
            str(getattr(self._config, "mode", "disabled")),
            str(getattr(self._config, "capture_backend", "serial")),
            bool(getattr(self._config, "strict_real_mode", True)),
        )
        ok = self._recorder.start(action_dir)
        with self._lock:
            self._status.recording = ok
        self._emit_status(
            f"EMG recording started: {action_dir}"
            if ok
            else f"EMG recording start failed: {action_dir}"
        )
        return ok

    def stop_recording(self) -> None:
        self._recorder.set_final_status(self.runtime_status())
        self._recorder.stop()
        with self._lock:
            self._status.recording = False
        self._emit_status("EMG recording stopped")

    def runtime_status(self) -> EmgRuntimeStatus:
        with self._lock:
            status = replace(self._status)
        status.ble_connected = status.mock_mode or self._ble.is_connected or self._serial.is_connected
        status.rpmsg_connected = self._rpmsg.is_connected
        status.invalid_payloads = self._ble.invalid_payloads
        status.dropped_packets = self._ble.dropped_packets
        status.duplicate_packets = self._ble.duplicate_packets
        status.out_of_order_packets = self._ble.out_of_order_packets
        status.ble_command_errors = self._ble.command_errors
        status.ble_status_timeouts = self._ble.status_timeouts
        status.invalid_feature_packets = self._rpmsg.invalid_feature_packets
        status.parse_errors = self._serial.parse_errors
        status.estimated_sample_rate_hz = self._serial.estimated_sample_rate_hz
        return status

    def latest_feature(self) -> Optional[EmgFeatureFrame]:
        return self._fusion.latest()

    def nearest_feature(
        self, host_ts_ns: int, max_delta_ns: int = 300_000_000
    ) -> Optional[EmgFeatureFrame]:
        return self._fusion.nearest(host_ts_ns, max_delta_ns)

    def summary_for_interval(self, start_ns: int, end_ns: int) -> EmgIntervalSummary:
        return self._fusion.summary_for_interval(start_ns, end_ns)

    def scan_devices(self, seconds: int = 4) -> EmgBluetoothScanResult:
        return self._scanner.scan(seconds)

    def connect_ble(self, address_or_name_or_device_path: str) -> bool:
        value = str(address_or_name_or_device_path).strip()
        if not value:
            return False
        if value.startswith("/dev/") or value.upper().startswith("COM"):
            self._config.serial_device = value
        else:
            self._config.ble_address = value
        self._emit_status("EMG BLE selection updated; restart pipeline to apply")
        return True

    def disconnect_ble(self) -> None:
        self._serial.stop()
        self._ble.stop()
        with self._lock:
            self._status.ble_connected = False
        self._emit_status("EMG BLE disconnected")

    def _mock_loop(self) -> None:
        interval = 0.02
        while self._running.is_set():
            now = time.monotonic_ns()
            phase = time.monotonic()
            values = (
                int(900 + 420 * abs(math.sin(phase * 2.1))),
                int(760 + 360 * abs(math.sin(phase * 1.8 + 0.7))),
            )
            raw = EmgRawSample(now, self._mock_seq, 0, values)
            self._mock_seq = (self._mock_seq + 1) & 0xFFFFFFFF
            self._handle_raw_sample(raw)
            features = tuple(
                EmgChannelFeature(
                    channel=index,
                    rms=float(abs(value)),
                    zcr=0.04,
                    cv=0.12,
                    fatigue_index=0.18,
                    state=EmgMuscleState.SMOOTH_FLEX,
                )
                for index, value in enumerate(values)
            )
            self._publish_feature(EmgFeatureFrame(now, raw.packet_seq, 1000, features))
            time.sleep(interval)

    def _handle_raw_sample(self, sample: EmgRawSample) -> None:
        chunk = None
        self._recorder.record_raw_sample(sample)
        with self._lock:
            self._status.raw_samples += 1
            if not self._status.mock_mode and self._rpmsg.is_connected:
                channel_count = max(
                    1, min(EmgRpmsgProtocol.MAX_CHANNELS, int(getattr(self._config, "channel_count", 2)))
                )
                if not self._pending_values:
                    self._pending_ts_ns = sample.host_ts_ns
                    self._pending_seq = sample.packet_seq
                self._pending_values.extend(
                    sample.channels[index] if index < len(sample.channels) else 0
                    for index in range(channel_count)
                )
                target = max(
                    1,
                    min(
                        EmgRpmsgProtocol.MAX_RAW_SAMPLES,
                        int(getattr(self._config, "raw_chunk_samples", 25)),
                    ),
                )
                if len(self._pending_values) // channel_count >= target:
                    chunk = EmgRawChunk(
                        host_ts_ns=self._pending_ts_ns,
                        seq=self._pending_seq,
                        sample_rate_hz=int(getattr(self._config, "sample_rate_hz", 1000)),
                        channel_count=channel_count,
                        interleaved_samples=tuple(self._pending_values),
                    )
                    self._pending_values.clear()
        if chunk is not None:
            if self._rpmsg.send_raw_chunk(chunk):
                with self._lock:
                    self._status.raw_chunks += 1
            else:
                with self._lock:
                    self._status.rpmsg_errors += 1
                self._emit_status("EMG RPMsg raw chunk send failed")

    def _publish_feature(self, frame: EmgFeatureFrame) -> None:
        if not frame.valid:
            return
        self._fusion.push_feature(frame)
        self._recorder.record_feature_frame(frame)
        with self._lock:
            self._status.feature_frames += 1
            self._status.latest_feature = frame
        if self._feature_callback:
            self._feature_callback(frame)

    def _handle_transport_status(self, message: str) -> None:
        lowered = message.lower()
        with self._lock:
            self._status.ble_connected = self._ble.is_connected or self._serial.is_connected
            self._status.rpmsg_connected = self._rpmsg.is_connected
            if "streaming started" in lowered or (
                "serial connected" in lowered and self._rpmsg.is_connected
            ):
                self._status.link_state = "real-ok"
            elif any(word in lowered for word in ("failed", "timeout", "rejected", "without")):
                strict = bool(getattr(self._config, "strict_real_mode", True))
                self._status.link_state = "real-failed" if strict else "real-degraded"
        self._emit_status(message)

    def _set_failed(self, message: str) -> None:
        strict = bool(getattr(self._config, "strict_real_mode", True))
        with self._lock:
            self._status.running = False
            self._status.link_state = "real-failed" if strict else "real-degraded"
        self._emit_status(message)

    def _emit_status(self, message: str) -> None:
        with self._lock:
            self._status.message = message
            status = replace(self._status)
        if self._status_callback:
            self._status_callback(status)


__all__ = [
    "BleGattCapture",
    "BleSequenceResult",
    "EmgBleNotifyParser",
    "EmgChannelFeature",
    "EmgFeatureFrame",
    "EmgManager",
    "EmgMuscleState",
    "EmgRawSample",
    "EmgRawChunk",
    "EmgRpmsgClient",
    "EmgRpmsgProtocol",
    "EmgRuntimeStatus",
    "SerialEmgCapture",
]
