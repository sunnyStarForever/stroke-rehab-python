"""Python-owned RGB/Depth capture orchestration and synchronization.

Low-level V4L2/OpenNI drivers may remain native, but frames, timestamps,
nearest-neighbour pairing, queue bounds, lifecycle and diagnostics live here.
The synchronizer mirrors ``core/sync/SyncManager.cpp``.
"""

from __future__ import annotations

import threading
import time
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Deque, Dict, Generic, Optional, TypeVar

from .performance import CallbackPerformanceMonitor


class FrameSource(str, Enum):
    RGB = "rgb"
    DEPTH = "depth"


@dataclass(frozen=True)
class FrameEnvelope:
    source: FrameSource
    image: Any
    width: int
    height: int
    sync_ts_ns: int
    arrival_ts_ns: int = 0
    device_ts_us: int = 0
    frame_id: int = 0
    depth_unit_to_meter: float = 0.001
    pixel_format_name: str = ""
    device_time_unit: str = "us"
    clock_quality: str = "host_fallback"
    clock_reason: str = ""
    clock_reset_count: int = 0

    @property
    def host_ts_ns(self) -> int:
        """Compatibility alias; synchronization uses ``sync_ts_ns``."""
        return self.arrival_ts_ns or self.sync_ts_ns

    @property
    def valid(self) -> bool:
        size = getattr(self.image, "size", None)
        has_data = bool(size) if size is not None else bool(self.image)
        return self.width > 0 and self.height > 0 and has_data


@dataclass(frozen=True)
class SyncedFramePair:
    rgb: FrameEnvelope
    depth: FrameEnvelope
    delta_ns: int


@dataclass(frozen=True)
class SyncStats:
    rgb_pushed: int = 0
    depth_pushed: int = 0
    matched: int = 0
    rgb_trimmed: int = 0
    depth_trimmed: int = 0
    threshold_misses: int = 0
    rgb_queued: int = 0
    depth_queued: int = 0


class TimestampNormalizer:
    @staticmethod
    def stamp(
        frame: FrameEnvelope, arrival_ts_ns: int, device_ts_us: int = 0
    ) -> FrameEnvelope:
        return FrameEnvelope(
            source=frame.source,
            image=frame.image,
            width=frame.width,
            height=frame.height,
            sync_ts_ns=arrival_ts_ns,
            arrival_ts_ns=arrival_ts_ns,
            device_ts_us=device_ts_us,
            frame_id=frame.frame_id,
            depth_unit_to_meter=frame.depth_unit_to_meter,
            pixel_format_name=frame.pixel_format_name,
            device_time_unit=frame.device_time_unit,
            clock_quality="host_fallback",
            clock_reason="python_stamp",
            clock_reset_count=frame.clock_reset_count,
        )


class FrameSynchronizer:
    """Bounded nearest-host-timestamp RGB/Depth matcher."""

    def __init__(self, config: Any):
        self.match_threshold_ns = max(0, int(getattr(config, "match_threshold_ns", 20_000_000)))
        self.queue_size = max(1, int(getattr(config, "queue_size", 30)))
        self._rgb: Deque[FrameEnvelope] = deque()
        self._depth: Deque[FrameEnvelope] = deque()
        self._lock = threading.Lock()
        self._callback: Optional[Callable[[SyncedFramePair], None]] = None
        self._rgb_pushed = self._depth_pushed = self._matched = 0
        self._rgb_trimmed = self._depth_trimmed = self._threshold_misses = 0

    def set_on_pair_ready(self, callback: Optional[Callable[[SyncedFramePair], None]]) -> None:
        with self._lock:
            self._callback = callback

    def push_frame(self, frame: FrameEnvelope) -> Optional[SyncedFramePair]:
        if not frame.valid:
            return None
        callback = None
        pair = None
        with self._lock:
            queue = self._rgb if frame.source is FrameSource.RGB else self._depth
            queue.append(frame)
            if frame.source is FrameSource.RGB:
                self._rgb_pushed += 1
            else:
                self._depth_pushed += 1
            self._trim(queue, frame.source)
            pair = self._try_match(frame.source)
            callback = self._callback
        # Keep capture callbacks outside the lock, matching the original C++.
        if pair is not None and callback is not None:
            callback(pair)
        return pair

    def _try_match(self, incoming: FrameSource) -> Optional[SyncedFramePair]:
        anchor_queue = self._rgb if incoming is FrameSource.RGB else self._depth
        other_queue = self._depth if incoming is FrameSource.RGB else self._rgb
        if not anchor_queue or not other_queue:
            return None
        anchor = anchor_queue[-1]
        best_index = min(
            range(len(other_queue)),
            key=lambda index: abs(anchor.sync_ts_ns - other_queue[index].sync_ts_ns),
        )
        other = other_queue[best_index]
        if abs(anchor.sync_ts_ns - other.sync_ts_ns) > self.match_threshold_ns:
            self._threshold_misses += 1
            return None
        del other_queue[best_index]
        anchor_queue.pop()
        rgb, depth = (anchor, other) if incoming is FrameSource.RGB else (other, anchor)
        self._matched += 1
        return SyncedFramePair(rgb, depth, rgb.sync_ts_ns - depth.sync_ts_ns)

    def _trim(self, queue: Deque[FrameEnvelope], source: FrameSource) -> None:
        while len(queue) > self.queue_size:
            queue.popleft()
            if source is FrameSource.RGB:
                self._rgb_trimmed += 1
            else:
                self._depth_trimmed += 1

    def clear(self) -> None:
        with self._lock:
            self._rgb.clear()
            self._depth.clear()

    def reset(self) -> None:
        with self._lock:
            self._rgb.clear()
            self._depth.clear()
            self._rgb_pushed = self._depth_pushed = self._matched = 0
            self._rgb_trimmed = self._depth_trimmed = self._threshold_misses = 0

    def stats(self) -> SyncStats:
        with self._lock:
            return SyncStats(
                self._rgb_pushed,
                self._depth_pushed,
                self._matched,
                self._rgb_trimmed,
                self._depth_trimmed,
                self._threshold_misses,
                len(self._rgb),
                len(self._depth),
            )


T = TypeVar("T")


class LatestFrameQueue(Generic[T]):
    """Single-slot latest-only queue matching ``LatestFrameQueue.h``."""

    def __init__(self):
        self._condition = threading.Condition()
        self._latest: Optional[T] = None
        self._stopped = False
        self.pushed = self.popped = self.dropped = 0

    def push(self, value: T) -> None:
        with self._condition:
            if self._latest is not None:
                self.dropped += 1
            self._latest = value
            self.pushed += 1
            self._condition.notify()

    def pop_latest(self, timeout: Optional[float] = None) -> Optional[T]:
        with self._condition:
            if self._latest is None and not self._stopped:
                self._condition.wait_for(
                    lambda: self._latest is not None or self._stopped, timeout=timeout
                )
            if self._stopped or self._latest is None:
                return None
            value, self._latest = self._latest, None
            self.popped += 1
            return value

    def stop(self) -> None:
        with self._condition:
            self._stopped = True
            self._condition.notify_all()

    def clear(self) -> None:
        with self._condition:
            self._latest = None
            self._stopped = False
            self.pushed = self.popped = self.dropped = 0


def _native_device_config(core: Any, config: Any):
    native = core.DeviceConfig()
    mappings = {
        "openni_device_uri": "openni_device_uri",
        "rgb_device_path": "rgb_device_path",
        "rgb_pixel_format": "rgb_pixel_format",
        "rgb_device_index": "rgb_device_index",
        "rgb_width": "rgb_width",
        "rgb_height": "rgb_height",
        "rgb_fps": "rgb_fps",
        "mirror_rgb_at_capture": "mirror_rgb_at_capture",
        "depth_pixel_format": "depth_pixel_format",
        "depth_width": "depth_width",
        "depth_height": "depth_height",
        "depth_fps": "depth_fps",
        "enable_hardware_d2c": "enable_hardware_d2c",
        "enable_openni_color_stream_for_debug": "enable_openni_color_stream_for_debug",
        "enable_openni_depth_color_sync": "enable_openni_depth_color_sync",
        "latest_queue_size": "latest_queue_size",
        "raw_perf_log_interval_sec": "raw_perf_log_interval_sec",
        "enable_cpu_affinity": "enable_cpu_affinity",
        "rgb_capture_cpu": "rgb_capture_cpu",
        "depth_capture_cpu": "depth_capture_cpu",
    }
    for python_name, native_name in mappings.items():
        if hasattr(config, python_name):
            try:
                setattr(native, native_name, getattr(config, python_name))
            except AttributeError:
                pass
    if hasattr(native, "rgb_device_path") and not getattr(native, "rgb_device_path", ""):
        native.rgb_device_path = f"/dev/video{int(getattr(config, 'rgb_device_index', 0))}"
    return native


class NativeRgbDepthBackend:
    """Adapt separate native V4L2/OpenNI drivers to Python synchronization."""

    def __init__(self, core: Any, device_config: Any, sync_config: Any):
        self._core = core
        self._device_config = device_config
        self._sync = FrameSynchronizer(sync_config)
        self._rgb = None
        self._depth = None
        self._running = False
        self._status_callback: Optional[Callable[[str], None]] = None
        self._started_at = 0.0
        self._rgb_callback_ms: Deque[float] = deque(maxlen=512)
        self._depth_callback_ms: Deque[float] = deque(maxlen=512)
        self._sync_arrivals: Deque[float] = deque()
        self._perf_monitor = CallbackPerformanceMonitor(
            window_seconds=getattr(device_config, "callback_perf_window_sec", 5.0),
            min_samples=getattr(device_config, "callback_perf_min_samples", 30),
            normal_p95_ms=getattr(device_config, "callback_normal_p95_ms", 8.0),
            warn_p95_ms=getattr(device_config, "callback_warn_p95_ms", 10.0),
            critical_p95_ms=getattr(device_config, "callback_critical_p95_ms", 25.0),
            warn_sustain_seconds=getattr(device_config, "callback_warn_sustain_sec", 5.0),
            low_fps_sustain_seconds=getattr(device_config, "callback_low_fps_sustain_sec", 3.0),
            recovery_seconds=getattr(device_config, "callback_recovery_sec", 5.0),
            target_fps={
                "rgb": float(getattr(device_config, "rgb_fps", 30)),
                "depth": float(getattr(device_config, "depth_fps", 30)),
            },
        )
        self._clock_state: Dict[FrameSource, tuple[str, str, int]] = {}
        self._clock_quality_counts: Dict[str, int] = {}

    @property
    def is_running(self) -> bool:
        return self._running

    def set_on_status(self, callback: Optional[Callable[[str], None]]) -> None:
        self._status_callback = callback

    def start(self, pair_callback: Callable[[SyncedFramePair], None]) -> bool:
        if self._running:
            return True
        self._sync.reset()
        self._rgb_callback_ms.clear()
        self._depth_callback_ms.clear()
        self._sync_arrivals.clear()
        self._perf_monitor.reset()
        self._clock_state.clear()
        self._clock_quality_counts.clear()
        def observed_pair_callback(pair: SyncedFramePair) -> None:
            now = time.monotonic()
            self._sync_arrivals.append(now)
            cutoff = now - max(0.1, float(getattr(
                self._device_config, "callback_perf_window_sec", 5.0)))
            while self._sync_arrivals and self._sync_arrivals[0] < cutoff:
                self._sync_arrivals.popleft()
            pair_callback(pair)

        self._sync.set_on_pair_ready(observed_pair_callback)
        native_config = _native_device_config(self._core, self._device_config)
        self._rgb = self._core.RgbCaptureV4L2()
        self._depth = self._core.DepthCaptureOpenNI()
        if self._status_callback and hasattr(self._rgb, "set_on_status"):
            self._rgb.set_on_status(self._status_callback)

        # Do not accept callbacks until the native depth driver has explicitly
        # attested that a real hardware stream is active.  This also prevents
        # a legacy module from leaking synthetic startup frames before it is
        # rejected below.
        self._running = False
        rgb_ok = self._rgb.start(native_config, self._on_rgb)
        depth_started = self._depth.start(native_config, self._on_depth)
        depth_ok = False
        if depth_started and hasattr(self._depth, "real_depth_active"):
            deadline = time.monotonic() + 5.0
            while (not self._depth.real_depth_active()
                   and self._depth.is_running()
                   and time.monotonic() < deadline):
                time.sleep(0.02)
            depth_ok = bool(self._depth.real_depth_active())
        if not rgb_ok or not depth_ok:
            self._emit(
                "Native capture start failed: "
                f"rgb={bool(rgb_ok)} real_depth={bool(depth_ok)}"
            )
            self.stop()
            return False
        self._running = True
        self._started_at = time.monotonic()
        self._emit("Native RGB/Depth drivers started; Python timestamp sync active")
        return True

    def stop(self) -> None:
        # Disable acceptance before blocking in driver stop calls.
        self._running = False
        errors = []
        for name, driver in (("rgb", self._rgb), ("depth", self._depth)):
            if driver is not None:
                try:
                    driver.stop()
                except Exception as exc:
                    errors.append(f"{name}: {exc}")
        self._rgb = self._depth = None
        self._sync.clear()
        if errors:
            raise RuntimeError("; ".join(errors))

    def _on_rgb(
        self, image, width, height, sync_ts_ns, frame_id,
        device_ts_us=0, _depth_unit=0.001, pixel_format="", _source="rgb",
        arrival_ts_ns=0, device_time_unit="us", clock_quality="host_fallback",
        clock_reason="", clock_reset_count=0,
    ) -> None:
        if self._running:
            if not self._valid_native_array(FrameSource.RGB, image, width, height):
                return
            observed_at = time.monotonic()
            self._perf_monitor.observe_arrival("rgb", observed_at)
            started = time.perf_counter()
            self._observe_clock(
                FrameSource.RGB, str(clock_quality), str(clock_reason),
                int(clock_reset_count))
            self._sync.push_frame(
                FrameEnvelope(
                    source=FrameSource.RGB,
                    image=image,
                    width=int(width),
                    height=int(height),
                    sync_ts_ns=int(sync_ts_ns),
                    arrival_ts_ns=int(arrival_ts_ns or sync_ts_ns),
                    device_ts_us=device_ts_us,
                    frame_id=frame_id,
                    pixel_format_name=str(pixel_format),
                    device_time_unit=str(device_time_unit),
                    clock_quality=str(clock_quality),
                    clock_reason=str(clock_reason),
                    clock_reset_count=int(clock_reset_count),
                )
            )
            duration_ms = (time.perf_counter() - started) * 1000.0
            self._rgb_callback_ms.append(duration_ms)
            event = self._perf_monitor.observe_callback("rgb", duration_ms, observed_at)
            if event is not None:
                self._emit(event.log_message())

    def _on_depth(
        self, image, width, height, sync_ts_ns, frame_id,
        device_ts_us=0, depth_unit=0.001, pixel_format="", _source="depth",
        arrival_ts_ns=0, device_time_unit="us", clock_quality="host_fallback",
        clock_reason="", clock_reset_count=0,
    ) -> None:
        if self._running:
            if not self._valid_native_array(FrameSource.DEPTH, image, width, height):
                return
            observed_at = time.monotonic()
            self._perf_monitor.observe_arrival("depth", observed_at)
            started = time.perf_counter()
            self._observe_clock(
                FrameSource.DEPTH, str(clock_quality), str(clock_reason),
                int(clock_reset_count))
            self._sync.push_frame(
                FrameEnvelope(
                    source=FrameSource.DEPTH,
                    image=image,
                    width=int(width),
                    height=int(height),
                    sync_ts_ns=int(sync_ts_ns),
                    arrival_ts_ns=int(arrival_ts_ns or sync_ts_ns),
                    device_ts_us=device_ts_us,
                    frame_id=frame_id,
                    depth_unit_to_meter=depth_unit,
                    pixel_format_name=str(pixel_format),
                    device_time_unit=str(device_time_unit),
                    clock_quality=str(clock_quality),
                    clock_reason=str(clock_reason),
                    clock_reset_count=int(clock_reset_count),
                )
            )
            duration_ms = (time.perf_counter() - started) * 1000.0
            self._depth_callback_ms.append(duration_ms)
            event = self._perf_monitor.observe_callback("depth", duration_ms, observed_at)
            if event is not None:
                self._emit(event.log_message())

    def hardware_d2c_active(self) -> bool:
        return bool(self._depth and self._depth.hardware_d2c_active())

    def sync_stats(self) -> SyncStats:
        return self._sync.stats()

    def performance_stats(self) -> Dict[str, Any]:
        elapsed = max(0.0, time.monotonic() - self._started_at) if self._started_at else 0.0
        stats = self._sync.stats()

        now = time.monotonic()
        for source in ("rgb", "depth"):
            event = self._perf_monitor.evaluate(source, now)
            if event is not None:
                self._emit(event.log_message())
        rgb_perf = self._perf_monitor.snapshot("rgb", now)
        depth_perf = self._perf_monitor.snapshot("depth", now)
        sync_fps = 0.0
        if len(self._sync_arrivals) >= 2:
            span = self._sync_arrivals[-1] - self._sync_arrivals[0]
            sync_fps = (len(self._sync_arrivals) - 1) / span if span > 0 else 0.0

        return {
            "elapsed_seconds": elapsed,
            "raw_rgb_fps": rgb_perf.raw_fps,
            "raw_depth_fps": depth_perf.raw_fps,
            "sync_fps": sync_fps,
            "rgb_callback_state": rgb_perf.state,
            "depth_callback_state": depth_perf.state,
            "rgb_callback_avg_ms": rgb_perf.callback_avg_ms,
            "depth_callback_avg_ms": depth_perf.callback_avg_ms,
            "rgb_callback_p50_ms": rgb_perf.callback_p50_ms,
            "depth_callback_p50_ms": depth_perf.callback_p50_ms,
            "rgb_callback_p95_ms": rgb_perf.callback_p95_ms,
            "depth_callback_p95_ms": depth_perf.callback_p95_ms,
            "rgb_callback_max_ms": rgb_perf.callback_max_ms,
            "depth_callback_max_ms": depth_perf.callback_max_ms,
            "clock_quality_counts": dict(self._clock_quality_counts),
            "clock_states": {
                source.value: {
                    "quality": state[0], "reason": state[1],
                    "reset_count": state[2],
                }
                for source, state in self._clock_state.items()
            },
        }

    def _observe_clock(
        self, source: FrameSource, quality: str, reason: str, reset_count: int
    ) -> None:
        key = f"{source.value}:{quality}"
        self._clock_quality_counts[key] = self._clock_quality_counts.get(key, 0) + 1
        state = (quality, reason, reset_count)
        if self._clock_state.get(source) == state:
            return
        self._clock_state[source] = state
        detail = f" reason={reason}" if reason else ""
        self._emit(
            f"Clock {source.value}: quality={quality} resets={reset_count}{detail}")

    def _valid_native_array(
        self, source: FrameSource, image: Any, width: int, height: int
    ) -> bool:
        expected_shape = (
            (int(height), int(width), 3)
            if source is FrameSource.RGB else (int(height), int(width))
        )
        expected_dtype = "uint8" if source is FrameSource.RGB else "uint16"
        valid = (
            getattr(image, "shape", None) == expected_shape
            and str(getattr(image, "dtype", "")) == expected_dtype
            and bool(getattr(getattr(image, "flags", None), "c_contiguous", False))
        )
        if not valid:
            self._emit(
                f"Rejected {source.value} callback array: "
                f"shape={getattr(image, 'shape', None)} "
                f"dtype={getattr(image, 'dtype', None)} expected={expected_shape}/{expected_dtype}")
        elif hasattr(image, "setflags"):
            image.setflags(write=False)
        return valid

    def _emit(self, message: str) -> None:
        if self._status_callback:
            self._status_callback(message)


__all__ = [
    "FrameEnvelope",
    "FrameSource",
    "FrameSynchronizer",
    "LatestFrameQueue",
    "NativeRgbDepthBackend",
    "SyncStats",
    "SyncedFramePair",
    "TimestampNormalizer",
]
