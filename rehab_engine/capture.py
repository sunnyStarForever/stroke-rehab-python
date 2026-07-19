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


class FrameSource(str, Enum):
    RGB = "rgb"
    DEPTH = "depth"


@dataclass(frozen=True)
class FrameEnvelope:
    source: FrameSource
    payload: bytes
    width: int
    height: int
    host_ts_ns: int
    device_ts_us: int = 0
    frame_id: int = 0
    encoding: str = ""
    depth_unit_to_meter: float = 0.001
    pixel_format_name: str = ""

    @property
    def valid(self) -> bool:
        return self.width > 0 and self.height > 0 and bool(self.payload)


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
        frame: FrameEnvelope, host_ts_ns: int, device_ts_us: int = 0
    ) -> FrameEnvelope:
        return FrameEnvelope(
            source=frame.source,
            payload=frame.payload,
            width=frame.width,
            height=frame.height,
            host_ts_ns=host_ts_ns,
            device_ts_us=device_ts_us,
            frame_id=frame.frame_id,
            encoding=frame.encoding,
            depth_unit_to_meter=frame.depth_unit_to_meter,
            pixel_format_name=frame.pixel_format_name,
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
            key=lambda index: abs(anchor.host_ts_ns - other_queue[index].host_ts_ns),
        )
        other = other_queue[best_index]
        if abs(anchor.host_ts_ns - other.host_ts_ns) > self.match_threshold_ns:
            self._threshold_misses += 1
            return None
        del other_queue[best_index]
        anchor_queue.pop()
        rgb, depth = (anchor, other) if incoming is FrameSource.RGB else (other, anchor)
        self._matched += 1
        return SyncedFramePair(rgb, depth, rgb.host_ts_ns - depth.host_ts_ns)

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

    @property
    def is_running(self) -> bool:
        return self._running

    def set_on_status(self, callback: Optional[Callable[[str], None]]) -> None:
        self._status_callback = callback

    def start(self, pair_callback: Callable[[SyncedFramePair], None]) -> bool:
        if self._running:
            return True
        self._sync.reset()
        self._sync.set_on_pair_ready(pair_callback)
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

    def _on_rgb(self, payload, width, height, ts_ns, frame_id, *extra) -> None:
        if self._running:
            device_ts_us = int(extra[0]) if len(extra) >= 2 else 0
            pixel_format = str(extra[2]) if len(extra) >= 4 else ""
            self._sync.push_frame(
                FrameEnvelope(
                    FrameSource.RGB,
                    bytes(payload),
                    width,
                    height,
                    ts_ns,
                    device_ts_us=device_ts_us,
                    frame_id=frame_id,
                    encoding="jpeg",
                    pixel_format_name=pixel_format,
                )
            )

    def _on_depth(self, payload, width, height, ts_ns, frame_id, *extra) -> None:
        if self._running:
            device_ts_us = int(extra[0]) if len(extra) >= 2 else 0
            depth_unit = float(extra[1]) if len(extra) >= 4 else 0.001
            pixel_format = str(extra[2]) if len(extra) >= 4 else ""
            self._sync.push_frame(
                FrameEnvelope(
                    FrameSource.DEPTH,
                    bytes(payload),
                    width,
                    height,
                    ts_ns,
                    device_ts_us=device_ts_us,
                    frame_id=frame_id,
                    encoding="png16",
                    depth_unit_to_meter=depth_unit,
                    pixel_format_name=pixel_format,
                )
            )

    def hardware_d2c_active(self) -> bool:
        return bool(self._depth and self._depth.hardware_d2c_active())

    def sync_stats(self) -> SyncStats:
        return self._sync.stats()

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
