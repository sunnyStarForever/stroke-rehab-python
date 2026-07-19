"""
Skeleton 3D recorder and EMG recorder.
Replaces core/record/Skeleton3DRecorder.cpp + core/emg/EmgRecorder.cpp.
Pure Python, uses csv module.
"""

from __future__ import annotations

import csv
import concurrent.futures
import json
import os
import queue
import threading
import time
from collections import deque
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, TextIO


@dataclass
class _SessionMeta:
    start_time: str = ""
    rgb_width: int = 0
    rgb_height: int = 0
    depth_width: int = 0
    depth_height: int = 0
    rgb_fps: int = 0
    depth_fps: int = 0
    pose_model_path: str = ""
    detector_model_path: str = ""
    pose_interval: int = 0
    hardware_d2c_enabled: bool = False
    mirror_rgb_at_capture: bool = True
    mirror_preview: bool = False


@dataclass
class RecorderStats:
    recording: bool = False
    frames: int = 0
    rows: int = 0
    skipped_frames: int = 0
    session_dir: str = ""
    csv_path: str = ""


@dataclass(frozen=True)
class VideoRecorderStats:
    recording: bool = False
    rgb_frames: int = 0
    depth_frames: int = 0
    rgb_path: str = ""
    depth_path: str = ""
    last_write_ms: float = 0.0
    received: int = 0
    written: int = 0
    dropped: int = 0
    failed: int = 0
    stop_dropped: int = 0
    queue_depth: int = 0
    queue_high_watermark: int = 0
    write_fps: float = 0.0
    write_avg_ms: float = 0.0
    write_p95_ms: float = 0.0
    metadata_path: str = ""
    last_error: str = ""


@dataclass(frozen=True)
class RecordingFrame:
    """Read-only references to one canonical synchronized RGB-D pair."""

    bgr_image: Any
    depth_image: Any
    sync_ts_ns: int
    rgb_frame_id: int
    depth_frame_id: int
    depth_unit_to_meter: float = 0.001
    align_mode: str = "hardware"


class RgbDepthVideoRecorder:
    """Bounded asynchronous RGB-D encoder, independent of the pose worker."""

    _STOP = object()

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._rgb_writer = None
        self._depth_writer = None
        self._recording = False
        self._rgb_frames = 0
        self._depth_frames = 0
        self._rgb_path = ""
        self._depth_path = ""
        self._last_write_ms = 0.0
        self._size = (0, 0)
        self._queue: Optional[queue.Queue] = None
        self._thread: Optional[threading.Thread] = None
        self._ready = threading.Event()
        self._abort = threading.Event()
        self._accepting = False
        self._record_rgb = True
        self._record_depth = False
        self._fps = 30.0
        self._received = self._written = self._dropped = self._failed = 0
        self._stop_dropped = self._queue_high_watermark = 0
        self._write_samples = deque(maxlen=512)
        self._write_timestamps = deque(maxlen=512)
        self._metadata_path = ""
        self._last_error = ""
        self._depth_transform = None
        self._encode_pool = None

    def start(
        self,
        session_dir: str,
        fps: int,
        width: int,
        height: int,
        record_rgb: bool = True,
        record_depth: bool = False,
        queue_capacity: int = 90,
        ready_timeout_sec: float = 5.0,
        depth_transform: Optional[Callable[[Any, float], Any]] = None,
    ) -> bool:
        self.stop()
        root = Path(session_dir)
        root.mkdir(parents=True, exist_ok=True)
        with self._lock:
            self._size = (max(1, int(width)), max(1, int(height)))
            self._fps = float(max(1, int(fps)))
            self._record_rgb = bool(record_rgb)
            self._record_depth = bool(record_depth)
            self._rgb_path = str(root / "rgb.mp4") if record_rgb else ""
            self._depth_path = str(root / "depth.avi") if record_depth else ""
            self._metadata_path = str(root / "video_frames.csv")
            self._queue = queue.Queue(maxsize=max(1, int(queue_capacity)))
            self._received = self._written = self._dropped = self._failed = 0
            self._stop_dropped = self._queue_high_watermark = 0
            self._rgb_frames = self._depth_frames = 0
            self._last_write_ms = 0.0
            self._write_samples.clear()
            self._write_timestamps.clear()
            self._last_error = ""
            self._depth_transform = depth_transform
            self._accepting = True
            self._recording = False
            self._ready.clear()
            self._abort.clear()
            self._thread = threading.Thread(
                target=self._run, name="rgbd-video-recorder", daemon=True)
            self._thread.start()
        if not self._ready.wait(max(0.1, float(ready_timeout_sec))):
            self._last_error = "recording thread startup timed out"
            self.stop()
            return False
        with self._lock:
            return self._recording

    def stop(self, drain_timeout_sec: float = 5.0) -> None:
        with self._lock:
            self._accepting = False
            work_queue = self._queue
            thread = self._thread
        if work_queue is not None and thread is not None and thread.is_alive():
            deadline = time.monotonic() + max(0.0, float(drain_timeout_sec))
            while thread.is_alive():
                try:
                    work_queue.put(self._STOP, timeout=0.05)
                    break
                except queue.Full:
                    if time.monotonic() >= deadline:
                        self._abort.set()
                        self._discard_pending(work_queue)
                        try:
                            work_queue.put_nowait(self._STOP)
                        except queue.Full:
                            pass
                        break
            thread.join(timeout=max(0.1, deadline - time.monotonic() + 0.5))
        with self._lock:
            self._recording = False
            self._thread = None
            self._queue = None

    def submit(self, frame: RecordingFrame) -> bool:
        with self._lock:
            if not self._accepting or self._queue is None:
                return False
            work_queue = self._queue
            self._received += 1
        try:
            work_queue.put_nowait(frame)
        except queue.Full:
            try:
                work_queue.get_nowait()
                work_queue.task_done()
                with self._lock:
                    self._dropped += 1
            except queue.Empty:
                pass
            try:
                work_queue.put_nowait(frame)
            except queue.Full:
                with self._lock:
                    self._dropped += 1
                return False
        with self._lock:
            self._queue_high_watermark = max(
                self._queue_high_watermark, work_queue.qsize())
        return True

    def record(self, rgb_image=None, depth_image=None) -> float:
        """Deprecated compatibility shim; encoding remains asynchronous."""
        self.submit(RecordingFrame(rgb_image, depth_image, time.monotonic_ns(), 0, 0))
        return 0.0

    def _run(self) -> None:
        metadata_file = None
        try:
            import cv2
            rgb_writer = depth_writer = None
            if self._record_rgb:
                rgb_writer = cv2.VideoWriter(
                    self._rgb_path, cv2.VideoWriter_fourcc(*"mp4v"),
                    self._fps, self._size, True)
                if not rgb_writer.isOpened():
                    raise OSError(f"cannot open RGB video: {self._rgb_path}")
            if self._record_depth:
                depth_writer = cv2.VideoWriter(
                    self._depth_path, cv2.VideoWriter_fourcc(*"mp4v"),
                    self._fps, self._size, True)
                if not depth_writer.isOpened():
                    raise OSError(f"cannot open depth video: {self._depth_path}")
            metadata_file = open(self._metadata_path, "w", newline="", encoding="utf-8")
            metadata = csv.writer(metadata_file)
            metadata.writerow((
                "written_index", "sync_ts_ns", "rgb_frame_id", "depth_frame_id",
                "depth_unit_to_meter", "align_mode", "write_ms"))
            with self._lock:
                self._rgb_writer = rgb_writer
                self._depth_writer = depth_writer
                self._recording = True
                if rgb_writer is not None and depth_writer is not None:
                    self._encode_pool = concurrent.futures.ThreadPoolExecutor(
                        max_workers=2, thread_name_prefix="rgbd-encode")
            self._ready.set()
            while True:
                work_queue = self._queue
                if work_queue is None:
                    break
                item = work_queue.get()
                try:
                    if item is self._STOP:
                        break
                    if self._abort.is_set():
                        with self._lock:
                            self._stop_dropped += 1
                        continue
                    write_ms = self._write_frame(item, cv2)
                    with self._lock:
                        self._written += 1
                        written_index = self._written
                    metadata.writerow((
                        written_index, item.sync_ts_ns, item.rgb_frame_id,
                        item.depth_frame_id, item.depth_unit_to_meter,
                        item.align_mode, f"{write_ms:.3f}"))
                    if written_index % 30 == 0:
                        metadata_file.flush()
                except Exception as exc:
                    with self._lock:
                        self._failed += 1
                        self._last_error = str(exc)
                finally:
                    work_queue.task_done()
        except Exception as exc:
            with self._lock:
                self._failed += 1
                self._last_error = str(exc)
        finally:
            self._ready.set()
            if metadata_file is not None:
                metadata_file.flush()
                metadata_file.close()
            with self._lock:
                encode_pool = self._encode_pool
                self._encode_pool = None
            if encode_pool is not None:
                encode_pool.shutdown(wait=True, cancel_futures=False)
            with self._lock:
                for writer in (self._rgb_writer, self._depth_writer):
                    if writer is not None:
                        writer.release()
                self._rgb_writer = self._depth_writer = None
                self._recording = False
                self._accepting = False

    def _write_frame(self, item: RecordingFrame, cv2) -> float:
        started = time.perf_counter()
        jobs = []
        if self._encode_pool is not None:
            if self._rgb_writer is not None and item.bgr_image is not None:
                jobs.append(self._encode_pool.submit(self._write_rgb, item.bgr_image, cv2))
            if self._depth_writer is not None and item.depth_image is not None:
                jobs.append(self._encode_pool.submit(
                    self._write_depth, item.depth_image,
                    item.depth_unit_to_meter, cv2))
            for job in jobs:
                job.result()
        else:
            if self._rgb_writer is not None and item.bgr_image is not None:
                self._write_rgb(item.bgr_image, cv2)
            if self._depth_writer is not None and item.depth_image is not None:
                self._write_depth(item.depth_image, item.depth_unit_to_meter, cv2)
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        now = time.monotonic()
        with self._lock:
            self._last_write_ms = elapsed_ms
            self._write_samples.append(elapsed_ms)
            self._write_timestamps.append(now)
        return elapsed_ms

    def _write_rgb(self, image, cv2) -> None:
        if self._rgb_writer is not None:
            rgb = image
            if rgb.shape[1::-1] != self._size:
                rgb = cv2.resize(rgb, self._size, interpolation=cv2.INTER_LINEAR)
            self._rgb_writer.write(rgb)
            self._rgb_frames += 1

    def _write_depth(self, image, depth_unit_to_meter: float, cv2) -> None:
        if self._depth_writer is not None:
            depth_source = image
            if self._depth_transform is not None:
                depth_source = self._depth_transform(
                    depth_source, depth_unit_to_meter)
            depth = cv2.resize(depth_source, self._size, interpolation=cv2.INTER_NEAREST)
            valid = depth > 0
            max_depth = float(depth[valid].max()) if valid.any() else 0.0
            if max_depth > 0.0:
                depth8 = (depth.astype("float32") * (255.0 / max_depth)).clip(0, 255).astype("uint8")
                depth8[~valid] = 0
                colored = cv2.applyColorMap(depth8, cv2.COLORMAP_JET)
            else:
                import numpy as np
                colored = np.zeros((self._size[1], self._size[0], 3), dtype=np.uint8)
            self._depth_writer.write(colored)
            self._depth_frames += 1

    def _discard_pending(self, work_queue: queue.Queue) -> None:
        while True:
            try:
                item = work_queue.get_nowait()
            except queue.Empty:
                return
            if item is not self._STOP:
                with self._lock:
                    self._stop_dropped += 1
            work_queue.task_done()

    def stats(self) -> VideoRecorderStats:
        with self._lock:
            values = list(self._write_samples)
            ordered = sorted(values)
            p95 = ordered[min(len(ordered) - 1, int((len(ordered) - 1) * 0.95 + 0.999999))] if ordered else 0.0
            write_fps = 0.0
            if len(self._write_timestamps) >= 2:
                span = self._write_timestamps[-1] - self._write_timestamps[0]
                write_fps = (len(self._write_timestamps) - 1) / span if span > 0 else 0.0
            return VideoRecorderStats(
                self._recording,
                self._rgb_frames,
                self._depth_frames,
                self._rgb_path,
                self._depth_path,
                self._last_write_ms,
                self._received,
                self._written,
                self._dropped,
                self._failed,
                self._stop_dropped,
                self._queue.qsize() if self._queue is not None else 0,
                self._queue_high_watermark,
                write_fps,
                sum(values) / len(values) if values else 0.0,
                p95,
                self._metadata_path,
                self._last_error,
            )


@dataclass(frozen=True)
class PairRecorderStats:
    recording: bool = False
    pairs: int = 0
    session_dir: str = ""
    index_path: str = ""


class PairDebugRecorder:
    """Save raw/aligned RGB-D pairs and the original ``pairs.csv`` index."""

    HEADER = (
        "pair_id", "rgb_frame_id", "depth_frame_id", "rgb_host_ts_ns",
        "depth_host_ts_ns", "rgb_device_ts_us", "depth_device_ts_us",
        "rgb_arrival_ts_ns", "depth_arrival_ts_ns",
        "rgb_sync_ts_ns", "depth_sync_ts_ns",
        "rgb_clock_quality", "depth_clock_quality",
        "rgb_clock_reason", "depth_clock_reason",
        "rgb_clock_reset_count", "depth_clock_reset_count",
        "delta_ns", "align_mode", "rgb_file", "depth_raw_file",
        "depth_aligned_file",
    )

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._file = None
        self._writer = None
        self._session_dir = None
        self._pairs = 0

    def start(self, root: str) -> bool:
        self.stop()
        now = datetime.now()
        session = Path(root) / f"{now:%Y%m%d_%H%M%S}_{now.microsecond // 1000:03d}"
        try:
            session.mkdir(parents=True, exist_ok=False)
            self._file = open(session / "pairs.csv", "w", newline="", encoding="utf-8")
            self._writer = csv.DictWriter(self._file, fieldnames=self.HEADER)
            self._writer.writeheader()
            self._file.flush()
        except OSError:
            if self._file is not None:
                self._file.close()
            self._file = self._writer = None
            return False
        self._session_dir = session
        self._pairs = 0
        return True

    def stop(self) -> None:
        with self._lock:
            if self._file is not None:
                self._file.flush()
                self._file.close()
            self._file = self._writer = None

    def record(
        self,
        *,
        rgb_image,
        depth_raw,
        depth_aligned,
        rgb_frame_id: int,
        depth_frame_id: int,
        rgb_host_ts_ns: int,
        depth_host_ts_ns: int,
        rgb_device_ts_us: int = 0,
        depth_device_ts_us: int = 0,
        rgb_arrival_ts_ns: int = 0,
        depth_arrival_ts_ns: int = 0,
        rgb_sync_ts_ns: int = 0,
        depth_sync_ts_ns: int = 0,
        rgb_clock_quality: str = "host_fallback",
        depth_clock_quality: str = "host_fallback",
        rgb_clock_reason: str = "",
        depth_clock_reason: str = "",
        rgb_clock_reset_count: int = 0,
        depth_clock_reset_count: int = 0,
        delta_ns: int = 0,
        align_mode: str = "software",
    ) -> bool:
        with self._lock:
            if self._writer is None or self._session_dir is None:
                return False
            import cv2
            self._pairs += 1
            base = f"pair_{self._pairs:06d}"
            names = {
                "rgb_file": f"{base}_rgb.png",
                "depth_raw_file": f"{base}_depth_raw_u16.png",
                "depth_aligned_file": f"{base}_depth_aligned_u16.png",
            }
            try:
                if not cv2.imwrite(str(self._session_dir / names["rgb_file"]), rgb_image):
                    raise OSError("RGB PNG write failed")
                if not cv2.imwrite(str(self._session_dir / names["depth_raw_file"]), depth_raw):
                    raise OSError("raw depth PNG write failed")
                if not cv2.imwrite(str(self._session_dir / names["depth_aligned_file"]), depth_aligned):
                    raise OSError("aligned depth PNG write failed")
                self._writer.writerow({
                    "pair_id": self._pairs,
                    "rgb_frame_id": rgb_frame_id,
                    "depth_frame_id": depth_frame_id,
                    "rgb_host_ts_ns": rgb_host_ts_ns,
                    "depth_host_ts_ns": depth_host_ts_ns,
                    "rgb_device_ts_us": rgb_device_ts_us,
                    "depth_device_ts_us": depth_device_ts_us,
                    "rgb_arrival_ts_ns": rgb_arrival_ts_ns,
                    "depth_arrival_ts_ns": depth_arrival_ts_ns,
                    "rgb_sync_ts_ns": rgb_sync_ts_ns,
                    "depth_sync_ts_ns": depth_sync_ts_ns,
                    "rgb_clock_quality": rgb_clock_quality,
                    "depth_clock_quality": depth_clock_quality,
                    "rgb_clock_reason": rgb_clock_reason,
                    "depth_clock_reason": depth_clock_reason,
                    "rgb_clock_reset_count": rgb_clock_reset_count,
                    "depth_clock_reset_count": depth_clock_reset_count,
                    "delta_ns": delta_ns,
                    "align_mode": align_mode,
                    **names,
                })
                if self._pairs % 30 == 0:
                    self._file.flush()
                return True
            except Exception:
                self._pairs -= 1
                return False

    def stats(self) -> PairRecorderStats:
        with self._lock:
            return PairRecorderStats(
                self._writer is not None,
                self._pairs,
                str(self._session_dir) if self._session_dir else "",
                str(self._session_dir / "pairs.csv") if self._session_dir else "",
            )


# ============================================================
# Skeleton3DRecorder
# ============================================================

_REHAB22_JOINT_NAMES = [
    "Waist", "Spine", "Chest", "Neck", "Head", "HeadTip",
    "LeftCollar", "LeftUpperArm", "LeftForearm", "LeftHand",
    "RightCollar", "RightUpperArm", "RightForearm", "RightHand",
    "LeftUpperLeg", "LeftLowerLeg", "LeftFoot", "LeftToes",
    "RightUpperLeg", "RightLowerLeg", "RightFoot", "RightToes",
]

_WIDE_JOINT_NAMES = [
    "waist", "spine", "chest", "neck", "head", "head_tip",
    "l_collar", "l_shoulder", "l_elbow", "l_hand", "r_collar",
    "r_shoulder", "r_elbow", "r_hand", "l_hip", "l_knee",
    "l_foot", "l_toe", "r_hip", "r_knee", "r_foot", "r_toe",
]


class Skeleton3DRecorder:
    """Records skeleton_3d.csv, skeleton3d_debug.csv, and meta.json."""

    def __init__(self):
        self._lock = threading.Lock()
        self._csv: Optional[TextIO] = None
        self._csv_writer: Any = None
        self._detailed_csv: Optional[TextIO] = None
        self._detailed_writer: Any = None
        self._debug_csv: Optional[TextIO] = None
        self._debug_writer: Any = None
        self._session_dir: Optional[Path] = None
        self._meta = _SessionMeta()
        self._recording: bool = False
        self._frame_count: int = 0
        self._row_count: int = 0
        self._skipped: int = 0

    def start(self, session_dir: str, meta: Optional[Dict] = None) -> bool:
        """Begin recording to the given session directory."""
        self.stop()
        sd = Path(session_dir)
        sd.mkdir(parents=True, exist_ok=True)

        csv_path = sd / "skeleton_3d.csv"
        detailed_path = sd / "skeleton_3d_detailed.csv"
        debug_path = sd / "skeleton3d_debug.csv"

        try:
            self._csv = open(csv_path, "w", newline="", encoding="utf-8")
            self._detailed_csv = open(detailed_path, "w", newline="", encoding="utf-8")
            self._debug_csv = open(debug_path, "w", newline="", encoding="utf-8")
        except OSError:
            if self._csv:
                self._csv.close()
                self._csv = None
            if self._debug_csv:
                self._debug_csv.close()
                self._debug_csv = None
            if self._detailed_csv:
                self._detailed_csv.close()
                self._detailed_csv = None
            return False

        # Build CSV header: timestamp_ns,frame_id,..., plus 22 joints × (x,y,z,score,valid)
        wide_header = ["frame_idx"]
        for index, name in enumerate(_WIDE_JOINT_NAMES):
            wide_header.extend(
                f"{index:02d}_{name}_{axis}" for axis in ("x", "y", "z")
            )
        self._csv_writer = csv.writer(self._csv)
        self._csv_writer.writerow(wide_header)

        header = ["timestamp_ns", "frame_id", "pair_id", "dt_seconds", "bbox_mode"]
        for name in _REHAB22_JOINT_NAMES:
            header.extend([f"{name}_x", f"{name}_y", f"{name}_z",
                          f"{name}_score", f"{name}_valid"])
        self._detailed_writer = csv.DictWriter(self._detailed_csv, fieldnames=header)
        self._detailed_writer.writeheader()

        debug_header = [
            "frame_idx", "frame_id", "timestamp_ns", "dt_seconds", "pair_id",
            "joint_index", "joint_name", "canonical_name", "u", "v",
            "pose_conf", "raw_pose_conf", "raw_depth_single_point_mm",
            "sampled_depth_mm", "depth_valid", "sample_method", "sample_reason",
            "body_depth_ref_mm", "background_depth_ref_mm",
            "rejected_as_background", "edge_ambiguous", "foreground_recovered",
            "foreground_pixel_count", "rejected_background_count", "used_radius",
            "x_raw", "y_raw", "z_raw", "raw_valid", "x_ema", "y_ema",
            "z_ema", "ema_valid", "ema_alpha", "ema_reason", "invalid_hold_count",
        ]
        self._debug_writer = csv.writer(self._debug_csv)
        self._debug_writer.writerow(debug_header)

        # Meta
        self._meta = _SessionMeta(
            start_time=datetime.now(timezone.utc).isoformat(),
            **(meta or {}),
        )
        self._session_dir = sd
        self._write_meta("")

        self._recording = True
        self._frame_count = 0
        self._row_count = 0
        self._skipped = 0
        return True

    def stop(self) -> None:
        # Serialize close with record().  Without this lock the UI could close
        # the CSV while the pipeline worker was inside writerow().
        with self._lock:
            if not self._recording:
                return
            self._recording = False
            self._write_meta(datetime.now(timezone.utc).isoformat())
            if self._csv:
                self._csv.flush()
                self._csv.close()
                self._csv = None
            if self._debug_csv:
                self._debug_csv.flush()
                self._debug_csv.close()
                self._debug_csv = None
            if self._detailed_csv:
                self._detailed_csv.flush()
                self._detailed_csv.close()
                self._detailed_csv = None

    @property
    def is_recording(self) -> bool:
        return self._recording

    def stats(self) -> RecorderStats:
        return RecorderStats(
            recording=self._recording,
            frames=self._frame_count,
            rows=self._row_count,
            skipped_frames=self._skipped,
            session_dir=str(self._session_dir) if self._session_dir else "",
            csv_path=str(self._session_dir / "skeleton_3d.csv") if self._session_dir else "",
        )

    def record(self, timestamp_ns: int, frame_id: int, pair_id: int,
               dt_seconds: float, bbox_mode: str,
               joints_3d: List[List[float]], joints_2d=None,
               raw_joints_3d=None, ema_joints_3d=None,
               depth_debug=None, ema_debug=None) -> bool:
        """
        Record one frame of 3D skeleton data.
        joints_3d: list of 22 [x, y, z] arrays.
        """
        with self._lock:
            # Re-check after acquiring the lock: stop() may have run between
            # the caller's frame check and this write.
            if not self._recording or self._csv is None or self._csv_writer is None:
                return False
            if len(joints_3d) != 22:
                self._skipped += 1
                return False

            frame_index = self._frame_count + self._skipped

            def value(item, name, default=""):
                if item is None:
                    return default
                if isinstance(item, dict):
                    result = item.get(name, default)
                else:
                    result = getattr(item, name, default)
                return getattr(result, "value", result)

            if self._debug_writer is not None:
                for index in range(22):
                    point2d = joints_2d[index] if joints_2d and index < len(joints_2d) else None
                    raw = raw_joints_3d[index] if raw_joints_3d and index < len(raw_joints_3d) else None
                    ema = ema_joints_3d[index] if ema_joints_3d and index < len(ema_joints_3d) else None
                    depth = depth_debug[index] if depth_debug and index < len(depth_debug) else None
                    ema_info = ema_debug[index] if ema_debug and index < len(ema_debug) else None
                    self._debug_writer.writerow([
                        frame_index, frame_id, timestamp_ns, f"{dt_seconds:.6f}", pair_id,
                        index, _REHAB22_JOINT_NAMES[index], value(point2d, "name"),
                        value(point2d, "x"), value(point2d, "y"),
                        value(point2d, "score"), value(point2d, "raw_score"),
                        value(depth, "depth_raw_mm"), value(depth, "depth_meters", 0) * 1000,
                        str(bool(value(depth, "valid", False))).lower(),
                        str(value(depth, "method")), value(depth, "reason"),
                        value(depth, "body_depth_ref_mm"), value(depth, "background_depth_ref_mm"),
                        str(bool(value(depth, "rejected_as_background", False))).lower(),
                        str(bool(value(depth, "edge_ambiguous", False))).lower(),
                        str(bool(value(depth, "foreground_recovered", False))).lower(),
                        value(depth, "foreground_pixel_count"),
                        value(depth, "rejected_background_count"), value(depth, "used_radius"),
                        value(raw, "x"), value(raw, "y"), value(raw, "z"),
                        str(bool(value(raw, "valid", False))).lower(),
                        value(ema, "x"), value(ema, "y"), value(ema, "z"),
                        str(bool(value(ema, "valid", False))).lower(),
                        value(ema_info, "alpha"), value(ema_info, "reason"),
                        value(ema_info, "invalid_hold_count", 0),
                    ])

            row = {
                "timestamp_ns": timestamp_ns,
                "frame_id": frame_id,
                "pair_id": pair_id,
                "dt_seconds": f"{dt_seconds:.6f}",
                "bbox_mode": bbox_mode,
            }
            for i, name in enumerate(_REHAB22_JOINT_NAMES):
                j = joints_3d[i] if i < len(joints_3d) else [0, 0, 0, 0, 0]
                x, y, z = j[0], j[1], j[2]
                score = j[3] if len(j) > 3 else 0.0
                valid = j[4] if len(j) > 4 else int(bool(x or y or z))
                row[f"{name}_x"] = f"{x:.6f}"
                row[f"{name}_y"] = f"{y:.6f}"
                row[f"{name}_z"] = f"{z:.6f}"
                row[f"{name}_score"] = f"{score:.4f}"
                row[f"{name}_valid"] = valid

            try:
                if self._detailed_writer is not None:
                    self._detailed_writer.writerow(row)
                if not any(bool(joint[4]) for joint in joints_3d):
                    self._skipped += 1
                    return False
                wide_row = [self._frame_count]
                for joint in joints_3d:
                    if bool(joint[4]):
                        wide_row.extend(
                            (-float(joint[0]), -float(joint[1]), -float(joint[2]))
                        )
                    else:
                        wide_row.extend(("", "", ""))
                self._csv_writer.writerow(wide_row)
                self._frame_count += 1
                self._row_count += 1
            except Exception:
                self._skipped += 1
                return False

        return True

    def _write_meta(self, end_time: str) -> None:
        """Write or update meta.json."""
        if not self._session_dir:
            return
        meta_path = self._session_dir / "meta.json"
        data = asdict(self._meta)
        data["end_time"] = end_time
        data["total_frames"] = self._frame_count
        data["total_rows"] = self._row_count
        try:
            meta_path.write_text(json.dumps(data, indent=2, ensure_ascii=False),
                                encoding="utf-8")
        except OSError:
            pass


# ============================================================
# EmgRecorder
# ============================================================

class EmgRecorder:
    """Records emg_raw.csv, emg_features.csv, and emg_summary.json."""

    def __init__(self):
        self._lock = threading.RLock()
        self._action_dir: Optional[Path] = None
        self._raw_csv: Optional[TextIO] = None
        self._raw_writer: Any = None
        self._feat_csv: Optional[TextIO] = None
        self._feat_writer: Any = None
        self._raw_count: int = 0
        self._feat_count: int = 0
        self._feature_frames: int = 0
        self._state_counts = {name: 0 for name in ("REST", "SMOOTH_FLEX", "TREMOR", "FATIGUE")}
        self._rms_sum = 0.0
        self._max_rms = 0.0
        self._fatigue_sum = 0.0
        self._channel_rms = {0: [0, 0.0], 1: [0, 0.0]}
        self._source_mode = "disabled"
        self._capture_backend = "serial"
        self._strict_real_mode = True
        self._final_status: Any = None

    def start(self, action_dir: str) -> bool:
        with self._lock:
            self.stop()
            ad = Path(action_dir)
            try:
                ad.mkdir(parents=True, exist_ok=True)
                self._raw_csv = open(ad / "emg_raw.csv", "w", newline="", encoding="utf-8")
                self._feat_csv = open(ad / "emg_features.csv", "w", newline="", encoding="utf-8")
            except OSError:
                for output in (self._raw_csv, self._feat_csv):
                    if output:
                        output.close()
                self._raw_csv = self._feat_csv = None
                return False
            self._action_dir = ad
            self._raw_writer = csv.writer(self._raw_csv)
            self._raw_writer.writerow(
                ["timestamp_ns", "packet_seq", "sample_index", "ch0", "ch1", "source_mode"]
            )
            self._feat_writer = csv.writer(self._feat_csv)
            self._feat_writer.writerow(
                ["timestamp_ns", "seq", "ch", "rms", "zcr", "cv", "fatigue_index", "state"]
            )
            self._raw_count = self._feat_count = self._feature_frames = 0
            self._state_counts = {name: 0 for name in self._state_counts}
            self._rms_sum = self._max_rms = self._fatigue_sum = 0.0
            self._channel_rms = {0: [0, 0.0], 1: [0, 0.0]}
            self._final_status = None
            return True

    def stop(self) -> None:
        with self._lock:
            self._write_summary_locked()
            for output in (self._raw_csv, self._feat_csv):
                if output:
                    output.flush()
                    output.close()
            self._raw_csv = self._feat_csv = None
            self._raw_writer = self._feat_writer = None

    @property
    def is_recording(self) -> bool:
        with self._lock:
            return self._raw_writer is not None and self._feat_writer is not None

    def set_link_context(self, mode: str, capture_backend: str, strict_real_mode: bool) -> None:
        with self._lock:
            self._source_mode = mode
            self._capture_backend = capture_backend
            self._strict_real_mode = strict_real_mode

    def set_final_status(self, status: Any) -> None:
        with self._lock:
            self._final_status = status

    def record_raw(self, timestamp_ns: int, seq: int, channels: List[int]) -> None:
        self._record_raw(timestamp_ns, seq, 0, channels)

    def record_raw_sample(self, sample: Any) -> None:
        self._record_raw(
            sample.host_ts_ns,
            sample.packet_seq,
            sample.sample_index,
            list(sample.channels),
        )

    def _record_raw(self, timestamp_ns: int, seq: int, sample_index: int, channels: List[int]) -> None:
        with self._lock:
            if not self._raw_writer:
                return
            values = list(channels[:2])
            values.extend([0] * (2 - len(values)))
            self._raw_writer.writerow(
                [timestamp_ns, seq, sample_index, values[0], values[1], self._source_mode]
            )
            self._raw_count += 1
            if self._raw_count % 256 == 0:
                self._raw_csv.flush()

    def record_feature(self, timestamp_ns: int, seq: int, channel: int,
                       rms: float, zcr: float, cv: float,
                       fatigue_index: float, state: str) -> None:
        with self._lock:
            self._record_feature_row(timestamp_ns, seq, channel, rms, zcr, cv, fatigue_index, state)

    def record_feature_frame(self, frame: Any) -> None:
        with self._lock:
            if not self._feat_writer or not frame.valid:
                return
            self._feature_frames += 1
            for channel in frame.channels:
                state = getattr(channel.state, "name", str(channel.state))
                self._record_feature_row(
                    frame.host_ts_ns,
                    frame.seq,
                    channel.channel,
                    channel.rms,
                    channel.zcr,
                    channel.cv,
                    channel.fatigue_index,
                    state,
                )

    def _record_feature_row(self, timestamp_ns, seq, channel, rms, zcr, cv, fatigue, state):
        if not self._feat_writer:
            return
        state = str(state).upper()
        self._feat_writer.writerow(
            [timestamp_ns, seq, channel, f"{rms:.6f}", f"{zcr:.6f}",
             f"{cv:.6f}", f"{fatigue:.6f}", state]
        )
        self._feat_count += 1
        self._rms_sum += float(rms)
        self._max_rms = max(self._max_rms, float(rms))
        self._fatigue_sum += float(fatigue)
        if state in self._state_counts:
            self._state_counts[state] += 1
        if channel in self._channel_rms:
            self._channel_rms[channel][0] += 1
            self._channel_rms[channel][1] += float(rms)
        if self._feat_count % 128 == 0:
            self._feat_csv.flush()

    def _write_summary_locked(self) -> None:
        if not self._action_dir:
            return
        observations = sum(self._state_counts.values())
        denominator = observations or 1
        dominant = max(self._state_counts, key=self._state_counts.get)
        status = self._final_status
        link_state = getattr(status, "link_state", "unavailable")
        raw_chunks = int(getattr(status, "raw_chunks", 0))
        feature_frames = int(getattr(status, "feature_frames", self._feature_frames))
        strict_passed = (
            self._source_mode == "real"
            and link_state == "real-ok"
            and self._raw_count > 0
            and raw_chunks > 0
            and feature_frames > 0
        )
        summary = {
            "source_mode": self._source_mode,
            "capture_backend": self._capture_backend,
            "strict_real_mode": self._strict_real_mode,
            "strict_validation_passed": strict_passed,
            "link_state": link_state,
            "parse_errors": int(getattr(status, "parse_errors", 0)),
            "invalid_payloads": int(getattr(status, "invalid_payloads", 0)),
            "dropped_packets": int(getattr(status, "dropped_packets", 0)),
            "rpmsg_errors": int(getattr(status, "rpmsg_errors", 0)),
            "invalid_feature_packets": int(getattr(status, "invalid_feature_packets", 0)),
            "ble_command_errors": int(getattr(status, "ble_command_errors", 0)),
            "ble_status_timeouts": int(getattr(status, "ble_status_timeouts", 0)),
            "raw_rows": self._raw_count,
            "feature_frames": self._feature_frames,
            "feature_rows": self._feat_count,
            "active_ratio": (
                self._state_counts["SMOOTH_FLEX"]
                + self._state_counts["TREMOR"]
                + self._state_counts["FATIGUE"]
            ) / denominator,
            "fatigue_ratio": self._state_counts["FATIGUE"] / denominator,
            "tremor_ratio": self._state_counts["TREMOR"] / denominator,
            "avg_rms": self._rms_sum / denominator,
            "max_rms": self._max_rms,
            "avg_fatigue_index": self._fatigue_sum / denominator,
            "ch0_avg_rms": self._channel_average(0),
            "ch1_avg_rms": self._channel_average(1),
            "active_muscle_avg_rms": self._channel_average(0),
            "antagonist_muscle_avg_rms": self._channel_average(1),
            "dominant_state": dominant,
        }
        try:
            (self._action_dir / "emg_summary.json").write_text(
                json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
            )
        except OSError:
            pass

    def _channel_average(self, channel: int) -> float:
        count, total = self._channel_rms[channel]
        return total / count if count else 0.0
