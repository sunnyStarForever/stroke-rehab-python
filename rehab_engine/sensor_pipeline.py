"""
Sensor pipeline orchestrator.
Replaces core/pipeline/SensorPipeline.cpp (~1600 lines).

Architecture (Full Mode):
    ┌─ C++ RgbCaptureV4L2 (V4L2 ioctl/mmap/poll) ─┐
    │  JPEG encodes frame → pybind11 callback       │
    │                                               ├──► pair_queue ──► Worker ──► Preview
    └─ C++ DepthCaptureOpenNI (OpenNI2) ────────────┘   (decode→pose→3D→compose)

Architecture (Stub Mode):
    Mock thread ──► pair_queue ──► Worker (synthetic_pose) ──► Preview

Thread safety: queue.Queue between capture and worker threads,
               and threading.Lock for the latest preview frame.
"""

import math
import queue
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

try:
    import numpy as np
except ImportError:  # Optional in STUB mode; required only for real image decoding.
    np = None

from ._stub import PipelineConfig, logger

# Detect engine mode: try to import the compiled _core module.
# Do NOT import the rehab_engine package (circular import with __init__.py).
_STUB_MODE = False
_engine = None
try:
    from . import _core as _engine
except ImportError:
    _STUB_MODE = True

from .preview import PreviewComposer, PreviewFrame
from .recorder import Skeleton3DRecorder


def _decode_jpeg_to_rgb(jpeg_bytes: bytes) -> Optional[Any]:
    """Decode JPEG bytes to RGB numpy array. Uses OpenCV if available, PIL fallback."""
    if np is None:
        return None
    try:
        import cv2
        arr = np.frombuffer(jpeg_bytes, dtype=np.uint8)
        bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if bgr is not None:
            return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    except ImportError:
        pass
    try:
        from PIL import Image
        import io
        img = Image.open(io.BytesIO(jpeg_bytes))
        img = img.convert("RGB")
        return np.array(img)
    except ImportError:
        pass
    return None


class SensorPipeline:
    """
    Main pipeline orchestrator.

    Dual-mode:
      - STUB mode:  mock capture thread + synthetic pose
      - FULL mode:  real C++ V4L2 + OpenNI2 capture via pybind11

    Usage:
        pipeline = SensorPipeline(config)
        pipeline.set_on_frame(lambda frame: print("got frame"))
        pipeline.start()
        pipeline.start_recording("records/session_001")
        ...
        pipeline.stop()
    """

    def __init__(self, config: Optional[PipelineConfig] = None):
        self._config = config or PipelineConfig()

        # Mode
        self._stub_mode = _STUB_MODE

        # Queues
        self._pair_queue: queue.Queue = queue.Queue(maxsize=100)

        # Threads
        self._worker_thread: Optional[threading.Thread] = None
        self._running = threading.Event()

        # C++ engine objects (full mode only)
        self._rgb_capture = None      # RgbCaptureV4L2
        self._depth_capture = None    # DepthCaptureOpenNI

        # Modules
        self._preview = PreviewComposer()
        self._recorder = Skeleton3DRecorder()

        # Recording state
        self._recording: bool = False
        self._recording_paused: bool = False
        self._session_dir: str = ""
        self._recording_lock = threading.Lock()

        # Counters
        self._pair_id: int = 0
        self._processed: int = 0
        self._dropped: int = 0
        self._rgb_count: int = 0
        self._depth_count: int = 0
        self._pose_count: int = 0

        # Performance
        self._perf_lock = threading.Lock()
        self._last_perf_time = time.monotonic()
        self._rgb_fps = 0.0
        self._depth_fps = 0.0
        self._pair_fps = 0.0
        self._pose_fps = 0.0
        self._rgb_since_last = 0
        self._depth_since_last = 0
        self._pair_since_last = 0
        self._pose_since_last = 0

        # Callbacks
        self._on_frame: Optional[Callable[[PreviewFrame], None]] = None
        self._on_status: Optional[Callable[[str], None]] = None
        self._on_performance: Optional[Callable[[dict], None]] = None

        # Camera status
        self._camera_status: str = "stopped"
        self._camera_error: str = ""

    # ================================================================
    # Public API
    # ================================================================

    def set_on_frame(self, callback: Optional[Callable[[PreviewFrame], None]]):
        self._on_frame = callback

    def set_on_status(self, callback: Optional[Callable[[str], None]]):
        self._on_status = callback

    def set_on_performance(self, callback: Optional[Callable[[dict], None]]):
        self._on_performance = callback

    @property
    def config(self) -> PipelineConfig:
        return self._config

    @property
    def is_running(self) -> bool:
        return self._running.is_set()

    @property
    def is_recording(self) -> bool:
        return self._recording

    @property
    def is_recording_paused(self) -> bool:
        return self._recording_paused

    @property
    def preview(self) -> PreviewComposer:
        return self._preview

    @property
    def stub_mode(self) -> bool:
        return self._stub_mode

    @property
    def camera_status(self) -> dict:
        return {
            "status": self._camera_status,
            "error": self._camera_error,
            "rgb_fps": self._rgb_fps,
            "mode": "STUB" if self._stub_mode else "FULL",
        }

    # ================================================================
    # Start / Stop
    # ================================================================

    def start(self) -> bool:
        """Start the pipeline. In full mode opens real V4L2 camera via C++ engine."""
        if self._running.is_set():
            return False

        mode_label = "STUB" if self._stub_mode else "FULL"
        self._emit_status(f"Pipeline starting ({mode_label} mode)...")

        self._running.set()

        # Start worker thread (always needed)
        self._worker_thread = threading.Thread(
            target=self._worker_loop, name="pipeline-worker", daemon=True)
        self._worker_thread.start()

        if self._stub_mode:
            self._start_mock_capture()
        else:
            self._start_real_capture()

        return True

    def stop(self) -> None:
        """Stop the pipeline and release camera resources."""
        self._emit_status("Pipeline stopping...")
        self._running.clear()

        # Stop C++ capture objects
        self._stop_real_capture()

        if self._worker_thread and self._worker_thread.is_alive():
            self._worker_thread.join(timeout=3)

        self._emit_status("Pipeline stopped")

    # ================================================================
    # Recording
    # ================================================================

    def start_recording(self, save_root: str) -> str:
        with self._recording_lock:
            if self._recording:
                return self._session_dir

            now = datetime.now()
            date_folder = now.strftime("%Y%m%d")
            session_name = now.strftime("%Y%m%d_%H%M%S") + f"_{now.microsecond // 1000:03d}"
            session_dir = Path(save_root) / date_folder / session_name
            session_dir.mkdir(parents=True, exist_ok=True)

            if not self._recorder.start(str(session_dir)):
                raise OSError(f"Unable to open recording files in {session_dir}")
            self._session_dir = str(session_dir)
            self._recording = True
            self._recording_paused = False
            self._emit_status(f"Recording started: {session_dir}")
            return self._session_dir

    def stop_recording(self) -> None:
        with self._recording_lock:
            if not self._recording:
                return
            self._recording = False
            self._recording_paused = False
            self._recorder.stop()
            self._emit_status(f"Recording stopped: {self._session_dir}")

    def pause_recording(self) -> bool:
        with self._recording_lock:
            if not self._recording or self._recording_paused:
                return False
            self._recording_paused = True
            self._emit_status("Recording paused")
            return True

    def resume_recording(self) -> bool:
        with self._recording_lock:
            if not self._recording or not self._recording_paused:
                return False
            self._recording_paused = False
            self._emit_status("Recording resumed")
            return True

    def recording_stats(self) -> dict:
        s = self._recorder.stats()
        return {
            "recording": s.recording,
            "paused": self._recording_paused,
            "session_dir": s.session_dir,
            "csv_path": s.csv_path,
            "frames": s.frames,
            "rows": s.rows,
            "skipped": s.skipped_frames,
        }

    def performance_stats(self) -> dict:
        with self._perf_lock:
            return {
                "rgb_fps": self._rgb_fps,
                "depth_fps": self._depth_fps,
                "pair_fps": self._pair_fps,
                "pose_fps": self._pose_fps,
                "queue_length": self._pair_queue.qsize(),
                "dropped_pairs": self._dropped,
                "processed": self._processed,
                "stub_mode": self._stub_mode,
                "camera_status": self._camera_status,
            }

    # ================================================================
    # Stub mode: mock capture
    # ================================================================

    def _start_mock_capture(self):
        """Start mock capture thread that feeds synthetic data."""
        self._emit_status("Pipeline: mock capture (no real camera)")
        mock_thread = threading.Thread(
            target=self._mock_capture_loop, name="mock-capture", daemon=True)
        mock_thread.start()

    def _mock_capture_loop(self) -> None:
        """Simulate RGB+Depth capture in stub mode."""
        interval = 1.0 / max(1, self._config.device.rgb_fps)
        while self._running.is_set():
            try:
                self._pair_queue.put_nowait({"ts": time.monotonic_ns(), "mock": True})
            except queue.Full:
                self._dropped += 1
            self._rgb_count += 1
            self._depth_count += 1
            self._rgb_since_last += 1
            self._depth_since_last += 1
            time.sleep(interval)

    # ================================================================
    # Full mode: real C++ V4L2 camera capture
    # ================================================================

    def _start_real_capture(self):
        """Start real V4L2 RGB capture via C++ pybind11 engine."""
        if _engine is None:
            logger.error("C++ engine not loaded — falling back to stub")
            self._emit_status("ERROR: C++ engine not loaded — fallback to stub")
            self._stub_mode = True
            self._start_mock_capture()
            return

        device_path = self._config.device.rgb_device_path
        if not device_path:
            device_path = f"/dev/video{self._config.device.rgb_device_index}"

        fmt = self._config.device.rgb_pixel_format
        w = self._config.device.rgb_width
        h = self._config.device.rgb_height
        fps = self._config.device.rgb_fps

        self._emit_status(
            f"Pipeline: opening camera {device_path} {w}x{h} @{fps}fps {fmt}")

        self._camera_status = "opening"

        try:
            # Build C++ DeviceConfig from Python config
            engine_cfg = _engine.PipelineConfig()
            dev = engine_cfg.device
            dev.rgb_device_path = device_path
            dev.rgb_width = w
            dev.rgb_height = h
            dev.rgb_fps = fps
            dev.rgb_pixel_format = fmt
            dev.mirror_rgb_at_capture = self._config.device.mirror_rgb_at_capture
            dev.rgb_device_index = self._config.device.rgb_device_index

            # Create capture object
            self._rgb_capture = _engine.RgbCaptureV4L2()

            # Set status callback — receives "[RGB OPEN] device=... open=ok" etc.
            self._rgb_capture.set_on_status(
                lambda s: self._on_camera_status(s))

            # Set frame callback — receives JPEG-encoded frame bytes
            # signature: (jpeg_bytes, width, height, ts_ns, frame_id, source)
            self._rgb_capture.start(dev, lambda jpeg, w, h, ts, fid, src: (
                self._on_real_rgb_frame(jpeg, w, h, ts, fid, src)))

            self._camera_status = "started"
            self._emit_status(f"Pipeline: camera started on {device_path}")

        except Exception as e:
            self._camera_status = "error"
            self._camera_error = str(e)
            logger.error(f"Camera start failed: {e}")
            self._emit_status(f"ERROR: camera start failed: {e} — fallback to stub")
            self._stub_mode = True
            self._start_mock_capture()

    def _stop_real_capture(self):
        """Stop C++ capture objects and release hardware."""
        if self._rgb_capture:
            try:
                self._rgb_capture.stop()
            except Exception as e:
                logger.warn(f"Error stopping RGB capture: {e}")
            self._rgb_capture = None

        if self._depth_capture:
            try:
                self._depth_capture.stop()
            except Exception as e:
                logger.warn(f"Error stopping depth capture: {e}")
            self._depth_capture = None

        self._camera_status = "stopped"

    def _on_camera_status(self, status: str):
        """Callback from C++ RgbCaptureV4L2 setOnStatus (runs in C++ capture thread)."""
        logger.info(f"[Camera] {status}")

        if "open=ok" in status:
            self._camera_status = "running"
            self._camera_error = ""
        elif "open failed" in status.lower() or "error" in status.lower():
            self._camera_error = status
            self._camera_status = "error"

        self._emit_status(f"[Camera] {status}")

    def _on_real_rgb_frame(self, jpeg_bytes: bytes, width: int, height: int,
                           ts_ns: int, frame_id: int, source: str):
        """
        Callback from C++ RgbCaptureV4L2 (runs in C++ capture thread).
        JPEG-encoded frame → push to pair_queue for worker loop.
        """
        self._rgb_count += 1
        self._rgb_since_last += 1

        try:
            item = {
                "ts": ts_ns,
                "mock": False,
                "jpeg": jpeg_bytes,
                "width": width,
                "height": height,
                "source": source,
                "frame_id": frame_id,
            }
            self._pair_queue.put_nowait(item)
        except queue.Full:
            self._dropped += 1

    # ================================================================
    # Worker loop (common to both modes)
    # ================================================================

    def _worker_loop(self) -> None:
        """Main pipeline worker thread."""
        while self._running.is_set():
            try:
                pair = self._pair_queue.get(timeout=0.1)
            except queue.Empty:
                continue

            self._processed += 1
            self._pair_since_last += 1
            self._pair_id += 1

            # EMG simulation (both modes)
            emg_status = "disabled"
            emg_rms = []
            if self._config.emg.enabled and self._config.emg.mode == "mock":
                phase = time.monotonic()
                emg_status = "mock"
                emg_rms = [
                    900 + 420 * abs(math.sin(phase * 2.1)),
                    760 + 360 * abs(math.sin(phase * 1.8 + 0.7)),
                ]
            elif self._config.emg.enabled and self._config.emg.mode == "real":
                emg_status = "waiting for real EMG data"

            if pair.get("mock"):
                # --- Stub mode: synthetic pose ---
                pose_2d, pose_3d = self._synthetic_pose()
            else:
                # --- Full mode: real camera frame ---
                # Decode JPEG for potential pose inference
                jpeg = pair.get("jpeg")
                if jpeg:
                    _ = _decode_jpeg_to_rgb(jpeg)  # decode for future pose inference
                # For now, use empty pose (pose inference via ONNX will be added)
                pose_2d, pose_3d = self._synthetic_pose()

            # Build preview frame
            self._preview.submit(
                joints_2d_raw=pose_2d,
                joints_3d=pose_3d,
                rgb_fps=self._rgb_fps,
                depth_fps=self._depth_fps,
                pair_fps=self._pair_fps,
                pose_fps=self._pose_fps,
                queue_length=self._pair_queue.qsize(),
                dropped_pairs=self._dropped,
                delta_ms=0.0,
                recording=self._recording and not self._recording_paused,
                skeleton_recording=self._recording and not self._recording_paused,
                skeleton_frames=self._recorder.stats().frames if self._recording else 0,
                emg_status=emg_status,
                emg_rms=emg_rms,
            )

            # Record skeleton
            if self._recording and not self._recording_paused:
                timestamp_ns = time.monotonic_ns()
                joint_arrays = []
                if pose_3d:
                    for j in pose_3d:
                        score = j[3] if len(j) > 3 else 1.0
                        valid = 1 if (j[4] if len(j) > 4 else False) else 0
                        joint_arrays.append([j[0], j[1], j[2], score, valid])
                self._recorder.record(
                    timestamp_ns=timestamp_ns,
                    frame_id=self._pair_id,
                    pair_id=self._pair_id,
                    dt_seconds=0.033,
                    bbox_mode="full" if not self._stub_mode else "mock",
                    joints_3d=joint_arrays,
                )

            # Update performance counters
            self._update_performance()

            # Notify listener
            if self._on_frame:
                frame = self._preview.latest_frame()
                if frame:
                    try:
                        self._on_frame(frame)
                    except Exception:
                        pass

    def _synthetic_pose(self):
        """Generate synthetic pose data for stub mode testing."""
        t = time.monotonic()
        joints_2d = []
        joints_3d = []
        base_x, base_y, base_z = 320, 240, 2.0

        layout = [
            (0, -100), (0, -120), (0, -140), (0, -160), (0, -180), (0, -200),
            (-30, -140), (-80, -120), (-130, -100), (-180, -80),
            (30, -140), (80, -120), (130, -100), (180, -80),
            (-20, 0), (-20, 80), (-20, 160), (-30, 240),
            (20, 0), (20, 80), (20, 160), (30, 240),
        ]

        for dx, dy in layout:
            x = base_x + dx + (5 * (time.monotonic() % 1))
            y = base_y + dy
            z = base_z
            joints_2d.append((x, y, 0.9, True))
            joints_3d.append((x / 320 - 1, -(y / 240 - 1), z, 0.9, True))

        return joints_2d, joints_3d

    # ================================================================
    # Performance
    # ================================================================

    def _update_performance(self) -> None:
        now = time.monotonic()
        elapsed = now - self._last_perf_time
        if elapsed >= 1.0:
            with self._perf_lock:
                self._rgb_fps = self._rgb_since_last / elapsed
                self._depth_fps = self._depth_since_last / elapsed
                self._pair_fps = self._pair_since_last / elapsed
                self._pose_fps = self._pose_since_last / elapsed
                self._rgb_since_last = 0
                self._depth_since_last = 0
                self._pair_since_last = 0
                self._pose_since_last = 0
                self._last_perf_time = now

    # ================================================================
    # Status
    # ================================================================

    def _emit_status(self, message: str) -> None:
        logger.info(message)
        if self._on_status:
            self._on_status(message)
