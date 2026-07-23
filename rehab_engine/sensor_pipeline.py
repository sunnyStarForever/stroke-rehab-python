"""
Sensor pipeline orchestrator.
Replaces core/pipeline/SensorPipeline.cpp (~1600 lines).

Full mode data flow:
  SyncedCapture(C++: RGB+Depth+SyncManager) → pair_queue → Worker
    → YOLO detect → RTMPose infer → Halpe→Rehab22 map
    → DepthSample → JointProject3D → EMA filter → Smoother
    → Preview + Recording + Scoring

Real-data policy:
  Synthetic/mock camera, depth and skeleton data are disabled.  If native
  capture support or real devices are unavailable, start() fails instead of
  substituting fake frames.
"""

from __future__ import annotations

import math
import json
import os as _os
import queue
import threading
import time
from collections import deque
from datetime import datetime
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

try:
    import numpy as np
except ImportError:
    np = None

from . import PipelineConfig, logger

_STUB_MODE = False
_REAL_CAPTURE_READY = False
_engine = None
try:
    from . import _core as _engine
except ImportError:
    _engine = None
else:
    _REAL_CAPTURE_READY = all(
        hasattr(_engine, name)
        for name in ("RgbCaptureV4L2", "DepthCaptureOpenNI", "DeviceConfig")
    )

from .preview import PreviewComposer, PreviewFrame
from .recorder import (
    PairDebugRecorder, RecordingFrame, RgbDepthVideoRecorder, Skeleton3DRecorder,
)
from .emg import EmgManager
from .inference import (
    AdaptiveRoiTracker,
    BoundingBox2D as PythonBoundingBox,
    PersonDetector as PythonPersonDetector,
    RtmposeEstimator as PythonRtmposeEstimator,
    map_halpe26_to_rehab22,
)
from .capture import NativeRgbDepthBackend
from .alignment import SoftwareRegistrationAligner, load_calibration as load_registration_calibration
from .pose3d import (
    DepthSampler as PythonDepthSampler,
    EmaSkeletonFilter,
    JointProjector3D as PythonJointProjector3D,
    SkeletonSmoother as PythonSkeletonSmoother,
    make_rehab22_joints,
)


@dataclass(frozen=True)
class RecordingOptions:
    save_root: str = "records"
    record_skeleton: bool = True
    record_rgb: bool = True
    record_depth: bool = False
    mirror_preview: bool = False
    record_valid_3d_only: bool = False


class SensorPipeline:
    """Python-owned pipeline orchestrator for real RGB-D hardware only."""

    def __init__(self, config: Optional[PipelineConfig] = None):
        self._config = config or PipelineConfig()
        self._stub_mode = False
        self._real_capture_ready = _REAL_CAPTURE_READY
        self._pair_queue: queue.Queue = queue.Queue(
            maxsize=max(1, int(self._config.pose.max_pair_queue))
        )
        self._worker_thread: Optional[threading.Thread] = None
        self._running = threading.Event()
        self._accept_frames = threading.Event()
        self._stopping = threading.Event()
        self._start_done = threading.Event()
        self._start_done.set()
        self._stop_lock = threading.Lock()
        self._stop_callbacks: List[Callable[[bool, str], None]] = []

        # Python-owned capture orchestration; low-level drivers are adapters.
        self._capture_backend = None

        # Python owns pose inference and all 2D -> 3D processing. Native pose
        # objects exist only in explicitly built legacy compatibility modules.
        self._person_detector = None
        self._pose_estimator = None
        self._roi_tracker = None
        self._halpe_mapper = None
        self._depth_sampler = PythonDepthSampler(self._config.depth_sampler)
        self._joint_projector = PythonJointProjector3D()
        self._ema_filter = EmaSkeletonFilter(self._config.skeleton_filter)
        self._smoother = PythonSkeletonSmoother(self._config.pose.smoothing_alpha)
        self._pose_models_ready = False
        self._python_inference = False
        self._frame_counter = 0
        self._last_pose_time = 0.0

        self._preview = PreviewComposer()
        self._recorder = Skeleton3DRecorder()
        self._video_recorder = RgbDepthVideoRecorder()
        self._pair_recorder = PairDebugRecorder()
        self._emg = EmgManager(self._config.emg)
        self._emg.set_on_status(lambda status: self._emit_status(status.message))

        self._recording = False
        self._recording_paused = False
        self._session_dir = ""
        self._recording_options = RecordingOptions()
        self._recording_started_at = ""
        self._recording_lock = threading.Lock()

        self._pair_id = 0
        self._processed = 0
        self._dropped = 0
        self._rgb_count = 0
        self._depth_count = 0
        self._pose_count = 0

        self._perf_lock = threading.Lock()
        self._last_perf_time = time.monotonic()
        self._rgb_fps = 0.0
        self._depth_fps = 0.0
        self._sync_fps = 0.0
        self._worker_fps = 0.0
        self._pose_fps = 0.0
        self._last_yolo_ms = 0.0
        self._last_pose_ms = 0.0
        self._last_record_write_ms = 0.0
        self._pose_ms_samples = deque(maxlen=300)
        self._yolo_ms_samples = deque(maxlen=300)
        self._rgb_since_last = 0
        self._depth_since_last = 0
        self._sync_since_last = 0
        self._worker_since_last = 0
        self._pose_since_last = 0

        self._on_frame: Optional[Callable] = None
        self._on_status: Optional[Callable] = None
        self._on_performance: Optional[Callable] = None
        self._camera_status = "stopped"
        self._camera_error = ""

        # Skeleton cache — pose runs every Nth frame, reuse last result in between
        self._last_j2d: list = []
        self._last_j3d: list = []
        self._last_bbox = None
        self._last_rehab2d = []
        self._last_depth_debug = []
        self._last_raw_j3d = []
        self._last_ema_j3d = []
        self._last_ema_debug = []

        # Depth diagnostic counters
        self._depth_has_data = False
        self._depth_empty_count = 0

        # Spatial alignment (depth → RGB coordinate frame)
        self._align_ready = False
        self._software_aligner = None
        self._hardware_d2c_active = False

    # ── Public API ────────────────────────────────────────────────

    def set_on_frame(self, cb): self._on_frame = cb
    def set_on_status(self, cb): self._on_status = cb
    def set_on_performance(self, cb): self._on_performance = cb

    @property
    def config(self): return self._config
    @property
    def is_running(self): return self._running.is_set()
    @property
    def is_stopping(self): return self._stopping.is_set()
    @property
    def is_recording(self): return self._recording
    @property
    def is_recording_paused(self): return self._recording_paused
    @property
    def preview(self): return self._preview
    @property
    def stub_mode(self): return False
    @property
    def camera_status(self):
        mode = "FULL" if self._real_capture_ready else "REAL_UNAVAILABLE"
        return {"status": self._camera_status, "error": self._camera_error,
                "rgb_fps": self._rgb_fps, "mode": mode}
    @property
    def emg_status(self): return self._emg.runtime_status()

    def start_action_recording(self, action_dir: str) -> bool:
        """Start EMG artifacts for one course action, matching the original layout."""
        return self._emg.start_recording(action_dir)

    def stop_action_recording(self) -> None:
        self._emg.stop_recording()

    # ── Start / Stop ──────────────────────────────────────────────

    def start(self) -> bool:
        if self._running.is_set() or self._stopping.is_set():
            return False
        if self._camera_status == "stop_error":
            self._emit_status(
                "ERROR: previous camera shutdown was incomplete; restart the application")
            return False
        if not self._real_capture_ready:
            self._camera_status = "configuration_error"
            self._camera_error = (
                "真实 RGB-D 采集核心不可用：未加载包含 RgbCaptureV4L2/"
                "DepthCaptureOpenNI 的 _core 扩展")
            self._emit_status(f"ERROR: {self._camera_error}")
            return False
        if self._config.device.rgb_fps != 30 or self._config.device.depth_fps != 30:
            self._camera_status = "configuration_error"
            self._camera_error = "真实 RGB 与 Depth 采集必须同时配置为 30 FPS"
            self._emit_status(f"ERROR: {self._camera_error}")
            return False
        self._emit_status("Pipeline starting (FULL real-data mode)...")
        self._start_done.clear()
        self._drain_pair_queue()
        self._reset_performance_counters()
        self._ema_filter.reset()
        self._smoother.reset()
        if self._roi_tracker is not None:
            self._roi_tracker.reset()
        self._last_pose_time = 0.0
        self._last_j2d = []
        self._last_j3d = []
        self._last_bbox = None
        self._last_rehab2d = []
        self._last_depth_debug = []
        self._last_raw_j3d = []
        self._last_ema_j3d = []
        self._last_ema_debug = []
        if self._config.record_pairs and not self._pair_recorder.start(
            str(self._config.record_path)
        ):
            self._emit_status("WARNING: RGB-D pair debug recorder failed to start")
        self._accept_frames.set()
        self._running.set()
        self._worker_thread = threading.Thread(
            target=self._worker_loop, name="pipeline-worker", daemon=True)
        self._worker_thread.start()
        if not self._start_real_capture():
            self._accept_frames.clear()
            self._running.clear()
            if self._worker_thread and self._worker_thread.is_alive():
                self._worker_thread.join(timeout=1.0)
            self._pair_recorder.stop()
            self._start_done.set()
            return False
        emg_ok = self._emg.start()
        strict_real = (
            self._config.emg.enabled
            and bool(getattr(self._config.emg, "strict_real_mode", True))
        )
        if not emg_ok and strict_real:
            self._accept_frames.clear()
            self._running.clear()
            try:
                self._stop_real_capture()
            except Exception as exc:
                self._camera_error = str(exc)
            if self._worker_thread and self._worker_thread.is_alive():
                self._worker_thread.join(timeout=1.0)
            self._pair_recorder.stop()
            self._camera_status = "error"
            self._camera_error = self._emg.runtime_status().message
            self._start_done.set()
            self._emit_status(f"ERROR: {self._camera_error}")
            return False
        self._start_done.set()
        return True

    def stop(self, on_complete: Optional[Callable[[bool, str], None]] = None) -> None:
        """Request an ordered stop and return immediately.

        Completion means capture callbacks are disabled, the camera stop was
        requested, the worker exited, and recording files were flushed/closed.
        """
        call_immediately = False
        with self._stop_lock:
            if on_complete:
                self._stop_callbacks.append(on_complete)
            if self._stopping.is_set():
                return
            if not self._running.is_set() and not self._recording:
                call_immediately = True
            else:
                self._stopping.set()
                self._accept_frames.clear()
                self._running.clear()
                self._camera_status = "stopping"
                self._emit_status("Pipeline stopping...")
                threading.Thread(
                    target=self._finish_stop,
                    name="pipeline-stop", daemon=True,
                ).start()
        if call_immediately:
            self._notify_stop_complete(True, "Pipeline already stopped")

    def _finish_stop(self) -> None:
        errors = []
        if not self._start_done.wait(timeout=10.0):
            errors.append("pipeline start did not finish within 10 seconds")
        try:
            self._emg.stop()
        except Exception as exc:
            errors.append(f"emg: {exc}")
        try:
            self._stop_real_capture()
        except Exception as exc:
            errors.append(f"camera: {exc}")

        self._drain_pair_queue()
        worker = self._worker_thread
        if worker and worker is not threading.current_thread() and worker.is_alive():
            worker.join(timeout=5.0)
            if worker.is_alive():
                errors.append("pipeline worker did not exit within 5 seconds")

        # No worker may start a new write after _running was cleared. Recorder
        # stop shares its write lock, so an in-flight row finishes first.
        try:
            self.stop_recording()
        except Exception as exc:
            errors.append(f"recorder: {exc}")
        self._pair_recorder.stop()

        self._worker_thread = None
        self._ema_filter.reset()
        self._smoother.reset()
        self._last_pose_time = 0.0
        self._camera_status = "stop_error" if errors else "stopped"
        self._camera_error = "; ".join(errors)
        self._stopping.clear()
        success = not errors
        message = self._camera_error or "Pipeline stopped"
        self._emit_status(message)
        self._notify_stop_complete(success, message)

    def _notify_stop_complete(self, success: bool, message: str) -> None:
        with self._stop_lock:
            callbacks = self._stop_callbacks[:]
            self._stop_callbacks.clear()
        for callback in callbacks:
            try:
                callback(success, message)
            except Exception:
                pass

    def _drain_pair_queue(self) -> None:
        try:
            while True:
                self._pair_queue.get_nowait()
        except queue.Empty:
            pass

    def _enqueue_pair(self, pair: dict) -> None:
        """Keep the newest pairs, matching the original bounded deque."""
        try:
            self._pair_queue.put_nowait(pair)
            return
        except queue.Full:
            pass
        try:
            self._pair_queue.get_nowait()
            self._dropped += 1
        except queue.Empty:
            pass
        try:
            self._pair_queue.put_nowait(pair)
        except queue.Full:
            self._dropped += 1

    def _reset_performance_counters(self) -> None:
        with self._perf_lock:
            self._pair_id = self._processed = self._dropped = 0
            self._rgb_count = self._depth_count = self._pose_count = 0
            self._rgb_fps = self._depth_fps = self._sync_fps = 0.0
            self._worker_fps = self._pose_fps = 0.0
            self._rgb_since_last = self._depth_since_last = 0
            self._sync_since_last = self._worker_since_last = 0
            self._pose_since_last = 0
            self._last_perf_time = time.monotonic()
            self._pose_ms_samples.clear()
            self._yolo_ms_samples.clear()

    # ── Recording ─────────────────────────────────────────────────

    def start_recording(self, save_root) -> str:
        with self._recording_lock:
            if self._recording: return self._session_dir
            options = (
                save_root if isinstance(save_root, RecordingOptions)
                else RecordingOptions(save_root=str(save_root))
            )
            if not (options.record_skeleton or options.record_rgb or options.record_depth):
                raise ValueError("At least one recording output must be enabled")
            now = datetime.now()
            date_folder = now.strftime("%Y%m%d")
            output_root = Path(options.save_root)
            if output_root.name != "output":
                output_root = output_root / "output"
            day_dir = output_root / date_folder
            session_dir = next(
                day_dir / f"session_{index:03d}"
                for index in range(1, 10_000)
                if not (day_dir / f"session_{index:03d}").exists()
            )
            session_dir.mkdir(parents=True, exist_ok=True)
            depth_transform = None
            if options.record_depth and not self._hardware_d2c_active:
                recorder_aligner = SoftwareRegistrationAligner(
                    load_registration_calibration(self._calibration_path()))
                rgb_size = (
                    self._config.device.rgb_width,
                    self._config.device.rgb_height,
                )
                depth_transform = lambda depth, unit: recorder_aligner.align(
                    depth, rgb_size, unit)
            video_ok = self._video_recorder.start(
                str(session_dir),
                self._config.device.rgb_fps,
                self._config.device.rgb_width,
                self._config.device.rgb_height,
                options.record_rgb,
                options.record_depth,
                queue_capacity=self._config.recording_queue_capacity,
                depth_transform=depth_transform,
            )
            if not video_ok:
                raise OSError(f"Cannot open video recording files in {session_dir}")
            meta = {
                "rgb_width": self._config.device.rgb_width,
                "rgb_height": self._config.device.rgb_height,
                "depth_width": self._config.device.depth_width,
                "depth_height": self._config.device.depth_height,
                "rgb_fps": self._config.device.rgb_fps,
                "depth_fps": self._config.device.depth_fps,
                "pose_model_path": self._config.pose.model_path,
                "detector_model_path": self._config.pose.detector_model_path,
                "pose_interval": self._config.pose.pose_interval,
                "hardware_d2c_enabled": self._config.device.enable_hardware_d2c,
                "mirror_rgb_at_capture": self._config.device.mirror_rgb_at_capture,
                "mirror_preview": options.mirror_preview,
            }
            if options.record_skeleton and not self._recorder.start(str(session_dir), meta):
                self._video_recorder.stop()
                raise OSError(f"Cannot open recording files in {session_dir}")
            self._session_dir = str(session_dir)
            self._recording_options = options
            self._recording_started_at = datetime.now().astimezone().isoformat()
            self._recording = True
            self._recording_paused = False
            self._write_recording_meta("")
            self._emit_status(f"Recording started: {session_dir}")
            return self._session_dir

    def stop_recording(self):
        with self._recording_lock:
            if not self._recording: return
            self._recording = False; self._recording_paused = False
            self._emg.stop_recording()
            self._recorder.stop()
            self._video_recorder.stop(self._config.recording_drain_timeout_sec)
            self._write_recording_meta(datetime.now().astimezone().isoformat())
            self._emit_status(f"Recording stopped: {self._session_dir}")

    def pause_recording(self):
        with self._recording_lock:
            if not self._recording or self._recording_paused: return False
            self._recording_paused = True
            self._emit_status("Recording paused"); return True

    def resume_recording(self):
        with self._recording_lock:
            if not self._recording or not self._recording_paused: return False
            self._recording_paused = False
            self._emit_status("Recording resumed"); return True

    def recording_stats(self):
        s = self._recorder.stats()
        video = self._video_recorder.stats()
        return {"recording": self._recording, "paused": self._recording_paused,
                "session_dir": self._session_dir, "csv_path": s.csv_path,
                "frames": s.frames, "rows": s.rows, "skipped": s.skipped_frames,
                "rgb_frames": video.rgb_frames, "depth_frames": video.depth_frames,
                "rgb_path": video.rgb_path, "depth_path": video.depth_path,
                "last_write_ms": video.last_write_ms,
                "received": video.received, "written": video.written,
                "dropped": video.dropped, "failed": video.failed,
                "stop_dropped": video.stop_dropped,
                "queue_depth": video.queue_depth,
                "queue_high_watermark": video.queue_high_watermark,
                "write_fps": video.write_fps,
                "write_avg_ms": video.write_avg_ms,
                "write_p95_ms": video.write_p95_ms,
                "metadata_path": video.metadata_path,
                "last_error": video.last_error}

    def _write_recording_meta(self, end_time: str) -> None:
        if not self._session_dir:
            return
        skeleton = self._recorder.stats()
        video = self._video_recorder.stats()
        options = self._recording_options
        data = {
            "start_time": self._recording_started_at,
            "end_time": end_time,
            "rgb_device": self._config.device.rgb_device_path,
            "depth_device": self._config.device.openni_device_uri or "OpenNI2 ANY_DEVICE",
            "rgb_format": self._config.device.rgb_pixel_format,
            "depth_format": self._config.device.depth_pixel_format,
            "rgb_width": self._config.device.rgb_width,
            "rgb_height": self._config.device.rgb_height,
            "depth_width": self._config.device.depth_width,
            "depth_height": self._config.device.depth_height,
            "rgb_fps": self._config.device.rgb_fps,
            "depth_fps": self._config.device.depth_fps,
            "hardware_d2c_enabled": self._config.device.enable_hardware_d2c,
            "rgb_mirror_at_capture": self._config.device.mirror_rgb_at_capture,
            "mirror_preview": options.mirror_preview,
            "record_skeleton": options.record_skeleton,
            "record_rgb": options.record_rgb,
            "record_depth": options.record_depth,
            "record_valid_3d_only": options.record_valid_3d_only,
            "skeleton_csv": "skeleton_3d.csv" if options.record_skeleton else None,
            "rgb_video": "rgb.mp4" if options.record_rgb else None,
            "depth_video": "depth.avi" if options.record_depth else None,
            "saved_skeleton_frames": skeleton.frames,
            "saved_rgb_frames": video.rgb_frames,
            "saved_depth_frames": video.depth_frames,
            "recording_declared_fps": self._config.device.rgb_fps,
            "recording_actual_fps": video.write_fps,
            "recording_received": video.received,
            "recording_written": video.written,
            "recording_dropped": video.dropped,
            "recording_failed": video.failed,
            "recording_stop_dropped": video.stop_dropped,
            "recording_queue_high_watermark": video.queue_high_watermark,
            "recording_final_state": "stopped" if end_time else "recording",
        }
        try:
            (Path(self._session_dir) / "meta.json").write_text(
                json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8"
            )
        except OSError as exc:
            logger.warn(f"Recording metadata write failed: {exc}")

    def performance_stats(self):
        with self._perf_lock:
            stats = {"raw_rgb_fps": self._rgb_fps, "raw_depth_fps": self._depth_fps,
                    "sync_fps": self._sync_fps,
                    "worker_fps": self._worker_fps,
                    "pair_fps": self._worker_fps, "pose_fps": self._pose_fps,
                    "rgb_fps": self._rgb_fps, "depth_fps": self._depth_fps,
                    "yolo_ms": self._last_yolo_ms,
                    "pose_ms": self._last_pose_ms,
                    "record_write_ms": self._last_record_write_ms,
                    "queue_length": self._pair_queue.qsize(),
                    "dropped_pairs": self._dropped, "processed": self._processed,
                    "stub_mode": False, "real_data": True,
                    "target_fps": 30.0,
                    "rgb_30fps_ok": 27.0 <= self._rgb_fps <= 33.5,
                    "depth_30fps_ok": 27.0 <= self._depth_fps <= 33.5,
                    "sync_30fps_ok": 27.0 <= self._sync_fps <= 33.5,
                    "pair_30fps_ok": 27.0 <= self._worker_fps <= 33.5,
                    "camera_status": self._camera_status,
                    "stopping": self._stopping.is_set()}
            pair_recording = self._pair_recorder.stats()
            video = self._video_recorder.stats()
            pose_values = list(self._pose_ms_samples)
            yolo_values = list(self._yolo_ms_samples)

            def p95(values):
                ordered = sorted(values)
                if not ordered:
                    return 0.0
                return ordered[min(
                    len(ordered) - 1,
                    int((len(ordered) - 1) * 0.95 + 0.999999),
                )]

            stats.update({
                "debug_pairs_recording": pair_recording.recording,
                "debug_pairs_saved": pair_recording.pairs,
                "debug_pairs_dir": pair_recording.session_dir,
                "pose_avg_ms": sum(pose_values) / len(pose_values) if pose_values else 0.0,
                "pose_p95_ms": p95(pose_values),
                "yolo_avg_ms": sum(yolo_values) / len(yolo_values) if yolo_values else 0.0,
                "yolo_p95_ms": p95(yolo_values),
                "record_write_ms": video.write_avg_ms,
                "record_write_avg_ms": video.write_avg_ms,
                "record_write_p95_ms": video.write_p95_ms,
                "recording_queue_depth": video.queue_depth,
                "recording_queue_high_watermark": video.queue_high_watermark,
                "recording_received": video.received,
                "recording_written": video.written,
                "recording_dropped": video.dropped,
                "recording_failed": video.failed,
                "recording_write_fps": video.write_fps,
            })
            if self._capture_backend is not None:
                sync = self._capture_backend.sync_stats()
                capture_perf = self._capture_backend.performance_stats()
                stats.update({
                    "sync_matched": sync.matched,
                    "sync_threshold_misses": sync.threshold_misses,
                    "sync_rgb_trimmed": sync.rgb_trimmed,
                    "sync_depth_trimmed": sync.depth_trimmed,
                    "raw_rgb_fps": capture_perf["raw_rgb_fps"],
                    "raw_depth_fps": capture_perf["raw_depth_fps"],
                    "rgb_fps": capture_perf["raw_rgb_fps"],
                    "depth_fps": capture_perf["raw_depth_fps"],
                    "sync_fps": capture_perf["sync_fps"],
                    "rgb_callback_p95_ms": capture_perf["rgb_callback_p95_ms"],
                    "depth_callback_p95_ms": capture_perf["depth_callback_p95_ms"],
                    "rgb_callback_state": capture_perf["rgb_callback_state"],
                    "depth_callback_state": capture_perf["depth_callback_state"],
                    "rgb_callback_avg_ms": capture_perf["rgb_callback_avg_ms"],
                    "depth_callback_avg_ms": capture_perf["depth_callback_avg_ms"],
                    "rgb_callback_max_ms": capture_perf["rgb_callback_max_ms"],
                    "depth_callback_max_ms": capture_perf["depth_callback_max_ms"],
                    "clock_quality_counts": capture_perf["clock_quality_counts"],
                    "clock_states": capture_perf["clock_states"],
                })
            return stats

    # ── ONNX pose model init ──────────────────────────────────────

    def _model_assets(self):
        python_root = Path(__file__).resolve().parent.parent
        repo_root = python_root.parent
        roots = [
            repo_root / "stroke-rehab" / "including",
            repo_root / "including",
            python_root / "including",
        ]

        def configured_or_default(configured, relative):
            if configured:
                path = Path(configured).expanduser()
                if not path.is_absolute():
                    for root in (python_root, repo_root):
                        candidate = (root / path).resolve()
                        if candidate.exists():
                            return str(candidate)
                return str(path.resolve())
            for root in roots:
                candidate = root / relative
                if candidate.exists():
                    return str(candidate)
            return str(roots[0] / relative)

        pose = self._config.pose
        return {
            "detector": configured_or_default(
                pose.detector_model_path, Path("yolov8n") / "yolov8n.onnx"
            ),
            "pose": configured_or_default(
                pose.model_path, Path("rtmpose-t") / "end2end.onnx"
            ),
            "pipeline": configured_or_default(
                pose.pipeline_json_path, Path("rtmpose-t") / "pipeline.json"
            ),
            "detail": configured_or_default(
                pose.detail_json_path, Path("rtmpose-t") / "detail.json"
            ),
            "deploy": configured_or_default(
                pose.deploy_json_path, Path("rtmpose-t") / "deploy.json"
            ),
        }

    def _init_pose_models(self):
        if not self._config.pose.enable_pose:
            self._pose_models_ready = False
            self._person_detector = self._pose_estimator = self._roi_tracker = None
            logger.info("Pose inference disabled by configuration")
            return False
        assets = self._model_assets()
        backend = str(getattr(self._config.pose, "inference_backend", "python")).lower()
        if backend not in ("python", "native", "auto"):
            logger.error(
                f"Unsupported inference_backend={backend!r}; expected python, native, or auto")
            return False
        self._python_inference = False
        try:
            if backend in ("python", "auto"):
                detector = PythonPersonDetector(self._config.pose)
                pose_estimator = PythonRtmposeEstimator(self._config.pose)
                detector_ok = detector.initialize(assets["detector"])
                pose_ok = pose_estimator.initialize(
                    assets["pose"], assets["pipeline"], assets["detail"]
                )
                if pose_ok:
                    self._person_detector = detector if detector_ok else None
                    self._pose_estimator = pose_estimator
                    self._roi_tracker = (
                        AdaptiveRoiTracker(detector, self._config.pose)
                        if detector_ok and self._config.pose.enable_adaptive_roi
                        else None
                    )
                    self._python_inference = True
                    logger.info(
                        f"Python ONNX pose ready: detector={detector_ok} pose={assets['pose']}"
                    )
                elif backend == "python":
                    logger.error(
                        "Python ONNX backend unavailable; install onnxruntime and verify model paths"
                    )

            native_pose_api = (
                _engine is not None
                and all(hasattr(_engine, name) for name in (
                    "PersonDetectorOrt", "PoseEstimatorRTMPoseOrt",
                    "PoseEstimatorConfig", "Halpe26ToRehab22Mapper",
                ))
            )
            if not self._python_inference and backend in ("native", "auto") and native_pose_api:
                self._person_detector = _engine.PersonDetectorOrt()
                if not self._person_detector.initialize(assets["detector"]):
                    self._person_detector = None
                self._pose_estimator = _engine.PoseEstimatorRTMPoseOrt()
                cfg = _engine.PoseEstimatorConfig()
                cfg.model_path = assets["pose"]
                cfg.pipeline_json_path = assets["pipeline"]
                cfg.detail_json_path = assets["detail"]
                cfg.deploy_json_path = assets["deploy"]
                if not self._pose_estimator.initialize(cfg):
                    self._pose_estimator = None
                self._halpe_mapper = _engine.Halpe26ToRehab22Mapper()
            elif not self._python_inference and backend in ("native", "auto"):
                logger.error(
                    "Native pose backend is unavailable in the hardware-only _core build; "
                    "use inference_backend=python or rebuild with "
                    "STROKE_BUILD_LEGACY_NATIVE_PIPELINE=ON for compatibility testing")

            # Use RGB intrinsics from calibration.yaml for accurate 3D projection
            calib = self._load_calibration()
            if calib:
                rgb = calib["rgb_intrinsics"]
                self._joint_projector.set_intrinsics(rgb["fx"], rgb["fy"], rgb["cx"], rgb["cy"])
                logger.info(f"Projector intrinsics from calibration: fx={rgb['fx']:.1f} fy={rgb['fy']:.1f}")
            else:
                self._joint_projector.set_intrinsics(0.0, 0.0, 0.0, 0.0)
                logger.warn("RGB intrinsics invalid; Python 3D projection disabled")

            # Build depth→RGB remap LUT for spatial alignment
            registration = load_registration_calibration(self._calibration_path())
            self._software_aligner = SoftwareRegistrationAligner(registration)
            self._align_ready = self._software_aligner.valid
            if self._align_ready:
                logger.info(f"Python software registration ready: {registration.source_path}")
            else:
                logger.warn("Software registration calibration unavailable; nearest resize fallback")

            self._pose_models_ready = self._pose_estimator is not None
            logger.info(
                f"Pose pipeline ready: {self._pose_models_ready} "
                f"backend={'python' if self._python_inference else 'native'}"
            )
            return self._pose_models_ready
        except Exception as e:
            logger.error(f"Pose model init failed: {e}")
            return False

    # ── Full capture (SyncedCapture + ONNX) ───────────────────────

    def _start_real_capture(self):
        if _engine is None:
            self._camera_status = "error"
            self._camera_error = "C++ engine not loaded"
            logger.error(self._camera_error)
            self._emit_status(f"ERROR: {self._camera_error}")
            return False

        self._init_pose_models()
        self._camera_status = "opening"

        try:
            self._capture_backend = NativeRgbDepthBackend(
                _engine, self._config.device, self._config.sync
            )
            self._capture_backend.set_on_status(self._on_camera_status)

            def _on_pair(pair):
                if not self._accept_frames.is_set():
                    return
                pair_data = {
                        "ts": pair.rgb.sync_ts_ns, "mock": False,
                        "bgr_image": pair.rgb.image,
                        "width": pair.rgb.width,
                        "height": pair.rgb.height,
                        "depth_image": pair.depth.image,
                        "depth_width": pair.depth.width,
                        "depth_height": pair.depth.height,
                        "delta_ns": pair.delta_ns,
                        "frame_id": pair.rgb.frame_id,
                        "depth_frame_id": pair.depth.frame_id,
                        "rgb_host_ts_ns": pair.rgb.host_ts_ns,
                        "depth_host_ts_ns": pair.depth.host_ts_ns,
                        "rgb_arrival_ts_ns": pair.rgb.arrival_ts_ns,
                        "depth_arrival_ts_ns": pair.depth.arrival_ts_ns,
                        "rgb_sync_ts_ns": pair.rgb.sync_ts_ns,
                        "depth_sync_ts_ns": pair.depth.sync_ts_ns,
                        "rgb_device_ts_us": pair.rgb.device_ts_us,
                        "depth_device_ts_us": pair.depth.device_ts_us,
                        "depth_unit_to_meter": pair.depth.depth_unit_to_meter,
                        "rgb_pixel_format_name": pair.rgb.pixel_format_name,
                        "depth_pixel_format_name": pair.depth.pixel_format_name,
                        "rgb_clock_quality": pair.rgb.clock_quality,
                        "depth_clock_quality": pair.depth.clock_quality,
                        "rgb_clock_reason": pair.rgb.clock_reason,
                        "depth_clock_reason": pair.depth.clock_reason,
                        "rgb_clock_reset_count": pair.rgb.clock_reset_count,
                        "depth_clock_reset_count": pair.depth.clock_reset_count,
                        "source": "python-sync",
                    }
                if (self._config.async_video_recording
                        and self._recording and not self._recording_paused):
                    self._video_recorder.submit(RecordingFrame(
                        bgr_image=pair.rgb.image,
                        depth_image=pair.depth.image,
                        sync_ts_ns=pair.rgb.sync_ts_ns,
                        rgb_frame_id=pair.rgb.frame_id,
                        depth_frame_id=pair.depth.frame_id,
                        depth_unit_to_meter=pair.depth.depth_unit_to_meter,
                        align_mode=(
                            "hardware" if self._hardware_d2c_active else "software"),
                    ))
                self._enqueue_pair(pair_data)

            if not self._capture_backend.start(_on_pair):
                raise RuntimeError("Native RGB/Depth backend failed to start")
            self._camera_status = "running"
            hw_d2c = self._capture_backend.hardware_d2c_active()
            self._hardware_d2c_active = hw_d2c
            align_mode = "HW_D2C" if hw_d2c else ("SW_REMAP" if self._align_ready else "NONE")
            self._emit_status(
                f"Pipeline: started | sync=PYTHON_NEAREST | align={align_mode} | HW_D2C={hw_d2c}")
            return True
        except Exception as e:
            self._camera_status = "error"; self._camera_error = str(e)
            logger.error(f"SyncedCapture start failed: {e}")
            self._emit_status(f"ERROR: SyncedCapture failed: {e}")
            capture = self._capture_backend
            self._capture_backend = None
            if capture is not None:
                try:
                    capture.stop()
                except Exception:
                    pass
            return False

    def _stop_real_capture(self):
        capture = self._capture_backend
        self._capture_backend = None
        if capture is None:
            self._hardware_d2c_active = False
            return
        done = threading.Event()
        failure = []

        def _stop_capture():
            try:
                capture.stop()
            except Exception as exc:
                failure.append(exc)
            finally:
                done.set()

        threading.Thread(
            target=_stop_capture, name="synced-stop-call", daemon=True).start()
        if not done.wait(timeout=5.0):
            raise TimeoutError("SyncedCapture.stop timed out after 5 seconds")
        if failure:
            raise RuntimeError(str(failure[0]))
        self._hardware_d2c_active = False

    def _on_camera_status(self, status: str):
        if status.startswith("[PERF]"):
            logger.performance(status)
            return
        logger.info(f"[Camera] {status}")
        self._emit_status(f"[Camera] {status}")

    # ── Worker loop ───────────────────────────────────────────────

    def _worker_loop(self):
        if self._config.pose.enable_cpu_affinity and hasattr(_os, "sched_setaffinity"):
            try:
                _os.sched_setaffinity(0, {int(self._config.pose.pose_cpu)})
                logger.info(f"bind thread pose_worker to CPU{self._config.pose.pose_cpu}")
            except (OSError, ValueError) as exc:
                logger.warn(f"bind thread pose_worker failed: {exc}")
        pose_interval = max(1, self._config.pose.pose_interval)
        while self._running.is_set():
            try: pair = self._pair_queue.get(timeout=0.1)
            except queue.Empty: continue

            self._processed += 1; self._worker_since_last += 1; self._pair_id += 1
            emg_status, emg_rms, emg_fatigue = self._tick_emg(
                int(pair.get("ts", 0))
            )
            depth_is_hardware = not bool(pair.get("mock"))

            if pair.get("mock"):
                self._emit_status("ERROR: mock RGB-D frame rejected; real data is required")
                continue
            else:
                depth_unit_to_meter = float(pair.get("depth_unit_to_meter", 0.001))
                bgr_image = pair.get("bgr_image")
                depth_raw = pair.get("depth_image")
                rgb_image = None
                if bgr_image is not None:
                    try:
                        import cv2
                        rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
                    except Exception:
                        rgb_image = None
                depth_image = self._align_depth(depth_raw, depth_unit_to_meter)
                pose_2d, pose_3d, bbox, pose_ms, yolo_ms = \
                    self._infer_pose_full(
                        bgr_image, depth_image, pose_interval, depth_unit_to_meter)

            # Only real hardware depth may reach depth-dependent consumers.
            if not depth_is_hardware:
                depth_raw = None
                depth_image = None
                pose_3d = []
                self._last_raw_j3d = []
                self._last_ema_j3d = []

            if self._pair_recorder.stats().recording and all(
                image is not None for image in (bgr_image, depth_raw, depth_image)
            ):
                self._pair_recorder.record(
                    rgb_image=bgr_image,
                    depth_raw=depth_raw,
                    depth_aligned=depth_image,
                    rgb_frame_id=int(pair.get("frame_id", self._pair_id)),
                    depth_frame_id=int(pair.get("depth_frame_id", self._pair_id)),
                    rgb_host_ts_ns=int(pair.get("rgb_host_ts_ns", pair.get("ts", 0))),
                    depth_host_ts_ns=int(pair.get("depth_host_ts_ns", pair.get("ts", 0))),
                    rgb_device_ts_us=int(pair.get("rgb_device_ts_us", 0)),
                    depth_device_ts_us=int(pair.get("depth_device_ts_us", 0)),
                    rgb_arrival_ts_ns=int(pair.get("rgb_arrival_ts_ns", 0)),
                    depth_arrival_ts_ns=int(pair.get("depth_arrival_ts_ns", 0)),
                    rgb_sync_ts_ns=int(pair.get("rgb_sync_ts_ns", pair.get("ts", 0))),
                    depth_sync_ts_ns=int(pair.get("depth_sync_ts_ns", pair.get("ts", 0))),
                    rgb_clock_quality=str(pair.get("rgb_clock_quality", "host_fallback")),
                    depth_clock_quality=str(pair.get("depth_clock_quality", "host_fallback")),
                    rgb_clock_reason=str(pair.get("rgb_clock_reason", "")),
                    depth_clock_reason=str(pair.get("depth_clock_reason", "")),
                    rgb_clock_reset_count=int(pair.get("rgb_clock_reset_count", 0)),
                    depth_clock_reset_count=int(pair.get("depth_clock_reset_count", 0)),
                    delta_ns=int(pair.get("delta_ns", 0)),
                    align_mode="hardware" if self._hardware_d2c_active else "software",
                )

            if self._recording and not self._recording_paused:
                if not self._config.async_video_recording:
                    self._video_recorder.submit(RecordingFrame(
                        bgr_image=bgr_image,
                        depth_image=depth_image,
                        sync_ts_ns=int(pair.get("ts", 0)),
                        rgb_frame_id=int(pair.get("frame_id", self._pair_id)),
                        depth_frame_id=int(pair.get("depth_frame_id", self._pair_id)),
                        depth_unit_to_meter=depth_unit_to_meter,
                        align_mode=(
                            "hardware" if self._hardware_d2c_active else "software"),
                    ))
                if (depth_is_hardware
                        and self._recording_options.record_skeleton):
                    self._record_skeleton(pose_3d, int(pair.get("ts", 0)))
            video_stats = self._video_recorder.stats()
            record_write_ms = video_stats.write_avg_ms
            with self._perf_lock:
                self._last_yolo_ms = yolo_ms
                if pose_ms > 0.0:
                    self._last_pose_ms = pose_ms
                    self._pose_ms_samples.append(pose_ms)
                if yolo_ms > 0.0:
                    self._yolo_ms_samples.append(yolo_ms)
                self._last_record_write_ms = record_write_ms
            raw_j3d = [
                (point.x, point.y, point.z, point.score, point.valid)
                for point in self._last_raw_j3d
            ]
            ema_j3d = [
                (point.x, point.y, point.z, point.score, point.valid)
                for point in self._last_ema_j3d
            ]

            perf_snapshot = self.performance_stats()
            self._preview.submit(
                pair_id=self._pair_id,
                rgb_frame_id=int(pair.get("frame_id", self._pair_id)),
                depth_frame_id=int(pair.get("depth_frame_id", self._pair_id)),
                host_ts_ns=int(pair.get("rgb_host_ts_ns", pair.get("ts", 0))),
                rgb_width=int(pair.get("width", self._config.device.rgb_width)),
                rgb_height=int(pair.get("height", self._config.device.rgb_height)),
                depth_width=int(pair.get("depth_width", self._config.device.depth_width)),
                depth_height=int(pair.get("depth_height", self._config.device.depth_height)),
                pose_interval=pose_interval,
                joints_2d_raw=pose_2d, joints_3d=pose_3d,
                raw_joints_3d=raw_j3d, ema_joints_3d=ema_j3d,
                raw_rgb_fps=perf_snapshot.get("raw_rgb_fps", 0.0),
                raw_depth_fps=perf_snapshot.get("raw_depth_fps", 0.0),
                sync_fps=perf_snapshot.get("sync_fps", 0.0),
                worker_fps=self._worker_fps, pair_fps=self._worker_fps,
                pose_fps=self._pose_fps,
                yolo_ms=yolo_ms, pose_ms=pose_ms,
                record_write_ms=record_write_ms,
                queue_length=self._pair_queue.qsize(),
                dropped_pairs=self._dropped,
                delta_ms=float(pair.get("delta_ns", 0)) / 1_000_000.0,
                bbox=bbox,
                recording=self._recording and not self._recording_paused,
                skeleton_recording=self._recording and not self._recording_paused,
                skeleton_frames=self._recorder.stats().frames if self._recording else 0,
                rgb_frames=video_stats.rgb_frames,
                depth_frames=video_stats.depth_frames,
                emg_status=emg_status, emg_rms=emg_rms,
                emg_fatigue=emg_fatigue,
                rgb_image=rgb_image, depth_image=depth_image,
                depth_is_hardware=depth_is_hardware)

            self._update_performance()

            if self._on_frame:
                frame = self._preview.latest_frame()
                if frame:
                    try: self._on_frame(frame)
                    except Exception: pass

    # ── Calibration & spatial alignment ────────────────────────────

    def _calibration_path(self) -> str:
        configured = str(getattr(self._config, "calibration_file", "")).strip()
        if configured:
            path = Path(configured).expanduser()
            if not path.is_absolute():
                path = Path(__file__).resolve().parent.parent / path
            return str(path.resolve())
        python_root = Path(__file__).resolve().parent.parent
        candidates = [
            python_root / "configs" / "calibration.yaml",
            python_root.parent / "stroke-rehab" / "configs" / "calibration.yaml",
        ]
        return str(next((path for path in candidates if path.is_file()), candidates[0]))

    def _load_calibration(self) -> Optional[dict]:
        calibration = load_registration_calibration(self._calibration_path())
        if calibration is None:
            return None
        return {
            "rgb_intrinsics": {
                "fx": calibration.rgb.fx,
                "fy": calibration.rgb.fy,
                "cx": calibration.rgb.cx,
                "cy": calibration.rgb.cy,
            },
            "depth_intrinsics": {
                "fx": calibration.depth.fx,
                "fy": calibration.depth.fy,
                "cx": calibration.depth.cx,
                "cy": calibration.depth.cy,
            },
            "R": calibration.rotation.reshape(-1).tolist(),
            "T": calibration.translation_m.tolist(),
            "width": calibration.rgb.width or self._config.device.rgb_width,
            "height": calibration.rgb.height or self._config.device.rgb_height,
        }

    def _align_depth(self, depth_image, depth_unit_to_meter: float = 0.001):
        """Align depth to RGB using hardware D2C or calibrated point projection."""
        if depth_image is None or np is None:
            return depth_image
        if self._capture_backend is not None and not self._hardware_d2c_active:
            self._hardware_d2c_active = self._capture_backend.hardware_d2c_active()
        if self._hardware_d2c_active:
            return depth_image
        if self._software_aligner is None:
            self._software_aligner = SoftwareRegistrationAligner(
                load_registration_calibration(self._calibration_path())
            )
        return self._software_aligner.align(
            depth_image,
            (self._config.device.rgb_width, self._config.device.rgb_height),
            depth_unit_to_meter,
        )

    # ── Full ONNX pose inference ──────────────────────────────────

    def _infer_pose_full(
        self, bgr, depth_image, pose_interval, depth_unit_to_meter: float = 0.001
    ):
        # If pose is disabled or models aren't ready, return last cached skeleton
        if bgr is None or not self._pose_models_ready:
            return self._last_j2d, self._last_j3d, self._last_bbox, 0.0, 0.0

        self._frame_counter += 1
        should_run_pose = (
            not self._last_j2d
            or not self._config.pose.enable_pose_reuse
            or self._frame_counter % max(1, pose_interval) == 0
        )
        if not should_run_pose:
            # Match the original pipeline: reuse 2D, but sample the current
            # aligned depth so the 3D skeleton is not frozen between inferences.
            if self._last_j2d and depth_image is not None:
                self._last_j3d = self._lift_pose_to_3d(
                    self._last_j2d, depth_image, self._last_bbox,
                    time.monotonic_ns(), depth_unit_to_meter)
            return self._last_j2d, self._last_j3d, self._last_bbox, 0.0, 0.0

        # ── Run full ONNX inference ──
        j2d, j3d = [], []
        bbox = None; pose_ms = 0.0; yolo_ms = 0.0

        try:
            if self._python_inference:
                t0 = time.monotonic()
                box = (
                    self._roi_tracker.get_primary_box(bgr)
                    if self._roi_tracker is not None
                    else PythonBoundingBox(
                        0.0, 0.0, float(bgr.shape[1]), float(bgr.shape[0]), 1.0, True
                    )
                )
                yolo_ms = (time.monotonic() - t0) * 1000.0
                if not box.valid:
                    box = PythonBoundingBox(
                        0.0, 0.0, float(bgr.shape[1]), float(bgr.shape[0]), 1.0, True
                    )
                result = self._pose_estimator.infer(bgr, box)
                pose_ms = result.pose_ms
                if self._roi_tracker is not None:
                    self._roi_tracker.update_from_pose(result.used_box, result.keypoints)
                if not result.model_loaded:
                    return self._last_j2d, self._last_j3d, self._last_bbox, pose_ms, yolo_ms
                rehab_points = map_halpe26_to_rehab22(result.keypoints)
                used = result.used_box
                if used.valid:
                    bbox = (
                        used.x, used.y, used.w, used.h, True,
                        self._roi_tracker.debug_state if self._roi_tracker else "full_fallback",
                    )
            else:
                t0 = time.monotonic()
                box = (
                    self._person_detector.detect_largest_person_bgr(bgr)
                    if self._person_detector
                    else None
                )
                yolo_ms = (time.monotonic() - t0) * 1000.0
                t1 = time.monotonic()
                if box:
                    bb = _engine.BoundingBox2D()
                    bb.x = box.x; bb.y = box.y; bb.w = box.w; bb.h = box.h
                    bb.valid = box.valid; bb.score = box.score
                    self._pose_estimator.set_bounding_box_provider_fallback(bb)
                result = self._pose_estimator.infer_bgr(bgr)
                pose_ms = (time.monotonic() - t1) * 1000.0
                if result is None or not result.model_loaded:
                    return self._last_j2d, self._last_j3d, self._last_bbox, pose_ms, yolo_ms
                rehab_points = self._halpe_mapper.map(result.keypoints)
                used = result.used_box
                if used.valid:
                    bbox = (used.x, used.y, used.w, used.h, True, "detect")

            rehab2d = make_rehab22_joints(rehab_points, self._config.pose.min_score)
            j2d = [(p.x, p.y, p.score, p.valid) for p in rehab2d]
            j3d = self._lift_pose_to_3d(
                rehab2d, depth_image, bbox, time.monotonic_ns(),
                depth_unit_to_meter)

        except Exception as e:
            logger.warn(f"Pose inference error: {e}")
            return self._last_j2d, self._last_j3d, self._last_bbox, 0.0, 0.0

        # ── Cache for skipped frames ──
        self._pose_count += 1; self._pose_since_last += 1
        if any(point[3] for point in j2d):
            self._last_j2d, self._last_j3d, self._last_bbox = j2d, j3d, bbox
        else:
            self._last_j2d = self._last_j3d = []
            self._last_bbox = None
            self._ema_filter.reset()
            self._smoother.reset()
        return j2d, j3d, bbox, pose_ms, yolo_ms

    def _lift_pose_to_3d(
        self, points, depth_image, bbox, timestamp_ns,
        depth_unit_to_meter: float = 0.001,
    ):
        rehab2d = make_rehab22_joints(points, self._config.pose.min_score)
        if depth_image is None or not self._joint_projector.intrinsics_valid():
            self._last_rehab2d = rehab2d
            self._last_depth_debug = []
            self._last_raw_j3d = []
            self._last_ema_j3d = []
            self._last_ema_debug = []
            return [(0.0, 0.0, 0.0, point.score, False) for point in rehab2d]
        roi = None
        if bbox and len(bbox) >= 5 and bbox[4]:
            roi = tuple(int(round(value)) for value in bbox[:4])
        samples = self._depth_sampler.sample_skeleton(
            depth_image, rehab2d, depth_unit_to_meter, roi)
        raw = self._joint_projector.project(rehab2d, samples)
        self._last_rehab2d = rehab2d
        self._last_depth_debug = samples
        self._last_raw_j3d = raw
        now = timestamp_ns / 1.0e9
        dt = max(0.001, now - self._last_pose_time) if self._last_pose_time > 0 else (
            1.0 / max(1, self._config.device.rgb_fps))
        mode = str(self._config.skeleton_filter.mode).lower()
        if not self._config.pose.enable_smoothing or mode == "none":
            filtered = raw
            ema_debug = [
                {"alpha": 1.0, "reason": "smoothing_disabled", "invalid_hold_count": 0}
                for _ in range(22)
            ]
        elif mode == "legacy_stabilizer":
            filtered, _ = self._smoother.smooth(raw, timestamp_ns)
            ema_debug = [
                {"alpha": self._config.pose.smoothing_alpha,
                 "reason": "legacy_stabilizer", "invalid_hold_count": 0}
                for _ in range(22)
            ]
        else:
            filtered = self._ema_filter.filter(raw, dt)
            ema_debug = self._ema_filter.last_debug
        self._last_ema_j3d = filtered
        self._last_ema_debug = ema_debug
        self._last_pose_time = now
        return [
            (point.x, point.y, point.z, point.score, point.valid)
            for point in filtered
        ]

    # ── Helpers ───────────────────────────────────────────────────

    def _tick_emg(self, host_ts_ns: int = 0):
        status = self._emg.runtime_status()
        frame = (
            self._emg.nearest_feature(host_ts_ns, 300_000_000)
            if host_ts_ns > 0
            else self._emg.latest_feature()
        )
        if frame is None:
            return self._format_emg_status(status), [], []
        return (
            self._format_emg_status(status),
            [channel.rms for channel in frame.channels],
            [channel.fatigue_index for channel in frame.channels],
        )

    @staticmethod
    def _format_emg_status(status) -> str:
        return (
            f"mode={status.mode} "
            f"rpmsg={'connected' if status.rpmsg_connected else 'off'} "
            f"ble={'connected' if status.ble_connected else 'off'} "
            f"{status.message}"
        ).strip()

    def _record_skeleton(self, pose_3d, timestamp_ns: int = 0):
        ts = timestamp_ns or time.monotonic_ns(); joints = []
        for j in pose_3d:
            score = j[3] if len(j) > 3 else 1.0
            valid = 1 if (j[4] if len(j) > 4 else False) else 0
            joints.append([j[0], j[1], j[2], score, valid])
        if self._recording_options.record_valid_3d_only and not any(
            bool(joint[4]) for joint in joints
        ):
            return
        self._recorder.record(
            timestamp_ns=ts, frame_id=self._pair_id, pair_id=self._pair_id,
            dt_seconds=0.033, bbox_mode="full",
            joints_3d=joints,
            joints_2d=self._last_rehab2d,
            raw_joints_3d=self._last_raw_j3d,
            ema_joints_3d=self._last_ema_j3d,
            depth_debug=self._last_depth_debug,
            ema_debug=self._last_ema_debug,
        )

    # ── Performance ───────────────────────────────────────────────

    def _update_performance(self):
        now = time.monotonic(); elapsed = now - self._last_perf_time
        if elapsed >= 1.0:
            with self._perf_lock:
                self._rgb_fps = self._rgb_since_last / elapsed
                self._depth_fps = self._depth_since_last / elapsed
                self._sync_fps = self._sync_since_last / elapsed
                self._worker_fps = self._worker_since_last / elapsed
                self._pose_fps = self._pose_since_last / elapsed
                self._rgb_since_last = 0; self._depth_since_last = 0
                self._sync_since_last = 0; self._worker_since_last = 0
                self._pose_since_last = 0
                self._last_perf_time = now

    def _emit_status(self, message: str):
        logger.info(message)
        if self._on_status: self._on_status(message)
