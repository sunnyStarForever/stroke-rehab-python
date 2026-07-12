"""
Sensor pipeline orchestrator.
Replaces core/pipeline/SensorPipeline.cpp (~1600 lines).

Full mode data flow:
  SyncedCapture(C++: RGB+Depth+SyncManager) → pair_queue → Worker
    → YOLO detect → RTMPose infer → Halpe→Rehab22 map
    → DepthSample → JointProject3D → EMA filter → Smoother
    → Preview + Recording + Scoring

Stub mode: mock capture thread + synthetic pose.
"""

import math
import os as _os
import queue
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

try:
    import numpy as np
except ImportError:
    np = None

from . import PipelineConfig, logger

_STUB_MODE = False
_engine = None
try:
    from . import _core as _engine
except ImportError:
    _STUB_MODE = True

from .preview import PreviewComposer, PreviewFrame
from .recorder import Skeleton3DRecorder


def _decode_jpeg_to_rgb(jpeg: bytes) -> Any:
    if np is None:
        return None
    try:
        import cv2
        arr = np.frombuffer(jpeg, dtype=np.uint8)
        bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB) if bgr is not None else None
    except Exception:
        return None


class SensorPipeline:
    """Main pipeline orchestrator: dual-mode (stub / full ONNX)."""

    def __init__(self, config: Optional[PipelineConfig] = None):
        self._config = config or PipelineConfig()
        self._stub_mode = _STUB_MODE
        self._pair_queue: queue.Queue = queue.Queue(maxsize=100)
        self._worker_thread: Optional[threading.Thread] = None
        self._running = threading.Event()
        self._accept_frames = threading.Event()
        self._stopping = threading.Event()
        self._start_done = threading.Event()
        self._start_done.set()
        self._stop_lock = threading.Lock()
        self._stop_callbacks: List[Callable[[bool, str], None]] = []

        # C++ capture
        self._synced_capture = None

        # C++ pose pipeline (ONNX)
        self._person_detector = None
        self._pose_estimator = None
        self._halpe_mapper = None
        self._depth_sampler = None
        self._joint_projector = None
        self._ema_filter = None
        self._smoother = None
        self._pose_models_ready = False
        self._frame_counter = 0
        self._last_pose_time = 0.0

        self._preview = PreviewComposer()
        self._recorder = Skeleton3DRecorder()

        self._recording = False
        self._recording_paused = False
        self._session_dir = ""
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
        self._pair_fps = 0.0
        self._pose_fps = 0.0
        self._rgb_since_last = 0
        self._depth_since_last = 0
        self._pair_since_last = 0
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

        # Depth diagnostic counters
        self._depth_has_data = False
        self._depth_empty_count = 0

        # Spatial alignment — precomputed remap LUT (depth → RGB coordinate frame)
        self._align_map_x = None
        self._align_map_y = None
        self._align_ready = False

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
    def stub_mode(self): return self._stub_mode
    @property
    def camera_status(self):
        return {"status": self._camera_status, "error": self._camera_error,
                "rgb_fps": self._rgb_fps, "mode": "STUB" if self._stub_mode else "FULL"}

    # ── Start / Stop ──────────────────────────────────────────────

    def start(self) -> bool:
        if self._running.is_set() or self._stopping.is_set():
            return False
        if self._camera_status == "stop_error":
            self._emit_status(
                "ERROR: previous camera shutdown was incomplete; restart the application")
            return False
        if (not self._stub_mode and
                (self._config.device.rgb_fps != 30 or self._config.device.depth_fps != 30)):
            self._camera_status = "configuration_error"
            self._camera_error = "真实 RGB 与 Depth 采集必须同时配置为 30 FPS"
            self._emit_status(f"ERROR: {self._camera_error}")
            return False
        mode_label = "STUB" if self._stub_mode else "FULL"
        self._emit_status(f"Pipeline starting ({mode_label} mode)...")
        self._start_done.clear()
        self._drain_pair_queue()
        self._reset_performance_counters()
        self._accept_frames.set()
        self._running.set()
        self._worker_thread = threading.Thread(
            target=self._worker_loop, name="pipeline-worker", daemon=True)
        self._worker_thread.start()
        if self._stub_mode:
            self._start_mock_capture()
        else:
            if not self._start_real_capture():
                self._accept_frames.clear()
                self._running.clear()
                if self._worker_thread and self._worker_thread.is_alive():
                    self._worker_thread.join(timeout=1.0)
                self._start_done.set()
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

        self._worker_thread = None
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

    def _reset_performance_counters(self) -> None:
        with self._perf_lock:
            self._pair_id = self._processed = self._dropped = 0
            self._rgb_count = self._depth_count = self._pose_count = 0
            self._rgb_fps = self._depth_fps = self._pair_fps = self._pose_fps = 0.0
            self._rgb_since_last = self._depth_since_last = 0
            self._pair_since_last = self._pose_since_last = 0
            self._last_perf_time = time.monotonic()

    # ── Recording ─────────────────────────────────────────────────

    def start_recording(self, save_root: str) -> str:
        with self._recording_lock:
            if self._recording: return self._session_dir
            now = datetime.now()
            date_folder = now.strftime("%Y%m%d")
            session_name = f"{now:%Y%m%d_%H%M%S}_{now.microsecond // 1000:03d}"
            session_dir = Path(save_root) / date_folder / session_name
            session_dir.mkdir(parents=True, exist_ok=True)
            if not self._recorder.start(str(session_dir)):
                raise OSError(f"Cannot open recording files in {session_dir}")
            self._session_dir = str(session_dir)
            self._recording = True
            self._recording_paused = False
            self._emit_status(f"Recording started: {session_dir}")
            return self._session_dir

    def stop_recording(self):
        with self._recording_lock:
            if not self._recording: return
            self._recording = False; self._recording_paused = False
            self._recorder.stop()
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
        return {"recording": s.recording, "paused": self._recording_paused,
                "session_dir": s.session_dir, "csv_path": s.csv_path,
                "frames": s.frames, "rows": s.rows, "skipped": s.skipped_frames}

    def performance_stats(self):
        with self._perf_lock:
            return {"rgb_fps": self._rgb_fps, "depth_fps": self._depth_fps,
                    "pair_fps": self._pair_fps, "pose_fps": self._pose_fps,
                    "queue_length": self._pair_queue.qsize(),
                    "dropped_pairs": self._dropped, "processed": self._processed,
                    "stub_mode": self._stub_mode, "real_data": not self._stub_mode,
                    "target_fps": 30.0,
                    "rgb_30fps_ok": 27.0 <= self._rgb_fps <= 33.5,
                    "depth_30fps_ok": 27.0 <= self._depth_fps <= 33.5,
                    "pair_30fps_ok": 27.0 <= self._pair_fps <= 33.5,
                    "camera_status": self._camera_status,
                    "stopping": self._stopping.is_set()}

    # ── Stub capture ──────────────────────────────────────────────

    def _start_mock_capture(self):
        self._emit_status("Pipeline: mock capture (no real camera)")
        threading.Thread(target=self._mock_capture_loop, name="mock-capture", daemon=True).start()

    def _mock_capture_loop(self):
        interval = 1.0 / max(1, self._config.device.rgb_fps)
        while self._running.is_set():
            try: self._pair_queue.put_nowait({"ts": time.monotonic_ns(), "mock": True})
            except queue.Full: self._dropped += 1
            self._rgb_count += 1; self._depth_count += 1
            self._rgb_since_last += 1; self._depth_since_last += 1
            time.sleep(interval)

    # ── ONNX pose model init ──────────────────────────────────────

    def _init_pose_models(self):
        if _engine is None: return False
        try:
            model_dir = _os.path.normpath(_os.path.join(
                _os.path.dirname(_os.path.dirname(__file__)), "..", "including"))
            yolo_path = _os.path.join(model_dir, "yolov8n", "yolov8n.onnx")
            pose_path = _os.path.join(model_dir, "rtmpose-t", "end2end.onnx")

            self._person_detector = _engine.PersonDetectorOrt()
            if _os.path.exists(yolo_path) and self._person_detector.initialize(yolo_path):
                logger.info(f"YOLO initialized: {yolo_path}")
            else:
                logger.warn(f"YOLO not available: {yolo_path}")
                self._person_detector = None

            self._pose_estimator = _engine.PoseEstimatorRTMPoseOrt()
            if _os.path.exists(pose_path):
                cfg = _engine.PoseEstimatorConfig()
                cfg.model_path = pose_path
                cfg.pipeline_json_path = _os.path.join(model_dir, "rtmpose-t", "pipeline.json")
                cfg.detail_json_path = _os.path.join(model_dir, "rtmpose-t", "detail.json")
                cfg.deploy_json_path = _os.path.join(model_dir, "rtmpose-t", "deploy.json")
                if self._pose_estimator.initialize(cfg):
                    logger.info(f"RTMPose initialized: {pose_path}")
                else:
                    self._pose_estimator = None
            else:
                logger.warn(f"RTMPose model not found: {pose_path}")
                self._pose_estimator = None

            self._halpe_mapper = _engine.Halpe26ToRehab22Mapper()
            self._depth_sampler = _engine.DepthSampler()
            self._joint_projector = _engine.JointProjector3D()
            self._ema_filter = _engine.EMASkeletonFilter()
            self._smoother = _engine.SkeletonSmoother()
            # Use RGB intrinsics from calibration.yaml for accurate 3D projection
            calib = self._load_calibration()
            if calib:
                rgb = calib["rgb_intrinsics"]
                self._joint_projector.set_intrinsics(rgb["fx"], rgb["fy"], rgb["cx"], rgb["cy"])
                logger.info(f"Projector intrinsics from calibration: fx={rgb['fx']:.1f} fy={rgb['fy']:.1f}")
            else:
                self._joint_projector.set_intrinsics(570.34, 570.34, 319.5, 239.5)
                logger.warn("Using fallback intrinsics (no calibration.yaml)")

            # Build depth→RGB remap LUT for spatial alignment
            self._build_alignment_remap(calib)

            self._pose_models_ready = self._pose_estimator is not None
            logger.info(f"Pose pipeline ready: {self._pose_models_ready}")
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
            cfg = _engine.PipelineConfig()
            dev = cfg.device
            dev.rgb_device_path = self._config.device.rgb_device_path or \
                f"/dev/video{self._config.device.rgb_device_index}"
            dev.rgb_width = self._config.device.rgb_width
            dev.rgb_height = self._config.device.rgb_height
            dev.rgb_fps = self._config.device.rgb_fps
            dev.rgb_pixel_format = self._config.device.rgb_pixel_format
            dev.mirror_rgb_at_capture = self._config.device.mirror_rgb_at_capture
            dev.rgb_device_index = self._config.device.rgb_device_index
            dev.depth_width = self._config.device.depth_width
            dev.depth_height = self._config.device.depth_height
            dev.depth_fps = self._config.device.depth_fps
            dev.depth_pixel_format = self._config.device.depth_pixel_format
            dev.enable_hardware_d2c = self._config.device.enable_hardware_d2c

            self._synced_capture = _engine.SyncedCapture()
            self._synced_capture.set_on_status(lambda s: self._on_camera_status(s))

            def _on_pair(jpeg, png, rw, rh, dw, dh, rts, dts, delta):
                if not self._accept_frames.is_set():
                    return
                self._rgb_count += 1; self._rgb_since_last += 1
                self._depth_count += 1; self._depth_since_last += 1
                try:
                    self._pair_queue.put_nowait({
                        "ts": rts, "mock": False,
                        "jpeg": jpeg, "width": rw, "height": rh,
                        "depth_png": png, "depth_width": dw, "depth_height": dh,
                        "delta_ns": delta, "frame_id": 0, "source": "synced",
                    })
                except queue.Full: self._dropped += 1

            if not self._synced_capture.start(dev, _on_pair):
                raise RuntimeError("SyncedCapture.start() returned False")
            self._camera_status = "running"
            hw_d2c = self._synced_capture.hardware_d2c_active()
            align_mode = "HW_D2C" if hw_d2c else ("SW_REMAP" if self._align_ready else "NONE")
            self._emit_status(
                f"Pipeline: started | align={align_mode} | HW_D2C={hw_d2c}")
            return True
        except Exception as e:
            self._camera_status = "error"; self._camera_error = str(e)
            logger.error(f"SyncedCapture start failed: {e}")
            self._emit_status(f"ERROR: SyncedCapture failed: {e}")
            capture = self._synced_capture
            self._synced_capture = None
            if capture is not None:
                try:
                    capture.stop()
                except Exception:
                    pass
            return False

    def _stop_real_capture(self):
        capture = self._synced_capture
        self._synced_capture = None
        if capture is None:
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

    def _on_camera_status(self, status: str):
        logger.info(f"[Camera] {status}")
        self._emit_status(f"[Camera] {status}")

    # ── Worker loop ───────────────────────────────────────────────

    def _worker_loop(self):
        pose_interval = max(1, self._config.pose.pose_interval)
        while self._running.is_set():
            try: pair = self._pair_queue.get(timeout=0.1)
            except queue.Empty: continue

            self._processed += 1; self._pair_since_last += 1; self._pair_id += 1
            emg_status, emg_rms = self._tick_emg()

            if pair.get("mock"):
                pose_2d, pose_3d, bbox = self._synthetic_pose_full()
                rgb_image = None; depth_image = None
                pose_ms = 0.0; yolo_ms = 0.0
            else:
                jpeg = pair.get("jpeg"); depth_png = pair.get("depth_png")
                rgb_image = _decode_jpeg_to_rgb(jpeg) if jpeg else None
                depth_raw = self._decode_depth_png(depth_png)
                depth_image = self._align_depth(depth_raw)
                # Re-encode aligned depth for C++ DepthSampler (uses PNG bytes)
                aligned_png = self._encode_depth_png(depth_image)
                pose_2d, pose_3d, bbox, pose_ms, yolo_ms = \
                    self._infer_pose_full(jpeg, aligned_png or depth_png, depth_image, pose_interval)

            self._preview.submit(
                joints_2d_raw=pose_2d, joints_3d=pose_3d,
                rgb_fps=self._rgb_fps, depth_fps=self._depth_fps,
                pair_fps=self._pair_fps, pose_fps=self._pose_fps,
                yolo_ms=yolo_ms, pose_ms=pose_ms,
                queue_length=self._pair_queue.qsize(),
                dropped_pairs=self._dropped, delta_ms=0.0, bbox=bbox,
                recording=self._recording and not self._recording_paused,
                skeleton_recording=self._recording and not self._recording_paused,
                skeleton_frames=self._recorder.stats().frames if self._recording else 0,
                emg_status=emg_status, emg_rms=emg_rms,
                rgb_image=rgb_image, depth_image=depth_image)

            if self._recording and not self._recording_paused:
                self._record_skeleton(pose_3d)

            self._update_performance()

            if self._on_frame:
                frame = self._preview.latest_frame()
                if frame:
                    try: self._on_frame(frame)
                    except Exception: pass

    # ── Calibration & spatial alignment ────────────────────────────

    def _load_calibration(self) -> Optional[dict]:
        """Return Astra Pro calibration constants from configs/calibration.yaml."""
        calib_path = _os.path.normpath(_os.path.join(
            _os.path.dirname(_os.path.dirname(__file__)), "..", "configs", "calibration.yaml"))
        try:
            if _os.path.exists(calib_path):
                import yaml as _yaml  # requires PyYAML (already installed)
                with open(calib_path, "r") as f:
                    data = _yaml.safe_load(f)
                R = data.get("depth_to_rgb_extrinsics", {}).get("R", [])
                T = data.get("depth_to_rgb_extrinsics", {}).get("T", [])
                rgb = data.get("rgb_intrinsics", {})
                depth = data.get("depth_intrinsics", {})
                if R and T:
                    return {
                        "rgb_intrinsics": {"fx": float(rgb.get("fx", 592)), "fy": float(rgb.get("fy", 592)),
                                           "cx": float(rgb.get("cx", 320)), "cy": float(rgb.get("cy", 240))},
                        "depth_intrinsics": {"fx": float(depth.get("fx", 576)), "fy": float(depth.get("fy", 575)),
                                             "cx": float(depth.get("cx", 325)), "cy": float(depth.get("cy", 261))},
                        "R": [float(x) for x in R], "T": [float(x) for x in T],
                        "width": 640, "height": 480,
                    }
        except Exception as e:
            logger.warn(f"YAML calibration load failed ({e}), using hardcoded defaults")

        # Hardcoded fallback (same values as calibration.yaml)
        return {
            "rgb_intrinsics": {"fx": 592.966, "fy": 591.582, "cx": 319.775, "cy": 252.137},
            "depth_intrinsics": {"fx": 575.688, "fy": 574.712, "cx": 325.402, "cy": 260.708},
            "R": [0.999665, -0.025406, 0.004879, 0.025396, 0.999675, 0.002035, -0.004930, -0.001910, 0.999986],
            "T": [26.476, 3.024, -2.781],
            "width": 640, "height": 480,
        }

    def _build_alignment_remap(self, calib: Optional[dict]):
        """Precompute cv2.remap LUT for depth → RGB spatial alignment.

        Uses inverse projection: for each RGB pixel, find the corresponding
        depth-pixel source.  Approximates Z = 1.5 m (negligible parallax
        error for rehab-range distances with Astro Pro's 25 mm baseline).
        """
        self._align_ready = False
        if calib is None or np is None:
            return
        try:
            import cv2
            w, h = calib["width"], calib["height"]
            Z_REF = 1.5  # metres

            R = np.array(calib["R"]).reshape(3, 3).astype(np.float64)
            T = np.array(calib["T"]).reshape(3, 1).astype(np.float64)
            R_inv = R.T
            Kd = np.array([[calib["depth_intrinsics"]["fx"], 0, calib["depth_intrinsics"]["cx"]],
                           [0, calib["depth_intrinsics"]["fy"], calib["depth_intrinsics"]["cy"]],
                           [0, 0, 1]], dtype=np.float64)
            Kr = np.array([[calib["rgb_intrinsics"]["fx"], 0, calib["rgb_intrinsics"]["cx"]],
                           [0, calib["rgb_intrinsics"]["fy"], calib["rgb_intrinsics"]["cy"]],
                           [0, 0, 1]], dtype=np.float64)

            # For each RGB output pixel, inverse-project to 3D, then to depth pixel
            ur = np.arange(w, dtype=np.float32)
            vr = np.arange(h, dtype=np.float32)
            ur_grid, vr_grid = np.meshgrid(ur, vr)  # (h, w) — RGB grid

            # RGB pixel → 3D ray (at Z_REF)
            Xr = (ur_grid - Kr[0, 2]) * Z_REF / Kr[0, 0]
            Yr = (vr_grid - Kr[1, 2]) * Z_REF / Kr[1, 1]
            Zr = np.full_like(Xr, Z_REF)

            # Transform to depth camera frame: P_d = R^T · (P_r - T)
            pts_r = np.stack([Xr, Yr, Zr], axis=-1).reshape(-1, 3).T  # (3, h*w)
            pts_d = R_inv @ (pts_r - T)  # (3, h*w)
            pts_d = pts_d.reshape(3, h, w)
            Zd = np.maximum(pts_d[2], 1e-6)
            ud = (pts_d[0] / Zd) * Kd[0, 0] + Kd[0, 2]
            vd = (pts_d[1] / Zd) * Kd[1, 1] + Kd[1, 2]

            # Clamp to image bounds
            map_x = np.clip(ud, 0, w - 1).astype(np.float32)
            map_y = np.clip(vd, 0, h - 1).astype(np.float32)

            self._align_map_x = map_x
            self._align_map_y = map_y
            self._align_ready = True
            logger.info(f"Alignment remap built: depth({w}x{h}) → RGB")
        except Exception as e:
            logger.warn(f"Failed to build alignment remap: {e}")
            self._align_ready = False

    def _align_depth(self, depth_image):
        """Warp depth to RGB coordinate frame using precomputed remap LUT."""
        if depth_image is None or not self._align_ready or np is None:
            return depth_image
        try:
            import cv2
            aligned = cv2.remap(depth_image, self._align_map_x, self._align_map_y,
                                cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
            return aligned
        except Exception:
            return depth_image

    # ── Full ONNX pose inference ──────────────────────────────────

    def _infer_pose_full(self, jpeg, depth_png, depth_image, pose_interval):
        # If pose is disabled or models aren't ready, return last cached skeleton
        if jpeg is None or not self._pose_models_ready:
            return self._last_j2d, self._last_j3d, self._last_bbox, 0.0, 0.0

        self._frame_counter += 1
        if self._frame_counter % pose_interval != 0:
            # ── Skip: reuse last cached skeleton (no flicker) ──
            return self._last_j2d, self._last_j3d, self._last_bbox, 0.0, 0.0

        # ── Run full ONNX inference ──
        j2d, j3d = [], []
        bbox = None; pose_ms = 0.0; yolo_ms = 0.0

        try:
            # 1. YOLO
            t0 = time.monotonic()
            box = self._person_detector.detect_largest_person_jpeg(jpeg) if self._person_detector else None
            yolo_ms = (time.monotonic() - t0) * 1000.0

            # 2. RTMPose
            t1 = time.monotonic()
            if box:
                bb = _engine.BoundingBox2D()
                bb.x = box.x; bb.y = box.y; bb.w = box.w; bb.h = box.h
                bb.valid = box.valid; bb.score = box.score
                self._pose_estimator.set_bounding_box_provider_fallback(bb)
            result = self._pose_estimator.infer_jpeg(jpeg)
            pose_ms = (time.monotonic() - t1) * 1000.0

            if result is None or not result.model_loaded:
                return self._last_j2d, self._last_j3d, self._last_bbox, pose_ms, yolo_ms

            # 3. Halpe26 → Rehab22
            rehab2d = self._halpe_mapper.map(result.keypoints)

            # 4. Depth → 3D
            depth_meters = [0.0] * 22
            if depth_png is not None and self._joint_projector.intrinsics_valid():
                depth_meters = self._depth_sampler.sample_png(depth_png, rehab2d, 0.001)

            j3d_raw = self._joint_projector.project(rehab2d, depth_meters)
            kn = time.monotonic()
            dt = max(0.001, kn - self._last_pose_time) if self._last_pose_time > 0 else 0.033
            j3d_ema = self._ema_filter.filter(j3d_raw, dt)
            j3d_smooth = self._smoother.smooth(j3d_ema)
            self._last_pose_time = kn

            for i in range(22):
                k = j3d_smooth[i]
                j3d.append((k.x, k.y, k.z, k.score, k.valid))
            for i in range(22):
                k = rehab2d[i]
                j2d.append((k.x, k.y, k.score, k.valid))

            used = result.used_box
            if used.valid:
                bbox = (used.x, used.y, used.w, used.h, True, "detect")

        except Exception as e:
            logger.warn(f"Pose inference error: {e}")
            return self._last_j2d, self._last_j3d, self._last_bbox, 0.0, 0.0

        # ── Cache for skipped frames ──
        self._last_j2d, self._last_j3d, self._last_bbox = j2d, j3d, bbox
        self._pose_count += 1; self._pose_since_last += 1
        return j2d, j3d, bbox, pose_ms, yolo_ms

    # ── Helpers ───────────────────────────────────────────────────

    @staticmethod
    def _decode_depth_png(png):
        if png is None or np is None: return None
        try:
            import cv2
            return cv2.imdecode(np.frombuffer(png, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
        except Exception: return None

    @staticmethod
    def _encode_depth_png(depth_img):
        """Encode aligned (or raw) depth back to PNG for C++ DepthSampler."""
        if depth_img is None or np is None: return None
        try:
            import cv2
            ok, buf = cv2.imencode(".png", depth_img, [cv2.IMWRITE_PNG_COMPRESSION, 1])
            return bytes(buf) if ok else None
        except Exception: return None

    def _tick_emg(self):
        if not self._config.emg.enabled: return "disabled", []
        if self._config.emg.mode == "mock":
            phase = time.monotonic()
            return "mock", [900 + 420 * abs(math.sin(phase * 2.1)),
                            760 + 360 * abs(math.sin(phase * 1.8 + 0.7))]
        return "waiting for real EMG data", []

    def _record_skeleton(self, pose_3d):
        ts = time.monotonic_ns(); joints = []
        for j in pose_3d:
            score = j[3] if len(j) > 3 else 1.0
            valid = 1 if (j[4] if len(j) > 4 else False) else 0
            joints.append([j[0], j[1], j[2], score, valid])
        self._recorder.record(
            timestamp_ns=ts, frame_id=self._pair_id, pair_id=self._pair_id,
            dt_seconds=0.033, bbox_mode="full" if not self._stub_mode else "mock",
            joints_3d=joints)

    def _synthetic_pose_full(self):
        t = time.monotonic()
        j2d, j3d = [], []
        layout = [(0,-100),(0,-120),(0,-140),(0,-160),(0,-180),(0,-200),
                  (-30,-140),(-80,-120),(-130,-100),(-180,-80),
                  (30,-140),(80,-120),(130,-100),(180,-80),
                  (-20,0),(-20,80),(-20,160),(-30,240),
                  (20,0),(20,80),(20,160),(30,240)]
        for dx, dy in layout:
            x = 320 + dx + (5 * (t % 1)); y = 240 + dy; z = 2.0
            j2d.append((x, y, 0.9, True))
            j3d.append((x/320-1, -(y/240-1), z, 0.9, True))
        bbox = (100.0, 50.0, 440.0, 430.0, True, "mock")
        return j2d, j3d, bbox

    # ── Performance ───────────────────────────────────────────────

    def _update_performance(self):
        now = time.monotonic(); elapsed = now - self._last_perf_time
        if elapsed >= 1.0:
            with self._perf_lock:
                self._rgb_fps = self._rgb_since_last / elapsed
                self._depth_fps = self._depth_since_last / elapsed
                self._pair_fps = self._pair_since_last / elapsed
                self._pose_fps = self._pose_since_last / elapsed
                self._rgb_since_last = 0; self._depth_since_last = 0
                self._pair_since_last = 0; self._pose_since_last = 0
                self._last_perf_time = now

    def _emit_status(self, message: str):
        logger.info(message)
        if self._on_status: self._on_status(message)
