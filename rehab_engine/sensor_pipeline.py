"""
Sensor pipeline orchestrator.
Replaces core/pipeline/SensorPipeline.cpp (~1600 lines).

Architecture:
    RGB thread ──┐
                 ├── SyncManager (nearest-neighbor) ──► Worker thread ──► Callbacks
    Depth thread ┘

The Worker thread runs: align → pose → map → depth sample → 3D project → filter → record → preview.

In stub mode (no C++ engine), this module runs with synthetic/mock data.
In full mode, it drives the C++ rehab_engine capture & pose classes via pybind11.

Thread safety: uses queue.Queue between capture and worker threads,
               and threading.Lock for the latest preview frame.
"""

import queue
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from ._stub import PipelineConfig, logger
from .preview import PreviewComposer, PreviewFrame
from .recorder import Skeleton3DRecorder


class SensorPipeline:
    """
    Main pipeline orchestrator.

    Usage:
        pipeline = SensorPipeline(config)
        pipeline.set_on_frame(lambda frame: print("got frame"))
        pipeline.start()
        pipeline.start_recording("records/session_001")
        ...
        pipeline.stop_recording()
        pipeline.stop()
    """

    def __init__(self, config: Optional[PipelineConfig] = None):
        self._config = config or PipelineConfig()

        # Queues
        self._pair_queue: queue.Queue = queue.Queue(maxsize=100)

        # Threads
        self._worker_thread: Optional[threading.Thread] = None
        self._running = threading.Event()

        # Modules
        self._preview = PreviewComposer()
        self._recorder = Skeleton3DRecorder()

        # Recording state
        self._recording: bool = False
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
    def preview(self) -> PreviewComposer:
        return self._preview

    def start(self) -> bool:
        """Start the pipeline (in stub mode, starts mock capture)."""
        if self._running.is_set():
            return False
        self._emit_status("Pipeline starting...")

        self._running.set()
        self._worker_thread = threading.Thread(
            target=self._worker_loop, name="pipeline-worker", daemon=True)
        self._worker_thread.start()

        # In stub mode, also start a mock capture thread
        if self._config.emg.mode == "mock" or True:
            mock_thread = threading.Thread(
                target=self._mock_capture_loop, name="mock-capture", daemon=True)
            mock_thread.start()

        self._emit_status("Pipeline started (stub/mock mode)")
        return True

    def stop(self) -> None:
        """Stop the pipeline."""
        self._emit_status("Pipeline stopping...")
        self._running.clear()
        if self._worker_thread and self._worker_thread.is_alive():
            self._worker_thread.join(timeout=3)
        self._emit_status("Pipeline stopped")

    def start_recording(self, save_root: str) -> str:
        """
        Start skeleton recording.
        Returns the session directory path.
        """
        with self._recording_lock:
            if self._recording:
                return self._session_dir

            # Create session directory
            now = datetime.now()
            date_folder = now.strftime("%Y%m%d")
            session_name = now.strftime("%Y%m%d_%H%M%S") + f"_{now.microsecond // 1000:03d}"
            session_dir = Path(save_root) / date_folder / session_name
            session_dir.mkdir(parents=True, exist_ok=True)

            self._recorder.start(str(session_dir))
            self._session_dir = str(session_dir)
            self._recording = True
            self._emit_status(f"Recording started: {session_dir}")
            return self._session_dir

    def stop_recording(self) -> None:
        with self._recording_lock:
            if not self._recording:
                return
            self._recording = False
            self._recorder.stop()
            self._emit_status(f"Recording stopped: {self._session_dir}")

    def recording_stats(self) -> dict:
        s = self._recorder.stats()
        return {
            "recording": s.recording,
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
            }

    # ================================================================
    # Worker loop
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

            # In stub mode, create synthetic pose data
            pose_2d, pose_3d = self._synthetic_pose()

            # Build preview frame
            self._preview.submit(
                joints_2d_raw=pose_2d,
                joints_3d=pose_3d,
                rgb_fps=self._rgb_fps,
                depth_fps=self._depth_fps,
                pair_fps=self._pair_fps,
                pose_fps=0.0,
                queue_length=self._pair_queue.qsize(),
                dropped_pairs=self._dropped,
                delta_ms=0.0,
                recording=self._recording,
                skeleton_recording=self._recording,
                skeleton_frames=self._recorder.stats().frames if self._recording else 0,
            )

            # Record skeleton
            if self._recording:
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
                    bbox_mode="mock",
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
        # Simple 22-joint standing pose centered around (320, 240) at ~2m depth
        joints_2d = []
        joints_3d = []
        base_x, base_y, base_z = 320, 240, 2.0

        # Simplified 22-joint layout
        layout = [
            (0, -100), (0, -120), (0, -140), (0, -160), (0, -180), (0, -200),  # torso→head
            (-30, -140), (-80, -120), (-130, -100), (-180, -80),  # left arm
            (30, -140), (80, -120), (130, -100), (180, -80),      # right arm
            (-20, 0), (-20, 80), (-20, 160), (-30, 240),          # left leg
            (20, 0), (20, 80), (20, 160), (30, 240),              # right leg
        ]

        for dx, dy in layout:
            x = base_x + dx + (5 * (time.monotonic() % 1))  # slight motion
            y = base_y + dy
            z = base_z
            joints_2d.append((x, y, 0.9, True))
            joints_3d.append((x / 320 - 1, -(y / 240 - 1), z, 0.9, True))

        return joints_2d, joints_3d

    # ================================================================
    # Mock capture (stub mode)
    # ================================================================

    def _mock_capture_loop(self) -> None:
        """Simulate RGB+Depth capture in stub mode."""
        interval = 1.0 / max(1, self._config.device.rgb_fps)
        while self._running.is_set():
            # Push a mock synced pair
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