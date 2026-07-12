"""
Skeleton 3D recorder and EMG recorder.
Replaces core/record/Skeleton3DRecorder.cpp + core/emg/EmgRecorder.cpp.
Pure Python, uses csv module.
"""

import csv
import json
import os
import threading
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


class Skeleton3DRecorder:
    """Records skeleton_3d.csv, skeleton3d_debug.csv, and meta.json."""

    def __init__(self):
        self._lock = threading.Lock()
        self._csv: Optional[TextIO] = None
        self._csv_writer: Any = None
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
        debug_path = sd / "skeleton3d_debug.csv"

        try:
            self._csv = open(csv_path, "w", newline="", encoding="utf-8")
            self._debug_csv = open(debug_path, "w", newline="", encoding="utf-8")
        except OSError:
            if self._csv:
                self._csv.close()
                self._csv = None
            if self._debug_csv:
                self._debug_csv.close()
                self._debug_csv = None
            return False

        # Build CSV header: timestamp_ns,frame_id,..., plus 22 joints × (x,y,z,score,valid)
        header = ["timestamp_ns", "frame_id", "pair_id", "dt_seconds", "bbox_mode"]
        for name in _REHAB22_JOINT_NAMES:
            header.extend([f"{name}_x", f"{name}_y", f"{name}_z",
                          f"{name}_score", f"{name}_valid"])
        self._csv_writer = csv.DictWriter(self._csv, fieldnames=header)
        self._csv_writer.writeheader()

        # Debug CSV (simplified)
        debug_header = ["timestamp_ns", "frame_id"] + [
            f"{n}_{suffix}" for n in _REHAB22_JOINT_NAMES
            for suffix in ("sample_mm", "sample_method", "alpha", "ema_reason")]
        self._debug_writer = csv.DictWriter(self._debug_csv, fieldnames=debug_header)
        self._debug_writer.writeheader()

        # Meta
        self._meta = _SessionMeta(
            start_time=datetime.now(timezone.utc).isoformat(),
            **(meta or {}),
        )
        self._write_meta("")

        self._session_dir = sd
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
               joints_3d: List[List[float]]) -> bool:
        """
        Record one frame of 3D skeleton data.
        joints_3d: list of 22 [x, y, z] arrays.
        """
        with self._lock:
            # Re-check after acquiring the lock: stop() may have run between
            # the caller's frame check and this write.
            if not self._recording or self._csv is None or self._csv_writer is None:
                return False
            self._frame_count += 1

            if len(joints_3d) != 22:
                self._skipped += 1
                return False

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
                self._csv_writer.writerow(row)
                self._row_count += 1
            except Exception:
                self._skipped += 1

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
        self._action_dir: Optional[Path] = None
        self._raw_csv: Optional[TextIO] = None
        self._raw_writer: Any = None
        self._feat_csv: Optional[TextIO] = None
        self._feat_writer: Any = None
        self._raw_count: int = 0
        self._feat_count: int = 0

    def start(self, action_dir: str) -> bool:
        self.stop()
        ad = Path(action_dir)
        ad.mkdir(parents=True, exist_ok=True)
        self._action_dir = ad
        self._raw_csv = open(ad / "emg_raw.csv", "w", newline="", encoding="utf-8")
        self._raw_writer = csv.writer(self._raw_csv)
        self._raw_writer.writerow(["timestamp_ns", "seq", "channel_0", "channel_1"])
        self._feat_csv = open(ad / "emg_features.csv", "w", newline="", encoding="utf-8")
        self._feat_writer = csv.writer(self._feat_csv)
        self._feat_writer.writerow(["timestamp_ns", "seq", "channel", "rms", "zcr", "cv",
                                    "fatigue_index", "muscle_state"])
        self._raw_count = 0
        self._feat_count = 0
        return True

    def stop(self) -> None:
        for f in (self._raw_csv, self._feat_csv):
            if f:
                f.close()
        self._raw_csv = None
        self._feat_csv = None
        self._write_summary()

    def record_raw(self, timestamp_ns: int, seq: int, channels: List[int]) -> None:
        if self._raw_writer:
            self._raw_writer.writerow([timestamp_ns, seq] + list(channels))
            self._raw_count += 1

    def record_feature(self, timestamp_ns: int, seq: int, channel: int,
                       rms: float, zcr: float, cv: float,
                       fatigue_index: float, state: str) -> None:
        if self._feat_writer:
            self._feat_writer.writerow(
                [timestamp_ns, seq, channel, f"{rms:.4f}", f"{zcr:.4f}",
                 f"{cv:.4f}", f"{fatigue_index:.4f}", state])
            self._feat_count += 1

    def _write_summary(self) -> None:
        if not self._action_dir:
            return
        summary = {
            "raw_samples": self._raw_count,
            "feature_frames": self._feat_count,
        }
        (self._action_dir / "emg_summary.json").write_text(
            json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
