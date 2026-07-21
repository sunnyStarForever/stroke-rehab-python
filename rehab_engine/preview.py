"""
Preview composer: converts pipeline output into displayable frames.
Replaces core/pipeline/PreviewComposer.cpp + core/pipeline/PreviewFrame.h.

In the Python architecture, this is a pure computation module — no Qt.
The UI layer (PySide6 in Stage 3) will read PreviewFrame and paint it.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import Any, List, Optional, Tuple


@dataclass
class Joint2DDisplay:
    """A single joint for display purposes."""
    x: float = 0.0
    y: float = 0.0
    score: float = 0.0
    valid: bool = False
    name: str = ""


@dataclass
class Joint3DDisplay:
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0
    score: float = 0.0
    valid: bool = False
    name: str = ""


@dataclass
class PreviewFrame:
    """
    Lightweight preview data snapshot for the UI.
    Analogous to the C++ PreviewFrame struct.
    """
    seq: int = 0
    pair_id: int = 0
    rgb_frame_id: int = 0
    depth_frame_id: int = 0
    host_ts_ns: int = 0
    rgb_width: int = 0
    rgb_height: int = 0
    depth_width: int = 0
    depth_height: int = 0
    pose_interval: int = 0
    # Pose
    joints_2d: List[Joint2DDisplay] = field(default_factory=list)    # 22 Rehab22 joints
    joints_3d: List[Joint3DDisplay] = field(default_factory=list)
    raw_joints_3d: List[Joint3DDisplay] = field(default_factory=list)
    ema_joints_3d: List[Joint3DDisplay] = field(default_factory=list)
    has_valid_2d: bool = False
    has_valid_3d: bool = False

    # FPS / performance
    raw_rgb_fps: float = 0.0
    raw_depth_fps: float = 0.0
    sync_fps: float = 0.0
    worker_fps: float = 0.0
    rgb_fps: float = 0.0  # compatibility alias for raw_rgb_fps
    depth_fps: float = 0.0  # compatibility alias for raw_depth_fps
    pair_fps: float = 0.0
    pose_fps: float = 0.0
    yolo_ms: float = 0.0
    pose_ms: float = 0.0
    record_write_ms: float = 0.0
    queue_length: int = 0
    dropped_pairs: int = 0
    delta_ms: float = 0.0

    # Bounding box
    bbox_valid: bool = False
    bbox_x: float = 0.0
    bbox_y: float = 0.0
    bbox_w: float = 0.0
    bbox_h: float = 0.0
    bbox_mode: str = "full_fallback"

    # Recording status
    recording: bool = False
    skeleton_recording: bool = False
    skeleton_saved_frames: int = 0
    rgb_recorded_frames: int = 0
    depth_recorded_frames: int = 0

    # EMG
    emg_status: str = ""
    emg_rms: List[float] = field(default_factory=list)
    emg_fatigue_index: List[float] = field(default_factory=list)

    # Mirror
    mirror: bool = False

    # RGB image data (full mode: decoded JPEG frame for UI display)
    rgb_image: Any = None

    # Depth image data (full mode: decoded 16-bit PNG for overlay, mm scale)
    depth_image: Any = None
    depth_is_hardware: bool = False

    # Bone connections for skeleton drawing (index pairs into joints_2d)
    bones: List[Tuple[int, int]] = field(default_factory=list)


_REHAB22_BONES = [
    (0, 1), (1, 2), (2, 3), (3, 4), (4, 5),           # torso → head
    (3, 6), (6, 7), (7, 8), (8, 9),                    # left arm
    (3, 10), (10, 11), (11, 12), (12, 13),              # right arm
    (0, 14), (14, 15), (15, 16), (16, 17),              # left leg
    (0, 18), (18, 19), (19, 20), (20, 21),              # right leg
]

_REHAB22_JOINT_NAMES = [
    "Waist", "Spine", "Chest", "Neck", "Head", "HeadTip",
    "LeftCollar", "LeftUpperArm", "LeftForearm", "LeftHand",
    "RightCollar", "RightUpperArm", "RightForearm", "RightHand",
    "LeftUpperLeg", "LeftLowerLeg", "LeftFoot", "LeftToes",
    "RightUpperLeg", "RightLowerLeg", "RightFoot", "RightToes",
]


class PreviewComposer:
    """
    Converts aligned frame data to a PreviewFrame for UI display.
    Thread-safe latest-frame-only pattern.

    The UI (Stage 3 PySide6) calls latest_frame() on a timer.
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._latest: Optional[PreviewFrame] = None
        self._seq = 0

    def submit(self,
               pair_id: int = 0,
               rgb_frame_id: int = 0,
               depth_frame_id: int = 0,
               host_ts_ns: int = 0,
               rgb_width: int = 0,
               rgb_height: int = 0,
               depth_width: int = 0,
               depth_height: int = 0,
               pose_interval: int = 0,
               # Pose arrays: 22-element lists of [x,y,z], [x,y], etc.
               joints_2d_raw: Optional[List[Tuple[float, float, float, bool]]] = None,
               joints_3d: Optional[List[Tuple[float, float, float, float, bool]]] = None,
               raw_joints_3d: Optional[List[Tuple[float, float, float, float, bool]]] = None,
               ema_joints_3d: Optional[List[Tuple[float, float, float, float, bool]]] = None,
               # Performance
               raw_rgb_fps: float = 0.0,
               raw_depth_fps: float = 0.0,
               sync_fps: float = 0.0,
               worker_fps: float = 0.0,
               rgb_fps: float = 0.0,
               depth_fps: float = 0.0,
               pair_fps: float = 0.0,
               pose_fps: float = 0.0,
               yolo_ms: float = 0.0,
               pose_ms: float = 0.0,
               record_write_ms: float = 0.0,
               queue_length: int = 0,
               dropped_pairs: int = 0,
               delta_ms: float = 0.0,
               # BBox
               bbox: Optional[Tuple[float, float, float, float, bool, str]] = None,
               # Recording
               recording: bool = False,
               skeleton_recording: bool = False,
               skeleton_frames: int = 0,
               rgb_frames: int = 0,
               depth_frames: int = 0,
               # EMG
               emg_status: str = "",
               emg_rms: Optional[List[float]] = None,
               emg_fatigue: Optional[List[float]] = None,
               mirror: bool = False,
               # Real RGB image (numpy array, full mode)
               rgb_image=None,
               # Real depth image (numpy 16-bit array, full mode)
               depth_image=None,
               depth_is_hardware: bool = False,
               ) -> None:
        """Submit new frame data. Called from the pipeline worker thread."""
        frame = PreviewFrame()
        self._seq += 1
        frame.seq = self._seq
        frame.pair_id = pair_id
        frame.rgb_frame_id = rgb_frame_id
        frame.depth_frame_id = depth_frame_id
        frame.host_ts_ns = host_ts_ns
        frame.rgb_width = rgb_width
        frame.rgb_height = rgb_height
        frame.depth_width = depth_width
        frame.depth_height = depth_height
        frame.pose_interval = pose_interval
        frame.bones = list(_REHAB22_BONES)

        if joints_2d_raw:
            for i, j in enumerate(joints_2d_raw):
                jd = Joint2DDisplay(x=j[0], y=j[1], score=j[2] if len(j) > 2 else 0,
                                    valid=j[3] if len(j) > 3 else False,
                                    name=_REHAB22_JOINT_NAMES[i] if i < 22 else "")
                frame.joints_2d.append(jd)
            frame.has_valid_2d = any(j.valid for j in frame.joints_2d)

        for source, target in [
            (joints_3d if depth_is_hardware else None, frame.joints_3d),
            (raw_joints_3d if depth_is_hardware else None, frame.raw_joints_3d),
            (ema_joints_3d if depth_is_hardware else None, frame.ema_joints_3d),
        ]:
            if source:
                for i, j in enumerate(source):
                    jd = Joint3DDisplay(
                        x=j[0], y=j[1], z=j[2],
                        score=j[3] if len(j) > 3 else 0,
                        valid=j[4] if len(j) > 4 else False,
                        name=_REHAB22_JOINT_NAMES[i] if i < 22 else "")
                    target.append(jd)
        frame.has_valid_3d = bool(depth_is_hardware) and any(
            j.valid for j in frame.joints_3d)

        frame.raw_rgb_fps = raw_rgb_fps or rgb_fps
        frame.raw_depth_fps = raw_depth_fps or depth_fps
        frame.sync_fps = sync_fps
        frame.worker_fps = worker_fps or pair_fps
        frame.rgb_fps = frame.raw_rgb_fps
        frame.depth_fps = frame.raw_depth_fps
        frame.pair_fps = frame.worker_fps
        frame.pose_fps = pose_fps
        frame.yolo_ms = yolo_ms
        frame.pose_ms = pose_ms
        frame.record_write_ms = record_write_ms
        frame.queue_length = queue_length
        frame.dropped_pairs = dropped_pairs
        frame.delta_ms = delta_ms

        if bbox:
            frame.bbox_x, frame.bbox_y, frame.bbox_w, frame.bbox_h, frame.bbox_valid, frame.bbox_mode = bbox

        frame.recording = recording
        frame.skeleton_recording = skeleton_recording
        frame.skeleton_saved_frames = skeleton_frames
        frame.rgb_recorded_frames = rgb_frames
        frame.depth_recorded_frames = depth_frames

        frame.emg_status = emg_status
        frame.emg_rms = emg_rms or []
        frame.emg_fatigue_index = emg_fatigue or []

        frame.mirror = mirror
        frame.rgb_image = rgb_image
        frame.depth_image = depth_image if depth_is_hardware else None
        frame.depth_is_hardware = bool(depth_is_hardware)

        with self._lock:
            self._latest = frame

    def latest_frame(self) -> Optional[PreviewFrame]:
        """Get the latest preview frame (called from UI timer)."""
        with self._lock:
            return self._latest
