"""
Stroke Rehab Python Application Layer.

Python owns configuration, lifecycle, business logic and UI.  The optional
``rehab_engine._core`` module is a hardware/inference adapter only.

To use in full mode (with compiled C++ engine):
    from rehab_engine._core import PipelineConfig, ...
"""

import os
import sys

# Try to import the compiled C++ module (named _core.so / _core.pyd).
# It is built from bindings/module.cpp with PYBIND11_MODULE(_core, m) {...}
_STUB_MODE = False
_core_loaded = False
_CORE_READY = False
try:
    from . import _core  # noqa: F401 — compiled pybind11 module
    _core_loaded = True
except ImportError:
    _core_loaded = False

if _core_loaded:
    _CORE_READY = all(
        hasattr(_core, name)
        for name in ("RgbCaptureV4L2", "DepthCaptureOpenNI", "DeviceConfig")
    )

# ---- Logger adapter ----
# C++ Logger bindings only export set_callback().  Add info/warn/error so all
# Python code that calls logger.info(msg) etc. works in full mode as well.

class _FullLogger:
    """Logger wrapper for C++ _core.logger — adds info/warn/error + stderr output."""

    def __init__(self, core_logger):
        self._core = core_logger
        self._cb = None   # (level, msg) -> None

    def set_callback(self, callback):
        self._cb = callback
        if callback is not None:
            self._core.set_callback(callback)

    # -- public logging API (mirrors _StubLogger) --
    def info(self, msg: str):
        print(f"[INFO] {msg}", flush=True, file=sys.stderr)
        if self._cb:
            self._cb("INFO", msg)

    def warn(self, msg: str):
        print(f"[WARN] {msg}", flush=True, file=sys.stderr)
        if self._cb:
            self._cb("WARN", msg)

    def error(self, msg: str):
        print(f"[ERROR] {msg}", flush=True, file=sys.stderr)
        if self._cb:
            self._cb("ERROR", msg)

    def performance(self, msg: str):
        print(f"[PERF] {msg}", flush=True, file=sys.stderr)
        if self._cb:
            self._cb("PERF", msg)


# Configuration is always Python-owned.  This prevents loading the native
# adapter from silently changing which runtime fields are available.
from ._stub import (  # noqa: F401
    DeviceConfig,
    SyncConfig,
    PoseConfig,
    DepthSamplerConfig,
    SkeletonFilterConfig,
    DebugConfig,
    VoiceConfig,
    EmgConfig,
    PipelineConfig,
    logger as _stub_logger,
    __engine_version__ as _stub_engine_version,
)

if _CORE_READY:
    logger = _FullLogger(_core.logger)
    __engine_version__ = getattr(_core, "__version__", "unknown")
else:
    logger = _stub_logger
    __engine_version__ = "real-core-unavailable"

# Diagnostics (run before UI to check hardware status)
from .diagnostics import Diagnostics, DiagItem, run_diagnostics, print_diagnostics

# Stage 2 application modules (pure Python)
from .config_loader import load_pipeline_config, save_pipeline_config
from .course import CourseRepository, CourseRunner, Course, CourseAction
from .scoring import (
    OfflineReportRunner,
    ScoreBridge,
    ScoreResult,
    ScoringCsvRecorder,
    ScoringSkeletonAdapter,
)
from .recorder import Skeleton3DRecorder, EmgRecorder
from .emg import (
    BleGattCapture,
    EmgBleNotifyParser,
    EmgFeatureFrame,
    EmgFeatureProcessor,
    EmgFusionBuffer,
    EmgIntervalSummary,
    EmgBluetoothDevice,
    EmgBluetoothScanResult,
    EmgBluetoothScanner,
    EmgManager,
    EmgRawSample,
    EmgRawChunk,
    EmgRpmsgClient,
    EmgRpmsgProtocol,
    SerialEmgCapture,
    EmgRuntimeStatus,
)
from .sensor_pipeline import RecordingOptions, SensorPipeline
from .preview import PreviewComposer, PreviewFrame
from .reporting import analyze_skeleton_csv, generate_session_report
from .inference import (
    AdaptiveRoiTracker,
    BoundingBox2D,
    Keypoint2D,
    PersonDetector,
    PoseInferenceResult,
    RtmposeEstimator,
    map_halpe26_to_rehab22,
)
from .capture import (
    FrameEnvelope,
    FrameSource,
    FrameSynchronizer,
    LatestFrameQueue,
    NativeRgbDepthBackend,
    SyncedFramePair,
    TimestampNormalizer,
)
from .alignment import (
    CameraIntrinsics,
    RegistrationCalibration,
    SoftwareRegistrationAligner,
    load_calibration,
)
from .pose3d import (
    DepthSampleContext,
    DepthSampleMethod,
    DepthSampleResult,
    DepthSampler,
    EmaSkeletonFilter,
    Joint3D,
    JointProjector3D,
    SkeletonSmoother,
)
from .voice import VoiceAssistant

__all__ = [
    # Config
    "PipelineConfig",
    "DeviceConfig",
    "SyncConfig",
    "PoseConfig",
    "DepthSamplerConfig",
    "SkeletonFilterConfig",
    "DebugConfig",
    "VoiceConfig",
    "EmgConfig",
    "load_pipeline_config",
    "save_pipeline_config",
    "logger",
    # Diagnostics
    "Diagnostics",
    "DiagItem",
    "run_diagnostics",
    "print_diagnostics",
    # Pipeline
    "SensorPipeline",
    "RecordingOptions",
    "PreviewComposer",
    "PreviewFrame",
    # Business
    "CourseRepository",
    "CourseRunner",
    "Course",
    "CourseAction",
    # Scoring
    "ScoreBridge",
    "ScoreResult",
    "OfflineReportRunner",
    "ScoringCsvRecorder",
    "ScoringSkeletonAdapter",
    # Recording
    "Skeleton3DRecorder",
    "EmgRecorder",
    "BleGattCapture",
    "EmgBleNotifyParser",
    "EmgFeatureFrame",
    "EmgFeatureProcessor",
    "EmgFusionBuffer",
    "EmgIntervalSummary",
    "EmgBluetoothDevice",
    "EmgBluetoothScanResult",
    "EmgBluetoothScanner",
    "EmgManager",
    "EmgRawSample",
    "EmgRawChunk",
    "EmgRpmsgClient",
    "EmgRpmsgProtocol",
    "SerialEmgCapture",
    "EmgRuntimeStatus",
    "analyze_skeleton_csv",
    "generate_session_report",
    "BoundingBox2D",
    "AdaptiveRoiTracker",
    "Keypoint2D",
    "PersonDetector",
    "PoseInferenceResult",
    "RtmposeEstimator",
    "map_halpe26_to_rehab22",
    "FrameEnvelope",
    "FrameSource",
    "FrameSynchronizer",
    "LatestFrameQueue",
    "NativeRgbDepthBackend",
    "SyncedFramePair",
    "TimestampNormalizer",
    "CameraIntrinsics",
    "RegistrationCalibration",
    "SoftwareRegistrationAligner",
    "load_calibration",
    "DepthSampleContext",
    "DepthSampleMethod",
    "DepthSampleResult",
    "DepthSampler",
    "EmaSkeletonFilter",
    "Joint3D",
    "JointProjector3D",
    "SkeletonSmoother",
    "VoiceAssistant",
]
