"""
Stroke Rehab Python Application Layer.

Stage 1 (C++ engine):  hardware drivers, AI inference  →  `rehab_engine._core`
Stage 2 (Python app):  pipeline orchestration, scoring, courses, recording
Stage 3 (future):      PySide6 UI

To use in stub mode (desktop dev):
    from rehab_engine._stub import PipelineConfig, ...

To use in full mode (with compiled C++ engine):
    from rehab_engine._core import PipelineConfig, ...
"""

import sys

# Try to import the compiled C++ module (named _core.so / _core.pyd).
# It is built from bindings/module.cpp with PYBIND11_MODULE(_core, m) {...}
_STUB_MODE = False
_core_loaded = False
try:
    from . import _core  # noqa: F401 — compiled pybind11 module
    _core_loaded = True
except ImportError:
    _STUB_MODE = True

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


if _STUB_MODE:
    # Stub mode: use pure-Python dataclass config types
    from ._stub import (  # noqa: F401
        DeviceConfig,
        SyncConfig,
        PoseConfig,
        DepthSamplerConfig,
        SkeletonFilterConfig,
        DebugConfig,
        EmgConfig,
        PipelineConfig,
        logger,
        __engine_version__,
    )
else:
    # Full mode: the compiled _core module provides all types
    DeviceConfig = _core.DeviceConfig
    SyncConfig = _core.SyncConfig
    PoseConfig = _core.PoseConfig
    DepthSamplerConfig = _core.DepthSamplerConfig
    SkeletonFilterConfig = _core.SkeletonFilterConfig
    DebugConfig = _core.DebugConfig
    EmgConfig = _core.EmgConfig
    PipelineConfig = _core.PipelineConfig
    logger = _FullLogger(_core.logger)
    __engine_version__ = getattr(_core, "__version__", "unknown")

# Diagnostics (run before UI to check hardware status)
from .diagnostics import Diagnostics, DiagItem, run_diagnostics, print_diagnostics

# Stage 2 application modules (pure Python)
from .config_loader import load_pipeline_config, save_pipeline_config
from .course import CourseRepository, CourseRunner, Course, CourseAction
from .scoring import ScoreBridge, OfflineReportRunner, ScoreResult
from .recorder import Skeleton3DRecorder, EmgRecorder
from .sensor_pipeline import SensorPipeline
from .preview import PreviewComposer, PreviewFrame
from .reporting import analyze_skeleton_csv, generate_session_report

__all__ = [
    # Config
    "PipelineConfig",
    "DeviceConfig",
    "SyncConfig",
    "PoseConfig",
    "DepthSamplerConfig",
    "SkeletonFilterConfig",
    "DebugConfig",
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
    # Recording
    "Skeleton3DRecorder",
    "EmgRecorder",
    "analyze_skeleton_csv",
    "generate_session_report",
]
