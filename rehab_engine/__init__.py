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

# Try to import the compiled C++ module; fall back to stub if not found.
_STUB_MODE = False
try:
    from . import _core  # noqa: F401 — compiled pybind11 module
except ImportError:
    _STUB_MODE = True

# Re-export from appropriate backend
if _STUB_MODE:
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
    # When the compiled engine is available, import directly.
    # The user is expected to have the .pyd/.so in sys.path.
    import rehab_engine as _core  # type: ignore[import-not-found]

    DeviceConfig = _core.DeviceConfig
    SyncConfig = _core.SyncConfig
    PoseConfig = _core.PoseConfig
    DepthSamplerConfig = _core.DepthSamplerConfig
    SkeletonFilterConfig = _core.SkeletonFilterConfig
    DebugConfig = _core.DebugConfig
    EmgConfig = _core.EmgConfig
    PipelineConfig = _core.PipelineConfig
    logger = _core.logger
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
