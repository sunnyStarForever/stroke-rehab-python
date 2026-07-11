"""
Load PipelineConfig from YAML/JSON files and environment variables.
Replaces the config-loading logic scattered across core/common/Config.h
and the SensorPipeline constructor.
"""

import json
import os
import sys
from pathlib import Path
from typing import Optional

try:
    import yaml as _yaml
except ImportError:
    _yaml = None

from ._stub import (
    DebugConfig,
    DepthSamplerConfig,
    DeviceConfig,
    EmgConfig,
    PipelineConfig,
    PoseConfig,
    SkeletonFilterConfig,
    SyncConfig,
    logger,
)


def _find_project_root() -> Path:
    """Locate the stroke-rehab project root."""
    # Try environment variable first
    root = os.environ.get("STROKE_REHAB_ROOT")
    if root:
        return Path(root)
    # Walk up from this file (rehab_engine/config_loader.py → python_version/ → stroke-rehab/)
    p = Path(__file__).resolve().parent  # rehab_engine/
    for _ in range(5):
        # Check if this directory contains python_version/ AND configs/
        if (p / "python_version").is_dir() and (p / "configs").is_dir():
            return p
        # Also check if configs/ exists directly (for alternate layouts)
        if (p / "configs").is_dir():
            return p
        if (p / "configs" / "device.yaml").exists():
            return p
        p = p.parent
    return Path.cwd()


def _deep_update(base: dict, overlay: dict) -> dict:
    """Recursively merge overlay into base."""
    for key, value in overlay.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            base[key] = _deep_update(base[key], value)
        else:
            base[key] = value
    return base


def _load_yaml(path: Path) -> dict:
    """Load a YAML file, falling back to basic parsing if yaml not installed."""
    text = path.read_text(encoding="utf-8")
    if _yaml is not None:
        try:
            return _yaml.safe_load(text) or {}
        except Exception:
            pass
    # Fallback: very basic YAML-ish parser for simple key:value files
    result: dict = {}
    indent = 0
    stack = [(result, indent)]
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or stripped.startswith("%"):
            continue
        if ":" not in stripped:
            continue
        key, _, value = stripped.partition(":")
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if value == "":
            # nested dict
            sub = {}
            result[key] = sub
            stack.append((result, indent))
            result = sub
            indent = len(line) - len(line.lstrip())
        else:
            # typed conversion
            if value.lower() in ("true", "yes"):
                val = True
            elif value.lower() in ("false", "no"):
                val = False
            elif value.replace(".", "", 1).replace("-", "", 1).isdigit():
                val = float(value) if "." in value else int(value)
            else:
                val = value
            result[key] = val
    return stack[0][0]


def load_pipeline_config(
    project_root: Optional[Path] = None,
    device_yaml: str = "configs/device.yaml",
    emg_yaml: str = "configs/emg.yaml",
) -> PipelineConfig:
    """
    Load pipeline configuration from config files and environment.

    Priority: env vars > YAML files > defaults in PipelineConfig.
    """
    root = project_root or _find_project_root()
    config = PipelineConfig()

    # --- Load device.yaml ---
    device_path = root / device_yaml
    if device_path.exists():
        ydata = _load_yaml(device_path)

        # Map YAML keys to DeviceConfig fields
        if "depth_capture" in ydata:
            dc = ydata["depth_capture"]
            config.device.openni_device_uri = dc.get("device_uri", config.device.openni_device_uri)
            config.device.enable_hardware_d2c = dc.get("enable_hardware_d2c", True)
            config.device.enable_openni_depth_color_sync = dc.get("enable_depth_color_sync", False)
            config.device.depth_width = dc.get("width", config.device.depth_width)
            config.device.depth_height = dc.get("height", config.device.depth_height)
            config.device.depth_fps = dc.get("fps", config.device.depth_fps)

        if "depth_sampler" in ydata:
            ds = ydata["depth_sampler"]
            for k, v in ds.items():
                if hasattr(config.depth_sampler, k):
                    setattr(config.depth_sampler, k, v)

        if "skeleton_filter" in ydata:
            sf = ydata["skeleton_filter"]
            for k, v in sf.items():
                if hasattr(config.skeleton_filter, k):
                    setattr(config.skeleton_filter, k, v)

        if "debug" in ydata:
            dbg = ydata["debug"]
            for k, v in dbg.items():
                if hasattr(config.debug, k):
                    setattr(config.debug, k, v)

    # --- Load emg.yaml ---
    emg_path = root / emg_yaml
    if emg_path.exists():
        edata = _load_yaml(emg_path)
        for k, v in edata.items():
            if hasattr(config.emg, k):
                setattr(config.emg, k, v)

    # --- Environment variable overrides ---
    _env_override(config)

    return config


def _env_override(config: PipelineConfig) -> None:
    """Apply STROKE_* environment variable overrides."""
    if os.environ.get("STROKE_EMG_ENABLED"):
        config.emg.enabled = os.environ["STROKE_EMG_ENABLED"] in ("1", "true", "True")
    if os.environ.get("STROKE_EMG_MODE"):
        config.emg.mode = os.environ["STROKE_EMG_MODE"]
    if os.environ.get("STROKE_EMG_SERIAL_DEVICE"):
        config.emg.serial_device = os.environ["STROKE_EMG_SERIAL_DEVICE"]
    if os.environ.get("STROKE_EMG_RPMSG_CTRL"):
        config.emg.rpmsg_ctrl_device = os.environ["STROKE_EMG_RPMSG_CTRL"]
    if os.environ.get("STROKE_EMG_RPMSG_DATA"):
        config.emg.rpmsg_data_device = os.environ["STROKE_EMG_RPMSG_DATA"]
    if os.environ.get("STROKE_EMG_ENDPOINT"):
        config.emg.rpmsg_endpoint_name = os.environ["STROKE_EMG_ENDPOINT"]

    # Camera
    if os.environ.get("STROKE_RGB_DEVICE"):
        config.device.rgb_device_path = os.environ["STROKE_RGB_DEVICE"]
    if os.environ.get("STROKE_MIRROR_RGB"):
        config.device.mirror_rgb_at_capture = os.environ["STROKE_MIRROR_RGB"] != "0"

    # Calibration
    if os.environ.get("STROKE_CALIBRATION_FILE"):
        config.calibration_file = os.environ["STROKE_CALIBRATION_FILE"]