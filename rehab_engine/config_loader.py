"""
Load PipelineConfig from YAML/JSON files and environment variables.
Replaces the config-loading logic scattered across core/common/Config.h
and the SensorPipeline constructor.
"""

import json
import os
import sys
from dataclasses import asdict, is_dataclass
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
    VoiceConfig,
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
    stack = [(-1, result)]
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or stripped.startswith("%"):
            continue
        if ":" not in stripped:
            continue
        indent = len(line) - len(line.lstrip())
        key, _, raw_value = stripped.partition(":")
        key = key.strip()
        raw_value = raw_value.strip()
        while len(stack) > 1 and indent <= stack[-1][0]:
            stack.pop()
        parent = stack[-1][1]
        if raw_value == "":
            # nested dict
            sub = {}
            parent[key] = sub
            stack.append((indent, sub))
        else:
            value = raw_value.strip('"').strip("'")
            # typed conversion
            if value.lower() in ("true", "yes"):
                val = True
            elif value.lower() in ("false", "no"):
                val = False
            elif value.lower() in ("null", "none", "~"):
                val = None
            elif value.replace(".", "", 1).replace("-", "", 1).isdigit():
                val = float(value) if "." in value else int(value)
            else:
                val = value
            parent[key] = val
    return result


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

        if "pose" in ydata:
            pose = ydata["pose"]
            for k, v in pose.items():
                if hasattr(config.pose, k):
                    setattr(config.pose, k, v)

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

        if "voice" in ydata:
            voice = ydata["voice"]
            for k, v in voice.items():
                if hasattr(config.voice, k):
                    setattr(config.voice, k, v)

    # --- Load emg.yaml ---
    emg_path = root / emg_yaml
    if emg_path.exists():
        edata = _load_yaml(emg_path)
        for k, v in edata.items():
            if hasattr(config.emg, k):
                setattr(config.emg, k, v)

    # --- Load local user preferences written by the settings page ---
    user_path = _user_config_path()
    if user_path.exists():
        try:
            _apply_config_dict(config, json.loads(user_path.read_text(encoding="utf-8")))
        except (OSError, json.JSONDecodeError) as exc:
            logger.warn(f"Unable to load user config {user_path}: {exc}")

    # --- Environment variable overrides ---
    _env_override(config)

    return config


def _user_config_path() -> Path:
    override = os.environ.get("STROKE_USER_CONFIG")
    if override:
        return Path(override).expanduser()
    return Path(__file__).resolve().parent.parent / "config.user.json"


def _apply_config_dict(target, values: dict) -> None:
    """Recursively apply known JSON keys to the config dataclasses."""
    for key, value in values.items():
        if not hasattr(target, key):
            continue
        current = getattr(target, key)
        if is_dataclass(current) and isinstance(value, dict):
            _apply_config_dict(current, value)
        else:
            setattr(target, key, value)


def save_pipeline_config(config: PipelineConfig, path: Optional[Path] = None) -> Path:
    """Persist user-editable preferences without modifying shipped YAML files."""
    output = path or _user_config_path()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(
        json.dumps(asdict(config), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    temporary.replace(output)
    return output


def _env_override(config: PipelineConfig) -> None:
    """Apply STROKE_* environment variable overrides."""
    if os.environ.get("STROKE_EMG_ENABLED"):
        config.emg.enabled = os.environ["STROKE_EMG_ENABLED"] in ("1", "true", "True")
    if os.environ.get("STROKE_EMG_MODE"):
        config.emg.mode = os.environ["STROKE_EMG_MODE"]
    if os.environ.get("STROKE_EMG_CAPTURE_BACKEND"):
        config.emg.capture_backend = os.environ["STROKE_EMG_CAPTURE_BACKEND"]
    if os.environ.get("STROKE_EMG_SERIAL_DEVICE"):
        config.emg.serial_device = os.environ["STROKE_EMG_SERIAL_DEVICE"]
    if os.environ.get("STROKE_EMG_BLE_ADDRESS"):
        config.emg.ble_address = os.environ["STROKE_EMG_BLE_ADDRESS"]
    if os.environ.get("STROKE_EMG_STRICT_REAL"):
        config.emg.strict_real_mode = os.environ["STROKE_EMG_STRICT_REAL"] in (
            "1", "true", "True"
        )
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
    if os.environ.get("STROKE_INFERENCE_BACKEND"):
        config.pose.inference_backend = os.environ["STROKE_INFERENCE_BACKEND"]
    if os.environ.get("STROKE_ONNX_PROVIDER"):
        config.pose.onnx_execution_provider = os.environ["STROKE_ONNX_PROVIDER"]
    if os.environ.get("STROKE_VOICE_ENABLED"):
        config.voice.enabled = os.environ["STROKE_VOICE_ENABLED"].lower() in (
            "1", "true", "yes", "on")
