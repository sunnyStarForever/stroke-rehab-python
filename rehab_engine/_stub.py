"""
Stub config types for desktop development.
Mirrors the C++ Config.h structs as pure Python dataclasses.
"""

from dataclasses import dataclass, field
from typing import Optional

__engine_version__ = "0.1.0-stub"

# Re-exported for convenience — actual value is set by __init__.py
_STUB_MODE = True


# ============================================================
# Logger stub
# ============================================================
class _StubLogger:
    """Callback-based logger matching the C++ Logger API."""

    def __init__(self):
        self._callback = None

    def set_callback(self, callback):
        """Set callback(level: str, message: str) or None to disable."""
        self._callback = callback

    def _log(self, level: str, msg: str):
        if self._callback:
            self._callback(level, msg)

    def info(self, msg: str):
        self._log("INFO", msg)

    def warn(self, msg: str):
        self._log("WARN", msg)

    def error(self, msg: str):
        self._log("ERROR", msg)

    def performance(self, msg: str):
        self._log("PERF", msg)


logger = _StubLogger()


# ============================================================
# Config dataclasses
# ============================================================

@dataclass
class DeviceConfig:
    openni_device_uri: str = ""
    rgb_device_path: str = ""
    rgb_pixel_format: str = "MJPG"
    rgb_device_index: int = 0
    rgb_width: int = 640
    rgb_height: int = 480
    rgb_fps: int = 30
    mirror_rgb_at_capture: bool = True
    depth_pixel_format: str = "DEPTH_1_MM"
    depth_width: int = 640
    depth_height: int = 480
    depth_fps: int = 30
    enable_hardware_d2c: bool = True
    enable_openni_color_stream_for_debug: bool = False
    enable_openni_depth_color_sync: bool = False
    latest_queue_size: int = 1
    raw_perf_log_interval_sec: float = 1.0
    callback_perf_window_sec: float = 5.0
    callback_perf_min_samples: int = 30
    callback_normal_p95_ms: float = 8.0
    callback_warn_p95_ms: float = 10.0
    callback_critical_p95_ms: float = 25.0
    callback_warn_sustain_sec: float = 5.0
    callback_low_fps_sustain_sec: float = 3.0
    callback_recovery_sec: float = 5.0
    enable_cpu_affinity: bool = True
    rgb_capture_cpu: int = 0
    depth_capture_cpu: int = 0


@dataclass
class SyncConfig:
    match_threshold_ns: int = 20_000_000  # 20ms
    queue_size: int = 30


@dataclass
class PoseConfig:
    enable_pose: bool = True
    enable_adaptive_roi: bool = True
    model_path: str = ""
    detector_model_path: str = ""
    pipeline_json_path: str = ""
    detail_json_path: str = ""
    deploy_json_path: str = ""
    min_score: float = 0.15
    max_pair_queue: int = 2
    pose_interval: int = 6
    enable_pose_reuse: bool = True
    enable_cpu_affinity: bool = True
    sync_or_ui_cpu: int = 1
    detector_cpu: int = 2
    pose_cpu: int = 3
    detector_interval: int = 30
    roi_margin_ratio: float = 0.20
    min_track_mean_score: float = 0.25
    min_track_valid_points: int = 6
    max_consecutive_misses: int = 3
    motion_trigger_ratio: float = 0.35
    detector_input_size: int = 320
    detector_conf_threshold: float = 0.35
    detector_nms_threshold: float = 0.45
    depth_median_window: int = 5
    enable_smoothing: bool = True
    smoothing_alpha: float = 0.35
    inference_backend: str = "python"  # python / native / auto
    onnx_execution_provider: str = "auto"
    onnx_intra_op_threads: int = 1
    onnx_inter_op_threads: int = 1


@dataclass
class DepthSamplerConfig:
    min_depth_mm: int = 300
    max_depth_mm: int = 5000
    body_depth_band_mm: int = 700
    edge_body_depth_band_mm: int = 900
    background_reject_margin_mm: int = 500
    background_match_band_mm: int = 300
    foreground_percentile: float = 0.20
    min_foreground_pixels: int = 3
    hip_radius: int = 2
    knee_radius: int = 3
    ankle_radius: int = 4
    toe_radius: int = 5
    wrist_radius: int = 4
    default_radius: int = 2
    limb_inward_search_enabled: bool = True
    limb_inward_steps: int = 10
    limb_inward_step_px: int = 3
    limb_inward_radius: int = 3


@dataclass
class SkeletonFilterConfig:
    mode: str = "ema"
    alpha_good: float = 0.65
    alpha_low_confidence: float = 0.35
    alpha_recovered: float = 0.45
    alpha_invalid: float = 0.0
    max_z_jump_m: float = 0.45
    max_joint_speed_mps: float = 2.5
    hold_last_when_invalid: bool = True
    max_hold_frames: int = 5


@dataclass
class DebugConfig:
    save_depth_sampling_overlay: bool = False
    save_skeleton_raw_csv: bool = True
    save_skeleton_ema_csv: bool = True


@dataclass
class VoiceConfig:
    enabled: bool = True
    backend: str = "auto"
    rate: int = 175
    volume: float = 0.9
    cooldown_seconds: float = 2.0
    queue_size: int = 12


@dataclass
class EmgConfig:
    enabled: bool = False
    mode: str = "mock"  # disabled / mock / real
    capture_backend: str = "serial"  # serial / bluez
    serial_device: str = "/dev/rfcomm0"
    serial_baud_rate: int = 115200
    ble_name_prefix: str = "ESP32_EMG"
    ble_address: str = ""
    ble_service_uuid: str = ""
    ble_command_rx_uuid: str = ""
    ble_status_tx_uuid: str = ""
    ble_notify_char_uuid: str = ""
    ble_command_timeout_ms: int = 5000
    sample_rate_hz: int = 1000
    channel_count: int = 2
    raw_chunk_samples: int = 25
    strict_real_mode: bool = True
    rpmsg_enabled: bool = True
    rpmsg_ctrl_device: str = "/dev/rpmsg_ctrl0"
    rpmsg_data_device: str = "/dev/rpmsg0"
    rpmsg_endpoint_name: str = "emg_rpmsg"
    rpmsg_poll_timeout_ms: int = 5
    active_threshold: float = 800.0
    noise_threshold: float = 15.0


@dataclass
class PipelineConfig:
    device: DeviceConfig = field(default_factory=DeviceConfig)
    sync: SyncConfig = field(default_factory=SyncConfig)
    pose: PoseConfig = field(default_factory=PoseConfig)
    depth_sampler: DepthSamplerConfig = field(default_factory=DepthSamplerConfig)
    skeleton_filter: SkeletonFilterConfig = field(default_factory=SkeletonFilterConfig)
    debug: DebugConfig = field(default_factory=DebugConfig)
    voice: VoiceConfig = field(default_factory=VoiceConfig)
    emg: EmgConfig = field(default_factory=EmgConfig)
    calibration_file: str = ""
    record_pairs: bool = False
    record_path: str = "recordings"
    async_video_recording: bool = True
    recording_queue_capacity: int = 90
    recording_drain_timeout_sec: float = 5.0
    selected_course_id: str = ""
    patient_name: str = ""
    patient_id: str = "P0001"
    patient_gender: str = ""
    patient_age: int = 0
    patient_diagnosis: str = ""
    ui_debug_enabled: bool = False
