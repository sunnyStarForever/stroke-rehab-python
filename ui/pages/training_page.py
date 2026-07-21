"""
Training page — the core interface for rehabilitation training.
Replaces app/pages/TrainingPage.cpp (~1100 lines).

Layout:
  ┌─────────────────────────────────────────────────────┐
  │  Top bar: course title · action · status · timer   │
  ├──────────────────────────┬──────────────────────────┤
  │                          │  Course info             │
  │    Preview (720×480)     │  · current action        │
  │    + skeleton overlay    │  · target reps / progress│
  │    + performance HUD     │  · rest countdown        │
  │                          │  ─────────────────       │
  │                          │  Scores (6-dimension)    │
  │                          │  ─────────────────       │
  │                          │  EMG status              │
  ├──────────────────────────┴──────────────────────────┤
  │  Feedback log (read-only)                           │
  ├─────────────────────────────────────────────────────┤
  │  [Start] [Pause] [Stop]  ☑ Record RGB    [Report]  │
  └─────────────────────────────────────────────────────┘
"""

import os
import json
import threading
import time
from collections import deque
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Optional

from PyQt5.QtCore import Qt, QTimer, pyqtSignal
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
    QLabel, QSizePolicy,
)

from qfluentwidgets import (
    CardWidget, SimpleCardWidget,
    PrimaryPushButton, PushButton,
    TitleLabel, SubtitleLabel, BodyLabel, StrongBodyLabel, CaptionLabel,
    InfoBar, InfoBarPosition,
    MessageBox,
    LineEdit, ProgressRing, IndeterminateProgressRing,
    TextEdit, ScrollArea,
)

from rehab_engine.course import CourseRepository, CourseRunner, RunnerState
from rehab_engine.scoring import (
    OfflineReportRunner,
    ScoreBridge,
    ScoreResult,
    ScoringCsvRecorder,
    ScoringSkeletonAdapter,
)
from rehab_engine.sensor_pipeline import SensorPipeline
from rehab_engine.preview import PreviewComposer, PreviewFrame
from rehab_engine.voice import VoiceAssistant
from rehab_engine import PipelineConfig, logger
import rehab_engine  # for rehab_engine._STUB_MODE


from ..widgets.preview_widget import PreviewWidget
from ..widgets.score_panel import ScorePanel
from ..widgets.emg_panel import EmgPanel
from ..widgets.debug_panel import DebugPanel
from ..theme import COLORS, PAGE_STYLE, pill_style, state_badge_style


ACTION_TIPS = {
    "M2": "跨步时保持躯干直立，脚尖朝前，动作稳定后再返回。",
    "M3": "前后脚保持稳定，屈膝时膝盖方向与脚尖一致。",
    "M4": "向侧方转移重心，另一侧腿伸直，避免躯干过度前倾。",
    "M5": "身体略向前倾后站起，双脚均匀发力，缓慢坐回。",
    "M6": "保持支撑腿稳定，抬腿时不要借助躯干摆动。",
    "M7": "手臂从身体两侧缓慢抬起，避免耸肩和快速下落。",
    "M8": "手臂向后伸展时保持躯干直立，不要向前代偿。",
    "M9": "肘部贴近身体完成旋转，动作幅度以舒适为准。",
    "M10": "沿肩胛平面缓慢上举，保持肩颈放松并自然呼吸。",
}


class TrainingState(Enum):
    IDLE = "待采集"
    STARTING_CAPTURE = "正在启动采集"
    CAPTURING = "采集中（未训练）"
    TRAINING = "训练中"
    RESTING = "休息中"
    PAUSED = "已暂停"
    STOPPING = "正在停止"
    FINISHED = "已完成"


class TrainingPage(QWidget):
    """Core training page with preview, scoring, and controls."""

    report_requested = pyqtSignal(str, str)  # session_dir, csv_path
    status_message = pyqtSignal(str)
    score_received = pyqtSignal(object)
    score_failed = pyqtSignal(str)
    action_changed_received = pyqtSignal(object)
    action_completed_received = pyqtSignal(object, int, float)
    rest_started_received = pyqtSignal(object, int)
    rest_tick_received = pyqtSignal(int)
    course_finished_received = pyqtSignal()
    runner_state_received = pyqtSignal(object)
    runtime_stopped = pyqtSignal(int, bool, str)
    score_start_finished = pyqtSignal(int, object, bool)
    shutdown_ready = pyqtSignal()
    pipeline_started = pyqtSignal(int, bool, str)
    offline_report_ready = pyqtSignal(int, str)
    offline_report_failed = pyqtSignal(int, str)

    def __init__(self, config: PipelineConfig, parent=None):
        super().__init__(parent)
        self._config = config
        self._state = TrainingState.IDLE
        self._elapsed_seconds = 0
        self._session_dir = ""
        self._frame_index = 0
        self._ending = False
        self._end_generation = 0
        self._pending_end = None
        self._shutdown_requested = False
        self._score_generation = 0
        self._start_generation = 0
        self._fps_warning_active = False
        self._paused_from = TrainingState.IDLE
        self._displayed_action_reps = 0
        self._scoring_submit_times = deque(maxlen=120)
        self._last_scoring_fps_update = 0.0
        self._calibrated_scoring_fps = 0.0
        self._session_start_time = ""
        self._current_action_dir = ""
        self._current_action_csv = ""
        self._action_summaries = []
        self._offline_report_runners = {}

        # Camera detection
        self._cameras_found: list = []
        self._camera_checked: bool = False

        # Pipeline modules
        self._pipeline = SensorPipeline(config)
        self._course_runner = CourseRunner()
        self._score_bridge: Optional[ScoreBridge] = None
        self._scoring_recorder = ScoringCsvRecorder()
        self._voice = VoiceAssistant(config.voice)
        self._voice.on_status = lambda message: logger.info(f"[Voice] {message}")
        self._voice.start()
        self._course_repo = CourseRepository()

        # Load course
        self._course_repo.load()
        courses = self._course_repo.courses
        self._current_course = (
            self._course_repo.find_by_id(config.selected_course_id)
            if config.selected_course_id else None
        )
        if self._current_course is None:
            self._current_course = courses[0] if courses else None

        self._init_ui()
        self._wire_signals()

        # Preview timer — 30fps refresh
        self._preview_timer = QTimer(self)
        self._preview_timer.timeout.connect(self._refresh_preview)
        self._preview_timer.start(33)

        # Training timer — 1Hz
        self._training_timer = QTimer(self)
        self._training_timer.timeout.connect(self._tick_training)

        self._update_state(TrainingState.IDLE)

        # ---- Startup diagnostic log ----
        self._log_startup_diagnostics()

    # ================================================================
    # Startup diagnostics
    # ================================================================

    def _log_startup_diagnostics(self):
        """Log system status to the feedback area at startup."""
        self._append_feedback("═══ 系统启动诊断 ═══")

        # Engine mode
        if rehab_engine._STUB_MODE:
            self._append_feedback("⚠ 引擎模式: STUB（模拟数据）— 未加载 C++ .so")
            self._preview.set_engine_mode("STUB")
        else:
            self._append_feedback("✓ 引擎模式: FULL（真实 C++ 引擎已加载）")
            self._preview.set_engine_mode("FULL")

        # Python
        import sys
        self._append_feedback(f"✓ Python {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}")
        conda_env = os.environ.get("CONDA_DEFAULT_ENV", "")
        if conda_env:
            self._append_feedback(f"✓ Conda 环境: {conda_env}")

        # Cameras
        self._check_cameras()
        if self._cameras_found:
            self._append_feedback(f"✓ 检测到 {len(self._cameras_found)} 个摄像头: {', '.join(self._cameras_found)}")
        else:
            self._append_feedback("✗ 未检测到摄像头设备 (/dev/video*)")

        # Config
        self._append_feedback(f"✓ 配置: RGB={self._config.device.rgb_device_path} "
                             f"{self._config.device.rgb_width}x{self._config.device.rgb_height} "
                             f"@{self._config.device.rgb_fps}fps")
        self._append_feedback(f"✓ 配置: Depth={self._config.device.depth_width}x{self._config.device.depth_height}")
        self._append_feedback(
            f"✓ 配置: EMG={'启用（真实采集）' if self._config.emg.enabled else '禁用'}")

        self._append_feedback("════════════════════")

    # ================================================================
    # Camera detection
    # ================================================================

    def _check_cameras(self):
        """Detect available V4L2 cameras."""
        self._cameras_found = []
        for idx in range(4):
            dev = f"/dev/video{idx}"
            if os.path.exists(dev):
                self._cameras_found.append(dev)
        self._camera_checked = True

    def camera_status(self) -> str:
        """Return a human-readable camera status string."""
        if not self._camera_checked:
            self._check_cameras()

        if self._pipeline.stub_mode:
            return "📷 模拟画面（非真实相机数据）"
        if self._pipeline.is_stopping:
            return "📷 正在释放 RGB/Depth 设备"
        if self._pipeline.is_running:
            stats = self._pipeline.performance_stats()
            marker = "✓" if (stats["rgb_30fps_ok"] and stats["depth_30fps_ok"]
                               and stats["pair_30fps_ok"]) else "⚠"
            return (f"{marker} 真实数据 RGB {stats['rgb_fps']:.1f}/30 | "
                    f"Depth {stats['depth_fps']:.1f}/30 | "
                    f"显示 {stats['pair_fps']:.1f}/30 fps")
        status = self._pipeline.camera_status
        return f"📷 真实相机 {status['status']}"

    # ---- Properties ----

    @property
    def is_training(self) -> bool:
        return self._state in (TrainingState.TRAINING, TrainingState.RESTING)

    def pipeline_stats(self) -> dict:
        return self._pipeline.performance_stats()

    def set_course(self, course_id: str) -> bool:
        """Apply a settings-page course selection when no session is active."""
        course = self._course_repo.find_by_id(course_id)
        if course is None:
            return False
        if self._state in (TrainingState.TRAINING, TrainingState.RESTING, TrainingState.PAUSED):
            self._append_feedback("课程设置已保存，将在当前训练结束后生效。")
            return False
        self._current_course = course
        self._course_label.setText(course.course_name)
        self._action_label.setText(f"共 {len(course.actions)} 个动作 · 预计 {course.estimated_minutes} 分钟")
        self._append_feedback(f"已选择课程：{course.course_name}")
        return True

    def set_debug_enabled(self, enabled: bool):
        self._preview.set_show_debug(enabled)

    def _capture_preflight_checks(self):
        """Validate only resources required for capture and pose inference."""
        errors, warnings = [], []
        self._check_cameras()
        if not rehab_engine._STUB_MODE and not self._cameras_found:
            errors.append("未检测到 RGB 摄像头")
        elif rehab_engine._STUB_MODE:
            warnings.append("当前为 STUB 模拟模式")
        if (not rehab_engine._STUB_MODE and
                (self._config.device.rgb_fps != 30 or self._config.device.depth_fps != 30)):
            errors.append("真实 RGB 与 Depth 相机必须同时设置为 30 FPS")
        return errors, warnings

    def _training_preflight_checks(self):
        """Validate resources first needed when a formal session begins."""
        errors = []
        if not self._current_course or not self._current_course.actions:
            errors.append("未选择有效训练课程")
        try:
            save_root = self._record_sessions_dir()
            save_root.mkdir(parents=True, exist_ok=True)
            probe = save_root / ".write_test"
            probe.write_text("ok", encoding="utf-8")
            probe.unlink()
        except OSError as exc:
            errors.append(f"训练记录目录不可写：{exc}")
        return errors

    def _record_sessions_dir(self) -> Path:
        """Resolve relative recording paths against the Python project root."""
        record_root = Path(self._config.record_path).expanduser()
        if not record_root.is_absolute():
            record_root = Path(__file__).resolve().parents[2] / record_root
        return record_root / "sessions"

    # ---- UI Construction ----

    def _init_ui(self):
        self.setStyleSheet(PAGE_STYLE)
        root = QVBoxLayout(self)
        root.setContentsMargins(18, 12, 18, 12)
        root.setSpacing(10)

        # --- Training summary / hero ---
        top_card = CardWidget(self)
        top_layout = QHBoxLayout(top_card)
        top_layout.setContentsMargins(20, 10, 20, 10)
        top_layout.setSpacing(20)

        title_box = QVBoxLayout()
        title_box.setSpacing(2)
        eyebrow = QLabel("ACTIVE REHABILITATION")
        eyebrow.setObjectName("pageEyebrow")
        title_box.addWidget(eyebrow)

        self._course_label = TitleLabel(
            self._current_course.course_name if self._current_course
            else "未选择课程")
        self._course_label.setStyleSheet(
            f"color:{COLORS['ink']}; font-size:21px; font-weight:700;")
        self._action_label = BodyLabel("当前动作：—")
        self._action_label.setStyleSheet(
            f"color:{COLORS['muted']}; font-size:12px;")
        title_box.addWidget(self._course_label)
        title_box.addWidget(self._action_label)

        self._state_badge = CaptionLabel("待开始")
        self._state_badge.setObjectName("stateBadge")
        self._state_badge.setAlignment(Qt.AlignCenter)
        self._state_badge.setStyleSheet(state_badge_style("IDLE"))

        timer_box = QVBoxLayout()
        timer_box.setSpacing(0)
        timer_caption = CaptionLabel("本次训练时长")
        timer_caption.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        timer_caption.setStyleSheet(f"color:{COLORS['muted']};")
        self._timer_label = SubtitleLabel("00:00")
        self._timer_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self._timer_label.setStyleSheet(
            f"color:{COLORS['primary']}; font-size:30px; font-weight:700;")
        timer_box.addWidget(timer_caption)
        timer_box.addWidget(self._timer_label)

        top_layout.addLayout(title_box, 1)
        top_layout.addWidget(self._state_badge)
        top_layout.addLayout(timer_box)
        root.addWidget(top_card)

        # --- Main row: preview + side panel ---
        main_row = QHBoxLayout()
        main_row.setSpacing(12)

        # -- Preview area --
        preview_card = CardWidget(self)
        preview_layout = QVBoxLayout(preview_card)
        preview_layout.setContentsMargins(14, 12, 14, 14)
        preview_layout.setSpacing(10)

        tag_row = QHBoxLayout()
        preview_titles = QVBoxLayout()
        preview_titles.setSpacing(1)
        preview_title = QLabel("实时动作捕捉")
        preview_title.setObjectName("sectionTitle")
        preview_hint = QLabel("保持全身位于画面中央，系统将同步追踪骨骼与训练质量")
        preview_hint.setObjectName("sectionHint")
        preview_titles.addWidget(preview_title)
        preview_titles.addWidget(preview_hint)
        tag_row.addLayout(preview_titles)
        tag_row.addStretch()

        self._preview_mode_buttons = {}
        for mode, tag in [("rgb", "RGB"), ("depth", "深度"), ("skeleton", "骨骼")]:
            button = PushButton(tag, self)
            button.setCheckable(True)
            button.setFixedHeight(30)
            button.clicked.connect(
                lambda checked=False, selected=mode: self._set_preview_mode(selected))
            self._preview_mode_buttons[mode] = button
            tag_row.addWidget(button)
        preview_layout.addLayout(tag_row)

        self._preview = PreviewWidget(self)
        self._preview.setMinimumSize(680, 360)
        self._preview.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self._set_preview_mode("skeleton")
        preview_layout.addWidget(self._preview, 1)

        # -- Side panel --
        side = QWidget(self)
        side.setFixedWidth(336)
        side_layout = QVBoxLayout(side)
        side_layout.setContentsMargins(0, 0, 0, 0)
        side_layout.setSpacing(12)

        # Course info card
        info_card = SimpleCardWidget(side)
        info_layout = QVBoxLayout(info_card)
        info_layout.setContentsMargins(16, 12, 16, 12)
        info_layout.setSpacing(6)
        info_title = QLabel("训练进程")
        info_title.setObjectName("sectionTitle")
        info_layout.addWidget(info_title)
        self._info_action = BodyLabel("等待开始训练")
        self._info_action.setWordWrap(True)
        self._info_action.setStyleSheet(
            f"color:{COLORS['ink']}; font-size:17px; font-weight:700;")
        self._instruction_label = CaptionLabel("选择动作后显示训练要点")
        self._instruction_label.setWordWrap(True)
        self._instruction_label.setStyleSheet(
            f"color:{COLORS['muted']}; line-height:1.5;")

        metric_row = QHBoxLayout()
        metric_row.setSpacing(8)
        target_box = QVBoxLayout()
        target_box.addWidget(QLabel("本动作目标", objectName="metricLabel"))
        self._info_target = QLabel("— 次")
        self._info_target.setObjectName("metricValue")
        target_box.addWidget(self._info_target)
        progress_box = QVBoxLayout()
        progress_box.addWidget(QLabel("课程进度", objectName="metricLabel"))
        self._info_progress = QLabel("— / —")
        self._info_progress.setObjectName("metricValue")
        progress_box.addWidget(self._info_progress)
        metric_row.addLayout(target_box, 1)
        metric_row.addLayout(progress_box, 1)

        self._rest_label = BodyLabel("休息：—")
        self._rest_label.setStyleSheet(pill_style("warning"))
        info_layout.addWidget(self._info_action)
        info_layout.addWidget(self._instruction_label)
        info_layout.addLayout(metric_row)
        info_layout.addWidget(self._rest_label)
        side_layout.addWidget(info_card)

        # Score panel
        self._score_panel = ScorePanel(side)
        side_layout.addWidget(self._score_panel)

        # EMG panel
        self._emg_panel = EmgPanel(side)
        side_layout.addWidget(self._emg_panel)

        # Debug panel (collapsible via toggle button on control bar)
        self._debug_panel = DebugPanel(side)
        self._debug_panel.setVisible(False)
        side_layout.addWidget(self._debug_panel)

        side_layout.addStretch()

        main_row.addWidget(preview_card, 1)
        main_row.addWidget(side)
        root.addLayout(main_row, 1)

        # --- Feedback card ---
        feedback_card = SimpleCardWidget(self)
        feedback_layout = QVBoxLayout(feedback_card)
        feedback_layout.setContentsMargins(12, 8, 12, 9)
        feedback_layout.setSpacing(4)
        feedback_header = QHBoxLayout()
        feedback_title = QLabel("训练反馈")
        feedback_title.setObjectName("sectionTitle")
        feedback_hint = QLabel("实时状态与动作建议")
        feedback_hint.setObjectName("sectionHint")
        feedback_header.addWidget(feedback_title)
        feedback_header.addSpacing(8)
        feedback_header.addWidget(feedback_hint)
        feedback_header.addStretch()
        feedback_layout.addLayout(feedback_header)
        self._feedback = TextEdit(self)
        self._feedback.setObjectName("feedbackLog")
        self._feedback.setReadOnly(True)
        self._feedback.setPlaceholderText("训练反馈将显示在这里…")
        self._feedback.setMaximumHeight(68)
        feedback_layout.addWidget(self._feedback)
        root.addWidget(feedback_card)

        # --- Control bar ---
        ctrl_card = CardWidget(self)
        ctrl = QHBoxLayout(ctrl_card)
        ctrl.setContentsMargins(16, 8, 16, 8)
        ctrl.setSpacing(10)

        self._btn_capture = PrimaryPushButton("开始采集")
        self._btn_start = PrimaryPushButton("开始训练")
        self._btn_pause = PushButton("暂停")
        self._btn_stop = PushButton("停止")
        self._btn_report = PrimaryPushButton("结束并生成报告")
        self._btn_open_report = PushButton("查看报告")
        self._btn_debug = PushButton("调试面板")
        self._btn_debug.setCheckable(True)
        self._btn_debug.setFixedHeight(30)

        for button in [self._btn_capture, self._btn_start, self._btn_pause, self._btn_stop,
                       self._btn_report, self._btn_open_report]:
            button.setMinimumHeight(36)
        self._btn_start.setMinimumWidth(112)
        self._btn_report.setMinimumWidth(146)
        self._btn_stop.setStyleSheet(
            f"PushButton{{color:{COLORS['danger']};}}")

        self._btn_capture.clicked.connect(self._on_start_capture)
        self._btn_start.clicked.connect(self._on_start)
        self._btn_pause.clicked.connect(self._on_pause)
        self._btn_stop.clicked.connect(self._on_stop)
        self._btn_report.clicked.connect(self._on_finish)
        self._btn_open_report.clicked.connect(self._on_open_report)
        self._btn_debug.clicked.connect(self._on_toggle_debug)

        ctrl.addWidget(self._btn_capture)
        ctrl.addWidget(self._btn_start)
        ctrl.addWidget(self._btn_pause)
        ctrl.addWidget(self._btn_stop)
        ctrl.addStretch()
        ctrl.addWidget(self._btn_debug)
        ctrl.addWidget(self._btn_report)
        ctrl.addWidget(self._btn_open_report)
        root.addWidget(ctrl_card)

    # ---- Course runner signals ----

    def _wire_signals(self):
        # CourseRunner and ScoreBridge may invoke callbacks from worker threads.
        # Relay every UI mutation through queued Qt signals.
        self._course_runner.on_action_changed = self.action_changed_received.emit
        self._course_runner.on_action_completed = self.action_completed_received.emit
        self._course_runner.on_rest_started = self.rest_started_received.emit
        self._course_runner.on_rest_tick = self.rest_tick_received.emit
        self._course_runner.on_course_finished = self.course_finished_received.emit
        self._course_runner.on_state_changed = self.runner_state_received.emit
        self.action_changed_received.connect(self._on_action_changed)
        self.action_completed_received.connect(self._on_action_completed)
        self.rest_started_received.connect(self._on_rest_started)
        self.rest_tick_received.connect(self._on_rest_tick)
        self.course_finished_received.connect(self._on_course_finished)
        self.runner_state_received.connect(self._on_runner_state)
        self.score_received.connect(self._on_score)
        self.score_failed.connect(self._on_score_error)
        self.runtime_stopped.connect(self._on_runtime_stopped)
        self.score_start_finished.connect(self._on_score_start_finished)
        self.pipeline_started.connect(self._on_pipeline_started)
        self.offline_report_ready.connect(self._on_offline_report_ready)
        self.offline_report_failed.connect(self._on_offline_report_failed)
        self._pipeline.set_on_frame(self._on_pipeline_frame)

    # ---- State machine ----

    def _update_state(self, state: TrainingState):
        self._state = state
        self._state_badge.setText(state.value)
        self._state_badge.setStyleSheet(state_badge_style(state.name))
        self._update_buttons()

    def _update_buttons(self):
        s = self._state
        self._btn_capture.setEnabled(s in (TrainingState.IDLE, TrainingState.FINISHED))
        self._btn_start.setEnabled(
            s == TrainingState.CAPTURING and self._pipeline.is_running)
        self._btn_start.setText("开始训练")
        self._btn_pause.setEnabled(
            s in (TrainingState.TRAINING, TrainingState.RESTING, TrainingState.PAUSED))
        self._btn_pause.setText("继续训练" if s == TrainingState.PAUSED else "暂停")
        self._btn_stop.setEnabled(s in (
            TrainingState.CAPTURING, TrainingState.TRAINING,
            TrainingState.RESTING, TrainingState.PAUSED))
        self._btn_report.setEnabled(
            s in (TrainingState.TRAINING, TrainingState.RESTING, TrainingState.PAUSED))
        self._btn_open_report.setEnabled(bool(self._session_dir))

    # ---- Button handlers ----

    def _set_preview_mode(self, mode: str):
        self._preview.set_view_mode(mode)
        for name, button in self._preview_mode_buttons.items():
            button.setChecked(name == mode)

    def _on_start_capture(self):
        if self._state not in (TrainingState.IDLE, TrainingState.FINISHED):
            self._append_feedback("采集已在启动或运行中，请勿重复点击。")
            return
        errors, warnings = self._capture_preflight_checks()
        if errors:
            message = "；".join(errors)
            self._append_feedback(f"采集前检查失败：{message}")
            InfoBar.error("无法开始采集", message, duration=6000,
                          position=InfoBarPosition.TOP_RIGHT, parent=self)
            return
        if warnings:
            self._append_feedback("训练前提示：" + "；".join(warnings))

        # Log pipeline start details
        self._append_feedback("═══ 启动采集 Pipeline ═══")
        self._append_feedback(f"引擎模式: {'STUB (模拟)' if rehab_engine._STUB_MODE else 'FULL (真实)'}")
        self._append_feedback(f"RGB 设备: {self._config.device.rgb_device_path}")
        self._append_feedback(f"分辨率: {self._config.device.rgb_width}x{self._config.device.rgb_height} @ {self._config.device.rgb_fps}fps")
        self._append_feedback(
            f"EMG: {'启用（真实采集）' if self._config.emg.enabled else '禁用'}")

        self._append_feedback("正在启动摄像头与骨骼推理预览…")

        self._ending = False
        self._fps_warning_active = False
        self._frame_index = 0
        self._start_generation += 1
        generation = self._start_generation
        self._update_state(TrainingState.STARTING_CAPTURE)
        self._append_feedback("正在后台连接真实 RGB/Depth 设备并初始化模型…")

        def _start_pipeline():
            try:
                ok = self._pipeline.start()
                message = "" if ok else (
                    self._pipeline.camera_status.get("error") or "传感器流水线未能启动")
            except Exception as exc:
                ok, message = False, str(exc)
            try:
                self.pipeline_started.emit(generation, ok, message)
            except RuntimeError:
                pass

        threading.Thread(
            target=_start_pipeline, name="pipeline-start", daemon=True).start()

    def _on_pipeline_started(self, generation: int, ok: bool, message: str):
        if generation != self._start_generation:
            if ok:
                self._pipeline.stop()
            return
        if self._state != TrainingState.STARTING_CAPTURE:
            if ok:
                self._pipeline.stop()
            return
        if not ok:
            self._update_state(TrainingState.IDLE)
            self._append_feedback(f"Pipeline 启动失败：{message}")
            InfoBar.error("启动失败", message,
                          position=InfoBarPosition.TOP_RIGHT, parent=self)
            return
        self._preview.set_recording(False)
        self._preview.set_show_debug(self._config.ui_debug_enabled)
        self._update_state(TrainingState.CAPTURING)
        self._append_feedback(
            "采集已启动，请确认摄像头和骨骼显示正常后点击“开始训练”。")
        self._voice.speak(
            "采集已启动，请确认骨骼显示后开始训练",
            key="capture_ready", priority=2, force=True)

    def _on_start(self):
        if self._state != TrainingState.CAPTURING:
            self._append_feedback("请先点击“开始采集”并等待摄像头与骨骼就绪。")
            return
        if not self._pipeline.is_running:
            self._append_feedback("采集 Pipeline 已停止，请重新开始采集。")
            self._update_state(TrainingState.IDLE)
            return
        errors = self._training_preflight_checks()
        if errors:
            message = "；".join(errors)
            self._append_feedback(f"训练前检查失败：{message}")
            InfoBar.error("无法开始训练", message, duration=6000,
                          position=InfoBarPosition.TOP_RIGHT, parent=self)
            return
        self._ending = False
        self._fps_warning_active = False
        self._frame_index = 0
        self._score_panel.reset()
        self._append_feedback(f"启动正式训练：{self._current_course.course_name}")
        try:
            self._session_dir = self._pipeline.start_recording(
                str(self._record_sessions_dir()))
        except OSError as exc:
            self._append_feedback(f"录制启动失败：{exc}")
            InfoBar.error("录制启动失败", str(exc),
                          position=InfoBarPosition.TOP_RIGHT, parent=self)
            self._end_session(
                TrainingState.IDLE, generate_report=False,
                reason="录制启动失败，设备已安全停止。")
            return
        self._session_start_time = datetime.now().isoformat(timespec="seconds")
        self._current_action_dir = ""
        self._current_action_csv = ""
        self._action_summaries = [
            {
                "action_id": action.action_id,
                "movement_id": action.movement_id,
                "name_cn": action.name_cn,
                "target_reps": action.target_reps,
                "actual_reps": 0,
                "average_score": 0.0,
                "action_dir": "",
                "csv_path": "",
                "report_path": "",
            }
            for action in self._current_course.actions
        ]
        self._write_course_summary(False)
        self._preview.set_recording(True)
        self._preview.set_show_debug(self._config.ui_debug_enabled)

        self._elapsed_seconds = 0

        if not self._course_runner.start_course(self._current_course):
            self._append_feedback("课程不包含有效动作，训练已取消。")
            self._end_session(
                TrainingState.IDLE, generate_report=False,
                reason="课程无有效动作，设备已安全停止。")
            return
        self._training_timer.start(1000)
        self._update_state(TrainingState.TRAINING)
        self._voice.speak(
            f"训练开始，课程是{self._current_course.course_name}",
            key="session_start", priority=2, force=True)

    def _on_pause(self):
        if self._state == TrainingState.PAUSED:
            if self._course_runner.resume_course():
                self._pipeline.resume_recording()
                self._training_timer.start(1000)
                resume_state = (
                    TrainingState.RESTING
                    if self._course_runner.state == RunnerState.RESTING
                    else TrainingState.TRAINING
                )
                self._update_state(resume_state)
                self._preview.set_recording(True)
                self._append_feedback("训练已继续。")
                self._voice.speak("训练继续", key="resume", priority=3, force=True)
            return

        if self._state not in (TrainingState.TRAINING, TrainingState.RESTING):
            return
        self._paused_from = self._state
        if self._course_runner.pause_course():
            self._training_timer.stop()
            self._pipeline.pause_recording()
            self._preview.set_recording(False)
            self._update_state(TrainingState.PAUSED)
            self._append_feedback("训练已暂停，采集画面保留但数据不计入训练。")
            self._voice.speak("训练已暂停", key="pause", priority=2, force=True)

    def _on_stop(self):
        if self._state in (TrainingState.TRAINING, TrainingState.RESTING, TrainingState.PAUSED):
            box = MessageBox(
                "停止当前训练？",
                "停止后会保存已采集的骨骼数据，但本次课程将标记为未完整完成。",
                self,
            )
            if not box.exec():
                return
        reason = ("采集已停止。" if self._state == TrainingState.CAPTURING
                  else "训练已停止。")
        self._end_session(TrainingState.IDLE, generate_report=False, reason=reason)

    def _on_finish(self):
        self._end_session(
            TrainingState.FINISHED, generate_report=True,
            reason="训练完成，正在生成报告…")

    def _on_open_report(self):
        if self._session_dir:
            csv_path = str(Path(self._session_dir) / "skeleton_3d.csv")
            self.report_requested.emit(self._session_dir, csv_path)

    def _end_session(self, final_state: TrainingState,
                     generate_report: bool, reason: str):
        """Begin an ordered, non-blocking shutdown of the active runtime."""
        if self._ending:
            return
        self._ending = True
        self._end_generation += 1
        generation = self._end_generation
        self._pending_end = (final_state, generate_report, reason)
        self._training_timer.stop()
        self._update_state(TrainingState.STOPPING)
        self._append_feedback("正在安全停止采集、评分和录制，请稍候…")
        try:
            self._course_runner.stop_course()
        except Exception as exc:
            self._append_feedback(f"课程计时器停止异常：{exc}")
        self._stop_current_action_artifacts()
        self._score_generation += 1
        if self._score_bridge:
            try:
                self._score_bridge.stop()
            except Exception as exc:
                self._append_feedback(f"评分进程停止异常：{exc}")
            self._score_bridge = None
        self._preview.set_recording(False)
        self._pipeline.stop(
            on_complete=lambda ok, msg: self.runtime_stopped.emit(
                generation, ok, msg))

    def _on_runtime_stopped(self, generation: int, success: bool, message: str):
        if generation != self._end_generation or not self._pending_end:
            return
        final_state, generate_report, reason = self._pending_end
        self._pending_end = None
        self._write_session_metadata(final_state)
        self._update_state(final_state)
        self._append_feedback(reason)
        if not self._shutdown_requested:
            self._voice.clear()
            self._voice.speak(
                reason.replace("…", ""), key="session_end", priority=1, force=True)
        self._log_stop_stats()
        if not success:
            self._append_feedback(f"后台停止不完整：{message}")
            InfoBar.error(
                "设备停止异常", message + "。为避免设备冲突，请重启应用后再训练。",
                duration=8000, position=InfoBarPosition.TOP_RIGHT, parent=self)

        self._ending = False
        if generate_report and self._session_dir:
            csv_path = str(Path(self._session_dir) / "skeleton_3d.csv")
            # Signal emission is cheap and remains on the Qt thread. The report
            # page performs file work in its own worker and returns via signal.
            self.report_requested.emit(self._session_dir, csv_path)

        if self._shutdown_requested:
            self.shutdown_ready.emit()

    def _write_session_metadata(self, final_state: TrainingState):
        if not self._session_dir:
            return
        metadata = {
            "patient_id": self._config.patient_id,
            "patient_name": self._config.patient_name,
            "patient_gender": self._config.patient_gender,
            "patient_age": self._config.patient_age,
            "patient_diagnosis": self._config.patient_diagnosis,
            "course_id": self._current_course.course_id if self._current_course else "",
            "course_name": self._current_course.course_name if self._current_course else "",
            "start_time": self._session_start_time,
            "elapsed_seconds": self._elapsed_seconds,
            "finished": final_state == TrainingState.FINISHED,
            "end_time": datetime.now().isoformat(timespec="seconds"),
            "engine_mode": "stub" if rehab_engine._STUB_MODE else "full",
            "pipeline_csv_path": str(Path(self._session_dir) / "skeleton_3d.csv"),
            "course_summary_path": str(Path(self._session_dir) / "course_summary.json"),
        }
        try:
            path = Path(self._session_dir) / "session_ui_meta.json"
            path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2),
                            encoding="utf-8")
        except OSError as exc:
            self._append_feedback(f"会话元数据保存失败：{exc}")
        self._write_course_summary(final_state == TrainingState.FINISHED)

    def _write_course_summary(self, finished: bool):
        if not self._session_dir or not self._current_course:
            return
        payload = {
            "patient_id": self._config.patient_id,
            "patient_name": self._config.patient_name,
            "course_id": self._current_course.course_id,
            "course_name": self._current_course.course_name,
            "category": self._current_course.category,
            "difficulty": self._current_course.difficulty,
            "estimated_minutes": self._current_course.estimated_minutes,
            "description": self._current_course.description,
            "start_time": self._session_start_time,
            "end_time": datetime.now().isoformat(timespec="seconds") if finished else "",
            "status": "finished" if finished else self._state.value,
            "session_dir": str(Path(self._session_dir).resolve()),
            "actions": self._action_summaries,
        }
        output = Path(self._session_dir) / "course_summary.json"
        temporary = output.with_suffix(".json.tmp")
        try:
            temporary.write_text(
                json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
            temporary.replace(output)
        except OSError as exc:
            self._append_feedback(f"课程摘要保存失败：{exc}")

    # ---- Stop stats ----

    def _log_stop_stats(self):
        """Log pipeline statistics when stopping."""
        stats = self._pipeline.performance_stats()
        self._append_feedback(f"─── 训练统计 ───")
        self._append_feedback(f"已处理帧: {stats['processed']}")
        self._append_feedback(f"丢弃帧: {stats['dropped_pairs']}")
        self._append_feedback(f"队列长度: {stats['queue_length']}")

    # ---- Course callbacks ----

    def _on_action_changed(self, action):
        self._displayed_action_reps = 0
        self._scoring_submit_times.clear()
        self._last_scoring_fps_update = 0.0
        self._calibrated_scoring_fps = 0.0
        self._action_label.setText(f"当前动作：{action.name_cn} ({action.action_id})")
        self._info_action.setText(f"{action.name_cn}\n{action.action_id}")
        self._instruction_label.setText(
            ACTION_TIPS.get(action.action_id, "请在舒适范围内缓慢完成动作，保持自然呼吸。"))
        self._info_target.setText(f"{action.target_reps} 次")
        self._info_progress.setText(
            f"{self._course_runner.current_action_index + 1} / {self._course_runner.total_actions}")
        self._score_panel.set_target(action.target_reps)
        self._preview.set_training_progress(0, action.target_reps, "准备开始动作")
        self._rest_label.setText("休息：—")
        instruction = ACTION_TIPS.get(
            action.action_id, "请在舒适范围内缓慢完成动作，保持自然呼吸。")
        self._voice.speak(
            f"下一动作，{action.name_cn}。目标{action.target_reps}次。{instruction}",
            key=f"action_{action.order}", priority=4, force=True)

        movement = action.movement_id or "movement"
        action_dir = Path(self._session_dir) / "actions" / (
            f"{action.order:02d}_{action.action_id}_{movement}")
        action_dir.mkdir(parents=True, exist_ok=True)
        self._current_action_dir = str(action_dir.resolve())
        self._current_action_csv = str((action_dir / "skeleton3d.csv").resolve())
        if not self._scoring_recorder.start(self._current_action_dir):
            self._append_feedback(f"动作 CSV 启动失败：{self._current_action_dir}")
        if not self._pipeline.start_action_recording(self._current_action_dir):
            self._append_feedback(f"动作 EMG 录制启动失败：{self._current_action_dir}")
        summary = self._summary_for_action(action.order)
        if summary is not None:
            summary["action_dir"] = self._current_action_dir
            summary["csv_path"] = self._current_action_csv
        self._write_course_summary(False)

        # Start scoring for this action
        if self._pipeline.is_running:
            fps = self._skeleton_fps()
            if self._score_bridge:
                self._score_bridge.stop()
            bridge = ScoreBridge()
            bridge.on_score_updated = self.score_received.emit
            bridge.on_error = self.score_failed.emit
            bridge.on_debug_state = self._on_debug_state_received
            self._score_bridge = bridge
            self._score_generation += 1
            generation = self._score_generation
            self._append_feedback("正在后台启动实时评分…")

            # Reset debug panel data for the new action
            self._debug_panel.reset()

            def _start_scoring():
                ok = bridge.start(action.action_id, fps)
                try:
                    self.score_start_finished.emit(generation, bridge, ok)
                except RuntimeError:
                    bridge.stop()

            threading.Thread(
                target=_start_scoring,
                name=f"score-start-{action.action_id}", daemon=True,
            ).start()

    def _on_score_start_finished(self, generation: int, bridge: ScoreBridge, ok: bool):
        if (generation != self._score_generation or bridge is not self._score_bridge
                or self._state not in (TrainingState.TRAINING, TrainingState.RESTING,
                                       TrainingState.PAUSED)):
            bridge.stop()
            return
        if ok:
            self._append_feedback("实时评分已启动。")
        else:
            self._append_feedback("实时评分不可用，训练录制仍将继续。")

    def _on_action_completed(self, action, actual_reps, avg_score):
        action_dir = self._current_action_dir
        csv_path = self._current_action_csv
        self._stop_current_action_artifacts()
        self._score_generation += 1
        if self._score_bridge:
            self._score_bridge.stop()
            self._score_bridge = None
        summary = self._summary_for_action(action.order)
        if summary is not None:
            summary.update({
                "actual_reps": actual_reps,
                "average_score": avg_score,
                "action_dir": action_dir,
                "csv_path": csv_path,
            })
        self._append_feedback(
            f"完成 {action.name_cn}：{actual_reps}/{action.target_reps} 次，"
            f"平均评分 {avg_score:.1f}")
        self._voice.speak(
            f"{action.name_cn}完成，平均评分{avg_score:.0f}分",
            key=f"action_complete_{action.order}", priority=3, force=True)
        if action_dir and csv_path and Path(csv_path).exists():
            report_dir = Path(action_dir) / "report"
            report_dir.mkdir(parents=True, exist_ok=True)
            expected = str((report_dir / "offline_action_report.html").resolve())
            if summary is not None:
                summary["report_path"] = expected
            runner = OfflineReportRunner()
            self._offline_report_runners[action.order] = runner
            runner.on_ready = lambda path, order=action.order: (
                self.offline_report_ready.emit(order, path))
            runner.on_error = lambda message, order=action.order: (
                self.offline_report_failed.emit(order, message))
            if runner.run(csv_path, action.action_id, str(report_dir), self._skeleton_fps()):
                self._append_feedback(f"动作 {action.action_id} 已完成，正在生成离线报告。")
            else:
                self._offline_report_runners.pop(action.order, None)
        self._write_course_summary(False)

    def _on_offline_report_ready(self, order: int, html_path: str):
        self._offline_report_runners.pop(order, None)
        summary = self._summary_for_action(order)
        if summary is not None:
            summary["report_path"] = html_path
        self._write_course_summary(self._state == TrainingState.FINISHED)
        self._append_feedback(f"离线动作报告已生成：{html_path}")

    def _on_offline_report_failed(self, order: int, message: str):
        self._offline_report_runners.pop(order, None)
        self._append_feedback(f"动作离线报告失败：{message}")

    def _summary_for_action(self, order: int):
        index = int(order) - 1
        return self._action_summaries[index] if 0 <= index < len(self._action_summaries) else None

    def _stop_current_action_artifacts(self):
        self._scoring_recorder.stop()
        try:
            self._pipeline.stop_action_recording()
        except Exception as exc:
            self._append_feedback(f"动作 EMG 录制停止异常：{exc}")

    def _skeleton_fps(self) -> float:
        actual = float(self._pipeline.performance_stats().get("worker_fps", 0.0))
        if actual >= 0.5:
            return actual
        return self._config.device.rgb_fps / max(1, self._config.pose.pose_interval)

    def _on_rest_started(self, action, rest_sec):
        self._update_state(TrainingState.RESTING)
        self._rest_label.setText(f"休息：{rest_sec} 秒")
        next_index = self._course_runner.current_action_index + 1
        if self._current_course and next_index < len(self._current_course.actions):
            next_action = self._current_course.actions[next_index]
            self._instruction_label.setText(f"下一动作：{next_action.name_cn}。请调整站位并准备。")
        self._append_feedback(f"动作完成，休息 {rest_sec} 秒…")
        self._voice.speak(
            f"动作完成，请休息{rest_sec}秒",
            key=f"rest_{action.order}", priority=3, force=True)

    def _on_rest_tick(self, remaining):
        self._rest_label.setText(f"休息：{remaining} 秒")
        if remaining in (10, 5, 3, 2, 1):
            self._voice.speak(
                f"还有{remaining}秒", key=f"rest_tick_{remaining}", priority=8)

    def _on_course_finished(self):
        self._append_feedback("全部动作已完成！")
        self._voice.speak("全部动作已完成", key="course_finished", priority=1, force=True)
        self._end_session(
            TrainingState.FINISHED, generate_report=True,
            reason="课程已自动完成，正在生成报告…")

    def _on_runner_state(self, state: RunnerState):
        if state == RunnerState.TRAINING:
            self._update_state(TrainingState.TRAINING)

    def _on_score(self, result: ScoreResult):
        self._score_panel.set_score(result)
        reported = max(result.count, result.completed_count)
        if reported > self._displayed_action_reps + 1:
            self._displayed_action_reps += 1
        else:
            self._displayed_action_reps = max(self._displayed_action_reps, reported)
        count = self._displayed_action_reps
        self._score_panel.set_display_count(count)
        quality = (
            "动作质量：优秀" if result.overall_score >= 85 else
            "动作质量：良好" if result.overall_score >= 70 else
            "动作质量：请放慢并保持稳定" if result.overall_score > 0 else
            "正在识别动作"
        )
        action = self._course_runner.current_action
        target = action.target_reps if action else 0
        self._preview.set_training_progress(count, target, quality)
        if self._course_runner:
            self._course_runner.on_score_updated(result)
        if result.status == "new_completed_cycle":
            self._voice.speak(
                f"完成第{count}次，评分{result.overall_score:.0f}分",
                key=f"score_cycle_{count}", priority=7, force=True)
        # Also feed FPS info to debug panel
        if self._debug_panel.isVisible() and self._calibrated_scoring_fps > 0:
            self._debug_panel.set_fps_info(
                self._calibrated_scoring_fps, self._skeleton_fps())

    def _on_score_error(self, message: str):
        self._append_feedback(f"评分提示：{message}")

    def _on_toggle_debug(self):
        """Toggle the debug panel visibility and connect refresh callback."""
        visible = not self._debug_panel.isVisible()
        self._debug_panel.setVisible(visible)
        self._btn_debug.setChecked(visible)
        if visible:
            self._debug_panel.set_refresh_callback(self._on_debug_refresh)
            self._debug_panel.request_refresh()

    def _on_debug_refresh(self):
        """Callback from DebugPanel when user clicks 'refresh state'."""
        bridge = self._score_bridge
        if bridge is not None and bridge._running:
            bridge.on_debug_state = self._on_debug_state_received
            bridge.request_debug_state()
        else:
            self._append_feedback("评分服务未运行，无法获取调试状态")

    def _on_debug_state_received(self, debug_state: dict):
        """Receive debug state from ScoreBridge and forward to panel."""
        # debug_state comes from RealtimeJointActionScorer.get_debug_state() JSON
        self._debug_panel.set_debug_state(debug_state)

    def _on_pipeline_frame(self, frame: PreviewFrame):
        """Runs on the pipeline worker; submit only valid active-training frames."""
        bridge = self._score_bridge
        if (self._state != TrainingState.TRAINING
                or not frame.depth_is_hardware
                or not frame.has_valid_3d or len(frame.joints_3d) < 22):
            return
        self._frame_index += 1
        joints = frame.joints_3d[:22]
        if ScoringSkeletonAdapter.valid_joint_count(joints) >= 5:
            self._scoring_recorder.append(
                self._frame_index, ScoringSkeletonAdapter.convert(joints))
        # ScoreBridge owns the original Rehab22 -> P-Coder coordinate transform,
        # validity gate and synthetic joint fallbacks.
        if bridge is not None:
            now_ns = time.monotonic_ns()
            if bridge.submit_skeleton(self._frame_index, now_ns, joints):
                now = now_ns / 1_000_000_000.0
                self._scoring_submit_times.append(now)
                if (len(self._scoring_submit_times) >= 10
                        and now - self._last_scoring_fps_update >= 2.0):
                    elapsed = now - self._scoring_submit_times[0]
                    actual_fps = ((len(self._scoring_submit_times) - 1) / elapsed
                                  if elapsed > 0.0 else 0.0)
                    if (actual_fps >= 0.5
                            and (self._calibrated_scoring_fps <= 0.0
                                 or abs(actual_fps - self._calibrated_scoring_fps)
                                 / self._calibrated_scoring_fps >= 0.15)):
                        if bridge.set_fps(actual_fps):
                            self._calibrated_scoring_fps = actual_fps
                    self._last_scoring_fps_update = now

    # ---- Timer ticks ----

    def _tick_training(self):
        self._elapsed_seconds += 1
        m, s = divmod(self._elapsed_seconds, 60)
        self._timer_label.setText(f"{m:02d}:{s:02d}")
        if not self._pipeline.stub_mode and self._elapsed_seconds >= 3:
            stats = self._pipeline.performance_stats()
            healthy = (stats["rgb_30fps_ok"] and stats["depth_30fps_ok"]
                       and stats["pair_30fps_ok"])
            if not healthy and not self._fps_warning_active:
                self._fps_warning_active = True
                self._append_feedback(
                    f"⚠ 真实采集帧率未达到30：RGB {stats['rgb_fps']:.1f}，"
                    f"Depth {stats['depth_fps']:.1f}，显示 {stats['pair_fps']:.1f} fps")
                InfoBar.warning(
                    "相机帧率下降",
                    "当前显示的仍是真实数据，请检查 USB 带宽、分辨率和设备负载。",
                    duration=5000, position=InfoBarPosition.TOP_RIGHT, parent=self)
            elif healthy:
                self._fps_warning_active = False

    def _refresh_preview(self):
        if (self._state == TrainingState.CAPTURING
                and not self._pipeline.is_running
                and not self._pipeline.is_stopping):
            self._append_feedback("采集 Pipeline 已意外停止，请重新开始采集。")
            self._update_state(TrainingState.IDLE)
        frame = self._pipeline.preview.latest_frame()
        if frame:
            self._preview.set_frame(frame)

        # Update EMG from preview
        if frame and frame.emg_status:
            self._emg_panel.set_frame(frame)

    # ---- Feedback ----

    def _append_feedback(self, message: str):
        ts = datetime.now().strftime("%H:%M:%S")
        self._feedback.append(f"[{ts}] {message}")

    # ---- Shutdown ----

    def shutdown(self):
        """Start application shutdown and emit shutdown_ready when safe to exit."""
        print("[TrainingPage] 关闭 Pipeline...", flush=True)
        self._shutdown_requested = True
        self._preview_timer.stop()
        self._training_timer.stop()
        self._voice.stop()
        if self._ending:
            return
        active = (self._pipeline.is_running or self._pipeline.is_recording
                  or self._pipeline.is_stopping)
        if active:
            self._end_session(
                TrainingState.IDLE, generate_report=False,
                reason="应用关闭，训练数据已安全保存。")
            return
        try:
            self._course_runner.stop_course()
        except Exception:
            pass
        if self._score_bridge:
            self._score_bridge.stop()
            self._score_bridge = None
        print("[TrainingPage] Pipeline 已关闭", flush=True)
        QTimer.singleShot(0, self.shutdown_ready.emit)
