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
import time
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Optional

from PyQt5.QtCore import Qt, QTimer, pyqtSignal
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
    QLabel, QCheckBox, QSizePolicy,
)

from qfluentwidgets import (
    CardWidget, SimpleCardWidget,
    PrimaryPushButton, PushButton,
    TitleLabel, SubtitleLabel, BodyLabel, StrongBodyLabel, CaptionLabel,
    InfoBar, InfoBarPosition,
    LineEdit, ProgressRing, IndeterminateProgressRing,
    TextEdit, ScrollArea,
)

from rehab_engine.course import CourseRepository, CourseRunner, RunnerState
from rehab_engine.scoring import ScoreBridge, ScoreResult
from rehab_engine.recorder import Skeleton3DRecorder, EmgRecorder
from rehab_engine.sensor_pipeline import SensorPipeline
from rehab_engine.preview import PreviewComposer, PreviewFrame
from rehab_engine._stub import PipelineConfig, logger
import rehab_engine  # for rehab_engine._STUB_MODE


from ..widgets.preview_widget import PreviewWidget
from ..widgets.score_panel import ScorePanel
from ..widgets.emg_panel import EmgPanel


class TrainingState(Enum):
    IDLE = "待开始"
    TRAINING = "训练中"
    RESTING = "休息中"
    PAUSED = "已暂停"
    FINISHED = "已完成"


class TrainingPage(QWidget):
    """Core training page with preview, scoring, and controls."""

    report_requested = pyqtSignal(str, str)  # session_dir, csv_path
    status_message = pyqtSignal(str)

    def __init__(self, config: PipelineConfig, parent=None):
        super().__init__(parent)
        self._config = config
        self._state = TrainingState.IDLE
        self._elapsed_seconds = 0
        self._session_dir = ""

        # Camera detection
        self._cameras_found: list = []
        self._camera_checked: bool = False

        # Pipeline modules
        self._pipeline = SensorPipeline(config)
        self._recorder = Skeleton3DRecorder()
        self._emg_recorder = EmgRecorder()
        self._course_runner = CourseRunner()
        self._score_bridge: Optional[ScoreBridge] = None
        self._course_repo = CourseRepository()

        # Load course
        self._course_repo.load()
        courses = self._course_repo.courses
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
        self._append_feedback(f"✓ 配置: EMG={'启用' if self._config.emg.enabled else '禁用'} "
                             f"模式={self._config.emg.mode}")

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

        if rehab_engine._STUB_MODE:
            return "📷 模拟画面 (STUB模式)"

        if self._cameras_found:
            running = "运行中" if self._pipeline.is_running else "待机"
            return f"📷 {len(self._cameras_found)}个摄像头 ({running})"
        else:
            return "📷 无摄像头"

    # ---- Properties ----

    @property
    def is_training(self) -> bool:
        return self._state in (TrainingState.TRAINING, TrainingState.RESTING)

    def pipeline_stats(self) -> dict:
        return self._pipeline.performance_stats()

    # ---- UI Construction ----

    def _init_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(16, 12, 16, 12)
        root.setSpacing(12)

        # --- Top bar ---
        top_card = CardWidget(self)
        top_layout = QHBoxLayout(top_card)
        top_layout.setContentsMargins(16, 10, 16, 10)

        self._course_label = TitleLabel(
            self._current_course.course_name if self._current_course
            else "未选择课程")
        self._action_label = BodyLabel("当前动作：—")
        self._state_badge = CaptionLabel("待开始")
        self._state_badge.setObjectName("stateBadge")
        self._timer_label = SubtitleLabel("00:00")
        self._timer_label.setStyleSheet("color: #2F80ED; font-size: 28px; font-weight: 700;")

        top_layout.addWidget(self._course_label, 1)
        top_layout.addWidget(self._action_label)
        top_layout.addWidget(self._state_badge)
        top_layout.addWidget(self._timer_label)
        root.addWidget(top_card)

        # --- Main row: preview + side panel ---
        main_row = QHBoxLayout()
        main_row.setSpacing(12)

        # -- Preview area --
        preview_card = CardWidget(self)
        preview_layout = QVBoxLayout(preview_card)
        preview_layout.setContentsMargins(12, 8, 12, 12)

        tag_row = QHBoxLayout()
        for tag in ["RGB", "Depth", "Skeleton"]:
            lbl = CaptionLabel(tag)
            lbl.setStyleSheet("background:#EAF3FF; color:#2F80ED; border-radius:6px; padding:2px 10px;")
            tag_row.addWidget(lbl)
        tag_row.addStretch()
        preview_layout.addLayout(tag_row)

        self._preview = PreviewWidget(self)
        self._preview.setMinimumSize(720, 480)
        self._preview.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        preview_layout.addWidget(self._preview, 1)

        # -- Side panel --
        side = QWidget(self)
        side.setFixedWidth(320)
        side_layout = QVBoxLayout(side)
        side_layout.setContentsMargins(0, 0, 0, 0)
        side_layout.setSpacing(10)

        # Course info card
        info_card = SimpleCardWidget(side)
        info_layout = QVBoxLayout(info_card)
        info_layout.setContentsMargins(16, 14, 16, 14)
        info_layout.setSpacing(8)
        info_layout.addWidget(StrongBodyLabel("课程信息"))
        self._info_action = BodyLabel("动作：—")
        self._info_target = BodyLabel("目标次数：—")
        self._info_progress = BodyLabel("进度：— / —")
        self._rest_label = BodyLabel("休息：—")
        self._rest_label.setStyleSheet("color: #B45309;")
        info_layout.addWidget(self._info_action)
        info_layout.addWidget(self._info_target)
        info_layout.addWidget(self._info_progress)
        info_layout.addWidget(self._rest_label)
        side_layout.addWidget(info_card)

        # Score panel
        self._score_panel = ScorePanel(side)
        side_layout.addWidget(self._score_panel)

        # EMG panel
        self._emg_panel = EmgPanel(side)
        side_layout.addWidget(self._emg_panel)

        side_layout.addStretch()

        main_row.addWidget(preview_card, 1)
        main_row.addWidget(side)
        root.addLayout(main_row, 1)

        # --- Feedback log ---
        self._feedback = TextEdit(self)
        self._feedback.setReadOnly(True)
        self._feedback.setPlaceholderText("训练反馈将显示在这里…")
        self._feedback.setMaximumHeight(110)
        root.addWidget(self._feedback)

        # --- Control bar ---
        ctrl_card = CardWidget(self)
        ctrl = QHBoxLayout(ctrl_card)
        ctrl.setContentsMargins(16, 10, 16, 10)
        ctrl.setSpacing(10)

        self._btn_start = PrimaryPushButton("开始训练")
        self._btn_pause = PushButton("暂停")
        self._btn_stop = PushButton("停止")
        self._chk_rgb = QCheckBox("录制 RGB 视频")
        self._chk_rgb.setChecked(True)
        self._lbl_skeleton = CaptionLabel("3D骨骼CSV：强制录制")
        self._lbl_skeleton.setStyleSheet("color: #27AE60;")
        self._btn_report = PrimaryPushButton("结束并生成报告")
        self._btn_open_report = PushButton("查看报告")

        self._btn_start.clicked.connect(self._on_start)
        self._btn_pause.clicked.connect(self._on_pause)
        self._btn_stop.clicked.connect(self._on_stop)
        self._btn_report.clicked.connect(self._on_finish)
        self._btn_open_report.clicked.connect(self._on_open_report)

        ctrl.addWidget(self._btn_start)
        ctrl.addWidget(self._btn_pause)
        ctrl.addWidget(self._btn_stop)
        ctrl.addSpacing(16)
        ctrl.addWidget(self._chk_rgb)
        ctrl.addWidget(self._lbl_skeleton)
        ctrl.addStretch()
        ctrl.addWidget(self._btn_report)
        ctrl.addWidget(self._btn_open_report)
        root.addWidget(ctrl_card)

    # ---- Course runner signals ----

    def _wire_signals(self):
        self._course_runner.on_action_changed = self._on_action_changed
        self._course_runner.on_action_completed = self._on_action_completed
        self._course_runner.on_rest_started = self._on_rest_started
        self._course_runner.on_rest_tick = self._on_rest_tick
        self._course_runner.on_course_finished = self._on_course_finished
        self._course_runner.on_state_changed = self._on_runner_state

    # ---- State machine ----

    def _update_state(self, state: TrainingState):
        self._state = state
        self._state_badge.setText(state.value)
        self._update_buttons()

    def _update_buttons(self):
        s = self._state
        self._btn_start.setEnabled(s == TrainingState.IDLE)
        self._btn_pause.setEnabled(s == TrainingState.TRAINING)
        self._btn_stop.setEnabled(s in (TrainingState.TRAINING, TrainingState.RESTING, TrainingState.PAUSED))
        self._btn_report.setEnabled(s in (TrainingState.TRAINING, TrainingState.RESTING, TrainingState.PAUSED, TrainingState.FINISHED))
        self._btn_open_report.setEnabled(bool(self._session_dir))
        self._chk_rgb.setEnabled(s == TrainingState.IDLE)

    # ---- Button handlers ----

    def _on_start(self):
        if not self._current_course:
            self._append_feedback("请先在设置页面选择课程。")
            return

        # Log pipeline start details
        self._append_feedback(f"═══ 启动训练 Pipeline ═══")
        self._append_feedback(f"引擎模式: {'STUB (模拟)' if rehab_engine._STUB_MODE else 'FULL (真实)'}")
        self._append_feedback(f"RGB 设备: {self._config.device.rgb_device_path}")
        self._append_feedback(f"分辨率: {self._config.device.rgb_width}x{self._config.device.rgb_height} @ {self._config.device.rgb_fps}fps")
        self._append_feedback(f"课程: {self._current_course.course_name}")
        self._append_feedback(f"EMG: {'启用' if self._config.emg.enabled else '禁用'} ({self._config.emg.mode})")

        self._append_feedback(f"启动训练：{self._current_course.course_name}")

        self._pipeline.start()
        self._session_dir = self._pipeline.start_recording(
            str(Path(self._config.record_path) / "sessions"))

        # Log pipeline status after start
        if rehab_engine._STUB_MODE:
            self._append_feedback("⚠ Pipeline 运行在模拟模式 — 画面为合成数据")
        else:
            self._append_feedback("✓ Pipeline 已启动 — 等待相机数据...")

        self._elapsed_seconds = 0

        self._course_runner.start_course(self._current_course)
        self._training_timer.start(1000)
        self._update_state(TrainingState.TRAINING)

    def _on_pause(self):
        self._training_timer.stop()
        self._update_state(TrainingState.PAUSED)
        self._append_feedback("训练已暂停。")

    def _on_stop(self):
        self._training_timer.stop()
        self._pipeline.stop_recording()
        self._pipeline.stop()
        self._score_bridge = None
        self._update_state(TrainingState.IDLE)
        self._append_feedback("训练已停止。")
        self._log_stop_stats()

    def _on_finish(self):
        self._training_timer.stop()
        self._update_state(TrainingState.FINISHED)
        self._pipeline.stop_recording()
        self._pipeline.stop()
        self._score_bridge = None
        self._append_feedback("训练完成，正在生成报告…")
        csv_path = str(Path(self._session_dir) / "skeleton_3d.csv")
        self.report_requested.emit(self._session_dir, csv_path)

    def _on_open_report(self):
        if self._session_dir:
            csv_path = str(Path(self._session_dir) / "skeleton_3d.csv")
            self.report_requested.emit(self._session_dir, csv_path)

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
        self._action_label.setText(f"当前动作：{action.name_cn} ({action.action_id})")
        self._info_action.setText(f"动作：{action.name_cn} ({action.action_id})")
        self._info_target.setText(f"目标次数：{action.target_reps} 次")
        self._info_progress.setText(f"进度：{self._course_runner.current_action_index + 1} / {self._course_runner.total_actions}")
        self._rest_label.setText("休息：—")

        # Start scoring for this action
        if self._pipeline.is_running:
            fps = self._config.device.rgb_fps / max(1, self._config.pose.pose_interval)
            self._score_bridge = ScoreBridge()
            self._score_bridge.on_score_updated = self._on_score
            self._score_bridge.start(action.action_id, fps)

    def _on_action_completed(self, action, actual_reps, avg_score):
        self._append_feedback(
            f"完成 {action.name_cn}：{actual_reps}/{action.target_reps} 次，"
            f"平均评分 {avg_score:.1f}")

    def _on_rest_started(self, action, rest_sec):
        self._update_state(TrainingState.RESTING)
        self._rest_label.setText(f"休息：{rest_sec} 秒")
        self._append_feedback(f"动作完成，休息 {rest_sec} 秒…")

    def _on_rest_tick(self, remaining):
        self._rest_label.setText(f"休息：{remaining} 秒")

    def _on_course_finished(self):
        self._update_state(TrainingState.FINISHED)
        self._append_feedback("全部动作已完成！")

    def _on_runner_state(self, state: RunnerState):
        if state == RunnerState.TRAINING:
            self._update_state(TrainingState.TRAINING)

    def _on_score(self, result: ScoreResult):
        self._score_panel.set_score(result)
        if self._course_runner:
            self._course_runner.on_score_updated(result)

    # ---- Timer ticks ----

    def _tick_training(self):
        self._elapsed_seconds += 1
        m, s = divmod(self._elapsed_seconds, 60)
        self._timer_label.setText(f"{m:02d}:{s:02d}")

    def _refresh_preview(self):
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
        print("[TrainingPage] 关闭 Pipeline...", flush=True)
        self._pipeline.stop()
        self._preview_timer.stop()
        self._training_timer.stop()
        if self._score_bridge:
            self._score_bridge.stop()
        print("[TrainingPage] Pipeline 已关闭", flush=True)