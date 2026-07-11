"""
Settings page — device configuration and system preferences.
Replaces app/dialogs/DeviceSettingsDialog.cpp + RecordingSettingsDialog.cpp.
"""

from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGridLayout, QGroupBox,
)
from PyQt5.QtCore import Qt

from qfluentwidgets import (
    CardWidget, SimpleCardWidget, ScrollArea,
    PrimaryPushButton, PushButton,
    TitleLabel, SubtitleLabel, BodyLabel, StrongBodyLabel, CaptionLabel,
    ComboBox, EditableComboBox, LineEdit, SwitchButton, InfoBar, InfoBarPosition,
    SpinBox,
)

from rehab_engine._stub import PipelineConfig
from rehab_engine.config_loader import load_pipeline_config
from rehab_engine.course import CourseRepository


_DEVICE_ITEMS = ["/dev/video0", "/dev/video1", "/dev/video2"]
_RGB_FORMATS = ["MJPG", "YUYV"]
_RESOLUTIONS = ["640x480", "1280x720", "1920x1080"]
_DEPTH_RESOLUTIONS = ["640x480", "320x240", "1280x720"]
_FPS_VALUES = ["15", "30", "60"]
_EMG_MODES = ["disabled", "mock", "real"]


class SettingsPage(ScrollArea):
    """Device and system settings."""

    def __init__(self, config: PipelineConfig, parent=None):
        super().__init__(parent)
        self._config = config
        self._course_repo = CourseRepository()
        self._course_repo.load()

        self._init_ui()
        self._load_config()

    def _init_ui(self):
        self.setWidgetResizable(True)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

        w = QWidget(self)
        self.setWidget(w)
        root = QVBoxLayout(w)
        root.setContentsMargins(16, 12, 16, 12)
        root.setSpacing(12)

        # --- Camera settings ---
        cam_card = CardWidget(w)
        cam_layout = QVBoxLayout(cam_card)
        cam_layout.setContentsMargins(20, 16, 20, 16)
        cam_layout.setSpacing(12)
        cam_layout.addWidget(TitleLabel("相机设置"))

        grid = QGridLayout()
        grid.setSpacing(10)

        self._rgb_device = EditableComboBox()
        self._rgb_device.addItems(_DEVICE_ITEMS)
        self._rgb_format = ComboBox()
        self._rgb_format.addItems(_RGB_FORMATS)
        self._rgb_resolution = ComboBox()
        self._rgb_resolution.addItems(_RESOLUTIONS)
        self._rgb_fps = ComboBox()
        self._rgb_fps.addItems(_FPS_VALUES)

        grid.addWidget(BodyLabel("RGB 设备"), 0, 0)
        grid.addWidget(self._rgb_device, 0, 1)
        grid.addWidget(BodyLabel("格式"), 0, 2)
        grid.addWidget(self._rgb_format, 0, 3)
        grid.addWidget(BodyLabel("分辨率"), 1, 0)
        grid.addWidget(self._rgb_resolution, 1, 1)
        grid.addWidget(BodyLabel("FPS"), 1, 2)
        grid.addWidget(self._rgb_fps, 1, 3)
        cam_layout.addLayout(grid)

        depth_row = QHBoxLayout()
        self._depth_device = LineEdit()
        self._depth_device.setPlaceholderText("OpenNI2 设备 URI（留空为自动检测）")
        self._depth_resolution = ComboBox()
        self._depth_resolution.addItems(_DEPTH_RESOLUTIONS)
        self._depth_fps = ComboBox()
        self._depth_fps.addItems(_FPS_VALUES)
        self._hw_d2c = SwitchButton("硬件 D2C 对齐")
        self._hw_d2c.setChecked(True)

        depth_row.addWidget(BodyLabel("Depth"))
        depth_row.addWidget(self._depth_device, 1)
        depth_row.addWidget(self._depth_resolution)
        depth_row.addWidget(self._depth_fps)
        depth_row.addWidget(self._hw_d2c)
        cam_layout.addLayout(depth_row)

        root.addWidget(cam_card)

        # --- Course selection ---
        course_card = CardWidget(w)
        course_layout = QVBoxLayout(course_card)
        course_layout.setContentsMargins(20, 16, 20, 16)
        course_layout.setSpacing(8)
        course_layout.addWidget(TitleLabel("课程选择"))

        self._course_combo = ComboBox()
        for course in self._course_repo.courses:
            self._course_combo.addItem(
                f"{course.course_name} ({course.estimated_minutes}分钟)")
        course_layout.addWidget(self._course_combo)

        course_info = CaptionLabel(
            "训练课程定义了动作序列、目标次数和休息时间。"
            "切换课程后需要重新开始训练。")
        course_info.setWordWrap(True)
        course_layout.addWidget(course_info)
        root.addWidget(course_card)

        # --- EMG settings ---
        emg_card = CardWidget(w)
        emg_layout = QVBoxLayout(emg_card)
        emg_layout.setContentsMargins(20, 16, 20, 16)
        emg_layout.setSpacing(10)
        emg_layout.addWidget(TitleLabel("肌电设置"))

        emg_row1 = QHBoxLayout()
        self._emg_enabled = SwitchButton()
        self._emg_enabled.setText("启用 EMG")
        self._emg_mode = ComboBox()
        self._emg_mode.addItems(_EMG_MODES)
        self._emg_serial = LineEdit()
        self._emg_serial.setText("/dev/rfcomm0")
        emg_row1.addWidget(self._emg_enabled)
        emg_row1.addWidget(BodyLabel("模式"))
        emg_row1.addWidget(self._emg_mode)
        emg_row1.addWidget(BodyLabel("串口"))
        emg_row1.addWidget(self._emg_serial, 1)
        emg_layout.addLayout(emg_row1)

        emg_row2 = QHBoxLayout()
        self._emg_rpmsg_ctrl = LineEdit()
        self._emg_rpmsg_ctrl.setText("/dev/rpmsg_ctrl0")
        self._emg_rpmsg_ctrl.setPlaceholderText("RPMsg control device")
        self._emg_rpmsg_data = LineEdit()
        self._emg_rpmsg_data.setText("/dev/rpmsg0")
        self._emg_rpmsg_data.setPlaceholderText("RPMsg data device")
        self._emg_endpoint = LineEdit()
        self._emg_endpoint.setText("emg_rpmsg")
        emg_row2.addWidget(BodyLabel("RPMsg Ctrl"))
        emg_row2.addWidget(self._emg_rpmsg_ctrl, 1)
        emg_row2.addWidget(BodyLabel("Data"))
        emg_row2.addWidget(self._emg_rpmsg_data, 1)
        emg_row2.addWidget(BodyLabel("Endpoint"))
        emg_row2.addWidget(self._emg_endpoint)
        emg_layout.addLayout(emg_row2)

        root.addWidget(emg_card)

        # --- Apply button ---
        btn_row = QHBoxLayout()
        btn_row.addStretch()
        self._btn_apply = PrimaryPushButton("应用设置")
        self._btn_apply.clicked.connect(self._apply)
        btn_row.addWidget(self._btn_apply)
        root.addLayout(btn_row)

        root.addStretch()

    def _load_config(self):
        c = self._config
        self._rgb_device.setCurrentText(c.device.rgb_device_path or "/dev/video0")
        self._rgb_format.setCurrentText(c.device.rgb_pixel_format)
        self._rgb_resolution.setCurrentText(
            f"{c.device.rgb_width}x{c.device.rgb_height}")
        self._rgb_fps.setCurrentText(str(c.device.rgb_fps))

        self._depth_device.setText(c.device.openni_device_uri)
        self._depth_resolution.setCurrentText(
            f"{c.device.depth_width}x{c.device.depth_height}")
        self._depth_fps.setCurrentText(str(c.device.depth_fps))
        self._hw_d2c.setChecked(c.device.enable_hardware_d2c)

        self._emg_enabled.setChecked(c.emg.enabled)
        self._emg_mode.setCurrentText(c.emg.mode)
        self._emg_serial.setText(c.emg.serial_device)
        self._emg_rpmsg_ctrl.setText(c.emg.rpmsg_ctrl_device)
        self._emg_rpmsg_data.setText(c.emg.rpmsg_data_device)
        self._emg_endpoint.setText(c.emg.rpmsg_endpoint_name)

    def _apply(self):
        c = self._config

        c.device.rgb_device_path = self._rgb_device.currentText()
        c.device.rgb_pixel_format = self._rgb_format.currentText()
        res = self._rgb_resolution.currentText().split("x")
        if len(res) == 2:
            c.device.rgb_width = int(res[0])
            c.device.rgb_height = int(res[1])
        c.device.rgb_fps = int(self._rgb_fps.currentText())

        c.device.openni_device_uri = self._depth_device.text().strip()
        dres = self._depth_resolution.currentText().split("x")
        if len(dres) == 2:
            c.device.depth_width = int(dres[0])
            c.device.depth_height = int(dres[1])
        c.device.depth_fps = int(self._depth_fps.currentText())
        c.device.enable_hardware_d2c = self._hw_d2c.isChecked()

        c.emg.enabled = self._emg_enabled.isChecked()
        c.emg.mode = self._emg_mode.currentText()
        c.emg.serial_device = self._emg_serial.text().strip()
        c.emg.rpmsg_ctrl_device = self._emg_rpmsg_ctrl.text().strip()
        c.emg.rpmsg_data_device = self._emg_rpmsg_data.text().strip()
        c.emg.rpmsg_endpoint_name = self._emg_endpoint.text().strip()

        InfoBar.success("设置已应用", "设备配置已更新，下次训练生效。",
                        position=InfoBarPosition.BOTTOM_RIGHT, parent=self)