"""
Settings page — device configuration and system preferences.
Replaces app/dialogs/DeviceSettingsDialog.cpp + RecordingSettingsDialog.cpp.
"""

import threading

from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGridLayout, QGroupBox, QLabel,
)
from PyQt5.QtCore import Qt, pyqtSignal

from qfluentwidgets import (
    CardWidget, SimpleCardWidget, ScrollArea,
    PrimaryPushButton, PushButton,
    TitleLabel, SubtitleLabel, BodyLabel, StrongBodyLabel, CaptionLabel,
    ComboBox, EditableComboBox, LineEdit, SwitchButton, InfoBar, InfoBarPosition,
    SpinBox,
)

from rehab_engine._stub import PipelineConfig
from rehab_engine.config_loader import save_pipeline_config
from rehab_engine.course import CourseRepository
from rehab_engine.diagnostics import run_diagnostics
from rehab_engine.emg import EmgBluetoothScanner
from ..theme import COLORS, PAGE_STYLE, pill_style


_DEVICE_ITEMS = ["/dev/video0", "/dev/video1", "/dev/video2"]
_RGB_FORMATS = ["MJPG", "YUYV"]
_RESOLUTIONS = ["640x480", "1280x720", "1920x1080"]
_DEPTH_RESOLUTIONS = ["640x480", "320x240", "1280x720"]
# RGB and depth are intentionally locked to the synchronized 30 FPS profile.
# If hardware cannot sustain it, the runtime reports the measured rate instead
# of substituting mock frames.
_FPS_VALUES = ["30"]
_EMG_BACKENDS = ["bluez", "serial"]


class SettingsPage(ScrollArea):
    """Device and system settings."""

    course_changed = pyqtSignal(str)
    debug_changed = pyqtSignal(bool)
    settings_applied = pyqtSignal(str)
    ble_scan_finished = pyqtSignal(object)
    log_requested = pyqtSignal()
    performance_requested = pyqtSignal()

    def __init__(self, config: PipelineConfig, parent=None):
        super().__init__(parent)
        self._config = config
        self._course_repo = CourseRepository()
        self._course_repo.load()
        self._ble_devices = {}

        self._init_ui()
        self.ble_scan_finished.connect(self._finish_ble_scan)
        self._load_config()

    def _init_ui(self):
        self.setObjectName("settingsPage")
        self.setStyleSheet(PAGE_STYLE)
        self.setWidgetResizable(True)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

        w = QWidget(self)
        self.setWidget(w)
        root = QVBoxLayout(w)
        root.setContentsMargins(28, 20, 28, 24)
        root.setSpacing(16)

        eyebrow = QLabel("SYSTEM PREFERENCES")
        eyebrow.setObjectName("pageEyebrow")
        page_title = QLabel("设备与训练设置")
        page_title.setObjectName("pageTitle")
        page_hint = QLabel("配置采集设备、训练课程和肌电链路；修改将在下一次训练时生效")
        page_hint.setObjectName("pageSubtitle")
        root.addWidget(eyebrow)
        root.addWidget(page_title)
        root.addWidget(page_hint)

        # --- Camera settings ---
        cam_card = CardWidget(w)
        cam_layout = QVBoxLayout(cam_card)
        cam_layout.setContentsMargins(22, 18, 22, 20)
        cam_layout.setSpacing(14)
        cam_title = QLabel("相机与深度采集")
        cam_title.setObjectName("sectionTitle")
        cam_hint = QLabel("设置 RGB 与深度相机参数，建议两路帧率保持一致")
        cam_hint.setObjectName("sectionHint")
        cam_layout.addWidget(cam_title)
        cam_layout.addWidget(cam_hint)

        grid = QGridLayout()
        grid.setHorizontalSpacing(14)
        grid.setVerticalSpacing(10)
        grid.setColumnStretch(1, 3)
        grid.setColumnStretch(3, 2)

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

        depth_grid = QGridLayout()
        depth_grid.setHorizontalSpacing(14)
        depth_grid.setVerticalSpacing(10)
        depth_grid.setColumnStretch(1, 3)
        self._depth_device = LineEdit()
        self._depth_device.setPlaceholderText("OpenNI2 设备 URI（留空为自动检测）")
        self._depth_resolution = ComboBox()
        self._depth_resolution.addItems(_DEPTH_RESOLUTIONS)
        self._depth_fps = ComboBox()
        self._depth_fps.addItems(_FPS_VALUES)
        self._hw_d2c = SwitchButton("硬件 D2C 对齐")
        self._hw_d2c.setChecked(True)

        depth_grid.addWidget(BodyLabel("深度设备"), 0, 0)
        depth_grid.addWidget(self._depth_device, 0, 1)
        depth_grid.addWidget(BodyLabel("分辨率"), 0, 2)
        depth_grid.addWidget(self._depth_resolution, 0, 3)
        depth_grid.addWidget(BodyLabel("FPS"), 1, 0)
        depth_grid.addWidget(self._depth_fps, 1, 1)
        depth_grid.addWidget(self._hw_d2c, 1, 3)
        cam_layout.addLayout(depth_grid)

        root.addWidget(cam_card)

        # --- Course selection ---
        course_card = CardWidget(w)
        course_layout = QVBoxLayout(course_card)
        course_layout.setContentsMargins(22, 18, 22, 20)
        course_layout.setSpacing(10)
        course_title = QLabel("默认训练课程")
        course_title.setObjectName("sectionTitle")
        course_layout.addWidget(course_title)

        self._course_combo = ComboBox()
        self._course_ids = []
        for course in self._course_repo.courses:
            self._course_ids.append(course.course_id)
            self._course_combo.addItem(
                f"{course.course_name} ({course.estimated_minutes}分钟)")

        course_grid = QGridLayout()
        course_grid.setHorizontalSpacing(14)
        self._patient_name = LineEdit()
        self._patient_id = LineEdit()
        self._patient_id.setPlaceholderText("P0001")
        self._patient_gender = ComboBox()
        self._patient_gender.addItems(["", "男", "女", "其他"])
        self._patient_age = SpinBox()
        self._patient_age.setRange(0, 120)
        self._patient_age.setSpecialValueText("未填写")
        self._patient_diagnosis = LineEdit()
        self._patient_diagnosis.setPlaceholderText("例如：卒中后上肢康复")
        self._patient_name.setPlaceholderText("可选，用于区分训练报告")
        self._debug_switch = SwitchButton("显示性能调试信息")
        course_grid.addWidget(BodyLabel("训练对象"), 0, 0)
        course_grid.addWidget(self._patient_name, 0, 1)
        course_grid.addWidget(BodyLabel("训练课程"), 1, 0)
        course_grid.addWidget(self._course_combo, 1, 1)
        course_grid.addWidget(self._debug_switch, 2, 1)
        course_grid.setColumnStretch(1, 1)
        course_grid.addWidget(BodyLabel("患者编号"), 3, 0)
        course_grid.addWidget(self._patient_id, 3, 1)
        course_grid.addWidget(BodyLabel("性别"), 4, 0)
        course_grid.addWidget(self._patient_gender, 4, 1)
        course_grid.addWidget(BodyLabel("年龄"), 5, 0)
        course_grid.addWidget(self._patient_age, 5, 1)
        course_grid.addWidget(BodyLabel("诊断/训练说明"), 6, 0)
        course_grid.addWidget(self._patient_diagnosis, 6, 1)
        course_layout.addLayout(course_grid)

        course_info = CaptionLabel(
            "训练课程定义了动作序列、目标次数和休息时间。"
            "切换课程后需要重新开始训练。")
        course_info.setWordWrap(True)
        course_info.setStyleSheet(f"color:{COLORS['muted']};")
        course_layout.addWidget(course_info)
        root.addWidget(course_card)

        # --- EMG settings ---
        emg_card = CardWidget(w)
        emg_layout = QVBoxLayout(emg_card)
        emg_layout.setContentsMargins(22, 18, 22, 20)
        emg_layout.setSpacing(12)
        emg_title = QLabel("肌电采集")
        emg_title.setObjectName("sectionTitle")
        emg_hint = QLabel("选择采集模式并配置设备链路；不使用肌电时可保持关闭")
        emg_hint.setObjectName("sectionHint")
        emg_layout.addWidget(emg_title)
        emg_layout.addWidget(emg_hint)

        emg_row1 = QHBoxLayout()
        self._emg_enabled = SwitchButton()
        self._emg_enabled.setText("启用 EMG")
        self._emg_backend = ComboBox()
        self._emg_backend.addItems(_EMG_BACKENDS)
        self._emg_serial = LineEdit()
        self._emg_serial.setText("/dev/rfcomm0")
        emg_row1.addWidget(self._emg_enabled)
        emg_row1.addWidget(BodyLabel("后端"))
        emg_row1.addWidget(self._emg_backend)
        emg_row1.addWidget(BodyLabel("串口"))
        emg_row1.addWidget(self._emg_serial, 1)
        emg_layout.addLayout(emg_row1)

        emg_ble_row = QHBoxLayout()
        self._emg_ble_device = EditableComboBox()
        self._emg_ble_device.setPlaceholderText("BLE 名称或 MAC 地址")
        self._btn_scan_ble = PushButton("扫描 BLE")
        self._btn_scan_ble.clicked.connect(self._scan_ble)
        emg_ble_row.addWidget(BodyLabel("BLE 设备"))
        emg_ble_row.addWidget(self._emg_ble_device, 1)
        emg_ble_row.addWidget(self._btn_scan_ble)
        emg_layout.addLayout(emg_ble_row)

        self._advanced_toggle = SwitchButton("显示高级链路参数")
        self._advanced_toggle.setChecked(False)
        emg_layout.addWidget(self._advanced_toggle)
        self._advanced_widget = QWidget()
        emg_row2 = QHBoxLayout(self._advanced_widget)
        emg_row2.setContentsMargins(0, 0, 0, 0)
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
        emg_layout.addWidget(self._advanced_widget)
        self._advanced_widget.setVisible(False)
        self._advanced_toggle.checkedChanged.connect(self._advanced_widget.setVisible)

        root.addWidget(emg_card)

        # --- Apply button ---
        btn_row = QHBoxLayout()
        apply_hint = QLabel("应用后不会中断当前页面，新的设备参数将在下次训练生效")
        apply_hint.setObjectName("sectionHint")
        btn_row.addWidget(apply_hint)
        btn_row.addStretch()
        self._btn_test = PushButton("测试设备")
        self._btn_test.setMinimumSize(110, 38)
        self._btn_test.clicked.connect(self._test_devices)
        btn_row.addWidget(self._btn_test)
        self._btn_logs = PushButton("运行日志")
        self._btn_logs.clicked.connect(self.log_requested)
        btn_row.addWidget(self._btn_logs)
        self._btn_performance = PushButton("性能监控")
        self._btn_performance.clicked.connect(self.performance_requested)
        btn_row.addWidget(self._btn_performance)
        self._btn_apply = PrimaryPushButton("应用设置")
        self._btn_apply.setMinimumSize(128, 38)
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
        self._emg_backend.setCurrentText(c.emg.capture_backend)
        self._emg_serial.setText(c.emg.serial_device)
        self._emg_ble_device.setCurrentText(c.emg.ble_address)
        self._emg_rpmsg_ctrl.setText(c.emg.rpmsg_ctrl_device)
        self._emg_rpmsg_data.setText(c.emg.rpmsg_data_device)
        self._emg_endpoint.setText(c.emg.rpmsg_endpoint_name)
        self._patient_name.setText(c.patient_name)
        self._patient_id.setText(c.patient_id)
        self._patient_gender.setCurrentText(c.patient_gender)
        self._patient_age.setValue(c.patient_age)
        self._patient_diagnosis.setText(c.patient_diagnosis)
        self._debug_switch.setChecked(c.ui_debug_enabled)
        if c.selected_course_id in self._course_ids:
            self._course_combo.setCurrentIndex(
                self._course_ids.index(c.selected_course_id))

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
        c.emg.capture_backend = self._emg_backend.currentText()
        c.emg.serial_device = self._emg_serial.text().strip()
        selected_ble = self._emg_ble_device.currentText().strip()
        c.emg.ble_address = self._ble_devices.get(selected_ble, selected_ble)
        c.emg.rpmsg_ctrl_device = self._emg_rpmsg_ctrl.text().strip()
        c.emg.rpmsg_data_device = self._emg_rpmsg_data.text().strip()
        c.emg.rpmsg_endpoint_name = self._emg_endpoint.text().strip()

        c.patient_name = self._patient_name.text().strip()
        c.patient_id = self._patient_id.text().strip()
        c.patient_gender = self._patient_gender.currentText().strip()
        c.patient_age = self._patient_age.value()
        c.patient_diagnosis = self._patient_diagnosis.text().strip()
        c.ui_debug_enabled = self._debug_switch.isChecked()
        index = self._course_combo.currentIndex()
        if 0 <= index < len(self._course_ids):
            c.selected_course_id = self._course_ids[index]

        try:
            config_path = save_pipeline_config(c)
        except OSError as exc:
            InfoBar.error("保存失败", str(exc),
                          position=InfoBarPosition.BOTTOM_RIGHT, parent=self)
            return

        self.course_changed.emit(c.selected_course_id)
        self.debug_changed.emit(c.ui_debug_enabled)
        self.settings_applied.emit(str(config_path))
        parent = self.window()
        if hasattr(parent, "refresh_diagnostics"):
            parent.refresh_diagnostics()

        InfoBar.success("设置已保存", f"配置已写入 {config_path.name}，下次训练生效。",
                        position=InfoBarPosition.BOTTOM_RIGHT, parent=self)

    def _scan_ble(self):
        self._btn_scan_ble.setEnabled(False)
        self._btn_scan_ble.setText("扫描中…")

        def worker():
            self.ble_scan_finished.emit(EmgBluetoothScanner().scan(4))

        threading.Thread(target=worker, name="settings-ble-scan", daemon=True).start()

    def _finish_ble_scan(self, result):
        self._btn_scan_ble.setEnabled(True)
        self._btn_scan_ble.setText("扫描 BLE")
        if not result.ok:
            InfoBar.error(
                "BLE 扫描失败", result.message, duration=5000,
                position=InfoBarPosition.TOP_RIGHT, parent=self,
            )
            return
        current = self._emg_ble_device.currentText().strip()
        self._ble_devices = {}
        self._emg_ble_device.clear()
        for device in result.devices:
            label = f"{device.name or 'Unknown'} [{device.address}] ({device.rssi} dBm)"
            self._ble_devices[label] = device.address
            self._emg_ble_device.addItem(label)
        if current:
            self._emg_ble_device.setCurrentText(current)
        if result.devices:
            InfoBar.success(
                "BLE 扫描完成", f"发现 {len(result.devices)} 个设备",
                position=InfoBarPosition.TOP_RIGHT, parent=self,
            )
        else:
            InfoBar.warning(
                "BLE 扫描完成", "未发现设备，请检查蓝牙适配器和设备广播状态",
                position=InfoBarPosition.TOP_RIGHT, parent=self,
            )

    def _test_devices(self):
        """Run the same diagnostics used at startup and summarize the result."""
        self._btn_test.setEnabled(False)
        try:
            diagnostics = run_diagnostics(self._config)
            errors = diagnostics.errors()
            warnings = diagnostics.warnings()
            if errors:
                detail = "；".join(item.name for item in errors[:4])
                InfoBar.error("设备检查未通过", detail, duration=6000,
                              position=InfoBarPosition.TOP_RIGHT, parent=self)
            elif warnings:
                detail = "；".join(item.name for item in warnings[:4])
                InfoBar.warning("设备可用但存在警告", detail, duration=5000,
                                position=InfoBarPosition.TOP_RIGHT, parent=self)
            else:
                InfoBar.success("设备检查通过", "相机、引擎与保存路径状态正常。",
                                position=InfoBarPosition.TOP_RIGHT, parent=self)
        finally:
            self._btn_test.setEnabled(True)
