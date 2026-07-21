"""Compact EMG status and time-domain feature panel."""

from __future__ import annotations

from collections import deque
from math import isfinite

from PyQt5.QtCore import QRectF, Qt
from PyQt5.QtGui import QColor, QPainter, QPen
from PyQt5.QtWidgets import (
    QDialog,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from qfluentwidgets import BodyLabel, CaptionLabel, PushButton, SimpleCardWidget

from rehab_engine.preview import PreviewFrame
from ..theme import COLORS, pill_style


class EmgWaveform(QWidget):
    """Lightweight two-channel EMG waveform painter for recent feature values."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._history = [deque(maxlen=240), deque(maxlen=240)]
        self.setMinimumHeight(180)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

    def add_sample(self, values):
        for index in range(2):
            value = float(values[index]) if index < len(values) else 0.0
            self._history[index].append(max(0.0, value))
        self.update()

    def reset(self):
        for channel in self._history:
            channel.clear()
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        rect = self.rect().adjusted(1, 1, -1, -1)
        painter.setPen(QPen(QColor(COLORS["border"]), 1))
        painter.setBrush(QColor("#F8FAFD"))
        painter.drawRoundedRect(rect, 8, 8)

        plot = QRectF(rect).adjusted(12, 16, -12, -18)
        max_value = max(
            [1.0] + [max(channel) for channel in self._history if channel]
        )
        colors = [QColor(COLORS["primary"]), QColor(COLORS["success"])]
        labels = ["CH1 RMS", "CH2 RMS"]

        painter.setPen(QPen(QColor("#E5EAF2"), 1))
        for ratio in (0.25, 0.50, 0.75):
            y = int(plot.top() + plot.height() * ratio)
            painter.drawLine(int(plot.left()), y, int(plot.right()), y)

        for index, channel in enumerate(self._history):
            points = list(channel)
            lane_height = plot.height() / 2.0
            lane_top = plot.top() + index * lane_height
            baseline = lane_top + lane_height * 0.86
            amplitude = lane_height * 0.70

            painter.setPen(QPen(colors[index], 1.8))
            painter.drawText(int(plot.left()), int(lane_top + 14), labels[index])
            if len(points) < 2:
                continue
            step = plot.width() / max(1, len(points) - 1)
            last_x = plot.left()
            last_y = baseline - (points[0] / max_value) * amplitude
            for i, value in enumerate(points[1:], start=1):
                x = plot.left() + i * step
                y = baseline - (value / max_value) * amplitude
                painter.drawLine(int(last_x), int(last_y), int(x), int(y))
                last_x, last_y = x, y


class EmgPanel(SimpleCardWidget):
    """Compact EMG monitor with time-domain features and waveform dialog."""

    _FEATURE_ROWS = [
        ("rms", "RMS", 1),
        ("mav", "MAV", 1),
        ("iemg", "IEMG", 0),
        ("wl", "WL", 0),
        ("zc", "ZC", 0),
    ]

    def __init__(self, parent=None):
        super().__init__(parent)
        self._wave_dialog = None
        self._waveform = EmgWaveform(self)
        self._feature_labels = {}
        self._init_ui()

    def _init_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(14, 12, 14, 12)
        layout.setSpacing(8)

        header = QHBoxLayout()
        title_box = QVBoxLayout()
        title_box.setSpacing(1)
        title = QLabel("肌电时域特征")
        title.setObjectName("sectionTitle")
        hint = QLabel("RMS / MAV / IEMG / WL / ZC")
        hint.setObjectName("sectionHint")
        title_box.addWidget(title)
        title_box.addWidget(hint)
        header.addLayout(title_box, 1)
        self._status_dot = QLabel("未连接")
        self._status_dot.setStyleSheet(pill_style("neutral"))
        header.addWidget(self._status_dot)
        layout.addLayout(header)

        self._status_label = CaptionLabel("未接入肌电数据")
        self._status_label.setStyleSheet(f"color:{COLORS['muted']};")
        self._status_label.setWordWrap(True)
        layout.addWidget(self._status_label)

        grid = QGridLayout()
        grid.setHorizontalSpacing(8)
        grid.setVerticalSpacing(6)
        grid.addWidget(self._grid_header("特征"), 0, 0)
        grid.addWidget(self._grid_header("CH1"), 0, 1)
        grid.addWidget(self._grid_header("CH2"), 0, 2)
        for row, (key, label, _decimals) in enumerate(self._FEATURE_ROWS, start=1):
            name = CaptionLabel(label)
            name.setStyleSheet(f"color:{COLORS['muted']};")
            grid.addWidget(name, row, 0)
            for channel in range(2):
                value = BodyLabel("—")
                value.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
                value.setMinimumWidth(58)
                value.setStyleSheet(f"color:{COLORS['ink']}; font-weight:700;")
                grid.addWidget(value, row, channel + 1)
                self._feature_labels[(key, channel)] = value
        layout.addLayout(grid)

        bottom = QHBoxLayout()
        self._fatigue_label = CaptionLabel("疲劳指数：—")
        self._fatigue_label.setStyleSheet(f"color:{COLORS['muted']};")
        bottom.addWidget(self._fatigue_label, 1)
        self._wave_btn = PushButton("实时波形")
        self._wave_btn.setFixedHeight(30)
        self._wave_btn.clicked.connect(self._open_waveform_dialog)
        bottom.addWidget(self._wave_btn)
        layout.addLayout(bottom)

    def _grid_header(self, text: str):
        label = CaptionLabel(text)
        label.setStyleSheet(f"color:{COLORS['muted']}; font-weight:700;")
        label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        return label

    def _open_waveform_dialog(self):
        if self._wave_dialog is None:
            self._wave_dialog = QDialog(self)
            self._wave_dialog.setWindowTitle("肌电实时波形")
            self._wave_dialog.resize(760, 420)
            root = QVBoxLayout(self._wave_dialog)
            root.setContentsMargins(16, 16, 16, 16)
            root.setSpacing(10)
            title = BodyLabel("肌电实时波形（最近 RMS 趋势）")
            title.setStyleSheet(f"color:{COLORS['ink']}; font-weight:700;")
            root.addWidget(title)
            root.addWidget(self._waveform, 1)
        self._wave_dialog.show()
        self._wave_dialog.raise_()
        self._wave_dialog.activateWindow()

    def set_frame(self, frame: PreviewFrame):
        """Update EMG display from a PreviewFrame."""
        status = frame.emg_status or "肌电未接入"
        self._apply_status(status)

        features = self._normalize_features(frame)
        rms_values = [item.get("rms", 0.0) for item in features]
        self._waveform.add_sample(rms_values)

        for key, _label, decimals in self._FEATURE_ROWS:
            for channel in range(2):
                label = self._feature_labels[(key, channel)]
                value = features[channel].get(key, 0.0) if channel < len(features) else 0.0
                label.setText(self._format_value(value, decimals))

        fatigue = [item.get("fatigue", 0.0) for item in features]
        fatigue_text = " / ".join(self._format_value(value, 2) for value in fatigue[:2])
        self._fatigue_label.setText(f"疲劳指数：{fatigue_text or '—'}")

    def _apply_status(self, status: str):
        lowered = status.lower()
        if "waiting" in lowered or "error" in lowered or "failed" in lowered:
            self._status_label.setText("肌电：等待真实设备数据")
            self._status_label.setStyleSheet(f"color:{COLORS['warning']};")
            self._status_dot.setText("等待中")
            self._status_dot.setStyleSheet(pill_style("warning"))
        elif "mock" in lowered:
            self._status_label.setText("肌电：模拟模式")
            self._status_label.setStyleSheet(f"color:{COLORS['warning']};")
            self._status_dot.setText("模拟")
            self._status_dot.setStyleSheet(pill_style("warning"))
        elif "real" in lowered or "connected" in lowered:
            self._status_label.setText("肌电：真实链路正常")
            self._status_label.setStyleSheet(f"color:{COLORS['success']};")
            self._status_dot.setText("已连接")
            self._status_dot.setStyleSheet(pill_style("success"))
        elif "disabled" in lowered or not status.strip():
            self._status_label.setText("肌电：未启用")
            self._status_label.setStyleSheet(f"color:{COLORS['muted']};")
            self._status_dot.setText("未启用")
            self._status_dot.setStyleSheet(pill_style("neutral"))
        else:
            self._status_label.setText(status[:80])
            self._status_label.setStyleSheet(f"color:{COLORS['text']};")

    def _normalize_features(self, frame: PreviewFrame):
        features = [dict(item) for item in (frame.emg_features or [])[:2]]
        rms_values = frame.emg_rms or []
        fatigue_values = frame.emg_fatigue_index or []
        while len(features) < 2:
            features.append({})
        for channel in range(2):
            if channel < len(rms_values):
                features[channel].setdefault("rms", float(rms_values[channel]))
            if channel < len(fatigue_values):
                features[channel].setdefault("fatigue", float(fatigue_values[channel]))
            features[channel].setdefault("mav", None)
            features[channel].setdefault("iemg", None)
            features[channel].setdefault("wl", None)
            if "zc" not in features[channel] and features[channel].get("zcr") is not None:
                features[channel]["zc"] = float(features[channel].get("zcr", 0.0)) * 100.0
            features[channel].setdefault("zc", None)
        return features

    @staticmethod
    def _format_value(value, decimals: int) -> str:
        try:
            number = float(value)
        except (TypeError, ValueError):
            return "—"
        if not isfinite(number):
            return "—"
        return f"{number:.{decimals}f}"
