"""
EMG status panel — displays EMG connection status and per-channel metrics.
Replaces the EMG section in TrainingPage.cpp.
"""

from collections import deque
from typing import Optional

from PyQt5.QtCore import QRectF, Qt
from PyQt5.QtGui import QColor, QFont, QPainter, QPen
from PyQt5.QtWidgets import (
    QDialog, QGridLayout, QHBoxLayout, QLabel, QProgressBar, QVBoxLayout,
    QWidget,
)

from qfluentwidgets import BodyLabel, CaptionLabel, PushButton, SimpleCardWidget

from rehab_engine.preview import PreviewFrame
from ..theme import COLORS, pill_style


class EmgWaveform(QWidget):
    """Lightweight two-channel EMG trend painter for RMS samples."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._history = [deque(maxlen=180), deque(maxlen=180)]
        self.setMinimumHeight(78)
        self.setMaximumHeight(100)

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

        plot = QRectF(rect).adjusted(10, 8, -10, -8)
        painter.setPen(QPen(QColor("#E5EAF2"), 1))
        painter.drawLine(
            int(plot.left()), int(plot.center().y()),
            int(plot.right()), int(plot.center().y()))

        max_value = max(
            [1.0]
            + [max(channel) for channel in self._history if len(channel) > 0]
        )
        colors = [QColor(COLORS["primary"]), QColor(COLORS["success"])]
        labels = ["通道一", "通道二"]
        for index, channel in enumerate(self._history):
            points = list(channel)
            if len(points) < 2:
                continue
            painter.setPen(QPen(colors[index], 1.8))
            step = plot.width() / max(1, len(points) - 1)
            baseline = plot.bottom() - index * (plot.height() * 0.48)
            amplitude = plot.height() * 0.42
            last_x = plot.left()
            last_y = baseline - (points[0] / max_value) * amplitude
            for i, value in enumerate(points[1:], start=1):
                x = plot.left() + i * step
                y = baseline - (value / max_value) * amplitude
                painter.drawLine(int(last_x), int(last_y), int(x), int(y))
                last_x, last_y = x, y
            painter.drawText(int(plot.left()), int(baseline - amplitude - 2), labels[index])


class EmgFeatureWaveform(QWidget):
    """Stacked feature waveform painter for one EMG channel."""

    FEATURES = [
        ("rms", "均方根强度", COLORS["primary"]),
        ("zcr", "过零率", COLORS["warning"]),
        ("cv", "波动系数", COLORS["cyan"]),
        ("fatigue", "疲劳指数", COLORS["danger"]),
        ("envelope", "包络均值", COLORS["success"]),
    ]

    def __init__(self, channel_name: str, parent=None):
        super().__init__(parent)
        self._channel_name = channel_name
        self._history = {key: deque(maxlen=360) for key, _, _ in self.FEATURES}
        self._state = "—"
        self.setMinimumHeight(330)

    def add_values(self, values: dict, state: str = ""):
        for key, _, _ in self.FEATURES:
            value = values.get(key)
            if value is None:
                self._history[key].append(0.0)
            else:
                try:
                    self._history[key].append(float(value))
                except (TypeError, ValueError):
                    self._history[key].append(0.0)
        if state:
            self._state = state
        self.update()

    def reset(self):
        for values in self._history.values():
            values.clear()
        self._state = "—"
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        rect = self.rect().adjusted(1, 1, -1, -1)
        painter.setPen(QPen(QColor(COLORS["border"]), 1))
        painter.setBrush(QColor("#F8FAFD"))
        painter.drawRoundedRect(rect, 12, 12)

        painter.setFont(QFont("Microsoft YaHei", 11, QFont.Bold))
        painter.setPen(QColor(COLORS["ink"]))
        painter.drawText(rect.adjusted(12, 8, -12, 0), Qt.AlignTop | Qt.AlignLeft,
                         f"{self._channel_name}  ·  状态 {self._state}")

        plot = QRectF(rect).adjusted(12, 36, -12, -10)
        band_h = plot.height() / max(1, len(self.FEATURES))
        painter.setFont(QFont("Microsoft YaHei", 9))

        for index, (key, label, color) in enumerate(self.FEATURES):
            top = plot.top() + index * band_h
            bottom = top + band_h - 6
            mid = (top + bottom) / 2.0
            painter.setPen(QPen(QColor("#E5EAF2"), 1))
            painter.drawLine(int(plot.left()), int(bottom), int(plot.right()), int(bottom))
            painter.drawLine(int(plot.left()), int(mid), int(plot.right()), int(mid))

            points = list(self._history[key])
            last_value = points[-1] if points else 0.0
            painter.setPen(QColor(COLORS["muted"]))
            painter.drawText(
                int(plot.left()), int(top + 14),
                f"{label}: {last_value:.3g}")
            if len(points) < 2:
                continue

            min_v = min(points)
            max_v = max(points)
            span = max(max_v - min_v, 1e-9)
            usable_h = max(1.0, bottom - top - 18)
            step = plot.width() / max(1, len(points) - 1)
            painter.setPen(QPen(QColor(color), 1.8))
            last_x = plot.left()
            last_y = bottom - 4 - ((points[0] - min_v) / span) * usable_h
            for i, value in enumerate(points[1:], start=1):
                x = plot.left() + i * step
                y = bottom - 4 - ((value - min_v) / span) * usable_h
                painter.drawLine(int(last_x), int(last_y), int(x), int(y))
                last_x, last_y = x, y


class EmgFeatureDialog(QDialog):
    """Popup dialog for two-channel EMG feature trends."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("肌电特征实时波形")
        self.resize(980, 620)
        self._last_seq = -1

        root = QVBoxLayout(self)
        root.setContentsMargins(18, 16, 18, 18)
        root.setSpacing(12)

        header = QHBoxLayout()
        title_box = QVBoxLayout()
        title = QLabel("肌电特征实时波形")
        title.setObjectName("sectionTitle")
        hint = CaptionLabel(
            "左右分别显示通道一 / 通道二；每个通道展示均方根强度、过零率、波动系数、疲劳指数与包络均值趋势")
        hint.setStyleSheet(f"color:{COLORS['muted']};")
        title_box.addWidget(title)
        title_box.addWidget(hint)
        header.addLayout(title_box, 1)
        self._sample_rate = QLabel("采样率：—")
        self._sample_rate.setStyleSheet(pill_style("primary"))
        header.addWidget(self._sample_rate)
        root.addLayout(header)

        self._status = CaptionLabel("肌电状态：等待数据")
        self._status.setWordWrap(True)
        self._status.setStyleSheet(f"color:{COLORS['muted']};")
        root.addWidget(self._status)

        grid = QGridLayout()
        grid.setHorizontalSpacing(14)
        self._channels = [
            EmgFeatureWaveform("通道一", self),
            EmgFeatureWaveform("通道二", self),
        ]
        grid.addWidget(self._channels[0], 0, 0)
        grid.addWidget(self._channels[1], 0, 1)
        grid.setColumnStretch(0, 1)
        grid.setColumnStretch(1, 1)
        root.addLayout(grid, 1)

        footer = CaptionLabel("注：当前展示的是上位机接收到的肌电特征帧，不生成模拟肌电数据。")
        footer.setStyleSheet(f"color:{COLORS['muted']};")
        root.addWidget(footer)

    def update_frame(self, frame: PreviewFrame):
        if frame.seq == self._last_seq:
            return
        self._last_seq = frame.seq
        rate = int(getattr(frame, "emg_sample_rate_hz", 0) or 0)
        self._sample_rate.setText(f"采样率：{rate} Hz" if rate > 0 else "采样率：—")
        self._status.setText(f"肌电状态：{frame.emg_status or '未接入'}")

        states = list(getattr(frame, "emg_state", []) or [])
        for index, widget in enumerate(self._channels):
            values = {
                "rms": _value_at(frame.emg_rms, index),
                "zcr": _value_at(getattr(frame, "emg_zcr", []), index),
                "cv": _value_at(getattr(frame, "emg_cv", []), index),
                "fatigue": _value_at(frame.emg_fatigue_index, index),
                "envelope": _value_at(getattr(frame, "emg_envelope_mean", []), index),
            }
            widget.add_values(
                values,
                states[index] if index < len(states) else "",
            )


def _value_at(values, index: int):
    return values[index] if values and index < len(values) else None


class EmgPanel(SimpleCardWidget):
    """Compact EMG status display."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._last_frame: Optional[PreviewFrame] = None
        self._feature_dialog: Optional[EmgFeatureDialog] = None
        self._init_ui()

    def _init_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 12, 16, 12)
        layout.setSpacing(7)

        header = QHBoxLayout()
        title = QLabel("肌电监测")
        title.setObjectName("sectionTitle")
        header.addWidget(title)
        header.addStretch()
        self._btn_features = PushButton("查看肌电曲线")
        self._btn_features.setFixedHeight(28)
        self._btn_features.setToolTip("弹出大窗口查看两个肌电通道的实时特征曲线")
        self._btn_features.clicked.connect(self._open_feature_dialog)
        header.addWidget(self._btn_features)
        self._status_dot = QLabel("未连接")
        self._status_dot.setStyleSheet(pill_style("neutral"))
        header.addWidget(self._status_dot)
        layout.addLayout(header)

        self._status_label = CaptionLabel("未接入肌电数据")
        self._status_label.setStyleSheet(f"color:{COLORS['muted']};")
        self._status_label.setWordWrap(True)
        layout.addWidget(self._status_label)

        self._waveform = EmgWaveform(self)
        self._waveform.setToolTip("小波形展示最近一段时间两个肌电通道的均方根强度变化，用于快速观察当前发力是否稳定。")
        layout.addWidget(self._waveform)
        waveform_hint = CaptionLabel("小波形：最近肌电均方根强度变化，蓝色为通道一，绿色为通道二")
        waveform_hint.setStyleSheet(f"color:{COLORS['muted']};")
        waveform_hint.setWordWrap(True)
        layout.addWidget(waveform_hint)

        self._ch_bars = []
        for ch_name in ["通道一", "通道二"]:
            row = QHBoxLayout()
            row.addWidget(BodyLabel(ch_name))
            bar = QProgressBar()
            bar.setRange(0, 3000)
            bar.setValue(0)
            bar.setTextVisible(False)
            row.addWidget(bar, 1)
            val = CaptionLabel("—")
            val.setFixedWidth(50)
            val.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
            row.addWidget(val)
            layout.addLayout(row)
            self._ch_bars.append((bar, val))

        self._fatigue_label = CaptionLabel("疲劳指数：—")
        self._fatigue_label.setStyleSheet(f"color:{COLORS['muted']};")
        layout.addWidget(self._fatigue_label)

    def set_frame(self, frame: PreviewFrame):
        """Update EMG display from a PreviewFrame."""
        self._last_frame = frame
        status = frame.emg_status or "肌电未接入"
        lowered = status.lower()

        if any(token in lowered for token in ("waiting", "error", "failed", "mock")):
            self._status_label.setText("肌电：等待真实设备数据")
            self._status_label.setStyleSheet("color: #B45309;")
            self._status_dot.setText("等待中")
            self._status_dot.setStyleSheet(pill_style("warning"))
        elif "real" in lowered or "connected" in lowered:
            self._status_label.setText("肌电：真实链路正常")
            self._status_label.setStyleSheet("color: #27AE60;")
            self._status_dot.setText("已连接")
            self._status_dot.setStyleSheet(pill_style("success"))
        elif "disabled" in lowered or not status.strip():
            self._status_label.setText("肌电：未启用")
            self._status_label.setStyleSheet("color: #9AA8B4;")
            self._status_dot.setText("未启用")
            self._status_dot.setStyleSheet(pill_style("neutral"))
        else:
            self._status_label.setText(status[:80])
            self._status_label.setStyleSheet("color: #1F2933;")

        rms_values = frame.emg_rms or []
        self._waveform.add_sample(rms_values)
        fatigue_values = frame.emg_fatigue_index or []
        if fatigue_values:
            fatigue_text = " / ".join(f"{float(v):.2f}" for v in fatigue_values[:2])
            self._fatigue_label.setText(f"疲劳指数：{fatigue_text}")
        else:
            self._fatigue_label.setText("疲劳指数：—")
        for i, (bar, val_label) in enumerate(self._ch_bars):
            if i < len(rms_values):
                v = int(rms_values[i])
                bar.setValue(min(v, 3000))
                val_label.setText(f"{v:.1f}")
            else:
                bar.setValue(0)
                val_label.setText("—")
        if self._feature_dialog is not None:
            self._feature_dialog.update_frame(frame)

    def _open_feature_dialog(self):
        if self._feature_dialog is None:
            self._feature_dialog = EmgFeatureDialog(self.window())
        if self._last_frame is not None:
            self._feature_dialog.update_frame(self._last_frame)
        self._feature_dialog.show()
        self._feature_dialog.raise_()
        self._feature_dialog.activateWindow()
