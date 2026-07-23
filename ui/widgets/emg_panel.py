"""
EMG status panel — displays EMG connection status and per-channel metrics.
Replaces the EMG section in TrainingPage.cpp.
"""

from collections import deque

from PyQt5.QtCore import QRectF, Qt
from PyQt5.QtGui import QColor, QPainter, QPen
from PyQt5.QtWidgets import QHBoxLayout, QLabel, QProgressBar, QVBoxLayout, QWidget

from qfluentwidgets import BodyLabel, CaptionLabel, SimpleCardWidget

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
        labels = ["CH1", "CH2"]
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


class EmgPanel(SimpleCardWidget):
    """Compact EMG status display."""

    def __init__(self, parent=None):
        super().__init__(parent)
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
        self._status_dot = QLabel("未连接")
        self._status_dot.setStyleSheet(pill_style("neutral"))
        header.addWidget(self._status_dot)
        layout.addLayout(header)

        self._status_label = CaptionLabel("未接入肌电数据")
        self._status_label.setStyleSheet(f"color:{COLORS['muted']};")
        self._status_label.setWordWrap(True)
        layout.addWidget(self._status_label)

        self._waveform = EmgWaveform(self)
        layout.addWidget(self._waveform)

        self._ch_bars = []
        for ch_name in ["CH1", "CH2"]:
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
        status = frame.emg_status or "肌电未接入"
        lowered = status.lower()

        if any(token in lowered for token in ("waiting", "error", "failed")):
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
