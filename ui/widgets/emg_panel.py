"""
EMG status panel — displays EMG connection status and per-channel metrics.
Replaces the EMG section in TrainingPage.cpp.
"""

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QLabel, QProgressBar

from qfluentwidgets import SimpleCardWidget, StrongBodyLabel, BodyLabel, CaptionLabel

from rehab_engine.preview import PreviewFrame
from ..theme import COLORS, pill_style


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

        # Per-channel RMS bars
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

    def set_frame(self, frame: PreviewFrame):
        """Update EMG display from a PreviewFrame."""
        status = frame.emg_status or "肌电未接入"

        # Parse EMG status text
        if "mock" in status.lower():
            self._status_label.setText("肌电：模拟模式（演示数据）")
            self._status_label.setStyleSheet("color: #B45309;")
            self._status_dot.setText("模拟")
            self._status_dot.setStyleSheet(pill_style("warning"))
        elif "real" in status.lower():
            self._status_label.setText("肌电：真实链路正常")
            self._status_label.setStyleSheet("color: #27AE60;")
            self._status_dot.setText("已连接")
            self._status_dot.setStyleSheet(pill_style("success"))
        elif "disabled" in status.lower() or not status.strip():
            self._status_label.setText("肌电：未启用")
            self._status_label.setStyleSheet("color: #9AA8B4;")
            self._status_dot.setText("未启用")
            self._status_dot.setStyleSheet(pill_style("neutral"))
        else:
            self._status_label.setText(status[:80])
            self._status_label.setStyleSheet("color: #1F2933;")

        # Update RMS bars
        rms_values = frame.emg_rms or []
        for i, (bar, val_label) in enumerate(self._ch_bars):
            if i < len(rms_values):
                v = int(rms_values[i])
                bar.setValue(min(v, 3000))
                val_label.setText(f"{v:.1f}")
            else:
                bar.setValue(0)
                val_label.setText("—")
