"""
Debug panel — real-time waveform of segmentation feature with peak/center markers.

Connects to ScoreBridge debug_state to visualize:
  - The segmentation feature signal (detrended, smoothed)
  - Detected peaks
  - Accepted centers
  - Internal parameters
  - FPS calibration status

Adds a toggle button to TrainingPage to hide/show this panel.
"""

from __future__ import annotations

from typing import Optional

from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtWidgets import (
    QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QCheckBox,
)

from qfluentwidgets import (
    SimpleCardWidget, BodyLabel, CaptionLabel,
)

from ..theme import COLORS, pill_style

import numpy as np

# Matplotlib backend must be set before any other matplotlib import.
try:
    import matplotlib
    matplotlib.use("Qt5Agg")
    from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
    from matplotlib.figure import Figure
except Exception:
    FigureCanvasQTAgg = None  # fallback when matplotlib is missing


class DebugPanel(SimpleCardWidget):
    """Real-time debug panel for the scoring engine state."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._signal_data: list[float] = []
        self._peaks: list[int] = []
        self._accepted_peaks: list[int] = []
        self._centers: list[int] = []
        self._debug_state_raw: dict = {}
        self._fs: float = 20.0
        self._n_frames: int = 0
        self._completed_count: int = 0
        self._count: int = 0

        self._init_ui()

        # Auto-refresh timer for param labels (lightweight, no subprocess call)
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._tick_labels)
        self._timer.start(100)  # 10 Hz

    def _init_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 12, 16, 12)
        layout.setSpacing(8)

        # ── Header ──
        header = QHBoxLayout()
        title = QLabel("评分引擎调试")
        title.setObjectName("sectionTitle")
        header.addWidget(title)
        header.addStretch()
        self._refresh_btn = QPushButton("刷新状态")
        self._refresh_btn.setFixedHeight(28)
        self._refresh_btn.clicked.connect(self.request_refresh)
        header.addWidget(self._refresh_btn)
        layout.addLayout(header)

        hint = CaptionLabel("实时显示分割特征波形与寻峰状态，点击「刷新状态」从子进程拉取最新数据")
        hint.setStyleSheet(f"color:{COLORS['muted']};")
        layout.addWidget(hint)

        # ── Matplotlib canvas ──
        if FigureCanvasQTAgg is not None:
            self._fig = Figure(figsize=(5, 2.0))
            self._fig.set_tight_layout(True)
            self._canvas = FigureCanvasQTAgg(self._fig)
            self._canvas.setMinimumHeight(120)
            layout.addWidget(self._canvas, 1)
            self._ax = self._fig.add_subplot(111)
            self._setup_axes()
        else:
            self._canvas = None
            self._ax = None
            no_plot = BodyLabel("需要 matplotlib 以显示波形图")
            no_plot.setStyleSheet(f"color:{COLORS['muted']}; padding: 16px;")
            no_plot.setAlignment(Qt.AlignCenter)
            layout.addWidget(no_plot)

        # ── Status row ──
        row1 = QHBoxLayout()
        row1.setSpacing(16)
        self._nframes_label = CaptionLabel("帧数: —")
        self._count_label = CaptionLabel("计数: —")
        self._completed_label = CaptionLabel("完成: —")
        self._fps_label = CaptionLabel("帧率：—")
        for lbl in (self._nframes_label, self._count_label,
                    self._completed_label, self._fps_label):
            lbl.setStyleSheet(f"color:{COLORS['text']}; font-weight:600;")
            row1.addWidget(lbl)
        row1.addStretch()
        layout.addLayout(row1)

        # ── Parameters ──
        row2 = QHBoxLayout()
        row2.setSpacing(16)
        self._feat_label = CaptionLabel("分割特征: —")
        self._pkcf_label = CaptionLabel("确认帧数: —")
        self._proem_label = CaptionLabel("峰显著比: —")
        self._div_label = CaptionLabel("计数除数: —")
        for lbl in (self._feat_label, self._pkcf_label,
                    self._proem_label, self._div_label):
            lbl.setStyleSheet(f"color:{COLORS['text']};")
            row2.addWidget(lbl)
        row2.addStretch()
        layout.addLayout(row2)

        # ── Metric history row ──
        self._metric_label = CaptionLabel("指标历史: —")
        self._metric_label.setStyleSheet(f"color:{COLORS['muted']};")
        self._metric_label.setWordWrap(True)
        layout.addWidget(self._metric_label)

    def _setup_axes(self):
        if self._ax is None:
            return
        self._ax.set_facecolor("#F8FAFD")
        self._ax.set_xlabel("帧序号")
        self._ax.set_ylabel("Feature value")
        self._ax.grid(True, alpha=0.3)

    # ── Public API ──

    def request_refresh(self):
        """Signal the parent TrainingPage to call ScoreBridge.request_debug_state()."""
        self._refresh_btn.setEnabled(False)
        self._refresh_btn.setText("请求中…")
        on_refresh = getattr(self, "_on_debug_refresh_cb", None)
        if on_refresh:
            on_refresh()
        QTimer.singleShot(2500, self._restore_refresh_button)

    def _restore_refresh_button(self):
        if not self._refresh_btn.isEnabled():
            self._refresh_btn.setEnabled(True)
            self._refresh_btn.setText("刷新状态")

    def set_refresh_callback(self, callback):
        """Set a callable that the panel invokes when the user clicks refresh."""
        self._on_debug_refresh_cb = callback

    def set_debug_state(self, debug_state: dict):
        """Update all displays from a debug_state dict."""
        self._debug_state_raw = debug_state
        self._n_frames = debug_state.get("n_frames", 0)
        self._count = debug_state.get("count", 0)
        self._completed_count = debug_state.get("completed_count", 0)
        params = debug_state.get("params", {})

        # Signal data
        signal = debug_state.get("segment_signal")
        peaks = debug_state.get("detected_peaks", [])
        accepted_peaks = debug_state.get("accepted_peaks", [])
        centers = debug_state.get("accepted_centers", [])

        # Update stored data
        if signal is not None:
            self._signal_data = list(signal)
        else:
            self._signal_data = []
        self._peaks = [int(p) for p in (peaks or [])]
        self._accepted_peaks = [int(p) for p in (accepted_peaks or [])]
        self._centers = [int(c) for c in (centers or [])]

        # Redraw
        self._redraw_plot()

        # Refresh labels via _tick_labels
        self._tick_labels()

        # Re-enable button
        self._refresh_btn.setEnabled(True)
        self._refresh_btn.setText("刷新状态")

    def set_fps_info(self, calibrated_fps: float, skeleton_fps: float):
        """Set FPS info without a full debug state refresh."""
        self._fs = calibrated_fps if calibrated_fps > 0 else skeleton_fps

    def _redraw_plot(self):
        if self._ax is None:
            return
        self._ax.clear()
        self._setup_axes()

        data = np.asarray(self._signal_data, dtype=float)
        if len(data) == 0:
            self._ax.set_title("无信号数据（点击刷新）")
            self._canvas.draw_idle()
            return

        x = np.arange(len(data))
        self._ax.plot(x, data, color=COLORS["primary"], linewidth=1.2, label="分割特征曲线")

        # Overlay detected peaks (all)
        if len(self._peaks) > 0:
            valid_peaks = [p for p in self._peaks if 0 <= p < len(data)]
            if valid_peaks:
                self._ax.scatter(
                    valid_peaks, data[valid_peaks],
                    color="#93C5FD", marker="v", s=30, zorder=5,
                    label=f"检测峰值（{len(valid_peaks)}）",
                )

        # Overlay accepted peaks (confirmed)
        if len(self._accepted_peaks) > 0:
            valid_ap = [p for p in self._accepted_peaks if 0 <= p < len(data)]
            if valid_ap:
                self._ax.scatter(
                    valid_ap, data[valid_ap],
                    color=COLORS["success"], marker="v", s=50, zorder=6,
                    label=f"计数峰值（{len(valid_ap)}）",
                )

        # Overlay accepted centers
        if len(self._centers) > 0:
            valid_centers = [c for c in self._centers if 0 <= c < len(data)]
            if valid_centers:
                self._ax.scatter(
                    valid_centers, data[valid_centers],
                    color=COLORS["danger"], marker="*", s=80, zorder=7,
                    label=f"动作中心（{len(valid_centers)}）",
                )

        self._ax.legend(fontsize=7, loc="upper right")
        self._canvas.draw_idle()

    def _tick_labels(self):
        """Update lightweight label fields (called by timer, no subprocess)."""
        if self._n_frames:
            self._nframes_label.setText(f"帧数: {self._n_frames}")
            self._count_label.setText(f"计数: {self._count}")
            self._completed_label.setText(f"完成: {self._completed_count}")

        # Params from last debug state
        debug = self._debug_state_raw
        params = debug.get("params", {})
        feat = debug.get("segment_feature_name", "")
        if feat:
            self._feat_label.setText(f"分割特征: {feat}")

        pkcf = params.get("peak_confirm_frames")
        if pkcf is not None:
            self._pkcf_label.setText(f"确认帧数: {pkcf}")

        proem = params.get("prominence_ratio")
        if proem is not None:
            self._proem_label.setText(f"峰显著比: {proem:.2f}")

        div = params.get("count_divisor")
        if div is not None:
            self._div_label.setText(f"计数除数: {div}")

        # Metric history summary
        mh = debug.get("metric_history", {})
        if mh:
            parts = []
            for key in ("amplitude_metric_rom_deg", "smoothness_metric",
                        "trunk_stability_metric", "symmetry_metric_deg"):
                vals = mh.get(key, [])
                if vals:
                    mean_val = np.mean(vals[-10:])
                    parts.append(f"{key.split('_')[0]}({len(vals)}):{mean_val:.1f}")
            self._metric_label.setText("指标历史: " + " | ".join(parts))
        else:
            self._metric_label.setText("指标历史: 无")

    def reset(self):
        """Clear all debug data."""
        self._signal_data = []
        self._peaks = []
        self._accepted_peaks = []
        self._centers = []
        self._debug_state_raw = {}
        self._n_frames = 0
        self._count = 0
        self._completed_count = 0
        if self._ax is not None:
            self._ax.clear()
            self._setup_axes()
            self._canvas.draw_idle()
        self._tick_labels()

    def closeEvent(self, event):
        self._timer.stop()
        super().closeEvent(event)
