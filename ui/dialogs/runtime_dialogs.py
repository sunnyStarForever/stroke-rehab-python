"""Non-blocking runtime log and performance views from the original client."""

from dataclasses import dataclass

from PyQt5.QtWidgets import (
    QComboBox,
    QDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QPlainTextEdit,
    QVBoxLayout,
)


@dataclass(frozen=True)
class LogEntry:
    timestamp: str
    level: str
    message: str


class LogDialog(QDialog):
    _LEVELS = {
        "全部": None,
        "信息": "INFO",
        "警告": "WARN",
        "错误": "ERROR",
        "性能": "PERF",
    }

    def __init__(self, parent=None):
        super().__init__(parent)
        self._entries = []
        self.setWindowTitle("运行日志")
        self.resize(840, 520)
        root = QVBoxLayout(self)
        row = QHBoxLayout()
        row.addWidget(QLabel("级别筛选"))
        self._level_combo = QComboBox()
        self._level_combo.addItems(self._LEVELS)
        self._level_combo.currentIndexChanged.connect(self.refresh_view)
        row.addWidget(self._level_combo)
        row.addStretch()
        root.addLayout(row)
        self._log_edit = QPlainTextEdit()
        self._log_edit.setReadOnly(True)
        self._log_edit.setMaximumBlockCount(2000)
        root.addWidget(self._log_edit, 1)

    @property
    def entries(self):
        return list(self._entries)

    def set_entries(self, entries):
        self._entries = list(entries)[-2000:]
        self.refresh_view()

    def append_entry(self, entry: LogEntry):
        self._entries.append(entry)
        if len(self._entries) > 2000:
            del self._entries[:len(self._entries) - 2000]
        if self._accepts(entry):
            self._log_edit.appendPlainText(self._format(entry))
            bar = self._log_edit.verticalScrollBar()
            bar.setValue(bar.maximum())

    def _accepts(self, entry):
        selected = self._LEVELS.get(self._level_combo.currentText())
        return selected is None or entry.level == selected

    @staticmethod
    def _format(entry):
        return f"[{entry.timestamp}] [{entry.level}] {entry.message}"

    def refresh_view(self):
        self._log_edit.setPlainText("\n".join(
            self._format(entry) for entry in self._entries if self._accepts(entry)))
        bar = self._log_edit.verticalScrollBar()
        bar.setValue(bar.maximum())


class PerformanceDialog(QDialog):
    _FIELDS = (
        ("rgb_fps", "RGB FPS", "{:.1f}"),
        ("depth_fps", "Depth FPS", "{:.1f}"),
        ("pair_fps", "同步 Pair FPS", "{:.1f}"),
        ("pose_fps", "Pose FPS", "{:.1f}"),
        ("yolo_ms", "YOLO 耗时", "{:.1f} ms"),
        ("pose_ms", "Pose 耗时", "{:.1f} ms"),
        ("queue_length", "队列长度", "{}"),
        ("dropped_pairs", "丢帧数", "{}"),
        ("record_write_ms", "录制写盘耗时", "{:.1f} ms"),
    )

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("性能监控")
        self.resize(380, 330)
        layout = QVBoxLayout(self)
        form = QFormLayout()
        self._labels = {}
        for key, caption, _ in self._FIELDS:
            label = QLabel("-")
            form.addRow(caption, label)
            self._labels[key] = label
        layout.addLayout(form)
        layout.addStretch()

    def set_snapshot(self, snapshot):
        for key, _, pattern in self._FIELDS:
            value = snapshot.get(key, 0)
            try:
                text = pattern.format(value)
            except (TypeError, ValueError):
                text = str(value)
            self._labels[key].setText(text)


__all__ = ["LogDialog", "LogEntry", "PerformanceDialog"]
