"""
Score panel — displays real-time 6-dimension scoring results.
Replaces the score GridLayout in TrainingPage.cpp.
"""

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QLabel, QGridLayout

from qfluentwidgets import SimpleCardWidget, StrongBodyLabel, BodyLabel, CaptionLabel

from rehab_engine.scoring import ScoreResult


class ScorePanel(SimpleCardWidget):
    """Compact score display with 6 sub-scores."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._init_ui()

    def _init_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(14, 12, 14, 12)
        layout.setSpacing(8)

        layout.addWidget(StrongBodyLabel("实时评分"))

        # Counter row
        counter_row = QHBoxLayout()
        self._count_label = BodyLabel("0 / 0")
        counter_row.addWidget(BodyLabel("计数:"))
        counter_row.addWidget(self._count_label)
        counter_row.addStretch()
        self._overall_label = BodyLabel("—")
        self._overall_label.setStyleSheet("color:#2F80ED; font-size:22px; font-weight:700;")
        counter_row.addWidget(self._overall_label)
        layout.addLayout(counter_row)

        # Sub-scores grid
        grid = QGridLayout()
        grid.setSpacing(6)
        self._scores = {}
        items = [
            ("amplitude", "幅度"), ("smoothness", "平滑性"),
            ("trunk", "躯干稳定"), ("symmetry", "对称性"),
            ("rhythm", "节奏性"),
        ]
        for i, (key, name) in enumerate(items):
            row, col = divmod(i, 2)
            lbl_name = CaptionLabel(name)
            lbl_val = BodyLabel("—")
            lbl_val.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
            grid.addWidget(lbl_name, row, col * 2)
            grid.addWidget(lbl_val, row, col * 2 + 1)
            self._scores[key] = lbl_val

        layout.addLayout(grid)

    def set_score(self, result: ScoreResult):
        """Update all score displays from a ScoreResult."""
        self._count_label.setText(f"{result.count}")
        self._overall_label.setText(
            f"{result.overall_score:.1f}" if result.overall_score > 0 else "—")

        for key, attr in [
            ("amplitude", result.amplitude_score),
            ("smoothness", result.smoothness_score),
            ("trunk", result.trunk_score),
            ("symmetry", result.symmetry_score),
            ("rhythm", result.rhythm_score),
        ]:
            if key in self._scores:
                self._scores[key].setText(
                    f"{attr:.1f}" if attr > 0 else "—")

    def reset(self):
        """Reset all scores to default."""
        self._count_label.setText("0 / 0")
        self._overall_label.setText("—")
        for lbl in self._scores.values():
            lbl.setText("—")