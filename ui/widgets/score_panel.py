"""
Score panel — displays real-time 6-dimension scoring results.
Replaces the score GridLayout in TrainingPage.cpp.
"""

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QVBoxLayout, QHBoxLayout, QLabel, QProgressBar

from qfluentwidgets import SimpleCardWidget, StrongBodyLabel, BodyLabel, CaptionLabel

from rehab_engine.scoring import ScoreResult
from ..theme import COLORS, pill_style


class ScorePanel(SimpleCardWidget):
    """Compact score display with 6 sub-scores."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._init_ui()

    def _init_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 12, 16, 12)
        layout.setSpacing(7)

        header = QHBoxLayout()
        title_box = QVBoxLayout()
        title = QLabel("实时训练质量")
        title.setObjectName("sectionTitle")
        hint = QLabel("综合动作完成度")
        hint.setObjectName("sectionHint")
        title_box.addWidget(title)
        title_box.addWidget(hint)
        header.addLayout(title_box)
        header.addStretch()
        self._overall_label = BodyLabel("—")
        self._overall_label.setAlignment(Qt.AlignCenter)
        self._overall_label.setMinimumSize(54, 40)
        self._overall_label.setStyleSheet(
            f"color:{COLORS['primary']}; background:{COLORS['primary_soft']}; "
            "border-radius:12px; font-size:22px; font-weight:700;")
        header.addWidget(self._overall_label)
        layout.addLayout(header)

        # Counter row
        counter_row = QHBoxLayout()
        counter_row.addWidget(CaptionLabel("动作计数"))
        counter_row.addStretch()
        self._count_label = BodyLabel("0 / —")
        self._count_label.setStyleSheet(pill_style("success"))
        counter_row.addWidget(self._count_label)
        layout.addLayout(counter_row)

        # Sub-score bars make relative quality readable at a glance.
        self._scores = {}
        self._bars = {}
        items = [
            ("amplitude", "幅度"), ("smoothness", "平滑性"),
            ("trunk", "躯干稳定"), ("symmetry", "对称性"),
            ("rhythm", "节奏性"),
        ]
        for key, name in items:
            row = QHBoxLayout()
            row.setSpacing(8)
            lbl_name = CaptionLabel(name)
            lbl_name.setFixedWidth(54)
            bar = QProgressBar()
            bar.setRange(0, 100)
            bar.setValue(0)
            bar.setTextVisible(False)
            lbl_val = BodyLabel("—")
            lbl_val.setFixedWidth(34)
            lbl_val.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
            row.addWidget(lbl_name)
            row.addWidget(bar, 1)
            row.addWidget(lbl_val)
            layout.addLayout(row)
            self._scores[key] = lbl_val
            self._bars[key] = bar

        self._target = 0

    def set_target(self, target: int):
        self._target = max(0, int(target))
        self._count_label.setText(f"0 / {self._target or '—'}")

    def set_score(self, result: ScoreResult):
        """Update all score displays from a ScoreResult."""
        count = result.count
        self._count_label.setText(f"{count} / {self._target or '—'}")
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
                self._bars[key].setValue(max(0, min(100, int(attr))))

    def set_display_count(self, count: int):
        """Apply the original UI's one-step jump guard to the visible count."""
        self._count_label.setText(f"{max(0, int(count))} / {self._target or '—'}")

    def reset(self):
        """Reset all scores to default."""
        self._count_label.setText(f"0 / {self._target or '—'}")
        self._overall_label.setText("—")
        for key, lbl in self._scores.items():
            lbl.setText("—")
            self._bars[key].setValue(0)
