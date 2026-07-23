"""Patient home page migrated from the original Qt/C++ patient client."""

import json
from pathlib import Path

from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtWidgets import (
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from qfluentwidgets import CardWidget, PrimaryPushButton, SimpleCardWidget

from rehab_engine.course import Course, CourseRepository
from ..theme import COLORS, PAGE_STYLE, pill_style


class PatientInfoCard(SimpleCardWidget):
    """Read-only patient summary backed by ``PipelineConfig``."""

    def __init__(self, config, parent=None):
        super().__init__(parent)
        self._config = config
        layout = QVBoxLayout(self)
        layout.setContentsMargins(18, 18, 18, 18)
        layout.setSpacing(10)
        title = QLabel("患者信息")
        title.setObjectName("sectionTitle")
        layout.addWidget(title)

        grid = QGridLayout()
        grid.setHorizontalSpacing(10)
        grid.setVerticalSpacing(8)
        self._values = {}
        for row, (key, label) in enumerate((
            ("name", "姓名"),
            ("gender", "性别"),
            ("age", "年龄"),
            ("diagnosis", "诊断"),
            ("patient_id", "编号"),
        )):
            caption = QLabel(label)
            caption.setStyleSheet(f"color:{COLORS['muted']};")
            value = QLabel("-")
            value.setWordWrap(True)
            grid.addWidget(caption, row, 0, Qt.AlignTop)
            grid.addWidget(value, row, 1)
            self._values[key] = value
        grid.setColumnStretch(1, 1)
        layout.addLayout(grid)
        self.refresh()

    def refresh(self):
        config = self._config
        self._values["name"].setText(config.patient_name or "未填写")
        self._values["gender"].setText(config.patient_gender or "未填写")
        self._values["age"].setText(
            f"{config.patient_age} 岁" if config.patient_age > 0 else "未填写")
        self._values["diagnosis"].setText(config.patient_diagnosis or "未填写")
        self._values["patient_id"].setText(config.patient_id or "未填写")


class CourseCard(CardWidget):
    start_requested = pyqtSignal(str)

    def __init__(self, course: Course, selected=False, parent=None):
        super().__init__(parent)
        self.course = course
        self.setMinimumWidth(300)
        self.setMaximumWidth(460)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 18, 20, 18)
        layout.setSpacing(10)

        header = QHBoxLayout()
        title = QLabel(course.course_name)
        title.setObjectName("sectionTitle")
        title.setWordWrap(True)
        header.addWidget(title, 1)
        selected_label = QLabel("当前课程")
        selected_label.setStyleSheet(pill_style("success" if selected else "neutral"))
        selected_label.setVisible(selected)
        header.addWidget(selected_label)
        layout.addLayout(header)

        description = QLabel(course.description or "康复训练课程")
        description.setWordWrap(True)
        description.setStyleSheet(f"color:{COLORS['muted']};")
        layout.addWidget(description)

        action_text = "、".join(
            f"{action.name_cn} × {action.target_reps}" for action in course.actions)
        actions = QLabel(f"动作内容：{action_text}")
        actions.setWordWrap(True)
        layout.addWidget(actions)

        difficulty = "★" * max(0, min(5, course.difficulty))
        difficulty += "☆" * (5 - max(0, min(5, course.difficulty)))
        details = QLabel(
            f"难度：{difficulty}    预计用时：{max(0, course.estimated_minutes)} 分钟")
        details.setStyleSheet(f"color:{COLORS['muted']};")
        layout.addWidget(details)
        layout.addStretch()

        button = PrimaryPushButton("进入训练")
        button.setMinimumHeight(40)
        button.clicked.connect(lambda: self.start_requested.emit(course.course_id))
        layout.addWidget(button)


class PatientHomePage(QWidget):
    """Patient summary, real local history and course entry point."""

    course_selected = pyqtSignal(str)
    report_requested = pyqtSignal(str, str)

    def __init__(self, config, parent=None):
        super().__init__(parent)
        self._config = config
        self._course_repo = CourseRepository()
        self._course_repo.load()
        self._history_entries = []
        self.setObjectName("patientHomePage")
        self.setStyleSheet(PAGE_STYLE)

        root = QHBoxLayout(self)
        root.setContentsMargins(24, 20, 24, 24)
        root.setSpacing(20)

        sidebar = QFrame(self)
        sidebar.setFixedWidth(285)
        side_layout = QVBoxLayout(sidebar)
        side_layout.setContentsMargins(0, 0, 0, 0)
        side_layout.setSpacing(14)
        self._patient_card = PatientInfoCard(config, sidebar)
        side_layout.addWidget(self._patient_card)

        history_card = SimpleCardWidget(sidebar)
        history_layout = QVBoxLayout(history_card)
        history_layout.setContentsMargins(16, 16, 16, 16)
        history_layout.setSpacing(8)
        history_title = QLabel("最近训练")
        history_title.setObjectName("sectionTitle")
        self._history_list = QListWidget()
        self._history_list.itemDoubleClicked.connect(self._open_history_item)
        history_layout.addWidget(history_title)
        history_layout.addWidget(self._history_list, 1)
        side_layout.addWidget(history_card, 1)
        root.addWidget(sidebar)

        scroll = QScrollArea(self)
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        content = QWidget(scroll)
        self._content_layout = QVBoxLayout(content)
        self._content_layout.setContentsMargins(4, 0, 4, 0)
        self._content_layout.setSpacing(18)

        eyebrow = QLabel("患者康复训练")
        eyebrow.setObjectName("pageEyebrow")
        title = QLabel("患者训练首页")
        title.setObjectName("pageTitle")
        subtitle = QLabel("选择今日训练课程；训练结束后可从最近训练或报告中心查看结果。")
        subtitle.setObjectName("pageSubtitle")
        self._content_layout.addWidget(eyebrow)
        self._content_layout.addWidget(title)
        self._content_layout.addWidget(subtitle)

        self._course_host = QWidget(content)
        self._course_grid = QGridLayout(self._course_host)
        self._course_grid.setContentsMargins(0, 0, 0, 0)
        self._course_grid.setHorizontalSpacing(16)
        self._course_grid.setVerticalSpacing(16)
        self._content_layout.addWidget(self._course_host)

        advice = CardWidget(content)
        advice_layout = QVBoxLayout(advice)
        advice_layout.setContentsMargins(20, 18, 20, 18)
        advice_layout.setSpacing(8)
        advice_title = QLabel("医生建议和训练计划")
        advice_title.setObjectName("sectionTitle")
        advice_layout.addWidget(advice_title)
        for text in (
            "今日建议：按课程顺序完成训练，动作保持缓慢、稳定。",
            "注意事项：出现明显疼痛、眩晕或疲劳时，请立即停止训练。",
            "训练计划：建议每日 1 次，每次 5–10 分钟，并由专业人员调整强度。",
        ):
            label = QLabel(text)
            label.setWordWrap(True)
            advice_layout.addWidget(label)
        self._content_layout.addWidget(advice)
        self._content_layout.addStretch()
        scroll.setWidget(content)
        root.addWidget(scroll, 1)

        self.refresh_from_config()

    def refresh_from_config(self):
        self._patient_card.refresh()
        while self._course_grid.count():
            item = self._course_grid.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        courses = self._course_repo.courses
        if not courses:
            self._course_grid.addWidget(QLabel(self._course_repo.last_error), 0, 0)
        for index, course in enumerate(courses):
            card = CourseCard(
                course, course.course_id == self._config.selected_course_id,
                self._course_host)
            card.start_requested.connect(self.course_selected)
            self._course_grid.addWidget(card, index // 2, index % 2)
        self._course_grid.setColumnStretch(0, 1)
        self._course_grid.setColumnStretch(1, 1)
        self.refresh_history()

    def refresh_history(self):
        roots = []
        configured = Path(self._config.record_path).expanduser()
        if not configured.is_absolute():
            configured = Path(__file__).resolve().parents[2] / configured
        roots.extend((configured, configured / "output", configured / "sessions"))
        seen = set()
        entries = []
        for root in roots:
            if not root.exists():
                continue
            for csv_path in root.rglob("skeleton_3d.csv"):
                session = csv_path.parent.resolve()
                if session in seen:
                    continue
                seen.add(session)
                label = session.name
                meta_path = session / "session_ui_meta.json"
                if not meta_path.exists():
                    meta_path = session / "meta.json"
                if meta_path.exists():
                    try:
                        meta = json.loads(meta_path.read_text(encoding="utf-8"))
                        label = " · ".join(filter(None, (
                            str(meta.get("start_time", ""))[:16].replace("T", " "),
                            meta.get("course_name", ""),
                        ))) or label
                    except (OSError, ValueError, TypeError):
                        pass
                entries.append((session.stat().st_mtime, str(session), str(csv_path), label))
        entries.sort(reverse=True)
        self._history_entries = entries[:20]
        self._history_list.clear()
        for _, _, _, label in self._history_entries:
            self._history_list.addItem(label)
        if not self._history_entries:
            self._history_list.addItem("暂无训练记录")

    def _open_history_item(self, item):
        row = self._history_list.row(item)
        if 0 <= row < len(self._history_entries):
            _, session, csv_path, _ = self._history_entries[row]
            self.report_requested.emit(session, csv_path)


__all__ = ["CourseCard", "PatientHomePage", "PatientInfoCard"]
