"""
Course system: repository and runner.
Replaces core/course/CourseRepository.cpp + CourseRunner.cpp.
Pure Python, no Qt dependency.
"""

import json
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from threading import Timer
from typing import Callable, List, Optional


# ============================================================
# Data types
# ============================================================

@dataclass
class CourseAction:
    order: int = 0
    action_id: str = ""         # M1-M10
    movement_id: str = ""       # m01-m10
    name_cn: str = ""
    name_en: str = ""
    target_reps: int = 0
    rest_sec_after: int = 0
    side_mode: str = "none"     # none / left_right


@dataclass
class Course:
    course_id: str = ""
    course_name: str = ""
    category: str = ""
    difficulty: int = 0
    estimated_minutes: int = 0
    description: str = ""
    actions: List[CourseAction] = field(default_factory=list)


# ============================================================
# Repository
# ============================================================

class CourseRepository:
    """Loads courses from a JSON config file (configs/courses.json)."""

    def __init__(self, config_path: Optional[str] = None):
        self._courses: List[Course] = []
        self._config_path = config_path or ""
        self._last_error = ""
        if config_path:
            self.load(config_path)

    def load(self, config_path: Optional[str] = None) -> bool:
        path = config_path or self._config_path
        if not path:
            # Search default locations
            for candidate in ["configs/courses.json", "../configs/courses.json"]:
                if Path(candidate).exists():
                    path = candidate
                    break
        if not path or not Path(path).exists():
            self._last_error = f"Course config not found: {path}"
            return False

        try:
            data = json.loads(Path(path).read_text(encoding="utf-8"))
            self._courses = []
            for obj in data.get("courses", []):
                course = Course(
                    course_id=obj.get("course_id", ""),
                    course_name=obj.get("course_name", ""),
                    category=obj.get("category", ""),
                    difficulty=obj.get("difficulty", 0),
                    estimated_minutes=obj.get("estimated_minutes", 0),
                    description=obj.get("description", ""),
                )
                for action_obj in obj.get("actions", []):
                    action = CourseAction(
                        order=action_obj.get("order", 0),
                        action_id=action_obj.get("action_id", ""),
                        movement_id=action_obj.get("movement_id", ""),
                        name_cn=action_obj.get("name_cn", ""),
                        name_en=action_obj.get("name_en", ""),
                        target_reps=action_obj.get("target_reps", 0),
                        rest_sec_after=action_obj.get("rest_sec_after", 0),
                        side_mode=action_obj.get("side_mode", "none"),
                    )
                    course.actions.append(action)
                self._courses.append(course)
            self._config_path = path
            return True
        except (json.JSONDecodeError, KeyError) as e:
            self._last_error = str(e)
            return False

    @property
    def courses(self) -> List[Course]:
        return list(self._courses)

    def find_by_id(self, course_id: str) -> Optional[Course]:
        cid = self._canonical(course_id)
        for c in self._courses:
            if self._canonical(c.course_id) == cid:
                return c
        return None

    @property
    def last_error(self) -> str:
        return self._last_error

    @staticmethod
    def _canonical(course_id: str) -> str:
        return course_id.strip().lower()

    @staticmethod
    def is_valid_action_id(action_id: str) -> bool:
        """Check if action_id matches the M1-M10 pattern."""
        import re
        return bool(re.match(r'^M([1-9]|10)$', action_id))

    @staticmethod
    def default_config_path() -> str:
        for p in ["configs/courses.json", "../configs/courses.json"]:
            if Path(p).exists():
                return p
        return "configs/courses.json"


# ============================================================
# Runner (state machine, replaces CourseRunner C++ QObject)
# ============================================================

class RunnerState(Enum):
    IDLE = "idle"
    TRAINING = "training"
    RESTING = "resting"
    FINISHED = "finished"


class CourseRunner:
    """
    Training state machine.
    Drives action sequencing and rest countdown.
    Callbacks replace Qt signals.
    """

    def __init__(self):
        self._course: Optional[Course] = None
        self._action_index: int = -1
        self._state: RunnerState = RunnerState.IDLE
        # Per-action scoring state
        self._actual_reps: int = 0
        self._score_sum: float = 0.0
        self._score_samples: int = 0
        # Rest timer
        self._rest_remaining: int = 0
        self._rest_timer: Optional[Timer] = None

        # Callbacks (set by the caller, replaces Qt signals)
        self.on_action_changed: Optional[Callable[[CourseAction], None]] = None
        self.on_action_completed: Optional[Callable[[CourseAction, int, float], None]] = None
        self.on_rest_started: Optional[Callable[[CourseAction, int], None]] = None
        self.on_rest_tick: Optional[Callable[[int], None]] = None
        self.on_course_finished: Optional[Callable[[], None]] = None
        self.on_state_changed: Optional[Callable[[RunnerState], None]] = None

    # --- Properties ---
    @property
    def state(self) -> RunnerState:
        return self._state

    @property
    def current_action(self) -> Optional[CourseAction]:
        if self._course and 0 <= self._action_index < len(self._course.actions):
            return self._course.actions[self._action_index]
        return None

    @property
    def current_action_index(self) -> int:
        return self._action_index

    @property
    def total_actions(self) -> int:
        return len(self._course.actions) if self._course else 0

    @property
    def rest_remaining_sec(self) -> int:
        return self._rest_remaining

    # --- State machine ---

    def start_course(self, course: Course) -> bool:
        if not course.actions:
            return False
        self._course = course
        self._action_index = 0
        self._actual_reps = 0
        self._score_sum = 0.0
        self._score_samples = 0
        self._set_state(RunnerState.TRAINING)
        self._notify_action()
        return True

    def stop_course(self) -> None:
        self._cancel_rest_timer()
        self._set_state(RunnerState.FINISHED)

    def on_score_updated(self, score: "ScoreResult") -> None:
        """
        Receive a real-time score update from ScoreBridge.
        Tracks reps and average score; triggers action completion.
        """
        if self._state != RunnerState.TRAINING:
            return
        action = self.current_action
        if action is None:
            return

        self._actual_reps = max(self._actual_reps, score.completed_count)
        self._score_sum += score.overall_score if score.count > 0 else 0.0
        self._score_samples += 1

        # Complete action when target reps reached
        if self._actual_reps >= action.target_reps > 0:
            self._complete_current_action()

    def _complete_current_action(self) -> None:
        action = self.current_action
        if action is None:
            return
        avg_score = self._score_sum / max(1, self._score_samples)

        if self.on_action_completed:
            self.on_action_completed(action, self._actual_reps, avg_score)

        # Check if this is the last action
        if self._action_index + 1 >= self.total_actions:
            self._set_state(RunnerState.FINISHED)
            if self.on_course_finished:
                self.on_course_finished()
            return

        # Enter rest period before next action
        if action.rest_sec_after > 0:
            self._set_state(RunnerState.RESTING)
            self._rest_remaining = action.rest_sec_after
            if self.on_rest_started:
                self.on_rest_started(action, action.rest_sec_after)
            self._start_rest_timer()
        else:
            self._advance_to_next()

    def _advance_to_next(self) -> None:
        self._action_index += 1
        self._actual_reps = 0
        self._score_sum = 0.0
        self._score_samples = 0
        self._set_state(RunnerState.TRAINING)
        self._notify_action()

    def _notify_action(self) -> None:
        action = self.current_action
        if action and self.on_action_changed:
            self.on_action_changed(action)

    # --- Rest timer ---

    def _start_rest_timer(self) -> None:
        self._cancel_rest_timer()
        self._rest_tick()

    def _rest_tick(self) -> None:
        if self._state != RunnerState.RESTING:
            return
        if self._rest_remaining <= 0:
            self._advance_to_next()
            return
        if self.on_rest_tick:
            self.on_rest_tick(self._rest_remaining)
        self._rest_remaining -= 1
        self._rest_timer = Timer(1.0, self._rest_tick)
        self._rest_timer.daemon = True
        self._rest_timer.start()

    def _cancel_rest_timer(self) -> None:
        if self._rest_timer:
            self._rest_timer.cancel()
            self._rest_timer = None

    def _set_state(self, state: RunnerState) -> None:
        if state != self._state:
            self._state = state
            if self.on_state_changed:
                self.on_state_changed(state)


# Re-export ScoreResult type hint (circular import avoidance)
from .scoring import ScoreResult  # noqa: E402