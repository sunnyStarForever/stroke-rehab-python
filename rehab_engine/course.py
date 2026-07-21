"""
Course system: repository and runner.
Replaces core/course/CourseRepository.cpp + CourseRunner.cpp.
Pure Python, no Qt dependency.
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from threading import RLock, Timer
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
            package_root = Path(__file__).resolve().parent.parent
            for candidate in [
                package_root / "configs/courses.json",
                Path("configs/courses.json"),
                Path("../configs/courses.json"),
            ]:
                if Path(candidate).exists():
                    path = str(candidate)
                    break
        if not path or not Path(path).exists():
            self._last_error = f"Course config not found: {path}"
            return False

        self._courses = []
        self._last_error = ""
        try:
            data = json.loads(Path(path).read_text(encoding="utf-8"))
            entries = data.get("courses", []) if isinstance(data, dict) else data
            if not isinstance(entries, list) or not entries:
                raise ValueError(f"课程配置为空: {path}")
            loaded = [self._parse_course(obj) for obj in entries]
            self._courses = loaded
            self._config_path = path
            return True
        except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError) as e:
            self._last_error = str(e)
            self._courses = []
            return False

    def _parse_course(self, obj) -> Course:
        if not isinstance(obj, dict):
            raise ValueError("课程配置包含非对象条目")
        course = Course(
            course_id=str(obj.get("course_id", "")).strip(),
            course_name=str(obj.get("course_name", "")).strip(),
            category=str(obj.get("category", "")).strip(),
            difficulty=int(obj.get("difficulty", 0) or 0),
            estimated_minutes=int(obj.get("estimated_minutes", 0) or 0),
            description=str(obj.get("description", "")),
        )
        if not course.course_id or not course.course_name:
            raise ValueError("课程缺少 course_id 或 course_name")
        actions = obj.get("actions", [])
        if not isinstance(actions, list) or not actions:
            raise ValueError(f"课程 {course.course_id} 没有动作")
        for action_obj in actions:
            if not isinstance(action_obj, dict):
                raise ValueError(f"课程 {course.course_id} 包含非对象动作")
            side_mode = str(action_obj.get("side_mode", "none")).strip() or "none"
            action = CourseAction(
                order=int(action_obj.get("order", 0) or 0),
                action_id=str(action_obj.get("action_id", "")).strip(),
                movement_id=str(action_obj.get("movement_id", "")).strip(),
                name_cn=str(action_obj.get("name_cn", "")).strip(),
                name_en=str(action_obj.get("name_en", "")).strip(),
                target_reps=int(action_obj.get("target_reps", 0) or 0),
                rest_sec_after=int(action_obj.get("rest_sec_after", 0) or 0),
                side_mode=side_mode,
            )
            if not self.is_valid_action_id(action.action_id):
                raise ValueError(
                    f"课程 {course.course_id} 的动作 {action.action_id} 非法，P-Coder 只接受 M1-M10")
            if action.order <= 0 or action.target_reps <= 0:
                raise ValueError(
                    f"课程 {course.course_id} 的动作 {action.action_id} 缺少合法 order 或 target_reps")
            course.actions.append(action)
        course.actions.sort(key=lambda action: action.order)
        return course

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
        course_id = course_id.strip()
        aliases = {
            "shoulder_basic": "upper_limb_shoulder_rom_basic",
            "lower_limb_stability": "lower_limb_balance_transfer_basic",
        }
        return aliases.get(course_id, course_id)

    @staticmethod
    def is_valid_action_id(action_id: str) -> bool:
        """Check if action_id matches the M1-M10 pattern."""
        return bool(re.fullmatch(r'M([1-9]|10)', action_id.strip()))

    @staticmethod
    def default_config_path() -> str:
        package_root = Path(__file__).resolve().parent.parent
        for p in [package_root / "configs/courses.json",
                  Path("configs/courses.json"), Path("../configs/courses.json")]:
            if Path(p).exists():
                return str(p)
        return "configs/courses.json"


# ============================================================
# Runner (state machine, replaces CourseRunner C++ QObject)
# ============================================================

class RunnerState(Enum):
    IDLE = "idle"
    TRAINING = "training"
    RESTING = "resting"
    PAUSED = "paused"
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
        self._state_before_pause: RunnerState = RunnerState.IDLE
        self._state_lock = RLock()
        self._rest_generation = 0

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
        with self._state_lock:
            if not course.actions:
                return False
            self._cancel_rest_timer()
            self._course = course
            self._action_index = 0
            self._actual_reps = 0
            self._score_sum = 0.0
            self._score_samples = 0
            self._set_state(RunnerState.TRAINING)
            self._notify_action()
            return True

    def stop_course(self) -> None:
        with self._state_lock:
            self._cancel_rest_timer()
            self._course = None
            self._action_index = -1
            self._rest_remaining = 0
            self._actual_reps = 0
            self._score_sum = 0.0
            self._score_samples = 0
            self._state_before_pause = RunnerState.IDLE
            self._set_state(RunnerState.IDLE)

    def pause_course(self) -> bool:
        """Pause scoring progression and preserve an active rest countdown."""
        with self._state_lock:
            if self._state not in (RunnerState.TRAINING, RunnerState.RESTING):
                return False
            self._state_before_pause = self._state
            self._cancel_rest_timer()
            self._set_state(RunnerState.PAUSED)
            return True

    def resume_course(self) -> bool:
        """Resume the state that was active before pause."""
        with self._state_lock:
            if self._state != RunnerState.PAUSED:
                return False
            resume_state = self._state_before_pause
            if resume_state not in (RunnerState.TRAINING, RunnerState.RESTING):
                resume_state = RunnerState.TRAINING
            self._set_state(resume_state)
            if resume_state == RunnerState.RESTING:
                self._start_rest_timer()
            return True

    def on_score_updated(self, score: "ScoreResult") -> None:
        """
        Receive a real-time score update from ScoreBridge.
        Tracks reps and average score; triggers action completion.
        """
        with self._state_lock:
            if self._state != RunnerState.TRAINING:
                return
            action = self.current_action
            if action is None:
                return

            # Realtime counting is strictly peak-based: one detected peak is
            # one repetition. Keep course progress identical to score.count.
            self._actual_reps = max(0, int(score.count))

            # Only completed cycles carry a meaningful last_cycle score.
            if score.status == "new_completed_cycle":
                self._score_sum += score.overall_score
                self._score_samples += 1

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
        generation = self._rest_generation
        self._rest_tick(generation)

    def _rest_tick(self, generation: int) -> None:
        with self._state_lock:
            if generation != self._rest_generation or self._state != RunnerState.RESTING:
                return
            if self._rest_remaining <= 0:
                self._advance_to_next()
                return
            if self.on_rest_tick:
                self.on_rest_tick(self._rest_remaining)
            self._rest_remaining -= 1
            self._rest_timer = Timer(1.0, self._rest_tick, args=(generation,))
            self._rest_timer.daemon = True
            self._rest_timer.start()

    def _cancel_rest_timer(self) -> None:
        self._rest_generation += 1
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
