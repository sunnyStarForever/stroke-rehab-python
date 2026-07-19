"""Windowed pipeline performance statistics and transition-only alerts."""

from __future__ import annotations

import threading
import time
from collections import deque
from dataclasses import asdict, dataclass
from enum import Enum
from typing import Deque, Dict, Optional


class PerformanceState(str, Enum):
    WARMING = "WARMING"
    NORMAL = "NORMAL"
    WARN = "WARN"
    CRITICAL = "CRITICAL"
    RECOVERING = "RECOVERING"


@dataclass(frozen=True)
class PerformanceEvent:
    category: str
    code: str
    source: str
    state: str
    actual_value: float
    threshold: float
    duration_seconds: float
    capture_fps: float

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)

    def log_message(self) -> str:
        return (
            f"[PERF] code={self.code} source={self.source} state={self.state} "
            f"actual={self.actual_value:.2f} threshold={self.threshold:.2f} "
            f"duration={self.duration_seconds:.1f}s fps={self.capture_fps:.2f}"
        )


@dataclass(frozen=True)
class PerformanceSnapshot:
    state: str = PerformanceState.WARMING.value
    samples: int = 0
    callback_avg_ms: float = 0.0
    callback_p50_ms: float = 0.0
    callback_p95_ms: float = 0.0
    callback_max_ms: float = 0.0
    raw_fps: float = 0.0


class CallbackPerformanceMonitor:
    """Per-source callback and raw-FPS window with hysteretic alert states."""

    def __init__(
        self,
        *,
        window_seconds: float = 5.0,
        min_samples: int = 30,
        normal_p95_ms: float = 8.0,
        warn_p95_ms: float = 10.0,
        critical_p95_ms: float = 25.0,
        warn_sustain_seconds: float = 5.0,
        low_fps_sustain_seconds: float = 3.0,
        recovery_seconds: float = 5.0,
        target_fps: Optional[Dict[str, float]] = None,
    ) -> None:
        self.window_seconds = max(0.1, float(window_seconds))
        self.min_samples = max(1, int(min_samples))
        self.normal_p95_ms = float(normal_p95_ms)
        self.warn_p95_ms = float(warn_p95_ms)
        self.critical_p95_ms = float(critical_p95_ms)
        self.warn_sustain_seconds = max(0.0, float(warn_sustain_seconds))
        self.low_fps_sustain_seconds = max(0.0, float(low_fps_sustain_seconds))
        self.recovery_seconds = max(0.0, float(recovery_seconds))
        self.target_fps = target_fps or {"rgb": 30.0, "depth": 30.0}
        self._lock = threading.Lock()
        self._callbacks: Dict[str, Deque[tuple[float, float]]] = {}
        self._arrivals: Dict[str, Deque[float]] = {}
        self._states: Dict[str, PerformanceState] = {}
        self._warn_since: Dict[str, Optional[float]] = {}
        self._low_fps_since: Dict[str, Optional[float]] = {}
        self._recovery_since: Dict[str, Optional[float]] = {}

    def reset(self) -> None:
        with self._lock:
            self._callbacks.clear()
            self._arrivals.clear()
            self._states.clear()
            self._warn_since.clear()
            self._low_fps_since.clear()
            self._recovery_since.clear()

    def observe_arrival(self, source: str, now: Optional[float] = None) -> None:
        timestamp = time.monotonic() if now is None else float(now)
        with self._lock:
            arrivals = self._arrivals.setdefault(source, deque())
            arrivals.append(timestamp)
            self._prune(arrivals, timestamp)

    def observe_callback(
        self, source: str, duration_ms: float, now: Optional[float] = None
    ) -> Optional[PerformanceEvent]:
        timestamp = time.monotonic() if now is None else float(now)
        with self._lock:
            samples = self._callbacks.setdefault(source, deque())
            samples.append((timestamp, max(0.0, float(duration_ms))))
            self._prune(samples, timestamp)
            return self._evaluate_locked(source, timestamp)

    def evaluate(
        self, source: str, now: Optional[float] = None
    ) -> Optional[PerformanceEvent]:
        timestamp = time.monotonic() if now is None else float(now)
        with self._lock:
            self._prune(self._callbacks.setdefault(source, deque()), timestamp)
            self._prune(self._arrivals.setdefault(source, deque()), timestamp)
            return self._evaluate_locked(source, timestamp)

    def snapshot(self, source: str, now: Optional[float] = None) -> PerformanceSnapshot:
        timestamp = time.monotonic() if now is None else float(now)
        with self._lock:
            callbacks = self._callbacks.setdefault(source, deque())
            arrivals = self._arrivals.setdefault(source, deque())
            self._prune(callbacks, timestamp)
            self._prune(arrivals, timestamp)
            values = [sample[1] for sample in callbacks]
            state = self._states.get(source, PerformanceState.WARMING)
            return PerformanceSnapshot(
                state=state.value,
                samples=len(values),
                callback_avg_ms=sum(values) / len(values) if values else 0.0,
                callback_p50_ms=self._percentile(values, 0.50),
                callback_p95_ms=self._percentile(values, 0.95),
                callback_max_ms=max(values, default=0.0),
                raw_fps=self._fps(arrivals),
            )

    def _evaluate_locked(self, source: str, now: float) -> Optional[PerformanceEvent]:
        callbacks = self._callbacks.setdefault(source, deque())
        arrivals = self._arrivals.setdefault(source, deque())
        values = [sample[1] for sample in callbacks]
        fps = self._fps(arrivals)
        state = self._states.get(source, PerformanceState.WARMING)
        if len(values) < self.min_samples:
            self._states[source] = PerformanceState.WARMING
            return None
        if state is PerformanceState.WARMING:
            state = PerformanceState.NORMAL
            self._states[source] = state

        p95 = self._percentile(values, 0.95)
        target = max(0.0, float(self.target_fps.get(source, 0.0)))
        fps_threshold = target * 0.90
        fps_window_ready = len(arrivals) >= 2 and now - arrivals[0] >= self.low_fps_sustain_seconds

        if p95 > self.critical_p95_ms:
            self._warn_since[source] = None
            self._low_fps_since[source] = None
            return self._transition(
                source, state, PerformanceState.CRITICAL, "CALLBACK_P95_CRITICAL",
                p95, self.critical_p95_ms, 0.0, fps)

        low_since = self._low_fps_since.get(source)
        if target and fps_window_ready and fps < fps_threshold:
            low_since = now if low_since is None else low_since
            self._low_fps_since[source] = low_since
            if now - low_since >= self.low_fps_sustain_seconds:
                return self._transition(
                    source, state, PerformanceState.CRITICAL, "RAW_FPS_CRITICAL",
                    fps, fps_threshold, now - low_since, fps)
        else:
            self._low_fps_since[source] = None

        warn_since = self._warn_since.get(source)
        if p95 > self.warn_p95_ms:
            warn_since = now if warn_since is None else warn_since
            self._warn_since[source] = warn_since
            if now - warn_since >= self.warn_sustain_seconds:
                return self._transition(
                    source, state, PerformanceState.WARN, "CALLBACK_P95_WARN",
                    p95, self.warn_p95_ms, now - warn_since, fps)
        else:
            self._warn_since[source] = None

        healthy = p95 <= self.normal_p95_ms and (not target or fps >= fps_threshold)
        if state in (PerformanceState.WARN, PerformanceState.CRITICAL):
            if healthy:
                self._recovery_since[source] = now
                return self._transition(
                    source, state, PerformanceState.RECOVERING, "PERFORMANCE_RECOVERING",
                    p95, self.normal_p95_ms, 0.0, fps)
        elif state is PerformanceState.RECOVERING:
            recovery_since = self._recovery_since.get(source, now)
            if not healthy:
                self._recovery_since[source] = None
                # Re-evaluation will transition only after the applicable threshold.
                self._states[source] = PerformanceState.NORMAL
            elif now - recovery_since >= self.recovery_seconds:
                self._recovery_since[source] = None
                return self._transition(
                    source, state, PerformanceState.NORMAL, "PERFORMANCE_RECOVERED",
                    p95, self.normal_p95_ms, now - recovery_since, fps)
        return None

    def _transition(
        self, source: str, old: PerformanceState, new: PerformanceState,
        code: str, value: float, threshold: float, duration: float, fps: float
    ) -> Optional[PerformanceEvent]:
        if old is new:
            return None
        self._states[source] = new
        return PerformanceEvent(
            category="performance", code=code, source=source, state=new.value,
            actual_value=value, threshold=threshold,
            duration_seconds=duration, capture_fps=fps,
        )

    def _prune(self, values: Deque, now: float) -> None:
        cutoff = now - self.window_seconds
        while values:
            timestamp = values[0][0] if isinstance(values[0], tuple) else values[0]
            if timestamp >= cutoff:
                break
            values.popleft()

    @staticmethod
    def _percentile(values, quantile: float) -> float:
        if not values:
            return 0.0
        ordered = sorted(values)
        index = min(len(ordered) - 1, max(0, int((len(ordered) - 1) * quantile + 0.999999)))
        return float(ordered[index])

    @staticmethod
    def _fps(arrivals: Deque[float]) -> float:
        if len(arrivals) < 2:
            return 0.0
        span = arrivals[-1] - arrivals[0]
        return (len(arrivals) - 1) / span if span > 0.0 else 0.0
