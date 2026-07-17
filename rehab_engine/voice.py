"""Non-blocking, failure-tolerant training voice prompts."""

from dataclasses import dataclass, field
import heapq
import itertools
import threading
import time
from typing import Any, Callable, Optional


@dataclass(order=True)
class _Prompt:
    priority: int
    sequence: int
    text: str = field(compare=False)
    key: str = field(compare=False)


class VoiceAssistant:
    """Serialize TTS on a worker thread without blocking sensor or UI threads."""

    def __init__(
        self,
        config: Any,
        engine_factory: Optional[Callable[[], Any]] = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._config = config
        self._engine_factory = engine_factory
        self._clock = clock
        self._condition = threading.Condition()
        self._queue = []
        self._sequence = itertools.count()
        self._pending_keys = set()
        self._last_spoken = {}
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._engine = None
        self._available: Optional[bool] = None
        self._last_error = ""
        self.on_status: Optional[Callable[[str], None]] = None

    @property
    def enabled(self) -> bool:
        return bool(getattr(self._config, "enabled", True))

    @property
    def available(self) -> Optional[bool]:
        return self._available

    @property
    def last_error(self) -> str:
        return self._last_error

    def start(self) -> bool:
        if not self.enabled:
            self._available = False
            self._emit("Voice prompts disabled")
            return False
        with self._condition:
            if self._running:
                return True
            self._running = True
            self._thread = threading.Thread(
                target=self._worker, name="voice-assistant", daemon=True)
            self._thread.start()
        return True

    def speak(
        self, text: str, key: str = "", priority: int = 10, force: bool = False
    ) -> bool:
        text = str(text).strip()
        if not text or not self.enabled:
            return False
        if not self._running:
            self.start()
        prompt_key = key.strip() or text
        now = self._clock()
        cooldown = max(0.0, float(getattr(self._config, "cooldown_seconds", 2.0)))
        with self._condition:
            if not self._running:
                return False
            if not force and (
                prompt_key in self._pending_keys
                or now - self._last_spoken.get(prompt_key, float("-inf")) < cooldown
            ):
                return False
            limit = max(1, int(getattr(self._config, "queue_size", 12)))
            if len(self._queue) >= limit:
                worst = max(range(len(self._queue)), key=lambda i: self._queue[i].priority)
                if self._queue[worst].priority <= int(priority):
                    return False
                removed = self._queue.pop(worst)
                heapq.heapify(self._queue)
                self._pending_keys.discard(removed.key)
            prompt = _Prompt(int(priority), next(self._sequence), text, prompt_key)
            heapq.heappush(self._queue, prompt)
            self._pending_keys.add(prompt_key)
            self._condition.notify()
        return True

    def clear(self) -> None:
        with self._condition:
            self._queue.clear()
            self._pending_keys.clear()

    def stop(self, timeout: float = 2.0) -> None:
        with self._condition:
            self._running = False
            self._queue.clear()
            self._pending_keys.clear()
            self._condition.notify_all()
        engine = self._engine
        if engine is not None:
            try:
                engine.stop()
            except Exception:
                pass
        thread = self._thread
        if thread and thread is not threading.current_thread():
            thread.join(timeout=max(0.0, timeout))
        self._thread = None

    def _worker(self) -> None:
        try:
            self._engine = self._create_engine()
            self._available = True
            self._emit("Voice prompts ready")
        except Exception as exc:
            self._available = False
            self._last_error = str(exc)
            self._emit(f"Voice prompts unavailable: {exc}")
            with self._condition:
                self._running = False
                self._queue.clear()
                self._pending_keys.clear()
            return

        while True:
            with self._condition:
                while self._running and not self._queue:
                    self._condition.wait()
                if not self._running:
                    break
                prompt = heapq.heappop(self._queue)
                self._pending_keys.discard(prompt.key)
            try:
                self._engine.say(prompt.text)
                self._engine.runAndWait()
                self._last_spoken[prompt.key] = self._clock()
            except Exception as exc:
                self._available = False
                self._last_error = str(exc)
                self._emit(f"Voice prompt failed; continuing silently: {exc}")
                with self._condition:
                    self._running = False
                    self._queue.clear()
                    self._pending_keys.clear()
                break
        self._engine = None

    def _create_engine(self):
        if self._engine_factory is not None:
            engine = self._engine_factory()
        else:
            backend = str(getattr(self._config, "backend", "auto")).strip().lower()
            if backend not in ("", "auto", "pyttsx3"):
                raise ValueError(f"Unsupported voice backend: {backend}")
            import pyttsx3
            engine = pyttsx3.init()
        engine.setProperty("rate", int(getattr(self._config, "rate", 175)))
        engine.setProperty(
            "volume", min(1.0, max(0.0, float(getattr(self._config, "volume", 0.9)))))
        return engine

    def _emit(self, message: str) -> None:
        if self.on_status:
            try:
                self.on_status(message)
            except Exception:
                pass


__all__ = ["VoiceAssistant"]
