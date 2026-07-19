"""Compatibility routing for the legacy ``(level, message)`` log callback."""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from typing import Dict, Optional


_CODE = re.compile(r"\bcode=([A-Za-z0-9_.:-]+)")


def event_category(level: str, message: str) -> str:
    text = message.strip()
    if level.upper() == "PERF" or text.startswith("[PERF]"):
        return "performance"
    lowered = text.lower()
    if "record" in lowered or "录制" in text:
        return "recording"
    if "camera" in lowered or "device" in lowered or "采集" in text or "设备" in text:
        return "device"
    return "application"


def event_code(level: str, message: str) -> str:
    match = _CODE.search(message)
    if match:
        return match.group(1)
    normalized = re.sub(r"\d+(?:\.\d+)?", "#", message.lower())
    normalized = re.sub(r"\s+", " ", normalized).strip()
    return f"{event_category(level, message)}:{normalized[:96]}"


@dataclass
class UserNotificationGate:
    cooldown_seconds: float = 30.0
    _last_shown: Dict[str, float] = field(default_factory=dict)

    def should_notify(
        self, level: str, message: str, now: Optional[float] = None
    ) -> bool:
        if event_category(level, message) == "performance":
            return False
        if level.upper() not in ("WARN", "ERROR"):
            return False
        timestamp = time.monotonic() if now is None else float(now)
        code = event_code(level, message)
        previous = self._last_shown.get(code)
        if previous is not None and timestamp - previous < self.cooldown_seconds:
            return False
        self._last_shown[code] = timestamp
        return True

    def recover(self, level: str, message: str) -> None:
        self._last_shown.pop(event_code(level, message), None)
