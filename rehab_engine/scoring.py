"""
Real-time scoring bridge and offline report runner.
Replaces core/scoring/ScoreBridge.cpp + OfflineReportRunner.cpp.

Previous C++ architecture: QProcess -> Python subprocess (JSON Lines over stdin/stdout)
New architecture:       subprocess.Popen -> Python subprocess (same protocol)
                        or direct import if scoring engine is in PYTHONPATH.
"""

from __future__ import annotations

import json
import math
import os
import re
import csv
import subprocess
import sys
import threading
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from . import logger


@dataclass
class ScoreResult:
    """Mirrors the C++ ScoreResult struct."""
    status: str = ""
    count: int = 0
    completed_count: int = 0
    overall_score: float = 0.0
    amplitude_score: float = 0.0
    smoothness_score: float = 0.0
    trunk_score: float = 0.0
    symmetry_score: float = 0.0
    rhythm_score: float = 0.0


def _find_scoring_engine() -> Optional[Path]:
    """Locate tools/scoring_engine/ relative to project root."""
    python_root = Path(__file__).resolve().parent.parent
    repo_root = python_root.parent
    configured_root = Path(os.environ.get("STROKE_REHAB_ROOT", "")).expanduser()
    for base in [
        python_root / "tools" / "scoring_engine",
        repo_root / "tools" / "scoring_engine",
        repo_root / "stroke-rehab" / "tools" / "scoring_engine",
        Path.cwd() / "tools" / "scoring_engine",
        configured_root / "tools" / "scoring_engine",
        configured_root / "stroke-rehab" / "tools" / "scoring_engine",
    ]:
        if (base / "score_server.py").exists():
            return base.resolve()
    return None


def _normalize_action_id(action_id: str) -> Optional[str]:
    normalized = str(action_id).strip()
    return normalized if re.fullmatch(r"M(?:[1-9]|10)", normalized) else None


def _invalid_action_message(action_id: str) -> str:
    normalized = str(action_id).strip()
    if normalized == "shoulder_basic" or "_" in normalized:
        return f"ScoreBridge 只接受 M1-M10，当前收到的是课程ID {normalized}"
    return f"ScoreBridge 只接受 M1-M10，当前收到的是 {normalized}"


class ScoringSkeletonAdapter:
    """Convert Rehab22 joints to the exact coordinate contract used by P-Coder."""

    JOINT_COUNT = 22

    @staticmethod
    def _joint(item: Any) -> tuple[float, float, float, bool]:
        if hasattr(item, "x"):
            x = float(item.x)
            y = float(item.y)
            z = float(item.z)
            valid = bool(getattr(item, "valid", True))
        else:
            values = list(item)
            if len(values) < 3:
                return 0.0, 0.0, 0.0, False
            x, y, z = map(float, values[:3])
            valid = bool(values[3]) if len(values) >= 4 else True
        valid = valid and all(math.isfinite(value) for value in (x, y, z))
        return x, y, z, valid

    @classmethod
    def valid_joint_count(cls, joints: List[Any]) -> int:
        return sum(cls._joint(item)[3] for item in joints[:cls.JOINT_COUNT])

    @classmethod
    def convert(cls, joints: List[Any]) -> List[List[float]]:
        if len(joints) < cls.JOINT_COUNT:
            raise ValueError("Rehab22 scoring requires 22 joints")
        source = [cls._joint(item) for item in joints[:cls.JOINT_COUNT]]
        output = [
            [x, -y, -z] if valid else [0.0, 0.0, 0.0]
            for x, y, z, valid in source
        ]

        def midpoint(dst: int, left: int, right: int) -> None:
            if not source[dst][3] and source[left][3] and source[right][3]:
                a, b = source[left], source[right]
                output[dst] = [
                    0.5 * (a[0] + b[0]),
                    -0.5 * (a[1] + b[1]),
                    -0.5 * (a[2] + b[2]),
                ]

        # HeadTip extends Head away from Neck by half a head-neck vector.
        if not source[5][3] and source[4][3] and source[3][3]:
            head, neck = source[4], source[3]
            output[5] = [
                head[0] + 0.5 * (head[0] - neck[0]),
                -(head[1] + 0.5 * (head[1] - neck[1])),
                -(head[2] + 0.5 * (head[2] - neck[2])),
            ]
        midpoint(6, 3, 7)   # LeftCollar <- Neck/LeftUpperArm
        midpoint(10, 3, 11) # RightCollar <- Neck/RightUpperArm
        if not source[17][3] and source[16][3]:
            output[17] = list(output[16])
        if not source[21][3] and source[20][3]:
            output[21] = list(output[20])
        return output


class ScoringCsvRecorder:
    """Write the per-action wide CSV consumed by offline_report_cli.py."""

    JOINT_NAMES = (
        "waist", "spine", "chest", "neck", "head", "head_tip",
        "l_collar", "l_shoulder", "l_elbow", "l_hand", "r_collar",
        "r_shoulder", "r_elbow", "r_hand", "l_hip", "l_knee",
        "l_foot", "l_toe", "r_hip", "r_knee", "r_foot", "r_toe",
    )

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._file = None
        self._writer = None
        self._path = ""
        self._rows = 0

    def start(self, action_dir: str) -> bool:
        self.stop()
        directory = Path(action_dir)
        try:
            directory.mkdir(parents=True, exist_ok=True)
            output = directory / "skeleton3d.csv"
            handle = output.open("w", newline="", encoding="utf-8")
        except OSError:
            return False
        header = ["frame_idx"]
        for index, name in enumerate(self.JOINT_NAMES):
            header.extend(
                f"{index:02d}_{name}_{axis}" for axis in ("x", "y", "z"))
        with self._lock:
            self._file = handle
            self._writer = csv.writer(handle, lineterminator="\n")
            self._writer.writerow(header)
            handle.flush()
            self._path = str(output.resolve())
            self._rows = 0
        return True

    def append(self, frame_index: int, scoring_joints: List[List[float]]) -> bool:
        if len(scoring_joints) != ScoringSkeletonAdapter.JOINT_COUNT:
            return False
        with self._lock:
            if self._writer is None:
                return False
            row: List[Any] = [int(frame_index)]
            for joint in scoring_joints:
                if len(joint) < 3:
                    return False
                row.extend(f"{float(value):.6f}" for value in joint[:3])
            try:
                self._writer.writerow(row)
                self._rows += 1
                if self._rows % 64 == 0:
                    self._file.flush()
                return True
            except (OSError, ValueError):
                return False

    def stop(self) -> None:
        with self._lock:
            if self._file:
                try:
                    self._file.flush()
                    self._file.close()
                except OSError:
                    pass
            self._file = None
            self._writer = None

    @property
    def is_recording(self) -> bool:
        with self._lock:
            return self._writer is not None

    @property
    def csv_path(self) -> str:
        return self._path

    @property
    def rows(self) -> int:
        return self._rows


# ============================================================
# ScoreBridge - real-time scoring
# ============================================================

class ScoreBridge:
    """
    Real-time scoring via subprocess (JSON Lines protocol).

    Spawns `score_server.py` and communicates via stdin/stdout JSON Lines.
    Same wire protocol as the C++ ScoreBridge.
    """

    def __init__(self, ready_timeout_seconds: Optional[float] = None):
        self._process: Optional[subprocess.Popen] = None
        self._action_id: str = ""
        self._skeleton_fps: float = 20.0
        self._running: bool = False
        self._reader_thread: Optional[threading.Thread] = None
        self._stdout_thread: Optional[threading.Thread] = None
        self._state_lock = threading.Lock()
        self._io_lock = threading.Lock()
        self._cancel_start = threading.Event()
        self._ready_event = threading.Event()
        self._stderr_tail = deque(maxlen=8)
        if ready_timeout_seconds is None:
            try:
                ready_timeout_seconds = float(
                    os.environ.get("STROKE_SCORE_READY_TIMEOUT_SECONDS", "20"))
            except ValueError:
                ready_timeout_seconds = 20.0
        self._ready_timeout_seconds = max(1.0, min(120.0, ready_timeout_seconds))

        # Callbacks
        self.on_score_updated: Optional[Callable[[ScoreResult], None]] = None
        self.on_error: Optional[Callable[[str], None]] = None

    def start(self, action_id: str, skeleton_fps: float = 20.0) -> bool:
        """Launch score_server.py and negotiate the initial state."""
        normalized_action_id = _normalize_action_id(action_id)
        if normalized_action_id is None:
            self._emit_error(_invalid_action_message(action_id))
            return False
        self.stop()
        self._cancel_start.clear()
        self._ready_event.clear()
        self._stderr_tail.clear()

        engine_dir = _find_scoring_engine()
        if engine_dir is None:
            self._emit_error("Scoring engine not found (tools/scoring_engine/)")
            return False

        server_path = engine_dir / "score_server.py"
        try:
            proc = subprocess.Popen(
                [sys.executable, "-u", str(server_path),
                 "--action", normalized_action_id,
                 "--fs", str(skeleton_fps)],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                cwd=str(engine_dir),
                text=True,
                encoding="utf-8",
                errors="replace",
                bufsize=1,
            )
        except FileNotFoundError:
            self._emit_error(f"Python not found: {sys.executable}")
            return False
        except Exception as e:
            self._emit_error(f"Failed to start score_server: {e}")
            return False

        with self._state_lock:
            if self._cancel_start.is_set():
                self._terminate_process(proc)
                return False
            self._process = proc

        self._stdout_thread = threading.Thread(
            target=self._read_stdout, args=(proc,), name="score-stdout", daemon=True)
        self._stdout_thread.start()
        self._reader_thread = threading.Thread(
            target=self._read_stderr, args=(proc,), name="score-stderr", daemon=True)
        self._reader_thread.start()

        if (not self._ready_event.wait(timeout=self._ready_timeout_seconds)
                or proc.poll() is not None):
            detail = " | ".join(self._stderr_tail)
            message = "score_server exited or timed out before ready handshake"
            if detail:
                message += f": {detail}"
            self._emit_error(message)
            self.stop()
            return False

        with self._state_lock:
            if self._cancel_start.is_set() or self._process is not proc:
                self._terminate_process(proc)
                return False
            self._action_id = normalized_action_id
            self._skeleton_fps = skeleton_fps
            self._running = True

        return True

    def stop(self) -> None:
        """Request subprocess termination without blocking the Qt thread."""
        with self._state_lock:
            self._cancel_start.set()
            self._running = False
            proc = self._process
            self._process = None
        if proc:
            threading.Thread(
                target=self._terminate_process, args=(proc,),
                name="score-kill", daemon=True,
            ).start()

    @staticmethod
    def _terminate_process(proc: subprocess.Popen) -> None:
        try:
            if proc.poll() is None:
                proc.kill()
            proc.wait(timeout=3)
        except Exception:
            pass
        finally:
            for stream in (proc.stdin, proc.stdout, proc.stderr):
                try:
                    if stream:
                        stream.close()
                except Exception:
                    pass

    def reset(self) -> None:
        """Send a reset command."""
        self._send_command({"cmd": "reset"})

    def set_action(self, action_id: str, skeleton_fps: float = 20.0) -> bool:
        """Change the current action mid-session."""
        normalized_action_id = _normalize_action_id(action_id)
        if normalized_action_id is None:
            self._emit_error(_invalid_action_message(action_id))
            return False
        ok = self._send_command({
            "cmd": "set_action", "action": normalized_action_id, "fs": skeleton_fps})
        if ok:
            self._action_id = normalized_action_id
            self._skeleton_fps = skeleton_fps
        return ok

    def submit_skeleton(self, frame_index: int, timestamp_ns: int,
                        joints: List[List[float]]) -> bool:
        """
        Submit a 22x3 joint array for real-time scoring.
        joints: list of [x, y, z] for each of the 22 Rehab22 joints.
        """
        if len(joints) < ScoringSkeletonAdapter.JOINT_COUNT:
            return False
        if ScoringSkeletonAdapter.valid_joint_count(joints) < 18:
            return False
        try:
            scoring_joints = ScoringSkeletonAdapter.convert(joints)
        except (TypeError, ValueError):
            return False
        msg = {
            "cmd": "frame",
            "frame_index": frame_index,
            "timestamp_ns": str(timestamp_ns),
            "joints": scoring_joints,
        }
        return self._send_command(msg)

    def _send_command(self, data: dict) -> bool:
        with self._io_lock:
            with self._state_lock:
                if not self._running or self._process is None:
                    return False
                proc = self._process
            try:
                if proc.poll() is not None or not proc.stdin or not proc.stdout:
                    return False
                proc.stdin.write(json.dumps(data, ensure_ascii=False) + "\n")
                proc.stdin.flush()
                return True
            except (BrokenPipeError, OSError, ValueError) as e:
                with self._state_lock:
                    still_running = self._running and self._process is proc
                if still_running:
                    self._emit_error(f"score_server write error: {e}")
                return False

    def _read_stdout(self, proc: subprocess.Popen) -> None:
        if not proc.stdout:
            self._ready_event.set()
            return
        for line in proc.stdout:
            try:
                raw = json.loads(line)
            except (ValueError, json.JSONDecodeError):
                self._emit_error(f"评分服务返回非法 JSON：{line[:500].strip()}")
                continue
            if raw.get("service") == "score_server" and raw.get("message") == "ready":
                if raw.get("ok", True):
                    self._ready_event.set()
                else:
                    self._emit_error(str(raw.get("error", raw.get("message", "评分服务未就绪"))))
                continue
            if not raw.get("ok", True) or raw.get("status") == "error":
                self._emit_error(str(raw.get("error", raw.get("message", "评分服务返回错误"))))
                continue
            status = str(raw.get("status", ""))
            if status and status != "ok" and self.on_score_updated:
                self.on_score_updated(_parse_score_result(raw))
        self._ready_event.set()
        with self._state_lock:
            unexpected = self._process is proc and not self._cancel_start.is_set()
            if self._process is proc:
                self._running = False
        if unexpected:
            self._emit_error(f"评分服务意外退出，退出码：{proc.poll()}")

    def _read_stderr(self, proc: Optional[subprocess.Popen]) -> None:
        """Read stderr from the subprocess for logging."""
        if proc and proc.stderr:
            for line in proc.stderr:
                with self._state_lock:
                    active = (self._process is proc
                              and not self._cancel_start.is_set())
                if not active:
                    break
                message = line.strip()
                if message:
                    self._stderr_tail.append(message)
                    logger.warn(f"[ScoreBridge stderr] {message}")

    def _emit_error(self, message: str) -> None:
        if self.on_error:
            self.on_error(message)


def _parse_score_result(raw: dict) -> ScoreResult:
    """Parse the JSON response from score_server into a ScoreResult."""
    cycle = raw.get("last_cycle") if isinstance(raw.get("last_cycle"), dict) else {}
    dimensions = cycle.get("dimension_scores", {}) if isinstance(cycle, dict) else {}
    scores = raw.get("scores", raw)
    return ScoreResult(
        status=raw.get("status", ""),
        count=raw.get("count", 0),
        completed_count=raw.get("completed_count", raw.get("completedCount", 0)),
        overall_score=float(cycle.get("overall_score", scores.get("overall_score", scores.get("overallScore", 0.0)))),
        amplitude_score=float(dimensions.get("amplitude_score", scores.get("amplitude_score", scores.get("amplitudeScore", 0.0)))),
        smoothness_score=float(dimensions.get("smoothness_score", scores.get("smoothness_score", scores.get("smoothnessScore", 0.0)))),
        trunk_score=float(dimensions.get("trunk_score", scores.get("trunk_score", scores.get("trunkScore", 0.0)))),
        symmetry_score=float(dimensions.get("symmetry_score", scores.get("symmetry_score", scores.get("symmetryScore", 0.0)))),
        rhythm_score=float(dimensions.get("rhythm_score", scores.get("rhythm_score", scores.get("rhythmScore", 0.0)))),
    )


# ============================================================
# OfflineReportRunner - post-training report
# ============================================================

class OfflineReportRunner:
    """
    Run offline_action_report CLI after training completes.
    Replaces core/scoring/OfflineReportRunner.cpp.
    """

    def __init__(self):
        self._process: Optional[subprocess.Popen] = None
        self._lock = threading.Lock()
        self._generation = 0
        self.on_ready: Optional[Callable[[str], None]] = None  # html_path
        self.on_error: Optional[Callable[[str], None]] = None

    def run(self, csv_path: str, action_id: str,
            output_path: str, fs: float = 20.0) -> bool:
        """Generate an offline HTML report for a single action."""
        normalized_action_id = _normalize_action_id(action_id)
        if normalized_action_id is None:
            self._emit_error(_invalid_action_message(action_id))
            return False
        engine_dir = _find_scoring_engine()
        if engine_dir is None:
            self._emit_error("Scoring engine not found")
            return False

        cli_path = engine_dir / "offline_report_cli.py"
        cmd = [
            sys.executable, "-u", str(cli_path),
            "--csv", csv_path,
            "--action", normalized_action_id,
            "--out", output_path,
            "--fs", str(fs),
        ]
        try:
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                cwd=str(engine_dir),
                text=True,
                encoding="utf-8",
                errors="replace",
            )
            with self._lock:
                self._generation += 1
                generation = self._generation
                previous = self._process
                self._process = process
            if previous and previous.poll() is None:
                previous.kill()
            threading.Thread(
                target=self._wait_for_report,
                args=(process, generation, output_path),
                name=f"offline-report-{normalized_action_id}",
                daemon=True,
            ).start()
            return True
        except Exception as e:
            self._emit_error(str(e))
            return False

    def stop(self) -> None:
        with self._lock:
            self._generation += 1
            process = self._process
            self._process = None
        if process and process.poll() is None:
            try:
                process.kill()
            except OSError:
                pass

    def _wait_for_report(
        self, process: subprocess.Popen, generation: int, output_path: str
    ) -> None:
        try:
            stdout, stderr = process.communicate(timeout=60)
        except subprocess.TimeoutExpired:
            process.kill()
            process.communicate()
            self._emit_if_current(generation, False, "Report generation timed out")
            return
        with self._lock:
            current = generation == self._generation and self._process is process
            if self._process is process:
                self._process = None
        if not current:
            return
        result = {}
        for line in reversed((stdout or "").splitlines()):
            try:
                candidate = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(candidate, dict):
                result = candidate
                break
        if process.returncode == 0 and result.get("ok", False):
            html_path = result.get("html_path") or result.get("output") or output_path
            if self.on_ready:
                self.on_ready(str(html_path))
            return
        message = result.get("error") or result.get("message") or (stderr or "")[-1000:]
        self._emit_error(message or "Report generation failed")

    def _emit_if_current(self, generation: int, ok: bool, message: str) -> None:
        with self._lock:
            current = generation == self._generation
            if current:
                self._process = None
        if current and not ok:
            self._emit_error(message)

    def _emit_error(self, message: str) -> None:
        if self.on_error:
            self.on_error(message)
