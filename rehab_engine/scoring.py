"""
Real-time scoring bridge and offline report runner.
Replaces core/scoring/ScoreBridge.cpp + OfflineReportRunner.cpp.

Previous C++ architecture: QProcess -> Python subprocess (JSON Lines over stdin/stdout)
New architecture:       subprocess.Popen -> Python subprocess (same protocol)
                        or direct import if scoring engine is in PYTHONPATH.
"""

import json
import os
import subprocess
import sys
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional


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
    for base in [
        Path(__file__).resolve().parent.parent.parent / "tools" / "scoring_engine",
        Path.cwd() / "tools" / "scoring_engine",
        Path(os.environ.get("STROKE_REHAB_ROOT", "")) / "tools" / "scoring_engine",
    ]:
        if (base / "score_server.py").exists():
            return base
    return None


# ============================================================
# ScoreBridge - real-time scoring
# ============================================================

class ScoreBridge:
    """
    Real-time scoring via subprocess (JSON Lines protocol).

    Spawns `score_server.py` and communicates via stdin/stdout JSON Lines.
    Same wire protocol as the C++ ScoreBridge.
    """

    def __init__(self):
        self._process: Optional[subprocess.Popen] = None
        self._action_id: str = ""
        self._skeleton_fps: float = 20.0
        self._running: bool = False
        self._reader_thread: Optional[threading.Thread] = None
        self._state_lock = threading.Lock()
        self._io_lock = threading.Lock()
        self._cancel_start = threading.Event()

        # Callbacks
        self.on_score_updated: Optional[Callable[[ScoreResult], None]] = None
        self.on_error: Optional[Callable[[str], None]] = None

    def start(self, action_id: str, skeleton_fps: float = 20.0) -> bool:
        """Launch score_server.py and negotiate the initial state."""
        self.stop()
        self._cancel_start.clear()

        engine_dir = _find_scoring_engine()
        if engine_dir is None:
            self._emit_error("Scoring engine not found (tools/scoring_engine/)")
            return False

        server_path = engine_dir / "score_server.py"
        try:
            proc = subprocess.Popen(
                [sys.executable, str(server_path),
                 "--action", action_id,
                 "--fs", str(skeleton_fps)],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                cwd=str(engine_dir),
                text=True,
                encoding="utf-8",
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

        # Read the initial "ready" message
        try:
            line = proc.stdout.readline()
            if not line:
                self._emit_error("score_server exited before ready handshake")
                self.stop()
                return False
            ready_msg = json.loads(line)
            if not ready_msg.get("ok"):
                self._emit_error(f"score_server not ready: {ready_msg}")
                self.stop()
                return False
        except Exception as e:
            self._emit_error(f"score_server handshake failed: {e}")
            self.stop()
            return False

        with self._state_lock:
            if self._cancel_start.is_set() or self._process is not proc:
                self._terminate_process(proc)
                return False
            self._action_id = action_id
            self._skeleton_fps = skeleton_fps
            self._running = True

        # Start stderr reader thread (for logging)
        self._reader_thread = threading.Thread(
            target=self._read_stderr, args=(proc,), name="score-stderr", daemon=True)
        self._reader_thread.start()

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
        self._send_command({"cmd": "set_action", "action": action_id, "fs": skeleton_fps})
        # Read the acknowledgment
        if self._process and self._process.stdout:
            try:
                line = self._process.stdout.readline()
                if line:
                    ack = json.loads(line)
                    if ack.get("ok"):
                        self._action_id = action_id
                        return True
            except Exception:
                pass
        return False

    def submit_skeleton(self, frame_index: int, timestamp_ns: int,
                        joints: List[List[float]]) -> bool:
        """
        Submit a 22x3 joint array for real-time scoring.
        joints: list of [x, y, z] for each of the 22 Rehab22 joints.
        """
        with self._state_lock:
            if not self._running or self._process is None:
                return False
            proc = self._process
        msg = {
            "cmd": "frame",
            "frame_index": frame_index,
            "joints": joints,
        }
        # A single request/response transaction owns the pipe. stop() can kill
        # the local process to unblock readline, but cannot swap in a new one.
        with self._io_lock:
            try:
                if proc.poll() is not None or not proc.stdin or not proc.stdout:
                    return False
                proc.stdin.write(json.dumps(msg, ensure_ascii=False) + "\n")
                proc.stdin.flush()
                line = proc.stdout.readline()
                if not line:
                    return False
                result = json.loads(line)
                if result.get("ok") and self.on_score_updated:
                    self.on_score_updated(_parse_score_result(result))
                    return True
                if not result.get("ok"):
                    self._emit_error(
                        f"score_server error: {result.get('message', 'unknown')}")
            except (BrokenPipeError, OSError, ValueError, json.JSONDecodeError) as e:
                # Expected when stop() interrupts a pending read.
                with self._state_lock:
                    still_running = self._running and self._process is proc
                if still_running:
                    self._emit_error(f"score_server read error: {e}")
                return False
        return True

    def _send_command(self, data: dict) -> None:
        if self._process and self._process.stdin:
            try:
                self._process.stdin.write(json.dumps(data, ensure_ascii=False) + "\n")
                self._process.stdin.flush()
            except (BrokenPipeError, OSError):
                self._emit_error("score_server pipe broken")

    def _read_stderr(self, proc: Optional[subprocess.Popen]) -> None:
        """Read stderr from the subprocess for logging."""
        if proc and proc.stderr:
            for line in proc.stderr:
                with self._state_lock:
                    active = self._running and self._process is proc
                if not active:
                    break
                # stderr contains log messages; forward to our logger
                pass  # Could hook into rehab_engine._stub.logger here

    def _emit_error(self, message: str) -> None:
        if self.on_error:
            self.on_error(message)


def _parse_score_result(raw: dict) -> ScoreResult:
    """Parse the JSON response from score_server into a ScoreResult."""
    scores = raw.get("scores", raw)  # handle both wrapped and flat formats
    return ScoreResult(
        status=raw.get("status", ""),
        count=raw.get("count", 0),
        completed_count=raw.get("completed_count", raw.get("completedCount", 0)),
        overall_score=float(scores.get("overall_score", scores.get("overallScore", 0.0))),
        amplitude_score=float(scores.get("amplitude_score", scores.get("amplitudeScore", 0.0))),
        smoothness_score=float(scores.get("smoothness_score", scores.get("smoothnessScore", 0.0))),
        trunk_score=float(scores.get("trunk_score", scores.get("trunkScore", 0.0))),
        symmetry_score=float(scores.get("symmetry_score", scores.get("symmetryScore", 0.0))),
        rhythm_score=float(scores.get("rhythm_score", scores.get("rhythmScore", 0.0))),
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
        self.on_ready: Optional[Callable[[str], None]] = None  # html_path
        self.on_error: Optional[Callable[[str], None]] = None

    def run(self, csv_path: str, action_id: str,
            output_path: str, fs: float = 20.0) -> bool:
        """Generate an offline HTML report for a single action."""
        engine_dir = _find_scoring_engine()
        if engine_dir is None:
            self._emit_error("Scoring engine not found")
            return False

        cli_path = engine_dir / "offline_report_cli.py"
        cmd = [
            sys.executable, str(cli_path),
            "--csv", csv_path,
            "--action", action_id,
            "--output", output_path,
            "--fs", str(fs),
        ]
        try:
            self._process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                cwd=str(engine_dir),
                text=True,
                encoding="utf-8",
            )
            stdout, stderr = self._process.communicate(timeout=60)
            if self._process.returncode == 0:
                # The CLI outputs one JSON line on stdout with the result path
                try:
                    result = json.loads(stdout.strip() or "{}")
                    html_path = result.get("output", result.get("html_path", output_path))
                except json.JSONDecodeError:
                    html_path = output_path
                if self.on_ready:
                    self.on_ready(html_path)
                return True
            else:
                self._emit_error(stderr[:500] if stderr else "Report generation failed")
                return False
        except subprocess.TimeoutExpired:
            self._process.kill()
            self._emit_error("Report generation timed out")
            return False
        except Exception as e:
            self._emit_error(str(e))
            return False

    def _emit_error(self, message: str) -> None:
        if self.on_error:
            self.on_error(message)
