#!/usr/bin/env python3
"""Unified preflight check for the Python-main rehabilitation runtime."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def _run_models() -> tuple[bool, str]:
    process = subprocess.run(
        [sys.executable, str(ROOT / "verify_models.py")],
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=120,
    )
    return process.returncode == 0, process.stdout.strip()


def _check_ui() -> tuple[bool, str]:
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    try:
        from PyQt5.QtWidgets import QApplication
        from ui.main_window import StrokeRehabWindow

        app = QApplication.instance() or QApplication([])
        window = StrokeRehabWindow()
        window.close()
        app.processEvents()
        return True, "StrokeRehabWindow constructed with the offscreen Qt backend"
    except Exception as exc:
        return False, str(exc)


def _run_capture_smoke(config, duration_seconds: float) -> tuple[bool, dict]:
    """Open both native drivers and prove Python receives synchronized pairs."""
    try:
        from rehab_engine import _core
        from rehab_engine.capture import NativeRgbDepthBackend
    except (ImportError, AttributeError) as exc:
        return False, {"error": f"native hardware adapter unavailable: {exc}"}

    backend = NativeRgbDepthBackend(_core, config.device, config.sync)
    first_pair = threading.Event()
    pairs = []

    def _on_pair(pair):
        pairs.append(pair)
        first_pair.set()

    started = False
    stop_error = ""
    ok = False
    detail = {}
    start_time = time.monotonic()
    try:
        started = backend.start(_on_pair)
        if not started:
            detail = {"error": "native RGB/Depth backend failed to start"}
        else:
            first_pair.wait(timeout=max(1.0, duration_seconds))
            deadline = start_time + max(0.0, duration_seconds)
            while time.monotonic() < deadline:
                time.sleep(min(0.05, deadline - time.monotonic()))
            sync = backend.sync_stats()
            elapsed = max(0.001, time.monotonic() - start_time)
            detail = {
                "pairs": len(pairs),
                "pair_fps": len(pairs) / elapsed,
                "sync_matched": sync.matched,
                "sync_threshold_misses": sync.threshold_misses,
                "sync_rgb_trimmed": sync.rgb_trimmed,
                "sync_depth_trimmed": sync.depth_trimmed,
                "hardware_d2c_active": backend.hardware_d2c_active(),
            }
            ok = bool(pairs)
    except Exception as exc:
        detail = {"error": str(exc), "pairs": len(pairs)}
    finally:
        if started:
            try:
                backend.stop()
            except Exception as exc:
                stop_error = str(exc)
        if stop_error:
            detail["stop_error"] = stop_error
            ok = False
    return ok, detail


def _run_emg_smoke(config, duration_seconds: float) -> tuple[bool, dict]:
    """Prove the configured real EMG capture and CPU1 feature path end to end."""
    if not config.emg.enabled:
        return False, {"error": "EMG must be configured with enabled=true"}
    from rehab_engine.emg import EmgManager

    manager = EmgManager(config.emg)
    started = False
    ok = False
    detail = {}
    try:
        started = manager.start()
        if not started:
            detail = {"error": manager.runtime_status().message}
        else:
            deadline = time.monotonic() + max(1.0, duration_seconds)
            while time.monotonic() < deadline and manager.latest_feature() is None:
                time.sleep(0.05)
            status = manager.runtime_status()
            detail = {
                "capture_backend": status.capture_backend,
                "link_state": status.link_state,
                "ble_connected": status.ble_connected,
                "rpmsg_connected": status.rpmsg_connected,
                "raw_samples": status.raw_samples,
                "raw_chunks": status.raw_chunks,
                "feature_frames": status.feature_frames,
                "parse_errors": status.parse_errors,
                "rpmsg_errors": status.rpmsg_errors,
                "message": status.message,
            }
            ok = bool(
                manager.latest_feature() is not None
                and status.ble_connected
                and status.rpmsg_connected
            )
    except Exception as exc:
        detail = {"error": str(exc)}
    finally:
        if started:
            try:
                manager.stop()
            except Exception as exc:
                detail["stop_error"] = str(exc)
                ok = False
    return ok, detail


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", action="store_true", help="load and run shipped ONNX models")
    parser.add_argument("--ui", action="store_true", help="construct the UI using offscreen Qt")
    parser.add_argument(
        "--require-hardware",
        action="store_true",
        help="require native hardware and perform a real RGB/Depth capture smoke test",
    )
    parser.add_argument(
        "--capture-smoke-seconds",
        type=float,
        default=3.0,
        help="real capture duration used with --require-hardware (0 disables active smoke)",
    )
    parser.add_argument(
        "--require-emg",
        action="store_true",
        help="require configured real BLE/RFCOMM -> RPMsg -> CPU1 feature flow",
    )
    parser.add_argument(
        "--emg-smoke-seconds",
        type=float,
        default=8.0,
        help="maximum time to wait for a real CPU1 EMG feature frame",
    )
    parser.add_argument("--json", action="store_true", help="emit JSON only")
    args = parser.parse_args()

    from rehab_engine import _CORE_READY, load_pipeline_config, run_diagnostics
    from rehab_engine.capture import FrameEnvelope, FrameSource, FrameSynchronizer

    config = load_pipeline_config()
    diagnostics = run_diagnostics(config)
    fatal = [item for item in diagnostics.items if item.status == "ERROR"]
    if not args.require_hardware:
        hardware_terms = ("V4L2", "OpenNI", "硬件", "相机", "EMG 真实")
        fatal = [item for item in fatal if not any(term in item.name for term in hardware_terms)]

    sync = FrameSynchronizer(config.sync)
    import numpy as np
    rgb = FrameEnvelope(
        FrameSource.RGB, np.zeros((1, 1, 3), dtype=np.uint8),
        1, 1, 1_000_000_000)
    depth = FrameEnvelope(
        FrameSource.DEPTH, np.zeros((1, 1), dtype=np.uint16),
        1, 1, 1_001_000_000)
    sync_ok = sync.push_frame(rgb) is None and sync.push_frame(depth) is not None

    checks = {
        "config_loaded": True,
        "python_sync_smoke": sync_ok,
        "native_hardware_adapter_loaded": bool(_CORE_READY),
        "diagnostic_errors": [f"{item.name}: {item.detail}" for item in fatal],
    }
    details = {}
    if args.models:
        checks["models"], details["models"] = _run_models()
    if args.ui:
        checks["ui"], details["ui"] = _check_ui()
    if args.require_hardware and args.capture_smoke_seconds > 0:
        checks["hardware_capture_smoke"], details["hardware_capture_smoke"] = \
            _run_capture_smoke(config, args.capture_smoke_seconds)
    if args.require_emg:
        checks["emg_smoke"], details["emg_smoke"] = \
            _run_emg_smoke(config, args.emg_smoke_seconds)

    required = [checks["config_loaded"], checks["python_sync_smoke"], not fatal]
    if args.models:
        required.append(bool(checks["models"]))
    if args.ui:
        required.append(bool(checks["ui"]))
    if args.require_hardware:
        required.append(bool(checks["native_hardware_adapter_loaded"]))
        if args.capture_smoke_seconds > 0:
            required.append(bool(checks["hardware_capture_smoke"]))
    if args.require_emg:
        required.append(bool(checks["emg_smoke"]))
    report = {"ok": all(required), "checks": checks, "details": details}

    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    else:
        print("Stroke Rehab Python-main runtime verification")
        for name, value in checks.items():
            print(f"  {name}: {value}")
        for name, value in details.items():
            print(f"\n[{name}]\n{value}")
        print(f"\nRESULT: {'PASS' if report['ok'] else 'FAIL'}")
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
