"""Measure synchronized RGB and attested hardware-depth capture on the board."""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rehab_engine import _core
from rehab_engine.capture import NativeRgbDepthBackend
from rehab_engine.config_loader import load_pipeline_config


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--duration", type=float, default=10.0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)

    config = load_pipeline_config()
    backend = NativeRgbDepthBackend(_core, config.device, config.sync)
    lock = threading.Lock()
    pairs = []
    statuses: list[str] = []
    native_logs: list[dict[str, str]] = []
    _core.logger.set_callback(
        lambda level, message: native_logs.append(
            {"level": str(level), "message": str(message)}))
    backend.set_on_status(statuses.append)

    def on_pair(pair):
        with lock:
            pairs.append({
                "rgb_id": int(pair.rgb.frame_id),
                "depth_id": int(pair.depth.frame_id),
                "rgb_ts": int(pair.rgb.host_ts_ns),
                "depth_ts": int(pair.depth.host_ts_ns),
                "delta_ns": int(pair.delta_ns),
                "rgb_format": str(pair.rgb.pixel_format_name),
                "depth_format": str(pair.depth.pixel_format_name),
                "depth_unit_to_meter": float(pair.depth.depth_unit_to_meter),
            })

    started = bool(backend.start(on_pair))
    started_at = time.monotonic()
    if started:
        time.sleep(max(0.1, args.duration))
    elapsed = time.monotonic() - started_at
    hardware_d2c = bool(backend.hardware_d2c_active()) if started else False
    stats = backend.sync_stats()
    backend.stop()
    _core.logger.set_callback(None)

    with lock:
        captured = list(pairs)
    deltas_ms = [abs(item["delta_ns"]) / 1_000_000.0 for item in captured]
    summary = {
        "started": started,
        "attested_hardware_depth": started,
        "hardware_d2c_active": hardware_d2c,
        "elapsed_seconds": elapsed,
        "pair_count": len(captured),
        "pair_fps": len(captured) / elapsed if elapsed > 0 else 0.0,
        "rgb_unique_frames": len({item["rgb_id"] for item in captured}),
        "depth_unique_frames": len({item["depth_id"] for item in captured}),
        "mean_abs_delta_ms": statistics.fmean(deltas_ms) if deltas_ms else 0.0,
        "p95_abs_delta_ms": (
            sorted(deltas_ms)[min(len(deltas_ms) - 1, int(len(deltas_ms) * 0.95))]
            if deltas_ms else 0.0
        ),
        "max_abs_delta_ms": max(deltas_ms) if deltas_ms else 0.0,
        "rgb_formats": sorted({item["rgb_format"] for item in captured}),
        "depth_formats": sorted({item["depth_format"] for item in captured}),
        "depth_units": sorted({item["depth_unit_to_meter"] for item in captured}),
        "sync_matched": int(stats.matched),
        "sync_threshold_misses": int(stats.threshold_misses),
        "sync_rgb_trimmed": int(stats.rgb_trimmed),
        "sync_depth_trimmed": int(stats.depth_trimmed),
        "statuses": statuses,
        "native_logs": native_logs,
    }
    args.output.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if started and captured else 1


if __name__ == "__main__":
    raise SystemExit(main())
