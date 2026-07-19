"""Measure synchronized RGB and attested hardware-depth capture on the board."""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rehab_engine import _core
from rehab_engine.capture import NativeRgbDepthBackend
from rehab_engine.config_loader import load_pipeline_config

try:
    import resource
except ImportError:  # pragma: no cover - Windows validation fallback
    resource = None


def current_rss_bytes() -> int:
    try:
        import psutil
        return int(psutil.Process(os.getpid()).memory_info().rss)
    except (ImportError, OSError):
        if resource is not None:
            return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1024
        return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--duration", type=float, default=10.0)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rgb-format", choices=("MJPG", "YUYV"))
    parser.add_argument("--depth-format", choices=("DEPTH_1_MM", "DEPTH_100_UM"))
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)

    config = load_pipeline_config()
    if args.rgb_format:
        config.device.rgb_pixel_format = args.rgb_format
    if args.depth_format:
        config.device.depth_pixel_format = args.depth_format
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
            first_pair = not pairs
            pairs.append({
                "rgb_id": int(pair.rgb.frame_id),
                "depth_id": int(pair.depth.frame_id),
                "rgb_arrival_ts_ns": int(pair.rgb.arrival_ts_ns),
                "depth_arrival_ts_ns": int(pair.depth.arrival_ts_ns),
                "rgb_sync_ts_ns": int(pair.rgb.sync_ts_ns),
                "depth_sync_ts_ns": int(pair.depth.sync_ts_ns),
                "delta_ns": int(pair.delta_ns),
                "rgb_format": str(pair.rgb.pixel_format_name),
                "depth_format": str(pair.depth.pixel_format_name),
                "depth_unit_to_meter": float(pair.depth.depth_unit_to_meter),
                "rgb_clock_quality": str(pair.rgb.clock_quality),
                "depth_clock_quality": str(pair.depth.clock_quality),
                "rgb_clock_reason": str(pair.rgb.clock_reason),
                "depth_clock_reason": str(pair.depth.clock_reason),
                "rgb_clock_reset_count": int(pair.rgb.clock_reset_count),
                "depth_clock_reset_count": int(pair.depth.clock_reset_count),
                "rgb_dtype": str(pair.rgb.image.dtype),
                "depth_dtype": str(pair.depth.image.dtype),
                "rgb_shape": list(pair.rgb.image.shape),
                "depth_shape": list(pair.depth.image.shape),
                "rgb_c_contiguous": bool(pair.rgb.image.flags.c_contiguous),
                "depth_c_contiguous": bool(pair.depth.image.flags.c_contiguous),
                "rgb_strides": list(pair.rgb.image.strides),
                "depth_strides": list(pair.depth.image.strides),
                "depth_min": int(pair.depth.image.min()) if first_pair else None,
                "depth_max": int(pair.depth.image.max()) if first_pair else None,
                "depth_zero_count": int((pair.depth.image == 0).sum()) if first_pair else None,
            })

    started = bool(backend.start(on_pair))
    started_at = time.monotonic()
    cpu_started = time.process_time()
    rss_samples = []
    if started:
        deadline = time.monotonic() + max(0.1, args.duration)
        while time.monotonic() < deadline:
            rss_samples.append(current_rss_bytes())
            time.sleep(min(1.0, max(0.0, deadline - time.monotonic())))
    elapsed = time.monotonic() - started_at
    cpu_seconds = time.process_time() - cpu_started
    hardware_d2c = bool(backend.hardware_d2c_active()) if started else False
    stats = backend.sync_stats()
    performance = backend.performance_stats()
    backend.stop()
    _core.logger.set_callback(None)

    with lock:
        captured = list(pairs)
    deltas_ms = [abs(item["delta_ns"]) / 1_000_000.0 for item in captured]
    cpu_percent = 100.0 * cpu_seconds / elapsed if elapsed > 0 else 0.0
    rss_bytes = max(rss_samples, default=0)
    first = captured[0] if captured else {}
    last = captured[-1] if captured else {}
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
        "rgb_array": {
            "dtype": first.get("rgb_dtype", ""),
            "shape": first.get("rgb_shape", []),
            "strides": first.get("rgb_strides", []),
            "c_contiguous": first.get("rgb_c_contiguous", False),
            "channel_order": "BGR",
        },
        "depth_array": {
            "dtype": first.get("depth_dtype", ""),
            "shape": first.get("depth_shape", []),
            "strides": first.get("depth_strides", []),
            "c_contiguous": first.get("depth_c_contiguous", False),
            "min": first.get("depth_min"),
            "max": first.get("depth_max"),
            "zero_count": first.get("depth_zero_count"),
        },
        "last_array_contract_unchanged": bool(captured) and all((
            first.get("rgb_dtype") == last.get("rgb_dtype"),
            first.get("depth_dtype") == last.get("depth_dtype"),
            first.get("rgb_shape") == last.get("rgb_shape"),
            first.get("depth_shape") == last.get("depth_shape"),
            last.get("rgb_c_contiguous", False),
            last.get("depth_c_contiguous", False),
        )),
        "rgb_clock_quality_counts": {
            quality: sum(item["rgb_clock_quality"] == quality for item in captured)
            for quality in sorted({item["rgb_clock_quality"] for item in captured})
        },
        "depth_clock_quality_counts": {
            quality: sum(item["depth_clock_quality"] == quality for item in captured)
            for quality in sorted({item["depth_clock_quality"] for item in captured})
        },
        "max_clock_reset_count": max(
            (max(item["rgb_clock_reset_count"], item["depth_clock_reset_count"])
             for item in captured), default=0),
        "sync_matched": int(stats.matched),
        "sync_threshold_misses": int(stats.threshold_misses),
        "sync_rgb_trimmed": int(stats.rgb_trimmed),
        "sync_depth_trimmed": int(stats.depth_trimmed),
        "raw_rgb_fps": performance["raw_rgb_fps"],
        "raw_depth_fps": performance["raw_depth_fps"],
        "rgb_callback_p95_ms": performance["rgb_callback_p95_ms"],
        "depth_callback_p95_ms": performance["depth_callback_p95_ms"],
        "cpu_percent": cpu_percent,
        "rss_bytes": rss_bytes,
        "rss_growth_bytes": (
            rss_samples[-1] - rss_samples[0] if len(rss_samples) >= 2 else 0),
        "statuses": statuses,
        "native_logs": native_logs,
    }
    args.output.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if started and captured else 1


if __name__ == "__main__":
    raise SystemExit(main())
