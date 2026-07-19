#!/usr/bin/env python3
"""Headless target-board acceptance for performance metrics and recording."""

from __future__ import annotations

import argparse
import csv
import json
import threading
import time
from pathlib import Path

import cv2

from rehab_engine import PipelineConfig
from rehab_engine.sensor_pipeline import RecordingOptions, SensorPipeline


def rss_mib() -> float:
    for line in Path("/proc/self/status").read_text(encoding="utf-8").splitlines():
        if line.startswith("VmRSS:"):
            return float(line.split()[1]) / 1024.0
    return 0.0


def video_readable(path: str) -> bool:
    if not path:
        return True
    capture = cv2.VideoCapture(path)
    ok, _frame = capture.read()
    capture.release()
    return bool(ok)


def timestamps_monotonic(path: str) -> bool:
    if not path or not Path(path).exists():
        return True
    with open(path, newline="", encoding="utf-8") as source:
        timestamps = [int(row["sync_ts_ns"]) for row in csv.DictReader(source)]
    return all(right >= left for left, right in zip(timestamps, timestamps[1:]))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--duration", type=float, default=30.0)
    parser.add_argument("--mode", choices=("none", "rgb", "depth", "both"), default="none")
    parser.add_argument("--queue-capacity", type=int, default=90)
    parser.add_argument("--worker-delay-ms", type=float, default=0.0)
    parser.add_argument("--output", default="validation-output")
    args = parser.parse_args()

    config = PipelineConfig()
    config.recording_queue_capacity = args.queue_capacity
    pipeline = SensorPipeline(config)
    if args.worker_delay_ms > 0:
        original = pipeline._infer_pose_full

        def delayed(*call_args, **call_kwargs):
            result = original(*call_args, **call_kwargs)
            time.sleep(args.worker_delay_ms / 1000.0)
            return result

        pipeline._infer_pose_full = delayed

    result = {
        "mode": args.mode,
        "duration_seconds": args.duration,
        "queue_capacity": args.queue_capacity,
        "worker_delay_ms": args.worker_delay_ms,
        "rss_start_mib": rss_mib(),
        "samples": [],
    }
    if not pipeline.start():
        result["error"] = "pipeline start failed"
        print(json.dumps(result, ensure_ascii=False))
        return 2

    session = ""
    if args.mode != "none":
        session = pipeline.start_recording(RecordingOptions(
            save_root=args.output,
            record_skeleton=False,
            record_rgb=args.mode in ("rgb", "both"),
            record_depth=args.mode in ("depth", "both"),
        ))

    started = time.monotonic()
    deadline = started + max(1.0, args.duration)
    try:
        while time.monotonic() < deadline:
            time.sleep(min(1.0, max(0.0, deadline - time.monotonic())))
            stats = pipeline.performance_stats()
            result["samples"].append({key: stats.get(key, 0) for key in (
                "raw_rgb_fps", "raw_depth_fps", "sync_fps", "worker_fps",
                "pose_fps", "rgb_callback_p95_ms", "depth_callback_p95_ms",
                "recording_write_fps", "recording_queue_depth",
                "recording_queue_high_watermark", "recording_dropped",
                "dropped_pairs",
            )})
            result["samples"][-1]["rss_mib"] = rss_mib()
    finally:
        if pipeline.is_recording:
            pipeline.stop_recording()
        recording = pipeline.recording_stats()
        stopped = threading.Event()
        stop_result = []
        pipeline.stop(lambda ok, message: (stop_result.append((ok, message)), stopped.set()))
        stopped.wait(10.0)

    result["elapsed_seconds"] = time.monotonic() - started
    result["rss_end_mib"] = rss_mib()
    result["rss_growth_mib"] = result["rss_end_mib"] - result["rss_start_mib"]
    result["recording"] = recording
    result["stop_result"] = stop_result
    result["rgb_video_readable"] = video_readable(recording.get("rgb_path", ""))
    result["depth_video_readable"] = video_readable(recording.get("depth_path", ""))
    result["timestamps_monotonic"] = timestamps_monotonic(
        recording.get("metadata_path", ""))
    result["session_dir"] = session
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if stop_result and stop_result[0][0] else 3


if __name__ == "__main__":
    raise SystemExit(main())
