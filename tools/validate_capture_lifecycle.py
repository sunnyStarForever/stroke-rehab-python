"""Repeat trusted RGB-D pipeline start/stop cycles and record release evidence."""

from __future__ import annotations

import argparse
import json
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rehab_engine.config_loader import load_pipeline_config
from rehab_engine.sensor_pipeline import SensorPipeline


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cycles", type=int, default=3)
    parser.add_argument("--run-seconds", type=float, default=3.0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    results = []
    for index in range(max(1, args.cycles)):
        config = load_pipeline_config()
        config.emg.enabled = False
        pipeline = SensorPipeline(config)
        hardware_frames = 0
        frame_lock = threading.Lock()

        def on_frame(frame):
            nonlocal hardware_frames
            if frame.depth_is_hardware:
                with frame_lock:
                    hardware_frames += 1

        pipeline.set_on_frame(on_frame)
        started = bool(pipeline.start())
        deadline = time.monotonic() + 15.0
        while started and time.monotonic() < deadline:
            with frame_lock:
                if hardware_frames:
                    break
            time.sleep(0.1)
        time.sleep(max(0.5, args.run_seconds))
        stopped = threading.Event()
        stop_result = []
        pipeline.stop(lambda ok, message: (
            stop_result.extend([bool(ok), str(message)]), stopped.set()))
        stop_completed = stopped.wait(20.0)
        with frame_lock:
            count = hardware_frames
        results.append({
            "cycle": index + 1,
            "started": started,
            "hardware_frames": count,
            "stop_completed": stop_completed,
            "stop_result": stop_result,
            "camera_status": pipeline.camera_status,
        })
        if not (started and count and stop_completed
                and stop_result and stop_result[0]):
            break
        time.sleep(1.0)

    summary = {
        "requested_cycles": max(1, args.cycles),
        "completed_cycles": len(results),
        "all_passed": len(results) == max(1, args.cycles) and all(
            item["started"] and item["hardware_frames"]
            and item["stop_completed"] and item["stop_result"][0]
            for item in results
        ),
        "cycles": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return 0 if summary["all_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
