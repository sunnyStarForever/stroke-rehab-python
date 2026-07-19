"""Exercise the real RGB-D pipeline, 3D output, scoring, and recording."""

from __future__ import annotations

import argparse
import json
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rehab_engine.config_loader import load_pipeline_config
from rehab_engine.scoring import ScoreBridge
from rehab_engine.sensor_pipeline import RecordingOptions, SensorPipeline


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--duration", type=float, default=15.0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    config = load_pipeline_config()
    config.emg.enabled = False
    pipeline = SensorPipeline(config)
    statuses: list[str] = []
    frames: list[dict[str, object]] = []
    scores: list[dict[str, object]] = []
    score_errors: list[str] = []
    lock = threading.Lock()
    score_bridge = ScoreBridge()
    score_bridge.on_score_updated = lambda result: scores.append({
        "overall_score": float(result.overall_score),
        "count": int(result.count),
        "status": str(result.status),
    })
    score_bridge.on_error = score_errors.append
    scoring_started = bool(score_bridge.start("M7", 5.0))
    frame_index = 0

    def on_frame(frame):
        nonlocal frame_index
        with lock:
            frames.append({
                "depth_is_hardware": bool(frame.depth_is_hardware),
                "has_depth": frame.depth_image is not None,
                "has_valid_3d": bool(frame.has_valid_3d),
                "valid_3d_count": sum(j.valid for j in frame.joints_3d),
                "pair_fps": float(frame.pair_fps),
                "delta_ms": float(frame.delta_ms),
            })
        if (scoring_started and frame.depth_is_hardware
                and frame.has_valid_3d and len(frame.joints_3d) >= 22):
            frame_index += 1
            score_bridge.submit_skeleton(
                frame_index, time.monotonic_ns(), frame.joints_3d[:22])

    pipeline.set_on_status(statuses.append)
    pipeline.set_on_frame(on_frame)
    started = bool(pipeline.start())
    session_dir = ""
    if started:
        deadline = time.monotonic() + 15.0
        while time.monotonic() < deadline:
            with lock:
                if any(item["depth_is_hardware"] for item in frames):
                    break
            time.sleep(0.1)
        session_dir = pipeline.start_recording(RecordingOptions(
            save_root=str(args.output),
            record_skeleton=True,
            record_rgb=True,
            record_depth=True,
            record_valid_3d_only=True,
        ))
        time.sleep(max(1.0, args.duration))
        pipeline.stop_recording()

    performance = pipeline.performance_stats()
    recording = pipeline.recording_stats()
    stopped = threading.Event()
    stop_result: list[object] = []
    pipeline.stop(lambda ok, message: (
        stop_result.extend([bool(ok), str(message)]), stopped.set()))
    stopped.wait(20.0)
    score_bridge.stop()

    with lock:
        captured = list(frames)
    summary = {
        "pipeline_started": started,
        "camera_status": pipeline.camera_status,
        "frame_count": len(captured),
        "hardware_frame_count": sum(
            bool(item["depth_is_hardware"]) for item in captured),
        "depth_image_frame_count": sum(bool(item["has_depth"]) for item in captured),
        "valid_3d_frame_count": sum(bool(item["has_valid_3d"]) for item in captured),
        "max_valid_3d_joints": max(
            (int(item["valid_3d_count"]) for item in captured), default=0),
        "performance": performance,
        "recording": recording,
        "session_dir": session_dir,
        "scoring_started": scoring_started,
        "score_count": len(scores),
        "scores": scores[-10:],
        "score_errors": score_errors,
        "stop_completed": stopped.is_set(),
        "stop_result": stop_result,
        "statuses": statuses,
    }
    output = args.output / "real_pipeline_summary.json"
    output.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if (started and summary["hardware_frame_count"]
                 and stopped.is_set()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
