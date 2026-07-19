"""Capture and characterize attested hardware depth without starting the GUI."""

from __future__ import annotations

import argparse
import json
import sys
import threading
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rehab_engine import _core


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--duration", type=float, default=5.0)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    driver = _core.DepthCaptureOpenNI()
    config = _core.DeviceConfig()
    lock = threading.Lock()
    frames: list[np.ndarray] = []
    frame_ids: list[int] = []
    timestamps: list[int] = []
    pixel_formats: list[str] = []
    logs: list[dict[str, str]] = []

    _core.logger.set_callback(
        lambda level, message: logs.append(
            {"level": str(level), "message": str(message)}))

    def on_depth(payload, width, height, ts_ns, frame_id, *extra):
        encoded = np.frombuffer(bytes(payload), dtype=np.uint8)
        image = cv2.imdecode(encoded, cv2.IMREAD_UNCHANGED)
        if image is None or image.dtype != np.uint16:
            return
        with lock:
            frames.append(image.copy())
            frame_ids.append(int(frame_id))
            timestamps.append(int(ts_ns))
            pixel_formats.append(str(extra[2]) if len(extra) >= 3 else "")

    started = bool(driver.start(config, on_depth))
    deadline = time.monotonic() + 5.0
    while (not driver.real_depth_active() and driver.is_running()
           and time.monotonic() < deadline):
        time.sleep(0.02)
    active_at_start = bool(driver.real_depth_active())
    if active_at_start:
        time.sleep(max(0.1, args.duration))
    active_before_stop = bool(driver.real_depth_active())
    driver.stop()

    with lock:
        captured = list(frames)
        ids = list(frame_ids)
        times = list(timestamps)
        formats = list(pixel_formats)

    summary: dict[str, object] = {
        "started": started,
        "real_depth_active_at_start": active_at_start,
        "real_depth_active_before_stop": active_before_stop,
        "real_depth_active_after_stop": bool(driver.real_depth_active()),
        "frame_count": len(captured),
        "frame_ids_monotonic": ids == sorted(ids),
        "pixel_formats": sorted(set(formats)),
        "native_logs": logs,
    }
    if captured:
        first = captured[0]
        last = captured[-1]
        valid = first[first > 0]
        diff_x = np.diff(first.astype(np.int32), axis=1)
        diff_y = np.diff(first.astype(np.int32), axis=0)
        synthetic_gradient_score = float(
            ((diff_x == 1).mean() + (diff_y == 1).mean()) / 2.0
        )
        changed = np.abs(last.astype(np.int32) - first.astype(np.int32))
        summary.update({
            "shape": list(first.shape),
            "dtype": str(first.dtype),
            "zero_fraction": float((first == 0).mean()),
            "valid_min_mm": int(valid.min()) if valid.size else 0,
            "valid_max_mm": int(valid.max()) if valid.size else 0,
            "unique_values": int(np.unique(first).size),
            "synthetic_gradient_score": synthetic_gradient_score,
            "changed_pixel_fraction": float((changed > 0).mean()),
            "max_temporal_change_mm": int(changed.max()),
            "duration_seconds": (
                (times[-1] - times[0]) / 1_000_000_000.0
                if len(times) > 1 else 0.0
            ),
        })
        for index in sorted({0, len(captured) // 2, len(captured) - 1}):
            cv2.imwrite(str(args.output / f"depth_{index:04d}.png"), captured[index])

    (args.output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    _core.logger.set_callback(None)
    return 0 if active_at_start and len(captured) > 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
