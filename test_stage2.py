"""
Stage 2 integration test.
Run from python_version/:  python test_stage2.py
"""
import os
import sys
import tempfile
import time

sys.path.insert(0, ".")

print("=" * 60)
print("Stage 2 Integration Test")
print("=" * 60)

# ---- 1. Config loader ----
print("\n[1] Config loader")
from rehab_engine.config_loader import load_pipeline_config

config = load_pipeline_config()
assert config.device.rgb_width == 640
assert config.pose.enable_adaptive_roi is True
print("  load_pipeline_config(): OK")

# ---- 2. Course system ----
print("\n[2] Course system")
from rehab_engine.course import CourseRepository, CourseRunner, RunnerState

repo = CourseRepository()
for p in ["../configs/courses.json", "configs/courses.json"]:
    if os.path.exists(p):
        assert repo.load(p), f"Failed to load {p}: {repo.last_error}"
        break
assert len(repo.courses) > 0, "No courses loaded"
print(f"  Loaded: {len(repo.courses)} courses")

course = repo.find_by_id("upper_limb_shoulder_rom_basic")
assert course is not None
print(f"  Course: {course.course_name} ({len(course.actions)} actions)")

runner = CourseRunner()
assert runner.state == RunnerState.IDLE
runner.on_rest_tick = lambda r: None  # suppress timer thread
runner.start_course(course)
assert runner.state == RunnerState.TRAINING
assert runner.current_action.action_id == "M7"

from rehab_engine.scoring import ScoreResult

for i in range(8):
    runner.on_score_updated(ScoreResult(completed_count=i + 1, overall_score=85.0 + i, count=i + 1))
assert runner.state == RunnerState.RESTING, f"Expected RESTING, got {runner.state}"
print("  CourseRunner state machine: OK")

# ---- 3. Preview ----
print("\n[3] Preview system")
from rehab_engine.preview import PreviewComposer

composer = PreviewComposer()
joints_2d = [(320 + i * 5, 240 + i * 3, 0.9, True) for i in range(22)]
joints_3d = [(i * 0.05 - 0.5, -(i * 0.04 - 0.4), 2.0, 0.85, True) for i in range(22)]

composer.submit(
    pair_id=12, rgb_frame_id=20, depth_frame_id=21,
    host_ts_ns=1_000_000_000, rgb_width=640, rgb_height=480,
    depth_width=640, depth_height=480, pose_interval=6,
    joints_2d_raw=joints_2d,
    joints_3d=joints_3d,
    rgb_fps=30.0, pair_fps=28.5,
    bbox=(100, 50, 200, 300, True, "detect"),
    emg_status="EMG mock active",
    emg_rms=[0.5, 0.3],
    mirror=True,
)

frame = composer.latest_frame()
assert frame is not None and len(frame.joints_2d) == 22
assert frame.depth_is_hardware is False
assert frame.depth_image is None and not frame.joints_3d and not frame.has_valid_3d
assert frame.pair_id == 12 and frame.rgb_frame_id == 20 and frame.depth_frame_id == 21
assert frame.host_ts_ns == 1_000_000_000 and frame.pose_interval == 6
assert frame.bbox_mode == "detect"
assert frame.mirror
print("  PreviewComposer: OK")

trusted_composer = PreviewComposer()
trusted_composer.submit(
    joints_3d=joints_3d, depth_image=[[1000]], depth_is_hardware=True)
trusted_frame = trusted_composer.latest_frame()
assert trusted_frame.depth_is_hardware is True
assert trusted_frame.depth_image == [[1000]]
assert len(trusted_frame.joints_3d) == 22 and trusted_frame.has_valid_3d

# ---- 4. Recorder ----
print("\n[4] Recorder")
tmpdir = tempfile.mkdtemp(prefix="stroke_test_")
from rehab_engine.recorder import Skeleton3DRecorder, EmgRecorder

skel_rec = Skeleton3DRecorder()
skel_rec.start(tmpdir)
for fid in range(5):
    joints = [[i * 0.01 + fid * 0.001, i * 0.02, 2.0, 0.9, 1] for i in range(22)]
    skel_rec.record(
        timestamp_ns=1000000000 + fid * 33000000, frame_id=fid + 1, pair_id=fid + 1,
        dt_seconds=0.033, bbox_mode="detect", joints_3d=joints,
    )

stats = skel_rec.stats()
skel_rec.stop()
assert stats.frames == 5 and stats.rows == 5

csv_path = os.path.join(tmpdir, "skeleton_3d.csv")
assert os.path.exists(csv_path)
with open(csv_path) as f:
    rows = f.readlines()
    assert len(rows) == 6
    assert len(rows[0].strip().split(",")) == 67
    assert rows[0].startswith("frame_idx,00_waist_x")
    assert float(rows[1].split(",")[4]) == -0.01
assert os.path.exists(os.path.join(tmpdir, "skeleton_3d_detailed.csv"))
with open(os.path.join(tmpdir, "skeleton3d_debug.csv")) as f:
    assert len(f.readlines()) == 1 + 5 * 22
print(f"  Skeleton3DRecorder: {stats.frames} frames -> {csv_path}")

emg_rec = EmgRecorder()
emg_rec.start(tmpdir)
emg_rec.record_raw(1_000_000_000, 1, [100, 200])
emg_rec.record_feature(1_000_033_000, 1, 0, 0.5, 10.0, 0.05, 0.1, "SMOOTH_FLEX")
emg_rec.stop()
assert os.path.exists(os.path.join(tmpdir, "emg_raw.csv"))
assert os.path.exists(os.path.join(tmpdir, "emg_summary.json"))
print("  EmgRecorder: OK")

# ---- 5. Pipeline stub ----
print("\n[5] SensorPipeline (stub mode)")
from rehab_engine.sensor_pipeline import SensorPipeline

pipeline = SensorPipeline(config)
frames = []
pipeline.set_on_frame(lambda f: frames.append(f))
pipeline.start()
time.sleep(1.0)
stats = pipeline.performance_stats()
pipeline.stop()
assert len(frames) > 0
assert all(frame.depth_is_hardware is False for frame in frames)
assert all(frame.depth_image is None for frame in frames)
assert all(not frame.joints_3d for frame in frames)
assert {"yolo_ms", "pose_ms", "record_write_ms"}.issubset(stats)
print(f"  Frames: {len(frames)}, pair_fps={stats['pair_fps']:.1f}, processed={stats['processed']}")
print("  SensorPipeline start/stop: OK")

# ---- 6. Pipeline + Recording ----
print("\n[6] Pipeline recording")
pipeline2 = SensorPipeline(config)
pipeline2.set_on_frame(lambda f: None)
pipeline2.start()
session_dir = pipeline2.start_recording(tmpdir)
time.sleep(2.0)
pipeline2.stop_recording()
pipeline2.stop()

rec_stats = pipeline2.recording_stats()
rec_csv = os.path.join(session_dir, "skeleton_3d.csv")
assert os.path.exists(rec_csv)
with open(rec_csv) as f:
    lines = f.readlines()
assert len(lines) == 1
assert rec_stats["frames"] == 0
assert rec_stats["depth_frames"] == 0
print(f"  Recorded: {rec_stats['frames']} frames, CSV: {len(lines)-1} rows")
print("  Pipeline recording: OK")

# ---- Cleanup ----
import shutil

shutil.rmtree(tmpdir)

print()
print("=" * 60)
print("Stage 2: ALL TESTS PASSED")
print("=" * 60)
