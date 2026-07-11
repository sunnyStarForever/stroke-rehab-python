#!/usr/bin/env python3
"""
Standalone environment check — run this BEFORE launching the UI.
No Qt dependency — works in a headless terminal via SSH.

Usage:
    python check_env.py              # Quick check
    python check_env.py --camera     # Also test V4L2 camera read
    python check_env.py --verbose    # Detailed output including all files
"""

import os
import sys
import ctypes
import platform
import subprocess
import shutil
from pathlib import Path

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent  # stroke-rehab/

# Use ASCII-safe symbols that work on all terminals
_MARK_OK = "[PASS]"
_MARK_WARN = "[WARN]"
_MARK_ERR = "[FAIL]"


def header(title: str):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


def ok(msg: str):
    print(f"  {_MARK_OK} {msg}")


def warn(msg: str):
    print(f"  {_MARK_WARN} {msg}")


def err(msg: str):
    print(f"  {_MARK_ERR} {msg}")


def info(msg: str):
    print(f"    {msg}")


def check_file_size(path: Path, label: str = "") -> bool:
    if path.exists():
        size = path.stat().st_size
        if size == 0:
            err(f"{label or path.name}: zero size (possibly broken symlink)")
            return False
        sz_mb = size / (1024 * 1024)
        ok(f"{label or path.name}: {sz_mb:.1f} MB")
        return True
    else:
        err(f"{label or path.name}: not found ({path})")
        return False


def run(verbose: bool = False, camera_test: bool = False):
    """Run the full environment check."""

    header("Stroke Rehab -- Environment Diagnostics")

    # 1. System
    header("1. System Info")
    ok(f"Arch: {platform.machine()}")
    ok(f"OS: {platform.system()} {platform.release()}")
    ok(f"Python: {sys.version}")
    ok(f"Python path: {sys.executable}")
    conda_env = os.environ.get("CONDA_ENV", os.environ.get("CONDA_DEFAULT_ENV", ""))
    if conda_env:
        ok(f"Conda env: {conda_env}")
    else:
        warn("Conda env not detected")

    # 2. Python packages
    header("2. Python Packages")
    for mod_name, display in [
        ("numpy", "NumPy"),
        ("cv2", "OpenCV Python"),
        ("yaml", "PyYAML"),
        ("PyQt5", "PyQt5"),
        ("qfluentwidgets", "QFluentWidgets"),
    ]:
        try:
            m = __import__(mod_name)
            ver = getattr(m, "__version__", "?")
            ok(f"{display}: v{ver}")
        except ImportError:
            err(f"{display}: not installed (pip install {mod_name})")

    # 3. C++ Engine .so
    header("3. C++ Engine (.so)")
    build_dir = HERE / "build"
    if build_dir.is_dir():
        so_files = list(build_dir.glob("rehab_engine*.so"))
        if so_files:
            for f in so_files:
                check_file_size(f, f"Engine .so: {f.name}")

            # Try to import
            sys.path.insert(0, str(HERE))
            try:
                from rehab_engine import _STUB_MODE
                if _STUB_MODE:
                    warn("Engine mode: STUB (C++ engine NOT loaded)")
                    warn(".so exists but Python could not import it")
                else:
                    ok("Engine mode: FULL (C++ engine loaded!)")
            except ImportError as e:
                warn(f"Import failed: {e}")
                warn(".so exists but Python cannot import (Python version mismatch?)")
        else:
            warn("build/ exists but no .so found")
            warn("Run: cmake --build build -j$(nproc)")
    else:
        warn("build/ directory does not exist")
        warn("Run: bash setup_board.sh")

    # 4. Cameras
    header("4. Cameras (/dev/video*)")
    cameras_found = []
    for idx in range(4):
        dev = f"/dev/video{idx}"
        if os.path.exists(dev):
            cameras_found.append(dev)
            if os.access(dev, os.R_OK):
                ok(f"{dev}: accessible")
            else:
                warn(f"{dev}: exists but no read permission (need video group)")

            if camera_test:
                try:
                    result = subprocess.run(
                        ["v4l2-ctl", "-d", dev, "--all"],
                        capture_output=True, text=True, timeout=5)
                    for line in result.stdout.splitlines()[:8]:
                        if line.strip():
                            info(f"  {line.strip()}")
                except FileNotFoundError:
                    info("  (v4l2-ctl not installed)")
                except subprocess.TimeoutExpired:
                    info("  (v4l2-ctl timeout)")
                except Exception as e:
                    info(f"  (v4l2-ctl error: {e})")

    if not cameras_found:
        err("No /dev/video* devices found")
        info("Check: lsusb (USB devices), ls -la /dev/video* (permissions)")

    # 5. Depth Camera (OpenNI2)
    header("5. Depth Camera (OpenNI2)")
    openni_paths = [
        PROJECT_ROOT / "including" / "OpenNI" / "sdk" / "libs" / "libOpenNI2.so",
        Path("/usr/lib/libOpenNI2.so"),
        Path("/usr/local/lib/libOpenNI2.so"),
    ]
    found_openni = False
    for p in openni_paths:
        if p.exists() and p.stat().st_size > 0:
            ok(f"OpenNI2: {p}")
            try:
                ctypes.cdll.LoadLibrary(str(p))
                ok("  Library loadable")
            except OSError as e:
                warn(f"  Library load failed: {e}")
            found_openni = True
            break
    if not found_openni:
        warn("OpenNI2 library not found (depth camera will not work)")
        info("Install: sudo apt install libopenni2-dev")

    # 6. ONNX Runtime
    header("6. ONNX Runtime (AI inference)")
    onnx_dir = PROJECT_ROOT / "including" / "onnxruntime" / "lib"
    if onnx_dir.is_dir():
        for so in sorted(onnx_dir.glob("libonnxruntime.so*")):
            size = so.stat().st_size
            if size == 0:
                err(f"{so.name}: 0 bytes (BROKEN SYMLINK!)")
                info(f"Fix: cd {onnx_dir}")
                info("  rm -f libonnxruntime.so libonnxruntime.so.1")
                info("  ln -s libonnxruntime.so.1.25.0 libonnxruntime.so.1")
                info("  ln -s libonnxruntime.so.1 libonnxruntime.so")
            else:
                ok(f"{so.name}: {size / (1024*1024):.1f} MB")
    else:
        err(f"ONNX Runtime directory not found: {onnx_dir}")

    # 7. AI Models
    header("7. AI Models")
    models = [
        (PROJECT_ROOT / "including" / "yolov8n" / "yolov8n.onnx", "YOLO detector"),
        (PROJECT_ROOT / "including" / "rtmpose-t" / "end2end.onnx", "RTMPose pose"),
    ]
    for model_path, name in models:
        check_file_size(model_path, name)

    # 8. EMG devices
    header("8. EMG Devices")
    for dev in ["/dev/rpmsg_ctrl0", "/dev/rpmsg0", "/dev/rfcomm0"]:
        if os.path.exists(dev):
            ok(f"{dev}: detected")
        else:
            info(f"{dev}: not found (EMG will use mock data)")

    # 9. Config files
    header("9. Config Files")
    configs_dir = PROJECT_ROOT / "configs"
    for cfg in ["device.yaml", "emg.yaml", "courses.json"]:
        p = configs_dir / cfg
        if p.exists():
            ok(f"{cfg}: {p.stat().st_size} bytes")
        else:
            err(f"{cfg}: not found")

    # 10. Disk space
    header("10. Disk Space")
    disk = shutil.disk_usage(str(PROJECT_ROOT))
    free_gb = disk.free / (1024**3)
    total_gb = disk.total / (1024**3)
    if free_gb > 1:
        ok(f"Free: {free_gb:.1f} GB / {total_gb:.1f} GB")
    else:
        err(f"Low disk space: {free_gb:.1f} GB / {total_gb:.1f} GB")

    records_dir = PROJECT_ROOT / "records"
    if records_dir.is_dir():
        ok(f"Recordings dir: {records_dir} (writable: {os.access(str(records_dir), os.W_OK)})")
    else:
        info(f"Recordings dir will be auto-created: {records_dir}")

    # Summary
    header("Diagnostics Complete")
    print()
    print("  If all items show [PASS], the system is ready.")
    print("  [WARN] items are non-critical but may affect functionality.")
    print("  [FAIL] items must be fixed before running.")
    print()
    print("  Start UI:     cd python_version && python main.py")
    print("  Build engine: bash setup_board.sh")
    print()


# ================================================================
if __name__ == "__main__":
    verbose = "--verbose" in sys.argv
    camera_test = "--camera" in sys.argv
    run(verbose=verbose, camera_test=camera_test)