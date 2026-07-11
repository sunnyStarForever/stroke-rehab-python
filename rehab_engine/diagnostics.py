"""
System diagnostics module — checks hardware, engine, and environment status.
Run at startup to provide clear feedback about what's working and what's not.
"""

import os
import sys
import platform
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple


@dataclass
class DiagItem:
    name: str
    status: str          # "OK", "WARN", "ERROR", "DISABLED"
    detail: str = ""
    hint: str = ""


@dataclass
class Diagnostics:
    """Complete system diagnostic report."""
    items: List[DiagItem] = field(default_factory=list)

    def add(self, name: str, status: str, detail: str = "", hint: str = ""):
        self.items.append(DiagItem(name, status, detail, hint))

    def summary(self) -> str:
        ok = sum(1 for i in self.items if i.status == "OK")
        warn = sum(1 for i in self.items if i.status == "WARN")
        err = sum(1 for i in self.items if i.status == "ERROR")
        disabled = sum(1 for i in self.items if i.status == "DISABLED")
        parts = []
        if ok:
            parts.append(f"{ok} OK")
        if warn:
            parts.append(f"{warn} 警告")
        if err:
            parts.append(f"{err} 错误")
        if disabled:
            parts.append(f"{disabled} 已禁用")
        return ", ".join(parts) if parts else "未检测"

    def all_ok(self) -> bool:
        return all(i.status in ("OK", "DISABLED") for i in self.items)

    def errors(self) -> List[DiagItem]:
        return [i for i in self.items if i.status == "ERROR"]

    def warnings(self) -> List[DiagItem]:
        return [i for i in self.items if i.status == "WARN"]

    def format_lines(self, ascii_only: bool = False) -> List[str]:
        lines = []
        if ascii_only:
            emoji = {"OK": "  [PASS]", "WARN": "  [WARN]", "ERROR": "  [FAIL]", "DISABLED": "  [OFF]"}
        else:
            # Try Unicode, fall back to ASCII if stdout doesn't support it
            try:
                '✓'.encode(sys.stdout.encoding or 'utf-8')
            except (UnicodeEncodeError, LookupError):
                ascii_only = True
                emoji = {"OK": "  [PASS]", "WARN": "  [WARN]", "ERROR": "  [FAIL]", "DISABLED": "  [OFF]"}
            else:
                emoji = {"OK": "  ✓", "WARN": "  ⚠", "ERROR": "  ✗", "DISABLED": "  ○"}
        for item in self.items:
            prefix = emoji.get(item.status, "  ?")
            line = f"{prefix} {item.name}: {item.detail}"
            if item.hint:
                line += f"  ({item.hint})"
            lines.append(line)
        return lines


def _find_project_root() -> Path:
    root = os.environ.get("STROKE_REHAB_ROOT")
    if root:
        return Path(root)
    p = Path(__file__).resolve().parent  # rehab_engine/
    for _ in range(5):
        if (p / "python_version").is_dir() and (p / "configs").is_dir():
            return p
        if (p / "configs").is_dir():
            return p
        p = p.parent
    return Path.cwd()


def _check_file(path: Path) -> Tuple[bool, str]:
    if path.exists():
        size_mb = path.stat().st_size / (1024 * 1024)
        return True, f"{path} ({size_mb:.1f} MB)"
    return False, f"不存在: {path}"


def run_diagnostics(config=None) -> Diagnostics:
    """
    Run a comprehensive system diagnostic.

    Returns a Diagnostics object that can be printed to console
    or displayed in the UI.
    """
    d = Diagnostics()
    root = _find_project_root()

    # ================================================================
    # 1. Platform
    # ================================================================
    d.add("操作系统", "OK",
          f"{platform.system()} {platform.release()}",
          f"架构: {platform.machine()}")

    # ================================================================
    # 2. Python environment
    # ================================================================
    py_ver = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    py_arch = "64-bit" if sys.maxsize > 2**32 else "32-bit"
    d.add("Python 版本", "OK", f"Python {py_ver} ({py_arch})")
    d.add("Python 路径", "OK", sys.executable)

    # Check if we're in a conda environment
    conda_env = os.environ.get("CONDA_DEFAULT_ENV", "")
    if conda_env:
        d.add("Conda 环境", "OK", conda_env)
    else:
        d.add("Conda 环境", "WARN", "未检测到 conda 环境",
              "建议在 stroke38 环境中运行")

    # ================================================================
    # 3. C++ Engine (rehab_engine._core)
    # ================================================================
    try:
        engine_so_path = None
        # Try to find the compiled .so
        build_dir = Path(__file__).resolve().parent.parent / "build"
        if build_dir.is_dir():
            so_files = list(build_dir.glob("rehab_engine*.so"))
            if so_files:
                engine_so_path = so_files[0]

        if engine_so_path:
            so_size_mb = engine_so_path.stat().st_size / (1024 * 1024)
            d.add("C++ 引擎 (.so)", "OK",
                  f"{engine_so_path.name} ({so_size_mb:.1f} MB)")
        else:
            # Check if we can import it
            try:
                from . import _core  # noqa: F401
                d.add("C++ 引擎", "OK", "已导入 (imported)")
            except ImportError:
                d.add("C++ 引擎 (.so)", "WARN",
                      "未找到编译产物 — 使用 STUB 模式",
                      "运行 setup_board.sh 编译引擎")
    except Exception as e:
        d.add("C++ 引擎 (.so)", "ERROR", str(e),
              "重新编译: cmake --build build -j$(nproc)")

    # Stub mode check
    try:
        from . import _STUB_MODE
        if _STUB_MODE:
            d.add("引擎模式", "WARN", "STUB（模拟数据）",
                  "编译 C++ 引擎以使用真实硬件")
        else:
            d.add("引擎模式", "OK", "FULL（真实引擎）")
    except Exception:
        d.add("引擎模式", "WARN", "未知", "检查 rehab_engine 导入")

    # ================================================================
    # 4. Python packages
    # ================================================================
    PKG_CHECKS = [
        ("numpy", "NumPy"),
        ("cv2", "OpenCV Python"),
        ("yaml", "PyYAML"),
        ("PyQt5", "PyQt5"),
        ("qfluentwidgets", "QFluentWidgets"),
    ]
    for mod_name, display in PKG_CHECKS:
        try:
            m = __import__(mod_name)
            ver = getattr(m, "__version__", "?")
            d.add(display, "OK", f"v{ver}")
        except ImportError:
            d.add(display, "ERROR", "未安装",
                  f"pip install {mod_name}")

    # ================================================================
    # 5. Camera devices (V4L2)
    # ================================================================
    camera_found = False
    for dev_idx in range(4):
        dev_path = f"/dev/video{dev_idx}"
        if os.path.exists(dev_path):
            camera_found = True
            # Try to get device name via v4l2-ctl
            detail = dev_path
            try:
                import subprocess
                result = subprocess.run(
                    ["v4l2-ctl", "-d", dev_path, "--all"],
                    capture_output=True, text=True, timeout=3)
                # Extract driver/card info
                for line in result.stdout.splitlines()[:5]:
                    line = line.strip()
                    if line:
                        detail += f" [{line}]"
                        break
            except Exception:
                pass
            d.add(f"摄像头 /dev/video{dev_idx}", "OK", detail)
    if not camera_found:
        d.add("摄像头", "ERROR", "未检测到任何 /dev/video* 设备",
              "确认 USB 摄像头已连接并已启用")

    # ================================================================
    # 6. Depth camera (OpenNI2)
    # ================================================================
    openni_lib = root / "including" / "OpenNI" / "sdk" / "libs" / "libOpenNI2.so"
    if openni_lib.exists():
        d.add("OpenNI2 库", "OK", str(openni_lib))
    else:
        # Check system paths
        import ctypes.util
        sys_openni = ctypes.util.find_library("OpenNI2")
        if sys_openni:
            d.add("OpenNI2 库", "OK", f"系统路径: {sys_openni}")
        else:
            d.add("OpenNI2 库", "WARN", "未找到",
                  "Depth 深度相机将不可用")

    # ================================================================
    # 7. ONNX Runtime + AI models
    # ================================================================
    onnx_lib = root / "including" / "onnxruntime" / "lib" / "libonnxruntime.so"
    exists, detail = _check_file(onnx_lib)
    if exists:
        # Check if it's a real file or broken symlink
        if onnx_lib.stat().st_size == 0:
            d.add("ONNX Runtime", "ERROR", "符号链接损坏 (0 字节)",
                  "修复: rm libonnxruntime.so && ln -s libonnxruntime.so.1.25.0 libonnxruntime.so")
        else:
            d.add("ONNX Runtime", "OK", detail)
    else:
        # Check for any version
        onnx_dir = root / "including" / "onnxruntime" / "lib"
        if onnx_dir.is_dir():
            found = list(onnx_dir.glob("libonnxruntime.so*"))
            if found:
                d.add("ONNX Runtime", "OK", f"已找到 {len(found)} 个文件")
            else:
                d.add("ONNX Runtime", "ERROR", "库文件不存在")
        else:
            d.add("ONNX Runtime", "ERROR", "目录不存在",
                  "检查 ~/stroke-rehab/including/onnxruntime/")

    # YOLO model
    yolo_model = root / "including" / "yolov8n" / "yolov8n.onnx"
    exists, detail = _check_file(yolo_model)
    d.add("YOLO 检测模型", "OK" if exists else "ERROR",
          detail, "检查 including/yolov8n/yolov8n.onnx" if not exists else "")

    # RTMPose model
    rtm_model = root / "including" / "rtmpose-t" / "end2end.onnx"
    exists, detail = _check_file(rtm_model)
    d.add("RTMPose 姿态模型", "OK" if exists else "ERROR",
          detail, "检查 including/rtmpose-t/end2end.onnx" if not exists else "")

    # ================================================================
    # 8. EMG devices
    # ================================================================
    rpmsg_ctrl = "/dev/rpmsg_ctrl0"
    if os.path.exists(rpmsg_ctrl):
        d.add("EMG (RPMsg)", "OK", f"{rpmsg_ctrl} 已检测到")
    else:
        d.add("EMG (RPMsg)", "DISABLED",
              f"{rpmsg_ctrl} 未检测到",
              "EMG 将使用模拟数据")

    rfcomm = "/dev/rfcomm0"
    if os.path.exists(rfcomm):
        d.add("EMG (Serial)", "OK", f"{rfcomm} 已检测到")
    else:
        d.add("EMG (Serial)", "DISABLED",
              "未检测到蓝牙串口设备",
              "EMG 串口模式不可用")

    # ================================================================
    # 9. Config files
    # ================================================================
    for cfg_name in ["device.yaml", "emg.yaml", "courses.json"]:
        cfg_path = root / "configs" / cfg_name
        exists, detail = _check_file(cfg_path)
        d.add(f"配置文件: {cfg_name}", "OK" if exists else "WARN",
              detail)

    # ================================================================
    # 10. Output directory
    # ================================================================
    records_dir = root / "records"
    if records_dir.is_dir():
        writable = os.access(str(records_dir), os.W_OK)
        if writable:
            d.add("录制输出目录", "OK", str(records_dir))
        else:
            d.add("录制输出目录", "ERROR", "目录不可写",
                  f"chmod 755 {records_dir}")
    else:
        d.add("录制输出目录", "OK", "将自动创建")

    # ================================================================
    # 11. OpenCV C++ (system library, used by engine)
    # ================================================================
    opencv_config = None
    for search in [
        "/usr/lib/aarch64-linux-gnu/cmake/opencv4/OpenCVConfig.cmake",
        "/usr/lib/x86_64-linux-gnu/cmake/opencv4/OpenCVConfig.cmake",
    ]:
        if os.path.exists(search):
            opencv_config = search
            break
    if opencv_config:
        d.add("OpenCV C++ 开发库", "OK", opencv_config)
    else:
        d.add("OpenCV C++ 开发库", "WARN", "未找到 cmake 配置",
              "安装: sudo apt install libopencv-dev")

    return d


def print_diagnostics(diag: Diagnostics) -> None:
    """Print diagnostics to console in a formatted manner."""
    # Detect if we can use Unicode characters
    use_unicode = True
    try:
        '✓'.encode(sys.stdout.encoding or 'utf-8')
    except (UnicodeEncodeError, LookupError):
        use_unicode = False

    sep = "=" * 58
    print(f"\n{sep}")
    print("  Stroke Rehab — System Diagnostics")
    print(f"  Status: {diag.summary()}")
    print(sep)
    for line in diag.format_lines(ascii_only=not use_unicode):
        print(line)
    print(sep)

    errors = diag.errors()
    warnings = diag.warnings()
    if errors:
        print(f"\n  !! {len(errors)} error(s) to fix:")
        for e in errors:
            print(f"     - {e.name}: {e.detail}")
            if e.hint:
                print(f"       -> {e.hint}")
    if warnings:
        print(f"\n  !! {len(warnings)} warning(s):")
        for w in warnings:
            print(f"     - {w.name}: {w.detail}")
    print()