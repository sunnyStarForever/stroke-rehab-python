"""Python-main runtime diagnostics with platform-aware hardware checks."""

from __future__ import annotations

import importlib
import importlib.metadata
import os
import platform
import shutil
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple


@dataclass
class DiagItem:
    name: str
    status: str
    detail: str = ""
    hint: str = ""


@dataclass
class Diagnostics:
    items: List[DiagItem] = field(default_factory=list)

    def add(self, name: str, status: str, detail: str = "", hint: str = ""):
        self.items.append(DiagItem(name, status, detail, hint))

    def summary(self) -> str:
        labels = (("OK", "正常"), ("WARN", "警告"), ("ERROR", "错误"),
                  ("DISABLED", "未启用"))
        parts = [f"{sum(item.status == status for item in self.items)}项{label}"
                 for status, label in labels
                 if any(item.status == status for item in self.items)]
        return " / ".join(parts) if parts else "未检测"

    def all_ok(self) -> bool:
        return all(item.status in ("OK", "DISABLED") for item in self.items)

    def errors(self) -> List[DiagItem]:
        return [item for item in self.items if item.status == "ERROR"]

    def warnings(self) -> List[DiagItem]:
        return [item for item in self.items if item.status == "WARN"]

    def format_lines(self, ascii_only: bool = False) -> List[str]:
        markers = ({"OK": "[PASS]", "WARN": "[WARN]", "ERROR": "[FAIL]",
                    "DISABLED": "[OFF]"} if ascii_only else
                   {"OK": "✓", "WARN": "⚠", "ERROR": "✗", "DISABLED": "○"})
        return [
            f"  {markers.get(item.status, '?')} {item.name}: {item.detail}"
            + (f"（{item.hint}）" if item.hint else "")
            for item in self.items
        ]


def _python_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _repo_root() -> Path:
    return _python_root().parent


def _configured_project_root() -> Optional[Path]:
    raw = os.environ.get("STROKE_REHAB_ROOT", "").strip()
    return Path(raw).expanduser().resolve() if raw else None


def _stroke_root() -> Path:
    configured = _configured_project_root()
    candidates = []
    if configured:
        candidates.extend((configured, configured / "stroke-rehab"))
    candidates.extend((_repo_root() / "stroke-rehab", _python_root()))
    return next((path for path in candidates if (path / "including").is_dir()), candidates[0])


def _check_file(path: Path) -> Tuple[bool, str]:
    if path.is_file():
        return True, f"{path}（{path.stat().st_size / 1024 / 1024:.1f} MB）"
    return False, f"不存在：{path}"


def _package_version(module_name: str) -> Tuple[bool, str]:
    distribution_names = {
        "cv2": ("opencv-python", "opencv-contrib-python", "opencv-python-headless"),
        "yaml": ("PyYAML",),
        "PyQt5": ("PyQt5",),
        "qfluentwidgets": ("PyQt-Fluent-Widgets",),
        "serial": ("pyserial",),
    }.get(module_name, (module_name,))
    for distribution_name in distribution_names:
        try:
            return True, importlib.metadata.version(distribution_name)
        except importlib.metadata.PackageNotFoundError:
            continue
    try:
        module = importlib.import_module(module_name)
        return True, str(getattr(module, "__version__", "已安装"))
    except Exception as exc:
        return False, str(exc)


def run_diagnostics(config=None) -> Diagnostics:
    if config is None:
        try:
            from .config_loader import load_pipeline_config
            config = load_pipeline_config()
        except Exception:
            config = None

    diagnostics = Diagnostics()
    python_root = _python_root()
    stroke_root = _stroke_root()

    diagnostics.add("操作系统", "OK", f"{platform.system()} {platform.release()} / {platform.machine()}")
    diagnostics.add("Python", "OK", f"{platform.python_version()} / {sys.executable}")

    try:
        from . import _CORE_READY
        if _CORE_READY:
            diagnostics.add("原生硬件适配器", "OK", "_core 已加载；仅接受真实 RGB-D 采集")
        else:
            diagnostics.add(
                "原生硬件适配器", "ERROR", "真实采集核心不可用，不会启用模拟采集",
                "请构建/部署包含 V4L2 RGB 与 OpenNI2 Depth 的 _core")
    except Exception as exc:
        diagnostics.add("原生硬件适配器", "ERROR", f"真实采集核心检查失败：{exc}")

    packages = (
        ("numpy", "NumPy", True), ("cv2", "OpenCV Python", True),
        ("yaml", "PyYAML", True), ("onnxruntime", "ONNX Runtime Python", True),
        ("PyQt5", "PyQt5", True), ("qfluentwidgets", "QFluentWidgets", True),
        ("bleak", "Bleak", False), ("serial", "pyserial", False),
        ("pyttsx3", "pyttsx3", False),
    )
    for module, label, required in packages:
        ok, detail = _package_version(module)
        diagnostics.add(
            label, "OK" if ok else ("ERROR" if required else "WARN"),
            detail if ok else "未安装",
            "执行 pip install -r requirements.txt" if not ok else "")

    model_files = (
        ("YOLO 检测模型", stroke_root / "including/yolov8n/yolov8n.onnx"),
        ("RTMPose 姿态模型", stroke_root / "including/rtmpose-t/end2end.onnx"),
        ("RTMPose pipeline", stroke_root / "including/rtmpose-t/pipeline.json"),
    )
    for name, path in model_files:
        exists, detail = _check_file(path)
        diagnostics.add(name, "OK" if exists else "ERROR", detail,
                        "检查模型资产或 STROKE_REHAB_ROOT" if not exists else "")

    scorer = stroke_root / "tools/scoring_engine/score_server.py"
    exists, detail = _check_file(scorer)
    diagnostics.add("实时评分服务", "OK" if exists else "ERROR", detail)

    for name in ("device.yaml", "emg.yaml", "courses.json", "calibration.yaml"):
        path = python_root / "configs" / name
        exists, detail = _check_file(path)
        diagnostics.add(f"配置文件 {name}", "OK" if exists else "ERROR", detail)

    if platform.system() == "Linux":
        videos = sorted(Path("/dev").glob("video*"))
        diagnostics.add(
            "V4L2 RGB 设备", "OK" if videos else "ERROR",
            ", ".join(map(str, videos)) if videos else "未找到 /dev/video*",
            "检查 USB、权限和 v4l2 驱动" if not videos else "")
        openni_candidates = (
            stroke_root / "including/OpenNI/sdk/libs/libOpenNI2.so",
            stroke_root / "including/OpenNI/sdk/Redist/libOpenNI2.so",
        )
        openni = next((path for path in openni_candidates if path.exists()), None)
        diagnostics.add(
            "OpenNI2", "OK" if openni else "WARN",
            str(openni) if openni else "项目目录未找到 libOpenNI2.so",
            "也可使用系统 OpenNI2 库")
    else:
        diagnostics.add("目标板相机", "DISABLED", "当前不是 Linux，未执行 /dev 与 OpenNI 硬件检查")

    emg = getattr(config, "emg", None)
    emg_enabled = bool(getattr(emg, "enabled", False))
    if not emg_enabled:
        diagnostics.add("EMG", "DISABLED", "配置未启用")
    elif platform.system() != "Linux":
        diagnostics.add("EMG 真实链路", "WARN", "需要在目标 Linux 板验证 RPMsg 与采集设备")
    else:
        rpmsg = Path(str(getattr(emg, "rpmsg_ctrl_device", "/dev/rpmsg_ctrl0")))
        backend = str(getattr(emg, "capture_backend", "serial")).lower()
        capture_ok = (backend == "bluez" or Path(str(
            getattr(emg, "serial_device", "/dev/rfcomm0"))).exists())
        diagnostics.add(
            "EMG 真实链路", "OK" if rpmsg.exists() and capture_ok else "ERROR",
            f"backend={backend}, rpmsg={rpmsg}",
            "启动 remoteproc/rpmsg_char 并连接 BLE 或 RFCOMM" if not (rpmsg.exists() and capture_ok) else "")

    voice = getattr(config, "voice", None)
    if voice is not None and not bool(getattr(voice, "enabled", True)):
        diagnostics.add("语音提示", "DISABLED", "配置未启用")
    else:
        ok, detail = _package_version("pyttsx3")
        diagnostics.add("语音提示", "OK" if ok else "WARN",
                        "pyttsx3 可用" if ok else "pyttsx3 未安装")

    record_path = Path(str(getattr(config, "record_path", "recordings") if config else "recordings"))
    if not record_path.is_absolute():
        record_path = python_root / record_path
    existing_parent = next((path for path in (record_path, *record_path.parents) if path.exists()), None)
    writable = bool(existing_parent and os.access(existing_parent, os.W_OK))
    diagnostics.add(
        "录制输出目录", "OK" if writable else "ERROR", str(record_path),
        "检查目录权限" if not writable else "")

    if platform.system() == "Linux":
        tools = [name for name in ("cmake", "v4l2-ctl") if shutil.which(name)]
        diagnostics.add("目标板工具", "OK" if tools else "WARN",
                        ", ".join(tools) if tools else "未找到 cmake/v4l2-ctl")
    return diagnostics


def print_diagnostics(diagnostics: Diagnostics) -> None:
    ascii_only = False
    try:
        "✓".encode(sys.stdout.encoding or "utf-8")
    except (UnicodeEncodeError, LookupError):
        ascii_only = True
    print("\n" + "=" * 58)
    print("  Stroke Rehab — Python 主框架启动诊断")
    print(f"  状态：{diagnostics.summary()}")
    print("=" * 58)
    for line in diagnostics.format_lines(ascii_only=ascii_only):
        print(line)
    print("=" * 58)
