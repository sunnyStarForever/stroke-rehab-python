# Python 主框架部署

当前运行边界如下：

- Python：配置、帧同步、空间对齐、ONNX 推理、2D/3D 姿态、肌电协议、训练流程、评分、报告、语音和 UI。
- `_core`：仅提供 Linux V4L2 RGB 与 OpenNI2 深度相机驱动。
- 旧 C++ 同步、姿态、3D 和肌电绑定默认不参与构建；仅兼容调试时可显式设置 `STROKE_BUILD_LEGACY_NATIVE_PIPELINE=ON`。

## Windows 开发机

在 `python_version` 中运行：

```bat
build_win.bat
.venv\Scripts\python.exe main.py
```

Windows 开发机不再默认使用模拟相机数据；未连接真实 RGB-D 设备或未加载真实采集核心时，采集不会启动。仍可单独执行非硬件 UI/模型检查：

```bat
.venv\Scripts\python.exe verify_runtime.py --models --ui
```

## Linux 目标板

从工作区运行（Git Bash、WSL 或 Linux）：

```bash
bash python_version/deploy.sh <board-ip> root
```

然后登录目标板：

```bash
cd /root/stroke-rehab-runtime/python_version
bash env_check.sh
bash setup_board.sh
.venv/bin/python main.py
```

`setup_board.sh` 会安装系统依赖、创建虚拟环境、安装 Python 依赖、编译硬件适配器，并执行模型、UI 与硬件自检。

## 手工构建硬件适配器

```bash
cd python_version
python3 -m venv --system-site-packages .venv
.venv/bin/pip install -r requirements.txt
PYTHON_BIN="$PWD/.venv/bin/python" bash build_linux.sh
```

构建产物会复制到 `rehab_engine/_core*.so`。目标板启动前可运行：

```bash
.venv/bin/python verify_runtime.py --models --ui --require-hardware
```

未连接 V4L2/OpenNI、BLE/RFCOMM 或 RPMsg 设备时，硬件链路无法在开发机上完成验证；这类检查必须在目标板连接真实设备后执行。
