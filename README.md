# Stroke Rehab Python

基于 PyQt5、QFluentWidgets 和 C++ 扩展的卒中康复训练桌面应用。系统提供实时骨骼预览、课程状态管理、动作评分、肌电状态、训练记录和 HTML 会话报告。

RGB/Depth ndarray callback、归一化时间戳与目标板验收方法见
[`docs/rgb_depth_array_transport.md`](docs/rgb_depth_array_transport.md)。

## 当前能力

- 上肢、下肢康复课程选择与持久化
- 训练前设备和保存目录检查
- 训练开始、停止和自动完成状态机
- 非阻塞的 `STARTING / STOPPING` 生命周期，界面不会等待相机或评分进程
- 22 关节骨骼 CSV 强制录制
- 实时骨骼帧送入 `ScoreBridge`，显示次数和五维动作质量
- 仅接受真实 RGB、Depth 与骨骼数据；EMG 默认关闭，启用后只连接真实设备
- 会话元数据、训练历史和始终可用的 HTML 摘要报告
- 相机、引擎、EMG 和保存环境诊断
- 真实模式强制 RGB/Depth 同步 30 FPS，并监测采集与显示帧率
- 普通训练视图和性能调试视图

> 会话摘要中的“有效关节比例”和“数据质量”描述采集完整度，不是临床诊断结果。

## 环境安装

推荐 Python 3.10 或更高版本：

```bash
python -m venv .venv
source .venv/bin/activate        # Linux
# .venv\Scripts\activate         # Windows
pip install -r requirements.txt
```

Windows Miniforge 示例：

```powershell
conda create -n stroke-rehab python=3.10
conda activate stroke-rehab
pip install -r requirements.txt
```

## 启动

```bash
python main.py
```

应用启动时会先运行环境诊断。没有加载真实采集所需的 C++ 扩展或硬件适配器时，采集会直接失败并提示原因，不会自动进入模拟模式。

## 配置

- 内置课程：`configs/courses.json`
- 用户设置：首次点击“应用设置”后写入 `config.user.json`
- 用户设置文件已被 Git 忽略，不会意外上传设备路径或患者名称
- 可用 `STROKE_USER_CONFIG` 指定其他用户配置路径
- 可用 `STROKE_REHAB_ROOT` 指定完整工程根目录

设置页提供课程、训练对象、相机、深度、EMG、调试显示和高级 RPMsg 参数，并支持“测试设备”。

真实引擎模式固定使用 RGB 30 FPS + Depth 30 FPS。系统会同时显示 RGB、Depth
和同步预览的实测帧率；低于目标时会告警，不会自动切换为模拟数据。

## 训练记录

默认保存目录：

```text
recordings/sessions/YYYYMMDD/YYYYMMDD_HHMMSS_mmm/
├── skeleton_3d.csv
├── session_ui_meta.json
└── session_report.html
```

`recordings/` 默认不进入 Git。

## 实时评分工具

`ScoreBridge` 会查找：

```text
tools/scoring_engine/score_server.py
```

完整工程中需保证 `tools/scoring_engine` 位于仓库父级工程，或通过 `STROKE_REHAB_ROOT` 指向包含该目录的工程根目录。评分器不可用时，训练和骨骼录制仍会继续，会话摘要报告仍可生成。

## C++ 扩展

Windows：

```powershell
build_win.bat
```

Linux / 开发板：

```bash
chmod +x build_linux.sh setup_board.sh
./setup_board.sh
./build_linux.sh
```

## 测试

```bash
python test_stage2.py
python test_ui.py
python test_workflows.py
```

无显示器的 Linux 环境可使用：

```bash
QT_QPA_PLATFORM=offscreen python test_ui.py
```

## 重要提示

- 正式训练前应确认患者站位、遮挡、相机距离和关节有效率。
- 停止或完成训练会统一关闭课程计时器、评分进程、录制器和传感器流水线。
- 本项目用于康复训练辅助和技术研究，不替代治疗师判断或医疗诊断。

## Python ONNX 模型部署

检测和姿态模型现在可由 Python ONNX Runtime 直接运行，原生 `_core` 作为相机、深度和可选 native 推理适配器。模型路径、Execution Provider、线程数及 `python/native/auto` 后端选择见 [`docs/model_deployment.md`](docs/model_deployment.md)。

安装依赖后可执行真实模型自检：

```bash
python verify_models.py
python test_inference.py
```

## Python 主框架迁移

当前 Python 层已经拥有配置、采集生命周期、RGB/Depth 时间戳同步、处理队列、深度软件配准、ONNX 推理以及 EMG 协议和 RPMsg 调度。`_core` 保留为 V4L2/OpenNI 等底层驱动及可选 native 推理适配器，不再作为主程序控制层。

- 采集、同步与深度对齐：[`docs/capture_pipeline.md`](docs/capture_pipeline.md)
- BLE/串口 EMG 与 CPU1 RPMsg：[`docs/emg_pipeline.md`](docs/emg_pipeline.md)
- Python ONNX 模型部署：[`docs/model_deployment.md`](docs/model_deployment.md)
- 非阻塞训练语音：[`docs/voice_assistant.md`](docs/voice_assistant.md)

核心逻辑回归：

```bash
python test_capture.py
python test_alignment.py
python test_emg.py
python test_inference.py
python test_pose3d.py
python test_scoring.py
python test_voice.py
python test_stage2.py
python test_workflows.py
QT_QPA_PLATFORM=offscreen python test_ui.py
```

真实相机、OpenNI 硬件 D2C、蓝牙射频、RFCOMM、remoteproc/RPMsg 与 CPU1 特征结果必须在目标 Linux 板上另行验收；桌面测试通过不能替代上板验证。
