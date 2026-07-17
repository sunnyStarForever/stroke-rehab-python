# Python 主框架迁移矩阵

本文档以 `stroke-rehab` 的原有 C++ 实现为行为基准，记录当前 Python
主框架的代码归属、兼容状态和仍需在目标 Linux 板卡完成的验收。

状态说明：

- **Python 主控**：业务状态、队列、时间戳或算法由 Python 实现。
- **硬件适配**：仅底层设备访问保留在 `_core`。
- **兼容保留**：旧 C++ 实现只在显式兼容构建中提供，不是默认运行路径。
- **待上板**：桌面测试已覆盖逻辑，但真实设备链路尚未验收。

## 运行边界

| 能力 | 原实现 | 当前 Python 归属 | 状态 |
|---|---|---|---|
| 配置加载与环境变量覆盖 | `Config.h`, `MainWindow.cpp` | `rehab_engine/config_loader.py`, `_stub.py` | Python 主控 |
| RGB 设备访问 | `RgbCaptureV4L2` | `capture.NativeRgbDepthBackend` 调用 `_core.RgbCaptureV4L2` | 硬件适配、待上板 |
| Depth/OpenNI 与硬件 D2C | `DepthCaptureOpenNI`, `HardwareD2CAligner` | Python 编排，`_core.DepthCaptureOpenNI` 访问设备 | 硬件适配、待上板 |
| 时间戳标准化与最近邻同步 | `TimestampNormalizer`, `SyncManager` | `capture.TimestampNormalizer`, `FrameSynchronizer` | Python 主控 |
| 有界处理队列与丢帧 | `LatestFrameQueue`, `SensorPipeline` | `SensorPipeline._enqueue_pair` | Python 主控 |
| 软件深度配准 | `SoftwareRegistrationAligner` | `alignment.SoftwareRegistrationAligner` | Python 主控 |
| 深度单位与像素格式 | `FrameEnvelope` | `capture.FrameEnvelope`，贯通对齐、采样和 3D 投影 | Python 主控、待上板 |
| YOLO 人体检测 | `PersonDetectorOrt` | `inference.PersonDetector` + ONNX Runtime | Python 主控 |
| RTMPose 推理 | `PoseEstimatorRTMPoseOrt` | `inference.RtmposeEstimator` + ONNX Runtime | Python 主控 |
| 自适应 ROI | `AdaptiveRoiBoundingBoxProvider` | `inference.AdaptiveRoiTracker` | Python 主控 |
| Halpe26 到 Rehab22 | `Halpe26ToRehab22Mapper` | `inference.map_halpe26_to_rehab22` | Python 主控 |
| 深度采样与 3D 投影 | `DepthSampler`, `JointProjector3D` | `pose3d.DepthSampler`, `JointProjector3D` | Python 主控 |
| EMA/旧平滑器 | `EMASkeletonFilter`, `SkeletonSmoother` | `pose3d.EmaSkeletonFilter`, `SkeletonSmoother` | Python 主控 |
| BLE 通知协议解析 | `EmgBleNotifyParser` | `emg.EmgBleNotifyParser` | Python 主控 |
| BLE GATT 与扫描 | `EmgBleGattCapture`, `EmgBluetoothScanner` | `emg.EmgBleGattCapture`, `EmgBluetoothScanner`（Bleak） | Python 主控、待上板 |
| RFCOMM/串口肌电 | `EmgBleSerialCapture` | `emg.EmgSerialCapture` | Python 主控、待上板 |
| RPMsg/remoteproc | `EmgRpmsgClient` | `emg.EmgRpmsgClient` | Python 主控、待上板 |
| 肌电特征与时间融合 | `EmgFeatureProcessor`, `EmgFusionBuffer` | `emg.EmgFeatureProcessor`, `EmgFusionBuffer` | Python 主控 |
| 课程与状态机 | `CourseRepository`, `CourseRunner` | `course.CourseRepository`, `CourseRunner` | Python 主控 |
| 实时评分与离线报告 | `ScoreBridge`, `OfflineReportRunner` | `scoring.py`, `reporting.py` | Python 主控 |
| 骨架/RGB/Depth 录制 | `Skeleton3DRecorder`, `SensorPipeline` | `recorder.py`, `SensorPipeline` | Python 主控 |
| 预览与性能数据 | `PreviewComposer` | `preview.py`, `ui/widgets/preview_widget.py` | Python 主控 |
| 训练、报告、设置界面 | Qt/C++ `TrainingPage`, `ReportPage`, 设置对话框 | PyQt `ui/pages/` 与 `ui/widgets/` | Python 主控 |
| 患者首页/患者资料卡 | `PatientHomePage`, `PatientInfoCard`, `CourseCardWidget` | `ui/pages/patient_home_page.py`，接入真实课程与本地历史 | Python 主控 |
| 独立日志/性能对话框 | `LogDialog`, `PerformanceDialog` | `ui/dialogs/runtime_dialogs.py`，从设置页打开并持续刷新 | Python 主控 |
| 语音提示 | 原界面流程 | `voice.VoiceAssistant` | Python 主控 |

默认构建使用 `STROKE_BUILD_LEGACY_NATIVE_PIPELINE=OFF`。因此 `_core` 不包含
旧同步、姿态、3D、肌电和课程实现。只有为差异比对显式打开该选项时，
`inference_backend=native` 才可用；产品运行应使用 `inference_backend=python`。

## 行为兼容要点

- RGB 驱动帧作为主时间线，RGB/Depth 以主机单调时间戳做最近邻配对。
- 同步和姿态队列达到上限时丢弃最旧帧，保留最新输入。
- 跳过姿态推理的帧复用最近 2D 关节，但重新采样当前深度，避免 3D 冻结。
- 深度单位来自设备帧；毫米和 `DEPTH_100_UM` 均按真实比例进入配准与投影。
- 肌电特征按最近 RGB 时间戳匹配，最大允许时间差为 300 ms。
- `skeleton_3d.csv` 保持旧评分工具需要的 67 列格式；详细时间戳和调试数据
  分别写入 `skeleton_3d_detailed.csv` 与 `skeleton3d_debug.csv`。
- 默认录制骨架与 `rgb.mp4`，深度视频按配置写入 `depth.avi`。

`pose.depth_median_window` 属于旧版、当前已停用的简单采样分支；当前高级采样器
由 `depth_sampler` 配置组控制。原有 `save_depth_sampling_overlay`、
`save_skeleton_raw_csv` 和 `save_skeleton_ema_csv` 在旧代码中也只有配置声明，
没有形成独立运行分支；Python 版保留字段以兼容配置，并统一输出可审计调试 CSV。

## 验证入口

开发机逻辑回归：

```bash
python -m unittest discover -v
python verify_runtime.py --models --ui
```

目标 Linux 板卡验收：

```bash
bash env_check.sh
python verify_runtime.py --models --ui --require-hardware
python main.py
```

`--require-hardware` 会实际打开 V4L2 与 OpenNI 驱动、等待 Python 最近邻同步
产生帧对，并在默认 3 秒采集后执行有序停止；可用
`--capture-smoke-seconds N` 调整持续时间。该检查不是单纯检查设备节点。

肌电配置为 `enabled: true`、`mode: real` 后，可追加 `--require-emg`。
该检查会真实启动 BLE/RFCOMM 采集、RPMsg endpoint 和 CPU1 特征回传，只有
同时建立采集与 RPMsg 链路并收到至少一帧特征才通过：

```bash
python verify_runtime.py --models --ui --require-hardware --require-emg
```

上板时需要逐项确认：V4L2 帧率与设备时间戳、OpenNI 深度格式和 D2C、
BLE 扫描/连接/通知、RFCOMM 备用链路、remoteproc/RPMsg 启停、真实模型延迟、
长时间录制完整性及停止过程资源释放。
