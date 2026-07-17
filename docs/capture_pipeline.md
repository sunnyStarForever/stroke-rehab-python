# Python 采集、同步与深度对齐

Python 应用层现在负责 RGB/Depth 采集生命周期、时间戳归一化、帧同步、处理队列和深度到 RGB 的软件配准。原生 `_core` 仅提供 V4L2 RGB 与 OpenNI Depth 的设备读帧能力，不再拥有同步策略或流水线调度。

## 数据路径

```text
V4L2 RGB driver  ──> FrameEnvelope ──┐
                                     ├─> Python nearest timestamp sync
OpenNI Depth driver ─> FrameEnvelope ┘        │
                                               v
                                      latest-only processing queue
                                               │
                              HW D2C active ───┴─── software registration
                                               │
                                               v
                         detector -> pose -> Rehab22 -> Python 3D lifting
```

实现入口：

- `rehab_engine/capture.py`：`FrameSynchronizer`、`LatestFrameQueue` 与 `NativeRgbDepthBackend`。
- `rehab_engine/alignment.py`：标定加载与 `SoftwareRegistrationAligner`。
- `rehab_engine/pose3d.py`：前景深度采样、RGB 内参反投影、EMA 与 legacy 平滑。
- `rehab_engine/sensor_pipeline.py`：设备生命周期、同步帧消费及推理调度。

## 同步逻辑

同步策略保持原 C++ `SyncManager` 的关键约束：

- RGB 与 Depth 分别以主机单调时钟记录 `host_ts_ns`，同时保留设备时间戳。
- 每次新帧到达时，在另一侧有界队列中选择主机时间戳差绝对值最小的一帧。
- 只有时间差不超过 `SyncConfig.match_threshold_ns` 才生成帧对；每帧最多匹配一次。
- 队列超过 `SyncConfig.queue_size` 时丢弃最旧帧并累计诊断计数。
- 帧对回调在同步锁外执行，避免采集线程被下游处理反向阻塞。
- 下游采用 latest-only 队列；处理追不上采集时替换旧帧，优先保证实时性。

默认参数是 20 ms 匹配阈值和每路 30 帧缓存，可通过 `PipelineConfig.sync` 调整。

## 深度对齐

当 OpenNI 报告硬件 D2C 已启用时，Python 直接使用设备给出的已对齐深度。否则读取 `configs/calibration.yaml`，将每个有效深度像素反投影到深度相机坐标，再通过外参投影到 RGB 图像，并用最近深度 z-buffer 处理多个点落入同一像素的情况。

标定加载器兼容原文件中的大写 `R/T` 和小写 `r/t`。`translation_unit: "mm"` 会显式把平移向量转换成米；缺少单位且数值明显为毫米时也会兼容推断。若标定不可用，则仅做最近邻尺寸适配，并在运行状态中标记没有有效软件配准，不能把该状态当作已标定对齐。

## 验证

桌面逻辑回归：

```bash
python test_capture.py
python test_alignment.py
python test_pose3d.py
```

## Python 3D 骨骼

姿态模型输出 Rehab22 之后，Python 会沿用原版的身体深度参考和背景深度参考，按关节类别选择窗口半径；手腕、脚踝和脚趾使用近前景百分位，膝、踝、趾和腕在采样失败时沿父关节方向向内搜索。有效深度随后按 RGB 内参执行 `X=(u-cx)*Z/fx`、`Y=(v-cy)*Z/fy`。

`skeleton_filter.mode` 支持 `none`、`legacy_stabilizer` 和默认 `ema`。EMA 保留低置信度、恢复点、Z 跳变、速度跳变和短期无效保持的原参数语义。未到姿态推理间隔的帧会复用上一帧 2D 关节，但使用当前对齐深度重新生成 3D，不会冻结空间位置。

没有有效 RGB 标定内参时仍可显示 2D，但会明确禁用 3D 反投影，不再使用无法证明准确的固定内参伪装为已标定结果。

目标板验收还必须覆盖：

1. V4L2 与 OpenNI 分别启动、部分启动失败时正确回滚。
2. 连续运行时同步匹配率、阈值 miss、队列裁剪和 latest-only 丢帧计数。
3. 硬件 D2C 开关两种路径下，RGB/Depth 轮廓是否实际重合。
4. 30 FPS 压力下停止流程不死锁、设备句柄可再次打开。

桌面单元测试通过不代表上述硬件验收已经完成。
