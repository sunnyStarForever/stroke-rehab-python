# RGB/Depth 数组传输与时钟同步

## 实时数据接口

原生采集 callback 不再传递 JPEG/PNG bytes，而是传递 Python-owned、C-contiguous NumPy 数组：

- RGB：`numpy.uint8`，shape=`(height, width, 3)`，通道顺序 BGR。
- Depth：`numpy.uint16`，shape=`(height, width)`，数值保持设备原始单位。

callback 参数顺序为：

```text
image, width, height, sync_ts_ns, frame_id,
device_ts_us, depth_unit_to_meter, pixel_format, source,
arrival_ts_ns, device_time_unit, clock_quality,
clock_reason, clock_reset_count
```

数组内存由 Python 持有，callback 返回、V4L2 缓冲区重新 QBUF 或 OpenNI frame 释放后仍可安全读取。实时采集、同步和 worker 路径不得执行 JPEG/PNG 编解码；压缩只允许发生在录像或调试导出边界。

## 时间戳含义

- `device_ts_us`：设备/驱动原始时间；缺失时为 0。
- `arrival_ts_ns`：帧从驱动返回附近采集的 `CLOCK_MONOTONIC_RAW` 时间。
- `sync_ts_ns`：映射到统一 monotonic-raw 时钟域、供 RGB-D 最近邻配对使用的时间。
- `clock_quality`：`native_monotonic`、`normalized_device` 或 `host_fallback`。
- `clock_reason`：发生降级或设备时钟重置时的原因。
- `clock_reset_count`：当前采集实例检测到的时钟重置累计次数。

V4L2 monotonic buffer timestamp 会映射到 monotonic-raw 域；无效或不单调时退回 arrival 时间。OpenNI 微秒时间使用滚动低分位偏移估计映射；时间倒退或异常跳变会重建映射。

## 验证

自动化回归：

```bash
python -m unittest discover -p 'test_*.py'
```

真实硬件同步验证：

```bash
python tools/validate_rgb_depth_sync.py \
  --duration 300 \
  --output recordings/rgb_depth_sync_validation.json
```

输出包括原始 RGB/Depth FPS、pair FPS、同步差值平均值/P95/最大值、时钟质量分布、callback P95、队列裁剪、CPU 和 RSS。目标板验收时还应检查原生日志中的 V4L2 timestamp flags 与 OpenNI 时间连续性。

## 回滚

这是破坏性 callback API 迁移，不能在运行时混用 bytes 与 ndarray 协议。若目标板完整构建或实机验证失败，应整体回滚以下部分到同一个变更前版本：

1. `engine/common/FrameEnvelope.h` 与 `engine/sync/TimestampNormalizer.*`
2. `engine/capture/RgbCaptureV4L2.*`、`DepthCaptureOpenNI.cpp`
3. `bindings/module.cpp`
4. `rehab_engine/capture.py`、`sensor_pipeline.py` 与相关 recorder/工具

回滚后必须重新构建 `_core`，不得把旧 `_core` 与新 Python 代码组合运行。

## 目标板时间戳参数依据（2026-07-19）

目标板 `/dev/video0` 在 MJPG 与 YUYV 两种模式下均报告
`flags=0x12001`、`type=monotonic`，因此 RGB 使用 V4L2 设备时间映射，
不使用 DQBUF 完成时刻代替采集时刻。OpenNI 连续帧设备时间增量约为
33.7 ms，和 30 FPS 配置一致；首帧设备时间为 0 时仅该帧降级到 arrival，
后续帧进入 `normalized_device`。

归一化器保留 31 帧滚动窗口，选取排序后约 10% 低分位的
`arrival - device` 偏移，以降低排队/复制延迟对偏移估计的污染；每帧偏移修正
限制为 200 us，避免主机调度抖动直接造成同步时间跳动。设备时间倒退，或设备
增量与 arrival 增量相差超过 1 s 时重建映射。这些阈值远大于实测约 33.7 ms
帧周期与毫秒级调度抖动，同时能快速识别流重启和异常跳变。

格式契约可分别用以下命令复验：

```bash
python tools/validate_rgb_depth_sync.py --duration 10 \
  --rgb-format MJPG --depth-format DEPTH_1_MM \
  --output recordings/array_mode_mjpg_1mm.json
python tools/validate_rgb_depth_sync.py --duration 10 \
  --rgb-format YUYV --depth-format DEPTH_100_UM \
  --output recordings/array_mode_yuyv_100um.json
```

## 目标板验收结果（2026-07-19）

- 180 秒无录制：RGB 29.86 FPS、Depth 29.64 FPS、pair 29.49 FPS；
  5309 个唯一 RGB/Depth pair，绝对同步差均值 7.92 ms、P95 17.19 ms、
  最大 20.00 ms；RGB 5309 帧全部为 `native_monotonic`，Depth 5309 帧
  全部为 `normalized_device`，时钟重置 0 次，RGB/Depth 队列裁剪分别为
  37/0。证据文件：`recordings/rgb_depth_sync_validation_180s.json`。
- 20 秒启用 RGB/Depth 录像：采集两路均为 29.91 FPS，分别写入 247 帧，
  停止回调成功；模型 worker 采用 latest-only 策略，推理较慢时丢弃旧 pair，
  不反压采集线程。证据文件：
  `recordings/array_recording_validation/real_pipeline_summary.json`。
- MJPG/YUYV 均得到 BGR `uint8[480,640,3]`、stride `[1920,3,1]`；
  `DEPTH_1_MM`/`DEPTH_100_UM` 均得到 `uint16[480,640]`、stride
  `[1280,2]`，单位元数据分别约为 `0.001`/`0.0001`，零值与非零最大值
  保持原始整数。末帧数组契约与首帧一致。
- 连续 3 轮真实 RGB-D 启动/停止全部成功，无设备占用残留或停止超时。

本次实施前没有保存同一目标板、同一配置下的 CPU/RSS/callback P95/pair FPS
基线，因此不能给出可信的前后百分比对比；变更后绝对值和生命周期验证已经记录，
OpenSpec 任务 8.5 保持未完成，直至补采可比的变更前版本数据。
