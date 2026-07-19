# 流水线性能监控与异步录制

性能窗口默认使用最近 5 秒数据，至少收集 30 个 callback 样本后才结束预热。单帧
callback 超过 2 ms 只计入性能统计，不产生 WARN。RGB 与 Depth 分别计算平均值、
P50、P95 和最大值。

- P95 ≤ 8 ms：正常。
- P95 > 10 ms 持续 5 秒：警告。
- P95 > 25 ms：严重。
- 任一路原始采集帧率持续 3 秒低于目标帧率的 90%：严重。
- 异常消失并连续正常 5 秒：恢复。

性能状态只在状态迁移时写入 `PERF` 运行日志和性能监控，不弹出 InfoBar。设备断开、
录制文件打开失败和应用错误仍会通知用户，同一事件码默认冷却 30 秒。

性能监控中的五类帧率含义如下：

- 原始 RGB 帧率：有效 V4L2 callback 到达速率。
- 原始深度帧率：有效 OpenNI callback 到达速率。
- 同步配对帧率：时间戳匹配成功的 RGB-D pair 速率。
- Worker 处理帧率：Pose worker 完成 pair 处理的速率。
- 姿态推理帧率：实际执行模型推理的速率；复用姿态的帧不计入。

旧字段 `pair_fps` 在迁移期等同于 `worker_fps`，新代码应使用明确字段。

## 录制线程

同步 pair callback 将同一组只读 BGR `uint8` 和 Depth `uint16` 数组引用非阻塞地提交给
Worker latest-only 队列和独立录制队列。录制线程负责 RGB resize/VideoWriter、Depth
对齐/resize/伪彩和写盘；Pose worker 的丢帧不会直接造成录制丢帧。队列满时淘汰最旧
pair，并分别统计接收、写入、溢出、编码失败、停止丢弃、队列水位、写入 FPS 与耗时
平均/P95。`video_frames.csv` 保存每个写入 pair 的同步时间戳及 RGB/Depth frame ID。

640×480 的 BGR+Depth 数组约占 1.46 MiB/pair，不含 Python/OpenCV 对象开销。队列容量
30、60、90 的仅数组内存预算约为 44、88、132 MiB。默认容量为 90，以满足 30 FPS 下
约 3 秒的短时编码抖动缓冲；目标板验收时必须结合 RSS 和双路编码 P95 再确认。

配置位于 `configs/device.yaml` 的 `performance` 和 `recording` 段。需要回滚提交点进行
A/B 比较时，可临时设置 `async_video_recording: false`；编码仍由录制线程完成，但 pair
会在 Worker 完成对齐后提交。正常部署使用 `true`。

Depth 视频采用 AVI 容器内的 `mp4v` 编码；目标板实测相较 MJPG 可使双路录制从约
24 FPS 提升到约 29.7 FPS。完整板端数据见 `BOARD_PERFORMANCE_VALIDATION.md`。
