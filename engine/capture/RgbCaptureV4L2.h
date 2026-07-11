/*
 * 模块作用：
 * 本文件声明 RGB 采集器。项目在 Linux 摄像头侧直接使用 V4L2，
 * 便于控制 MJPG/YUYV、分辨率、帧率和缓冲区行为。
 * 采集到的 RGB 帧会被封装为 FrameEnvelope，并交给 SyncManager 与 Depth 帧配对。
 */
#pragma once

#include <atomic>
#include <functional>
#include <mutex>
#include <thread>

#include "engine/common/Config.h"
#include "engine/common/FrameEnvelope.h"
#include "engine/sync/TimestampNormalizer.h"

namespace rehab {

/*
 * RgbCaptureV4L2
 * 职责：
 * 1. 打开并配置 /dev/video* RGB 摄像头；
 * 2. 在独立线程中持续读取 V4L2 缓冲区；
 * 3. 为每帧打统一主机时间戳，避免不同设备时钟无法直接比较；
 * 4. 通过回调把 FrameEnvelope 推给上层 pipeline。
 */
class RgbCaptureV4L2 {
 public:
  using FrameCallback = std::function<void(FrameEnvelope)>;
  using StatusCallback = std::function<void(const std::string&)>;

  ~RgbCaptureV4L2();

  bool start(const DeviceConfig& config, FrameCallback callback);
  void stop();
  void setOnStatus(StatusCallback callback);

  bool isRunning() const { return running_.load(); }

 private:
  void run();
  void runFallback();
  void emitStatus(const std::string& status);

  DeviceConfig config_;              // 采集参数：设备路径、MJPG/YUYV、分辨率、fps、镜像等
  FrameCallback callback_;           // 帧回调：采集线程把 RGB 帧交给同步层
  StatusCallback statusCallback_;    // 状态回调：只传递日志/性能信息，不参与计算链路
  std::mutex statusMutex_;           // 保护状态回调，避免 UI 线程和采集线程同时访问

  std::atomic<bool> running_{false}; // 采集线程退出标志，stop() 通过它通知 run() 收尾
  TimestampNormalizer tsNormalizer_; // 统一写入 host_ts_ns/device_ts_us
  std::thread worker_;               // RGB 采集工作线程，避免阻塞 UI 和姿态推理
};

}  // namespace rehab
