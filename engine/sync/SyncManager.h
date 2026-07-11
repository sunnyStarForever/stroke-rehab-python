/*
 * 模块作用：
 * 本文件声明 RGB/Depth 软件时间同步器。两路设备采集线程独立运行，
 * 因此必须用统一的 host_ts_ns 进行近邻匹配，形成可用于对齐和 3D 的 RGB-D pair。
 */
#pragma once

#include <deque>
#include <functional>
#include <mutex>
#include <optional>

#include "engine/common/Config.h"
#include "engine/common/FrameEnvelope.h"

namespace rehab {

/*
 * SyncManager
 * 职责：
 * 1. 分别缓存 RGB 和 Depth 的少量最新帧；
 * 2. 每来一帧就用 host_ts_ns 在另一队列中找最近邻；
 * 3. 时间差小于阈值时输出 SyncedFramePair；
 * 4. 队列超过上限时丢弃过旧帧，防止实时系统延迟累积。
 */
class SyncManager {
 public:
  using PairCallback = std::function<void(const SyncedFramePair&)>;

  explicit SyncManager(SyncConfig config);

  void setOnPairReady(PairCallback callback);
  void pushFrame(FrameEnvelope frame);
  void clear();

 private:
  std::optional<SyncedFramePair> tryMatchLocked(FrameSource incomingSource);
  static int64_t absDiffNs(uint64_t lhs, uint64_t rhs);
  void trimQueueLocked(std::deque<FrameEnvelope>& queue);

  SyncConfig config_;       // 同步阈值和队列长度配置
  PairCallback callback_;   // pair 输出回调，通常进入 SensorPipeline 的处理队列

  std::mutex mutex_;                 // 保护两路队列和回调复制
  std::deque<FrameEnvelope> rgbQueue_;   // 少量 RGB 最新帧，按 host_ts_ns 近似递增
  std::deque<FrameEnvelope> depthQueue_; // 少量 Depth 最新帧，按 host_ts_ns 近似递增
};

}  // namespace rehab
