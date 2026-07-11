/*
 * 模块作用：
 * 本文件实现 RGB/Depth 的软件时间同步。硬件即使支持深度和彩色同步，
 * 两路在程序中仍来自不同采集 API，因此主流程统一使用 host_ts_ns 做配对。
 */
#include "engine/sync/SyncManager.h"

#include <algorithm>
#include <limits>

namespace rehab {

SyncManager::SyncManager(SyncConfig config) : config_(config) {}

void SyncManager::setOnPairReady(PairCallback callback) {
  std::lock_guard<std::mutex> lock(mutex_);
  callback_ = std::move(callback);
}

void SyncManager::pushFrame(FrameEnvelope frame) {
  /*
   * pushFrame()
   * 输入：采集层送来的一帧 RGB 或 Depth。
   * 作用：放入对应队列，并立即尝试用另一队列中的最近 host_ts_ns 生成 pair。
   * 注意：回调在解锁后执行，避免上层处理耗时阻塞采集线程继续入队。
   */
  PairCallback callbackCopy;
  std::optional<SyncedFramePair> matchedPair;

  {
    std::lock_guard<std::mutex> lock(mutex_);
    if (frame.source == FrameSource::Rgb) {
      rgbQueue_.push_back(std::move(frame));
      trimQueueLocked(rgbQueue_);
      matchedPair = tryMatchLocked(FrameSource::Rgb);
    } else {
      depthQueue_.push_back(std::move(frame));
      trimQueueLocked(depthQueue_);
      matchedPair = tryMatchLocked(FrameSource::Depth);
    }
    callbackCopy = callback_;
  }

  if (matchedPair && callbackCopy) {
    callbackCopy(*matchedPair);
  }
}

void SyncManager::clear() {
  std::lock_guard<std::mutex> lock(mutex_);
  rgbQueue_.clear();
  depthQueue_.clear();
}

std::optional<SyncedFramePair> SyncManager::tryMatchLocked(
    FrameSource incomingSource) {
  std::deque<FrameEnvelope>& anchorQueue =
      (incomingSource == FrameSource::Rgb) ? rgbQueue_ : depthQueue_;
  std::deque<FrameEnvelope>& otherQueue =
      (incomingSource == FrameSource::Rgb) ? depthQueue_ : rgbQueue_;

  if (anchorQueue.empty() || otherQueue.empty()) {
    return std::nullopt;
  }

  // 以最新到达帧作为 anchor，保证输出尽量贴近当前时刻，而不是补处理历史帧。
  const FrameEnvelope anchor = anchorQueue.back();

  std::size_t bestIndex = 0;
  int64_t bestDelta = std::numeric_limits<int64_t>::max();
  for (std::size_t i = 0; i < otherQueue.size(); ++i) {
    // 最近邻匹配：在另一队列中找 host_ts_ns 时间差最小的帧。
    const int64_t delta = absDiffNs(anchor.hostTsNs, otherQueue[i].hostTsNs);
    if (delta < bestDelta) {
      bestDelta = delta;
      bestIndex = i;
    }
  }

  if (bestDelta > config_.matchThresholdNs) {
    // 时间差超过阈值说明两帧不是同一瞬间的人体状态，等待后续帧而不是强配。
    return std::nullopt;
  }

  // 已匹配的帧必须从队列移除，避免同一帧被重复配对导致录制和统计重复。
  FrameEnvelope matchedOther = std::move(otherQueue[bestIndex]);
  otherQueue.erase(otherQueue.begin() + static_cast<std::ptrdiff_t>(bestIndex));
  anchorQueue.pop_back();

  SyncedFramePair pair;
  if (incomingSource == FrameSource::Rgb) {
    pair.rgb = std::move(anchor);
    pair.depth = std::move(matchedOther);
  } else {
    pair.rgb = std::move(matchedOther);
    pair.depth = std::move(anchor);
  }
  pair.deltaNs = static_cast<int64_t>(pair.rgb.hostTsNs) -
                 static_cast<int64_t>(pair.depth.hostTsNs);
  return pair;
}

int64_t SyncManager::absDiffNs(uint64_t lhs, uint64_t rhs) {
  return static_cast<int64_t>((lhs > rhs) ? (lhs - rhs) : (rhs - lhs));
}

void SyncManager::trimQueueLocked(std::deque<FrameEnvelope>& queue) {
  while (queue.size() > config_.queueSize) {
    // 队列只保留近期帧；过旧帧即使未来被处理也只会增加延迟，不能代表当前动作。
    queue.pop_front();
  }
}

}  // namespace rehab
