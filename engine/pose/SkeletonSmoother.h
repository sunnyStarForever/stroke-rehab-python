/*
 * 模块作用：
 * 本文件声明 3D 骨架平滑器。深度噪声会导致关节 Z 值跳动，
 * EMA 平滑和跳变拒绝能让 UI 显示与录制数据更稳定。
 */
#pragma once

#include <array>
#include <cstdint>

#include "engine/pose/Rehab22Types.h"

namespace rehab {

struct SkeletonSmoothingStats {
  int validInput{0};    // 输入中有效 3D 关节数量
  int validOutput{0};   // 平滑后仍有效的 3D 关节数量
  int jumpRejected{0};  // 被判定为短时间深度突变而拒绝的关节数量
};

/*
 * SkeletonSmoother
 * 职责：
 * 1. 对每个关节维护上一帧状态；
 * 2. 使用 EMA 将新观测和历史状态融合，减少抖动；
 * 3. 对短时间内过大的 Z 跳变做拒绝，避免深度空洞污染骨架。
 */
class SkeletonSmoother {
 public:
  void reset();
  void setAlpha(float alpha);
  Rehab22Skeleton3D smooth(const Rehab22Skeleton3D& input);
  Rehab22Skeleton3D smooth(const Rehab22Skeleton3D& input,
                           uint64_t timestampNs,
                           SkeletonSmoothingStats* stats);

 private:
  float alpha_{0.35f};
  std::array<bool, kRehab22JointCount> hasState_{};
  std::array<uint64_t, kRehab22JointCount> lastTimestampNs_{};
  std::array<int, kRehab22JointCount> invalidStreak_{};
  Rehab22Skeleton3D state_{};
};

}  // namespace rehab
