/*
 * 模块作用：
 * 本文件实现 Rehab22 3D 骨架平滑。平滑只改变显示和录制使用的输出，
 * 不影响 2D 推理、深度采样或原始 alignedDepth。
 */
#include "engine/pose/SkeletonSmoother.h"

#include <algorithm>
#include <cmath>

namespace rehab {

namespace {

bool isDepthJump(float newZ, float lastZ, double dtSec) {
  // 短时间内 Z 方向突变通常来自深度空洞或背景误采样，不适合作为人体真实运动。
  if (lastZ <= 0.0f || newZ <= 0.0f) {
    return false;
  }
  const float dz = std::abs(newZ - lastZ);
  return dtSec > 0.0 && dtSec < 0.2 && dz > 0.35f;
}

}  // namespace

void SkeletonSmoother::reset() {
  hasState_.fill(false);
  lastTimestampNs_.fill(0);
  invalidStreak_.fill(0);
  state_ = {};
}

void SkeletonSmoother::setAlpha(float alpha) {
  // alpha 越大越跟随当前帧，越小越稳定但延迟更明显。
  alpha_ = std::clamp(alpha, 0.0f, 1.0f);
}

Rehab22Skeleton3D SkeletonSmoother::smooth(const Rehab22Skeleton3D& input) {
  return smooth(input, 0, nullptr);
}

Rehab22Skeleton3D SkeletonSmoother::smooth(
    const Rehab22Skeleton3D& input,
    uint64_t timestampNs,
    SkeletonSmoothingStats* stats) {
  SkeletonSmoothingStats localStats;
  Rehab22Skeleton3D output{};

  for (std::size_t i = 0; i < kRehab22JointCount; ++i) {
    if (!input[i].valid) {
      // 连续无效一段时间后清空历史状态，避免人物离开后还显示旧骨架。
      ++invalidStreak_[i];
      if (invalidStreak_[i] > 10) {
        hasState_[i] = false;
        state_[i] = {};
        lastTimestampNs_[i] = 0;
      }
      continue;
    }

    ++localStats.validInput;
    invalidStreak_[i] = 0;
    const double dtSec =
        (timestampNs > 0 && lastTimestampNs_[i] > 0 &&
         timestampNs > lastTimestampNs_[i])
            ? static_cast<double>(timestampNs - lastTimestampNs_[i]) / 1.0e9
            : 0.0;

    if (hasState_[i] && isDepthJump(input[i].z, state_[i].z, dtSec)) {
      ++localStats.jumpRejected;
      output[i] = {};
      ++invalidStreak_[i];
      if (invalidStreak_[i] > 10) {
        hasState_[i] = false;
        state_[i] = {};
        lastTimestampNs_[i] = 0;
      }
      continue;
    }

    if (!hasState_[i]) {
      // 第一次看到有效关节时直接初始化状态，不做平滑，避免从零点拖尾。
      state_[i] = input[i];
      hasState_[i] = true;
    } else {
      state_[i].x = alpha_ * input[i].x + (1.0f - alpha_) * state_[i].x;
      state_[i].y = alpha_ * input[i].y + (1.0f - alpha_) * state_[i].y;
      state_[i].z = alpha_ * input[i].z + (1.0f - alpha_) * state_[i].z;
      state_[i].score =
          alpha_ * input[i].score + (1.0f - alpha_) * state_[i].score;
      state_[i].valid = true;
    }

    lastTimestampNs_[i] = timestampNs;
    output[i] = state_[i];
    ++localStats.validOutput;
  }

  if (stats != nullptr) {
    *stats = localStats;
  }
  return output;
}

}  // namespace rehab
