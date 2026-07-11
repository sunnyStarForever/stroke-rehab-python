/*
 * 模块作用：
 * 本文件实现关节 3D 反投影。公式为：
 * X=(u-cx)*Z/fx, Y=(v-cy)*Z/fy, Z=depthMeters。
 */
#include "engine/pose/JointProjector3D.h"

#include "engine/util/Logger.h"

namespace rehab {

void JointProjector3D::setIntrinsics(const CameraIntrinsics& intrinsics) {
  intrinsics_ = intrinsics;
}

Keypoint3D JointProjector3D::projectSingle(const Keypoint2D& joint2d,
                                           float depthMeters) const {
  Keypoint3D joint3d;
  joint3d.score = joint2d.score;
  joint3d.valid = joint2d.valid && depthMeters > 0.0f && intrinsics_.valid();

  if (!joint3d.valid) {
    return joint3d;
  }

  // 使用 RGB 内参是因为 depth 已经对齐到 RGB；若内参错误，X/Y 会系统性偏移。
  joint3d.x = (joint2d.x - intrinsics_.cx) * depthMeters / intrinsics_.fx;
  joint3d.y = (joint2d.y - intrinsics_.cy) * depthMeters / intrinsics_.fy;
  joint3d.z = depthMeters;
  return joint3d;
}

Keypoint3D JointProjector3D::projectSingle(
    const Keypoint2D& joint2d,
    const DepthSampleResult& sample) const {
  Keypoint3D joint3d;
  joint3d.score = joint2d.score;
  joint3d.rawPoseScore = joint2d.rawScore;
  joint3d.u = joint2d.x;
  joint3d.v = joint2d.y;
  joint3d.sampledDepthMm = sample.depthRawMm;
  joint3d.sampleMethod = depthSampleMethodToString(sample.method);
  joint3d.sampleReason = sample.reason;
  joint3d.rejectedAsBackground = sample.rejectedAsBackground;
  joint3d.edgeAmbiguous = sample.edgeAmbiguous;
  joint3d.foregroundRecovered = sample.foregroundRecovered;
  joint3d.valid = joint2d.valid && sample.valid && sample.depthMeters > 0.0f &&
                  intrinsics_.valid();

  if (!intrinsics_.valid()) {
    Logger::warn("RGB intrinsics invalid, skip 3D projection");
  }
  if (!joint3d.valid) {
    return joint3d;
  }

  joint3d.x = (joint2d.x - intrinsics_.cx) * sample.depthMeters /
              intrinsics_.fx;
  joint3d.y = (joint2d.y - intrinsics_.cy) * sample.depthMeters /
              intrinsics_.fy;
  joint3d.z = sample.depthMeters;
  return joint3d;
}

Rehab22Skeleton3D JointProjector3D::project(
    const Rehab22Skeleton2D& joints2d,
    const Rehab22DepthSamples& depthsMeters) const {
  Rehab22Skeleton3D joints3d{};
  for (std::size_t i = 0; i < kRehab22JointCount; ++i) {
    joints3d[i] = projectSingle(joints2d[i], depthsMeters[i]);
  }
  return joints3d;
}

Rehab22Skeleton3D JointProjector3D::project(
    const Rehab22Skeleton2D& joints2d,
    const std::array<DepthSampleResult, kRehab22JointCount>& samples) const {
  Rehab22Skeleton3D joints3d{};
  if (!intrinsics_.valid()) {
    Logger::warn("RGB intrinsics invalid, skip 3D projection");
    return joints3d;
  }
  for (std::size_t i = 0; i < kRehab22JointCount; ++i) {
    joints3d[i] = projectSingle(joints2d[i], samples[i]);
  }
  return joints3d;
}

}  // namespace rehab
