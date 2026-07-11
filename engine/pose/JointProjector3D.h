/*
 * 模块作用：
 * 本文件声明 2D+Depth 到 3D 的反投影器。
 * 因为 alignedDepth 已经对齐到 RGB 坐标系，所以这里使用 RGB 内参计算 3D 坐标。
 */
#pragma once

#include <array>

#include "engine/pose/DepthSampler.h"
#include "engine/pose/Rehab22Types.h"

namespace rehab {

struct CameraIntrinsics {
  float fx{0.0f}; // RGB 相机 x 方向焦距，像素单位
  float fy{0.0f}; // RGB 相机 y 方向焦距，像素单位
  float cx{0.0f}; // RGB 主点 x 坐标
  float cy{0.0f}; // RGB 主点 y 坐标

  bool valid() const {
    return fx > 0.0f && fy > 0.0f && cx >= 0.0f && cy >= 0.0f;
  }
};

/*
 * JointProjector3D
 * 职责：
 * 1. 持有 RGB 相机内参；
 * 2. 使用 2D 像素坐标和采样深度 Z 反投影；
 * 3. 输出 RGB 相机坐标系下、单位为米的 Rehab22 3D 骨架。
 */
class JointProjector3D {
 public:
  void setIntrinsics(const CameraIntrinsics& intrinsics);
  bool intrinsicsValid() const { return intrinsics_.valid(); }
  Rehab22Skeleton3D project(const Rehab22Skeleton2D& joints2d,
                            const Rehab22DepthSamples& depthsMeters) const;
  Rehab22Skeleton3D project(
      const Rehab22Skeleton2D& joints2d,
      const std::array<DepthSampleResult, kRehab22JointCount>& samples) const;

 private:
  Keypoint3D projectSingle(const Keypoint2D& joint2d, float depthMeters) const;
  Keypoint3D projectSingle(const Keypoint2D& joint2d,
                           const DepthSampleResult& sample) const;

  CameraIntrinsics intrinsics_;
};

}  // namespace rehab
