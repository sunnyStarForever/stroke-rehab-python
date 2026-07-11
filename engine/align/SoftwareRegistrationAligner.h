/*
 * 模块作用：
 * 本文件声明软件标定对齐器。当硬件 D2C 不可用时，它使用 calibration.yaml
 * 中的 Depth/RGB 内参和 Depth->RGB 外参，把深度点重投影到 RGB 图像平面。
 */
#pragma once

#include <opencv2/core.hpp>
#include <string>

#include "engine/common/FrameEnvelope.h"

namespace rehab {

/*
 * SoftwareRegistrationAligner
 * 职责：
 * 1. 读取 RGB 内参、Depth 内参和 Depth 到 RGB 的外参；
 * 2. 将每个有效深度像素反投影到 Depth 相机三维坐标；
 * 3. 用外参转换到 RGB 相机坐标，再投影到 RGB 像素；
 * 4. 生成与 RGB 同尺寸的 alignedDepth，供 2D 关键点采样。
 */
class SoftwareRegistrationAligner {
 public:
  explicit SoftwareRegistrationAligner(const std::string& calibrationFile = "");
  cv::Mat align(const FrameEnvelope& depth, const FrameEnvelope& rgb) const;

 private:
  bool loadCalibration(const std::string& calibrationFile);
  cv::Matx33d depthIntrinsics_; // Depth 内参 fx/fy/cx/cy，用于深度像素反投影
  cv::Matx33d rgbIntrinsics_;   // RGB 内参 fx/fy/cx/cy，用于三维点投影到 RGB 像素
  cv::Matx33d rotation_;        // Depth 坐标系到 RGB 坐标系的旋转矩阵
  cv::Vec3d translation_;       // Depth 坐标系到 RGB 坐标系的平移，单位需与深度米制一致
  bool valid_{false};           // 标定是否可用；无效时只能尺寸缩放，3D 精度会下降
};

}  // namespace rehab
