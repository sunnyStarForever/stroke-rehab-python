/*
 * 模块作用：
 * 本文件实现软件 RGB-D 对齐。它把 Depth 图中每个有效深度点变换到 RGB
 * 坐标系后重新栅格化，保证后续 2D 关键点的 (u,v) 能在 alignedDepth 中取到同一人体点的深度。
 */
#include "engine/align/SoftwareRegistrationAligner.h"

#include <cmath>
#include <limits>
#include <vector>

#include <opencv2/core.hpp>
#include <opencv2/core/persistence.hpp>
#include <opencv2/core/types.hpp>
#include <opencv2/imgproc.hpp>

#include "engine/util/Logger.h"

namespace rehab {

namespace {

cv::Mat fallbackResize(const cv::Mat& depth, int width, int height) {
  // 标定缺失时只能做最近邻缩放；这能维持显示流程，但不能保证 RGB/Depth 几何一致。
  if (depth.empty()) {
    return {};
  }
  cv::Mat resized;
  cv::resize(depth, resized, cv::Size(width, height), 0, 0, cv::INTER_NEAREST);
  return resized;
}

}  // namespace

SoftwareRegistrationAligner::SoftwareRegistrationAligner(
    const std::string& calibrationFile) {
  valid_ = loadCalibration(calibrationFile);
}

bool SoftwareRegistrationAligner::loadCalibration(
    const std::string& calibrationFile) {
  if (calibrationFile.empty()) {
    Logger::warn("Software alignment calibration file not specified.");
    return false;
  }

  try {
    cv::FileStorage fs(calibrationFile, cv::FileStorage::READ);
    if (!fs.isOpened()) {
      Logger::warn("Failed to open calibration file: " + calibrationFile);
      return false;
    }

    double fx_d = 0.0, fy_d = 0.0, cx_d = 0.0, cy_d = 0.0;
    double fx_rgb = 0.0, fy_rgb = 0.0, cx_rgb = 0.0, cy_rgb = 0.0;
    std::vector<double> rvec, tvec;

    // Depth 内参描述深度像素如何变为深度相机坐标下的三维点。
    fs["depth_intrinsics"]["fx"] >> fx_d;
    fs["depth_intrinsics"]["fy"] >> fy_d;
    fs["depth_intrinsics"]["cx"] >> cx_d;
    fs["depth_intrinsics"]["cy"] >> cy_d;

    // RGB 内参描述 RGB 相机坐标下的三维点如何投影回 RGB 像素平面。
    fs["rgb_intrinsics"]["fx"] >> fx_rgb;
    fs["rgb_intrinsics"]["fy"] >> fy_rgb;
    fs["rgb_intrinsics"]["cx"] >> cx_rgb;
    fs["rgb_intrinsics"]["cy"] >> cy_rgb;

    // 外参把 Depth 相机坐标转换到 RGB 相机坐标；为空或无效会导致 3D 点偏移。
    fs["depth_to_rgb_extrinsics"]["r"] >> rvec;
    fs["depth_to_rgb_extrinsics"]["t"] >> tvec;

    if (fx_d <= 0.0 || fy_d <= 0.0 || fx_rgb <= 0.0 || fy_rgb <= 0.0 ||
        rvec.size() != 9 || tvec.size() != 3) {
      Logger::warn("Invalid calibration parameters in file: " + calibrationFile);
      return false;
    }

    depthIntrinsics_ = cv::Matx33d(fx_d, 0.0, cx_d, 0.0, fy_d, cy_d, 0.0, 0.0, 1.0);
    rgbIntrinsics_ = cv::Matx33d(fx_rgb, 0.0, cx_rgb, 0.0, fy_rgb, cy_rgb, 0.0, 0.0, 1.0);
    rotation_ = cv::Matx33d(rvec[0], rvec[1], rvec[2], rvec[3], rvec[4], rvec[5],
                            rvec[6], rvec[7], rvec[8]);
    translation_ = cv::Vec3d(tvec[0], tvec[1], tvec[2]);

    Logger::info("Loaded software registration calibration: " + calibrationFile);
    return true;
  } catch (const cv::Exception& e) {
    Logger::warn("Exception loading calibration file: " + calibrationFile + " - " + e.what());
    return false;
  }
}

cv::Mat SoftwareRegistrationAligner::align(const FrameEnvelope& depth,
                                           const FrameEnvelope& rgb) const {
  /*
   * align()
   * 输入：已同步但尚未统一坐标的 Depth/RGB 帧。
   * 输出：与 RGB 尺寸一致的 CV_16UC1 alignedDepth。
   * 关键点：姿态 2D 点来自 RGB 图，因此深度必须先对齐到 RGB 坐标系再采样。
   */
  if (depth.image.empty() || rgb.width <= 0 || rgb.height <= 0) {
    return {};
  }
  if (!valid_) {
    // 无标定参数时保底缩放，避免 pipeline 中断；但该结果不能作为高精度 3D 依据。
    return fallbackResize(depth.image, rgb.width, rgb.height);
  }

  cv::Mat depthImage;
  if (depth.image.type() == CV_16UC1) {
    depthImage = depth.image;
  } else {
    depth.image.convertTo(depthImage, CV_16UC1);
  }

  cv::Mat aligned(rgb.height, rgb.width, CV_16UC1, cv::Scalar(0));
  // zBuffer 保留同一 RGB 像素上离相机最近的深度，减少前后景重叠造成的覆盖错误。
  cv::Mat zBuffer(rgb.height, rgb.width, CV_64F, cv::Scalar(std::numeric_limits<double>::infinity()));
  std::size_t projectedPoints = 0;

  const double fx_d = depthIntrinsics_(0, 0);
  const double fy_d = depthIntrinsics_(1, 1);
  const double cx_d = depthIntrinsics_(0, 2);
  const double cy_d = depthIntrinsics_(1, 2);

  const double fx_rgb = rgbIntrinsics_(0, 0);
  const double fy_rgb = rgbIntrinsics_(1, 1);
  const double cx_rgb = rgbIntrinsics_(0, 2);
  const double cy_rgb = rgbIntrinsics_(1, 2);
  const double depthUnitToMeter =
      depth.depthUnitToMeter > 0.0f ? depth.depthUnitToMeter : 0.001;

  for (int y = 0; y < depthImage.rows; ++y) {
    const uint16_t* row = depthImage.ptr<uint16_t>(y);
    for (int x = 0; x < depthImage.cols; ++x) {
      const uint16_t depthValue = row[x];
      if (depthValue == 0) {
        continue;
      }
      const double z = static_cast<double>(depthValue) * depthUnitToMeter;
      // Depth 像素反投影：先从二维像素和 Z 深度恢复 Depth 相机坐标下的三维点。
      const cv::Vec3d depthPoint((static_cast<double>(x) - cx_d) * z / fx_d,
                                 (static_cast<double>(y) - cy_d) * z / fy_d,
                                 z);
      // 外参转换：把 Depth 相机坐标系的点移动到 RGB 相机坐标系。
      const cv::Vec3d rgbPoint = rotation_ * depthPoint + translation_;
      if (rgbPoint[2] <= 0.0) {
        continue;
      }
      // RGB 投影：得到 alignedDepth 中应该写入的 RGB 像素坐标。
      const int u = static_cast<int>(std::round((rgbPoint[0] * fx_rgb / rgbPoint[2]) + cx_rgb));
      const int v = static_cast<int>(std::round((rgbPoint[1] * fy_rgb / rgbPoint[2]) + cy_rgb));
      if (u < 0 || u >= aligned.cols || v < 0 || v >= aligned.rows) {
        continue;
      }
      const double depthMeters = rgbPoint[2];
      const double& currentDepth = zBuffer.at<double>(v, u);
      if (depthMeters < currentDepth) {
        zBuffer.at<double>(v, u) = depthMeters;
        aligned.at<uint16_t>(v, u) = depthValue;
        ++projectedPoints;
      }
    }
  }

  if (projectedPoints == 0) {
    return fallbackResize(depthImage, rgb.width, rgb.height);
  }

  return aligned;
}

}  // namespace rehab
