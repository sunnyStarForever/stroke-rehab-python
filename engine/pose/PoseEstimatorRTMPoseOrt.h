/*
 * 模块作用：
 * 本文件声明 RTMPose ONNX 姿态估计器。它接收 RGB 图和人体 ROI，
 * 输出 Halpe26 二维关键点，是后续 Rehab22 映射和 3D 反投影的 2D 来源。
 */
#pragma once

#include <array>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include <opencv2/core.hpp>

#include "engine/pose/PoseEstimator.h"

namespace rehab {

/*
 * PoseEstimatorRTMPoseOrt
 * 职责：
 * 1. 读取 RTMPose 导出参数并加载 ONNX 模型；
 * 2. 根据 bboxProvider_ 提供的人体框裁剪/仿射到模型输入；
 * 3. 解码 SimCC 输出得到 RGB 坐标系下的 Halpe26 关键点；
 * 4. 返回推理耗时、ROI 来源和关键点质量，供 pipeline 决定是否复用。
 */
class PoseEstimatorRTMPoseOrt final : public PoseEstimator {
 public:
  PoseEstimatorRTMPoseOrt();
  ~PoseEstimatorRTMPoseOrt() override;

  bool initialize(const PoseEstimatorConfig& config) override;
  PoseInferenceResult infer(const cv::Mat& bgr) override;
  bool isInitialized() const override { return initialized_; }
  void setBoundingBoxProvider(std::shared_ptr<BoundingBoxProvider> provider) override;

 private:
  struct RuntimeParams {
    int inputWidth{192};     // RTMPose 输入宽度
    int inputHeight{256};    // RTMPose 输入高度
    float padding{1.25f};    // ROI 外扩比例，给四肢运动留出边界
    float simccSplitRatio{2.0f}; // SimCC bin 到像素坐标的缩放比例
    std::array<float, 3> mean{{123.675f, 116.28f, 103.53f}}; // 模型训练均值
    std::array<float, 3> std{{58.395f, 57.12f, 57.375f}};    // 模型训练方差
    bool toRgb{true};        // OpenCV BGR 是否需转成模型训练时的 RGB
  };

  bool loadRuntimeParams();
  bool loadModel();
  BoundingBox2D sanitizeBox(const BoundingBox2D& box, const cv::Size& imageSize) const;
  bool preprocess(const cv::Mat& bgr,
                  const BoundingBox2D& box,
                  std::vector<float>* inputTensor,
                  cv::Matx23f* inverseAffine) const;
  bool decodeSimcc(const float* simccX,
                   const std::vector<int64_t>& xShape,
                   const float* simccY,
                   const std::vector<int64_t>& yShape,
                   const cv::Matx23f& inverseAffine,
                   Halpe26Skeleton2D* outJoints) const;

  PoseEstimatorConfig config_;                 // 模型路径、配置文件和关键点阈值
  RuntimeParams runtimeParams_;                // 预处理和 SimCC 解码参数
  bool initialized_{false};                    // ONNX Runtime session 是否可用
  std::shared_ptr<BoundingBoxProvider> bboxProvider_; // ROI 来源：全图或自适应人体框

#ifdef HAVE_ONNXRUNTIME
  class OrtSessionHolder;
  std::unique_ptr<OrtSessionHolder> ortHolder_;
#endif
};

}  // namespace rehab
