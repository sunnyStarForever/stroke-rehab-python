/*
 * 模块作用：
 * 本文件声明基于 ONNX Runtime 的 YOLO 人体检测器。
 * 它只输出人体框，不负责关键点；关键点由 RTMPose 在该框内完成。
 */
#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include <opencv2/core.hpp>

#include "engine/pose/PoseEstimator.h"

namespace rehab {

struct PersonDetectorConfig {
  std::string modelPath;       // YOLO ONNX 模型路径
  int inputSize{320};          // 方形输入尺寸，越大越准但耗时越高
  float confThreshold{0.35f};  // 过滤低置信度人体框
  float nmsThreshold{0.45f};   // NMS 阈值，抑制重复框
};

/*
 * PersonDetectorOrt
 * 职责：
 * 1. 加载 YOLO 人体检测模型；
 * 2. 将 RGB 图 letterbox 到模型输入尺寸；
 * 3. 解码 COCO person 类别框并做 NMS；
 * 4. 返回面积最大的单人框给 RTMPose ROI。
 */
class PersonDetectorOrt {
 public:
  PersonDetectorOrt();
  ~PersonDetectorOrt();

  bool initialize(const PersonDetectorConfig& config);
  bool isInitialized() const { return initialized_; }

  std::vector<BoundingBox2D> detect(const cv::Mat& bgr);
  BoundingBox2D detectLargestPerson(const cv::Mat& bgr);

 private:
  struct PreprocessInfo {
    float scale{1.0f};
    float padX{0.0f};
    float padY{0.0f};
    int inputSize{320};
  };

  bool preprocess(const cv::Mat& bgr,
                  std::vector<float>* inputTensor,
                  PreprocessInfo* preprocessInfo) const;
  std::vector<BoundingBox2D> decodeDetections(
      const float* outputData,
      const std::vector<int64_t>& outputShape,
      const PreprocessInfo& preprocessInfo,
      const cv::Size& imageSize) const;
  std::vector<BoundingBox2D> nms(const std::vector<BoundingBox2D>& boxes) const;
  BoundingBox2D selectLargestBox(const std::vector<BoundingBox2D>& boxes) const;

  PersonDetectorConfig config_;
  bool initialized_{false};

#ifdef HAVE_ONNXRUNTIME
  class OrtSessionHolder;
  std::unique_ptr<OrtSessionHolder> ortHolder_;
#endif
};

}  // namespace rehab
