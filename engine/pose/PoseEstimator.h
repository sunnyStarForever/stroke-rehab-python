/*
 * 模块作用：
 * 本文件定义姿态推理接口和人体框提供器接口。
 * YOLO/ROI 只负责给 RTMPose 提供人体区域，RTMPose 才负责输出关键点。
 */
#pragma once

#include <memory>
#include <string>

#include <opencv2/core.hpp>

#include "engine/pose/Halpe26Types.h"

namespace rehab {

struct PoseEstimatorConfig {
  std::string modelPath;        // RTMPose ONNX 模型路径
  std::string pipelineJsonPath; // 预处理参数来源，如输入尺寸、均值方差、padding
  std::string detailJsonPath;   // 模型导出细节参数来源
  std::string deployJsonPath;   // 部署配置路径，保留给模型配置追踪
  float minScore{0.05f};        // 关键点置信度阈值，低于该值标记 valid=false
};

struct BoundingBox2D {
  float x{0.0f};       // 人体框左上角 x，RGB 像素坐标
  float y{0.0f};       // 人体框左上角 y，RGB 像素坐标
  float w{0.0f};       // 人体框宽度，单位像素
  float h{0.0f};       // 人体框高度，单位像素
  float score{1.0f};   // 框置信度或跟踪质量
  bool valid{false};   // false 表示没有可靠人体框，RTMPose 不应基于该框推理
};

/*
 * BoundingBoxProvider
 * 职责：
 * 负责为 RTMPose 提供 ROI。实现可以是全图、YOLO 检测框，
 * 或基于上一帧关键点的自适应跟踪框。
 */
class BoundingBoxProvider {
 public:
  virtual ~BoundingBoxProvider() = default;
  virtual BoundingBox2D getPrimaryBox(const cv::Mat& bgr) = 0;
  virtual void updateFromPose(const BoundingBox2D& usedBox,
                              const Halpe26Skeleton2D& keypoints) = 0;
  virtual void reset() = 0;
  virtual std::string debugState() const = 0;
};

struct PoseInferenceResult {
  Halpe26Skeleton2D keypoints{}; // RTMPose 输出的 Halpe26 关键点
  BoundingBox2D usedBox{};       // 本次推理实际使用的人体 ROI
  float meanScore{0.0f};         // 有效关键点平均置信度，用于判断跟踪质量
  int validCount{0};             // 有效关键点数量
  double bboxMs{0.0};            // 获取 ROI 的耗时，包含 YOLO 或跟踪逻辑
  double poseMs{0.0};            // RTMPose 模型推理和后处理耗时
  bool modelLoaded{false};       // 姿态模型是否可用
};

/*
 * PoseEstimator
 * 职责：
 * 抽象姿态模型实现，主流程只依赖该接口，不关心底层是否为 ONNX Runtime。
 */
class PoseEstimator {
 public:
  virtual ~PoseEstimator();

  virtual bool initialize(const PoseEstimatorConfig& config) = 0;
  virtual PoseInferenceResult infer(const cv::Mat& bgr) = 0;
  virtual bool isInitialized() const = 0;
  virtual void setBoundingBoxProvider(std::shared_ptr<BoundingBoxProvider> provider) = 0;
};

/*
 * FullImageBoundingBoxProvider
 * 职责：
 * 当 YOLO 不可用或未启用自适应 ROI 时，使用整张 RGB 图作为 RTMPose 输入区域。
 */
class FullImageBoundingBoxProvider : public BoundingBoxProvider {
 public:
  BoundingBox2D getPrimaryBox(const cv::Mat& bgr) override;
  void updateFromPose(const BoundingBox2D& usedBox,
                      const Halpe26Skeleton2D& keypoints) override;
  void reset() override;
  std::string debugState() const override;
};

}  // namespace rehab
