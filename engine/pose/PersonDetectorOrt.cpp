/*
 * 模块作用：
 * 本文件实现 YOLO 人体框检测。检测结果用于缩小 RTMPose 的输入区域，
 * 但不会直接产生姿态关键点。
 */
#include "engine/pose/PersonDetectorOrt.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <filesystem>
#include <limits>
#include <string>
#include <vector>

#include <opencv2/imgproc.hpp>

#include "engine/util/Logger.h"

#ifdef HAVE_ONNXRUNTIME
#include <onnxruntime_cxx_api.h>
#endif

namespace rehab {

namespace {

constexpr int kCocoPersonClassId = 0;
constexpr int kCocoClassCount = 80;

float intersectionOverUnion(const BoundingBox2D& a, const BoundingBox2D& b) {
  const float x1 = std::max(a.x, b.x);
  const float y1 = std::max(a.y, b.y);
  const float x2 = std::min(a.x + a.w, b.x + b.w);
  const float y2 = std::min(a.y + a.h, b.y + b.h);
  const float w = std::max(0.0f, x2 - x1);
  const float h = std::max(0.0f, y2 - y1);
  const float inter = w * h;
  const float areaA = std::max(0.0f, a.w) * std::max(0.0f, a.h);
  const float areaB = std::max(0.0f, b.w) * std::max(0.0f, b.h);
  const float denom = areaA + areaB - inter;
  if (denom <= 1e-6f) {
    return 0.0f;
  }
  return inter / denom;
}

bool parseOutputLayout(const std::vector<int64_t>& shape,
                       int64_t* boxCount,
                       int64_t* attrCount,
                       bool* attrFirst) {
  // 不同 YOLO 导出版本可能是 [1, attr, box] 或 [1, box, attr]，这里统一识别布局。
  if (shape.size() < 2 || boxCount == nullptr || attrCount == nullptr ||
      attrFirst == nullptr) {
    return false;
  }

  int64_t leadingMul = 1;
  for (std::size_t i = 0; i + 2 < shape.size(); ++i) {
    if (shape[i] <= 0) {
      return false;
    }
    leadingMul *= shape[i];
  }
  if (leadingMul != 1) {
    return false;
  }

  const int64_t dimA = shape[shape.size() - 2];
  const int64_t dimB = shape[shape.size() - 1];
  if (dimA <= 0 || dimB <= 0) {
    return false;
  }

  if (dimA <= 256 && dimB > 256) {
    *attrCount = dimA;
    *boxCount = dimB;
    *attrFirst = true;
    return true;
  }
  if (dimB <= 256 && dimA > 256) {
    *attrCount = dimB;
    *boxCount = dimA;
    *attrFirst = false;
    return true;
  }

  if (dimA <= dimB) {
    *attrCount = dimA;
    *boxCount = dimB;
    *attrFirst = true;
  } else {
    *attrCount = dimB;
    *boxCount = dimA;
    *attrFirst = false;
  }
  return true;
}

}  // namespace

#ifdef HAVE_ONNXRUNTIME
class PersonDetectorOrt::OrtSessionHolder {
 public:
  OrtSessionHolder() : env(ORT_LOGGING_LEVEL_WARNING, "stroke_rehab_person_detector") {}

  Ort::Env env;
  Ort::SessionOptions sessionOptions;
  std::unique_ptr<Ort::Session> session;
  std::string inputName;
  std::string outputName;
};
#endif

PersonDetectorOrt::PersonDetectorOrt() = default;

PersonDetectorOrt::~PersonDetectorOrt() = default;

bool PersonDetectorOrt::initialize(const PersonDetectorConfig& config) {
  config_ = config;
  initialized_ = false;

#ifndef HAVE_ONNXRUNTIME
  Logger::warn("ONNX Runtime is not enabled, person detector disabled.");
  return false;
#else
  if (config_.modelPath.empty()) {
    Logger::warn("Person detector model path is empty.");
    return false;
  }
  if (!std::filesystem::exists(config_.modelPath)) {
    Logger::warn("Person detector model file not found: " + config_.modelPath);
    return false;
  }
  if (config_.inputSize <= 0) {
    Logger::warn("Invalid detector input size.");
    return false;
  }

  try {
    ortHolder_ = std::make_unique<OrtSessionHolder>();
    ortHolder_->sessionOptions.SetIntraOpNumThreads(1);
    ortHolder_->sessionOptions.SetInterOpNumThreads(1);
    ortHolder_->sessionOptions.SetGraphOptimizationLevel(
        GraphOptimizationLevel::ORT_ENABLE_EXTENDED);

    ortHolder_->session = std::make_unique<Ort::Session>(
        ortHolder_->env, std::filesystem::path(config_.modelPath).c_str(),
        ortHolder_->sessionOptions);

    Ort::AllocatorWithDefaultOptions allocator;
    if (ortHolder_->session->GetInputCount() < 1 ||
        ortHolder_->session->GetOutputCount() < 1) {
      Logger::warn("Person detector ONNX session missing input or output.");
      ortHolder_.reset();
      return false;
    }

    auto inputName = ortHolder_->session->GetInputNameAllocated(0, allocator);
    auto outputName = ortHolder_->session->GetOutputNameAllocated(0, allocator);
    const char* rawInput = inputName.get();
    const char* rawOutput = outputName.get();
    ortHolder_->inputName = rawInput != nullptr ? rawInput : "";
    ortHolder_->outputName = rawOutput != nullptr ? rawOutput : "";

    if (ortHolder_->inputName.empty() || ortHolder_->outputName.empty()) {
      Logger::warn("Person detector ONNX IO name is empty.");
      ortHolder_.reset();
      return false;
    }

    const auto outputShape =
        ortHolder_->session->GetOutputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape();
    int64_t boxCount = 0;
    int64_t attrCount = 0;
    bool attrFirst = false;
    if (!parseOutputLayout(outputShape, &boxCount, &attrCount, &attrFirst) ||
        attrCount < 4 + 1) {
      Logger::warn("Person detector output shape is not a supported YOLO layout.");
      ortHolder_.reset();
      return false;
    }

    initialized_ = true;
    Logger::info("Person detector ONNX model loaded: " + config_.modelPath);
    return true;
  } catch (const Ort::Exception& ex) {
    Logger::warn(std::string("Person detector ONNX load failed: ") + ex.what());
    ortHolder_.reset();
    return false;
  } catch (const std::exception& ex) {
    Logger::warn(std::string("Person detector init failed: ") + ex.what());
    ortHolder_.reset();
    return false;
  }
#endif
}

bool PersonDetectorOrt::preprocess(const cv::Mat& bgr,
                                   std::vector<float>* inputTensor,
                                   PreprocessInfo* preprocessInfo) const {
  /*
   * preprocess()
   * 使用 letterbox 保持原图比例，记录 scale/pad，后处理时再把模型坐标还原到 RGB 原图。
   */
  if (bgr.empty() || inputTensor == nullptr || preprocessInfo == nullptr) {
    return false;
  }

  const int inputSize = config_.inputSize;
  const float scale = std::min(static_cast<float>(inputSize) /
                                   static_cast<float>(bgr.cols),
                               static_cast<float>(inputSize) /
                                   static_cast<float>(bgr.rows));

  const int resizedW = std::max(1, static_cast<int>(std::round(bgr.cols * scale)));
  const int resizedH = std::max(1, static_cast<int>(std::round(bgr.rows * scale)));

  cv::Mat resized;
  cv::resize(bgr, resized, cv::Size(resizedW, resizedH), 0.0, 0.0, cv::INTER_LINEAR);

  cv::Mat canvas(inputSize, inputSize, CV_8UC3, cv::Scalar(114, 114, 114));
  const int padX = std::max(0, (inputSize - resizedW) / 2);
  const int padY = std::max(0, (inputSize - resizedH) / 2);
  const cv::Rect dstRoi(padX, padY, std::min(resizedW, inputSize - padX),
                        std::min(resizedH, inputSize - padY));
  const cv::Rect srcRoi(0, 0, dstRoi.width, dstRoi.height);
  resized(srcRoi).copyTo(canvas(dstRoi));

  cv::Mat rgb;
  cv::cvtColor(canvas, rgb, cv::COLOR_BGR2RGB);

  cv::Mat floatImage;
  rgb.convertTo(floatImage, CV_32FC3, 1.0f / 255.0f);

  inputTensor->assign(static_cast<std::size_t>(3 * inputSize * inputSize), 0.0f);
  for (int y = 0; y < inputSize; ++y) {
    const cv::Vec3f* row = floatImage.ptr<cv::Vec3f>(y);
    for (int x = 0; x < inputSize; ++x) {
      const cv::Vec3f px = row[x];
      for (int c = 0; c < 3; ++c) {
        const std::size_t idx =
            static_cast<std::size_t>(c * inputSize * inputSize + y * inputSize + x);
        (*inputTensor)[idx] = px[c];
      }
    }
  }

  preprocessInfo->scale = scale;
  preprocessInfo->padX = static_cast<float>(padX);
  preprocessInfo->padY = static_cast<float>(padY);
  preprocessInfo->inputSize = inputSize;
  return true;
}

std::vector<BoundingBox2D> PersonDetectorOrt::decodeDetections(
    const float* outputData,
    const std::vector<int64_t>& outputShape,
      const PreprocessInfo& preprocessInfo,
      const cv::Size& imageSize) const {
  /*
   * decodeDetections()
   * YOLO 输出是候选框和类别分数；这里只保留 COCO person 类别，
   * 并把输入图坐标反算回原始 RGB 坐标，供 RTMPose 裁剪 ROI。
   */
  std::vector<BoundingBox2D> boxes;
  if (outputData == nullptr || imageSize.width <= 0 || imageSize.height <= 0) {
    return boxes;
  }

  int64_t boxCount = 0;
  int64_t attrCount = 0;
  bool attrFirst = false;
  if (!parseOutputLayout(outputShape, &boxCount, &attrCount, &attrFirst) ||
      boxCount <= 0 || attrCount < 5) {
    return boxes;
  }

  const bool hasObjectness = attrCount >= (5 + kCocoClassCount);
  const int classStart = hasObjectness ? 5 : 4;
  const int64_t classCount = attrCount - classStart;
  if (classCount <= kCocoPersonClassId) {
    return boxes;
  }

  auto at = [&](int64_t boxIdx, int64_t attrIdx) -> float {
    if (attrFirst) {
      const std::size_t idx =
          static_cast<std::size_t>(attrIdx * boxCount + boxIdx);
      return outputData[idx];
    }
    const std::size_t idx =
        static_cast<std::size_t>(boxIdx * attrCount + attrIdx);
    return outputData[idx];
  };

  for (int64_t i = 0; i < boxCount; ++i) {
    const float cx = at(i, 0);
    const float cy = at(i, 1);
    const float w = at(i, 2);
    const float h = at(i, 3);
    if (w <= 1e-6f || h <= 1e-6f) {
      continue;
    }

    int bestClass = -1;
    float bestClassScore = -std::numeric_limits<float>::infinity();
    for (int64_t c = 0; c < classCount; ++c) {
      const float cls = at(i, classStart + c);
      if (cls > bestClassScore) {
        bestClassScore = cls;
        bestClass = static_cast<int>(c);
      }
    }

    if (bestClass != kCocoPersonClassId) {
      continue;
    }

    const float objScore = hasObjectness ? at(i, 4) : 1.0f;
    const float personScore = objScore * bestClassScore;
    if (personScore < config_.confThreshold) {
      continue;
    }

    const float x1Input = cx - 0.5f * w;
    const float y1Input = cy - 0.5f * h;
    const float x2Input = cx + 0.5f * w;
    const float y2Input = cy + 0.5f * h;

    float x1 = (x1Input - preprocessInfo.padX) / preprocessInfo.scale;
    float y1 = (y1Input - preprocessInfo.padY) / preprocessInfo.scale;
    float x2 = (x2Input - preprocessInfo.padX) / preprocessInfo.scale;
    float y2 = (y2Input - preprocessInfo.padY) / preprocessInfo.scale;

    x1 = std::clamp(x1, 0.0f, static_cast<float>(imageSize.width - 1));
    y1 = std::clamp(y1, 0.0f, static_cast<float>(imageSize.height - 1));
    x2 = std::clamp(x2, 0.0f, static_cast<float>(imageSize.width));
    y2 = std::clamp(y2, 0.0f, static_cast<float>(imageSize.height));

    const float outW = x2 - x1;
    const float outH = y2 - y1;
    if (outW <= 1.0f || outH <= 1.0f) {
      continue;
    }

    BoundingBox2D box;
    box.x = x1;
    box.y = y1;
    box.w = outW;
    box.h = outH;
    box.score = personScore;
    box.valid = true;
    boxes.push_back(box);
  }

  return boxes;
}

std::vector<BoundingBox2D> PersonDetectorOrt::nms(
    const std::vector<BoundingBox2D>& boxes) const {
  // NMS 删除同一人体上的重叠框，避免 ROI 在相邻候选之间抖动。
  std::vector<BoundingBox2D> result;
  if (boxes.empty()) {
    return result;
  }

  std::vector<BoundingBox2D> sorted = boxes;
  std::sort(sorted.begin(), sorted.end(),
            [](const BoundingBox2D& a, const BoundingBox2D& b) {
              return a.score > b.score;
            });

  for (const BoundingBox2D& candidate : sorted) {
    bool keep = true;
    for (const BoundingBox2D& picked : result) {
      if (intersectionOverUnion(candidate, picked) > config_.nmsThreshold) {
        keep = false;
        break;
      }
    }
    if (keep) {
      result.push_back(candidate);
    }
  }
  return result;
}

BoundingBox2D PersonDetectorOrt::selectLargestBox(
    const std::vector<BoundingBox2D>& boxes) const {
  BoundingBox2D largest;
  float bestArea = -std::numeric_limits<float>::infinity();
  for (const BoundingBox2D& box : boxes) {
    if (!box.valid) {
      continue;
    }
    const float area = box.w * box.h;
    if (area > bestArea) {
      bestArea = area;
      largest = box;
    }
  }
  return largest;
}

std::vector<BoundingBox2D> PersonDetectorOrt::detect(const cv::Mat& bgr) {
  std::vector<BoundingBox2D> out;
  if (!initialized_ || bgr.empty()) {
    return out;
  }

#ifndef HAVE_ONNXRUNTIME
  return out;
#else
  if (!ortHolder_ || !ortHolder_->session) {
    return out;
  }

  std::vector<float> inputTensor;
  PreprocessInfo prep;
  if (!preprocess(bgr, &inputTensor, &prep)) {
    return out;
  }

  try {
    const std::array<int64_t, 4> inputShape = {1, 3, prep.inputSize, prep.inputSize};
    Ort::MemoryInfo memoryInfo =
        Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
    Ort::Value input = Ort::Value::CreateTensor<float>(
        memoryInfo, inputTensor.data(), inputTensor.size(), inputShape.data(),
        inputShape.size());

    const char* inputName = ortHolder_->inputName.c_str();
    const char* outputName = ortHolder_->outputName.c_str();
    auto outputs = ortHolder_->session->Run(Ort::RunOptions{nullptr}, &inputName,
                                            &input, 1, &outputName, 1);
    if (outputs.empty() || !outputs[0].IsTensor()) {
      return out;
    }

    const auto outputShape = outputs[0].GetTensorTypeAndShapeInfo().GetShape();
    const float* outputData = outputs[0].GetTensorData<float>();
    const std::vector<BoundingBox2D> personBoxes =
        decodeDetections(outputData, outputShape, prep, bgr.size());
    const std::vector<BoundingBox2D> nmsBoxes = nms(personBoxes);
    const BoundingBox2D largest = selectLargestBox(nmsBoxes);
    if (largest.valid) {
      out.push_back(largest);
    }
    return out;
  } catch (const Ort::Exception& ex) {
    Logger::warn(std::string("Person detector inference failed: ") + ex.what());
    return out;
  } catch (const std::exception& ex) {
    Logger::warn(std::string("Person detector runtime failed: ") + ex.what());
    return out;
  }
#endif
}

BoundingBox2D PersonDetectorOrt::detectLargestPerson(const cv::Mat& bgr) {
  // 当前康复采集按单人训练设计，取最大人体框作为主被试。
  const std::vector<BoundingBox2D> boxes = detect(bgr);
  if (boxes.empty()) {
    return {};
  }
  return boxes.front();
}

}  // namespace rehab

