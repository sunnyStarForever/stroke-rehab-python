/*
 * 模块作用：
 * 本文件实现 RTMPose 姿态推理。它只在 pipeline 决定需要推理时运行，
 * 低频推理配合 2D 复用可以降低 CPU 开销，同时仍用当前深度生成每帧 3D。
 */
#include "engine/pose/PoseEstimatorRTMPoseOrt.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <limits>
#include <regex>
#include <sstream>
#include <string>
#include <vector>

#include <opencv2/imgproc.hpp>

#include "engine/util/Logger.h"

#ifdef HAVE_ONNXRUNTIME
#include <onnxruntime_cxx_api.h>
#endif

namespace rehab {

namespace {

double elapsedMs(const std::chrono::steady_clock::time_point& start,
                 const std::chrono::steady_clock::time_point& end) {
  return std::chrono::duration<double, std::milli>(end - start).count();
}

std::string readTextFile(const std::string& path) {
  std::ifstream ifs(path, std::ios::in);
  if (!ifs.is_open()) {
    return {};
  }
  std::ostringstream oss;
  oss << ifs.rdbuf();
  return oss.str();
}

bool parseFloatTriplet(const std::string& text,
                       const std::string& key,
                       std::array<float, 3>* values) {
  const std::regex pattern("\"" + key +
                           "\"\\s*:\\s*\\[\\s*([-+0-9.eE]+)\\s*,\\s*([-+0-9.eE]+)\\s*,\\s*([-+0-9.eE]+)\\s*\\]");
  std::smatch match;
  if (!std::regex_search(text, match, pattern) || match.size() < 4) {
    return false;
  }
  (*values)[0] = std::stof(match[1].str());
  (*values)[1] = std::stof(match[2].str());
  (*values)[2] = std::stof(match[3].str());
  return true;
}

bool parseFloatValue(const std::string& text, const std::string& key, float* out) {
  const std::regex pattern("\"" + key + "\"\\s*:\\s*([-+0-9.eE]+)");
  std::smatch match;
  if (!std::regex_search(text, match, pattern) || match.size() < 2) {
    return false;
  }
  *out = std::stof(match[1].str());
  return true;
}

bool parseBoolValue(const std::string& text, const std::string& key, bool* out) {
  const std::regex pattern("\"" + key + "\"\\s*:\\s*(true|false)");
  std::smatch match;
  if (!std::regex_search(text, match, pattern) || match.size() < 2) {
    return false;
  }
  *out = (match[1].str() == "true");
  return true;
}

bool parseFirstIntPair(const std::string& text,
                       const std::string& key,
                       int* first,
                       int* second) {
  const std::regex pattern("\"" + key +
                           "\"\\s*:\\s*\\[\\s*(\\d+)\\s*,\\s*(\\d+)\\s*\\]");
  std::smatch match;
  if (!std::regex_search(text, match, pattern) || match.size() < 3) {
    return false;
  }
  *first = std::stoi(match[1].str());
  *second = std::stoi(match[2].str());
  return true;
}

cv::Point2f thirdPoint(const cv::Point2f& a, const cv::Point2f& b) {
  const cv::Point2f direction = a - b;
  return b + cv::Point2f(-direction.y, direction.x);
}

std::string joinNameList(const std::vector<std::string>& names) {
  if (names.empty()) {
    return "(none)";
  }

  std::ostringstream oss;
  for (std::size_t i = 0; i < names.size(); ++i) {
    if (i > 0) {
      oss << ", ";
    }
    oss << names[i];
  }
  return oss.str();
}

}  // namespace

#ifdef HAVE_ONNXRUNTIME
class PoseEstimatorRTMPoseOrt::OrtSessionHolder {
 public:
  OrtSessionHolder() : env(ORT_LOGGING_LEVEL_WARNING, "stroke_rehab_pose") {}

  Ort::Env env;
  Ort::SessionOptions sessionOptions;
  std::unique_ptr<Ort::Session> session;
  std::vector<std::string> sessionInputNames;
  std::vector<std::string> sessionOutputNames;
  std::vector<std::string> runInputNames;
  std::vector<std::string> runOutputNames;
};
#endif

PoseEstimatorRTMPoseOrt::PoseEstimatorRTMPoseOrt()
    : bboxProvider_(std::make_shared<FullImageBoundingBoxProvider>()) {}

PoseEstimatorRTMPoseOrt::~PoseEstimatorRTMPoseOrt() = default;

bool PoseEstimatorRTMPoseOrt::initialize(const PoseEstimatorConfig& config) {
  config_ = config;
  initialized_ = false;
  if (!bboxProvider_) {
    bboxProvider_ = std::make_shared<FullImageBoundingBoxProvider>();
  }

  if (config_.modelPath.empty()) {
    Logger::warn("Pose model path is empty.");
    return false;
  }

  if (!loadRuntimeParams()) {
    Logger::warn("Pose runtime params fallback to built-in defaults.");
  }

  initialized_ = loadModel();
  if (initialized_) {
    Logger::info("RTMPose ONNX model loaded: " + config_.modelPath);
  }
  return initialized_;
}

void PoseEstimatorRTMPoseOrt::setBoundingBoxProvider(
    std::shared_ptr<BoundingBoxProvider> provider) {
  bboxProvider_ = provider ? std::move(provider)
                           : std::make_shared<FullImageBoundingBoxProvider>();
}

bool PoseEstimatorRTMPoseOrt::loadRuntimeParams() {
  // 从 RTMPose 导出的 json 中读取输入尺寸、归一化和 SimCC 参数，避免硬编码和模型不匹配。
  bool loadedAny = false;

  if (!config_.pipelineJsonPath.empty()) {
    const std::string pipelineText = readTextFile(config_.pipelineJsonPath);
    if (!pipelineText.empty()) {
      int width = 0;
      int height = 0;
      if (parseFirstIntPair(pipelineText, "image_size", &width, &height)) {
        runtimeParams_.inputWidth = width;
        runtimeParams_.inputHeight = height;
        loadedAny = true;
      }
      float padding = 0.0f;
      if (parseFloatValue(pipelineText, "padding", &padding)) {
        runtimeParams_.padding = padding;
        loadedAny = true;
      }
      std::array<float, 3> mean = runtimeParams_.mean;
      if (parseFloatTriplet(pipelineText, "mean", &mean)) {
        runtimeParams_.mean = mean;
        loadedAny = true;
      }
      std::array<float, 3> std = runtimeParams_.std;
      if (parseFloatTriplet(pipelineText, "std", &std)) {
        runtimeParams_.std = std;
        loadedAny = true;
      }
      bool toRgb = runtimeParams_.toRgb;
      if (parseBoolValue(pipelineText, "to_rgb", &toRgb)) {
        runtimeParams_.toRgb = toRgb;
        loadedAny = true;
      }
      float splitRatio = 0.0f;
      if (parseFloatValue(pipelineText, "simcc_split_ratio", &splitRatio)) {
        runtimeParams_.simccSplitRatio = splitRatio;
        loadedAny = true;
      }
    }
  }

  if (!config_.detailJsonPath.empty()) {
    const std::string detailText = readTextFile(config_.detailJsonPath);
    if (!detailText.empty()) {
      int width = 0;
      int height = 0;
      if (parseFirstIntPair(detailText, "input_shape", &width, &height)) {
        runtimeParams_.inputWidth = width;
        runtimeParams_.inputHeight = height;
        loadedAny = true;
      }
    }
  }

  return loadedAny;
}

bool PoseEstimatorRTMPoseOrt::loadModel() {
#ifndef HAVE_ONNXRUNTIME
  Logger::warn("ONNX Runtime is not enabled at build time, pose inference disabled.");
  return false;
#else
  try {
    ortHolder_ = std::make_unique<OrtSessionHolder>();
    ortHolder_->sessionOptions.SetIntraOpNumThreads(1);
    ortHolder_->sessionOptions.SetInterOpNumThreads(1);
    ortHolder_->sessionOptions.SetGraphOptimizationLevel(
        GraphOptimizationLevel::ORT_ENABLE_EXTENDED);

    const std::filesystem::path modelPath(config_.modelPath);
    if (!std::filesystem::exists(modelPath)) {
      Logger::warn("Pose model file not found: " + config_.modelPath);
      return false;
    }

    ortHolder_->session = std::make_unique<Ort::Session>(
        ortHolder_->env, modelPath.c_str(), ortHolder_->sessionOptions);

    Ort::AllocatorWithDefaultOptions allocator;
    const std::size_t inputCount = ortHolder_->session->GetInputCount();
    const std::size_t outputCount = ortHolder_->session->GetOutputCount();

    ortHolder_->sessionInputNames.clear();
    for (std::size_t i = 0; i < inputCount; ++i) {
      auto name = ortHolder_->session->GetInputNameAllocated(i, allocator);
      const char* rawName = name.get();
      ortHolder_->sessionInputNames.emplace_back(rawName != nullptr ? rawName : "");
    }

    ortHolder_->sessionOutputNames.clear();
    for (std::size_t i = 0; i < outputCount; ++i) {
      auto name = ortHolder_->session->GetOutputNameAllocated(i, allocator);
      const char* rawName = name.get();
      ortHolder_->sessionOutputNames.emplace_back(rawName != nullptr ? rawName : "");
    }

    Logger::info("RTMPose ONNX session inputs: " +
                 joinNameList(ortHolder_->sessionInputNames));
    Logger::info("RTMPose ONNX session outputs: " +
                 joinNameList(ortHolder_->sessionOutputNames));

    constexpr const char* kInputName = "input";
    constexpr const char* kOutputNameX = "simcc_x";
    constexpr const char* kOutputNameY = "simcc_y";
    ortHolder_->runInputNames = {kInputName};
    ortHolder_->runOutputNames = {kOutputNameX, kOutputNameY};

    auto hasName = [](const std::vector<std::string>& names,
                      const std::string& expected) {
      return std::find(names.begin(), names.end(), expected) != names.end();
    };
    if (!hasName(ortHolder_->sessionInputNames, kInputName) ||
        !hasName(ortHolder_->sessionOutputNames, kOutputNameX) ||
        !hasName(ortHolder_->sessionOutputNames, kOutputNameY)) {
      Logger::warn(
          "Unexpected ONNX IO names. Required input/output names: input, "
          "simcc_x, simcc_y.");
      ortHolder_.reset();
      return false;
    }

    if (ortHolder_->runInputNames.empty() || ortHolder_->runOutputNames.size() < 2) {
      Logger::warn("Unexpected ONNX IO shape: require >=1 input and >=2 outputs.");
      ortHolder_.reset();
      return false;
    }

    return true;
  } catch (const Ort::Exception& ex) {
    Logger::warn(std::string("ONNX Runtime load failed: ") + ex.what());
    ortHolder_.reset();
    return false;
  } catch (const std::exception& ex) {
    Logger::warn(std::string("Pose model load failed: ") + ex.what());
    ortHolder_.reset();
    return false;
  }
#endif
}

BoundingBox2D PoseEstimatorRTMPoseOrt::sanitizeBox(const BoundingBox2D& box,
                                                   const cv::Size& imageSize) const {
  // ROI 无效时回退到全图，保证模型仍可运行；同时把越界框裁剪回图像内部。
  BoundingBox2D out = box;
  if (!out.valid || out.w <= 1.0f || out.h <= 1.0f) {
    out.x = 0.0f;
    out.y = 0.0f;
    out.w = static_cast<float>(imageSize.width);
    out.h = static_cast<float>(imageSize.height);
    out.score = 1.0f;
    out.valid = true;
  }

  out.x = std::clamp(out.x, 0.0f, static_cast<float>(imageSize.width - 1));
  out.y = std::clamp(out.y, 0.0f, static_cast<float>(imageSize.height - 1));
  out.w = std::clamp(out.w, 1.0f, static_cast<float>(imageSize.width));
  out.h = std::clamp(out.h, 1.0f, static_cast<float>(imageSize.height));

  if (out.x + out.w > static_cast<float>(imageSize.width)) {
    out.w = static_cast<float>(imageSize.width) - out.x;
  }
  if (out.y + out.h > static_cast<float>(imageSize.height)) {
    out.h = static_cast<float>(imageSize.height) - out.y;
  }
  out.valid = out.w > 1.0f && out.h > 1.0f;
  return out;
}

bool PoseEstimatorRTMPoseOrt::preprocess(const cv::Mat& bgr,
                                         const BoundingBox2D& box,
                                         std::vector<float>* inputTensor,
                                         cv::Matx23f* inverseAffine) const {
  /*
   * preprocess()
   * RTMPose 不是直接缩放整图，而是围绕人体框做仿射裁剪。
   * inverseAffine 会在解码后把模型输入坐标还原为原 RGB 图像坐标。
   */
  if (bgr.empty() || inputTensor == nullptr || inverseAffine == nullptr) {
    return false;
  }

  const float aspect = static_cast<float>(runtimeParams_.inputWidth) /
                       static_cast<float>(runtimeParams_.inputHeight);
  const cv::Point2f center(box.x + box.w * 0.5f, box.y + box.h * 0.5f);

  float boxWidth = box.w;
  float boxHeight = box.h;
  if (boxWidth > aspect * boxHeight) {
    // 保持模型输入宽高比，防止人体被拉伸导致关键点坐标偏移。
    boxHeight = boxWidth / aspect;
  } else {
    boxWidth = boxHeight * aspect;
  }
  boxWidth *= runtimeParams_.padding;
  boxHeight *= runtimeParams_.padding;

  const cv::Point2f srcDir(0.0f, -0.5f * boxWidth);
  const cv::Point2f dstCenter(runtimeParams_.inputWidth * 0.5f,
                              runtimeParams_.inputHeight * 0.5f);
  const cv::Point2f dstDir(0.0f, -0.5f * runtimeParams_.inputWidth);

  std::array<cv::Point2f, 3> src = {
      center,
      center + srcDir,
      thirdPoint(center, center + srcDir),
  };
  std::array<cv::Point2f, 3> dst = {
      dstCenter,
      dstCenter + dstDir,
      thirdPoint(dstCenter, dstCenter + dstDir),
  };

  const cv::Mat affine = cv::getAffineTransform(src.data(), dst.data());
  cv::Mat warped;
  cv::warpAffine(bgr, warped, affine,
                 cv::Size(runtimeParams_.inputWidth, runtimeParams_.inputHeight),
                 cv::INTER_LINEAR, cv::BORDER_CONSTANT, cv::Scalar(0, 0, 0));

  cv::Mat affineInv;
  cv::invertAffineTransform(affine, affineInv);
  *inverseAffine =
      cv::Matx23f(static_cast<float>(affineInv.at<double>(0, 0)),
                  static_cast<float>(affineInv.at<double>(0, 1)),
                  static_cast<float>(affineInv.at<double>(0, 2)),
                  static_cast<float>(affineInv.at<double>(1, 0)),
                  static_cast<float>(affineInv.at<double>(1, 1)),
                  static_cast<float>(affineInv.at<double>(1, 2)));

  cv::Mat floatImage;
  warped.convertTo(floatImage, CV_32FC3);
  if (runtimeParams_.toRgb) {
    cv::cvtColor(floatImage, floatImage, cv::COLOR_BGR2RGB);
  }

  const int w = runtimeParams_.inputWidth;
  const int h = runtimeParams_.inputHeight;
  inputTensor->assign(static_cast<std::size_t>(3 * w * h), 0.0f);

  for (int y = 0; y < h; ++y) {
    const cv::Vec3f* row = floatImage.ptr<cv::Vec3f>(y);
    for (int x = 0; x < w; ++x) {
      const cv::Vec3f pixel = row[x];
      for (int c = 0; c < 3; ++c) {
        const float normalized =
            (pixel[c] - runtimeParams_.mean[c]) / runtimeParams_.std[c];
        const std::size_t index =
            static_cast<std::size_t>(c * h * w + y * w + x);
        (*inputTensor)[index] = normalized;
      }
    }
  }

  return true;
}

bool PoseEstimatorRTMPoseOrt::decodeSimcc(
    const float* simccX,
    const std::vector<int64_t>& xShape,
    const float* simccY,
    const std::vector<int64_t>& yShape,
                   const cv::Matx23f& inverseAffine,
                   Halpe26Skeleton2D* outJoints) const {
  /*
   * decodeSimcc()
   * SimCC 将 x/y 坐标分别表示成一维分布。每个关节取最大响应位置，
   * 再通过 inverseAffine 回到 RGB 像素坐标，作为后续 Rehab22 映射输入。
   */
  if (simccX == nullptr || simccY == nullptr || outJoints == nullptr) {
    return false;
  }

  auto parseShape = [](const std::vector<int64_t>& shape, int64_t* joints,
                       int64_t* bins) -> bool {
    if (shape.size() >= 3) {
      *joints = shape[shape.size() - 2];
      *bins = shape[shape.size() - 1];
      return true;
    }
    if (shape.size() == 2) {
      *joints = shape[0];
      *bins = shape[1];
      return true;
    }
    return false;
  };

  int64_t xJoints = 0;
  int64_t yJoints = 0;
  int64_t xBins = 0;
  int64_t yBins = 0;
  if (!parseShape(xShape, &xJoints, &xBins) ||
      !parseShape(yShape, &yJoints, &yBins) ||
      xBins <= 0 || yBins <= 0) {
    return false;
  }

  const std::size_t jointCount = static_cast<std::size_t>(
      std::min<int64_t>({xJoints, yJoints, static_cast<int64_t>(kHalpe26JointCount)}));
  for (std::size_t i = 0; i < kHalpe26JointCount; ++i) {
    (*outJoints)[i] = {};
  }

  for (std::size_t j = 0; j < jointCount; ++j) {
    const std::size_t xOffset = j * static_cast<std::size_t>(xBins);
    const std::size_t yOffset = j * static_cast<std::size_t>(yBins);

    float maxX = -std::numeric_limits<float>::infinity();
    float maxY = -std::numeric_limits<float>::infinity();
    int maxXIdx = 0;
    int maxYIdx = 0;

    for (int i = 0; i < xBins; ++i) {
      const float value = simccX[xOffset + static_cast<std::size_t>(i)];
      if (value > maxX) {
        maxX = value;
        maxXIdx = i;
      }
    }
    for (int i = 0; i < yBins; ++i) {
      const float value = simccY[yOffset + static_cast<std::size_t>(i)];
      if (value > maxY) {
        maxY = value;
        maxYIdx = i;
      }
    }

    const float px = static_cast<float>(maxXIdx) / runtimeParams_.simccSplitRatio;
    const float py = static_cast<float>(maxYIdx) / runtimeParams_.simccSplitRatio;

    Keypoint2D point;
    point.x = inverseAffine(0, 0) * px + inverseAffine(0, 1) * py + inverseAffine(0, 2);
    point.y = inverseAffine(1, 0) * px + inverseAffine(1, 1) * py + inverseAffine(1, 2);
    point.score = 0.5f * (maxX + maxY);
    point.rawScore = point.score;
    point.valid = point.score >= config_.minScore;
    // valid=false 表示该关节置信度不足，后续不会采深度，也不会生成有效 3D。
    point.valid = point.score >= config_.minScore;
    (*outJoints)[j] = point;
  }

  return true;
}

PoseInferenceResult PoseEstimatorRTMPoseOrt::infer(const cv::Mat& bgr) {
  /*
   * infer()
   * 输入：当前 RGB 图像。
   * 输出：Halpe26 2D 关键点和本次使用的人体框。
   * 关键点：bboxProvider_ 可能内部跑 YOLO，也可能复用跟踪框；RTMPose 只处理最终 ROI。
   */
  PoseInferenceResult result;
  result.modelLoaded = initialized_;

  if (!initialized_ || bgr.empty()) {
    return result;
  }

#ifndef HAVE_ONNXRUNTIME
  return result;
#else
  if (!ortHolder_ || !ortHolder_->session) {
    return result;
  }

  const auto bboxStart = std::chrono::steady_clock::now();
  const BoundingBox2D requestedBox =
      bboxProvider_ ? bboxProvider_->getPrimaryBox(bgr) : BoundingBox2D{};
  result.bboxMs = elapsedMs(bboxStart, std::chrono::steady_clock::now());
  if (!requestedBox.valid) {
    return result;
  }
  const BoundingBox2D box = sanitizeBox(requestedBox, bgr.size());
  result.usedBox = box;

  const auto poseStart = std::chrono::steady_clock::now();
  std::vector<float> inputData;
  cv::Matx23f inverseAffine;
  if (!preprocess(bgr, box, &inputData, &inverseAffine)) {
    result.poseMs = elapsedMs(poseStart, std::chrono::steady_clock::now());
    return result;
  }

  try {
    const std::array<int64_t, 4> inputShape = {
        1, 3, runtimeParams_.inputHeight, runtimeParams_.inputWidth};
    Ort::MemoryInfo memoryInfo =
        Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
    Ort::Value inputTensor = Ort::Value::CreateTensor<float>(
        memoryInfo, inputData.data(), inputData.size(), inputShape.data(),
        inputShape.size());

    std::vector<const char*> inputNames;
    inputNames.reserve(ortHolder_->runInputNames.size());
    for (const std::string& name : ortHolder_->runInputNames) {
      inputNames.push_back(name.c_str());
    }

    std::vector<const char*> outputNames;
    outputNames.reserve(ortHolder_->runOutputNames.size());
    for (const std::string& name : ortHolder_->runOutputNames) {
      outputNames.push_back(name.c_str());
    }

    auto outputs = ortHolder_->session->Run(
        Ort::RunOptions{nullptr}, inputNames.data(), &inputTensor,
        inputNames.size(), outputNames.data(), outputNames.size());
    if (outputs.size() < 2 || !outputs[0].IsTensor() || !outputs[1].IsTensor()) {
      result.poseMs = elapsedMs(poseStart, std::chrono::steady_clock::now());
      return result;
    }

    const auto xShape = outputs[0].GetTensorTypeAndShapeInfo().GetShape();
    const auto yShape = outputs[1].GetTensorTypeAndShapeInfo().GetShape();
    const float* simccX = outputs[0].GetTensorData<float>();
    const float* simccY = outputs[1].GetTensorData<float>();
    if (!decodeSimcc(simccX, xShape, simccY, yShape, inverseAffine,
                     &result.keypoints)) {
      result.poseMs = elapsedMs(poseStart, std::chrono::steady_clock::now());
      return result;
    }

    float scoreSum = 0.0f;
    int scoreCount = 0;
    for (const Keypoint2D& point : result.keypoints) {
      if (!point.valid) {
        continue;
      }
      scoreSum += point.score;
      ++scoreCount;
    }
    result.validCount = scoreCount;
    result.meanScore = scoreCount > 0 ? (scoreSum / static_cast<float>(scoreCount))
                                      : 0.0f;
    result.modelLoaded = true;
    result.poseMs = elapsedMs(poseStart, std::chrono::steady_clock::now());
    return result;
  } catch (const Ort::Exception& ex) {
    result.poseMs = elapsedMs(poseStart, std::chrono::steady_clock::now());
    Logger::warn(std::string("ONNX Runtime inference failed: ") + ex.what());
    return result;
  }
#endif
}

}  // namespace rehab
