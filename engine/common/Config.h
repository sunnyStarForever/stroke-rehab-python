#pragma once

#include <cstddef>
#include <cstdint>
#include <string>

namespace rehab {

struct DeviceConfig {
  std::string openniDeviceUri;
  std::string rgbDevicePath;
  std::string rgbPixelFormat{"MJPG"};
  int rgbDeviceIndex{0};
  int rgbWidth{640};
  int rgbHeight{480};
  int rgbFps{30};
  bool mirrorRgbAtCapture{true};

  std::string depthPixelFormat{"DEPTH_1_MM"};
  int depthWidth{640};
  int depthHeight{480};
  int depthFps{30};
  bool enableHardwareD2C{true};
  bool enableOpenNIColorStreamForDebug{false};
  bool enableOpenNIDepthColorSync{false};
  int latestQueueSize{1};
  double rawPerfLogIntervalSec{1.0};

  bool enableCpuAffinity{true};
  int rgbCaptureCpu{0};
  int depthCaptureCpu{0};
};

struct SyncConfig {
  int64_t matchThresholdNs{20LL * 1000LL * 1000LL};
  std::size_t queueSize{30};
};

struct PoseConfig {
  bool enablePose{true};
  bool enableAdaptiveRoi{true};
  std::string modelPath;
  std::string detectorModelPath;
  std::string pipelineJsonPath;
  std::string detailJsonPath;
  std::string deployJsonPath;
  float minScore{0.15f};
  int maxPairQueue{2};
  int poseInterval{6};
  bool enablePoseReuse{true};
  bool enableCpuAffinity{true};
  int syncOrUiCpu{1};
  int detectorCpu{2};
  int poseCpu{3};
  int detectorInterval{30};
  float roiMarginRatio{0.20f};
  float minTrackMeanScore{0.25f};
  int minTrackValidPoints{6};
  int maxConsecutiveMisses{3};
  float motionTriggerRatio{0.35f};
  int detectorInputSize{320};
  float detectorConfThreshold{0.35f};
  float detectorNmsThreshold{0.45f};
  int depthMedianWindow{5};
  bool enableSmoothing{true};
  float smoothingAlpha{0.35f};
};

struct DepthSamplerConfig {
  int minDepthMm{300};
  int maxDepthMm{5000};
  int bodyDepthBandMm{700};
  int edgeBodyDepthBandMm{900};
  int backgroundRejectMarginMm{500};
  int backgroundMatchBandMm{300};
  float foregroundPercentile{0.20f};
  int minForegroundPixels{3};
  int hipRadius{2};
  int kneeRadius{3};
  int ankleRadius{4};
  int toeRadius{5};
  int wristRadius{4};
  int defaultRadius{2};
  bool limbInwardSearchEnabled{true};
  int limbInwardSteps{10};
  int limbInwardStepPx{3};
  int limbInwardRadius{3};
};

struct SkeletonFilterConfig {
  std::string mode{"ema"};
  float alphaGood{0.65f};
  float alphaLowConfidence{0.35f};
  float alphaRecovered{0.45f};
  float alphaInvalid{0.0f};
  float maxZJumpM{0.45f};
  float maxJointSpeedMps{2.5f};
  bool holdLastWhenInvalid{true};
  int maxHoldFrames{5};
};

struct DebugConfig {
  bool saveDepthSamplingOverlay{false};
  bool saveSkeletonRawCsv{true};
  bool saveSkeletonEmaCsv{true};
};

struct EmgConfig {
  bool enabled{false};
  std::string mode{"mock"};              // disabled/mock/real
  std::string captureBackend{"serial"};  // serial/bluez
  std::string serialDevice{"/dev/rfcomm0"};
  int serialBaudRate{115200};
  std::string bleNamePrefix{"ESP32_EMG"};
  std::string bleAddress;
  std::string bleServiceUuid;
  std::string bleNotifyCharUuid;
  int sampleRateHz{1000};
  int channelCount{2};
  int rawChunkSamples{16};
  bool rpmsgEnabled{true};
  std::string rpmsgCtrlDevice{"/dev/rpmsg_ctrl0"};
  std::string rpmsgDataDevice{"/dev/rpmsg0"};
  std::string rpmsgEndpointName{"emg_rpmsg"};
  int rpmsgPollTimeoutMs{5};
  float activeThreshold{800.0f};
  float noiseThreshold{15.0f};
};

struct PipelineConfig {
  DeviceConfig device;
  SyncConfig sync;
  PoseConfig pose;
  DepthSamplerConfig depthSampler;
  SkeletonFilterConfig skeletonFilter;
  DebugConfig debug;
  EmgConfig emg;
  std::string calibrationFile;
  bool recordPairs{false};
  std::string recordPath{"recordings"};
};

}  // namespace rehab
