#pragma once

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace rehab {

// 肌肉状态枚举与 cpu1 裸机 emg.c 保持一致，便于 RPMsg 两端直接互认。
enum class EmgMuscleState : uint8_t {
  Rest = 0,
  SmoothFlex = 1,
  Tremor = 2,
  Fatigue = 3,
};

inline const char* emgMuscleStateName(EmgMuscleState state) {
  switch (state) {
    case EmgMuscleState::Rest:
      return "REST";
    case EmgMuscleState::SmoothFlex:
      return "SMOOTH_FLEX";
    case EmgMuscleState::Tremor:
      return "TREMOR";
    case EmgMuscleState::Fatigue:
      return "FATIGUE";
  }
  return "REST";
}

inline EmgMuscleState emgMuscleStateFromByte(uint8_t value) {
  switch (value) {
    case 1:
      return EmgMuscleState::SmoothFlex;
    case 2:
      return EmgMuscleState::Tremor;
    case 3:
      return EmgMuscleState::Fatigue;
    case 0:
    default:
      return EmgMuscleState::Rest;
  }
}

// 单个蓝牙/串口原始采样点。channels 默认两个通道，也允许后续扩展。
struct EmgRawSample {
  uint64_t hostTsNs{0};
  uint32_t seq{0};
  std::vector<int16_t> channels;
};

// 发往 cpu1 的原始采样块。samples 为按帧交错排列：frame0_ch0, frame0_ch1...
struct EmgRawChunk {
  uint64_t hostTsNs{0};
  uint32_t seq{0};
  int sampleRateHz{1000};
  int channelCount{2};
  std::vector<int16_t> interleavedSamples;

  int sampleCount() const {
    if (channelCount <= 0) {
      return 0;
    }
    return static_cast<int>(interleavedSamples.size()) / channelCount;
  }
};

// cpu1 返回的每通道肌电特征。
struct EmgChannelFeature {
  int channel{0};
  float rms{0.0f};
  float zcr{0.0f};
  float cv{0.0f};
  float fatigueIndex{0.0f};
  EmgMuscleState state{EmgMuscleState::Rest};
};

// 与骨骼帧融合时使用的特征帧，hostTsNs 使用对应 raw chunk 的主机时间戳。
struct EmgFeatureFrame {
  uint64_t hostTsNs{0};
  uint32_t seq{0};
  int sampleRateHz{1000};
  std::vector<EmgChannelFeature> channels;

  bool valid() const {
    return hostTsNs > 0 && !channels.empty();
  }
};

// 训练动作区间的肌电统计结果，写入 emg_summary.json 并供报告读取。
struct EmgIntervalSummary {
  int frameCount{0};
  int channelObservations{0};
  double activeRatio{0.0};
  double fatigueRatio{0.0};
  double tremorRatio{0.0};
  double avgRms{0.0};
  double maxRms{0.0};
  double avgFatigueIndex{0.0};
  EmgMuscleState dominantState{EmgMuscleState::Rest};
};

// 运行状态用于 UI 展示，不参与实时计算。
struct EmgRuntimeStatus {
  bool enabled{false};
  bool running{false};
  bool mockMode{false};
  bool bleConnected{false};
  bool rpmsgConnected{false};
  bool recording{false};
  std::string mode{"disabled"};
  std::string serialDevice;
  std::string rpmsgEndpointName;
  std::string message{"EMG disabled"};
  uint64_t rawSamples{0};
  uint64_t rawChunks{0};
  uint64_t featureFrames{0};
  uint64_t parseErrors{0};
  uint64_t rpmsgErrors{0};
  double estimatedSampleRateHz{0.0};
  std::optional<EmgFeatureFrame> latestFeature;
};

}  // namespace rehab
