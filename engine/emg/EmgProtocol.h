#pragma once

#include <cstddef>
#include <cstdint>

namespace rehab::emg_protocol {

// RPMsg 单包按 256 字节上限设计，兼容 rpmsg-demo-single.c 的 MAX_DATA_LENGTH。
constexpr std::size_t kMaxRpmsgPacketBytes = 256;
constexpr uint32_t kMagic = 0x31474D45u;  // "EMG1" 小端表示
constexpr uint8_t kVersion = 1;
constexpr uint8_t kMaxChannels = 2;
constexpr uint8_t kDefaultRawSamplesPerPacket = 16;
constexpr uint8_t kMaxRawSamplesPerPacket = 32;
constexpr uint16_t kHeaderSizeBytes = 28;

enum MessageType : uint8_t {
  EmgRawChunk = 1,
  EmgFeature = 2,
  EmgConfig = 3,
  EmgHeartbeat = 4,
  EmgError = 5,
};

enum MuscleState : uint8_t {
  Rest = 0,
  SmoothFlex = 1,
  Tremor = 2,
  Fatigue = 3,
};

#pragma pack(push, 1)

struct Header {
  uint32_t magic{kMagic};
  uint8_t version{kVersion};
  uint8_t msg_type{0};
  uint16_t header_size{kHeaderSizeBytes};
  uint32_t seq{0};
  uint64_t host_ts_ns{0};
  uint16_t sample_rate_hz{1000};
  uint8_t channel_count{kMaxChannels};
  uint8_t sample_count{0};
  uint16_t payload_bytes{0};
  uint16_t reserved{0};
};

struct ConfigPayload {
  uint16_t raw_chunk_samples{kDefaultRawSamplesPerPacket};
  uint16_t reserved{0};
  float active_threshold{800.0f};
  float noise_threshold{15.0f};
};

struct ConfigPacket {
  Header header;
  ConfigPayload config;
};

struct RawChunkPacket {
  Header header;
  int16_t samples[kMaxChannels * kMaxRawSamplesPerPacket]{};
};

struct ChannelFeaturePayload {
  float rms{0.0f};
  float zcr{0.0f};
  float cv{0.0f};
  float fatigue_index{0.0f};
  uint8_t state{Rest};
  uint8_t reserved[3]{};
};

struct FeaturePacket {
  Header header;
  ChannelFeaturePayload channels[kMaxChannels]{};
};

struct ErrorPacket {
  Header header;
  int32_t code{0};
  char message[96]{};
};

#pragma pack(pop)

static_assert(sizeof(Header) <= kMaxRpmsgPacketBytes, "EMG header too large");
static_assert(sizeof(Header) == kHeaderSizeBytes, "EMG header size changed");
static_assert(sizeof(ConfigPacket) <= kMaxRpmsgPacketBytes, "EMG config packet too large");
static_assert(sizeof(RawChunkPacket) <= kMaxRpmsgPacketBytes, "EMG raw packet too large");
static_assert(sizeof(FeaturePacket) <= kMaxRpmsgPacketBytes, "EMG feature packet too large");
static_assert(sizeof(ErrorPacket) <= kMaxRpmsgPacketBytes, "EMG error packet too large");

inline bool validHeader(const Header& header) {
  return header.magic == kMagic && header.version == kVersion &&
         header.header_size == sizeof(Header) &&
         header.payload_bytes <= kMaxRpmsgPacketBytes - sizeof(Header);
}

}  // namespace rehab::emg_protocol
