#include "engine/emg/EmgRpmsgClient.h"

#include <algorithm>
#include <cerrno>
#include <cstring>
#include <filesystem>
#include <utility>

#ifdef __linux__
#include <fcntl.h>
#include <poll.h>
#include <sys/ioctl.h>
#include <unistd.h>
#if defined(__has_include)
#if __has_include(<linux/rpmsg.h>)
#include <linux/rpmsg.h>
#endif
#endif
#endif

#include "engine/emg/EmgProtocol.h"

#ifdef __linux__
#ifndef RPMSG_ADDR_ANY
#define RPMSG_ADDR_ANY 0xFFFFFFFF
#endif
#ifndef RPMSG_CREATE_EPT_IOCTL
struct rpmsg_endpoint_info {
  char name[32];
  uint32_t src;
  uint32_t dst;
};
#define RPMSG_CREATE_EPT_IOCTL _IOW(0xb5, 0x1, struct rpmsg_endpoint_info)
#endif
#endif

namespace rehab {

namespace {

constexpr uint16_t clampU16(int value, int fallback) {
  const int safe = value > 0 ? value : fallback;
  return static_cast<uint16_t>(std::max(1, std::min(65535, safe)));
}

#ifdef __linux__
bool writeFull(int fd, const uint8_t* data, std::size_t size) {
  std::size_t written = 0;
  while (written < size) {
    const ssize_t n = ::write(fd, data + written, size - written);
    if (n < 0) {
      if (errno == EINTR) {
        continue;
      }
      return false;
    }
    if (n == 0) {
      return false;
    }
    written += static_cast<std::size_t>(n);
  }
  return true;
}
#endif

}  // namespace

EmgRpmsgClient::~EmgRpmsgClient() {
  close();
}

void EmgRpmsgClient::configure(const EmgConfig& config) {
  config_ = config;
}

void EmgRpmsgClient::setOnFeature(FeatureCallback callback) {
  std::lock_guard<std::mutex> lock(callbackMutex_);
  featureCallback_ = std::move(callback);
}

void EmgRpmsgClient::setOnStatus(StatusCallback callback) {
  std::lock_guard<std::mutex> lock(callbackMutex_);
  statusCallback_ = std::move(callback);
}

bool EmgRpmsgClient::connect() {
  if (connected_.load()) {
    return true;
  }
  if (!config_.rpmsgEnabled) {
    emitStatus("EMG RPMsg disabled by config");
    return false;
  }

#ifdef __linux__
  const std::string ctrlDevice =
      config_.rpmsgCtrlDevice.empty() ? "/dev/rpmsg_ctrl0" : config_.rpmsgCtrlDevice;
  const std::string dataDevice =
      config_.rpmsgDataDevice.empty() ? "/dev/rpmsg0" : config_.rpmsgDataDevice;
  const std::string endpoint =
      config_.rpmsgEndpointName.empty() ? "emg_rpmsg" : config_.rpmsgEndpointName;

  if (!std::filesystem::exists(ctrlDevice)) {
    emitStatus("RPMsg device not found, please start remoteproc0 and load rpmsg_char first");
    return false;
  }

  ctrlFd_ = ::open(ctrlDevice.c_str(), O_RDWR | O_CLOEXEC);
  if (ctrlFd_ < 0) {
    emitStatus("EMG RPMsg ctrl open failed: " + std::string(std::strerror(errno)));
    return false;
  }

  rpmsg_endpoint_info endpointInfo{};
  std::strncpy(endpointInfo.name, endpoint.c_str(), sizeof(endpointInfo.name) - 1);
  endpointInfo.src = RPMSG_ADDR_ANY;
  endpointInfo.dst = RPMSG_ADDR_ANY;
  if (::ioctl(ctrlFd_, RPMSG_CREATE_EPT_IOCTL, &endpointInfo) < 0) {
    emitStatus("EMG RPMsg endpoint create failed: " + std::string(std::strerror(errno)));
    ::close(ctrlFd_);
    ctrlFd_ = -1;
    return false;
  }

  dataFd_ = ::open(dataDevice.c_str(), O_RDWR | O_NONBLOCK | O_CLOEXEC);
  if (dataFd_ < 0) {
    emitStatus("EMG RPMsg data open failed: " + dataDevice + " (" +
               std::strerror(errno) + ")");
    ::close(ctrlFd_);
    ctrlFd_ = -1;
    return false;
  }

  connected_.store(true);
  readerRunning_.store(true);
  reader_ = std::thread(&EmgRpmsgClient::readLoop, this);
  emitStatus("EMG RPMsg connected endpoint=" + endpoint);
  sendConfig();
  return true;
#else
  emitStatus("EMG RPMsg is only available on Linux");
  return false;
#endif
}

void EmgRpmsgClient::close() {
  readerRunning_.store(false);
  if (reader_.joinable()) {
    reader_.join();
  }

#ifdef __linux__
  std::lock_guard<std::mutex> lock(ioMutex_);
  if (dataFd_ >= 0) {
    ::close(dataFd_);
    dataFd_ = -1;
  }
  if (ctrlFd_ >= 0) {
    ::close(ctrlFd_);
    ctrlFd_ = -1;
  }
#endif
  if (connected_.exchange(false)) {
    emitStatus("EMG RPMsg disconnected");
  }
}

bool EmgRpmsgClient::sendConfig() {
  emg_protocol::ConfigPacket packet;
  packet.header.msg_type = emg_protocol::EmgConfig;
  packet.header.sample_rate_hz = clampU16(config_.sampleRateHz, 1000);
  packet.header.channel_count = static_cast<uint8_t>(
      std::max(1, std::min<int>(emg_protocol::kMaxChannels, config_.channelCount)));
  packet.header.payload_bytes = sizeof(packet.config);
  packet.config.raw_chunk_samples = static_cast<uint16_t>(
      std::max(1, std::min<int>(emg_protocol::kMaxRawSamplesPerPacket,
                                config_.rawChunkSamples)));
  packet.config.active_threshold = config_.activeThreshold;
  packet.config.noise_threshold = config_.noiseThreshold;
  return writePacket(&packet, sizeof(packet.header) + sizeof(packet.config));
}

bool EmgRpmsgClient::sendRawChunk(const EmgRawChunk& chunk) {
  if (!connected_.load()) {
    return false;
  }

  emg_protocol::RawChunkPacket packet;
  packet.header.msg_type = emg_protocol::EmgRawChunk;
  packet.header.seq = chunk.seq;
  packet.header.host_ts_ns = chunk.hostTsNs;
  packet.header.sample_rate_hz = clampU16(chunk.sampleRateHz, 1000);
  const int channelCount =
      std::max(1, std::min<int>(emg_protocol::kMaxChannels, chunk.channelCount));
  const int sampleCount =
      std::max(0, std::min<int>(emg_protocol::kMaxRawSamplesPerPacket,
                                chunk.sampleCount()));
  packet.header.channel_count = static_cast<uint8_t>(channelCount);
  packet.header.sample_count = static_cast<uint8_t>(sampleCount);
  const std::size_t valueCount =
      static_cast<std::size_t>(channelCount) * static_cast<std::size_t>(sampleCount);
  packet.header.payload_bytes =
      static_cast<uint16_t>(valueCount * sizeof(packet.samples[0]));
  if (valueCount > 0) {
    std::copy_n(chunk.interleavedSamples.begin(),
                std::min(valueCount, chunk.interleavedSamples.size()),
                packet.samples);
  }
  return writePacket(&packet, sizeof(packet.header) + packet.header.payload_bytes);
}

void EmgRpmsgClient::readLoop() {
#ifdef __linux__
  uint8_t buffer[emg_protocol::kMaxRpmsgPacketBytes]{};
  while (readerRunning_.load()) {
    int fd = -1;
    {
      std::lock_guard<std::mutex> lock(ioMutex_);
      fd = dataFd_;
    }
    if (fd < 0) {
      break;
    }

    pollfd pfd{};
    pfd.fd = fd;
    pfd.events = POLLIN;
    const int ready = ::poll(&pfd, 1, std::max(1, config_.rpmsgPollTimeoutMs));
    if (!readerRunning_.load()) {
      break;
    }
    if (ready < 0) {
      if (errno == EINTR) {
        continue;
      }
      emitStatus("EMG RPMsg poll failed: " + std::string(std::strerror(errno)));
      break;
    }
    if (ready == 0 || (pfd.revents & POLLIN) == 0) {
      continue;
    }

    const ssize_t n = ::read(fd, buffer, sizeof(buffer));
    if (n < 0) {
      if (errno == EAGAIN || errno == EWOULDBLOCK || errno == EINTR) {
        continue;
      }
      emitStatus("EMG RPMsg read failed: " + std::string(std::strerror(errno)));
      break;
    }
    if (n == 0) {
      continue;
    }

    EmgFeatureFrame frame;
    if (parseFeaturePacket(buffer, static_cast<std::size_t>(n), &frame)) {
      FeatureCallback callback;
      {
        std::lock_guard<std::mutex> lock(callbackMutex_);
        callback = featureCallback_;
      }
      if (callback) {
        callback(frame);
      }
    }
  }
  connected_.store(false);
#endif
}

void EmgRpmsgClient::emitStatus(const std::string& status) {
  StatusCallback callback;
  {
    std::lock_guard<std::mutex> lock(callbackMutex_);
    callback = statusCallback_;
  }
  if (callback) {
    callback(status);
  }
}

bool EmgRpmsgClient::writePacket(const void* data, std::size_t size) {
#ifdef __linux__
  if (!connected_.load() || data == nullptr || size == 0 ||
      size > emg_protocol::kMaxRpmsgPacketBytes) {
    return false;
  }
  std::lock_guard<std::mutex> lock(ioMutex_);
  if (dataFd_ < 0) {
    return false;
  }
  if (!writeFull(dataFd_, static_cast<const uint8_t*>(data), size)) {
    emitStatus("EMG RPMsg write failed: " + std::string(std::strerror(errno)));
    connected_.store(false);
    return false;
  }
  return true;
#else
  (void)data;
  (void)size;
  return false;
#endif
}

bool EmgRpmsgClient::parseFeaturePacket(const uint8_t* data,
                                        std::size_t size,
                                        EmgFeatureFrame* outFrame) {
  if (data == nullptr || outFrame == nullptr ||
      size < sizeof(emg_protocol::Header)) {
    return false;
  }
  const auto* header = reinterpret_cast<const emg_protocol::Header*>(data);
  if (!emg_protocol::validHeader(*header) ||
      header->msg_type != emg_protocol::EmgFeature ||
      size < sizeof(emg_protocol::Header) + header->payload_bytes) {
    return false;
  }

  const auto* packet = reinterpret_cast<const emg_protocol::FeaturePacket*>(data);
  EmgFeatureFrame frame;
  frame.hostTsNs = header->host_ts_ns;
  frame.seq = header->seq;
  frame.sampleRateHz = header->sample_rate_hz;
  const int channelCount =
      std::max(0, std::min<int>(emg_protocol::kMaxChannels, header->channel_count));
  if (header->payload_bytes <
      channelCount * static_cast<int>(sizeof(emg_protocol::ChannelFeaturePayload))) {
    return false;
  }
  frame.channels.reserve(static_cast<std::size_t>(channelCount));
  for (int i = 0; i < channelCount; ++i) {
    EmgChannelFeature feature;
    feature.channel = i;
    feature.rms = packet->channels[i].rms;
    feature.zcr = packet->channels[i].zcr;
    feature.cv = packet->channels[i].cv;
    feature.fatigueIndex = packet->channels[i].fatigue_index;
    feature.state = emgMuscleStateFromByte(packet->channels[i].state);
    frame.channels.push_back(feature);
  }

  *outFrame = std::move(frame);
  return outFrame->valid();
}

}  // namespace rehab
