#pragma once

#include <cstdint>
#include <chrono>

#ifdef __linux__
#include <time.h>
#endif

namespace rehab {

inline uint64_t monotonicRawNowNs() {
#ifdef __linux__
  timespec ts{};
  if (clock_gettime(CLOCK_MONOTONIC_RAW, &ts) == 0) {
    return static_cast<uint64_t>(ts.tv_sec) * 1000000000ULL +
           static_cast<uint64_t>(ts.tv_nsec);
  }
#endif
  const auto now = std::chrono::steady_clock::now().time_since_epoch();
  return static_cast<uint64_t>(
      std::chrono::duration_cast<std::chrono::nanoseconds>(now).count());
}

inline uint64_t monotonicNowNs() {
#ifdef __linux__
  timespec ts{};
  if (clock_gettime(CLOCK_MONOTONIC, &ts) == 0) {
    return static_cast<uint64_t>(ts.tv_sec) * 1000000000ULL +
           static_cast<uint64_t>(ts.tv_nsec);
  }
#endif
  return monotonicRawNowNs();
}

}  // namespace rehab
