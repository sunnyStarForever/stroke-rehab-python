#include <cassert>
#include <cstdint>
#include <string>

#include "engine/common/FrameEnvelope.h"
#include "engine/sync/TimestampNormalizer.h"

using rehab::FrameEnvelope;
using rehab::TimestampNormalizer;

int main() {
  TimestampNormalizer normalizer(10);

  FrameEnvelope fallback;
  normalizer.stampDeviceMicroseconds(fallback, 5'000'000'000ULL, 0);
  assert(fallback.syncTsNs == fallback.arrivalTsNs);
  assert(fallback.clockQuality == "host_fallback");
  assert(fallback.clockReason == "missing_device_timestamp");

  FrameEnvelope first;
  normalizer.stampDeviceMicroseconds(first, 10'000'000'000ULL, 1'000'000ULL);
  assert(first.syncTsNs == first.arrivalTsNs);
  assert(first.clockQuality == "normalized_device");

  FrameEnvelope jittered;
  normalizer.stampDeviceMicroseconds(jittered, 10'035'000'000ULL, 1'033'000ULL);
  assert(jittered.syncTsNs <= jittered.arrivalTsNs);
  assert(jittered.clockQuality == "normalized_device");

  FrameEnvelope reset;
  normalizer.stampDeviceMicroseconds(reset, 11'000'000'000ULL, 500'000ULL);
  assert(reset.clockResetCount == 1);
  assert(reset.clockReason == "device_timestamp_backwards");

  FrameEnvelope native;
  normalizer.stampNativeMonotonic(native, 20'000'000'000ULL,
                                  19'990'000'000ULL, 19'990'000ULL);
  assert(native.syncTsNs == 19'990'000'000ULL);
  assert(native.clockQuality == "native_monotonic");

  FrameEnvelope invalidNative;
  normalizer.stampNativeMonotonic(invalidNative, 21'000'000'000ULL, 0, 0);
  assert(invalidNative.syncTsNs == invalidNative.arrivalTsNs);
  assert(invalidNative.clockQuality == "host_fallback");
  assert(invalidNative.clockReason == "invalid_native_monotonic");

  FrameEnvelope jump;
  normalizer.stampDeviceMicroseconds(jump, 30'000'000'000ULL, 2'000'000ULL);
  FrameEnvelope jumped;
  normalizer.stampDeviceMicroseconds(jumped, 31'033'000'000ULL, 4'500'000ULL);
  assert(jumped.clockResetCount >= 2);
  assert(jumped.clockReason == "device_timestamp_jump");
  return 0;
}
