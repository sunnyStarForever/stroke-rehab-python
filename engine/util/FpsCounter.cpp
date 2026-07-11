#include "engine/util/FpsCounter.h"

namespace rehab {

void FpsCounter::tick(uint64_t nowNs) {
  if (lastTsNs_ == 0) {
    lastTsNs_ = nowNs;
    frames_ = 1;
    return;
  }

  ++frames_;
  const auto elapsedNs = nowNs - lastTsNs_;
  if (elapsedNs >= 1000000000ULL) {
    fps_ = static_cast<double>(frames_) * 1000000000.0 /
           static_cast<double>(elapsedNs);
    frames_ = 0;
    lastTsNs_ = nowNs;
  }
}

}  // namespace rehab
