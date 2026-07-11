#pragma once

#include <cstdint>

namespace rehab {

class FpsCounter {
 public:
  void tick(uint64_t nowNs);
  double fps() const { return fps_; }

 private:
  uint64_t lastTsNs_{0};
  uint64_t frames_{0};
  double fps_{0.0};
};

}  // namespace rehab
