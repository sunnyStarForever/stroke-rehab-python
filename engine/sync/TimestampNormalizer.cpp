#include "engine/sync/TimestampNormalizer.h"

namespace rehab {

void TimestampNormalizer::stamp(FrameEnvelope& frame,
                                uint64_t hostTsNs,
                                uint64_t deviceTsUs) const {
  frame.hostTsNs = hostTsNs;
  frame.deviceTsUs = deviceTsUs;
}

}  // namespace rehab
