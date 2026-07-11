#pragma once

#include <string>
#include <vector>

namespace rehab {

struct Joint2D {
  std::string name;
  float x{0.0f};
  float y{0.0f};
  float score{0.0f};
  float rawScore{0.0f};
  bool valid{false};
};

struct Joint3D {
  std::string name;
  float x{0.0f};
  float y{0.0f};
  float z{0.0f};
  float score{0.0f};
  bool valid{false};
};

using Skeleton2D = std::vector<Joint2D>;
using Skeleton3D = std::vector<Joint3D>;

}  // namespace rehab
