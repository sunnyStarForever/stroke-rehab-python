#pragma once

#include <string>
#include <vector>

#include "engine/pose/SkeletonTypes.h"

namespace rehab {

std::string canonicalJointName(const std::string& name);

const Joint2D* findParentJoint(const std::vector<Joint2D>& joints,
                               const std::string& canonicalName);

}  // namespace rehab
