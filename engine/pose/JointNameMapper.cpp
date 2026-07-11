#include "engine/pose/JointNameMapper.h"

#include <algorithm>
#include <cctype>
#include <unordered_map>

namespace rehab {

namespace {

std::string normalizeKey(const std::string& name) {
  std::string out;
  out.reserve(name.size());
  for (char ch : name) {
    if (ch == '_' || ch == '-' || ch == ' ') {
      continue;
    }
    out.push_back(static_cast<char>(std::tolower(static_cast<unsigned char>(ch))));
  }
  return out;
}

const std::unordered_map<std::string, std::string>& aliases() {
  static const std::unordered_map<std::string, std::string> kAliases = {
      {"waist", "waist"},
      {"pelvis", "pelvis"},
      {"hip", "pelvis"},
      {"spine", "spine"},
      {"chest", "chest"},
      {"neck", "neck"},
      {"head", "head"},
      {"headtip", "head_tip"},
      {"leftcollar", "left_collar"},
      {"leftupperarm", "left_shoulder"},
      {"leftshoulder", "left_shoulder"},
      {"lshoulder", "left_shoulder"},
      {"leftforearm", "left_elbow"},
      {"leftelbow", "left_elbow"},
      {"lelbow", "left_elbow"},
      {"lefthand", "left_wrist"},
      {"leftwrist", "left_wrist"},
      {"lwrist", "left_wrist"},
      {"rightcollar", "right_collar"},
      {"rightupperarm", "right_shoulder"},
      {"rightshoulder", "right_shoulder"},
      {"rshoulder", "right_shoulder"},
      {"rightforearm", "right_elbow"},
      {"rightelbow", "right_elbow"},
      {"relbow", "right_elbow"},
      {"righthand", "right_wrist"},
      {"rightwrist", "right_wrist"},
      {"rwrist", "right_wrist"},
      {"leftupperleg", "left_hip"},
      {"lefthip", "left_hip"},
      {"lhip", "left_hip"},
      {"leftlowerleg", "left_knee"},
      {"leftknee", "left_knee"},
      {"lknee", "left_knee"},
      {"leftfoot", "left_ankle"},
      {"leftankle", "left_ankle"},
      {"lankle", "left_ankle"},
      {"lefttoes", "left_toe"},
      {"lefttoe", "left_toe"},
      {"ltoe", "left_toe"},
      {"rightupperleg", "right_hip"},
      {"righthip", "right_hip"},
      {"rhip", "right_hip"},
      {"rightlowerleg", "right_knee"},
      {"rightknee", "right_knee"},
      {"rknee", "right_knee"},
      {"rightfoot", "right_ankle"},
      {"rightankle", "right_ankle"},
      {"rankle", "right_ankle"},
      {"righttoes", "right_toe"},
      {"righttoe", "right_toe"},
      {"rtoe", "right_toe"},
  };
  return kAliases;
}

std::string parentNameFor(const std::string& canonical) {
  static const std::unordered_map<std::string, std::string> kParents = {
      {"left_knee", "left_hip"},
      {"left_ankle", "left_knee"},
      {"left_toe", "left_ankle"},
      {"right_knee", "right_hip"},
      {"right_ankle", "right_knee"},
      {"right_toe", "right_ankle"},
      {"left_elbow", "left_shoulder"},
      {"left_wrist", "left_elbow"},
      {"right_elbow", "right_shoulder"},
      {"right_wrist", "right_elbow"},
  };
  const auto it = kParents.find(canonical);
  return it == kParents.end() ? std::string{} : it->second;
}

}  // namespace

std::string canonicalJointName(const std::string& name) {
  const std::string key = normalizeKey(name);
  const auto it = aliases().find(key);
  if (it != aliases().end()) {
    return it->second;
  }
  return key;
}

const Joint2D* findParentJoint(const std::vector<Joint2D>& joints,
                               const std::string& canonicalName) {
  const std::string wanted = parentNameFor(canonicalName);
  if (wanted.empty()) {
    return nullptr;
  }
  for (const Joint2D& joint : joints) {
    if (canonicalJointName(joint.name) == wanted && joint.valid) {
      return &joint;
    }
  }
  return nullptr;
}

}  // namespace rehab
