# action_config.py

import re


# =========================================================
# 统一动作配置
# =========================================================

# =========================================================
# 1. 每个动作对应的推荐特征配置
#    best       : 红色 + 斜体 + 下划线，优先使用
#    secondary  : 红色，次优
#    half_count : 绿色，峰数 / 2 作为计数
#    only       : 黄色，只能选取该特征
# =========================================================
ACTION_FEATURE_CONFIG = {
    "M1": {
        # 实时 / 离线默认分割特征
        "segment_feature": "knee_flex_mean",
        "default": "knee_flex_mean",

        # ROM / 质量评分主角度
        "primary_angle": "knee_flexion_mean",

        # 特征优先级
        "best": ["knee_flex_mean", "trunk_sagittal_lean"],
        "secondary": [
            "body_com_height",
            "waist_height",
            "waist_forward_shift",
            "hand_up_max",
        ],
        "half_count": ["hip_flex_mean"],
        "only": [],
    },

    "M2": {
        "segment_feature": "knee_flex_mean",
        "default": "knee_flex_mean",
        "primary_angle": "knee_flexion_mean",

        "best": ["knee_flex_mean"],
        "secondary": [
            "body_com_height",
            "waist_height",
            "pelvis_drop",
            "foot_up_max",
        ],
        "half_count": [],
        "only": [],
    },

    "M3": {
        "segment_feature": "knee_flex_mean",
        "default": "knee_flex_mean",
        "primary_angle": "knee_flexion_mean",

        "best": ["knee_flex_mean"],
        "secondary": [
            "body_com_height",
            "waist_height",
        ],
        "half_count": [],
        "only": [],
    },

    "M4": {
        "segment_feature": "knee_flex_mean",
        "default": "knee_flex_mean",
        "primary_angle": "knee_flexion_mean",

        "best": ["knee_flex_mean", "trunk_sagittal_lean"],
        "secondary": [
            "body_com_height",
            "waist_height",
            "waist_lateral_shift",
            "waist_forward_shift",
            "hand_up_max",
        ],
        "half_count": [],
        "only": [],
    },

    "M5": {
        "segment_feature": "knee_flex_mean",
        "default": "knee_flex_mean",
        "primary_angle": "knee_flexion_mean",

        "best": ["knee_flex_mean"],
        "secondary": [
            "body_com_height",
            "waist_height",
            "waist_forward_shift",
            "foot_up_max",
            "hand_up_max",
        ],
        "half_count": [],
        "only": [],
    },

    "M6": {
        "segment_feature": "foot_forward_max",
        "default": "foot_forward_max",
        "primary_angle": "hip_flexion_mean",

        "best": [],
        "secondary": [
            "waist_height",
            "foot_up_max",
            "foot_forward_max",
        ],
        "half_count": [],
        "only": [],
    },

    "M7": {
        "segment_feature": "hand_up_max",
        "default": "hand_up_max",
        "primary_angle": "shoulder_abduction_mean",

        "best": [
            "body_com_height",
            "waist_height",
            "hand_up_max",
            "hands_dist",
        ],
        "secondary": [],
        "half_count": [],
        "only": [],
    },

    "M8": {
        "segment_feature": "hand_lateral_max",
        "default": "hand_lateral_max",
        "primary_angle": "shoulder_abduction_mean",

        "best": [],
        "secondary": [
            "hand_lateral_max",
            "hands_dist",
        ],
        "half_count": [],
        "only": [],
    },

    "M9": {
        "segment_feature": "hand_forward_max",
        "default": "hand_forward_max",
        "primary_angle": "shoulder_flexion_mean",

        "best": [],
        "secondary": [],
        "half_count": [],
        "only": ["hand_forward_max"],
    },

    "M10": {
        "segment_feature": "hand_forward_max",
        "default": "hand_forward_max",
        "primary_angle": "shoulder_flexion_mean",

        "best": [
            "body_com_height",
            "hand_up_max",
            "hand_lateral_max",
            "hand_forward_max",
            "hands_dist",
        ],
        "secondary": [
            "waist_height",
        ],
        "half_count": [],
        "only": [],
    },
}


# =========================================================
# 动作编号标准化
# =========================================================
def normalize_action_type(action_type):
    """
    支持输入：
        "M1", "m1", "m01", 1, "1"

    返回：
        "M1", "M2", ..., "M10"
    """

    s = str(action_type).strip().upper()

    match = re.search(r"M?0?(\d+)", s)

    if match is None:
        raise ValueError(f"无法解析动作类型: {action_type}")

    action_id = f"M{int(match.group(1))}"

    if action_id not in ACTION_FEATURE_CONFIG:
        raise ValueError(
            f"未知动作类型: {action_id}，应为 M1-M10"
        )

    return action_id


# =========================================================
# 获取完整动作配置
# =========================================================
def get_action_config(action_type):
    action_id = normalize_action_type(action_type)
    cfg = ACTION_FEATURE_CONFIG[action_id].copy()
    cfg["action_id"] = action_id
    return cfg


# =========================================================
# 获取分割特征与计数规则
# =========================================================
def get_segment_feature(action_type, feature_name=None):
    """
    获取用于分割计数的特征。

    参数：
        action_type:
            动作类型，例如 "M1"

        feature_name:
            如果为 None，则使用配置中的 segment_feature；
            如果手动指定，则使用手动指定特征。

    返回：
        feature_name, count_divisor, priority
    """

    action_id = normalize_action_type(action_type)
    cfg = ACTION_FEATURE_CONFIG[action_id]

    if feature_name is None:
        feature_name = cfg.get("segment_feature", cfg.get("default"))

    if feature_name in cfg.get("half_count", []):
        count_divisor = 2
        priority = "half_count"
    elif feature_name in cfg.get("only", []):
        count_divisor = 1
        priority = "only"
    elif feature_name in cfg.get("best", []):
        count_divisor = 1
        priority = "best"
    elif feature_name in cfg.get("secondary", []):
        count_divisor = 1
        priority = "secondary"
    elif feature_name == cfg.get("segment_feature") or feature_name == cfg.get("default"):
        count_divisor = 1
        priority = "default"
    else:
        count_divisor = 1
        priority = "manual_or_unlisted"

    return feature_name, count_divisor, priority


# =========================================================
# 兼容原 action_segmentation_template_eval.py 的 get_feature_rule
# =========================================================
def get_feature_rule(action_type, feature_name=None):
    """
    离线分割模块兼容接口。

    返回格式与原 get_feature_rule() 类似。
    """

    action_id = normalize_action_type(action_type)

    feature_name, count_divisor, priority = get_segment_feature(
        action_id,
        feature_name=feature_name,
    )

    return {
        "action_id": action_id,
        "feature_name": feature_name,
        "priority": priority,
        "count_divisor": count_divisor,
    }


# =========================================================
# 获取 ROM / 质量评分主角度
# =========================================================
def get_primary_angle(action_type):
    action_id = normalize_action_type(action_type)
    return ACTION_FEATURE_CONFIG[action_id]["primary_angle"]


# =========================================================
# 实时模块专用配置
# =========================================================
def get_realtime_action_config(action_type, segment_feature_name=None):
    """
    给 realtime_joint_action_scorer.py 使用。

    返回：
        action_id
        segment_feature
        count_divisor
        primary_angle
        priority
    """

    action_id = normalize_action_type(action_type)
    cfg = ACTION_FEATURE_CONFIG[action_id]

    feature_name, count_divisor, priority = get_segment_feature(
        action_id,
        feature_name=segment_feature_name,
    )

    return {
        "action_id": action_id,
        "segment_feature": feature_name,
        "count_divisor": count_divisor,
        "primary_angle": cfg["primary_angle"],
        "priority": priority,
    }


# =========================================================
# 获取模板特征
# =========================================================
def get_template_features_for_action(action_type, mode="default"):
    """
    给模板构建 / 模板相似度使用。

    mode:
        "default":
            使用默认分割特征。

        "best":
            使用 best 特征。
            如果 best 为空，则退回 default。

        "all_recommended":
            使用 best + secondary + only。
            如果都为空，则退回 default。
    """

    action_id = normalize_action_type(action_type)
    cfg = ACTION_FEATURE_CONFIG[action_id]

    if mode == "default":
        return [cfg["default"]]

    if mode == "best":
        feats = cfg.get("best", [])
        return feats if len(feats) > 0 else [cfg["default"]]

    if mode == "all_recommended":
        feats = []
        for key in ["best", "secondary", "only"]:
            feats.extend(cfg.get(key, []))

        # 去重但保持顺序
        feats = list(dict.fromkeys(feats))

        return feats if len(feats) > 0 else [cfg["default"]]

    raise ValueError(f"未知模板特征模式: {mode}")