from rom_quality_report import *
from realtime_joint_action_scorer import *
from offline_html_report import *


# 离线调用API
CSV_PATH = r"data\processed\M1\skeleton3d.csv"
ACTION_TYPE = "M1"
OUTPUT_DIR = r"data\outputs\rom_report\M1_example"

print(ACTION_FEATURE_CONFIG[ACTION_TYPE].keys())

result = analyze_action_csv(
    csv_path=CSV_PATH,
    action_type=ACTION_TYPE,
    template_npz=r"data\outputs\templates\action_templates.npz",
    template_meta_csv=r"data\outputs\templates\action_templates_meta.csv",
    template_feature_names=ACTION_FEATURE_CONFIG[ACTION_TYPE]["default"],
    fs=20,
    min_interval_sec=1.0,
    target_len=100,
    return_features=True,
)

report = generate_report_from_action_api_result(
    csv_path=CSV_PATH,
    action_id=ACTION_TYPE,
    api_result=result,
    output_dir=OUTPUT_DIR,
    plot=True,
    show_plots=True,
)

print(report["rom_summary_df"])
print(report["quality_df"])
print(report["quality_summary"])
print(report["offset_summary_df"])



# 离线报告生成调用API
CSV_PATH = r"data\processed\M1\skeleton3d.csv"
ACTION_TYPE = "M1"
OUTPUT_DIR = r"data\outputs\rom_report\M1_example"

out = generate_offline_action_html_report(
    csv_path=CSV_PATH,
    action_type=ACTION_TYPE,
    output_dir=OUTPUT_DIR,
    template_npz=r"data\outputs\templates\action_templates.npz",
    template_meta_csv=r"data\outputs\templates\action_templates_meta.csv",

    # template_feature_names=ACTION_FEATURE_CONFIG[ACTION_TYPE]["default"],
    template_feature_names=None,

    fs=20,
    min_interval_sec=1.0,
    target_len=100,

    plot=True,
    show_plots=False,

    html_filename="offline_action_report.html",
    html_title="离线动作分析报告 - M1",
    embed_images=True,
)

print("HTML 报告已生成：", out["html_path"])



# 实时调用API (这里采用模拟输入一帧frame 实际使用需要把)
analyzer = RealtimeJointActionScorer(action_type="M1", fs=30)

frame = {
    "waist": [0.0, 1.0, 2.0],
    "spine": [0.0, 1.1, 2.0],
    "chest": [0.0, 1.2, 2.0],
    "neck": [0.0, 1.4, 2.0],
    "head": [0.0, 1.55, 2.0],
    "head_tip": [0.0, 1.7, 2.0],
    "l_collar": [-0.1, 1.35, 2.0],
    "l_shoulder": [-0.25, 1.35, 2.0],
    "l_elbow": [-0.45, 1.1, 2.0],
    "l_hand": [-0.55, 0.9, 2.0],
    "r_collar": [0.1, 1.35, 2.0],
    "r_shoulder": [0.25, 1.35, 2.0],
    "r_elbow": [0.45, 1.1, 2.0],
    "r_hand": [0.55, 0.9, 2.0],
    "l_hip": [-0.15, 0.9, 2.0],
    "l_knee": [-0.15, 0.5, 2.0],
    "l_foot": [-0.15, 0.1, 2.0],
    "l_toe": [-0.15, 0.0, 2.1],
    "r_hip": [0.15, 0.9, 2.0],
    "r_knee": [0.15, 0.5, 2.0],
    "r_foot": [0.15, 0.1, 2.0],
    "r_toe": [0.15, 0.0, 2.1],
}

# while True:
for _ in range(10):
    # TODO 此处仅测试输入无报错 实际使用时应该每获得一帧就输入一次 然后查看评分结果（只要划分出来新的周期就会有结果 反之没有结果）
    frame = frame  # shape=(22,3) 或 dict
    result = analyzer.update(frame)

    current_count = result["count"]

    if result["status"] == "new_completed_cycle":  # 用这个判断是否产生了新的周期！
        last_score = result["last_cycle"]
        print("最后完整周期综合评分:", last_score["overall_score"])
        print("五维评分:", last_score["dimension_scores"])