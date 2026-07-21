import os

os.environ.setdefault("PYTHONUNBUFFERED", "1")
os.environ.setdefault("PYTHONIOENCODING", "utf-8")
os.environ.setdefault("MPLBACKEND", "Agg")

import argparse
import json
import logging
import sys
import traceback


logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)s] %(message)s",
    stream=sys.stderr,
    force=True,
)


def log_info(msg):
    print(f"[INFO] {msg}", file=sys.stderr, flush=True)


def log_warn(msg):
    print(f"[WARN] {msg}", file=sys.stderr, flush=True)


def log_error(msg):
    print(f"[ERROR] {msg}", file=sys.stderr, flush=True)


def emit_json(obj):
    print(json.dumps(obj, ensure_ascii=False), flush=True)


def json_error(message, **extra):
    payload = {
        "ok": False,
        "status": "error",
        "message": message,
        "error": message,
    }
    payload.update(extra)
    return payload


def load_dependencies():
    from action_config import normalize_action_type
    from offline_html_report import generate_offline_action_html_report

    return normalize_action_type, generate_offline_action_html_report


def csv_columns(csv_path):
    with open(csv_path, "r", encoding="utf-8-sig", errors="replace") as f:
        header = f.readline().strip()
    return [item.strip() for item in header.split(",") if item.strip()]


def validate_csv(csv_path):
    if not csv_path:
        return "缺少 --csv 参数"
    if not os.path.exists(csv_path):
        return f"CSV 文件不存在: {csv_path}"
    if os.path.getsize(csv_path) <= 0:
        return f"CSV 文件为空: {csv_path}"

    columns = csv_columns(csv_path)
    if not columns:
        return f"CSV 缺少表头: {csv_path}"

    long_required = {"frame_id", "joint_index", "x_m", "y_m", "z_m"}
    if long_required.issubset(set(columns)):
        return ""

    missing = []
    if "frame_idx" not in columns:
        missing.append("frame_idx")
    has_xyz = any(col.endswith("_x") for col in columns) and any(
        col.endswith("_y") for col in columns
    ) and any(col.endswith("_z") for col in columns)
    if not has_xyz:
        missing.append("*_x/*_y/*_z")
    if missing:
        return "CSV 关键列缺失: " + ", ".join(missing)
    return ""


def prepare_csv_for_report(csv_path, output_dir):
    columns = set(csv_columns(csv_path))
    long_required = {"frame_id", "joint_index", "x_m", "y_m", "z_m"}
    if not long_required.issubset(columns):
        return csv_path

    from transformCSV import convert_long_skeleton_csv_to_wide_csv

    converted_csv = os.path.join(output_dir, "_converted_skeleton3d_wide.csv")
    log_info(f"检测到长表 skeleton3d.csv，转换为宽表: {converted_csv}")
    convert_long_skeleton_csv_to_wide_csv(
        input_csv=csv_path,
        output_csv=converted_csv,
        only_valid=True,
        drop_incomplete_frames=True,
    )
    return converted_csv


def main():
    parser = argparse.ArgumentParser(description="P-Coder 离线报告生成 CLI")
    parser.add_argument("--csv", help="输入骨架 CSV 文件")
    parser.add_argument("--action", help="动作类型")
    parser.add_argument("--out", help="输出目录或 HTML 文件路径")
    parser.add_argument("--fs", type=float, default=20.0, help="骨骼帧率")
    parser.add_argument("--self-test", action="store_true", help="验证依赖和 JSON 协议")
    args = parser.parse_args()

    normalize_action_type, generate_offline_action_html_report = load_dependencies()

    if args.self_test:
        log_info("offline_report self-test passed")
        emit_json({"ok": True, "service": "offline_report", "self_test": True})
        return 0

    csv_error = validate_csv(args.csv)
    if csv_error:
        emit_json(json_error(csv_error, csv_path=args.csv or "", stage="load_csv"))
        return 1
    if not args.action:
        emit_json(json_error("缺少 --action 参数", csv_path=args.csv, stage="validate_args"))
        return 1
    if not args.out:
        emit_json(json_error("缺少 --out 参数", csv_path=args.csv, stage="validate_args"))
        return 1

    try:
        action_id = normalize_action_type(args.action)
    except Exception as ex:
        emit_json(
            json_error(
                str(ex),
                action_id=args.action,
                csv_path=args.csv,
                stage="validate_action",
            )
        )
        return 1

    output_path = args.out
    output_dir = output_path
    html_filename = "offline_action_report.html"
    if os.path.splitext(output_path)[1].lower() in [".html", ".htm"]:
        output_dir = os.path.dirname(output_path)
        html_filename = os.path.basename(output_path)

    os.makedirs(output_dir, exist_ok=True)

    try:
        base_dir = os.path.dirname(os.path.abspath(__file__))
        report_csv = prepare_csv_for_report(args.csv, output_dir)
        result = generate_offline_action_html_report(
            csv_path=report_csv,
            action_type=action_id,
            output_dir=output_dir,
            template_npz=os.path.join(
                base_dir, "data", "outputs", "templates", "action_templates.npz"
            ),
            template_meta_csv=os.path.join(
                base_dir, "data", "outputs", "templates", "action_templates_meta.csv"
            ),
            fs=args.fs,
            html_filename=html_filename,
        )
        html_path = result.get("html_path", "")
        emit_json(
            {
                "ok": True,
                "status": "ok",
                "service": "offline_report",
                "report_dir": output_dir,
                "html_path": html_path,
                "summary": {
                    "actions": 1,
                    "valid_frames": int(result.get("valid_frames", 0) or 0),
                },
            }
        )
        return 0
    except Exception as ex:
        traceback.print_exc(file=sys.stderr)
        emit_json(
            json_error(
                str(ex),
                input_dir=os.path.dirname(os.path.abspath(args.csv)),
                csv_path=args.csv,
                action_id=action_id,
                stage="generate_report",
                traceback_saved=True,
            )
        )
        return 1


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as e:
        traceback.print_exc(file=sys.stderr)
        try:
            emit_json(
                {
                    "ok": False,
                    "service": "offline_report",
                    "error": str(e),
                    "type": type(e).__name__,
                    "traceback_saved": True,
                }
            )
        except Exception:
            pass
        sys.exit(1)
