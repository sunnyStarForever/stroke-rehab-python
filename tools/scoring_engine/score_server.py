import os

os.environ.setdefault("PYTHONUNBUFFERED", "1")
os.environ.setdefault("PYTHONIOENCODING", "utf-8")
os.environ.setdefault("MPLBACKEND", "Agg")

import argparse
import json
import logging
import sys
import traceback

np = None
RealtimeJointActionScorer = None
normalize_action_type = None


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


def load_dependencies():
    global np, RealtimeJointActionScorer, normalize_action_type
    if np is not None:
        return

    import numpy as _np
    from action_config import normalize_action_type as _normalize_action_type
    from realtime_joint_action_scorer import (
        RealtimeJointActionScorer as _RealtimeJointActionScorer,
    )

    np = _np
    normalize_action_type = _normalize_action_type
    RealtimeJointActionScorer = _RealtimeJointActionScorer


def to_primitive(value):
    if isinstance(value, dict):
        return {k: to_primitive(v) for k, v in value.items()}
    if isinstance(value, list):
        return [to_primitive(v) for v in value]
    if np is not None and isinstance(value, np.ndarray):
        return to_primitive(value.tolist())
    if np is not None and isinstance(value, (np.integer, np.int32, np.int64)):
        return int(value)
    if np is not None and isinstance(value, (np.floating, np.float32, np.float64)):
        return float(value)
    return value


def parse_frame_data(frame_data):
    if not isinstance(frame_data, list):
        raise ValueError("joints 数据必须是列表")
    joints = np.asarray(frame_data, dtype=float)
    if joints.shape != (22, 3):
        raise ValueError(f"joints shape 必须为 (22, 3)，当前为 {joints.shape}")
    return joints


def json_error(message, **extra):
    payload = {
        "ok": False,
        "status": "error",
        "message": message,
        "error": message,
    }
    payload.update(extra)
    return payload


def handle_command(command, analyzer, default_fs):
    if not isinstance(command, dict):
        raise ValueError("输入必须是 JSON 对象")

    request_id = command.get("request_id")
    cmd = command.get("cmd")

    if cmd == "reset":
        analyzer.reset()
        payload = {"ok": True, "status": "ok", "message": "reset"}
        if request_id is not None:
            payload["request_id"] = request_id
        return payload, analyzer

    if cmd == "set_action":
        action_type = command.get("action")
        fs = float(command.get("fs", default_fs))
        if not action_type:
            raise ValueError("set_action 需要 action 字段")
        action_id = normalize_action_type(action_type)
        analyzer = RealtimeJointActionScorer(action_type=action_id, fs=fs)
        payload = {
            "ok": True,
            "status": "ok",
            "message": "action_set",
            "action_id": action_id,
        }
        if request_id is not None:
            payload["request_id"] = request_id
        return payload, analyzer

    if cmd == "frame":
        frame_index = int(command.get("frame_index", -1))
        joints = command.get("joints")
        if joints is None:
            raise ValueError("frame 需要 joints 字段")
        input_frame = parse_frame_data(joints)
        result = to_primitive(analyzer.update(input_frame))
        if not isinstance(result, dict):
            result = {"status": "ok", "result": result}
        result["ok"] = True
        result.setdefault("frame_index", frame_index)
        if request_id is not None:
            result["request_id"] = request_id
        return result, analyzer

    if cmd == "get_debug_state":
        debug_state = analyzer.get_debug_state()
        payload = {
            "ok": True,
            "status": "debug_state",
            "cmd": "get_debug_state",
            "debug_state": to_primitive(debug_state),
        }
        if request_id is not None:
            payload["request_id"] = request_id
        return payload, analyzer

    raise ValueError(f"未知命令: {cmd}")


def main():
    parser = argparse.ArgumentParser(description="P-Coder 实时评分服务")
    parser.add_argument("--action", default="M1", help="动作类型")
    parser.add_argument("--fs", type=float, default=10.0, help="骨骼帧率")
    parser.add_argument("--self-test", action="store_true", help="验证依赖和 JSON 协议")
    args = parser.parse_args()

    load_dependencies()

    if args.self_test:
        log_info("score_server self-test passed")
        emit_json({"ok": True, "service": "score_server", "self_test": True})
        return 0

    action_id = normalize_action_type(args.action)
    analyzer = RealtimeJointActionScorer(
                action_type=action_id,
                fs=args.fs,
            )
    emit_json(
        {
            "ok": True,
            "status": "ok",
            "service": "score_server",
            "message": "ready",
            "action_id": action_id,
        }
    )

    for raw_line in sys.stdin:
        line = raw_line.strip()
        if not line:
            continue

        request_id = None
        try:
            command = json.loads(line)
            if isinstance(command, dict):
                request_id = command.get("request_id")
            payload, analyzer = handle_command(command, analyzer, args.fs)
            emit_json(payload)
        except Exception as ex:
            traceback.print_exc(file=sys.stderr)
            payload = json_error(str(ex), traceback_saved=True)
            if request_id is not None:
                payload["request_id"] = request_id
            emit_json(payload)

    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as e:
        traceback.print_exc(file=sys.stderr)
        try:
            emit_json(
                {
                    "ok": False,
                    "service": "score_server",
                    "error": str(e),
                    "type": type(e).__name__,
                    "traceback_saved": True,
                }
            )
        except Exception:
            pass
        sys.exit(1)
