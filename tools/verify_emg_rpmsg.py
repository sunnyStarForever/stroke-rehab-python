"""Repeatable on-board smoke test for the CPU1 EMG RPMsg v1 service."""

from __future__ import annotations

import argparse
import fcntl
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rehab_engine import EmgConfig
from rehab_engine.emg import EmgRawChunk, EmgRpmsgClient


def cleanup_endpoints(paths: list[str]) -> None:
    for path in paths:
        descriptor = os.open(path, os.O_RDWR | os.O_NONBLOCK)
        try:
            fcntl.ioctl(descriptor, EmgRpmsgClient.RPMSG_DESTROY_EPT_IOCTL)
            print(f"destroyed={path}", flush=True)
        finally:
            os.close(descriptor)


def run_cycle(index: int, wait_seconds: float) -> bool:
    config = EmgConfig()
    config.rpmsg_enabled = True
    client = EmgRpmsgClient(config)
    features = []
    client.set_on_status(lambda message: print(f"cycle={index} status={message}", flush=True))
    client.set_on_feature(features.append)
    connected = client.connect()
    print(f"cycle={index} connected={connected}", flush=True)
    raw_sent = False
    if connected:
        sends = []
        chunks_for_window = max(
            1, int(getattr(config, "window_size", 200)) // config.raw_chunk_samples)
        for chunk_index in range(chunks_for_window):
            sends.append(client.send_raw_chunk(
                EmgRawChunk(
                    host_ts_ns=time.monotonic_ns(),
                    seq=index * 100 + chunk_index,
                    sample_rate_hz=config.sample_rate_hz,
                    channel_count=config.channel_count,
                    interleaved_samples=tuple(
                        range(config.raw_chunk_samples * config.channel_count)),
                )
            ))
        raw_sent = all(sends)
        time.sleep(wait_seconds)
    print(
        f"cycle={index} raw_sent={raw_sent} features={len(features)} "
        f"invalid={client.invalid_feature_packets} protocol_errors={client.protocol_errors}",
        flush=True,
    )
    client.close()
    return connected and raw_sent and bool(features)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cycles", type=int, default=3)
    parser.add_argument("--wait", type=float, default=1.0)
    parser.add_argument("--cleanup", nargs="*", default=[])
    args = parser.parse_args()
    cleanup_endpoints(args.cleanup)
    results = [run_cycle(index + 1, args.wait) for index in range(max(1, args.cycles))]
    print(f"result={sum(results)}/{len(results)} cycles passed", flush=True)
    return 0 if all(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
