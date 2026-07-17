#!/usr/bin/env bash
# Lightweight target-board check before installation.
set -u
echo "Stroke Rehab Python-main board preflight"
echo "arch: $(uname -m)"
echo "kernel: $(uname -r)"
for command_name in python3 cmake g++ v4l2-ctl; do
  if command -v "$command_name" >/dev/null 2>&1; then
    echo "[PASS] $command_name: $(command -v "$command_name")"
  else
    echo "[MISS] $command_name"
  fi
done
for device in /dev/video0 /dev/rpmsg_ctrl0 /dev/rpmsg0 /dev/rfcomm0; do
  if [[ -e "$device" ]]; then echo "[PASS] $device"; else echo "[INFO] $device not present"; fi
done
if ldconfig -p 2>/dev/null | grep -q libOpenNI2; then
  echo "[PASS] OpenNI2 runtime"
else
  echo "[MISS] OpenNI2 runtime"
fi
