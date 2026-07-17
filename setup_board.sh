#!/usr/bin/env bash
# One-time setup for the Linux target. Run from python_version.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

if command -v apt-get >/dev/null 2>&1; then
  SUDO=""; if [[ "$(id -u)" -ne 0 ]]; then SUDO="sudo"; fi
  $SUDO apt-get update
  $SUDO apt-get install -y build-essential cmake python3 python3-dev python3-pip \
    python3-venv libopencv-dev libopenni2-dev v4l-utils libgl1 libglib2.0-0 \
    libasound2-dev espeak-ng
fi

python3 -m venv --system-site-packages .venv
.venv/bin/python -m pip install --upgrade pip setuptools wheel
.venv/bin/python -m pip install -r requirements.txt
PYTHON_BIN="$SCRIPT_DIR/.venv/bin/python" bash build_linux.sh
.venv/bin/python verify_runtime.py --models --ui --require-hardware

echo "Setup complete. Start with: cd $SCRIPT_DIR && .venv/bin/python main.py"
