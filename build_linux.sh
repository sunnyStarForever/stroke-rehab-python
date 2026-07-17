#!/usr/bin/env bash
# Build only the low-level V4L2/OpenNI adapter used by the Python pipeline.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"
PYTHON_BIN="${PYTHON_BIN:-$SCRIPT_DIR/.venv/bin/python}"
if [[ ! -x "$PYTHON_BIN" ]]; then PYTHON_BIN="$(command -v python3)"; fi

cmake -S . -B build \
  -DCMAKE_BUILD_TYPE=Release \
  -DSTROKE_ENGINE_STUB=OFF \
  -DSTROKE_BUILD_LEGACY_NATIVE_PIPELINE=OFF \
  -DPython_EXECUTABLE="$PYTHON_BIN"
cmake --build build -j"$(nproc)"

CORE_FILE="$(find build -maxdepth 1 -type f -name '_core*.so' -print -quit)"
if [[ -z "$CORE_FILE" ]]; then
  echo "ERROR: build completed without _core*.so" >&2
  exit 1
fi
cp -f "$CORE_FILE" rehab_engine/
"$PYTHON_BIN" -c "import rehab_engine; assert not rehab_engine._STUB_MODE; print('native hardware adapter:', rehab_engine.__engine_version__)"
