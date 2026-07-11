#!/bin/bash
# Build the rehab_engine pybind11 module on Linux AArch64 (target device).
# Prerequisites:
#   - OpenCV 4.x (system or ../opencv-4.4.0/build)
#   - OpenNI2 (system or ../including/OpenNI)
#   - ONNX Runtime (system or ../including/onnxruntime)
#   - Python 3 + pybind11 (pip install pybind11)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

echo "=== Configuring CMake (full mode) ==="
rm -rf build
cmake -S . -B build \
    -DCMAKE_BUILD_TYPE=Release \
    -DSTROKE_ENGINE_STUB=OFF \
    -DPython_EXECUTABLE="$(which python3)" \
    2>&1 | tail -30

echo ""
echo "=== Building ==="
cmake --build build -j"$(nproc)" 2>&1

echo ""
echo "=== Build completed ==="
find build -name "rehab_engine*.so" -type f 2>/dev/null && echo "Library found!" || echo "ERROR: Library not found"

echo ""
echo "=== Quick test ==="
python3 -c "
import sys
sys.path.insert(0, 'build')
import rehab_engine
print('rehab_engine imported OK')
print('  version:', rehab_engine.__version__)
print('  stub_mode:', rehab_engine._stub_mode)
config = rehab_engine.PipelineConfig()
print('  PipelineConfig created OK')
# Check which subsystems are available
subsystems = [attr for attr in dir(rehab_engine) if not attr.startswith('_')]
print('  Available subsystems:', len(subsystems), 'symbols')
print('All tests passed!')
"