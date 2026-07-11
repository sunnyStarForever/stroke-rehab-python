#!/bin/bash
# ==========================================================================
# Run THIS script ON THE TARGET BOARD after deploy.sh has transferred files.
# It checks prerequisites, installs missing packages, and builds the engine.
#
# Uses the stroke38 conda environment (NOT system Python 3.8.10).
#
# Usage (on the target board, after deploy):
#   cd ~/stroke-rehab/python_version
#   bash setup_board.sh
#
# Or run the standalone diagnostics first:
#   python check_env.py
# ==========================================================================
set -euo pipefail

RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; CYAN='\033[0;36m'; NC='\033[0m'
info()  { echo -e "${GREEN}[INFO]${NC} $*"; }
warn()  { echo -e "${YELLOW}[WARN]${NC} $*"; }
error() { echo -e "${RED}[ERROR]${NC} $*"; }
step_header() { echo ""; echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"; echo -e "${CYAN}  $*${NC}"; echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"; }

BOARD_DIR="$(cd "$(dirname "$0")/.." && pwd)"
ENGINE_DIR="${BOARD_DIR}/python_version"
INCLUDING_DIR="${BOARD_DIR}/including"

info "============================================"
info " Stroke Rehab — Board Setup"
info " Working directory: ${BOARD_DIR}"
info "============================================"

# ================================================================
# Step 0: Find and activate stroke38 conda environment
# ================================================================
step_header "Step 0: Locate stroke38 conda environment"

CONDA_BASE=""
for candidate in \
    "${HOME}/miniforge3" \
    "${HOME}/miniconda3" \
    "${HOME}/anaconda3" \
    "/opt/conda" \
    "/opt/miniforge3" \
    "/opt/miniconda3"; do
    if [ -f "${candidate}/etc/profile.d/conda.sh" ]; then
        CONDA_BASE="$candidate"
        info "Found conda at: ${CONDA_BASE}"
        break
    fi
done

if [ -z "$CONDA_BASE" ]; then
    error "Could not find any conda installation."
    error "Please install miniforge3 or miniconda3 first."
    exit 1
fi

# Activate conda and then the stroke38 environment
source "${CONDA_BASE}/etc/profile.d/conda.sh"

if ! conda env list | grep -q "stroke38"; then
    error "stroke38 conda environment not found!"
    error "Available environments:"
    conda env list
    error ""
    error "Create it with:"
    error "  conda create -n stroke38 python=3.10 -y"
    error "  conda activate stroke38"
    error "  pip install pybind11 numpy PyQt5 qfluentwidgets pyyaml opencv-python"
    exit 1
fi

conda activate stroke38
info "Python: $(which python3)"
info "Python version: $(python3 --version)"
CONDA_PY=$(which python3)

# ================================================================
# Step 1: Check system prerequisites
# ================================================================
step_header "Step 1: Check system prerequisites"

check_cmd() {
    if command -v "$1" &>/dev/null; then
        info "  $1: $(command -v $1)"
    else
        error "  $1: NOT FOUND — will attempt to install"
        return 1
    fi
}

MISSING=""
check_cmd cmake   || MISSING="${MISSING} cmake"
check_cmd g++     || MISSING="${MISSING} g++"
check_cmd make    || MISSING="${MISSING} make"
check_cmd git     || true  # optional

if [ -n "$MISSING" ]; then
    warn "Installing missing packages:${MISSING}"
    sudo apt-get update -qq
    sudo apt-get install -y -qq ${MISSING}
fi

# ================================================================
# Step 2: Install Python dependencies (in conda env)
# ================================================================
step_header "Step 2: Install Python dependencies (stroke38 env)"

# Verify we're in the right conda env
CURRENT_ENV=$(conda info --envs | grep '*' | awk '{print $1}')
info "Current conda env: ${CURRENT_ENV}"
if [ "$CURRENT_ENV" != "stroke38" ]; then
    error "Not in stroke38 environment! Got: ${CURRENT_ENV}"
    exit 1
fi

pip install --quiet \
    pybind11 \
    numpy \
    PyQt5 \
    qfluentwidgets \
    pyyaml \
    opencv-python 2>/dev/null || true

info "Python packages installed."

# Verify pybind11 cmake files exist
PYBIND11_CMAKE_DIR=$(python3 -c "import pybind11; print(pybind11.get_cmake_dir())" 2>/dev/null || echo "")
if [ -n "$PYBIND11_CMAKE_DIR" ]; then
    info "pybind11 cmake dir: ${PYBIND11_CMAKE_DIR}"
else
    error "pybind11 not found in conda env — install: pip install pybind11"
    exit 1
fi

# ================================================================
# Step 3: Check hardware libraries
# ================================================================
step_header "Step 3: Check hardware libraries"

# OpenCV C++ dev
OCV_DIR=""
if find /usr -name "OpenCVConfig.cmake" 2>/dev/null | grep -q .; then
    OCV_DIR=$(dirname "$(find /usr -name "OpenCVConfig.cmake" 2>/dev/null | head -1)")
    info "OpenCV dev: FOUND (${OCV_DIR})"
else
    warn "OpenCV dev: NOT FOUND"
    warn "Install with: sudo apt install libopencv-dev"
    warn "Or set OpenCV_DIR in CMake to your cross-compiled OpenCV path."
fi

# OpenNI2
OPENNI2_OK=false
if find /usr -name "libOpenNI2.so" 2>/dev/null | grep -q .; then
    info "OpenNI2: system"
    OPENNI2_OK=true
elif [ -f "${INCLUDING_DIR}/OpenNI/sdk/libs/libOpenNI2.so" ]; then
    info "OpenNI2: vendored (${INCLUDING_DIR}/OpenNI)"
    OPENNI2_OK=true
else
    warn "OpenNI2: NOT FOUND — depth capture will use fallback"
fi

# ONNX Runtime — fix broken symlinks
ONNX_OK=false
ONNX_LIB_DIR="${INCLUDING_DIR}/onnxruntime/lib"
if [ -d "$ONNX_LIB_DIR" ]; then
    info "ONNX Runtime dir: ${ONNX_LIB_DIR}"

    # Fix broken symlinks (common issue)
    for so in "$ONNX_LIB_DIR"/libonnxruntime.so "$ONNX_LIB_DIR"/libonnxruntime.so.1; do
        if [ -f "$so" ] && [ ! -s "$so" ]; then
            warn "Fixing broken symlink: $(basename $so) (0 bytes)"
            rm -f "$so"
        fi
    done

    # Recreate symlinks if needed
    if [ -f "${ONNX_LIB_DIR}/libonnxruntime.so.1.25.0" ]; then
        if [ ! -f "${ONNX_LIB_DIR}/libonnxruntime.so.1" ] || [ ! -s "${ONNX_LIB_DIR}/libonnxruntime.so.1" ]; then
            rm -f "${ONNX_LIB_DIR}/libonnxruntime.so.1"
            ln -s libonnxruntime.so.1.25.0 "${ONNX_LIB_DIR}/libonnxruntime.so.1"
            info "Created symlink: libonnxruntime.so.1 -> libonnxruntime.so.1.25.0"
        fi
        if [ ! -f "${ONNX_LIB_DIR}/libonnxruntime.so" ] || [ ! -s "${ONNX_LIB_DIR}/libonnxruntime.so" ]; then
            rm -f "${ONNX_LIB_DIR}/libonnxruntime.so"
            ln -s libonnxruntime.so.1 "${ONNX_LIB_DIR}/libonnxruntime.so"
            info "Created symlink: libonnxruntime.so -> libonnxruntime.so.1"
        fi
        ONNX_OK=true
    else
        warn "libonnxruntime.so.1.25.0 not found in ${ONNX_LIB_DIR}"
    fi
    ls -lh "${ONNX_LIB_DIR}/"libonnxruntime.so* 2>/dev/null || true
else
    warn "ONNX Runtime: NOT FOUND — pose inference will be disabled"
fi

# V4L2
if [ -e /dev/video0 ]; then
    info "RGB camera: /dev/video0 exists"
else
    warn "RGB camera: /dev/video0 not found"
fi

# RPMsg
if [ -e /dev/rpmsg_ctrl0 ]; then
    info "RPMsg: /dev/rpmsg_ctrl0 exists"
else
    warn "RPMsg: /dev/rpmsg_ctrl0 not found (EMG will fall back to mock)"
fi

# ================================================================
# Step 4: Verify transferred files
# ================================================================
step_header "Step 4: Verify transferred files"

check_file() {
    if [ -e "$1" ]; then
        info "  $1"
    else
        error "  $1: MISSING — deploy may have failed"
        return 1
    fi
}

check_file "${BOARD_DIR}/configs/courses.json" || true
check_file "${BOARD_DIR}/configs/calibration.yaml" || true
check_file "${BOARD_DIR}/tools/scoring_engine/score_server.py" || true
check_file "${INCLUDING_DIR}/rtmpose-t/end2end.onnx" || warn "  Models not transferred (large files may have been skipped)"
check_file "${INCLUDING_DIR}/yolov8n/yolov8n.onnx" || true

# ================================================================
# Step 5: Build the C++ engine (using stroke38 conda Python)
# ================================================================
step_header "Step 5: Build C++ engine (stroke38 env)"

cd "${ENGINE_DIR}"
rm -rf build
mkdir -p build

CMAKE_ARGS=(
    -DCMAKE_BUILD_TYPE=Release
    -DSTROKE_ENGINE_STUB=OFF
    -DPython_EXECUTABLE="${CONDA_PY}"
    -Dpybind11_DIR="${PYBIND11_CMAKE_DIR}"
)

# OpenCV
if [ -n "$OCV_DIR" ]; then
    CMAKE_ARGS+=(-DOpenCV_DIR="$OCV_DIR")
fi

# ONNX Runtime
if [ "$ONNX_OK" = true ]; then
    CMAKE_ARGS+=(-DONNXRUNTIME_DIR="${INCLUDING_DIR}/onnxruntime")
fi

# OpenNI2
if [ "$OPENNI2_OK" = true ]; then
    CMAKE_ARGS+=(-DOPENNI2_ROOT="${INCLUDING_DIR}/OpenNI/sdk")
fi

echo ""
info "CMake command:"
echo "  cmake -S . -B build ${CMAKE_ARGS[*]}"
echo ""

cmake -S . -B build "${CMAKE_ARGS[@]}" 2>&1 | tail -30

# Check cmake result
if [ "${PIPESTATUS[0]}" -ne 0 ]; then
    error "CMake configuration failed. See output above."
    exit 1
fi

echo ""
info "Building with $(nproc) cores..."
cmake --build build -j"$(nproc)" 2>&1 | tail -20

# Find the built .so
SO_FILE=$(find build -name "rehab_engine*.so" -type f 2>/dev/null | head -1)
if [ -n "$SO_FILE" ]; then
    SO_SIZE=$(ls -lh "$SO_FILE" | awk '{print $5}')
    info "Engine built: ${SO_FILE} (${SO_SIZE})"

    # Copy to python_version root for easy import
    cp "$SO_FILE" "${ENGINE_DIR}/" 2>/dev/null || true
    info "Copied to: ${ENGINE_DIR}/$(basename "$SO_FILE")"
else
    warn "Engine .so not found — build may have partially failed"
    warn "Check build logs: less ${ENGINE_DIR}/build/CMakeFiles/CMakeError.log"
fi

# ================================================================
# Step 6: Verify the installation
# ================================================================
step_header "Step 6: Verify installation"

cd "${ENGINE_DIR}"

# Run the standalone diagnostic first
info "Running standalone diagnostics..."
python3 check_env.py 2>&1 | head -60
echo ""

# Full import test
info "Testing engine import..."
python3 -c "
import sys; sys.path.insert(0, '.')
import os
os.environ['STROKE_REHAB_ROOT'] = '${BOARD_DIR}'

print('  Working directory:', os.getcwd())
print('  Python:', sys.executable)
print('  Python version:', sys.version)

# Import rehab_engine
try:
    import rehab_engine as re
    print('  Module file:', re.__file__)
    stub = getattr(re, '_stub_mode', True)
    if stub:
        print('  Mode: STUB (C++ engine NOT loaded — using Python stubs)')
        print('  ⚠ WARNING: Real hardware drivers not available.')
        print('  ⚠ Check: did cmake find the right Python?')
    else:
        print('  Mode: FULL (C++ engine loaded successfully!)')
        print('  ✓ Engine .so imported correctly')

    # Test Config
    if stub:
        from rehab_engine._stub import PipelineConfig
        c = PipelineConfig()
        print('  PipelineConfig (stub): OK')
    else:
        c = re.PipelineConfig()
        print('  PipelineConfig: OK')
        print('  RGB device:', c.device.rgb_device_path)
        print('  RGB resolution:', c.device.rgb_width, 'x', c.device.rgb_height)
except Exception as e:
    import traceback
    print('  ERROR during import:')
    traceback.print_exc()
" || warn "  Python import test failed — check build output."

# ================================================================
# Complete!
# ================================================================
echo ""
info "============================================"
info " Setup complete!"
info ""
if [ -n "${SO_FILE:-}" ]; then
    info "  Engine:  $(ls -lh "${SO_FILE}" 2>/dev/null | awk '{print $5, $NF}')"
fi
info "  Conda env: stroke38"
info "  Python:    $(which python3)"
info ""
info "  Quick diagnostics:"
info "    cd ${ENGINE_DIR} && python3 check_env.py"
info ""
info "  Start UI:"
info "    cd ${ENGINE_DIR} && python3 main.py"
info ""
info "  Test without UI:"
info "    cd ${ENGINE_DIR}"
info "    python3 -c \"from rehab_engine.diagnostics import run_diagnostics, print_diagnostics; d=run_diagnostics(); print_diagnostics(d)\""
info "============================================"