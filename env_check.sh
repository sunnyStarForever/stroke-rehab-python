#!/bin/bash
# Quick environment check — run on the target board BEFORE full setup.
# Usage: ssh root@10.161.95.152 'bash -s' < env_check.sh
#    or: copy to board and run: bash env_check.sh

echo "============================================"
echo " Stroke Rehab — Environment Check"
echo " $(date)"
echo "============================================"

echo ""
echo "--- System ---"
echo "  arch:     $(uname -m)"
echo "  os:       $(cat /etc/os-release 2>/dev/null | grep PRETTY | cut -d= -f2 | tr -d '"' || echo 'unknown')"
echo "  kernel:   $(uname -r)"
echo "  memory:   $(free -h | grep Mem | awk '{print $2}')"
echo "  disk:     $(df -h / | tail -1 | awk '{print $4 " free of " $2}')"

echo ""
echo "--- Compilers ---"
for cmd in g++ cmake make; do
    if command -v $cmd &>/dev/null; then
        echo "  $cmd: $(command -v $cmd) ($($cmd --version 2>&1 | head -1))"
    else
        echo "  $cmd: MISSING"
    fi
done

echo ""
echo "--- Python ---"
for cmd in python3 pip3; do
    if command -v $cmd &>/dev/null; then
        echo "  $cmd: $(command -v $cmd) ($($cmd --version 2>&1))"
    else
        echo "  $cmd: MISSING"
    fi
done

echo ""
echo "--- Python packages ---"
python3 -c "
import importlib
for pkg in ['pybind11', 'numpy', 'yaml', 'cv2', 'PyQt5']:
    try:
        m = importlib.import_module(pkg)
        v = getattr(m, '__version__', '')
        print(f'  {pkg}: {v}')
    except ImportError:
        print(f'  {pkg}: MISSING')
" 2>/dev/null

echo ""
echo "--- Hardware libraries ---"
for lib in OpenCV OpenNI2 onnxruntime; do
    if [ "$lib" = "OpenCV" ]; then
        FOUND=$(find /usr -name "OpenCVConfig.cmake" 2>/dev/null | head -1)
        [ -n "$FOUND" ] && echo "  $lib: $(dirname $FOUND)" || echo "  $lib: MISSING"
    else
        FOUND=$(find /usr -name "lib${lib}.so*" 2>/dev/null | head -1)
        [ -n "$FOUND" ] && echo "  $lib: $FOUND" || echo "  $lib: MISSING (check ~/stroke-rehab/including/)"
    fi
done

echo ""
echo "--- Devices ---"
for dev in /dev/video0 /dev/rpmsg_ctrl0 /dev/rpmsg0 /dev/rfcomm0; do
    [ -e "$dev" ] && echo "  $dev: EXISTS" || echo "  $dev: not found"
done

echo ""
echo "--- Network ---"
echo "  hostname: $(hostname)"
ip addr show 2>/dev/null | grep "inet " | grep -v 127.0.0.1 | awk '{print "  ip: "$2}' || true

echo ""
echo "============================================"
echo " Done. If any items show MISSING, install:"
echo "   sudo apt install cmake g++ python3 python3-pip libopencv-dev"
echo "   pip3 install pybind11 numpy pyyaml opencv-python PyQt5 qfluentwidgets"
echo "============================================"