#!/bin/bash
# ==========================================================================
# Deploy stroke-rehab to target board (AArch64 Linux).
# Run from WINDOWS Git Bash, in the stroke-rehab/ root directory.
#
# Usage:
#   cd e:/ResearchWork/jichuangsai/stroke-rehab
#   bash python_version/deploy.sh 10.161.95.152 [root]
# ==========================================================================
set -euo pipefail

BOARD_IP="${1:?Usage: $0 <board-ip> [user]}"
BOARD_USER="${2:-root}"
BOARD_HOST="${BOARD_USER}@${BOARD_IP}"
BOARD_HOME="/home/${BOARD_USER}"
BOARD_DIR="${BOARD_HOME}/stroke-rehab"

echo "============================================"
echo " Deploying stroke-rehab to ${BOARD_HOST}"
echo " Target directory: ${BOARD_DIR}"
echo "============================================"

# ---- Step 1: Create directory structure on board ----
echo ""
echo "[1/5] Creating directories on target board..."
ssh "${BOARD_HOST}" "
    mkdir -p ${BOARD_DIR}/{python_version/build,including,configs,tools/scoring_engine,records}
" && echo "  Done."

# ---- Step 2: Transfer including/ (ONNX Runtime + models + OpenNI SDK) ----
echo ""
echo "[2/5] Transferring including/ (~48MB, this may take a minute)..."
rsync -avz --progress \
    --exclude='__pycache__' \
    --exclude='.git' \
    including/ \
    "${BOARD_HOST}:${BOARD_DIR}/including/"
echo "  Done."

# ---- Step 3: Transfer python_version/ (engine code + UI) ----
echo ""
echo "[3/5] Transferring python_version/ (source code)..."
rsync -avz --progress \
    --exclude='build/' \
    --exclude='__pycache__/' \
    --exclude='*.pyd' \
    --exclude='*.pyc' \
    python_version/ \
    "${BOARD_HOST}:${BOARD_DIR}/python_version/"
echo "  Done."

# ---- Step 4: Transfer configs/ ----
echo ""
echo "[4/5] Transferring configs/..."
rsync -avz --progress \
    configs/ \
    "${BOARD_HOST}:${BOARD_DIR}/configs/"
echo "  Done."

# ---- Step 5: Transfer tools/scoring_engine/ (Python scripts only, no data/) ----
echo ""
echo "[5/5] Transferring scoring engine (excluding large data/)..."
rsync -avz --progress \
    --exclude='data/' \
    --exclude='outputs/' \
    --exclude='__pycache__/' \
    tools/scoring_engine/ \
    "${BOARD_HOST}:${BOARD_DIR}/tools/scoring_engine/"
echo "  Done."

echo ""
echo "============================================"
echo " Deploy complete!"
echo ""
echo " Next steps on the target board:"
echo "   ssh ${BOARD_HOST}"
echo "   cd ${BOARD_DIR}/python_version"
echo "   bash setup_board.sh"
echo "============================================"