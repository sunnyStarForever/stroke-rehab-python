#!/usr/bin/env bash
# Deploy the Python-main runtime from this workspace to an AArch64 Linux board.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
BOARD_IP="${1:?Usage: $0 <board-ip> [user] [remote-dir]}"
BOARD_USER="${2:-root}"
if [[ -n "${3:-}" ]]; then
  REMOTE_DIR="$3"
elif [[ "$BOARD_USER" == "root" ]]; then
  REMOTE_DIR="/root/stroke-rehab-runtime"
else
  REMOTE_DIR="/home/$BOARD_USER/stroke-rehab-runtime"
fi
BOARD_HOST="${BOARD_USER}@${BOARD_IP}"

ssh "$BOARD_HOST" "mkdir -p '$REMOTE_DIR/python_version' '$REMOTE_DIR/stroke-rehab/including' '$REMOTE_DIR/stroke-rehab/tools/scoring_engine'"
rsync -az --progress --delete \
  --exclude='.venv/' --exclude='build/' --exclude='recordings/' \
  --exclude='__pycache__/' --exclude='*.pyc' --exclude='*.pyd' --exclude='_core*.so' \
  "$WORKSPACE_ROOT/python_version/" "$BOARD_HOST:$REMOTE_DIR/python_version/"
rsync -az --progress "$WORKSPACE_ROOT/stroke-rehab/including/" \
  "$BOARD_HOST:$REMOTE_DIR/stroke-rehab/including/"
rsync -az --progress --delete --exclude='outputs/' --exclude='__pycache__/' --exclude='*.pyc' \
  "$WORKSPACE_ROOT/stroke-rehab/tools/scoring_engine/" \
  "$BOARD_HOST:$REMOTE_DIR/stroke-rehab/tools/scoring_engine/"

echo "Deploy complete. On the board: cd $REMOTE_DIR/python_version && bash setup_board.sh"
