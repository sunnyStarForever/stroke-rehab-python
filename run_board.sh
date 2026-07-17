#!/usr/bin/env bash
# Launch the Python-main application on the target board.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/board_env.sh"
cd "$SCRIPT_DIR"
PYTHON_BIN="${STROKE_REHAB_PYTHON:-/usr/bin/python3}"
if [[ ! -x "$PYTHON_BIN" ]]; then
    echo "Python interpreter not found: $PYTHON_BIN" >&2
    exit 1
fi
exec "$PYTHON_BIN" main.py "$@"
