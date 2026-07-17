#!/usr/bin/env bash
# Runtime environment for the independently deployed Phytium Pi tree.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUNTIME_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
OPENNI_LIB="$RUNTIME_ROOT/stroke-rehab/including/OpenNI/sdk/libs"

export PYTHONPATH="$SCRIPT_DIR/.deps:$SCRIPT_DIR/build${PYTHONPATH:+:$PYTHONPATH}"
export LD_LIBRARY_PATH="$OPENNI_LIB${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export OPENNI2_REDIST="$OPENNI_LIB"
export OPENNI2_DRIVERS_PATH="$OPENNI_LIB/OpenNI2/Drivers"

