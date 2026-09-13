#!/usr/bin/env bash
# External-side wrapper for the 5E.2 image subscriber. MIRROR of the 5C.3
# controller-side wrapper (stage5c3_trivial_controller.sh): native
# /opt/ros/jazzy SOURCED (system Python 3.12), conda fully DEACTIVATED. This is
# the OPPOSITE of the sim-side wrapper (which activates conda + uses the bundle).
# The wrapper contrast IS the two-process architecture.
set -e

# --- Kill conda shadowing (the 5C.3 gotcha) --------------------------------
# conda base auto-activates in every terminal, prepending its Python to PATH.
# `conda deactivate` is a shell FUNCTION from conda's init; a non-interactive
# `bash foo.sh` never sources that init, so the function doesn't exist and a
# bare `conda deactivate` silently no-ops -- while the parent's shadowed PATH is
# still inherited. Fix: source conda.sh to LOAD the function, then deactivate
# every stacked env. Fallback: strip conda dirs from PATH directly.
CONDA_BASE="$(conda info --base 2>/dev/null || echo "$HOME/miniconda3")"
if [ -f "$CONDA_BASE/etc/profile.d/conda.sh" ]; then
    source "$CONDA_BASE/etc/profile.d/conda.sh"
    while [ -n "${CONDA_PREFIX:-}" ]; do
        conda deactivate
    done
else
    # Fallback: remove conda bin dirs from PATH so system python3 wins.
    export PATH="$(echo "$PATH" | tr ':' '\n' | grep -v "$CONDA_BASE" | paste -sd ':' -)"
    unset CONDA_PREFIX
fi

# --- Source native ROS 2 Jazzy (system Python 3.12) ------------------------
source /opt/ros/jazzy/setup.bash
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp   # match the sim side's RMW

# --- Prove the interpreter is the system one BEFORE launching --------------
echo "[wrapper] python3 -> $(command -v python3)"
echo "[wrapper] version: $(python3 --version 2>&1)"
echo "[wrapper] ROS_DISTRO=$ROS_DISTRO RMW=$RMW_IMPLEMENTATION"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
python3 "$SCRIPT_DIR/stage5e3_image_subscriber.py" "$@"
