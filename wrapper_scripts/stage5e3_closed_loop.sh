#!/usr/bin/env bash
# Stage 5E.3 Phase-2 - controller-side wrapper for the closed vision->control loop.
#
# SAME environment as stage5e3_image_subscriber.sh / stage5e3_owlvit_detect.sh:
# native /opt/ros/jazzy SOURCED (system Python 3.12), conda fully DEACTIVATED, so
# python3 -> /usr/bin/python3 with rclpy (system) + transformers/torch (~/.local).
# This node imports BOTH rclpy AND OWL-ViT -- it is the merged subscriber +
# detector + commander. Never imports Isaac; talks only over DDS topics.
set -e

# --- Kill conda shadowing (the 5C.3 gotcha) --------------------------------
CONDA_BASE="$(conda info --base 2>/dev/null || echo "$HOME/miniconda3")"
if [ -f "$CONDA_BASE/etc/profile.d/conda.sh" ]; then
    source "$CONDA_BASE/etc/profile.d/conda.sh"
    while [ -n "${CONDA_PREFIX:-}" ]; do
        conda deactivate
    done
else
    export PATH="$(echo "$PATH" | tr ':' '\n' | grep -v "$CONDA_BASE" | paste -sd ':' -)"
    unset CONDA_PREFIX
fi

# --- Source native ROS 2 Jazzy (system Python 3.12) ------------------------
source /opt/ros/jazzy/setup.bash
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp   # match the sim side's RMW

echo "[wrapper] python3 -> $(command -v python3)"
echo "[wrapper] version: $(python3 --version 2>&1)"
echo "[wrapper] ROS_DISTRO=$ROS_DISTRO RMW=$RMW_IMPLEMENTATION"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Pass all args through (--prompt, --margin-gate, --max-nudges, --target-xy, ...).
python3 "$SCRIPT_DIR/stage5e3_closed_loop.py" "$@"
