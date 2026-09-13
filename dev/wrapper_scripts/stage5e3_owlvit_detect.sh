#!/usr/bin/env bash
# Stage 5E.3 - controller-side wrapper for the OWL-ViT off-loop detector.
#
# SAME environment as stage5e2_image_subscriber.sh -- native /opt/ros/jazzy
# sourced (system Python 3.12), conda fully DEACTIVATED. This is where
# transformers/torch/OWL-ViT were installed (~/.local/lib/python3.12/), so this
# is the interpreter that must run the detector. (The detector does NOT import
# rclpy or ROS -- but we reuse this wrapper so the interpreter is guaranteed to
# be the one the packages live in, and the proof-print confirms it.)
#
# The conda-shadowing fix (from 5C.3): a script run non-interactively can't call
# the `conda` shell function unless we source conda.sh first. Without this,
# python3 may resolve to conda base's 3.13 and OWL-ViT "isn't found".
#
# All CLI args are passed straight through to the script, so prompt/threshold/
# frame vary without editing anything:
#   ./stage5e3_owlvit_detect.sh stage5e2_first_frame.png
#   ./stage5e3_owlvit_detect.sh frame.png --prompt "a red block" --threshold 0.05
set -e

# Source native Jazzy (system 3.12). Not strictly needed by the detector (no
# ROS import), but keeps this wrapper a true mirror of the subscriber's and
# guarantees the same PATH/interpreter resolution.
source /opt/ros/jazzy/setup.bash

# Kill conda shadowing: load the conda function, then deactivate every stacked
# env so python3 -> /usr/bin/python3 (3.12), NOT conda base (3.13).
if [ -f "$(conda info --base 2>/dev/null)/etc/profile.d/conda.sh" ]; then
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda deactivate 2>/dev/null || true
    conda deactivate 2>/dev/null || true
else
    # Fallback: strip conda dirs from PATH directly if conda.sh isn't found.
    export PATH=$(echo "$PATH" | tr ':' '\n' | grep -v miniconda | grep -v anaconda | paste -sd ':')
fi

# Proof-of-interpreter, same discipline as the subscriber wrapper.
echo "[wrapper] python3 -> $(which python3)"
echo "[wrapper] version: $(python3 --version 2>&1)"

cd "$(dirname "$0")"

# Pass ALL args through to the detector (frame, --prompt, --threshold, --outdir).
python3 stage5e3_owlvit_detect.py "$@"
