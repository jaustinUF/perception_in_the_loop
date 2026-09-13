#!/usr/bin/env bash
# Stage 5E.3 Phase-1 - sim-side wrapper for the OFFLINE joint probe.
#
# Simpler than the camera-node wrapper: this probe imports NO rclpy / ROS (no
# camera, no publish, no seam), so the bridge bundle LD_LIBRARY_PATH dance is
# NOT needed. We only need Isaac's own conda env + the local asset root (set in
# the script). If you ever add ROS to this probe, copy the camera-node wrapper
# instead.
set -e

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate env_isaaclab

cd "$(dirname "$0")"

python stage5e3_joint_probe.py
