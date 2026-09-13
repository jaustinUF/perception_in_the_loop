#!/usr/bin/env bash
# Stage 5E.3 - sim-side wrapper for the camera node + red cube.
#
# IDENTICAL environment to the proven stage5e2_camera_node.sh -- only the python
# script name changes (stage5e2_camera_node_cube.py). This is the "Option 1"
# sim-side environment settled in 5C.2 and reconfirmed every stage since:
#
#   - conda env_isaaclab ACTIVATED (Isaac Sim's bundled Python 3.11)
#   - native /opt/ros/jazzy DELIBERATELY NOT sourced (its 3.12 rclpy is ABI-
#     incompatible with Isaac's 3.11 interpreter)
#   - the bridge's BUNDLED rclpy is used instead; its lib/ dir is put on
#     LD_LIBRARY_PATH so the bundle libraries resolve EACH OTHER
#   - RMW = FastDDS (matches the native-side subscriber, so they discover)
#
# Boot log confirms the isolation every run:
#   "Could not import system rclpy" -> "internal rclpy for jazzy" -> "rclpy loaded"
# That "failure" line is the isolation working as designed.
set -e

BRIDGE_LIB="$HOME/miniconda3/envs/env_isaaclab/lib/python3.11/site-packages/isaacsim/exts/isaacsim.ros2.bridge/jazzy/lib"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate env_isaaclab

export ROS_DISTRO=jazzy
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:$BRIDGE_LIB"

# Run from the directory this wrapper lives in, so the script's __file__-relative
# paths behave the same as the proven pair.
cd "$(dirname "$0")"

python stage5e3_camera_node_cube.py
