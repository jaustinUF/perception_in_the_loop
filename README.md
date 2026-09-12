# perception_in_the_loop

A closed vision→control loop in NVIDIA Isaac Sim: an external OWL-ViT detector,
running over ROS 2, drives a Franka arm toward a target it perceives — and
refuses to act when the detection isn't trustworthy.

## What this does

A Franka Panda arm in Isaac Sim publishes camera frames and its measured gripper
pose over ROS 2. A separate external process — its own environment, talking only
over DDS topics — runs an OWL-ViT open-vocabulary detector on those frames. When
it confidently locates a red cube on the floor, it commands the arm over ROS 2 to
reduce the gripper-to-cube distance; in a successful run it closed 44 cm. A
margin-based confidence gate makes the controller **refuse to act** when the
detection is ambiguous, so the arm only moves on a detection it can trust.

The detector's behavior was characterized before the loop was closed (prompt
sensitivity, a non-monotonic confidence-vs-target-size effect, ~230 ms
per-detection latency), so the trust threshold rests on measured data rather than
assumption. See [`characterization/`](characterization/).

## The gate in action

Same system, same code, same prompt — two framings, two decisions:

| Run | Framing | Cube score | Arm score | Margin | Gate | Result |
|-----|---------|-----------|-----------|--------|------|--------|
| Success | mid (~40 px cube) | 0.312 | 0.202 | **+0.110** | OPEN | arm drives, 44 cm closed |
| Refusal | close (~90 px cube) | 0.223 | 0.273 | **−0.050** | REFUSE | no motion |

At the close framing the detector genuinely ranks the arm above the cube, so the
gate declines — protecting against a real, measured failure mode, not a
hypothetical one. Full frames, annotated detections, and console traces for both:
[`demos/`](demos/).

## Architecture

Two processes, two Python environments, one DDS graph between them — the seam
sits where it would on real hardware. The sim side (Isaac Sim + its bundled ROS 2
libraries) publishes `/camera/image_raw`, `/gripper_pose`, and `/joint_states`,
and subscribes `/joint_command`. The controller side (native ROS 2, system
Python) runs the detector and closes the loop — it never imports Isaac.

Live loop:
- **`stage5e3_camera_node_cube.py`** — the sim-side plant: arm, camera, red cube; publishes frames + gripper pose, applies joint commands.
- **`stage5e3_closed_loop.py`** — the external controller: perceives, gates on margin, drives the arm, reports.

`dev/` holds the tools used to build and characterize the loop (the joint-selection probe, the frame subscriber, the offline detector). `wrapper_scripts/` holds the environment launchers for each side.

## Context

Stage 5E of a self-directed robotics simulation & control track. It reuses the
two-process ROS 2 seam proven in an earlier joint-control stage ([isaac-ros2-external-control](https://github.com/jaustinUF/isaac-ros2-external-control)), carrying a
richer payload (camera images and a learned detector's output) across the same
architecture — the substrate a vision-language reasoning model would later plug
into.