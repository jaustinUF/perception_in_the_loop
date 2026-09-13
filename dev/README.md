# dev — tools used to build and characterize the loop

These are the development and characterization scripts behind the live loop, not
part of it. Each was used to settle one question before the closed loop in the
repo root was assembled.

- **`stage5e3_joint_probe.py`** — nudges each candidate arm joint a fixed amount and measures how far it moves the gripper toward the cube, choosing the drive joint from data (joint 2) rather than by assumption.
- **`stage5e3_joint_probe.csv`** — the probe's output: per-joint, per-direction distance change, with the controls (base/wrist joints) confirming the distance metric behaves sensibly.
- **`stage5e3_image_subscriber.py`** — subscribes to the camera topic and saves frames to disk, used to confirm images cross the ROS 2 seam intact and to capture the frames the detector runs on.
- **`stage5e3_owlvit_detect.py`** — runs OWL-ViT offline on a saved frame with a text prompt, reporting each detection's score and the margin between the best true and best false box; the characterization work was done with this.

Environment launchers for each are in [`wrapper_scripts/`](wrapper_scripts/).
