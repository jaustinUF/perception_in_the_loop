"""
Stage 5E.3 Phase 2 - Closed vision->control loop (the ER-2 socket).

WHAT THIS IS
    The external "brain": ONE controller-side node (native Jazzy, Python 3.12)
    that closes the loop over ROS 2 --
      subscribe /camera/image_raw   (live frames, the 5E.2 subscriber's job)
      subscribe /gripper_pose       (measured hand FK, NEW 5E.3 sim publisher)
      run OWL-ViT on a frame        (the detector's job, merged in)
      publish /joint_command        (drive the arm, the 5C controller's job)

    This IS the Stage-6 architecture with OWL-ViT standing in for ER 2: an
    external process reads the camera, decides, and commands the arm over the
    proven two-process / one-DDS-graph seam. Swap OWL-ViT for ER 2 later and the
    topology is unchanged.

THE LOOP (design decisions locked with Jim)
    A1  gripper pose is MEASURED (subscribed from sim), not estimated.
    C1  PERCEIVE ONCE to GATE, then the nudge loop closes on measured DISTANCE
        (not re-detecting every step). Perception's job here is confirm+gate;
        the margin metric from 5E.3 characterization becomes the SAFETY GATE.
    Approach B  position-based joint-space nudge; NO image Jacobian, NO pixel
        servo. The Phase-1 probe picked the joint EMPIRICALLY: panda_joint2,
        +direction, ~-3.4 cm per 0.1 rad (cleanest 'toward' mover; avoids
        joint4's 5C.4 gravity/limit history).

    Sequence:
      1. wait for a live frame + a gripper pose to arrive.
      2. PERCEIVE ONCE: run OWL-ViT (prompt PROMPT), compute margin
         (best on-target score - best off-target score). If margin < MARGIN_GATE
         -> REFUSE to act, report, stop. (Trust-calibration as control gate.)
      3. if gated open: NUDGE loop, up to MAX_NUDGES times --
           read current gripper pose -> distance to cube (d_before)
           publish /joint_command: joint2 += NUDGE_RAD (name-based, only joint2)
           wait SETTLE_SEC for the arm to reach the new target
           read gripper pose again -> d_after
           if d_after >= d_before - MIN_PROGRESS: distance stopped dropping, STOP
           else continue
      4. REPORT: start distance, end distance, closed-by, nudges used.

    ONE genuinely-coupled part is the nudge loop (distance read <-> stop logic).
    Per our batching rule, it PRINTS EVERY ITERATION (nudge #, target, d_before,
    d_after, delta, decision) so a failure localizes by inspection without
    splitting the build.

WHY NOT A GRASP
    Gripper starts ~0.94 m up; cube on the floor ~0.99 m away, mostly VERTICAL.
    joint2 +0.1 rad closes ~3.4 cm, so MAX_NUDGES=10 closes ~1/3 the gap -- a
    visible APPROACH, not a grasp. Grasping needs IK + coordinated multi-joint
    reach (deliberately out of scope -- the 5C.4 'don't fight dynamics' lesson).
    The honest claim is "perception drives the arm to REDUCE gripper-to-cube
    distance over ROS 2", which IS a closed vision->control loop.

ENVIRONMENT (mirror of the 5E.2 subscriber -- see the .sh wrapper)
    Native /opt/ros/jazzy SOURCED (3.12), conda DEACTIVATED, system rclpy +
    transformers/torch (~/.local). Never imports Isaac; talks only over DDS.

RUN
    # sim side first (must be publishing image + gripper_pose + accepting cmd):
    #   ./stage5e3_camera_node_cube.sh
    ./stage5e3_closed_loop.sh                    # this node
    #   optional overrides: --prompt "..." --margin-gate 0.05 --max-nudges 10
"""

import argparse
import time

import numpy as np

import torch
from transformers import OwlViTProcessor, OwlViTForObjectDetection

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, JointState
from geometry_msgs.msg import PoseStamped

MODEL_ID = "google/owlvit-base-patch32"

# ----------------------------------------------------------------------------
# CONFIG
# ----------------------------------------------------------------------------
IMAGE_TOPIC   = "/camera/image_raw"
GRIPPER_TOPIC = "/gripper_pose"
COMMAND_TOPIC = "/joint_command"
NODE_NAME     = "closed_loop_controller"

# Cube world position (known -- we placed it). Distance is measured to this.
CUBE_XYZ = np.array([0.5, 0.0, 0.04], dtype=np.float32)

# Actuation (from the Phase-1 probe): joint2, + direction, 0.1 rad/nudge.
DRIVE_JOINT   = "panda_joint2"
NUDGE_RAD     = 0.10
MAX_NUDGES    = 10
SETTLE_SEC    = 1.0        # wall time to let the arm reach each new target
MIN_PROGRESS  = 0.002      # m: if a nudge closes less than this, distance has
                           # stopped dropping -> stop (2 mm floor)

# Perception gate (from 5E.3 characterization): the robust prompt + a margin
# gate. 'a partially red object' was the ROBUST operating point (+0.11 margin).
PROMPT        = "a partially red object"
MARGIN_GATE   = 0.05       # require best_on - best_off >= this to ACT
SCORE_FLOOR   = 0.10       # a detection must at least clear this to count
# The cube's approximate pixel center depends on framing; default assumes the
# mid/close framing. Overridable. Used to classify on- vs off-target boxes.
# TARGET_XY     = (577.0, 500.0)
TARGET_XY     = (640.0, 567.0)
TARGET_TOL    = 80.0


class ClosedLoopController(Node):
    def __init__(self, args):
        super().__init__(NODE_NAME)
        self.args = args
        self.latest_frame = None      # (H,W,3) uint8
        self.latest_gripper = None    # np.array([x,y,z])
        self.cmd_pub = self.create_publisher(JointState, COMMAND_TOPIC, 10)
        self.create_subscription(Image, IMAGE_TOPIC, self._on_image, 1)
        self.create_subscription(PoseStamped, GRIPPER_TOPIC, self._on_gripper, 1)
        self.get_logger().info(
            f"controller up: sub {IMAGE_TOPIC} + {GRIPPER_TOPIC}, "
            f"pub {COMMAND_TOPIC}"
        )

    def _on_image(self, msg: Image):
        if msg.encoding != "rgb8" or msg.step != msg.width * 3:
            return
        self.latest_frame = np.frombuffer(msg.data, np.uint8).reshape(
            msg.height, msg.width, 3
        ).copy()

    def _on_gripper(self, msg: PoseStamped):
        p = msg.pose.position
        self.latest_gripper = np.array([p.x, p.y, p.z], dtype=np.float32)

    def gripper_distance(self):
        if self.latest_gripper is None:
            return None
        return float(np.linalg.norm(self.latest_gripper - CUBE_XYZ))

    def send_joint2(self, target_value):
        """Publish a name-based /joint_command addressing ONLY panda_joint2.
        Joints not named hold their current target (5C.3 name-based mapping)."""
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name = [DRIVE_JOINT]
        msg.position = [float(target_value)]
        self.cmd_pub.publish(msg)


def wait_for_inputs(node, timeout_sec=30.0):
    """Spin until both a frame and a gripper pose have arrived (or timeout)."""
    t0 = time.time()
    while rclpy.ok() and (node.latest_frame is None or node.latest_gripper is None):
        rclpy.spin_once(node, timeout_sec=0.1)
        if time.time() - t0 > timeout_sec:
            return False
    return True


def perceive_and_gate(node, processor, model, device):
    """Run OWL-ViT once on the latest frame; return (act:bool, info:dict)."""
    from PIL import Image as PILImage
    frame = node.latest_frame
    H, W = frame.shape[:2]
    image = PILImage.fromarray(frame)

    inputs = processor(text=[[node.args.prompt]], images=image,
                       return_tensors="pt").to(device)
    t0 = time.perf_counter()
    with torch.no_grad():
        outputs = model(**inputs)
    if device == "cuda":
        torch.cuda.synchronize()
    infer_ms = (time.perf_counter() - t0) * 1000.0

    target_sizes = torch.tensor([[H, W]], device=device)
    res = processor.post_process_grounded_object_detection(
        outputs=outputs, target_sizes=target_sizes, threshold=0.0,
        text_labels=[[node.args.prompt]],
    )[0]
    boxes = res["boxes"].cpu().numpy()
    scores = res["scores"].cpu().numpy()
    order = np.argsort(-scores)
    boxes, scores = boxes[order], scores[order]

    tx, ty = node.args.target_xy
    best_on = best_off = None
    for i in range(len(scores)):
        cx = 0.5 * (boxes[i][0] + boxes[i][2])
        cy = 0.5 * (boxes[i][1] + boxes[i][3])
        on = ((cx - tx) ** 2 + (cy - ty) ** 2) ** 0.5 <= node.args.target_tol
        if on and best_on is None:
            best_on = scores[i]
        elif (not on) and best_off is None:
            best_off = scores[i]
        if best_on is not None and best_off is not None:
            break

    on_v = float(best_on) if best_on is not None else 0.0
    off_v = float(best_off) if best_off is not None else 0.0
    margin = on_v - off_v

    print(f"[perceive] prompt '{node.args.prompt}'  infer {infer_ms:.0f} ms")
    print(f"[perceive] best on-target {on_v:.4f}  best off-target {off_v:.4f}  "
          f"margin {margin:+.4f}")

    act = (on_v >= node.args.score_floor) and (margin >= node.args.margin_gate)
    if not act:
        reason = ("on-target score below floor" if on_v < node.args.score_floor
                  else "margin below gate")
        print(f"[gate] REFUSE to act: {reason} "
              f"(need score>={node.args.score_floor}, "
              f"margin>={node.args.margin_gate})")
    else:
        print(f"[gate] OPEN: cube confirmed, margin sufficient -- proceeding.")
    return act, {"on": on_v, "off": off_v, "margin": margin}


def nudge_loop(node, home_joint2):
    """Drive joint2 + up to MAX_NUDGES, closing on measured distance.
    Prints every iteration (the coupled part -- per our batching rule)."""
    d_start = node.gripper_distance()
    print(f"\n[loop] start gripper-to-cube distance {d_start:.4f} m")
    target = home_joint2
    used = 0
    for i in range(node.args.max_nudges):
        d_before = node.gripper_distance()
        target = target + NUDGE_RAD
        node.send_joint2(target)

        # Wait for the arm to reach the new target, servicing callbacks so the
        # gripper-pose subscription refreshes.
        t0 = time.time()
        while time.time() - t0 < SETTLE_SEC:
            rclpy.spin_once(node, timeout_sec=0.05)

        d_after = node.gripper_distance()
        delta = d_after - d_before
        used = i + 1
        progressed = (d_before - d_after) >= MIN_PROGRESS
        decision = "continue" if progressed else "STOP (progress stalled)"
        print(f"  nudge {used:2d}  joint2->{target:+.3f}  "
              f"d {d_before:.4f} -> {d_after:.4f} m  "
              f"delta {delta:+.4f} m  [{decision}]")
        if not progressed:
            break

    d_end = node.gripper_distance()
    print(f"\n[report] nudges used {used}/{node.args.max_nudges}")
    print(f"[report] distance {d_start:.4f} m -> {d_end:.4f} m  "
          f"(closed by {d_start - d_end:.4f} m = {(d_start - d_end)*100:.1f} cm)")


def main():
    ap = argparse.ArgumentParser(description="Closed vision->control loop.")
    ap.add_argument("--prompt", default=PROMPT)
    ap.add_argument("--margin-gate", dest="margin_gate", type=float, default=MARGIN_GATE)
    ap.add_argument("--score-floor", dest="score_floor", type=float, default=SCORE_FLOOR)
    ap.add_argument("--max-nudges", dest="max_nudges", type=int, default=MAX_NUDGES)
    ap.add_argument("--target-xy", dest="target_xy", default=None, help="'x,y' cube pixel center (default from CONFIG)")
    ap.add_argument("--target-tol", dest="target_tol", type=float, default=TARGET_TOL)
    args = ap.parse_args()
    if args.target_xy:
        args.target_xy = tuple(float(v) for v in args.target_xy.split(","))
    else:
        args.target_xy = TARGET_XY

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[info] device: {device}")
    print(f"[info] loading OWL-ViT ({MODEL_ID})...")
    processor = OwlViTProcessor.from_pretrained(MODEL_ID)
    model = OwlViTForObjectDetection.from_pretrained(MODEL_ID).to(device)
    model.eval()

    rclpy.init()
    node = ClosedLoopController(args)

    print("[info] waiting for a live frame + gripper pose...")
    if not wait_for_inputs(node):
        print("[error] timed out waiting for /camera/image_raw + /gripper_pose. "
              "Is the sim node running and publishing both?")
        node.destroy_node(); rclpy.shutdown(); return

    # Record joint2's home value: we need the ABSOLUTE target to command, and the
    # sim holds joints at their last target. We don't subscribe /joint_states
    # here (kept minimal), so we command RELATIVE to a known home. The sim's home
    # joint2 was ~-0.0528 rad (from the Phase-1 probe). We command absolute
    # targets = home + k*NUDGE. Read it from CONFIG-consistent probe value.
    home_joint2 = -0.0528   # panda_joint2 home (Phase-1 probe reading)

    act, info = perceive_and_gate(node, processor, model, device)
    if act:
        nudge_loop(node, home_joint2)
    else:
        print("\n[report] no motion -- perception gate closed. "
              "Loop refused to act on an untrusted detection.")

    print("\n[done] closed-loop pass complete.")
    node.destroy_node()
    if rclpy.ok():
        rclpy.shutdown()


if __name__ == "__main__":
    main()
