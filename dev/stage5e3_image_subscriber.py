"""
Stage 5E.3 - Image subscriber : the external side of the perception seam.

WHAT THIS IS
    A fresh, MINIMAL external-process node (NOT a copy of the trivial controller
    -- it carried JointState plumbing we don't need here). One job: prove a raw
    frame crossed the proven 5C seam INTACT.

      - subscribe sensor_msgs/Image on /camera/image_raw
      - reshape the raw rgb8 bytes back into an (H,W,3) uint8 array
      - save the FIRST good frame to disk as a PNG (self-describing file)
      - print its shape / encoding / byte count as the sanity read
      - keep spinning (so you can watch the arrival rate); Ctrl+C to stop

    This is the MIRROR of the sim-side publisher's tobytes(): we do the inverse,
    turning the flat wire bytes back into a shaped array. No cv_bridge -- we
    KNOW this frame is rgb8, contiguous, unpadded (step == width*3), because the
    5E.2 sim node constructed it that way. Manual reshape is faithful to those
    known-clean bytes; cv_bridge is for consuming image topics of UNKNOWN
    provenance/encoding (a later stage), where its convention-adapting earns its
    dependency. Here it would only add a BGR-confusion risk and buy nothing.

WHY DECODE INSTEAD OF DUMPING msg.data
    msg.data is RAW PIXELS only -- R,G,B,R,G,B,... with no header saying "I am
    1280x720 rgb8". That shape lives in SEPARATE message fields (height, width,
    encoding, step). A file on disk must be self-describing, so we reunite the
    pixels with their shape (reshape) and let PIL wrap them in a PNG container
    (header + data) that opens as an image later, with no ROS message around.

ENVIRONMENT (mirror of the 5C.3 controller side -- see the .sh wrapper)
    Native /opt/ros/jazzy SOURCED (Python 3.12), conda fully DEACTIVATED (the
    conda-shadowing gotcha from 5C.3), system rclpy. The OPPOSITE of the sim-side
    wrapper. This node never imports Isaac; it talks only over DDS topics.

QoS
    The sim-side publisher is RELIABLE / VOLATILE (confirmed via
    `ros2 topic info /camera/image_raw --verbose`). We subscribe with the DEFAULT
    RELIABLE profile -> compatible (matches the 5C.3 choice that worked). If
    "no frames arriving", the FIRST thing to check is a QoS RxO mismatch:
    `ros2 topic info /camera/image_raw --verbose` and compare endpoints.

RUN
    # sim side first (must be publishing):  ./stage5e2_camera_node.sh
    ./stage5e2_image_subscriber.sh          # this node (native Jazzy)
    # -> writes stage5e2_first_frame.png next to this script; eyeball it:
    #    right shape, right colors, shows the robot, NOT grey/empty.
"""

import os
import argparse

import numpy as np
from PIL import Image as PILImage

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image

# ----------------------------------------------------------------------------
# CONFIG
# ----------------------------------------------------------------------------
IMAGE_TOPIC = "/camera/image_raw"
NODE_NAME   = "image_subscriber"
# Save next to this script, so the output lands wherever the pair lives.
OUT_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "demo_success_frame.png"
)


class ImageSubscriber(Node):
    """Subscribe /camera/image_raw; save N good frames, then request shutdown."""

    def __init__(self, n_save=1, skip=0):
        super().__init__(NODE_NAME)
        # Default QoS (RELIABLE) -- compatible with the RELIABLE sim publisher.
        self.create_subscription(Image, IMAGE_TOPIC, self._on_image, 10)
        self._n_received = 0
        self._n_saved = 0
        self._n_save = n_save        # how many good frames to save before exiting
        self._skip = skip            # discard this many good frames first (warm-up)
        self._done = False           # set True once we've saved enough
        self.get_logger().info(f"subscriber up: listening on {IMAGE_TOPIC}")
        self.get_logger().info(
            f"will skip {skip} frame(s), then save {n_save}, then exit."
        )
        self.get_logger().info("waiting for the first frame...")

    def _on_image(self, msg: Image):
        if self._done:
            return
        self._n_received += 1

        # --- Guard the assumptions BEFORE reshaping (verify, don't trust). ---
        if msg.encoding != "rgb8":
            self.get_logger().warn(
                f"unexpected encoding '{msg.encoding}' (expected 'rgb8') -- "
                f"manual reshape assumes 3x8-bit; skipping."
            )
            return
        expected_step = msg.width * 3
        if msg.step != expected_step:
            self.get_logger().warn(
                f"row padding detected: step={msg.step} != width*3="
                f"{expected_step}. Plain reshape unsafe; skipping. "
                f"(This is exactly the case cv_bridge handles for you.)"
            )
            return

        # --- Reunite pixels with shape: flat bytes -> (H, W, 3) uint8. ---
        arr = np.frombuffer(msg.data, dtype=np.uint8).reshape(
            msg.height, msg.width, 3
        )

        # --- Skip warm-up frames first (RTX lighting convergence, 5E.2). ---
        # Even with the sim node's 60-step warm-up, allow discarding a few more
        # here so the SAVED frame is guaranteed converged. Default skip=0.
        if self._n_received <= self._skip:
            self.get_logger().info(
                f"...skipping warm-up frame {self._n_received}/{self._skip} "
                f"(mean brightness {float(arr.mean()):.1f})"
            )
            return

        if self._n_received == self._skip + 1:
            self.get_logger().info(
                f"FIRST kept frame: {msg.width}x{msg.height} "
                f"encoding='{msg.encoding}' step={msg.step} "
                f"{len(msg.data)} bytes -> array shape {arr.shape} "
                f"dtype {arr.dtype}"
            )

        # --- Save this frame. First good frame -> OUT_PATH; if saving several,
        #     subsequent ones get an index so they don't overwrite. ---
        if self._n_saved == 0:
            out = OUT_PATH
        else:
            base, ext = os.path.splitext(OUT_PATH)
            out = f"{base}_{self._n_saved:02d}{ext}"
        PILImage.fromarray(arr.copy()).save(out)
        self._n_saved += 1
        self.get_logger().info(
            f"saved frame {self._n_saved}/{self._n_save} -> {out} "
            f"(mean brightness {float(arr.mean()):.1f})"
        )

        # --- Done? Request shutdown so the node exits on its own (no Ctrl+C). ---
        if self._n_saved >= self._n_save:
            self._done = True
            self.get_logger().info(
                "saved requested frame(s); shutting down cleanly. "
                "Eyeball the PNG: right shape, colors, cube visible, not grey."
            )
            # rclpy.shutdown() from inside a callback cleanly breaks spin().
            rclpy.shutdown()


def main():
    parser = argparse.ArgumentParser(
        description="Save N good camera frames from /camera/image_raw, then exit."
    )
    parser.add_argument(
        "--frames", type=int, default=1,
        help="number of good frames to save before exiting (default: 1)",
    )
    parser.add_argument(
        "--skip", type=int, default=0,
        help="discard this many good frames first, as extra RTX warm-up "
             "insurance (default: 0)",
    )
    args = parser.parse_args()

    rclpy.init()
    node = ImageSubscriber(n_save=args.frames, skip=args.skip)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.get_logger().info(
            f"received {node._n_received} frames, saved {node._n_saved}."
        )
        node.destroy_node()
        # Guard: shutdown() may already have been called from the callback.
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
