"""
Stage 5E.3 - Phase 1 : joint actuation probe (OFFLINE, no ROS, no camera).

WHAT THIS IS
    The characterization step BEFORE the closed loop. It answers, empirically:
    "if I nudge joint N by a small amount, does the gripper (panda_hand) move
    TOWARD or AWAY from the cube, and by how much?" -- so the Phase-2 loop can
    pick the right joint and direction from DATA, not from kinematic reasoning.

    This is the "plant is the oracle" design Jim chose: we don't compute the
    arm's kinematics analytically; we nudge a joint and READ the measured
    gripper-to-cube distance change. The sim is the model we probe.

WHY OFFLINE / NO ROS / NO CAMERA
    One variable at a time. Phase 1 changes ONLY the actuation question. No
    perception, no ROS seam, no image -- just load the arm + cube, nudge joints,
    measure. If this misbehaves, it's the arm/joint, not the transport or the
    detector. Phase 2 (the closed loop) adds perception and the ROS seam back.

METHOD (per candidate joint, per direction +/-):
    1. world.reset() -> known home pose.
    2. read panda_hand world position -> distance to cube = d_before.
    3. nudge: set that joint's position target to home +/- NUDGE_RAD; step the
       sim ~SETTLE steps so the internal position PD reaches the new target.
    4. read panda_hand again -> d_after.
    5. record joint, direction, d_before, d_after, delta = d_after - d_before.
    NEGATIVE delta = gripper moved TOWARD the cube (what we want).

JOINTS TESTED
    Candidates (expected to reduce distance): panda_joint2, panda_joint4
      -- these swing the arm in the vertical plane toward a floor cube in front.
    Controls (validate the distance metric itself):
      panda_joint1 (base yaw)  -- head-on cube is on the centerline, so rotating
        the base should swing the gripper SIDEWAYS -> distance flat or UP, never
        a good "approach". If joint1 REDUCED head-on distance, distrust the metric.
      panda_joint6 (wrist)     -- reorients the end-effector; ambiguous, tests
        whether small wrist motion is a confound.
    (Same arithmetic-ground-truth discipline as 5B: a control that SHOULDN'T
    help validates that the measurement behaves sensibly.)

GRIPPER POSE HANDLE (grep-verified, Isaac Sim 5.1)
    panda_hand world pose is read via a RigidPrim view on /World/Franka/panda_hand
    -- NOT Articulation.get_world_poses(), which returns the articulation ROOT
    (base) pose, not a per-link pose (verified: that method calls
    get_root_transforms()). RigidPrim + single get_world_pose() returns
    position (3,) directly, scalar-first quat (4,). Must be scene-added and the
    sim reset before the physics handle is valid.

OUTPUT
    A CSV (stage5e3_joint_probe.csv) + a printed table. Acquisition/analysis
    split: the CSV is the artifact; the printed table is the quick read.

RUN (sim-side wrapper: conda env_isaaclab, bundled rclpy NOT needed here since
     no ROS -- but the wrapper's conda activation + asset root still apply)
    ./stage5e3_joint_probe.sh
"""

# ----------------------------------------------------------------------------
# 1. LAUNCH THE APP FIRST (headless -- no viewport needed for a numeric probe).
# ----------------------------------------------------------------------------
from isaacsim import SimulationApp

simulation_app = SimulationApp({"headless": True})

import os
import csv
import numpy as np
import carb.settings

LOCAL_ASSET_ROOT = os.path.expanduser("~/isaacsim_assets/Assets/Isaac/5.1")
carb.settings.get_settings().set(
    "/persistent/isaac/asset_root/default", LOCAL_ASSET_ROOT
)

from isaacsim.core.api import World
from isaacsim.core.prims import Articulation, RigidPrim
from isaacsim.core.utils.stage import add_reference_to_stage
from isaacsim.core.api.objects.cuboid import VisualCuboid

# ----------------------------------------------------------------------------
# CONFIG
# ----------------------------------------------------------------------------
PHYSICS_DT = 1.0 / 60.0
SETTLE     = 90                     # steps to let the position PD reach target
NUDGE_RAD  = 0.10                   # +/- nudge per joint (~5.7 deg)

FRANKA_PRIM_PATH = "/World/Franka"
FRANKA_USD = os.path.join(
    LOCAL_ASSET_ROOT, "Isaac/Robots/FrankaRobotics/FrankaPanda/franka.usd"
)
HAND_PRIM_PATH = "/World/Franka/panda_hand"   # end-effector link (grep-verified)

CUBE_PRIM_PATH = "/World/red_cube"
CUBE_SIZE      = 0.08
CUBE_POSITION  = np.array([0.5, 0.0, CUBE_SIZE / 2.0], dtype=np.float32)  # (0.5,0,0.04)

# Joints to test: candidates first, then controls.
TEST_JOINTS = ["panda_joint2", "panda_joint4", "panda_joint1", "panda_joint6"]
ROLE = {
    "panda_joint2": "candidate",
    "panda_joint4": "candidate",
    "panda_joint1": "control (base yaw -- should NOT reduce head-on distance)",
    "panda_joint6": "control (wrist -- ambiguous)",
}

CSV_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "stage5e3_joint_probe.csv")


def hand_distance_to_cube(hand_view):
    """Read panda_hand world position and return Euclidean distance to cube.

    RigidPrim (a multi-prim view) exposes the PLURAL get_world_poses() ->
    positions (M,3), orientations (M,4). Our view holds one prim, so take row 0.
    (The singular get_world_pose() lives on the single-prim wrapper, a different
    class -- grep mismatch, corrected here.)
    """
    positions, _ = hand_view.get_world_poses()   # (1,3), (1,4) scalar-first
    pos = np.asarray(positions, dtype=np.float32).reshape(-1)[:3]
    return float(np.linalg.norm(pos - CUBE_POSITION)), pos

def main():
    if not os.path.isfile(FRANKA_USD):
        raise FileNotFoundError(f"Franka USD not found: {FRANKA_USD}")

    world = World(physics_dt=PHYSICS_DT, rendering_dt=PHYSICS_DT,
                  stage_units_in_meters=1.0)
    world.scene.add_default_ground_plane()

    add_reference_to_stage(usd_path=FRANKA_USD, prim_path=FRANKA_PRIM_PATH)
    franka = Articulation(prim_paths_expr=FRANKA_PRIM_PATH, name="franka")
    world.scene.add(franka)

    # Pure-visual cube -- no physics, just a fixed world position to measure to.
    red_cube = VisualCuboid(
        prim_path=CUBE_PRIM_PATH, name="red_cube",
        position=CUBE_POSITION, size=CUBE_SIZE,
        color=np.array([1.0, 0.0, 0.0], dtype=np.float32),
    )
    world.scene.add(red_cube)

    # RigidPrim view on the HAND link -- the correct per-link pose handle
    # (Articulation.get_world_poses would give the base, not the hand).
    hand = RigidPrim(prim_paths_expr=HAND_PRIM_PATH, name="panda_hand_view")
    world.scene.add(hand)

    world.reset()   # physics handles valid after this

    dof_names = franka.dof_names
    print(f"[info] dof_names: {dof_names}")
    print(f"[info] body_names: {franka.body_names}")

    # Home pose read AFTER reset (known, not gravity-drifted).
    home = np.asarray(franka.get_joint_positions(), dtype=np.float32).reshape(-1)
    print(f"[info] home joint positions: {np.round(home, 4)}")

    d_home, hand_home = hand_distance_to_cube(hand)
    print(f"[info] cube at {CUBE_POSITION.tolist()}")
    print(f"[info] home gripper pos {np.round(hand_home, 4).tolist()} "
          f"-> distance to cube {d_home:.4f} m")
    print(f"[info] nudge = +/- {NUDGE_RAD} rad, settle = {SETTLE} steps\n")

    rows = []

    def settle_to(target_positions):
        """Command a full-DOF position target and step until the PD settles."""
        tgt = target_positions.reshape(1, -1)
        for _ in range(SETTLE):
            franka.set_joint_position_targets(tgt)
            world.step(render=False)

    for jname in TEST_JOINTS:
        if jname not in dof_names:
            print(f"[warn] {jname} not in dof_names; skipping")
            continue
        jidx = dof_names.index(jname)

        for direction in (+1.0, -1.0):
            # Reset to home each trial so trials are independent.
            world.reset()
            settle_to(home.copy())                     # re-settle exactly at home
            d_before, _ = hand_distance_to_cube(hand)

            target = home.copy()
            target[jidx] = home[jidx] + direction * NUDGE_RAD
            settle_to(target)
            d_after, hand_after = hand_distance_to_cube(hand)

            delta = d_after - d_before
            toward = "TOWARD" if delta < 0 else "away"
            rows.append({
                "joint": jname, "role": ROLE[jname],
                "dir": f"{direction:+.0f}", "nudge_rad": round(NUDGE_RAD, 4),
                "d_before_m": round(d_before, 4),
                "d_after_m": round(d_after, 4),
                "delta_m": round(delta, 4),
                "toward": toward,
                "hand_x_m": round(float(hand_after[0]), 4),
                "hand_y_m": round(float(hand_after[1]), 4),
                "hand_z_m": round(float(hand_after[2]), 4),
            })
            print(f"  {jname:13s} {direction:+.0f}  "
                  f"d {d_before:.4f} -> {d_after:.4f} m  "
                  f"delta {delta:+.4f} m  [{toward}]")

    # ------------------------------------------------------------------
    # Write CSV (the artifact). Print a ranked summary (the quick read).
    # ------------------------------------------------------------------
    with open(CSV_PATH, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\n[info] wrote {len(rows)} rows -> {CSV_PATH}")

    print("\n[summary] most-negative delta = best 'approach' joint+direction:")
    for r in sorted(rows, key=lambda r: r["delta_m"])[:4]:
        print(f"  {r['joint']:13s} {r['dir']}  delta {r['delta_m']:+.4f} m  "
              f"[{r['toward']}]  ({r['role']})")

    print("\n[done] Phase-2 loop uses the candidate joint+direction with the "
          "cleanest, most-negative delta. Controls (joint1/6) should NOT be "
          "strong 'toward' movers -- if they are, distrust the distance metric.")

    simulation_app.close()


if __name__ == "__main__":
    main()
