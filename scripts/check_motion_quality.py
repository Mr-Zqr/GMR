#!/usr/bin/env python3
"""
Check retargeted motion NPZ files (G1 robot) for quality issues.

Reports per-file and aggregate frame-level ratios for:
  1. Joint jumps        - single-frame joint position discontinuities
  2. Self-intersection  - MuJoCo collision detection (excluding hands)
  3. Joint at limit     - any joint within limit_threshold of its URDF limit

Usage:
  python scripts/check_motion_quality.py \
      --input_dir /path/to/npz/directory
"""

import argparse
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import numpy as np

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
XML_PATH  = str(PROJECT_ROOT / "assets" / "unitree_g1" / "g1_mocap_29dof.xml")
URDF_PATH = str(PROJECT_ROOT / "assets" / "unitree_g1" / "g1_custom_collision_29dof.urdf")

# bmimic joint → mujoco DOF reordering
G1_JOINT_MAPPING = [
    0, 6, 12,
    1, 7, 13,
    2, 8, 14,
    3,  9, 15, 22,
    4, 10, 16, 23,
    5, 11, 17, 24,
          18, 25,
          19, 26,
          20, 27,
          21, 28,
]

# Bodies whose geoms should be excluded from self-intersection checks
EXCLUDE_BODIES = {
    "left_wrist_yaw_link", "right_wrist_yaw_link",
    "left_rubber_hand", "right_rubber_hand",
}

# bmimic body order (body_pos_w axis-1 indices)
G1_BMIMIC_BODY_NAMES = [
    "pelvis",               "left_hip_pitch_link",   "right_hip_pitch_link",
    "waist_yaw_link",       "left_hip_roll_link",    "right_hip_roll_link",
    "waist_roll_link",      "left_hip_yaw_link",     "right_hip_yaw_link",
    "torso_link",           "left_knee_link",         "right_knee_link",
    "left_shoulder_pitch_link",  "right_shoulder_pitch_link",
    "left_ankle_pitch_link",     "right_ankle_pitch_link",
    "left_shoulder_roll_link",   "right_shoulder_roll_link",
    "left_ankle_roll_link",      "right_ankle_roll_link",
    "left_shoulder_yaw_link",    "right_shoulder_yaw_link",
    "left_elbow_link",           "right_elbow_link",
    "left_wrist_roll_link",      "right_wrist_roll_link",
    "left_wrist_pitch_link",     "right_wrist_pitch_link",
    "left_wrist_yaw_link",       "right_wrist_yaw_link",
]


def load_joint_limits_from_urdf(urdf_path, mj_model):
    """Parse URDF joint limits and return arrays in bmimic joint order.

    Returns (lower, upper) each of shape (29,) float32.
    """
    tree = ET.parse(urdf_path)
    root = tree.getroot()
    urdf_limits = {}
    for joint in root.findall(".//joint"):
        if joint.get("type") != "revolute":
            continue
        lim = joint.find("limit")
        if lim is None:
            continue
        urdf_limits[joint.get("name")] = (float(lim.get("lower")), float(lim.get("upper")))

    # Collect limits in MuJoCo joint order (skip joint 0 = free root joint)
    lower_mujoco = np.empty(mj_model.njnt - 1, dtype=np.float32)
    upper_mujoco = np.empty(mj_model.njnt - 1, dtype=np.float32)
    for i in range(1, mj_model.njnt):
        name = mujoco.mj_id2name(mj_model, mujoco.mjtObj.mjOBJ_JOINT, i)
        lo, hi = urdf_limits[name]
        lower_mujoco[i - 1] = lo
        upper_mujoco[i - 1] = hi

    # Convert to bmimic order: G1_JOINT_MAPPING[bmimic_idx] = mujoco_dof_idx
    lower_bmimic = lower_mujoco[G1_JOINT_MAPPING]
    upper_bmimic = upper_mujoco[G1_JOINT_MAPPING]
    return lower_bmimic, upper_bmimic


def check_file(path, mj_model, mj_data, exclude_geom_ids, floor_geom_id,
               jump_threshold, skip_frames, joint_lower, joint_upper, limit_threshold):
    """Check a single npz file.

    Returns (n_frames, n_jump_frames, n_collision_frames, n_checked, n_limit).
    """
    data = np.load(path, allow_pickle=False)
    joint_pos   = data["joint_pos"].astype(np.float32)    # (N, 29) bmimic order
    body_pos_w  = data["body_pos_w"].astype(np.float32)   # (N, 30, 3)
    body_quat_w = data["body_quat_w"].astype(np.float32)  # (N, 30, 4) wxyz

    N = joint_pos.shape[0]
    if N < 3:
        return N, 0, 0, 0, 0

    # -- Joint jump detection --
    joint_diff = np.abs(np.diff(joint_pos, axis=0))  # (N-1, 29)
    max_jump_per_frame = joint_diff[2:].max(axis=1)   # skip first 2 transitions
    n_jump_frames = int(np.sum(max_jump_per_frame > jump_threshold))

    # -- Self-intersection via MuJoCo collision detection --
    mujoco_dof = np.empty_like(joint_pos)
    mujoco_dof[:, G1_JOINT_MAPPING] = joint_pos

    root_pos  = body_pos_w[:, 0, :]     # (N, 3)
    root_quat = body_quat_w[:, 0, :]    # (N, 4) wxyz

    n_collision = 0
    n_checked = 0

    for t in range(0, N, skip_frames):
        mj_data.qpos[:3] = root_pos[t]
        mj_data.qpos[3:7] = root_quat[t]
        mj_data.qpos[7:] = mujoco_dof[t]

        mujoco.mj_forward(mj_model, mj_data)

        n_checked += 1
        for i in range(mj_data.ncon):
            c = mj_data.contact[i]
            g1, g2 = c.geom1, c.geom2
            if g1 == floor_geom_id or g2 == floor_geom_id:
                continue
            if g1 in exclude_geom_ids or g2 in exclude_geom_ids:
                continue
            n_collision += 1
            break

    # -- Joint at limit detection (all frames) --
    at_lower = joint_pos < (joint_lower + limit_threshold)   # (N, 29)
    at_upper = joint_pos > (joint_upper - limit_threshold)   # (N, 29)
    n_limit = int(np.sum(np.any(at_lower | at_upper, axis=1)))

    return N, n_jump_frames, n_collision, n_checked, n_limit


def main():
    parser = argparse.ArgumentParser(description="Check retargeted NPZ motion files for quality issues.")
    parser.add_argument("--input_dir", required=True, help="Directory containing .npz motion files")
    parser.add_argument("--jump_threshold", type=float, default=0.5,
                        help="Max per-frame joint change (rad) to flag as jump")
    parser.add_argument("--skip_frames", type=int, default=2,
                        help="Check every N-th frame for collision")
    parser.add_argument("--limit_threshold", type=float, default=0.05,
                        help="Distance (rad) from joint limit to count as 'at limit' (default 0.05)")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    if not input_dir.exists():
        print(f"Error: directory not found: {input_dir}")
        sys.exit(1)

    npz_files = sorted(input_dir.rglob("*.npz"))
    if not npz_files:
        print(f"No .npz files found in {input_dir}")
        sys.exit(0)

    # Load MuJoCo model
    mj_model = mujoco.MjModel.from_xml_path(XML_PATH)
    mj_data = mujoco.MjData(mj_model)

    # Load joint limits from URDF
    joint_lower, joint_upper = load_joint_limits_from_urdf(URDF_PATH, mj_model)

    # Find geom IDs to exclude
    exclude_body_ids = set()
    for name in EXCLUDE_BODIES:
        bid = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY, name)
        if bid >= 0:
            exclude_body_ids.add(bid)

    exclude_geom_ids = set()
    for gid in range(mj_model.ngeom):
        if mj_model.geom_bodyid[gid] in exclude_body_ids:
            exclude_geom_ids.add(gid)

    floor_geom_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_GEOM, "floor")

    # Process files
    total_files = len(npz_files)
    total_frames = 0
    total_jump_frames = 0
    total_collision_frames = 0
    total_checked_frames = 0
    total_limit_frames = 0

    print(f"Found {total_files} NPZ files in {input_dir}")
    print(f"Jump threshold: {args.jump_threshold} rad  |  Skip frames: {args.skip_frames}  |  "
          f"Limit threshold: {args.limit_threshold} rad")
    W = 84
    print("-" * W)
    print(f"{'File':<50} {'Frames':>6} {'Jump':>6} {'Jump%':>7} {'Coll':>6} {'Coll%':>7} {'Limit':>6} {'Limit%':>7}")
    print("-" * W)

    for path in npz_files:
        try:
            n_frames, n_jump, n_coll, n_checked, n_limit = check_file(
                str(path), mj_model, mj_data, exclude_geom_ids, floor_geom_id,
                args.jump_threshold, args.skip_frames,
                joint_lower, joint_upper, args.limit_threshold)
        except Exception as e:
            print(f"{path.name:<50} ERROR: {e}")
            continue

        jump_ratio  = n_jump  / max(n_frames - 3, 1)
        coll_ratio  = n_coll  / max(n_checked, 1)
        limit_ratio = n_limit / max(n_frames, 1)

        total_frames           += n_frames
        total_jump_frames      += n_jump
        total_collision_frames += n_coll
        total_checked_frames   += n_checked
        total_limit_frames     += n_limit

        print(f"{path.name:<50} {n_frames:>6} {n_jump:>6} {jump_ratio:>6.1%} "
              f"{n_coll:>6} {coll_ratio:>6.1%} {n_limit:>6} {limit_ratio:>6.1%}")

    print("-" * W)
    agg_jump_ratio  = total_jump_frames      / max(total_frames, 1)
    agg_coll_ratio  = total_collision_frames / max(total_checked_frames, 1)
    agg_limit_ratio = total_limit_frames     / max(total_frames, 1)
    print(f"{'TOTAL':<50} {total_frames:>6} {total_jump_frames:>6} {agg_jump_ratio:>6.1%} "
          f"{total_collision_frames:>6} {agg_coll_ratio:>6.1%} {total_limit_frames:>6} {agg_limit_ratio:>6.1%}")


if __name__ == "__main__":
    main()
