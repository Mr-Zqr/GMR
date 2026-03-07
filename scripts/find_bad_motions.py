#!/usr/bin/env python3
"""
Filter retargeted motion NPZ files (G1 robot) by quality.

Checks for:
  1. Floating feet      - average foot clearance > threshold
  2. Joint jumps        - single-frame joint position discontinuities (IK solver flips)
  3. Self-intersection  - MuJoCo collision detection between robot body meshes
                          (excluding hands/wrist_yaw and rubber_hand)

Outputs:
  - valid_motions.pkl : list of paths that passed all checks
  - bad_motions.yaml  : paths that failed (with issue details), for debugging

The two outputs are disjoint and their union equals all input files.

Usage:
  python scripts/find_bad_motions.py \
      --input_dir /home/amax/devel/dataset/NeoBot/2_gmr_retarget \
      --output_pkl valid_motions.pkl \
      --output_yaml bad_motions.yaml \
      --num_workers 16
"""

import argparse
import multiprocessing
import pickle
import sys
from pathlib import Path

import mujoco
import numpy as np
import yaml

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
XML_PATH = str(PROJECT_ROOT / "assets" / "unitree_g1" / "g1_mocap_29dof.xml")

# bmimic body order (30 bodies)
BMIMIC_BODY_NAMES = [
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

FOOT_BMIMIC_INDICES = [14, 15, 18, 19]  # ankle pitch/roll L/R in bmimic order

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


# ---------------------------------------------------------------------------
# Worker process globals (initialized once per process)
# ---------------------------------------------------------------------------

_mj_model = None
_mj_data = None
_exclude_geom_ids = None
_floor_geom_id = None


def _worker_init():
    """Load MuJoCo model once per worker process."""
    global _mj_model, _mj_data, _exclude_geom_ids, _floor_geom_id

    _mj_model = mujoco.MjModel.from_xml_path(XML_PATH)
    _mj_data = mujoco.MjData(_mj_model)

    # Find geom IDs to exclude (hands + floor)
    exclude_body_ids = set()
    for name in EXCLUDE_BODIES:
        bid = mujoco.mj_name2id(_mj_model, mujoco.mjtObj.mjOBJ_BODY, name)
        if bid >= 0:
            exclude_body_ids.add(bid)

    _exclude_geom_ids = set()
    for gid in range(_mj_model.ngeom):
        if _mj_model.geom_bodyid[gid] in exclude_body_ids:
            _exclude_geom_ids.add(gid)

    _floor_geom_id = mujoco.mj_name2id(_mj_model, mujoco.mjtObj.mjOBJ_GEOM, "floor")


def _has_self_intersection():
    """Check if current mj_data has any self-intersection contacts (excluding hands & floor)."""
    for i in range(_mj_data.ncon):
        c = _mj_data.contact[i]
        g1, g2 = c.geom1, c.geom2
        if g1 == _floor_geom_id or g2 == _floor_geom_id:
            continue
        if g1 in _exclude_geom_ids or g2 in _exclude_geom_ids:
            continue
        return True
    return False


# ---------------------------------------------------------------------------
# Per-file check
# ---------------------------------------------------------------------------

def check_motion(args_tuple):
    """Worker function: returns (path_str, issues_dict, error_str)."""
    path_str, cfg = args_tuple
    try:
        data = np.load(path_str, allow_pickle=False)
        body_pos_w  = data["body_pos_w"].astype(np.float32)   # (N, 30, 3)
        joint_pos   = data["joint_pos"].astype(np.float32)    # (N, 29) bmimic order
        body_quat_w = data["body_quat_w"].astype(np.float32)  # (N, 30, 4) wxyz
    except Exception as e:
        return path_str, None, str(e)

    N = body_pos_w.shape[0]
    if N < 2:
        return path_str, None, "too few frames"

    issues = {}

    # -- 1. Floating feet ------------------------------------------------
    ground_z   = float(body_pos_w[:, :, 2].min())
    foot_z     = body_pos_w[:, FOOT_BMIMIC_INDICES, 2]   # (N, 4)
    min_foot_z = foot_z.min(axis=1)                       # (N,)
    clearance  = float((min_foot_z - ground_z).mean())
    if clearance > cfg["float_threshold"]:
        issues["floating_feet"] = clearance

    # -- 2. Joint jump detection ------------------------------------------
    joint_diff = np.abs(np.diff(joint_pos, axis=0))  # (N-1, 29)
    max_jump_per_frame = joint_diff[2:].max(axis=1)   # skip first 2 transitions
    n_jump_frames = int(np.sum(max_jump_per_frame > cfg["jump_threshold"]))
    if n_jump_frames > 0:
        max_jump = float(max_jump_per_frame.max())
        issues["joint_jump"] = {
            "max_jump_rad": round(max_jump, 4),
            "n_jump_frames": n_jump_frames,
        }

    # -- 3. Self-intersection via MuJoCo collision detection -------------
    mujoco_dof = np.empty_like(joint_pos)
    mujoco_dof[:, G1_JOINT_MAPPING] = joint_pos

    root_pos  = body_pos_w[:, 0, :]     # (N, 3)
    root_quat = body_quat_w[:, 0, :]    # (N, 4) wxyz

    cross_ratio = cfg["cross_ratio"]
    skip = cfg.get("skip_frames", 2)
    frame_indices = range(0, N, skip)

    n_intersect = 0
    n_checked = 0

    for t in frame_indices:
        _mj_data.qpos[:3] = root_pos[t]
        _mj_data.qpos[3:7] = root_quat[t]
        _mj_data.qpos[7:] = mujoco_dof[t]

        mujoco.mj_forward(_mj_model, _mj_data)

        n_checked += 1
        if _has_self_intersection():
            n_intersect += 1

    if n_checked > 0:
        self_ratio = n_intersect / n_checked
        if self_ratio > cross_ratio:
            issues["self_intersection"] = round(self_ratio, 4)

    return path_str, issues, None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Filter retargeted NPZ motion files by quality.")
    parser.add_argument("--input_dir",       default="/home/amax/devel/dataset/NeoBot/2_gmr_retarget")
    parser.add_argument("--output_pkl",      default="valid_motions.pkl",
                        help="Output pkl: list of valid motion file paths")
    parser.add_argument("--output_yaml",     default="bad_motions.yaml",
                        help="Output yaml: bad motion paths with issue details (for debug)")
    parser.add_argument("--num_workers",     type=int,   default=20)
    parser.add_argument("--float_threshold", type=float, default=0.10,
                        help="Mean foot clearance to flag as floating (m)")
    parser.add_argument("--cross_ratio",     type=float, default=0.05,
                        help="Fraction of frames with self-intersection to flag")
    parser.add_argument("--jump_threshold",  type=float, default=0.5,
                        help="Max per-frame joint position change (rad) to flag as joint jump")
    parser.add_argument("--skip_frames",     type=int,   default=2,
                        help="Check every N-th frame for collision (speed vs accuracy)")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    if not input_dir.exists():
        print(f"Error: input_dir not found: {input_dir}")
        sys.exit(1)

    npz_files = sorted(input_dir.rglob("*.npz"))
    total = len(npz_files)
    if total == 0:
        print(f"No .npz files found in {input_dir}")
        sys.exit(0)

    print(f"Found {total} NPZ files in {input_dir}")
    print(f"Workers: {args.num_workers}")
    print(f"Skip frames: {args.skip_frames}")

    cfg = {
        "float_threshold": args.float_threshold,
        "jump_threshold":  args.jump_threshold,
        "cross_ratio":     args.cross_ratio,
        "skip_frames":     args.skip_frames,
    }

    task_args = [(str(p), cfg) for p in npz_files]

    valid_paths = []                  # no issues at all
    bad_entries = {}                  # path -> issues dict (floating_feet / self_intersection)
    errors = []

    with multiprocessing.Pool(processes=args.num_workers, initializer=_worker_init) as pool:
        for i, (path_str, issues, err) in enumerate(pool.imap_unordered(check_motion, task_args), 1):
            if i % 500 == 0 or i == total:
                print(f"  {i}/{total}", flush=True)
            if err is not None:
                bad_entries[path_str] = {"error": err}
                continue
            if "floating_feet" in issues or "joint_jump" in issues or "self_intersection" in issues:
                bad_entries[path_str] = issues
            else:
                valid_paths.append(path_str)

    valid_paths.sort()
    bad_paths_sorted = sorted(bad_entries.keys())

    n_valid = len(valid_paths)
    n_bad = len(bad_paths_sorted)

    # -- Write valid_motions.pkl --
    pkl_path = Path(args.output_pkl)
    with open(pkl_path, "wb") as f:
        pickle.dump(valid_paths, f)

    # -- Write bad_motions.yaml --
    # Count per-issue
    n_floating    = sum(1 for v in bad_entries.values() if "floating_feet" in v)
    n_joint_jump  = sum(1 for v in bad_entries.values() if "joint_jump" in v)
    n_self_int    = sum(1 for v in bad_entries.values() if "self_intersection" in v)
    n_errors      = sum(1 for v in bad_entries.values() if "error" in v)

    yaml_output = {
        "summary": {
            "total":             total,
            "valid":             n_valid,
            "bad":               n_bad,
            "floating_feet":     n_floating,
            "joint_jump":        n_joint_jump,
            "self_intersection": n_self_int,
            "load_errors":       n_errors,
        },
        "bad_motions": {p: bad_entries[p] for p in bad_paths_sorted},
    }

    yaml_path = Path(args.output_yaml)
    with open(yaml_path, "w") as f:
        yaml.dump(yaml_output, f, default_flow_style=False, allow_unicode=True, sort_keys=False)

    print()
    print("=" * 60)
    print(f"  Total files       : {total}")
    print(f"  Valid (pkl)       : {n_valid}")
    print(f"  Bad (yaml)        : {n_bad}")
    print(f"    floating_feet   : {n_floating}")
    print(f"    joint_jump      : {n_joint_jump}")
    print(f"    self_intersection: {n_self_int}")
    print(f"    load_errors     : {n_errors}")
    print(f"  valid + bad       : {n_valid + n_bad}  (should == {total})")
    print(f"  Output pkl        : {pkl_path.resolve()}")
    print(f"  Output yaml       : {yaml_path.resolve()}")
    print("=" * 60)

    assert n_valid + n_bad == total, f"BUG: valid({n_valid}) + bad({n_bad}) != total({total})"


if __name__ == "__main__":
    main()
