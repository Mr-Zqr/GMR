#!/usr/bin/env python3
"""
Convert prediction pkl files (pred_g1_trans / pred_g1_root_ori_quat / pred_g1_dof)
to bmimic npz format (same as smplx_to_robot_gmr.py output).

Input pkl keys:
  pred_g1_trans          (N, 3)   root translation
  pred_g1_root_ori_quat  (N, 4)   root rotation, wxyz
  pred_g1_dof            (N, 29)  joint positions, lab order

Output npz keys:
  fps, joint_pos, joint_vel, body_pos_w, body_quat_w, body_lin_vel_w, body_ang_vel_w

Usage:
  # single file
  python pred_pkl_to_bmimic.py --input 137_pred.pkl --output 137.npz

  # directory (mirrors directory tree)
  python pred_pkl_to_bmimic.py --input pred/ --output bmimic/
"""

import argparse
import os
import pickle
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.spatial.transform import Rotation, Slerp
from scipy.interpolate import interp1d

sys.path.insert(0, str(Path(__file__).parent.parent))
from general_motion_retargeting.kinematics_model import KinematicsModel
from general_motion_retargeting.params import ROBOT_XML_DICT
from scripts.smplx_to_robot_gmr import build_bmimic_data


def resample(root_pos, root_rot_wxyz, dof_lab, src_fps, tgt_fps):
    """Resample motion from src_fps to tgt_fps using linear interp + slerp."""
    N = len(root_pos)
    t_src = np.arange(N) / src_fps
    duration = t_src[-1]
    N_tgt = int(round(duration * tgt_fps)) + 1
    t_tgt = np.linspace(0, duration, N_tgt)

    # linear interp for positions and dof
    root_pos_r = interp1d(t_src, root_pos, axis=0)(t_tgt).astype(np.float32)
    dof_r      = interp1d(t_src, dof_lab,  axis=0)(t_tgt).astype(np.float32)

    # slerp for quaternions (wxyz → xyzw for scipy, then back)
    rots = Rotation.from_quat(root_rot_wxyz[:, [1, 2, 3, 0]])  # wxyz→xyzw
    slerp = Slerp(t_src, rots)
    rot_r_xyzw = slerp(t_tgt).as_quat()                        # (N_tgt, 4) xyzw
    rot_r = rot_r_xyzw[:, [3, 0, 1, 2]].astype(np.float32)    # xyzw→wxyz

    return root_pos_r, rot_r, dof_r


def to_numpy(x):
    if hasattr(x, "cpu"):
        return x.detach().cpu().numpy()
    return np.asarray(x, dtype=np.float32)


# build_bmimic_data expects dof in mujoco order and applies G1_JOINT_MAPPING internally
# to produce lab-order joint_pos.  pred_g1_dof is already in lab order, so we convert
# lab → mujoco here so the internal mapping restores the correct lab order.
G1_JOINT_MAPPING = [
    0, 6, 12, 1, 7, 13, 2, 8, 14, 3, 9, 15, 22,
    4, 10, 16, 23, 5, 11, 17, 24, 18, 25, 19, 26, 20, 27, 21, 28,
]


def convert_file(input_path, output_path, km, src_fps, tgt_fps):
    with open(input_path, "rb") as f:
        d = pickle.load(f)

    root_pos      = to_numpy(d["pred_g1_trans"])          # (N, 3)
    root_rot_wxyz = to_numpy(d["pred_g1_root_ori_quat"])  # (N, 4) wxyz
    dof_lab       = to_numpy(d["pred_g1_dof"])            # (N, 29) lab order

    # resample src_fps → tgt_fps
    if src_fps != tgt_fps:
        root_pos, root_rot_wxyz, dof_lab = resample(root_pos, root_rot_wxyz, dof_lab, src_fps, tgt_fps)

    # lab → mujoco order (build_bmimic_data will convert back to lab internally)
    N = dof_lab.shape[0]
    dof_pos = np.zeros((N, 29), dtype=np.float32)
    dof_pos[:, G1_JOINT_MAPPING] = dof_lab

    motion_data = build_bmimic_data(root_pos, root_rot_wxyz, dof_pos, tgt_fps, km, "unitree_g1")

    os.makedirs(Path(output_path).parent, exist_ok=True)
    out = str(output_path).removesuffix(".npz")
    np.savez(out, **motion_data)
    print(f"  saved → {output_path}")


def main():
    parser = argparse.ArgumentParser(description="pred pkl → bmimic npz")
    parser.add_argument("--input",  required=True, help="Input pkl file or directory")
    parser.add_argument("--output", default=None,  help="Output npz file or directory")
    parser.add_argument("--src_fps", type=float, default=30.0, help="Source FPS of input pkl (default: 30)")
    parser.add_argument("--tgt_fps", type=float, default=50.0, help="Target FPS of output npz (default: 50)")
    parser.add_argument("--device", default=None,
                        help="Torch device (default: cuda if available, else cpu)")
    args = parser.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    xml_file = str(ROBOT_XML_DICT["unitree_g1"])
    km = KinematicsModel(xml_file, device=device)

    input_path = Path(args.input)

    if input_path.is_file():
        output_path = args.output or str(input_path.parent / input_path.name.replace(".pkl", ".npz"))
        print(f"Processing: {input_path}")
        convert_file(str(input_path), output_path, km, args.src_fps, args.tgt_fps)

    elif input_path.is_dir():
        pkl_files = sorted(input_path.rglob("*.pkl"))
        if not pkl_files:
            print(f"No .pkl files found in {input_path}")
            return
        output_root = Path(args.output) if args.output else input_path.parent / (input_path.name + "_bmimic")
        print(f"Found {len(pkl_files)} pkl files → {output_root}")
        for pkl_file in pkl_files:
            rel = pkl_file.relative_to(input_path)
            out = output_root / rel.with_suffix(".npz")
            print(f"[{pkl_file.name}]")
            try:
                convert_file(str(pkl_file), str(out), km, args.src_fps, args.tgt_fps)
            except Exception as e:
                print(f"  [ERROR] {e}")
        print("Done.")

    else:
        print(f"Error: {input_path} not found")


if __name__ == "__main__":
    main()
