"""Convert NeoBot pkl motion files to bmimic npz format.

Source pkl keys: root_trans (N,3), root_rot_quat (N,4 wxyz), dof (N,29), source_path
Output npz keys: fps, joint_pos, joint_vel, body_pos_w, body_quat_w, body_lin_vel_w, body_ang_vel_w

Note: source dof is used as-is (no joint reordering applied).
"""
import argparse
import pathlib

import joblib
import numpy as np
import torch
from scipy.spatial.transform import Rotation, Slerp

from general_motion_retargeting import ROBOT_XML_DICT
from general_motion_retargeting.kinematics_model import KinematicsModel


def resample_motion(root_pos, root_rot_wxyz, dof_pos, src_fps, tgt_fps=50.0):
    """Resample motion arrays from src_fps to tgt_fps using linear/SLERP interpolation."""
    if abs(src_fps - tgt_fps) < 0.5:
        return root_pos, root_rot_wxyz, dof_pos
    N = root_pos.shape[0]
    t_src = np.arange(N) / src_fps
    duration = t_src[-1]
    M = int(round(duration * tgt_fps)) + 1
    t_tgt = np.clip(np.arange(M) / tgt_fps, 0.0, duration)

    root_pos_new = np.stack(
        [np.interp(t_tgt, t_src, root_pos[:, i]) for i in range(3)], axis=1
    ).astype(np.float32)
    dof_pos_new = np.stack(
        [np.interp(t_tgt, t_src, dof_pos[:, i]) for i in range(dof_pos.shape[1])], axis=1
    ).astype(np.float32)

    root_rot_xyzw = root_rot_wxyz[:, [1, 2, 3, 0]]
    slerp = Slerp(t_src, Rotation.from_quat(root_rot_xyzw))
    root_rot_wxyz_new = slerp(t_tgt).as_quat()[:, [3, 0, 1, 2]].astype(np.float32)

    return root_pos_new, root_rot_wxyz_new, dof_pos_new


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


joint_mapping = [0, 6, 12,
                 1, 7, 13,
                 2, 8, 14,
                 3, 9, 15, 22,
                 4, 10, 16, 23,
                 5, 11, 17, 24,
                 18, 25,
                 19, 26,
                 20, 27,
                 21, 28]
def build_bmimic_data(root_pos, root_rot_wxyz, dof_pos, fps, kinematics_model):
    """Convert robot qpos arrays to bmimic npz format dict.

    dof_pos is used as-is for joint_pos (no reordering).
    """
    N = root_pos.shape[0]
    device = kinematics_model._device

    # KinematicsModel expects xyzw quaternion
    root_rot_xyzw = root_rot_wxyz[:, [1, 2, 3, 0]]
    root_pos_t = torch.from_numpy(root_pos).float().to(device)
    root_rot_t = torch.from_numpy(root_rot_xyzw).float().to(device)
    dof_pos_t_lab  = torch.from_numpy(dof_pos).float().to(device)
    
    dof_pos_t = torch.zeros((N, 29), dtype=torch.float32, device=device)
    dof_pos_t[:, joint_mapping] = dof_pos_t_lab

    body_pos_all, body_rot_all = kinematics_model.forward_kinematics(
        root_pos_t, root_rot_t, dof_pos_t
    )
    body_pos_all = body_pos_all.cpu().numpy()
    body_rot_all = body_rot_all.cpu().numpy()

    # NeoBot pkl comes from MuJoCo simulation and is already grounded (feet at z≈0).
    # Do NOT apply z_min height adjustment — it would use KinematicsModel FK offsets
    # which differ slightly from MuJoCo FK, causing feet to float after conversion.

    # XY origin at first frame
    root_pos = root_pos.copy()
    xy0 = root_pos[0, :2].copy()
    body_pos_all[:, :, :2] -= xy0
    root_pos[:, :2] -= xy0

    # Select 30 bmimic bodies
    km_names = kinematics_model.body_names
    body_idx = [km_names.index(n) for n in G1_BMIMIC_BODY_NAMES]
    B = len(body_idx)

    body_pos_w  = body_pos_all[:, body_idx, :].astype(np.float32)
    body_rot_sel = body_rot_all[:, body_idx, :]            # xyzw (KinematicsModel convention)
    body_quat_w  = body_rot_sel[:, :, [3, 0, 1, 2]].astype(np.float32)  # xyzw → wxyz

    # joint_pos: use dof as-is (no reordering)
    joint_pos = dof_pos.astype(np.float32)

    joint_vel = np.zeros_like(joint_pos)
    joint_vel[:-1] = (joint_pos[1:] - joint_pos[:-1]) * fps
    joint_vel[-1]  = joint_vel[-2]

    body_lin_vel_w = np.zeros_like(body_pos_w)
    body_lin_vel_w[:-1] = (body_pos_w[1:] - body_pos_w[:-1]) * fps
    body_lin_vel_w[-1]  = body_lin_vel_w[-2]

    body_ang_vel_w = np.zeros((N, B, 3), dtype=np.float32)
    for b in range(B):
        rots  = Rotation.from_quat(body_rot_sel[:, b, :])
        q_rel = rots[1:] * rots[:-1].inv()
        body_ang_vel_w[:-1, b] = (q_rel.as_rotvec() * fps).astype(np.float32)
    body_ang_vel_w[-1] = body_ang_vel_w[-2]

    return {
        "fps":            np.array([int(fps)], dtype=np.int64),
        "joint_pos":      joint_pos,
        "joint_vel":      joint_vel,
        "body_pos_w":     body_pos_w,
        "body_quat_w":    body_quat_w,
        "body_lin_vel_w": body_lin_vel_w,
        "body_ang_vel_w": body_ang_vel_w,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Convert NeoBot pkl files to bmimic npz format using G1 KinematicsModel."
    )
    parser.add_argument("--input_dir",  required=True,
                        help="Root dir containing pkl files (e.g. .../from_neobot)")
    parser.add_argument("--output_dir", required=True,
                        help="Output root dir; subdirectory structure is preserved")
    parser.add_argument("--fps", type=float, default=30.0,
                        help="Frame rate of source pkl files (default: 50)")
    parser.add_argument("--robot", default="unitree_g1",
                        help="Robot name (default: unitree_g1)")
    args = parser.parse_args()

    input_dir  = pathlib.Path(args.input_dir)
    output_dir = pathlib.Path(args.output_dir)

    pkl_files = sorted(input_dir.rglob("*.pkl"))
    if not pkl_files:
        print(f"No pkl files found in {input_dir}")
        return

    print(f"Found {len(pkl_files)} pkl files")

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    km = KinematicsModel(str(ROBOT_XML_DICT[args.robot]), device=device)
    print(f"KinematicsModel loaded for {args.robot} on {device}")

    ok, failed = 0, 0
    for pkl_path in pkl_files:
        rel      = pkl_path.relative_to(input_dir)
        out_path = output_dir / rel.with_suffix(".npz")
        out_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            d             = joblib.load(pkl_path)
            root_pos      = np.asarray(d["root_trans"],    dtype=np.float32)  # (N, 3)
            root_rot_wxyz = np.asarray(d["root_rot_quat"], dtype=np.float32)  # (N, 4) wxyz
            dof_pos       = np.asarray(d["dof"],           dtype=np.float32)  # (N, 29)

            tgt_fps = 50.0
            root_pos, root_rot_wxyz, dof_pos = resample_motion(root_pos, root_rot_wxyz, dof_pos, args.fps, tgt_fps)
            motion_data = build_bmimic_data(root_pos, root_rot_wxyz, dof_pos, tgt_fps, km)
            np.savez(str(out_path), **motion_data)
            print(f"  [ok] {rel}  frames={root_pos.shape[0]}")
            ok += 1
        except Exception as e:
            print(f"  [FAIL] {rel}: {e}")
            failed += 1

    print(f"\nDone: {ok} converted, {failed} failed → {output_dir}")


if __name__ == "__main__":
    main()
