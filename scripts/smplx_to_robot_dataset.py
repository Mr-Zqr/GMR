import argparse
import os
import multiprocessing as mp
import pathlib

import numpy as np
import torch
from scipy.spatial.transform import Rotation
from tqdm import tqdm
from natsort import natsorted
from rich import print
import gc
import time
import psutil
import tracemalloc

from general_motion_retargeting import GeneralMotionRetargeting as GMR
from general_motion_retargeting.utils.smpl import load_smplx_file, get_smplx_data_offline_fast
from general_motion_retargeting.kinematics_model import KinematicsModel


# ── G1 bmimic format constants ───────────────────────────────────────────────
# Reordering from MuJoCo DOF order → bmimic joint order (reverse: bmimic[:,i] = mujoco[:,JOINT_MAPPING[i]])
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

# 30 bodies in bmimic order for G1
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

# Robots that use G1 bmimic body layout
G1_ROBOTS = {"unitree_g1", "unitree_g1_with_hands"}


HERE = pathlib.Path(__file__).parent


def check_memory(threshold_gb=5):
    mem = psutil.virtual_memory()
    if mem.available / (1024 ** 3) < threshold_gb:
        print(f"[WARNING] Available memory {mem.available/(1024**3):.2f} GB < {threshold_gb} GB")
        return True
    return False


def build_bmimic_data(root_pos, root_rot_wxyz, dof_pos, fps, kinematics_model, tgt_robot):
    """
    Convert retargeted qpos arrays to bmimic npz format.

    Args:
        root_pos:       (N, 3) float32, world position
        root_rot_wxyz:  (N, 4) float32, quaternion in MuJoCo wxyz convention
        dof_pos:        (N, D) float32, MuJoCo DOF order
        fps:            float
        kinematics_model: KinematicsModel instance
        tgt_robot:      str, robot name

    Returns:
        dict matching the bmimic npz format:
          fps(1,), joint_pos(N,D'), joint_vel(N,D'),
          body_pos_w(N,B,3), body_quat_w(N,B,4 wxyz),
          body_lin_vel_w(N,B,3), body_ang_vel_w(N,B,3)
    """
    N = root_pos.shape[0]
    device = kinematics_model._device

    # KinematicsModel uses xyzw quaternions
    root_rot_xyzw = root_rot_wxyz[:, [1, 2, 3, 0]]

    root_pos_t   = torch.from_numpy(root_pos).float().to(device)
    root_rot_t   = torch.from_numpy(root_rot_xyzw).float().to(device)
    dof_pos_t    = torch.from_numpy(dof_pos).float().to(device)

    body_pos_all, body_rot_all = kinematics_model.forward_kinematics(
        root_pos_t, root_rot_t, dof_pos_t
    )
    body_pos_all = body_pos_all.cpu().numpy()   # (N, B_all, 3)
    body_rot_all = body_rot_all.cpu().numpy()   # (N, B_all, 4) xyzw

    # Height adjustment: lift lowest body part to z = 0
    z_min = float(np.min(body_pos_all[..., 2]))
    body_pos_all[:, :, 2] -= z_min
    root_pos = root_pos.copy()
    root_pos[:, 2] -= z_min

    # XY origin at first frame root position
    xy0 = root_pos[0, :2].copy()
    body_pos_all[:, :, :2] -= xy0
    root_pos[:, :2] -= xy0

    # Body selection / reordering
    km_names = kinematics_model.body_names
    if tgt_robot in G1_ROBOTS:
        body_names   = G1_BMIMIC_BODY_NAMES
        joint_mapping = G1_JOINT_MAPPING
    else:
        body_names   = km_names
        joint_mapping = list(range(dof_pos.shape[1]))

    body_idx = [km_names.index(n) for n in body_names]
    B = len(body_idx)

    body_pos_w  = body_pos_all[:, body_idx, :].astype(np.float32)   # (N, B, 3)
    body_rot_sel = body_rot_all[:, body_idx, :]                       # (N, B, 4) xyzw
    # Convert xyzw → wxyz for bmimic
    body_quat_w = body_rot_sel[:, :, [3, 0, 1, 2]].astype(np.float32)  # (N, B, 4)

    # Joint positions in bmimic order
    joint_pos = dof_pos[:, joint_mapping].astype(np.float32)          # (N, D')

    # Finite-difference velocities (forward difference, last frame repeats)
    joint_vel = np.zeros_like(joint_pos)
    joint_vel[:-1] = (joint_pos[1:] - joint_pos[:-1]) * fps
    joint_vel[-1]  = joint_vel[-2]

    body_lin_vel_w = np.zeros_like(body_pos_w)
    body_lin_vel_w[:-1] = (body_pos_w[1:] - body_pos_w[:-1]) * fps
    body_lin_vel_w[-1]  = body_lin_vel_w[-2]

    # Angular velocity via quaternion finite difference
    body_ang_vel_w = np.zeros((N, B, 3), dtype=np.float32)
    for b in range(B):
        rots  = Rotation.from_quat(body_rot_sel[:, b, :])   # scipy uses xyzw
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


def process_file(smplx_file_path, tgt_file_path, tgt_robot, SMPLX_FOLDER,
                 tgt_folder, total_files, src_fps, yup_to_zup=False, verbose=False):

    def log_memory(msg):
        if verbose:
            mb = psutil.Process(os.getpid()).memory_info().rss / (1024 ** 2)
            print(f"[MEMORY] {msg}: {mb:.0f} MB")

    if verbose:
        tracemalloc.start()

    num_pause = 0
    while check_memory():
        print(f"[PAUSE] Waiting for memory ({num_pause})")
        time.sleep(120)
        num_pause += 1
        if num_pause > 10:
            print(f"[ERROR] Memory still high after 10 pauses, skipping.")
            return

    try:
        smplx_data, body_model, smplx_output, actual_human_height = load_smplx_file(
            smplx_file_path, SMPLX_FOLDER
        )
    except Exception as e:
        print(f"[ERROR] Loading {smplx_file_path}: {e}")
        return
    log_memory("after load")

    tgt_fps = 50
    try:
        smplx_frames, aligned_fps = get_smplx_data_offline_fast(
            smplx_data, body_model, smplx_output, tgt_fps=tgt_fps, src_fps=src_fps,
            yup_to_zup=yup_to_zup
        )
    except Exception as e:
        import traceback
        print(f"[ERROR] Preprocessing {smplx_file_path}: {e}")
        traceback.print_exc()
        return

    retargeter = GMR(
        src_human="smplx",
        tgt_robot=tgt_robot,
        actual_human_height=actual_human_height,
        verbose=False,
    )

    qpos_list = []
    for frame in smplx_frames:
        qpos, _ = retargeter.retarget(frame)
        qpos_list.append(qpos)

    qpos_arr = np.array(qpos_list)          # (N, 7+D)
    root_pos      = qpos_arr[:, :3].copy()
    root_rot_wxyz = qpos_arr[:, 3:7].astype(np.float32)
    dof_pos       = qpos_arr[:, 7:].astype(np.float32)

    log_memory("after retarget")

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    km = KinematicsModel(retargeter.xml_file, device=device)

    try:
        motion_data = build_bmimic_data(
            root_pos, root_rot_wxyz, dof_pos, aligned_fps, km, tgt_robot
        )
    except Exception as e:
        print(f"[ERROR] Building bmimic data for {smplx_file_path}: {e}")
        return

    log_memory("after FK")

    os.makedirs(os.path.dirname(tgt_file_path), exist_ok=True)
    np.savez(tgt_file_path, **motion_data)

    done = sum(
        len([f for f in files if f.endswith('.npz')])
        for _, _, files in os.walk(tgt_folder)
    )
    print(f"Saved [{done}/{total_files}]: {tgt_file_path}")

    if verbose:
        snap = tracemalloc.take_snapshot()
        for stat in snap.statistics('lineno')[:5]:
            print(stat)
        tracemalloc.stop()

    torch.cuda.empty_cache()
    gc.collect()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--robot",      default="unitree_g1")
    parser.add_argument("--src_folder", type=str, required=True)
    parser.add_argument("--tgt_folder", type=str, required=True)
    parser.add_argument("--src_fps",    type=int, default=None,
                        help="Source FPS (auto-detected from file if possible, default 120)")
    parser.add_argument("--override",    default=False, action="store_true")
    parser.add_argument("--yup_to_zup", default=False, action="store_true",
                        help="Rotate input SMPL data from y-up to z-up coordinate system")
    parser.add_argument("--num_cpus",   default=2, type=int)
    args = parser.parse_args()

    print(f"Total CPUs: {mp.cpu_count()}, using {args.num_cpus}")

    SMPLX_FOLDER      = HERE / ".." / "assets" / "body_models"
    hard_motions_dir  = HERE / ".." / "assets" / "hard_motions"

    # Load exclusion lists
    hard_motions = set()
    for path in [hard_motions_dir / "0.txt", hard_motions_dir / "1.txt"]:
        if path.exists():
            with open(path) as f:
                for line in f:
                    if "Motion:" in line:
                        m = line.split(":")[1].strip().split(",")[0].strip().split(".")[0]
                        hard_motions.add(m)

    exclude_keywords = ["BMLrub", "EKUT", "crawl", "_lie", "upstairs", "downstairs"]

    # Collect input files
    args_list = []
    for dirpath, _, filenames in os.walk(args.src_folder):
        for filename in natsorted(filenames):
            if filename.endswith("_stagei.npz"):
                continue
            if not filename.endswith((".pkl", ".npz")):
                continue

            smplx_file = os.path.join(dirpath, filename)
            stem = os.path.splitext(filename)[0]
            tgt_file = os.path.join(
                dirpath.replace(args.src_folder, args.tgt_folder),
                stem + ".npz"
            )

            if os.path.exists(tgt_file) and not args.override:
                continue
            if stem in hard_motions:
                continue
            if any(kw in filename for kw in exclude_keywords):
                continue

            args_list.append((smplx_file, tgt_file, args.robot,
                               SMPLX_FOLDER, args.tgt_folder, args.src_fps, args.yup_to_zup))

    total = len(args_list)
    print(f"Files to process: {total}")

    with mp.Pool(args.num_cpus) as pool:
        pool.starmap(process_file,
                     [(smplx_file, tgt_file, robot, smplx_folder, tgt_folder, total, src_fps, yup_to_zup, False)
                      for smplx_file, tgt_file, robot, smplx_folder, tgt_folder, src_fps, yup_to_zup in args_list])

    print("Done. Saved to", args.tgt_folder)


if __name__ == "__main__":
    main()
