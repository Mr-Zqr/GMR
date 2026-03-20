import argparse
import pathlib
import os
import time
from multiprocessing import Pool

import numpy as np
import torch
from scipy.spatial.transform import Rotation

from general_motion_retargeting import GeneralMotionRetargeting as GMR
from general_motion_retargeting import RobotMotionViewer, ROBOT_XML_DICT
from general_motion_retargeting.utils.smpl import load_smplx_file, get_smplx_data_offline_fast
from general_motion_retargeting.kinematics_model import KinematicsModel

from rich import print

# ── G1 bmimic format constants (same as smplx_to_robot_dataset.py) ───────────
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

G1_ROBOTS = {"unitree_g1", "unitree_g1_with_hands"}


def retarget_segment(args):
    """Worker: retarget one segment of frames in a separate process."""
    segment_frames, actual_human_height, robot = args
    retarget = GMR(
        actual_human_height=actual_human_height,
        src_human="smplx",
        tgt_robot=robot,
        verbose=False,
    )
    qpos_list = []
    for frame in segment_frames:
        qpos, _ = retarget.retarget(frame)
        qpos_list.append(qpos.copy())
    return qpos_list


def build_bmimic_data(root_pos, root_rot_wxyz, dof_pos, fps, kinematics_model, tgt_robot):
    """Convert retargeted qpos arrays to bmimic npz format dict."""
    N = root_pos.shape[0]
    device = kinematics_model._device

    root_rot_xyzw = root_rot_wxyz[:, [1, 2, 3, 0]]
    root_pos_t  = torch.from_numpy(root_pos).float().to(device)
    root_rot_t  = torch.from_numpy(root_rot_xyzw).float().to(device)
    dof_pos_t   = torch.from_numpy(dof_pos).float().to(device)

    body_pos_all, body_rot_all = kinematics_model.forward_kinematics(
        root_pos_t, root_rot_t, dof_pos_t
    )
    body_pos_all = body_pos_all.cpu().numpy()
    body_rot_all = body_rot_all.cpu().numpy()

    # Height adjustment
    z_min = float(np.min(body_pos_all[..., 2]))
    body_pos_all[:, :, 2] -= z_min
    root_pos = root_pos.copy()
    root_pos[:, 2] -= z_min

    # XY origin at first frame
    xy0 = root_pos[0, :2].copy()
    body_pos_all[:, :, :2] -= xy0
    root_pos[:, :2] -= xy0

    # Body selection / reordering
    km_names = kinematics_model.body_names
    if tgt_robot in G1_ROBOTS:
        body_names    = G1_BMIMIC_BODY_NAMES
        joint_mapping = G1_JOINT_MAPPING
    else:
        body_names    = km_names
        joint_mapping = list(range(dof_pos.shape[1]))

    body_idx = [km_names.index(n) for n in body_names]
    B = len(body_idx)

    body_pos_w   = body_pos_all[:, body_idx, :].astype(np.float32)
    body_rot_sel = body_rot_all[:, body_idx, :]
    body_quat_w  = body_rot_sel[:, :, [3, 0, 1, 2]].astype(np.float32)  # xyzw→wxyz

    joint_pos = dof_pos[:, joint_mapping].astype(np.float32)

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


if __name__ == "__main__":

    HERE = pathlib.Path(__file__).parent

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--smplx_file",
        type=str,
        required=True
    )
    parser.add_argument(
        "--robot",
        choices=["unitree_g1", "unitree_g1_with_hands", "unitree_h1", "unitree_h1_2",
                 "booster_t1", "booster_t1_29dof", "stanford_toddy", "fourier_n1",
                 "engineai_pm01", "kuavo_s45", "hightorque_hi", "galaxea_r1pro",
                 "berkeley_humanoid_lite", "booster_k1", "pnd_adam_lite", "tienkung"],
        default="unitree_g1",
    )
    parser.add_argument("--save_path",  required=True,
                        help="Output .npz path (bmimic format). Directory auto-created.")
    parser.add_argument("--src_fps",    type=int, default=None,
                        help="Source FPS (auto-detected if not specified, default 120)")
    parser.add_argument("--loop",       default=False, action="store_true")
    parser.add_argument("--record_video", default=False, action="store_true")
    parser.add_argument("--rate_limit", default=True, action="store_true")
    parser.add_argument("--visualize_smplx", default=False, action="store_true")
    parser.add_argument("--viz_mode",
                        choices=["interactive", "animation", "grid"],
                        default="interactive")
    parser.add_argument("--headless", default=False, action="store_true",
                        help="Skip visualization, only retarget and save.")
    parser.add_argument("--num_workers", type=int, default=None,
                        help="Number of parallel workers (default: cpu_count). Set 1 to disable parallel.")
    parser.add_argument("--yup_to_zup", default=False, action="store_true",
                        help="Rotate SMPL joints from y-up to z-up before retargeting (for datasets in y-up convention).")

    args = parser.parse_args()

    SMPLX_FOLDER = HERE / ".." / "assets" / "body_models"

    tgt_fps = 30

    smplx_data, body_model, smplx_output, actual_human_height = load_smplx_file(
        args.smplx_file, SMPLX_FOLDER, downsample_fps=tgt_fps
    )
    if args.src_fps is not None:
        src_fps_detected = float(args.src_fps)
    elif "mocap_frame_rate" in smplx_data:
        src_fps_detected = float(np.array(smplx_data["mocap_frame_rate"]).squeeze())
    else:
        src_fps_detected = 120.0

    src_frame_count = int(np.asarray(smplx_data["trans"]).shape[0])
    src_duration = src_frame_count / src_fps_detected if src_fps_detected > 0 else 0.0

    # After downsample, smplx_data["mocap_frame_rate"] is already tgt_fps;
    # don't pass args.src_fps here as it refers to the original pre-downsample rate.
    smplx_data_frames, aligned_fps = get_smplx_data_offline_fast(
        smplx_data, body_model, smplx_output, tgt_fps=tgt_fps, yup_to_zup=args.yup_to_zup
    )
    mapped_frame_count = len(smplx_data_frames)
    mapped_duration = mapped_frame_count / aligned_fps if aligned_fps > 0 else 0.0

    print(f"[debug] source_fps={src_fps_detected:.3f}, target_fps={float(tgt_fps):.3f}, aligned_fps={float(aligned_fps):.3f}")
    print(
        f"[debug] before_mapping: frames={src_frame_count}, duration={src_duration:.3f}s | "
        f"after_mapping: frames={mapped_frame_count}, duration={mapped_duration:.3f}s"
    )

    if args.visualize_smplx:
        print(f"Total frames: {len(smplx_data_frames)}, FPS: {aligned_fps}, "
              f"Duration: {len(smplx_data_frames)/aligned_fps:.2f}s")
        try:
            from visualize_smplx_simple import (
                visualize_frames_interactive,
                visualize_frames_animation,
                visualize_frames_grid,
            )
            if args.viz_mode == "interactive":
                visualize_frames_interactive(smplx_data_frames, aligned_fps)
            elif args.viz_mode == "animation":
                visualize_frames_animation(smplx_data_frames, aligned_fps)
            else:
                visualize_frames_grid(smplx_data_frames, num_samples=6)
        except ImportError as e:
            print(f"[red]Cannot import visualize_smplx_simple: {e}[/red]")
            exit(1)

    # ── Retargeting ─────────────────────────────────────────────────────────
    if args.headless:
        # Headless mode: parallel segment retargeting, no visualization
        segment_sec = 4
        segment_size = int(segment_sec * aligned_fps)
        segments = [smplx_data_frames[i:i + segment_size]
                    for i in range(0, len(smplx_data_frames), segment_size)]
        print(f"[info] Splitting {len(smplx_data_frames)} frames into {len(segments)} segments "
              f"({segment_sec}s, {segment_size} frames each)")

        num_workers = args.num_workers or min(len(segments), os.cpu_count() or 1)

        if num_workers > 1 and len(segments) > 1:
            worker_args = [(seg, actual_human_height, args.robot) for seg in segments]
            t0 = time.time()
            with Pool(num_workers) as pool:
                results = pool.map(retarget_segment, worker_args)
            print(f"[info] Parallel retargeting done in {time.time() - t0:.1f}s with {num_workers} workers")
            qpos_list = []
            for seg_qpos in results:
                qpos_list.extend(seg_qpos)
        else:
            # Single-worker sequential
            retarget = GMR(
                actual_human_height=actual_human_height,
                src_human="smplx",
                tgt_robot=args.robot,
                verbose=False,
            )
            qpos_list = []
            t0 = time.time()
            for frame in smplx_data_frames:
                qpos, _ = retarget.retarget(frame)
                qpos_list.append(qpos.copy())
            print(f"[info] Sequential retargeting done in {time.time() - t0:.1f}s")
    else:
        # Interactive mode: sequential retargeting with visualization
        retarget = GMR(
            actual_human_height=actual_human_height,
            src_human="smplx",
            tgt_robot=args.robot,
            verbose=False,
        )

        robot_motion_viewer = RobotMotionViewer(
            robot_type=args.robot,
            motion_fps=aligned_fps,
            transparent_robot=0,
            record_video=args.record_video,
            video_path=f"videos/{args.robot}_{os.path.splitext(os.path.basename(args.smplx_file))[0]}.mp4",
        )

        qpos_list = []
        fps_counter = 0
        fps_start   = time.time()
        i = 0

        while True:
            if args.loop:
                i = (i + 1) % len(smplx_data_frames)
            else:
                i += 1
                if i >= len(smplx_data_frames):
                    break

            fps_counter += 1
            now = time.time()
            if now - fps_start >= 2.0:
                print(f"Retargeting FPS: {fps_counter / (now - fps_start):.1f}")
                fps_counter = 0
                fps_start   = now

            qpos, _ = retarget.retarget(smplx_data_frames[i])

            robot_motion_viewer.step(
                root_pos=qpos[:3],
                root_rot=qpos[3:7],
                dof_pos=qpos[7:],
                human_motion_data=None,
                rate_limit=args.rate_limit,
            )

            qpos_list.append(qpos.copy())

        robot_motion_viewer.close()

    # ── Save in bmimic format ────────────────────────────────────────────────
    if args.save_path is not None:
        qpos_arr      = np.array(qpos_list)
        root_pos      = qpos_arr[:, :3].astype(np.float32)
        root_rot_wxyz = qpos_arr[:, 3:7].astype(np.float32)
        dof_pos       = qpos_arr[:, 7:].astype(np.float32)

        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        km = KinematicsModel(str(ROBOT_XML_DICT[args.robot]), device=device)

        motion_data = build_bmimic_data(
            root_pos, root_rot_wxyz, dof_pos, aligned_fps, km, args.robot
        )

        save_path = args.save_path
        if not save_path.endswith(".npz"):
            save_path = os.path.splitext(save_path)[0] + ".npz"
        save_dir = os.path.dirname(save_path)
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)

        np.savez(save_path, **motion_data)
        print(f"Saved to {save_path}")
        for k, v in motion_data.items():
            print(f"  {k}: {v.shape}  {v.dtype}")
