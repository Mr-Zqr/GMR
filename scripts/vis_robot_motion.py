"""
Unified robot motion visualization script with automatic format detection.

Prints all available keys and shapes on load, then auto-selects the right keys.
Supports single file interactive viewing and batch directory rendering.

Auto-detected formats:
  GMR standard : {root_pos, root_rot(xyzw), dof_pos[, fps]}
  G1 debug     : {g1_trans, g1_root_rot, g1_dof}
  G1 gt        : {gt_trans, gt_g1_root_ori_quat, gt_g1_dof}
  G1 joints    : {g1_joints, g1_root_ori, g1_dof}

For unknown formats use --key_root_pos / --key_root_rot / --key_dof
to specify keys manually.
"""

import numpy as np
from general_motion_retargeting import RobotMotionViewer
import argparse
import os
import glob
from tqdm import tqdm


# G1 29-DOF joint reordering used in bmimic-style datasets
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


def to_numpy(x):
    if hasattr(x, "cpu"):
        return x.cpu().numpy()
    return np.asarray(x)


def apply_joint_mapping(dof, n_dof=29):
    result = np.zeros((dof.shape[0], n_dof), dtype=np.float32)
    result[:, G1_JOINT_MAPPING] = dof
    return result


def load_motion(motion_file, fps_override=None,
                key_root_pos=None, key_root_rot=None, key_dof=None,
                joint_mapping=False):
    """
    Load motion file and return (root_pos, root_rot_wxyz, dof_pos, fps).

    Tries np.load first; falls back to joblib for .pkl files.
    Prints all key names and shapes for inspection.
    """
    ext = os.path.splitext(motion_file)[1].lower()
    try:
        data = np.load(motion_file, allow_pickle=True)
        _ = list(data.keys())  # trigger load for npz
    except Exception:
        import joblib
        data = joblib.load(motion_file)

    keys = set(data.keys())
    print(f"\n[Info] Keys in '{os.path.basename(motion_file)}':")
    for k in sorted(keys):
        v = data[k]
        shape = v.shape if hasattr(v, "shape") else type(v).__name__
        dtype = getattr(v, "dtype", "")
        print(f"  {k:40s} {str(shape):20s} {dtype}")

    fps = fps_override

    # ── Manual key override ──────────────────────────────────────────────────
    if key_root_pos and key_root_rot and key_dof:
        print(f"\n[Info] Using manual keys: {key_root_pos} / {key_root_rot} / {key_dof}")
        raw = to_numpy(data[key_root_pos])
        root_pos = raw[:, 0, :] if raw.ndim == 3 else raw
        root_rot = to_numpy(data[key_root_rot])
        dof_raw  = to_numpy(data[key_dof])
        dof_pos  = apply_joint_mapping(dof_raw) if joint_mapping else dof_raw.astype(np.float32)
        fps = fps or 30
        return root_pos, root_rot, dof_pos, fps

    # ── GMR standard {root_pos, root_rot(xyzw), dof_pos} ────────────────────
    if {"root_pos", "root_rot", "dof_pos"} <= keys:
        print("\n[Info] Detected format: GMR standard")
        root_pos = to_numpy(data["root_pos"])
        root_rot = to_numpy(data["root_rot"])[:, [3, 0, 1, 2]]  # xyzw → wxyz
        dof_pos  = to_numpy(data["dof_pos"])
        fps = fps or (float(data["fps"]) if "fps" in keys else 30)
        return root_pos, root_rot, dof_pos, fps

    # ── G1 debug {g1_trans, g1_root_rot, g1_dof} ────────────────────────────
    if {"g1_trans", "g1_root_rot", "g1_dof"} <= keys:
        print("\n[Info] Detected format: G1 debug (g1_trans / g1_root_rot / g1_dof)")
        raw = to_numpy(data["g1_trans"])
        root_pos = raw[:, 0, :] if raw.ndim == 3 else raw
        root_rot = to_numpy(data["g1_root_rot"])
        dof_raw  = to_numpy(data["g1_dof"])
        dof_pos  = apply_joint_mapping(dof_raw) if joint_mapping else dof_raw.astype(np.float32)
        fps = fps or 30
        return root_pos, root_rot, dof_pos, fps

    # ── G1 gt {gt_trans, gt_g1_root_ori_quat, gt_g1_dof} ───────────────────
    if {"gt_trans", "gt_g1_root_ori_quat", "gt_g1_dof"} <= keys:
        print("\n[Info] Detected format: G1 gt (gt_trans / gt_g1_root_ori_quat / gt_g1_dof)")
        raw = to_numpy(data["gt_trans"])
        root_pos = raw[:, 0, :] if raw.ndim == 3 else raw
        root_rot = to_numpy(data["gt_g1_root_ori_quat"])
        dof_raw  = to_numpy(data["gt_g1_dof"])
        dof_pos  = apply_joint_mapping(dof_raw) if joint_mapping else dof_raw.astype(np.float32)
        fps = fps or 30
        return root_pos, root_rot, dof_pos, fps

    # ── G1 joints {g1_joints, g1_root_ori, g1_dof} ──────────────────────────
    if {"g1_joints", "g1_root_ori", "g1_dof"} <= keys:
        print("\n[Info] Detected format: G1 joints (g1_joints / g1_root_ori / g1_dof)")
        root_pos = to_numpy(data["g1_joints"])[:, 0, :]
        root_rot = to_numpy(data["g1_root_ori"])
        dof_raw  = to_numpy(data["g1_dof"])
        dof_pos  = apply_joint_mapping(dof_raw) if joint_mapping else dof_raw.astype(np.float32)
        fps = fps or 30
        return root_pos, root_rot, dof_pos, fps

    # ── Unknown ──────────────────────────────────────────────────────────────
    raise ValueError(
        f"\nUnknown format. Keys: {sorted(keys)}\n"
        "Specify --key_root_pos, --key_root_rot, --key_dof to load manually."
    )


def run_single(motion_file, robot_type, fps_override, key_root_pos, key_root_rot,
               key_dof, joint_mapping, record_video, video_path):
    root_pos, root_rot, dof_pos, fps = load_motion(
        motion_file, fps_override, key_root_pos, key_root_rot, key_dof, joint_mapping
    )
    n_frames = len(root_pos)
    print(f"\n[Info] Frames={n_frames}, FPS={fps}, dof_dim={dof_pos.shape[1]}")

    env = RobotMotionViewer(
        robot_type=robot_type,
        motion_fps=fps,
        camera_follow=True,
        record_video=record_video,
        video_path=video_path,
    )

    frame_idx = 0
    while True:
        env.step(root_pos[frame_idx], root_rot[frame_idx], dof_pos[frame_idx], rate_limit=True)
        frame_idx = (frame_idx + 1) % n_frames

    env.close()


def run_batch(motion_dir, robot_type, fps_override, key_root_pos, key_root_rot,
              key_dof, joint_mapping, video_dir):
    files = sorted(
        glob.glob(os.path.join(motion_dir, "*.pkl")) +
        glob.glob(os.path.join(motion_dir, "*.npz"))
    )
    if not files:
        raise FileNotFoundError(f"No .pkl/.npz files in '{motion_dir}'")

    os.makedirs(video_dir, exist_ok=True)
    print(f"[Info] Found {len(files)} files → saving videos to '{video_dir}'")

    env = None
    for motion_file in tqdm(files, desc="Rendering"):
        name = os.path.splitext(os.path.basename(motion_file))[0]
        video_path = os.path.join(video_dir, f"{name}.mp4")

        try:
            root_pos, root_rot, dof_pos, fps = load_motion(
                motion_file, fps_override, key_root_pos, key_root_rot, key_dof, joint_mapping
            )
        except Exception as e:
            print(f"  [Skip] {name}: {e}")
            continue

        if env is not None and hasattr(env, "mp4_writer"):
            env.mp4_writer.close()

        if env is None:
            env = RobotMotionViewer(
                robot_type=robot_type,
                motion_fps=fps,
                camera_follow=True,
                record_video=True,
                video_path=video_path,
                headless=True,
            )
        else:
            import imageio
            env.video_path = video_path
            env.motion_fps = fps
            env.mp4_writer = imageio.get_writer(video_path, fps=fps)

        for i in range(len(root_pos)):
            env.step(root_pos[i], root_rot[i], dof_pos[i], rate_limit=False)

    if env is not None:
        env.close()
    print("\n[Done]")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)

    # Input
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--robot_motion_path", type=str,
                     help="Single motion file (.pkl or .npz)")
    src.add_argument("--robot_motion_dir", type=str,
                     help="Directory of motion files for batch rendering")

    parser.add_argument("--robot", type=str, default="unitree_g1",
                        help="Robot type (default: unitree_g1)")
    parser.add_argument("--fps", type=float, default=None,
                        help="Override playback FPS (auto-detected if omitted)")

    # Manual key override
    parser.add_argument("--key_root_pos", type=str, default=None,
                        help="Key name for root position array")
    parser.add_argument("--key_root_rot", type=str, default=None,
                        help="Key name for root rotation array")
    parser.add_argument("--key_dof", type=str, default=None,
                        help="Key name for DOF position array")

    # Joint mapping
    parser.add_argument("--joint_mapping", action="store_true", default=False,
                        help="Apply G1 joint reordering (for bmimic-style datasets)")

    # Video
    parser.add_argument("--record_video", action="store_true")
    parser.add_argument("--video_path", type=str, default="videos/output.mp4",
                        help="Output video path (single-file mode)")
    parser.add_argument("--video_dir", type=str, default="videos/batch",
                        help="Output video directory (batch mode)")

    args = parser.parse_args()

    if args.robot_motion_path:
        if not os.path.exists(args.robot_motion_path):
            raise FileNotFoundError(args.robot_motion_path)
        run_single(
            args.robot_motion_path, args.robot, args.fps,
            args.key_root_pos, args.key_root_rot, args.key_dof,
            args.joint_mapping, args.record_video, args.video_path,
        )
    else:
        if not os.path.isdir(args.robot_motion_dir):
            raise FileNotFoundError(args.robot_motion_dir)
        run_batch(
            args.robot_motion_dir, args.robot, args.fps,
            args.key_root_pos, args.key_root_rot, args.key_dof,
            args.joint_mapping, args.video_dir,
        )
