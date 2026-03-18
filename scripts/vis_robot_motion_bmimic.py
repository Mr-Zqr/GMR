import numpy as np
from general_motion_retargeting import RobotMotionViewer
import argparse
import os
import joblib
import torch

def load_robot_motion(motion_file):
    """
    Load robot motion data from a pickle file.
    """
    with open(motion_file, "rb") as f:
        # motion_data = joblib.load(f)        
        motion_data = np.load(f, allow_pickle=True)
        motion_fps = 50
        # motion_id = 820
        # print(motion_data[motion_id]["caption"])
        # motion_root_pos = motion_data["root_pos"]
        motion_root_pos = motion_data["body_pos_w"][:, 0, :]
        motion_root_rot = motion_data["body_quat_w"][:, 0, :] 
        dof_indices = list(range(0, 20)) + list(range(23, 26))
        joint_mapping = list([0, 6, 12,
                          1, 7, 13,
                          2, 8, 14,
                          3, 9    , 15, 22,
                          4, 10   , 16, 23,
                          5, 11   , 17, 24,
                                    18, 25,
                                    19, 26,
                                    20, 27,
                                    21, 28])
        motion_dof_pos = np.zeros((motion_data["body_quat_w"].shape[0], 29), dtype = np.float32)
        motion_dof_pos[:, joint_mapping] = motion_data["joint_pos"]
        motion_body_pos = motion_data["body_pos_w"]
        motion_body_rot = motion_data["body_quat_w"]
    return motion_data, motion_fps, motion_root_pos, motion_root_rot, motion_dof_pos, motion_body_pos, motion_body_rot


def collect_motion_files(input_dir, ext=".npz"):
    """
    Recursively collect all motion files under input_dir.
    Subdirectories are traversed in sorted order, files within each
    directory are also sorted to ensure deterministic ordering.
    Returns a list of absolute file paths.
    """
    collected = []
    for root, dirs, files in os.walk(input_dir):
        dirs.sort()  # sort subdirectory traversal order in-place
        for fname in sorted(files):
            if fname.endswith(ext):
                collected.append(os.path.join(root, fname))
    return collected


def process_single_file(robot_type, motion_file, video_path, camera_azimuth=90):
    """
    Retarget and export one motion file to a video (headless).
    """
    motion_data, motion_fps, motion_root_pos, motion_root_rot, motion_dof_pos, \
        motion_local_body_pos, motion_link_body_list = load_robot_motion(motion_file)

    env = RobotMotionViewer(
        robot_type=robot_type,
        motion_fps=motion_fps,
        camera_follow=True,
        record_video=True,
        video_path=video_path,
        headless=True,
        camera_azimuth=camera_azimuth,
    )

    num_frames = len(motion_root_pos)
    for frame_idx in range(num_frames):
        running = env.step(
            motion_root_pos[frame_idx],
            motion_root_rot[frame_idx],
            motion_dof_pos[frame_idx],
            rate_limit=False,
        )
        if not running:
            break
    env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--robot", type=str, default="unitree_g1")

    # Single-file mode
    parser.add_argument("--robot_motion_path", type=str, default=None,
                        help="Path to a single motion .npz file.")

    # Directory mode
    parser.add_argument("--input_dir", type=str, default=None,
                        help="Directory containing motion .npz files to batch-export as videos.")
    parser.add_argument("--video_dir", type=str, default=None,
                        help="Output directory for batch-exported videos (mirrors input_dir structure). "
                             "Defaults to <input_dir>_videos.")

    # Single-file video options
    parser.add_argument("--record_video", action="store_true")
    parser.add_argument("--video_path", type=str,
                        default="videos/example.mp4")
    parser.add_argument("--azimuth", type=float, default=90,
                        help="Camera azimuth in degrees (90=right side, 180=front, 270=left, 0=back)")

    args = parser.parse_args()

    robot_type = args.robot

    # ------------------------------------------------------------------ #
    # Directory batch mode                                                 #
    # ------------------------------------------------------------------ #
    if args.input_dir is not None:
        input_dir = os.path.abspath(args.input_dir)
        if not os.path.isdir(input_dir):
            raise NotADirectoryError(f"input_dir not found: {input_dir}")

        video_dir = args.video_dir if args.video_dir is not None else input_dir + "_videos"
        video_dir = os.path.abspath(video_dir)

        motion_files = collect_motion_files(input_dir)
        if not motion_files:
            raise FileNotFoundError(f"No .npz files found under {input_dir}")

        print(f"Found {len(motion_files)} motion file(s) under '{input_dir}'.")
        print(f"Videos will be saved to '{video_dir}'.")

        for i, motion_file in enumerate(motion_files):
            # Compute output video path that mirrors the input directory structure.
            rel_path = os.path.relpath(motion_file, input_dir)
            rel_video = os.path.splitext(rel_path)[0] + ".mp4"
            video_path = os.path.join(video_dir, rel_video)

            print(f"[{i + 1}/{len(motion_files)}] {rel_path} -> {rel_video}")
            try:
                process_single_file(robot_type, motion_file, video_path, camera_azimuth=args.azimuth)
            except Exception as e:
                print(f"  [ERROR] Skipped due to exception: {e}")

        print("Batch export complete.")

    # ------------------------------------------------------------------ #
    # Single-file mode (original behaviour)                               #
    # ------------------------------------------------------------------ #
    else:
        if args.robot_motion_path is None:
            raise ValueError("Provide either --robot_motion_path or --input_dir.")

        robot_motion_path = args.robot_motion_path
        if not os.path.exists(robot_motion_path):
            raise FileNotFoundError(f"Motion file not found: {robot_motion_path}")

        motion_data, motion_fps, motion_root_pos, motion_root_rot, motion_dof_pos, \
            motion_local_body_pos, motion_link_body_list = load_robot_motion(robot_motion_path)

        env = RobotMotionViewer(
            robot_type=robot_type,
            motion_fps=motion_fps,
            camera_follow=True,
            record_video=args.record_video,
            video_path=args.video_path,
            camera_azimuth=args.azimuth,
        )

        num_frames = len(motion_root_pos)
        for frame_idx in range(num_frames):
            running = env.step(
                motion_root_pos[frame_idx],
                motion_root_rot[frame_idx],
                motion_dof_pos[frame_idx],
                rate_limit=True,
            )
            if not running:
                break
        env.close()