"""
Visualize robot motion from a CSV file.

CSV format expected:
  Frame, root_translateX, root_translateY, root_translateZ,
  root_rotateX, root_rotateY, root_rotateZ,
  <joint>_dof ...  (one column per DOF, in MuJoCo DOF order)

Translations are assumed to be in centimeters (divide by 100 for meters).
Root rotations are Euler angles in degrees (xyz extrinsic by default).
Joint DOFs are in degrees.

Usage:
  python scripts/vis_csv_motion.py /path/to/motion.csv
  python scripts/vis_csv_motion.py /path/to/motion.csv --robot unitree_g1 --fps 30
  python scripts/vis_csv_motion.py /path/to/motion.csv --translation_scale 1.0  # if already in meters
"""

import numpy as np
import pandas as pd
import argparse
import os
from scipy.spatial.transform import Rotation
from general_motion_retargeting import RobotMotionViewer


def load_csv_motion(csv_file, translation_scale=0.01, euler_seq='xyz'):
    df = pd.read_csv(csv_file)

    print(f"\n[Info] Loaded '{os.path.basename(csv_file)}': {len(df)} frames, {len(df.columns)} columns")

    # Root position
    root_pos = df[['root_translateX', 'root_translateY', 'root_translateZ']].values.astype(np.float64)
    root_pos *= translation_scale

    # Root rotation: Euler (deg) → wxyz quaternion
    euler = df[['root_rotateX', 'root_rotateY', 'root_rotateZ']].values
    xyzw = Rotation.from_euler(euler_seq, euler, degrees=True).as_quat()  # (N, 4) xyzw
    root_rot = xyzw[:, [3, 0, 1, 2]]  # → wxyz

    # Joint DOFs: degrees → radians, in MuJoCo DOF order (no reordering)
    dof_cols = [c for c in df.columns if c.endswith('_dof')]
    print(f"[Info] DOF columns ({len(dof_cols)}): {dof_cols}")
    dof_pos = np.deg2rad(df[dof_cols].values).astype(np.float32)

    return root_pos, root_rot, dof_pos


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('csv_file', type=str, help='Path to motion CSV file')
    parser.add_argument('--robot', type=str, default='unitree_g1',
                        help='Robot type (default: unitree_g1)')
    parser.add_argument('--fps', type=float, default=30,
                        help='Playback FPS (default: 30)')
    parser.add_argument('--translation_scale', type=float, default=0.01,
                        help='Scale factor for translations (default: 0.01 for cm→m; use 1.0 if already in meters)')
    parser.add_argument('--euler_seq', type=str, default='xyz',
                        help='Euler angle sequence for scipy (default: xyz)')
    parser.add_argument('--record_video', action='store_true',
                        help='Record visualization to video')
    parser.add_argument('--video_path', type=str, default='videos/output.mp4',
                        help='Output video path (default: videos/output.mp4)')
    parser.add_argument('--azimuth', type=float, default=90,
                        help='Camera azimuth in degrees (default: 90)')
    parser.add_argument('--white_background', action='store_true',
                        help='Set sky and floor to white')
    args = parser.parse_args()

    if not os.path.exists(args.csv_file):
        raise FileNotFoundError(args.csv_file)

    root_pos, root_rot, dof_pos_orig = load_csv_motion(
        args.csv_file,
        translation_scale=args.translation_scale,
        euler_seq=args.euler_seq,
    )
    n_frames = len(root_pos)
    print(f"[Info] Frames={n_frames}, FPS={args.fps}, dof_dim={dof_pos_orig.shape[1]}")

    env = RobotMotionViewer(
        robot_type=args.robot,
        motion_fps=args.fps,
        camera_follow=True,
        record_video=args.record_video,
        video_path=args.video_path,
        camera_azimuth=args.azimuth,
        white_background=args.white_background,
    )

    # Auto pad/truncate dof_pos to match robot's expected DOF count
    n_robot_dof = len(env.data.qpos) - 7
    dof_pos = np.zeros((n_frames, n_robot_dof), dtype=np.float32)
    n_src = dof_pos_orig.shape[1]
    if n_src <= n_robot_dof:
        dof_pos[:, :n_src] = dof_pos_orig
        if n_src < n_robot_dof:
            print(f"[Info] Padded dof_pos from {n_src} to {n_robot_dof} DOF with zeros")
    else:
        dof_pos = dof_pos_orig[:, :n_robot_dof]
        print(f"[Info] Truncated dof_pos to {n_robot_dof} DOF")

    frame_idx = 0
    while True:
        running = env.step(root_pos[frame_idx], root_rot[frame_idx], dof_pos[frame_idx], rate_limit=True)
        if not running:
            break
        frame_idx = (frame_idx + 1) % n_frames

    env.close()


if __name__ == '__main__':
    main()
