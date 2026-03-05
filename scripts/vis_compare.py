"""
One-command motion comparison script.

Given a pred/gt directory, a sequence index, and an output video folder, records:
  {idx}_pred.mp4   — predicted robot motion
  {idx}_gt.mp4     — ground truth robot motion
  {idx}_gmr.mp4    — GMR retarget motion (bmimic format)
  {idx}_smpl.mp4   — SMPL human motion

Usage:
  python scripts/vis_compare.py \
    --pred_dir /path/to/pred \
    --gt_dir /path/to/gt \
    --idx 90 \
    --video_dir /path/to/output_videos
"""

import argparse
import os
import re
import sys
import joblib
import subprocess
from pathlib import Path

SCRIPTS_DIR = Path(__file__).parent
PYTHON = sys.executable

DEFAULT_GMR_BASE = "/home/amax/devel/dataset/NeoBot/2_gmr_retarget"
DEFAULT_SMPL_BASE = "/home/amax/devel/dataset/NeoBot/1_filtered_smplx"


def extract_relative_path(source_path: str) -> str:
    """Extract relative path starting from the Motion* segment, with .npy -> .npz."""
    match = re.search(r'(Motion[^/]+/.+)', source_path)
    if not match:
        raise ValueError(
            f"Could not find Motion* segment in source_path: {source_path!r}\n"
            "Expected a path like '.../MotionLLAMA/interx/.../file.npy'"
        )
    rel = match.group(1)
    if rel.endswith('.npy'):
        rel = rel[:-4] + '.npz'
    return rel


def main():
    parser = argparse.ArgumentParser(description='Record comparison videos for pred/gt/gmr/smpl motions')
    parser.add_argument('--pred_dir', type=str, required=True, help='Directory containing {idx}_pred.pkl files')
    parser.add_argument('--gt_dir', type=str, required=True, help='Directory containing {idx}_gt.pkl files')
    parser.add_argument('--idx', type=int, required=True, help='Sequence index (e.g. 90)')
    parser.add_argument('--video_dir', type=str, required=True, help='Output directory for video files')
    parser.add_argument('--gmr_base', type=str, default=DEFAULT_GMR_BASE,
                        help=f'GMR retarget base directory (default: {DEFAULT_GMR_BASE})')
    parser.add_argument('--smpl_base', type=str, default=DEFAULT_SMPL_BASE,
                        help=f'SMPL base directory (default: {DEFAULT_SMPL_BASE})')
    args = parser.parse_args()

    idx = args.idx
    pred_file = Path(args.pred_dir) / f"{idx}_pred.pkl"
    gt_file = Path(args.gt_dir) / f"{idx}_gt.pkl"
    video_dir = Path(args.video_dir)

    # Validate input files
    if not pred_file.exists():
        print(f"ERROR: pred file not found: {pred_file}")
        sys.exit(1)
    if not gt_file.exists():
        print(f"ERROR: gt file not found: {gt_file}")
        sys.exit(1)

    # Extract source_path from pred pkl to derive gmr/smpl paths
    print(f"Loading pred pkl: {pred_file}")
    pred_data = joblib.load(pred_file)
    source_path = pred_data.get('source_path', '')
    if not source_path:
        print("ERROR: 'source_path' key not found in pred pkl")
        sys.exit(1)
    print(f"source_path: {source_path}")

    rel_path = extract_relative_path(source_path)
    gmr_path = Path(args.gmr_base) / rel_path
    smpl_path = Path(args.smpl_base) / rel_path
    print(f"GMR path: {gmr_path}")
    print(f"SMPL path: {smpl_path}")

    if not gmr_path.exists():
        print(f"WARNING: GMR file not found: {gmr_path}")
    if not smpl_path.exists():
        print(f"WARNING: SMPL file not found: {smpl_path}")

    os.makedirs(video_dir, exist_ok=True)

    cmds = [
        {
            'label': 'pred',
            'cmd': [
                PYTHON, str(SCRIPTS_DIR / "vis_robot_motion.py"),
                "--robot_motion_path", str(pred_file),
                "--key_root_pos", "pred_g1_trans",
                "--key_root_rot", "pred_g1_root_ori_quat",
                "--key_dof", "pred_g1_dof",
                "--joint_mapping",
                "--record_video", "--video_path", str(video_dir / f"{idx}_pred.mp4"),
            ],
        },
        {
            'label': 'gt',
            'cmd': [
                PYTHON, str(SCRIPTS_DIR / "vis_robot_motion.py"),
                "--robot_motion_path", str(gt_file),
                "--key_root_pos", "gt_g1_trans",
                "--key_root_rot", "gt_g1_root_ori_quat",
                "--key_dof", "gt_g1_dof",
                "--joint_mapping",
                "--record_video", "--video_path", str(video_dir / f"{idx}_gt.mp4"),
            ],
        },
        {
            'label': 'gmr',
            'cmd': [
                PYTHON, str(SCRIPTS_DIR / "vis_robot_motion_bmimic.py"),
                "--robot_motion_path", str(gmr_path),
                "--record_video", "--video_path", str(video_dir / f"{idx}_gmr.mp4"),
            ],
        },
        {
            'label': 'smpl',
            'cmd': [
                PYTHON, str(SCRIPTS_DIR / "main_humanoid.py"),
                "--smplx_file", str(smpl_path),
                "--record_video", "--video_path", str(video_dir / f"{idx}_smpl.mp4"),
            ],
        },
    ]

    for entry in cmds:
        label = entry['label']
        cmd = entry['cmd']
        print(f"\n--- Recording {label} ---")
        print(" ".join(str(c) for c in cmd))
        try:
            result = subprocess.run(cmd)
        except KeyboardInterrupt:
            # Ctrl+C closed the viewer window — treat as normal exit, continue to next
            print(f"  (interrupted, continuing)")
            result = None
        if result is not None and result.returncode not in (0, -2):
            print(f"WARNING: {label} recording exited with code {result.returncode}")

    print(f"\nDone. Videos written to: {video_dir}")
    for label in ['pred', 'gt', 'gmr', 'smpl']:
        p = video_dir / f"{idx}_{label}.mp4"
        status = "OK" if p.exists() else "MISSING"
        print(f"  {p.name}: {status}")


if __name__ == '__main__':
    main()
