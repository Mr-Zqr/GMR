#!/usr/bin/env python3
"""
Find the motion file with the maximum root translation velocity in a directory of pkl files.

Velocity = diff(trans) * fps, speed = L2 norm per frame.

Supports multiple pkl formats:
  - gt:   gt_g1_trans
  - pred: pred_g1_trans
  - GMR:  root_pos
  - bmimic: body_pos_w[:, 0, :]

Usage:
  python scripts/find_max_joint_vel.py --dir <path> [--fps 50] [--top_n 10]
"""

import argparse
import pickle
import numpy as np
from pathlib import Path


TRANS_KEY_CANDIDATES = [
    "gt_g1_trans",
    "pred_g1_trans",
    "root_pos",
    "g1_trans",
]


def to_numpy(v):
    try:
        return v.numpy().astype(np.float64)
    except AttributeError:
        return np.array(v, dtype=np.float64)


def detect_trans_key(data: dict):
    for key in TRANS_KEY_CANDIDATES:
        if key in data:
            return key, None
    # bmimic: body_pos_w (N, 30, 3) -> take root body [:, 0, :]
    if "body_pos_w" in data:
        return "body_pos_w", 0
    return None, None


def analyze_file(pkl_path: Path, fps: float):
    try:
        with open(pkl_path, "rb") as f:
            data = pickle.load(f)
    except Exception as e:
        return None, f"load error: {e}"

    if not isinstance(data, dict):
        return None, "not a dict"

    trans_key, body_idx = detect_trans_key(data)
    if trans_key is None:
        return None, f"no trans key found (keys: {list(data.keys())})"

    trans = to_numpy(data[trans_key])
    if body_idx is not None:
        trans = trans[:, body_idx, :]  # (N, 3)

    if trans.ndim != 2 or trans.shape[1] != 3:
        return None, f"unexpected trans shape {trans.shape}"

    n_frames = trans.shape[0]
    if n_frames < 2:
        return None, "too few frames"

    vel = np.diff(trans, axis=0) * fps   # (N-1, 3)
    speed = np.linalg.norm(vel, axis=1)  # (N-1,)

    max_speed = float(speed.max())
    max_frame = int(speed.argmax())
    mean_speed = float(speed.mean())
    rms_speed = float(np.sqrt((speed ** 2).mean()))

    return {
        "file": pkl_path.name,
        "max_speed": max_speed,
        "max_frame": max_frame,
        "mean_speed": mean_speed,
        "rms_speed": rms_speed,
        "n_frames": n_frames,
        "trans_key": trans_key,
    }, None


def main():
    parser = argparse.ArgumentParser(
        description="Find pkl motion files with maximum root translation velocity."
    )
    parser.add_argument("--dir", required=True, help="Directory containing pkl files")
    parser.add_argument("--fps", type=float, default=50.0, help="Assumed FPS (default: 50)")
    parser.add_argument("--top_n", type=int, default=10, help="Number of top results to show (default: 10)")
    parser.add_argument(
        "--sort_by",
        choices=["max_speed", "mean_speed", "rms_speed"],
        default="max_speed",
        help="Metric to sort by (default: max_speed)",
    )
    args = parser.parse_args()

    search_dir = Path(args.dir)
    if not search_dir.exists():
        print(f"Error: directory not found: {search_dir}")
        return

    pkl_files = sorted(search_dir.glob("*.pkl"))
    if not pkl_files:
        print(f"No pkl files found in {search_dir}")
        return

    print(f"Found {len(pkl_files)} pkl files in {search_dir}")
    print(f"FPS: {args.fps}  |  Sort by: {args.sort_by}\n")

    results, errors = [], []
    for pkl_path in pkl_files:
        info, err = analyze_file(pkl_path, args.fps)
        if info is None:
            errors.append((pkl_path.name, err))
        else:
            results.append(info)

    if errors:
        print(f"Skipped {len(errors)} files with errors:")
        for name, err in errors[:5]:
            print(f"  {name}: {err}")
        if len(errors) > 5:
            print(f"  ... and {len(errors) - 5} more")
        print()

    if not results:
        print("No valid results.")
        return

    results.sort(key=lambda x: x[args.sort_by], reverse=True)

    top = results[: args.top_n]
    col_file = max(max(len(r["file"]) for r in top), 20)
    header = (
        f"{'Rank':<5} {'File':<{col_file}} {'MaxSpeed(m/s)':<15} "
        f"{'MeanSpeed':<12} {'RMS_Speed':<12} {'Frame':<7} {'Frames':<7}"
    )
    print(header)
    print("-" * len(header))
    for rank, r in enumerate(top, 1):
        print(
            f"{rank:<5} {r['file']:<{col_file}} "
            f"{r['max_speed']:<15.4f} {r['mean_speed']:<12.4f} "
            f"{r['rms_speed']:<12.4f} {r['max_frame']:<7} {r['n_frames']:<7}"
        )

    best = results[0]
    print()
    print("=" * 60)
    print(f"  根节点速度最大的动作文件: {best['file']}")
    print(f"  最大速度: {best['max_speed']:.4f} m/s  (第 {best['max_frame']} 帧)")
    print(f"  平均速度: {best['mean_speed']:.4f} m/s")
    print(f"  RMS 速度: {best['rms_speed']:.4f} m/s")
    print(f"  总帧数: {best['n_frames']}  Trans键: {best['trans_key']}")
    print("=" * 60)


if __name__ == "__main__":
    main()
