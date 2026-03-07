"""
Compute jitter metrics comparing predicted vs ground truth robot motions.

Metrics:
1. MPJA - Mean Per-Joint Acceleration (primary jitter metric)
2. MPJJ - Mean Per-Joint Jerk (sensitive to high-frequency noise)
3. VDRR - Velocity Direction Reversal Rate (oscillation frequency)
4. HFER - High-Frequency Energy Ratio (FFT-based spectral analysis)
5. Root Translation Jitter (acceleration on root translation)

Usage:
    # Directory mode (pairs pred/gt files by index)
    python scripts/compute_jitter_metrics.py \
        --pred_dir /path/to/pred --gt_dir /path/to/gt

    # Single file mode
    python scripts/compute_jitter_metrics.py \
        --pred_file /path/to/pred.pkl --gt_file /path/to/gt.pkl

    # Save results to CSV
    python scripts/compute_jitter_metrics.py \
        --pred_dir /path/to/pred --gt_dir /path/to/gt --save_csv results.csv
"""

import argparse
import os
import pickle
import re
from pathlib import Path

import numpy as np
import torch


FPS = 50
FREQ_CUTOFF_HZ = 10.0  # cutoff for high-frequency energy ratio


def to_numpy(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def compute_mpja(q, fps):
    """Mean Per-Joint Acceleration: mean(|d²q/dt²|)."""
    if q.shape[0] < 3:
        return np.nan
    accel = np.diff(q, n=2, axis=0) * (fps ** 2)
    return np.mean(np.abs(accel))


def compute_mpjj(q, fps):
    """Mean Per-Joint Jerk: mean(|d³q/dt³|)."""
    if q.shape[0] < 4:
        return np.nan
    jerk = np.diff(q, n=3, axis=0) * (fps ** 3)
    return np.mean(np.abs(jerk))


def compute_vdrr(q, fps):
    """Velocity Direction Reversal Rate: fraction of frames where velocity sign changes."""
    if q.shape[0] < 3:
        return np.nan
    vel = np.diff(q, n=1, axis=0)  # (T-1, D)
    sign_changes = np.abs(np.diff(np.sign(vel), axis=0))  # (T-2, D), 0 or 2
    # Each sign change gives value 2, normalize to get fraction
    reversal_rate = np.mean(sign_changes > 0)
    return reversal_rate


def compute_hfer(q, fps, cutoff_hz=FREQ_CUTOFF_HZ):
    """High-Frequency Energy Ratio: energy above cutoff / total energy (FFT-based)."""
    if q.shape[0] < 4:
        return np.nan
    T, D = q.shape
    freqs = np.fft.rfftfreq(T, d=1.0 / fps)
    total_energy = 0.0
    high_energy = 0.0
    for j in range(D):
        spectrum = np.abs(np.fft.rfft(q[:, j])) ** 2
        total_energy += np.sum(spectrum)
        high_energy += np.sum(spectrum[freqs >= cutoff_hz])
    if total_energy < 1e-12:
        return 0.0
    return high_energy / total_energy


def compute_metrics_single(q, fps):
    """Compute all jitter metrics for a single signal (N, D)."""
    return {
        "mpja": compute_mpja(q, fps),
        "mpjj": compute_mpjj(q, fps),
        "vdrr": compute_vdrr(q, fps),
        "hfer": compute_hfer(q, fps),
    }


def load_pkl(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def extract_index(filename):
    """Extract numeric index from filename like '100_pred.pkl' or '100_gt.pkl'."""
    m = re.match(r"(\d+)", filename)
    return int(m.group(1)) if m else None


def process_pair(pred_path, gt_path, fps):
    """Process a single pred/gt pair and return metrics dict."""
    pred_data = load_pkl(pred_path)
    gt_data = load_pkl(gt_path)

    pred_dof = to_numpy(pred_data["pred_g1_dof"])       # (N, 29)
    gt_dof = to_numpy(gt_data["gt_g1_dof"])             # (N, 29)
    pred_trans = to_numpy(pred_data["pred_g1_trans"])    # (N, 3)
    gt_trans = to_numpy(gt_data["gt_g1_trans"])          # (N, 3)

    # Truncate to same length
    min_len = min(len(pred_dof), len(gt_dof))
    pred_dof, gt_dof = pred_dof[:min_len], gt_dof[:min_len]
    pred_trans, gt_trans = pred_trans[:min_len], gt_trans[:min_len]

    # DOF metrics
    pred_dof_metrics = compute_metrics_single(pred_dof, fps)
    gt_dof_metrics = compute_metrics_single(gt_dof, fps)

    # Root translation metrics (acceleration and jerk only)
    pred_trans_mpja = compute_mpja(pred_trans, fps)
    gt_trans_mpja = compute_mpja(gt_trans, fps)
    pred_trans_mpjj = compute_mpjj(pred_trans, fps)
    gt_trans_mpjj = compute_mpjj(gt_trans, fps)

    def safe_ratio(a, b):
        if b is None or np.isnan(b) or abs(b) < 1e-12:
            return np.nan
        return a / b

    result = {
        "num_frames": min_len,
        # DOF metrics
        "pred_dof_mpja": pred_dof_metrics["mpja"],
        "gt_dof_mpja": gt_dof_metrics["mpja"],
        "dof_mpja_ratio": safe_ratio(pred_dof_metrics["mpja"], gt_dof_metrics["mpja"]),
        "pred_dof_mpjj": pred_dof_metrics["mpjj"],
        "gt_dof_mpjj": gt_dof_metrics["mpjj"],
        "dof_mpjj_ratio": safe_ratio(pred_dof_metrics["mpjj"], gt_dof_metrics["mpjj"]),
        "pred_dof_vdrr": pred_dof_metrics["vdrr"],
        "gt_dof_vdrr": gt_dof_metrics["vdrr"],
        "dof_vdrr_ratio": safe_ratio(pred_dof_metrics["vdrr"], gt_dof_metrics["vdrr"]),
        "pred_dof_hfer": pred_dof_metrics["hfer"],
        "gt_dof_hfer": gt_dof_metrics["hfer"],
        "dof_hfer_ratio": safe_ratio(pred_dof_metrics["hfer"], gt_dof_metrics["hfer"]),
        # Root translation metrics
        "pred_trans_mpja": pred_trans_mpja,
        "gt_trans_mpja": gt_trans_mpja,
        "trans_mpja_ratio": safe_ratio(pred_trans_mpja, gt_trans_mpja),
        "pred_trans_mpjj": pred_trans_mpjj,
        "gt_trans_mpjj": gt_trans_mpjj,
        "trans_mpjj_ratio": safe_ratio(pred_trans_mpjj, gt_trans_mpjj),
        # GT motion characteristics
        "gt_dof_speed": np.mean(np.abs(np.diff(gt_dof, axis=0))) * fps if min_len > 1 else np.nan,
        "gt_dof_range": np.mean(np.ptp(gt_dof, axis=0)),
    }
    return result


def find_pairs(pred_dir, gt_dir):
    """Find matching pred/gt file pairs by index."""
    pred_files = {extract_index(f): f for f in os.listdir(pred_dir) if f.endswith(".pkl")}
    gt_files = {extract_index(f): f for f in os.listdir(gt_dir) if f.endswith(".pkl")}
    common = sorted(set(pred_files) & set(gt_files))
    pairs = []
    for idx in common:
        if idx is not None:
            pairs.append((
                os.path.join(pred_dir, pred_files[idx]),
                os.path.join(gt_dir, gt_files[idx]),
                idx,
            ))
    return pairs


def main():
    parser = argparse.ArgumentParser(description="Compute jitter metrics for pred vs GT robot motions")
    parser.add_argument("--pred_dir", type=str, help="Directory with pred pkl files")
    parser.add_argument("--gt_dir", type=str, help="Directory with GT pkl files")
    parser.add_argument("--pred_file", type=str, help="Single pred pkl file")
    parser.add_argument("--gt_file", type=str, help="Single GT pkl file")
    parser.add_argument("--fps", type=int, default=FPS, help=f"Frame rate (default: {FPS})")
    parser.add_argument("--save_csv", type=str, default=None, help="Save per-file results to CSV")
    args = parser.parse_args()

    if args.pred_file and args.gt_file:
        pairs = [(args.pred_file, args.gt_file, 0)]
    elif args.pred_dir and args.gt_dir:
        pairs = find_pairs(args.pred_dir, args.gt_dir)
    else:
        parser.error("Provide either --pred_dir/--gt_dir or --pred_file/--gt_file")

    print(f"Processing {len(pairs)} file pairs at {args.fps} FPS...")

    all_results = []
    for pred_path, gt_path, idx in pairs:
        result = process_pair(pred_path, gt_path, args.fps)
        result["file_index"] = idx
        all_results.append(result)

    # Aggregate statistics
    metric_keys = [
        ("dof_mpja_ratio", "DOF Acceleration Ratio (pred/gt)"),
        ("dof_mpjj_ratio", "DOF Jerk Ratio (pred/gt)"),
        ("dof_vdrr_ratio", "DOF Velocity Reversal Ratio"),
        ("dof_hfer_ratio", "DOF High-Freq Energy Ratio"),
        ("trans_mpja_ratio", "Root Trans Acceleration Ratio"),
        ("trans_mpjj_ratio", "Root Trans Jerk Ratio"),
    ]

    print("\n" + "=" * 65)
    print("AGGREGATE JITTER METRICS")
    print("=" * 65)
    print(f"{'Metric':<40} {'Mean':>8} {'Median':>8} {'Std':>8}")
    print("-" * 65)
    for key, label in metric_keys:
        vals = np.array([r[key] for r in all_results if not np.isnan(r[key])])
        if len(vals) > 0:
            print(f"{label:<40} {np.mean(vals):>8.3f} {np.median(vals):>8.3f} {np.std(vals):>8.3f}")

    # Absolute values
    abs_keys = [
        ("pred_dof_mpja", "Pred DOF Accel (rad/s²)"),
        ("gt_dof_mpja", "GT DOF Accel (rad/s²)"),
        ("pred_trans_mpja", "Pred Trans Accel (m/s²)"),
        ("gt_trans_mpja", "GT Trans Accel (m/s²)"),
        ("pred_dof_vdrr", "Pred DOF VDRR"),
        ("gt_dof_vdrr", "GT DOF VDRR"),
        ("pred_dof_hfer", "Pred DOF HFER"),
        ("gt_dof_hfer", "GT DOF HFER"),
    ]

    print("\n" + "=" * 65)
    print("ABSOLUTE VALUES")
    print("=" * 65)
    print(f"{'Metric':<40} {'Mean':>8} {'Median':>8} {'Std':>8}")
    print("-" * 65)
    for key, label in abs_keys:
        vals = np.array([r[key] for r in all_results if not np.isnan(r[key])])
        if len(vals) > 0:
            print(f"{label:<40} {np.mean(vals):>8.3f} {np.median(vals):>8.3f} {np.std(vals):>8.3f}")

    # Correlation analysis
    print("\n" + "=" * 65)
    print("CORRELATION WITH JITTER RATIO")
    print("=" * 65)
    jitter_ratios = np.array([r["dof_mpja_ratio"] for r in all_results])
    valid = ~np.isnan(jitter_ratios)
    corr_factors = [
        ("gt_dof_speed", "GT motion speed"),
        ("gt_dof_range", "GT motion range"),
        ("gt_dof_mpja", "GT absolute jitter"),
        ("num_frames", "Sequence length"),
    ]
    for key, label in corr_factors:
        vals = np.array([r[key] for r in all_results])
        mask = valid & ~np.isnan(vals)
        if mask.sum() > 2:
            corr = np.corrcoef(jitter_ratios[mask], vals[mask])[0, 1]
            print(f"  {label:<30} r = {corr:>7.3f}")

    # Top/bottom 5 files
    sorted_results = sorted(
        [r for r in all_results if not np.isnan(r["dof_mpja_ratio"])],
        key=lambda r: r["dof_mpja_ratio"],
        reverse=True,
    )
    print("\n" + "=" * 65)
    print("TOP 5 JITTERIEST FILES (highest DOF accel ratio)")
    print("-" * 65)
    for r in sorted_results[:5]:
        print(f"  File {r['file_index']:>4d}: DOF ratio={r['dof_mpja_ratio']:.2f}x, "
              f"Trans ratio={r['trans_mpja_ratio']:.2f}x, "
              f"GT speed={r['gt_dof_speed']:.3f}")

    print("\nBOTTOM 5 (lowest DOF accel ratio)")
    print("-" * 65)
    for r in sorted_results[-5:]:
        print(f"  File {r['file_index']:>4d}: DOF ratio={r['dof_mpja_ratio']:.2f}x, "
              f"Trans ratio={r['trans_mpja_ratio']:.2f}x, "
              f"GT speed={r['gt_dof_speed']:.3f}")

    # Save CSV
    if args.save_csv:
        import csv
        csv_keys = ["file_index", "num_frames",
                     "pred_dof_mpja", "gt_dof_mpja", "dof_mpja_ratio",
                     "pred_dof_mpjj", "gt_dof_mpjj", "dof_mpjj_ratio",
                     "pred_dof_vdrr", "gt_dof_vdrr", "dof_vdrr_ratio",
                     "pred_dof_hfer", "gt_dof_hfer", "dof_hfer_ratio",
                     "pred_trans_mpja", "gt_trans_mpja", "trans_mpja_ratio",
                     "pred_trans_mpjj", "gt_trans_mpjj", "trans_mpjj_ratio",
                     "gt_dof_speed", "gt_dof_range"]
        with open(args.save_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=csv_keys, extrasaction="ignore")
            writer.writeheader()
            for r in sorted(all_results, key=lambda x: x["file_index"]):
                writer.writerow(r)
        print(f"\nResults saved to {args.save_csv}")


if __name__ == "__main__":
    main()
