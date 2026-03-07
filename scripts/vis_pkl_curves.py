"""
Interactive PKL curve visualizer with Plotly.

Load any number of pkl files, select keys and dimensions to plot interactively.
Supports zoom, pan, and hover on all axes.

Usage:
    python scripts/vis_pkl_curves.py \
        /home/amax/Documents/test_results_split/pred/2_pred.pkl \
        /home/amax/Documents/test_results_split/gt/2_gt.pkl

    # NPZ files are also supported:
    python scripts/vis_pkl_curves.py \
        /home/amax/devel/dataset/NeoBot/2_gmr_retarget/MotionGV/folder8/490248.npz

    # Mix pkl and npz freely:
    python scripts/vis_pkl_curves.py file1.pkl file2.npz

    # Auto-open in browser (default). Use --no-browser to just save HTML.
    python scripts/vis_pkl_curves.py file1.pkl file2.pkl --no-browser
"""

import argparse
import pickle
import sys
from pathlib import Path

import numpy as np
import torch


def to_numpy(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy().astype(np.float64)
    return np.asarray(x, dtype=np.float64)


def load_pkl(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def load_npz(path):
    """Load a .npz file and return a plain dict of numpy arrays."""
    npz = np.load(path, allow_pickle=False)
    return {k: npz[k] for k in npz.files}


def load_file(path):
    """Load a .pkl or .npz file, dispatch by extension."""
    suffix = Path(path).suffix.lower()
    if suffix == ".npz":
        return load_npz(path)
    else:  # .pkl / .pickle / default
        return load_pkl(path)


def get_array_keys(data):
    """Return keys whose values are numeric arrays with ndim >= 1."""
    keys = []
    for k, v in data.items():
        if isinstance(v, (np.ndarray, torch.Tensor)):
            arr = to_numpy(v)
            if arr.ndim >= 1 and np.issubdtype(arr.dtype, np.number):
                keys.append(k)
    return sorted(keys)


def pick_multi(prompt, options):
    """Let user pick multiple items from a list. Returns list of selected indices."""
    for i, opt in enumerate(options):
        print(f"  [{i}] {opt}")
    print(f"  [a] All")
    raw = input(f"{prompt} (comma-separated, e.g. 0,2,3 or 'a' for all): ").strip()
    if raw.lower() == "a":
        return list(range(len(options)))
    indices = []
    for part in raw.split(","):
        part = part.strip()
        if "-" in part:
            lo, hi = part.split("-", 1)
            indices.extend(range(int(lo), int(hi) + 1))
        else:
            indices.append(int(part))
    return [i for i in indices if 0 <= i < len(options)]


def pick_dimensions(shape):
    """Let user pick which dimensions to plot for a multi-dim array.
    shape is the shape excluding the time axis (axis 0).
    Returns list of tuples representing dimension indices."""
    if len(shape) == 0:
        return [()]  # scalar per frame
    total = int(np.prod(shape))
    if total <= 32:
        labels = []
        indices = []
        for idx in np.ndindex(*shape):
            label = ",".join(str(i) for i in idx)
            labels.append(f"dim[{label}]")
            indices.append(idx)
        sel = pick_multi(f"  Select dimensions ({total} total)", labels)
        return [indices[i] for i in sel]
    else:
        print(f"  Shape {shape} has {total} elements. Enter indices manually.")
        raw = input("  Dimensions (e.g. '0,1,5' for 1D, or '0:0,1:2' for multi-D): ").strip()
        if raw.lower() == "a":
            return [idx for idx in np.ndindex(*shape)]
        result = []
        for part in raw.split(","):
            part = part.strip()
            if ":" in part:
                idx = tuple(int(x) for x in part.split(":"))
            else:
                idx = (int(part),)
            result.append(idx)
        return result


def main():
    parser = argparse.ArgumentParser(description="Interactive PKL curve visualizer")
    parser.add_argument("files", nargs="+", help="PKL or NPZ files to load")
    parser.add_argument("--no-browser", action="store_true", help="Don't auto-open browser")
    parser.add_argument("--output", type=str, default=None, help="Output HTML path")
    args = parser.parse_args()

    # Load all files
    file_data = {}
    for fpath in args.files:
        name = Path(fpath).stem
        data = load_file(fpath)
        file_data[name] = data
        keys = get_array_keys(data)
        print(f"\n{name}: {len(keys)} array keys")
        for k in keys:
            arr = to_numpy(data[k])
            print(f"  {k}: shape={arr.shape} dtype={arr.dtype}")

    # Collect all unique keys across files
    all_keys = sorted(set(k for d in file_data.values() for k in get_array_keys(d)))
    if not all_keys:
        print("No plottable array keys found.")
        sys.exit(1)

    # Select keys
    print("\nAvailable keys across all files:")
    sel_key_indices = pick_multi("Select keys to plot", all_keys)
    selected_keys = [all_keys[i] for i in sel_key_indices]

    if not selected_keys:
        print("No keys selected.")
        sys.exit(1)

    # For each key, select dimensions
    # Group by key -> list of (file_name, array)
    traces_config = []  # list of (label, array_1d)

    for key in selected_keys:
        print(f"\nKey: {key}")
        # Find a representative shape (first file that has this key)
        rep_shape = None
        for fname, data in file_data.items():
            if key in data:
                arr = to_numpy(data[key])
                rep_shape = arr.shape[1:]  # exclude time axis
                print(f"  Shape (excl. time): {rep_shape}, frames={arr.shape[0]}")
                break

        if rep_shape is None:
            continue

        dims = pick_dimensions(rep_shape)

        for fname, data in file_data.items():
            if key not in data:
                continue
            arr = to_numpy(data[key])
            for dim_idx in dims:
                dim_label = ",".join(str(d) for d in dim_idx) if dim_idx else "scalar"
                label = f"{fname}/{key}[{dim_label}]"
                series = arr[(slice(None),) + dim_idx]
                traces_config.append((label, series))

    if not traces_config:
        print("No traces to plot.")
        sys.exit(1)

    print(f"\nPlotting {len(traces_config)} traces...")

    # Build plotly figure
    try:
        import plotly.graph_objects as go
    except ImportError:
        print("plotly not installed. Install with: pip install plotly")
        sys.exit(1)

    fig = go.Figure()
    for label, series in traces_config:
        frames = np.arange(len(series))
        fig.add_trace(go.Scatter(
            x=frames, y=series, mode="lines", name=label,
        ))

    fig.update_layout(
        title="PKL Curve Viewer",
        xaxis_title="Frame",
        yaxis_title="Value",
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        xaxis=dict(rangeslider=dict(visible=True)),  # range slider for zoom
    )

    # Enable scroll zoom
    config = {"scrollZoom": True}

    if args.output:
        out_path = args.output
    else:
        out_path = "/tmp/vis_pkl_curves.html"

    fig.write_html(out_path, config=config, auto_open=not args.no_browser)
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
