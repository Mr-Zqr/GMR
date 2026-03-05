#!/usr/bin/env python3
"""Decode pkl / npz / npy files.

Default: brief summary (key names + shapes).
--keys KEY [KEY ...]: print every row (first-dim element) of the given key(s).
--keys all: same for every key.
"""

import argparse
import pickle
import numpy as np
from pathlib import Path

try:
    import joblib; HAS_JOBLIB = True
except ImportError:
    HAS_JOBLIB = False

try:
    import torch; HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


# ── helpers ──────────────────────────────────────────────────────────────────

def brief_info(v):
    if isinstance(v, np.ndarray):
        return f"ndarray {v.shape} {v.dtype}"
    if HAS_TORCH and isinstance(v, torch.Tensor):
        return f"Tensor {tuple(v.shape)} {v.dtype}"
    if isinstance(v, (list, tuple)):
        inner = type(v[0]).__name__ if v else "empty"
        return f"{type(v).__name__}[{len(v)}] of {inner}"
    if isinstance(v, dict):
        return f"dict({len(v)} keys)"
    return f"{type(v).__name__}: {v}"


def print_summary(data):
    if isinstance(data, np.lib.npyio.NpzFile):
        print(f"NPZ  {len(data.files)} keys:")
        for k in data.files:
            print(f"  {k}: {brief_info(data[k])}")

    elif isinstance(data, np.ndarray):
        print(f"NPY  {brief_info(data)}")

    elif isinstance(data, dict):
        print(f"dict  {len(data)} keys:")
        for k, v in data.items():
            print(f"  '{k}': {brief_info(v)}")
            if isinstance(v, dict):
                for kk, vv in v.items():
                    print(f"      '{kk}': {brief_info(vv)}")

    elif isinstance(data, (list, tuple)):
        print(f"{type(data).__name__}[{len(data)}]")
        if data:
            print(f"  [0]: {brief_info(data[0])}")
            if isinstance(data[0], dict):
                for k, v in data[0].items():
                    print(f"      '{k}': {brief_info(v)}")
    else:
        print(brief_info(data))


def all_keys(data):
    if isinstance(data, np.lib.npyio.NpzFile):
        return list(data.files)
    if isinstance(data, dict):
        return list(data.keys())
    return []


def print_key_detail(data, key, n=1):
    # resolve value
    if isinstance(data, np.lib.npyio.NpzFile):
        if key not in data.files:
            print(f"  key '{key}' not found. available: {data.files}")
            return
        v = data[key]
    elif isinstance(data, dict):
        if key not in data:
            print(f"  key '{key}' not found. available: {list(data.keys())}")
            return
        v = data[key]
    else:
        print(f"  cannot index type {type(data).__name__} by key")
        return

    print(f"\n=== '{key}'  {brief_info(v)} ===")

    if isinstance(v, np.ndarray):
        if v.ndim == 0:
            print(v.item())
        else:
            rows = v if n < 0 else v[:n]
            for i, row in enumerate(rows):
                print(f"[{i}]  {row}")
            if 0 <= n < len(v):
                print(f"... ({len(v) - n} more rows)")
    elif HAS_TORCH and isinstance(v, torch.Tensor):
        rows = len(v) if n < 0 else min(n, len(v))
        for i in range(rows):
            print(f"[{i}]  {v[i]}")
        if 0 <= n < len(v):
            print(f"... ({len(v) - n} more rows)")
    elif isinstance(v, (list, tuple)):
        rows = v if n < 0 else v[:n]
        for i, item in enumerate(rows):
            print(f"[{i}]  {item}")
        if 0 <= n < len(v):
            print(f"... ({len(v) - n} more rows)")
    else:
        print(v)


# ── loading ───────────────────────────────────────────────────────────────────

def load_file(path):
    p = str(path)
    suffix = path.suffix.lower()

    if suffix in ('.npz', '.npy'):
        return np.load(p, allow_pickle=True)

    # pkl / joblib / other
    try:
        with open(p, 'rb') as f:
            return pickle.load(f)
    except Exception as e:
        if HAS_JOBLIB:
            try:
                return joblib.load(p)
            except Exception:
                pass
        raise e


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Decode pkl / npz / npy — brief summary by default"
    )
    parser.add_argument("file", help="Path to pkl / npz / npy file")
    parser.add_argument(
        "--keys", nargs="+", metavar="KEY",
        help="Print rows of these key(s). Use 'all' to dump every key."
    )
    parser.add_argument(
        "-n", type=int, default=1, metavar="N",
        help="Number of rows to print per key (default: 1, -1 for all)"
    )
    args = parser.parse_args()

    path = Path(args.file)
    if not path.exists():
        print(f"File not found: {path}")
        return

    print(f"Loading: {path}")
    data = load_file(path)

    if args.keys:
        keys = all_keys(data) if args.keys == ["all"] else args.keys
        for k in keys:
            print_key_detail(data, k, n=args.n)
    else:
        print_summary(data)


if __name__ == "__main__":
    main()
