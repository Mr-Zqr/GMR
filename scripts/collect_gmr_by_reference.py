#!/usr/bin/env python3
"""
Mirror a reference directory structure using GMR-retargeted NPZ files.

For each file in REFERENCE_DIR (any extension), look up a same-stem NPZ in
SEARCH_DIR, and copy it to OUTPUT_DIR preserving the reference folder layout
(with 'smpl' subfolder renamed to 'gmr').

Usage:
    python scripts/collect_gmr_by_reference.py \
        --reference_dir /data/NeoBot/momillion_selected_for_web_neobot \
        --search_dir    /data/NeoBot/2_gmr_retarget/Mirror_MotionGV/folder0 \
        --output_dir    /data/NeoBot/momillion_selected_for_web_gmr
"""

import argparse
import shutil
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--reference_dir",
                        default="/home/amax/devel/dataset/NeoBot/momillion_selected_for_web_neobot")
    parser.add_argument("--search_dir",
                        default="/home/amax/devel/dataset/NeoBot/2_gmr_retarget/Mirror_MotionGV/folder0")
    parser.add_argument("--output_dir",
                        default="/home/amax/devel/dataset/NeoBot/momillion_selected_for_web_gmr")
    parser.add_argument("--smpl_subdir", default="smpl",
                        help="Subfolder name in reference dir to rename (default: smpl)")
    parser.add_argument("--gmr_subdir", default="gmr",
                        help="Replacement subfolder name in output dir (default: gmr)")
    args = parser.parse_args()

    ref_dir    = Path(args.reference_dir)
    search_dir = Path(args.search_dir)
    out_dir    = Path(args.output_dir)

    # Build stem → path index for the search dir (recursive, only .npz)
    index = {p.stem: p for p in search_dir.rglob("*.npz")}
    print(f"Indexed {len(index)} NPZ files in '{search_dir}'")

    ref_files = sorted(ref_dir.rglob("*"))
    ref_files = [f for f in ref_files if f.is_file()]
    print(f"Found {len(ref_files)} reference files in '{ref_dir}'")

    copied = 0
    missing = []

    for ref_file in ref_files:
        stem = ref_file.stem

        if stem not in index:
            missing.append(str(ref_file))
            continue

        src = index[stem]

        # Compute relative path from reference root, then replace smpl→gmr subfolder
        rel = ref_file.relative_to(ref_dir)
        parts = list(rel.parts)
        parts = [args.gmr_subdir if p == args.smpl_subdir else p for p in parts]
        # Replace extension with .npz
        parts[-1] = stem + ".npz"

        dst = out_dir.joinpath(*parts)
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        copied += 1

    print(f"\nCopied : {copied}")
    print(f"Missing: {len(missing)}")
    if missing:
        print("Files not found in search_dir:")
        for m in missing:
            print(f"  {m}")


if __name__ == "__main__":
    main()
