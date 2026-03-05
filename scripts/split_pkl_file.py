#!/usr/bin/env python3
"""
Script to split a pickle file containing a list of dictionaries into individual files.
Each element in the list will be saved as separate pred and gt pickle files.
"""

import pickle
import argparse
from pathlib import Path

try:
    import joblib
    HAS_JOBLIB = True
except ImportError:
    HAS_JOBLIB = False


def load_pickle_file(pkl_path):
    """Load a pickle file using pickle or joblib."""
    # Try standard pickle first
    try:
        with open(pkl_path, 'rb') as f:
            data = pickle.load(f)
        print("Loaded with standard pickle")
        return data
    except Exception as e:
        print(f"Standard pickle failed: {e}")
    
    # Try with different pickle protocols
    for encoding in ['latin1', 'ASCII', 'bytes']:
        try:
            with open(pkl_path, 'rb') as f:
                data = pickle.load(f, encoding=encoding)
            print(f"Loaded with pickle using encoding={encoding}")
            return data
        except Exception as e:
            print(f"Pickle with encoding={encoding} failed: {e}")
    
    # Try joblib if available
    if HAS_JOBLIB:
        print("Trying joblib...")
        try:
            data = joblib.load(pkl_path)
            print("Loaded with joblib")
            return data
        except Exception as e:
            print(f"Error loading with joblib: {e}")
    else:
        print("joblib not available. Try installing: pip install joblib")
    
    raise RuntimeError(f"Failed to load pickle file: {pkl_path}")


def split_pickle_file(input_pkl, output_dir):
    """
    Split a pickle file containing a list of dictionaries into individual files.
    
    Args:
        input_pkl: Path to the input pickle file
        output_dir: Directory to save the split files
    """
    # Load the pickle file
    print(f"Loading pickle file: {input_pkl}")
    data = load_pickle_file(input_pkl)
    
    # Verify the data structure
    if not isinstance(data, list):
        print(f"Error: Expected a list, but got {type(data).__name__}")
        return
    
    print(f"Loaded {len(data)} elements")
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Create subdirectories for pred and gmr``
    pred_path = output_path / "pred"
    gmr_path = output_path / "gmr"
    phuma_path = output_path / "phuma"
    gt_path = output_path / "gt"
    pred_path.mkdir(parents=True, exist_ok=True)
    gmr_path.mkdir(parents=True, exist_ok=True)
    phuma_path.mkdir(parents=True, exist_ok=True)
    gt_path.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_path}")
    print(f"  - Pred files: {pred_path}")
    print(f"  - GMR files: {gmr_path}")
    print(f"  - PHUMA files: {phuma_path}")
    print(f"  - GT files: {gt_path}")
    
    # Process each element
    for idx, element in enumerate(data):
        if not isinstance(element, dict):
            print(f"Warning: Element {idx} is not a dictionary, skipping...")
            continue
        
        # Split into pred and gt data
        pred_data = {}
        gt_data = {}
        phuma_data = {}

        source_path = element.get('source_path', None)

        for key, value in element.items():
            if key.startswith('pred_'):
                # Remove 'pred_' prefix for cleaner keys
                new_key = key  # Remove 'pred_' prefix
                pred_data[new_key] = value
            elif key.startswith('gmr_'):
                # Remove 'gmr_' prefix for cleaner keys
                new_key = key  # Remove 'gmr_' prefix
                gt_data[new_key] = value
            elif key.startswith('phuma_'):
                new_key = key
                phuma_data[new_key] = value
            elif key.startswith('gt_'):
                new_key = key
                gt_data[new_key] = value
            # else:
            #     # For keys without prefix, add to both
            #     pred_data[key] = value
            #     gt_data[key] = value

        if source_path is not None:
            for d in (pred_data, gt_data, phuma_data):
                if d:
                    d['source_path'] = source_path
        
        # Save pred file
        if pred_data:
            pred_file = pred_path / f"{idx + 1}_pred.pkl"
            with open(pred_file, 'wb') as f:
                pickle.dump(pred_data, f)
            print(f"Saved: pred/{pred_file.name}")
        
        # Save gt file
        if gt_data:
            gt_file = gmr_path / f"{idx + 1}_gmr.pkl"
            with open(gt_file, 'wb') as f:
                pickle.dump(gt_data, f)
            print(f"Saved: gmr/{gt_file.name}")
        
        if phuma_data:
            phuma_file = phuma_path / f"{idx + 1}_phuma.pkl"
            with open(phuma_file, 'wb') as f:
                pickle.dump(phuma_data, f)
            print(f"Saved: phuma/{phuma_file.name}")
        
        if gt_data:
            gt_file = gt_path / f"{idx + 1}_gt.pkl"
            with open(gt_file, 'wb') as f:
                pickle.dump(gt_data, f)
            print(f"Saved: gt/{gt_file.name}")
    
    print(f"\nCompleted! Split {len(data)} elements into {output_path}")
    print(f"Total files created: {len(list(output_path.glob('*.pkl')))}")


def main():
    parser = argparse.ArgumentParser(
        description="Split a pickle file containing a list into individual pred/gmr files"
    )
    parser.add_argument(
        "input_pkl",
        type=str,
        help="Path to the input pickle file"
    )
    parser.add_argument(
        "-o", "--output-dir",
        type=str,
        default=None,
        help="Output directory (default: same directory as input with '_split' suffix)"
    )
    
    args = parser.parse_args()
    
    input_path = Path(args.input_pkl)
    
    if not input_path.exists():
        print(f"Error: File '{input_path}' not found!")
        return
    
    # Determine output directory
    if args.output_dir:
        output_dir = args.output_dir
    else:
        # Use input file's parent directory with '_split' suffix
        output_dir = input_path.parent / f"{input_path.stem}_split"
    
    # Split the file
    split_pickle_file(input_path, output_dir)


if __name__ == "__main__":
    main()
