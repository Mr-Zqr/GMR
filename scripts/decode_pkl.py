#!/usr/bin/env python3
"""
Script to decode and display the structure of pickle files.
Recursively shows keys and dimensions for nested dictionaries and arrays.
"""

import pickle
import argparse
import numpy as np
import torch
from pathlib import Path

try:
    import joblib
    HAS_JOBLIB = True
except ImportError:
    HAS_JOBLIB = False


def get_shape_info(obj):
    """Get shape/dimension information of an object."""
    if isinstance(obj, np.ndarray):
        return f"numpy.ndarray, shape: {obj.shape}, dtype: {obj.dtype}"
    elif isinstance(obj, torch.Tensor):
        return f"torch.Tensor, shape: {obj.shape}, dtype: {obj.dtype}"
    elif isinstance(obj, list):
        if len(obj) > 0:
            first_elem = obj[0]
            if isinstance(first_elem, (np.ndarray, torch.Tensor)):
                return f"list of {len(obj)} elements, first element: {get_shape_info(first_elem)}"
            else:
                return f"list of {len(obj)} elements, type: {type(first_elem).__name__}"
        else:
            return "empty list"
    elif isinstance(obj, tuple):
        if len(obj) > 0:
            return f"tuple of {len(obj)} elements"
        else:
            return "empty tuple"
    elif isinstance(obj, dict):
        return f"dict with {len(obj)} keys"
    elif isinstance(obj, (int, float, str, bool)):
        return f"{type(obj).__name__}: {obj}"
    else:
        return f"{type(obj).__name__}"


def decode_structure(obj, indent=0, max_depth=10, current_depth=0):
    """
    Recursively decode and print the structure of a Python object.
    
    Args:
        obj: The object to decode
        indent: Current indentation level
        max_depth: Maximum recursion depth
        current_depth: Current recursion depth
    """
    prefix = "  " * indent
    
    if current_depth >= max_depth:
        print(f"{prefix}[Max depth reached]")
        return
    
    if isinstance(obj, dict):
        print(f"{prefix}Dictionary with {len(obj)} keys:")
        for key, value in obj.items():
            shape_info = get_shape_info(value)
            print(f"{prefix}  '{key}': {shape_info}")
            
            # Recursively decode nested structures
            if isinstance(value, dict):
                decode_structure(value, indent + 2, max_depth, current_depth + 1)
            elif isinstance(value, (list, tuple)) and len(value) > 0:
                # If it's a list/tuple of dicts, show the first one
                if isinstance(value[0], dict):
                    print(f"{prefix}    [First element structure:]")
                    decode_structure(value[0], indent + 3, max_depth, current_depth + 1)
                    
    elif isinstance(obj, (list, tuple)):
        print(f"{prefix}{type(obj).__name__} with {len(obj)} elements:")
        # print(obj[814])
        if len(obj) > 0:
            first_elem = obj[0]
            shape_info = get_shape_info(first_elem)
            print(f"{prefix}  First element: {shape_info}")
            
            if isinstance(first_elem, dict):
                decode_structure(first_elem, indent + 1, max_depth, current_depth + 1)
                
    else:
        shape_info = get_shape_info(obj)
        print(f"{prefix}{shape_info}")


def main():
    parser = argparse.ArgumentParser(
        description="Decode and display the structure of pickle files"
    )
    parser.add_argument(
        "pkl_file",
        type=str,
        help="Path to the pickle file to decode"
    )
    parser.add_argument(
        "--max-depth",
        type=int,
        default=10,
        help="Maximum recursion depth for nested structures (default: 10)"
    )
    parser.add_argument(
        "--show-values",
        action="store_true",
        help="Show actual values for small arrays (shape <= 10)"
    )
    parser.add_argument(
        "--show-key",
        type=str,
        default=None,
        help="Show all data for a specific key (e.g., 'link_body_list')"
    )
    
    args = parser.parse_args()
    
    pkl_path = Path(args.pkl_file)
    
    if not pkl_path.exists():
        print(f"Error: File '{pkl_path}' not found!")
        return
    
    print(f"Loading pickle file: {pkl_path}")
    print("=" * 80)
    
    try:
        with open(pkl_path, 'rb') as f:
            data = pickle.load(f)
    except (ModuleNotFoundError, Exception) as e:
        # Try loading with joblib if pickle fails
        if HAS_JOBLIB:
            print(f"Standard pickle failed ({e}), trying joblib...")
            try:
                data = joblib.load(pkl_path)
            except Exception as e2:
                print(f"Error loading with joblib: {e2}")
                import traceback
                traceback.print_exc()
                return
        else:
            print(f"Error: {e}")
            print("Try installing joblib: pip install joblib")
            import traceback
            traceback.print_exc()
            return
    
    print(f"\nTop-level structure: {type(data).__name__}")
    # print(data)
    print("=" * 80)
    
    decode_structure(data, indent=0, max_depth=args.max_depth)
    
    # Optional: show specific key data
    if args.show_key:
        print("\n" + "=" * 80)
        print(f"Data for key '{args.show_key}':")
        print("=" * 80)
        show_key_data(data, args.show_key)
    
    # Optional: show small array values
    if args.show_values:
        print("\n" + "=" * 80)
        print("Small array values (shape <= 10):")
        print("=" * 80)
        show_small_values(data)


def show_small_values(obj, prefix=""):
    """Show values for small arrays."""
    if isinstance(obj, dict):
        for key, value in obj.items():
            new_prefix = f"{prefix}.{key}" if prefix else key
            if isinstance(value, (np.ndarray, torch.Tensor)):
                if np.prod(value.shape) <= 40:
                    print(f"\n{new_prefix}:")
                    print(value)
            elif isinstance(value, dict):
                show_small_values(value, new_prefix)
    elif isinstance(obj, (list, tuple)) and len(obj) > 0:
        if isinstance(obj[0], dict):
            show_small_values(obj[0], f"{prefix}[0]")


def show_key_data(obj, target_key, prefix="", found_any=False):
    """
    Recursively search for and display data for a specific key.
    
    Args:
        obj: The object to search in
        target_key: The key to search for
        prefix: Current path prefix
        found_any: Whether any matching key has been found
    
    Returns:
        bool: Whether any matching key was found
    """
    if isinstance(obj, dict):
        for key, value in obj.items():
            current_path = f"{prefix}.{key}" if prefix else key
            
            # Check if this is the target key
            if key == target_key:
                print(f"\nFound at: {current_path}")
                print(f"Type: {type(value).__name__}")
                
                if isinstance(value, (np.ndarray, torch.Tensor)):
                    print(f"Shape: {value.shape}, dtype: {value.dtype}")
                    print("Data:")
                    print(value)
                elif isinstance(value, (list, tuple)):
                    print(f"Length: {len(value)}")
                    print("Data:")
                    for i, item in enumerate(value):
                        if isinstance(item, (np.ndarray, torch.Tensor)):
                            print(f"  [{i}]: shape={item.shape}, dtype={item.dtype}")
                            print(f"       {item}")
                        else:
                            print(f"  [{i}]: {item}")
                elif isinstance(value, dict):
                    print(f"Dictionary with {len(value)} keys:")
                    print(value)
                else:
                    print("Data:")
                    print(value)
                
                found_any = True
            
            # Recursively search nested structures
            if isinstance(value, dict):
                found_any = show_key_data(value, target_key, current_path, found_any) or found_any
            elif isinstance(value, (list, tuple)) and len(value) > 0:
                if isinstance(value[0], dict):
                    for i, item in enumerate(value):
                        if isinstance(item, dict):
                            found_any = show_key_data(item, target_key, f"{current_path}[{i}]", found_any) or found_any
    
    elif isinstance(obj, (list, tuple)):
        for i, item in enumerate(obj):
            if isinstance(item, dict):
                current_path = f"{prefix}[{i}]" if prefix else f"[{i}]"
                found_any = show_key_data(item, target_key, current_path, found_any) or found_any
    
    # Print message if nothing was found (only at top level)
    if prefix == "" and not found_any:
        print(f"\nKey '{target_key}' not found in the pickle file.")
    
    return found_any


if __name__ == "__main__":
    main()
