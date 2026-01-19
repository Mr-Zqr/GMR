import numpy as np
import pandas as pd
import argparse
import os
from pathlib import Path
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

def npz_to_csv(npz_file, output_csv=None):
    """
    Convert npz motion file to CSV format.
    
    Format:
    - Columns 0-2: base position (x, y, z)
    - Columns 3-6: base quaternion (x, y, z, w)
    - Columns 7+: joint angles
    
    Parameters:
    npz_file (str): Path to input npz file
    output_csv (str): Path to output csv file (optional, defaults to same name with .csv)
    """
    # Load npz file
    print(f"Loading: {npz_file}")
    data = np.load(npz_file, allow_pickle=True)
    
    # Extract data
    joint_pos = np.zeros((data['joint_pos'].shape[0], 29), dtype=np.float32)
    joint_pos[:, joint_mapping] = data['joint_pos']  # shape: (N, num_joints)
    root_pos = data['body_pos_w'][:, 0, :]  # shape: (N, 3)
    root_quat = data['body_quat_w'][:, 0, :]  # shape: (N, 4)
    
    num_frames = joint_pos.shape[0]
    num_joints = joint_pos.shape[1]
    
    print(f"Number of frames: {num_frames}")
    print(f"Number of joints: {num_joints}")
    print(f"Root position shape: {root_pos.shape}")
    print(f"Root quaternion shape: {root_quat.shape}")
    
    # Check quaternion format and convert wxyz to xyzw if needed
    # Assuming input is wxyz format, convert to xyzw
    if root_quat.shape[1] == 4:
        # Convert from wxyz to xyzw
        root_quat_xyzw = np.zeros_like(root_quat)
        root_quat_xyzw[:, 0] = root_quat[:, 1]  # x
        root_quat_xyzw[:, 1] = root_quat[:, 2]  # y
        root_quat_xyzw[:, 2] = root_quat[:, 3]  # z
        root_quat_xyzw[:, 3] = root_quat[:, 0]  # w
        root_quat = root_quat_xyzw
        print("Converted quaternion from wxyz to xyzw format")
    
    # Combine data: [base_xyz (3), quat_xyzw (4), joint_angles (num_joints)]
    combined_data = np.concatenate([root_pos, root_quat, joint_pos], axis=1)
    
    # Create column names
    column_names = (
        ['base_x', 'base_y', 'base_z'] +
        ['quat_x', 'quat_y', 'quat_z', 'quat_w'] +
        [f'joint_{i}' for i in range(num_joints)]
    )
    
    # Create DataFrame
    df = pd.DataFrame(combined_data, columns=column_names)
    
    # Determine output path
    if output_csv is None:
        output_csv = str(Path(npz_file).with_suffix('.csv'))
    
    # Save to CSV
    df.to_csv(output_csv, index=False)
    print(f"Saved to: {output_csv}")
    print(f"CSV shape: {df.shape}")
    print(f"Columns: {len(column_names)} (3 base_xyz + 4 quat_xyzw + {num_joints} joints)")
    
    return output_csv


def batch_convert(input_dir, output_dir=None):
    """
    Batch convert all npz files in a directory to CSV.
    
    Parameters:
    input_dir (str): Input directory containing npz files
    output_dir (str): Output directory (optional, defaults to input_dir)
    """
    input_path = Path(input_dir)
    
    if output_dir is None:
        output_path = input_path
    else:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
    
    # Find all npz files
    npz_files = list(input_path.rglob("*.npz"))
    
    if not npz_files:
        print(f"No .npz files found in {input_dir}")
        return
    
    print(f"Found {len(npz_files)} npz files")
    print("-" * 80)
    
    for i, npz_file in enumerate(npz_files, 1):
        print(f"\n[{i}/{len(npz_files)}]")
        
        # Calculate relative path and output path
        rel_path = npz_file.relative_to(input_path)
        output_file = output_path / rel_path.with_suffix('.csv')
        
        # Create output directory if needed
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            npz_to_csv(str(npz_file), str(output_file))
        except Exception as e:
            print(f"Error processing {npz_file}: {e}")
    
    print("\n" + "=" * 80)
    print(f"Batch conversion complete!")


def main():
    parser = argparse.ArgumentParser(
        description='Convert npz motion files to CSV format.\n'
                    'CSV format: base_xyz (3 cols) + quat_xyzw (4 cols) + joint_angles',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument('input', type=str,
                        help='Input npz file or directory')
    parser.add_argument('--output', type=str, default=None,
                        help='Output csv file or directory (optional)')
    parser.add_argument('--batch', action='store_true',
                        help='Batch process all npz files in input directory')
    
    args = parser.parse_args()
    
    # Check if input exists
    if not os.path.exists(args.input):
        print(f"Error: Input path does not exist: {args.input}")
        return
    
    # Batch mode or single file mode
    if args.batch or os.path.isdir(args.input):
        batch_convert(args.input, args.output)
    else:
        npz_to_csv(args.input, args.output)


if __name__ == "__main__":
    main()
