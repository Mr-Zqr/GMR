#!/usr/bin/env python3
"""
Script to find the gt file with maximum difference in joint[:, 0, 0] between first and last frame.
"""

import pickle
import numpy as np
from pathlib import Path

def analyze_gt_files(split_dir):
    """
    Analyze all gt files and find the one with maximum xy distance difference.
    
    Args:
        split_dir: Directory containing the split pickle files
    """
    split_path = Path(split_dir)
    
    if not split_path.exists():
        print(f"Error: Directory '{split_path}' not found!")
        return
    
    # Find all gt files
    gt_files = sorted(split_path.glob("*_gt.pkl"))
    
    if not gt_files:
        print(f"No gt files found in '{split_path}'")
        return
    
    print(f"Found {len(gt_files)} gt files")
    print("=" * 80)
    
    results = []
    
    for gt_file in gt_files:
        try:
            with open(gt_file, 'rb') as f:
                data = pickle.load(f)
            
            # Extract joint data
            if 'joints' in data:
                joint = data['joints']
                
                # Get first and last frame xy distance
                first_frame_xy = np.sqrt(joint[0, 0, 0]**2 + joint[0, 0, 1]**2)
                last_frame_xy = np.sqrt(joint[-1, 0, 0]**2 + joint[-1, 0, 1]**2)
                diff = abs(last_frame_xy - first_frame_xy)
                
                file_num = gt_file.stem.split('_')[0]
                results.append({
                    'file': gt_file.name,
                    'number': int(file_num),
                    'first_frame_xy': first_frame_xy,
                    'last_frame_xy': last_frame_xy,
                    'diff': diff,
                    'shape': joint.shape
                })
                
        except Exception as e:
            print(f"Error processing {gt_file.name}: {e}")
    
    # Sort by difference (descending)
    results.sort(key=lambda x: x['diff'], reverse=True)
    
    if not results:
        print("No valid results found!")
        return []
    
    print("\nTop 10 files with largest xy distance difference:")
    print("-" * 80)
    print(f"{'Rank':<6} {'File':<15} {'First XY Dist':<15} {'Last XY Dist':<15} {'Difference':<15} {'Shape':<20}")
    print("-" * 80)
    
    for i, result in enumerate(results[:10]):
        print(f"{i+1:<6} {result['file']:<15} {result['first_frame_xy']:<15.6f} {result['last_frame_xy']:<15.6f} {result['diff']:<15.6f} {str(result['shape']):<20}")
    
    print("=" * 80)
    print(f"\n最大差距的文件编号: {results[0]['number']}")
    print(f"文件名: {results[0]['file']}")
    print(f"首帧 XY 距离: {results[0]['first_frame_xy']:.6f}")
    print(f"末帧 XY 距离: {results[0]['last_frame_xy']:.6f}")
    print(f"差值: {results[0]['diff']:.6f}")
    print(f"Shape: {results[0]['shape']}")
    
    return results


if __name__ == "__main__":
    split_dir = "/home/amax/devel/dataset/251226yiheng_test/output (1)_split"
    analyze_gt_files(split_dir)
