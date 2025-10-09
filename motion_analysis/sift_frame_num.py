import argparse
import os
import pickle
from natsort import natsorted
import torch
from typing import Any
from concurrent.futures import ThreadPoolExecutor
import numpy as np

import pathlib

HERE = pathlib.Path(__file__).parent

def load_pkl_file(file_path):
    with open(file_path, "rb") as f:
        data = pickle.load(f)
    return data

def analyze_data_structure(data: Any, name: str = "root", level: int = 0) -> None:
    """
    递归分析数据结构，打印详细信息
    
    Args:
        data: 要分析的数据
        name: 数据名称
        level: 缩进级别
    """
    indent = "  " * level
    
    if isinstance(data, dict):
        print(f"{indent}{name}: dict (keys: {len(data)})")
        for key, value in data.items():
            analyze_data_structure(value, f"{key}", level + 1)
    
    elif isinstance(data, (list, tuple)):
        print(f"{indent}{name}: {type(data).__name__} (length: {len(data)})")
        if len(data) > 0:
            print(f"{indent}  └─ First element:")
            analyze_data_structure(data[0], f"[0]", level + 2)
            if len(data) > 1:
                print(f"{indent}  └─ Element types: {[type(x).__name__ for x in data[:5]]}")
    
    elif isinstance(data, np.ndarray):
        print(f"{indent}{name}: numpy.ndarray")
        print(f"{indent}  ├─ Shape: {data.shape}")
        print(f"{indent}  ├─ Dtype: {data.dtype}")
        print(f"{indent}  ├─ Size: {data.size}")
        print(f"{indent}  └─ Range: [{data.min():.6f}, {data.max():.6f}]")
        
        # 如果是小数组，显示前几个值
        if data.size <= 20:
            print(f"{indent}     Values: {data.flatten()[:10]}")
    
    elif isinstance(data, torch.Tensor):
        print(f"{indent}{name}: torch.Tensor")
        print(f"{indent}  ├─ Shape: {data.shape}")
        print(f"{indent}  ├─ Dtype: {data.dtype}")
        print(f"{indent}  ├─ Device: {data.device}")
        print(f"{indent}  └─ Range: [{data.min():.6f}, {data.max():.6f}]")
        
        # 如果是小张量，显示前几个值
        if data.numel() <= 20:
            print(f"{indent}     Values: {data.flatten()[:10]}")
    
    elif isinstance(data, (int, float)):
        print(f"{indent}{name}: {type(data).__name__} = {data}")
    
    elif isinstance(data, str):
        print(f"{indent}{name}: str = '{data[:50]}{'...' if len(data) > 50 else ''}'")
    
    else:
        print(f"{indent}{name}: {type(data).__name__} = {str(data)[:100]}")

def analyze_pkl_file(file_path):
    data = load_pkl_file(file_path)
    print(f"Analyzing file: {file_path}")
    print(f"Keys in the pickle file: {list(data.keys())}")
    # Add more analysis as needed
    return data

def load_excluded_motions(exclude_file):
    """从txt文件加载需要排除的动作列表"""
    excluded_motions = set()
    if os.path.exists(exclude_file):
        with open(exclude_file, 'r') as f:
            for line in f:
                line = line.strip()
                if line:
                    excluded_motions.add(line)
        print(f"从 {exclude_file} 加载了 {len(excluded_motions)} 个排除的动作")
    return excluded_motions

def save_excluded_motions(exclude_file, motion_paths):
    """将动作路径列表保存到txt文件"""
    with open(exclude_file, 'w') as f:
        for path in motion_paths:
            f.write(f"{path}\n")
    print(f"已将 {len(motion_paths)} 个动作保存到 {exclude_file}")

def get_frame_count(file_path):
    """获取pkl文件的帧数"""
    try:
        data = load_pkl_file(file_path)
        # 尝试从不同的键中获取帧数
        if 'poses' in data:
            frame_count = len(data['poses'])
        elif 'body_pose' in data:
            frame_count = len(data['body_pose'])
        elif 'global_orient' in data:
            frame_count = len(data['global_orient'])
        else:
            # 如果都没有，尝试找第一个list或array类型的数据
            for key, value in data.items():
                if isinstance(value, (list, tuple, np.ndarray, torch.Tensor)):
                    frame_count = len(value)
                    break
            else:
                frame_count = 0
        
        return file_path, frame_count
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return file_path, 0

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--motion_folder", type=str, required=True)
    parser.add_argument("--max_workers", type=int, default=4, help="多线程数量")
    parser.add_argument("--exclude_file", type=str, default="assets/hard_motions/less_than_30.txt", help="排除动作列表文件")
    parser.add_argument("--frame_threshold", type=int, default=30, help="帧数阈值，小于此值的动作将被排除")
    args = parser.parse_args()

    src_folder = args.motion_folder
    exclude_file = args.exclude_file
    
    # 加载已排除的动作列表
    excluded_motions = load_excluded_motions(exclude_file)

    args_list = []
    excluded_count = 0
    for dirpath, _, filenames in os.walk(src_folder):
        for filename in natsorted(filenames):
            if filename.endswith("_stagei.pkl"):
                file_path = os.path.join(dirpath, filename)
                print("skip stagei file:", file_path)
                continue
            if filename.endswith((".pkl")):
                file_path = os.path.join(dirpath, filename)
                # 检查是否在排除列表中
                if file_path in excluded_motions:
                    excluded_count += 1
                    continue
                args_list.append(file_path)

    print(f"file num: {len(args_list)}, excluded: {excluded_count}")
    
    # 使用多线程加载pkl文件并获取帧数
    motion_map = {}
    
    print("正在加载pkl文件...")
    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        results = list(executor.map(get_frame_count, args_list))
    
    # 构建motion_map
    for file_path, frame_count in results:
        motion_map[file_path] = {
            'frame_count': frame_count,
            'file_path': file_path
        }
    
    print(f"加载完成！共处理 {len(motion_map)} 个文件")
    
    # 筛选帧数小于阈值的动作
    short_motions = []
    short_motion_paths = []
    for file_path, info in motion_map.items():
        if info['frame_count'] < args.frame_threshold:
            short_motions.append(info)
            short_motion_paths.append(file_path)
    
    print(f"帧数小于{args.frame_threshold}的动作有 {len(short_motions)} 个:")
    for motion in short_motions:
        print(f"  {motion['file_path']}: {motion['frame_count']} 帧")
    
    # 保存筛选出的动作到txt文件
    if short_motion_paths:
        # 合并新筛选出的动作和之前已排除的动作
        all_excluded = excluded_motions.union(set(short_motion_paths))
        save_excluded_motions(exclude_file, sorted(all_excluded))
        print(f"已将帧数小于{args.frame_threshold}的动作添加到排除列表")

    # 打印一些统计信息
    frame_counts = [info['frame_count'] for info in motion_map.values()]
    if frame_counts:
        print(f"\n统计信息:")
        print(f"  最小帧数: {min(frame_counts)}")
        print(f"  最大帧数: {max(frame_counts)}")
        print(f"  平均帧数: {sum(frame_counts) / len(frame_counts):.2f}")
        print(f"  总排除动作数: {len(all_excluded) if short_motion_paths else len(excluded_motions)}")
    
    return motion_map

if __name__ == "__main__":
    main()