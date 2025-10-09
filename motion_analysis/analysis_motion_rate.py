import argparse
import os
import pickle
from natsort import natsorted
import torch
from typing import Any, Dict, List, Tuple
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from scipy.spatial.transform import Rotation as R
from scipy import stats
import pathlib

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

def quaternion_to_rotation_matrix(quaternion):
    """将四元数转换为旋转矩阵"""
    # quaternion: (N, 4) [x, y, z, w]
    return R.from_quat(quaternion).as_matrix()

def transform_local_to_global(local_body_pos, root_pos, root_rot):
    """
    将局部坐标系下的body位置转换到全局坐标系
    Args:
        local_body_pos: (N, num_joints, 3) 局部关节位置
        root_pos: (N, 3) 根节点全局位置
        root_rot: (N, 4) 根节点全局旋转(四元数)
    Returns:
        global_body_pos: (N, num_joints, 3) 全局关节位置
    """
    N, num_joints, _ = local_body_pos.shape
    
    # 转换四元数为旋转矩阵
    rot_matrices = quaternion_to_rotation_matrix(root_rot)  # (N, 3, 3)
    
    # 将局部位置转换到全局坐标系
    # local_body_pos: (N, num_joints, 3) -> (N, num_joints, 3, 1)
    local_pos_expanded = local_body_pos[..., np.newaxis]
    
    # 应用旋转: (N, 3, 3) @ (N, num_joints, 3, 1) -> (N, num_joints, 3, 1)
    rotated_pos = np.matmul(rot_matrices[:, np.newaxis, :, :], local_pos_expanded)
    rotated_pos = rotated_pos.squeeze(-1)  # (N, num_joints, 3)
    
    # 添加根节点位置
    global_body_pos = rotated_pos + root_pos[:, np.newaxis, :]  # (N, num_joints, 3)
    
    return global_body_pos

def calculate_mpjpe(pos1, pos2):
    """
    计算两帧之间的Mean Per Joint Position Error (MPJPE)
    Args:
        pos1, pos2: (num_joints, 3) 关节位置
    Returns:
        mpjpe: float
    """
    return np.mean(np.linalg.norm(pos1 - pos2, axis=1))

def calculate_motion_statistics(mpjpe_values):
    """
    计算MPJPE统计信息
    Args:
        mpjpe_values: List[float] MPJPE值列表
    Returns:
        stats_dict: Dict 包含各项统计信息
    """
    if not mpjpe_values:
        return None
    
    mpjpe_array = np.array(mpjpe_values)
    
    # 计算众数
    try:
        mode_result = stats.mode(mpjpe_array, keepdims=True)
        mode_value = mode_result.mode[0]
    except:
        # 如果没有众数，使用最频繁的近似值
        hist, bin_edges = np.histogram(mpjpe_array, bins=20)
        mode_idx = np.argmax(hist)
        mode_value = (bin_edges[mode_idx] + bin_edges[mode_idx + 1]) / 2
    
    return {
        'min': float(np.min(mpjpe_array)),
        'max': float(np.max(mpjpe_array)),
        'mean': float(np.mean(mpjpe_array)),
        'median': float(np.median(mpjpe_array)),
        'mode': float(mode_value),
        'std': float(np.std(mpjpe_array)),
        'frame_count': len(mpjpe_values) + 1  # +1 because we calculate N-1 MPJPE values for N frames
    }

def process_motion_file(file_path):
    """
    处理单个动作文件，计算MPJPE统计信息
    Args:
        file_path: str 文件路径
    Returns:
        tuple: (file_path, stats_dict or None)
    """
    try:
        with open(file_path, 'rb') as f:
            motion_data = pickle.load(f)
        
        # 提取数据
        root_pos = motion_data['root_pos']  # (N, 3)
        root_rot = motion_data['root_rot']  # (N, 4)
        local_body_pos = motion_data['local_body_pos']  # (N, num_joints, 3)
        
        # 转换为全局坐标
        global_body_pos = transform_local_to_global(local_body_pos, root_pos, root_rot)
        
        # 计算相邻帧之间的MPJPE
        mpjpe_values = []
        for i in range(len(global_body_pos) - 1):
            mpjpe = calculate_mpjpe(global_body_pos[i], global_body_pos[i + 1])
            mpjpe_values.append(mpjpe)
        
        # 计算统计信息
        stats_dict = calculate_motion_statistics(mpjpe_values)
        
        return (file_path, stats_dict)
        
    except Exception as e:
        print(f"处理文件 {file_path} 时出错: {e}")
        return (file_path, None)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--motion_folder", type=str, required=True)
    parser.add_argument("--max_workers", type=int, default=4, help="多线程数量")
    parser.add_argument("--exclude_file", type=str, default="assets/hard_motions/less_than_30.txt", help="排除动作列表文件")
    parser.add_argument("--frame_threshold", type=int, default=30, help="帧数阈值，小于此值的动作将被排除")
    parser.add_argument("--output_file", type=str, default="motion_mpjpe_stats.pkl", help="输出统计结果文件")
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
    
    # 使用多线程处理文件
    motion_stats_map = {}
    all_mpjpe_values = []
    valid_motions = 0
    
    print("正在处理动作文件并计算MPJPE统计信息...")
    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        results = list(executor.map(process_motion_file, args_list))
    
    # 收集结果
    for file_path, stats_dict in results:
        if stats_dict is not None:
            motion_stats_map[file_path] = stats_dict
            # 收集所有MPJPE值用于总体统计
            try:
                with open(file_path, 'rb') as f:
                    motion_data = pickle.load(f)
                root_pos = motion_data['root_pos']
                root_rot = motion_data['root_rot']
                local_body_pos = motion_data['local_body_pos']
                global_body_pos = transform_local_to_global(local_body_pos, root_pos, root_rot)
                
                for i in range(len(global_body_pos) - 1):
                    mpjpe = calculate_mpjpe(global_body_pos[i], global_body_pos[i + 1])
                    all_mpjpe_values.append(mpjpe)
                valid_motions += 1
            except:
                pass
    
    # 计算总体统计信息
    overall_stats = calculate_motion_statistics(all_mpjpe_values)
    
    print(f"\n处理完成！")
    print(f"有效动作数量: {valid_motions}")
    print(f"总帧间MPJPE数量: {len(all_mpjpe_values)}")
    
    if overall_stats:
        print(f"\n总体MPJPE统计信息:")
        print(f"  最小值: {overall_stats['min']:.6f}")
        print(f"  最大值: {overall_stats['max']:.6f}")
        print(f"  平均值: {overall_stats['mean']:.6f}")
        print(f"  中位数: {overall_stats['median']:.6f}")
        print(f"  众数: {overall_stats['mode']:.6f}")
        print(f"  标准差: {overall_stats['std']:.6f}")
    
    # 保存结果
    output_data = {
        'motion_stats': motion_stats_map,
        'overall_stats': overall_stats,
        'total_motions': valid_motions,
        'total_frame_pairs': len(all_mpjpe_values)
    }
    
    with open(args.output_file, 'wb') as f:
        pickle.dump(output_data, f)
    
    print(f"\n结果已保存到: {args.output_file}")
    
    # 显示一些示例统计信息
    print(f"\n前5个动作的MPJPE统计信息:")
    for i, (file_path, stats) in enumerate(list(motion_stats_map.items())[:5]):
        print(f"{i+1}. {os.path.basename(file_path)}:")
        print(f"   帧数: {stats['frame_count']}, 平均MPJPE: {stats['mean']:.6f}")
        print(f"   范围: [{stats['min']:.6f}, {stats['max']:.6f}]")

if __name__ == "__main__":
    main()