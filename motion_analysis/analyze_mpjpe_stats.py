#!/usr/bin/env python3
"""
分析motion_mpjpe_stats.pkl文件，找出平均MPJPE最低和最高的文件
"""

import pickle
import os
import numpy as np
from pathlib import Path

def load_mpjpe_stats(pkl_file):
    """加载MPJPE统计数据"""
    try:
        with open(pkl_file, 'rb') as f:
            stats = pickle.load(f)
        return stats
    except Exception as e:
        print(f"加载文件时出错: {e}")
        return None

def analyze_mpjpe_stats(data):
    """分析MPJPE统计数据"""
    if data is None:
        return
    
    # 获取motion_stats部分，这里包含了每个文件的统计信息
    motion_stats = data.get('motion_stats', {})
    overall_stats = data.get('overall_stats', {})
    total_motions = data.get('total_motions', 0)
    total_frame_pairs = data.get('total_frame_pairs', 0)
    
    print("=== MPJPE 统计分析 ===")
    print(f"总motion文件数: {total_motions}")
    print(f"总帧对数: {total_frame_pairs}")
    print(f"motion_stats条目数: {len(motion_stats)}")
    
    # 显示整体统计信息
    if overall_stats:
        print("\n=== 整体统计信息 ===")
        for key, value in overall_stats.items():
            if isinstance(value, float):
                print(f"{key}: {value:.4f}")
            else:
                print(f"{key}: {value}")
    
    # 计算每个文件的平均MPJPE
    file_avg_mpjpe = {}
    
    for file_path, file_stats in motion_stats.items():
        if isinstance(file_stats, dict):
            # 假设file_stats包含统计信息，如mean, std等
            if 'mean' in file_stats:
                file_avg_mpjpe[file_path] = file_stats['mean']
            elif 'mpjpe_values' in file_stats:
                # 如果有原始值数组
                mpjpe_values = file_stats['mpjpe_values']
                if isinstance(mpjpe_values, (list, np.ndarray)) and len(mpjpe_values) > 0:
                    file_avg_mpjpe[file_path] = np.mean(mpjpe_values)
            else:
                # 尝试找到数值型的统计值
                numeric_values = [v for v in file_stats.values() if isinstance(v, (int, float))]
                if numeric_values:
                    file_avg_mpjpe[file_path] = np.mean(numeric_values)
    
    if not file_avg_mpjpe:
        print("没有找到有效的MPJPE数据")
        # 让我们检查一个示例文件的结构
        if motion_stats:
            sample_key = list(motion_stats.keys())[0]
            sample_data = motion_stats[sample_key]
            print(f"\n示例文件统计结构: {sample_key}")
            print(f"统计数据: {sample_data}")
        return
    
    # 按平均MPJPE排序
    sorted_files = sorted(file_avg_mpjpe.items(), key=lambda x: x[1])
    
    print(f"\n有效分析文件数: {len(sorted_files)}")
    print(f"文件平均MPJPE的平均值: {np.mean(list(file_avg_mpjpe.values())):.4f}")
    print(f"文件平均MPJPE的标准差: {np.std(list(file_avg_mpjpe.values())):.4f}")
    print(f"最小平均MPJPE: {sorted_files[0][1]:.4f}")
    print(f"最大平均MPJPE: {sorted_files[-1][1]:.4f}")
    
    # 显示平均MPJPE最低的5个文件
    print("\n=== 平均MPJPE最低的5个文件 ===")
    for i, (file_path, avg_mpjpe) in enumerate(sorted_files[:5], 1):
        filename = Path(file_path).name
        print(f"{i}. {filename}")
        print(f"   路径: {file_path}")
        print(f"   平均MPJPE: {avg_mpjpe:.4f}")
        # 尝试获取更多统计信息
        if file_path in motion_stats:
            file_stat = motion_stats[file_path]
            if isinstance(file_stat, dict):
                for key, value in file_stat.items():
                    if key != 'mean' and isinstance(value, (int, float)):
                        print(f"   {key}: {value:.4f}" if isinstance(value, float) else f"   {key}: {value}")
        print()
    
    # 显示平均MPJPE最高的5个文件
    print("=== 平均MPJPE最高的5个文件 ===")
    for i, (file_path, avg_mpjpe) in enumerate(sorted_files[-5:], 1):
        filename = Path(file_path).name
        print(f"{i}. {filename}")
        print(f"   路径: {file_path}")
        print(f"   平均MPJPE: {avg_mpjpe:.4f}")
        # 尝试获取更多统计信息
        if file_path in motion_stats:
            file_stat = motion_stats[file_path]
            if isinstance(file_stat, dict):
                for key, value in file_stat.items():
                    if key != 'mean' and isinstance(value, (int, float)):
                        print(f"   {key}: {value:.4f}" if isinstance(value, float) else f"   {key}: {value}")
        print()
    
    return sorted_files

def main():
    # MPJPE统计文件路径
    pkl_file = "/home/zqr/devel/GMR/motion_mpjpe_stats.pkl"
    
    if not os.path.exists(pkl_file):
        print(f"错误: 文件 {pkl_file} 不存在")
        return
    
    print(f"正在加载文件: {pkl_file}")
    data = load_mpjpe_stats(pkl_file)
    
    if data is not None:
        analyze_mpjpe_stats(data)
    else:
        print("无法加载统计数据")

if __name__ == "__main__":
    main()