#!/usr/bin/env python3
"""
分析motion_mpjpe_stats.pkl文件，列出平均MPJPE最低和最高的文件
作者: 用于分析GMR项目的MPJPE统计数据
"""

import pickle
import os
import numpy as np
from pathlib import Path

def main():
    # 加载MPJPE统计文件
    pkl_file = "/home/zqr/devel/dataset/humanml3d_g1/motion_analysis.pkl"
    
    if not os.path.exists(pkl_file):
        print(f"错误: 文件 {pkl_file} 不存在")
        return
    
    print(f"正在加载文件: {pkl_file}")
    
    try:
        with open(pkl_file, 'rb') as f:
            data = pickle.load(f)
    except Exception as e:
        print(f"加载文件时出错: {e}")
        return
    
    # 获取数据
    motion_stats = data.get('motion_stats', {})
    overall_mpjpe_stats = data.get('overall_mpjpe_stats', {})
    overall_velocity_stats = data.get('overall_velocity_stats', {})
    total_motions = data.get('total_motions', 0)
    total_frame_pairs = data.get('total_frame_pairs', 0)
    total_velocity_values = data.get('total_velocity_values', 0)
    
    print("=" * 60)
    print("MPJPE & 速度统计分析结果")
    print("=" * 60)
    print(f"总motion文件数: {total_motions:,}")
    print(f"总帧对数: {total_frame_pairs:,}")
    print(f"总速度值数: {total_velocity_values:,}")
    
    # 显示整体MPJPE统计信息
    if overall_mpjpe_stats:
        print(f"\n整体MPJPE统计:")
        print(f"  最小值: {overall_mpjpe_stats.get('min', 0):.6f}")
        print(f"  最大值: {overall_mpjpe_stats.get('max', 0):.4f}")
        print(f"  平均值: {overall_mpjpe_stats.get('mean', 0):.4f}")
        print(f"  中位数: {overall_mpjpe_stats.get('median', 0):.4f}")
        print(f"  众数: {overall_mpjpe_stats.get('mode', 0):.6f}")
        print(f"  标准差: {overall_mpjpe_stats.get('std', 0):.4f}")
        print(f"  总帧数: {overall_mpjpe_stats.get('frame_count', 0):,}")
    
    # 显示整体速度统计信息
    if overall_velocity_stats:
        print(f"\n整体速度统计:")
        print(f"  最小值: {overall_velocity_stats.get('min', 0):.6f}")
        print(f"  最大值: {overall_velocity_stats.get('max', 0):.4f}")
        print(f"  平均值: {overall_velocity_stats.get('mean', 0):.4f}")
        print(f"  中位数: {overall_velocity_stats.get('median', 0):.4f}")
        print(f"  众数: {overall_velocity_stats.get('mode', 0):.6f}")
        print(f"  标准差: {overall_velocity_stats.get('std', 0):.4f}")
    
    # 提取每个文件的平均MPJPE
    file_avg_mpjpe = {}
    file_avg_velocity = {}
    for file_path, file_stats in motion_stats.items():
        if isinstance(file_stats, dict) and 'mpjpe_stats' in file_stats:
            mpjpe_stats = file_stats['mpjpe_stats']
            # 只分析帧数大于600的文件
            if mpjpe_stats.get('frame_count', 0) > 10:
                file_avg_mpjpe[file_path] = mpjpe_stats['max']
                
                # 同时提取速度信息
                if 'velocity_stats' in file_stats and file_stats['root_states']['mean'] > 0.6 and file_stats['velocity_stats']['max']>1:
                    file_avg_velocity[file_path] = file_stats['velocity_stats']['max']
    
    # 按平均MPJPE排序
    sorted_files_velocity = sorted(file_avg_velocity.items(), key=lambda x: x[1])

    print(f"\n分析的有效文件数: {len(sorted_files_velocity):,}")

    # 显示最大velocity最低的10个文件
    print("\n" + "=" * 60)
    print("最大velocity最低的10个文件")
    print("=" * 60)
    # for i, (file_path, avg_velocity) in enumerate(sorted_files_velocity[int(0.95*len(sorted_files_velocity)):][:20], 1):
    for i, (file_path, avg_velocity) in enumerate(sorted_files_velocity[-20:], 1):
        filename = Path(file_path).name
        print(f"\n{i}. {filename}")
        print(f"   完整路径: {file_path}")
        print(f"   平均MPJPE: {avg_velocity:.6f}")
        
        # 显示详细统计信息
        if file_path in motion_stats:
            file_data = motion_stats[file_path]
            mpjpe_stats = file_data.get('mpjpe_stats', {})
            velocity_stats = file_data.get('velocity_stats', {})
            root_states = file_data.get('root_states', {})
            
            print(f"   帧数: {file_data.get('frame_count', 'N/A')}")
            print(f"   FPS: {file_data.get('fps', 'N/A')}")
            print(f"   MPJPE - 最小值: {mpjpe_stats.get('min', 0):.6f}, 最大值: {mpjpe_stats.get('max', 0):.4f}, 标准差: {mpjpe_stats.get('std', 0):.4f}")
            print(f"   速度 - 平均值: {velocity_stats.get('mean', 0):.4f}, 标准差: {velocity_stats.get('std', 0):.4f}")
            print(f"   根状态 - 平均值: {root_states.get('mean', 0):.4f}, 标准差: {root_states.get('std', 0):.4f}")
    
    # # 显示平均MPJPE最高的10个文件
    # print("\n" + "=" * 60)
    # print("平均MPJPE最高的10个文件")
    # print("=" * 60)
    # for i, (file_path, avg_mpjpe) in enumerate(sorted_files_velocity[-10:], 1):
    #     filename = Path(file_path).name
    #     print(f"\n{i}. {filename}")
    #     print(f"   完整路径: {file_path}")
    #     print(f"   平均MPJPE: {avg_mpjpe:.4f}")
        
    #     # 显示详细统计信息
    #     if file_path in motion_stats:
    #         file_data = motion_stats[file_path]
    #         mpjpe_stats = file_data.get('mpjpe_stats', {})
    #         velocity_stats = file_data.get('velocity_stats', {})
    #         root_states = file_data.get('root_states', {})
            
    #         print(f"   帧数: {file_data.get('frame_count', 'N/A')}")
    #         print(f"   FPS: {file_data.get('fps', 'N/A')}")
    #         print(f"   MPJPE - 最小值: {mpjpe_stats.get('min', 0):.6f}, 最大值: {mpjpe_stats.get('max', 0):.4f}, 标准差: {mpjpe_stats.get('std', 0):.4f}")
    #         print(f"   速度 - 平均值: {velocity_stats.get('mean', 0):.4f}, 标准差: {velocity_stats.get('std', 0):.4f}")
    #         print(f"   根状态 - 平均值: {root_states.get('mean', 0):.4f}, 标准差: {root_states.get('std', 0):.4f}")
    
    # 如果有速度数据，也显示速度分析
    # if file_avg_velocity:
    #     sorted_files_velocity = sorted(file_avg_velocity.items(), key=lambda x: x[1])
        
    #     print("\n" + "=" * 60)
    #     print("平均速度最低的5个文件")
    #     print("=" * 60)
    #     for i, (file_path, avg_velocity) in enumerate(sorted_files_velocity[:5], 1):
    #         filename = Path(file_path).name
    #         print(f"\n{i}. {filename}")
    #         print(f"   平均速度: {avg_velocity:.4f}")
    #         print(f"   平均MPJPE: {file_avg_mpjpe.get(file_path, 'N/A'):.6f}")
        
    #     print("\n" + "=" * 60)
    #     print("平均速度最高的5个文件")
    #     print("=" * 60)
    #     for i, (file_path, avg_velocity) in enumerate(sorted_files_velocity[-5:], 1):
    #         filename = Path(file_path).name
    #         print(f"\n{i}. {filename}")
    #         print(f"   平均速度: {avg_velocity:.4f}")
    #         print(f"   平均MPJPE: {file_avg_mpjpe.get(file_path, 'N/A'):.4f}")
    
    print("\n" + "=" * 60)
    print("分析完成")
    print("=" * 60)

if __name__ == "__main__":
    main()