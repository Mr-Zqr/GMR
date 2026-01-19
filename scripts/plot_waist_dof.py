#!/usr/bin/env python3
"""
绘制npz文件中waist三个自由度的值
Waist DOF indices: 2, 5, 8 in dof_pos

用法: python plot_waist_dof.py
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# 设置中文字体支持
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'SimHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

def plot_waist_dof(npz_file, waist_indices=[2, 5, 8], show_plot=True, save_fig=False):
    """
    绘制waist DOF的时序图
    
    Args:
        npz_file: npz文件路径
        waist_indices: waist DOF在dof_pos中的索引
        show_plot: 是否显示图片
        save_fig: 是否保存图片
    """
    # 检查文件是否存在
    if not Path(npz_file).exists():
        print(f"错误: 文件 {npz_file} 不存在")
        return False
    
    print(f"正在加载文件: {npz_file}")
    
    try:
        data = np.load(npz_file, allow_pickle=True)
    except Exception as e:
        print(f"加载文件时出错: {e}")
        return False
    
    # 检查可用的键
    print(f"文件中的键: {data.files}")
    
    # 检查是否有dof_pos或joint_pos数据
    if 'dof_pos' in data.files:
        dof_pos = data['dof_pos']
        data_key = 'dof_pos'
    elif 'joint_pos' in data.files:
        dof_pos = data['joint_pos']
        data_key = 'joint_pos'
    else:
        # 尝试查找其他可能的键名
        possible_keys = [k for k in data.files if 'dof' in k.lower() or 'joint' in k.lower() or 'pos' in k.lower()]
        print(f"错误: 文件中没有dof_pos或joint_pos数据")
        if possible_keys:
            print(f"可能相关的键: {possible_keys}")
        return False
    
    print(f"使用数据键: {data_key}")
    
    print(f"dof_pos形状: {dof_pos.shape}")
    print(f"总帧数: {len(dof_pos)}")
    
    # 检查索引是否有效
    if dof_pos.shape[1] <= max(waist_indices):
        print(f"错误: dof_pos只有 {dof_pos.shape[1]} 个自由度，但需要访问索引 {max(waist_indices)}")
        return False
    
    # 提取waist DOF数据
    waist_dof1 = dof_pos[:, waist_indices[0]]
    waist_dof2 = dof_pos[:, waist_indices[1]]
    waist_dof3 = dof_pos[:, waist_indices[2]]
    
    # 假设50fps (可以根据实际情况调整)
    fps = 50.0
    time = np.arange(len(dof_pos)) / fps
    
    # 打印统计信息
    print(f"\n=== Waist DOF 统计信息 ===")
    print(f"时长: {len(dof_pos)/fps:.2f}秒")
    print(f"帧率: {fps} fps")
    
    for i, idx in enumerate(waist_indices, 1):
        dof_data = dof_pos[:, idx]
        print(f"\nWaist DOF {i} (索引 {idx}):")
        print(f"  范围: [{np.min(dof_data):.4f}, {np.max(dof_data):.4f}] rad")
        print(f"  范围: [{np.rad2deg(np.min(dof_data)):.2f}, {np.rad2deg(np.max(dof_data)):.2f}] deg")
        print(f"  均值: {np.mean(dof_data):.4f} rad ({np.rad2deg(np.mean(dof_data)):.2f} deg)")
        print(f"  标准差: {np.std(dof_data):.4f} rad ({np.rad2deg(np.std(dof_data)):.2f} deg)")
    
    # 创建图形
    fig, axes = plt.subplots(3, 1, figsize=(12, 10))
    fig.suptitle('Waist DOF 时序图', fontsize=16, fontweight='bold')
    
    dof_names = ['Waist DOF 1 (Roll)', 'Waist DOF 2 (Pitch)', 'Waist DOF 3 (Yaw)']
    dof_data_list = [waist_dof1, waist_dof2, waist_dof3]
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
    
    for i, (ax, dof_data, name, color, idx) in enumerate(zip(axes, dof_data_list, dof_names, colors, waist_indices)):
        # 绘制弧度值
        ax.plot(time, dof_data, color=color, linewidth=1.5, label=f'{name} (rad)')
        
        # 添加度数的参考刻度
        ax2 = ax.twinx()
        ax2.plot(time, np.rad2deg(dof_data), color=color, linewidth=0, alpha=0)
        ax2.set_ylabel('角度 (度)', fontsize=10)
        ax2.tick_params(axis='y', labelsize=9)
        ax2.grid(False)
        
        # 设置第一个y轴
        ax.set_xlabel('时间 (秒)', fontsize=10)
        ax.set_ylabel('角度 (弧度)', fontsize=10)
        ax.set_title(f'{name} (索引: {idx})', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper left', fontsize=9)
        ax.tick_params(axis='both', labelsize=9)
        
        # 添加零线
        ax.axhline(y=0, color='gray', linestyle='--', linewidth=0.8, alpha=0.5)
        
        # 添加数值范围文本
        text_str = f'范围: [{np.min(dof_data):.3f}, {np.max(dof_data):.3f}] rad\n'
        text_str += f'      [{np.rad2deg(np.min(dof_data)):.1f}, {np.rad2deg(np.max(dof_data)):.1f}] deg'
        ax.text(0.02, 0.98, text_str, transform=ax.transAxes, 
                fontsize=8, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
    
    plt.tight_layout()
    
    # 保存图片
    if save_fig:
        output_path = Path(npz_file).parent / f"{Path(npz_file).stem}_waist_dof.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"\n图片已保存至: {output_path}")
    
    # 显示图片
    if show_plot:
        plt.show()
    
    return True

if __name__ == "__main__":
    # NPZ文件路径
    npz_file = "/home/amax/devel/dataset/jiaqi_kuailechongbai_bmimic_follow/jiaqi_kuailechongbai_1m30s_bmimic_follow.npz"
    
    # Waist DOF indices (0-indexed)
    waist_indices = [2, 5, 8]
    
    # 绘制
    success = plot_waist_dof(
        npz_file=npz_file,
        waist_indices=waist_indices,
        show_plot=True,
        save_fig=True
    )
    
    if success:
        print("\n绘制完成!")
    else:
        print("\n绘制失败!")
