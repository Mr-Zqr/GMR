#!/usr/bin/env python3
"""
对waist三个自由度进行零相位滤波
并替换原始数据，保存到新的npz文件

Waist DOF indices: 2, 5, 8 in joint_pos

用法: python filter_waist_dof.py
"""

import numpy as np
from scipy.signal import butter, filtfilt
from pathlib import Path
import matplotlib.pyplot as plt

# 设置中文字体支持
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'SimHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False


def butter_lowpass_filter(data, cutoff, fs, order=4):
    """
    使用Butterworth低通滤波器进行零相位滤波
    
    Args:
        data: 输入信号
        cutoff: 截止频率 (Hz)
        fs: 采样频率 (Hz)
        order: 滤波器阶数
    
    Returns:
        filtered_data: 滤波后的信号
    """
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    filtered_data = filtfilt(b, a, data)  # 零相位滤波
    return filtered_data


def filter_waist_dof(input_npz, output_npz=None, waist_indices=[2, 5, 8], 
                     cutoff_freq=5.0, filter_order=4, plot_comparison=True):
    """
    对waist DOF进行滤波并保存到新文件
    
    Args:
        input_npz: 输入npz文件路径
        output_npz: 输出npz文件路径，如果为None则自动生成
        waist_indices: waist DOF在joint_pos中的索引
        cutoff_freq: 滤波器截止频率 (Hz)
        filter_order: 滤波器阶数
        plot_comparison: 是否绘制滤波前后对比图
    """
    # 检查文件是否存在
    input_path = Path(input_npz)
    if not input_path.exists():
        print(f"错误: 文件 {input_npz} 不存在")
        return False
    
    print(f"正在加载文件: {input_npz}")
    
    # 加载数据
    data = np.load(input_npz, allow_pickle=True)
    
    # 获取fps
    fps = float(data['fps']) if 'fps' in data.files else 50.0
    print(f"采样频率: {fps} Hz")
    
    # 获取joint_pos数据
    if 'joint_pos' in data.files:
        joint_pos = data['joint_pos'].copy()  # 复制一份，避免修改原数据
        data_key = 'joint_pos'
    elif 'dof_pos' in data.files:
        joint_pos = data['dof_pos'].copy()
        data_key = 'dof_pos'
    else:
        print("错误: 文件中没有joint_pos或dof_pos数据")
        return False
    
    print(f"数据形状: {joint_pos.shape}")
    print(f"总帧数: {len(joint_pos)}")
    print(f"时长: {len(joint_pos)/fps:.2f}秒")
    
    # 保存原始waist DOF数据用于对比
    original_waist_data = joint_pos[:, waist_indices].copy()
    
    # 对每个waist DOF进行滤波
    print(f"\n应用Butterworth低通滤波器:")
    print(f"  截止频率: {cutoff_freq} Hz")
    print(f"  滤波器阶数: {filter_order}")
    print(f"  方法: 零相位滤波 (filtfilt)")
    
    for i, idx in enumerate(waist_indices, 1):
        original_data = joint_pos[:, idx].copy()  # 保存原始数据的副本
        filtered_data = butter_lowpass_filter(joint_pos[:, idx], cutoff_freq, fps, filter_order)
        joint_pos[:, idx] = filtered_data
        
        # 计算滤波效果
        diff = original_data - filtered_data
        print(f"\nWaist DOF {i} (索引 {idx}):")
        print(f"  原始范围: [{np.min(original_data):.4f}, {np.max(original_data):.4f}] rad")
        print(f"  滤波范围: [{np.min(filtered_data):.4f}, {np.max(filtered_data):.4f}] rad")
        print(f"  最大差值: {np.max(np.abs(diff)):.6f} rad ({np.rad2deg(np.max(np.abs(diff))):.3f} deg)")
        print(f"  均方根差: {np.sqrt(np.mean(diff**2)):.6f} rad ({np.rad2deg(np.sqrt(np.mean(diff**2))):.3f} deg)")
    
    # 准备保存数据
    save_dict = {}
    for key in data.files:
        if key == data_key:
            save_dict[key] = joint_pos  # 使用滤波后的数据
        else:
            save_dict[key] = data[key]  # 保持其他数据不变
    
    # 确定输出文件路径
    if output_npz is None:
        output_npz = input_path.parent / f"{input_path.stem}_filtered.npz"
    else:
        output_npz = Path(output_npz)
    
    # 保存到新文件
    print(f"\n正在保存到: {output_npz}")
    np.savez(output_npz, **save_dict)
    print("保存成功!")
    
    # 绘制对比图
    if plot_comparison:
        plot_filter_comparison(original_waist_data, joint_pos[:, waist_indices], 
                              waist_indices, fps, output_npz.parent / f"{output_npz.stem}_comparison.png")
    
    return True


def plot_filter_comparison(original_data, filtered_data, waist_indices, fps, save_path):
    """
    绘制滤波前后的对比图
    
    Args:
        original_data: 原始数据 (N, 3)
        filtered_data: 滤波后数据 (N, 3)
        waist_indices: waist DOF索引
        fps: 采样频率
        save_path: 保存路径
    """
    time = np.arange(len(original_data)) / fps
    
    fig, axes = plt.subplots(3, 1, figsize=(14, 10))
    fig.suptitle('Waist DOF Filtering Comparison (Zero-Phase Filter)', 
                 fontsize=16, fontweight='bold')
    
    dof_names = ['Waist DOF 1 (Roll)', 'Waist DOF 2 (Pitch)', 'Waist DOF 3 (Yaw)']
    colors_orig = ['#1f77b4', '#ff7f0e', '#2ca02c']
    colors_filt = ['#d62728', '#9467bd', '#8c564b']
    
    for i, (ax, name, c_orig, c_filt, idx) in enumerate(zip(axes, dof_names, colors_orig, colors_filt, waist_indices)):
        orig = original_data[:, i]
        filt = filtered_data[:, i]
        
        # 绘制原始和滤波后的数据
        ax.plot(time, orig, color=c_orig, linewidth=1.0, alpha=0.6, label='Original')
        ax.plot(time, filt, color=c_filt, linewidth=1.5, label='Filtered')
        
        # 设置标签和标题
        ax.set_xlabel('Time (s)', fontsize=10)
        ax.set_ylabel('Angle (rad)', fontsize=10)
        ax.set_title(f'{name} (Index: {idx})', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper right', fontsize=9)
        ax.tick_params(axis='both', labelsize=9)
        
        # 添加零线
        ax.axhline(y=0, color='gray', linestyle='--', linewidth=0.8, alpha=0.5)
        
        # 计算并显示统计信息
        diff = orig - filt
        text_str = f'Max diff: {np.max(np.abs(diff)):.4f} rad ({np.rad2deg(np.max(np.abs(diff))):.2f} deg)\n'
        text_str += f'RMS diff: {np.sqrt(np.mean(diff**2)):.4f} rad ({np.rad2deg(np.sqrt(np.mean(diff**2))):.2f} deg)'
        ax.text(0.02, 0.02, text_str, transform=ax.transAxes, 
                fontsize=8, verticalalignment='bottom',
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"对比图已保存至: {save_path}")
    plt.close()


if __name__ == "__main__":
    # 输入文件路径
    input_npz = "/home/amax/devel/dataset/jiaqi_kuailechongbai_bmimic_follow/jiaqi_kuailechongbai_1m30s_bmimic_follow.npz"
    
    # 输出文件路径（自动生成）
    output_npz = None  # 将自动生成为 *_filtered.npz
    
    # Waist DOF indices
    waist_indices = [2, 5, 8]
    
    # 滤波器参数
    cutoff_freq = 3.0  # 截止频率 (Hz) - 降低以产生更明显的平滑效果
    filter_order = 4    # 滤波器阶数
    
    # 执行滤波
    success = filter_waist_dof(
        input_npz=input_npz,
        output_npz=output_npz,
        waist_indices=waist_indices,
        cutoff_freq=cutoff_freq,
        filter_order=filter_order,
        plot_comparison=True
    )
    
    if success:
        print("\n滤波处理完成!")
    else:
        print("\n滤波处理失败!")
