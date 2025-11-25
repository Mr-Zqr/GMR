#!/usr/bin/env python3
"""
绘制motion文件的root_pos轨迹图
作者: 用于分析GMR项目的root position数据
"""

import pickle
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os
from pathlib import Path

def plot_root_pos(pkl_file, output_dir=None):
    """
    绘制root_pos的轨迹图
    
    Args:
        pkl_file: pkl文件路径
        output_dir: 输出目录，如果None则不保存
    """
    # 检查文件是否存在
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
    
    # 检查是否有root_pos数据
    if 'root_pos' not in data:
        print("错误: 文件中没有root_pos数据")
        return
    
    root_pos = data['root_pos']
    fps = data.get('fps', 30.0)  # 默认30fps
    
    print(f"root_pos形状: {root_pos.shape}")
    print(f"FPS: {fps}")
    print(f"总帧数: {len(root_pos)}")
    print(f"时长: {len(root_pos)/fps:.2f}秒")
    
    # 提取XYZ坐标
    x = root_pos[:, 0]
    y = root_pos[:, 1] 
    z = root_pos[:, 2]
    
    # 创建时间轴
    time = np.arange(len(root_pos)) / fps
    
    # 计算一些统计信息
    total_distance = np.sum(np.sqrt(np.diff(x)**2 + np.diff(y)**2 + np.diff(z)**2))
    max_speed = np.max(np.sqrt(np.diff(x)**2 + np.diff(y)**2 + np.diff(z)**2) * fps)
    
    print(f"总移动距离: {total_distance:.2f}米")
    print(f"最大速度: {max_speed:.2f}米/秒")
    
    # 创建图形
    fig = plt.figure(figsize=(16, 12))
    
    # 1. 3D轨迹图
    ax1 = fig.add_subplot(2, 3, 1, projection='3d')
    ax1.plot(x, y, z, 'b-', linewidth=2, alpha=0.8)
    ax1.scatter(x[0], y[0], z[0], color='green', s=100, label='Start')
    ax1.scatter(x[-1], y[-1], z[-1], color='red', s=100, label='End')
    ax1.set_xlabel('X (m)')
    ax1.set_ylabel('Y (m)')
    ax1.set_zlabel('Z (m)')
    ax1.set_title('3D Root Position Trajectory')
    ax1.legend()
    ax1.grid(True)
    
    # 2. XY平面轨迹
    ax2 = fig.add_subplot(2, 3, 2)
    ax2.plot(x, y, 'b-', linewidth=2, alpha=0.8)
    ax2.scatter(x[0], y[0], color='green', s=100, label='Start')
    ax2.scatter(x[-1], y[-1], color='red', s=100, label='End')
    ax2.set_xlabel('X (m)')
    ax2.set_ylabel('Y (m)')
    ax2.set_title('XY Plane Trajectory (Top View)')
    ax2.legend()
    ax2.grid(True)
    ax2.axis('equal')
    
    # 3. X坐标随时间变化
    ax3 = fig.add_subplot(2, 3, 3)
    ax3.plot(time, x, 'r-', linewidth=2)
    ax3.set_xlabel('Time (s)')
    ax3.set_ylabel('X Position (m)')
    ax3.set_title('X Position vs Time')
    ax3.grid(True)
    
    # 4. Y坐标随时间变化
    ax4 = fig.add_subplot(2, 3, 4)
    ax4.plot(time, y, 'g-', linewidth=2)
    ax4.set_xlabel('Time (s)')
    ax4.set_ylabel('Y Position (m)')
    ax4.set_title('Y Position vs Time')
    ax4.grid(True)
    
    # 5. Z坐标随时间变化
    ax5 = fig.add_subplot(2, 3, 5)
    ax5.plot(time, z, 'b-', linewidth=2)
    ax5.set_xlabel('Time (s)')
    ax5.set_ylabel('Z Position (m)')
    ax5.set_title('Z Position vs Time')
    ax5.grid(True)
    
    # 6. 速度随时间变化
    ax6 = fig.add_subplot(2, 3, 6)
    # 计算每帧的速度
    dx = np.diff(x)
    dy = np.diff(y)
    dz = np.diff(z)
    speed = np.sqrt(dx**2 + dy**2 + dz**2) * fps
    speed_time = time[1:]  # 速度时间轴比位置少一个点
    
    ax6.plot(speed_time, speed, 'm-', linewidth=2)
    ax6.set_xlabel('Time (s)')
    ax6.set_ylabel('Speed (m/s)')
    ax6.set_title('Root Position Speed')
    ax6.grid(True)
    
    plt.tight_layout()
    
    # 添加总标题
    filename = Path(pkl_file).name
    fig.suptitle(f'Root Position Analysis - {filename}', fontsize=16, y=0.98)
    
    # 保存图片
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        output_file = os.path.join(output_dir, f"{Path(pkl_file).stem}_root_pos.png")
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"图片已保存到: {output_file}")
    
    # 显示图片
    plt.show()

def main():
    # 指定要分析的文件
    pkl_file = "/home/zqr/devel/dataset/AMASS_g1/CNRS/288/08_L_2_stageii.pkl"
    
    # 输出目录
    output_dir = "/home/zqr/devel/GMR/motion_analysis/plots"
    
    # 绘制图形
    plot_root_pos(pkl_file, output_dir)

if __name__ == "__main__":
    main()