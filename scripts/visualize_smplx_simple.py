"""
简化版的SMPL-X火柴人可视化工具
直接在 smplx_to_robot.py 中使用
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D


# SMPL-X 主要骨骼连接
SKELETON_CONNECTIONS = [
    ('pelvis', 'spine1'),
    ('spine1', 'spine2'),
    ('spine2', 'spine3'),
    ('spine3', 'neck'),
    ('neck', 'head'),
    ('spine3', 'left_collar'),
    ('left_collar', 'left_shoulder'),
    ('left_shoulder', 'left_elbow'),
    ('left_elbow', 'left_wrist'),
    ('spine3', 'right_collar'),
    ('right_collar', 'right_shoulder'),
    ('right_shoulder', 'right_elbow'),
    ('right_elbow', 'right_wrist'),
    ('pelvis', 'left_hip'),
    ('left_hip', 'left_knee'),
    ('left_knee', 'left_ankle'),
    ('pelvis', 'right_hip'),
    ('right_hip', 'right_knee'),
    ('right_knee', 'right_ankle'),
]


def visualize_frames_interactive(smplx_data_frames, fps=30):
    """交互式可视化 - 可以使用滑块控制帧"""
    from matplotlib.widgets import Slider
    
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    plt.subplots_adjust(bottom=0.15)
    
    # 滑块控制帧
    ax_slider = plt.axes([0.2, 0.05, 0.6, 0.03])
    slider = Slider(ax_slider, 'Frame', 0, len(smplx_data_frames)-1, 
                    valinit=0, valstep=1)
    
    def plot_frame(frame_idx):
        ax.clear()
        frame_data = smplx_data_frames[int(frame_idx)]
        
        # 绘制骨骼
        for parent, child in SKELETON_CONNECTIONS:
            if parent in frame_data and child in frame_data:
                p_pos = frame_data[parent][0]
                c_pos = frame_data[child][0]
                ax.plot([p_pos[0], c_pos[0]], 
                       [p_pos[1], c_pos[1]],
                       [p_pos[2], c_pos[2]], 
                       'b-', linewidth=2)
        
        # 绘制关节
        for joint_name, (position, _) in frame_data.items():
            ax.scatter(position[0], position[1], position[2], c='r', s=30)
        
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title(f'Frame {int(frame_idx)}/{len(smplx_data_frames)-1}')
        
        # 固定视角范围
        all_pos = np.array([pos for pos, _ in frame_data.values()])
        center = all_pos.mean(axis=0)
        ax.set_xlim(center[0]-1, center[0]+1)
        ax.set_ylim(center[1]-1, center[1]+1)
        ax.set_zlim(0, 2)
        
        fig.canvas.draw_idle()
    
    slider.on_changed(plot_frame)
    plot_frame(0)
    plt.show()


def visualize_frames_animation(smplx_data_frames, fps=30, save_path=None):
    """创建动画"""
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    def update(frame_idx):
        ax.clear()
        frame_data = smplx_data_frames[frame_idx]
        
        # 绘制骨骼
        for parent, child in SKELETON_CONNECTIONS:
            if parent in frame_data and child in frame_data:
                p_pos = frame_data[parent][0]
                c_pos = frame_data[child][0]
                ax.plot([p_pos[0], c_pos[0]], 
                       [p_pos[1], c_pos[1]],
                       [p_pos[2], c_pos[2]], 
                       'b-', linewidth=2)
        
        # 绘制关节
        for position, _ in frame_data.values():
            ax.scatter(position[0], position[1], position[2], c='r', s=20)
        
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title(f'Frame {frame_idx}/{len(smplx_data_frames)-1}')
        
        # 计算所有帧的范围以保持一致
        all_pos = np.array([pos for pos, _ in frame_data.values()])
        center = all_pos.mean(axis=0)
        ax.set_xlim(center[0]-1, center[0]+1)
        ax.set_ylim(center[1]-1, center[1]+1)
        ax.set_zlim(0, 2)
    
    anim = FuncAnimation(fig, update, frames=len(smplx_data_frames),
                        interval=1000/fps, repeat=True)
    
    if save_path:
        print(f"保存动画到 {save_path}...")
        anim.save(save_path, writer='pillow', fps=fps)
        print("保存完成！")
    else:
        plt.show()


def visualize_frames_grid(smplx_data_frames, num_samples=6):
    """网格显示多个关键帧"""
    num_frames = len(smplx_data_frames)
    frame_indices = np.linspace(0, num_frames-1, num_samples, dtype=int)
    
    cols = 3
    rows = (num_samples + cols - 1) // cols
    fig = plt.figure(figsize=(15, 5*rows))
    
    for i, frame_idx in enumerate(frame_indices):
        ax = fig.add_subplot(rows, cols, i+1, projection='3d')
        frame_data = smplx_data_frames[frame_idx]
        
        # 绘制骨骼
        for parent, child in SKELETON_CONNECTIONS:
            if parent in frame_data and child in frame_data:
                p_pos = frame_data[parent][0]
                c_pos = frame_data[child][0]
                ax.plot([p_pos[0], c_pos[0]], 
                       [p_pos[1], c_pos[1]],
                       [p_pos[2], c_pos[2]], 
                       'b-', linewidth=2)
        
        # 绘制关节
        for position, _ in frame_data.values():
            ax.scatter(position[0], position[1], position[2], c='r', s=20)
        
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title(f'Frame {frame_idx}')
        
        all_pos = np.array([pos for pos, _ in frame_data.values()])
        center = all_pos.mean(axis=0)
        ax.set_xlim(center[0]-1, center[0]+1)
        ax.set_ylim(center[1]-1, center[1]+1)
        ax.set_zlim(0, 2)
    
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    print("这是一个库文件，请从 smplx_to_robot.py 中导入使用")
    print("\n使用示例:")
    print("from visualize_smplx_simple import visualize_frames_interactive")
    print("visualize_frames_interactive(smplx_data_frames, fps=aligned_fps)")
