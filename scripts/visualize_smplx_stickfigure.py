import argparse
import pathlib
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D

from general_motion_retargeting.utils.smpl import load_smplx_file, get_smplx_data_offline_fast

# SMPL-X 骨骼连接定义 (父子关系)
SMPLX_SKELETON_CONNECTIONS = [
    # 脊柱
    ('pelvis', 'spine1'),
    ('spine1', 'spine2'),
    ('spine2', 'spine3'),
    ('spine3', 'neck'),
    ('neck', 'head'),
    
    # 左臂
    ('spine3', 'left_collar'),
    ('left_collar', 'left_shoulder'),
    ('left_shoulder', 'left_elbow'),
    ('left_elbow', 'left_wrist'),
    
    # 右臂
    ('spine3', 'right_collar'),
    ('right_collar', 'right_shoulder'),
    ('right_shoulder', 'right_elbow'),
    ('right_elbow', 'right_wrist'),
    
    # 左腿
    ('pelvis', 'left_hip'),
    ('left_hip', 'left_knee'),
    ('left_knee', 'left_ankle'),
    
    # 右腿
    ('pelvis', 'right_hip'),
    ('right_hip', 'right_knee'),
    ('right_knee', 'right_ankle'),
]


def plot_frame_2d(ax, frame_data, connections):
    """绘制单帧的2D火柴人"""
    ax.clear()
    
    # 绘制骨骼连接
    for parent, child in connections:
        if parent in frame_data and child in frame_data:
            parent_pos = frame_data[parent][0]
            child_pos = frame_data[child][0]
            
            # 投影到XZ平面 (侧视图)
            ax.plot([parent_pos[0], child_pos[0]], 
                   [parent_pos[2], child_pos[2]], 
                   'b-', linewidth=2)
    
    # 绘制关节点
    for joint_name, (position, _) in frame_data.items():
        ax.plot(position[0], position[2], 'ro', markersize=4)
    
    ax.set_xlabel('X')
    ax.set_ylabel('Z (Height)')
    ax.set_aspect('equal')
    ax.grid(True)
    ax.set_title('Side View (XZ Plane)')


def plot_frame_3d(ax, frame_data, connections):
    """绘制单帧的3D火柴人"""
    ax.clear()
    
    # 绘制骨骼连接
    for parent, child in connections:
        if parent in frame_data and child in frame_data:
            parent_pos = frame_data[parent][0]
            child_pos = frame_data[child][0]
            
            ax.plot([parent_pos[0], child_pos[0]], 
                   [parent_pos[1], child_pos[1]],
                   [parent_pos[2], child_pos[2]], 
                   'b-', linewidth=2)
    
    # 绘制关节点
    for joint_name, (position, _) in frame_data.items():
        ax.scatter(position[0], position[1], position[2], c='r', s=20)
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z (Height)')
    ax.set_title('3D View')
    
    # 设置坐标轴范围
    all_positions = np.array([pos for pos, _ in frame_data.values()])
    center = all_positions.mean(axis=0)
    max_range = 1.0  # 设置固定范围
    
    ax.set_xlim(center[0] - max_range, center[0] + max_range)
    ax.set_ylim(center[1] - max_range, center[1] + max_range)
    ax.set_zlim(0, 2.0)  # Z轴从地面开始


def create_animation(smplx_data_frames, fps, save_path=None, mode='3d'):
    """创建动画"""
    fig = plt.figure(figsize=(10, 8))
    
    if mode == '3d':
        ax = fig.add_subplot(111, projection='3d')
    else:
        ax = fig.add_subplot(111)
    
    def update(frame_idx):
        if mode == '3d':
            plot_frame_3d(ax, smplx_data_frames[frame_idx], SMPLX_SKELETON_CONNECTIONS)
        else:
            plot_frame_2d(ax, smplx_data_frames[frame_idx], SMPLX_SKELETON_CONNECTIONS)
        ax.set_title(f'Frame {frame_idx}/{len(smplx_data_frames)-1}')
        return ax,
    
    anim = FuncAnimation(fig, update, frames=len(smplx_data_frames),
                        interval=1000/fps, blit=False, repeat=True)
    
    if save_path:
        print(f"Saving animation to {save_path}...")
        anim.save(save_path, writer='pillow', fps=fps)
        print("Animation saved!")
    else:
        plt.show()
    
    return anim


def visualize_static_frames(smplx_data_frames, num_samples=5, mode='3d'):
    """可视化几个静态帧"""
    num_frames = len(smplx_data_frames)
    frame_indices = np.linspace(0, num_frames-1, num_samples, dtype=int)
    
    if mode == '3d':
        fig = plt.figure(figsize=(15, 3))
        for i, frame_idx in enumerate(frame_indices):
            ax = fig.add_subplot(1, num_samples, i+1, projection='3d')
            plot_frame_3d(ax, smplx_data_frames[frame_idx], SMPLX_SKELETON_CONNECTIONS)
            ax.set_title(f'Frame {frame_idx}')
    else:
        fig, axes = plt.subplots(1, num_samples, figsize=(15, 3))
        for i, frame_idx in enumerate(frame_indices):
            plot_frame_2d(axes[i], smplx_data_frames[frame_idx], SMPLX_SKELETON_CONNECTIONS)
            axes[i].set_title(f'Frame {frame_idx}')
    
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    HERE = pathlib.Path(__file__).parent

    parser = argparse.ArgumentParser(description='Visualize SMPL-X motion as stick figure')
    parser.add_argument(
        "--smplx_file",
        help="SMPLX motion file to load.",
        type=str,
        default="/home/yanjieze/projects/g1_wbc/GMR/motion_data/ACCAD/Male1General_c3d/General_A1_-_Stand_stageii.npz",
    )
    
    parser.add_argument(
        "--mode",
        choices=["2d", "3d"],
        default="3d",
        help="Visualization mode: 2d (side view) or 3d",
    )
    
    parser.add_argument(
        "--animate",
        action="store_true",
        help="Create animation instead of static frames",
    )
    
    parser.add_argument(
        "--save_gif",
        type=str,
        default=None,
        help="Path to save animation as GIF",
    )
    
    parser.add_argument(
        "--num_samples",
        type=int,
        default=5,
        help="Number of sample frames to display (for static mode)",
    )

    args = parser.parse_args()

    SMPLX_FOLDER = HERE / ".." / "assets" / "body_models"
    
    # Load SMPLX trajectory
    print("Loading SMPL-X file...")
    smplx_data, body_model, smplx_output, actual_human_height = load_smplx_file(
        args.smplx_file, SMPLX_FOLDER
    )
    
    # Align fps
    tgt_fps = 30
    print("Processing frames...")
    smplx_data_frames, aligned_fps = get_smplx_data_offline_fast(
        smplx_data, body_model, smplx_output, tgt_fps=tgt_fps
    )
    
    print(f"Total frames: {len(smplx_data_frames)}")
    print(f"FPS: {aligned_fps}")
    print(f"Duration: {len(smplx_data_frames)/aligned_fps:.2f}s")
    
    # 打印可用的关节名称
    print("\nAvailable joints:")
    for joint_name in smplx_data_frames[0].keys():
        print(f"  - {joint_name}")
    
    if args.animate:
        print("\nCreating animation...")
        create_animation(smplx_data_frames, aligned_fps, args.save_gif, args.mode)
    else:
        print("\nDisplaying static frames...")
        visualize_static_frames(smplx_data_frames, args.num_samples, args.mode)
