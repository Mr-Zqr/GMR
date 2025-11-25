#!/usr/bin/env python3
"""
快速测试脚本 - 可视化 SMPL-X 动作
"""
import pathlib
import sys

# 添加项目路径
HERE = pathlib.Path(__file__).parent.parent
sys.path.insert(0, str(HERE))

from general_motion_retargeting.utils.smpl import load_smplx_file, get_smplx_data_offline_fast
from scripts.visualize_smplx_simple import (
    visualize_frames_interactive,
    visualize_frames_animation, 
    visualize_frames_grid
)

def test_visualization(smplx_file):
    """测试可视化功能"""
    
    SMPLX_FOLDER = HERE / "assets" / "body_models"
    
    print("=" * 60)
    print("SMPL-X 动作可视化测试")
    print("=" * 60)
    print(f"\n文件: {smplx_file}")
    
    # 加载数据
    print("\n1. 加载 SMPL-X 文件...")
    smplx_data, body_model, smplx_output, actual_human_height = load_smplx_file(
        smplx_file, SMPLX_FOLDER
    )
    print(f"   ✓ 人体高度: {actual_human_height:.2f}m")
    
    # 处理帧
    print("\n2. 处理帧数据...")
    tgt_fps = 30
    smplx_data_frames, aligned_fps = get_smplx_data_offline_fast(
        smplx_data, body_model, smplx_output, tgt_fps=tgt_fps
    )
    print(f"   ✓ 总帧数: {len(smplx_data_frames)}")
    print(f"   ✓ FPS: {aligned_fps}")
    print(f"   ✓ 时长: {len(smplx_data_frames)/aligned_fps:.2f}秒")
    
    # 显示可用关节
    print("\n3. 可用关节 (前10个):")
    joint_names = list(smplx_data_frames[0].keys())
    for i, name in enumerate(joint_names[:10]):
        print(f"   - {name}")
    if len(joint_names) > 10:
        print(f"   ... 还有 {len(joint_names)-10} 个关节")
    
    # 可视化选项
    print("\n" + "=" * 60)
    print("选择可视化模式:")
    print("=" * 60)
    print("1. 交互式 (推荐) - 使用滑块浏览")
    print("2. 动画 - 自动播放")
    print("3. 网格 - 显示关键帧")
    print("4. 全部显示")
    print("0. 退出")
    
    choice = input("\n请选择 [1-4, 0]: ").strip()
    
    if choice == "1":
        print("\n启动交互式可视化...")
        visualize_frames_interactive(smplx_data_frames, aligned_fps)
    elif choice == "2":
        print("\n播放动画...")
        visualize_frames_animation(smplx_data_frames, aligned_fps)
    elif choice == "3":
        print("\n显示关键帧网格...")
        visualize_frames_grid(smplx_data_frames, num_samples=6)
    elif choice == "4":
        print("\n先显示网格，关闭后再显示交互式...")
        visualize_frames_grid(smplx_data_frames, num_samples=6)
        visualize_frames_interactive(smplx_data_frames, aligned_fps)
    else:
        print("\n退出")
        return
    
    print("\n可视化完成！")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="测试 SMPL-X 可视化")
    parser.add_argument(
        "--smplx_file",
        type=str,
        help="SMPL-X motion file (.npz)",
        default=None
    )
    
    args = parser.parse_args()
    
    if args.smplx_file:
        test_visualization(args.smplx_file)
    else:
        print("请提供 SMPL-X 文件:")
        print("python scripts/test_visualization.py --smplx_file /path/to/file.npz")
        print("\n或者修改脚本中的默认路径")
