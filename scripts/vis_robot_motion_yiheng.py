import numpy as np
from general_motion_retargeting import RobotMotionViewer
import argparse
import os
from tqdm import tqdm
import glob

def load_robot_motion(motion_file):
    """
    Load robot motion data from a pickle file.
    """
    with open(motion_file, "rb") as f:
        # motion_data = joblib.load(f)        
        motion_data = np.load(f, allow_pickle=True)
        # motion_data = torch.load(f)
        # motion_fps = motion_data["fps"]
        motion_fps = 50
        # motion_id = 820
        # print(motion_data[motion_id]["caption"])
        # motion_root_pos = motion_data["root_pos"]
        # motion_root_rot = motion_data["root_rot"][:, [3, 0, 1, 2]] # from xyzw to wxyz
        motion_root_pos = motion_data["body_pos_w"][:, 0, :]
        motion_root_rot = motion_data["body_quat_w"][:, 0, :] # from xyzw to wxyz
        # motion_root_rot = motion_data["root_rot"] # from xyzw to wxyz
        dof_indices = list(range(0, 20)) + list(range(23, 26))
        joint_mapping = list([0, 6, 12,
                          1, 7, 13,
                          2, 8, 14,
                          3, 9    , 15, 22,
                          4, 10   , 16, 23,
                          5, 11   , 17, 24,
                                    18, 25,
                                    19, 26,
                                    20, 27,
                                    21, 28])
        motion_dof_pos = np.zeros((motion_data["body_quat_w"].shape[0], 29), dtype = np.float32)
        motion_dof_pos[:, joint_mapping] = motion_data["joint_pos"]
    return motion_data, motion_fps, motion_root_pos, motion_root_rot, motion_dof_pos, None, None

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--robot", type=str, default="unitree_g1")
                        
    parser.add_argument("--robot_motion_dir", type=str, 
                        default="/home/amax/devel/dataset/yiheng_g1/TWIST",
                        help="Directory containing motion files")

    parser.add_argument("--video_dir", type=str, 
                        default="videos/twist_batch",
                        help="Directory to save videos")
                        
    args = parser.parse_args()
    
    robot_type = args.robot
    robot_motion_dir = args.robot_motion_dir
    video_dir = args.video_dir
    
    if not os.path.exists(robot_motion_dir):
        raise FileNotFoundError(f"Motion directory {robot_motion_dir} not found")
    
    # Create video output directory
    os.makedirs(video_dir, exist_ok=True)
    
    # Find all motion files (assuming .pkl or .npz format)
    motion_files = glob.glob(os.path.join(robot_motion_dir, "*.pkl")) + \
                   glob.glob(os.path.join(robot_motion_dir, "*.npz"))
    
    if len(motion_files) == 0:
        raise FileNotFoundError(f"No motion files found in {robot_motion_dir}")
    
    print(f"Found {len(motion_files)} motion files to process")
    
    # Create a single viewer instance for all videos (reuse renderer)
    env = None
    
    # Process each motion file
    for motion_path in tqdm(motion_files, desc="Processing motions"):
        motion_filename = os.path.basename(motion_path)
        motion_name = os.path.splitext(motion_filename)[0]
        video_path = os.path.join(video_dir, f"{motion_name}.mp4")
        
        print(f"\nProcessing: {motion_filename}")
        
        try:
            # Load motion data
            motion_data, motion_fps, motion_root_pos, motion_root_rot, motion_dof_pos, motion_local_body_pos, motion_link_body_list = load_robot_motion(motion_path)
            
            # Close previous video writer if exists
            if env is not None and hasattr(env, 'mp4_writer'):
                env.mp4_writer.close()
                print(f"✓ Saved previous video")
            
            # Create or update viewer for new video
            if env is None:
                # First time: create viewer with renderer
                env = RobotMotionViewer(robot_type=robot_type,
                                        motion_fps=motion_fps,
                                        camera_follow=True,
                                        record_video=True, 
                                        video_path=video_path,
                                        headless=True)
            else:
                # Reuse existing viewer, just create new video writer
                import imageio
                env.video_path = video_path
                env.motion_fps = motion_fps
                env.mp4_writer = imageio.get_writer(video_path, fps=motion_fps)
                print(f"Recording video to {video_path}")
            
            # Render all frames
            for frame_idx in range(len(motion_root_pos)):
                env.step(motion_root_pos[frame_idx], 
                        motion_root_rot[frame_idx], 
                        motion_dof_pos[frame_idx], 
                        rate_limit=False)
            
        except Exception as e:
            print(f"✗ Error processing {motion_filename}: {e}")
            continue
    
    # Close final video and cleanup
    if env is not None:
        env.close()
    
    print(f"\n✓ All done! Videos saved to {video_dir}")