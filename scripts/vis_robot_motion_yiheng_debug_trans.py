from ntpath import join
import numpy as np
from scipy.spatial.transform import Rotation as R
from general_motion_retargeting import RobotMotionViewer
import argparse
import os
import joblib
import torch

def load_robot_motion(motion_file):
    """
    Load robot motion data from a pickle file.
    """
    def to_numpy(tensor):
        """Convert torch tensor to numpy array, handling CUDA tensors."""
        if hasattr(tensor, 'cpu'):  # Check if it's a torch tensor
            return tensor.cpu().numpy()
        return np.array(tensor)
    with open(motion_file, "rb") as f:
        motion_data = np.load(f, allow_pickle=True)
        motion_fps = 30
        g1_trans = to_numpy(motion_data["g1_trans"])
        motion_root_pos = g1_trans[:, 0, :] if g1_trans.ndim == 3 else g1_trans
        # motion_root_rot = to_numpy(motion_data["g1_root_rot"])  # xyzw -> wxyz
        motion_root_rot =to_numpy(motion_data["g1_root_rot"])  
        # motion_root_rot = R.from_rotvec(to_numpy(motion_data["g1_root_rot"])).as_quat()[:, [3, 0, 1, 2]]  # axis-angle -> xyzw
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
        motion_dof_pos = np.zeros((motion_data["g1_dof"].shape[0], 29), dtype = np.float32)
        motion_dof_pos[:, joint_mapping] = motion_data["g1_dof"]
    return motion_data, motion_fps, motion_root_pos, motion_root_rot, motion_dof_pos, None, None

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--robot", type=str, default="unitree_g1")
                        
    parser.add_argument("--robot_motion_path", type=str, required=True)

    parser.add_argument("--record_video", action="store_true")
    parser.add_argument("--video_path", type=str, 
                        default="videos/example.mp4")
                        
    args = parser.parse_args()
    
    robot_type = args.robot
    robot_motion_path = args.robot_motion_path
    
    if not os.path.exists(robot_motion_path):
        raise FileNotFoundError(f"Motion file  not found")
    
    motion_data, motion_fps, motion_root_pos, motion_root_rot, motion_dof_pos, motion_local_body_pos, motion_link_body_list = load_robot_motion(robot_motion_path)
    
    env = RobotMotionViewer(robot_type=robot_type,
                            motion_fps=motion_fps,
                            camera_follow=False,
                            record_video=args.record_video, video_path=args.video_path)
    
    frame_idx = 0
    while True:
        env.step(motion_root_pos[frame_idx], 
                motion_root_rot[frame_idx], 
                motion_dof_pos[frame_idx], 
                rate_limit=True)
        frame_idx += 1
        if frame_idx >= len(motion_root_pos):
            frame_idx = 0
    env.close()