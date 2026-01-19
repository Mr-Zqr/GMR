
import pickle
import numpy as np
import torch
def load_robot_motion(motion_file):
    """
    Load robot motion data from a pickle file.
    """
    # joint_mapping = [0, 6, 12,
    #                  1, 7, 13,
    #                  2, 8, 14,
    #                  3, 9, 15, 22,
    #                  4, 10, 16, 23,
    #                  5, 11, 17, 24,
    #                  18, 25,
    #                  19, 26,
    #                  20, 27,
    #                  21, 28]
    with open(motion_file, "rb") as f:
        # motion_data = torch.load(f)
        # motion_data = pickle.load(f)
        motion_data = np.load(f, allow_pickle=True)
        # motion_fps = motion_data["fps"]
        motion_fps = 30
        motion_root_pos = motion_data["root_pos"]
        motion_root_rot = motion_data["root_rot"][:, [3, 0, 1, 2]]# from xyzw to wxyz
        dof_indices = list(range(0, 20)) + list(range(23, 26))
        motion_dof_pos = np.zeros((motion_data["root_rot"].shape[0], 29), dtype = np.float32)
        # motion_dof_pos[:, joint_mapping] = motion_data["joint_pos"]
        # motion_dof_pos[:, dof_indices] = motion_data["dof_pos"]
        motion_dof_pos = motion_data["dof_pos"]
        # motion_local_body_pos = motion_data["local_body_pos"]
        # motion_link_body_list = motion_data["link_body_list"]
    return motion_data, motion_fps, motion_root_pos, motion_root_rot, motion_dof_pos, None, None


