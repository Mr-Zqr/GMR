import numpy as np
import smplx
import torch
from scipy.spatial.transform import Rotation as R
from smplx.joint_names import JOINT_NAMES
from scipy.interpolate import interp1d

import general_motion_retargeting.utils.lafan_vendor.utils as utils

def load_smpl_file(smpl_file):
    smpl_data = np.load(smpl_file, allow_pickle=True)
    return smpl_data

def load_smplx_file(smplx_file, smplx_body_model_path):
    smplx_data = np.load(smplx_file, allow_pickle=True)
    body_model = smplx.create(
        smplx_body_model_path,
        "smplx",
        gender=str("male"),
        use_pca=False,
    )
    # print(smplx_data["pose_body"].shape)
    # print(smplx_data["betas"].shape)
    # print(smplx_data["root_orient"].shape)
    # print(smplx_data["trans"].shape)
    
    # trackings = smplx_data['trackings']
    # num_frames = trackings.item()['smpl_trans'].shape[0]

    # smplx_trans = trackings.item()['smpl_trans_wd']
    # smplx_global_orient_rot_matrix = trackings.item()['smpl_root_orient_wd']
    # Apply rotation of 90 degrees clockwise around X-axis to all frames
    # Rotation matrix for 90 degrees clockwise around X-axis: [[1, 0, 0], [0, cos(-90), -sin(-90)], [0, sin(-90), cos(-90)]]
    # first_frame_inv = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]], dtype=np.float32).reshape(1, 3, 3)
    # smplx_global_orient_rot_matrix =  first_frame_inv @smplx_global_orient_rot_matrix 
    # Apply the same rotation to translation
    # smplx_trans = (first_frame_inv.squeeze() @ smplx_trans.T).T
    # Convert rotation matrix (N, 1, 3, 3) to axis-angle (N, 3)
    # smplx_global_orient = R.from_matrix(smplx_global_orient_rot_matrix.squeeze(1)).as_rotvec()
    # print(smplx_global_orient)
    # smplx_body_pos_rot_matrix = trackings.item()['smpl_body_pose']
    # Apply the same rotation to the first joint of body pose only
    # smplx_body_pos_rot_matrix[:, 0:2] =  first_frame_inv @smplx_body_pos_rot_matrix[:, 0:1]
    # Convert body pose rotation matrix (N, 23, 3, 3) to axis-angle (N, 63), taking first 21 joints
    # smplx_body_pose = R.from_matrix(smplx_body_pos_rot_matrix[:, :21].reshape(-1, 3, 3)).as_rotvec().reshape(num_frames, -1)
    # smplx_betas = np.zeros(16)
    # smplx_betas[:10] = trackings.item()['smpl_shapes']

    # smplx_output = body_model(
    #     betas=torch.tensor(smplx_betas).float().view(1, -1), # (16,)
    #     global_orient=torch.tensor(smplx_global_orient).float(), # (N, 3)
    #     body_pose=torch.tensor(smplx_body_pose).float(), # (N, 63)
    #     transl=torch.tensor(smplx_trans).float(), # (N, 3)
    #     left_hand_pose=torch.zeros(num_frames, 45).float(),
    #     right_hand_pose=torch.zeros(num_frames, 45).float(),
    #     jaw_pose=torch.zeros(num_frames, 3).float(),
    #     leye_pose=torch.zeros(num_frames, 3).float(),
    #     reye_pose=torch.zeros(num_frames, 3).float(),
    #     # expression=torch.zeros(num_frames, 10).float(),
    #     return_full_pose=True,
    # )

    def _extract_trackings_dict(npz_data):
        if "trackings" not in npz_data:
            return None
        trackings = npz_data["trackings"]
        if isinstance(trackings, np.ndarray) and trackings.dtype == object:
            if trackings.shape == ():
                trackings = trackings.item()
            elif trackings.shape[0] > 0:
                trackings = trackings[0]
        return trackings if isinstance(trackings, dict) else None

    def _to_axis_angle_global_orient(global_orient):
        global_orient = np.asarray(global_orient)
        if global_orient.ndim == 2 and global_orient.shape[1] == 3:
            return global_orient
        if global_orient.ndim == 3 and global_orient.shape[1:] == (3, 3):
            return R.from_matrix(global_orient).as_rotvec()
        if global_orient.ndim == 4 and global_orient.shape[1:] == (1, 3, 3):
            return R.from_matrix(global_orient[:, 0]).as_rotvec()
        raise ValueError(
            f"Unsupported global_orient shape: {global_orient.shape}, expected (N,3) or rotation matrices"
        )

    def _to_axis_angle_body_pose(body_pose):
        body_pose = np.asarray(body_pose)
        if body_pose.ndim == 2 and body_pose.shape[1] >= 63:
            return body_pose[:, :63]
        if body_pose.ndim == 3 and body_pose.shape[1:] == (21, 3):
            return body_pose.reshape(body_pose.shape[0], -1)
        if body_pose.ndim == 4 and body_pose.shape[2:] == (3, 3):
            body_pose_aa = R.from_matrix(body_pose[:, :21].reshape(-1, 3, 3)).as_rotvec()
            return body_pose_aa.reshape(body_pose.shape[0], -1)
        raise ValueError(
            f"Unsupported body_pose/body_pos shape: {body_pose.shape}, expected (N,63), (N,21,3) or rotation matrices"
        )

    if "fullpose" in smplx_data and "trans" in smplx_data and "betas" in smplx_data:
        smplx_trans = np.asarray(smplx_data["trans"], dtype=np.float32)
        global_orient = np.asarray(smplx_data["fullpose"][:, :3], dtype=np.float32)
        smpl_body_pos = np.asarray(smplx_data["fullpose"][:, 3:66], dtype=np.float32)
        smplx_betas = np.asarray(smplx_data["betas"])
        if "mocap_frame_rate" in smplx_data:
            mocap_frame_rate = smplx_data["mocap_frame_rate"]
        else:
            mocap_frame_rate = np.array(120, dtype=np.int64)
    else:
        trackings = _extract_trackings_dict(smplx_data)
        data_source = trackings if trackings is not None else smplx_data

        transl_key = "transl" if "transl" in data_source else "trans"
        body_pose_key = "body_pose" if "body_pose" in data_source else "body_pos"

        required_keys = [transl_key, "global_orient", body_pose_key, "betas"]
        for key in required_keys:
            if key not in data_source:
                raise KeyError(
                    f"Missing key '{key}' in SMPL-X file: {smplx_file}. Available keys: {list(data_source.keys())}"
                )

        smplx_trans = np.asarray(data_source[transl_key], dtype=np.float32)
        global_orient = _to_axis_angle_global_orient(data_source["global_orient"]).astype(np.float32)
        smpl_body_pos = _to_axis_angle_body_pose(data_source[body_pose_key]).astype(np.float32)
        smplx_betas = np.asarray(data_source["betas"]) 

        if "mocap_frame_rate" in data_source:
            mocap_frame_rate = np.array(data_source["mocap_frame_rate"]).squeeze()
        elif "fps" in data_source:
            mocap_frame_rate = np.array(data_source["fps"]).squeeze()
        else:
            mocap_frame_rate = np.array(30, dtype=np.int64)

    num_frames = smplx_trans.shape[0]

    smplx_betas = np.asarray(smplx_betas, dtype=np.float32)
    if smplx_betas.ndim > 1:
        smplx_betas = smplx_betas[0]
    if smplx_betas.shape[0] < 16:
        smplx_betas = np.pad(smplx_betas, (0, 16 - smplx_betas.shape[0]))
    else:
        smplx_betas = smplx_betas[:16]

    smplx_output = body_model(
        betas=torch.tensor(smplx_betas).float().view(1, -1), # (16,)
        global_orient=torch.tensor(global_orient).float(), # (N, 3)
        body_pose=torch.tensor(smpl_body_pos).float(), # (N, 63)
        transl=torch.tensor(smplx_trans).float(), # (N, 3)
        left_hand_pose=torch.zeros(num_frames, 45).float(),
        right_hand_pose=torch.zeros(num_frames, 45).float(),
        jaw_pose=torch.zeros(num_frames, 3).float(),
        leye_pose=torch.zeros(num_frames, 3).float(),
        reye_pose=torch.zeros(num_frames, 3).float(),
        # expression=torch.zeros(num_frames, 10).float(),
        return_full_pose=True,
    )

    if hasattr(smplx_data, "files"):
        smplx_data_dict = {k: smplx_data[k] for k in smplx_data.files}
    else:
        smplx_data_dict = dict(smplx_data)
    smplx_data_dict["trans"] = smplx_trans
    smplx_data_dict["mocap_frame_rate"] = np.array(mocap_frame_rate, dtype=np.int64)
    
    if len(smplx_betas.shape)==1:
        human_height = 1.66 + 0.1 * smplx_betas[0]
    else:
        human_height = 1.66 + 0.1 * smplx_betas[0, 0]
    # if len(smplx_betas.shape)==1:
    #     human_height = 1.66 + 0.1 * smplx_betas[0]
    # else:
    #     human_height = 1.66 + 0.1 * smplx_betas[0, 0]
    
    return smplx_data_dict, body_model, smplx_output, human_height


def load_gvhmr_pred_file(gvhmr_pred_file, smplx_body_model_path):
    gvhmr_pred = torch.load(gvhmr_pred_file)
    smpl_params_global = gvhmr_pred['smpl_params_global']
    # print(smpl_params_global['body_pose'].shape)
    # print(smpl_params_global['betas'].shape)
    # print(smpl_params_global['global_orient'].shape)
    # print(smpl_params_global['transl'].shape)
    
    betas = np.pad(smpl_params_global['betas'][0], (0,6))
    
    # correct rotations
    # rotation_matrix = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]])
    # rotation_quat = R.from_matrix(rotation_matrix).as_quat(scalar_first=True)
    
    # smpl_params_global['body_pose'] = smpl_params_global['body_pose'] @ rotation_matrix
    # smpl_params_global['global_orient'] = smpl_params_global['global_orient'] @ rotation_quat
    
    smplx_data = {
        'pose_body': smpl_params_global['body_pose'].numpy(),
        'betas': betas,
        'root_orient': smpl_params_global['global_orient'].numpy(),
        'trans': smpl_params_global['transl'].numpy(),
        "mocap_frame_rate": torch.tensor(30),
    }

    body_model = smplx.create(
        smplx_body_model_path,
        "smplx",
        gender="neutral",
        use_pca=False,
    )
    
    num_frames = smpl_params_global['body_pose'].shape[0]
    smplx_output = body_model(
        betas=torch.tensor(smplx_data["betas"]).float().view(1, -1), # (16,)
        global_orient=torch.tensor(smplx_data["root_orient"]).float(), # (N, 3)
        body_pose=torch.tensor(smplx_data["pose_body"]).float(), # (N, 63)
        transl=torch.tensor(smplx_data["trans"]).float(), # (N, 3)
        left_hand_pose=torch.zeros(num_frames, 45).float(),
        right_hand_pose=torch.zeros(num_frames, 45).float(),
        jaw_pose=torch.zeros(num_frames, 3).float(),
        leye_pose=torch.zeros(num_frames, 3).float(),
        reye_pose=torch.zeros(num_frames, 3).float(),
        # expression=torch.zeros(num_frames, 10).float(),
        return_full_pose=True,
    )
    
    if len(smplx_data['betas'].shape)==1:
        human_height = 1.66 + 0.1 * smplx_data['betas'][0]
    else:
        human_height = 1.66 + 0.1 * smplx_data['betas'][0, 0]
    
    return smplx_data, body_model, smplx_output, human_height


def get_smplx_data(smplx_data, body_model, smplx_output, curr_frame):
    """
    Must return a dictionary with the following structure:
    {
        "Hips": (position, orientation),
        "Spine": (position, orientation),
        ...
    }
    """
    global_orient = smplx_output.global_orient[curr_frame].squeeze()
    full_body_pose = smplx_output.full_pose[curr_frame].reshape(-1, 3)
    joints = smplx_output.joints[curr_frame].detach().numpy().squeeze()
    joint_names = JOINT_NAMES[: len(body_model.parents)]
    parents = body_model.parents

    result = {}
    joint_orientations = []
    for i, joint_name in enumerate(joint_names):
        if i == 0:
            rot = R.from_rotvec(global_orient)
        else:
            rot = joint_orientations[parents[i]] * R.from_rotvec(
                full_body_pose[i].squeeze()
            )
        joint_orientations.append(rot)
        result[joint_name] = (joints[i], rot.as_quat(scalar_first=True))

  
    return result


def slerp(rot1, rot2, t):
    """Spherical linear interpolation between two rotations."""
    # Convert to quaternions
    q1 = rot1.as_quat()
    q2 = rot2.as_quat()
    
    # Normalize quaternions
    q1 = q1 / np.linalg.norm(q1)
    q2 = q2 / np.linalg.norm(q2)
    
    # Compute dot product
    dot = np.sum(q1 * q2)
    
    # If the dot product is negative, slerp won't take the shorter path
    if dot < 0.0:
        q2 = -q2
        dot = -dot
    
    # If the inputs are too close, linearly interpolate
    if dot > 0.9995:
        return R.from_quat(q1 + t * (q2 - q1))
    
    # Perform SLERP
    theta_0 = np.arccos(dot)
    theta = theta_0 * t
    sin_theta = np.sin(theta)
    sin_theta_0 = np.sin(theta_0)
    
    s0 = np.cos(theta) - dot * sin_theta / sin_theta_0
    s1 = sin_theta / sin_theta_0
    q = s0 * q1 + s1 * q2
    
    return R.from_quat(q)

def get_smplx_data_offline_fast(smplx_data, body_model, smplx_output, tgt_fps=30, src_fps=None):
    """
    Must return a dictionary with the following structure:
    {
        "Hips": (position, orientation),
        "Spine": (position, orientation),
        ...
    }
    src_fps: source FPS of the motion data. If None, tries smplx_data["mocap_frame_rate"],
             falls back to 120.
    """
    if src_fps is None:
        if "mocap_frame_rate" in smplx_data:
            src_fps = smplx_data["mocap_frame_rate"].item()
        else:
            src_fps = 120
    num_frames = smplx_data['trans'].shape[0]
    global_orient = smplx_output.global_orient.squeeze()
    full_body_pose = smplx_output.full_pose.reshape(num_frames, -1, 3)
    joints = smplx_output.joints.detach().numpy().squeeze()
    joint_names = JOINT_NAMES[: len(body_model.parents)]
    parents = body_model.parents
    
    if tgt_fps != src_fps:
        # perform fps alignment with proper interpolation
        # Calculate correct number of output frames based on time duration
        new_num_frames = int(num_frames * tgt_fps / src_fps)
        
        # Create time points for interpolation
        original_time = np.arange(num_frames)
        target_time = np.linspace(0, num_frames-1, new_num_frames)
        
        # Interpolate global orientation using SLERP
        global_orient_interp = []
        for i in range(len(target_time)):
            t = target_time[i]
            idx1 = int(np.floor(t))
            idx2 = min(idx1 + 1, num_frames - 1)
            alpha = t - idx1
            
            rot1 = R.from_rotvec(global_orient[idx1])
            rot2 = R.from_rotvec(global_orient[idx2])
            interp_rot = slerp(rot1, rot2, alpha)
            global_orient_interp.append(interp_rot.as_rotvec())
        global_orient = np.stack(global_orient_interp, axis=0)
        
        # Interpolate full body pose using SLERP
        full_body_pose_interp = []
        for i in range(full_body_pose.shape[1]):  # For each joint
            joint_rots = []
            for j in range(len(target_time)):
                t = target_time[j]
                idx1 = int(np.floor(t))
                idx2 = min(idx1 + 1, num_frames - 1)
                alpha = t - idx1
                
                rot1 = R.from_rotvec(full_body_pose[idx1, i])
                rot2 = R.from_rotvec(full_body_pose[idx2, i])
                interp_rot = slerp(rot1, rot2, alpha)
                joint_rots.append(interp_rot.as_rotvec())
            full_body_pose_interp.append(np.stack(joint_rots, axis=0))
        full_body_pose = np.stack(full_body_pose_interp, axis=1)
        
        # Interpolate joint positions using linear interpolation
        joints_interp = []
        for i in range(joints.shape[1]):  # For each joint
            for j in range(3):  # For each coordinate
                interp_func = interp1d(original_time, joints[:, i, j], kind='linear')
                joints_interp.append(interp_func(target_time))
        joints = np.stack(joints_interp, axis=1).reshape(new_num_frames, -1, 3)
        
        aligned_fps = len(global_orient) / num_frames * src_fps
    else:
        aligned_fps = tgt_fps
        
    smplx_data_frames = []
    for curr_frame in range(len(global_orient)):
        result = {}
        single_global_orient = global_orient[curr_frame]
        single_full_body_pose = full_body_pose[curr_frame]
        single_joints = joints[curr_frame]
        joint_orientations = []
        for i, joint_name in enumerate(joint_names):
            if i == 0:
                rot = R.from_rotvec(single_global_orient)
            else:
                rot = joint_orientations[parents[i]] * R.from_rotvec(
                    single_full_body_pose[i].squeeze()
                )
            joint_orientations.append(rot)
            result[joint_name] = (single_joints[i], rot.as_quat(scalar_first=True))


        smplx_data_frames.append(result)

    return smplx_data_frames, aligned_fps



def get_gvhmr_data_offline_fast(smplx_data, body_model, smplx_output, tgt_fps=30):
    """
    Must return a dictionary with the following structure:
    {
        "Hips": (position, orientation),
        "Spine": (position, orientation),
        ...
    }
    """
    src_fps = smplx_data["mocap_frame_rate"].item()
    num_frames = smplx_data["pose_body"].shape[0]
    global_orient = smplx_output.global_orient.squeeze()
    full_body_pose = smplx_output.full_pose.reshape(num_frames, -1, 3)
    joints = smplx_output.joints.detach().numpy().squeeze()
    joint_names = JOINT_NAMES[: len(body_model.parents)]
    parents = body_model.parents
    
    if tgt_fps != src_fps:
        # perform fps alignment with proper interpolation
        # Calculate correct number of output frames based on time duration
        new_num_frames = int(num_frames * tgt_fps / src_fps)
        
        # Create time points for interpolation
        original_time = np.arange(num_frames)
        target_time = np.linspace(0, num_frames-1, new_num_frames)
        
        # Interpolate global orientation using SLERP
        global_orient_interp = []
        for i in range(len(target_time)):
            t = target_time[i]
            idx1 = int(np.floor(t))
            idx2 = min(idx1 + 1, num_frames - 1)
            alpha = t - idx1
            
            rot1 = R.from_rotvec(global_orient[idx1])
            rot2 = R.from_rotvec(global_orient[idx2])
            interp_rot = slerp(rot1, rot2, alpha)
            global_orient_interp.append(interp_rot.as_rotvec())
        global_orient = np.stack(global_orient_interp, axis=0)
        
        # Interpolate full body pose using SLERP
        full_body_pose_interp = []
        for i in range(full_body_pose.shape[1]):  # For each joint
            joint_rots = []
            for j in range(len(target_time)):
                t = target_time[j]
                idx1 = int(np.floor(t))
                idx2 = min(idx1 + 1, num_frames - 1)
                alpha = t - idx1
                
                rot1 = R.from_rotvec(full_body_pose[idx1, i])
                rot2 = R.from_rotvec(full_body_pose[idx2, i])
                interp_rot = slerp(rot1, rot2, alpha)
                joint_rots.append(interp_rot.as_rotvec())
            full_body_pose_interp.append(np.stack(joint_rots, axis=0))
        full_body_pose = np.stack(full_body_pose_interp, axis=1)
        
        # Interpolate joint positions using linear interpolation
        joints_interp = []
        for i in range(joints.shape[1]):  # For each joint
            for j in range(3):  # For each coordinate
                interp_func = interp1d(original_time, joints[:, i, j], kind='linear')
                joints_interp.append(interp_func(target_time))
        joints = np.stack(joints_interp, axis=1).reshape(new_num_frames, -1, 3)
        
        aligned_fps = len(global_orient) / num_frames * src_fps
    else:
        aligned_fps = tgt_fps
        
    smplx_data_frames = []
    for curr_frame in range(len(global_orient)):
        result = {}
        single_global_orient = global_orient[curr_frame]
        single_full_body_pose = full_body_pose[curr_frame]
        single_joints = joints[curr_frame]
        joint_orientations = []
        for i, joint_name in enumerate(joint_names):
            if i == 0:
                rot = R.from_rotvec(single_global_orient)
            else:
                rot = joint_orientations[parents[i]] * R.from_rotvec(
                    single_full_body_pose[i].squeeze()
                )
            joint_orientations.append(rot)
            result[joint_name] = (single_joints[i], rot.as_quat(scalar_first=True))


        smplx_data_frames.append(result)
        
    # add correct rotations
    rotation_matrix = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]])
    rotation_quat = R.from_matrix(rotation_matrix).as_quat(scalar_first=True)
    for result in smplx_data_frames:
        for joint_name in result.keys():
            orientation = utils.quat_mul(rotation_quat, result[joint_name][1])
            position = result[joint_name][0] @ rotation_matrix.T
            result[joint_name] = (position, orientation)
            

    return smplx_data_frames, aligned_fps
