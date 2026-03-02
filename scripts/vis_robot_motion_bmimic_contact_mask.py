import numpy as np
from general_motion_retargeting import RobotMotionViewer
import argparse
import os
import joblib
import torch
from scipy.spatial.transform import Rotation as R
import mujoco as mj

# Lab body name list for contact mask mapping
lab_body_name_list = [
  "pelvis",
  "left_hip_pitch_link",
  "right_hip_pitch_link",
  "waist_yaw_link",
  "left_hip_roll_link",
  "right_hip_roll_link",
  "waist_roll_link",
  "left_hip_yaw_link",
  "right_hip_yaw_link",
  "torso_link",
  "left_knee_link",
  "right_knee_link",
  "left_shoulder_pitch_link",
  "right_shoulder_pitch_link",
  "left_ankle_pitch_link",
  "right_ankle_pitch_link",
  "left_shoulder_roll_link",
  "right_shoulder_roll_link",
  "left_ankle_roll_link",
  "right_ankle_roll_link",
  "left_shoulder_yaw_link",
  "right_shoulder_yaw_link",
  "left_elbow_link",
  "right_elbow_link",
  "left_wrist_roll_link",
  "right_wrist_roll_link",
  "left_wrist_pitch_link",
  "right_wrist_pitch_link",
  "left_wrist_yaw_link",
  "right_wrist_yaw_link",
]

def draw_contact_frame(
    pos,
    mat,
    v,
    size,
    joint_name=None,
    orientation_correction=R.from_euler("xyz", [0, 0, 0]),
    pos_offset=np.array([0, 0, 0]),
    contact_color=[1, 0, 0, 1],  # 默认红色表示接触
):
    """
    绘制接触点的坐标框架，使用指定的颜色
    """
    # 为接触点使用单一颜色（红色）
    rgba_list = [contact_color, contact_color, contact_color]
    for i in range(3):
        geom = v.user_scn.geoms[v.user_scn.ngeom]
        mj.mjv_initGeom(
            geom,
            type=mj.mjtGeom.mjGEOM_ARROW,
            size=[0.01, 0.01, 0.01],
            pos=pos + pos_offset,
            mat=mat.flatten(),
            rgba=rgba_list[i],
        )
        # if joint_name is not None:
        #     geom.label = joint_name
        fix = orientation_correction.as_matrix()
        mj.mjv_connector(
            v.user_scn.geoms[v.user_scn.ngeom],
            type=mj.mjtGeom.mjGEOM_ARROW,
            width=0.005,
            from_=pos + pos_offset,
            to=pos + pos_offset + size * (mat @ fix)[:, i],
        )
        v.user_scn.ngeom += 1

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
        motion_root_pos = motion_data["body_pos_w"][:, 0, :].copy()
        motion_root_rot = motion_data["body_quat_w"][:, 0, :].copy() # from xyzw to wxyz
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
        # motion_dof_pos[:, joint_mapping] = motion_data["joint_pos"]
        motion_dof_pos[:, joint_mapping] = motion_data["joint_pos"]
        print(motion_dof_pos)
        
        # Load body positions and quaternions for contact visualization
        body_pos_w = motion_data["body_pos_w"].copy()
        body_quat_w = motion_data["body_quat_w"].copy()
        
        # Load contact mask if available
        contact_mask = motion_data.get("contact_mask", None)
        if contact_mask is not None:
            contact_mask = contact_mask.copy()
        
    return motion_fps, motion_root_pos, motion_root_rot, motion_dof_pos, contact_mask, body_pos_w, body_quat_w

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
    
    motion_fps, motion_root_pos, motion_root_rot, motion_dof_pos, contact_mask, body_pos_w, body_quat_w = load_robot_motion(robot_motion_path)
    
    if contact_mask is not None:
        print(f"Contact mask loaded: shape={contact_mask.shape}")
    else:
        print("No contact_mask found in motion data")
    
    env = RobotMotionViewer(robot_type=robot_type,
                            motion_fps=motion_fps,
                            camera_follow=False,
                            record_video=args.record_video, video_path=args.video_path)
    
    frame_idx = 0
    while True:
        # Update robot pose
        env.data.qpos[:3] = motion_root_pos[frame_idx]
        env.data.qpos[3:7] = motion_root_rot[frame_idx]
        env.data.qpos[7:] = motion_dof_pos[frame_idx]
        mj.mj_forward(env.model, env.data)
        
        if not env.headless:
            # Update camera
            if env.camera_follow:
                env.viewer.cam.lookat = env.data.xpos[env.model.body(env.robot_base).id]
                env.viewer.cam.distance = env.viewer_cam_distance
                env.viewer.cam.elevation = -10
            
            # Draw contact visualization
            if contact_mask is not None:
                # Clean custom geometry
                env.viewer.user_scn.ngeom = 0
                # Draw contact points in red
                for body_idx in range(contact_mask.shape[1]):
                    if contact_mask[frame_idx, body_idx] == 1:
                        body_name = lab_body_name_list[body_idx]
                        # Get body position and rotation from motion data
                        body_pos = body_pos_w[frame_idx, body_idx, :]
                        body_quat = body_quat_w[frame_idx, body_idx, :]  # wxyz format
                        
                        # Draw red contact frame
                        draw_contact_frame(
                            body_pos,
                            R.from_quat(body_quat, scalar_first=True).as_matrix(),
                            env.viewer,
                            0.15,  # size
                            joint_name=body_name,
                            contact_color=[1, 0, 0, 1]  # 红色表示接触
                        )
            
            env.viewer.sync()
            if env.rate_limiter is not None:
                env.rate_limiter.sleep()
        
        if env.record_video:
            # Handle video recording
            if env.headless:
                if env.camera_follow:
                    env.camera.lookat = env.data.xpos[env.model.body(env.robot_base).id]
                env.renderer.update_scene(env.data, camera=env.camera)
            else:
                env.renderer.update_scene(env.data, camera=env.viewer.cam)
            img = env.renderer.render()
            env.mp4_writer.append_data(img)
        
        frame_idx += 1
        if frame_idx >= len(motion_root_pos):
            frame_idx = 0
    env.close()