import numpy as np
from general_motion_retargeting import RobotMotionViewer
import argparse
import os
import mujoco as mj
import mujoco.viewer as mjv
from scipy.spatial.transform import Rotation as R


def load_robot_motion(motion_file):
    """
    Load robot motion data from a pickle file.
    """
    with open(motion_file, "rb") as f:
        motion_data = np.load(f, allow_pickle=True)
        motion_fps = 50
        
        motion_root_pos = motion_data["body_pos_w"][:, 0, :]
        motion_root_rot = motion_data["body_quat_w"][:, 0, :]
        
        # Get first frame global pose for alignment
        first_frame_pos = motion_data["body_pos_w"][0, 0, :].copy()
        first_frame_rot = motion_data["body_quat_w"][0, 0, :].copy()
        
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
        motion_body_pos = motion_data["body_pos_w"]
        motion_body_rot = motion_data["body_quat_w"]
        
    return {
        'root_pos': motion_root_pos,
        'root_rot': motion_root_rot,
        'dof_pos': motion_dof_pos,
        'first_frame_pos': first_frame_pos,
        'first_frame_rot': first_frame_rot,
        'fps': motion_fps
    }


def align_motion_to_reference(motion_data, ref_pos, ref_rot):
    """
    Align motion data so that its first frame matches the reference pose.
    
    Args:
        motion_data: dict with 'root_pos', 'root_rot', 'first_frame_pos', 'first_frame_rot'
        ref_pos: reference position (3,)
        ref_rot: reference rotation quaternion wxyz (4,)
    
    Returns:
        Aligned root_pos and root_rot
    """
    # Get first frame pose
    first_pos = motion_data['first_frame_pos']
    first_rot = motion_data['first_frame_rot']
    
    # Calculate position offset
    pos_offset = ref_pos - first_pos
    
    # Calculate rotation offset: ref_rot * inv(first_rot)
    first_rot_scipy = R.from_quat([first_rot[1], first_rot[2], first_rot[3], first_rot[0]])  # wxyz to xyzw
    ref_rot_scipy = R.from_quat([ref_rot[1], ref_rot[2], ref_rot[3], ref_rot[0]])  # wxyz to xyzw
    rot_offset = ref_rot_scipy * first_rot_scipy.inv()
    
    # Apply offset to all frames
    aligned_pos = motion_data['root_pos'].copy()
    aligned_rot = motion_data['root_rot'].copy()
    
    for i in range(len(aligned_pos)):
        # Rotate position around reference point
        relative_pos = aligned_pos[i] - first_pos
        rotated_relative_pos = rot_offset.apply(relative_pos)
        aligned_pos[i] = ref_pos + rotated_relative_pos
        
        # Rotate orientation
        curr_rot_scipy = R.from_quat([aligned_rot[i, 1], aligned_rot[i, 2], aligned_rot[i, 3], aligned_rot[i, 0]])
        new_rot_scipy = rot_offset * curr_rot_scipy
        new_rot_xyzw = new_rot_scipy.as_quat()  # xyzw
        aligned_rot[i] = [new_rot_xyzw[3], new_rot_xyzw[0], new_rot_xyzw[1], new_rot_xyzw[2]]  # wxyz
    
    return aligned_pos, aligned_rot


class MultiRobotMotionViewer:
    """
    Viewer for visualizing multiple robot motions simultaneously with different colors.
    Robots are semi-transparent and aligned at first frame.
    """
    def __init__(self, 
                 robot_type,
                 motion_files,
                 motion_fps=50,
                 transparency=0.6,
                 record_video=False,
                 video_path=None,
                 video_width=1280,
                 video_height=720):
        """
        Args:
            robot_type: Type of robot to visualize
            motion_files: List of paths to motion files (1-3 files)
            motion_fps: FPS for playback
            transparency: Alpha value for robot transparency (0=invisible, 1=opaque)
            record_video: Whether to record video
            video_path: Path to save video
            video_width: Video width
            video_height: Video height
        """
        self.robot_type = robot_type
        self.motion_fps = motion_fps
        self.transparency = transparency
        
        # Predefined colors for up to 3 robots (with transparency)
        self.colors = [
            [1.0, 0.3, 0.3, transparency],  # Red
            [0.3, 0.7, 1.0, transparency],  # Blue
            [0.3, 1.0, 0.3, transparency],  # Green
        ]
        
        # Load all motion data
        self.motions = []
        self.motion_names = []
        
        for i, motion_file in enumerate(motion_files):
            if not os.path.exists(motion_file):
                raise FileNotFoundError(f"Motion file {motion_file} not found")
            
            motion_data = load_robot_motion(motion_file)
            self.motions.append(motion_data)
            self.motion_names.append(os.path.basename(motion_file))
        
        num_robots = len(self.motions)
        if num_robots > 3:
            raise ValueError("Maximum 3 robots supported")
        
        # Align all motions to first motion's first frame
        if num_robots > 1:
            ref_pos = self.motions[0]['first_frame_pos']
            ref_rot = self.motions[0]['first_frame_rot']
            
            for i in range(1, num_robots):
                aligned_pos, aligned_rot = align_motion_to_reference(
                    self.motions[i], ref_pos, ref_rot
                )
                self.motions[i]['root_pos'] = aligned_pos
                self.motions[i]['root_rot'] = aligned_rot
            
            print(f"Aligned {num_robots} motions to first motion's first frame")
        
        # Create MuJoCo model and viewer
        from general_motion_retargeting import ROBOT_XML_DICT, ROBOT_BASE_DICT, VIEWER_CAM_DISTANCE_DICT
        
        self.xml_path = ROBOT_XML_DICT[robot_type]
        self.robot_base = ROBOT_BASE_DICT[robot_type]
        self.viewer_cam_distance = VIEWER_CAM_DISTANCE_DICT[robot_type]
        
        # Create models for each robot
        self.models = []
        self.datas = []
        
        for i in range(num_robots):
            model = mj.MjModel.from_xml_path(str(self.xml_path))
            data = mj.MjData(model)
            
            # Apply color with transparency to all geoms
            color = self.colors[i]
            for j in range(model.ngeom):
                model.geom_rgba[j] = color
            
            self.models.append(model)
            self.datas.append(data)
        
        # Create viewer with first model
        self.viewer = mjv.launch_passive(
            model=self.models[0],
            data=self.datas[0],
            show_left_ui=False,
            show_right_ui=False
        )
        
        # Enable transparency rendering
        self.viewer.opt.flags[mj.mjtVisFlag.mjVIS_TRANSPARENT] = True
        
        # Create perturbation object for mjv_addGeoms (required parameter)
        self.pert = mj.MjvPerturb()
        
        self.record_video = record_video
        if self.record_video:
            assert video_path is not None, "Please provide video path for recording"
            self.video_path = video_path
            video_dir = os.path.dirname(self.video_path)
            
            if video_dir and not os.path.exists(video_dir):
                os.makedirs(video_dir)
            
            import imageio
            self.mp4_writer = imageio.get_writer(self.video_path, fps=self.motion_fps)
            print(f"Recording video to {self.video_path}")
            
            # Initialize renderer for video recording
            self.renderer = mj.Renderer(self.models[0], height=video_height, width=video_width)
            
            # Create a separate scene for rendering with all robots
            self.render_scene = mj.MjvScene(self.models[0], maxgeom=10000)
            self.render_cam = mj.MjvCamera()
            self.render_opt = mj.MjvOption()
        
        # Rate limiter
        from loop_rate_limiters import RateLimiter
        self.rate_limiter = RateLimiter(frequency=self.motion_fps, warn=False)
    
    def step(self, frame_idx, rate_limit=True):
        """
        Update all robots for the given frame index
        """
        # Clear custom geometry
        self.viewer.user_scn.ngeom = 0
        
        # Update each robot's state first
        for i, (model, data, motion) in enumerate(zip(self.models, self.datas, self.motions)):
            # Handle looping
            idx = frame_idx % len(motion['root_pos'])
            
            # Update robot state (no offset - already aligned)
            data.qpos[:3] = motion['root_pos'][idx]
            data.qpos[3:7] = motion['root_rot'][idx]
            data.qpos[7:] = motion['dof_pos'][idx]
            
            mj.mj_forward(model, data)
        
        # Add labels for all robots
        for i, (model, data, motion) in enumerate(zip(self.models, self.datas, self.motions)):
            # Add label above robot (slightly offset for visibility)
            label_offset = np.array([0.1 * i, 0.1 * i, 0.3 + 0.15 * i])
            label_pos = data.xpos[model.body(self.robot_base).id] + label_offset
            geom = self.viewer.user_scn.geoms[self.viewer.user_scn.ngeom]
            mj.mjv_initGeom(
                geom,
                type=mj.mjtGeom.mjGEOM_SPHERE,
                size=[0.03, 0.03, 0.03],
                pos=label_pos,
                mat=np.eye(3).flatten(),
                rgba=[self.colors[i][0], self.colors[i][1], self.colors[i][2], 1.0],  # Solid color for label
            )
            geom.label = f"{i+1}: {self.motion_names[i]}"
            self.viewer.user_scn.ngeom += 1
        
        # Add additional robots to the scene using mjv_addGeoms
        for i in range(1, len(self.models)):
            model = self.models[i]
            data = self.datas[i]
            
            # Use mjv_addGeoms to properly add all geoms including meshes
            mj.mjv_addGeoms(
                model, 
                data, 
                self.viewer.opt,  # visualization options
                self.pert,        # perturbation object (required)
                mj.mjtCatBit.mjCAT_ALL,  # all categories
                self.viewer.user_scn     # add to user scene
            )
        
        # Update camera to follow first robot
        self.viewer.cam.lookat = self.datas[0].xpos[self.models[0].body(self.robot_base).id]
        self.viewer.cam.distance = self.viewer_cam_distance
        self.viewer.cam.elevation = -10
        
        self.viewer.sync()
        
        if rate_limit:
            self.rate_limiter.sleep()
        
        if self.record_video:
            # Update camera settings
            self.render_cam.lookat = self.datas[0].xpos[self.models[0].body(self.robot_base).id].copy()
            self.render_cam.distance = self.viewer_cam_distance
            self.render_cam.elevation = -10
            self.render_cam.azimuth = self.viewer.cam.azimuth  # Match viewer azimuth
            
            # Enable transparency in render options
            self.render_opt.flags[mj.mjtVisFlag.mjVIS_TRANSPARENT] = True
            
            # Update scene with first robot
            mj.mjv_updateScene(
                self.models[0], 
                self.datas[0], 
                self.render_opt, 
                self.pert, 
                self.render_cam, 
                mj.mjtCatBit.mjCAT_ALL, 
                self.render_scene
            )
            
            # Add additional robots to the render scene
            for i in range(1, len(self.models)):
                mj.mjv_addGeoms(
                    self.models[i], 
                    self.datas[i], 
                    self.render_opt,
                    self.pert,
                    mj.mjtCatBit.mjCAT_ALL,
                    self.render_scene
                )
            
            # Add labels to render scene
            for i, (model, data, motion) in enumerate(zip(self.models, self.datas, self.motions)):
                if self.render_scene.ngeom < self.render_scene.maxgeom:
                    label_offset = np.array([0.1 * i, 0.1 * i, 0.3 + 0.15 * i])
                    label_pos = data.xpos[model.body(self.robot_base).id] + label_offset
                    geom = self.render_scene.geoms[self.render_scene.ngeom]
                    mj.mjv_initGeom(
                        geom,
                        type=mj.mjtGeom.mjGEOM_SPHERE,
                        size=[0.03, 0.03, 0.03],
                        pos=label_pos,
                        mat=np.eye(3).flatten(),
                        rgba=[self.colors[i][0], self.colors[i][1], self.colors[i][2], 1.0],
                    )
                    geom.label = f"{i+1}: {self.motion_names[i]}"
                    self.render_scene.ngeom += 1
            
            # Render the scene directly
            self.renderer._scene = self.render_scene
            img = self.renderer.render()
            self.mp4_writer.append_data(img)
    
    def close(self):
        if self.viewer is not None:
            self.viewer.close()
            import time
            time.sleep(0.5)
        if self.record_video:
            self.mp4_writer.close()
            print(f"Video saved to {self.video_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize multiple robot motions simultaneously")
    parser.add_argument("--robot", type=str, default="unitree_g1",
                        help="Robot type")
    parser.add_argument("--motion_files", type=str, nargs='+', required=True,
                        help="Paths to motion files (1-3 files)")
    parser.add_argument("--transparency", type=float, default=0.6,
                        help="Robot transparency (0=invisible, 1=opaque)")
    parser.add_argument("--record_video", action="store_true",
                        help="Record video of visualization")
    parser.add_argument("--video_path", type=str, 
                        default="videos/multi_robot_comparison.mp4",
                        help="Path to save video")
    
    args = parser.parse_args()
    
    if len(args.motion_files) < 1 or len(args.motion_files) > 3:
        raise ValueError("Please provide 1-3 motion files")
    
    print(f"Loading {len(args.motion_files)} motion files:")
    for i, f in enumerate(args.motion_files):
        print(f"  {i+1}. {f}")
    
    viewer = MultiRobotMotionViewer(
        robot_type=args.robot,
        motion_files=args.motion_files,
        transparency=args.transparency,
        record_video=args.record_video,
        video_path=args.video_path
    )
    
    frame_idx = 0
    try:
        while True:
            viewer.step(frame_idx, rate_limit=True)
            frame_idx += 1
    except KeyboardInterrupt:
        print("\nStopping visualization...")
    finally:
        viewer.close()
