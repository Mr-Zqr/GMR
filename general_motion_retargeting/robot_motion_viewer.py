import os
import time
import mujoco as mj
import mujoco.viewer as mjv
import imageio
from scipy.spatial.transform import Rotation as R
from general_motion_retargeting import ROBOT_XML_DICT, ROBOT_BASE_DICT, VIEWER_CAM_DISTANCE_DICT
from loop_rate_limiters import RateLimiter
import numpy as np
from rich import print


def draw_frame(
    pos,
    mat,
    v,
    size,
    joint_name=None,
    orientation_correction=R.from_euler("xyz", [0, 0, 0]),
    pos_offset=np.array([0, 0, 0]),
):
    rgba_list = [[1, 0, 0, 1], [0, 1, 0, 1], [0, 0, 1, 1]]
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
        if joint_name is not None:
            geom.label = joint_name  # 这里赋名字
        fix = orientation_correction.as_matrix()
        mj.mjv_connector(
            v.user_scn.geoms[v.user_scn.ngeom],
            type=mj.mjtGeom.mjGEOM_ARROW,
            width=0.005,
            from_=pos + pos_offset,
            to=pos + pos_offset + size * (mat @ fix)[:, i],
        )
        v.user_scn.ngeom += 1

class RobotMotionViewer:
    def __init__(self,
                robot_type,
                camera_follow=True,
                motion_fps=30,
                transparent_robot=0,
                # robot color and offset for multiple robots
                robot_color=None,
                position_offset=None,
                robot_label=None,
                # video recording
                record_video=False,
                video_path=None,
                video_width=1920,
                video_height=1080,
                headless=False,
                camera_azimuth=90,
                camera_elevation=-5,
                camera_lookat_z_offset=-0.3,
                # joint sphere visualization
                show_joint_spheres=False,
                joint_sphere_color=None,
                joint_sphere_radius=0.05,
                joint_sphere_bodies=None,  # list of body names to show; None = all bodies
                # white background and floor
                white_background=False):
        
        self.robot_type = robot_type
        self.xml_path = ROBOT_XML_DICT[robot_type]
        self.model = mj.MjModel.from_xml_path(str(self.xml_path))
        self.data = mj.MjData(self.model)
        self.robot_base = ROBOT_BASE_DICT[robot_type]
        self.viewer_cam_distance = VIEWER_CAM_DISTANCE_DICT[robot_type]
        
        # Set robot color if provided
        self.robot_color = robot_color if robot_color is not None else [0.8, 0.8, 0.8, 1.0]
        self.position_offset = position_offset if position_offset is not None else np.array([0.0, 0.0, 0.0])
        self.robot_label = robot_label
        
        # Apply color to all geoms
        for i in range(self.model.ngeom):
            self.model.geom_rgba[i] = self.robot_color

        # Joint sphere visualization settings
        self.show_joint_spheres = show_joint_spheres
        self.joint_sphere_color = joint_sphere_color if joint_sphere_color is not None else [0.2, 0.5, 0.9, 1.0]
        self.joint_sphere_radius = joint_sphere_radius
        # Pre-compute body indices to render (skip world body 0)
        if joint_sphere_bodies is not None:
            self.joint_sphere_body_ids = [self.model.body(name).id for name in joint_sphere_bodies]
        else:
            self.joint_sphere_body_ids = list(range(1, self.model.nbody))

        # White background: sky + haze → white, floor plane geoms → white (no texture)
        if white_background:
            self.model.vis.rgba.sky[:] = [1.0, 1.0, 1.0, 1.0]
            self.model.vis.rgba.haze[:] = [1.0, 1.0, 1.0, 1.0]
            for i in range(self.model.ngeom):
                if self.model.geom_type[i] == mj.mjtGeom.mjGEOM_PLANE:
                    self.model.geom_rgba[i] = [1.0, 1.0, 1.0, 1.0]
                    self.model.geom_matid[i] = -1

        mj.mj_step(self.model, self.data)
        
        self.motion_fps = motion_fps
        self.rate_limiter = RateLimiter(frequency=self.motion_fps, warn=False)
        self.camera_follow = camera_follow
        self.record_video = record_video
        self.headless = headless

        if not self.headless:
            self.viewer = mjv.launch_passive(
                model=self.model,
                data=self.data,
                show_left_ui=False,
                show_right_ui=False)      

            self.viewer.opt.flags[mj.mjtVisFlag.mjVIS_TRANSPARENT] = transparent_robot
        else:
            self.viewer = None
        
        self.camera_azimuth = camera_azimuth
        self.camera_elevation = camera_elevation
        self.camera_lookat_z_offset = camera_lookat_z_offset

        if self.record_video:
            assert video_path is not None, "Please provide video path for recording"
            # Ensure video path has .mp4 extension
            if not os.path.splitext(video_path)[1]:
                video_path = video_path + ".mp4"
            self.video_path = video_path
            video_dir = os.path.dirname(self.video_path)

            if not os.path.exists(video_dir):
                os.makedirs(video_dir)
            self.mp4_writer = imageio.get_writer(
                self.video_path, fps=self.motion_fps, quality=9, macro_block_size=1)
            print(f"Recording video to {self.video_path}")

            # Ensure MuJoCo offscreen framebuffer is large enough for the requested resolution
            self.model.vis.global_.offwidth = max(self.model.vis.global_.offwidth, video_width)
            self.model.vis.global_.offheight = max(self.model.vis.global_.offheight, video_height)

            # Initialize renderer for video recording
            self.renderer = mj.Renderer(self.model, height=video_height, width=video_width)

            # Set up camera for headless mode
            if self.headless:
                self.camera = mj.MjvCamera()
                self.camera.type = mj.mjtCamera.mjCAMERA_FREE
                self.camera.lookat[:] = self.data.xpos[self.model.body(self.robot_base).id]
                self.camera.lookat[2] += self.camera_lookat_z_offset
                self.camera.distance = self.viewer_cam_distance
                self.camera.elevation = self.camera_elevation
                self.camera.azimuth = self.camera_azimuth
        
    def step(self, 
            # robot data
            root_pos, root_rot, dof_pos, 
            # human data
            human_motion_data=None, 
            show_human_body_name=False,
            # scale for human point visualization
            human_point_scale=0.1,
            # human pos offset add for visualization    
            human_pos_offset=np.array([0.0, 0.0, 0]),
            # rate limit
            rate_limit=True, 
            follow_camera=True,
            # for multi-robot support
            show_label=False,
            ):
        """
        by default visualize robot motion.
        also support visualize human motion by providing human_motion_data, to compare with robot motion.
        
        human_motion_data is a dict of {"human body name": (3d global translation, 3d global rotation)}.

        if rate_limit is True, the motion will be visualized at the same rate as the motion data.
        else, the motion will be visualized as fast as possible.
        """
        
        # Apply position offset
        self.data.qpos[:3] = root_pos + self.position_offset
        self.data.qpos[3:7] = root_rot # quat need to be scalar first! for mujoco
        self.data.qpos[7:] = dof_pos
        
        mj.mj_forward(self.model, self.data)

        if not self.headless and self.show_joint_spheres:
            self.viewer.user_scn.ngeom = 0  # clear previous frame's custom geoms
            for i in self.joint_sphere_body_ids:
                if self.viewer.user_scn.ngeom >= self.viewer.user_scn.maxgeom:
                    break
                geom = self.viewer.user_scn.geoms[self.viewer.user_scn.ngeom]
                mj.mjv_initGeom(
                    geom,
                    type=mj.mjtGeom.mjGEOM_SPHERE,
                    size=[self.joint_sphere_radius] * 3,
                    pos=self.data.xpos[i].copy(),
                    mat=np.eye(3).flatten(),
                    rgba=self.joint_sphere_color,
                )
                self.viewer.user_scn.ngeom += 1

        if not self.headless:
            if follow_camera:
                self.viewer.cam.lookat[:] = self.data.xpos[self.model.body(self.robot_base).id]
                self.viewer.cam.lookat[2] += self.camera_lookat_z_offset
                self.viewer.cam.distance = self.viewer_cam_distance
                self.viewer.cam.elevation = self.camera_elevation
                self.viewer.cam.azimuth = self.camera_azimuth
            
            if human_motion_data is not None:
                # Draw the task targets for reference
                for human_body_name, (pos, rot) in human_motion_data.items():
                    draw_frame(
                        pos,
                        R.from_quat(rot, scalar_first=True).as_matrix(),
                        self.viewer,
                        human_point_scale,
                        pos_offset=human_pos_offset,
                        joint_name=human_body_name if show_human_body_name else None
                        )
            
            # Draw robot label if provided
            if show_label and self.robot_label is not None:
                label_pos = self.data.xpos[self.model.body(self.robot_base).id] + np.array([0, 0, 0.3])
                geom = self.viewer.user_scn.geoms[self.viewer.user_scn.ngeom]
                mj.mjv_initGeom(
                    geom,
                    type=mj.mjtGeom.mjGEOM_SPHERE,
                    size=[0.02, 0.02, 0.02],
                    pos=label_pos,
                    mat=np.eye(3).flatten(),
                    rgba=self.robot_color,
                )
                geom.label = self.robot_label
                self.viewer.user_scn.ngeom += 1

            if not self.viewer.is_running():
                return False
            self.viewer.sync()
            if rate_limit is True:
                self.rate_limiter.sleep()

        if self.record_video:
            # Use renderer for proper offscreen rendering
            if self.headless:
                # Update camera position for headless mode
                if follow_camera:
                    self.camera.lookat[:] = self.data.xpos[self.model.body(self.robot_base).id]
                    self.camera.lookat[2] += self.camera_lookat_z_offset
                    self.camera.distance = self.viewer_cam_distance
                    self.camera.elevation = self.camera_elevation
                    self.camera.azimuth = self.camera_azimuth
                self.renderer.update_scene(self.data, camera=self.camera)
            else:
                self.renderer.update_scene(self.data, camera=self.viewer.cam)
            # Draw joint spheres into renderer scene
            if self.show_joint_spheres:
                for i in self.joint_sphere_body_ids:
                    if self.renderer.scene.ngeom >= self.renderer.scene.maxgeom:
                        break
                    geom = self.renderer.scene.geoms[self.renderer.scene.ngeom]
                    mj.mjv_initGeom(
                        geom,
                        type=mj.mjtGeom.mjGEOM_SPHERE,
                        size=[self.joint_sphere_radius] * 3,
                        pos=self.data.xpos[i].copy(),
                        mat=np.eye(3).flatten(),
                        rgba=self.joint_sphere_color,
                    )
                    self.renderer.scene.ngeom += 1
            img = self.renderer.render()
            self.mp4_writer.append_data(img)
        return True

    def close(self):
        if self.viewer is not None:
            self.viewer.close()
            time.sleep(0.5)
        if self.record_video:
            self.mp4_writer.close()
            print(f"Video saved to {self.video_path}")
