"""Visualize the IK optimization process across all frames of motion retargeting.

Flattens all IK iterations across all frames into a single timeline.
Step through with '.' (forward) and ',' (backward) like a video.

Each frame shows: initial pose -> Phase 1 iterations -> Phase 2 iterations -> next frame...

Features:
- Target skeleton (green stick figure)
- Sphere size/color per joint based on IK weights
- Error lines from robot end-effectors to targets
- Lazy frame computation (next frame computed on demand)
- Optional video recording with fixed world-space camera

Usage:
    python scripts/vis_ik_optimization.py \
        --smplx_file <path.npz> \
        --robot unitree_g1 \
        [--yup_to_zup] \
        [--auto_play] \
        [--play_interval 0.5] \
        [--record_video --video_path out.mp4] \
        [--video_width 1920 --video_height 1080 --video_quality 8] \
        [--cam_lookat X Y Z] [--cam_distance D] [--cam_azimuth A] [--cam_elevation E]
"""
import argparse
import pathlib
import time

import imageio
import numpy as np
import mujoco as mj
import mujoco.viewer as mjv
from PIL import Image, ImageDraw, ImageFont
from scipy.spatial.transform import Rotation as R

from general_motion_retargeting import GeneralMotionRetargeting as GMR
from general_motion_retargeting import ROBOT_XML_DICT, ROBOT_BASE_DICT, VIEWER_CAM_DISTANCE_DICT
from general_motion_retargeting.utils.smpl import load_smplx_file, get_smplx_data_offline_fast
from rich import print


HERE = pathlib.Path(__file__).parent
SMPLX_FOLDER = HERE / ".." / "assets" / "body_models"

# Human skeleton connectivity for stick figure
HUMAN_SKELETON_EDGES = [
    ("pelvis", "spine3"),
    ("pelvis", "left_hip"),
    ("pelvis", "right_hip"),
    ("left_hip", "left_knee"),
    ("right_hip", "right_knee"),
    ("left_knee", "left_foot"),
    ("right_knee", "right_foot"),
    ("spine3", "left_shoulder"),
    ("spine3", "right_shoulder"),
    ("left_shoulder", "left_elbow"),
    ("right_shoulder", "right_elbow"),
    ("left_elbow", "left_wrist"),
    ("right_elbow", "right_wrist"),
]


def weight_to_color(pos_w, rot_w):
    total = pos_w + rot_w
    if total == 0:
        return [0.5, 0.5, 0.5, 0.4]
    t = min(1.0, np.log10(max(total, 1)) / np.log10(200))
    if t < 0.5:
        s = t * 2
        return [s, s, 1.0 - s, 0.6 + 0.4 * t]
    else:
        s = (t - 0.5) * 2
        return [1.0, 1.0 - s, 0.0, 0.8 + 0.2 * s]


def weight_to_radius(pos_w, rot_w):
    total = pos_w + rot_w
    if total == 0:
        return 0.015
    return 0.015 + 0.035 * min(1.0, np.log10(max(total, 1)) / np.log10(200))


def get_weights_for_phase(retarget, phase):
    if phase <= 1:
        table = retarget.ik_match_table1
    else:
        table = retarget.ik_match_table2
    weights = {}
    for robot_frame, entry in table.items():
        weights[entry[0]] = (entry[1], entry[2])
    return weights


COLOR_RGB = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]       # target: red/green/blue
COLOR_CMY = [[0, 1, 1], [1, 0, 1], [1, 1, 0]]       # robot:  cyan/magenta/yellow
COLOR_RGB_LIGHT = [[1, 0.5, 0.5], [0.5, 1, 0.5], [0.5, 0.5, 1]]  # robot hands: light RGB

# Only draw coordinate frames for these human body names
FRAME_BODIES = {"pelvis", "left_wrist", "right_wrist", "left_foot", "right_foot"}
HAND_BODIES = {"left_wrist", "right_wrist"}


def _rotation_mat_z_to(from_pt: np.ndarray, to_pt: np.ndarray) -> np.ndarray:
    """3×3 rotation matrix whose z-column points from from_pt toward to_pt."""
    d = np.asarray(to_pt, np.float64) - np.asarray(from_pt, np.float64)
    n = np.linalg.norm(d)
    if n < 1e-9:
        return np.eye(3)
    z = d / n
    ref = np.array([1.0, 0.0, 0.0]) if abs(z[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    x = ref - np.dot(ref, z) * z
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    return np.column_stack([x, y, z])


def _add_connector(scene, type_: int, width: float, from_pt, to_pt, rgba) -> None:
    """Drop-in for mjv_connector using mjv_initGeom — works on both viewer.user_scn and renderer.scene."""
    if scene.ngeom >= scene.maxgeom - 1:
        return
    from_pt = np.asarray(from_pt, np.float64)
    to_pt = np.asarray(to_pt, np.float64)
    length = np.linalg.norm(to_pt - from_pt)
    if length < 1e-9:
        return
    pos = (from_pt + to_pt) / 2.0
    mat = _rotation_mat_z_to(from_pt, to_pt)

    if type_ == mj.mjtGeom.mjGEOM_CAPSULE:
        half_cyl = max(1e-6, length / 2.0 - width)
        size = [width, half_cyl, 0.0]
    elif type_ in (mj.mjtGeom.mjGEOM_ARROW, mj.mjtGeom.mjGEOM_ARROW1, mj.mjtGeom.mjGEOM_ARROW2):
        size = [width, length / 2.0, 0.0]
        type_ = mj.mjtGeom.mjGEOM_CYLINDER
    else:  # LINE and others → thin capsule
        size = [width * 0.5, max(1e-6, length / 2.0 - width * 0.5), 0.0]
        type_ = mj.mjtGeom.mjGEOM_CAPSULE

    geom = scene.geoms[scene.ngeom]
    mj.mjv_initGeom(geom, type=type_, size=size, pos=pos,
                    mat=mat.flatten(), rgba=np.asarray(rgba, np.float32))
    scene.ngeom += 1


# MjvOption with labels/frames disabled — used for offscreen renderer
_SCENE_OPT = mj.MjvOption()
_SCENE_OPT.label = mj.mjtLabel.mjLABEL_NONE
_SCENE_OPT.frame = mj.mjtFrame.mjFRAME_NONE

# System fonts to try for HUD text overlay (in order of preference)
_FONT_CANDIDATES = [
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
    "/usr/share/fonts/truetype/freefont/FreeSansBold.ttf",
    "/usr/share/fonts/truetype/ubuntu/Ubuntu-B.ttf",
]


def _load_font(size: int):
    for path in _FONT_CANDIDATES:
        if pathlib.Path(path).exists():
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


def overlay_hud(img: np.ndarray, text_left: str, text_right: str,
                font_size: int = 32) -> np.ndarray:
    """Overlay HUD text onto a (H, W, 3) uint8 RGB numpy array."""
    pil = Image.fromarray(img)
    draw = ImageDraw.Draw(pil)
    font = _load_font(font_size)
    pad = 16
    # Shadow then white text (top-left)
    for dx, dy in ((1, 1), (-1, -1), (1, -1), (-1, 1)):
        draw.text((pad + dx, pad + dy), text_left, font=font, fill=(0, 0, 0))
    draw.text((pad, pad), text_left, font=font, fill=(255, 255, 255))
    # Right-aligned text (top-right)
    try:
        bbox = font.getbbox(text_right)
        tw = bbox[2] - bbox[0]
    except AttributeError:
        tw = font.getlength(text_right)
    rx = pil.width - tw - pad
    for dx, dy in ((1, 1), (-1, -1), (1, -1), (-1, 1)):
        draw.text((rx + dx, pad + dy), text_right, font=font, fill=(0, 0, 0))
    draw.text((rx, pad), text_right, font=font, fill=(255, 255, 255))
    return np.array(pil)


def draw_frame_styled(pos, mat, scene, size, width=0.005, alpha=1.0, label=None,
                      colors=None):
    """Draw orientation frame arrows into a mjvScene. colors: list of 3 RGB triples."""
    if colors is None:
        colors = COLOR_RGB
    for i in range(3):
        if scene.ngeom >= scene.maxgeom - 1:
            break
        rgba = [colors[i][0], colors[i][1], colors[i][2], alpha]
        tip = pos + size * mat[:, i]
        _add_connector(scene, mj.mjtGeom.mjGEOM_ARROW, width, pos, tip, rgba)
        if i == 0 and label is not None:
            scene.geoms[scene.ngeom - 1].label = label


def draw_skeleton_and_targets(scene, scaled_human_data, weights, human_to_robot_frame,
                               model, data, reset=True):
    """Draw skeleton overlay into a mjvScene (viewer.user_scn or renderer.scene).

    reset=True clears the scene first (use for viewer.user_scn).
    reset=False appends after update_scene (use for renderer.scene).
    """
    if reset:
        scene.ngeom = 0

    # 1) Target skeleton edges (green capsules)
    for parent, child in HUMAN_SKELETON_EDGES:
        if parent not in scaled_human_data or child not in scaled_human_data:
            continue
        if scene.ngeom >= scene.maxgeom - 1:
            break
        pos_from = np.asarray(scaled_human_data[parent][0], dtype=np.float64)
        pos_to = np.asarray(scaled_human_data[child][0], dtype=np.float64)
        _add_connector(scene, mj.mjtGeom.mjGEOM_CAPSULE, 0.008,
                       pos_from, pos_to, [0.2, 0.8, 0.2, 0.5])

    # 2) Target spheres + orientation frames (bright, thick arrows)
    for human_body_name, (pos, rot) in scaled_human_data.items():
        if scene.ngeom >= scene.maxgeom - 10:
            break
        pos_w, rot_w = weights.get(human_body_name, (0, 0))
        color = weight_to_color(pos_w, rot_w)
        radius = weight_to_radius(pos_w, rot_w)

        # Sphere
        geom = scene.geoms[scene.ngeom]
        mj.mjv_initGeom(geom, type=mj.mjtGeom.mjGEOM_SPHERE,
                         size=[radius] * 3,
                         pos=np.asarray(pos, dtype=np.float64),
                         mat=np.eye(3).flatten(), rgba=color)
        scene.ngeom += 1

        # Target orientation frame: only for hands, feet, root
        if human_body_name in FRAME_BODIES:
            frame_scale = 0.20 + 0.10 * min(1.0, (pos_w + rot_w) / 100.0)
            target_mat = R.from_quat(rot, scalar_first=True).as_matrix()
            draw_frame_styled(np.asarray(pos, dtype=np.float64), target_mat,
                              scene, frame_scale, width=0.010, alpha=1.0)

    # 3) Robot joint orientation frames (dim, thin arrows) + error lines
    for human_name, robot_frame_name in human_to_robot_frame.items():
        if human_name not in scaled_human_data:
            continue
        if scene.ngeom >= scene.maxgeom - 8:
            break
        robot_body_id = model.body(robot_frame_name).id
        robot_pos = data.xpos[robot_body_id].copy()
        robot_mat = data.xmat[robot_body_id].reshape(3, 3).copy()
        target_pos = np.asarray(scaled_human_data[human_name][0], dtype=np.float64)

        pos_w, rot_w = weights.get(human_name, (0, 0))

        # Robot joint orientation frame: only for hands, feet, root
        if human_name in FRAME_BODIES:
            robot_frame_scale = 0.18 + 0.10 * min(1.0, (pos_w + rot_w) / 100.0)
            robot_colors = COLOR_RGB_LIGHT if human_name in HAND_BODIES else COLOR_CMY
            draw_frame_styled(robot_pos, robot_mat, scene, robot_frame_scale,
                              width=0.008, alpha=0.8, colors=robot_colors)

        # Error line connecting robot joint to target
        dist = np.linalg.norm(robot_pos - target_pos)
        err_t = min(1.0, dist / 0.3)
        line_width = 0.001 + 0.004 * min(1.0, (pos_w + rot_w) / 100.0)
        _add_connector(scene, mj.mjtGeom.mjGEOM_LINE, line_width,
                       robot_pos, target_pos, [1.0, 1.0 - 0.7 * err_t, 0.0, 0.7])


class LazyTraceTimeline:
    """Lazily computes IK traces frame-by-frame, flattening into a single step list."""

    def __init__(self, retarget, smplx_frames):
        self.retarget = retarget
        self.smplx_frames = smplx_frames
        self.total_frames = len(smplx_frames)
        self.next_frame_to_compute = 0

        # Flat list of steps: each is (frame_idx, snap_dict, scaled_human_data)
        self.steps = []
        # Index of first step for each frame
        self.frame_start_indices = []

    @property
    def num_steps(self):
        return len(self.steps)

    def ensure_frame(self, frame_idx):
        """Ensure all frames up to frame_idx are computed."""
        while self.next_frame_to_compute <= frame_idx and self.next_frame_to_compute < self.total_frames:
            self._compute_frame(self.next_frame_to_compute)
            self.next_frame_to_compute += 1

    def _compute_frame(self, frame_idx):
        self.frame_start_indices.append(len(self.steps))
        trace, scaled_human_data = self.retarget.retarget_with_trace(
            self.smplx_frames[frame_idx]
        )
        for snap in trace:
            snap["frame_idx"] = frame_idx
            self.steps.append((frame_idx, snap, scaled_human_data))

    def get_step(self, step_idx):
        """Get step at index, computing more frames if needed."""
        # Compute frames until we have enough steps
        while step_idx >= len(self.steps) and self.next_frame_to_compute < self.total_frames:
            self._compute_frame(self.next_frame_to_compute)
            self.next_frame_to_compute += 1
        if step_idx < len(self.steps):
            return self.steps[step_idx]
        return None

    def is_exhausted(self):
        return self.next_frame_to_compute >= self.total_frames

    def max_known_step(self):
        return len(self.steps) - 1


def main():
    parser = argparse.ArgumentParser(description="Visualize IK optimization process")
    parser.add_argument("--smplx_file", type=str, required=True)
    parser.add_argument("--robot", type=str, default="unitree_g1")
    parser.add_argument("--yup_to_zup", default=False, action="store_true")
    parser.add_argument("--auto_play", default=False, action="store_true")
    parser.add_argument("--play_interval", type=float, default=0.5)
    # Camera (fixed world-space)
    parser.add_argument("--cam_lookat", type=float, nargs=3, default=None,
                        metavar=("X", "Y", "Z"),
                        help="Fixed camera lookat in world coords (default: first frame base pos)")
    parser.add_argument("--cam_distance", type=float, default=None,
                        help="Camera distance (default: robot-specific)")
    parser.add_argument("--cam_azimuth", type=float, default=90.0)
    parser.add_argument("--cam_elevation", type=float, default=-10.0)
    # Video recording
    parser.add_argument("--record_video", action="store_true")
    parser.add_argument("--video_path", type=str, default="output_ik_opt.mp4")
    parser.add_argument("--video_width", type=int, default=1920)
    parser.add_argument("--video_height", type=int, default=1080)
    parser.add_argument("--video_quality", type=int, default=8,
                        help="Video quality 0-10 (imageio/ffmpeg, default: 8)")
    parser.add_argument("--headless", action="store_true",
                        help="No interactive viewer; render video headlessly (implies --record_video --auto_play)")
    args = parser.parse_args()

    # --headless implies --record_video and --auto_play
    if args.headless:
        args.record_video = True
        args.auto_play = True

    # Load SMPL-X data
    tgt_fps = 30
    smplx_data, body_model, smplx_output, actual_human_height = load_smplx_file(
        args.smplx_file, SMPLX_FOLDER, downsample_fps=tgt_fps
    )
    smplx_data_frames, aligned_fps = get_smplx_data_offline_fast(
        smplx_data, body_model, smplx_output, tgt_fps=tgt_fps, yup_to_zup=args.yup_to_zup
    )
    print(f"Loaded {len(smplx_data_frames)} frames at {aligned_fps} FPS")

    # Create retarget (state carries over between frames, just like real pipeline)
    retarget = GMR(
        actual_human_height=actual_human_height,
        src_human="smplx",
        tgt_robot=args.robot,
        verbose=False,
    )

    # Build mapping
    human_to_robot_frame = {}
    for human_name, task in retarget.human_body_to_task1.items():
        human_to_robot_frame[human_name] = task.frame_name
    for human_name, task in retarget.human_body_to_task2.items():
        human_to_robot_frame[human_name] = task.frame_name

    # Print weight table
    print("[bold]IK Weights (pos/rot):[/bold]")
    w1 = get_weights_for_phase(retarget, 1)
    w2 = get_weights_for_phase(retarget, 2)
    all_names = sorted(set(w1.keys()) | set(w2.keys()))
    print(f"  {'body':<16s}  {'Phase1':>10s}  {'Phase2':>10s}")
    for name in all_names:
        p1, r1 = w1.get(name, (0, 0))
        p2, r2 = w2.get(name, (0, 0))
        print(f"  {name:<16s}  {p1:>3}/{r1:<3}    {p2:>3}/{r2:<3}")
    print()

    # Lazy timeline
    timeline = LazyTraceTimeline(retarget, smplx_data_frames)
    # Pre-compute frame 0
    timeline.ensure_frame(0)
    print(f"Frame 0: {timeline.num_steps} steps")

    # Normalize video path extension
    video_path = args.video_path
    if args.record_video and not pathlib.Path(video_path).suffix:
        video_path = video_path + ".mp4"

    # MuJoCo model shared by viewer (or used standalone in headless mode)
    model = mj.MjModel.from_xml_path(str(ROBOT_XML_DICT[args.robot]))
    data = mj.MjData(model)
    robot_base_id = model.body(ROBOT_BASE_DICT[args.robot]).id
    cam_distance = args.cam_distance or VIEWER_CAM_DISTANCE_DICT[args.robot]

    # Determine fixed camera lookat from first frame's robot base position
    if args.cam_lookat is not None:
        cam_lookat = np.array(args.cam_lookat)
    else:
        first_step = timeline.get_step(0)
        _, first_snap, _ = first_step
        tmp_data = mj.MjData(model)
        tmp_data.qpos[:] = first_snap["qpos"]
        mj.mj_forward(model, tmp_data)
        cam_lookat = tmp_data.xpos[robot_base_id].copy()

    # ── Headless mode: no viewer, pure offscreen recording ──────────────────
    if args.headless:
        # Must enlarge offscreen framebuffer BEFORE creating Renderer —
        # otherwise the internal GL framebuffer is too small and update_scene segfaults.
        model.vis.global_.offwidth = max(model.vis.global_.offwidth, args.video_width)
        model.vis.global_.offheight = max(model.vis.global_.offheight, args.video_height)
        renderer = mj.Renderer(model, height=args.video_height, width=args.video_width)
        rec_cam = mj.MjvCamera()
        rec_cam.type = mj.mjtCamera.mjCAMERA_FREE
        rec_cam.lookat[:] = cam_lookat
        rec_cam.distance = cam_distance
        rec_cam.elevation = args.cam_elevation
        rec_cam.azimuth = args.cam_azimuth

        video_dir = pathlib.Path(video_path).parent
        video_dir.mkdir(parents=True, exist_ok=True)
        mp4_writer = imageio.get_writer(
            video_path, fps=tgt_fps,
            quality=args.video_quality, macro_block_size=1,
        )
        print(f"Headless recording to [bold]{video_path}[/bold] "
              f"({args.video_width}x{args.video_height}, quality={args.video_quality})")

        # Each IK step is held for this many video frames to match play_interval speed
        n_repeat = max(1, round(args.play_interval * tgt_fps))

        step_idx = 0
        total_steps = None
        while True:
            result = timeline.get_step(step_idx)
            if result is None:
                break
            frame_idx, snap, scaled_human_data = result

            data.qpos[:] = snap["qpos"]
            mj.mj_forward(model, data)

            phase, it, err = snap["phase"], snap["iteration"], snap["error"]
            if phase == 0:
                phase_label = "Initial"
            elif it == -1:
                phase_label = f"Phase {phase} start"
            else:
                phase_label = f"Phase {phase}, Iter {it}"
            if total_steps is None and timeline.is_exhausted():
                total_steps = timeline.max_known_step()
            total_str = f"/{total_steps}" if total_steps is not None else "+..."
            print(f"[Step {step_idx}{total_str}] Frame {frame_idx}/{timeline.total_frames-1}, "
                  f"{phase_label}, Error: {err:.4f}")

            weights = get_weights_for_phase(retarget, snap["phase"])
            renderer.update_scene(data, camera=rec_cam, scene_option=_SCENE_OPT)
            draw_skeleton_and_targets(renderer.scene, scaled_human_data, weights,
                                      human_to_robot_frame, model, data, reset=False)
            raw = renderer.render()

            # HUD overlay: phase info top-left, step number top-right
            hud_left = f"Frame {frame_idx}/{timeline.total_frames-1}\n{phase_label}"
            hud_right = f"Step {step_idx}{total_str}"
            frame_img = overlay_hud(raw, hud_left, hud_right)

            for _ in range(n_repeat):
                mp4_writer.append_data(frame_img)

            step_idx += 1

        mp4_writer.close()
        print(f"Video saved to [bold]{video_path}[/bold]")
        return

    # ── Interactive viewer mode ──────────────────────────────────────────────
    current_step = [0]
    prev_step = [-1]

    def key_callback(keycode):
        if keycode == 262 or keycode == ord('.'):  # right arrow or '.'
            next_idx = current_step[0] + 1
            result = timeline.get_step(next_idx)
            if result is not None:
                current_step[0] = next_idx
        elif keycode == 263 or keycode == ord(','):  # left arrow or ','
            current_step[0] = max(current_step[0] - 1, 0)

    viewer = mjv.launch_passive(
        model, data, key_callback=key_callback,
        show_left_ui=False, show_right_ui=False
    )

    # Set fixed camera once — user can still orbit/zoom interactively
    viewer.cam.lookat[:] = cam_lookat
    viewer.cam.distance = cam_distance
    viewer.cam.elevation = args.cam_elevation
    viewer.cam.azimuth = args.cam_azimuth

    # Video recording in interactive mode: separate model/data to avoid
    # GL context conflicts between GLFW viewer and EGL renderer.
    renderer = None
    rec_model = None
    rec_data = None
    mp4_writer = None
    rec_cam = None
    if args.record_video:
        rec_model = mj.MjModel.from_xml_path(str(ROBOT_XML_DICT[args.robot]))
        rec_data = mj.MjData(rec_model)
        rec_model.vis.global_.offwidth = max(rec_model.vis.global_.offwidth, args.video_width)
        rec_model.vis.global_.offheight = max(rec_model.vis.global_.offheight, args.video_height)
        renderer = mj.Renderer(rec_model, height=args.video_height, width=args.video_width)

        rec_cam = mj.MjvCamera()
        rec_cam.type = mj.mjtCamera.mjCAMERA_FREE
        rec_cam.lookat[:] = cam_lookat
        rec_cam.distance = cam_distance
        rec_cam.elevation = args.cam_elevation
        rec_cam.azimuth = args.cam_azimuth

        video_dir = pathlib.Path(video_path).parent
        video_dir.mkdir(parents=True, exist_ok=True)
        mp4_writer = imageio.get_writer(
            video_path, fps=tgt_fps,
            quality=args.video_quality, macro_block_size=1,
        )
        print(f"Recording video to [bold]{video_path}[/bold] "
              f"({args.video_width}x{args.video_height}, quality={args.video_quality})")

    print("[bold]Controls:[/bold]")
    print("  '.' / Right arrow: next step (computes next frame automatically)")
    print("  ',' / Left arrow:  previous step")
    print("  Sphere: [blue]blue[/blue]=low weight  [yellow]yellow[/yellow]=medium  [red]red[/red]=high")
    print("  Green capsules = target skeleton, Yellow/Red lines = error vectors")
    print(f"  Camera fixed at world lookat={cam_lookat.tolist()}")

    while viewer.is_running():
        step_idx = current_step[0]
        result = timeline.get_step(step_idx)
        if result is None:
            time.sleep(0.02)
            continue

        frame_idx, snap, scaled_human_data = result

        if step_idx != prev_step[0]:
            data.qpos[:] = snap["qpos"]
            mj.mj_forward(model, data)

            phase, it, err = snap["phase"], snap["iteration"], snap["error"]
            if phase == 0:
                label = "Initial"
            elif it == -1:
                label = f"Phase {phase} start"
            else:
                label = f"Phase {phase}, Iter {it}"
            sorted_errs = sorted(snap["task_errors"].items(), key=lambda x: -x[1])
            top3 = "  ".join(f"{name}={val:.3f}" for name, val in sorted_errs[:3])
            total_str = f"/{timeline.max_known_step()}" if timeline.is_exhausted() else "+..."
            print(f"[Step {step_idx}{total_str}] Frame {frame_idx}/{timeline.total_frames-1}, "
                  f"{label}, Error: {err:.4f}  | {top3}")
            prev_step[0] = step_idx

        weights = get_weights_for_phase(retarget, snap["phase"])

        # HUD text in top-left corner
        phase, it, err = snap["phase"], snap["iteration"], snap["error"]
        if phase == 0:
            phase_str = "Initial"
        elif it == -1:
            phase_str = f"Phase {phase} start"
        else:
            phase_str = f"Phase {phase}, Iter {it}"
        hud_left = f"Frame {frame_idx}/{timeline.total_frames-1}\n{phase_str}"
        hud_right = f"Step {step_idx}"
        viewer.set_texts((mj.mjtFontScale.mjFONTSCALE_200, mj.mjtGridPos.mjGRID_TOPLEFT, hud_left, hud_right))

        # Draw overlays into interactive viewer
        draw_skeleton_and_targets(viewer.user_scn, scaled_human_data, weights,
                                  human_to_robot_frame, model, data, reset=True)
        viewer.sync()

        # Capture frame for video
        if mp4_writer is not None:
            rec_data.qpos[:] = data.qpos[:]
            mj.mj_forward(rec_model, rec_data)
            renderer.update_scene(rec_data, camera=rec_cam)
            draw_skeleton_and_targets(renderer.scene, scaled_human_data, weights,
                                      human_to_robot_frame, rec_model, rec_data, reset=False)
            mp4_writer.append_data(renderer.render())

        if args.auto_play:
            time.sleep(args.play_interval)
            next_idx = current_step[0] + 1
            if timeline.get_step(next_idx) is not None:
                current_step[0] = next_idx
        else:
            time.sleep(0.02)

    if mp4_writer is not None:
        mp4_writer.close()
        print(f"Video saved to [bold]{video_path}[/bold]")

    print("Viewer closed.")


if __name__ == "__main__":
    main()
