"""Visualize the IK optimization process across all frames of motion retargeting.

Flattens all IK iterations across all frames into a single timeline.
Step through with '.' (forward) and ',' (backward) like a video.

Each frame shows: initial pose -> Phase 1 iterations -> Phase 2 iterations -> next frame...

Features:
- Target skeleton (green stick figure)
- Sphere size/color per joint based on IK weights
- Error lines from robot end-effectors to targets
- Lazy frame computation (next frame computed on demand)

Usage:
    python scripts/vis_ik_optimization.py \
        --smplx_file <path.npz> \
        --robot unitree_g1 \
        [--yup_to_zup] \
        [--auto_play] \
        [--play_interval 0.5]
"""
import argparse
import pathlib
import time

import numpy as np
import mujoco as mj
import mujoco.viewer as mjv
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


def draw_frame_styled(pos, mat, viewer, size, width=0.005, alpha=1.0, label=None,
                      colors=None):
    """Draw orientation frame arrows. colors: list of 3 RGB triples (default: RGB)."""
    if colors is None:
        colors = COLOR_RGB
    rgba_list = [[c[0], c[1], c[2], alpha] for c in colors]
    for i in range(3):
        if viewer.user_scn.ngeom >= viewer.user_scn.maxgeom - 1:
            break
        geom = viewer.user_scn.geoms[viewer.user_scn.ngeom]
        mj.mjv_connector(
            geom, type=mj.mjtGeom.mjGEOM_ARROW, width=width,
            from_=pos, to=pos + size * mat[:, i],
        )
        geom.rgba[:] = rgba_list[i]
        if i == 0 and label is not None:
            geom.label = label
        viewer.user_scn.ngeom += 1


def draw_skeleton_and_targets(viewer, scaled_human_data, weights, human_to_robot_frame, model, data):
    viewer.user_scn.ngeom = 0

    # 1) Target skeleton edges (green capsules)
    for parent, child in HUMAN_SKELETON_EDGES:
        if parent not in scaled_human_data or child not in scaled_human_data:
            continue
        if viewer.user_scn.ngeom >= viewer.user_scn.maxgeom - 1:
            break
        pos_from = np.asarray(scaled_human_data[parent][0], dtype=np.float64)
        pos_to = np.asarray(scaled_human_data[child][0], dtype=np.float64)
        geom = viewer.user_scn.geoms[viewer.user_scn.ngeom]
        mj.mjv_connector(geom, type=mj.mjtGeom.mjGEOM_CAPSULE, width=0.008,
                          from_=pos_from, to=pos_to)
        geom.rgba[:] = [0.2, 0.8, 0.2, 0.5]
        viewer.user_scn.ngeom += 1

    # 2) Target spheres + orientation frames (bright, thick arrows)
    for human_body_name, (pos, rot) in scaled_human_data.items():
        if viewer.user_scn.ngeom >= viewer.user_scn.maxgeom - 10:
            break
        pos_w, rot_w = weights.get(human_body_name, (0, 0))
        color = weight_to_color(pos_w, rot_w)
        radius = weight_to_radius(pos_w, rot_w)

        # Sphere
        geom = viewer.user_scn.geoms[viewer.user_scn.ngeom]
        mj.mjv_initGeom(geom, type=mj.mjtGeom.mjGEOM_SPHERE,
                         size=[radius] * 3,
                         pos=np.asarray(pos, dtype=np.float64),
                         mat=np.eye(3).flatten(), rgba=color)
        # no label — too cluttered
        viewer.user_scn.ngeom += 1

        # Target orientation frame: bright RGB, thick arrows, large
        frame_scale = 0.20 + 0.10 * min(1.0, (pos_w + rot_w) / 100.0)
        target_mat = R.from_quat(rot, scalar_first=True).as_matrix()
        draw_frame_styled(np.asarray(pos, dtype=np.float64), target_mat,
                          viewer, frame_scale, width=0.010, alpha=1.0)

    # 3) Robot joint orientation frames (dim, thin arrows) + error lines
    for human_name, robot_frame_name in human_to_robot_frame.items():
        if human_name not in scaled_human_data:
            continue
        if viewer.user_scn.ngeom >= viewer.user_scn.maxgeom - 8:
            break
        robot_body_id = model.body(robot_frame_name).id
        robot_pos = data.xpos[robot_body_id].copy()
        robot_mat = data.xmat[robot_body_id].reshape(3, 3).copy()
        target_pos = np.asarray(scaled_human_data[human_name][0], dtype=np.float64)

        pos_w, rot_w = weights.get(human_name, (0, 0))

        # Robot joint orientation frame: CMY colors to distinguish from target RGB
        robot_frame_scale = 0.18 + 0.10 * min(1.0, (pos_w + rot_w) / 100.0)
        draw_frame_styled(robot_pos, robot_mat, viewer, robot_frame_scale,
                          width=0.008, alpha=0.8, colors=COLOR_CMY)

        # Error line connecting robot joint to target
        line_width = 0.001 + 0.004 * min(1.0, (pos_w + rot_w) / 100.0)
        geom = viewer.user_scn.geoms[viewer.user_scn.ngeom]
        mj.mjv_connector(geom, type=mj.mjtGeom.mjGEOM_LINE, width=line_width,
                          from_=robot_pos, to=target_pos)
        dist = np.linalg.norm(robot_pos - target_pos)
        err_t = min(1.0, dist / 0.3)
        geom.rgba[:] = [1.0, 1.0 - 0.7 * err_t, 0.0, 0.7]
        viewer.user_scn.ngeom += 1


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
    args = parser.parse_args()

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

    # Setup MuJoCo viewer
    model = mj.MjModel.from_xml_path(str(ROBOT_XML_DICT[args.robot]))
    data = mj.MjData(model)
    robot_base_id = model.body(ROBOT_BASE_DICT[args.robot]).id
    cam_distance = VIEWER_CAM_DISTANCE_DICT[args.robot]

    current_step = [0]
    prev_step = [-1]

    def key_callback(keycode):
        if keycode == 262 or keycode == ord('.'):  # right arrow or '.'
            # Try to advance; compute next frame if needed
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

    print("[bold]Controls:[/bold]")
    print("  '.' / Right arrow: next step (computes next frame automatically)")
    print("  ',' / Left arrow:  previous step")
    print("  Sphere: [blue]blue[/blue]=low weight  [yellow]yellow[/yellow]=medium  [red]red[/red]=high")
    print("  Green capsules = target skeleton, Yellow/Red lines = error vectors")

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
        sorted_errs = sorted(snap["task_errors"].items(), key=lambda x: -x[1])
        top3_str = "\n".join(f"  {name}: {val:.3f}" for name, val in sorted_errs[:5])
        hud_left = f"Frame {frame_idx}/{timeline.total_frames-1}\n{phase_str}"
        hud_right = f"Step {step_idx}"
        viewer.set_texts((mj.mjtFontScale.mjFONTSCALE_200, mj.mjtGridPos.mjGRID_TOPLEFT, hud_left, hud_right))

        draw_skeleton_and_targets(viewer, scaled_human_data, weights,
                                  human_to_robot_frame, model, data)

        viewer.cam.lookat = data.xpos[robot_base_id].copy()
        viewer.cam.distance = cam_distance
        viewer.cam.elevation = -10
        viewer.cam.azimuth = 90

        viewer.sync()

        if args.auto_play:
            time.sleep(args.play_interval)
            next_idx = current_step[0] + 1
            if timeline.get_step(next_idx) is not None:
                current_step[0] = next_idx
        else:
            time.sleep(0.02)

    print("Viewer closed.")


if __name__ == "__main__":
    main()
