"""
Visualize SMPL-X body shapes and Unitree G1 side-by-side from the same motion.

Current view keeps one SMPL mesh body (SMPL_std), while SMPL_asap_jth and G1
are shown as joint pose (joint spheres + parent-child bone lines).

We render SMPL-X meshes by updating model.mesh_vert + mjr_uploadMesh every
frame. This needs an explicit mjrContext, which mujoco.viewer.launch_passive
doesn't expose, so we run our own GLFW + MjrContext loop.
"""
import argparse
import math
import os
import pathlib
import pickle
import tempfile

import glfw
import mujoco as mj
import numpy as np
import smplx
import torch
from loop_rate_limiters import RateLimiter
from scipy.spatial.transform import Rotation as R
from smplx.joint_names import JOINT_NAMES

from general_motion_retargeting import GeneralMotionRetargeting, ROBOT_XML_DICT
from general_motion_retargeting.utils.smpl import (
    get_smplx_data_offline_fast,
    load_smplx_file,
)


HERE = pathlib.Path(__file__).parent
SMPLX_MODEL_ROOT = (HERE / ".." / "assets" / "body_models").resolve()
DEFAULT_BETAS_DIR = "/home/amax/devel/dataset/NeoBot/smpl_betas"

# (label, pkl_filename or None for standard, rgba, x_offset)
SMPL_BODY_CONFIGS = [
    ("SMPL_std",       None,                             [0.75, 0.75, 0.75, 1.0], -3.0),
    ("SMPL_asap_jth",  "shape_optimized_asap_jth.pkl",   [0.35, 0.85, 0.40, 1.0],  1.0),
]
G1_X_OFFSET = 5.0
SMPL_MESH_LABEL = "SMPL_std"
SMPL_JOINT_LABEL = "SMPL_asap_jth"

SMPL_JOINT_RADIUS = 0.016
SMPL_BONE_WIDTH = 0.006
G1_JOINT_RADIUS = 0.018
G1_BONE_WIDTH = 0.0065

# y-up -> z-up (rotate +90deg around X)
YUP_TO_ZUP_MAT = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]], dtype=np.float32)


def load_betas(betas_dir):
    """Return list of (label, betas_16_tensor[1,16], rgba, x_offset, pkl_scale)."""
    out = []
    for label, pkl_name, rgba, x_off in SMPL_BODY_CONFIGS:
        betas_16 = torch.zeros(1, 16)
        pkl_scale = 1.0
        if pkl_name is not None:
            with open(os.path.join(betas_dir, pkl_name), "rb") as f:
                betas_raw, scale_raw = pickle.load(f)
            n = min(16, betas_raw.shape[1])
            betas_16[0, :n] = betas_raw[0, :n].detach().cpu()
            scale_arr = scale_raw.detach().cpu().numpy() if isinstance(scale_raw, torch.Tensor) \
                else np.asarray(scale_raw)
            pkl_scale = float(scale_arr.reshape(-1)[0])
        out.append((label, betas_16, rgba, x_off, pkl_scale))
    return out


def compute_vertices_batched(body_model, betas_16, global_orient, body_pose, transl, chunk=256):
    """Run SMPL-X forward for a single betas over all T frames. Returns (T,V,3) float32."""
    T = global_orient.shape[0]
    go_t = torch.from_numpy(global_orient).float()
    bp_t = torch.from_numpy(body_pose).float()
    tr_t = torch.from_numpy(transl).float()
    out_chunks = []
    for start in range(0, T, chunk):
        end = min(start + chunk, T)
        k = end - start
        zeros_hand = torch.zeros(k, 45)
        zeros_3 = torch.zeros(k, 3)
        with torch.no_grad():
            out = body_model(
                betas=betas_16,
                global_orient=go_t[start:end],
                body_pose=bp_t[start:end],
                transl=tr_t[start:end],
                left_hand_pose=zeros_hand,
                right_hand_pose=zeros_hand,
                jaw_pose=zeros_3,
                leye_pose=zeros_3,
                reye_pose=zeros_3,
            )
        out_chunks.append(out.vertices.cpu().numpy().astype(np.float32))
    return np.concatenate(out_chunks, axis=0)


def compute_joints_batched(body_model, betas_16, global_orient, body_pose, transl, chunk=256):
    """Run SMPL-X forward for a single betas over all T frames. Returns (T,J,3) float32."""
    T = global_orient.shape[0]
    go_t = torch.from_numpy(global_orient).float()
    bp_t = torch.from_numpy(body_pose).float()
    tr_t = torch.from_numpy(transl).float()
    out_chunks = []
    for start in range(0, T, chunk):
        end = min(start + chunk, T)
        k = end - start
        zeros_hand = torch.zeros(k, 45)
        zeros_3 = torch.zeros(k, 3)
        with torch.no_grad():
            out = body_model(
                betas=betas_16,
                global_orient=go_t[start:end],
                body_pose=bp_t[start:end],
                transl=tr_t[start:end],
                left_hand_pose=zeros_hand,
                right_hand_pose=zeros_hand,
                jaw_pose=zeros_3,
                leye_pose=zeros_3,
                reye_pose=zeros_3,
            )
        out_chunks.append(out.joints.cpu().numpy().astype(np.float32))
    return np.concatenate(out_chunks, axis=0)


def build_combined_xml(g1_xml_path, t_pose_meshes, faces, xml_dir):
    """
    g1_xml_path: absolute path to g1_mocap_29dof.xml
    t_pose_meshes: list of (label, verts_Vx3_in_world, rgba)
    faces: shared faces (F,3) int
    xml_dir: directory to write the temp xml (must be G1 XML dir so includes resolve)
    Returns path to the temp xml file.
    """
    g1_filename = os.path.basename(g1_xml_path)
    face_str = " ".join(str(int(x)) for x in faces.flatten())

    asset_parts = []
    body_parts = []
    for i, (label, verts, rgba) in enumerate(t_pose_meshes):
        vstr = " ".join(f"{v:.6f}" for v in verts.flatten())
        rgba_str = " ".join(f"{c:.3f}" for c in rgba)
        asset_parts.append(
            f'    <mesh name="smpl_{i}" vertex="{vstr}" face="{face_str}"/>'
        )
        body_parts.append(
            f'    <body name="smpl_body_{i}" pos="0 0 0">'
            f'<geom type="mesh" mesh="smpl_{i}" rgba="{rgba_str}" '
            f'contype="0" conaffinity="0" group="1"/></body>'
        )

    xml = (
        '<mujoco>\n'
        f'  <include file="{g1_filename}"/>\n'
        '  <asset>\n' + "\n".join(asset_parts) + "\n  </asset>\n"
        '  <worldbody>\n' + "\n".join(body_parts) + "\n  </worldbody>\n"
        '</mujoco>\n'
    )
    tf = tempfile.NamedTemporaryFile(
        prefix="vis_multi_smpl_", suffix=".xml", dir=xml_dir, delete=False, mode="w"
    )
    tf.write(xml)
    tf.close()
    return tf.name


def _add_joint_sphere(scene, pos, rgba, radius, label=None):
    if scene.ngeom >= scene.maxgeom:
        return None
    g = scene.geoms[scene.ngeom]
    mj.mjv_initGeom(
        g,
        type=mj.mjtGeom.mjGEOM_SPHERE,
        size=[radius, radius, radius],
        pos=np.asarray(pos, np.float64),
        mat=np.eye(3).flatten(),
        rgba=np.asarray(rgba, np.float32),
    )
    if label is not None:
        g.label = label
    scene.ngeom += 1
    return g


def _add_bone(scene, p0, p1, rgba, width):
    if scene.ngeom >= scene.maxgeom:
        return
    p0 = np.asarray(p0, np.float64)
    p1 = np.asarray(p1, np.float64)
    if np.linalg.norm(p1 - p0) < 1e-8:
        return
    g = scene.geoms[scene.ngeom]
    mj.mjv_initGeom(
        g,
        type=mj.mjtGeom.mjGEOM_CAPSULE,
        size=[width, width, width],
        pos=np.zeros(3),
        mat=np.eye(3).flatten(),
        rgba=np.asarray(rgba, np.float32),
    )
    mj.mjv_connector(
        g,
        type=mj.mjtGeom.mjGEOM_CAPSULE,
        width=width,
        from_=p0,
        to=p1,
    )
    scene.ngeom += 1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--smplx_motion_file", required=True,
                        help="Path to the SMPL-X motion file (npz/pkl).")
    parser.add_argument("--betas_dir", default=DEFAULT_BETAS_DIR)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--src_fps", type=float, default=None)
    parser.add_argument("--yup_to_zup", action="store_true", default=True,
                        help="Rotate Y-up motion to Z-up (default: on).")
    parser.add_argument("--no_yup_to_zup", dest="yup_to_zup", action="store_false")
    parser.add_argument("--loop", action="store_true", default=True)
    parser.add_argument("--win_width", type=int, default=1600)
    parser.add_argument("--win_height", type=int, default=900)
    parser.add_argument("--screenshot_frame", type=int, default=None,
                        help="If set, render that frame, save screenshot, and exit.")
    parser.add_argument("--screenshot_path", type=str, default="/tmp/vis_multi_smpl.png")
    parser.add_argument("--smpl_scale", type=float, default=1.0,
                        help="Global multiplier applied on top of each body's "
                             "per-pkl scale. 1.0 = use pkl scale as-is.")
    parser.add_argument("--show_g1_mesh", action="store_true", default=False,
                        help="Show G1 robot mesh in addition to G1 joint pose.")
    parser.add_argument("--overlap_g1_to_smpl_joints", action="store_true", default=False,
                        help="Align G1 joint pose to SMPL_asap_jth by per-frame foot-link translation (fallback: pelvis).")
    args = parser.parse_args()

    # ── 1. Load motion ───────────────────────────────────────────────────────
    smplx_data, body_model, smplx_output, actual_human_height = load_smplx_file(
        args.smplx_motion_file, str(SMPLX_MODEL_ROOT), downsample_fps=args.fps
    )

    # Raw per-frame arrays (already downsampled if src > tgt)
    trans_np = np.asarray(smplx_data["trans"], dtype=np.float32)
    fullpose = np.asarray(smplx_output.full_pose.detach().cpu().numpy(), dtype=np.float32)
    # smplx_output is computed over the whole (downsampled) sequence
    # fullpose shape (T, Jx3) -- extract global_orient (first 3) and body_pose (next 63)
    fullpose = fullpose.reshape(fullpose.shape[0], -1)
    global_orient = fullpose[:, :3].copy()
    body_pose = fullpose[:, 3:66].copy()
    T = trans_np.shape[0]
    assert global_orient.shape[0] == T, f"shape mismatch {global_orient.shape} vs {T}"
    print(f"[info] Loaded {T} frames, human_height={actual_human_height:.3f}")

    # ── 2. Per-frame skeleton dicts for GMR retargeter ──────────────────────
    #    (reuses the existing utility so yup_to_zup handling is consistent)
    frames_for_retarget, aligned_fps = get_smplx_data_offline_fast(
        smplx_data, body_model, smplx_output, tgt_fps=args.fps, src_fps=args.src_fps,
        yup_to_zup=args.yup_to_zup,
    )
    T = min(T, len(frames_for_retarget))
    frames_for_retarget = frames_for_retarget[:T]

    # ── 3. Load betas and compute SMPL vertices across all frames ────────────
    betas_entries = load_betas(args.betas_dir)
    print(f"[info] Computing SMPL-X vertices for {len(betas_entries)} body shapes × {T} frames ...")
    per_body_verts = []  # list of (T, V, 3) float32 in final coord system
    smpl_joint_seq = None  # (T, J, 3), only for SMPL_asap_jth
    smpl_joint_parent = None  # (J,)
    smpl_joint_color = None
    smpl_joint_names = None  # list[str], matched with smpl_joint_seq dim-1
    for label, betas_16, rgba, x_off, _pkl_scale in betas_entries:
        verts = compute_vertices_batched(body_model, betas_16, global_orient, body_pose, trans_np)
        verts = verts[:T]
        if args.yup_to_zup:
            # (T, V, 3) @ R.T to rotate positions
            verts = verts @ YUP_TO_ZUP_MAT.T
        per_body_verts.append(verts)
        print(f"  {label}: verts shape={verts.shape}")
        if label == SMPL_JOINT_LABEL:
            joints = compute_joints_batched(body_model, betas_16, global_orient, body_pose, trans_np)[:T]
            if args.yup_to_zup:
                joints = joints @ YUP_TO_ZUP_MAT.T
            n_joint = min(joints.shape[1], len(body_model.parents))
            smpl_joint_seq = joints[:, :n_joint, :]
            smpl_joint_parent = np.asarray(body_model.parents[:n_joint], dtype=np.int32)
            smpl_joint_color = np.asarray(rgba, dtype=np.float32)
            if len(JOINT_NAMES) >= n_joint:
                smpl_joint_names = list(JOINT_NAMES[:n_joint])
            else:
                smpl_joint_names = [f"smpl_joint_{i}" for i in range(n_joint)]

    # ── 3.5. Apply per-body scale (pkl × CLI) around each frame's root ──────
    # SMPL `transl` is the per-frame root (pelvis) position; scaling around it
    # preserves the motion trajectory and only changes body size.
    root_zup = trans_np.copy()
    if args.yup_to_zup:
        root_zup = root_zup @ YUP_TO_ZUP_MAT.T
    root_zup = root_zup[:T]  # (T, 3)
    for i, (label, _b, _c, _x, pkl_scale) in enumerate(betas_entries):
        eff_scale = pkl_scale * args.smpl_scale
        if abs(eff_scale - 1.0) < 1e-6:
            continue
        per_body_verts[i] = (per_body_verts[i] - root_zup[:, None, :]) * eff_scale \
                            + root_zup[:, None, :]
        print(f"  {label}: applied scale={eff_scale:.4f} "
              f"(pkl={pkl_scale:.4f}, cli={args.smpl_scale:.4f})")
        if betas_entries[i][0] == SMPL_JOINT_LABEL and smpl_joint_seq is not None:
            smpl_joint_seq = (smpl_joint_seq - root_zup[:, None, :]) * eff_scale + root_zup[:, None, :]

    # ── 4. Ground alignment: find z_min across all bodies+frames, shift ──────
    z_min_all = min(v[..., 2].min() for v in per_body_verts)
    print(f"[info] z_min across all SMPL bodies = {z_min_all:.4f}; shifting to z=0")
    for i in range(len(per_body_verts)):
        per_body_verts[i][..., 2] -= z_min_all
    if smpl_joint_seq is not None:
        smpl_joint_seq[..., 2] -= z_min_all

    # ── 5. Apply per-body x-offset ───────────────────────────────────────────
    for i, (label, _b, _c, x_off, _s) in enumerate(betas_entries):
        per_body_verts[i][..., 0] += x_off
        if label == SMPL_JOINT_LABEL and smpl_joint_seq is not None:
            smpl_joint_seq[..., 0] += x_off

    # ── 6. Build combined MuJoCo XML with initial T-pose meshes ─────────────
    g1_xml_path = str(ROBOT_XML_DICT["unitree_g1"].resolve())
    g1_xml_dir = os.path.dirname(g1_xml_path)
    # Use frame-0 vertices as the "reference" mesh
    t_pose_meshes = [
        (betas_entries[i][0], per_body_verts[i][0].copy(), betas_entries[i][2])
        for i in range(len(betas_entries))
        if betas_entries[i][0] == SMPL_MESH_LABEL
    ]
    faces = body_model.faces.astype(np.int32)
    combined_xml_path = build_combined_xml(g1_xml_path, t_pose_meshes, faces, g1_xml_dir)
    try:
        model = mj.MjModel.from_xml_path(combined_xml_path)
    finally:
        os.unlink(combined_xml_path)
    data = mj.MjData(model)
    print(f"[info] Combined model: nbody={model.nbody}, ngeom={model.ngeom}, nmesh={model.nmesh}, nq={model.nq}")

    # ── 7. Cache SMPL mesh_pos / mesh_quat transforms ───────────────────────
    smpl_mesh_indices = [i for i, (label, *_rest) in enumerate(betas_entries) if label == SMPL_MESH_LABEL]
    smpl_mesh_info = []  # list of (src_idx, mesh_id, vertadr, vertnum, mesh_pos, R_mesh)
    for mesh_slot, src_idx in enumerate(smpl_mesh_indices):
        mesh_name = f"smpl_{mesh_slot}"
        mesh_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_MESH, mesh_name)
        assert mesh_id >= 0, f"mesh {mesh_name} not found"
        adr = int(model.mesh_vertadr[mesh_id])
        num = int(model.mesh_vertnum[mesh_id])
        m_pos = np.array(model.mesh_pos[mesh_id], dtype=np.float32)
        m_quat = np.array(model.mesh_quat[mesh_id], dtype=np.float32)  # wxyz
        R_mesh = R.from_quat([m_quat[1], m_quat[2], m_quat[3], m_quat[0]]).as_matrix().astype(np.float32)
        smpl_mesh_info.append((src_idx, mesh_id, adr, num, m_pos, R_mesh))
        print(f"  {mesh_name}: id={mesh_id}, vertadr={adr}, vertnum={num}")

    # G1 qpos slice (it's the freejoint + 29 DoFs at the start of qpos)
    g1_pelvis_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, "pelvis")
    g1_qpos_adr = int(model.jnt_qposadr[model.body_jntadr[g1_pelvis_id]])
    assert g1_qpos_adr == 0, "expected G1 at qpos start"
    g1_qpos_len = 3 + 4 + 29

    # G1 skeleton bodies: all non-world and non-smpl_body_* bodies.
    g1_body_ids = [
        bid for bid in range(1, model.nbody)
        if not mj.mj_id2name(model, mj.mjtObj.mjOBJ_BODY, bid).startswith("smpl_body_")
    ]
    g1_body_set = set(g1_body_ids)
    g1_parent = np.asarray(model.body_parentid, dtype=np.int32)
    g1_joint_color = np.array([1.0, 0.85, 0.2, 1.0], dtype=np.float32)
    g1_body_name = {
        bid: mj.mj_id2name(model, mj.mjtObj.mjOBJ_BODY, bid) for bid in g1_body_ids
    }

    # Overlap anchor: prefer foot links/joints; fallback to pelvis/root.
    def _find_name_idx(name_list, candidates):
        for c in candidates:
            if c in name_list:
                return name_list.index(c)
        return None

    g1_name_to_id = {name: bid for bid, name in g1_body_name.items()}
    g1_foot_ids = [
        g1_name_to_id[n]
        for n in ("left_ankle_roll_link", "right_ankle_roll_link")
        if n in g1_name_to_id
    ]

    smpl_foot_joint_ids = []
    if smpl_joint_names is not None:
        li = _find_name_idx(
            smpl_joint_names, ["left_ankle", "left_foot", "left_toe", "left_toe_base"]
        )
        ri = _find_name_idx(
            smpl_joint_names, ["right_ankle", "right_foot", "right_toe", "right_toe_base"]
        )
        if li is not None:
            smpl_foot_joint_ids.append(li)
        if ri is not None:
            smpl_foot_joint_ids.append(ri)

    # Hide G1 mesh geoms by default; keep only joint pose visualization for G1.
    if not args.show_g1_mesh:
        for gid in range(model.ngeom):
            b_id = int(model.geom_bodyid[gid])
            if b_id in g1_body_set:
                model.geom_rgba[gid, 3] = 0.0

    # ── 8. Init GMR retargeter (uses its own internal MjModel for the robot) ─
    retargeter = GeneralMotionRetargeting(
        src_human="smplx", tgt_robot="unitree_g1",
        actual_human_height=actual_human_height, verbose=False,
    )

    # ── 9. GLFW window + mjrContext ─────────────────────────────────────────
    if not glfw.init():
        raise RuntimeError("glfw.init() failed")
    glfw.window_hint(glfw.VISIBLE, glfw.TRUE)
    window = glfw.create_window(args.win_width, args.win_height,
                                 "Multi-SMPL betas + G1", None, None)
    if not window:
        glfw.terminate()
        raise RuntimeError("glfw.create_window failed")
    glfw.make_context_current(window)
    glfw.swap_interval(1)

    cam = mj.MjvCamera()
    opt = mj.MjvOption()
    scn = mj.MjvScene(model, maxgeom=20000)
    ctx = mj.MjrContext(model, mj.mjtFontScale.mjFONTSCALE_150.value)

    cam.lookat = np.array([1.0, 0.0, 1.0])
    cam.distance = 9.0
    cam.elevation = -10.0
    cam.azimuth = 90.0

    # Mouse interaction: left-drag orbit, right-drag pan, wheel zoom.
    mouse_state = {
        "last_x": 0.0,
        "last_y": 0.0,
        "left_down": False,
        "right_down": False,
    }

    def _mouse_button_cb(_window, button, action, _mods):
        if button == glfw.MOUSE_BUTTON_LEFT:
            mouse_state["left_down"] = (action == glfw.PRESS)
        elif button == glfw.MOUSE_BUTTON_RIGHT:
            mouse_state["right_down"] = (action == glfw.PRESS)

    def _cursor_pos_cb(_window, xpos, ypos):
        dx = xpos - mouse_state["last_x"]
        dy = ypos - mouse_state["last_y"]
        mouse_state["last_x"] = xpos
        mouse_state["last_y"] = ypos

        if mouse_state["left_down"]:
            cam.azimuth += dx * 0.35
            cam.elevation = float(np.clip(cam.elevation - dy * 0.25, -89.0, 89.0))
        elif mouse_state["right_down"]:
            pan_scale = 0.003 * float(cam.distance)
            az = math.radians(float(cam.azimuth))
            right = np.array([math.cos(az), math.sin(az), 0.0], dtype=np.float64)
            up = np.array([0.0, 0.0, 1.0], dtype=np.float64)
            cam.lookat[:] = cam.lookat[:] - right * (dx * pan_scale) + up * (dy * pan_scale)

    def _scroll_cb(_window, _xoff, yoff):
        zoom_scale = math.exp(-0.12 * yoff)
        cam.distance = float(np.clip(cam.distance * zoom_scale, 1.0, 60.0))

    glfw.set_mouse_button_callback(window, _mouse_button_cb)
    glfw.set_cursor_pos_callback(window, _cursor_pos_cb)
    glfw.set_scroll_callback(window, _scroll_cb)

    # ── 10. Main loop ───────────────────────────────────────────────────────
    if args.screenshot_frame is not None:
        print(f"[info] Screenshot mode: frame {args.screenshot_frame} -> {args.screenshot_path}")
    else:
        print(f"[info] Entering render loop at {args.fps} fps. Close window to exit.")
    rate = RateLimiter(frequency=args.fps, warn=False)

    frame_idx = 0
    while not glfw.window_should_close(window):
        if args.screenshot_frame is not None:
            t = args.screenshot_frame % T
        else:
            t = frame_idx % T

        # 10a. Update SMPL mesh vertices (to local mesh frame) and upload
        for src_idx, mesh_id, adr, num, m_pos, R_mesh in smpl_mesh_info:
            v_world = per_body_verts[src_idx][t]  # (V, 3), already x-offset + z-shifted
            # local = R_mesh^T * (world - mesh_pos)
            v_local = (v_world - m_pos) @ R_mesh  # == (R_mesh^T @ (v-p).T).T
            model.mesh_vert[adr:adr + num] = v_local
            mj.mjr_uploadMesh(model, ctx, mesh_id)

        # 10b. Retarget this frame for G1
        qpos, _ = retargeter.retarget(frames_for_retarget[t])
        qpos = qpos.copy()
        qpos[0] += G1_X_OFFSET
        data.qpos[g1_qpos_adr:g1_qpos_adr + g1_qpos_len] = qpos

        mj.mj_forward(model, data)

        # 10c. Update scene + add labels
        mj.mjv_updateScene(
            model, data, opt, None, cam, mj.mjtCatBit.mjCAT_ALL.value, scn,
        )
        # 10c-1. SMPL_asap_jth joint pose (spheres + bones)
        g1_draw_delta = np.zeros(3, dtype=np.float64)
        show_joint_labels = args.overlap_g1_to_smpl_joints
        if smpl_joint_seq is not None:
            jpos = smpl_joint_seq[t]
            for j in range(jpos.shape[0]):
                label = None
                if show_joint_labels and smpl_joint_names is not None:
                    label = f"S:{smpl_joint_names[j]}"
                _add_joint_sphere(scn, jpos[j], smpl_joint_color, SMPL_JOINT_RADIUS, label=label)
            for j in range(1, jpos.shape[0]):
                p = int(smpl_joint_parent[j])
                if p < 0:
                    continue
                _add_bone(scn, jpos[p], jpos[j], smpl_joint_color, SMPL_BONE_WIDTH)
            if args.overlap_g1_to_smpl_joints:
                if len(smpl_foot_joint_ids) > 0 and len(g1_foot_ids) > 0:
                    smpl_anchor = np.mean(jpos[smpl_foot_joint_ids], axis=0).astype(np.float64)
                    g1_anchor = np.mean(data.xpos[g1_foot_ids], axis=0).astype(np.float64)
                else:
                    smpl_anchor = jpos[0].astype(np.float64)
                    g1_anchor = data.xpos[g1_pelvis_id].astype(np.float64)
                g1_draw_delta = smpl_anchor - g1_anchor

        # 10c-2. G1 joint pose (spheres + bones)
        for bid in g1_body_ids:
            g_label = f"G:{g1_body_name[bid]}" if show_joint_labels else None
            _add_joint_sphere(
                scn,
                data.xpos[bid] + g1_draw_delta,
                g1_joint_color,
                G1_JOINT_RADIUS,
                label=g_label,
            )
        for bid in g1_body_ids:
            p = int(g1_parent[bid])
            if p <= 0 or p not in g1_body_set:
                continue
            _add_bone(
                scn,
                data.xpos[p] + g1_draw_delta,
                data.xpos[bid] + g1_draw_delta,
                g1_joint_color,
                G1_BONE_WIDTH,
            )

        # 10c-3. Labels
        for label, _b, rgba, x_off, _s in betas_entries:
            if scn.ngeom >= scn.maxgeom:
                break
            if label not in {SMPL_MESH_LABEL, SMPL_JOINT_LABEL}:
                continue
            label_pos = np.array([x_off, 0.0, 2.3], dtype=np.float64)
            g = scn.geoms[scn.ngeom]
            mj.mjv_initGeom(
                g, type=mj.mjtGeom.mjGEOM_SPHERE, size=[0.04, 0.04, 0.04],
                pos=label_pos, mat=np.eye(3).flatten(),
                rgba=np.array([rgba[0], rgba[1], rgba[2], 1.0], dtype=np.float32),
            )
            g.label = label
            scn.ngeom += 1
        # G1 label
        if scn.ngeom < scn.maxgeom:
            g = scn.geoms[scn.ngeom]
            mj.mjv_initGeom(
                g, type=mj.mjtGeom.mjGEOM_SPHERE, size=[0.04, 0.04, 0.04],
                pos=np.array([G1_X_OFFSET, 0.0, 2.3], dtype=np.float64),
                mat=np.eye(3).flatten(),
                rgba=np.array([1.0, 0.85, 0.2, 1.0], dtype=np.float32),
            )
            if args.overlap_g1_to_smpl_joints:
                g.pos[:] = g.pos + g1_draw_delta
            g.label = "G1"
            scn.ngeom += 1

        # 10d. Render
        fb_w, fb_h = glfw.get_framebuffer_size(window)
        viewport = mj.MjrRect(0, 0, fb_w, fb_h)
        mj.mjr_render(viewport, scn, ctx)

        glfw.swap_buffers(window)
        glfw.poll_events()

        if args.screenshot_frame is not None:
            # Render a couple frames to let GL settle, then grab the framebuffer
            if frame_idx >= 2:
                import imageio.v2 as imageio
                rgb = np.zeros((fb_h, fb_w, 3), dtype=np.uint8)
                depth = np.zeros((fb_h, fb_w, 1), dtype=np.float32)
                mj.mjr_readPixels(rgb=rgb, depth=None, viewport=viewport, con=ctx)
                imageio.imwrite(args.screenshot_path, np.flipud(rgb))
                print(f"[info] Saved screenshot to {args.screenshot_path}")
                break

        frame_idx += 1
        if not args.loop and frame_idx >= T:
            break
        rate.sleep()

    glfw.terminate()
    print("[info] exited cleanly.")


if __name__ == "__main__":
    main()
