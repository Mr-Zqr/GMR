"""Visualize representative frames from a motion cluster.

Given a cluster ID, randomly picks N files from that cluster, finds the most
similar frame across them (closest to the global mean joint pose), and renders
each as a PNG using MuJoCo offscreen rendering.
"""
import argparse
import os
import random
import tempfile

import joblib
import imageio
import mujoco as mj
import numpy as np

from general_motion_retargeting import ROBOT_XML_DICT, ROBOT_BASE_DICT, VIEWER_CAM_DISTANCE_DICT

# G1 bmimic → mujoco DOF reorder (same as vis_robot_motion_bmimic.py)
G1_JOINT_MAPPING = [
    0, 6, 12,
    1, 7, 13,
    2, 8, 14,
    3, 9, 15, 22,
    4, 10, 16, 23,
    5, 11, 17, 24,
    18, 25, 19, 26, 20, 27, 21, 28,
]

DATA_ROOT = "/home/amax/devel/dataset/NeoBot/2_gmr_retarget_filtered-0122-0035"
FILE_LIST_PATH = os.path.join(DATA_ROOT, "file_path_list_cluster500.txt")
LABELS_PATH = os.path.join(DATA_ROOT, "labels_500.pkl")


def load_cluster_data():
    with open(FILE_LIST_PATH, "r") as f:
        file_paths = [line.strip() for line in f if line.strip()]
    labels = joblib.load(LABELS_PATH)
    return file_paths, labels


def load_motion(rel_path):
    full_path = os.path.join(DATA_ROOT, rel_path + ".npz")
    data = np.load(full_path)
    return data


def find_similar_frame_indices(motion_list):
    """Find the frame in each motion closest to the global mean joint pose."""
    # Compute global mean joint_pos across all frames of all motions
    all_joints = np.concatenate([m["joint_pos"] for m in motion_list], axis=0)
    global_mean = all_joints.mean(axis=0)  # (29,)

    frame_indices = []
    for m in motion_list:
        joint_pos = m["joint_pos"]  # (N, 29)
        dists = np.linalg.norm(joint_pos - global_mean, axis=1)
        frame_indices.append(int(np.argmin(dists)))
    return frame_indices


def load_model_classic_scene(xml_path):
    """Load G1 model with classic MuJoCo blue reflective floor.

    The G1 XML already has 'texplane' (blue checker) and 'MatPlane' (reflectance=0.5)
    but the floor geom uses the white 'groundplane' material. We swap it via XML patching.
    """
    with open(xml_path) as f:
        xml_text = f.read()
    # Switch floor to the blue checker material already defined in the XML
    xml_text = xml_text.replace('material="groundplane"', 'material="MatPlane"')
    # Fix MatPlane texrepeat: original "1 1" makes tiles huge, classic look uses 5 5
    xml_text = xml_text.replace(
        'texrepeat="1 1" texture="texplane"',
        'texrepeat="5 5" texture="texplane"',
    )
    # Lighten the floor checker colors
    xml_text = xml_text.replace(
        'rgb1=".2 .3 .4" rgb2=".1 .15 .2"',
        'rgb1=".6 .72 .8" rgb2=".45 .57 .66"',
    )
    # Boost ambient for better overall illumination
    xml_text = xml_text.replace('ambient="0.1 0.1 0.1"', 'ambient="0.3 0.3 0.3"')
    xml_dir = os.path.dirname(xml_path)
    with tempfile.NamedTemporaryFile(mode="w", suffix=".xml", dir=xml_dir, delete=False) as f:
        f.write(xml_text)
        tmp_path = f.name
    try:
        model = mj.MjModel.from_xml_path(tmp_path)
    finally:
        os.unlink(tmp_path)
    return model


def render_frame(model, data, renderer, root_pos, root_rot_wxyz, dof_pos_mujoco):
    """Render a single frame and return RGB image."""
    data.qpos[:3] = root_pos
    data.qpos[3:7] = root_rot_wxyz
    data.qpos[7:] = dof_pos_mujoco
    mj.mj_forward(model, data)

    camera = mj.MjvCamera()
    base_id = model.body(ROBOT_BASE_DICT["unitree_g1"]).id
    camera.lookat = np.array([data.xpos[base_id][0], data.xpos[base_id][1], 0.6])
    camera.distance = VIEWER_CAM_DISTANCE_DICT["unitree_g1"]
    camera.elevation = 8
    camera.azimuth = 90

    renderer.update_scene(data, camera=camera)
    return renderer.render().copy()


def main():
    parser = argparse.ArgumentParser(description="Render representative frames from a motion cluster")
    parser.add_argument("--cluster_id", type=int, required=True, help="Cluster index (0-499)")
    parser.add_argument("--save_dir", type=str, required=True, help="Output directory for PNGs")
    parser.add_argument("--num_files", type=int, default=20, help="Number of files to sample")
    parser.add_argument("--width", type=int, default=640, help="Render width")
    parser.add_argument("--height", type=int, default=480, help="Render height")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility")
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)

    # Load cluster data
    file_paths, labels = load_cluster_data()
    cluster_indices = np.where(labels == args.cluster_id)[0]
    if len(cluster_indices) == 0:
        print(f"Cluster {args.cluster_id} has no files.")
        return

    # Filter to existing files first, then sample
    all_paths = [file_paths[i] for i in cluster_indices]
    existing_paths = [p for p in all_paths if os.path.exists(os.path.join(DATA_ROOT, p + ".npz"))]
    print(f"Cluster {args.cluster_id}: {len(cluster_indices)} total files, {len(existing_paths)} found on disk")
    n = min(args.num_files, len(existing_paths))
    selected_paths = random.sample(existing_paths, n)

    # Load motions
    motions = [load_motion(p) for p in selected_paths]
    print(f"Loaded {len(motions)} motion files")

    # Find similar frames
    frame_indices = find_similar_frame_indices(motions)

    # Set up MuJoCo offscreen rendering with classic blue scene
    xml_path = str(ROBOT_XML_DICT["unitree_g1"])
    model = load_model_classic_scene(xml_path)
    data = mj.MjData(model)
    for i in range(model.ngeom):
        if model.geom_type[i] != mj.mjtGeom.mjGEOM_PLANE:
            model.geom_rgba[i] = [0.55, 0.55, 0.55, 1.0]
    renderer = mj.Renderer(model, height=args.height, width=args.width)

    # Render and save
    os.makedirs(args.save_dir, exist_ok=True)
    for i, (motion, fidx, rel_path) in enumerate(zip(motions, frame_indices, selected_paths)):
        root_pos = motion["body_pos_w"][fidx, 0, :]
        root_rot = motion["body_quat_w"][fidx, 0, :]  # wxyz
        joint_pos_bmimic = motion["joint_pos"][fidx, :]

        # Inverse mapping: bmimic order → mujoco DOF order
        dof_pos = np.zeros(29, dtype=np.float32)
        dof_pos[G1_JOINT_MAPPING] = joint_pos_bmimic

        img = render_frame(model, data, renderer, root_pos, root_rot, dof_pos)
        # Use the relative path as filename, replacing '/' with '-'
        filename = rel_path.replace("/", "-") + ".png"
        out_path = os.path.join(args.save_dir, filename)
        imageio.imwrite(out_path, img)
        print(f"  [{i}] frame {fidx}/{motion['joint_pos'].shape[0]} from {rel_path} → {out_path}")

    print(f"Done. {n} PNGs saved to {args.save_dir}")


if __name__ == "__main__":
    main()
