
import argparse
import numpy as np
import pathlib
import torch
from scipy.spatial.transform import Rotation, Slerp
from general_motion_retargeting import ROBOT_XML_DICT
from scripts.smplx_to_robot_gmr import build_bmimic_data

from general_motion_retargeting.kinematics_model import KinematicsModel


def resample_motion(root_pos, root_rot_wxyz, dof_pos, src_fps, tgt_fps=50.0):
    """Resample motion arrays from src_fps to tgt_fps using linear/SLERP interpolation."""
    if abs(src_fps - tgt_fps) < 0.5:
        return root_pos, root_rot_wxyz, dof_pos
    N = root_pos.shape[0]
    t_src = np.arange(N) / src_fps
    duration = t_src[-1]
    M = int(round(duration * tgt_fps)) + 1
    t_tgt = np.clip(np.arange(M) / tgt_fps, 0.0, duration)

    root_pos_new = np.stack(
        [np.interp(t_tgt, t_src, root_pos[:, i]) for i in range(3)], axis=1
    ).astype(np.float32)
    dof_pos_new = np.stack(
        [np.interp(t_tgt, t_src, dof_pos[:, i]) for i in range(dof_pos.shape[1])], axis=1
    ).astype(np.float32)

    root_rot_xyzw = root_rot_wxyz[:, [1, 2, 3, 0]]
    slerp = Slerp(t_src, Rotation.from_quat(root_rot_xyzw))
    root_rot_wxyz_new = slerp(t_tgt).as_quat()[:, [3, 0, 1, 2]].astype(np.float32)

    return root_pos_new, root_rot_wxyz_new, dof_pos_new
def main():
    parser = argparse.ArgumentParser(description="Convert GMR .pkl motion files to BMimic-compatible .npz files")
    parser.add_argument("--input_folder", type=str, required=True, help="Folder containing GMR .pkl files")
    parser.add_argument("--robot", default="unitree_g1",
                        help="Robot name (default: unitree_g1)")
    parser.add_argument("--fps", type=float, default=30.0,
                        help="Frame rate of source pkl files (default: 30)")
    parser.add_argument("--output_folder", type=str, required=True, help="Folder to save BMimic .npz files")
    args = parser.parse_args()  

    input_dir = pathlib.Path(args.input_folder)
    output_dir = pathlib.Path(args.output_folder)

    pkl_files = sorted(input_dir.rglob("*.npy"))
    if not pkl_files:
        print(f"No .npy files found in {input_dir}")
        return
    print(f"Found {len(pkl_files)} .npy files in {input_dir}")

    
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    km = KinematicsModel(str(ROBOT_XML_DICT[args.robot]), device=device)
    print(f"KinematicsModel loaded for {args.robot} on {device}")

    ok, failed = 0, 0
    for pkl_path in pkl_files:
        rel      = pkl_path.relative_to(input_dir)
        out_path = output_dir / rel.with_suffix(".npz")
        out_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            d             = np.load(pkl_path, allow_pickle=True).item()
            root_pos      = np.asarray(d["root_trans"],             dtype=np.float32)  # (N, 3)
            root_rot_wxyz = np.asarray(d["root_ori"][:, [3, 0, 1, 2]], dtype=np.float32)  # xyzw -> wxyz
            dof_pos_orig       = np.asarray(d["dof_pos"],           dtype=np.float32)  # (N, 23)
            dof_pos = np.zeros((dof_pos_orig.shape[0], 29), dtype=np.float32)
            joint_mapping = list(range(0, 20)) + list(range(23, 26))
            dof_pos[:, joint_mapping] = dof_pos_orig
            # dof_pos = dof_pos_orig
            # fps = float(d["fps"]) if "fps" in d else args.fps
            fps = args.fps

            tgt_fps = 50.0
            root_pos, root_rot_wxyz, dof_pos = resample_motion(root_pos, root_rot_wxyz, dof_pos, fps, tgt_fps)
            motion_data = build_bmimic_data(root_pos, root_rot_wxyz, dof_pos, tgt_fps, km, args.robot)
            np.savez(str(out_path), **motion_data)
            print(f"  [ok] {rel}  frames={root_pos.shape[0]}")
            ok += 1
        except Exception as e:
            print(f"  [FAIL] {rel}: {e}")
            failed += 1
    

    print(f"\nDone: {ok} converted, {failed} failed → {output_dir}")


if __name__ == "__main__":
    main()