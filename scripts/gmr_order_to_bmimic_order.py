
import argparse
import numpy as np
import pathlib
import torch
from general_motion_retargeting import ROBOT_XML_DICT
from scripts.smplx_to_robot_gmr import build_bmimic_data

from general_motion_retargeting.kinematics_model import KinematicsModel
import joblib
def main():
    parser = argparse.ArgumentParser(description="Convert GMR .pkl motion files to BMimic-compatible .npz files")
    parser.add_argument("--input_folder", type=str, required=True, help="Folder containing GMR .pkl files")
    parser.add_argument("--robot", default="unitree_g1",
                        help="Robot name (default: unitree_g1)")
    parser.add_argument("--fps", type=float, default=30.0,
                        help="Frame rate of source pkl files (default: 50)")
    parser.add_argument("--output_folder", type=str, required=True, help="Folder to save BMimic .npz files")
    args = parser.parse_args()  

    input_dir = pathlib.Path(args.input_folder)
    output_dir = pathlib.Path(args.output_folder)

    pkl_files = sorted(input_dir.rglob("*.pkl"))
    if not pkl_files:
        print(f"No .pkl files found in {input_dir}")
        return
    print(f"Found {len(pkl_files)} .pkl files in {input_dir}")

    
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    km = KinematicsModel(str(ROBOT_XML_DICT[args.robot]), device=device)
    print(f"KinematicsModel loaded for {args.robot} on {device}")

    ok, failed = 0, 0
    for pkl_path in pkl_files:
        rel      = pkl_path.relative_to(input_dir)
        out_path = output_dir / rel.with_suffix(".npz")
        out_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            d             = joblib.load(pkl_path)
            root_pos      = np.asarray(d["root_pos"],    dtype=np.float32)  # (N, 3)
            root_rot_wxyz = np.asarray(d["root_rot"][:, [3, 0, 1, 2]], dtype=np.float32)  # (N, 4) wxyz
            dof_pos       = np.asarray(d["dof_pos"],           dtype=np.float32)  # (N, 29)

            motion_data = build_bmimic_data(root_pos, root_rot_wxyz, dof_pos, args.fps, km, args.robot)
            np.savez(str(out_path), **motion_data)
            print(f"  [ok] {rel}  frames={root_pos.shape[0]}")
            ok += 1
        except Exception as e:
            print(f"  [FAIL] {rel}: {e}")
            failed += 1
    

    print(f"\nDone: {ok} converted, {failed} failed → {output_dir}")


if __name__ == "__main__":
    main()