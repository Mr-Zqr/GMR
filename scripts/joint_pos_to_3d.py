#!/usr/bin/env python3
"""
Convert G1 joint_pos / joint_vel from (N, 29) to (N, 29, 3).

Each DOF scalar is placed on its rotation axis:  [x=roll, y=pitch, z=yaw]
  roll  joints → [val, 0,   0  ]
  pitch joints → [0,   val, 0  ]
  yaw   joints → [0,   0,   val]

NPZ joint order (sim/lab order), 29 joints:
  [ 0] left_hip_pitch_joint        pitch (y)  → mujoco[ 0]
  [ 1] right_hip_pitch_joint       pitch (y)  → mujoco[ 6]
  [ 2] waist_yaw_joint             yaw   (z)  → mujoco[12]
  [ 3] left_hip_roll_joint         roll  (x)  → mujoco[ 1]
  [ 4] right_hip_roll_joint        roll  (x)  → mujoco[ 7]
  [ 5] waist_roll_joint            roll  (x)  → mujoco[13]
  [ 6] left_hip_yaw_joint          yaw   (z)  → mujoco[ 2]
  [ 7] right_hip_yaw_joint         yaw   (z)  → mujoco[ 8]
  [ 8] waist_pitch_joint           pitch (y)  → mujoco[14]
  [ 9] left_knee_joint             pitch (y)  → mujoco[ 3]
  [10] right_knee_joint            pitch (y)  → mujoco[ 9]
  [11] left_shoulder_pitch_joint   pitch (y)  → mujoco[15]
  [12] right_shoulder_pitch_joint  pitch (y)  → mujoco[22]
  [13] left_ankle_pitch_joint      pitch (y)  → mujoco[ 4]
  [14] right_ankle_pitch_joint     pitch (y)  → mujoco[10]
  [15] left_shoulder_roll_joint    roll  (x)  → mujoco[16]
  [16] right_shoulder_roll_joint   roll  (x)  → mujoco[23]
  [17] left_ankle_roll_joint       roll  (x)  → mujoco[ 5]
  [18] right_ankle_roll_joint      roll  (x)  → mujoco[11]
  [19] left_shoulder_yaw_joint     yaw   (z)  → mujoco[17]
  [20] right_shoulder_yaw_joint    yaw   (z)  → mujoco[24]
  [21] left_elbow_joint            pitch (y)  → mujoco[18]
  [22] right_elbow_joint           pitch (y)  → mujoco[25]
  [23] left_wrist_roll_joint       roll  (x)  → mujoco[19]
  [24] right_wrist_roll_joint      roll  (x)  → mujoco[26]
  [25] left_wrist_pitch_joint      pitch (y)  → mujoco[20]
  [26] right_wrist_pitch_joint     pitch (y)  → mujoco[27]
  [27] left_wrist_yaw_joint        yaw   (z)  → mujoco[21]
  [28] right_wrist_yaw_joint       yaw   (z)  → mujoco[28]
"""

import numpy as np
import argparse
import os
from pathlib import Path

# Axis index per DOF in sim/lab order: 0=x(roll), 1=y(pitch), 2=z(yaw)
_AXIS = [1,1,2, 0,0,0, 2,2,1, 1,1, 1,1, 1,1, 0,0, 0,0, 2,2, 1,1, 0,0, 1,1, 2,2]

# One-hot mask: shape (29, 3).  joint_pos_3d = joint_pos[:,:,None] * AXIS_MASK
AXIS_MASK = np.zeros((29, 3), dtype=np.float32)
for _j, _ax in enumerate(_AXIS):
    AXIS_MASK[_j, _ax] = 1.0


def process_file(input_path, output_path):
    data = np.load(input_path, allow_pickle=True)
    if "joint_pos" not in data or "joint_vel" not in data:
        print(f"  [SKIP] {input_path}: missing joint_pos or joint_vel")
        return False

    joint_pos_3d = data["joint_pos"][:, :, None] * AXIS_MASK   # (N, 29, 3)
    joint_vel_3d = data["joint_vel"][:, :, None] * AXIS_MASK   # (N, 29, 3)

    os.makedirs(Path(output_path).parent, exist_ok=True)
    out = str(output_path).removesuffix(".npz")
    np.savez(out, **{k: data[k] for k in data.files},
             joint_pos_3d=joint_pos_3d, joint_vel_3d=joint_vel_3d)
    print(f"  Saved → {output_path}")
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Convert G1 joint_pos/joint_vel from (N,29) to (N,29,3)"
    )
    parser.add_argument("--input", type=str, required=True,
                        help="Input npz file or directory")
    parser.add_argument("--output", type=str, default=None,
                        help="Output npz file or directory")
    args = parser.parse_args()

    input_path = Path(args.input)

    if input_path.is_file():
        output_path = args.output or str(input_path.parent / (input_path.stem + "_3d.npz"))
        print(f"Processing: {input_path}")
        process_file(str(input_path), output_path)

    elif input_path.is_dir():
        npz_files = sorted(input_path.rglob("*.npz"))
        if not npz_files:
            print(f"No .npz files found in {input_path}")
            return
        output_root = Path(args.output) if args.output else input_path.parent / (input_path.name + "_3d")
        print(f"Found {len(npz_files)} files → {output_root}")
        for npz_file in npz_files:
            process_file(str(npz_file), str(output_root / npz_file.relative_to(input_path)))
        print(f"Done.")

    else:
        print(f"Error: {input_path} is not a valid file or directory")


if __name__ == "__main__":
    main()
