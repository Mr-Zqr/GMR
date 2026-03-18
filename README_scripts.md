# GMR Scripts — Quick Reference

## main_humanoid.py

Visualize or export SMPL-X human motion (aitviewer).

**Single file — interactive view:**
```bash
python scripts/main_humanoid.py --smplx_file path/to/motion.npz
```

**Single file — export video:**
```bash
python scripts/main_humanoid.py \
    --smplx_file path/to/motion.npz \
    --record_video --video_path output.mp4
```

**Batch folder — export all .npz as videos (mirrors folder structure):**
```bash
python scripts/main_humanoid.py \
    --smplx_dir  /data/1_filtered_smplx \
    --video_dir  /data/smplx_videos \
    --zup
```

> `--zup`: input uses Z-up convention (e.g. AMASS/NeoBot); auto-rotates to Y-up for aitviewer.

---

## find_bad_motions.py

Filter retargeted G1 bmimic NPZ files by quality. Detects three issues:
- **floating_feet** — average foot clearance above ground exceeds threshold
- **joint_jump** — single-frame joint discontinuity (IK solver flip)
- **self_intersection** — MuJoCo collision between robot body meshes

**Basic usage:**
```bash
python scripts/find_bad_motions.py \
    --input_dir  /data/2_gmr_retarget \
    --output_pkl valid_motions.pkl \
    --output_yaml bad_motions.yaml \
    --num_workers 16
```

**Also record videos of bad motions (organised by issue type):**
```bash
python scripts/find_bad_motions.py \
    --input_dir    /data/2_gmr_retarget \
    --output_pkl   valid_motions.pkl \
    --output_yaml  bad_motions.yaml \
    --record_videos \
    --video_dir    bad_motion_videos \
    --smplx_dir    /data/1_filtered_smplx \
    --num_workers  16
```

Output video structure:
```
bad_motion_videos/
  floating_feet/
    robot/<rel_path>.mp4
    human/<rel_path>.mp4
  joint_jump/
    robot/<rel_path>.mp4
    human/<rel_path>.mp4
  self_intersection/
    robot/<rel_path>.mp4
    human/<rel_path>.mp4
```

A motion with multiple issues appears in all relevant issue folders.
`--smplx_dir` defaults to the sibling `1_filtered_smplx/` directory of `--input_dir`.

Key thresholds (all have defaults):

| Flag | Default | Meaning |
|------|---------|---------|
| `--float_threshold` | 0.10 m | mean foot clearance to flag floating |
| `--jump_threshold` | 0.5 rad | per-frame joint change to flag as jump |
| `--cross_ratio` | 0.05 | fraction of frames with self-intersection to flag |
| `--skip_frames` | 2 | check every N-th frame for collision |

---

## vis_robot_motion.py

Visualize retargeted robot motion (GMR standard pkl/npz or bmimic-style).
Auto-detects format from file keys; prints all key shapes on load.

**Interactive single file:**
```bash
python scripts/vis_robot_motion.py \
    --robot_motion_path output.pkl \
    --robot unitree_g1
```

**Export single video:**
```bash
python scripts/vis_robot_motion.py \
    --robot_motion_path output.pkl \
    --robot unitree_g1 \
    --record_video --video_path output.mp4
```

**Batch directory → videos:**
```bash
python scripts/vis_robot_motion.py \
    --robot_motion_dir /data/2_gmr_retarget \
    --robot unitree_g1 \
    --video_dir /data/robot_videos
```

Auto-detected formats:

| Format | Keys |
|--------|------|
| GMR standard | `root_pos`, `root_rot` (xyzw), `dof_pos` |
| G1 debug | `g1_trans`, `g1_root_rot`, `g1_dof` |
| G1 gt | `gt_trans`, `gt_g1_root_ori_quat`, `gt_g1_dof` |
| G1 joints | `g1_joints`, `g1_root_ori`, `g1_dof` |

For unknown formats use `--key_root_pos / --key_root_rot / --key_dof`.
Use `--joint_mapping` for bmimic-order DOFs (G1 reordering).

---

## vis_robot_motion_bmimic.py

Visualize bmimic-format NPZ files (G1 robot, `body_pos_w` / `body_quat_w` / `joint_pos`).

**Interactive single file:**
```bash
python scripts/vis_robot_motion_bmimic.py \
    --robot_motion_path /data/2_gmr_retarget/walk/01.npz
```

**Batch directory → videos:**
```bash
python scripts/vis_robot_motion_bmimic.py \
    --input_dir /data/2_gmr_retarget \
    --video_dir /data/robot_videos
```

Output mirrors the input directory structure (`<rel_path>.npz` → `<rel_path>.mp4`).
