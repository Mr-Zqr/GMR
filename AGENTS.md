# Repository Guidelines

## Project Structure & Module Organization
- `general_motion_retargeting/`: core library code (retargeting logic, kinematics, utilities, robot configs).
- `scripts/`: CLI workflows for conversion, retargeting, visualization, and dataset filtering.
- `assets/`: robot assets (`.xml`, `.urdf`, meshes) and body model resources (for example `assets/body_models/smplx`).
- `motion_analysis/`: analysis utilities and plotting scripts.
- `third_party/poselib/`: vendored dependency with its own tests; avoid broad refactors here.
- `test/`: local sample data files used for quick checks, not a full automated test suite.

## Build, Test, and Development Commands
- `conda create -n gmr python=3.10 -y && conda activate gmr`: create the recommended environment.
- `pip install -e .`: install in editable mode from `setup.py`.
- `python scripts/smplx_to_robot_gmr.py --smplx_file <in.npz> --robot unitree_g1 --save_path <out.npz> --headless`: run a retargeting pass.
- `python scripts/vis_robot_motion.py --robot_motion_path <out.npz> --robot unitree_g1`: visualize a result.
- `python scripts/find_bad_motions.py --input_dir <retarget_dir> --output_pkl valid_motions.pkl --output_yaml bad_motions.yaml`: quality filter batch outputs.

## Coding Style & Naming Conventions
- Python style: 4-space indentation, snake_case for functions/variables/files, PascalCase for classes.
- Keep scripts argument-driven (`argparse`) and prefer explicit flag names over implicit behavior.
- No project-wide formatter config is committed; follow existing style in nearby files and keep imports/grouping consistent.
- Keep generated artifacts (`*.mp4`, large `*.pkl`/`*.npz`) out of commits unless intentionally versioned.

## Testing Guidelines
- There is no single `pytest` entry point for the whole repo.
- Use targeted validation:
  - smoke test one motion through `smplx_to_robot_gmr.py`;
  - verify visualization with `vis_robot_motion.py`;
  - optionally run vendored checks: `python -m pytest third_party/poselib/core/tests`.
- For new features, include a reproducible command and expected output path in the PR description.

## Commit & Pull Request Guidelines
- Recent commits are short and imperative (examples: `update video resolution`, `add motion analysis`, `fix t1_29dof ...`).
- Prefer `<verb> <scope>` style, e.g., `add g1 wrist mapping fix`.
- PRs should include:
  - what changed and why;
  - key commands run for validation;
  - linked issue (if any);
  - screenshots/videos when behavior or visualization changes.
