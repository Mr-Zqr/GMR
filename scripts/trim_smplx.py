"""Trim SMPL-X motion files by start/end time in seconds, with optional FPS resampling."""

import argparse
import numpy as np


def main():
    parser = argparse.ArgumentParser(description="Trim SMPL-X motion file by time range")
    parser.add_argument("--smplx_file", type=str, required=True, help="Input SMPL-X file (.npz or .pkl)")
    parser.add_argument("--start", type=float, default=0.0, help="Start time in seconds (default: 0)")
    parser.add_argument("--end", type=float, default=None, help="End time in seconds (default: end of file)")
    parser.add_argument("--fps", type=float, default=None, help="Output FPS, resample if different from source")
    parser.add_argument("--save_path", type=str, required=True, help="Output .npz path")
    args = parser.parse_args()

    # Load
    data = np.load(args.smplx_file, allow_pickle=True)
    data_dict = {k: data[k] for k in data.files}

    # Detect source FPS
    if "mocap_frame_rate" in data_dict:
        src_fps = float(data_dict["mocap_frame_rate"])
    elif "fps" in data_dict:
        src_fps = float(data_dict["fps"])
    else:
        src_fps = 30.0
        print(f"Warning: no FPS key found, assuming {src_fps}")

    # Detect number of frames from a known per-frame key
    frame_keys = ["transl", "trans", "global_orient", "root_orient", "pose_body", "body_pose"]
    n_frames = None
    for k in frame_keys:
        if k in data_dict and hasattr(data_dict[k], "shape") and len(data_dict[k].shape) >= 1:
            n_frames = data_dict[k].shape[0]
            break
    if n_frames is None:
        raise ValueError("Cannot determine number of frames from file")

    total_duration = n_frames / src_fps
    print(f"Source: {n_frames} frames, {src_fps} FPS, {total_duration:.2f}s")

    # Compute frame range
    start_frame = int(args.start * src_fps)
    end_frame = int(args.end * src_fps) if args.end is not None else n_frames
    start_frame = max(0, min(start_frame, n_frames))
    end_frame = max(start_frame, min(end_frame, n_frames))

    print(f"Trimming: frames [{start_frame}, {end_frame}) = {end_frame - start_frame} frames, "
          f"[{start_frame / src_fps:.2f}s, {end_frame / src_fps:.2f}s)")

    # Slice per-frame arrays
    trimmed = {}
    for k, v in data_dict.items():
        if hasattr(v, "shape") and len(v.shape) >= 1 and v.shape[0] == n_frames:
            trimmed[k] = v[start_frame:end_frame]
        else:
            trimmed[k] = v

    trimmed_frames = end_frame - start_frame

    # Resample FPS if requested
    out_fps = src_fps
    if args.fps is not None and args.fps != src_fps:
        out_fps = args.fps
        n_out = int(trimmed_frames * out_fps / src_fps)
        indices = np.linspace(0, trimmed_frames - 1, n_out).astype(int)
        for k, v in trimmed.items():
            if hasattr(v, "shape") and len(v.shape) >= 1 and v.shape[0] == trimmed_frames:
                trimmed[k] = v[indices]
        print(f"Resampled: {trimmed_frames} -> {n_out} frames at {out_fps} FPS")
        trimmed_frames = n_out

    # Update FPS key
    if "mocap_frame_rate" in trimmed:
        trimmed["mocap_frame_rate"] = np.array(out_fps)
    elif "fps" in trimmed:
        trimmed["fps"] = np.array(out_fps)
    else:
        trimmed["mocap_frame_rate"] = np.array(out_fps)

    np.savez(args.save_path, **trimmed)
    print(f"Saved to {args.save_path}: {trimmed_frames} frames at {out_fps} FPS")


if __name__ == "__main__":
    main()
