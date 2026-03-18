# pip install moderngl-window==2.4.6 pyglet chumpy aitviewer
from aitviewer.renderables.smpl import SMPLSequence
import numpy as np
from aitviewer.viewer import Viewer
from aitviewer.models.smpl import SMPLLayer
from aitviewer.configuration import CONFIG
import torch
import joblib
import argparse
import glob
import os

CONFIG.smplx_models = '/home/amax/devel/GMR/assets/body_models/smplx/SMPLX_NEUTRAL.npz'
CONFIG.playback_fps = 30


def process_one(smplx_file, video_path, zup=False):
    smpl_data = np.load(smplx_file)

    # Support both SMPL-X ('body_pose'/'global_orient'/'transl') and
    # AMASS/SMPL-H ('pose_body'/'root_orient'/'trans') key conventions.
    body_pose = smpl_data['body_pose'] if 'body_pose' in smpl_data else smpl_data['pose_body']
    global_orient = smpl_data['global_orient'] if 'global_orient' in smpl_data else smpl_data['root_orient']
    transl = smpl_data['transl'] if 'transl' in smpl_data else smpl_data['trans']

    # Reshape from (T, 21, 3) or (T, 63) to (T, 63)
    body_pose = body_pose.reshape(body_pose.shape[0], -1)
    if body_pose.shape[1] > 63:
        body_pose = body_pose[:, :63]

    # Z-up → Y-up: pre-multiply root orientation by Rx(-90°), rotate translation axes
    if zup:
        from scipy.spatial.transform import Rotation as R
        r_zup2yup = R.from_euler('x', -90, degrees=True)
        global_orient = (r_zup2yup * R.from_rotvec(global_orient)).as_rotvec().astype(np.float32)
        # (x, y, z) → (x, z, -y)
        transl = transl[:, [0, 2, 1]].copy().astype(np.float32)
        transl[:, 2] *= -1

    # # 降采样为原来的1/2
    # body_pose = body_pose[::2]
    # global_orient = global_orient[::2]
    # transl = transl[::2]

    v = Viewer()
    smpl_seq = SMPLSequence(
        smpl_layer=SMPLLayer(ext='npz'),
        poses_root=global_orient,  # T, 3
        poses_body=body_pose,      # T, 63
        trans=transl,
    )
    v.scene.add(smpl_seq)
    v._init_scene()
    cam = v.scene.camera
    new_target = cam.target.copy()
    new_target[1] += 0.8  # y is up in aitviewer
    dir_vec = cam.position - cam.target
    cam.target = new_target
    cam.position = new_target + dir_vec * 1.50  # pull back to 80% zoom
    v.export_video(output_path=video_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Visualize SMPL-X human motion')
    parser.add_argument('--smplx_file', type=str, default=None,
                        help='Path to a single SMPL-X .npz file')
    parser.add_argument('--smplx_dir', type=str, default=None,
                        help='Input folder; recursively process all .npz files')
    parser.add_argument('--record_video', action='store_true', help='Export video instead of interactive view')
    parser.add_argument('--video_path', type=str, default='videos/smpl_output.mp4', help='Output video path (single-file mode)')
    parser.add_argument('--video_dir', type=str, default=None,
                        help='Output root folder for batch mode (mirrors input folder structure)')
    parser.add_argument('--zup', action='store_true',
                        help='Input motion is Z-up; rotate to Y-up before visualization')
    args = parser.parse_args()

    if args.smplx_dir:
        # Batch mode: spawn a subprocess per file to avoid aitviewer GL context singleton issue
        import subprocess
        import sys

        if args.video_dir is None:
            parser.error('--video_dir is required when using --smplx_dir')

        smplx_dir = os.path.abspath(args.smplx_dir)
        video_dir = os.path.abspath(args.video_dir)
        npz_files = sorted(glob.glob(os.path.join(smplx_dir, '**', '*.npz'), recursive=True))

        if not npz_files:
            print(f'No .npz files found in {smplx_dir}')
        else:
            print(f'Found {len(npz_files)} files, exporting to {video_dir}')
            script = os.path.abspath(__file__)
            for i, npz_path in enumerate(npz_files):
                rel = os.path.relpath(npz_path, smplx_dir)
                video_path = os.path.join(video_dir, os.path.splitext(rel)[0] + '.mp4')
                os.makedirs(os.path.dirname(video_path), exist_ok=True)
                print(f'[{i+1}/{len(npz_files)}] {rel}')
                cmd = [sys.executable, script,
                       '--smplx_file', npz_path,
                       '--record_video',
                       '--video_path', video_path]
                if args.zup:
                    cmd.append('--zup')
                env = os.environ.copy()
                env['CUDA_VISIBLE_DEVICES'] = ''  # force CPU in subprocess; avoids parent-process VRAM conflict
                result = subprocess.run(cmd, capture_output=True, text=True, env=env)
                if result.returncode != 0:
                    print(f'  ERROR: {result.stderr[-300:] if result.stderr else "unknown"}')

    else:
        # Single-file mode
        smplx_file = args.smplx_file or '/home/amax/devel/dataset/NeoBot/neobot-testset/smpl/ulom/long/02_07_stageii.npz'

        if args.record_video:
            CONFIG.window_type = 'headless'
            process_one(smplx_file, args.video_path, zup=args.zup)
        else:
            smpl_data = np.load(smplx_file)
            body_pose = smpl_data['body_pose'] if 'body_pose' in smpl_data else smpl_data['pose_body']
            global_orient = smpl_data['global_orient'] if 'global_orient' in smpl_data else smpl_data['root_orient']
            transl = smpl_data['transl'] if 'transl' in smpl_data else smpl_data['trans']

            body_pose = body_pose.reshape(body_pose.shape[0], -1)
            if body_pose.shape[1] > 63:
                body_pose = body_pose[:, :63]

            from scipy.spatial.transform import Rotation as R
            r_zup2yup = R.from_euler('x', -90, degrees=True)
            global_orient = (r_zup2yup * R.from_rotvec(global_orient)).as_rotvec().astype(np.float32)
            transl = transl[:, [0, 2, 1]].copy().astype(np.float32)
            transl[:, 2] *= -1

            body_pose = body_pose[::2]
            global_orient = global_orient[::2]
            transl = transl[::2]

            v = Viewer()
            smpl_seq = SMPLSequence(
                smpl_layer=SMPLLayer(ext='npz'),
                poses_root=global_orient,
                poses_body=body_pose,
                trans=transl,
            )
            v.scene.add(smpl_seq)
            v.run()
