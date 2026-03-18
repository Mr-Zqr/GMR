# pip install moderngl-window==2.4.6 pyglet chumpy aitviewer
from aitviewer.renderables.smpl import SMPLSequence
from aitviewer.viewer import Viewer
from aitviewer.models.smpl import SMPLLayer
from aitviewer.configuration import CONFIG
import torch, joblib
import numpy as np
from aitviewer.renderables.lines import Lines

CONFIG.smplx_models = '/home/amax/devel/GMR/assets/body_models/smplx/SMPLX_NEUTRAL.npz'
CONFIG.playback_fps = 30


def process_one(pkl_path, video_path=None):
    smpl_data = joblib.load(pkl_path)

    trans = smpl_data['smpl_trans_wd']
    T = trans.shape[0]
    root_orient = torch.from_numpy(smpl_data['smpl_root_orient_wd']).squeeze(1)
    body_pose = torch.from_numpy(smpl_data['smpl_body_pose']).flatten(1, 2)[:, :63]
    betas = torch.from_numpy(smpl_data['smpl_shapes'])[None].repeat(T, 1)

    v = Viewer()
    smpl_seq = SMPLSequence(
        smpl_layer=SMPLLayer(ext='npz'),
        poses_root=root_orient,
        poses_body=body_pose,
        betas=betas,
        trans=trans,
    )
    v.scene.add(smpl_seq)

    line = Lines(
        trans,
        color=(1.0, 0.0, 0.0, 1.0),
    )
    v.scene.add(line)

    v._init_scene()

    # Match main_humanoid.py camera angle
    cam = v.scene.camera
    new_target = cam.target.copy()
    new_target[1] += 0.8
    dir_vec = cam.position - cam.target
    cam.target = new_target
    cam.position = new_target + dir_vec * 1.50

    if video_path is not None:
        import os
        os.makedirs(os.path.dirname(video_path) if os.path.dirname(video_path) else '.', exist_ok=True)
        v.export_video(output_path=video_path)
    else:
        v.run()


if __name__ == '__main__':
    import argparse
    import glob
    import os
    import subprocess
    import sys

    parser = argparse.ArgumentParser()
    parser.add_argument('--pkl_file', type=str,
                        default='/home/amax/devel/dataset/NeoBot/6_bad_motion/bad_motion/6_pjpe.pkl')
    parser.add_argument('--pkl_dir', type=str, default=None,
                        help='Directory to scan for *.pkl files (batch mode)')
    parser.add_argument('--record_video', action='store_true')
    parser.add_argument('--video_path', type=str, default='videos/smpl_output.mp4')
    parser.add_argument('--video_dir', type=str, default='videos',
                        help='Output directory for batch video export')
    args = parser.parse_args()

    if args.pkl_dir is not None:
        # Batch mode: scan directory and spawn subprocess per file
        pkl_files = sorted(glob.glob(os.path.join(args.pkl_dir, '**/*.pkl'), recursive=True))
        total = len(pkl_files)
        print(f'Found {total} pkl files in {args.pkl_dir}')
        os.makedirs(args.video_dir, exist_ok=True)
        script_path = os.path.abspath(__file__)
        for i, pkl_path in enumerate(pkl_files):
            stem = os.path.splitext(os.path.basename(pkl_path))[0]
            video_path = os.path.join(args.video_dir, stem + '.mp4')
            print(f'[{i+1}/{total}] {stem}')
            env = os.environ.copy()
            env['CUDA_VISIBLE_DEVICES'] = ''
            cmd = [sys.executable, script_path,
                   '--pkl_file', pkl_path,
                   '--record_video',
                   '--video_path', video_path]
            result = subprocess.run(cmd, capture_output=True, text=True, env=env)
            if result.returncode != 0:
                print(f'  ERROR: {result.stderr[-300:]}')
    else:
        if args.record_video:
            CONFIG.window_type = 'headless'
            process_one(args.pkl_file, video_path=args.video_path)
        else:
            process_one(args.pkl_file, video_path=None)
