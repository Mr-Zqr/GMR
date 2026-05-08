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


def process_one(smplx_file, video_path, zup=False, rot_y180=False):
    from scipy.spatial.transform import Rotation as _R
    # smpl_data = np.load(smplx_file)
    smpl_data = joblib.load(smplx_file)  # for .pkl files saved by joblib.dump()

    # Unwrap NeoBot 'trackings' nesting
    src = smpl_data
    if isinstance(smpl_data, dict) and 'trackings' in smpl_data and \
            'body_pose' not in smpl_data and 'pose_body' not in smpl_data and \
            'fullpose' not in smpl_data:
        trackings = smpl_data['trackings']
        if isinstance(trackings, (list, tuple)):
            src = trackings[0]
        elif isinstance(trackings, dict):
            src = trackings
        elif isinstance(trackings, np.ndarray) and trackings.dtype == object:
            src = trackings.item() if trackings.shape == () else trackings[0]

    # body_pose
    if 'body_pose' in src:
        body_pose = np.asarray(src['body_pose'])
    elif 'pose_body' in src:
        body_pose = np.asarray(src['pose_body'])
    elif 'fullpose' in src:
        body_pose = np.array(src['fullpose'])[:, 3:66]
    elif 'smpl_body_pose' in src:
        bp = np.asarray(src['smpl_body_pose'])
        if bp.ndim == 4:  # (T, J, 3, 3)
            T = bp.shape[0]
            body_pose = _R.from_matrix(bp[:, :21].reshape(-1, 3, 3)).as_rotvec().reshape(T, -1)
        elif bp.ndim == 3 and bp.shape[2] == 3 and bp.shape[1] > 3:  # (T, J, 3)
            body_pose = bp[:, :21].reshape(bp.shape[0], -1)
        else:
            body_pose = bp
    else:
        raise KeyError(f"Cannot find body_pose. Available keys: {list(src.keys())}")

    # global_orient
    if 'global_orient' in src:
        global_orient = np.asarray(src['global_orient'])
    elif 'root_orient' in src:
        global_orient = np.asarray(src['root_orient'])
    elif 'fullpose' in src:
        global_orient = np.array(src['fullpose'])[:, :3]
    elif 'smpl_root_orient_wd' in src:
        go = np.asarray(src['smpl_root_orient_wd'])
        if go.ndim == 4:  # (T, 1, 3, 3)
            global_orient = _R.from_matrix(go[:, 0]).as_rotvec()
        elif go.ndim == 3 and go.shape[1] == 1 and go.shape[2] == 3:  # (T, 1, 3)
            global_orient = go[:, 0, :]
        elif go.ndim == 3 and go.shape[1:] == (3, 3):  # (T, 3, 3)
            global_orient = _R.from_matrix(go).as_rotvec()
        else:
            global_orient = go
    else:
        raise KeyError(f"Cannot find global_orient. Available keys: {list(src.keys())}")

    # transl
    if 'transl' in src:
        transl = np.asarray(src['transl'])
    elif 'trans' in src:
        transl = np.asarray(src['trans'])
    elif 'smpl_trans_wd' in src:
        transl = np.asarray(src['smpl_trans_wd'])
    else:
        raise KeyError(f"Cannot find transl. Available keys: {list(src.keys())}")

    # Reshape from (T, 21, 3) or (T, 63) to (T, 63)
    body_pose = body_pose.reshape(body_pose.shape[0], -1)
    if body_pose.shape[1] > 63:
        body_pose = body_pose[:, :63]

    # Z-up → Y-up: pre-multiply root orientation by Rx(-90°), rotate translation axes
    if zup:
        r_zup2yup = _R.from_euler('x', -90, degrees=True)
        global_orient = (r_zup2yup * _R.from_rotvec(global_orient)).as_rotvec().astype(np.float32)
        # (x, y, z) → (x, z, -y)
        transl = transl[:, [0, 2, 1]].copy().astype(np.float32)
        transl[:, 2] *= -1

    # Rotate 180° around Y axis: flip x and z of translation, pre-multiply orientation
    if rot_y180:
        r_y180 = _R.from_euler('y', 180, degrees=True)
        global_orient = (r_y180 * _R.from_rotvec(global_orient)).as_rotvec().astype(np.float32)
        transl = transl.copy().astype(np.float32)
        transl[:, 0] *= -1
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
    parser.add_argument('--rot_y180', action='store_true',
                        help='Rotate motion 180° around Y axis (flip facing direction)')
    args = parser.parse_args()

    if args.smplx_dir:
        # Batch mode: spawn a subprocess per file to avoid aitviewer GL context singleton issue
        import subprocess
        import sys

        if args.video_dir is None:
            parser.error('--video_dir is required when using --smplx_dir')

        smplx_dir = os.path.abspath(args.smplx_dir)
        video_dir = os.path.abspath(args.video_dir)
        npz_files = sorted(glob.glob(os.path.join(smplx_dir, '**', '*.pkl'), recursive=True))

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
                if args.rot_y180:
                    cmd.append('--rot_y180')
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
            process_one(smplx_file, args.video_path, zup=args.zup, rot_y180=args.rot_y180)
        else:
            smpl_data = np.load(smplx_file, allow_pickle=True)
            # smpl_data = joblib.load(smplx_file)  # for .pkl files saved by joblib.dump()
            if isinstance(smpl_data, np.ndarray):
                smpl_data = smpl_data.item()
            # NeoBot video pkl: SMPL data nested under 'trackings'
            src = smpl_data
            if 'trackings' in smpl_data and 'body_pose' not in smpl_data and 'pose_body' not in smpl_data and 'fullpose' not in smpl_data:
                trackings = smpl_data['trackings']
                if isinstance(trackings, (list, tuple)):
                    src = trackings[0]
                elif isinstance(trackings, dict):
                    src = trackings
                elif isinstance(trackings, np.ndarray) and trackings.dtype == object:
                    src = trackings.item() if trackings.shape == () else trackings[0]

            if 'body_pose' in src:
                body_pose = np.asarray(src['body_pose'])
            elif 'pose_body' in src:
                body_pose = np.asarray(src['pose_body'])
            elif 'fullpose' in src:
                fullpose = np.array(src['fullpose'])
                body_pose = fullpose[:, 3:66]
            elif 'smpl_body_pose' in src:
                from scipy.spatial.transform import Rotation as _R
                bp = np.asarray(src['smpl_body_pose'])
                if bp.ndim == 4:  # (T, J, 3, 3) rotation matrices
                    T = bp.shape[0]
                    body_pose = _R.from_matrix(bp[:, :21].reshape(-1, 3, 3)).as_rotvec().reshape(T, -1)
                elif bp.ndim == 3 and bp.shape[2] == 3 and bp.shape[1] > 3:  # (T, J, 3) axis-angle
                    body_pose = bp[:, :21].reshape(bp.shape[0], -1)
                else:
                    body_pose = bp
            else:
                raise KeyError(f"Cannot find body_pose. Available keys: {list(src.keys())}")

            if 'global_orient' in src:
                global_orient = np.asarray(src['global_orient'])
            elif 'root_orient' in src:
                global_orient = np.asarray(src['root_orient'])
            elif 'fullpose' in src:
                fullpose = np.array(src['fullpose'])
                global_orient = fullpose[:, :3]
            elif 'smpl_root_orient_wd' in src:
                from scipy.spatial.transform import Rotation as _R
                go = np.asarray(src['smpl_root_orient_wd'])
                if go.ndim == 4:  # (T, 1, 3, 3)
                    global_orient = _R.from_matrix(go[:, 0]).as_rotvec()
                elif go.ndim == 3 and go.shape[1] == 1 and go.shape[2] == 3:  # (T, 1, 3)
                    global_orient = go[:, 0, :]
                elif go.ndim == 3 and go.shape[1:] == (3, 3):  # (T, 3, 3)
                    global_orient = _R.from_matrix(go).as_rotvec()
                else:
                    global_orient = go
            else:
                raise KeyError(f"Cannot find global_orient. Available keys: {list(src.keys())}")

            if 'transl' in src:
                transl = np.asarray(src['transl'])
            elif 'trans' in src:
                transl = np.asarray(src['trans'])
            elif 'smpl_trans_wd' in src:
                transl = np.asarray(src['smpl_trans_wd'])
            else:
                raise KeyError(f"Cannot find transl. Available keys: {list(src.keys())}")

            body_pose = body_pose.reshape(body_pose.shape[0], -1)
            if body_pose.shape[1] > 63:
                body_pose = body_pose[:, :63]

            if args.zup:
                from scipy.spatial.transform import Rotation as _R
                r_zup2yup = _R.from_euler('x', -90, degrees=True)
                global_orient = (r_zup2yup * _R.from_rotvec(global_orient)).as_rotvec().astype(np.float32)
                transl = transl[:, [0, 2, 1]].copy().astype(np.float32)
                transl[:, 2] *= -1

            if args.rot_y180:
                from scipy.spatial.transform import Rotation as _R
                r_y180 = _R.from_euler('y', 180, degrees=True)
                global_orient = (r_y180 * _R.from_rotvec(global_orient)).as_rotvec().astype(np.float32)
                transl = transl.copy().astype(np.float32)
                transl[:, 0] *= -1
                transl[:, 2] *= -1

            body_pose = body_pose[::2]
            global_orient = global_orient[::2]
            transl = transl[::2]

            smpl_g1 = joblib.load('/home/amax/devel/dataset/NeoBot/smpl_betas/shape_optimized_asap.pkl')
            smpl_g1 = np.zeros((10,), dtype=np.float32)
            v = Viewer()
            smpl_seq = SMPLSequence(
                smpl_layer=SMPLLayer(ext='npz'),
                poses_root=global_orient,
                poses_body=body_pose,
                trans=transl,
                betas=smpl_g1
            )
            v.scene.add(smpl_seq)
            v.run()
