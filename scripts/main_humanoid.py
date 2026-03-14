# pip install moderngl-window==2.4.6 pyglet chumpy aitviewer
from aitviewer.renderables.smpl import SMPLSequence
import numpy as np
from aitviewer.viewer import Viewer
from aitviewer.models.smpl import SMPLLayer
from aitviewer.configuration import CONFIG
import torch
import joblib
import argparse

CONFIG.smplx_models = '/home/amax/devel/GMR/assets/body_models/smplx/SMPLX_NEUTRAL.npz'
CONFIG.playback_fps = 30

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Visualize SMPL-X human motion')
    parser.add_argument('--smplx_file', type=str, default='/home/amax/devel/dataset/NeoBot/neobot-testset/smpl/ulom/long/02_07_stageii.npz',
                        help='Path to SMPL-X .npz file')
    parser.add_argument('--record_video', action='store_true', help='Export video instead of interactive view')
    parser.add_argument('--video_path', type=str, default='videos/smpl_output.mp4', help='Output video path')
    parser.add_argument('--zup', action='store_true',
                        help='Input motion is Z-up; rotate to Y-up before visualization')
    args = parser.parse_args()

    if args.record_video:
        CONFIG.window_type = 'headless'

    smpl_data = np.load(args.smplx_file)

    args.zup = True
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
    if args.zup:
        from scipy.spatial.transform import Rotation as R
        r_zup2yup = R.from_euler('x', -90, degrees=True)
        global_orient = (r_zup2yup * R.from_rotvec(global_orient)).as_rotvec().astype(np.float32)
        # (x, y, z) → (x, z, -y)
        transl = transl[:, [0, 2, 1]].copy().astype(np.float32)
        transl[:, 2] *= -1

    # 降采样为原来的1/2
    body_pose = body_pose[::2]
    global_orient = global_orient[::2]
    transl = transl[::2]
    v = Viewer()
    smpl_seq = SMPLSequence(
        smpl_layer=SMPLLayer(ext='npz'),
        poses_root=global_orient, # T, 3
        poses_body=body_pose, # T, 63
        trans=transl,
    )
    v.scene.add(smpl_seq)
    if args.record_video:
        v._init_scene()
        cam = v.scene.camera
        new_target = cam.target.copy()
        new_target[1] += 0.8  # y is up in aitviewer
        dir_vec = cam.position - cam.target
        cam.target = new_target
        cam.position = new_target + dir_vec * 1.50  # pull back to 80% zoom
        v.export_video(output_path=args.video_path)
    else:
        v.run()
