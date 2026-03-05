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
    parser.add_argument('--smplx_file', type=str, default='/home/amax/devel/dataset/NeoBot/1_filtered_smplx/Mirror_MotionGV/folder1/079786.npz',
                        help='Path to SMPL-X .npz file')
    parser.add_argument('--record_video', action='store_true', help='Export video instead of interactive view')
    parser.add_argument('--video_path', type=str, default='videos/smpl_output.mp4', help='Output video path')
    args = parser.parse_args()

    if args.record_video:
        CONFIG.window_type = 'headless'

    smpl_data = np.load(args.smplx_file)

    body_pose = smpl_data['body_pose']
    global_orient = smpl_data['global_orient']
    transl = smpl_data['transl']

    # Reshape from (T, 21, 3) to (T, 63)
    body_pose = body_pose.reshape(body_pose.shape[0], -1)

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
