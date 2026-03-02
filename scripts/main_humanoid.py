# pip install moderngl-window==2.4.6 pyglet chumpy aitviewer
from aitviewer.renderables.smpl import SMPLSequence
import numpy as np
from aitviewer.viewer import Viewer
from aitviewer.models.smpl import SMPLLayer
from aitviewer.configuration import CONFIG
import torch
import joblib

CONFIG.smplx_models = '/home/amax/devel/GMR/assets/body_models/smplx/SMPLX_NEUTRAL.npz'
CONFIG.playback_fps = 30

if __name__ == '__main__':
    # smpl_data = joblib.load('/home/amax/devel/dataset/jinitaimei/99360891-1-30032_scene_0_1775.pkl')[145]
    smpl_data = joblib.load('/run/user/1000/gvfs/sftp:host=114.212.175.192,port=10063/home/zhaoqr/devel/neobot_neural_retarget/humanoid/data/tmp_data/zhaoqr/dataset/neobot/2_gmr_retarget_filtered-0105-0040/MotionUnion/fitness/000116.npz')
    # smpl_data = np.load('/home/amax/devel/Neobot_project/neobot/1_filtered_smplx/MotionGV/folder3/200042.npz')
    # smpl_beta = joblib.load( '/home/amax/devel/dataset/NeoBot/smpl_betas/shape_optimized_asap.pkl')
    # body_pose = smpl_data['smpl_params_global']['body_pose'] # T, 63
    # global_orient = smpl_data['smpl_params_global']['global_orient'] # T, 3
    # transl = smpl_data['smpl_params_global']['transl'] # T, 3

    body_pose = smpl_data['smplx_body_pose']
    global_orient = smpl_data['smplx_global_orient']
    transl = smpl_data['smplx_transl']
    # smpl_beta = smpl_beta[0].detach().cpu().numpy().copy()

    # Reshape from (T, 21, 3) to (T, 63) and (T, 1, 3) to (T, 3)
    body_pose = body_pose.reshape(body_pose.shape[0], -1)
    # global_orient = global_orient.squeeze(1)

    v = Viewer()
    smpl_seq = SMPLSequence(
        smpl_layer=SMPLLayer(ext='npz'), 
        poses_root=global_orient, # T, 3
        poses_body=body_pose, # T, 63
        trans=transl, 
        # betas = smpl_beta
    )
    v.scene.add(smpl_seq)
    v.run()
