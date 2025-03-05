import os
import sys
import numpy as np
from tqdm import tqdm

sys.path.append('.')

from segmentation import ClothSegemntation
from TailorNet import TailorNet
from hugs.cfg.constants import *

def mocap_path(scene_name):
    # ./data/MoSh/MPI_mosh/50027/misc_dancing_hiphop_poses.npz
    if os.path.basename(scene_name) == 'seattle': # and opt.motion_name == 'moonwalk':
        # return './data/SFU/0018/0018_Moonwalk001_poses.npz', 0, 400, 4
        # return './data/SFU/0005/0005_Stomping001_poses.npz', 0, 800, 4
        return './data/SFU/0005/0005_SideSkip001_poses.npz', 0, 800, 4
    elif os.path.basename(scene_name) == 'citron': # and opt.motion_name == 'speedvault':
        # return './data/SFU/0008/0008_ChaCha001_poses.npz', 0, 1000, 4
        return './data/MPI_mosh/00093/irish_dance_poses.npz', 0, 1000, 4
        # return './data/SFU/0012/0012_SpeedVault001_poses.npz', 0, 340, 2
        # return './data/MPI_mosh/50027/misc_dancing_hiphop_poses.npz', 0, 2000, 4
        # return './data/SFU/0017/0017_ParkourRoll001_poses.npz', 140, 500, 4
    elif os.path.basename(scene_name) == 'parkinglot': # and opt.motion_name == 'yoga':
        return './data/SFU/0005/0005_2FeetJump001_poses.npz', 0, 1200, 4
        # return './data/SFU/0008/0008_Yoga001_poses.npz', 300, 1900, 8
    elif os.path.basename(scene_name) == 'bike': # and opt.motion_name == 'jumpandroll':
        return './data/MPI_mosh/50002/misc_poses.npz', 0, 250, 1
        # return './data/SFU/0018/0018_Moonwalk001_poses.npz', 0, 600, 4
        # return './data/SFU/0012/0012_JumpAndRoll001_poses.npz', 100, 400, 3
    elif os.path.basename(scene_name) == 'jogging': # and opt.motion_name == 'cartwheel':
        return './data/SFU/0007/0007_Cartwheel001_poses.npz', 200, 1000, 8
    elif os.path.basename(scene_name) == 'lab': # and opt.motion_name == 'chacha':
        return './data/SFU/0008/0008_ChaCha001_poses.npz', 0, 1000, 4
    else:
        raise ValueError('Define new elif branch')

if __name__=='__main__':

    seq = 'jogging'

    neu_data_path = "/mnt/data1/yu/data/ml-hugs/dataset/neuman/dataset/"

    load_smpl_path = neu_data_path+seq+"/4d_humans/smpl_optimized_aligned_scale.npz"

    smpl_params_path = load_smpl_path
    smpl_params = np.load(smpl_params_path)
    smpl_params = {f: smpl_params[f] for f in smpl_params.files}


    motion_path, start_idx, end_idx, skip = mocap_path(seq)
    motions = np.load(motion_path)
    poses = motions['poses'][start_idx:end_idx:skip, AMASS_SMPLH_TO_SMPL_JOINTS]
    transl = motions['trans'][start_idx:end_idx:skip]
    betas = smpl_params['betas'][0]
    smpl_params = {
        'global_orient': poses[:, :3],
        'body_pose': poses[:, 3:],
        'transl': transl,
        'scale': np.array([1.0] * poses.shape[0]),
        'betas': betas[None].repeat(poses.shape[0], 0)[:, :10],
    }





    #tailorNet.run_demo()


    image_dir = neu_data_path+seq+"/images"
    human_segmented_image_dir = neu_data_path+seq+"/segmentations"
    result_dir = neu_data_path+seq+"/cloth_segmented_images"

    need_to_run_cloth_segmentation = True
    if need_to_run_cloth_segmentation:
        clothSegemntation = ClothSegemntation(image_dir, result_dir,human_segmented_image_dir)
        clothSegemntation.infer()

    need_to_run_tailornet = True
    if need_to_run_tailornet:
        tailorNet = TailorNet('male','t-shirt')
        tailorNet.out_path = os.path.join(os.getcwd(),"cloth_mesh_output")

        vert_indeces = tailorNet.get_vert_indices()
        f = tailorNet.get_f()

        # run inference for every frame
        pbar = tqdm(total=poses.shape[0])
        for i in range(poses.shape[0]):
            pbar.update(1)
            theta = np.concatenate([smpl_params['global_orient'][i], smpl_params['body_pose'][i]]).astype(np.float32)
            beta = smpl_params['betas'][i].astype(np.float32)
            tailorNet.run_tailornet(theta,beta,"cloth_{:04d}".format(i))

        zero_theta = np.zeros(poses.shape[1])
        # let zero_theta[0] = smpl_params['global_orient'][i]
        zero_theta[0:3] = smpl_params['global_orient'][0]


        tailorNet.run_tailornet(zero_theta,smpl_params['betas'][0],"canon_cloth")
