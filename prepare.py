import os
import sys
import numpy as np
import glob
import cv2
from tqdm import tqdm

sys.path.append('.')

from segmentation import ClothSegemntation
from TailorNet import TailorNet
from hugs.cfg.constants import *
from segment_anything import SamPredictor, sam_model_registry
from scipy.ndimage import label, sum as ndi_sum

CHECKPOINT = os.path.expanduser("~/project/segment-anything/ckpts/sam_vit_h_4b8939.pth")
MODEL = "vit_h"


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
    
def remove_small_objects(mask, min_size):
    # 连通域标记
    labeled_mask, num_features = label(mask)
    
    # 计算每个连通域的大小
    sizes = ndi_sum(mask, labeled_mask, index=np.arange(1, num_features + 1))
    
    # 保留面积大于等于 min_size 的连通域
    valid_labels = np.where(sizes >= min_size)[0] + 1  # 标签从1开始
    cleaned_mask = np.isin(labeled_mask, valid_labels).astype(np.uint8)
    
    return cleaned_mask

def calculateBoundingBox(mask):
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)

    # 如果 mask 全为 0，返回空的包围盒
    if not np.any(rows) or not np.any(cols):
        return np.array([0, 0, 0, 0])

    # 找到非零值在行和列的范围
    y_min, y_max = np.where(rows)[0][[0, -1]]
    x_min, x_max = np.where(cols)[0][[0, -1]]
    
    # 返回包围盒的坐标 [x_min, y_min, x_max, y_max] 作为 np.ndarray
    return np.array([x_min, y_min, x_max + 1, y_max + 1])


if __name__=='__main__':

    seq = 'seattle'

    neu_data_path = "/mnt/data1/yu/data/ClothGS/dataset/neuman/dataset/"

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
    
    sam = sam_model_registry[MODEL](checkpoint=CHECKPOINT)
    sam.to("cuda")
    predictor = SamPredictor(sam)

    image_dir = neu_data_path+seq+"/images"
    human_segmented_image_dir = neu_data_path+seq+"/segmentations"
    keypoints_dir = neu_data_path+seq+"/keypoints"
    result_dir = neu_data_path+seq+"/cloth_segmented_images"
    
    upperbody_img_path = os.path.join(result_dir,"upperbody")
    lowerbody_img_path = os.path.join(result_dir,"lowerbody")
    humanbody_img_path = os.path.join(result_dir,"humanbody")
    
    
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
    if not os.path.exists(upperbody_img_path):
        os.makedirs(upperbody_img_path)
    if not os.path.exists(lowerbody_img_path):
        os.makedirs(lowerbody_img_path)
    if not os.path.exists(humanbody_img_path):
        os.makedirs(humanbody_img_path)
        
    image_list = os.listdir(image_dir)
    
    img_lists = sorted(glob.glob(f"{image_dir}/*.png"))
    
    pbar = tqdm(total=len(image_list))
    
    clothSegemntation = ClothSegemntation() 
    
    for image_name in img_lists:

        # get basename of image
        basename = os.path.basename(image_name)

        # get segemnt mask of human in human_segmented_image_dir and same name as image
        human_mask = cv2.imread(os.path.join(human_segmented_image_dir,basename))

        # make human mask in one channel
        human_mask = human_mask[:,:,0]

        img = cv2.imread(image_name)
        predictor.set_image(img)
        
        pbar.update(1)
        upperbody_mask,lowerbody_mask= clothSegemntation.inferOne(image_name)
        
        upperbody_mask = remove_small_objects(upperbody_mask, 1000)
        upperbody_box = calculateBoundingBox(upperbody_mask)
        
        lowerbody_mask = remove_small_objects(lowerbody_mask, 1000)
        lowerbody_box = calculateBoundingBox(lowerbody_mask)
        
        upperbody_mask, _, _ = predictor.predict(box=upperbody_box)
        upperbody_mask = upperbody_mask.sum(axis=0) > 0
        upperbody_mask = remove_small_objects(upperbody_mask, 1000)
        
        # get file name without extension
        img_name = os.path.basename(image_name).split(".")[0]

        upperbody_mask_name = img_name +"_upperbody_mask" + ".png"
        # save file name 
        save_name = os.path.join(upperbody_img_path,upperbody_mask_name)
        upperbody_mask = upperbody_mask.astype(np.bool_)
        upperbody_mask = ~upperbody_mask
        upperbody_mask = upperbody_mask.astype(np.int8)
        cv2.imwrite(save_name, upperbody_mask * 255)

        # # make cloth mask in bool
        # upperbody_mask = upperbody_mask.astype(np.bool)

        # upperbody_mask_img = cv2.imread(image_name)
        # upperbody_mask_img[~upperbody_mask] = 0

        # # draw box in img
        # # cv2.rectangle(img, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
        # cv2.imwrite(fn.replace("images", "cloth_masked_sam_images/upperbody"), upperbody_mask_img)
        
        lowerbody_mask, _, _ = predictor.predict(box=lowerbody_box)
        lowerbody_mask = lowerbody_mask.sum(axis=0) > 0
        lowerbody_mask = remove_small_objects(lowerbody_mask, 1000)
        # draw box in cloth_mask in tensor
        lowerbody_mask = lowerbody_mask.astype(np.bool_)
        lowerbody_mask = ~lowerbody_mask
        lowerbody_mask = lowerbody_mask.astype(np.int8)
        lowerbody_mask_name = img_name +"_lowerbody_mask" + ".png"
        # save file name
        save_name = os.path.join(lowerbody_img_path,lowerbody_mask_name)
        cv2.imwrite(save_name, lowerbody_mask * 255)
        
        # # make cloth mask in bool
        # lowwerbody_mask = lowwerbody_mask.astype(np.bool)
        
        # lowwerbody_mask_img = cv2.imread(fn)
        # lowwerbody_mask_img[~lowwerbody_mask] = 0
        
        # # draw box in img
        # # cv2.rectangle(img, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
        
        # cv2.imwrite(fn.replace("images", "cloth_masked_sam_images/lowerbody"), lowwerbody_mask_img)

        # humanbody_mask = human_mask - upperbody_mask - lowwerbody_mask
        # need bitwise_and to get the intersection of two masks

        upperbody_mask = upperbody_mask.astype(np.bool_)
        lowerbody_mask = lowerbody_mask.astype(np.bool_)
        human_mask = human_mask.astype(np.bool_)

        
        
        human_mask = ~human_mask
        humanbody_mask = human_mask & upperbody_mask & lowerbody_mask

        humanbody_mask_name = img_name +"_humanbody_mask" + ".png"
        # save file name
        save_name = os.path.join(humanbody_img_path,humanbody_mask_name)

        # make humanbody_mask in 0 and 255


        remove_small_objects(humanbody_mask, 1000)

        humanbody_mask = ~humanbody_mask

        humanbody_mask = humanbody_mask.astype(np.uint8) * 255

        cv2.imwrite(save_name, humanbody_mask)
        

    
    
    

    need_to_run_cloth_segmentation = False
    if need_to_run_cloth_segmentation:
        clothSegemntation = ClothSegemntation(image_dir, result_dir,human_segmented_image_dir)
        clothSegemntation.infer()

    need_to_run_tailornet = False
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
