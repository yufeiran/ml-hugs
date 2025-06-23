import numpy as np
import os
import pickle
import torch


print(os.getcwd())

from rotations import (
    axis_angle_to_rotation_6d,
    matrix_to_quaternion,
    matrix_to_rotation_6d,
    quaternion_multiply,
    quaternion_to_matrix,
    rotation_6d_to_axis_angle,
    rotation_6d_to_matrix,
    torch_rotation_matrix_from_vectors,
)

def get_data_splits(scene_length):
    num_val = scene_length // 5
    length = int(1 / (num_val) * scene_length)
    offset = length // 2
    val_list = list(range(scene_length))[offset::length]
    train_list = list(set(range(scene_length)) - set(val_list))
    test_list = val_list[:len(val_list) // 2]
    val_list = val_list[len(val_list) // 2:]
    assert len(train_list) > 0
    assert len(test_list) > 0
    assert len(val_list) > 0    
    return train_list, val_list, test_list

if __name__=='__main__':
    t_iter = 0
    
    print("pwd: ", os.getcwd())

    base_path = os.getcwd()+'/results/smpl_result/'
    
    
    seq = 'parkinglot/'

    pre_process_path = base_path + seq + 'smpl_optimized_aligned_scale.npz'
    
    our_path = base_path + seq + 'human_params_014000.pth.npy'

    # with open(pre_process_path, 'rb') as f:
    #     pre_process = pickle.load(f)
    smpl_params = np.load(pre_process_path)
    smpl_params = {f: smpl_params[f] for f in smpl_params.files}
    
    # Load the human parameters
    our_params = np.load(our_path, allow_pickle=True)
    our_params = our_params.item()
    
    train_split, _, val_split = get_data_splits(len(smpl_params['global_orient']))
    
    # 添加一个map 映射训练集图片id 到原始视频帧id
    
    map_dict = {}
    for i in range(len(train_split)):
        map_dict[train_split[i]] = i

    
    
    # split the data

    for key in smpl_params.keys():
        smpl_params[key] = smpl_params[key][train_split]
        
    pretrain_global_orient = smpl_params['global_orient']
    
    pretrain_body_pose = smpl_params['body_pose']
    
    
    pretrain_transl = smpl_params['transl']
    
    # to tensor
    pretrain_global_orient = torch.tensor(pretrain_global_orient, dtype=torch.float32)
    
    pretrain_body_pose = torch.tensor(pretrain_body_pose, dtype=torch.float32)
    
    pretrain_transl = torch.tensor(pretrain_transl, dtype=torch.float32)
    
    
    
    
    global_orient = our_params['global_orient']

    # to tensor 
    global_orient = torch.tensor(global_orient, dtype=torch.float32)
    
    global_orient = rotation_6d_to_axis_angle(
        global_orient.reshape(-1, 6)).reshape(-1,3)
    
    global_orient_mat = our_params['global_orient']
    global_orient_mat = torch.tensor(global_orient_mat, dtype=torch.float32)
    global_orient_mat = rotation_6d_to_matrix(
        global_orient_mat.reshape(-1, 6)).reshape(-1,3,3)

    # save global_orient_mat to file
    path = base_path + seq + 'global_orient_mat.npy'
    np.save(path, global_orient_mat)

    body_pose = our_params['body_pose']
    
    # to tensor
    body_pose = torch.tensor(body_pose, dtype=torch.float32)
    
    body_pose = rotation_6d_to_axis_angle(
        body_pose.reshape(-1, 6)).reshape(-1,23 * 3)
    
    # save body_pose to file

    path = base_path + seq + 'body_pose.npy'
    np.save(path, body_pose)
    
    body_pose_mat = our_params['body_pose']
    body_pose_mat = torch.tensor(body_pose_mat, dtype=torch.float32)
    
    body_pose_mat = rotation_6d_to_matrix(
        body_pose_mat.reshape(-1, 6)).reshape(-1,23,3,3)
    
    
    transl = our_params['transl']
    
    # to tensor
    transl = torch.tensor(transl, dtype=torch.float32)
    
    # show difference
    print("Global Orient Difference: ", torch.norm(global_orient - pretrain_global_orient))
    print("Body Pose Difference: ", torch.norm(body_pose - pretrain_body_pose))
    print("Transl Difference: ", torch.norm(transl - pretrain_transl))
    
    
    for i in range(len(smpl_params['global_orient'])):

        print("index: ", i)
        print("img_id", train_split[i])
        
        # print val and diff
        # print("Global Orient: ", global_orient[i])
        # print("Pretrain Global Orient: ", pretrain_global_orient[i])
        print("Global Orient Difference: ", torch.norm(global_orient[i] - pretrain_global_orient[i]))
        
        # print("Body Pose: ", body_pose[i])
        # print("Pretrain Body Pose: ", pretrain_body_pose[i])
        print("Body Pose Difference: ", torch.norm(body_pose[i] - pretrain_body_pose[i]))
        
        # print('Transl: ', transl[i])
        # print('Pretrain Transl: ', pretrain_transl[i])
        print('Transl Difference: ', torch.norm(transl[i] - pretrain_transl[i]))
        
    # find max k num difference and index

    max_diff = torch.norm(global_orient - pretrain_global_orient, dim=1)

    max_diff_idx = torch.argmax(max_diff)




            
    # print("Max Diff: ", max_diff)
    # print("Max Diff Index: ", max_diff_idx)
    
    # print("Global Orient: ", global_orient[max_diff_idx])
    # print("Pretrain Global Orient: ", pretrain_global_orient[max_diff_idx])
    
    # print("Body Pose: ", body_pose[max_diff_idx])
    # print("Pretrain Body Pose: ", pretrain_body_pose[max_diff_idx])
    
    # print("Body Pose Mat: ", body_pose_mat[max_diff_idx])

    
    # print("Transl: ", transl[max_diff_idx])
    # print("Pretrain Transl: ", pretrain_transl[max_diff_idx])
    
    # # 计算最大插值的图片所属的原始视频帧id
    # print("Max Diff Index: ", train_split[max_diff_idx])
    
    
    # change body_pose_mat to numpy
    body_pose_mat = body_pose_mat.detach().numpy()
    
    # save the body_pose_mat to file
    path = base_path + seq + 'body_pose_mat.npy'
    np.save(path, body_pose_mat)
        
    # smpl_output = self.smpl(
    #     betas=betas.unsqueeze(0),
    #     body_pose=body_pose.unsqueeze(0),
    #     global_orient=global_orient.unsqueeze(0),
    #     disable_posedirs=False,
    #     return_full_pose=True,
    # )
    
    pass