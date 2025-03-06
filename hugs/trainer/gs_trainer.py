#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.
#

import time
import os
import glob
import shutil
import torch
import itertools
import torchvision
import numpy as np
from PIL import Image
from tqdm import tqdm
from lpips import LPIPS
from loguru import logger
import matplotlib.pyplot as plt
import torch.nn as nn

from hugs.datasets.utils import (
    get_rotating_camera,
    get_smpl_canon_params,
    get_smpl_static_params, 
    get_static_camera
)
from hugs.losses.utils import ssim
from hugs.datasets import NeumanDataset
from hugs.losses.loss import HumanSceneLoss
from hugs.models.hugs_trimlp import HUGS_TRIMLP
from hugs.models.hugs_wo_trimlp import HUGS_WO_TRIMLP
from hugs.models.clothGS import ClothGS
from hugs.models import SceneGS
from hugs.utils.init_opt import optimize_init
from hugs.renderer.gs_renderer import render_human_scene
from hugs.utils.vis import save_ply
from hugs.utils.image import psnr, save_image
from hugs.utils.general import RandomIndexIterator, load_human_ckpt, save_images, create_video
from hugs.utils.export_gs_to_uv import project_3dgs_with_covariance_torch


def get_train_dataset(cfg):
    if cfg.dataset.name == 'neuman':
        logger.info(f'Loading NeuMan dataset {cfg.dataset.seq}-train')
        dataset = NeumanDataset(
            cfg.dataset.seq, 'train', 
            render_mode=cfg.mode,
            add_bg_points=cfg.scene.add_bg_points,
            num_bg_points=cfg.scene.num_bg_points,
            bg_sphere_dist=cfg.scene.bg_sphere_dist,
            clean_pcd=cfg.scene.clean_pcd,
        )
    
    return dataset


def get_val_dataset(cfg):
    if cfg.dataset.name == 'neuman':
        logger.info(f'Loading NeuMan dataset {cfg.dataset.seq}-val')
        dataset = NeumanDataset(cfg.dataset.seq, 'val', cfg.mode)
   
    return dataset


def get_anim_dataset(cfg):
    if cfg.dataset.name == 'neuman':
        logger.info(f'Loading NeuMan dataset {cfg.dataset.seq}-anim')
        dataset = NeumanDataset(cfg.dataset.seq, 'anim', cfg.mode)
    elif cfg.dataset.name == 'zju':
        dataset = None
        
    return dataset


class GaussianTrainer():
    def __init__(self, cfg) -> None:
        self.cfg = cfg

        self.is_human_with_cloth_separate = True
        
        # get dataset
        if not cfg.eval:
            self.train_dataset = get_train_dataset(cfg)
        self.val_dataset = get_val_dataset(cfg)
        self.anim_dataset = get_anim_dataset(cfg)
        
        self.eval_metrics = {}
        self.lpips = LPIPS(net="alex", pretrained=True).to('cuda')
        # get models
        self.human_gs, self.scene_gs = None, None
        
        if cfg.mode in ['human', 'human_scene']:
            if cfg.human.name == 'hugs_wo_trimlp':
                self.human_gs = HUGS_WO_TRIMLP(
                    sh_degree=cfg.human.sh_degree, 
                    n_subdivision=cfg.human.n_subdivision,  
                    use_surface=cfg.human.use_surface,
                    init_2d=cfg.human.init_2d,
                    rotate_sh=cfg.human.rotate_sh,
                    isotropic=cfg.human.isotropic,
                    init_scale_multiplier=cfg.human.init_scale_multiplier,
                )
                init_betas = torch.stack([x['betas'] for x in self.train_dataset.cached_data], dim=0)
                self.human_gs.create_betas(init_betas[0], cfg.human.optim_betas)
                self.human_gs.initialize()
            elif cfg.human.name == 'hugs_trimlp':
                init_betas = torch.stack([x['betas'] for x in self.val_dataset.cached_data], dim=0)
                self.human_gs = HUGS_TRIMLP(
                    sh_degree=cfg.human.sh_degree, 
                    n_subdivision=0,
                    use_surface=cfg.human.use_surface,
                    init_2d=cfg.human.init_2d,
                    rotate_sh=cfg.human.rotate_sh,
                    isotropic=cfg.human.isotropic,
                    init_scale_multiplier=cfg.human.init_scale_multiplier,
                    n_features=32,
                    use_deformer=cfg.human.use_deformer,
                    disable_posedirs=cfg.human.disable_posedirs,
                    triplane_res=cfg.human.triplane_res,
                    betas=init_betas[0]
                )
                self.human_gs.create_betas(init_betas[0], cfg.human.optim_betas)
                if not cfg.eval:
                    self.human_gs.initialize()
                    self.human_gs = optimize_init(self.human_gs, num_steps=self.cfg.train.init_opimization_iters)
            self.upperbody_gs = ClothGS(
                    garment_class='t-shirt',
                    sh_degree=cfg.human.sh_degree,
                    n_subdivision=cfg.human.n_subdivision,
                    use_surface=cfg.human.use_surface,
                    init_2d=cfg.human.init_2d,
                    rotate_sh=cfg.human.rotate_sh,
                    isotropic=cfg.human.isotropic,
                    init_scale_multiplier=cfg.human.init_scale_multiplier,
                    n_features=32,
                    use_deformer=cfg.human.use_deformer,
                    disable_posedirs=cfg.human.disable_posedirs,
                    triplane_res=cfg.human.triplane_res,
                    betas=init_betas[0])
            if not cfg.eval:
                self.upperbody_gs.initialize()
                self.upperbody_gs = optimize_init(self.upperbody_gs, num_steps=self.cfg.train.init_opimization_iters)

            self.lowerbody_gs = ClothGS(
                    garment_class='pant',
                    sh_degree=cfg.human.sh_degree,
                    n_subdivision=cfg.human.n_subdivision,
                    use_surface=cfg.human.use_surface,
                    init_2d=cfg.human.init_2d,
                    rotate_sh=cfg.human.rotate_sh,
                    isotropic=cfg.human.isotropic,
                    init_scale_multiplier=cfg.human.init_scale_multiplier,
                    n_features=32,
                    use_deformer=cfg.human.use_deformer,
                    disable_posedirs=cfg.human.disable_posedirs,
                    triplane_res=cfg.human.triplane_res,
                    betas=init_betas[0])
            if not cfg.eval:
                self.lowerbody_gs.initialize()
                self.lowerbody_gs = optimize_init(self.lowerbody_gs, num_steps=self.cfg.train.init_opimization_iters)
        
        if cfg.mode in ['scene', 'human_scene']:
            self.scene_gs = SceneGS(
                sh_degree=cfg.scene.sh_degree,
            )
            
        # setup the optimizers
        if self.human_gs:
            self.human_gs.setup_optimizer(cfg=cfg.human.lr)
            logger.info(self.human_gs)
            if cfg.human.ckpt:
                # load_human_ckpt(self.human_gs, cfg.human.ckpt)
                self.human_gs.load_state_dict(torch.load(cfg.human.ckpt))
                logger.info(f'Loaded human model from {cfg.human.ckpt}')
            else:
                ckpt_files = sorted(glob.glob(f'{cfg.logdir_ckpt}/*human*.pth'))
                if len(ckpt_files) > 0:
                    ckpt = torch.load(ckpt_files[-1])
                    self.human_gs.load_state_dict(ckpt)
                    logger.info(f'Loaded human model from {ckpt_files[-1]}')

            if not cfg.eval:
                init_smpl_global_orient = torch.stack([x['global_orient'] for x in self.train_dataset.cached_data])
                init_smpl_body_pose = torch.stack([x['body_pose'] for x in self.train_dataset.cached_data])
                init_smpl_trans = torch.stack([x['transl'] for x in self.train_dataset.cached_data], dim=0)
                init_betas = torch.stack([x['betas'] for x in self.train_dataset.cached_data], dim=0)
                init_eps_offsets = torch.zeros((len(self.train_dataset), self.human_gs.n_gs, 3), 
                                            dtype=torch.float32, device="cuda")

                self.human_gs.create_betas(init_betas[0], cfg.human.optim_betas)
                
                self.human_gs.create_body_pose(init_smpl_body_pose, cfg.human.optim_pose)
                self.human_gs.create_global_orient(init_smpl_global_orient, cfg.human.optim_pose)
                self.human_gs.create_transl(init_smpl_trans, cfg.human.optim_trans)
                
                self.human_gs.setup_optimizer(cfg=cfg.human.lr)

        if self.upperbody_gs:
            self.upperbody_gs.setup_optimizer(cfg=cfg.human.lr)
            logger.info(self.upperbody_gs)
            if cfg.human.ckpt:
                # load_human_ckpt(self.human_gs, cfg.human.ckpt)
                self.upperbody_gs.load_state_dict(torch.load(cfg.human.ckpt))
                logger.info(f'Loaded human model from {cfg.human.ckpt}')
            else:
                ckpt_files = sorted(glob.glob(f'{cfg.logdir_ckpt}/*human*.pth'))
                if len(ckpt_files) > 0:
                    ckpt = torch.load(ckpt_files[-1])
                    self.upperbody_gs.load_state_dict(ckpt)
                    logger.info(f'Loaded human model from {ckpt_files[-1]}')

            if not cfg.eval:
                init_smpl_global_orient = torch.stack([x['global_orient'] for x in self.train_dataset.cached_data])
                init_smpl_body_pose = torch.stack([x['body_pose'] for x in self.train_dataset.cached_data])
                init_smpl_trans = torch.stack([x['transl'] for x in self.train_dataset.cached_data], dim=0)
                init_betas = torch.stack([x['betas'] for x in self.train_dataset.cached_data], dim=0)
                init_eps_offsets = torch.zeros((len(self.train_dataset), self.human_gs.n_gs, 3),
                                            dtype=torch.float32, device="cuda")

                self.upperbody_gs.create_betas(init_betas[0], cfg.human.optim_betas)

                self.upperbody_gs.create_body_pose(init_smpl_body_pose, cfg.human.optim_pose)
                self.upperbody_gs.create_global_orient(init_smpl_global_orient, cfg.human.optim_pose)
                self.upperbody_gs.create_transl(init_smpl_trans, cfg.human.optim_trans)

                self.upperbody_gs.setup_optimizer(cfg=cfg.human.lr)

        if self.lowerbody_gs:
            self.lowerbody_gs.setup_optimizer(cfg=cfg.human.lr)
            logger.info(self.lowerbody_gs)
            if cfg.human.ckpt:
                # load_human_ckpt(self.human_gs, cfg.human.ckpt)
                self.lowerbody_gs.load_state_dict(torch.load(cfg.human.ckpt))
                logger.info(f'Loaded human model from {cfg.human.ckpt}')
            else:
                ckpt_files = sorted(glob.glob(f'{cfg.logdir_ckpt}/*human*.pth'))
                if len(ckpt_files) > 0:
                    ckpt = torch.load(ckpt_files[-1])
                    self.lowerbody_gs.load_state_dict(ckpt)
                    logger.info(f'Loaded human model from {ckpt_files[-1]}')

            if not cfg.eval:
                init_smpl_global_orient = torch.stack([x['global_orient'] for x in self.train_dataset.cached_data])
                init_smpl_body_pose = torch.stack([x['body_pose'] for x in self.train_dataset.cached_data])
                init_smpl_trans = torch.stack([x['transl'] for x in self.train_dataset.cached_data], dim=0)
                init_betas = torch.stack([x['betas'] for x in self.train_dataset.cached_data], dim=0)
                init_eps_offsets = torch.zeros((len(self.train_dataset), self.human_gs.n_gs, 3),
                                            dtype=torch.float32, device="cuda")

                self.lowerbody_gs.create_betas(init_betas[0], cfg.human.optim_betas)

                self.lowerbody_gs.create_body_pose(init_smpl_body_pose, cfg.human.optim_pose)
                self.lowerbody_gs.create_global_orient(init_smpl_global_orient, cfg.human.optim_pose)
                self.lowerbody_gs.create_transl(init_smpl_trans, cfg.human.optim_trans)

                self.lowerbody_gs.setup_optimizer(cfg=cfg.human.lr)
                    
        if self.scene_gs:
            logger.info(self.scene_gs)
            if cfg.scene.ckpt:
                ckpt = torch.load(cfg.scene.ckpt)
                self.scene_gs.restore(ckpt, cfg.scene.lr)
                logger.info(f'Loaded scene model from {cfg.scene.ckpt}')
            else:
                ckpt_files = sorted(glob.glob(f'{cfg.logdir_ckpt}/*scene*.pth'))
                if len(ckpt_files) > 0:
                    ckpt = torch.load(ckpt_files[-1])
                    self.scene_gs.restore(ckpt, cfg.scene.lr)
                    logger.info(f'Loaded scene model from {cfg.scene.ckpt}')
                else:
                    pcd = self.train_dataset.init_pcd
                    spatial_lr_scale = self.train_dataset.radius
                    self.scene_gs.create_from_pcd(pcd, spatial_lr_scale)
                
            self.scene_gs.setup_optimizer(cfg=cfg.scene.lr)
        
        bg_color = cfg.bg_color
        if bg_color == 'white':
            self.bg_color = torch.tensor([1, 1, 1], dtype=torch.float32, device="cuda")
        elif bg_color == 'black':
            self.bg_color = torch.tensor([0, 0, 0], dtype=torch.float32, device="cuda")
        else:
            raise ValueError(f"Unknown background color {bg_color}")
        
        if cfg.mode in ['human', 'human_scene']:
            l = cfg.human.loss




            self.loss_fn = HumanSceneLoss(
                l_ssim_w=l.ssim_w,
                l_l1_w=l.l1_w,
                l_lpips_w=l.lpips_w,
                l_lbs_w=l.lbs_w,
                l_humansep_w=l.humansep_w,
                l_clothsep_w=l.humansep_w, # FIX ME
                num_patches=l.num_patches,
                patch_size=l.patch_size,
                use_patches=l.use_patches,
                bg_color=self.bg_color,
            )
        else:
            self.cfg.train.optim_scene = True
            l = cfg.scene.loss
            self.loss_fn = HumanSceneLoss(
                l_ssim_w=l.ssim_w,
                l_l1_w=l.l1_w,
                bg_color=self.bg_color,
            )
                
        if cfg.mode in ['human', 'human_scene']:
            self.canon_camera_params = get_rotating_camera(
                dist=5.0, img_size=512, 
                nframes=cfg.human.canon_nframes, device='cuda',
                angle_limit=2*torch.pi,
            )
            betas = self.human_gs.betas.detach() if hasattr(self.human_gs, 'betas') else self.train_dataset.betas[0]
            self.static_smpl_params = get_smpl_static_params(
                betas=betas,
                pose_type=self.cfg.human.canon_pose_type
            )

    def train(self):
        if self.human_gs:
            self.human_gs.train()

        pbar = tqdm(range(self.cfg.train.num_steps+1), desc="Training" ,dynamic_ncols=True)
        
        rand_idx_iter = RandomIndexIterator(len(self.train_dataset))
        sgrad_means, sgrad_stds = [], []
        for t_iter in range(self.cfg.train.num_steps+1):
            render_mode = self.cfg.mode
            
            if self.scene_gs and self.cfg.train.optim_scene:
                self.scene_gs.update_learning_rate(t_iter)
            
            if hasattr(self.human_gs, 'update_learning_rate'):
                self.human_gs.update_learning_rate(t_iter)
        
            rnd_idx = next(rand_idx_iter)
            data = self.train_dataset[rnd_idx]
            
            human_gs_out, upperbody_gs_out,lowerbody_gs_out, scene_gs_out = None, None, None, None
            
            if self.human_gs:
                human_gs_out = self.human_gs.forward(
                    smpl_scale=data['smpl_scale'][None],
                    dataset_idx=rnd_idx,
                    is_train=True,
                    ext_tfs=None,
                )
            if self.upperbody_gs:
                upperbody_gs_out = self.upperbody_gs.forward(
                    smpl_scale=data['smpl_scale'][None],
                    dataset_idx=rnd_idx,
                    is_train=True,
                    ext_tfs=None,
                )
            if self.lowerbody_gs:
                lowerbody_gs_out = self.lowerbody_gs.forward(
                    smpl_scale=data['smpl_scale'][None],
                    dataset_idx=rnd_idx,
                    is_train=True,
                    ext_tfs=None,
                )
            
            if self.scene_gs:
                if t_iter >= self.cfg.scene.opt_start_iter:
                    scene_gs_out = self.scene_gs.forward()
                else:
                    render_mode = 'human'
            
            bg_color = torch.rand(3, dtype=torch.float32, device="cuda")
            
            
            if self.cfg.human.loss.humansep_w > 0.0 and render_mode == 'human_scene':
                render_human_separate = True
                human_bg_color = torch.rand(3, dtype=torch.float32, device="cuda")
                cloth_bg_color = torch.rand(3, dtype=torch.float32, device="cuda")
            else:
                human_bg_color = None
                cloth_bg_color = None
                render_human_separate = False
            
            render_pkg = render_human_scene(
                data=data, 
                human_gs_out=human_gs_out,
                upperbody_gs_out=upperbody_gs_out,
                lowerbody_gs_out=lowerbody_gs_out,
                scene_gs_out=scene_gs_out, 
                bg_color=bg_color,
                human_bg_color=human_bg_color,
                cloth_bg_color=cloth_bg_color,
                render_mode=render_mode,
                render_human_separate=render_human_separate,
                render_cloth_separate=self.is_human_with_cloth_separate,

            )
            
            if self.human_gs:
                self.human_gs.init_values['edges'] = self.human_gs.edges
                        
            loss, loss_dict, loss_extras = self.loss_fn(
                data,
                render_pkg,
                human_gs_out,
                render_mode=render_mode,
                human_gs_init_values=self.human_gs.init_values if self.human_gs else None,
                bg_color=bg_color,
                human_bg_color=human_bg_color,
                upperbody_gs_out=upperbody_gs_out,
                upperbody_gs_init_values=self.upperbody_gs.init_values if self.upperbody_gs else None,
                lowerbody_gs_out=lowerbody_gs_out,
                lowerbody_gs_init_values=self.lowerbody_gs.init_values if self.lowerbody_gs else None,
                cloth_bg_color=cloth_bg_color,
                is_human_with_cloth_seprate=self.is_human_with_cloth_separate,
            )
            
            loss.backward()
            
            loss_dict['loss'] = loss

            if t_iter % 10 == 0:
                postfix_dict = {
                    "#hp": f"{self.human_gs.n_gs/1000 if self.human_gs else 0:.1f}K",
                    # "#sp": f"{self.scene_gs.get_xyz.shape[0]/1000 if self.scene_gs else 0:.1f}K",
                    # 'h_sh_d': self.human_gs.active_sh_degree if self.human_gs else 0,
                    # 's_sh_d': self.scene_gs.active_sh_degree if self.scene_gs else 0,
                }
                for k, v in loss_dict.items():
                    #if k == 'loss':
                    postfix_dict["l_" + k] = f"{v.item():.4f}"
                

                pbar.set_postfix(postfix_dict)
                pbar.update(10)
            
            if t_iter % 500 == 0:
                postfix_dict = {
                    # "#hp": f"{self.human_gs.n_gs/1000 if self.human_gs else 0:.1f}K",
                    # "#sp": f"{self.scene_gs.get_xyz.shape[0]/1000 if self.scene_gs else 0:.1f}K",
                    # 'h_sh_d': self.human_gs.active_sh_degree if self.human_gs else 0,
                    # 's_sh_d': self.scene_gs.active_sh_degree if self.scene_gs else 0,
                }
                for k, v in loss_dict.items():
                    postfix_dict["l_"+k] = f"{v.item():.4f}"
                        
                pbar.set_postfix(postfix_dict)
                pbar.update(10)
                
            if t_iter == self.cfg.train.num_steps:
                pbar.close()

            if t_iter % 1000 == 0:
                with torch.no_grad():
                    pred_img = loss_extras['pred_img']
                    gt_img = loss_extras['gt_img']
                    log_pred_img = (pred_img.cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
                    log_gt_img = (gt_img.cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
                    log_img = np.concatenate([log_gt_img, log_pred_img], axis=1)
                    save_images(log_img, f'{self.cfg.logdir}/train/{t_iter:06d}.png')


            export_smpl_uv_color = True 
            export_smpl_uv_color_interval = 400
            if export_smpl_uv_color is True and t_iter != 0 and t_iter % export_smpl_uv_color_interval == 0:
                smpl_mesh_path = './assets/template_mesh_smpl_uv.obj'
                feats = upperbody_gs_out['shs']
                xyz = self.upperbody_gs._xyz
                means3D = upperbody_gs_out['xyz']
                opacity = upperbody_gs_out['opacity']
                scales = upperbody_gs_out['scales']
                rotations = upperbody_gs_out['rotq']
                active_sh_degree = upperbody_gs_out['active_sh_degree']
                
                uv_image = project_3dgs_with_covariance_torch(
                    smpl_mesh_path=smpl_mesh_path,
                    gs_positions=xyz,
                    gs_rotations=rotations,
                    gs_scales=scales,
                    gs_sh_coeffs=feats,
                    gs_opacity=opacity,
                    device='cuda'
                )
                SAVE_PATH =  self.cfg.train.smpl_uv_result_path
                
                # 4. 验证UV图是否包含非空像素
                non_zero_pixels = np.any(uv_image > 0, axis=-1).sum()
                print(f"非空像素数量: {non_zero_pixels}")
                # assert non_zero_pixels >= len(test_vertex_indices), \
                #     f"至少应有{len(test_vertex_indices)}个高斯点的颜色被投影到UV图"
                
                # 5. 可选：保存UV图供肉眼检查
                import cv2
                # cv2.imwrite(SAVE_PATH+"test_uv_output.png", cv2.cvtColor(uv_image, cv2.COLOR_RGB2BGR))
                name = f'{SAVE_PATH}test_uv_output_{t_iter:06d}.png'
                cv2.imwrite(name, cv2.cvtColor(uv_image, cv2.COLOR_RGB2BGR))
                print("测试通过！生成的UV图已保存为 %s" % name)
            
            
            if t_iter % self.cfg.train.gs_save_to_disk_interval == 0 and t_iter != 0:
                save_dict = {
                    'gs_positions': upperbody_gs_out['xyz_canon'],
                    'gs_rotations': upperbody_gs_out['rotq_canon'],
                    'gs_scales': upperbody_gs_out['scales_canon'],
                    'gs_sh_coeffs': upperbody_gs_out['shs'],
                    'gs_opacity': upperbody_gs_out['opacity'],
                    'gs_active_sh_degree': upperbody_gs_out['active_sh_degree'],
                }
                # check if the debug base path exists
                if not os.path.exists(self.cfg.train.results_base_path):
                    os.makedirs(self.cfg.train.results_base_path)
                    
                if t_iter == 1000:
                    pass
                # check if the directory exists
                if not os.path.exists(self.cfg.train.gs_save_to_disk_path):
                    os.makedirs(self.cfg.train.gs_save_to_disk_path)
                
                torch.save(save_dict, f'{self.cfg.train.gs_save_to_disk_path}/gs_{t_iter:06d}.pth')
                
                if self.upperbody_gs:
                    save_ply(upperbody_gs_out, f'{self.cfg.train.gs_save_to_disk_path}/upperbody_{t_iter:06d}_splat.ply')

            save_result_img_interval = 100
            if t_iter % save_result_img_interval == 0:
                # show pred_img and gt_img by plt in one window
                pred_img = loss_extras['pred_img']
                gt_img = loss_extras['gt_img']

                # 可视化差异
                diff_image = torch.abs(pred_img - gt_img)
                diff_image = torch.mean(diff_image, dim=0)
                diff_image = diff_image.unsqueeze(0)
                # 变回rgb通道
                diff_image = diff_image.repeat(3, 1, 1)

                mask = data['mask'].unsqueeze(0)
                gt_human_with_colth_image = gt_img * mask + human_bg_color[:, None, None] * (1. - mask)

                need_show_window = False
                if self.is_human_with_cloth_separate is False:
                    pred_img = torchvision.transforms.ToPILImage()(pred_img)
                    gt_img = torchvision.transforms.ToPILImage()(gt_img)
                    diff_image = torchvision.transforms.ToPILImage()(diff_image)

                    human_with_cloth_image = render_pkg['human_with_cloth_img']
                    diff_human_with_cloth = torch.abs(human_with_cloth_image - gt_human_with_colth_image)
                    human_with_cloth_mse_loss = torch.mean(diff_human_with_cloth, dim=0)
                    diff_human_with_cloth_image = human_with_cloth_mse_loss.unsqueeze(0)

                    human_with_cloth_image = torchvision.transforms.ToPILImage()(human_with_cloth_image)
                    gt_human_with_colth_image = torchvision.transforms.ToPILImage()(gt_human_with_colth_image)
                    diff_human_with_cloth_image = torchvision.transforms.ToPILImage()(diff_human_with_cloth_image)

                    plt.figure(dpi=108, figsize=(24, 24))
                    #add title for iteration
                    plt.suptitle(f'Iteration: {t_iter}', fontsize=16)

                    # add loss value for each iteration
                    plt.text(0.5, 0.5, f'Loss: {loss.item():.4f}', fontsize=16, ha='center')
                    # add time for each iteration

                    plt.text(0.5, 0.4, f'Time: {time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())}', fontsize=16, ha='center')
                    plt.subplot(231)
                    plt.imshow(pred_img)
                    plt.axis('off')
                    plt.title("Rendered Image")
                    plt.subplot(232)
                    plt.imshow(gt_img)
                    plt.axis('off')
                    plt.title("Ground Truth")
                    plt.subplot(233)
                    plt.imshow(diff_image)
                    plt.axis('off')
                    plt.title("Difference")

                    plt.subplot(234)
                    plt.imshow(human_with_cloth_image)
                    plt.axis('off')
                    plt.title("Human with Cloth Image")
                    plt.subplot(235)
                    plt.imshow(gt_human_with_colth_image)
                    plt.axis('off')
                    plt.title("Human with Cloth Ground Truth")
                    plt.subplot(236)
                    plt.imshow(diff_human_with_cloth_image)
                    

                if self.is_human_with_cloth_separate:

                    humanbody_mask = data['humanbody_mask'].unsqueeze(0)
                    gt_human_image = gt_img * humanbody_mask + human_bg_color[:, None, None] * (1. - humanbody_mask)

                    upperbody_mask = data['upperbody_mask']
                    # cloth area in upperbody_mask is r = 1, g = 0, b = 0,get the cloth area image
                    # cloth_area = torch.zeros_like(gt_img)
                    # cloth_area[upperbody_mask == 0] = gt_img[upperbody_mask == 0]
                    # cloth_area = torchvision.transforms.ToPILImage()(cloth_area)
                    gt_upperbody_img = gt_img * upperbody_mask + cloth_bg_color[:, None, None] * (1. - upperbody_mask)

                    upperbody_img = render_pkg['upperbody_img']

                    upperbody_diff = torch.abs(upperbody_img - gt_upperbody_img)
                    upperbody_mse_loss = torch.mean(upperbody_diff, dim=0)
                    upperbody_diff_image = upperbody_mse_loss.unsqueeze(0)
                    upperbody_diff_image = upperbody_diff_image.repeat(3, 1, 1)

                    lowerbody_mask = data['lowerbody_mask']
                    gt_lowerbody_img = gt_img * lowerbody_mask + cloth_bg_color[:, None, None] * (1. - lowerbody_mask)

                    lowerbody_img = render_pkg['lowerbody_img']

                    lowerbody_diff = torch.abs(lowerbody_img - gt_lowerbody_img)
                    lowerbody_mse_loss = torch.mean(lowerbody_diff, dim=0)
                    lowerbody_diff_image = lowerbody_mse_loss.unsqueeze(0)
                    lowerbody_diff_image = lowerbody_diff_image.repeat(3, 1, 1)


                    human_img = render_pkg['human_img']

                    human_diff = torch.abs(human_img - gt_human_image)
                    human_mse_loss = torch.mean(human_diff, dim=0)
                    human_diff_image = human_mse_loss.unsqueeze(0)
                    human_diff_image = human_diff_image.repeat(3, 1, 1)

                    pred_img = torchvision.transforms.ToPILImage()(pred_img)
                    gt_img = torchvision.transforms.ToPILImage()(gt_img)

                    gt_upperbody_img = torchvision.transforms.ToPILImage()(gt_upperbody_img)
                    upperbody_img = torchvision.transforms.ToPILImage()(upperbody_img)

                    gt_lowerbody_img = torchvision.transforms.ToPILImage()(gt_lowerbody_img)
                    lowerbody_img = torchvision.transforms.ToPILImage()(lowerbody_img)
                    lowerbody_diff_image_cpu = torchvision.transforms.ToPILImage()(lowerbody_diff_image)

                    human_img = torchvision.transforms.ToPILImage()(human_img)
                    gt_human_image = torchvision.transforms.ToPILImage()(gt_human_image)

                    diff_image_cpu = torchvision.transforms.ToPILImage()(diff_image)
                    upperbody_diff_image_cpu = torchvision.transforms.ToPILImage()(upperbody_diff_image)
                    human_diff_image_cpu = torchvision.transforms.ToPILImage()(human_diff_image)

                    # show three image in GUI every 100 iterations

                    plt.figure(dpi=108, figsize=(24, 24))

                    # add title for iteration
                    plt.suptitle(f'Iteration: {t_iter}', fontsize=16)

                    # add loss value for each iteration
                    plt.text(0.5, 0.5, f'Loss: {loss.item():.4f}', fontsize=16, ha='center')
                    # add time for each iteration

                    plt.text(0.5, 0.4, f'Time: {time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())}', fontsize=16, ha='center')



                    plt.subplot(431)
                    plt.imshow(pred_img)
                    plt.axis('off')
                    plt.title("Rendered Image")
                    plt.subplot(432)
                    plt.imshow(gt_img)
                    plt.axis('off')
                    plt.title("Ground Truth")
                    plt.subplot(433)
                    plt.imshow(diff_image_cpu)
                    plt.axis('off')
                    plt.title("Difference")

                    plt.subplot(434)
                    plt.imshow(human_img)
                    plt.axis('off')
                    plt.title("Human Image")
                    plt.subplot(435)
                    plt.imshow(gt_human_image)
                    plt.axis('off')
                    plt.title("Human Image Ground Truth")
                    plt.subplot(436)
                    plt.imshow(human_diff_image_cpu)
                    plt.axis('off')
                    plt.title("Human Difference")

                    plt.subplot(437)
                    plt.imshow(upperbody_img)
                    plt.axis('off')
                    plt.title("Upperbody Image")
                    plt.subplot(438)
                    plt.imshow(gt_upperbody_img)
                    plt.axis('off')
                    plt.title("Upperbody Ground Truth")
                    plt.subplot(439)
                    plt.imshow(upperbody_diff_image_cpu)
                    plt.axis('off')
                    plt.title("Upperbody Difference")

                    plt.subplot(4, 3, 10)
                    plt.imshow(lowerbody_img)
                    plt.axis('off')
                    plt.title("Lowerbody Image")
                    plt.subplot(4, 3, 11)
                    plt.imshow(gt_lowerbody_img)
                    plt.axis('off')
                    plt.title("Lowerbody Ground Truth")
                    plt.subplot(4, 3, 12)
                    plt.imshow(lowerbody_diff_image_cpu)
                    plt.axis('off')
                    plt.title("Lowerbody Difference")

                path = self.cfg.train.train_process_img_path
                
                # check if the directory exists
                if not os.path.exists(path):
                    os.makedirs(path)
                
                plt.savefig(path + f'{t_iter:06d}.png')

                if need_show_window:
                    plt.show()
                plt.close()

            if t_iter >= self.cfg.scene.opt_start_iter:
                if (t_iter - self.cfg.scene.opt_start_iter) < self.cfg.scene.densify_until_iter and self.cfg.mode in ['scene', 'human_scene']:
                    render_pkg['scene_viewspace_points'] = render_pkg['viewspace_points']
                    render_pkg['scene_viewspace_points'].grad = render_pkg['viewspace_points'].grad
                        
                    sgrad_mean, sgrad_std = render_pkg['scene_viewspace_points'].grad.mean(), render_pkg['scene_viewspace_points'].grad.std()
                    sgrad_means.append(sgrad_mean.item())
                    sgrad_stds.append(sgrad_std.item())
                    with torch.no_grad():
                        self.scene_densification(
                            visibility_filter=render_pkg['scene_visibility_filter'],
                            radii=render_pkg['scene_radii'],
                            viewspace_point_tensor=render_pkg['scene_viewspace_points'],
                            iteration=(t_iter - self.cfg.scene.opt_start_iter) + 1,
                        )
                        
            if self.is_human_with_cloth_separate is True and t_iter < self.cfg.human.densify_until_iter and self.cfg.mode in ['human', 'human_scene']:
                render_pkg['human_viewspace_points'] = render_pkg['viewspace_points'][:human_gs_out['xyz'].shape[0]]
                render_pkg['human_viewspace_points'].grad = render_pkg['viewspace_points'].grad[:human_gs_out['xyz'].shape[0]]
                with torch.no_grad():
                    self.human_densification(
                        human_gs_out=human_gs_out,
                        visibility_filter=render_pkg['human_visibility_filter'],
                        radii=render_pkg['human_radii'],
                        viewspace_point_tensor=render_pkg['human_viewspace_points'],
                        iteration=t_iter+1,
                    )
                    
            if self.is_human_with_cloth_separate is True and t_iter < self.cfg.human.densify_until_iter and self.cfg.mode in ['human', 'human_scene']:
                with torch.no_grad():
                    self.upperbody_densification(
                        upperbody_gs_out=upperbody_gs_out,
                        visibility_filter=render_pkg['upperbody_visibility_filter'],
                        radii=render_pkg['upperbody_radii'],
                        viewspace_point_tensor=render_pkg['upperbody_viewspace_points'],
                        iteration=t_iter+1,
                    )
                    
            if self.is_human_with_cloth_separate is True and t_iter < self.cfg.human.densify_until_iter and self.cfg.mode in ['human', 'human_scene']:
                with torch.no_grad():
                    self.lowerbody_densification(
                        lowerbody_gs_out=lowerbody_gs_out,
                        visibility_filter=render_pkg['lowerbody_visibility_filter'],
                        radii=render_pkg['lowerbody_radii'],
                        viewspace_point_tensor=render_pkg['lowerbody_viewspace_points'],
                        iteration=t_iter+1,
                    )
            
            if self.human_gs:
                self.human_gs.optimizer.step()
                self.human_gs.optimizer.zero_grad(set_to_none=True)

            if self.upperbody_gs:
                self.upperbody_gs.optimizer.step()
                self.upperbody_gs.optimizer.zero_grad(set_to_none=True)

            if self.lowerbody_gs:
                self.lowerbody_gs.optimizer.step()
                self.lowerbody_gs.optimizer.zero_grad(set_to_none=True)
                
            if self.scene_gs and self.cfg.train.optim_scene:
                if t_iter >= self.cfg.scene.opt_start_iter:
                    self.scene_gs.optimizer.step()
                    self.scene_gs.optimizer.zero_grad(set_to_none=True)
                
            # save checkpoint
            if (t_iter % self.cfg.train.save_ckpt_interval == 0 and t_iter > 0) or \
                (t_iter == self.cfg.train.num_steps and t_iter > 0):
                self.save_ckpt(t_iter)

            # run validation
            if t_iter % self.cfg.train.val_interval == 0 and t_iter > 0:
                self.validate(t_iter)
            
            if t_iter == 0 or t_iter % 1000 == 0:
                if self.scene_gs:
                    self.scene_gs.save_ply(f'{self.cfg.logdir}/meshes/scene_{t_iter:06d}_splat.ply')
                if self.human_gs:
                    save_ply(human_gs_out, f'{self.cfg.logdir}/meshes/human_{t_iter:06d}_splat.ply')
                if self.upperbody_gs:
                    save_ply(upperbody_gs_out, f'{self.cfg.logdir}/meshes/upperbody_{t_iter:06d}_splat.ply')
                if self.lowerbody_gs:
                    save_ply(lowerbody_gs_out, f'{self.cfg.logdir}/meshes/lowerbody_{t_iter:06d}_splat.ply')


                if self.cfg.mode in ['human', 'human_scene']:
                    self.render_canonical(t_iter, nframes=self.cfg.human.canon_nframes)

            if t_iter % self.cfg.train.anim_interval == 0 and t_iter > 0 and self.cfg.train.anim_interval > 0:
                if self.human_gs:
                    save_ply(human_gs_out, f'{self.cfg.logdir}/meshes/human_{t_iter:06d}_splat.ply')
                if self.anim_dataset is not None:
                    self.animate(t_iter)
                    
                if self.cfg.mode in ['human', 'human_scene']:
                    self.render_canonical(t_iter, nframes=self.cfg.human.canon_nframes)
            
            if t_iter % 1000 == 0 and t_iter > 0:
                if self.human_gs: self.human_gs.oneupSHdegree()
                if self.scene_gs: self.scene_gs.oneupSHdegree()
                
            if self.cfg.train.save_progress_images and t_iter % self.cfg.train.progress_save_interval == 0 and self.cfg.mode in ['human', 'human_scene']:
                self.render_canonical(t_iter, nframes=2, is_train_progress=True)
        
        # train progress images
        if self.cfg.train.save_progress_images:
            video_fname = f'{self.cfg.logdir}/train_{self.cfg.dataset.name}_{self.cfg.dataset.seq}.mp4'
            create_video(f'{self.cfg.logdir}/train_progress/', video_fname, fps=10)
            shutil.rmtree(f'{self.cfg.logdir}/train_progress/')
            
    def save_ckpt(self, iter=None):
        
        iter_s = 'final' if iter is None else f'{iter:06d}'
        
        if self.human_gs:
            torch.save(self.human_gs.state_dict(), f'{self.cfg.logdir_ckpt}/human_{iter_s}.pth')
            
        if self.scene_gs:
            torch.save(self.scene_gs.state_dict(), f'{self.cfg.logdir_ckpt}/scene_{iter_s}.pth')
            self.scene_gs.save_ply(f'{self.cfg.logdir}/meshes/scene_{iter_s}_splat.ply')
            
        logger.info(f'Saved checkpoint {iter_s}')
                
    def scene_densification(self, visibility_filter, radii, viewspace_point_tensor, iteration):
        self.scene_gs.max_radii2D[visibility_filter] = torch.max(
            self.scene_gs.max_radii2D[visibility_filter], 
            radii[visibility_filter]
        )
        self.scene_gs.add_densification_stats(viewspace_point_tensor, visibility_filter)

        if iteration > self.cfg.scene.densify_from_iter and iteration % self.cfg.scene.densification_interval == 0:
            size_threshold = 20 if iteration > self.cfg.scene.opacity_reset_interval else None
            self.scene_gs.densify_and_prune(
                self.cfg.scene.densify_grad_threshold, 
                min_opacity=self.cfg.scene.prune_min_opacity, 
                extent=self.train_dataset.radius, 
                max_screen_size=size_threshold,
                max_n_gs=self.cfg.scene.max_n_gaussians,
            )
        
        is_white = self.bg_color.sum().item() == 3.
        
        if iteration % self.cfg.scene.opacity_reset_interval == 0 or (is_white and iteration == self.cfg.scene.densify_from_iter):
            logger.info(f"[{iteration:06d}] Resetting opacity!!!")
            self.scene_gs.reset_opacity()
    
    def human_densification(self, human_gs_out, visibility_filter, radii, viewspace_point_tensor, iteration):
        self.human_gs.max_radii2D[visibility_filter] = torch.max(
            self.human_gs.max_radii2D[visibility_filter], 
            radii[visibility_filter]
        )
        
        self.human_gs.add_densification_stats(viewspace_point_tensor, visibility_filter)

        if iteration > self.cfg.human.densify_from_iter and iteration % self.cfg.human.densification_interval == 0:
            size_threshold = 20
            self.human_gs.densify_and_prune(
                human_gs_out,
                self.cfg.human.densify_grad_threshold, 
                min_opacity=self.cfg.human.prune_min_opacity, 
                extent=self.cfg.human.densify_extent, 
                max_screen_size=size_threshold,
                max_n_gs=self.cfg.human.max_n_gaussians,
            )
            
    def upperbody_densification(self, upperbody_gs_out, visibility_filter, radii, viewspace_point_tensor, iteration):
        self.upperbody_gs.max_radii2D[visibility_filter] = torch.max(
            self.upperbody_gs.max_radii2D[visibility_filter], 
            radii[visibility_filter]
        )
        
        self.upperbody_gs.add_densification_stats(viewspace_point_tensor, visibility_filter)

        if iteration > self.cfg.upperbody.densify_from_iter and iteration % self.cfg.upperbody.densification_interval == 0:
            size_threshold = 20
            self.upperbody_gs.densify_and_prune(
                upperbody_gs_out,
                self.cfg.upperbody.densify_grad_threshold, 
                min_opacity=self.cfg.upperbody.prune_min_opacity, 
                extent=self.cfg.upperbody.densify_extent, 
                max_screen_size=size_threshold,
                max_n_gs=self.cfg.upperbody.max_n_gaussians,
            )
    def lowerbody_densification(self, lowerbody_gs_out, visibility_filter, radii, viewspace_point_tensor, iteration):
        self.lowerbody_gs.max_radii2D[visibility_filter] = torch.max(
            self.lowerbody_gs.max_radii2D[visibility_filter], 
            radii[visibility_filter]
        )
        
        self.lowerbody_gs.add_densification_stats(viewspace_point_tensor, visibility_filter)

        if iteration > self.cfg.lowerbody.densify_from_iter and iteration % self.cfg.lowerbody.densification_interval == 0:
            size_threshold = 20
            self.lowerbody_gs.densify_and_prune(
                lowerbody_gs_out,
                self.cfg.lowerbody.densify_grad_threshold, 
                min_opacity=self.cfg.lowerbody.prune_min_opacity, 
                extent=self.cfg.lowerbody.densify_extent, 
                max_screen_size=size_threshold,
                max_n_gs=self.cfg.lowerbody.max_n_gaussians,
            )
    
    @torch.no_grad()
    def validate(self, iter=None):
        
        iter_s = 'final' if iter is None else f'{iter:06d}'
        
        bg_color = torch.zeros(3, dtype=torch.float32, device="cuda")
        
        if self.human_gs:
            self.human_gs.eval()
                
        methods = ['hugs', 'hugs_human']
        metrics = ['lpips', 'psnr', 'ssim']
        metrics = dict.fromkeys(['_'.join(x) for x in itertools.product(methods, metrics)])
        metrics = {k: [] for k in metrics}
        
        for idx, data in enumerate(tqdm(self.val_dataset, desc="Validation")):
            human_gs_out,upperbody_gs_out,lowerbody_gs_out, scene_gs_out = None,None,None, None
            render_mode = self.cfg.mode
            
            if self.human_gs:
                human_gs_out = self.human_gs.forward(
                    global_orient=data['global_orient'], 
                    body_pose=data['body_pose'], 
                    betas=data['betas'], 
                    transl=data['transl'], 
                    smpl_scale=data['smpl_scale'][None],
                    dataset_idx=-1,
                    is_train=False,
                    ext_tfs=None,
                )

            if self.upperbody_gs:
                upperbody_gs_out = self.upperbody_gs.forward(
                    global_orient=data['global_orient'],
                    body_pose=data['body_pose'],
                    betas=data['betas'],
                    transl=data['transl'],
                    smpl_scale=data['smpl_scale'][None],
                    dataset_idx=-1,
                    is_train=False,
                    ext_tfs=None,
                )
            if self.lowerbody_gs:
                lowerbody_gs_out = self.lowerbody_gs.forward(
                    global_orient=data['global_orient'],
                    body_pose=data['body_pose'],
                    betas=data['betas'],
                    transl=data['transl'],
                    smpl_scale=data['smpl_scale'][None],
                    dataset_idx=-1,
                    is_train=False,
                    ext_tfs=None,
                )
                
            if self.scene_gs:
                if iter is not None:
                    if iter >= self.cfg.scene.opt_start_iter:
                        scene_gs_out = self.scene_gs.forward()
                    else:
                        render_mode = 'human'
                else:
                    scene_gs_out = self.scene_gs.forward()
                    
            render_pkg = render_human_scene(
                data=data, 
                human_gs_out=human_gs_out,
                upperbody_gs_out=upperbody_gs_out,
                lowerbody_gs_out=lowerbody_gs_out,
                scene_gs_out=scene_gs_out,
                bg_color=bg_color,
                render_mode=render_mode,
            )
            
            gt_image = data['rgb']
            
            image = render_pkg["render"]
            if self.cfg.dataset.name == 'zju':
                image = image * data['mask']
                gt_image = gt_image * data['mask']
            
            metrics['hugs_psnr'].append(psnr(image, gt_image).mean().double())
            metrics['hugs_ssim'].append(ssim(image, gt_image).mean().double())
            metrics['hugs_lpips'].append(self.lpips(image.clip(max=1), gt_image).mean().double())
            
            log_img = torchvision.utils.make_grid([gt_image, image], nrow=2, pad_value=1)
            imf = f'{self.cfg.logdir}/val/full_{iter_s}_{idx:03d}.png'
            os.makedirs(os.path.dirname(imf), exist_ok=True)
            torchvision.utils.save_image(log_img, imf)
            
            log_img = []
            if self.cfg.mode in ['human', 'human_scene']:
                bbox = data['bbox'].to(int)
                cropped_gt_image = gt_image[:, bbox[0]:bbox[2], bbox[1]:bbox[3]]
                cropped_image = image[:, bbox[0]:bbox[2], bbox[1]:bbox[3]]
                log_img += [cropped_gt_image, cropped_image]
                
                metrics['hugs_human_psnr'].append(psnr(cropped_image, cropped_gt_image).mean().double())
                metrics['hugs_human_ssim'].append(ssim(cropped_image, cropped_gt_image).mean().double())
                metrics['hugs_human_lpips'].append(self.lpips(cropped_image.clip(max=1), cropped_gt_image).mean().double())
            
            if len(log_img) > 0:
                log_img = torchvision.utils.make_grid(log_img, nrow=len(log_img), pad_value=1)
                torchvision.utils.save_image(log_img, f'{self.cfg.logdir}/val/human_{iter_s}_{idx:03d}.png')
        
        
        self.eval_metrics[iter_s] = {}
        
        for k, v in metrics.items():
            if v == []:
                continue
            
            logger.info(f"{iter_s} - {k.upper()}: {torch.stack(v).mean().item():.4f}")
            self.eval_metrics[iter_s][k] = torch.stack(v).mean().item()
        
        torch.save(metrics, f'{self.cfg.logdir}/val/eval_{iter_s}.pth')
    
    @torch.no_grad()
    def animate(self, iter=None, keep_images=False):
        if self.anim_dataset is None:
            logger.info("No animation dataset found")
            return 0
        
        iter_s = 'final' if iter is None else f'{iter:06d}'
        if self.human_gs:
            self.human_gs.eval()
        
        os.makedirs(f'{self.cfg.logdir}/anim/', exist_ok=True)
        
        for idx, data in enumerate(tqdm(self.anim_dataset, desc="Animation")):
            human_gs_out, scene_gs_out = None, None
            
            if self.human_gs:
                ext_tfs = (data['manual_trans'], data['manual_rotmat'], data['manual_scale'])
                human_gs_out = self.human_gs.forward(
                    global_orient=data['global_orient'],
                    body_pose=data['body_pose'],
                    betas=data['betas'],
                    transl=data['transl'],
                    smpl_scale=data['smpl_scale'][None],
                    dataset_idx=-1,
                    is_train=False,
                    ext_tfs=ext_tfs,
                )
            if self.upperbody_gs:
                ext_tfs = (data['manual_trans'], data['manual_rotmat'], data['manual_scale'])
                upperbody_gs_out = self.upperbody_gs.forward(
                    global_orient=data['global_orient'],
                    body_pose=data['body_pose'],
                    betas=data['betas'],
                    transl=data['transl'],
                    smpl_scale=data['smpl_scale'][None],
                    dataset_idx=-1,
                    is_train=False,
                    ext_tfs=ext_tfs,
                )

            if self.lowerbody_gs:
                ext_tfs = (data['manual_trans'], data['manual_rotmat'], data['manual_scale'])
                lowerbody_gs_out = self.lowerbody_gs.forward(
                    global_orient=data['global_orient'],
                    body_pose=data['body_pose'],
                    betas=data['betas'],
                    transl=data['transl'],
                    smpl_scale=data['smpl_scale'][None],
                    dataset_idx=-1,
                    is_train=False,
                    ext_tfs=ext_tfs,
                )
            
            if self.scene_gs:
                scene_gs_out = self.scene_gs.forward()
                    
            render_pkg = render_human_scene(
                data=data, 
                human_gs_out=human_gs_out,
                upperbody_gs_out=upperbody_gs_out,
                lowerbody_gs_out=lowerbody_gs_out,
                scene_gs_out=scene_gs_out, 
                bg_color=self.bg_color,
                render_mode=self.cfg.mode,
            )
            
            image = render_pkg["render"]
            
            torchvision.utils.save_image(image, f'{self.cfg.logdir}/anim/{idx:05d}.png')
            
        video_fname = f'{self.cfg.logdir}/anim_{self.cfg.dataset.name}_{self.cfg.dataset.seq}_{iter_s}.mp4'
        create_video(f'{self.cfg.logdir}/anim/', video_fname, fps=20)
        if not keep_images:
            shutil.rmtree(f'{self.cfg.logdir}/anim/')
            os.makedirs(f'{self.cfg.logdir}/anim/')
    
    @torch.no_grad()
    def render_canonical(self, iter=None, nframes=100, is_train_progress=False, pose_type=None):
        iter_s = 'final' if iter is None else f'{iter:06d}'
        iter_s += f'_{pose_type}' if pose_type is not None else ''
        
        if self.human_gs:
            self.human_gs.eval()
        
        os.makedirs(f'{self.cfg.logdir}/canon/', exist_ok=True)
        os.makedirs(f'{self.cfg.logdir}/upperbody_canon/', exist_ok=True)
        os.makedirs(f'{self.cfg.logdir}/lowerbody_canon/', exist_ok=True)
        
        camera_params = get_rotating_camera(
            dist=5.0, img_size=256 if is_train_progress else 512, 
            nframes=nframes, device='cuda',
            angle_limit=torch.pi if is_train_progress else 2*torch.pi,
        )
        
        betas = self.human_gs.betas.detach() if hasattr(self.human_gs, 'betas') else self.train_dataset.betas[0]
        
        static_smpl_params = get_smpl_static_params(
            betas=betas,
            pose_type=self.cfg.human.canon_pose_type if pose_type is None else pose_type,
        )
        
        if is_train_progress:
            progress_imgs = []
            upperbody_progress_imgs = []
            lowerbody_progress_imgs = []
        
        pbar = range(nframes) if is_train_progress else tqdm(range(nframes), desc="Canonical:")
        
        for idx in pbar:
            human_gs_out,upperbody_gs_out,lowerbody_gs_out, scene_gs_out = None, None, None, None
            
            cam_p = camera_params[idx]
            data = dict(static_smpl_params, **cam_p)

            if self.human_gs:
                human_gs_out = self.human_gs.forward(
                    global_orient=data['global_orient'],
                    body_pose=data['body_pose'],
                    betas=data['betas'],
                    transl=data['transl'],
                    smpl_scale=data['smpl_scale'],
                    dataset_idx=-1,
                    is_train=False,
                    ext_tfs=None,
                )
            if self.upperbody_gs:
                upperbody_gs_out = self.upperbody_gs.forward(
                    global_orient=data['global_orient'],
                    body_pose=data['body_pose'],
                    betas=data['betas'],
                    transl=data['transl'],
                    smpl_scale=data['smpl_scale'],
                    dataset_idx=-1,
                    is_train=False,
                    ext_tfs=None,
                )
            if self.lowerbody_gs:
                lowerbody_gs_out = self.lowerbody_gs.forward(
                    global_orient=data['global_orient'],
                    body_pose=data['body_pose'],
                    betas=data['betas'],
                    transl=data['transl'],
                    smpl_scale=data['smpl_scale'],
                    dataset_idx=-1,
                    is_train=False,
                    ext_tfs=None,
                )
                
            if is_train_progress:
                scale_mod = 0.5
                render_pkg = render_human_scene(
                    data=data, 
                    human_gs_out=human_gs_out, 
                    scene_gs_out=scene_gs_out, 
                    bg_color=self.bg_color,
                    render_mode='human',
                    scaling_modifier=scale_mod,
                )

                image = render_pkg["render"]

                progress_imgs.append(image)

                upperbody_render_pkg = render_human_scene(
                    data=data,
                    human_gs_out=human_gs_out,
                    scene_gs_out=scene_gs_out,
                    upperbody_gs_out=upperbody_gs_out,
                    lowerbody_gs_out=lowerbody_gs_out,
                    bg_color=self.bg_color,
                    render_mode='upperbody',
                    scaling_modifier=scale_mod,
                )

                upperbody_image = upperbody_render_pkg["render"]

                upperbody_progress_imgs.append(upperbody_image)

                lowerbody_render_pkg = render_human_scene(
                    data=data,
                    human_gs_out=human_gs_out,
                    scene_gs_out=scene_gs_out,
                    upperbody_gs_out=upperbody_gs_out,
                    lowerbody_gs_out=lowerbody_gs_out,
                    bg_color=self.bg_color,
                    render_mode='lowerbody',
                    scaling_modifier=scale_mod,
                )

                lowerbody_image = lowerbody_render_pkg["render"]

                lowerbody_progress_imgs.append(lowerbody_image)
                
                render_pkg = render_human_scene(
                    data=data, 
                    human_gs_out=human_gs_out, 
                    scene_gs_out=scene_gs_out, 
                    bg_color=self.bg_color,
                    render_mode='human',
                )

                image = render_pkg["render"]
                
                progress_imgs.append(image)

                upperbody_render_pkg = render_human_scene(
                    data=data,
                    human_gs_out=human_gs_out,
                    scene_gs_out=scene_gs_out,
                    upperbody_gs_out=upperbody_gs_out,
                    lowerbody_gs_out=lowerbody_gs_out,
                    bg_color=self.bg_color,
                    render_mode='upperbody',
                )

                upperbody_image = upperbody_render_pkg["render"]

                upperbody_progress_imgs.append(upperbody_image)

                lowerbody_render_pkg = render_human_scene(
                    data=data,
                    human_gs_out=human_gs_out,
                    scene_gs_out=scene_gs_out,
                    upperbody_gs_out=upperbody_gs_out,
                    lowerbody_gs_out=lowerbody_gs_out,
                    bg_color=self.bg_color,
                    render_mode='lowerbody',
                )

                lowerbody_image = lowerbody_render_pkg["render"]

                lowerbody_progress_imgs.append(lowerbody_image)
                
            else:
                render_pkg = render_human_scene(
                    data=data, 
                    human_gs_out=human_gs_out, 
                    scene_gs_out=scene_gs_out, 
                    bg_color=self.bg_color,
                    render_mode='human',
                )
                
                image = render_pkg["render"]
                
                torchvision.utils.save_image(image, f'{self.cfg.logdir}/canon/{idx:05d}.png')

                upperbody_render_pkg = render_human_scene(
                    data=data,
                    human_gs_out=human_gs_out,
                    scene_gs_out=scene_gs_out,
                    upperbody_gs_out=upperbody_gs_out,
                    lowerbody_gs_out=lowerbody_gs_out,
                    bg_color=self.bg_color,
                    render_mode='upperbody',
                )

                upperbody_image = upperbody_render_pkg["render"]

                torchvision.utils.save_image(upperbody_image, f'{self.cfg.logdir}/upperbody_canon/{idx:05d}.png')

                lowerbody_render_pkg = render_human_scene(
                    data=data,
                    human_gs_out=human_gs_out,
                    scene_gs_out=scene_gs_out,
                    upperbody_gs_out=upperbody_gs_out,
                    lowerbody_gs_out=lowerbody_gs_out,
                    bg_color=self.bg_color,
                    render_mode='lowerbody',
                )

                lowerbody_image = lowerbody_render_pkg["render"]

                torchvision.utils.save_image(lowerbody_image, f'{self.cfg.logdir}/lowerbody_canon/{idx:05d}.png')
        
        if is_train_progress:
            os.makedirs(f'{self.cfg.logdir}/train_progress/', exist_ok=True)
            log_img = torchvision.utils.make_grid(progress_imgs, nrow=4, pad_value=0)
            save_image(log_img, f'{self.cfg.logdir}/train_progress/{iter:06d}.png', 
                       text_labels=f"{iter:06d}, n_gs={self.human_gs.n_gs}")

            os.makedirs(f'{self.cfg.logdir}/train_progress/', exist_ok=True)
            upperbody_log_img = torchvision.utils.make_grid(upperbody_progress_imgs, nrow=4, pad_value=0)
            save_image(upperbody_log_img, f'{self.cfg.logdir}/train_progress/upperbody{iter:06d}.png',
                       text_labels=f"{iter:06d}, n_gs={self.upperbody_gs.n_gs}")

            os.makedirs(f'{self.cfg.logdir}/train_progress/', exist_ok=True)
            lowerbody_log_img = torchvision.utils.make_grid(lowerbody_progress_imgs, nrow=4, pad_value=0)
            save_image(lowerbody_log_img, f'{self.cfg.logdir}/train_progress/lowerbody{iter:06d}.png',
                       text_labels=f"{iter:06d}, n_gs={self.lowerbody_gs.n_gs}")

            return
        
        video_fname = f'{self.cfg.logdir}/canon_{self.cfg.dataset.name}_{self.cfg.dataset.seq}_{iter_s}.mp4'
        create_video(f'{self.cfg.logdir}/canon/', video_fname, fps=10)
        shutil.rmtree(f'{self.cfg.logdir}/canon/')
        os.makedirs(f'{self.cfg.logdir}/canon/')

        video_fname = f'{self.cfg.logdir}/upperbody_canon_{self.cfg.dataset.name}_{self.cfg.dataset.seq}_{iter_s}.mp4'
        create_video(f'{self.cfg.logdir}/upperbody_canon/', video_fname, fps=10)
        shutil.rmtree(f'{self.cfg.logdir}/upperbody_canon/')
        os.makedirs(f'{self.cfg.logdir}/upperbody_canon/')

        video_fname = f'{self.cfg.logdir}/lowerbody_canon_{self.cfg.dataset.name}_{self.cfg.dataset.seq}_{iter_s}.mp4'
        create_video(f'{self.cfg.logdir}/lowerbody_canon/', video_fname, fps=10)
        shutil.rmtree(f'{self.cfg.logdir}/lowerbody_canon/')
        os.makedirs(f'{self.cfg.logdir}/lowerbody_canon/')

        
    def render_poses(self, camera_params, smpl_params, pose_type='a_pose', bg_color='white'):
    
        if self.human_gs:
            self.human_gs.eval()
        
        betas = self.human_gs.betas.detach() if hasattr(self.human_gs, 'betas') else self.val_dataset.betas[0]
        
        nframes = len(camera_params)
        
        canon_forward_out = None
        if hasattr(self.human_gs, 'canon_forward'):
            canon_forward_out = self.human_gs.canon_forward()
        
        pbar = tqdm(range(nframes), desc="Canonical:")
        if bg_color == 'white':
            bg_color = torch.tensor([1, 1, 1], dtype=torch.float32, device="cuda")
        elif bg_color == 'black':
            bg_color = torch.tensor([0, 0, 0], dtype=torch.float32, device="cuda")
            
            
        imgs = []
        for idx in pbar:
            human_gs_out, scene_gs_out = None, None
            
            cam_p = camera_params[idx]
            data = dict(smpl_params, **cam_p)

            if self.human_gs:
                if canon_forward_out is not None:
                    human_gs_out = self.human_gs.forward_test(
                        canon_forward_out,
                        global_orient=data['global_orient'],
                        body_pose=data['body_pose'],
                        betas=data['betas'],
                        transl=data['transl'],
                        smpl_scale=data['smpl_scale'],
                        dataset_idx=-1,
                        is_train=False,
                        ext_tfs=None,
                    )
                else:
                    human_gs_out = self.human_gs.forward(
                        global_orient=data['global_orient'],
                        body_pose=data['body_pose'],
                        betas=data['betas'],
                        transl=data['transl'],
                        smpl_scale=data['smpl_scale'],
                        dataset_idx=-1,
                        is_train=False,
                        ext_tfs=None,
                    )

            render_pkg = render_human_scene(
                data=data, 
                human_gs_out=human_gs_out, 
                scene_gs_out=scene_gs_out, 
                bg_color=self.bg_color,
                render_mode='human',
            )
            image = render_pkg["render"]
            imgs.append(image)
        return imgs