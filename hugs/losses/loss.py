#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.
#

import torch
from lpips import LPIPS
import torch.nn as nn
import torch.nn.functional as F

from hugs.utils.sampler import PatchSampler
from hugs.models.clothGS import ClothGS
from pytorch3d.structures import Meshes
from pytorch3d.loss.point_mesh_distance import point_face_distance
from pytorch3d.structures import Pointclouds

from .utils import l1_loss, ssim


class HumanSceneLoss(nn.Module):
    def __init__(
        self,
        l_ssim_w=0.2,
        l_l1_w=0.8,
        l_lpips_w=0.0,
        l_lbs_w=0.0,
        l_humansep_w=0.0,
        l_clothsep_w=0.0,
        l_geo_dist_w=200.0,
        l_upperbody_t_offset_w=0.1,
        l_lowerbody_t_offset_w=0.1,
        num_patches=4,
        patch_size=32,
        use_patches=True,
        bg_color='white',

    ):
        super(HumanSceneLoss, self).__init__()
        
        self.l_ssim_w = l_ssim_w
        self.l_l1_w = l_l1_w
        self.l_lpips_w = l_lpips_w
        self.l_lbs_w = l_lbs_w
        self.l_humansep_w = l_humansep_w
        self.l_clothsep_w = l_clothsep_w
        self.l_geo_dist_w = l_geo_dist_w
        self.l_upperbody_t_offset_w = l_upperbody_t_offset_w
        self.l_lowerbody_t_offset_w = l_lowerbody_t_offset_w
        self.use_patches = use_patches
        
        self.lambda_t = 0.1

        
        self.bg_color = bg_color
        self.lpips = LPIPS(net="vgg", pretrained=True).to('cuda')
    
        for param in self.lpips.parameters(): param.requires_grad=False
        
        if self.use_patches:
            self.patch_sampler = PatchSampler(num_patch=num_patches, patch_size=patch_size, ratio_mask=0.9, dilate=0)
        
    def forward(
        self, 
        data, 
        render_pkg,
        human_gs_out,
        render_mode, 
        human_gs_init_values=None,
        bg_color=None,
        human_bg_color=None,
        upperbody_gs_out=None,
        lowerbody_gs_out=None,
        upperbody_gs:ClothGS = None ,
        lowerbody_gs:ClothGS = None,
        upperbody_gs_init_values=None,
        lowerbody_gs_init_values=None,
        cloth_bg_color=None,
        is_human_with_cloth_seprate=False,
        upperbody_gs_target_mesh=None,
        lowerbody_gs_target_mesh=None,
    ):
        loss_dict = {}
        extras_dict = {}
        
        if bg_color is not None:
            self.bg_color = bg_color
            
        if human_bg_color is None:
            human_bg_color = self.bg_color

        if cloth_bg_color is None:
            cloth_bg_color = self.bg_color
            
        gt_image = data['rgb']
        mask = data['mask'].unsqueeze(0)
        
        pred_img = render_pkg['render']

        fullbody_mask = data['humanbody_mask'].unsqueeze(0)

        upperbody_mask = data['upperbody_mask'].unsqueeze(0)

        lowerbody_mask = data['lowerbody_mask'].unsqueeze(0)
        
        if render_mode == "human":
            gt_image = gt_image * fullbody_mask + human_bg_color[:, None, None] * (1. - fullbody_mask)
            extras_dict['gt_img'] = gt_image
            extras_dict['pred_img'] = pred_img
        elif render_mode == "scene":
            # invert the mask
            extras_dict['pred_img'] = pred_img
            
            mask = (1. - data['mask'].unsqueeze(0))
            gt_image = gt_image * mask
            pred_img = pred_img * mask
            
            extras_dict['gt_img'] = gt_image
        elif render_mode == "cloth":
            gt_image = gt_image * upperbody_mask + cloth_bg_color[:, None, None] * (1. - upperbody_mask)
            extras_dict['gt_img'] = gt_image
            extras_dict['pred_img'] = pred_img

        else:
            extras_dict['gt_img'] = gt_image
            extras_dict['pred_img'] = pred_img

        if render_mode == "human_cloth":
            human_cloth_gt_image = gt_image * fullbody_mask + bg_color[:, None, None] * (1. - fullbody_mask)
            extras_dict['human_cloth_gt_img'] = human_cloth_gt_image
        
        if self.l_l1_w > 0.0:
            if render_mode == "human":
                Ll1 = l1_loss(pred_img, gt_image, mask)
            elif render_mode == "scene":
                Ll1 = l1_loss(pred_img, gt_image, 1 - mask)
            elif render_mode == "human_scene":
                Ll1 = l1_loss(pred_img, gt_image)
            elif render_mode == "human_cloth":
                Ll1 = l1_loss(pred_img, human_cloth_gt_image, mask)
            else:
                raise NotImplementedError
            loss_dict['l1'] = self.l_l1_w * Ll1

        if self.l_ssim_w > 0.0:
            loss_ssim = 1.0 - ssim(pred_img, gt_image)
            if render_mode == "human":
                loss_ssim = loss_ssim * (mask.sum() / (pred_img.shape[-1] * pred_img.shape[-2]))
            elif render_mode == "human_cloth":
                loss_ssim = loss_ssim * (mask.sum() / (pred_img.shape[-1] * pred_img.shape[-2]))
            elif render_mode == "scene":
                loss_ssim = loss_ssim * ((1 - mask).sum() / (pred_img.shape[-1] * pred_img.shape[-2]))
            elif render_mode == "human_scene":
                loss_ssim = 1.0 - ssim(pred_img,human_cloth_gt_image)
                loss_ssim = loss_ssim * (mask.sum() / (pred_img.shape[-1] * pred_img.shape[-2]))
            
                
            loss_dict['ssim'] = self.l_ssim_w * loss_ssim
        
        if self.l_lpips_w > 0.0 and not render_mode == "scene":
            if self.use_patches:
                if render_mode == "human" or render_mode == "human_cloth":
                    bg_color_lpips = torch.rand_like(pred_img)
                    image_bg = pred_img * mask + bg_color_lpips * (1. - mask)
                    gt_image_bg = gt_image * mask + bg_color_lpips * (1. - mask)
                    _, pred_patches, gt_patches = self.patch_sampler.sample(mask, image_bg, gt_image_bg)
                else:
                    _, pred_patches, gt_patches = self.patch_sampler.sample(mask, pred_img, gt_image)
                    
                loss_lpips = self.lpips(pred_patches.clip(max=1), gt_patches).mean()
                loss_dict['lpips_patch'] = self.l_lpips_w * loss_lpips
            else:
                bbox = data['bbox'].to(int)
                cropped_gt_image = gt_image[:, bbox[0]:bbox[2], bbox[1]:bbox[3]]
                cropped_pred_img = pred_img[:, bbox[0]:bbox[2], bbox[1]:bbox[3]]
                loss_lpips = self.lpips(cropped_pred_img.clip(max=1), cropped_gt_image).mean()
                loss_dict['lpips'] = self.l_lpips_w * loss_lpips
                
        if is_human_with_cloth_seprate and self.l_humansep_w > 0.0 and (render_mode == "human_scene" or render_mode == 'human_cloth') : 
            pred_human_img = render_pkg['human_img']
            gt_human_image = gt_image * fullbody_mask + human_bg_color[:, None, None] * (1. - fullbody_mask)
            
            Ll1_human = l1_loss(pred_human_img, gt_human_image, fullbody_mask)
            loss_dict['l1_human'] = self.l_l1_w * Ll1_human * self.l_humansep_w
            
            loss_ssim_human = 1.0 - ssim(pred_human_img, gt_human_image)
            loss_ssim_human = loss_ssim_human * (fullbody_mask.sum() / (pred_human_img.shape[-1] * pred_human_img.shape[-2]))
            loss_dict['ssim_human'] = self.l_ssim_w * loss_ssim_human * self.l_humansep_w
            
            bg_color_lpips = torch.rand_like(pred_human_img)
            image_bg = pred_human_img * fullbody_mask + bg_color_lpips * (1. - fullbody_mask)
            gt_image_bg = gt_human_image * fullbody_mask + bg_color_lpips * (1. - fullbody_mask)
            _, pred_patches, gt_patches = self.patch_sampler.sample(fullbody_mask, image_bg, gt_image_bg)
            loss_lpips_human = self.lpips(pred_patches.clip(max=1), gt_patches).mean()
            loss_dict['lpips_patch_human'] = self.l_lpips_w * loss_lpips_human * self.l_humansep_w

        if is_human_with_cloth_seprate and self.l_clothsep_w > 0.0 and (render_mode == "human_scene" or render_mode == 'human_cloth'):
            pred_upperbody_img = render_pkg['upperbody_img']
            gt_upperbody_image = gt_image * upperbody_mask + cloth_bg_color[:, None, None] * (1 - upperbody_mask)

            Ll1_upperbody = l1_loss(pred_upperbody_img, gt_upperbody_image, upperbody_mask)
            loss_dict['Ll1_upperbody'] = self.l_l1_w * Ll1_upperbody * self.l_clothsep_w

            loss_ssim_cupperbody = 1.0 - ssim(pred_upperbody_img, gt_upperbody_image)
            loss_ssim_cupperbody = loss_ssim_cupperbody * (upperbody_mask.sum() / (pred_upperbody_img.shape[-1] * pred_upperbody_img.shape[-2]))
            loss_dict['ssim_upperbody'] = self.l_ssim_w * loss_ssim_cupperbody * self.l_clothsep_w

            bg_color_lpips = torch.rand_like(pred_upperbody_img)
            image_bg = pred_upperbody_img * upperbody_mask + bg_color_lpips * (1 - upperbody_mask)
            gt_image_bg = gt_upperbody_image * upperbody_mask + bg_color_lpips * (1 - upperbody_mask)
            _, pred_patches, gt_patches = self.patch_sampler.sample(upperbody_mask, image_bg, gt_image_bg)
            loss_lpips_cloth = self.lpips(pred_patches.clip(max=1), gt_patches).mean()
            loss_dict['lpips_patch_upperbody'] = self.l_lpips_w * loss_lpips_cloth * self.l_clothsep_w
            
            
            gs_points = upperbody_gs_out['xyz_canon']
            
            pcls = Pointclouds(points=[gs_points])
            
            # 假设已定义 Meshes 和 Pointclouds 对象
            points = pcls.points_packed()  # (P, 3)
            tris = upperbody_gs_target_mesh.verts_packed()[upperbody_gs_target_mesh.faces_packed()]  # (T, 3, 3)
            
            point_to_face = point_face_distance(
                points,
                pcls.cloud_to_packed_first_idx(),
                tris,
                upperbody_gs_target_mesh.mesh_to_faces_packed_first_idx(),
                pcls.num_points_per_cloud().max().item(),
                1e-6
            )  # 形状 (P,)
            dist_point_to_face = point_to_face.sqrt()
            loss_distance = torch.mean(dist_point_to_face)
            # nan check
            if torch.isnan(loss_distance):
                loss_distance = torch.tensor(0.0).to(loss_distance.device)
                print("ldistance_upperbody is nan!")
            
            loss_dict['ldistance_upperbody'] = loss_distance * self.l_geo_dist_w
            
            # # 计算在当前姿态下，点云到目标网格的距离，如果距离大于阈值，则将该点剔除
            
            # now_gs_points = upperbody_gs_out['xyz']
            
            # now_pcls = Pointclouds(points=[now_gs_points])
            
            # # 假设已定义 Meshes 和 Pointclouds 对象
            # now_points = now_pcls.points_packed()
            # now_mesh = Meshes(upperbody_gs_out['tailorNet_output'].tailornet_v[None], upperbody_gs_out['tailorNet_output'].tailornet_f[None])
            # now_tris = now_mesh.verts_packed()[now_mesh.faces_packed()]  # (T, 3, 3)
            
            
            # now_point_to_face = point_face_distance(
            #     now_points,
            #     now_pcls.cloud_to_packed_first_idx(),
            #     now_tris,
            #     now_mesh.mesh_to_faces_packed_first_idx(),
            #     now_pcls.num_points_per_cloud().max().item(),
            #     1e-6
            # )
            # now_loss_distance = torch.mean(now_point_to_face)
            
            
            # loss_dict['lnow_distance_upperbody'] = now_loss_distance * self.l_geo_dist_w
            
            
            
            # dists_per_face = point_mesh_face_distance(
            #     meshes=upperbody_gs_target_mesh,
            #     pcls=pcls
            # )
            # dists = dists_per_face.min(dim=1).values  # 形状 (N,)
            # loss_distance = torch.mean(dists)  # 平均距离损失
            # loss_dict['distance_upperbody'] = loss_distance * l_geo_dist_w
            
            
            upperbody_t_offset_loss = torch.max(torch.abs(upperbody_gs.t) - self.lambda_t * upperbody_gs.avg_edge_len,torch.tensor(0.0).to(upperbody_gs.t.device)) ** 2
            upperbody_t_offset_loss = upperbody_t_offset_loss.mean()
            # nan check
            if torch.isnan(upperbody_t_offset_loss):
                upperbody_t_offset_loss = torch.tensor(0.0).to(upperbody_t_offset_loss.device)
                print("t_offset_upperbody is nan!")
            
            loss_dict['t_offset_upperbody'] = self.l_upperbody_t_offset_w * upperbody_t_offset_loss

        if is_human_with_cloth_seprate and self.l_clothsep_w > 0.0 and (render_mode == "human_scene" or render_mode == 'human_cloth'):
            pred_lowerbody_img = render_pkg['lowerbody_img']
            gt_lowerbody_image = gt_image * lowerbody_mask + cloth_bg_color[:, None, None] * (1 - lowerbody_mask)

            # # 使用lowerbody_mask，只计算lowerbody_mask区域的loss
            # lowerbody_mask_bool = lowerbody_mask.bool().repeat(3, 1, 1)
            # pred_lowerbody_img_masked = pred_lowerbody_img[lowerbody_mask_bool].view(3, -1)
            # gt_lowerbody_image_masked = gt_lowerbody_image[lowerbody_mask_bool].view(3, -1)

            Ll1_lowerbody = l1_loss(pred_lowerbody_img, gt_lowerbody_image, lowerbody_mask)
            loss_dict['Ll1_lowerbody'] = self.l_l1_w * Ll1_lowerbody * self.l_clothsep_w

            loss_ssim_clowerbody = 1.0 - ssim(pred_lowerbody_img, gt_lowerbody_image)
            loss_ssim_clowerbody = loss_ssim_clowerbody * (lowerbody_mask.sum() / (pred_lowerbody_img.shape[-1] * pred_lowerbody_img.shape[-2]))
            loss_dict['ssim_lowerbody'] = self.l_ssim_w * loss_ssim_clowerbody * self.l_clothsep_w

            bg_color_lpips = torch.rand_like(pred_lowerbody_img)
            image_bg = pred_lowerbody_img * lowerbody_mask + bg_color_lpips * (1 - lowerbody_mask)
            gt_image_bg = gt_lowerbody_image * lowerbody_mask + bg_color_lpips * (1 - lowerbody_mask)
            _, pred_patches, gt_patches = self.patch_sampler.sample(lowerbody_mask, image_bg, gt_image_bg)
            loss_lpips_cloth = self.lpips(pred_patches.clip(max=1), gt_patches).mean()
            loss_dict['lpips_patch_lowerbody'] = self.l_lpips_w * loss_lpips_cloth * self.l_clothsep_w
            
            gs_points = lowerbody_gs_out['xyz_canon']
            
            pcls = Pointclouds(points=[gs_points])
            
            # 假设已定义 Meshes 和 Pointclouds 对象
            points = pcls.points_packed()  # (P, 3)
            tris = lowerbody_gs_target_mesh.verts_packed()[lowerbody_gs_target_mesh.faces_packed()]  # (T, 3, 3)
            
            point_to_face = point_face_distance(
                points,
                pcls.cloud_to_packed_first_idx(),
                tris,
                lowerbody_gs_target_mesh.mesh_to_faces_packed_first_idx(),
                pcls.num_points_per_cloud().max().item(),
                1e-6
            )  # 形状 (P,)
            dist_point_to_face = point_to_face.sqrt()
            loss_distance = torch.mean(dist_point_to_face)
            
            # nan check
            if torch.isnan(loss_distance):
                loss_distance = torch.tensor(0.0).to(loss_distance.device)
                print("ldistance_lowerbody is nan!")
            
            loss_dict['ldistance_lowwerbody'] = loss_distance * self.l_geo_dist_w
            
            # # 计算在当前姿态下，点云到目标网格的距离，如果距离大于阈值，则将该点剔除
            
            
            # now_gs_points = lowerbody_gs_out['xyz']
            
            # now_pcls = Pointclouds(points=[now_gs_points])
            
            # # 假设已定义 Meshes 和 Pointclouds 对象
            
            # now_points = now_pcls.points_packed()
            
            # now_mesh = Meshes(lowerbody_gs_out['tailorNet_output'].tailornet_v[None], lowerbody_gs_out['tailorNet_output'].tailornet_f[None])
            
            # now_tris = now_mesh.verts_packed()[now_mesh.faces_packed()]  # (T, 3, 3)
            
            
            # now_point_to_face = point_face_distance(
            #     now_points,
            #     now_pcls.cloud_to_packed_first_idx(),
            #     now_tris,
            #     now_mesh.mesh_to_faces_packed_first_idx(),
            #     now_pcls.num_points_per_cloud().max().item(),
            #     1e-6
            # )
            
            # now_loss_distance = torch.mean(now_point_to_face)
            
            
            # loss_dict['lnow_distance_lowwerbody'] = now_loss_distance * self.l_geo_dist_w
            
            
            lowerbody_t_offset_loss = torch.max(torch.abs(lowerbody_gs.t) - self.lambda_t * lowerbody_gs.avg_edge_len,torch.tensor(0.0).to(lowerbody_gs.t.device)) ** 2
            lowerbody_t_offset_loss = lowerbody_t_offset_loss.mean()
            
            # nan check
            if torch.isnan(lowerbody_t_offset_loss):
                lowerbody_t_offset_loss = torch.tensor(0.0).to(lowerbody_t_offset_loss.device)
                print("t_offset_lowerbody is nan!")
            
            loss_dict['t_offset_lowerbody'] = self.l_lowerbody_t_offset_w * lowerbody_t_offset_loss
            

        if is_human_with_cloth_seprate is False and self.l_clothsep_w > 0.0 and (render_mode == "human_scene" or render_mode == 'human_cloth'):
            pred_human_with_cloth_img = render_pkg['human_with_cloth_img']
            gt_human_with_cloth_image = gt_image * mask + human_bg_color[:, None, None] * (1 - mask)
            
            Ll1_human_with_cloth = l1_loss(pred_human_with_cloth_img, gt_human_with_cloth_image, mask)
            loss_dict['Ll1_human_with_cloth'] = self.l_l1_w * Ll1_human_with_cloth * self.l_clothsep_w

            loss_ssim_human_with_cloth = 1.0 - ssim(pred_human_with_cloth_img, gt_human_with_cloth_image)
            loss_ssim_human_with_cloth = loss_ssim_human_with_cloth * (mask.sum() / (pred_human_with_cloth_img.shape[-1] * pred_human_with_cloth_img.shape[-2]))
            loss_dict['ssim_human_with_cloth'] = self.l_ssim_w * loss_ssim_human_with_cloth * self.l_clothsep_w

            bg_color_lpips = torch.rand_like(pred_human_with_cloth_img)
            image_bg = pred_human_with_cloth_img * mask + bg_color_lpips * (1 - mask)
            gt_image_bg = gt_human_with_cloth_image * mask + bg_color_lpips * (1 - mask)
            _, pred_patches, gt_patches = self.patch_sampler.sample(mask, image_bg, gt_image_bg)
            loss_lpips_human_with_cloth = self.lpips(pred_patches.clip(max=1), gt_patches).mean()
            loss_dict['lpips_patch_human_with_cloth'] = self.l_lpips_w * loss_lpips_human_with_cloth * self.l_clothsep_w
        

        if self.l_lbs_w > 0.0 and 'lbs_weights'in human_gs_out.keys() and not render_mode == "scene":
            if 'gt_lbs_weights' in human_gs_out.keys():
                loss_lbs = F.mse_loss(
                    human_gs_out['lbs_weights'], 
                    human_gs_out['gt_lbs_weights'].detach()).mean()
            else:
                loss_lbs = F.mse_loss(
                    human_gs_out['lbs_weights'], 
                    human_gs_init_values['lbs_weights']).mean()
            loss_dict['lbs'] = self.l_lbs_w * loss_lbs

        if self.l_lbs_w > 0.0 and upperbody_gs_out is not None and 'lbs_weights'in human_gs_out.keys() and not render_mode == "scene":
            if 'gt_lbs_weights' in upperbody_gs_out.keys():
                loss_lbs = F.mse_loss(
                    upperbody_gs_out['lbs_weights'],
                    upperbody_gs_out['gt_lbs_weights'].detach()).mean()
            else:
                loss_lbs = F.mse_loss(
                    upperbody_gs_out['lbs_weights'],
                    upperbody_gs_init_values['lbs_weights']).mean()
            loss_dict['lbs_upperbody'] = self.l_lbs_w * loss_lbs

        if self.l_lbs_w > 0.0 and lowerbody_gs_out is not None and 'lbs_weights'in human_gs_out.keys() and not render_mode == "scene":
            if 'gt_lbs_weights' in lowerbody_gs_out.keys():
                loss_lbs = F.mse_loss(
                    lowerbody_gs_out['lbs_weights'],
                    lowerbody_gs_out['gt_lbs_weights'].detach()).mean()
            else:
                loss_lbs = F.mse_loss(
                    lowerbody_gs_out['lbs_weights'],
                    lowerbody_gs_init_values['lbs_weights']).mean()
            loss_dict['lbs_lowerbody'] = self.l_lbs_w * loss_lbs
        
        loss = 0.0
        for k, v in loss_dict.items():
            loss += v
        
        return loss, loss_dict, extras_dict
    