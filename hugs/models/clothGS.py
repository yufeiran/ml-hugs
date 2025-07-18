#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.
#

import torch
import trimesh
from torch import nn
from loguru import logger
import torch.nn.functional as F
from hugs.models.hugs_wo_trimlp import smpl_lbsmap_top_k, smpl_lbsweight_top_k
from pytorch3d.structures import Meshes
from pytorch3d.loss import point_mesh_distance
from pytorch3d.loss.point_mesh_distance import point_face_distance
from pytorch3d.structures import Pointclouds
import numpy as np


from hugs.utils.general import (
    inverse_sigmoid,
    get_expon_lr_func,
    strip_symmetric,
    build_scaling_rotation,
)
from hugs.utils.rotations import (
    axis_angle_to_rotation_6d,
    matrix_to_quaternion,
    matrix_to_rotation_6d,
    quaternion_multiply,
    quaternion_to_matrix,
    rotation_6d_to_axis_angle,
    rotation_6d_to_matrix,
    torch_rotation_matrix_from_vectors,
)
from hugs.cfg.constants import SMPL_PATH
from hugs.utils.subdivide_smpl import subdivide_smpl_model
from hugs.utils.compute_local_frame import compute_all_local_frames

from .modules.lbs import lbs_extra
from .modules.smpl_layer import SMPL
from .modules.triplane import TriPlane
from .modules.MultiResolutionHashGrid import MultiResolutionHashGrid
from .modules.decoders import AppearanceDecoder, DeformationDecoder, GeometryDecoder

from .modules.tailorNet_layer import TailorNet_Layer



SCALE_Z = 1e-5


class ClothGS:

    def setup_functions(self):
        def build_covariance_from_scaling_rotation(scaling, scaling_modifier, rotation):
            L = build_scaling_rotation(scaling_modifier * scaling, rotation)
            actual_covariance = L @ L.transpose(1, 2)
            symm = strip_symmetric(actual_covariance)
            return symm

        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log

        self.covariance_activation = build_covariance_from_scaling_rotation

        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = inverse_sigmoid

        self.rotation_activation = torch.nn.functional.normalize

    def __init__(
            self,
            garment_class: str,
            gender: str,
            sh_degree: int,
            only_rgb: bool = False,
            n_subdivision: int = 0,
            opacity_start_iter: int = 0,
            opacity_end_iter: int = 0,
            use_surface=False,
            init_2d=False,
            rotate_sh=False,
            isotropic=False,
            init_scale_multiplier=0.5,
            n_features=32,
            use_deformer=False,
            disable_posedirs=False,
            triplane_res=512,
            betas=None,
            use_multires_hashgrid=True,
    ):
        self.opacity_start_iter = opacity_start_iter
        self.opacity_end_iter = opacity_end_iter
        self.only_rgb = only_rgb
        self.active_sh_degree = 0
        self.max_sh_degree = sh_degree
        self._xyz = torch.empty(0)
        self.scaling_multiplier = torch.empty(0)

        self.max_radii2D = torch.empty(0)
        self.xyz_gradient_accum = torch.empty(0)
        self.denom = torch.empty(0)
        self.optimizer = None
        self.percent_dense = 0
        self.spatial_lr_scale = 0
        self.device = 'cuda'
        self.use_surface = use_surface
        self.init_2d = init_2d
        self.rotate_sh = rotate_sh
        self.isotropic = isotropic
        self.init_scale_multiplier = init_scale_multiplier
        self.use_deformer = use_deformer
        self.disable_posedirs = disable_posedirs
        self.use_multires_hashgrid = use_multires_hashgrid
        self.use_triplane = (use_multires_hashgrid is False)

        self.deformer = 'smpl'

        if betas is not None:
            self.create_betas(betas, requires_grad=False)

        if use_multires_hashgrid:
            n_levels = 3
            self.hashgrid = MultiResolutionHashGrid(
                n_levels=n_levels,
                n_features_per_level=n_features,
                log2_hashmap_size=18,
                base_resolution=16,
                max_resolution=512,
                input_range=(-1, 1),
            ).to('cuda')
            self.appearance_dec = AppearanceDecoder(n_features=n_features * n_levels).to('cuda')
            self.deformation_dec = DeformationDecoder(n_features=n_features * n_levels,
                                                    disable_posedirs=disable_posedirs).to('cuda')
            self.geometry_dec = GeometryDecoder(n_features=n_features * n_levels, use_surface=use_surface,scale_dim=1).to('cuda')
        elif self.use_triplane:
            self.triplane = TriPlane(n_features, resX=triplane_res, resY=triplane_res, resZ=triplane_res).to('cuda')
            self.appearance_dec = AppearanceDecoder(n_features=n_features * 3).to('cuda')
            self.deformation_dec = DeformationDecoder(n_features=n_features * 3,
                                                    disable_posedirs=disable_posedirs).to('cuda')
            self.geometry_dec = GeometryDecoder(n_features=n_features * 3, use_surface=use_surface,scale_dim=1).to('cuda')
        else:
            # use xyz as input
            self.appearance_dec = AppearanceDecoder(n_features=3).to('cuda')
            self.deformation_dec = DeformationDecoder(n_features=3,
                                                    disable_posedirs=disable_posedirs).to('cuda')
            self.geometry_dec = GeometryDecoder(n_features=3, use_surface=use_surface,scale_dim=1).to('cuda')


        if n_subdivision > 0:
            logger.info(f"Subdividing SMPL model {n_subdivision} times")
            self.smpl_template = subdivide_smpl_model(smoothing=True, n_iter=n_subdivision).to(self.device)
        else:
            self.smpl_template = SMPL(SMPL_PATH).to(self.device)

        #self.cloth_template = TailorNet_Layer().to(self.device)
        #self.cloth = TailorNet_Layer().to(self.device)


        self.smpl = SMPL(SMPL_PATH).to(self.device)

        edges = trimesh.Trimesh(
            vertices=self.smpl_template.v_template.detach().cpu().numpy(),
            faces=self.smpl_template.faces, process=False
        ).edges_unique
        self.edges = torch.from_numpy(edges).to(self.device).long()

        self.tailorNet_layer = TailorNet_Layer(garment_class,SMPL_PATH,gender).to(self.device)

        self.init_values = {}
        self.get_vitruvian_verts()

        self.setup_functions()

    def create_body_pose(self, body_pose, requires_grad=False):
        body_pose = axis_angle_to_rotation_6d(body_pose.reshape(-1, 3)).reshape(-1, 23 * 6)
        self.body_pose = nn.Parameter(body_pose, requires_grad=requires_grad)
        logger.info(f"Created body pose with shape: {body_pose.shape}, requires_grad: {requires_grad}")

    def create_global_orient(self, global_orient, requires_grad=False):
        global_orient = axis_angle_to_rotation_6d(global_orient.reshape(-1, 3)).reshape(-1, 6)
        self.global_orient = nn.Parameter(global_orient, requires_grad=requires_grad)
        logger.info(f"Created global_orient with shape: {global_orient.shape}, requires_grad: {requires_grad}")

    def create_betas(self, betas, requires_grad=False):
        self.betas = nn.Parameter(betas, requires_grad=requires_grad)
        logger.info(f"Created betas with shape: {betas.shape}, requires_grad: {requires_grad}")

    def create_transl(self, transl, requires_grad=False):
        self.transl = nn.Parameter(transl, requires_grad=requires_grad)
        logger.info(f"Created transl with shape: {transl.shape}, requires_grad: {requires_grad}")

    def create_eps_offsets(self, eps_offsets, requires_grad=False):
        logger.info(f"NOT CREATED eps_offsets with shape: {eps_offsets.shape}, requires_grad: {requires_grad}")

    def save_opimizer_params(self,path):
        # save pose/global_orient/transl to disk 

        save_dict = {
            'global_orient': self.global_orient,
            'body_pose': self.body_pose,
            'transl': self.transl,
        }

        # change to numpy
        for key in save_dict.keys():
            save_dict[key] = save_dict[key].detach().cpu().numpy()

        np.save(path,save_dict)

    @property
    def get_xyz(self):
        return self._xyz
    
    @property
    def get_xyz_by_uv(self,):
        # get xyz by calculate uv to xyz
        xyz = torch.zeros(self.u.shape[0], 3).to('cuda')
        
        vert = self.cloth_verts
        normal = self.cloth_normals
        
        tri_v0 = vert[self.tri_vert_ids[:, 0]]
        tri_v1 = vert[self.tri_vert_ids[:, 1]]
        tri_v2 = vert[self.tri_vert_ids[:, 2]]        
        
        normal_v0 = normal[self.tri_vert_ids[:, 0]]
        normal_v1 = normal[self.tri_vert_ids[:, 1]]
        normal_v2 = normal[self.tri_vert_ids[:, 2]]
        
        
        u = self.u
        v = self.v
        k = 1 - u - v
        
        xyz = u * tri_v0 + v * tri_v1 + k * tri_v2
        point_normal = u * normal_v0 + v * normal_v1 + k * normal_v2
        
        xyz += self.t * point_normal
        
        return xyz
        
        
        
        

    def state_dict(self):
        save_dict = {
            'active_sh_degree': self.active_sh_degree,
            'xyz': self._xyz,
            'triplane': self.triplane.state_dict(),
            'appearance_dec': self.appearance_dec.state_dict(),
            'geometry_dec': self.geometry_dec.state_dict(),
            'deformation_dec': self.deformation_dec.state_dict(),
            'scaling_multiplier': self.scaling_multiplier,
            'max_radii2D': self.max_radii2D,
            'xyz_gradient_accum': self.xyz_gradient_accum,
            'denom': self.denom,
            'optimizer': self.optimizer.state_dict(),
            'spatial_lr_scale': self.spatial_lr_scale,
        }
        return save_dict

    def load_state_dict(self, state_dict, cfg=None):
        self.active_sh_degree = state_dict['active_sh_degree']
        self._xyz = state_dict['xyz']
        self.max_radii2D = state_dict['max_radii2D']
        xyz_gradient_accum = state_dict['xyz_gradient_accum']
        denom = state_dict['denom']
        opt_dict = state_dict['optimizer']
        self.spatial_lr_scale = state_dict['spatial_lr_scale']

        self.triplane.load_state_dict(state_dict['triplane'])
        self.appearance_dec.load_state_dict(state_dict['appearance_dec'])
        self.geometry_dec.load_state_dict(state_dict['geometry_dec'])
        self.deformation_dec.load_state_dict(state_dict['deformation_dec'])
        self.scaling_multiplier = state_dict['scaling_multiplier']

        if cfg is None:
            from hugs.cfg.config import cfg as default_cfg
            cfg = default_cfg.human.lr

        self.setup_optimizer(cfg)
        self.xyz_gradient_accum = xyz_gradient_accum
        self.denom = denom
        try:
            self.optimizer.load_state_dict(opt_dict)
        except ValueError as e:
            logger.warning(f"Optimizer load failed: {e}")
            logger.warning("Continue without a pretrained optimizer")

    def __repr__(self):
        repr_str = "ClothGS: \n"
        repr_str += "xyz: {} \n".format(self._xyz.shape)
        repr_str += "max_radii2D: {} \n".format(self.max_radii2D.shape)
        repr_str += "xyz_gradient_accum: {} \n".format(self.xyz_gradient_accum.shape)
        repr_str += "denom: {} \n".format(self.denom.shape)
        return repr_str

    def canon_forward(self):
        if self.use_multires_hashgrid:
            grid_feats = self.hashgrid(self.get_xyz_by_uv)
            appearance_out = self.appearance_dec(grid_feats)
            geometry_out = self.geometry_dec(grid_feats)
        elif self.use_triplane:
            tri_feats = self.triplane(self.get_xyz_by_uv)
            appearance_out = self.appearance_dec(tri_feats)
            geometry_out = self.geometry_dec(tri_feats) 
        else:
            appearance_out = self.appearance_dec(self.get_xyz_by_uv)
            geometry_out = self.geometry_dec(self.get_xyz_by_uv)

        xyz_offsets = geometry_out['xyz']
        gs_rot6d = geometry_out['rotations']
        # FIXME
        # gs_scales = geometry_out['scales'] * self.scaling_multiplier
        gs_scales = geometry_out['scales']

        gs_scales = torch.clamp(gs_scales, max=2 * torch.mean(gs_scales))
        gs_scales = gs_scales.repeat(1, 3)  # 3 scales for each vertex
        
        # gs_scales 原本是 [N,2]的，现在改成[N,3]的，其中第三个维度的计算为 系数乘以前两个维度的最小值
        # 比如 gs_scales[0] = [1,2] -> gs_scales[0] = [1,2,0.5] 
        # scale_factor = 0.000
        # gs_scales = torch.cat([gs_scales,torch.min(gs_scales,dim=1)[0].unsqueeze(1)*scale_factor],dim=1)

        gs_opacity = appearance_out['opacity']
        


        gs_shs = appearance_out['shs'].reshape(-1, 16, 3)

        if self.use_deformer:
            if self.use_multires_hashgrid:
                deformation_out = self.deformation_dec(grid_feats)
            elif self.use_triplane:
                deformation_out = self.deformation_dec(tri_feats)
            else:
                deformation_out = self.deformation_dec(self.get_xyz_by_uv)
            lbs_weights = deformation_out['lbs_weights']
            lbs_weights = F.softmax(lbs_weights / 0.1, dim=-1)
            posedirs = deformation_out['posedirs']
            if abs(lbs_weights.sum(-1).mean().item() - 1) < 1e-7:
                pass
            else:
                logger.warning(f"LBS weights should sum to 1, but it is: {lbs_weights.sum(-1).mean().item()}")
        else:
            lbs_weights = None
            posedirs = None

        return {
            'xyz_offsets': xyz_offsets,
            'scales': gs_scales,
            'rot6d_canon': gs_rot6d,
            'shs': gs_shs,
            'opacity': gs_opacity,
            'lbs_weights': lbs_weights,
            'posedirs': posedirs,
        }

    def forward_test(
            self,
            canon_forward_out,
            global_orient=None,
            body_pose=None,
            betas=None,
            transl=None,
            smpl_scale=None,
            dataset_idx=-1,
            is_train=False,
            ext_tfs=None,
    ):
        xyz_offsets = canon_forward_out['xyz_offsets']
        gs_rot6d = canon_forward_out['rot6d_canon']
        gs_scales = canon_forward_out['scales']

        gs_xyz = self.get_xyz_by_uv + xyz_offsets

        gs_rotmat = rotation_6d_to_matrix(gs_rot6d)
        gs_rotq = matrix_to_quaternion(gs_rotmat)

        gs_opacity = canon_forward_out['opacity']
        gs_shs = canon_forward_out['shs'].reshape(-1, 16, 3)

        if self.isotropic:
            gs_scales = torch.ones_like(gs_scales) * torch.mean(gs_scales, dim=-1, keepdim=True)

        gs_scales_canon = gs_scales.clone()

        if self.use_deformer:
            lbs_weights = canon_forward_out['lbs_weights']
            posedirs = canon_forward_out['posedirs']
            if abs(lbs_weights.sum(-1).mean().item() - 1) < 1e-7:
                pass
            else:
                logger.warning(f"LBS weights should sum to 1, but it is: {lbs_weights.sum(-1).mean().item()}")
        else:
            lbs_weights = None
            posedirs = None

        if hasattr(self, 'global_orient') and global_orient is None:
            global_orient = rotation_6d_to_axis_angle(
                self.global_orient[dataset_idx].reshape(-1, 6)).reshape(3)

        if hasattr(self, 'body_pose') and body_pose is None:
            body_pose = rotation_6d_to_axis_angle(
                self.body_pose[dataset_idx].reshape(-1, 6)).reshape(23 * 3)

        if hasattr(self, 'betas') and betas is None:
            betas = self.betas

        if hasattr(self, 'transl') and transl is None:
            transl = self.transl[dataset_idx]

        # vitruvian -> t-pose -> posed
        # remove and reapply the blendshape
        smpl_output = self.smpl(
            betas=betas.unsqueeze(0),
            body_pose=body_pose.unsqueeze(0),
            global_orient=global_orient.unsqueeze(0),
            disable_posedirs=False,
            return_full_pose=True,
        )

        gt_lbs_weights = None
        if self.use_deformer:
            A_t2pose = smpl_output.A[0]
            A_vitruvian2pose = A_t2pose @ self.inv_A_t2vitruvian
            deformed_xyz, _, lbs_T, _, _ = lbs_extra(
                A_vitruvian2pose[None], gs_xyz[None], posedirs, lbs_weights,
                smpl_output.full_pose, disable_posedirs=self.disable_posedirs, pose2rot=True
            )
            deformed_xyz = deformed_xyz.squeeze(0)
            lbs_T = lbs_T.squeeze(0)

            with torch.no_grad():
                # gt lbs is needed for lbs regularization loss
                # predicted lbs should be close to gt lbs
                _, gt_lbs_weights = smpl_lbsweight_top_k(
                    lbs_weights=self.tailorNet_layer.lbs_weights,
                    points=gs_xyz.unsqueeze(0),
                    template_points=self.vitruvian_verts.unsqueeze(0),
                )
                gt_lbs_weights = gt_lbs_weights.squeeze(0)
                if abs(gt_lbs_weights.sum(-1).mean().item() - 1) < 1e-7:
                    pass
                else:
                    logger.warning(f"GT LBS weights should sum to 1, but it is: {gt_lbs_weights.sum(-1).mean().item()}")
        else:
            curr_offsets = (smpl_output.shape_offsets + smpl_output.pose_offsets)[0]
            T_t2pose = smpl_output.T[0]
            T_vitruvian2t = self.inv_T_t2vitruvian.clone()
            T_vitruvian2t[..., :3, 3] = T_vitruvian2t[..., :3, 3] + self.canonical_offsets - curr_offsets
            T_vitruvian2pose = T_t2pose @ T_vitruvian2t

            _, lbs_T = smpl_lbsmap_top_k(
                lbs_weights=self.smpl.lbs_weights,
                verts_transform=T_vitruvian2pose.unsqueeze(0),
                points=gs_xyz.unsqueeze(0),
                template_points=self.vitruvian_verts.unsqueeze(0),
                K=6,
            )
            lbs_T = lbs_T.squeeze(0)

            homogen_coord = torch.ones_like(gs_xyz[..., :1])
            gs_xyz_homo = torch.cat([gs_xyz, homogen_coord], dim=-1)
            deformed_xyz = torch.matmul(lbs_T, gs_xyz_homo.unsqueeze(-1))[..., :3, 0]

        if smpl_scale is not None:
            deformed_xyz = deformed_xyz * smpl_scale.unsqueeze(0)
            gs_scales = gs_scales * smpl_scale.unsqueeze(0)

        if transl is not None:
            deformed_xyz = deformed_xyz + transl.unsqueeze(0)

        deformed_gs_rotmat = lbs_T[:, :3, :3] @ gs_rotmat
        deformed_gs_rotq = matrix_to_quaternion(deformed_gs_rotmat)

        if ext_tfs is not None:
            tr, rotmat, sc = ext_tfs
            deformed_xyz = (tr[..., None] + (sc[None] * (rotmat @ deformed_xyz[..., None]))).squeeze(-1)
            gs_scales = sc * gs_scales

            rotq = matrix_to_quaternion(rotmat)
            deformed_gs_rotq = quaternion_multiply(rotq, deformed_gs_rotq)
            deformed_gs_rotmat = quaternion_to_matrix(deformed_gs_rotq)

        self.normals = torch.zeros_like(gs_xyz)
        self.normals[:, 2] = 1.0

        canon_normals = (gs_rotmat @ self.normals.unsqueeze(-1)).squeeze(-1)
        deformed_normals = (deformed_gs_rotmat @ self.normals.unsqueeze(-1)).squeeze(-1)

        deformed_gs_shs = gs_shs.clone()

        return {
            'xyz': deformed_xyz,
            'xyz_canon': gs_xyz,
            'xyz_offsets': xyz_offsets,
            'scales': gs_scales,
            'scales_canon': gs_scales_canon,
            'rotq': deformed_gs_rotq,
            'rotq_canon': gs_rotq,
            'rotmat': deformed_gs_rotmat,
            'rotmat_canon': gs_rotmat,
            'shs': deformed_gs_shs,
            'opacity': gs_opacity,
            'normals': deformed_normals,
            'normals_canon': canon_normals,
            'active_sh_degree': self.active_sh_degree,
            'rot6d_canon': gs_rot6d,
            'lbs_weights': lbs_weights,
            'posedirs': posedirs,
            'gt_lbs_weights': gt_lbs_weights,
        }

    def forward(
            self,
            iter_num,
            global_orient=None,
            body_pose=None,
            betas=None,
            transl=None,
            smpl_scale=None,
            dataset_idx=-1,
            is_train=False,
            ext_tfs=None,
    ):

        if self.use_multires_hashgrid:
            grid_feats = self.hashgrid(self.get_xyz_by_uv)
            appearance_out = self.appearance_dec(grid_feats)
            geometry_out = self.geometry_dec(grid_feats)
        elif self.use_triplane:
            tri_feats = self.triplane(self.get_xyz_by_uv)
            appearance_out = self.appearance_dec(tri_feats)
            geometry_out = self.geometry_dec(tri_feats) 
        else:
            appearance_out = self.appearance_dec(self.get_xyz_by_uv)
            geometry_out = self.geometry_dec(self.get_xyz_by_uv)

        xyz_offsets = geometry_out['xyz']
        gs_rot6d = geometry_out['rotations']
        
        gs_scales = geometry_out['scales'] * self.scaling_multiplier

        gs_scales = torch.clamp(gs_scales, max=2 * torch.mean(gs_scales))

        gs_scales = gs_scales.repeat(1, 3)  # 3 scales for each vertex
        
        # gs_scales 原本是 [N,2]的，现在改成[N,3]的，其中第三个维度的计算为 系数乘以前两个维度的最小值
        # 比如 gs_scales[0] = [1,2] -> gs_scales[0] = [1,2,0.5] 
        # scale_factor = 0.001
        # gs_scales = torch.cat([gs_scales,torch.min(gs_scales,dim=1)[0].unsqueeze(1)*scale_factor],dim=1)

        gs_xyz = self.get_xyz_by_uv + xyz_offsets
        

        gs_rotmat = rotation_6d_to_matrix(gs_rot6d)
        
        # make gs_rotmat to identity
        gs_rotmat = torch.eye(3).repeat(gs_rotmat.shape[0], 1, 1).to('cuda')
        
        gs_rotq = matrix_to_quaternion(gs_rotmat)

        if iter_num is None:
            iter_num = 3000

        # gs_opacity = appearance_out['opacity']
        
        # if iter_num > 2000:
        #  # make gs_opacity to 1
        #     gs_opacity = torch.ones(gs_opacity.shape).to('cuda')
        if iter_num > self.opacity_start_iter and iter_num < self.opacity_end_iter:
            gs_opacity = appearance_out['opacity']
        else:
            gs_opacity = torch.ones(appearance_out['opacity'].shape).to('cuda')
        
        gs_shs = appearance_out['shs'].reshape(-1, 16, 3)
        
        gs_rgb = appearance_out['rgb'].reshape(-1,3)

        if self.isotropic:
            gs_scales = torch.ones_like(gs_scales) * torch.mean(gs_scales, dim=-1, keepdim=True)

        gs_scales_canon = gs_scales.clone()

        if self.use_deformer:
            if self.use_multires_hashgrid:
                deformation_out = self.deformation_dec(grid_feats)
            elif self.use_triplane:
                deformation_out = self.deformation_dec(tri_feats)
            else:
                deformation_out = self.deformation_dec(self.get_xyz_by_uv)
            lbs_weights = deformation_out['lbs_weights']
            lbs_weights = F.softmax(lbs_weights / 0.1, dim=-1)
            posedirs = deformation_out['posedirs']
            if abs(lbs_weights.sum(-1).mean().item() - 1) < 1e-7:
                pass
            else:
                logger.warning(f"LBS weights should sum to 1, but it is: {lbs_weights.sum(-1).mean().item()}")
        else:
            lbs_weights = None
            posedirs = None

        if hasattr(self, 'global_orient') and global_orient is None:
            global_orient = rotation_6d_to_axis_angle(
                self.global_orient[dataset_idx].reshape(-1, 6)).reshape(3)

        if hasattr(self, 'body_pose') and body_pose is None:
            body_pose = rotation_6d_to_axis_angle(
                self.body_pose[dataset_idx].reshape(-1, 6)).reshape(23 * 3)

        if hasattr(self, 'betas') and betas is None:
            betas = self.betas

        if hasattr(self, 'transl') and transl is None:
            transl = self.transl[dataset_idx]

        # vitruvian -> t-pose -> posed
        # remove and reapply the blendshape
        smpl_output = self.smpl(
            betas=betas.unsqueeze(0),
            body_pose=body_pose.unsqueeze(0),
            global_orient=global_orient.unsqueeze(0),
            disable_posedirs=False,
            return_full_pose=True,
        )

        tailorNet_output = self.tailorNet_layer(
            betas=betas.unsqueeze(0),
            body_pose=body_pose.unsqueeze(0),
            global_orient=global_orient.unsqueeze(0),
            disable_posedirs=False,
            return_full_pose=True,
            transl=transl.unsqueeze(0),
        )

        gt_lbs_weights = None
        if self.use_deformer:
            A_t2pose = smpl_output.A[0]
            A_vitruvian2pose = A_t2pose @ self.inv_A_t2vitruvian
            deformed_xyz, _, lbs_T, _, _ = lbs_extra(
                A_vitruvian2pose[None], gs_xyz[None], posedirs, lbs_weights,
                smpl_output.full_pose, disable_posedirs=self.disable_posedirs, pose2rot=True
            )
            deformed_xyz = deformed_xyz.squeeze(0)
            lbs_T = lbs_T.squeeze(0)

            with torch.no_grad():
                # gt lbs is needed for lbs regularization loss
                # predicted lbs should be close to gt lbs
                # _, gt_lbs_weights = smpl_lbsweight_top_k(
                #     lbs_weights=self.smpl.lbs_weights,
                #     points=gs_xyz.unsqueeze(0),
                #     template_points=self.vitruvian_verts.unsqueeze(0),
                # )
                _, gt_lbs_weights = smpl_lbsweight_top_k(
                    lbs_weights=self.tailorNet_layer.tailorNet_lbs_weights,
                    points=gs_xyz.unsqueeze(0),
                    template_points=self.vitruvian_verts.unsqueeze(0),
                )
                gt_lbs_weights = gt_lbs_weights.squeeze(0)
                if abs(gt_lbs_weights.sum(-1).mean().item() - 1) < 1e-7:
                    pass
                else:
                    logger.warning(f"GT LBS weights should sum to 1, but it is: {gt_lbs_weights.sum(-1).mean().item()}")
        else:
            curr_offsets = (smpl_output.shape_offsets + smpl_output.pose_offsets)[0]
            T_t2pose = smpl_output.T[0]
            T_vitruvian2t = self.inv_T_t2vitruvian.clone()
            T_vitruvian2t[..., :3, 3] = T_vitruvian2t[..., :3, 3] + self.canonical_offsets - curr_offsets
            T_vitruvian2pose = T_t2pose @ T_vitruvian2t

            _, lbs_T = smpl_lbsmap_top_k(
                lbs_weights=self.smpl.lbs_weights,
                verts_transform=T_vitruvian2pose.unsqueeze(0),
                points=gs_xyz.unsqueeze(0),
                template_points=self.vitruvian_verts.unsqueeze(0),
                K=6,
            )
            lbs_T = lbs_T.squeeze(0)

            homogen_coord = torch.ones_like(gs_xyz[..., :1])
            gs_xyz_homo = torch.cat([gs_xyz, homogen_coord], dim=-1)
            deformed_xyz = torch.matmul(lbs_T, gs_xyz_homo.unsqueeze(-1))[..., :3, 0]

        # deformed_xyz = tailorNet_output.tailornet_v.float().to('cuda')

        if smpl_scale is not None:
            deformed_xyz = deformed_xyz * smpl_scale.unsqueeze(0)
            gs_scales = gs_scales * smpl_scale.unsqueeze(0)

        if transl is not None:
            deformed_xyz = deformed_xyz + transl.unsqueeze(0)

        deformed_gs_rotmat = lbs_T[:, :3, :3] @ gs_rotmat
        deformed_gs_rotq = matrix_to_quaternion(deformed_gs_rotmat)

        if ext_tfs is not None:
            tr, rotmat, sc = ext_tfs
            deformed_xyz = (tr[..., None] + (sc[None] * (rotmat @ deformed_xyz[..., None]))).squeeze(-1)
            gs_scales = sc * gs_scales

            rotq = matrix_to_quaternion(rotmat)
            deformed_gs_rotq = quaternion_multiply(rotq, deformed_gs_rotq)
            deformed_gs_rotmat = quaternion_to_matrix(deformed_gs_rotq)

        self.normals = torch.zeros_like(gs_xyz)
        self.normals[:, 2] = 1.0

        canon_normals = (gs_rotmat @ self.normals.unsqueeze(-1)).squeeze(-1)
        deformed_normals = (deformed_gs_rotmat @ self.normals.unsqueeze(-1)).squeeze(-1)

        deformed_gs_shs = gs_shs.clone()

        return {
            'xyz': deformed_xyz,
            'xyz_canon': gs_xyz,
            'xyz_offsets': xyz_offsets,
            'scales': gs_scales,
            'scales_canon': gs_scales_canon,
            'rotq': deformed_gs_rotq,
            'rotq_canon': gs_rotq,
            'rotmat': deformed_gs_rotmat,
            'rotmat_canon': gs_rotmat,
            'shs': deformed_gs_shs,
            'rgb': gs_rgb,
            'opacity': gs_opacity,
            'normals': deformed_normals,
            'normals_canon': canon_normals,
            'active_sh_degree': self.active_sh_degree,
            'rot6d_canon': gs_rot6d,
            'lbs_weights': lbs_weights,
            'posedirs': posedirs,
            'gt_lbs_weights': gt_lbs_weights,
            'tailorNet_output': tailorNet_output,
        }

    def oneupSHdegree(self):
        if self.active_sh_degree < self.max_sh_degree:
            logger.info(f"Going from SH degree {self.active_sh_degree} to {self.active_sh_degree + 1}")
            self.active_sh_degree += 1

    @torch.no_grad()
    def get_vitruvian_verts(self):
        vitruvian_pose = torch.zeros(69, dtype=self.smpl.dtype, device=self.device)
        vitruvian_pose[2] = 1.0
        vitruvian_pose[5] = -1.0
        tailor_net_output = self.tailorNet_layer(body_pose=vitruvian_pose[None], betas=self.betas[None], disable_posedirs=False)
        vitruvian_verts = tailor_net_output.tailornet_v
        self.A_t2vitruvian = tailor_net_output.A[0].detach()
        self.T_t2vitruvian = tailor_net_output.T[0].detach()
        self.inv_T_t2vitruvian = torch.inverse(self.T_t2vitruvian)
        self.inv_A_t2vitruvian = torch.inverse(self.A_t2vitruvian)
        self.canonical_offsets = tailor_net_output.shape_offsets + tailor_net_output.pose_offsets
        self.canonical_offsets = self.canonical_offsets[0].detach()
        self.vitruvian_verts = vitruvian_verts.detach()
        
        self.target_mesh = Meshes(verts= tailor_net_output.tailornet_v[None], faces= tailor_net_output.tailornet_f[None])
        
        return vitruvian_verts.detach()

    @torch.no_grad()
    def get_vitruvian_verts_template(self):
        vitruvian_pose = torch.zeros(69, dtype=self.smpl_template.dtype, device=self.device)
        vitruvian_pose[2] = 1.0
        vitruvian_pose[5] = -1.0
        smpl_output = self.smpl_template(body_pose=vitruvian_pose[None], betas=self.betas[None], disable_posedirs=False)
        vitruvian_verts = smpl_output.vertices[0]
        tailor_net_output =  self.tailorNet_layer(body_pose=vitruvian_pose[None], betas=self.betas[None], disable_posedirs=False)

        cloth_verts = tailor_net_output.tailornet_v
        cloth_faces = tailor_net_output.tailornet_f

        return vitruvian_verts.detach(), cloth_verts.detach().cuda(), cloth_faces.detach().cuda()

    def train(self):
        pass

    def eval(self):
        pass

    def add_gs_point_in_triangle(self,n, verts, faces, scales,rotmats, lbs_weights ):
        # 现在在mesh顶点上生成了GS点属性,现在需要在三角形内随机生成n个gs点，同时插值scales和rotmats、lbs_weights
        
        mesh = trimesh.Trimesh(vertices=verts.detach().cpu().numpy(), faces=faces.cpu().numpy())
        edges = mesh.edges_unique
        self.edges = torch.from_numpy(edges).to(self.device).long()
        
        gs_xyzs = []
        gs_scales = []
        gs_rotmats = []
        gs_lbs_weights = []
        
        gs_us = []
        gs_vs = []
        gs_face_ids = []
        gs_tri_vert_ids = []
        
        for f in range(faces.shape[0]):
            face = faces[f]
            v1 = verts[face[0]]
            v2 = verts[face[1]]
            v3 = verts[face[2]]
            scale1 = scales[face[0]]
            scale2 = scales[face[1]]
            scale3 = scales[face[2]]
            rotmat1 = rotmats[face[0]]
            rotmat2 = rotmats[face[1]]
            rotmat3 = rotmats[face[2]]
            lbs_weight1 = lbs_weights[face[0]]
            lbs_weight2 = lbs_weights[face[1]]
            lbs_weight3 = lbs_weights[face[2]]
            
            for i in range(n):
                
                # 生成重心坐标,r1+r2+r3=1
                r1 = np.random.rand()
                r2 = np.random.rand()
                r3 = np.random.rand()
                norm = r1 + r2 + r3
                r1 = r1 / norm
                r2 = r2 / norm
                r3 = r3 / norm

                gs_xyz = r1 * v1 + r2 * v2 + r3 * v3
                gs_scale = r1 * scale1 + r2 * scale2 + r3 * scale3
                gs_scale *= 1/(n * n)
                gs_rotmat = r1 * rotmat1 + r2 * rotmat2 + r3 * rotmat3
                gs_lbs_weight = r1 * lbs_weight1 + r2 * lbs_weight2 + r3 * lbs_weight3
                
                gs_xyzs.append(gs_xyz)
                gs_scales.append(gs_scale)
                gs_rotmats.append(gs_rotmat)
                gs_lbs_weights.append(gs_lbs_weight)
                
                gs_us.append(r1)
                gs_vs.append(r2)
                gs_face_ids.append(f)
                gs_tri_vert_ids.append([face[0],face[1],face[2]])
                
        return gs_xyzs, gs_scales, gs_rotmats, gs_lbs_weights, gs_us,gs_vs, gs_face_ids, gs_tri_vert_ids


    def initialize(self):
        
        add_point_num = 1
        
        t_pose_verts,cloth_t_pose_verts,cloth_faces = self.get_vitruvian_verts_template()
        cloth_t_pose_verts = cloth_t_pose_verts.float()
        t_pose_verts = cloth_t_pose_verts

        self.scaling_multiplier = torch.ones((t_pose_verts.shape[0], 1), device="cuda")

        xyz_offsets = torch.zeros_like(t_pose_verts)
        colors = torch.ones_like(t_pose_verts) * 0.5

        shs = torch.zeros((colors.shape[0], 3, 16)).float().cuda()
        shs[:, :3, 0] = colors
        shs[:, 3:, 1:] = 0.0
        shs = shs.transpose(1, 2).contiguous()

        scales = torch.zeros_like(t_pose_verts)

        import trimesh
        mesh = trimesh.Trimesh(vertices=t_pose_verts.detach().cpu().numpy(), faces=cloth_faces.cpu().numpy())
        edges = mesh.edges_unique
        self.edges = torch.from_numpy(edges).to(self.device).long()
        self.cloth_verts = cloth_t_pose_verts
        # get mesh normals
        vert_normals = torch.tensor(mesh.vertex_normals).float().cuda()
        self.cloth_normals = vert_normals
        
        avg_edge_len = torch.mean(torch.norm(t_pose_verts[self.edges[:, 0]] - t_pose_verts[self.edges[:, 1]], dim=-1)).cuda()
        self.avg_edge_len = avg_edge_len
        
        # for v in range(t_pose_verts.shape[0]):
        #     selected_edges = torch.any(self.edges == v, dim=-1)
        #     selected_edges_len = torch.norm(
        #         t_pose_verts[self.edges[selected_edges][0]] - t_pose_verts[self.edges[selected_edges][1]],
        #         dim=-1
        #     )
        #     selected_edges_len *= self.init_scale_multiplier
        #     scales[v, 0] = torch.log(torch.max(selected_edges_len))
        #     scales[v, 1] = torch.log(torch.max(selected_edges_len))

        #     if not self.use_surface:
        #         scales[v, 2] = torch.log(torch.max(selected_edges_len))

        # if self.use_surface or self.init_2d:
        #     scales = scales[..., :2]

        # scales = torch.exp(scales)

        # if self.use_surface or self.init_2d:
        #     scale_z = torch.ones_like(scales[:, -1:]) * SCALE_Z
        #     scales = torch.cat([scales, scale_z], dim=-1)


        vert_normals = torch.tensor(mesh.vertex_normals).float().cuda()

        gs_normals = torch.zeros_like(vert_normals)
        gs_normals[:, 2] = 1.0

        norm_rotmat = torch_rotation_matrix_from_vectors(gs_normals, vert_normals)

        rotq = matrix_to_quaternion(norm_rotmat)
        rot6d = matrix_to_rotation_6d(norm_rotmat)
        
        rot_mats,scales=compute_all_local_frames(t_pose_verts, cloth_faces,global_orient=None)
        rotq = matrix_to_quaternion(rot_mats)
        rot6d = matrix_to_rotation_6d(rot_mats)

        self.normals = gs_normals
        deformed_normals = (norm_rotmat @ gs_normals.unsqueeze(-1)).squeeze(-1)

        opacity = 0.1 * torch.ones((t_pose_verts.shape[0], 1), dtype=torch.float, device="cuda")

        posedirs = self.smpl_template.posedirs.detach().clone()
        lbs_weights = self.tailorNet_layer.tailorNet.get_w()
        lbs_weights = torch.from_numpy(lbs_weights).float().cuda()
        self.lbs_weights = lbs_weights
        
        self.n_gs = t_pose_verts.shape[0]
        self._xyz = nn.Parameter(t_pose_verts.detach(), requires_grad=False)
        self._xyz = nn.Parameter(cloth_t_pose_verts.requires_grad_(True))
        
        # 如果需要使用重心坐标表示xyz，就必须用add_point_num来生成gs点
        if add_point_num > 0:
            
            gs_xyzs, gs_scales, gs_rotmats, gs_lbs_weights,gs_us,gs_vs, gs_face_ids, gs_tri_vert_ids = self.add_gs_point_in_triangle(add_point_num, t_pose_verts, cloth_faces, scales, rot_mats, lbs_weights)
            
            gs_rotmats = torch.stack(gs_rotmats).float().cuda()
            addition_rotq = matrix_to_quaternion(gs_rotmats)
            addition_rot6d = matrix_to_rotation_6d(gs_rotmats)
            
            self.lbs_weights = torch.stack(gs_lbs_weights).float().cuda()
            scales = torch.stack(gs_scales).float().cuda()
            rot6d = addition_rot6d
            rotq = addition_rotq
            
            self.n_gs = len(gs_xyzs)
            self._xyz = nn.Parameter(torch.stack(gs_xyzs).float().cuda(), requires_grad=True)
            self.scaling_multiplier = torch.ones((self._xyz.shape[0], 1), device="cuda")
            shs = torch.zeros((self._xyz.shape[0], 3, 16)).float().cuda() 
            shs[:, :3, 0] = torch.ones_like(self._xyz) * 0.5
            shs[:, 3:, 1:] = 0.0
            shs = shs.transpose(1, 2).contiguous()
            opacity =  0.1 * torch.ones((self._xyz.shape[0], 1), dtype=torch.float, device="cuda")
            xyz_offsets = torch.zeros_like(self._xyz)
            
            u = torch.tensor(gs_us).float().cuda()
            v = torch.tensor(gs_vs).float().cuda()
            u = u.unsqueeze(-1)
            v = v.unsqueeze(-1)
            self.u = nn.Parameter(u, requires_grad=True)
            self.v = nn.Parameter(v, requires_grad=True)
            self.t = nn.Parameter(torch.zeros_like(u), requires_grad=True)

            self.face_ids = torch.tensor(gs_face_ids).long().cuda().unsqueeze(-1)
            self.tri_vert_ids = torch.tensor(gs_tri_vert_ids).long().cuda()
            
        

        self.max_radii2D = torch.zeros((self.get_xyz_by_uv.shape[0]), device="cuda")
        return {
            'xyz_offsets': xyz_offsets,
            'scales': scales,
            'rot6d_canon': rot6d,
            'shs': shs,
            'opacity': opacity,
            'lbs_weights': self.lbs_weights,
            'posedirs': posedirs,
            'deformed_normals': deformed_normals,
            'faces': self.smpl.faces_tensor,
            'edges': self.edges,
        }

    def setup_optimizer(self, cfg):
        self.percent_dense = cfg.percent_dense
        self.xyz_gradient_accum = torch.zeros((self.get_xyz_by_uv.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz_by_uv.shape[0], 1), device="cuda")
        self.spatial_lr_scale = cfg.smpl_spatial

        params = [
            {'params': [self._xyz], 'lr': cfg.position_init * cfg.smpl_spatial, "name": "xyz"},
            {"params": self.u, "lr": 1e-4,"name":"u"},
            {"params": self.v, "lr": 1e-4,"name":"v"},
            {"params": self.t, "lr": 1e-5,"name":"t"},
            # {'params': self.triplane.parameters(), 'lr': cfg.vembed, 'name': 'v_embed'},
            {'params': self.geometry_dec.parameters(), 'lr': cfg.geometry, 'name': 'geometry_dec'},
            {'params': self.appearance_dec.parameters(), 'lr': cfg.appearance, 'name': 'appearance_dec'},
            {'params': self.deformation_dec.parameters(), 'lr': cfg.deformation, 'name': 'deform_dec'},
        ]
        if self.use_multires_hashgrid:
            params.append({'params': self.hashgrid.parameters(), 'lr': cfg.hashgrid, 'name': 'hashgrid'})
        if self.use_triplane:
            params.append({'params': self.triplane.parameters(), 'lr': cfg.vembed, 'name': 'v_embed'})
        else:
            pass

        if hasattr(self, 'global_orient') and self.global_orient.requires_grad:
            params.append({'params': self.global_orient, 'lr': cfg.smpl_pose, 'name': 'global_orient'})

        if hasattr(self, 'body_pose') and self.body_pose.requires_grad:
            params.append({'params': self.body_pose, 'lr': cfg.smpl_pose, 'name': 'body_pose'})

        if hasattr(self, 'betas') and self.betas.requires_grad:
            params.append({'params': self.betas, 'lr': cfg.smpl_betas, 'name': 'betas'})

        if hasattr(self, 'transl') and self.betas.requires_grad:
            params.append({'params': self.transl, 'lr': cfg.smpl_trans, 'name': 'transl'})

        self.non_densify_params_keys = [
            'global_orient', 'body_pose', 'betas', 'transl',
            'v_embed', 'geometry_dec', 'appearance_dec', 'deform_dec',
        ]

        for param in params:
            logger.info(f"Parameter: {param['name']}, lr: {param['lr']}")

        self.optimizer = torch.optim.Adam(params, lr=0.0, eps=1e-15)
        self.xyz_scheduler_args = get_expon_lr_func(
            lr_init=cfg.position_init * cfg.smpl_spatial,
            lr_final=cfg.position_final * cfg.smpl_spatial,
            lr_delay_mult=cfg.position_delay_mult,
            max_steps=cfg.position_max_steps,
        )

    def step(self):
        self.optimizer.step()
        
        # keep u,v in [0,1]
        self.u.data = torch.clamp(self.u.data, 0, 1)
        self.v.data = torch.clamp(self.v.data, 0, 1)

    def update_learning_rate(self, iteration):
        ''' Learning rate scheduling per step '''
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "xyz":
                lr = self.xyz_scheduler_args(iteration)
                param_group['lr'] = lr
                return lr

    def replace_tensor_to_optimizer(self, tensor, name):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] == name:
                stored_state = self.optimizer.state.get(group['params'][0], None)
                stored_state["exp_avg"] = torch.zeros_like(tensor)
                stored_state["exp_avg_sq"] = torch.zeros_like(tensor)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(tensor.requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def _prune_optimizer(self, mask):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] in self.non_densify_params_keys:
                continue
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = stored_state["exp_avg"][mask]
                stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask]

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter((group["params"][0][mask].requires_grad_(True)))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(group["params"][0][mask].requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors
    
    def _prune_tensors(self, valid_points_mask, keys_to_prune):
        optimizable_tensors = {}
        for key in keys_to_prune:
            if hasattr(self, key):
                tensor = getattr(self, key)
                if isinstance(tensor, nn.Parameter):
                    # 对于 nn.Parameter，剪枝后仍保持 Parameter 类型
                    pruned_tensor = tensor.data[valid_points_mask]
                    optimizable_tensors[key] = nn.Parameter(pruned_tensor.to(self.device))
                else:
                    # 普通张量直接剪枝
                    pruned_tensor = tensor[valid_points_mask]
                    optimizable_tensors[key] = pruned_tensor.to(self.device)
        return optimizable_tensors
    
    def _prune_tensors(self, valid_points_mask, keys_to_prune):
        optimizable_tensors = {}
        for key in keys_to_prune:
            if hasattr(self, key):
                tensor = getattr(self, key)
                if isinstance(tensor, nn.Parameter):
                    # 对于 nn.Parameter，剪枝后仍保持 Parameter 类型
                    pruned_tensor = tensor.data[valid_points_mask]
                    optimizable_tensors[key] = nn.Parameter(pruned_tensor.to(self.device))
                else:
                    # 普通张量直接剪枝
                    pruned_tensor = tensor[valid_points_mask]
                    optimizable_tensors[key] = pruned_tensor.to(self.device)
        return optimizable_tensors
    

    def prune_points(self, mask):
        valid_points_mask = ~mask
        
        # 定义需要剪枝的键（包含普通张量和 Parameter）
        keys_to_prune = {
            "u", "v","t",           # nn.Parameter
            "face_ids", "tri_vert_ids",   # 普通张量
            "xyz", "features_dc", "opacity", "scaling", "rotation"  # 其他可能的参数
        }
        
        optimizable_tensors = self._prune_tensors(valid_points_mask, keys_to_prune)

        # 重新赋值给类的属性
        for key in optimizable_tensors:
            if hasattr(self, key):
                # 如果是 Parameter，直接替换（已在 _prune_tensors 中处理）
                if isinstance(getattr(self, key), nn.Parameter):
                    setattr(self, key, optimizable_tensors[key])
                else:
                    # 普通张量直接覆盖
                    setattr(self, key, optimizable_tensors[key])
        # self._xyz = optimizable_tensors["xyz"]
        self.scaling_multiplier = self.scaling_multiplier[valid_points_mask]

        self.scales_tmp = self.scales_tmp[valid_points_mask]
        self.opacity_tmp = self.opacity_tmp[valid_points_mask]
        self.rotmat_tmp = self.rotmat_tmp[valid_points_mask]

        self.xyz_gradient_accum = self.xyz_gradient_accum[valid_points_mask]

        self.denom = self.denom[valid_points_mask]
        self.max_radii2D = self.max_radii2D[valid_points_mask]

    def cat_tensors_to_optimizer(self, tensors_dict):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] in self.non_densify_params_keys:
                continue

            assert len(group["params"]) == 1
            extension_tensor = tensors_dict[group["name"]]
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:

                stored_state["exp_avg"] = torch.cat((stored_state["exp_avg"], torch.zeros_like(extension_tensor)),
                                                    dim=0)
                stored_state["exp_avg_sq"] = torch.cat((stored_state["exp_avg_sq"], torch.zeros_like(extension_tensor)),
                                                       dim=0)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(
                    torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(
                    torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]

        return optimizable_tensors

    def densification_postfix(self, new_xyz, new_scaling_multiplier, new_opacity_tmp, new_scales_tmp, new_rotmat_tmp):
        d = {
            "xyz": new_xyz,
        }

        optimizable_tensors = self.cat_tensors_to_optimizer(d)
        self._xyz = optimizable_tensors["xyz"]
        self.scaling_multiplier = torch.cat((self.scaling_multiplier, new_scaling_multiplier), dim=0)
        self.opacity_tmp = torch.cat([self.opacity_tmp, new_opacity_tmp], dim=0)
        self.scales_tmp = torch.cat([self.scales_tmp, new_scales_tmp], dim=0)
        self.rotmat_tmp = torch.cat([self.rotmat_tmp, new_rotmat_tmp], dim=0)

        self.xyz_gradient_accum = torch.zeros((self.get_xyz_by_uv.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz_by_uv.shape[0], 1), device="cuda")
        self.max_radii2D = torch.zeros((self.get_xyz_by_uv.shape[0]), device="cuda")
        
    def densification_postfix(self,new_u,new_v,new_t,new_face_id, new_tri_vert_id,new_scaling_multiplier, new_opacity_tmp, new_scales_tmp, new_rotmat_tmp):
        self.u = nn.Parameter(torch.cat([self.u, new_u], dim=0))
        self.v = nn.Parameter(torch.cat([self.v, new_v], dim=0))
        self.t = nn.Parameter(torch.cat([self.t, new_t], dim=0))
        
        self.scaling_multiplier = torch.cat((self.scaling_multiplier, new_scaling_multiplier), dim=0)
        self.opacity_tmp = torch.cat([self.opacity_tmp, new_opacity_tmp], dim=0)
        self.scales_tmp = torch.cat([self.scales_tmp, new_scales_tmp], dim=0)
        self.rotmat_tmp = torch.cat([self.rotmat_tmp, new_rotmat_tmp], dim=0)
        
        # 合并其他属性
        self.face_ids = torch.cat([self.face_ids, new_face_id], dim=0)
        self.tri_vert_ids = torch.cat([self.tri_vert_ids, new_tri_vert_id], dim=0)
        
        # 更新优化器
        self.optimizer.add_param_group({'params': self.u[-len(new_u):], 'lr': 0.01})
        self.optimizer.add_param_group({'params': self.v[-len(new_v):], 'lr': 0.01})
        
        # 重置梯度统计
        self.xyz_gradient_accum = torch.zeros((self.u.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.u.shape[0], 1), device="cuda")
        self.max_radii2D = torch.zeros((self.u.shape[0]), device="cuda")

    def densify_and_split(self, grads, grad_threshold, scene_extent,distance_threshold, N=2):
        n_init_points = self.get_xyz_by_uv.shape[0]
        scales = self.scales_tmp
        rotation = self.rotmat_tmp
        # Extract points that satisfy the gradient condition
        padded_grad = torch.zeros((n_init_points), device="cuda")
        padded_grad[:grads.shape[0]] = grads.squeeze()
        selected_pts_mask = torch.where(padded_grad >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(scales, dim=1).values > self.percent_dense * scene_extent)
        # filter elongated gaussians
        med = scales.median(dim=1, keepdim=True).values
        stdmed_mask = (((scales - med) / med).squeeze(-1) >= 1.0).any(dim=-1)
        selected_pts_mask = torch.logical_and(selected_pts_mask, stdmed_mask)
        
        # pcls = Pointclouds(points=[self.get_xyz])
            
        # points = pcls.points_packed()  # (P, 3)
        # tris = self.target_mesh.verts_packed()[self.target_mesh.faces_packed()]  # (T, 3, 3)
        
        # dis_point_to_face = point_face_distance(
        #     points,
        #     pcls.cloud_to_packed_first_idx(),
        #     tris,
        #     self.target_mesh.mesh_to_faces_packed_first_idx(),
        #     pcls.num_points_per_cloud().max().item(),
        #     1e-8,
        # )  # 形状 (P,)

        
        # distance_mask = dis_point_to_face < distance_threshold * distance_threshold
        
        # third_mask = distance_mask
        
        # selected_pts_mask = torch.logical_and(selected_pts_mask, third_mask)

        stds = scales[selected_pts_mask].repeat(N, 1)
        means = torch.zeros((stds.size(0), 3), device="cuda")
        samples = torch.normal(mean=means, std=torch.relu(stds))
        rots = rotation[selected_pts_mask].repeat(N, 1, 1)
        # new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self.get_xyz[selected_pts_mask].repeat(N, 1)
        
        new_u = self.u[selected_pts_mask]
        new_v = self.v[selected_pts_mask]
        
        noise_scale = 0.1
        new_u += torch.randn_like(new_u) * noise_scale * 0.5
        new_v += torch.randn_like(new_v) * noise_scale * 0.5
        
        # 确保 u 在 [0, 0.499] 范围内
        new_u = torch.clamp(new_u, min=0.0, max=0.499)  # 标量 min/max
        
        # 确保 max_v 是张量且非负
        max_v = (0.499 - new_u).clamp(min=0.0)  # 保证 max_v >= 0
        min_tensor = torch.tensor(0.0, device=new_v.device)
        new_v = torch.clamp(new_v, min=min_tensor, max=max_v)  # 张量 min/max ✅

        new_u = new_u.repeat(N, 1)
        new_v = new_v.repeat(N, 1)
        
        new_t = self.t[selected_pts_mask].repeat(N, 1) + torch.rand_like(self.t[selected_pts_mask].repeat(N, 1)) * noise_scale * 0.05
        
        new_face_id = self.face_ids[selected_pts_mask].repeat(N, 1)
        new_tri_vert_id = self.tri_vert_ids[selected_pts_mask].repeat(N, 1)
        
        new_scaling_multiplier = self.scaling_multiplier[selected_pts_mask].repeat(N, 1) / (0.8 * N)
        new_opacity_tmp = self.opacity_tmp[selected_pts_mask].repeat(N, 1)
        new_scales_tmp = self.scales_tmp[selected_pts_mask].repeat(N, 1)
        new_rotmat_tmp = self.rotmat_tmp[selected_pts_mask].repeat(N, 1, 1)

        self.densification_postfix(new_u= new_u,new_v=new_v,new_t=new_t,new_face_id=new_face_id,
                                   new_tri_vert_id=new_tri_vert_id,
                                   new_scaling_multiplier=new_scaling_multiplier,
                                   new_opacity_tmp=new_opacity_tmp,new_scales_tmp= new_scales_tmp,new_rotmat_tmp= new_rotmat_tmp)

        prune_filter = torch.cat(
            (selected_pts_mask, torch.zeros(N * selected_pts_mask.sum(), device="cuda", dtype=bool)))
        self.prune_points(prune_filter)

    def densify_and_clone(self, grads, grad_threshold, scene_extent,distance_threshold):
        # Extract points that satisfy the gradient condition
        scales = self.scales_tmp
        grad_cond = torch.norm(grads, dim=-1) >= grad_threshold
        scale_cond = torch.max(scales, dim=1).values <= self.percent_dense * scene_extent


        # pcls = Pointclouds(points=[self.get_xyz])
            
        # points = pcls.points_packed()  # (P, 3)
        # tris = self.target_mesh.verts_packed()[self.target_mesh.faces_packed()]  # (T, 3, 3)
        
        # dis_point_to_face = point_face_distance(
        #     points,
        #     pcls.cloud_to_packed_first_idx(),
        #     tris,
        #     self.target_mesh.mesh_to_faces_packed_first_idx(),
        #     pcls.num_points_per_cloud().max().item(),
        #     1e-8,
        # )  # 形状 (P,)
        
        # distance_mask = dis_point_to_face < distance_threshold * distance_threshold
        
        # third_mask = distance_mask
        
        selected_pts_mask = torch.where(grad_cond, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask, scale_cond)
        # selected_pts_mask = torch.logical_and(selected_pts_mask, third_mask)
        

        # new_xyz = self._xyz[selected_pts_mask]
        
        new_u = self.u[selected_pts_mask]
        new_v = self.v[selected_pts_mask]
        
        noise_scale = 0.1
        new_u += torch.randn_like(new_u) * noise_scale * 0.5
        new_v += torch.randn_like(new_v) * noise_scale * 0.5
        
        # 确保 u 在 [0, 0.499] 范围内
        new_u = torch.clamp(new_u, min=0.0, max=0.499)  # 标量 min/max
        
        # 确保 max_v 是张量且非负
        max_v = (0.499 - new_u).clamp(min=0.0)  # 保证 max_v >= 0
        min_tensor = torch.tensor(0.0, device=new_v.device)
        new_v = torch.clamp(new_v, min=min_tensor, max=max_v)  # 张量 min/max ✅
        
        new_t = self.t[selected_pts_mask] + torch.rand_like(self.t[selected_pts_mask]) * noise_scale * 0.05

        
        new_face_id = self.face_ids[selected_pts_mask]
        new_tri_vert_id = self.tri_vert_ids[selected_pts_mask]
        
        new_scaling_multiplier = self.scaling_multiplier[selected_pts_mask]
        new_opacity_tmp = self.opacity_tmp[selected_pts_mask]
        new_scales_tmp = self.scales_tmp[selected_pts_mask]
        new_rotmat_tmp = self.rotmat_tmp[selected_pts_mask]

        # self.densification_postfix(new_xyz, new_scaling_multiplier, new_opacity_tmp, new_scales_tmp, new_rotmat_tmp)
        self.densification_postfix(new_u= new_u,new_v=new_v,new_t=new_t,new_face_id=new_face_id,
                            new_tri_vert_id=new_tri_vert_id,
                            new_scaling_multiplier=new_scaling_multiplier,
                            new_opacity_tmp=new_opacity_tmp,new_scales_tmp= new_scales_tmp,new_rotmat_tmp= new_rotmat_tmp)

    def densify_and_prune(self, human_gs_out, max_grad, min_opacity, extent, max_screen_size, add_gs_distance_threshold,remove_gs_distance_threshold, max_n_gs=None):
        grads = self.xyz_gradient_accum / self.denom
        grads[grads.isnan()] = 0.0

        self.opacity_tmp = human_gs_out['opacity']
        self.scales_tmp = human_gs_out['scales_canon']
        self.rotmat_tmp = human_gs_out['rotmat_canon']

        max_n_gs = max_n_gs if max_n_gs else self.get_xyz.shape[0] + 1

        if self.get_xyz.shape[0] <= max_n_gs:
            self.densify_and_clone(grads, max_grad, extent, add_gs_distance_threshold)
            self.densify_and_split(grads, max_grad, extent, add_gs_distance_threshold)
            
        need_prune = False
        if need_prune:
            prune_mask = (self.opacity_tmp < min_opacity).squeeze()
            
            if max_screen_size:
                big_points_vs = self.max_radii2D > max_screen_size
                big_points_ws = self.scales_tmp.max(dim=1).values > 0.1 * extent
                prune_mask = torch.logical_or(torch.logical_or(prune_mask, big_points_vs), big_points_ws)
            
            # 计算在标准姿态下，点云到目标网格的距离，如果距离大于阈值，则将该点剔除
            
            pcls = Pointclouds(points=[self.get_xyz_by_uv])
                
            points = pcls.points_packed()  # (P, 3)
            tris = self.target_mesh.verts_packed()[self.target_mesh.faces_packed()]  # (T, 3, 3)
            
            dis_point_to_face = point_face_distance(
                points,
                pcls.cloud_to_packed_first_idx(),
                tris,
                self.target_mesh.mesh_to_faces_packed_first_idx(),
                pcls.num_points_per_cloud().max().item(),
                1e-8,
            )  # 形状 (P,)
            
            distance_mask = dis_point_to_face > remove_gs_distance_threshold * remove_gs_distance_threshold
            
            prune_mask = torch.logical_or(prune_mask, distance_mask)
            
            self.prune_points(prune_mask)
        self.n_gs = self.u.shape[0]
        torch.cuda.empty_cache()

    def prune(self,human_gs_out,min_opacity,max_screen_size,extent,remove_gs_distance_threshold):
        grads = self.xyz_gradient_accum / self.denom
        grads[grads.isnan()] = 0.0

        self.opacity_tmp = human_gs_out['opacity']
        self.scales_tmp = human_gs_out['scales_canon']
        self.rotmat_tmp = human_gs_out['rotmat_canon']

        
        prune_mask = (self.opacity_tmp < min_opacity).squeeze()
        
        print("ClothGS::prune() prune_count:",prune_mask.sum())
        
        if max_screen_size:
            big_points_vs = self.max_radii2D > max_screen_size
            big_points_ws = self.scales_tmp.max(dim=1).values > 0.1 * extent
            prune_mask = torch.logical_or(torch.logical_or(prune_mask, big_points_vs), big_points_ws)
        
        # 计算在标准姿态下，点云到目标网格的距离，如果距离大于阈值，则将该点剔除
        
        pcls = Pointclouds(points=[self.get_xyz_by_uv])
            
        points = pcls.points_packed()  # (P, 3)
        tris = self.target_mesh.verts_packed()[self.target_mesh.faces_packed()]  # (T, 3, 3)
        
        dis_point_to_face = point_face_distance(
            points,
            pcls.cloud_to_packed_first_idx(),
            tris,
            self.target_mesh.mesh_to_faces_packed_first_idx(),
            pcls.num_points_per_cloud().max().item(),
            1e-8,
        )  # 形状 (P,)
        
        distance_mask = dis_point_to_face > remove_gs_distance_threshold * remove_gs_distance_threshold
        
        prune_mask = torch.logical_or(prune_mask, distance_mask)
        
        self.prune_points(prune_mask)
        self.n_gs = self.u.shape[0]
        torch.cuda.empty_cache()
    
    def add_densification_stats(self, viewspace_point_tensor, update_filter):
        self.xyz_gradient_accum[update_filter] += torch.norm(
            viewspace_point_tensor.grad[:update_filter.shape[0]][update_filter, :2], dim=-1, keepdim=True)
        self.denom[update_filter] += 1
