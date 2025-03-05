# Code adapted from 3DGS: https://github.com/graphdeco-inria/gaussian-splatting/blob/main/gaussian_renderer/__init__.py
# License from 3DGS: https://github.com/graphdeco-inria/gaussian-splatting/blob/main/LICENSE.md

#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.
#

import math
import torch
from diff_gaussian_rasterization import (
    GaussianRasterizationSettings, 
    GaussianRasterizer
)

from hugs.utils.spherical_harmonics import SH2RGB
from hugs.utils.rotations import quaternion_to_matrix


def render_human_scene(
    data, 
    human_gs_out,
    scene_gs_out,
    bg_color, 
    human_bg_color=None,
    scaling_modifier=1.0, 
    render_mode='human_scene',
    render_human_separate=False,
    upperbody_gs_out =None,
    lowerbody_gs_out =None,
    cloth_bg_color=None,
    render_cloth_separate=True,
):

    feats = None
    if render_mode == 'human_scene':
        # cat human \upperbody_gs\ lowerbody_gs scene Gaussians


        feats = torch.cat([human_gs_out['shs'], upperbody_gs_out['shs'],lowerbody_gs_out['shs'],scene_gs_out['shs']], dim=0)
        means3D = torch.cat([human_gs_out['xyz'],upperbody_gs_out['xyz'],lowerbody_gs_out['xyz'], scene_gs_out['xyz']], dim=0)
        opacity = torch.cat([human_gs_out['opacity'],upperbody_gs_out['opacity'],lowerbody_gs_out['opacity'], scene_gs_out['opacity']], dim=0)
        scales = torch.cat([human_gs_out['scales'],upperbody_gs_out['scales'],lowerbody_gs_out['scales'], scene_gs_out['scales']], dim=0)
        rotations = torch.cat([human_gs_out['rotq'],upperbody_gs_out['rotq'],lowerbody_gs_out['rotq'], scene_gs_out['rotq']], dim=0)
        active_sh_degree = human_gs_out['active_sh_degree']
    elif render_mode == 'human':
        feats = human_gs_out['shs']
        means3D = human_gs_out['xyz']
        opacity = human_gs_out['opacity']
        scales = human_gs_out['scales']
        rotations = human_gs_out['rotq']
        active_sh_degree = human_gs_out['active_sh_degree']
    elif render_mode == 'scene':
        feats = scene_gs_out['shs']
        means3D = scene_gs_out['xyz']
        opacity = scene_gs_out['opacity']
        scales = scene_gs_out['scales']
        rotations = scene_gs_out['rotq']
        active_sh_degree = scene_gs_out['active_sh_degree']
    elif render_mode == 'upperbody':
        feats = upperbody_gs_out['shs']
        means3D = upperbody_gs_out['xyz']
        opacity = upperbody_gs_out['opacity']
        scales = upperbody_gs_out['scales']
        rotations = upperbody_gs_out['rotq']
        active_sh_degree = upperbody_gs_out['active_sh_degree']
    elif render_mode == 'lowerbody':
        feats = lowerbody_gs_out['shs']
        means3D = lowerbody_gs_out['xyz']
        opacity = lowerbody_gs_out['opacity']
        scales = lowerbody_gs_out['scales']
        rotations = lowerbody_gs_out['rotq']
        active_sh_degree = lowerbody_gs_out['active_sh_degree']
    else:
        raise ValueError(f'Unknown render mode: {render_mode}')
    
    render_pkg = render(
        means3D=means3D,
        feats=feats,
        opacity=opacity,
        scales=scales,
        rotations=rotations,
        data=data,
        scaling_modifier=scaling_modifier,
        bg_color=bg_color,
        active_sh_degree=active_sh_degree,
    )


    if render_cloth_separate == False and render_mode == 'human_scene':
        human_with_cloth_feats = torch.cat([human_gs_out['shs'], upperbody_gs_out['shs'],lowerbody_gs_out['shs']], dim=0)
        human_with_cloth_means3D = torch.cat([human_gs_out['xyz'],upperbody_gs_out['xyz'],lowerbody_gs_out['xyz']], dim=0)
        human_with_cloth_opacity = torch.cat([human_gs_out['opacity'],upperbody_gs_out['opacity'],lowerbody_gs_out['opacity']], dim=0)
        human_with_cloth_scales = torch.cat([human_gs_out['scales'],upperbody_gs_out['scales'],lowerbody_gs_out['scales']], dim=0)
        human_with_cloth_rotations = torch.cat([human_gs_out['rotq'],upperbody_gs_out['rotq'],lowerbody_gs_out['rotq']], dim=0)
        human_with_cloth_active_sh_degree = human_gs_out['active_sh_degree']


        render_human_with_cloth_pkg = render(
            means3D=human_with_cloth_means3D,
            feats=human_with_cloth_feats,
            opacity=human_with_cloth_opacity,
            scales=human_with_cloth_scales,
            rotations=human_with_cloth_rotations,
            data=data,
            scaling_modifier=scaling_modifier,
            bg_color=human_bg_color if human_bg_color is not None else bg_color,
            active_sh_degree=human_with_cloth_active_sh_degree,
        )

        render_pkg['human_with_cloth_img'] = render_human_with_cloth_pkg['render']
        render_pkg['human_with_cloth_visibility_filter'] = render_human_with_cloth_pkg['visibility_filter']
        render_pkg['human_with_cloth_radii'] = render_human_with_cloth_pkg['radii']

        
    if render_human_separate and render_cloth_separate is True and render_mode == 'human_scene':
        render_human_pkg = render(
            means3D=human_gs_out['xyz'],
            feats=human_gs_out['shs'],
            opacity=human_gs_out['opacity'],
            scales=human_gs_out['scales'],
            rotations=human_gs_out['rotq'],
            data=data,
            scaling_modifier=scaling_modifier,
            bg_color=human_bg_color if human_bg_color is not None else bg_color,
            active_sh_degree=human_gs_out['active_sh_degree'],
        )
        render_pkg['human_img'] = render_human_pkg['render']
        render_pkg['human_visibility_filter'] = render_human_pkg['visibility_filter']
        render_pkg['human_radii'] = render_human_pkg['radii']

    if upperbody_gs_out != None and render_cloth_separate and render_mode == 'human_scene':
        render_cloth_pkg = render(
            means3D=upperbody_gs_out['xyz'],
            feats=upperbody_gs_out['shs'],
            opacity=upperbody_gs_out['opacity'],
            scales=upperbody_gs_out['scales'],
            rotations=upperbody_gs_out['rotq'],
            data=data,
            scaling_modifier=scaling_modifier,
            bg_color=cloth_bg_color if cloth_bg_color is not None else bg_color,
            active_sh_degree=upperbody_gs_out['active_sh_degree'],
        )
        render_pkg['upperbody_img'] = render_cloth_pkg['render']
        render_pkg['upperbody_viewspace_points'] = render_cloth_pkg['viewspace_points']
        render_pkg['upperbody_visibility_filter'] = render_cloth_pkg['visibility_filter']
        render_pkg['upperbody_radii'] = render_cloth_pkg['radii']

    if lowerbody_gs_out != None and render_cloth_separate and render_mode == 'human_scene':
        render_cloth_pkg = render(
            means3D=lowerbody_gs_out['xyz'],
            feats=lowerbody_gs_out['shs'],
            opacity=lowerbody_gs_out['opacity'],
            scales=lowerbody_gs_out['scales'],
            rotations=lowerbody_gs_out['rotq'],
            data=data,
            scaling_modifier=scaling_modifier,
            bg_color=cloth_bg_color if cloth_bg_color is not None else bg_color,
            active_sh_degree=lowerbody_gs_out['active_sh_degree'],
        )
        render_pkg['lowerbody_img'] = render_cloth_pkg['render']
        render_pkg['lowerbody_viewspace_points'] = render_cloth_pkg['viewspace_points']
        render_pkg['lowerbody_visibility_filter'] = render_cloth_pkg['visibility_filter']
        render_pkg['lowerbody_radii'] = render_cloth_pkg['radii']
        
    if render_mode == 'human':
        render_pkg['human_visibility_filter'] = render_pkg['visibility_filter']
        render_pkg['human_radii'] = render_pkg['radii']
    elif render_mode == 'upperbody':
        render_pkg['upperbody_visibility_filter'] = render_pkg['visibility_filter']
        render_pkg['upperbody_viewspace_points'] = render_pkg['viewspace_points']
        render_pkg['upperbody_radii'] = render_pkg['radii']
    elif render_mode == 'lowerbody':
        render_pkg['lowerbody_visibility_filter'] = render_pkg['visibility_filter']
        render_pkg['lowerbody_viewspace_points'] = render_pkg['viewspace_points']
        render_pkg['lowerbody_radii'] = render_pkg
    elif render_mode == 'human_scene':
        human_n_gs = human_gs_out['xyz'].shape[0]
        upperbody_n_gs = upperbody_gs_out['xyz'].shape[0]
        lowerbody_n_gs = lowerbody_gs_out['xyz'].shape[0]
        scene_n_gs = scene_gs_out['xyz'].shape[0]
        render_pkg['scene_visibility_filter'] = render_pkg['visibility_filter'][human_n_gs+upperbody_n_gs+lowerbody_n_gs:]
        render_pkg['scene_radii'] = render_pkg['radii'][human_n_gs+upperbody_n_gs+lowerbody_n_gs:]
        if not 'human_visibility_filter' in render_pkg.keys():
            render_pkg['human_visibility_filter'] = render_pkg['visibility_filter'][:-scene_n_gs]
            render_pkg['human_radii'] = render_pkg['radii'][:-scene_n_gs]
            
    elif render_mode == 'scene':
        render_pkg['scene_visibility_filter'] = render_pkg['visibility_filter']
        render_pkg['scene_radii'] = render_pkg['radii']
        
    return render_pkg
    
    
def render(means3D, feats, opacity, scales, rotations, data, scaling_modifier=1.0, bg_color=None, active_sh_degree=0):
    if bg_color is None:
        bg_color = torch.zeros(3, dtype=torch.float32, device="cuda")
        
    screenspace_points = torch.zeros_like(means3D, dtype=means3D.dtype, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except Exception as e:
        pass

    means2D = screenspace_points

    # Set up rasterization configuration
    tanfovx = math.tan(data['fovx'] * 0.5)
    tanfovy = math.tan(data['fovy'] * 0.5)

    shs, rgb = None, None
    if len(feats.shape) == 2:
        rgb = feats
    else:
        shs = feats

    
    raster_settings = GaussianRasterizationSettings(
        image_height=int(data['image_height']),
        image_width=int(data['image_width']),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=data['world_view_transform'],
        projmatrix=data['full_proj_transform'],
        sh_degree=active_sh_degree,
        campos=data['camera_center'],
        prefiltered=False,
        debug=False,
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)
        
    # Rasterize visible Gaussians to image, obtain their radii (on screen). 
    rendered_image, radii = rasterizer(
        means3D=means3D,
        means2D=means2D,
        shs=shs,
        opacities=opacity,
        scales=scales,
        rotations=rotations,
        colors_precomp=rgb,
    )
    rendered_image = torch.clamp(rendered_image, 0.0, 1.0)
    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.
    return {
        "render": rendered_image,
        "viewspace_points": screenspace_points,
        "visibility_filter" : radii > 0,
        "radii": radii,
    }
