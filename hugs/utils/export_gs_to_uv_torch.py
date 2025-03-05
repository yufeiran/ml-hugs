import torch
import torch.nn.functional as F
from pytorch3d.structures import Meshes
from pytorch3d.ops import sample_points_from_meshes
from pytorch3d.io import load_obj

# ------------------------------
# 1. 加载SMPL Mesh和UV数据
# ------------------------------
# 假设已从SMPL模型中加载顶点、面片和UV坐标
# verts: [N_verts, 3], faces: [N_faces, 3], uv_coords: [N_verts, 2]
# 例如通过load_obj加载：
# verts, faces, aux = load_obj("smpl_mesh.obj")
# uv_coords = aux.verts_uvs  # [N_verts, 2]

# 示例数据（需替换为实际SMPL数据）
verts = torch.randn(6890, 3).cuda()  # 假设SMPL顶点
faces = torch.randint(0, 6890, (13776, 3)).cuda()  # 假设SMPL面片
uv_coords = torch.rand(6890, 2).cuda()  # 假设UV坐标

# 转换为PyTorch3D的Meshes对象
smpl_mesh = Meshes(verts=[verts], faces=[faces])

# ------------------------------
# 2. 生成3D高斯点及颜色（示例数据）
# ------------------------------
num_gs_points = 100000
gs_points = torch.randn(num_gs_points, 3).cuda()  # 3D高斯点位置
gs_colors = torch.rand(num_gs_points, 3).cuda()    # 颜色值（RGB）

# ------------------------------
# 3. 查询每个高斯点的最近面片和重心坐标
# ------------------------------
# 使用PyTorch3D的最近面片查询
# 返回值: 面片索引, 权重（重心坐标）
_, face_idx, bary_coords = sample_points_from_meshes(
    smpl_mesh, gs_points.unsqueeze(0), return_normals=False, return_textures=False
)
face_idx = face_idx.squeeze(0)  # [num_gs_points]
bary_coords = bary_coords.squeeze(0)  # [num_gs_points, 3]

# ------------------------------
# 4. 计算UV坐标（通过重心坐标插值）
# ------------------------------
# 获取面片对应的三个顶点的UV坐标
face_uvs = uv_coords[faces[face_idx]]  # [num_gs_points, 3, 2]

# 使用重心坐标插值UV坐标
uvs = torch.sum(face_uvs * bary_coords.unsqueeze(-1), dim=1)  # [num_gs_points, 2]

# ------------------------------
# 5. 将颜色投影到UV图（GPU加速）
# ------------------------------
# 定义UV图分辨率（例如512x512）
uv_resolution = 512
uv_map = torch.zeros((uv_resolution, uv_resolution, 3), device="cuda")

# 将UV坐标归一化到[0, 1]并缩放到分辨率
uv_pixels = (uvs * (uv_resolution - 1)).long()  # [num_gs_points, 2]

# 使用原子操作避免竞争（可能需要自定义CUDA内核）
# 这里简化处理：直接累加颜色（后续归一化）
for i in range(num_gs_points):
    u, v = uv_pixels[i]
    uv_map[v, u] += gs_colors[i]

# ------------------------------
# 6. 归一化处理（可选）
# ------------------------------
# 统计每个像素被命中的次数，计算平均颜色
count_map = torch.zeros_like(uv_map)
for i in range(num_gs_points):
    u, v = uv_pixels[i]
    count_map[v, u] += 1

uv_map = uv_map / (count_map + 1e-8)  # 避免除以零

# 转换为CPU并保存
uv_map_cpu = uv_map.cpu().numpy()
# 保存为图片（例如用OpenCV）
# import cv2
# cv2.imwrite("uv_color.png", uv_map_cpu * 255)