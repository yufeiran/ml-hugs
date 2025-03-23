import torch
from torch.nn.utils.rnn import pad_sequence

def axis_angle_to_rotation_matrix(axis_angle):
    """
    将轴角表示转换为旋转矩阵。
    axis_angle: (3,) 的 tensor，表示旋转轴乘以旋转角度（弧度）。
    当角度趋近于0时返回单位矩阵。
    """
    angle = torch.norm(axis_angle)
    if angle < 1e-8:
        return torch.eye(3, device=axis_angle.device, dtype=axis_angle.dtype)
    axis = axis_angle / angle
    x, y, z = axis[0], axis[1], axis[2]
    cos_val = torch.cos(angle)
    sin_val = torch.sin(angle)
    one_minus_cos = 1 - cos_val
    R = torch.tensor([
        [cos_val + x*x*one_minus_cos,    x*y*one_minus_cos - z*sin_val, x*z*one_minus_cos + y*sin_val],
        [y*x*one_minus_cos + z*sin_val,    cos_val + y*y*one_minus_cos,   y*z*one_minus_cos - x*sin_val],
        [z*x*one_minus_cos - y*sin_val,    z*y*one_minus_cos + x*sin_val, cos_val + z*z*one_minus_cos]
    ], device=axis_angle.device, dtype=axis_angle.dtype)
    return R

def matrix_to_quaternion(R):
    """
    将批量旋转矩阵 R (N, 3, 3) 转换为四元数表示，返回 (N, 4) 张量，四元数顺序为 (w, x, y, z)。
    """
    # R: (N, 3, 3)
    N = R.shape[0]
    q = torch.empty((N, 4), device=R.device, dtype=R.dtype)
    
    # 计算迹
    trace = R[:, 0, 0] + R[:, 1, 1] + R[:, 2, 2]
    # 条件：trace > 0
    cond = trace > 0
    if cond.any():
        S = torch.sqrt(trace[cond] + 1.0) * 2.0  # S shape: (n_cond,)
        q[cond, 0] = 0.25 * S                     # qw
        q[cond, 1] = (R[cond, 2, 1] - R[cond, 1, 2]) / S  # qx
        q[cond, 2] = (R[cond, 0, 2] - R[cond, 2, 0]) / S  # qy
        q[cond, 3] = (R[cond, 1, 0] - R[cond, 0, 1]) / S  # qz

    # 对于 trace <= 0 的情况，分3种情况处理：
    cond_not = ~cond
    if cond_not.any():
        # 提取对应的 R
        R_not = R[cond_not]  # (n_rem, 3, 3)
        diag0 = R_not[:, 0, 0]
        diag1 = R_not[:, 1, 1]
        diag2 = R_not[:, 2, 2]

        # 为了后面取全局索引
        not_idx = torch.nonzero(cond_not, as_tuple=False).squeeze(1)

        # 情况1：R[0,0] 最大
        cond0 = (diag0 >= diag1) & (diag0 >= diag2)
        if cond0.any():
            idx0 = torch.nonzero(cond0, as_tuple=False).squeeze(1)
            S = torch.sqrt(1.0 + diag0[idx0] - diag1[idx0] - diag2[idx0]) * 2.0
            global_idx0 = not_idx[idx0]
            q[global_idx0, 0] = (R[global_idx0, 2, 1] - R[global_idx0, 1, 2]) / S
            q[global_idx0, 1] = 0.25 * S
            q[global_idx0, 2] = (R[global_idx0, 0, 1] + R[global_idx0, 1, 0]) / S
            q[global_idx0, 3] = (R[global_idx0, 0, 2] + R[global_idx0, 2, 0]) / S

        # 情况2：R[1,1] 最大
        cond1 = (diag1 >= diag0) & (diag1 >= diag2)
        if cond1.any():
            idx1 = torch.nonzero(cond1, as_tuple=False).squeeze(1)
            S = torch.sqrt(1.0 + diag1[idx1] - diag0[idx1] - diag2[idx1]) * 2.0
            global_idx1 = not_idx[idx1]
            q[global_idx1, 0] = (R[global_idx1, 0, 2] - R[global_idx1, 2, 0]) / S
            q[global_idx1, 1] = (R[global_idx1, 0, 1] + R[global_idx1, 1, 0]) / S
            q[global_idx1, 2] = 0.25 * S
            q[global_idx1, 3] = (R[global_idx1, 1, 2] + R[global_idx1, 2, 1]) / S

        # 情况3：R[2,2] 最大
        cond2 = (diag2 >= diag0) & (diag2 >= diag1)
        if cond2.any():
            idx2 = torch.nonzero(cond2, as_tuple=False).squeeze(1)
            S = torch.sqrt(1.0 + diag2[idx2] - diag0[idx2] - diag1[idx2]) * 2.0
            global_idx2 = not_idx[idx2]
            q[global_idx2, 0] = (R[global_idx2, 1, 0] - R[global_idx2, 0, 1]) / S
            q[global_idx2, 1] = (R[global_idx2, 0, 2] + R[global_idx2, 2, 0]) / S
            q[global_idx2, 2] = (R[global_idx2, 1, 2] + R[global_idx2, 2, 1]) / S
            q[global_idx2, 3] = 0.25 * S

    return q

def build_neighbor_list(num_vertices, faces):
    """
    构建每个顶点的邻域列表（返回 Python list，每个元素为 torch.LongTensor）。
    """
    neighbor_list = [set() for _ in range(num_vertices)]
    for face in faces:
        i0, i1, i2 = face.tolist()
        neighbor_list[i0].update([i1, i2])
        neighbor_list[i1].update([i0, i2])
        neighbor_list[i2].update([i0, i1])
    neighbor_list = [torch.tensor(list(nbr), dtype=torch.long, device=faces.device) for nbr in neighbor_list]
    return neighbor_list

def compute_vertex_normals_torch(vertices, faces):
    """
    使用面法向累加计算每个顶点的法向（已在 GPU 上实现）。
    """
    v0 = vertices[faces[:, 0]]
    v1 = vertices[faces[:, 1]]
    v2 = vertices[faces[:, 2]]
    face_normals = torch.cross(v1 - v0, v2 - v0, dim=1)
    face_norms = face_normals.norm(dim=1, keepdim=True)
    face_normals = face_normals / (face_norms + 1e-8)
    
    vertex_normals = torch.zeros_like(vertices)
    vertex_normals = vertex_normals.index_add(0, faces[:, 0], face_normals)
    vertex_normals = vertex_normals.index_add(0, faces[:, 1], face_normals)
    vertex_normals = vertex_normals.index_add(0, faces[:, 2], face_normals)
    vertex_normals = vertex_normals / (vertex_normals.norm(dim=1, keepdim=True) + 1e-8)
    return vertex_normals

def compute_all_local_frames_batched(vertices, faces,theta,neighbor_list,padded_indices,mask, scale_normal=0.01):
    """
    批量计算整个 mesh 所有顶点的局部标架，主要步骤：
      1. 利用邻域列表构造一个填充后的张量（mask 标记有效邻域）。
      2. 计算每个顶点邻域点相对于顶点的差值，并投影到顶点法平面上。
      3. 计算每个顶点的 3x3 协方差矩阵，并批量求解特征值分解。
      4. 利用特征向量（对应最大和次大特征值）与顶点法向构造局部标架。
    
    参数：
      vertices: (N, 3) 张量，顶点坐标
      faces: (F, 3) 张量，面索引（dtype=torch.long）
      scale_normal: 法向方向的尺度
    返回：
      R_all: (N, 3, 3) 每个顶点的局部旋转矩阵
      scales_all: (N, 3) 每个顶点在三个方向的尺度
    """
    num_vertices = vertices.shape[0]
    device = vertices.device
    
    rotated_vertices = vertices
    
    if theta is not None:
        
        # 1. 将输入的 theta 转换为 tensor，并用轴角转换为旋转矩阵
        if not torch.is_tensor(theta):
            theta = torch.tensor(theta, device=device, dtype=vertices.dtype)
        else:
            theta = theta.to(device)
        R_theta = axis_angle_to_rotation_matrix(theta)  # (3, 3)

        # 2. 将输入顶点绕原点旋转
        rotated_vertices = vertices @ R_theta.T

    # 预先构造邻域列表（可缓存）
    #neighbor_list = build_neighbor_list(num_vertices, faces)
    
    # 将变长的邻域列表填充为 (N, max_neighbors) 的张量，同时生成 mask
    # 假设 neighbor_list 是一个长度为 N 的列表，每个元素是形如 (n_i,) 的 LongTensor
    # 使用 pad_sequence 填充，padding_value 设为 -1
    # padded_indices = pad_sequence(neighbor_list, batch_first=True, padding_value=-1)  # (N, max_neighbors)
    # mask = padded_indices != -1

    # 取出每个顶点的邻域坐标，注意无效位置暂时为 -1，所以 clamp(-1 -> 0) 后再用 mask 过滤
    neighbor_coords = rotated_vertices[padded_indices.clamp(min=0)]
    # 将无效数据置零
    neighbor_coords = neighbor_coords * mask.unsqueeze(-1).float()


    # 计算每个顶点与其邻域点的相对坐标
    diff = neighbor_coords - rotated_vertices.unsqueeze(1)  # (N, max_neighbors, 3)

    # 计算所有顶点的法向
    vertex_normals = compute_vertex_normals_torch(rotated_vertices, faces)  # (N, 3)

    # 将 diff 投影到法平面上：diff_proj = diff - (diff·n) * n
    dots = (diff * vertex_normals.unsqueeze(1)).sum(dim=-1, keepdim=True)  # (N, max_neighbors, 1)
    projected = diff - dots * vertex_normals.unsqueeze(1)  # (N, max_neighbors, 3)
    # 无效位置置零（已由 mask 保证）

    # 计算每个顶点的协方差矩阵：cov = sum_j( x_j * x_j^T ) / (num_valid)
    projected = projected * mask.unsqueeze(-1).float()  # (N, max_neighbors, 3)
    # 计算每个顶点的协方差矩阵：利用投影后的邻域点（已填充的 projected，形状 (N, max_neighbors, 3)）
    # 这里我们对邻域（m 维度）求和，得到每个顶点的 3x3 协方差矩阵
    cov = torch.einsum("nmi,nmj->nij", projected, projected)  # 结果 shape: (N, 3, 3)
    valid_counts = mask.sum(dim=1).view(num_vertices, 1, 1).float()  # (N,1,1)
    cov = cov / (valid_counts + 1e-8)

    # 对每个 3x3 协方差矩阵批量求解特征值分解（eigh 返回升序排列的特征值）
    eigvals, eigvecs = torch.linalg.eigh(cov)  # eigvals: (N, 3), eigvecs: (N, 3, 3)
    # 取出对应最大（最后一列）和次大（倒数第二列）的特征向量作为切向量
    axis1 = eigvecs[:, :, 2]  # (N, 3)
    axis2 = eigvecs[:, :, 1]  # (N, 3)

    # 确保构成右手系：如果 axis1 x axis2 与 vertex_normal 方向不一致，则反转 axis2
    cross = torch.cross(axis1, axis2, dim=1)  # (N, 3)
    flip_mask = (torch.sum(cross * vertex_normals, dim=1) < 0).unsqueeze(-1)  # (N, 1)
    axis2 = torch.where(flip_mask, -axis2, axis2)

    # 构造局部旋转矩阵：将 axis1, axis2, vertex_normals 作为列向量
    R_all = torch.stack([axis1, axis2, vertex_normals], dim=2)  # (N, 3, 3)
    
    # 将旋转矩阵转换为四元数表示
    quat_all = matrix_to_quaternion(R_all)  # (N, 4)


    # 构造尺度：对切向使用特征值的平方根（可以视具体需求调节），法向使用 scale_normal
    scales_all = torch.stack([
        torch.sqrt(eigvals[:, 2] + 1e-8),
        torch.sqrt(eigvals[:, 1] + 1e-8),
        torch.full((num_vertices,), scale_normal, device=device)
    ], dim=1)  # (N, 3)

    scales_all *=0.3

    return R_all, scales_all

def compute_all_local_frames(vertices,faces,global_orient=None):
    faces=faces.long()
    neighbor_list = build_neighbor_list(vertices.shape[0], faces)
    padded_indices = pad_sequence(neighbor_list, batch_first=True, padding_value=-1)  # (N, max_neighbors)
    padded_indices_mask = padded_indices != -1
    
    R_all, scales_all = compute_all_local_frames_batched(vertices, faces,global_orient,neighbor_list,padded_indices,padded_indices_mask, scale_normal=0.01)

    return R_all, scales_all

# 示例运行
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 示例顶点和面（请替换为实际数据）
    vertices = torch.tensor([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [1.0, 1.0, 0.0],
    ], device=device)
    faces = torch.tensor([
        [0, 1, 2],
        [1, 3, 2]
    ], dtype=torch.long, device=device)
    
    neighbor_list = build_neighbor_list(vertices.shape[0], faces)
    padded_indices = pad_sequence(neighbor_list, batch_first=True, padding_value=-1)  # (N, max_neighbors)
    padded_indices_mask = padded_indices != -1
    
    R_all, scales_all = compute_all_local_frames_batched(vertices, faces,None,neighbor_list,padded_indices,padded_indices_mask, scale_normal=0.01)

    R_all, scales_all = compute_all_local_frames(vertices, faces,None)

    for i in range(vertices.shape[0]):
        print(f"顶点 {i}:")
        print("局部标架旋转矩阵 R:")
        print(R_all[i])
        print("局部标架缩放因子:")
        print(scales_all[i])
        print("------------")
