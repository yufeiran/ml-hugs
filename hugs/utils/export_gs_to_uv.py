import trimesh
import numpy as np
import torch
from scipy.spatial.transform import Rotation
import trimesh
from scipy.spatial import KDTree
import cv2

SAVE_PATH = './results/smpl_uv_result/'

def find_closest_surface_point(
    point: np.ndarray, 
    mesh: trimesh.Trimesh, 
    direction=None, 
    use_raycasting=True
) -> (np.ndarray, int):
    """
    找到点投影到mesh表面的最近位置及其面片索引
    Args:
        point: 输入点坐标 [3]
        mesh: SMPL mesh对象
        direction: 投影方向（若为None，则使用mesh顶点法线方向）
        use_raycasting: 是否优先用射线投射（否则用最近面片）
    Returns:
        closest_point: 表面最近点 [3]
        face_idx: 面片索引
    """
    if use_raycasting:
        # 方法1：沿法线方向射线投射（需预计算顶点法线）
        if direction is None:
            # 获取最近的顶点法线（简化处理，实际应插值面片法线）
            closest_vertex = mesh.vertices[np.argmin(np.linalg.norm(mesh.vertices - point, axis=1))]
            direction = mesh.vertex_normals[np.argmin(np.linalg.norm(mesh.vertices - point, axis=1))]
        
        # 发射射线：从点沿法线反方向投射
        ray_origin = point + direction * 1e-3  # 避免自交
        ray_direction = -direction
        locations, _, face_indices = mesh.ray.intersects_location(
            [ray_origin], [ray_direction]
        )
        if len(locations) > 0:
            # 选择第一个交点
            closest_point = locations[0]
            face_idx = face_indices[0]
            return closest_point, face_idx
    
    # 方法2：直接找最近面片（若射线投射失败）
    # 计算所有面片的距离并找到最近的面片
    from trimesh.proximity import closest_point
    closest_point, distance, face_idx = closest_point(mesh, np.array([point]))
    return closest_point[0], face_idx[0]

def compute_tbn(mesh: trimesh.Trimesh, face_idx: int) -> (np.ndarray, np.ndarray, np.ndarray):
    """
    计算面片的TBN矩阵（切向量T, 副切向量B, 法线N）
    Args:
        mesh: 包含顶点、面片、UV的mesh
        face_idx: 面片索引
    Returns:
        T: 切向量 [3]
        B: 副切向量 [3]
        N: 法线向量 [3]
    """
    # 获取面片的三个顶点坐标和UV坐标
    face = mesh.faces[face_idx]
    v0, v1, v2 = mesh.vertices[face]
    uv0, uv1, uv2 = mesh.visual.uv[face]
    
    # 计算边向量和UV差异
    edge1 = v1 - v0
    edge2 = v2 - v0
    delta_uv1 = uv1 - uv0
    delta_uv2 = uv2 - uv0
    
    # 解线性方程组计算T和B
    # [delta_uv1.x, delta_uv1.y] [T] = edge1
    # [delta_uv2.x, delta_uv2.y] [B]   edge2
    # 使用最小二乘法求解
    A = np.array([delta_uv1, delta_uv2]).T  # [2, 2]
    B = np.array([edge1, edge2]).T           # [3, 2]
    try:
        inv_A = np.linalg.inv(A)
        T, B_vec = (B @ inv_A).T  # [3, 2]
    except np.linalg.LinAlgError:
        # 行列式为零时退化为正交化
        T = edge1 / np.linalg.norm(edge1)
        B_vec = np.cross(edge2, T)
        B_vec /= np.linalg.norm(B_vec)
    
    # 法线向量（归一化）
    N = np.cross(edge1, edge2)
    N /= np.linalg.norm(N)
    
    # 正交化T和B（Gram-Schmidt）
    T = T - np.dot(T, N) * N
    T /= np.linalg.norm(T)
    B_vec = np.cross(N, T)
    
    return T, B_vec, N

def get_uv(mesh: trimesh.Trimesh, point: np.ndarray, face_idx: int) -> np.ndarray:
    """
    通过重心坐标插值得到表面点的UV坐标
    Args:
        mesh: 包含UV的mesh
        point: 表面点坐标 [3]
        face_idx: 面片索引
    Returns:
        uv: 插值后的UV坐标 [2]
    """
    face = mesh.faces[face_idx]
    v0, v1, v2 = mesh.vertices[face]
    uv0, uv1, uv2 = mesh.visual.uv[face]
    
    # 计算重心坐标
    vec0 = v1 - v0
    vec1 = v2 - v0
    vec_p = point - v0
    det = np.dot(vec0, vec0) * np.dot(vec1, vec1) - np.dot(vec0, vec1)**2
    if det == 0:
        return uv0  # 退化为第一个顶点
    
    u = (np.dot(vec_p, vec0) * np.dot(vec1, vec1) - np.dot(vec_p, vec1) * np.dot(vec0, vec1)) / det
    v = (np.dot(vec_p, vec1) * np.dot(vec0, vec0) - np.dot(vec_p, vec0) * np.dot(vec0, vec1)) / det
    w = 1 - u - v
    
    # 插值UV
    uv = w * uv0 + u * uv1 + v * uv2
    return uv

def eval_sh(degree, dir):
    """计算三阶球谐基函数值"""
    x, y, z = dir / (np.linalg.norm(dir) + 1e-8)
    result = []
    # l=0
    result.append(0.5 * np.sqrt(1.0 / np.pi))
    # l=1
    result.append(np.sqrt(3/(4*np.pi)) * y)
    result.append(np.sqrt(3/(4*np.pi)) * z)
    result.append(np.sqrt(3/(4*np.pi)) * x)
    # l=2
    result.append(0.5 * np.sqrt(15/np.pi) * x * y)
    result.append(0.5 * np.sqrt(15/np.pi) * y * z)
    result.append(0.25 * np.sqrt(5/np.pi) * (3*z**2-1))
    result.append(0.5 * np.sqrt(15/np.pi) * x * z)
    result.append(0.25 * np.sqrt(15/np.pi) * (x**2 - y**2))
    # l=3
    result.append( 0.25 * np.sqrt(35/(2*np.pi))  * y * (3*x**2 - y**2))
    result.append( 0.5  * np.sqrt(105/np.pi)     * x*y*z)
    result.append( 0.25 * np.sqrt(21/(2*np.pi))  * y * (4*z**2 - x**2 - y**2))
    result.append( 0.25 * np.sqrt(7/np.pi)       * z * (2*z**2 - 3*x**2 - 3*y**2))
    result.append( 0.25 * np.sqrt(21/(2*np.pi))  * x * (4*z**2 - x**2 - y**2))
    result.append( 0.25 * np.sqrt(105/np.pi)     * z * (x**2 - y**2))
    result.append( 0.25 * np.sqrt(35/(2*np.pi))   * x * (x**2 - 3*y**2))
    
    return np.array(result)

def eval_sh_basis(view_dir):
    """
    计算 4 阶（0 阶到 3 阶，共 16 个基函数）实球谐基函数值。
    输入：
        view_dir: 包含 (x, y, z) 的向量，要求为归一化的单位向量。
    输出：
        一个 shape 为 (16,) 的 numpy 数组，顺序为：
        [Y_{0,0},
         Y_{1,-1}, Y_{1,0}, Y_{1,1},
         Y_{2,-2}, Y_{2,-1}, Y_{2,0}, Y_{2,1}, Y_{2,2},
         Y_{3,-3}, Y_{3,-2}, Y_{3,-1}, Y_{3,0}, Y_{3,1}, Y_{3,2}, Y_{3,3}]
    """
    x, y, z = view_dir
    basis = np.empty(16)
    
    # 0阶
    basis[0] = 0.282095
    
    # 1阶
    basis[1] = 0.488603 * y
    basis[2] = 0.488603 * z
    basis[3] = 0.488603 * x
    
    # 2阶
    basis[4] = 1.092548 * x * y
    basis[5] = 1.092548 * y * z
    basis[6] = 0.315392 * (3 * z**2 - 1)
    basis[7] = 1.092548 * x * z
    basis[8] = 0.546274 * (x**2 - y**2)
    
    # 3阶
    basis[9]  = 0.590044 * y * (3*x**2 - y**2)
    basis[10] = 2.890611 * x * y * z
    basis[11] = 0.457046 * y * (5*z**2 - 1)
    basis[12] = 0.373176 * (5*z**3 - 3*z)
    basis[13] = 0.457046 * x * (5*z**2 - 1)
    basis[14] = 1.445306 * z * (x**2 - y**2)
    basis[15] = 0.590044 * x * (x**2 - 3*y**2)
    
    return basis

def sh_color(view_dir, sh_coeffs):
    """
    计算给定视角方向下的 RGB 颜色。
    输入：
        view_dir: 一个包含 (x, y, z) 的向量或列表，需归一化为单位向量。
        sh_coeffs: 一个 shape 为 (16, 3) 的 numpy 数组，每行为对应球谐基的 RGB 系数。
    输出：
        一个长度为 3 的 numpy 数组，表示计算得到的 RGB 颜色（注意：可能会包含负值，
        渲染器通常会对最终颜色进行 clamp、归一化、曝光和 gamma 校正等处理）。
    """
    view_dir = np.array(view_dir, dtype=np.float32)
    view_dir = view_dir / np.linalg.norm(view_dir)
    
    basis = eval_sh_basis(view_dir)
    # 对每个颜色通道做点积，得到 RGB 颜色
    color = np.dot(basis, sh_coeffs)  # sh_coeffs 的 shape 应为 (16, 3)
    return color

def splat_elliptical_gaussian(
    uv_center: np.ndarray, 
    Sigma_2D: np.ndarray, 
    normal: np.ndarray,      # 当前高斯点的法线方向
    uv_image: np.ndarray, 
    weight_buffer: np.ndarray, 
    opacity: float = 1.0,
    uv_size=(1024, 1024), 
    radius=3,
    color: np.ndarray = None, 
    sh_coeffs: np.ndarray = None,  # [3, num_coeffs]
):
    # 计算椭圆参数：特征值分解确定覆盖范围
    eigvals, eigvecs = np.linalg.eigh(Sigma_2D)
    angle = np.degrees(np.arctan2(eigvecs[1, 0], eigvecs[0, 0]))
    std_dev = np.sqrt(eigvals)
    major_axis = std_dev[1] * radius
    minor_axis = std_dev[0] * radius
    
    # 生成椭圆覆盖的像素范围（简化：矩形包围盒）
    u_center = uv_center[0] * uv_size[0]
    v_center = (1 - uv_center[1]) * uv_size[1]  # UV坐标系转换为图像坐标系
    u_min = int(np.clip(u_center - major_axis, 0, uv_size[0] - 1))
    u_max = int(np.clip(u_center + major_axis, 0, uv_size[0] - 1))
    v_min = int(np.clip(v_center - major_axis, 0, uv_size[1] - 1))
    v_max = int(np.clip(v_center + major_axis, 0, uv_size[1] - 1))
    
    if sh_coeffs is not None:
        # 计算SH颜色
        Y = eval_sh(3, normal)  # 三阶SH
        color = np.array([
            np.dot(sh_coeffs[0], Y),
            np.dot(sh_coeffs[1], Y),
            np.dot(sh_coeffs[2], Y)
        ])
        # clamp color
        
        color = 1 / (1 + np.exp(-color))  # Sigmoid激活
        color = np.clip(color, 0, 1)
        
        t_sh_coeffs = sh_coeffs.transpose()
        new_color = sh_color(normal, t_sh_coeffs)
        new_color = np.clip(new_color, 0, 1)
    
    # 遍历椭圆覆盖的像素
    for u_pixel in range(u_min, u_max + 1):
        for v_pixel in range(v_min, v_max + 1):
            # 计算像素中心在UV坐标系中的位置
            u = (u_pixel + 0.5) / uv_size[0]
            v = (v_pixel + 0.5) / uv_size[1]
            delta = np.array([u - uv_center[0], (1 - v) - uv_center[1]])  # 转换到UV空间
            # 计算马氏距离
            inv_Sigma = np.linalg.inv(Sigma_2D)
            distance = delta.T @ inv_Sigma @ delta
            if distance > radius**2:
                continue
            # 计算高斯权重
            weight = np.exp(-0.5 * distance)
            # 累加颜色和权重
            uv_image[v_pixel, u_pixel] += new_color * weight * opacity
            weight_buffer[v_pixel, u_pixel] += weight
            
def compute_jacobian_with_scaling(mesh, face_idx):
    # 计算原始雅可比矩阵 J
    T, B, N = compute_tbn(mesh, face_idx)
    J = np.vstack([T, B])
    
    # 计算面积缩放因子
    face = mesh.faces[face_idx]
    v0, v1, v2 = mesh.vertices[face]
    uv0, uv1, uv2 = mesh.visual.uv[face]
    
    # 物体空间三角形面积
    obj_area = np.linalg.norm(np.cross(v1 - v0, v2 - v0)) / 2
    # UV空间三角形面积
    uv_area = np.abs( (uv1[0]-uv0[0])*(uv2[1]-uv0[1]) - (uv2[0]-uv0[0])*(uv1[1]-uv0[1]) ) / 2
    scale_factor = np.sqrt(uv_area / (obj_area + 1e-8))
    
    # 缩放雅可比矩阵
    J_scaled = J * scale_factor
    return J_scaled

def project_3dgs_with_covariance(
    smpl_mesh_path: str, 
    gs_positions: np.ndarray, 
    gs_rotations: np.ndarray,  # 四元数 [K, 4]
    gs_scales: np.ndarray,     # [K, 3]
    gs_opacity: np.ndarray,    # [K]
    gs_colors: np.ndarray = None,     # [K, 3]
    gs_sh_coeffs: np.ndarray = None,  # [K, 3, num_coeffs]
    uv_size=(1024, 1024)
) -> np.ndarray:
    # 加载SMPL Mesh
    smpl_mesh = trimesh.load(smpl_mesh_path, process=False)
    assert smpl_mesh.visual.uv is not None, "Mesh必须包含UV坐标"
    
    
    
    # 初始化UV图和权重缓存
    uv_image = np.zeros((uv_size[1], uv_size[0], 3), dtype=np.float32)

    weight_buffer = np.zeros((uv_size[1], uv_size[0]), dtype=np.float32)
    
    # 添加tqdm进度条
    from tqdm import tqdm
    
    # 对每个高斯点进行投影
    for i in tqdm(range(len(gs_positions)), desc="Projecting 3D Gaussians"):
        point = gs_positions[i]
        # 1. 找到最近的表面点和面片
        closest_point, face_idx = find_closest_surface_point(point, smpl_mesh)
        
        # 2. 计算TBN矩阵
        T, B, N = compute_tbn(smpl_mesh, face_idx)
        J = np.vstack([T, B])  # 雅可比矩阵 [2, 3]
        
        # 3. 构建3D协方差矩阵
        from scipy.spatial.transform import Rotation
        R = Rotation.from_quat(gs_rotations[i]).as_matrix()  # [3, 3]
        S = np.diag(gs_scales[i])
        Sigma_3D = R @ S @ S.T @ R.T  # [3, 3]
        
        # 4. 投影到2D切平面
        Sigma_2D = J @ Sigma_3D @ J.T  # [2, 2]
        Sigma_2D += np.eye(2) * 1e-6   # 防止奇异矩阵
        
        # 5. 计算UV坐标
        uv_center = get_uv(smpl_mesh, closest_point, face_idx)
        
        # 6. 在UV图上泼溅椭圆（考虑协方差）
        
        if gs_colors is not None:
            
            splat_elliptical_gaussian(
                uv_center= uv_center, 
                Sigma_2D=Sigma_2D, 
                uv_image=uv_image, 
                weight_buffer=weight_buffer, 
                uv_size=uv_size,
                color=gs_colors[i],
                opacity = gs_opacity[i],
                normal=-N,
            )
        if gs_sh_coeffs is not None:
            splat_elliptical_gaussian(
                uv_center= uv_center, 
                Sigma_2D=Sigma_2D, 
                uv_image=uv_image, 
                weight_buffer=weight_buffer, 
                uv_size=uv_size,
                color=None,
                opacity=gs_opacity[i],
                normal=-N,
                sh_coeffs=gs_sh_coeffs[i]
            )
    
    # 归一化颜色
    # 归一化前处理可能的除零错误
    valid_mask = weight_buffer > 1e-8
    uv_image[valid_mask] = np.divide(
        uv_image[valid_mask], 
        weight_buffer[valid_mask][:, np.newaxis]
    )
    # 未覆盖区域设为0
    # uv_image[~valid_mask] = 0.0
    # u未覆盖区域设为黄种人肤色 （0.90, 0.75, 0.65）
    uv_image[~valid_mask, 0] = 0.90
    uv_image[~valid_mask, 1] = 0.75
    uv_image[~valid_mask, 2] = 0.65
    return (uv_image * 255).astype(np.uint8)



# add a adpated version of project_3dgs_with_covariance
def project_3dgs_with_covariance_torch(
    smpl_mesh_path: str,
    gs_positions: torch.Tensor,
    gs_rotations: torch.Tensor,
    gs_scales: torch.Tensor,
    gs_opacity: torch.Tensor,
    gs_colors: torch.Tensor = None,
    gs_sh_coeffs: torch.Tensor = None,
    uv_size=(1024, 1024),
    device='cuda'
) -> np.ndarray:
    
    # convert torch tensors to numpy arrays
    gs_positions = gs_positions.detach().cpu().numpy()
    gs_rotations = gs_rotations.detach().cpu().numpy()
    gs_scales = gs_scales.detach().cpu().numpy()
    if gs_colors is not None:
        gs_colors = gs_colors.detach().cpu().numpy()
    if gs_sh_coeffs is not None:
        gs_sh_coeffs = gs_sh_coeffs.detach().cpu().numpy()
        # if gs_sh_coeffs shape is [K, 16, 3], then convert it to [K, 3, 16]
        if gs_sh_coeffs.shape[1] == 16:
            gs_sh_coeffs = np.transpose(gs_sh_coeffs, (0, 2, 1))
        
    gs_opacity = gs_opacity.detach().cpu().numpy()
    
    # change uv_size to numpy array,and type to int
    uv_size = np.array(uv_size, dtype=np.int64)
    
    return project_3dgs_with_covariance(
        smpl_mesh_path=smpl_mesh_path,
        gs_positions=gs_positions,
        gs_rotations=gs_rotations,
        gs_scales=gs_scales,
        gs_colors=gs_colors,
        gs_sh_coeffs=gs_sh_coeffs,
        gs_opacity=gs_opacity,
        uv_size=uv_size
    )
    
class GSLoader(torch.nn.Module):
    """ 版本适配加载器 """
    def __init__(self):
        super().__init__()
        self.version_mapping = {
            '1.0': self._load_v1,
            '2.0': self._load_v2
        }

    def load(self, path):
        raw = torch.load(path)
        version = raw.get('metadata', {}).get('version', '1.0')
        return self.version_mapping[version](raw)

    def _load_v1(self, data):
        # 旧版数据处理逻辑
        return data

    def _load_v2(self, data):
        # 新版数据增强逻辑
        data['sh_coeffs'] = data['sh_coeffs'].unsqueeze(-1)
        return data


def test_gs_projection_with_smpl_torch():
    # 0. 定义输入路径
    smpl_mesh_path = './assets/template_mesh_smpl_uv.obj'  # 你的SMPL路径
    
    
    
    # 1. 加载你的SMPL模型（确保文件存在）
    smpl_mesh = trimesh.load(smpl_mesh_path)
    assert smpl_mesh.visual.uv is not None, "SMPL模型必须包含UV坐标！"
    
    # 2. 生成所有顶点的高斯点（示例：前100个顶点加速测试）
    # ------------------------------------------
    # 选择顶点范围（避免全量测试耗时）
    # test_vertex_indices = range(1000)  # 测试前100个顶点
    test_vertex_indices = range(len(smpl_mesh.vertices))  # 全量顶点（慎用！）
    
    # 高斯参数设置
    gs_positions = torch.tensor(smpl_mesh.vertices[test_vertex_indices], device='cuda',dtype=torch.float32)  # [K,3]
    gs_rotations = torch.tensor([[1.0, 0.0, 0.0, 0.0]] * len(test_vertex_indices), device='cuda')  # 无旋转
    gs_scales = torch.tensor([[0.02, 0.02, 0.02]] * len(test_vertex_indices), device='cuda')       # 小尺寸
    gs_colors = torch.tensor([[1.0, 1.0, 1.0]] * len(test_vertex_indices), device='cuda')        # 白色
    # 生成SH系数（三阶，每通道16个系数）和法线
    num_sh_bands = 3  # 三阶SH
    num_sh_coeffs = (num_sh_bands + 1)**2  # 16个系数
    gs_sh_coeffs = np.random.uniform(-0.5, 0.5, (len(test_vertex_indices), 3, num_sh_coeffs))  # [K,3,16]
    gs_sh_coeffs = torch.tensor(gs_sh_coeffs, device='cuda', dtype=torch.float32)
    
    # 使用适配器加载
    loader = GSLoader()
    compatible_data = loader.load('./results/gs_data/gs_000500.pth')

    gs_positions = compatible_data['gs_positions']
    gs_rotations = compatible_data['gs_rotations']
    gs_scales = compatible_data['gs_scales']
    gs_scales = torch.abs(gs_scales)  # 修正尺度
    
    gs_sh_coeffs = compatible_data['gs_sh_coeffs']
    gs_opacity = compatible_data['gs_opacity']
    
    # 3. 运行投影函数（小尺寸UV图加速测试）
    uv_image = project_3dgs_with_covariance_torch(
        smpl_mesh_path=smpl_mesh_path,
        gs_positions=gs_positions,
        gs_rotations=gs_rotations,
        gs_scales=gs_scales,
        # gs_colors=gs_colors,
        gs_sh_coeffs=gs_sh_coeffs,
        gs_opacity=gs_opacity,
        uv_size=(128, 128)  # 小尺寸加速测试
    )
    
    # 4. 验证UV图是否包含非空像素
    non_zero_pixels = np.any(uv_image > 0, axis=-1).sum()
    print(f"非空像素数量: {non_zero_pixels}")
    # assert non_zero_pixels >= len(test_vertex_indices), \
    #     f"至少应有{len(test_vertex_indices)}个高斯点的颜色被投影到UV图"
    
    # 5. 可选：保存UV图供肉眼检查
    import cv2
    # cv2.imwrite(SAVE_PATH+"test_uv_output.png", cv2.cvtColor(uv_image, cv2.COLOR_RGB2BGR))
    cv2.imwrite(SAVE_PATH + "test_uv_output_torch.png", cv2.cvtColor(uv_image, cv2.COLOR_RGB2BGR))
    print("测试通过！生成的UV图已保存为 test_uv_output_torch.png")
    
    

def test_gs_projection_with_smpl():
    # 0. 定义输入路径
    smpl_mesh_path = './assets/template_mesh_smpl_uv.obj'  # 你的SMPL路径
    
    # 1. 加载你的SMPL模型（确保文件存在）
    smpl_mesh = trimesh.load(smpl_mesh_path)
    assert smpl_mesh.visual.uv is not None, "SMPL模型必须包含UV坐标！"
    
    
    # 2. 生成所有顶点的高斯点（示例：前100个顶点加速测试）
    # ------------------------------------------
    # 选择顶点范围（避免全量测试耗时）
    # test_vertex_indices = range(1000)  # 测试前100个顶点
    test_vertex_indices = range(len(smpl_mesh.vertices))  # 全量顶点（慎用！）
    
    # 高斯参数设置
    gs_positions = smpl_mesh.vertices[test_vertex_indices]  # [K,3]
    gs_rotations = np.tile([1.0, 0.0, 0.0, 0.0], (len(test_vertex_indices), 1))  # 无旋转
    gs_scales = np.array([[0.02, 0.02, 0.02]] * len(test_vertex_indices))       # 小尺寸
    #make gs_scales random
    gs_scales = np.random.uniform(0.01, 0.5, (len(test_vertex_indices), 3))
    gs_colors = np.tile([1.0, 1.0, 1.0], (len(test_vertex_indices), 1))        # 白色
    # 红色
    gs_colors = np.tile([1.0, 0, 0], (len(test_vertex_indices), 1)) 
    
    # 生成SH系数（三阶，每通道16个系数）和法线
    num_sh_bands = 3  # 三阶SH
    num_sh_coeffs = (num_sh_bands + 1)**2  # 16个系数
    gs_sh_coeffs = np.random.uniform(-0.5, 0.5, (len(test_vertex_indices), 3, num_sh_coeffs))  # [K,3,16]

    
    # 3. 运行投影函数（小尺寸UV图加速测试）
    uv_image = project_3dgs_with_covariance(
        smpl_mesh_path=smpl_mesh_path,
        gs_positions=gs_positions,
        gs_rotations=gs_rotations,
        gs_scales=gs_scales,
        gs_colors=gs_colors,
        # gs_sh_coeffs=gs_sh_coeffs,
        uv_size=(256, 256)  # 小尺寸加速测试
    )
    
    # 4. 验证UV图是否包含非空像素
    non_zero_pixels = np.any(uv_image > 0, axis=-1).sum()
    print(f"非空像素数量: {non_zero_pixels}")
    # assert non_zero_pixels >= len(test_vertex_indices), \
    #     f"至少应有{len(test_vertex_indices)}个高斯点的颜色被投影到UV图"
    
    # 5. 可选：保存UV图供肉眼检查
    import cv2
    # cv2.imwrite(SAVE_PATH+"test_uv_output.png", cv2.cvtColor(uv_image, cv2.COLOR_RGB2BGR))
    cv2.imwrite(SAVE_PATH + "test_uv_output.png", cv2.cvtColor(uv_image, cv2.COLOR_RGB2BGR))
    print("测试通过！生成的UV图已保存为 test_uv_output.png")
            
if __name__ == "__main__":
    # 构造一个平面面片（顶点和UV均为直角三角形）
    # vertices = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]])
    # faces = np.array([[0, 1, 2]])
    # uv = np.array([[0, 0], [1, 0], [0, 1]])
    # mesh = trimesh.Trimesh(vertices=vertices, faces=faces, visual=trimesh.visual.TextureVisuals(uv=uv))
    # T, B, N = compute_tbn(mesh, 0)
    # assert np.allclose(np.dot(T, B), 0), "T和B必须正交"
    # assert np.allclose(np.dot(T, N), 0), "T和N必须正交"
    # assert np.allclose(np.dot(B, N), 0), "B和N必须正交"
    
    # # 在面片中心点应插值为(0.5, 0.5)
    # point = np.array([0.5, 0.5, 0])
    # closest_point, face_idx = find_closest_surface_point(point, mesh)
    # uv = get_uv(mesh, closest_point, face_idx)
    # assert np.allclose(uv, [0.5, 0.5]), "中心点UV应为(0.5, 0.5)"
    
    #test_gs_projection_with_smpl()
    
    test_gs_projection_with_smpl_torch()