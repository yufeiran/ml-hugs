import torch
import torch.nn as nn
import math

class MultiResolutionHashGrid(nn.Module):
    def __init__(
        self,
        n_levels=4,
        n_features_per_level=2,
        log2_hashmap_size=8, # TO BE tested !
        base_resolution=16,
        max_resolution=512,
        input_range=(-1, 1),  # 新增参数，指定输入范围
    ):
        super().__init__()
        self.input_min, self.input_max = input_range  # 输入坐标范围

        # 分辨率层级计算（原代码不变）
        self.n_levels = n_levels
        self.base_resolution = base_resolution
        self.max_resolution = max_resolution
        b = math.exp((math.log(max_resolution) - math.log(base_resolution)) / (n_levels-1))
        self.resolutions = [int(base_resolution * (b**l)) for l in range(n_levels)]
        
        self.log2_hashmap_size = log2_hashmap_size  # 哈希表大小（2的幂次方）

        # 哈希表初始化（原代码不变）
        self.hash_tables = nn.ParameterList([
            nn.Parameter(1e-4 * torch.randn(2**log2_hashmap_size, n_features_per_level))
            for _ in range(n_levels)
        ])


    def _hash_function(self, grid_indices, level):
        # 将3D坐标映射到哈希索引（参考Instant NGP的哈希函数）
        primes = [1, 2654435761, 805459861, 3674653429]
        h = torch.bitwise_xor(grid_indices[..., 0], grid_indices[..., 1] * primes[1])
        h = torch.bitwise_xor(h, grid_indices[..., 2] * primes[2])
        return h % (2**self.log2_hashmap_size)

    def forward(self, x):
        # 将输入从 [-1,1] 映射到 [0,1]
        x_normalized = (x - self.input_min) / (self.input_max - self.input_min)  # 归一化到[0,1]

        features = []
        for level in range(self.n_levels):
            resolution = self.resolutions[level]
            
            # 计算网格索引（关键调整！）
            scaled_coords = x_normalized * (resolution - 1)
            grid_indices = torch.floor(scaled_coords).long()
            
            # 后续哈希计算与原代码一致
            hash_indices = self._hash_function(grid_indices, level)
            feature = self.hash_tables[level][hash_indices]
            features.append(feature)

        return torch.cat(features, dim=-1)

# 测试代码
if __name__ == "__main__":
    # 参数设置
    grid_encoder = MultiResolutionHashGrid(
        n_levels=4,
        n_features_per_level=2,
        log2_hashmap_size=18,  # 哈希表大小=2^18=262144
        base_resolution=16,
        max_resolution=512,
        input_range=(-1,1),
    )

    # 输入坐标（假设范围[0,1]）
    x = torch.tensor([[0.1, 0.2, 0.3], [0.8, 0.7, 0.5]])  # [2,3]
    features = grid_encoder(x)
    print(f"Output features shape: {features.shape}")  # 预期 [2, 4 * 2=8]
    print(f"Output features: {features}")