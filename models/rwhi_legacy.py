# ==============================================================================
# RWHI v2.1 模块 (Robust Hybrid Anchor Initialization - Budget Inflation)
# ==============================================================================
# 版本: 2.1
# 策略: 预算膨胀 + 叠加策略 (P_total = P_base + P_radar)
# 
# 版本演进:
# - v1.0: 分割策略 (近场=安全流, 远场=雷达) -> 远场召回率崩溃
# - v2.0: 叠加策略 (num_base=500, num_radar=400) -> 基础密度下降44%
# - v2.1: 预算膨胀 (num_base=900, num_radar=300) -> 匹配基线+纯增益
#
# v2.1 关键改进:
# - num_base = 900: 完全匹配原始RaCFormer的空间密度
# - num_radar = 300: 额外的雷达增益，作为纯收益
# - num_query = 1200: 膨胀的总预算 (900 + 300)
#
# 数学模型:
# 1. P_base(r) = max(α·r^{-1}, ε_floor)  -- 逆深度分布 + 最小密度保证
# 2. P_radar(x,y) = Σ w_i · N(x_i, y_i)  -- 高RCS区域增益
#
# 预期效果: mAP > 基线 (因为基础覆盖相同，雷达增益是纯收益)
#
# 设计原则: 全向量化，无Python for循环，TensorRT兼容
# ==============================================================================

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from mmcv.runner import BaseModule
from .bbox.utils import compute_map_size_and_radius


class RWHIModule(BaseModule):
    """
    RWHI v2.1: Robust Hybrid Anchor Initialization Module (Budget Inflation)
    
    30m 分界双流策略:
    - Safety Stream (0-30m): 固定锚点池，逆深度采样，近密远疏
    - Saliency Stream (>30m): 雷达驱动，向量化加权 + scatter_add + TopK
    
    输入:
        - radar_points: 雷达点云 [B, M, C] (x, y, z, rcs, v_r, ...)
        
    输出:
        - query_positions: 混合锚点位置 [B, N_total, 10]
        - anchor_mask: 锚点类型掩码 [B, N_total] (False=Safety, True=Saliency)
    """
    
    def __init__(self,
                 num_query=900,
                 num_safety=300,
                 num_saliency=600,
                 embed_dims=256,
                 pc_range=(-51.2, -51.2, -5.0, 51.2, 51.2, 3.0),
                 bev_grid_size=100,      # 用于雷达增益的BEV网格大小
                 max_range=55.0,
                 min_range=2.0,
                 safety_max_range=30.0,
                 rcs_threshold=0.0,
                 rcs_noise=0.0,
                 rcs_clip=10.0,
                 dist_ref=30.0,
                 dist_gamma=1.0,
                 vel_beta=0.0,
                 vel_threshold=0.0,
                 vel_scale=1.0,
                 radar_channel_map=None,
                 height_hypotheses=(0.0, 1.0),  # 高度假设 (米)
                 diffusion_kernel_size=3,
                 noise_eps_train=1e-6,
                 noise_eps_test=0.0,
                 enabled=True,
                 map_size=None,
                 polar_radius=None,
                 # 兼容v1.0的参数 (忽略但不报错)
                 num_base=None,
                 num_radar=None,
                 safety_ratio=None,
                 init_cfg=None):
        """
        初始化 RWHI 30m 双流模块
        
        Args:
            num_query: 总Query数量
            num_safety: Safety锚点数量 (0-30m 固定)
            num_saliency: Saliency锚点数量 (>30m 雷达驱动)
            embed_dims: 嵌入维度
            pc_range: 点云范围 [x_min, y_min, z_min, x_max, y_max, z_max]
            bev_grid_size: BEV网格大小
            max_range: 最大感知范围 (米)
            min_range: 最小感知范围 (米)
            safety_max_range: Safety范围上限 (米)
            rcs_threshold: RCS过滤阈值
            rcs_noise: RCS噪声阈值
            rcs_clip: RCS对数压缩上限
            dist_ref: 距离补偿参考值
            dist_gamma: 距离补偿幂次
            vel_beta: 速度权重系数
            vel_threshold: 速度阈值
            vel_scale: 速度尺度
            height_hypotheses: 高度假设 (米)
            diffusion_kernel_size: 不确定性扩散核大小
            noise_eps_train: 训练时TopK噪声
            noise_eps_test: 推理时TopK噪声
            enabled: 是否启用RWHI
        """
        super(RWHIModule, self).__init__(init_cfg=init_cfg)
        
        # 兼容旧参数
        if num_base is not None:
            num_safety = num_base
        if num_radar is not None:
            num_saliency = num_radar

        self.num_query = num_query
        self.num_safety = num_safety
        self.num_saliency = num_saliency
        self.embed_dims = embed_dims
        self.pc_range = pc_range
        self.bev_grid_size = bev_grid_size
        self.max_range = max_range
        self.min_range = min_range
        self.safety_max_range = safety_max_range
        self.rcs_threshold = rcs_threshold
        self.rcs_noise = rcs_noise
        self.rcs_clip = rcs_clip
        self.dist_ref = dist_ref
        self.dist_gamma = dist_gamma
        self.vel_beta = vel_beta
        self.vel_threshold = vel_threshold
        self.vel_scale = vel_scale
        self.radar_channel_map = radar_channel_map or {
            'x': 0, 'y': 1, 'z': 2, 'rcs': 3, 'v_r': 4
        }
        self.height_hypotheses = height_hypotheses
        self.diffusion_kernel_size = diffusion_kernel_size
        self.noise_eps_train = noise_eps_train
        self.noise_eps_test = noise_eps_test
        self.enabled = enabled
        self.map_size, self.polar_radius = compute_map_size_and_radius(
            self.pc_range, map_size=map_size, polar_radius=polar_radius
        )
        
        # 验证锚点数量配置
        if self.num_safety + self.num_saliency != self.num_query:
            print(f"[RWHI] Warning: num_safety({self.num_safety}) + num_saliency({self.num_saliency}) "
                  f"!= num_query({self.num_query}). Adjusting num_saliency.")
            self.num_saliency = self.num_query - self.num_safety
        
        # ============================================================
        # 计算BEV网格参数
        # ============================================================
        self.bev_x_range = pc_range[3] - pc_range[0]  # 102.4
        self.bev_y_range = pc_range[4] - pc_range[1]  # 102.4
        self.cell_size_x = self.bev_x_range / bev_grid_size
        self.cell_size_y = self.bev_y_range / bev_grid_size
        
        # ============================================================
        # 预计算安全锚点 (0-30m 逆深度分布)
        # ============================================================
        safety_anchors = self._precompute_base_anchors()
        self.register_buffer('base_anchors', safety_anchors)
        
        # 不确定性扩散层 (MaxPool2d)
        self.diffusion = nn.MaxPool2d(
            kernel_size=diffusion_kernel_size,
            stride=1,
            padding=diffusion_kernel_size // 2
        )
        
        # 预计算BEV近场mask (用于Saliency屏蔽 0-30m)
        self.register_buffer('bev_dist_grid', self._precompute_bev_distance_grid())

        print(f"[RWHI] Initialized: num_safety={self.num_safety}, num_saliency={self.num_saliency}, "
              f"safety_range=[{self.min_range}, {self.safety_max_range}]m, "
              f"polar_radius={self.polar_radius:.2f}")
    
    def _precompute_base_anchors(self):
        """
        预计算安全锚点 - 逆深度采样 (0-30m)
        
        使用 1/r 空间均匀采样，保证近场更密、远场更疏。
        """
        aspect_ratio = 1.2
        num_rings = int(math.sqrt(self.num_safety / aspect_ratio))
        num_angles = max(self.num_safety // max(num_rings, 1), 1)

        while num_rings * num_angles < self.num_safety:
            num_angles += 1

        r_min = max(self.min_range, 1e-3)
        r_max = max(self.safety_max_range, r_min + 1e-3)
        rho_min = 1.0 / r_min
        rho_max = 1.0 / r_max

        rho_vals = torch.linspace(rho_min, rho_max, num_rings)
        r_vals = 1.0 / rho_vals
        d_vals = r_vals / self.polar_radius
        d_vals = torch.clamp(d_vals, 0.0, 1.0)

        angles = torch.linspace(0, 1, num_angles + 1)[:-1]
        theta_grid = angles.view(1, num_angles).expand(num_rings, num_angles)
        d_grid = d_vals.view(num_rings, 1).expand(num_rings, num_angles)

        theta = theta_grid.reshape(-1)
        d = d_grid.reshape(-1)

        current_num = theta.shape[0]
        if current_num >= self.num_safety:
            theta = theta[:self.num_safety]
            d = d[:self.num_safety]
        else:
            extra_needed = self.num_safety - current_num
            extra_angles = torch.linspace(0.5 / num_angles, 1 - 0.5 / num_angles, extra_needed)
            extra_d = torch.full((extra_needed,), d_vals[-1])
            theta = torch.cat([theta, extra_angles])
            d = torch.cat([d, extra_d])
        
        # ============================================================
        # Step 6: 组合锚点属性
        # ============================================================
        theta_d = torch.stack([theta, d], dim=-1)  # [num_safety, 2]
        
        num_anchors = theta_d.shape[0]
        z = torch.full((num_anchors, 1), 0.5)  # 归一化z坐标 (中间高度)
        w = torch.full((num_anchors, 1), 0.0)  # log(w)
        l = torch.full((num_anchors, 1), 0.0)  # log(l)  
        h = torch.full((num_anchors, 1), 0.2)  # log(h)
        sin_rot = torch.zeros((num_anchors, 1))
        cos_rot = torch.ones((num_anchors, 1))
        vx = torch.zeros((num_anchors, 1))
        vy = torch.zeros((num_anchors, 1))
        
        base_anchors = torch.cat([theta_d, z, w, l, h, sin_rot, cos_rot, vx, vy], dim=-1)
        
        # 打印分布统计
        print(f"[RWHI] Safety anchors: {num_anchors} points, "
              f"rings={num_rings}, angles={num_angles}")
        print(f"[RWHI] Safety distance range: [{d.min().item()*self.polar_radius:.1f}m, "
              f"{d.max().item()*self.polar_radius:.1f}m]")
        
        return base_anchors  # [num_safety, 10]

    def _precompute_bev_distance_grid(self):
        """
        预计算BEV网格中心到ego的距离 (米)
        """
        device = self.base_anchors.device if hasattr(self, 'base_anchors') else torch.device('cpu')
        x_coords = torch.linspace(
            self.pc_range[0] + 0.5 * self.cell_size_x,
            self.pc_range[3] - 0.5 * self.cell_size_x,
            self.bev_grid_size,
            device=device,
        )
        y_coords = torch.linspace(
            self.pc_range[1] + 0.5 * self.cell_size_y,
            self.pc_range[4] - 0.5 * self.cell_size_y,
            self.bev_grid_size,
            device=device,
        )
        yy, xx = torch.meshgrid(y_coords, x_coords, indexing='ij')
        dist = torch.sqrt(xx ** 2 + yy ** 2)
        return dist.unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]
    
    # ============================================================
    # 雷达权重计算
    # ============================================================
    def _validate_channel_map(self, radar_points):
        max_idx = max(self.radar_channel_map.values())
        if radar_points.shape[-1] <= max_idx:
            raise ValueError(
                f"radar_points last dim={radar_points.shape[-1]} < required idx {max_idx}."
            )

    def compute_radar_weights(self, radar_points):
        """
        计算雷达点权重 (向量化)
        
        W = W_rcs * W_dist * W_vel
        
        Args:
            radar_points: [B, M, C] (x, y, z, rcs, v_r, ...)
            
        Returns:
            weights: [B, M]
        """
        self._validate_channel_map(radar_points)
        x = radar_points[..., self.radar_channel_map['x']]
        y = radar_points[..., self.radar_channel_map['y']]
        rcs = radar_points[..., self.radar_channel_map['rcs']]
        v_r = radar_points[..., self.radar_channel_map['v_r']]

        # RCS权重: log(1 + ReLU(RCS - noise)) + clip/归一化
        rcs_weight = torch.log1p(F.relu(rcs - self.rcs_noise))
        if self.rcs_clip is not None and self.rcs_clip > 0:
            rcs_weight = torch.clamp(rcs_weight, max=self.rcs_clip) / self.rcs_clip

        # 距离补偿: 鼓励远场
        dist = torch.sqrt(x ** 2 + y ** 2).clamp(min=1e-3)
        dist_weight = torch.pow(dist / max(self.dist_ref, 1e-3), self.dist_gamma)
        dist_weight = torch.clamp(dist_weight, min=1.0)

        # 速度权重: 1 + beta * sigmoid((|v|-v_th)/v_scale)
        if self.vel_beta > 0:
            vel_weight = 1.0 + self.vel_beta * torch.sigmoid(
                (torch.abs(v_r) - self.vel_threshold) / max(self.vel_scale, 1e-3)
            )
        else:
            vel_weight = torch.ones_like(dist_weight)

        weights = rcs_weight * dist_weight * vel_weight
        
        return weights
    
    def filter_by_rcs(self, radar_points, weights):
        """
        根据RCS阈值过滤雷达点 (向量化)
        
        仅保留高置信度点 (RCS > threshold) 的权重
        
        Args:
            radar_points: [B, M, C]
            weights: [B, M]
            
        Returns:
            filtered_weights: [B, M] (低RCS点的权重置为0)
        """
        rcs = radar_points[..., self.radar_channel_map['rcs']]  # [B, M]
        
        # 创建掩码: RCS > threshold
        valid_mask = (rcs > self.rcs_threshold).float()  # [B, M]
        
        # 应用掩码
        filtered_weights = weights * valid_mask  # [B, M]
        
        return filtered_weights
    
    # ============================================================
    # 体素化和采样
    # ============================================================
    def scatter_add_voxelize(self, radar_points, weights):
        """
        散点累加体素化 (全向量化，无循环)
        
        将雷达点权重累加到BEV网格中
        
        Args:
            radar_points: [B, M, C]
            weights: [B, M]
            
        Returns:
            bev_grid: [B, 1, H, W]
        """
        B, M, _ = radar_points.shape
        device = radar_points.device
        
        # 提取xy坐标
        x = radar_points[..., self.radar_channel_map['x']]  # [B, M]
        y = radar_points[..., self.radar_channel_map['y']]  # [B, M]
        
        # 计算网格索引
        x_idx = ((x - self.pc_range[0]) / self.cell_size_x).long()  # [B, M]
        y_idx = ((y - self.pc_range[1]) / self.cell_size_y).long()  # [B, M]
        
        # 裁剪到有效范围
        x_idx = x_idx.clamp(0, self.bev_grid_size - 1)
        y_idx = y_idx.clamp(0, self.bev_grid_size - 1)
        
        # 计算一维索引: batch_idx * H * W + y_idx * W + x_idx
        batch_idx = torch.arange(B, device=device).view(B, 1).expand(B, M)  # [B, M]
        flat_idx = batch_idx * (self.bev_grid_size * self.bev_grid_size) + \
                   y_idx * self.bev_grid_size + x_idx  # [B, M]
        flat_idx = flat_idx.view(-1)  # [B*M]
        
        # 展平权重
        flat_weights = weights.view(-1)  # [B*M]
        
        # 创建输出网格
        bev_grid = torch.zeros(
            B * self.bev_grid_size * self.bev_grid_size,
            device=device,
            dtype=weights.dtype
        )
        
        # scatter_add_ 累加权重
        bev_grid.scatter_add_(0, flat_idx, flat_weights)
        
        # 重塑为 [B, 1, H, W]
        bev_grid = bev_grid.view(B, self.bev_grid_size, self.bev_grid_size)
        bev_grid = bev_grid.unsqueeze(1)  # [B, 1, H, W]
        
        return bev_grid
    
    def apply_diffusion(self, bev_grid):
        """
        应用不确定性扩散 (MaxPool2d)
        
        模拟雷达角分辨率的不确定性
        
        Args:
            bev_grid: [B, 1, H, W]
            
        Returns:
            diffused_grid: [B, 1, H, W]
        """
        return self.diffusion(bev_grid)
    
    def topk_sampling(self, bev_grid, num_samples):
        """
        Top-K采样 (向量化)
        
        从BEV网格中选取响应最高的位置作为雷达增益锚点
        
        Args:
            bev_grid: [B, 1, H, W]
            num_samples: 采样数量
            
        Returns:
            selected_xy: [B, num_samples, 2] 归一化xy坐标 [0, 1]
        """
        B, _, H, W = bev_grid.shape
        device = bev_grid.device
        
        # 添加微小噪声防止TopK重复 (训练开启，推理关闭)
        noise_eps = self.noise_eps_train if self.training else self.noise_eps_test
        if noise_eps > 0:
            noise = torch.rand_like(bev_grid) * noise_eps
            noisy_grid = bev_grid + noise
        else:
            noisy_grid = bev_grid
        
        # 展平并TopK
        flat_grid = noisy_grid.view(B, -1)  # [B, H*W]
        
        # 确保num_samples不超过有效网格数
        num_samples = min(num_samples, H * W)
        
        _, top_indices = torch.topk(flat_grid, num_samples, dim=1)  # [B, num_samples]
        
        # 将一维索引转换回二维坐标
        y_idx = top_indices // W  # [B, num_samples]
        x_idx = top_indices % W   # [B, num_samples]
        
        # 转换为归一化坐标 [0, 1]
        x_norm = (x_idx.float() + 0.5) / W  # [B, num_samples]
        y_norm = (y_idx.float() + 0.5) / H  # [B, num_samples]
        
        selected_xy = torch.stack([x_norm, y_norm], dim=-1)  # [B, num_samples, 2]
        
        return selected_xy
    
    # ============================================================
    # 坐标转换
    # ============================================================
    def _xy_to_theta_d(self, xy_norm):
        """
        将归一化xy坐标转换为theta-d极坐标
        
        坐标系约定:
        - xy_norm ∈ [0, 1]，BEV网格归一化位置
        - (0.5, 0.5) = ego车辆位置 (BEV中心)
        - theta ∈ [0, 1]: 归一化角度 [0, 2π)
        - d ∈ [0, 1]: 归一化距离
        
        Args:
            xy_norm: [B, N, 2] 归一化坐标 [0, 1]
            
        Returns:
            theta_d: [B, N, 2] 极坐标
        """
        map_size = self.map_size
        r = self.polar_radius
        center = map_size / 2
        
        # 反归一化到实际坐标 (米)
        x = xy_norm[..., 0:1] * map_size  # [B, N, 1]
        y = xy_norm[..., 1:2] * map_size  # [B, N, 1]
        
        # 计算相对于中心的偏移
        dx = x - center
        dy = y - center
        
        # 计算极坐标
        distance = torch.sqrt(dx ** 2 + dy ** 2) / r  # 归一化距离
        theta = torch.atan2(dy, dx)  # [-π, π]
        theta = ((theta + 2 * math.pi) % (2 * math.pi)) / (2 * math.pi)  # 归一化到[0, 1]
        
        # Clamp 到 [0, 1] 范围
        theta = torch.clamp(theta, 0.0, 1.0)
        distance = torch.clamp(distance, 0.0, 1.0)
        
        theta_d = torch.cat([theta, distance], dim=-1)  # [B, N, 2]
        
        return theta_d
    
    # ============================================================
    # 锚点获取方法
    # ============================================================
    def get_base_anchors(self, batch_size, device):
        """
        获取安全锚点 (静态，覆盖0-30m)
        
        Args:
            batch_size: batch大小
            device: 设备
            
        Returns:
            base_anchors: [B, num_safety, 10]
        """
        base_anchors = self.base_anchors.to(device)
        base_anchors = base_anchors.unsqueeze(0).expand(batch_size, -1, -1).clone()
        return base_anchors

    def _empty_saliency_anchors(self, batch_size, device):
        anchors = torch.zeros(batch_size, self.num_saliency, 10, device=device)
        anchors[..., 2] = 0.5  # z
        anchors[..., 7] = 1.0  # cos
        return anchors
    
    def get_radar_anchors(self, radar_points):
        """
        获取雷达增益锚点 (动态，基于高RCS区域)
        
        流程:
        1. 计算雷达点权重 W = log(1+RCS) × (1 + α×sigmoid(|v_r|))
        2. 按RCS阈值过滤
        3. scatter_add 体素化到BEV网格
        4. MaxPool 不确定性扩散
        5. TopK 采样高价值区域
        6. 多高度假设生成
        
        Args:
            radar_points: [B, M, C] 雷达点云
            
        Returns:
            radar_anchors: [B, num_saliency, 10]
        """
        B = radar_points.shape[0]
        device = radar_points.device
        if radar_points.shape[1] == 0:
            return self._empty_saliency_anchors(B, device)
        
        # Step 1: 计算雷达点权重
        weights = self.compute_radar_weights(radar_points)  # [B, M]
        
        # Step 2: 根据RCS阈值过滤
        weights = self.filter_by_rcs(radar_points, weights)  # [B, M]
        
        # Step 3: 散点累加体素化
        bev_grid = self.scatter_add_voxelize(radar_points, weights)  # [B, 1, H, W]
        
        # Step 4: 不确定性扩散
        bev_grid = self.apply_diffusion(bev_grid)  # [B, 1, H, W]

        # Step 4.5: 近场mask (<=30m) 置零，避免覆盖Safety
        bev_mask = (self.bev_dist_grid.to(device) > self.safety_max_range).type_as(bev_grid)
        bev_grid = bev_grid * bev_mask

        grid_sum = bev_grid.view(B, -1).sum(dim=1)  # [B]
        invalid_batch = grid_sum <= 0

        # Step 5: TopK采样
        # 考虑高度假设，每个假设采样 num_saliency / num_heights 个点
        num_heights = len(self.height_hypotheses)
        num_samples_per_height = self.num_saliency // num_heights
        
        selected_xy = self.topk_sampling(bev_grid, num_samples_per_height)  # [B, N, 2]
        
        # Step 6: 多假设高度生成
        N = selected_xy.shape[1]
        
        # 复制xy坐标用于不同高度假设 (向量化)
        # [B, N, 2] -> [B, N, num_heights, 2] -> [B, N*num_heights, 2]
        selected_xy_expanded = selected_xy.unsqueeze(2).expand(B, N, num_heights, 2)
        selected_xy_expanded = selected_xy_expanded.reshape(B, N * num_heights, 2)
        
        # 生成高度 (归一化到[0, 1])
        z_min, z_max = self.pc_range[2], self.pc_range[5]
        heights_norm = torch.tensor(
            [(h - z_min) / (z_max - z_min) for h in self.height_hypotheses],
            device=device, dtype=selected_xy.dtype
        )  # [num_heights]
        
        # 扩展高度
        # [num_heights] -> [1, 1, num_heights] -> [B, N, num_heights] -> [B, N*num_heights, 1]
        heights = heights_norm.view(1, 1, num_heights).expand(B, N, num_heights)
        heights = heights.reshape(B, N * num_heights, 1)
        
        # 转换xy坐标为theta-d极坐标
        theta_d = self._xy_to_theta_d(selected_xy_expanded)  # [B, N*num_heights, 2]
        
        # 组合成完整的锚点表示
        num_anchors = N * num_heights
        w = torch.zeros((B, num_anchors, 1), device=device)
        l = torch.zeros((B, num_anchors, 1), device=device)
        h = torch.full((B, num_anchors, 1), 0.2, device=device)
        sin_rot = torch.zeros((B, num_anchors, 1), device=device)
        cos_rot = torch.ones((B, num_anchors, 1), device=device)
        vx = torch.zeros((B, num_anchors, 1), device=device)
        vy = torch.zeros((B, num_anchors, 1), device=device)
        
        radar_anchors = torch.cat([
            theta_d, heights, w, l, h, sin_rot, cos_rot, vx, vy
        ], dim=-1)  # [B, num_anchors, 10]
        
        # 确保锚点数量正确 (截断或填充)
        if radar_anchors.shape[1] > self.num_saliency:
            radar_anchors = radar_anchors[:, :self.num_saliency, :]
        elif radar_anchors.shape[1] < self.num_saliency:
            # 用默认值填充
            padding_size = self.num_saliency - radar_anchors.shape[1]
            padding = torch.zeros(B, padding_size, 10, device=device, dtype=radar_anchors.dtype)
            padding[..., 2] = 0.5  # z
            padding[..., 7] = 1.0  # cos
            radar_anchors = torch.cat([radar_anchors, padding], dim=1)

        if invalid_batch.any():
            empty_anchors = self._empty_saliency_anchors(B, device)
            radar_anchors[invalid_batch] = empty_anchors[invalid_batch]
        
        return radar_anchors
    
    # ============================================================
    # 前向传播
    # ============================================================
    def forward(self, radar_points=None):
        """
        RWHI v2.0 前向传播
        
        双流策略: query_bbox = concat(safety_anchors, saliency_anchors)
        - safety_anchors: 固定安全网 (0-30m)
        - saliency_anchors: 雷达驱动 (>30m)
        
        Args:
            radar_points: [B, M, C] 雷达点云 (x, y, z, rcs, v_r, ...)
                         如果为None，仅返回基础锚点 (复制以填满num_query)
            
        Returns:
            hybrid_anchors: [B, num_query, 10] 混合锚点
            anchor_mask: [B, num_query] bool 区分Safety(False)和Saliency(True)
        """
        # ============================================================
        # Case 1: RWHI禁用
        # ============================================================
        if not self.enabled:
            B = 1 if radar_points is None else radar_points.shape[0]
            device = self.base_anchors.device if radar_points is None else radar_points.device
            
            base_anchors = self.base_anchors.to(device)
            base_anchors = base_anchors.unsqueeze(0).expand(B, -1, -1).clone()
            
            repeat_times = (self.num_query + self.num_safety - 1) // self.num_safety
            full_anchors = base_anchors.repeat(1, repeat_times, 1)[:, :self.num_query, :]
            anchor_mask = torch.zeros(B, self.num_query, dtype=torch.bool, device=device)
            return full_anchors, anchor_mask
        
        # ============================================================
        # Case 2: 无雷达数据
        # ============================================================
        if radar_points is None:
            B = 1
            device = self.base_anchors.device
            
            base_anchors = self.base_anchors.to(device)
            base_anchors = base_anchors.unsqueeze(0).expand(B, -1, -1).clone()
            
            safety_anchors = self.get_base_anchors(B, device)
            saliency_anchors = self._empty_saliency_anchors(B, device)
            hybrid_anchors = torch.cat([safety_anchors, saliency_anchors], dim=1)
            anchor_mask = torch.zeros(B, self.num_query, dtype=torch.bool, device=device)
            anchor_mask[:, self.num_safety:] = True
            return hybrid_anchors, anchor_mask
        
        # ============================================================
        # Case 3: 正常叠加策略
        # ============================================================
        B = radar_points.shape[0]
        device = radar_points.device
        
        # 获取安全锚点 (静态，0-30m)
        base_anchors = self.get_base_anchors(B, device)  # [B, num_safety, 10]
        
        # 获取显著锚点 (动态，基于雷达)
        radar_anchors = self.get_radar_anchors(radar_points)  # [B, num_saliency, 10]
        
        # 拼接: Safety -> Saliency
        hybrid_anchors = torch.cat([base_anchors, radar_anchors], dim=1)  # [B, num_query, 10]
        
        # 创建掩码 (Safety=False, Saliency=True)
        anchor_mask = torch.zeros(B, self.num_query, dtype=torch.bool, device=device)
        anchor_mask[:, self.num_safety:] = True
        
        return hybrid_anchors, anchor_mask
    
    # ============================================================
    # 兼容性属性 (供 racformer_head.py 使用)
    # ============================================================
    @property
    def safety_anchors(self):
        """
        兼容性属性: 返回基础锚点作为"安全锚点"
        
        供 racformer_head.py 初始化 init_query_bbox 使用
        """
        return self.base_anchors
    
    @property
    def num_safety_anchors(self):
        """
        兼容性属性: 返回安全锚点数量
        
        供 racformer_head.py 使用
        """
        return self.num_safety


# ============================================================
# RWHIQueryGenerator - 已废弃
# ============================================================
class RWHIQueryGenerator(BaseModule):
    """
    [已废弃] RWHI Query生成器
    
    警告: 此类已废弃，请直接使用 RWHIModule。
    保留此类定义以保持向后兼容，但不建议使用。
    """
    
    def __init__(self, *args, **kwargs):
        raise DeprecationWarning(
            "RWHIQueryGenerator 已废弃。"
            "请直接使用 RWHIModule，并在 RaCFormer_head 中使用 pos2content MLP。"
            "参考: RaCFormer/models/racformer_head.py"
        )
