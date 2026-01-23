"""
RWHI v7 (Radar-Weighted Hybrid Initialization) 模块

根据 rwhi代码实现.md 文档实现，用于替代 RaCFormer 原有的均匀极坐标分布锚点。

核心思想：
1. 构建 BEV 打分图 S(x,y)，雷达高置信区域得分高
2. Top-K 选取得分最高的 K 个位置作为 Query 锚点
3. 为每个锚点计算置信度 α，用于特征增强
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.runner import BaseModule

# 从 bbox/utils.py 导入 R_MAX，确保全链路统一
from .bbox.utils import R_MAX

# 固定常量
# R_MAX 从 bbox/utils.py 导入，值为 65.0
EPS = 1e-5    # 数值稳定性常量


class AlphaMLP(nn.Module):
    """
    点置信度预测网络
    
    输入: f_in = [log1p(max(rcs,0)), d_norm, v_norm]  # 3维
    输出: α ∈ (0, 1)
    结构: Linear(3→32) → ReLU → Linear(32→32) → ReLU → Linear(32→1) → Sigmoid
    """
    
    def __init__(self, in_dim=3, hidden_dim=32, init_bias=1.0):
        """
        Args:
            in_dim: 输入维度 (默认 3: [log1p(rcs), d_norm, v_norm])
            hidden_dim: 隐藏层维度
            init_bias: 最后一层偏置初始化值，使初始 α ≈ 0.73，偏向信任雷达
        """
        super().__init__()
        
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)
        
        # 初始化最后一层偏置，使初始 α 偏向信任雷达
        nn.init.constant_(self.fc3.bias, init_bias)
    
    def forward(self, x):
        """
        Args:
            x: [B, M, in_dim] 输入特征
        Returns:
            alpha: [B, M, 1] 点置信度，范围 (0, 1)
        """
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        alpha = torch.sigmoid(self.fc3(x))
        return alpha


class AlphaEncoder(nn.Module):
    """
    α 特征编码器
    
    将 α 编码为 d_alpha 维 embedding，用于 Query 特征增强。
    不修改 bbox_proposal 的 10 维结构。
    
    输入: α ∈ (0, 1)
    处理: α_center = (α - 0.5) * 2  → [-1, 1]
          u = [α_center, α_center²]  → 2维
    输出: Linear(2→hidden) → ReLU → Linear(hidden→d_alpha)
    """
    
    def __init__(self, d_alpha=2, hidden_dim=8):
        """
        Args:
            d_alpha: 输出 embedding 维度
            hidden_dim: 隐藏层维度
        """
        super().__init__()
        
        self.fc1 = nn.Linear(2, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, d_alpha)
        self.d_alpha = d_alpha
    
    def forward(self, alpha):
        """
        Args:
            alpha: [B, K, 1] 置信度值，范围 (0, 1)
        Returns:
            embedding: [B, K, d_alpha] α 的特征编码
        """
        # α_center = (α - 0.5) * 2, 映射到 [-1, 1]
        alpha_center = (alpha - 0.5) * 2.0
        
        # 构建输入: [α_center, α_center²]
        u = torch.cat([alpha_center, alpha_center ** 2], dim=-1)  # [B, K, 2]
        
        # MLP 编码
        x = F.relu(self.fc1(u))
        embedding = self.fc2(x)
        
        return embedding


class RWHIModule(BaseModule):
    """
    RWHI v7 核心模块
    
    功能：
    1. 从雷达点云构建 BEV 打分图
    2. Top-K 选取锚点位置
    3. 输出 10 维归一化 bbox 和 α 置信度
    
    打分公式：
    S(x,y) = C_base + I_radar(x,y) + ε * J(x,y)
    S_final = S + γ * (MaxPool3x3(S) - S)
    
    I_radar 公式：
    I_radar(点) = α * log(1 + σ_proc * w_v * w_d)
    """
    
    def __init__(
        self,
        num_query,
        pc_range,
        bev_grid_size=100,
        # I_radar 参数
        d_ref=30.0,
        d_lambda=1.0,
        w_d_max=2.5,
        d_min=2.0,
        v_max=30.0,
        v_ref=10.0,
        beta=1.0,
        # 打分场参数
        base_bias=1.0,
        epsilon=0.01,
        diffusion_gamma=0.5,
        diffusion_s_max=5.0,
        # 默认值
        z_default=0.5,
        w_default=1.8,
        l_default=4.0,
        h_default=1.5,
        # AlphaMLP 参数
        alpha_mlp_in_dim=3,
        alpha_mlp_hidden=32,
        alpha_init_bias=1.0,
        # AlphaEncoder 参数
        d_alpha=2,
        alpha_encoder_hidden=8,
        # 其他
        num_clusters=6,
        max_points=5000,
        enabled=True,
        init_cfg=None,
        **kwargs
    ):
        """
        Args:
            num_query: Query 数量 (K)
            pc_range: 点云范围 [x_min, y_min, z_min, x_max, y_max, z_max]
            bev_grid_size: BEV 网格尺寸
            d_ref: 距离权重参考值
            d_lambda: 距离权重指数
            w_d_max: 距离权重上限
            d_min: 最小有效距离
            v_max: 最大速度
            v_ref: 速度权重参考值
            beta: 速度权重系数
            base_bias: 基础分数 C_base
            epsilon: 微扰层系数
            diffusion_gamma: 空间扩散系数
            diffusion_s_max: 得分上限
            z_default: 默认归一化高度
            w_default: 默认物体宽度 (物理值，会转为 log)
            l_default: 默认物体长度 (物理值，会转为 log)
            h_default: 默认物体高度 (物理值，会转为 log)
            alpha_mlp_in_dim: AlphaMLP 输入维度
            alpha_mlp_hidden: AlphaMLP 隐藏层维度
            alpha_init_bias: AlphaMLP 初始偏置
            d_alpha: α embedding 维度
            alpha_encoder_hidden: AlphaEncoder 隐藏层维度
            num_clusters: 距离层数量 (用于 safety_anchors 生成)
            max_points: 最大雷达点数
            enabled: 是否启用 RWHI
        """
        super().__init__(init_cfg)
        
        self.num_query = num_query
        self.pc_range = pc_range
        self.bev_grid_size = bev_grid_size
        self.enabled = enabled
        self.num_clusters = num_clusters
        
        # 关键：固定使用 R_MAX = 65.0
        self.polar_radius = R_MAX
        
        # I_radar 参数
        self.d_ref = d_ref
        self.d_lambda = d_lambda
        self.w_d_max = w_d_max
        self.d_min = d_min
        self.v_max = v_max
        self.v_ref = v_ref
        self.beta = beta
        
        # 打分场参数
        self.base_bias = base_bias
        self.epsilon = epsilon
        self.diffusion_gamma = diffusion_gamma
        self.diffusion_s_max = diffusion_s_max
        
        # 默认值
        self.z_default = z_default
        self.w_default = w_default
        self.l_default = l_default
        self.h_default = h_default
        
        # 其他参数
        self.max_points = max_points
        self._d_alpha = d_alpha
        
        # 计算 BEV 网格参数
        # 【修复】分别计算 x/y 范围，并断言正方形
        x_range = pc_range[3] - pc_range[0]
        y_range = pc_range[4] - pc_range[1]
        assert abs(x_range - y_range) < 1e-6, \
            f"RWHI 要求 pc_range x/y 对称，但 x_range={x_range}, y_range={y_range}"
        self.map_size = x_range  # 102.4
        self.grid_resolution = self.map_size / bev_grid_size
        
        # 初始化子模块
        self.alpha_mlp = AlphaMLP(
            in_dim=alpha_mlp_in_dim,
            hidden_dim=alpha_mlp_hidden,
            init_bias=alpha_init_bias
        )
        
        self.alpha_encoder = AlphaEncoder(
            d_alpha=d_alpha,
            hidden_dim=alpha_encoder_hidden
        )
        
        # 空间扩散层 (MaxPool3x3)
        self.diffusion = nn.MaxPool2d(
            kernel_size=3,
            stride=1,
            padding=1
        )
        
        # 预计算 BEV 网格坐标
        self._init_grid()
        
        # 预计算确定性微扰场 (hash-based)
        self._init_jitter_field()
    
    def _init_grid(self):
        """初始化 BEV 网格坐标"""
        H = W = self.bev_grid_size
        
        # 计算每个格子中心的物理坐标
        x_min, y_min = self.pc_range[0], self.pc_range[1]
        
        # 生成网格坐标
        y_coords = torch.linspace(
            y_min + self.grid_resolution / 2,
            y_min + self.map_size - self.grid_resolution / 2,
            H
        )
        x_coords = torch.linspace(
            x_min + self.grid_resolution / 2,
            x_min + self.map_size - self.grid_resolution / 2,
            W
        )
        
        # [H, W, 2] 物理坐标
        yy, xx = torch.meshgrid(y_coords, x_coords, indexing='ij')
        grid_xy = torch.stack([xx, yy], dim=-1)  # [H, W, 2]
        
        # 注册为 buffer (不参与梯度更新)
        self.register_buffer('grid_xy', grid_xy.reshape(-1, 2))  # [H*W, 2]
    
    def _init_jitter_field(self):
        """初始化确定性微扰场 (hash-based)"""
        H = W = self.bev_grid_size
        
        # 使用固定种子生成确定性微扰
        generator = torch.Generator()
        generator.manual_seed(42)
        
        jitter = torch.rand(1, 1, H, W, generator=generator) - 0.5  # [-0.5, 0.5]
        
        # 注册为 buffer
        self.register_buffer('jitter_field', jitter)
    
    @property
    def d_alpha(self):
        """返回 α embedding 维度"""
        return self._d_alpha
    
    @property
    def safety_anchors(self):
        """
        返回安全锚点 [num_query, 10]，用于初始化 init_query_bbox
        
        生成均匀极坐标分布作为安全默认值
        
        【注意】处理 num_query 不能被 num_clusters 整除的情况：
        - 使用 ceil 确保生成足够的锚点
        - 最后截取 num_query 个
        """
        # 【修复】使用配置的 num_clusters 而不是硬编码的 6
        num_clusters = self.num_clusters
        # 【修复】使用 ceil 处理不整除情况，确保锚点数量 >= num_query
        num_angles = math.ceil(self.num_query / num_clusters)
        
        angles = torch.linspace(0, 1, num_angles + 1)[:-1]  # [0, 1)
        distances = torch.linspace(0, 1, num_clusters + 2)[1:-1]  # (0, 1)
        
        # 组合
        angles = angles.view(num_angles, 1).expand(num_angles, num_clusters)
        distances = distances.view(1, num_clusters).expand(num_angles, num_clusters)
        
        theta_d = torch.stack([angles, distances], dim=-1).flatten(0, 1)  # [num_angles * num_clusters, 2]
        
        # 【修复】截取到 num_query 个，处理不整除情况
        if theta_d.shape[0] > self.num_query:
            theta_d = theta_d[:self.num_query]
        
        # 确保 d 在 (EPS, 1-EPS) 范围内
        theta_d[:, 1] = theta_d[:, 1].clamp(min=EPS, max=1.0 - EPS)
        
        # 构建完整 10 维
        Q = theta_d.shape[0]
        anchors = torch.zeros(Q, 10)
        
        # θ, d
        anchors[:, 0] = theta_d[:, 0]
        anchors[:, 1] = theta_d[:, 1]
        
        # z (归一化)
        anchors[:, 2] = self.z_default
        
        # w, l, h (log 空间)
        anchors[:, 3] = math.log(max(self.w_default, 0.1))
        anchors[:, 4] = math.log(max(self.l_default, 0.1))
        anchors[:, 5] = math.log(max(self.h_default, 0.1))
        
        # sin, cos (yaw=0)
        anchors[:, 6] = 0.0  # sin(0)
        anchors[:, 7] = 1.0  # cos(0)
        
        # vx, vy
        anchors[:, 8] = 0.0
        anchors[:, 9] = 0.0
        
        return anchors
    
    def _prepare_alpha_input(self, radar_points, radar_mask):
        """
        准备 AlphaMLP 的输入特征
        
        Args:
            radar_points: [B, M, C] 雷达点 (x, y, z, rcs, v_r, ...)
            radar_mask: [B, M] 有效点掩码
        
        Returns:
            f_in: [B, M, 3] AlphaMLP 输入 [log1p(rcs), d_norm, v_norm]
        """
        x = radar_points[..., 0]
        y = radar_points[..., 1]
        rcs = radar_points[..., 3]
        v_r = radar_points[..., 4]
        
        # 距离
        dist = torch.sqrt(x ** 2 + y ** 2).clamp(min=self.d_min, max=R_MAX)
        d_norm = dist / R_MAX
        
        # 速度归一化
        v_norm = v_r.abs() / self.v_max
        v_norm = v_norm.clamp(max=1.0)
        
        # RCS 预处理
        rcs_proc = torch.log1p(F.relu(rcs))
        
        # 组合输入
        f_in = torch.stack([rcs_proc, d_norm, v_norm], dim=-1)  # [B, M, 3]
        
        return f_in
    
    def _compute_i_radar(self, radar_points, radar_mask, alpha):
        """
        计算 I_radar = α * log(1 + σ * w_v * w_d)
        
        Args:
            radar_points: [B, M, C] 雷达点
            radar_mask: [B, M] 有效点掩码
            alpha: [B, M, 1] 点置信度
        
        Returns:
            i_radar: [B, M] 每个点的 I_radar 值
        """
        x = radar_points[..., 0]
        y = radar_points[..., 1]
        rcs = radar_points[..., 3]
        v_r = radar_points[..., 4]
        
        dist = torch.sqrt(x ** 2 + y ** 2).clamp(min=self.d_min, max=R_MAX)
        
        # RCS 预处理
        sigma_proc = F.relu(rcs)
        
        # 速度权重: w_v = 1 + β * sigmoid(|v|/v_ref)
        w_v = 1.0 + self.beta * torch.sigmoid(v_r.abs() / self.v_ref)
        
        # 距离权重: w_d = min((d/d_ref)^λ, w_d_max)
        w_d = (dist / self.d_ref).pow(self.d_lambda).clamp(max=self.w_d_max)
        
        # I_radar = α * log(1 + σ * w_v * w_d)
        phys_term = 1.0 + sigma_proc * w_v * w_d
        alpha_squeezed = alpha.squeeze(-1)  # [B, M]
        i_radar = alpha_squeezed * torch.log(phys_term)
        
        # 应用 mask
        if radar_mask is not None:
            i_radar = i_radar * radar_mask.float()
        
        return i_radar
    
    def _scatter_to_bev(self, radar_points, values, radar_mask, batch_size, device):
        """
        将点级别的值散射到 BEV 网格
        
        Args:
            radar_points: [B, M, C] 雷达点
            values: [B, M] 要散射的值
            radar_mask: [B, M] 有效点掩码
            batch_size: batch 大小
            device: 设备
        
        Returns:
            bev_map: [B, 1, H, W] BEV 网格值
        """
        H = W = self.bev_grid_size
        x_min, y_min = self.pc_range[0], self.pc_range[1]
        
        x = radar_points[..., 0]
        y = radar_points[..., 1]
        
        # 计算网格索引
        grid_x = ((x - x_min) / self.grid_resolution).long()
        grid_y = ((y - y_min) / self.grid_resolution).long()
        
        # 裁剪到有效范围
        grid_x = grid_x.clamp(0, W - 1)
        grid_y = grid_y.clamp(0, H - 1)
        
        # 线性索引
        linear_idx = grid_y * W + grid_x  # [B, M]
        
        # 【修复】AMP/FP16 兼容性：使用 values 的 dtype 初始化 bev_map
        bev_map = torch.zeros(batch_size, H * W, device=device, dtype=values.dtype)
        
        # 应用 mask
        if radar_mask is not None:
            values = values * radar_mask.to(values.dtype)
        
        # Scatter-add
        for b in range(batch_size):
            bev_map[b].scatter_add_(0, linear_idx[b], values[b])
        
        bev_map = bev_map.view(batch_size, 1, H, W)
        
        return bev_map
    
    def _build_score_map(self, i_radar_map, batch_size, device):
        """
        构建三层打分场
        
        S = C_base + I_radar + ε*J
        S_final = S + γ*(MaxPool(S) - S)
        
        Args:
            i_radar_map: [B, 1, H, W] I_radar 网格
            batch_size: batch 大小
            device: 设备
        
        Returns:
            S_final: [B, 1, H, W] 最终打分图
        """
        H = W = self.bev_grid_size
        dtype = i_radar_map.dtype  # 【修复】AMP/FP16 兼容性
        
        # 安全层
        base_field = torch.full(
            (batch_size, 1, H, W),
            self.base_bias,
            device=device,
            dtype=dtype  # 【修复】使用与输入相同的 dtype
        )
        
        # 微扰层 (确定性 hash-based)
        jitter = self.epsilon * self.jitter_field.to(device=device, dtype=dtype)
        
        # 组合
        S = base_field + i_radar_map + jitter
        
        # 残差式空间扩散
        S_pool = self.diffusion(S)
        S_final = S + self.diffusion_gamma * (S_pool - S)
        
        # 得分上限裁剪
        if self.diffusion_s_max > 0:
            S_final = S_final.clamp(max=self.diffusion_s_max)
        
        return S_final
    
    def _xy_to_theta_d(self, xy):
        """
        将 (x, y) 物理坐标转换为归一化极坐标
        
        关键:
        - 使用固定 R_MAX = 65.0
        - d 必须 clamp 到 (EPS, 1-EPS)
        
        Args:
            xy: [..., 2] 物理坐标
        
        Returns:
            theta_d: [..., 2] 归一化极坐标 (θ, d)
        """
        x, y = xy[..., 0], xy[..., 1]
        
        # θ: atan2 → [0, 2π) → [0, 1)
        theta_rad = torch.atan2(y, x)
        theta_rad = (theta_rad + 2 * math.pi) % (2 * math.pi)
        theta_norm = theta_rad / (2 * math.pi)
        theta_norm = theta_norm.clamp(min=0.0, max=1.0 - EPS)
        
        # d: 使用固定 R_MAX = 65.0
        dist = torch.sqrt(x ** 2 + y ** 2).clamp(min=1e-6)
        d_norm = (dist / R_MAX).clamp(min=EPS, max=1.0 - EPS)
        
        return torch.stack([theta_norm, d_norm], dim=-1)
    
    def _topk_to_anchors(self, S, alpha_map, batch_size, device):
        """
        从打分图提取 Top-K 位置，转换为 10 维锚点
        
        关键:
        - 输出归一化值，不做 inverse_sigmoid
        - d/z 必须 clamp 到 (EPS, 1-EPS)
        
        Args:
            S: [B, 1, H, W] 打分图
            alpha_map: [B, 1, H, W] α 网格
            batch_size: batch 大小
            device: 设备
        
        Returns:
            anchors: [B, K, 10] 10 维锚点
            alpha_topk: [B, K, 1] Top-K 位置的 α 值
        """
        H = W = self.bev_grid_size
        K = self.num_query
        dtype = S.dtype  # 【修复】AMP/FP16 兼容性
        
        # Top-K
        S_flat = S.view(batch_size, -1)  # [B, H*W]
        _, topk_idx = torch.topk(S_flat, K, dim=1)  # [B, K]
        
        # 获取 α
        alpha_flat = alpha_map.view(batch_size, -1)  # [B, H*W]
        alpha_topk = torch.gather(alpha_flat, 1, topk_idx)  # [B, K]
        alpha_topk = alpha_topk.clamp(0.0, 1.0).unsqueeze(-1)  # [B, K, 1]
        
        # 坐标转换: 网格索引 → 物理坐标 → 极坐标
        # grid_xy: [H*W, 2]
        topk_xy = self.grid_xy.to(device=device, dtype=dtype)[topk_idx]  # [B, K, 2]
        theta_d = self._xy_to_theta_d(topk_xy)  # [B, K, 2]
        
        # z_norm (已 clamp)
        z = torch.full(
            (batch_size, K, 1),
            self.z_default,
            device=device,
            dtype=dtype  # 【修复】
        ).clamp(min=EPS, max=1.0 - EPS)
        
        # wlh (log 空间)
        w_log = math.log(max(self.w_default, 0.1))
        l_log = math.log(max(self.l_default, 0.1))
        h_log = math.log(max(self.h_default, 0.1))
        
        # 组装 10 维 【修复】所有 torch.full 添加 dtype
        anchors = torch.cat([
            theta_d,  # [B, K, 2] (θ, d)
            z,        # [B, K, 1]
            torch.full((batch_size, K, 1), w_log, device=device, dtype=dtype),  # w (log)
            torch.full((batch_size, K, 1), l_log, device=device, dtype=dtype),  # l (log)
            torch.full((batch_size, K, 1), h_log, device=device, dtype=dtype),  # h (log)
            torch.full((batch_size, K, 1), 0.0, device=device, dtype=dtype),    # sin(yaw)
            torch.full((batch_size, K, 1), 1.0, device=device, dtype=dtype),    # cos(yaw)
            torch.full((batch_size, K, 1), 0.0, device=device, dtype=dtype),    # vx
            torch.full((batch_size, K, 1), 0.0, device=device, dtype=dtype),    # vy
        ], dim=-1)  # [B, K, 10]
        
        return anchors, alpha_topk
    
    def encode_alpha(self, alpha_values):
        """
        将 α 编码为 embedding
        
        Args:
            alpha_values: [B, K, 1] α 值
        
        Returns:
            embedding: [B, K, d_alpha] α embedding
        """
        return self.alpha_encoder(alpha_values)
    
    def forward(self, radar_points, radar_mask):
        """
        RWHI 前向传播
        
        Args:
            radar_points: [B, M, C] 雷达点 (x, y, z, rcs, v_r, ...)
            radar_mask: [B, M] 有效点掩码 (1=有效, 0=padding)
        
        Returns:
            anchors: [B, K, 10] 归一化 bbox，不做 inverse_sigmoid
            alpha_values: [B, K, 1] 每个锚点的 α 值
        """
        if not self.enabled:
            # 返回安全锚点
            B = radar_points.shape[0]
            device = radar_points.device
            dtype = radar_points.dtype  # 【修复】AMP/FP16 兼容性
            anchors = self.safety_anchors.unsqueeze(0).expand(B, -1, -1).to(device=device, dtype=dtype)
            alpha_values = torch.full((B, self.num_query, 1), 0.5, device=device, dtype=dtype)
            return anchors, alpha_values
        
        B, M, C = radar_points.shape
        device = radar_points.device
        
        # 1. 计算每个点的 α
        f_in = self._prepare_alpha_input(radar_points, radar_mask)  # [B, M, 3]
        alpha = self.alpha_mlp(f_in)  # [B, M, 1]
        
        # 2. 计算 I_radar
        i_radar = self._compute_i_radar(radar_points, radar_mask, alpha)  # [B, M]
        
        # 3. Scatter-Add 聚合到 BEV 网格
        i_radar_map = self._scatter_to_bev(
            radar_points, i_radar, radar_mask, B, device
        )  # [B, 1, H, W]
        
        # 4. 同时聚合 α (用于后续提取)
        # 这里简单取每个格子的平均 α
        alpha_squeezed = alpha.squeeze(-1)  # [B, M]
        alpha_sum_map = self._scatter_to_bev(
            radar_points, alpha_squeezed, radar_mask, B, device
        )
        count_map = self._scatter_to_bev(
            radar_points,
            torch.ones_like(alpha_squeezed),
            radar_mask, B, device
        )
        alpha_map = alpha_sum_map / (count_map + 1e-6)
        
        # 对无雷达点的区域设置默认 α
        no_radar_mask = (count_map < 0.5)
        # 【修复】AMP/FP16 兼容性：使用与 alpha_map 相同的 dtype
        alpha_map = torch.where(no_radar_mask, 
                                torch.tensor(0.5, device=device, dtype=alpha_map.dtype), 
                                alpha_map)
        
        # 5. 构建三层打分场
        S_final = self._build_score_map(i_radar_map, B, device)  # [B, 1, H, W]
        
        # 6. Top-K 选取
        anchors, alpha_topk = self._topk_to_anchors(
            S_final, alpha_map, B, device
        )  # [B, K, 10], [B, K, 1]
        
        return anchors, alpha_topk


# 工厂函数，用于根据版本创建 RWHI 模块
def build_rwhi(rwhi_version='v7', **kwargs):
    """
    构建 RWHI 模块
    
    Args:
        rwhi_version: RWHI 版本 ('v7', 'v5.3' 等)
        **kwargs: 传递给 RWHIModule 的参数
    
    Returns:
        rwhi_module: RWHI 模块实例
    """
    # 目前只有 v7 版本，后续可扩展
    return RWHIModule(**kwargs)

