# -*- coding: utf-8 -*-
# ==============================================================================
# RWHI v7 (Radar-Weighted Hybrid Initialization)
# ==============================================================================
# 基于 gpt-rwhi-v7.txt 设计文档实现
#
# 核心设计 - 三层打分场 + 残差式空间扩散:
#   S(x,y) = C_base + I_radar(x,y) + ε * J(x,y)
#   S_final = S + γ * (MaxPool3x3(S) - S)
#
# I_radar 公式:
#   I_radar = α * log(1 + σ_proc * w_v * w_d)
#   - α: 雷达置信门控 (MLP 学习)
#   - w_v = 1 + β * sigmoid(|v|/v_ref): 速度权重
#   - w_d = min((d/d_ref)^λ, w_d_max): 距离权重
#
# 完全兼容 RaCFormer bbox 结构 (见 racformer_query_generation_points.md)
# ==============================================================================

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.runner import BaseModule

from .bbox.utils import compute_map_size_and_radius


class AlphaMLP(nn.Module):
    """
    轻量级 MLP 用于计算每个雷达点的置信度 α ∈ (0, 1)
    
    输入特征 f_in:
    - log1p(max(rcs_raw, 0)): RCS 强度 (log1p 压缩)
    - d_norm: 距离归一化 d / R_MAX
    - v_norm: 速度归一化 v / v_max
    
    输出:
    - α ∈ (0, 1): 点置信度
    """
    
    def __init__(self, in_dim=3, hidden_dim=32, init_bias=1.0):
        """
        Args:
            in_dim: 输入特征维度 (默认 3: rcs_log, d_norm, v_norm)
            hidden_dim: 隐藏层维度
            init_bias: 最后一层 bias 初始化值 (正值使初始 α 偏大)
        """
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 1),
        )
        
        # 初始化: 使初始 α ≈ 0.7-0.9
        for module in self.net:
            if isinstance(module, nn.Linear):
                nn.init.kaiming_uniform_(module.weight, a=math.sqrt(5))
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        
        # 设置最后一层 bias 使初始 α 偏大
        self.net[-1].bias.data.fill_(init_bias)
    
    def forward(self, f_in):
        """
        Args:
            f_in: [N_points, in_dim] 或 [B, M, in_dim]
            
        Returns:
            alpha: [N_points, 1] 或 [B, M, 1], 值域 (0, 1)
        """
        logits = self.net(f_in)
        alpha = torch.sigmoid(logits)
        return alpha


class AlphaEncoder(nn.Module):
    """
    将 α 值编码为 d_alpha 维 embedding，用于 Feature-Guided Initialization
    
    不修改 bbox_proposal 的维度，仅用于 Query 特征增强
    """
    
    def __init__(self, d_alpha=2, hidden_dim=8):
        """
        Args:
            d_alpha: 输出 embedding 维度 (推荐 2 或 4)
            hidden_dim: 隐藏层维度
        """
        super().__init__()
        self.d_alpha = d_alpha
        
        # 输入: [alpha_center, alpha_center^2]
        self.fc1 = nn.Linear(2, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, d_alpha)
        
        # 初始化
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.zeros_(self.fc1.bias)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)
    
    def forward(self, alpha):
        """
        Args:
            alpha: [N_query] 或 [B, K] 或 [B, K, 1], 值域 (0, 1)
            
        Returns:
            emb: [N_query, d_alpha] 或 [B, K, d_alpha]
        """
        # 移除最后一维如果是 1
        if alpha.dim() >= 2 and alpha.shape[-1] == 1:
            alpha = alpha.squeeze(-1)
        
        # 中心化 + 缩放: (0, 1) -> (-1, 1)
        SCALE = 2.0
        alpha_center = (alpha - 0.5) * SCALE
        
        # 构造 2 维输入: [alpha_center, alpha_center^2]
        u = torch.stack([alpha_center, alpha_center ** 2], dim=-1)
        
        # MLP
        h = F.relu(self.fc1(u))
        emb = self.fc2(h)
        
        return emb


class RWHI_v53(BaseModule):
    """
    RWHI v7: Radar-Weighted Hybrid Initialization
    
    基于三层打分场 + 残差式空间扩散:
        S(x,y) = C_base + I_radar(x,y) + ε * J(x,y)
        S_final = S + γ * (MaxPool3x3(S) - S)
    
    I_radar 公式 (v7 完整版):
        I_radar = α * log(1 + σ_proc * w_v * w_d)
        - w_v = 1 + β * sigmoid(|v|/v_ref): 速度权重，强调动态目标
        - w_d = min((d/d_ref)^λ, w_d_max): 距离权重，兼顾远场覆盖
    
    输入:
        - radar_points: [B, M, C] (x, y, z, rcs, v_r, ...)
        - radar_mask: [B, M] padding 掩码 (1=有效, 0=padding)
    
    输出:
        - anchors: [B, K, 10] (theta, d, z_norm, w_log, l_log, h_log, sin, cos, vx, vy)
        - alpha_values: [B, K, 1] 每个锚点的 α 值
    
    bbox 10维结构 (兼容 racformer_query_generation_points.md):
        - 0: θ (归一化角度 [0,1] → [0, 2π])
        - 1: d (归一化距离 [0,1] → [0, r=65m], 需 clamp(eps, 1-eps))
        - 2: z (归一化高度 [0,1] → [-5, 3]m, 需 clamp(eps, 1-eps))
        - 3-5: w, l, h (Log 空间)
        - 6-7: sin(yaw), cos(yaw) (独立预测, 初始化 sin=0, cos=1)
        - 8-9: vx, vy (速度)
    """
    
    def __init__(
        self,
        num_query=900,
        pc_range=(-51.2, -51.2, -5.0, 51.2, 51.2, 3.0),
        bev_grid_size=100,
        grid_size_h=None,
        grid_size_w=None,
        # 物理参数 - 距离权重 w_d
        d_ref=30.0,
        d_lambda=1.0,
        w_d_max=2.5,
        d_min=2.0,
        # 物理参数 - 速度权重 w_v (v7 新增)
        v_max=30.0,
        v_ref=10.0,      # v7: 速度参考值 (m/s)
        beta=1.0,        # v7: 速度权重强度 β, w_v ∈ [1, 1+β]
        # 安全场参数
        base_bias=1.0,
        epsilon=0.01,
        # 空间扩散参数 (v7 新增)
        diffusion_gamma=0.5,   # v7: 残差融合系数 γ ∈ [0.3, 0.7]
        diffusion_s_max=5.0,   # v7: 可选得分上限裁剪 S_max ≈ 4.5~5.0, 0 表示不裁剪
        # 默认值
        z_default=0.5,
        w_default=1.8,
        l_default=4.0,
        h_default=1.5,
        # AlphaMLP 配置
        alpha_mlp_hidden=32,
        alpha_mlp_in_dim=3,
        alpha_init_bias=1.0,
        # AlphaEncoder 配置
        d_alpha=2,
        alpha_encoder_hidden=8,
        # 其他配置
        max_points=None,
        radar_channel_map=None,
        enabled=True,
        init_cfg=None,
        **kwargs,
    ):
        super().__init__(init_cfg=init_cfg)
        
        # 忽略未知参数
        _ = kwargs
        
        self.num_query = int(num_query)
        self.pc_range = pc_range
        self.enabled = enabled
        
        # 物理参数 - 距离权重 w_d
        self.d_ref = float(max(d_ref, 1e-6))
        self.d_lambda = float(d_lambda)
        self.w_d_max = float(w_d_max)
        self.d_min = float(d_min)
        
        # 物理参数 - 速度权重 w_v (v7)
        self.v_max = float(v_max)
        self.v_ref = float(max(v_ref, 1e-6))
        self.beta = float(beta)
        
        # 安全场参数
        self.base_bias = float(base_bias)
        self.epsilon = float(epsilon)
        
        # 空间扩散参数 (v7)
        self.diffusion_gamma = float(diffusion_gamma)
        self.diffusion_s_max = float(diffusion_s_max)
        
        # 默认值
        self.z_default = float(z_default)
        self.w_default = float(w_default)
        self.l_default = float(l_default)
        self.h_default = float(h_default)
        
        # α 相关
        self.d_alpha = d_alpha
        
        # 其他
        self.max_points = max_points
        
        # BEV 网格尺寸配置
        if bev_grid_size is not None:
            if isinstance(bev_grid_size, (list, tuple)):
                grid_size_h, grid_size_w = bev_grid_size
            else:
                grid_size_h = bev_grid_size
                grid_size_w = bev_grid_size
        if grid_size_h is None or grid_size_w is None:
            raise ValueError('grid_size_h/grid_size_w must be set.')
        self.grid_size_h = int(grid_size_h)
        self.grid_size_w = int(grid_size_w)
        
        # 计算 map_size 和 polar_radius (R_MAX = 65m)
        self.map_size, self.polar_radius = compute_map_size_and_radius(self.pc_range)
        
        # 雷达通道映射
        self.radar_channel_map = radar_channel_map or {
            'x': 0, 'y': 1, 'z': 2, 'rcs': 3, 'v_r': 4
        }
        
        # 预计算网格
        self._init_bev_grid()
        
        # 预计算 jitter field
        self._init_jitter_field()
        
        # 构建初始 anchors
        self.register_buffer('init_anchors', self._build_init_anchors())
        
        # AlphaMLP - 计算每个点的置信度
        self.alpha_mlp = AlphaMLP(
            in_dim=alpha_mlp_in_dim,
            hidden_dim=alpha_mlp_hidden,
            init_bias=alpha_init_bias
        )
        
        # AlphaEncoder - 将 α 编码为 embedding
        self.alpha_encoder = AlphaEncoder(
            d_alpha=d_alpha,
            hidden_dim=alpha_encoder_hidden
        )
        
        # 空间扩散 (MaxPool2d)
        self.diffusion = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        
        # 验证
        total_cells = self.grid_size_h * self.grid_size_w
        if self.num_query > total_cells:
            raise ValueError(
                f'num_query({self.num_query}) > H*W({total_cells}).'
            )
    
    def _init_bev_grid(self):
        """预计算 BEV 网格坐标"""
        x_min, y_min, _, x_max, y_max, _ = self.pc_range
        cell_size_x = (x_max - x_min) / float(self.grid_size_w)
        cell_size_y = (y_max - y_min) / float(self.grid_size_h)
        
        # 网格中心坐标
        x_coords = (torch.arange(self.grid_size_w, dtype=torch.float32) + 0.5) * cell_size_x + x_min
        y_coords = (torch.arange(self.grid_size_h, dtype=torch.float32) + 0.5) * cell_size_y + y_min
        yy, xx = torch.meshgrid(y_coords, x_coords, indexing='ij')
        grid_xy = torch.stack([xx, yy], dim=-1).reshape(-1, 2)  # [H*W, 2]
        
        self.register_buffer('grid_xy', grid_xy)
        self.cell_size_x = cell_size_x
        self.cell_size_y = cell_size_y
    
    def _init_jitter_field(self):
        """
        预计算确定性 hash-based jitter field
        J(i, j) = 2 * frac(sin(a*x + b*y + c) * d) - 1 ∈ [-1, 1]
        """
        H, W = self.grid_size_h, self.grid_size_w
        device = torch.device('cpu')  # 将在 forward 中移动到正确设备
        
        a, b, c, d = 12.9898, 78.233, 37.719, 43758.5453
        
        ys, xs = torch.meshgrid(
            torch.arange(H, device=device, dtype=torch.float32),
            torch.arange(W, device=device, dtype=torch.float32),
            indexing='ij'
        )
        
        h = torch.sin(a * xs + b * ys + c) * d
        frac = h - torch.floor(h)
        J = 2.0 * frac - 1.0
        
        self.register_buffer('jitter_field', J)
    
    def _build_init_anchors(self):
        """
        构建初始均匀安全锚点 (用于 init_query_bbox 初始化)
        输出为 Logits 形式
        """
        device = self.grid_xy.device
        total_cells = self.grid_size_h * self.grid_size_w
        
        # 均匀步进采样索引
        base_idx = (torch.arange(self.num_query, dtype=torch.long, device=device) * total_cells) // self.num_query
        xy = self.grid_xy[base_idx]
        
        # 转换为极坐标
        theta_d = self._xy_to_theta_d(xy)  # [num_query, 2], 归一化到 [0, 1]
        z = torch.full((self.num_query, 1), self.z_default, device=device, dtype=theta_d.dtype)
        
        # 转换为 Logits
        theta_d = self._inverse_sigmoid(theta_d)
        z = self._inverse_sigmoid(z)
        
        return torch.cat([theta_d, z], dim=-1)
    
    @staticmethod
    def _inverse_sigmoid(x, eps=1e-5):
        """逆 Sigmoid 函数"""
        x = x.clamp(min=eps, max=1.0 - eps)
        return torch.log(x / (1.0 - x))
    
    def _xy_to_theta_d(self, xy):
        """
        将 (x, y) 物理坐标转换为归一化极坐标 (theta_norm, d_norm)
        
        - theta_rad = atan2(y, x) → [0, 2π) → theta_norm = theta_rad / (2π) ∈ [0, 1)
        - d = sqrt(x² + y²) → d_norm = d / R_MAX ∈ (EPS, 1-EPS)
        """
        x = xy[..., 0]
        y = xy[..., 1]
        
        # 使用 atan2 计算角度
        theta_rad = torch.atan2(y, x)
        theta_rad = (theta_rad + 2 * math.pi) % (2 * math.pi)  # [0, 2π)
        theta_norm = theta_rad / (2 * math.pi)  # [0, 1)
        
        # 计算距离
        dist = torch.sqrt(x ** 2 + y ** 2).clamp(min=1e-6)
        d_norm = (dist / self.polar_radius).clamp(min=1e-5, max=1.0 - 1e-5)
        
        theta_norm = theta_norm.clamp(min=0.0, max=1.0 - 1e-5)
        
        return torch.stack([theta_norm, d_norm], dim=-1)
    
    def _prepare_point_features(self, radar_points, radar_mask):
        """
        准备 AlphaMLP 的输入特征
        
        f_in = [log1p(max(rcs, 0)), d_norm, v_norm]
        """
        ch = self.radar_channel_map
        
        x = radar_points[..., ch['x']]
        y = radar_points[..., ch['y']]
        rcs = radar_points[..., ch['rcs']]
        v_r = radar_points[..., ch['v_r']]
        
        # RCS 特征: log1p 压缩
        rcs_log = torch.log1p(F.relu(rcs))
        
        # 距离归一化
        dist = torch.sqrt(x ** 2 + y ** 2).clamp(min=self.d_min, max=self.polar_radius)
        d_norm = dist / self.polar_radius
        
        # 速度归一化
        v_norm = (v_r / self.v_max).clamp(min=-1.0, max=1.0)
        
        # 构造特征向量
        f_in = torch.stack([rcs_log, d_norm, v_norm], dim=-1)
        
        return f_in, dist
    
    def _compute_alpha(self, radar_points, radar_mask):
        """
        使用 AlphaMLP 计算每个雷达点的置信度 α
        """
        f_in, _ = self._prepare_point_features(radar_points, radar_mask)
        
        # AlphaMLP 前向
        alpha = self.alpha_mlp(f_in)  # [B, M, 1]
        
        # 应用 mask
        if radar_mask is not None:
            alpha = alpha * radar_mask.unsqueeze(-1)
        
        return alpha
    
    def _compute_i_radar(self, radar_points, radar_mask, alpha):
        """
        计算改进的物理增强项 I_radar (v7 完整版)
        
        I_radar(d, σ, v) = α * log(1 + σ_proc * w_v(v) * w_d(d))
        
        其中:
        - σ_proc = max(0, rcs): RCS 预处理，确保非负
        - w_v(v) = 1 + β * sigmoid(|v|/v_ref): 速度权重，w_v ∈ [1, 1+β]
        - w_d(d) = min((d/d_ref)^λ, w_d_max): 距离权重
        
        外层 log(1+...) 做整体压缩，防止极端数值放大
        """
        ch = self.radar_channel_map
        
        x = radar_points[..., ch['x']]
        y = radar_points[..., ch['y']]
        rcs = radar_points[..., ch['rcs']]
        v_r = radar_points[..., ch['v_r']]
        
        # 距离
        dist = torch.sqrt(x ** 2 + y ** 2).clamp(min=self.d_min, max=self.polar_radius)
        
        # RCS 处理 (σ_proc): 确保非负
        sigma_proc = F.relu(rcs)
        
        # 速度权重 w_v (v7): w_v = 1 + β * sigmoid(|v|/v_ref)
        # 对有显著径向速度的目标给轻微增益，静态目标 w_v ≈ 1
        v_abs_norm = torch.abs(v_r) / self.v_ref
        w_v = 1.0 + self.beta * torch.sigmoid(v_abs_norm)
        
        # 距离权重 w_d: min((d/d_ref)^λ, w_d_max)
        w_d = (dist / self.d_ref).pow(self.d_lambda)
        w_d = w_d.clamp(max=self.w_d_max)
        
        # 物理项: 1 + σ * w_v * w_d ≥ 1
        phys_term = 1.0 + sigma_proc * w_v * w_d
        
        # I_radar = α * log(phys_term)
        # log(1+...) 做整体压缩，防止极端放大
        alpha_squeezed = alpha.squeeze(-1) if alpha.dim() > 2 else alpha
        i_radar_point = alpha_squeezed * torch.log(phys_term)
        
        # 应用 mask
        if radar_mask is not None:
            i_radar_point = i_radar_point * radar_mask
        
        return i_radar_point
    
    def _scatter_to_bev(self, radar_points, values, batch_size, device):
        """
        使用 scatter-add 将点级别的值累加到 BEV 网格
        """
        ch = self.radar_channel_map
        x = radar_points[..., ch['x']]
        y = radar_points[..., ch['y']]
        
        x_min, y_min, _, x_max, y_max, _ = self.pc_range
        
        # 计算网格索引
        ix = torch.floor((x - x_min) / (x_max - x_min) * self.grid_size_w).long()
        iy = torch.floor((y - y_min) / (y_max - y_min) * self.grid_size_h).long()
        ix = ix.clamp(0, self.grid_size_w - 1)
        iy = iy.clamp(0, self.grid_size_h - 1)
        
        # 展平索引
        flat_idx = iy * self.grid_size_w + ix
        batch_offset = torch.arange(batch_size, device=device).view(batch_size, 1) * (self.grid_size_h * self.grid_size_w)
        flat_idx = (flat_idx + batch_offset).view(-1)
        flat_values = values.view(-1)
        
        # scatter-add
        field_flat = torch.zeros(
            batch_size * self.grid_size_h * self.grid_size_w,
            device=device,
            dtype=radar_points.dtype,
        )
        field_flat.scatter_add_(0, flat_idx, flat_values)
        field = field_flat.view(batch_size, 1, self.grid_size_h, self.grid_size_w)
        
        return field
    
    def _build_score_map(self, i_radar_map, device):
        """
        构建打分图并应用残差式空间扩散 (v7)
        
        Step 1: 三层打分场
            S(x, y) = C_base + I_radar(x, y) + ε * J(x, y)
        
        Step 2: 残差式空间扩散 (v7 新增)
            S_pool = MaxPool3x3(S)
            S_diff = S + γ * (S_pool - S)
            可选: S_final = min(S_diff, S_max)
        
        空间扩散的关键性质:
        - S_pool >= S, 因此 S_diff >= S
        - 扩散不会降低任何格子的得分
        - 原本能进 Top-K 的雷达峰值不会因扩散而掉出 Top-K
        """
        B = i_radar_map.shape[0]
        
        # ===== Step 1: 三层打分场 =====
        
        # 安全层: C_base ≈ 1.0
        base_field = i_radar_map.new_full(
            (B, 1, self.grid_size_h, self.grid_size_w),
            self.base_bias
        )
        
        # 稳定层: ε * J(x, y), J ∈ [-1, 1], ε ≈ 0.01
        # 作用: 打破平局，为无雷达区域引入平滑细粒度差异
        J = self.jitter_field.to(device=device, dtype=i_radar_map.dtype)
        J = J.unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]
        jitter = self.epsilon * J
        
        # 组合打分图
        S = base_field + i_radar_map + jitter
        
        # ===== Step 2: 残差式空间扩散 (v7) =====
        # S_diff = S + γ * (MaxPool3x3(S) - S)
        # 推荐 γ ∈ [0.3, 0.7]
        
        S_pool = self.diffusion(S)  # MaxPool3x3
        S_final = S + self.diffusion_gamma * (S_pool - S)
        
        # 可选: 得分上限裁剪
        if self.diffusion_s_max > 0:
            S_final = S_final.clamp(max=self.diffusion_s_max)
        
        return S_final
    
    def _topk_to_anchors(self, S, alpha_map, batch_size, device):
        """
        从打分图中提取 Top-K 位置并转换为 anchors
        """
        EPS = 1e-5
        
        # Top-K 选取
        S_flat = S.view(batch_size, -1)
        _, topk_idx = torch.topk(S_flat, self.num_query, dim=1, largest=True, sorted=True)
        
        # 获取 Top-K 位置的 α 值
        alpha_flat = alpha_map.squeeze(1).view(batch_size, -1)
        alpha_topk = torch.gather(alpha_flat, 1, topk_idx)
        alpha_topk = alpha_topk.clamp(0.0, 1.0).unsqueeze(-1)  # [B, K, 1]
        
        # 从预计算网格解码坐标
        topk_xy = self.grid_xy[topk_idx]  # [B, K, 2]
        
        # 转换为极坐标
        theta_d = self._xy_to_theta_d(topk_xy)  # [B, K, 2]
        
        # z_norm 初始化
        z = torch.full(
            (batch_size, self.num_query, 1),
            self.z_default,
            device=device,
            dtype=theta_d.dtype
        )
        
        # 转换为 Logits
        theta_d = self._inverse_sigmoid(theta_d)
        z = self._inverse_sigmoid(z)
        
        # w, l, h 使用 log 形式
        w_log = math.log(max(self.w_default, 0.1))
        l_log = math.log(max(self.l_default, 0.1))
        h_log = math.log(max(self.h_default, 0.1))
        
        # sin/cos 初始化 (yaw=0)
        sin_init = 0.0
        cos_init = 1.0
        
        # vx, vy 初始化
        vx_init = 0.0
        vy_init = 0.0
        
        # 构建 10 维 bbox_proposal
        # [theta_norm, d_norm, z_norm, w_log, l_log, h_log, sin, cos, vx, vy]
        anchors = torch.cat([
            theta_d,                                                        # [B, K, 2]
            z,                                                              # [B, K, 1]
            torch.full((batch_size, self.num_query, 1), w_log, device=device, dtype=theta_d.dtype),
            torch.full((batch_size, self.num_query, 1), l_log, device=device, dtype=theta_d.dtype),
            torch.full((batch_size, self.num_query, 1), h_log, device=device, dtype=theta_d.dtype),
            torch.full((batch_size, self.num_query, 1), sin_init, device=device, dtype=theta_d.dtype),
            torch.full((batch_size, self.num_query, 1), cos_init, device=device, dtype=theta_d.dtype),
            torch.full((batch_size, self.num_query, 1), vx_init, device=device, dtype=theta_d.dtype),
            torch.full((batch_size, self.num_query, 1), vy_init, device=device, dtype=theta_d.dtype),
        ], dim=-1)
        
        return anchors, alpha_topk
    
    def _fix_length(self, radar_points, radar_mask):
        """固定长度补齐 (保证静态图)"""
        if self.max_points is None:
            return radar_points, radar_mask
        
        B, M, C = radar_points.shape
        
        if radar_mask is not None:
            radar_mask = radar_mask.to(radar_points.dtype)
        
        if M == self.max_points:
            return radar_points, radar_mask
        
        if M > self.max_points:
            radar_points = radar_points[:, :self.max_points, :]
            if radar_mask is not None:
                radar_mask = radar_mask[:, :self.max_points]
            return radar_points, radar_mask
        
        pad_len = self.max_points - M
        radar_points = F.pad(radar_points, (0, 0, 0, pad_len))
        if radar_mask is not None:
            radar_mask = F.pad(radar_mask, (0, pad_len))
        
        return radar_points, radar_mask
    
    def forward(self, radar_points=None, radar_mask=None):
        """
        前向传播
        
        Args:
            radar_points: [B, M, C] 雷达点云 (x, y, z, rcs, v_r, ...)
            radar_mask: [B, M] 有效点掩码 (1=有效, 0=padding)
        
        Returns:
            anchors: [B, K, 10] bbox_proposal
            alpha_values: [B, K, 1] 每个锚点的 α 值 (用于 Feature-Guided Init)
        """
        if not self.enabled:
            B = 1 if radar_points is None else radar_points.shape[0]
            anchors = self.init_anchors.unsqueeze(0).expand(B, -1, -1).clone()
            # 扩展到 10 维
            anchors = self._expand_anchors_to_10d(anchors)
            alpha_values = torch.ones(B, self.num_query, 1, device=anchors.device, dtype=anchors.dtype) * 0.5
            return anchors, alpha_values
        
        if radar_points is None:
            B = 1
            anchors = self.init_anchors.unsqueeze(0).expand(B, -1, -1).clone()
            anchors = self._expand_anchors_to_10d(anchors)
            alpha_values = torch.ones(B, self.num_query, 1, device=anchors.device, dtype=anchors.dtype) * 0.5
            return anchors, alpha_values
        
        # 固定长度
        radar_points, radar_mask = self._fix_length(radar_points, radar_mask)
        
        # 处理 mask
        if radar_mask is None:
            radar_mask = (radar_points.abs().sum(dim=-1) > 0).to(radar_points.dtype)
        else:
            radar_mask = radar_mask.to(radar_points.dtype)
        
        B, M, _ = radar_points.shape
        device = radar_points.device
        
        # 1. 计算每个点的置信度 α
        alpha = self._compute_alpha(radar_points, radar_mask)  # [B, M, 1]
        
        # 2. 计算 I_radar = α * log(1 + σ * w_v * w_d)
        i_radar_point = self._compute_i_radar(radar_points, radar_mask, alpha)  # [B, M]
        
        # 3. Scatter-Add 聚合到 BEV 网格
        # 对真实目标: 多个点落在同一网格，累加后形成稳定高峰
        # 对噪声: 点位分散，少有格子能聚到同等量级
        i_radar_map = self._scatter_to_bev(radar_points, i_radar_point, B, device)  # [B, 1, H, W]
        alpha_weight = alpha.squeeze(-1) * radar_mask
        alpha_map = self._scatter_to_bev(radar_points, alpha_weight, B, device)  # [B, 1, H, W]
        
        # 4. 对 alpha_map 也做扩散 (用于获取 Top-K 位置的 α 值)
        alpha_map = self.diffusion(alpha_map)
        
        # 5. 构建打分图 S 并应用残差式空间扩散 (v7)
        # S = C_base + I_radar + ε*J
        # S_final = S + γ*(MaxPool(S) - S)
        S = self._build_score_map(i_radar_map, device)
        
        # 6. Top-K 选取并转换为 anchors
        anchors, alpha_topk = self._topk_to_anchors(S, alpha_map, B, device)
        
        return anchors, alpha_topk
    
    def _expand_anchors_to_10d(self, anchors_3d):
        """将 [B, K, 3] 扩展到 [B, K, 10]"""
        if anchors_3d.shape[-1] == 10:
            return anchors_3d
        
        B, K, _ = anchors_3d.shape
        device = anchors_3d.device
        dtype = anchors_3d.dtype
        
        anchors_10d = torch.zeros((B, K, 10), device=device, dtype=dtype)
        anchors_10d[..., :3] = anchors_3d
        
        # w, l, h 使用 log 形式
        anchors_10d[..., 3] = math.log(max(self.w_default, 0.1))
        anchors_10d[..., 4] = math.log(max(self.l_default, 0.1))
        anchors_10d[..., 5] = math.log(max(self.h_default, 0.1))
        
        # sin=0, cos=1
        anchors_10d[..., 6] = 0.0
        anchors_10d[..., 7] = 1.0
        
        # vx, vy = 0
        anchors_10d[..., 8:10] = 0.0
        
        return anchors_10d
    
    def encode_alpha(self, alpha_values):
        """
        将 α 值编码为 embedding
        
        Args:
            alpha_values: [B, K, 1] 或 [B, K]
        
        Returns:
            alpha_emb: [B, K, d_alpha]
        """
        return self.alpha_encoder(alpha_values)
    
    @property
    def safety_anchors(self):
        """
        输出 10 维锚点用于 init_query_bbox
        """
        anchors_3d = self.init_anchors  # [num_query, 3]
        num = anchors_3d.shape[0]
        device = anchors_3d.device
        dtype = anchors_3d.dtype
        
        anchors_10d = torch.zeros((num, 10), device=device, dtype=dtype)
        anchors_10d[:, 0:3] = anchors_3d
        
        # w, l, h 使用 log 形式
        anchors_10d[:, 3] = math.log(max(self.w_default, 0.1))
        anchors_10d[:, 4] = math.log(max(self.l_default, 0.1))
        anchors_10d[:, 5] = math.log(max(self.h_default, 0.1))
        
        # sin=0, cos=1
        anchors_10d[:, 6] = 0.0
        anchors_10d[:, 7] = 1.0
        
        return anchors_10d
    
    @property
    def num_safety_anchors(self):
        return self.num_query

