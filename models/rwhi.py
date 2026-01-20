# -*- coding: utf-8 -*-
# ==============================================================================
# RWHI v5.2 (Ultimate)
# - 各向同性安全场 (Uniform Safety Net)
# - 物理增强雷达增益 (d/d_ref)^4
# - 自适应 α-MLP 门控 (抑制远场噪声)
# - ScatterAdd + MaxPool + Global Top-K (TRT 静态图友好)
# ==============================================================================

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.runner import BaseModule

from .bbox.utils import compute_map_size_and_radius

try:
    from .rwhi_legacy import RWHIModule as LegacyRWHIModule
except Exception:
    LegacyRWHIModule = None


class RWHI_Ultimate(BaseModule):
    """
    RWHI v5.2: Isotropic Safety Field + Physics-Enhanced Radar Gain + α-MLP Gating

    输入:
        - radar_points: [B, M, 5] (x, y, z, rcs, v_r)
        - radar_mask:  [B, M] padding 掩码 (1=有效, 0=padding)

    输出:
        - anchors: [B, K, 10] (theta, d, z_norm, ..., vx)
    """

    def __init__(
        self,
        num_query=900,
        pc_range=(-51.2, -51.2, -5.0, 51.2, 51.2, 3.0),
        bev_grid_size=100,
        grid_size_h=None,
        grid_size_w=None,
        d_ref=30.0,
        beta=0.0,
        base_bias=1.0,
        z_default=0.5,
        max_points=None,
        radar_channel_map=None,
        enabled=True,
        init_cfg=None,
        **kwargs,
    ):
        super().__init__(init_cfg=init_cfg)

        # 兼容旧参数但不使用
        _ = kwargs

        self.num_query = int(num_query)
        self.pc_range = pc_range
        self.d_ref = float(max(d_ref, 1e-6))
        self.beta = float(beta)
        self.base_bias = float(base_bias)
        self.z_default = float(z_default)
        self.max_points = max_points
        self.enabled = enabled

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

        self.map_size, self.polar_radius = compute_map_size_and_radius(self.pc_range)

        # 雷达通道映射
        self.radar_channel_map = radar_channel_map or {
            'x': 0, 'y': 1, 'z': 2, 'rcs': 3, 'v_r': 4
        }

        # 预计算网格与初始化 anchors
        self._init_bev_grid()
        self.register_buffer('init_anchors', self._build_init_anchors())

        # α-MLP (point-wise, 无循环)
        self.alpha_fc1 = nn.Linear(5, 64)
        # FIX: 对应改进项 - LayerNorm 替代 BatchNorm
        self.alpha_ln1 = nn.LayerNorm(64)
        self.alpha_fc2 = nn.Linear(64, 64)
        # FIX: 对应改进项 - LayerNorm 替代 BatchNorm
        self.alpha_ln2 = nn.LayerNorm(64)
        self.alpha_fc3 = nn.Linear(64, 1)
        # 冷启动降低雷达置信度，避免未训练噪声污染
        nn.init.constant_(self.alpha_fc3.bias, -4.59)

        # 空间扩散 (MaxPool2d, TRT 友好)
        self.diffusion = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)

        # 静态形状约束
        total_cells = self.grid_size_h * self.grid_size_w
        if self.num_query > total_cells:
            raise ValueError(
                f'num_query({self.num_query}) > H*W({total_cells}).'
            )

    # ============================================================
    # 预计算 BEV 网格
    # ============================================================
    def _init_bev_grid(self):
        x_min, y_min, _, x_max, y_max, _ = self.pc_range
        cell_size_x = (x_max - x_min) / float(self.grid_size_w)
        cell_size_y = (y_max - y_min) / float(self.grid_size_h)

        x_coords = (torch.arange(self.grid_size_w, dtype=torch.float32) + 0.5) * cell_size_x + x_min
        y_coords = (torch.arange(self.grid_size_h, dtype=torch.float32) + 0.5) * cell_size_y + y_min
        yy, xx = torch.meshgrid(y_coords, x_coords, indexing='ij')
        grid_xy = torch.stack([xx, yy], dim=-1).reshape(-1, 2)  # [H*W, 2]

        self.register_buffer('grid_xy', grid_xy)
        # FIX: 对应改进项 - Deterministic Spatial Jitter 固定纹理噪声
        x_coords_norm = (x_coords - x_min) / (x_max - x_min)
        y_coords_norm = (y_coords - y_min) / (y_max - y_min)
        yy_norm, xx_norm = torch.meshgrid(y_coords_norm, x_coords_norm, indexing='ij')
        fixed_noise = (torch.sin(100.0 * xx_norm) + torch.cos(100.0 * yy_norm)) * 1e-3
        self.register_buffer('fixed_noise', fixed_noise)
        self.cell_size_x = cell_size_x
        self.cell_size_y = cell_size_y

    def _build_init_anchors(self):
        """
        初始化使用的均匀安全锚点 (用于 init_query_bbox)，不参与前向逻辑。
        输出为 Logits 形式，与 forward 保持一致。
        """
        device = self.grid_xy.device
        total_cells = self.grid_size_h * self.grid_size_w
        # 均匀步进采样索引，确保初始化覆盖全域
        base_idx = (torch.arange(self.num_query, dtype=torch.long, device=device) * total_cells) // self.num_query
        xy = self.grid_xy[base_idx]
        theta_d = self._xy_to_theta_d(xy)  # [0, 1] 范围
        z = torch.full((self.num_query, 1), self.z_default, device=device, dtype=theta_d.dtype)

        # [FIX] 坐标域修正: 转换为 Logits (与 forward 保持一致)
        theta_d = self.inverse_sigmoid(theta_d)
        z = self.inverse_sigmoid(z)

        return torch.cat([theta_d, z], dim=-1)

    # ============================================================
    # 逆 Sigmoid (Logit) 变换
    # ============================================================
    @staticmethod
    def inverse_sigmoid(x, eps=1e-5):
        """
        将归一化坐标 [0, 1] 转换为 logits，使得 sigmoid(logits) ≈ x。
        公式: log(x / (1 - x))
        使用 clamp 防止 log(0) 或 log(inf) 产生 NaN。
        """
        x = x.clamp(min=eps, max=1.0 - eps)
        return torch.log(x / (1.0 - x))

    # ============================================================
    # 坐标转换 (xy -> theta/d)
    # ============================================================
    def _xy_to_theta_d(self, xy):
        x = xy[..., 0]
        y = xy[..., 1]
        # FIX: 对应改进项 - 使用 atan2 并归一化到 [0, 1]
        theta = torch.atan2(y, x)
        theta = (theta + math.pi) / (2 * math.pi)
        dist = torch.sqrt(x ** 2 + y ** 2).clamp(min=1e-6)
        d = torch.clamp(dist / self.polar_radius, 0.0, 1.0)
        theta = torch.clamp(theta, 0.0, 1.0)
        return torch.stack([theta, d], dim=-1)

    # ============================================================
    # α-MLP (Adaptive Confidence Gating)
    # ============================================================
    def _compute_alpha(self, radar_points, radar_mask):
        B, M, _ = radar_points.shape
        feat = torch.stack(
            [
                radar_points[..., self.radar_channel_map['x']],
                radar_points[..., self.radar_channel_map['y']],
                radar_points[..., self.radar_channel_map['z']],
                radar_points[..., self.radar_channel_map['rcs']],
                radar_points[..., self.radar_channel_map['v_r']],
            ],
            dim=-1,
        )
        x = feat.reshape(B * M, 5)

        x = self.alpha_fc1(x)
        x = self.alpha_ln1(x)
        x = F.relu(x, inplace=True)

        x = self.alpha_fc2(x)
        x = self.alpha_ln2(x)
        x = F.relu(x, inplace=True)

        x = self.alpha_fc3(x)
        x = torch.sigmoid(x)

        alpha = x.view(B, M, 1)
        return alpha * radar_mask.unsqueeze(-1)

    # ============================================================
    # 物理增强增益 (含 d^4 距离补偿)
    # ============================================================
    def _compute_phys_gain(self, radar_points):
        x = radar_points[..., self.radar_channel_map['x']]
        y = radar_points[..., self.radar_channel_map['y']]
        rcs = radar_points[..., self.radar_channel_map['rcs']]
        v_r = radar_points[..., self.radar_channel_map['v_r']]

        # RCS 对数压缩: 抑制大目标垄断 (Swerling 抑制)
        rcs_term = torch.log1p(F.relu(rcs))

        # Doppler 增益: 动态目标优先
        doppler_term = 1.0 + self.beta * torch.sigmoid(v_r)

        # 距离补偿: 抵消雷达方程 1/R^4 衰减 (关键物理项)
        dist = torch.sqrt(x ** 2 + y ** 2).clamp(min=1e-6)
        dist_term = (dist / self.d_ref).pow(4).clamp(max=20.0)

        return rcs_term * doppler_term * dist_term

    # ============================================================
    # 固定长度补齐 (保证静态图)
    # ============================================================
    def _fix_length(self, radar_points, radar_mask):
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

    # ============================================================
    # 前向传播
    # ============================================================
    def forward(self, radar_points=None, radar_mask=None):
        if not self.enabled:
            B = 1 if radar_points is None else radar_points.shape[0]
            anchors = self.init_anchors.unsqueeze(0).expand(B, -1, -1).clone()
            return anchors, None

        if radar_points is None:
            B = 1
            anchors = self.init_anchors.unsqueeze(0).expand(B, -1, -1).clone()
            return anchors, None

        radar_points, radar_mask = self._fix_length(radar_points, radar_mask)

        if radar_mask is None:
            # padding 点全 0 时可由此得到静态 mask
            radar_mask = (radar_points.abs().sum(dim=-1) > 0).to(radar_points.dtype)
        else:
            radar_mask = radar_mask.to(radar_points.dtype)

        B, M, _ = radar_points.shape
        device = radar_points.device

        # α-MLP 门控 (抑制远场噪声被 d^4 放大污染 Top-K)
        alpha = self._compute_alpha(radar_points, radar_mask)

        # 物理增益 (含 d^4)
        w_phys = self._compute_phys_gain(radar_points)
        alpha_weight = alpha.squeeze(-1) * radar_mask
        i_radar = alpha_weight * w_phys

        # ScatterAdd 注入 BEV 网格 (无循环)
        x = radar_points[..., self.radar_channel_map['x']]
        y = radar_points[..., self.radar_channel_map['y']]
        x_min, y_min, _, x_max, y_max, _ = self.pc_range
        ix = torch.floor((x - x_min) / (x_max - x_min) * self.grid_size_w).long()
        iy = torch.floor((y - y_min) / (y_max - y_min) * self.grid_size_h).long()
        ix = ix.clamp(0, self.grid_size_w - 1)
        iy = iy.clamp(0, self.grid_size_h - 1)

        flat_idx = iy * self.grid_size_w + ix
        batch_offset = (torch.arange(B, device=device).view(B, 1) * (self.grid_size_h * self.grid_size_w))
        flat_idx = (flat_idx + batch_offset).view(-1)
        flat_weight = i_radar.view(-1)
        flat_alpha = alpha_weight.view(-1)

        radar_field_flat = torch.zeros(
            B * self.grid_size_h * self.grid_size_w,
            device=device,
            dtype=radar_points.dtype,
        )
        radar_field_flat.scatter_add_(0, flat_idx, flat_weight)
        radar_field = radar_field_flat.view(B, 1, self.grid_size_h, self.grid_size_w)
        alpha_field_flat = torch.zeros(
            B * self.grid_size_h * self.grid_size_w,
            device=device,
            dtype=radar_points.dtype,
        )
        alpha_field_flat.scatter_add_(0, flat_idx, flat_alpha)
        alpha_field = alpha_field_flat.view(B, 1, self.grid_size_h, self.grid_size_w)

        # 空间扩散 (MaxPool2d) - 模拟雷达位置不确定性
        radar_field = self.diffusion(radar_field)
        alpha_field = self.diffusion(alpha_field)

        # 各向同性安全场 (Uniform Safety Net)
        base_field = radar_field.new_full(
            (B, 1, self.grid_size_h, self.grid_size_w),
            self.base_bias,
        )
        fixed_noise = self.fixed_noise.to(dtype=base_field.dtype, device=base_field.device)
        fixed_noise = fixed_noise.unsqueeze(0).unsqueeze(0)
        # FIX: 对应改进项 - 训练态加入随机噪声，评估态严格确定性
        if self.training:
            base_field = base_field + fixed_noise + (torch.rand_like(base_field) * 1e-3)
        else:
            base_field = base_field + fixed_noise

        # 加性融合
        fused = base_field + radar_field

        # Global Top-K (静态输出 K)
        fused_flat = fused.view(B, -1)
        _, topk_idx = torch.topk(fused_flat, self.num_query, dim=1, largest=True, sorted=True)

        # 从预计算网格解码坐标
        topk_xy = self.grid_xy[topk_idx]
        theta_d = self._xy_to_theta_d(topk_xy)  # 当前在 [0, 1] 范围

        z = torch.full((B, self.num_query, 1), self.z_default, device=device, dtype=theta_d.dtype)

        # [FIX] 坐标域修正: 转换为 Logits 因为 Transformer 内部会应用 sigmoid
        # sigmoid(inverse_sigmoid(x)) ≈ x，确保参考点覆盖完整空间
        theta_d = self.inverse_sigmoid(theta_d)
        z = self.inverse_sigmoid(z)

        # FIX: 对应改进项 - Score Injection 保留梯度
        alpha_map = alpha_field.squeeze(1)
        alpha_flat = alpha_map.view(B, -1)
        alpha_topk = torch.gather(alpha_flat, 1, topk_idx)
        alpha_topk = torch.clamp(alpha_topk, 0.0, 1.0)
        alpha_topk = alpha_topk.unsqueeze(-1)

        zero_col = torch.zeros((B, self.num_query, 1), device=device, dtype=theta_d.dtype)
        anchors = torch.cat(
            [
                theta_d,
                z,
                zero_col,
                zero_col,
                zero_col + 0.2,
                zero_col,
                zero_col + 1.0,
                alpha_topk,
                zero_col,
            ],
            dim=-1,
        )

        return anchors, None

    # ============================================================
    # 兼容性属性 (供 racformer_head.py 使用)
    # ============================================================
    @property
    def safety_anchors(self):
        # 输出 10 维锚点用于 init_query_bbox (wlh/sincos/vxvy 使用默认值)
        anchors_3d = self.init_anchors
        num = anchors_3d.shape[0]
        device = anchors_3d.device
        dtype = anchors_3d.dtype
        anchors_10d = torch.zeros((num, 10), device=device, dtype=dtype)
        anchors_10d[:, 0:3] = anchors_3d
        anchors_10d[:, 5] = 0.2
        anchors_10d[:, 7] = 1.0
        return anchors_10d

    @property
    def num_safety_anchors(self):
        return self.num_query


class RWHIModule(BaseModule):
    """
    对外统一入口: 默认启用 v3, 可通过 rwhi_version='legacy' 回滚
    """

    def __init__(self, rwhi_version='v3', init_cfg=None, **kwargs):
        super().__init__(init_cfg=init_cfg)
        self.rwhi_version = rwhi_version

        if rwhi_version in ('legacy', 'v2', 'v2.1'):
            if LegacyRWHIModule is None:
                raise RuntimeError('Legacy RWHI module not found.')
            self.impl = LegacyRWHIModule(init_cfg=init_cfg, **kwargs)
            self.is_legacy = True
        else:
            self.impl = RWHI_Ultimate(init_cfg=init_cfg, **kwargs)
            self.is_legacy = False

    def forward(self, radar_points=None, radar_mask=None):
        if self.is_legacy:
            return self.impl(radar_points)
        return self.impl(radar_points=radar_points, radar_mask=radar_mask)

    @property
    def safety_anchors(self):
        return self.impl.safety_anchors

    @property
    def num_safety_anchors(self):
        return self.impl.num_safety_anchors


class RWHIQueryGenerator(BaseModule):
    """
    [已废弃] RWHI Query生成器
    """

    def __init__(self, *args, **kwargs):
        raise DeprecationWarning(
            'RWHIQueryGenerator 已废弃。'
            '请直接使用 RWHIModule。'
        )
