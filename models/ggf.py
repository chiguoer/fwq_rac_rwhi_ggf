"""
GGF2.0 (Geometry-Guided Fusion) 模块

核心思想：
1. 构建统一的几何场（基于雷达高斯分布），供 RWHI 和 Decoder 共享
2. MGC (Visual Correcting Radar Geometry): 在图像采样时约束采样区域
3. GGA (Geometry-Guided Attention): 在注意力计算中加入几何偏置

所有子模块都可通过配置开关单独启用/禁用，方便消融实验。
"""

import math
import time
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.runner import BaseModule, get_dist_info

from .bbox.utils import R_MAX, theta_d2xy_coods, xy2theta_d_coods

# ============================================================
# 常量定义
# ============================================================
EPS = 1e-6  # 数值稳定性常量
LOG_EPS = 1e-10  # 对数计算的下界
GEOMETRY_BIAS_MIN = -100.0  # GGA 几何偏置的下限，防止 softmax 下溢


# ============================================================
# NativeRGF: 原生高斯场实现（仅用 PyTorch 原生算子）
# ============================================================
class NativeRGF(nn.Module):
    """
    Native Radar Gaussian Field
    
    使用 PyTorch 原生算子实现高斯场，不依赖自定义 CUDA。
    
    高斯参数：
    - 中心：雷达点的 (x, y) 坐标
    - 协方差：可预测或使用简单对角形式
    
    输出：
    - BEV 网格上的高斯场密度
    """
    
    def __init__(
        self,
        bev_grid_size=100,
        pc_range=None,
        default_sigma_x=3.0,  # 默认 x 方向标准差 (m)
        default_sigma_y=3.0,  # 默认 y 方向标准差 (m)
        use_velocity_anisotropy=True,  # 是否根据速度调整各向异性
        velocity_scale=0.1,  # 速度影响协方差的比例
        sigma_min=0.5,  # 最小标准差 (m)
        sigma_max=10.0,  # 最大标准差 (m)
        amplitude_mode='rcs',  # 振幅模式: 'rcs', 'uniform', 'learned'
        rcs_scale=0.1,  # RCS 到振幅的缩放系数
        kernel_size=7,  # 局部散射核尺寸 (奇数)
        predict_params=True,  # 是否预测高斯参数 (sx, sy, theta)
        input_indices=None,  # 雷达点特征索引
        hidden_dims=64,  # MLP 隐藏层维度
        use_rotation=True,  # 是否使用旋转高斯
        theta_scale=math.pi,  # 角度输出缩放
        profile_rgf=False,  # 是否输出简单耗时统计
        profile_rgf_every=100,  # profiler 输出间隔
        chunk_size=128,  # 分块大小：控制显存占用
        init_cfg=None,
    ):
        super().__init__()
        
        self.bev_grid_size = bev_grid_size
        self.pc_range = pc_range if pc_range is not None else [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]
        self.map_size = self.pc_range[3] - self.pc_range[0]
        self.grid_resolution = self.map_size / bev_grid_size
        
        self.default_sigma_x = default_sigma_x
        self.default_sigma_y = default_sigma_y
        self.use_velocity_anisotropy = use_velocity_anisotropy
        self.velocity_scale = velocity_scale
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.amplitude_mode = amplitude_mode
        self.rcs_scale = rcs_scale
        self.kernel_size = kernel_size
        self.predict_params = predict_params
        self.input_indices = input_indices
        self.hidden_dims = hidden_dims
        self.use_rotation = use_rotation
        self.theta_scale = theta_scale
        self.profile_rgf = profile_rgf
        self.profile_rgf_every = profile_rgf_every
        self._rgf_step = 0
        self.chunk_size = chunk_size

        # 预计算 BEV 网格坐标
        self._init_grid()

        # 预计算局部散射网格
        self._init_local_grid()

        # 参数预测器
        if self.predict_params:
            in_dim = len(input_indices) if input_indices is not None else 5
            hidden = hidden_dims
            self.param_predictor = nn.Sequential(
                nn.Linear(in_dim, hidden),
                nn.ReLU(inplace=True),
                nn.Linear(hidden, hidden),
                nn.ReLU(inplace=True),
                nn.Linear(hidden, 3),
            )
        else:
            self.param_predictor = None
    
    def _init_grid(self):
        """初始化 BEV 网格坐标"""
        H = W = self.bev_grid_size
        x_min, y_min = self.pc_range[0], self.pc_range[1]
        
        # 生成网格中心坐标
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
        
        self.register_buffer('grid_xy', grid_xy)

    def _init_local_grid(self):
        """初始化局部散射网格 (KxK)"""
        k = max(int(self.kernel_size), 1)
        if k % 2 == 0:
            k = k + 1
        self.kernel_size = k
        radius = k // 2
        offset_range = torch.arange(-radius, radius + 1)
        oy, ox = torch.meshgrid(offset_range, offset_range, indexing='ij')
        offsets = torch.stack([ox, oy], dim=-1).view(-1, 2)  # [K*K, 2]
        self.register_buffer('local_offsets', offsets)
    
    def _compute_gaussian_params(self, radar_points, radar_mask):
        """
        从雷达点计算高斯参数
        
        Args:
            radar_points: [B, M, C] 雷达点 (x, y, z, rcs, v_r, ...)
            radar_mask: [B, M] 有效点掩码
        
        Returns:
            centers: [B, M, 2] 高斯中心 (x, y)
            sigmas: [B, M, 2] 标准差 (sigma_x, sigma_y)
            amplitudes: [B, M] 振幅
            mask_bool: [B, M] 有效点布尔掩码或 None
            theta: [B, M] 旋转角
            precision: [B, M, 2, 2] 精度矩阵 (Sigma^-1)
        """
        B, M, C = radar_points.shape
        device = radar_points.device
        dtype = radar_points.dtype
        
        # 高斯中心
        centers = radar_points[..., :2]  # [B, M, 2]
        
        # 计算标准差与旋转角
        sigma_x = torch.full((B, M), self.default_sigma_x, device=device, dtype=dtype)
        sigma_y = torch.full((B, M), self.default_sigma_y, device=device, dtype=dtype)
        theta = torch.zeros((B, M), device=device, dtype=dtype)

        if self.predict_params and self.param_predictor is not None:
            if self.input_indices is not None:
                feat = radar_points[..., self.input_indices]
            else:
                feat = radar_points
            in_dim = self.param_predictor[0].in_features
            if feat.shape[-1] < in_dim:
                pad = in_dim - feat.shape[-1]
                feat = torch.cat([feat, torch.zeros(B, M, pad, device=device, dtype=dtype)], dim=-1)
            else:
                feat = feat[..., :in_dim]
            pred = self.param_predictor(feat)
            sigma_pred = F.softplus(pred[..., :2]) + 1e-2
            theta = torch.tanh(pred[..., 2]) * self.theta_scale
            sigma_x = sigma_pred[..., 0]
            sigma_y = sigma_pred[..., 1]
        else:
            if self.use_velocity_anisotropy and C > 4:
                # 根据速度方向调整各向异性
                v_r = radar_points[..., 4]  # 径向速度

                # 速度越大，沿运动方向的不确定性越大
                v_factor = 1.0 + self.velocity_scale * v_r.abs()

                # 径向速度影响径向方向的不确定性
                sigma_radial = self.default_sigma_x * v_factor
                # 使用已有 tensor 作为切向 sigma，避免 float -> tensor 的类型混乱
                sigma_tangent = sigma_y

                # 简化：直接使用径向/切向作为 x/y 方向
                sigma_x = sigma_radial
                sigma_y = sigma_tangent
        
        # Clamp 标准差
        sigma_x = sigma_x.clamp(min=self.sigma_min, max=self.sigma_max)
        sigma_y = sigma_y.clamp(min=self.sigma_min, max=self.sigma_max)
        sigmas = torch.stack([sigma_x, sigma_y], dim=-1)  # [B, M, 2]

        # 计算精度矩阵
        precision = self._build_precision(sigmas, theta)
        
        # 计算振幅
        if self.amplitude_mode == 'rcs' and C > 3:
            rcs = radar_points[..., 3]
            amplitudes = F.relu(rcs) * self.rcs_scale + 1.0
        elif self.amplitude_mode == 'uniform':
            amplitudes = torch.ones(B, M, device=device, dtype=dtype)
        else:
            amplitudes = torch.ones(B, M, device=device, dtype=dtype)
        
        # 应用掩码并清理无效点，避免 NaN 传播到高斯场或 GGA/MGC
        mask_bool = None
        if radar_mask is not None:
            mask_bool = radar_mask > 0
            # 无效点置零，避免 0 * NaN 产生 NaN
            centers = torch.where(
                mask_bool.unsqueeze(-1), centers, torch.zeros_like(centers)
            )
            # 使用默认 sigma 作为无效点的安全值
            fallback = torch.tensor(
                [self.default_sigma_x, self.default_sigma_y], device=device, dtype=dtype
            ).view(1, 1, 2).expand_as(sigmas)
            sigmas = torch.where(mask_bool.unsqueeze(-1), sigmas, fallback)
            theta = torch.where(mask_bool, theta, torch.zeros_like(theta))
            fallback_precision = self._build_precision(fallback, torch.zeros_like(theta))
            precision = torch.where(
                mask_bool.unsqueeze(-1).unsqueeze(-1),
                precision,
                fallback_precision
            )
            amplitudes = torch.where(mask_bool, amplitudes, torch.zeros_like(amplitudes))

        return centers, sigmas, amplitudes, mask_bool, theta, precision

    def _build_precision(self, sigmas, theta):
        """
        构建 2x2 精度矩阵 (Sigma^-1)
        """
        inv_s2 = 1.0 / (sigmas ** 2 + EPS)  # [B, M, 2]
        if not self.use_rotation:
            precision = torch.zeros(sigmas.shape[0], sigmas.shape[1], 2, 2, device=sigmas.device, dtype=sigmas.dtype)
            precision[..., 0, 0] = inv_s2[..., 0]
            precision[..., 1, 1] = inv_s2[..., 1]
            return precision

        cos_t = torch.cos(theta)
        sin_t = torch.sin(theta)
        # 旋转矩阵 R
        r11 = cos_t
        r12 = -sin_t
        r21 = sin_t
        r22 = cos_t
        # 计算 R * diag(inv_s2) * R^T
        precision = torch.zeros(sigmas.shape[0], sigmas.shape[1], 2, 2, device=sigmas.device, dtype=sigmas.dtype)
        precision[..., 0, 0] = r11 * r11 * inv_s2[..., 0] + r12 * r12 * inv_s2[..., 1]
        precision[..., 1, 1] = r21 * r21 * inv_s2[..., 0] + r22 * r22 * inv_s2[..., 1]
        precision[..., 0, 1] = r11 * r21 * inv_s2[..., 0] + r12 * r22 * inv_s2[..., 1]
        precision[..., 1, 0] = precision[..., 0, 1]
        return precision
    
    def forward(self, radar_points, radar_mask, return_params=False):
        """
        构建高斯场
        
        Args:
            radar_points: [B, M, C] 雷达点
            radar_mask: [B, M] 有效点掩码
            return_params: 是否返回高斯参数
        
        Returns:
            gaussian_field: [B, 1, H, W] BEV 网格上的高斯场
            (可选) params: dict 包含 centers, sigmas, amplitudes
        """
        B, M, C = radar_points.shape
        H = W = self.bev_grid_size
        device = radar_points.device
        dtype = radar_points.dtype
        
        t_start = time.perf_counter() if self.profile_rgf else None

        # 计算高斯参数
        centers, sigmas, amplitudes, mask_bool, theta, precision = self._compute_gaussian_params(radar_points, radar_mask)

        # 初始化高斯场（局部散射）
        field = torch.zeros(B, H * W, device=device, dtype=dtype)

        if radar_mask is not None:
            amplitudes = amplitudes * radar_mask.to(dtype)

        # 预计算局部偏移
        offsets = self.local_offsets.to(device=device)  # [K*K, 2]
        offsets_x = offsets[:, 0].view(1, 1, -1)
        offsets_y = offsets[:, 1].view(1, 1, -1)

        x_min, y_min = self.pc_range[0], self.pc_range[1]
        res = self.grid_resolution

        chunk_size = self.chunk_size if self.chunk_size and self.chunk_size > 0 else M
        for start in range(0, M, chunk_size):
            end = min(M, start + chunk_size)
            if radar_mask is not None and not radar_mask[:, start:end].any():
                continue

            centers_chunk = centers[:, start:end]  # [B, mc, 2]
            amps_chunk = amplitudes[:, start:end]  # [B, mc]
            precision_chunk = precision[:, start:end]  # [B, mc, 2, 2]

            # 计算网格索引
            grid_x = ((centers_chunk[..., 0] - x_min) / res).long()
            grid_y = ((centers_chunk[..., 1] - y_min) / res).long()

            # 局部邻域索引
            nx = grid_x.unsqueeze(-1) + offsets_x  # [B, mc, K2]
            ny = grid_y.unsqueeze(-1) + offsets_y  # [B, mc, K2]

            valid = (nx >= 0) & (nx < W) & (ny >= 0) & (ny < H)

            # 物理坐标
            x_phys = x_min + (nx.to(dtype) + 0.5) * res
            y_phys = y_min + (ny.to(dtype) + 0.5) * res

            dx = x_phys - centers_chunk[..., 0].unsqueeze(-1)
            dy = y_phys - centers_chunk[..., 1].unsqueeze(-1)
            diff = torch.stack([dx, dy], dim=-1)  # [B, mc, K2, 2]

            # 马氏距离
            diff_vec = diff.unsqueeze(-1)  # [B, mc, K2, 2, 1]
            prec = precision_chunk.unsqueeze(2)  # [B, mc, 1, 2, 2]
            term = torch.matmul(prec, diff_vec)
            mahal_sq = torch.matmul(diff_vec.transpose(-1, -2), term).squeeze(-1).squeeze(-1)  # [B, mc, K2]

            gaussian = amps_chunk.unsqueeze(-1) * torch.exp(-0.5 * mahal_sq)  # [B, mc, K2]
            gaussian = gaussian * valid.to(dtype)

            linear_idx = (ny * W + nx).long()  # [B, mc, K2]
            for b in range(B):
                valid_b = valid[b]
                if not valid_b.any():
                    continue
                idx = linear_idx[b][valid_b]
                val = gaussian[b][valid_b]
                field[b].scatter_add_(0, idx.view(-1), val.view(-1))

        gaussian_field = field.view(B, H, W).unsqueeze(1)

        self._rgf_step += 1
        if self.profile_rgf and (self._rgf_step % self.profile_rgf_every == 0) and t_start is not None:
            elapsed = (time.perf_counter() - t_start) * 1000.0
            print(f"[NativeRGF] forward time={elapsed:.2f} ms (B={B}, M={M}, K={self.kernel_size})")
        
        if return_params:
            params = {
                'centers': centers,
                'sigmas': sigmas,
                'theta': theta,
                'precision': precision,
                'amplitudes': amplitudes,
            }
            if mask_bool is not None:
                params['mask'] = mask_bool
            return gaussian_field, params
        return gaussian_field


# ============================================================
# GeometryFieldBuilder: 几何场构建与统一场积分
# ============================================================
class GeometryFieldBuilder(nn.Module):
    """
    几何场构建与统一场积分
    
    输入：雷达点及相关属性
    输出：
    - 线性空间几何场：用于 RWHI 选点（数值尺度 1-5）
    - 对数空间几何偏置：用于 GGA
    
    可以使用 NativeRGF 或简单的散射方式构建几何场。
    """
    
    def __init__(
        self,
        bev_grid_size=100,
        pc_range=None,
        use_native_rgf=True,
        # NativeRGF 参数
        rgf_sigma_x=3.0,
        rgf_sigma_y=3.0,
        rgf_use_velocity=True,
        rgf_velocity_scale=0.1,
        rgf_kernel_size=7,
        rgf_predict_params=True,
        rgf_input_indices=None,
        rgf_hidden_dims=64,
        rgf_use_rotation=True,
        rgf_theta_scale=math.pi,
        rgf_profile=False,
        rgf_profile_every=100,
        # 场缩放参数
        linear_scale=1.0,  # 线性场缩放
        linear_bias=1.0,   # 线性场基础偏置（对应背景分数）
        linear_max=5.0,    # 线性场上限
        log_temperature=1.0,  # 对数场温度参数
        # 融合参数
        rwhi_fusion_mode='replace',  # 'replace', 'add', 'gate'
        rwhi_fusion_weight=1.0,
        init_cfg=None,
    ):
        super().__init__()
        
        self.bev_grid_size = bev_grid_size
        self.pc_range = pc_range if pc_range is not None else [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]
        self.use_native_rgf = use_native_rgf
        
        self.linear_scale = linear_scale
        self.linear_bias = linear_bias
        self.linear_max = linear_max
        self.log_temperature = log_temperature
        
        self.rwhi_fusion_mode = rwhi_fusion_mode
        self.rwhi_fusion_weight = rwhi_fusion_weight
        
        # 构建 NativeRGF
        if use_native_rgf:
            self.native_rgf = NativeRGF(
                bev_grid_size=bev_grid_size,
                pc_range=pc_range,
                default_sigma_x=rgf_sigma_x,
                default_sigma_y=rgf_sigma_y,
                use_velocity_anisotropy=rgf_use_velocity,
                velocity_scale=rgf_velocity_scale,
                kernel_size=rgf_kernel_size,
                predict_params=rgf_predict_params,
                input_indices=rgf_input_indices,
                hidden_dims=rgf_hidden_dims,
                use_rotation=rgf_use_rotation,
                theta_scale=rgf_theta_scale,
                profile_rgf=rgf_profile,
                profile_rgf_every=rgf_profile_every,
            )
        else:
            self.native_rgf = None
    
    def forward(self, radar_points, radar_mask, rwhi_i_radar_map=None):
        """
        构建几何场
        
        Args:
            radar_points: [B, M, C] 雷达点
            radar_mask: [B, M] 有效点掩码
            rwhi_i_radar_map: [B, 1, H, W] RWHI 原始的 I_radar_map（可选，用于融合）
        
        Returns:
            linear_field: [B, 1, H, W] 线性空间几何场（用于 RWHI 选点）
            log_field: [B, 1, H, W] 对数空间几何偏置（用于 GGA）
            params: dict 高斯参数（如果使用 NativeRGF）
        """
        B = radar_points.shape[0]
        H = W = self.bev_grid_size
        device = radar_points.device
        dtype = radar_points.dtype
        
        params = {}
        
        if self.use_native_rgf and self.native_rgf is not None:
            # 使用 NativeRGF 构建高斯场
            gaussian_field, rgf_params = self.native_rgf(
                radar_points, radar_mask, return_params=True
            )
            params.update(rgf_params)
        else:
            # 简单的散射方式
            gaussian_field = torch.zeros(B, 1, H, W, device=device, dtype=dtype)
        
        # 构建线性空间几何场
        # 注意：当提供 rwhi_i_radar_map 时，输出应处于 I_radar 空间（不含 base_bias）
        linear_field_no_bias = gaussian_field * self.linear_scale
        linear_field = self.linear_bias + linear_field_no_bias
        
        # 与 RWHI 原始 I_radar_map 融合
        if rwhi_i_radar_map is not None:
            if self.rwhi_fusion_mode == 'replace':
                # 完全替换（保持 I_radar 空间）
                linear_field = linear_field_no_bias
            elif self.rwhi_fusion_mode == 'add':
                # 加性融合（I_radar 空间）
                linear_field = rwhi_i_radar_map + linear_field_no_bias * self.rwhi_fusion_weight
            elif self.rwhi_fusion_mode == 'gate':
                # 门控融合（I_radar 空间）
                gate = torch.sigmoid(gaussian_field)
                linear_field = gate * linear_field_no_bias + (1 - gate) * rwhi_i_radar_map
        
        # 裁剪到合理范围
        linear_field = linear_field.clamp(max=self.linear_max)
        
        # 构建对数空间几何偏置（用于 GGA）
        # log(density) / temperature，用于注意力权重
        log_eps = LOG_EPS if gaussian_field.dtype in (torch.float32, torch.float64) else 1e-6
        log_field = torch.log(gaussian_field.clamp(min=log_eps)) / self.log_temperature
        # Clamp 到合理范围
        log_field = log_field.clamp(min=GEOMETRY_BIAS_MIN)
        
        return linear_field, log_field, params


# ============================================================
# MGCModule: 视觉修正雷达几何
# ============================================================
class MGCModule(nn.Module):
    """
    MGC (Visual Correcting Radar Geometry)
    
    根据雷达几何约束图像采样位置。
    
    输入：
    - Query 空间位置（极坐标或笛卡尔坐标）
    - 雷达高斯参数
    - 相机外参/内参
    
    输出：
    - 采样 offset 调整或采样网格

    位置：
    - 位于解码器图像采样分支，用雷达高斯椭圆约束图像采样，再与雷达/LSS BEV 融合

    已知问题（修复点）：
    - sigma_uv 在训练早期可能非正定，导致 torch.linalg.cholesky 报错
      “input is not positive-definite”
    """
    
    def __init__(
        self,
        embed_dims=256,
        num_points=4,
        pc_range=None,
        # 约束模式
        constraint_mode='soft',  # 'soft' (偏置学习offset), 'hard' (直接覆盖)
        constraint_strength=1.0,
        # 椭圆投影参数
        ellipse_scale=2.0,  # 椭圆半径倍数（相对于 sigma）
        project_to_image=True,
        use_image_sampling=False,  # 是否启用图像分支采样
        sample_res=None,  # 图像采样分辨率 (h, w) 或 int
        view_select='first_valid',  # 视角选择策略
        min_depth=1e-5,  # 最小投影深度
        align_corners=True,
        spd_eig_min=1e-4,  # SPD 特征值下界
        spd_eig_max=None,  # SPD 特征值上界（None 表示不设）
        fallback_scale=1e-2,  # Cholesky 失败或无效时回退尺度
        max_dist=None,  # 超过该距离（m）不启用 MGC
        debug_mgc=False,  # 是否打印/统计调试信息
        debug_mgc_every=100,  # 调试输出间隔
        debug_mgc_max_print=5,  # 最大打印次数
        profile_mgc=False,  # 是否输出简单耗时统计
        profile_mgc_every=100,  # profiler 输出间隔
        # 学习参数
        learnable_strength=True,
        init_cfg=None,
    ):
        super().__init__()
        
        self.embed_dims = embed_dims
        self.num_points = num_points
        self.pc_range = pc_range if pc_range is not None else [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]
        
        self.constraint_mode = constraint_mode
        self.constraint_strength = constraint_strength
        self.ellipse_scale = ellipse_scale
        self.project_to_image = project_to_image
        self.use_image_sampling = use_image_sampling
        if sample_res is None:
            self.sample_res = None
        elif isinstance(sample_res, int):
            self.sample_res = (sample_res, sample_res)
        else:
            self.sample_res = tuple(sample_res)
        self.view_select = view_select
        self.min_depth = min_depth
        self.align_corners = align_corners
        self.spd_eig_min = spd_eig_min
        self.spd_eig_max = spd_eig_max
        self.fallback_scale = fallback_scale
        self.max_dist = max_dist
        self.debug_mgc = debug_mgc
        self.debug_mgc_every = debug_mgc_every
        self.debug_mgc_max_print = debug_mgc_max_print
        self.profile_mgc = profile_mgc
        self.profile_mgc_every = profile_mgc_every
        self._mgc_debug_step = 0
        self._mgc_debug_prints = 0
        self.logger = logging.getLogger()
        try:
            rank, _ = get_dist_info()
        except Exception:
            rank = 0
        self._is_main_process = (rank == 0)
        
        if learnable_strength:
            self.strength_param = nn.Parameter(torch.tensor(constraint_strength))
        else:
            self.register_buffer('strength_param', torch.tensor(constraint_strength))
        
        # 可选：从 query 特征预测约束强度
        self.strength_predictor = nn.Linear(embed_dims, 1)
        nn.init.zeros_(self.strength_predictor.weight)
        nn.init.constant_(self.strength_predictor.bias, constraint_strength)
    
    def compute_ellipse_bounds(self, centers, sigmas, lidar2img=None, image_h=None, image_w=None):
        """
        计算雷达椭圆在图像平面的投影边界
        
        Args:
            centers: [B, Q, 2] 雷达点中心 (x, y) in 物理坐标
            sigmas: [B, Q, 2] 标准差 (sigma_x, sigma_y)
            lidar2img: [B, N, 4, 4] 投影矩阵
            image_h, image_w: 图像尺寸
        
        Returns:
            bounds_2d: [B, Q, N, 4] 每个 Query 在每个视角的边界框 (u_min, v_min, u_max, v_max)
        """
        B, Q, _ = centers.shape
        device = centers.device
        dtype = centers.dtype
        
        if lidar2img is None:
            # 返回默认的全图边界
            return None
        
        N = lidar2img.shape[1] // 8 if lidar2img.dim() == 3 else 6  # 默认6个视角
        
        # 计算 3D 椭圆的 8 个角点
        # 简化：只用 x, y 平面的椭圆，z 使用默认高度范围
        z_low, z_high = self.pc_range[2], self.pc_range[5]
        
        ellipse_radius = sigmas * self.ellipse_scale  # [B, Q, 2]
        
        # 生成椭圆边界点（4 个主方向）
        offsets = torch.tensor([
            [1, 0], [-1, 0], [0, 1], [0, -1]
        ], device=device, dtype=dtype)  # [4, 2]
        
        # [B, Q, 4, 2]
        boundary_xy = centers.unsqueeze(2) + offsets.unsqueeze(0).unsqueeze(0) * ellipse_radius.unsqueeze(2)
        
        # 转换为 3D 点（添加 z 坐标）
        # [B, Q, 4, 2, 3] - 每个 xy 点有 z_low 和 z_high 两个版本
        boundary_3d = torch.zeros(B, Q, 4, 2, 3, device=device, dtype=dtype)
        boundary_3d[..., :2] = boundary_xy.unsqueeze(3).expand(-1, -1, -1, 2, -1)
        boundary_3d[..., 0, 2] = z_low
        boundary_3d[..., 1, 2] = z_high
        
        # 简化输出：返回 sigma 作为采样范围指示
        return {
            'centers': centers,
            'sigmas': sigmas,
            'ellipse_radius': ellipse_radius,
        }
    
    def compute_sampling_constraint(self, query_bbox, query_feat, gaussian_params, d_region):
        """
        计算采样约束
        
        Args:
            query_bbox: [B, Q, 10] Query 的 bbox（极坐标格式）
            query_feat: [B, Q, C] Query 特征
            gaussian_params: dict 包含 centers, sigmas 等
            d_region: float 当前层的采样区域半径
        
        Returns:
            constraint_mask: [B, Q, P] 约束掩码（软约束时为权重）
            adjusted_d_region: [B, Q] 调整后的采样区域半径
        """
        B, Q, _ = query_bbox.shape
        device = query_bbox.device
        dtype = query_bbox.dtype
        
        if gaussian_params is None or 'centers' not in gaussian_params:
            # 无约束
            return None, None
        
        # 预测每个 Query 的约束强度
        strength = torch.sigmoid(self.strength_predictor(query_feat)).squeeze(-1)  # [B, Q]
        strength = strength * self.strength_param
        
        # 获取 Query 的物理坐标
        query_bbox_xy = theta_d2xy_coods(query_bbox)
        query_centers = query_bbox_xy[..., :2]  # 归一化坐标
        
        # 转换为物理坐标
        query_centers_phys = query_centers.clone()
        map_size = self.pc_range[3] - self.pc_range[0]
        query_centers_phys[..., 0] = query_centers[..., 0] * map_size + self.pc_range[0]
        query_centers_phys[..., 1] = query_centers[..., 1] * map_size + self.pc_range[1]
        
        # 获取最近的雷达高斯
        radar_centers = gaussian_params['centers']  # [B, M, 2]
        radar_sigmas = gaussian_params['sigmas']    # [B, M, 2]
        radar_mask = gaussian_params.get('mask', None)  # [B, M] or None
        
        # 计算 Query 到每个雷达点的距离
        # [B, Q, 1, 2] - [B, 1, M, 2] -> [B, Q, M]
        dist_sq = ((query_centers_phys.unsqueeze(2) - radar_centers.unsqueeze(1)) ** 2).sum(dim=-1)
        if radar_mask is not None:
            # 对无效点设置为无穷，避免被选为最近点
            dist_sq = dist_sq.masked_fill(~radar_mask.unsqueeze(1), float('inf'))
        
        # 找最近的雷达点
        min_dist_sq, min_idx = dist_sq.min(dim=-1)  # [B, Q]
        min_dist_sq = torch.nan_to_num(min_dist_sq, posinf=1e6)
        
        # 获取对应的 sigma
        # [B, Q, 2]
        batch_idx = torch.arange(B, device=device).unsqueeze(1).expand(B, Q)
        nearest_sigma = radar_sigmas[batch_idx, min_idx]  # [B, Q, 2]
        
        # 根据 sigma 调整 d_region
        # d_region 是极坐标空间的半径，需要转换
        sigma_avg = nearest_sigma.mean(dim=-1)  # [B, Q]
        
        # 当 Query 靠近雷达点时，收紧采样区域
        adjusted_d_region = d_region * (1.0 - strength * 0.5 * torch.exp(-min_dist_sq / (2 * sigma_avg ** 2 + EPS)))
        adjusted_d_region = adjusted_d_region.clamp(min=d_region * 0.3, max=d_region)
        
        return strength, adjusted_d_region

    def _safe_spd_cholesky(self, sigma_uv, valid_mask=None):
        """
        将 sigma_uv 投影到 SPD 并做 Cholesky 分解。
        Args:
            sigma_uv: [..., 2, 2]
            valid_mask: [...], True 表示有效
        Returns:
            sigma_spd: [..., 2, 2]
            L: [..., 2, 2]
            stats: dict (可选)
            fallback_mask: [...] True 表示回退
        """
        dtype = sigma_uv.dtype
        device = sigma_uv.device

        # 若全无有效点，直接返回退化尺度
        if valid_mask is not None and not valid_mask.any():
            fallback_L = torch.zeros_like(sigma_uv)
            fallback_L[..., 0, 0] = self.fallback_scale
            fallback_L[..., 1, 1] = self.fallback_scale
            fallback_sigma = fallback_L @ fallback_L.transpose(-1, -2)
            stats = None
            if self.debug_mgc:
                stats = {
                    'eig_min': self.fallback_scale ** 2,
                    'eig_mean': self.fallback_scale ** 2,
                    'eig_max': self.fallback_scale ** 2,
                    'fallback_ratio': 1.0,
                }
            return fallback_sigma, fallback_L, stats, torch.ones_like(valid_mask, dtype=torch.bool)

        # 对称化
        sigma_sym = 0.5 * (sigma_uv + sigma_uv.transpose(-1, -2))

        eye = torch.eye(2, device=device, dtype=dtype).view(*([1] * (sigma_sym.dim() - 2)), 2, 2)
        if valid_mask is not None:
            sigma_sym = torch.where(
                valid_mask.unsqueeze(-1).unsqueeze(-1),
                sigma_sym,
                eye * (self.fallback_scale ** 2)
            )

        sigma_sym_f = sigma_sym.float() if sigma_sym.dtype in (torch.float16, torch.bfloat16) else sigma_sym
        eigvals, eigvecs = torch.linalg.eigh(sigma_sym_f)

        eigvals_clamped = eigvals.clamp(min=self.spd_eig_min)
        if self.spd_eig_max is not None:
            eigvals_clamped = eigvals_clamped.clamp(max=self.spd_eig_max)

        sigma_spd_f = eigvecs @ torch.diag_embed(eigvals_clamped) @ eigvecs.transpose(-1, -2)
        sigma_spd = sigma_spd_f.to(dtype=dtype)

        fallback_mask = torch.zeros_like(eigvals_clamped[..., 0], dtype=torch.bool)
        try:
            L_f = torch.linalg.cholesky(sigma_spd_f)
            info = None
        except RuntimeError as exc:
            if self.debug_mgc and self._is_main_process and self._mgc_debug_prints < self.debug_mgc_max_print:
                self.logger.warning('[MGC] cholesky failed, fallback to cholesky_ex: %s', exc)
                self._mgc_debug_prints += 1
            L_f, info = torch.linalg.cholesky_ex(sigma_spd_f, check_errors=False)

        if info is not None:
            fallback_mask = info > 0
            if fallback_mask.any() and self.debug_mgc and self._is_main_process and self._mgc_debug_prints < self.debug_mgc_max_print:
                self.logger.warning('[MGC] cholesky_ex fallback triggered for some elements.')
                self._mgc_debug_prints += 1

        if fallback_mask.any():
            fallback_L = torch.zeros_like(L_f)
            fallback_L[..., 0, 0] = self.fallback_scale
            fallback_L[..., 1, 1] = self.fallback_scale
            L_f = torch.where(fallback_mask.unsqueeze(-1).unsqueeze(-1), fallback_L, L_f)
            sigma_spd = torch.where(
                fallback_mask.unsqueeze(-1).unsqueeze(-1),
                eye * (self.fallback_scale ** 2),
                sigma_spd
            )

        L = L_f.to(dtype=dtype)

        stats = None
        if self.debug_mgc:
            stats = {
                'eig_min': eigvals_clamped.min().item(),
                'eig_mean': eigvals_clamped.mean().item(),
                'eig_max': eigvals_clamped.max().item(),
                'fallback_ratio': fallback_mask.float().mean().item(),
            }

        return sigma_spd, L, stats, fallback_mask

    def build_image_sampling_locations(self, query_bbox, query_feat, gaussian_params, img_metas, pc_range, sample_res=None):
        """
        基于雷达高斯椭圆构建图像采样位置 (affine_grid)

        Returns:
            sampling_locations: [B, Q, P, 3] in [0, 1] (x, y, view)
            mgc_info: dict
        """
        if gaussian_params is None or 'centers' not in gaussian_params:
            return None, {}

        t_start = time.perf_counter() if self.profile_mgc else None

        B, Q, _ = query_bbox.shape
        device = query_bbox.device
        dtype = query_bbox.dtype

        lidar2img = img_metas[0]['lidar2img']  # [B, N, 4, 4]
        image_h, image_w, _ = img_metas[0]['img_shape'][0]
        N = lidar2img.shape[1]

        # 采样分辨率
        if sample_res is None:
            if self.sample_res is None:
                sample_h, sample_w = 3, 4
            else:
                sample_h, sample_w = self.sample_res
        else:
            if isinstance(sample_res, int):
                sample_h, sample_w = sample_res, sample_res
            else:
                sample_h, sample_w = sample_res
        P = sample_h * sample_w

        # 预测每个 Query 的约束强度
        strength = torch.sigmoid(self.strength_predictor(query_feat)).squeeze(-1)  # [B, Q]
        strength = strength * self.strength_param

        # 获取 Query 的物理坐标
        query_bbox_xy = theta_d2xy_coods(query_bbox)
        query_centers = query_bbox_xy[..., :2]
        map_size = pc_range[3] - pc_range[0]
        query_centers_phys = query_centers.clone()
        query_centers_phys[..., 0] = query_centers[..., 0] * map_size + pc_range[0]
        query_centers_phys[..., 1] = query_centers[..., 1] * map_size + pc_range[1]
        query_z_phys = query_bbox[..., 2:3] * (pc_range[5] - pc_range[2]) + pc_range[2]

        # 最近雷达高斯
        radar_centers = gaussian_params['centers']  # [B, M, 2]
        radar_sigmas = gaussian_params['sigmas']    # [B, M, 2]
        radar_theta = gaussian_params.get('theta', None)
        radar_mask = gaussian_params.get('mask', None)

        dist_sq = ((query_centers_phys.unsqueeze(2) - radar_centers.unsqueeze(1)) ** 2).sum(dim=-1)
        if radar_mask is not None:
            dist_sq = dist_sq.masked_fill(~radar_mask.unsqueeze(1), float('inf'))

        min_dist_sq, min_idx = dist_sq.min(dim=-1)  # [B, Q]
        batch_idx = torch.arange(B, device=device).unsqueeze(1).expand(B, Q)
        nearest_center = radar_centers[batch_idx, min_idx]  # [B, Q, 2]
        nearest_sigma = radar_sigmas[batch_idx, min_idx].clamp(min=1e-3)  # [B, Q, 2]
        if radar_theta is None:
            nearest_theta = torch.zeros(B, Q, device=device, dtype=dtype)
        else:
            nearest_theta = radar_theta[batch_idx, min_idx]

        # 椭圆轴向 (物理坐标)
        cos_t = torch.cos(nearest_theta)
        sin_t = torch.sin(nearest_theta)
        ellipse_scale = self.ellipse_scale * (1.0 + 0.5 * strength)
        axis1 = torch.stack([cos_t * nearest_sigma[..., 0] * ellipse_scale,
                             sin_t * nearest_sigma[..., 0] * ellipse_scale], dim=-1)
        axis2 = torch.stack([-sin_t * nearest_sigma[..., 1] * ellipse_scale,
                             cos_t * nearest_sigma[..., 1] * ellipse_scale], dim=-1)

        # 3D 点：中心与轴端点
        center_xyz = torch.cat([nearest_center, query_z_phys], dim=-1)  # [B, Q, 3]
        axis1_xyz = torch.cat([axis1, torch.zeros(B, Q, 1, device=device, dtype=dtype)], dim=-1)
        axis2_xyz = torch.cat([axis2, torch.zeros(B, Q, 1, device=device, dtype=dtype)], dim=-1)
        pts = torch.stack([center_xyz, center_xyz + axis1_xyz, center_xyz + axis2_xyz], dim=2)  # [B, Q, 3, 3]
        pts_h = torch.cat([pts, torch.ones(B, Q, 3, 1, device=device, dtype=dtype)], dim=-1)  # [B, Q, 3, 4]

        # 投影到各视角
        pts_h = pts_h.unsqueeze(1).unsqueeze(-1)  # [B, 1, Q, 3, 4, 1]
        lidar2img = lidar2img.unsqueeze(2).unsqueeze(3)  # [B, N, 1, 1, 4, 4]
        proj = torch.matmul(lidar2img, pts_h).squeeze(-1)  # [B, N, Q, 3, 4]

        u = proj[..., 0]
        v = proj[..., 1]
        depth = proj[..., 2].clamp(min=self.min_depth)
        u = u / depth
        v = v / depth

        # 视角选择（基于中心点）
        u_center = u[..., 0]
        v_center = v[..., 0]
        depth_center = depth[..., 0]
        valid = (depth_center > self.min_depth) & (u_center > 0) & (u_center < image_w) & (v_center > 0) & (v_center < image_h)

        view_idx = valid.float().argmax(dim=1)  # [B, Q]
        valid_any = valid.any(dim=1)  # [B, Q]
        if self.max_dist is not None:
            valid_any = valid_any & (min_dist_sq <= (self.max_dist ** 2))

        # 选择对应视角的投影结果
        u_perm = u.permute(0, 2, 1, 3)  # [B, Q, N, 3]
        v_perm = v.permute(0, 2, 1, 3)
        gather_idx = view_idx.unsqueeze(-1).unsqueeze(-1).expand(B, Q, 1, 3)
        u_sel = u_perm.gather(2, gather_idx).squeeze(2)  # [B, Q, 3]
        v_sel = v_perm.gather(2, gather_idx).squeeze(2)  # [B, Q, 3]
        u_sel = torch.nan_to_num(u_sel, nan=0.0, posinf=0.0, neginf=0.0)
        v_sel = torch.nan_to_num(v_sel, nan=0.0, posinf=0.0, neginf=0.0)

        # 计算仿射矩阵 (归一化到 [-1, 1])
        u0 = u_sel[..., 0]
        v0 = v_sel[..., 0]
        u1 = u_sel[..., 1]
        v1 = v_sel[..., 1]
        u2 = u_sel[..., 2]
        v2 = v_sel[..., 2]

        center_u = u0 / image_w * 2 - 1
        center_v = v0 / image_h * 2 - 1
        axis1_u = (u1 - u0) / image_w * 2
        axis1_v = (v1 - v0) / image_h * 2
        axis2_u = (u2 - u0) / image_w * 2
        axis2_v = (v2 - v0) / image_h * 2

        # 无有效雷达支持的 query 直接置零，避免数值异常
        valid_any_f = valid_any.to(dtype=dtype)
        axis1_u = axis1_u * valid_any_f
        axis1_v = axis1_v * valid_any_f
        axis2_u = axis2_u * valid_any_f
        axis2_v = axis2_v * valid_any_f

        # 构建 Sigma_uv 并做 Cholesky 分解
        A = torch.zeros(B, Q, 2, 2, device=device, dtype=dtype)
        A[..., 0, 0] = axis1_u
        A[..., 0, 1] = axis2_u
        A[..., 1, 0] = axis1_v
        A[..., 1, 1] = axis2_v
        sigma_uv = torch.matmul(A, A.transpose(-1, -2))
        sigma_uv, L, stats, fallback_mask = self._safe_spd_cholesky(sigma_uv, valid_mask=valid_any)

        theta = torch.zeros(B, Q, 2, 3, device=device, dtype=dtype)
        theta[..., 0:2, 0:2] = L
        theta[..., 0, 2] = center_u
        theta[..., 1, 2] = center_v

        # 无有效视角时使用退化采样
        fallback = torch.zeros_like(theta)
        fallback[..., 0, 0] = 0.01
        fallback[..., 1, 1] = 0.01
        theta = torch.where(valid_any.unsqueeze(-1).unsqueeze(-1), theta, fallback)

        # affine_grid 生成采样点
        grid = F.affine_grid(theta.view(B * Q, 2, 3), torch.Size((B * Q, 1, sample_h, sample_w)),
                             align_corners=self.align_corners)  # [-1, 1]
        grid = grid.view(B, Q, sample_h * sample_w, 2)
        grid = (grid + 1.0) * 0.5  # [0, 1]

        # 视角索引归一化
        view_coord = view_idx.to(dtype=dtype) / max(float(N - 1), 1.0)
        view_coord = view_coord.view(B, Q, 1, 1).expand(B, Q, P, 1)
        sampling_locations = torch.cat([grid, view_coord], dim=-1)  # [B, Q, P, 3]

        self._mgc_debug_step += 1
        if (self.debug_mgc and self._is_main_process and
                (self._mgc_debug_step % self.debug_mgc_every == 0) and stats is not None):
            axis_min = math.sqrt(stats['eig_min']) if stats['eig_min'] > 0 else 0.0
            axis_max = math.sqrt(stats['eig_max']) if stats['eig_max'] > 0 else 0.0
            self.logger.info(
                "[MGC] eig(min/mean/max)=(%.4e, %.4e, %.4e) axis(min/max)=(%.4e, %.4e) "
                "fallback_ratio=%.3f valid_ratio=%.3f",
                stats['eig_min'], stats['eig_mean'], stats['eig_max'],
                axis_min, axis_max, stats['fallback_ratio'], valid_any.float().mean().item()
            )

        if (self.profile_mgc and self._is_main_process and
                (self._mgc_debug_step % self.profile_mgc_every == 0) and t_start is not None):
            elapsed = (time.perf_counter() - t_start) * 1000.0
            self.logger.info("[MGC] build_image_sampling_locations time=%.2f ms", elapsed)

        return sampling_locations, {
            'view_idx': view_idx,
            'valid_mask': valid_any,
            'fallback_mask': fallback_mask,
            'sample_res': (sample_h, sample_w),
        }
    
    def forward(self, query_bbox, query_feat, sampling_offset, gaussian_params, d_region):
        """
        应用 MGC 约束
        
        Args:
            query_bbox: [B, Q, 10] Query bbox
            query_feat: [B, Q, C] Query 特征
            sampling_offset: [B, Q, P, 3] 原始采样 offset
            gaussian_params: dict 高斯参数
            d_region: float 采样区域
        
        Returns:
            adjusted_offset: [B, Q, P, 3] 调整后的采样 offset
            mgc_info: dict MGC 相关信息
        """
        if gaussian_params is None:
            return sampling_offset, {}
        
        strength, adjusted_d_region = self.compute_sampling_constraint(
            query_bbox, query_feat, gaussian_params, d_region
        )
        
        if strength is None:
            return sampling_offset, {}
        
        # 软约束：根据强度缩放 offset
        if self.constraint_mode == 'soft':
            # strength: [B, Q], offset: [B, Q, P, 3]
            scale = 1.0 - strength.unsqueeze(-1).unsqueeze(-1) * 0.3
            adjusted_offset = sampling_offset * scale
        else:
            adjusted_offset = sampling_offset
        
        return adjusted_offset, {
            'strength': strength,
            'adjusted_d_region': adjusted_d_region,
        }


# ============================================================
# GGAModule: 几何引导注意力
# ============================================================
class GGAModule(nn.Module):
    """
    GGA (Geometry-Guided Attention)
    
    在注意力计算中加入几何偏置，使模型更关注几何上相关的区域。
    
    几何偏置计算：
    - 基于 Query 中心与雷达高斯中心之间的马氏距离
    - B_geom = -0.5 * d_mahal^2 / temperature
    """
    
    def __init__(
        self,
        embed_dims=256,
        num_heads=8,
        temperature=1.0,  # 温度参数，控制偏置强度
        learnable_temperature=True,
        bias_scale=1.0,  # 偏置缩放系数
        bias_min=GEOMETRY_BIAS_MIN,  # 偏置下限
        use_query_projection=False,  # 是否对 Query 做投影
        soft_clamp_min=None,  # 软截断下限（None 表示关闭）
        soft_clamp_beta=1.0,  # 软截断平滑系数
        debug_gga=False,  # 是否输出 GGA 调试信息
        debug_gga_every=100,  # 调试输出间隔
        init_cfg=None,
    ):
        super().__init__()
        
        self.embed_dims = embed_dims
        self.num_heads = num_heads
        self.bias_scale = bias_scale
        self.bias_min = bias_min
        self.use_query_projection = use_query_projection
        self.soft_clamp_min = soft_clamp_min
        self.soft_clamp_beta = soft_clamp_beta
        self.debug_gga = debug_gga
        self.debug_gga_every = debug_gga_every
        self._gga_debug_step = 0
        self.logger = logging.getLogger()
        try:
            rank, _ = get_dist_info()
        except Exception:
            rank = 0
        self._is_main_process = (rank == 0)
        
        if learnable_temperature:
            self.temperature = nn.Parameter(torch.tensor(temperature))
        else:
            self.register_buffer('temperature', torch.tensor(temperature))
        
        # 可学习的偏置缩放
        self.bias_scale_param = nn.Parameter(torch.tensor(bias_scale))
        
        # 可选：从 Query 特征预测偏置权重
        if use_query_projection:
            self.query_proj = nn.Linear(embed_dims, num_heads)
            nn.init.zeros_(self.query_proj.weight)
            nn.init.ones_(self.query_proj.bias)
    
    def compute_mahalanobis_distance(self, query_centers, gaussian_centers, gaussian_sigmas):
        """
        计算马氏距离
        
        Args:
            query_centers: [B, Q, 2] Query 中心 (物理坐标)
            gaussian_centers: [B, M, 2] 高斯中心
            gaussian_sigmas: [B, M, 2] 高斯标准差
        
        Returns:
            mahal_dist_sq: [B, Q, M] 马氏距离平方
        """
        # [B, Q, 1, 2] - [B, 1, M, 2] -> [B, Q, M, 2]
        diff = query_centers.unsqueeze(2) - gaussian_centers.unsqueeze(1)
        
        # 对角协方差的逆
        inv_var = 1.0 / (gaussian_sigmas.unsqueeze(1) ** 2 + EPS)  # [B, 1, M, 2]
        
        # 马氏距离平方
        mahal_dist_sq = (diff ** 2 * inv_var).sum(dim=-1)  # [B, Q, M]
        
        return mahal_dist_sq

    def _soft_clamp_min(self, x):
        if self.soft_clamp_min is None:
            return x
        beta = max(float(self.soft_clamp_beta), 1e-6)
        return self.soft_clamp_min + F.softplus((x - self.soft_clamp_min) * beta) / beta
    
    def compute_geometry_bias(self, query_bbox, query_feat, gaussian_params, pc_range):
        """
        计算几何偏置
        
        Args:
            query_bbox: [B, Q, 10] Query bbox（极坐标格式）
            query_feat: [B, Q, C] Query 特征
            gaussian_params: dict 高斯参数
            pc_range: list 点云范围
        
        Returns:
            geometry_bias: [B, num_heads, Q, M] 几何偏置（用于加到 attention logits 上）
        """
        B, Q, _ = query_bbox.shape
        device = query_bbox.device
        dtype = query_bbox.dtype
        
        if gaussian_params is None or 'centers' not in gaussian_params:
            return None
        
        gaussian_centers = gaussian_params['centers']  # [B, M, 2]
        gaussian_sigmas = gaussian_params['sigmas']    # [B, M, 2]
        gaussian_precision = gaussian_params.get('precision', None)  # [B, M, 2, 2]
        gaussian_mask = gaussian_params.get('mask', None)  # [B, M] or None
        M = gaussian_centers.shape[1]
        
        # 获取 Query 的物理坐标
        query_bbox_xy = theta_d2xy_coods(query_bbox)
        query_centers = query_bbox_xy[..., :2]  # 归一化坐标 [B, Q, 2]
        
        # 转换为物理坐标
        query_centers_phys = query_centers.clone()
        map_size = pc_range[3] - pc_range[0]
        query_centers_phys[..., 0] = query_centers[..., 0] * map_size + pc_range[0]
        query_centers_phys[..., 1] = query_centers[..., 1] * map_size + pc_range[1]
        
        # 计算马氏距离
        # 防御性处理，避免 NaN/Inf 传播
        gaussian_centers = torch.nan_to_num(gaussian_centers, nan=0.0)
        gaussian_sigmas = torch.nan_to_num(gaussian_sigmas, nan=1.0)

        if gaussian_precision is not None:
            diff = query_centers_phys.unsqueeze(2) - gaussian_centers.unsqueeze(1)  # [B, Q, M, 2]
            diff_vec = diff.unsqueeze(-1)  # [B, Q, M, 2, 1]
            prec = gaussian_precision.unsqueeze(1)  # [B, 1, M, 2, 2]
            term = torch.matmul(prec, diff_vec)
            mahal_dist_sq = torch.matmul(diff_vec.transpose(-1, -2), term).squeeze(-1).squeeze(-1)  # [B, Q, M]
        else:
            mahal_dist_sq = self.compute_mahalanobis_distance(
                query_centers_phys, gaussian_centers, gaussian_sigmas
            )  # [B, Q, M]
        
        # 计算几何偏置
        # B_geom = -0.5 * d^2 / temperature * scale
        temperature = self.temperature.clamp(min=0.1)
        geometry_bias = -0.5 * mahal_dist_sq / temperature * self.bias_scale_param
        
        # 软截断或硬截断
        geometry_bias = self._soft_clamp_min(geometry_bias)
        geometry_bias = geometry_bias.clamp(min=self.bias_min)
        
        # 扩展到 num_heads 维度
        # [B, Q, M] -> [B, num_heads, Q, M]
        geometry_bias = geometry_bias.unsqueeze(1).expand(-1, self.num_heads, -1, -1)

        if gaussian_mask is not None:
            geometry_bias = geometry_bias.masked_fill(
                ~gaussian_mask.unsqueeze(1).unsqueeze(2), self.bias_min
            )
        
        # 可选：根据 Query 特征调整偏置权重
        if self.use_query_projection:
            query_weight = self.query_proj(query_feat)  # [B, Q, num_heads]
            query_weight = query_weight.permute(0, 2, 1).unsqueeze(-1)  # [B, num_heads, Q, 1]
            geometry_bias = geometry_bias * query_weight
        
        return geometry_bias
    
    def forward(self, attn_logits, query_bbox, query_feat, gaussian_params, pc_range):
        """
        应用 GGA 几何偏置
        
        Args:
            attn_logits: [B, num_heads, Q, K] 原始注意力 logits
            query_bbox: [B, Q, 10] Query bbox
            query_feat: [B, Q, C] Query 特征
            gaussian_params: dict 高斯参数
            pc_range: list 点云范围
        
        Returns:
            adjusted_logits: [B, num_heads, Q, K] 调整后的 logits
            gga_info: dict GGA 相关信息
        """
        geometry_bias = self.compute_geometry_bias(
            query_bbox, query_feat, gaussian_params, pc_range
        )
        
        if geometry_bias is None:
            return attn_logits, {}

        B, num_heads, Q, K = attn_logits.shape
        M = geometry_bias.shape[-1]
        
        # 如果 K != M，需要处理维度不匹配
        # 简化处理：如果 K > M，对 bias 做广播；如果 K < M，截断或求最近
        if K == M:
            adjusted_logits = attn_logits + geometry_bias
            bias_used = geometry_bias
        elif K > M:
            # 扩展 geometry_bias
            padding = torch.zeros(B, num_heads, Q, K - M, device=geometry_bias.device, dtype=geometry_bias.dtype)
            geometry_bias_padded = torch.cat([geometry_bias, padding], dim=-1)
            adjusted_logits = attn_logits + geometry_bias_padded
            bias_used = geometry_bias_padded
        else:
            # 截断 geometry_bias
            adjusted_logits = attn_logits + geometry_bias[..., :K]
            bias_used = geometry_bias[..., :K]

        self._gga_debug_step += 1
        if self.debug_gga and self._is_main_process and (self._gga_debug_step % self.debug_gga_every == 0):
            b_min = bias_used.min().item()
            b_mean = bias_used.mean().item()
            b_max = bias_used.max().item()
            self.logger.info("[GGA] bias(min/mean/max)=(%.2f, %.2f, %.2f)", b_min, b_mean, b_max)
        
        return adjusted_logits, {
            'geometry_bias': geometry_bias,
        }


# ============================================================
# GGFModule: GGF2.0 总管理类
# ============================================================
class GGFModule(BaseModule):
    """
    GGF2.0 总管理类
    
    协调所有 GGF 子模块：
    - GeometryFieldBuilder: 几何场构建
    - NativeRGF: 原生高斯场
    - MGCModule: 视觉修正雷达几何
    - GGAModule: 几何引导注意力
    
    所有子模块都可通过配置开关单独启用/禁用。
    """
    
    def __init__(
        self,
        enabled=True,
        embed_dims=256,
        pc_range=None,
        bev_grid_size=100,
        # 子模块开关
        use_mgc=True,
        use_gga=True,
        use_unified_field=True,
        use_native_rgf=True,
        # GeometryFieldBuilder 参数
        field_cfg=None,
        # MGC 参数
        mgc_cfg=None,
        # GGA 参数
        gga_cfg=None,
        init_cfg=None,
    ):
        super().__init__(init_cfg)
        
        self.enabled = enabled
        self.embed_dims = embed_dims
        self.pc_range = pc_range if pc_range is not None else [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]
        self.bev_grid_size = bev_grid_size
        
        # 子模块开关
        self.use_mgc = use_mgc
        self.use_gga = use_gga
        self.use_unified_field = use_unified_field
        self.use_native_rgf = use_native_rgf
        
        # 默认配置
        field_cfg = field_cfg if field_cfg is not None else {}
        mgc_cfg = mgc_cfg if mgc_cfg is not None else {}
        gga_cfg = gga_cfg if gga_cfg is not None else {}
        
        # 构建子模块
        if enabled and (use_unified_field or use_native_rgf):
            self.geometry_field_builder = GeometryFieldBuilder(
                bev_grid_size=bev_grid_size,
                pc_range=pc_range,
                use_native_rgf=use_native_rgf,
                **field_cfg,
            )
        else:
            self.geometry_field_builder = None
        
        if enabled and use_mgc:
            self.mgc = MGCModule(
                embed_dims=embed_dims,
                pc_range=pc_range,
                **mgc_cfg,
            )
        else:
            self.mgc = None
        
        if enabled and use_gga:
            self.gga = GGAModule(
                embed_dims=embed_dims,
                **gga_cfg,
            )
        else:
            self.gga = None
        
        # 缓存几何场和参数
        self._cached_linear_field = None
        self._cached_log_field = None
        self._cached_params = None
    
    def build_geometry_field(self, radar_points, radar_mask, rwhi_i_radar_map=None):
        """
        构建几何场（供 RWHI 使用）
        
        Args:
            radar_points: [B, M, C] 雷达点
            radar_mask: [B, M] 有效点掩码
            rwhi_i_radar_map: [B, 1, H, W] RWHI 原始 I_radar_map
        
        Returns:
            linear_field: [B, 1, H, W] 线性空间几何场
            log_field: [B, 1, H, W] 对数空间几何偏置
        """
        if not self.enabled or self.geometry_field_builder is None:
            return rwhi_i_radar_map, None
        
        linear_field, log_field, params = self.geometry_field_builder(
            radar_points, radar_mask, rwhi_i_radar_map
        )
        
        # 缓存
        self._cached_linear_field = linear_field
        self._cached_log_field = log_field
        self._cached_params = params
        
        return linear_field, log_field
    
    def apply_mgc(self, query_bbox, query_feat, sampling_offset, d_region):
        """
        应用 MGC 约束（供采样模块使用）
        
        Args:
            query_bbox: [B, Q, 10] Query bbox
            query_feat: [B, Q, C] Query 特征
            sampling_offset: [B, Q, P, 3] 采样 offset
            d_region: float 采样区域
        
        Returns:
            adjusted_offset: [B, Q, P, 3] 调整后的 offset
            mgc_info: dict MGC 信息
        """
        if not self.enabled or self.mgc is None or not self.use_mgc:
            return sampling_offset, {}
        
        return self.mgc(
            query_bbox, query_feat, sampling_offset,
            self._cached_params, d_region
        )
    
    def apply_gga(self, attn_logits, query_bbox, query_feat):
        """
        应用 GGA 几何偏置（供注意力模块使用）
        
        Args:
            attn_logits: [B, num_heads, Q, K] 注意力 logits
            query_bbox: [B, Q, 10] Query bbox
            query_feat: [B, Q, C] Query 特征
        
        Returns:
            adjusted_logits: [B, num_heads, Q, K] 调整后的 logits
            gga_info: dict GGA 信息
        """
        if not self.enabled or self.gga is None or not self.use_gga:
            return attn_logits, {}
        
        return self.gga(
            attn_logits, query_bbox, query_feat,
            self._cached_params, self.pc_range
        )
    
    def get_cached_params(self):
        """获取缓存的高斯参数"""
        return self._cached_params
    
    def get_cached_fields(self):
        """获取缓存的几何场"""
        return self._cached_linear_field, self._cached_log_field
    
    def clear_cache(self):
        """清除缓存"""
        self._cached_linear_field = None
        self._cached_log_field = None
        self._cached_params = None
    
    def forward(self, radar_points, radar_mask, rwhi_i_radar_map=None):
        """
        主前向传播：构建几何场
        
        Args:
            radar_points: [B, M, C] 雷达点
            radar_mask: [B, M] 有效点掩码
            rwhi_i_radar_map: [B, 1, H, W] RWHI 原始 I_radar_map
        
        Returns:
            linear_field: [B, 1, H, W] 线性空间几何场
            log_field: [B, 1, H, W] 对数空间几何偏置
        """
        return self.build_geometry_field(radar_points, radar_mask, rwhi_i_radar_map)


# ============================================================
# 工厂函数
# ============================================================
def build_ggf(ggf_cfg):
    """
    构建 GGF 模块
    
    Args:
        ggf_cfg: dict GGF 配置
    
    Returns:
        ggf_module: GGFModule 实例
    """
    if ggf_cfg is None:
        return None
    
    return GGFModule(**ggf_cfg)


# ============================================================
# 调试和可视化工具
# ============================================================
class GGFDebugger:
    """
    GGF 调试和可视化工具
    """
    
    @staticmethod
    def visualize_geometry_field(linear_field, log_field, radar_points=None, save_path=None):
        """
        可视化几何场
        
        Args:
            linear_field: [B, 1, H, W] 线性空间几何场
            log_field: [B, 1, H, W] 对数空间几何偏置
            radar_points: [B, M, C] 雷达点（可选）
            save_path: str 保存路径（可选）
        """
        import matplotlib.pyplot as plt
        
        B = linear_field.shape[0]
        
        fig, axes = plt.subplots(B, 2, figsize=(12, 6 * B))
        if B == 1:
            axes = axes.reshape(1, 2)
        
        for b in range(B):
            # 线性场
            ax1 = axes[b, 0]
            im1 = ax1.imshow(linear_field[b, 0].cpu().numpy(), cmap='viridis')
            ax1.set_title(f'Batch {b}: Linear Field')
            plt.colorbar(im1, ax=ax1)
            
            # 对数场
            ax2 = axes[b, 1]
            im2 = ax2.imshow(log_field[b, 0].cpu().numpy(), cmap='RdBu_r')
            ax2.set_title(f'Batch {b}: Log Field (GGA Bias)')
            plt.colorbar(im2, ax=ax2)
            
            # 绘制雷达点
            if radar_points is not None:
                H, W = linear_field.shape[-2:]
                pts = radar_points[b].cpu().numpy()
                # 转换为网格坐标
                x_idx = ((pts[:, 0] + 51.2) / 102.4 * W).astype(int)
                y_idx = ((pts[:, 1] + 51.2) / 102.4 * H).astype(int)
                valid = (x_idx >= 0) & (x_idx < W) & (y_idx >= 0) & (y_idx < H)
                ax1.scatter(x_idx[valid], y_idx[valid], c='red', s=10, alpha=0.5)
                ax2.scatter(x_idx[valid], y_idx[valid], c='red', s=10, alpha=0.5)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
            plt.close()
        else:
            plt.show()
    
    @staticmethod
    def print_statistics(ggf_module, linear_field, log_field):
        """
        打印统计信息
        """
        print("=" * 50)
        print("GGF Statistics")
        print("=" * 50)
        
        if linear_field is not None:
            print(f"Linear Field - min: {linear_field.min().item():.4f}, "
                  f"max: {linear_field.max().item():.4f}, "
                  f"mean: {linear_field.mean().item():.4f}")
        
        if log_field is not None:
            print(f"Log Field - min: {log_field.min().item():.4f}, "
                  f"max: {log_field.max().item():.4f}, "
                  f"mean: {log_field.mean().item():.4f}")
        
        params = ggf_module.get_cached_params()
        if params:
            if 'sigmas' in params:
                sigmas = params['sigmas']
                print(f"Gaussian Sigmas - mean: {sigmas.mean().item():.4f}")
            if 'amplitudes' in params:
                amps = params['amplitudes']
                print(f"Gaussian Amplitudes - mean: {amps.mean().item():.4f}")
        
        print("=" * 50)
