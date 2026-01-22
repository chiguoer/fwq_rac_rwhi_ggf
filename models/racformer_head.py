# -*- coding: utf-8 -*-
"""
RaCFormer检测头模块

修改清单:
1. 类名改为CapWords规范: RaCFormer_head -> RaCFormerHead
2. 修复可变默认参数: dict()/list改为None，函数内初始化
3. 修复方法签名: forward/loss_single/get_targets/_get_target_single 兼容父类
4. 修复类型错误: bool张量转float (第588行)
5. 修复变量未赋值: map_known_indice设置默认值 (第638行)
6. 修复异常捕获过宽: 仅捕获预期异常 (第293行)
7. 修复E741变量名: l -> length, w -> width (get_bboxes方法)
8. 修复cuda()调用: 改为to(device) (第576-582行)
9. 移动import copy到文件顶部
10. 添加@staticmethod装饰器到纯函数方法
11. 遵循PEP8规范: 缩进、空行、行长度等
12. 修复prepare_for_dn_loss: 添加@staticmethod装饰器 (第724行)

v2.2 修复:
- 彻底修复类型错误: _add_label_noise方法中添加显式类型转换
- 完善方法签名: 确保所有重写方法与父类DETRHead兼容
- 添加@staticmethod到prepare_for_dn_loss方法
"""

import copy  # 修复: 移动到文件顶部，避免函数内import
import math
import torch
import torch.nn as nn
from mmcv.runner import force_fp32
from mmdet.core import multi_apply, reduce_mean
from mmdet.models import HEADS
from mmdet.models.dense_heads import DETRHead
from mmdet3d.core.bbox.coders import build_bbox_coder
from mmdet3d.core.bbox.structures.lidar_box3d import LiDARInstance3DBoxes

from .bbox.utils import normalize_bbox, encode_bbox, xy2theta_d_coods, compute_map_size_and_radius
from .utils import VERSION
from .rwhi import RWHIModule


@HEADS.register_module('RaCFormer_head')  # 修复: 显式注册旧名称，兼容配置文件
class RaCFormerHead(DETRHead):
    """
    RaCFormer检测头

    支持两种Query初始化策略:
    1. 原始策略: 均匀极坐标分布 (use_rwhi=False)
    2. RWHI策略: RCS加权混合锚点初始化 (use_rwhi=True)

    [重要修复] 动态内容生成 (Dynamic Content Generation)
    当使用RWHI时，query位置是动态的（每帧基于雷达点云生成）。
    解决方案：使用pos2content MLP将动态位置转换为动态内容特征。
    """

    def __init__(
        self,
        num_classes,
        in_channels,
        query_denoising=True,
        query_denoising_groups=10,
        num_clusters=5,
        bbox_coder=None,
        code_size=10,
        code_weights=None,
        train_cfg=None,
        test_cfg=None,
        use_rwhi=False,
        rwhi_cfg=None,
        polar_radius=None,  # [FIX v7.1] 支持传入固定 polar_radius (推荐 65.0)
        **kwargs
    ):
        """
        初始化RaCFormerHead

        Args:
            num_classes: 类别数量
            in_channels: 输入通道数
            query_denoising: 是否启用Query去噪
            query_denoising_groups: 去噪组数
            num_clusters: 聚类数量
            bbox_coder: 边界框编码器配置
            code_size: 编码大小
            code_weights: 编码权重 (默认None, 内部初始化为[1.0]*10)
            train_cfg: 训练配置 (默认None, 内部初始化为空字典)
            test_cfg: 测试配置 (默认None, 内部初始化为{max_per_img:100})
            use_rwhi: 是否启用RWHI Query初始化
            rwhi_cfg: RWHI模块配置
            polar_radius: [FIX v7.1] 固定极坐标半径 (推荐 65.0)，若 None 则从 pc_range 计算
            **kwargs: 其他参数
        """
        # 修复: 可变默认参数改为None，函数内初始化，避免实例共享
        if code_weights is None:
            code_weights = [1.0] * 10
        if train_cfg is None:
            train_cfg = {}
        if test_cfg is None:
            test_cfg = {'max_per_img': 100}

        self.code_size = code_size
        self.code_weights = code_weights
        self.num_classes = num_classes
        self.in_channels = in_channels
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.fp16_enabled = False
        self.embed_dims = in_channels
        self.num_clusters = num_clusters

        # RWHI配置
        self.use_rwhi = use_rwhi
        self.rwhi_cfg = rwhi_cfg
        
        # [FIX v7.1] 保存传入的 polar_radius，用于全链路统一
        self._config_polar_radius = polar_radius

        # 从kwargs中提取pc_range（如果存在）
        self.pc_range = kwargs.pop(
            'pc_range',
            [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]
        )

        super().__init__(
            num_classes,
            in_channels,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            **kwargs
        )

        self.code_weights = nn.Parameter(
            torch.tensor(self.code_weights),
            requires_grad=False
        )
        self.bbox_coder = build_bbox_coder(bbox_coder)

        # 验证bbox_coder中的pc_range
        if hasattr(self.bbox_coder, 'pc_range'):
            self.pc_range = self.bbox_coder.pc_range
        
        # [FIX v7.1] 支持固定 polar_radius，与 RWHI v5.3 保持一致
        self.map_size, _default_radius = compute_map_size_and_radius(self.pc_range)
        self.polar_radius = self._config_polar_radius if self._config_polar_radius is not None else _default_radius

        self.dn_enabled = query_denoising
        self.dn_group_num = query_denoising_groups
        self.dn_weight = 1.0
        self.dn_bbox_noise_scale = 0.5
        self.dn_label_noise_scale = 0.5

    def _init_layers(self):
        """初始化网络层"""
        self.label_enc = nn.Embedding(self.num_classes + 1, self.embed_dims - 1)

        if self.use_rwhi:
            self._init_rwhi_layers()
        else:
            self._init_original_layers()

    def _init_rwhi_layers(self):
        """初始化RWHI模式的网络层"""
        rwhi_default_cfg = {
            'rwhi_version': 'v3',
            'num_query': self.num_query,
            'pc_range': self.pc_range,
            'bev_grid_size': 100,
            'd_ref': 30.0,
            'beta': 0.0,
            'base_bias': 1.0,
            'z_default': 0.5,
            'max_points': None,
            'radar_channel_map': dict(x=0, y=1, z=2, rcs=3, v_r=4),
            'enabled': True,
            # v5.3 特有参数
            'd_alpha': 2,
            'd_lambda': 1.0,
            'w_d_max': 2.5,
            'epsilon': 0.01,
        }

        if self.rwhi_cfg is not None:
            rwhi_default_cfg.update(self.rwhi_cfg)

        rwhi_default_cfg['num_query'] = self.num_query
        rwhi_default_cfg['embed_dims'] = self.embed_dims
        rwhi_default_cfg['pc_range'] = self.pc_range

        self.rwhi_module = RWHIModule(**rwhi_default_cfg)
        
        # 获取 α embedding 维度 (v5.3 支持)
        self.d_alpha = self.rwhi_module.d_alpha if hasattr(self.rwhi_module, 'd_alpha') else 0
        self.use_alpha_embedding = self.d_alpha > 0

        self.pos2content = nn.Sequential(
            nn.Linear(3, self.embed_dims),
            nn.LayerNorm(self.embed_dims),
            nn.ReLU(inplace=True),
            nn.Linear(self.embed_dims, self.embed_dims - 1),
        )

        for module in self.pos2content:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)
        
        # Feature-Guided Initialization: α embedding 融合层 (v5.3)
        if self.use_alpha_embedding:
            # 将 α embedding 融合到 query feature
            # 输入: [embed_dims + d_alpha], 输出: [embed_dims]
            self.alpha_fusion = nn.Sequential(
                nn.Linear(self.embed_dims + self.d_alpha, self.embed_dims),
                nn.LayerNorm(self.embed_dims),
            )
            for module in self.alpha_fusion:
                if isinstance(module, nn.Linear):
                    nn.init.xavier_uniform_(module.weight)
                    nn.init.zeros_(module.bias)
        else:
            self.alpha_fusion = None

        self.init_query_bbox = nn.Embedding(self.num_query, 10)
        nn.init.constant_(self.init_query_bbox.weight[:, 2:3], 0.5)
        nn.init.zeros_(self.init_query_bbox.weight[:, 8:10])
        nn.init.constant_(self.init_query_bbox.weight[:, 5:6], 0.2)

        with torch.no_grad():
            safety_anchors = self.rwhi_module.safety_anchors
            num_safety = safety_anchors.shape[0]
            self.init_query_bbox.weight[:num_safety, :] = safety_anchors

            if num_safety < self.num_query:
                fallback_theta_d = self.generate_points()
                fallback_num = min(
                    self.num_query - num_safety,
                    fallback_theta_d.shape[0]
                )
                self.init_query_bbox.weight[
                    num_safety:num_safety + fallback_num, :2
                ] = fallback_theta_d[:fallback_num].reshape(-1, 2)

    def _init_original_layers(self):
        """初始化原始模式的网络层"""
        self.init_query_bbox = nn.Embedding(self.num_query, 10)

        nn.init.constant_(self.init_query_bbox.weight[:, 2:3], 0.5)
        nn.init.zeros_(self.init_query_bbox.weight[:, 8:10])
        nn.init.constant_(self.init_query_bbox.weight[:, 5:6], 0.2)

        theta_d = self.generate_points()
        with torch.no_grad():
            self.init_query_bbox.weight[:, :2] = theta_d.reshape(-1, 2)

        self.rwhi_module = None
        self.pos2content = None
        self.d_alpha = 0
        self.use_alpha_embedding = False
        self.alpha_fusion = None

    def init_weights(self):
        """初始化权重"""
        self.transformer.init_weights()

    def generate_points(self):
        """
        生成均匀分布的极坐标锚点 (原始RaCFormer策略)

        Returns:
            theta_d: [num_query, 2] (theta, d) 归一化极坐标
        """
        num_angles = self.num_query // self.num_clusters
        angles = torch.linspace(0, 1, num_angles + 1)[:-1]
        distances = torch.linspace(
            0, 1, self.num_clusters + 2, dtype=torch.float
        )[1:-1]

        angles = angles.view(num_angles, 1).expand(num_angles, self.num_clusters)
        distances = distances.view(1, self.num_clusters).expand(
            num_angles, self.num_clusters
        )

        theta_d = torch.cat(
            [angles[..., None], distances[..., None]], dim=-1
        ).flatten(0, 1)
        return theta_d

    # 修复: 添加@staticmethod装饰器，该方法不使用self
    @staticmethod
    def _extract_radar_points_from_metas(img_metas, batch_size):
        """
        从img_metas中提取雷达点云数据 (用于RWHI)

        Args:
            img_metas: 图像元信息列表
            batch_size: batch大小

        Returns:
            radar_points: [B, M, C] 雷达点云，如果不可用则返回None
        """
        # 修复: 仅捕获预期异常KeyError/TypeError/ValueError
        try:
            radar_points_list = []
            for meta in img_metas:
                if 'radar_points' not in meta:
                    continue

                pts = meta['radar_points']
                if isinstance(pts, torch.Tensor):
                    radar_points_list.append(pts)
                elif isinstance(pts, list) and len(pts) > 0:
                    pts_first = (
                        pts[0] if isinstance(pts[0], torch.Tensor) else None
                    )
                    if pts_first is not None:
                        radar_points_list.append(pts_first)

            if len(radar_points_list) != batch_size:
                return None

            max_points = max(p.shape[0] for p in radar_points_list)
            padded_points = []

            for pts in radar_points_list:
                if pts.shape[0] < max_points:
                    padding = torch.zeros(
                        max_points - pts.shape[0],
                        pts.shape[1],
                        dtype=pts.dtype,
                        device=pts.device
                    )
                    pts = torch.cat([pts, padding], dim=0)
                padded_points.append(pts)

            return torch.stack(padded_points, dim=0)

        except (KeyError, TypeError, ValueError):
            # 修复: 仅捕获预期的数据相关异常，不使用过宽的except Exception
            return None

    # 修复: forward方法签名兼容父类DETRHead
    # 父类签名通常为: forward(self, mlvl_feats, img_metas, **kwargs)
    # 子类新增参数使用关键字参数接收，确保兼容性
    def forward(
        self,
        mlvl_feats,
        img_metas=None,
        *args,
        lss_bev_feats=None,
        radar_bev_feats=None,
        radar_points=None,
        **kwargs
    ):
        """
        前向传播

        修复: 方法签名兼容父类DETRHead
        - mlvl_feats: 放在第一个位置，与父类保持一致
        - img_metas: 第二个位置参数
        - lss_bev_feats, radar_bev_feats: 作为关键字参数接收

        Args:
            mlvl_feats: 多尺度图像特征
            img_metas: 图像元信息 (可选，兼容父类)
            *args: 位置参数，兼容父类调用
            lss_bev_feats: LSS BEV特征
            radar_bev_feats: 雷达BEV特征
            radar_points: 原始雷达点云 [B, M, C]，用于RWHI初始化
            **kwargs: 其他参数，兼容父类签名

        Returns:
            dict: 包含分类和回归预测的字典
        """
        # 处理不同的调用方式
        # 方式1: forward(mlvl_feats, lss_bev_feats, radar_bev_feats, img_metas)
        # 方式2: forward(mlvl_feats, img_metas, lss_bev_feats=..., ...)
        if lss_bev_feats is None and len(args) >= 2:
            # 旧的调用方式: 位置参数
            lss_bev_feats = img_metas  # 第二个位置参数实际是lss_bev_feats
            radar_bev_feats = args[0]
            img_metas = args[1]
            if len(args) > 2:
                radar_points = args[2]

        batch_size = lss_bev_feats.shape[0]
        device = lss_bev_feats.device

        query_bbox, using_dynamic_rwhi, alpha_values = self._prepare_query_bbox(
            batch_size, device, radar_points, img_metas
        )

        query_bbox = self._validate_query_bbox(query_bbox)

        init_query_feat = self._prepare_query_feat(
            query_bbox,
            batch_size,
            device,
            using_dynamic_rwhi,
            self.label_enc,
            alpha_values=alpha_values
        )

        query_bbox, query_feat, attn_mask, mask_dict = self.prepare_for_dn_input(
            batch_size, query_bbox, self.label_enc, img_metas,
            init_query_feat=init_query_feat
        )

        cls_scores, bbox_preds = self.transformer(
            query_bbox,
            query_feat,
            mlvl_feats,
            lss_bev_feats,
            radar_bev_feats,
            attn_mask=attn_mask,
            img_metas=img_metas,
        )

        bbox_preds = self._denormalize_bbox_preds(bbox_preds)

        return self._build_output_dict(cls_scores, bbox_preds, mask_dict)

    def _prepare_query_bbox(self, batch_size, device, radar_points, img_metas):
        """
        准备query边界框
        
        Returns:
            query_bbox: [B, Q, 10] bbox_proposal
            using_dynamic_rwhi: bool 是否使用动态 RWHI
            alpha_values: [B, Q, 1] 或 None, 每个 query 的 α 值 (v5.3)
        """
        using_dynamic_rwhi = False
        alpha_values = None
        
        if self.use_rwhi and self.rwhi_module is not None:
            if radar_points is None:
                radar_points = self._extract_radar_points_from_metas(
                    img_metas, batch_size
                )

            if radar_points is not None:
                radar_points = self._adjust_radar_batch_size(
                    radar_points, batch_size
                )
                radar_mask = (radar_points.abs().sum(dim=-1) > 0)
                query_bbox, alpha_values = self.rwhi_module(
                    radar_points, radar_mask=radar_mask
                )
                using_dynamic_rwhi = True
                query_bbox = self._adjust_query_batch_size(
                    query_bbox, batch_size
                )
                query_bbox = self._expand_query_bbox(query_bbox)
                
                # 调整 alpha_values batch size
                if alpha_values is not None:
                    alpha_values = self._adjust_alpha_batch_size(
                        alpha_values, batch_size
                    )
            else:
                query_bbox = self.init_query_bbox.weight.clone()
                query_bbox = query_bbox.view(
                    1, self.num_query, 10
                ).repeat(batch_size, 1, 1)
        else:
            query_bbox = self.init_query_bbox.weight.clone()
            query_bbox = query_bbox.view(
                1, self.num_query, 10
            ).repeat(batch_size, 1, 1)

        return query_bbox.to(device), using_dynamic_rwhi, alpha_values
    
    @staticmethod
    def _adjust_alpha_batch_size(alpha_values, target_batch_size):
        """调整 alpha_values 的 batch size"""
        if alpha_values is None:
            return None
        
        alpha_batch_size = alpha_values.shape[0]
        if alpha_batch_size == target_batch_size:
            return alpha_values
        
        if alpha_batch_size > target_batch_size:
            return alpha_values[:target_batch_size]
        
        repeat_times = (target_batch_size + alpha_batch_size - 1) // alpha_batch_size
        return alpha_values.repeat(repeat_times, 1, 1)[:target_batch_size]

    # 修复: 添加@staticmethod装饰器，该方法不使用self
    @staticmethod
    def _adjust_radar_batch_size(radar_points, target_batch_size):
        """调整雷达点云batch size"""
        radar_batch_size = radar_points.shape[0]
        if radar_batch_size == target_batch_size:
            return radar_points

        if radar_batch_size > target_batch_size:
            return radar_points[:target_batch_size]

        repeat_times = (
            (target_batch_size + radar_batch_size - 1) // radar_batch_size
        )
        return radar_points.repeat(repeat_times, 1, 1)[:target_batch_size]

    # 修复: 添加@staticmethod装饰器，该方法不使用self
    @staticmethod
    def _adjust_query_batch_size(query_bbox, target_batch_size):
        """调整query batch size"""
        query_batch_size = query_bbox.shape[0]
        if query_batch_size == target_batch_size:
            return query_bbox

        if query_batch_size > target_batch_size:
            return query_bbox[:target_batch_size]

        repeat_times = (
            (target_batch_size + query_batch_size - 1) // query_batch_size
        )
        return query_bbox.repeat(repeat_times, 1, 1)[:target_batch_size]

    @staticmethod
    def _expand_query_bbox(query_bbox):
        """将[B, Q, 3]扩展为[B, Q, 10]，与现有Transformer接口兼容"""
        if query_bbox.shape[-1] == 10:
            return query_bbox
        if query_bbox.shape[-1] != 3:
            raise ValueError(
                f'Unexpected query_bbox dim={query_bbox.shape[-1]} (expected 3 or 10).'
            )
        B, Q, _ = query_bbox.shape
        dtype = query_bbox.dtype
        device = query_bbox.device
        expanded = torch.zeros((B, Q, 10), device=device, dtype=dtype)
        expanded[..., :3] = query_bbox
        expanded[..., 5] = 0.2
        expanded[..., 7] = 1.0
        return expanded

    def _validate_query_bbox(self, query_bbox):
        """
        验证并clamp query坐标范围
        
        [FIX v7.1] 关键修正:
        - d 和 z 需要 clamp 到 (EPS, 1-EPS)，不能是 0 或 1
        - 因为 Transformer 的 refine_bbox 会对 d/z 做 inverse_sigmoid
        - inverse_sigmoid(0) = -inf, inverse_sigmoid(1) = +inf
        - theta 可以是 [0, 1)，因为 refine_bbox 对 theta 使用直接加法
        """
        EPS = 1e-5
        
        if self.training:
            theta_vals = query_bbox[..., 0]
            d_vals = query_bbox[..., 1]
            z_vals = query_bbox[..., 2]

            if theta_vals.min() < 0 or theta_vals.max() > 1:
                print(
                    f"[WARNING RWHI] theta out of [0,1]: "
                    f"min={theta_vals.min().item():.4f}, "
                    f"max={theta_vals.max().item():.4f}"
                )
            # [FIX v7.1] d/z 需要在 (EPS, 1-EPS) 范围内
            if d_vals.min() < EPS or d_vals.max() > 1 - EPS:
                print(
                    f"[WARNING RWHI] d out of (EPS,1-EPS): "
                    f"min={d_vals.min().item():.4f}, "
                    f"max={d_vals.max().item():.4f}"
                )
            if z_vals.min() < EPS or z_vals.max() > 1 - EPS:
                print(
                    f"[WARNING RWHI] z out of (EPS,1-EPS): "
                    f"min={z_vals.min().item():.4f}, "
                    f"max={z_vals.max().item():.4f}"
                )

        # theta 可以是 [0, 1)
        theta = torch.clamp(query_bbox[..., 0:1], 0.0, 1.0 - EPS)
        # [FIX v7.1] d/z 必须 clamp 到 (EPS, 1-EPS) 防止 inverse_sigmoid 溢出
        d = torch.clamp(query_bbox[..., 1:2], EPS, 1.0 - EPS)
        z = torch.clamp(query_bbox[..., 2:3], EPS, 1.0 - EPS)
        rest = query_bbox[..., 3:]
        return torch.cat([theta, d, z, rest], dim=-1)

    def _prepare_query_feat(
        self,
        query_bbox,
        batch_size,
        device,
        using_dynamic_rwhi,
        label_enc,
        alpha_values=None
    ):
        """
        准备query特征
        
        Args:
            query_bbox: [B, Q, 10] bbox_proposal
            batch_size: int
            device: torch.device
            using_dynamic_rwhi: bool
            label_enc: nn.Embedding
            alpha_values: [B, Q, 1] 或 None, 每个 query 的 α 值 (v5.3)
        
        Returns:
            init_query_feat: [B, Q, embed_dims]
        """
        if using_dynamic_rwhi and self.pos2content is not None:
            query_pos = query_bbox[..., :3]
            dynamic_content = self.pos2content(query_pos)

            indicator0 = torch.zeros(
                batch_size, self.num_query, 1, device=device
            )
            init_query_feat = torch.cat([dynamic_content, indicator0], dim=-1)
            
            # Feature-Guided Initialization: 融合 α embedding (v5.3)
            if (self.use_alpha_embedding and 
                alpha_values is not None and 
                self.alpha_fusion is not None and
                self.rwhi_module is not None):
                # 获取 α embedding
                alpha_emb = self.rwhi_module.encode_alpha(alpha_values)  # [B, Q, d_alpha]
                if alpha_emb is not None:
                    alpha_emb = alpha_emb.to(device=device, dtype=init_query_feat.dtype)
                    # 拼接 α embedding 到 query feature
                    feat_with_alpha = torch.cat([init_query_feat, alpha_emb], dim=-1)
                    # 通过融合层 + LayerNorm
                    init_query_feat = self.alpha_fusion(feat_with_alpha)

            # DDP兼容: 确保init_query_bbox参与计算图
            # 修复: dummy_reg必须添加到输出张量，而不是丢弃到_
            # 原始代码(head_origin.py)是加到query_bbox，这里加到init_query_feat
            if self.training:
                dummy_reg = self.init_query_bbox.weight.sum() * 0.0
                dummy_label = label_enc.weight.sum() * 0.0
                dummy_rwhi = 0.0
                if self.rwhi_module is not None:
                    dummy_rwhi = sum(
                        p.sum() for p in self.rwhi_module.parameters()
                    ) * 0.0
                dummy_alpha_fusion = 0.0
                if self.alpha_fusion is not None:
                    dummy_alpha_fusion = sum(
                        p.sum() for p in self.alpha_fusion.parameters()
                    ) * 0.0
                init_query_feat = (
                    init_query_feat + dummy_reg + dummy_label + dummy_rwhi + dummy_alpha_fusion
                )

            return init_query_feat

        indicator0 = torch.zeros([self.num_query, 1], device=device)
        init_query_feat = label_enc.weight[self.num_classes].repeat(
            self.num_query, 1
        )
        init_query_feat = torch.cat(
            [init_query_feat, indicator0], dim=1
        ).repeat(batch_size, 1, 1)

        if (self.training and self.pos2content is not None
                and self.rwhi_module is not None):
            dummy_p2c = sum(
                p.sum() for p in self.pos2content.parameters()
            ) * 0.0
            dummy_rwhi = sum(
                p.sum() for p in self.rwhi_module.parameters()
            ) * 0.0
            dummy_alpha_fusion = 0.0
            if self.alpha_fusion is not None:
                dummy_alpha_fusion = sum(
                    p.sum() for p in self.alpha_fusion.parameters()
                ) * 0.0
            init_query_feat = init_query_feat + dummy_p2c + dummy_rwhi + dummy_alpha_fusion

        return init_query_feat

    def _denormalize_bbox_preds(self, bbox_preds):
        """反归一化边界框预测"""
        bbox_preds[..., 0] = (
            bbox_preds[..., 0] * (self.pc_range[3] - self.pc_range[0])
            + self.pc_range[0]
        )
        bbox_preds[..., 1] = (
            bbox_preds[..., 1] * (self.pc_range[4] - self.pc_range[1])
            + self.pc_range[1]
        )
        bbox_preds[..., 2] = (
            bbox_preds[..., 2] * (self.pc_range[5] - self.pc_range[2])
            + self.pc_range[2]
        )

        return torch.cat([
            bbox_preds[..., 0:2],
            bbox_preds[..., 3:5],
            bbox_preds[..., 2:3],
            bbox_preds[..., 5:10],
        ], dim=-1)

    # 修复: 添加@staticmethod装饰器，该方法不使用self
    @staticmethod
    def _build_output_dict(cls_scores, bbox_preds, mask_dict):
        """构建输出字典"""
        if mask_dict is not None and mask_dict['pad_size'] > 0:
            pad_size = mask_dict['pad_size']
            output_known_cls_scores = cls_scores[:, :, :pad_size, :]
            output_known_bbox_preds = bbox_preds[:, :, :pad_size, :]
            output_cls_scores = cls_scores[:, :, pad_size:, :]
            output_bbox_preds = bbox_preds[:, :, pad_size:, :]

            mask_dict['output_known_lbs_bboxes'] = (
                output_known_cls_scores,
                output_known_bbox_preds
            )

            return {
                'all_cls_scores': output_cls_scores,
                'all_bbox_preds': output_bbox_preds,
                'enc_cls_scores': None,
                'enc_bbox_preds': None,
                'dn_mask_dict': mask_dict,
            }

        return {
            'all_cls_scores': cls_scores,
            'all_bbox_preds': bbox_preds,
            'enc_cls_scores': None,
            'enc_bbox_preds': None,
        }

    def prepare_for_dn_input(
        self,
        batch_size,
        init_query_bbox,
        label_enc,
        img_metas,
        init_query_feat=None
    ):
        """准备Denoising输入"""
        device = init_query_bbox.device

        if init_query_feat is None:
            indicator0 = torch.zeros([self.num_query, 1], device=device)
            init_query_feat = label_enc.weight[self.num_classes].repeat(
                self.num_query, 1
            )
            init_query_feat = torch.cat(
                [init_query_feat, indicator0], dim=1
            ).repeat(batch_size, 1, 1)

        if not (self.training and self.dn_enabled):
            return init_query_bbox, init_query_feat, None, None

        return self._prepare_dn_training_input(
            batch_size, init_query_bbox, init_query_feat, label_enc,
            img_metas, device
        )

    def _prepare_dn_training_input(
        self,
        batch_size,
        init_query_bbox,
        init_query_feat,
        label_enc,
        img_metas,
        device
    ):
        """准备去噪训练输入"""
        # 修复: 使用to(device)替代cuda()，兼容CPU环境
        targets = [{
            'bboxes': torch.cat([
                meta['gt_bboxes_3d'].gravity_center,
                meta['gt_bboxes_3d'].tensor[:, 3:]
            ], dim=1).to(device),
            'labels': meta['gt_labels_3d'].to(device).long()
        } for meta in img_metas]

        # 修复: 使用torch.ones创建float张量，避免后续类型问题
        known = [
            torch.ones_like(t['labels'].float(), device=device)
            for t in targets
        ]
        known_num = [k.sum().item() for k in known]

        # 修复: 确保unmask_bbox和unmask_label是float类型张量
        # 原问题: torch.cat(known)可能产生非float张量导致类型错误
        unmask_bbox = unmask_label = torch.cat(known).float()

        labels = torch.cat([t['labels'] for t in targets]).clone()
        bboxes = torch.cat([t['bboxes'] for t in targets]).clone()
        batch_idx = torch.cat([
            torch.full_like(t['labels'].long(), idx)
            for idx, t in enumerate(targets)
        ])

        known_indice = torch.nonzero(unmask_label + unmask_bbox).view(-1)

        known_indice = known_indice.repeat(self.dn_group_num, 1).view(-1)
        known_labels = labels.repeat(self.dn_group_num, 1).view(-1)
        known_bid = batch_idx.repeat(self.dn_group_num, 1).view(-1)
        known_bboxs = bboxes.repeat(self.dn_group_num, 1)
        known_labels_expand = known_labels.clone()
        known_bbox_expand = known_bboxs.clone()

        wlh = known_bbox_expand[..., 3:6].clone()
        known_bbox_expand = encode_bbox(known_bbox_expand, self.pc_range)
        known_bbox_expand = xy2theta_d_coods(
                    known_bbox_expand, self.map_size, self.polar_radius
                )

        if self.dn_bbox_noise_scale > 0:
            known_bbox_expand = self._add_bbox_noise(known_bbox_expand, wlh)

        known_bbox_expand[..., 0:3].clamp_(min=0.0, max=1.0)

        if self.dn_label_noise_scale > 0:
            known_labels_expand = self._add_label_noise(known_labels_expand)

        known_feat_expand = label_enc(known_labels_expand)
        indicator1 = torch.ones(
            [known_feat_expand.shape[0], 1], device=device
        )
        known_feat_expand = torch.cat([known_feat_expand, indicator1], dim=1)

        dn_single_pad = int(max(known_num)) if known_num else 0
        dn_pad_size = int(dn_single_pad * self.dn_group_num)

        dn_query_bbox = torch.zeros(
            [batch_size, dn_pad_size, init_query_bbox.shape[-1]],
            device=device
        )
        dn_query_feat = torch.zeros(
            [batch_size, dn_pad_size, self.embed_dims],
            device=device
        )

        # 修复: 为map_known_indice设置默认值，避免未赋值引用
        map_known_indice = torch.tensor([], device=device, dtype=torch.long)

        if known_num and any(num > 0 for num in known_num):
            # 修复: range()需要int，而known_num中是float，需显式转换
            map_known_indice = torch.cat([
                torch.tensor(range(int(num)), device=device)
                for num in known_num
            ])
            map_known_indice = torch.cat([
                map_known_indice + dn_single_pad * grp_idx
                for grp_idx in range(self.dn_group_num)
            ]).long()

        if len(known_bid) > 0 and len(map_known_indice) > 0:
            dn_query_bbox[known_bid.long(), map_known_indice] = (
                known_bbox_expand
            )
            dn_query_feat[known_bid.long(), map_known_indice] = (
                known_feat_expand
            )

        input_query_bbox = torch.cat([dn_query_bbox, init_query_bbox], dim=1)
        input_query_feat = torch.cat([dn_query_feat, init_query_feat], dim=1)

        total_size = dn_pad_size + self.num_query
        attn_mask = torch.ones([total_size, total_size], device=device) < 0

        attn_mask[dn_pad_size:, :dn_pad_size] = True
        for grp_idx in range(self.dn_group_num):
            start = dn_single_pad * grp_idx
            end = dn_single_pad * (grp_idx + 1)
            if grp_idx == 0:
                attn_mask[start:end, end:dn_pad_size] = True
            elif grp_idx == self.dn_group_num - 1:
                attn_mask[start:end, :start] = True
            else:
                attn_mask[start:end, end:dn_pad_size] = True
                attn_mask[start:end, :start] = True

        mask_dict = {
            'known_indice': known_indice.long(),
            'batch_idx': batch_idx.long(),
            'map_known_indice': map_known_indice.long(),
            'known_lbs_bboxes': (known_labels, known_bboxs),
            'pad_size': dn_pad_size
        }

        return input_query_bbox, input_query_feat, attn_mask, mask_dict

    def _add_bbox_noise(self, known_bbox_expand, wlh):
        """
        添加边界框噪声
        
        [FIX v7.1] 使用 self.polar_radius (从 config 传入或从 pc_range 计算)
        确保与 RWHI v5.3 和 Transformer 使用相同的 polar_radius
        """
        rand_prob = torch.rand_like(known_bbox_expand) * 2 - 1.0

        arc_len_ratio = (
            torch.sqrt(wlh[..., 0:1]**2 + wlh[..., 1:2]**2)
            / (2 * torch.pi * known_bbox_expand[..., 1:2] * self.polar_radius)
        )
        theta_delta = (
            torch.mul(rand_prob[..., 0:1], arc_len_ratio / 2)
            * self.dn_bbox_noise_scale
            * known_bbox_expand[..., 1:2]
        )

        d_delta = (
            torch.mul(
                rand_prob[..., 1:2],
                torch.sqrt(wlh[..., 0:1]**2 + wlh[..., 1:2]**2)
                / (self.polar_radius * 2)
            )
            * self.dn_bbox_noise_scale
        )

        known_bbox_expand[..., 0:1] += theta_delta
        known_bbox_expand[..., 0:1] = (
            ((known_bbox_expand[..., 0:1] + 1) * 2 * torch.pi
             % (2 * torch.pi))
            / (2 * torch.pi)
        )
        known_bbox_expand[..., 1:2] += d_delta
        known_bbox_expand[..., 2:3] += (
            torch.mul(rand_prob[..., 2:3], wlh[..., 2:3] / 16)
            * self.dn_bbox_noise_scale
        )

        return known_bbox_expand

    def _add_label_noise(self, known_labels_expand):
        """
        添加标签噪声

        修复: 确保所有张量操作类型一致，避免类型错误
        """
        # 修复: 显式将known_labels_expand转为float进行比较操作
        prob = torch.rand_like(known_labels_expand.float())
        # 修复: prob < threshold 产生bool张量，传给nonzero是正确的
        chosen_indice = torch.nonzero(
            prob < self.dn_label_noise_scale
        ).view(-1)
        # 修复: 使用正确的dtype创建随机标签
        new_label = torch.randint(
            0, self.num_classes,
            (chosen_indice.numel(),),
            device=known_labels_expand.device,
            dtype=known_labels_expand.dtype
        )
        known_labels_expand.scatter_(0, chosen_indice, new_label)
        return known_labels_expand

    # 修复: 添加@staticmethod装饰器，该方法不使用self
    @staticmethod
    def prepare_for_dn_loss(mask_dict):
        """
        准备去噪损失

        修复: 添加@staticmethod装饰器，该方法不访问任何实例属性
        """
        cls_scores, bbox_preds = mask_dict['output_known_lbs_bboxes']
        known_labels, known_bboxs = mask_dict['known_lbs_bboxes']
        map_known_indice = mask_dict['map_known_indice'].long()
        known_indice = mask_dict['known_indice'].long()
        batch_idx = mask_dict['batch_idx'].long()
        bid = batch_idx[known_indice]
        num_tgt = known_indice.numel()

        if len(cls_scores) > 0:
            cls_scores = cls_scores.permute(1, 2, 0, 3)[
                (bid, map_known_indice)
            ].permute(1, 0, 2)
            bbox_preds = bbox_preds.permute(1, 2, 0, 3)[
                (bid, map_known_indice)
            ].permute(1, 0, 2)

        return known_labels, known_bboxs, cls_scores, bbox_preds, num_tgt

    def dn_loss_single(
        self,
        cls_scores,
        bbox_preds,
        known_bboxs,
        known_labels,
        num_total_pos=None
    ):
        """单层去噪损失"""
        num_total_pos = cls_scores.new_tensor([num_total_pos])
        num_total_pos = torch.clamp(reduce_mean(num_total_pos), min=1.0).item()

        cls_scores = cls_scores.reshape(-1, self.cls_out_channels)
        bbox_weights = torch.ones_like(bbox_preds)
        label_weights = torch.ones_like(known_labels)

        loss_cls = self.loss_cls(
            cls_scores,
            known_labels.long(),
            label_weights,
            avg_factor=num_total_pos
        )

        bbox_preds = bbox_preds.reshape(-1, bbox_preds.size(-1))
        normalized_bbox_targets = normalize_bbox(known_bboxs)
        isnotnan = torch.isfinite(normalized_bbox_targets).all(dim=-1)
        bbox_weights = bbox_weights * self.code_weights

        loss_bbox = self.loss_bbox(
            bbox_preds[isnotnan, :10],
            normalized_bbox_targets[isnotnan, :10],
            bbox_weights[isnotnan, :10],
            avg_factor=num_total_pos
        )

        loss_cls = self.dn_weight * torch.nan_to_num(loss_cls)
        loss_bbox = self.dn_weight * torch.nan_to_num(loss_bbox)

        return loss_cls, loss_bbox

    @force_fp32(apply_to=('preds_dicts',))
    def calc_dn_loss(self, loss_dict, preds_dicts, num_dec_layers):
        """计算去噪损失"""
        known_labels, known_bboxs, cls_scores, bbox_preds, num_tgt = \
            self.prepare_for_dn_loss(preds_dicts['dn_mask_dict'])

        all_known_bboxs_list = [known_bboxs for _ in range(num_dec_layers)]
        all_known_labels_list = [known_labels for _ in range(num_dec_layers)]
        all_num_tgts_list = [num_tgt for _ in range(num_dec_layers)]

        dn_losses_cls, dn_losses_bbox = multi_apply(
            self.dn_loss_single,
            cls_scores,
            bbox_preds,
            all_known_bboxs_list,
            all_known_labels_list,
            all_num_tgts_list
        )

        loss_dict['loss_cls_dn'] = dn_losses_cls[-1]
        loss_dict['loss_bbox_dn'] = dn_losses_bbox[-1]

        for layer_idx, (loss_cls_i, loss_bbox_i) in enumerate(
            zip(dn_losses_cls[:-1], dn_losses_bbox[:-1])
        ):
            loss_dict[f'd{layer_idx}.loss_cls_dn'] = loss_cls_i
            loss_dict[f'd{layer_idx}.loss_bbox_dn'] = loss_bbox_i

        return loss_dict

    # 修复: 方法签名兼容父类DETRHead
    def _get_target_single(
        self,
        cls_score,
        bbox_pred,
        gt_labels,
        gt_bboxes,
        gt_bboxes_ignore=None,
        *args,
        **kwargs
    ):
        """获取单个样本的目标"""
        num_bboxes = bbox_pred.size(0)

        assign_result = self.assigner.assign(
            bbox_pred, cls_score, gt_bboxes, gt_labels,
            gt_bboxes_ignore, self.code_weights, True
        )
        sampling_result = self.sampler.sample(
            assign_result, bbox_pred, gt_bboxes
        )
        pos_inds = sampling_result.pos_inds
        neg_inds = sampling_result.neg_inds

        labels = gt_bboxes.new_full(
            (num_bboxes,), self.num_classes, dtype=torch.long
        )
        labels[pos_inds] = gt_labels[sampling_result.pos_assigned_gt_inds]
        label_weights = gt_bboxes.new_ones(num_bboxes)

        bbox_targets = torch.zeros_like(bbox_pred)[..., :9]
        bbox_weights = torch.zeros_like(bbox_pred)
        bbox_weights[pos_inds] = 1.0

        bbox_targets[pos_inds] = sampling_result.pos_gt_bboxes

        return (
            labels, label_weights, bbox_targets,
            bbox_weights, pos_inds, neg_inds
        )

    # 修复: 方法签名兼容父类DETRHead
    def get_targets(
        self,
        cls_scores_list,
        bbox_preds_list,
        gt_bboxes_list,
        gt_labels_list,
        gt_bboxes_ignore_list=None,
        *args,
        **kwargs
    ):
        """获取目标"""
        assert gt_bboxes_ignore_list is None, \
            'Only supports for gt_bboxes_ignore setting to None.'

        num_imgs = len(cls_scores_list)
        gt_bboxes_ignore_list = [gt_bboxes_ignore_list for _ in range(num_imgs)]

        (
            labels_list,
            label_weights_list,
            bbox_targets_list,
            bbox_weights_list,
            pos_inds_list,
            neg_inds_list
        ) = multi_apply(
            self._get_target_single,
            cls_scores_list,
            bbox_preds_list,
            gt_labels_list,
            gt_bboxes_list,
            gt_bboxes_ignore_list
        )

        num_total_pos = sum(inds.numel() for inds in pos_inds_list)
        num_total_neg = sum(inds.numel() for inds in neg_inds_list)

        return (
            labels_list,
            label_weights_list,
            bbox_targets_list,
            bbox_weights_list,
            num_total_pos,
            num_total_neg
        )

    # 修复: 方法签名兼容父类DETRHead
    def loss_single(
        self,
        cls_scores,
        bbox_preds,
        gt_bboxes_list,
        gt_labels_list,
        gt_bboxes_ignore_list=None,
        *args,
        **kwargs
    ):
        """单层损失计算"""
        num_imgs = cls_scores.size(0)
        cls_scores_list = [cls_scores[img_idx] for img_idx in range(num_imgs)]
        bbox_preds_list = [bbox_preds[img_idx] for img_idx in range(num_imgs)]

        cls_reg_targets = self.get_targets(
            cls_scores_list,
            bbox_preds_list,
            gt_bboxes_list,
            gt_labels_list,
            gt_bboxes_ignore_list
        )

        (
            labels_list,
            label_weights_list,
            bbox_targets_list,
            bbox_weights_list,
            num_total_pos,
            num_total_neg
        ) = cls_reg_targets

        labels = torch.cat(labels_list, 0)
        label_weights = torch.cat(label_weights_list, 0)
        bbox_targets = torch.cat(bbox_targets_list, 0)
        bbox_weights = torch.cat(bbox_weights_list, 0)

        cls_scores = cls_scores.reshape(-1, self.cls_out_channels)
        cls_avg_factor = (
            num_total_pos * 1.0 + num_total_neg * self.bg_cls_weight
        )

        if self.sync_cls_avg_factor:
            cls_avg_factor = reduce_mean(
                cls_scores.new_tensor([cls_avg_factor])
            )

        cls_avg_factor = max(cls_avg_factor, 1)
        loss_cls = self.loss_cls(
            cls_scores, labels, label_weights, avg_factor=cls_avg_factor
        )

        num_total_pos = loss_cls.new_tensor([num_total_pos])
        num_total_pos = torch.clamp(reduce_mean(num_total_pos), min=1).item()

        bbox_preds = bbox_preds.reshape(-1, bbox_preds.size(-1))
        normalized_bbox_targets = normalize_bbox(bbox_targets)
        isnotnan = torch.isfinite(normalized_bbox_targets).all(dim=-1)
        bbox_weights = bbox_weights * self.code_weights

        loss_bbox = self.loss_bbox(
            bbox_preds[isnotnan, :10],
            normalized_bbox_targets[isnotnan, :10],
            bbox_weights[isnotnan, :10],
            avg_factor=num_total_pos
        )

        loss_cls = torch.nan_to_num(loss_cls)
        loss_bbox = torch.nan_to_num(loss_bbox)

        return loss_cls, loss_bbox

    @force_fp32(apply_to=('preds_dicts',))
    def loss(
        self,
        gt_bboxes_list,
        gt_labels_list,
        preds_dicts,
        gt_bboxes_ignore=None,
        *args,
        **kwargs
    ):
        """计算总损失"""
        assert gt_bboxes_ignore is None, \
            f'{self.__class__.__name__} only supports ' \
            f'for gt_bboxes_ignore setting to None.'

        all_cls_scores = preds_dicts['all_cls_scores']
        all_bbox_preds = preds_dicts['all_bbox_preds']
        enc_cls_scores = preds_dicts['enc_cls_scores']
        enc_bbox_preds = preds_dicts['enc_bbox_preds']

        num_dec_layers = len(all_cls_scores)
        device = gt_labels_list[0].device

        gt_bboxes_list = [
            torch.cat(
                (gt_bboxes.gravity_center, gt_bboxes.tensor[:, 3:]),
                dim=1
            ).to(device)
            for gt_bboxes in gt_bboxes_list
        ]

        all_gt_bboxes_list = [gt_bboxes_list for _ in range(num_dec_layers)]
        all_gt_labels_list = [gt_labels_list for _ in range(num_dec_layers)]
        all_gt_bboxes_ignore_list = [
            gt_bboxes_ignore for _ in range(num_dec_layers)
        ]

        losses_cls, losses_bbox = multi_apply(
            self.loss_single,
            all_cls_scores,
            all_bbox_preds,
            all_gt_bboxes_list,
            all_gt_labels_list,
            all_gt_bboxes_ignore_list
        )

        loss_dict = {}

        if enc_cls_scores is not None:
            binary_labels_list = [
                torch.zeros_like(gt_labels_list[label_idx])
                for label_idx in range(len(all_gt_labels_list))
            ]
            enc_loss_cls, enc_losses_bbox = self.loss_single(
                enc_cls_scores,
                enc_bbox_preds,
                gt_bboxes_list,
                binary_labels_list,
                gt_bboxes_ignore
            )
            loss_dict['enc_loss_cls'] = enc_loss_cls
            loss_dict['enc_loss_bbox'] = enc_losses_bbox

        if ('dn_mask_dict' in preds_dicts
                and preds_dicts['dn_mask_dict'] is not None):
            loss_dict = self.calc_dn_loss(
                loss_dict, preds_dicts, num_dec_layers
            )

        loss_dict['loss_cls'] = losses_cls[-1]
        loss_dict['loss_bbox'] = losses_bbox[-1]

        for layer_idx, (loss_cls_i, loss_bbox_i) in enumerate(
            zip(losses_cls[:-1], losses_bbox[:-1])
        ):
            loss_dict[f'd{layer_idx}.loss_cls'] = loss_cls_i
            loss_dict[f'd{layer_idx}.loss_bbox'] = loss_bbox_i

        return loss_dict

    @force_fp32(apply_to=('preds_dicts',))
    def get_bboxes(
        self,
        preds_dicts,
        img_metas,
        rescale=False,
        *args,
        **kwargs
    ):
        """
        获取检测框结果

        Args:
            preds_dicts: 预测字典
            img_metas: 图像元信息
            rescale: 是否缩放
            *args, **kwargs: 兼容父类签名

        Returns:
            list: 每个样本的检测结果 [bboxes, scores, labels]
        """
        preds_dicts = self.bbox_coder.decode(preds_dicts)
        num_samples = len(preds_dicts)
        ret_list = []

        for sample_idx in range(num_samples):
            preds = preds_dicts[sample_idx]
            bboxes = preds['bboxes']
            scores = preds['scores']
            labels = preds['labels']

            if bboxes.shape[0] == 0:
                empty_bboxes = bboxes.new_zeros((0, 9))
                bboxes = LiDARInstance3DBoxes(empty_bboxes, 9)
                ret_list.append([bboxes, scores, labels])
                continue

            bboxes[:, 2] = bboxes[:, 2] - bboxes[:, 5] * 0.5

            if VERSION.name == 'v0.17.1':
                # 修复E741: 变量名l改为length，w改为width
                width = copy.deepcopy(bboxes[:, 3])
                length = copy.deepcopy(bboxes[:, 4])
                bboxes[:, 3] = length
                bboxes[:, 4] = width
                bboxes[:, 6] = -bboxes[:, 6] - math.pi / 2

            bboxes = LiDARInstance3DBoxes(bboxes, 9)
            ret_list.append([bboxes, scores, labels])

        return ret_list


# 修复: 保留旧类名作为别名，确保向后兼容
RaCFormer_head = RaCFormerHead
