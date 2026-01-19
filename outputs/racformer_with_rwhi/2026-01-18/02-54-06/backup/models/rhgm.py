# ==============================================================================
# RHGM模块 (Radar Hybrid Generation Module)
# 来源: HGSFusion/hybrid_pts/hybrid_radar_pts_vod.py
# 功能: 雷达点云混合生成，包含前景点筛选、混合概率分布生成、虚拟点云生成
# 接入位置: RaCFormer雷达分支最前端
# ==============================================================================

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from mmcv.runner import BaseModule


class RHGM(BaseModule):
    """
    Radar Hybrid Generation Module (RHGM)
    
    雷达混合点云生成模块，用于在RaCFormer中增强雷达点云表示。
    
    功能:
    1. 前景点筛选: 基于相机语义掩码筛选前景雷达点
    2. 混合概率分布生成: 高斯分布 + 均匀分布采样
    3. 虚拟点云生成: 从2D掩码区域生成虚拟3D点
    
    输入:
        - radar_points: 原始雷达点云 [N, C] (x, y, z, ...)
        - semantic_masks: 相机语义分割掩码 [N_masks, H, W]
        - lidar2img: 雷达到图像的投影矩阵 [4, 4]
        - img2lidar: 图像到雷达的投影矩阵 [4, 4]
        - intrinsics: 相机内参 [3, 3]
        
    输出:
        - hybrid_points: 混合点云 (原始点 + 前景点 + 生成点) [M, C]
    """
    
    def __init__(self,
                 num_virtual_points=100,
                 dist_thresh=3000,
                 gauss_sigma=7,
                 gauss_kernel_size=51,
                 gauss_uniform_ratio=[1, 4],
                 input_channels=7,
                 output_channels=7,
                 enabled=True,
                 init_cfg=None):
        """
        初始化RHGM模块
        
        Args:
            num_virtual_points: 每个掩码生成的虚拟点数量
            dist_thresh: 虚拟点与真实点的最大距离阈值
            gauss_sigma: 高斯分布的标准差
            gauss_kernel_size: 高斯核大小
            gauss_uniform_ratio: [高斯采样k值, 均匀采样k值]
            input_channels: 输入点云通道数 (x,y,z,rcs,vr,vr_comp,time)
            output_channels: 输出点云通道数
            enabled: 是否启用RHGM模块
        """
        super(RHGM, self).__init__(init_cfg=init_cfg)
        
        self.num_virtual_points = num_virtual_points
        self.dist_thresh = dist_thresh
        self.gauss_sigma = gauss_sigma
        self.gauss_kernel_size = gauss_kernel_size
        self.gauss_uniform_ratio = gauss_uniform_ratio
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.enabled = enabled
        
        # 预计算高斯核
        self.register_buffer('gauss_kernel', self._create_gaussian_kernel())
        
        # # 维度转换层 (如果输入输出通道不一致)
        # if input_channels != output_channels:
        #     self.channel_adapter = nn.Linear(input_channels, output_channels)
        # else:
        #     self.channel_adapter = None
    
    def _create_gaussian_kernel(self):
        """创建2D高斯核"""
        size = self.gauss_kernel_size
        sigma = self.gauss_sigma
        
        m, n = [(size - 1.) / 2. for _ in range(2)]
        y, x = np.ogrid[-m:m + 1, -n:n + 1]
        
        h = np.exp(-(x * x + y * y) / (2 * sigma * sigma))
        h[h < np.finfo(h.dtype).eps * h.max()] = 0
        
        return torch.from_numpy(h).float()
    
    def project_points_to_image(self, points, lidar2img, img_shape):
        """
        将雷达点投影到图像平面
        
        Args:
            points: 雷达点云 [N, 3+] (至少包含x,y,z)
            lidar2img: 雷达到图像的变换矩阵 [4, 4]
            img_shape: 图像尺寸 (H, W)
            
        Returns:
            projected_points: 投影后的2D坐标 [N, 5] (u, v, depth, valid, camera_id)
        """
        H, W = img_shape
        
        # 扩展为齐次坐标
        points_xyz = points[:, :3]  # [N, 3]
        ones = torch.ones((points_xyz.shape[0], 1), device=points.device, dtype=points.dtype)
        points_homo = torch.cat([points_xyz, ones], dim=1)  # [N, 4]
        
        # 投影到图像平面
        points_img = torch.matmul(lidar2img.float(), points_homo.T).T  # [N, 4]
        
        # 归一化
        depth = points_img[:, 2:3]
        depth = torch.clamp(depth, min=1e-5)  # 避免除零
        points_2d = points_img[:, :2] / depth  # [N, 2]
        
        # 检查有效性 (在图像范围内且深度为正)
        valid = (points_2d[:, 0] >= 0) & (points_2d[:, 0] < W) & \
                (points_2d[:, 1] >= 0) & (points_2d[:, 1] < H) & \
                (depth.squeeze() > 0)
        
        # 组合输出
        camera_id = torch.zeros((points_2d.shape[0], 1), device=points.device)
        projected = torch.cat([points_2d, depth, valid.float().unsqueeze(1), camera_id], dim=1)
        
        return projected  # [N, 5]
    
    def is_within_mask(self, points_uv, masks, img_shape):
        """
        判断点是否在语义掩码内
        
        Args:
            points_uv: 2D投影点 [N, 2] (u, v)
            masks: 语义掩码 [N_masks, H, W]
            img_shape: 图像尺寸 (H, W)
            
        Returns:
            valid_matrix: 有效性矩阵 [N_points, N_masks]
        """
        H, W = img_shape
        
        # 确保坐标在有效范围内
        points_uv = points_uv.long()
        points_uv[:, 0] = torch.clamp(points_uv[:, 0], 0, W - 1)
        points_uv[:, 1] = torch.clamp(points_uv[:, 1], 0, H - 1)
        
        # 检查每个点是否在每个掩码内
        # masks: [N_masks, H, W]
        # 使用索引获取每个点对应位置的掩码值
        valid_matrix = masks[:, points_uv[:, 1], points_uv[:, 0]]  # [N_masks, N_points]
        
        return valid_matrix.T  # [N_points, N_masks]
    
    def generate_virtual_points(self, masks, valid_matrix, points, raw_points,
                                intrinsics, lidar2img, img_shape):
        """
        生成虚拟点云
        
        Args:
            masks: 语义掩码 [N_masks, H, W]
            valid_matrix: 有效性矩阵 [N_points, N_masks]
            points: 投影后的点 [N, 5]
            raw_points: 原始雷达点 [N, C]
            intrinsics: 相机内参 [3, 3]
            lidar2img: 雷达到图像变换 [4, 4]
            img_shape: 图像尺寸 (H, W)
            
        Returns:
            virtual_points: 虚拟点云 [M, C]
            foreground_mask: 前景点掩码 [N]
        """
        H, W = img_shape
        device = masks.device
        num_masks = masks.shape[0]
        
        if num_masks == 0:
            return None, torch.zeros(raw_points.shape[0], dtype=torch.bool, device=device)
        
        # 前景点掩码
        foreground_mask = valid_matrix.sum(dim=1) > 0  # [N_points]
        
        # 为每个掩码生成采样点
        virtual_points_list = []
        gauss = self.gauss_kernel.to(device)
        
        for mask_idx in range(num_masks):
            mask = masks[mask_idx]  # [H, W]
            
            # 找到该掩码内的真实点
            mask_points_idx = torch.where(valid_matrix[:, mask_idx] > 0)[0]
            
            if len(mask_points_idx) > 0:
                # 有真实点的情况：使用高斯分布采样
                cur_points = points[mask_points_idx]  # [K, 5]
                
                # 创建概率图
                prob_map = torch.zeros_like(mask, dtype=torch.float32)
                kr = self.gauss_kernel_size // 2
                
                for point in cur_points:
                    x, y = int(point[0]), int(point[1])
                    
                    # 计算高斯核覆盖区域
                    x1, x2 = max(0, x - kr), min(W, x + kr + 1)
                    y1, y2 = max(0, y - kr), min(H, y + kr + 1)
                    
                    gx1, gx2 = kr - (x - max(0, x - kr)), kr + min(W, x + kr + 1) - x
                    gy1, gy2 = kr - (y - max(0, y - kr)), kr + min(H, y + kr + 1) - y
                    
                    if x2 > x1 and y2 > y1:
                        prob_map[y1:y2, x1:x2] += gauss[gy1:gy2, gx1:gx2]
                
                # 在掩码内采样
                prob_map = prob_map * mask
                mask_indices = mask.nonzero(as_tuple=False)  # [M, 2] (y, x)
                
                if len(mask_indices) > 0:
                    prob_values = prob_map[mask_indices[:, 0], mask_indices[:, 1]] + 1e-6
                    
                    # 高斯采样
                    num_gauss = self.num_virtual_points // 2
                    gauss_indices = torch.multinomial(prob_values, num_gauss, replacement=True)
                    
                    # 均匀采样
                    num_uniform = self.num_virtual_points - num_gauss
                    uniform_indices = torch.randperm(len(mask_indices), device=device)[:num_uniform]
                    
                    selected_indices = torch.cat([gauss_indices, uniform_indices])
                    selected_points_2d = mask_indices[selected_indices]  # [num_virtual, 2] (y, x)
                    
                    # 从最近的真实点获取深度
                    selected_xy = selected_points_2d[:, [1, 0]].float()  # [num_virtual, 2] (x, y)
                    real_xy = cur_points[:, :2]  # [K, 2]
                    
                    # 计算最近邻
                    dist = torch.cdist(selected_xy, real_xy)  # [num_virtual, K]
                    nearest_idx = dist.argmin(dim=1)  # [num_virtual]
                    nearest_depth = cur_points[nearest_idx, 2]  # [num_virtual]
                    
                    # 反投影到3D
                    virtual_3d = self._unproject_to_3d(
                        selected_xy, nearest_depth, intrinsics, lidar2img
                    )
                    
                    # 复制最近点的属性
                    virtual_attrs = raw_points[mask_points_idx[nearest_idx], 3:]
                    virtual_points = torch.cat([virtual_3d, virtual_attrs], dim=1)
                    virtual_points_list.append(virtual_points)
            else:
                # 无真实点的情况：从掩码均匀采样
                mask_indices = mask.nonzero(as_tuple=False)  # [M, 2]
                if len(mask_indices) > 0:
                    selected_indices = torch.randperm(len(mask_indices), device=device)[:self.num_virtual_points]
                    if len(selected_indices) < self.num_virtual_points:
                        # 重复填充
                        repeat_times = self.num_virtual_points // len(selected_indices) + 1
                        selected_indices = selected_indices.repeat(repeat_times)[:self.num_virtual_points]
                    # 这种情况下没有深度信息，跳过
                    pass
        
        if len(virtual_points_list) > 0:
            virtual_points = torch.cat(virtual_points_list, dim=0)
        else:
            virtual_points = None
        
        return virtual_points, foreground_mask
    
    def _unproject_to_3d(self, points_2d, depth, intrinsics, lidar2img):
        """
        将2D点反投影回3D空间
        
        Args:
            points_2d: 2D点坐标 [N, 2] (x, y)
            depth: 深度值 [N]
            intrinsics: 相机内参 [3, 3]
            lidar2img: 雷达到图像变换 [4, 4]
            
        Returns:
            points_3d: 3D点坐标 [N, 3]
        """
        N = points_2d.shape[0]
        device = points_2d.device
        
        # 反内参变换
        fx, fy = intrinsics[0, 0], intrinsics[1, 1]
        cx, cy = intrinsics[0, 2], intrinsics[1, 2]
        
        x = (points_2d[:, 0] - cx) * depth / fx
        y = (points_2d[:, 1] - cy) * depth / fy
        z = depth
        
        # 相机坐标系下的3D点
        points_cam = torch.stack([x, y, z], dim=1)  # [N, 3]
        
        # 变换到雷达坐标系
        img2lidar = torch.inverse(lidar2img.float())
        ones = torch.ones((N, 1), device=device, dtype=points_cam.dtype)
        points_homo = torch.cat([points_cam, ones], dim=1)  # [N, 4]
        points_3d = torch.matmul(img2lidar, points_homo.T).T[:, :3]  # [N, 3]
        
        return points_3d
    
    def forward(self, radar_points, semantic_masks=None, lidar2img=None, 
                intrinsics=None, img_shape=None):
        """
        RHGM前向传播
        
        Args:
            radar_points: 原始雷达点云 [N, C]
            semantic_masks: 语义分割掩码 [N_masks, H, W]，如果为None则直接返回原始点云
            lidar2img: 雷达到图像的变换矩阵 [4, 4]
            intrinsics: 相机内参 [3, 3]
            img_shape: 图像尺寸 (H, W)
            
        Returns:
            hybrid_points: 混合点云 [M, C]
            foreground_mask: 前景点掩码 [N]
        """
        # 如果模块未启用或没有语义掩码，直接返回原始点云
        if not self.enabled or semantic_masks is None or len(semantic_masks) == 0:
            foreground_mask = torch.zeros(radar_points.shape[0], dtype=torch.bool, 
                                         device=radar_points.device)
            return radar_points, foreground_mask
        
        device = radar_points.device
        
        # 确保所有张量在同一设备上
        if semantic_masks.device != device:
            semantic_masks = semantic_masks.to(device)
        if lidar2img.device != device:
            lidar2img = lidar2img.to(device)
        if intrinsics.device != device:
            intrinsics = intrinsics.to(device)
        
        # 1. 将雷达点投影到图像平面
        projected_points = self.project_points_to_image(radar_points, lidar2img, img_shape)
        
        # 2. 检查点是否在掩码内
        valid_points = projected_points[:, 3] > 0  # 有效投影点
        points_uv = projected_points[:, :2]
        
        valid_matrix = self.is_within_mask(points_uv, semantic_masks, img_shape)
        valid_matrix = valid_matrix * valid_points.unsqueeze(1)  # 只考虑有效投影点
        
        # 3. 生成虚拟点
        virtual_points, foreground_mask = self.generate_virtual_points(
            semantic_masks, valid_matrix, projected_points, radar_points,
            intrinsics, lidar2img, img_shape
        )
        
        # 4. 组合混合点云
        if virtual_points is not None:
            hybrid_points = torch.cat([radar_points, virtual_points], dim=0)
        else:
            hybrid_points = radar_points
        
        return hybrid_points, foreground_mask


class RHGMWrapper(BaseModule):
    """
    RHGM包装器，用于批量处理和与RaCFormer集成
    
    # RHGM模块接入，来自HGSFusion
    # 原代码位置: HGSFusion/hybrid_pts/hybrid_radar_pts_vod.py
    """
    
    def __init__(self,
                 rhgm_cfg=None,
                 init_cfg=None):
        """
        初始化RHGM包装器
        
        Args:
            rhgm_cfg: RHGM模块配置
        """
        super(RHGMWrapper, self).__init__(init_cfg=init_cfg)
        
        if rhgm_cfg is None:
            rhgm_cfg = dict(
                num_virtual_points=100,
                dist_thresh=3000,
                gauss_sigma=7,
                gauss_kernel_size=51,
                gauss_uniform_ratio=[1, 4],
                input_channels=7,
                output_channels=7,
                enabled=True
            )
        
        self.rhgm = RHGM(**rhgm_cfg)
        self.enabled = rhgm_cfg.get('enabled', True)
    
    def forward(self, batch_radar_points, semantic_masks_list=None,
                lidar2img_list=None, intrinsics_list=None, img_shape=None):
        """
        批量处理雷达点云
        
        Args:
            batch_radar_points: 雷达点云列表 [B] x [N_i, C]
            semantic_masks_list: 语义掩码列表 [B] x [N_masks, H, W]
            lidar2img_list: 变换矩阵列表 [B] x [4, 4]
            intrinsics_list: 内参列表 [B] x [3, 3]
            img_shape: 图像尺寸 (H, W)
            
        Returns:
            hybrid_points_list: 混合点云列表 [B] x [M_i, C]
            foreground_masks_list: 前景掩码列表 [B] x [N_i]
        """
        if not self.enabled:
            # 如果模块未启用，直接返回原始点云
            foreground_masks = [
                torch.zeros(pts.shape[0], dtype=torch.bool, device=pts.device)
                for pts in batch_radar_points
            ]
            return batch_radar_points, foreground_masks
        
        hybrid_points_list = []
        foreground_masks_list = []
        
        batch_size = len(batch_radar_points)
        
        for i in range(batch_size):
            radar_points = batch_radar_points[i]
            
            # 获取当前样本的掩码和变换矩阵
            semantic_masks = semantic_masks_list[i] if semantic_masks_list is not None else None
            lidar2img = lidar2img_list[i] if lidar2img_list is not None else None
            intrinsics = intrinsics_list[i] if intrinsics_list is not None else None
            
            # 调用RHGM处理
            hybrid_points, foreground_mask = self.rhgm(
                radar_points,
                semantic_masks=semantic_masks,
                lidar2img=lidar2img,
                intrinsics=intrinsics,
                img_shape=img_shape
            )
            
            hybrid_points_list.append(hybrid_points)
            foreground_masks_list.append(foreground_mask)
        
        return hybrid_points_list, foreground_masks_list

