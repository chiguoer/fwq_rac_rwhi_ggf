# ==============================================================================
# RadarBEVNet模块
# 来源: rcbevdet-master/mmdet3d/models/backbones/radar_encoder.py
# 功能: 双流雷达骨干网络 + RCS-aware BEV编码器
# 替换位置: RaCFormer原有的pillar_encoder和middle_encoder
# ==============================================================================

import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
from mmcv.runner import BaseModule
from mmcv.cnn import build_norm_layer
from mmcv.cnn.bricks.transformer import MultiheadAttention
try:
    from timm.models.layers import DropPath, Mlp, to_2tuple
except ImportError:
    from timm.layers import DropPath, Mlp, to_2tuple


def get_paddings_indicator(actual_num, max_num, axis=0):
    """
    创建填充指示器掩码
    
    # 来自RCBEVDet的radar_encoder.py
    
    Args:
        actual_num: 每个voxel的实际点数 [N]
        max_num: 最大点数
        axis: 轴
        
    Returns:
        paddings_indicator: 填充指示器 [N, max_num]
    """
    actual_num = torch.unsqueeze(actual_num, axis + 1)
    max_num_shape = [1] * len(actual_num.shape)
    max_num_shape[axis + 1] = -1
    max_num = torch.arange(max_num, dtype=torch.int, device=actual_num.device).view(max_num_shape)
    paddings_indicator = actual_num.int() > max_num
    return paddings_indicator


class RFNLayer(nn.Module):
    """
    Radar Feature Network Layer
    
    # 替换为RCBEVDet的RadarBEVNet模块
    # 原代码位置: rcbevdet-master/mmdet3d/models/backbones/radar_encoder.py
    
    基础雷达特征网络层，类似于PFNLayer但针对雷达点云优化
    """
    
    def __init__(self, in_channels, out_channels, norm_cfg=None, last_layer=False):
        """
        初始化RFN层
        
        Args:
            in_channels: 输入通道数
            out_channels: 输出通道数
            norm_cfg: 归一化配置
            last_layer: 是否为最后一层
        """
        super().__init__()
        self.name = "RFNLayer"
        self.last_vfe = last_layer
        self.units = out_channels

        if norm_cfg is None:
            norm_cfg = dict(type="BN1d", eps=1e-3, momentum=0.01)
        self.norm_cfg = norm_cfg

        self.linear = nn.Linear(in_channels, self.units, bias=False)
        self.norm = build_norm_layer(self.norm_cfg, self.units)[1]

    def forward(self, inputs):
        """
        前向传播
        
        Args:
            inputs: 输入特征 [N, max_points, C]
            
        Returns:
            输出特征 [N, max_points, out_channels] 或 [N, 1, out_channels]
        """
        x = self.linear(inputs)
        torch.backends.cudnn.enabled = False
        x = self.norm(x.permute(0, 2, 1).contiguous()).permute(0, 2, 1).contiguous()
        torch.backends.cudnn.enabled = True
        x = F.relu(x)

        if self.last_vfe:
            x_max = torch.max(x, dim=1, keepdim=True)[0]
            return x_max
        else:
            return x


class PointEmbed(nn.Module):
    """
    点云嵌入模块
    
    # 来自RCBEVDet的RadarBEVNet
    # 用于将原始点云特征嵌入到高维空间
    """
    
    def __init__(self, in_c, out_c):
        super().__init__()
        self.in_c = in_c
        self.out_c = out_c
        self.conv1 = nn.Sequential(
            nn.Conv1d(in_c, out_c, 1),
            nn.BatchNorm1d(out_c),
            nn.ReLU(inplace=True),
            nn.Conv1d(out_c, out_c, 1)
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(out_c * 2, out_c * 2, 1),
            nn.BatchNorm1d(out_c * 2),
            nn.ReLU(inplace=True),
            nn.Conv1d(out_c * 2, out_c, 1)
        )

    def forward(self, points):
        """
        前向传播
        
        Args:
            points: 输入点云 [B, N, C]
            
        Returns:
            嵌入特征 [B, N, out_c]
        """
        bs, n, c = points.shape
        feature = self.conv1(points.transpose(2, 1))  # [bs, out_c, n]
        feature_global = torch.max(feature, dim=2, keepdim=True)[0]  # [bs, out_c, 1]
        
        feature = torch.cat([feature_global.expand(-1, -1, n), feature], dim=1)  # [bs, out_c*2, n]
        feature = self.conv2(feature)

        return feature.transpose(2, 1)  # [bs, n, out_c]


class CrossAttention(nn.Module):
    """
    交叉注意力模块
    
    # 来自RCBEVDet的RadarBEVNet
    # 用于双流特征交互
    """
    
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, c):
        """
        前向传播
        
        Args:
            x: 查询特征 [B, N, C]
            c: 上下文特征 [B, N, C]
            
        Returns:
            输出特征 [B, N, C]
        """
        B, N, C = x.shape
        kv = self.kv(c).reshape(B, N, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        k, v = kv.unbind(0)

        q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Extractor(nn.Module):
    """
    特征提取器模块
    
    # 来自RCBEVDet的RadarBEVNet
    # 从辅助流提取特征到主流
    """
    
    def __init__(self, dim, num_heads=6, n_points=4, n_levels=1, deform_ratio=1.0,
                 with_cffn=True, cffn_ratio=0.25, drop=0., drop_path=0.,
                 norm_layer=partial(nn.LayerNorm, eps=1e-6), with_cp=False):
        super().__init__()
        self.query_norm = norm_layer(dim)
        self.feat_norm = norm_layer(dim)
        self.attn = CrossAttention(dim, num_heads, qkv_bias=False, attn_drop=drop, proj_drop=drop)
        self.with_cffn = with_cffn
        self.with_cp = with_cp
        if with_cffn:
            self.ffn = Mlp(in_features=dim, hidden_features=int(dim * cffn_ratio), act_layer=nn.GELU, drop=drop)
            self.ffn_norm = norm_layer(dim)
            self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, query, feat):
        """
        前向传播
        
        Args:
            query: 查询特征 [B, N, C]
            feat: 上下文特征 [B, N, C]
            
        Returns:
            输出特征 [B, N, C]
        """
        attn = self.attn(self.query_norm(query), self.feat_norm(feat))
        query = query + attn
        
        if self.with_cffn:
            query = query + self.drop_path(self.ffn(self.ffn_norm(query)))
        return query


class Injector(nn.Module):
    """
    特征注入器模块
    
    # 来自RCBEVDet的RadarBEVNet
    # 向主流注入来自辅助流的特征
    """
    
    def __init__(self, dim, num_heads=6, n_points=4, n_levels=1, deform_ratio=1.0,
                 norm_layer=partial(nn.LayerNorm, eps=1e-6), init_values=0., with_cp=False, drop=0.):
        super().__init__()
        self.with_cp = with_cp
        self.query_norm = norm_layer(dim)
        self.feat_norm = norm_layer(dim)
        self.attn = CrossAttention(dim, num_heads, qkv_bias=False, attn_drop=drop, proj_drop=drop)
        self.gamma = nn.Parameter(init_values * torch.ones((dim)), requires_grad=True)

    def forward(self, query, feat):
        """
        前向传播
        
        Args:
            query: 查询特征 [B, N, C]
            feat: 上下文特征 [B, N, C]
            
        Returns:
            注入的特征增量 [B, N, C]
        """
        attn = self.attn(self.query_norm(query), self.feat_norm(feat))
        return self.gamma * attn


class DMSA(nn.Module):
    """
    Distance-aware Multi-head Self-Attention
    距离感知多头自注意力
    
    # 来自RCBEVDet的RadarBEVNet
    # 根据点云空间距离调节注意力权重
    """
    
    def __init__(self, embed_dims=256, num_heads=8, dropout=0.1):
        super().__init__()
        self.attention = MultiheadAttention(embed_dims, num_heads, dropout, batch_first=True)
        self.beta = nn.Linear(embed_dims, num_heads)

    @torch.no_grad()
    def init_weights(self):
        nn.init.zeros_(self.beta.weight)
        nn.init.uniform_(self.beta.bias, 0.0, 2.0)

    def inner_forward(self, query_bbox, query_feat, pre_attn_mask):
        dist = self.center_dists(query_bbox)
        beta = self.beta(query_feat)

        beta = beta.permute(0, 2, 1)
        attn_mask = dist[:, None, :, :] * beta[..., None]
        if pre_attn_mask is not None:
            attn_mask[:, :, pre_attn_mask] = float('-inf')
        attn_mask = attn_mask.flatten(0, 1)
        return self.attention(query_feat, attn_mask=attn_mask)

    def forward(self, query_bbox, query_feat, pre_attn_mask=None):
        return self.inner_forward(query_bbox, query_feat, pre_attn_mask)

    @torch.no_grad()
    def center_dists(self, points):
        centers = points[..., :2]
        dist = []
        for b in range(centers.shape[0]):
            dist_b = torch.norm(centers[b].reshape(-1, 1, 2) - centers[b].reshape(1, -1, 2), dim=-1)
            dist.append(dist_b[None, ...])

        dist = torch.cat(dist, dim=0)
        dist = -dist

        return dist


class SelfAttentionBlock(nn.Module):
    """
    自注意力块
    
    # 来自RCBEVDet的RadarBEVNet
    # 使用DMSA进行自注意力计算
    """
    
    def __init__(self, dim, num_heads=6, n_points=4, n_levels=1, deform_ratio=1.0,
                 with_cffn=True, cffn_ratio=0.25, drop=0., drop_path=0.,
                 norm_layer=partial(nn.LayerNorm, eps=1e-6), with_cp=False):
        super().__init__()
        self.query_norm = norm_layer(dim)
        self.attn = DMSA(dim, num_heads, dropout=drop)
        self.with_cffn = with_cffn
        self.with_cp = with_cp
        if with_cffn:
            self.ffn = Mlp(in_features=dim, hidden_features=int(dim * 2), act_layer=nn.GELU, drop=drop)
            self.ffn_norm = norm_layer(dim)
            self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, feat, points):
        """
        前向传播
        
        Args:
            feat: 输入特征 [B, N, C]
            points: 点坐标 [B, N, 3]
            
        Returns:
            输出特征 [B, N, C]
        """
        identity = feat
        feat = self.query_norm(feat)
        feat = self.attn(points, feat)
        feat = feat + identity

        if self.with_cffn:
            feat = feat + self.drop_path(self.ffn(self.ffn_norm(feat)))
        return feat


class RadarBEVNet(BaseModule):
    """
    RadarBEVNet - 双流雷达骨干网络
    
    # 替换为RCBEVDet的RadarBEVNet模块
    # 原代码位置: rcbevdet-master/mmdet3d/models/backbones/radar_encoder.py
    # 替换RaCFormer原有的: radar_voxel_encoder + radar_middle_encoder
    
    特点:
    1. 双流架构: 主流(点特征) + 辅助流(上下文嵌入)
    2. 注入-提取交互: Injector和Extractor模块
    3. 距离感知自注意力: DMSA模块
    4. RCS-aware编码: 保留RCS信息用于后续处理
    """
    
    def __init__(
        self,
        in_channels=7,
        feat_channels=(64,),
        with_distance=False,
        voxel_size=(0.5, 0.5, 8),
        point_cloud_range=(-51.2, -51.2, -5.0, 51.2, 51.2, 3.0),
        norm_cfg=None,
        with_pos_embed=False,
        return_rcs=False,
        drop=0.0,
        init_cfg=None,
    ):
        """
        初始化RadarBEVNet
        
        Args:
            in_channels: 输入点云通道数 (默认7: x,y,z,rcs,vr,vr_comp,time)
            feat_channels: 特征通道列表
            with_distance: 是否添加距离特征
            voxel_size: 体素大小
            point_cloud_range: 点云范围
            norm_cfg: 归一化配置
            with_pos_embed: 是否使用位置编码
            return_rcs: 是否返回RCS特征
            drop: Dropout比率
        """
        super().__init__(init_cfg=init_cfg)
        
        self.return_rcs = return_rcs
        assert len(feat_channels) > 0

        self.in_channels = in_channels
        in_channels_with_offset = in_channels + 2  # 添加中心偏移
        self._with_distance = with_distance

        # 创建RFN层
        feat_channels = [in_channels_with_offset] + list(feat_channels)
        point_block = []
        for i in range(len(feat_channels) - 1):
            in_filters = feat_channels[i]
            out_filters = feat_channels[i + 1]
            last_layer = False
            point_block.append(
                RFNLayer(in_filters, out_filters, norm_cfg=norm_cfg, last_layer=last_layer)
            )
        self.point_block = nn.ModuleList(point_block)

        # 创建Extractor模块
        num_heads = 2
        extractor = []
        for i in range(1, len(feat_channels)):
            extractor.append(
                Extractor(feat_channels[i], num_heads=num_heads, cffn_ratio=1, drop=drop, drop_path=drop)
            )
        self.extractor = nn.ModuleList(extractor)

        # 创建Injector模块
        injector = []
        for i in range(1, len(feat_channels)):
            injector.append(
                Injector(feat_channels[i], num_heads=num_heads, drop=drop)
            )
        self.injector = nn.ModuleList(injector)

        # 创建Transformer块
        transformer_block = []
        for i in range(1, len(feat_channels)):
            transformer_block.append(
                SelfAttentionBlock(feat_channels[i], num_heads=num_heads, cffn_ratio=1, drop=drop, drop_path=drop)
            )
        self.transformer_block = nn.ModuleList(transformer_block)

        # 线性变换模块
        linear_module = []
        for i in range(1, len(feat_channels) - 1):
            linear_module.append(
                nn.Linear(feat_channels[i], feat_channels[i + 1])
            )
        self.linear_module = nn.ModuleList(linear_module)

        # 输出线性层
        self.out_linear = nn.Linear(feat_channels[-1] * 2, feat_channels[-1])

        # 体素参数
        self.vx = voxel_size[0]
        self.vy = voxel_size[1]
        self.x_offset = self.vx / 2 + point_cloud_range[0]
        self.y_offset = self.vy / 2 + point_cloud_range[1]
        self.pc_range = point_cloud_range

        # 位置编码
        if with_pos_embed:
            embed_dims = feat_channels[1]
            self.pos_embed = nn.Sequential(
                nn.Linear(3, embed_dims),
                nn.LayerNorm(embed_dims),
                nn.ReLU(inplace=True),
                nn.Linear(embed_dims, embed_dims),
                nn.LayerNorm(embed_dims),
                nn.ReLU(inplace=True),
            )
        self.with_pos_embed = with_pos_embed

        # 点云嵌入
        self.point_embed = PointEmbed(in_channels_with_offset + 2, feat_channels[1])
        
        # 输出通道数
        self.out_channels = feat_channels[-1]

    def compress(self, x):
        """压缩特征 (取max)"""
        x = x.max(dim=1)[0]
        x = x.unsqueeze(dim=0)
        return x

    def forward(self, features, num_voxels, coors):
        """
        前向传播
        
        Args:
            features: 体素特征 [N_voxels, max_points, in_channels]
            num_voxels: 每个体素的点数 [N_voxels]
            coors: 体素坐标 [N_voxels, 3] (batch_idx, y_idx, x_idx)
            
        Returns:
            output: 体素特征 [N_voxels, out_channels]
            rcs (可选): RCS特征 [N_voxels, in_channels]
        """
        dtype = features.dtype
        
        # 计算中心偏移
        f_center = torch.zeros_like(features[:, :, :2])
        f_center[:, :, 0] = features[:, :, 0] - (
            coors[:, 2].to(dtype).unsqueeze(1) * self.vx + self.x_offset
        )
        f_center[:, :, 1] = features[:, :, 1] - (
            coors[:, 1].to(dtype).unsqueeze(1) * self.vy + self.y_offset
        )

        # 归一化坐标到[0, 1]
        features[:, :, 0:1] = (features[:, :, 0:1] - self.pc_range[0]) / (self.pc_range[3] - self.pc_range[0])
        features[:, :, 1:2] = (features[:, :, 1:2] - self.pc_range[1]) / (self.pc_range[4] - self.pc_range[1])
        features[:, :, 2:3] = (features[:, :, 2:3] - self.pc_range[2]) / (self.pc_range[5] - self.pc_range[2])

        voxel_count = features.shape[1]
        mask = get_paddings_indicator(num_voxels, voxel_count, axis=0)
        mask = torch.unsqueeze(mask, -1).type_as(features)

        # 计算均值偏移
        features_mean = torch.zeros_like(features[:, :, :2])
        mask_sum = mask.squeeze().sum(dim=1, keepdim=True).clamp(min=1)
        features_mean[:, :, 0] = features[:, :, 0] - ((features[:, :, 0] * mask.squeeze()).sum(dim=1, keepdim=True) / mask_sum)
        features_mean[:, :, 1] = features[:, :, 1] - ((features[:, :, 1] * mask.squeeze()).sum(dim=1, keepdim=True) / mask_sum)

        rcs_features = features.clone()
        
        # 构建输入特征
        c = torch.cat([features, features_mean, f_center], dim=-1)
        x = torch.cat([features, f_center], dim=-1)

        # 应用掩码
        x *= mask
        c *= mask

        # 点云嵌入
        c = self.point_embed(c)
        if self.with_pos_embed:
            c = c + self.pos_embed(features[:, :, 0:3])
        points_coors = features[:, :, 0:3].detach()

        # 获取batch信息
        batch_size = coors[-1, 0] + 1
        if batch_size > 1:
            bs_list = [0]
            bs_info = coors[:, 0]
            pre = bs_info[0]
            for i in range(1, len(bs_info)):
                if pre != bs_info[i]:
                    bs_list.append(i)
                    pre = bs_info[i]
            bs_list.append(len(bs_info))
            bs_list = [bs_list[i + 1] - bs_list[i] for i in range(len(bs_list) - 1)]
        elif batch_size == 1:
            bs_list = [len(coors[:, 0])]
        else:
            bs_list = [len(coors[:, 0])]

        points_coors_split = torch.split(points_coors, bs_list)

        # 双流处理
        i = 0
        for rfn in self.point_block:
            x = rfn(x)
            x_split = torch.split(x, bs_list)
            c_split = torch.split(c, bs_list)

            x_out_list = []
            c_out_list = []
            for bs in range(len(x_split)):
                c_tmp = c_split[bs]
                x_tmp = x_split[bs]
                points_coors_tmp = points_coors_split[bs]
                
                # 注入-提取交互
                c_tmp = c_tmp + self.injector[i](self.compress(c_tmp), self.compress(x_tmp)).transpose(1, 0).expand_as(c_tmp)
                x_tmp = x_tmp + self.extractor[i](self.compress(x_tmp), self.compress(c_tmp)).transpose(1, 0).expand_as(x_tmp)
                
                # 自注意力
                c_tmp = self.transformer_block[i](self.compress(c_tmp), self.compress(points_coors_tmp)).transpose(1, 0).expand_as(c_tmp)
                
                # 维度变换
                if i < len(self.point_block) - 1:
                    c_tmp = self.linear_module[i](c_tmp)

                c_out_list.append(c_tmp)
                x_out_list.append(x_tmp)

            x = torch.cat(x_out_list, dim=0)
            c = torch.cat(c_out_list, dim=0)
            i += 1

        # 融合双流特征
        c = self.out_linear(torch.cat([c, x], dim=-1))

        # Max pooling
        c = torch.max(c, dim=1, keepdim=True)[0]
        
        if not self.return_rcs:
            return c.squeeze()
        else:
            rcs = (rcs_features * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
            return c.squeeze(), rcs.squeeze()


class RadarBEVNetWrapper(BaseModule):
    """
    RadarBEVNet包装器
    
    # 用于与RaCFormer集成的包装器类
    # 处理维度适配和BEV特征生成
    """
    
    def __init__(
        self,
        radar_bev_net_cfg=None,
        out_channels=256,
        bev_h=128,
        bev_w=128,
        init_cfg=None,
    ):
        """
        初始化RadarBEVNet包装器
        
        Args:
            radar_bev_net_cfg: RadarBEVNet配置
            out_channels: 输出通道数 (需要与RaCFormer的embed_dims匹配)
            bev_h: BEV特征图高度
            bev_w: BEV特征图宽度
        """
        super().__init__(init_cfg=init_cfg)
        
        if radar_bev_net_cfg is None:
            radar_bev_net_cfg = dict(
                in_channels=7,
                feat_channels=(64,),
                with_distance=False,
                voxel_size=(0.5, 0.5, 8),
                point_cloud_range=(-51.2, -51.2, -5.0, 51.2, 51.2, 3.0),
                norm_cfg=None,
                with_pos_embed=False,
                return_rcs=False,
                drop=0.0,
            )
        
        self.radar_bev_net = RadarBEVNet(**radar_bev_net_cfg)
        self.bev_h = bev_h
        self.bev_w = bev_w
        self.out_channels = out_channels
        
        # # 新增维度转换层适配RaCFormer期望的输出维度
        # # RadarBEVNet输出: [N_voxels, 64]
        # # RaCFormer期望: [B, 256, H, W]
        # radar_out_channels = radar_bev_net_cfg.get('feat_channels', (64,))[-1]
        # if radar_out_channels != out_channels:
        #     self.channel_adapter = nn.Sequential(
        #         nn.Linear(radar_out_channels, out_channels),
        #         nn.LayerNorm(out_channels),
        #         nn.ReLU(inplace=True)
        #     )
        # else:
        #     self.channel_adapter = None
    
    def forward(self, voxel_features, num_points, coors, batch_size):
        """
        前向传播
        
        Args:
            voxel_features: 体素特征 [N_voxels, max_points, in_channels]
            num_points: 每个体素的点数 [N_voxels]
            coors: 体素坐标 [N_voxels, 4] (batch_idx, z_idx, y_idx, x_idx)
            batch_size: batch大小
            
        Returns:
            bev_features: BEV特征 [B, C, H, W]
        """
        # 调用RadarBEVNet
        # 注意: RadarBEVNet期望的coors格式是 [N, 3] (batch_idx, y_idx, x_idx)
        # 需要处理坐标格式
        if coors.shape[1] == 4:
            # 从 (batch, z, y, x) 转换为 (batch, y, x)
            coors_3d = torch.stack([coors[:, 0], coors[:, 2], coors[:, 3]], dim=1)
        else:
            coors_3d = coors
        
        # 获取体素特征
        voxel_feats = self.radar_bev_net(voxel_features, num_points, coors_3d)
        
        # # 维度适配
        # if self.channel_adapter is not None:
        #     voxel_feats = self.channel_adapter(voxel_feats)
        
        return voxel_feats

