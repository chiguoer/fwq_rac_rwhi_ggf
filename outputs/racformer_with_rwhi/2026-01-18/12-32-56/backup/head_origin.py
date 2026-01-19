import math

import torch

import torch.nn as nn

import torch.nn.functional as F

from mmcv.runner import force_fp32

from mmdet.core import multi_apply, reduce_mean

from mmdet.models import HEADS

from mmdet.models.dense_heads import DETRHead

from mmdet3d.core.bbox.coders import build_bbox_coder

from mmdet3d.core.bbox.structures.lidar_box3d import LiDARInstance3DBoxes

from .bbox.utils import normalize_bbox, encode_bbox, theta_d2xy_coods, xy2theta_d_coods, compute_map_size_and_radius

from .utils import VERSION

from .rwhi import RWHIModule



@HEADS.register_module()

class RaCFormer_head(DETRHead):

    """

    RaCFormer检测头

    

    支持两种Query初始化策略:

    1. 原始策略: 均匀极坐标分布 (use_rwhi=False)

    2. RWHI策略: RCS加权混合锚点初始化 (use_rwhi=True)

    

    ============================================================

    [重要修复] 动态内容生成 (Dynamic Content Generation)

    ============================================================

    当使用RWHI时，query位置是动态的（每帧基于雷达点云生成）。

    如果仍使用静态的nn.Embedding作为query内容特征，会导致：

    - 梯度冲突：模型试图为随机跳动的query学习固定特征

    - 训练崩溃：loss在epoch 7左右爆炸（从~23到>1000）

    

    解决方案：使用pos2content MLP将动态位置转换为动态内容特征。

    ============================================================

    """

    

    # def __init__(self,

    #              *args,

    #              num_classes,

    #              in_channels,

    #              query_denoising=True,

    #              query_denoising_groups=10,

    #              num_clusters=5,

    #              bbox_coder=None,

    #              code_size=10,

    #              code_weights=[1.0] * 10,

    #              train_cfg=dict(),

    #              test_cfg=dict(max_per_img=100),

    #              # ============================================================

    #              # RWHI (RCS-Weighted Hybrid Anchor Initialization) 配置

    #              # ============================================================

    #              use_rwhi=False,  # 是否启用RWHI Query初始化

    #              rwhi_cfg=None,   # RWHI模块配置

    #              **kwargs):

    #     self.code_size = code_size

    #     self.code_weights = code_weights

    #     self.num_classes = num_classes

    #     self.in_channels = in_channels

    #     self.train_cfg = train_cfg

    #     self.test_cfg = test_cfg

    #     self.fp16_enabled = False

    #     self.embed_dims = in_channels



    #     self.num_clusters = num_clusters

        

    #     # RWHI配置 (需要在super().__init__之前设置)

    #     self.use_rwhi = use_rwhi

    #     self.rwhi_cfg = rwhi_cfg



    #     super(RaCFormer_head, self).__init__(num_classes, in_channels, train_cfg=train_cfg, test_cfg=test_cfg, **kwargs)



    #     self.code_weights = nn.Parameter(torch.tensor(self.code_weights), requires_grad=False)

    #     self.bbox_coder = build_bbox_coder(bbox_coder)

    #     self.pc_range = self.bbox_coder.pc_range



    #     self.dn_enabled = query_denoising

    #     self.dn_group_num = query_denoising_groups

    #     self.dn_weight = 1.0

    #     self.dn_bbox_noise_scale = 0.5

    #     self.dn_label_noise_scale = 0.5



    def __init__(self,

             *args,

             num_classes,

             in_channels,

             query_denoising=True,

             query_denoising_groups=10,

             num_clusters=5,

             bbox_coder=None,

             code_size=10,

             code_weights=[1.0] * 10,

             train_cfg=dict(),

             test_cfg=dict(max_per_img=100),

             use_rwhi=False,

             rwhi_cfg=None,

             **kwargs):

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



        # 关键修复：从 kwargs 中提取 pc_range（如果存在）

        self.pc_range = kwargs.pop('pc_range', [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0])



        super(RaCFormer_head, self).__init__(num_classes, in_channels, train_cfg=train_cfg, test_cfg=test_cfg, **kwargs)



        self.code_weights = nn.Parameter(torch.tensor(self.code_weights), requires_grad=False)

        self.bbox_coder = build_bbox_coder(bbox_coder)

        

        # 验证 bbox_coder 中的 pc_range

        if hasattr(self.bbox_coder, 'pc_range'):

            bbox_pc_range = self.bbox_coder.pc_range

            # 可以选择使用 bbox_coder 的 pc_range

            self.pc_range = bbox_pc_range



        self.dn_enabled = query_denoising

        self.dn_group_num = query_denoising_groups

        self.dn_weight = 1.0

        self.dn_bbox_noise_scale = 0.5

        self.dn_label_noise_scale = 0.5



    def _init_layers(self):

        """

        初始化网络层

        

        支持两种Query初始化模式:

        1. 原始模式 (use_rwhi=False): 使用nn.Embedding + generate_points()

        2. RWHI模式 (use_rwhi=True): 使用RWHIModule动态生成锚点 + pos2content MLP

        

        ============================================================

        [关键修复] 动态Query需要动态内容特征

        ============================================================

        当use_rwhi=True时，query_bbox是动态的（每帧不同）。

        如果使用静态的label_enc embedding作为query_feat，会导致：

        - 模型试图为"随机跳动的Embedding #5"学习"它是一辆车"

        - 梯度冲突 -> 训练崩溃 (epoch 7, loss从23爆炸到1000+)

        

        解决方案：使用pos2content MLP从位置生成内容特征。

        ============================================================

        """

        self.label_enc = nn.Embedding(self.num_classes + 1, self.embed_dims - 1)  # DAB-DETR

        

        if self.use_rwhi:

            # ============================================================

            # RWHI模式: 使用RCS加权混合锚点初始化

            # ============================================================

            # ============================================================

            # RWHI v2.0 默认配置 - 叠加策略

            # ============================================================

            rwhi_default_cfg = dict(

                num_query=self.num_query,

                # v2.0 新参数: 锚点预算分配

                num_base=500,                    # 基础锚点 (静态安全网，覆盖全图)

                num_radar=400,                   # 雷达锚点 (动态增益)

                # v2.0 新参数: 空间范围

                max_range=55.0,                  # 最大感知范围 (米) - 覆盖整个BEV!

                min_range=2.0,                   # 最小感知范围 (米)

                rcs_threshold=0.0,               # RCS过滤阈值

                # 共用参数

                embed_dims=self.embed_dims,

                pc_range=self.pc_range,

                bev_grid_size=100,

                velocity_alpha=0.5,

                height_hypotheses=(0.0, 1.0),

                diffusion_kernel_size=3,

                noise_eps=1e-6,

                epsilon_floor=0.02,

                enabled=True,

            )

            

            # 合并用户配置

            if self.rwhi_cfg is not None:

                rwhi_default_cfg.update(self.rwhi_cfg)

            

            # 确保关键参数正确

            rwhi_default_cfg['num_query'] = self.num_query

            rwhi_default_cfg['embed_dims'] = self.embed_dims

            rwhi_default_cfg['pc_range'] = self.pc_range

            

            self.rwhi_module = RWHIModule(**rwhi_default_cfg)

            

            # ============================================================

            # [核心修复] Position-to-Content MLP

            # 将动态位置 (theta, d, z) 转换为动态内容特征

            # 输出维度是 embed_dims - 1，因为后续要拼接1维的indicator

            # ============================================================

            self.pos2content = nn.Sequential(

                nn.Linear(3, self.embed_dims),

                nn.LayerNorm(self.embed_dims),

                nn.ReLU(inplace=True),

                nn.Linear(self.embed_dims, self.embed_dims - 1),  # -1 for indicator

            )

            # 初始化权重 (Xavier for stable training)

            for module in self.pos2content:

                if isinstance(module, nn.Linear):

                    nn.init.xavier_uniform_(module.weight)

                    nn.init.zeros_(module.bias)

            

            # 仍然需要init_query_bbox作为后备 (当没有雷达点时)

            self.init_query_bbox = nn.Embedding(self.num_query, 10)

            nn.init.constant_(self.init_query_bbox.weight[:, 2:3], 0.5)

            nn.init.zeros_(self.init_query_bbox.weight[:, 8:10])

            nn.init.constant_(self.init_query_bbox.weight[:, 5:6], 0.2)

            

            # 使用RWHI的安全流锚点初始化embedding

            with torch.no_grad():

                safety_anchors = self.rwhi_module.safety_anchors  # [num_safety, 10]

                num_safety = safety_anchors.shape[0]

                # 用安全流锚点填充embedding的前部分

                self.init_query_bbox.weight[:num_safety, :] = safety_anchors

                # 后续部分使用generate_points作为后备

                if num_safety < self.num_query:

                    fallback_theta_d = self.generate_points()

                    fallback_num = min(self.num_query - num_safety, fallback_theta_d.shape[0])

                    self.init_query_bbox.weight[num_safety:num_safety + fallback_num, :2] = \

                        fallback_theta_d[:fallback_num].reshape(-1, 2)

        else:

            # ============================================================

            # 原始模式: 均匀极坐标分布 (静态embedding)

            # ============================================================

            self.init_query_bbox = nn.Embedding(self.num_query, 10)  # (x, y, z, w, l, h, sin, cos, vx, vy)

            

            nn.init.constant_(self.init_query_bbox.weight[:, 2:3], 0.5)

            nn.init.zeros_(self.init_query_bbox.weight[:, 8:10])

            nn.init.constant_(self.init_query_bbox.weight[:, 5:6], 0.2)



            theta_d = self.generate_points()

            with torch.no_grad():

                self.init_query_bbox.weight[:, :2] = theta_d.reshape(-1, 2)  # [Q, 2]

            

            self.rwhi_module = None

            self.pos2content = None  # 原始模式不需要pos2content



    def init_weights(self):

        self.transformer.init_weights()





    def generate_points(self):

        """

        生成均匀分布的极坐标锚点 (原始RaCFormer策略)

        

        Returns:

            theta_d: [num_query, 2] (theta, d) 归一化极坐标

        """

        num_angles = self.num_query//self.num_clusters

        angles = torch.linspace(0, 1, num_angles+1)[:-1]

        distances = torch.linspace(0, 1, self.num_clusters + 2,  dtype=torch.float)[1:-1]



        angles = angles.view(num_angles, 1).expand(num_angles, self.num_clusters)

        distances = distances.view(1, self.num_clusters).expand(num_angles, self.num_clusters)



        theta_d = torch.cat([angles[..., None], distances[..., None]], dim=-1).flatten(0,1)

        return theta_d

    

    def _extract_radar_points_from_metas(self, img_metas, batch_size):

        """

        从img_metas中提取雷达点云数据 (用于RWHI)

        

        Args:

            img_metas: 图像元信息列表

            batch_size: batch大小

            

        Returns:

            radar_points: [B, M, C] 雷达点云，如果不可用则返回None

        """

        # 尝试从img_metas获取雷达点云

        # 注意: 这需要数据加载器在img_metas中包含radar_points

        try:

            radar_points_list = []

            for meta in img_metas:

                if 'radar_points' in meta:

                    pts = meta['radar_points']

                    if isinstance(pts, torch.Tensor):

                        radar_points_list.append(pts)

                    elif isinstance(pts, list) and len(pts) > 0:

                        # 如果是多帧数据，取第一帧

                        pts_first = pts[0] if isinstance(pts[0], torch.Tensor) else None

                        if pts_first is not None:

                            radar_points_list.append(pts_first)

            

            if len(radar_points_list) == batch_size:

                # 需要统一点数 (取最小点数或padding)

                max_points = max(p.shape[0] for p in radar_points_list)

                padded_points = []

                for pts in radar_points_list:

                    if pts.shape[0] < max_points:

                        # Zero-padding

                        padding = torch.zeros(

                            max_points - pts.shape[0], pts.shape[1],

                            dtype=pts.dtype, device=pts.device

                        )

                        pts = torch.cat([pts, padding], dim=0)

                    padded_points.append(pts)

                

                radar_points = torch.stack(padded_points, dim=0)  # [B, M, C]

                return radar_points

        except Exception:

            pass

        

        return None





    def forward(self, mlvl_feats, lss_bev_feats, radar_bev_feats, img_metas, radar_points=None):

        """

        前向传播

        

        Args:

            mlvl_feats: 多尺度图像特征

            lss_bev_feats: LSS BEV特征

            radar_bev_feats: 雷达BEV特征

            img_metas: 图像元信息

            radar_points: 原始雷达点云 [B, M, C]，用于RWHI初始化

        

        Returns:

            dict: 包含分类和回归预测的字典

        """

        B = lss_bev_feats.shape[0]

        device = lss_bev_feats.device

        

        # ============================================================

        # Query初始化: RWHI模式 vs 原始模式

        # ============================================================

        # 标记是否使用了动态RWHI (用于决定是否生成动态content)

        using_dynamic_rwhi = False

        

        if self.use_rwhi and self.rwhi_module is not None:

            # 尝试从img_metas获取雷达点云 (如果未显式传入)

            if radar_points is None:

                radar_points = self._extract_radar_points_from_metas(img_metas, B)

            

            if radar_points is not None:

                # ============================================================

                # 修复：确保雷达点云的batch size与B一致

                # 如果radar_points的batch size与B不匹配，进行调整

                # ============================================================

                radar_batch_size = radar_points.shape[0]

                

                if radar_batch_size != B:

                    # 情况1: 雷达点云是多帧数据拼接的，需要取对应帧

                    if radar_batch_size > B:

                        # 假设多帧数据是按帧拼接的，取每隔(radar_batch_size//B)帧的数据

                        # 或者简单取前B个样本

                        radar_points = radar_points[:B]

                    else:

                        # 情况2: 雷达点云少于batch size，复制填充

                        repeat_times = (B + radar_batch_size - 1) // radar_batch_size

                        radar_points = radar_points.repeat(repeat_times, 1, 1)[:B]

                

                # RWHI模式: 使用雷达点云动态生成锚点

                query_bbox, _ = self.rwhi_module(radar_points)  # [B, Q, 10]

                using_dynamic_rwhi = True  # 标记使用了动态RWHI

                

                # 再次确保batch size一致

                if query_bbox.shape[0] != B:

                    if query_bbox.shape[0] > B:

                        query_bbox = query_bbox[:B]

                    else:

                        repeat_times = (B + query_bbox.shape[0] - 1) // query_bbox.shape[0]

                        query_bbox = query_bbox.repeat(repeat_times, 1, 1)[:B]

            else:

                # 后备: 使用预计算的embedding

                query_bbox = self.init_query_bbox.weight.clone()  # [Q, 10]

                query_bbox = query_bbox.view(1, self.num_query, 10).repeat(B, 1, 1)

        else:

            # 原始模式: 使用固定的embedding

            query_bbox = self.init_query_bbox.weight.clone()  # [Q, 10]

            query_bbox = query_bbox.view(1, self.num_query, 10).repeat(B, 1, 1)

        

        # 确保query_bbox在正确的设备上

        query_bbox = query_bbox.to(device)

        

        # ============================================================

        # DEBUG: 验证query_bbox坐标范围 (可在稳定后移除)

        # Transformer期望 (theta, d, z) 都在 [0, 1] 范围内

        # ============================================================

        if self.training:

            theta_vals = query_bbox[..., 0]

            d_vals = query_bbox[..., 1]

            z_vals = query_bbox[..., 2]

            # 只在范围异常时打印警告

            if theta_vals.min() < 0 or theta_vals.max() > 1:

                print(f"[WARNING RWHI] theta out of [0,1]: min={theta_vals.min().item():.4f}, max={theta_vals.max().item():.4f}")

            if d_vals.min() < 0 or d_vals.max() > 1:

                print(f"[WARNING RWHI] d out of [0,1]: min={d_vals.min().item():.4f}, max={d_vals.max().item():.4f}")

            if z_vals.min() < 0 or z_vals.max() > 1:

                print(f"[WARNING RWHI] z out of [0,1]: min={z_vals.min().item():.4f}, max={z_vals.max().item():.4f}")

        

        # ============================================================

        # 关键修复: 强制clamp所有坐标到[0, 1]

        # 确保即使RWHI输出有轻微超范围，也不会导致Transformer失效

        # ============================================================

        query_bbox = query_bbox.clone()

        query_bbox[..., 0] = torch.clamp(query_bbox[..., 0], 0.0, 1.0)  # theta

        query_bbox[..., 1] = torch.clamp(query_bbox[..., 1], 0.0, 1.0)  # d

        query_bbox[..., 2] = torch.clamp(query_bbox[..., 2], 0.0, 1.0)  # z



        # ============================================================

        # [核心修复] 动态内容生成 (Dynamic Content Generation)

        # ============================================================

        # 当使用动态RWHI时，query位置每帧都不同，

        # 必须使用pos2content MLP从位置生成内容特征，

        # 而不是使用静态的label_enc embedding。

        # 这是防止训练崩溃的关键！

        # ============================================================

        if using_dynamic_rwhi and self.pos2content is not None:

            # 动态模式: 从位置生成内容特征

            query_pos = query_bbox[..., :3]  # [B, Q, 3] (theta, d, z)

            dynamic_content = self.pos2content(query_pos)  # [B, Q, embed_dims-1]

            

            # 添加indicator (0表示非DN query)

            indicator0 = torch.zeros(B, self.num_query, 1, device=device)

            init_query_feat = torch.cat([dynamic_content, indicator0], dim=-1)  # [B, Q, embed_dims]

            

            # ============================================================

            # [DDP兼容修复] 确保 init_query_bbox 参与梯度计算

            # ============================================================

            # 问题: 在RWHI模式下，init_query_bbox (后备embedding) 不被使用，

            #       导致DDP报错: "Parameter indices which did not receive grad"

            # 解决: 添加一个零梯度的虚拟正则化项，使其参与计算图

            #       乘以0确保不影响实际输出

            # ============================================================

            if self.training:

                dummy_regularizer = self.init_query_bbox.weight.sum() * 0.0

                # 将虚拟正则化项加到query_bbox的某个不影响输出的位置

                query_bbox = query_bbox + dummy_regularizer

        else:

            # 静态模式: 使用原始的label_enc embedding

            # 这是原始RaCFormer的行为，保持向后兼容

            init_query_feat = None  # 让prepare_for_dn_input使用默认逻辑



        query_bbox, query_feat, attn_mask, mask_dict = self.prepare_for_dn_input(

            B, query_bbox, self.label_enc, img_metas, init_query_feat=init_query_feat

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



        bbox_preds[..., 0] = bbox_preds[..., 0] * (self.pc_range[3] - self.pc_range[0]) + self.pc_range[0]

        bbox_preds[..., 1] = bbox_preds[..., 1] * (self.pc_range[4] - self.pc_range[1]) + self.pc_range[1]

        bbox_preds[..., 2] = bbox_preds[..., 2] * (self.pc_range[5] - self.pc_range[2]) + self.pc_range[2]



        bbox_preds = torch.cat([

            bbox_preds[..., 0:2],

            bbox_preds[..., 3:5],

            bbox_preds[..., 2:3],

            bbox_preds[..., 5:10],

        ], dim=-1)  # [cx, cy, w, l, cz, h, sin, cos, vx, vy]



        if mask_dict is not None and mask_dict['pad_size'] > 0:  # if using query denoising

            output_known_cls_scores = cls_scores[:, :, :mask_dict['pad_size'], :]

            output_known_bbox_preds = bbox_preds[:, :, :mask_dict['pad_size'], :]

            output_cls_scores = cls_scores[:, :, mask_dict['pad_size']:, :]

            output_bbox_preds = bbox_preds[:, :, mask_dict['pad_size']:, :]

            mask_dict['output_known_lbs_bboxes'] = (output_known_cls_scores, output_known_bbox_preds)

            outs = {

                'all_cls_scores': output_cls_scores,

                'all_bbox_preds': output_bbox_preds,

                'enc_cls_scores': None,

                'enc_bbox_preds': None, 

                'dn_mask_dict': mask_dict,

            }

        else:

            outs = {

                'all_cls_scores': cls_scores,

                'all_bbox_preds': bbox_preds,

                'enc_cls_scores': None,

                'enc_bbox_preds': None, 

            }



        return outs



    def prepare_for_dn_input(self, batch_size, init_query_bbox, label_enc, img_metas, init_query_feat=None):

        """

        准备Denoising输入

        

        Args:

            batch_size: batch大小

            init_query_bbox: 初始query位置 [B, Q, 10]

            label_enc: 标签编码器

            img_metas: 图像元信息

            init_query_feat: 可选的预计算query特征 [B, Q, embed_dims]

                            如果为None，使用静态的label_enc embedding (原始行为)

                            如果提供，使用动态生成的特征 (RWHI模式)

        

        ============================================================

        [关键修复说明]

        当使用RWHI时，init_query_feat由pos2content MLP动态生成，

        而不是使用静态的label_enc.weight[num_classes]。

        这解决了"动态位置+静态内容"导致的训练崩溃问题。

        ============================================================

        

        References:

        - https://github.com/IDEA-Research/DN-DETR/blob/main/models/DN_DAB_DETR/dn_components.py

        - https://github.com/megvii-research/PETR/blob/main/projects/mmdet3d_plugin/models/dense_heads/petrv2_dnhead.py

        """

        device = init_query_bbox.device

        

        # ============================================================

        # Query内容特征初始化

        # ============================================================

        if init_query_feat is None:

            # 静态模式 (原始RaCFormer行为): 使用label_enc的背景类embedding

            indicator0 = torch.zeros([self.num_query, 1], device=device)

            init_query_feat = label_enc.weight[self.num_classes].repeat(self.num_query, 1)

            init_query_feat = torch.cat([init_query_feat, indicator0], dim=1).repeat(batch_size, 1, 1)

        # else: 使用传入的动态init_query_feat (由pos2content生成)



        if self.training and self.dn_enabled:

            targets = [{

                'bboxes': torch.cat([m['gt_bboxes_3d'].gravity_center,

                                     m['gt_bboxes_3d'].tensor[:, 3:]], dim=1).cuda(),

                'labels': m['gt_labels_3d'].cuda().long()

            } for m in img_metas]



            known = [torch.ones_like(t['labels'], device=device) for t in targets]

            known_num = [sum(k) for k in known]



            # can be modified to selectively denosie some label or boxes; also known label prediction

            unmask_bbox = unmask_label = torch.cat(known)

            labels = torch.cat([t['labels'] for t in targets]).clone()

            bboxes = torch.cat([t['bboxes'] for t in targets]).clone()

            batch_idx = torch.cat([torch.full_like(t['labels'].long(), i) for i, t in enumerate(targets)])



            known_indice = torch.nonzero(unmask_label + unmask_bbox)

            known_indice = known_indice.view(-1)



            # add noise

            known_indice = known_indice.repeat(self.dn_group_num, 1).view(-1)

            known_labels = labels.repeat(self.dn_group_num, 1).view(-1)

            known_bid = batch_idx.repeat(self.dn_group_num, 1).view(-1)

            known_bboxs = bboxes.repeat(self.dn_group_num, 1) # 9

            known_labels_expand = known_labels.clone()

            known_bbox_expand = known_bboxs.clone()



            wlh = known_bbox_expand[..., 3:6].clone()

            known_bbox_expand = encode_bbox(known_bbox_expand, self.pc_range)

            map_size, polar_radius = compute_map_size_and_radius(self.pc_range)
            known_bbox_expand = xy2theta_d_coods(
                known_bbox_expand, map_size, polar_radius
            )



            # noise on the box

            if self.dn_bbox_noise_scale > 0:

                r = 65.0

                rand_prob = torch.rand_like(known_bbox_expand) * 2 - 1.0

                arc_len_ratio = torch.sqrt(wlh[...,0:1]**2+wlh[...,1:2]**2) / (2*torch.pi*known_bbox_expand[..., 1:2]*r)

                theta_delta = torch.mul(rand_prob[..., 0:1], arc_len_ratio/2) * self.dn_bbox_noise_scale * known_bbox_expand[..., 1:2]

                

                d_delta = torch.mul(rand_prob[..., 1:2], torch.sqrt(wlh[...,0:1]**2+wlh[...,1:2]**2) / (r*2))  * self.dn_bbox_noise_scale



                known_bbox_expand[..., 0:1] += theta_delta

                known_bbox_expand[..., 0:1] = ((known_bbox_expand[..., 0:1]+1) * 2*torch.pi % (2 * torch.pi)) / (2 * torch.pi)

                known_bbox_expand[..., 1:2] += d_delta



                known_bbox_expand[..., 2:3] += torch.mul(rand_prob[..., 2:3], wlh[..., 2:3] / (8*2)) * self.dn_bbox_noise_scale

            

            known_bbox_expand[..., 0:3].clamp_(min=0.0, max=1.0)

            # noise on the label

            if self.dn_label_noise_scale > 0:

                p = torch.rand_like(known_labels_expand.float())

                chosen_indice = torch.nonzero(p < self.dn_label_noise_scale).view(-1)  # usually half of bbox noise

                new_label = torch.randint_like(chosen_indice, 0, self.num_classes)  # randomly put a new one here

                known_labels_expand.scatter_(0, chosen_indice, new_label)



            known_feat_expand = label_enc(known_labels_expand)

            indicator1 = torch.ones([known_feat_expand.shape[0], 1], device=device)  # add dn part indicator

            known_feat_expand = torch.cat([known_feat_expand, indicator1], dim=1)



            # construct final query

            dn_single_pad = int(max(known_num))

            dn_pad_size = int(dn_single_pad * self.dn_group_num)

            dn_query_bbox = torch.zeros([batch_size, dn_pad_size, init_query_bbox.shape[-1]], device=device)

            dn_query_feat = torch.zeros([batch_size, dn_pad_size, self.embed_dims], device=device)

            input_query_bbox = torch.cat([dn_query_bbox, init_query_bbox], dim=1)

            input_query_feat = torch.cat([dn_query_feat, init_query_feat], dim=1)



            if len(known_num):

                map_known_indice = torch.cat([torch.tensor(range(num)) for num in known_num])  # [1,2, 1,2,3]

                map_known_indice = torch.cat([map_known_indice + dn_single_pad * i for i in range(self.dn_group_num)]).long()



            if len(known_bid):

                input_query_bbox[known_bid.long(), map_known_indice] = known_bbox_expand

                input_query_feat[(known_bid.long(), map_known_indice)] = known_feat_expand



            total_size = dn_pad_size + self.num_query

            attn_mask = torch.ones([total_size, total_size], device=device) < 0



            # match query cannot see the reconstruct

            attn_mask[dn_pad_size:, :dn_pad_size] = True

            for i in range(self.dn_group_num):

                if i == 0:

                    attn_mask[dn_single_pad * i:dn_single_pad * (i + 1), dn_single_pad * (i + 1):dn_pad_size] = True

                if i == self.dn_group_num - 1:

                    attn_mask[dn_single_pad * i:dn_single_pad * (i + 1), :dn_single_pad * i] = True

                else:

                    attn_mask[dn_single_pad * i:dn_single_pad * (i + 1), dn_single_pad * (i + 1):dn_pad_size] = True

                    attn_mask[dn_single_pad * i:dn_single_pad * (i + 1), :dn_single_pad * i] = True



            mask_dict = {

                'known_indice': torch.as_tensor(known_indice).long(),

                'batch_idx': torch.as_tensor(batch_idx).long(),

                'map_known_indice': torch.as_tensor(map_known_indice).long(),

                'known_lbs_bboxes': (known_labels, known_bboxs),

                'pad_size': dn_pad_size

            }

        else:

            input_query_bbox = init_query_bbox

            input_query_feat = init_query_feat

            attn_mask = None

            mask_dict = None



        return input_query_bbox, input_query_feat, attn_mask, mask_dict



    def prepare_for_dn_loss(self, mask_dict):

        cls_scores, bbox_preds = mask_dict['output_known_lbs_bboxes']

        known_labels, known_bboxs = mask_dict['known_lbs_bboxes']

        map_known_indice = mask_dict['map_known_indice'].long()

        known_indice = mask_dict['known_indice'].long()

        batch_idx = mask_dict['batch_idx'].long()

        bid = batch_idx[known_indice]

        num_tgt = known_indice.numel()



        if len(cls_scores) > 0:

            cls_scores = cls_scores.permute(1, 2, 0, 3)[(bid, map_known_indice)].permute(1, 0, 2)

            bbox_preds = bbox_preds.permute(1, 2, 0, 3)[(bid, map_known_indice)].permute(1, 0, 2)



        return known_labels, known_bboxs, cls_scores, bbox_preds, num_tgt



    def dn_loss_single(self,

                       cls_scores,

                       bbox_preds,

                       known_bboxs,

                       known_labels,

                       num_total_pos=None):        

        # Compute the average number of gt boxes accross all gpus

        num_total_pos = cls_scores.new_tensor([num_total_pos])

        num_total_pos = torch.clamp(reduce_mean(num_total_pos), min=1.0).item()



        # cls loss

        cls_scores = cls_scores.reshape(-1, self.cls_out_channels)

        bbox_weights = torch.ones_like(bbox_preds)

        label_weights = torch.ones_like(known_labels)

        loss_cls = self.loss_cls(

            cls_scores,

            known_labels.long(),

            label_weights,

            avg_factor=num_total_pos

        )



        # regression L1 loss

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



    @force_fp32(apply_to=('preds_dicts'))

    def calc_dn_loss(self, loss_dict, preds_dicts, num_dec_layers):

        known_labels, known_bboxs, cls_scores, bbox_preds, num_tgt = \

            self.prepare_for_dn_loss(preds_dicts['dn_mask_dict'])



        all_known_bboxs_list = [known_bboxs for _ in range(num_dec_layers)]

        all_known_labels_list = [known_labels for _ in range(num_dec_layers)]

        all_num_tgts_list = [num_tgt for _ in range(num_dec_layers)]



        dn_losses_cls, dn_losses_bbox = multi_apply(

            self.dn_loss_single, cls_scores, bbox_preds,

            all_known_bboxs_list, all_known_labels_list, all_num_tgts_list)



        loss_dict['loss_cls_dn'] = dn_losses_cls[-1]

        loss_dict['loss_bbox_dn'] = dn_losses_bbox[-1]



        num_dec_layer = 0

        for loss_cls_i, loss_bbox_i in zip(dn_losses_cls[:-1], dn_losses_bbox[:-1]):

            loss_dict[f'd{num_dec_layer}.loss_cls_dn'] = loss_cls_i

            loss_dict[f'd{num_dec_layer}.loss_bbox_dn'] = loss_bbox_i

            num_dec_layer += 1



        return loss_dict



    def _get_target_single(self,

                           cls_score,

                           bbox_pred,

                           gt_labels,

                           gt_bboxes,

                           gt_bboxes_ignore=None):

        num_bboxes = bbox_pred.size(0)



        # assigner and sampler

        assign_result = self.assigner.assign(bbox_pred, cls_score, gt_bboxes, gt_labels, gt_bboxes_ignore, self.code_weights, True)

        sampling_result = self.sampler.sample(assign_result, bbox_pred, gt_bboxes)

        pos_inds = sampling_result.pos_inds

        neg_inds = sampling_result.neg_inds



        # label targets

        labels = gt_bboxes.new_full((num_bboxes, ), self.num_classes, dtype=torch.long)

        labels[pos_inds] = gt_labels[sampling_result.pos_assigned_gt_inds]

        label_weights = gt_bboxes.new_ones(num_bboxes)



        # bbox targets

        bbox_targets = torch.zeros_like(bbox_pred)[..., :9]

        bbox_weights = torch.zeros_like(bbox_pred)

        bbox_weights[pos_inds] = 1.0

        

        # DETR

        bbox_targets[pos_inds] = sampling_result.pos_gt_bboxes

        return (labels, label_weights, bbox_targets, bbox_weights, pos_inds, neg_inds)



    def get_targets(self,

                    cls_scores_list,

                    bbox_preds_list,

                    gt_bboxes_list,

                    gt_labels_list,

                    gt_bboxes_ignore_list=None):

        assert gt_bboxes_ignore_list is None, \

            'Only supports for gt_bboxes_ignore setting to None.'

        num_imgs = len(cls_scores_list)

        gt_bboxes_ignore_list = [gt_bboxes_ignore_list for _ in range(num_imgs)]



        (labels_list, label_weights_list, bbox_targets_list,

         bbox_weights_list, pos_inds_list, neg_inds_list) = multi_apply(

                self._get_target_single, cls_scores_list, bbox_preds_list,

             gt_labels_list, gt_bboxes_list, gt_bboxes_ignore_list)

        num_total_pos = sum((inds.numel() for inds in pos_inds_list))

        num_total_neg = sum((inds.numel() for inds in neg_inds_list))

        return (labels_list, label_weights_list, bbox_targets_list,

                bbox_weights_list, num_total_pos, num_total_neg)



    def loss_single(self,

                    cls_scores,

                    bbox_preds,

                    gt_bboxes_list,

                    gt_labels_list,

                    gt_bboxes_ignore_list=None):

        num_imgs = cls_scores.size(0)

        cls_scores_list = [cls_scores[i] for i in range(num_imgs)]

        bbox_preds_list = [bbox_preds[i] for i in range(num_imgs)]

        cls_reg_targets = self.get_targets(cls_scores_list, bbox_preds_list,

                gt_bboxes_list, gt_labels_list, gt_bboxes_ignore_list)

        (labels_list, label_weights_list, bbox_targets_list, bbox_weights_list,

         num_total_pos, num_total_neg) = cls_reg_targets



        labels = torch.cat(labels_list, 0)

        label_weights = torch.cat(label_weights_list, 0)

        bbox_targets = torch.cat(bbox_targets_list, 0)

        bbox_weights = torch.cat(bbox_weights_list, 0)



        # classification loss

        cls_scores = cls_scores.reshape(-1, self.cls_out_channels)

        # construct weighted avg_factor to match with the official DETR repo

        cls_avg_factor = num_total_pos * 1.0 + \

            num_total_neg * self.bg_cls_weight

        if self.sync_cls_avg_factor:

            cls_avg_factor = reduce_mean(

                cls_scores.new_tensor([cls_avg_factor]))



        cls_avg_factor = max(cls_avg_factor, 1)

        loss_cls = self.loss_cls(

            cls_scores, labels, label_weights, avg_factor=cls_avg_factor)



        # Compute the average number of gt boxes accross all gpus, for

        # normalization purposes

        num_total_pos = loss_cls.new_tensor([num_total_pos])

        num_total_pos = torch.clamp(reduce_mean(num_total_pos), min=1).item()



        # regression L1 loss

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



    @force_fp32(apply_to=('preds_dicts'))

    def loss(self,

             gt_bboxes_list,

             gt_labels_list,

             preds_dicts,

             gt_bboxes_ignore=None):

        assert gt_bboxes_ignore is None, \

            f'{self.__class__.__name__} only supports ' \

            f'for gt_bboxes_ignore setting to None.'



        all_cls_scores = preds_dicts['all_cls_scores']

        all_bbox_preds = preds_dicts['all_bbox_preds']

        enc_cls_scores = preds_dicts['enc_cls_scores']

        enc_bbox_preds = preds_dicts['enc_bbox_preds']



        num_dec_layers = len(all_cls_scores)

        device = gt_labels_list[0].device

        gt_bboxes_list = [torch.cat(

            (gt_bboxes.gravity_center, gt_bboxes.tensor[:, 3:]),

            dim=1).to(device) for gt_bboxes in gt_bboxes_list]



        all_gt_bboxes_list = [gt_bboxes_list for _ in range(num_dec_layers)]

        all_gt_labels_list = [gt_labels_list for _ in range(num_dec_layers)]

        all_gt_bboxes_ignore_list = [gt_bboxes_ignore for _ in range(num_dec_layers)]



        losses_cls, losses_bbox = multi_apply(

            self.loss_single, all_cls_scores, all_bbox_preds,

            all_gt_bboxes_list, all_gt_labels_list, 

            all_gt_bboxes_ignore_list)



        loss_dict = dict()

        # loss of proposal generated from encode feature map

        if enc_cls_scores is not None:

            binary_labels_list = [

                torch.zeros_like(gt_labels_list[i])

                for i in range(len(all_gt_labels_list))

            ]

            enc_loss_cls, enc_losses_bbox = \

                self.loss_single(enc_cls_scores, enc_bbox_preds,

                                 gt_bboxes_list, binary_labels_list, gt_bboxes_ignore)

            loss_dict['enc_loss_cls'] = enc_loss_cls

            loss_dict['enc_loss_bbox'] = enc_losses_bbox



        if 'dn_mask_dict' in preds_dicts and preds_dicts['dn_mask_dict'] is not None:

            loss_dict = self.calc_dn_loss(loss_dict, preds_dicts, num_dec_layers)



        # loss from the last decoder layer

        loss_dict['loss_cls'] = losses_cls[-1]

        loss_dict['loss_bbox'] = losses_bbox[-1]



        # loss from other decoder layers

        num_dec_layer = 0

        for loss_cls_i, loss_bbox_i in zip(losses_cls[:-1], losses_bbox[:-1]):

            loss_dict[f'd{num_dec_layer}.loss_cls'] = loss_cls_i

            loss_dict[f'd{num_dec_layer}.loss_bbox'] = loss_bbox_i

            num_dec_layer += 1

        return loss_dict



    @force_fp32(apply_to=('preds_dicts'))

    def get_bboxes(self, preds_dicts, img_metas, rescale=False):

        """

        获取检测框结果

        

        Args:

            preds_dicts: 预测字典

            img_metas: 图像元信息

            rescale: 是否缩放

            

        Returns:

            list: 每个样本的检测结果 [bboxes, scores, labels]

                  即使没有检测到目标，也会返回空的LiDARInstance3DBoxes，不会返回None

        """

        preds_dicts = self.bbox_coder.decode(preds_dicts)

        num_samples = len(preds_dicts)

        ret_list = []

        for i in range(num_samples):

            preds = preds_dicts[i]

            bboxes = preds['bboxes']

            scores = preds['scores']

            labels = preds['labels']

            

            # ============================================================

            # 防护代码：处理没有检测到目标的情况

            # 确保返回空的LiDARInstance3DBoxes而不是None

            # ============================================================

            if bboxes.shape[0] == 0:

                # 创建空的bboxes tensor，维度为 [0, 9]

                empty_bboxes = bboxes.new_zeros((0, 9))

                bboxes = LiDARInstance3DBoxes(empty_bboxes, 9)

                ret_list.append([bboxes, scores, labels])

                continue

            

            # 正常情况：有检测结果

            bboxes[:, 2] = bboxes[:, 2] - bboxes[:, 5] * 0.5



            if VERSION.name == 'v0.17.1':

                import copy

                w, l = copy.deepcopy(bboxes[:, 3]), copy.deepcopy(bboxes[:, 4])

                bboxes[:, 3], bboxes[:, 4] = l, w

                bboxes[:, 6] = -bboxes[:, 6] - math.pi / 2



            bboxes = LiDARInstance3DBoxes(bboxes, 9)

            ret_list.append([bboxes, scores, labels])

        return ret_list
