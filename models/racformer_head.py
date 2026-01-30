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
from .bbox.utils import normalize_bbox, encode_bbox, theta_d2xy_coods, xy2theta_d_coods, R_MAX
from .utils import VERSION

# RWHI 相关常量
EPS = 1e-5
# R_MAX 从 bbox/utils.py 导入，确保全链路统一


@HEADS.register_module()
class RaCFormer_head(DETRHead):
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
                 # RWHI 相关参数
                 use_rwhi=False,
                 use_alpha=True,
                 rwhi_gate_init=0.2,
                 rwhi_gate_const=0.7,
                 rwhi_affect_query=True,
                 rwhi_cfg=None,
                 polar_radius=None,
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

        # RWHI 相关
        self.use_rwhi = use_rwhi
        self.use_alpha = use_alpha
        self.rwhi_affect_query = rwhi_affect_query
        self.rwhi_gate_init = rwhi_gate_init
        self._rwhi_gate_const_value = rwhi_gate_const
        self.rwhi_cfg = rwhi_cfg if rwhi_cfg is not None else {}
        self.rwhi_cfg.setdefault('use_alpha', self.use_alpha)
        
        # 全链路统一的极坐标半径
        self.polar_radius = polar_radius if polar_radius is not None else R_MAX

        # 【关键修复】必须在 super().__init__() 之前构建 bbox_coder 并设置 pc_range
        # 因为 super().__init__() 会调用 _init_layers()，而 _init_rwhi_layers() 需要 self.pc_range
        self.bbox_coder = build_bbox_coder(bbox_coder)
        self.pc_range = self.bbox_coder.pc_range

        super(RaCFormer_head, self).__init__(num_classes, in_channels, train_cfg=train_cfg, test_cfg=test_cfg, **kwargs)

        self.code_weights = nn.Parameter(torch.tensor(self.code_weights), requires_grad=False)

        self.dn_enabled = query_denoising
        self.dn_group_num = query_denoising_groups
        self.dn_weight = 1.0
        self.dn_bbox_noise_scale = 0.5
        self.dn_label_noise_scale = 0.5

    def _init_layers(self):
        self.init_query_bbox = nn.Embedding(self.num_query, 10)  # (x, y, z, w, l, h, sin, cos, vx, vy)
        self.label_enc = nn.Embedding(self.num_classes + 1, self.embed_dims - 1)  # DAB-DETR

        # nn.init.zeros_(self.init_query_bbox.weight[:, 2:3])
        nn.init.constant_(self.init_query_bbox.weight[:, 2:3], 0.5)
        nn.init.zeros_(self.init_query_bbox.weight[:, 8:10])
        # nn.init.constant_(self.init_query_bbox.weight[:, 5:6], 1.5)
        nn.init.constant_(self.init_query_bbox.weight[:, 5:6], 0.2)

        theta_d = self.generate_points()
        with torch.no_grad():
            self.init_query_bbox.weight[:, :2] = theta_d.reshape(-1, 2)  # [Q, 2]

        # 初始化 RWHI 相关层
        if self.use_rwhi:
            self._init_rwhi_layers()

    def _init_rwhi_layers(self):
        """初始化 RWHI 相关的网络层"""
        from .rwhi import build_rwhi
        
        # 构建 RWHI 模块
        rwhi_params = dict(
            num_query=self.num_query,
            pc_range=self.pc_range,
            **self.rwhi_cfg
        )
        self.rwhi_module = build_rwhi(**rwhi_params)
        
        # ✅ 使用 RWHI 的 safety_anchors 初始化 init_query_bbox
        # 仅复制 θ, d, z, h, sin, cos, vx, vy，保留 w/l 的随机初始化
        with torch.no_grad():
            safety_anchors = self.rwhi_module.safety_anchors
            # θ, d (indices 0, 1)
            self.init_query_bbox.weight[:, 0:2].copy_(safety_anchors[:, 0:2])
            # z (index 2)
            self.init_query_bbox.weight[:, 2].copy_(safety_anchors[:, 2])
            # w, l (indices 3, 4) - 保留随机初始化，不复制！
            # h (index 5)
            self.init_query_bbox.weight[:, 5].copy_(safety_anchors[:, 5])
            # sin, cos, vx, vy (indices 6, 7, 8, 9)
            self.init_query_bbox.weight[:, 6:10].copy_(safety_anchors[:, 6:10])
        
        # pos2content: 只用位置生成动态内容
        self.pos2content = nn.Sequential(
            nn.Linear(3, self.embed_dims),
            nn.LayerNorm(self.embed_dims),
            nn.ReLU(inplace=True),
            nn.Linear(self.embed_dims, self.embed_dims - 1),
        )
        nn.init.zeros_(self.pos2content[-1].weight)
        nn.init.zeros_(self.pos2content[-1].bias)
        
        self.alpha_fusion = None
        self.rwhi_gate = nn.Parameter(torch.tensor(self.rwhi_gate_init, dtype=torch.float32))
        self.register_buffer('rwhi_gate_const', torch.tensor(self._rwhi_gate_const_value, dtype=torch.float32))

    def init_weights(self):
        self.transformer.init_weights()

    def generate_points(self):
        # 【修复】使用 ceil 处理不整除情况，与 safety_anchors 保持一致
        num_angles = math.ceil(self.num_query / self.num_clusters)
        angles = torch.linspace(0, 1, num_angles+1)[:-1]
        distances = torch.linspace(0, 1, self.num_clusters + 2,  dtype=torch.float)[1:-1]

        angles = angles.view(num_angles, 1).expand(num_angles, self.num_clusters)
        distances = distances.view(1, self.num_clusters).expand(num_angles, self.num_clusters)

        theta_d = torch.cat([angles[..., None], distances[..., None]], dim=-1).flatten(0,1)
        
        # 【修复】截断到 num_query，处理不整除情况
        if theta_d.shape[0] > self.num_query:
            theta_d = theta_d[:self.num_query]
        
        return theta_d

    def _validate_query_bbox(self, query_bbox):
        """
        验证并 clamp query_bbox 的值范围
        
        关键:
        - theta: [0, 1-EPS]
        - d: (EPS, 1-EPS)  ← 必须严格开区间
        - z: (EPS, 1-EPS)  ← 必须严格开区间
        
        Args:
            query_bbox: [B, K, 10] query bbox
        
        Returns:
            validated_bbox: [B, K, 10] 验证后的 bbox
        """
        validated = query_bbox.clone()
        
        # out-of-place clamp 避免版本冲突
        theta = validated[..., 0].clamp(min=0.0, max=1.0 - EPS)
        d = validated[..., 1].clamp(min=EPS, max=1.0 - EPS)
        z = validated[..., 2].clamp(min=EPS, max=1.0 - EPS)
        
        validated = torch.cat([
            theta.unsqueeze(-1),
            d.unsqueeze(-1),
            z.unsqueeze(-1),
            validated[..., 3:],
        ], dim=-1)
        
        return validated

    def _prepare_query_bbox(self, radar_points, radar_mask, batch_size, device):
        """
        准备 query bbox
        
        如果启用 RWHI 且有有效雷达点，使用 RWHI 生成动态锚点；
        否则使用静态的 init_query_bbox。
        
        Args:
            radar_points: [B, M, C] 雷达点或 None
            radar_mask: [B, M] 有效点掩码或 None
            batch_size: batch 大小
            device: 设备
        
        Returns:
            query_bbox: [B, K, 10]
            alpha_values: [B, K, 1] 或 None
            using_dynamic_rwhi: bool
        """
        using_dynamic_rwhi = False
        alpha_values = None
        num_rwhi = self.rwhi_cfg.get('num_rwhi', self.num_query)
        
        if self.use_rwhi and radar_points is not None and num_rwhi > 0:
            # 【修复】检查是否有有效雷达点，使用 .item() 避免 tensor 在布尔上下文中的问题
            has_valid_points = radar_mask is not None and radar_mask.sum().item() > 0
            
            if has_valid_points:
                # 使用 RWHI 生成动态锚点
                query_bbox, alpha_values = self.rwhi_module(radar_points, radar_mask)
                
                # ✅ 使用 init_query_bbox 的 w/l 替代动态采样，保持确定性和可学习性
                # 动态路径仅提供 θ/d/z（雷达引导），w/l 来自 init_query_bbox（与静态路径一致）
                # 注意：必须在slice后立即clone()，断开与原始weight的view关系，避免计算图版本冲突
                wl_from_init = self.init_query_bbox.weight[:, 3:5].clone()  # [K, 2] - clone断开view
                wl_from_init = wl_from_init.unsqueeze(0).repeat(batch_size, 1, 1)  # [B, K, 2]
                query_bbox = torch.cat([
                    query_bbox[..., :3],      # θ, d, z (来自 RWHI)
                    wl_from_init,             # w, l (来自 init_query_bbox)
                    query_bbox[..., 5:],      # h, sin, cos, vx, vy
                ], dim=-1)
                
                query_bbox = self._validate_query_bbox(query_bbox)
                using_dynamic_rwhi = True
            else:
                # 无有效雷达点，使用静态锚点
                query_bbox = self.init_query_bbox.weight.clone()
                query_bbox = query_bbox.unsqueeze(0).repeat(batch_size, 1, 1)
                query_bbox = self._validate_query_bbox(query_bbox)
        else:
            # 不使用 RWHI，使用原始静态锚点
            query_bbox = self.init_query_bbox.weight.clone()
            query_bbox = query_bbox.unsqueeze(0).repeat(batch_size, 1, 1)
        
        return query_bbox, alpha_values, using_dynamic_rwhi

    def _prepare_query_feat(self, query_bbox, alpha_values, batch_size, device, using_dynamic_rwhi):
        """
        准备 query 特征
        
        如果使用动态 RWHI：
        1. pos2content(query_bbox[..., :3]) → 动态内容
        2. 添加 indicator
        
        否则使用原始方式。
        
        Args:
            query_bbox: [B, K, 10]
            alpha_values: [B, K, 1] 或 None
            batch_size: batch 大小
            device: 设备
            using_dynamic_rwhi: 是否使用动态 RWHI
        
        Returns:
            query_feat: [B, K, embed_dims]
        """
        # 【修复】指定 dtype 确保 AMP/FP16 兼容性
        dtype = query_bbox.dtype
        indicator0 = torch.zeros([self.num_query, 1], device=device, dtype=dtype)
        
        # 当 rwhi_affect_query=False 时，RWHI 只影响锚点分布，不再注入 query 内容
        if using_dynamic_rwhi and self.use_rwhi and self.rwhi_affect_query:
            query_pos = query_bbox[..., :3]  # [B, K, 3] (θ, d, z)
            dynamic_content = self.pos2content(query_pos)  # [B, K, embed_dims-1]
            label_enc_base = self.label_enc.weight[self.num_classes].repeat(self.num_query, 1)
            label_enc_base = label_enc_base.unsqueeze(0).repeat(batch_size, 1, 1)
            label_enc_base = label_enc_base.to(dtype=dynamic_content.dtype)
            gate = self.rwhi_gate.to(dtype=dynamic_content.dtype)
            query_feat_content = label_enc_base + gate * dynamic_content
            indicator = indicator0.unsqueeze(0).repeat(batch_size, 1, 1)  # [B, K, 1]
            query_feat = torch.cat([query_feat_content, indicator], dim=-1)  # [B, K, embed_dims]
        else:
            # 原始方式
            init_query_feat = self.label_enc.weight[self.num_classes].repeat(self.num_query, 1)
            init_query_feat = torch.cat([init_query_feat, indicator0], dim=1).repeat(batch_size, 1, 1)
            query_feat = init_query_feat
        
        # 【关键修复】DDP 兼容性：确保所有参数都参与计算图
        # 当 use_rwhi=True 时：rwhi_module, pos2content, alpha_fusion 可能不参与（空雷达点）
        # 当 using_dynamic_rwhi=True 时：init_query_bbox 不参与（使用 RWHI 生成的锚点）
        if self.training:
            dummy = None  # 使用 None 而不是 0.0，避免 tensor 与 float 比较问题
            
            if self.use_rwhi:
                modules = [self.pos2content]
                for module in modules:
                    for param in module.parameters():
                        term = param.sum() * 0.0
                        dummy = term if dummy is None else dummy + term
                for param in self.rwhi_module.parameters():
                    term = param.sum() * 0.0
                    dummy = term if dummy is None else dummy + term
                if isinstance(self.rwhi_gate, nn.Parameter):
                    term = self.rwhi_gate.sum() * 0.0
                    dummy = term if dummy is None else dummy + term
                
                # 【新增修复】当使用动态 RWHI 时，init_query_bbox 未参与计算
                # 需要添加 dummy sum 确保其参与计算图
                if using_dynamic_rwhi:
                    for param in self.init_query_bbox.parameters():
                        term = param.sum() * 0.0
                        dummy = term if dummy is None else dummy + term
            
            # 【修复】使用 isinstance 检查，避免 tensor 与 Python 值比较问题
            if dummy is not None:
                query_feat = query_feat + dummy
        
        return query_feat

    def forward(self, mlvl_feats, lss_bev_feats, radar_bev_feats, img_metas, radar_points=None, radar_mask=None):
        """
        前向传播
        
        Args:
            mlvl_feats: 多尺度图像特征
            lss_bev_feats: LSS BEV 特征
            radar_bev_feats: 雷达 BEV 特征
            img_metas: 图像元信息
            radar_points: [B, M, C] 雷达点云 (可选，用于 RWHI)
            radar_mask: [B, M] 雷达点有效掩码 (可选)
        """
        B = lss_bev_feats.shape[0]
        device = lss_bev_feats.device

        # 准备 query_bbox
        query_bbox, alpha_values, using_dynamic_rwhi = self._prepare_query_bbox(
            radar_points, radar_mask, B, device
        )
        
        # 准备 query_feat
        query_feat = self._prepare_query_feat(
            query_bbox, alpha_values, B, device, using_dynamic_rwhi
        )

        # query denoising
        # 【关键修复】传入 RWHI 构建的 query_feat，而不是让 prepare_for_dn_input 重新创建
        query_bbox, query_feat, attn_mask, mask_dict = self.prepare_for_dn_input(
            B, query_bbox, query_feat, self.label_enc, img_metas
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
                'alpha_values': alpha_values,
            }
        else:
            outs = {
                'all_cls_scores': cls_scores,
                'all_bbox_preds': bbox_preds,
                'enc_cls_scores': None,
                'enc_bbox_preds': None, 
                'alpha_values': alpha_values,
            }

        return outs

    def prepare_for_dn_input(self, batch_size, init_query_bbox, init_query_feat, label_enc, img_metas):
        """
        准备 Query Denoising 输入
        
        【关键修复】接收外部传入的 init_query_feat，而不是内部重建
        这样 RWHI 的 pos2content 特征增强才能传递到 transformer
        
        Args:
            batch_size: batch 大小
            init_query_bbox: [B, Q, 10] 初始 query bbox (可能来自 RWHI)
            init_query_feat: [B, Q, embed_dims] 初始 query 特征 (可能包含 RWHI alpha 增强)
            label_enc: 标签编码器
            img_metas: 图像元信息
        """
        # mostly borrowed from:
        #  - https://github.com/IDEA-Research/DN-DETR/blob/main/models/DN_DAB_DETR/dn_components.py
        #  - https://github.com/megvii-research/PETR/blob/main/projects/mmdet3d_plugin/models/dense_heads/petrv2_dnhead.py

        device = init_query_bbox.device
        # 【删除】不再内部重建 init_query_feat，使用外部传入的版本
        # 这样 RWHI 的特征增强才能生效

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
            known_bbox_expand = xy2theta_d_coods(known_bbox_expand)

            # noise on the box
            if self.dn_bbox_noise_scale > 0:
                # 使用全链路统一的 polar_radius
                r = self.polar_radius
                rand_prob = torch.rand_like(known_bbox_expand) * 2 - 1.0
                arc_len_ratio = torch.sqrt(wlh[...,0:1]**2+wlh[...,1:2]**2) / (2*torch.pi*known_bbox_expand[..., 1:2]*r)
                theta_delta = torch.mul(rand_prob[..., 0:1], arc_len_ratio/2) * self.dn_bbox_noise_scale * known_bbox_expand[..., 1:2]
                
                d_delta = torch.mul(rand_prob[..., 1:2], torch.sqrt(wlh[...,0:1]**2+wlh[...,1:2]**2) / (r*2))  * self.dn_bbox_noise_scale

                known_bbox_expand[..., 0:1] += theta_delta
                known_bbox_expand[..., 0:1] = ((known_bbox_expand[..., 0:1]+1) * 2*torch.pi % (2 * torch.pi)) / (2 * torch.pi)
                known_bbox_expand[..., 1:2] += d_delta

                known_bbox_expand[..., 2:3] += torch.mul(rand_prob[..., 2:3], wlh[..., 2:3] / (8*2)) * self.dn_bbox_noise_scale
            
            # 【关键修复】θ/d/z 都使用 [0, 1-EPS] 或 (EPS, 1-EPS)
            # θ: [0, 1-EPS] - 避免边界值 1.0，与 RWHI 保持一致
            # d/z: (EPS, 1-EPS) - 因为 refine_bbox 使用 inverse_sigmoid，0/1 会导致 inf/NaN
            known_bbox_expand[..., 0:1].clamp_(min=0.0, max=1.0 - EPS)  # θ: [0, 1-EPS]
            known_bbox_expand[..., 1:3].clamp_(min=EPS, max=1.0 - EPS)  # d, z: (EPS, 1-EPS)
            # noise on the label
            if self.dn_label_noise_scale > 0:
                p = torch.rand_like(known_labels_expand.float())
                chosen_indice = torch.nonzero(p < self.dn_label_noise_scale).view(-1)  # usually half of bbox noise
                new_label = torch.randint_like(chosen_indice, 0, self.num_classes)  # randomly put a new one here
                known_labels_expand.scatter_(0, chosen_indice, new_label)

            known_feat_expand = label_enc(known_labels_expand)
            # 【修复】指定 dtype 确保 AMP/FP16 兼容性
            indicator1 = torch.ones([known_feat_expand.shape[0], 1], device=device, dtype=known_feat_expand.dtype)
            known_feat_expand = torch.cat([known_feat_expand, indicator1], dim=1)

            # construct final query
            dn_single_pad = int(max(known_num))
            dn_pad_size = int(dn_single_pad * self.dn_group_num)
            # 【修复】指定 dtype 确保 AMP/FP16 兼容性
            dn_query_bbox = torch.zeros([batch_size, dn_pad_size, init_query_bbox.shape[-1]], device=device, dtype=init_query_bbox.dtype)
            dn_query_feat = torch.zeros([batch_size, dn_pad_size, self.embed_dims], device=device, dtype=init_query_feat.dtype)
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
            input_query_bbox = init_query_bbox.repeat(batch_size, 1, 1) if init_query_bbox.dim() == 2 else init_query_bbox
            input_query_feat = init_query_feat.repeat(batch_size, 1, 1) if init_query_feat.dim() == 2 else init_query_feat
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
        preds_dicts = self.bbox_coder.decode(preds_dicts)
        num_samples = len(preds_dicts)
        ret_list = []
        for i in range(num_samples):
            preds = preds_dicts[i]
            bboxes = preds['bboxes']
            bboxes[:, 2] = bboxes[:, 2] - bboxes[:, 5] * 0.5

            if VERSION.name == 'v0.17.1':
                import copy
                w, l = copy.deepcopy(bboxes[:, 3]), copy.deepcopy(bboxes[:, 4])
                bboxes[:, 3], bboxes[:, 4] = l, w
                bboxes[:, 6] = -bboxes[:, 6] - math.pi / 2

            bboxes = LiDARInstance3DBoxes(bboxes, 9)
            scores = preds['scores']
            labels = preds['labels']
            ret_list.append([bboxes, scores, labels])
        return ret_list
