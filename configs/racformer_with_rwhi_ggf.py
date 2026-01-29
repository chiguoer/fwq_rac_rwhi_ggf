"""
RaCFormer with RWHI v7 配置文件

基于 racformer_r50_nuimg_704x256_f8.py，添加 RWHI v7 模块支持。
RWHI (Radar-Weighted Hybrid Initialization) 用雷达引导的动态锚点替代均匀极坐标分布锚点。
"""

import torch
pi = torch.pi

dataset_type = 'CustomNuScenesDataset_radar'
dataset_root = 'data/nuscenes/'

input_modality = dict(
    use_lidar=False,
    use_camera=True,
    use_radar=True,
    use_map=False,
    use_external=True
)

# For nuScenes we usually do 10-class detection
class_names = [
    'car', 'truck', 'trailer', 'bus', 'construction_vehicle', 'bicycle',
    'motorcycle', 'pedestrian', 'traffic_cone', 'barrier'
]

# If point cloud range is changed, the models should also change their point
# cloud range accordingly
point_cloud_range = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]
voxel_size = [0.2, 0.2, 8]

# arch config
embed_dims = 256
num_layers = 6

num_frames = 8
num_levels = 4
num_points = 4
num_points_bev = 4
img_depth_num = 3

bev_depth_num = 5 

d_region_list = [0.08, 0.07, 0.06, 0.05, 0.04, 0.03]

num_clusters = 6
num_ray = 900 // num_clusters
num_query = 900
num_rwhi = 600
use_alpha = True
alpha_const = 0.7
rwhi_gate_init = 0.2
rwhi_gate_const = 0.7
loss_alpha_anchor_weight = 0.2
# 控制 RWHI 是否影响 query 特征（默认 True，保持旧行为）
rwhi_affect_query = False

# ============ RWHI v7 关键配置 ============
# 全链路统一的极坐标半径 - 必须在所有地方保持一致
R_MAX = 65.0

# RWHI 模块配置
rwhi_cfg = dict(
    # RWHI 版本
    rwhi_version='v7',
    
    # BEV 网格
    bev_grid_size=100,
    
    # I_radar 参数
    d_ref=30.0,           # 距离权重参考值
    d_lambda=1.0,         # 距离权重指数
    w_d_max=2.5,          # 距离权重上限
    d_min=2.0,            # 最小有效距离
    v_max=30.0,           # 最大速度
    v_ref=10.0,           # 速度权重参考值
    beta=1.0,             # 速度权重系数
    
    # 打分场参数
    base_bias=1.0,        # 基础分数 C_base
    epsilon=0.01,         # 微扰层系数
    diffusion_type='avg', # 扩散类型: 'max'/'avg'/'none'
    diffusion_kernel=2,   # 扩散核尺寸: 1/2/3；为1或type='none'时跳过池化
    diffusion_gamma=0.2,  # 扩散系数 λ
    diffusion_s_max=5.0,  # 得分上限
    
    # 默认值 (用于锚点初始化)
    z_default=0.5,        # 默认归一化高度 (对应物理高度 -1.0m)
    w_default=1.8,        # 默认物体宽度 (物理值，会转为 log)
    l_default=4.0,        # 默认物体长度 (物理值，会转为 log)
    h_default=1.5,        # 默认物体高度 (物理值，会转为 log)
    
    # AlphaMLP 参数
    alpha_mlp_in_dim=3,   # 输入维度 [log1p(rcs), d_norm, v_norm]
    alpha_mlp_hidden=32,  # 隐藏层维度
    alpha_init_bias=1.0,  # 初始偏置，使初始 α ≈ 0.73
    alpha_const=alpha_const,  # use_alpha=False 时的常数 α
    use_alpha=use_alpha,      # 控制是否启用 AlphaMLP/Encoder
    
    # AlphaEncoder 参数
    d_alpha=2,            # α embedding 维度
    alpha_encoder_hidden=8,

    # 其他
    num_clusters=num_clusters,  # 距离层数量 (与 RaCFormer 一致)
    max_points=5000,      # 最大雷达点数
    enabled=True,         # 是否启用 RWHI
    num_rwhi=num_rwhi,    # 雷达引导锚点数量
    enable_diverse_topk=False,
    coarse_factor=4,
    max_per_cell=5,
)
# ============ RWHI v7 配置结束 ============

# ============ GGF2.0 配置 ============
# GGF (Geometry-Guided Fusion) 模块配置
# 所有子模块都可单独开关，方便消融实验
# 典型组合示例（建议用 --override 快速切换）：
# A: RGF-only
#   --override ggf_cfg.enabled=True ggf_cfg.use_native_rgf=True ggf_cfg.use_gga=False ggf_cfg.use_mgc=False
# B: RGF + GGA
#   --override ggf_cfg.enabled=True ggf_cfg.use_native_rgf=True ggf_cfg.use_gga=True ggf_cfg.use_mgc=False
# C: Full UPG (RGF + GGA + MGC)
#   --override ggf_cfg.enabled=True ggf_cfg.use_native_rgf=True ggf_cfg.use_gga=True ggf_cfg.use_mgc=True
ggf_cfg = dict(
    # 总开关
    enabled=True,  # 默认开启，启用 GGF2.0
    
    # 子模块开关
    use_mgc=True,           # MGC: 视觉修正雷达几何
    use_gga=True,           # GGA: 几何引导注意力
    use_unified_field=True, # 统一场积分（用于 RWHI）
    use_native_rgf=True,    # 原生高斯场实现
    
    # embed_dims（与模型一致）
    embed_dims=embed_dims,
    
    # 几何场构建参数
    field_cfg=dict(
        # NativeRGF 参数
        rgf_sigma_x=3.0,        # 默认 x 方向标准差 (m)
        rgf_sigma_y=3.0,        # 默认 y 方向标准差 (m)
        rgf_use_velocity=True,  # 根据速度调整各向异性
        rgf_velocity_scale=0.1, # 速度影响协方差的比例
        rgf_kernel_size=7,      # 局部散射核尺寸
        rgf_predict_params=True,  # 预测 (sx, sy, theta)
        rgf_input_indices=[0, 1, 2, 3, 4],  # x,y,z,rcs,v_r
        rgf_hidden_dims=64,
        rgf_use_rotation=True,
        rgf_theta_scale=3.1415926,
        rgf_profile=False,
        rgf_profile_every=100,
        chunk_size=512,
        
        # 场缩放参数（与 RWHI 打分场数值尺度保持一致）
        linear_scale=1.0,       # 线性场缩放
        linear_bias=1.0,        # 线性场基础偏置（背景分数）
        linear_max=5.0,         # 线性场上限（与 diffusion_s_max 一致）
        log_temperature=1.0,    # 对数场温度参数
        
        # 融合参数
        rwhi_fusion_mode='add', # 融合模式: 'replace', 'add', 'gate'
        rwhi_fusion_weight=0.5, # 融合权重
    ),
    
    # MGC 参数
    mgc_cfg=dict(
        constraint_mode='soft',     # 约束模式: 'soft', 'hard'
        constraint_strength=1.0,    # 约束强度
        ellipse_scale=2.0,          # 椭圆半径倍数
        learnable_strength=True,    # 可学习的约束强度
        use_image_sampling=True,    # 仅作用于图像分支采样
        sample_res=(3, 4),          # 与 num_points*depth_num 对齐 (3*4=12)
        view_select='first_valid',
        min_depth=1e-5,
        align_corners=True,
        spd_eig_min=1e-4,
        spd_eig_max=None,
        fallback_scale=1e-2,
        max_dist=30.0,
        debug_mgc=False,
        debug_mgc_every=1000,
        debug_mgc_max_print=5,
        profile_mgc=False,
        profile_mgc_every=100,
    ),
    
    # GGA 参数
    gga_cfg=dict(
        num_heads=8,                # 注意力头数（与模型一致）
        temperature=1.0,            # 温度参数
        learnable_temperature=True, # 可学习温度
        bias_scale=1.0,             # 偏置缩放系数
        bias_min=-100.0,            # 偏置下限
        use_query_projection=False, # 是否对 Query 做投影
        soft_clamp_min=-100.0,      # 软截断下限
        soft_clamp_beta=1.0,
        chunk_size=256,             # 按 M 维分块，避免巨大 bias 张量
        return_geometry_bias=False, # 不返回完整 bias，避免显存暴涨
        debug_gga=True,
        debug_gga_every=1000,
    ),
)
# ============ GGF2.0 配置结束 ============

ida_aug_conf = {
    'resize_lim': (0.38, 0.55),
    'final_dim': (256, 704),
    'bot_pct_lim': (0.0, 0.0),
    'rot_lim': (0.0, 0.0),
    'H': 900, 'W': 1600,
    'rand_flip': True,
}

# Model
grid_config = {
    'x': [-51.2, 51.2, 0.8],
    'y': [-51.2, 51.2, 0.8],
    'z': [-5, 3, 8],
    'depth': [1.0, 65.0, 96.0],
    'rcs': [-64, 64, 64]
}

numC_Trans = 256
file_client_args = dict(backend='disk')

img_backbone = dict(
    type='ResNet',
    depth=50,
    num_stages=4,
    out_indices=(0, 1, 2, 3),
    frozen_stages=1,
    norm_cfg=dict(type='BN2d', requires_grad=True),
    norm_eval=True,
    style='pytorch',
    with_cp=False)

img_neck = dict(
    type='FPN',
    in_channels=[256, 512, 1024, 2048],
    out_channels=embed_dims,
    num_outs=num_levels)

img_norm_cfg = dict(
    mean=[123.675, 116.280, 103.530],
    std=[58.395, 57.120, 57.375],
    to_rgb=True)

img_lss_neck=dict(
    type='CustomFPN',
    in_channels=[1024, 2048],
    out_channels=256,
    num_outs=1,
    start_level=0,
    out_ids=[0])

img_lss_view_transformer=dict(
    type='LSSViewTransformerBEVDepth_racformer',
    grid_config=grid_config,
    input_size=ida_aug_conf['final_dim'],
    in_channels=256,
    out_channels=numC_Trans,
    depthnet_cfg=dict(use_dcn=False),
    downsample=16,
    loss_depth_weight=2.0)

pre_process=None
model = dict(
    type='RaCFormer',
    data_aug=dict(
        img_color_aug=True,  # Move some augmentations to GPU
        img_norm_cfg=img_norm_cfg,
        img_pad_cfg=dict(size_divisor=32)),
    stop_prev_grad=0,
    img_backbone=img_backbone,
    img_neck=img_neck,
    img_lss_neck=img_lss_neck,
    img_lss_view_transformer=img_lss_view_transformer,
    num_lss_fpn=2,
    dep_downsample=16,
    pre_process=pre_process,
    radar_voxel_layer=dict(
        max_num_points=10,
        voxel_size=[0.8, 0.8, 8],
        max_voxels=(30000, 40000),
        point_cloud_range=point_cloud_range,
        deterministic=False,), 

    radar_voxel_encoder=dict(
        type='PillarFeatureNet',
        in_channels=7,
        feat_channels=[64],
        with_distance=False,
        voxel_size=[0.8, 0.8, 8],
        norm_cfg=dict(type='BN1d', eps=1e-3, momentum=0.01),
        legacy=False),

    radar_middle_encoder=dict(
        type='PointPillarsScatter', in_channels=64, output_shape=(128, 128)),

    pts_bbox_head=dict(
        type='RaCFormer_head',
        num_classes=10,
        num_clusters=num_clusters,
        in_channels=embed_dims,
        num_query=num_query,
        query_denoising=True,
        query_denoising_groups=10,
        code_size=10,
        code_weights=[2.0, 2.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        sync_cls_avg_factor=True,
        
        # ============ RWHI v7 关键配置 ============
        use_rwhi=True,                    # 启用 RWHI
        use_alpha=use_alpha,              # 是否启用 α 学习与特征融合
        rwhi_gate_init=rwhi_gate_init,    # use_alpha=True 时的可学习门控初值
        rwhi_gate_const=rwhi_gate_const,  # use_alpha=False 时的固定门控
        rwhi_affect_query=rwhi_affect_query,  # 从 config 控制是否启用
        loss_alpha_anchor_weight=loss_alpha_anchor_weight,
        rwhi_cfg=rwhi_cfg,                # RWHI 配置
        polar_radius=R_MAX,               # 全链路统一 polar_radius
        # ============ RWHI v7 配置结束 ============
        
        # ============ GGF2.0 配置 ============
        ggf_cfg=ggf_cfg,                  # GGF 配置（默认关闭）
        # ============ GGF2.0 配置结束 ============
        
        transformer=dict(
            type='RaCFormerTransformer',
            embed_dims=embed_dims,
            num_frames=num_frames,
            num_points=num_points,
            num_points_bev=num_points_bev,
            img_depth_num=img_depth_num, 
            bev_depth_num=bev_depth_num,
            num_layers=num_layers,
            num_levels=num_levels,
            num_ray=num_ray,
            num_classes=10,
            code_size=10,
            pc_range=point_cloud_range,
            d_region_list=d_region_list,
            # ============ RWHI v7: 传入 polar_radius ============
            polar_radius=R_MAX,
            # ============ GGF2.0: 传入配置 ============
            ggf_cfg=ggf_cfg,
        ),
        bbox_coder=dict(
            type='NMSFreeCoder',
            post_center_range=[-61.2, -61.2, -10.0, 61.2, 61.2, 10.0],
            pc_range=point_cloud_range,
            max_num=300,
            voxel_size=voxel_size,
            score_threshold=0.05,
            num_classes=10),
        positional_encoding=dict(
            type='SinePositionalEncoding',
            num_feats=embed_dims // 2,
            normalize=True,
            offset=-0.5),
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=2.0),
        loss_bbox=dict(type='L1Loss', loss_weight=0.25),
        loss_iou=dict(type='GIoULoss', loss_weight=0.0)),
    train_cfg=dict(pts=dict(
        grid_size=[512, 512, 1],
        voxel_size=voxel_size,
        point_cloud_range=point_cloud_range,
        out_size_factor=4,
        assigner=dict(
            type='PolarHungarianAssigner3D',
            cls_cost=dict(type='FocalLossCost', weight=2.0),
            reg_cost=dict(type='BBox3DL1Cost', weight=0.25),
            theta_cost=dict(type='ThetaL1Cost', weight=3.0),
            iou_cost=dict(type='IoUCost', weight=0.0),
        )
    ))
)


train_pipeline = [
    dict(type='LoadMultiViewImageFromFiles', to_float32=False, color_type='color'),
    dict(type='LoadMultiViewImageFromMultiSweeps', sweeps_num=num_frames - 1),
    dict(type='LoadAnnotations3D', with_bbox_3d=True, with_label_3d=True, with_attr_label=False,
        with_label=False, with_bbox_depth=False),
    dict(type='ObjectRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='ObjectNameFilter', classes=class_names),
    dict(type='RandomTransformImage', ida_aug_conf=ida_aug_conf, training=True),
    dict(type='Loadnuradarpoints', coord_type='RADAR', num_sweeps=5, file_client_args=file_client_args),
    dict(type='LoadradarpointsFromMultiSweeps', sweeps_num=num_frames-1, num_aggr_sweeps=5, test_mode=False),
    dict(type='LoadPointsFromFile', coord_type='LIDAR', load_dim=5, use_dim=5, file_client_args=file_client_args),
    dict(type='RaCGlobalRotScaleTransImage', rot_range=[-0.3925, 0.3925], scale_ratio_range=[0.95, 1.05]),
    dict(type='PointToMultiViewDepth', downsample=1, grid_config=grid_config),
    dict(type='RadarPointToMultiViewDepth', downsample=1, grid_config=grid_config, test_mode=False),
    dict(type='RaCFormatBundle3D', class_names=class_names),
    dict(type='Collect3D', keys=['gt_bboxes_3d', 'gt_labels_3d', 'img', 'gt_depth', 'radar_depth', 'radar_rcs', 'radar_points'], meta_keys=(
        'filename', 'ori_shape', 'img_shape', 'pad_shape', 'lidar2img', 'img_timestamp', 'intrinsics'))
]

test_pipeline = [
    dict(type='LoadMultiViewImageFromFiles', to_float32=False, color_type='color'),
    dict(type='LoadMultiViewImageFromMultiSweeps', sweeps_num=num_frames - 1, test_mode=True),
    dict(type='RandomTransformImage', ida_aug_conf=ida_aug_conf, training=False),
    dict(type='Loadnuradarpoints', coord_type='RADAR', num_sweeps=5, file_client_args=file_client_args),
    dict(type='LoadradarpointsFromMultiSweeps', sweeps_num=num_frames-1, num_aggr_sweeps=5, test_mode=True),
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=5,
        use_dim=5,
        file_client_args=file_client_args),
    dict(type='PointToMultiViewDepth', downsample=1, grid_config=grid_config),
    dict(type='RadarPointToMultiViewDepth', downsample=1, grid_config=grid_config, test_mode=True),
    dict(
        type='MultiScaleFlipAug3D',
        img_scale=(1600, 900),
        pts_scale_ratio=1,
        flip=False,
        transforms=[
            dict(type='RaCFormatBundle3D', class_names=class_names, with_label=False),
            dict(type='Collect3D', keys=['img', 'gt_depth', 'radar_points', 'radar_depth', 'radar_rcs'], meta_keys=(
                'filename', 'box_type_3d', 'ori_shape', 'img_shape', 'pad_shape',
                'lidar2img', 'img_timestamp', 'intrinsics'))
        ])
]

data = dict(
    workers_per_gpu=2,
    train=dict(
        type=dataset_type,
        data_root=dataset_root,
        ann_file=dataset_root + 'nuscenes_infos_train_sweep.pkl',
        pipeline=train_pipeline,
        classes=class_names,
        modality=input_modality,
        test_mode=False,
        use_valid_flag=True,
        box_type_3d='LiDAR'),
    val=dict(
        type=dataset_type,
        data_root=dataset_root,
        ann_file=dataset_root + 'nuscenes_infos_val_sweep.pkl',
        pipeline=test_pipeline,
        classes=class_names,
        modality=input_modality,
        test_mode=True,
        box_type_3d='LiDAR'),
    test=dict(
        type=dataset_type,
        data_root=dataset_root,
        ann_file=dataset_root + 'nuscenes_infos_test_sweep.pkl',
        pipeline=test_pipeline,
        classes=class_names,
        modality=input_modality,
        test_mode=True,
        box_type_3d='LiDAR')
)

optimizer = dict(
    type='AdamW',
    lr=4e-4,
    paramwise_cfg=dict(custom_keys={
        'img_backbone': dict(lr_mult=0.1),
        'sampling_offset': dict(lr_mult=0.1),
        # RWHI 模块可以使用较小的学习率
        'rwhi_module': dict(lr_mult=0.8),
    }),
    weight_decay=0.01
)

optimizer_config = dict(
    type='Fp16OptimizerHook',
    loss_scale=512.0,
    grad_clip=dict(max_norm=35, norm_type=2)
)

# learning policy
lr_config = dict(
    policy='CosineAnnealing',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=1.0 / 3,
    min_lr_ratio=1e-3
)

total_epochs = 20
batch_size = 2

# load pretrained weights
load_from = 'pretrain/cascade_mask_rcnn_r50_fpn_coco-20e_20e_nuim_20201009_124951-40963960.pth'
#load_from = 'pretrain/rwhi-v7-e2.pth'
revise_keys = [('backbone', 'img_backbone')]

# resume the last training
resume_from = None

# checkpointing
default_hooks = dict(
    checkpoint = None
)

checkpoint_config = dict(interval=1, max_keep_ckpts=4)

# logging
log_config = dict(
    interval=1,
    hooks=[
        dict(type='MyTextLoggerHook', interval=50, reset_flag=True),
        dict(type='MyTensorboardLoggerHook', interval=500, reset_flag=True)
    ]
)

# evaluation
eval_config = dict(interval=2)

# other flags
debug = False
find_unused_parameters = False

# DDP 配置
# 注意：static_graph 不适用于此模型，因为计算图可能根据雷达数据变化
# 通过 dummy sum 确保所有参数在每次迭代中都参与计算图
static_graph = False

custom_hooks = [
    dict(
        type='SequentialControlHook',
        start_epoch=18,
    ),
]
