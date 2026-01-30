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
num_rwhi = 750
use_rwhi = True
use_alpha = True
alpha_const = 0.7
rwhi_gate_init = 0.2
rwhi_gate_const = 0.7
# 控制 RWHI 是否影响 query 特征（默认 True，保持旧行为）
rwhi_affect_query = True

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
    
    # AlphaMLP 参数（仅 RWHI 内部雷达置信度）
    alpha_mlp_in_dim=3,   # 输入维度 [log1p(rcs), d_norm, v_norm]
    alpha_mlp_hidden=32,  # 隐藏层维度
    alpha_init_bias=1.0,  # 初始偏置，使初始 α ≈ 0.73
    alpha_const=alpha_const,  # use_alpha=False 时的常数 α
    use_alpha=use_alpha,      # 控制是否启用 AlphaMLP
    st_tau=0.05,             # Straight-Through 可微 Top-K 温度

    # 其他
    num_clusters=num_clusters,  # 距离层数量 (与 RaCFormer 一致)
    max_points=5000,      # 最大雷达点数
    enabled=True,         # 是否启用 RWHI
    num_rwhi=num_rwhi,    # 雷达引导锚点数量
    enable_diverse_topk=True,
    coarse_factor=4,
    max_per_cell=5,
)
# ============ RWHI v7 配置结束 ============

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
    with_cp=True)

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
        use_rwhi=use_rwhi,                    # 启用 RWHI
        use_alpha=use_alpha,              # 是否启用 RWHI 内部 α（雷达置信度）
        rwhi_gate_init=rwhi_gate_init,    # 可学习门控初值
        rwhi_gate_const=rwhi_gate_const,  # 固定门控（备用）
        rwhi_affect_query=rwhi_affect_query,  # 从 config 控制是否启用
        rwhi_cfg=rwhi_cfg,                # RWHI 配置
        polar_radius=R_MAX,               # 全链路统一 polar_radius
        # ============ RWHI v7 配置结束 ============
        
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
    workers_per_gpu=4,
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
        'rwhi_module': dict(lr_mult=1.0),
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
batch_size = 4

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

custom_hooks = [
    dict(
        type='SequentialControlHook',
        start_epoch=18,
    ),
]

