# 🔄 RaCFormer 模型切换指南 (Model Switching Guide)

本文档说明如何在**原始 RaCFormer 基线**和**增强版 (RHGM + RadarBEVNet)** 之间进行切换。

---

## 📊 模型版本对比

| 版本 | 雷达点云增强 | 雷达编码器 | 配置文件 |
|------|-------------|-----------|---------|
| **原始基线** | ❌ 无 | PillarFeatureNet | `racformer_r50_nuimg_704x256_f8.py` |
| **+RHGM** | ✅ RHGM | PillarFeatureNet | 自定义配置 |
| **+RadarBEVNet** | ❌ 无 | RadarBEVNet | 自定义配置 |
| **完整增强版** | ✅ RHGM | RadarBEVNet | `racformer_with_rhgm_radarbevnet.py` |

---

## 🎯 快速切换方法

### 方法 1：使用不同配置文件（推荐）

```bash
# 原始基线
python train.py --config configs/racformer_r50_nuimg_704x256_f8.py

# 或使用显式基线配置
python train.py --config configs/racformer_baseline.py

# 完整增强版 (RHGM + RadarBEVNet)
python train.py --config configs/racformer_with_rhgm_radarbevnet.py
```

### 方法 2：修改配置文件中的开关

在 `configs/racformer_with_rhgm_radarbevnet.py` 中修改以下参数：

```python
model = dict(
    type='RaCFormer',
    # ...其他配置...
    
    # ============ 模块开关 ============
    use_rhgm=True,              # True=启用RHGM, False=禁用
    use_radar_bev_net=True,     # True=使用RadarBEVNet, False=使用原始PillarFeatureNet
    
    # ...
)
```

---

## 🔧 详细配置说明

### 1. 原始 RaCFormer 基线

**特征**：
- 雷达编码器：`PillarFeatureNet` → `PointPillarsScatter`
- 无虚拟点云增强
- 这是 CVPR 2025 论文的原始实现

**配置方式**：

```python
# 配置文件中不添加 RHGM 和 RadarBEVNet 相关参数
# 或显式设置：
model = dict(
    type='RaCFormer',
    use_rhgm=False,           # ❌ 禁用RHGM
    use_radar_bev_net=False,  # ❌ 使用原始编码器
    # ...
)
```

**关键代码位置** (`models/racformer.py`)：
```python
# 第54-57行：模块开关参数
use_rhgm=False,              # 默认禁用
use_radar_bev_net=False,     # 默认禁用

# 第158-161行：使用原始编码器
if not use_radar_bev_net:
    self.radar_voxel_encoder = builder.build_voxel_encoder(radar_voxel_encoder)
    self.radar_middle_encoder = builder.build_middle_encoder(radar_middle_encoder)
```

---

### 2. 启用 RHGM 模块

**功能**：雷达-相机混合点云生成，增加前景区域的雷达点密度

**来源**：[HGSFusion](https://arxiv.org/abs/2406.04083)

**配置方式**：

```python
# 第1步：定义 RHGM 配置
rhgm_cfg = dict(
    num_virtual_points=100,      # 每个前景区域生成的虚拟点数
    dist_thresh=3000,            # 距离阈值 (mm)
    gauss_sigma=7,               # 高斯分布标准差
    gauss_kernel_size=51,        # 高斯核大小
    gauss_uniform_ratio=[1, 4],  # 高斯:均匀采样比例
    input_channels=7,            # 输入通道数
    output_channels=7,           # 输出通道数
    enabled=True,                # 模块内部开关
)

# 第2步：在模型配置中启用
model = dict(
    type='RaCFormer',
    use_rhgm=True,               # ✅ 启用RHGM
    rhgm_cfg=rhgm_cfg,           # 配置字典
    # ...
)
```

**关键代码位置** (`models/racformer.py`)：
```python
# 第92-105行：RHGM 初始化
if use_rhgm:
    self.rhgm = RHGMWrapper(rhgm_cfg=rhgm_cfg)

# 第241-251行：RHGM 调用
if self.use_rhgm and semantic_masks is not None:
    hybrid_points, foreground_masks = self.rhgm(
        radar_points,
        semantic_masks_list=semantic_masks,
        # ...
    )
    radar_points = hybrid_points  # 使用混合点云
```

---

### 3. 启用 RadarBEVNet 模块

**功能**：双流雷达骨干网络 + RCS-aware BEV 编码器

**来源**：[RCBEVDet](https://arxiv.org/abs/2403.01578)

**配置方式**：

```python
# 第1步：定义 RadarBEVNet 配置
radar_bev_net_cfg = dict(
    in_channels=7,                    # 输入通道数
    feat_channels=[64],               # ⚠️ 必须与 radar_middle_encoder.in_channels 匹配
    voxel_size=[0.8, 0.8, 8],         # 体素大小
    point_cloud_range=[-51.2, -51.2, -5.0, 51.2, 51.2, 3.0],
    with_distance=False,
    with_pos_embed=True,              # 使用位置编码
    return_rcs=False,                 # 是否返回 RCS 特征
    drop=0.0,
)

# 第2步：在模型配置中启用
model = dict(
    type='RaCFormer',
    use_radar_bev_net=True,           # ✅ 启用 RadarBEVNet
    radar_bev_net_cfg=radar_bev_net_cfg,
    # ⚠️ 仍需保留原始编码器配置（模型内部需要）
    radar_voxel_encoder=dict(...),
    radar_middle_encoder=dict(in_channels=64, ...),  # 必须与 feat_channels[-1] 一致
    # ...
)
```

**关键代码位置** (`models/racformer.py`)：
```python
# 第116-130行：RadarBEVNet 初始化
if use_radar_bev_net:
    self.radar_bev_net = RadarBEVNet(**radar_bev_net_cfg)

# 第265-280行：RadarBEVNet 调用
if self.use_radar_bev_net:
    radar_bev_output = self.radar_bev_net(voxels, num_points, coors)
    # ...
else:
    radar_features = self.radar_voxel_encoder(voxels, num_points, coors)
```

---

## 📁 代码文件说明

| 文件 | 说明 |
|------|------|
| `models/racformer.py` | 主检测器，包含模块切换逻辑 |
| `models/rhgm.py` | RHGM 模块实现 (来自 HGSFusion) |
| `models/radar_bev_net.py` | RadarBEVNet 模块实现 (来自 RCBEVDet) |
| `models/__init__.py` | 模块注册 |
| `configs/racformer_r50_nuimg_704x256_f8.py` | 原始基线配置 |
| `configs/racformer_baseline.py` | 显式基线配置 (用于消融实验) |
| `configs/racformer_with_rhgm_radarbevnet.py` | 完整增强版配置 |

---

## ⚠️ 维度匹配注意事项

使用 RadarBEVNet 时，必须确保维度匹配：

```
RadarBEVNet.feat_channels[-1] == radar_middle_encoder.in_channels
```

**示例**：
```python
# ✅ 正确：两者都是 64
radar_bev_net_cfg = dict(feat_channels=[64], ...)
radar_middle_encoder = dict(in_channels=64, ...)

# ❌ 错误：维度不匹配
radar_bev_net_cfg = dict(feat_channels=[64, 128], ...)  # 输出 128
radar_middle_encoder = dict(in_channels=64, ...)        # 期望 64
```

如果维度不匹配，模型会自动创建适配层并打印警告信息。

---

## 🧪 消融实验配置表

| 实验 | use_rhgm | use_radar_bev_net | 预期效果 |
|------|----------|-------------------|---------|
| Baseline | `False` | `False` | 基准性能 |
| +RHGM | `True` | `False` | 点云密度↑, 小目标检测↑ |
| +RadarBEVNet | `False` | `True` | 特征表达能力↑ |
| +Both | `True` | `True` | 最佳性能 |

---

## 🚀 快速开始脚本

```bash
#!/bin/bash

# 运行消融实验
CONFIG_DIR="configs"

# 基线
echo "Running Baseline..."
python train.py --config ${CONFIG_DIR}/racformer_r50_nuimg_704x256_f8.py

# 完整增强版
echo "Running Enhanced Version..."
python train.py --config ${CONFIG_DIR}/racformer_with_rhgm_radarbevnet.py
```

---

## 📚 相关文档

- [README.md](./README.md) - 项目总览
- [代码修改总览.md](../代码修改总览.md) - 所有代码修改说明
- [运行指南.md](../运行指南.md) - 详细运行教程

---

## 🎯 RWHI Query初始化策略

### 概述

RWHI (RCS-Weighted Hybrid Anchor Initialization) 是一种新型Query初始化策略，利用雷达点云信息动态生成检测锚点。

### 架构设计

```
Query初始化 = 安全流 (30%) + 显著流 (70%)

安全流 (Safety Stream):
  - 基于逆深度分布 (1/r)
  - 近场密集，远场稀疏
  - 预计算，不依赖输入

显著流 (Saliency Stream):
  1. 权重计算: W = log(1 + ReLU(RCS)) × (1 + α × sigmoid(|v_r|))
  2. Scatter-Add体素化: 将权重累加到BEV网格
  3. MaxPool扩散: 模拟雷达角分辨率不确定性
  4. TopK采样: 选取高响应区域
  5. 多高度假设: z=0m 和 z=1.5m
```

### 配置方法

```python
# 配置文件中
model = dict(
    pts_bbox_head=dict(
        use_rwhi=True,  # ✅ 启用RWHI
        rwhi_cfg=dict(
            safety_ratio=0.3,          # 安全流占比
            bev_grid_size=100,         # BEV网格分辨率
            safety_max_range=30.0,     # 安全流最大范围(米)
            velocity_alpha=0.5,        # 速度权重系数
            height_hypotheses=(0.0, 1.5),  # 高度假设
            diffusion_kernel_size=3,   # 扩散核大小
            enabled=True,
        ),
    ),
)
```

### 代码文件

| 文件 | 说明 |
|------|------|
| `models/rwhi.py` | RWHI模块实现 |
| `models/racformer_head.py` | 集成RWHI的检测头 |
| `configs/racformer_with_rwhi.py` | RWHI配置示例 |

### 实现特点

1. **全向量化**: 无Python for循环，使用 `scatter_add_`、`topk`、`index_select`
2. **TensorRT兼容**: 静态图友好，无动态shape操作
3. **GPU友好**: 所有操作在GPU上执行

---

*最后更新: 2025-12-29*

