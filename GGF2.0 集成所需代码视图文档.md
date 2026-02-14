# GGF2.0 集成所需代码视图文档

> 本文档系统分析了「已集成 RWHI 的 RaCFormer 改进版」代码，目的是找出未来集成 **GGF2.0 模块** 所需要依赖的所有关键接口、数据流和张量格式。

---

## 一、整体结构与入口文件

### 1.1 项目地图

| 功能 | 文件路径 | 类/函数名 |
|------|----------|-----------|
| **模型主入口** | `models/racformer.py` | `RaCFormer` |
| **检测头** | `models/racformer_head.py` | `RaCFormer_head` |
| **Transformer** | `models/racformer_transformer.py` | `RaCFormerTransformer`, `RaCFormerTransformerDecoder`, `RaCFormerTransformerDecoderLayer` |
| **RWHI模块** | `models/rwhi.py` | `RWHIModule`, `AlphaMLP`, `AlphaEncoder`, `build_rwhi()` |
| **坐标工具** | `models/bbox/utils.py` | `theta_d2xy_coods()`, `xy2theta_d_coods()`, `R_MAX`, `MAP_SIZE` |
| **训练入口** | `train.py` | `main()` |
| **配置文件** | `configs/racformer_with_rwhi.py` | 完整配置 |

### 1.2 框架基座

- **基于 mmdet3d**，继承自 `MVXTwoStageDetector`
- **模型注册方式**：
  ```python
  @DETECTORS.register_module()   # RaCFormer
  @HEADS.register_module()        # RaCFormer_head
  @TRANSFORMER.register_module() # RaCFormerTransformer
  ```
- **构建调用链**：
  ```python
  # train.py line 122
  model = build_model(cfgs.model)  # 通过 mmdet3d.models.build_model
  ```

---

## 二、RaCFormer 主干结构梳理

### 2.1 核心子模块位置

```
models/racformer.py::RaCFormer
├── img_backbone        # ResNet50, 来自 mmdet
├── img_neck            # FPN, 输出 4 级特征
├── img_lss_neck        # CustomFPN, 用于 LSS
├── img_lss_view_transformer  # LSSViewTransformerBEVDepth_racformer
├── radar_voxel_layer   # Voxelization
├── radar_voxel_encoder # PillarFeatureNet
├── radar_middle_encoder# PointPillarsScatter
├── radar_bev_conv      # 3层Conv2d, 输出 embed_dims=256
└── pts_bbox_head       # RaCFormer_head
    ├── rwhi_module     # RWHIModule (RWHI v7)
    ├── transformer     # RaCFormerTransformer
    │   └── decoder     # RaCFormerTransformerDecoder
    │       └── decoder_layer  # RaCFormerTransformerDecoderLayer
    │           ├── self_attn      # ScaleAdaptiveSelfAttention
    │           ├── sampling       # RaCFormerSampling (图像采样)
    │           ├── sampling_lss_bev   # BEVSampling (LSS BEV)
    │           ├── sampling_radar_bev # BEVSampling (雷达 BEV)
    │           ├── mixing         # AdaptiveMixing
    │           └── fusion         # nn.Linear(embed_dims*3, embed_dims)
    └── cls_branch/reg_branch  # 分类/回归头
```

### 2.2 Forward 数据流（简易版）

```
Input Batch
    │
    ├─► extract_img_feat(img) ──────────────────────────────────────┐
    │       ├── img_backbone(img) → img_feats                       │
    │       ├── img_neck(img_feats) → img_feats_fpn [B,NT,C,H,W]    │
    │       └── img_lss_neck() → img_lss_feats                      │
    │                                                               │
    ├─► extract_pts_feat(radar_points) ─────────────────────────────┤
    │       ├── radar_voxelize() → voxels, coors                    │
    │       ├── radar_voxel_encoder() → pillar features             │
    │       ├── radar_middle_encoder() → rad_bev_feas [B,64,128,128]│
    │       └── radar_bev_conv() → radar_bev_feats [B,T,256,128,128]│
    │                                                               │
    └─► img_lss_view_transformer() ─────────────────────────────────┤
            └── bev_feats [B,T,256,128,128]                         │
                                                                    │
    ┌───────────────────────────────────────────────────────────────┘
    ▼
pts_bbox_head.forward(img_feats, bev_feats, radar_bev_feats, img_metas,
                      radar_points, radar_mask)
    │
    ├─► _prepare_query_bbox() ──────────────────────────────────────┐
    │       └── rwhi_module(radar_points, radar_mask)               │
    │               → query_bbox [B,900,10], alpha_values [B,900,1] │
    │                                                               │
    ├─► _prepare_query_feat() ──────────────────────────────────────┤
    │       └── query_feat [B,900,256]                              │
    │                                                               │
    └─► transformer.forward() ──────────────────────────────────────┤
            │                                                       │
            └─► decoder.forward() (6 layers)                        │
                    │                                               │
                    ├── self_attn(query_bbox, query_feat)           │
                    ├── sampling_radar_bev() → query_radar_feat     │
                    ├── sampling_lss_bev() → query_lss_feat         │
                    ├── sampling() → sampled_feat (from images)     │
                    ├── mixing(sampled_feat, query_feat)            │
                    ├── fusion(concat[query, radar, lss])           │
                    ├── ffn()                                       │
                    └── cls_branch(), reg_branch() → cls, bbox      │
                                                                    │
    Output: cls_scores [L,B,Q,10], bbox_preds [L,B,Q,10]            ▼
```

### 2.3 关键张量 Shape 汇总

| 张量名 | Shape | 含义 |
|--------|-------|------|
| `img` | `[B, NT, C, H, W]` | 多相机多帧图像，NT=6*8=48 |
| `img_feats_fpn` | `List[[B,NT,C,H,W]]` | 4 级 FPN 特征 |
| `bev_feats` (LSS) | `[B, T, 256, 128, 128]` | LSS BEV 特征 |
| `radar_bev_feats` | `[B, T, 256, 128, 128]` | 雷达 BEV 特征 |
| `radar_points` | `List[[N, 7]]` | 雷达点云 (x,y,z,rcs,vx,vy,timestamp) |
| `query_bbox` | `[B, 900, 10]` | Query bbox: (θ,d,z,w,l,h,sin,cos,vx,vy) |
| `query_feat` | `[B, 900, 256]` | Query 特征 |
| `reference_points` | 从 `query_bbox[:,:,:2]` 获取 | 极坐标参考点 (θ, d) |

---

## 三、RWHI 相关实现：几何场与 Query 初始化

### 3.1 RWHI 实现文件/类

```
models/rwhi.py
├── RWHIModule (BaseModule)     # 核心模块
├── AlphaMLP (nn.Module)        # 点置信度预测
├── AlphaEncoder (nn.Module)    # α 特征编码
└── build_rwhi()                # 工厂函数
```

### 3.2 雷达打分场构建流程

**核心函数调用链（`RWHIModule.forward()`）**：

```python
# models/rwhi.py line 733-825

def forward(self, radar_points, radar_mask):
    # 1. 计算每个点的 α 置信度
    f_in = self._prepare_alpha_input(radar_points, radar_mask)  # [B,M,3]
    alpha = self.alpha_mlp(f_in)  # [B,M,1]
    
    # 2. 计算 I_radar = α * log(1 + σ * w_v * w_d)
    i_radar = self._compute_i_radar(radar_points, radar_mask, alpha)  # [B,M]
    
    # 3. Scatter-Add 聚合到 BEV 网格
    i_radar_map = self._scatter_to_bev(radar_points, i_radar, ...)  # [B,1,H,W]
    
    # 4. 构建三层打分场
    S_final = self._build_score_map(i_radar_map, ...)  # [B,1,H,W]
    
    # 5. Top-K 选取锚点
    anchors, alpha_topk = self._topk_to_anchors(S_final, alpha_map, ...)
    
    return anchors, alpha_values  # [B,K,10], [B,K,1]
```

**打分公式实现**（`_build_score_map()`, line 514-564）：

```python
# S(x,y) = C_base + I_radar(x,y) + ε*J(x,y)
S = base_field + i_radar_map + self.epsilon * self.jitter_field

# S_final = S + γ*(Diffusion(S) - S)
S_pool = self.diffusion(S)  # MaxPool2d 或 AvgPool2d
S_final = S + self.diffusion_gamma * (S_pool - S)
S_final = S_final.clamp(max=self.diffusion_s_max)
```

**I_radar 计算**（`_compute_i_radar()`, line 429-466）：

```python
# w_v = 1 + β * sigmoid(|v|/v_ref)
w_v = 1.0 + self.beta * torch.sigmoid(v_r.abs() / self.v_ref)

# w_d = min((d/d_ref)^λ, w_d_max)
w_d = (dist / self.d_ref).pow(self.d_lambda).clamp(max=self.w_d_max)

# I_radar = α * log(1 + σ * w_v * w_d)
i_radar = alpha * torch.log(1.0 + sigma_proc * w_v * w_d)
```

### 3.3 数值刻度与配置项

```python
# configs/racformer_with_rwhi.py line 59-113
rwhi_cfg = dict(
    # 打分场参数
    base_bias=1.0,        # C_base (背景基础分)
    epsilon=0.01,         # 微扰系数
    diffusion_gamma=0.2,  # 扩散系数 λ
    diffusion_s_max=5.0,  # 得分上限裁剪
    
    # I_radar 参数
    d_ref=30.0,           # 距离权重参考值
    w_d_max=2.5,          # 距离权重上限
    ...
)
```

**目标范围**：
- 背景区域：S ≈ `base_bias` = 1.0
- 雷达覆盖区域：S ≈ 1.0 + I_radar（典型 2.0～4.0）
- 上限：S_max = 5.0

### 3.4 Query 初始化（Top-K 选点）

**核心函数**（`_topk_to_anchors()`, line 652-715）：

```python
def _topk_to_anchors(self, S, alpha_map, batch_size, device, topk_k=None):
    K = topk_k if topk_k is not None else self.num_query
    
    # Top-K 选取
    S_flat = S.view(batch_size, -1)  # [B, H*W]
    topk_idx = self._select_topk_indices(S_flat, H, W, K)  # [B, K]
    
    # 获取 Top-K 位置的物理坐标
    topk_xy = self.grid_xy[topk_idx]  # [B, K, 2] 物理坐标
    
    # 转换为极坐标
    theta_d = self._xy_to_theta_d(topk_xy)  # [B, K, 2]
    
    # 组装 10 维 bbox
    anchors = torch.cat([
        theta_d,    # [B,K,2] (θ, d)
        z,          # [B,K,1]
        w, l, h,    # [B,K,3] 尺寸
        sin, cos,   # [B,K,2] 朝向
        vx, vy      # [B,K,2] 速度
    ], dim=-1)  # [B,K,10]
```

**reference_points 存储与使用**：

- **存储格式**：`query_bbox[:,:,0:2]` = `(θ_norm, d_norm)`，均在 `[0,1]` 范围内归一化
- **物理坐标转换**：通过 `theta_d2xy_coods()` 转为笛卡尔坐标
- **在 Decoder 中使用**：

```python
# models/racformer_transformer.py line 250-290
def forward(self, query_bbox, query_feat, ...):
    # 位置编码
    query_pos = self.position_encoder(query_bbox[..., :3])  # [θ, d, z]
    query_feat = query_feat + query_pos
    
    # 传给各 sampling 模块
    sampled_feat = self.sampling(query_bbox, query_feat, mlvl_feats, ...)
    query_radar_feat = self.sampling_radar_bev(query_bbox, query_feat, radar_bev_feats, ...)
```

### 3.5 RWHI 张量 Shape 与 BEV 分辨率关系

| 张量 | Shape | 说明 |
|------|-------|------|
| `S_final` (打分图) | `[B, 1, 100, 100]` | `bev_grid_size=100` |
| `I_radar_map` | `[B, 1, 100, 100]` | 雷达强度图 |
| `alpha_map` | `[B, 1, 100, 100]` | 置信度图 |
| `grid_xy` (buffer) | `[10000, 2]` | 物理坐标 |

**BEV 网格映射**：
```python
# RWHI 使用 100x100 网格，覆盖 [-51.2, 51.2]
grid_resolution = 102.4 / 100 = 1.024 m/cell
# 主干 BEV 使用 128x128 网格
main_bev_resolution = 102.4 / 128 = 0.8 m/cell
```

---

## 四、雷达分支与高斯/BEV 表示

### 4.1 雷达数据加载

**数据格式**（`loaders/nuscenes_dataset.py`）：

```python
# get_nu_radar() 返回
radar_points: [18, N]  # 原始 nuscenes 格式
# 维度: x, y, z, dyn_prop, id, rcs, vx, vy, vx_comp, vy_comp, ...

# Pipeline 处理后 (loaders/pipelines/loading.py)
radar_points: List[[N, 7]]  # 每帧
# 维度: x, y, z, rcs, v_r (径向速度), ... 
```

### 4.2 雷达 Encoder 流程

```python
# models/racformer.py line 127-146

def extract_pts_feat(self, radar_points):
    # 1. Voxelization
    voxels, num_points, coors = self.radar_voxelize(radar_points)
    # voxels: [N_voxels, max_points, 7]
    
    # 2. PillarFeatureNet
    radar_features = self.radar_voxel_encoder(voxels, num_points, coors)
    # radar_features: [N_voxels, 64]
    
    # 3. PointPillarsScatter
    rad_bev_feas = self.radar_middle_encoder(radar_features, coors, batch_size)
    # rad_bev_feas: [B, 64, 128, 128]
    
    # 4. Conv layers
    rad_bev_feas = self.radar_bev_conv(rad_bev_feas)
    # rad_bev_feas: [B, 256, 128, 128]
    
    return rad_bev_feas
```

**配置参数**：

```python
# configs/racformer_with_rwhi.py line 192-209
radar_voxel_layer=dict(
    max_num_points=10,
    voxel_size=[0.8, 0.8, 8],  # 与 BEV 分辨率一致
    point_cloud_range=point_cloud_range,
)

radar_middle_encoder=dict(
    type='PointPillarsScatter',
    in_channels=64,
    output_shape=(128, 128),  # BEV 网格尺寸
)
```

### 4.3 雷达 BEV 特征图参数

| 参数 | 值 | 说明 |
|------|-----|------|
| Shape | `[B, T, 256, 128, 128]` | T=8 帧 |
| Stride | 0.8 m/pixel | 102.4m / 128 |
| 范围 | `[-51.2, 51.2]` x `[-51.2, 51.2]` m | 与 LSS BEV 对齐 |

### 4.4 ⚠️ 当前无高斯参数

**搜索结果**：代码中**没有**现成的高斯相关实现。
- 无 `Gaussian`, `PGE`, `sigma`, `cov`, `splat` 等关键字
- RWHI 仅使用 Scatter-Add 聚合，无高斯展开

**GGF2.0 插入点建议**：在 `RWHIModule._scatter_to_bev()` 或 `_compute_i_radar()` 处引入高斯参数。

---

## 五、Decoder / Ray-sampling / Deformable Attention 接口

### 5.1 Decoder 结构

```python
# models/racformer_transformer.py

class RaCFormerTransformerDecoderLayer:
    def forward(self, query_bbox, query_feat, mlvl_feats, lss_bev_feats, 
                radar_bev_feats, attn_mask, img_metas, layer=0):
        
        # 1. Position Encoding + Self-Attention
        query_pos = self.position_encoder(query_bbox[..., :3])  # [B,Q,256]
        query_feat = query_feat + query_pos
        query_feat = self.norm1(self.self_attn(query_bbox, query_feat, attn_mask))
        
        # 2. BEV Sampling (雷达 + LSS)
        query_radar_feat = self.sampling_radar_bev(query_bbox, query_feat, 
                                                    radar_bev_feats, img_metas, 
                                                    d_region=self.d_region_list[layer])
        query_lss_feat = self.sampling_lss_bev(query_bbox, query_feat,
                                                lss_bev_feats, img_metas,
                                                d_region=self.d_region_list[layer])
        
        # 3. Image Ray-Sampling
        sampled_feat = self.sampling(query_bbox, query_feat, mlvl_feats, 
                                      img_metas, d_region=self.d_region_list[layer])
        
        # 4. Adaptive Mixing + Fusion
        query_feat = self.norm2(self.mixing(sampled_feat, query_feat))
        query_feat = self.norm_fusion(self.fusion(
            torch.cat((query_feat, query_radar_feat, query_lss_feat), dim=-1)
        ))
        query_feat = self.norm3(self.ffn(query_feat))
        
        # 5. Classification + Regression
        cls_score = self.cls_branch(query_feat)
        bbox_pred = self.reg_branch(query_feat)
        bbox_pred = self.refine_bbox(query_bbox, bbox_pred)
        
        return query_feat, cls_score, bbox_pred
```

**关键张量 Shape**：

| 张量 | Shape | 位置 |
|------|-------|------|
| `query_feat` | `[B, Q, 256]` | Input/Output |
| `query_bbox` | `[B, Q, 10]` | Reference points |
| `sampled_feat` | `[B, Q, G, FP, C]` = `[B,Q,4,96,64]` | 图像采样后 |
| `query_radar_feat` | `[B, Q, 256]` | 雷达 BEV 采样后 |
| `query_lss_feat` | `[B, Q, 256]` | LSS BEV 采样后 |

### 5.2 Ray-Sampling 详解

**`RaCFormerSampling` 类**（line 349-438）：

```python
class RaCFormerSampling(BaseModule):
    def inner_forward(self, query_ray, query_feat, mlvl_feats, img_metas, d_region=0.1):
        # 输入
        # query_ray: [B, Q, 10] 极坐标 bbox
        # query_feat: [B, Q, C]
        # mlvl_feats: 4 级特征 [B*T*G, C, N, H, W]
        
        # 转为笛卡尔坐标
        query_bbox = theta_d2xy_coods(query_ray).clone()  # [B, Q, 10]
        
        # 计算 sampling offset (可学习)
        sampling_offset = self.sampling_offset(query_feat)  # [B, Q, D*G*P, 3]
        
        # 生成采样点
        sampling_points = make_sample_points(query_bbox, sampling_offset, self.pc_range)
        # sampling_points: [B, Q, T, G, P*D, 3]
        
        # 沿射线方向采样 (d_region 控制深度范围)
        sampling_points_d = torch.linspace(-d_region, d_region, self.depth_num)
        # 添加可学习的深度偏移
        sampling_points_d += (self.ray_points_offset(query_feat).sigmoid()*2-1) * d_region/self.depth_num/2
        
        # Multi-scale Multi-view Grid Sample
        sampled_feats = sampling_4d(sampling_points, mlvl_feats, scale_weights, ...)
        
        return sampled_feats  # [B, Q, G, FP, C]
```

**采样逻辑特点**：
- **Purely Learnable Offset**：通过 `self.sampling_offset` 学习偏移量
- **深度采样**：沿射线在 `[-d_region, d_region]` 范围内采样
- **已有几何先验**：`d_region_list = [0.08, 0.07, 0.06, 0.05, 0.04, 0.03]`，逐层缩小

### 5.3 BEVSampling（`BEVSelfAttention` 接口）

```python
# models/bev_self_attention.py

class BEVSelfAttention(BaseModule):
    def forward(self, query, value, sampling_locations, attention_weights, ...):
        # query: [B, Q, C]
        # value: [B*T, C, H, W] BEV 特征
        # sampling_locations: [B, Q, H, T, L, P, 2] 采样位置
        # attention_weights: [B, Q, H, T, L, P] 注意力权重
        
        # 使用 MultiScaleDeformableAttnFunction
        output = MultiScaleDeformableAttnFunction.apply(
            value, spatial_shapes, level_start_index,
            sampling_locations, attention_weights, self.im2col_step
        )
        
        return self.output_proj(output) + identity
```

### 5.4 🎯 GGF 改动建议

**如果要在 Decoder 中改动 ray-sampling / 加几何 bias**：

| 目标 | 修改位置 | 接入张量 |
|------|----------|----------|
| **几何引导深度采样** | `RaCFormerSampling.inner_forward()` line 406-411 | `sampling_points_d` |
| **注意力权重调制** | `BEVSelfAttention.forward()` line 179-186 | `attention_weights` |
| **采样偏移调制** | `RaCFormerSampling.inner_forward()` line 383-384 | `sampling_offset` |

---

## 六、BEV 特征、跨模态融合（Adaptive Mixer）接口

### 6.1 融合模块定位

```python
# models/racformer_transformer.py line 204
self.fusion = nn.Linear(embed_dims*3, embed_dims)  # 768 → 256
```

### 6.2 融合公式

```python
# line 267-268
# 三路 concat + MLP 融合
query_feat = self.norm_fusion(
    self.fusion(torch.cat((query_feat, query_radar_feat, query_lss_feat), dim=-1))
)
```

**输入张量**：

| 张量名 | Shape | 来源 |
|--------|-------|------|
| `query_feat` | `[B, Q, 256]` | 图像采样 → AdaptiveMixing |
| `query_radar_feat` | `[B, Q, 256]` | `sampling_radar_bev()` |
| `query_lss_feat` | `[B, Q, 256]` | `sampling_lss_bev()` |

**当前无几何权重/gate**：简单 concat + Linear，无条件融合。

### 6.3 AdaptiveMixing 详解

```python
# models/racformer_transformer.py line 560-627

class AdaptiveMixing(nn.Module):
    def inner_forward(self, x, query):
        # x: [B, Q, G, P, C] 采样特征
        # query: [B, Q, C] Query 特征
        
        # 生成 mixing 参数
        params = self.parameter_generator(query)  # [B*Q, G, M+S]
        M, S = params.split([self.m_parameters, self.s_parameters], 2)
        
        # Adaptive Channel Mixing
        out = torch.matmul(out, M)  # [B*Q, G, P, C'] @ [G, C, C']
        
        # Adaptive Point Mixing
        out = torch.matmul(S, out)  # [G, P', P] @ [G, P, C']
        
        return query + self.out_proj(out)
```

### 6.4 🎯 GGF Fusion 插入点建议

**最适合的位置**：在 `decoder_layer.forward()` 中，**cross-attn 输出后、FFN 前**：

```python
# 建议在 line 267 之后、line 269 之前插入 GGF Fusion
query_feat = self.norm2(self.mixing(sampled_feat, query_feat))
# ========== GGF Fusion 插入点 ==========
# query_feat = ggf_fusion(query_feat, query_radar_feat, query_lss_feat, geometry_field)
# =========================================
query_feat = self.norm_fusion(self.fusion(torch.cat(...)))
query_feat = self.norm3(self.ffn(query_feat))
```

---

## 七、坐标/张量约定与坐标系转换

### 7.1 坐标转换函数

```python
# models/bbox/utils.py

# 全局常量
R_MAX = 65.0      # 极坐标最大半径 (m)
MAP_SIZE = 102.4  # BEV 地图边长 (m)

def theta_d2xy_coods(theta_d_coords, map_size=102.4, r=65.0):
    """极坐标 (θ, d) → 笛卡尔坐标 (x, y)"""
    center = map_size / 2
    x = (center + d_norm * r * cos(θ * 2π)) / map_size  # → [0, 1]
    y = (center + d_norm * r * sin(θ * 2π)) / map_size  # → [0, 1]
    
def xy2theta_d_coods(xy_coords_norm, map_size=102.4, r=65.0):
    """笛卡尔坐标 (x, y) → 极坐标 (θ, d)"""
    θ = atan2(y - center, x - center) / 2π  # → [0, 1]
    d = sqrt((x-center)² + (y-center)²) / r  # → [0, ~1.1]
```

### 7.2 BEV 网格分辨率

| 组件 | 网格尺寸 | 分辨率 | 范围 |
|------|----------|--------|------|
| 雷达 BEV | 128 × 128 | 0.8 m/cell | [-51.2, 51.2] |
| LSS BEV | 128 × 128 | 0.8 m/cell | [-51.2, 51.2] |
| RWHI 打分图 | 100 × 100 | 1.024 m/cell | [-51.2, 51.2] |

**坐标映射公式**：
```python
# BEV index (u, v) → 物理坐标 (x, y)
x = pc_range[0] + (u + 0.5) * grid_resolution
y = pc_range[1] + (v + 0.5) * grid_resolution

# 物理坐标 (x, y) → BEV index (u, v)
u = (x - pc_range[0]) / grid_resolution
v = (y - pc_range[1]) / grid_resolution
```

### 7.3 Query reference_points 约定

- **格式**：归一化 `[0, 1]` 范围
- **极坐标**：`query_bbox[:,:,0]` = θ ∈ [0, 1)，`query_bbox[:,:,1]` = d ∈ (ε, 1-ε)
- **使用时转换**：`theta_d2xy_coods()` 转为归一化笛卡尔，再乘以 `pc_range` 得物理坐标

### 7.4 🎯 GGF 坐标对齐指南

**如果要从高斯场画 I_radar(x,y)**：
```python
# 1. 高斯中心 (x, y) 是物理坐标
# 2. 转换到 RWHI 网格 index
u = (x - pc_range[0]) / (map_size / bev_grid_size)  # bev_grid_size=100
v = (y - pc_range[1]) / (map_size / bev_grid_size)

# 3. 或转换到主干 BEV 网格
u = (x - pc_range[0]) / 0.8  # 128x128 网格
v = (y - pc_range[1]) / 0.8
```

**从 Query 坐标读取几何特征**：
```python
# query_bbox: [B, Q, 10] 极坐标格式
query_xy = theta_d2xy_coods(query_bbox)[:, :, :2]  # [B, Q, 2] 归一化 [0,1]
query_xy_physical = query_xy * map_size - map_size/2  # [B, Q, 2] 物理坐标

# 从几何场采样
grid = query_xy_physical / (map_size / geometry_field_size) + geometry_field_size / 2
sampled_geometry = F.grid_sample(geometry_field, grid, ...)
```

---

## 八、总结 & 建议的插入点

### 8.1 GGF2.0 依赖的关键代码位置

| 功能 | 文件 | 类/函数 | 输入 Shape | 输出 Shape |
|------|------|---------|------------|------------|
| **Query 锚点生成** | `models/rwhi.py` | `RWHIModule.forward()` | `[B,M,7]`, `[B,M]` | `[B,K,10]`, `[B,K,1]` |
| **打分场构建** | `models/rwhi.py` | `_build_score_map()` | `[B,1,H,W]` | `[B,1,H,W]` |
| **雷达 BEV 编码** | `models/racformer.py` | `extract_pts_feat()` | `List[[N,7]]` | `[B,T,256,128,128]` |
| **图像采样** | `models/racformer_transformer.py` | `RaCFormerSampling.inner_forward()` | `[B,Q,10]`, `[B,Q,C]` | `[B,Q,G,FP,C]` |
| **BEV 采样** | `models/racformer_transformer.py` | `BEVSampling.inner_forward()` | `[B,Q,10]`, `[B,T,C,H,W]` | `[B,Q,C]` |
| **模态融合** | `models/racformer_transformer.py` | `decoder_layer.forward()` line 268 | 3×`[B,Q,256]` | `[B,Q,256]` |
| **坐标转换** | `models/bbox/utils.py` | `theta_d2xy_coods()`, `xy2theta_d_coods()` | `[...,2]` | `[...,2]` |

### 8.2 建议的 GGF2.0 插入点（优先级排序）

#### 插入点 1：雷达几何场替换 RWHI 打分图

**位置**：`models/rwhi.py::RWHIModule._build_score_map()` 或 `_compute_i_radar()`

**改动**：
```python
# 在 _compute_i_radar() 中引入高斯参数
def _compute_i_radar_with_gaussian(self, radar_points, radar_mask, alpha, gaussian_params):
    # gaussian_params: [B, M, 5] = (μx, μy, σx, σy, scale)
    # 使用高斯展开替代 Scatter-Add
    i_radar_map = gaussian_splat(gaussian_params, grid_xy)
    return i_radar_map
```

#### 插入点 2：几何引导 Ray-Sampling 深度采样

**位置**：`models/racformer_transformer.py::RaCFormerSampling.inner_forward()` line 406-411

**改动**：
```python
# 引入几何场作为深度先验
def inner_forward(self, query_ray, query_feat, mlvl_feats, img_metas, 
                  geometry_field=None, d_region=0.1):
    ...
    # 原始: 均匀深度采样
    sampling_points_d = torch.linspace(-d_region, d_region, self.depth_num)
    
    # GGF: 几何引导的非均匀采样
    if geometry_field is not None:
        geometry_depth_prior = sample_geometry_depth(geometry_field, query_bbox)
        sampling_points_d = modulate_depth_sampling(sampling_points_d, geometry_depth_prior)
```

#### 插入点 3：Query 级 GGF Fusion

**位置**：`models/racformer_transformer.py::RaCFormerTransformerDecoderLayer.forward()` line 267-268 之间

**改动**：
```python
# 在 mixing 后、fusion 前插入 GGF Fusion
query_feat = self.norm2(self.mixing(sampled_feat, query_feat))

# ========== GGF Fusion 插入 ==========
if hasattr(self, 'ggf_fusion') and geometry_field is not None:
    geometry_weights = self.geometry_gate(geometry_field, query_bbox)  # [B, Q, 1]
    query_feat = query_feat + geometry_weights * self.geometry_proj(
        sample_geometry_features(geometry_field, query_bbox)
    )
# =====================================

query_feat = self.norm_fusion(self.fusion(torch.cat(...)))
```

#### 插入点 4：替换整个雷达 BEV 编码管线

**位置**：`models/racformer.py::RaCFormer.extract_pts_feat()`

**改动**：
```python
def extract_pts_feat(self, radar_points, use_ggf=True):
    if use_ggf:
        # GGF2.0: 高斯场 → BEV
        gaussian_params = self.ggf_encoder(radar_points)
        rad_bev_feas = self.gaussian_to_bev(gaussian_params)
    else:
        # 原始: Pillar → BEV
        rad_bev_feas = self.pillar_encoder(radar_points)
    
    rad_bev_feas = self.radar_bev_conv(rad_bev_feas)
    return rad_bev_feas
```

### 8.3 快速检查清单

在开始 GGF2.0 集成前，确保：

- [ ] `R_MAX = 65.0` 在 GGF 模块中保持一致
- [ ] BEV 网格分辨率选择：100×100 (RWHI) 或 128×128 (主干)
- [ ] `pc_range = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]` 与 GGF 坐标系对齐
- [ ] Query reference_points 使用归一化极坐标 `(θ, d)` ∈ `[0,1]`
- [ ] 高斯参数张量 dtype 与 AMP/FP16 兼容
- [ ] 新增模块需要在 `models/__init__.py` 中注册

---

## 附录 A：核心文件快速索引

| 文件路径 | 核心内容 |
|----------|----------|
| `models/racformer.py` | 主检测器，特征提取，forward 入口 |
| `models/racformer_head.py` | Query 初始化，RWHI 集成，loss 计算 |
| `models/racformer_transformer.py` | Decoder 层，采样模块，融合模块 |
| `models/rwhi.py` | RWHI v7 完整实现，打分场，Top-K |
| `models/bbox/utils.py` | 坐标转换，R_MAX/MAP_SIZE 常量 |
| `models/sparsebev_sampling.py` | 4D 采样核心函数 |
| `models/bev_self_attention.py` | BEV Deformable Attention |
| `models/necks/view_transformer_racformer.py` | LSS 视角转换 |
| `configs/racformer_with_rwhi.py` | 完整配置文件 |
| `train.py` | 训练入口 |

---

## 附录 B：数据流图（ASCII Art）

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              INPUT BATCH                                     │
│  img: [B,48,3,H,W]  radar_points: List[[N,7]]  radar_depth/rcs: [B,6,T,H,W] │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
         ┌────────────────────────────┼────────────────────────────┐
         │                            │                            │
         ▼                            ▼                            ▼
┌─────────────────┐        ┌─────────────────────┐      ┌─────────────────┐
│  Image Branch   │        │    Radar Branch     │      │   LSS Branch    │
│  ───────────    │        │    ────────────     │      │   ──────────    │
│  img_backbone   │        │  radar_voxelize     │      │ view_transformer│
│  img_neck (FPN) │        │  pillar_encoder     │      │  + depth_net    │
│  img_lss_neck   │        │  scatter → BEV      │      │                 │
└────────┬────────┘        └──────────┬──────────┘      └────────┬────────┘
         │                            │                            │
         │                            │                            │
         ▼                            ▼                            ▼
┌─────────────────┐        ┌─────────────────────┐      ┌─────────────────┐
│ img_feats_fpn   │        │  radar_bev_feats    │      │   bev_feats     │
│ [B,NT,C,H,W]×4  │        │ [B,T,256,128,128]   │      │ [B,T,256,128,128]│
└────────┬────────┘        └──────────┬──────────┘      └────────┬────────┘
         │                            │                            │
         └────────────────────────────┼────────────────────────────┘
                                      │
                                      ▼
                        ┌─────────────────────────┐
                        │   pts_bbox_head.forward │
                        │   ─────────────────────  │
                        │                         │
                        │  ┌───────────────────┐  │
                        │  │  RWHI Module      │  │
                        │  │  radar_points ──► │  │
                        │  │  query_bbox [B,Q,10]│ │
                        │  │  alpha [B,Q,1]    │  │
                        │  └───────────────────┘  │
                        │           │             │
                        │           ▼             │
                        │  ┌───────────────────┐  │
                        │  │  Transformer      │  │
                        │  │  Decoder (×6)     │  │
                        │  │  ─────────────    │  │
                        │  │  • self_attn      │  │
                        │  │  • img_sampling   │  │
                        │  │  • radar_bev_samp │  │
                        │  │  • lss_bev_samp   │  │
                        │  │  • mixing         │  │
                        │  │  • fusion         │  │
                        │  │  • ffn            │  │
                        │  │  • cls/reg branch │  │
                        │  └───────────────────┘  │
                        │           │             │
                        └───────────┼─────────────┘
                                    │
                                    ▼
                        ┌─────────────────────────┐
                        │        OUTPUT           │
                        │  cls_scores [L,B,Q,10]  │
                        │  bbox_preds [L,B,Q,10]  │
                        └─────────────────────────┘
```

---

> **文档生成时间**：2026-01-27
> 
> **适用版本**：RaCFormer + RWHI v7
> 
> **目标**：GGF2.0 模块集成
