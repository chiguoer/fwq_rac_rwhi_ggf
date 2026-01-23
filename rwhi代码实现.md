# RWHI v7 (Radar-Weighted Hybrid Initialization) 代码实现指南

> 本文档是经过多轮调试后总结的完整实现规范，供 AI 重新生成代码时参考。

---

## 目录

1. [核心设计概述](#1-核心设计概述)
2. [关键常量与规范](#2-关键常量与规范)
3. [RaCFormer 原始 Query 生成机制](#3-racformer-原始-query-生成机制)
4. [RWHI v7 核心算法](#4-rwhi-v7-核心算法)
5. [严重易错点（血泪教训）](#5-严重易错点血泪教训)
6. [文件结构与接口设计](#6-文件结构与接口设计)
7. [详细实现规范](#7-详细实现规范)
8. [配置文件规范](#8-配置文件规范)
9. [测试验证清单](#9-测试验证清单)

---

## 1. 核心设计概述

### 1.1 RWHI 的目标

将 RaCFormer 原有的**均匀极坐标分布锚点**替换为**雷达引导的动态锚点**，核心思想：

1. 构建 BEV 打分图 `S(x,y)`，雷达高置信区域得分高
2. Top-K 选取得分最高的 K 个位置作为 Query 锚点
3. 为每个锚点计算置信度 α，用于特征增强

### 1.2 三层打分场公式

```
S(x,y) = C_base + I_radar(x,y) + ε * J(x,y)
S_final = S + γ * (MaxPool3x3(S) - S)
```

其中：
- `C_base ≈ 1.0`：均匀安全层，确保无雷达区域也有基础分数
- `I_radar`：雷达增益层，真目标区域得分高
- `ε * J`：确定性微扰层（hash-based），打破平局
- 残差式空间扩散：邻域传播，不降低原有峰值

### 1.3 I_radar 公式

```
I_radar(点) = α * log(1 + σ_proc * w_v * w_d)
```

- `α ∈ (0, 1)`：点置信度，由 AlphaMLP 学习
- `σ_proc = max(0, rcs)`：RCS 预处理
- `w_v = 1 + β * sigmoid(|v|/v_ref)`：速度权重，动态目标增益
- `w_d = min((d/d_ref)^λ, w_d_max)`：距离权重，远场补偿

---

## 2. 关键常量与规范

### 2.1 固定常量（不可更改）

| 常量 | 值 | 说明 |
|------|-----|------|
| **R_MAX** | **65.0 m** | 极坐标最大半径，**全链路必须统一** |
| EPS | 1e-5 | 数值稳定性常量 |
| map_size | 102.4 m | BEV 地图边长 (pc_range 推导) |
| pc_range | [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0] | 点云范围 |

### 2.2 Query BBox 10 维定义（RaCFormer 原有规范）

| 索引 | 符号 | 物理含义 | 空间类型 | 范围 | RWHI 输出要求 |
|------|------|----------|----------|------|---------------|
| 0 | θ | 极坐标角度 | Normalized | [0, 1) | 直接归一化，可循环 |
| 1 | d | 极坐标距离 | Normalized | **(EPS, 1-EPS)** | **必须 clamp** |
| 2 | z | 归一化高度 | Normalized | **(EPS, 1-EPS)** | **必须 clamp** |
| 3 | w (log) | 物体宽度 | Log-Space | (-∞, +∞) | `log(width_m.clamp(min=0.1))` |
| 4 | l (log) | 物体长度 | Log-Space | (-∞, +∞) | `log(length_m.clamp(min=0.1))` |
| 5 | h (log) | 物体高度 | Log-Space | (-∞, +∞) | `log(height_m.clamp(min=0.1))` |
| 6 | sin(yaw) | 航向角正弦 | Linear | [-1, 1] | 初始化 0.0 |
| 7 | cos(yaw) | 航向角余弦 | Linear | [-1, 1] | 初始化 1.0 |
| 8 | vx | X方向速度 | Linear | 任意 | 初始化 0.0 |
| 9 | vy | Y方向速度 | Linear | 任意 | 初始化 0.0 |

### 2.3 归一化基准

| 量 | 归一化分母 | 说明 |
|----|-----------|------|
| θ | 2π | `theta_norm = theta_rad / (2π)` |
| d | **R_MAX = 65.0 m** | `d_norm = distance / 65.0`，**不是 map_size** |
| z | pc_range z 范围 (8m) | `z_norm = (z_phys + 5) / 8` |

---

## 3. RaCFormer 原始 Query 生成机制

### 3.1 原始均匀极坐标初始化

```python
# racformer_head.py 原始代码
def generate_points(self):
    theta = torch.linspace(0, 1, num_ray)  # [0, 1]
    d = torch.linspace(0, 1, num_depth)    # [0, 1]
    # 组合成 num_ray * num_depth 个锚点
```

### 3.2 Transformer 的 refine_bbox 机制

**关键代码 (racformer_transformer.py)**：

```python
def refine_bbox(self, bbox_proposal, bbox_delta):
    # θ: 直接加法（有界 delta）
    theta = bbox_proposal[..., 0:1] + (sigmoid(bbox_delta[..., 0:1])*2-1) / num_ray
    
    # d, z: inverse_sigmoid + sigmoid 配对
    dz = inverse_sigmoid(bbox_proposal[..., 1:3])  # ← 输入必须在 (0, 1)
    dz_new = sigmoid(dz_delta + dz)
    
    # w, l, h, sin, cos, vx, vy: 直接替换
    return cat([theta, dz_new, bbox_delta[..., 3:]], dim=-1)
```

### 3.3 坐标转换函数

```python
# theta_d2xy_coods: 极坐标 → 笛卡尔 (用于采样)
x = (center + d * R_MAX * cos(θ * 2π)) / map_size
y = (center + d * R_MAX * sin(θ * 2π)) / map_size

# xy2theta_d_coods: 笛卡尔 → 极坐标 (用于编码)
d = sqrt(x² + y²) / R_MAX
θ = atan2(y, x) / (2π)
```

---

## 4. RWHI v7 核心算法

### 4.1 AlphaMLP（点置信度预测）

```python
class AlphaMLP(nn.Module):
    """
    输入: f_in = [log1p(max(rcs,0)), d_norm, v_norm]  # 3维
    输出: α ∈ (0, 1)
    结构: Linear(3→32) → ReLU → Linear(32→32) → ReLU → Linear(32→1) → Sigmoid
    """
    def __init__(self, in_dim=3, hidden_dim=32, init_bias=1.0):
        # init_bias=1.0 使初始 α ≈ 0.73，偏向信任雷达
```

### 4.2 AlphaEncoder（α 特征编码）

```python
class AlphaEncoder(nn.Module):
    """
    将 α 编码为 d_alpha 维 embedding，用于 Query 特征增强
    不修改 bbox_proposal 的 10 维结构
    
    输入: α ∈ (0, 1)
    处理: α_center = (α - 0.5) * 2  → [-1, 1]
          u = [α_center, α_center²]  → 2维
    输出: Linear(2→8) → ReLU → Linear(8→d_alpha)
    """
```

### 4.3 打分图构建流程

```python
def forward(radar_points, radar_mask):
    # 1. 计算每个点的 α
    alpha = alpha_mlp(f_in)  # [B, M, 1]
    
    # 2. 计算 I_radar = α * log(1 + σ * w_v * w_d)
    w_v = 1 + beta * sigmoid(|v_r| / v_ref)
    w_d = (dist / d_ref).pow(d_lambda).clamp(max=w_d_max)
    i_radar = alpha * log(1 + rcs_proc * w_v * w_d)
    
    # 3. Scatter-Add 聚合到 BEV 网格
    i_radar_map = scatter_add(i_radar, grid_indices)  # [B, 1, H, W]
    
    # 4. 构建三层打分场
    S = C_base + i_radar_map + epsilon * J
    
    # 5. 残差式空间扩散
    S_final = S + gamma * (MaxPool3x3(S) - S)
    
    # 6. Top-K 选取
    _, topk_idx = topk(S_final.flatten(), K)
    
    # 7. 转换为极坐标锚点
    anchors = topk_to_anchors(topk_idx)  # [B, K, 10]
    
    return anchors, alpha_topk
```

---

## 5. 严重易错点（血泪教训）

### 🔴 易错点 1: 双重 inverse_sigmoid

**错误做法**：
```python
# RWHI 输出
theta_d = inverse_sigmoid(theta_d)  # ❌ 转为 logits
z = inverse_sigmoid(z)

# Transformer refine_bbox
dz = inverse_sigmoid(bbox_proposal[..., 1:3])  # ❌ 再次转换
```

**后果**：坐标被完全扭曲，模型无法收敛。

**正确做法**：
```python
# RWHI 输出归一化值 [0, 1]，不做 inverse_sigmoid
# d 和 z 必须 clamp 到 (EPS, 1-EPS)
d_norm = (dist / R_MAX).clamp(min=EPS, max=1.0 - EPS)
z_norm = z_default.clamp(min=EPS, max=1.0 - EPS)
# 直接输出，让 Transformer 的 refine_bbox 做 inverse_sigmoid
```

### 🔴 易错点 2: polar_radius 不一致

**错误做法**：
```python
# RWHI 使用固定值
self.polar_radius = 65.0

# Transformer 从 pc_range 计算
self.polar_radius = sqrt(51.2² + 51.2²) = 72.4  # ❌ 不一致！
```

**后果**：采样位置偏移 11%，query 和 GT 不对齐。

**正确做法**：
```python
# 全链路统一使用 R_MAX = 65.0
# 在 config 中明确指定：
model = dict(
    pts_bbox_head=dict(
        polar_radius=65.0,
        transformer=dict(polar_radius=65.0),
    ),
)
```

### 🔴 易错点 3: d 归一化分母错误

**错误做法**：
```python
d_norm = dist / map_size  # ❌ map_size = 102.4
d_norm = dist / polar_radius_from_pc_range  # ❌ = 72.4
```

**正确做法**：
```python
R_MAX = 65.0
d_norm = dist / R_MAX  # ✅ 固定 65.0
```

### 🔴 易错点 4: α embedding 注入位置错误

**错误做法**：
```python
# 将 α 直接放入 bbox_proposal 的第 11 维
bbox_proposal = cat([theta_d, z, ..., alpha], dim=-1)  # ❌ 破坏 10 维结构
```

**正确做法**：
```python
# α embedding 注入到 query_feat，不改变 bbox_proposal
alpha_emb = alpha_encoder(alpha_values)  # [B, K, d_alpha]
query_feat = cat([pos2content(bbox[..., :3]), alpha_emb], dim=-1)
query_feat = alpha_fusion(query_feat)  # Linear + LayerNorm
```

### 🔴 易错点 5: DDP 兼容性

**错误做法**：
```python
# 新模块参数未参与计算图，DDP 报错
```

**正确做法**：
```python
if self.training:
    # 添加 dummy sum 确保所有参数参与计算图
    dummy = alpha_encoder.fc1.weight.sum() * 0.0
    query_feat = query_feat + dummy
```

### 🔴 易错点 6: z_default 物理含义

**错误做法**：
```python
z_default = 0.0  # ❌ 对应物理高度 -5m（地下）
```

**正确做法**：
```python
z_default = 0.5  # ✅ 对应物理高度 -1m（接近地面）
# z_phys = z_norm * 8 - 5 = 0.5 * 8 - 5 = -1m
```

### 🔴 易错点 7: wlh 使用物理值而非 log 值

**错误做法**：
```python
anchors[..., 3] = 1.8  # ❌ 物理宽度
```

**正确做法**：
```python
anchors[..., 3] = math.log(max(1.8, 0.1))  # ✅ log 空间
```

### 🔴 易错点 8: θ 的归一化范围

**注意**：θ 归一化到 [0, 1)，不是 (EPS, 1-EPS)，因为 refine_bbox 对 θ 使用直接加法，不做 inverse_sigmoid。

```python
theta_norm = theta_rad / (2 * math.pi)  # [0, 1)
theta_norm = theta_norm.clamp(min=0.0, max=1.0 - EPS)  # 避免 = 1.0
```

---

## 6. 文件结构与接口设计

### 6.1 推荐文件结构

```
models/
├── rwhi.py              # 统一入口（门面模式）
├── rwhi_v53.py          # RWHI v7 核心实现
├── racformer_head.py    # 检测头（集成 RWHI）
├── racformer_transformer.py  # Transformer（需传入 polar_radius）
└── bbox/
    └── utils.py         # 坐标转换函数
```

### 6.2 RWHI 模块接口

```python
class RWHI_v53(BaseModule):
    def __init__(self, num_query, pc_range, bev_grid_size, ...):
        """
        必须参数:
        - num_query: Query 数量 (K)
        - pc_range: 点云范围
        - bev_grid_size: BEV 网格尺寸
        
        关键: self.polar_radius = R_MAX = 65.0 (固定)
        """
    
    def forward(self, radar_points, radar_mask):
        """
        输入:
        - radar_points: [B, M, C] (x, y, z, rcs, v_r, ...)
        - radar_mask: [B, M] (1=有效, 0=padding)
        
        输出:
        - anchors: [B, K, 10] 归一化值，不做 inverse_sigmoid
        - alpha_values: [B, K, 1] 每个锚点的 α
        """
    
    def encode_alpha(self, alpha_values):
        """将 α 编码为 embedding"""
        return self.alpha_encoder(alpha_values)
    
    @property
    def safety_anchors(self):
        """返回 [num_query, 10] 用于初始化 init_query_bbox"""
    
    @property
    def d_alpha(self):
        """返回 α embedding 维度"""
```

### 6.3 RaCFormerHead 集成接口

```python
class RaCFormerHead:
    def __init__(self, ..., use_rwhi=False, rwhi_cfg=None, polar_radius=None):
        """
        polar_radius: 全链路统一的极坐标半径 (推荐 65.0)
        """
        if use_rwhi:
            self._init_rwhi_layers()
    
    def _init_rwhi_layers(self):
        """
        初始化:
        - self.rwhi_module: RWHI 模块
        - self.pos2content: 位置 → 内容 MLP
        - self.alpha_fusion: α embedding 融合层 (Linear + LayerNorm)
        """
    
    def _prepare_query_bbox(self, radar_points, radar_mask, batch_size, device):
        """
        返回:
        - query_bbox: [B, K, 10]
        - alpha_values: [B, K, 1]
        - using_dynamic_rwhi: bool
        """
    
    def _prepare_query_feat(self, query_bbox, ..., alpha_values=None):
        """
        构建 query 特征:
        1. pos2content(query_bbox[..., :3]) → 动态内容
        2. 如果有 α: alpha_emb = encode_alpha(alpha_values)
        3. 融合: cat([content, alpha_emb]) → alpha_fusion
        """
    
    def _validate_query_bbox(self, query_bbox):
        """
        验证并 clamp:
        - theta: [0, 1-EPS]
        - d: (EPS, 1-EPS)  ← 必须严格开区间
        - z: (EPS, 1-EPS)  ← 必须严格开区间
        """
```

---

## 7. 详细实现规范

### 7.1 _xy_to_theta_d 实现

```python
def _xy_to_theta_d(self, xy):
    """
    将 (x, y) 物理坐标转换为归一化极坐标
    
    关键:
    - 使用固定 R_MAX = 65.0
    - d 必须 clamp 到 (EPS, 1-EPS)
    """
    x, y = xy[..., 0], xy[..., 1]
    
    # θ: atan2 → [0, 2π) → [0, 1)
    theta_rad = torch.atan2(y, x)
    theta_rad = (theta_rad + 2 * math.pi) % (2 * math.pi)
    theta_norm = theta_rad / (2 * math.pi)
    theta_norm = theta_norm.clamp(min=0.0, max=1.0 - EPS)
    
    # d: 使用固定 R_MAX = 65.0
    dist = torch.sqrt(x**2 + y**2).clamp(min=1e-6)
    d_norm = (dist / R_MAX).clamp(min=EPS, max=1.0 - EPS)
    
    return torch.stack([theta_norm, d_norm], dim=-1)
```

### 7.2 _compute_i_radar 实现

```python
def _compute_i_radar(self, radar_points, radar_mask, alpha):
    """
    I_radar = α * log(1 + σ * w_v * w_d)
    """
    x = radar_points[..., 0]
    y = radar_points[..., 1]
    rcs = radar_points[..., 3]
    v_r = radar_points[..., 4]
    
    dist = torch.sqrt(x**2 + y**2).clamp(min=self.d_min, max=R_MAX)
    
    # RCS 预处理
    sigma_proc = F.relu(rcs)
    
    # 速度权重: w_v = 1 + β * sigmoid(|v|/v_ref)
    w_v = 1.0 + self.beta * torch.sigmoid(v_r.abs() / self.v_ref)
    
    # 距离权重: w_d = min((d/d_ref)^λ, w_d_max)
    w_d = (dist / self.d_ref).pow(self.d_lambda).clamp(max=self.w_d_max)
    
    # I_radar
    phys_term = 1.0 + sigma_proc * w_v * w_d
    alpha_squeezed = alpha.squeeze(-1)
    i_radar = alpha_squeezed * torch.log(phys_term)
    
    # 应用 mask
    if radar_mask is not None:
        i_radar = i_radar * radar_mask
    
    return i_radar
```

### 7.3 _build_score_map 实现

```python
def _build_score_map(self, i_radar_map, device):
    """
    S = C_base + I_radar + ε*J
    S_final = S + γ*(MaxPool(S) - S)
    """
    B = i_radar_map.shape[0]
    
    # 安全层
    base_field = torch.full((B, 1, H, W), self.base_bias, device=device)
    
    # 微扰层 (确定性 hash-based)
    jitter = self.epsilon * self.jitter_field.to(device)
    
    # 组合
    S = base_field + i_radar_map + jitter
    
    # 残差式空间扩散
    S_pool = self.diffusion(S)  # MaxPool3x3
    S_final = S + self.diffusion_gamma * (S_pool - S)
    
    # 可选: 得分上限裁剪
    if self.diffusion_s_max > 0:
        S_final = S_final.clamp(max=self.diffusion_s_max)
    
    return S_final
```

### 7.4 _topk_to_anchors 实现

```python
def _topk_to_anchors(self, S, alpha_map, batch_size, device):
    """
    从打分图提取 Top-K 位置，转换为 10 维锚点
    
    关键:
    - 输出归一化值，不做 inverse_sigmoid
    - d/z 必须 clamp 到 (EPS, 1-EPS)
    """
    # Top-K
    S_flat = S.view(batch_size, -1)
    _, topk_idx = torch.topk(S_flat, self.num_query, dim=1)
    
    # 获取 α
    alpha_flat = alpha_map.view(batch_size, -1)
    alpha_topk = torch.gather(alpha_flat, 1, topk_idx)
    alpha_topk = alpha_topk.clamp(0.0, 1.0).unsqueeze(-1)
    
    # 坐标转换
    topk_xy = self.grid_xy[topk_idx]
    theta_d = self._xy_to_theta_d(topk_xy)  # 已 clamp
    
    # z_norm
    z = torch.full((batch_size, self.num_query, 1), self.z_default, device=device)
    z = z.clamp(min=EPS, max=1.0 - EPS)
    
    # wlh (log 空间)
    w_log = math.log(max(self.w_default, 0.1))
    l_log = math.log(max(self.l_default, 0.1))
    h_log = math.log(max(self.h_default, 0.1))
    
    # 组装 10 维
    anchors = torch.cat([
        theta_d,  # [B, K, 2]
        z,        # [B, K, 1]
        torch.full(..., w_log),
        torch.full(..., l_log),
        torch.full(..., h_log),
        torch.full(..., 0.0),  # sin
        torch.full(..., 1.0),  # cos
        torch.full(..., 0.0),  # vx
        torch.full(..., 0.0),  # vy
    ], dim=-1)
    
    return anchors, alpha_topk
```

---

## 8. 配置文件规范

```python
# configs/racformer_with_rwhi_v53.py

# 全链路统一的极坐标半径
R_MAX = 65.0

rwhi_cfg = dict(
    rwhi_version='v5.3',
    
    # BEV 网格
    bev_grid_size=100,
    
    # I_radar 参数
    d_ref=30.0,
    d_lambda=1.0,
    w_d_max=2.5,
    d_min=2.0,
    v_max=30.0,
    v_ref=10.0,
    beta=1.0,
    
    # 打分场参数
    base_bias=1.0,
    epsilon=0.01,
    diffusion_gamma=0.5,
    diffusion_s_max=5.0,
    
    # 默认值
    z_default=0.5,
    w_default=1.8,
    l_default=4.0,
    h_default=1.5,
    
    # AlphaMLP
    alpha_mlp_in_dim=3,
    alpha_mlp_hidden=32,
    alpha_init_bias=1.0,
    
    # AlphaEncoder
    d_alpha=2,
    alpha_encoder_hidden=8,
    
    # 其他
    max_points=5000,
    enabled=True,
)

model = dict(
    pts_bbox_head=dict(
        num_query=900,
        use_rwhi=True,
        rwhi_cfg=rwhi_cfg,
        
        # [关键] 全链路统一 polar_radius
        polar_radius=R_MAX,
        
        transformer=dict(
            type='RaCFormerTransformer',
            polar_radius=R_MAX,  # 传入固定值
        ),
    ),
)
```

---

## 9. 测试验证清单

### 9.1 数值范围测试

```python
def test_output_ranges():
    rwhi = RWHI_v53(...)
    anchors, alpha = rwhi(radar_points, radar_mask)
    
    # θ: [0, 1)
    assert anchors[..., 0].min() >= 0
    assert anchors[..., 0].max() < 1
    
    # d: (EPS, 1-EPS)
    assert anchors[..., 1].min() > 1e-5
    assert anchors[..., 1].max() < 1 - 1e-5
    
    # z: (EPS, 1-EPS)
    assert anchors[..., 2].min() > 1e-5
    assert anchors[..., 2].max() < 1 - 1e-5
    
    # α: (0, 1)
    assert alpha.min() > 0
    assert alpha.max() < 1
```

### 9.2 梯度流测试

```python
def test_gradient_flow():
    rwhi = RWHI_v53(...)
    anchors, alpha = rwhi(radar_points, radar_mask)
    
    loss = anchors.sum() + alpha.sum()
    loss.backward()
    
    # 检查所有参数有梯度
    for name, param in rwhi.named_parameters():
        assert param.grad is not None, f"{name} has no gradient"
```

### 9.3 全链路 polar_radius 一致性测试

```python
def test_polar_radius_consistency():
    head = RaCFormerHead(..., polar_radius=65.0)
    
    assert head.polar_radius == 65.0
    assert head.rwhi_module.polar_radius == 65.0
    # transformer 需要在完整模型中测试
```

---

## 附录: 数值量级参考

| 组成部分 | 数值范围 | 说明 |
|---------|---------|------|
| C_base | 1.0 | 均匀基础分数 |
| 背景格子 S | ≈ 1.0 ± 0.01 | C_base + ε*J |
| w_v | 1.0 ~ 2.0 | 速度权重 |
| w_d | 0 ~ 2.5 | 距离权重 |
| 真目标 I_radar | 1.5 ~ 3.0 | 高 RCS + 高 α |
| 真目标 S | 2.5 ~ 4.0 | C_base + I_radar |
| 杂波 I_radar | 0.01 ~ 0.05 | 低 α 抑制 |
| 扩散后背景 | 1.0 ~ 1.1 | 邻域提升 |
| 扩散后真目标 | 2.5 ~ 4.2 | 峰值保持 |

---

## 10. 原始 RaCFormer 代码结构参考

### 10.1 关键文件和函数

```
models/
├── racformer_head.py
│   ├── RaCFormerHead.__init__()     # 初始化检测头
│   ├── RaCFormerHead._init_layers() # 初始化网络层
│   ├── RaCFormerHead.generate_points() # 原始均匀极坐标生成
│   ├── RaCFormerHead.forward()      # 前向传播入口
│   └── RaCFormerHead.forward_single() # 单帧处理
│
├── racformer_transformer.py
│   ├── RaCFormerTransformer.__init__()
│   ├── RaCFormerTransformerDecoder.__init__()
│   └── RaCFormerTransformerDecoderLayer.refine_bbox() # 关键：坐标更新
│
└── bbox/utils.py
    ├── compute_map_size_and_radius() # 计算 map_size 和 polar_radius
    ├── theta_d2xy_coods()            # 极坐标 → 笛卡尔
    ├── xy2theta_d_coods()            # 笛卡尔 → 极坐标
    ├── encode_bbox()                 # bbox 编码
    └── decode_bbox()                 # bbox 解码
```

### 10.2 原始 forward 流程

```python
# racformer_head.py 简化流程
def forward_single(self, img_feats, radar_feats, img_metas, ...):
    # 1. 获取 query_bbox（原始：均匀极坐标）
    query_bbox = self.init_query_bbox.weight.clone()  # [num_query, 10]
    query_bbox = query_bbox.unsqueeze(0).repeat(B, 1, 1)
    
    # 2. 获取 query_feat（原始：类别嵌入 + indicator）
    query_feat = label_enc(torch.zeros(...))  # 类别 0 的嵌入
    query_feat = cat([query_feat, indicator0], dim=-1)
    
    # 3. (可选) Query Denoising
    query_bbox, query_feat, attn_mask, mask_dict = self.prepare_for_dn(...)
    
    # 4. Transformer 前向
    cls_scores, bbox_preds = self.transformer(
        query_bbox, query_feat, img_feats, lss_feats, radar_feats, ...
    )
    
    # 5. 后处理
    ...
```

### 10.3 RWHI 集成点

RWHI 需要修改的地方：

1. **query_bbox 生成**：替换 `self.init_query_bbox.weight` 为 `self.rwhi_module(radar_points)`
2. **query_feat 生成**：添加 `pos2content` MLP 和 `alpha_fusion`
3. **init_query_bbox 初始化**：使用 `rwhi_module.safety_anchors` 初始化权重

```python
# RWHI 集成后的流程
def forward_single(self, img_feats, radar_feats, radar_points, ...):
    if self.use_rwhi and radar_points is not None:
        # 动态 RWHI
        query_bbox, alpha_values = self.rwhi_module(radar_points, radar_mask)
        query_bbox = self._validate_query_bbox(query_bbox)  # clamp
        
        # 动态 query_feat
        query_pos = query_bbox[..., :3]
        dynamic_content = self.pos2content(query_pos)
        
        # α embedding 融合
        if alpha_values is not None:
            alpha_emb = self.rwhi_module.encode_alpha(alpha_values)
            feat_with_alpha = cat([dynamic_content, alpha_emb], dim=-1)
            query_feat = self.alpha_fusion(feat_with_alpha)
    else:
        # 原始路径
        query_bbox = self.init_query_bbox.weight.clone()
        query_feat = label_enc(...)
```

### 10.4 需要新增的网络层

```python
# 在 RaCFormerHead._init_rwhi_layers() 中初始化
self.rwhi_module = RWHIModule(rwhi_version='v5.3', **rwhi_cfg)

# pos2content: 将位置 [θ, d, z] 转换为内容特征
self.pos2content = nn.Sequential(
    nn.Linear(3, embed_dims),
    nn.LayerNorm(embed_dims),
    nn.ReLU(inplace=True),
    nn.Linear(embed_dims, embed_dims - 1),  # 留 1 维给 indicator
)

# alpha_fusion: 融合 α embedding
d_alpha = self.rwhi_module.d_alpha  # 通常为 2
self.alpha_fusion = nn.Sequential(
    nn.Linear(embed_dims - 1 + d_alpha, embed_dims),
    nn.LayerNorm(embed_dims),
)
```

---

## 11. 调试建议与常见问题

### 11.1 训练不收敛

**症状**：loss 不下降或 NaN

**可能原因**：
1. d/z 未 clamp 到 (EPS, 1-EPS)，inverse_sigmoid 溢出
2. RWHI 输出了 logits 而不是归一化值
3. polar_radius 不一致，采样位置偏移

**调试方法**：
```python
# 在 forward 中添加
print(f"d range: [{query_bbox[..., 1].min():.4f}, {query_bbox[..., 1].max():.4f}]")
print(f"z range: [{query_bbox[..., 2].min():.4f}, {query_bbox[..., 2].max():.4f}]")
# 期望输出: d/z 在 (0.00001, 0.99999) 范围内
```

### 11.2 mAP 很低

**可能原因**：
1. polar_radius 不一致（RWHI 用 65，Transformer 用 72.4）
2. DN 噪声尺度与 d 归一化不匹配
3. α embedding 注入位置错误

**调试方法**：
```python
# 检查 polar_radius 一致性
print(f"RWHI polar_radius: {self.rwhi_module.polar_radius}")
print(f"Head polar_radius: {self.polar_radius}")
print(f"Transformer polar_radius: {self.transformer.polar_radius}")
# 三者必须相等 = 65.0
```

### 11.3 DDP 报错 "unused parameters"

**解决方法**：
```python
if self.training:
    # 确保所有新参数参与计算图
    dummy = 0.0
    for module in [self.alpha_mlp, self.alpha_encoder, self.alpha_fusion]:
        if module is not None:
            for param in module.parameters():
                dummy = dummy + param.sum() * 0.0
    query_feat = query_feat + dummy
```

### 11.4 打分图验证

```python
# 可视化打分图
import matplotlib.pyplot as plt

S = rwhi._build_score_map(i_radar_map, device)
plt.imshow(S[0, 0].cpu().numpy())
plt.colorbar()
plt.title(f"Score map: min={S.min():.2f}, max={S.max():.2f}")
plt.savefig("score_map.png")

# 期望: 雷达目标位置有明显峰值 (2.5~4.0)，背景 ≈ 1.0
```

---

## 12. 快速实现检查清单

### 实现前检查

- [ ] 理解 RaCFormer 原始 query 生成流程
- [ ] 理解 refine_bbox 的 inverse_sigmoid + sigmoid 机制
- [ ] 确认 polar_radius = 65.0 全链路统一

### 实现中检查

- [ ] RWHI 输出归一化值，不做 inverse_sigmoid
- [ ] d/z 使用 clamp(EPS, 1-EPS)
- [ ] wlh 使用 log 空间
- [ ] θ 使用 clamp(0, 1-EPS)
- [ ] α embedding 不放入 bbox，只放入 query_feat
- [ ] 所有新参数添加 DDP dummy sum

### 实现后验证

- [ ] 数值范围测试通过
- [ ] 梯度流测试通过
- [ ] 单 GPU 训练正常启动
- [ ] DDP 训练无 unused parameter 错误

---

## 13. ⚠️ 常见易错点（经验教训）

### 13.1 AttributeError: 'RaCFormer_head' has no attribute 'pc_range'

**错误现象**：
```
AttributeError: 'RaCFormer_head' object has no attribute 'pc_range'
```
训练启动时立即报错，在模型初始化阶段。

**原因分析**：

mmdet/mmdet3d 的检测头基类 (DETRHead) 在 `__init__()` 中会调用 `_init_layers()` 方法。代码执行顺序如下：

```
__init__ 开始
  ├── 设置 self.use_rwhi, self.rwhi_cfg 等
  ├── 调用 super().__init__(...)  ← 这里触发 _init_layers()
  │     └── _init_layers() 
  │           └── _init_rwhi_layers() 
  │                 └── 使用 self.pc_range ❌ 此时尚未定义！
  └── self.pc_range = self.bbox_coder.pc_range  ← 太晚了！
```

**修复方式**：

将 `bbox_coder` 构建和 `pc_range` 设置移到 `super().__init__()` 调用**之前**：

```python
# 【关键修复】必须在 super().__init__() 之前设置
self.bbox_coder = build_bbox_coder(bbox_coder)
self.pc_range = self.bbox_coder.pc_range

super(RaCFormer_head, self).__init__(...)  # 这里会调用 _init_layers()
```

**建议检查项**：

- [ ] 凡是在 `_init_layers()` 中使用的属性，必须在 `super().__init__()` 之前设置
- [ ] 涉及 voxel/grid 归一化的模块（如 RWHI），必须确保 `pc_range` 在初始化时可用
- [ ] 检查所有 `self.xxx` 属性的定义顺序，尤其是与父类初始化交互的属性

### 13.2 参数初始化顺序通用规则

对于 mmdet/mmdet3d 框架中的检测头，属性初始化的正确顺序应为：

```python
def __init__(self, ..., bbox_coder=None, **kwargs):
    # 1. 先设置简单属性
    self.code_size = code_size
    self.num_classes = num_classes
    
    # 2. 构建需要在 _init_layers() 中使用的模块和属性
    self.bbox_coder = build_bbox_coder(bbox_coder)
    self.pc_range = self.bbox_coder.pc_range
    
    # 3. 再调用父类初始化（会触发 _init_layers()）
    super().__init__(...)
    
    # 4. 最后设置不依赖 _init_layers() 的属性
    self.code_weights = nn.Parameter(...)
```

### 13.3 DDP unused-parameter 风险 (High)

**错误现象**：
```
RuntimeError: Expected to have finished reduction in the prior iteration 
before starting a new one. (find_unused_parameters=False)
```

**原因分析**：

当 `use_rwhi=True` 但 `using_dynamic_rwhi=False`（如空雷达点时），原代码的 dummy sum 只在动态分支执行：

```python
# ❌ 错误：dummy sum 在条件分支内
if using_dynamic_rwhi and alpha_values is not None:
    ...
    if self.training:
        dummy = ...  # 只在这里执行
        query_feat = query_feat + dummy
else:
    # 这里 rwhi_module, pos2content, alpha_fusion 参数未参与计算图！
    query_feat = ...
```

**修复方式**：

将 dummy sum 移到条件分支外，确保无论是否使用动态 RWHI 都执行：

```python
# ✅ 正确：dummy sum 在所有 RWHI 路径后执行
if using_dynamic_rwhi:
    ...
else:
    ...

# 无论哪个分支，都确保 RWHI 参数参与计算图
if self.training and self.use_rwhi:
    dummy = 0.0
    for module in [self.pos2content, self.alpha_fusion]:
        for param in module.parameters():
            dummy = dummy + param.sum() * 0.0
    for param in self.rwhi_module.parameters():
        dummy = dummy + param.sum() * 0.0
    query_feat = query_feat + dummy
```

### 13.4 DN bbox clamp 导致 inverse_sigmoid 溢出 (High)

**错误现象**：
```
RuntimeWarning: invalid value encountered in log
# 或 loss 变为 NaN/Inf
```

**原因分析**：

Query Denoising 中对 bbox 的 clamp 使用 `[0, 1]`：

```python
# ❌ 错误：允许 d/z 为 0 或 1
known_bbox_expand[..., 0:3].clamp_(min=0.0, max=1.0)
```

但 `refine_bbox` 对 d/z 使用 `inverse_sigmoid`：

```python
dz = inverse_sigmoid(bbox_proposal[..., 1:3])  # d=0 或 d=1 时 → ±inf
```

**修复方式**：

对 θ 和 d/z 使用不同的 clamp 范围：

```python
# ✅ 正确：θ 用 [0, 1]，d/z 用 (EPS, 1-EPS)
known_bbox_expand[..., 0:1].clamp_(min=0.0, max=1.0)       # θ
known_bbox_expand[..., 1:3].clamp_(min=EPS, max=1.0 - EPS) # d, z
```

### 13.5 Tensor 布尔上下文问题 (Medium)

**错误现象**：
```
RuntimeError: Boolean value of Tensor with more than one value is ambiguous
```

**原因分析**：

```python
# ❌ 可能出错：tensor > 0 返回 tensor，在布尔上下文中不安全
has_valid_points = radar_mask is not None and radar_mask.sum() > 0
```

**修复方式**：

```python
# ✅ 正确：显式转为 Python bool
has_valid_points = radar_mask is not None and radar_mask.sum().item() > 0
```

### 13.6 坐标转换函数 polar_radius 硬编码 (Medium)

**错误现象**：

修改配置中的 `polar_radius` 后，预测坐标偏移。

**原因分析**：

`bbox/utils.py` 中的 `theta_d2xy_coods` 和 `xy2theta_d_coods` 使用硬编码默认值 `r=65.0`，调用点未传入配置的 `polar_radius`。

**修复方式**：

在 `bbox/utils.py` 中定义模块常量，并在函数中使用：

```python
# bbox/utils.py
R_MAX = 65.0
MAP_SIZE = 102.4

def theta_d2xy_coods(theta_d_coords, map_size=None, r=None):
    if map_size is None:
        map_size = MAP_SIZE
    if r is None:
        r = R_MAX
    ...
```

### 13.7 safety_anchors 硬编码 num_clusters (Low)

**错误现象**：

当 `num_clusters != 6` 时，safety_anchors 数量与 num_query 不匹配。

**原因分析**：

```python
# ❌ 硬编码 6
num_angles = self.num_query // 6
num_clusters = 6
```

**修复方式**：

在 RWHIModule 中添加 `num_clusters` 参数：

```python
def __init__(self, ..., num_clusters=6, ...):
    self.num_clusters = num_clusters

@property
def safety_anchors(self):
    num_clusters = self.num_clusters  # ✅ 使用配置值
    num_angles = math.ceil(self.num_query / num_clusters)  # ✅ 使用 ceil 处理不整除
    ...
    # 截取到 num_query 个
    if theta_d.shape[0] > self.num_query:
        theta_d = theta_d[:self.num_query]
```

### 13.8 init_query_bbox 未参与计算导致 DDP 报错 (Critical)

**错误现象**：
```
RuntimeError: Expected to have finished reduction in the prior iteration before starting a new one.
Parameter indices which did not receive grad for rank 0: 96
```

**原因分析**：

当 `using_dynamic_rwhi=True` 时，`query_bbox` 由 RWHI 模块生成，而 `self.init_query_bbox`（`nn.Embedding`）完全没有被使用：

```python
# ❌ 问题代码：当 RWHI 生效时，init_query_bbox 完全不参与
if has_valid_points:
    query_bbox, alpha_values = self.rwhi_module(radar_points, radar_mask)  # 使用 RWHI
    # init_query_bbox.weight 没有被使用！
```

**修复方式**：

在 `_prepare_query_feat` 中添加 `init_query_bbox` 的 dummy sum：

```python
if self.training:
    dummy = 0.0
    if self.use_rwhi:
        # RWHI 相关模块
        for module in [self.pos2content, self.alpha_fusion]:
            for param in module.parameters():
                dummy = dummy + param.sum() * 0.0
        for param in self.rwhi_module.parameters():
            dummy = dummy + param.sum() * 0.0
        
        # 【新增修复】当使用动态 RWHI 时，init_query_bbox 未参与计算
        if using_dynamic_rwhi:
            for param in self.init_query_bbox.parameters():
                dummy = dummy + param.sum() * 0.0
    
    if dummy != 0.0:
        query_feat = query_feat + dummy
```

**DDP 参数覆盖矩阵**：

| 场景 | init_query_bbox | rwhi_module | pos2content | alpha_fusion |
|------|-----------------|-------------|-------------|--------------|
| `use_rwhi=False` | ✅ 正常使用 | N/A | N/A | N/A |
| `use_rwhi=True, dynamic=False` | ✅ 正常使用 | ⚠️ dummy sum | ⚠️ dummy sum | ⚠️ dummy sum |
| `use_rwhi=True, dynamic=True` | ⚠️ **dummy sum** | ✅ 正常使用 | ✅ 正常使用 | ✅ 正常使用 |

### 13.9 dummy != 0.0 条件判断错误 (Critical)

**错误现象**：
DDP 仍然报错 unused parameters，即使已添加 dummy sum 逻辑。

**原因分析**：

```python
# ❌ 问题代码
dummy = 0.0
dummy = dummy + param.sum() * 0.0  # 结果是 tensor(0.0)，不是 Python float

if dummy != 0.0:  # tensor(0.0) != 0.0 的结果是 tensor(False)，在布尔上下文中不确定
    query_feat = query_feat + dummy
```

Python 中 `tensor(0.0) != 0.0` 返回的是 `tensor(False)`，而不是 Python `bool`。在条件判断中这个行为是不确定的。

**修复方式**：

```python
# ✅ 正确代码
dummy = None  # 使用 None 作为初始值

for param in module.parameters():
    term = param.sum() * 0.0
    dummy = term if dummy is None else dummy + term

if dummy is not None:  # 使用 is not None 检查
    query_feat = query_feat + dummy
```

### 13.10 坐标转换函数 polar_radius 配置同步 (残余风险)

**风险描述**：

`bbox/utils.py` 中的 `theta_d2xy_coods` 和 `xy2theta_d_coods` 使用模块常量 `R_MAX = 65.0`，但 `racformer_transformer.py` 中的调用点未传入 `polar_radius` 参数。如果未来需要修改 `polar_radius`，需要同步修改多处。

**当前设计**：

```python
# bbox/utils.py
R_MAX = 65.0  # 模块级常量
MAP_SIZE = 102.4

def theta_d2xy_coods(theta_d_coords, map_size=None, r=None):
    if map_size is None:
        map_size = MAP_SIZE
    if r is None:
        r = R_MAX  # 使用模块常量
    ...
```

**已修复**（2026-01-22）：

R_MAX 现已集中定义在 `bbox/utils.py`，其他模块通过导入获取：

```python
# bbox/utils.py - 单一来源
R_MAX = 65.0

# 其他模块导入
from .bbox.utils import R_MAX  # rwhi.py, racformer_head.py, racformer_transformer.py
```

如需修改 `polar_radius`，只需更新 `bbox/utils.py` 中的 `R_MAX` 值。

---

## 14. 修复记录

| 日期 | 问题 | 严重程度 | 修复文件 | 修复内容 |
|-----|------|---------|---------|---------|
| 2026-01-22 | pc_range 初始化顺序 | High | racformer_head.py | 移动 bbox_coder 构建到 super().__init__() 之前 |
| 2026-01-22 | DDP unused-parameter | High | racformer_head.py | dummy sum 移到条件分支外 |
| 2026-01-22 | DN bbox clamp | High | racformer_head.py | d/z 使用 (EPS, 1-EPS) |
| 2026-01-22 | Tensor 布尔上下文 | Medium | racformer_head.py | 使用 .item() |
| 2026-01-22 | polar_radius 硬编码 | Medium | bbox/utils.py | 使用模块常量 |
| 2026-01-22 | num_clusters 硬编码 | Low | rwhi.py | 添加 num_clusters 参数 |
| 2026-01-22 | num_query 整除问题 | Low | rwhi.py | 使用 ceil 并截取确保锚点数量正确 |
| 2026-01-22 | R_MAX 分散定义 | Medium | 多文件 | 集中到 bbox/utils.py，其他模块导入 |
| 2026-01-22 | generate_points 整除 | High | racformer_head.py | 使用 ceil 并截取确保初始点数量正确 |
| 2026-01-22 | AMP/FP16 dtype 兼容 | Medium | rwhi.py | 所有 torch.zeros/full 添加 dtype=values.dtype |
| 2026-01-22 | pc_range x/y 对称假设 | Low | rwhi.py | 添加断言检查 x_range == y_range |
| 2026-01-22 | **RWHI query_feat 被覆盖** | **Critical** | racformer_head.py | prepare_for_dn_input 接收外部 init_query_feat |
| 2026-01-23 | **init_query_bbox 未参与计算** | **Critical** | racformer_head.py | 当 using_dynamic_rwhi=True 时添加 init_query_bbox 的 dummy sum |
| 2026-01-23 | **dummy != 0.0 条件判断错误** | **Critical** | racformer_head.py | 0.0 + tensor*0 仍是 tensor，使用 `dummy is not None` 替代 |
| 2026-01-23 | DN/indicator dtype 不兼容 | Medium | racformer_head.py | torch.zeros/ones 添加 dtype 参数 |
| 2026-01-23 | radar padding dtype 不兼容 | Medium | racformer.py | torch.zeros 添加 dtype=radar_pts[0].dtype |
| 2026-01-23 | DN theta clamp max=1.0 | Low | racformer_head.py | 改为 max=1.0-EPS 与 RWHI 一致 |

---

**文档版本**: v1.8  
**更新日期**: 2026-01-22  
**适用范围**: RWHI v7 在 RaCFormer 上的实现

