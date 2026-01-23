# RaCFormer Query 生成技术规范文档

> 本文档为 RWHI 模块适配 RaCFormer Transformer 接口提供详细技术规范

---

## 目录
1. [关键参数表](#1-关键参数表)
2. [五大核心问题解答](#2-五大核心问题解答)
3. [更新机制图解](#3-更新机制图解)
4. [RWHI 适配清单](#4-rwhi-适配清单)
5. [代码证据索引](#5-代码证据索引)

---

## 1. 关键参数表

### 1.1 Query BBox 10 维度完整定义

| 索引 | 符号 | 物理含义 | 空间类型 | 范围 | 默认初始化 | RWHI 输出要求 |
|------|------|----------|----------|------|------------|---------------|
| 0 | θ (theta) | 极坐标角度 | Normalized | [0, 1] → [0, 2π] | `linspace(0,1,N)` | **直接归一化** |
| 1 | d | 极坐标距离 | Normalized | [0, 1] → [0, r] | `linspace(0,1,K)` | **需 clamp(eps, 1-eps)** |
| 2 | z | 归一化高度 | Normalized | [0, 1] → [z_min, z_max] | `0.5` | **直接归一化** |
| 3 | w (log) | 物体宽度 | **Log-Space** | (-∞, +∞) → exp() | `~N(0,1)` | **需 .log()** |
| 4 | l (log) | 物体长度 | **Log-Space** | (-∞, +∞) → exp() | `~N(0,1)` | **需 .log()** |
| 5 | h (log) | 物体高度 | **Log-Space** | (-∞, +∞) → exp() | `0.2` | **需 .log()** |
| 6 | sin(yaw) | 航向角正弦 | Linear | [-1, 1] | `~N(0,1)` | **直接输出** |
| 7 | cos(yaw) | 航向角余弦 | Linear | [-1, 1] | `~N(0,1)` | **直接输出** |
| 8 | vx | X方向速度 | Linear | 任意 | `0.0` | **直接输出** |
| 9 | vy | Y方向速度 | Linear | 任意 | `0.0` | **直接输出** |

### 1.2 关键常量

| 常量 | 值 | 来源 | 说明 |
|------|-----|------|------|
| `map_size` | 102.4 m | `bbox/utils.py:82` | BEV 地图边长 |
| `r` | 65.0 m | `bbox/utils.py:82` | 极坐标最大半径 |
| `center` | 51.2 m | `map_size / 2` | BEV 中心点 |
| `pc_range` | [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0] | config | 点云范围 |
| `num_ray` | 150 | config | 角度细化精度 |
| `eps` | 1e-5 | `utils.py:86` | inverse_sigmoid 数值保护 |

---

## 2. 五大核心问题解答

### 2.1 尺寸维度的定义 (Physical vs. Log-Space)

**判决：✅ RWHI 输出的 w, l, h 必须 Log 化**

#### 证据链

**初始化端** (`racformer_head.py:51-59`)：
```python
self.init_query_bbox = nn.Embedding(self.num_query, 10)
# ...
nn.init.constant_(self.init_query_bbox.weight[:, 5:6], 0.2)  # h 初始化为 0.2 (log空间)
```
- 第 5 维 (h) 显式设为 0.2，对应 `exp(0.2) ≈ 1.22m` 高度
- 第 3-4 维 (w, l) 使用默认随机初始化 ~N(0,1)

**编码端** (`bbox/utils.py:49-63`)：
```python
def encode_bbox(bboxes, pc_range=None):
    wlh = bboxes[..., 3:6].log()  # ← 物理尺寸被 log 变换
    ...
```

**解码端** (`bbox/utils.py:66-80`, `nms_free_coder.py:57`)：
```python
def denormalize_bbox(normalized_bboxes):
    w = normalized_bboxes[..., 2:3].exp()  # ← exp 恢复物理尺寸
    l = normalized_bboxes[..., 3:4].exp()
    h = normalized_bboxes[..., 5:6].exp()
```

**结论**：Transformer 内部处理的是 **Log 空间**的尺寸值。

| 物理尺寸 (m) | Log 值 | 备注 |
|--------------|--------|------|
| 0.5 | -0.69 | 小物体 |
| 1.0 | 0.0 | 中等 |
| 2.0 | 0.69 | 大物体 |
| 5.0 | 1.61 | 卡车 |

---

### 2.2 坐标更新机制的"双重标准"

**判决：是的，存在双重标准。d/z 需要 clamp 保护，θ 可自由取值**

#### 核心代码 (`racformer_transformer.py:226-232`)

```python
def refine_bbox(self, bbox_proposal, bbox_delta):
    # ========== d, z 使用 inverse_sigmoid + sigmoid 配对 ==========
    dz = inverse_sigmoid(bbox_proposal[..., 1:3])   # 索引 1, 2 → d, z
    dz_delta = bbox_delta[..., 1:3]
    dz_new = torch.sigmoid(dz_delta + dz)           # sigmoid 恢复到 [0, 1]
    
    # ========== θ 使用直接加法（有界 delta）==========
    theta = bbox_proposal[..., 0:1] + (torch.sigmoid(bbox_delta[..., 0:1])*2-1) / self.num_ray
    #                                  ↑ delta 范围 [-1/150, +1/150]
    
    # ========== 其余维度直接替换 ==========
    return torch.cat([theta, dz_new, bbox_delta[..., 3:]], dim=-1)
```

#### `inverse_sigmoid` 函数实现 (`utils.py:86-101`)

```python
def inverse_sigmoid(x, eps=1e-5):
    x = x.clamp(min=0, max=1)       # 输入限制在 [0, 1]
    x1 = x.clamp(min=eps)           # 防止 log(0)
    x2 = (1 - x).clamp(min=eps)     # 防止 log(0)
    return torch.log(x1 / x2)       # logit 变换
```

#### RWHI 适配要求

| 维度 | 初始值要求 | 原因 |
|------|------------|------|
| d | `clamp(eps, 1-eps)` | 避免 inverse_sigmoid 溢出 |
| z | `clamp(eps, 1-eps)` | 同上 |
| θ | `[0, 1]` 即可，可循环 | 直接加法，不经过 sigmoid |

---

### 2.3 Z 轴的参照系

**判决：Z 是归一化高度，范围 [0,1] 对应 pc_range 的 Z 区间**

#### 证据

**初始化** (`racformer_head.py:56`)：
```python
nn.init.constant_(self.init_query_bbox.weight[:, 2:3], 0.5)
```

**反归一化** (`racformer_head.py:104`)：
```python
bbox_preds[..., 2] = bbox_preds[..., 2] * (self.pc_range[5] - self.pc_range[2]) + self.pc_range[2]
#                   = z_norm * (3.0 - (-5.0)) + (-5.0)
#                   = z_norm * 8.0 - 5.0
```

#### Z 值对照表

| z (归一化) | 物理高度 (m) | 典型场景 |
|------------|--------------|----------|
| 0.0 | -5.0 | 地下 |
| 0.5 | -1.0 | 地面 |
| 0.625 | 0.0 | 地面 |
| 0.75 | 1.0 | 行人头部 |
| 1.0 | 3.0 | 最高点 |

**默认 z=0.5 对应物理高度 -1.0m**（接近地面）

---

### 2.4 航向角的几何一致性

**判决：sin/cos 不从 θ 计算，是独立预测的值**

#### 证据

**初始化** (`racformer_head.py:51-63`)：
```python
self.init_query_bbox = nn.Embedding(self.num_query, 10)
# 索引 6, 7 (sin, cos) 没有显式初始化
# 使用 nn.Embedding 默认：~N(0, 1)
```

**更新机制** (`racformer_transformer.py:232`)：
```python
return torch.cat([theta, dz_new, bbox_delta[..., 3:]], dim=-1)
#                                ↑ 索引 3-9: w, l, h, sin, cos, vx, vy
#                                  直接使用 reg_branch 输出，非迭代细化
```

#### RWHI 适配要点

1. **sin, cos 是独立预测的**，不需要根据极坐标 θ 计算
2. **初始值推荐**：
   - 若已知物体朝向，计算 `sin(yaw), cos(yaw)`
   - 若未知，可设为 `sin=0, cos=1`（对应 yaw=0）
3. **约束**：理论上应满足 `sin² + cos² = 1`，但代码中未强制

---

### 2.5 极坐标系的归一化基准

**判决：θ 除以 2π，d 除以 r (65.0m)**

#### 核心公式 (`bbox/utils.py:82-90`)

```python
def theta_d2xy_coods(theta_d_coords, map_size=102.4, r=65.0):
    center = map_size / 2  # = 51.2
    
    # θ 范围 [0, 1] → [0, 2π]
    x = (center + d * r * cos(θ * 2π)) / map_size
    y = (center + d * r * sin(θ * 2π)) / map_size
```

#### 逆变换 (`bbox/utils.py:93-106`)

```python
def xy2theta_d_coods(xy_coords_norm, map_size=102.4, r=65.0):
    # 笛卡尔 → 极坐标
    d = sqrt((x*map_size - center)² + (y*map_size - center)²) / r  # ÷ r
    θ = atan2(y - center, x - center)
    θ = ((θ + 2π) % 2π) / 2π  # ÷ 2π
```

#### 归一化总结

| 量 | 归一化分母 | 物理最大值 | 说明 |
|----|-----------|-----------|------|
| θ | 2π | 360° | 角度，范围 [0, 1] |
| d | r = 65.0 m | 65 m | **不是 map_size！** |

**⚠️ 关键注意**：d 的归一化基准是 `r=65.0m`，**不是** `map_size=102.4m`。这意味着 d=1 对应 65 米，而不是地图边长。

---

## 3. 更新机制图解

### 3.1 Decoder Layer 数据流

```
                    ┌─────────────────────────────────────────────────────────┐
                    │              RaCFormerTransformerDecoderLayer           │
                    │                                                          │
 query_bbox [B,Q,10]│    ┌────────────────┐     ┌─────────────┐               │
 ─────────────────▶ │    │  Position      │     │  reg_branch │               │
      (θ,d,z,w,l,h, │    │  Encoder       │     │  (Linear)   │               │
       sin,cos,vx,vy)    │  [前3维]        │     └──────┬──────┘               │
                    │    └───────┬────────┘            │                       │
                    │            │                     │ bbox_delta [B,Q,10]   │
                    │            ▼                     ▼                       │
                    │    ┌───────────────────────────────┐                     │
                    │    │         refine_bbox()         │                     │
                    │    │  ┌─────────────────────────┐  │                     │
                    │    │  │ θ: 直接加法              │  │                     │
                    │    │  │ d,z: inv_sig + sig 配对 │  │                     │
                    │    │  │ 其余: 直接替换           │  │                     │
                    │    │  └─────────────────────────┘  │                     │
                    │    └───────────────┬───────────────┘                     │
                    │                    │                                     │
                    │                    ▼                                     │
                    │           bbox_pred [B,Q,10]                             │
                    └────────────────────┼─────────────────────────────────────┘
                                         │
                                         ▼
                              theta_d2xy_coods()  ──────▶  [x,y,z,w,l,h,sin,cos,vx,vy]
                                                           (极坐标 → 笛卡尔)
```

### 3.2 各维度更新方式对比

```
┌────────────────────────────────────────────────────────────────────────────┐
│                        refine_bbox() 更新策略                               │
├─────────┬──────────────────────────────────────────────────────────────────┤
│   θ     │  θ_new = θ_old + (sigmoid(δ)*2-1) / num_ray                      │
│  [idx0] │  ↳ 直接加法，delta 范围 [-1/150, +1/150]                          │
│         │  ↳ 理论上可能超出 [0,1]，但转换时会被 % 2π 处理                    │
├─────────┼──────────────────────────────────────────────────────────────────┤
│   d     │  d_new = sigmoid(inverse_sigmoid(d_old) + δ)                     │
│  [idx1] │  ↳ 保证输出始终在 [0, 1]                                          │
│         │  ↳ 输入必须在 (0, 1)，否则 inverse_sigmoid 溢出                    │
├─────────┼──────────────────────────────────────────────────────────────────┤
│   z     │  z_new = sigmoid(inverse_sigmoid(z_old) + δ)                     │
│  [idx2] │  ↳ 与 d 相同的更新机制                                            │
├─────────┼──────────────────────────────────────────────────────────────────┤
│ w,l,h   │  直接使用 reg_branch 输出（不迭代细化）                           │
│ [idx3-5]│  ↳ Log 空间，通过 exp() 解码为物理尺寸                            │
├─────────┼──────────────────────────────────────────────────────────────────┤
│ sin,cos │  直接使用 reg_branch 输出（不迭代细化）                           │
│ [idx6-7]│  ↳ 线性空间，范围 [-1, 1]                                         │
├─────────┼──────────────────────────────────────────────────────────────────┤
│ vx, vy  │  直接使用 reg_branch 输出（不迭代细化）                           │
│ [idx8-9]│  ↳ 线性空间，表示速度 (m/s)                                       │
└─────────┴──────────────────────────────────────────────────────────────────┘
```

---

## 4. RWHI 适配清单

### 4.1 输出张量格式

```python
class RWHI(nn.Module):
    def forward(self, radar_heatmap, img_features):
        """
        Returns:
            query_bbox: [B, Q, 10]
            query_feat: [B, Q, embed_dims]
        """
        # ... 生成 anchor ...
        return query_bbox, query_feat
```

### 4.2 各维度处理清单

#### ✅ 维度 0: θ (角度)
```python
# 从 Radar Heatmap 提取角度
theta_rad = torch.atan2(y_offset, x_offset)  # [-π, π]
theta_norm = ((theta_rad + 2*torch.pi) % (2*torch.pi)) / (2*torch.pi)  # [0, 1]

# 或从极坐标直接获取
theta_norm = polar_angle / (2 * torch.pi)
```

#### ✅ 维度 1: d (距离) ⚠️ 需要 Clamp
```python
EPS = 1e-5
r = 65.0  # 极坐标半径，不是 map_size!

# 从笛卡尔坐标计算
distance_m = torch.sqrt(x**2 + y**2)
d_norm = (distance_m / r).clamp(EPS, 1 - EPS)  # ⚠️ 关键：防止 inverse_sigmoid 溢出
```

#### ✅ 维度 2: z (高度) ⚠️ 需要 Clamp
```python
EPS = 1e-5
pc_range_z = [-5.0, 3.0]

# 从物理高度计算
z_norm = (z_physical - pc_range_z[0]) / (pc_range_z[1] - pc_range_z[0])
z_norm = z_norm.clamp(EPS, 1 - EPS)  # ⚠️ 防止 inverse_sigmoid 溢出

# 默认值（未知高度）
z_default = 0.5  # 对应物理高度 -1.0m
```

#### ✅ 维度 3-5: w, l, h (尺寸) ⚠️ 需要 Log
```python
# 从物理尺寸转换
w_log = torch.log(width_m.clamp(min=0.1))   # 防止 log(0)
l_log = torch.log(length_m.clamp(min=0.1))
h_log = torch.log(height_m.clamp(min=0.1))

# 默认值（典型车辆）
w_default = torch.log(torch.tensor(1.8))  # ≈ 0.59
l_default = torch.log(torch.tensor(4.5))  # ≈ 1.50
h_default = torch.log(torch.tensor(1.5))  # ≈ 0.40
# 或使用 RaCFormer 默认值
h_default = 0.2  # 对应 exp(0.2) ≈ 1.22m
```

#### ✅ 维度 6-7: sin, cos (航向角)
```python
# 如果已知航向角
sin_yaw = torch.sin(yaw_rad)
cos_yaw = torch.cos(yaw_rad)

# 如果从 Radar Doppler 推断（物体运动方向）
heading = torch.atan2(vy, vx)
sin_yaw = torch.sin(heading)
cos_yaw = torch.cos(heading)

# 默认值（未知朝向）
sin_default = 0.0
cos_default = 1.0  # 对应 yaw = 0
```

#### ✅ 维度 8-9: vx, vy (速度)
```python
# 从 Radar Doppler 提取
# 通常 Radar 只能测径向速度，需要假设或估计
vx = radial_velocity * cos(theta)
vy = radial_velocity * sin(theta)

# 默认值
vx_default = 0.0
vy_default = 0.0
```

### 4.3 完整 RWHI 输出示例

```python
def generate_rwhi_query(self, radar_points, img_features):
    """
    Args:
        radar_points: [B, N, 7] (x, y, z, rcs, vr, vr_comp, timestamp)
    Returns:
        query_bbox: [B, Q, 10]
        query_feat: [B, Q, embed_dims]
    """
    B, N, _ = radar_points.shape
    EPS = 1e-5
    
    # 提取雷达点信息
    x, y, z = radar_points[..., 0], radar_points[..., 1], radar_points[..., 2]
    rcs = radar_points[..., 3]
    vr = radar_points[..., 4]  # 径向速度
    
    # ===== 计算极坐标 =====
    # θ: 角度归一化
    theta_rad = torch.atan2(y, x)
    theta_norm = ((theta_rad + 2*torch.pi) % (2*torch.pi)) / (2*torch.pi)
    
    # d: 距离归一化 (注意除以 r=65.0)
    d_raw = torch.sqrt(x**2 + y**2) / 65.0
    d_norm = d_raw.clamp(EPS, 1 - EPS)  # ⚠️ 关键
    
    # z: 高度归一化
    z_raw = (z - (-5.0)) / 8.0  # pc_range z: [-5, 3]
    z_norm = z_raw.clamp(EPS, 1 - EPS)  # ⚠️ 关键
    
    # ===== 尺寸估计 (Log 空间) =====
    # 根据 RCS 估计尺寸（示例逻辑）
    size_scale = torch.sigmoid(rcs / 10)  # RCS → 尺寸比例
    w_log = torch.log(torch.tensor(1.8)) + size_scale * 0.5
    l_log = torch.log(torch.tensor(4.5)) + size_scale * 0.5
    h_log = 0.2  # 固定默认值
    
    # ===== 航向角估计 =====
    # 假设物体沿运动方向
    heading = torch.atan2(y, x)  # 简化：指向原点
    sin_yaw = torch.sin(heading)
    cos_yaw = torch.cos(heading)
    
    # ===== 速度分解 =====
    vx = vr * torch.cos(theta_rad)
    vy = vr * torch.sin(theta_rad)
    
    # ===== 组装 query_bbox =====
    query_bbox = torch.stack([
        theta_norm,  # 0: θ
        d_norm,      # 1: d
        z_norm,      # 2: z
        w_log,       # 3: w (log)
        l_log,       # 4: l (log)
        h_log,       # 5: h (log)
        sin_yaw,     # 6: sin
        cos_yaw,     # 7: cos
        vx,          # 8: vx
        vy,          # 9: vy
    ], dim=-1)  # [B, Q, 10]
    
    return query_bbox, query_feat
```

---

## 5. 代码证据索引

| 问题 | 关键代码位置 | 行号 |
|------|-------------|------|
| 尺寸 Log 变换 | `bbox/utils.py` `encode_bbox` | 51 |
| 尺寸 Exp 解码 | `bbox/utils.py` `denormalize_bbox` | 35-37 |
| θ 更新机制 | `racformer_transformer.py` `refine_bbox` | 230 |
| d/z 更新机制 | `racformer_transformer.py` `refine_bbox` | 227-229 |
| inverse_sigmoid | `utils.py` `inverse_sigmoid` | 86-101 |
| z 初始化 | `racformer_head.py` `_init_layers` | 56 |
| h 初始化 | `racformer_head.py` `_init_layers` | 59 |
| θ 生成 | `racformer_head.py` `generate_points` | 72 |
| d 生成 | `racformer_head.py` `generate_points` | 73 |
| 极坐标转换 | `bbox/utils.py` `theta_d2xy_coods` | 82-90 |
| 笛卡尔转换 | `bbox/utils.py` `xy2theta_d_coods` | 93-106 |
| num_ray 参数 | `racformer_transformer.py` | 31, 154, 211 |
| r 常量 | `bbox/utils.py` | 82 (默认 65.0) |

---

## 附录：常见错误与解决方案

### ❌ 错误 1: d 值为 0 或 1
```
RuntimeError: inverse_sigmoid 输入溢出
```
**解决**：`d = d.clamp(1e-5, 1-1e-5)`

### ❌ 错误 2: 尺寸使用物理值
```
预测的 bbox 尺寸异常大/小
```
**解决**：`w_log = torch.log(w_physical)`

### ❌ 错误 3: d 归一化使用 map_size
```
锚点位置偏移
```
**解决**：`d_norm = distance / 65.0`（不是 102.4）

### ❌ 错误 4: θ 使用弧度值
```
角度计算错误
```
**解决**：`theta_norm = theta_rad / (2*pi)`

---

**文档版本**: v1.0  
**更新日期**: 2026-01-21  
**适用范围**: RaCFormer Query 接口适配

