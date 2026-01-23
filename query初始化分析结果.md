# RaCFormer Object Query 初始化策略深度审计报告

## 1. 代码定位

### 1.1 关键文件路径
| 文件 | 作用 |
|------|------|
| `models/racformer_head.py` | Query 初始化的核心实现 |
| `configs/racformer_r50_nuimg_704x256_f8.py` | 默认配置文件 |
| `models/bbox/utils.py` | 极坐标与笛卡尔坐标转换函数 |

### 1.2 关键函数/类
- `RaCFormer_head._init_layers()` — 初始化 Query Embedding
- `RaCFormer_head.generate_points()` — 生成几何先验点
- `theta_d2xy_coods()` — 极坐标 (θ, d) → 笛卡尔坐标 (x, y)
- `xy2theta_d_coods()` — 笛卡尔坐标 → 极坐标

---

## 2. 实现机制分析

### 2.1 Query 定义方式

**类型：** 混合模式（几何先验 + 可学习 Embedding）

```python
# models/racformer_head.py, Line 52
self.init_query_bbox = nn.Embedding(self.num_query, 10)  # (x, y, z, w, l, h, sin, cos, vx, vy)
```

- Query 的 10 个维度中：
  - **前 2 维 (theta, d)**：由几何规则生成，作为初始化先验
  - **后 8 维 (z, w, l, h, sin, cos, vx, vy)**：可学习参数

### 2.2 几何初始化的关键代码

```python
# models/racformer_head.py, Lines 61-63
def _init_layers(self):
    self.init_query_bbox = nn.Embedding(self.num_query, 10)
    # ...
    theta_d = self.generate_points()
    with torch.no_grad():
        self.init_query_bbox.weight[:, :2] = theta_d.reshape(-1, 2)  # [Q, 2]
```

### 2.3 `generate_points()` 函数完整分析

```python
# models/racformer_head.py, Lines 69-79
def generate_points(self):
    num_angles = self.num_query // self.num_clusters  # 每个圆环的点数
    angles = torch.linspace(0, 1, num_angles+1)[:-1]  # 均匀角度 [0, 1)
    distances = torch.linspace(0, 1, self.num_clusters + 2,  dtype=torch.float)[1:-1]  # 均匀距离

    angles = angles.view(num_angles, 1).expand(num_angles, self.num_clusters)
    distances = distances.view(1, self.num_clusters).expand(num_angles, self.num_clusters)

    theta_d = torch.cat([angles[..., None], distances[..., None]], dim=-1).flatten(0,1)
    return theta_d
```

---

## 3. 数量分布验证（关键）

### 3.1 默认配置参数

```python
# configs/racformer_r50_nuimg_704x256_f8.py, Lines 41-43
num_clusters = 6    # 圆环数量
num_ray = 150       # 每个圆环的采样点数
num_query = num_ray * num_clusters  # 900
```

### 3.2 每个圆环的 Query 数量计算

```python
num_angles = self.num_query // self.num_clusters
# = 900 // 6 = 150
```

**结论：每个圆环的点数固定为 150，所有圆环的 Query 数量完全相同。**

### 3.3 代码生成的分布形式

| 圆环索引 | 距离 d (归一化) | Query 数量 |
|----------|-----------------|-----------|
| 0 (内圈) | 0.143 | **150** |
| 1 | 0.286 | **150** |
| 2 | 0.429 | **150** |
| 3 | 0.571 | **150** |
| 4 | 0.714 | **150** |
| 5 (外圈) | 0.857 | **150** |

> ⚠️ **所有圆环的 Query 数量相同，并非"外圈比内圈密"的线性递增分布。**

---

## 4. 论文声称 vs 代码实现对比

### 4.1 论文描述

论文第 3.2 节 "Linearly Increasing Circular Query Initialization" 声称：

> "We employ a linearly increasing number of object queries distributed along the radial direction, which enables denser sampling at farther distances where objects tend to be smaller and more challenging to detect."
>
> 公式 (4): $N_c = N_1 + (c-1) \cdot \alpha$，其中 $\alpha$ 为增长因子

论文 Figure 3 也展示了外圈点更密集的示意图。

### 4.2 代码实际实现

```python
# 实际代码逻辑
num_angles = self.num_query // self.num_clusters  # 固定值，所有圆环相同
angles = torch.linspace(0, 1, num_angles+1)[:-1]  # 均匀角度
distances = torch.linspace(0, 1, self.num_clusters + 2)[1:-1]  # 均匀距离
```

**代码实现的是：**
- ✅ 同心圆分布（Circular Distribution）
- ✅ 使用极坐标 (θ, d) 表示
- ❌ **每个圆环点数相同（均匀分布）**
- ❌ **未实现线性递增因子 α**

---

## 5. 坐标转换机制

代码使用极坐标 (θ, d) 作为内部表示，在需要时转换为笛卡尔坐标：

```python
# models/bbox/utils.py, Lines 82-90
def theta_d2xy_coods(theta_d_coords, map_size=102.4, r=65.0):
    center = map_size / 2
    xy_coords = theta_d_coords[..., :2].clone()
    xy_coords[..., 0:1] = (center + theta_d_coords[..., 1:2]*r * torch.cos(theta_d_coords[..., 0:1]*(2 * torch.pi))) / map_size
    xy_coords[..., 1:2] = (center + theta_d_coords[..., 1:2]*r * torch.sin(theta_d_coords[..., 0:1]*(2 * torch.pi))) / map_size
    return torch.cat([xy_coords, theta_d_coords[..., 2:]], dim=-1)
```

- `theta` 范围：[0, 1] 映射到 [0, 2π]
- `d` 范围：[0, 1] 映射到 [0, r]（r=65.0m）

---

## 6. 其他初始化细节

### 6.1 固定初始化值

```python
# models/racformer_head.py, Lines 56-59
nn.init.constant_(self.init_query_bbox.weight[:, 2:3], 0.5)  # z 高度
nn.init.zeros_(self.init_query_bbox.weight[:, 8:10])         # vx, vy 速度
nn.init.constant_(self.init_query_bbox.weight[:, 5:6], 0.2)  # h 高度
```

### 6.2 Query Feature 初始化

```python
# models/racformer_head.py, Line 53
self.label_enc = nn.Embedding(self.num_classes + 1, self.embed_dims - 1)
```

Query Feature 使用类别嵌入的"未知类"向量作为初始化。

---

## 7. 搜索结果汇总

| 搜索项 | 结果 |
|--------|------|
| `query_embed` | 未使用此命名 |
| `object_query` | 未使用此命名 |
| `reference_points` | 仅在 BEV Attention 中出现 |
| `anchor` | 未使用 |
| `nn.Parameter` | 用于 `code_weights`，非 Query |
| `register_buffer` | 用于 `frustum`，非 Query |
| `torch.meshgrid` | 未使用 |
| `polar` | 仅用于 `PolarHungarianAssigner3D` |
| `circle` | 未找到 |
| `linspace` | 用于 `generate_points()` 和采样点生成 |

---

## 8. 结论

### 8.1 实现机制总结

| 属性 | 实际实现 |
|------|----------|
| 存储方式 | `nn.Embedding` (可学习) |
| 位置初始化 | 几何先验 (同心圆) |
| 坐标系统 | 极坐标 (θ, d) |
| 分布类型 | **均匀同心圆分布** |
| 是否线性递增 | **❌ 否** |

### 8.2 代码与论文一致性判定

| 论文声称 | 代码实现 | 一致性 |
|----------|----------|--------|
| 同心圆分布 | ✅ 同心圆分布 | ✅ 一致 |
| 极坐标初始化 | ✅ 极坐标 (θ, d) | ✅ 一致 |
| 线性递增 (外圈更密) | ❌ 各圆环点数相同 | ❌ **不一致** |
| 增长因子 α | ❌ 未实现 | ❌ **不一致** |

### 8.3 最终结论

> **代码实现了"同心圆初始化"（Circular Query Initialization），但未实现论文声称的"线性递增"（Linearly Increasing）策略。**
>
> 实际代码中，所有圆环的 Query 数量相同（均为 `num_query // num_clusters = 150`），属于**均匀分布的同心圆初始化**，而非**线性递增的同心圆初始化**。

---

## 9. 如果需要实现"线性递增"的修改建议

若要使代码与论文一致，可修改 `generate_points()` 函数如下：

```python
def generate_points(self):
    # 线性递增分配每个圆环的点数
    # 例如: alpha = 2, 内圈基数 base = 100
    # 则各圈点数: 100, 102, 104, 106, 108, 110
    alpha = 2  # 增长因子
    base_points = (self.num_query - alpha * self.num_clusters * (self.num_clusters - 1) // 2) // self.num_clusters
    
    all_theta_d = []
    distances = torch.linspace(0, 1, self.num_clusters + 2,  dtype=torch.float)[1:-1]
    
    for c in range(self.num_clusters):
        num_points_c = base_points + c * alpha
        angles = torch.linspace(0, 1, num_points_c + 1)[:-1]
        d = distances[c].expand(num_points_c)
        theta_d = torch.stack([angles, d], dim=-1)
        all_theta_d.append(theta_d)
    
    return torch.cat(all_theta_d, dim=0)
```

---

**报告生成时间：** 2026-01-19  
**审计工具：** 代码静态分析  
**审计范围：** RaCFormer-main/models 目录

