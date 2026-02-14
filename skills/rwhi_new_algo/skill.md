# skills.md

## 0. 触发规则：skills 模式

- **触发条件**：每次你“读取 / 看到”本文件内容时，立即进入 **skills 模式**。
- **skills 模式的基础行为：**
  0、接下来的回复请用中文。
  1. **优先完成**当前对话中用户给出的具体指令（当前 chat 的任务永远最高优先级）。
  2. 当操作者明确说出以下指令时：

     > 请执行固定步骤之后，再开始执行下面的“强制性工作流程”

     必须先完整执行本文件中的【1. 强制性工作流程】，再继续后续任务。


---

## 1. 强制性工作流程

### 步骤 1：自审（/review 模式）

**触发条件**：只要你对代码**有任何修改**，立刻进入 `/review` 模式进行自审。

1. **Git 差异检查**

   执行：

   ```bash
   git --no-pager diff HEAD
````

用 `diff` 输出对照以下两点进行自查：

* 你的修改是否 **完整实现了用户指令** 要求的功能 / 修复点。
* 是否误改了无关代码（尤其是配置、初始化和数据流相关部分）。

2. **重点检查内容**

   在审阅代码时，重点关注以下常见故障源：

   * **BN / 归一化维度问题**

     * `BatchNorm` / `LayerNorm` 等的 `num_features` 与输入维度是否一致。
     * 是否在不稳定的 batch（稀疏、batch size = 1）上使用 BN。

   * **梯度相关问题**

     * 是否存在错误的梯度截断、`detach()`、就地操作（in-place）等导致梯度丢失。
     * 静态图中是否对同一 Tensor 做多次就地修改影响计算图版本。

   * **静态形状 / 维度约束**

     * 所有 `view` / `reshape` / `permute` / `cat` / `stack` 操作是否自洽。
     * 多分支 / 条件分支是否保证输出形状一致。

   * **数值稳定性**

     * 除法、对数、开方等操作是否有 `eps` 防护。
     * 是否可能产生 NaN / inf（例如 `(0,0)` 上做 `atan2`、`log(0)` 等）。

   如需参考，可以对照本文档末尾的【2. 故障排查经验表】中已有问题与解决方案。

---

### 步骤 2：冒烟测试

**环境准备：**

1. 激活指定 conda 环境：

   ```bash
   conda activate racrwhi
   ```

2. 根据修改模块选择对应的配置文件运行 **短程冒烟训练**：

   * 通用示例（RWHI 相关修改）：

     ```bash
     torchrun --nproc_per_node 2 train.py --config configs/racformer_with_rwhi.py
     ```

   * **本次会话的固定冒烟流程**（优先使用）：

     ```bash
     conda activate racrwhi && python train.py --config configs/racformer_with_rwhi.py
     ```

**验证要求：**

* 训练必须至少跑满 **10+ iterations**。
* 日志中 **不得出现梯度相关报错** 或 RuntimeError。
* 开始正式训练前，请**仔细检查训练日志**，确认无以下问题：

  * DDP / reduction 相关错误；
  * BN / 维度不匹配；
  * NaN / inf；
  * 其他 RuntimeError。

**重要约束：**

* 如果 **冒烟测试未跑通**：

  * **禁止立刻动代码**。
  * 必须先在对话中输出：

    * 当前错误现象；
    * 你对问题原因的初步判断；
    * 接下来计划的“最小修改方案”。
  * 等操作者确认后，再开始修改代码。

* 如果 **冒烟测试跑通**：

  * 可以继续进行：

    * 逻辑正确性检查；
    * 性能 / 数值稳定性分析；
    * 必要的微调与优化。

* 修改原则：

  * 采用 **“最小修改原则”**：

    * 只修改真正导致错误的那一小块逻辑；
    * 不重写整个文件；
    * 不随意改变原本的设计意图和数据流。

---

### 步骤 3：代码提交（Git Push）

当：

* 自审通过；
* 冒烟测试通过（训练正常运行，无错误）；

则可以执行 Git 提交与推送。

1. 添加修改：

   ```bash
   git add .
   ```

2. 使用**清晰的英文提交信息**提交，例如：

   ```bash
   git commit -m "fix: deterministic jitter for RWHI v5.2"
   ```

3. 推送到远端：

   ```bash
   git push
   ```

> 说明：Git 环境已预先配置 SSH / Token，不需要再输入用户名或密码。

---

## 2. 故障排查经验表（持续更新）

> 规则：
>
> * 每次审阅代码或运行测试时，如果出现新的 **故障现象** 且你已经有了 **明确解决方案**，请将其**简要概括**并按下表格式追加到 `skills.md` 中。
> * **仅在出现真实故障时**才记录；不要把正常流程当作故障写入。

| 现象                                                        | 解决方案                                                                                                                                  |
| --------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------- |
| ONNX 导出失败                                                 | 检查并移除 ONNX 不支持或带动态行为的算子（如动态形状、条件循环等），保证导出图中算子均为静态、可导出的运算。                                                                             |
| BN 运行时错误                                                  | 将 `nn.BatchNorm1d` 等对 batch 依赖较强的层替换为 `nn.LayerNorm` 或更稳定的归一化方式，以避免在稀疏 batch / 小 batch 下崩溃。                                           |
| 评估阶段 mAP 波动异常                                             | 确保评估阶段只使用**固定噪声**或禁用随机性（如禁用 `torch.rand` 等），避免评估过程中随机性导致 mAP 不稳定。                                                                     |
| MLP 中梯度消失                                                 | 检查 `alpha` / `anchors` 等关键中间量是否被错误地 `detach()`，确保这些张量始终保持 `requires_grad=True`，并参与反向传播。                                               |
| `Atan2Backward0` NaN / 梯度崩溃（sin/cos 为零或 NaN）              | 在 `decode_bbox` 中对 `sin` / `cos` 先做 `nan_to_num`，再对接近零的情况用掩码强制设置为 (0,1)，最后重新归一化后再调用 `atan2`，避免在 (0,0) 上反传产生 NaN。                      |
| `ClampBackward` 版本冲突（`_validate_query_bbox` 中就地 `clamp`）  | 将就地 `clamp` 改为 **out-of-place**：对 `theta` / `d` / `z` 分别 `clamp` 得到新张量后再 `concat`，避免对同一张量的就地修改破坏计算图版本。                                |
| GGF 单测导入失败（`bev_pool_v2_ext` 缺失）                          | 在 `models/__init__.py` 中对 `necks` 相关导入包裹 `try/except`，在缺失 CUDA 扩展时优雅跳过，避免非必要扩展导致 GGF 单测中断。                                            |
| `NativeRGF` 报错：`'float' object has no attribute 'clamp'`  | 在 `use_velocity_anisotropy` 分支中，保持 `sigma_y` 为 Tensor 类型（例如 `sigma_tangent = sigma_y`），避免被转换成 float 后再调用 `.clamp()`。                  |
| 冒烟训练失败：`data/nuscenes/v1.0-trainval` 不存在                  | 确认并补齐 nuScenes 数据集路径，或修正配置中的 `dataroot` 指向正确目录后，再重新运行训练。                                                                              |
| Cholesky 分解失败（`Sigma_uv` 非正定）                             | `Sigma_uv` 由 MGC 投影得到，训练早期可能非 SPD：通过「对称化 + 特征值 clamp + SPD 投影 + fallback（对角缩放）」使矩阵强制为 SPD；若新数据集仍报错，可适当调大 `eps` / 限制最大特征值。             |
| MGC debug 日志缺失（`debug_mgc=True` 仍无输出）                     | 检查 GGF cached params 是否已生成；确认 `use_image_sampling=True`；必要时将 MGC debug 输出改为 `logging` 或用 `tee` 将 stdout 保存到文件。                        |
| DataLoader worker 启动报错：`cannot pickle 'dict_keys' object` | 在 `CustomNuScenesDataset.__init__` 中，将 `dataset.eval_detection_configs.class_names` 从 `dict_keys` 转为 `list`（例如 `list(...)`），以便被正确序列化。 |
| **System RAM OOM (Exit -9) / Server Freeze** | Reduce `workers_per_gpu` (e.g., 4->2) in config to lower memory pressure from multi-sweep high-res data loading. |
| **Extremely slow training with GGF/MGC enabled** | Replace `torch.linalg` operations on small fixed-size matrices (2x2) with closed-form analytical solutions to avoid CUDA kernel launch overhead. |
| **VRAM explosion in Attention (GGA) modules** | Apply `torch.utils.checkpoint` to heavy bias calculation layers (GGA) during training to trade compute for memory. |
| **DDP training startup hang or extreme slowness** | Disable `SyncBatchNorm` (`sync_bn=False`) unless strictly necessary; set `find_unused_parameters=False` if manual dummy loss is handled. |
| **DDP RuntimeError: "parameters were not used in producing loss"** | Set `find_unused_parameters=True` in config; only set to False if you are 100% sure all params are used or manually handled. |
| **System RAM OOM / Server Freeze** | Reduce `workers_per_gpu` (4->2) to lower memory pressure from multi-sweep high-res data loading. |
| **Slow training (Kernel Overhead)** | Use scalar expansion for 2x2 Mahalanobis distance and analytical 2x2 Cholesky to avoid tiny-matmul overhead. |
| **VRAM OOM in Attention** | Use chunked geometry-bias computation to avoid materializing `[B, H, Q, M]` tensors. |

---

> 提示：
>
> * 之后如果有新的典型错误（特别是和 DDP、梯度、数值稳定性相关），可以继续按“现象 / 解决方案”的格式往上面表格追加。
> * 修改本 `skills.md` 时，同样要遵守【自审 + 冒烟测试】的流程，只是无需为纯文档变更单独跑训练。

```

---

如果你有后续想加入的新条目（比如刚刚那个 DDP 未使用参数的问题），也可以告诉我，我帮你用同样的格式写到“故障排查经验表”里。
::contentReference[oaicite:0]{index=0}
```
