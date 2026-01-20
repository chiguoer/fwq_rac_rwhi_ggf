# RWHI 运行文档（工程同事可直接照做）

## 背景与版本说明
RWHI 是 RaCFormer 检测头中的 Query 初始化改动（30m 双流安全/显著流）。训练/评估入口 **不变**，仅通过切换配置启用/禁用。结论：**命令不变，只换 config**。

## 环境准备
最小依赖以仓库 README 为准（无 requirements/conda yml）：
- Python 3.8
- PyTorch 2.0.0 + CUDA 11.8
- mmcv-full 1.6.0、mmdet 2.28.2、mmsegmentation 0.30.0、mmdet3d 1.0.0rc6
- 其他：setuptools==59.5.0、numpy==1.23.5、timm==0.9.2
- 可选加速：libturbojpeg + pillow-simd
- 编译 CUDA 扩展：`cd models/csrc && python setup.py build_ext --inplace`

（具体安装命令见 `README.md` 中 Environment/Compile CUDA extensions 部分）

## 配置选择
### Baseline（原始 RaCFormer）
- 配置：`configs/racformer_r50_nuimg_704x256_f8.py`

### RWHI（启用）
- 配置：`configs/racformer_with_rwhi.py`
- 该配置在 `model.pts_bbox_head` 中设置 `use_rwhi=True`，并提供 `rwhi_cfg`。

### RWHI（禁用）
RWHI 开关已存在：
- 开关位置：`model.pts_bbox_head.use_rwhi`（默认 False，在 `models/racformer_head.py` 中定义）
- RWHI 配置内部还有 `rwhi_cfg.enabled`（默认 True）

关闭方式（推荐直接关闭 `use_rwhi`，退化为原始 RaCFormer 初始化）：
```bash
python train.py --config configs/racformer_with_rwhi.py \
  --override model.pts_bbox_head.use_rwhi=False
```

关闭后回到 baseline 的等价方式（直接换 config）：
```bash
python train.py --config configs/racformer_r50_nuimg_704x256_f8.py
```

## 训练命令
### 单卡
```bash
python train.py --config configs/racformer_with_rwhi.py
```
- `--config`：训练配置文件路径。

### 多卡/分布式（DDP）
```bash
torchrun --nproc_per_node 8 train.py --config configs/racformer_with_rwhi.py
```
- `--nproc_per_node`：GPU 数量。

或直接使用脚本（默认是 baseline 配置，需自行改 config）：
```bash
bash dist_train.sh
```

### Resume / 从 checkpoint 继续
`train.py` 支持 `resume_from`，可通过 `--override` 注入：
```bash
python train.py --config configs/racformer_with_rwhi.py \
  --override resume_from=outputs/racformer_with_rwhi/2025-01-01/00-00-00/epoch_12.pth
```
- `resume_from`：已有 checkpoint 路径。

## 评估/测试命令
### 单卡
```bash
python val.py --config configs/racformer_with_rwhi.py --weights checkpoints/xxx.pth
```
- `--weights`：待评估权重路径。

### 多卡/分布式
```bash
torchrun --nproc_per_node 8 val.py --config configs/racformer_with_rwhi.py --weights checkpoints/xxx.pth
```

或直接使用脚本（默认是 baseline 配置，需自行改 config/weights）：
```bash
bash dist_test.sh
```

## 日志与产物目录说明
- 训练输出目录由 `train.py` 自动创建：`outputs/<config文件名>/<日期>/<时间>/`。
- 目录内包含：
  - `train.log`（训练日志）
  - `epoch_*.pth`（checkpoint，受 `checkpoint_config` 控制）
  - TensorBoard 日志（由 `log_config` 中的 `MyTensorboardLoggerHook` 写入）
  - 代码备份（`utils.backup_code`）

## 常见问题（<=5）
1) 报错 CUDA 不可用（`assert torch.cuda.is_available()`）
- 说明当前环境无 GPU 或 CUDA 未正确安装；需按 README 安装 PyTorch + CUDA。

2) 评估/训练报错找不到数据或 `nuscenes_infos_*.pkl`
- 确保数据位于 `data/nuscenes/`，并生成/下载 `nuscenes_infos_train_sweep.pkl`、`nuscenes_infos_test_sweep.pkl`（见 `README.md`）。

3) 训练报错找不到预训练权重
- 配置里 `load_from` 指向 `pretrain/cascade_mask_rcnn_r50_fpn_coco-20e_20e_nuim_20201009_124951-40963960.pth`，请确认文件存在或改为本地路径。

4) RWHI 报错 `Sizes of tensors must match except in dimension 1`
- 与 `radar_points` batch size 不匹配有关；确保雷达点云形状为 `[B, M, C]`，B 与图像 batch 一致（仓库已做过修复）。

5) 报错 `'RaCFormer_head' object has no attribute 'pc_range'`
- 配置中 `pts_bbox_head` 需显式传入 `pc_range`（该 repo 已在主配置中包含）。

## 附录：关键配置项解释（RWHI）
以下字段来自 `configs/racformer_with_rwhi.py` 与 `models/rwhi.py`：
- `safety_max_range`：安全流半径上限（米），0-30m 逆深度固定锚点范围。
- `num_safety` / `num_saliency`：安全流/显著流锚点数量，且 `num_query = num_safety + num_saliency`。
- `noise_eps_train` / `noise_eps_test`：TopK 采样的扰动噪声，训练与推理分别控制。
- `rcs_noise` / `rcs_clip`：RCS 噪声阈值与对数压缩上限，控制雷达显著性分布。
- `dist_gamma` / `dist_ref`：距离补偿参数，用于远场权重调制。
- `radar_channel_map`：雷达点云通道索引（x/y/z/rcs/v_r）。
- `pc_range` 推导的 `polar_radius`：在 `models/bbox/utils.py` 中由 `pc_range` 计算（`sqrt(max(|x|)^2 + max(|y|)^2)`），用于极坐标归一化。
