# RWHI / GGF 消融与参数扫描（脚本化）

## 脚本位置

- 训练调度脚本：`tools/exp/run_rwhi_ggf_ablation_scan.sh`
- 指标提取脚本：`tools/exp/extract_ablation_metrics.py`

## 4 组消融实验

脚本内置 4 组：

1. Baseline（`configs/racformer_r50_nuimg_704x256_f8.py`）
2. 伪 Baseline（新路径，RWHI/GGF 全关）
3. RWHI-only（开 RWHI，关 GGF）
4. RWHI+GGF（开 RGF+GGA，先关 MGC）

运行：

```bash
bash tools/exp/run_rwhi_ggf_ablation_scan.sh --phase ablation --gpus 0,1
```

默认训练排程（已按当前需求固化到脚本）：

- Baseline：`total_epochs=2`，`batch_size=4`
- 伪 Baseline：`total_epochs=20`，`batch_size=2`
- 后续实验（RWHI-only / RWHI+GGF / scan）：`total_epochs=20`，`batch_size=2`

## 参数扫描

脚本内置扫描范围：

- RWHI：
  - `num_rwhi`: `300, 450, 600`
  - `diffusion_gamma`: `0.1, 0.2, 0.3`
  - `st_tau`: `0.03, 0.05, 0.08`
- GGF 组合：
  - `RGF only`
  - `RGF + GGA`
  - `RGF + GGA + MGC`
- GGA 参数（在 `RGF + GGA` 上）：
  - `bias_scale`: `0.3, 0.5, 1.0`
  - `bias_min`: `-20, -40, -80`

运行：

```bash
bash tools/exp/run_rwhi_ggf_ablation_scan.sh --phase scan --gpus 0,1
```

## 一键全跑（消融 + 扫描）

```bash
bash tools/exp/run_rwhi_ggf_ablation_scan.sh --phase all --gpus 0,1
```

## 冒烟模式（2 卡）

```bash
bash tools/exp/run_rwhi_ggf_ablation_scan.sh \
  --phase ablation \
  --gpus 0,1 \
  --max-iters 1 \
  --batch-size 2 \
  --workers-per-gpu 0 \
  --eval-interval 0
```

## 指标汇总

脚本结束后会生成 `manifest.tsv`，可直接提取指标为 CSV：

```bash
python tools/exp/extract_ablation_metrics.py \
  --manifest outputs/ablation_scan/<run_id>/manifest.tsv \
  --print-table
```
