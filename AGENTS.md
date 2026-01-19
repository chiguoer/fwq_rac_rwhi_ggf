# Repository Guidelines

## 🗣️ Language / 语言
**Core Instruction: You must always and exclusively use Chinese (Simplified Chinese) for all responses in this project.**

## Project Structure & Module Organization
- `configs/`: experiment configs (e.g., `configs/racformer_with_rhgm_radarbevnet.py`).
- `models/`: model code, including fusion modules (`models/racformer.py`, `models/rhgm.py`, `models/radar_bev_net.py`) and CUDA ops in `models/csrc/`.
- `loaders/`: dataset builders and pipelines (nuScenes and VOD).
- `tools/`: data prep and visualization scripts (e.g., `tools/gen_sweep_info.py`).
- Entrypoints: `train.py`, `val.py`, `utils.py`.
- Assets/docs: `arch.jpg`, `MODEL_SWITCHING.md`, `RWHI_INTEGRATION_CHECK.md`.

## Build, Test, and Development Commands
- Build CUDA extensions:
  - `cd models/csrc && python setup.py build_ext --inplace`
- Single-GPU training:
  - `python train.py --config configs/racformer_with_rhgm_radarbevnet.py`
- Multi-GPU training (example):
  - `torchrun --nproc_per_node 8 train.py --config configs/racformer_r50_nuimg_704x256_f8.py`
  - `./dist_train.sh` runs the same with preset GPUs.
- Evaluation:
  - `python val.py --config configs/racformer_r50_nuimg_704x256_f8.py --weights checkpoints/racformer_r50_f8.pth`
  - `./dist_test.sh` for multi-GPU evaluation.

## Coding Style & Naming Conventions
- Python, 4-space indentation, UTF-8 headers where already present.
- Use `snake_case` for functions/variables, `CamelCase` for classes, and descriptive config dict names (`rhgm_cfg`, `radar_bev_net_cfg`).
- Keep module toggles explicit in configs (e.g., `use_rhgm=True`).

## Testing Guidelines
- Config sanity checks:
  - `python test_module_config.py --config configs/racformer_with_rhgm_radarbevnet.py`
- There is no formal unit-test suite; validate changes by running a small training/eval smoke test.

## Commit & Pull Request Guidelines
- No Git history is available in this checkout; use conventional commits for clarity (e.g., `feat: add RHGM config switch`, `fix: align radar batch size`).
- PRs should include: purpose, config(s) touched, training/eval command used, and key metrics or logs; add screenshots for visualization changes.

## Data & Configuration Tips
- Place nuScenes under `data/nuscenes/` and pretrained weights under `pretrain/`.
- Keep checkpoints in `checkpoints/` and reference them explicitly in `val.py` commands.
