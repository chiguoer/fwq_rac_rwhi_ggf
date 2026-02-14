# nuScenes 可视化脚本使用说明

脚本路径：`tools/nuscenes_vis.py`

该脚本提供三类图：
- Figure A：GT + 雷达构造的 proxy RWHI 场与 Query 分布（示意图）
- Figure B：三模型对比绘图管线（需要你后续提供真实预测 JSON）
- Figure C：GT + 雷达拟合的 MGC 椭圆采样区域（示意图）

> 说明：Figure A / C 均为 **illustrative**，不代表真实网络中间输出。

---

## 1. 环境依赖

至少需要：
- Python 3
- `nuscenes-devkit`
- `numpy`
- `matplotlib`
- `pyquaternion`
- `opencv-python`（可选）

示例安装命令：

```bash
python -m pip install --user nuscenes-devkit pyquaternion opencv-python-headless
```

---

## 2. 配置路径

打开 `tools/nuscenes_vis.py`，修改 CONFIG 区域：

```python
NUSC_VERSION = "v1.0-trainval"
NUSC_DATAROOT = "/你的/nuscenes/根目录"
DEFAULT_CAM = "CAM_FRONT"
```

如需使用 Figure B，再填预测结果 JSON 路径：

```python
CAMERA_ONLY_RESULT_JSON = "/path/to/camera_only_results.json"
RACFORMER_RESULT_JSON = "/path/to/racformer_results.json"
UPGFORMER_RESULT_JSON = "/path/to/upgformer_results.json"
```

---

## 3. 运行 Figure A / Figure C（推荐）

### 方式 A：在脚本 main 里直接填 sample token

编辑 `tools/nuscenes_vis.py` 末尾：

```python
sample_tokens = [
    "你的_sample_token",
]
```

然后执行：

```bash
python tools/nuscenes_vis.py
```

### 方式 B：在 Python 中直接调用函数（更灵活）

```bash
python - <<'PY'
from tools.nuscenes_vis import load_nuscenes, plot_figure_A_bev, draw_mgc_ellipses_on_image

nusc = load_nuscenes("v1.0-trainval", "/你的/nuscenes/根目录")
token = "你的_sample_token"

plot_figure_A_bev(token, nusc, f"figure_visualize/figure_A_bev_{token}.png")
draw_mgc_ellipses_on_image(nusc, token, "CAM_FRONT", f"figure_visualize/figure_C_mgc_{token}.png")
PY
```

---

## 4. Figure B 调用方式（有真实预测后）

```python
from tools.nuscenes_vis import (
    load_nuscenes, prediction_json_to_boxes, draw_figure_B_for_sample
)

nusc = load_nuscenes("v1.0-trainval", "/你的/nuscenes/根目录")
results_cam = prediction_json_to_boxes("/path/to/camera_only_results.json")
results_rac = prediction_json_to_boxes("/path/to/racformer_results.json")
results_upg = prediction_json_to_boxes("/path/to/upgformer_results.json")

token = "你的_sample_token"
draw_figure_B_for_sample(
    nusc, token, results_cam, results_rac, results_upg,
    f"figure_visualize/figure_B_img_{token}.png",
    f"figure_visualize/figure_B_bev_{token}.png",
)
```

---

## 5. 常见问题

1. `FileNotFoundError: nuScenes dataroot 不存在`  
   - 检查 `NUSC_DATAROOT` 是否指向包含 `samples/`, `sweeps/`, `v1.0-trainval/` 的目录。

2. 图片没画出椭圆（Figure C）  
   - 该目标可能 GT 内雷达点不足（<3）或投影后在图像外，脚本会自动跳过。

3. Figure B 没有预测框  
   - 先确认 JSON 是 nuScenes result 格式，且包含对应 `sample_token`。
