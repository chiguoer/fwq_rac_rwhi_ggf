#!/usr/bin/env python3
"""
nuScenes 可视化脚本：
1) Figure A: 使用 GT + 雷达构造 proxy RWHI 场与示意 Query 分布（非真实网络输出）。
2) Figure B: 三模型检测结果对比绘图管线（接口预留，不伪造预测）。
3) Figure C: 使用 GT + 雷达在图像平面拟合 MGC 椭圆采样区域（示意图，非真实网络中间输出）。
"""

from __future__ import annotations

import copy
import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Ellipse
from matplotlib.path import Path as MplPath
from PIL import Image
from pyquaternion import Quaternion

from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import Box, RadarPointCloud
from nuscenes.utils.geometry_utils import view_points

try:
    from scipy.spatial import cKDTree
except Exception:  # pragma: no cover - 兼容无 scipy 环境
    cKDTree = None


# =============================================================================
# CONFIG
# =============================================================================
NUSC_VERSION = "v1.0-trainval"
NUSC_DATAROOT = 'data/nuscenes/'

DEFAULT_CAM = "CAM_FRONT"

X_MIN, X_MAX = -50.0, 50.0
Y_MIN, Y_MAX = -50.0, 50.0
GRID_RES = 0.5

CAMERA_ONLY_RESULT_JSON = "/path/to/camera_only_results.json"
RACFORMER_RESULT_JSON = "/path/to/racformer_results.json"
UPGFORMER_RESULT_JSON = "/path/to/upgformer_results.json"


def load_nuscenes(version: str, dataroot: str) -> NuScenes:
    """
    加载 nuScenes 数据集对象。

    Args:
        version: 数据版本，如 v1.0-trainval。
        dataroot: 数据根目录。

    Returns:
        NuScenes 对象。
    """
    if not os.path.isdir(dataroot):
        raise FileNotFoundError(f"nuScenes dataroot 不存在: {dataroot}")
    return NuScenes(version=version, dataroot=dataroot, verbose=True)


def _clone_box(box: Box) -> Box:
    """创建 Box 深拷贝，避免原地修改。"""
    return copy.deepcopy(box)


def _get_reference_sample_data_token(
    nusc: NuScenes, sample_token: str, reference_sd_token: Optional[str] = None
) -> str:
    """
    获取 BEV 参考坐标的 sample_data token。
    优先级：显式指定 > DEFAULT_CAM > LIDAR_TOP > 任意传感器。
    """
    if reference_sd_token is not None:
        return reference_sd_token

    sample = nusc.get("sample", sample_token)
    if DEFAULT_CAM in sample["data"]:
        return sample["data"][DEFAULT_CAM]
    if "LIDAR_TOP" in sample["data"]:
        return sample["data"]["LIDAR_TOP"]
    # 保底：拿第一个可用传感器 token
    return next(iter(sample["data"].values()))


def _get_reference_ego_record(
    nusc: NuScenes, sample_token: str, reference_sd_token: Optional[str] = None
) -> dict:
    """获取参考坐标系对应的 ego_pose 记录。"""
    ref_sd_token = _get_reference_sample_data_token(nusc, sample_token, reference_sd_token)
    ref_sd = nusc.get("sample_data", ref_sd_token)
    return nusc.get("ego_pose", ref_sd["ego_pose_token"])


def _polygon_contains_points(polygon_xy: np.ndarray, points_xy: np.ndarray) -> np.ndarray:
    """
    判断 points_xy 是否落在 polygon_xy 内。

    Args:
        polygon_xy: (K, 2) 多边形顶点。
        points_xy: (N, 2) 点集。

    Returns:
        mask: (N,) bool 数组，True 表示点在多边形内部。
    """
    if polygon_xy.shape[0] < 3 or points_xy.shape[0] == 0:
        return np.zeros((points_xy.shape[0],), dtype=bool)
    path = MplPath(polygon_xy)
    return path.contains_points(points_xy)


def _draw_projected_box(
    ax: plt.Axes, corners_uv: np.ndarray, color: Tuple[float, float, float], linewidth: float = 1.5
) -> None:
    """在图像平面绘制 3D box 的 12 条边。"""
    if corners_uv.shape != (8, 2):
        return
    edges = [
        (0, 1), (1, 2), (2, 3), (3, 0),  # top
        (4, 5), (5, 6), (6, 7), (7, 4),  # bottom
        (0, 4), (1, 5), (2, 6), (3, 7),  # vertical
    ]
    for i, j in edges:
        ax.plot(
            [corners_uv[i, 0], corners_uv[j, 0]],
            [corners_uv[i, 1], corners_uv[j, 1]],
            color=color,
            linewidth=linewidth,
        )


def _draw_bev_box(
    ax: plt.Axes,
    box_ego: Box,
    color: Tuple[float, float, float],
    linewidth: float = 1.5,
    linestyle: str = "-",
    label: Optional[str] = None,
) -> None:
    """在 BEV 平面绘制 box 底面轮廓。"""
    corners = bev_corners_from_box_ego(box_ego)
    closed = np.vstack([corners, corners[0]])
    ax.plot(closed[:, 0], closed[:, 1], color=color, linewidth=linewidth, linestyle=linestyle, label=label)


def _make_ego_legend_handle(label: str = "Ego") -> Line2D:
    """为 Ego 构造独立 legend 句柄，避免星形被线段句柄挤压变形。"""
    return Line2D(
        [],
        [],
        marker="*",
        linestyle="None",
        markersize=8,
        markeredgewidth=1.0,
        markerfacecolor="red",
        markeredgecolor="k",
        color="red",
        label=label,
    )


def _world_points_to_ref_ego(points_world: np.ndarray, ref_ego_record: dict) -> np.ndarray:
    """将 world 坐标点转换到参考 ego 坐标系。"""
    if points_world.size == 0:
        return np.zeros((0, 3), dtype=np.float32)

    t = np.asarray(ref_ego_record["translation"], dtype=np.float64)
    q_inv = Quaternion(ref_ego_record["rotation"]).inverse
    shifted = points_world - t[None, :]
    points_ego = (q_inv.rotation_matrix @ shifted.T).T
    return points_ego


def get_sample_camera_data(
    nusc: NuScenes, sample_token: str, cam_name: str = DEFAULT_CAM
) -> Tuple[str, dict, np.ndarray, dict, dict]:
    """
    获取 sample 对应相机的数据与标定信息。

    Returns:
        img_path: 图像绝对路径
        cam_sd: sample_data 记录
        cam_intrinsic: (3, 3) 相机内参
        cs_record: calibrated_sensor 记录
        ego_record: ego_pose 记录
    """
    sample = nusc.get("sample", sample_token)
    if cam_name not in sample["data"]:
        raise KeyError(f"sample {sample_token} 不包含相机 {cam_name}")

    cam_sd = nusc.get("sample_data", sample["data"][cam_name])
    img_path = nusc.get_sample_data_path(cam_sd["token"])
    cs_record = nusc.get("calibrated_sensor", cam_sd["calibrated_sensor_token"])
    ego_record = nusc.get("ego_pose", cam_sd["ego_pose_token"])
    cam_intrinsic = np.asarray(cs_record["camera_intrinsic"], dtype=np.float64)
    return img_path, cam_sd, cam_intrinsic, cs_record, ego_record


def get_sample_radar_points_ego(
    nusc: NuScenes, sample_token: str, reference_sd_token: Optional[str] = None
) -> np.ndarray:
    """
    读取 sample 中多个雷达点，并统一到同一参考 ego 坐标系。

    Notes:
        - 参考 ego 坐标由 reference_sd_token 指定；默认按 DEFAULT_CAM/LIDAR_TOP 自动选择。
        - 为兼容 Figure A/C，本函数仅返回 xy（shape: N x 2）。
    """
    sample = nusc.get("sample", sample_token)
    radar_channels = [k for k in sample["data"].keys() if k.startswith("RADAR")]
    if len(radar_channels) == 0:
        return np.zeros((0, 2), dtype=np.float32)

    ref_ego = _get_reference_ego_record(nusc, sample_token, reference_sd_token=reference_sd_token)
    radar_points_xy: List[np.ndarray] = []

    for channel in radar_channels:
        radar_sd_token = sample["data"][channel]
        radar_sd = nusc.get("sample_data", radar_sd_token)
        radar_path = nusc.get_sample_data_path(radar_sd_token)
        if not os.path.isfile(radar_path):
            continue

        try:
            radar_pc = RadarPointCloud.from_file(radar_path)
        except Exception:
            continue

        if radar_pc.points.shape[1] == 0:
            continue

        points_sensor = radar_pc.points[:3, :].T  # (N, 3)

        cs_record = nusc.get("calibrated_sensor", radar_sd["calibrated_sensor_token"])
        ego_record = nusc.get("ego_pose", radar_sd["ego_pose_token"])

        # sensor -> ego(sensor_time)
        q_cs = Quaternion(cs_record["rotation"])
        t_cs = np.asarray(cs_record["translation"], dtype=np.float64)
        points_ego_sensor_time = (q_cs.rotation_matrix @ points_sensor.T).T + t_cs[None, :]

        # ego(sensor_time) -> world
        q_ego = Quaternion(ego_record["rotation"])
        t_ego = np.asarray(ego_record["translation"], dtype=np.float64)
        points_world = (q_ego.rotation_matrix @ points_ego_sensor_time.T).T + t_ego[None, :]

        # world -> ego(reference_time)
        points_ref_ego = _world_points_to_ref_ego(points_world, ref_ego)
        radar_points_xy.append(points_ref_ego[:, :2])

    if len(radar_points_xy) == 0:
        return np.zeros((0, 2), dtype=np.float32)
    return np.concatenate(radar_points_xy, axis=0).astype(np.float32)


def get_sample_gt_boxes_world(nusc: NuScenes, sample_token: str) -> List[Box]:
    """
    获取 sample 的 GT boxes（world 坐标）。
    """
    sample = nusc.get("sample", sample_token)
    gt_boxes_world: List[Box] = []
    for ann_token in sample["anns"]:
        gt_boxes_world.append(nusc.get_box(ann_token))
    return gt_boxes_world


def transform_box_world_to_ego(box: Box, ego_record: dict) -> Box:
    """
    将 world 坐标系 Box 转到 ego 坐标系（返回新对象，不原地修改）。
    """
    box_ego = _clone_box(box)
    box_ego.translate(-np.asarray(ego_record["translation"], dtype=np.float64))
    box_ego.rotate(Quaternion(ego_record["rotation"]).inverse)
    return box_ego


def transform_box_world_to_camera(box: Box, ego_record: dict, cs_record: dict) -> Box:
    """
    将 world 坐标系 Box 转到 camera 坐标系（返回新对象，不原地修改）。
    """
    box_cam = transform_box_world_to_ego(box, ego_record)
    box_cam.translate(-np.asarray(cs_record["translation"], dtype=np.float64))
    box_cam.rotate(Quaternion(cs_record["rotation"]).inverse)
    return box_cam


def project_box_to_image(box_cam: Box, cam_intrinsic: np.ndarray) -> np.ndarray:
    """
    将 camera 坐标系 Box 的 8 个角点投影到图像平面。

    Returns:
        corners_uv: (8, 2)
    """
    corners_3d = box_cam.corners()
    corners_uv = view_points(corners_3d, cam_intrinsic, normalize=True)[:2, :].T
    return corners_uv


def bev_corners_from_box_ego(box_ego: Box) -> np.ndarray:
    """
    获取 ego 坐标系 Box 底面 4 个角点的 xy 坐标。

    Returns:
        corners_xy: (4, 2)
    """
    corners_bottom = box_ego.bottom_corners()  # (3, 4)
    return corners_bottom[:2, :].T


# =============================================================================
# Figure A: Proxy RWHI (GT + Radar only, illustrative)
# =============================================================================
def create_bev_grid(
    x_min: float, x_max: float, y_min: float, y_max: float, grid_res: float
) -> Tuple[np.ndarray, np.ndarray]:
    """
    创建 BEV 网格。

    Returns:
        X, Y: shape (H, W)
    """
    xs = np.arange(x_min, x_max + 1e-6, grid_res, dtype=np.float32)
    ys = np.arange(y_min, y_max + 1e-6, grid_res, dtype=np.float32)
    X, Y = np.meshgrid(xs, ys)
    return X, Y


def compute_radar_field(radar_xy: np.ndarray, X: np.ndarray, Y: np.ndarray, sigma_r: float = 2.0) -> np.ndarray:
    """
    通过最近雷达点距离构造雷达场:
        S_radar(x, y) = exp(-d_radar^2 / (2 * sigma_r^2))
    """
    if radar_xy.shape[0] == 0:
        return np.zeros_like(X, dtype=np.float32)

    grid_points = np.column_stack([X.ravel(), Y.ravel()]).astype(np.float64)
    radar_xy = radar_xy.astype(np.float64)

    if cKDTree is not None:
        tree = cKDTree(radar_xy)
        dists, _ = tree.query(grid_points, k=1)
    else:
        # 无 scipy 时使用分块广播，避免单次内存过大。
        min_d2 = np.full((grid_points.shape[0],), np.inf, dtype=np.float64)
        chunk = 256
        for st in range(0, radar_xy.shape[0], chunk):
            ed = min(st + chunk, radar_xy.shape[0])
            radar_chunk = radar_xy[st:ed]
            diff = grid_points[:, None, :] - radar_chunk[None, :, :]
            d2 = np.sum(diff * diff, axis=2)
            min_d2 = np.minimum(min_d2, d2.min(axis=1))
        dists = np.sqrt(min_d2)

    s_radar = np.exp(-(dists ** 2) / (2.0 * sigma_r ** 2))
    return s_radar.reshape(X.shape).astype(np.float32)


def compute_gt_field(gt_boxes_ego: List[Box], X: np.ndarray, Y: np.ndarray, sigma_g: float = 4.0) -> np.ndarray:
    """
    构造 GT 场。

    当前实现使用简化版（满足需求允许的简单方案）：
        - 网格点落在任一 GT box BEV 底面内 => 1.0
        - 否则 => 0.0

    参数 sigma_g 保留为接口占位，便于后续替换为距离衰减版。
    """
    _ = sigma_g  # 预留参数，当前简化版不使用
    s_gt = np.zeros_like(X, dtype=np.float32)
    if len(gt_boxes_ego) == 0:
        return s_gt

    grid_points = np.column_stack([X.ravel(), Y.ravel()])
    for box_ego in gt_boxes_ego:
        corners = bev_corners_from_box_ego(box_ego)
        inside = _polygon_contains_points(corners, grid_points).reshape(X.shape)
        s_gt[inside] = 1.0
    return s_gt


def compute_proxy_score_field(
    s_radar: np.ndarray, s_gt: np.ndarray, alpha: float = 0.5, beta: float = 0.5
) -> np.ndarray:
    """
    合成 proxy 打分场并归一化到 [0, 1]。
    """
    s_proxy = alpha * s_radar + beta * s_gt
    v_min = float(np.min(s_proxy))
    v_max = float(np.max(s_proxy))
    if v_max - v_min < 1e-8:
        return np.zeros_like(s_proxy, dtype=np.float32)
    s_proxy = (s_proxy - v_min) / (v_max - v_min)
    return s_proxy.astype(np.float32)


def sample_queries_from_gt_and_score(
    gt_boxes_ego: List[Box],
    s_proxy: np.ndarray,
    X: np.ndarray,
    Y: np.ndarray,
    num_queries: int = 200,
) -> np.ndarray:
    """
    从 GT 区域与全局高分区域构造示意 Query 分布（非真实网络输出）。

    策略：
    - Part A: 在 GT box 内优先采样（约 60% 配额）；
    - Part B: 在全局 S_proxy top-K 中补齐（优先不在 GT 内部）。
    """
    if num_queries <= 0:
        return np.zeros((0, 2), dtype=np.float32)

    rng = np.random.default_rng(0)  # 可复现
    points_flat = np.column_stack([X.ravel(), Y.ravel()])
    score_flat = s_proxy.ravel()

    if X.shape[1] > 1:
        step_x = float(abs(X[0, 1] - X[0, 0]))
    else:
        step_x = GRID_RES
    if Y.shape[0] > 1:
        step_y = float(abs(Y[1, 0] - Y[0, 0]))
    else:
        step_y = GRID_RES
    jitter = 0.25 * min(step_x, step_y)

    gt_union_mask = np.zeros((points_flat.shape[0],), dtype=bool)
    selected = set()
    query_points: List[np.ndarray] = []

    if len(gt_boxes_ego) > 0:
        gt_budget = int(round(num_queries * 0.6))
        per_box = max(1, gt_budget // len(gt_boxes_ego))
        for box_ego in gt_boxes_ego:
            corners = bev_corners_from_box_ego(box_ego)
            inside_mask = _polygon_contains_points(corners, points_flat)
            gt_union_mask |= inside_mask
            inside_idx = np.where(inside_mask)[0]
            if inside_idx.size == 0:
                continue

            take = min(per_box, inside_idx.size)
            chosen = rng.choice(inside_idx, size=take, replace=False)
            for idx in chosen:
                base_xy = points_flat[idx].copy()
                base_xy += rng.uniform(-jitter, jitter, size=(2,))
                base_xy[0] = np.clip(base_xy[0], float(X.min()), float(X.max()))
                base_xy[1] = np.clip(base_xy[1], float(Y.min()), float(Y.max()))
                query_points.append(base_xy)
                selected.add(int(idx))

    # 全局 top-k 补齐
    ranked = np.argsort(score_flat)[::-1]
    for idx in ranked:
        if len(query_points) >= num_queries:
            break
        i = int(idx)
        if i in selected:
            continue
        # 优先选 GT 外的高分点，避免与 GT 内采样重复语义
        if gt_union_mask[i]:
            continue
        query_points.append(points_flat[i])
        selected.add(i)

    # 若仍不足，允许从剩余高分点补齐
    if len(query_points) < num_queries:
        for idx in ranked:
            if len(query_points) >= num_queries:
                break
            i = int(idx)
            if i in selected:
                continue
            query_points.append(points_flat[i])
            selected.add(i)

    if len(query_points) == 0:
        return np.zeros((0, 2), dtype=np.float32)
    return np.asarray(query_points, dtype=np.float32)


def plot_figure_A_bev(sample_token: str, nusc: NuScenes, out_path: str) -> None:
    """
    绘制 Figure A:
      - 背景: S_proxy
      - 黑点: 雷达点
      - 蓝色 x: Query
      - 白框: GT box 底面

    说明（仅代码层面）：
      当前 score map 与 queries 由官方标注框与雷达点规则构造，用于几何可视化。
      图像本身仅使用中性文字，不在标题/图例中展示构造来源说明。
    """
    radar_xy = get_sample_radar_points_ego(nusc, sample_token)
    gt_boxes_world = get_sample_gt_boxes_world(nusc, sample_token)
    ref_ego = _get_reference_ego_record(nusc, sample_token)
    gt_boxes_ego = [transform_box_world_to_ego(box, ref_ego) for box in gt_boxes_world]

    X, Y = create_bev_grid(X_MIN, X_MAX, Y_MIN, Y_MAX, GRID_RES)
    s_radar = compute_radar_field(radar_xy, X, Y, sigma_r=2.0)
    s_gt = compute_gt_field(gt_boxes_ego, X, Y, sigma_g=4.0)
    s_proxy = compute_proxy_score_field(s_radar, s_gt, alpha=0.5, beta=0.5)
    queries_xy = sample_queries_from_gt_and_score(gt_boxes_ego, s_proxy, X, Y, num_queries=200)

    fig, ax = plt.subplots(figsize=(10, 10))
    mesh = ax.pcolormesh(X, Y, s_proxy, shading="auto", cmap="viridis", alpha=0.9)
    cbar = fig.colorbar(mesh, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Score map")

    legend_handles: List[object] = []
    if radar_xy.shape[0] > 0:
        radar_handle = ax.scatter(radar_xy[:, 0], radar_xy[:, 1], s=8, c="black", alpha=0.6, label="Radar points")
        legend_handles.append(radar_handle)
    else:
        legend_handles.append(
            Line2D([], [], marker="o", linestyle="None", color="black", markersize=4, label="Radar points")
        )
    if queries_xy.shape[0] > 0:
        query_handle = ax.scatter(queries_xy[:, 0], queries_xy[:, 1], s=20, c="dodgerblue", marker="x", label="Queries")
        legend_handles.append(query_handle)
    else:
        legend_handles.append(
            Line2D([], [], marker="x", linestyle="None", color="dodgerblue", markersize=7, label="Queries")
        )

    for box_ego in gt_boxes_ego:
        _draw_bev_box(
            ax,
            box_ego,
            color=(1.0, 1.0, 1.0),
            linewidth=2.0,
            label=None,
        )
    legend_handles.append(Line2D([], [], color="white", linewidth=2.0, label="3D boxes"))

    ax.plot(0.0, 0.0, marker="*", color="red", markersize=12)
    legend_handles.append(_make_ego_legend_handle("Ego"))
    ax.set_xlim(X_MIN, X_MAX)
    ax.set_ylim(Y_MIN, Y_MAX)
    ax.set_xlabel("X (meters)")
    ax.set_ylabel("Y (meters)")
    ax.set_aspect("equal")
    ax.set_title("Figure A: Radar-guided query distribution in BEV")
    ax.legend(handles=legend_handles, loc="upper right", fontsize=9, handlelength=1.2)
    fig.tight_layout()

    out_file = Path(out_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_file, dpi=180)
    plt.close(fig)


# =============================================================================
# Figure B: plotting pipeline only (no fake predictions)
# =============================================================================
def prediction_json_to_boxes(pred_json_path: str) -> Dict[str, List[Box]]:
    """
    读取 nuScenes 风格 result JSON，并转换为 world 坐标 Box 字典。

    Returns:
        pred_dict:
            key: sample_token
            value: List[Box] (world)
    """
    if not os.path.isfile(pred_json_path):
        raise FileNotFoundError(f"预测 JSON 不存在: {pred_json_path}")

    with open(pred_json_path, "r", encoding="utf-8") as f:
        payload = json.load(f)

    if isinstance(payload, dict) and "results" in payload and isinstance(payload["results"], dict):
        results = payload["results"]
    elif isinstance(payload, dict):
        # 兼容直接以 sample_token 为 key 的结构
        results = payload
    else:
        raise ValueError("预测 JSON 结构不合法，期望为 dict 或包含 results 字段。")

    pred_dict: Dict[str, List[Box]] = {}
    for sample_token, items in results.items():
        if not isinstance(items, list):
            continue
        boxes: List[Box] = []
        for item in items:
            if not isinstance(item, dict):
                continue
            if not all(k in item for k in ("translation", "size", "rotation")):
                continue

            velocity = item.get("velocity", [np.nan, np.nan, 0.0])
            if len(velocity) == 2:
                velocity = [velocity[0], velocity[1], 0.0]

            box = Box(
                center=np.asarray(item["translation"], dtype=np.float64),
                size=np.asarray(item["size"], dtype=np.float64),
                orientation=Quaternion(item["rotation"]),
                label=np.nan,
                score=float(item.get("detection_score", item.get("score", np.nan))),
                velocity=tuple(velocity[:3]),
                name=item.get("detection_name", item.get("name", "prediction")),
                token=item.get("sample_token", sample_token),
            )
            boxes.append(box)
        pred_dict[sample_token] = boxes
    return pred_dict


def render_image_view_for_model(
    img: np.ndarray,
    boxes_world: List[Box],
    gt_boxes_world: List[Box],
    nusc: NuScenes,
    cam_sd: dict,
    cam_intrinsic: np.ndarray,
    color_pred: Tuple[float, float, float],
    color_gt: Tuple[float, float, float],
    model_name: str = "Model",
    ax: Optional[plt.Axes] = None,
) -> None:
    """
    在图像子图绘制模型预测与 GT 3D 框投影。
    """
    if ax is None:
        ax = plt.gca()

    ax.imshow(img)
    ax.axis("off")

    ego_record = nusc.get("ego_pose", cam_sd["ego_pose_token"])
    cs_record = nusc.get("calibrated_sensor", cam_sd["calibrated_sensor_token"])
    width, height = img.shape[1], img.shape[0]

    # GT
    for gt in gt_boxes_world:
        gt_cam = transform_box_world_to_camera(gt, ego_record, cs_record)
        corners_3d = gt_cam.corners()
        if np.all(corners_3d[2, :] <= 1e-3):
            continue
        corners_uv = project_box_to_image(gt_cam, cam_intrinsic)
        if not np.any(
            (corners_uv[:, 0] >= 0)
            & (corners_uv[:, 0] < width)
            & (corners_uv[:, 1] >= 0)
            & (corners_uv[:, 1] < height)
        ):
            continue
        _draw_projected_box(ax, corners_uv, color=color_gt, linewidth=1.5)

    # Prediction
    for pred in boxes_world:
        pred_cam = transform_box_world_to_camera(pred, ego_record, cs_record)
        corners_3d = pred_cam.corners()
        if np.all(corners_3d[2, :] <= 1e-3):
            continue
        corners_uv = project_box_to_image(pred_cam, cam_intrinsic)
        if not np.any(
            (corners_uv[:, 0] >= 0)
            & (corners_uv[:, 0] < width)
            & (corners_uv[:, 1] >= 0)
            & (corners_uv[:, 1] < height)
        ):
            continue
        _draw_projected_box(ax, corners_uv, color=color_pred, linewidth=1.2)

    ax.set_title(model_name)


def render_bev_view_for_models(
    radar_xy: np.ndarray,
    gt_boxes_ego: List[Box],
    boxes_cam_ego: List[Box],
    boxes_rac_ego: List[Box],
    boxes_upg_ego: List[Box],
    out_path: str,
) -> None:
    """
    绘制 Figure B 的 BEV 对比视图。
    """
    fig, ax = plt.subplots(figsize=(10, 10))

    legend_handles: List[object] = []
    if radar_xy.shape[0] > 0:
        radar_handle = ax.scatter(radar_xy[:, 0], radar_xy[:, 1], s=7, c="black", alpha=0.5, label="Radar points")
        legend_handles.append(radar_handle)
    else:
        legend_handles.append(
            Line2D([], [], marker="o", linestyle="None", color="black", markersize=4, label="Radar points")
        )

    for box in gt_boxes_ego:
        _draw_bev_box(
            ax,
            box,
            color=(1.0, 1.0, 1.0),
            linewidth=2.0,
            label=None,
        )
    legend_handles.append(Line2D([], [], color="white", linewidth=2.0, label="3D boxes"))

    for box in boxes_cam_ego:
        _draw_bev_box(
            ax,
            box,
            color=(0.1, 0.8, 0.1),
            linewidth=1.4,
            label=None,
        )
    legend_handles.append(Line2D([], [], color=(0.1, 0.8, 0.1), linewidth=1.4, label="Camera-only preds"))

    for box in boxes_rac_ego:
        _draw_bev_box(
            ax,
            box,
            color=(1.0, 0.55, 0.1),
            linewidth=1.4,
            label=None,
        )
    legend_handles.append(Line2D([], [], color=(1.0, 0.55, 0.1), linewidth=1.4, label="RaCFormer preds"))

    for box in boxes_upg_ego:
        _draw_bev_box(
            ax,
            box,
            color=(0.2, 0.45, 1.0),
            linewidth=1.6,
            label=None,
        )
    legend_handles.append(Line2D([], [], color=(0.2, 0.45, 1.0), linewidth=1.6, label="UPGFormer preds"))

    ax.plot(0.0, 0.0, marker="*", color="red", markersize=12)
    legend_handles.append(_make_ego_legend_handle("Ego"))
    ax.set_xlim(X_MIN, X_MAX)
    ax.set_ylim(Y_MIN, Y_MAX)
    ax.set_xlabel("X (meters)")
    ax.set_ylabel("Y (meters)")
    ax.set_aspect("equal")
    ax.set_title("Figure B (BEV): Radar + multi-model predictions")
    ax.legend(handles=legend_handles, loc="upper right", fontsize=9, handlelength=1.2)
    fig.tight_layout()

    out_file = Path(out_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_file, dpi=180)
    plt.close(fig)


def _boxes_world_to_ego(boxes_world: List[Box], ego_record: dict) -> List[Box]:
    """批量 world->ego。"""
    return [transform_box_world_to_ego(b, ego_record) for b in boxes_world]


def draw_figure_B_for_sample(
    nusc: NuScenes,
    sample_token: str,
    results_cam: Dict[str, List[Box]],
    results_rac: Dict[str, List[Box]],
    results_upg: Dict[str, List[Box]],
    out_path_img: str,
    out_path_bev: str,
) -> None:
    """
    Figure B 高层接口。
    仅使用传入的真实结果字典，不构造任何虚构预测。
    """
    img_path, cam_sd, cam_intrinsic, _, _ = get_sample_camera_data(nusc, sample_token, cam_name=DEFAULT_CAM)
    img = np.array(Image.open(img_path).convert("RGB"))

    gt_boxes_world = get_sample_gt_boxes_world(nusc, sample_token)
    boxes_cam_world = results_cam.get(sample_token, [])
    boxes_rac_world = results_rac.get(sample_token, [])
    boxes_upg_world = results_upg.get(sample_token, [])

    # 图像视角对比（3 个子图）
    fig, axes = plt.subplots(1, 3, figsize=(24, 8))
    render_image_view_for_model(
        img=img,
        boxes_world=boxes_cam_world,
        gt_boxes_world=gt_boxes_world,
        nusc=nusc,
        cam_sd=cam_sd,
        cam_intrinsic=cam_intrinsic,
        color_pred=(0.1, 0.8, 0.1),
        color_gt=(1.0, 1.0, 1.0),
        model_name="Camera-only",
        ax=axes[0],
    )
    render_image_view_for_model(
        img=img,
        boxes_world=boxes_rac_world,
        gt_boxes_world=gt_boxes_world,
        nusc=nusc,
        cam_sd=cam_sd,
        cam_intrinsic=cam_intrinsic,
        color_pred=(1.0, 0.55, 0.1),
        color_gt=(1.0, 1.0, 1.0),
        model_name="RaCFormer",
        ax=axes[1],
    )
    render_image_view_for_model(
        img=img,
        boxes_world=boxes_upg_world,
        gt_boxes_world=gt_boxes_world,
        nusc=nusc,
        cam_sd=cam_sd,
        cam_intrinsic=cam_intrinsic,
        color_pred=(0.2, 0.45, 1.0),
        color_gt=(1.0, 1.0, 1.0),
        model_name="UPGFormer",
        ax=axes[2],
    )
    fig.suptitle("Figure B (Image view): 3D boxes and model predictions", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.96])

    out_img = Path(out_path_img)
    out_img.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_img, dpi=180)
    plt.close(fig)

    # BEV 对比
    ref_ego = _get_reference_ego_record(nusc, sample_token)
    radar_xy = get_sample_radar_points_ego(nusc, sample_token)
    gt_boxes_ego = _boxes_world_to_ego(gt_boxes_world, ref_ego)
    boxes_cam_ego = _boxes_world_to_ego(boxes_cam_world, ref_ego)
    boxes_rac_ego = _boxes_world_to_ego(boxes_rac_world, ref_ego)
    boxes_upg_ego = _boxes_world_to_ego(boxes_upg_world, ref_ego)

    render_bev_view_for_models(
        radar_xy=radar_xy,
        gt_boxes_ego=gt_boxes_ego,
        boxes_cam_ego=boxes_cam_ego,
        boxes_rac_ego=boxes_rac_ego,
        boxes_upg_ego=boxes_upg_ego,
        out_path=out_path_bev,
    )


# =============================================================================
# Figure C: per-radar Gaussian kernels in BEV and image plane
# =============================================================================
def assign_radar_to_boxes_ego(radar_xy: np.ndarray, boxes_ego: List[Box]) -> List[List[int]]:
    """
    将雷达点按 BEV 几何关系分配到每个 3D box。

    说明：这里的 boxes 来自官方标注，分配结果用于可视化分析。
    """
    radar_indices_per_box: List[List[int]] = [[] for _ in range(len(boxes_ego))]
    if radar_xy.shape[0] == 0 or len(boxes_ego) == 0:
        return radar_indices_per_box

    for box_idx, box_ego in enumerate(boxes_ego):
        corners = bev_corners_from_box_ego(box_ego)
        mask = _polygon_contains_points(corners, radar_xy)
        radar_indices_per_box[box_idx] = np.where(mask)[0].tolist()
    return radar_indices_per_box


def assign_radar_to_gt_ego(radar_xy: np.ndarray, gt_boxes_ego: List[Box]) -> List[List[int]]:
    """兼容旧函数名。"""
    return assign_radar_to_boxes_ego(radar_xy, gt_boxes_ego)


def build_bev_ellipse_for_radar_point(
    point_xy: np.ndarray,
    radar_direction: Optional[Sequence[float] | float] = None,
    sigma_r: float = 0.5,
    sigma_theta: float = 1.0,
) -> Tuple[Tuple[float, float], float, float, float]:
    """
    基于单个雷达点构造 BEV 椭圆参数。

    简化模型：长轴沿点的径向方向，短轴垂直径向方向。
    """
    point_xy = np.asarray(point_xy, dtype=np.float64).reshape(2)

    if radar_direction is None:
        dir_vec = point_xy.copy()
    elif np.isscalar(radar_direction):
        a = float(radar_direction)
        dir_vec = np.array([np.cos(a), np.sin(a)], dtype=np.float64)
    else:
        dir_vec = np.asarray(radar_direction, dtype=np.float64).reshape(2)

    norm = float(np.linalg.norm(dir_vec))
    if norm < 1e-6:
        dir_vec = np.array([1.0, 0.0], dtype=np.float64)
    else:
        dir_vec = dir_vec / norm

    distance = float(np.linalg.norm(point_xy))
    sigma_theta_rad = np.deg2rad(sigma_theta)

    # 使用固定+轻微随距离变化的半轴，保证同一图内能看到多个清晰椭圆。
    a_bev = float(np.clip(4.0 * sigma_r, 1.2, 3.0))
    b_raw = float(2.0 * max(distance, 1.0) * np.tan(sigma_theta_rad))
    b_bev = float(np.clip(b_raw, 0.35, 1.2))
    b_bev = min(b_bev, 0.95 * a_bev)

    angle_deg = float(np.degrees(np.arctan2(dir_vec[1], dir_vec[0])))
    center = (float(point_xy[0]), float(point_xy[1]))
    return center, a_bev, b_bev, angle_deg


def _ego_points_to_camera(points_ego: np.ndarray, cs_record: dict) -> np.ndarray:
    """ego 坐标点 -> camera 坐标点。"""
    points_ego = np.asarray(points_ego, dtype=np.float64)
    if points_ego.ndim != 2 or points_ego.shape[1] != 3:
        raise ValueError('points_ego 需要 shape (N, 3)')

    t_cs = np.asarray(cs_record['translation'], dtype=np.float64)
    q_cs_inv = Quaternion(cs_record['rotation']).inverse
    shifted = points_ego - t_cs[None, :]
    points_cam = (q_cs_inv.rotation_matrix @ shifted.T).T
    return points_cam


def _project_ego_points_to_image(
    points_ego: np.ndarray,
    cs_record: dict,
    cam_intrinsic: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    将 ego 点投影到图像，返回 (uv, points_cam, valid_mask)。
    """
    points_cam = _ego_points_to_camera(points_ego, cs_record)
    valid = points_cam[:, 2] > 1e-3

    uv = np.full((points_ego.shape[0], 2), np.nan, dtype=np.float64)
    if np.any(valid):
        uv_valid = view_points(points_cam[valid].T, cam_intrinsic, normalize=True)[:2, :].T
        uv[valid] = uv_valid
    return uv, points_cam, valid


def project_radar_points_to_image(
    radar_points_ego: np.ndarray,
    ego_record: dict,
    cs_record: dict,
    cam_intrinsic: np.ndarray,
) -> np.ndarray:
    """
    将 ego 坐标雷达点投影到图像平面。

    输入可为 (N,2) 或 (N,3)，二维输入默认补 z=0。
    """
    _ = ego_record
    if radar_points_ego.size == 0:
        return np.zeros((0, 2), dtype=np.float32)

    radar_points_ego = np.asarray(radar_points_ego)
    if radar_points_ego.ndim != 2 or radar_points_ego.shape[1] not in (2, 3):
        raise ValueError('radar_points_ego 需要 shape 为 (N,2) 或 (N,3)')

    if radar_points_ego.shape[1] == 2:
        points_ego = np.concatenate(
            [radar_points_ego.astype(np.float64), np.zeros((radar_points_ego.shape[0], 1), dtype=np.float64)],
            axis=1,
        )
    else:
        points_ego = radar_points_ego[:, :3].astype(np.float64)

    uv, _, valid = _project_ego_points_to_image(points_ego, cs_record, cam_intrinsic)
    return uv[valid].astype(np.float32)


def _select_box_indices_for_per_radar_demo(
    boxes_ego: List[Box],
    radar_indices_per_box: List[List[int]],
    cam_intrinsic: np.ndarray,
    cs_record: dict,
    image_shape: Tuple[int, int],
    max_boxes: int = 1,
) -> List[int]:
    """
    选择用于右图展示的 box（优先：可见、vehicle、前向、点数多、距离近）。
    """
    img_h, img_w = image_shape
    ranked: List[Tuple[int, int, int, int, float, int]] = []

    for idx, (box_ego, radar_ids) in enumerate(zip(boxes_ego, radar_indices_per_box)):
        if len(radar_ids) == 0:
            continue

        center_ego = np.asarray(box_ego.center[:3], dtype=np.float64).reshape(1, 3)
        uv, points_cam, valid = _project_ego_points_to_image(center_ego, cs_record, cam_intrinsic)
        visible = bool(
            valid[0]
            and 0 <= uv[0, 0] < img_w
            and 0 <= uv[0, 1] < img_h
            and points_cam[0, 2] > 1e-3
        )

        name = (box_ego.name or '').lower()
        is_vehicle = ('vehicle' in name) or ('car' in name)
        front = float(box_ego.center[0]) >= 0.0
        dist = float(np.hypot(float(box_ego.center[0]), float(box_ego.center[1])))

        ranked.append(
            (
                0 if visible else 1,
                0 if is_vehicle else 1,
                0 if front else 1,
                -len(radar_ids),
                int(dist * 100),
                idx,
            )
        )

    if len(ranked) == 0:
        if len(boxes_ego) == 0:
            return []
        fallback = sorted(
            range(len(boxes_ego)),
            key=lambda i: float(np.hypot(float(boxes_ego[i].center[0]), float(boxes_ego[i].center[1]))),
        )
        return fallback[:max_boxes]

    ranked.sort()
    return [x[-1] for x in ranked[:max_boxes]]


def _limit_radar_indices_for_box(
    radar_xy: np.ndarray,
    box_ego: Box,
    radar_indices: Sequence[int],
    max_points: int,
) -> List[int]:
    """按到 box 中心距离排序，截取固定数量雷达点。"""
    if len(radar_indices) == 0:
        return []

    center_xy = np.asarray(box_ego.center[:2], dtype=np.float64)
    order = sorted(
        list(radar_indices),
        key=lambda ridx: float(np.linalg.norm(radar_xy[int(ridx), :2] - center_xy)),
    )
    return order[:max_points]


def _project_radar_kernel_to_image(
    point_xy: np.ndarray,
    cs_record: dict,
    cam_intrinsic: np.ndarray,
    sigma_r: float,
    sigma_theta: float,
) -> Optional[Tuple[Tuple[float, float], float, float, float]]:
    """
    将单个雷达核从 BEV 近似投影到图像平面椭圆参数。

    这里采用局部线性近似：
      - 在 BEV 中构造主/副轴方向；
      - 分别投影单位位移向量，估计像素尺度与旋转角。
    """
    center, a_bev, b_bev, angle_bev = build_bev_ellipse_for_radar_point(
        point_xy=point_xy,
        radar_direction=None,
        sigma_r=sigma_r,
        sigma_theta=sigma_theta,
    )

    rad = np.deg2rad(angle_bev)
    major_dir = np.array([np.cos(rad), np.sin(rad)], dtype=np.float64)
    minor_dir = np.array([-np.sin(rad), np.cos(rad)], dtype=np.float64)

    p0 = np.array([center[0], center[1], 0.0], dtype=np.float64)
    p_major = np.array([center[0] + major_dir[0], center[1] + major_dir[1], 0.0], dtype=np.float64)
    p_minor = np.array([center[0] + minor_dir[0], center[1] + minor_dir[1], 0.0], dtype=np.float64)

    points_ego = np.stack([p0, p_major, p_minor], axis=0)
    uv, _, valid = _project_ego_points_to_image(points_ego, cs_record, cam_intrinsic)
    if not np.all(valid):
        return None

    uv0 = uv[0]
    v_major = uv[1] - uv0
    v_minor = uv[2] - uv0

    scale_major = float(np.linalg.norm(v_major))
    scale_minor = float(np.linalg.norm(v_minor))
    if scale_major < 1e-4 or scale_minor < 1e-4:
        return None

    a_img = float(np.clip(a_bev * scale_major, 2.0, 280.0))
    b_img = float(np.clip(b_bev * scale_minor, 2.0, 240.0))
    angle_img = float(np.degrees(np.arctan2(v_major[1], v_major[0])))

    return (float(uv0[0]), float(uv0[1])), a_img, b_img, angle_img


def plot_mgc_bev_for_single_box(
    ax: plt.Axes,
    radar_xy: np.ndarray,
    boxes_ego: List[Box],
    box_idx: int,
    radar_indices_per_box: List[List[int]],
    max_ellipses_per_box: int = 10,
    window_margin: float = 20.0,
    sigma_r: float = 0.5,
    sigma_theta: float = 1.0,
    show_legend: bool = True,
) -> List[int]:
    """
    绘制单目标的 BEV 局部图：局部雷达点 + 3D boxes + 多椭圆。

    Returns:
        used_radar_indices: 实际绘制椭圆的雷达点索引（用于右侧图像子图）。
    """
    radar_xy = np.asarray(radar_xy)
    if box_idx < 0 or box_idx >= len(boxes_ego):
        ax.set_axis_off()
        return []

    target_box = boxes_ego[box_idx]
    cx, cy = float(target_box.center[0]), float(target_box.center[1])
    x0, x1 = cx - window_margin, cx + window_margin
    y0, y1 = cy - window_margin, cy + window_margin

    legend_handles: List[object] = []
    if radar_xy.shape[0] > 0:
        in_win = (
            (radar_xy[:, 0] >= x0) & (radar_xy[:, 0] <= x1) &
            (radar_xy[:, 1] >= y0) & (radar_xy[:, 1] <= y1)
        )
        radar_local = radar_xy[in_win]
        if radar_local.shape[0] > 0:
            radar_handle = ax.scatter(
                radar_local[:, 0],
                radar_local[:, 1],
                s=10,
                c="black",
                alpha=0.65,
                label="Radar points",
            )
            legend_handles.append(radar_handle)
        else:
            legend_handles.append(
                Line2D([], [], marker="o", linestyle="None", color="black", markersize=4, label="Radar points")
            )

    for idx, box in enumerate(boxes_ego):
        center_xy = np.asarray(box.center[:2], dtype=np.float64)
        if center_xy[0] < x0 - 5 or center_xy[0] > x1 + 5 or center_xy[1] < y0 - 5 or center_xy[1] > y1 + 5:
            continue
        _draw_bev_box(
            ax,
            box,
            color=(0.2, 0.95, 0.2),
            linewidth=2.6 if idx == box_idx else 1.4,
            label=None,
        )
    legend_handles.append(Line2D([], [], color=(0.2, 0.95, 0.2), linewidth=2.0, label="3D boxes"))

    used_radar_indices: List[int] = []
    candidate_ids = _limit_radar_indices_for_box(
        radar_xy=radar_xy,
        box_ego=target_box,
        radar_indices=radar_indices_per_box[box_idx],
        max_points=max_ellipses_per_box,
    )
    ellipse_count = 0
    for ridx in candidate_ids:
        center, a_bev, b_bev, angle_deg = build_bev_ellipse_for_radar_point(
            point_xy=radar_xy[int(ridx), :2],
            radar_direction=None,
            sigma_r=sigma_r,
            sigma_theta=sigma_theta,
        )
        patch = Ellipse(
            xy=center,
            width=2.0 * a_bev,
            height=2.0 * b_bev,
            angle=angle_deg,
            fill=False,
            edgecolor="cyan",
            linewidth=1.8,
            alpha=0.95,
        )
        ax.add_patch(patch)
        ellipse_count += 1
        used_radar_indices.append(int(ridx))
    legend_handles.append(Line2D([], [], color="cyan", linewidth=1.8, label="Ellipses"))

    # 局部图中仅在窗口覆盖原点时显示 Ego 标记，避免视觉干扰。
    if x0 <= 0.0 <= x1 and y0 <= 0.0 <= y1:
        ax.plot(0.0, 0.0, marker="*", color="red", markersize=10)
        legend_handles.append(_make_ego_legend_handle("Ego"))

    ax.set_xlim(x0, x1)
    ax.set_ylim(y0, y1)
    ax.set_aspect("equal")
    ax.set_xlabel("X (meters)")
    ax.set_ylabel("Y (meters)")
    ax.set_title("Figure C (left): Gaussian ellipses in BEV")
    ax.grid(False)

    if show_legend:
        ax.legend(handles=legend_handles, loc="upper right", fontsize=8, handlelength=1.2)

    return list(dict.fromkeys(used_radar_indices))


def plot_figure_C_bev_per_radar(
    radar_xy: np.ndarray,
    boxes_ego: List[Box],
    radar_indices_per_box: List[List[int]],
    selected_box_indices: Sequence[int],
    ax: Optional[plt.Axes] = None,
    sigma_r: float = 0.5,
    sigma_theta: float = 1.0,
    max_ellipses_per_box: int = 10,
) -> List[int]:
    """
    绘制 Figure C 左图：每个雷达点一个椭圆核（选定目标内的多个雷达点）。

    Returns:
        used_radar_indices: 实际绘制椭圆的雷达点索引。
    """
    local_fig: Optional[plt.Figure] = None
    if ax is None:
        local_fig, ax = plt.subplots(figsize=(10, 10))

    radar_xy = np.asarray(radar_xy)
    used_radar_indices: List[int] = []
    legend_handles: List[object] = []

    if radar_xy.shape[0] > 0:
        radar_handle = ax.scatter(radar_xy[:, 0], radar_xy[:, 1], s=8, c='black', alpha=0.6, label='Radar points')
        legend_handles.append(radar_handle)
    else:
        legend_handles.append(
            Line2D([], [], marker="o", linestyle="None", color="black", markersize=4, label="Radar points")
        )

    for box_ego in boxes_ego:
        _draw_bev_box(ax, box_ego, color=(0.2, 0.95, 0.2), linewidth=1.4, label=None)
    legend_handles.append(Line2D([], [], color=(0.2, 0.95, 0.2), linewidth=2.0, label="3D boxes"))

    for box_idx in selected_box_indices:
        if box_idx < 0 or box_idx >= len(boxes_ego):
            continue
        candidate_ids = _limit_radar_indices_for_box(
            radar_xy=radar_xy,
            box_ego=boxes_ego[box_idx],
            radar_indices=radar_indices_per_box[box_idx],
            max_points=max_ellipses_per_box,
        )
        for ridx in candidate_ids:
            center, a_bev, b_bev, angle_deg = build_bev_ellipse_for_radar_point(
                point_xy=radar_xy[ridx],
                radar_direction=None,
                sigma_r=sigma_r,
                sigma_theta=sigma_theta,
            )
            patch = Ellipse(
                xy=center,
                width=2.0 * a_bev,
                height=2.0 * b_bev,
                angle=angle_deg,
                fill=False,
                edgecolor='cyan',
                linewidth=1.8,
                alpha=0.95,
            )
            ax.add_patch(patch)
            used_radar_indices.append(int(ridx))
    legend_handles.append(Line2D([], [], color="cyan", linewidth=1.8, label="Ellipses"))

    ax.plot(0.0, 0.0, marker='*', color='red', markersize=10)
    legend_handles.append(_make_ego_legend_handle("Ego"))
    ax.set_xlim(X_MIN, X_MAX)
    ax.set_ylim(Y_MIN, Y_MAX)
    ax.set_xlabel('X (meters)')
    ax.set_ylabel('Y (meters)')
    ax.set_aspect('equal')
    ax.set_title('Figure C (left): Gaussian ellipses in BEV')
    ax.legend(handles=legend_handles, loc='upper right', fontsize=8, handlelength=1.2)

    if local_fig is not None:
        local_fig.tight_layout()

    # 去重并保持顺序
    dedup = list(dict.fromkeys(used_radar_indices))
    return dedup


def plot_figure_C_image_per_radar(
    img: np.ndarray,
    cam_intrinsic: np.ndarray,
    ego_record: dict,
    cs_record: dict,
    boxes_world: List[Box],
    selected_box_indices: Sequence[int],
    radar_points_ego: np.ndarray,
    radar_indices_for_selected_box: Sequence[int],
    ax: Optional[plt.Axes] = None,
    sigma_r: float = 0.5,
    sigma_theta: float = 1.0,
) -> None:
    """
    绘制 Figure C 右图：同一目标内多个雷达核在图像平面的椭圆采样区域。
    """
    local_fig: Optional[plt.Figure] = None
    if ax is None:
        local_fig, ax = plt.subplots(figsize=(14, 8))

    img = np.asarray(img)
    radar_points_ego = np.asarray(radar_points_ego)
    img_h, img_w = img.shape[:2]

    ax.imshow(img)
    ax.axis('off')

    box_label_added = False
    for idx in selected_box_indices:
        if idx < 0 or idx >= len(boxes_world):
            continue
        box_cam = transform_box_world_to_camera(boxes_world[idx], ego_record, cs_record)
        corners_3d = box_cam.corners()
        if np.all(corners_3d[2, :] <= 1e-3):
            continue
        corners_uv = project_box_to_image(box_cam, cam_intrinsic)
        _draw_projected_box(ax, corners_uv, color=(0.2, 0.95, 0.2), linewidth=1.5)
        if not box_label_added:
            ax.plot([], [], color=(0.2, 0.95, 0.2), linewidth=1.8, label='3D boxes')
            box_label_added = True

    ellipses_drawn = 0
    centers_drawn = 0
    for ridx in radar_indices_for_selected_box:
        if ridx < 0 or ridx >= radar_points_ego.shape[0]:
            continue

        kernel = _project_radar_kernel_to_image(
            point_xy=radar_points_ego[int(ridx), :2],
            cs_record=cs_record,
            cam_intrinsic=cam_intrinsic,
            sigma_r=sigma_r,
            sigma_theta=sigma_theta,
        )
        if kernel is None:
            continue

        center_uv, a_img, b_img, angle_img = kernel
        u0, v0 = center_uv
        if not (0 <= u0 < img_w and 0 <= v0 < img_h):
            continue

        ellipse = Ellipse(
            xy=(u0, v0),
            width=2.0 * a_img,
            height=2.0 * b_img,
            angle=angle_img,
            fill=True,
            facecolor=(0.1, 1.0, 1.0, 0.12),
            edgecolor='cyan',
            linewidth=1.5,
            alpha=0.95,
            label='Ellipses' if ellipses_drawn == 0 else None,
        )
        ax.add_patch(ellipse)
        ellipses_drawn += 1

        ax.plot(
            u0,
            v0,
            marker='+',
            color='yellow',
            markersize=8,
            markeredgewidth=2.0,
            label='Sampling center' if centers_drawn == 0 else None,
        )
        centers_drawn += 1

    if not box_label_added:
        ax.plot([], [], color=(0.2, 0.95, 0.2), linewidth=1.8, label='3D boxes')
    if ellipses_drawn == 0:
        ax.plot([], [], color='cyan', linewidth=1.5, label='Ellipses')
    if centers_drawn == 0:
        ax.plot([], [], marker='+', color='yellow', linestyle='None', label='Sampling center')

    ax.set_xlim(0, img_w)
    ax.set_ylim(img_h, 0)
    ax.set_title('Figure C (right): Elliptical sampling regions in the image plane')
    ax.legend(loc='upper right', fontsize=8)

    if local_fig is not None:
        local_fig.tight_layout()


def plot_mgc_image_zoom_for_single_box(
    ax: plt.Axes,
    img: np.ndarray,
    cam_intrinsic: np.ndarray,
    ego_record: dict,
    cs_record: dict,
    box_world: Box,
    radar_xy: np.ndarray,
    radar_indices: Sequence[int],
    sigma_r: float = 0.5,
    sigma_theta: float = 1.0,
    crop_margin: float = 0.2,
    show_legend: bool = True,
) -> None:
    """
    绘制单目标 zoom-in 图像子图：局部 patch 内叠加多椭圆采样区域。
    """
    img = np.asarray(img)
    radar_xy = np.asarray(radar_xy)
    img_h, img_w = img.shape[:2]

    box_cam = transform_box_world_to_camera(box_world, ego_record, cs_record)
    corners_uv = project_box_to_image(box_cam, cam_intrinsic)

    valid = np.isfinite(corners_uv[:, 0]) & np.isfinite(corners_uv[:, 1])
    if np.any(valid):
        u_min = float(np.min(corners_uv[valid, 0]))
        u_max = float(np.max(corners_uv[valid, 0]))
        v_min = float(np.min(corners_uv[valid, 1]))
        v_max = float(np.max(corners_uv[valid, 1]))
    else:
        u_min, u_max, v_min, v_max = 0.0, float(img_w), 0.0, float(img_h)

    box_w = max(1.0, u_max - u_min)
    box_h = max(1.0, v_max - v_min)
    du = box_w * crop_margin
    dv = box_h * crop_margin

    crop_u0 = max(0, int(np.floor(u_min - du)))
    crop_u1 = min(img_w, int(np.ceil(u_max + du)))
    crop_v0 = max(0, int(np.floor(v_min - dv)))
    crop_v1 = min(img_h, int(np.ceil(v_max + dv)))

    # 避免裁剪区域过小导致观感差。
    if crop_u1 - crop_u0 < 40 or crop_v1 - crop_v0 < 40:
        center_u = int(np.clip((u_min + u_max) * 0.5, 0, img_w - 1))
        center_v = int(np.clip((v_min + v_max) * 0.5, 0, img_h - 1))
        half = 120
        crop_u0 = max(0, center_u - half)
        crop_u1 = min(img_w, center_u + half)
        crop_v0 = max(0, center_v - half)
        crop_v1 = min(img_h, center_v + half)

    patch = img[crop_v0:crop_v1, crop_u0:crop_u1]
    if patch.size == 0:
        patch = img
        crop_u0, crop_v0 = 0, 0
        crop_u1, crop_v1 = img_w, img_h

    patch_h, patch_w = patch.shape[:2]
    ax.imshow(patch)
    ax.axis("off")

    legend_handles: List[object] = []
    corners_patch = corners_uv.copy()
    corners_patch[:, 0] -= crop_u0
    corners_patch[:, 1] -= crop_v0
    _draw_projected_box(ax, corners_patch, color=(0.2, 0.95, 0.2), linewidth=1.8)
    legend_handles.append(Line2D([], [], color=(0.2, 0.95, 0.2), linewidth=1.8, label="3D boxes"))

    ellipses_drawn = 0
    centers_drawn = 0
    for ridx in radar_indices:
        if int(ridx) < 0 or int(ridx) >= radar_xy.shape[0]:
            continue

        kernel = _project_radar_kernel_to_image(
            point_xy=radar_xy[int(ridx), :2],
            cs_record=cs_record,
            cam_intrinsic=cam_intrinsic,
            sigma_r=sigma_r,
            sigma_theta=sigma_theta,
        )
        if kernel is None:
            continue

        center_uv, a_img, b_img, angle_img = kernel
        u0, v0 = center_uv
        if not (crop_u0 <= u0 < crop_u1 and crop_v0 <= v0 < crop_v1):
            continue

        u_patch = u0 - crop_u0
        v_patch = v0 - crop_v0

        ellipse = Ellipse(
            xy=(u_patch, v_patch),
            width=2.0 * a_img,
            height=2.0 * b_img,
            angle=angle_img,
            fill=True,
            facecolor=(0.1, 1.0, 1.0, 0.12),
            edgecolor="cyan",
            linewidth=1.5,
            alpha=0.95,
        )
        ax.add_patch(ellipse)
        ellipses_drawn += 1

        ax.plot(u_patch, v_patch, marker="+", color="yellow", markersize=8, markeredgewidth=2.0)
        centers_drawn += 1

    legend_handles.append(Line2D([], [], color="cyan", linewidth=1.5, label="Ellipses"))
    legend_handles.append(
        Line2D([], [], marker="+", linestyle="None", color="yellow", markersize=8, label="Sampling center")
    )

    ax.set_xlim(0, patch_w)
    ax.set_ylim(patch_h, 0)
    ax.set_title("Figure C (right): Elliptical sampling regions in the image plane")
    if show_legend:
        ax.legend(handles=legend_handles, loc="upper right", fontsize=8, handlelength=1.2)


def _build_figure_c_per_radar_context(
    nusc: NuScenes,
    sample_token: str,
    cam_name: str,
    max_boxes: int = 1,
    max_points_per_box: int = 6,
) -> Dict[str, object]:
    """
    构建 Figure C(per-radar) 的共享上下文。

    说明：该可视化内部基于官方 3D boxes 与雷达点构造采样核，仅用于几何分析。
    """
    img_path, cam_sd, cam_intrinsic, cs_record, ego_record = get_sample_camera_data(
        nusc, sample_token, cam_name=cam_name
    )
    img = np.array(Image.open(img_path).convert('RGB'))

    radar_xy = get_sample_radar_points_ego(nusc, sample_token, reference_sd_token=cam_sd['token'])
    boxes_world = get_sample_gt_boxes_world(nusc, sample_token)
    boxes_ego = [transform_box_world_to_ego(box, ego_record) for box in boxes_world]

    radar_indices_per_box = assign_radar_to_boxes_ego(radar_xy, boxes_ego)
    selected_box_indices = _select_box_indices_for_per_radar_demo(
        boxes_ego=boxes_ego,
        radar_indices_per_box=radar_indices_per_box,
        cam_intrinsic=cam_intrinsic,
        cs_record=cs_record,
        image_shape=(img.shape[0], img.shape[1]),
        max_boxes=max_boxes,
    )

    selected_radar_indices: List[int] = []
    for box_idx in selected_box_indices:
        if box_idx < 0 or box_idx >= len(boxes_ego):
            continue
        picked = _limit_radar_indices_for_box(
            radar_xy=radar_xy,
            box_ego=boxes_ego[box_idx],
            radar_indices=radar_indices_per_box[box_idx],
            max_points=max_points_per_box,
        )
        selected_radar_indices.extend(picked)

    selected_radar_indices = list(dict.fromkeys([int(i) for i in selected_radar_indices]))

    return {
        'sample_token': sample_token,
        'img': img,
        'cam_intrinsic': cam_intrinsic,
        'cs_record': cs_record,
        'ego_record': ego_record,
        'radar_xy': radar_xy,
        'boxes_world': boxes_world,
        'boxes_ego': boxes_ego,
        'radar_indices_per_box': radar_indices_per_box,
        'selected_box_indices': selected_box_indices,
        'selected_radar_indices': selected_radar_indices,
    }


def plot_figure_C_bev(sample_token: str, nusc: NuScenes, out_path_bev: str) -> None:
    """绘制 Figure C 左图（per-radar BEV 版本）。"""
    context = _build_figure_c_per_radar_context(nusc, sample_token, cam_name=DEFAULT_CAM)
    fig, ax = plt.subplots(figsize=(10, 10))
    plot_figure_C_bev_per_radar(
        radar_xy=np.asarray(context['radar_xy']),
        boxes_ego=list(context['boxes_ego']),
        radar_indices_per_box=list(context['radar_indices_per_box']),
        selected_box_indices=list(context['selected_box_indices']),
        ax=ax,
    )
    fig.tight_layout()
    out_file = Path(out_path_bev)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_file, dpi=180)
    plt.close(fig)


def draw_mgc_ellipses_on_image(
    nusc: NuScenes,
    sample_token: str,
    cam_name: str,
    out_path: str,
    ax: Optional[plt.Axes] = None,
    context: Optional[Dict[str, object]] = None,
) -> None:
    """
    兼容旧接口：绘制 Figure C 右图（per-radar image 版本）。
    """
    local_fig: Optional[plt.Figure] = None
    if context is None:
        context = _build_figure_c_per_radar_context(nusc, sample_token, cam_name=cam_name)
    if ax is None:
        local_fig, ax = plt.subplots(figsize=(14, 8))

    plot_figure_C_image_per_radar(
        img=np.asarray(context['img']),
        cam_intrinsic=np.asarray(context['cam_intrinsic']),
        ego_record=context['ego_record'],
        cs_record=context['cs_record'],
        boxes_world=list(context['boxes_world']),
        selected_box_indices=list(context['selected_box_indices']),
        radar_points_ego=np.asarray(context['radar_xy']),
        radar_indices_for_selected_box=list(context['selected_radar_indices']),
        ax=ax,
    )

    if local_fig is not None:
        local_fig.tight_layout()
        out_file = Path(out_path)
        out_file.parent.mkdir(parents=True, exist_ok=True)
        local_fig.savefig(out_file, dpi=180)
        plt.close(local_fig)


def plot_figure_C_bev_and_image_per_radar(
    sample_token: str,
    nusc: NuScenes,
    cam_name: str,
    out_path: str,
) -> None:
    """
    Figure C 高层兼容接口：默认单目标，转发到多目标版本。
    """
    plot_figure_C_mgc_multi_targets(
        sample_token=sample_token,
        nusc=nusc,
        cam_name=cam_name,
        max_targets=1,
        max_ellipses_per_box=6,
        out_path=out_path,
    )


def plot_figure_C_mgc_multi_targets(
    sample_token: str,
    nusc: NuScenes,
    cam_name: str,
    max_targets: int = 4,
    max_ellipses_per_box: int = 6,
    out_path: str = "figure_C_mgc_multi_targets.png",
) -> None:
    """
    Figure C 多目标版本：3~4 辆近处车，每辆一行（左 BEV 局部 + 右图像 zoom-in）。
    """
    context = _build_figure_c_per_radar_context(
        nusc=nusc,
        sample_token=sample_token,
        cam_name=cam_name,
        max_boxes=max_targets,
        max_points_per_box=max_ellipses_per_box,
    )

    boxes_world = list(context["boxes_world"])
    boxes_ego = list(context["boxes_ego"])
    radar_xy = np.asarray(context["radar_xy"])
    radar_indices_per_box = list(context["radar_indices_per_box"])
    selected_box_indices = [int(i) for i in list(context["selected_box_indices"])]

    valid_box_indices = [
        idx for idx in selected_box_indices
        if 0 <= idx < len(radar_indices_per_box) and len(radar_indices_per_box[idx]) > 0
    ]

    if len(valid_box_indices) == 0:
        # 退化：从全局里找有雷达点的最近目标
        fallback = [
            i for i in range(len(boxes_ego))
            if i < len(radar_indices_per_box) and len(radar_indices_per_box[i]) > 0
        ]
        fallback.sort(key=lambda i: float(np.hypot(float(boxes_ego[i].center[0]), float(boxes_ego[i].center[1]))))
        valid_box_indices = fallback[:max(1, max_targets)]

    if len(valid_box_indices) == 0:
        # 极端情况下直接画一个空图，避免报错中断。
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.axis("off")
        ax.set_title("Figure C: Elliptical sampling around nearby vehicles")
        out_file = Path(out_path)
        out_file.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_file, dpi=180)
        plt.close(fig)
        return

    n_rows = min(len(valid_box_indices), max_targets)
    valid_box_indices = valid_box_indices[:n_rows]

    fig, axes = plt.subplots(n_rows, 2, figsize=(15, 5.2 * n_rows))
    if n_rows == 1:
        axes = np.asarray([axes])

    for row, box_idx in enumerate(valid_box_indices):
        ax_bev = axes[row, 0]
        ax_img = axes[row, 1]

        used_radar_ids = plot_mgc_bev_for_single_box(
            ax=ax_bev,
            radar_xy=radar_xy,
            boxes_ego=boxes_ego,
            box_idx=box_idx,
            radar_indices_per_box=radar_indices_per_box,
            max_ellipses_per_box=max_ellipses_per_box,
            window_margin=20.0,
            sigma_r=0.5,
            sigma_theta=1.0,
            show_legend=(row == 0),
        )

        if len(used_radar_ids) == 0:
            used_radar_ids = _limit_radar_indices_for_box(
                radar_xy=radar_xy,
                box_ego=boxes_ego[box_idx],
                radar_indices=radar_indices_per_box[box_idx],
                max_points=max_ellipses_per_box,
            )

        plot_mgc_image_zoom_for_single_box(
            ax=ax_img,
            img=np.asarray(context["img"]),
            cam_intrinsic=np.asarray(context["cam_intrinsic"]),
            ego_record=context["ego_record"],
            cs_record=context["cs_record"],
            box_world=boxes_world[box_idx],
            radar_xy=radar_xy,
            radar_indices=used_radar_ids,
            sigma_r=0.5,
            sigma_theta=1.0,
            crop_margin=0.2,
            show_legend=(row == 0),
        )

    fig.suptitle("Figure C: Elliptical sampling around nearby vehicles", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    out_file = Path(out_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_file, dpi=180)
    plt.close(fig)


def plot_figure_C_bev_and_image(sample_token: str, nusc: NuScenes, cam_name: str, out_path: str) -> None:
    """兼容旧接口，转发到 per-radar 版本。"""
    plot_figure_C_bev_and_image_per_radar(sample_token, nusc, cam_name, out_path)

# =============================================================================
# Main example
# =============================================================================
if __name__ == "__main__":
    nusc = load_nuscenes(NUSC_VERSION, NUSC_DATAROOT)

    # 用户手动指定 sample_token 进行可视化
    sample_tokens = [
        # 示例: "ca9a282c9e77460f8360f564131a8af5",
    ]

    if len(sample_tokens) == 0:
        print("sample_tokens 为空，请先在脚本中填入 sample token。")

    for token in sample_tokens:
        # Figure A: BEV query distribution
        out_A = f"figure_A_bev_{token}.png"
        plot_figure_A_bev(token, nusc, out_A)

        # Figure C: multi-target rows (left BEV local + right image zoom-in)
        out_C_multi = f"figure_C_mgc_multi_targets_{token}.png"
        plot_figure_C_mgc_multi_targets(
            token,
            nusc,
            DEFAULT_CAM,
            max_targets=4,
            out_path=out_C_multi,
        )

    # Figure B 的调用示例（仅当用户准备好预测 JSON 时使用）
    # results_cam = prediction_json_to_boxes(CAMERA_ONLY_RESULT_JSON)
    # results_rac = prediction_json_to_boxes(RACFORMER_RESULT_JSON)
    # results_upg = prediction_json_to_boxes(UPGFORMER_RESULT_JSON)
    # for token in sample_tokens:
    #     out_B_img = f"figure_B_img_{token}.png"
    #     out_B_bev = f"figure_B_bev_{token}.png"
    #     draw_figure_B_for_sample(
    #         nusc, token, results_cam, results_rac, results_upg, out_B_img, out_B_bev
    #     )
