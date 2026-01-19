import os
import sys
import types
import torch

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, REPO_ROOT)

# Avoid importing models/__init__.py to keep test lightweight.
models_pkg = types.ModuleType("models")
models_pkg.__path__ = [os.path.join(REPO_ROOT, "models")]
sys.modules["models"] = models_pkg

bbox_pkg = types.ModuleType("models.bbox")
bbox_pkg.__path__ = [os.path.join(REPO_ROOT, "models", "bbox")]
sys.modules["models.bbox"] = bbox_pkg

import models.bbox.match_costs  # noqa: F401
import models.bbox.assigners  # noqa: F401
from models.bbox.utils import normalize_bbox
from mmdet.core.bbox.match_costs import build_match_cost
from mmdet.core.bbox.builder import build_assigner


def make_dummy_boxes(num, pc_range, device):
    x = torch.empty(num, device=device).uniform_(pc_range[0], pc_range[3])
    y = torch.empty(num, device=device).uniform_(pc_range[1], pc_range[4])
    z = torch.zeros(num, device=device)
    w = torch.full((num,), 1.6, device=device)
    l = torch.full((num,), 3.9, device=device)
    h = torch.full((num,), 1.5, device=device)
    rot = torch.zeros(num, device=device)
    vx = torch.zeros(num, device=device)
    vy = torch.zeros(num, device=device)
    return torch.stack([x, y, z, w, l, h, rot, vx, vy], dim=-1)


def main():
    device = torch.device("cpu")
    pc_range = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]

    theta_cost_cfg = dict(type="ThetaL1Cost", weight=3.0, pc_range=pc_range)
    reg_cost_cfg = dict(type="BBox3DL1Cost", weight=0.25)
    iou_cost_cfg = dict(type="IoUCost", weight=0.0)
    cls_cost_cfg = dict(type="FocalLossCost", weight=2.0)

    theta_cost = build_match_cost(theta_cost_cfg)
    reg_cost = build_match_cost(reg_cost_cfg)
    _ = build_match_cost(iou_cost_cfg)
    print("[build] match_cost ok")

    num_query, num_gt, num_classes = 5, 3, 3
    bbox_pred = make_dummy_boxes(num_query, pc_range, device)
    gt_bboxes = make_dummy_boxes(num_gt, pc_range, device)
    normalized_gt = normalize_bbox(gt_bboxes)

    theta_matrix = theta_cost(bbox_pred.clone(), normalized_gt.clone())
    print(f"[theta_cost] shape={tuple(theta_matrix.shape)}")
    assert theta_matrix.shape == (num_query, num_gt)

    reg_matrix = reg_cost(bbox_pred[:, :8], normalized_gt[:, :8])
    print(f"[reg_cost] shape={tuple(reg_matrix.shape)}")

    assigner_cfg = dict(
        type="PolarHungarianAssigner3D",
        cls_cost=cls_cost_cfg,
        reg_cost=reg_cost_cfg,
        theta_cost=theta_cost_cfg,
        iou_cost=iou_cost_cfg,
    )
    assigner = build_assigner(assigner_cfg)
    print(f"[build] assigner ok: {type(assigner).__name__}")

    cls_pred = torch.randn(num_query, num_classes, device=device)
    gt_labels = torch.randint(0, num_classes, (num_gt,), device=device)
    assign_result = assigner.assign(bbox_pred, cls_pred, gt_bboxes, gt_labels)
    print(f"[assign] num_gts={assign_result.num_gts}")

    print("[done] min assigner/cost test completed")


if __name__ == "__main__":
    main()
