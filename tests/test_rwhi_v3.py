# -*- coding: utf-8 -*-
import os
import sys

import torch

ROOT = os.path.dirname(os.path.dirname(__file__))
if ROOT not in sys.path:
    sys.path.append(ROOT)

from models.rwhi import RWHIModule
from models.racformer_head import RaCFormerHead


def _build_mask(B, M, valid_count, device):
    mask = torch.zeros(B, M, device=device, dtype=torch.bool)
    mask[:, :valid_count] = True
    return mask


def test_v3_output_shape_and_finite():
    device = torch.device('cpu')
    B, M, K = 2, 64, 32
    torch.manual_seed(0)

    rwhi = RWHIModule(
        rwhi_version='v3',
        num_query=K,
        pc_range=(-51.2, -51.2, -5.0, 51.2, 51.2, 3.0),
        bev_grid_size=32,
        d_ref=30.0,
        beta=0.2,
        base_bias=1.0,
        z_default=0.5,
        max_points=M,
        radar_channel_map=dict(x=0, y=1, z=2, rcs=3, v_r=4),
        enabled=True,
    ).to(device)
    rwhi.eval()

    radar_points = torch.randn(B, M, 5, device=device)
    mask = _build_mask(B, M, M // 2, device)
    radar_points = radar_points * mask.unsqueeze(-1)

    with torch.no_grad():
        anchors, _ = rwhi(radar_points, radar_mask=mask)

    assert anchors.shape == (B, K, 10)
    assert torch.isfinite(anchors).all()


def test_legacy_and_head_compat():
    device = torch.device('cpu')
    B, M, K = 2, 64, 40
    torch.manual_seed(1)

    legacy = RWHIModule(
        rwhi_version='legacy',
        num_query=K,
        num_safety=K // 2,
        num_saliency=K - K // 2,
        bev_grid_size=32,
        enabled=True,
    ).to(device)
    legacy.eval()

    radar_points = torch.randn(B, M, 5, device=device)
    anchors, anchor_mask = legacy(radar_points)

    assert anchors.shape == (B, K, 10)
    assert anchor_mask.shape == (B, K)

    expanded = RaCFormerHead._expand_query_bbox(anchors)
    head = RaCFormerHead.__new__(RaCFormerHead)
    head.training = False
    validated = head._validate_query_bbox(expanded)

    assert validated.shape == (B, K, 10)
    assert torch.isfinite(validated[..., :3]).all()


def test_numerical_stability_large_distance():
    device = torch.device('cpu')
    B, M, K = 1, 32, 16

    rwhi = RWHIModule(
        rwhi_version='v3',
        num_query=K,
        pc_range=(-200.0, -200.0, -5.0, 200.0, 200.0, 3.0),
        bev_grid_size=32,
        d_ref=30.0,
        beta=0.0,
        base_bias=1.0,
        z_default=0.5,
        max_points=M,
        radar_channel_map=dict(x=0, y=1, z=2, rcs=3, v_r=4),
        enabled=True,
    ).to(device)
    rwhi.eval()

    radar_points = torch.zeros(B, M, 5, device=device)
    radar_points[..., 0] = 1e4
    radar_points[..., 1] = -1e4
    radar_points[..., 3] = 1e3
    radar_points[..., 4] = 50.0
    mask = torch.ones(B, M, device=device, dtype=torch.bool)

    with torch.no_grad():
        anchors, _ = rwhi(radar_points, radar_mask=mask)

    assert anchors.shape == (B, K, 10)
    assert torch.isfinite(anchors).all()


def test_scatter_add_path_profile():
    device = torch.device('cpu')
    B, M, K = 1, 32, 16

    rwhi = RWHIModule(
        rwhi_version='v3',
        num_query=K,
        pc_range=(-51.2, -51.2, -5.0, 51.2, 51.2, 3.0),
        bev_grid_size=16,
        d_ref=30.0,
        beta=0.0,
        base_bias=1.0,
        z_default=0.5,
        max_points=M,
        radar_channel_map=dict(x=0, y=1, z=2, rcs=3, v_r=4),
        enabled=True,
    ).to(device)
    rwhi.eval()

    radar_points = torch.randn(B, M, 5, device=device)
    mask = torch.ones(B, M, device=device, dtype=torch.bool)

    with torch.autograd.profiler.profile(use_cuda=False) as prof:
        with torch.no_grad():
            rwhi(radar_points, radar_mask=mask)

    names = [evt.name for evt in prof.function_events]
    assert any('scatter_add' in name for name in names)
