# -*- coding: utf-8 -*-
import os
import sys
import torch

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from models.rwhi import RWHIModule


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    B = 2
    M = 128
    K = 64
    torch.manual_seed(0)

    rwhi = RWHIModule(
        rwhi_version='v3',
        num_query=K,
        pc_range=(-51.2, -51.2, -5.0, 51.2, 51.2, 3.0),
        bev_grid_size=64,
        d_ref=30.0,
        beta=0.2,
        base_bias=1.0,
        z_default=0.5,
        max_points=M,
        radar_channel_map=dict(x=0, y=1, z=2, rcs=3, v_r=4),
        enabled=True,
    ).to(device)
    rwhi.eval()

    valid_count = M // 2
    mask = (torch.arange(M, device=device).unsqueeze(0) < valid_count).expand(B, M)

    radar_points = torch.zeros(B, M, 5, device=device)
    radar_points[:, :valid_count] = torch.randn(B, valid_count, 5, device=device)

    radar_points_clean = radar_points.clone()
    radar_points_dirty = radar_points.clone()
    radar_points_dirty[:, valid_count:] = 1e4

    with torch.no_grad():
        anchors_clean, _ = rwhi(radar_points_clean, radar_mask=mask)
        anchors_dirty, _ = rwhi(radar_points_dirty, radar_mask=mask)

    assert anchors_clean.shape == (B, K, 3), f'anchors shape mismatch: {anchors_clean.shape}'
    assert anchors_dirty.shape == (B, K, 3), f'anchors shape mismatch: {anchors_dirty.shape}'
    assert torch.isfinite(anchors_clean).all(), 'anchors_clean contains NaN/Inf'
    assert torch.isfinite(anchors_dirty).all(), 'anchors_dirty contains NaN/Inf'

    diff = (anchors_clean - anchors_dirty).abs().max().item()
    assert torch.allclose(anchors_clean, anchors_dirty, atol=1e-5, rtol=1e-4), (
        f'mask not effective, max diff={diff:.6f}'
    )

    print('smoke_test_rwhi: OK', anchors_clean.shape, anchors_clean.dtype, anchors_clean.device)


if __name__ == '__main__':
    main()
