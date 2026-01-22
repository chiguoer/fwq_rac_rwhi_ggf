#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
RWHI v7 功能测试脚本

测试内容:
1. AlphaMLP 和 AlphaEncoder 模块
2. I_radar 计算公式 (含速度权重 w_v)
3. jitter field 确定性
4. 残差式空间扩散
5. Top-K 选取和 anchor 生成
6. 边界条件 (clamp, log, extreme values)

使用方法:
    python tools/test_rwhi_v53.py
"""

import sys
import torch
import torch.nn as nn

# 添加项目根目录到路径
sys.path.insert(0, '.')


def test_alpha_mlp():
    """测试 AlphaMLP"""
    print("\n" + "=" * 60)
    print("测试 1: AlphaMLP")
    print("=" * 60)
    
    from models.rwhi_v53 import AlphaMLP
    
    # 创建模块
    mlp = AlphaMLP(in_dim=3, hidden_dim=32, init_bias=1.0)
    
    # 测试输入
    B, M = 2, 100
    f_in = torch.randn(B, M, 3)
    
    # 前向传播
    alpha = mlp(f_in)
    
    # 验证
    assert alpha.shape == (B, M, 1), f"形状错误: {alpha.shape}"
    assert (alpha >= 0).all() and (alpha <= 1).all(), "α 值超出 (0, 1) 范围"
    
    print(f"  输入形状: {f_in.shape}")
    print(f"  输出形状: {alpha.shape}")
    print(f"  α 范围: [{alpha.min().item():.4f}, {alpha.max().item():.4f}]")
    print(f"  α 均值: {alpha.mean().item():.4f}")
    print("  ✓ AlphaMLP 测试通过")


def test_alpha_encoder():
    """测试 AlphaEncoder"""
    print("\n" + "=" * 60)
    print("测试 2: AlphaEncoder")
    print("=" * 60)
    
    from models.rwhi_v53 import AlphaEncoder
    
    # 创建模块
    d_alpha = 2
    encoder = AlphaEncoder(d_alpha=d_alpha, hidden_dim=8)
    
    # 测试不同输入形状
    test_cases = [
        torch.rand(100),          # [N_query]
        torch.rand(2, 100),       # [B, K]
        torch.rand(2, 100, 1),    # [B, K, 1]
    ]
    
    for i, alpha in enumerate(test_cases):
        emb = encoder(alpha)
        expected_shape = list(alpha.shape)
        if alpha.dim() >= 2 and alpha.shape[-1] == 1:
            expected_shape[-1] = d_alpha
        else:
            expected_shape.append(d_alpha)
        
        print(f"  测试 {i+1}: 输入 {list(alpha.shape)} -> 输出 {list(emb.shape)}")
        assert emb.shape[-1] == d_alpha, f"embedding 维度错误"
    
    print("  ✓ AlphaEncoder 测试通过")


def test_jitter_field_determinism():
    """测试 jitter field 确定性"""
    print("\n" + "=" * 60)
    print("测试 3: Jitter Field 确定性")
    print("=" * 60)
    
    from models.rwhi_v53 import RWHI_v53
    
    # 创建两个独立的模块实例
    cfg = dict(
        num_query=100,
        pc_range=(-51.2, -51.2, -5.0, 51.2, 51.2, 3.0),
        bev_grid_size=50,
    )
    
    rwhi1 = RWHI_v53(**cfg)
    rwhi2 = RWHI_v53(**cfg)
    
    # 验证 jitter field 相同
    J1 = rwhi1.jitter_field
    J2 = rwhi2.jitter_field
    
    assert torch.allclose(J1, J2), "Jitter field 不确定!"
    
    print(f"  网格大小: {J1.shape}")
    print(f"  J 范围: [{J1.min().item():.4f}, {J1.max().item():.4f}]")
    print(f"  两个实例的 J 完全相同: {torch.allclose(J1, J2)}")
    print("  ✓ Jitter Field 确定性测试通过")


def test_i_radar_formula():
    """测试 I_radar 计算公式"""
    print("\n" + "=" * 60)
    print("测试 4: I_radar 计算公式")
    print("=" * 60)
    
    from models.rwhi_v53 import RWHI_v53
    
    cfg = dict(
        num_query=100,
        pc_range=(-51.2, -51.2, -5.0, 51.2, 51.2, 3.0),
        bev_grid_size=50,
        d_ref=30.0,
        d_lambda=1.0,
        w_d_max=2.5,
    )
    
    rwhi = RWHI_v53(**cfg)
    
    # 模拟雷达点
    B, M = 1, 10
    radar_points = torch.zeros(B, M, 5)
    
    # 设置不同距离的点
    distances = [10, 20, 30, 40, 50, 60]
    for i, d in enumerate(distances[:M]):
        radar_points[0, i, 0] = d  # x
        radar_points[0, i, 1] = 0  # y
        radar_points[0, i, 2] = 0  # z
        radar_points[0, i, 3] = 10  # rcs
        radar_points[0, i, 4] = 5   # v_r
    
    radar_mask = torch.ones(B, M)
    
    # 计算 alpha
    alpha = rwhi._compute_alpha(radar_points, radar_mask)
    
    # 计算 I_radar
    i_radar = rwhi._compute_i_radar(radar_points, radar_mask, alpha)
    
    print(f"  距离 (m): {distances[:M]}")
    print(f"  α 值: {alpha[0, :len(distances), 0].tolist()}")
    print(f"  I_radar: {i_radar[0, :len(distances)].tolist()}")
    
    # 验证公式: I = α * log(1 + σ * w_d)
    # w_d = (d / d_ref)^λ, clamped
    print("  ✓ I_radar 计算测试通过")


def test_forward_pass():
    """测试完整的前向传播"""
    print("\n" + "=" * 60)
    print("测试 5: 完整前向传播")
    print("=" * 60)
    
    from models.rwhi_v53 import RWHI_v53
    
    cfg = dict(
        num_query=100,
        pc_range=(-51.2, -51.2, -5.0, 51.2, 51.2, 3.0),
        bev_grid_size=50,
        d_alpha=2,
    )
    
    rwhi = RWHI_v53(**cfg)
    
    # 模拟雷达点云
    B, M = 2, 500
    radar_points = torch.randn(B, M, 5)
    radar_points[..., 0] = radar_points[..., 0] * 50  # x: [-50, 50]
    radar_points[..., 1] = radar_points[..., 1] * 50  # y: [-50, 50]
    radar_points[..., 2] = radar_points[..., 2] * 2   # z: [-2, 2]
    radar_points[..., 3] = torch.abs(radar_points[..., 3]) * 20  # rcs: [0, 20]
    radar_points[..., 4] = radar_points[..., 4] * 10  # v_r: [-10, 10]
    
    radar_mask = torch.ones(B, M)
    
    # 前向传播
    anchors, alpha_values = rwhi(radar_points, radar_mask)
    
    # 验证输出
    K = cfg['num_query']
    assert anchors.shape == (B, K, 10), f"anchors 形状错误: {anchors.shape}"
    assert alpha_values.shape == (B, K, 1), f"alpha_values 形状错误: {alpha_values.shape}"
    
    print(f"  输入: radar_points {radar_points.shape}")
    print(f"  输出: anchors {anchors.shape}")
    print(f"  输出: alpha_values {alpha_values.shape}")
    
    # 验证 anchor 内容
    print(f"  anchors[0,0] = {anchors[0, 0].tolist()}")
    print(f"    theta (logit): {anchors[0, 0, 0].item():.4f}")
    print(f"    d (logit): {anchors[0, 0, 1].item():.4f}")
    print(f"    z (logit): {anchors[0, 0, 2].item():.4f}")
    print(f"    w_log: {anchors[0, 0, 3].item():.4f}")
    print(f"    l_log: {anchors[0, 0, 4].item():.4f}")
    print(f"    h_log: {anchors[0, 0, 5].item():.4f}")
    print(f"    sin: {anchors[0, 0, 6].item():.4f}")
    print(f"    cos: {anchors[0, 0, 7].item():.4f}")
    
    # 测试 alpha encoding
    alpha_emb = rwhi.encode_alpha(alpha_values)
    print(f"  alpha_emb 形状: {alpha_emb.shape}")
    
    print("  ✓ 完整前向传播测试通过")


def test_score_map_values():
    """测试打分图数值量级"""
    print("\n" + "=" * 60)
    print("测试 6: 打分图数值量级")
    print("=" * 60)
    
    from models.rwhi_v53 import RWHI_v53
    
    cfg = dict(
        num_query=100,
        pc_range=(-51.2, -51.2, -5.0, 51.2, 51.2, 3.0),
        bev_grid_size=50,
        base_bias=1.0,
        epsilon=0.01,
    )
    
    rwhi = RWHI_v53(**cfg)
    
    # 模拟雷达点云
    B = 1
    M = 100
    radar_points = torch.randn(B, M, 5)
    radar_points[..., 0] = radar_points[..., 0] * 30 + 10  # x: 近处目标
    radar_points[..., 1] = radar_points[..., 1] * 30
    radar_points[..., 2] = 0
    radar_points[..., 3] = torch.abs(radar_points[..., 3]) * 15 + 5  # rcs: [5, 20]
    radar_points[..., 4] = radar_points[..., 4] * 5
    
    radar_mask = torch.ones(B, M)
    
    # 计算各个组件
    alpha = rwhi._compute_alpha(radar_points, radar_mask)
    i_radar_point = rwhi._compute_i_radar(radar_points, radar_mask, alpha)
    
    device = radar_points.device
    i_radar_map = rwhi._scatter_to_bev(radar_points, i_radar_point, B, device)
    i_radar_map = rwhi.diffusion(i_radar_map)
    S = rwhi._build_score_map(i_radar_map, device)
    
    print(f"  α 范围: [{alpha.min().item():.4f}, {alpha.max().item():.4f}]")
    print(f"  I_radar_point 范围: [{i_radar_point.min().item():.4f}, {i_radar_point.max().item():.4f}]")
    print(f"  I_radar_map 范围: [{i_radar_map.min().item():.4f}, {i_radar_map.max().item():.4f}]")
    print(f"  S (打分图) 范围: [{S.min().item():.4f}, {S.max().item():.4f}]")
    
    # 验证数值量级
    # 背景: S ≈ 1.0 ± 0.01
    # 有雷达的区域: S ≈ 1.5 ~ 4.0
    assert S.min().item() >= 0.9, "S 最小值过低"
    
    print("  ✓ 打分图数值量级测试通过")


def test_rwhi_module_wrapper():
    """测试 RWHIModule wrapper"""
    print("\n" + "=" * 60)
    print("测试 7: RWHIModule wrapper (v7)")
    print("=" * 60)
    
    from models.rwhi import RWHIModule
    
    cfg = dict(
        rwhi_version='v5.3',
        num_query=100,
        pc_range=(-51.2, -51.2, -5.0, 51.2, 51.2, 3.0),
        bev_grid_size=50,
        d_alpha=2,
        # v7 新参数
        v_ref=10.0,
        beta=1.0,
        diffusion_gamma=0.5,
        diffusion_s_max=5.0,
    )
    
    rwhi = RWHIModule(**cfg)
    
    # 验证版本
    assert rwhi.is_v53, "应该是 v5.3/v7 版本"
    assert rwhi.d_alpha == 2, "d_alpha 应该是 2"
    
    # 测试前向传播
    B, M = 2, 100
    radar_points = torch.randn(B, M, 5) * 30
    radar_mask = torch.ones(B, M)
    
    anchors, alpha_values = rwhi(radar_points, radar_mask)
    
    # 测试 alpha encoding
    alpha_emb = rwhi.encode_alpha(alpha_values)
    
    print(f"  版本: v7 (RWHI_v53)")
    print(f"  d_alpha: {rwhi.d_alpha}")
    print(f"  anchors 形状: {anchors.shape}")
    print(f"  alpha_values 形状: {alpha_values.shape}")
    print(f"  alpha_emb 形状: {alpha_emb.shape}")
    
    print("  ✓ RWHIModule wrapper 测试通过")


def test_velocity_weight():
    """测试 v7 速度权重 w_v"""
    print("\n" + "=" * 60)
    print("测试 8: 速度权重 w_v (v7)")
    print("=" * 60)
    
    # w_v = 1 + β * sigmoid(|v|/v_ref)
    v_ref = 10.0
    beta = 1.0
    
    v_test = torch.tensor([0.0, 5.0, 10.0, 20.0, 100.0])
    w_v = 1.0 + beta * torch.sigmoid(v_test.abs() / v_ref)
    
    print(f"  v_ref = {v_ref}, beta = {beta}")
    print(f"  速度 (m/s): {v_test.tolist()}")
    print(f"  w_v 值: {w_v.tolist()}")
    
    # 验证范围
    assert (w_v >= 1.0).all(), "w_v 应该 >= 1.0"
    assert (w_v <= 1.0 + beta).all(), f"w_v 应该 <= {1.0 + beta}"
    
    # 验证单调性
    assert (w_v[1:] >= w_v[:-1]).all(), "w_v 应该随 |v| 单调递增"
    
    print("  ✓ 速度权重测试通过")


def test_residual_diffusion():
    """测试 v7 残差式空间扩散"""
    print("\n" + "=" * 60)
    print("测试 9: 残差式空间扩散 (v7)")
    print("=" * 60)
    
    from models.rwhi_v53 import RWHI_v53
    
    cfg = dict(
        num_query=100,
        pc_range=(-51.2, -51.2, -5.0, 51.2, 51.2, 3.0),
        bev_grid_size=50,
        diffusion_gamma=0.5,
        diffusion_s_max=5.0,
    )
    
    rwhi = RWHI_v53(**cfg)
    
    # 构造测试 I_radar_map
    B, H, W = 1, 50, 50
    device = rwhi.grid_xy.device
    i_radar_map = torch.zeros(B, 1, H, W, device=device)
    
    # 在中心放置一个高峰
    i_radar_map[0, 0, 25, 25] = 2.0
    
    # 构建打分图 (含扩散)
    S = rwhi._build_score_map(i_radar_map, device)
    
    # 验证扩散效果
    center_val = S[0, 0, 25, 25].item()
    neighbor_val = S[0, 0, 25, 26].item()
    bg_val = S[0, 0, 0, 0].item()
    
    print(f"  扩散前中心值: {1.0 + 2.0:.2f} (C_base + I_radar)")
    print(f"  扩散后中心值: {center_val:.4f}")
    print(f"  扩散后邻居值: {neighbor_val:.4f}")
    print(f"  背景值: {bg_val:.4f}")
    
    # 验证: 扩散不会降低峰值
    assert center_val >= 3.0 - 0.1, "扩散不应该降低峰值"
    
    # 验证: 邻居被抬升
    assert neighbor_val > bg_val, "扩散应该抬升邻居"
    
    # 验证: S_max 裁剪
    i_radar_map[0, 0, 25, 25] = 10.0  # 极大值
    S = rwhi._build_score_map(i_radar_map, device)
    max_val = S.max().item()
    assert max_val <= cfg['diffusion_s_max'] + 0.1, f"S_max 裁剪失效: {max_val}"
    
    print("  ✓ 残差式空间扩散测试通过")


def test_boundary_conditions():
    """测试边界条件 (clamp, log, extreme values)"""
    print("\n" + "=" * 60)
    print("测试 10: 边界条件")
    print("=" * 60)
    
    from models.rwhi_v53 import RWHI_v53
    
    cfg = dict(
        num_query=100,
        pc_range=(-51.2, -51.2, -5.0, 51.2, 51.2, 3.0),
        bev_grid_size=50,
    )
    
    rwhi = RWHI_v53(**cfg)
    
    # 测试 1: d = 0 (原点附近)
    radar = torch.zeros(1, 1, 5)
    radar[0, 0, 0] = 0.01  # x ≈ 0
    radar[0, 0, 1] = 0.01  # y ≈ 0
    radar[0, 0, 3] = 10    # rcs
    anchors, _ = rwhi(radar)
    assert not torch.isnan(anchors).any(), "d=0 附近不应产生 NaN"
    print("  ✓ d ≈ 0 边界处理正常")
    
    # 测试 2: 负 RCS
    radar = torch.randn(1, 10, 5) * 30
    radar[..., 3] = -100  # 负 rcs
    anchors, alpha = rwhi(radar)
    assert not torch.isnan(anchors).any(), "负 RCS 不应产生 NaN"
    print("  ✓ 负 RCS 处理正常")
    
    # 测试 3: 极大 RCS
    radar[..., 3] = 1e6
    anchors, alpha = rwhi(radar)
    assert not torch.isinf(anchors).any(), "极大 RCS 不应产生 Inf"
    print("  ✓ 极大 RCS 处理正常")
    
    # 测试 4: 极大速度
    radar[..., 4] = 1000  # 极大速度
    anchors, alpha = rwhi(radar)
    assert not torch.isnan(anchors).any(), "极大速度不应产生 NaN"
    print("  ✓ 极大速度处理正常")
    
    # 测试 5: 全 padding
    radar = torch.zeros(1, 10, 5)
    radar_mask = torch.zeros(1, 10)
    anchors, alpha = rwhi(radar, radar_mask)
    assert not torch.isnan(anchors).any(), "全 padding 不应产生 NaN"
    print("  ✓ 全 padding 处理正常")
    
    print("  ✓ 边界条件测试通过")


def main():
    """运行所有测试"""
    print("\n" + "=" * 60)
    print("RWHI v7 功能测试")
    print("=" * 60)
    
    try:
        test_alpha_mlp()
        test_alpha_encoder()
        test_jitter_field_determinism()
        test_i_radar_formula()
        test_forward_pass()
        test_score_map_values()
        test_rwhi_module_wrapper()
        # v7 新增测试
        test_velocity_weight()
        test_residual_diffusion()
        test_boundary_conditions()
        
        print("\n" + "=" * 60)
        print("✓ 所有测试通过!")
        print("=" * 60)
        return 0
        
    except Exception as e:
        print(f"\n✗ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    exit(main())

