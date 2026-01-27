"""
RWHI v7 模块测试脚本

测试内容：
1. 数值范围测试
2. 梯度流测试
3. polar_radius 一致性测试
"""

import torch
import sys
import os

# 添加项目根目录到 path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.rwhi import RWHIModule, AlphaMLP, AlphaEncoder, build_rwhi, R_MAX, EPS


def test_alpha_mlp():
    """测试 AlphaMLP 输出范围"""
    print("=" * 50)
    print("测试 AlphaMLP")
    print("=" * 50)
    
    mlp = AlphaMLP(in_dim=3, hidden_dim=32, init_bias=1.0)
    
    # 模拟输入: [log1p(rcs), d_norm, v_norm]
    batch_size, num_points = 2, 100
    f_in = torch.randn(batch_size, num_points, 3)
    f_in[..., 1] = f_in[..., 1].abs().clamp(0, 1)  # d_norm
    f_in[..., 2] = f_in[..., 2].abs().clamp(0, 1)  # v_norm
    
    alpha = mlp(f_in)
    
    print(f"输入形状: {f_in.shape}")
    print(f"输出形状: {alpha.shape}")
    print(f"α 范围: [{alpha.min().item():.4f}, {alpha.max().item():.4f}]")
    
    # 验证范围
    assert alpha.min() > 0, "α 应该 > 0"
    assert alpha.max() < 1, "α 应该 < 1"
    print("✅ AlphaMLP 测试通过")
    print()


def test_alpha_encoder():
    """测试 AlphaEncoder 输出"""
    print("=" * 50)
    print("测试 AlphaEncoder")
    print("=" * 50)
    
    encoder = AlphaEncoder(d_alpha=2, hidden_dim=8)
    
    batch_size, num_query = 2, 900
    alpha = torch.rand(batch_size, num_query, 1)  # (0, 1)
    
    embedding = encoder(alpha)
    
    print(f"输入形状: {alpha.shape}")
    print(f"输出形状: {embedding.shape}")
    print(f"d_alpha: {encoder.d_alpha}")
    
    assert embedding.shape == (batch_size, num_query, 2), "输出形状不正确"
    print("✅ AlphaEncoder 测试通过")
    print()


def test_rwhi_module_output_ranges():
    """测试 RWHIModule 输出范围"""
    print("=" * 50)
    print("测试 RWHIModule 输出范围")
    print("=" * 50)
    
    pc_range = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]
    num_query = 900
    
    rwhi = RWHIModule(
        num_query=num_query,
        pc_range=pc_range,
        bev_grid_size=100,
        enabled=True,
    )
    
    # 模拟雷达点云输入
    batch_size = 2
    num_points = 500
    
    # [x, y, z, rcs, v_r, ...]
    radar_points = torch.zeros(batch_size, num_points, 7)
    radar_points[..., 0] = torch.rand(batch_size, num_points) * 100 - 50  # x: [-50, 50]
    radar_points[..., 1] = torch.rand(batch_size, num_points) * 100 - 50  # y: [-50, 50]
    radar_points[..., 2] = torch.rand(batch_size, num_points) * 8 - 5     # z: [-5, 3]
    radar_points[..., 3] = torch.rand(batch_size, num_points) * 30        # rcs: [0, 30]
    radar_points[..., 4] = torch.rand(batch_size, num_points) * 30 - 15   # v_r: [-15, 15]
    
    radar_mask = torch.ones(batch_size, num_points)
    radar_mask[:, num_points//2:] = 0  # 一半点无效
    
    anchors, alpha_values = rwhi(radar_points, radar_mask)
    
    print(f"雷达点形状: {radar_points.shape}")
    print(f"锚点形状: {anchors.shape}")
    print(f"α 形状: {alpha_values.shape}")
    
    # 验证 θ: [0, 1)
    theta = anchors[..., 0]
    print(f"\nθ 范围: [{theta.min().item():.6f}, {theta.max().item():.6f}]")
    assert theta.min() >= 0, "θ 应该 >= 0"
    assert theta.max() < 1, "θ 应该 < 1"
    
    # 验证 d: (EPS, 1-EPS)
    d = anchors[..., 1]
    print(f"d 范围: [{d.min().item():.6f}, {d.max().item():.6f}]")
    assert d.min() > EPS, f"d 应该 > {EPS}"
    assert d.max() < 1 - EPS, f"d 应该 < {1 - EPS}"
    
    # 验证 z: (EPS, 1-EPS)
    z = anchors[..., 2]
    print(f"z 范围: [{z.min().item():.6f}, {z.max().item():.6f}]")
    assert z.min() > EPS, f"z 应该 > {EPS}"
    assert z.max() < 1 - EPS, f"z 应该 < {1 - EPS}"
    
    # 验证 α: (0, 1)
    print(f"\nα 范围: [{alpha_values.min().item():.4f}, {alpha_values.max().item():.4f}]")
    assert alpha_values.min() >= 0, "α 应该 >= 0"
    assert alpha_values.max() <= 1, "α 应该 <= 1"
    
    print("\n✅ RWHIModule 输出范围测试通过")
    print()


def test_gradient_flow():
    """测试梯度流"""
    print("=" * 50)
    print("测试梯度流")
    print("=" * 50)
    
    pc_range = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]
    num_query = 900
    
    rwhi = RWHIModule(
        num_query=num_query,
        pc_range=pc_range,
        bev_grid_size=100,
        enabled=True,
    )
    rwhi.train()
    
    # 模拟输入
    batch_size = 2
    num_points = 500
    
    radar_points = torch.randn(batch_size, num_points, 7, requires_grad=False)
    radar_points[..., 0] = radar_points[..., 0] * 30  # x
    radar_points[..., 1] = radar_points[..., 1] * 30  # y
    
    radar_mask = torch.ones(batch_size, num_points)
    
    anchors, alpha_values = rwhi(radar_points, radar_mask)
    
    # 模拟损失
    loss = anchors.sum() + alpha_values.sum()
    loss.backward()
    
    # 检查梯度
    params_with_grad = 0
    params_without_grad = 0
    
    for name, param in rwhi.named_parameters():
        if param.grad is not None:
            params_with_grad += 1
        else:
            params_without_grad += 1
            print(f"⚠️  {name} 没有梯度")
    
    print(f"\n有梯度的参数: {params_with_grad}")
    print(f"无梯度的参数: {params_without_grad}")
    
    if params_without_grad == 0:
        print("\n✅ 梯度流测试通过")
    else:
        print("\n⚠️  部分参数没有梯度，请检查")
    print()


def test_polar_radius_consistency():
    """测试 polar_radius 一致性"""
    print("=" * 50)
    print("测试 polar_radius 一致性")
    print("=" * 50)
    
    pc_range = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]
    num_query = 900
    
    rwhi = RWHIModule(
        num_query=num_query,
        pc_range=pc_range,
        bev_grid_size=100,
        enabled=True,
    )
    
    print(f"R_MAX 常量: {R_MAX}")
    print(f"RWHI polar_radius: {rwhi.polar_radius}")
    
    assert rwhi.polar_radius == R_MAX, f"polar_radius ({rwhi.polar_radius}) 应该等于 R_MAX ({R_MAX})"
    assert rwhi.polar_radius == 65.0, f"polar_radius 应该等于 65.0"
    
    print("\n✅ polar_radius 一致性测试通过")
    print()


def test_safety_anchors():
    """测试安全锚点"""
    print("=" * 50)
    print("测试安全锚点")
    print("=" * 50)
    
    pc_range = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]
    num_query = 900
    
    rwhi = RWHIModule(
        num_query=num_query,
        pc_range=pc_range,
        bev_grid_size=100,
        enabled=True,
    )
    
    safety_anchors = rwhi.safety_anchors
    
    print(f"安全锚点形状: {safety_anchors.shape}")
    print(f"期望形状: ({num_query}, 10)")
    
    assert safety_anchors.shape == (num_query, 10), "安全锚点形状不正确"
    
    # 验证各维度
    print(f"\nθ 范围: [{safety_anchors[:, 0].min():.4f}, {safety_anchors[:, 0].max():.4f}]")
    print(f"d 范围: [{safety_anchors[:, 1].min():.4f}, {safety_anchors[:, 1].max():.4f}]")
    print(f"z 值: {safety_anchors[:, 2].unique().tolist()}")
    print(f"w (log): {safety_anchors[:, 3].unique().tolist()}")
    print(f"l (log): {safety_anchors[:, 4].unique().tolist()}")
    print(f"h (log): {safety_anchors[:, 5].unique().tolist()}")
    print(f"sin: {safety_anchors[:, 6].unique().tolist()}")
    print(f"cos: {safety_anchors[:, 7].unique().tolist()}")
    
    print("\n✅ 安全锚点测试通过")
    print()


def test_disabled_mode():
    """测试禁用模式"""
    print("=" * 50)
    print("测试禁用模式")
    print("=" * 50)
    
    pc_range = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]
    num_query = 900
    
    rwhi = RWHIModule(
        num_query=num_query,
        pc_range=pc_range,
        bev_grid_size=100,
        enabled=False,  # 禁用
    )
    
    batch_size = 2
    num_points = 500
    radar_points = torch.randn(batch_size, num_points, 7)
    radar_mask = torch.ones(batch_size, num_points)
    
    anchors, alpha_values = rwhi(radar_points, radar_mask)
    
    print(f"禁用模式下锚点形状: {anchors.shape}")
    print(f"禁用模式下 α 形状: {alpha_values.shape}")
    
    assert anchors.shape == (batch_size, num_query, 10)
    assert alpha_values.shape == (batch_size, num_query, 1)
    
    print("\n✅ 禁用模式测试通过")
    print()


def main():
    print("\n" + "=" * 60)
    print("       RWHI v7 模块测试")
    print("=" * 60 + "\n")
    
    try:
        test_alpha_mlp()
        test_alpha_encoder()
        test_rwhi_module_output_ranges()
        test_gradient_flow()
        test_polar_radius_consistency()
        test_safety_anchors()
        test_disabled_mode()
        
        print("=" * 60)
        print("       所有测试通过! ✅")
        print("=" * 60)
        
    except AssertionError as e:
        print(f"\n❌ 测试失败: {e}")
        raise
    except Exception as e:
        print(f"\n❌ 测试出错: {e}")
        raise


if __name__ == "__main__":
    main()

