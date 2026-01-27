"""
GGF2.0 单元测试和验证脚本

用于验证：
1. GGF 模块各组件的正确性
2. 不同配置组合下模型能正常 forward
3. 几何场的数值范围和分布

使用方法：
    python tools/test_ggf.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Optional, Tuple


def create_mock_radar_data(batch_size: int = 2, num_points: int = 100, device: str = 'cuda'):
    """
    创建模拟的雷达点云数据
    
    Returns:
        radar_points: [B, M, C] 雷达点 (x, y, z, rcs, v_r, ...)
        radar_mask: [B, M] 有效点掩码
    """
    # 在 BEV 范围内随机生成点
    x = torch.rand(batch_size, num_points, device=device) * 80 - 40  # [-40, 40]
    y = torch.rand(batch_size, num_points, device=device) * 80 - 40  # [-40, 40]
    z = torch.rand(batch_size, num_points, device=device) * 2 - 1   # [-1, 1]
    rcs = torch.rand(batch_size, num_points, device=device) * 30 - 10  # [-10, 20] dBsm
    v_r = torch.rand(batch_size, num_points, device=device) * 40 - 20  # [-20, 20] m/s
    
    # 添加一些额外的特征（占位）
    extra = torch.zeros(batch_size, num_points, 2, device=device)
    
    radar_points = torch.stack([x, y, z, rcs, v_r], dim=-1)
    radar_points = torch.cat([radar_points, extra], dim=-1)  # [B, M, 7]
    
    # 随机掩码（80% 有效）
    radar_mask = torch.rand(batch_size, num_points, device=device) > 0.2
    radar_mask = radar_mask.float()
    
    return radar_points, radar_mask


def create_mock_query_data(batch_size: int = 2, num_query: int = 100, embed_dims: int = 256, device: str = 'cuda'):
    """
    创建模拟的 Query 数据
    
    Returns:
        query_bbox: [B, Q, 10] Query bbox (极坐标格式)
        query_feat: [B, Q, C] Query 特征
    """
    # 极坐标格式：[theta, d, z, w, l, h, sin, cos, vx, vy]
    theta = torch.rand(batch_size, num_query, device=device)  # [0, 1]
    d = torch.rand(batch_size, num_query, device=device) * 0.8 + 0.1  # [0.1, 0.9]
    z = torch.full((batch_size, num_query), 0.5, device=device)
    w = torch.zeros(batch_size, num_query, device=device)  # log scale
    l = torch.zeros(batch_size, num_query, device=device)  # log scale
    h = torch.full((batch_size, num_query), 0.2, device=device)  # log scale
    sin_yaw = torch.zeros(batch_size, num_query, device=device)
    cos_yaw = torch.ones(batch_size, num_query, device=device)
    vx = torch.zeros(batch_size, num_query, device=device)
    vy = torch.zeros(batch_size, num_query, device=device)
    
    query_bbox = torch.stack([theta, d, z, w, l, h, sin_yaw, cos_yaw, vx, vy], dim=-1)
    query_feat = torch.randn(batch_size, num_query, embed_dims, device=device)
    
    return query_bbox, query_feat


def test_native_rgf():
    """测试 NativeRGF 模块"""
    print("\n" + "=" * 60)
    print("测试 NativeRGF 模块")
    print("=" * 60)
    
    from models.ggf import NativeRGF
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"使用设备: {device}")
    
    # 创建模块
    rgf = NativeRGF(
        bev_grid_size=100,
        pc_range=[-51.2, -51.2, -5.0, 51.2, 51.2, 3.0],
        default_sigma_x=3.0,
        default_sigma_y=3.0,
    ).to(device)
    
    # 创建模拟数据
    radar_points, radar_mask = create_mock_radar_data(device=device)
    
    # 前向传播
    gaussian_field, params = rgf(radar_points, radar_mask, return_params=True)
    
    print(f"输入形状: radar_points={radar_points.shape}, radar_mask={radar_mask.shape}")
    print(f"输出形状: gaussian_field={gaussian_field.shape}")
    print(f"高斯场统计: min={gaussian_field.min().item():.4f}, max={gaussian_field.max().item():.4f}, mean={gaussian_field.mean().item():.4f}")
    print(f"高斯参数: centers={params['centers'].shape}, sigmas={params['sigmas'].shape}")
    
    # 验证输出范围
    assert gaussian_field.min() >= 0, "高斯场应该非负"
    assert not torch.isnan(gaussian_field).any(), "高斯场不应包含 NaN"
    
    print("✓ NativeRGF 测试通过")
    return True


def test_geometry_field_builder():
    """测试 GeometryFieldBuilder 模块"""
    print("\n" + "=" * 60)
    print("测试 GeometryFieldBuilder 模块")
    print("=" * 60)
    
    from models.ggf import GeometryFieldBuilder
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # 创建模块
    builder = GeometryFieldBuilder(
        bev_grid_size=100,
        pc_range=[-51.2, -51.2, -5.0, 51.2, 51.2, 3.0],
        use_native_rgf=True,
        linear_scale=1.0,
        linear_bias=1.0,
        linear_max=5.0,
    ).to(device)
    
    # 创建模拟数据
    radar_points, radar_mask = create_mock_radar_data(device=device)
    
    # 前向传播
    linear_field, log_field, params = builder(radar_points, radar_mask)
    
    print(f"线性场: min={linear_field.min().item():.4f}, max={linear_field.max().item():.4f}, mean={linear_field.mean().item():.4f}")
    print(f"对数场: min={log_field.min().item():.4f}, max={log_field.max().item():.4f}, mean={log_field.mean().item():.4f}")
    
    # 验证数值范围
    assert linear_field.max() <= 5.0, "线性场应该被裁剪到 linear_max"
    assert log_field.min() >= -100.0, "对数场应该被裁剪到 bias_min"
    
    print("✓ GeometryFieldBuilder 测试通过")
    return True


def test_mgc_module():
    """测试 MGC 模块"""
    print("\n" + "=" * 60)
    print("测试 MGC 模块")
    print("=" * 60)
    
    from models.ggf import MGCModule
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # 创建模块
    mgc = MGCModule(
        embed_dims=256,
        num_points=4,
        pc_range=[-51.2, -51.2, -5.0, 51.2, 51.2, 3.0],
        constraint_mode='soft',
        constraint_strength=1.0,
    ).to(device)
    
    # 创建模拟数据
    query_bbox, query_feat = create_mock_query_data(device=device)
    B, Q, _ = query_bbox.shape
    sampling_offset = torch.randn(B, Q, 12, 3, device=device)  # 4 points * 3 depth
    
    # 模拟高斯参数
    radar_points, radar_mask = create_mock_radar_data(device=device)
    M = radar_points.shape[1]
    gaussian_params = {
        'centers': radar_points[..., :2],  # [B, M, 2]
        'sigmas': torch.full((B, M, 2), 3.0, device=device),
        'amplitudes': torch.ones(B, M, device=device),
    }
    
    # 前向传播
    adjusted_offset, mgc_info = mgc(query_bbox, query_feat, sampling_offset, gaussian_params, d_region=0.1)
    
    print(f"原始 offset 范围: [{sampling_offset.min().item():.4f}, {sampling_offset.max().item():.4f}]")
    print(f"调整后 offset 范围: [{adjusted_offset.min().item():.4f}, {adjusted_offset.max().item():.4f}]")
    
    # 验证输出形状
    assert adjusted_offset.shape == sampling_offset.shape, "输出形状应与输入相同"
    
    print("✓ MGC 测试通过")
    return True


def test_gga_module():
    """测试 GGA 模块"""
    print("\n" + "=" * 60)
    print("测试 GGA 模块")
    print("=" * 60)
    
    from models.ggf import GGAModule
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # 创建模块
    gga = GGAModule(
        embed_dims=256,
        num_heads=8,
        temperature=1.0,
        bias_scale=1.0,
    ).to(device)
    
    # 创建模拟数据
    query_bbox, query_feat = create_mock_query_data(device=device)
    B, Q, _ = query_bbox.shape
    
    # 模拟 attention logits
    attn_logits = torch.randn(B, 8, Q, Q, device=device)
    
    # 模拟高斯参数
    radar_points, radar_mask = create_mock_radar_data(device=device)
    M = radar_points.shape[1]
    gaussian_params = {
        'centers': radar_points[..., :2],
        'sigmas': torch.full((B, M, 2), 3.0, device=device),
    }
    
    # 前向传播
    adjusted_logits, gga_info = gga(attn_logits, query_bbox, query_feat, gaussian_params, 
                                    pc_range=[-51.2, -51.2, -5.0, 51.2, 51.2, 3.0])
    
    print(f"原始 logits 范围: [{attn_logits.min().item():.4f}, {attn_logits.max().item():.4f}]")
    print(f"调整后 logits 范围: [{adjusted_logits.min().item():.4f}, {adjusted_logits.max().item():.4f}]")
    
    if 'geometry_bias' in gga_info:
        gb = gga_info['geometry_bias']
        print(f"几何偏置范围: [{gb.min().item():.4f}, {gb.max().item():.4f}]")
    
    print("✓ GGA 测试通过")
    return True


def test_ggf_module():
    """测试 GGF 总模块"""
    print("\n" + "=" * 60)
    print("测试 GGFModule 总模块")
    print("=" * 60)
    
    from models.ggf import GGFModule
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # 创建模块
    ggf = GGFModule(
        enabled=True,
        embed_dims=256,
        pc_range=[-51.2, -51.2, -5.0, 51.2, 51.2, 3.0],
        bev_grid_size=100,
        use_mgc=True,
        use_gga=True,
        use_unified_field=True,
        use_native_rgf=True,
    ).to(device)
    
    # 创建模拟数据
    radar_points, radar_mask = create_mock_radar_data(device=device)
    
    # 构建几何场
    linear_field, log_field = ggf(radar_points, radar_mask)
    
    print(f"线性场形状: {linear_field.shape}")
    print(f"对数场形状: {log_field.shape}")
    
    # 测试 MGC
    query_bbox, query_feat = create_mock_query_data(device=device)
    B, Q, _ = query_bbox.shape
    sampling_offset = torch.randn(B, Q, 12, 3, device=device)
    
    adjusted_offset, mgc_info = ggf.apply_mgc(query_bbox, query_feat, sampling_offset, d_region=0.1)
    print(f"MGC 调整后 offset 形状: {adjusted_offset.shape}")
    
    # 测试 GGA
    attn_logits = torch.randn(B, 8, Q, Q, device=device)
    adjusted_logits, gga_info = ggf.apply_gga(attn_logits, query_bbox, query_feat)
    print(f"GGA 调整后 logits 形状: {adjusted_logits.shape}")
    
    print("✓ GGFModule 测试通过")
    return True


def test_config_combinations():
    """测试不同配置组合"""
    print("\n" + "=" * 60)
    print("测试不同配置组合")
    print("=" * 60)
    
    from models.ggf import GGFModule
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # 创建模拟数据
    radar_points, radar_mask = create_mock_radar_data(device=device)
    query_bbox, query_feat = create_mock_query_data(device=device)
    
    configs = [
        # (enabled, use_unified_field, use_mgc, use_gga, use_native_rgf)
        (False, True, True, True, True),    # 全部关闭
        (True, True, False, False, True),   # 只开 unified_field
        (True, True, True, False, True),    # unified_field + MGC
        (True, True, False, True, True),    # unified_field + GGA
        (True, True, True, True, True),     # 完全体
        (True, True, True, True, False),    # 不使用 NativeRGF
    ]
    
    for i, (enabled, use_uf, use_mgc, use_gga, use_rgf) in enumerate(configs):
        config_name = f"配置 {i+1}: enabled={enabled}, unified_field={use_uf}, mgc={use_mgc}, gga={use_gga}, rgf={use_rgf}"
        print(f"\n测试 {config_name}")
        
        try:
            ggf = GGFModule(
                enabled=enabled,
                embed_dims=256,
                pc_range=[-51.2, -51.2, -5.0, 51.2, 51.2, 3.0],
                bev_grid_size=100,
                use_mgc=use_mgc,
                use_gga=use_gga,
                use_unified_field=use_uf,
                use_native_rgf=use_rgf,
            ).to(device)
            
            # 测试 forward
            linear_field, log_field = ggf(radar_points, radar_mask)
            
            # 测试 MGC
            B, Q, _ = query_bbox.shape
            sampling_offset = torch.randn(B, Q, 12, 3, device=device)
            adjusted_offset, _ = ggf.apply_mgc(query_bbox, query_feat, sampling_offset, d_region=0.1)
            
            # 测试 GGA
            attn_logits = torch.randn(B, 8, Q, Q, device=device)
            adjusted_logits, _ = ggf.apply_gga(attn_logits, query_bbox, query_feat)
            
            print(f"  ✓ 通过")
        except Exception as e:
            print(f"  ✗ 失败: {e}")
            return False
    
    print("\n✓ 所有配置组合测试通过")
    return True


def test_rwhi_integration():
    """测试 RWHI 与 GGF 的集成"""
    print("\n" + "=" * 60)
    print("测试 RWHI 与 GGF 的集成")
    print("=" * 60)
    
    from models.rwhi import RWHIModule
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    pc_range = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]
    
    # 创建带 GGF 的 RWHI 模块
    ggf_cfg = dict(
        enabled=True,
        embed_dims=256,
        use_mgc=True,
        use_gga=True,
        use_unified_field=True,
        use_native_rgf=True,
    )
    
    rwhi = RWHIModule(
        num_query=100,
        pc_range=pc_range,
        bev_grid_size=100,
        enabled=True,
        ggf_cfg=ggf_cfg,
    ).to(device)
    
    # 创建模拟数据
    radar_points, radar_mask = create_mock_radar_data(device=device)
    
    # 前向传播
    anchors, alpha_values = rwhi(radar_points, radar_mask)
    
    print(f"输出锚点形状: {anchors.shape}")
    print(f"输出 alpha 形状: {alpha_values.shape}")
    print(f"锚点 theta 范围: [{anchors[..., 0].min().item():.4f}, {anchors[..., 0].max().item():.4f}]")
    print(f"锚点 d 范围: [{anchors[..., 1].min().item():.4f}, {anchors[..., 1].max().item():.4f}]")
    
    # 验证 GGF 模块被正确创建
    ggf_module = rwhi.get_ggf_module()
    assert ggf_module is not None, "GGF 模块应该被创建"
    print(f"GGF 模块已创建: {type(ggf_module).__name__}")
    
    # 对比：不使用 GGF 的 RWHI
    rwhi_no_ggf = RWHIModule(
        num_query=100,
        pc_range=pc_range,
        bev_grid_size=100,
        enabled=True,
        ggf_cfg=None,
    ).to(device)
    
    anchors_no_ggf, alpha_no_ggf = rwhi_no_ggf(radar_points, radar_mask)
    
    # 比较输出差异（由于 GGF 几何场的影响，结果应该不同）
    diff = (anchors - anchors_no_ggf).abs().mean().item()
    print(f"有/无 GGF 的锚点平均差异: {diff:.6f}")
    
    print("✓ RWHI 与 GGF 集成测试通过")
    return True


def visualize_ggf_output(save_dir: str = 'ggf_debug'):
    """可视化 GGF 输出"""
    print("\n" + "=" * 60)
    print("可视化 GGF 输出")
    print("=" * 60)
    
    import os
    os.makedirs(save_dir, exist_ok=True)
    
    from models.ggf import GGFModule, GGFDebugger
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # 创建模块
    ggf = GGFModule(
        enabled=True,
        embed_dims=256,
        pc_range=[-51.2, -51.2, -5.0, 51.2, 51.2, 3.0],
        bev_grid_size=100,
        use_mgc=True,
        use_gga=True,
        use_unified_field=True,
        use_native_rgf=True,
    ).to(device)
    
    # 创建模拟数据
    radar_points, radar_mask = create_mock_radar_data(batch_size=1, device=device)
    
    # 构建几何场
    linear_field, log_field = ggf(radar_points, radar_mask)
    
    # 打印统计信息
    GGFDebugger.print_statistics(ggf, linear_field, log_field)
    
    # 可视化
    try:
        save_path = os.path.join(save_dir, 'geometry_field.png')
        GGFDebugger.visualize_geometry_field(
            linear_field, log_field, radar_points, save_path=save_path
        )
        print(f"可视化已保存到: {save_path}")
    except ImportError:
        print("跳过可视化（matplotlib 未安装）")
    
    return True


def main():
    """运行所有测试"""
    print("=" * 60)
    print("GGF2.0 单元测试")
    print("=" * 60)
    
    # 检查 CUDA 可用性
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"使用设备: {device}")
    
    tests = [
        ("NativeRGF", test_native_rgf),
        ("GeometryFieldBuilder", test_geometry_field_builder),
        ("MGC", test_mgc_module),
        ("GGA", test_gga_module),
        ("GGFModule", test_ggf_module),
        ("配置组合", test_config_combinations),
        ("RWHI 集成", test_rwhi_integration),
        ("可视化", visualize_ggf_output),
    ]
    
    results = []
    for name, test_func in tests:
        try:
            success = test_func()
            results.append((name, success))
        except Exception as e:
            print(f"\n✗ {name} 测试失败: {e}")
            import traceback
            traceback.print_exc()
            results.append((name, False))
    
    # 打印总结
    print("\n" + "=" * 60)
    print("测试总结")
    print("=" * 60)
    
    all_passed = True
    for name, success in results:
        status = "✓ 通过" if success else "✗ 失败"
        print(f"  {name}: {status}")
        if not success:
            all_passed = False
    
    if all_passed:
        print("\n所有测试通过！")
    else:
        print("\n部分测试失败，请检查错误信息。")
    
    return all_passed


if __name__ == '__main__':
    success = main()
    exit(0 if success else 1)
