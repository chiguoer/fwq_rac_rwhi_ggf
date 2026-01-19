#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
调试脚本：定位DDP中未使用的参数

使用方法：
    python tools/debug_find_unused_param.py --config configs/racformer_with_rwhi.py

输出：
    列出所有模型参数及其索引，帮助定位 "Parameter indices which did not receive grad: XXX" 报错
"""

import argparse
import sys
import os

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def parse_args():
    parser = argparse.ArgumentParser(description='定位DDP未使用的参数')
    parser.add_argument('--config', type=str, 
                        default='configs/racformer_with_rwhi.py',
                        help='配置文件路径')
    parser.add_argument('--target-idx', type=int, default=103,
                        help='要定位的参数索引 (从DDP报错中获取)')
    return parser.parse_args()


def main():
    args = parse_args()
    
    print("=" * 70)
    print("DDP 未使用参数调试工具")
    print("=" * 70)
    
    # 尝试导入必要的模块
    try:
        import torch
        from mmcv import Config
        from mmdet3d.models import build_model
    except ImportError as e:
        print(f"[错误] 无法导入必要模块: {e}")
        print("请确保在正确的conda环境中运行此脚本")
        return
    
    # 加载配置
    print(f"\n[1] 加载配置文件: {args.config}")
    cfg = Config.fromfile(args.config)
    
    # 构建模型
    print(f"\n[2] 构建模型...")
    model = build_model(cfg.model)
    model.init_weights()
    
    # 遍历所有参数
    print(f"\n[3] 遍历模型参数 (共 {sum(1 for _ in model.named_parameters())} 个)...")
    print("-" * 70)
    
    target_param_name = None
    target_param_shape = None
    
    for idx, (name, param) in enumerate(model.named_parameters()):
        # 标记目标参数
        marker = " <-- 目标参数!" if idx == args.target_idx else ""
        
        # 检查是否需要梯度
        grad_info = "requires_grad=True" if param.requires_grad else "requires_grad=False (冻结)"
        
        # 打印参数信息
        if idx == args.target_idx:
            print(f"\n{'='*70}")
            print(f">>> INDEX {idx}: {name}")
            print(f"    Shape: {list(param.shape)}")
            print(f"    Dtype: {param.dtype}")
            print(f"    Device: {param.device}")
            print(f"    {grad_info}")
            print(f"{'='*70}\n")
            target_param_name = name
            target_param_shape = list(param.shape)
        elif abs(idx - args.target_idx) <= 5:
            # 打印目标附近的参数以提供上下文
            print(f"[{idx:4d}] {name:60s} | shape={list(param.shape)}{marker}")
    
    print("-" * 70)
    
    # 总结
    print(f"\n[4] 总结")
    print("=" * 70)
    if target_param_name:
        print(f"目标参数 (index={args.target_idx}):")
        print(f"  名称: {target_param_name}")
        print(f"  形状: {target_param_shape}")
        
        # 分析参数所属模块
        parts = target_param_name.split('.')
        print(f"\n模块层级:")
        for i, part in enumerate(parts):
            print(f"  {'  ' * i}└─ {part}")
        
        # 提供修复建议
        print(f"\n[5] 修复建议")
        print("-" * 70)
        if 'init_query_bbox' in target_param_name:
            print("问题: init_query_bbox 是后备embedding，仅在无雷达点时使用。")
            print("解决: 已通过 find_unused_parameters=True 处理。")
        elif 'rwhi' in target_param_name.lower():
            print("问题: RWHI模块中存在条件分支，某些参数可能未参与计算。")
            print("解决: 检查对应层是否在forward中被调用。")
        elif 'pos_embed' in target_param_name or 'pos2content' in target_param_name:
            print("问题: 位置编码相关层可能存在未使用的分支。")
            print("解决: 检查forward中的条件分支。")
        else:
            print("请检查该参数所属模块的forward方法，确认是否参与计算。")
    else:
        print(f"[警告] 未找到索引 {args.target_idx} 的参数。")
        print(f"模型共有 {sum(1 for _ in model.named_parameters())} 个参数。")
    
    print("=" * 70)


if __name__ == '__main__':
    main()

