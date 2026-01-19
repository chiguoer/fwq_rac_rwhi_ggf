#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
模型配置测试脚本
================
功能：检查配置文件与模型代码的参数是否匹配，提前发现问题

使用方法：
    python test_model_config.py --config configs/racformer_with_rhgm_radarbevnet.py

作者：AI Assistant
日期：2025-12-23
"""

import argparse
import sys
import os
import inspect

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def load_config(config_path):
    """加载配置文件"""
    import importlib.util
    spec = importlib.util.spec_from_file_location("config", config_path)
    config = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config)
    return config


def get_class_init_params(cls):
    """获取类的__init__方法的参数列表"""
    sig = inspect.signature(cls.__init__)
    params = {}
    for name, param in sig.parameters.items():
        if name == 'self':
            continue
        params[name] = {
            'default': param.default if param.default != inspect.Parameter.empty else None,
            'has_default': param.default != inspect.Parameter.empty,
            'kind': str(param.kind)
        }
    return params


def check_config_params(config_dict, cls, module_name):
    """检查配置参数是否与类的__init__参数匹配"""
    print(f"\n{'='*60}")
    print(f"检查模块: {module_name}")
    print(f"{'='*60}")
    
    # 获取类接受的参数
    class_params = get_class_init_params(cls)
    print(f"\n📋 {cls.__name__} 类接受的参数:")
    for name, info in class_params.items():
        default_str = f" = {info['default']}" if info['has_default'] else " (必需)"
        print(f"   - {name}{default_str}")
    
    # 检查配置中的参数
    print(f"\n📄 配置文件中的参数:")
    for key, value in config_dict.items():
        print(f"   - {key} = {value}")
    
    # 找出不匹配的参数
    errors = []
    warnings = []
    
    # 检查配置中是否有类不接受的参数
    for key in config_dict.keys():
        if key not in class_params:
            errors.append(f"❌ 配置中的 '{key}' 不是 {cls.__name__}.__init__() 的有效参数")
    
    # 检查必需参数是否都提供了
    for name, info in class_params.items():
        if not info['has_default'] and name not in config_dict:
            warnings.append(f"⚠️  必需参数 '{name}' 未在配置中提供")
    
    # 输出结果
    if errors:
        print(f"\n🚨 发现 {len(errors)} 个错误:")
        for err in errors:
            print(f"   {err}")
    else:
        print(f"\n✅ 参数检查通过！")
    
    if warnings:
        print(f"\n⚠️  发现 {len(warnings)} 个警告:")
        for warn in warnings:
            print(f"   {warn}")
    
    return len(errors) == 0


def test_rhgm_module(config):
    """测试RHGM模块配置"""
    try:
        from models.rhgm import RHGM, RHGMWrapper
        
        if hasattr(config, 'rhgm_cfg'):
            rhgm_cfg = config.rhgm_cfg
            print("\n" + "="*60)
            print("测试 RHGM 模块")
            print("="*60)
            
            # RHGMWrapper 接受 rhgm_cfg 作为参数
            print(f"\n📄 rhgm_cfg 配置内容:")
            for key, value in rhgm_cfg.items():
                print(f"   - {key} = {value}")
            
            # 检查 RHGM 类的参数
            rhgm_params = get_class_init_params(RHGM)
            print(f"\n📋 RHGM 类接受的参数:")
            for name, info in rhgm_params.items():
                default_str = f" = {info['default']}" if info['has_default'] else " (必需)"
                print(f"   - {name}{default_str}")
            
            # 找出不匹配的参数
            errors = []
            for key in rhgm_cfg.keys():
                if key not in rhgm_params:
                    errors.append(f"❌ rhgm_cfg 中的 '{key}' 不是 RHGM.__init__() 的有效参数")
            
            if errors:
                print(f"\n🚨 发现 {len(errors)} 个错误:")
                for err in errors:
                    print(f"   {err}")
                return False
            else:
                print(f"\n✅ RHGM 参数检查通过！")
                return True
        else:
            print("\n⚠️  配置中没有 rhgm_cfg")
            return True
            
    except ImportError as e:
        print(f"\n❌ 无法导入 RHGM 模块: {e}")
        return False


def test_radar_bev_net_module(config):
    """测试RadarBEVNet模块配置"""
    try:
        from models.radar_bev_net import RadarBEVNet, RadarBEVNetWrapper
        
        if hasattr(config, 'radar_bev_net_cfg'):
            cfg = config.radar_bev_net_cfg
            return check_config_params(cfg, RadarBEVNet, "RadarBEVNet")
        else:
            print("\n⚠️  配置中没有 radar_bev_net_cfg")
            return True
            
    except ImportError as e:
        print(f"\n❌ 无法导入 RadarBEVNet 模块: {e}")
        return False


def test_racformer_model(config):
    """测试RaCFormer主模型配置"""
    try:
        from models.racformer import RaCFormer
        
        if hasattr(config, 'model'):
            model_cfg = dict(config.model)  # 转换为普通字典
            
            print("\n" + "="*60)
            print("测试 RaCFormer 主模型")
            print("="*60)
            
            # 获取RaCFormer的参数
            racformer_params = get_class_init_params(RaCFormer)
            print(f"\n📋 RaCFormer 类接受的参数:")
            for name, info in racformer_params.items():
                default_str = f" = {info['default']}" if info['has_default'] else " (必需)"
                print(f"   - {name}{default_str}")
            
            # 检查配置中的参数
            print(f"\n📄 model 配置中的顶层参数:")
            errors = []
            for key in model_cfg.keys():
                if key == 'type':
                    continue  # type 是 mmcv 的特殊参数
                value = model_cfg[key]
                value_str = str(value)[:50] + "..." if len(str(value)) > 50 else str(value)
                print(f"   - {key} = {value_str}")
                
                if key not in racformer_params:
                    errors.append(f"❌ model 配置中的 '{key}' 不是 RaCFormer.__init__() 的有效参数")
            
            if errors:
                print(f"\n🚨 发现 {len(errors)} 个错误:")
                for err in errors:
                    print(f"   {err}")
                return False
            else:
                print(f"\n✅ RaCFormer 参数检查通过！")
                return True
        else:
            print("\n❌ 配置中没有 model")
            return False
            
    except ImportError as e:
        print(f"\n❌ 无法导入 RaCFormer 模块: {e}")
        return False


def test_model_instantiation(config):
    """尝试实例化模型（不加载权重）"""
    print("\n" + "="*60)
    print("尝试实例化模型")
    print("="*60)
    
    try:
        import torch
        from mmdet3d.models import build_model
        
        # 构建模型配置 - 使用普通 dict
        model_cfg = dict(config.model)
        
        # 设置 train_cfg 和 test_cfg
        train_cfg = model_cfg.pop('train_cfg', None)
        test_cfg = model_cfg.pop('test_cfg', None)
        
        print("\n🔨 正在构建模型...")
        
        # 尝试构建模型 - mmdet3d 的 build_model 需要 dict
        model = build_model(
            model_cfg,
            train_cfg=train_cfg,
            test_cfg=test_cfg
        )
        
        print("✅ 模型构建成功！")
        print(f"\n📊 模型信息:")
        print(f"   - 类型: {type(model).__name__}")
        
        # 统计参数量
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"   - 总参数量: {total_params:,}")
        print(f"   - 可训练参数量: {trainable_params:,}")
        
        # 检查关键模块是否正确初始化
        print(f"\n🔍 关键模块检查:")
        print(f"   - use_rhgm: {getattr(model, 'use_rhgm', 'N/A')}")
        print(f"   - use_radar_bev_net: {getattr(model, 'use_radar_bev_net', 'N/A')}")
        
        if hasattr(model, 'rhgm'):
            print(f"   - RHGM模块: ✅ 已初始化")
        else:
            print(f"   - RHGM模块: ❌ 未初始化")
            
        if hasattr(model, 'radar_bev_net'):
            print(f"   - RadarBEVNet模块: ✅ 已初始化")
        else:
            print(f"   - RadarBEVNet模块: ❌ 未初始化")
        
        return True
        
    except Exception as e:
        print(f"\n❌ 模型构建失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    parser = argparse.ArgumentParser(description='测试模型配置')
    parser.add_argument('--config', type=str, 
                        default='configs/racformer_with_rhgm_radarbevnet.py',
                        help='配置文件路径')
    parser.add_argument('--no-instantiate', action='store_true',
                        help='跳过模型实例化测试（更快）')
    args = parser.parse_args()
    
    print("="*60)
    print("🔍 模型配置测试工具")
    print("="*60)
    print(f"配置文件: {args.config}")
    
    # 加载配置
    try:
        config = load_config(args.config)
        print("✅ 配置文件加载成功")
    except Exception as e:
        print(f"❌ 配置文件加载失败: {e}")
        return 1
    
    # 运行测试
    all_passed = True
    
    # 测试 RHGM 模块
    if not test_rhgm_module(config):
        all_passed = False
    
    # 测试 RadarBEVNet 模块
    if not test_radar_bev_net_module(config):
        all_passed = False
    
    # 测试 RaCFormer 主模型
    if not test_racformer_model(config):
        all_passed = False
    
    # 尝试实例化模型
    if not args.no_instantiate:
        if not test_model_instantiation(config):
            all_passed = False
    
    # 输出总结
    print("\n" + "="*60)
    print("📊 测试总结")
    print("="*60)
    
    if all_passed:
        print("✅ 所有测试通过！可以开始训练。")
        return 0
    else:
        print("❌ 部分测试失败，请根据上面的错误信息修复配置。")
        return 1


if __name__ == '__main__':
    sys.exit(main())

