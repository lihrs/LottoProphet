#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
测试模型加载脚本
"""
import os
import sys

# 添加项目根目录到路径
script_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(script_dir)
if project_dir not in sys.path:
    sys.path.append(project_dir)

from model_utils import load_resources_pytorch

def test_model_loading():
    """测试模型加载"""
    print("测试加载大乐透模型...")
    try:
        red_model, blue_model, scaler_X = load_resources_pytorch('dlt')
        print("✓ 大乐透模型加载成功")
        print(f"  - 红球模型输出尺寸: {red_model.output_dim}")
        print(f"  - 蓝球模型输出尺寸: {blue_model.output_dim}")
        print(f"  - 特征缩放器特征数: {scaler_X.n_features_in_}")
    except Exception as e:
        print(f"✗ 大乐透模型加载失败: {str(e)}")
    
    print("\n测试加载双色球模型...")
    try:
        red_model, blue_model, scaler_X = load_resources_pytorch('ssq')
        print("✓ 双色球模型加载成功")
        print(f"  - 红球模型输出尺寸: {red_model.output_dim}")
        print(f"  - 蓝球模型输出尺寸: {blue_model.output_dim}")
        print(f"  - 特征缩放器特征数: {scaler_X.n_features_in_}")
    except Exception as e:
        print(f"✗ 双色球模型加载失败: {str(e)}")

if __name__ == "__main__":
    test_model_loading() 