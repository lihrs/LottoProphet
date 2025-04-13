#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
生成占位符模型文件，用于测试
"""
import os
import sys
import time
import torch
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler


script_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(script_dir)
if project_dir not in sys.path:
    sys.path.append(project_dir)

from model import LstmCRFModel

def generate_placeholder_model(model_dir, model_type):
    """
    生成占位符模型文件
    
    Args:
        model_dir: 模型保存目录
        model_type: 模型类型，'dlt'或'ssq'
    """
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    
  
    if model_type == 'dlt':
        input_dim = 7  # 5红球 + 2蓝球
        hidden_dim = 128
        red_output_dim = 35  # 红球输出维度 (1-35)
        blue_output_dim = 12  # 蓝球输出维度 (1-12)
        red_seq_length = 5  # 红球序列长度
        blue_seq_length = 2  # 蓝球序列长度
        model_file = 'dlt_model.pth'
    elif model_type == 'ssq':
        input_dim = 7  # 6红球 + 1蓝球
        hidden_dim = 128
        red_output_dim = 33  # 红球输出维度 (1-33)
        blue_output_dim = 16  # 蓝球输出维度 (1-16)
        red_seq_length = 6  # 红球序列长度
        blue_seq_length = 1  # 蓝球序列长度
        model_file = 'ssq_model.pth'
    else:
        raise ValueError(f"不支持的模型类型：{model_type}")
    
 
    red_model = LstmCRFModel(
        input_dim, hidden_dim,
        red_output_dim, red_seq_length,
        num_layers=1, dropout=0.3
    )
    
   
    blue_model = LstmCRFModel(
        input_dim, hidden_dim,
        blue_output_dim, blue_seq_length,
        num_layers=1, dropout=0.3
    )
    
  
    combined_model = {
        'red_model': red_model.state_dict(),
        'blue_model': blue_model.state_dict(),
        'metadata': {
            'timestamp': time.time(),
            'input_dim': input_dim,
            'hidden_dim': hidden_dim,
            'red_output_dim': red_output_dim,
            'blue_output_dim': blue_output_dim,
            'red_seq_length': red_seq_length,
            'blue_seq_length': blue_seq_length
        }
    }
    
   
    torch.save(combined_model, os.path.join(model_dir, model_file))
    
 
    scaler = StandardScaler()
  
    sample_data = np.random.randn(100, input_dim)
    scaler.fit(sample_data)
    
   
    joblib.dump(scaler, os.path.join(model_dir, 'scaler_X.pkl'))
    
    print(f"已在 {model_dir} 生成 {model_type} 模型占位符文件")

def main():
    """主函数"""
  
    generate_placeholder_model(os.path.join(script_dir, 'dlt'), 'dlt')
    
 
    generate_placeholder_model(os.path.join(script_dir, 'ssq'), 'ssq')
    
    print("占位符模型文件生成完成！")

if __name__ == "__main__":
    main() 