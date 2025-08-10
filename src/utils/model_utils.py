# -*- coding:utf-8 -*-
"""
模型工具模块，包含与模型加载、预测相关的功能
"""
import os
import torch
import joblib
import numpy as np
# 避免循环导入，将导入移到函数内部

# 添加安全的全局变量，以允许numpy._core.multiarray._reconstruct
try:
    torch.serialization.add_safe_globals(['numpy._core.multiarray._reconstruct'])
except (AttributeError, ImportError):
    # 兼容旧版PyTorch
    pass

# 模型和路径配置
name_path = {
    "dlt": {
        "name": "大乐透",
        "path": "./models/dlt/",
        "model_file": "dlt_model.pth",
        "scaler_X_file": "scaler_X.pkl",
        "train_script": "./src/models/trainers/dlt_trainer.py",
        "fetch_script": "./src/data/fetchers/dlt_fetcher.py"
    },
    "ssq": {
        "name": "双色球",
        "path": "./models/ssq/",
        "model_file": "ssq_model.pth",
        "scaler_X_file": "scaler_X.pkl",
        "train_script": "./src/models/trainers/ssq_trainer.py",
        "fetch_script": "./src/data/fetchers/ssq_fetcher.py"
    }
}

def load_pytorch_model(model_path, input_dim, hidden_dim, output_dim, output_seq_length, lottery_type, 
                     bidirectional=False, attention=False, residual=False, num_layers=1, dropout=0.3):
    """
    加载 PyTorch 模型及缩放器
    
    Args:
        model_path: 模型文件路径
        input_dim: 输入特征维度
        hidden_dim: LSTM隐藏层维度
        output_dim: 输出维度字典 {'red': int, 'blue': int}
        output_seq_length: 输出序列长度字典 {'red': int, 'blue': int}
        lottery_type: 彩票类型
        bidirectional: 是否使用双向LSTM
        attention: 是否使用注意力机制
        residual: 是否使用残差连接
        num_layers: LSTM层数
        dropout: Dropout比例
    
    Returns:
        red_model: 红球预测模型
        blue_model: 蓝球预测模型
        scaler_X: 特征缩放器
    """
    # 在函数内部导入，避免循环导入
    from src.models.lstm_crf import LstmCRFModel
    
    # 设置weights_only=False以兼容PyTorch 2.6+
    checkpoint = torch.load(model_path, map_location=torch.device('cpu'), weights_only=False)
    
    # 检查checkpoint中是否包含模型配置信息
    model_config = checkpoint.get('model_config', {})
    
    # 如果checkpoint中有配置信息，则使用它们覆盖默认值
    bidirectional = model_config.get('bidirectional', bidirectional)
    attention = model_config.get('attention', attention)
    residual = model_config.get('residual', residual)
    num_layers = model_config.get('num_layers', num_layers)
    dropout = model_config.get('dropout', dropout)

    # 加载红球模型
    red_model = LstmCRFModel(
        input_size=input_dim, 
        hidden_size=hidden_dim, 
        num_layers=num_layers,
        num_classes=output_dim['red'],
        lottery_type=lottery_type,
        dropout=dropout,
        bidirectional=bidirectional,
        use_attention=attention,
        use_residual=residual,
        output_seq_length=output_seq_length['red']
    )
    red_model.load_state_dict(checkpoint['red_model'])
    red_model.eval()

    # 加载蓝球模型
    blue_model = LstmCRFModel(
        input_size=input_dim, 
        hidden_size=hidden_dim, 
        num_layers=num_layers,
        num_classes=output_dim['blue'],
        lottery_type=lottery_type,
        dropout=dropout,
        bidirectional=bidirectional,
        use_attention=attention,
        use_residual=residual,
        output_seq_length=output_seq_length['blue']
    )
    blue_model.load_state_dict(checkpoint['blue_model'])
    blue_model.eval()

    # 加载缩放器
    scaler_X_path = os.path.join(os.path.dirname(model_path), name_path[lottery_type]['scaler_X_file'])
    if not os.path.exists(scaler_X_path):
        raise FileNotFoundError(f"特征缩放器文件不存在：{scaler_X_path}")
    scaler_X = joblib.load(scaler_X_path)

    return red_model, blue_model, scaler_X

def load_resources_pytorch(lottery_type):
    """
    根据彩票类型加载模型和资源
    """
    if lottery_type not in name_path:
        raise ValueError(f"不支持的彩票类型：{lottery_type}，请检查输入。")
    if lottery_type == "dlt":
        hidden_dim = 128
        output_dim = {
            'red': 35,
            'blue': 12
        }
        output_seq_length = {
            'red': 5,
            'blue': 2
        }
    elif lottery_type == "ssq":
        hidden_dim = 128
        output_dim = {
            'red': 33,
            'blue': 16
        }
        output_seq_length = {
            'red': 6,
            'blue': 1
        }

    model_path = os.path.join(name_path[lottery_type]['path'], name_path[lottery_type]['model_file'])
    scaler_path = os.path.join(name_path[lottery_type]['path'], name_path[lottery_type]['scaler_X_file'])

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"模型文件不存在：{model_path}")
    if not os.path.exists(scaler_path):
        raise FileNotFoundError(f"特征缩放器文件不存在：{scaler_path}")

    # 从scaler_X文件中获取input_dim
    scaler_X = joblib.load(scaler_path)
    input_dim = scaler_X.n_features_in_

    red_model, blue_model, scaler_X = load_pytorch_model(
        model_path, input_dim, hidden_dim, output_dim, output_seq_length, lottery_type
    )

    return red_model, blue_model, scaler_X

# 这些函数已移动到models/lstm_crf.py中
# 在这里保留注释以便于理解代码结构