#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
示例脚本：使用LSTMTimeStepModel进行彩票预测
"""

import os
import sys
import torch
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# 添加项目根目录到Python路径
script_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(script_dir)
if project_dir not in sys.path:
    sys.path.append(project_dir)

from src.models import LSTMTimeStepModel
from src.utils.device_utils import check_device_availability
from src.core.prediction import process_predictions, randomize_numbers


def load_data(lottery_type='ssq'):
    """
    加载彩票历史数据
    
    Args:
        lottery_type: 彩票类型，'ssq'或'dlt'
        
    Returns:
        历史数据DataFrame
    """
    data_path = os.path.join(project_dir, 'data', lottery_type, f'{lottery_type}_history.csv')
    return pd.read_csv(data_path)


def prepare_sequence_data(data, window_size=10, test_size=0.2):
    """
    准备序列数据用于LSTM模型训练
    
    Args:
        data: 历史数据DataFrame
        window_size: 滑动窗口大小，使用多少期数据作为特征
        test_size: 测试集比例
        
    Returns:
        训练集和测试集的特征和标签
    """
    # 提取号码数据
    if 'red_1' in data.columns:
        # 双色球格式
        red_cols = [f'red_{i}' for i in range(1, 7)]
        blue_cols = ['blue']
    else:
        # 大乐透格式
        red_cols = [f'front_{i}' for i in range(1, 6)]
        blue_cols = [f'back_{i}' for i in range(1, 3)]
    
    # 合并红蓝球数据
    ball_cols = red_cols + blue_cols
    ball_data = data[ball_cols].values
    
    # 创建序列数据
    X, y = [], []
    for i in range(len(ball_data) - window_size):
        X.append(ball_data[i:i+window_size])
        y.append(ball_data[i+window_size])
    
    X = np.array(X)
    y = np.array(y)
    
    # 标准化特征
    scaler = StandardScaler()
    n_samples, n_timesteps, n_features = X.shape
    X_reshaped = X.reshape(n_samples * n_timesteps, n_features)
    X_scaled = scaler.fit_transform(X_reshaped).reshape(n_samples, n_timesteps, n_features)
    
    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=test_size, shuffle=False)
    
    return X_train, X_test, y_train, y_test, scaler


def train_lstm_timestep_model(lottery_type='ssq', epochs=50, batch_size=32):
    """
    训练LSTM TimeStep模型
    
    Args:
        lottery_type: 彩票类型，'ssq'或'dlt'
        epochs: 训练轮数
        batch_size: 批次大小
        
    Returns:
        训练好的模型
    """
    # 加载数据
    data = load_data(lottery_type)
    X_train, X_test, y_train, y_test, scaler = prepare_sequence_data(data)
    
    # 设置模型参数
    input_size = X_train.shape[2]  # 特征维度
    hidden_size = 64
    num_layers = 2
    
    # 设置输出类别数
    if lottery_type == 'ssq':
        num_classes = 33  # 双色球红球最大号码
    else:  # dlt
        num_classes = 35  # 大乐透前区最大号码
    
    # 检查设备可用性
    device = check_device_availability()
    
    # 创建模型
    model = LSTMTimeStepModel(
        input_size=input_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        num_classes=num_classes,
        lottery_type=lottery_type,
        dropout=0.5,
        bidirectional=True,
        use_time_attention=True,
        use_residual=True
    ).to(device)
    
    # 定义损失函数和优化器
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # 转换数据为PyTorch张量
    X_train_tensor = torch.FloatTensor(X_train).to(device)
    y_train_tensor = torch.LongTensor(y_train).to(device)
    X_test_tensor = torch.FloatTensor(X_test).to(device)
    y_test_tensor = torch.LongTensor(y_test).to(device)
    
    # 训练模型
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        
        # 批次训练
        for i in range(0, len(X_train), batch_size):
            batch_X = X_train_tensor[i:i+batch_size]
            batch_y = y_train_tensor[i:i+batch_size]
            
            # 前向传播
            outputs = model(batch_X)
            
            # 计算损失
            loss = criterion(outputs, batch_y[:, 0])  # 简化：只预测第一个红球
            
            # 反向传播和优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        # 打印训练进度
        if (epoch + 1) % 5 == 0:
            print(f'Epoch [{epoch+1}/{epochs}], Loss: {total_loss/len(X_train):.4f}')
    
    # 保存模型
    model_dir = os.path.join(project_dir, 'models', lottery_type)
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, 'lstm_timestep_model.pth')
    model.save_model(model_path)
    
    print(f'模型已保存到: {model_path}')
    return model


def predict_with_lstm_timestep(model, lottery_type='ssq', num_predictions=5):
    """
    使用训练好的LSTM TimeStep模型进行预测
    
    Args:
        model: 训练好的模型
        lottery_type: 彩票类型，'ssq'或'dlt'
        num_predictions: 生成的预测数量
        
    Returns:
        预测结果列表
    """
    # 加载最近的历史数据作为输入
    data = load_data(lottery_type)
    
    # 提取最近的数据作为输入
    window_size = 10  # 与训练时保持一致
    if 'red_1' in data.columns:
        # 双色球格式
        red_cols = [f'red_{i}' for i in range(1, 7)]
        blue_cols = ['blue']
    else:
        # 大乐透格式
        red_cols = [f'front_{i}' for i in range(1, 6)]
        blue_cols = [f'back_{i}' for i in range(1, 3)]
    
    # 合并红蓝球数据
    ball_cols = red_cols + blue_cols
    recent_data = data[ball_cols].values[-window_size:]
    
    # 标准化数据
    scaler = StandardScaler()
    recent_data_scaled = scaler.fit_transform(recent_data)
    
    # 转换为PyTorch张量
    device = check_device_availability()
    input_tensor = torch.FloatTensor(recent_data_scaled).unsqueeze(0).to(device)  # 添加批次维度
    
    # 进行预测
    model.eval()
    predictions = []
    
    with torch.no_grad():
        for _ in range(num_predictions):
            # 获取模型输出概率分布
            output_probs = model(input_tensor)
            
            # 根据概率分布采样预测结果
            if lottery_type == 'ssq':
                # 双色球：6个红球，1个蓝球
                red_preds = torch.multinomial(output_probs, 6, replacement=False).cpu().numpy()[0]
                blue_preds = [np.random.randint(0, 16)]  # 简化：随机生成蓝球
            else:  # dlt
                # 大乐透：5个红球，2个蓝球
                red_preds = torch.multinomial(output_probs, 5, replacement=False).cpu().numpy()[0]
                blue_preds = [np.random.randint(0, 12) for _ in range(2)]  # 简化：随机生成蓝球
            
            # 处理预测结果
            prediction = process_predictions(red_preds, blue_preds, lottery_type)
            
            # 增加随机性
            prediction = randomize_numbers(prediction, lottery_type)
            
            predictions.append(prediction)
    
    return predictions


def main():
    # 设置彩票类型
    lottery_type = 'ssq'  # 或 'dlt'
    
    # 训练模型
    print(f'开始训练{lottery_type}的LSTM TimeStep模型...')
    model = train_lstm_timestep_model(lottery_type, epochs=30, batch_size=32)
    
    # 使用模型进行预测
    print('\n生成预测结果:')
    predictions = predict_with_lstm_timestep(model, lottery_type, num_predictions=5)
    
    # 打印预测结果
    for i, pred in enumerate(predictions, 1):
        if lottery_type == 'ssq':
            red_nums = pred[:6]
            blue_nums = pred[6:]
            print(f'预测 {i}: 红球: {red_nums}, 蓝球: {blue_nums}')
        else:  # dlt
            front_nums = pred[:5]
            back_nums = pred[5:]
            print(f'预测 {i}: 前区: {front_nums}, 后区: {back_nums}')


if __name__ == '__main__':
    main()