# -*- coding: utf-8 -*-
"""
Base model interface for lottery prediction models
"""

import os
import sys
import time
import numpy as np
import pandas as pd
import pickle
import joblib
import json
import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple, Union, Callable
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# 添加项目根目录到Python路径
script_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(os.path.dirname(script_dir))
if project_dir not in sys.path:
    sys.path.append(project_dir)

from utils.device_utils import check_device_availability

# 尝试导入可选的模型库
try:
    from lightgbm import LGBMRegressor
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False

try:
    from catboost import CatBoostRegressor
    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False

# 导入期望值模型
try:
    from core.expected_value import ExpectedValueLotteryModel
    EXPECTED_VALUE_MODEL_AVAILABLE = True
except ImportError:
    EXPECTED_VALUE_MODEL_AVAILABLE = False

# 定义支持的模型类型
MODEL_TYPES = {
    'lstm_timestep': 'LSTMTimeStep',
    'random_forest': '随机森林',
    'xgboost': 'XGBoost',
    'gbdt': '梯度提升树',
    'ensemble': '集成模型'
}

# 如果可选库可用，添加到支持的模型中
if LIGHTGBM_AVAILABLE:
    MODEL_TYPES['lightgbm'] = 'LightGBM'
if CATBOOST_AVAILABLE:
    MODEL_TYPES['catboost'] = 'CatBoost'
if EXPECTED_VALUE_MODEL_AVAILABLE:
    MODEL_TYPES['expected_value'] = '期望值模型'


class BaseLotteryModel(ABC):
    """
    Base abstract class for all lottery prediction models
    """
    
    def __init__(self, lottery_type: str):
        """
        Initialize the base model
        
        Args:
            lottery_type: Type of lottery ('ssq' for 双色球, 'dlt' for 大乐透)
        """
        self.lottery_type = lottery_type
        self.is_trained = False
        
    @abstractmethod
    def train(self, data: Any, **kwargs) -> None:
        """
        Train the model with given data
        
        Args:
            data: Training data
            **kwargs: Additional training parameters
        """
        pass
        
    @abstractmethod
    def predict(self, **kwargs) -> Tuple[List[int], List[int]]:
        """
        Make predictions
        
        Args:
            **kwargs: Prediction parameters
            
        Returns:
            Tuple of (red_numbers, blue_numbers)
        """
        pass
        
    @abstractmethod
    def save_model(self, filepath: str) -> None:
        """
        Save the trained model
        
        Args:
            filepath: Path to save the model
        """
        pass
        
    @abstractmethod
    def load_model(self, filepath: str) -> None:
        """
        Load a trained model
        
        Args:
            filepath: Path to load the model from
        """
        pass
        
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get model information
        
        Returns:
            Dictionary containing model information
        """
        return {
            'lottery_type': self.lottery_type,
            'is_trained': self.is_trained,
            'model_type': self.__class__.__name__
        }


class BaseMLModel:
    """
    基础机器学习模型类，提供通用功能
    """
    
    def __init__(self, lottery_type='dlt', feature_window=10, log_callback=None, use_gpu=False):
        """
        初始化模型
        
        Args:
            lottery_type: 彩票类型，'dlt'或'ssq'
            feature_window: 特征窗口大小，使用多少期数据作为特征
            log_callback: 日志回调函数，用于将日志发送到UI
            use_gpu: 是否使用GPU训练
        """
        self.lottery_type = lottery_type
        self.feature_window = feature_window
        self.models = {}
        self.scalers = {}
        self.feature_cols = []
        self.log_callback = log_callback
        self.use_gpu = use_gpu
        
        # 初始化日志记录器
        self.logger = logging.getLogger(f"ml_models_{lottery_type}")
        self.logger.setLevel(logging.INFO)
        
        # 设置彩票参数
        if lottery_type == 'dlt':
            self.red_count = 5
            self.blue_count = 2
            self.red_range = 35
            self.blue_range = 12
        else:  # ssq
            self.red_count = 6
            self.blue_count = 1
            self.red_range = 33
            self.blue_range = 16
            
        # 设置模型保存目录
        self.models_dir = os.path.join(project_dir, 'models', lottery_type)
        os.makedirs(self.models_dir, exist_ok=True)
    
    def log(self, message):
        """记录日志并发送到UI（如果有回调）"""
        self.logger.info(message)
        if self.log_callback:
            self.log_callback(message)
    
    @staticmethod
    def process_multidim_prediction(raw_preds):
        """
        处理多维预测结果，选择概率最高的类别
        
        Args:
            raw_preds: 原始预测结果，可能是概率分布或类别索引
            
        Returns:
            处理后的预测结果，通常是类别索引列表
        """
        # 如果是一维数组，直接返回
        if len(raw_preds.shape) == 1:
            return [np.argmax(raw_preds)]
        
        # 如果是二维数组（样本数 x 类别数），为每个样本选择概率最高的类别
        if len(raw_preds.shape) == 2:
            # 添加随机性：有20%的概率不选择最高概率的类别，而是从前3个最高概率中随机选择
            if np.random.random() < 0.2 and raw_preds.shape[1] > 2:
                # 获取每个样本的前3个最高概率的索引
                top_indices = np.argsort(raw_preds, axis=1)[:, -3:]
                # 随机选择一个索引
                selected_indices = []
                for i in range(top_indices.shape[0]):
                    selected_indices.append(np.random.choice(top_indices[i]))
                return selected_indices
            else:
                # 直接选择最高概率的类别
                return np.argmax(raw_preds, axis=1).tolist()
        
        # 其他情况，尝试转换为列表后处理
        return [np.argmax(p) if hasattr(p, '__iter__') else p for p in raw_preds]
    
    def prepare_data(self, df, test_size=0.2):
        """准备训练数据"""
    
        self.log("准备训练数据...")
        
        # 设置特征窗口大小
        window_size = self.feature_window
        
        # 确保数据按期数排序
        df = df.sort_values('期数').reset_index(drop=True)
        
        # 提取红蓝球列名
        if self.lottery_type == 'dlt':
            red_cols = [col for col in df.columns if col.startswith('红球_')][:5]
            blue_cols = [col for col in df.columns if col.startswith('蓝球_')][:2]
        else:  # ssq
            red_cols = [col for col in df.columns if col.startswith('红球_')][:6]
            blue_cols = [col for col in df.columns if col.startswith('蓝球_')][:1]
        
        # 创建特征和标签
        X_data = []
        y_red_data = []
        y_blue_data = []
        
        # 使用滑动窗口创建序列数据
        for i in range(len(df) - window_size):
            # 特征：过去window_size期的开奖号码
            features = []
            for j in range(window_size):
                row_features = []
                for col in red_cols + blue_cols:
                    row_features.append(df.iloc[i + j][col])
                features.append(row_features)
            
            # 标签：下一期的红球和蓝球号码（转换为0-based索引）
            red_labels = []
            blue_labels = []
            for col in red_cols:
                # 减1转换为0-based索引，并确保在有效范围内
                value = df.iloc[i + window_size][col] - 1
                # 确保红球值在有效范围内 [0, red_range-1]
                value = max(0, min(value, self.red_range - 1))
                red_labels.append(value)
            for col in blue_cols:
                # 获取原始值
                original_value = df.iloc[i + window_size][col]
                # 减1转换为0-based索引
                value = original_value - 1
                
                # 检查蓝球值是否在有效范围内 [0, blue_range-1]
                if value < 0 or value >= self.blue_range:
                    self.log(f"警告: 蓝球原始值{original_value}(索引{value})超出范围[1-{self.blue_range}]，已调整为有效范围")
                    # 修正到有效范围
                    value = max(0, min(value, self.blue_range - 1))
                    self.log(f"  - 已调整为: {value} (原始值对应: {value+1})")
                
                blue_labels.append(value)
            
            X_data.append(features)
            y_red_data.append(red_labels)
            y_blue_data.append(blue_labels)
        
        # 转换为NumPy数组
        X = np.array(X_data)
        y_red = np.array(y_red_data, dtype=int)
        y_blue = np.array(y_blue_data, dtype=int)
        
        # 检查数据是否为空
        if len(X_data) == 0 or len(y_red_data) == 0 or len(y_blue_data) == 0:
            self.log(f"错误: 生成的训练数据为空。请检查数据集大小({len(df)}行)是否小于特征窗口大小({window_size})。")
            raise ValueError(f"训练数据为空，无法继续训练。请确保数据集大小大于特征窗口大小({window_size})。")
        
        # 验证标签范围
        if len(y_red) > 0:
            self.log(f"红球标签范围: {np.min(y_red)} - {np.max(y_red)}, 预期范围: 0 - {self.red_range-1}")
        else:
            self.log("警告: 红球标签数组为空，无法计算范围")
            
        if len(y_blue) > 0:
            self.log(f"蓝球标签范围: {np.min(y_blue)} - {np.max(y_blue)}, 预期范围: 0 - {self.blue_range-1}")
        else:
            self.log("警告: 蓝球标签数组为空，无法计算范围")
        
        # 检查是否有超出范围的值
        if len(y_red) > 0:
            red_out_of_range = (y_red < 0) | (y_red >= self.red_range)
            if np.any(red_out_of_range):
                self.log(f"警告: 发现{np.sum(red_out_of_range)}个超出范围的红球标签")
        
        if len(y_blue) > 0:
            blue_out_of_range = (y_blue < 0) | (y_blue >= self.blue_range)
            if np.any(blue_out_of_range):
                self.log(f"警告: 发现{np.sum(blue_out_of_range)}个超出范围的蓝球标签")
        
        # 重塑特征以适合传统ML模型
        X_reshaped = X.reshape(X.shape[0], -1)
        
        # 标准化特征
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_reshaped)
        
        # 保存缩放器
        self.scalers['X'] = scaler
        
        # 分割训练集和测试集
        # 不使用分层抽样，因为可能存在某些类别样本数量不足的情况
        if test_size == 0 or len(X_scaled) == 1:
            # 如果test_size为0或只有一个样本，不进行分割
            X_train, X_test = X_scaled, np.array([])
            y_red_train, y_red_test = y_red, np.array([])
            y_blue_train, y_blue_test = y_blue, np.array([])
        else:
            # 检查是否有足够的样本进行分割
            if len(X_scaled) * test_size < 1:
                self.log(f"警告: 数据量({len(X_scaled)})太少，无法按test_size={test_size}进行分割，将不进行测试集分割")
                X_train, X_test = X_scaled, np.array([])
                y_red_train, y_red_test = y_red, np.array([])
                y_blue_train, y_blue_test = y_blue, np.array([])
            else:
                X_train, X_test, y_red_train, y_red_test, y_blue_train, y_blue_test = train_test_split(
                    X_scaled, y_red, y_blue, test_size=test_size, random_state=42, stratify=None
                )
        
        # 对于红球和蓝球，我们需要将多标签转换为单标签
        # 例如，对于红球，我们有5个标签，每个标签表示一个位置的号码
        # 我们需要将其转换为5个独立的分类问题
        
        # 红球
        if len(y_red_train) > 0 and y_red_train.shape[1] > 1:
            # 多个红球位置，需要分别处理每个位置
            red_train_data = []
            red_test_data = []
            for i in range(y_red_train.shape[1]):
                red_train_data.append(y_red_train[:, i].astype(np.int32))
                if len(y_red_test) > 0:
                    red_test_data.append(y_red_test[:, i].astype(np.int32))
                else:
                    red_test_data.append(np.array([], dtype=np.int32))
        else:
            # 只有一个红球位置或空数据
            if len(y_red_train) > 0:
                red_train_data = [y_red_train.flatten().astype(np.int32)]
            else:
                red_train_data = [np.array([], dtype=np.int32)]
            
            if len(y_red_test) > 0:
                red_test_data = [y_red_test.flatten().astype(np.int32)]
            else:
                red_test_data = [np.array([], dtype=np.int32)]
        
        # 蓝球
        if len(y_blue_train) > 0 and y_blue_train.shape[1] > 1:
            # 多个蓝球位置，需要分别处理每个位置
            blue_train_data = []
            blue_test_data = []
            for i in range(y_blue_train.shape[1]):
                blue_train_data.append(y_blue_train[:, i].astype(np.int32))
                if len(y_blue_test) > 0:
                    blue_test_data.append(y_blue_test[:, i].astype(np.int32))
                else:
                    blue_test_data.append(np.array([], dtype=np.int32))
        else:
            # 只有一个蓝球位置或空数据
            if len(y_blue_train) > 0:
                blue_train_data = [y_blue_train.flatten().astype(np.int32)]
            else:
                blue_train_data = [np.array([], dtype=np.int32)]
            
            if len(y_blue_test) > 0:
                blue_test_data = [y_blue_test.flatten().astype(np.int32)]
            else:
                blue_test_data = [np.array([], dtype=np.int32)]
        
        return X_train, X_test, red_train_data, red_test_data, blue_train_data, blue_test_data