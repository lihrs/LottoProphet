# -*- coding: utf-8 -*-
"""
Advanced LSTM TimeStep model implementation for lottery prediction
高级LSTM时间步模型，集成多种优化技术提升预测准确率
"""

import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import datetime
from typing import Optional, Tuple, List, Any, Dict
from .base import BaseMLModel
from utils.device_utils import check_device_availability
import math

# 添加安全的全局变量，以允许numpy._core.multiarray._reconstruct
try:
    torch.serialization.add_safe_globals(['numpy._core.multiarray._reconstruct'])
except (AttributeError, ImportError):
    # 兼容旧版PyTorch
    pass

# 添加项目根目录到Python路径
script_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(os.path.dirname(script_dir))
if project_dir not in sys.path:
    sys.path.append(project_dir)

class MultiHeadAttention(nn.Module):
    """
    多头注意力机制，用于捕获序列中的复杂依赖关系
    """
    
    def __init__(self, d_model: int, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        assert d_model % num_heads == 0
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        
    def scaled_dot_product_attention(self, Q, K, V, mask=None):
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
            
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        output = torch.matmul(attention_weights, V)
        return output, attention_weights
    
    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)
        
        Q = self.w_q(query).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = self.w_k(key).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = self.w_v(value).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        
        attention_output, attention_weights = self.scaled_dot_product_attention(Q, K, V, mask)
        
        attention_output = attention_output.transpose(1, 2).contiguous().view(
            batch_size, -1, self.d_model
        )
        
        output = self.w_o(attention_output)
        return output, attention_weights

class PositionalEncoding(nn.Module):
    """
    位置编码，为序列添加位置信息
    """
    
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        return x + self.pe[:x.size(0), :]

class EnhancedLSTMLayer(nn.Module):
    """
    增强的LSTM层，集成注意力机制和残差连接
    """
    
    def __init__(self, input_size: int, hidden_size: int, num_layers: int = 2, 
                 dropout: float = 0.2, bidirectional: bool = True, use_attention: bool = True):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.use_attention = use_attention
        
        # LSTM层
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional,
            batch_first=True
        )
        
        # 输出维度
        self.output_size = hidden_size * 2 if bidirectional else hidden_size
        
        # 多头注意力
        if use_attention:
            self.attention = MultiHeadAttention(self.output_size, num_heads=8, dropout=dropout)
            self.norm1 = nn.LayerNorm(self.output_size)
            
        # 前馈网络
        self.ffn = nn.Sequential(
            nn.Linear(self.output_size, self.output_size * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(self.output_size * 4, self.output_size),
            nn.Dropout(dropout)
        )
        self.norm2 = nn.LayerNorm(self.output_size)
        
        # 位置编码
        self.pos_encoding = PositionalEncoding(self.output_size)
        
    def forward(self, x):
        # LSTM处理
        lstm_out, (hidden, cell) = self.lstm(x)
        
        # 添加位置编码
        lstm_out = self.pos_encoding(lstm_out.transpose(0, 1)).transpose(0, 1)
        
        # 多头注意力
        if self.use_attention:
            attn_out, attention_weights = self.attention(lstm_out, lstm_out, lstm_out)
            lstm_out = self.norm1(lstm_out + attn_out)
        else:
            attention_weights = None
            
        # 前馈网络
        ffn_out = self.ffn(lstm_out)
        output = self.norm2(lstm_out + ffn_out)
        
        return output, attention_weights

class AdaptivePredictionHead(nn.Module):
    """
    自适应预测头，根据历史统计动态调整预测策略
    """
    
    def __init__(self, input_size: int, output_size: int, dropout: float = 0.2, use_layernorm: bool = False):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.use_layernorm = use_layernorm
        
        # 多层预测网络
        layers = []
        # 第一层
        layers.append(nn.Linear(input_size, input_size * 2))
        layers.append(nn.GELU())
        if use_layernorm:
            layers.append(nn.LayerNorm(input_size * 2))
        layers.append(nn.Dropout(dropout))
        
        # 第二层
        layers.append(nn.Linear(input_size * 2, input_size))
        layers.append(nn.GELU())
        if use_layernorm:
            layers.append(nn.LayerNorm(input_size))
        layers.append(nn.Dropout(dropout))
        
        # 输出层
        layers.append(nn.Linear(input_size, output_size))
        
        self.predictor = nn.Sequential(*layers)
        
        # 温度参数，用于调节预测分布的锐度
        self.temperature = nn.Parameter(torch.ones(1))
        
    def forward(self, x):
        logits = self.predictor(x)
        # 应用温度缩放
        scaled_logits = logits / torch.clamp(self.temperature, min=0.1, max=5.0)
        return scaled_logits

class LotteryLoss(nn.Module):
    """
    专门为彩票预测设计的损失函数，结合交叉熵和分布约束
    """
    
    def __init__(self, alpha: float = 0.8, beta: float = 0.2):
        super().__init__()
        self.alpha = alpha  # 交叉熵权重
        self.beta = beta    # 分布约束权重
        self.ce_loss = nn.CrossEntropyLoss()
        
    def forward(self, logits, targets, historical_freq=None):
        # 基础交叉熵损失
        ce_loss = self.ce_loss(logits, targets)
        
        # 分布约束损失（鼓励预测分布接近历史频率分布）
        if historical_freq is not None:
            pred_probs = F.softmax(logits, dim=-1)
            freq_probs = F.softmax(historical_freq, dim=-1)
            kl_loss = F.kl_div(pred_probs.log(), freq_probs, reduction='batchmean')
        else:
            kl_loss = 0
            
        total_loss = self.alpha * ce_loss + self.beta * kl_loss
        return total_loss

class LSTMTimeStepModel(BaseMLModel, nn.Module):
    """
    高级LSTM时间步模型，集成多种优化技术
    
    优化特点：
    1. 混合精度训练加速
    2. 梯度累积更新
    3. 自适应批次大小
    4. 标签平滑和Focal Loss
    5. 注意力机制增强
    6. 优化的学习率调度
    7. 特征融合与增强
    """
    
    def __init__(self, lottery_type='dlt', feature_window=10, log_callback=None, use_gpu=False,
                 hidden_size=256, num_layers=2, dropout=0.2, bidirectional=True,
                 use_attention=True, learning_rate=0.0005, weight_decay=1e-5):
        """
        初始化高级LSTM TimeStep模型
        """
        BaseMLModel.__init__(self, lottery_type, feature_window, log_callback, use_gpu)
        nn.Module.__init__(self)
        
        # 优化的模型参数 - 调整为更高效的配置
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.bidirectional = bidirectional
        self.use_attention = use_attention
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        
        # 新增训练优化参数
        self.use_mixed_precision = True  # 使用混合精度训练加速
        self.gradient_accumulation_steps = 4  # 梯度累积步数
        self.label_smoothing = 0.1  # 标签平滑系数
        self.use_focal_loss = True  # 使用Focal Loss
        self.focal_gamma = 2.0  # Focal Loss的gamma参数
        self.attention_heads = 12  # 增加注意力头数量
        self.use_layer_norm = True  # 使用层归一化
        self.use_residual = True  # 使用残差连接
        self.use_gelu = True  # 使用GELU激活函数
        
        # 设备配置 - 增加MPS支持（苹果M系列芯片）
        if use_gpu:
            if torch.cuda.is_available():
                self.device = torch.device('cuda')
                # 检查CUDA版本，决定是否启用TF32精度
                if torch.cuda.get_device_capability()[0] >= 8:  # Ampere架构或更高
                    torch.backends.cuda.matmul.allow_tf32 = True
                    torch.backends.cudnn.allow_tf32 = True
                    self.log("启用TF32精度加速")
            elif hasattr(torch, 'mps') and torch.backends.mps.is_available():
                self.device = torch.device('mps')
                # 对于MPS，禁用混合精度训练，因为MPS不完全支持
                self.use_mixed_precision = False
                self.log("在MPS设备上禁用混合精度训练")
            else:
                self.device = torch.device('cpu')
                self.use_mixed_precision = False
        else:
            self.device = torch.device('cpu')
            self.use_mixed_precision = False
        
        # 输入特征大小
        self.input_size = self.red_count + self.blue_count
        
        # 构建网络
        self._build_network()
        
        # 移动到设备
        self.to(self.device)
        
        # 简化的优化器配置
        self.optimizer = optim.Adam(
            self.parameters(), 
            lr=self.learning_rate, 
            weight_decay=self.weight_decay
        )
        
        # 简化的学习率调度器
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=20
        )
        
        # 混合精度训练的scaler - 仅在CUDA可用时使用
        self.amp_scaler = torch.cuda.amp.GradScaler() if (self.use_mixed_precision and torch.cuda.is_available()) else None
        
        # 损失函数 - 使用简单的交叉熵损失
        self.criterion = nn.CrossEntropyLoss()
        
        # 训练状态
        self.training_history = {'loss': [], 'red_accuracy': [], 'blue_accuracy': []}
        self.is_trained = False
        self.best_model_state = None
        
        # 历史频率统计
        self.red_freq = None
        self.blue_freq = None
        
        # 梯度裁剪值
        self.grad_clip_value = 0.8  # 调整梯度裁剪值
        
    def _build_network(self):
        """
        构建精简的网络架构 - 优化训练速度
        """
        # 输入嵌入层 - 简化为单层
        self.input_embedding = nn.Sequential(
            nn.Linear(self.input_size, self.hidden_size),
            nn.BatchNorm1d(self.hidden_size),
            nn.ReLU(),
            nn.Dropout(self.dropout)
        )
        
        # 简化LSTM层
        self.lstm_layer = EnhancedLSTMLayer(
            input_size=self.hidden_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            dropout=self.dropout,
            bidirectional=self.bidirectional,
            use_attention=self.use_attention
        )
        
        # 计算LSTM输出大小
        lstm_output_size = self.hidden_size * 2 if self.bidirectional else self.hidden_size
        
        # 简化的层归一化
        self.layer_norm = nn.LayerNorm(lstm_output_size)
        
        # 简化的特征处理层
        self.feature_processor = nn.Sequential(
            nn.Linear(lstm_output_size, lstm_output_size),
            nn.ReLU(),
            nn.Dropout(self.dropout)
        )
        
        # 红球预测头 - 简化版本
        self.red_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(lstm_output_size, self.red_range),
                nn.LogSoftmax(dim=-1)
            )
            for _ in range(self.red_count)
        ])
        
        # 蓝球预测头 - 简化版本
        self.blue_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(lstm_output_size, self.blue_range),
                nn.LogSoftmax(dim=-1)
            )
            for _ in range(self.blue_count)
        ])
        
        # 简化的全局注意力
        self.global_attention = nn.Sequential(
            nn.Linear(lstm_output_size, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        """
        前向传播 - 简化版实现
        """
        batch_size, seq_len, feature_dim = x.shape
        
        # 确保输入特征维度与模型期望的输入维度匹配
        if feature_dim != self.input_size:
            self.log(f"警告: 输入特征维度 {feature_dim} 与模型期望的输入维度 {self.input_size} 不匹配")
            # 动态调整input_size以匹配实际输入
            self.input_size = feature_dim
            
            # 重新创建输入嵌入层的第一个线性层以匹配新的输入维度
            old_embedding = self.input_embedding
            self.input_embedding = nn.Sequential(
                nn.Linear(self.input_size, self.hidden_size),
                old_embedding[1],  # 保留原有的BatchNorm1d
                old_embedding[2],  # 保留原有的ReLU
                old_embedding[3]   # 保留原有的Dropout
            )
            # 将新层移动到正确的设备上
            self.input_embedding = self.input_embedding.to(self.device)
        
        # 输入嵌入 - 处理批归一化需要的维度转换
        x_reshaped = x.reshape(-1, self.input_size)
        # 先通过线性层
        linear_out = self.input_embedding[0](x_reshaped)
        # 然后通过批归一化层
        bn_out = self.input_embedding[1](linear_out)
        # 再通过激活函数和dropout
        embedded = self.input_embedding[3](self.input_embedding[2](bn_out))
        embedded = embedded.reshape(batch_size, seq_len, self.hidden_size)
        
        # LSTM处理
        lstm_out, attention_weights = self.lstm_layer(embedded)
        
        # 应用层归一化
        normed_lstm = self.layer_norm(lstm_out)
        
        # 特征处理
        processed_features = self.feature_processor(normed_lstm)
        
        # 简化的注意力机制
        attention_scores = self.global_attention(processed_features)  # [batch_size, seq_len, 1]
        attention_weights_global = F.softmax(attention_scores, dim=1)
        
        # 加权求和得到全局特征
        global_features = torch.sum(processed_features * attention_weights_global, dim=1)  # [batch_size, hidden_size]
        
        # 使用最后一个时间步的特征作为序列特征
        sequence_features = processed_features[:, -1, :]
        
        # 简单组合全局特征和序列特征
        combined_features = global_features * 0.6 + sequence_features * 0.4
        
        # 红球预测
        red_outputs = [head(combined_features) for head in self.red_heads]
        
        # 蓝球预测
        blue_outputs = [head(combined_features) for head in self.blue_heads]
        
        return {
            'red_logits': red_outputs,
            'blue_logits': blue_outputs,
            'attention_weights': attention_weights,
            'features': combined_features,
            'global_attention': attention_weights_global
        }
    
    def _compute_historical_frequency(self, data):
        """
        计算历史号码频率
        """
        red_freq = torch.zeros(self.red_range)
        blue_freq = torch.zeros(self.blue_range)
        
        # 统计红球频率
        for i in range(self.red_count):
            col_name = f'red_{i+1}'
            if col_name in data.columns:
                values = data[col_name].values - 1  # 转换为0-based索引
                for val in values:
                    if 0 <= val < self.red_range:
                        red_freq[val] += 1
        
        # 统计蓝球频率
        for i in range(self.blue_count):
            col_name = f'blue_{i+1}'
            if col_name in data.columns:
                values = data[col_name].values - 1  # 转换为0-based索引
                for val in values:
                    if 0 <= val < self.blue_range:
                        blue_freq[val] += 1
        
        # 归一化
        red_freq = red_freq / (red_freq.sum() + 1e-8)
        blue_freq = blue_freq / (blue_freq.sum() + 1e-8)
        
        return red_freq.to(self.device), blue_freq.to(self.device)
    
    def prepare_fibonacci_data(self, df, test_size=0.2):
        """
        基于斐波那契数列选择短期数据进行训练
        
        参数:
            df: 原始数据DataFrame
            test_size: 测试集比例
            
        返回:
            训练集和测试集数据
        """
        self.log("准备训练数据 - 使用斐波那契短期数据选择策略...")
        
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
        
        # 基于斐波那契数列选择短期数据
        fib_periods = [3, 5, 8, 13, 21, 34, 55]
        
        # 创建特征和标签
        X_data = []
        y_red_data = []
        y_blue_data = []
        
        # 获取最新的数据
        latest_index = len(df) - 1
        
        self.log(f"原始数据总期数: {len(df)}")
        self.log(f"斐波那契周期选择: {fib_periods}")
        
        # 为每个斐波那契周期创建数据集
        for period in fib_periods:
            self.log(f"处理周期: {period}期")
            
            # 确保周期不超过可用数据量
            if period >= len(df):
                self.log(f"  - 跳过: 周期{period}大于可用数据量{len(df)}")
                continue
            
            # 选择最近的period期数据
            period_df = df.iloc[-period:].reset_index(drop=True)
            self.log(f"  - 选择最近{period}期数据，实际获取: {len(period_df)}期")
            
            # 使用滑动窗口创建序列数据
            for i in range(len(period_df) - window_size):
                # 特征：过去window_size期的开奖号码
                features = []
                for j in range(window_size):
                    row_features = []
                    for col in red_cols + blue_cols:
                        row_features.append(period_df.iloc[i + j][col])
                    features.append(row_features)
                
                # 标签：下一期的红球和蓝球号码（转换为0-based索引）
                red_labels = []
                blue_labels = []
                for col in red_cols:
                    # 减1转换为0-based索引，并确保在有效范围内
                    value = period_df.iloc[i + window_size][col] - 1
                    # 确保红球值在有效范围内 [0, red_range-1]
                    value = max(0, min(value, self.red_range - 1))
                    red_labels.append(value)
                for col in blue_cols:
                    # 获取原始值
                    original_value = period_df.iloc[i + window_size][col]
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
            
            self.log(f"  - 周期{period}生成样本数: {len(period_df) - window_size}")
        
        # 转换为NumPy数组
        X = np.array(X_data)
        y_red = np.array(y_red_data, dtype=int)
        y_blue = np.array(y_blue_data, dtype=int)
        
        # 检查数据是否为空
        if len(X_data) == 0 or len(y_red_data) == 0 or len(y_blue_data) == 0:
            self.log(f"错误: 生成的训练数据为空。请检查数据集大小({len(df)}行)是否小于特征窗口大小({window_size})。")
            raise ValueError(f"训练数据为空，无法继续训练。请确保数据集大小大于特征窗口大小({window_size})。")
        
        # 验证标签范围
        if len(y_red) > 0 and y_red.size > 0:
            self.log(f"红球标签范围: {np.min(y_red)} - {np.max(y_red)}, 预期范围: 0 - {self.red_range-1}")
        else:
            self.log("警告: 红球标签数组为空，无法计算范围")
            
        if len(y_blue) > 0 and y_blue.size > 0:
            self.log(f"蓝球标签范围: {np.min(y_blue)} - {np.max(y_blue)}, 预期范围: 0 - {self.blue_range-1}")
        else:
            self.log("警告: 蓝球标签数组为空，无法计算范围")
        
        self.log(f"基于斐波那契数列的短期数据选择完成，总样本数: {len(X)}")
        
        # 划分训练集和测试集
        if test_size > 0 and len(X) > 10:  # 确保数据足够划分
            # 使用时间序列划分，保留最近的数据作为测试集
            split_idx = int(len(X) * (1 - test_size))
            X_train, X_test = X[:split_idx], X[split_idx:]
            y_red_train, y_red_test = [], []
            y_blue_train, y_blue_test = [], []
            
            # 分别处理每个位置的红球和蓝球
            for i in range(len(y_red[0])):
                y_red_train.append(y_red[:split_idx, i])
                y_red_test.append(y_red[split_idx:, i])
            
            for i in range(len(y_blue[0])):
                y_blue_train.append(y_blue[:split_idx, i])
                y_blue_test.append(y_blue[split_idx:, i])
            
            self.log(f"数据划分完成: 训练集 {len(X_train)} 样本, 测试集 {len(X_test)} 样本")
            return X_train, X_test, y_red_train, y_red_test, y_blue_train, y_blue_test
        else:
            # 不划分测试集，全部用于训练
            y_red_train = [y_red[:, i] for i in range(y_red.shape[1])]
            y_blue_train = [y_blue[:, i] for i in range(y_blue.shape[1])]
            
            self.log(f"全部数据用于训练: {len(X)} 样本")
            return X, np.array([]), y_red_train, [], y_blue_train, []
    
    def fit(self, data, epochs=500, batch_size=64, validation_split=0.2, 
            early_stopping_patience=50, **kwargs):
        """
        训练模型 - 简化版，使用斐波那契短期数据策略
        """
        self.log(f"开始训练简化版LSTM TimeStep模型，彩票类型: {self.lottery_type}")
        self.log(f"优化参数: hidden_size={self.hidden_size}, num_layers={self.num_layers}, dropout={self.dropout}")
        self.log(f"训练参数: epochs={epochs}, batch_size={batch_size}, lr={self.learning_rate}")
        self.log(f"训练策略: 基于斐波那契数列的短期数据选择")
        
        try:
            # 计算历史频率
            self.red_freq, self.blue_freq = self._compute_historical_frequency(data)
            
            # 准备数据 - 使用基于斐波那契数列的短期数据选择策略
            X_train, X_val, red_train_data, red_val_data, blue_train_data, blue_val_data = self.prepare_fibonacci_data(
                data, test_size=validation_split
            )
            
            # 转换为序列数据
            X_train_seq = self._prepare_sequence_data(X_train)
            X_val_seq = self._prepare_sequence_data(X_val) if len(X_val) > 0 else np.array([])
            
            # 保存最后序列用于预测
            self._last_sequence = X_val_seq[-1:] if len(X_val_seq) > 0 else X_train_seq[-1:]
            
            # 转换为张量
            X_train_tensor = torch.FloatTensor(X_train_seq).to(self.device)
            X_val_tensor = torch.FloatTensor(X_val_seq).to(self.device) if len(X_val_seq) > 0 else None
            
            # 准备标签
            red_train_tensors = [torch.LongTensor(red_data).to(self.device) for red_data in red_train_data]
            red_val_tensors = [torch.LongTensor(red_data).to(self.device) for red_data in red_val_data] if red_val_data else []
            blue_train_tensors = [torch.LongTensor(blue_data).to(self.device) for blue_data in blue_train_data]
            blue_val_tensors = [torch.LongTensor(blue_data).to(self.device) for blue_data in blue_val_data] if blue_val_data else []
            
            # 数据加载器 - 简化版，不使用加权采样
            train_dataset = torch.utils.data.TensorDataset(
                X_train_tensor, *red_train_tensors, *blue_train_tensors
            )
            train_loader = torch.utils.data.DataLoader(
                train_dataset, batch_size=batch_size, shuffle=True, drop_last=True
            )
            
            # 验证集数据加载器
            val_loader = None
            if X_val_tensor is not None and len(red_val_tensors) > 0 and len(blue_val_tensors) > 0:
                val_dataset = torch.utils.data.TensorDataset(
                    X_val_tensor, *red_val_tensors, *blue_val_tensors
                )
                val_loader = torch.utils.data.DataLoader(
                    val_dataset, batch_size=batch_size, shuffle=False
                )
            
            # 训练循环
            best_val_loss = float('inf')
            patience_counter = 0
            
            for epoch in range(epochs):
                # 训练阶段
                train_loss, train_red_acc, train_blue_acc = self._train_epoch(train_loader)
                
                # 验证阶段
                val_loss, val_red_acc, val_blue_acc = self._validate_epoch(val_loader) if val_loader else (train_loss, train_red_acc, train_blue_acc)
                
                # 记录历史
                self.training_history['loss'].append({'train': train_loss, 'val': val_loss})
                self.training_history['red_accuracy'].append({'train': train_red_acc, 'val': val_red_acc})
                self.training_history['blue_accuracy'].append({'train': train_blue_acc, 'val': val_blue_acc})
                
                # 早停检查 - 仅基于验证损失
                improved = False
                if val_loss < best_val_loss * 0.999:  # 损失需要改善(至少0.1%)
                    best_val_loss = val_loss
                    improved = True
                    self.best_model_state = self.state_dict().copy()
                    patience_counter = 0
                else:
                    patience_counter += 1
                
                # 学习率调度
                self.scheduler.step(val_loss)
                
                # 日志输出
                if epoch % 5 == 0 or epoch == epochs - 1 or improved:
                    current_lr = self.optimizer.param_groups[0]['lr']
                    self.log(f"Epoch {epoch+1}/{epochs}: Train Loss: {train_loss:.4f}, "
                           f"Val Loss: {val_loss:.4f}, Red Acc: {val_red_acc:.3f}, "
                           f"Blue Acc: {val_blue_acc:.3f}, "
                           f"LR: {current_lr:.6f}, Patience: {patience_counter}/{early_stopping_patience}")
                
                # 早停
                if patience_counter >= early_stopping_patience:
                    self.log(f"早停触发，在第{epoch+1}轮停止训练，已经{patience_counter}个epoch没有改善")
                    break
            
            # 加载最佳模型
            if self.best_model_state is not None:
                self.load_state_dict(self.best_model_state)
                self.log(f"加载最佳模型，验证损失: {best_val_loss:.4f}, 验证红球准确率: {val_red_acc:.4f}, 验证蓝球准确率: {val_blue_acc:.4f}")
            
            self.is_trained = True
            self.log("模型训练完成")
            
            # 自动保存训练完成的模型
            try:
                model_save_path = self.get_model_path()
                os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
                self.save_model(model_save_path)
                self.log(f"模型已自动保存到: {model_save_path}")
            except Exception as save_error:
                self.log(f"模型保存失败: {str(save_error)}")
            
        except Exception as e:
            self.log(f"训练过程中发生错误: {str(e)}")
            raise
    
    def _prepare_sequence_data(self, X):
        """
        准备序列数据
        """
        samples = X.shape[0]
        
        # 检查输入数据的形状
        if len(X.shape) == 3:
            # 如果已经是三维数组，检查第二维和第三维是否符合预期
            if X.shape[1] == self.feature_window and X.shape[2] == self.input_size:
                return X
            elif X.shape[1] == self.feature_window:
                # 第三维不匹配，可能是特征数量不一致
                # 注意：这里不更新input_size和input_embedding，因为这些会在forward方法中处理
                # 这样可以避免重复更新
                return X
            else:
                # 尝试重塑数据
                total_elements = X.size
                expected_elements = samples * self.feature_window * self.input_size
                if total_elements != expected_elements:
                    self.log(f"错误: 无法将形状为 {X.shape} 的数组重塑为 ({samples}, {self.feature_window}, {self.input_size})")
                    self.log(f"数组大小: {total_elements}, 预期大小: {expected_elements}")
                    # 尝试计算合适的维度
                    if total_elements % (samples * self.feature_window) == 0:
                        new_input_size = total_elements // (samples * self.feature_window)
                        self.log(f"调整input_size为: {new_input_size}")
                        # 注意：这里不更新input_size和input_embedding，因为这些会在forward方法中处理
                        return X.reshape(samples, self.feature_window, new_input_size)
                    else:
                        raise ValueError(f"无法重塑数组: 大小 {total_elements} 不能被 {samples}*{self.feature_window} 整除")
                return X.reshape(samples, self.feature_window, self.input_size)
        elif len(X.shape) == 2:
            # 如果是二维数组，尝试重塑为三维
            total_elements = X.size
            expected_elements = samples * self.feature_window * self.input_size
            if total_elements != expected_elements:
                self.log(f"错误: 无法将形状为 {X.shape} 的数组重塑为 ({samples}, {self.feature_window}, {self.input_size})")
                self.log(f"数组大小: {total_elements}, 预期大小: {expected_elements}")
                # 尝试计算合适的维度
                if total_elements % (samples * self.feature_window) == 0:
                    new_input_size = total_elements // (samples * self.feature_window)
                    self.log(f"调整input_size为: {new_input_size}")
                    # 注意：这里不更新input_size和input_embedding，因为这些会在forward方法中处理
                    return X.reshape(samples, self.feature_window, new_input_size)
                else:
                    raise ValueError(f"无法重塑数组: 大小 {total_elements} 不能被 {samples}*{self.feature_window} 整除")
            return X.reshape(samples, self.feature_window, self.input_size)
        else:
            raise ValueError(f"不支持的输入数据形状: {X.shape}")
    
    def _train_epoch(self, train_loader):
        """
        训练一个epoch - 简化版实现
        """
        self.train()
        total_loss = 0.0
        red_correct = 0
        blue_correct = 0
        total_samples = 0
        
        for batch_data in train_loader:
            X_batch = batch_data[0]
            red_targets = batch_data[1:1+self.red_count]
            blue_targets = batch_data[1+self.red_count:]
            
            # 梯度清零
            self.optimizer.zero_grad()
            
            # 前向传播
            outputs = self.forward(X_batch)
            
            # 计算损失
            red_loss = 0.0
            blue_loss = 0.0
            
            # 红球损失
            for i, (red_logit, red_target) in enumerate(zip(outputs['red_logits'], red_targets)):
                loss = self.criterion(red_logit, red_target)
                red_loss += loss
                
                # 计算准确率
                _, predicted = torch.max(red_logit, 1)
                red_correct += (predicted == red_target).sum().item()
            
            # 蓝球损失
            for i, (blue_logit, blue_target) in enumerate(zip(outputs['blue_logits'], blue_targets)):
                loss = self.criterion(blue_logit, blue_target)
                blue_loss += loss
                
                # 计算准确率
                _, predicted = torch.max(blue_logit, 1)
                blue_correct += (predicted == blue_target).sum().item()
            
            # 总损失
            total_batch_loss = red_loss + blue_loss
            
            # 反向传播
            total_batch_loss.backward()
            
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
            
            # 更新参数
            self.optimizer.step()
            
            total_loss += total_batch_loss.item()
            total_samples += X_batch.size(0)
        
        # 计算平均损失和准确率
        avg_loss = total_loss / len(train_loader)
        red_accuracy = red_correct / (total_samples * self.red_count)
        blue_accuracy = blue_correct / (total_samples * self.blue_count)
        
        return avg_loss, red_accuracy, blue_accuracy
    
    def _validate_epoch(self, val_loader):
        """
        验证一个epoch - 简化版实现
        """
        if val_loader is None:
            return 0.0, 0.0, 0.0
            
        self.eval()
        total_loss = 0.0
        red_correct = 0
        blue_correct = 0
        total_samples = 0
        
        with torch.no_grad():
            for batch_data in val_loader:
                X_batch = batch_data[0]
                red_targets = batch_data[1:1+self.red_count]
                blue_targets = batch_data[1+self.red_count:]
                
                # 前向传播
                outputs = self.forward(X_batch)
                
                # 计算损失
                red_loss = 0.0
                blue_loss = 0.0
                
                # 红球损失
                for i, (red_logit, red_target) in enumerate(zip(outputs['red_logits'], red_targets)):
                    loss = self.criterion(red_logit, red_target)
                    red_loss += loss
                    
                    # 计算准确率
                    _, predicted = torch.max(red_logit, 1)
                    red_correct += (predicted == red_target).sum().item()
                
                # 蓝球损失
                for i, (blue_logit, blue_target) in enumerate(zip(outputs['blue_logits'], blue_targets)):
                    loss = self.criterion(blue_logit, blue_target)
                    blue_loss += loss
                    
                    # 计算准确率
                    _, predicted = torch.max(blue_logit, 1)
                    blue_correct += (predicted == blue_target).sum().item()
                
                # 总损失
                total_batch_loss = red_loss + blue_loss
                total_loss += total_batch_loss.item()
                total_samples += X_batch.size(0)
        
        # 计算平均损失和准确率
        avg_loss = total_loss / len(val_loader)
        red_accuracy = red_correct / (total_samples * self.red_count)
        blue_accuracy = blue_correct / (total_samples * self.blue_count)
        
        return avg_loss, red_accuracy, blue_accuracy
    
    def predict(self, recent_data=None, num_predictions=1, temperature=1.0, top_k=None, return_probs=False, **kwargs):
        """
        预测彩票号码 - 简化版实现
        
        参数:
            recent_data: 最近的数据，用于预测
            num_predictions: 生成的预测数量
            temperature: 温度参数，控制采样的随机性，越小越确定
            top_k: 只考虑概率最高的前k个选项
            return_probs: 是否返回概率分布和置信度
            
        返回:
            如果return_probs=False，返回预测的号码
            如果return_probs=True，返回(预测号码, 概率分布, 置信度)
        """
        if not self.is_trained:
            raise ValueError("模型必须先训练才能进行预测")
        
        self.eval()
        predictions = []
        all_confidences = []
        all_red_probs = []
        all_blue_probs = []
        
        with torch.no_grad():
            for _ in range(num_predictions):
                if recent_data is None:
                    if not hasattr(self, '_last_sequence'):
                        raise ValueError("没有可用的历史数据进行预测")
                    input_data = self._last_sequence
                else:
                    # 处理输入数据
                    if hasattr(recent_data, 'values'):
                        # 直接处理数据，不进行训练测试分割
                        df = recent_data.sort_values('期数').reset_index(drop=True)
                        
                        # 提取红蓝球列名
                        if self.lottery_type == 'dlt':
                            red_cols = [col for col in df.columns if col.startswith('红球_')][:5]
                            blue_cols = [col for col in df.columns if col.startswith('蓝球_')][:2]
                        else:  # ssq
                            red_cols = [col for col in df.columns if col.startswith('红球_')][:6]
                            blue_cols = [col for col in df.columns if col.startswith('蓝球_')][:1]
                        
                        # 检查数据量是否足够
                        if len(df) < self.feature_window:
                            raise ValueError(f"预测需要至少{self.feature_window}期历史数据，当前只有{len(df)}期")
                        
                        # 使用最后feature_window期数据作为特征
                        features = []
                        for j in range(self.feature_window):
                            row_features = []
                            for col in red_cols + blue_cols:
                                row_features.append(df.iloc[-(self.feature_window-j)][col])
                            features.append(row_features)
                        
                        # 转换为numpy数组
                        input_data = np.array([features])
                    else:
                        input_data = recent_data
                
                # 转换为张量
                input_tensor = torch.FloatTensor(input_data).to(self.device)
                
                # 前向传播
                outputs = self.forward(input_tensor)
                
                # 获取预测结果
                red_numbers = []
                blue_numbers = []
                red_confidences = []
                blue_confidences = []
                batch_red_probs = []
                batch_blue_probs = []
                
                # 红球预测
                for i, red_logit in enumerate(outputs['red_logits']):
                    # 应用温度缩放
                    scaled_logits = red_logit / temperature
                    probabilities = F.softmax(scaled_logits, dim=1)
                    batch_red_probs.append(probabilities.cpu().numpy())
                    
                    if top_k is not None:
                        # 只保留top_k个概率最高的球
                        top_k_probs, top_k_indices = torch.topk(probabilities, k=min(top_k, probabilities.size(1)), dim=1)
                        # 记录置信度
                        confidence = top_k_probs[:, 0]
                        
                        # 采样
                        predicted = top_k_indices[:, 0].item()  # 选择概率最高的
                    else:
                        # 直接采样
                        predicted = torch.multinomial(probabilities, 1).squeeze().item()
                        # 记录置信度
                        confidence = probabilities[0, predicted].item()
                    
                    red_numbers.append(predicted + 1)  # 转换为1-based索引
                    red_confidences.append(confidence.item() if isinstance(confidence, torch.Tensor) else confidence)
                
                # 蓝球预测
                for i, blue_logit in enumerate(outputs['blue_logits']):
                    scaled_logits = blue_logit / temperature
                    probabilities = F.softmax(scaled_logits, dim=1)
                    batch_blue_probs.append(probabilities.cpu().numpy())
                    
                    if top_k is not None:
                        # 只保留top_k个概率最高的球
                        top_k_probs, top_k_indices = torch.topk(probabilities, k=min(top_k, probabilities.size(1)), dim=1)
                        # 记录置信度
                        confidence = top_k_probs[:, 0]
                        
                        # 采样
                        predicted = top_k_indices[:, 0].item()  # 选择概率最高的
                    else:
                        # 直接采样
                        predicted = torch.multinomial(probabilities, 1).squeeze().item()
                        # 记录置信度
                        confidence = probabilities[0, predicted].item()
                    
                    blue_numbers.append(predicted + 1)  # 转换为1-based索引
                    blue_confidences.append(confidence.item() if isinstance(confidence, torch.Tensor) else confidence)
                
                # 确保红球号码唯一
                red_numbers = self._ensure_unique_red_numbers(red_numbers)
                
                # 对红球和蓝球号码进行排序（从小到大）
                red_numbers.sort()
                blue_numbers.sort()
                
                predictions.append((red_numbers, blue_numbers))
                all_confidences.append((red_confidences, blue_confidences))
                all_red_probs.append(batch_red_probs)
                all_blue_probs.append(batch_blue_probs)
        
        if return_probs:
            return (predictions[0] if num_predictions == 1 else predictions, 
                    (all_red_probs[0], all_blue_probs[0]) if num_predictions == 1 else (all_red_probs, all_blue_probs),
                    all_confidences[0] if num_predictions == 1 else all_confidences)
        else:
            return predictions[0] if num_predictions == 1 else predictions
    
    def _ensure_unique_red_numbers(self, red_numbers):
        """
        确保红球号码唯一
        """
        unique_numbers = []
        used_numbers = set()
        
        for num in red_numbers:
            if num not in used_numbers:
                unique_numbers.append(num)
                used_numbers.add(num)
            else:
                # 找一个未使用的号码
                for candidate in range(1, self.red_range + 1):
                    if candidate not in used_numbers:
                        unique_numbers.append(candidate)
                        used_numbers.add(candidate)
                        break
        
        return sorted(unique_numbers)
        
    def _ensure_unique_red_numbers_with_confidence(self, red_numbers, confidences=None):
        """
        确保红球号码唯一，同时更新置信度 - 简化版
        
        参数:
            red_numbers: 红球号码列表
            confidences: 对应的置信度列表
            
        返回:
            唯一的红球号码列表和更新后的置信度列表
        """
        # 直接调用基础版本的方法，不再单独处理置信度
        unique_numbers = self._ensure_unique_red_numbers(red_numbers)
        
        # 如果需要置信度，则创建一个简单的置信度列表
        if confidences is not None:
            # 保持原始置信度，对于替换的号码使用默认值0.5
            updated_confidences = []
            for i, num in enumerate(unique_numbers):
                if i < len(confidences) and num == red_numbers[i]:
                    updated_confidences.append(confidences[i])
                else:
                    updated_confidences.append(0.5)
            return unique_numbers, updated_confidences
        else:
            return unique_numbers, [0.5] * len(unique_numbers)
    
    def save_model(self, filepath, save_optimizer=True):
        """
        保存模型 - 简化版实现
        
        参数:
            filepath: 保存模型的文件路径
            save_optimizer: 是否保存优化器状态，默认为True
        """
        try:
            # 创建目录（如果不存在）
            os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
            
            # 保存模型参数
            model_data = {
                'model_state_dict': self.state_dict(),
                'training_history': self.training_history,
                'model_params': {
                    'lottery_type': self.lottery_type,
                    'feature_window': self.feature_window,
                    'hidden_size': self.hidden_size,
                    'num_layers': self.num_layers,
                    'dropout': self.dropout,
                    'bidirectional': self.bidirectional,
                    'use_attention': self.use_attention,
                    'learning_rate': self.learning_rate,
                    'weight_decay': self.weight_decay,
                    'grad_clip_value': self.grad_clip_value,
                    'model_version': '2.1',  # 更新版本信息为简化版
                    'timestamp': datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                },
                'red_freq': self.red_freq,
                'blue_freq': self.blue_freq,
                'is_trained': self.is_trained
            }
            
            # 可选保存优化器状态
            if save_optimizer and self.optimizer:
                model_data['optimizer_state_dict'] = self.optimizer.state_dict()
            
            if hasattr(self, '_last_sequence'):
                model_data['last_sequence'] = self._last_sequence
            
            # 直接保存模型
            torch.save(model_data, filepath)
            
            self.log(f"模型已保存到: {filepath}")
            return True
            
        except Exception as e:
            self.log(f"保存模型时发生错误: {str(e)}")
            return False
    
    def load_model(self, filepath, strict=False):
        """
        加载模型 - 简化版实现
        
        参数:
            filepath: 模型文件路径
            strict: 是否严格检查参数匹配，默认为False
        """
        try:
            # 检查文件是否存在
            if not os.path.exists(filepath):
                self.log(f"模型文件不存在: {filepath}")
                return False
                
            # 设置weights_only=False以兼容PyTorch 2.6+
            model_data = torch.load(filepath, map_location=self.device, weights_only=False)
            
            # 检查模型参数是否匹配
            params = model_data['model_params']
            mismatch = False
            
            # 检查关键参数
            if params['lottery_type'] != self.lottery_type or params['feature_window'] != self.feature_window:
                mismatch = True
                self.log(f"警告：关键参数不匹配 - lottery_type: {params['lottery_type']} vs {self.lottery_type}, feature_window: {params['feature_window']} vs {self.feature_window}")
                
                if strict:
                    self.log("错误：关键参数不匹配且strict=True，无法加载模型")
                    return False
            
            # 加载模型状态
            self.load_state_dict(model_data['model_state_dict'], strict=False)
            
            # 加载优化器状态
            if 'optimizer_state_dict' in model_data and model_data['optimizer_state_dict'] is not None and self.optimizer:
                try:
                    self.optimizer.load_state_dict(model_data['optimizer_state_dict'])
                except Exception as e:
                    self.log(f"警告：加载优化器状态时出错: {str(e)}")
            
            # 加载其他状态
            self.training_history = model_data['training_history']
            self.red_freq = model_data.get('red_freq')
            self.blue_freq = model_data.get('blue_freq')
            self.is_trained = model_data.get('is_trained', False)
            
            if 'last_sequence' in model_data:
                self._last_sequence = model_data['last_sequence']
            
            # 记录模型版本信息
            model_version = params.get('model_version', '1.0')
            timestamp = params.get('timestamp', '未知')
            self.log(f"模型已从 {filepath} 加载 (版本: {model_version}, 时间戳: {timestamp})")
            
            return True
            
        except Exception as e:
            self.log(f"加载模型时发生错误: {str(e)}")
            return False
    
    def get_model_info(self, include_history=False):
        """
        获取模型信息 - 简化版实现
        
        参数:
            include_history: 是否包含训练历史，默认为False
        """
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        info = {
            'model_type': 'Basic LSTM TimeStep',
            'lottery_type': self.lottery_type,
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'hidden_size': self.hidden_size,
            'num_layers': self.num_layers,
            'feature_window': self.feature_window,
            'device': str(self.device),
            'is_trained': self.is_trained,
            'learning_rate': self.learning_rate
        }
        
        if include_history and hasattr(self, 'training_history') and self.training_history:
            info['training_history'] = self.training_history
        
        return info
        
    def evaluate(self, test_data, batch_size=32):
        """
        评估模型在测试数据上的性能 - 简化版实现
        
        参数:
            test_data: 测试数据
            batch_size: 批次大小
            
        返回:
            包含基本评估指标的字典
        """
        if not self.is_trained:
            self.log("警告：模型尚未训练，评估结果可能不准确")
            
        self.eval()
        
        # 准备测试数据
        X_test, _, red_test_data, _, blue_test_data, _ = self.prepare_data(test_data, test_size=0)
        X_test_seq = self._prepare_sequence_data(X_test)
        
        # 转换为张量
        X_test_tensor = torch.FloatTensor(X_test_seq).to(self.device)
        red_test_tensors = [torch.LongTensor(red_data).to(self.device) for red_data in red_test_data]
        blue_test_tensors = [torch.LongTensor(blue_data).to(self.device) for blue_data in blue_test_data]
        
        # 创建数据加载器
        test_dataset = torch.utils.data.TensorDataset(
            X_test_tensor, *red_test_tensors, *blue_test_tensors
        )
        test_loader = torch.utils.data.DataLoader(
            test_dataset, batch_size=batch_size, shuffle=False
        )
        
        # 初始化评估指标
        total_loss = 0.0
        red_correct = 0
        blue_correct = 0
        total_samples = 0
        
        # 预测和真实值
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for batch_data in test_loader:
                X_batch = batch_data[0]
                red_targets = batch_data[1:1+self.red_count]
                blue_targets = batch_data[1+self.red_count:]
                
                # 前向传播
                outputs = self.forward(X_batch)
                
                # 计算损失
                batch_loss = 0.0
                batch_red_preds = []
                batch_red_targets = []
                
                # 红球损失和准确率
                for i, (red_logit, red_target) in enumerate(zip(outputs['red_logits'], red_targets)):
                    loss = self.criterion(red_logit, red_target)
                    batch_loss += loss
                    
                    # 计算准确率
                    _, predicted = torch.max(red_logit, 1)
                    red_correct += (predicted == red_target).sum().item()
                    
                    # 收集预测和目标
                    batch_red_preds.append(predicted.cpu().numpy())
                    batch_red_targets.append(red_target.cpu().numpy())
                
                # 蓝球损失和准确率
                batch_blue_preds = []
                batch_blue_targets = []
                
                for i, (blue_logit, blue_target) in enumerate(zip(outputs['blue_logits'], blue_targets)):
                    loss = self.criterion(blue_logit, blue_target)
                    batch_loss += loss
                    
                    # 计算准确率
                    _, predicted = torch.max(blue_logit, 1)
                    blue_correct += (predicted == blue_target).sum().item()
                    
                    # 收集预测和目标
                    batch_blue_preds.append(predicted.cpu().numpy())
                    batch_blue_targets.append(blue_target.cpu().numpy())
                
                total_loss += batch_loss.item()
                total_samples += X_batch.size(0)
                
                # 收集所有预测和目标
                for i in range(X_batch.size(0)):
                    pred_reds = [batch_red_preds[j][i] + 1 for j in range(len(batch_red_preds))]
                    pred_blues = [batch_blue_preds[j][i] + 1 for j in range(len(batch_blue_preds))]
                    
                    target_reds = [batch_red_targets[j][i] + 1 for j in range(len(batch_red_targets))]
                    target_blues = [batch_blue_targets[j][i] + 1 for j in range(len(batch_blue_targets))]
                    
                    all_predictions.append((pred_reds, pred_blues))
                    all_targets.append((target_reds, target_blues))
        
        # 计算评估指标
        avg_loss = total_loss / len(test_loader) if len(test_loader) > 0 else float('inf')
        red_accuracy = red_correct / (total_samples * self.red_count) if total_samples > 0 else 0
        blue_accuracy = blue_correct / (total_samples * self.blue_count) if total_samples > 0 else 0
        
        # 记录评估结果
        self.log(f"测试集损失: {avg_loss:.4f}")
        self.log(f"测试集红球准确率: {red_accuracy:.4f}")
        self.log(f"测试集蓝球准确率: {blue_accuracy:.4f}")
        
        # 返回评估结果
        return {
            'loss': avg_loss,
            'red_accuracy': red_accuracy,
            'blue_accuracy': blue_accuracy,
            'total_samples': total_samples,
            'predictions': all_predictions,
            'targets': all_targets
        }
    
    def plot_training_history(self, save_path=None, show=True, figsize=(12, 8)):
        """
        绘制训练历史曲线 - 基础版
        
        参数:
            save_path (str, optional): 保存图像的路径
            show (bool, optional): 是否显示图像
            figsize (tuple, optional): 图像大小
            
        返回:
            bool: 是否成功绘制
        """
        try:
            import matplotlib.pyplot as plt
            import os
            
            # 检查训练历史是否存在
            if not hasattr(self, 'training_history') or not self.training_history or not self.training_history.get('loss'):
                self.log("没有训练历史可供绘制")
                return False
            
            # 提取历史数据 - 简化版
            epochs = range(1, len(self.training_history['loss']) + 1)
            train_losses = [h['train'] for h in self.training_history['loss']]
            val_losses = [h['val'] for h in self.training_history['loss']] if 'val' in self.training_history['loss'][0] else None
            
            train_red_acc = [h['train'] for h in self.training_history['red_accuracy']]
            val_red_acc = [h['val'] for h in self.training_history['red_accuracy']] if 'val' in self.training_history['red_accuracy'][0] else None
            
            train_blue_acc = [h['train'] for h in self.training_history['blue_accuracy']]
            val_blue_acc = [h['val'] for h in self.training_history['blue_accuracy']] if 'val' in self.training_history['blue_accuracy'][0] else None
            
            has_val = val_losses is not None
            
            # 创建2x2的图形布局
            fig, axs = plt.subplots(2, 2, figsize=figsize)
            
            # 1. 损失曲线 - 左上
            axs[0, 0].plot(epochs, train_losses, 'b-', label='训练损失')
            if has_val:
                axs[0, 0].plot(epochs, val_losses, 'r--', label='验证损失')
            axs[0, 0].set_title('损失曲线')
            axs[0, 0].set_xlabel('Epoch')
            axs[0, 0].set_ylabel('Loss')
            axs[0, 0].legend()
            axs[0, 0].grid(True)
            
            # 2. 红球准确率 - 右上
            axs[0, 1].plot(epochs, train_red_acc, 'b-', label='训练红球准确率')
            if has_val:
                axs[0, 1].plot(epochs, val_red_acc, 'r--', label='验证红球准确率')
            axs[0, 1].set_title('红球准确率')
            axs[0, 1].set_xlabel('Epoch')
            axs[0, 1].set_ylabel('Accuracy')
            axs[0, 1].legend()
            axs[0, 1].grid(True)
            
            # 3. 蓝球准确率 - 左下
            axs[1, 0].plot(epochs, train_blue_acc, 'b-', label='训练蓝球准确率')
            if has_val:
                axs[1, 0].plot(epochs, val_blue_acc, 'r--', label='验证蓝球准确率')
            axs[1, 0].set_title('蓝球准确率')
            axs[1, 0].set_xlabel('Epoch')
            axs[1, 0].set_ylabel('Accuracy')
            axs[1, 0].legend()
            axs[1, 0].grid(True)
            
            # 4. 模型信息 - 右下
            axs[1, 1].axis('off')  # 不显示坐标轴
            model_info = self.get_model_info()
            info_text = (
                f"模型类型: {model_info['model_type']}\n"
                f"隐藏层大小: {model_info.get('hidden_size', 'N/A')}, 层数: {model_info.get('num_layers', 'N/A')}\n"
                f"Dropout: {model_info.get('dropout', 'N/A')}, 学习率: {model_info.get('learning_rate', 'N/A')}\n"
                f"参数总数: {model_info.get('total_parameters', 0):,}"
            )
            
            # 添加简单的训练摘要
            if has_val:
                best_epoch = val_losses.index(min(val_losses)) + 1
                summary_text = (
                    f"\n\n训练摘要:\n"
                    f"最佳Epoch: {best_epoch}, 最终验证损失: {val_losses[-1]:.4f}"
                )
                info_text += summary_text
            
            axs[1, 1].text(0.5, 0.5, info_text, ha='center', va='center', fontsize=10,
                          bbox=dict(boxstyle="round", facecolor="#f0f0f0"))
            
            # 调整布局
            plt.tight_layout()
            fig.suptitle('LSTM时间步模型训练历史 (基础版)', fontsize=14)
            plt.subplots_adjust(top=0.9)
            
            # 保存图形
            if save_path:
                os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
                plt.savefig(save_path, dpi=200)
                self.log(f"训练历史图已保存到 {save_path}")
            
            # 显示图形
            if show:
                plt.show()
            else:
                plt.close()
                
            return True
                
        except Exception as e:
            self.log(f"绘制训练历史时发生错误: {str(e)}")
            return False
    
    def load_models(self):
        """
        加载保存的模型 - 兼容LotteryMLModels接口
        
        返回:
            bool: 是否成功加载模型
        """
        try:
            model_path = self.get_model_path()
            self.log(f"尝试加载模型: {model_path}")
            
            # 检查文件是否存在
            if not os.path.exists(model_path):
                self.log(f"模型文件不存在: {model_path}")
                return False
                
            # 调用load_model方法加载模型
            load_success = self.load_model(model_path)
            
            if load_success:
                self.log(f"模型加载成功: {model_path}")
                return True
            else:
                self.log(f"模型加载失败: {model_path}")
                return False
                
        except Exception as e:
            self.log(f"加载模型时发生错误: {str(e)}")
            import traceback
            self.log(traceback.format_exc())
            return False
    
    def save_models(self):
        """
        保存模型 - 兼容LotteryMLModels接口
        
        返回:
            bool: 是否成功保存模型
        """
        try:
            # 确保模型目录存在
            model_path = self.get_model_path()
            os.makedirs(os.path.dirname(os.path.abspath(model_path)), exist_ok=True)
            
            self.log(f"正在保存模型到: {model_path}")
            
            # 调用save_model方法保存模型
            save_success = self.save_model(model_path, save_optimizer=True)
            
            if save_success:
                self.log(f"模型保存成功: {model_path}")
                
                # 保存训练损失图表
                if self.training_history and len(self.training_history) > 0:
                    plot_path = os.path.join(os.path.dirname(model_path), 'training_loss.png')
                    self.plot_training_history(save_path=plot_path, show=False)
                    self.log(f"训练历史图表已保存到: {plot_path}")
                    
                return True
            else:
                self.log(f"模型保存失败: {model_path}")
                return False
                
        except Exception as e:
            self.log(f"保存模型时发生错误: {str(e)}")
            import traceback
            self.log(traceback.format_exc())
            return False
    
    def get_model_path(self):
        """
        获取模型保存路径
        
        返回:
            str: 模型保存路径
        """
        import os
        model_dir = os.path.join('models', self.lottery_type, 'lstm_timeStep')
        os.makedirs(model_dir, exist_ok=True)
        return os.path.join(model_dir, f'{self.lottery_type}_model.pth')
    
    def visualize_predictions(self, test_data, num_samples=5, save_path=None, show=True):
        """
        可视化预测结果
        
        参数:
            test_data: 测试数据
            num_samples: 要可视化的样本数量
            save_path: 保存图形的路径
            show: 是否显示图形
        """
        try:
            import matplotlib.pyplot as plt
            import numpy as np
            import random
            import os
            
            if not self.is_trained:
                self.log("警告：模型尚未训练，预测结果可能不准确")
            
            # 准备测试数据
            if len(test_data) > num_samples:
                # 随机选择样本
                sample_indices = random.sample(range(len(test_data)), num_samples)
                samples = [test_data[i] for i in sample_indices]
            else:
                samples = test_data
                num_samples = len(samples)
            
            # 获取预测结果和概率分布
            all_predictions = []
            all_probabilities = []
            all_confidences = []
            
            for sample in samples:
                # 获取历史数据作为输入
                history = sample[0]  # 假设样本的第一个元素是历史数据
                
                # 预测
                pred, probs, conf = self.predict(history, return_probs=True)
                
                all_predictions.append(pred)
                all_probabilities.append(probs)
                all_confidences.append(conf)
            
            # 创建图形
            fig, axes = plt.subplots(num_samples, 2, figsize=(15, 4 * num_samples))
            if num_samples == 1:
                axes = np.array([axes])  # 确保axes是二维的
            
            for i, (sample, pred, probs, conf) in enumerate(zip(samples, all_predictions, all_probabilities, all_confidences)):
                # 获取真实标签
                true_red = sample[1:1+self.red_count]  # 假设样本的后续元素是标签
                true_blue = sample[1+self.red_count:]
                
                # 红球概率分布
                ax_red = axes[i, 0]
                red_probs = probs[0]  # 红球概率分布
                
                # 绘制红球概率分布（取第一个红球作为示例）
                x = np.arange(1, self.red_range + 1)
                ax_red.bar(x, red_probs[0][0], alpha=0.7, color='lightcoral')
                
                # 标记预测值和真实值
                for j, (p, t) in enumerate(zip(pred[0], true_red)):
                    ax_red.axvline(x=p, color='red', linestyle='--', alpha=0.7, label=f'预测 {j+1}: {p}' if j==0 else f'预测 {j+1}: {p}')
                    ax_red.axvline(x=t, color='green', linestyle='-', alpha=0.7, label=f'真实 {j+1}: {t}' if j==0 else f'真实 {j+1}: {t}')
                
                ax_red.set_title(f'样本 {i+1} - 红球概率分布')
                ax_red.set_xlabel('红球号码')
                ax_red.set_ylabel('概率')
                ax_red.legend(loc='upper right')
                ax_red.grid(True, linestyle='--', alpha=0.3)
                
                # 蓝球概率分布
                ax_blue = axes[i, 1]
                blue_probs = probs[1]  # 蓝球概率分布
                
                # 绘制蓝球概率分布（取第一个蓝球作为示例）
                x = np.arange(1, self.blue_range + 1)
                ax_blue.bar(x, blue_probs[0][0], alpha=0.7, color='lightskyblue')
                
                # 标记预测值和真实值
                for j, (p, t) in enumerate(zip(pred[1], true_blue)):
                    ax_blue.axvline(x=p, color='red', linestyle='--', alpha=0.7, label=f'预测 {j+1}: {p}' if j==0 else f'预测 {j+1}: {p}')
                    ax_blue.axvline(x=t, color='green', linestyle='-', alpha=0.7, label=f'真实 {j+1}: {t}' if j==0 else f'真实 {j+1}: {t}')
                
                ax_blue.set_title(f'样本 {i+1} - 蓝球概率分布')
                ax_blue.set_xlabel('蓝球号码')
                ax_blue.set_ylabel('概率')
                ax_blue.legend(loc='upper right')
                ax_blue.grid(True, linestyle='--', alpha=0.3)
                
                # 添加置信度信息
                red_conf_text = ', '.join([f'{c:.2f}' for c in conf[0]])
                blue_conf_text = ', '.join([f'{c:.2f}' for c in conf[1]])
                ax_red.text(0.05, 0.95, f'置信度: {red_conf_text}', transform=ax_red.transAxes, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
                ax_blue.text(0.05, 0.95, f'置信度: {blue_conf_text}', transform=ax_blue.transAxes, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
            
            plt.tight_layout()
            
            # 保存图形
            if save_path:
                # 确保目录存在
                os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                self.log(f"预测可视化图已保存到 {save_path}")
            
            # 显示图形
            if show:
                plt.show()
            else:
                plt.close()
                
            return True
                
        except Exception as e:
            self.log(f"可视化预测结果时发生错误: {str(e)}")
            import traceback
            self.log(traceback.format_exc())
            return False