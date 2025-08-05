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
    
    def __init__(self, input_size: int, output_size: int, dropout: float = 0.2):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        
        # 多层预测网络
        self.predictor = nn.Sequential(
            nn.Linear(input_size, input_size * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(input_size * 2, input_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(input_size, output_size)
        )
        
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
    """
    
    def __init__(self, lottery_type='dlt', feature_window=15, log_callback=None, use_gpu=False,
                 hidden_size=384, num_layers=3, dropout=0.2, bidirectional=True,
                 use_attention=True, learning_rate=0.0002, weight_decay=1e-5):
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
        
        # 设备配置 - 增加MPS支持（苹果M系列芯片）
        if use_gpu:
            if torch.cuda.is_available():
                self.device = torch.device('cuda')
            elif hasattr(torch, 'mps') and torch.backends.mps.is_available():
                self.device = torch.device('mps')
            else:
                self.device = torch.device('cpu')
        else:
            self.device = torch.device('cpu')
        
        # 输入特征大小
        self.input_size = self.red_count + self.blue_count
        
        # 构建网络
        self._build_network()
        
        # 移动到设备
        self.to(self.device)
        
        # 优化器配置 - 使用更稳定的配置
        self.optimizer = optim.AdamW(
            self.parameters(), 
            lr=self.learning_rate, 
            weight_decay=self.weight_decay,
            betas=(0.9, 0.999),
            eps=1e-8
        )
        
        # 学习率调度器 - 更平滑的学习率调整
        self.scheduler = optim.lr_scheduler.OneCycleLR(
            self.optimizer,
            max_lr=self.learning_rate * 3,  # 进一步降低最大学习率倍数
            epochs=800,                    # 大幅增加总epoch数
            steps_per_epoch=50,            # 减少每个epoch的步数
            pct_start=0.3,                 # 增加预热阶段
            anneal_strategy='cos',
            div_factor=20.0,               # 初始学习率除数
            final_div_factor=5000.0        # 最终学习率除数
        )
        
        # 损失函数 - 调整权重比例
        self.criterion = LotteryLoss(alpha=0.7, beta=0.3)  # 增加分布约束权重
        
        # 训练状态
        self.training_history = {'loss': [], 'red_accuracy': [], 'blue_accuracy': []}
        self.is_trained = False
        self.best_model_state = None
        
        # 历史频率统计
        self.red_freq = None
        self.blue_freq = None
        
        # 梯度裁剪值
        self.grad_clip_value = 0.5  # 添加更严格的梯度裁剪值
        
    def _build_network(self):
        """
        构建优化的网络架构
        """
        # 输入嵌入层 - 增加批归一化提高稳定性
        self.input_embedding = nn.Sequential(
            nn.Linear(self.input_size, self.hidden_size),
            nn.BatchNorm1d(self.hidden_size),
            nn.GELU(),
            nn.Dropout(self.dropout)
        )
        
        # 增强LSTM层 - 保持不变但参数已优化
        self.lstm_layer = EnhancedLSTMLayer(
            input_size=self.hidden_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            dropout=self.dropout,
            bidirectional=self.bidirectional,
            use_attention=self.use_attention
        )
        
        # 特征融合层 - 增加残差连接和层归一化
        lstm_output_size = self.hidden_size * 2 if self.bidirectional else self.hidden_size
        self.layer_norm1 = nn.LayerNorm(lstm_output_size)
        self.feature_fusion = nn.Sequential(
            nn.Linear(lstm_output_size, lstm_output_size),
            nn.GELU(),
            nn.Dropout(self.dropout),
            nn.Linear(lstm_output_size, lstm_output_size)
        )
        self.layer_norm2 = nn.LayerNorm(lstm_output_size)
        
        # 红球预测头 - 共享部分特征提取
        self.shared_red_features = nn.Sequential(
            nn.Linear(lstm_output_size, lstm_output_size),
            nn.LayerNorm(lstm_output_size),
            nn.GELU(),
            nn.Dropout(self.dropout * 0.5)  # 降低dropout以减少过拟合
        )
        self.red_heads = nn.ModuleList([
            AdaptivePredictionHead(lstm_output_size, self.red_range, self.dropout * 0.5)
            for _ in range(self.red_count)
        ])
        
        # 蓝球预测头 - 共享部分特征提取
        self.shared_blue_features = nn.Sequential(
            nn.Linear(lstm_output_size, lstm_output_size),
            nn.LayerNorm(lstm_output_size),
            nn.GELU(),
            nn.Dropout(self.dropout * 0.5)  # 降低dropout以减少过拟合
        )
        self.blue_heads = nn.ModuleList([
            AdaptivePredictionHead(lstm_output_size, self.blue_range, self.dropout * 0.5)
            for _ in range(self.blue_count)
        ])
        
        # 全局特征提取器 - 增加注意力池化
        self.global_attention = nn.Sequential(
            nn.Linear(lstm_output_size, 1),
            nn.Sigmoid()
        )
        self.global_feature_extractor = nn.Sequential(
            nn.Linear(lstm_output_size, lstm_output_size),
            nn.LayerNorm(lstm_output_size),
            nn.GELU(),
            nn.Dropout(self.dropout * 0.5)
        )
        
    def forward(self, x):
        """
        前向传播 - 优化的实现
        """
        batch_size, seq_len, _ = x.shape
        
        # 输入嵌入 - 处理批归一化需要的维度转换
        x_reshaped = x.reshape(-1, self.input_size)
        embedded = self.input_embedding(x_reshaped)
        embedded = embedded.reshape(batch_size, seq_len, self.hidden_size)
        
        # LSTM处理
        lstm_out, attention_weights = self.lstm_layer(embedded)
        
        # 特征融合 - 增加残差连接和层归一化
        normed_lstm = self.layer_norm1(lstm_out)
        fusion_out = self.feature_fusion(normed_lstm)
        fused_features = self.layer_norm2(normed_lstm + fusion_out)  # 残差连接
        
        # 全局特征 - 使用注意力加权池化
        # 计算注意力权重
        attention_scores = self.global_attention(fused_features)  # [batch_size, seq_len, 1]
        attention_weights_global = F.softmax(attention_scores, dim=1)
        
        # 加权求和得到全局特征
        global_context = torch.sum(fused_features * attention_weights_global, dim=1)  # [batch_size, hidden_size]
        global_features = self.global_feature_extractor(global_context)
        
        # 序列特征（使用最后时间步和倒数第二时间步的加权组合）
        if seq_len > 1:
            sequence_features = fused_features[:, -1, :] * 0.7 + fused_features[:, -2, :] * 0.3
        else:
            sequence_features = fused_features[:, -1, :]
        
        # 组合特征 - 使用门控机制融合全局和序列特征
        gate = torch.sigmoid(global_features + sequence_features)  # 自适应门控
        combined_features = gate * global_features + (1 - gate) * sequence_features
        
        # 红球预测 - 使用共享特征提取
        red_shared = self.shared_red_features(combined_features)
        red_outputs = [head(red_shared) for head in self.red_heads]
        
        # 蓝球预测 - 使用共享特征提取
        blue_shared = self.shared_blue_features(combined_features)
        blue_outputs = [head(blue_shared) for head in self.blue_heads]
        
        return {
            'red_logits': red_outputs,
            'blue_logits': blue_outputs,
            'attention_weights': attention_weights,
            'features': combined_features,
            'global_attention': attention_weights_global  # 返回全局注意力权重用于可视化
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
    
    def fit(self, data, epochs=800, batch_size=64, validation_split=0.2, 
            early_stopping_patience=150, **kwargs):
        """
        训练模型
        """
        self.log(f"开始训练高级LSTM TimeStep模型，彩票类型: {self.lottery_type}")
        self.log(f"优化参数: hidden_size={self.hidden_size}, num_layers={self.num_layers}")
        self.log(f"训练参数: epochs={epochs}, batch_size={batch_size}, lr={self.learning_rate}")
        self.log(f"训练数据: {data}")
        
        try:
            # 计算历史频率
            self.red_freq, self.blue_freq = self._compute_historical_frequency(data)
            self.log(f"红球频率: {self.red_freq}")
            self.log(f"蓝球频率: {self.blue_freq}")
            # 准备数据
            X_train, X_val, red_train_data, red_val_data, blue_train_data, blue_val_data = self.prepare_data(
                data, test_size=validation_split
            )
            
            # 转换为序列数据
            X_train_seq = self._prepare_sequence_data(X_train)
            X_val_seq = self._prepare_sequence_data(X_val)
            
            # 保存最后序列用于预测
            self._last_sequence = X_val_seq[-1:] if len(X_val_seq) > 0 else X_train_seq[-1:]
            
            # 转换为张量
            X_train_tensor = torch.FloatTensor(X_train_seq).to(self.device)
            X_val_tensor = torch.FloatTensor(X_val_seq).to(self.device)
            
            # 准备标签
            red_train_tensors = [torch.LongTensor(red_data).to(self.device) for red_data in red_train_data]
            red_val_tensors = [torch.LongTensor(red_data).to(self.device) for red_data in red_val_data]
            blue_train_tensors = [torch.LongTensor(blue_data).to(self.device) for blue_data in blue_train_data]
            blue_val_tensors = [torch.LongTensor(blue_data).to(self.device) for blue_data in blue_val_data]
            
            # 数据加载器 - 添加数据增强和加权采样
            # 使用加权采样，增加最近数据的权重
            weights = torch.linspace(0.5, 1.0, len(X_train_tensor))
            sampler = torch.utils.data.WeightedRandomSampler(weights, len(weights), replacement=True)
            
            train_dataset = torch.utils.data.TensorDataset(
                X_train_tensor, *red_train_tensors, *blue_train_tensors
            )
            train_loader = torch.utils.data.DataLoader(
                train_dataset, batch_size=batch_size, sampler=sampler, drop_last=True
            )
            
            val_dataset = torch.utils.data.TensorDataset(
                X_val_tensor, *red_val_tensors, *blue_val_tensors
            )
            val_loader = torch.utils.data.DataLoader(
                val_dataset, batch_size=batch_size, shuffle=False
            )
            
            # 训练循环
            best_val_loss = float('inf')
            best_val_combined_acc = 0.0  # 添加综合准确率指标
            patience_counter = 0
            
            # 学习率预热 - 增加预热轮数
            warmup_epochs = 10
            initial_lr = self.learning_rate / 5
            
            # 训练前设置较小的学习率进行预热
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = initial_lr
            
            for epoch in range(epochs):
                # 学习率预热
                if epoch < warmup_epochs:
                    lr = initial_lr + (self.learning_rate - initial_lr) * epoch / warmup_epochs
                    for param_group in self.optimizer.param_groups:
                        param_group['lr'] = lr
                
                # 训练阶段
                train_loss, train_red_acc, train_blue_acc = self._train_epoch(train_loader)
                
                # 验证阶段
                val_loss, val_red_acc, val_blue_acc = self._validate_epoch(val_loader)
                
                # 记录历史
                self.training_history['loss'].append({'train': train_loss, 'val': val_loss})
                self.training_history['red_accuracy'].append({'train': train_red_acc, 'val': val_red_acc})
                self.training_history['blue_accuracy'].append({'train': train_blue_acc, 'val': val_blue_acc})
                
                # 计算综合准确率 - 红球权重0.7，蓝球权重0.3
                val_combined_acc = val_red_acc * 0.7 + val_blue_acc * 0.3
                
                # 早停检查 - 使用更宽松的综合指标
                improved = False
                if val_loss < best_val_loss * 0.999:  # 损失需要改善(至少0.1%)
                    best_val_loss = val_loss
                    improved = True
                
                if val_combined_acc > best_val_combined_acc * 1.001:  # 准确率需要改善(至少0.1%)
                    best_val_combined_acc = val_combined_acc
                    improved = True
                
                if improved:
                    patience_counter = 0
                    self.best_model_state = self.state_dict().copy()
                else:
                    patience_counter += 1
                
                # 学习率调度 - 预热期后使用
                if epoch >= warmup_epochs:
                    self.scheduler.step()
                
                # 动态早停耐心值 - 增加整体耐心值
                dynamic_patience = early_stopping_patience * 2  # 基础耐心值翻倍
                if epoch < epochs // 3:  # 训练初期使用更大的耐心值
                    dynamic_patience = early_stopping_patience * 4
                elif epoch > epochs * 2 // 3:  # 训练后期仍保持较大耐心值
                    dynamic_patience = early_stopping_patience * 2
                
                # 日志输出 - 增加输出频率
                if epoch % 10 == 0 or epoch == epochs - 1:
                    current_lr = self.optimizer.param_groups[0]['lr']
                    self.log(f"Epoch {epoch+1}/{epochs}: Train Loss: {train_loss:.4f}, "
                           f"Val Loss: {val_loss:.4f}, Red Acc: {val_red_acc:.3f}, "
                           f"Blue Acc: {val_blue_acc:.3f}, Combined Acc: {val_combined_acc:.3f}, "
                           f"LR: {current_lr:.6f}, Patience: {patience_counter}/{dynamic_patience}")
                
                # 早停
                if patience_counter >= dynamic_patience:
                    self.log(f"早停触发，在第{epoch+1}轮停止训练，已经{patience_counter}个epoch没有改善")
                    break
            
            # 加载最佳模型
            if self.best_model_state is not None:
                self.load_state_dict(self.best_model_state)
                self.log(f"加载最佳模型，验证损失: {best_val_loss:.4f}, 验证综合准确率: {best_val_combined_acc:.4f}")
            
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
        return X.reshape(samples, self.feature_window, self.input_size)
    
    def _train_epoch(self, train_loader):
        """
        训练一个epoch - 优化的实现
        """
        self.train()
        total_loss = 0.0
        red_correct = 0
        blue_correct = 0
        total_samples = 0
        
        # 使用tqdm显示进度条
        for batch_data in train_loader:
            X_batch = batch_data[0]
            red_targets = batch_data[1:1+self.red_count]
            blue_targets = batch_data[1+self.red_count:]
            
            # 使用更高效的梯度清零方式
            for param in self.parameters():
                param.grad = None
            
            # 前向传播
            outputs = self.forward(X_batch)
            
            # 计算损失 - 使用更高效的损失计算
            red_loss = 0.0
            blue_loss = 0.0
            
            # 红球损失 - 使用权重衰减处理不同位置的球
            for i, (red_logit, red_target) in enumerate(zip(outputs['red_logits'], red_targets)):
                # 位置权重 - 前面的球更重要
                position_weight = 1.0 - (i * 0.05)  # 位置权重从1.0逐渐减小
                position_weight = max(0.7, position_weight)  # 确保最小权重不低于0.7
                
                loss = self.criterion(red_logit, red_target, self.red_freq) * position_weight
                red_loss += loss
                
                # 计算准确率
                _, predicted = torch.max(red_logit, 1)
                red_correct += (predicted == red_target).sum().item()
            
            # 蓝球损失 - 蓝球通常更重要，增加权重
            blue_weight = 1.2  # 蓝球权重增加20%
            for i, (blue_logit, blue_target) in enumerate(zip(outputs['blue_logits'], blue_targets)):
                loss = self.criterion(blue_logit, blue_target, self.blue_freq) * blue_weight
                blue_loss += loss
                
                # 计算准确率
                _, predicted = torch.max(blue_logit, 1)
                blue_correct += (predicted == blue_target).sum().item()
            
            # 总损失 - 添加L2正则化
            total_batch_loss = red_loss + blue_loss
            
            # 反向传播
            total_batch_loss.backward()
            
            # 梯度裁剪 - 使用更严格的值
            torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=self.grad_clip_value)
            
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
        验证一个epoch - 优化的实现
        """
        self.eval()
        total_loss = 0.0
        red_correct = 0
        blue_correct = 0
        total_samples = 0
        
        # 添加更多评估指标
        red_top3_correct = 0  # 红球Top-3准确率
        blue_top2_correct = 0  # 蓝球Top-2准确率
        red_position_correct = [0] * self.red_count  # 每个位置的红球准确率
        
        with torch.no_grad():
            for batch_data in val_loader:
                X_batch = batch_data[0]
                red_targets = batch_data[1:1+self.red_count]
                blue_targets = batch_data[1+self.red_count:]
                
                # 前向传播
                outputs = self.forward(X_batch)
                
                # 计算损失 - 与训练过程保持一致
                red_loss = 0.0
                blue_loss = 0.0
                
                # 红球损失 - 使用与训练相同的权重
                for i, (red_logit, red_target) in enumerate(zip(outputs['red_logits'], red_targets)):
                    # 位置权重
                    position_weight = 1.0 - (i * 0.05)
                    position_weight = max(0.7, position_weight)
                    
                    loss = self.criterion(red_logit, red_target, self.red_freq) * position_weight
                    red_loss += loss
                    
                    # 计算准确率
                    _, predicted = torch.max(red_logit, 1)
                    batch_correct = (predicted == red_target).sum().item()
                    red_correct += batch_correct
                    red_position_correct[i] += batch_correct
                    
                    # 计算Top-3准确率
                    _, top3_indices = torch.topk(red_logit, 3, dim=1)
                    for j in range(X_batch.size(0)):
                        if red_target[j].item() in top3_indices[j]:
                            red_top3_correct += 1
                
                # 蓝球损失 - 使用与训练相同的权重
                blue_weight = 1.2
                for i, (blue_logit, blue_target) in enumerate(zip(outputs['blue_logits'], blue_targets)):
                    loss = self.criterion(blue_logit, blue_target, self.blue_freq) * blue_weight
                    blue_loss += loss
                    
                    # 计算准确率
                    _, predicted = torch.max(blue_logit, 1)
                    blue_correct += (predicted == blue_target).sum().item()
                    
                    # 计算Top-2准确率
                    _, top2_indices = torch.topk(blue_logit, 2, dim=1)
                    for j in range(X_batch.size(0)):
                        if blue_target[j].item() in top2_indices[j]:
                            blue_top2_correct += 1
                
                # 总损失
                total_batch_loss = red_loss + blue_loss
                total_loss += total_batch_loss.item()
                total_samples += X_batch.size(0)
        
        # 计算平均损失和准确率
        avg_loss = total_loss / len(val_loader)
        red_accuracy = red_correct / (total_samples * self.red_count)
        blue_accuracy = blue_correct / (total_samples * self.blue_count)
        
        # 计算额外的评估指标
        red_top3_accuracy = red_top3_correct / (total_samples * self.red_count)
        blue_top2_accuracy = blue_top2_correct / (total_samples * self.blue_count)
        red_position_accuracy = [pos_correct / total_samples for pos_correct in red_position_correct]
        
        # 记录额外的评估指标
        self.log(f"验证集红球Top-3准确率: {red_top3_accuracy:.4f}")
        self.log(f"验证集蓝球Top-2准确率: {blue_top2_accuracy:.4f}")
        for i, acc in enumerate(red_position_accuracy):
            self.log(f"验证集红球位置{i+1}准确率: {acc:.4f}")
        
        return avg_loss, red_accuracy, blue_accuracy
    
    def predict(self, recent_data=None, num_predictions=1, temperature=0.8, top_k=None, return_probs=False, **kwargs):
        """
        预测彩票号码 - 优化的实现
        
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
                        
                        # 转换为numpy数组并标准化
                        X_data = np.array([features])
                        X_reshaped = X_data.reshape(X_data.shape[0], -1)
                        
                        # 使用保存的缩放器进行标准化
                        if hasattr(self, 'scalers') and 'X' in self.scalers:
                            X_scaled = self.scalers['X'].transform(X_reshaped)
                        else:
                            # 如果没有缩放器，直接使用原始数据
                            X_scaled = X_reshaped
                            self.log("警告: 未找到特征缩放器，使用原始数据进行预测")
                        
                        input_data = X_scaled.reshape(1, self.feature_window, self.input_size)
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
                
                # 红球预测（使用自适应温度采样增加多样性）
                for i, red_logit in enumerate(outputs['red_logits']):
                    # 前面的球使用较低的温度（更确定），后面的球使用较高的温度（更随机）
                    position_temp = temperature * (0.8 + i * 0.1)  # 温度从0.8*temperature逐渐增加
                    position_temp = min(position_temp, temperature * 1.5)  # 最大不超过1.5倍基础温度
                    
                    # 应用温度缩放
                    scaled_logits = red_logit / position_temp
                    probabilities = F.softmax(scaled_logits, dim=1)
                    batch_red_probs.append(probabilities.cpu().numpy())
                    
                    if top_k is not None:
                        # 只保留top_k个概率最高的球
                        top_k_probs, top_k_indices = torch.topk(probabilities, k=min(top_k, probabilities.size(1)), dim=1)
                        # 记录置信度 - 选中概率与第二高概率的比值
                        if top_k >= 2:
                            confidence = top_k_probs[:, 0] / (top_k_probs[:, 1] + 1e-6)
                        else:
                            confidence = top_k_probs[:, 0]
                        
                        # 重新归一化概率
                        top_k_probs = top_k_probs / top_k_probs.sum(dim=1, keepdim=True)
                        # 采样
                        sampled_indices = torch.multinomial(top_k_probs, 1)
                        predicted = top_k_indices.gather(1, sampled_indices).squeeze().item()
                    else:
                        # 直接采样
                        predicted = torch.multinomial(probabilities, 1).squeeze().item()
                        # 记录置信度 - 选中号码的概率
                        confidence = probabilities[0, predicted].item()
                    
                    red_numbers.append(predicted + 1)  # 转换为1-based索引
                    red_confidences.append(confidence.item() if isinstance(confidence, torch.Tensor) else confidence)
                
                # 蓝球预测（使用较低的温度，因为通常蓝球更重要）
                for i, blue_logit in enumerate(outputs['blue_logits']):
                    blue_temp = temperature * 0.9  # 蓝球使用较低的温度
                    scaled_logits = blue_logit / blue_temp
                    probabilities = F.softmax(scaled_logits, dim=1)
                    batch_blue_probs.append(probabilities.cpu().numpy())
                    
                    if top_k is not None:
                        # 只保留top_k个概率最高的球
                        top_k_probs, top_k_indices = torch.topk(probabilities, k=min(top_k, probabilities.size(1)), dim=1)
                        # 记录置信度
                        if top_k >= 2:
                            confidence = top_k_probs[:, 0] / (top_k_probs[:, 1] + 1e-6)
                        else:
                            confidence = top_k_probs[:, 0]
                        
                        # 重新归一化概率
                        top_k_probs = top_k_probs / top_k_probs.sum(dim=1, keepdim=True)
                        # 采样
                        sampled_indices = torch.multinomial(top_k_probs, 1)
                        predicted = top_k_indices.gather(1, sampled_indices).squeeze().item()
                    else:
                        # 直接采样
                        predicted = torch.multinomial(probabilities, 1).squeeze().item()
                        # 记录置信度
                        confidence = probabilities[0, predicted].item()
                    
                    blue_numbers.append(predicted + 1)  # 转换为1-based索引
                    blue_confidences.append(confidence.item() if isinstance(confidence, torch.Tensor) else confidence)
                
                # 确保红球号码唯一，同时更新置信度
                red_numbers, red_confidences = self._ensure_unique_red_numbers_with_confidence(red_numbers, red_confidences)
                
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
        确保红球号码唯一，同时更新置信度
        
        参数:
            red_numbers: 红球号码列表
            confidences: 对应的置信度列表
            
        返回:
            唯一的红球号码列表和更新后的置信度列表
        """
        if confidences is None:
            confidences = [1.0] * len(red_numbers)
            
        # 创建号码-置信度对，并按置信度降序排序
        num_conf_pairs = sorted(zip(red_numbers, confidences), key=lambda x: x[1], reverse=True)
        
        unique_numbers = []
        updated_confidences = []
        used_numbers = set()
        
        # 首先处理高置信度的号码
        for num, conf in num_conf_pairs:
            if num not in used_numbers:
                unique_numbers.append(num)
                updated_confidences.append(conf)
                used_numbers.add(num)
            else:
                # 找一个未使用的号码，置信度降低50%
                for candidate in range(1, self.red_range + 1):
                    if candidate not in used_numbers:
                        unique_numbers.append(candidate)
                        updated_confidences.append(conf * 0.5)  # 降低置信度
                        used_numbers.add(candidate)
                        break
        
        # 按原始顺序重新排列
        original_order = list(range(len(red_numbers)))
        ordered_pairs = sorted(zip(original_order, unique_numbers, updated_confidences), key=lambda x: x[0])
        
        ordered_numbers = [pair[1] for pair in ordered_pairs]
        ordered_confidences = [pair[2] for pair in ordered_pairs]
        
        return ordered_numbers, ordered_confidences
    
    def save_model(self, filepath, save_optimizer=True):
        """
        保存模型 - 优化的实现
        
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
                    'model_version': '2.0',  # 添加版本信息
                    'timestamp': datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                },
                'red_freq': self.red_freq,
                'blue_freq': self.blue_freq,
                'is_trained': self.is_trained
            }
            
            # 可选保存优化器和调度器状态
            if save_optimizer:
                model_data['optimizer_state_dict'] = self.optimizer.state_dict() if self.optimizer else None
                model_data['scheduler_state_dict'] = self.scheduler.state_dict() if self.scheduler else None
            
            if hasattr(self, '_last_sequence'):
                model_data['last_sequence'] = self._last_sequence
            
            # 使用临时文件保存，然后重命名，避免保存过程中断导致文件损坏
            temp_filepath = filepath + ".tmp"
            torch.save(model_data, temp_filepath)
            if os.path.exists(filepath):
                os.remove(filepath)
            os.rename(temp_filepath, filepath)
            
            self.log(f"模型已保存到: {filepath}")
            return True
            
        except Exception as e:
            self.log(f"保存模型时发生错误: {str(e)}")
            import traceback
            self.log(traceback.format_exc())
            return False
    
    def load_model(self, filepath, strict=False):
        """
        加载模型 - 优化的实现
        
        参数:
            filepath: 模型文件路径
            strict: 是否严格检查参数匹配，默认为False
        """
        try:
            # 检查文件是否存在
            if not os.path.exists(filepath):
                self.log(f"模型文件不存在: {filepath}")
                return False
                
            model_data = torch.load(filepath, map_location=self.device, weights_only=False)
            
            # 检查模型参数是否匹配
            params = model_data['model_params']
            mismatch = False
            mismatch_params = []
            
            # 检查关键参数
            critical_params = ['lottery_type', 'feature_window']
            for param in critical_params:
                if param in params and getattr(self, param) != params[param]:
                    mismatch = True
                    mismatch_params.append(f"{param}: 模型={params[param]}, 当前={getattr(self, param)}")
            
            # 检查非关键参数
            non_critical_params = ['hidden_size', 'num_layers', 'dropout', 'bidirectional', 'use_attention']
            for param in non_critical_params:
                if param in params and getattr(self, param) != params[param]:
                    mismatch_params.append(f"{param}: 模型={params[param]}, 当前={getattr(self, param)}")
            
            if mismatch and strict:
                self.log(f"错误：加载的模型参数与当前模型不匹配: {', '.join(mismatch_params)}")
                return False
            elif mismatch:
                self.log(f"警告：加载的模型参数与当前模型不完全匹配: {', '.join(mismatch_params)}")
            
            # 重建网络
            if mismatch and not strict:
                self.__init__(
                    lottery_type=params['lottery_type'],
                    feature_window=params['feature_window'],
                    log_callback=self.log_callback,
                    use_gpu=self.use_gpu,
                    hidden_size=params['hidden_size'],
                    num_layers=params['num_layers'],
                    dropout=params['dropout'],
                    bidirectional=params['bidirectional'],
                    use_attention=params['use_attention'],
                    learning_rate=params['learning_rate'],
                    weight_decay=params['weight_decay']
                )
            
            # 加载模型状态
            missing_keys, unexpected_keys = self.load_state_dict(model_data['model_state_dict'], strict=not mismatch)
            if missing_keys:
                self.log(f"警告：模型缺少以下键: {missing_keys}")
            if unexpected_keys:
                self.log(f"警告：模型包含意外的键: {unexpected_keys}")
            
            # 加载优化器状态
            if 'optimizer_state_dict' in model_data and model_data['optimizer_state_dict'] is not None:
                try:
                    self.optimizer.load_state_dict(model_data['optimizer_state_dict'])
                    # 确保优化器状态在正确的设备上
                    for state in self.optimizer.state.values():
                        for k, v in state.items():
                            if isinstance(v, torch.Tensor):
                                state[k] = v.to(self.device)
                except Exception as e:
                    self.log(f"警告：加载优化器状态时出错: {str(e)}")
            
            # 加载调度器状态
            if 'scheduler_state_dict' in model_data and model_data['scheduler_state_dict'] is not None and self.scheduler is not None:
                try:
                    self.scheduler.load_state_dict(model_data['scheduler_state_dict'])
                except Exception as e:
                    self.log(f"警告：加载学习率调度器状态时出错: {str(e)}")
            
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
            import traceback
            self.log(traceback.format_exc())
            return False
    
    def get_model_info(self, include_history=False):
        """
        获取模型信息 - 优化的实现
        
        参数:
            include_history: 是否包含训练历史，默认为False
        """
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        info = {
            'model_type': 'Enhanced LSTM TimeStep',
            'lottery_type': self.lottery_type,
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'hidden_size': self.hidden_size,
            'num_layers': self.num_layers,
            'feature_window': self.feature_window,
            'use_attention': self.use_attention,
            'bidirectional': self.bidirectional,
            'device': str(self.device),
            'is_trained': self.is_trained,
            'grad_clip_value': self.grad_clip_value,
            'learning_rate': self.learning_rate,
            'weight_decay': self.weight_decay
        }
        
        if include_history and hasattr(self, 'training_history') and self.training_history:
            info['training_history'] = self.training_history
        
        return info
        
    def evaluate(self, test_data, batch_size=32):
        """
        评估模型在测试数据上的性能
        
        参数:
            test_data: 测试数据
            batch_size: 批次大小
            
        返回:
            包含各种评估指标的字典
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
        red_top3_correct = 0
        blue_top2_correct = 0
        red_position_correct = [0] * self.red_count
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
                    loss = self.criterion(red_logit, red_target, self.red_freq)
                    batch_loss += loss
                    
                    # 计算准确率
                    _, predicted = torch.max(red_logit, 1)
                    batch_correct = (predicted == red_target).sum().item()
                    red_correct += batch_correct
                    red_position_correct[i] += batch_correct
                    
                    # 计算Top-3准确率
                    _, top3_indices = torch.topk(red_logit, 3, dim=1)
                    for j in range(X_batch.size(0)):
                        if red_target[j].item() in top3_indices[j]:
                            red_top3_correct += 1
                    
                    # 收集预测和目标
                    batch_red_preds.append(predicted.cpu().numpy())
                    batch_red_targets.append(red_target.cpu().numpy())
                
                # 蓝球损失和准确率
                batch_blue_preds = []
                batch_blue_targets = []
                
                for i, (blue_logit, blue_target) in enumerate(zip(outputs['blue_logits'], blue_targets)):
                    loss = self.criterion(blue_logit, blue_target, self.blue_freq)
                    batch_loss += loss
                    
                    # 计算准确率
                    _, predicted = torch.max(blue_logit, 1)
                    blue_correct += (predicted == blue_target).sum().item()
                    
                    # 计算Top-2准确率
                    _, top2_indices = torch.topk(blue_logit, 2, dim=1)
                    for j in range(X_batch.size(0)):
                        if blue_target[j].item() in top2_indices[j]:
                            blue_top2_correct += 1
                    
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
        red_top3_accuracy = red_top3_correct / (total_samples * self.red_count) if total_samples > 0 else 0
        blue_top2_accuracy = blue_top2_correct / (total_samples * self.blue_count) if total_samples > 0 else 0
        red_position_accuracy = [pos_correct / total_samples for pos_correct in red_position_correct] if total_samples > 0 else [0] * self.red_count
        
        # 计算完全匹配率（所有球都预测正确）
        exact_matches = 0
        for pred, target in zip(all_predictions, all_targets):
            if pred[0] == target[0] and pred[1] == target[1]:
                exact_matches += 1
        exact_match_rate = exact_matches / total_samples if total_samples > 0 else 0
        
        # 计算部分匹配率
        red_matches = 0
        blue_matches = 0
        for pred, target in zip(all_predictions, all_targets):
            if pred[0] == target[0]:
                red_matches += 1
            if pred[1] == target[1]:
                blue_matches += 1
        
        red_match_rate = red_matches / total_samples if total_samples > 0 else 0
        blue_match_rate = blue_matches / total_samples if total_samples > 0 else 0
        
        # 记录评估结果
        self.log(f"测试集损失: {avg_loss:.4f}")
        self.log(f"测试集红球准确率: {red_accuracy:.4f}")
        self.log(f"测试集蓝球准确率: {blue_accuracy:.4f}")
        self.log(f"测试集红球Top-3准确率: {red_top3_accuracy:.4f}")
        self.log(f"测试集蓝球Top-2准确率: {blue_top2_accuracy:.4f}")
        self.log(f"测试集完全匹配率: {exact_match_rate:.4f}")
        self.log(f"测试集红球完全匹配率: {red_match_rate:.4f}")
        self.log(f"测试集蓝球完全匹配率: {blue_match_rate:.4f}")
        
        for i, acc in enumerate(red_position_accuracy):
            self.log(f"测试集红球位置{i+1}准确率: {acc:.4f}")
        
        # 返回评估结果
        return {
            'loss': avg_loss,
            'red_accuracy': red_accuracy,
            'blue_accuracy': blue_accuracy,
            'red_top3_accuracy': red_top3_accuracy,
            'blue_top2_accuracy': blue_top2_accuracy,
            'red_position_accuracy': red_position_accuracy,
            'exact_match_rate': exact_match_rate,
            'red_match_rate': red_match_rate,
            'blue_match_rate': blue_match_rate,
            'total_samples': total_samples,
            'predictions': all_predictions,
            'targets': all_targets
        }
    
    def plot_training_history(self, save_path=None, show=True, figsize=(12, 18)):
        """
        绘制训练历史 - 优化的实现
        
        参数:
            save_path: 保存图形的路径，默认为None
            show: 是否显示图形，默认为True
            figsize: 图形大小，默认为(12, 18)
        """
        try:
            import matplotlib.pyplot as plt
            import numpy as np
            from matplotlib.gridspec import GridSpec
            import os
            
            # 检查训练历史是否存在
            if not hasattr(self, 'training_history') or not self.training_history or not self.training_history.get('loss'):
                self.log("警告：没有训练历史可供绘制")
                return False
            
            # 提取历史数据
            epochs = range(1, len(self.training_history['loss']) + 1)
            train_losses = [h['train'] for h in self.training_history['loss']]
            val_losses = [h['val'] for h in self.training_history['loss']] if 'val' in self.training_history['loss'][0] else None
            
            train_red_acc = [h['train'] for h in self.training_history['red_accuracy']]
            val_red_acc = [h['val'] for h in self.training_history['red_accuracy']] if 'val' in self.training_history['red_accuracy'][0] else None
            
            train_blue_acc = [h['train'] for h in self.training_history['blue_accuracy']]
            val_blue_acc = [h['val'] for h in self.training_history['blue_accuracy']] if 'val' in self.training_history['blue_accuracy'][0] else None
            
            has_val = val_losses is not None
            
            # 创建图形 - 使用GridSpec实现更灵活的布局
            fig = plt.figure(figsize=figsize)
            gs = GridSpec(4, 2, figure=fig)
            
            # 设置全局样式
            plt.style.use('seaborn-v0_8-darkgrid' if 'seaborn-v0_8-darkgrid' in plt.style.available else 'seaborn-darkgrid')
            colors = ['#2C3E50', '#E74C3C', '#3498DB', '#2ECC71', '#F39C12']
            
            # 1. 损失曲线 - 左上角，占据两行
            ax_loss = fig.add_subplot(gs[0:2, 0])
            ax_loss.plot(epochs, train_losses, color=colors[0], marker='o', linestyle='-', markersize=3, label='训练损失')
            if has_val:
                ax_loss.plot(epochs, val_losses, color=colors[1], marker='s', linestyle='--', markersize=3, label='验证损失')
            
            # 添加最小值标记
            if has_val:
                min_val_loss = min(val_losses)
                min_epoch = val_losses.index(min_val_loss) + 1
                ax_loss.scatter(min_epoch, min_val_loss, color='red', s=100, zorder=5, alpha=0.8, label=f'最小验证损失: {min_val_loss:.4f} (Epoch {min_epoch})')
            
            ax_loss.set_title('损失曲线', fontsize=14, fontweight='bold')
            ax_loss.set_xlabel('Epoch', fontsize=12)
            ax_loss.set_ylabel('Loss', fontsize=12)
            ax_loss.legend(loc='upper right', fontsize=10)
            ax_loss.grid(True, linestyle='--', alpha=0.7)
            
            # 2. 红球准确率 - 右上角
            ax_red = fig.add_subplot(gs[0, 1])
            ax_red.plot(epochs, train_red_acc, color=colors[2], marker='o', linestyle='-', markersize=3, label='训练红球准确率')
            if has_val:
                ax_red.plot(epochs, val_red_acc, color=colors[3], marker='s', linestyle='--', markersize=3, label='验证红球准确率')
            
            # 添加最大值标记
            if has_val:
                max_val_red_acc = max(val_red_acc)
                max_epoch = val_red_acc.index(max_val_red_acc) + 1
                ax_red.scatter(max_epoch, max_val_red_acc, color='green', s=100, zorder=5, alpha=0.8, label=f'最高验证准确率: {max_val_red_acc:.4f}')
            
            ax_red.set_title('红球准确率', fontsize=14, fontweight='bold')
            ax_red.set_xlabel('Epoch', fontsize=12)
            ax_red.set_ylabel('Accuracy', fontsize=12)
            ax_red.legend(loc='lower right', fontsize=10)
            ax_red.grid(True, linestyle='--', alpha=0.7)
            
            # 3. 蓝球准确率 - 右中
            ax_blue = fig.add_subplot(gs[1, 1])
            ax_blue.plot(epochs, train_blue_acc, color=colors[3], marker='o', linestyle='-', markersize=3, label='训练蓝球准确率')
            if has_val:
                ax_blue.plot(epochs, val_blue_acc, color=colors[4], marker='s', linestyle='--', markersize=3, label='验证蓝球准确率')
            
            # 添加最大值标记
            if has_val:
                max_val_blue_acc = max(val_blue_acc)
                max_epoch = val_blue_acc.index(max_val_blue_acc) + 1
                ax_blue.scatter(max_epoch, max_val_blue_acc, color='green', s=100, zorder=5, alpha=0.8, label=f'最高验证准确率: {max_val_blue_acc:.4f}')
            
            ax_blue.set_title('蓝球准确率', fontsize=14, fontweight='bold')
            ax_blue.set_xlabel('Epoch', fontsize=12)
            ax_blue.set_ylabel('Accuracy', fontsize=12)
            ax_blue.legend(loc='lower right', fontsize=10)
            ax_blue.grid(True, linestyle='--', alpha=0.7)
            
            # 4. 组合准确率 - 左下
            ax_combined = fig.add_subplot(gs[2, 0])
            combined_train_acc = [0.7 * r + 0.3 * b for r, b in zip(train_red_acc, train_blue_acc)]
            ax_combined.plot(epochs, combined_train_acc, color=colors[0], marker='o', linestyle='-', markersize=3, label='训练组合准确率')
            
            if has_val:
                combined_val_acc = [0.7 * r + 0.3 * b for r, b in zip(val_red_acc, val_blue_acc)]
                ax_combined.plot(epochs, combined_val_acc, color=colors[1], marker='s', linestyle='--', markersize=3, label='验证组合准确率')
                
                # 添加最大值标记
                max_val_combined_acc = max(combined_val_acc)
                max_epoch = combined_val_acc.index(max_val_combined_acc) + 1
                ax_combined.scatter(max_epoch, max_val_combined_acc, color='green', s=100, zorder=5, alpha=0.8, label=f'最高组合准确率: {max_val_combined_acc:.4f}')
            
            ax_combined.set_title('组合准确率 (红球0.7 + 蓝球0.3)', fontsize=14, fontweight='bold')
            ax_combined.set_xlabel('Epoch', fontsize=12)
            ax_combined.set_ylabel('Accuracy', fontsize=12)
            ax_combined.legend(loc='lower right', fontsize=10)
            ax_combined.grid(True, linestyle='--', alpha=0.7)
            
            # 5. 学习率变化 - 右下
            if hasattr(self, 'lr_history') and self.lr_history:
                ax_lr = fig.add_subplot(gs[2, 1])
                ax_lr.plot(epochs, self.lr_history, color=colors[4], marker='o', linestyle='-', markersize=3)
                ax_lr.set_title('学习率变化', fontsize=14, fontweight='bold')
                ax_lr.set_xlabel('Epoch', fontsize=12)
                ax_lr.set_ylabel('Learning Rate', fontsize=12)
                ax_lr.grid(True, linestyle='--', alpha=0.7)
                # 使用对数刻度更好地显示学习率变化
                ax_lr.set_yscale('log')
            
            # 6. 模型信息 - 底部，跨越两列
            ax_info = fig.add_subplot(gs[3, :])
            ax_info.axis('off')  # 不显示坐标轴
            
            # 收集模型信息
            model_info = self.get_model_info()
            info_text = (
                f"模型类型: {model_info['model_type']}\n"
                f"隐藏层大小: {model_info.get('hidden_size', 'N/A')}, 层数: {model_info.get('num_layers', 'N/A')}, "
                f"双向: {model_info.get('bidirectional', False)}, 注意力: {model_info.get('use_attention', False)}\n"
                f"Dropout: {model_info.get('dropout', 'N/A')}, 学习率: {model_info.get('learning_rate', 'N/A')}, "
                f"权重衰减: {model_info.get('weight_decay', 'N/A')}, 梯度裁剪: {getattr(self, 'grad_clip_value', 'N/A')}\n"
                f"参数总数: {model_info.get('total_parameters', 0):,}, 可训练参数: {model_info.get('trainable_parameters', 0):,}\n"
                f"设备: {model_info.get('device', 'N/A')}, 训练状态: {'已训练' if model_info.get('is_trained', False) else '未训练'}"
            )
            
            # 添加训练结果摘要
            if has_val:
                best_epoch = val_losses.index(min(val_losses)) + 1
                final_val_loss = val_losses[-1]
                final_val_red_acc = val_red_acc[-1]
                final_val_blue_acc = val_blue_acc[-1]
                
                summary_text = (
                    f"\n\n训练摘要:\n"
                    f"最佳Epoch: {best_epoch}, 最终验证损失: {final_val_loss:.4f}\n"
                    f"最终红球准确率: {final_val_red_acc:.4f}, 最终蓝球准确率: {final_val_blue_acc:.4f}\n"
                    f"最高红球准确率: {max(val_red_acc):.4f}, 最高蓝球准确率: {max(val_blue_acc):.4f}"
                )
                info_text += summary_text
            
            ax_info.text(0.5, 0.5, info_text, ha='center', va='center', fontsize=12, 
                        bbox=dict(boxstyle="round,pad=1", facecolor="#f0f0f0", alpha=0.8, edgecolor="#cccccc"))
            
            # 调整布局
            plt.tight_layout()
            fig.subplots_adjust(hspace=0.3, wspace=0.3)
            
            # 添加总标题
            fig.suptitle('LSTM时间步模型训练历史', fontsize=16, fontweight='bold', y=0.98)
            plt.subplots_adjust(top=0.95)
            
            # 保存图形
            if save_path:
                # 确保目录存在
                os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                self.log(f"训练历史图已保存到 {save_path}")
            
            # 显示图形
            if show:
                plt.show()
            else:
                plt.close()
                
            return True
                
        except Exception as e:
            self.log(f"绘制训练历史时发生错误: {str(e)}")
            import traceback
            self.log(traceback.format_exc())
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