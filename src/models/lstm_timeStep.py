# -*- coding: utf-8 -*-
"""
Advanced LSTM TimeStep model implementation for lottery prediction
高级LSTM时间步模型，集成多种优化技术提升预测准确率
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, Tuple, List, Any, Dict
from .base import BaseMLModel
from src.utils.device_utils import check_device_availability
import math

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
                 hidden_size=256, num_layers=3, dropout=0.2, bidirectional=True,
                 use_attention=True, learning_rate=0.0005, weight_decay=1e-4):
        """
        初始化高级LSTM TimeStep模型
        """
        BaseMLModel.__init__(self, lottery_type, feature_window, log_callback, use_gpu)
        nn.Module.__init__(self)
        
        # 优化的模型参数
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.bidirectional = bidirectional
        self.use_attention = use_attention
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        
        # 设备配置
        self.device = torch.device('cuda' if use_gpu and torch.cuda.is_available() else 'cpu')
        
        # 输入特征大小
        self.input_size = self.red_count + self.blue_count
        
        # 构建网络
        self._build_network()
        
        # 移动到设备
        self.to(self.device)
        
        # 优化器配置
        self.optimizer = optim.AdamW(
            self.parameters(), 
            lr=self.learning_rate, 
            weight_decay=self.weight_decay,
            betas=(0.9, 0.999)
        )
        
        # 学习率调度器
        self.scheduler = optim.lr_scheduler.OneCycleLR(
            self.optimizer,
            max_lr=self.learning_rate * 10,
            epochs=200,
            steps_per_epoch=100,
            pct_start=0.1,
            anneal_strategy='cos'
        )
        
        # 损失函数
        self.criterion = LotteryLoss(alpha=0.7, beta=0.3)
        
        # 训练状态
        self.training_history = {'loss': [], 'red_accuracy': [], 'blue_accuracy': []}
        self.is_trained = False
        self.best_model_state = None
        
        # 历史频率统计
        self.red_freq = None
        self.blue_freq = None
        
    def _build_network(self):
        """
        构建优化的网络架构
        """
        # 输入嵌入层
        self.input_embedding = nn.Sequential(
            nn.Linear(self.input_size, self.hidden_size),
            nn.GELU(),
            nn.Dropout(self.dropout)
        )
        
        # 增强LSTM层
        self.lstm_layer = EnhancedLSTMLayer(
            input_size=self.hidden_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            dropout=self.dropout,
            bidirectional=self.bidirectional,
            use_attention=self.use_attention
        )
        
        # 特征融合层
        lstm_output_size = self.hidden_size * 2 if self.bidirectional else self.hidden_size
        self.feature_fusion = nn.Sequential(
            nn.Linear(lstm_output_size, lstm_output_size),
            nn.GELU(),
            nn.Dropout(self.dropout),
            nn.Linear(lstm_output_size, lstm_output_size)
        )
        
        # 红球预测头
        self.red_heads = nn.ModuleList([
            AdaptivePredictionHead(lstm_output_size, self.red_range, self.dropout)
            for _ in range(self.red_count)
        ])
        
        # 蓝球预测头
        self.blue_heads = nn.ModuleList([
            AdaptivePredictionHead(lstm_output_size, self.blue_range, self.dropout)
            for _ in range(self.blue_count)
        ])
        
        # 全局特征提取器
        self.global_feature_extractor = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(lstm_output_size, lstm_output_size // 2),
            nn.GELU(),
            nn.Dropout(self.dropout)
        )
        
    def forward(self, x):
        """
        前向传播
        """
        batch_size, seq_len, _ = x.shape
        
        # 输入嵌入
        embedded = self.input_embedding(x)
        
        # LSTM处理
        lstm_out, attention_weights = self.lstm_layer(embedded)
        
        # 特征融合
        fused_features = self.feature_fusion(lstm_out)
        
        # 全局特征
        global_features = self.global_feature_extractor(fused_features.transpose(1, 2))
        
        # 序列特征（使用最后时间步）
        sequence_features = fused_features[:, -1, :]
        
        # 组合特征
        combined_features = sequence_features + global_features.unsqueeze(1).expand(-1, sequence_features.size(1))
        
        # 多头预测
        red_outputs = [head(combined_features) for head in self.red_heads]
        blue_outputs = [head(combined_features) for head in self.blue_heads]
        
        return {
            'red_logits': red_outputs,
            'blue_logits': blue_outputs,
            'attention_weights': attention_weights,
            'features': combined_features
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
    
    def fit(self, data, epochs=200, batch_size=64, validation_split=0.2, 
            early_stopping_patience=25, **kwargs):
        """
        训练模型
        """
        self.log(f"开始训练高级LSTM TimeStep模型，彩票类型: {self.lottery_type}")
        self.log(f"优化参数: hidden_size={self.hidden_size}, num_layers={self.num_layers}")
        self.log(f"训练参数: epochs={epochs}, batch_size={batch_size}, lr={self.learning_rate}")
        
        try:
            # 计算历史频率
            self.red_freq, self.blue_freq = self._compute_historical_frequency(data)
            
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
            
            # 数据加载器
            train_dataset = torch.utils.data.TensorDataset(
                X_train_tensor, *red_train_tensors, *blue_train_tensors
            )
            train_loader = torch.utils.data.DataLoader(
                train_dataset, batch_size=batch_size, shuffle=True, drop_last=True
            )
            
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
                val_loss, val_red_acc, val_blue_acc = self._validate_epoch(val_loader)
                
                # 记录历史
                self.training_history['loss'].append({'train': train_loss, 'val': val_loss})
                self.training_history['red_accuracy'].append({'train': train_red_acc, 'val': val_red_acc})
                self.training_history['blue_accuracy'].append({'train': train_blue_acc, 'val': val_blue_acc})
                
                # 学习率调度
                self.scheduler.step()
                
                # 早停检查
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    self.best_model_state = self.state_dict().copy()
                else:
                    patience_counter += 1
                
                # 日志输出
                if epoch % 20 == 0 or epoch == epochs - 1:
                    current_lr = self.scheduler.get_last_lr()[0]
                    self.log(f"Epoch {epoch+1}/{epochs}: Train Loss: {train_loss:.4f}, "
                           f"Val Loss: {val_loss:.4f}, Red Acc: {val_red_acc:.3f}, "
                           f"Blue Acc: {val_blue_acc:.3f}, LR: {current_lr:.6f}")
                
                # 早停
                if patience_counter >= early_stopping_patience:
                    self.log(f"早停触发，在第{epoch+1}轮停止训练")
                    break
            
            # 加载最佳模型
            if self.best_model_state is not None:
                self.load_state_dict(self.best_model_state)
            
            self.is_trained = True
            self.log("模型训练完成")
            
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
        训练一个epoch
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
            
            self.optimizer.zero_grad()
            
            # 前向传播
            outputs = self.forward(X_batch)
            
            # 计算损失
            total_batch_loss = 0.0
            
            # 红球损失
            for i, (red_logit, red_target) in enumerate(zip(outputs['red_logits'], red_targets)):
                loss = self.criterion(red_logit, red_target, self.red_freq)
                total_batch_loss += loss
                
                # 计算准确率
                _, predicted = torch.max(red_logit, 1)
                red_correct += (predicted == red_target).sum().item()
            
            # 蓝球损失
            for i, (blue_logit, blue_target) in enumerate(zip(outputs['blue_logits'], blue_targets)):
                loss = self.criterion(blue_logit, blue_target, self.blue_freq)
                total_batch_loss += loss
                
                # 计算准确率
                _, predicted = torch.max(blue_logit, 1)
                blue_correct += (predicted == blue_target).sum().item()
            
            # 反向传播
            total_batch_loss.backward()
            
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            total_loss += total_batch_loss.item()
            total_samples += X_batch.size(0)
        
        avg_loss = total_loss / len(train_loader)
        red_accuracy = red_correct / (total_samples * self.red_count)
        blue_accuracy = blue_correct / (total_samples * self.blue_count)
        
        return avg_loss, red_accuracy, blue_accuracy
    
    def _validate_epoch(self, val_loader):
        """
        验证一个epoch
        """
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
                
                outputs = self.forward(X_batch)
                
                total_batch_loss = 0.0
                
                # 红球损失和准确率
                for i, (red_logit, red_target) in enumerate(zip(outputs['red_logits'], red_targets)):
                    loss = self.criterion(red_logit, red_target, self.red_freq)
                    total_batch_loss += loss
                    
                    _, predicted = torch.max(red_logit, 1)
                    red_correct += (predicted == red_target).sum().item()
                
                # 蓝球损失和准确率
                for i, (blue_logit, blue_target) in enumerate(zip(outputs['blue_logits'], blue_targets)):
                    loss = self.criterion(blue_logit, blue_target, self.blue_freq)
                    total_batch_loss += loss
                    
                    _, predicted = torch.max(blue_logit, 1)
                    blue_correct += (predicted == blue_target).sum().item()
                
                total_loss += total_batch_loss.item()
                total_samples += X_batch.size(0)
        
        avg_loss = total_loss / len(val_loader)
        red_accuracy = red_correct / (total_samples * self.red_count)
        blue_accuracy = blue_correct / (total_samples * self.blue_count)
        
        return avg_loss, red_accuracy, blue_accuracy
    
    def predict(self, recent_data=None, num_predictions=1, **kwargs):
        """
        预测彩票号码
        """
        if not self.is_trained:
            raise ValueError("模型必须先训练才能进行预测")
        
        self.eval()
        predictions = []
        
        with torch.no_grad():
            for _ in range(num_predictions):
                if recent_data is None:
                    if not hasattr(self, '_last_sequence'):
                        raise ValueError("没有可用的历史数据进行预测")
                    input_data = self._last_sequence
                else:
                    # 处理输入数据
                    if hasattr(recent_data, 'values'):
                        X_train, X_test, _, _, _, _ = self.prepare_data(recent_data)
                        X_all = np.vstack([X_train, X_test]) if len(X_test) > 0 else X_train
                        input_data = X_all[-1:].reshape(1, self.feature_window, self.input_size)
                    else:
                        input_data = recent_data
                
                # 转换为张量
                input_tensor = torch.FloatTensor(input_data).to(self.device)
                
                # 前向传播
                outputs = self.forward(input_tensor)
                
                # 获取预测结果
                red_numbers = []
                blue_numbers = []
                
                # 红球预测（使用温度采样增加多样性）
                for red_logit in outputs['red_logits']:
                    probabilities = F.softmax(red_logit / 1.2, dim=1)  # 温度采样
                    predicted = torch.multinomial(probabilities, 1).squeeze().item()
                    red_numbers.append(predicted + 1)
                
                # 蓝球预测
                for blue_logit in outputs['blue_logits']:
                    probabilities = F.softmax(blue_logit / 1.2, dim=1)
                    predicted = torch.multinomial(probabilities, 1).squeeze().item()
                    blue_numbers.append(predicted + 1)
                
                # 确保红球号码唯一
                red_numbers = self._ensure_unique_red_numbers(red_numbers)
                
                predictions.append((red_numbers, blue_numbers))
        
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
    
    def save_models(self, filepath):
        """
        保存模型
        """
        try:
            model_data = {
                'model_state_dict': self.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
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
                    'weight_decay': self.weight_decay
                },
                'red_freq': self.red_freq,
                'blue_freq': self.blue_freq,
                'is_trained': self.is_trained
            }
            
            if hasattr(self, '_last_sequence'):
                model_data['last_sequence'] = self._last_sequence
            
            torch.save(model_data, filepath)
            self.log(f"模型已保存到: {filepath}")
            
        except Exception as e:
            self.log(f"保存模型时发生错误: {str(e)}")
            raise
    
    def load_model(self, filepath):
        """
        加载模型
        """
        try:
            model_data = torch.load(filepath, map_location=self.device)
            
            # 重建网络
            params = model_data['model_params']
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
            
            # 加载状态
            self.load_state_dict(model_data['model_state_dict'])
            self.optimizer.load_state_dict(model_data['optimizer_state_dict'])
            self.training_history = model_data['training_history']
            self.red_freq = model_data.get('red_freq')
            self.blue_freq = model_data.get('blue_freq')
            self.is_trained = model_data.get('is_trained', False)
            
            if 'last_sequence' in model_data:
                self._last_sequence = model_data['last_sequence']
            
            self.log(f"模型已从 {filepath} 加载")
            
        except Exception as e:
            self.log(f"加载模型时发生错误: {str(e)}")
            raise
    
    def get_model_info(self):
        """
        获取模型信息
        """
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
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
            'is_trained': self.is_trained
        }
    
    def plot_training_history(self, save_path=None):
        """
        绘制训练历史
        """
        if not self.training_history['loss']:
            self.log("没有训练历史可以绘制")
            return
        
        try:
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            
            # 损失曲线
            epochs = range(1, len(self.training_history['loss']) + 1)
            train_losses = [h['train'] for h in self.training_history['loss']]
            val_losses = [h['val'] for h in self.training_history['loss']]
            
            axes[0, 0].plot(epochs, train_losses, 'b-', label='训练损失')
            axes[0, 0].plot(epochs, val_losses, 'r-', label='验证损失')
            axes[0, 0].set_title('损失曲线')
            axes[0, 0].set_xlabel('Epoch')
            axes[0, 0].set_ylabel('Loss')
            axes[0, 0].legend()
            axes[0, 0].grid(True)
            
            # 红球准确率
            train_red_acc = [h['train'] for h in self.training_history['red_accuracy']]
            val_red_acc = [h['val'] for h in self.training_history['red_accuracy']]
            
            axes[0, 1].plot(epochs, train_red_acc, 'b-', label='训练准确率')
            axes[0, 1].plot(epochs, val_red_acc, 'r-', label='验证准确率')
            axes[0, 1].set_title('红球预测准确率')
            axes[0, 1].set_xlabel('Epoch')
            axes[0, 1].set_ylabel('Accuracy')
            axes[0, 1].legend()
            axes[0, 1].grid(True)
            
            # 蓝球准确率
            train_blue_acc = [h['train'] for h in self.training_history['blue_accuracy']]
            val_blue_acc = [h['val'] for h in self.training_history['blue_accuracy']]
            
            axes[1, 0].plot(epochs, train_blue_acc, 'b-', label='训练准确率')
            axes[1, 0].plot(epochs, val_blue_acc, 'r-', label='验证准确率')
            axes[1, 0].set_title('蓝球预测准确率')
            axes[1, 0].set_xlabel('Epoch')
            axes[1, 0].set_ylabel('Accuracy')
            axes[1, 0].legend()
            axes[1, 0].grid(True)
            
            # 综合指标
            combined_acc = [(r['val'] + b['val']) / 2 for r, b in 
                          zip(self.training_history['red_accuracy'], 
                              self.training_history['blue_accuracy'])]
            
            axes[1, 1].plot(epochs, combined_acc, 'g-', label='综合准确率')
            axes[1, 1].set_title('综合预测准确率')
            axes[1, 1].set_xlabel('Epoch')
            axes[1, 1].set_ylabel('Accuracy')
            axes[1, 1].legend()
            axes[1, 1].grid(True)
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                self.log(f"训练历史图已保存到: {save_path}")
            
            plt.show()
            
        except Exception as e:
            self.log(f"绘制训练历史时发生错误: {str(e)}")