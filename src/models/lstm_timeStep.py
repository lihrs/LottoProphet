# -*- coding: utf-8 -*-
"""
LSTM TimeStep model implementation for lottery prediction
优化的LSTM模型，包含时间注意力机制、残差连接和多头输出
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Optional, Tuple, List, Any, Dict
from .base import BaseMLModel
from src.utils.device_utils import check_device_availability

class LSTMTimeStep(nn.Module):
    """
    LSTM TimeStep layer for sequence modeling with focus on time dependencies
    """
    
    def __init__(self, input_size: int, hidden_size: int, num_layers: int = 1, 
                 dropout: float = 0.0, bidirectional: bool = False, batch_first: bool = True):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.bidirectional = bidirectional
        self.batch_first = batch_first
        
        # LSTM layer
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional,
            batch_first=batch_first
        )
        
        # Output dimension
        self.output_size = hidden_size * 2 if bidirectional else hidden_size
        
        # Time-aware attention mechanism
        self.time_attention = nn.Sequential(
            nn.Linear(self.output_size, self.output_size),
            nn.Tanh(),
            nn.Linear(self.output_size, 1)
        )
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through LSTM with time attention
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, input_size)
            
        Returns:
            Tuple of (output, attention_weights)
        """
        # LSTM forward pass
        lstm_out, (hidden, cell) = self.lstm(x)  # (batch_size, seq_len, hidden_size)
        
        # Apply time attention
        attention_scores = self.time_attention(lstm_out)  # (batch_size, seq_len, 1)
        attention_weights = torch.softmax(attention_scores, dim=1)  # (batch_size, seq_len, 1)
        
        # Weighted sum of LSTM outputs
        context_vector = torch.sum(lstm_out * attention_weights, dim=1)  # (batch_size, hidden_size)
        
        return context_vector, attention_weights


class LSTMTimeStepModel(BaseMLModel, nn.Module):
    """
    优化的LSTM时间步模型，用于彩票号码预测
    包含时间注意力机制、残差连接和多头输出结构
    """
    
    def __init__(self, lottery_type='dlt', feature_window=10, log_callback=None, use_gpu=False,
                 hidden_size=128, num_layers=2, dropout=0.3, bidirectional=True,
                 use_time_attention=True, use_residual=True, learning_rate=0.001):
        """
        初始化LSTM TimeStep模型
        
        Args:
            lottery_type: 彩票类型 ('dlt' 或 'ssq')
            feature_window: 特征窗口大小
            log_callback: 日志回调函数
            use_gpu: 是否使用GPU
            hidden_size: LSTM隐藏层大小
            num_layers: LSTM层数
            dropout: Dropout比率
            bidirectional: 是否使用双向LSTM
            use_time_attention: 是否使用时间注意力机制
            use_residual: 是否使用残差连接
            learning_rate: 学习率
        """
        # 初始化基类
        BaseMLModel.__init__(self, lottery_type, feature_window, log_callback, use_gpu)
        nn.Module.__init__(self)
        
        # 模型参数
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.bidirectional = bidirectional
        self.use_time_attention = use_time_attention
        self.use_residual = use_residual
        self.learning_rate = learning_rate
        
        # 设备配置
        self.device = torch.device('cuda' if use_gpu and torch.cuda.is_available() else 'cpu')
        
        # 计算输入特征大小（红球数量 + 蓝球数量）
        self.input_size = self.red_count + self.blue_count
        
        # 构建网络结构
        self._build_network()
        
        # 移动模型到指定设备
        self.to(self.device)
        
        # 初始化优化器
        self.optimizer = optim.Adam(self.parameters(), lr=self.learning_rate, weight_decay=1e-5)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=10
        )
        
        # 损失函数
        self.criterion = nn.CrossEntropyLoss()
        
        # 训练历史
        self.training_history = {'loss': [], 'red_accuracy': [], 'blue_accuracy': []}
        
        # 训练状态
        self.is_trained = False
        
    def _build_network(self):
        """
        构建网络结构
        """
        # LSTM时间步层
        self.lstm_layer = LSTMTimeStep(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            dropout=self.dropout,
            bidirectional=self.bidirectional
        )
        
        # 计算LSTM输出维度
        lstm_output_size = self.hidden_size * 2 if self.bidirectional else self.hidden_size
        
        # 残差连接投影层
        if self.use_residual:
            self.residual_projection = nn.Linear(self.input_size, lstm_output_size)
        
        # 特征提取层
        self.feature_extractor = nn.Sequential(
            nn.Linear(lstm_output_size, lstm_output_size * 2),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(lstm_output_size * 2, lstm_output_size),
            nn.ReLU(),
            nn.Dropout(self.dropout)
        )
        
        # 红球预测头（多个输出头，每个位置一个）
        self.red_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(lstm_output_size, lstm_output_size // 2),
                nn.ReLU(),
                nn.Dropout(self.dropout // 2),
                nn.Linear(lstm_output_size // 2, self.red_range)
            ) for _ in range(self.red_count)
        ])
        
        # 蓝球预测头
        self.blue_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(lstm_output_size, lstm_output_size // 2),
                nn.ReLU(),
                nn.Dropout(self.dropout // 2),
                nn.Linear(lstm_output_size // 2, self.blue_range)
            ) for _ in range(self.blue_count)
        ])
        
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        前向传播
        
        Args:
            x: 输入张量，形状为 (batch_size, seq_len, input_size)
            
        Returns:
            包含红球和蓝球预测结果的字典
        """
        batch_size, seq_len, _ = x.shape
        
        # LSTM时间步处理
        if self.use_time_attention:
            lstm_out, attention_weights = self.lstm_layer(x)
        else:
            # 不使用注意力机制，直接使用最后时间步的输出
            lstm_out, _ = self.lstm_layer.lstm(x)
            lstm_out = lstm_out[:, -1, :]  # 取最后一个时间步
            attention_weights = None
        
        # 残差连接
        if self.use_residual:
            residual = self.residual_projection(x[:, -1, :])  # 使用最后一个时间步的输入
            lstm_out = lstm_out + residual
        
        # 特征提取
        features = self.feature_extractor(lstm_out)
        
        # 多头预测
        red_outputs = []
        for head in self.red_heads:
            red_outputs.append(head(features))
        
        blue_outputs = []
        for head in self.blue_heads:
            blue_outputs.append(head(features))
        
        return {
            'red_logits': red_outputs,
            'blue_logits': blue_outputs,
            'attention_weights': attention_weights,
            'features': features
        }
            
    def fit(self, data: Any, epochs: int = 100, batch_size: int = 32, 
            validation_split: float = 0.2, early_stopping_patience: int = 15, **kwargs) -> None:
        """
        训练LSTM TimeStep模型
        
        Args:
            data: 训练数据 (pandas DataFrame)
            epochs: 训练轮数
            batch_size: 批次大小
            validation_split: 验证集比例
            early_stopping_patience: 早停耐心值
            **kwargs: 其他训练参数
        """
        self.log(f"开始训练LSTM TimeStep模型，彩票类型: {self.lottery_type}")
        self.log(f"模型参数: hidden_size={self.hidden_size}, num_layers={self.num_layers}, dropout={self.dropout}")
        
        try:
            # 准备训练数据
            X_train, X_val, red_train_data, red_val_data, blue_train_data, blue_val_data = self.prepare_data(
                data, test_size=validation_split
            )
            
            # 转换为序列数据
            X_train_seq = self._prepare_sequence_data(X_train)
            X_val_seq = self._prepare_sequence_data(X_val)
            
            # 转换为PyTorch张量
            X_train_tensor = torch.FloatTensor(X_train_seq).to(self.device)
            X_val_tensor = torch.FloatTensor(X_val_seq).to(self.device)
            
            # 准备标签张量
            red_train_tensors = [torch.LongTensor(red_data).to(self.device) for red_data in red_train_data]
            red_val_tensors = [torch.LongTensor(red_data).to(self.device) for red_data in red_val_data]
            blue_train_tensors = [torch.LongTensor(blue_data).to(self.device) for blue_data in blue_train_data]
            blue_val_tensors = [torch.LongTensor(blue_data).to(self.device) for blue_data in blue_val_data]
            
            # 创建数据加载器
            train_dataset = torch.utils.data.TensorDataset(
                X_train_tensor, *red_train_tensors, *blue_train_tensors
            )
            train_loader = torch.utils.data.DataLoader(
                train_dataset, batch_size=batch_size, shuffle=True
            )
            
            val_dataset = torch.utils.data.TensorDataset(
                X_val_tensor, *red_val_tensors, *blue_val_tensors
            )
            val_loader = torch.utils.data.DataLoader(
                val_dataset, batch_size=batch_size, shuffle=False
            )
            
            # 早停机制
            best_val_loss = float('inf')
            patience_counter = 0
            
            # 训练循环
            for epoch in range(epochs):
                # 训练阶段
                self.train_mode = True
                train_loss, train_red_acc, train_blue_acc = self._train_epoch(train_loader)
                
                # 验证阶段
                self.eval()
                val_loss, val_red_acc, val_blue_acc = self._validate_epoch(val_loader)
                
                # 记录训练历史
                self.training_history['loss'].append({'train': train_loss, 'val': val_loss})
                self.training_history['red_accuracy'].append({'train': train_red_acc, 'val': val_red_acc})
                self.training_history['blue_accuracy'].append({'train': train_blue_acc, 'val': val_blue_acc})
                
                # 学习率调度
                self.scheduler.step(val_loss)
                
                # 早停检查
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    # 保存最佳模型
                    self._save_best_model()
                else:
                    patience_counter += 1
                
                # 日志输出
                if epoch % 10 == 0 or epoch == epochs - 1:
                    self.log(f"Epoch {epoch+1}/{epochs}: "
                           f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, "
                           f"Red Acc: {val_red_acc:.3f}, Blue Acc: {val_blue_acc:.3f}")
                
                # 早停
                if patience_counter >= early_stopping_patience:
                    self.log(f"早停触发，在第{epoch+1}轮停止训练")
                    break
            
            # 加载最佳模型
            self._load_best_model()
            self.is_trained = True
            self.log("模型训练完成")
            
        except Exception as e:
            self.log(f"训练过程中发生错误: {str(e)}")
            raise
    
    def _prepare_sequence_data(self, X: np.ndarray) -> np.ndarray:
        """
        将扁平化的特征数据重新整形为序列数据
        
        Args:
            X: 扁平化的特征数据
            
        Returns:
            序列化的特征数据
        """
        # X的形状应该是 (samples, feature_window * input_size)
        # 需要重新整形为 (samples, feature_window, input_size)
        samples = X.shape[0]
        return X.reshape(samples, self.feature_window, self.input_size)
    
    def _train_epoch(self, train_loader) -> Tuple[float, float, float]:
        """
        训练一个epoch
        
        Returns:
            Tuple of (loss, red_accuracy, blue_accuracy)
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
                loss = self.criterion(red_logit, red_target)
                total_batch_loss += loss
                
                # 计算准确率
                _, predicted = torch.max(red_logit, 1)
                red_correct += (predicted == red_target).sum().item()
            
            # 蓝球损失
            for i, (blue_logit, blue_target) in enumerate(zip(outputs['blue_logits'], blue_targets)):
                loss = self.criterion(blue_logit, blue_target)
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
    
    def _validate_epoch(self, val_loader) -> Tuple[float, float, float]:
        """
        验证一个epoch
        
        Returns:
            Tuple of (loss, red_accuracy, blue_accuracy)
        """
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
                total_batch_loss = 0.0
                
                # 红球损失和准确率
                for i, (red_logit, red_target) in enumerate(zip(outputs['red_logits'], red_targets)):
                    loss = self.criterion(red_logit, red_target)
                    total_batch_loss += loss
                    
                    _, predicted = torch.max(red_logit, 1)
                    red_correct += (predicted == red_target).sum().item()
                
                # 蓝球损失和准确率
                for i, (blue_logit, blue_target) in enumerate(zip(outputs['blue_logits'], blue_targets)):
                    loss = self.criterion(blue_logit, blue_target)
                    total_batch_loss += loss
                    
                    _, predicted = torch.max(blue_logit, 1)
                    blue_correct += (predicted == blue_target).sum().item()
                
                total_loss += total_batch_loss.item()
                total_samples += X_batch.size(0)
        
        avg_loss = total_loss / len(val_loader)
        red_accuracy = red_correct / (total_samples * self.red_count)
        blue_accuracy = blue_correct / (total_samples * self.blue_count)
        
        return avg_loss, red_accuracy, blue_accuracy
    
    def predict(self, recent_data: Optional[np.ndarray] = None, **kwargs) -> Tuple[List[int], List[int]]:
        """
        使用训练好的模型进行预测
        
        Args:
            recent_data: 最近的历史数据，如果为None则使用训练时的最后几期数据
            **kwargs: 其他预测参数
            
        Returns:
            Tuple of (red_numbers, blue_numbers)
        """
        if not self.is_trained:
            raise ValueError("模型必须先训练才能进行预测")
        
        self.eval()
        
        with torch.no_grad():
            if recent_data is None:
                # 使用训练时保存的最后几期数据
                if not hasattr(self, '_last_sequence'):
                    raise ValueError("没有可用的历史数据进行预测")
                input_data = self._last_sequence
            else:
                # 处理DataFrame输入
                if hasattr(recent_data, 'values'):  # DataFrame
                    # 使用prepare_data方法处理数据
                    X_train, X_test, _, _, _, _ = self.prepare_data(recent_data)
                    # 合并训练和测试数据
                    X_all = np.vstack([X_train, X_test]) if len(X_test) > 0 else X_train
                    if len(X_all) == 0:
                        raise ValueError("提供的数据不足以生成预测序列")
                    # 重新整形为序列数据
                    input_data = X_all[-1:].reshape(1, self.feature_window, self.input_size)
                elif isinstance(recent_data, np.ndarray):
                    # 使用提供的数据
                    if len(recent_data.shape) == 1:
                        # 如果是一维数据，需要重新整形
                        input_data = recent_data.reshape(1, self.feature_window, self.input_size)
                    else:
                        input_data = recent_data
                else:
                    raise ValueError("不支持的数据类型")
            
            # 转换为张量
            input_tensor = torch.FloatTensor(input_data).to(self.device)
            
            # 前向传播
            outputs = self.forward(input_tensor)
            
            # 获取预测结果
            red_numbers = []
            blue_numbers = []
            
            # 红球预测
            for red_logit in outputs['red_logits']:
                probabilities = torch.softmax(red_logit, dim=1)
                # 使用概率采样而不是简单的argmax，增加随机性
                if np.random.random() < 0.3:  # 30%的概率使用随机采样
                    predicted = torch.multinomial(probabilities, 1).squeeze().item()
                else:
                    predicted = torch.argmax(probabilities, dim=1).item()
                red_numbers.append(predicted + 1)  # 转换回1-based索引
            
            # 蓝球预测
            for blue_logit in outputs['blue_logits']:
                probabilities = torch.softmax(blue_logit, dim=1)
                if np.random.random() < 0.3:  # 30%的概率使用随机采样
                    predicted = torch.multinomial(probabilities, 1).squeeze().item()
                else:
                    predicted = torch.argmax(probabilities, dim=1).item()
                blue_numbers.append(predicted + 1)  # 转换回1-based索引
            
            # 确保红球号码不重复（对于需要不重复的彩票类型）
            if len(set(red_numbers)) != len(red_numbers):
                # 如果有重复，使用更智能的方法
                red_numbers = self._ensure_unique_numbers(outputs['red_logits'], self.red_range)
            
            # 确保号码在有效范围内
            red_numbers = [max(1, min(num, self.red_range)) for num in red_numbers]
            blue_numbers = [max(1, min(num, self.blue_range)) for num in blue_numbers]
            
            self.log(f"预测结果 - 红球: {red_numbers}, 蓝球: {blue_numbers}")
            
            return red_numbers, blue_numbers
    
    def _ensure_unique_numbers(self, red_logits: List[torch.Tensor], red_range: int) -> List[int]:
        """
        确保红球号码不重复
        
        Args:
            red_logits: 红球预测logits
            red_range: 红球范围
            
        Returns:
            不重复的红球号码列表
        """
        selected_numbers = []
        used_numbers = set()
        
        for logit in red_logits:
            probabilities = torch.softmax(logit, dim=1).squeeze()
            
            # 按概率排序
            sorted_indices = torch.argsort(probabilities, descending=True)
            
            # 选择第一个未使用的号码
            for idx in sorted_indices:
                number = idx.item() + 1  # 转换为1-based
                if number not in used_numbers:
                    selected_numbers.append(number)
                    used_numbers.add(number)
                    break
        
        return selected_numbers
        
    def _save_best_model(self) -> None:
        """
        保存当前最佳模型到临时文件
        """
        import tempfile
        import os
        
        if not hasattr(self, '_best_model_path'):
            # 创建临时文件
            temp_dir = tempfile.gettempdir()
            self._best_model_path = os.path.join(temp_dir, f'best_lstm_model_{self.lottery_type}.pth')
        
        torch.save({
            'model_state_dict': self.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'training_history': self.training_history
        }, self._best_model_path)
    
    def _load_best_model(self) -> None:
        """
        加载最佳模型
        """
        if hasattr(self, '_best_model_path') and os.path.exists(self._best_model_path):
            checkpoint = torch.load(self._best_model_path, map_location=self.device)
            self.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            self.training_history = checkpoint.get('training_history', self.training_history)
    
    def save_models(self, filepath: str) -> None:
        """
        保存训练好的模型
        
        Args:
            filepath: 模型保存路径
        """
        import os
        
        # 确保目录存在
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # 保存模型和相关信息
        torch.save({
            'model_state_dict': self.state_dict(),
            'model_config': {
                'lottery_type': self.lottery_type,
                'feature_window': self.feature_window,
                'hidden_size': self.hidden_size,
                'num_layers': self.num_layers,
                'dropout': self.dropout,
                'bidirectional': self.bidirectional,
                'use_time_attention': self.use_time_attention,
                'use_residual': self.use_residual,
                'learning_rate': self.learning_rate,
                'input_size': self.input_size,
                'red_count': self.red_count,
                'blue_count': self.blue_count,
                'red_range': self.red_range,
                'blue_range': self.blue_range
            },
            'optimizer_state_dict': self.optimizer.state_dict() if hasattr(self, 'optimizer') else None,
            'scheduler_state_dict': self.scheduler.state_dict() if hasattr(self, 'scheduler') else None,
            'training_history': self.training_history,
            'scalers': self.scalers if hasattr(self, 'scalers') else {},
            'is_trained': self.is_trained
        }, filepath)
        
        self.log(f"模型已保存到: {filepath}")
        
    def load_model(self, filepath: str) -> None:
        """
        加载训练好的模型
        
        Args:
            filepath: 模型文件路径
        """
        import os
        
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"模型文件不存在: {filepath}")
        
        try:
            # 加载检查点
            checkpoint = torch.load(filepath, map_location=self.device)
            
            # 加载模型配置
            if 'model_config' in checkpoint:
                config = checkpoint['model_config']
                
                # 更新模型参数
                self.hidden_size = config.get('hidden_size', self.hidden_size)
                self.num_layers = config.get('num_layers', self.num_layers)
                self.dropout = config.get('dropout', self.dropout)
                self.bidirectional = config.get('bidirectional', self.bidirectional)
                self.use_time_attention = config.get('use_time_attention', self.use_time_attention)
                self.use_residual = config.get('use_residual', self.use_residual)
                self.learning_rate = config.get('learning_rate', self.learning_rate)
                
                # 重新构建网络（如果参数发生变化）
                self._build_network()
                self.to(self.device)
            
            # 加载模型权重
            self.load_state_dict(checkpoint['model_state_dict'])
            
            # 加载训练状态
            self.is_trained = checkpoint.get('is_trained', False)
            self.training_history = checkpoint.get('training_history', {'loss': [], 'red_accuracy': [], 'blue_accuracy': []})
            
            # 加载缩放器
            if 'scalers' in checkpoint:
                self.scalers = checkpoint['scalers']
            
            # 重新初始化优化器和调度器
            if self.is_trained:
                self.optimizer = optim.Adam(self.parameters(), lr=self.learning_rate, weight_decay=1e-5)
                self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                    self.optimizer, mode='min', factor=0.5, patience=10, verbose=True
                )
                
                # 如果有保存的优化器状态，则加载
                if 'optimizer_state_dict' in checkpoint and checkpoint['optimizer_state_dict'] is not None:
                    self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                
                if 'scheduler_state_dict' in checkpoint and checkpoint['scheduler_state_dict'] is not None:
                    self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            
            self.log(f"模型已从 {filepath} 加载完成")
            
        except Exception as e:
            self.log(f"加载模型时发生错误: {str(e)}")
            raise
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        获取模型信息
        
        Returns:
            包含模型信息的字典
        """
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            'model_type': 'LSTM TimeStep',
            'lottery_type': self.lottery_type,
            'feature_window': self.feature_window,
            'hidden_size': self.hidden_size,
            'num_layers': self.num_layers,
            'dropout': self.dropout,
            'bidirectional': self.bidirectional,
            'use_time_attention': self.use_time_attention,
            'use_residual': self.use_residual,
            'learning_rate': self.learning_rate,
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'device': str(self.device),
            'is_trained': self.is_trained,
            'training_epochs': len(self.training_history['loss']) if self.training_history['loss'] else 0
        }
    
    def save_models(self) -> None:
        """
        保存模型到默认路径
        """
        # 创建模型目录
        model_dir = os.path.join(self.models_dir, self.model_type)
        os.makedirs(model_dir, exist_ok=True)
        
        # 保存模型文件
        model_path = os.path.join(model_dir, f"{self.lottery_type}_model.pth")
        self.save_model(model_path)
        
        return True
    
    def load_models(self) -> bool:
        """
        从默认路径加载模型
        
        Returns:
            bool: 是否成功加载模型
        """
        model_dir = os.path.join(self.models_dir, self.model_type)
        model_path = os.path.join(model_dir, f"{self.lottery_type}_model.pth")
        
        if not os.path.exists(model_path):
            self.log(f"模型文件不存在: {model_path}")
            return False
            
        try:
            self.load_model(model_path)
            return True
        except Exception as e:
            self.log(f"加载模型时发生错误: {str(e)}")
            return False
    
    def plot_training_history(self, save_path: Optional[str] = None) -> None:
        """
        绘制训练历史
        
        Args:
            save_path: 保存图片的路径，如果为None则显示图片
        """
        try:
            import matplotlib.pyplot as plt
            
            if not self.training_history['loss']:
                self.log("没有训练历史数据可以绘制")
                return
            
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            fig.suptitle(f'LSTM TimeStep Model Training History - {self.lottery_type.upper()}', fontsize=16)
            
            epochs = range(1, len(self.training_history['loss']) + 1)
            
            # 损失曲线
            axes[0, 0].plot(epochs, [h['train'] for h in self.training_history['loss']], 'b-', label='Train Loss')
            axes[0, 0].plot(epochs, [h['val'] for h in self.training_history['loss']], 'r-', label='Val Loss')
            axes[0, 0].set_title('Loss')
            axes[0, 0].set_xlabel('Epoch')
            axes[0, 0].set_ylabel('Loss')
            axes[0, 0].legend()
            axes[0, 0].grid(True)
            
            # 红球准确率
            axes[0, 1].plot(epochs, [h['train'] for h in self.training_history['red_accuracy']], 'b-', label='Train Red Acc')
            axes[0, 1].plot(epochs, [h['val'] for h in self.training_history['red_accuracy']], 'r-', label='Val Red Acc')
            axes[0, 1].set_title('Red Ball Accuracy')
            axes[0, 1].set_xlabel('Epoch')
            axes[0, 1].set_ylabel('Accuracy')
            axes[0, 1].legend()
            axes[0, 1].grid(True)
            
            # 蓝球准确率
            axes[1, 0].plot(epochs, [h['train'] for h in self.training_history['blue_accuracy']], 'b-', label='Train Blue Acc')
            axes[1, 0].plot(epochs, [h['val'] for h in self.training_history['blue_accuracy']], 'r-', label='Val Blue Acc')
            axes[1, 0].set_title('Blue Ball Accuracy')
            axes[1, 0].set_xlabel('Epoch')
            axes[1, 0].set_ylabel('Accuracy')
            axes[1, 0].legend()
            axes[1, 0].grid(True)
            
            # 学习率（如果有记录）
            if hasattr(self.scheduler, 'get_last_lr'):
                current_lr = self.scheduler.get_last_lr()[0]
                axes[1, 1].axhline(y=current_lr, color='g', linestyle='--', label=f'Current LR: {current_lr:.6f}')
                axes[1, 1].set_title('Learning Rate')
                axes[1, 1].set_xlabel('Epoch')
                axes[1, 1].set_ylabel('Learning Rate')
                axes[1, 1].legend()
                axes[1, 1].grid(True)
            else:
                axes[1, 1].text(0.5, 0.5, 'Learning Rate\nHistory Not Available', 
                               ha='center', va='center', transform=axes[1, 1].transAxes)
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                self.log(f"训练历史图已保存到: {save_path}")
            else:
                plt.show()
                
        except ImportError:
            self.log("matplotlib未安装，无法绘制训练历史")
        except Exception as e:
            self.log(f"绘制训练历史时发生错误: {str(e)}")