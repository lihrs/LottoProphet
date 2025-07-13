#!/usr/bin/env python
import torch
import torch.nn as nn
import torch.nn.functional as F
from TorchCRF import CRF

class LstmCRFModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, output_seq_length, num_layers=1, dropout=0.5, 
                 bidirectional=True, attention=True, residual=False):
        super(LstmCRFModel, self).__init__()
        
        # 配置参数
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.output_seq_length = output_seq_length
        self.bidirectional = bidirectional
        self.attention = attention
        self.residual = residual
        
        # LSTM层
        self.lstm = nn.LSTM(
            input_dim, 
            hidden_dim, 
            num_layers=num_layers, 
            batch_first=True, 
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional
        )
        
        # 计算LSTM输出维度
        lstm_output_dim = hidden_dim * 2 if bidirectional else hidden_dim
        
        # 注意力机制
        if attention:
            self.attention_layer = nn.Sequential(
                nn.Linear(lstm_output_dim, lstm_output_dim // 2),
                nn.Tanh(),
                nn.Linear(lstm_output_dim // 2, 1)
            )
        
        # 全连接层
        self.fc = nn.Linear(lstm_output_dim, output_dim * output_seq_length)
        
        # 残差连接的额外层
        if residual:
            self.residual_layer = nn.Linear(input_dim, lstm_output_dim)
        
        # CRF层
        self.crf = CRF(output_dim)
        
        # Dropout层
        self.dropout = nn.Dropout(dropout)

    def _apply_attention(self, lstm_out):
        # 计算注意力权重
        attn_weights = self.attention_layer(lstm_out)
        attn_weights = F.softmax(attn_weights, dim=1)
        
        # 应用注意力权重
        context = torch.bmm(attn_weights.transpose(1, 2), lstm_out)
        return context.squeeze(1)

    def forward(self, x, labels=None, mask=None, return_emissions=False):
        # LSTM前向传播
        lstm_out, _ = self.lstm(x)
        
        # 应用dropout
        lstm_out = self.dropout(lstm_out)
        
        # 应用残差连接
        if self.residual and x.size(2) == lstm_out.size(2):
            lstm_out = lstm_out + x
        
        # 提取特征表示
        if self.attention:
            # 使用注意力机制
            features = self._apply_attention(lstm_out)
        else:
            # 使用最后一个时间步的隐藏状态
            features = lstm_out[:, -1, :]
        
        # 生成logits
        logits = self.fc(features)
        logits = logits.view(-1, self.output_seq_length, self.output_dim)
        
        # 训练模式：计算损失
        if labels is not None:
            if mask is not None:
                mask = mask.bool()
            loss = -self.crf(logits, labels, mask=mask).mean()  # 对每个样本的损失求平均
            return loss
        # 预测模式
        else:
            predictions = self.crf.viterbi_decode(logits, mask=mask)
            if return_emissions:
                # 返回预测结果和发射概率
                return predictions, logits
            return predictions
    
    def get_emissions(self, x):
        """获取发射概率矩阵，用于采样和概率分析"""
        with torch.no_grad():
            lstm_out, _ = self.lstm(x)
            lstm_out = self.dropout(lstm_out)
            
            if self.residual and x.size(2) == lstm_out.size(2):
                lstm_out = lstm_out + x
            
            if self.attention:
                features = self._apply_attention(lstm_out)
            else:
                features = lstm_out[:, -1, :]
            
            logits = self.fc(features)
            logits = logits.view(-1, self.output_seq_length, self.output_dim)
            
            # 应用softmax获取概率分布
            emissions = F.softmax(logits, dim=-1)
            return emissions