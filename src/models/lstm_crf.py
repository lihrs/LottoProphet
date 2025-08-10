# -*- coding: utf-8 -*-
"""
LSTM-CRF model implementation for lottery prediction
"""

import torch
import torch.nn as nn
import numpy as np
import random
from typing import Optional, Tuple, List, Any, Dict, Union
from .base import BaseLotteryModel

# 添加安全的全局变量，以允许numpy._core.multiarray._reconstruct
try:
    # 尝试添加numpy相关的安全全局变量
    torch.serialization.add_safe_globals([
        'numpy._core.multiarray._reconstruct',
        'numpy.core.multiarray._reconstruct',
        'numpy._core._multiarray_umath',
        'numpy.core._multiarray_umath'
    ])
except (AttributeError, ImportError) as e:
    # 兼容旧版PyTorch
    print(f"注意: 无法添加numpy安全全局变量: {str(e)}")
    pass


class CRF(nn.Module):
    """
    Conditional Random Field layer
    """
    
    def __init__(self, num_tags: int, batch_first: bool = False):
        if num_tags <= 0:
            raise ValueError(f"invalid number of tags: {num_tags}")
        super().__init__()
        self.num_tags = num_tags
        self.batch_first = batch_first
        self.start_trans = nn.Parameter(torch.empty(num_tags))
        self.end_trans = nn.Parameter(torch.empty(num_tags))
        self.trans_matrix = nn.Parameter(torch.empty(num_tags, num_tags))
        
        self.reset_parameters()
        
    def reset_parameters(self) -> None:
        """Initialize the transition parameters"""
        nn.init.uniform_(self.start_trans, -0.1, 0.1)
        nn.init.uniform_(self.end_trans, -0.1, 0.1)
        nn.init.uniform_(self.trans_matrix, -0.1, 0.1)
        
    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(num_tags={self.num_tags})'
        
    def forward(self, emissions: torch.Tensor, tags: torch.LongTensor,
                mask: Optional[torch.ByteTensor] = None,
                reduction: str = 'sum') -> torch.Tensor:
        """Compute the conditional log likelihood of a sequence of tags given emission scores"""
        self._validate(emissions, tags=tags, mask=mask)
        if reduction not in ('none', 'sum', 'mean', 'token_mean'):
            raise ValueError(f'invalid reduction: {reduction}')
        if mask is None:
            mask = torch.ones_like(tags, dtype=torch.uint8)
            
        if self.batch_first:
            emissions = emissions.transpose(0, 1)
            tags = tags.transpose(0, 1)
            mask = mask.transpose(0, 1)
            
        # compute the log sum exp of all possible paths
        numerator = self._compute_score(emissions, tags, mask)
        denominator = self._compute_normalizer(emissions, mask)
        llh = numerator - denominator
        
        if reduction == 'none':
            return llh
        elif reduction == 'sum':
            return llh.sum()
        elif reduction == 'mean':
            return llh.mean()
        else:  # reduction == 'token_mean'
            return llh.sum() / mask.float().sum()
            
    def decode(self, emissions: torch.Tensor,
               mask: Optional[torch.ByteTensor] = None) -> List[List[int]]:
        """Find the most likely tag sequence using Viterbi algorithm"""
        self._validate(emissions, mask=mask)
        if mask is None:
            mask = emissions.new_ones(emissions.shape[:2], dtype=torch.uint8)
            
        if self.batch_first:
            emissions = emissions.transpose(0, 1)
            mask = mask.transpose(0, 1)
            
        return self._viterbi_decode(emissions, mask)
        
    def _validate(self, emissions: torch.Tensor, tags: Optional[torch.LongTensor] = None,
                  mask: Optional[torch.ByteTensor] = None) -> None:
        if emissions.dim() != 3:
            raise ValueError(f'emissions must have dimension of 3, got {emissions.dim()}')
        if emissions.size(2) != self.num_tags:
            raise ValueError(
                f'expected last dimension of emissions is {self.num_tags}, '
                f'got {emissions.size(2)}')
                
        if tags is not None:
            if emissions.shape[:2] != tags.shape:
                raise ValueError(
                    'the first two dimensions of emissions and tags must match, '
                    f'got {tuple(emissions.shape[:2])} and {tuple(tags.shape)}')
                    
        if mask is not None:
            if emissions.shape[:2] != mask.shape:
                raise ValueError(
                    'the first two dimensions of emissions and mask must match, '
                    f'got {tuple(emissions.shape[:2])} and {tuple(mask.shape)}')
            no_empty_seq = not self.batch_first and mask[0].all()
            no_empty_seq_bf = self.batch_first and mask[:, 0].all()
            if not no_empty_seq and not no_empty_seq_bf:
                raise ValueError('mask of the first timestep must all be on')
                
    def _compute_score(self, emissions: torch.Tensor, tags: torch.LongTensor,
                       mask: torch.ByteTensor) -> torch.Tensor:
        # emissions: (seq_length, batch_size, num_tags)
        # tags: (seq_length, batch_size)
        # mask: (seq_length, batch_size)
        assert emissions.dim() == 3 and tags.dim() == 2
        assert emissions.shape[:2] == tags.shape
        assert emissions.size(2) == self.num_tags
        assert mask.shape == tags.shape
        assert mask[0].all()
        
        seq_length, batch_size = tags.shape
        mask = mask.float()
        
        # Start transition score and first emission
        score = self.start_trans[tags[0]]
        score += emissions[0, torch.arange(batch_size), tags[0]]
        
        # Transition scores
        for i in range(1, seq_length):
            score += self.trans_matrix[tags[i - 1], tags[i]] * mask[i]
            score += emissions[i, torch.arange(batch_size), tags[i]] * mask[i]
            
        # End transition score
        seq_ends = mask.long().sum(dim=0) - 1
        last_tags = tags[seq_ends, torch.arange(batch_size)]
        score += self.end_trans[last_tags]
        
        return score
        
    def _compute_normalizer(self, emissions: torch.Tensor,
                            mask: torch.ByteTensor) -> torch.Tensor:
        # emissions: (seq_length, batch_size, num_tags)
        # mask: (seq_length, batch_size)
        assert emissions.dim() == 3 and mask.dim() == 2
        assert emissions.shape[:2] == mask.shape
        assert emissions.size(2) == self.num_tags
        assert mask[0].all()
        
        seq_length = emissions.size(0)
        
        # Start transition score and first emission; score has size (batch_size, num_tags)
        score = self.start_trans + emissions[0]
        
        for i in range(1, seq_length):
            # Broadcast score for every possible next tag
            broadcast_score = score.unsqueeze(2)
            
            # Broadcast emission score for every possible current tag
            broadcast_emissions = emissions[i].unsqueeze(1)
            
            # Compute the score tensor of size (batch_size, num_tags, num_tags) where
            # next_score[i, j, k] = score[i, j] + transition_score[j, k] + emission_score[i, k]
            next_score = broadcast_score + self.trans_matrix + broadcast_emissions
            
            # Sum over all possible current tags, but we're in score space, so a sum
            # becomes a log-sum-exp: next_score has size (batch_size, num_tags)
            next_score = torch.logsumexp(next_score, dim=1)
            
            # Set score to the next score if this timestep is valid (mask == 1)
            # shape: (batch_size, num_tags)
            score = torch.where(mask[i].unsqueeze(1), next_score, score)
            
        # End transition score
        score += self.end_trans
        
        return torch.logsumexp(score, dim=1)
        
    def _viterbi_decode(self, emissions: torch.Tensor,
                        mask: torch.ByteTensor) -> List[List[int]]:
        # emissions: (seq_length, batch_size, num_tags)
        # mask: (seq_length, batch_size)
        assert emissions.dim() == 3 and mask.dim() == 2
        assert emissions.shape[:2] == mask.shape
        assert emissions.size(2) == self.num_tags
        assert mask[0].all()
        
        seq_length, batch_size = mask.shape
        
        # Start transition and first emission
        score = self.start_trans + emissions[0]
        history = []
        
        # score is a tensor of size (batch_size, num_tags) where for every batch,
        # value at column j stores the score of the best tag sequence so far that ends
        # with tag j
        # history saves where the best tags candidate transitioned from; this is used
        # when we trace back the best tag sequence
        
        # Viterbi algorithm recursive case: we compute the score of the best tag sequence
        # for every possible last tag
        for i in range(1, seq_length):
            # Broadcast viterbi score for every possible next tag
            broadcast_score = score.unsqueeze(2)
            
            # Broadcast emission score for every possible current tag
            broadcast_emission = emissions[i].unsqueeze(1)
            
            # Compute the score tensor of size (batch_size, num_tags, num_tags) where
            # next_score[i, j, k] = score[i, j] + transition_score[j, k] + emission_score[i, k]
            next_score = broadcast_score + self.trans_matrix + broadcast_emission
            
            # Find the maximum score over all possible current tag
            next_score, indices = next_score.max(dim=1)
            
            # Set score to the next score if this timestep is valid (mask == 1)
            # and save the index that produces the next score
            score = torch.where(mask[i].unsqueeze(1), next_score, score)
            history.append(indices)
            
        # End transition score
        score += self.end_trans
        
        # Now, compute the best path for each sample
        
        # shape: (batch_size,)
        seq_ends = mask.long().sum(dim=0) - 1
        best_tags_list = []
        
        for idx in range(batch_size):
            # Find the tag which maximizes the score at the last timestep; this is our best tag
            # for the last timestep
            _, best_last_tag = score[idx].max(dim=0)
            best_tags = [best_last_tag.item()]
            
            # We trace back where the best last tag comes from, append that to our best tag
            # sequence, and trace it back again, and so on
            for hist in reversed(history[:seq_ends[idx]]):
                best_last_tag = hist[idx][best_tags[-1]]
                best_tags.append(best_last_tag.item())
                
            # Reverse the order because we start from the last timestep
            best_tags.reverse()
            best_tags_list.append(best_tags)
            
        return best_tags_list


def sample_crf_sequences(crf_model, emissions, mask, num_samples=1, temperature=1.0, top_k=0, diversity=0.0):
    """
    从CRF模型中采样序列，支持多样性采样和温度调节
    
    Args:
        crf_model: CRF模型
        emissions: 发射概率
        mask: 掩码
        num_samples: 采样数量
        temperature: 温度参数，控制随机性（较高的值增加随机性）
        top_k: 如果>0，只从概率最高的k个标签中采样
        diversity: 多样性参数，控制不同样本之间的差异（0-1之间）
        
    Returns:
        采样的序列列表，每个批次有num_samples个样本
    """
    batch_size, seq_length, num_tags = emissions.size()
    emissions = emissions.cpu().numpy()
    mask = mask.cpu().numpy()

    all_sampled_sequences = []

    for i in range(batch_size):
        batch_samples = []
        seq_mask = mask[i]
        seq_emissions = emissions[i][:seq_mask.sum()]
        
        for sample_idx in range(num_samples):
            seq_sample = []
            
            # 对每个时间步进行采样
            for t, emission in enumerate(seq_emissions):
                # 应用温度缩放
                scaled_emission = emission / temperature
                
                # 计算概率分布
                probs = np.exp(scaled_emission - np.max(scaled_emission))
                probs = probs / probs.sum()
                
                # 应用top-k过滤
                if top_k > 0 and top_k < num_tags:
                    # 获取top-k索引和概率
                    top_indices = np.argsort(-probs)[:top_k]
                    top_probs = probs[top_indices]
                    top_probs = top_probs / top_probs.sum()  # 重新归一化
                    
                    # 从top-k中采样
                    sampled_tag = top_indices[np.random.choice(len(top_indices), p=top_probs)]
                else:
                    # 从完整分布中采样
                    sampled_tag = np.random.choice(num_tags, p=probs)
                
                seq_sample.append(sampled_tag)
            
            # 应用多样性增强
            if diversity > 0 and sample_idx > 0:
                # 与之前的样本比较，如果太相似则重新采样部分标签
                for prev_sample in batch_samples:
                    similarity = sum(1 for a, b in zip(seq_sample, prev_sample) if a == b) / len(seq_sample)
                    if similarity > (1 - diversity):
                        # 随机选择一些位置重新采样
                        positions_to_resample = np.random.choice(
                            len(seq_sample), 
                            size=max(1, int(diversity * len(seq_sample))), 
                            replace=False
                        )
                        for pos in positions_to_resample:
                            emission = seq_emissions[pos] / temperature
                            probs = np.exp(emission - np.max(emission))
                            probs /= probs.sum()
                            seq_sample[pos] = np.random.choice(num_tags, p=probs)
            
            batch_samples.append(seq_sample)
        
        all_sampled_sequences.extend(batch_samples)

    return all_sampled_sequences


def process_predictions(red_predictions, blue_predictions, lottery_type, check_history=True, similarity_rules=None):
    """
    处理预测结果，确保号码在有效范围内且为整数
    
    Args:
        red_predictions: 红球预测的类别索引
        blue_predictions: 蓝球预测的类别索引
        lottery_type: 彩票类型 ('ssq' 或 'dlt')
        check_history: 是否检查历史数据避免重复
        similarity_rules: 相似度规则列表，默认为None使用默认规则
        
    Returns:
        预测的开奖号码列表
    """
    if lottery_type == "dlt":
        # 大乐透前区：1-35，后区：1-12
        front_numbers = [min(max(int(num) + 1, 1), 35) for num in red_predictions[:5]]
        back_numbers = [min(max(int(num) + 1, 1), 12) for num in blue_predictions[:2]]

        # 确保前区号码唯一
        front_numbers = list(set(front_numbers))
        while len(front_numbers) < 5:
            additional_num = np.random.randint(1, 36)
            if additional_num not in front_numbers:
                front_numbers.append(additional_num)
        front_numbers = sorted(front_numbers)[:5]

        # 随机交换前区号码以增加多样性
        if np.random.rand() > 0.5:
            idx1, idx2 = np.random.choice(5, 2, replace=False)
            front_numbers[idx1], front_numbers[idx2] = front_numbers[idx2], front_numbers[idx1]

    elif lottery_type == "ssq":
        # 双色球红球：1-33，蓝球：1-16
        front_numbers = [min(max(int(num) + 1, 1), 33) for num in red_predictions[:6]]
        back_number = min(max(int(blue_predictions[0]) + 1, 1), 16)

        # 确保红球号码唯一
        front_numbers = list(set(front_numbers))
        while len(front_numbers) < 6:
            additional_num = np.random.randint(1, 34)
            if additional_num not in front_numbers:
                front_numbers.append(additional_num)
        front_numbers = sorted(front_numbers)[:6]

        # 随机交换红球号码以增加多样性
        if np.random.rand() > 0.5:
            idx1, idx2 = np.random.choice(6, 2, replace=False)
            front_numbers[idx1], front_numbers[idx2] = front_numbers[idx2], front_numbers[idx1]

    else:
        raise ValueError("不支持的彩票类型！请选择 'ssq' 或 'dlt'。")

    # 组合红蓝球号码
    if lottery_type == "dlt":
        prediction = front_numbers + back_numbers
    elif lottery_type == "ssq":
        prediction = front_numbers + [back_number]
        
    # 检查历史数据
    if check_history:
        try:
            # 导入历史检查相关函数
            from src.core.history_check import check_prediction_against_history, adjust_prediction_to_avoid_history, get_default_similarity_rules
            
            # 设置默认的相似度规则
            if similarity_rules is None:
                similarity_rules = get_default_similarity_rules(lottery_type)
            
            # 检查预测是否与历史相似
            is_similar = False
            for rule in similarity_rules:
                similar, match_info = check_prediction_against_history(prediction, lottery_type, rule)
                if similar:
                    is_similar = True
                    # 提取匹配的期数信息
                    match_periods = [match['期数'] for match in match_info]
                    print(f"预测结果与历史相似: {prediction}, 匹配期数: {match_periods}")
                    break
            
            # 如果与历史相似，调整预测
            if is_similar:
                adjusted_prediction = adjust_prediction_to_avoid_history(prediction, lottery_type, similarity_rules)
                print(f"调整后的预测结果: {adjusted_prediction}")
                return adjusted_prediction
        except ImportError as e:
            print(f"警告: 无法导入历史检查模块: {str(e)}")
            print("继续使用未经历史检查的预测结果")
        except Exception as e:
            print(f"警告: 历史检查过程中出错: {str(e)}")
            print("继续使用未经历史检查的预测结果")
    
    return prediction


def randomize_numbers(numbers, lottery_type, check_history=True, similarity_rules=None):
    """
    为预测号码增加随机性，以产生更多样化的结果
    
    Args:
        numbers: 原始预测号码列表
        lottery_type: 彩票类型 ('ssq' 或 'dlt')
        check_history: 是否检查预测结果与历史数据的相似性
        similarity_rules: 自定义相似度规则，如果为None则使用默认规则
        
    Returns:
        处理后的号码列表
    """
    # 如果不是已知的彩票类型，直接返回原始号码
    if lottery_type != "dlt" and lottery_type != "ssq":
        return numbers
        
    if lottery_type == "dlt":
        # 大乐透: 前区5个红球(1-35)，后区2个蓝球(1-12)
        red_numbers = numbers[:5]
        blue_numbers = numbers[5:]
        
        # 为前区号码增加随机性，但保持号码在合法范围内
        for i in range(len(red_numbers)):
            if random.random() < 0.3:  # 30%的几率修改号码
                offset = random.randint(-2, 2)
                red_numbers[i] = max(1, min(35, red_numbers[i] + offset))
        
        # 确保前区号码唯一
        while len(set(red_numbers)) < 5:
            for i in range(len(red_numbers)):
                if red_numbers.count(red_numbers[i]) > 1:
                    red_numbers[i] = random.randint(1, 35)
                    break
        
        # 为后区号码增加随机性
        for i in range(len(blue_numbers)):
            if random.random() < 0.3:
                offset = random.randint(-1, 1)
                blue_numbers[i] = max(1, min(12, blue_numbers[i] + offset))
                
        # 确保后区号码唯一
        while len(set(blue_numbers)) < 2:
            for i in range(len(blue_numbers)):
                if blue_numbers.count(blue_numbers[i]) > 1:
                    blue_numbers[i] = random.randint(1, 12)
                    break
        
        result = sorted(red_numbers) + sorted(blue_numbers)
        
    elif lottery_type == "ssq":
        # 双色球: 红球6个(1-33)，蓝球1个(1-16)
        red_numbers = numbers[:6]
        blue_number = numbers[6]
        
        # 为红球号码增加随机性
        for i in range(len(red_numbers)):
            if random.random() < 0.3:  # 30%的几率修改号码
                offset = random.randint(-2, 2)
                red_numbers[i] = max(1, min(33, red_numbers[i] + offset))
        
        # 确保红球号码唯一
        while len(set(red_numbers)) < 6:
            for i in range(len(red_numbers)):
                if red_numbers.count(red_numbers[i]) > 1:
                    red_numbers[i] = random.randint(1, 33)
                    break
        
        # 为蓝球增加随机性
        if random.random() < 0.3:
            offset = random.randint(-1, 1)
            blue_number = max(1, min(16, blue_number + offset))
            
        result = sorted(red_numbers) + [blue_number]
    
    # 检查历史相似性
    if check_history:
        try:
            # 导入历史检查相关函数
            from src.core.history_check import check_prediction_against_history, adjust_prediction_to_avoid_history, get_default_similarity_rules
            
            # 设置默认的相似度规则
            if similarity_rules is None:
                similarity_rules = get_default_similarity_rules(lottery_type)
            
            # 检查预测是否与历史相似
            is_similar = False
            for rule in similarity_rules:
                similar, match_info = check_prediction_against_history(result, lottery_type, rule)
                if similar:
                    is_similar = True
                    # 提取匹配的期数信息
                    match_periods = [match['期数'] for match in match_info]
                    print(f"随机化后的预测结果与历史相似: {result}, 匹配期数: {match_periods}")
                    break
            
            # 如果与历史相似，调整预测
            if is_similar:
                adjusted_result = adjust_prediction_to_avoid_history(result, lottery_type, similarity_rules)
                print(f"调整后的随机化预测结果: {adjusted_result}")
                return adjusted_result
        except ImportError as e:
            print(f"警告: 无法导入历史检查模块: {str(e)}")
            print("继续使用未经历史检查的随机化预测结果")
        except Exception as e:
            print(f"警告: 历史检查过程中出错: {str(e)}")
            print("继续使用未经历史检查的随机化预测结果")
    
    return result


class LstmCRFModel(BaseLotteryModel, nn.Module):
    """
    LSTM-CRF model for lottery number prediction
    """
    
    def __init__(self, input_size: int, hidden_size: int, num_layers: int, 
                 num_classes: int, lottery_type: str, dropout: float = 0.5,
                 bidirectional: bool = True, use_attention: bool = False,
                 use_residual: bool = False, output_seq_length: int = None):
        BaseLotteryModel.__init__(self, lottery_type)
        nn.Module.__init__(self)
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_classes = num_classes
        self.dropout = dropout
        self.bidirectional = bidirectional
        self.use_attention = use_attention
        self.use_residual = use_residual
        self.output_seq_length = output_seq_length
        self.output_dim = num_classes
        
        # 根据彩票类型调整输出维度，以兼容训练脚本中的模型
        if lottery_type == 'dlt':
            if output_seq_length == 5:  # 红球
                self.output_dim = 175  # 35 * 5
            elif output_seq_length == 2:  # 蓝球
                self.output_dim = 24  # 12 * 2
        elif lottery_type == 'ssq':
            if output_seq_length == 6:  # 红球
                self.output_dim = 198  # 33 * 6
            elif output_seq_length == 1:  # 蓝球
                self.output_dim = 16  # 16 * 1
        
        # LSTM layer
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional,
            batch_first=True
        )
        
        # Calculate LSTM output size
        lstm_output_size = hidden_size * 2 if bidirectional else hidden_size
        
        # Attention mechanism
        if use_attention:
            self.attention = nn.MultiheadAttention(
                embed_dim=lstm_output_size,
                num_heads=8,
                dropout=dropout,
                batch_first=True
            )
            
        # Residual connection
        if use_residual and input_size == lstm_output_size:
            self.residual_projection = None
        elif use_residual:
            self.residual_projection = nn.Linear(input_size, lstm_output_size)
        else:
            self.residual_projection = None
            
        # Fully connected layer
        self.fc = nn.Linear(lstm_output_size, self.output_dim)
        self.dropout_layer = nn.Dropout(dropout)
        
        # CRF layer
        self.crf = CRF(self.num_classes, batch_first=True)
        
    def forward(self, x: torch.Tensor, tags: Optional[torch.LongTensor] = None,
                mask: Optional[torch.ByteTensor] = None) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, input_size)
            tags: Target tags for training (batch_size, seq_len)
            mask: Mask for variable length sequences (batch_size, seq_len)
            
        Returns:
            If tags is provided (training), returns negative log likelihood
            If tags is None (inference), returns predicted tag sequences
        """
        batch_size, seq_len, _ = x.shape
        
        # LSTM forward pass
        lstm_out, _ = self.lstm(x)
        
        # Apply attention if enabled
        if self.use_attention:
            lstm_out, _ = self.attention(lstm_out, lstm_out, lstm_out)
            
        # Apply residual connection if enabled
        if self.use_residual and self.residual_projection is not None:
            residual = self.residual_projection(x)
            lstm_out = lstm_out + residual
        elif self.use_residual and self.residual_projection is None:
            lstm_out = lstm_out + x
            
        # Apply dropout
        lstm_out = self.dropout_layer(lstm_out)
        
        # 处理输出，兼容TorchCRF库
        # 使用最后一个时间步的输出，与训练脚本中的模型一致
        fc_out = self.fc(lstm_out[:, -1])
        
        # 重塑为(batch_size, output_seq_length, num_classes)形式
        emissions = fc_out.view(batch_size, self.output_seq_length, self.num_classes)
        
        if tags is not None:
            # Training mode: return negative log likelihood
            return -self.crf(emissions, tags, mask=mask).mean()
        else:
            # Inference mode: return predicted sequences
            return self.crf.viterbi_decode(emissions, mask=mask)
            
    def fit(self, data: Any, **kwargs) -> None:
        """
        Train the LSTM-CRF model
        
        Args:
            data: Training data
            **kwargs: Additional training parameters
        """
        # This method should be implemented based on specific training requirements
        self.is_trained = True
        
    def train(self, data: Any, **kwargs) -> None:
        """
        Train the LSTM-CRF model - implementation of abstract method from BaseLotteryModel
        
        Args:
            data: Training data
            **kwargs: Additional training parameters
        """
        # Call the fit method to maintain compatibility
        self.fit(data, **kwargs)
        
    def predict(self, recent_data=None, num_predictions=1, temperature=1.0, top_k=0, diversity=0.0, randomize=True, check_history=True, similarity_rules=None, **kwargs) -> Tuple[List[int], List[int]]:
        """
        Make predictions using the trained model
        
        Args:
            recent_data: 最近的数据，用于预测
            num_predictions: 生成的预测数量
            temperature: 温度参数，控制采样的随机性，越小越确定
            top_k: 只考虑概率最高的前k个选项
            diversity: 多样性参数，控制不同样本之间的差异（0-1之间）
            randomize: 是否对预测结果进行随机化处理
            check_history: 是否检查历史数据避免重复
            similarity_rules: 相似度规则列表，用于定义何种程度的相似需要避免
            
        Returns:
            Tuple of (red_numbers, blue_numbers)
        """
        # This method should be implemented based on specific prediction requirements
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        # 实现预测逻辑
        # 这里应该根据模型类型和彩票类型实现具体的预测逻辑
        # 以下是一个示例实现
        
        # 获取输入数据
        if recent_data is None:
            # 使用默认数据或者最近的训练数据
            # 这里需要根据实际情况实现
            pass
        
        # 根据彩票类型确定红球和蓝球的数量
        if self.lottery_type == "dlt":
            red_count = 5
            blue_count = 2
        elif self.lottery_type == "ssq":
            red_count = 6
            blue_count = 1
        else:
            raise ValueError(f"Unsupported lottery type: {self.lottery_type}")
        
        # 生成预测结果
        red_predictions = [0] * red_count  # 示例预测，实际应该使用模型输出
        blue_predictions = [0] * blue_count  # 示例预测，实际应该使用模型输出
        
        # 处理预测结果
        prediction = process_predictions(red_predictions, blue_predictions, self.lottery_type, check_history, similarity_rules)
        
        # 如果需要随机化处理
        if randomize:
            prediction = randomize_numbers(prediction, self.lottery_type, check_history, similarity_rules)
        
        # 根据彩票类型拆分红球和蓝球
        if self.lottery_type == "dlt":
            red_numbers = prediction[:5]
            blue_numbers = prediction[5:]
        elif self.lottery_type == "ssq":
            red_numbers = prediction[:6]
            blue_numbers = prediction[6:]
        else:
            red_numbers = []
            blue_numbers = []
        
        return red_numbers, blue_numbers
        
    def get_emissions(self, x: torch.Tensor) -> torch.Tensor:
        """
        获取发射概率矩阵，用于采样和概率分析
        
        Args:
            x: 输入张量，形状为 (batch_size, seq_len, input_size)
            
        Returns:
            发射概率矩阵，形状为 (batch_size, seq_len, num_classes)
        """
        with torch.no_grad():
            batch_size, seq_len, _ = x.shape
            
            # LSTM forward pass
            lstm_out, _ = self.lstm(x)
            
            # Apply attention if enabled
            if self.use_attention:
                lstm_out, _ = self.attention(lstm_out, lstm_out, lstm_out)
                
            # Apply residual connection if enabled
            if self.use_residual and self.residual_projection is not None:
                residual = self.residual_projection(x)
                lstm_out = lstm_out + residual
            elif self.use_residual and self.residual_projection is None:
                lstm_out = lstm_out + x
                
            # Apply dropout
            lstm_out = self.dropout_layer(lstm_out)
            
            # 使用最后一个时间步的输出，与训练脚本中的模型一致
            fc_out = self.fc(lstm_out[:, -1])
            
            # 重塑为(batch_size, output_seq_length, num_classes)形式
            emissions = fc_out.view(batch_size, self.output_seq_length, self.num_classes)
                
            # 应用softmax获取概率分布
            emissions = torch.nn.functional.softmax(emissions, dim=-1)
            return emissions
        
    def save_model(self, filepath: str) -> None:
        """
        Save the trained model
        
        Args:
            filepath: Path to save the model
        """
        torch.save({
            'model_state_dict': self.state_dict(),
            'model_config': {
                'input_size': self.input_size,
                'hidden_size': self.hidden_size,
                'num_layers': self.num_layers,
                'num_classes': self.num_classes,
                'lottery_type': self.lottery_type,
                'dropout': self.dropout,
                'bidirectional': self.bidirectional,
                'use_attention': self.use_attention,
                'use_residual': self.use_residual
            },
            'is_trained': self.is_trained
        }, filepath)
        
    def load_model(self, filepath: str) -> None:
        """
        Load a trained model
        
        Args:
            filepath: Path to load the model from
        """
        # 设置weights_only=False以兼容PyTorch 2.6+
        checkpoint = torch.load(filepath, map_location='cpu', weights_only=False)
        self.load_state_dict(checkpoint['model_state_dict'])
        self.is_trained = checkpoint.get('is_trained', False)