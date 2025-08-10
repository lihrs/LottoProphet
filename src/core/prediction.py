#!/usr/bin/env python
import numpy as np
import random
from src.core.history_check import check_prediction_against_history, adjust_prediction_to_avoid_history, get_default_similarity_rules

def process_predictions(red_predictions, blue_predictions, lottery_type, check_history=True, similarity_rules=None):
    """处理预测结果，确保号码在有效范围内且为整数，并可选择性地检查历史数据避免重复
    :param red_predictions: list, 红球预测的类别索引
    :param blue_predictions: list, 蓝球预测的类别索引
    :param lottery_type: 'ssq' 或 'dlt'
    :param check_history: bool, 是否检查历史数据避免重复
    :param similarity_rules: list of dict, 相似度规则列表，用于定义何种程度的相似需要避免
    :return: list, 预测的开奖号码
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

    # 组合完整的预测号码
    if lottery_type == "dlt":
        prediction = front_numbers + back_numbers
    elif lottery_type == "ssq":
        prediction = front_numbers + [back_number]
    
    # 如果需要检查历史数据
    if check_history:
        # 设置默认的相似度规则
        if similarity_rules is None:
            similarity_rules = get_default_similarity_rules(lottery_type)
        
        # 检查预测是否与历史相似
        is_similar = False
        for rule in similarity_rules:
            similar, match_info = check_prediction_against_history(prediction, lottery_type, rule)
            if similar:
                is_similar = True
                break
        
        # 如果与历史相似，调整预测
        if is_similar:
            prediction = adjust_prediction_to_avoid_history(prediction, lottery_type, similarity_rules)
    
    return prediction

def randomize_numbers(numbers, lottery_type, check_history=True, similarity_rules=None):
    """为预测号码增加随机性，以产生更多样化的结果，并可选择性地检查历史数据避免重复
    
    Args:
        numbers: 原始预测号码列表
        lottery_type: 'ssq' 或 'dlt'
        check_history: bool, 是否检查历史数据避免重复
        similarity_rules: list of dict, 相似度规则列表，用于定义何种程度的相似需要避免
    
    Returns:
        处理后的号码列表
    """
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
        
        randomized_numbers = sorted(red_numbers) + sorted(blue_numbers)
        
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
            
        randomized_numbers = sorted(red_numbers) + [blue_number]
    
    else:
        return numbers  # 未知类型，返回原始号码
    
    # 如果需要检查历史数据
    if check_history:
        # 设置默认的相似度规则
        if similarity_rules is None:
            similarity_rules = get_default_similarity_rules(lottery_type)
        
        # 检查预测是否与历史相似
        is_similar = False
        for rule in similarity_rules:
            similar, match_info = check_prediction_against_history(randomized_numbers, lottery_type, rule)
            if similar:
                is_similar = True
                break
        
        # 如果与历史相似，调整预测
        if is_similar:
            randomized_numbers = adjust_prediction_to_avoid_history(randomized_numbers, lottery_type, similarity_rules)
    
    return randomized_numbers

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