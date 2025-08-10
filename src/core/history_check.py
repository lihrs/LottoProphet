#!/usr/bin/env python
"""
历史号码检查模块
用于检查预测号码是否与历史号码重复或相似
"""
import pandas as pd
import numpy as np
import logging
from src.data.analysis import load_lottery_data


def get_default_similarity_rules(lottery_type):
    """获取默认的相似度规则
    :param lottery_type: 'ssq' 或 'dlt'
    :return: list of dict, 相似度规则列表
    """
    if lottery_type == 'dlt':
        return [
            {'front_match': 5, 'back_match': 2},  # 完全相同
            {'front_match': 5, 'back_match': 1},  # 前区5个相同，后区1个相同
            {'front_match': 5, 'back_match': 0},  # 前区5个相同，后区0个相同
            {'front_match': 4, 'back_match': 2}   # 前区4个相同，后区2个相同
        ]
    else:  # ssq
        return [
            {'front_match': 6, 'back_match': 1},  # 完全相同
            {'front_match': 6, 'back_match': 0},  # 前区6个相同，后区0个相同
            {'front_match': 5, 'back_match': 1},  # 前区5个相同，后区1个相同
            {'front_match': 5, 'back_match': 0}   # 前区5个相同，后区0个相同
        ]


def check_prediction_against_history(prediction, lottery_type, similarity_threshold=None):
    """
    检查预测号码是否与历史号码重复或相似
    
    Args:
        prediction: list, 预测的号码列表
        lottery_type: str, 'ssq' 或 'dlt'
        similarity_threshold: dict, 相似度阈值设置，默认为None表示只检查完全相同
            例如：{'front_match': 5, 'back_match': 0} 表示前区匹配5个，后区匹配0个
    
    Returns:
        tuple: (is_duplicate, match_info)
            is_duplicate: bool, 是否重复或相似
            match_info: dict, 匹配信息，包含匹配的期数、匹配的前区数量、匹配的后区数量
    """

        # 初始化日志记录器
    logger = logging.getLogger(f"history_check_{lottery_type}")
    logger.info(f"检查预测号码 {prediction} 是否与历史号码重复或相似")
    # 加载历史数据
    df = load_lottery_data(lottery_type)
    if df is None or len(df) == 0:
        return False, {}
    
    # 确定前区和后区的数量
    if lottery_type == 'dlt':
        front_count = 5
        back_count = 2
    elif lottery_type == 'ssq':
        front_count = 6
        back_count = 1
    else:
        raise ValueError(f"不支持的彩票类型: {lottery_type}")
    
    # 分离预测号码的前区和后区
    front_prediction = prediction[:front_count]
    back_prediction = prediction[front_count:]
    
    # 设置默认相似度阈值
    if similarity_threshold is None:
        # 默认只检查完全相同的情况
        similarity_threshold = {'front_match': front_count, 'back_match': back_count}
    
    # 遍历历史数据检查是否有相似或相同的号码
    match_results = []
    
    for _, row in df.iterrows():
        # 获取历史号码
        if lottery_type == 'dlt':
            front_history = [row['红球_1'], row['红球_2'], row['红球_3'], row['红球_4'], row['红球_5']]
            back_history = [row['蓝球_1'], row['蓝球_2']]
        else:  # ssq
            front_history = [row['红球_1'], row['红球_2'], row['红球_3'], row['红球_4'], row['红球_5'], row['红球_6']]
            back_history = [row['蓝球_1']]
        
        # 计算前区和后区匹配的数量
        front_match_count = len(set(front_prediction) & set(front_history))
        back_match_count = len(set(back_prediction) & set(back_history))
        
        # 检查是否达到相似度阈值
        if (front_match_count >= similarity_threshold.get('front_match', front_count) and
            back_match_count >= similarity_threshold.get('back_match', back_count)):
            match_results.append({
                '期数': row['期数'],
                'front_match': front_match_count,
                'back_match': back_match_count,
                'front_history': front_history,
                'back_history': back_history
            })
    
    # 如果有匹配结果，返回True和匹配信息
    if match_results:
        return True, match_results
    
    return False, {}


def filter_predictions_by_history(predictions, lottery_type, similarity_rules=None):
    """
    根据历史数据过滤预测结果，去除与历史数据相似或相同的预测
    
    Args:
        predictions: list of list, 预测号码列表的列表
        lottery_type: str, 'ssq' 或 'dlt'
        similarity_rules: list of dict, 相似度规则列表，每个规则是一个字典
            例如：[{'front_match': 5, 'back_match': 2}, {'front_match': 5, 'back_match': 1}]
    
    Returns:
        list: 过滤后的预测号码列表
    """
    # 如果没有指定相似度规则，使用默认规则
    if similarity_rules is None:
        similarity_rules = get_default_similarity_rules(lottery_type)
    
    filtered_predictions = []
    
    for prediction in predictions:
        should_filter = False
        
        # 检查每个相似度规则
        for rule in similarity_rules:
            is_similar, match_info = check_prediction_against_history(
                prediction, lottery_type, rule
            )
            
            if is_similar:
                should_filter = True
                break
        
        # 如果没有被过滤，添加到结果中
        if not should_filter:
            filtered_predictions.append(prediction)
    
    return filtered_predictions


def adjust_prediction_to_avoid_history(prediction, lottery_type, similarity_rules=None, max_attempts=10):
    """
    调整预测号码，避免与历史号码相似或相同
    
    Args:
        prediction: list, 预测的号码列表
        lottery_type: str, 'ssq' 或 'dlt'
        similarity_rules: list of dict, 相似度规则列表
        max_attempts: int, 最大尝试次数
    
    Returns:
        list: 调整后的预测号码列表
    """
    # 如果没有指定相似度规则，使用默认规则
    if similarity_rules is None:
        similarity_rules = get_default_similarity_rules(lottery_type)
    
    # 确定前区和后区的数量和范围
    if lottery_type == 'dlt':
        front_count = 5
        back_count = 2
        front_range = 35
        back_range = 12
    elif lottery_type == 'ssq':
        front_count = 6
        back_count = 1
        front_range = 33
        back_range = 16
    else:
        raise ValueError(f"不支持的彩票类型: {lottery_type}")
    
    # 复制原始预测
    adjusted_prediction = prediction.copy()
    
    # 尝试调整预测，直到不再与历史相似或达到最大尝试次数
    attempts = 0
    while attempts < max_attempts:
        # 检查是否与历史相似
        is_similar = False
        for rule in similarity_rules:
            similar, _ = check_prediction_against_history(
                adjusted_prediction, lottery_type, rule
            )
            if similar:
                is_similar = True
                break
        
        # 如果不相似，返回调整后的预测
        if not is_similar:
            return adjusted_prediction
        
        # 随机调整一个前区号码和一个后区号码
        front_idx = np.random.randint(0, front_count)
        front_new_value = np.random.randint(1, front_range + 1)
        
        # 确保前区号码唯一
        front_numbers = adjusted_prediction[:front_count]
        while front_new_value in front_numbers and front_new_value != front_numbers[front_idx]:
            front_new_value = np.random.randint(1, front_range + 1)
        
        # 更新前区号码
        front_numbers[front_idx] = front_new_value
        adjusted_prediction[:front_count] = sorted(front_numbers)
        
        # 调整后区号码
        if back_count > 0:
            back_idx = np.random.randint(0, back_count)
            back_new_value = np.random.randint(1, back_range + 1)
            
            # 确保后区号码唯一（如果需要）
            back_numbers = adjusted_prediction[front_count:]
            if back_count > 1:
                while back_new_value in back_numbers and back_new_value != back_numbers[back_idx]:
                    back_new_value = np.random.randint(1, back_range + 1)
            
            # 更新后区号码
            back_numbers[back_idx] = back_new_value
            if back_count > 1:
                adjusted_prediction[front_count:] = sorted(back_numbers)
            else:
                adjusted_prediction[front_count:] = back_numbers
        
        attempts += 1
    
    # 如果达到最大尝试次数仍未找到不相似的预测，返回最后一次调整的结果
    return adjusted_prediction