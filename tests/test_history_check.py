#!/usr/bin/env python
"""
测试历史号码检查功能
"""
import sys
import os

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.core.history_check import (
    check_prediction_against_history,
    filter_predictions_by_history,
    adjust_prediction_to_avoid_history
)
from src.core.prediction import process_predictions, randomize_numbers
from src.data.analysis import load_lottery_data


def test_check_prediction_against_history():
    """测试检查预测与历史的功能"""
    # 测试双色球
    lottery_type = 'ssq'
    # 使用一个已知的历史号码进行测试
    test_prediction = [1, 2, 6, 12, 16, 18, 8]  # 这应该是一个历史号码
    
    # 检查完全匹配
    is_similar, match_info = check_prediction_against_history(
        test_prediction, lottery_type, {'front_match': 6, 'back_match': 1}
    )
    print(f"完全匹配测试: {is_similar}")
    if is_similar:
        print(f"匹配信息: {match_info}")
    
    # 检查部分匹配
    is_similar, match_info = check_prediction_against_history(
        test_prediction, lottery_type, {'front_match': 5, 'back_match': 0}
    )
    print(f"部分匹配测试: {is_similar}")
    if is_similar:
        print(f"匹配信息: {match_info}")


def test_filter_predictions_by_history():
    """测试过滤预测结果的功能"""
    lottery_type = 'ssq'
    # 创建一些测试预测，包括一个已知的历史号码
    test_predictions = [
        [1, 2, 6, 12, 16, 18, 8],  # 历史号码
        [3, 7, 11, 21, 30, 33, 7],  # 可能是历史号码
        [5, 10, 15, 20, 25, 30, 10]  # 随机号码
    ]
    
    # 使用默认规则过滤
    filtered_predictions = filter_predictions_by_history(test_predictions, lottery_type)
    print(f"过滤前: {len(test_predictions)} 组预测")
    print(f"过滤后: {len(filtered_predictions)} 组预测")
    print(f"过滤后的预测: {filtered_predictions}")


def test_adjust_prediction_to_avoid_history():
    """测试调整预测以避免历史重复的功能"""
    lottery_type = 'ssq'
    # 使用一个已知的历史号码
    test_prediction = [1, 2, 6, 12, 16, 18, 8]
    
    # 调整预测
    adjusted_prediction = adjust_prediction_to_avoid_history(test_prediction, lottery_type)
    print(f"原始预测: {test_prediction}")
    print(f"调整后的预测: {adjusted_prediction}")
    
    # 验证调整后的预测不再与历史匹配
    is_similar, _ = check_prediction_against_history(
        adjusted_prediction, lottery_type, {'front_match': 6, 'back_match': 1}
    )
    print(f"调整后是否仍然完全匹配: {is_similar}")
    
    is_similar, _ = check_prediction_against_history(
        adjusted_prediction, lottery_type, {'front_match': 5, 'back_match': 0}
    )
    print(f"调整后是否仍然部分匹配: {is_similar}")


def test_process_predictions_with_history_check():
    """测试带有历史检查的预测处理功能"""
    lottery_type = 'ssq'
    # 模拟一些预测结果
    red_predictions = [0, 1, 5, 11, 15, 17]  # 对应 [1, 2, 6, 12, 16, 18]
    blue_predictions = [7]  # 对应 [8]
    
    # 不检查历史
    prediction_no_check = process_predictions(
        red_predictions, blue_predictions, lottery_type, check_history=False
    )
    print(f"不检查历史的预测: {prediction_no_check}")
    
    # 检查历史
    prediction_with_check = process_predictions(
        red_predictions, blue_predictions, lottery_type, check_history=True
    )
    print(f"检查历史的预测: {prediction_with_check}")
    
    # 验证检查历史的预测不再与历史匹配
    is_similar, _ = check_prediction_against_history(
        prediction_with_check, lottery_type, {'front_match': 5, 'back_match': 0}
    )
    print(f"检查历史的预测是否仍然部分匹配: {is_similar}")


def test_randomize_numbers_with_history_check():
    """测试带有历史检查的随机化号码功能"""
    lottery_type = 'ssq'
    # 使用一个已知的历史号码
    test_numbers = [1, 2, 6, 12, 16, 18, 8]
    
    # 不检查历史
    randomized_no_check = randomize_numbers(
        test_numbers, lottery_type, check_history=False
    )
    print(f"不检查历史的随机化: {randomized_no_check}")
    
    # 检查历史
    randomized_with_check = randomize_numbers(
        test_numbers, lottery_type, check_history=True
    )
    print(f"检查历史的随机化: {randomized_with_check}")
    
    # 验证检查历史的随机化不再与历史匹配
    is_similar, _ = check_prediction_against_history(
        randomized_with_check, lottery_type, {'front_match': 5, 'back_match': 0}
    )
    print(f"检查历史的随机化是否仍然部分匹配: {is_similar}")


if __name__ == "__main__":
    print("测试检查预测与历史的功能...")
    test_check_prediction_against_history()
    print("\n测试过滤预测结果的功能...")
    test_filter_predictions_by_history()
    print("\n测试调整预测以避免历史重复的功能...")
    test_adjust_prediction_to_avoid_history()
    print("\n测试带有历史检查的预测处理功能...")
    test_process_predictions_with_history_check()
    print("\n测试带有历史检查的随机化号码功能...")
    test_randomize_numbers_with_history_check()