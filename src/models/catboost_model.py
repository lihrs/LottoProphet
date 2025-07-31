# -*- coding:utf-8 -*-
"""
CatBoost model implementation for lottery prediction
优化版本：改进数据处理、模型架构和预测策略
"""

import os
import numpy as np
import pandas as pd
import pickle
import joblib
import json
import time
import warnings
from datetime import datetime
from functools import reduce
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, KFold
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder
from collections import Counter
import itertools
from typing import List, Tuple, Dict, Any

# 条件导入CatBoost
try:
    import catboost as cb
    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False

from .base import BaseMLModel


class WrappedCatBoostModel:
    """
    CatBoost模型包装器，用于统一预测接口
    """
    def __init__(self, model, processor=None):
        self.model = model
        self.processor = processor
        
    def predict(self, X):
        if self.processor:
            X = self.processor(X)
        return self.model.predict(X)


class CatBoostModel(BaseMLModel):
    """
    CatBoost模型实现
    """
    
    def __init__(self, lottery_type='dlt', feature_window=10, log_callback=None, use_gpu=False):
        """
        初始化CatBoost模型
        
        Args:
            lottery_type: 彩票类型，'dlt'或'ssq'
            feature_window: 特征窗口大小，使用多少期数据作为特征
            log_callback: 日志回调函数，用于将日志发送到UI
            use_gpu: 是否使用GPU训练
        """
        super().__init__(lottery_type, feature_window, log_callback, use_gpu)
        self.model_type = 'catboost'
        
        # 检查CatBoost是否可用
        if not CATBOOST_AVAILABLE:
            self.log("警告: CatBoost未安装或不可用，无法使用CatBoost模型")
    
    def train(self, df, optimize_params=False):
        """
        训练CatBoost模型 - 优化版本
        
        Args:
            df: 包含历史开奖数据的DataFrame
            optimize_params: 是否进行超参数优化，默认为False
            
        Returns:
            训练好的模型
        """
        if not CATBOOST_AVAILABLE:
            self.log("错误: CatBoost未安装或不可用，无法训练CatBoost模型")
            raise ImportError("CatBoost未安装或不可用，请先安装CatBoost")
            
        self.log("\n----- 开始训练优化版CatBoost模型 -----")
        
        # 准备增强特征数据
        X_train, X_test, y_red_train, y_red_test, y_blue_train, y_blue_test = self.prepare_enhanced_data(df)
        
        # 初始化模型信息字典，用于保存训练参数和性能指标
        model_info = {
            'n_features_in_': X_train.shape[1],
            'n_samples_train': X_train.shape[0],
            'n_samples_test': X_test.shape[0],
            'hyperparameter_optimization': optimize_params,
            'training_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
        # 如果启用超参数优化，则先进行优化
        if optimize_params:
            self.log("\n----- 开始超参数优化 -----")
            red_params = self.optimize_hyperparameters(X_train, y_red_train, 'red')
            blue_params = self.optimize_hyperparameters(X_train, y_blue_train, 'blue')
            
            # 保存优化后的参数到模型信息
            model_info['red_ball_params'] = red_params
            model_info['blue_ball_params'] = blue_params
            
            # 使用优化后的参数训练红球模型
            self.log("使用优化后的参数训练红球模型...")
            self.train_red_ball_models(X_train, y_red_train, X_test, y_red_test, custom_params=red_params)
            
            # 使用优化后的参数训练蓝球模型
            self.log("使用优化后的参数训练蓝球模型...")
            self.train_blue_ball_models(X_train, y_blue_train, X_test, y_blue_test, custom_params=blue_params)
        else:
            # 使用默认参数训练模型
            self.log("使用默认参数训练模型...")
            # 训练红球模型组合
            self.log("训练红球CatBoost模型组合...")
            self.train_red_ball_models(X_train, y_red_train, X_test, y_red_test)
            
            # 训练蓝球模型组合
            self.log("训练蓝球CatBoost模型组合...")
            self.train_blue_ball_models(X_train, y_blue_train, X_test, y_blue_test)
        
        # 评估模型性能
        evaluation_results = self.evaluate_enhanced(X_test, y_red_test, y_blue_test)
        
        # 将评估指标添加到模型信息
        if evaluation_results:
            model_info['performance_metrics'] = evaluation_results
        
        # 保存模型
        self.save_models(recent_data=df, model_info=model_info)
        
        return self.models
    
    def prepare_enhanced_data(self, df, test_size=0.2):
        """
        准备增强特征数据 - 包含更丰富的特征工程
        
        Args:
            df: 历史开奖数据
            test_size: 测试集比例
            
        Returns:
            训练和测试数据
        """
        self.log("准备增强特征数据...")
        
        # 确保数据按期数排序（如果有期数列）
        if '期数' in df.columns:
            df = df.sort_values('期数').reset_index(drop=True)
        else:
            df = df.reset_index(drop=True)
        
        # 提取红蓝球列名 - 更灵活的列名识别
        if self.lottery_type == 'dlt':
            # 尝试多种可能的列名格式
            red_cols = [col for col in df.columns if col.startswith('红球_')][:5]
            if not red_cols:
                red_cols = [col for col in df.columns if col.startswith('red_')][:5]
            if not red_cols:
                red_cols = [f'red_{i}' for i in range(1, 6) if f'red_{i}' in df.columns]
            
            blue_cols = [col for col in df.columns if col.startswith('蓝球_')][:2]
            if not blue_cols:
                blue_cols = [col for col in df.columns if col.startswith('blue_')][:2]
            if not blue_cols:
                blue_cols = [f'blue_{i}' for i in range(1, 3) if f'blue_{i}' in df.columns]
        else:  # ssq
            red_cols = [col for col in df.columns if col.startswith('红球_')][:6]
            if not red_cols:
                red_cols = [col for col in df.columns if col.startswith('red_')][:6]
            if not red_cols:
                red_cols = [f'red_{i}' for i in range(1, 7) if f'red_{i}' in df.columns]
            
            blue_cols = [col for col in df.columns if col.startswith('蓝球_')][:1]
            if not blue_cols:
                blue_cols = [col for col in df.columns if col.startswith('blue_')][:1]
            if not blue_cols:
                blue_cols = [f'blue_{i}' for i in range(1, 2) if f'blue_{i}' in df.columns]
        
        # 验证列名是否找到
        if not red_cols or not blue_cols:
            raise ValueError(f"无法找到足够的红球或蓝球列。找到的红球列: {red_cols}, 蓝球列: {blue_cols}")
        
        self.log(f"使用的红球列: {red_cols}")
        self.log(f"使用的蓝球列: {blue_cols}")
        
        # 创建增强特征
        enhanced_features = []
        red_targets = []
        blue_targets = []
        
        window_size = self.feature_window
        
        for i in range(len(df) - window_size):
            # 基础特征：历史号码
            basic_features = []
            for j in range(window_size):
                for col in red_cols + blue_cols:
                    basic_features.append(df.iloc[i + j][col])
            
            # 统计特征
            stat_features = self._extract_statistical_features(df.iloc[i:i+window_size], red_cols, blue_cols)
            
            # 趋势特征
            trend_features = self._extract_trend_features(df.iloc[i:i+window_size], red_cols, blue_cols)
            
            # 组合所有特征
            all_features = basic_features + stat_features + trend_features
            enhanced_features.append(all_features)
            
            # 目标值：下一期的号码组合
            next_row = df.iloc[i + window_size]
            red_combo = tuple(sorted([next_row[col] for col in red_cols]))
            blue_combo = tuple([next_row[col] for col in blue_cols])
            
            red_targets.append(red_combo)
            blue_targets.append(blue_combo)
        
        # 转换为numpy数组
        X = np.array(enhanced_features)
        
        # 编码目标值 - 将组合转换为字符串格式
        self.red_encoder = LabelEncoder()
        self.blue_encoder = LabelEncoder()
        
        # 将元组转换为字符串以便LabelEncoder处理
        red_targets_str = [str(combo) for combo in red_targets]
        blue_targets_str = [str(combo) for combo in blue_targets]
        
        y_red_encoded = self.red_encoder.fit_transform(red_targets_str)
        y_blue_encoded = self.blue_encoder.fit_transform(blue_targets_str)
        
        # 特征标准化
        from sklearn.preprocessing import StandardScaler
        self.feature_scaler = StandardScaler()
        X_scaled = self.feature_scaler.fit_transform(X)
        
        # 分割数据
        from sklearn.model_selection import train_test_split
        
        # 检查是否可以使用分层抽样
        from collections import Counter
        red_class_counts = Counter(y_red_encoded)
        min_red_count = min(red_class_counts.values())
        
        # 只有当每个类别至少有2个样本时才使用分层抽样
        if min_red_count >= 2:
            stratify_param = y_red_encoded
            self.log("使用分层抽样进行数据分割")
        else:
            stratify_param = None
            self.log(f"某些类别样本数不足（最少{min_red_count}个），使用随机抽样")
        
        X_train, X_test, y_red_train, y_red_test, y_blue_train, y_blue_test = train_test_split(
            X_scaled, y_red_encoded, y_blue_encoded, test_size=test_size, random_state=42, stratify=stratify_param
        )
        
        self.log(f"数据准备完成: 训练集{X_train.shape[0]}样本, 测试集{X_test.shape[0]}样本")
        self.log(f"特征维度: {X_train.shape[1]}, 红球类别数: {len(self.red_encoder.classes_)}, 蓝球类别数: {len(self.blue_encoder.classes_)}")
        
        return X_train, X_test, y_red_train, y_red_test, y_blue_train, y_blue_test
    
    def _extract_statistical_features(self, window_df, red_cols, blue_cols):
        """
        提取增强版统计特征
        """
        features = []
        
        # 红球统计特征 - 增强版
        red_values = []
        for col in red_cols:
            red_values.extend(window_df[col].tolist())
        
        # 基础统计特征
        features.extend([
            np.mean(red_values),  # 平均值
            np.std(red_values),   # 标准差
            np.min(red_values),   # 最小值
            np.max(red_values),   # 最大值
            len(set(red_values)), # 唯一值数量
            np.median(red_values), # 中位数
            np.percentile(red_values, 25), # 25分位数
            np.percentile(red_values, 75), # 75分位数
            np.var(red_values),   # 方差
            np.ptp(red_values),   # 极差
        ])
        
        # 频率特征 - 红球
        red_counts = Counter(red_values)
        max_count = max(red_counts.values()) if red_counts else 0
        min_count = min(red_counts.values()) if red_counts else 0
        features.extend([
            max_count,  # 最高频率
            min_count,  # 最低频率
            max_count - min_count,  # 频率差
            len([v for v in red_counts.values() if v > 1])  # 重复出现的号码数量
        ])
        
        # 蓝球统计特征 - 增强版
        blue_values = []
        for col in blue_cols:
            blue_values.extend(window_df[col].tolist())
        
        # 基础统计特征
        features.extend([
            np.mean(blue_values),
            np.std(blue_values),
            np.min(blue_values),
            np.max(blue_values),
            len(set(blue_values)),
            np.median(blue_values), # 中位数
            np.percentile(blue_values, 25), # 25分位数
            np.percentile(blue_values, 75), # 75分位数
            np.var(blue_values),   # 方差
            np.ptp(blue_values),   # 极差
        ])
        
        # 频率特征 - 蓝球
        blue_counts = Counter(blue_values)
        max_count = max(blue_counts.values()) if blue_counts else 0
        min_count = min(blue_counts.values()) if blue_counts else 0
        features.extend([
            max_count,  # 最高频率
            min_count,  # 最低频率
            max_count - min_count,  # 频率差
            len([v for v in blue_counts.values() if v > 1])  # 重复出现的号码数量
        ])
        
        # 红蓝球关系特征
        if red_values and blue_values:
            features.extend([
                np.mean(red_values) / np.mean(blue_values) if np.mean(blue_values) != 0 else 0,  # 红蓝平均值比
                np.std(red_values) / np.std(blue_values) if np.std(blue_values) != 0 else 0,  # 红蓝标准差比
                np.corrcoef(red_values[:len(blue_values)], blue_values[:len(red_values)])[0, 1] if len(red_values) == len(blue_values) and len(red_values) > 1 else 0  # 相关系数
            ])
        
        return features
    
    def _extract_trend_features(self, window_df, red_cols, blue_cols):
        """
        提取增强版趋势特征
        """
        features = []
        
        # 红球趋势 - 增强版
        red_sums = []
        red_means = []
        red_stds = []
        red_mins = []
        red_maxs = []
        red_ranges = []
        
        for _, row in window_df.iterrows():
            red_values = [row[col] for col in red_cols]
            red_sums.append(sum(red_values))
            red_means.append(np.mean(red_values))
            red_stds.append(np.std(red_values))
            red_mins.append(min(red_values))
            red_maxs.append(max(red_values))
            red_ranges.append(max(red_values) - min(red_values))
        
        # 计算各种趋势指标
        if len(red_sums) > 1:
            x = np.arange(len(red_sums))
            # 线性趋势（一阶多项式拟合）
            red_slope, red_intercept = np.polyfit(x, red_sums, 1)
            features.append(red_slope)  # 斜率
            features.append(red_intercept)  # 截距
            
            # 二阶多项式拟合（捕捉非线性趋势）
            if len(red_sums) > 2:
                red_poly2 = np.polyfit(x, red_sums, 2)
                features.extend(red_poly2)  # 二阶多项式系数
            else:
                features.extend([0, 0, 0])  # 填充默认值
            
            # 均值、标准差、最小值、最大值、范围的趋势
            features.append(np.polyfit(x, red_means, 1)[0])
            features.append(np.polyfit(x, red_stds, 1)[0])
            features.append(np.polyfit(x, red_mins, 1)[0])
            features.append(np.polyfit(x, red_maxs, 1)[0])
            features.append(np.polyfit(x, red_ranges, 1)[0])
            
            # 移动平均趋势
            window_size = min(3, len(red_sums))
            if window_size > 1:
                moving_avg = np.convolve(red_sums, np.ones(window_size)/window_size, mode='valid')
                if len(moving_avg) > 1:
                    features.append(moving_avg[-1] - moving_avg[0])  # 移动平均变化
                else:
                    features.append(0)
            else:
                features.append(0)
        else:
            # 如果数据不足，填充默认值
            features.extend([0] * 11)  # 对应上面添加的11个特征
        
        # 蓝球趋势 - 增强版
        blue_sums = []
        blue_means = []
        blue_stds = []
        blue_mins = []
        blue_maxs = []
        blue_ranges = []
        
        for _, row in window_df.iterrows():
            blue_values = [row[col] for col in blue_cols]
            blue_sums.append(sum(blue_values))
            blue_means.append(np.mean(blue_values))
            blue_stds.append(np.std(blue_values) if len(blue_values) > 1 else 0)
            blue_mins.append(min(blue_values))
            blue_maxs.append(max(blue_values))
            blue_ranges.append(max(blue_values) - min(blue_values))
        
        # 计算各种趋势指标
        if len(blue_sums) > 1:
            x = np.arange(len(blue_sums))
            # 线性趋势
            blue_slope, blue_intercept = np.polyfit(x, blue_sums, 1)
            features.append(blue_slope)
            features.append(blue_intercept)
            
            # 二阶多项式拟合
            if len(blue_sums) > 2:
                blue_poly2 = np.polyfit(x, blue_sums, 2)
                features.extend(blue_poly2)
            else:
                features.extend([0, 0, 0])
            
            # 均值、标准差、最小值、最大值、范围的趋势
            features.append(np.polyfit(x, blue_means, 1)[0])
            if all(std > 0 for std in blue_stds):
                features.append(np.polyfit(x, blue_stds, 1)[0])
            else:
                features.append(0)
            features.append(np.polyfit(x, blue_mins, 1)[0])
            features.append(np.polyfit(x, blue_maxs, 1)[0])
            features.append(np.polyfit(x, blue_ranges, 1)[0])
            
            # 移动平均趋势
            window_size = min(3, len(blue_sums))
            if window_size > 1:
                moving_avg = np.convolve(blue_sums, np.ones(window_size)/window_size, mode='valid')
                if len(moving_avg) > 1:
                    features.append(moving_avg[-1] - moving_avg[0])
                else:
                    features.append(0)
            else:
                features.append(0)
        else:
            # 如果数据不足，填充默认值
            features.extend([0] * 11)
        
        # 奇偶比例特征 - 增强版
        red_odd_count = sum(1 for _, row in window_df.iterrows() for col in red_cols if row[col] % 2 == 1)
        red_even_count = len(red_cols) * len(window_df) - red_odd_count
        red_odd_ratio = red_odd_count / (red_odd_count + red_even_count) if (red_odd_count + red_even_count) > 0 else 0
        features.append(red_odd_ratio)
        
        blue_odd_count = sum(1 for _, row in window_df.iterrows() for col in blue_cols if row[col] % 2 == 1)
        blue_even_count = len(blue_cols) * len(window_df) - blue_odd_count
        blue_odd_ratio = blue_odd_count / (blue_odd_count + blue_even_count) if (blue_odd_count + blue_even_count) > 0 else 0
        features.append(blue_odd_ratio)
        
        # 奇偶比例变化趋势
        red_odd_ratios = []
        blue_odd_ratios = []
        for i, row in window_df.iterrows():
            red_values = [row[col] for col in red_cols]
            red_odd = sum(1 for v in red_values if v % 2 == 1)
            red_odd_ratios.append(red_odd / len(red_values) if len(red_values) > 0 else 0)
            
            blue_values = [row[col] for col in blue_cols]
            blue_odd = sum(1 for v in blue_values if v % 2 == 1)
            blue_odd_ratios.append(blue_odd / len(blue_values) if len(blue_values) > 0 else 0)
        
        if len(red_odd_ratios) > 1:
            x = np.arange(len(red_odd_ratios))
            features.append(np.polyfit(x, red_odd_ratios, 1)[0])  # 红球奇偶比例变化趋势
        else:
            features.append(0)
            
        if len(blue_odd_ratios) > 1:
            x = np.arange(len(blue_odd_ratios))
            features.append(np.polyfit(x, blue_odd_ratios, 1)[0])  # 蓝球奇偶比例变化趋势
        else:
            features.append(0)
        
        # 大小比例特征（大于中间值的比例）
        red_mid = self.red_range / 2
        blue_mid = self.blue_range / 2
        
        red_large_count = sum(1 for _, row in window_df.iterrows() for col in red_cols if row[col] > red_mid)
        red_total = len(red_cols) * len(window_df)
        features.append(red_large_count / red_total if red_total > 0 else 0)
        
        blue_large_count = sum(1 for _, row in window_df.iterrows() for col in blue_cols if row[col] > blue_mid)
        blue_total = len(blue_cols) * len(window_df)
        features.append(blue_large_count / blue_total if blue_total > 0 else 0)
        
        return features
    
    def train_red_ball_models(self, X_train, y_red_train, X_test, y_red_test, custom_params=None):
        """
        训练红球模型组合
        
        Args:
            X_train: 训练特征
            y_red_train: 训练标签
            X_test: 测试特征
            y_red_test: 测试标签
            custom_params: 自定义参数，如果提供则使用这些参数，否则使用默认参数
        """
        self.log("训练红球CatBoost模型...")
        
        # 优化的超参数 - 更新参数以提高性能
        default_params = {
            'iterations': 1500,  # 增加迭代次数以提高模型收敛
            'depth': 8,
            'learning_rate': 0.03,  # 降低学习率以减少过拟合风险
            'l2_leaf_reg': 5,  # 增加正则化
            'border_count': 128,
            'bagging_temperature': 1.2,  # 增加随机性
            'random_strength': 1.5,  # 增加随机性
            'od_type': 'Iter',
            'od_wait': 100,  # 增加早停等待轮数
            'max_ctr_complexity': 3,  # 控制分类特征处理复杂度
            'leaf_estimation_method': 'Newton'  # 使用牛顿法估计叶子值
        }
        
        # 如果提供了自定义参数，则使用自定义参数
        best_params = custom_params if custom_params else default_params
        self.log(f"使用参数: {best_params}")
        
        # 设置GPU或CPU
        if self.use_gpu and 'task_type' not in best_params:
            best_params['task_type'] = 'GPU'
            best_params['devices'] = '0'
        
        # 训练主模型
        red_model = cb.CatBoostClassifier(
            **best_params,
            random_seed=42,
            verbose=False
        )
        
        # 过滤测试集，只保留训练集中存在的类别
        train_classes = set(y_red_train)
        valid_test_indices = [i for i, label in enumerate(y_red_test) if label in train_classes]
        
        if len(valid_test_indices) > 0:
            X_test_filtered = X_test[valid_test_indices]
            y_red_test_filtered = y_red_test[valid_test_indices]
            
            # 使用早停训练
            red_model.fit(
                X_train, y_red_train,
                eval_set=[(X_test_filtered, y_red_test_filtered)],
                early_stopping_rounds=50,
                verbose=False
            )
        else:
            # 如果没有有效的测试样本，不使用验证集
            self.log("警告: 测试集中没有训练集包含的类别，不使用验证集")
            red_model.fit(X_train, y_red_train, verbose=False)
        
        # 评估性能
        train_acc = red_model.score(X_train, y_red_train)
        test_acc = red_model.score(X_test, y_red_test)
        self.log(f"红球模型 - 训练准确率: {train_acc:.4f}, 测试准确率: {test_acc:.4f}")
        
        self.models['red'] = WrappedCatBoostModel(red_model)
    
    def train_blue_ball_models(self, X_train, y_blue_train, X_test, y_blue_test, custom_params=None):
        """
        训练蓝球模型组合
        
        Args:
            X_train: 训练特征
            y_blue_train: 训练标签
            X_test: 测试特征
            y_blue_test: 测试标签
            custom_params: 自定义参数，如果提供则使用这些参数，否则使用默认参数
        """
        self.log("训练蓝球CatBoost模型...")
        
        # 优化的超参数 - 更新参数以提高蓝球预测性能
        default_params = {
            'iterations': 1200,  # 增加迭代次数
            'depth': 7,  # 增加树深度以捕获更复杂的模式
            'learning_rate': 0.05,  # 降低学习率以减少过拟合
            'l2_leaf_reg': 6,  # 增加正则化
            'border_count': 96,  # 增加分箱数量以提高精度
            'bagging_temperature': 0.8,  # 增加随机性但保持适度
            'random_strength': 12,  # 增加随机性
            'od_type': 'Iter',
            'od_wait': 60,  # 增加早停等待轮数
            'max_ctr_complexity': 2,  # 控制分类特征处理复杂度
            'leaf_estimation_method': 'Newton',  # 使用牛顿法估计叶子值
            'bootstrap_type': 'Bernoulli',  # 使用伯努利采样
            'subsample': 0.85  # 每次迭代使用85%的数据
        }
        
        # 如果提供了自定义参数，则使用自定义参数
        best_params = custom_params if custom_params else default_params
        self.log(f"使用参数: {best_params}")
        
        # 设置GPU或CPU
        if self.use_gpu and 'task_type' not in best_params:
            best_params['task_type'] = 'GPU'
            best_params['devices'] = '0'
        
        # 训练主模型
        blue_model = cb.CatBoostClassifier(
            **best_params,
            random_seed=42,
            verbose=False
        )
        
        # 过滤测试集，只保留训练集中存在的类别
        train_classes = set(y_blue_train)
        valid_test_indices = [i for i, label in enumerate(y_blue_test) if label in train_classes]
        
        if len(valid_test_indices) > 0:
            X_test_filtered = X_test[valid_test_indices]
            y_blue_test_filtered = y_blue_test[valid_test_indices]
            
            # 使用早停训练
            blue_model.fit(
                X_train, y_blue_train,
                eval_set=[(X_test_filtered, y_blue_test_filtered)],
                early_stopping_rounds=30,
                verbose=False
            )
        else:
            # 如果没有有效的测试样本，不使用验证集
            self.log("警告: 测试集中没有训练集包含的类别，不使用验证集")
            blue_model.fit(X_train, y_blue_train, verbose=False)
        
        # 评估性能
        train_acc = blue_model.score(X_train, y_blue_train)
        test_acc = blue_model.score(X_test, y_blue_test)
        self.log(f"蓝球模型 - 训练准确率: {train_acc:.4f}, 测试准确率: {test_acc:.4f}")
        
        self.models['blue'] = WrappedCatBoostModel(blue_model)
    
    def optimize_hyperparameters(self, X_train, y_train, ball_type):
        """
        使用随机搜索优化CatBoost模型的超参数
        
        Args:
            X_train: 训练特征
            y_train: 训练标签
            ball_type: 球类型，'red'或'blue'
            
        Returns:
            dict: 优化后的超参数
        """
        if not CATBOOST_AVAILABLE:
            self.log("错误: CatBoost未安装或不可用，无法进行超参数优化")
            raise ImportError("CatBoost未安装或不可用，请先安装CatBoost")
            
        self.log(f"开始{ball_type}球模型超参数优化...")
        self.log(f"数据维度: 特征={X_train.shape}, 标签={y_train.shape}")
        
        # 确保y_train是整数类型
        y_train = y_train.astype(int)
        
        # 检查是否有类别样本数量过少的情况
        class_counts = Counter(y_train)
        min_samples = min(class_counts.values())
        
        # 设置交叉验证策略
        if min_samples < 2:
            # 如果有类别样本数量过少，使用KFold代替StratifiedKFold
            self.log(f"警告: 某些类别样本数量过少(最少{min_samples}个)，使用KFold代替StratifiedKFold")
            cv = KFold(n_splits=5, shuffle=True, random_state=42)
        else:
            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        
        # 设置GPU或CPU使用
        task_type = 'GPU' if self.use_gpu else 'CPU'
        self.log(f"使用{task_type}进行超参数优化")
        
        # 根据球类型设置不同的超参数搜索空间
        if ball_type == 'red':
            param_grid = {
                'iterations': [800, 1000, 1500, 2000],
                'depth': [6, 8, 10, 12],
                'learning_rate': [0.01, 0.03, 0.05, 0.1],
                'l2_leaf_reg': [3, 5, 7, 9],
                'border_count': [64, 96, 128, 254],
                'bagging_temperature': [0.8, 1.0, 1.2, 1.5],
                'random_strength': [1, 1.5, 2, 3],
                'max_ctr_complexity': [2, 3, 4],
                'leaf_estimation_method': ['Newton', 'Gradient'],
                'od_type': ['Iter'],
                'od_wait': [50, 100, 150]
            }
        else:  # blue
            param_grid = {
                'iterations': [600, 800, 1000, 1200],
                'depth': [5, 6, 7, 8],
                'learning_rate': [0.03, 0.05, 0.07, 0.1],
                'l2_leaf_reg': [4, 6, 8, 10],
                'border_count': [64, 96, 128],
                'bagging_temperature': [0.6, 0.8, 1.0, 1.2],
                'random_strength': [8, 10, 12, 15],
                'max_ctr_complexity': [1, 2, 3],
                'leaf_estimation_method': ['Newton', 'Gradient'],
                'bootstrap_type': ['Bernoulli', 'Bayesian'],
                'subsample': [0.75, 0.8, 0.85, 0.9],
                'od_type': ['Iter'],
                'od_wait': [30, 60, 90]
            }
        
        # 创建基础模型
        base_params = {
            'task_type': task_type,
            'random_seed': 42,
            'verbose': 0,
            'loss_function': 'MultiClass',
            'eval_metric': 'Accuracy',
            'od_type': 'Iter',
            'od_wait': 50
        }
        
        # 创建自定义日志回调
        class CustomCallback(object):
            def __init__(self, log_func):
                self.log_func = log_func
                self.iteration = 0
                self.start_time = time.time()
                
            def after_iteration(self, info):
                self.iteration += 1
                if self.iteration % 100 == 0:
                    elapsed = time.time() - self.start_time
                    self.log_func(f"超参数优化进度: 迭代 {self.iteration}, 耗时 {elapsed:.2f}秒")
                return True
        
        callback = CustomCallback(self.log)
        
        try:
            # 创建随机搜索对象
            self.log(f"开始随机搜索最佳超参数，搜索空间大小: {len(param_grid)}个参数")
            n_iter = min(20, reduce(lambda x, y: x * len(y), param_grid.values(), 1))
            n_iter = max(10, min(n_iter, 30))  # 确保至少10次，最多30次迭代
            self.log(f"将执行{n_iter}次随机搜索迭代")
            
            random_search = RandomizedSearchCV(
                estimator=cb.CatBoostClassifier(**base_params),
                param_distributions=param_grid,
                n_iter=n_iter,
                scoring='accuracy',
                cv=cv,
                random_state=42,
                n_jobs=1,  # CatBoost内部已经并行化
                verbose=0
            )
            
            # 执行随机搜索
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                random_search.fit(X_train, y_train, cat_features=None, callbacks=[callback])
            
            # 获取最佳参数
            best_params = random_search.best_params_
            best_score = random_search.best_score_
            
            self.log(f"超参数优化完成! 最佳验证准确率: {best_score:.4f}")
            self.log(f"最佳参数: {best_params}")
            
            # 合并基础参数和最佳参数
            final_params = {**base_params, **best_params}
            return final_params
            
        except Exception as e:
            self.log(f"超参数优化失败: {str(e)}")
            self.log("使用默认参数作为回退方案")
            
            # 回退到默认参数
            if ball_type == 'red':
                default_params = {
                    'iterations': 1000,
                    'depth': 8,
                    'learning_rate': 0.05,
                    'l2_leaf_reg': 5,
                    'border_count': 128,
                    'bagging_temperature': 1.0,
                    'random_strength': 1.5,
                    'max_ctr_complexity': 3,
                    'leaf_estimation_method': 'Newton',
                    'od_type': 'Iter',
                    'od_wait': 100
                }
            else:  # blue
                default_params = {
                    'iterations': 800,
                    'depth': 6,
                    'learning_rate': 0.05,
                    'l2_leaf_reg': 6,
                    'border_count': 96,
                    'bagging_temperature': 0.8,
                    'random_strength': 10,
                    'max_ctr_complexity': 2,
                    'leaf_estimation_method': 'Newton',
                    'bootstrap_type': 'Bayesian',
                    'subsample': 0.8,
                    'od_type': 'Iter',
                    'od_wait': 60
                }
            
            # 合并基础参数和默认参数
            final_params = {**base_params, **default_params}
            return final_params
        
    def train_catboost(self, X_train, y_train, ball_type):
        """
        训练CatBoost模型，使用交叉验证和超参数调优
        
        Args:
            X_train: 训练特征
            y_train: 训练标签
            ball_type: 球类型，'red'或'blue'
            
        Returns:
            训练好的CatBoost模型
        """
        self.log(f"训练{ball_type}球CatBoost模型...")
        self.log(f"数据维度: 特征={X_train.shape}, 标签={y_train.shape}")
        
        # 确保y_train是整数类型
        y_train = y_train.astype(int)
        
        # 检查是否有类别样本数量过少的情况
        class_counts = Counter(y_train)
        min_samples = min(class_counts.values())
        
        # 设置交叉验证策略
        if min_samples < 2:
            # 如果有类别样本数量过少，使用KFold代替StratifiedKFold
            self.log(f"警告: 某些类别样本数量过少(最少{min_samples}个)，使用KFold代替StratifiedKFold")
            cv = KFold(n_splits=5, shuffle=True, random_state=42)
        else:
            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        
        # 设置GPU或CPU使用
        task_type = 'GPU' if self.use_gpu else 'CPU'
        self.log(f"使用{task_type}训练CatBoost模型")
        
        # 设置超参数搜索空间
        param_grid = {
            'iterations': [100, 200, 300, 500],
            'depth': [4, 6, 8, 10],
            'learning_rate': [0.01, 0.05, 0.1, 0.2],
            'l2_leaf_reg': [1, 3, 5, 7],
            'border_count': [32, 64, 128],
            'bagging_temperature': [0, 1, 10],
            'random_strength': [1, 10, 100]
        }
        
        # 创建一个自定义回调函数，用于记录超参数搜索进度
        class LoggingCallback:
            def __init__(self, logger):
                self.logger = logger
                self.iteration = 0
                
            def after_iteration(self, info):
                self.iteration += 1
                self.logger(f"训练进度: 第{self.iteration}轮")
                return True  # 返回True继续训练，False停止训练
                
            def __call__(self, params, train_pool, test_pool):
                self.iteration += 1
                self.logger(f"超参数搜索进度: {self.iteration}/20")
                return False
        
        logging_callback = LoggingCallback(self.log)
        
        # 使用随机搜索进行超参数调优
        random_search = RandomizedSearchCV(
            estimator=cb.CatBoostClassifier(
                task_type=task_type,
                devices='0' if self.use_gpu else None,
                random_seed=42,
                verbose=0
            ),
            param_distributions=param_grid,
            n_iter=20,  # 尝试20种不同的组合
            scoring='accuracy',
            cv=cv,
            verbose=0,
            random_state=42,
            n_jobs=1  # CatBoost内部已经并行化，所以这里设为1
        )
        
        # 训练模型
        try:
            self.log("开始超参数搜索...")
            random_search.fit(X_train, y_train)
            
            # 获取最佳模型
            best_cb = random_search.best_estimator_
            
            # 记录最佳参数
            self.log(f"最佳参数: {random_search.best_params_}")
            self.log(f"交叉验证最佳得分: {random_search.best_score_:.4f}")
            
            # 使用最佳参数训练最终模型，启用早停
            best_params = random_search.best_params_
            final_model = cb.CatBoostClassifier(
                **best_params,
                task_type=task_type,
                devices='0' if self.use_gpu else None,
                random_seed=42,
                verbose=0
            )
            
            # 创建验证集
            from sklearn.model_selection import train_test_split
            X_train_final, X_val, y_train_final, y_val = train_test_split(
                X_train, y_train, test_size=0.2, random_state=42, stratify=y_train if min_samples >= 2 else None
            )
            
            # 训练最终模型，使用早停
            self.log("开始训练最终模型...")
            final_model.fit(
                X_train_final, y_train_final,
                eval_set=[(X_val, y_val)],
                early_stopping_rounds=50,
                verbose=0,
                callbacks=[logging_callback]
            )
            
            # 评估验证集准确率
            val_preds = final_model.predict(X_val)
            val_accuracy = accuracy_score(y_val, val_preds)
            self.log(f"验证集准确率: {val_accuracy:.4f}")
            
            # 记录特征重要性
            feature_importances = final_model.feature_importances_
            top_n = 10  # 只显示前10个最重要的特征
            indices = np.argsort(feature_importances)[-top_n:]
            self.log(f"前{top_n}个最重要特征的重要性:")
            for i in indices:
                self.log(f"特征 {i}: {feature_importances[i]:.4f}")
            
            # 包装模型以统一接口
            wrapped_model = WrappedCatBoostModel(final_model)
            
            return wrapped_model
        except Exception as e:
            self.log(f"CatBoost超参数搜索失败: {e}")
            self.log("使用默认参数训练CatBoost模型")
            
            # 使用默认参数训练模型
            default_model = cb.CatBoostClassifier(
                iterations=500,
                depth=6,
                learning_rate=0.1,
                task_type=task_type,
                devices='0' if self.use_gpu else None,
                random_seed=42,
                verbose=0
            )
            
            # 创建新的回调实例用于默认模型训练
            default_logging_callback = LoggingCallback(self.log)
            default_model.fit(X_train, y_train, verbose=0, callbacks=[default_logging_callback])
            
            # 包装模型以统一接口
            wrapped_model = WrappedCatBoostModel(default_model)
            
            return wrapped_model
    
    def evaluate_enhanced(self, X_test, y_red_test, y_blue_test, visualize=False):
        """
        评估增强模型性能 - 包含更多评估指标和可视化选项
        
        Args:
            X_test: 测试特征
            y_red_test: 红球测试标签
            y_blue_test: 蓝球测试标签
            visualize: 是否生成可视化结果
            
        Returns:
            包含各种评估指标的字典
        """
        self.log("评估增强模型性能...")
        
        # 初始化评估结果字典
        evaluation_results = {
            'red_metrics': {},
            'blue_metrics': {},
            'red_confusion_matrix': None,
            'blue_confusion_matrix': None,
            'red_feature_importance': None,
            'blue_feature_importance': None
        }
        
        # 评估红球模型
        if 'red' in self.models:
            red_model = self.models['red'].model
            # 预测类别
            red_preds = self.models['red'].predict(X_test)
            # 尝试获取预测概率
            try:
                red_probs = red_model.predict_proba(X_test)
            except:
                red_probs = None
                
            # 计算基本指标
            red_accuracy = accuracy_score(y_red_test, red_preds)
            evaluation_results['red_metrics']['accuracy'] = red_accuracy
            self.log(f"红球组合模型准确率: {red_accuracy:.4f}")
            
            # 计算更多指标
            if len(set(y_red_test)) > 1:
                from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
                
                # 计算精确率、召回率和F1分数
                red_precision = precision_score(y_red_test, red_preds, average='weighted')
                red_recall = recall_score(y_red_test, red_preds, average='weighted')
                red_f1 = f1_score(y_red_test, red_preds, average='weighted')
                
                evaluation_results['red_metrics']['precision'] = red_precision
                evaluation_results['red_metrics']['recall'] = red_recall
                evaluation_results['red_metrics']['f1'] = red_f1
                
                self.log(f"红球模型精确率: {red_precision:.4f}")
                self.log(f"红球模型召回率: {red_recall:.4f}")
                self.log(f"红球模型F1分数: {red_f1:.4f}")
                
                # 计算混淆矩阵
                red_cm = confusion_matrix(y_red_test, red_preds)
                evaluation_results['red_confusion_matrix'] = red_cm
                
                # 计算对数损失
                if red_probs is not None:
                    from sklearn.metrics import log_loss
                    try:
                        red_log_loss = log_loss(y_red_test, red_probs)
                        evaluation_results['red_metrics']['log_loss'] = red_log_loss
                        self.log(f"红球模型对数损失: {red_log_loss:.4f}")
                    except:
                        self.log("无法计算红球模型对数损失")
                
                # 显示详细分类报告
                self.log("红球模型分类报告:")
                report = classification_report(y_red_test, red_preds, output_dict=True)
                self.log(f"宏平均F1: {report['macro avg']['f1-score']:.4f}")
                self.log(f"加权平均F1: {report['weighted avg']['f1-score']:.4f}")
                
                # 获取特征重要性
                if hasattr(red_model, 'feature_importances_'):
                    evaluation_results['red_feature_importance'] = red_model.feature_importances_
                    # 显示前10个最重要的特征
                    importances = red_model.feature_importances_
                    indices = np.argsort(importances)[-10:]
                    self.log("红球模型前10个最重要特征:")
                    for i in reversed(indices):
                        self.log(f"特征 {i}: {importances[i]:.4f}")
        
        # 评估蓝球模型
        if 'blue' in self.models:
            blue_model = self.models['blue'].model
            # 预测类别
            blue_preds = self.models['blue'].predict(X_test)
            # 尝试获取预测概率
            try:
                blue_probs = blue_model.predict_proba(X_test)
            except:
                blue_probs = None
                
            # 计算基本指标
            blue_accuracy = accuracy_score(y_blue_test, blue_preds)
            evaluation_results['blue_metrics']['accuracy'] = blue_accuracy
            self.log(f"蓝球组合模型准确率: {blue_accuracy:.4f}")
            
            # 计算更多指标
            if len(set(y_blue_test)) > 1:
                from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
                
                # 计算精确率、召回率和F1分数
                blue_precision = precision_score(y_blue_test, blue_preds, average='weighted')
                blue_recall = recall_score(y_blue_test, blue_preds, average='weighted')
                blue_f1 = f1_score(y_blue_test, blue_preds, average='weighted')
                
                evaluation_results['blue_metrics']['precision'] = blue_precision
                evaluation_results['blue_metrics']['recall'] = blue_recall
                evaluation_results['blue_metrics']['f1'] = blue_f1
                
                self.log(f"蓝球模型精确率: {blue_precision:.4f}")
                self.log(f"蓝球模型召回率: {blue_recall:.4f}")
                self.log(f"蓝球模型F1分数: {blue_f1:.4f}")
                
                # 计算混淆矩阵
                blue_cm = confusion_matrix(y_blue_test, blue_preds)
                evaluation_results['blue_confusion_matrix'] = blue_cm
                
                # 计算对数损失
                if blue_probs is not None:
                    from sklearn.metrics import log_loss
                    try:
                        blue_log_loss = log_loss(y_blue_test, blue_probs)
                        evaluation_results['blue_metrics']['log_loss'] = blue_log_loss
                        self.log(f"蓝球模型对数损失: {blue_log_loss:.4f}")
                    except:
                        self.log("无法计算蓝球模型对数损失")
                
                # 显示详细分类报告
                self.log("蓝球模型分类报告:")
                report = classification_report(y_blue_test, blue_preds, output_dict=True)
                self.log(f"宏平均F1: {report['macro avg']['f1-score']:.4f}")
                self.log(f"加权平均F1: {report['weighted avg']['f1-score']:.4f}")
                
                # 获取特征重要性
                if hasattr(blue_model, 'feature_importances_'):
                    evaluation_results['blue_feature_importance'] = blue_model.feature_importances_
                    # 显示前10个最重要的特征
                    importances = blue_model.feature_importances_
                    indices = np.argsort(importances)[-10:]
                    self.log("蓝球模型前10个最重要特征:")
                    for i in reversed(indices):
                        self.log(f"特征 {i}: {importances[i]:.4f}")
        
        # 可视化评估结果
        if visualize:
            try:
                self._visualize_evaluation_results(evaluation_results)
            except Exception as e:
                self.log(f"可视化评估结果失败: {str(e)}")
        
        return evaluation_results
    
    def _visualize_evaluation_results(self, evaluation_results):
        """
        可视化评估结果
        
        Args:
            evaluation_results: 评估结果字典
        """
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
            import os
            
            # 创建输出目录
            output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'output', 'evaluation')
            os.makedirs(output_dir, exist_ok=True)
            
            # 设置绘图样式
            sns.set(style="whitegrid")
            plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
            plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
            
            # 1. 绘制红球和蓝球模型的各项指标对比图
            if 'red_metrics' in evaluation_results and 'blue_metrics' in evaluation_results:
                metrics = ['accuracy', 'precision', 'recall', 'f1']
                red_values = [evaluation_results['red_metrics'].get(m, 0) for m in metrics]
                blue_values = [evaluation_results['blue_metrics'].get(m, 0) for m in metrics]
                
                plt.figure(figsize=(10, 6))
                x = np.arange(len(metrics))
                width = 0.35
                
                plt.bar(x - width/2, red_values, width, label='红球')
                plt.bar(x + width/2, blue_values, width, label='蓝球')
                
                plt.title('红蓝球模型评估指标对比')
                plt.xticks(x, [m.capitalize() for m in metrics])
                plt.ylim(0, 1)
                plt.legend()
                plt.tight_layout()
                plt.savefig(os.path.join(output_dir, 'metrics_comparison.png'), dpi=300)
                plt.close()
            
            # 2. 绘制混淆矩阵热力图
            if evaluation_results['red_confusion_matrix'] is not None:
                plt.figure(figsize=(10, 8))
                sns.heatmap(evaluation_results['red_confusion_matrix'], annot=True, fmt='d', cmap='Blues')
                plt.title('红球模型混淆矩阵')
                plt.xlabel('预测类别')
                plt.ylabel('真实类别')
                plt.tight_layout()
                plt.savefig(os.path.join(output_dir, 'red_confusion_matrix.png'), dpi=300)
                plt.close()
            
            if evaluation_results['blue_confusion_matrix'] is not None:
                plt.figure(figsize=(10, 8))
                sns.heatmap(evaluation_results['blue_confusion_matrix'], annot=True, fmt='d', cmap='Blues')
                plt.title('蓝球模型混淆矩阵')
                plt.xlabel('预测类别')
                plt.ylabel('真实类别')
                plt.tight_layout()
                plt.savefig(os.path.join(output_dir, 'blue_confusion_matrix.png'), dpi=300)
                plt.close()
            
            # 3. 绘制特征重要性图
            if evaluation_results['red_feature_importance'] is not None:
                plt.figure(figsize=(12, 6))
                # 只显示前20个最重要的特征
                importance = evaluation_results['red_feature_importance']
                top_indices = np.argsort(importance)[-20:]
                top_importance = importance[top_indices]
                
                plt.barh(range(len(top_indices)), top_importance)
                plt.yticks(range(len(top_indices)), [f'特征 {idx}' for idx in top_indices])
                plt.title('红球模型 - 特征重要性 (Top 20)')
                plt.tight_layout()
                plt.savefig(os.path.join(output_dir, 'red_feature_importance.png'), dpi=300)
                plt.close()
            
            if evaluation_results['blue_feature_importance'] is not None:
                plt.figure(figsize=(12, 6))
                # 只显示前20个最重要的特征
                importance = evaluation_results['blue_feature_importance']
                top_indices = np.argsort(importance)[-20:]
                top_importance = importance[top_indices]
                
                plt.barh(range(len(top_indices)), top_importance)
                plt.yticks(range(len(top_indices)), [f'特征 {idx}' for idx in top_indices])
                plt.title('蓝球模型 - 特征重要性 (Top 20)')
                plt.tight_layout()
                plt.savefig(os.path.join(output_dir, 'blue_feature_importance.png'), dpi=300)
                plt.close()
            
            self.log(f"评估可视化结果已保存到: {output_dir}")
            
        except Exception as e:
            self.log(f"可视化过程发生错误: {str(e)}")
            import traceback
            self.log(f"错误详情: {traceback.format_exc()}")
    
    def evaluate(self, X_test, y_red_test, y_blue_test):
        """
        保持向后兼容的评估方法
        """
        return self.evaluate_enhanced(X_test, y_red_test, y_blue_test)
    
    def save_models(self, red_models=None, blue_model=None, recent_data=None, model_info=None):
        """
        保存模型、缩放器、编码器、模型权重和历史模式分析结果
        
        Args:
            red_models: 红球模型，如果为None则使用self.models['red']
            blue_model: 蓝球模型，如果为None则使用self.models['blue']
            recent_data: 包含最近开奖数据的DataFrame，用于分析历史模式
            model_info: 模型信息字典，包含训练参数、性能指标等
        """
        self.log("\n----- 保存模型和缩放器 -----")
        
        # 创建模型目录
        model_dir = os.path.join(self.models_dir, self.model_type)
        os.makedirs(model_dir, exist_ok=True)
        
        # 保存红球模型
        if red_models is not None:
            self.models['red'] = red_models
        if 'red' in self.models:
            model_path = os.path.join(model_dir, 'red_model.pkl')
            with open(model_path, 'wb') as f:
                pickle.dump(self.models['red'], f)
            self.log(f"红球模型保存到: {model_path}")
        
        # 保存蓝球模型
        if blue_model is not None:
            self.models['blue'] = blue_model
        if 'blue' in self.models:
            model_path = os.path.join(model_dir, 'blue_model.pkl')
            with open(model_path, 'wb') as f:
                pickle.dump(self.models['blue'], f)
            self.log(f"蓝球模型保存到: {model_path}")
        
        # 保存编码器
        if hasattr(self, 'red_encoder') and hasattr(self, 'blue_encoder'):
            encoders = {
                'red_encoder': self.red_encoder,
                'blue_encoder': self.blue_encoder
            }
            encoder_path = os.path.join(model_dir, 'encoders.pkl')
            with open(encoder_path, 'wb') as f:
                pickle.dump(encoders, f)
            self.log(f"编码器保存到: {encoder_path}")
        
        # 保存特征缩放器
        if hasattr(self, 'feature_scaler'):
            scaler_path = os.path.join(model_dir, 'feature_scaler.pkl')
            with open(scaler_path, 'wb') as f:
                pickle.dump(self.feature_scaler, f)
            self.log(f"特征缩放器保存到: {scaler_path}")
        elif hasattr(self, 'scalers') and 'X' in self.scalers:
            scaler_path = os.path.join(model_dir, 'scaler.pkl')
            with open(scaler_path, 'wb') as f:
                pickle.dump(self.scalers['X'], f)
            self.log(f"特征缩放器保存到: {scaler_path}")
        
        # 分析并保存历史模式
        if recent_data is not None:
            try:
                self.log("分析历史模式并保存结果...")
                pattern_results = self.analyze_historical_patterns(recent_data, visualize=False)
                
                # 保存历史模式分析结果
                pattern_path = os.path.join(model_dir, 'pattern_analysis.pkl')
                with open(pattern_path, 'wb') as f:
                    pickle.dump(pattern_results, f)
                self.log(f"历史模式分析结果保存到: {pattern_path}")
                
                # 将关键模式信息添加到模型信息中
                pattern_summary = {
                    'high_frequency_red': pattern_results['frequency']['high_frequency_red'],
                    'high_frequency_blue': pattern_results['frequency']['high_frequency_blue'],
                    'average_odd_ratio': pattern_results['odd_even']['average_odd_ratio'],
                    'average_big_ratio': pattern_results['big_small']['average_big_ratio'],
                    'average_repeat': pattern_results['repeats']['average_repeat'] if 'repeats' in pattern_results else None
                }
            except Exception as e:
                self.log(f"保存历史模式分析结果时出错: {e}")
                pattern_summary = None
        else:
            pattern_summary = None
        
        # 准备模型信息
        base_model_info = {
            'model_type': self.model_type,
            'lottery_type': self.lottery_type,
            'feature_window': self.feature_window,
            'training_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'pattern_summary': pattern_summary
        }
        
        # 如果提供了额外的模型信息，合并它
        if model_info is not None:
            base_model_info.update(model_info)
        
        # 保存模型信息
        info_path = os.path.join(model_dir, 'model_info.json')
        with open(info_path, 'w') as f:
            json.dump(base_model_info, f)
        self.log(f"模型信息保存到: {info_path}")
    
    def load_models(self):
        """
        加载保存的模型、缩放器和历史模式分析结果
        
        Returns:
            bool: 是否成功加载模型
        """
        if not CATBOOST_AVAILABLE:
            self.log("错误: CatBoost未安装或不可用，无法加载CatBoost模型")
            return False
            
        self.log(f"尝试加载{self.lottery_type}的CatBoost模型...")
        
        try:
            # 检查模型目录是否存在
            model_dir = os.path.join(self.models_dir, self.model_type)
            if not os.path.exists(model_dir):
                self.log(f"模型目录不存在: {model_dir}")
                return False
            
            # 检查模型信息文件是否存在
            info_path = os.path.join(model_dir, 'model_info.json')
            if not os.path.exists(info_path):
                self.log(f"模型信息文件不存在: {info_path}")
                return False
            
            # 加载模型信息
            with open(info_path, 'r') as f:
                model_info = json.load(f)
            self.log(f"加载模型信息成功: {model_info}")
            
            # 更新模型参数
            if 'feature_window' in model_info:
                self.feature_window = model_info['feature_window']
                self.log(f"更新特征窗口大小: {self.feature_window}")
            
            # 加载编码器
            encoder_path = os.path.join(model_dir, 'encoders.pkl')
            if os.path.exists(encoder_path):
                with open(encoder_path, 'rb') as f:
                    encoders = pickle.load(f)
                self.red_encoder = encoders['red_encoder']
                self.blue_encoder = encoders['blue_encoder']
                self.log("编码器加载成功")
            else:
                self.log("警告: 编码器文件不存在")
                return False
            
            # 加载特征缩放器
            feature_scaler_path = os.path.join(model_dir, 'feature_scaler.pkl')
            if os.path.exists(feature_scaler_path):
                with open(feature_scaler_path, 'rb') as f:
                    self.feature_scaler = pickle.load(f)
                self.log("特征缩放器加载成功")
            else:
                # 尝试加载旧版本的缩放器
                scaler_path = os.path.join(model_dir, 'scaler.pkl')
                if os.path.exists(scaler_path):
                    with open(scaler_path, 'rb') as f:
                        self.feature_scaler = pickle.load(f)
                    self.log("加载旧版本特征缩放器成功")
                else:
                    self.log("警告: 特征缩放器文件不存在，将使用未缩放的特征进行预测")
            
            # 加载红球模型
            red_model_path = os.path.join(model_dir, 'red_model.pkl')
            if os.path.exists(red_model_path):
                with open(red_model_path, 'rb') as f:
                    self.models['red'] = pickle.load(f)
                self.log("红球模型加载成功")
            else:
                self.log("警告: 红球模型文件不存在")
                return False
                
            # 加载蓝球模型
            blue_model_path = os.path.join(model_dir, 'blue_model.pkl')
            if os.path.exists(blue_model_path):
                with open(blue_model_path, 'rb') as f:
                    self.models['blue'] = pickle.load(f)
                self.log("蓝球模型加载成功")
            else:
                self.log("警告: 蓝球模型文件不存在")
                return False
            
            # 尝试加载历史模式分析结果
            pattern_path = os.path.join(model_dir, 'pattern_analysis.pkl')
            if os.path.exists(pattern_path):
                try:
                    with open(pattern_path, 'rb') as f:
                        self.pattern_results = pickle.load(f)
                    self.log("历史模式分析结果加载成功")
                    
                    # 打印一些关键的模式信息
                    if 'frequency' in self.pattern_results:
                        self.log(f"高频红球: {self.pattern_results['frequency']['high_frequency_red']}")
                        self.log(f"高频蓝球: {self.pattern_results['frequency']['high_frequency_blue']}")
                    if 'odd_even' in self.pattern_results:
                        self.log(f"平均奇数比例: {self.pattern_results['odd_even']['average_odd_ratio']:.2f}")
                    if 'repeats' in self.pattern_results:
                        self.log(f"平均重复红球数: {self.pattern_results['repeats']['average_repeat']:.2f}")
                except Exception as e:
                    self.log(f"加载历史模式分析结果时出错: {e}，将在预测时重新分析")
                    self.pattern_results = None
            else:
                self.log("历史模式分析结果文件不存在，将在预测时重新分析")
                self.pattern_results = None
            
            self.log("所有模型和编码器加载成功")
            return True
                
        except Exception as e:
            self.log(f"加载模型时出错: {e}")
            import traceback
            self.log(traceback.format_exc())
            return False
    
    def predict(self, recent_data):
        """
        生成预测结果 - 增强优化版本
        
        Args:
            recent_data: 包含最近开奖数据的DataFrame
            
        Returns:
            预测的红球和蓝球号码
        """
        # 使用历史模式分析结果或重新分析
        if hasattr(self, 'pattern_results') and self.pattern_results is not None:
            self.log("使用已加载的历史模式分析结果...")
            pattern_results = self.pattern_results
        else:
            self.log("分析历史模式以辅助预测...")
            pattern_results = self.analyze_historical_patterns(recent_data, visualize=False)
            # 保存分析结果以便下次使用
            self.pattern_results = pattern_results
        if not CATBOOST_AVAILABLE:
            self.log("错误: CatBoost未安装或不可用，无法使用CatBoost模型进行预测")
            raise ImportError("CatBoost未安装或不可用，请先安装CatBoost")
            
        # 检查模型和编码器是否已加载
        if 'red' not in self.models or 'blue' not in self.models:
            self.log("模型未加载，尝试重新加载...")
            load_success = self.load_models()
            if not load_success:
                self.log("错误：模型加载失败，请先训练模型")
                raise ValueError("模型未正确加载，请先训练或加载模型。")
        
        # 检查编码器是否存在
        if not hasattr(self, 'red_encoder') or not hasattr(self, 'blue_encoder'):
            self.log("编码器未加载，尝试重新加载...")
            load_success = self.load_models()
            if not load_success:
                self.log("错误：编码器加载失败")
                raise ValueError("编码器未正确加载，请先训练模型。")
        
        # 提取红蓝球列名
        if self.lottery_type == 'dlt':
            red_cols = [col for col in recent_data.columns if col.startswith('红球_')][:5]
            blue_cols = [col for col in recent_data.columns if col.startswith('蓝球_')][:2]
        else:  # ssq
            red_cols = [col for col in recent_data.columns if col.startswith('红球_')][:6]
            blue_cols = [col for col in recent_data.columns if col.startswith('蓝球_')][:1]
            
        # 确保数据按期数降序排列
        recent_data = recent_data.sort_values('期数', ascending=False).reset_index(drop=True)
        
        # 确保有足够的历史数据
        if len(recent_data) < self.feature_window:
            self.log(f"历史数据不足，需要至少 {self.feature_window} 期")
            raise ValueError(f"历史数据不足，需要至少 {self.feature_window} 期数据。当前仅有 {len(recent_data)} 期。")
        
        # 准备预测特征
        window_df = recent_data.head(self.feature_window)
        
        # 基础特征
        basic_features = []
        for j in range(self.feature_window):
            for col in red_cols + blue_cols:
                basic_features.append(window_df.iloc[j][col])
        
        # 统计特征
        stat_features = self._extract_statistical_features(window_df, red_cols, blue_cols)
        
        # 趋势特征
        trend_features = self._extract_trend_features(window_df, red_cols, blue_cols)
        
        # 组合所有特征
        all_features = basic_features + stat_features + trend_features
        X = np.array([all_features])
        
        # 应用特征缩放
        try:
            if hasattr(self, 'feature_scaler'):
                X_scaled = self.feature_scaler.transform(X)
                self.log("使用训练时的特征缩放器进行预测")
            else:
                self.log("警告: 未找到特征缩放器，使用原始特征")
                X_scaled = X
        except Exception as e:
            self.log(f"应用特征缩放时出错: {e}，使用原始特征")
            X_scaled = X
        
        try:
            # 预测红球组合并获取概率
            red_model = self.models['red'].model
            red_pred_encoded = red_model.predict(X_scaled)[0]
            red_probs = red_model.predict_proba(X_scaled)[0]
            
            # 获取前15个最高概率的类别
            top_indices = np.argsort(red_probs)[-15:]
            top_probs = red_probs[top_indices]
            
            # 解码为实际号码组合
            red_combo_candidates = []
            for idx, prob in zip(top_indices[::-1], top_probs[::-1]):  # 反转以获得降序
                combo_str = self.red_encoder.inverse_transform([idx])[0]
                combo = eval(combo_str)  # 解析字符串表示的元组
                red_combo_candidates.append((list(combo), prob))
            
            # 记录概率信息
            self.log(f"红球组合候选(概率):\n{', '.join([f'{c}({p:.4f})' for c, p in red_combo_candidates[:5]])}")
            
            # 使用历史模式分析结果优化预测
            high_freq_red = pattern_results['frequency']['high_frequency_red']
            low_freq_red = pattern_results['frequency']['low_frequency_red']
            avg_odd_ratio = pattern_results['odd_even']['average_odd_ratio']
            avg_big_ratio = pattern_results['big_small']['average_big_ratio']
            avg_intervals = pattern_results['intervals']['average_intervals']
            avg_repeat = pattern_results['repeats']['average_repeat']
            
            # 计算每个候选组合的模式匹配分数
            red_combo_scores = []
            for combo, prob in red_combo_candidates:
                score = prob  # 基础分数是模型预测概率
                
                # 1. 高频号码匹配加分
                high_freq_match = len(set(combo).intersection(set(high_freq_red)))
                score += 0.02 * high_freq_match
                
                # 2. 低频号码匹配减分（适度）
                low_freq_match = len(set(combo).intersection(set(low_freq_red)))
                score -= 0.01 * low_freq_match
                
                # 3. 奇偶比例接近历史平均值加分
                odd_count = sum(1 for num in combo if num % 2 == 1)
                combo_odd_ratio = odd_count / len(combo)
                odd_ratio_diff = abs(combo_odd_ratio - avg_odd_ratio)
                score += 0.05 * (1 - odd_ratio_diff)  # 差异越小，加分越多
                
                # 4. 大小比例接近历史平均值加分
                red_mid = self.red_range / 2
                big_count = sum(1 for num in combo if num > red_mid)
                combo_big_ratio = big_count / len(combo)
                big_ratio_diff = abs(combo_big_ratio - avg_big_ratio)
                score += 0.05 * (1 - big_ratio_diff)  # 差异越小，加分越多
                
                # 5. 区间分布接近历史平均值加分
                interval_size = self.red_range // 3
                interval_1 = sum(1 for num in combo if 1 <= num <= interval_size)
                interval_2 = sum(1 for num in combo if interval_size < num <= 2*interval_size)
                interval_3 = sum(1 for num in combo if 2*interval_size < num <= self.red_range)
                combo_intervals = [interval_1, interval_2, interval_3]
                interval_diff = sum(abs(a - b) for a, b in zip(combo_intervals, avg_intervals))
                score += 0.05 * (3 - interval_diff) / 3  # 差异越小，加分越多
                
                # 6. 重复号码接近历史平均值加分
                if len(recent_data) > 0:
                    last_draw = [recent_data.iloc[0][col] for col in red_cols]
                    repeat_count = len(set(combo).intersection(set(last_draw)))
                    repeat_diff = abs(repeat_count - avg_repeat)
                    score += 0.05 * (1 - min(repeat_diff, 1))  # 差异越小，加分越多
                
                # 7. 连号分析
                sorted_combo = sorted(combo)
                consecutive_count = sum(1 for i in range(1, len(sorted_combo)) if sorted_combo[i] == sorted_combo[i-1] + 1)
                # 适度惩罚过多的连号
                if consecutive_count > 2:
                    score -= 0.03 * (consecutive_count - 2)
                
                red_combo_scores.append((combo, score))
            
            # 按优化后的分数排序
            red_combo_scores.sort(key=lambda x: x[1], reverse=True)
            
            # 记录优化后的分数
            self.log(f"红球组合优化后分数:\n{', '.join([f'{c}({s:.4f})' for c, s in red_combo_scores[:5]])}")
            
            # 选择最高分数的组合，但检查是否与历史重复
            red_predictions = list(red_combo_scores[0][0])
            
            # 检查是否与历史数据重复
            is_duplicate = False
            for i in range(min(10, len(recent_data))):
                history_red = [recent_data.iloc[i][col] for col in red_cols]
                if set(red_predictions) == set(history_red):
                    is_duplicate = True
                    self.log(f"警告: 预测的红球组合与第{i+1}期历史数据相同")
                    break
            
            # 如果重复，尝试使用第二高分数的组合
            if is_duplicate and len(red_combo_scores) > 1:
                red_predictions = list(red_combo_scores[1][0])
                self.log(f"使用第二高分数的红球组合: {red_predictions}")
            
            # 预测蓝球组合并获取概率
            blue_model = self.models['blue'].model
            blue_pred_encoded = blue_model.predict(X_scaled)[0]
            blue_probs = blue_model.predict_proba(X_scaled)[0]
            
            # 获取前8个最高概率的类别
            top_blue_indices = np.argsort(blue_probs)[-8:]
            top_blue_probs = blue_probs[top_blue_indices]
            
            # 解码为实际号码组合
            blue_combo_candidates = []
            for idx, prob in zip(top_blue_indices[::-1], top_blue_probs[::-1]):
                combo_str = self.blue_encoder.inverse_transform([idx])[0]
                combo = eval(combo_str)  # 解析字符串表示的元组
                blue_combo_candidates.append((list(combo), prob))
            
            # 记录概率信息
            self.log(f"蓝球组合候选(概率):\n{', '.join([f'{c}({p:.4f})' for c, p in blue_combo_candidates[:3]])}")
            
            # 使用历史模式分析结果优化蓝球预测
            high_freq_blue = pattern_results['frequency']['high_frequency_blue']
            low_freq_blue = pattern_results['frequency']['low_frequency_blue']
            
            # 计算每个蓝球候选组合的模式匹配分数
            blue_combo_scores = []
            for combo, prob in blue_combo_candidates:
                score = prob  # 基础分数是模型预测概率
                
                # 1. 高频号码匹配加分
                high_freq_match = len(set(combo).intersection(set(high_freq_blue)))
                score += 0.03 * high_freq_match
                
                # 2. 低频号码匹配减分（适度）
                low_freq_match = len(set(combo).intersection(set(low_freq_blue)))
                score -= 0.015 * low_freq_match
                
                blue_combo_scores.append((combo, score))
            
            # 按优化后的分数排序
            blue_combo_scores.sort(key=lambda x: x[1], reverse=True)
            
            # 记录优化后的分数
            self.log(f"蓝球组合优化后分数:\n{', '.join([f'{c}({s:.4f})' for c, s in blue_combo_scores[:3]])}")
            
            # 如果最高分数显著高于第二高分数，直接选择最高分数组合
            if len(blue_combo_scores) > 1 and blue_combo_scores[0][1] > 1.5 * blue_combo_scores[1][1]:
                blue_predictions = list(blue_combo_scores[0][0])
                self.log("蓝球最高分数显著高于其他，直接选择最高分数组合")
            else:
                # 否则，根据分数权重随机选择前3个
                weights = np.array([s for _, s in blue_combo_scores[:3]])
                weights = weights / np.sum(weights)  # 归一化权重
                selected_idx = np.random.choice(min(3, len(blue_combo_scores)), p=weights)
                blue_predictions = list(blue_combo_scores[selected_idx][0])
                self.log(f"蓝球根据分数权重随机选择组合: {blue_predictions}")
            
            self.log(f"预测的红球组合: {red_predictions}")
            self.log(f"预测的蓝球组合: {blue_predictions}")
            
        except Exception as e:
            self.log(f"预测过程中出错: {e}")
            import traceback
            self.log(traceback.format_exc())
            
            # 回退到随机预测
            self.log("使用回退策略生成预测...")
            red_predictions = self._generate_fallback_red_prediction()
            blue_predictions = self._generate_fallback_blue_prediction()
        
        # 验证和调整预测结果
        red_predictions = self._validate_red_prediction(red_predictions)
        blue_predictions = self._validate_blue_prediction(blue_predictions)
        
        # 保存本次预测的概率信息，用于后续分析
        self.last_prediction_info = {
            'red_candidates': red_combo_candidates if 'red_combo_candidates' in locals() else [],
            'blue_candidates': blue_combo_candidates if 'blue_combo_candidates' in locals() else [],
            'final_red': red_predictions,
            'final_blue': blue_predictions
        }
        
        self.log(f"最终预测结果: 红球 {red_predictions}, 蓝球 {blue_predictions}")
        
        return red_predictions, blue_predictions
    
    def _generate_fallback_red_prediction(self):
        """
        生成回退红球预测
        """
        red_predictions = []
        while len(red_predictions) < self.red_count:
            new_num = np.random.randint(1, self.red_range + 1)
            if new_num not in red_predictions:
                red_predictions.append(new_num)
        return sorted(red_predictions)
    
    def _generate_fallback_blue_prediction(self):
        """
        生成回退蓝球预测
        """
        blue_predictions = []
        while len(blue_predictions) < self.blue_count:
            new_num = np.random.randint(1, self.blue_range + 1)
            if new_num not in blue_predictions:
                blue_predictions.append(new_num)
        return blue_predictions
    
    def _validate_red_prediction(self, red_predictions):
        """
        验证和调整红球预测
        """
        # 确保是列表
        if not isinstance(red_predictions, list):
            red_predictions = list(red_predictions)
        
        # 去重并排序
        red_predictions = sorted(list(set(red_predictions)))
        
        # 确保在有效范围内
        red_predictions = [num for num in red_predictions if 1 <= num <= self.red_range]
        
        # 补充到所需数量
        while len(red_predictions) < self.red_count:
            new_num = np.random.randint(1, self.red_range + 1)
            if new_num not in red_predictions:
                red_predictions.append(new_num)
        
        # 截取到所需数量并排序
        return sorted(red_predictions[:self.red_count])
    
    def _validate_blue_prediction(self, blue_predictions):
        """
        验证和调整蓝球预测
        """
        # 确保是列表
        if not isinstance(blue_predictions, list):
            blue_predictions = list(blue_predictions)
        
        # 去重
        blue_predictions = list(set(blue_predictions))
        
        # 确保在有效范围内
        blue_predictions = [num for num in blue_predictions if 1 <= num <= self.blue_range]
        
        # 补充到所需数量
        while len(blue_predictions) < self.blue_count:
            new_num = np.random.randint(1, self.blue_range + 1)
            if new_num not in blue_predictions:
                blue_predictions.append(new_num)
        
        # 截取到所需数量
        return blue_predictions[:self.blue_count]
        
    def update_pattern_analysis(self, recent_data, save_results=True):
        """
        更新历史模式分析结果
        
        Args:
            recent_data: 包含最近开奖数据的DataFrame
            save_results: 是否保存更新后的结果
            
        Returns:
            dict: 更新后的历史模式分析结果
        """
        self.log("\n----- 更新历史模式分析 -----")
        
        # 重新分析历史模式
        self.pattern_results = self.analyze_historical_patterns(recent_data, visualize=False)
        
        # 保存更新后的结果
        if save_results:
            try:
                # 创建模型目录
                model_dir = os.path.join(self.models_dir, self.model_type)
                os.makedirs(model_dir, exist_ok=True)
                
                # 保存历史模式分析结果
                pattern_path = os.path.join(model_dir, 'pattern_analysis.pkl')
                with open(pattern_path, 'wb') as f:
                    pickle.dump(self.pattern_results, f)
                self.log(f"更新后的历史模式分析结果保存到: {pattern_path}")
                
                # 更新模型信息中的模式摘要
                info_path = os.path.join(model_dir, 'model_info.json')
                if os.path.exists(info_path):
                    with open(info_path, 'r') as f:
                        model_info = json.load(f)
                    
                    # 更新模式摘要
                    pattern_summary = {
                        'high_frequency_red': self.pattern_results['frequency']['high_frequency_red'],
                        'high_frequency_blue': self.pattern_results['frequency']['high_frequency_blue'],
                        'average_odd_ratio': self.pattern_results['odd_even']['average_odd_ratio'],
                        'average_big_ratio': self.pattern_results['big_small']['average_big_ratio'],
                        'average_repeat': self.pattern_results['repeats']['average_repeat'] if 'repeats' in self.pattern_results else None
                    }
                    model_info['pattern_summary'] = pattern_summary
                    
                    # 保存更新后的模型信息
                    with open(info_path, 'w') as f:
                        json.dump(model_info, f)
                    self.log(f"模型信息中的模式摘要已更新")
            except Exception as e:
                self.log(f"保存更新后的历史模式分析结果时出错: {e}")
                import traceback
                self.log(traceback.format_exc())
        
        return self.pattern_results
        
    def analyze_historical_patterns(self, data, output_dir=None, visualize=True):
        """
        分析历史数据的模式和趋势
        
        Args:
            data: 包含历史开奖数据的DataFrame
            output_dir: 可视化结果保存目录，默认为None（不保存）
            visualize: 是否生成可视化结果
            
        Returns:
            dict: 包含分析结果的字典
        """
        self.log("\n----- 分析历史数据模式 -----")
        
        # 确保数据按期数升序排列
        data = data.sort_values('期数', ascending=True).reset_index(drop=True)
        
        # 提取红蓝球列名
        if self.lottery_type == 'dlt':
            red_cols = [col for col in data.columns if col.startswith('红球_')][:5]
            blue_cols = [col for col in data.columns if col.startswith('蓝球_')][:2]
        else:  # ssq
            red_cols = [col for col in data.columns if col.startswith('红球_')][:6]
            blue_cols = [col for col in data.columns if col.startswith('蓝球_')][:1]
        
        results = {}
        
        # 1. 频率分析
        red_freq = {}
        for i in range(1, self.red_range + 1):
            red_freq[i] = 0
            
        blue_freq = {}
        for i in range(1, self.blue_range + 1):
            blue_freq[i] = 0
            
        for _, row in data.iterrows():
            for col in red_cols:
                num = row[col]
                if 1 <= num <= self.red_range:
                    red_freq[num] += 1
                    
            for col in blue_cols:
                num = row[col]
                if 1 <= num <= self.blue_range:
                    blue_freq[num] += 1
        
        # 计算出现频率
        total_draws = len(data)
        red_freq_pct = {k: v/total_draws/len(red_cols) for k, v in red_freq.items()}
        blue_freq_pct = {k: v/total_draws/len(blue_cols) for k, v in blue_freq.items()}
        
        # 找出高频和低频号码
        red_sorted = sorted(red_freq.items(), key=lambda x: x[1], reverse=True)
        blue_sorted = sorted(blue_freq.items(), key=lambda x: x[1], reverse=True)
        
        high_freq_red = [k for k, v in red_sorted[:self.red_count]]
        low_freq_red = [k for k, v in red_sorted[-self.red_count:]]
        high_freq_blue = [k for k, v in blue_sorted[:self.blue_count]]
        low_freq_blue = [k for k, v in blue_sorted[-self.blue_count:]]
        
        results['frequency'] = {
            'red_frequency': red_freq,
            'blue_frequency': blue_freq,
            'red_frequency_pct': red_freq_pct,
            'blue_frequency_pct': blue_freq_pct,
            'high_frequency_red': high_freq_red,
            'low_frequency_red': low_freq_red,
            'high_frequency_blue': high_freq_blue,
            'low_frequency_blue': low_freq_blue
        }
        
        self.log(f"高频红球: {high_freq_red}")
        self.log(f"低频红球: {low_freq_red}")
        self.log(f"高频蓝球: {high_freq_blue}")
        self.log(f"低频蓝球: {low_freq_blue}")
        
        # 2. 奇偶比例分析
        odd_even_ratios = []
        for _, row in data.iterrows():
            red_numbers = [row[col] for col in red_cols]
            odd_count = sum(1 for num in red_numbers if num % 2 == 1)
            even_count = len(red_numbers) - odd_count
            odd_even_ratios.append(odd_count / len(red_numbers) if len(red_numbers) > 0 else 0)
        
        avg_odd_ratio = np.mean(odd_even_ratios)
        results['odd_even'] = {
            'odd_even_ratios': odd_even_ratios,
            'average_odd_ratio': avg_odd_ratio
        }
        
        self.log(f"平均奇数比例: {avg_odd_ratio:.2f}")
        
        # 3. 大小比例分析 (大: >红球范围/2, 小: <=红球范围/2)
        big_small_ratios = []
        red_mid = self.red_range / 2
        for _, row in data.iterrows():
            red_numbers = [row[col] for col in red_cols]
            big_count = sum(1 for num in red_numbers if num > red_mid)
            small_count = len(red_numbers) - big_count
            big_small_ratios.append(big_count / len(red_numbers) if len(red_numbers) > 0 else 0)
        
        avg_big_ratio = np.mean(big_small_ratios)
        results['big_small'] = {
            'big_small_ratios': big_small_ratios,
            'average_big_ratio': avg_big_ratio
        }
        
        self.log(f"平均大号比例: {avg_big_ratio:.2f}")
        
        # 4. 区间分布分析
        # 将红球范围分为3个区间
        interval_size = self.red_range // 3
        interval_counts = []
        
        for _, row in data.iterrows():
            red_numbers = [row[col] for col in red_cols]
            interval_1 = sum(1 for num in red_numbers if 1 <= num <= interval_size)
            interval_2 = sum(1 for num in red_numbers if interval_size < num <= 2*interval_size)
            interval_3 = sum(1 for num in red_numbers if 2*interval_size < num <= self.red_range)
            interval_counts.append([interval_1, interval_2, interval_3])
        
        interval_counts = np.array(interval_counts)
        avg_intervals = np.mean(interval_counts, axis=0)
        
        results['intervals'] = {
            'interval_counts': interval_counts.tolist(),
            'average_intervals': avg_intervals.tolist()
        }
        
        self.log(f"区间分布平均值: {avg_intervals}")
        
        # 5. 和值趋势分析
        sum_values = []
        for _, row in data.iterrows():
            red_sum = sum(row[col] for col in red_cols)
            blue_sum = sum(row[col] for col in blue_cols)
            sum_values.append((red_sum, blue_sum))
        
        red_sums = [x[0] for x in sum_values]
        blue_sums = [x[1] for x in sum_values]
        
        avg_red_sum = np.mean(red_sums)
        std_red_sum = np.std(red_sums)
        avg_blue_sum = np.mean(blue_sums)
        std_blue_sum = np.std(blue_sums)
        
        results['sum_values'] = {
            'red_sums': red_sums,
            'blue_sums': blue_sums,
            'average_red_sum': avg_red_sum,
            'std_red_sum': std_red_sum,
            'average_blue_sum': avg_blue_sum,
            'std_blue_sum': std_blue_sum
        }
        
        self.log(f"红球和值平均值: {avg_red_sum:.2f}, 标准差: {std_red_sum:.2f}")
        self.log(f"蓝球和值平均值: {avg_blue_sum:.2f}, 标准差: {std_blue_sum:.2f}")
        
        # 6. 重复号码趋势
        repeat_counts = []
        for i in range(1, len(data)):
            prev_red = set(data.iloc[i-1][col] for col in red_cols)
            curr_red = set(data.iloc[i][col] for col in red_cols)
            repeat_count = len(prev_red.intersection(curr_red))
            repeat_counts.append(repeat_count)
        
        avg_repeat = np.mean(repeat_counts) if repeat_counts else 0
        results['repeats'] = {
            'repeat_counts': repeat_counts,
            'average_repeat': avg_repeat
        }
        
        self.log(f"平均重复红球数: {avg_repeat:.2f}")
        
        # 7. 连号分析
        consecutive_counts = []
        for _, row in data.iterrows():
            red_numbers = sorted([row[col] for col in red_cols])
            consecutive_count = 0
            for i in range(1, len(red_numbers)):
                if red_numbers[i] == red_numbers[i-1] + 1:
                    consecutive_count += 1
            consecutive_counts.append(consecutive_count)
        
        avg_consecutive = np.mean(consecutive_counts)
        results['consecutive'] = {
            'consecutive_counts': consecutive_counts,
            'average_consecutive': avg_consecutive
        }
        
        self.log(f"平均连号数: {avg_consecutive:.2f}")
        
        # 可视化结果
        if visualize:
            try:
                import matplotlib.pyplot as plt
                import seaborn as sns
                
                # 创建输出目录
                if output_dir:
                    os.makedirs(output_dir, exist_ok=True)
                
                # 设置风格
                sns.set(style="whitegrid")
                plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文
                plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
                
                # 1. 频率分析图
                plt.figure(figsize=(12, 6))
                plt.bar(red_freq.keys(), red_freq.values(), color='red', alpha=0.7)
                plt.title('红球出现频率')
                plt.xlabel('号码')
                plt.ylabel('出现次数')
                plt.xticks(range(1, self.red_range + 1))
                plt.grid(True, linestyle='--', alpha=0.7)
                if output_dir:
                    plt.savefig(os.path.join(output_dir, 'red_frequency.png'), dpi=300, bbox_inches='tight')
                plt.close()
                
                plt.figure(figsize=(12, 6))
                plt.bar(blue_freq.keys(), blue_freq.values(), color='blue', alpha=0.7)
                plt.title('蓝球出现频率')
                plt.xlabel('号码')
                plt.ylabel('出现次数')
                plt.xticks(range(1, self.blue_range + 1))
                plt.grid(True, linestyle='--', alpha=0.7)
                if output_dir:
                    plt.savefig(os.path.join(output_dir, 'blue_frequency.png'), dpi=300, bbox_inches='tight')
                plt.close()
                
                # 2. 奇偶比例趋势
                plt.figure(figsize=(12, 6))
                plt.plot(range(len(odd_even_ratios)), odd_even_ratios, marker='o', linestyle='-', alpha=0.7)
                plt.axhline(y=0.5, color='r', linestyle='--', alpha=0.7)
                plt.title('奇数比例趋势')
                plt.xlabel('期数')
                plt.ylabel('奇数比例')
                plt.grid(True, linestyle='--', alpha=0.7)
                if output_dir:
                    plt.savefig(os.path.join(output_dir, 'odd_even_ratio.png'), dpi=300, bbox_inches='tight')
                plt.close()
                
                # 3. 和值趋势
                plt.figure(figsize=(12, 6))
                plt.plot(range(len(red_sums)), red_sums, marker='o', linestyle='-', color='red', alpha=0.7)
                plt.axhline(y=avg_red_sum, color='r', linestyle='--', alpha=0.7, label=f'平均值: {avg_red_sum:.2f}')
                plt.title('红球和值趋势')
                plt.xlabel('期数')
                plt.ylabel('和值')
                plt.legend()
                plt.grid(True, linestyle='--', alpha=0.7)
                if output_dir:
                    plt.savefig(os.path.join(output_dir, 'red_sum_trend.png'), dpi=300, bbox_inches='tight')
                plt.close()
                
                # 4. 区间分布热图
                plt.figure(figsize=(12, 6))
                sns.heatmap(interval_counts.T, cmap='YlOrRd', cbar=True)
                plt.title('区间分布热图')
                plt.xlabel('期数')
                plt.ylabel('区间')
                plt.yticks([0.5, 1.5, 2.5], ['区间1', '区间2', '区间3'])
                if output_dir:
                    plt.savefig(os.path.join(output_dir, 'interval_heatmap.png'), dpi=300, bbox_inches='tight')
                plt.close()
                
                # 5. 重复号码趋势
                plt.figure(figsize=(12, 6))
                plt.plot(range(len(repeat_counts)), repeat_counts, marker='o', linestyle='-', alpha=0.7)
                plt.axhline(y=avg_repeat, color='r', linestyle='--', alpha=0.7, label=f'平均值: {avg_repeat:.2f}')
                plt.title('重复号码趋势')
                plt.xlabel('期数')
                plt.ylabel('重复数量')
                plt.legend()
                plt.grid(True, linestyle='--', alpha=0.7)
                if output_dir:
                    plt.savefig(os.path.join(output_dir, 'repeat_trend.png'), dpi=300, bbox_inches='tight')
                plt.close()
                
                self.log(f"分析可视化结果已保存到: {output_dir if output_dir else '未保存'}")
                
            except Exception as e:
                self.log(f"可视化过程发生错误: {str(e)}")
                import traceback
                self.log(f"错误详情: {traceback.format_exc()}")
        
        return results