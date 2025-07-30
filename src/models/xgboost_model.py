# -*- coding: utf-8 -*-
"""
XGBoost model for lottery prediction
优化版本：增强参数调优、特征工程和模型评估
"""

import os
import sys
import time
import uuid
import json
import pickle
import platform
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Any, Dict, List, Optional, Tuple, Union, Callable
from datetime import datetime
from collections import Counter
from scipy import stats
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split, RandomizedSearchCV, KFold, StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, log_loss, confusion_matrix
from xgboost import XGBClassifier, callback

# 添加项目根目录到Python路径
script_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(os.path.dirname(script_dir))
if project_dir not in sys.path:
    sys.path.append(project_dir)

from src.models.base import BaseMLModel
from src.utils.device_utils import check_device_availability


class WrappedXGBoostModel:
    """
    包装XGBoost模型，提供统一的预测接口
    支持概率预测和top-k预测
    """
    
    def __init__(self, model, reverse_mapping=None):
        """
        初始化包装模型
        
        Args:
            model: XGBoost模型
            reverse_mapping: 标签逆映射字典，用于将预测的类别索引转换回原始标签
        """
        self.model = model
        self.reverse_mapping = reverse_mapping
    
    def predict(self, X):
        """
        预测类别
        
        Args:
            X: 特征数据
            
        Returns:
            预测的类别
        """
        return self.model.predict(X)
    
    def predict_proba(self, X):
        """
        预测概率分布
        
        Args:
            X: 特征数据
            
        Returns:
            预测的概率分布
        """
        return self.model.predict_proba(X)
    
    def predict_top_k(self, X, k=5):
        """
        预测概率最高的k个类别
        
        Args:
            X: 特征数据
            k: 返回的类别数量
            
        Returns:
            (top_k_indices, top_k_probs): 概率最高的k个类别索引和对应的概率
        """
        proba = self.predict_proba(X)
        top_k_indices = np.argsort(proba, axis=1)[:, -k:][:, ::-1]
        top_k_probs = np.array([proba[i, top_k_indices[i]] for i in range(len(X))])
        
        # 如果有逆映射，应用到索引上
        if self.reverse_mapping is not None:
            mapped_indices = np.zeros_like(top_k_indices)
            for i in range(top_k_indices.shape[0]):
                for j in range(top_k_indices.shape[1]):
                    idx = top_k_indices[i, j]
                    mapped_indices[i, j] = self.reverse_mapping.get(idx, idx)
            return mapped_indices, top_k_probs
        
        return top_k_indices, top_k_probs


class XGBoostModel(BaseMLModel):
    """
    XGBoost模型用于彩票预测
    优化版本：增强参数调优、特征工程和模型评估
    """
    
    def __init__(self, lottery_type='dlt', feature_window=10, log_callback=None, use_gpu=False):
        """
        初始化XGBoost模型
        
        Args:
            lottery_type: 彩票类型，'dlt'或'ssq'
            feature_window: 特征窗口大小，使用多少期数据作为特征
            log_callback: 日志回调函数，用于将日志发送到UI
            use_gpu: 是否使用GPU训练
        """
        super().__init__(lottery_type, feature_window, log_callback, use_gpu)
        self.model_type = 'xgboost'
        self.current_model_version = None
        
        # 检查GPU可用性
        if use_gpu:
            device_info = check_device_availability()
            self.use_gpu = device_info['gpu_available']
            if self.use_gpu:
                self.log(f"将使用GPU训练XGBoost模型: {device_info['gpu_name']}")
            else:
                self.log("GPU不可用，将使用CPU训练XGBoost模型")
        else:
            self.use_gpu = False
            self.log("将使用CPU训练XGBoost模型")
    
    def train(self, data, test_size=0.2, random_state=42, n_iter_search=50, cv=5, early_stopping_rounds=20):
        """
        训练XGBoost模型
        
        Args:
            data: 训练数据DataFrame
            test_size: 测试集比例
            random_state: 随机种子
            n_iter_search: 随机搜索迭代次数
            cv: 交叉验证折数
            early_stopping_rounds: 早停轮数
            
        Returns:
            训练结果字典
        """
        self.log(f"开始训练{self.lottery_type}的XGBoost模型...")
        start_time = time.time()
        
        # 准备训练数据
        X_train, X_test, red_train_data, red_test_data, blue_train_data, blue_test_data = self.prepare_data(data, test_size)
        
        # 训练红球模型
        self.log("训练红球模型...")
        red_models = []
        red_train_results = []
        
        for i, (y_train, y_test) in enumerate(zip(red_train_data, red_test_data)):
            position = i + 1
            self.log(f"训练红球位置{position}模型...")
            
            model, train_result = self.train_xgboost(
                X_train, y_train, X_test, y_test,
                ball_type='red', position=position,
                n_iter_search=n_iter_search, cv=cv,
                early_stopping_rounds=early_stopping_rounds,
                random_state=random_state
            )
            
            red_models.append(model)
            red_train_results.append(train_result)
        
        # 训练蓝球模型
        self.log("训练蓝球模型...")
        blue_models = []
        blue_train_results = []
        
        for i, (y_train, y_test) in enumerate(zip(blue_train_data, blue_test_data)):
            position = i + 1
            self.log(f"训练蓝球位置{position}模型...")
            
            model, train_result = self.train_xgboost(
                X_train, y_train, X_test, y_test,
                ball_type='blue', position=position,
                n_iter_search=n_iter_search, cv=cv,
                early_stopping_rounds=early_stopping_rounds,
                random_state=random_state
            )
            
            blue_models.append(model)
            blue_train_results.append(train_result)
        
        # 保存模型
        self.models['red'] = red_models
        self.models['blue'] = blue_models
        
        # 计算总训练时间
        train_time = time.time() - start_time
        self.log(f"XGBoost模型训练完成，耗时{train_time:.2f}秒")
        
        # 评估模型
        self.log("评估模型性能...")
        evaluation_results = self.evaluate(X_test, red_test_data, blue_test_data)
        
        # 保存模型和训练结果
        self.log("保存模型...")
        save_result = self.save_models(red_train_results, blue_train_results, evaluation_results)
        
        # 返回训练结果
        return {
            'red_models': red_models,
            'blue_models': blue_models,
            'red_train_results': red_train_results,
            'blue_train_results': blue_train_results,
            'evaluation_results': evaluation_results,
            'train_time': train_time,
            'save_result': save_result
        }
    
    def train_xgboost(self, X_train, y_train, X_test, y_test, ball_type, position, 
                     n_iter_search=50, cv=5, early_stopping_rounds=20, random_state=42):
        """
        训练单个XGBoost模型
        
        Args:
            X_train: 训练特征
            y_train: 训练标签
            X_test: 测试特征
            y_test: 测试标签
            ball_type: 球类型 ('red' 或 'blue')
            position: 位置索引
            n_iter_search: 随机搜索迭代次数
            cv: 交叉验证折数
            early_stopping_rounds: 早停轮数
            random_state: 随机种子
            
        Returns:
            (model, train_result): 训练好的模型和训练结果
        """
        # 处理标签映射，确保标签是连续的0-based索引
        # 获取训练数据中的唯一标签
        unique_train_labels = np.unique(y_train)
        unique_test_labels = np.unique(y_test)
        all_unique_labels = np.unique(np.concatenate([unique_train_labels, unique_test_labels]))
        
        # 创建连续的标签映射：将实际标签映射到0, 1, 2, ..., n-1
        label_mapping = {label: i for i, label in enumerate(sorted(all_unique_labels))}
        reverse_mapping = {i: label for label, i in label_mapping.items()}
        
        # 应用标签映射，确保标签是连续的
        y_train_mapped = np.array([label_mapping[label] for label in y_train], dtype=int)
        y_test_mapped = np.array([label_mapping[label] for label in y_test], dtype=int)
        
        # 确定映射后的标签范围
        max_label_value = len(all_unique_labels) - 1
        
        self.log(f"原始标签范围: {sorted(all_unique_labels)}")
        self.log(f"映射后标签范围: 0-{max_label_value}")
        self.log(f"标签映射: {label_mapping}")
        
        # 检查标签是否在合理范围内
        if ball_type == 'red':
            max_label = self.red_range - 1
        else:  # blue
            max_label = self.blue_range - 1
        
        # 检查是否有超出范围的标签
        out_of_range = (y_train_mapped < 0) | (y_train_mapped > max_label)
        if np.any(out_of_range):
            self.log(f"警告: 发现{np.sum(out_of_range)}个超出范围的{ball_type}球标签，已调整为有效范围")
            y_train_mapped = np.clip(y_train_mapped, 0, max_label)
        
        out_of_range = (y_test_mapped < 0) | (y_test_mapped > max_label)
        if np.any(out_of_range):
            self.log(f"警告: 发现{np.sum(out_of_range)}个超出范围的{ball_type}球测试标签，已调整为有效范围")
            y_test_mapped = np.clip(y_test_mapped, 0, max_label)
        
        # 设置XGBoost参数搜索空间
        if ball_type == 'red':
            if self.lottery_type == 'ssq':
                # 双色球红球参数搜索空间
                param_dist = {
                    'n_estimators': [100, 200, 300, 500, 800, 1000],
                    'learning_rate': [0.01, 0.03, 0.05, 0.08, 0.1, 0.2],
                    'max_depth': [3, 4, 5, 6, 8, 10],
                    'min_child_weight': [1, 3, 5, 7],
                    'gamma': [0, 0.1, 0.2, 0.3, 0.5],
                    'subsample': [0.6, 0.7, 0.8, 0.9, 1.0],
                    'colsample_bytree': [0.6, 0.7, 0.8, 0.9, 1.0],
                    'reg_alpha': [0, 0.001, 0.01, 0.1, 1, 10],
                    'reg_lambda': [0.01, 0.1, 1, 10, 100],
                    'scale_pos_weight': [1, 3, 5]
                }
            else:  # dlt
                # 大乐透红球参数搜索空间
                param_dist = {
                    'n_estimators': [100, 200, 300, 500, 800, 1000],
                    'learning_rate': [0.01, 0.03, 0.05, 0.08, 0.1, 0.2],
                    'max_depth': [3, 4, 5, 6, 8],
                    'min_child_weight': [1, 3, 5, 7],
                    'gamma': [0, 0.1, 0.2, 0.3, 0.5],
                    'subsample': [0.6, 0.7, 0.8, 0.9, 1.0],
                    'colsample_bytree': [0.6, 0.7, 0.8, 0.9, 1.0],
                    'reg_alpha': [0, 0.001, 0.01, 0.1, 1],
                    'reg_lambda': [0.01, 0.1, 1, 10],
                    'scale_pos_weight': [1, 3, 5]
                }
        else:  # blue
            if self.lottery_type == 'ssq':
                # 双色球蓝球参数搜索空间
                param_dist = {
                    'n_estimators': [100, 200, 300, 500, 800],
                    'learning_rate': [0.01, 0.03, 0.05, 0.08, 0.1, 0.2],
                    'max_depth': [3, 4, 5, 6, 8],
                    'min_child_weight': [1, 2, 3, 5],
                    'gamma': [0, 0.1, 0.2, 0.3],
                    'subsample': [0.6, 0.7, 0.8, 0.9, 1.0],
                    'colsample_bytree': [0.6, 0.7, 0.8, 0.9, 1.0],
                    'reg_alpha': [0, 0.001, 0.01, 0.1],
                    'reg_lambda': [0.01, 0.1, 1, 10],
                    'scale_pos_weight': [1, 2, 3]
                }
            else:  # dlt
                # 大乐透蓝球参数搜索空间
                param_dist = {
                    'n_estimators': [100, 200, 300, 500, 800],
                    'learning_rate': [0.01, 0.03, 0.05, 0.08, 0.1, 0.2],
                    'max_depth': [3, 4, 5, 6],
                    'min_child_weight': [1, 2, 3, 5],
                    'gamma': [0, 0.1, 0.2, 0.3],
                    'subsample': [0.6, 0.7, 0.8, 0.9, 1.0],
                    'colsample_bytree': [0.6, 0.7, 0.8, 0.9, 1.0],
                    'reg_alpha': [0, 0.001, 0.01, 0.1],
                    'reg_lambda': [0.01, 0.1, 1, 10],
                    'scale_pos_weight': [1, 2, 3]
                }
        
        # 根据数据集大小动态调整搜索次数
        adjusted_n_iter = min(n_iter_search, len(X_train) // 10)
        adjusted_n_iter = max(adjusted_n_iter, 10)  # 至少10次
        
        # 创建基础XGBoost分类器
        # 设置num_class参数确保XGBoost知道所有可能的类别数量
        num_classes = max_label_value + 1
        if self.use_gpu:
            base_model = XGBClassifier(tree_method='gpu_hist', gpu_id=0, random_state=random_state, 
                                     eval_metric=['mlogloss', 'merror'], num_class=num_classes)
        else:
            base_model = XGBClassifier(tree_method='hist', random_state=random_state, 
                                     eval_metric=['mlogloss', 'merror'], num_class=num_classes)
        
        # 设置交叉验证策略
        # 检查每个类别的样本数量，确保分层抽样可行
        unique_classes, class_counts = np.unique(y_train_mapped, return_counts=True)
        min_samples_per_class = np.min(class_counts)
        
        # 如果最小类别样本数小于2或只有一个类别，则使用普通K折交叉验证
        if len(unique_classes) > 1 and min_samples_per_class >= 2:
            # 使用分层K折交叉验证
            cv_strategy = StratifiedKFold(n_splits=cv, shuffle=True, random_state=random_state)
        else:
            # 如果只有一个类别或某类别样本数不足，使用普通K折交叉验证
            cv_strategy = KFold(n_splits=cv, shuffle=True, random_state=random_state)
        
        # 设置评分指标
        scoring = {
            'accuracy': 'accuracy',
            'f1_weighted': 'f1_weighted',
            'precision_weighted': 'precision_weighted',
            'recall_weighted': 'recall_weighted',
            'neg_log_loss': 'neg_log_loss'
        }
        
        # 创建随机搜索对象
        random_search = RandomizedSearchCV(
            base_model,
            param_distributions=param_dist,
            n_iter=adjusted_n_iter,
            scoring=scoring,
            refit='neg_log_loss',  # 使用负对数损失作为主要优化指标
            cv=cv_strategy,
            random_state=random_state,
            n_jobs=-1,  # 使用所有可用CPU
            verbose=0,
            return_train_score=True
        )
        
        # 执行随机搜索
        self.log(f"开始{ball_type}球位置{position}的参数搜索，迭代次数: {adjusted_n_iter}")
        search_start_time = time.time()
        random_search.fit(X_train, y_train_mapped)
        search_time = time.time() - search_start_time
        
        # 获取最佳参数和分数
        best_params = random_search.best_params_
        best_score = random_search.best_score_
        cv_results = random_search.cv_results_
        
        self.log(f"{ball_type}球位置{position}参数搜索完成，耗时{search_time:.2f}秒")
        self.log(f"最佳参数: {best_params}")
        self.log(f"最佳交叉验证分数: {best_score:.4f}")
        
        # 使用最佳参数创建模型
        # 在模型初始化时设置eval_metric参数，而不是在fit方法中
        # 设置num_class参数确保XGBoost知道所有可能的类别数量
        num_classes = max_label_value + 1
        if self.use_gpu:
            model = XGBClassifier(tree_method='gpu_hist', gpu_id=0, random_state=random_state, 
                                eval_metric=['mlogloss', 'merror'], num_class=num_classes, **best_params)
        else:
            model = XGBClassifier(tree_method='hist', random_state=random_state, 
                                eval_metric=['mlogloss', 'merror'], num_class=num_classes, **best_params)
        
        # 分割训练集和验证集
        # 检查每个类别的样本数量，确保分层抽样可行
        unique_classes, class_counts = np.unique(y_train_mapped, return_counts=True)
        min_samples_per_class = np.min(class_counts)
        
        # 如果最小类别样本数小于2，则不使用分层抽样
        use_stratify = len(unique_classes) > 1 and min_samples_per_class >= 2
        
        X_train_split, X_val, y_train_split, y_val = train_test_split(
            X_train, y_train_mapped, test_size=0.2, random_state=random_state,
            stratify=y_train_mapped if use_stratify else None
        )
        
        # 训练最终模型，使用验证集进行早停
        train_start_time = time.time()
        # 尝试不同的早停实现方式
        early_stop_success = False
        
        # 方法1: 尝试使用callbacks方式（XGBoost 1.6+）
        try:
            early_stop_callback = callback.EarlyStopping(rounds=early_stopping_rounds, save_best=True)
            model.fit(
                X_train_split, y_train_split,
                eval_set=[(X_val, y_val)],
                callbacks=[early_stop_callback],
                verbose=False
            )
            early_stop_success = True
            self.log(f"使用callbacks方式实现早停")
        except (AttributeError, TypeError) as e:
            self.log(f"callbacks方式不支持: {str(e)}")
        
        # 方法2: 如果callbacks不支持，尝试旧的early_stopping_rounds方式
        if not early_stop_success:
            try:
                model.fit(
                    X_train_split, y_train_split,
                    eval_set=[(X_val, y_val)],
                    early_stopping_rounds=early_stopping_rounds,
                    verbose=False
                )
                early_stop_success = True
                self.log(f"使用early_stopping_rounds参数实现早停")
            except TypeError as e:
                self.log(f"early_stopping_rounds参数不支持: {str(e)}")
        
        # 方法3: 如果都不支持，则不使用早停
        if not early_stop_success:
            self.log(f"警告: 当前XGBoost版本不支持早停功能，将使用完整训练")
            model.fit(
                X_train_split, y_train_split,
                eval_set=[(X_val, y_val)],
                verbose=False
            )
        train_time = time.time() - train_start_time
        
        # 在测试集上评估模型
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)
        
        # 计算评估指标
        accuracy = accuracy_score(y_test_mapped, y_pred)
        precision = precision_score(y_test_mapped, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y_test_mapped, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_test_mapped, y_pred, average='weighted', zero_division=0)
        
        # 计算对数损失
        try:
            log_loss_value = log_loss(y_test_mapped, y_pred_proba)
        except ValueError:
            # 如果预测概率中有0或1，可能会导致对数损失计算出错
            log_loss_value = float('inf')
        
        # 计算混淆矩阵
        cm = confusion_matrix(y_test_mapped, y_pred)
        
        # 获取特征重要性
        feature_importances = model.feature_importances_
        
        # 分析特征重要性
        importance_analysis = self._analyze_feature_importance(feature_importances)
        
        # 检查过拟合
        train_score = model.score(X_train, y_train_mapped)
        test_score = model.score(X_test, y_test_mapped)
        overfit_ratio = train_score / max(test_score, 1e-10)  # 避免除以0
        
        # 如果过拟合严重，尝试调整正则化参数
        if overfit_ratio > 1.3:  # 训练集得分比测试集高30%以上
            self.log(f"检测到{ball_type}球位置{position}模型可能存在过拟合，训练集/测试集得分比: {overfit_ratio:.2f}")
            self.log("尝试增加正则化强度...")
            
            # 增加正则化参数
            adjusted_params = best_params.copy()
            adjusted_params['reg_alpha'] = best_params.get('reg_alpha', 0) * 2 + 0.1
            adjusted_params['reg_lambda'] = best_params.get('reg_lambda', 1) * 2
            adjusted_params['learning_rate'] = best_params.get('learning_rate', 0.1) * 0.8
            
            # 创建调整后的模型
            if self.use_gpu:
                adjusted_model = XGBClassifier(tree_method='gpu_hist', gpu_id=0, random_state=random_state, eval_metric=['mlogloss', 'merror'], **adjusted_params)
            else:
                adjusted_model = XGBClassifier(tree_method='hist', random_state=random_state, eval_metric=['mlogloss', 'merror'], **adjusted_params)
            
            # 训练调整后的模型，使用兼容的早停方式
            try:
                # 方法1: 尝试使用新版本的callbacks方式
                early_stop_callback = callback.EarlyStopping(rounds=early_stopping_rounds, save_best=True)
                adjusted_model.fit(
                    X_train_split, y_train_split,
                    eval_set=[(X_val, y_val)],
                    callbacks=[early_stop_callback],
                    verbose=False
                )
                self.log(f"使用callbacks方式实现早停")
            except Exception as e:
                try:
                    # 方法2: 如果callbacks不支持，尝试旧的early_stopping_rounds方式
                    adjusted_model.fit(
                        X_train_split, y_train_split,
                        eval_set=[(X_val, y_val)],
                        early_stopping_rounds=early_stopping_rounds,
                        verbose=False
                    )
                    self.log(f"使用early_stopping_rounds参数实现早停")
                except Exception as e2:
                    # 方法3: 如果都不支持，则不使用早停
                    self.log(f"callbacks方式不支持: {str(e)}")
                    self.log(f"early_stopping_rounds参数不支持: {str(e2)}")
                    self.log("警告: 当前XGBoost版本不支持早停功能，将使用完整训练")
                    adjusted_model.fit(
                        X_train_split, y_train_split,
                        verbose=False
                    )
            
            # 评估调整后的模型
            adjusted_train_score = adjusted_model.score(X_train, y_train_mapped)
            adjusted_test_score = adjusted_model.score(X_test, y_test_mapped)
            adjusted_overfit_ratio = adjusted_train_score / max(adjusted_test_score, 1e-10)
            
            self.log(f"调整后的训练集/测试集得分比: {adjusted_overfit_ratio:.2f}")
            
            # 如果调整后的模型更好，使用调整后的模型
            if adjusted_test_score >= test_score * 0.95 and adjusted_overfit_ratio < overfit_ratio:
                self.log("使用调整后的模型，过拟合程度降低")
                model = adjusted_model
                train_score = adjusted_train_score
                test_score = adjusted_test_score
                overfit_ratio = adjusted_overfit_ratio
            else:
                self.log("保留原始模型，调整后的模型性能下降")
        
        # 创建包装模型
        wrapped_model = WrappedXGBoostModel(model, reverse_mapping)
        
        # 收集训练结果
        train_result = {
            'ball_type': ball_type,
            'position': position,
            'best_params': best_params,
            'best_cv_score': best_score,
            'cv_results': {
                'mean_test_accuracy': np.mean(cv_results['mean_test_accuracy']),
                'mean_test_f1_weighted': np.mean(cv_results['mean_test_f1_weighted']),
                'mean_test_precision_weighted': np.mean(cv_results['mean_test_precision_weighted']),
                'mean_test_recall_weighted': np.mean(cv_results['mean_test_recall_weighted']),
                'mean_test_neg_log_loss': np.mean(cv_results['mean_test_neg_log_loss'])
            },
            'test_metrics': {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'log_loss': log_loss_value
            },
            'confusion_matrix': cm.tolist(),
            'feature_importances': feature_importances.tolist(),
            'importance_analysis': importance_analysis,
            'train_score': train_score,
            'test_score': test_score,
            'overfit_ratio': overfit_ratio,
            'n_features': X_train.shape[1],
            'n_samples': X_train.shape[0],
            'class_distribution': Counter(y_train_mapped),
            'search_time': search_time,
            'train_time': train_time,
            'model_complexity': {
                'n_estimators': model.n_estimators,
                'max_depth': model.max_depth,
                'n_classes': model.n_classes_
            },
            'label_mapping': label_mapping,
            'reverse_mapping': reverse_mapping
        }
        
        return wrapped_model, train_result
    
    def _analyze_feature_importance(self, importances):
        """
        分析特征重要性
        
        Args:
            importances: 特征重要性数组
            
        Returns:
            特征重要性分析结果字典
        """
        # 计算前N个特征的重要性占比
        sorted_idx = np.argsort(importances)[::-1]
        sorted_importances = importances[sorted_idx]
        total_importance = np.sum(importances)
        
        # 计算前10个特征的累计重要性
        top_n = min(10, len(importances))
        top_n_importance = np.sum(sorted_importances[:top_n]) / total_importance
        
        # 计算零重要性特征的比例
        zero_importance = np.sum(importances == 0) / len(importances)
        
        # 计算特征重要性的统计信息
        stats_dict = {
            'max': np.max(importances),
            'min': np.min(importances),
            'mean': np.mean(importances),
            'median': np.median(importances),
            'variance': np.var(importances),
            'std': np.std(importances),
            'skewness': stats.skew(importances) if len(importances) > 2 else 0,
            'kurtosis': stats.kurtosis(importances) if len(importances) > 2 else 0,
            'entropy': stats.entropy(importances + 1e-10) if len(importances) > 1 else 0,
            'quantile_25': np.percentile(importances, 25),
            'quantile_75': np.percentile(importances, 75),
            'quantile_90': np.percentile(importances, 90)
        }
        
        # 检查是否符合80/20法则（帕累托原则）
        # 即20%的特征是否贡献了80%的重要性
        n_features = len(importances)
        n_top_20_percent = max(1, int(n_features * 0.2))
        top_20_percent_importance = np.sum(sorted_importances[:n_top_20_percent]) / total_importance
        pareto_principle_satisfied = top_20_percent_importance >= 0.8
        
        return {
            'top_n_importance_ratio': top_n_importance,
            'top_n': top_n,
            'zero_importance_ratio': zero_importance,
            'stats': stats_dict,
            'pareto_principle': {
                'satisfied': pareto_principle_satisfied,
                'top_20_percent_importance': top_20_percent_importance
            }
        }
    
    def evaluate(self, X_test, red_test_data, blue_test_data):
        """
        评估模型性能
        
        Args:
            X_test: 测试特征
            red_test_data: 红球测试标签
            blue_test_data: 蓝球测试标签
            
        Returns:
            评估结果字典
        """
        self.log("评估红球模型性能...")
        red_metrics = []
        
        for i, (model, y_test) in enumerate(zip(self.models['red'], red_test_data)):
            position = i + 1
            self.log(f"评估红球位置{position}模型...")
            
            # 预测
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)
            
            # 应用标签映射
            if hasattr(model.model, 'classes_'):
                classes = model.model.classes_
                y_test_mapped = np.array([np.where(classes == y)[0][0] if y in classes else -1 for y in y_test])
                # 处理不在classes中的标签
                y_test_mapped = np.array([y if y >= 0 else 0 for y in y_test_mapped])
            else:
                y_test_mapped = y_test
            
            # 计算评估指标
            accuracy = accuracy_score(y_test_mapped, y_pred)
            precision = precision_score(y_test_mapped, y_pred, average='weighted', zero_division=0)
            recall = recall_score(y_test_mapped, y_pred, average='weighted', zero_division=0)
            f1 = f1_score(y_test_mapped, y_pred, average='weighted', zero_division=0)
            
            # 计算对数损失
            try:
                log_loss_value = log_loss(y_test_mapped, y_pred_proba)
            except ValueError:
                log_loss_value = float('inf')
            
            # 计算混淆矩阵
            cm = confusion_matrix(y_test_mapped, y_pred)
            
            # 分析混淆矩阵
            cm_analysis = self._analyze_confusion_matrix(cm, y_test_mapped, y_pred)
            
            # 收集指标
            metrics = {
                'position': position,
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'log_loss': log_loss_value,
                'confusion_matrix': cm.tolist(),
                'confusion_analysis': cm_analysis
            }
            
            red_metrics.append(metrics)
            
            self.log(f"红球位置{position}评估结果: 准确率={accuracy:.4f}, F1={f1:.4f}")
        
        # 计算红球平均指标
        red_avg_metrics = {
            'accuracy': np.mean([m['accuracy'] for m in red_metrics]),
            'precision': np.mean([m['precision'] for m in red_metrics]),
            'recall': np.mean([m['recall'] for m in red_metrics]),
            'f1': np.mean([m['f1'] for m in red_metrics]),
            'log_loss': np.mean([m['log_loss'] for m in red_metrics if m['log_loss'] != float('inf')])
        }
        
        # 计算红球位置间的差异
        red_position_variance = {
            'accuracy': np.var([m['accuracy'] for m in red_metrics]),
            'f1': np.var([m['f1'] for m in red_metrics])
        }
        
        self.log(f"红球平均评估结果: 准确率={red_avg_metrics['accuracy']:.4f}, F1={red_avg_metrics['f1']:.4f}")
        
        # 评估蓝球模型
        self.log("评估蓝球模型性能...")
        blue_metrics = []
        
        for i, (model, y_test) in enumerate(zip(self.models['blue'], blue_test_data)):
            position = i + 1
            self.log(f"评估蓝球位置{position}模型...")
            
            # 预测
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)
            
            # 应用标签映射
            if hasattr(model.model, 'classes_'):
                classes = model.model.classes_
                y_test_mapped = np.array([np.where(classes == y)[0][0] if y in classes else -1 for y in y_test])
                # 处理不在classes中的标签
                y_test_mapped = np.array([y if y >= 0 else 0 for y in y_test_mapped])
            else:
                y_test_mapped = y_test
            
            # 计算评估指标
            accuracy = accuracy_score(y_test_mapped, y_pred)
            precision = precision_score(y_test_mapped, y_pred, average='weighted', zero_division=0)
            recall = recall_score(y_test_mapped, y_pred, average='weighted', zero_division=0)
            f1 = f1_score(y_test_mapped, y_pred, average='weighted', zero_division=0)
            
            # 计算对数损失
            try:
                log_loss_value = log_loss(y_test_mapped, y_pred_proba)
            except ValueError:
                log_loss_value = float('inf')
            
            # 计算混淆矩阵
            cm = confusion_matrix(y_test_mapped, y_pred)
            
            # 分析混淆矩阵
            cm_analysis = self._analyze_confusion_matrix(cm, y_test_mapped, y_pred)
            
            # 收集指标
            metrics = {
                'position': position,
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'log_loss': log_loss_value,
                'confusion_matrix': cm.tolist(),
                'confusion_analysis': cm_analysis
            }
            
            blue_metrics.append(metrics)
            
            self.log(f"蓝球位置{position}评估结果: 准确率={accuracy:.4f}, F1={f1:.4f}")
        
        # 计算蓝球平均指标
        blue_avg_metrics = {
            'accuracy': np.mean([m['accuracy'] for m in blue_metrics]),
            'precision': np.mean([m['precision'] for m in blue_metrics]),
            'recall': np.mean([m['recall'] for m in blue_metrics]),
            'f1': np.mean([m['f1'] for m in blue_metrics]),
            'log_loss': np.mean([m['log_loss'] for m in blue_metrics if m['log_loss'] != float('inf')])
        }
        
        # 计算蓝球位置间的差异
        blue_position_variance = {
            'accuracy': np.var([m['accuracy'] for m in blue_metrics]),
            'f1': np.var([m['f1'] for m in blue_metrics])
        }
        
        self.log(f"蓝球平均评估结果: 准确率={blue_avg_metrics['accuracy']:.4f}, F1={blue_avg_metrics['f1']:.4f}")
        
        # 返回评估结果
        return {
            'red_metrics': red_metrics,
            'red_avg_metrics': red_avg_metrics,
            'red_position_variance': red_position_variance,
            'blue_metrics': blue_metrics,
            'blue_avg_metrics': blue_avg_metrics,
            'blue_position_variance': blue_position_variance
        }
    
    def _analyze_confusion_matrix(self, cm, y_true, y_pred):
        """
        分析混淆矩阵
        
        Args:
            cm: 混淆矩阵
            y_true: 真实标签
            y_pred: 预测标签
            
        Returns:
            混淆矩阵分析结果字典
        """
        # 找出最常混淆的类别对
        n_classes = cm.shape[0]
        confusion_pairs = []
        
        for i in range(n_classes):
            for j in range(n_classes):
                if i != j and cm[i, j] > 0:
                    confusion_pairs.append(((i, j), cm[i, j]))
        
        # 按混淆次数排序
        confusion_pairs.sort(key=lambda x: x[1], reverse=True)
        
        # 取前5个最常混淆的类别对
        top_confusion_pairs = confusion_pairs[:5]
        
        # 计算每个类别的预测准确率
        class_accuracy = {}
        for i in range(n_classes):
            if np.sum(cm[i, :]) > 0:
                class_accuracy[i] = cm[i, i] / np.sum(cm[i, :])
        
        # 找出预测最准确和最不准确的类别
        if class_accuracy:
            most_accurate_class = max(class_accuracy.items(), key=lambda x: x[1])
            least_accurate_class = min(class_accuracy.items(), key=lambda x: x[1])
        else:
            most_accurate_class = (0, 0)
            least_accurate_class = (0, 0)
        
        return {
            'top_confusion_pairs': top_confusion_pairs,
            'most_accurate_class': most_accurate_class,
            'least_accurate_class': least_accurate_class,
            'class_accuracy': class_accuracy
        }
    
    def save_models(self, red_train_results, blue_train_results, evaluation_results):
        """
        保存训练好的模型和缩放器
        支持位置模型格式
        
        Args:
            red_train_results: 红球训练结果
            blue_train_results: 蓝球训练结果
            evaluation_results: 评估结果
            
        Returns:
            保存结果字典
        """
        # 创建版本目录
        timestamp = int(time.time())
        version_dir = os.path.join(self.models_dir, f"v_{timestamp}")
        os.makedirs(version_dir, exist_ok=True)
        
        # 创建符号链接指向最新版本
        latest_link = os.path.join(self.models_dir, "latest")
        if os.path.exists(latest_link):
            if os.path.islink(latest_link):
                os.unlink(latest_link)
            else:
                os.remove(latest_link)
        
        # 在Windows上创建文本文件而不是符号链接
        if platform.system() == "Windows":
            with open(latest_link, "w") as f:
                f.write(version_dir)
        else:
            os.symlink(version_dir, latest_link, target_is_directory=True)
        
        # 收集模型统计信息
        model_statistics = {
            'red_count': len(red_train_results),
            'blue_count': len(blue_train_results),
            'red_avg_score': evaluation_results['red_avg_metrics']['accuracy'],
            'blue_avg_score': evaluation_results['blue_avg_metrics']['accuracy'],
            'red_avg_f1': evaluation_results['red_avg_metrics']['f1'],
            'blue_avg_f1': evaluation_results['blue_avg_metrics']['f1'],
            'red_best_params': [result['best_params'] for result in red_train_results],
            'blue_best_params': [result['best_params'] for result in blue_train_results],
            'red_feature_importance': [result['importance_analysis'] for result in red_train_results],
            'blue_feature_importance': [result['importance_analysis'] for result in blue_train_results]
        }
        
        # 收集环境信息
        env_info = {
            'python_version': platform.python_version(),
            'xgboost_version': getattr(XGBClassifier, '__version__', 'unknown'),
            'sklearn_version': getattr(train_test_split, '__version__', 'unknown'),
            'numpy_version': np.__version__,
            'pandas_version': pd.__version__,
            'platform': platform.platform(),
            'processor': platform.processor(),
            'gpu_used': self.use_gpu
        }
        
        # 创建模型信息
        model_info = {
            'model_type': self.model_type,
            'lottery_type': self.lottery_type,
            'feature_window': self.feature_window,
            'created_time': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'model_version': f"v_{timestamp}",
            'model_id': str(uuid.uuid4()),
            'model_statistics': model_statistics,
            'environment_info': env_info,
            'optimization_features': {
                'parameter_tuning': True,
                'early_stopping': True,
                'feature_importance_analysis': True,
                'overfit_detection': True,
                'position_based_models': True
            }
        }
        
        # 保存模型信息
        with open(os.path.join(version_dir, 'model_info.json'), 'w') as f:
            json.dump(model_info, f, indent=2)
        
        # 保存红球位置模型
        for i, model in enumerate(self.models['red']):
            model_path = os.path.join(version_dir, f'red_model_pos_{i+1}.pkl')
            with open(model_path, 'wb') as f:
                pickle.dump(model, f)
        
        # 保存蓝球位置模型
        for i, model in enumerate(self.models['blue']):
            model_path = os.path.join(version_dir, f'blue_model_pos_{i+1}.pkl')
            with open(model_path, 'wb') as f:
                pickle.dump(model, f)
        
        # 保存特征缩放器
        for scaler_name, scaler in self.scalers.items():
            scaler_path = os.path.join(version_dir, f'scaler_{scaler_name}.pkl')
            with open(scaler_path, 'wb') as f:
                pickle.dump(scaler, f)
        
        # 保存特征名称列表
        if hasattr(self, 'feature_cols') and self.feature_cols:
            with open(os.path.join(version_dir, 'feature_names.json'), 'w') as f:
                json.dump(self.feature_cols, f)
        
        self.log(f"模型已保存到: {version_dir}")
        self.current_model_version = os.path.basename(version_dir)
        
        return {
            'version_dir': version_dir,
            'model_info': model_info
        }
    
    def load_models(self, version=None):
        """
        加载训练好的模型
        支持加载指定版本或最新版本
        
        Args:
            version: 模型版本，如果为None则加载最新版本
            
        Returns:
            是否加载成功
        """
        self.log(f"加载{self.lottery_type}的XGBoost模型...")
        
        # 确定模型目录
        if version is None:
            # 尝试加载最新版本
            latest_link = os.path.join(self.models_dir, "latest")
            if os.path.exists(latest_link):
                if os.path.islink(latest_link):
                    model_dir = os.readlink(latest_link)
                else:
                    # 在Windows上读取文本文件
                    with open(latest_link, "r") as f:
                        model_dir = f.read().strip()
            else:
                # 如果没有latest链接，查找最新的版本目录
                model_dir = self._find_latest_version_dir(self.models_dir)
        else:
            # 加载指定版本
            model_dir = os.path.join(self.models_dir, version)
        
        if model_dir is None or not os.path.exists(model_dir):
            self.log(f"找不到模型目录: {model_dir}")
            return False
        
        self.log(f"从{model_dir}加载模型...")
        
        # 加载模型信息
        info_path = os.path.join(model_dir, 'model_info.json')
        if os.path.exists(info_path):
            try:
                with open(info_path, 'r') as f:
                    model_info = json.load(f)
                self.log(f"加载模型信息成功: {model_info['model_version']}")
            except Exception as e:
                self.log(f"加载模型信息失败: {e}")
                model_info = None
        else:
            self.log("模型信息文件不存在")
            model_info = None
        
        # 加载特征名称列表
        feature_names_path = os.path.join(model_dir, 'feature_names.json')
        if os.path.exists(feature_names_path):
            try:
                with open(feature_names_path, 'r') as f:
                    self.feature_cols = json.load(f)
                self.log(f"加载特征名称列表成功，共{len(self.feature_cols)}个特征")
            except Exception as e:
                self.log(f"加载特征名称列表失败: {e}")
        
        # 加载特征缩放器
        for scaler_name in ['X', 'red', 'blue']:
            scaler_path = os.path.join(model_dir, f'scaler_{scaler_name}.pkl')
            if os.path.exists(scaler_path):
                try:
                    with open(scaler_path, 'rb') as f:
                        self.scalers[scaler_name] = pickle.load(f)
                    self.log(f"加载{scaler_name}特征缩放器成功")
                except Exception as e:
                    self.log(f"加载{scaler_name}特征缩放器失败: {e}")
        
        # 加载模型
        models_loaded = True
        balls_loaded = 0
        
        for ball_type in ['red', 'blue']:
            ball_models = []
            ball_count = self.red_count if ball_type == 'red' else self.blue_count
            
            # 尝试加载位置模型（新格式）
            position_models_found = True
            for i in range(ball_count):
                model_path = os.path.join(model_dir, f'{ball_type}_model_pos_{i+1}.pkl')
                if os.path.exists(model_path):
                    try:
                        with open(model_path, 'rb') as f:
                            model = pickle.load(f)
                        ball_models.append(model)
                        self.log(f"加载{ball_type}球位置{i+1}模型成功")
                    except Exception as e:
                        self.log(f"加载{ball_type}球位置{i+1}模型失败: {e}")
                        position_models_found = False
                        break
                else:
                    position_models_found = False
                    break
            
            # 如果位置模型加载成功
            if position_models_found and len(ball_models) == ball_count:
                self.models[ball_type] = ball_models
                balls_loaded += 1
                self.log(f"成功加载{ball_count}个{ball_type}球位置模型")
            else:
                # 尝试加载旧格式的单一模型
                model_path = os.path.join(model_dir, f'{ball_type}_model.pkl')
                if os.path.exists(model_path):
                    try:
                        with open(model_path, 'rb') as f:
                            self.models[ball_type] = pickle.load(f)
                        self.log(f"加载{ball_type}球模型成功（旧格式）")
                        balls_loaded += 1
                    except Exception as e:
                        self.log(f"加载{ball_type}球模型失败: {e}")
                        models_loaded = False
                else:
                    self.log(f"警告: {ball_type}球模型文件不存在")
                    models_loaded = False
        
        # 判断加载结果
        if balls_loaded >= 2:  # 至少加载了红球和蓝球
            self.log(f"XGBoost模型加载成功，加载了{balls_loaded}种球类型的模型")
            # 记录当前加载的模型版本
            self.current_model_version = os.path.basename(model_dir)
            return True
        else:
            self.log(f"XGBoost模型加载失败，未找到必要的红球和蓝球模型")
            return False
    
    def _find_latest_version_dir(self, base_dir):
        """
        查找最新的模型版本目录
        
        Args:
            base_dir: 基础模型目录
            
        Returns:
            最新版本的目录路径，如果没有找到则返回None
        """
        try:
            # 查找所有以v_开头的目录
            version_dirs = [d for d in os.listdir(base_dir) 
                           if os.path.isdir(os.path.join(base_dir, d)) and d.startswith('v_')]
            
            if not version_dirs:
                self.log(f"在{base_dir}中未找到任何版本目录")
                # 尝试直接使用基础目录（兼容旧版本）
                if os.path.exists(os.path.join(base_dir, 'model_info.json')):
                    self.log(f"找到旧版本模型结构，将使用基础目录")
                    return base_dir
                return None
            
            # 按照版本号排序（实际上是按照时间戳排序）
            version_dirs.sort(reverse=True)  # 降序排列，最新的在前面
            latest_dir = os.path.join(base_dir, version_dirs[0])
            self.log(f"找到最新的模型版本目录: {latest_dir}")
            return latest_dir
        except Exception as e:
            self.log(f"查找最新版本目录时出错: {e}")
            return None
    
    def list_available_versions(self):
        """
        列出所有可用的模型版本
        
        Returns:
            版本信息列表，每个元素是一个字典，包含版本ID和创建时间
        """
        base_model_dir = os.path.join(self.models_dir, self.model_type)
        if not os.path.exists(base_model_dir):
            self.log(f"模型基础目录不存在: {base_model_dir}")
            return []
        
        versions = []
        try:
            # 查找所有以v_开头的目录
            version_dirs = [d for d in os.listdir(base_model_dir) 
                           if os.path.isdir(os.path.join(base_model_dir, d)) and d.startswith('v_')]
            
            for v_dir in version_dirs:
                info_path = os.path.join(base_model_dir, v_dir, 'model_info.json')
                if os.path.exists(info_path):
                    try:
                        with open(info_path, 'r') as f:
                            model_info = json.load(f)
                        
                        version_info = {
                            'version_id': v_dir,
                            'created_time': model_info.get('created_time', '未知'),
                            'model_version': model_info.get('model_version', '未知'),
                            'lottery_type': model_info.get('lottery_type', '未知')
                        }
                        
                        # 添加性能指标（如果有）
                        if 'model_statistics' in model_info:
                            stats = model_info['model_statistics']
                            if 'red_avg_score' in stats:
                                version_info['red_score'] = stats['red_avg_score']
                            if 'blue_avg_score' in stats:
                                version_info['blue_score'] = stats['blue_avg_score']
                        
                        versions.append(version_info)
                    except Exception as e:
                        self.log(f"读取版本{v_dir}的信息时出错: {e}")
            
            # 按创建时间排序（降序）
            versions.sort(key=lambda x: x.get('created_time', ''), reverse=True)
            
        except Exception as e:
            self.log(f"列出可用版本时出错: {e}")
        
        return versions
    
    def prepare_features(self, recent_data):
        """
        从最近的开奖数据中提取预测所需的特征
        
        Args:
            recent_data: 包含最近开奖数据的DataFrame
            
        Returns:
            处理后的特征数据，用于预测
        """
        self.log("从最近数据中提取预测特征...")
        
        # 设置特征窗口大小
        window_size = self.feature_window
        
        # 确保数据按期数排序
        if '期数' in recent_data.columns:
            recent_data = recent_data.sort_values('期数').reset_index(drop=True)
        else:
            recent_data = recent_data.reset_index(drop=True)
        
        # 提取红蓝球列名
        if self.lottery_type == 'dlt':
            red_cols = [col for col in recent_data.columns if col.startswith('红球_')][:5]
            blue_cols = [col for col in recent_data.columns if col.startswith('蓝球_')][:2]
        else:  # ssq
            red_cols = [col for col in recent_data.columns if col.startswith('红球_')][:6]
            blue_cols = [col for col in recent_data.columns if col.startswith('蓝球_')][:1]
        
        # 检查数据量是否足够
        if len(recent_data) < window_size:
            self.log(f"错误: 提供的数据量({len(recent_data)}行)小于特征窗口大小({window_size})")
            return None
        
        # 创建特征
        features = []
        # 使用最近的window_size期数据作为特征
        for j in range(window_size):
            row_features = []
            idx = len(recent_data) - window_size + j
            for col in red_cols + blue_cols:
                row_features.append(recent_data.iloc[idx][col])
            features.append(row_features)
        
        # 转换为NumPy数组并重塑
        X = np.array([features])
        X_reshaped = X.reshape(X.shape[0], -1)
        
        # 检查特征缩放器是否存在
        if 'X' not in self.scalers:
            self.log("警告: 特征缩放器不存在，尝试加载模型")
            load_success = self.load_models()
            if not load_success or 'X' not in self.scalers:
                self.log("错误: 无法加载特征缩放器，无法进行预测")
                return None
        
        # 应用特征缩放
        X_scaled = self.scalers['X'].transform(X_reshaped)
        
        return X_scaled
        
    def predict(self, recent_data, num_predictions=1, position_based=True, top_k=10, use_probability=True):
        """
        生成预测结果
        优化版本：支持位置模型、概率预测和智能号码选择
        
        Args:
            recent_data: 包含最近开奖数据的DataFrame
            num_predictions: 预测的组数
            position_based: 是否使用位置模型，默认为True
            top_k: 每个位置考虑的候选号码数量
            use_probability: 是否使用概率进行智能选择
            
        Returns:
            预测的红球和蓝球号码，单组或多组
        """
        # 首先检查模型是否已加载
        if 'red' not in self.models or 'blue' not in self.models:
            # 尝试重新加载模型
            self.log("模型未加载，尝试重新加载...")
            load_success = self.load_models()
            if not load_success:
                self.log("错误：模型加载失败，请先训练模型")
                raise ValueError("模型未正确加载，请先训练或加载模型")
        
        # 准备特征数据
        self.log("准备预测特征...")
        X_features = self.prepare_features(recent_data)
        
        # 检查特征是否准备成功
        if X_features is None or len(X_features) == 0:
            self.log("错误：特征准备失败")
            raise ValueError("特征准备失败，无法进行预测")
        
        # 根据模型类型选择预测方法
        if position_based and isinstance(self.models['red'], list) and isinstance(self.models['blue'], list):
            self.log("使用位置模型进行预测...")
            return self._predict_with_position_models(X_features, num_predictions, top_k, use_probability)
        else:
            self.log("使用传统模型进行预测...")
            return self._predict_with_traditional_models(X_features, num_predictions, top_k, use_probability)
    
    def _predict_with_position_models(self, X_features, num_predictions=1, top_k=10, use_probability=True):
        """
        使用位置模型进行预测
        
        Args:
            X_features: 特征数据
            num_predictions: 预测的组数
            top_k: 每个位置考虑的候选号码数量
            use_probability: 是否使用概率进行智能选择
            
        Returns:
            预测的红球和蓝球号码，单组或多组
        """
        # 获取红球和蓝球模型
        red_models = self.models['red']
        blue_models = self.models['blue']
        
        # 检查模型数量是否符合要求
        if len(red_models) != self.red_count or len(blue_models) != self.blue_count:
            self.log(f"警告：模型数量与要求不符，红球模型数量: {len(red_models)}，蓝球模型数量: {len(blue_models)}")
        
        # 存储每个位置的预测结果和概率
        red_predictions = []
        red_probabilities = []
        blue_predictions = []
        blue_probabilities = []
        
        # 预测红球
        for i, model in enumerate(red_models):
            position = i + 1
            self.log(f"预测红球位置{position}...")
            
            # 获取top-k预测结果和概率
            indices, probs = model.predict_top_k(X_features, k=top_k)
            red_predictions.append(indices[0])  # 取第一个样本的预测结果
            red_probabilities.append(probs[0])  # 取第一个样本的概率
        
        # 预测蓝球
        for i, model in enumerate(blue_models):
            position = i + 1
            self.log(f"预测蓝球位置{position}...")
            
            # 获取top-k预测结果和概率
            indices, probs = model.predict_top_k(X_features, k=top_k)
            blue_predictions.append(indices[0])  # 取第一个样本的预测结果
            blue_probabilities.append(probs[0])  # 取第一个样本的概率
        
        # 生成多组预测结果
        all_predictions = []
        for _ in range(num_predictions):
            # 智能选择红球号码
            red_balls = self._select_balls(
                red_predictions, red_probabilities, 
                count=self.red_count, max_value=self.red_range-1, 
                use_probability=use_probability
            )
            
            # 智能选择蓝球号码
            blue_balls = self._select_balls(
                blue_predictions, blue_probabilities, 
                count=self.blue_count, max_value=self.blue_range-1, 
                use_probability=use_probability
            )
            
            # 添加到预测结果列表
            all_predictions.append((red_balls, blue_balls))
        
        # 如果只需要一组预测，直接返回第一组
        if num_predictions == 1:
            return all_predictions[0]
        
        return all_predictions
    
    def _predict_with_traditional_models(self, X_features, num_predictions=1, top_k=10, use_probability=True):
        """
        使用传统模型进行预测（非位置模型）
        
        Args:
            X_features: 特征数据
            num_predictions: 预测的组数
            top_k: 考虑的候选号码数量
            use_probability: 是否使用概率进行智能选择
            
        Returns:
            预测的红球和蓝球号码，单组或多组
        """
        # 获取红球和蓝球模型
        red_model = self.models['red']
        blue_model = self.models['blue']
        
        # 预测红球
        self.log("预测红球...")
        if hasattr(red_model, 'predict_proba'):
            red_proba = red_model.predict_proba(X_features)
            # 获取概率最高的top_k个红球号码
            red_indices = np.argsort(red_proba[0])[-top_k:]
            red_probs = red_proba[0][red_indices]
            # 按概率降序排列
            sort_idx = np.argsort(red_probs)[::-1]
            red_indices = red_indices[sort_idx]
            red_probs = red_probs[sort_idx]
        else:
            # 如果模型不支持概率预测，使用简单预测
            red_indices = np.array([red_model.predict(X_features)[0]])
            red_probs = np.array([1.0])
        
        # 预测蓝球
        self.log("预测蓝球...")
        if hasattr(blue_model, 'predict_proba'):
            blue_proba = blue_model.predict_proba(X_features)
            # 获取概率最高的top_k个蓝球号码
            blue_indices = np.argsort(blue_proba[0])[-top_k:]
            blue_probs = blue_proba[0][blue_indices]
            # 按概率降序排列
            sort_idx = np.argsort(blue_probs)[::-1]
            blue_indices = blue_indices[sort_idx]
            blue_probs = blue_probs[sort_idx]
        else:
            # 如果模型不支持概率预测，使用简单预测
            blue_indices = np.array([blue_model.predict(X_features)[0]])
            blue_probs = np.array([1.0])
        
        # 生成多组预测结果
        all_predictions = []
        for _ in range(num_predictions):
            # 从候选红球中选择self.red_count个
            if use_probability and len(red_indices) > self.red_count:
                # 根据概率加权选择
                red_balls = np.random.choice(
                    red_indices, size=self.red_count, replace=False, 
                    p=red_probs/np.sum(red_probs)
                )
                # 排序
                red_balls = np.sort(red_balls)
            else:
                # 直接选择概率最高的几个
                red_balls = red_indices[:self.red_count]
                red_balls = np.sort(red_balls)
            
            # 从候选蓝球中选择self.blue_count个
            if use_probability and len(blue_indices) > self.blue_count:
                # 根据概率加权选择
                blue_balls = np.random.choice(
                    blue_indices, size=self.blue_count, replace=False, 
                    p=blue_probs/np.sum(blue_probs)
                )
                # 排序
                blue_balls = np.sort(blue_balls)
            else:
                # 直接选择概率最高的几个
                blue_balls = blue_indices[:self.blue_count]
                blue_balls = np.sort(blue_balls)
            
            # 添加到预测结果列表
            all_predictions.append((red_balls, blue_balls))
        
        # 如果只需要一组预测，直接返回第一组
        if num_predictions == 1:
            return all_predictions[0]
        
        return all_predictions
    
    def _select_balls(self, predictions, probabilities, count, max_value, use_probability=True):
        """
        从候选号码中智能选择球号
        
        Args:
            predictions: 每个位置的候选号码列表
            probabilities: 每个位置的候选号码概率列表
            count: 需要选择的球数
            max_value: 球号的最大值
            use_probability: 是否使用概率进行选择
            
        Returns:
            选择的球号列表，已排序
        """
        # 创建候选号码池
        candidates = set()
        candidate_probs = {}
        
        # 将所有位置的候选号码添加到池中
        for pos_idx, (pos_preds, pos_probs) in enumerate(zip(predictions, probabilities)):
            for ball_idx, (ball, prob) in enumerate(zip(pos_preds, pos_probs)):
                # 确保球号在有效范围内
                if 0 <= ball <= max_value:
                    candidates.add(ball)
                    # 如果号码已存在，取最高概率
                    if ball in candidate_probs:
                        candidate_probs[ball] = max(candidate_probs[ball], prob)
                    else:
                        candidate_probs[ball] = prob
        
        # 转换为列表
        candidates = list(candidates)
        
        # 如果候选号码不足，添加随机号码
        if len(candidates) < count:
            self.log(f"警告：候选号码不足，添加随机号码。当前候选数量: {len(candidates)}，需要: {count}")
            # 找出缺失的号码
            missing = set(range(max_value + 1)) - set(candidates)
            # 随机选择缺失的号码
            additional = np.random.choice(list(missing), size=min(count - len(candidates), len(missing)), replace=False)
            candidates.extend(additional)
            # 为新添加的号码分配较低的概率
            min_prob = min(candidate_probs.values()) if candidate_probs else 0.1
            for ball in additional:
                candidate_probs[ball] = min_prob * 0.5
        
        # 选择球号
        if use_probability and len(candidates) > count:
            # 提取概率
            probs = np.array([candidate_probs[ball] for ball in candidates])
            # 归一化概率
            probs = probs / np.sum(probs)
            # 根据概率加权选择
            selected = np.random.choice(candidates, size=count, replace=False, p=probs)
        else:
            # 按概率排序
            candidates_with_probs = [(ball, candidate_probs[ball]) for ball in candidates]
            candidates_with_probs.sort(key=lambda x: x[1], reverse=True)
            # 选择概率最高的几个
            selected = np.array([ball for ball, _ in candidates_with_probs[:count]])
        
        # 排序并返回
        return np.sort(selected)