# -*- coding:utf-8 -*-
"""
XGBoost model implementation for lottery prediction
优化版本：提升性能、速度和预测准确性
"""

import os
import time
import numpy as np
import pandas as pd
import pickle
import joblib
import json
import xgboost as xgb
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, KFold
from sklearn.metrics import accuracy_score, classification_report
from collections import Counter
import warnings
from typing import List, Tuple, Dict, Any

from .base import BaseMLModel

# 忽略XGBoost的警告信息
warnings.filterwarnings('ignore', category=UserWarning, module='xgboost')


class WrappedXGBoostModel:
    """
    XGBoost模型包装器，用于统一预测接口
    优化版本：支持概率预测和更好的预测策略
    """
    def __init__(self, model, processor=None):
        self.model = model
        self.processor = processor
        
    def predict(self, X):
        """预测类别"""
        if self.processor:
            X = self.processor(X)
        predictions = self.model.predict(X)
        
        # 如果模型有标签映射，需要将预测结果映射回原始标签
        if hasattr(self.model, 'reverse_mapping'):
            mapped_predictions = np.array([self.model.reverse_mapping.get(pred, pred) for pred in predictions])
            return mapped_predictions
        
        return predictions
    
    def predict_proba(self, X):
        """预测概率分布"""
        if self.processor:
            X = self.processor(X)
        return self.model.predict_proba(X)
    
    def predict_top_k(self, X, k=3):
        """预测前k个最可能的类别"""
        probas = self.predict_proba(X)
        if len(probas.shape) == 1:
            return [np.argsort(probas)[-k:][::-1]]
        else:
            top_k_indices = []
            for i in range(probas.shape[0]):
                top_k = np.argsort(probas[i])[-k:][::-1]
                # 如果模型有标签映射，映射回原始标签
                if hasattr(self.model, 'reverse_mapping'):
                    top_k = [self.model.reverse_mapping.get(idx, idx) for idx in top_k]
                top_k_indices.append(top_k)
            return top_k_indices


class XGBoostModel(BaseMLModel):
    """
    XGBoost模型实现
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
    
    def train(self, df):
        """
        训练XGBoost模型
        
        Args:
            df: 包含历史开奖数据的DataFrame
            
        Returns:
            训练好的模型
        """
        self.log("\n----- 开始训练XGBoost模型 -----")
        
        # 准备数据
        X_train, X_test, red_train_data, red_test_data, blue_train_data, blue_test_data = self.prepare_data(df)
        
        # 训练红球模型 - 为每个红球位置分别训练
        self.log("训练红球XGBoost模型...")
        red_models = []
        for i, red_data in enumerate(red_train_data):
            self.log(f"训练第{i+1}个红球位置的模型...")
            red_model = self.train_xgboost(X_train, red_data, 'red')
            red_models.append(red_model)
        self.models['red'] = red_models
        
        # 训练蓝球模型 - 为每个蓝球位置分别训练
        self.log("训练蓝球XGBoost模型...")
        blue_models = []
        for i, blue_data in enumerate(blue_train_data):
            self.log(f"训练第{i+1}个蓝球位置的模型...")
            blue_model = self.train_xgboost(X_train, blue_data, 'blue')
            blue_models.append(blue_model)
        self.models['blue'] = blue_models
        
        # 评估模型
        self.evaluate(X_test, red_test_data, blue_test_data)
        
        # 保存模型
        self.save_models()
        
        return self.models
    
    def train_xgboost(self, X_train, y_train, ball_type):
        """
        训练XGBoost模型，使用优化的超参数调优和早停机制
        
        Args:
            X_train: 训练特征
            y_train: 训练标签
            ball_type: 球类型，'red'或'blue'
            
        Returns:
            训练好的XGBoost模型
        """
        self.log(f"训练{ball_type}球XGBoost模型...")
        self.log(f"原始数据维度: 特征={X_train.shape}, 标签={y_train.shape}")
        
        # 确保y_train是整数类型，并转换为一维数组
        y_train = np.array(y_train).flatten().astype(np.int32)
        self.log(f"标签类型: {y_train.dtype}, 形状: {y_train.shape}, 唯一值: {np.unique(y_train)}")
        
        # 确定预期的类别范围（0-based索引）
        if ball_type == 'red':
            expected_classes = np.arange(self.red_range)  # 0 到 red_range-1
            max_valid_class = self.red_range - 1
        else:  # blue
            expected_classes = np.arange(self.blue_range)  # 0 到 blue_range-1
            max_valid_class = self.blue_range - 1
        
        # 在训练前强制修正所有超出范围的标签
        original_y_train = y_train.copy()
        y_train = np.clip(y_train, 0, max_valid_class)
        
        # 检查是否有修正
        modified_count = np.sum(original_y_train != y_train)
        if modified_count > 0:
            self.log(f"修正了{modified_count}个超出范围的标签")
            self.log(f"修正前范围: {np.min(original_y_train)} - {np.max(original_y_train)}")
            self.log(f"修正后范围: {np.min(y_train)} - {np.max(y_train)}")
        
        # 检查实际类别是否与预期类别匹配
        actual_classes = np.unique(y_train)
        self.log(f"预期类别: {expected_classes}")
        self.log(f"实际类别: {actual_classes}")
        
        # 验证所有类别都在预期范围内
        if np.min(actual_classes) < 0 or np.max(actual_classes) >= len(expected_classes):
            self.log(f"错误: 仍然存在超出范围的类别，这不应该发生")
            self.log(f"实际类别范围: {np.min(actual_classes)} - {np.max(actual_classes)}")
            self.log(f"预期类别范围: 0 - {len(expected_classes)-1}")
            raise ValueError(f"标签范围验证失败: 实际范围 {np.min(actual_classes)}-{np.max(actual_classes)}, 预期范围 0-{len(expected_classes)-1}")
        
        # 检查是否有缺失的类别
        missing_classes = np.setdiff1d(expected_classes, actual_classes)
        if len(missing_classes) > 0:
            self.log(f"缺失类别: {missing_classes}（这是正常的，不是所有类别都必须出现在训练数据中）")
        
        # 检查是否有类别样本数量过少的情况
        class_counts = Counter(y_train)
        min_samples = min(class_counts.values()) if class_counts else 0
        self.log(f"类别数量: {len(class_counts)}, 最少样本数: {min_samples}")
        
        # 设置交叉验证策略
        if min_samples < 2:
            # 如果有类别样本数量过少，使用KFold代替StratifiedKFold
            self.log(f"警告: 某些类别样本数量过少(最少{min_samples}个)，使用KFold代替StratifiedKFold")
            cv = KFold(n_splits=5, shuffle=True, random_state=42)
        else:
            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        
        # 优化的超参数搜索空间 - 更加精细和高效
        param_grid = {
            'n_estimators': [200, 300, 500, 800],  # 增加树的数量范围
            'max_depth': [4, 6, 8, 10],  # 调整深度范围
            'learning_rate': [0.05, 0.1, 0.15, 0.2],  # 优化学习率范围
            'subsample': [0.7, 0.8, 0.9],  # 减少搜索空间但保持有效范围
            'colsample_bytree': [0.7, 0.8, 0.9],  # 特征采样比例
            'gamma': [0, 0.1, 0.2, 0.3],  # 最小分裂损失
            'min_child_weight': [1, 3, 5, 7],  # 叶子节点最小权重
            'reg_alpha': [0, 0.1, 0.5],  # L1正则化
            'reg_lambda': [1, 1.5, 2]   # L2正则化
        }
        
        # 首先创建标签映射，确保所有标签都映射到连续的0-based索引
        unique_labels = np.unique(y_train)
        label_mapping = {label: idx for idx, label in enumerate(unique_labels)}
        reverse_mapping = {idx: label for label, idx in label_mapping.items()}
        
        # 应用标签映射
        y_train_mapped = np.array([label_mapping[label] for label in y_train])
        self.log(f"标签映射: {dict(list(label_mapping.items())[:10])}...")
        self.log(f"映射后标签范围: {np.min(y_train_mapped)} - {np.max(y_train_mapped)}")
        
        # 创建优化的XGBoost分类器
        base_params = {
            'random_state': 42,
            'num_class': len(unique_labels),
            'objective': 'multi:softprob',
            'eval_metric': 'mlogloss',
            'early_stopping_rounds': 50,  # 早停机制
            'verbosity': 0  # 减少输出
        }
        
        if self.use_gpu:
            self.log("使用GPU训练XGBoost模型")
            base_params.update({
                'tree_method': 'gpu_hist',
                'gpu_id': 0
            })
        else:
            base_params['tree_method'] = 'hist'  # 使用直方图算法加速
            
        xgb_model = xgb.XGBClassifier(**base_params)
        
        # 使用优化的随机搜索进行超参数调优
        n_iter = min(30, len(y_train_mapped) // 10)  # 根据数据量动态调整搜索次数
        n_iter = max(10, n_iter)  # 至少搜索10次
        
        random_search = RandomizedSearchCV(
            estimator=xgb_model,
            param_distributions=param_grid,
            n_iter=n_iter,
            scoring='accuracy',
            cv=cv,
            verbose=0,
            random_state=42,
            n_jobs=-1 if not self.use_gpu else 1,  # GPU时使用单进程，CPU时使用多进程
            return_train_score=True  # 返回训练分数用于分析
        )
        
        # 训练模型
        self.log(f"开始训练{ball_type}球XGBoost模型，搜索{n_iter}种参数组合...")
        start_time = time.time()
        
        # 分割验证集用于早停
        from sklearn.model_selection import train_test_split
        
        # 检查是否可以使用分层抽样
        class_counts_mapped = Counter(y_train_mapped)
        min_samples_mapped = min(class_counts_mapped.values()) if class_counts_mapped else 0
        
        if min_samples_mapped < 2:
            # 如果有类别样本数量过少，不使用分层抽样
            self.log(f"警告: 某些类别样本数量过少(最少{min_samples_mapped}个)，train_test_split不使用分层抽样")
            X_train_fit, X_val, y_train_fit, y_val = train_test_split(
                X_train, y_train_mapped, test_size=0.2, random_state=42
            )
        else:
            # 使用分层抽样
            X_train_fit, X_val, y_train_fit, y_val = train_test_split(
                X_train, y_train_mapped, test_size=0.2, random_state=42, stratify=y_train_mapped
            )
        
        # 设置验证集用于早停
        eval_set = [(X_val, y_val)]
        
        # 拟合模型
        random_search.fit(
            X_train_fit, y_train_fit,
            eval_set=eval_set,
            verbose=False
        )
        
        training_time = time.time() - start_time
        self.log(f"训练完成，耗时: {training_time:.2f}秒")
        
        # 获取最佳模型
        best_xgb = random_search.best_estimator_
        
        # 保存标签映射信息到模型中，以便预测时使用
        best_xgb.label_mapping = label_mapping
        best_xgb.reverse_mapping = reverse_mapping
        
        # 记录详细的训练结果
        self.log(f"最佳参数: {random_search.best_params_}")
        self.log(f"交叉验证最佳得分: {random_search.best_score_:.4f}")
        
        # 分析过拟合情况
        if hasattr(random_search, 'cv_results_'):
            best_idx = random_search.best_index_
            train_score = random_search.cv_results_['mean_train_score'][best_idx]
            val_score = random_search.cv_results_['mean_test_score'][best_idx]
            overfitting = train_score - val_score
            self.log(f"训练得分: {train_score:.4f}, 验证得分: {val_score:.4f}")
            if overfitting > 0.1:
                self.log(f"警告: 可能存在过拟合 (差异: {overfitting:.4f})")
        
        # 记录特征重要性
        feature_importances = best_xgb.feature_importances_
        top_n = min(15, len(feature_importances))  # 显示前15个最重要的特征
        indices = np.argsort(feature_importances)[-top_n:][::-1]
        self.log(f"前{top_n}个最重要特征的重要性:")
        for rank, i in enumerate(indices, 1):
            self.log(f"第{rank}位 - 特征{i}: {feature_importances[i]:.4f}")
        
        # 计算特征重要性统计
        importance_sum = np.sum(feature_importances)
        top_10_importance = np.sum(feature_importances[indices[:10]])
        self.log(f"前10个特征重要性占比: {top_10_importance/importance_sum:.2%}")
        
        # 包装模型以统一接口
        wrapped_model = WrappedXGBoostModel(best_xgb)
        
        return wrapped_model
    
    def evaluate(self, X_test, y_red_test, y_blue_test):
        """
        评估模型性能
        
        Args:
            X_test: 测试特征
            y_red_test: 红球测试标签列表
            y_blue_test: 蓝球测试标签列表
            
        Returns:
            红球和蓝球的准确率
        """
        self.log("评估模型性能...")
        
        red_accuracies = []
        blue_accuracies = []
        
        # 评估红球模型
        if 'red' in self.models and isinstance(self.models['red'], list):
            for i, (red_model, y_red_test_pos) in enumerate(zip(self.models['red'], y_red_test)):
                red_preds = red_model.predict(X_test)
                red_accuracy = accuracy_score(y_red_test_pos, red_preds)
                red_accuracies.append(red_accuracy)
                self.log(f"红球位置{i+1}模型准确率: {red_accuracy:.4f}")
        
        # 评估蓝球模型
        if 'blue' in self.models and isinstance(self.models['blue'], list):
            for i, (blue_model, y_blue_test_pos) in enumerate(zip(self.models['blue'], y_blue_test)):
                blue_preds = blue_model.predict(X_test)
                blue_accuracy = accuracy_score(y_blue_test_pos, blue_preds)
                blue_accuracies.append(blue_accuracy)
                self.log(f"蓝球位置{i+1}模型准确率: {blue_accuracy:.4f}")
        
        # 计算平均准确率
        avg_red_accuracy = np.mean(red_accuracies) if red_accuracies else 0
        avg_blue_accuracy = np.mean(blue_accuracies) if blue_accuracies else 0
        
        self.log(f"红球平均准确率: {avg_red_accuracy:.4f}")
        self.log(f"蓝球平均准确率: {avg_blue_accuracy:.4f}")
        
        return avg_red_accuracy, avg_blue_accuracy
    
    def save_models(self):
        """
        保存训练好的模型和缩放器
        优化版本：支持位置模型格式和更详细的模型信息记录
        
        Returns:
            bool: 是否成功保存模型
        """
        if not self.models:
            self.log("没有训练好的模型可以保存")
            return False
        
        self.log(f"开始保存{self.lottery_type}的XGBoost模型...")
        
        # 创建模型目录
        model_dir = os.path.join(self.models_dir, self.model_type)
        os.makedirs(model_dir, exist_ok=True)
        
        # 收集模型统计信息
        model_stats = {}
        for ball_type in ['red', 'blue']:
            if ball_type in self.models:
                models = self.models[ball_type]
                if isinstance(models, list):
                    # 位置模型统计
                    model_stats[f'{ball_type}_model_type'] = 'position_models'
                    model_stats[f'{ball_type}_model_count'] = len(models)
                    # 收集每个位置模型的性能指标
                    if hasattr(models[0], 'model') and hasattr(models[0].model, 'best_score_'):
                        best_scores = [getattr(m.model, 'best_score_', 0) for m in models if hasattr(m.model, 'best_score_')]
                        if best_scores:
                            model_stats[f'{ball_type}_avg_score'] = sum(best_scores) / len(best_scores)
                            model_stats[f'{ball_type}_best_score'] = max(best_scores)
                else:
                    # 单一模型统计
                    model_stats[f'{ball_type}_model_type'] = 'single_model'
                    if hasattr(models, 'model') and hasattr(models.model, 'best_score_'):
                        model_stats[f'{ball_type}_best_score'] = models.model.best_score_
        
        # 保存模型信息
        model_info = {
            'model_type': self.model_type,
            'lottery_type': self.lottery_type,
            'feature_window': self.feature_window,
            'red_count': self.red_count,
            'blue_count': self.blue_count,
            'red_range': self.red_range,
            'blue_range': self.blue_range,
            'created_time': time.strftime('%Y-%m-%d %H:%M:%S'),
            'model_version': '2.0',  # 升级版本号
            'optimization_features': [
                'position_based_prediction',
                'probability_prediction',
                'early_stopping',
                'hyperparameter_tuning',
                'regularization'
            ],
            'model_statistics': model_stats
        }
        
        info_path = os.path.join(model_dir, 'model_info.json')
        try:
            with open(info_path, 'w') as f:
                json.dump(model_info, f, indent=2, ensure_ascii=False)
            self.log(f"保存模型信息成功: {info_path}")
        except Exception as e:
            self.log(f"保存模型信息失败: {e}")
            return False
        
        # 保存模型
        models_saved = True
        
        for ball_type in ['red', 'blue']:
            if ball_type in self.models:
                models = self.models[ball_type]
                
                if isinstance(models, list):
                    # 保存位置模型（新格式）
                    for i, model in enumerate(models):
                        try:
                            model_path = os.path.join(model_dir, f'{ball_type}_model_pos_{i+1}.pkl')
                            with open(model_path, 'wb') as f:
                                pickle.dump(model, f)
                            self.log(f"保存{ball_type}球位置{i+1}模型成功")
                        except Exception as e:
                            self.log(f"保存{ball_type}球位置{i+1}模型失败: {e}")
                            models_saved = False
                else:
                    # 保存单一模型（兼容旧格式）
                    try:
                        model_path = os.path.join(model_dir, f'{ball_type}_model.pkl')
                        with open(model_path, 'wb') as f:
                            pickle.dump(models, f)
                        self.log(f"保存{ball_type}球模型成功: {model_path}")
                    except Exception as e:
                        self.log(f"保存{ball_type}球模型失败: {e}")
                        models_saved = False
        
        # 保存特征缩放器
        if 'X' in self.scalers:
            try:
                scaler_path = os.path.join(model_dir, 'scaler.pkl')
                with open(scaler_path, 'wb') as f:
                    pickle.dump(self.scalers['X'], f)
                self.log(f"保存特征缩放器成功: {scaler_path}")
            except Exception as e:
                self.log(f"保存特征缩放器失败: {e}")
                models_saved = False
        
        if models_saved:
            self.log(f"XGBoost模型保存成功: {model_dir}")
            self.log(f"模型统计信息: {model_stats}")
            return True
        else:
            self.log(f"XGBoost模型保存失败")
            return False
    
    def load_models(self):
        """
        加载保存的模型和缩放器
        优化版本：支持新的位置模型格式和更好的错误处理
        
        Returns:
            bool: 是否成功加载模型
        """
        self.log(f"尝试加载{self.lottery_type}的XGBoost模型...")
        
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
        try:
            with open(info_path, 'r') as f:
                model_info = json.load(f)
            self.log(f"加载模型信息成功: {model_info}")
            
            # 更新模型参数
            if 'feature_window' in model_info:
                self.feature_window = model_info['feature_window']
                self.log(f"更新特征窗口大小: {self.feature_window}")
        except Exception as e:
            self.log(f"加载模型信息失败: {e}")
            return False
        
        # 加载特征缩放器（只需要加载一次）
        scaler_loaded = False
        scaler_path = os.path.join(model_dir, 'scaler.pkl')
        if os.path.exists(scaler_path):
            try:
                with open(scaler_path, 'rb') as f:
                    self.scalers['X'] = pickle.load(f)
                self.log(f"加载特征缩放器成功")
                scaler_loaded = True
            except Exception as e:
                self.log(f"加载特征缩放器失败: {e}")
        
        if not scaler_loaded:
            self.log(f"警告: 特征缩放器文件不存在，将创建默认缩放器")
            from sklearn.preprocessing import StandardScaler
            self.scalers['X'] = StandardScaler()
        
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
            return True
        else:
            self.log(f"XGBoost模型加载失败，未找到必要的红球和蓝球模型")
            return False
    
    def predict(self, recent_data, num_predictions=1):
        """
        生成预测结果
        优化版本：支持位置模型、概率预测和智能号码选择
        
        Args:
            recent_data: 包含最近开奖数据的DataFrame
            num_predictions: 预测的组数
            
        Returns:
            预测的红球和蓝球号码
        """
        # 首先检查模型是否已加载
        if 'red' not in self.models or 'blue' not in self.models:
            # 尝试重新加载模型
            self.log("模型未加载，尝试重新加载...")
            load_success = self.load_models()
            if not load_success:
                self.log("错误：模型加载失败，请先训练模型")
                raise ValueError(f"模型未正确加载，请先训练或加载模型。")
        
        # 确保模型已加载
        if 'red' not in self.models or 'blue' not in self.models:
            self.log("模型未加载，无法预测")
            raise ValueError(f"模型未正确加载，请先训练或加载模型。")
        
        self.log(f"开始预测{num_predictions}组{self.lottery_type}彩票号码...")
        
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
        
        # 创建特征序列
        X_data = []
        
        # 使用滑动窗口创建序列数据
        features = []
        for j in range(self.feature_window):
            row_features = []
            for col in red_cols + blue_cols:
                row_features.append(recent_data.iloc[j][col])
            features.append(row_features)
            
        X_data.append(features)
        
        # 转换为NumPy数组
        X = np.array(X_data)
        
        # 重塑特征以适合模型
        X_reshaped = X.reshape(X.shape[0], -1)
        
        # 应用特征缩放
        try:
            if 'X' in self.scalers:
                X_scaled = self.scalers['X'].transform(X_reshaped)
                self.log("使用通用特征缩放器进行预测")
            else:
                # 回退到红球缩放器
                X_scaled = self.scalers['red'].transform(X_reshaped)
                self.log("使用红球特征缩放器进行预测")
        except Exception as e:
            self.log(f"应用特征缩放时出错: {e}")
            self.log("使用未缩放的特征进行预测")
            X_scaled = X_reshaped
        
        predictions = []
        
        for i in range(num_predictions):
            try:
                # 预测红球
                red_predictions = self._predict_balls('red', X_scaled)
                
                # 预测蓝球
                blue_predictions = self._predict_balls('blue', X_scaled)
                
                # 组合预测结果
                if red_predictions and blue_predictions:
                    if num_predictions == 1:
                        # 单组预测，返回原格式
                        return red_predictions, blue_predictions
                    else:
                        # 多组预测，返回字典格式
                        prediction = {
                            'red': red_predictions,
                            'blue': blue_predictions
                        }
                        predictions.append(prediction)
                        self.log(f"第{i+1}组预测: 红球 {red_predictions}, 蓝球 {blue_predictions}")
                else:
                    self.log(f"第{i+1}组预测失败：缺少红球或蓝球预测")
                    
            except Exception as e:
                self.log(f"第{i+1}组预测过程中出现错误: {e}")
                continue
        
        if num_predictions == 1:
            # 如果单组预测失败，返回空结果
            return [], []
        else:
            self.log(f"预测完成，共生成{len(predictions)}组号码")
            return predictions
    
    def _predict_balls(self, ball_type: str, features: np.ndarray) -> List[int]:
        """
        预测指定类型的球号码
        支持位置模型和单一模型两种格式
        
        Args:
            ball_type: 球类型 ('red' 或 'blue')
            features: 特征数据
            
        Returns:
            List[int]: 预测的号码列表
        """
        if ball_type not in self.models:
            self.log(f"未找到{ball_type}球模型")
            return []
        
        ball_count = self.red_count if ball_type == 'red' else self.blue_count
        ball_range = self.red_range if ball_type == 'red' else self.blue_range
        models = self.models[ball_type]
        
        # 检查是否为位置模型（列表格式）
        if isinstance(models, list) and len(models) == ball_count:
            # 位置模型预测
            predictions = []
            for pos, model in enumerate(models):
                try:
                    # 使用概率预测获取top-k候选
                    if hasattr(model, 'predict_proba'):
                        proba = model.predict_proba(features)[0]
                        # 获取概率最高的前3个候选
                        top_indices = np.argsort(proba)[-3:][::-1]
                        # 转换为实际号码，考虑标签映射
                        if hasattr(model.model, 'reverse_mapping'):
                            candidates = [model.model.reverse_mapping.get(idx, idx) + 1 for idx in top_indices]
                        else:
                            candidates = [idx + 1 for idx in top_indices]
                        # 确保在有效范围内
                        candidates = [c for c in candidates if 1 <= c <= ball_range]
                    else:
                        # 普通预测
                        pred = model.predict(features)[0]
                        candidates = [max(1, min(int(pred) + 1, ball_range))]
                    
                    # 选择不重复的号码
                    selected = None
                    for candidate in candidates:
                        if candidate not in predictions:
                            selected = candidate
                            break
                    
                    # 如果所有候选都重复，随机选择一个未使用的号码
                    if selected is None:
                        import random
                        available = [n for n in range(1, ball_range + 1) if n not in predictions]
                        if available:
                            selected = random.choice(available)
                        else:
                            selected = candidates[0] if candidates else pos + 1  # 最后的选择
                    
                    predictions.append(selected)
                    
                except Exception as e:
                    self.log(f"位置{pos+1}预测失败: {e}")
                    # 随机选择一个未使用的号码作为备选
                    import random
                    available = [n for n in range(1, ball_range + 1) if n not in predictions]
                    if available:
                        predictions.append(random.choice(available))
                    else:
                        predictions.append((pos % ball_range) + 1)
            
            # 确保有足够的预测结果
            while len(predictions) < ball_count:
                import random
                available = [n for n in range(1, ball_range + 1) if n not in predictions]
                if available:
                    predictions.append(random.choice(available))
                else:
                    break
            
            predictions.sort()
            return predictions[:ball_count]
        
        else:
            # 单一模型预测（旧格式）
            try:
                if hasattr(models, 'predict_proba'):
                    # 使用概率预测
                    proba = models.predict_proba(features)[0]
                    # 获取概率最高的号码作为候选
                    top_indices = np.argsort(proba)[-ball_count*2:][::-1]
                    # 考虑标签映射
                    if hasattr(models.model, 'reverse_mapping'):
                        candidates = [models.model.reverse_mapping.get(idx, idx) + 1 for idx in top_indices]
                    else:
                        candidates = [idx + 1 for idx in top_indices]
                    # 确保在有效范围内
                    candidates = [c for c in candidates if 1 <= c <= ball_range]
                else:
                    # 普通预测
                    pred = models.predict(features)
                    candidates = [max(1, min(int(p) + 1, ball_range)) for p in pred]
                
                # 去重并选择
                predictions = []
                for candidate in candidates:
                    if candidate not in predictions:
                        predictions.append(candidate)
                    if len(predictions) >= ball_count:
                        break
                
                # 补充不足的号码
                while len(predictions) < ball_count:
                    import random
                    candidate = random.randint(1, ball_range)
                    if candidate not in predictions:
                        predictions.append(candidate)
                
                predictions.sort()
                return predictions[:ball_count]
                
            except Exception as e:
                self.log(f"{ball_type}球预测失败: {e}")
                # 返回随机号码作为备选
                import random
                return sorted(random.sample(range(1, ball_range + 1), ball_count))
    
    def evaluate_model_performance(self, X_test, y_test):
        """
        评估模型性能
        优化版本：支持位置模型和详细的性能指标
        
        Args:
            X_test: 测试特征
            y_test: 测试标签
            
        Returns:
            dict: 性能评估结果
        """
        if not self.models:
            self.log("模型未训练，无法评估性能")
            return {}
        
        self.log("开始评估模型性能...")
        performance = {}
        
        for ball_type in ['red', 'blue']:
            if ball_type not in self.models:
                continue
                
            ball_count = self.red_count if ball_type == 'red' else self.blue_count
            models = self.models[ball_type]
            
            if isinstance(models, list):
                # 位置模型评估
                position_scores = []
                for i, model in enumerate(models):
                    try:
                        if i < len(y_test[ball_type]):
                            y_true = y_test[ball_type][i]
                            y_pred = model.predict(X_test)
                            
                            # 计算准确率
                            accuracy = accuracy_score(y_true, y_pred)
                            position_scores.append(accuracy)
                            
                            self.log(f"{ball_type}球位置{i+1}准确率: {accuracy:.4f}")
                    except Exception as e:
                        self.log(f"评估{ball_type}球位置{i+1}时出错: {e}")
                        position_scores.append(0.0)
                
                performance[f'{ball_type}_position_scores'] = position_scores
                performance[f'{ball_type}_avg_accuracy'] = np.mean(position_scores) if position_scores else 0.0
                performance[f'{ball_type}_best_position'] = np.argmax(position_scores) + 1 if position_scores else 0
                
            else:
                # 单一模型评估
                try:
                    y_true = y_test[ball_type][0] if isinstance(y_test[ball_type], list) else y_test[ball_type]
                    y_pred = models.predict(X_test)
                    
                    accuracy = accuracy_score(y_true, y_pred)
                    performance[f'{ball_type}_accuracy'] = accuracy
                    
                    # 如果支持概率预测，计算更多指标
                    if hasattr(models, 'predict_proba'):
                        y_proba = models.predict_proba(X_test)
                        # 计算top-k准确率
                        for k in [3, 5, 10]:
                            top_k_acc = self._calculate_top_k_accuracy(y_true, y_proba, k)
                            performance[f'{ball_type}_top_{k}_accuracy'] = top_k_acc
                    
                    self.log(f"{ball_type}球模型准确率: {accuracy:.4f}")
                    
                except Exception as e:
                    self.log(f"评估{ball_type}球模型时出错: {e}")
                    performance[f'{ball_type}_accuracy'] = 0.0
        
        # 计算整体性能
        red_acc = performance.get('red_avg_accuracy', performance.get('red_accuracy', 0.0))
        blue_acc = performance.get('blue_avg_accuracy', performance.get('blue_accuracy', 0.0))
        performance['overall_accuracy'] = (red_acc + blue_acc) / 2
        
        self.log(f"模型整体性能评估完成，平均准确率: {performance['overall_accuracy']:.4f}")
        return performance
    
    def _calculate_top_k_accuracy(self, y_true, y_proba, k):
        """
        计算Top-K准确率
        
        Args:
            y_true: 真实标签
            y_proba: 预测概率
            k: Top-K的K值
            
        Returns:
            float: Top-K准确率
        """
        try:
            correct = 0
            total = len(y_true)
            
            for i in range(total):
                # 获取概率最高的k个预测
                top_k_indices = np.argsort(y_proba[i])[-k:]
                if y_true[i] in top_k_indices:
                    correct += 1
            
            return correct / total if total > 0 else 0.0
        except Exception as e:
            self.log(f"计算Top-{k}准确率时出错: {e}")
            return 0.0