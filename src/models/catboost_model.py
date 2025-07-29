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
    
    def train(self, df):
        """
        训练CatBoost模型 - 优化版本
        
        Args:
            df: 包含历史开奖数据的DataFrame
            
        Returns:
            训练好的模型
        """
        if not CATBOOST_AVAILABLE:
            self.log("错误: CatBoost未安装或不可用，无法训练CatBoost模型")
            raise ImportError("CatBoost未安装或不可用，请先安装CatBoost")
            
        self.log("\n----- 开始训练优化版CatBoost模型 -----")
        
        # 准备增强特征数据
        X_train, X_test, y_red_train, y_red_test, y_blue_train, y_blue_test = self.prepare_enhanced_data(df)
        
        # 训练红球模型组合
        self.log("训练红球CatBoost模型组合...")
        self.train_red_ball_models(X_train, y_red_train, X_test, y_red_test)
        
        # 训练蓝球模型组合
        self.log("训练蓝球CatBoost模型组合...")
        self.train_blue_ball_models(X_train, y_blue_train, X_test, y_blue_test)
        
        # 评估模型性能
        self.evaluate_enhanced(X_test, y_red_test, y_blue_test)
        
        # 保存模型
        self.save_models()
        
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
        提取统计特征
        """
        features = []
        
        # 红球统计特征
        red_values = []
        for col in red_cols:
            red_values.extend(window_df[col].tolist())
        
        features.extend([
            np.mean(red_values),  # 平均值
            np.std(red_values),   # 标准差
            np.min(red_values),   # 最小值
            np.max(red_values),   # 最大值
            len(set(red_values)), # 唯一值数量
        ])
        
        # 蓝球统计特征
        blue_values = []
        for col in blue_cols:
            blue_values.extend(window_df[col].tolist())
        
        features.extend([
            np.mean(blue_values),
            np.std(blue_values),
            np.min(blue_values),
            np.max(blue_values),
            len(set(blue_values)),
        ])
        
        return features
    
    def _extract_trend_features(self, window_df, red_cols, blue_cols):
        """
        提取趋势特征
        """
        features = []
        
        # 红球趋势
        red_sums = []
        for _, row in window_df.iterrows():
            red_sums.append(sum([row[col] for col in red_cols]))
        
        # 计算趋势斜率
        if len(red_sums) > 1:
            x = np.arange(len(red_sums))
            red_slope = np.polyfit(x, red_sums, 1)[0]
        else:
            red_slope = 0
        
        features.append(red_slope)
        
        # 蓝球趋势
        blue_sums = []
        for _, row in window_df.iterrows():
            blue_sums.append(sum([row[col] for col in blue_cols]))
        
        if len(blue_sums) > 1:
            x = np.arange(len(blue_sums))
            blue_slope = np.polyfit(x, blue_sums, 1)[0]
        else:
            blue_slope = 0
        
        features.append(blue_slope)
        
        # 奇偶比例特征
        red_odd_count = sum(1 for _, row in window_df.iterrows() for col in red_cols if row[col] % 2 == 1)
        red_even_count = len(red_cols) * len(window_df) - red_odd_count
        features.append(red_odd_count / (red_odd_count + red_even_count) if (red_odd_count + red_even_count) > 0 else 0)
        
        blue_odd_count = sum(1 for _, row in window_df.iterrows() for col in blue_cols if row[col] % 2 == 1)
        blue_even_count = len(blue_cols) * len(window_df) - blue_odd_count
        features.append(blue_odd_count / (blue_odd_count + blue_even_count) if (blue_odd_count + blue_even_count) > 0 else 0)
        
        return features
    
    def train_red_ball_models(self, X_train, y_red_train, X_test, y_red_test):
        """
        训练红球模型组合
        """
        self.log("训练红球CatBoost模型...")
        
        # 优化的超参数
        best_params = {
            'iterations': 1000,
            'depth': 8,
            'learning_rate': 0.05,
            'l2_leaf_reg': 3,
            'border_count': 128,
            'bagging_temperature': 1,
            'random_strength': 1,
            'od_type': 'Iter',
            'od_wait': 50
        }
        
        # 设置GPU或CPU
        if self.use_gpu:
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
    
    def train_blue_ball_models(self, X_train, y_blue_train, X_test, y_blue_test):
        """
        训练蓝球模型组合
        """
        self.log("训练蓝球CatBoost模型...")
        
        # 优化的超参数
        best_params = {
            'iterations': 800,
            'depth': 6,
            'learning_rate': 0.08,
            'l2_leaf_reg': 5,
            'border_count': 64,
            'bagging_temperature': 0.5,
            'random_strength': 10,
            'od_type': 'Iter',
            'od_wait': 30
        }
        
        # 设置GPU或CPU
        if self.use_gpu:
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
    
    def evaluate_enhanced(self, X_test, y_red_test, y_blue_test):
        """
        评估增强模型性能
        
        Args:
            X_test: 测试特征
            y_red_test: 红球测试标签
            y_blue_test: 蓝球测试标签
            
        Returns:
            红球和蓝球的准确率
        """
        self.log("评估增强模型性能...")
        
        red_accuracy = 0
        blue_accuracy = 0
        
        if 'red' in self.models:
            red_preds = self.models['red'].predict(X_test)
            red_accuracy = accuracy_score(y_red_test, red_preds)
            self.log(f"红球组合模型准确率: {red_accuracy:.4f}")
            
            # 显示详细分类报告
            if len(set(y_red_test)) > 1:
                self.log("红球模型分类报告:")
                report = classification_report(y_red_test, red_preds, output_dict=True)
                self.log(f"宏平均F1: {report['macro avg']['f1-score']:.4f}")
                self.log(f"加权平均F1: {report['weighted avg']['f1-score']:.4f}")
        
        if 'blue' in self.models:
            blue_preds = self.models['blue'].predict(X_test)
            blue_accuracy = accuracy_score(y_blue_test, blue_preds)
            self.log(f"蓝球组合模型准确率: {blue_accuracy:.4f}")
            
            # 显示详细分类报告
            if len(set(y_blue_test)) > 1:
                self.log("蓝球模型分类报告:")
                report = classification_report(y_blue_test, blue_preds, output_dict=True)
                self.log(f"宏平均F1: {report['macro avg']['f1-score']:.4f}")
                self.log(f"加权平均F1: {report['weighted avg']['f1-score']:.4f}")
        
        return red_accuracy, blue_accuracy
    
    def evaluate(self, X_test, y_red_test, y_blue_test):
        """
        保持向后兼容的评估方法
        """
        return self.evaluate_enhanced(X_test, y_red_test, y_blue_test)
    
    def save_models(self):
        """
        保存模型、缩放器、编码器和模型权重
        """
        self.log("\n----- 保存模型和缩放器 -----")
        
        # 创建模型目录
        model_dir = os.path.join(self.models_dir, self.model_type)
        os.makedirs(model_dir, exist_ok=True)
        
        # 保存红球模型
        if 'red' in self.models:
            model_path = os.path.join(model_dir, 'red_model.pkl')
            with open(model_path, 'wb') as f:
                pickle.dump(self.models['red'], f)
            self.log(f"红球模型保存到: {model_path}")
        
        # 保存蓝球模型
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
        elif 'X' in self.scalers:
            scaler_path = os.path.join(model_dir, 'scaler.pkl')
            with open(scaler_path, 'wb') as f:
                pickle.dump(self.scalers['X'], f)
            self.log(f"特征缩放器保存到: {scaler_path}")
        
        # 保存模型信息
        model_info = {
            'model_type': self.model_type,
            'lottery_type': self.lottery_type,
            'feature_window': self.feature_window,
            'n_features_in_': X_train.shape[1] if 'X_train' in locals() else None
        }
        
        info_path = os.path.join(model_dir, 'model_info.json')
        with open(info_path, 'w') as f:
            json.dump(model_info, f)
        self.log(f"模型信息保存到: {info_path}")
    
    def load_models(self):
        """
        加载保存的模型和缩放器
        
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
            
            self.log("所有模型和编码器加载成功")
            return True
                
        except Exception as e:
            self.log(f"加载模型时出错: {e}")
            import traceback
            self.log(traceback.format_exc())
            return False
    
    def predict(self, recent_data):
        """
        生成预测结果 - 优化版本
        
        Args:
            recent_data: 包含最近开奖数据的DataFrame
            
        Returns:
            预测的红球和蓝球号码
        """
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
            # 预测红球组合
            red_pred_encoded = self.models['red'].predict(X_scaled)[0]
            red_combo_str = self.red_encoder.inverse_transform([red_pred_encoded])[0]
            # 将字符串转换回数字元组
            red_combo = eval(red_combo_str)  # 安全地解析字符串表示的元组
            red_predictions = list(red_combo)
            
            # 预测蓝球组合
            blue_pred_encoded = self.models['blue'].predict(X_scaled)[0]
            blue_combo_str = self.blue_encoder.inverse_transform([blue_pred_encoded])[0]
            # 将字符串转换回数字元组
            blue_combo = eval(blue_combo_str)  # 安全地解析字符串表示的元组
            blue_predictions = list(blue_combo)
            
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