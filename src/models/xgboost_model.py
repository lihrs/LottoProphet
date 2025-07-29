# -*- coding:utf-8 -*-
"""
XGBoost model implementation for lottery prediction
"""

import os
import numpy as np
import pandas as pd
import pickle
import joblib
import json
import xgboost as xgb
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, KFold
from sklearn.metrics import accuracy_score
from collections import Counter

from .base import BaseMLModel


class WrappedXGBoostModel:
    """
    XGBoost模型包装器，用于统一预测接口
    """
    def __init__(self, model, processor=None):
        self.model = model
        self.processor = processor
        
    def predict(self, X):
        if self.processor:
            X = self.processor(X)
        predictions = self.model.predict(X)
        
        # 如果模型有标签映射，需要将预测结果映射回原始标签
        if hasattr(self.model, 'reverse_mapping'):
            mapped_predictions = np.array([self.model.reverse_mapping.get(pred, pred) for pred in predictions])
            return mapped_predictions
        
        return predictions


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
        训练XGBoost模型，使用交叉验证和超参数调优
        
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
        
        # 设置超参数搜索空间
        param_grid = {
            'n_estimators': [100, 200, 300, 500],
            'max_depth': [3, 5, 7, 9],
            'learning_rate': [0.01, 0.05, 0.1, 0.2],
            'subsample': [0.6, 0.8, 1.0],
            'colsample_bytree': [0.6, 0.8, 1.0],
            'gamma': [0, 0.1, 0.2],
            'min_child_weight': [1, 3, 5]
        }
        
        # 首先创建标签映射，确保所有标签都映射到连续的0-based索引
        unique_labels = np.unique(y_train)
        label_mapping = {label: idx for idx, label in enumerate(unique_labels)}
        reverse_mapping = {idx: label for label, idx in label_mapping.items()}
        
        # 应用标签映射
        y_train_mapped = np.array([label_mapping[label] for label in y_train])
        self.log(f"标签映射: {dict(list(label_mapping.items())[:10])}...")
        self.log(f"映射后标签范围: {np.min(y_train_mapped)} - {np.max(y_train_mapped)}")
        
        # 创建XGBoost分类器，使用映射后的类别数量
        if self.use_gpu:
            self.log("使用GPU训练XGBoost模型")
            xgb_model = xgb.XGBClassifier(
                random_state=42, 
                tree_method='gpu_hist', 
                gpu_id=0,
                num_class=len(unique_labels),
                objective='multi:softprob'
            )
        else:
            xgb_model = xgb.XGBClassifier(
                random_state=42,
                num_class=len(unique_labels),
                objective='multi:softprob'
            )
        
        # 使用随机搜索进行超参数调优
        random_search = RandomizedSearchCV(
            estimator=xgb_model,
            param_distributions=param_grid,
            n_iter=20,  # 尝试20种不同的组合
            scoring='accuracy',
            cv=cv,
            verbose=0,
            random_state=42,
            n_jobs=1  # 使用单进程避免并发问题
        )
        
        # 训练模型
        self.log("开始训练XGBoost模型...")
        random_search.fit(X_train, y_train_mapped)
        # 获取最佳模型
        best_xgb = random_search.best_estimator_
        
        # 保存标签映射信息到模型中，以便预测时使用
        best_xgb.label_mapping = label_mapping
        best_xgb.reverse_mapping = reverse_mapping
        
        # 记录最佳参数
        self.log(f"最佳参数: {random_search.best_params_}")
        self.log(f"交叉验证最佳得分: {random_search.best_score_:.4f}")
        
        # 记录特征重要性
        feature_importances = best_xgb.feature_importances_
        top_n = 10  # 只显示前10个最重要的特征
        indices = np.argsort(feature_importances)[-top_n:]
        self.log(f"前{top_n}个最重要特征的重要性:")
        for i in indices:
            self.log(f"特征 {i}: {feature_importances[i]:.4f}")
        
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
        保存模型、缩放器和模型权重
        """
        self.log("\n----- 保存模型和缩放器 -----")
        
        # 创建模型目录
        model_dir = os.path.join(self.models_dir, self.model_type)
        os.makedirs(model_dir, exist_ok=True)
        
        # 保存红球模型
        if 'red' in self.models:
            if isinstance(self.models['red'], list):
                # 保存多个红球位置模型
                for i, red_model in enumerate(self.models['red']):
                    model_path = os.path.join(model_dir, f'red_model_pos_{i+1}.pkl')
                    with open(model_path, 'wb') as f:
                        pickle.dump(red_model, f)
                    self.log(f"红球位置{i+1}模型保存到: {model_path}")
            else:
                # 保存单个红球模型（向后兼容）
                model_path = os.path.join(model_dir, 'red_model.pkl')
                with open(model_path, 'wb') as f:
                    pickle.dump(self.models['red'], f)
                self.log(f"红球模型保存到: {model_path}")
        
        # 保存蓝球模型
        if 'blue' in self.models:
            if isinstance(self.models['blue'], list):
                # 保存多个蓝球位置模型
                for i, blue_model in enumerate(self.models['blue']):
                    model_path = os.path.join(model_dir, f'blue_model_pos_{i+1}.pkl')
                    with open(model_path, 'wb') as f:
                        pickle.dump(blue_model, f)
                    self.log(f"蓝球位置{i+1}模型保存到: {model_path}")
            else:
                # 保存单个蓝球模型（向后兼容）
                model_path = os.path.join(model_dir, 'blue_model.pkl')
                with open(model_path, 'wb') as f:
                    pickle.dump(self.models['blue'], f)
                self.log(f"蓝球模型保存到: {model_path}")
        
        # 保存特征缩放器
        if 'X' in self.scalers:
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
        
        # 加载模型和缩放器
        models_loaded = True
        balls_loaded = 0
        
        for ball_type in ['red', 'blue']:
            try:
                # 加载模型
                model_path = os.path.join(model_dir, f'{ball_type}_model.pkl')
                if os.path.exists(model_path):
                    with open(model_path, 'rb') as f:
                        self.models[ball_type] = pickle.load(f)
                    self.log(f"加载{ball_type}球模型成功")
                    balls_loaded += 1
                else:
                    self.log(f"警告: {ball_type}球模型文件不存在: {model_path}")
                    models_loaded = False
                
                # 加载特征缩放器
                scaler_path = os.path.join(model_dir, 'scaler.pkl')
                if os.path.exists(scaler_path):
                    with open(scaler_path, 'rb') as f:
                        self.scalers['X'] = pickle.load(f)
                    self.log(f"加载特征缩放器成功")
                else:
                    # 尝试加载特定的球特征缩放器
                    scaler_path = os.path.join(model_dir, f'{ball_type}_scaler.pkl')
                    if os.path.exists(scaler_path):
                        with open(scaler_path, 'rb') as f:
                            self.scalers[ball_type] = pickle.load(f)
                        self.log(f"加载{ball_type}球特征缩放器成功")
                    else:
                        self.log(f"警告: 特征缩放器文件不存在")
                        # 如果没有缩放器，创建一个默认的缩放器
                        from sklearn.preprocessing import StandardScaler
                        self.scalers[ball_type] = StandardScaler()
                        self.log(f"创建了{ball_type}球默认特征缩放器作为替代")
            except Exception as e:
                self.log(f"加载{ball_type}球模型失败: {e}")
                models_loaded = False
        
        # 如果所有模型都成功加载，返回True
        if models_loaded:
            self.log(f"XGBoost模型加载成功")
            return True
        elif balls_loaded >= 2:  # 至少加载了红球和蓝球
            # 即使有警告，只要基础模型存在，我们也认为模型可用
            self.log(f"XGBoost模型加载成功，但有一些警告")
            return True
        else:
            self.log(f"XGBoost模型加载失败，未找到必要的红球和蓝球模型")
            return False
    
    def predict(self, recent_data):
        """
        生成预测结果
        
        Args:
            recent_data: 包含最近开奖数据的DataFrame
            
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
        
        try:
            # 预测红球
            red_pred = self.models['red'].predict(X_scaled)[0]
            red_predictions = [int(red_pred) + 1]  # +1 转回原始号码范围
            
            # 预测蓝球
            blue_pred = self.models['blue'].predict(X_scaled)[0]
            blue_predictions = [int(blue_pred) + 1]  # +1 转回原始号码范围
        except Exception as e:
            self.log(f"预测过程中出错: {e}")
            import traceback
            self.log(traceback.format_exc())
            raise ValueError(f"预测过程中出错: {e}")
            
        # 生成完整号码集
        while len(red_predictions) < self.red_count:
            # 如果预测的红球数量不足，随机补充
            new_num = np.random.randint(1, self.red_range + 1)
            if new_num not in red_predictions:
                red_predictions.append(new_num)
                
        while len(blue_predictions) < self.blue_count:
            # 如果预测的蓝球数量不足，随机补充
            new_num = np.random.randint(1, self.blue_range + 1)
            if new_num not in blue_predictions:
                blue_predictions.append(new_num)
                
        # 确保号码不重复且按升序排列
        red_predictions = sorted(list(set(red_predictions)))[:self.red_count]
        blue_predictions = sorted(list(set(blue_predictions)))[:self.blue_count]
        
        return red_predictions, blue_predictions