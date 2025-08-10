# -*- coding:utf-8 -*-
"""
XGBoost model implementation for lottery prediction
"""

import os
import numpy as np
import pandas as pd
import pickle
import torch
import xgboost as xgb
from sklearn.metrics import accuracy_score

from .base import BaseMLModel


class WrappedXGBoostModel:
    """
    XGBoost模型包装器，用于统一预测接口
    """
    def __init__(self, model, processor=None):
        self.model = model
        self.processor = processor
        
    def predict(self, X):
        if not isinstance(X, xgb.DMatrix):
            X = xgb.DMatrix(X)
        raw_preds = self.model.predict(X)
        if self.processor:
            return self.processor(raw_preds)
        return raw_preds


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
        
        # 训练红球和蓝球模型
        self.train_xgboost_models(X_train, X_test, red_train_data, red_test_data, blue_train_data, blue_test_data)
        
        # 保存模型
        self.save_models()
        
        return self.models
    
    def train_xgboost_models(self, X_train, X_test, red_train_data, red_test_data, blue_train_data, blue_test_data):
        """
        训练XGBoost红球和蓝球模型
        
        Args:
            X_train: 训练特征
            X_test: 测试特征
            red_train_data: 红球训练数据
            red_test_data: 红球测试数据
            blue_train_data: 蓝球训练数据
            blue_test_data: 蓝球测试数据
        """
        self.log("训练XGBoost模型...")
        
        # 训练红球模型
        self.log("训练红球XGBoost模型...")
        red_models = []
        for i in range(self.red_count):
            self.log(f"训练第{i+1}个红球模型...")
            model = self.train_xgboost(X_train, red_train_data[i], 'red')
            red_models.append(model)
        
        # 训练蓝球模型
        self.log("训练蓝球XGBoost模型...")
        blue_models = []
        for i in range(self.blue_count):
            self.log(f"训练第{i+1}个蓝球模型...")
            model = self.train_xgboost(X_train, blue_train_data[i], 'blue')
            blue_models.append(model)
        
        # 存储模型
        self.models = {
            'red': red_models,
            'blue': blue_models
        }
        
        # 评估模型
        self.evaluate_model(X_test, red_test_data, blue_test_data)
    
    def train_xgboost(self, X_train, y_train, ball_type):
        """
        训练单个XGBoost模型
        
        Args:
            X_train: 训练特征
            y_train: 训练标签
            ball_type: 球类型，'red'或'blue'
            
        Returns:
            训练好的XGBoost模型
        """
        self.log(f"训练{ball_type}球XGBoost模型...")
        self.log(f"数据维度: 特征={X_train.shape}, 标签={y_train.shape}")
        
        # 设置参数
        params = {
            'objective': 'multi:softmax',
            'num_class': self.red_range + 1 if ball_type == 'red' else self.blue_range + 1,
            'learning_rate': 0.1,
            'max_depth': 6,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'eval_metric': 'mlogloss',
            'verbosity': 0
        }
        
        self.log(f"XGBoost参数: {params}")
        
        # 如果使用GPU并且GPU可用，则添加GPU参数
        cuda_available = torch.cuda.is_available()
        mps_available = hasattr(torch, 'mps') and torch.backends.mps.is_available()
        
        if self.use_gpu and cuda_available:
            params['tree_method'] = 'gpu_hist'  # 使用CUDA GPU加速
            self.log("XGBoost使用CUDA GPU加速训练")
        elif self.use_gpu and mps_available:
            # 注意：XGBoost目前可能不直接支持MPS，但我们保留这个检查以便未来兼容
            self.log("警告：XGBoost可能不支持MPS后端，将使用CPU训练")
        
        # 创建DMatrix数据结构
        dtrain = xgb.DMatrix(X_train, label=y_train)
        
        # 设置评估列表，用于记录训练进度
        watchlist = [(dtrain, 'train')]
        
        # 使用正确的XGBoost回调函数API
        class XGBCallback(xgb.callback.TrainingCallback):
            def __init__(self, log_func):
                self.log_func = log_func
                self.iteration = 0
                
            def after_iteration(self, model, epoch, evals_log):
                if (self.iteration + 1) % 10 == 0 or self.iteration == 0:  # 每10次迭代输出一次
                    # 从evals_log获取最新的评估结果
                    metric_values = evals_log.get('train', {}).get('mlogloss', [])
                    if metric_values:
                        msg = f'XGBoost迭代 {self.iteration + 1:3d}: {metric_values[-1]:.6f}'
                        self.log_func(msg)
                self.iteration += 1
                return False
        
        # 创建回调函数
        callbacks = [XGBCallback(self.log)]
        
        # 训练模型
        self.log(f"开始训练XGBoost模型...")
        model = xgb.train(
            params, 
            dtrain, 
            num_boost_round=100, 
            evals=watchlist,
            callbacks=callbacks,
            verbose_eval=False  # 禁用内置的输出，使用我们的回调
        )
        
        # 输出特征重要性
        if model.feature_names:
            importance = model.get_score(importance_type='gain')
            sorted_importance = sorted(importance.items(), key=lambda x: x[1], reverse=True)
            
            self.log(f"XGBoost特征重要性 (前10个):")
            for i, (feature, score) in enumerate(sorted_importance[:10]):
                self.log(f"  {i+1}. {feature}: {score:.4f}")
        
        # 创建包装模型 - XGBoost的multi:softmax直接返回类别索引，不需要额外处理
        wrapped_model = WrappedXGBoostModel(model, None)
        
        self.log(f"{ball_type}球XGBoost模型训练完成")
        return wrapped_model
    
    def prepare_prediction_data(self, recent_data):
        """
        准备预测数据
        
        Args:
            recent_data: 包含最近开奖数据的DataFrame
            
        Returns:
            处理后的预测特征数据
        """
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
                self.log("警告: 未找到特征缩放器，使用未缩放的特征进行预测")
                X_scaled = X_reshaped
        except Exception as e:
            self.log(f"应用特征缩放时出错: {e}")
            self.log("使用未缩放的特征进行预测")
            X_scaled = X_reshaped
            
        return X_scaled
    
    def predict(self, recent_data, check_history=False, similarity_rules=None):
        """
        使用训练好的模型进行预测
        
        Args:
            recent_data: 包含最近开奖数据的DataFrame
            check_history: 是否检查历史相似性
            similarity_rules: 相似性规则
            
        Returns:
            预测的红球和蓝球号码
        """
        if not self.models:
            if not self.load_models():
                raise ValueError("模型未训练且无法加载已保存的模型")
        
        self.log("\n----- 开始XGBoost预测 -----")
        
        # 准备预测数据
        X_pred = self.prepare_prediction_data(recent_data)
        
        # 预测红球
        red_predictions = []
        for i, model in enumerate(self.models['red']):
            pred = model.predict(X_pred)[0]  # 取第一个预测结果
            red_num = int(pred) + 1  # +1 转回原始号码范围
            # 确保号码在有效范围内
            red_num = max(1, min(red_num, self.red_range))
            red_predictions.append(red_num)
            self.log(f"红球{i+1}预测: {red_num}")
        
        # 预测蓝球
        blue_predictions = []
        for i, model in enumerate(self.models['blue']):
            pred = model.predict(X_pred)[0]  # 取第一个预测结果
            blue_num = int(pred) + 1  # +1 转回原始号码范围
            # 确保号码在有效范围内
            blue_num = max(1, min(blue_num, self.blue_range))
            blue_predictions.append(blue_num)
            self.log(f"蓝球{i+1}预测: {blue_num}")
        
        self.log(f"XGBoost预测完成: 红球={red_predictions}, 蓝球={blue_predictions}")
        
        # 检查历史相似性
        if check_history:
            from src.core.history_check import check_prediction_against_history, adjust_prediction_to_avoid_history
            
            # 组合红蓝球号码
            prediction = red_predictions + blue_predictions
            
            # 检查是否与历史数据相似
            is_similar, similarity_info = check_prediction_against_history(
                prediction, self.lottery_type, similarity_rules
            )
            
            # 如果相似，调整预测结果
            if is_similar:
                self.log(f"预测结果与历史数据相似: {similarity_info}")
                adjusted_prediction = adjust_prediction_to_avoid_history(
                    prediction, self.lottery_type, similarity_rules
                )
                
                # 分离调整后的红蓝球
                if self.lottery_type == 'dlt':
                    red_predictions = sorted(adjusted_prediction[:5])
                    blue_predictions = sorted(adjusted_prediction[5:])
                else:  # ssq
                    red_predictions = sorted(adjusted_prediction[:6])
                    blue_predictions = sorted(adjusted_prediction[6:])
                
                self.log(f"调整后的预测结果: 红球={red_predictions}, 蓝球={blue_predictions}")
        
        return red_predictions, blue_predictions
    
    def save_models(self):
        """
        保存训练好的模型和缩放器
        """
        self.log("\n----- 保存XGBoost模型和缩放器 -----")
        
        # 创建模型目录
        model_dir = os.path.join(self.models_dir, self.model_type)
        os.makedirs(model_dir, exist_ok=True)
        
        # 保存红球模型
        if 'red' in self.models:
            for i, model in enumerate(self.models['red']):
                model_path = os.path.join(model_dir, f'red_model_{i+1}.pkl')
                with open(model_path, 'wb') as f:
                    pickle.dump(model, f)
                self.log(f"红球模型{i+1}保存到: {model_path}")
        
        # 保存蓝球模型
        if 'blue' in self.models:
            for i, model in enumerate(self.models['blue']):
                model_path = os.path.join(model_dir, f'blue_model_{i+1}.pkl')
                with open(model_path, 'wb') as f:
                    pickle.dump(model, f)
                self.log(f"蓝球模型{i+1}保存到: {model_path}")
        
        # 保存特征缩放器
        if 'X' in self.scalers:
            scaler_path = os.path.join(model_dir, 'scaler_X.pkl')
            with open(scaler_path, 'wb') as f:
                pickle.dump(self.scalers['X'], f)
            self.log(f"特征缩放器保存到: {scaler_path}")
    
    def load_models(self):
        """
        加载保存的模型和缩放器
        
        Returns:
            bool: 是否成功加载模型
        """
        self.log(f"尝试加载{self.lottery_type}的XGBoost模型...")
        
        model_dir = os.path.join(self.models_dir, self.model_type)
        if not os.path.exists(model_dir):
            self.log(f"模型目录不存在: {model_dir}")
            return False
        
        try:
            # 加载红球模型
            red_models = []
            for i in range(self.red_count):
                model_path = os.path.join(model_dir, f'red_model_{i+1}.pkl')
                if os.path.exists(model_path):
                    with open(model_path, 'rb') as f:
                        model = pickle.load(f)
                    red_models.append(model)
                    self.log(f"红球模型{i+1}加载成功: {model_path}")
                else:
                    self.log(f"红球模型{i+1}文件不存在: {model_path}")
                    return False
            
            # 加载蓝球模型
            blue_models = []
            for i in range(self.blue_count):
                model_path = os.path.join(model_dir, f'blue_model_{i+1}.pkl')
                if os.path.exists(model_path):
                    with open(model_path, 'rb') as f:
                        model = pickle.load(f)
                    blue_models.append(model)
                    self.log(f"蓝球模型{i+1}加载成功: {model_path}")
                else:
                    self.log(f"蓝球模型{i+1}文件不存在: {model_path}")
                    return False
            
            # 加载特征缩放器
            scaler_path = os.path.join(model_dir, 'scaler_X.pkl')
            if os.path.exists(scaler_path):
                with open(scaler_path, 'rb') as f:
                    scaler = pickle.load(f)
                self.scalers['X'] = scaler
                self.log(f"特征缩放器加载成功: {scaler_path}")
            else:
                self.log(f"特征缩放器文件不存在: {scaler_path}")
                return False
            
            # 存储加载的模型
            self.models = {
                'red': red_models,
                'blue': blue_models
            }
            
            self.log("XGBoost模型加载完成")
            return True
            
        except Exception as e:
            self.log(f"加载XGBoost模型时出错: {e}")
            return False
    
    def evaluate_model(self, X_test, red_test_data, blue_test_data):
        """
        评估模型性能
        
        Args:
            X_test: 测试特征
            red_test_data: 红球测试数据
            blue_test_data: 蓝球测试数据
        """
        self.log("\n----- 评估XGBoost模型 -----")
        
        # 评估红球模型
        for i, model in enumerate(self.models['red']):
            # 直接使用XGBoost模型进行预测，不使用processor
            if not isinstance(X_test, xgb.DMatrix):
                X_test_dmatrix = xgb.DMatrix(X_test)
            else:
                X_test_dmatrix = X_test
            y_pred = model.model.predict(X_test_dmatrix)
            # 确保y_pred是整数类型
            y_pred = y_pred.astype(int)
            accuracy = accuracy_score(red_test_data[i], y_pred)
            self.log(f"红球{i+1}模型准确率: {accuracy:.4f}")
        
        # 评估蓝球模型
        for i, model in enumerate(self.models['blue']):
            # 直接使用XGBoost模型进行预测，不使用processor
            if not isinstance(X_test, xgb.DMatrix):
                X_test_dmatrix = xgb.DMatrix(X_test)
            else:
                X_test_dmatrix = X_test
            y_pred = model.model.predict(X_test_dmatrix)
            # 确保y_pred是整数类型
            y_pred = y_pred.astype(int)
            accuracy = accuracy_score(blue_test_data[i], y_pred)
            self.log(f"蓝球{i+1}模型准确率: {accuracy:.4f}")