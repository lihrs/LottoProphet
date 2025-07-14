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
        return self.model.predict(X)


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
        
        # 训练红球模型
        self.log("训练红球XGBoost模型...")
        red_model = self.train_xgboost(X_train, red_train_data[0], 'red')
        self.models['red'] = red_model
        
        # 训练蓝球模型
        self.log("训练蓝球XGBoost模型...")
        blue_model = self.train_xgboost(X_train, blue_train_data[0], 'blue')
        self.models['blue'] = blue_model
        
        # 评估模型
        self.evaluate(X_test, red_test_data[0].reshape(-1, 1), blue_test_data[0].reshape(-1, 1))
        
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
        
        # 创建XGBoost分类器
        if self.use_gpu:
            self.log("使用GPU训练XGBoost模型")
            xgb_model = xgb.XGBClassifier(random_state=42, tree_method='gpu_hist', gpu_id=0)
        else:
            xgb_model = xgb.XGBClassifier(random_state=42)
        
        # 使用随机搜索进行超参数调优
        random_search = RandomizedSearchCV(
            estimator=xgb_model,
            param_distributions=param_grid,
            n_iter=20,  # 尝试20种不同的组合
            scoring='accuracy',
            cv=cv,
            verbose=0,
            random_state=42,
            n_jobs=-1
        )
        
        # 训练模型
        try:
            self.log("开始训练XGBoost模型...")
            random_search.fit(X_train, y_train)
            # 获取最佳模型
            best_xgb = random_search.best_estimator_
        except ValueError as e:
            if "Invalid classes inferred from unique values of `y`" in str(e):
                self.log(f"警告: 类别推断错误，尝试更强的类型转换。错误信息: {str(e)}")
                # 记录当前标签信息
                self.log(f"当前标签信息: 类型={y_train.dtype}, 形状={y_train.shape}, 唯一值={np.unique(y_train)}")
                
                # 尝试更强的类型转换
                y_train = np.array(y_train, dtype=np.int32).flatten()
                self.log(f"重新转换后的标签类型: {y_train.dtype}, 形状: {y_train.shape}, 唯一值: {np.unique(y_train)}")
                
                # 重新创建随机搜索对象
                if self.use_gpu:
                    xgb_model = xgb.XGBClassifier(random_state=42, tree_method='gpu_hist', gpu_id=0)
                else:
                    xgb_model = xgb.XGBClassifier(random_state=42)
                    
                random_search = RandomizedSearchCV(
                    estimator=xgb_model,
                    param_distributions=param_grid,
                    n_iter=20,
                    scoring='accuracy',
                    cv=cv,
                    verbose=0,
                    random_state=42,
                    n_jobs=-1
                )
                
                self.log("使用转换后的标签重新训练XGBoost模型...")
                random_search.fit(X_train, y_train)
                best_xgb = random_search.best_estimator_
            else:
                self.log(f"训练过程中出现错误: {str(e)}")
                raise
        
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
            y_red_test: 红球测试标签
            y_blue_test: 蓝球测试标签
            
        Returns:
            红球和蓝球的准确率
        """
        self.log("评估模型性能...")
        
        if len(y_red_test.shape) == 1:
            y_red_test = y_red_test.reshape(-1, 1)
        if len(y_blue_test.shape) == 1:
            y_blue_test = y_blue_test.reshape(-1, 1)
            
        red_accuracy = 0
        blue_accuracy = 0
        
        if 'red' in self.models:
            red_preds = self.models['red'].predict(X_test)
            red_accuracy = accuracy_score(y_red_test, red_preds.reshape(-1, 1))
            self.log(f"红球模型准确率: {red_accuracy:.4f}")
        
        if 'blue' in self.models:
            blue_preds = self.models['blue'].predict(X_test)
            blue_accuracy = accuracy_score(y_blue_test, blue_preds.reshape(-1, 1))
            self.log(f"蓝球模型准确率: {blue_accuracy:.4f}")
        
        return red_accuracy, blue_accuracy
    
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