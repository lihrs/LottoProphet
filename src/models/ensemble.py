# -*- coding:utf-8 -*-
"""
Ensemble model implementation for lottery prediction
"""

import os
import numpy as np
import pandas as pd
import pickle
import joblib
import json
import logging
from sklearn.metrics import accuracy_score
from collections import Counter

# 导入各个子模型
from .random_forest import RandomForestModel
from .xgboost_model import XGBoostModel
from .base import BaseMLModel

# 条件导入LightGBM和CatBoost模型
try:
    from .lightgbm_model import LightGBMModel, LIGHTGBM_AVAILABLE
except ImportError:
    LIGHTGBM_AVAILABLE = False

try:
    from .catboost_model import CatBoostModel, CATBOOST_AVAILABLE
except ImportError:
    CATBOOST_AVAILABLE = False


class EnsembleModel(BaseMLModel):
    """
    集成模型实现，结合多个基础模型的预测结果
    """
    
    def __init__(self, lottery_type='dlt', feature_window=10, log_callback=None, use_gpu=False):
        """
        初始化集成模型
        
        Args:
            lottery_type: 彩票类型，'dlt'或'ssq'
            feature_window: 特征窗口大小，使用多少期数据作为特征
            log_callback: 日志回调函数，用于将日志发送到UI
            use_gpu: 是否使用GPU训练
        """
        super().__init__(lottery_type, feature_window, log_callback, use_gpu)
        self.model_type = 'ensemble'
        self.sub_models = {}
        self.model_weights = {}
    
    def train(self, df):
        """
        训练集成模型，包括训练各个子模型
        
        Args:
            df: 包含历史开奖数据的DataFrame
            
        Returns:
            训练好的模型
        """
        self.log("\n----- 开始训练集成模型 -----")
        
        # 准备数据
        X_train, X_test, red_train_data, red_test_data, blue_train_data, blue_test_data = self.prepare_data(df)
        
        # 训练各个子模型
        self.train_ensemble(X_train, X_test, red_train_data, red_test_data, blue_train_data, blue_test_data)
        
        # 保存模型
        self.save_models()
        
        return self.models
    
    def train_ensemble(self, X_train, X_test, red_train_data, red_test_data, blue_train_data, blue_test_data):
        """
        训练集成模型，包括训练各个子模型并计算权重
        
        Args:
            X_train: 训练特征
            X_test: 测试特征
            red_train_data: 红球训练数据
            red_test_data: 红球测试数据
            blue_train_data: 蓝球训练数据
            blue_test_data: 蓝球测试数据
            
        Returns:
            训练好的集成模型
        """
        self.log("训练集成模型...")
        
        # 初始化子模型和权重
        self.sub_models = {
            'red': {},
            'blue': {}
        }
        self.model_weights = {
            'red': {},
            'blue': {}
        }
        
        # 训练随机森林模型
        self.log("\n----- 训练随机森林子模型 -----")
        rf_model = RandomForestModel(self.lottery_type, self.feature_window, self.log, self.use_gpu)
        rf_model.scalers = self.scalers  # 共享缩放器
        
        # 训练红球模型
        self.log("训练红球随机森林模型...")
        red_rf = rf_model.train_random_forest(X_train, red_train_data[0], 'red')
        self.sub_models['red']['random_forest'] = red_rf
        
        # 训练蓝球模型
        self.log("训练蓝球随机森林模型...")
        blue_rf = rf_model.train_random_forest(X_train, blue_train_data[0], 'blue')
        self.sub_models['blue']['random_forest'] = blue_rf
        
        # 评估随机森林模型
        red_preds = red_rf.predict(X_test)
        red_accuracy = accuracy_score(red_test_data[0], red_preds)
        self.model_weights['red']['random_forest'] = red_accuracy
        self.log(f"随机森林红球模型准确率: {red_accuracy:.4f}")
        
        blue_preds = blue_rf.predict(X_test)
        blue_accuracy = accuracy_score(blue_test_data[0], blue_preds)
        self.model_weights['blue']['random_forest'] = blue_accuracy
        self.log(f"随机森林蓝球模型准确率: {blue_accuracy:.4f}")
        
        # 训练XGBoost模型
        self.log("\n----- 训练XGBoost子模型 -----")
        xgb_model = XGBoostModel(self.lottery_type, self.feature_window, self.log, self.use_gpu)
        xgb_model.scalers = self.scalers  # 共享缩放器
        
        # 训练红球模型
        self.log("训练红球XGBoost模型...")
        red_xgb = xgb_model.train_xgboost(X_train, red_train_data[0], 'red')
        self.sub_models['red']['xgboost'] = red_xgb
        
        # 训练蓝球模型
        self.log("训练蓝球XGBoost模型...")
        blue_xgb = xgb_model.train_xgboost(X_train, blue_train_data[0], 'blue')
        self.sub_models['blue']['xgboost'] = blue_xgb
        
        # 评估XGBoost模型
        red_preds = red_xgb.predict(X_test)
        red_accuracy = accuracy_score(red_test_data[0], red_preds)
        self.model_weights['red']['xgboost'] = red_accuracy
        self.log(f"XGBoost红球模型准确率: {red_accuracy:.4f}")
        
        blue_preds = blue_xgb.predict(X_test)
        blue_accuracy = accuracy_score(blue_test_data[0], blue_preds)
        self.model_weights['blue']['xgboost'] = blue_accuracy
        self.log(f"XGBoost蓝球模型准确率: {blue_accuracy:.4f}")
        
        # 训练LightGBM模型（如果可用）
        if LIGHTGBM_AVAILABLE:
            self.log("\n----- 训练LightGBM子模型 -----")
            lgb_model = LightGBMModel(self.lottery_type, self.feature_window, self.log, self.use_gpu)
            lgb_model.scalers = self.scalers  # 共享缩放器
            
            # 训练红球模型
            self.log("训练红球LightGBM模型...")
            red_lgb = lgb_model.train_lightgbm(X_train, red_train_data[0], 'red')
            self.sub_models['red']['lightgbm'] = red_lgb
            
            # 训练蓝球模型
            self.log("训练蓝球LightGBM模型...")
            blue_lgb = lgb_model.train_lightgbm(X_train, blue_train_data[0], 'blue')
            self.sub_models['blue']['lightgbm'] = blue_lgb
            
            # 评估LightGBM模型
            red_preds = red_lgb.predict(X_test)
            red_accuracy = accuracy_score(red_test_data[0], red_preds)
            self.model_weights['red']['lightgbm'] = red_accuracy
            self.log(f"LightGBM红球模型准确率: {red_accuracy:.4f}")
            
            blue_preds = blue_lgb.predict(X_test)
            blue_accuracy = accuracy_score(blue_test_data[0], blue_preds)
            self.model_weights['blue']['lightgbm'] = blue_accuracy
            self.log(f"LightGBM蓝球模型准确率: {blue_accuracy:.4f}")
        else:
            self.log("LightGBM不可用，跳过训练LightGBM模型")
        
        # 训练CatBoost模型（如果可用）
        if CATBOOST_AVAILABLE:
            self.log("\n----- 训练CatBoost子模型 -----")
            cb_model = CatBoostModel(self.lottery_type, self.feature_window, self.log, self.use_gpu)
            cb_model.scalers = self.scalers  # 共享缩放器
            
            # 训练红球模型
            self.log("训练红球CatBoost模型...")
            red_cb = cb_model.train_catboost(X_train, red_train_data[0], 'red')
            self.sub_models['red']['catboost'] = red_cb
            
            # 训练蓝球模型
            self.log("训练蓝球CatBoost模型...")
            blue_cb = cb_model.train_catboost(X_train, blue_train_data[0], 'blue')
            self.sub_models['blue']['catboost'] = blue_cb
            
            # 评估CatBoost模型
            red_preds = red_cb.predict(X_test)
            red_accuracy = accuracy_score(red_test_data[0], red_preds)
            self.model_weights['red']['catboost'] = red_accuracy
            self.log(f"CatBoost红球模型准确率: {red_accuracy:.4f}")
            
            blue_preds = blue_cb.predict(X_test)
            blue_accuracy = accuracy_score(blue_test_data[0], blue_preds)
            self.model_weights['blue']['catboost'] = blue_accuracy
            self.log(f"CatBoost蓝球模型准确率: {blue_accuracy:.4f}")
        else:
            self.log("CatBoost不可用，跳过训练CatBoost模型")
        
        # 计算模型权重（使用softmax将准确率转换为权重）
        self.log("\n----- 计算模型权重 -----")
        
        # 计算红球模型权重
        red_weights = np.array(list(self.model_weights['red'].values()))
        if len(red_weights) > 0:
            # 使用softmax计算权重
            red_weights = np.exp(red_weights * 10) / np.sum(np.exp(red_weights * 10))  # 乘以10使差异更明显
            for i, model_name in enumerate(self.model_weights['red'].keys()):
                self.model_weights['red'][model_name] = float(red_weights[i])
                self.log(f"红球{model_name}模型权重: {red_weights[i]:.4f}")
        else:
            self.log("警告: 没有可用的红球模型权重")
            # 如果没有权重，使用均等权重
            for model_name in self.sub_models['red'].keys():
                self.model_weights['red'][model_name] = 1.0 / len(self.sub_models['red'])
        
        # 计算蓝球模型权重
        blue_weights = np.array(list(self.model_weights['blue'].values()))
        if len(blue_weights) > 0:
            # 使用softmax计算权重
            blue_weights = np.exp(blue_weights * 10) / np.sum(np.exp(blue_weights * 10))  # 乘以10使差异更明显
            for i, model_name in enumerate(self.model_weights['blue'].keys()):
                self.model_weights['blue'][model_name] = float(blue_weights[i])
                self.log(f"蓝球{model_name}模型权重: {blue_weights[i]:.4f}")
        else:
            self.log("警告: 没有可用的蓝球模型权重")
            # 如果没有权重，使用均等权重
            for model_name in self.sub_models['blue'].keys():
                self.model_weights['blue'][model_name] = 1.0 / len(self.sub_models['blue'])
        
        # 评估集成模型
        self.log("\n----- 评估集成模型 -----")
        
        # 使用加权投票进行预测
        red_votes = {}
        for model_name, model in self.sub_models['red'].items():
            preds = model.predict(X_test)
            weight = self.model_weights['red'][model_name]
            for i, pred in enumerate(preds):
                if i not in red_votes:
                    red_votes[i] = {}
                if pred not in red_votes[i]:
                    red_votes[i][pred] = 0
                red_votes[i][pred] += weight
        
        red_ensemble_preds = []
        for i in range(len(X_test)):
            if i in red_votes:
                # 选择得票最高的类别
                pred = max(red_votes[i].items(), key=lambda x: x[1])[0]
                red_ensemble_preds.append(pred)
            else:
                # 如果没有投票，随机选择
                red_ensemble_preds.append(np.random.randint(0, self.red_range))
        
        red_ensemble_accuracy = accuracy_score(red_test_data[0], red_ensemble_preds)
        self.log(f"集成模型红球准确率: {red_ensemble_accuracy:.4f}")
        
        blue_votes = {}
        for model_name, model in self.sub_models['blue'].items():
            preds = model.predict(X_test)
            weight = self.model_weights['blue'][model_name]
            for i, pred in enumerate(preds):
                if i not in blue_votes:
                    blue_votes[i] = {}
                if pred not in blue_votes[i]:
                    blue_votes[i][pred] = 0
                blue_votes[i][pred] += weight
        
        blue_ensemble_preds = []
        for i in range(len(X_test)):
            if i in blue_votes:
                # 选择得票最高的类别
                pred = max(blue_votes[i].items(), key=lambda x: x[1])[0]
                blue_ensemble_preds.append(pred)
            else:
                # 如果没有投票，随机选择
                blue_ensemble_preds.append(np.random.randint(0, self.blue_range))
        
        blue_ensemble_accuracy = accuracy_score(blue_test_data[0], blue_ensemble_preds)
        self.log(f"集成模型蓝球准确率: {blue_ensemble_accuracy:.4f}")
        
        # 将子模型存储到models字典中，以便保存
        self.models = {
            'red': self.sub_models['red'],
            'blue': self.sub_models['blue'],
            'weights': self.model_weights
        }
        
        return self.models
    
    def save_models(self):
        """
        保存模型、缩放器和模型权重
        """
        self.log("\n----- 保存集成模型和缩放器 -----")
        
        # 创建模型目录
        model_dir = os.path.join(self.models_dir, self.model_type)
        os.makedirs(model_dir, exist_ok=True)
        
        # 保存红球子模型
        if 'red' in self.models:
            red_dir = os.path.join(model_dir, 'red')
            os.makedirs(red_dir, exist_ok=True)
            
            for model_name, model in self.models['red'].items():
                model_path = os.path.join(red_dir, f'{model_name}_model.pkl')
                with open(model_path, 'wb') as f:
                    pickle.dump(model, f)
                self.log(f"红球{model_name}模型保存到: {model_path}")
        
        # 保存蓝球子模型
        if 'blue' in self.models:
            blue_dir = os.path.join(model_dir, 'blue')
            os.makedirs(blue_dir, exist_ok=True)
            
            for model_name, model in self.models['blue'].items():
                model_path = os.path.join(blue_dir, f'{model_name}_model.pkl')
                with open(model_path, 'wb') as f:
                    pickle.dump(model, f)
                self.log(f"蓝球{model_name}模型保存到: {model_path}")
        
        # 保存模型权重
        if 'weights' in self.models:
            weights_path = os.path.join(model_dir, 'model_weights.pkl')
            with open(weights_path, 'wb') as f:
                pickle.dump(self.models['weights'], f)
            self.log(f"模型权重保存到: {weights_path}")
        
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
        self.log(f"尝试加载{self.lottery_type}的集成模型...")
        
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
        
        # 初始化子模型和权重
        self.sub_models = {
            'red': {},
            'blue': {}
        }
        self.model_weights = {
            'red': {},
            'blue': {}
        }
        
        # 加载模型权重
        weights_path = os.path.join(model_dir, 'model_weights.pkl')
        if os.path.exists(weights_path):
            try:
                with open(weights_path, 'rb') as f:
                    self.model_weights = pickle.load(f)
                self.log(f"加载模型权重成功")
            except Exception as e:
                self.log(f"加载模型权重失败: {e}")
                # 如果加载失败，使用均等权重
                self.log("使用均等权重作为替代")
        else:
            self.log(f"警告: 模型权重文件不存在: {weights_path}")
            self.log("使用均等权重作为替代")
        
        # 加载红球子模型
        red_dir = os.path.join(model_dir, 'red')
        if os.path.exists(red_dir):
            for model_name in ['random_forest', 'xgboost', 'lightgbm', 'catboost']:
                model_path = os.path.join(red_dir, f'{model_name}_model.pkl')
                if os.path.exists(model_path):
                    try:
                        with open(model_path, 'rb') as f:
                            self.sub_models['red'][model_name] = pickle.load(f)
                        self.log(f"加载红球{model_name}模型成功")
                        
                        # 如果没有权重，设置均等权重
                        if 'red' not in self.model_weights or model_name not in self.model_weights['red']:
                            if 'red' not in self.model_weights:
                                self.model_weights['red'] = {}
                            self.model_weights['red'][model_name] = 1.0
                    except Exception as e:
                        self.log(f"加载红球{model_name}模型失败: {e}")
        else:
            self.log(f"警告: 红球模型目录不存在: {red_dir}")
        
        # 加载蓝球子模型
        blue_dir = os.path.join(model_dir, 'blue')
        if os.path.exists(blue_dir):
            for model_name in ['random_forest', 'xgboost', 'lightgbm', 'catboost']:
                model_path = os.path.join(blue_dir, f'{model_name}_model.pkl')
                if os.path.exists(model_path):
                    try:
                        with open(model_path, 'rb') as f:
                            self.sub_models['blue'][model_name] = pickle.load(f)
                        self.log(f"加载蓝球{model_name}模型成功")
                        
                        # 如果没有权重，设置均等权重
                        if 'blue' not in self.model_weights or model_name not in self.model_weights['blue']:
                            if 'blue' not in self.model_weights:
                                self.model_weights['blue'] = {}
                            self.model_weights['blue'][model_name] = 1.0
                    except Exception as e:
                        self.log(f"加载蓝球{model_name}模型失败: {e}")
        else:
            self.log(f"警告: 蓝球模型目录不存在: {blue_dir}")
        
        # 加载特征缩放器
        scaler_path = os.path.join(model_dir, 'scaler.pkl')
        if os.path.exists(scaler_path):
            try:
                with open(scaler_path, 'rb') as f:
                    self.scalers['X'] = pickle.load(f)
                self.log(f"加载特征缩放器成功")
            except Exception as e:
                self.log(f"加载特征缩放器失败: {e}")
                # 如果没有缩放器，创建一个默认的缩放器
                from sklearn.preprocessing import StandardScaler
                self.scalers['X'] = StandardScaler()
                self.log(f"创建了默认特征缩放器作为替代")
        else:
            self.log(f"警告: 特征缩放器文件不存在: {scaler_path}")
            # 如果没有缩放器，创建一个默认的缩放器
            from sklearn.preprocessing import StandardScaler
            self.scalers['X'] = StandardScaler()
            self.log(f"创建了默认特征缩放器作为替代")
        
        # 归一化权重
        for ball_type in ['red', 'blue']:
            if ball_type in self.model_weights and len(self.model_weights[ball_type]) > 0:
                total_weight = sum(self.model_weights[ball_type].values())
                if total_weight > 0:
                    for model_name in self.model_weights[ball_type]:
                        self.model_weights[ball_type][model_name] /= total_weight
                    self.log(f"归一化{ball_type}球模型权重成功")
                else:
                    # 如果权重和为0，使用均等权重
                    for model_name in self.model_weights[ball_type]:
                        self.model_weights[ball_type][model_name] = 1.0 / len(self.model_weights[ball_type])
                    self.log(f"警告: {ball_type}球模型权重和为0，使用均等权重")
        
        # 将子模型存储到models字典中，以便使用
        self.models = {
            'red': self.sub_models['red'],
            'blue': self.sub_models['blue'],
            'weights': self.model_weights
        }
        
        # 检查是否成功加载了至少一个红球和蓝球模型
        if len(self.sub_models['red']) > 0 and len(self.sub_models['blue']) > 0:
            self.log(f"集成模型加载成功，红球模型: {len(self.sub_models['red'])}个，蓝球模型: {len(self.sub_models['blue'])}个")
            return True
        else:
            self.log(f"集成模型加载失败，未找到必要的红球和蓝球模型")
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
        if 'red' not in self.models or 'blue' not in self.models or 'weights' not in self.models:
            # 尝试重新加载模型
            self.log("模型未加载，尝试重新加载...")
            load_success = self.load_models()
            if not load_success:
                self.log("错误：模型加载失败，请先训练模型")
                raise ValueError(f"模型未正确加载，请先训练或加载模型。")
        
        # 确保模型已加载
        if len(self.models['red']) == 0 or len(self.models['blue']) == 0:
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
            # 使用加权投票进行红球预测
            red_votes = {}
            for model_name, model in self.models['red'].items():
                pred = model.predict(X_scaled)[0]
                weight = self.model_weights['red'].get(model_name, 1.0)
                if pred not in red_votes:
                    red_votes[pred] = 0
                red_votes[pred] += weight
                self.log(f"红球{model_name}模型预测: {pred+1}，权重: {weight:.4f}")
            
            # 选择得票最高的类别
            if red_votes:
                red_pred = max(red_votes.items(), key=lambda x: x[1])[0]
                red_predictions = [int(red_pred) + 1]  # +1 转回原始号码范围
            else:
                # 如果没有投票，随机选择
                red_predictions = [np.random.randint(1, self.red_range + 1)]
            
            # 使用加权投票进行蓝球预测
            blue_votes = {}
            for model_name, model in self.models['blue'].items():
                pred = model.predict(X_scaled)[0]
                weight = self.model_weights['blue'].get(model_name, 1.0)
                if pred not in blue_votes:
                    blue_votes[pred] = 0
                blue_votes[pred] += weight
                self.log(f"蓝球{model_name}模型预测: {pred+1}，权重: {weight:.4f}")
            
            # 选择得票最高的类别
            if blue_votes:
                blue_pred = max(blue_votes.items(), key=lambda x: x[1])[0]
                blue_predictions = [int(blue_pred) + 1]  # +1 转回原始号码范围
            else:
                # 如果没有投票，随机选择
                blue_predictions = [np.random.randint(1, self.blue_range + 1)]
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
        
        # 有5%的概率完全随机选择蓝球
        if np.random.random() < 0.05:
            self.log("随机选择蓝球（5%概率）")
            blue_predictions = []
            while len(blue_predictions) < self.blue_count:
                new_num = np.random.randint(1, self.blue_range + 1)
                if new_num not in blue_predictions:
                    blue_predictions.append(new_num)
            blue_predictions = sorted(blue_predictions)
        
        return red_predictions, blue_predictions