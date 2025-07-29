# -*- coding:utf-8 -*-
"""
LightGBM model implementation for lottery prediction
"""

import os
import numpy as np
import pandas as pd
import pickle
import joblib
import json
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, KFold
from sklearn.metrics import accuracy_score
from collections import Counter

# 条件导入LightGBM
try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False

from .base import BaseMLModel


class WrappedLightGBMModel:
    """
    LightGBM模型包装器，用于统一预测接口
    """
    def __init__(self, model, processor=None):
        self.model = model
        self.processor = processor
        
    def predict(self, X):
        if self.processor:
            X = self.processor(X)
        return self.model.predict(X)


class LightGBMModel(BaseMLModel):
    """
    LightGBM模型实现
    """
    
    def __init__(self, lottery_type='dlt', feature_window=10, log_callback=None, use_gpu=False):
        """
        初始化LightGBM模型
        
        Args:
            lottery_type: 彩票类型，'dlt'或'ssq'
            feature_window: 特征窗口大小，使用多少期数据作为特征
            log_callback: 日志回调函数，用于将日志发送到UI
            use_gpu: 是否使用GPU训练
        """
        super().__init__(lottery_type, feature_window, log_callback, use_gpu)
        self.model_type = 'lightgbm'
        
        # 检查LightGBM是否可用
        if not LIGHTGBM_AVAILABLE:
            self.log("警告: LightGBM未安装或不可用，无法使用LightGBM模型")
    
    def train(self, df):
        """
        训练LightGBM模型
        
        Args:
            df: 包含历史开奖数据的DataFrame
            
        Returns:
            训练好的模型
        """
        if not LIGHTGBM_AVAILABLE:
            self.log("错误: LightGBM未安装或不可用，无法训练LightGBM模型")
            raise ImportError("LightGBM未安装或不可用，请先安装LightGBM")
            
        self.log("\n----- 开始训练LightGBM模型 -----")
        
        # 准备数据
        X_train, X_test, red_train_data, red_test_data, blue_train_data, blue_test_data = self.prepare_data(df)
        
        # 训练红球模型 - 为每个位置训练独立的模型
        self.log("训练红球LightGBM模型...")
        red_models = []
        for i in range(len(red_train_data)):
            self.log(f"训练红球第{i+1}个位置的模型...")
            model = self.train_lightgbm(X_train, red_train_data[i], f'red_{i+1}')
            red_models.append(model)
        self.models['red'] = red_models
        
        # 训练蓝球模型 - 为每个位置训练独立的模型
        self.log("训练蓝球LightGBM模型...")
        blue_models = []
        for i in range(len(blue_train_data)):
            self.log(f"训练蓝球第{i+1}个位置的模型...")
            model = self.train_lightgbm(X_train, blue_train_data[i], f'blue_{i+1}')
            blue_models.append(model)
        self.models['blue'] = blue_models
        
        # 评估模型
        self.evaluate_multi_position(X_test, red_test_data, blue_test_data)
        
        # 保存模型
        self.save_models()
        
        return self.models
    
    def train_lightgbm(self, X_train, y_train, ball_type):
        """
        训练LightGBM模型，使用交叉验证和超参数调优
        
        Args:
            X_train: 训练特征
            y_train: 训练标签
            ball_type: 球类型，'red'或'blue'
            
        Returns:
            训练好的LightGBM模型
        """
        self.log(f"训练{ball_type}球LightGBM模型...")
        self.log(f"数据维度: 特征={X_train.shape}, 标签={y_train.shape}")
        
        # 确保y_train是整数类型
        y_train = y_train.astype(int)
        
        # 检查数据质量
        unique_labels = np.unique(y_train)
        self.log(f"{ball_type}球标签唯一值数量: {len(unique_labels)}, 范围: {unique_labels.min()} - {unique_labels.max()}")
        
        # 检查是否有类别样本数量过少的情况
        class_counts = Counter(y_train)
        min_samples = min(class_counts.values())
        self.log(f"最少样本数的类别有{min_samples}个样本")
        
        # 如果数据质量不佳，进行数据增强
        if min_samples < 5 or len(unique_labels) < 3:
            self.log(f"警告: 数据质量不佳，进行数据增强处理")
            X_train, y_train = self._augment_data(X_train, y_train, ball_type)
            class_counts = Counter(y_train)
            min_samples = min(class_counts.values())
            self.log(f"数据增强后: 样本数={len(y_train)}, 最少样本数={min_samples}")
        
        # 设置交叉验证策略
        if min_samples < 2:
            self.log(f"警告: 某些类别样本数量过少(最少{min_samples}个)，使用KFold代替StratifiedKFold")
            cv = KFold(n_splits=min(3, len(y_train)//10), shuffle=True, random_state=42)
        else:
            cv = StratifiedKFold(n_splits=min(5, min_samples), shuffle=True, random_state=42)
        
        # 优化的超参数搜索空间，避免过拟合
        param_grid = {
            'n_estimators': [50, 100, 200],
            'max_depth': [3, 5, 7],
            'learning_rate': [0.05, 0.1, 0.15],
            'subsample': [0.8, 0.9, 1.0],
            'colsample_bytree': [0.8, 0.9, 1.0],
            'min_child_samples': [10, 20, 30],
            'min_child_weight': [1e-3, 1e-2, 1e-1],
            'reg_alpha': [0, 0.1, 0.3],
            'reg_lambda': [0, 0.1, 0.3],
            'num_leaves': [15, 31, 63],
            'min_split_gain': [0.0, 0.1, 0.2]
        }
        
        # 创建LightGBM分类器，添加更多防止过拟合的参数
        base_params = {
            'random_state': 42,
            'objective': 'multiclass',
            'metric': 'multi_logloss',
            'verbosity': -1,
            'force_col_wise': True,
            'min_data_in_leaf': max(5, min_samples // 10),
            'feature_fraction': 0.8,
            'bagging_fraction': 0.8,
            'bagging_freq': 5
        }
        
        if self.use_gpu:
            self.log("使用GPU训练LightGBM模型")
            base_params['device'] = 'gpu'
            base_params['gpu_platform_id'] = 0
            base_params['gpu_device_id'] = 0
        
        lgb_model = lgb.LGBMClassifier(**base_params)
        
        # 使用随机搜索进行超参数调优
        random_search = RandomizedSearchCV(
            estimator=lgb_model,
            param_distributions=param_grid,
            n_iter=15,  # 减少搜索次数
            scoring='accuracy',
            cv=cv,
            verbose=0,
            random_state=42,
            n_jobs=1 if self.use_gpu else -1  # GPU模式下使用单进程
        )
        
        # 训练模型
        try:
            random_search.fit(X_train, y_train)
            best_params = random_search.best_params_
            best_score = random_search.best_score_
            self.log(f"最佳参数: {best_params}")
            self.log(f"交叉验证最佳得分: {best_score:.4f}")
        except Exception as e:
            self.log(f"超参数搜索失败: {e}，使用默认参数")
            best_params = {}
            best_score = 0.0
        
        # 使用最佳参数训练最终模型
        final_params = base_params.copy()
        final_params.update(best_params)
        
        final_model = lgb.LGBMClassifier(**final_params)
        
        # 创建验证集
        from sklearn.model_selection import train_test_split
        try:
            X_train_final, X_val, y_train_final, y_val = train_test_split(
                X_train, y_train, test_size=0.2, random_state=42, 
                stratify=y_train if min_samples >= 2 else None
            )
        except ValueError:
            # 如果分层失败，使用普通分割
            X_train_final, X_val, y_train_final, y_val = train_test_split(
                X_train, y_train, test_size=0.2, random_state=42
            )
        
        # 训练最终模型，使用早停
        try:
            final_model.fit(
                X_train_final, y_train_final,
                eval_set=[(X_val, y_val)],
                eval_metric='multi_logloss',
                callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)]
            )
        except Exception as e:
            self.log(f"早停训练失败: {e}，使用普通训练")
            final_model.fit(X_train_final, y_train_final)
        
        # 评估验证集准确率
        val_preds = final_model.predict(X_val)
        val_accuracy = accuracy_score(y_val, val_preds)
        self.log(f"验证集准确率: {val_accuracy:.4f}")
        
        # 记录特征重要性
        if hasattr(final_model, 'feature_importances_'):
            feature_importances = final_model.feature_importances_
            top_n = min(10, len(feature_importances))  # 显示前10个或所有特征
            indices = np.argsort(feature_importances)[-top_n:]
            self.log(f"前{top_n}个最重要特征的重要性:")
            for i in indices:
                self.log(f"特征 {i}: {feature_importances[i]:.4f}")
        
        # 包装模型以统一接口
        wrapped_model = WrappedLightGBMModel(final_model)
        
        return wrapped_model
    
    def _augment_data(self, X_train, y_train, ball_type):
        """
        数据增强，解决样本不足问题
        
        Args:
            X_train: 训练特征
            y_train: 训练标签
            ball_type: 球类型
            
        Returns:
            增强后的特征和标签
        """
        self.log(f"对{ball_type}球数据进行增强...")
        
        # 计算每个类别的样本数
        class_counts = Counter(y_train)
        max_samples = max(class_counts.values())
        min_samples_needed = max(10, max_samples // 2)  # 每个类别至少需要10个样本
        
        X_augmented = [X_train]
        y_augmented = [y_train]
        
        for label, count in class_counts.items():
            if count < min_samples_needed:
                # 找到该类别的所有样本
                label_indices = np.where(y_train == label)[0]
                label_samples = X_train[label_indices]
                
                # 需要生成的样本数
                samples_to_generate = min_samples_needed - count
                
                # 使用噪声增强
                for _ in range(samples_to_generate):
                    # 随机选择一个该类别的样本
                    base_sample = label_samples[np.random.randint(len(label_samples))]
                    # 添加小量噪声
                    noise = np.random.normal(0, 0.01, base_sample.shape)
                    augmented_sample = base_sample + noise
                    
                    X_augmented.append(augmented_sample.reshape(1, -1))
                    y_augmented.append([label])
        
        # 合并所有数据
        X_final = np.vstack(X_augmented)
        y_final = np.hstack(y_augmented)
        
        # 打乱数据
        indices = np.random.permutation(len(X_final))
        X_final = X_final[indices]
        y_final = y_final[indices]
        
        return X_final, y_final
    
    def evaluate_multi_position(self, X_test, red_test_data, blue_test_data):
        """
        评估多位置模型性能
        
        Args:
            X_test: 测试特征
            red_test_data: 红球测试标签列表
            blue_test_data: 蓝球测试标签列表
            
        Returns:
            红球和蓝球的平均准确率
        """
        self.log("评估多位置模型性能...")
        
        red_accuracies = []
        blue_accuracies = []
        
        # 评估红球模型
        if 'red' in self.models and isinstance(self.models['red'], list):
            for i, model in enumerate(self.models['red']):
                if i < len(red_test_data):
                    y_test = red_test_data[i]
                    preds = model.predict(X_test)
                    accuracy = accuracy_score(y_test, preds)
                    red_accuracies.append(accuracy)
                    self.log(f"红球第{i+1}位模型准确率: {accuracy:.4f}")
        
        # 评估蓝球模型
        if 'blue' in self.models and isinstance(self.models['blue'], list):
            for i, model in enumerate(self.models['blue']):
                if i < len(blue_test_data):
                    y_test = blue_test_data[i]
                    preds = model.predict(X_test)
                    accuracy = accuracy_score(y_test, preds)
                    blue_accuracies.append(accuracy)
                    self.log(f"蓝球第{i+1}位模型准确率: {accuracy:.4f}")
        
        # 计算平均准确率
        avg_red_accuracy = np.mean(red_accuracies) if red_accuracies else 0
        avg_blue_accuracy = np.mean(blue_accuracies) if blue_accuracies else 0
        
        self.log(f"红球平均准确率: {avg_red_accuracy:.4f}")
        self.log(f"蓝球平均准确率: {avg_blue_accuracy:.4f}")
        
        return avg_red_accuracy, avg_blue_accuracy
    
    def evaluate(self, X_test, y_red_test, y_blue_test):
        """
        评估模型性能（兼容旧接口）
        
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
        
        # 处理多位置模型
        if 'red' in self.models:
            if isinstance(self.models['red'], list):
                # 多位置模型，只评估第一个位置
                if len(self.models['red']) > 0:
                    red_preds = self.models['red'][0].predict(X_test)
                    red_accuracy = accuracy_score(y_red_test[:, 0] if y_red_test.shape[1] > 1 else y_red_test.flatten(), red_preds)
            else:
                # 单一模型
                red_preds = self.models['red'].predict(X_test)
                red_accuracy = accuracy_score(y_red_test, red_preds.reshape(-1, 1))
            self.log(f"红球模型准确率: {red_accuracy:.4f}")
        
        if 'blue' in self.models:
            if isinstance(self.models['blue'], list):
                # 多位置模型，只评估第一个位置
                if len(self.models['blue']) > 0:
                    blue_preds = self.models['blue'][0].predict(X_test)
                    blue_accuracy = accuracy_score(y_blue_test[:, 0] if y_blue_test.shape[1] > 1 else y_blue_test.flatten(), blue_preds)
            else:
                # 单一模型
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
            if isinstance(self.models['red'], list):
                # 多位置模型
                for i, model in enumerate(self.models['red']):
                    model_path = os.path.join(model_dir, f'red_model_{i+1}.pkl')
                    with open(model_path, 'wb') as f:
                        pickle.dump(model, f)
                    self.log(f"红球第{i+1}位模型保存到: {model_path}")
            else:
                # 单一模型
                model_path = os.path.join(model_dir, 'red_model.pkl')
                with open(model_path, 'wb') as f:
                    pickle.dump(self.models['red'], f)
                self.log(f"红球模型保存到: {model_path}")
        
        # 保存蓝球模型
        if 'blue' in self.models:
            if isinstance(self.models['blue'], list):
                # 多位置模型
                for i, model in enumerate(self.models['blue']):
                    model_path = os.path.join(model_dir, f'blue_model_{i+1}.pkl')
                    with open(model_path, 'wb') as f:
                        pickle.dump(model, f)
                    self.log(f"蓝球第{i+1}位模型保存到: {model_path}")
            else:
                # 单一模型
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
            'multi_position': True,  # 标记为多位置模型
            'red_positions': len(self.models['red']) if 'red' in self.models and isinstance(self.models['red'], list) else 1,
            'blue_positions': len(self.models['blue']) if 'blue' in self.models and isinstance(self.models['blue'], list) else 1
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
        if not LIGHTGBM_AVAILABLE:
            self.log("错误: LightGBM未安装或不可用，无法加载LightGBM模型")
            return False
            
        self.log(f"尝试加载{self.lottery_type}的LightGBM模型...")
        
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
        
        # 检查是否为多位置模型
        is_multi_position = model_info.get('multi_position', False)
        red_positions = model_info.get('red_positions', 1)
        blue_positions = model_info.get('blue_positions', 1)
        
        # 加载模型和缩放器
        models_loaded = True
        balls_loaded = 0
        
        # 加载红球模型
        try:
            if is_multi_position and red_positions > 1:
                # 多位置模型
                red_models = []
                for i in range(red_positions):
                    model_path = os.path.join(model_dir, f'red_model_{i+1}.pkl')
                    if os.path.exists(model_path):
                        with open(model_path, 'rb') as f:
                            model = pickle.load(f)
                        red_models.append(model)
                        self.log(f"加载红球第{i+1}位模型成功")
                    else:
                        self.log(f"警告: 红球第{i+1}位模型文件不存在: {model_path}")
                        models_loaded = False
                        break
                if red_models:
                    self.models['red'] = red_models
                    balls_loaded += 1
            else:
                # 单一模型
                model_path = os.path.join(model_dir, 'red_model.pkl')
                if os.path.exists(model_path):
                    with open(model_path, 'rb') as f:
                        self.models['red'] = pickle.load(f)
                    self.log(f"加载红球模型成功")
                    balls_loaded += 1
                else:
                    self.log(f"警告: 红球模型文件不存在: {model_path}")
                    models_loaded = False
        except Exception as e:
            self.log(f"加载红球模型失败: {e}")
            models_loaded = False
        
        # 加载蓝球模型
        try:
            if is_multi_position and blue_positions > 1:
                # 多位置模型
                blue_models = []
                for i in range(blue_positions):
                    model_path = os.path.join(model_dir, f'blue_model_{i+1}.pkl')
                    if os.path.exists(model_path):
                        with open(model_path, 'rb') as f:
                            model = pickle.load(f)
                        blue_models.append(model)
                        self.log(f"加载蓝球第{i+1}位模型成功")
                    else:
                        self.log(f"警告: 蓝球第{i+1}位模型文件不存在: {model_path}")
                        models_loaded = False
                        break
                if blue_models:
                    self.models['blue'] = blue_models
                    balls_loaded += 1
            else:
                # 单一模型
                model_path = os.path.join(model_dir, 'blue_model.pkl')
                if os.path.exists(model_path):
                    with open(model_path, 'rb') as f:
                        self.models['blue'] = pickle.load(f)
                    self.log(f"加载蓝球模型成功")
                    balls_loaded += 1
                else:
                    self.log(f"警告: 蓝球模型文件不存在: {model_path}")
                    models_loaded = False
        except Exception as e:
            self.log(f"加载蓝球模型失败: {e}")
            models_loaded = False
        
        # 加载特征缩放器
        try:
            scaler_path = os.path.join(model_dir, 'scaler.pkl')
            if os.path.exists(scaler_path):
                with open(scaler_path, 'rb') as f:
                    self.scalers['X'] = pickle.load(f)
                self.log(f"加载特征缩放器成功")
            else:
                self.log(f"警告: 特征缩放器文件不存在")
                # 如果没有缩放器，创建一个默认的缩放器
                from sklearn.preprocessing import StandardScaler
                self.scalers['X'] = StandardScaler()
                self.log(f"创建了默认特征缩放器作为替代")
        except Exception as e:
            self.log(f"加载特征缩放器失败: {e}")
            # 创建默认缩放器
            from sklearn.preprocessing import StandardScaler
            self.scalers['X'] = StandardScaler()
            self.log(f"创建了默认特征缩放器作为替代")
        
        # 如果所有模型都成功加载，返回True
        if models_loaded:
            self.log(f"LightGBM模型加载成功")
            return True
        elif balls_loaded >= 2:  # 至少加载了红球和蓝球
            # 即使有警告，只要基础模型存在，我们也认为模型可用
            self.log(f"LightGBM模型加载成功，但有一些警告")
            return True
        else:
            self.log(f"LightGBM模型加载失败，未找到必要的红球和蓝球模型")
            return False
    
    def predict(self, recent_data):
        """
        生成预测结果
        
        Args:
            recent_data: 包含最近开奖数据的DataFrame
            
        Returns:
            预测的红球和蓝球号码
        """
        if not LIGHTGBM_AVAILABLE:
            self.log("错误: LightGBM未安装或不可用，无法使用LightGBM模型进行预测")
            raise ImportError("LightGBM未安装或不可用，请先安装LightGBM")
            
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
            red_predictions = []
            blue_predictions = []
            
            # 预测红球
            if 'red' in self.models:
                if isinstance(self.models['red'], list):
                    # 多位置模型
                    for i, model in enumerate(self.models['red']):
                        pred = model.predict(X_scaled)[0]
                        red_num = int(pred) + 1  # +1 转回原始号码范围
                        # 确保号码在有效范围内
                        red_num = max(1, min(red_num, self.red_range))
                        red_predictions.append(red_num)
                else:
                    # 单一模型
                    red_pred = self.models['red'].predict(X_scaled)[0]
                    red_predictions = [int(red_pred) + 1]  # +1 转回原始号码范围
            
            # 预测蓝球
            if 'blue' in self.models:
                if isinstance(self.models['blue'], list):
                    # 多位置模型
                    for i, model in enumerate(self.models['blue']):
                        pred = model.predict(X_scaled)[0]
                        blue_num = int(pred) + 1  # +1 转回原始号码范围
                        # 确保号码在有效范围内
                        blue_num = max(1, min(blue_num, self.blue_range))
                        blue_predictions.append(blue_num)
                else:
                    # 单一模型
                    blue_pred = self.models['blue'].predict(X_scaled)[0]
                    blue_predictions = [int(blue_pred) + 1]  # +1 转回原始号码范围
                    
        except Exception as e:
            self.log(f"预测过程中出错: {e}")
            import traceback
            self.log(traceback.format_exc())
            raise ValueError(f"预测过程中出错: {e}")
        
        # 处理重复号码和数量不足的情况
        red_predictions = self._process_predictions(red_predictions, self.red_count, self.red_range, '红球')
        blue_predictions = self._process_predictions(blue_predictions, self.blue_count, self.blue_range, '蓝球')
        
        return red_predictions, blue_predictions
    
    def _process_predictions(self, predictions, required_count, number_range, ball_type):
        """
        处理预测结果，确保号码不重复且数量正确
        
        Args:
            predictions: 原始预测结果
            required_count: 需要的号码数量
            number_range: 号码范围
            ball_type: 球类型（用于日志）
            
        Returns:
            处理后的预测结果
        """
        # 去除重复号码
        unique_predictions = list(dict.fromkeys(predictions))  # 保持顺序的去重
        
        # 如果预测的号码数量不足，智能补充
        while len(unique_predictions) < required_count:
            # 优先使用统计学方法补充
            candidates = list(range(1, number_range + 1))
            # 移除已有的号码
            available = [num for num in candidates if num not in unique_predictions]
            
            if available:
                # 使用加权随机选择，偏向中间范围的号码
                weights = []
                for num in available:
                    # 中间号码权重更高
                    distance_from_center = abs(num - (number_range + 1) / 2)
                    weight = 1.0 / (1.0 + distance_from_center * 0.1)
                    weights.append(weight)
                
                # 归一化权重
                total_weight = sum(weights)
                weights = [w / total_weight for w in weights]
                
                # 随机选择一个号码
                selected = np.random.choice(available, p=weights)
                unique_predictions.append(selected)
            else:
                # 如果没有可用号码（理论上不应该发生），随机选择
                new_num = np.random.randint(1, number_range + 1)
                if new_num not in unique_predictions:
                    unique_predictions.append(new_num)
        
        # 确保数量正确
        final_predictions = unique_predictions[:required_count]
        
        # 对于红球，需要排序；对于蓝球，保持预测顺序
        if ball_type == '红球':
            final_predictions = sorted(final_predictions)
        
        self.log(f"{ball_type}预测结果: {final_predictions}")
        
        return final_predictions
        
        return red_predictions, blue_predictions