# -*- coding:utf-8 -*-
"""
Machine Learning Models for Lottery Prediction
Author: Yang Zhao

多种机器学习模型支持模块
包含XGBoost、随机森林等多种预测模型以及集成学习功能
"""
import os
import time
import torch
import numpy as np
import pandas as pd
import pickle
import joblib
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error
import xgboost as xgb
import logging
import json

# 尝试导入可选的模型库
try:
    from lightgbm import LGBMRegressor
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False

try:
    from catboost import CatBoostRegressor
    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False

# 导入期望值模型
try:
    from expected_value_model import ExpectedValueLotteryModel
    EXPECTED_VALUE_MODEL_AVAILABLE = True
except ImportError:
    EXPECTED_VALUE_MODEL_AVAILABLE = False

# 定义支持的模型类型
MODEL_TYPES = {
    'random_forest': '随机森林',
    'xgboost': 'XGBoost',
    'gbdt': '梯度提升树',
    'ensemble': '集成模型'
}

# 如果可选库可用，添加到支持的模型中
if LIGHTGBM_AVAILABLE:
    MODEL_TYPES['lightgbm'] = 'LightGBM'
if CATBOOST_AVAILABLE:
    MODEL_TYPES['catboost'] = 'CatBoost'
if EXPECTED_VALUE_MODEL_AVAILABLE:
    MODEL_TYPES['expected_value'] = '期望值模型'

class WrappedXGBoostModel:
    def __init__(self, model, processor):
        self.model = model
        self.process_prediction = processor
        
    def predict(self, data):
        if not isinstance(data, xgb.DMatrix):
            data = xgb.DMatrix(data)
        raw_preds = self.model.predict(data)
        return self.process_prediction(raw_preds)

class WrappedGBDTModel:
    def __init__(self, model, processor):
        self.model = model
        self.process_prediction = processor
        
    def predict(self, data):
        raw_preds = self.model.predict_proba(data)
        return self.process_prediction(raw_preds)

class WrappedLightGBMModel:
    def __init__(self, model, processor):
        self.model = model
        self.process_prediction = processor
        
    def predict(self, data):
        raw_preds = self.model.predict(data)
        return self.process_prediction(raw_preds)

class WrappedCatBoostModel:
    def __init__(self, model, processor):
        self.model = model
        self.process_prediction = processor
        
    def predict(self, data):
        raw_preds = self.model.predict(data)
        return self.process_prediction(raw_preds)

class LotteryMLModels:
    """彩票预测机器学习模型类"""
    
    def __init__(self, lottery_type='dlt', model_type='ensemble', feature_window=10, log_callback=None, use_gpu=False):
        """
        初始化模型
        
        Args:
            lottery_type: 彩票类型，'dlt'或'ssq'
            model_type: 模型类型，可选值见MODEL_TYPES
            feature_window: 特征窗口大小，使用多少期数据作为特征
            log_callback: 日志回调函数，用于将日志发送到UI
            use_gpu: 是否使用GPU训练
        """
        self.lottery_type = lottery_type
        self.model_type = model_type
        self.feature_window = feature_window
        self.models = {}
        self.scalers = {}
        self.feature_cols = []
        self.log_callback = log_callback
        self.use_gpu = use_gpu
        
        # 初始化模型权重字典
        self.model_weights = {}
        
        self.raw_models = {}
        
   
        self.logger = logging.getLogger(f"ml_models_{lottery_type}")
        self.logger.setLevel(logging.INFO)
   
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
        
     
        if self.use_gpu:
            cuda_available = torch.cuda.is_available()
            mps_available = hasattr(torch, 'mps') and torch.backends.mps.is_available()
            
            if cuda_available:
                self.log(f"ML模型使用CUDA GPU: {torch.cuda.get_device_name(0)}")
            elif mps_available:
                self.log("ML模型使用Apple M系列芯片GPU (MPS)")
            else:
                self.log("GPU不可用，ML模型使用CPU")
                self.use_gpu = False
        else:
            self.log("ML模型使用CPU")
        
    
        if lottery_type == 'dlt':
        
            self.red_range = 35
            self.blue_range = 12
            self.red_count = 5
            self.blue_count = 2
        else:  
     
            self.red_range = 33
            self.blue_range = 16
            self.red_count = 6
            self.blue_count = 1
        
   
        self.models_dir = os.path.join(f'./model/{lottery_type}')
        os.makedirs(self.models_dir, exist_ok=True)
    
    def log(self, message):
        """记录日志并发送到UI（如果有回调）"""
        self.logger.info(message)
        if self.log_callback:
            self.log_callback(message)
    
    def prepare_data(self, df, test_size=0.2):
        """准备训练数据"""
    
        self.log("准备训练数据...")
        
       
        window_size = self.feature_window
        
   
        df = df.sort_values('期数').reset_index(drop=True)
        
   
        if self.lottery_type == 'dlt':
            red_cols = [col for col in df.columns if col.startswith('红球_')][:5]
            blue_cols = [col for col in df.columns if col.startswith('蓝球_')][:2]
        else:  # ssq
            red_cols = [col for col in df.columns if col.startswith('红球_')][:6]
  
            blue_cols = []
            for col in df.columns:
                if col.startswith('蓝球_') or col == '蓝球':
                    blue_cols.append(col)
                    if len(blue_cols) >= 1:  # 双色球只有1个蓝球
                        break
        
        self.feature_cols = red_cols + blue_cols
        
     
        X_data = []
        y_red = []
        y_blue = []
        
        for i in range(len(df) - window_size):

            features = []
            for j in range(window_size):
                row_features = []
                for col in red_cols + blue_cols:
                    row_features.append(df.iloc[i + j][col])
                features.append(row_features)
            
            # 添加特征
            X_data.append(features)
            
            # 添加标签
            red_labels = []
            for col in red_cols:
                red_labels.append(df.iloc[i + window_size][col] - 1)  # 减1使号码从0开始，适合分类模型
            y_red.append(red_labels)
            
            # 添加蓝球标签
            blue_labels = []
            for col in blue_cols:
                blue_labels.append(df.iloc[i + window_size][col] - 1)  # 减1使号码从0开始，适合分类模型
            y_blue.append(blue_labels)
        
        # 转换为NumPy数组
        X = np.array(X_data)
        y_red = np.array(y_red)
        y_blue = np.array(y_blue)
        
        # 记录数据形状
        self.log(f"特征形状: {X.shape}")
        self.log(f"红球标签形状: {y_red.shape}")
        self.log(f"蓝球标签形状: {y_blue.shape}")
        
        # 将3D特征展平为2D，以便用于传统ML模型
        X_reshaped = X.reshape(X.shape[0], -1)
        
        # 保存特征维度，用于预测时的兼容性检查
        self.expected_feature_count = X_reshaped.shape[1]
        self.log(f"特征维度: {self.expected_feature_count}")
        
        # 数据标准化，仅对输入特征进行处理，不对标签进行处理
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_reshaped)
        
        # 保存缩放器以便预测时使用
        self.scalers['X'] = scaler
        
        # 拆分训练集和测试集
        # 检查是否有类别样本数量过少的情况
        from collections import Counter
        if y_red.ndim > 1:
            # 对于多个红球，检查第一个位置的分布
            red_class_counts = Counter(y_red[:, 0])
        else:
            red_class_counts = Counter(y_red)
        min_red_samples = min(red_class_counts.values())
        
        if min_red_samples < 2:
            self.log(f"警告: 红球数据中某些类别样本数量过少(最少{min_red_samples}个)，使用随机拆分")
            # 直接使用随机拆分，不使用分层抽样
            X_train, X_test, y_red_train, y_red_test, y_blue_train, y_blue_test = train_test_split(
                X_scaled, y_red, y_blue, test_size=test_size, random_state=42, stratify=None
            )
        else:
            try:
                # 尝试使用默认拆分
                X_train, X_test, y_red_train, y_red_test, y_blue_train, y_blue_test = train_test_split(
                    X_scaled, y_red, y_blue, test_size=test_size, random_state=42
                )
            except ValueError as e:
                self.log(f"警告: 数据拆分失败，原因: {str(e)}")
                # 如果失败，尝试不使用stratify参数
                X_train, X_test, y_red_train, y_red_test, y_blue_train, y_blue_test = train_test_split(
                    X_scaled, y_red, y_blue, test_size=test_size, random_state=42, stratify=None
                )
        
        # 对于ML模型，需要将多维标签展平为单维
        if y_red_train.ndim > 1:
            # 对于多个红球，随机选择一个位置进行预测
            red_pos = np.random.randint(0, y_red_train.shape[1])
            self.red_pos = red_pos
            y_red_class = y_red_train[:, red_pos]
        else:
            y_red_class = y_red_train
        
        # 对于蓝球标签，检查维度并确保安全访问
        if y_blue_train.shape[1] > 0:  # 确保数组有列可以访问
            # 对于多个蓝球，随机选择一个位置进行预测
            blue_pos = np.random.randint(0, y_blue_train.shape[1])
            self.blue_pos = blue_pos
            y_blue_class = y_blue_train[:, blue_pos]
        else:
            # 处理双色球可能没有蓝球的情况
            self.log("警告: 蓝球标签维度为0, 使用默认值0")
            y_blue_class = np.zeros(y_blue_train.shape[0], dtype=int)
            self.blue_pos = 0
        
        self.log(f"训练集特征形状: {X_train.shape}")
        
        return X_train, X_test, y_red_class, y_red_test[:, self.red_pos], y_blue_class, y_blue_test[:, self.blue_pos] if y_blue_test.shape[1] > 0 else np.zeros(y_blue_test.shape[0], dtype=int)
    
    def train_random_forest(self, X_train, y_train, ball_type, n_estimators=100):
        """训练随机森林模型，使用交叉验证和超参数调优"""
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
        from scipy.stats import randint, uniform
        
        self.log(f"训练{ball_type}球随机森林模型...")
        self.log(f"数据维度: 特征={X_train.shape}, 标签={y_train.shape}")
        
        # 创建基础模型
        base_model = RandomForestClassifier(
            random_state=42,
            n_jobs=-1,
            class_weight='balanced',  # 处理类别不平衡问题
            bootstrap=True,  # 使用bootstrap样本
            oob_score=True,  # 使用袋外样本评估模型
            verbose=0  # 减少输出
        )
        
        # 设置超参数搜索空间
        param_dist = {
            'n_estimators': randint(100, 300),
            'max_depth': [None, 10, 15, 20, 25],
            'min_samples_split': randint(2, 11),
            'min_samples_leaf': randint(1, 5),
            'max_features': ['sqrt', 'log2', None]
        }
        
        # 检查是否有类别样本数量过少的情况
        from collections import Counter
        class_counts = Counter(y_train)
        min_samples = min(class_counts.values())
        
        # 创建交叉验证对象
        if min_samples < 2:
            self.log(f"警告: 某些类别样本数量过少(最少{min_samples}个)，使用普通KFold代替StratifiedKFold")
            from sklearn.model_selection import KFold
            cv = KFold(n_splits=5, shuffle=True, random_state=42)
        else:
            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        
        # 创建随机搜索对象
        self.log("开始随机森林超参数随机搜索...")
        random_search = RandomizedSearchCV(
            estimator=base_model,
            param_distributions=param_dist,
            n_iter=20,  # 搜索20组参数
            cv=cv,
            scoring='accuracy',
            verbose=0,
            n_jobs=-1,
            random_state=42
        )
        
        try:
            # 执行随机搜索
            random_search.fit(X_train, y_train)
        except ValueError as e:
            if "The least populated class in y has only 1 member" in str(e):
                self.log(f"警告: 随机森林参数搜索失败，使用默认参数。原因: {str(e)}")
                # 使用默认参数创建模型
                model = RandomForestClassifier(
                    n_estimators=100,
                    max_depth=None,
                    min_samples_split=2,
                    min_samples_leaf=1,
                    max_features='sqrt',
                    random_state=42,
                    n_jobs=-1,
                    class_weight='balanced'
                )
                model.fit(X_train, y_train)
                self.log(f"{ball_type}球随机森林模型训练完成(使用默认参数)")
                return model
            else:
                raise
        
        # 获取最佳参数和模型
        best_params = random_search.best_params_
        best_score = random_search.best_score_
        model = random_search.best_estimator_
        
        self.log(f"随机森林最佳参数: {best_params}")
        self.log(f"随机森林交叉验证最佳得分: {best_score:.4f}")
        
        # 如果有袋外分数，输出它
        if hasattr(model, 'oob_score_'):
            self.log(f"随机森林袋外分数: {model.oob_score_:.4f}")
        
        # 输出特征重要性
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
            indices = np.argsort(importances)[::-1]
            
            self.log(f"随机森林特征重要性 (前10个):")
            max_features = min(10, len(indices))
            for i in range(max_features):
                feature_idx = indices[i]
                feature_name = f"特征_{feature_idx}" if feature_idx >= len(self.feature_cols) else self.feature_cols[feature_idx]
                self.log(f"  {i+1}. {feature_name}: {importances[feature_idx]:.4f}")
        
        self.log(f"{ball_type}球随机森林模型训练完成")
        return model
    
    # 添加处理多维预测的静态方法，使其可序列化
    @staticmethod
    def process_multidim_prediction(raw_preds):
        """处理多维预测结果，返回类别索引"""
        if len(raw_preds.shape) > 1 and raw_preds.shape[1] > 1:
            # 获取前3个最可能的类别，然后随机选择一个，添加随机性
            # 这样可以避免始终返回相同的预测结果
            top_n = min(3, raw_preds.shape[1])
            if np.random.random() < 0.7:  # 70%的概率使用最高概率类别
                return np.argmax(raw_preds, axis=1)
            else:  # 30%的概率从前N个最可能的类别中随机选择
                top_indices = np.argsort(-raw_preds, axis=1)[:, :top_n]
                selected_indices = np.zeros(raw_preds.shape[0], dtype=int)
                for i in range(raw_preds.shape[0]):
                    selected_indices[i] = np.random.choice(top_indices[i])
                return selected_indices
        return raw_preds
    
    def train_xgboost(self, X_train, y_train, ball_type):
        """训练XGBoost模型，使用交叉验证和超参数调优"""
        import xgboost as xgb
        from sklearn.model_selection import StratifiedKFold
        import numpy as np
        
        self.log(f"训练{ball_type}球XGBoost模型...")
        self.log(f"数据维度: 特征={X_train.shape}, 标签={y_train.shape}")
        
        # 设置基础参数
        base_params = {
            'objective': 'multi:softmax',
            'num_class': self.red_range + 1 if ball_type == 'red' else self.blue_range + 1,
            'verbosity': 0,
            'eval_metric': ['mlogloss', 'merror'],  # 添加多个评估指标
        }
        
        # 如果使用GPU并且GPU可用，则添加GPU参数
        cuda_available = torch.cuda.is_available()
        mps_available = hasattr(torch, 'mps') and torch.backends.mps.is_available()
        
        if self.use_gpu and cuda_available:
            base_params['tree_method'] = 'gpu_hist'  # 使用CUDA GPU加速
            self.log("XGBoost使用CUDA GPU加速训练")
        elif self.use_gpu and mps_available:
            self.log("警告：XGBoost可能不支持MPS后端，将使用CPU训练")
        else:
            base_params['tree_method'] = 'hist'  # 使用快速直方图算法
        
        # 创建DMatrix数据结构
        dtrain = xgb.DMatrix(X_train, label=y_train)
        
        # 定义超参数搜索空间
        param_grid = [
            {'max_depth': 3, 'learning_rate': 0.1, 'subsample': 0.8, 'colsample_bytree': 0.8, 'min_child_weight': 1},
            {'max_depth': 5, 'learning_rate': 0.05, 'subsample': 0.7, 'colsample_bytree': 0.7, 'min_child_weight': 3},
            {'max_depth': 7, 'learning_rate': 0.01, 'subsample': 0.9, 'colsample_bytree': 0.9, 'min_child_weight': 5},
            {'max_depth': 6, 'learning_rate': 0.03, 'subsample': 0.85, 'colsample_bytree': 0.85, 'min_child_weight': 2},
            {'max_depth': 4, 'learning_rate': 0.07, 'subsample': 0.75, 'colsample_bytree': 0.75, 'min_child_weight': 4}
        ]
        
        # 设置交叉验证
        cv_folds = 5
        
        # 检查是否有类别样本数量过少的情况
        from collections import Counter
        class_counts = Counter(y_train)
        min_samples = min(class_counts.values())
        
        # 根据样本数量决定是否使用分层抽样
        use_stratified = min_samples >= 2
        if not use_stratified:
            self.log(f"警告: 某些类别样本数量过少(最少{min_samples}个)，XGBoost交叉验证将不使用分层抽样")
            from sklearn.model_selection import KFold
            kfold = KFold(n_splits=cv_folds, shuffle=True, random_state=42)
        else:
            kfold = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
        
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
        
        # 添加早停回调
        early_stopping = xgb.callback.EarlyStopping(
            rounds=10,
            metric_name='merror',
            data_name='eval',
            save_best=True
        )
        callbacks.append(early_stopping)
        
        # 执行网格搜索
        self.log("开始XGBoost超参数搜索...")
        best_score = float('inf')
        best_params = None
        best_model = None
        
        for params in param_grid:
            # 合并基础参数和当前参数集
            current_params = {**base_params, **params}
            self.log(f"测试参数: {params}")
            
            # 创建交叉验证数据集
            cv_results = xgb.cv(
                current_params,
                dtrain,
                num_boost_round=200,
                nfold=cv_folds,
                stratified=use_stratified,  # 根据样本数量决定是否使用分层抽样
                early_stopping_rounds=20,
                metrics=['merror'],
                seed=42
            )
            
            # 获取最佳迭代次数和错误率
            best_iteration = len(cv_results)
            best_error = cv_results['test-merror-mean'].iloc[-1]
            self.log(f"参数 {params} 的最佳迭代次数: {best_iteration}, 错误率: {best_error:.6f}")
            
            # 更新最佳参数
            if best_error < best_score:
                best_score = best_error
                best_params = current_params
                best_params['num_boost_round'] = best_iteration
        
        # 使用最佳参数训练最终模型
        self.log(f"使用最佳参数训练最终模型: {best_params}")
        num_boost_round = best_params.pop('num_boost_round', 100)
        
        # 创建验证集用于早停
        from sklearn.model_selection import train_test_split
        from collections import Counter
        
        # 检查是否有类别样本数量过少的情况
        class_counts = Counter(y_train)
        min_samples = min(class_counts.values())
        
        # 根据样本数量决定是否使用分层抽样
        use_stratified = min_samples >= 2
        if use_stratified:
            X_train_final, X_val, y_train_final, y_val = train_test_split(
                X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
            )
        else:
            self.log(f"警告: 某些类别样本数量过少(最少{min_samples}个)，使用随机抽样代替分层抽样")
            X_train_final, X_val, y_train_final, y_val = train_test_split(
                X_train, y_train, test_size=0.2, random_state=42, stratify=None
            )
        dtrain_final = xgb.DMatrix(X_train_final, label=y_train_final)
        dval = xgb.DMatrix(X_val, label=y_val)
        
        # 设置评估列表
        watchlist = [(dtrain_final, 'train'), (dval, 'eval')]
        
        # 训练最终模型
        self.log("训练最终XGBoost模型...")
        model = xgb.train(
            best_params, 
            dtrain_final, 
            num_boost_round=num_boost_round, 
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
        
        # 评估最终模型
        y_pred = model.predict(dval)
        accuracy = np.mean(y_pred == y_val)
        self.log(f"XGBoost验证集准确率: {accuracy:.4f}")
        
        # 保存原始模型
        self.raw_models[f'xgboost_{ball_type}'] = model
        
        # 包装模型
        wrapped_model = WrappedXGBoostModel(model, self.process_multidim_prediction)
        
        self.log(f"{ball_type}球XGBoost模型训练完成")
        return wrapped_model
    
    def train_gbdt(self, X_train, y_train, ball_type):
        """训练梯度提升决策树模型，使用交叉验证和超参数调优"""
        from sklearn.ensemble import GradientBoostingClassifier
        from sklearn.model_selection import GridSearchCV, StratifiedKFold
        from sklearn.metrics import accuracy_score, make_scorer
        from sklearn.model_selection import train_test_split, KFold
        
        self.log(f"训练{ball_type}球GBDT模型...")
        self.log(f"数据维度: 特征={X_train.shape}, 标签={y_train.shape}")
        
        # 创建验证集用于最终评估
        # 首先检查是否有类别样本数量过少的情况
        from collections import Counter
        class_counts = Counter(y_train)
        min_samples = min(class_counts.values())
        
        if min_samples < 2:
            # 如果有类别样本数量过少，直接使用随机抽样
            self.log(f"警告: 某些类别样本数量过少(最少{min_samples}个)，使用随机抽样代替分层抽样")
            X_train_main, X_val, y_train_main, y_val = train_test_split(
                X_train, y_train, test_size=0.2, random_state=42, stratify=None
            )
        else:
            try:
                # 尝试使用分层抽样
                X_train_main, X_val, y_train_main, y_val = train_test_split(
                    X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
                )
            except ValueError as e:
                # 如果出现其他错误，则不使用分层抽样
                self.log(f"警告: 分层抽样失败，使用随机抽样代替。原因: {str(e)}")
                X_train_main, X_val, y_train_main, y_val = train_test_split(
                    X_train, y_train, test_size=0.2, random_state=42, stratify=None
                )
        
        # 基础模型参数
        base_model = GradientBoostingClassifier(
            random_state=42,
            verbose=0,  # 减少自带的输出
            validation_fraction=0.2,  # 用于早停的验证集比例
            n_iter_no_change=10,  # 如果验证分数在10轮内没有改善，则停止
            tol=1e-4  # 改善的容忍度
        )
        
        # 设置超参数搜索空间 - 使用更小的搜索空间
        param_grid = {
            'n_estimators': [100, 200],
            'learning_rate': [0.05, 0.1],
            'max_depth': [3, 5],
            'min_samples_split': [2, 5],
            'min_samples_leaf': [1, 2],
            'subsample': [0.8, 1.0]
        }
        
        # 创建交叉验证对象
        # 检查数据中是否存在样本数量少于2的类别
        from collections import Counter
        class_counts = Counter(y_train_main)
        min_samples = min(class_counts.values())
        
        if min_samples < 2:
            self.log(f"警告: 某些类别样本数量过少(最少{min_samples}个)，使用普通KFold代替StratifiedKFold")
            cv = KFold(n_splits=3, shuffle=True, random_state=42)  # 使用普通KFold
        else:
            cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)  # 减少折数
        
        # 使用随机搜索替代网格搜索，大幅减少计算量
        from sklearn.model_selection import RandomizedSearchCV
        self.log("开始GBDT超参数随机搜索...")
        
        # 创建回调函数用于显示进度
        class ProgressCallback:
            def __init__(self, log_func, total_iters):
                self.log_func = log_func
                self.total_iters = total_iters
                self.current_iter = 0
                
            def __call__(self, *args, **kwargs):
                self.current_iter += 1
                if self.current_iter % 5 == 0 or self.current_iter == self.total_iters:
                    self.log_func(f"GBDT参数搜索进度: {self.current_iter}/{self.total_iters} ({self.current_iter/self.total_iters*100:.1f}%)")
                return 0
        
        # 计算总迭代次数 (n_iter * cv折数)
        n_iter = 10  # 随机搜索的迭代次数
        total_iters = n_iter * cv.get_n_splits()
        progress_callback = ProgressCallback(self.log, total_iters)
        
        random_search = RandomizedSearchCV(
            estimator=base_model,
            param_distributions=param_grid,
            n_iter=n_iter,  # 只尝试10组参数组合
            cv=cv,
            scoring=make_scorer(accuracy_score),
            n_jobs=-1,  # 使用所有可用的CPU核心
            verbose=0,
            random_state=42
        )
        
        # 执行随机搜索
        import time
        start_time = time.time()
        self.log("开始执行GBDT参数搜索...")
        
        # 由于sklearn的搜索没有回调机制，我们使用一个简单的定时器来显示进度
        import threading
        stop_event = threading.Event()
        
        def progress_monitor():
            iter_count = 0
            while not stop_event.is_set():
                iter_count += 1
                if iter_count % 5 == 0:
                    elapsed = time.time() - start_time
                    self.log(f"GBDT参数搜索进行中... 已用时: {elapsed:.1f}秒")
                time.sleep(5)  # 每5秒更新一次
        
        # 启动进度监控线程
        monitor_thread = threading.Thread(target=progress_monitor)
        monitor_thread.daemon = True
        monitor_thread.start()
        
        try:
            # 检查是否有类别样本数量过少的情况
            from collections import Counter
            class_counts = Counter(y_train_main)
            min_samples = min(class_counts.values())
            
            if min_samples < 2:
                self.log(f"警告: 训练数据中某些类别样本数量过少(最少{min_samples}个)，跳过参数搜索")
                # 直接使用默认参数创建模型，跳过参数搜索
                # 不抛出错误，而是直接返回默认模型
                model = GradientBoostingClassifier(
                    n_estimators=100,
                    learning_rate=0.1,
                    max_depth=3,
                    random_state=42,
                    verbose=0
                )
                model.fit(X_train, y_train)
                # 包装模型并返回
                wrapped_model = WrappedGBDTModel(model, self.process_multidim_prediction)
                self.raw_models[f'gbdt_{ball_type}'] = model
                self.log(f"{ball_type}球GBDT模型训练完成(使用默认参数)")
                return wrapped_model
            else:
                # 如果类别样本数量足够，添加error_score参数
                random_search.error_score = 'raise'
                # 尝试拟合模型
                random_search.fit(X_train_main, y_train_main)
            
        except ValueError as e:
            # 处理所有拟合失败的情况
            self.log(f"GBDT参数搜索失败: {str(e)}")
            self.log("使用默认参数训练模型...")
            # 使用默认参数创建模型
            model = GradientBoostingClassifier(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=3,
                random_state=42,
                verbose=0
            )
            model.fit(X_train, y_train)
            # 包装模型并返回
            wrapped_model = WrappedGBDTModel(model, self.process_multidim_prediction)
            self.raw_models[f'gbdt_{ball_type}'] = model
            self.log(f"{ball_type}球GBDT模型训练完成(使用默认参数)")
            return wrapped_model
        finally:
            stop_event.set()  # 停止监控线程
            elapsed = time.time() - start_time
            self.log(f"GBDT参数搜索完成，总用时: {elapsed:.1f}秒")
        
        # 获取最佳参数和模型
        best_params = random_search.best_params_
        best_score = random_search.best_score_
        
        self.log(f"GBDT最佳参数: {best_params}")
        self.log(f"GBDT交叉验证最佳得分: {best_score:.4f}")
        
        # 显示所有评估的参数组合及其得分
        self.log("评估的参数组合及得分 (前5个):")
        results = list(zip(random_search.cv_results_['params'], random_search.cv_results_['mean_test_score']))
        results.sort(key=lambda x: x[1], reverse=True)
        for i, (params, score) in enumerate(results[:5]):
            self.log(f"  {i+1}. 得分: {score:.4f}, 参数: {params}")
        
        # 使用最佳参数创建最终模型
        final_params = best_params.copy()
        
        # 检查是否有类别样本数量过少的情况
        from collections import Counter
        class_counts = Counter(y_train)
        min_samples = min(class_counts.values())
        
        if min_samples < 2:
            self.log(f"警告: 最终训练数据中某些类别样本数量过少(最少{min_samples}个)，禁用早停功能和内部交叉验证")
            # 只添加基本参数，禁用所有可能导致分层问题的功能
            final_params.update({
                'random_state': 42,
                'verbose': 0
            })
        else:
            # 如果样本数量足够，添加早停和验证功能
            final_params.update({
                'random_state': 42,
                'verbose': 0,
                'validation_fraction': 0.2,  # 用于早停的验证集比例
                'n_iter_no_change': 10,  # 如果验证分数在10轮内没有改善，则停止
                'tol': 1e-4  # 改善的容忍度
            })
        
        self.log(f"使用最佳参数训练最终GBDT模型...")
            
        model = GradientBoostingClassifier(**final_params)
        
        # 使用分批训练方式，显示进度
        n_estimators_per_batch = 20
        total_estimators = final_params['n_estimators']
        
        init_model = GradientBoostingClassifier(
            n_estimators=n_estimators_per_batch,
            learning_rate=final_params['learning_rate'],
            max_depth=final_params['max_depth'],
            min_samples_split=final_params['min_samples_split'],
            min_samples_leaf=final_params['min_samples_leaf'],
            subsample=final_params['subsample'],
            random_state=42,
            warm_start=True,  # 允许增量训练
            verbose=0
        )
        
        init_model.fit(X_train_main, y_train_main)
        
        for i in range(n_estimators_per_batch, total_estimators, n_estimators_per_batch):
            self.log(f"GBDT训练进度: {i}/{total_estimators} 棵树已完成 ({i/total_estimators*100:.1f}%)")
            init_model.n_estimators = min(i + n_estimators_per_batch, total_estimators)
            init_model.fit(X_train_main, y_train_main)
        
        self.log(f"GBDT训练进度: {total_estimators}/{total_estimators} 棵树已完成 (100%)")
        
        model = init_model
        
        # 在验证集上评估模型
        y_pred = model.predict(X_val)
        val_accuracy = accuracy_score(y_val, y_pred)
        self.log(f"GBDT验证集准确率: {val_accuracy:.4f}")
        
        # 输出特征重要性
        if hasattr(model, 'feature_importances_'):
            feature_names = [f"特征_{i}" for i in range(X_train.shape[1])]
            if len(self.feature_cols) == X_train.shape[1]:
                feature_names = self.feature_cols
                
            importance_data = sorted(zip(feature_names, model.feature_importances_), key=lambda x: x[1], reverse=True)
            
            self.log(f"GBDT特征重要性 (前10个):")
            for i, (feature, importance) in enumerate(importance_data[:10]):
                self.log(f"  {i+1}. {feature}: {importance:.4f}")
        
        # 保存原始模型
        self.raw_models[f'gbdt_{ball_type}'] = model
        
        # 包装模型
        wrapped_model = WrappedGBDTModel(model, self.process_multidim_prediction)
        
        self.log(f"{ball_type}球GBDT模型训练完成")
        return wrapped_model
    
    def train_lightgbm(self, X_train, y_train, ball_type):
        """训练LightGBM模型，使用交叉验证和超参数调优"""
        if not LIGHTGBM_AVAILABLE:
            self.log("LightGBM未安装，跳过此模型训练")
            return None
            
        import lightgbm as lgb
        from sklearn.model_selection import train_test_split
        import numpy as np
            
        self.log(f"训练{ball_type}球LightGBM模型...")
        self.log(f"数据维度: 特征={X_train.shape}, 标签={y_train.shape}")
        
        # 创建验证集用于早停和最终评估
        # 检查是否有类别样本数量过少的情况
        from collections import Counter
        class_counts = Counter(y_train)
        min_samples = min(class_counts.values())
        
        # 根据样本数量决定是否使用分层抽样
        use_stratified = min_samples >= 2
        if use_stratified:
            X_train_main, X_val, y_train_main, y_val = train_test_split(
                X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
            )
        else:
            self.log(f"警告: 某些类别样本数量过少(最少{min_samples}个)，使用随机抽样代替分层抽样")
            X_train_main, X_val, y_train_main, y_val = train_test_split(
                X_train, y_train, test_size=0.2, random_state=42, stratify=None
            )
        
        # 基础参数
        base_params = {
            'objective': 'multiclass',
            'num_class': self.red_range + 1 if ball_type == 'red' else self.blue_range + 1,
            'boosting_type': 'gbdt',
            'metric': ['multi_logloss', 'multi_error'],  # 添加多个评估指标
            'verbose': -1,  # 减少自带的输出
            'pred_early_stop': True,  # 早停预测
            'predict_disable_shape_check': True,  # 禁用形状检查
            'seed': 42,  # 设置随机种子
            'deterministic': True,  # 确保结果可重现
            'feature_pre_filter': False  # 允许动态改变min_data_in_leaf参数
        }
        
        # 检查GPU可用性
        cuda_available = torch.cuda.is_available()
        mps_available = hasattr(torch, 'mps') and torch.backends.mps.is_available()
        
        if self.use_gpu and cuda_available:
            base_params['device'] = 'gpu'  # 使用CUDA GPU加速
            self.log("LightGBM使用CUDA GPU加速训练")
        elif self.use_gpu and mps_available:
            self.log("警告：LightGBM可能不支持MPS后端，将使用CPU训练")
        
        # 创建数据集
        train_data = lgb.Dataset(X_train_main, label=y_train_main)
        valid_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
        
        # 定义超参数搜索空间
        param_grid = [
            {'learning_rate': 0.01, 'num_leaves': 31, 'max_depth': 6, 'min_data_in_leaf': 20, 'feature_fraction': 0.8, 'bagging_fraction': 0.8},
            {'learning_rate': 0.05, 'num_leaves': 63, 'max_depth': 8, 'min_data_in_leaf': 15, 'feature_fraction': 0.7, 'bagging_fraction': 0.7},
            {'learning_rate': 0.1, 'num_leaves': 127, 'max_depth': 10, 'min_data_in_leaf': 10, 'feature_fraction': 0.9, 'bagging_fraction': 0.9},
            {'learning_rate': 0.03, 'num_leaves': 31, 'max_depth': -1, 'min_data_in_leaf': 30, 'feature_fraction': 0.85, 'bagging_fraction': 0.85},
            {'learning_rate': 0.07, 'num_leaves': 63, 'max_depth': 12, 'min_data_in_leaf': 5, 'feature_fraction': 0.75, 'bagging_fraction': 0.75}
        ]
        
        # 进度回调函数
        def progress_callback(env):
            if env.iteration % 10 == 0 or env.iteration == env.end_iteration - 1:
                try:
                    eval_result = env.evaluation_result_list
                    if eval_result and len(eval_result) > 0:
                        metric_name = eval_result[0][1]
                        metric_value = eval_result[0][2]
                        self.log(f"LightGBM迭代 {env.iteration+1}/{env.end_iteration}: {metric_name}={metric_value:.6f}")
                    else:
                        self.log(f"LightGBM迭代 {env.iteration+1}/{env.end_iteration}")
                except Exception as e:
                    self.log(f"记录LightGBM进度时出错: {str(e)}")
            return False
        
        # 执行超参数搜索
        self.log("开始LightGBM超参数搜索...")
        best_score = float('inf')
        best_params = None
        best_rounds = 100
        
        for params in param_grid:
            # 合并基础参数和当前参数集
            current_params = {**base_params, **params}
            self.log(f"测试参数: {params}")
            
            # 执行交叉验证
            cv_results = lgb.cv(
                current_params,
                train_data,
                num_boost_round=300,
                nfold=5,
                stratified=True,
                # early_stopping_rounds参数在某些版本的LightGBM中不支持，移除此参数
                # early_stopping_rounds=20,
                metrics=['multi_error'],
                seed=42
                # verbose_eval参数在某些版本的LightGBM中不支持，移除此参数
                # verbose_eval=False
            )
            
            # 获取最佳迭代次数和错误率
            # 检查cv_results的键名格式
            metric_key = None
            for key in cv_results.keys():
                if 'multi_error' in key and 'mean' in key:
                    metric_key = key
                    break
            
            if not metric_key:
                self.log(f"警告: 在cv_results中找不到multi_error相关的键，可用的键: {list(cv_results.keys())}")
                # 使用默认值继续
                best_iteration = 100
                best_error = float('inf')
            else:
                best_iteration = len(cv_results[metric_key])
                best_error = cv_results[metric_key][-1]
            
            self.log(f"参数 {params} 的最佳迭代次数: {best_iteration}, 错误率: {best_error:.6f}")
            
            # 更新最佳参数
            if best_error < best_score:
                best_score = best_error
                best_params = current_params
                best_rounds = best_iteration
        
        # 使用最佳参数训练最终模型
        self.log(f"使用最佳参数训练最终模型: {best_params}")
        self.log(f"最佳迭代次数: {best_rounds}")
        
        # 设置验证集和回调
        valid_sets = [train_data, valid_data]
        valid_names = ['train', 'valid']
        callbacks = [progress_callback]
        
        # 训练最终模型
        self.log("训练最终LightGBM模型...")
        # 创建早停回调
        early_stopping = lgb.early_stopping(stopping_rounds=20)
        if early_stopping not in callbacks:
            callbacks.append(early_stopping)
            
        model = lgb.train(
            best_params, 
            train_data, 
            num_boost_round=best_rounds,
            valid_sets=valid_sets,
            valid_names=valid_names,
            callbacks=callbacks
            # early_stopping_rounds参数在新版本中已移至callbacks
        )
        
        # 在验证集上评估模型
        y_pred = model.predict(X_val)
        y_pred_class = np.argmax(y_pred, axis=1)
        accuracy = np.mean(y_pred_class == y_val)
        self.log(f"LightGBM验证集准确率: {accuracy:.4f}")
        
        # 输出特征重要性
        if hasattr(model, 'feature_importance'):
            try:
                importances = model.feature_importance(importance_type='gain')
                feature_names = model.feature_name()
                feature_importance = sorted(zip(feature_names, importances), key=lambda x: x[1], reverse=True)
                
                self.log(f"LightGBM特征重要性 (前10个):")
                for i, (feature, importance) in enumerate(feature_importance[:10]):
                    self.log(f"  {i+1}. {feature}: {importance:.4f}")
            except Exception as e:
                self.log(f"获取LightGBM特征重要性时出错: {str(e)}")
        
        # 保存原始模型
        self.raw_models[f'lightgbm_{ball_type}'] = model
        
        # 包装模型
        wrapped_model = WrappedLightGBMModel(model, self.process_multidim_prediction)
        
        self.log(f"{ball_type}球LightGBM模型训练完成")
        return wrapped_model
    
    def train_catboost(self, X_train, y_train, ball_type):
        """训练CatBoost模型，使用交叉验证和超参数调优"""
        if not CATBOOST_AVAILABLE:
            self.log("CatBoost未安装，跳过此模型训练")
            return None
            
        import catboost as cb
        from sklearn.model_selection import train_test_split
        import numpy as np
            
        self.log(f"训练{ball_type}球CatBoost模型...")
        self.log(f"数据维度: 特征={X_train.shape}, 标签={y_train.shape}")
        
        # 创建验证集用于早停和最终评估
        # 检查是否有类别样本数量过少的情况
        from collections import Counter
        class_counts = Counter(y_train)
        min_samples = min(class_counts.values())
        
        # 根据样本数量决定是否使用分层抽样
        use_stratified = min_samples >= 2
        if use_stratified:
            X_train_main, X_val, y_train_main, y_val = train_test_split(
                X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
            )
        else:
            self.log(f"警告: 某些类别样本数量过少(最少{min_samples}个)，使用随机抽样代替分层抽样")
            X_train_main, X_val, y_train_main, y_val = train_test_split(
                X_train, y_train, test_size=0.2, random_state=42, stratify=None
            )
        
        # 设置输出目录
        train_dir = f"catboost_info_{ball_type}"
        os.makedirs(train_dir, exist_ok=True)
        
        # 确定分类数
        if ball_type == 'red':
            classes_count = self.red_range + 1
        else:  # blue
            classes_count = self.blue_range + 1
        
        # 基础参数
        base_params = {
            'loss_function': 'MultiClass',
            'classes_count': classes_count,
            'random_seed': 42,
            'verbose': False,
            'train_dir': train_dir,
            'eval_metric': 'MultiClass',  # 使用多分类评估指标
            'use_best_model': True,  # 使用验证集上表现最好的模型
            'od_type': 'Iter',  # 迭代次数早停类型
            'od_wait': 20,  # 如果20轮内没有改善，则停止
            'metric_period': 10  # 每10轮评估一次指标
        }
        
        # 如果GPU可用，启用GPU加速
        if self.use_gpu and hasattr(cb, 'CatBoostClassifier') and hasattr(cb.CatBoostClassifier, 'is_cuda_supported') and cb.CatBoostClassifier.is_cuda_supported():
            base_params['task_type'] = 'GPU'
            self.log("CatBoost使用GPU加速训练")
        
        # 定义超参数搜索空间
        param_grid = [
            {'iterations': 200, 'learning_rate': 0.03, 'depth': 6, 'l2_leaf_reg': 3, 'bootstrap_type': 'Bayesian', 'random_strength': 1},
            {'iterations': 300, 'learning_rate': 0.01, 'depth': 8, 'l2_leaf_reg': 1, 'bootstrap_type': 'Bernoulli', 'subsample': 0.8},
            {'iterations': 150, 'learning_rate': 0.1, 'depth': 4, 'l2_leaf_reg': 5, 'bootstrap_type': 'MVS'},
            {'iterations': 250, 'learning_rate': 0.05, 'depth': 7, 'l2_leaf_reg': 2, 'bootstrap_type': 'Bernoulli', 'subsample': 0.85},
            {'iterations': 200, 'learning_rate': 0.07, 'depth': 5, 'l2_leaf_reg': 4, 'bootstrap_type': 'Bayesian', 'random_strength': 0.5}
        ]
        
        # 创建自定义回调，用于日志和暂停支持
        class LoggerCallback(object):
            def __init__(self, logger, progress_interval=10):
                self.logger = logger
                self.progress_interval = progress_interval
                self.iteration = 0
                
            def after_iteration(self, info):
                self.iteration += 1
                if self.iteration % self.progress_interval == 0:
                    self.logger(f"CatBoost训练进度: {self.iteration}/{info.params.iterations} 迭代已完成 ({self.iteration/info.params.iterations*100:.1f}%)")
                return False  # 返回False表示继续训练
        
        try:
            # 执行超参数搜索
            self.log("开始CatBoost超参数搜索...")
            best_score = float('inf')
            best_params = None
            best_model = None
            
            # 准备训练和验证数据集
            train_pool = cb.Pool(X_train_main, y_train_main)
            val_pool = cb.Pool(X_val, y_val)
            
            for params in param_grid:
                # 合并基础参数和当前参数集
                current_params = {**base_params, **params}
                self.log(f"测试参数: {params}")
                
                # 创建模型
                model = cb.CatBoostClassifier(**current_params)
                
                # 训练模型
                model.fit(
                    train_pool,
                    eval_set=val_pool,
                    verbose=False,
                    plot=False
                )
                
                # 评估模型
                val_predictions = model.predict(X_val)
                val_accuracy = np.mean(val_predictions == y_val)
                val_error = 1 - val_accuracy
                
                self.log(f"参数 {params} 的验证集准确率: {val_accuracy:.4f}, 错误率: {val_error:.4f}")
                
                # 更新最佳参数
                if val_error < best_score:
                    best_score = val_error
                    best_params = current_params
                    best_model = model
            
            # 使用最佳参数训练最终模型
            if best_model is None:
                self.log("超参数搜索未找到有效模型，使用默认参数")
                best_params = {**base_params, **param_grid[0]}
                
            self.log(f"使用最佳参数训练最终模型: {best_params}")
            
            # 如果已经有最佳模型，直接使用
            if best_model is not None:
                model = best_model
                self.log("使用超参数搜索中找到的最佳模型")
            else:
                # 创建新模型并训练
                model = cb.CatBoostClassifier(**best_params)
                
                # 创建回调
                callbacks = [LoggerCallback(self.log)]
                
                # 训练模型
                self.log("训练最终CatBoost模型...")
                model.fit(
                    train_pool,
                    eval_set=val_pool,
                    callbacks=callbacks,
                    verbose=False,
                    plot=False
                )
            
            # 在验证集上评估最终模型
            val_predictions = model.predict(X_val)
            val_accuracy = np.mean(val_predictions == y_val)
            self.log(f"CatBoost验证集准确率: {val_accuracy:.4f}")
            
            # 特征重要性
            if hasattr(model, 'get_feature_importance'):
                importances = model.get_feature_importance()
                feature_names = [f"特征_{i}" for i in range(X_train.shape[1])]
                
                importance_data = sorted(zip(feature_names, importances), key=lambda x: x[1], reverse=True)
                
                self.log(f"CatBoost特征重要性 (前10个):")
                for i, (feature, importance) in enumerate(importance_data[:10]):
                    self.log(f"  {i+1}. {feature}: {importance:.4f}")
            
            # 保存原始模型
            self.raw_models[f'catboost_{ball_type}'] = model
            
            # 包装模型
            wrapped_model = WrappedCatBoostModel(model, self.process_multidim_prediction)
            
            self.log(f"{ball_type}球CatBoost模型训练完成")
            return wrapped_model
            
        except Exception as e:
            self.log(f"训练CatBoost模型时出错: {str(e)}")
            import traceback
            self.log(traceback.format_exc())
            return None
    
    def train_ensemble(self, X_train, y_train, ball_type):
        """训练集成模型（包含多种基础模型），使用加权投票机制"""
        self.log(f"训练{ball_type}球集成模型...")
        self.log(f"数据维度: 特征={X_train.shape}, 标签={y_train.shape}")
        
        # 划分验证集用于评估各个模型性能
        from sklearn.model_selection import train_test_split
        import numpy as np
        
        # 检查是否有类别样本数量过少的情况
        from collections import Counter
        class_counts = Counter(y_train)
        min_samples = min(class_counts.values())
        
        # 根据样本数量决定是否使用分层抽样
        use_stratified = min_samples >= 2
        if use_stratified:
            X_train_main, X_val, y_train_main, y_val = train_test_split(
                X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
            )
        else:
            self.log(f"警告: 某些类别样本数量过少(最少{min_samples}个)，使用随机抽样代替分层抽样")
            X_train_main, X_val, y_train_main, y_val = train_test_split(
                X_train, y_train, test_size=0.2, random_state=42, stratify=None
            )
        
        ensemble_models = {}
        model_weights = {}
        model_accuracies = {}
        total_models = 5 if LIGHTGBM_AVAILABLE and CATBOOST_AVAILABLE else 3
        current_model = 0
        
        self.log(f"集成模型将训练以下子模型: 随机森林, XGBoost, GBDT{', LightGBM' if LIGHTGBM_AVAILABLE else ''}{', CatBoost' if CATBOOST_AVAILABLE else ''}")
        self.log("使用加权投票机制，根据验证集性能为每个模型分配权重")
        
        # 训练随机森林模型
        current_model += 1
        self.log(f"集成模型进度: 开始训练子模型 ({current_model}/{total_models}) - 随机森林")
        rf_model = self.train_random_forest(X_train_main, y_train_main, ball_type)
        if rf_model:
            ensemble_models['random_forest'] = rf_model
            # 评估模型在验证集上的性能
            try:
                y_pred = rf_model.predict(X_val)
                if len(y_pred.shape) > 1 and y_pred.shape[1] > 1:
                    y_pred = np.argmax(y_pred, axis=1)
                accuracy = np.mean(y_pred == y_val)
                model_accuracies['random_forest'] = accuracy
                self.log(f"随机森林模型在验证集上的准确率: {accuracy:.4f}")
            except Exception as e:
                self.log(f"评估随机森林模型时出错: {str(e)}")
                model_accuracies['random_forest'] = 0.5  # 默认权重
            self.log(f"集成模型进度: 随机森林模型添加完成 ({current_model}/{total_models})")
            
        # 训练XGBoost模型
        current_model += 1
        self.log(f"集成模型进度: 开始训练子模型 ({current_model}/{total_models}) - XGBoost")
        xgb_model = self.train_xgboost(X_train_main, y_train_main, ball_type)
        if xgb_model:
            ensemble_models['xgboost'] = xgb_model
            # 评估模型在验证集上的性能
            try:
                y_pred = xgb_model.predict(X_val)
                if len(y_pred.shape) > 1 and y_pred.shape[1] > 1:
                    y_pred = np.argmax(y_pred, axis=1)
                accuracy = np.mean(y_pred == y_val)
                model_accuracies['xgboost'] = accuracy
                self.log(f"XGBoost模型在验证集上的准确率: {accuracy:.4f}")
            except Exception as e:
                self.log(f"评估XGBoost模型时出错: {str(e)}")
                model_accuracies['xgboost'] = 0.5  # 默认权重
            self.log(f"集成模型进度: XGBoost模型添加完成 ({current_model}/{total_models})")
            
        # 训练GBDT模型
        current_model += 1
        self.log(f"集成模型进度: 开始训练子模型 ({current_model}/{total_models}) - GBDT")
        gbdt_model = self.train_gbdt(X_train_main, y_train_main, ball_type)
        if gbdt_model:
            ensemble_models['gbdt'] = gbdt_model
            # 评估模型在验证集上的性能
            try:
                y_pred = gbdt_model.predict(X_val)
                if len(y_pred.shape) > 1 and y_pred.shape[1] > 1:
                    y_pred = np.argmax(y_pred, axis=1)
                accuracy = np.mean(y_pred == y_val)
                model_accuracies['gbdt'] = accuracy
                self.log(f"GBDT模型在验证集上的准确率: {accuracy:.4f}")
            except Exception as e:
                self.log(f"评估GBDT模型时出错: {str(e)}")
                model_accuracies['gbdt'] = 0.5  # 默认权重
            self.log(f"集成模型进度: GBDT模型添加完成 ({current_model}/{total_models})")
            
        # 训练LightGBM模型
        if LIGHTGBM_AVAILABLE:
            current_model += 1
            self.log(f"集成模型进度: 开始训练子模型 ({current_model}/{total_models}) - LightGBM")
            lgb_model = self.train_lightgbm(X_train_main, y_train_main, ball_type)
            if lgb_model:
                ensemble_models['lightgbm'] = lgb_model
                # 评估模型在验证集上的性能
                try:
                    y_pred = lgb_model.predict(X_val)
                    if len(y_pred.shape) > 1 and y_pred.shape[1] > 1:
                        y_pred = np.argmax(y_pred, axis=1)
                    accuracy = np.mean(y_pred == y_val)
                    model_accuracies['lightgbm'] = accuracy
                    self.log(f"LightGBM模型在验证集上的准确率: {accuracy:.4f}")
                except Exception as e:
                    self.log(f"评估LightGBM模型时出错: {str(e)}")
                    model_accuracies['lightgbm'] = 0.5  # 默认权重
                self.log(f"集成模型进度: LightGBM模型添加完成 ({current_model}/{total_models})")
                
        # 训练CatBoost模型
        if CATBOOST_AVAILABLE:
            current_model += 1
            self.log(f"集成模型进度: 开始训练子模型 ({current_model}/{total_models}) - CatBoost")
            cb_model = self.train_catboost(X_train_main, y_train_main, ball_type)
            if cb_model:
                ensemble_models['catboost'] = cb_model
                # 评估模型在验证集上的性能
                try:
                    y_pred = cb_model.predict(X_val)
                    if len(y_pred.shape) > 1 and y_pred.shape[1] > 1:
                        y_pred = np.argmax(y_pred, axis=1)
                    accuracy = np.mean(y_pred == y_val)
                    model_accuracies['catboost'] = accuracy
                    self.log(f"CatBoost模型在验证集上的准确率: {accuracy:.4f}")
                except Exception as e:
                    self.log(f"评估CatBoost模型时出错: {str(e)}")
                    model_accuracies['catboost'] = 0.5  # 默认权重
                self.log(f"集成模型进度: CatBoost模型添加完成 ({current_model}/{total_models})")
        
        # 计算模型权重
        if model_accuracies:
            # 使用softmax函数将准确率转换为权重
            accuracies = np.array(list(model_accuracies.values()))
            # 添加温度参数使权重分布更加明显
            temperature = 2.0
            exp_accuracies = np.exp((accuracies - np.min(accuracies)) / temperature)
            softmax_weights = exp_accuracies / np.sum(exp_accuracies)
            
            # 将权重分配给各个模型
            for i, model_name in enumerate(model_accuracies.keys()):
                model_weights[model_name] = softmax_weights[i]
        else:
            # 如果没有准确率信息，使用均等权重
            for model_name in ensemble_models.keys():
                model_weights[model_name] = 1.0 / len(ensemble_models)
        
        # 保存模型权重
        self.model_weights[f'{ball_type}_weights'] = model_weights
        
        # 输出模型权重信息
        self.log(f"集成模型完成，包含 {len(ensemble_models)} 个子模型")
        self.log("模型权重分配:")
        for model_name, weight in model_weights.items():
            self.log(f"- {model_name}: {weight:.4f} (准确率: {model_accuracies.get(model_name, 0):.4f})")
        
        # 使用加权投票在验证集上评估集成模型性能
        if X_val is not None and y_val is not None and ensemble_models:
            self.log("在验证集上评估集成模型性能...")
            try:
                # 获取每个模型的预测结果
                predictions = {}
                for model_name, model in ensemble_models.items():
                    y_pred = model.predict(X_val)
                    if len(y_pred.shape) > 1 and y_pred.shape[1] > 1:
                        y_pred = np.argmax(y_pred, axis=1)
                    predictions[model_name] = y_pred
                
                # 使用加权投票进行集成预测
                if ball_type == 'red':
                    num_classes = self.red_range + 1
                else:  # blue
                    num_classes = self.blue_range + 1
                
                # 初始化投票矩阵
                vote_matrix = np.zeros((len(y_val), num_classes))
                
                # 对每个模型的预测进行加权投票
                for model_name, preds in predictions.items():
                    weight = model_weights[model_name]
                    for i, pred in enumerate(preds):
                        if pred < num_classes:  # 确保预测类别在有效范围内
                            vote_matrix[i, pred] += weight
                
                # 获取得票最多的类别作为最终预测
                ensemble_preds = np.argmax(vote_matrix, axis=1)
                
                # 计算集成模型准确率
                ensemble_accuracy = np.mean(ensemble_preds == y_val)
                self.log(f"集成模型在验证集上的准确率: {ensemble_accuracy:.4f}")
                
                # 与最佳单一模型比较
                best_single_model = max(model_accuracies.items(), key=lambda x: x[1])
                self.log(f"最佳单一模型 ({best_single_model[0]}) 准确率: {best_single_model[1]:.4f}")
                
                if ensemble_accuracy > best_single_model[1]:
                    improvement = (ensemble_accuracy - best_single_model[1]) / best_single_model[1] * 100
                    self.log(f"集成模型相比最佳单一模型提升: {improvement:.2f}%")
                else:
                    decrease = (best_single_model[1] - ensemble_accuracy) / best_single_model[1] * 100
                    self.log(f"警告: 集成模型相比最佳单一模型下降: {decrease:.2f}%")
            except Exception as e:
                self.log(f"评估集成模型时出错: {str(e)}")
                import traceback
                self.log(traceback.format_exc())
        
        return ensemble_models
    
    def train(self, df):
        """训练模型"""
        self.log("============ 开始训练模型 ============")
        self.log(f"彩票类型: {self.lottery_type.upper()}")
        self.log(f"选择的模型类型: {self.model_type}")
        
        training_start_time = time.time()
        
        if self.model_type == 'expected_value' and EXPECTED_VALUE_MODEL_AVAILABLE:
            self.log("\n====== 使用期望值模型 ======")
            self.train_expected_value_model(df)
            total_time = time.time() - training_start_time
            self.log(f"\n训练完成，总耗时: {total_time:.2f}秒")
            return self.models
        
        self.log("\n====== 第1阶段: 数据准备 ======")
        data_prep_start = time.time()
        X_train, X_test, y_red_train, y_red_test, y_blue_train, y_blue_test = self.prepare_data(df)
        data_prep_time = time.time() - data_prep_start
        self.log(f"数据准备完成，耗时: {data_prep_time:.2f}秒")
        
        self.models = {}
        
        self.log("\n====== 第2阶段: 模型训练 ======")
        model_train_start = time.time()
        
        if self.model_type == 'random_forest':
            self.log("\n----- 使用随机森林模型 -----")
            self.models['red'] = self.train_random_forest(X_train, y_red_train, 'red')
            self.models['blue'] = self.train_random_forest(X_train, y_blue_train, 'blue')
        
        elif self.model_type == 'xgboost':
            self.log("\n----- 使用XGBoost模型 -----")
            self.models['red'] = self.train_xgboost(X_train, y_red_train, 'red')
            self.models['blue'] = self.train_xgboost(X_train, y_blue_train, 'blue')
        
        elif self.model_type == 'gbdt':
            self.log("\n----- 使用梯度提升决策树 -----")
            self.models['red'] = self.train_gbdt(X_train, y_red_train, 'red')
            self.models['blue'] = self.train_gbdt(X_train, y_blue_train, 'blue')
        
        elif self.model_type == 'lightgbm' and LIGHTGBM_AVAILABLE:
            self.log("\n----- 使用LightGBM模型 -----")
            self.models['red'] = self.train_lightgbm(X_train, y_red_train, 'red')
            self.models['blue'] = self.train_lightgbm(X_train, y_blue_train, 'blue')
        
        elif self.model_type == 'catboost' and CATBOOST_AVAILABLE:
            self.log("\n----- 使用CatBoost模型 -----")
            self.models['red'] = self.train_catboost(X_train, y_red_train, 'red')
            self.models['blue'] = self.train_catboost(X_train, y_blue_train, 'blue')
        
        elif self.model_type == 'ensemble':
            self.log("\n----- 使用集成模型 -----")
            self.models['red'] = self.train_ensemble(X_train, y_red_train, 'red')
            self.models['blue'] = self.train_ensemble(X_train, y_blue_train, 'blue')
        
        model_train_time = time.time() - model_train_start
        self.log(f"\n模型训练完成，耗时: {model_train_time:.2f}秒")
        
        self.log("\n====== 第3阶段: 模型评估 ======")
        if X_test is not None and y_red_test is not None and y_blue_test is not None:
            eval_start = time.time()
            self.evaluate(X_test, y_red_test, y_blue_test)
            eval_time = time.time() - eval_start
            self.log(f"模型评估完成，耗时: {eval_time:.2f}秒")
        
        self.log("\n====== 第4阶段: 模型保存 ======")
        save_start = time.time()
        self.save_models()
        save_time = time.time() - save_start
        self.log(f"模型保存完成，耗时: {save_time:.2f}秒")
        
        total_time = time.time() - training_start_time
        self.log(f"\n训练完成，总耗时: {total_time:.2f}秒")
        
        return self.models
        
    def train_expected_value_model(self, df):
        """训练期望值模型"""
        self.log(f"开始训练期望值模型...")
        
        self.models = {}
        
        red_probs_file = os.path.join(self.models_dir, 'ev_red_probabilities.pkl')
        if os.path.exists(red_probs_file):
            self.log("期望值模型文件已存在，尝试加载...")
            
            ev_model = ExpectedValueLotteryModel(
                lottery_type=self.lottery_type,
                log_callback=self.log,
                use_gpu=self.use_gpu
            )
                
            load_success = ev_model.load()
            if load_success:
                self.log("期望值模型加载成功")
                self.models['red'] = ev_model
                self.models['blue'] = ev_model
                # 保存到原始模型中以便序列化
                self.raw_models['expected_value_model'] = ev_model
                return
            else:
                self.log("期望值模型加载失败，将重新训练...")
        
 
        ev_model = ExpectedValueLotteryModel(
            lottery_type=self.lottery_type,
            log_callback=self.log,
            use_gpu=self.use_gpu
        )
        
       
        ev_model.train(df)
        
        self.models['red'] = ev_model
        self.models['blue'] = ev_model
        self.raw_models['expected_value_model'] = ev_model
        
        self.log("期望值模型训练完成")
    
    def evaluate(self, X_test, y_red_test, y_blue_test):
        """评估模型性能"""
        self.log("评估模型性能...")
        
        if len(y_red_test.shape) == 1:
            y_red_test = y_red_test.reshape(-1, 1)
        if len(y_blue_test.shape) == 1:
            y_blue_test = y_blue_test.reshape(-1, 1)
            
        red_accuracy = 0
        blue_accuracy = 0
        
        if 'red' in self.models:
            # 处理不同类型的模型
            if self.model_type == 'ensemble':
                # 集成模型需要单独处理，对每个子模型进行预测，然后进行加权投票
                red_votes = {}
                red_weighted_votes = {}
                
                # 检查是否有模型权重
                has_weights = hasattr(self, 'model_weights') and 'red' in self.model_weights
                
                for model_name, model in self.models['red'].items():
                    self.log(f"评估{model_name}模型...")
                    try:
                        y_pred = model.predict(X_test)
                        
                        # 预测结果处理，确保能够进行投票
                        if len(y_pred.shape) > 1 and y_pred.shape[1] > 1:
                            y_pred = np.argmax(y_pred, axis=1)
                        
                        # 获取当前模型的权重
                        weight = 1.0  # 默认权重
                        if has_weights and model_name in self.model_weights['red']:
                            weight = self.model_weights['red'][model_name]
                            self.log(f"{model_name}模型权重: {weight:.4f}")
                        
                        # 记录每个样本的加权投票
                        for i, pred in enumerate(y_pred):
                            if i not in red_votes:
                                red_votes[i] = {}
                                red_weighted_votes[i] = {}
                            if pred not in red_votes[i]:
                                red_votes[i][pred] = 0
                                red_weighted_votes[i][pred] = 0.0
                            red_votes[i][pred] += 1
                            red_weighted_votes[i][pred] += weight
                    except Exception as e:
                        self.log(f"评估{model_name}模型时出错: {e}")
                
                # 根据加权投票确定最终预测结果
                y_pred_red = []
                for i in range(len(X_test)):
                    if i in red_weighted_votes and red_weighted_votes[i]:
                        # 使用加权投票结果
                        pred_class = max(red_weighted_votes[i].items(), key=lambda x: x[1])[0]
                        y_pred_red.append(pred_class)
                    elif i in red_votes and red_votes[i]:
                        # 如果没有权重信息，回退到普通投票
                        pred_class = max(red_votes[i].items(), key=lambda x: x[1])[0]
                        y_pred_red.append(pred_class)
                    else:
                        # 如果没有投票，默认预测0
                        y_pred_red.append(0)
                
                y_pred_red = np.array(y_pred_red)
            else:
                # 单一模型
                y_pred_red = self.models['red'].predict(X_test)
            
            if len(y_pred_red.shape) > 1 and y_pred_red.shape[1] > 1:
                self.log(f"处理多维预测结果，形状: {y_pred_red.shape}")
                y_pred_red = np.argmax(y_pred_red, axis=1)
            
            y_red_test_flat = y_red_test.flatten()
            y_pred_red_flat = y_pred_red.flatten()
            
            self.log(f"红球预测形状: {y_pred_red_flat.shape}, 真实值形状: {y_red_test_flat.shape}")
            
            red_accuracy = np.mean(y_pred_red_flat == y_red_test_flat)
            self.log(f"红球预测准确率: {red_accuracy:.4f}")
        
        if 'blue' in self.models:
            # 处理不同类型的模型
            if self.model_type == 'ensemble':
                # 集成模型需要单独处理，对每个子模型进行预测，然后进行加权投票
                blue_votes = {}
                blue_weighted_votes = {}
                
                # 检查是否有模型权重
                has_weights = hasattr(self, 'model_weights') and 'blue' in self.model_weights
                
                for model_name, model in self.models['blue'].items():
                    self.log(f"评估{model_name}模型...")
                    try:
                        y_pred = model.predict(X_test)
                        
                        # 预测结果处理，确保能够进行投票
                        if len(y_pred.shape) > 1 and y_pred.shape[1] > 1:
                            y_pred = np.argmax(y_pred, axis=1)
                        
                        # 获取当前模型的权重
                        weight = 1.0  # 默认权重
                        if has_weights and model_name in self.model_weights['blue']:
                            weight = self.model_weights['blue'][model_name]
                            self.log(f"{model_name}模型权重: {weight:.4f}")
                        
                        # 记录每个样本的加权投票
                        for i, pred in enumerate(y_pred):
                            if i not in blue_votes:
                                blue_votes[i] = {}
                                blue_weighted_votes[i] = {}
                            if pred not in blue_votes[i]:
                                blue_votes[i][pred] = 0
                                blue_weighted_votes[i][pred] = 0.0
                            blue_votes[i][pred] += 1
                            blue_weighted_votes[i][pred] += weight
                    except Exception as e:
                        self.log(f"评估{model_name}模型时出错: {e}")
                
                # 根据加权投票确定最终预测结果
                y_pred_blue = []
                for i in range(len(X_test)):
                    if i in blue_weighted_votes and blue_weighted_votes[i]:
                        # 使用加权投票结果
                        pred_class = max(blue_weighted_votes[i].items(), key=lambda x: x[1])[0]
                        y_pred_blue.append(pred_class)
                    elif i in blue_votes and blue_votes[i]:
                        # 如果没有权重信息，回退到普通投票
                        pred_class = max(blue_votes[i].items(), key=lambda x: x[1])[0]
                        y_pred_blue.append(pred_class)
                    else:
                        # 如果没有投票，默认预测0
                        y_pred_blue.append(0)
                
                y_pred_blue = np.array(y_pred_blue)
            else:
                # 单一模型
                y_pred_blue = self.models['blue'].predict(X_test)
            
            if len(y_pred_blue.shape) > 1 and y_pred_blue.shape[1] > 1:
                self.log(f"处理多维预测结果，形状: {y_pred_blue.shape}")
                y_pred_blue = np.argmax(y_pred_blue, axis=1)
            
            y_blue_test_flat = y_blue_test.flatten()
            y_pred_blue_flat = y_pred_blue.flatten()
            
            # 记录预测和实际值的形状以便调试
            self.log(f"蓝球预测形状: {y_pred_blue_flat.shape}, 真实值形状: {y_blue_test_flat.shape}")
            
            # 计算准确率
            blue_accuracy = np.mean(y_pred_blue_flat == y_blue_test_flat)
            self.log(f"蓝球预测准确率: {blue_accuracy:.4f}")
        
        # 计算整体准确率
        overall_accuracy = (red_accuracy + blue_accuracy) / 2
        self.log(f"整体预测准确率: {overall_accuracy:.4f}")
        
        return red_accuracy, blue_accuracy
    
    def save_models(self):
        """保存模型、缩放器和模型权重"""
        self.log("\n----- 保存模型、缩放器和模型权重 -----")
        
        # 对于期望值模型，创建信息文件以确保目录存在
        if self.model_type == 'expected_value' and EXPECTED_VALUE_MODEL_AVAILABLE:
            # 期望值模型在train_expected_value_model中已经保存
            self.log("期望值模型已在训练过程中自动保存")
            
            # 创建模型信息文件，确保目录存在
            model_dir = os.path.join(self.models_dir, self.model_type)
            os.makedirs(model_dir, exist_ok=True)
            
            model_info = {
                'model_type': self.model_type,
                'lottery_type': self.lottery_type,
                'created_at': time.strftime('%Y-%m-%d %H:%M:%S')
            }
            info_path = os.path.join(model_dir, 'model_info.json')
            with open(info_path, 'w') as f:
                json.dump(model_info, f, indent=4)
            
            return True
        
        # 为其他模型创建保存目录
        model_dir = os.path.join(self.models_dir, self.model_type)
        os.makedirs(model_dir, exist_ok=True)
        
        # 保存模型和缩放器
        for ball_type in ['red', 'blue']:
            if ball_type not in self.models:
                continue
                
            model = self.models[ball_type]
            
            # 对于不同类型的模型使用不同的保存方式
            if self.model_type == 'ensemble':
                # 为集成模型保存每个子模型
                for model_name, sub_model in model.items():
                    sub_model_path = os.path.join(model_dir, f'{ball_type}_{model_name}_model.pkl')
                    with open(sub_model_path, 'wb') as f:
                        pickle.dump(sub_model, f)
                    self.log(f"保存{ball_type}球{model_name}模型: {sub_model_path}")
                
                # 保存模型权重（如果存在）
                if hasattr(self, 'model_weights') and ball_type in self.model_weights:
                    weights_path = os.path.join(model_dir, f'{ball_type}_model_weights.json')
                    with open(weights_path, 'w') as f:
                        json.dump(self.model_weights[ball_type], f, indent=4)
                    self.log(f"保存{ball_type}球模型权重: {weights_path}")
            elif hasattr(model, 'predict'):
                # 对于sklearn或类似模型使用pickle保存
                model_path = os.path.join(model_dir, f'{ball_type}_model.pkl')
                with open(model_path, 'wb') as f:
                    pickle.dump(model, f)
                self.log(f"保存{ball_type}球模型: {model_path}")
            else:
                self.log(f"警告: {ball_type}球模型不支持序列化")
        
        # 保存特征缩放器
        if 'X' in self.scalers:
            # 保存X缩放器，这是在训练过程中创建的主要缩放器
            x_scaler_path = os.path.join(model_dir, 'X_scaler.pkl')
            with open(x_scaler_path, 'wb') as f:
                pickle.dump(self.scalers['X'], f)
            self.log(f"保存特征缩放器: {x_scaler_path}")
            
            # 将X缩放器复制到red和blue球的缩放器中
            for ball_type in ['red', 'blue']:
                self.scalers[ball_type] = self.scalers['X']
                scaler_path = os.path.join(model_dir, f'{ball_type}_scaler.pkl')
                with open(scaler_path, 'wb') as f:
                    pickle.dump(self.scalers['X'], f)
                self.log(f"保存{ball_type}球特征缩放器: {scaler_path}")
        else:
            # 保存单独的球缩放器（如果存在）
            for ball_type in ['red', 'blue']:
                if ball_type in self.scalers:
                    scaler_path = os.path.join(model_dir, f'{ball_type}_scaler.pkl')
                    with open(scaler_path, 'wb') as f:
                        pickle.dump(self.scalers[ball_type], f)
                    self.log(f"保存{ball_type}球特征缩放器: {scaler_path}")
                else:
                    self.log(f"警告: 没有找到{ball_type}球的特征缩放器")
        
        # 保存模型信息
        model_info = {
            'model_type': self.model_type,
            'lottery_type': self.lottery_type,
            'feature_window': self.feature_window,
            'created_at': time.strftime('%Y-%m-%d %H:%M:%S')
        }
        
        # 添加特征数量信息
        expected_feature_count = getattr(self, 'expected_feature_count', None)
        if expected_feature_count:
            model_info['expected_feature_count'] = expected_feature_count
            self.log(f"保存特征数量信息: {expected_feature_count}")
        
        # 尝试从模型中获取特征数量信息
        if 'red' in self.models:
            model_obj = self.models['red']
            # 处理不同类型的模型
            if self.model_type == 'ensemble' and 'random_forest' in model_obj:
                model_obj = model_obj['random_forest']
                
            if hasattr(model_obj, 'n_features_in_'):
                model_info['n_features_in'] = model_obj.n_features_in_
                self.log(f"从模型中获取的特征数量: {model_obj.n_features_in_}")
            # 尝试获取更多的模型属性
            elif hasattr(model_obj, 'estimators_') and model_obj.estimators_:
                first_estimator = model_obj.estimators_[0]
                if hasattr(first_estimator, 'n_features_in_'):
                    model_info['n_features_in'] = first_estimator.n_features_in_
                    self.log(f"从模型第一个估计器中获取的特征数量: {first_estimator.n_features_in_}")
            # 最后尝试使用特征重要性的维度
            elif hasattr(model_obj, 'feature_importances_') and hasattr(model_obj.feature_importances_, 'shape'):
                model_info['n_features_in'] = model_obj.feature_importances_.shape[0]
                self.log(f"从模型特征重要性中获取的特征数量: {model_obj.feature_importances_.shape[0]}")
        
        # 如果没有从任何源获取到特征数量，默认保存为70
        if 'expected_feature_count' not in model_info and 'n_features_in' not in model_info:
            model_info['expected_feature_count'] = 70
            self.log(f"使用默认特征数量: 70")
            
        # 保存预测使用的预处理方法
        model_info['feature_data'] = {
            'window_size': self.feature_window,
            'red_count': self.red_count,
            'blue_count': self.blue_count,
            'red_range': self.red_range,
            'blue_range': self.blue_range
        }
        
        info_path = os.path.join(model_dir, 'model_info.json')
        with open(info_path, 'w') as f:
            json.dump(model_info, f, indent=4)
        
        self.log(f"保存模型信息: {info_path}")
        return True
        
    def load_models(self):
        """
        加载保存的模型和缩放器
        
        Returns:
            bool: 是否成功加载模型
        """
        self.log(f"尝试加载{self.lottery_type}的{MODEL_TYPES[self.model_type]}模型...")
        
        # 对于期望值模型，使用特殊的加载方法
        if self.model_type == 'expected_value' and EXPECTED_VALUE_MODEL_AVAILABLE:
            self.log("尝试加载期望值模型...")
            ev_model = ExpectedValueLotteryModel(
                lottery_type=self.lottery_type,
                log_callback=self.log,
                use_gpu=self.use_gpu
            )
            
            load_success = ev_model.load()
            if load_success:
                self.models['red'] = ev_model
                self.models['blue'] = ev_model
                self.raw_models['expected_value_model'] = ev_model
                self.log("期望值模型加载成功")
                return True
            else:
                self.log("期望值模型加载失败")
                return False
        
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
                
            self.feature_window = model_info.get('feature_window', self.feature_window)
            self.log(f"模型信息: 类型={model_info.get('model_type')}, 创建时间={model_info.get('created_at')}")
            
            # 加载特征数量信息
            if 'expected_feature_count' in model_info:
                self.expected_feature_count = model_info['expected_feature_count']
                self.log(f"已加载特征数量信息: {self.expected_feature_count}")
            elif 'n_features_in' in model_info:
                self.expected_feature_count = model_info['n_features_in']
                self.log(f"已加载特征数量信息(从n_features_in): {self.expected_feature_count}")
            
        except Exception as e:
            self.log(f"加载模型信息失败: {e}")
            return False
        
        # 加载X缩放器（通用特征缩放器）
        x_scaler_path = os.path.join(model_dir, 'X_scaler.pkl')
        if os.path.exists(x_scaler_path):
            try:
                with open(x_scaler_path, 'rb') as f:
                    self.scalers['X'] = pickle.load(f)
                self.log(f"加载通用特征缩放器成功")
                # 同时设置红蓝球的缩放器
                self.scalers['red'] = self.scalers['X']
                self.scalers['blue'] = self.scalers['X']
            except Exception as e:
                self.log(f"加载通用特征缩放器失败: {e}")
                # 创建一个默认的缩放器
                self.scalers['X'] = StandardScaler()
                self.scalers['red'] = StandardScaler()
                self.scalers['blue'] = StandardScaler()
                self.log("创建了默认特征缩放器作为替代")
        
        # 加载红球和蓝球模型
        models_loaded = True
        balls_loaded = 0  # 记录加载成功的球数量
        
        for ball_type in ['red', 'blue']:
            try:
                if self.model_type == 'ensemble':
                    # 对于集成模型，加载每个子模型
                    self.models[ball_type] = {}
                    ensemble_loaded = False
                    for model_name in ['random_forest', 'gbdt', 'xgboost', 'lightgbm', 'catboost']:
                        model_path = os.path.join(model_dir, f'{ball_type}_{model_name}_model.pkl')
                        if os.path.exists(model_path):
                            try:
                                with open(model_path, 'rb') as f:
                                    self.models[ball_type][model_name] = pickle.load(f)
                            except (AttributeError, ImportError) as e:
                                if "MT19937" in str(e) or "BitGenerator" in str(e):
                                    self.log(f"检测到numpy.random序列化问题，尝试修复...")
                                    # 修复numpy.random序列化问题
                                    # 确保pickle已经被正确导入
                                    import numpy as np
                                    
                                    # 自定义unpickler来处理MT19937
                                    class CustomUnpickler(pickle.Unpickler):
                                        def find_class(self, module, name):
                                            if module == 'numpy.random._mt19937' and name == 'MT19937':
                                                return np.random.MT19937
                                            return super().find_class(module, name)
                                    
                                    with open(model_path, 'rb') as f:
                                        self.models[ball_type][model_name] = CustomUnpickler(f).load()
                                    self.log(f"numpy.random序列化问题修复成功")
                                else:
                                    raise
                            self.log(f"加载{ball_type}球{model_name}模型成功")
                            
                            # 尝试从模型中获取特征数量
                            if not hasattr(self, 'expected_feature_count'):
                                model_obj = self.models[ball_type][model_name]
                                if hasattr(model_obj, 'n_features_in_'):
                                    self.expected_feature_count = model_obj.n_features_in_
                                    self.log(f"从{model_name}模型中获取特征数量: {self.expected_feature_count}")
                            
                            ensemble_loaded = True
                        else:
                            self.log(f"警告: {ball_type}球{model_name}模型文件不存在")
                    
                    # 加载模型权重（如果存在）
                    weights_path = os.path.join(model_dir, f'{ball_type}_model_weights.json')
                    if os.path.exists(weights_path):
                        try:
                            with open(weights_path, 'r') as f:
                                if not hasattr(self, 'model_weights'):
                                    self.model_weights = {}
                                if ball_type not in self.model_weights:
                                    self.model_weights[ball_type] = {}
                                self.model_weights[ball_type] = json.load(f)
                            self.log(f"加载{ball_type}球模型权重成功")
                            # 打印权重信息
                            for model_name, weight in self.model_weights[ball_type].items():
                                self.log(f"{model_name}模型权重: {weight:.4f}")
                        except Exception as e:
                            self.log(f"加载{ball_type}球模型权重失败: {e}")
                            # 创建默认权重
                            if not hasattr(self, 'model_weights'):
                                self.model_weights = {}
                            if ball_type not in self.model_weights:
                                self.model_weights[ball_type] = {}
                            # 为所有加载的模型设置相等的权重
                            for model_name in self.models[ball_type].keys():
                                self.model_weights[ball_type][model_name] = 1.0 / len(self.models[ball_type])
                            self.log(f"创建了默认{ball_type}球模型权重")
                    else:
                        self.log(f"警告: {ball_type}球模型权重文件不存在，使用均等权重")
                        # 创建默认权重
                        if not hasattr(self, 'model_weights'):
                            self.model_weights = {}
                        if ball_type not in self.model_weights:
                            self.model_weights[ball_type] = {}
                        # 为所有加载的模型设置相等的权重
                        for model_name in self.models[ball_type].keys():
                            self.model_weights[ball_type][model_name] = 1.0 / len(self.models[ball_type])
                    if ensemble_loaded:
                        balls_loaded += 1
                    else:
                        models_loaded = False
                else:
                    # 对于其他模型，直接加载
                    model_path = os.path.join(model_dir, f'{ball_type}_model.pkl')
                    if os.path.exists(model_path):
                        try:
                            # 添加处理numpy.random._mt19937.MT19937序列化问题的代码
                            try:
                                with open(model_path, 'rb') as f:
                                    model_obj = pickle.load(f)
                            except (AttributeError, ImportError) as e:
                                if "MT19937" in str(e) or "BitGenerator" in str(e):
                                    self.log(f"检测到numpy.random序列化问题，尝试修复...")
                                    # 修复numpy.random序列化问题
                                    # 确保pickle已经被正确导入
                                    import io
                                    import numpy as np
                                    
                                    # 自定义unpickler来处理MT19937
                                    class CustomUnpickler(pickle.Unpickler):
                                        def find_class(self, module, name):
                                            if module == 'numpy.random._mt19937' and name == 'MT19937':
                                                return np.random.MT19937
                                            return super().find_class(module, name)
                                    
                                    with open(model_path, 'rb') as f:
                                        model_obj = CustomUnpickler(f).load()
                                    self.log(f"numpy.random序列化问题修复成功")
                                else:
                                    raise
                                
                            # 特别处理 LightGBM 和其他可能返回原始对象而不是包装后对象的模型
                            if self.model_type == 'lightgbm' and hasattr(model_obj, 'predict'):
                                # 对于 LightGBM 和其他带有 predict 方法的库，创建一个包装器
                                if 'lightgbm' in str(type(model_obj)).lower():
                                    self.log(f"检测到原始 LightGBM 模型，创建包装器")
                                    self.models[ball_type] = WrappedLightGBMModel(model_obj, self.process_multidim_prediction)
                                else:
                                    self.models[ball_type] = model_obj
                            else:
                                self.models[ball_type] = model_obj
                                
                            self.log(f"加载{ball_type}球模型成功")
                            balls_loaded += 1
                            
                            # 尝试从模型中获取特征数量
                            if not hasattr(self, 'expected_feature_count'):
                                model_obj_inner = self.models[ball_type]
                                # 如果是包装类，获取内部模型
                                if hasattr(model_obj_inner, 'model'):
                                    model_obj_inner = model_obj_inner.model
                                
                                if hasattr(model_obj_inner, 'n_features_in_'):
                                    self.expected_feature_count = model_obj_inner.n_features_in_
                                    self.log(f"从模型中获取特征数量: {self.expected_feature_count}")
                        except Exception as e:
                            self.log(f"加载{ball_type}球模型失败: {e}, 尝试创建包装器")
                            # 尝试创建适当的包装器
                            try:
                                try:
                                    with open(model_path, 'rb') as f:
                                        raw_model = pickle.load(f)
                                except (AttributeError, ImportError) as e:
                                    if "MT19937" in str(e) or "BitGenerator" in str(e):
                                        self.log(f"检测到numpy.random序列化问题，尝试修复...")
                                        # 修复numpy.random序列化问题
                                        # 确保pickle已经被正确导入
                                        import numpy as np
                                        
                                        # 自定义unpickler来处理MT19937
                                        class CustomUnpickler(pickle.Unpickler):
                                            def find_class(self, module, name):
                                                if module == 'numpy.random._mt19937' and name == 'MT19937':
                                                    return np.random.MT19937
                                                return super().find_class(module, name)
                                        
                                        with open(model_path, 'rb') as f:
                                            raw_model = CustomUnpickler(f).load()
                                        self.log(f"numpy.random序列化问题修复成功")
                                    else:
                                        raise
                                
                                if self.model_type == 'lightgbm':
                                    self.models[ball_type] = WrappedLightGBMModel(raw_model, self.process_multidim_prediction)
                                elif self.model_type == 'catboost':
                                    self.models[ball_type] = WrappedCatBoostModel(raw_model, self.process_multidim_prediction)
                                elif self.model_type == 'xgboost':
                                    self.models[ball_type] = WrappedXGBoostModel(raw_model, self.process_multidim_prediction)
                                elif self.model_type == 'gbdt':
                                    self.models[ball_type] = WrappedGBDTModel(raw_model, self.process_multidim_prediction)
                                else:
                                    # 创建一个通用包装器
                                    class GenericWrapper:
                                        def __init__(self, model, processor):
                                            self.model = model
                                            self.process_prediction = processor
                                        
                                        def predict(self, data):
                                            if hasattr(self.model, 'predict_proba'):
                                                raw_preds = self.model.predict_proba(data)
                                            else:
                                                raw_preds = self.model.predict(data)
                                            return self.process_prediction(raw_preds)
                                    
                                    self.models[ball_type] = GenericWrapper(raw_model, self.process_multidim_prediction)
                                
                                self.log(f"成功为{ball_type}球创建了{self.model_type}包装器")
                                balls_loaded += 1
                            except Exception as e2:
                                self.log(f"创建包装器也失败: {e2}")
                                models_loaded = False
                    else:
                        self.log(f"警告: {ball_type}球模型文件不存在: {model_path}")
                        models_loaded = False
                
                # 尝试加载特定的球特征缩放器(如果还没有通用缩放器的话)
                if ball_type not in self.scalers:
                    scaler_path = os.path.join(model_dir, f'{ball_type}_scaler.pkl')
                    if os.path.exists(scaler_path):
                        try:
                            with open(scaler_path, 'rb') as f:
                                self.scalers[ball_type] = pickle.load(f)
                            self.log(f"加载{ball_type}球特征缩放器成功")
                        except Exception as e:
                            self.log(f"加载{ball_type}球特征缩放器失败: {e}")
                            self.scalers[ball_type] = StandardScaler()
                            self.log(f"创建了{ball_type}球默认特征缩放器作为替代")
                    else:
                        self.log(f"警告: {ball_type}球特征缩放器文件不存在")
                        # 如果没有缩放器，创建一个默认的缩放器
                        self.scalers[ball_type] = StandardScaler()
                        self.log(f"创建了{ball_type}球默认特征缩放器作为替代")
                        
            except Exception as e:
                self.log(f"加载{ball_type}球模型失败: {e}")
                models_loaded = False
        
        # 如果所有模型都成功加载，返回True
        if models_loaded:
            self.log(f"{MODEL_TYPES[self.model_type]}模型加载成功")
            return True
        elif balls_loaded >= 2:  # 至少加载了红球和蓝球
            # 即使有警告，只要基础模型存在，我们也认为模型可用
            self.log(f"{MODEL_TYPES[self.model_type]}模型加载成功，但有一些警告")
            return True
        elif 'red' in self.models and 'blue' in self.models:
            # 模型存在但可能有其他问题
            self.log(f"{MODEL_TYPES[self.model_type]}模型加载成功，但可能存在兼容性问题")
            return True
        else:
            self.log(f"{MODEL_TYPES[self.model_type]}模型加载失败，未找到必要的红球和蓝球模型")
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
        
        # 检查是否使用期望值模型
        if self.model_type == 'expected_value' and EXPECTED_VALUE_MODEL_AVAILABLE:
            # 检查模型是否已经加载
            if ('red' not in self.models or not isinstance(self.models['red'], ExpectedValueLotteryModel) or
                'blue' not in self.models or not isinstance(self.models['blue'], ExpectedValueLotteryModel)):
                # 尝试重新加载期望值模型
                self.log("期望值模型未加载或类型不正确，正在尝试重新加载...")
                ev_model = ExpectedValueLotteryModel(
                    lottery_type=self.lottery_type,
                    log_callback=self.log,
                    use_gpu=self.use_gpu
                )
                
                load_success = ev_model.load()
                if load_success:
                    self.models['red'] = ev_model
                    self.models['blue'] = ev_model
                    self.raw_models['expected_value_model'] = ev_model
                    self.log("期望值模型重新加载成功")
                else:
                    self.log("错误：期望值模型加载失败，请先训练模型")
                    return None, None
            
            # 使用期望值模型进行预测
            self.log("使用期望值模型进行预测...")
            try:
                red_preds, blue_preds = self.models['red'].predict(recent_data, num_predictions=1)
                # 期望值模型返回的是索引列表的列表，需要处理成号码
                if not red_preds or not blue_preds:
                    raise ValueError("期望值模型返回了空的预测结果")
                    
                red_numbers = [idx + 1 for idx in red_preds[0]]  # 索引转换为号码
                blue_numbers = [idx + 1 for idx in blue_preds[0]]
                
                # 确保红球和蓝球号码数量符合要求
                red_numbers = sorted(list(set(red_numbers)))[:self.red_count]
                blue_numbers = sorted(list(set(blue_numbers)))[:self.blue_count]
                
                # 如果数量不足，补充随机号码
                while len(red_numbers) < self.red_count:
                    new_num = np.random.randint(1, self.red_range + 1)
                    if new_num not in red_numbers:
                        red_numbers.append(new_num)
                red_numbers.sort()
                        
                while len(blue_numbers) < self.blue_count:
                    new_num = np.random.randint(1, self.blue_range + 1)
                    if new_num not in blue_numbers:
                        blue_numbers.append(new_num)
                blue_numbers.sort()
                
                return red_numbers, blue_numbers
            except Exception as e:
                self.log(f"期望值模型预测失败: {str(e)}")
                raise ValueError(f"期望值模型预测失败: {str(e)}")
        
        # 确保模型已加载
        if 'red' not in self.models or 'blue' not in self.models:
            self.log("模型未加载，无法预测")
            raise ValueError(f"模型未正确加载，请先训练或加载模型。")
        
        # 对于其他模型的处理保持不变
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
        
        # 获取模型详细信息以进行调试
        red_model_info = ""
        if 'red' in self.models:
            if self.model_type == 'ensemble':
                if 'random_forest' in self.models['red']:
                    red_model_info = f"特征数量: {getattr(self.models['red']['random_forest'], 'n_features_in_', '未知')}, 类型: {type(self.models['red']['random_forest']).__name__}"
            else:
                red_model_info = f"特征数量: {getattr(self.models['red'], 'n_features_in_', '未知')}, 类型: {type(self.models['red']).__name__}"
        self.log(f"红球模型信息: {red_model_info}")
        
        # 检查特征数量是否与训练时的特征数量匹配
        # 从模型信息或模型本身获取预期的特征数量
        expected_features = getattr(self, 'expected_feature_count', 70)  # 默认70，与训练时保持一致
        
        # 也可以尝试从模型中获取
        if 'red' in self.models:
            model_obj = self.models['red']
            # 处理不同类型的模型
            if self.model_type == 'ensemble' and 'random_forest' in model_obj:
                model_obj = model_obj['random_forest']
                
            if hasattr(model_obj, 'n_features_in_'):
                expected_features = model_obj.n_features_in_
                self.log(f"从模型中获取的特征数量: {expected_features}")
            elif hasattr(model_obj, 'feature_importances_') and hasattr(model_obj.feature_importances_, 'shape'):
                expected_features = model_obj.feature_importances_.shape[0]
                self.log(f"从模型特征重要性中获取的特征数量: {expected_features}")
            # 尝试获取更多的模型属性
            elif hasattr(model_obj, 'estimators_') and model_obj.estimators_:
                first_estimator = model_obj.estimators_[0]
                if hasattr(first_estimator, 'n_features_in_'):
                    expected_features = first_estimator.n_features_in_
                    self.log(f"从模型第一个估计器中获取的特征数量: {expected_features}")
        
        actual_features = X_reshaped.shape[1]
        
        if actual_features != expected_features:
            self.log(f"警告: 特征数量不匹配，预期{expected_features}个，实际{actual_features}个")
            # 根据情况补充或截断特征
            if actual_features < expected_features:
                # 如果特征不足，填充零
                padding = np.zeros((X_reshaped.shape[0], expected_features - actual_features))
                X_reshaped = np.concatenate([X_reshaped, padding], axis=1)
                self.log(f"已将特征填充至{X_reshaped.shape[1]}个")
            else:
                # 如果特征过多，截断
                X_reshaped = X_reshaped[:, :expected_features]
                self.log(f"已将特征截断至{X_reshaped.shape[1]}个")
        
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
            if self.model_type == 'ensemble':
                # 集成模型需要单独处理
                if not isinstance(self.models['red'], dict) or not self.models['red']:
                    self.log("错误：集成模型未正确加载，红球模型为空")
                    raise ValueError("集成模型未正确加载，红球模型为空。请重新训练模型。")
                red_votes = {}
                for name, model in self.models['red'].items():
                    preds = model.predict(X_scaled)[0]
                    if hasattr(preds, "__iter__"):
                        for pred in preds:
                            if pred not in red_votes:
                                red_votes[pred] = 0
                            red_votes[pred] += 1
                    else:
                        # 单一预测值
                        if preds not in red_votes:
                            red_votes[preds] = 0
                        red_votes[preds] += 1
                red_predictions = sorted(red_votes.items(), key=lambda x: x[1], reverse=True)[:self.red_count]
                red_predictions = [p[0] + 1 for p in red_predictions]  # +1 转回原始号码范围
            else:
                # 单一模型预测
                if 'red' not in self.models or not hasattr(self.models['red'], 'predict'):
                    self.log("错误：红球模型未正确加载或缺少predict方法")
                    raise ValueError("红球模型未正确加载或缺少predict方法。请重新训练模型。")
                red_pred = self.models['red'].predict(X_scaled)[0]
                if hasattr(red_pred, "__iter__"):
                    red_predictions = [int(p) + 1 for p in red_pred]  # +1 转回原始号码范围
                else:
                    red_predictions = [int(red_pred) + 1]  # +1 转回原始号码范围
            
            # 预测蓝球
            if self.model_type == 'ensemble':
                # 集成模型需要单独处理
                if not isinstance(self.models['blue'], dict) or not self.models['blue']:
                    self.log("错误：集成模型未正确加载，蓝球模型为空")
                    raise ValueError("集成模型未正确加载，蓝球模型为空。请重新训练模型。")
                blue_votes = {}
                for name, model in self.models['blue'].items():
                    preds = model.predict(X_scaled)[0]
                    if hasattr(preds, "__iter__"):
                        for pred in preds:
                            if pred not in blue_votes:
                                blue_votes[pred] = 0
                            blue_votes[pred] += 1
                    else:
                        # 单一预测值
                        if preds not in blue_votes:
                            blue_votes[preds] = 0
                        blue_votes[preds] += 1
                
                # 从得票前3的蓝球中随机选择，而不是总是选择得票最高的
                top_blue_votes = sorted(blue_votes.items(), key=lambda x: x[1], reverse=True)
                top_count = min(3, len(top_blue_votes))
                
                # 有60%概率使用票数最高的蓝球，40%概率从票数前3的蓝球中随机选择
                if np.random.random() < 0.6 or top_count == 1:
                    blue_predictions = [p[0] + 1 for p in top_blue_votes[:self.blue_count]]  # +1 转回原始号码范围
                else:
                    # 随机选择前top_count个中的blue_count个
                    selected_indices = np.random.choice(top_count, size=min(self.blue_count, top_count), replace=False)
                    blue_predictions = [top_blue_votes[i][0] + 1 for i in selected_indices]  # +1 转回原始号码范围
            else:
                # 单一模型预测
                if 'blue' not in self.models or not hasattr(self.models['blue'], 'predict'):
                    self.log("错误：蓝球模型未正确加载或缺少predict方法")
                    raise ValueError("蓝球模型未正确加载或缺少predict方法。请重新训练模型。")
                blue_pred = self.models['blue'].predict(X_scaled)[0]
                if hasattr(blue_pred, "__iter__"):
                    # 随机设定阈值，增加随机性
                    if np.random.random() < 0.3 and len(blue_pred) > 1:
                        # 30%的概率，从前3个最高概率蓝球中随机选择
                        top_indices = np.argsort(blue_pred)[-3:] if len(blue_pred) >= 3 else np.argsort(blue_pred)
                        selected_idx = np.random.choice(top_indices)
                        blue_predictions = [int(selected_idx) + 1]  # +1 转回原始号码范围
                    else:
                        # 70%的概率，使用原始预测
                        blue_predictions = [int(p) + 1 for p in blue_pred]  # +1 转回原始号码范围
                else:
                    # 直接使用并添加随机性
                    if np.random.random() < 0.25:  # 25%概率使用随机蓝球而不是模型预测
                        # 根据彩票类型确定蓝球范围
                        blue_range = self.blue_range
                        blue_predictions = [np.random.randint(1, blue_range + 1)]
                    else:
                        blue_predictions = [int(blue_pred) + 1]  # +1 转回原始号码范围
        except Exception as e:
            self.log(f"预测过程中出错: {e}")
            import traceback
            self.log(traceback.format_exc())
            # 不再返回随机号码，而是抛出异常
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
        
        # 增加随机性：有5%的概率完全随机生成一个蓝球号码
        if np.random.random() < 0.05:
            self.log("随机生成蓝球号码以增加多样性")
            blue_predictions = []
            for _ in range(self.blue_count):
                blue_predictions.append(np.random.randint(1, self.blue_range + 1))
            blue_predictions = sorted(list(set(blue_predictions)))
            
            # 如果随机生成后数量不足，继续随机补充
            while len(blue_predictions) < self.blue_count:
                new_num = np.random.randint(1, self.blue_range + 1)
                if new_num not in blue_predictions:
                    blue_predictions.append(new_num)
            blue_predictions = sorted(blue_predictions)[:self.blue_count]
        
        return red_predictions, blue_predictions

# 使用示例
def demo():
    # 加载数据
    from scripts.data_analysis import load_lottery_data
    lottery_type = 'dlt'  # 或 'ssq'
    df = load_lottery_data(lottery_type)
    
    # 初始化模型
    model = LotteryMLModels(lottery_type=lottery_type, model_type='ensemble')
    
    # 训练模型
    model.train(df)
    
    # 预测下一期号码
    recent_data = df.sort_values('期数', ascending=False).head(10)
    red_predictions, blue_predictions = model.predict(recent_data)
    
    print(f"预测红球号码: {red_predictions}")
    print(f"预测蓝球号码: {blue_predictions}")

if __name__ == "__main__":
    demo()