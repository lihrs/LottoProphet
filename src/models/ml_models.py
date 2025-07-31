# -*- coding:utf-8 -*-
"""
兼容层 - 为了保持向后兼容性
这个文件导入并重新导出所有从原始ml_models.py拆分出来的类
"""

import logging

# 导入所有拆分后的模型类
from .base import BaseMLModel, MODEL_TYPES
from .random_forest import RandomForestModel
from .xgboost_model import XGBoostModel, WrappedXGBoostModel
from .lstm_timeStep import LSTMTimeStepModel

# 条件导入LightGBM和CatBoost
try:
    from .lightgbm_model import LightGBMModel, WrappedLightGBMModel, LIGHTGBM_AVAILABLE
except ImportError:
    LIGHTGBM_AVAILABLE = False
    logging.warning("LightGBM not available. Some features will be disabled.")

try:
    from .catboost_model import CatBoostModel, WrappedCatBoostModel, CATBOOST_AVAILABLE
except ImportError:
    CATBOOST_AVAILABLE = False
    logging.warning("CatBoost not available. Some features will be disabled.")

from .ensemble import EnsembleModel

# 为了向后兼容，重新导出原始类名
class LotteryMLModels(BaseMLModel):
    """
    兼容层 - 保持与原始LotteryMLModels类的接口兼容
    这个类将根据model_type参数选择并实例化相应的具体模型类
    """
    
    def __init__(self, lottery_type='dlt', model_type='random_forest', feature_window=10, log_callback=None, use_gpu=False):
        """
        初始化模型
        
        Args:
            lottery_type: 彩票类型，'dlt'或'ssq'
            model_type: 模型类型，'random_forest', 'xgboost', 'lightgbm', 'catboost', 'ensemble'
            feature_window: 特征窗口大小
            log_callback: 日志回调函数
            use_gpu: 是否使用GPU
        """
        super().__init__(lottery_type, feature_window, log_callback, use_gpu)
        self.model_type = model_type
        
        # 根据model_type选择具体模型类
        if model_type == 'random_forest':
            self.model = RandomForestModel(lottery_type, feature_window, log_callback, use_gpu)
        elif model_type == 'lstm_timestep':
            # 为LSTM模型设置适当的参数
            self.model = LSTMTimeStepModel(lottery_type, feature_window, log_callback, use_gpu)
        elif model_type == 'xgboost':
            self.model = XGBoostModel(lottery_type, feature_window, log_callback, use_gpu)
        elif model_type == 'gbdt':
            # GBDT模型使用XGBoost实现
            self.model = XGBoostModel(lottery_type, feature_window, log_callback, use_gpu)
        elif model_type == 'lightgbm':
            if LIGHTGBM_AVAILABLE:
                self.model = LightGBMModel(lottery_type, feature_window, log_callback, use_gpu)
            else:
                raise ImportError("LightGBM is not available. Please install lightgbm package.")
        elif model_type == 'catboost':
            if CATBOOST_AVAILABLE:
                self.model = CatBoostModel(lottery_type, feature_window, log_callback, use_gpu)
            else:
                raise ImportError("CatBoost is not available. Please install catboost package.")
        elif model_type == 'ensemble':
            self.model = EnsembleModel(lottery_type, feature_window, log_callback, use_gpu)
        elif model_type == 'expected_value':
            # 直接使用BaseMLModel，因为expected_value模型在predict方法中有特殊处理
            pass
        else:
            raise ValueError(f"Unsupported model type: {model_type}. Supported types: {list(MODEL_TYPES.keys())}")
    
    def train(self, df):
        """
        训练模型
        
        Args:
            df: 包含历史开奖数据的DataFrame
            
        Returns:
            训练好的模型
        """
        if self.model_type == 'expected_value':
            self.log("Expected value model does not require training.")
            return None
        return self.model.train(df)
    
    def predict(self, recent_data):
        """
        生成预测结果
        
        Args:
            recent_data: 包含最近开奖数据的DataFrame
            
        Returns:
            预测的红球和蓝球号码
        """
        if self.model_type == 'expected_value':
            # 特殊处理expected_value模型
            from src.core.expected_value import ExpectedValueLotteryModel
            ev_model = ExpectedValueLotteryModel(self.lottery_type)
            return ev_model.predict()
        return self.model.predict(recent_data)
    
    def load_models(self):
        """
        加载保存的模型
        
        Returns:
            bool: 是否成功加载模型
        """
        if self.model_type == 'expected_value':
            return True  # expected_value模型不需要加载
        return self.model.load_models()
    
    def save_models(self):
        """
        保存模型
        """
        if self.model_type == 'expected_value':
            self.log("Expected value model does not require saving.")
            return
        self.model.save_models()
    
    def evaluate_model(self, X_test, red_test_data, blue_test_data):
        """
        评估模型
        
        Args:
            X_test: 测试特征
            red_test_data: 红球测试数据
            blue_test_data: 蓝球测试数据
            
        Returns:
            评估结果
        """
        if self.model_type == 'expected_value':
            self.log("Expected value model does not support evaluation.")
            return None
        return self.model.evaluate_model(X_test, red_test_data, blue_test_data)

# 为了向后兼容，重新导出原始类和函数
WrappedGBDTModel = WrappedXGBoostModel  # 在原代码中这两个是相同的

# 导出一个demo函数，与原始ml_models.py中的demo函数保持一致
def demo():
    """
    演示如何使用模型
    """
    import pandas as pd
    import os
    
    # 加载数据
    lottery_type = 'dlt'
    data_path = os.path.join('data', lottery_type, f'{lottery_type}_history.csv')
    df = pd.read_csv(data_path)
    
    # 创建模型
    model = LotteryMLModels(lottery_type=lottery_type, model_type='random_forest')
    
    # 训练模型
    model.train(df)
    
    # 预测
    predictions = model.predict(df.head(10))
    print(f"预测结果: {predictions}")
    
    return model

# 如果直接运行此文件，执行demo
if __name__ == "__main__":
    demo()