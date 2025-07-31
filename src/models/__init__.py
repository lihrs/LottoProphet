# -*- coding: utf-8 -*-
"""
Machine learning models
"""

# 导出基础模型接口
from .base import BaseLotteryModel, BaseMLModel, MODEL_TYPES

# 导出各种具体模型实现
from .random_forest import RandomForestModel
from .xgboost_model import XGBoostModel, WrappedXGBoostModel

# 条件导入LightGBM和CatBoost
try:
    from .lightgbm_model import LightGBMModel, WrappedLightGBMModel, LIGHTGBM_AVAILABLE
except ImportError:
    LIGHTGBM_AVAILABLE = False

try:
    from .catboost_model import CatBoostModel, WrappedCatBoostModel, CATBOOST_AVAILABLE
except ImportError:
    CATBOOST_AVAILABLE = False

# 导出集成模型
from .ensemble import EnsembleModel

# 导出兼容层
from .ml_models import LotteryMLModels, WrappedGBDTModel, demo

# 导出LSTM-CRF模型
from .lstm_crf import LstmCRFModel as LSTMCRFModel

# 导出LSTM-TimeStep模型
from .lstm_timeStep import LSTMTimeStepModel

__all__ = [
    # 基础模型接口
    'BaseLotteryModel',
    'BaseMLModel',
    'MODEL_TYPES',
    
    # 具体模型实现
    'RandomForestModel',
    'XGBoostModel',
    'WrappedXGBoostModel',
    'LightGBMModel',
    'WrappedLightGBMModel',
    'CatBoostModel',
    'WrappedCatBoostModel',
    'EnsembleModel',
    'LSTMCRFModel',
    'LSTMTimeStepModel',
    
    # 兼容层
    'LotteryMLModels',
    'WrappedGBDTModel',
    'demo',
    
    # 可用性标志
    'LIGHTGBM_AVAILABLE',
    'CATBOOST_AVAILABLE'
]