## 一、目录结构优化
### 1. 建立清晰的模块化目录结构
LottoProphet/
├── src/                           # 源代码目录
│   ├── core/                      # 核心功能模块
│   │   ├── __init__.py
│   │   ├── model.py               # 模型定义（从原model.py移动）
│   │   ├── prediction.py          # 预测功能（从prediction_utils.py移动）
│   │   └── expected_value.py      # 期望值模型（从expected_value_model.py移动）
│   ├── data/                      # 数据处理模块
│   │   ├── __init__.py
│   │   ├── processing.py          # 数据处理（从data_processing.py移动）
│   │   ├── analysis.py            # 基础分析功能（从scripts/data_analysis.py移动）
│   │   ├── statistics.py          # 高级统计（从scripts/advanced_statistics.py移动）
│   │   ├── fetchers/              # 数据获取模块
│   │   │   ├── __init__.py
│   │   │   ├── ssq_fetcher.py     # 双色球数据获取（从scripts/ssq/fetch_ssq_data.py移动）
│   │   │   └── dlt_fetcher.py     # 大乐透数据获取（从scripts/dlt/fetch_dlt_data.py移动）
│   ├── models/                    # 模型实现模块
│   │   ├── __init__.py
│   │   ├── base.py                # 基础模型接口
│   │   ├── lstm_crf.py            # LSTM-CRF模型实现
│   │   ├── ml_models.py           # 机器学习模型（从ml_models.py拆分）
│   │   ├── trainers/              # 模型训练模块
│   │   │   ├── __init__.py
│   │   │   ├── ssq_trainer.py     # 双色球训练器（从scripts/ssq/train_ssq_model.py移动）
│   │   │   └── dlt_trainer.py     # 大乐透训练器（从scripts/dlt/train_dlt_model.py移动）
│   ├── ui/                        # 用户界面模块
│   │   ├── __init__.py
│   │   ├── components.py          # UI组件（从ui_components.py移动）
│   │   ├── theme.py               # 主题管理（从theme_manager.py移动）
│   │   └── app.py                 # 主应用程序（从lottery_predictor_app_new.py移动）
│   └── utils/                     # 工具模块
│       ├── __init__.py
│       ├── config.py              # 配置管理
│       ├── logging.py             # 日志工具
│       └── threading.py           # 线程工具（从thread_utils.py移动）
├── scripts/                       # 脚本目录
│   ├── build.py                   # 构建脚本（从build.py移动）
│   └── convert_icon.py            # 图标转换脚本
├── data/                          # 数据目录
│   ├── ssq/                       # 双色球数据
│   │   └── ssq_history.csv        # 双色球历史数据
│   └── dlt/                       # 大乐透数据
│       └── dlt_history.csv        # 大乐透历史数据
├── models/                        # 模型存储目录
│   ├── ssq/                       # 双色球模型
│   └── dlt/                       # 大乐透模型
├── tests/                         # 测试目录
│   ├── __init__.py
│   ├── test_data_processing.py    # 数据处理测试
│   └── test_models.py             # 模型测试
├── docs/                          # 文档目录
│   ├── README.md                  # 项目说明
│   └── updates/                   # 更新文档
├── main.py                        # 主入口点
└── requirements.txt               # 项目依赖

## 二、大文件拆分方案
### 1. 拆分 ml_models.py（2531行）
将这个大文件拆分为多个专注于特定功能的模块：

- src/models/base.py - 基础模型接口和通用功能
- src/models/random_forest.py - 随机森林模型实现
- src/models/xgboost_model.py - XGBoost模型实现
- src/models/lightgbm_model.py - LightGBM模型实现
- src/models/catboost_model.py - CatBoost模型实现
- src/models/ensemble.py - 集成模型实现

### 2. 拆分 lottery_predictor_app_new.py（1278行）
将这个大文件拆分为多个UI组件和控制器：

- src/ui/app.py - 主应用程序框架
- src/ui/tabs/main_tab.py - 主标签页实现
- src/ui/tabs/analysis_tab.py - 分析标签页实现
- src/ui/tabs/expected_value_tab.py - 期望值模型标签页实现
- src/ui/tabs/advanced_stats_tab.py - 高级统计标签页实现
- src/ui/controllers/prediction_controller.py - 预测控制器
- src/ui/controllers/training_controller.py - 训练控制器
- src/ui/controllers/analysis_controller.py - 分析控制器

## 三、功能模块化改进
### 1. 数据获取模块化
创建统一的数据获取接口，使不同彩票类型的数据获取逻辑更加一致：
# src/data/fetchers/__init__.py
from abc import ABC, abstractmethod

class DataFetcher(ABC):
    """数据获取基类"""
    
    @abstractmethod
    def fetch_data(self):
        """获取数据"""
        pass
        
    @abstractmethod
    def save_data(self, data, path):
        """保存数据"""
        pass

### 2. 模型训练模块化
创建统一的模型训练接口，使不同彩票类型的模型训练逻辑更加一致：
# src/models/trainers/__init__.py
from abc import ABC, abstractmethod

class ModelTrainer(ABC):
    """模型训练器基类"""
    
    @abstractmethod
    def load_data(self):
        """加载数据"""
        pass
        
    @abstractmethod
    def preprocess_data(self, data):
        """预处理数据"""
        pass
        
    @abstractmethod
    def train(self, X_train, y_train):
        """训练模型"""
        pass
        
    @abstractmethod
    def save_model(self, model, path):
        """保存模型"""
        pass

### 3. 配置管理
创建统一的配置管理模块，集中管理所有配置信息：
# src/utils/config.py
class Config:
    """配置管理类"""
    
    # 彩票类型配置
    LOTTERY_TYPES = {
        "ssq": {
            "name": "双色球",
            "red_range": 33,
            "blue_range": 16,
            "red_count": 6,
            "blue_count": 1
        },
        "dlt": {
            "name": "大乐透",
            "red_range": 35,
            "blue_range": 12,
            "red_count": 5,
            "blue_count": 2
        }
    }
    
    # 路径配置
    DATA_DIR = "data"
    MODEL_DIR = "models"
    
    # 模型配置
    MODEL_TYPES = {
        "lstm_crf": "LSTM-CRF",
        "random_forest": "随机森林",
        "xgboost": "XGBoost",
        "lightgbm": "LightGBM",
        "catboost": "CatBoost",
        "ensemble": "集成模型",
        "expected_value": "期望值模型"
    }