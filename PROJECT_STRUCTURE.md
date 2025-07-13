# LottoProphet 项目结构说明

本文档描述了 LottoProphet 彩票预测项目的目录结构和模块组织方式。

## 项目根目录

```
LottoProphet/
├── src/                    # 源代码目录
├── data/                   # 原始数据目录
├── models/                 # 训练好的模型文件
├── scripts/                # 独立脚本文件
├── tests/                  # 测试文件
├── docs/                   # 文档目录
├── update/                 # 更新日志
├── venv/                   # 虚拟环境
├── main.py                 # 主程序入口
├── requirements.txt        # 依赖包列表
└── README.md              # 项目说明
```

## src/ 源代码目录结构

### core/ - 核心功能模块
- `model.py` - LSTM-CRF 神经网络模型定义
- `expected_value.py` - 期望值模型实现
- `prediction.py` - 预测结果处理工具
- `__init__.py` - 模块导入配置

### data/ - 数据处理模块
- `analysis.py` - 数据分析功能（原 scripts/data_analysis.py）
- `processing.py` - 数据处理工具（原 data_processing.py）
- `statistics.py` - 高级统计分析（原 scripts/advanced_statistics.py）
- `fetchers/` - 数据获取脚本目录
  - `fetch_and_train.py` - 数据获取和训练脚本
- `__init__.py` - 模块导入配置

### models/ - 机器学习模型
- `ml_models.py` - 多种机器学习模型实现
- `trainers/` - 模型训练脚本目录
  - `train_models.py` - 模型训练脚本
- `__init__.py` - 模块导入配置

### ui/ - 用户界面模块
- `main_app.py` - 主应用程序（原 lottery_predictor_app_new.py）
- `components.py` - UI 组件（原 ui_components.py）
- `theme_manager.py` - 主题管理器
- `controllers/` - 控制器目录（预留）
- `tabs/` - 标签页组件目录（预留）
- `__init__.py` - 模块导入配置

### utils/ - 工具模块
- `model_utils.py` - 模型工具函数
- `thread_utils.py` - 线程工具
- `__init__.py` - 模块导入配置

## 模块导入说明

每个子目录的 `__init__.py` 文件都配置了相应的导入语句，可以通过以下方式使用：

```python
# 核心功能
from src.core import LstmCRFModel, ExpectedValueLotteryModel, process_predictions

# 数据处理
from src.data import load_lottery_data, calculate_statistics, process_analysis_data

# 机器学习模型
from src.models import LotteryMLModels, MODEL_TYPES

# UI 组件
from src.ui import ThemeManager, create_main_tab

# 工具函数
from src.utils import load_pytorch_model, TrainModelThread
```

## 主要改进

1. **模块化组织**：将相关功能按照职责分组到不同目录
2. **清晰的层次结构**：核心功能、数据处理、模型、UI、工具分离
3. **统一的导入接口**：通过 `__init__.py` 提供清晰的模块导入
4. **减少根目录混乱**：将所有源代码移动到 `src/` 目录下
5. **保持向后兼容**：主要入口文件 `main.py` 保留在根目录

## 使用建议

- 新功能开发时，请按照模块职责将代码放入相应目录
- 导入模块时优先使用 `src.*` 的方式
- 保持各模块的独立性，避免循环导入
- 在添加新模块时，记得更新对应的 `__init__.py` 文件