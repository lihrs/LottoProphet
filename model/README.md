# 模型目录结构

这个目录包含了训练好的模型文件和相关资源，用于彩票预测。

## 目录结构

```
model/
├── dlt/            # 大乐透模型文件
│   ├── dlt_model.pth    # 组合模型文件（包含红球和蓝球模型）
│   └── scaler_X.pkl     # 特征缩放器
├── ssq/            # 双色球模型文件
│   ├── ssq_model.pth    # 组合模型文件（包含红球和蓝球模型）
│   └── scaler_X.pkl     # 特征缩放器
├── generate_placeholder_models.py  # 生成占位符模型脚本
├── test_model_loading.py          # 测试模型加载脚本
└── README.md       # 本文件
```

## 模型文件格式

每个模型文件（.pth）包含以下内容：

```python
{
    'red_model': red_model.state_dict(),   # 红球模型状态
    'blue_model': blue_model.state_dict(), # 蓝球模型状态
    'metadata': {                          # 元数据
        'timestamp': timestamp,            # 训练时间戳
        'input_dim': input_dim,            # 输入维度
        'hidden_dim': hidden_dim,          # 隐藏层维度
        'red_output_dim': red_output_dim,  # 红球输出维度
        'blue_output_dim': blue_output_dim,# 蓝球输出维度
        'red_seq_length': red_seq_length,  # 红球序列长度
        'blue_seq_length': blue_seq_length # 蓝球序列长度
    }
}
```

## 使用方法

模型加载和预测功能在 `model_utils.py` 文件中实现。请参考该文件中的 `load_resources_pytorch` 和相关函数。

## 模型训练

要训练新的模型，可以使用以下两种方式：

### 方式一：使用 train_models.py（仅训练）

```bash
# 训练所有模型（大乐透和双色球）
python train_models.py

# 只训练大乐透模型
python train_models.py --type dlt

# 只训练双色球模型
python train_models.py --type ssq

# 使用GPU训练并指定轮数
python train_models.py --gpu --epochs 200
```

### 方式二：使用 fetch_and_train.py（数据获取+训练）

此脚本会自动完成数据获取和模型训练的完整流程：

```bash
# 获取数据并训练所有模型
python fetch_and_train.py

# 只处理大乐透
python fetch_and_train.py --type dlt

# 只处理双色球
python fetch_and_train.py --type ssq

# 使用GPU训练并指定轮数
python fetch_and_train.py --gpu --epochs 200

# 跳过数据获取步骤，仅训练模型
python fetch_and_train.py --skip-fetch
```

训练完成后，模型文件将自动保存到对应的目录中。 