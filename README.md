# LottoProphet

一个使用深度学习模型进行彩票号码预测的应用程序。本项目支持两种主要的彩票类型：**双色球 (SSQ)** 和 **大乐透 (DLT)**，并使用先进的机器学习技术（如条件随机场 CRF）进行序列建模。

---

## 功能

- 支持双色球 (SSQ) 和大乐透 (DLT) 的彩票号码预测。
- 使用 LSTM 和 CRF 模型进行训练，实现序列化建模。
- 提供基于 PyQt5 的图形用户界面 (GUI)，便于操作。
- 支持数据自动抓取和实时训练日志显示。
---

## 环境要求

- Python 3.9 或更高版本
- PyTorch
- torchcrf
- PyQt5
- pandas
- numpy
- scikit-learn

---

## 安装步骤

1. **克隆仓库**：
   ```bash
   git clone git@github.com:zhaoyangpp/LottoProphet.git
   cd LottoProphet
