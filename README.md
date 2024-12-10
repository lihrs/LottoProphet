# LottoProphet


一个使用深度学习模型进行彩票号码预测的应用程序。本项目支持两种主要的彩票类型：双色球 (SSQ) 和 大乐透 (DLT)，并使用先进的机器学习技术（如条件随机场 CRF）进行序列建模。

功能
支持双色球 (SSQ) 和大乐透 (DLT) 的彩票号码预测。
使用 LSTM 和 CRF 模型进行训练，实现序列化建模。
提供基于 PyQt5 的图形用户界面 (GUI)，便于操作。
支持数据自动抓取和实时训练日志显示。
集成 Git，用于版本控制和协作开发。
环境要求
Python 3.9 或更高版本
PyTorch
torchcrf
PyQt5
pandas
numpy
scikit-learn
安装步骤
克隆仓库：

bash
Copy code
git clone git@github.com:zhaoyangpp/LottoProphet.git
cd LottoProphet
创建并激活虚拟环境：

bash
Copy code
python -m venv venv
source venv/bin/activate   # Windows 下：venv\Scripts\activate
安装依赖：

bash
Copy code
pip install -r requirements.txt
使用方法
1. 训练模型
运行训练脚本，选择需要训练的彩票类型：

bash
Copy code
# 对于双色球
python scripts/train_ssq_model.py

# 对于大乐透
python scripts/train_dlt_model.py
2. 运行 GUI 应用
启动图形界面应用：

bash
Copy code
python scripts/lottery_predictor_app.py
通过 GUI，您可以：

训练模型。
预测彩票号码。
查看实时的训练和预测日志。
开发流程
设置 Git 仓库
初始化仓库：

bash
Copy code
git init
添加远程仓库：

bash
Copy code
git remote add origin git@github.com:zhaoyangpp/LottoProphet.git
提交并推送更改：

bash
Copy code
git add .
git commit -m "首次提交"
git push -u origin main
常用 Git 命令
查看远程仓库：

bash
Copy code
git remote -v
拉取最新更改：

bash
Copy code
git pull origin main --rebase
推送本地更改：

bash
Copy code
git push origin master:main
测试 SSH 连接
确保 SSH 密钥配置正确：

bash
Copy code
ssh -T git@github.com
常见问题
错误：ModuleNotFoundError: No module named 'torchcrf'

确保已安装 torchcrf：

bash
Copy code
pip install torchcrf
错误：No matching distribution found for PyTorch

根据系统安装兼容的 PyTorch 版本：

bash
Copy code
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
错误：Git Push Rejected

如果 Git 推送被拒绝，请同步您的仓库：

bash
Copy code
git pull origin main --rebase
git push origin master:main
授权协议
本项目基于 MIT 协议开源。

作者信息
Zhaoyangpp
邮箱：zhaoyangpp@gmail.com
此 README.md 提供了详细的中文指南，方便开发者快速上手或协作开发。






