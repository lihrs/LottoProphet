# LottoProphet 安装指南

## 方法一：直接运行源代码

### 前提条件

- Python 3.8 或更高版本
- pip 包管理器

### 安装步骤

1. 克隆或下载项目代码

```bash
git clone https://github.com/zhaoyangpp/LottoProphet.git
cd LottoProphet
```

2. 安装依赖

```bash
pip install -r requirements.txt
```

3. 运行程序

```bash
python main.py
```

## 方法二：使用构建好的可执行文件

### Windows

1. 下载最新的 `LottoProphet.exe` 文件
2. 双击运行即可

### macOS

1. 下载最新的 `LottoProphet.dmg` 文件
2. 双击打开 DMG 文件
3. 将 LottoProphet 应用拖到 Applications 文件夹
4. 从启动器或 Applications 文件夹启动应用

### Linux

1. 下载最新的 `LottoProphet` 可执行文件
2. 添加执行权限：`chmod +x LottoProphet`
3. 运行程序：`./LottoProphet`

## 方法三：自行构建可执行文件

项目提供了构建脚本，可以将源代码打包为可执行文件。详细说明请参考 [BUILD.md](BUILD.md)。

### 快速构建命令

```bash
# 安装 PyInstaller
pip install pyinstaller

# 使用构建脚本构建
python build.py
```

## 常见问题

### 运行时缺少数据或模型文件

首次运行时，程序会自动下载必要的数据并训练模型。如果遇到问题，可以尝试手动运行以下命令：

```bash
python fetch_and_train.py
```

### 图形界面无法显示

确保已正确安装 PyQt5：

```bash
pip install PyQt5
```

### 模型训练失败

确保已安装所有机器学习相关依赖：

```bash
pip install torch torchvision torchaudio torchcrf scikit-learn xgboost lightgbm catboost
```

## 系统要求

- **操作系统**：Windows 10/11、macOS 10.15+、Linux（主流发行版）
- **处理器**：双核处理器或更高
- **内存**：4GB RAM（推荐 8GB 或更高）
- **存储空间**：至少 500MB 可用空间
- **Python**：3.8 或更高版本（如果从源代码运行）

## 更新说明

程序会定期检查更新。您也可以通过以下方式手动更新：

1. 从源代码运行：`git pull` 获取最新代码
2. 使用可执行文件：下载并替换为最新版本

## 支持与反馈

如有问题或建议，请通过以下方式联系我们：

- GitHub Issues: [https://github.com/zhaoyangpp/LottoProphet/issues](https://github.com/zhaoyangpp/LottoProphet/issues)
- 电子邮件：[example@example.com](mailto:example@example.com)（请替换为实际联系邮箱）