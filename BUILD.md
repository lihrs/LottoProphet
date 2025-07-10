# LottoProphet 构建指南

本文档提供了如何将 LottoProphet 项目编译为可执行文件和安装包的详细说明。

## 前提条件

- Python 3.8 或更高版本
- pip 包管理器
- 已安装项目依赖（在 requirements.txt 中列出）

## 构建方法

### 方法一：使用构建脚本（推荐）

项目提供了一个自动化构建脚本 `build.py`，可以简化构建过程。

1. 打开终端或命令提示符，进入项目根目录
2. 运行构建脚本：

```bash
python build.py
```

#### 构建选项

构建脚本支持以下命令行选项：

- `--onedir`：构建为目录而非单文件（默认为单文件模式）
- `--debug`：启用调试模式，保留控制台窗口（默认隐藏控制台）
- `--skip-installer`：跳过创建安装程序/DMG（仅适用于 Windows/macOS）

示例：

```bash
# 构建为目录模式并启用调试
python build.py --onedir --debug

# 构建单文件但跳过创建安装程序/DMG
python build.py --skip-installer
```

### 方法二：直接使用 PyInstaller

如果您需要更精细的控制，可以直接使用 PyInstaller：

1. 确保已安装 PyInstaller：

```bash
pip install pyinstaller
```

2. 使用提供的 spec 文件构建：

```bash
pyinstaller lottoprophet.spec
```

或者自定义构建命令：

```bash
pyinstaller --clean --noconfirm --onefile --windowed main.py
```

## 构建产物

成功构建后，您可以在 `dist` 目录中找到以下文件：

- **Windows**：`LottoProphet.exe`（单文件模式）或 `LottoProphet` 目录（目录模式）
- **macOS**：`LottoProphet.app`（应用程序包）和 `LottoProphet.dmg`（磁盘映像，如果未使用 `--skip-installer`）
- **Linux**：`LottoProphet`（可执行文件）

## 安装包创建

### Windows 安装程序

目前需要手动使用 NSIS 或 Inno Setup 创建 Windows 安装程序。未来版本可能会添加自动创建安装程序的功能。

### macOS DMG

在 macOS 上，构建脚本会自动创建 DMG 文件（除非使用了 `--skip-installer` 选项）。

## 常见问题

### 缺少依赖

如果构建过程报告缺少依赖，请确保已安装所有必要的包：

```bash
pip install -r requirements.txt
pip install pyinstaller
```

### 图标问题

确保项目根目录中存在以下图标文件：

- Windows：`icon.ico`
- macOS：`icon.icns`

如果没有这些文件，构建过程会使用默认图标。

### 打包后无法运行

如果打包后的应用程序无法运行，可以尝试以下方法：

1. 使用 `--debug` 选项重新构建，查看控制台输出的错误信息
2. 检查是否缺少必要的数据文件或模型文件
3. 确保 spec 文件中包含了所有必要的隐藏导入和数据文件

## 自定义构建

如需自定义构建过程，可以修改以下文件：

- `setup.py`：修改项目元数据和依赖
- `lottoprophet.spec`：修改 PyInstaller 打包配置
- `build.py`：修改构建脚本逻辑

## 注意事项

- 构建过程可能需要几分钟时间，具体取决于您的计算机性能和项目大小
- 首次构建时，PyInstaller 需要下载和缓存一些文件，可能会比后续构建慢
- 确保您有足够的磁盘空间（至少需要 1GB 的可用空间）