#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
构建脚本 - 使用PyInstaller将LottoProphet打包为可执行文件
"""

import os
import sys
import shutil
import subprocess
import platform
import argparse
from pathlib import Path


def setup_environment():
    """设置构建环境"""
    print("正在设置构建环境...")
    
    # 确保PyInstaller已安装
    try:
        import PyInstaller
        print(f"PyInstaller版本: {PyInstaller.__version__}")
    except ImportError:
        print("正在安装PyInstaller...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "PyInstaller>=5.13.0"])
    
    # 确保所有依赖已安装
    print("正在安装项目依赖...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
    
    # 创建构建目录
    os.makedirs("dist", exist_ok=True)
    os.makedirs("build", exist_ok=True)


def build_application(one_file=True, debug=False):
    """构建应用程序"""
    print("开始构建应用程序...")
    
    # 构建命令
    cmd = [
        "pyinstaller",
        "--clean",
        "--noconfirm",
        "--name", "LottoProphet",  # 设置应用程序名称
    ]
    
    # 是否打包为单文件
    if one_file:
        cmd.append("--onefile")
    else:
        cmd.append("--onedir")
    
    # 是否显示控制台
    if debug:
        cmd.append("--console")
    else:
        cmd.append("--windowed")
    
    # 添加数据文件
    cmd.extend([
        "--add-data", f"README.md{os.pathsep}.",
        "--add-data", f"README_EN.md{os.pathsep}.",
        "--add-data", f"requirements.txt{os.pathsep}.",
        "--add-data", f"data{os.pathsep}data",
        "--add-data", f"model{os.pathsep}model",
        "--add-data", f"update{os.pathsep}update",
    ])
    
    # 如果存在图标文件，添加图标
    if os.path.exists("icon.ico") and platform.system() == "Windows":
        cmd.extend(["--icon", "icon.ico"])
    elif os.path.exists("icon.icns") and platform.system() == "Darwin":
        cmd.extend(["--icon", "icon.icns"])
    elif os.path.exists("icon.svg"):
        # 尝试转换SVG为平台特定图标
        try:
            if platform.system() == "Darwin" and not os.path.exists("icon.icns"):
                print("尝试将SVG转换为ICNS图标...")
                # 这里可以添加SVG到ICNS的转换代码
                pass
            elif platform.system() == "Windows" and not os.path.exists("icon.ico"):
                print("尝试将SVG转换为ICO图标...")
                # 这里可以添加SVG到ICO的转换代码
                pass
        except Exception as e:
            print(f"转换图标时出错: {e}")
    
    # 添加主程序
    cmd.append("main.py")
    
    # 执行构建
    print(f"执行命令: {' '.join(cmd)}")
    subprocess.check_call(cmd)
    
    print("应用程序构建完成!")


def copy_additional_files():
    """复制额外的文件到构建目录"""
    print("复制额外文件...")
    
    # 确保数据目录存在
    os.makedirs("dist/data/dlt", exist_ok=True)
    os.makedirs("dist/data/ssq", exist_ok=True)
    os.makedirs("dist/model/dlt", exist_ok=True)
    os.makedirs("dist/model/ssq", exist_ok=True)


def create_dmg():
    """创建DMG文件（仅限macOS）"""
    if platform.system() != "Darwin":
        print("跳过创建DMG（仅支持macOS）")
        return
    
    print("创建macOS DMG文件...")
    
    # 检查是否有.app文件
    app_path = list(Path("dist").glob("*.app"))
    if not app_path:
        print("错误: 未找到.app文件，无法创建DMG")
        return
    
    app_path = app_path[0]
    dmg_name = f"{app_path.stem}.dmg"
    
    # 使用hdiutil创建DMG
    cmd = [
        "hdiutil",
        "create",
        "-volname", "LottoProphet",
        "-srcfolder", str(app_path),
        "-ov",
        "-format", "UDZO",
        f"dist/{dmg_name}"
    ]
    
    try:
        subprocess.check_call(cmd)
        print(f"DMG文件已创建: dist/{dmg_name}")
    except subprocess.CalledProcessError as e:
        print(f"创建DMG文件时出错: {e}")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="LottoProphet构建脚本")
    parser.add_argument("--onedir", action="store_true", help="构建为目录而非单文件")
    parser.add_argument("--debug", action="store_true", help="启用调试模式（显示控制台）")
    parser.add_argument("--skip-installer", action="store_true", help="跳过创建安装程序/DMG")
    parser.add_argument("--keep-build", action="store_true", help="保留构建临时目录")
    
    args = parser.parse_args()
    
    # 设置环境
    setup_environment()
    
    # 构建应用程序
    build_application(one_file=not args.onedir, debug=args.debug)
    
    # 复制额外文件
    copy_additional_files()
    
    # 创建DMG（仅限macOS）
    if not args.skip_installer and platform.system() == "Darwin":
        create_dmg()
    
    # 删除build目录
    if not args.keep_build:
        print("正在清理build目录...")
        if os.path.exists("build") and os.path.isdir("build"):
            try:
                shutil.rmtree("build")
                print("build目录已删除")
            except Exception as e:
                print(f"删除build目录时出错: {e}")
    else:
        print("保留build目录（使用了--keep-build选项）")
    
    print("构建过程完成!")


if __name__ == "__main__":
    main()