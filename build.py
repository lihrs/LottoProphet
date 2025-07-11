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
        "--add-data", f"scripts/dlt{os.pathsep}data/dlt",
        "--add-data", f"scripts/ssq{os.pathsep}data/ssq",
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
    
    # 确保目录存在
    os.makedirs("dist/model/dlt", exist_ok=True)
    os.makedirs("dist/model/ssq", exist_ok=True)
    os.makedirs("dist/update", exist_ok=True)
    
    # 复制README和LICENSE文件
    for file in ["README.md", "README_EN.md", "requirements.txt"]:
        if os.path.exists(file):
            shutil.copy(file, f"dist/{file}")
            print(f"已复制 {file} 到 dist 目录")
    
    # 创建空的LICENSE文件（如果不存在）
    if not os.path.exists("LICENSE"):
        with open("LICENSE", "w", encoding="utf-8") as f:
            f.write("Copyright (c) 2023 Yang Zhao\n\nAll rights reserved.\n")
        print("已创建默认LICENSE文件")
    
    # 复制LICENSE文件
    if os.path.exists("LICENSE"):
        shutil.copy("LICENSE", "dist/LICENSE")
        print("已复制LICENSE文件到dist目录")
    
    # 复制图标文件（如果存在）
    if os.path.exists("icon.ico"):
        shutil.copy("icon.ico", "dist/icon.ico")
        print("已复制图标文件到dist目录")


def convert_svg_to_ico():
    """将SVG图标转换为ICO格式"""
    if not os.path.exists("icon.svg"):
        print("警告: 未找到icon.svg文件，无法创建图标")
        return False
    
    print("正在将SVG图标转换为ICO格式...")
    
    # 尝试导入必要的库
    try:
        # 先尝试安装必要的库
        subprocess.check_call([sys.executable, "-m", "pip", "install", "cairosvg", "pillow"])
        
        # 运行转换脚本
        if os.path.exists("convert_icon.py"):
            subprocess.check_call([sys.executable, "convert_icon.py"])
            if os.path.exists("icon.ico"):
                # 确保dist目录存在
                os.makedirs("dist", exist_ok=True)
                # 复制图标到dist目录
                shutil.copy("icon.ico", "dist/icon.ico")
                print("已复制图标文件到dist目录")
                return True
            else:
                print("错误: 图标转换失败，未生成icon.ico文件")
                return False
        else:
            print("错误: 未找到convert_icon.py脚本")
            return False
    except Exception as e:
        print(f"转换图标时出错: {e}")
        return False

def create_windows_installer():
    """创建Windows安装程序（仅限Windows）"""
    if platform.system() != "Windows":
        print("跳过创建Windows安装程序（仅支持Windows）")
        return
    
    print("创建Windows安装程序...")
    
    # 检查是否有exe文件
    exe_path = os.path.join("dist", "LottoProphet.exe")
    if not os.path.exists(exe_path):
        print("错误: 未找到LottoProphet.exe文件，无法创建安装程序")
        return
    
    # 检查是否安装了NSIS
    nsis_path = None
    possible_paths = [
        "C:\\Program Files\\NSIS\\makensis.exe",
        "C:\\Program Files (x86)\\NSIS\\makensis.exe"
    ]
    
    for path in possible_paths:
        if os.path.exists(path):
            nsis_path = path
            break
    
    if nsis_path is None:
        print("警告: 未找到NSIS，尝试使用命令行调用makensis...")
        nsis_path = "makensis.exe"
    
    # 检查安装脚本是否存在
    if not os.path.exists("installer.nsi"):
        print("错误: 未找到installer.nsi脚本，无法创建安装程序")
        return
    
    # 运行NSIS编译安装程序
    try:
        subprocess.check_call([nsis_path, "installer.nsi"])
        installer_path = list(Path("dist").glob("*-Setup.exe"))
        if installer_path:
            print(f"安装程序已创建: {installer_path[0]}")
        else:
            print("安装程序可能已创建，但未找到输出文件")
    except subprocess.CalledProcessError as e:
        print(f"创建安装程序时出错: {e}")
    except FileNotFoundError:
        print("错误: 未找到NSIS编译器，请安装NSIS或将其添加到PATH环境变量")
        print("您可以从 https://nsis.sourceforge.io/Download 下载NSIS")

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
    parser.add_argument("--installer-only", action="store_true", help="仅创建安装程序，不重新构建应用")
    
    args = parser.parse_args()
    
    if not args.installer_only:
        # 设置环境
        setup_environment()
        
        # 构建应用程序
        build_application(one_file=not args.onedir, debug=args.debug)
        
        # 复制额外文件
        copy_additional_files()
    else:
        print("跳过构建应用程序，仅创建安装程序")
        
    # 转换图标（无论是否仅创建安装程序）
    if platform.system() == "Windows":
        convert_svg_to_ico()
    
    # 创建安装程序
    if not args.skip_installer:
        # 创建DMG（仅限macOS）
        if platform.system() == "Darwin":
            create_dmg()
        
        # 创建Windows安装程序（仅限Windows）
        if platform.system() == "Windows":
            create_windows_installer()
    
    # 删除build目录
    if not args.keep_build and not args.installer_only:
        print("正在清理build目录...")
        if os.path.exists("build") and os.path.isdir("build"):
            try:
                shutil.rmtree("build")
                print("build目录已删除")
            except Exception as e:
                print(f"删除build目录时出错: {e}")
    else:
        if args.keep_build:
            print("保留build目录（使用了--keep-build选项）")
    
    print("构建过程完成!")
    
    # 提示用户安装NSIS（如果在Windows上且没有跳过安装程序创建）
    if platform.system() == "Windows" and not args.skip_installer:
        print("\n注意: 要创建Windows安装程序，您需要安装NSIS (Nullsoft Scriptable Install System)")
        print("如果您尚未安装NSIS，请从 https://nsis.sourceforge.io/Download 下载并安装")
        print("安装后，您可以再次运行此脚本以创建安装程序，或使用 --installer-only 选项仅创建安装程序")


if __name__ == "__main__":
    main()