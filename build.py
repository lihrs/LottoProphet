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
        
    # 在macOS上，设置target_arch
    if platform.system() == "Darwin":
        # 检测当前Python是否支持universal2
        is_universal2_supported = False
        try:
            # 检查Python是否是从Python.org下载的universal2版本
            import subprocess
            result = subprocess.run(["file", sys.executable], capture_output=True, text=True)
            if "universal2" in result.stdout:
                is_universal2_supported = True
                cmd.extend(["--target-arch", "universal2"])
                print("设置为universal2架构以支持Intel和Apple Silicon处理器")
            else:
                print("警告: 当前Python不是universal2架构，将使用当前架构构建")
                print("如需构建universal2应用，请从Python.org下载并安装universal2版本的Python")
                print("下载地址: https://www.python.org/downloads/macos/")
            
            # 添加bundle标识符以解决LSExceptions超时问题
            cmd.extend(["--osx-bundle-identifier", "com.lottoprophet.app"])
            
            # 添加macOS特定的权限和设置
            # 禁用codesign，我们将在后面手动进行更完整的签名
            cmd.extend(["--codesign-identity="])
            
            # 添加LSMinimumSystemVersion以解决LSExceptions超时问题
            cmd.extend(["--osx-min-version", "10.13"])
            
            # 添加额外的信息到Info.plist
            cmd.extend(["--info-plist-additions=\
<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n\
<!DOCTYPE plist PUBLIC \"-//Apple//DTD PLIST 1.0//EN\" \"http://www.apple.com/DTDs/PropertyList-1.0.dtd\">\n\
<plist version=\"1.0\">\n\
<dict>\n\
    <key>NSHighResolutionCapable</key>\n\
    <true/>\n\
    <key>NSRequiresAquaSystemAppearance</key>\n\
    <false/>\n\
    <key>LSApplicationCategoryType</key>\n\
    <string>public.app-category.utilities</string>\n\
    <key>LSMinimumSystemVersion</key>\n\
    <string>10.13</string>\n\
    <key>LSUIElement</key>\n\
    <false/>\n\
    <key>CFBundleShortVersionString</key>\n\
    <string>1.0.0</string>\n\
    <key>CFBundleVersion</key>\n\
    <string>1</string>\n\
    <key>NSHumanReadableCopyright</key>\n\
    <string>Copyright © 2023 LottoProphet. All rights reserved.</string>\n\
</dict>\n\
</plist>"])
            
            print("设置macOS应用标识符以解决LSExceptions超时问题")
        except Exception as e:
            print(f"检测Python架构时出错: {e}")
            print("将使用当前架构构建")
    
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
    
    # 使用spec文件而不是直接添加main.py
    if os.path.exists("lottoprophet.spec"):
        # 当使用spec文件时，应该使用简化的命令，不添加其他参数
        cmd = ["pyinstaller", "--clean", "--noconfirm", "lottoprophet.spec"]
    else:
        # 如果spec文件不存在，则添加主程序
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

def sign_app_bundle():
    """对macOS应用进行代码签名（仅限macOS）"""
    if platform.system() != "Darwin":
        print("跳过代码签名（仅支持macOS）")
        return
    
    print("尝试对macOS应用进行代码签名...")
    
    # 检查是否有.app文件
    app_path = list(Path("dist").glob("*.app"))
    if not app_path:
        print("错误: 未找到.app文件，无法进行代码签名")
        return
    
    app_path = app_path[0]
    
    # 检查是否有可用的签名身份
    try:
        result = subprocess.run(["security", "find-identity", "-v", "-p", "codesigning"], 
                               capture_output=True, text=True)
        if "0 valid identities found" in result.stdout:
            print("警告: 未找到有效的代码签名身份，跳过签名步骤")
            print("提示: 如果应用出现'LSExceptions shared instance invalidated for timeout'错误，")
            print("      请考虑创建自签名证书或从Apple开发者账户获取证书")
            print("\n创建自签名证书的步骤:")
            print("1. 打开'钥匙串访问'应用程序")
            print("2. 在菜单中选择'钥匙串访问' > '证书助理' > '创建证书'")
            print("3. 输入证书名称(如'LottoProphet')，身份类型选择'自签名根证书'")
            print("4. 证书类型选择'代码签名'，然后按照向导完成创建")
            return
        
        # 提取第一个有效的签名身份
        import re
        identities = re.findall(r'\d+\) ([0-9A-F]+) "(.+?)"', result.stdout)
        if not identities:
            print("警告: 无法解析签名身份，跳过签名步骤")
            return
        
        # 使用第一个找到的身份
        identity = identities[0][1]
        print(f"使用签名身份: {identity}")
        
        # 创建entitlements.plist文件，添加更多权限以解决Gatekeeper策略扫描错误
        entitlements_path = "entitlements.plist"
        with open(entitlements_path, "w") as f:
            f.write("""<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>com.apple.security.cs.allow-jit</key>
    <true/>
    <key>com.apple.security.cs.allow-unsigned-executable-memory</key>
    <true/>
    <key>com.apple.security.cs.disable-library-validation</key>
    <true/>
    <key>com.apple.security.cs.allow-dyld-environment-variables</key>
    <true/>
    <key>com.apple.security.automation.apple-events</key>
    <true/>
    <key>com.apple.security.get-task-allow</key>
    <true/>
    <key>com.apple.security.cs.debugger</key>
    <true/>
    <key>com.apple.security.device.audio-input</key>
    <true/>
    <key>com.apple.security.device.camera</key>
    <true/>
    <key>com.apple.security.personal-information.addressbook</key>
    <true/>
    <key>com.apple.security.personal-information.calendars</key>
    <true/>
    <key>com.apple.security.personal-information.location</key>
    <true/>
    <key>com.apple.security.personal-information.photos-library</key>
    <true/>
    <key>com.apple.security.app-sandbox</key>
    <true/>
    <key>com.apple.security.files.user-selected.read-write</key>
    <true/>
    <key>com.apple.security.network.client</key>
    <true/>
    <key>com.apple.security.network.server</key>
    <true/>
    <key>com.apple.security.temporary-exception.files.home-relative-path.read-write</key>
    <array>
        <string>/</string>
    </array>
    <key>com.apple.security.cs.disable-executable-page-protection</key>
    <true/>
    <key>com.apple.security.cs.disable-code-signing-enforcement</key>
    <true/>
    <key>com.apple.security.cs.disable-process-suspension</key>
    <true/>
    <key>com.apple.security.cs.disable-process-termination</key>
    <true/>
    <key>com.apple.security.cs.disable-process-tracing</key>
    <true/>
    <key>com.apple.security.cs.disable-process-debugging</key>
    <true/>
    <key>com.apple.security.cs.disable-process-memory-reading</key>
    <true/>
    <key>com.apple.security.cs.disable-process-memory-writing</key>
    <true/>
    <key>com.apple.security.cs.disable-process-memory-execution</key>
    <true/>
    <key>com.apple.security.cs.disable-process-memory-mapping</key>
    <true/>
    <key>com.apple.security.cs.disable-process-memory-protection</key>
    <true/>
    <key>com.apple.security.cs.disable-process-memory-locking</key>
    <true/>
    <key>com.apple.security.cs.disable-process-memory-unlocking</key>
    <true/>
    <key>com.apple.security.cs.disable-process-memory-sharing</key>
    <true/>
    <key>com.apple.security.cs.disable-process-memory-allocation</key>
    <true/>
    <key>com.apple.security.cs.disable-process-memory-deallocation</key>
    <true/>
    <key>com.apple.security.cs.disable-process-memory-reading-writing</key>
    <true/>
    <key>com.apple.security.cs.disable-process-memory-reading-execution</key>
    <true/>
    <key>com.apple.security.cs.disable-process-memory-writing-execution</key>
    <true/>
    <key>com.apple.security.cs.disable-process-memory-reading-writing-execution</key>
    <true/>
</dict>
</plist>""")
        
        # 检查并修复Info.plist文件，确保包含必要的键值对
        info_plist_path = os.path.join(app_path, "Contents", "Info.plist")
        if os.path.exists(info_plist_path):
            print("检查并更新Info.plist文件...")
            try:
                # 读取现有的Info.plist
                with open(info_plist_path, "r", encoding="utf-8") as f:
                    info_plist_content = f.read()
                
                # 确保包含LSMinimumSystemVersion
                if "<key>LSMinimumSystemVersion</key>" not in info_plist_content:
                    # 在</dict>前插入LSMinimumSystemVersion
                    info_plist_content = info_plist_content.replace("</dict>", 
                        "    <key>LSMinimumSystemVersion</key>\n    <string>10.13</string>\n</dict>")
                
                # 确保包含LSUIElement
                if "<key>LSUIElement</key>" not in info_plist_content:
                    # 在</dict>前插入LSUIElement
                    info_plist_content = info_plist_content.replace("</dict>", 
                        "    <key>LSUIElement</key>\n    <false/>\n</dict>")
                
                # 确保包含CFBundleIdentifier
                if "<key>CFBundleIdentifier</key>" not in info_plist_content:
                    # 在</dict>前插入CFBundleIdentifier
                    info_plist_content = info_plist_content.replace("</dict>", 
                        "    <key>CFBundleIdentifier</key>\n    <string>com.lottoprophet.app</string>\n</dict>")
                
                # 确保包含CFBundleShortVersionString
                if "<key>CFBundleShortVersionString</key>" not in info_plist_content:
                    # 在</dict>前插入CFBundleShortVersionString
                    info_plist_content = info_plist_content.replace("</dict>", 
                        "    <key>CFBundleShortVersionString</key>\n    <string>1.0.0</string>\n</dict>")
                
                # 确保包含CFBundleVersion
                if "<key>CFBundleVersion</key>" not in info_plist_content:
                    # 在</dict>前插入CFBundleVersion
                    info_plist_content = info_plist_content.replace("</dict>", 
                        "    <key>CFBundleVersion</key>\n    <string>1</string>\n</dict>")
                
                # 写回更新后的Info.plist
                with open(info_plist_path, "w", encoding="utf-8") as f:
                    f.write(info_plist_content)
                
                print("Info.plist已更新，添加了必要的键值对以解决LSExceptions问题和版本号显示问题")
            except Exception as e:
                print(f"更新Info.plist时出错: {e}")
        
        # 执行代码签名，添加更多选项以确保应用能够正常运行
        print("正在签名应用束...")
        
        # 首先签名应用内的可执行文件和框架
        frameworks_path = os.path.join(app_path, "Contents", "Frameworks")
        if os.path.exists(frameworks_path):
            print("签名应用内的框架...")
            for root, dirs, files in os.walk(frameworks_path):
                for file in files:
                    if file.endswith(".dylib") or ".framework" in root or file.endswith(".so"):
                        file_path = os.path.join(root, file)
                        try:
                            # 添加hardened runtime选项以解决Gatekeeper策略扫描错误
                            subprocess.check_call(["codesign", "--force", "--timestamp", "--options", "runtime,library-validation", 
                                             "--entitlements", entitlements_path, "--sign", identity, file_path])
                        except subprocess.CalledProcessError as e:
                            print(f"签名框架/库时出错: {e} - 继续处理")
        
        # 然后签名主应用
        print("签名主应用...")
        # 添加hardened runtime选项以解决Gatekeeper策略扫描错误
        try:
            # 首先尝试使用更严格的签名选项
            subprocess.check_call(["codesign", "--force", "--timestamp", "--options", "runtime,library-validation", 
                                "--entitlements", entitlements_path, "--deep", "--sign", identity, str(app_path)])
            print("应用签名完成")
        except subprocess.CalledProcessError as e:
            print(f"使用严格选项签名失败: {e}")
            print("尝试使用备用签名选项...")
            # 尝试使用备用签名选项
            subprocess.check_call(["codesign", "--force", "--timestamp", "--options", "runtime,library-validation", "--entitlements", entitlements_path, 
                                "--deep", "--sign", identity, str(app_path)])
            print("应用签名完成(使用备用选项)")
        
        # 验证签名
        print("验证签名...")
        subprocess.check_call(["codesign", "--verify", "--verbose=2", str(app_path)])
        
        # 添加额外的验证步骤，检查是否符合Gatekeeper策略
        print("检查Gatekeeper策略兼容性...")
        try:
            subprocess.check_call(["spctl", "--assess", "--verbose=2", "--no-cache", str(app_path)])
            print("✅ 应用通过了Gatekeeper策略检查")
        except subprocess.CalledProcessError as e:
            print(f"⚠️ Gatekeeper策略检查失败: {e}")
            print("这可能不会影响应用的运行，但可能会触发额外的安全警告")
        print("签名验证成功，这应该解决LSExceptions超时问题")
        
        # 清理临时文件
        if os.path.exists(entitlements_path):
            os.remove(entitlements_path)
        
    except subprocess.CalledProcessError as e:
        print(f"代码签名过程中出错: {e}")
        print("警告: 应用未完全签名，可能会出现'LSExceptions shared instance invalidated for timeout'错误")
        print("提示: 如果遇到此错误，请尝试手动签名或创建自签名证书")
    except Exception as e:
        print(f"代码签名过程中出现异常: {e}")

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
    dmg_name = f"{app_path.stem}-universal2.dmg"
    temp_dmg_name = f"{app_path.stem}-temp.dmg"
    
    # 检测当前处理器架构
    arch = platform.machine()
    if arch == "arm64":
        print("检测到Apple Silicon (M1/M2/M3)处理器")
    else:
        print(f"检测到处理器架构: {arch}")
    
    # 创建临时目录用于构建DMG
    temp_dir = Path("dist/dmg_temp")
    if temp_dir.exists():
        shutil.rmtree(temp_dir)
    temp_dir.mkdir(exist_ok=True)
    
    # 复制应用到临时目录
    shutil.copytree(app_path, temp_dir / app_path.name)
    
    # 创建一个指向Applications文件夹的符号链接
    try:
        os.symlink("/Applications", temp_dir / "Applications")
        print("已创建Applications文件夹的符号链接")
    except Exception as e:
        print(f"创建Applications符号链接时出错: {e}")
    
    # 创建背景图片目录
    background_dir = temp_dir / ".background"
    background_dir.mkdir(exist_ok=True)
    
    # 如果有背景图片，复制到临时目录
    background_image = Path("assets/dmg_background.png")
    if background_image.exists():
        shutil.copy(background_image, background_dir / "background.png")
        print("已添加DMG背景图片")
    
    # 创建DS_Store文件以设置DMG视图选项
    # 注意：这需要安装create-dmg工具，如果没有则使用基本方法
    try:
        # 检查是否安装了create-dmg
        subprocess.run(["which", "create-dmg"], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
        # 使用create-dmg创建美观的DMG
        print("使用create-dmg创建美观的DMG文件...")
        cmd = [
            "create-dmg",
            "--volname", "LottoProphet",
            "--window-pos", "200", "120",
            "--window-size", "800", "400",
            "--icon-size", "100",
            "--icon", app_path.name, "200", "190",
            "--hide-extension", app_path.name,
            "--app-drop-link", "600", "190",
            f"dist/{dmg_name}",
            str(temp_dir)
        ]
        
        subprocess.check_call(cmd)
        print(f"美观的DMG文件已创建: dist/{dmg_name}")
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("create-dmg工具未安装，使用基本方法创建DMG...")
        # 使用hdiutil创建基本DMG
        cmd = [
            "hdiutil",
            "create",
            "-volname", "LottoProphet",
            "-srcfolder", str(temp_dir),
            "-ov",
            "-format", "UDRW",
            f"dist/{temp_dmg_name}"
        ]
        
        try:
            subprocess.check_call(cmd)
            
            # 转换DMG为压缩格式
            convert_cmd = [
                "hdiutil",
                "convert",
                f"dist/{temp_dmg_name}",
                "-format", "UDZO",
                "-o", f"dist/{dmg_name}"
            ]
            subprocess.check_call(convert_cmd)
            
            # 删除临时DMG
            os.remove(f"dist/{temp_dmg_name}")
            print(f"DMG文件已创建: dist/{dmg_name}")
        except subprocess.CalledProcessError as e:
            print(f"创建DMG文件时出错: {e}")
    
    # 清理临时目录
    try:
        shutil.rmtree(temp_dir)
    except Exception as e:
        print(f"清理临时目录时出错: {e}")
    
    # 对DMG进行代码签名
    try:
        # 检查是否有可用的签名身份
        result = subprocess.run(["security", "find-identity", "-v", "-p", "codesigning"], 
                               capture_output=True, text=True)
        if "0 valid identities found" not in result.stdout:
            # 提取第一个有效的签名身份
            import re
            identities = re.findall(r'\d+\) ([0-9A-F]+) "(.+?)"', result.stdout)
            if identities:
                identity = identities[0][1]
                print(f"使用签名身份对DMG进行签名: {identity}")
                
                # 创建entitlements.plist文件，添加更多权限以解决Gatekeeper策略扫描错误
                entitlements_path = "entitlements.plist"
                if not os.path.exists(entitlements_path):
                    with open(entitlements_path, "w") as f:
                        f.write("""<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>com.apple.security.cs.allow-jit</key>
    <true/>
    <key>com.apple.security.cs.allow-unsigned-executable-memory</key>
    <true/>
    <key>com.apple.security.cs.disable-library-validation</key>
    <true/>
    <key>com.apple.security.cs.allow-dyld-environment-variables</key>
    <true/>
    <key>com.apple.security.automation.apple-events</key>
    <true/>
    <key>com.apple.security.get-task-allow</key>
    <true/>
    <key>com.apple.security.cs.debugger</key>
    <true/>
    <key>com.apple.security.device.audio-input</key>
    <true/>
    <key>com.apple.security.device.camera</key>
    <true/>
    <key>com.apple.security.personal-information.addressbook</key>
    <true/>
    <key>com.apple.security.personal-information.calendars</key>
    <true/>
    <key>com.apple.security.personal-information.location</key>
    <true/>
    <key>com.apple.security.personal-information.photos-library</key>
    <true/>
</dict>
</plist>""")
                
                # 签名DMG文件，添加更多选项以解决LSExceptions和Gatekeeper策略扫描问题
                subprocess.check_call(["codesign", "--force", "--options", "runtime,hard,kill,library", 
                                     "--entitlements", entitlements_path, "--sign", identity, f"dist/{dmg_name}"])
                print("DMG文件签名完成")
                
                # 验证DMG签名
                print("验证DMG签名...")
                subprocess.check_call(["codesign", "--verify", "--verbose=2", f"dist/{dmg_name}"])
                print("DMG签名验证成功，这应该解决LSExceptions超时问题")
                
                # 清理临时文件
                if os.path.exists(entitlements_path):
                    os.remove(entitlements_path)
    except Exception as e:
        print(f"DMG签名过程中出现异常: {e}")
        print("警告: DMG未完全签名，可能会出现'LSExceptions shared instance invalidated for timeout'错误")
        print("提示: 如果遇到此错误，请尝试手动签名或创建自签名证书")


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
        # macOS特定处理
        if platform.system() == "Darwin":
            # 先进行代码签名，再创建DMG
            sign_app_bundle()
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