#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
将SVG图标转换为ICO格式的脚本
"""

import os
import sys
import subprocess
import platform

def convert_svg_to_ico(svg_path, ico_path):
    """
    将SVG图标转换为ICO格式
    需要安装cairosvg和pillow库
    """
    try:
        # 尝试导入必要的库
        from cairosvg import svg2png
        from PIL import Image
        import io
        
        print(f"正在将 {svg_path} 转换为 {ico_path}...")
        
        # 创建不同尺寸的图标
        sizes = [16, 32, 48, 64, 128, 256]
        images = []
        
        for size in sizes:
            # 将SVG转换为PNG
            png_data = svg2png(url=svg_path, output_width=size, output_height=size)
            
            # 将PNG数据转换为PIL图像
            img = Image.open(io.BytesIO(png_data))
            images.append(img)
        
        # 保存为ICO文件
        images[0].save(
            ico_path,
            format='ICO',
            sizes=[(img.width, img.height) for img in images],
            append_images=images[1:]
        )
        
        print(f"图标已成功转换并保存到 {ico_path}")
        return True
    except ImportError as e:
        print(f"错误: 缺少必要的库 - {e}")
        print("请安装必要的库: pip install cairosvg pillow")
        return False
    except Exception as e:
        print(f"转换图标时出错: {e}")
        return False

def main():
    # 检查文件是否存在
    svg_path = "icon.svg"
    ico_path = "icon.ico"
    
    if not os.path.exists(svg_path):
        print(f"错误: 找不到SVG图标文件 {svg_path}")
        return False
    
    # 如果ICO文件已存在，则跳过转换
    if os.path.exists(ico_path):
        print(f"ICO图标文件 {ico_path} 已存在，跳过转换")
        return True
    
    # 转换图标
    return convert_svg_to_ico(svg_path, ico_path)

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)