#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Matplotlib配置工具
用于统一处理matplotlib的字体和样式配置
"""

import matplotlib
import matplotlib.pyplot as plt
import warnings
import logging
import platform
import os

logger = logging.getLogger(__name__)

def configure_matplotlib():
    """
    配置matplotlib的字体和样式，解决中文显示和字体警告问题
    """
    # 过滤matplotlib字体相关警告
    warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib")
    
    # 设置更好的图表风格
    plt.style.use('ggplot')
    
    # 根据不同操作系统设置合适的中文字体
    system = platform.system()
    
    # 中文字体列表，按优先级排序
    if system == 'Windows':
        font_list = ['SimHei', 'Microsoft YaHei', 'SimSun', 'FangSong', 'KaiTi', 'Arial Unicode MS']
    elif system == 'Darwin':  # macOS
        font_list = ['PingFang SC', 'Heiti SC', 'STHeiti', 'Apple LiGothic', 'Hiragino Sans GB', 'Microsoft YaHei', 'Arial Unicode MS']
    else:  # Linux
        font_list = ['WenQuanYi Micro Hei', 'WenQuanYi Zen Hei', 'Droid Sans Fallback', 'Noto Sans CJK SC', 'Source Han Sans CN', 'Microsoft YaHei', 'SimHei']
    
    # 添加通用字体作为后备
    font_list.extend(['DejaVu Sans', 'sans-serif'])
    
    try:
        plt.rcParams['font.sans-serif'] = font_list
        plt.rcParams['axes.unicode_minus'] = False
        logger.info(f"已配置matplotlib字体: {font_list[0]}等")
    except Exception as e:
        logger.warning(f"配置matplotlib字体时出错: {str(e)}")

    # 设置DPI以获得更好的显示效果
    plt.rcParams['figure.dpi'] = 100
    
    # 设置更好的默认颜色循环
    plt.rcParams['axes.prop_cycle'] = plt.cycler('color', 
                                               ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', 
                                                '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', 
                                                '#bcbd22', '#17becf'])