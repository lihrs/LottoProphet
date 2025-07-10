#!/usr/bin/env python
# -*- coding: utf-8 -*-

from setuptools import setup, find_packages
import os

# 读取requirements.txt文件内容
with open('requirements.txt') as f:
    requirements = f.read().splitlines()

# 添加PyInstaller作为开发依赖
develop_requires = [
    'PyInstaller>=5.13.0',
]

setup(
    name="LottoProphet",
    version="1.0.0",
    author="Yang Zhao",
    author_email="example@example.com",  # 替换为实际邮箱
    description="彩票预测系统",
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/zhaoyangpp/LottoProphet",  # 替换为实际URL
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": develop_requires,
    },
    entry_points={
        "console_scripts": [
            "lottoprophet=main:main",
        ],
    },
    include_package_data=True,
    package_data={
        "": ["*.md", "*.txt"],
        "data": ["*/*.csv"],
        "model": ["*/*.pth", "*/*.pkl"],
    },
)