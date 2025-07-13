#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
设备选择工具函数
用于统一处理GPU/CPU设备选择逻辑，特别针对Mac M1/M2的MPS性能问题
"""

import platform
import logging

logger = logging.getLogger(__name__)

def get_optimal_device(use_gpu=True, force_mps=False):
    """
    获取最优的计算设备
    
    Args:
        use_gpu (bool): 是否尝试使用GPU
        force_mps (bool): 是否强制使用MPS（仅在Mac M1/M2上有效）
    
    Returns:
        torch.device: 最优的计算设备
        str: 设备描述信息
    """
    try:
        import torch
    except ImportError:
        return None, "PyTorch未安装"
    
    if not use_gpu:
        return torch.device("cpu"), "使用CPU"
    
    # 检查CUDA可用性
    if torch.cuda.is_available():
        device_name = torch.cuda.get_device_name(0)
        return torch.device("cuda"), f"使用CUDA GPU: {device_name}"
    
    # 检查MPS可用性
    if hasattr(torch, 'mps') and torch.backends.mps.is_available():
        is_mac_arm = platform.system() == 'Darwin' and platform.processor() == 'arm'
        
        if is_mac_arm and not force_mps:
            # Mac M1/M2: 默认使用CPU，因为MPS可能存在性能问题
            logger.warning("检测到Mac M1/M2处理器，由于MPS可能存在性能问题，建议使用CPU以获得更好的性能")
            logger.info("如需强制使用MPS，请设置force_mps=True")
            return torch.device("cpu"), "Mac M1/M2使用CPU (MPS可能较慢)"
        else:
            # 非Mac ARM或强制使用MPS
            return torch.device("mps"), "使用Apple M系列芯片GPU (MPS)"
    
    # 没有可用的GPU
    return torch.device("cpu"), "GPU不可用，使用CPU"

def check_device_availability():
    """
    检查设备可用性
    
    Returns:
        dict: 包含设备可用性信息的字典
    """
    try:
        import torch
    except ImportError:
        return {
            'cuda_available': False,
            'mps_available': False,
            'gpu_available': False,
            'device_info': 'PyTorch未安装',
            'is_mac_arm': False
        }
    
    cuda_available = torch.cuda.is_available()
    mps_available = hasattr(torch, 'mps') and torch.backends.mps.is_available()
    is_mac_arm = platform.system() == 'Darwin' and platform.processor() == 'arm'
    
    # 对于Mac M1/M2，不将MPS视为真正可用的GPU
    if is_mac_arm and mps_available:
        gpu_available = cuda_available
        if cuda_available:
            device_info = f"CUDA可用 ({torch.cuda.get_device_name(0)})"
        else:
            device_info = "Apple M系列芯片 (建议使用CPU，MPS可能较慢)"
    else:
        gpu_available = cuda_available or mps_available
        if cuda_available:
            device_info = f"CUDA可用 ({torch.cuda.get_device_name(0)})"
        elif mps_available:
            device_info = "Apple M系列芯片GPU (MPS)可用"
        else:
            device_info = "GPU不可用"
    
    return {
        'cuda_available': cuda_available,
        'mps_available': mps_available,
        'gpu_available': gpu_available,
        'device_info': device_info,
        'is_mac_arm': is_mac_arm
    }

def get_device_performance_tips():
    """
    获取设备性能优化建议
    
    Returns:
        list: 性能优化建议列表
    """
    device_info = check_device_availability()
    tips = []
    
    if device_info['is_mac_arm'] and device_info['mps_available']:
        tips.extend([
            "Mac M1/M2用户注意：",
            "• MPS在某些深度学习任务中可能比CPU慢",
            "• 建议先使用CPU训练，如果速度不满意再尝试MPS",
            "• 可以通过设置force_mps=True强制使用MPS",
            "• 对于小型模型，CPU通常表现更好"
        ])
    elif device_info['cuda_available']:
        tips.extend([
            "CUDA GPU可用：",
            "• 建议使用GPU加速训练",
            "• 确保模型和数据都移动到GPU上",
            "• 可以使用更大的批次大小"
        ])
    else:
        tips.extend([
            "仅CPU可用：",
            "• 建议使用较小的批次大小",
            "• 考虑使用数据并行处理",
            "• 优化模型结构以减少计算量"
        ])
    
    return tips