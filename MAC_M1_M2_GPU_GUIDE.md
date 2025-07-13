# Mac M1/M2 GPU使用指南

## 问题描述

Mac M1/M2处理器虽然支持PyTorch的MPS (Metal Performance Shaders) 后端，但在实际使用中可能存在以下问题：

1. **性能问题**：MPS在某些深度学习任务中可能比CPU还要慢 <mcreference link="https://github.com/pytorch/pytorch/issues/77799" index="2">2</mcreference>
2. **兼容性问题**：部分PyTorch操作在MPS上可能不稳定
3. **内存管理**：MPS的内存管理机制与CUDA不同，可能导致意外的性能下降

## 解决方案

### 自动优化策略

本项目已经实现了智能设备选择策略：

- **Mac M1/M2用户**：默认使用CPU进行训练，即使MPS可用
- **其他平台**：正常使用CUDA或MPS
- **性能提示**：在日志中提供设备选择的详细说明

### 手动控制选项

如果您想强制使用MPS，可以通过以下方式：

```python
from src.utils.device_utils import get_optimal_device

# 强制使用MPS（如果可用）
device, info = get_optimal_device(use_gpu=True, force_mps=True)
print(f"使用设备: {device}, 信息: {info}")
```

### 性能对比建议

建议Mac M1/M2用户进行以下测试：

1. **CPU训练**：使用默认设置
2. **MPS训练**：设置`force_mps=True`
3. **对比结果**：选择训练速度更快的方案

## 技术细节

### 设备检测逻辑

```python
def get_optimal_device(use_gpu=True, force_mps=False):
    # 1. 优先使用CUDA（如果可用）
    # 2. 检测Mac M1/M2平台
    # 3. 根据平台和用户设置选择最优设备
    # 4. 提供详细的设备信息和建议
```

### 受影响的组件

- `src/models/trainers/dlt_trainer.py`
- `src/models/trainers/ssq_trainer.py`
- `src/models/ml_models.py`
- `src/ui/components.py`
- `src/ui/app.py`

## 性能优化建议

### 对于Mac M1/M2用户

1. **使用CPU训练**：通常比MPS更稳定和快速
2. **优化批次大小**：使用较小的批次大小以适应CPU内存
3. **并行处理**：利用多核CPU的优势
4. **模型优化**：考虑使用更轻量级的模型结构

### 通用优化建议

1. **数据预处理**：在CPU上进行数据预处理，减少设备间数据传输
2. **混合精度**：如果使用GPU，考虑使用半精度浮点数
3. **内存管理**：及时清理不需要的变量和缓存

## 故障排除

### 常见问题

**Q: 为什么我的Mac M1显示"建议使用CPU"？**
A: 这是基于社区反馈的优化策略。MPS在某些情况下可能比CPU慢，特别是对于小型模型。

**Q: 如何强制使用MPS？**
A: 在训练代码中设置`force_mps=True`参数。

**Q: 如何判断哪种设备更快？**
A: 建议进行实际的性能测试，比较CPU和MPS的训练时间。

### 性能测试代码

```python
import time
import torch
from src.utils.device_utils import get_optimal_device

# 测试CPU性能
start_time = time.time()
device_cpu, _ = get_optimal_device(use_gpu=False)
# 运行您的训练代码
cpu_time = time.time() - start_time

# 测试MPS性能（如果可用）
start_time = time.time()
device_mps, _ = get_optimal_device(use_gpu=True, force_mps=True)
# 运行您的训练代码
mps_time = time.time() - start_time

print(f"CPU训练时间: {cpu_time:.2f}秒")
print(f"MPS训练时间: {mps_time:.2f}秒")
print(f"推荐使用: {'CPU' if cpu_time < mps_time else 'MPS'}")
```

## 参考资料

- [PyTorch MPS Backend Documentation](https://pytorch.org/docs/stable/notes/mps.html)
- [Apple Developer MPS Guide](https://developer.apple.com/metal/pytorch/) <mcreference link="https://developer.apple.com/metal/pytorch/" index="4">4</mcreference>
- [PyTorch GitHub Issues - MPS Performance](https://github.com/pytorch/pytorch/issues?q=is%3Aissue+is%3Aopen+mps+performance)

## 更新日志

- **2024-01-XX**: 实现智能设备选择策略
- **2024-01-XX**: 添加强制MPS选项
- **2024-01-XX**: 优化设备检测逻辑
- **2024-01-XX**: 添加性能测试工具