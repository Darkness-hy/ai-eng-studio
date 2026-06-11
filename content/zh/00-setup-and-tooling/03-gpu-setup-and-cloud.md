# GPU 环境搭建与云端方案

> 用 CPU 训练拿来学习没问题，但真正做训练就需要 GPU。

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 0, Lesson 01
**Time:** ~45 minutes

## 学习目标

- 使用 `nvidia-smi` 和 PyTorch 的 CUDA API 检查本地 GPU 是否可用
- 在 Google Colab 中配置 T4 GPU，免费进行云端实验
- 对比 CPU 与 GPU 上的矩阵乘法性能，测量加速比
- 用 fp16 经验法则估算你的显存（VRAM）最多能装下多大的模型

## 问题背景

第 1-3 阶段的大多数课程用 CPU 就能跑。但一旦开始训练 CNN、Transformer 或 LLM（第 4 阶段及以后），你就需要 GPU 加速。一次在 CPU 上要跑 8 小时的训练，在 GPU 上只需 10 分钟。

你有三个选择：本地 GPU、云端 GPU，或免费的 Google Colab。

## 核心概念

```
Your options:

1. Local NVIDIA GPU
   Cost: $0 (you already have it)
   Setup: Install CUDA + cuDNN
   Best for: Regular use, large datasets

2. Google Colab (free tier)
   Cost: $0
   Setup: None
   Best for: Quick experiments, no GPU at home

3. Cloud GPU (Lambda, RunPod, Vast.ai)
   Cost: $0.20-2.00/hr
   Setup: SSH + install
   Best for: Serious training, large models
```

## 从零实现

### 方案 1：本地 NVIDIA GPU

先检查你是否有 NVIDIA GPU：

```bash
nvidia-smi
```

安装支持 CUDA 的 PyTorch：

```python
import torch

print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA version: {torch.version.cuda}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
```

### 方案 2：Google Colab

1. 打开 [colab.research.google.com](https://colab.research.google.com)
2. Runtime > Change runtime type > T4 GPU
3. 运行 `!nvidia-smi` 进行验证

本课程的 notebook 可以直接上传到 Colab 使用。

### 方案 3：云端 GPU

对于 Lambda Labs、RunPod 或 Vast.ai：

```bash
ssh user@your-gpu-instance

pip install torch torchvision torchaudio
python -c "import torch; print(torch.cuda.get_device_name(0))"
```

### 没有 GPU？没关系。

大多数课程在 CPU 上就能跑。需要 GPU 的课程会明确说明，并附上 Colab 链接。

```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using: {device}")
```

## 从零实现：GPU 与 CPU 性能对比

```python
import torch
import time

size = 5000

a_cpu = torch.randn(size, size)
b_cpu = torch.randn(size, size)

start = time.time()
c_cpu = a_cpu @ b_cpu
cpu_time = time.time() - start
print(f"CPU: {cpu_time:.3f}s")

if torch.cuda.is_available():
    a_gpu = a_cpu.to("cuda")
    b_gpu = b_cpu.to("cuda")

    torch.cuda.synchronize()
    start = time.time()
    c_gpu = a_gpu @ b_gpu
    torch.cuda.synchronize()
    gpu_time = time.time() - start
    print(f"GPU: {gpu_time:.3f}s")
    print(f"Speedup: {cpu_time / gpu_time:.0f}x")
```

## 练习

1. 运行上面的基准测试，比较 CPU 与 GPU 的耗时
2. 如果你没有 GPU，就在 Google Colab 上运行并对比结果
3. 查看你有多少 GPU 显存，估算最多能装下多大的模型（经验法则：fp16 下每个参数占 2 字节）

## 关键术语

| 术语 | 大家怎么说 | 实际含义 |
|------|----------------|----------------------|
| CUDA | “GPU 编程” | NVIDIA 的并行计算平台，让你能在 GPU 上运行代码 |
| VRAM | “GPU 内存” | GPU 上的显存（Video RAM），独立于系统内存，决定了模型大小的上限 |
| fp16 | “半精度” | 16 位浮点数，内存占用只有 fp32 的一半，精度损失极小 |
| Tensor Core | “快速矩阵硬件” | 专为矩阵乘法设计的 GPU 核心，比普通核心快 4-8 倍 |
