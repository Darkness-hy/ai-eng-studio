# 梯度检查点与激活重计算

> 反向传播会保留每一个中间激活值。对于 70B 参数、128K 上下文的模型，每个 rank 要存 3 TB 的激活值。检查点技术用 FLOPs 换内存：与其保存，不如重算。问题在于该丢弃哪些段，而答案并不是"全部丢掉"。

**Type:** Build
**Languages:** Python (with numpy, optional torch)
**Prerequisites:** Phase 10 Lesson 04 (Pre-Training Mini-GPT), Phase 10 Lesson 05 (Scaling & Distributed)
**Time:** ~70 minutes

## 问题背景

训练 Transformer 时，每一层都要保存反向传播中所有需要求导的算子的输入：注意力的输入、Q/K/V 投影、softmax 输出、FFN 的输入、归一化层的输出，以及残差流。对于隐藏维度为 `d`、序列长度为 `L`、批大小为 `B` 的一层，这个量级约为每层 `12 * B * L * d` 个浮点数。

当 `d=8192, L=8192, B=1` 时，BF16 下每层就是 800 MB。一个 64 层的模型光激活值就要 51 GB——这还没乘上微批数量，没算注意力 softmax 的中间结果（每个头 `L^2`），也没算张量并行带来的部分副本。

账要从两头算：BF16 权重加上优化器状态也许能塞进 80GB，但激活值会把你顶爆。梯度检查点（gradient checkpointing，又称激活重计算 activation recomputation）是标准解法。丢掉大部分激活值；在反向传播时重跑一遍前向，把它们算回来。代价：额外的 FLOPs。收益：内存按检查点段数与总层数之比下降。

简单粗暴地做检查点，每步前向 FLOPs 大约多出 33%。做得好——按 Korthikanti 等人的"smart selection"做选择性检查点——你可以用不到 5% 的 FLOP 开销换 5 倍的内存节省。而在 FP8 矩阵乘、FSDP 卸载和专家并行 MoE 的场景下这一点格外重要：你既浪费不起内存，也浪费不起算力。

## 核心概念

### 反向传播到底需要什么

`output = layer(input)`。反向传播要算 `grad_input` 和 `grad_params`。为此它需要：

- `input`（线性层需要它来计算 `grad_params = input.T @ grad_output`）
- 一些激活函数的导数中间量（ReLU/GELU/softmax 的导数依赖于激活值本身）

前向传播会自动把这些存进自动求导图。每个 `tensor.retain_grad()` 和每个需要保留输入的算子都会持有一份引用。

### 朴素的全量检查点

把网络切成 `N` 段。前向时只保存每段的*输入*。当反向传播需要中间量时，重跑该段的前向把它们物化出来，再求导。

例子：32 层 Transformer 切成 32 段，每段 1 层。

- 内存：32 个层输入（很小）对比 32 * （每层的激活量）（巨大）。
- 额外计算：每段多跑 1 次前向，即总前向 FLOPs 增加约 33%（因为反向是前向的 2 倍，整步从 1 + 2 = 3 个单位变成 1 + 1 + 2 = 4 个单位）。

这就是 Chen 等人 2016 年的原始方案：每 `sqrt(L)` 层放一个检查点，以平衡内存与计算。L=64 时就是 8 个检查点。

### 选择性检查点（Korthikanti 2022）

并非所有激活值的代价都相同。注意力 softmax 的输出是 `B*L*L*heads`，随序列长度*二次*增长。FFN 隐藏层激活是 `B*L*4d`，线性增长。序列一长，softmax 就成了大头。

选择性检查点保留那些存储便宜的激活值（线性投影、残差），只重算昂贵的部分（注意力）。你只付出极少的重算 FLOPs，却省下了 O(L^2) 的内存。

Megatron-Core 把它实现为 "selective" 激活重计算模式。2024 年以后的大多数前沿训练都在用。

### 卸载

重计算之外的另一条路：在前向和反向之间把激活值搬到 CPU 内存。这需要 PCIe 带宽；当空闲带宽的成本低于重新物化的成本时就划算。混合策略很常见：一部分层做检查点，另一部分做卸载。

FSDP2 把卸载作为一等公民选项提供。当 GPU 卡在内存上、而 CPU-GPU 传输还有余量时，卸载最能发挥作用。

### 重计算成本模型

在 `L` 层中每 `k` 层做一次朴素检查点时，每步 FLOPs：

```
flops_fwd_normal = L * f_layer
flops_bwd_normal = 2 * L * f_layer
flops_total_normal = 3 * L * f_layer

flops_fwd_ckpt = L * f_layer
flops_recompute = L * f_layer  # one extra forward per layer in the segment
flops_bwd_ckpt = 2 * L * f_layer
flops_total_ckpt = 4 * L * f_layer
overhead = 4 / 3 - 1 = 0.33 = 33%
```

采用选择性检查点时，只重算注意力核函数而非整层：

```
flops_recompute_selective = L * f_attention ~= L * f_layer * 0.15
overhead_selective = (3 + 0.15) / 3 - 1 = 0.05 = 5%
```

### 内存节省模型

每层激活量记为 `A`。`L` 层的总激活内存为 `L * A`。

全量检查点（段大小为 1）：只存 `L * input_volume`（标准 Transformer 约为 `L * 1/10 A`）。节省约 `9 * L * A * 1/10`。

每 `k` 层一个检查点：存 `L/k * A`，再加上当前活跃段内 `k-1` 层的激活量。

当 `k = sqrt(L)` 时，内存和重算成本都按 `sqrt(L)` 增长——这是各层成本均匀时的最优折中。

### 什么时候不该做检查点

- 流水线阶段中已经在执行的最内层。它们反正要算完。
- 首层和末层，如果它们占据了该阶段的主要计算量（在 Transformer 中很少见）。
- 已经使用 FlashAttention 的注意力核——Flash 本身就会快速重算 softmax，再叠加层级检查点收益甚微。

### 实现模式

1. **函数包装器：** 用 `torch.utils.checkpoint.checkpoint(fn, input)` 把一段包起来。PyTorch 只存 `input`，反向时把其余的都重算出来。

2. **基于装饰器：** 给层打上"可检查点"标签；由训练器在配置阶段决定哪些段被包装。

3. **手动显式重算：** 自己写反向传播，调用一个自定义的 `recompute_forward`，用保存的输入复刻一遍前向。

三种方式在功能上结果相同。包装器是标准惯用法。

### 与 TP / PP / FP8 的交互

- **张量并行：** 检查点输入在重算时必须重新 gather 或 scatter；要把通信成本算进去。
- **流水线并行：** 典型做法是对每个流水线阶段的前向做检查点，这样逆序处理的微批可以复用激活内存。
- **FP8 重算：** 重算期间更新的 amax 历史必须与原始前向一致，否则 FP8 缩放因子会漂移。多数框架会对缩放因子做快照。

## 从零实现

### Step 1: 一个带分段的玩具模型

```python
import numpy as np


def linear_forward(x, w, b):
    return x @ w + b


def relu(x):
    return np.maximum(x, 0)


def layer_forward(x, w1, b1, w2, b2):
    h = relu(linear_forward(x, w1, b1))
    return linear_forward(h, w2, b2)


def model_forward(x, params):
    activations = [x]
    h = x
    for w1, b1, w2, b2 in params:
        h = layer_forward(h, w1, b1, w2, b2)
        activations.append(h)
    return h, activations
```

### Step 2: 需要全部激活值的朴素反向传播

```python
def model_backward(grad_output, activations, params):
    grads = [None] * len(params)
    g = grad_output
    for i in range(len(params) - 1, -1, -1):
        w1, b1, w2, b2 = params[i]
        x_in = activations[i]
        h_pre = linear_forward(x_in, w1, b1)
        h = relu(h_pre)
        gh = g @ w2.T
        gw2 = h.T @ g
        gb2 = g.sum(axis=0)
        g_pre = gh * (h_pre > 0)
        gx = g_pre @ w1.T
        gw1 = x_in.T @ g_pre
        gb1 = g_pre.sum(axis=0)
        grads[i] = (gw1, gb1, gw2, gb2)
        g = gx
    return g, grads
```

### Step 3: 每 k 层做检查点的内存方案

```python
def model_forward_checkpointed(x, params, k=4):
    saved_inputs = [x]
    h = x
    for i, (w1, b1, w2, b2) in enumerate(params):
        h = layer_forward(h, w1, b1, w2, b2)
        if (i + 1) % k == 0:
            saved_inputs.append(h)
    return h, saved_inputs


def model_backward_checkpointed(grad_output, saved_inputs, params, k=4):
    grads = [None] * len(params)
    g = grad_output
    segments = [(j * k, min((j + 1) * k, len(params))) for j in range(len(saved_inputs))]
    for seg_idx in range(len(saved_inputs) - 1, -1, -1):
        start, end = segments[seg_idx]
        if start >= end:
            continue
        x_in = saved_inputs[seg_idx]
        _, seg_acts = model_forward(x_in, params[start:end])
        g, seg_grads = model_backward(g, seg_acts, params[start:end])
        for j, gr in enumerate(seg_grads):
            grads[start + j] = gr
    return g, grads
```

### Step 4: 成本模型

```python
def checkpoint_cost(n_layers, segment_size, flops_per_layer=1.0):
    fwd = n_layers * flops_per_layer
    recompute = n_layers * flops_per_layer
    bwd = 2 * n_layers * flops_per_layer
    return {
        "fwd": fwd,
        "recompute": recompute,
        "bwd": bwd,
        "total": fwd + recompute + bwd,
        "overhead_vs_no_ckpt": (fwd + recompute + bwd) / (fwd + bwd) - 1.0,
    }


def selective_checkpoint_cost(n_layers, attention_fraction=0.15,
                              flops_per_layer=1.0):
    fwd = n_layers * flops_per_layer
    recompute = n_layers * attention_fraction * flops_per_layer
    bwd = 2 * n_layers * flops_per_layer
    return {
        "fwd": fwd,
        "recompute": recompute,
        "bwd": bwd,
        "total": fwd + recompute + bwd,
        "overhead_vs_no_ckpt": (fwd + recompute + bwd) / (fwd + bwd) - 1.0,
    }
```

### Step 5: 内存估算器

```python
def activation_memory_mb(n_layers, hidden=8192, seq=8192,
                        batch=1, bytes_per_value=2):
    per_layer = 12 * batch * seq * hidden * bytes_per_value
    return n_layers * per_layer / 1e6


def memory_after_checkpoint(n_layers, segment_size, hidden=8192,
                           seq=8192, batch=1, bytes_per_value=2):
    n_seg = max(1, n_layers // segment_size)
    saved = (n_seg + segment_size) * 1 * batch * seq * hidden * bytes_per_value
    return saved / 1e6
```

### Step 6: 最优段大小

```python
def optimal_segment(n_layers):
    return int(round(np.sqrt(n_layers)))
```

### Step 7: 选择性检查点决策

```python
def should_recompute(layer_type, activation_bytes, recompute_flops_ratio):
    if layer_type == "attention" and activation_bytes > 100 * 1e6:
        return True
    if layer_type == "ffn" and activation_bytes > 500 * 1e6:
        return recompute_flops_ratio < 0.1
    return False
```

## 生产实践

- **torch.utils.checkpoint**：`from torch.utils.checkpoint import checkpoint`——PyTorch 中的标准包装器。包装一个函数；只保存输入，反向时重算。
- **Megatron-Core 激活重计算**：支持 `selective`、`full` 和 `block` 三种模式。2024 年以后前沿训练的标配。
- **FSDP2 卸载**：在 FSDP2 中用 `module.to_empty(device="cpu")` 配合 `offload_policy`，把激活值分片到 CPU 而不是重算。
- **DeepSpeed ZeRO-Offload**：对优化器状态和激活值做 CPU 卸载，与检查点技术互补。

## 交付产物

本课产出 `outputs/prompt-activation-recompute-policy.md`——一个提示词，输入你的模型配置（层数、隐藏维度、序列长度、批大小）和可用 GPU 内存，输出逐层的重算策略（none / selective / full / offload）。

## 练习

1. 验证正确性。分别运行 `model_forward` + `model_backward`（全量激活）和 `model_forward_checkpointed` + `model_backward_checkpointed`（分段）。参数梯度必须在机器精度内完全一致。

2. 把段大小 `k` 从 1 扫到 `L`。画出 FLOP 开销和内存的曲线。找出曲线的拐点。

3. 实现选择性检查点：保存注意力模块的输入但不存其中间量。在 32 层、seq=8192 的模型上，测量它相对整层检查点的 FLOP 开销。

4. 加入卸载。把段输入保存到一个模拟的"CPU 缓冲区"（一个独立的 list）。用字节数/时间来度量"PCIe 带宽"，找出卸载与重算之间的盈亏平衡点。

5. 在真实的 PyTorch Transformer 上分别开关 `torch.utils.checkpoint` 做基准测试。测量内存（通过 `torch.cuda.max_memory_allocated`）和单步耗时。

## 关键术语

| 术语 | 大家怎么说 | 实际含义 |
|------|----------------|----------------------|
| 梯度检查点（Gradient checkpointing） | "靠重跑前向来省内存" | 只保存段输入；反向传播时重算中间量，得到梯度计算所需的张量 |
| 激活重计算（Activation recomputation） | "和检查点是一回事" | 同一技术的 HPC 风格叫法 |
| 段大小（k） | "每个检查点管几层" | 中间量被一并丢弃并一并重新物化的层数 |
| 选择性检查点 | "Korthikanti 的招" | 只重算存储昂贵的激活值（注意力 softmax）；便宜的照常保留 |
| 全量检查点 | "朴素版本" | 每个段里每一层的中间量都重算 |
| 块检查点（Block checkpointing） | "粗粒度" | 以整个 Transformer 块为单位做检查点；粒度最大 |
| FLOP 开销 | "算力税" | 每步额外 FLOPs = (重算 FLOPs) / (前向 + 反向 FLOPs)；朴素 33%，选择性 5% |
| 激活卸载（Activation offload） | "搬去 CPU" | 在前向到反向之间把激活值移到 CPU 内存；重算的替代方案 |
| sqrt-L 法则 | "经典最优解" | 各层成本均匀时，最优检查点间隔是 sqrt(L) 层 |
| 注意力 softmax 体积 | "O(L^2) 问题" | L^2 * heads * batch 个浮点数；长上下文下主导激活内存 |

## 延伸阅读

- [Chen et al., 2016 -- "Training Deep Nets with Sublinear Memory Cost"](https://arxiv.org/abs/1604.06174) -- 形式化梯度检查点的开山论文
- [Korthikanti et al., 2022 -- "Reducing Activation Recomputation in Large Transformer Models"](https://arxiv.org/abs/2205.05198) -- 选择性激活重计算及其正式的成本分析
- [Pudipeddi et al., 2020 -- "Training Large Neural Networks with Constant Memory using a New Execution Algorithm"](https://arxiv.org/abs/2002.05645) -- 基于反向模式重物化的常数内存替代方案
- [Ren et al., 2021 -- "ZeRO-Offload: Democratizing Billion-Scale Model Training"](https://arxiv.org/abs/2101.06840) -- 大规模激活卸载
- [PyTorch torch.utils.checkpoint docs](https://pytorch.org/docs/stable/checkpoint.html) -- 标准 API
- [Megatron-Core activation recomputation documentation](https://docs.nvidia.com/nemo-framework/user-guide/latest/nemotoolkit/features/memory_optimizations.html) -- selective、full 和 block 模式
