# 规模化：分布式训练、FSDP 与 DeepSpeed

> 你的 1.24 亿参数模型在一块 GPU 上训练完成了。现在试试 70 亿参数。模型放不进显存，数据在单台机器上要跑好几周。在大规模场景下，分布式训练不是可选项，而是唯一的出路。

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 10, Lesson 04 (Pre-Training a Mini GPT)
**Time:** ~120 minutes

## 学习目标

- 解释三种并行方式（数据并行、张量并行、流水线并行），并能根据模型规模和集群规模判断各自的适用场景
- 使用 PyTorch DDP 实现数据并行训练，在多块 GPU 之间同步梯度
- 计算给定模型规模的显存预算（权重 + 优化器状态 + 梯度 + 激活值），以确定最低硬件要求
- 配置 FSDP 或 DeepSpeed ZeRO 各阶段，将模型状态分片到多块 GPU 上，从而装下超出单卡显存的模型

## 问题背景

一个 70 亿（7B）参数的模型，仅 FP16 权重就需要 14GB。Adam 优化器还要为每个参数额外存两份副本（一阶矩和二阶矩估计），又是 28GB。反向传播期间的梯度再加 14GB。在存储任何一个激活值之前，你已经用掉了 56GB。

一块 NVIDIA A100 有 80GB 显存。

80GB 中已消耗 56GB，留给激活值（activations）的只有 24GB——激活值是前向传播过程中计算出的中间结果，必须保留到反向传播时使用。对于 2048 个 token 的序列、4096 维的模型，单层激活值约占 64MB。32 层下来，每个样本需要 2GB。批大小为 8 就需要 16GB。你只有 24GB，批大小到 12 就会爆显存。

再试试 700 亿（70B）参数。仅权重就要 140GB（FP16），单块 GPU 装不下。光是放权重就至少需要 2 块 A100（2 x 80GB = 160GB）。加上优化器状态和梯度，需求会大得多：至少 3 块 GPU 起步，实际取决于分片策略，通常需要 8 到 16 块。

Llama 3 405B 是在 16,384 块 NVIDIA H100 GPU 上训练的，这次训练的算力成本估计高达 1 亿美元。DeepSeek V3 训练了一个水平相当的模型，只花了约 560 万美元——靠的是巧妙的架构设计（混合专家（Mixture of Experts）意味着每个 token 只激活一小部分参数）和更高的训练效率。

这节课讲解让大规模训练成为可能的四种策略：数据并行、张量并行、流水线并行，以及完全分片数据并行。在接触任何分布式训练框架之前，你将先用纯 Python 模拟每一种策略，搞清楚其中的机制。

## 核心概念

### 为什么必须分布式

下面是真实模型的显存计算。每个数字都是算出来的，不是估的。

| 模型 | 参数量 | 权重（FP16） | Adam 状态 | 梯度（FP16） | 总计（不含激活值） |
|-------|--------|----------------|-------------|------------------|----------------------|
| GPT-2 Small | 124M | 248 MB | 992 MB | 248 MB | 1.5 GB |
| Llama 3 8B | 8B | 16 GB | 64 GB | 16 GB | 96 GB |
| Llama 3 70B | 70B | 140 GB | 560 GB | 140 GB | 840 GB |
| Llama 3 405B | 405B | 810 GB | 3,240 GB | 810 GB | 4,860 GB |

"Adam 状态"这一列才是致命的。Adam 为每个参数维护一个滑动均值（m）和一个滑动方差（v），且都用 FP32 存储。对一个 70B 模型来说，就是 70B x 4 字节 x 2 = 560GB。仅优化器就需要七块 A100。

单块 H100 有 80GB。Llama 3 405B 仅存放权重、优化器和梯度就至少需要 61 块 H100，加上激活值还会更多。Meta 用了 16,384 块 GPU，不是因为他们想用——是因为不得不用。

### 数据并行

最简单的分布式策略。把整个模型复制到 N 块 GPU 上，把每个训练批次切成 N 等份。每块 GPU 在自己那份数据分片上执行前向和反向传播。反向传播结束后，对所有 GPU 的梯度求平均。每块 GPU 用同样的平均梯度更新自己的权重副本，使所有副本保持同步。

**优点：** 吞吐量线性扩展。N 块 GPU 每步处理 N 倍的数据。通信只发生在梯度平均环节，并且可以与计算重叠。

**缺点：** 每块 GPU 都要持有模型、优化器状态和梯度的完整副本。对一个 70B 模型来说，每块 GPU 需要 840GB。数据并行完全不能降低单卡显存占用，它只能缩短训练时间。

**计算公式：** 有效批大小 = per_gpu_batch_size x N。N=64 块 GPU、单卡批大小为 16 时，有效批大小是 1,024。Llama 3 每步使用的有效批大小是 1600 万个 token。

```mermaid
graph TD
    subgraph DataParallel["Data Parallelism (N=4 GPUs)"]
        B["Full Batch\n(1024 samples)"] --> S["Split"]
        S --> G1["GPU 1\nFull Model Copy\n256 samples"]
        S --> G2["GPU 2\nFull Model Copy\n256 samples"]
        S --> G3["GPU 3\nFull Model Copy\n256 samples"]
        S --> G4["GPU 4\nFull Model Copy\n256 samples"]
        G1 --> AR["AllReduce\nAverage Gradients"]
        G2 --> AR
        G3 --> AR
        G4 --> AR
        AR --> U["Update\n(identical on all GPUs)"]
    end

    style B fill:#1a1a2e,stroke:#e94560,color:#fff
    style G1 fill:#1a1a2e,stroke:#0f3460,color:#fff
    style G2 fill:#1a1a2e,stroke:#0f3460,color:#fff
    style G3 fill:#1a1a2e,stroke:#0f3460,color:#fff
    style G4 fill:#1a1a2e,stroke:#0f3460,color:#fff
    style AR fill:#1a1a2e,stroke:#51cf66,color:#fff
    style U fill:#1a1a2e,stroke:#51cf66,color:#fff
```

### 张量并行

把单个层拆分到多块 GPU 上。一次矩阵乘法被分摊给多块 GPU，每块只算结果的一部分。

考虑前馈层中一个形状为 (8192, 8192) 的权重矩阵。采用 4 路张量并行时，每块 GPU 持有一个 (8192, 2048) 的分片。每块 GPU 用输入乘以自己的分片，得到一个部分结果。这些部分结果再通过 all-reduce 或 all-gather 合并成完整输出。

**优点：** 降低单卡上模型权重的显存占用。一个 70B 模型拆到 8 块 GPU 上，每块只需持有约 8.75B 参数对应的权重。

**缺点：** 每过一层都需要 GPU 之间的高速通信。每次矩阵乘法之后的 all-reduce 都会增加延迟。这在 NVLink 上（同节点 GPU 之间 900 GB/s）表现良好，但在用 InfiniBand 连接的跨节点场景下（400 Gb/s，约 50 GB/s）就很糟。张量并行几乎总是限制在单个节点内部（8 块 GPU）。

**实际应用：** Megatron-LM 开创了张量并行。Llama 3 405B 在每个节点内使用 8 路张量并行。

### 流水线并行

按层切分模型。GPU 1 跑第 1-8 层，GPU 2 跑第 9-16 层，GPU 3 跑第 17-24 层，GPU 4 跑第 25-32 层。数据沿流水线流动：GPU 1 算完自己的层后把激活值发给 GPU 2，GPU 2 算完发给 GPU 3，依此类推。

**优点：** GPU 之间的通信极少——只在层边界传递激活值，而激活值相比梯度或权重要小得多。由于带宽要求低，跨节点也能正常工作。

**缺点：** 流水线气泡（pipeline bubble）。当 GPU 4 在对第 1 个微批次做前向传播时，GPU 1、2、3 处于空闲（它们已经把自己的部分算完传走了）。反向传播时模式正好反过来。在最朴素的流水线方案下，N 个流水线阶段的 GPU 利用率只有 1/N。

**GPipe 和 PipeDream** 通过把批次拆成微批次（micro-batch）来解决气泡问题。GPU 1 一完成微批次 1 的前向传播，立刻开始处理微批次 2。这样各流水线阶段的计算可以相互重叠。对于 M 个微批次和 N 个阶段，气泡占比降到 (N-1)/M。用 M=16 个微批次配 N=4 个阶段，气泡就是 3/16 = 18.75% 的空闲时间。

### FSDP：完全分片数据并行

FSDP（Fully Sharded Data Parallel，完全分片数据并行）把数据并行的可扩展性与分片的显存效率结合了起来。每块 GPU 不再持有完整的模型副本，而是只持有 1/N 的参数、梯度和优化器状态。

在某一层的前向传播之前，FSDP 执行一次 **all-gather**，把完整参数从所有 GPU 收集到每块 GPU 的显存中。前向传播结束后，每块 GPU 丢弃非本地的参数。反向传播时，all-gather 再次执行，重建参数以计算梯度。反向传播结束后，一次 **reduce-scatter** 把梯度分片分发出去，每块 GPU 只存 1/N 的梯度。

**70B 模型放在 8 块 GPU 上的计算：**

| 组成部分 | 不使用 FSDP | 使用 FSDP |
|-----------|-------------|-----------|
| 权重（FP16） | 每 GPU 140 GB | 每 GPU 17.5 GB |
| Adam 状态（FP32） | 每 GPU 560 GB | 每 GPU 70 GB |
| 梯度（FP16） | 每 GPU 140 GB | 每 GPU 17.5 GB |
| **总计** | **每 GPU 840 GB** | **每 GPU 105 GB** |

没有 FSDP，70B 模型无论如何也装不进一块 80GB 的 GPU。用 FSDP 跑在 8 块 GPU 上，每块占用 105GB——等等，这还是装不下。你至少需要 16 块 GPU 才能把单卡占用压到 80GB 以下，或者把 FSDP 和激活检查点（activation checkpointing，在反向传播时重新计算激活值而不是存储它们）结合使用。

由于每层之前都要做一次 all-gather，通信开销比普通数据并行更高。但省下的显存让原本不可能的训练任务变成了可能。

```mermaid
graph TD
    subgraph FSDP["FSDP: Fully Sharded Data Parallel (4 GPUs)"]
        direction TB
        S["Model: 4 layers, sharded"]

        subgraph GPU1["GPU 1"]
            G1S["Shard: 1/4 params\n1/4 optimizer\n1/4 gradients"]
        end
        subgraph GPU2["GPU 2"]
            G2S["Shard: 1/4 params\n1/4 optimizer\n1/4 gradients"]
        end
        subgraph GPU3["GPU 3"]
            G3S["Shard: 1/4 params\n1/4 optimizer\n1/4 gradients"]
        end
        subgraph GPU4["GPU 4"]
            G4S["Shard: 1/4 params\n1/4 optimizer\n1/4 gradients"]
        end

        AG["All-Gather\n(reconstruct full params\nbefore each layer)"]
        FW["Forward Pass\n(full params temporarily)"]
        RS["Reduce-Scatter\n(distribute gradient shards\nafter backward)"]

        S --> GPU1
        S --> GPU2
        S --> GPU3
        S --> GPU4
        GPU1 --> AG
        GPU2 --> AG
        GPU3 --> AG
        GPU4 --> AG
        AG --> FW
        FW --> RS
    end

    style G1S fill:#1a1a2e,stroke:#0f3460,color:#fff
    style G2S fill:#1a1a2e,stroke:#0f3460,color:#fff
    style G3S fill:#1a1a2e,stroke:#0f3460,color:#fff
    style G4S fill:#1a1a2e,stroke:#0f3460,color:#fff
    style AG fill:#1a1a2e,stroke:#e94560,color:#fff
    style FW fill:#1a1a2e,stroke:#51cf66,color:#fff
    style RS fill:#1a1a2e,stroke:#e94560,color:#fff
```

### DeepSpeed ZeRO

DeepSpeed 的 ZeRO（Zero Redundancy Optimizer，零冗余优化器）在概念上与 FSDP 完全一致，只是由 Microsoft 独立开发。它定义了三个阶段，每个阶段的分片力度逐级加大：

| 阶段 | 分片对象 | 显存节省 | 通信开销 |
|-------|--------|---------------|---------------|
| ZeRO-1 | 仅优化器状态 | 约 4 倍缩减 | 与数据并行相同 |
| ZeRO-2 | + 梯度 | 约 8 倍缩减 | 略有增加 |
| ZeRO-3 | + 参数 | 约 N 倍缩减（N 块 GPU） | 每层一次 all-gather |

ZeRO-3 与 FSDP 等价，只是名字不同，机制完全一样。在 DeepSpeed 验证了这套思路之后，PyTorch 才把 FSDP 加入官方原生实现。

DeepSpeed 还推出了 ZeRO-Offload（把优化器状态卸载到更便宜、容量更大的 CPU 内存）和 ZeRO-Infinity（卸载到 NVMe SSD）。它们用计算速度换显存容量——被卸载的操作更慢，但能腾出 GPU 显存。

### 混合精度训练

现代训练会同时使用多种浮点格式：

- **前向传播**：FP16 或 BF16（16 位）。显存只有 FP32 的一半，矩阵乘法在 tensor core 上快 2 倍。
- **主权重（master weights）**：FP32（32 位）。由优化器维护，保证权重更新时的数值精度。
- **损失缩放（loss scaling）**：在反向传播前把损失乘以一个大常数，防止 FP16 梯度下溢为零；在优化器更新前再除回同一常数。

BF16（Brain Float 16）的指数范围与 FP32 相同（8 位指数位），但精度更低（7 位尾数位，FP32 是 23 位）。由于能表示同样范围的数值，它几乎不需要损失缩放。FP16 有 5 位指数位和 10 位尾数位——它能表示更细粒度的数值，但在极端量级下会上溢或下溢。

Google 的 TPU 原生使用 BF16。NVIDIA 的 A100 和 H100 同时支持 FP16 和 BF16。业界已基本转向 BF16，因为它省去了损失缩放的各种麻烦。

**7B 模型的显存对比：**

| 精度 | 权重 | 优化器 | 梯度 | 总计 |
|-----------|---------|-----------|-----------|-------|
| 全程 FP32 | 28 GB | 56 GB | 28 GB | 112 GB |
| 混合精度（BF16 + FP32 主权重） | 14 GB | 56 GB | 14 GB | 84 GB |

在这个模型上混合精度省了 28GB。无论用什么精度，优化器状态始终保持 FP32——大部分显存都耗在这里。

### Megatron-LM 与 3D 并行

真实的大规模训练会把三种并行全部组合起来：

- **数据并行**：跨节点组使用（扩大批大小）
- **张量并行**：节点内部使用（把层拆到 8 块 GPU 上）
- **流水线并行**：跨节点使用（把层组拆到多台机器上）

Llama 3 405B 在 16,384 块 H100 上：
- 节点内 8 路张量并行（每节点 8 块 GPU）
- 跨节点 16 路流水线并行（16 个流水线阶段）
- 剩余维度上 128 路数据并行（16,384 / 8 / 16 = 128）

这种 3D 分解（8 x 16 x 128 = 16,384）就是把规模扩展到数千块 GPU 的方法。每块 GPU 看到不同的数据分片（数据并行），持有每层的一个切片（张量并行），并计算不同的一组层（流水线并行）。

DeepSeek V3 走了另一条路。它的混合专家架构在每个 token 上只激活 671B 参数中的 37B。这意味着每块 GPU 只需要为激活的参数计算（并存储激活值）。他们用 2,048 块 H800 GPU 完成训练——不到 Meta GPU 数量的 1/8——花费 560 万美元，而 Meta 的估计是 1 亿美元。

```mermaid
graph TD
    subgraph ThreeD["3D Parallelism (Llama 3 405B)"]
        direction TB
        subgraph DP["Data Parallel (128-way)\nSplit batch across 128 groups"]
            subgraph PP["Pipeline Parallel (16-way)\nSplit layers across 16 stages"]
                subgraph TP["Tensor Parallel (8-way)\nSplit each layer across 8 GPUs"]
                    G1["GPU 1\nSlice of layers 1-N"]
                    G2["GPU 2\nSlice of layers 1-N"]
                    G8["GPU 8\nSlice of layers 1-N"]
                end
            end
        end
    end

    N1["Total: 8 x 16 x 128 = 16,384 GPUs"]

    style G1 fill:#1a1a2e,stroke:#0f3460,color:#fff
    style G2 fill:#1a1a2e,stroke:#0f3460,color:#fff
    style G8 fill:#1a1a2e,stroke:#0f3460,color:#fff
    style N1 fill:#1a1a2e,stroke:#e94560,color:#fff
```

```figure
paged-kv-cache
```

## 从零实现

### 第 1 步：模拟数据并行

把一个批次拆分到多个模拟 GPU 上。每个 GPU 在自己的分片上执行前向计算，然后对"梯度"求平均（这里用损失值来模拟）。

```python
import numpy as np

def simulate_data_parallelism(data, num_gpus, model_fn):
    batch_size = len(data)
    shard_size = batch_size // num_gpus
    remainder = batch_size % num_gpus

    gpu_losses = []
    gpu_gradients = []

    offset = 0
    for gpu_id in range(num_gpus):
        extra = 1 if gpu_id < remainder else 0
        shard = data[offset:offset + shard_size + extra]
        offset += shard_size + extra

        loss, grad = model_fn(shard)
        gpu_losses.append(loss)
        gpu_gradients.append(grad)

    avg_loss = np.mean(gpu_losses)
    avg_gradient = np.mean(gpu_gradients, axis=0)

    return avg_loss, avg_gradient
```

all-reduce 操作（梯度平均）是数据并行中唯一的通信。在实践中，NVIDIA GPU 上通过 NCCL 库完成，它实现了环形 all-reduce（ring all-reduce）：每块 GPU 把自己 1/N 的梯度发给一侧邻居，从另一侧邻居接收 1/N，经过 N-1 步后，每块 GPU 都拿到了完整的平均值。总通信量为 2 x gradient_size x (N-1)/N，当 N 较大时趋近于梯度大小的 2 倍。

### 第 2 步：模拟张量并行

把一个权重矩阵拆分到多个 GPU 上。每个 GPU 计算一部分矩阵乘法，再把结果合并。

```python
def simulate_tensor_parallelism(input_data, weight_matrix, num_gpus):
    d_in, d_out = weight_matrix.shape
    assert d_out % num_gpus == 0, f"d_out {d_out} not divisible by num_gpus {num_gpus}"
    shard_size = d_out // num_gpus

    partial_results = []
    for gpu_id in range(num_gpus):
        start = gpu_id * shard_size
        end = start + shard_size
        weight_shard = weight_matrix[:, start:end]

        partial = input_data @ weight_shard
        partial_results.append(partial)

    full_output = np.concatenate(partial_results, axis=-1)

    direct_output = input_data @ weight_matrix
    error = np.abs(full_output - direct_output).max()

    return full_output, error
```

误差应当严格为零（或在机器精度量级）。张量并行在数学上是精确的——它产生的结果与在单块 GPU 上算完整矩阵乘法完全相同。这里沿输出维度切分，每块 GPU 生成不同的一段列，拼接后即可还原完整结果。

列并行（column-parallel）线性层（切分输出维度）用拼接合并结果；行并行（row-parallel，切分输入维度）则用求和。在 Transformer 的 FFN 中，第一个线性层（升维）用列并行，第二个线性层（降维）用行并行，这样两层之间就省掉了一次 all-reduce。

### 第 3 步：模拟流水线并行

把模型的层拆分到多个虚拟 GPU 上，展示气泡问题：后面的阶段在计算时，前面的阶段只能空等。

```python
def simulate_pipeline_parallelism(num_layers, num_stages, num_microbatches):
    layers_per_stage = num_layers // num_stages

    timeline = {}
    clock = 0

    for mb in range(num_microbatches):
        for stage in range(num_stages):
            start_time = max(
                timeline.get((stage, mb - 1, "fwd"), (0, 0))[1] if mb > 0 else 0,
                timeline.get((stage - 1, mb, "fwd"), (0, 0))[1] if stage > 0 else 0,
            )
            end_time = start_time + layers_per_stage
            timeline[(stage, mb, "fwd")] = (start_time, end_time)

    last_fwd_end = max(v[1] for v in timeline.values())

    for mb in range(num_microbatches - 1, -1, -1):
        for stage in range(num_stages - 1, -1, -1):
            deps = [last_fwd_end]
            if mb < num_microbatches - 1 and (stage, mb + 1, "bwd") in timeline:
                deps.append(timeline[(stage, mb + 1, "bwd")][1])
            if stage < num_stages - 1 and (stage + 1, mb, "bwd") in timeline:
                deps.append(timeline[(stage + 1, mb, "bwd")][1])
            start_time = max(deps)
            end_time = start_time + layers_per_stage
            timeline[(stage, mb, "bwd")] = (start_time, end_time)

    total_time = max(v[1] for v in timeline.values())
    compute_time = num_microbatches * num_stages * layers_per_stage * 2
    bubble_fraction = 1.0 - compute_time / (total_time * num_stages)

    return timeline, total_time, bubble_fraction
```

4 个阶段、1 个微批次时，气泡占比是 75%——任意时刻四块 GPU 中有三块在空转。用 16 个微批次后降到约 19%。消除气泡的代价是显存：所有在途微批次的激活值必须同时保存。

### 第 4 步：显存计算器

精确计算任意模型规模训练时的显存需求。

```python
def memory_calculator(
    params_billions,
    precision_bytes=2,
    optimizer="adam",
    num_gpus=1,
    sharding="none",
    sequence_length=2048,
    batch_size_per_gpu=1,
    hidden_dim=None,
    num_layers=None,
):
    params = params_billions * 1e9

    weight_memory = params * precision_bytes

    if optimizer == "adam":
        optimizer_memory = params * 4 * 2
    elif optimizer == "sgd":
        optimizer_memory = params * 4
    else:
        optimizer_memory = 0

    gradient_memory = params * precision_bytes

    total_no_activation = weight_memory + optimizer_memory + gradient_memory

    if hidden_dim and num_layers:
        activation_per_layer = (
            sequence_length * batch_size_per_gpu * hidden_dim * precision_bytes * 4
        )
        activation_memory = activation_per_layer * num_layers
    else:
        activation_memory = params * precision_bytes * 0.5

    if sharding == "fsdp" or sharding == "zero3":
        weight_memory /= num_gpus
        optimizer_memory /= num_gpus
        gradient_memory /= num_gpus
    elif sharding == "zero2":
        optimizer_memory /= num_gpus
        gradient_memory /= num_gpus
    elif sharding == "zero1":
        optimizer_memory /= num_gpus

    per_gpu_total = weight_memory + optimizer_memory + gradient_memory + activation_memory

    return {
        "params_billions": params_billions,
        "weights_gb": weight_memory / 1e9,
        "optimizer_gb": optimizer_memory / 1e9,
        "gradients_gb": gradient_memory / 1e9,
        "activations_gb": activation_memory / 1e9,
        "per_gpu_total_gb": per_gpu_total / 1e9,
        "total_across_gpus_gb": per_gpu_total * num_gpus / 1e9,
        "fits_on_80gb": per_gpu_total / 1e9 <= 80,
        "num_gpus": num_gpus,
        "sharding": sharding,
    }
```

这个计算器回答了每个 ML 工程师都会问的问题："我需要几块 GPU？"输入模型规模，看看是否装得下，再调整分片策略，直到单卡总占用降到 80GB 以下。

### 第 5 步：混合精度模拟

对比 FP32、FP16 和混合精度训练的显存占用。

```python
def mixed_precision_comparison(params_billions):
    params = params_billions * 1e9

    fp32_weights = params * 4
    fp32_optimizer = params * 4 * 2
    fp32_gradients = params * 4
    fp32_total = fp32_weights + fp32_optimizer + fp32_gradients

    fp16_weights = params * 2
    fp16_master = params * 4
    fp16_optimizer = params * 4 * 2
    fp16_gradients = params * 2
    fp16_total = fp16_weights + fp16_master + fp16_optimizer + fp16_gradients

    mixed_weights = params * 2
    mixed_optimizer = params * 4 * 2
    mixed_gradients = params * 2
    mixed_total = mixed_weights + mixed_optimizer + mixed_gradients

    return {
        "fp32_total_gb": fp32_total / 1e9,
        "fp16_with_master_gb": fp16_total / 1e9,
        "mixed_bf16_gb": mixed_total / 1e9,
        "savings_vs_fp32": 1 - mixed_total / fp32_total,
    }
```

大多数人最意外的一点：混合精度并不会把显存砍半。无论用什么精度，优化器状态（Adam 的 m 和 v）始终保持 FP32。一个 7B 模型，FP32 训练用 112GB，混合精度用 84GB——降幅是 25%，不是 50%。优化器才是显存大头。

## 生产实践

### 运行全部模拟

```python
def run_all_demos():
    print("=" * 70)
    print("DATA PARALLELISM SIMULATION")
    print("=" * 70)

    np.random.seed(42)
    data = np.random.randn(64, 32)
    weight = np.random.randn(32, 16)

    def model_fn(batch):
        output = batch @ weight
        loss = np.mean(output ** 2)
        grad = 2 * batch.T @ (batch @ weight) / len(batch)
        return loss, grad

    for n_gpus in [1, 2, 4, 8]:
        loss, grad = simulate_data_parallelism(data, n_gpus, model_fn)
        print(f"  {n_gpus} GPUs: loss={loss:.4f}, grad_norm={np.linalg.norm(grad):.4f}")

    print()
    print("=" * 70)
    print("TENSOR PARALLELISM SIMULATION")
    print("=" * 70)

    x = np.random.randn(4, 8192)
    W = np.random.randn(8192, 8192)

    for n_gpus in [1, 2, 4, 8]:
        output, error = simulate_tensor_parallelism(x, W, n_gpus)
        print(f"  {n_gpus} GPUs: output_shape={output.shape}, max_error={error:.2e}")

    print()
    print("=" * 70)
    print("PIPELINE PARALLELISM SIMULATION")
    print("=" * 70)

    for n_mb in [1, 4, 8, 16, 32]:
        _, total_t, bubble = simulate_pipeline_parallelism(32, 4, n_mb)
        print(f"  {n_mb:2d} micro-batches: total_time={total_t:4d}, bubble={bubble:.1%}")

    print()
    print("=" * 70)
    print("MEMORY CALCULATOR")
    print("=" * 70)

    configs = [
        (7, "none", 1),
        (7, "fsdp", 8),
        (70, "none", 1),
        (70, "fsdp", 8),
        (70, "fsdp", 16),
        (405, "fsdp", 64),
        (405, "fsdp", 128),
    ]

    print(f"  {'Model':>8} {'Sharding':>8} {'GPUs':>5} {'Per-GPU':>10} {'Fits 80GB':>10}")
    print("  " + "-" * 50)
    for params, shard, gpus in configs:
        result = memory_calculator(params, num_gpus=gpus, sharding=shard)
        fits = "Yes" if result["fits_on_80gb"] else "No"
        print(f"  {params:>6}B {shard:>8} {gpus:>5} {result['per_gpu_total_gb']:>8.1f}GB {fits:>10}")

    print()
    print("=" * 70)
    print("MIXED PRECISION COMPARISON")
    print("=" * 70)

    for params_b in [7, 13, 70, 405]:
        result = mixed_precision_comparison(params_b)
        print(f"  {params_b}B: FP32={result['fp32_total_gb']:.0f}GB, "
              f"Mixed BF16={result['mixed_bf16_gb']:.0f}GB, "
              f"Savings={result['savings_vs_fp32']:.0%}")
```

## 交付产物

本课的产出是 `outputs/prompt-distributed-training-planner.md`——一个提示词，输入模型规模和可用硬件后，它会生成一份完整的分布式训练方案：并行策略、显存预算、通信开销和预期吞吐量。

## 练习

1. 修改显存计算器，加入激活检查点支持。启用检查点后，只在每隔 K 层的位置存储激活值（典型值 K=1，即全部重算）。展示显存与计算的权衡：检查点能省多少显存，又会让训练慢多少（完全检查点大约增加 33% 的计算量）？

2. 扩展流水线并行模拟，实现 PipeDream 使用的 1F1B（one forward, one backward）调度。在 4 个阶段、8 个微批次的设置下，对比 1F1B 与朴素调度的气泡占比。1F1B 调度由于更早开始反向传播，峰值显存应该更低。

3. 实现一个梯度累积模拟器。不在每个微批次后都做 all-reduce，而是先在本地累积 K 步梯度，再统一 all-reduce。展示这样做能把通信量减少 K 倍，同时最终梯度完全相同（因此训练结果也完全相同）。

4. 构建一个成本估算器。给定模型规模、目标 token 数、GPU 类型（A100 每小时 2 美元，H100 每小时 3.5 美元）和并行策略，估算总训练成本（美元）。用已知数据验证：Llama 3 405B 据报道花费约 1 亿美元，DeepSeek V3 约 560 万美元。

5. 给显存计算器加上 ZeRO-Offload。假设每节点 CPU 内存为 512GB，NVMe 为 2TB。展示把优化器状态卸载到 CPU 后，70B 模型只需 4 块 GPU 即可训练（而不是 16 块），代价是优化器步骤慢 30-50%。

## 关键术语

| 术语 | 人们常说的 | 实际含义 |
|------|----------------|----------------------|
| 数据并行 | "把模型复制到每块 GPU" | 每块 GPU 处理不同的数据分片；每步之后通过 all-reduce 对梯度求平均 |
| 张量并行 | "把一层拆到多块 GPU 上" | 切分权重矩阵，每块 GPU 计算矩阵乘法的一部分；需要 NVLink 级别的高速互连 |
| 流水线并行 | "把各层拆到多块 GPU 上" | 每块 GPU 运行不同的层组；数据沿流水线流动，用微批次减少气泡 |
| FSDP | "把所有东西都分片" | 完全分片数据并行——每块 GPU 持有 1/N 的权重、梯度和优化器状态；计算前先 all-gather |
| ZeRO | "DeepSpeed 版的 FSDP" | 零冗余优化器，分 3 个阶段：分片优化器状态（Stage 1）、+ 梯度（Stage 2）、+ 参数（Stage 3） |
| All-reduce | "跨 GPU 求平均" | 一种集合通信操作，结束时每块 GPU 都得到所有 GPU 输入的总和（或平均值）——通常用环形 all-reduce 实现 |
| All-gather | "从所有 GPU 收集" | 一种集合通信操作，结束时每块 GPU 都得到所有 GPU 数据的拼接结果——FSDP 用它重建完整参数 |
| Reduce-scatter | "求和并分发" | 一种集合通信操作，先对数据做归约（求和），再把不同的块分发到不同 GPU——FSDP 用它做梯度分片 |
| 混合精度 | "用半精度训练" | 前向/反向用 FP16/BF16，优化器状态用 FP32——节省约 25% 的显存而不是 50%，因为优化器才是大头 |
| 流水线气泡 | "流水线中的空闲时间" | GPU 等待上一阶段数据而空闲的时间占比——通过增加微批次数量来降低 |

## 延伸阅读

- [Rajbhandari et al., 2020 -- "ZeRO: Memory Optimizations Toward Training Trillion Parameter Models"](https://arxiv.org/abs/1910.02054)——定义了三个分片阶段的 DeepSpeed ZeRO 论文
- [Shoeybi et al., 2020 -- "Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism"](https://arxiv.org/abs/1909.08053)——NVIDIA 面向 Transformer 的张量并行
- [Narayanan et al., 2021 -- "Efficient Large-Scale Language Model Training on GPU Clusters Using Megatron-LM"](https://arxiv.org/abs/2104.04473)——结合数据、张量和流水线的 3D 并行
- [Zhao et al., 2023 -- "PyTorch FSDP: Experiences on Scaling Fully Sharded Data Parallel"](https://arxiv.org/abs/2304.11277)——PyTorch 官方原生 FSDP 实现
- [Llama 3 Technical Report](https://arxiv.org/abs/2407.21783)——16,384 块 GPU 训练及 3D 并行细节
- [DeepSeek-V3 Technical Report](https://arxiv.org/abs/2412.19437)——MoE 架构如何把训练成本降低一个数量级
