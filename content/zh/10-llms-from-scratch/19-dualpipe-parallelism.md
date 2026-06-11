# DualPipe 并行

> DeepSeek-V3 在 2,048 块 H800 GPU 上训练，MoE 专家分散在多个节点上。跨节点的专家 all-to-all 通信开销极高：每 1 个 GPU 小时的计算就对应 1 个 GPU 小时的通信，GPU 有一半时间在空转。DualPipe（DeepSeek，2024 年 12 月）是一种双向流水线，它把前向与反向计算和它们触发的 all-to-all 通信重叠起来。气泡减少了，吞吐量上去了；而保留两份模型参数副本（名字里的「dual」即由此而来）的代价并不高——因为专家并行（Expert Parallelism）本来就已经把专家摊到各个 rank 上了。本节课是一篇 Learn 类型的讲解，剖析 DualPipe 到底做了什么，以及为什么 Sea AI Lab 的 DualPipeV 改进版能去掉 2 倍参数开销，代价只是气泡略微变大。

**Type:** Learn
**Languages:** Python (stdlib, schedule simulator)
**Prerequisites:** Phase 10 · 05 (distributed training, FSDP, DeepSpeed), Phase 10 · 14 (open-model architectures and MoE)
**Time:** ~60 minutes

## 学习目标

- 说出 DualPipe 前向-反向块（chunk）的四个组成部分，以及为什么每个部分都有自己的重叠窗口。
- 解释大规模训练中的流水线气泡问题，以及「无气泡」在实践中和在宣传中分别意味着什么。
- 手动推演一个 8 个 PP rank、16 个 micro-batch 的 DualPipe 调度，并确认正向流与反向流恰好填补了彼此的空闲时隙。
- 陈述 DualPipeV（Sea AI Lab，2025）做出的取舍：去掉 2 倍参数复制，代价是在专家并行未启用时气泡略微变大。

## 问题背景

在 2 千块 H800 GPU 上训练一个 671B 的 MoE 模型，会遇到三个相互叠加的瓶颈：

1. **显存压力。** 每块 GPU 只持有模型的一个切片。在 8k 序列长度、61 层、128 个注意力头的配置下，激活值占用的显存非常庞大。
2. **流水线气泡。** 传统流水线并行（GPipe、1F1B）会让 GPU 在等待本阶段的输入或梯度时空转。在 8 个阶段的情况下，即便采用 1F1B 调度，气泡也可能占到约 12% 的 GPU 时间。
3. **跨节点 all-to-all。** 采用专家并行的 MoE 把专家分散到各个节点。每次前向传播都会触发一次 all-to-all 把 token 分发给对应的专家，再触发一次 all-to-all 把结果聚合回来。在 2 千块 GPU 的规模下，计算与通信之比很容易达到 1:1。

这三个问题各有独立的解法：用梯度检查点解决显存问题，用 Zero Bubble（Sea AI Lab，2023）解决流水线气泡，用专家并行通信内核解决 all-to-all。DualPipe 做的事情是让它们协同工作。它的调度在单个前向-反向块内部重叠计算与通信，同时从流水线的两端注入 micro-batch，并利用由此产生的调度把 all-to-all 藏进计算窗口里。

论文报告的结果：流水线气泡几乎被消除，在 DeepSeek-V3 的 14.8T token 训练中 GPU 利用率超过 95%。

## 核心概念

### 流水线并行回顾

把一个 N 层模型切分到 P 台设备上。设备 `i` 持有第 `i * N/P .. (i+1) * N/P - 1` 层。一个 micro-batch 先从设备 0 到 P-1 完成前向传播，再从 P-1 到 0 完成反向传播。每台设备必须等上游设备发来输出才能开始自己的前向阶段，必须等下游设备发来上游梯度才能开始反向阶段。

GPipe（Huang et al., 2019）一次只调度一个 micro-batch，浪费了大部分 GPU 时间。1F1B（Narayanan et al., 2021）让多个 micro-batch 的前向和反向交错执行。Zero Bubble（Qi et al., 2023）把反向传播拆成两部分——对输入求梯度（B）和对权重求梯度（W）——并把它们调度到气泡的位置上。在 Zero Bubble 之后，流水线已经几乎排满了。

DualPipe 是下一步。它在此之上叠加了两个想法：

### 想法 1：块分解

每个前向块被拆成四个组成部分：

- **注意力。** Q/K/V 投影、注意力计算、输出投影。
- **All-to-all dispatch。** 把 token 发送给对应专家的跨节点通信。
- **MLP。** MoE 专家计算。
- **All-to-all combine。** 把专家输出带回来的跨节点通信。

反向块则增加了上述各部分的梯度版本。DualPipe 把它们调度成：all-to-all dispatch 与下一个块的注意力计算并行执行，all-to-all combine 与再下一个块的 MLP 计算并行执行。

### 想法 2：双向调度

大多数流水线调度都是从阶段 0 注入 micro-batch，向阶段 P-1 流动。DualPipe 则从**两端**同时注入 micro-batch。阶段 0 有从它出发的前向 micro-batch；阶段 P-1 同样有从它出发的前向 micro-batch。两条流在中间汇合。

要实现这一点，设备 `i` 必须同时持有流水线前段的第 `i` 层**和**流水线后段的第 `P - 1 - i` 层。这就是 DualPipe 中「dual」的含义：每台设备为它需要服务的模型层保留两份副本（每个方向一份）。在 DeepSeek-V3 的规模下，这相当于 2 倍的参数复制开销。这个代价之所以可以接受，是因为专家并行已经把 MoE 专家摊得很薄，把非专家层复制两份只是九牛一毛。

关键在于，一个方向的前向流和另一个方向的反向流，恰好重叠在单向调度中本该出现气泡的位置上。气泡就此消失。

### 手动推演的调度

考虑 P = 4 个 rank、8 个 micro-batch，分成 4 个正向 / 4 个反向。时间从左向右推进，每一行是一个设备 rank。

```
           Time →
rank 0:  F1 F2 F3 F4  F5R F6R F7R F8R  B1 B2 B3 B4  ...
rank 1:     F1 F2 F3  F4/F5R F6R F7R   B1 B2 ...
rank 2:        F1 F2  F3/F5R F4/F6R    B1 ...
rank 3:           F1  F2/F5R F3/F6R    ...
```

如何读「F4/F5R」这个记法：rank 1 在同一个时隙里既在运行 micro-batch 4 的前向（在流水线中从左向右走），又在运行 micro-batch 5 的前向（从右向左走）。这就是「双向」在操作层面的含义。

在 rank 2 处，两条交叉流最早开始重叠；在 rank 0 和 P-1 处重叠得最晚。在调度的稳定中间阶段，每个 rank 都在执行 X 方向的前向并与 Y 方向的反向重叠。计算始终处于忙碌状态。前向传播的 all-to-all dispatch 藏在反向计算里，all-to-all combine 藏在前向计算里。气泡被挤了出去。

### 气泡核算

标准 1F1B 流水线气泡（每个 rank 浪费的时间）：

```
bubble_1F1B = (P - 1) * forward_chunk_time
```

Zero Bubble 的改进把它降下来了，但没有降到零。DualPipe 在稳定阶段，只要 micro-batch 数能被流水线深度的 2 倍整除，气泡就是零。在稳定阶段之外（预热和收尾阶段）存在一些气泡，但它不随 micro-batch 数量增长——这是论文强调的一个关键性质。

用宣传的说法：「无气泡」。用技术的说法：气泡不随 micro-batch 数量增长。Sea AI Lab 的后续分析（DualPipeV / Cut-in-half）表明，只有在专家并行不是瓶颈时才能实现完全的零气泡；一旦 EP 驱动的 all-to-all 介入，调度上总会存在一些妥协。

### DualPipeV——改进版

Sea AI Lab（2025）观察到，当 EP 通信重叠不是重点时，2 倍参数复制是一种浪费。他们的 DualPipeV 调度把双向注入折叠成一种「V 形」调度，只需单份参数副本即可运行。气泡比 DualPipe 略大，但显存节省非常可观。DeepSeek 在其开源的 DualPipe 实现中采纳了 DualPipeV，作为 EP 关闭模式。

取舍如下：

| 特性 | DualPipe | DualPipeV | 1F1B | Zero Bubble |
|---------|---------|-----------|------|------------|
| 每台设备的参数副本数 | 2 | 1 | 1 | 1 |
| 气泡随 micro-batch 数的变化 | 恒定 | 小幅增长 | 增长 | 增长 |
| 计算-通信重叠 | 完全 | 部分 | 极少 | 部分 |
| 适用场景 | EP 密集的 MoE | 稠密模型或 EP 较轻 | 基线 | 任意流水线 |

### 对 14.8T token 训练的意义

DeepSeek-V3 的预训练在 2,048 块 H800 GPU 上消耗了 14.8T token，约合 280 万 GPU 小时。如果用朴素的 1F1B，他们会有 12-15% 的算力损失在流水线气泡上——即 34-42 万 GPU 小时，足够完整训练一个 70B 模型。DualPipe 把其中大部分挽回了。在没有内部日志的情况下很难直接量化它的贡献，但论文中的说法是整个训练过程平均 GPU 利用率超过 95%。

对于较小规模的训练（1 千块 GPU 以下），DualPipe 属于杀鸡用牛刀——流水线气泡占总成本的比例更小，而且稠密模型训练很少触及 all-to-all 瓶颈。但对于数千 GPU 规模的前沿 MoE 训练，它实际上是必需品。

### 它在技术栈中的位置

- 与 **FSDP**（Phase 10 · 05）互补。FSDP 把模型参数分片到各个 rank 上；DualPipe 负责跨 rank 调度计算。二者可以组合使用。
- 与 **ZeRO-3** 梯度分片兼容。双副本复制的簿记需要与 ZeRO 的分片梯度协同工作。
- 需要针对具体集群拓扑调优的**定制 all-to-all 内核**。DeepSeek 的开源内核是参考实现。

```figure
expert-capacity
```

## 生产实践

`code/main.py` 是一个流水线调度模拟器。它接收 `(P, n_micro_batches, schedule)`，并打印 1F1B、Zero Bubble、DualPipe 和 DualPipeV 各自在稳定阶段的利用率。它是一个教学工具——这些数字与论文中的定性结论一致，但并不代表生产环境中实测的加速比。

这个模拟器的价值在于：用不同的 P 和 micro-batch 数量运行它，观察气泡占比如何在 1F1B 下增长，而在 DualPipe 下保持不变。

在真实训练中集成 DualPipe 时的考量：

- 选择一个能整除 micro-batch 数量的流水线并行深度。
- 确保你的专家并行网格支持双向 all-to-all。DeepSeek 的内核是参考实现。
- 第一次上手时，预计要在调度本身上花一周的调试时间。其中的簿记工作非常琐碎。
- 监控每个 rank 的 GPU 利用率，而不只是总体值。DualPipe 的收益来自于把拖后腿的 rank 收紧。

## 交付产物

本节课产出 `outputs/skill-dualpipe-planner.md`。给定一份训练集群规格（GPU 数量、拓扑、互连、模型形状），它会推荐一种流水线并行策略、应使用的调度算法，以及在目标规模下的预期气泡占比。

## 练习

1. 用 `(P=8, micro_batches=16, schedule=dualpipe)` 和 `(P=8, micro_batches=16, schedule=1f1b)` 运行 `code/main.py`。计算两者的 GPU 利用率差异，并把它换算成每训练一百万 token 可挽回的 GPU 小时数。

2. 手动画出 `(P=4, micro_batches=8, schedule=dualpipe)` 的调度表。在每个时隙标注 micro-batch 编号和方向。找出第一个不存在气泡的时隙。

3. 阅读 DeepSeek-V3 技术报告（arXiv:2412.19437）的 Figure 5。找出 DualPipe 前向块内部 all-to-all dispatch 的重叠窗口，并解释计算调度是如何把它隐藏起来的。

4. 分别计算 DualPipe 的 2 倍参数开销：一个 70B 稠密模型、P=8 个流水线阶段，以及一个 671B MoE 模型、P=16 个流水线阶段。说明为什么 MoE 情形的开销在比例上更小（大部分参数是专家，被分片到一个很大的 EP 组中）。

5. 把 DualPipe 与 Chimera（2021 年的一个同类双向调度器）进行比较。以论文的 Section 3.4 为参考，找出 DualPipe 新增而 Chimera 不具备的两个具体性质。

## 关键术语

| 术语 | 通俗说法 | 实际含义 |
|------|----------------|------------------------|
| 流水线气泡 | 「每个 rank 的空闲时间」 | 因流水线阶段在等待输入或梯度而浪费的 GPU 周期 |
| 1F1B | 「默认的流水线调度」 | 一前向一反向交错的调度方式；DualPipe 所对标的基线 |
| Zero Bubble | 「Sea AI Lab 2023」 | 把反向拆成 B（输入梯度）和 W（权重梯度）；几乎把流水线完全排满 |
| DualPipe | 「DeepSeek-V3 的调度」 | 双向流水线 + 计算-通信重叠；气泡不随 micro-batch 数量增长 |
| DualPipeV | 「Cut-in-half」 | V 形改进版，去掉 2 倍参数复制，代价是气泡略微变大 |
| 块（Chunk） | 「流水线工作的单元」 | 一个 micro-batch 在一个流水线阶段上的一次前向或反向传播 |
| All-to-all dispatch | 「把 token 发给专家」 | 把 token 路由到其指定 MoE 专家的跨节点通信 |
| All-to-all combine | 「把专家输出带回来」 | 在 MLP 之后收集专家输出的跨节点通信 |
| 专家并行（Expert Parallelism, EP） | 「专家分散在各 GPU 上」 | 把 MoE 专家分片到各个 rank，不同 GPU 持有不同专家 |
| 流水线并行（Pipeline Parallelism, PP） | 「层分散在各 GPU 上」 | 把模型层分片到各个 rank；DualPipe 所调度的维度 |
| 气泡占比 | 「浪费的 GPU 时间」 | (bubble_time / total_time)；DualPipe 力图压向零的比值 |

## 延伸阅读

- [DeepSeek-AI — DeepSeek-V3 Technical Report (arXiv:2412.19437), Section 3.3.2 and Figure 5](https://arxiv.org/abs/2412.19437) — DualPipe 的第一手参考资料
- [DeepSeek — DualPipe GitHub repository](https://github.com/deepseek-ai/DualPipe) — 开源参考实现，包含 DualPipeV（Cut-in-half）模式
- [Qi et al. — Zero Bubble Pipeline Parallelism (arXiv:2401.10241, Sea AI Lab 2023)](https://arxiv.org/abs/2401.10241) — Zero Bubble 这一前序工作
- [Sea AI Lab — DualPipe could be better without the Dual](https://sail.sea.com/blog/articles/63) — 促成 DeepSeek 加入 EP 关闭模式的 DualPipeV 分析
- [Narayanan et al. — PipeDream / 1F1B (arXiv:1806.03377, 2018-2021)](https://arxiv.org/abs/1806.03377) — DualPipe 用于对比的 1F1B 调度
- [Huang et al. — GPipe (arXiv:1811.06965, 2018)](https://arxiv.org/abs/1811.06965) — 最早的流水线并行论文及气泡问题
