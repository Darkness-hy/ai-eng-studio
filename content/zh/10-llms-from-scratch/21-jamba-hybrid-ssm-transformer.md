# Jamba — SSM-Transformer 混合架构

> 状态空间模型（SSM）和 Transformer 各有所长。Transformer 靠注意力换取质量，代价是二次方复杂度；SSM 靠递推换取线性时间推理和常数内存，但质量稍逊。AI21 的 Jamba（2024 年 3 月）和 Jamba 1.5（2024 年 8 月）把两者放进了同一个模型：每 7 层 Mamba 配 1 层 Transformer，每隔一层使用 MoE，256k 上下文窗口可装进单张 80GB GPU。Mamba-3（ICLR 2026）则用复数值状态空间和 MIMO 投影进一步强化了 SSM 这一侧。本课将完整解读这两种架构，并解释为什么在纯 SSM 和纯 Transformer 的长上下文尝试相继折戟的三年里，混合配方却能在规模化中存活下来。

**Type:** Learn
**Languages:** Python (stdlib, layer-mix calculator)
**Prerequisites:** Phase 10 · 14 (open-model architectures), Phase 10 · 17 (native sparse attention)
**Time:** ~60 minutes

## 学习目标

- 解释 Jamba 块中的三种基本组件——Transformer 层、Mamba 层、MoE——以及 1:7 加隔层 MoE 的交错配方。
- 从宏观层面描述 SSM 的递推形式，并说明它为何能实现常数内存推理。
- 计算 Jamba 模型在 256k 上下文下的 KV 缓存占用，并与纯 Transformer 模型所需的内存做对比。
- 说出 Mamba-3 的三项创新（指数-梯形离散化、复数值状态更新、MIMO），以及每项创新针对的问题。

## 问题背景

注意力的复杂度随序列长度呈二次方增长，而状态空间模型是线性的。这一差距会随规模放大：在 256k token 时，Transformer 每个注意力头的注意力矩阵有 650 亿个元素；而 SSM 的递推状态无论序列多长都是固定大小。

纯 SSM 模型（Mamba、Mamba-2）在小规模上能追平 Transformer 的困惑度，但在状态追踪任务上落后，在某些类型的上下文检索上则直接失败。直观原因是：SSM 把历史压缩进固定大小的状态，历史一长，信息就会泄漏。注意力则精确记住一切，但要付出二次方的代价。

显而易见的解法：两者都用。在需要精确回忆的位置放 Transformer 层，其余位置用 SSM 层，再调好两者的比例。Jamba 是第一个把这套混合配方做到生产级规模的模型（总参数 52B、激活 12B、256k 上下文、单张 80GB GPU）。Jamba 1.5 把这一家族扩展到 398B 总参数 / 94B 激活参数。Mamba-3（ICLR 2026）则是当前最强的纯 SSM 基线，未来的混合模型可以围绕它重建。

本课将通读这三篇论文，建立"选对比例"的心智模型。

## 核心概念

### 一页讲清 SSM

状态空间模型通过一个固定大小的状态 `h` 来处理序列 `x_1, ..., x_N`：

```
h_t = A h_{t-1} + B x_t
y_t = C h_t
```

每一步中，状态通过线性动力学 `A` 演化，接收输入 `B x_t`，并输出 `C h_t`。`A, B, C` 可以是学习得到的。注意一个关键性质：计算 `y_t` 只需要 `h_{t-1}` 和 `x_t`，不需要任何更早的 `x`。内存是常数级的，推理是每 token O(1)。

建模质量的诀窍在于 `A` 的结构。S4（Gu 2021）使用了高度结构化的矩阵，训练时可以高效地作为长卷积来计算。Mamba（Gu, Dao 2023）把固定的 `A, B, C` 替换成了数据依赖的版本（即"选择性"机制）。Mamba-2（2024）进一步简化了结构。Mamba-3（2026）则在特定位置重新引入了复杂性。

关键性质：对于解码器型 LLM，SSM 层可以直接替换注意力层，把不断增长的 KV 缓存换成每层固定大小的状态。

### Jamba 块

一个 Jamba 块按两个数字来交错排布各层：

- `l`：注意力与 Mamba 的比例。Jamba 取 `l = 8`，即每 7 层 Mamba 配 1 层 Transformer（7 层 Mamba + 1 层注意力 = 每组 8 层）。
- `e`：MoE 的频率。Jamba 取 `e = 2`，即每隔一层应用 MoE。

块内的层序列：

```
M  M  M  M  M  M  M  A    (7 Mamba + 1 Attention)
|  M  |  M  |  M  |  M    (where | marks MoE applied)
```

每个 Jamba 块有 8 层。堆叠 4 个块（共 32 层）后，你得到 28 层 Mamba 和 4 层注意力，其中 16 层使用 MoE。

### 为什么是 1:7

AI21 做了消融实验：在他们的长上下文评测上，什么样的注意力-Mamba 比例能同时给出最优的单位参数困惑度和上下文召回能力？

- 注意力太多（1:1）：质量上升，但内存和速度恶化。
- 注意力太少（1:15）：内存表现极佳，但上下文检索失败。
- 最佳区间：1:7 或 1:8。

直观理解：Transformer 层负责精确回忆和状态追踪，Mamba 层负责承担廉价的大批量处理。

### 位置编码

Mamba 层本身就是位置感知的（通过递推机制）。早期基于 Mamba 的混合模型中，注意力层不使用 RoPE——位置信息由 SSM 层提供。Jamba 1.5 给注意力层加上了 RoPE 以改善更长上下文的泛化能力，这是基于长上下文实测结果的事后改进。

### 内存预算

对于 Jamba-1 的模型形态（32 层：28 层 Mamba + 4 层注意力，hidden 4096，32 个注意力头）：

- KV 缓存（仅注意力层）：256k 上下文 BF16 下为 `2 * 4 * 32 * 128 * 256k * 2 = 8.4 GB`。只有那 4 层注意力层产生开销。
- SSM 状态：每个 token 前缀需要 `28 * hidden * state_size`，但这是每层固定大小，不随序列长度增长。典型的 Mamba 状态为每特征 16，hidden 4096：总计 `28 * 4096 * 16 * 2 = 3.7 MB`。

对比同样 32 层、相同 hidden、32 头全量 MHA 的纯 Transformer：256k 上下文 BF16 下为 `2 * 32 * 32 * 128 * 256k * 2 = 128 GB`。KV 缓存缩小了 8 倍。即使对比 2024 年多数模型采用的 GQA(8) 基线（`2 * 32 * 8 * 128 * 256k * 2 = 32 GB`），Jamba 1:7 混合架构的 16 GB 仍然小一半。

这就是 AI21 所说的"单张 80GB GPU 上跑 256k 上下文"的含义。全量 MHA 纯 Transformer 的 KV 缓存根本装不下；即便是 GQA 基线，剩下的空间也容不下权重和激活值；而 Jamba 的可以。

### Mamba-3：2026 年的纯 SSM 基线

Mamba-3（ICLR 2026，arXiv:2603.15569）在纯 SSM 这一侧引入了三项创新：

1. **指数-梯形离散化（Exponential-trapezoidal discretization）。** 用一种表达力更强的递推取代 Mamba-2 中的欧拉法离散化。类似卷积的操作直接作用于核心递推内部的状态-输入上，而不是作为对 `x_t` 的外部卷积。

2. **复数值状态更新（Complex-valued state update）。** 此前的 Mamba 系列把状态矩阵从复数（S4）简化为实数对角（Mamba），再简化为缩放单位阵（Mamba-2）。Mamba-3 重新引入复数值——等价于对状态施加数据依赖的旋转位置嵌入。这恢复了先前实数值简化所牺牲的状态追踪能力。

3. **多输入多输出（MIMO）投影。** 用矩阵值投影取代逐特征的标量投影。在不增加解码延迟的前提下，提升了建模能力和推理时的硬件利用率。

在 1.5B 参数规模上，Mamba-3 的平均下游准确率比 Gated DeltaNet 高 0.6 个点；MIMO 变体再加 1.2 个点，总计领先 1.8 个点。在相同状态大小下，Mamba-3 只用一半的状态就能追平 Mamba-2。

Mamba-3 尚未进入大规模生产级混合模型——但它显然是下一代 Jamba 级模型 SSM 侧的最佳候选。

### 什么时候该选混合架构

混合架构占优的场景：

- 上下文长到纯 Transformer 的 KV 缓存开始吃紧（64k 以上）。
- 任务混合了短程结构（SSM 擅长）与长程回忆（需要 Transformer）。
- 你想在单 GPU 内存预算下部署，而 Transformer 的 KV 缓存单独就装不下。

混合架构吃亏的场景：

- 上下文很短（低于 16k）。SSM 的开销被浪费了，纯 Transformer 就够。
- 任务需要全局两两交互的注意力（深度推理、多文档交叉引用）。混合架构中注意力层的稀疏性会造成损失。
- 你要把规模推向万亿参数的前沿模型。纯 Transformer + MLA + MoE（DeepSeek-V3 风格）目前在能力竞赛中领先。

### 竞争格局

| 模型 | 类别 | 规模 | 独特卖点 |
|-------|--------|------|-------------|
| Mamba-2 | 纯 SSM | 3B | 线性时间，常数内存 |
| Jamba | 混合 | 52B/12B | 80GB 上跑 256k |
| Jamba 1.5 Large | 混合 | 398B/94B | 企业级长上下文 |
| Mamba-3 | 纯 SSM | 1.5B（论文） | 恢复状态追踪能力 |
| DeepSeek-V3 | 纯 Transformer + MoE | 671B/37B | 前沿能力 |

2026 年的格局：纯 Transformer MoE 主宰前沿，但混合架构占据了 256k 以上上下文的细分市场。Mamba-3 在状态追踪上的进展，可能会让下一代混合模型把比例进一步压低（更多 SSM，更少注意力）。

```figure
swiglu-ffn
```

## 生产实践

`code/main.py` 是一个面向混合架构的内存计算器。给定 SSM-Transformer 比例和 hidden 大小 / 层数配置，它会计算：

- 目标上下文下的 KV 缓存。
- SSM 状态内存。
- 一系列模型形态在上下文长度 N 下的总内存。

该计算器支持：

- 纯 Transformer 基线（KV 缓存随 N 增长）。
- Jamba 风格的 1:7 混合架构。
- 纯 SSM（完全没有 KV 缓存）。

已发布模型形态的数字直接取自 Jamba-1 和 Jamba-1.5 论文，假想变体的数字则为外推值。

真实部署时的集成考量：

- 多数生产级推理服务器（vLLM、SGLang）支持 Jamba 和 Mamba。请核对具体版本。
- 在 256k 上下文下，Jamba 的内存优势体现在并发请求吞吐上。同样的显存里，能塞下的 Jamba 序列比 Transformer 序列更多。
- Mamba-3 作为独立模型尚未进入生产——目前是 1.5B 规模的研究预览版。

## 交付产物

本课产出 `outputs/skill-hybrid-picker.md`。给定一份工作负载说明（上下文长度分布、任务组合、内存预算），它会在纯 Transformer、Jamba 风格混合架构和纯 SSM 之间给出推荐，并对内存与质量的权衡给出明确的推理过程。

## 练习

1. 运行 `code/main.py`，计算 32 层纯 Transformer（hidden 4096，32 头）和同等形态的 Jamba-1 混合架构在 256k 上下文下的 KV 缓存。验证 AI21 论文声称的约 8 倍内存缩减。

2. 修改计算器，建模 1:3 混合（4 层 Mamba : 1 层注意力）和 1:15 混合（14 层 Mamba : 1 层注意力）。绘制 KV 缓存随比例变化的曲线。在什么比例下 KV 缓存与 SSM 状态内存相等？

3. 阅读 Jamba 论文（arXiv:2403.19887）第 3 节。解释为什么 AI21 选择 Mamba-1 而不是更快的 Mamba-2。提示：混合架构消融实验一节记录了原因。

4. 计算 Jamba 1.5 Large（总参数 398B，激活 94B）中隔层 MoE 带来的参数开销。将其激活比例与 DeepSeek-V3（37B/671B）对比，并解释为什么 Jamba 的架构会把激活比例推得更高。

5. 阅读 Mamba-3 论文（arXiv:2603.15569）第 3 节。用三句话解释为什么复数值状态更新等价于数据依赖的旋转位置嵌入。把答案与 Phase 7 · Lesson 04 的 RoPE 推导联系起来。

## 关键术语

| 术语 | 大家怎么说 | 实际含义 |
|------|----------------|------------------------|
| 状态空间模型（SSM） | "带固定状态的递推" | 带有学习得到的递推 `h_t = A h_{t-1} + B x_t` 的层；每 token 常数内存 |
| 选择性 SSM | "Mamba 的诀窍" | 数据依赖的 A、B、C 参数，让模型在线性时间内获得类似门控的选择能力 |
| 注意力-Mamba 比例 | "有多少注意力层" | 在 Jamba 中，`l = 8` 意味着每 7 层 Mamba 配 1 层注意力 |
| Jamba 块 | "那个 8 层一组" | 一层注意力 + 七层 Mamba + 隔层位置上的 MoE |
| SSM 状态 | "那个隐藏缓冲区" | 每层固定大小的状态，在 Mamba 层中替代 KV 缓存 |
| 256k 上下文 | "Jamba 的招牌数字" | Jamba-1 能装进单张 80GB GPU 的序列长度；纯 Transformer 在这个规模下装不下 |
| Mamba-3 | "2026 年的纯 SSM" | 当前最强的纯 SSM 架构，带复数状态 + MIMO；混合模型重建时的基线 |
| MIMO | "多输入多输出" | Mamba-3 的创新，用矩阵值投影取代逐特征标量投影 |
| 指数-梯形离散化 | "Mamba-3 的递推" | 表达力更强的递推，涵盖了 Mamba-2 的欧拉法离散化 |
| 混合架构 | "混合注意力和 SSM" | 任何交错排布 Transformer 和 SSM 层的模型；Jamba 是生产级的原型范例 |

## 延伸阅读

- [Lieber et al. — Jamba: A Hybrid Transformer-Mamba Language Model (arXiv:2403.19887)](https://arxiv.org/abs/2403.19887) — Jamba 原始论文，比例消融实验，256k 上下文主张
- [AI21 — Jamba 1.5: Hybrid Transformer-Mamba at Scale (arXiv:2408.12570)](https://arxiv.org/abs/2408.12570) — 规模化扩展的家族，398B/94B 和 12B/52B 公开发布版本
- [Gu, Dao — Mamba: Linear-Time Sequence Modeling with Selective State Spaces (arXiv:2312.00752)](https://arxiv.org/abs/2312.00752) — Jamba 所依赖的选择性 SSM 论文
- [Dao, Gu — Mamba-2 (arXiv:2405.21060)](https://arxiv.org/abs/2405.21060) — 简化结构化状态空间的后继版本
- [Lahoti et al. — Mamba-3 (arXiv:2603.15569, ICLR 2026)](https://arxiv.org/abs/2603.15569) — 复数值状态、MIMO，2026 年的纯 SSM 前沿
- [Gu et al. — Efficiently Modeling Long Sequences with Structured State Spaces (arXiv:2111.00396)](https://arxiv.org/abs/2111.00396) — S4 论文，LLM 领域 SSM 谱系的起点
