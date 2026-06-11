# 异步与 Hogwild! 推理

> 投机解码（Phase 10 · 15）在单个序列内部并行化 token 的生成。多智能体框架在整条序列之间并行，但要求显式协调（投票、子任务拆分）。Hogwild! Inference（Rodionov et al., arXiv:2504.06261）走了另一条路：让同一个 LLM 的 N 个实例并行运行，共享同一个键值缓存（KV cache）。每个 worker 都能立刻看到其他所有 worker 生成的 token。现代推理模型——QwQ、DeepSeek-R1——无需任何微调，就能通过这个共享缓存实现自我协调。这一方法仍处于实验阶段，但它开辟了一条全新的推理并行维度，与投机解码完全正交。本课用纯 Python 标准库实现一个双 worker 的 Hogwild! 模拟器，并解释为什么共享缓存上的协作能从模型既有的推理能力中自然涌现。

**Type:** Build
**Languages:** Python (stdlib)
**Prerequisites:** Phase 10 · 12 (inference optimization), Phase 10 · 15 (speculative decoding)
**Time:** ~60 minutes

## 学习目标

- 描述三种常见的并行 LLM 拓扑（投票、子任务、Hogwild!），并说明各自针对哪类问题。
- 陈述 Hogwild! 的核心设定：多个 worker、一个共享 KV 缓存、通过自我提示（self-prompting）涌现协调。
- 计算 Hogwild! 的实际耗时加速比，将其表示为 worker 数量 `N`、任务级并行度 `p` 和协调开销 `c` 的函数。
- 在一个玩具问题上实现双 worker 的 Hogwild! 模拟器，并观察任务分工的涌现。

## 问题背景

现代 LLM 靠产出长链推理来解决困难问题——5000 个 token 的逐步推导很常见，深度数学问题动辄数万 token。在 70B 模型上以 35 tokens/sec 的速度解码，5 万个 token 需要 24 分钟。这根本谈不上交互式体验。

投机解码（Phase 10 · 15）通过在单个序列内部并行化获得 3-5 倍加速。再往上，自回归解码的顺序依赖就是硬天花板。每个新 token 都依赖之前的全部 token。

显而易见的问题：能不能跨序列并行？让同一个模型的多个副本同时处理同一个问题，让它们相互配合、分担工作？

已有的工作包括：投票集成（运行 N 个模型，取多数答案）、思维树（tree-of-thought，分叉推理路径再合并）、多智能体框架（给每个智能体分配子任务，用一个协调者统筹）。它们在特定任务领域都有效，但也都引入了显式的协调机制——投票规则、分支剪枝逻辑、智能体间消息协议。

Hogwild! Inference 采取了不同的思路。N 个 worker 共享一个 KV 缓存。每个 worker 都能立即看到其他所有 worker 生成的 token，就像那些 token 出现在自己的上下文里一样。这些 worker 在没有任何训练或微调的情况下，自己摸索出了分工方式。现代推理模型（QwQ、DeepSeek-R1、Claude 系列的推理模式）能够读取共享缓存，并说出类似「我看到 worker 2 已经处理了基础情形，那我来做归纳步骤」这样的话。

截至 2026 年 4 月，加速效果依赖具体工作负载，且仍属实验性质。但这个想法值得了解，因为它开辟了推理并行的新维度。

## 核心概念

### 基本设定

初始化 N 个 worker 进程，全部运行同一个 LLM。不为每个 worker 单独维护 KV 缓存，而是只维护一个共享缓存。当 worker `i` 生成 token `t_j` 时，这个 token 会被写入共享缓存的下一个位置。当 worker `k` 执行下一步时，它读取缓存的当前状态（其中包含所有 N 个 worker 到目前为止生成的全部内容）。

在每个步骤上，worker 之间竞争写入 token。没有按 worker 划分的位置索引——缓存就是一条不断增长的序列，顺序由写入到达的时间决定。

### 为什么协调会涌现

所有 worker 共享同一个提示词。通常类似于：「你是共同处理这个问题的 N 个实例之一。每个实例都能读取共享内存，看到其他实例写入的内容。请避免重复劳动。」提示词加上共享缓存就足够了。推理模型会阅读缓存，注意到问题的哪些部分已经有人尝试过，然后（通常但并非总是）转向尚未探索的部分。

Hogwild! 论文（Rodionov et al., 2025）报告了如下观察：

- worker 会制定计划，并通过缓存把计划传达给其他 worker。
- worker 会注意到其他 worker 推理中的错误并指出来。
- worker 会在计划失败时做出调整并提出替代方案。
- 在被提示检查冗余时，worker 能发现冗余并转换方向。

这一切都不需要微调。这种涌现行为来自模型本身已有的推理能力。

### 命名由来

论文名称致敬了 Hogwild! SGD（Recht et al., 2011），一种异步更新的优化器。两者的类比关系是：SGD 的异步 worker 都向同一个共享参数向量写入；Hogwild! Inference 的 worker 都向同一个共享 KV 缓存写入。两者都依赖经验上的收敛，而非同步保证。

### RoPE 让这件事变得可行

旋转位置编码（Rotary Position Embeddings，RoPE，Su et al. 2021）通过对 Q 和 K 向量做旋转来编码位置信息。因为位置是旋转而不是固化的偏移量，一个 token 的位置可以移动而无需重新计算其 KV 缓存条目。当 worker `i` 在位置 `p` 写入共享缓存时，其他 worker 读取该位置可以直接使用缓存条目——不需要重新旋转。

如果换成可学习位置编码或绝对位置编码的模型，Hogwild! 在每次并发写入时都需要让缓存失效。RoPE 让缓存保持稳定。

### 耗时计算

设 `T_serial` 为单个 worker 独自解决问题所需的时间。设 `p` 为任务级可并行的比例。设 `c` 为每步的协调开销（读取变长的缓存、决定写什么）。

单 worker 时间：`T_serial`。
N 个 worker 的 Hogwild! 时间，若协调零成本：`T_serial * ((1 - p) + p / N)`。经典的 Amdahl 定律。
计入协调开销：`T_serial * ((1 - p) + p / N) + c * steps_per_worker`。

要让 worker 真正有产出，`c` 必须远小于每步解码时间。在产出 5k+ token 的推理模型上，worker 即便花费数百个 token 的协调开销仍然划算。在简短的聊天任务上，协调成本占主导，Hogwild! 反而不如串行。

### 具体例子

推理问题：1 万 token 的思维链。假设问题的可并行内容占比 `p = 0.7`（不同的证明策略、不同的分类讨论），每个 worker 的协调开销 `c = 200` 个 token。在 `N = 4` 个 worker 下：

- 串行时间：10000 个解码步。
- Hogwild! 时间：10000 * (0.3 + 0.7 / 4) + 200 * 4 = 10000 * 0.475 + 800 = 5550 个解码步。
- 加速比：10000 / 5550 = 1.8 倍。

这个数字并不惊艳。但在更长的推理问题上（5 万 token），协调开销被摊薄，加速比能逼近 2.5-3 倍。Hogwild! 之于推理，就相当于线程级并行之于一门能自然写多线程代码的编程语言。

### 什么时候该用 Hogwild!

- 长推理问题（数千 token 起），且任务能拆成相互独立的子目标并行处理。
- 经过逐步思考训练的推理模型。非推理模型无法很好地自我协调。
- 单节点部署，且 VRAM 足以容纳共享缓存外加 N 个 worker 进程。缓存是共享的，但每个 worker 有自己的激活内存。

### 什么时候不该用

- 简短的交互式聊天。协调开销占主导。
- 无法并行的任务（单条线性证明、单次编译）。N=1 就是上限。
- 非推理模型。不会涌现任何协调。
- 多节点部署。共享缓存需要非常快的跨 worker 同步。节点内没问题；跨节点是延迟灾难。

### 实验性现状

截至 2026 年 4 月，Hogwild! 仍是一个研究方法，有开源的 PyTorch 实现。生产环境尚无落地。三个阻碍：

1. 跨并发进程的共享 KV 缓存管理是不小的工程难题。
2. 涌现式协调依赖具体任务；基准测试体系仍在建设中。
3. 相比投机解码已有的收益，其加速幅度不算大，两者可以叠加，但叠加的工程实现又是一层复杂度。

值得了解，值得做实验，但还不值得把产品押在上面。

```figure
continuous-batching
```

## 从零实现

`code/main.py` 实现了一个玩具版 Hogwild! 模拟器：

- 两个 worker 进程，每个都是一个确定性的「LLM」，按已知概率产出若干 token 类别之一（工作 token、观察 token、协调 token）。
- 一个共享缓存（就是一个 token 列表），两个 worker 都读写它。
- 一套简单的协调逻辑：当一个 worker 看到对方已在某个类别上产出了足够多的工作 token，它就改选另一个类别。

模拟器在固定的步数预算内运行，并报告：

- 产出的工作 token 总数。
- 总耗时（worker 步数）。
- 相对单 worker 的有效加速比。
- 每个 token 由哪个 worker 写入的轨迹。

### 第 1 步：共享缓存

一个两个 worker 都向其追加的列表。真实实现中用简单的加锁（Python `threading.Lock`）；我们用一个计数器来模拟。

### 第 2 步：worker 循环

每个 worker 在每一步：

- 读取当前的共享缓存。
- 根据缓存里已有的内容，决定要写入哪个类别的 token。
- 写入一个 token。

### 第 3 步：协调启发式

如果类别 X 在缓存中已有 K 个 token，而 worker 原本打算写类别 X，那么它改写类别 Y。这是对推理模型「发现这部分已经有人覆盖了，那我换个方向」行为的玩具化替代。

### 第 4 步：测量加速比

分别用 N=1 和 N=2 个 worker 运行模拟器，总步数预算相同。统计产出的工作 token 数。N=2 应当多产出大约 1.5-1.8 倍的工作 token，这归功于协调驱动的任务分工。

### 第 5 步：给协调施压

降低协调启发式的灵敏度，再跑一次。可以观察到：缺乏良好协调时，N=2 会重复产出相同的 token，加速比跌破 1。这与论文的观察一致：这个技巧只有在 worker 具备自我协调的推理能力时才奏效。

## 生产实践

截至 2026 年 4 月，Hogwild! 在生产环境的集成仍停留在研究级别。来自 Yandex/HSE/IST 的参考实现基于 PyTorch，目标是 DeepSeek-R1 和 QwQ 模型上的单节点多进程部署。

务实的采用路径：

1. 剖析你的推理任务负载。测量探索性 token（多策略、分类讨论、搜索）与线性 token 的占比。
2. 如果探索占主导，跑一个双 worker 的 Hogwild! 实验，测量实际耗时的改善。
3. 如果改善低于 1.3 倍，说明你处在协调主导的区间，退回单 worker。
4. 如果改善超过 1.5 倍，推进到 N=4 再测一次。收益递减通常出现在 N=4-8 附近。

与投机解码组合：每个 Hogwild! worker 可以独立使用投机解码。两个加速比（近似）相乘，3 倍的投机解码加上 1.8 倍的 Hogwild!，相对朴素的单 worker 解码可达到约 5.4 倍的有效加速。

## 交付产物

本课产出 `outputs/skill-parallel-inference-router.md`。给定一个推理负载画像（token 预算、任务并行度特征、模型家族、部署目标），它在投票、思维树、多智能体、Hogwild! 和投机解码等策略之间做路由选择。

## 练习

1. 用默认设置运行 `code/main.py`。确认在相同实际耗时下，N=2 的 Hogwild! 配置比 N=1 基线产出更多工作 token。

2. 削弱协调启发式的强度（设置 `coordination_weight=0.1`），重新运行。展示加速比崩塌的现象，并解释原因：当 worker 无法协调时，它们会重复劳动。

3. 计算一个 5 万 token 推理任务在 `p=0.8, c=500`、N=4 个 worker 下的预期 Hogwild! 加速比。再对一个 1 千 token 聊天任务在 `p=0.3, c=200`、N=4 下做同样计算。为什么一个赚、一个赔？

4. 阅读 Hogwild! 论文的第 4 节（初步评估）。找出作者报告的两种失败模式，并说明更好的协调提示词分别可以如何缓解它们。

5. 在玩具模拟器中把 Hogwild! 与投机解码组合：每个 worker 内部使用 2-token 的投机解码。报告相乘后的加速比。当两个 worker 都想扩展同一个共享缓存前缀时，会出现什么记账（bookkeeping）问题？

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|----------------|------------------------|
| Hogwild! | 「并行 worker，共享缓存」 | 同一个 LLM 的 N 个实例并发运行，共用一个 KV 缓存；通过自我提示涌现协调 |
| 共享 KV 缓存 | 「协调的媒介」 | 一个所有 worker 共同读写的、持续增长的 KV 缓冲区；让 token 在 worker 之间即时可见 |
| 涌现式协调 | 「无需训练」 | 具备推理能力的 LLM 可以读取共享缓存并分工，不需要任何微调或显式协议 |
| 协调开销（c） | 「花在定位上的 token」 | 每个 worker 读取变长缓存并决定下一步动作的成本；必须远小于总解码时间 |
| 可并行比例（p） | 「能并行跑的部分」 | 任务级并行度：总工作量中并非本质上串行的那部分比例 |
| RoPE 成就了 Hogwild! | 「旋转位置具有平移不变性」 | 因为位置是旋转，写入共享缓存不需要重新计算之前的 token |
| 投票集成 | 「跑 N 个，取多数」 | 最简单的并行推理拓扑；适合分类任务，不太适合长篇推理 |
| 思维树 | 「分支再剪枝」 | 探索多条分支并剪枝的推理策略；带有显式协调逻辑 |
| 多智能体框架 | 「分配子任务」 | 每个智能体有一个角色，由协调者统筹；协议开销沉重 |

## 延伸阅读

- [Rodionov et al. — Hogwild! Inference: Parallel LLM Generation via Concurrent Attention (arXiv:2504.06261)](https://arxiv.org/abs/2504.06261) — Hogwild! 论文，在 QwQ 和 DeepSeek-R1 上的初步评估
- [Recht, Re, Wright, Niu — Hogwild!: A Lock-Free Approach to Parallelizing Stochastic Gradient Descent (arXiv:1106.5730, NeurIPS 2011)](https://arxiv.org/abs/1106.5730) — 原版 Hogwild!，命名的出处
- [Su et al. — RoFormer: Enhanced Transformer with Rotary Position Embedding (arXiv:2104.09864)](https://arxiv.org/abs/2104.09864) — RoPE，正是它的性质让共享缓存推理变得可行
- [Yao et al. — Tree of Thoughts: Deliberate Problem Solving with Large Language Models (arXiv:2305.10601)](https://arxiv.org/abs/2305.10601) — 思维树推理策略，与 Hogwild! 正交
- [Leviathan et al. — Fast Inference from Transformers via Speculative Decoding (arXiv:2211.17192)](https://arxiv.org/abs/2211.17192) — 投机解码，Hogwild! 可与之组合的序列内并行方法
- [Hogwild! reference PyTorch implementation](https://github.com/eqimp/hogwild_llm) — 论文实验的唯一权威实现
