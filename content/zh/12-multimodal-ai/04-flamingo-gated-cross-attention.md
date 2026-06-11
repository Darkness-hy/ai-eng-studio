# Flamingo 与门控交叉注意力：少样本 VLM 的开端

> DeepMind 的 Flamingo（2022）做了两件前无古人的事。它证明了单个模型可以处理图像、视频、文本任意交错排列的序列；也证明了 VLM 可以进行上下文学习（in-context learning）——给一个包含三组（图像，描述）示例的少样本提示，模型无需任何梯度更新就能为新图像生成描述。其机制是门控交叉注意力（gated cross-attention）层：插入在冻结 LLM 的现有层之间，配以一个从零开始的可学习 tanh 门控，从而在初始化时完整保留 LLM 的文本能力。本课将带你走一遍 Flamingo 的 Perceiver 重采样器（Perceiver resampler）与门控交叉注意力架构——它正是 Gemini 交错输入与 Idefics2 视觉 token 的鼻祖。

**Type:** Learn
**Languages:** Python (stdlib, gated cross-attention + Perceiver resampler demo)
**Prerequisites:** Phase 12 · 03 (BLIP-2 Q-Former)
**Time:** ~120 minutes

## 学习目标

- 解释门控交叉注意力如何通过 tanh(gate) = 0 在初始化时保留冻结 LLM 的文本能力。
- 走通 Perceiver 重采样器的流程：N 个图像 patch 经交叉注意力变为 K 个固定的「潜变量」查询。
- 描述 Flamingo 如何用尊重图像位置的因果掩码处理图文交错序列。
- 复现一个少样本多模态提示结构（3 组图像-描述示例后接一张查询图像）。

## 问题背景

BLIP-2 把 32 个视觉 token 送入冻结 LLM 的输入层。每个提示只有一张图时没问题。但如果你想把*多张*图像与文本交错输入，比如「这是图 A，描述它；这是图 B，描述它；现在这是图 C，描述它」呢？LLM 的自注意力必须在同一个序列流中同时处理图像 token 和文本 token，而「哪些位置可以关注哪些图像」的问题会变得很棘手。

Flamingo 的答案：完全不动 LLM 的输入流。在现有 LLM 块之间插入额外的交叉注意力层。文本 token 照旧流经 LLM 的因果自注意力。每隔几个 LLM 块，文本 token 还会通过一个新的门控层交叉关注图像特征。门控（初始化为零）意味着第零步时这些新层等于空操作——模型的行为与预训练 LLM 完全一致。随着训练推进，门控逐渐打开，视觉信息开始流入。

Flamingo 回答的第二个问题：如何处理每个提示中数量不定的图像（0 张、1 张或多张）？用 Perceiver 重采样器——一个小型交叉注意力模块，无论你有多少个 patch，都输出固定数量的视觉潜变量 token。无论提示中有多少张图像，LLM 的交叉注意力层看到的形状始终相同。

## 核心概念

### 冻结的 LLM

Flamingo 以一个冻结的 Chinchilla 70B LLM 为起点。全部 70B 权重原封不动。原有的文本自注意力和 FFN 照常运作。

### Perceiver 重采样器

对提示中的每张图像，ViT 产生 N 个 patch token。Perceiver 重采样器拥有 K 个固定的可学习潜变量（Flamingo 用 K=64）。每个重采样器块包含两个子步骤：

1. 交叉注意力：K 个潜变量对 N 个 patch token 做注意力（Q 来自潜变量，K/V 来自 patch）。
2. 潜变量内部的自注意力 + FFN。

经过 6 个重采样器块后，输出是 K=64 个维度为 1024 的视觉 token，与 ViT 产生了多少个 patch 无关。一张 224x224 的图像（196 个 patch）和一张 480x480 的图像（900 个 patch）出口处都是 64 个重采样器 token。

对于视频，重采样器按时间维度逐帧应用：每帧的 patch 产生 64 个潜变量，再用时间位置编码让模型区分 t=0 与 t=N。整段视频最终变成 T * 64 个视觉 token。

### 门控交叉注意力

在冻结 LLM 的每 M 层之间（Flamingo 用 M=4），插入一个新的门控交叉注意力块：

```
x_after_llm_block = llm_block(x_before)
cross = cross_attn(x_after, resampler_output)
gated = tanh(alpha) * cross + x_after
x_before_next_block = gated
```

- `alpha` 是一个初始化为零的可学习标量。
- `tanh(0) = 0`，因此初始化时门控分支的贡献为零。
- 随着 `alpha` 偏离零，交叉注意力的贡献平滑增长。
- 残差连接意味着即使门控完全打开，也不会覆盖 LLM 的文本表示，只是在其之上叠加视觉信息。

这是 Flamingo 全篇最重要的设计决策：视觉条件注入是加性的、有门控的、初始化为零的。第 0 步的 Flamingo 在纯文本输入上就是一个完美的 Chinchilla 70B。

### 面向交错输入的掩码交叉注意力

在「<image A> caption A <image B> caption B <image C> ?」这样的提示中，每个文本 token 应该只看到序列中位于它之前的图像。交叉注意力掩码的约束是：位置 `t` 的文本 token 只能关注图像索引满足 `i < i_t` 的图像重采样器 token，其中 `i_t` 是位置 `t` 之前最近的图像。「只看到紧邻的前一张图像」和「看到之前的所有图像」都是合理选择；Flamingo 选了前者。

### 上下文少样本学习

一个 Flamingo 提示长这样：

```
<image1> A photo of a cat. <image2> A photo of a dog. <image3> A photo of a
```

模型识别出这个补全模式，输出 "bird"（或 image3 实际展示的内容）。不需要任何梯度更新。冻结 LLM 的上下文学习能力穿过门控交叉注意力得以保留——这正是论文的点睛之笔，也是它重要的原因。

### 训练数据

Flamingo 在三个数据集上训练：

1. MultiModal MassiveWeb（M3W）：4300 万个图文交错的网页，按阅读顺序重建。
2. 图文对（ALIGN + LTIP）：44 亿对。
3. 视频-文本对（VTP）：2700 万段短视频。

OBELICS（2023）是该交错网页语料的开源复现，Idefics、Idefics2 以及大多数开源「类 Flamingo」模型都在其上训练。

### OpenFlamingo 与 Otter

OpenFlamingo（2023）是开源复现。架构完全相同（Perceiver 重采样器 + 作用于冻结 LLaMA 或 MPT 的门控交叉注意力）。提供 3B、4B、9B 三种 checkpoint。由于基座 LLM 更小、数据更少，质量落后于 Flamingo。

Otter（2023）在 OpenFlamingo 之上用 MIMIC-IT（一个多模态指令数据集）做了指令微调，证明门控交叉注意力同样适用于指令遵循。

### 后继者

- Idefics / Idefics2 / Idefics3：Hugging Face 的门控交叉注意力谱系，一代比一代简单（Idefics2 去掉了重采样器，改用带自适应池化的直接 patch token）。
- 从 Flamingo 到 Chameleon 的转变：到 2024 年许多团队转向了早期融合（第 12.11 课）；在必须冻结主干网络的场景中，Flamingo 式门控交叉注意力仍在生产环境中使用。
- Gemini 的交错输入：在概念上继承了 Flamingo 的交错格式灵活性，尽管确切机制是闭源的。

### 与 BLIP-2 的对比

| | BLIP-2 | Flamingo |
|---|---|---|
| 视觉桥接 | 仅在输入处用一次 Q-Former | 每 M 层一个门控交叉注意力 |
| 视觉 token | 每张图 32 个 | 每张图每个交叉注意力层 64 个 |
| 冻结 LLM | 是 | 是 |
| 少样本上下文学习 | 弱 | 强——论文的核心卖点 |
| 交错输入 | 无原生支持 | 有，正是设计目标 |
| 训练数据 | 1.3 亿对 | 13 亿对 + 4300 万交错网页 |
| 训练参数量 | 1.88 亿 | 约 100 亿（交叉注意力层） |
| 算力 | 8 张 A100 数天 | 数千张 TPUv4 数周 |

预算有限的单图 VQA 选 BLIP-2。交错输入、少样本或多图推理选 Flamingo/Idefics2。

## 生产实践

`code/main.py` 演示了：

1. 一个作用于 36 个伪造 patch token、带 8 个可学习潜变量的 Perceiver 重采样器（纯 Python 交叉注意力）。
2. 一次门控交叉注意力计算：`alpha = 0` 时输出等于输入（LLM 不变），随后 `alpha = 2.0` 时视觉贡献被混入。
3. 一个交错掩码构建器，为「(image 1) (text 1) (image 2) (text 2)」序列生成 2D 注意力掩码。

## 交付产物

本课产出 `outputs/skill-gated-bridge-diagnostic.md`。给定一个开源 VLM 的配置（有无重采样器、交叉注意力频率、门控方案），它能识别其中的 Flamingo 谱系元素并解释冻结策略。在排查「微调为什么把文本性能搞砸了」时很有用（答案：门控开得太宽太快）。

## 练习

1. 计算 Flamingo-9B 的视觉参数量：9B LLM + 1.4B 门控交叉注意力层 + 64M 重采样器。被训练的参数占总参数量的多少？

2. 用 PyTorch 实现门控残差 `y = tanh(alpha) * cross + x`。通过实验证明初始化时 `alpha=0` 使 `y==x` 严格成立。

3. 阅读 OpenFlamingo 第 3.2 节（arXiv:2308.01390），了解当一个 batch 中每个提示的图像数量不同时如何处理多张图像。描述其填充（padding）策略。

4. 为什么 Flamingo 的交叉注意力掩码让文本 token 只关注*紧邻的前一张*图像，而不是之前的所有图像？阅读 Flamingo 论文第 2.4 节并解释其中的权衡。

5. 上下文少样本：为某个新的 Flamingo 变体构造一个包含 4 个「图像 → 主要物体颜色」示例的提示。描述当示例数量从 0 变到 8 时，预期的准确率变化模式。

## 关键术语

| 术语 | 大家怎么说 | 实际含义 |
|------|----------------|------------------------|
| Perceiver 重采样器 | 「固定潜变量交叉注意力」 | 从数量不定的输入 patch 中产生 K 个固定 token 的模块 |
| 门控交叉注意力 | 「tanh 门控桥」 | 残差层 `y = tanh(alpha)*cross + x`，alpha 可学习，初始化为 0 |
| 交错输入 | 「混合序列」 | 图像与文本按阅读顺序自由混排的提示格式 |
| 冻结 LLM | 「LLM 无梯度」 | 文本 LLM 的权重不更新；只训练重采样器 + 交叉注意力层 |
| 少样本 | 「上下文示例」 | 在提示中给出少量（图像，答案）对；模型无需微调即可泛化 |
| OBELICS | 「交错网页语料」 | 包含 1.41 亿个按阅读顺序保留图文的网页的开源数据集 |
| Chinchilla | 「70B 冻结基座」 | Flamingo 的冻结文本 LLM，出自 DeepMind 的 Chinchilla 论文 |
| 门控调度 | 「alpha 怎么动」 | 训练过程中交叉注意力门控打开的速率 |
| 交叉注意力频率 | 「每 M 层一个」 | 门控交叉注意力块插入的间隔；Flamingo 用 M=4 |
| OpenFlamingo | 「开源复现」 | MosaicML/LAION 的 3-9B 开源 checkpoint；架构与 Flamingo 完全一致 |

## 延伸阅读

- [Alayrac et al. — Flamingo (arXiv:2204.14198)](https://arxiv.org/abs/2204.14198) — 原始论文。
- [Awadalla et al. — OpenFlamingo (arXiv:2308.01390)](https://arxiv.org/abs/2308.01390) — 开源复现。
- [Laurençon et al. — OBELICS (arXiv:2306.16527)](https://arxiv.org/abs/2306.16527) — 交错网页语料。
- [Jaegle et al. — Perceiver IO (arXiv:2107.14795)](https://arxiv.org/abs/2107.14795) — 通用 Perceiver 架构。
- [Li et al. — Otter (arXiv:2305.03726)](https://arxiv.org/abs/2305.03726) — 指令微调的 Flamingo 后继者。
- [Laurençon et al. — Idefics2 (arXiv:2405.02246)](https://arxiv.org/abs/2405.02246) — Flamingo 方法的现代简化版。
