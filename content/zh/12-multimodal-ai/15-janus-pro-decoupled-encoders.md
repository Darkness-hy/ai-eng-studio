# Janus-Pro：统一多模态模型的解耦编码器

> 统一多模态模型存在一个无法回避的矛盾。理解任务需要语义特征——SigLIP 或 DINOv2 输出的向量富含概念级信息；生成任务需要利于重建的编码——能够还原出清晰像素的 VQ token。这两个目标无法在单一编码器中兼得。Janus（DeepSeek，2024 年 10 月）和 Janus-Pro（DeepSeek，2025 年 1 月）的主张是：干脆别再勉强——把两个编码器解耦。Transformer 主干在任务间共享，但理解走 SigLIP，生成走 VQ 分词器。在 7B 规模下，Janus-Pro 在 GenEval 上击败 DALL-E 3，同时在 MMMU 上与 LLaVA 持平。本课解读为什么双编码器能成功，而单编码器会失败。

**Type:** Build
**Languages:** Python (stdlib, dual-encoder routing + shared-body signal)
**Prerequisites:** Phase 12 · 13 (Transfusion), Phase 12 · 14 (Show-o)
**Time:** ~120 minutes

## 学习目标

- 解释为什么单一共享编码器必然会牺牲理解质量或生成质量之一。
- 描述 Janus-Pro 的路由方式：理解任务在输入侧使用 SigLIP 特征，生成任务在输入和输出两侧都使用 VQ token。
- 梳理使 Janus-Pro 成功（而 Janus 未能成功）的数据配比扩展过程。
- 比较解耦式（Janus-Pro）、耦合-连续式（Transfusion）和耦合-离散式（Show-o）三种架构。

## 问题背景

统一模型在理解和生成任务之间共享一个 Transformer 主干。此前的尝试（Chameleon、Show-o、Transfusion）都为两个方向使用同一个视觉分词器。这个分词器是一种折中：

- 为重建（生成）优化：VQ-VAE 能捕捉细粒度的像素细节，但产出的 token 语义连贯性很弱。
- 为语义（理解）优化：SigLIP 嵌入能把「猫」的图像聚到「猫」的 token 附近，却无法支持良好的重建。

Show-o 和 Transfusion 为此付出了代价：总有一个方向的质量明显打折。Janus-Pro 反问：既然两个任务的需求不同，为什么非要用同一个分词器？

## 核心概念

### 解耦的视觉编码

Janus-Pro 的架构把两个编码器分开：

- 理解路径：输入图像 → SigLIP-SO400m → 2 层 MLP → Transformer 主干。
- 生成路径：输入图像（当以已有图像为条件时）→ VQ 分词器 → token ID → Transformer 主干。
- 输出生成：Transformer 预测的图像 token → VQ 解码器 → 像素。

Transformer 主干是共享的。主干上游和下游的所有部分都是任务专属的。

输入通过提示词格式来区分：`<understand>` 标签走 SigLIP 路由；`<generate>` 走 VQ 路由。或者根据任务隐式确定路由。

### 为什么这样有效

理解损失获得的是 SigLIP 特征，而 CLIP 式预训练已经把这些特征调校得擅长语义相似度。模型的感知基准测试成绩超过 Show-o / Transfusion，因为输入特征对这个任务来说更合适。

生成损失获得的是 VQ token，而分词器已经把它们调校得擅长重建。图像质量超过 Show-o，因为 VQ 编码能干净地还原回像素。

共享的 Transformer 主干面对两种输入分布（SigLIP 和 VQ），并学会同时驾驭两者。其核心论断是：只要数据够多、参数够大，主干就能消化这种切换。

### 数据扩展——Janus 与 Janus-Pro 对比

Janus（初版，arXiv 2410.13848）首先提出了解耦思想，但规模较小（1.3B 参数，数据有限）。Janus-Pro（arXiv 2501.17811）做了扩展：

- 7B 参数（对比 1.3B）。
- 阶段 1（对齐）使用 9000 万图文对，高于此前的 7200 万。
- 阶段 2（统一训练）使用 7200 万，高于此前的 2600 万。
- 阶段 3 新增 20 万条图像生成指令样本。

结果是：Janus-Pro-7B 在 MMMU 上与 LLaVA 持平（60.3 对约 58），在 GenEval 上击败 DALL-E 3（0.80 对 0.67）。一个开源模型，在统一谱系的两端都具备竞争力。

### JanusFlow——整流流变体

JanusFlow（arXiv 2411.07975）把 VQ 生成路径换成了整流流（rectified flow）生成路径（连续式）。划分变为：理解用 SigLIP + 生成用整流流。质量上限进一步提升。架构仍然是「解耦编码器 + 共享主干」。

### 共享主干的职责

Transformer 主干处理统一的序列，但面对两种输入分布。它的职责是：

- 理解任务：消费 SigLIP 特征 + 文本 token → 自回归地输出文本。
- 生成任务：消费文本 token +（可选的图像 VQ token）→ 自回归地输出图像 VQ token。

主干的每个 block 中都没有模态专属的权重。它就是你在 Qwen 或 Llama 内部会看到的那种文本式 Transformer，外加两个输入适配器。

有意思的是，这意味着 Janus-Pro 的主干可以用预训练 LLM 来初始化。Janus-Pro 确实从 DeepSeek-MoE-7B 初始化。这个选择很关键：LLM 带来的推理能力，是纯从零训练的统一模型难以企及的。

### 与 InternVL-U 的对比

InternVL-U（第 12.10 课）是 2026 年的后续工作。它结合了：

- 原生多模态预训练（InternVL3 骨干网络）。
- 解耦编码器路由（输入侧 SigLIP，输出侧 VQ + 扩散头）。
- 统一的理解 + 生成 + 编辑。

InternVL-U 把 Janus-Pro 的架构选择吸收进了一个更大的框架。解耦编码器思想如今已成为大规模统一模型的默认方案。

### 局限

解耦编码器增加了架构复杂度：要训练两个分词器，要维护两条输入路径，要面对两套失效模式。对于不需要生成能力的产品，Janus-Pro 是过度设计——选一个 LLaVA 系的理解模型即可。

对于不需要理解能力的产品，Janus-Pro 则是大材小用——选 Stable Diffusion 3 / Flux 模型即可。

对于两者都需要的产品，Janus-Pro 如今是开源架构的参考标准。

## 生产实践

`code/main.py` 模拟 Janus-Pro 的路由：

- 两个模拟编码器：类 SigLIP（产出 256 维语义向量）和类 VQ（产出整数编码）。
- 一个提示词路由器，根据任务标签选择编码器。
- 一个共享主干（替身实现），无论 token 序列来自哪个编码器都能处理。
- 一个从阶段 1（对齐）切换到阶段 3（指令微调）的加权采样调度。

为 3 个示例打印路由路径：图像问答、文生图（T2I）、图像编辑。

## 交付产物

本课产出 `outputs/skill-decoupled-encoder-picker.md`。给定一个想要接近前沿质量的统一生成 + 理解能力的产品，它会在 Janus-Pro、JanusFlow 或 InternVL-U 中做出选择，并给出具体的数据规模建议。

## 练习

1. Janus-Pro-7B 在 GenEval 上击败了 DALL-E 3。解释为什么一个 7B 开源模型在生成上能比肩前沿闭源模型，但在理解上做不到。

2. 实现一个路由函数：给定提示词文本，分类为 `understand` 或 `generate`。对于「先描述再画出来」这类有歧义的提示词，你如何处理？

3. JanusFlow 用整流流替换了 VQ 路径。Transformer 主干现在输出什么？损失函数有什么变化？

4. 提出一个 Janus-Pro 架构再加一个解耦编码器就能处理的第四种任务。例如：图像分割（DINO 风格）、深度估计（MiDaS 风格）。

5. 阅读 Janus-Pro 论文第 4.2 节关于数据扩展的内容。相比 Janus，哪个数据阶段对 T2I 质量提升贡献最大？

## 关键术语

| 术语 | 通俗说法 | 实际含义 |
|------|-----------------|------------------------|
| 解耦编码（Decoupled encoding） | 「两个视觉编码器」 | 每个方向使用独立的分词器或编码器：理解用语义编码器，生成用重建编码器 |
| 共享主干（Shared body） | 「一个 Transformer」 | 单个 Transformer 处理任一编码器的输出；没有模态专属权重 |
| SigLIP 用于理解 | 「语义特征」 | CLIP 家族的视觉塔，提供丰富的概念级特征，但重建能力差 |
| VQ 用于生成 | 「重建编码」 | 向量量化的 token，能干净地解码回像素 |
| JanusFlow | 「整流流变体」 | 用连续流匹配生成头替换 VQ 的 Janus-Pro |
| 路由标签（Routing tag） | 「任务标签」 | 用于选择输入编码器的提示词标记（`<understand>` / `<generate>`） |

## 延伸阅读

- [Wu et al. — Janus (arXiv:2410.13848)](https://arxiv.org/abs/2410.13848)
- [Chen et al. — Janus-Pro (arXiv:2501.17811)](https://arxiv.org/abs/2501.17811)
- [Ma et al. — JanusFlow (arXiv:2411.07975)](https://arxiv.org/abs/2411.07975)
- [InternVL-U (arXiv:2603.09877)](https://arxiv.org/abs/2603.09877)
- [Dong et al. — DreamLLM (arXiv:2309.11499)](https://arxiv.org/abs/2309.11499)
