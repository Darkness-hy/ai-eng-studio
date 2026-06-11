# 视觉自回归建模（VAR）：下一尺度预测

> 扩散模型在时间维度上迭代采样（去噪步骤），而 VAR 在尺度维度上迭代采样——先预测一个 1x1 的 token，然后是 2x2、4x4，直到最终分辨率，每个尺度都以前一个尺度为条件。2024 年的论文表明，VAR 在图像生成上呈现出与 GPT 风格相同的缩放定律，并且在相同计算预算下击败了 DiT。本课将实现其核心机制。

**Type:** Build
**Languages:** Python (with PyTorch)
**Prerequisites:** Phase 7 Lesson 03 (Multi-Head Attention), Phase 8 Lesson 06 (DDPM)
**Time:** ~90 minutes

## 问题背景

自回归生成之所以主导了语言建模，是因为它具有可预测的扩展性：更多计算、更多参数，就能换来更低的困惑度和更好的输出。在 2024 年之前，图像生成有过两类主要的自回归（AR）尝试：PixelRNN/PixelCNN（逐像素生成）和 DALL-E 1 / Parti / MuseGAN（在 VQ-VAE 编码上逐 token 生成）。

两者都受困于生成顺序问题。像素和 token 排列在二维网格上，但 AR 模型必须按一维光栅顺序访问它们。位于角落的早期像素完全不知道整张图像最终会变成什么样。其生成质量的扩展性比 GPT 在文本上的表现更差，并且在相同计算量下从未达到扩散模型的质量。

VAR 通过改变"生成的对象"来解决生成顺序问题。它不再在空间中逐个预测图像 token，而是以递增的分辨率预测整张图像。第 1 步：预测一个 1x1 的 token（整张图像的"摘要"）。第 2 步：预测一个 2x2 的 token 网格（较粗的特征）。第 3 步：预测一个 4x4 的网格。第 K 步：预测最终的 (H/8)x(W/8) 网格。

每个尺度关注所有先前的尺度（按"尺度顺序"保持因果性），并在自身尺度内并行生成。顺序问题随之消失：尺度 k 上的整张图像在一次 Transformer 前向传递中产生。

## 核心概念

### VQ-VAE 多尺度分词器

VAR 需要一个**多尺度离散分词器**（multi-scale discrete tokenizer）。对于一张图像 x，它会生成一系列分辨率逐步提高的 token 网格：

```
x -> encoder -> latent f
f -> tokenize at 1x1: token grid z_1 of shape (1, 1)
f -> tokenize at 2x2: token grid z_2 of shape (2, 2)
...
f -> tokenize at (H/p)x(W/p): token grid z_K of shape (H/p, W/p)
```

每个 z_k 使用同一个码本（codebook，典型大小为 4096-16384）。各尺度的 token 化并非相互独立——训练目标是让各尺度残差之和能够重建 f：

```
f ≈ upsample(embed(z_1), target_size) + ... + upsample(embed(z_K), target_size)
```

这是一种**残差 VQ**（residual VQ）变体。尺度 k 捕获尺度 1..k-1 遗漏的信息。解码器接收所有尺度嵌入之和，并输出图像。

多尺度 VQ 分词器只训练一次（类似 VQGAN），随后冻结。所有生成工作都由其之上的自回归模型完成。

### 下一尺度预测

生成模型是一个 Transformer，它看到所有先前尺度的 token，并预测下一个尺度的 token。

输入序列结构：
```
[START, z_1 tokens, z_2 tokens, z_3 tokens, ..., z_K tokens]
```

位置嵌入同时编码尺度索引和尺度内的空间位置。注意力在尺度顺序上是因果的：位于尺度 k、位置 (i, j) 的 token 可以关注尺度 1..k 上的所有 token，以及尺度 k 内部按某种尺度内顺序排在前面的 token（VAR 使用固定位置注意力，尺度内没有因果约束——一个尺度内的所有位置并行预测）。

训练损失：在每个尺度 k 上，给定所有先前尺度的 token 来预测 token z_k。对离散 VQ 编码计算交叉熵损失。结构与 GPT 相同，只是"序列"如今按尺度组织。

### 生成

推理时：
```
generate z_1 = sample from p(z_1)                    # 1 token
generate z_2 = sample from p(z_2 | z_1)              # 4 tokens in parallel
generate z_3 = sample from p(z_3 | z_1, z_2)         # 16 tokens in parallel
...
decode: f = sum of embed-and-upsample scales 1..K
image = VAE_decoder(f)
```

当 K = 10 个尺度时，生成只需 10 次 Transformer 前向传递。每次传递并行产出整个尺度——尺度内没有逐 token 的自回归。对于一张 256x256 的图像，大约只需 10 次传递，而 DiT 需要 28-50 次。

### 为什么下一尺度预测胜过下一 token 预测

三个结构性优势：
1. **由粗到细契合自然图像的统计特性。** 人类视觉感知和图像数据集都呈现出与尺度相关的规律：低频结构稳定且可预测；高频细节则取决于低频内容。下一尺度预测正好利用了这一点。
2. **尺度内并行生成。** 不同于 GPT 式的逐 token 自回归，VAR 在一步内产出一个尺度上的所有 token。有效生成长度是对数级而非线性级。
3. **没有生成顺序偏置。** 尺度 k 上的 token 能看到尺度 k-1 的全部内容；不存在"左侧"或"上方"偏置，不会迫使早期 token 在后续上下文可用之前就做出承诺。

### 缩放定律

Tian 等人证明，VAR 在 ImageNet 上的 FID 遵循幂律缩放曲线——就像 GPT 的困惑度一样。参数或计算量翻倍，误差就可靠地减半。这是第一个像语言模型一样干净地展现这种缩放行为的图像生成模型。其结果是：VAR 规模下的性能可以从计算量直接预测出来，而不再是针对每种架构的经验性猜测。

### 与扩散模型的关系

VAR 和扩散模型共享同一套数据压缩思路：两者都把生成问题拆解为一系列更容易的子问题。

- 扩散：逐步添加噪声，学习撤销其中一步。
- VAR：逐步增加分辨率，学习预测下一个尺度。

它们是穿过同一问题的不同轴线。两者都能得到易处理的条件分布。在实践中，VAR 的推理速度更快（传递次数更少，尺度内全部并行），并且在类条件 ImageNet 上与 DiT 持平或更优。文本条件 VAR（VARclip、HART）是一个活跃的研究方向。

## 从零实现

在 `code/main.py` 中你将：
1. 在合成"图像"数据（二维高斯环）上构建一个微型**多尺度 VQ 分词器**。
2. 训练一个 **VAR 风格的 Transformer** 来对 token 做下一尺度预测。
3. 通过调用 Transformer 4 次（4 个尺度）并解码来采样。
4. 验证按尺度顺序的训练能让生成在尺度内并行进行。

这是一个玩具实现。重点在于亲眼看到尺度结构化的注意力掩码和尺度内并行生成确实在工作。

## 交付产物

本课产出 `outputs/skill-var-tokenizer-designer.md`——一个用于设计多尺度分词器的技能：尺度数量、尺度比例、码本大小、残差共享、解码器架构。

## 练习

1. **尺度数量消融。** 分别用 4、6、8、10 个尺度训练 VAR。测量重建质量与自回归传递次数的关系。尺度越多 = 残差越精细 = 质量越好，但传递次数也越多。

2. **码本大小。** 分别用 512、4096、16384 的码本大小训练分词器。更大的码本带来更好的重建，但预测更困难。找出拐点。

3. **尺度内并行性检查。** 对一个训练好的 VAR，显式测量其注意力模式。在尺度 k 内，模型是否关注跨尺度位置而不关注尺度内位置？验证掩码实现。

4. **VAR 与 DiT 的缩放对比。** 在同一个 ImageNet 类条件任务上，以相同参数预算（如 33M、130M、458M）训练 VAR 和 DiT。绘制 FID 与计算量的曲线。VAR 应在每个规模上领先 DiT——在小规模上复现论文的结果。

5. **文本条件化。** 扩展 VAR，通过 adaLN 接收文本嵌入（CLIP 池化向量）作为额外的条件输入。这是 HART 的做法。在文本对齐采样上 FID 能提升多少？

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|----------------|----------------------|
| VAR | "Visual AutoRegressive" | 通过在 VQ token 网格金字塔上做下一尺度预测来生成图像 |
| 下一尺度预测 | "先预测粗的，再预测细的" | 模型以递增的分辨率尺度预测 token，以所有先前尺度为条件 |
| 多尺度 VQ 分词器 | "残差 VQ" | 产出 K 个分辨率递增的 token 网格的 VQ-VAE，解码器对所有尺度求和 |
| 尺度 k | "金字塔第 k 层" | K 个分辨率层级之一，从 k=1 时的 1x1 到 k=K 时的 (H/p)x(W/p) |
| 尺度内并行 | "每个尺度一次前向" | 尺度 k 上的所有 token 在一次 Transformer 传递中预测，而非自回归地预测 |
| 跨尺度因果 | "按尺度排序的注意力" | 尺度 k 上的 token 可以关注尺度 1..k 的全部内容，但不能关注尺度 k+1..K |
| 残差 VQ | "加性 token 化" | 每个尺度的 token 编码更低尺度留下的残差；解码器对所有尺度嵌入求和 |
| VAR 缩放定律 | "图像版 GPT 缩放" | FID 随计算量遵循可预测的幂律，就像语言模型的困惑度一样 |
| HART | "混合 VAR + 文本" | 文本条件 VAR 变体，将 MaskGIT 式迭代解码与 VAR 的尺度结构相结合 |
| 尺度位置嵌入 | "(scale, row, col) 三元组" | 位置编码同时携带尺度索引和尺度内的空间坐标 |

## 延伸阅读

- [Tian et al., 2024 — "Visual Autoregressive Modeling: Scalable Image Generation via Next-Scale Prediction"](https://arxiv.org/abs/2404.02905) —— VAR 论文，权威参考文献
- [Peebles and Xie, 2022 — "Scalable Diffusion Models with Transformers"](https://arxiv.org/abs/2212.09748) —— DiT，扩散模型对比基线
- [Esser et al., 2021 — "Taming Transformers for High-Resolution Image Synthesis"](https://arxiv.org/abs/2012.09841) —— VQGAN，VAR 多尺度分词器所扩展的分词器家族
- [van den Oord et al., 2017 — "Neural Discrete Representation Learning"](https://arxiv.org/abs/1711.00937) —— VQ-VAE，离散图像 token 化的基础
- [Tang et al., 2024 — "HART: Efficient Visual Generation with Hybrid Autoregressive Transformer"](https://arxiv.org/abs/2410.10812) —— 文本条件 VAR
