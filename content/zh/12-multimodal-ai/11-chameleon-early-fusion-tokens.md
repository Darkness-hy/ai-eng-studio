# Chameleon 与早期融合的纯 token 多模态模型

> 到目前为止我们见过的所有 VLM 都把图像和文本分开处理。视觉 token 来自视觉编码器，经过投影器，再在 LLM 内部与文本汇合。视觉词表和文本词表从不重叠。Chameleon（Meta，2024 年 5 月）提出了一个问题：如果让它们重叠会怎样？训练一个 VQ-VAE，把图像转换成来自共享词表的离散 token 序列。这样一来，每个多模态文档都变成了一条序列——文本 token 和图像 token 交错排列，用同一个自回归损失训练。附带效果：模型可以生成混合模态输出——在一次推理调用中交替产出文本和图像 token。本课解读早期融合（early fusion）这一主张，并端到端地构建一个玩具版实现。

**Type:** Build
**Languages:** Python (stdlib, VQ-VAE tokenizer + interleaved decoder)
**Prerequisites:** Phase 12 · 05, Phase 8 (Generative AI)
**Time:** ~180 minutes

## 学习目标

- 解释为什么共享词表加单一损失会改变模型的能力边界。
- 描述 VQ-VAE 如何把图像分词成与 Transformer 下一个 token 目标兼容的离散序列。
- 说出 Chameleon 的训练稳定性技巧：QK-Norm、dropout 放置位置、LayerNorm 排列顺序。
- 对比 Chameleon 与 BLIP-2 的 Q-Former 方案，并说明各自适用的场景。

## 问题背景

基于适配器的 VLM（LLaVA、BLIP-2、Qwen-VL）把文本和图像当成两种不同的东西。文本 token 走 `embed(text_token)`；图像走 `visual_encoder(image) → projector → ... pseudo_tokens`。模型有两条输入通路，在中途才合并。

由此带来三个后果：

1. LLM 只能消费图像，不能产出图像。输出只有文本。
2. 混合模态文档（像文章那样段落和图片交替出现）处理起来很别扭——你要么在模型外解析多模态输入，要么把多次生成串起来。
3. 分布失配。视觉 token 和文本 token 位于隐空间的不同区域，会带来微妙的对齐问题。

Chameleon 直接否定了这个前提：图像不过是来自共享词表的离散 token 序列。在交错文档上训练模型，一个损失、一个自回归解码器，混合模态生成能力就免费到手了。

## 核心概念

### 把 VQ-VAE 用作图像分词器

这个分词器是一个向量量化变分自编码器（vector-quantized variational autoencoder）。架构如下：

- 编码器：CNN + ViT，把图像映射成空间特征图，比如 32x32 个维度为 256 的特征。
- 码本（codebook）：一个学习得到的、由 K 个向量组成的词表（Chameleon 用 8192 个），维度同样是 256。
- 量化：对每个空间特征，按 L2 距离查找最近的码本条目。用整数索引替换连续特征。
- 解码器：CNN，把量化后的特征还原回像素。

训练：VAE 重建损失 + 承诺损失（commitment loss）+ 码本损失。码本索引构成了图像的离散字母表。

对 Chameleon 而言：一张图像变成 32*32 = 1024 个 token，取自大小为 8192 的词表。再与文本 token（来自 LLM 的 BPE 词表，假设 32000）拼接。最终词表大小：40192。Transformer 看到的是一条序列、一个损失。

### 共享词表

Chameleon 的词表由文本 token、图像 token 和模态分隔符组成。每个 token 有唯一的 ID。输入嵌入层把每个 ID 映射到 D 维隐向量。输出投影把隐向量映射回词表 logits。softmax 选出下一个 token——无论它属于哪种模态。

分隔符很重要：`<image>` 和 `</image>` 标签把图像 token 序列括起来。生成时，一旦模型产出 `<image>`，下游软件就知道接下来的 1024 个 token 是 VQ 索引，要送进解码器渲染成像素。

### 混合模态生成

推理就是在共享词表上做下一个 token 预测。示例提示词："Draw a cat and describe it."Chameleon 输出：

```
<image> 4821 1029 2891 ... (1024 image tokens) </image>
The cat is orange, sitting on a windowsill...
```

模型自主决定顺序——可能先图后文、先文后图，或者交错输出。同一个解码器，同一个损失。

对比之下，适配器型 VLM 的生成只有文本。Chameleon 重新打开了"模型输出可以是什么模态"这个问题。

### 训练稳定性——QK-Norm、dropout、LayerNorm 排列

早期融合训练在大规模下不稳定。Chameleon 论文记录了三个技巧：

- QK-Norm。在注意力内部对 query 和 key 投影应用 LayerNorm，放在点积之前。防止深层网络中 logit 量级爆炸。多个 2024 年之后的大模型都用了它。
- dropout 放置位置。在每个残差相加之后都放 dropout，而不是只放在注意力和 MLP 之后。当图像 token 的梯度可能占据主导时，需要更强的正则化。
- LayerNorm 排列顺序。残差分支上用 Pre-LN（标准做法），并在最后一个 block 的跳跃连接上额外加一层 LN。稳定最后一层的梯度流。

没有这些技巧，34B 参数的 Chameleon 训练在多个 checkpoint 处发散。有了它们，训练收敛。这套训练配方与架构本身同等重要，都是论文的贡献。

### 分词器的重建上限

VQ-VAE 是有损的。在 8192 个码本条目、每张 512x512 图像 1024 个 token 的设定下，重建 PSNR 上限大约在 26-28 dB。这足以生成可辨认的图像，但明显逊色于连续空间的扩散模型（Stable Diffusion 3 能达到 32+ dB）。

分词器就是瓶颈。更好的分词器（MAGVIT-v2、IBQ、SBER-MoVQGAN）能抬高这个上限。Emu3（第 12.12 课）仅靠更好的分词器就达到了 SDXL 级别的生成质量。

### Chameleon vs BLIP-2 / LLaVA

Chameleon（早期融合，共享词表）：

- 一个损失，一个解码器。
- 能生成混合模态输出。
- 分词器决定质量上限。
- 开销大：推理路径上每生成一张图都要跑一次 VQ-VAE 解码器。

BLIP-2 / LLaVA（晚期融合，双塔分离）：

- 图像进、纯文本出。
- 复用预训练 LLM。
- 理解任务上没有分词器瓶颈。
- 开销小：单次前向传播。

按任务来选。需要图像生成，选 Chameleon 系；只需要理解能力，适配器型 VLM 更简单，也能复用更多预训练算力。

### Fuyu 与 AnyGPT

Fuyu（Adept，2023）是一个相关的方案：完全跳过独立的视觉编码器，把原始图像 patch 当作 token 直接喂进 LLM 的输入投影层，不用分词器。比 Chameleon 更简单，但失去了基于共享词表的输出生成能力。

AnyGPT（Zhan et al., 2024）把 Chameleon 扩展到四种模态：文本、图像、语音、音乐。对每种模态用同样的 VQ-VAE 技巧，共享一个 Transformer。实现任意模态到任意模态的生成。第 12.16 课会更详细地讲。

## 生产实践

`code/main.py` 构建了一个端到端的玩具版早期融合模型：

- 一个微型 VQ-VAE 风格量化器，把 8x8 patch 映射到码本索引（K=16）。
- 一个共享词表：（文本 id 0..31）+（图像 id 32..47）+（分隔符 48、49）。
- 一个玩具自回归解码器（bigram 表），在合成的字幕 + 图像 token 序列上训练。
- 一个采样循环，给定提示词后交替输出文本和图像 token。

代码刻意把 Transformer 缩到极小（bigram），方便你端到端地追踪信号流。

## 交付产物

本课产出 `outputs/skill-tokenizer-vs-adapter-picker.md`。给定一份产品需求（只做理解 vs 理解 + 生成、所需图像质量、成本预算），它会在 Chameleon 系（早期融合）和 LLaVA 系（晚期融合）之间做选择，并用可量化的经验法则给出理由。

## 练习

1. Chameleon 用 K=8192 个码本条目，每张 512x512 图像 1024 个 token。估算它相对 24 位 RGB 图像的压缩比。它是有损的吗？损失有多大？

2. 一张 4K 图像（3840x2160）在同样的 VQ-VAE 密度下会产生多少图像 token？Chameleon 式模型能在一次推理调用中生成 4K 图像吗？最先崩掉的是什么——上下文、分词器质量，还是 KV 缓存？

3. 用纯 Python 实现 QK-Norm。给定一个 64 维的 query 和 key，展示 LayerNorm 前后的点积。为什么在深层网络中控制量级很重要？

4. 阅读 Chameleon 论文 2.3 节关于训练稳定性的内容。描述论文在 34B 规模下不用 QK-Norm 时观察到的确切失败模式。"范数爆炸"的特征信号是什么？

5. 扩展玩具解码器，使其在纯文本提示词下输出混合模态回复。在训练数据分布为 60% 先文本 / 40% 先图像的设定下，测量模型选择先图像与先文本的频率。

## 关键术语

| 术语 | 大家的叫法 | 实际含义 |
|------|-----------------|------------------------|
| 早期融合 | "统一 token" | 图像从第一步起就被转换成与 Transformer 共享词表的离散 token |
| VQ-VAE | "图像分词器" | CNN + ViT + 码本，把图像映射成 Transformer 可预测的整数索引 |
| 共享词表 | "一本词典" | 覆盖文本 + 图像 + 模态分隔符的单一 token ID 空间 |
| QK-Norm | "注意力稳定器" | 在 query 和 key 点积之前对二者应用 LayerNorm，防止范数爆炸 |
| 混合模态生成 | "文本 + 图像输出" | 在一次推理中自主产出交错的文本和图像 token |
| 码本大小 | "K 个条目" | VQ-VAE 可量化到的离散向量数量；在压缩率和保真度之间权衡 |
| 分词器上限 | "重建极限" | 解码 VQ token 所能达到的最佳 PSNR；约束了模型的图像质量 |

## 延伸阅读

- [Chameleon Team — Chameleon: Mixed-Modal Early-Fusion Foundation Models (arXiv:2405.09818)](https://arxiv.org/abs/2405.09818)
- [Aghajanyan et al. — CM3 (arXiv:2201.07520)](https://arxiv.org/abs/2201.07520)
- [Yu et al. — CM3Leon (arXiv:2309.02591)](https://arxiv.org/abs/2309.02591)
- [Zhan et al. — AnyGPT (arXiv:2402.12226)](https://arxiv.org/abs/2402.12226)
- [Adept — Fuyu-8B blog (adept.ai)](https://www.adept.ai/blog/fuyu-8b)
