# CLIP 与对比式视觉-语言预训练

> OpenAI 的 CLIP（2021）证明了一个足以驱动随后五年发展的想法：只用嘈杂的网络图像-描述文本对和一个对比损失，把图像编码器和文本编码器对齐到同一个向量空间。零监督标签。4 亿对数据。由此得到的嵌入空间可以做零样本分类、图文检索，并作为视觉塔（vision tower）接入 2026 年的每一个 VLM。SigLIP 2（2025）用 sigmoid 替换了 softmax，以更低的成本超越了 CLIP 的规模。本课从 InfoNCE 到 sigmoid 成对损失逐步推导数学原理，并用 Python 标准库实现训练步骤。

**Type:** Build
**Languages:** Python (stdlib, InfoNCE + sigmoid loss implementations)
**Prerequisites:** Phase 12 · 01 (ViT patches), Phase 7 (Transformers)
**Time:** ~180 minutes

## 学习目标

- 从互信息出发推导 InfoNCE 损失，并实现一个数值稳定的向量化版本。
- 解释为什么 sigmoid 成对损失（SigLIP）能扩展到 32768+ 的批大小，而无需 softmax 所要求的 all-gather 开销。
- 通过构造文本模板（`a photo of a {class}`）并对余弦相似度取 argmax，运行零样本 ImageNet 分类。
- 说出 CLIP / SigLIP 预训练给你的四个调节杠杆：批大小、温度、提示模板、数据质量。

## 问题背景

CLIP 之前的视觉是监督式的。收集带标签的数据集（ImageNet：120 万张图像，1000 个类别），训练一个 CNN，然后上线。标签很昂贵，标签会偏向标注者能达成一致的内容，而且不经过微调标签无法迁移到新任务。

图像-描述文本的网络数据有十亿以上的松散标注对，而且是免费的。一张金毛犬的图片配上 alt 文本 "my dog Max in the park"，本身就携带了监督信号——文本描述了图像。问题在于：你能把它变成有用的训练吗？

CLIP 的答案：把图像-描述对当作匹配任务。给定一批 N 张图像和 N 条描述，学习把每张图像与它自己的描述匹配起来，对抗 N-1 个干扰项。监督信号是"这两个东西属于一对；那 N-1 个不属于"。没有类别标签。没有人工标注。只有一个对比损失。

由此得到的嵌入空间能做的远超 CLIP 的训练目标。ImageNet 零样本之所以可行，是因为 "a photo of a cat" 的嵌入会靠近那些从未被显式标注为猫的猫的图片。这就是催生 2026 年每一个 VLM 的那场押注。

## 核心概念

### 双编码器

CLIP 有两座塔：

- 图像编码器 `f`：ViT 或 ResNet，每张图像输出一个 D 维向量。
- 文本编码器 `g`：小型 transformer，每条描述输出一个 D 维向量。

两座塔都把输出归一化到单位长度。相似度为 `cos(f(x), g(y)) = f(x)^T g(y)`，因为两者都是单位范数。

对一批 N 个（图像，描述）对，构造形状为 `(N, N)` 的相似度矩阵 `S`：

```
S[i, j] = cos(f(x_i), g(y_j)) / tau
```

其中 `tau` 是一个可学习的温度（CLIP 初始化为 0.07；在对数空间中学习）。

### InfoNCE 损失

CLIP 在行和列上使用对称的交叉熵：

```
loss_i2t = CE(S, labels=identity)     # each image's positive is its own caption
loss_t2i = CE(S^T, labels=identity)   # each caption's positive is its own image
loss = (loss_i2t + loss_t2i) / 2
```

这就是 InfoNCE。CE 中的 softmax 迫使每张图像与自己的描述匹配的程度超过批内的所有其他描述。"负样本"是批内的所有其他条目。更大的批 = 更多的负样本 = 更强的信号。CLIP 用 32k 的批大小训练；规模至关重要。

### 温度

`tau` 控制 softmax 的尖锐程度。低 tau → 分布尖锐，产生难负样本挖掘的效果。高 tau → 分布平缓，所有样本都有贡献。CLIP 学习的是 log(1/tau)，并加以裁剪防止坍缩。SigLIP 2 固定初始 tau，改用一个可学习的偏置。

### 为什么 sigmoid 更易扩展（SigLIP）

Softmax 需要整个相似度矩阵保持同步。在分布式训练中，你必须把每个嵌入 all-gather 到每个副本，然后再做 softmax。通信量随 world size 呈二次增长。

SigLIP 用逐元素的 sigmoid 替换 softmax：对每个对 `(i, j)`，损失是"这两个是匹配对吗？"的二分类。正类标签在对角线上，其余全是负类。损失为：

```
L = -1/N sum over (i, j) [ y_ij log sigmoid(S[i,j]) + (1-y_ij) log sigmoid(-S[i,j]) ]
```

当 `i == j` 时 `y_ij = 1`，否则为 0。每个对的损失彼此独立。不需要 all-gather。每块 GPU 计算自己的本地分块然后求和。SigLIP 2 能以低成本扩展到 32k-512k 的批大小，而 CLIP 需要按比例增加的通信量。

### 零样本分类

给定 N 个类别名称，为每个类别构造一个文本模板：

```
"a photo of a {class}"
```

用文本编码器嵌入每个模板。用图像编码器嵌入你的图像。对余弦相似度取 argmax = 预测类别。无需在目标类别上训练。

提示模板很重要。CLIP 的原始论文为每个类别使用 80 个模板（普通、艺术、照片、绘画等）并对嵌入取平均，带来了 ImageNet 上 +3 个百分点的提升。现代用法通常只挑选一两个模板。

### 线性探针与微调

零样本只是基线。线性探针（linear probe，在冻结的 CLIP 特征之上为目标类别训练一个线性层）在领域内任务上胜过零样本。完整微调在领域内胜过线性探针，但可能损害零样本迁移能力。三种方案，三种权衡。

### SigLIP 2：NaFlex 与稠密特征

SigLIP 2（2025）新增了：
- NaFlex：单个模型即可处理可变长宽比和分辨率。
- 更好的稠密特征，用于分割和深度估计，目标是在 VLM 中作为冻结骨干网络使用。
- 多语言：在 100 多种语言上训练，而 CLIP 只支持英语。
- 10 亿参数规模，而 CLIP 的上限是 4 亿。

在 2026 年的开源 VLM 中，SigLIP 2 SO400m/14 是默认的视觉塔。在纯图文检索场景中，如果 LAION-2B 的训练分布恰好匹配你的查询模式，CLIP 仍是默认选择。

### ALIGN、BASIC、OpenCLIP、EVA-CLIP

ALIGN（Google，2021）：与 CLIP 同样的思路，18 亿对的规模，90% 是噪声。证明了嘈杂数据可以规模化。OpenCLIP（LAION）：在 LAION-400M / 2B 上对 CLIP 的开源复现，提供多种规模，是首选的开源检查点。EVA-CLIP：从掩码图像建模初始化；是 VLM 的强力骨干网络。BASIC：Google 的 CLIP+ALIGN 混合方案。全是同一家族，只是数据和调参不同。

### 零样本的天花板

CLIP 类模型的 ImageNet 零样本准确率封顶在 76% 左右（CLIP-G、OpenCLIP-G）。要突破，要么需要大得多的数据（SigLIP 2 达到 80%+），要么需要架构改动（监督头、更多参数）。这个基准正在饱和；真正的价值在于供下游 VLM 消费的嵌入空间。

```figure
multimodal-fusion
```

## 生产实践

`code/main.py` 实现了：

1. 一个玩具双编码器（基于哈希的图像特征、文本字符特征），让你不依赖 numpy 就能看清 InfoNCE 的形态。
2. 纯 Python 的 InfoNCE 损失（通过 log-sum-exp 保证数值稳定）。
3. 用于对比的 sigmoid 成对损失。
4. 一个零样本分类流程：计算与一组文本提示的余弦相似度，取 argmax 作为预测。

运行它，观察损失曲线。绝对数值是玩具级的；但曲线形状与真实 CLIP 训练器输出的一致。

## 交付产物

本课产出 `outputs/skill-clip-zero-shot.md`。给定一组图像（通过路径）和一个目标类别列表，它用 CLIP 模板构造文本提示，用指定的检查点（例如 `openai/clip-vit-large-patch14`）嵌入两侧，并返回带相似度分数的 top-1 / top-5 预测。该技能拒绝对提示列表之外的类别做出任何判断。

## 练习

1. 手工为一批 4 个对实现 InfoNCE。构造 4x4 相似度矩阵，运行 softmax，取出对角线，计算交叉熵。用这次手工计算验证你的 Python 实现。

2. SigLIP 除温度外还使用一个偏置参数 `b`：`S'[i,j] = S[i,j]/tau + b`。当批内存在严重的类别不平衡（每行的负样本远多于正样本）时，`b` 起什么作用？阅读 SigLIP 第 3 节（arXiv:2303.15343）。

3. 构建一个猫狗零样本分类器。尝试两个提示模板：`a photo of a {class}` 和 `a picture of a {class}`。在 100 张测试图像上测量准确率。模板集成是否优于单一模板？

4. 计算在 512 块 GPU、批大小 32k 的训练中，softmax InfoNCE 与 sigmoid 成对损失的通信成本。哪个按 O(N) 扩展，哪个按 O(N^2)？引用 SigLIP 第 4 节。

5. 阅读 OpenCLIP 缩放定律论文（arXiv:2212.07143，Cherti et al.）。从图中复现他们关于数据规模化的结论：在固定模型规模下，ImageNet 零样本准确率与训练数据量之间的对数线性关系是什么？

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|----------------|------------------------|
| InfoNCE | "对比损失" | 对一批数据的相似度矩阵做交叉熵；每个条目的正样本是与它配对的条目，负样本是其余所有条目 |
| Sigmoid 损失 | "SigLIP 损失" | 逐对的二元交叉熵；没有 softmax，没有 all-gather，分布式训练中扩展成本低 |
| 温度 | "tau" | 在 softmax/sigmoid 之前缩放 logits 的标量；控制分布的尖锐程度 |
| 零样本 | "免微调分类" | 用文本提示构造类别嵌入，按余弦相似度分类；无需在目标类别上训练 |
| 提示模板 | "a photo of a ..." | 包裹类别名称的文本框架；对零样本准确率的影响在 1-5 个百分点 |
| 双编码器 | "双塔" | 一个图像编码器 + 一个文本编码器，输出位于共享的 D 维空间 |
| 难负样本 | "棘手的干扰项" | 与正样本足够相似的负样本，模型必须下功夫才能将它们分开 |
| 线性探针 | "冻结 + 一层" | 只在冻结特征之上训练一个线性分类器；衡量特征质量 |
| NaFlex | "原生灵活分辨率" | SigLIP 2 的能力：无需缩放即可接收任意长宽比和分辨率的图像 |
| 温度缩放 | "对数参数化的 tau" | CLIP 参数化 `log(1/tau)` 使梯度表现良好；通过裁剪防止 tau 坍缩到接近零 |

## 延伸阅读

- [Radford et al. — Learning Transferable Visual Models From Natural Language Supervision (arXiv:2103.00020)](https://arxiv.org/abs/2103.00020) — CLIP 论文。
- [Zhai et al. — Sigmoid Loss for Language Image Pre-Training (arXiv:2303.15343)](https://arxiv.org/abs/2303.15343) — SigLIP。
- [Tschannen et al. — SigLIP 2 (arXiv:2502.14786)](https://arxiv.org/abs/2502.14786) — 多语言 + NaFlex。
- [Jia et al. — ALIGN (arXiv:2102.05918)](https://arxiv.org/abs/2102.05918) — 用嘈杂网络数据实现规模化。
- [Cherti et al. — Reproducible scaling laws for contrastive language-image learning (arXiv:2212.07143)](https://arxiv.org/abs/2212.07143) — OpenCLIP 缩放定律。
