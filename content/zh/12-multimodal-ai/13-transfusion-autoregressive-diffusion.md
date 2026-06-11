# Transfusion：在一个 Transformer 里同时做自回归文本与扩散图像

> Chameleon 和 Emu3 把全部筹码押在离散 token 上。它们确实可行，但量化瓶颈显而易见——图像质量始终停留在连续空间扩散模型之下。Transfusion（Meta，Zhou 等人，2024 年 8 月）下了相反的赌注：让图像保持连续，彻底抛弃 VQ-VAE，用两个损失训练同一个 Transformer。文本 token 用下一 token 预测，图像 patch 用流匹配 / 扩散损失。两个目标优化同一套权重。Stable Diffusion 3 底层的架构（MMDiT）是它的近亲。本课会精读 Transfusion 的核心论点，构建一个玩具级双损失训练器，并梳理让一个 Transformer 身兼两职的注意力掩码。

**Type:** Build
**Languages:** Python (stdlib, two-loss trainer on MNIST-scale toy)
**Prerequisites:** Phase 12 · 11 (Chameleon), Phase 8 (Generative AI)
**Time:** ~180 minutes

## 学习目标

- 搭建一个在同一主干上跑两个损失（文本 token 上的 NTP、图像 patch 上的扩散 MSE）的 Transformer。
- 解释为什么"图像 patch 之间双向注意力 + 文本 token 上因果注意力"是正确的掩码选择。
- 在算力、质量和代码复杂度三个维度上，对比 Transfusion 式（连续图像 + 扩散损失）与 Chameleon 式（离散图像 + NTP）。
- 说出 MMDiT 的贡献：每个 block 内模态专属的权重，残差流处的联合注意力。

## 问题背景

离散与连续图像 token 之争比 LLM 还要古老。连续表示（原始像素、VAE 隐变量）保留细节；离散 token（VQ 索引）契合 Transformer 的原生词表，却在量化这一步丢失细节。

Chameleon / Emu3 选择了离散路线：一个损失、一套架构，但图像保真度被分词器质量封顶。

扩散模型选择了连续路线：图像质量出众，但模型与 LLM 相互独立，噪声调度工程复杂，且无法与文本生成干净地整合。

Transfusion 提出的问题是：能否两者兼得？让图像保持连续，仍然只训练一个模型，把两个损失缝合进同一次梯度更新。

## 核心概念

### 双损失架构

单个 decoder-only Transformer 处理的序列包含：

- 文本 token（离散，来自 BPE 词表）。
- 图像 patch（连续，16x16 像素块经线性嵌入投影到隐藏维度——与 ViT 编码器的输入相同）。
- `<image>` 和 `</image>` 标签，标记连续 patch 所在的位置。

前向传播只跑一次。损失按 token 类型在两个头中二选一：

- 文本 token：在词表 logits 头上做标准交叉熵。
- 图像 patch：对连续 patch 做扩散损失——预测加到每个 patch 上的噪声。

梯度流经共享的 Transformer 主体。两个损失同时改进这套共享权重。

### 注意力掩码：因果文本 + 双向图像

文本 token 必须是因果的——不能让一个文本 token 看到未来的文本，否则教师强制（teacher forcing）会失效。而图像 patch 表示的是同一个瞬间的快照，同一图像块内的 patch 应该相互双向注意。

掩码定义：

```
M[i, j] = 1 if:
  (i is text and j is text and j <= i)   # causal for text
  OR (i is image and j is image and same_image_block(i, j))   # bidirectional within image
  OR (i is text and j is image and j < i_image_end)   # text attends to previous images
  OR (i is image and j is text and j < i_image_start)   # image attends to preceding text
```

训练和推理时都实现为块三角掩码（block-triangular mask）。

### Transformer 内部的扩散损失

这个扩散损失是标准做法：给图像 patch 加噪，让模型预测噪声（或等价地预测干净 patch）。Transfusion 的版本采用流匹配（flow matching）——预测从带噪到干净的速度场。

训练时：
1. 对每个图像 patch x0，采样随机时间步 t。
2. 采样噪声 ε，计算 xt = (1-t) * x0 + t * ε（流匹配的线性插值）。
3. Transformer 预测 v_theta(xt, t)；损失 = MSE(v_theta(xt, t), ε - x0)。
4. 与同一序列里的文本 NTP 损失一起反向传播。

推理时的生成过程：
- 文本 token：标准自回归采样。
- 图像 patch：以前面的文本 token 为条件，跑扩散采样循环（通常 10-30 步）。

### MMDiT：Stable Diffusion 3 的变体

Stable Diffusion 3（Esser 等人，2024 年 3 月）发布的 MMDiT（Multimodal Diffusion Transformer）与 Transfusion 几乎同期问世。两套架构是亲兄弟。

MMDiT 的关键差异：

- 每个 block 内模态专属的权重。每个 Transformer block 对文本 token 和图像 patch 分别持有独立的 Q、K、V 和 MLP 权重。注意力是联合的（跨模态），其余部分均为模态专属。
- 修正流（rectified flow）训练。一种特定的流匹配变体，采样过程明确，数学比 DDPM 更简洁。
- 规模。MMDiT 是 SD3 的主干（有 2B 和 8B 参数两个变体）。Transfusion 的论文则扩展到了 7B。

二者殊途同归于同一个核心想法：用一个 Transformer 在文本上跑 NTP，在连续图像表示上跑扩散。

### 为什么优于 Chameleon 式方案

在图像生成上，连续扩散与离散 NTP 之间的质量差距是可量化的。Transfusion 论文报告：

- 在 7B 参数规模下，FID 比同尺寸的 Chameleon 式模型好 3-5 个点。
- 无需训练分词器——图像编码器更简单（线性投影到隐藏维度，与 ViT 的输入层一致）。
- 推理时图像 patch 去噪可以并行，而自回归图像 token 做不到。

缺点：Transfusion 是双损失模型，训练动态更棘手。损失权重需要调参；NTP 与扩散之间的调度失配可能导致某个头压过另一个。

### 下游演进

Janus-Pro（第 12.15 课）改进了 Transfusion 的思路：把理解和生成所用的视觉编码器解耦——一个用 SigLIP，另一个用 VQ——同时共享 Transformer 主体。Show-o（第 12.14 课）则把扩散换成离散扩散（掩码预测）。在 Transfusion 之后，统一生成这个家族迅速开枝散叶。

2026 年能生成图像的生产级 VLM——Gemini 3 Pro、GPT-5、Claude Opus 4.7 的图像生成路径——几乎可以肯定用的是这个家族的某种后代。细节是闭源的。

## 生产实践

`code/main.py` 在一个微型的类 MNIST 问题上构建玩具版 Transfusion：

- 文本描述是描述某个数字（0-9）的短整数序列。
- 图像是 4x4 的字节网格。
- 一对共享权重的线性投影充当 Transformer 的替身；文本上用 NTP 损失，带噪 patch 上用 MSE 损失。
- 训练循环交替两个损失，注意力掩码显式构造。
- 生成阶段在一次前向传播里同时产出文本描述和一张 4x4 图像。

这个 Transformer 是玩具，但双损失的管线、注意力掩码的构造和推理循环才是真正的产出物。

## 交付产物

本课产出 `outputs/skill-two-loss-trainer-designer.md`。给定一个新的多模态训练任务（文本 + 图像、文本 + 音频、文本 + 视频），它会设计双损失方案（损失权重、掩码形状、共享还是模态专属的 block），并标出实现风险。

## 练习

1. 一个 Transfusion 式模型的训练数据中 70% 是文本 token，30% 是图像 patch。图像扩散损失的量级约为文本 NTP 损失的 10 倍。用什么损失权重能让二者平衡？

2. 为序列 `[T, T, <image>, P, P, P, P, </image>, T]` 实现块三角掩码。把每个位置标为 0 或 1。

3. MMDiT 使用模态专属的 QKV 权重。相比 Transfusion 完全共享的 Transformer，这会带来多少参数量开销？在 7B 参数规模下值得吗？

4. 生成过程：给定一个文本提示，模型先用 NTP 生成 50 个 token，然后遇到 `<image>`，接着对 256 个 patch 做 20 步去噪扩散。总共需要多少次前向传播？

5. 阅读 SD3 论文第 3 节。描述修正流（rectified flow），并解释它为什么比 DDPM 在更少的推理步数内收敛。

## 关键术语

| 术语 | 大家怎么说 | 实际含义 |
|------|-----------------|------------------------|
| 双损失训练 | "NTP + 扩散" | 单个 Transformer 在同一次梯度更新中同时优化文本 token 上的交叉熵和连续图像 patch 上的 MSE |
| 流匹配 | "修正流" | 预测从噪声到干净数据的速度场的扩散变体；数学比 DDPM 更简洁 |
| MMDiT | "多模态 DiT" | Stable Diffusion 3 的架构：联合注意力，模态专属的 MLP 和归一化层 |
| 块三角掩码 | "因果文本 + 双向图像" | 在文本上因果、在图像区域内双向的注意力掩码 |
| 连续图像表示 | "不用 VQ" | 图像 patch 是实值向量，而不是整数码本索引 |
| 速度预测 | "v 参数化" | 网络输出的是噪声与数据之间的速度场，而不是噪声本身 |

## 延伸阅读

- [Zhou et al. — Transfusion (arXiv:2408.11039)](https://arxiv.org/abs/2408.11039)
- [Esser et al. — Stable Diffusion 3 / MMDiT (arXiv:2403.03206)](https://arxiv.org/abs/2403.03206)
- [Peebles & Xie — DiT (arXiv:2212.09748)](https://arxiv.org/abs/2212.09748)
- [Zhao et al. — MonoFormer (arXiv:2409.16280)](https://arxiv.org/abs/2409.16280)
- [Xie et al. — Show-o (arXiv:2408.12528)](https://arxiv.org/abs/2408.12528)
