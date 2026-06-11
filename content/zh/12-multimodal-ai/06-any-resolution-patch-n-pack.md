# 任意分辨率视觉：Patch-n'-Pack 与 NaFlex

> 真实世界的图像不是 224x224 的正方形。一张收据是 9:16，一张图表是 16:9，一张医学影像可能是 4096x4096，一张手机截图是 9:19.5。2024 年之前 VLM 的标准做法——把所有图像缩放成固定正方形——丢掉了让 OCR、文档理解和高分辨率场景解析得以成立的关键信号。NaViT（Google，2023）证明了可以用块对角掩码把可变分辨率的图像块打包进同一个 Transformer 批次。Qwen2-VL 的 M-RoPE（2024）则彻底抛弃了绝对位置表。LLaVA-NeXT 的 AnyRes 把高分辨率图像切分为一张基础图加若干子图。SigLIP 2 的 NaFlex 变体（2025）如今已成为开源 VLM 的默认编码器选择——一个 checkpoint 即可服务所有宽高比。本课将端到端实现 patch-n'-pack。

**Type:** Build
**Languages:** Python (stdlib, patch packer + block-diagonal mask)
**Prerequisites:** Phase 12 · 01 (ViT patches), Phase 12 · 05 (LLaVA)
**Time:** ~120 minutes

## 学习目标

- 把一批可变分辨率图像的图像块（patch）打包成一个序列，并构建块对角注意力掩码。
- 针对给定任务，在 AnyRes 切片（LLaVA-NeXT）、NaFlex（SigLIP 2）和 M-RoPE（Qwen2-VL）之间做出选择。
- 在不缩放图像的前提下，计算 OCR、图表和摄影任务的 token 预算。
- 说出正方形缩放的三种失败模式：文字被压扁、内容被裁剪、token 浪费在填充上。

## 问题背景

Transformer 期望输入是序列。一个批次就是一摞等长的序列。如果你的图像都是 224x224，每次都会得到 196 个 patch token，无需填充，万事大吉。224 训练、224 推理，从此不必再操心分辨率。

但现实世界并不配合。文档是竖版的（8.5x11 英寸，约 2:3）。图表截图是横版的（16:9）。收据又高又窄（1:3）。医学影像通常是 2048x2048 甚至更大。手机截图是 1170x2532（0.46:1）。

2024 年之前有三种选项，以及它们各自失败的原因：

1. 缩放到固定正方形（224x224 或 336x336）。压扁会扭曲文字和人脸，下采样会毁掉图表标签和 OCR 内容。这是 LLaVA-1.5 之前的标准做法。
2. 裁剪到固定宽高比。你会丢掉图像的大部分内容，而且选择裁剪位置本身就是一个独立的视觉难题。
3. 按最长边填充。解决了形变问题，但对竖版图像会把 50% 以上的 token 浪费在填充上，而这些填充 token 还要付出二次方的注意力开销。

2024-2025 年的答案是：让 Transformer 直接消化图像原生分辨率下的 patch，并想办法把一个尺寸各异的批次打包进同一个序列，且不浪费算力。

## 核心概念

### NaViT 与 patch-n'-pack

NaViT（Dehghani et al., 2023）是首篇证明这种做法在大规模下可行的论文。其思路非常机械化：

1. 对批次中的每张图像，按选定的 patch 大小（比如 14）计算其原生 patch 网格。
2. 把每张图像的 patch 展平成各自的可变长度序列。
3. 把所有图像的 patch 拼接成该批次的一条长序列。
4. 构建块对角注意力掩码，使图像 A 的 patch 只在图像 A 内部互相注意。
5. 为每个 patch 携带位置信息（2D RoPE 或分数位置嵌入）。

一个批次包含三张图像：336x336（576 个 token）、224x224（256 个 token）、448x336（768 个 token），打包后变成一条 1600 个 token 的序列，配上一个 1600x1600 的块对角掩码。没有填充。没有浪费的算力。Transformer 就能处理任意宽高比。

NaViT 还引入了训练期的分数 patch 丢弃——在整个批次中随机丢弃 50% 的 patch——既起到正则化作用，又加快了训练。SigLIP 2 继承了这一做法。

### AnyRes（LLaVA-NeXT）

LLaVA-NeXT 的 AnyRes 是务实派的替代方案。给定一张高分辨率图像和一个固定分辨率的编码器（336 分辨率的 CLIP 或 SigLIP），对图像做切片：

1. 从预定义集合中选择最贴合图像宽高比的网格布局——(1x1)、(1x2)、(2x1)、(1x3)、(3x1)、(2x2) 等。
2. 把完整图像切分进该网格；每个切片成为一个 336x336 的裁剪块。
3. 同时生成一张缩略图：整张图像缩放到 336x336，作为全局上下文 token。
4. 用冻结的 336 编码器编码每个切片，再把切片 token 与缩略图 token 拼接起来。

一张 672x672 的图像用 2x2 网格加缩略图：4 * 576 + 576 = 2880 个视觉 token。代价高昂但确实有效——LLM 同时看到了局部细节和全局上下文。

当你的编码器被冻结且只支持单一分辨率时，AnyRes 是首选路线。但它会让大图的 token 数量爆炸（一张 1344x1344 的图像用 4x4 网格是 9216 + 576 ≈ 9800 个 token，几乎填满一个 8k 上下文的 LLM）。

### M-RoPE（Qwen2-VL）

Qwen2-VL 引入了多模态旋转位置编码（Multimodal Rotary Position Embedding）。它既不用 NaViT 的分数位置，也不用 AnyRes 的切片加缩略图，而是让每个 patch 携带一个 3D 位置（时间、高度、宽度）。query/key 的旋转操作可以处理任意的 H、W 和时间长度。

M-RoPE 无需重新训练即可原生支持动态分辨率。推理时输入任意 HxW 的图像，patch 嵌入器产生 H/14 x W/14 个 token，每个 token 获得其 (t=0, r=行, c=列) 位置，RoPE 用相应频率旋转注意力，就完成了。Qwen2.5-VL 和 Qwen3-VL 延续了这一设计。InternVL3 的 V2PE 是同样的思路，只是对每种模态采用可变编码。

与 AnyRes 不同，M-RoPE 在原生分辨率下是 O(H x W / P^2) 个 token——没有切片带来的成倍开销。与 NaViT 不同，它每次前向仍只接受单张图像。要跨分辨率做批处理，还得在其上叠加 patch-n'-pack。

### NaFlex（SigLIP 2）

NaFlex 是 SigLIP 2 checkpoint 的原生弹性（native-flex）模式。单一模型在推理时可服务多种序列长度（256、729、1024 个 token）。其内部在训练时采用 NaViT 式的 patch-n'-pack，并为每个 patch 使用绝对分数位置。卖点在于：一个 checkpoint，推理时按任务自选 token 预算。

语义任务（分类、检索）用 256 个 token。OCR 或图表理解用 1024 个 token。无需重新训练。

### 打包掩码

块对角掩码是大多数实现栽跟头的地方。对于一条长度为 `N_total` 的打包序列，覆盖图像 `i=0..B-1`，各自长度为 `n_i`，形状为 `(N_total, N_total)` 的掩码 `M` 在两个索引落在同一图像块内时为 1，否则为 0。可以用累积长度列表来构建：

```
offsets = [0, n_0, n_0+n_1, ..., N_total]
M[i, j] = 1 iff there exists b where offsets[b] <= i < offsets[b+1] and offsets[b] <= j < offsets[b+1]
```

在 PyTorch 里用 `torch.block_diag` 或一次显式 gather 就是一行代码。FlashAttention 的变长路径（`cu_seqlens`）则完全跳过掩码，直接用累积长度张量在各序列内部做注意力——在典型批次上比稠密掩码快约 10 倍。

### Token 预算

按任务选择策略：

- OCR / 文档：1024-4096 个 token。SigLIP 2 NaFlex 用 1024，或 AnyRes 3x3 加缩略图。
- 图表与 UI：384-448 原生分辨率下 729-1024 个 token。Qwen2.5-VL 动态分辨率配合最大像素上限。
- 自然照片：256-576 个 token 就够了。下游 LLM 能看到足够的信息。把 token 花在内容密度高的地方。
- 视频：空间池化后每帧 64-128 个 token，2-8 FPS。第 12.17 课会讲。

2026 年的生产法则：为每类任务设定一个最大像素上限，在该上限内按原生宽高比编码，打包批次，跳过填充。Qwen2.5-VL 暴露的 `min_pixels` 和 `max_pixels` 正是为此而设的旋钮。

## 生产实践

`code/main.py` 用整数像素坐标实现了针对异构图像批次的 patch-n'-pack。它会：

- 接收一组 (H, W) 图像尺寸。
- 按 patch 大小 14 计算每张图像的 patch 序列长度。
- 把它们打包成一条总长度为 `sum(n_i)` 的序列。
- 构建块对角注意力掩码（为清晰起见用稠密形式）。
- 对比打包成本与正方形缩放、AnyRes 切片的成本。
- 为一个混合批次（收据、图表、截图、照片）打印一张 token 预算表。

运行它。跑出来的那些数字，就是 2026 年所有开源 VLM 都采用 patch-n'-pack 的原因。

## 交付产物

本课产出 `outputs/skill-resolution-budget-planner.md`。给定一个混合宽高比的工作负载（OCR、图表、照片、视频帧）和总 token 预算，它会选择合适的策略（NaFlex、AnyRes、M-RoPE 或固定正方形）并输出每个请求的配置。当你为产品评估 VLM 规模时使用这个技能——它能防止那种悄无声息的 10 倍 token 暴涨摧毁你的延迟预算。

## 练习

1. 一张收据是 600x1500（1:2.5）。在 patch 大小 14 下，原生分辨率有多少个 token？正方形缩放到 336 后有多少个？实践中哪种做法损失的 OCR 精度更多？

2. 为一个包含四张图像、长度分别为 256、576、729、1024 的批次构建块对角掩码。验证注意力矩阵为 2585x2585，且非零元素恰好为 `256^2 + 576^2 + 729^2 + 1024^2` 个。

3. 对一张 1792x896 的图像在 patch 大小 14 下比较：(a) 正方形缩放到 336 再编码，(b) AnyRes 2x1 加缩略图，(c) M-RoPE 原生分辨率。哪种用的 token 最少？哪种保留的细节最多？

4. 实现分数 patch 丢弃：给定一条打包序列，均匀随机丢弃 50% 的 token，并相应更新块对角掩码。测量掩码稀疏度的变化。

5. 阅读 Qwen2-VL 论文（arXiv:2409.12191）的 3.2 节。用两句话描述 `min_pixels` 和 `max_pixels` 各控制什么，以及为什么两个边界都很重要。

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|-----------------|------------------------|
| Patch-n'-pack | "NaViT 式打包" | 把来自不同图像的可变长度 patch 序列拼接进同一个批次维度 |
| 块对角掩码 | "打包掩码" | 一种注意力掩码，把每张图像的 patch 限制为只注意自身，不注意打包序列中的邻居 |
| AnyRes | "LLaVA-NeXT 切片" | 把高分辨率图像切分为固定尺寸切片网格外加一张全局缩略图；用固定编码器编码每个切片 |
| NaFlex | "SigLIP 2 native-flex" | 单一 SigLIP 2 checkpoint 在推理时服务 256/729/1024 token 预算，无需重新训练 |
| M-RoPE | "多模态 RoPE" | 3D 旋转位置编码（时间、行、列），无需位置表即可处理任意 H、W、T |
| cu_seqlens | "FlashAttention 打包" | FlashAttention 变长路径使用的累积长度张量，替代稠密块对角掩码 |
| min_pixels / max_pixels | "分辨率边界" | Qwen2.5-VL 的逐请求旋钮，用于限制极小或极大输入的 token 数量 |
| 视觉 token 预算 | "每张图多少 token" | 每张图像产出的 patch token 的大致数量；决定 LLM 的提示词预算和注意力开销 |

## 延伸阅读

- [Dehghani et al. — Patch n' Pack: NaViT (arXiv:2307.06304)](https://arxiv.org/abs/2307.06304)
- [Wang et al. — Qwen2-VL (arXiv:2409.12191)](https://arxiv.org/abs/2409.12191)
- [Laurençon et al. — What matters when building vision-language models? (Idefics2, arXiv:2405.02246)](https://arxiv.org/abs/2405.02246)
- [Tschannen et al. — SigLIP 2 (arXiv:2502.14786)](https://arxiv.org/abs/2502.14786)
- [Qwen Team — Qwen2.5-VL Technical Report (arXiv:2502.13923)](https://arxiv.org/abs/2502.13923)
