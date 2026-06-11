# Vision Transformer 与图像块 token 原语

> 在进入任何多模态工作之前，图像必须先变成 Transformer 能消化的 token 序列。2020 年的 ViT 论文给出的答案是：16x16 像素的图像块、一次线性投影、再加一个位置嵌入。五年过去，2026 年的每一个前沿模型（2576px 原生分辨率的 Claude Opus 4.7、Gemini 3.1 Pro、Qwen3.5-Omni）依然从这里开始——编码器从 ViT 换成了 DINOv2 再到 SigLIP 2，加入了寄存器 token，位置编码方案变成了 2D-RoPE，但这个原语始终未变。本课将端到端读懂图像块 token 流水线，并用纯标准库 Python 把它实现出来，让第 12 阶段的后续内容对「视觉 token」有一个具体的心智模型。

**Type:** Learn
**Languages:** Python (stdlib, patch tokenizer + geometry calculator)
**Prerequisites:** Phase 7 (Transformers), Phase 4 (Computer Vision)
**Time:** ~120 minutes

## 学习目标

- 把一张 HxWx3 的图像转换成带有正确位置编码的图像块 token 序列。
- 对给定（patch 大小、分辨率、隐藏维度、深度）的 ViT，计算其序列长度、参数量和 FLOPs。
- 说出让 ViT 从 2020 年的研究成果走向 2026 年生产环境的三项升级：自监督预训练（DINO / MAE）、寄存器 token（register tokens）、原生分辨率打包（native-resolution packing）。
- 针对下游任务，在 CLS 池化、均值池化和寄存器 token 之间做出选择。

## 问题背景

Transformer 处理的是向量序列。文本天然就是序列（字节或 token）。而图像是带三个颜色通道的二维像素网格——不是序列。如果把每个像素都展平，一张 224x224 的 RGB 图像会变成 150,528 个 token，在这个长度上做自注意力根本行不通（自注意力的开销随序列长度平方增长）。

2020 年之前的做法是在前面接一个 CNN 特征提取器：用 ResNet 产出 7x7 的特征图（每个位置是 2048 维向量），再把这 49 个 token 喂给 Transformer。这能用，但继承了 CNN 的归纳偏置（平移等变性、局部感受野），也丢掉了 Transformer 对规模的胃口。

Dosovitskiy 等人（2020）直截了当地问：要是干脆跳过 CNN 呢？把图像切成固定大小的图像块（比如 16x16 像素），对每个图像块做线性投影得到一个向量，加上位置嵌入，再把序列喂给一个普通的 Transformer。在当时这是离经叛道——做视觉竟然不用卷积。但只要数据够多（先是 JFT-300M，后来是 LAION），它就在 ImageNet 上击败了 ResNet，并且持续提升。

到 2026 年，ViT 原语已经是无可争议的基石。每一个开放权重 VLM 的视觉塔（vision tower）都是它的某种后裔（DINOv2、SigLIP 2、CLIP、EVA、InternViT）。问题不再是「该不该用图像块」，而是「用多大的 patch、什么样的分辨率策略、什么预训练目标、什么位置编码」。

## 核心概念

### 图像块即 token

给定形状为 `(H, W, 3)` 的图像 `x` 和 patch 大小 `P`，把图像切成 `(H/P) x (W/P)` 的非重叠图像块网格。每个图像块是一个 `P x P x 3` 的像素立方体。把每个立方体展平成 `3 P^2` 维向量。再用一个形状为 `(3 P^2, D)` 的共享线性投影 `W_E`，把每个图像块映射到模型的隐藏维度 `D`。

对于 ViT-B/16 这一经典配置：
- 分辨率 224，patch 大小 16 → 网格 14x14 → 196 个图像块 token。
- 每个图像块是 `16 x 16 x 3 = 768` 个像素值，投影到 `D = 768`。
- 加一个可学习的 `[CLS]` token → 序列长度 197。

这个图像块投影在数学上完全等价于一个 kernel 大小为 `P`、步长为 `P`、输出通道数为 `D` 的二维卷积。生产代码实际上就是这么实现的——`nn.Conv2d(3, D, kernel_size=P, stride=P)`。「线性投影」的说法是概念层面的；卷积核的写法才是高效的实现。

### 位置嵌入

图像块本身没有固有顺序——在 Transformer 眼里它们就是一袋无序的 token。早期 ViT 加的是可学习的一维位置嵌入（每个位置一个 768 维向量，共 197 个）。能用，但把模型绑死在训练分辨率上：推理时一旦改变网格，就得对位置表做插值。

现代视觉骨干网络用的是 2D-RoPE（Qwen2-VL 的 M-RoPE、SigLIP 2 的默认方案）或分解式二维位置编码。2D-RoPE 根据图像块的（行，列）索引旋转查询和键向量，模型从旋转角度推断相对二维位置。不需要位置表。推理时模型可以处理任意大小的网格。

### CLS token、池化输出与寄存器 token

图像级表示是什么？目前有三种方案并存：

1. `[CLS]` token。在图像块序列前面加一个可学习向量。经过所有 Transformer 块之后，CLS token 的隐藏状态就是图像表示。继承自 BERT。原始 ViT、CLIP 用的是它。
2. 均值池化。对图像块 token 的输出隐藏状态取平均。SigLIP、DINOv2 以及多数现代 VLM 用的是它。
3. 寄存器 token。Darcet 等人（2023）观察到，训练时没有显式 sink token 的 ViT 会出现高范数的「伪影」图像块，劫持自注意力。加入 4–16 个可学习的寄存器 token 可以吸收这部分负载，并提升密集预测（分割、深度估计）的质量。DINOv2 和 SigLIP 2 都自带寄存器。

这个选择对下游任务很重要。做分类，CLS 就够了。对于把图像块 token 喂进 LLM 的 VLM，则完全跳过池化——每个图像块都成为一个 LLM 输入 token。寄存器在交接之前会被丢弃（它们是脚手架，不是内容）。

### 预训练：监督、对比、掩码、自蒸馏

2020 年的 ViT 是在 JFT-300M 上用监督分类预训练的。很快就被以下方法取代：

- CLIP（2021）：在 4 亿图文对上做对比学习。见第 12.02 课。
- MAE（2021，He 等人）：掩盖 75% 的图像块，重建像素。自监督，纯图像即可训练。
- DINO（2021）/ DINOv2（2023）：学生-教师自蒸馏，不需要标签，也不需要图像描述。2023 年的 DINOv2 ViT-g/14 是最强的纯视觉骨干网络，也是「密集特征」场景的默认选择。
- SigLIP / SigLIP 2（2023、2025）：换用 sigmoid 损失的 CLIP，外加支持原生宽高比的 NaFlex。是 2026 年开放 VLM（Qwen、Idefics2、LLaVA-OneVision）中占主导地位的视觉塔。

预训练方式决定了骨干网络擅长什么：CLIP/SigLIP 擅长与文本做语义匹配，DINOv2 擅长密集视觉特征，MAE 适合作为下游微调的起点。

### 扩展规律

ViT 扩展研究（Zhai 等人，2022）确立了一点：ViT 的质量在模型规模、数据规模和计算量上服从可预测的规律。在计算量固定时：
- 更大的模型 + 更多的数据 → 更好的质量。
- Patch 大小是序列长度与保真度之间的调节杆。Patch 14（DINOv2 / SigLIP SO400m 的典型值）比 patch 16 给出更多的单图 token 数；对 OCR 和密集任务更好，但速度更慢。
- 分辨率是另一个大杠杆。从 224 提到 384 再到 512 几乎总是有收益，代价是 FLOPs 平方增长。

ViT-g/14（10 亿参数，patch 14，分辨率 224 → 256 个 token）和 SigLIP SO400m/14（4 亿参数，patch 14）是 2026 年开放 VLM 的两大主力编码器。

### ViT 的参数量

完整计算在 `code/main.py` 里。对于 224 分辨率下的 ViT-B/16：

```
patch_embed = 3 * 16 * 16 * 768 + 768  =  591k
cls + pos    = 768 + 197 * 768          =  152k
block        = 4 * 768^2 (QKVO) + 2 * 4 * 768^2 (MLP) + 2 * 2*768 (LN)
             = 12 * 768^2 + 3k          =  7.1M
12 blocks    = 85M
final LN    = 1.5k
total       ≈ 86M
```

加载 checkpoint 之前，先用这种方式估算每个 ViT 的量级。骨干网络的大小决定了任何下游 VLM 的显存下限。

### 2026 年的生产配置

2026 年多数开放 VLM 搭载的编码器是原生分辨率（NaFlex）的 SigLIP 2 SO400m/14。它具备：
- 4 亿参数。
- Patch 大小 14，默认分辨率 384 → 每张图 729 个图像块 token。
- 图像级任务用均值池化；做 VQA 时全部 729 个图像块流入 LLM。
- 4 个寄存器 token，在交给 LLM 之前丢弃。
- 带图像级缩放的 2D-RoPE，支持原生宽高比。

这套配置里的每个决策都能追溯到一篇可供查阅的论文。

```figure
image-patch-tokens
```

## 生产实践

`code/main.py` 是一个图像块分词器兼几何计算器。它接收（图像 H、W、patch P、隐藏维度 D、深度 L），并输出：

- 切块后的网格形状和序列长度。
- 一张 8x8 像素合成玩具图像的 token 序列（完整走一遍展平 + 投影的路径）。
- 按图像块嵌入、位置嵌入、Transformer 块和输出头拆分的参数量。
- 目标分辨率下单次前向传播的 FLOPs。
- 一张横跨 ViT-B/16 @ 224、ViT-L/14 @ 336、DINOv2 ViT-g/14 @ 224、SigLIP SO400m/14 @ 384 的对比表。

跑一遍。把参数量和公开发布的数字对上。调一调 patch 大小和分辨率，亲身感受 token 数量的代价。

## 交付产物

本课产出 `outputs/skill-patch-geometry-reader.md`。给定一个 ViT 配置（patch 大小、分辨率、隐藏维度、深度），它会给出 token 数、参数量和显存估算，并附上推导依据。每次为 VLM 挑选视觉骨干网络时都用上这个技能——它能避免「token 爆炸、LLM 上下文被塞满」这类意外。

## 练习

1. 计算 Qwen2.5-VL 在原生 1280x720 输入、patch 大小 14 下的图像块 token 序列长度。它与只用 CLS 的表示相比如何？

2. 一帧 1080p 画面（1920x1080）在 patch 14 下产生多少 token？按 30 FPS 计，一段 5 分钟的视频总共有多少视觉 token？哪种手段最能省成本：池化、抽帧采样，还是 token 合并？

3. 用纯 Python 实现对图像块 token 的均值池化。验证对 DINOv2 输出的 196 个 token 做均值池化的结果，与向模型 `forward` 请求池化嵌入时返回的结果一致。

4. 阅读 "Vision Transformers Need Registers"（arXiv:2309.16588）第 3 节。用两句话描述寄存器吸收的是什么伪影，以及它为什么对下游密集预测很重要。

5. 修改 `code/main.py` 以支持 patch-n'-pack：给定一组不同分辨率的图像，生成单条打包序列和对应的块对角注意力掩码。到第 12.06 课时再对照验证。

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|----------------|------------------------|
| Patch（图像块） | 「16x16 像素方块」 | 输入图像中固定大小、互不重叠的区域；对应一个 token |
| Patch embedding（图像块嵌入） | 「线性投影」 | 一个共享的学习矩阵（或 stride=P 的 Conv2d），把展平的图像块像素映射为 D 维向量 |
| CLS token | 「类别 token」 | 前置的可学习向量，其最终隐藏状态代表整张图像；在 2026 年已是可选项 |
| Register token（寄存器 token） | 「sink token」 | 额外的可学习 token，用于吸收 ViT 在预训练中产生的高范数注意力伪影 |
| Position embedding（位置嵌入） | 「位置信息」 | 让序列具备顺序感知的逐位置向量或旋转操作；2D-RoPE 是现代默认方案 |
| Grid（网格） | 「图像块网格」 | 给定分辨率和 patch 大小下 (H/P) x (W/P) 的二维图像块阵列 |
| NaFlex | 「原生灵活分辨率」 | SigLIP 2 的特性：单个模型无需重训即可支持多种宽高比和分辨率 |
| Backbone（骨干网络） | 「视觉塔」 | 预训练的图像编码器，其图像块 token 输出在 VLM 中送入 LLM |
| Pooling（池化） | 「图像级摘要」 | 把图像块 token 变成单个向量的策略：CLS、均值、注意力池化或基于寄存器 |
| Patch 14 vs 16 | 「更细 vs 更粗的网格」 | Patch 14 每张图产生更多 token，对 OCR 保真度更好，但更慢；patch 16 是经典默认值 |

## 延伸阅读

- [Dosovitskiy et al. — An Image is Worth 16x16 Words (arXiv:2010.11929)](https://arxiv.org/abs/2010.11929) — ViT 原始论文。
- [He et al. — Masked Autoencoders Are Scalable Vision Learners (arXiv:2111.06377)](https://arxiv.org/abs/2111.06377) — MAE，自监督预训练。
- [Oquab et al. — DINOv2 (arXiv:2304.07193)](https://arxiv.org/abs/2304.07193) — 大规模自蒸馏，无需标签。
- [Darcet et al. — Vision Transformers Need Registers (arXiv:2309.16588)](https://arxiv.org/abs/2309.16588) — 寄存器 token 与伪影分析。
- [Tschannen et al. — SigLIP 2 (arXiv:2502.14786)](https://arxiv.org/abs/2502.14786) — 2026 年的默认视觉塔。
- [Zhai et al. — Scaling Vision Transformers (arXiv:2106.04560)](https://arxiv.org/abs/2106.04560) — 经验性扩展规律。
