# 视觉 Transformer（ViT）

> 一张图像是由 patch 组成的网格，一个句子是由 token 组成的网格。同一个 Transformer 两者通吃。

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 7 · 05 (Full Transformer), Phase 4 · 03 (CNNs), Phase 4 · 14 (Vision Transformers intro)
**Time:** ~45 minutes

## 问题背景

2020 年之前，计算机视觉就等于卷积。ImageNet、COCO 以及各类检测基准上的每一个 SOTA 都用 CNN 作为骨干网络。Transformer 是属于语言领域的东西。

Dosovitskiy 等人（2020）的论文 "An Image is Worth 16x16 Words" 证明了卷积可以完全丢掉：把图像切成固定大小的图像块（patch），用线性投影把每个 patch 变成一个嵌入向量，再把这个序列喂给一个普通的 Transformer 编码器。在足够大的规模下（ImageNet-21k 或更大规模的预训练），ViT 能够追平甚至超越基于 ResNet 的模型。

ViT 开启了一个在 2026 年已经普遍化的模式：一种架构，多种模态。Whisper 把音频做成 token，ViT 把图像做成 token，机器人领域有动作 token，视频领域有像素 token。Transformer 并不在乎输入是什么——只要喂给它一个序列，它就会学习。

到 2026 年，ViT 及其后继者（DeiT、Swin、DINOv2、ViT-22B、SAM 3）已占据视觉领域的大半江山。CNN 仍在边缘设备和延迟敏感任务上占优，除此之外的几乎所有场景，技术栈里都有 ViT 的身影。

## 核心概念

![Image → patches → tokens → transformer](../assets/vit.svg)

### 第 1 步——切分图像块（patchify）

把一张 `H × W × C` 的图像切分成 `N × (P·P·C)` 的展平 patch 序列。典型配置：`224 × 224` 图像、`16 × 16` patch → 196 个 patch，每个包含 768 个数值。

```
image (224, 224, 3) → 14 × 14 grid of 16x16x3 patches → 196 vectors of length 768
```

patch 大小是关键的调节杆。patch 越小 = token 越多、分辨率越高、注意力开销按平方增长。patch 越大 = 粒度越粗、计算越便宜。

### 第 2 步——线性嵌入

用一个共享的可学习矩阵把每个展平的 patch 投影到 `d_model` 维。这等价于一个卷积核大小为 `P`、步长为 `P` 的卷积。在 PyTorch 里它就是 `nn.Conv2d(C, d_model, kernel_size=P, stride=P)`——两行代码即可实现。

### 第 3 步——前置 `[CLS]` token，加上位置嵌入

- 在序列前面加一个可学习的 `[CLS]` token，它的最终隐藏状态就是用于分类的图像表示。
- 加上可学习的位置嵌入（ViT 原版）或二维正弦位置编码（后续变体）。
- 2024 年以后，RoPE 被扩展到二维用于编码位置，有时不再需要显式的位置嵌入。

### 第 4 步——标准 Transformer 编码器

堆叠 L 个 `LayerNorm → Self-Attention → + → LayerNorm → MLP → +` 模块，与 BERT 完全相同，没有任何视觉专用层。这正是这篇论文在教学意义上的点睛之笔。

### 第 5 步——输出头

分类任务：取 `[CLS]` 的隐藏状态 → 线性层 → softmax。对于 DINOv2 或 SAM，则丢弃 `[CLS]`，直接使用各个 patch 的嵌入。

### 重要的变体

| 模型 | 年份 | 改动 |
|-------|------|--------|
| ViT | 2020 | 原版。固定 patch 大小，完整的全局注意力。 |
| DeiT | 2021 | 蒸馏；仅用 ImageNet-1k 即可训练。 |
| Swin | 2021 | 层级结构加移位窗口。计算开销固定为亚平方级。 |
| DINOv2 | 2023 | 自监督（无需标签）。最好的通用视觉特征。 |
| ViT-22B | 2023 | 220 亿参数；缩放定律同样适用。 |
| SigLIP | 2023 | ViT + 语言配对，sigmoid 对比损失。 |
| SAM 3 | 2025 | 分割一切；ViT-Large + 可提示的掩码解码器。 |

### 为什么花了这么久

ViT 需要*大量*数据才能追平 CNN，因为它完全没有 CNN 的归纳偏置（inductive bias）——平移不变性、局部性。在缺少超过 1 亿张标注图像或强自监督预训练的情况下，相同算力下 CNN 仍然占优。DeiT 在 2021 年用蒸馏技巧解决了这个问题；DINOv2 在 2023 年用自监督彻底解决了它。

## 从零实现

见 `code/main.py`。纯标准库实现 patchify + 线性嵌入 + 完整性检查。不做训练——任何现实规模的 ViT 都需要 PyTorch 和数小时的 GPU 时间。

### 第 1 步：构造假图像

一张 24 × 24 的 RGB 图像，表示为由 `(R, G, B)` 元组组成的行列表。我们使用 6×6 的 patch → 16 个 patch，每个对应一个 108 维的嵌入向量。

### 第 2 步：切分图像块

```python
def patchify(image, P):
    H = len(image)
    W = len(image[0])
    patches = []
    for i in range(0, H, P):
        for j in range(0, W, P):
            patch = []
            for di in range(P):
                for dj in range(P):
                    patch.extend(image[i + di][j + dj])
            patches.append(patch)
    return patches
```

光栅顺序：在网格上按行优先排列。所有 ViT 都使用这种顺序。

### 第 3 步：线性嵌入

把每个展平的 patch 乘以一个随机的 `(patch_flat_size, d_model)` 矩阵。验证在前置 `[CLS]` 之后输出形状为 `(N_patches + 1, d_model)`。

### 第 4 步：计算真实规模 ViT 的参数量

打印 ViT-Base 的参数量：12 层、12 个注意力头、d=768、patch=16。和 ResNet-50（约 2500 万参数）对比。ViT-Base 大约是 8600 万参数，ViT-Large 约 3.07 亿，ViT-Huge 约 6.32 亿。

## 生产实践

```python
from transformers import ViTImageProcessor, ViTModel
import torch
from PIL import Image

processor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224-in21k")
model = ViTModel.from_pretrained("google/vit-base-patch16-224-in21k")

img = Image.open("cat.jpg")
inputs = processor(img, return_tensors="pt")
out = model(**inputs).last_hidden_state   # (1, 197, 768): [CLS] + 196 patches
cls_emb = out[:, 0]                       # image representation
```

**DINOv2 嵌入是 2026 年图像特征的默认选择。** 冻结骨干网络，只训练一个很小的输出头。分类、检索、检测、图像描述都适用。Meta 的 DINOv2 checkpoint 在所有非文本视觉任务上都优于 CLIP。

**patch 大小的选择。** 小模型用 16×16（ViT-B/16）。密集预测任务（分割）用 8×8 或 14×14（SAM、DINOv2）。超大模型用 14×14。

## 交付产物

见 `outputs/skill-vit-configurator.md`。该 skill 根据数据集规模、分辨率和算力预算，为新的视觉任务选择 ViT 变体和 patch 大小。

## 练习

1. **简单。** 运行 `code/main.py`。验证 patch 数量等于 `(H/P) * (W/P)`，且展平后的 patch 维度等于 `P*P*C`。
2. **中等。** 实现二维正弦位置嵌入——为每个 patch 的 `row` 和 `col` 分别生成独立的正弦编码并拼接。把它们喂给一个小型 PyTorch ViT，在 CIFAR-10 上对比它与可学习位置嵌入的准确率。
3. **困难。** 用 PyTorch 构建一个 3 层 ViT，用 4×4 的 patch 在 1,000 张 MNIST 图像上训练，测量测试准确率。然后在同样的 1,000 张图像上加入 DINOv2 式预训练（简化版：只训练编码器从被掩码的 patch 预测 patch 嵌入）。准确率有提升吗？

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|-----------------|-----------------------|
| Patch | "视觉 Transformer 的 token" | 图像中一个 `P × P × C` 区域的像素值展平后得到的向量。 |
| Patchify | "切块 + 展平" | 把图像切分成互不重叠的 patch，并将每个 patch 展平成向量。 |
| `[CLS]` token | "图像摘要" | 前置的可学习 token；其最终嵌入就是图像表示。 |
| 归纳偏置（Inductive bias） | "模型的先验假设" | ViT 的先验比 CNN 少；需要更多数据来弥补差距。 |
| DINOv2 | "自监督 ViT" | 不用标签训练，依靠图像增强 + 动量教师。2026 年最好的通用图像特征。 |
| SigLIP | "CLIP 的继任者" | ViT + 文本编码器，用 sigmoid 对比损失训练；相同算力下优于 CLIP。 |
| Swin | "窗口化 ViT" | 层级式 ViT，局部注意力 + 移位窗口；亚平方复杂度。 |
| Register token | "2023 年的技巧" | 额外加入的几个可学习 token，用来吸收注意力汇点（attention sink）；能改善 DINOv2 的特征。 |

## 延伸阅读

- [Dosovitskiy et al. (2020). An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale](https://arxiv.org/abs/2010.11929) —— ViT 论文。
- [Touvron et al. (2021). Training data-efficient image transformers & distillation through attention](https://arxiv.org/abs/2012.12877) —— DeiT。
- [Liu et al. (2021). Swin Transformer: Hierarchical Vision Transformer using Shifted Windows](https://arxiv.org/abs/2103.14030) —— Swin。
- [Oquab et al. (2023). DINOv2: Learning Robust Visual Features without Supervision](https://arxiv.org/abs/2304.07193) —— DINOv2。
- [Darcet et al. (2023). Vision Transformers Need Registers](https://arxiv.org/abs/2309.16588) —— 修复 DINOv2 的 register token 方法。
