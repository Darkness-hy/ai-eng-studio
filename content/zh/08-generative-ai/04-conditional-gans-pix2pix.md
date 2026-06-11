# 条件 GAN 与 Pix2Pix

> 2014-2017 年间的第一个重大突破，是学会控制 GAN 生成什么。给它接上一个标签、一张图像或一句话。Pix2Pix 做的是图像版本，而且在窄域图像到图像任务上，它至今仍胜过所有通用的文本生成图像模型。

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 8 · 03 (GANs), Phase 4 · 06 (U-Net), Phase 3 · 07 (CNNs)
**Time:** ~75 minutes

## 问题背景

无条件 GAN 只能随机采样人脸。做演示有用，放到生产环境毫无价值。你真正想要的是：*把素描映射成照片*、*把地图映射成航拍图*、*把白天场景映射成夜晚*、*给灰度图上色*。这些任务都有一个共同点：给定输入图像 `x`，要求输出与之语义对应的 `y`。而每个 `x` 对应许多合理的 `y`。均方误差会把它们平均成一团糊。对抗损失则不会，因为「看起来真实」这个判据是锐利的。

条件 GAN（Conditional GAN，Mirza & Osindero, 2014）把条件 `c` 同时作为 `G` 和 `D` 的输入。Pix2Pix（Isola et al., 2017）将其特化：条件是一整张输入图像，生成器用 U-Net，判别器是*基于图块的*分类器（PatchGAN），损失是对抗损失 + L1。这套配方在窄域图像到图像任务上的表现，即便到了 2026 年也胜过从零训练的文本生成图像模型，因为它用的是*成对数据*——你恰好拥有所需的全部信号。

## 核心概念

![Pix2Pix: U-Net generator, PatchGAN discriminator](../assets/pix2pix.svg)

**条件生成器 G。** `G(x, z) → y`。在 Pix2Pix 中，`z` 是 G 内部的 dropout（没有输入噪声——Isola 发现显式噪声会被网络直接忽略）。

**条件判别器 D。** `D(x, y) → [0, 1]`。输入是（条件，输出）这个*对*。这是关键差异：D 必须判断 `y` 是否与 `x` 一致，而不只是 `y` 看起来真不真实。

**U-Net 生成器。** 带跨瓶颈跳跃连接的编码器-解码器。对于输入和输出共享低层结构（边缘、轮廓）的任务至关重要。没有跳跃连接，高频细节会消失殆尽。

**PatchGAN 判别器。** D 不输出单个真/假分数，而是输出一个 `N×N` 网格，每个单元判断约 70×70 像素的感受野，再取平均。这背后是马尔可夫随机场假设：真实感是局部的。训练快得多、参数更少、输出更锐利。

**损失函数。**

```
loss_G = -log D(x, G(x)) + λ · ||y - G(x)||_1
loss_D = -log D(x, y) - log (1 - D(x, G(x)))
```

L1 项稳定训练，并把 G 推向已知目标。L1 给出的边缘比 L2 更锐利（中位数而非均值）。Pix2Pix 的默认值是 `λ = 100`。

## CycleGAN——当你没有成对数据时

Pix2Pix 需要成对的 `(x, y)` 数据。CycleGAN（Zhu et al., 2017）放弃了这一要求，代价是多一项损失：*循环一致性*（cycle consistency）损失。两个生成器 `G: X → Y` 和 `F: Y → X`，训练目标是让 `F(G(x)) ≈ x` 且 `G(F(y)) ≈ y`。这让你可以在没有成对样本的情况下，把马变成斑马、把夏天变成冬天。

到 2026 年，无配对图像到图像的任务大多改用扩散模型（ControlNet、IP-Adapter）而非 CycleGAN，但循环一致性这一思想几乎在每篇无配对域适应的论文里都还活着。

## 从零实现

`code/main.py` 在一维数据上实现了一个极简条件 GAN。条件 `c` 是类别标签（0 或 1）。任务：针对给定类别，从其条件分布中产生样本。

### 第 1 步：把条件拼接到 G 和 D 的输入上

```python
def G(z, c, params):
    return mlp(concat([z, one_hot(c)]), params)

def D(x, c, params):
    return mlp(concat([x, one_hot(c)]), params)
```

独热编码（one-hot encoding）是最简单的方式。更大的模型会用可学习的嵌入、FiLM 调制或交叉注意力。

### 第 2 步：条件训练

```python
for step in range(steps):
    x, c = sample_real_conditional()
    noise = sample_noise()
    update_D(x_real=x, x_fake=G(noise, c), c=c)
    update_G(noise, c)
```

生成器必须匹配*给定条件下*的真实分布，而不是边缘分布。

### 第 3 步：验证逐类别输出

```python
for c in [0, 1]:
    samples = [G(noise, c) for noise in batch]
    mean_c = mean(samples)
    assert_near(mean_c, real_mean_for_class_c)
```

## 常见陷阱

- **条件被忽略。** G 学会了边缘化，而 D 由于条件信号太弱从不惩罚它。解法：更激进地给 D 注入条件（在早期层而不只是后期层），或使用投影判别器（projection discriminator，Miyato & Koyama 2018）。
- **L1 权重太低。** G 漂移到任意看似真实的输出，而非忠实于目标的输出。对 Pix2Pix 类任务从 λ≈100 起步。
- **L1 权重太高。** G 输出模糊，因为 L1 终究还是 L_p 范数。训练稳定后逐步退火降低。
- **D 中的真值泄漏。** D 的输入要拼接 `(x, y)`，而不能只给 `y`。否则 D 无法检查一致性。
- **逐类别模式崩塌。** 每个类别都可能独立崩塌。要做按类别的条件多样性检查。

## 生产实践

2026 年图像到图像任务的现状：

| 任务 | 最佳方案 |
|------|---------------|
| 素描 → 照片，同域、有成对数据 | Pix2Pix / Pix2PixHD（依然快、依然锐利） |
| 素描 → 照片，无配对 | ControlNet 搭配 Scribble 条件模型 |
| 语义分割图 → 照片 | SPADE / GauGAN2 或 SD + ControlNet-Seg |
| 风格迁移 | 扩散模型配 IP-Adapter 或 LoRA；GAN 方法已是遗产技术 |
| 深度图 → 照片 | 基于 Stable Diffusion 的 ControlNet-Depth |
| 超分辨率 | Real-ESRGAN（GAN）、ESRGAN-Plus 或 SD-Upscale（扩散） |
| 上色 | ColTran、基于扩散的上色器，或 Pix2Pix-color |
| 白天 → 夜晚、季节、天气 | CycleGAN 或基于 ControlNet 的方案 |

满足以下条件时，Pix2Pix 仍是正确的工具：(a) 你有数千个成对样本，(b) 任务窄且可重复，(c) 你需要快速推理。在通用开放域任务上，扩散模型胜出。

## 交付产物

保存 `outputs/skill-img2img-chooser.md`。该 skill 接收任务描述、数据可用性（成对还是无配对、N 个样本）以及延迟/质量预算，然后输出：方案（Pix2Pix、CycleGAN、某个 ControlNet 变体、SDXL + IP-Adapter）、训练数据需求、推理成本，以及评估方案（LPIPS、FID、任务特定指标）。

## 练习

1. **简单。** 修改 `code/main.py`，增加第三个类别。确认 G 仍能把每个类别的噪声映射到正确的模式。
2. **中等。** 在一维设定下用感知风格的损失替换 L1（例如用一个冻结的小 D 充当特征提取器）。它是否改变了条件分布的锐利程度？
3. **困难。** 在一维设定下勾画一个 CycleGAN：两个分布、两个生成器、循环损失。证明它能在没有成对数据的情况下学会两者之间的映射。

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|-----------------|-----------------------|
| 条件 GAN（Conditional GAN） | 「带标签的 GAN」 | G(z, c)、D(x, c)。两个网络都能看到条件。 |
| Pix2Pix | 「图像到图像的 GAN」 | 成对数据的 cGAN，U-Net 生成器 + PatchGAN 判别器 + L1 损失。 |
| U-Net | 「带跳跃连接的编码器-解码器」 | 对称卷积网络；跳跃连接保留高频信息。 |
| PatchGAN | 「局部真实感分类器」 | D 输出逐图块分数而非全局分数。 |
| CycleGAN | 「无配对图像翻译」 | 两个生成器 + 循环一致性损失；无需成对数据。 |
| SPADE | 「GauGAN」 | 用语义图归一化中间激活；分割图到图像。 |
| FiLM | 「逐特征线性调制」 | 由条件生成的逐特征仿射变换；廉价的条件注入方式。 |

## 生产笔记：把 Pix2Pix 当作延迟受限的基线

当你有成对数据且任务足够窄（素描 → 渲染图、语义图 → 照片、白天 → 夜晚）时，Pix2Pix 的单次前向推理在延迟上比扩散模型快一个数量级。生产中的典型对比如下：

| 路径 | 步数 | 单张 L4 上 512² 的典型延迟 |
|------|-------|----------------------------------------|
| Pix2Pix（U-Net 前向） | 1 | ~30 ms |
| SD-Inpaint 或 SD-Img2Img | 20 | ~1.2 s |
| SDXL-Turbo Img2Img | 1-4 | ~0.15-0.35 s |
| ControlNet + SDXL base | 20-30 | ~3-5 s |

在静态批处理场景下 Pix2Pix 吞吐量占优（每个请求的 FLOPs 完全相同）。扩散模型则在质量和泛化上胜出。现代的常见打法是：针对窄域任务上线一个 Pix2Pix 风格的蒸馏模型，再用扩散模型兜底处理长尾输入。

## 延伸阅读

- [Mirza & Osindero (2014). Conditional Generative Adversarial Nets](https://arxiv.org/abs/1411.1784) —— cGAN 论文。
- [Isola et al. (2017). Image-to-Image Translation with Conditional Adversarial Networks](https://arxiv.org/abs/1611.07004) —— Pix2Pix。
- [Zhu et al. (2017). Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks](https://arxiv.org/abs/1703.10593) —— CycleGAN。
- [Wang et al. (2018). High-Resolution Image Synthesis with Conditional GANs](https://arxiv.org/abs/1711.11585) —— Pix2PixHD。
- [Park et al. (2019). Semantic Image Synthesis with Spatially-Adaptive Normalization](https://arxiv.org/abs/1903.07291) —— SPADE / GauGAN。
- [Miyato & Koyama (2018). cGANs with Projection Discriminator](https://arxiv.org/abs/1802.05637) —— 投影判别器。
