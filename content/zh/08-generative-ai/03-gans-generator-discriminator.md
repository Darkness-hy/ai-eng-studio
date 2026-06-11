# GAN——生成器与判别器的对抗

> Goodfellow 在 2014 年的妙招是彻底绕开密度估计。两个网络：一个造假，一个打假。它们互相博弈，直到假样本与真样本无法区分。这听起来不该奏效，而且也经常失败。但一旦成功，在窄域任务上它的样本至今仍是文献中最锐利的。

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 3 · 02 (Backprop), Phase 3 · 08 (Optimizers), Phase 8 · 02 (VAE)
**Time:** ~75 minutes

## 问题背景

VAE 生成的样本之所以模糊，是因为其 MSE 解码器损失在贝叶斯意义上的最优解是*平均*图像——而许多合理数字图像的平均是一个糊掉的数字。你需要的损失应当奖励*合理性（plausibility）*，而不是与某个特定目标在像素上的接近程度。合理性没有解析表达式，你必须把它学出来。

Goodfellow 的思路：训练一个分类器 `D(x)` 区分真实图像与伪造图像，再训练一个生成器 `G(z)` 去骗过 `D`。`G` 的损失信号就是 `D` 当下认为"像真的"的标准。这个信号会随着 `G` 的进步而更新，追逐一个移动的目标。如果两个网络都收敛，`G` 就在从未显式写出 `log p(x)` 的情况下学会了数据分布。

这就是对抗训练（adversarial training）。其数学形式是一个极小极大博弈：

```
min_G max_D  E_real[log D(x)] + E_fake[log(1 - D(G(z)))]
```

到了 2026 年，GAN 已不再是 SOTA 的生成器（扩散模型和流匹配夺走了这顶王冠）。但 StyleGAN 2/3 仍是迄今发布过的最锐利的人脸模型，GAN 判别器被用作扩散训练中的*感知损失（perceptual loss）*，而对抗训练驱动着快速一步蒸馏（SDXL-Turbo、SD3-Turbo、LCM），让你能够部署实时扩散模型。

## 核心概念

![GAN training: generator and discriminator in minimax](../assets/gan.svg)

**生成器 `G(z)`。** 把噪声向量 `z ~ N(0, I)` 映射成样本 `x̂`。形态上是一个解码器式网络（全连接或转置卷积）。

**判别器 `D(x)`。** 把样本映射为一个标量概率（或分数）。真实 → 1，伪造 → 0。

**损失。** 两步交替更新：

- **训练 `D`：** `loss_D = -[ log D(x) + log(1 - D(G(z))) ]`。即在真实=1、伪造=0 上做二元交叉熵。
- **训练 `G`：** `loss_G = -log D(G(z))`。这是 Goodfellow 实际使用的*非饱和（non-saturating）*形式（原始的 `log(1 - D(G(z)))` 在 `D` 信心十足时会饱和，从而扼杀梯度）。

**训练循环。** `D` 走一步，`G` 走一步。如此往复。

**为什么有效。** 如果 `G` 完美拟合了 `p_data`，那么 `D` 的表现不会好于随机猜测，处处输出 0.5；`G` 也就再无梯度可拿。达到均衡。

**为什么会崩。** 模式坍缩（`G` 找到一个 `D` 分不出来的模式后就反复生产它）、梯度消失（`D` 学得太快导致 `log D` 饱和）、训练不稳定（学习率、批大小，什么都可能出问题）。

## 让 GAN 真正可用的变体

| 年份 | 创新 | 解决了什么 |
|------|------------|-----|
| 2015 | DCGAN | 卷积/反卷积、批归一化、LeakyReLU——第一个稳定的架构。 |
| 2017 | WGAN, WGAN-GP | 用 Wasserstein 距离 + 梯度惩罚替代 BCE。解决梯度消失。 |
| 2017 | 谱归一化（Spectral normalization） | 给判别器施加 Lipschitz 约束。2026 年的判别器仍在使用。 |
| 2018 | Progressive GAN | 先训低分辨率，再逐层添加。首次产出百万像素级结果。 |
| 2019 | StyleGAN / StyleGAN2 | 映射网络 + 自适应实例归一化。固定域照片级真实感的最先进水平。 |
| 2021 | StyleGAN3 | 无混叠、平移等变——2026 年依然是人脸生成的金标准。 |
| 2022 | StyleGAN-XL | 条件生成、类别感知、更大规模。 |
| 2024 | R3GAN | 以更强的正则化重塑旗鼓；不靠技巧即可在 1024² 上工作。 |

```figure
gan-minimax
```

## 从零实现

`code/main.py` 在一维数据上训练一个微型 GAN：数据是两个高斯分布的混合。生成器和判别器都是单隐藏层 MLP。我们手写前向、反向和极小极大循环。目标是亲眼看到两种关键失败模式（模式坍缩 + 梯度消失）发生的过程。

### 第 1 步：非饱和损失

原始的 Goodfellow 损失 `log(1 - D(G(z)))` 在 D 高置信度地把 G 的假样本判为假时趋向于 0。此时 G 的梯度基本为零——G 无法再改进。非饱和形式 `-log D(G(z))` 的渐近行为正好相反：D 越自信，它的值就越大，给 G 提供强烈的信号。

```python
def g_loss(d_fake):
    # maximize log D(G(z))  <=>  minimize -log D(G(z))
    return -sum(math.log(max(p, 1e-8)) for p in d_fake) / len(d_fake)
```

### 第 2 步：每个生成器步对应一个判别器步

```python
for step in range(steps):
    # train D
    real_batch = sample_real(batch_size)
    fake_batch = [G(z) for z in sample_noise(batch_size)]
    update_D(real_batch, fake_batch)

    # train G
    fake_batch = [G(z) for z in sample_noise(batch_size)]  # fresh fakes
    update_G(fake_batch)
```

给 G 用新生成的假样本，否则梯度就是过期的。

### 第 3 步：监控模式坍缩

```python
if step % 200 == 0:
    samples = [G(z) for z in sample_noise(500)]
    mode_a = sum(1 for s in samples if s < 0)
    mode_b = 500 - mode_a
    if min(mode_a, mode_b) < 50:
        print("  [!] mode collapse: one mode is starved")
```

经典症状：真实数据的两个模式之一不再被生成。判别器也不再纠正它，因为这个模式从未以假样本的身份出现过。

## 常见陷阱

- **判别器太强。** 把 D 的学习率降低 2-5 倍，或加入实例/层级噪声。如果 D 的准确率超过 95%，G 就死了。
- **生成器死记某个模式。** 给 D 的输入加噪声，使用 minibatch-discriminator 层，或改用 WGAN-GP。
- **批归一化泄漏统计量。** 真实批次和伪造批次流经同一个 BN 层会混合二者的统计量。改用实例归一化或谱归一化。
- **刷 Inception 分数。** FID 和 IS 在样本量小的时候噪声很大。评估时使用 ≥10k 个样本。
- **条件生成任务里"一步采样就够"是个谎言。** 你仍然需要 CFG 系数、截断技巧和重采样才能得到可用的输出。

## 生产实践

2026 年的 GAN 技术选型：

| 场景 | 选择 |
|-----------|------|
| 照片级真实人脸、固定姿态 | StyleGAN3（最锐利、最小） |
| 动漫/风格化人脸 | StyleGAN-XL 或 Stable Diffusion LoRA |
| 图像到图像翻译 | Pix2Pix / CycleGAN（Phase 8 · 04）或 ControlNet（Phase 8 · 08） |
| 快速一步文生图 | 扩散模型的对抗蒸馏（SDXL-Turbo、SD3-Turbo） |
| 扩散训练器内部的感知损失 | 在图像裁剪块上跑的小型 GAN 判别器 |
| 任何多模态、开放式生成 | 别用 GAN——用扩散或流匹配 |

GAN 锐利但狭窄。一旦你的领域开放起来——照片、任意文本提示、视频——就换扩散模型。对抗这一招以组件的形式延续下来（感知损失、蒸馏），而不再是独立的生成器。

## 交付产物

保存 `outputs/skill-gan-debugger.md`。该技能接收一次失败的 GAN 训练（损失曲线、样本网格、数据集规模），输出按可能性排序的原因清单、一行式修复方案和重跑流程。

## 练习

1. **简单。** 用默认设置运行 `code/main.py`。然后设置 `D_LR = 5 * G_LR` 再跑一次。G 的损失多快坍缩为常数？
2. **中等。** 把 Goodfellow 的 BCE 损失换成 WGAN 损失：`loss_D = E[D(fake)] - E[D(real)]`、`loss_G = -E[D(fake)]`，并把 D 的权重裁剪到 `[-0.01, 0.01]`。训练更稳定了吗？对比墙钟时间下的收敛速度。
3. **困难。** 把一维例子扩展到二维数据（环上 8 个高斯的混合）。统计生成器在第 1k、5k、10k 步分别捕获了 8 个模式中的几个。实现 minibatch discrimination 并重新测量。

## 关键术语

| 术语 | 大家怎么叫 | 实际含义 |
|------|-----------------|-----------------------|
| 生成器（Generator） | "G" | 噪声到样本的网络，`G: z → x̂`。 |
| 判别器（Discriminator） | "D" | 分类器 `D: x → [0, 1]`，区分真假。 |
| 极小极大（Minimax） | "那场博弈" | 对联合目标取 `min_G max_D`。 |
| 非饱和损失 | "那个修复" | G 用 `-log D(G(z))` 替代 `log(1 - D(G(z)))`。 |
| 模式坍缩（Mode collapse） | "G 只会一招" | 数据多样但生成器只产出极少几种不同输出。 |
| WGAN | "Wasserstein" | 用 Earth-Mover 距离 + 梯度惩罚替代 BCE；梯度更平滑。 |
| 谱归一化 | "Lipschitz 技巧" | 约束 D 的权重范数以限制其斜率；稳定训练。 |
| StyleGAN | "唯一管用的那个" | 映射网络 + AdaIN；人脸生成同类最佳，2026 年仍是。 |

## 生产笔记：一步推理是 GAN 留下的持久优势

在开放域生成上，GAN 已不再以样本质量取胜，但在推理成本上依然占优。用生产推理文献的术语来说，GAN 具备：

- **没有 prefill、没有 decode 阶段。** 只需一次 `G(z)` 前向。TTFT ≈ 总延迟。
- **没有 KV 缓存压力。** 唯一的状态就是权重。批大小受限于激活内存，而非缓存。
- **连续批处理（continuous batching）变得平凡。** 由于每个请求消耗相同的固定 FLOPs，在服务器目标占用率下的静态批通常就是最优解。不需要在途调度器。

这就是为什么 GAN 蒸馏（SDXL-Turbo、SD3-Turbo、ADD、LCM）是 2026 年快速文生图的主流技术：它把 20-50 步的扩散流水线压缩成 1-4 次 GAN 式前向，同时保留扩散基座的分布。对抗损失作为一个训练期旋钮存活下来，专门把慢的生成器变快。

## 延伸阅读

- [Goodfellow et al. (2014). Generative Adversarial Nets](https://arxiv.org/abs/1406.2661) —— GAN 原始论文。
- [Radford et al. (2015). Unsupervised Representation Learning with DCGAN](https://arxiv.org/abs/1511.06434) —— 第一个稳定的架构。
- [Arjovsky, Chintala, Bottou (2017). Wasserstein GAN](https://arxiv.org/abs/1701.07875) —— WGAN。
- [Miyato et al. (2018). Spectral Normalization for GANs](https://arxiv.org/abs/1802.05957) —— 谱归一化。
- [Karras et al. (2020). Analyzing and Improving the Image Quality of StyleGAN](https://arxiv.org/abs/1912.04958) —— StyleGAN2。
- [Karras et al. (2021). Alias-Free Generative Adversarial Networks](https://arxiv.org/abs/2106.12423) —— StyleGAN3。
- [Sauer et al. (2023). Adversarial Diffusion Distillation](https://arxiv.org/abs/2311.17042) —— SDXL-Turbo。
