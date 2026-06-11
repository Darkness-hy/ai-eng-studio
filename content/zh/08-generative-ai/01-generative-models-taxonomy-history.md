# 生成模型——分类与历史

> 每一个图像模型、文本模型、视频模型和 3D 模型，都能归入五个类别之一。选错类别，你会跟数学搏斗好几周；选对类别，过去十二年这个领域的全部进展都能在你脑中清晰成形。

**Type:** Learn
**Languages:** Python
**Prerequisites:** Phase 2 (ML Fundamentals), Phase 3 (Deep Learning Core), Phase 7 · 14 (Transformers)
**Time:** ~45 minutes

## 问题背景

生成模型只做一件事：给定从某个未知分布 `p_data(x)` 中采样得到的训练样本，输出看起来像是来自同一分布的新样本。人脸、句子、MIDI 文件、蛋白质结构——眯起眼睛看，都是同一个问题。

麻烦在于 `p_data` 存在于一个有数百万维的空间中（一张 512x512 的 RGB 图像约有 78.6 万维），样本只分布在该空间内的一个薄薄的流形上，而你手头大概只有 1000 万个样本。暴力求解密度毫无希望。每一种生成模型都是一种妥协：用一个稍微不那么难的问题，换掉一个极难的问题。

过去十二年里有五个家族存活了下来。了解每个家族各自做了什么妥协，就能明白它为什么在某些任务上取胜、在另一些任务上崩溃。

## 核心概念

![Five families of generative models — taxonomy by what they model](../assets/taxonomy.svg)

**1. 显式密度，可解析求解（tractable）。** 把 `log p(x)` 写成一个你真的能计算的求和式。自回归模型（PixelCNN、WaveNet、GPT）将其分解为 `p(x) = ∏ p(x_i | x_<i)`。归一化流（normalizing flows，如 RealNVP、Glow）把 `p(x)` 构造成对一个简单基础分布的可逆变换。优点：精确的似然，干净的训练损失。缺点：自回归推理是串行的（长序列时很慢），流模型需要可逆架构（架构上限制很大）。

**2. 显式密度，近似求解。** 给 `log p(x)` 找一个下界（ELBO），然后优化这个下界。VAE（Kingma 2013）使用带变分后验的编码器-解码器结构。扩散模型（DDPM，Ho 2020）训练一个去噪器，隐式地优化一个加权 ELBO。到 2026 年，扩散模型是图像、视频和 3D 领域占主导地位的骨干。

**3. 隐式密度。** 彻底跳过密度建模；学习一个生成器 `G(z)` 来生成样本，再学一个判别器 `D(x)` 来区分真假。这就是 GAN（Goodfellow 2014）。推理很快（一次前向传播），但训练出了名地不稳定。即使在 2026 年，StyleGAN 1/2/3 在固定领域的照片级真实感（人脸、卧室）上仍是最先进水平。

**4. 基于分数（score-based）/ 连续时间。** 直接学习对数密度的梯度 `∇_x log p(x)`（即分数）。Song & Ermon（2019）证明了分数匹配把扩散推广为一个 SDE。流匹配（flow matching，Lipman 2023）是 2024-2026 年的热点：无需模拟的训练、更直的路径、比 DDPM 快 4-10 倍的采样。Stable Diffusion 3、Flux、AudioCraft 2 都使用流匹配。

**5. 基于离散编码的 token 自回归。** 用 VQ-VAE 或残差量化器把高维数据压缩成一个短的离散 token 序列，然后用 Transformer 对该 token 序列建模。Parti、MuseNet、AudioLM、VALL-E、Sora 的图块分词器都用这种方法。这其实就是类别 1 加上一个学习得到的分词器。

## 简史

| 年份 | 模型 | 为什么重要 |
|------|-------|-----------------|
| 2013 | VAE (Kingma) | 第一个拥有可用训练损失的深度生成模型。 |
| 2014 | GAN (Goodfellow) | 隐式密度、不需要似然——样本清晰得令人震惊。 |
| 2015 | DRAW, PixelCNN | 序列式图像生成。 |
| 2017 | Glow, RealNVP | 可逆流；用深度换取精确似然。 |
| 2017 | Progressive GAN | 第一批百万像素级人脸。 |
| 2019 | StyleGAN / StyleGAN2 | 照片级真实人脸，在这个单一领域至今难以超越。 |
| 2020 | DDPM (Ho) | 扩散模型变得实用。 |
| 2021 | CLIP, DALL-E 1, VQGAN | 文生图走向主流。 |
| 2022 | Imagen, Stable Diffusion 1, DALL-E 2 | 潜空间扩散 + 文本条件控制 = 大众商品。 |
| 2022 | ControlNet, LoRA | 对预训练扩散模型的精细控制。 |
| 2023 | SDXL, Midjourney v5, Flow matching | 规模化 + 更好的训练动力学。 |
| 2024 | Sora, Stable Diffusion 3, Flux.1 | 视频扩散；流匹配胜出。 |
| 2025 | Veo 2, Kling 1.5, Runway Gen-3, Nano Banana | 生产级视频。 |
| 2026 | Consistency + Rectified Flow | 基于扩散骨干的一步采样。 |

## 五问分诊法

每当一篇新的生成模型论文发布，先回答以下五个问题，再去读方法部分。

1. **建模对象是什么？** 像素、潜变量、离散 token、3D 高斯、网格，还是波形？
2. **密度是显式还是隐式？** 他们有没有写出 `log p(x)`？
3. **采样是一步到位还是迭代式的？** 迭代式意味着推理更慢；一步到位通常意味着对抗式或蒸馏式方法。
4. **条件控制是哪种：无条件、类别、文本、图像、姿态？** 这决定了损失函数和架构的脚手架。
5. **评估指标用什么：FID、CLIP score、IS、人类偏好，还是任务准确率？** 每一种都有已知的失效模式（见第 14 课）。

本阶段的每一课你都会重新回答这五个问题。等到结束时，它们会成为你的条件反射。

## 从零实现

本课的代码是一个轻量级可视化：用三种玩具方法（核密度估计、离散直方图，以及一个最近样本式的"类 GAN"生成器）从样本中拟合一个一维高斯混合分布，让你在一个能打印在一屏内的问题上，直观看到显式密度与隐式密度的区别。

运行 `code/main.py`。它从一个双峰高斯混合分布中抽取 2000 个样本，然后打印：

```
explicit density (histogram): p(x in [-0.5, 0.5]) ≈ 0.38
approximate density (KDE):     p(x in [-0.5, 0.5]) ≈ 0.41
implicit (nearest-sample gen): 20 new samples printed, no p(x)
```

注意：前两种方法允许你问"这个点的可能性有多大？"，第三种做不到。这就是*显式与隐式*的区别，它在之后的每一课中都至关重要。

## 生产实践

2026 年，哪个家族适合哪类任务？

| 任务 | 最佳家族 | 原因 |
|------|-------------|-----|
| 照片级人脸，窄领域 | StyleGAN 2/3 | 仍然最清晰，推理最快。 |
| 通用文生图 | 潜空间扩散 + 流匹配 | SD3、Flux.1、DALL-E 3。 |
| 快速文生图 | Rectified flow + 蒸馏 | SDXL-Turbo、SD3-Turbo、LCM。 |
| 文生视频 | Diffusion Transformer + 流匹配 | Sora、Veo 2、Kling。 |
| 语音 + 音乐 | 基于 token 的自回归（AudioLM、VALL-E、MusicGen）或流匹配（AudioCraft 2） | 离散 token 的扩展成本低。 |
| 3D 场景 | Gaussian Splatting 拟合，扩散先验 | 3D-GS 用于重建，扩散用于新视角合成。 |
| 密度估计（不需要采样） | 流模型 | 唯一拥有精确 `log p(x)` 的家族。 |
| 仿真 / 物理 | 流匹配、score SDE | 直线路径，平滑的向量场。 |

## 交付产物

保存为 `outputs/skill-model-chooser.md`。

该技能接收一段任务描述，输出：(1) 应该使用哪个家族，(2) 三个开源方案和三个托管方案的排序列表，(3) 你需要警惕的最可能的失效模式，以及 (4) 算力/时间预算。

## 练习

1. **简单。** 针对以下五个产品，识别其所属家族与骨干模型：ChatGPT 图像生成、Midjourney v7、Sora、Runway Gen-3、ElevenLabs。证据应来自公开的技术报告。
2. **中等。** 你明天要读的论文声称采样速度比扩散模型快 100 倍。写下三个问题，用来检验这个加速在加入条件控制和高分辨率后是否依然成立。
3. **困难。** 选一个你关心的领域（例如蛋白质结构、CAD、分子、轨迹）。对该领域当前的 SOTA 模型回答五问分诊法，并勾勒出一个更好的模型会改变什么。

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|-----------------|-----------------------|
| 生成模型 | "它能造新东西" | 学习 `p_data(x)` 的采样器，可选地暴露 `log p(x)`。 |
| 显式密度 | "你能计算它" | 模型提供闭式或可解析求解的 `log p(x)`。 |
| 隐式密度 | "GAN 风格" | 只有采样器——无法计算给定点的 `p(x)`。 |
| ELBO | "证据下界" | `log p(x)` 的一个可解析求解的下界；VAE 和扩散模型优化的就是它。 |
| 分数（score） | "对数密度的梯度" | `∇_x log p(x)`；扩散模型和 SDE 模型学习的就是这个场。 |
| 流形假设 | "数据躺在一个曲面上" | 高维数据集中在一个低维流形上；这是降维之所以有效的原因。 |
| 自回归 | "预测下一块" | 把联合分布分解为条件分布的乘积。 |
| 潜变量 | "压缩编码" | 一种低维表示，解码器可以由它重建输入。 |

## 生产笔记：五个家族，五种推理形态

每个家族对应一条不同的推理服务成本曲线。生产推理领域的文献把 LLM 推理拆解为预填充（prefill）+ 解码（decode）；同样的分解也适用于这里：

- **自回归（类别 1 和 5）。** 串行解码主导延迟；KV 缓存、连续批处理（continuous batching）和投机解码（speculative decoding）都可以直接套用。
- **VAE / 扩散 / 流匹配（类别 2 和 4）。** 不存在 LLM 意义上的解码。成本 = `num_steps × step_cost`，其中 `step_cost` 是一次完整潜空间分辨率下的 Transformer 或 U-Net 前向传播。生产调优的旋钮是步数（DDIM / DPM-Solver / 蒸馏）、批大小和精度（bf16 / fp8 / int4）。
- **GAN（类别 3）。** 一次前向传播。没有调度，没有 KV 缓存。TTFT ≈ 总延迟。这就是 StyleGAN 在窄领域用户体验上至今仍占优的原因。

当你在论文摘要里看到"比扩散更快"时，把它翻译成"更少的步数 × 相同的单步成本"或"相同的步数 × 更便宜的单步成本"。其余的都是营销话术。

## 延伸阅读

- [Goodfellow et al. (2014). Generative Adversarial Nets](https://arxiv.org/abs/1406.2661) —— GAN 论文。
- [Kingma & Welling (2013). Auto-Encoding Variational Bayes](https://arxiv.org/abs/1312.6114) —— VAE 论文。
- [Ho, Jain, Abbeel (2020). Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2006.11239) —— DDPM 论文。
- [Song et al. (2021). Score-Based Generative Modeling through SDEs](https://arxiv.org/abs/2011.13456) —— 把扩散视为 SDE。
- [Lipman et al. (2023). Flow Matching for Generative Modeling](https://arxiv.org/abs/2210.02747) —— 流匹配论文。
- [Esser et al. (2024). Scaling Rectified Flow Transformers for High-Resolution Image Synthesis](https://arxiv.org/abs/2403.03206) —— Stable Diffusion 3。
