# 视频生成

> 图像是二维张量，视频则是三维的。理论完全相同，算力却要高出 10-100 倍。OpenAI 的 Sora（2024 年 2 月）证明了这条路走得通。到 2026 年，Veo 2、Kling 1.5、Runway Gen-3、Pika 2.0 和 WAN 2.2 都已能从文本生成 1080p 的生产级视频——而开放权重阵营（CogVideoX、HunyuanVideo、Mochi-1、WAN 2.2）只落后 12 个月。

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 8 · 07 (Latent Diffusion), Phase 7 · 09 (ViT), Phase 8 · 06 (DDPM)
**Time:** ~45 minutes

## 问题背景

一段 24fps、时长 10 秒的 1080p 视频包含 240 帧，每帧 1920×1080×3 像素，每个片段约 1.5 GB 原始数据。像素空间扩散根本不可行。你需要：

1. **时空压缩。** 一个对整段视频（而非逐帧）编码的 VAE，将其压缩成一串时空图块（spatial-temporal patch）。
2. **时序一致性。** 各帧需要在数秒内保持内容、光照和物体身份一致。网络必须建模运动。
3. **算力预算。** 同等模型规模下，视频训练比图像训练贵 10-100 倍。
4. **条件控制。** 文本、图像（首帧）、音频，或另一段视频。多数生产级模型四种都支持。

解决这个问题的架构是作用于时空图块的**扩散 Transformer（Diffusion Transformer, DiT）**，在海量（提示词、字幕、视频）数据集上训练。扩散损失与第 06 课完全相同。

## 核心概念

![Video diffusion: patchify, DiT, decode](../assets/video-generation.svg)

### 图块化（Patchify）

用 3D VAE（学习得到的时空压缩）对视频编码。潜变量的形状为 `[T_latent, H_latent, W_latent, C_latent]`。再切分成大小为 `[t_p, h_p, w_p]` 的图块。对于 Sora 风格的模型，`t_p = 1`（逐帧图块）或 `t_p = 2`（每两帧一块）。一段 10 秒 1080p 视频会压缩成约 20,000-100,000 个图块。

### 时空 DiT

一个 Transformer 处理展平后的图块序列。每个图块带有 3D 位置嵌入（时间 + y + x）。注意力通常做因子分解：

- **空间注意力**：在每一帧内部的图块之间计算。
- **时间注意力**：在不同帧的同一空间位置之间计算。
- **完整 3D 注意力**贵 16-100 倍，只在低分辨率或研究场景下使用。

### 文本条件

通过与大型文本编码器的交叉注意力实现（Sora 用 T5-XXL，CogVideoX-5B 也用 T5-XXL）。长提示词很重要——Sora 的训练集使用了 GPT 生成的稠密重标注字幕，每个片段平均 200 个 token。

### 训练

在时空潜变量上使用标准扩散损失（ε 或 v 预测）。数据：网络视频 + 约 1 亿条精选片段 + 合成文本字幕。算力：哪怕一次小规模研究性训练也要 10,000+ GPU 小时；Sora 级别则是 100,000+。

## 2026 年的生产格局

| 模型 | 时间 | 最长时长 | 最高分辨率 | 开放权重？ | 亮点 |
|-------|------|--------------|---------|---------------|---------|
| Sora (OpenAI) | 2024-02 | 60 秒 | 1080p | 否 | 首个在规模化下展现世界模拟器特性的模型 |
| Sora Turbo | 2024-12 | 20 秒 | 1080p | 否 | 生产版 Sora，推理速度快 5 倍 |
| Veo 2 (Google) | 2024-12 | 8 秒 | 4K | 否 | 2025 年画质与物理表现最佳 |
| Veo 3 | 2025 Q3 | 15 秒 | 4K | 否 | 原生音频与更强的镜头控制 |
| Kling 1.5 / 2.1 (Kuaishou) | 2024-2025 | 10 秒 | 1080p | 否 | 2025 Q1 人体运动表现最佳 |
| Runway Gen-3 Alpha | 2024-06 | 10 秒 | 768p | 否 | 上层配套专业视频工具 |
| Pika 2.0 | 2024-10 | 5 秒 | 1080p | 否 | 角色一致性最强 |
| CogVideoX (THUDM) | 2024 | 10 秒 | 720p | 是（2B、5B） | 首个开放的 5B 级视频模型 |
| HunyuanVideo (Tencent) | 2024-12 | 5 秒 | 720p | 是（13B） | 2024 年末的开源 SOTA |
| Mochi-1 (Genmo) | 2024-10 | 5.4 秒 | 480p | 是（10B） | 许可证最宽松 |
| WAN 2.2 (Alibaba) | 2025-07 | 5 秒 | 720p | 是 | 2025 年中最强开源模型 |

开放权重追赶的速度比当年图像领域更快：到 2026 年中，HunyuanVideo 加 WAN 2.2 的 LoRA 已经支撑了绝大多数开源工作流。

## 从零实现

`code/main.py` 模拟了时空 DiT 的核心思想：把一段小型合成视频做图块化，给每个图块加上位置嵌入，再用 Transformer 风格的图块间注意力对整个序列去噪。不用 numpy，纯 Python。我们将展示：哪怕在一维场景下，只要相邻帧的图块共享同一个去噪器和位置嵌入，时序一致性就会自然涌现。

### 第 1 步：图块化一段一维合成「视频」

```python
def make_video(T_frames=8, rng=None):
    # a "video" is a sequence of 1-D values following a smooth trajectory
    base = rng.gauss(0, 1)
    return [base + 0.3 * t + rng.gauss(0, 0.1) for t in range(T_frames)]
```

### 第 2 步：逐帧位置嵌入

```python
def pos_embed(t, dim):
    return sinusoidal(t, dim)
```

### 第 3 步：去噪器看到完整序列

我们的小网络不再逐帧独立去噪，而是把所有帧的值与各自的位置嵌入拼接起来，对所有帧的噪声做联合预测。

### 第 4 步：时序一致性测试

训练完成后采样一段视频，测量帧间差值。如果模型学到了时序结构，这些差值会比逐帧独立采样时更小。

## 常见陷阱

- **逐帧独立采样 = 闪烁。** 如果对每一帧单独跑图像扩散，输出会闪烁，因为每帧的噪声彼此独立。视频扩散通过注意力或共享噪声把各帧耦合起来，从而解决这个问题。
- **朴素 3D 注意力 = 显存爆炸（OOM）。** 在 10 秒 1080p 的潜变量上做完整 3D 注意力需要数千亿次运算。要分解为空间 + 时间。
- **数据标注比数据规模更重要。** Sora 相比此前工作的最大升级，是在约 10 倍更详细的字幕（GPT-4 重新标注的片段）上训练。OpenAI 的技术报告对此说得很明确。
- **首帧条件。** 多数生产级模型还接受一张图像作为首帧。这就是「图生视频（image-to-video）」模式；训练时会包含这一变体。
- **物理漂移。** 长片段（>10 秒）会累积细微的不一致。滑动窗口生成 + 关键帧锚定能缓解。

## 生产实践

| 使用场景 | 2026 年首选 |
|----------|-----------|
| 最高质量文生视频，托管服务 | Veo 3 或 Sora |
| 镜头可控的电影级画面 | Runway Gen-3 配合运动笔刷 |
| 跨片段角色一致性 | Pika 2.0 或 Kling 2.1 |
| 开放权重、快速微调 | WAN 2.2 + LoRA |
| 图生视频 | WAN 2.2-I2V、Kling 2.1 I2V 或 Runway |
| 音频驱动唇形同步 | Veo 3（原生音频）或专用唇同步模型 |
| 视频编辑 | Runway Act-Two、Kling Motion Brush、Flux-Kontext（静帧） |

在画质相当的前提下，每秒视频的生成成本从 2024 年到 2026 年下降了 20 倍。

## 交付产物

保存 `outputs/skill-video-brief.md`。该 Skill 接收一份视频需求简报（时长、宽高比、风格、镜头方案、主体一致性、音频），输出：模型与托管方案、提示词脚手架（镜头语言、主体描述、运动描述词）、种子与可复现协议，以及一份帧级 QA 检查清单。

## 练习

1. **简单。** 在 `code/main.py` 中，比较 (a) 逐帧独立采样与 (b) 整序列联合采样的帧间差值。报告差值的均值和方差。
2. **中等。** 添加首帧条件：把第 0 帧固定为给定值，再采样其余帧。测量这个固定值如何向后传播。
3. **困难。** 用 HuggingFace diffusers 在本地 GPU 上运行 CogVideoX-2B。在 720p 下为 6 秒片段计时 20 步推理。对时空注意力做性能剖析，找出瓶颈。

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|-----------------|-----------------------|
| 视频 VAE | 「3D VAE」 | 把 `(T, H, W, C)` 压缩成时空潜变量的编码器。 |
| 图块（Patch） | 「就是 token」 | 潜变量中固定大小的三维块；DiT 的输入。 |
| 因子分解注意力 | 「空间 + 时间」 | 先在空间维度做注意力，再在时间维度做；跳过完整 3D 注意力。 |
| 图生视频（I2V） | 「让这张照片动起来」 | 模型接收一张图像 + 文本，输出一段以该图像开头的视频。 |
| 关键帧条件 | 「锚定帧」 | 固定特定帧，以控制视频的整体走向。 |
| 运动笔刷 | 「方向提示」 | 一种 UI 输入方式，用户在图像上绘制运动向量。 |
| 重标注（Re-captioning） | 「稠密字幕」 | 用 LLM 给训练片段重新打上详细提示词标签。 |
| 闪烁（Flicker） | 「时序伪影」 | 帧间不一致；通过耦合去噪来修复。 |

## 生产笔记：视频潜变量是一个显存带宽问题

一段 24 fps、10 秒的 1080p 片段是 240 帧 × 1920 × 1080 × 3 ≈ 1.5 GB 原始像素。经过 4 倍视频 VAE 压缩（`2 × spatial × 2 × temporal`）后，每个请求的潜变量约 100 MB。把它送进时空 DiT 跑 30 步、batch 为 1，每步要在 HBM 中搬运约 3 GB——瓶颈是显存带宽，而不是 FLOPs。

三个生产调节手段，全部来自生产推理文献的推理章节：

- **对 DiT 做张量并行（TP）。** 文生视频模型动辄 ≥10B 参数。在 4 张 H100 上 TP=4 是标配；405B 级模型用 PP=2 × TP=2。在撞上 all-reduce 墙之前，每步延迟随 TP 大致线性下降。
- **帧批处理 = 连续批处理（continuous batching）。** 在生成阶段，视频在概念上就是一批由注意力连接的帧。连续批处理（在途调度）同样适用：如果模型架构支持滑动窗口生成，可以在返回第 `t-1` 帧的同时开始渲染第 `t+1` 帧。
- **片段级 prefill 缓存。** 对图生视频而言，首帧条件类似于 LLM 的提示词 prefill：算一次，在多轮时序解码中复用。这实际上就是视频版的 KV 缓存。

## 延伸阅读

- [Brooks et al. (2024). Video generation models as world simulators](https://openai.com/index/video-generation-models-as-world-simulators/) — Sora 技术报告。
- [Yang et al. (2024). CogVideoX: Text-to-Video Diffusion Models with An Expert Transformer](https://arxiv.org/abs/2408.06072) — CogVideoX。
- [Kong et al. (2024). HunyuanVideo: A Systematic Framework for Large Video Generative Models](https://arxiv.org/abs/2412.03603) — HunyuanVideo。
- [Genmo (2024). Mochi-1 Technical Report](https://www.genmo.ai/blog/mochi) — Mochi-1。
- [Alibaba (2025). WAN 2.2](https://wanvideo.io/) — 2025 年中的开源 SOTA。
- [Ho, Salimans, Gritsenko et al. (2022). Video Diffusion Models](https://arxiv.org/abs/2204.03458) — 视频扩散的开山之作。
- [Blattmann et al. (2023). Align your Latents (Video LDM)](https://arxiv.org/abs/2304.08818) — Stable Video Diffusion 的前身。
