# 图像修复、外扩与图像编辑

> 文生图负责创造新内容，图像修复（inpainting）负责修补旧内容。在生产环境中，70% 的付费图像工作都是编辑——换背景、去 logo、扩画布、重新生成一只手。图像修复正是扩散模型真正创造价值的地方。

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 8 · 07 (Latent Diffusion), Phase 8 · 08 (ControlNet & LoRA)
**Time:** ~75 minutes

## 问题背景

客户发来一张完美的产品照片，但背景里有一块碍眼的招牌。你想把招牌抹掉，同时让其余部分的像素一动不动。你不能从头跑一遍文生图——结果会有不同的颜色、不同的光照、不同的产品角度。你想要的是*只*重新生成被遮罩的区域，并且让重新生成的内容与周围上下文保持协调。

这就是图像修复（inpainting）。它有几个变体：

- **图像修复（Inpainting）。** 在遮罩内部重新生成，保留外部像素。
- **图像外扩（Outpainting）。** 在遮罩外部（或画布之外）重新生成，保留内部。
- **图像编辑（Image editing）。** 重新生成整张图像，但在语义或结构上保持对原图的忠实度（SDEdit、InstructPix2Pix）。

2026 年的每一个扩散模型流水线都自带图像修复模式：Flux.1-Fill、Stable Diffusion Inpaint、SDXL-Inpaint、DALL-E 3 Edit。它们的工作原理是一样的。

## 核心概念

![Inpainting: mask-aware denoising with context-preserving reinjection](../assets/inpainting.svg)

### 朴素做法（以及为什么它行不通）

带着遮罩跑标准的文生图。在每个采样步，把含噪潜变量中未遮罩区域替换为干净图像经前向扩散后的版本。它能跑……但效果很差。边界伪影会渗出来，因为模型对遮罩区域内是什么一无所知。

### 真正的修复模型

训练一个改造过的 U-Net，输入通道从 4 个变为 9 个：

```
input = concat([ noisy_latent (4ch), encoded_image (4ch), mask (1ch) ], dim=channel)
```

额外的通道是一份经 VAE 编码的源图像，再加上单通道的遮罩。训练时，随机遮住图像的某些区域，训练模型只对遮罩区域去噪，而未遮罩区域作为干净的条件信号提供。推理时，模型能"看到"遮罩区域周围有什么，从而生成连贯的补全。

SD-Inpaint、SDXL-Inpaint、Flux-Fill 都使用这种 9 通道（或类似）的输入。Diffusers 中对应 `StableDiffusionInpaintPipeline`、`FluxFillPipeline`。

### SDEdit（Meng et al., 2022）——免费的编辑

把源图像加噪到某个中间时刻 `t`，然后用新的提示词从 `t` 反向运行去噪链直到 0。无需重新训练。起始 `t` 的选择在忠实度与创作自由度之间权衡：

- `t/T = 0.3` → 与源图几乎一致，仅有细微的风格变化
- `t/T = 0.6` → 中等程度的编辑，保留粗粒度结构
- `t/T = 0.9` → 几乎从纯噪声生成，对源图的保留极少

### InstructPix2Pix（Brooks et al., 2023）

在 `(input_image, instruction, output_image)` 三元组上微调扩散模型。推理时同时以输入图像和文字指令（"改成日落"、"加一条龙"）作为条件。有两个 CFG 系数：图像系数和文本系数。

### RePaint（Lugmayr et al., 2022）

沿用标准的无条件扩散模型。在每个反向步骤中进行重采样——偶尔跳回到噪声更大的状态再重新生成。这样能避免边界伪影。当你手头没有训练好的修复模型时使用。

## 从零实现

`code/main.py` 在 5 维数据上实现了一个玩具版的一维图像修复方案。我们在 5 维混合数据上训练一个 DDPM，每个样本是来自两个簇之一的 5 个浮点数。推理时，我们"遮住"5 个维度中的 2 个，在每一步注入未遮罩的 3 个维度经前向加噪后的版本，只重新生成被遮罩的维度。

### 第 1 步：5 维 DDPM 数据

```python
def sample_data(rng):
    cluster = rng.choice([0, 1])
    center = [-1.0] * 5 if cluster == 0 else [1.0] * 5
    return [c + rng.gauss(0, 0.2) for c in center], cluster
```

### 第 2 步：在全部 5 个维度上训练去噪器

标准 DDPM。网络对 5 维含噪输入输出 5 维的噪声预测。

### 第 3 步：推理时执行遮罩感知的反向过程

```python
def inpaint_step(x_t, mask, clean_image, alpha_bars, t, rng):
    # replace unmasked dims with a freshly noised version of the clean source
    a_bar = alpha_bars[t]
    for i in range(len(x_t)):
        if not mask[i]:
            x_t[i] = math.sqrt(a_bar) * clean_image[i] + math.sqrt(1 - a_bar) * rng.gauss(0, 1)
    # ...then run the normal reverse step on x_t
```

这就是朴素做法，在玩具级的一维数据上它是可行的。真实的图像修复使用 9 通道输入，因为纹理连贯性更重要。

### 第 4 步：图像外扩

外扩就是把遮罩取反的图像修复：遮住新的（原先不存在的）画布区域，其余部分用原图填充。训练目标完全相同。

## 常见陷阱

- **接缝。** 朴素做法会留下可见的边界，因为梯度信息无法跨越遮罩流动。解决：把遮罩膨胀 8-16 个像素，或者使用真正的修复模型。
- **遮罩泄漏。** 如果条件图像中未遮罩区域质量低或有噪点，会污染遮罩内的生成结果。可以先轻微去噪或模糊处理。
- **CFG 与遮罩大小相互影响。** 小遮罩配高 CFG = 过饱和的色块。小范围编辑时降低 CFG。
- **SDEdit 的忠实度悬崖。** 从 `t/T = 0.5` 调到 `t/T = 0.6` 可能就丢失了主体的身份特征。逐档扫参并保存检查点。
- **提示词不匹配。** 提示词应该描述*整张*图像，而不只是新增的内容。要写"一只猫坐在椅子上"，而不是"一只猫"。

## 生产实践

| 任务 | 流水线 |
|------|----------|
| 移除物体，小遮罩 | SD-Inpaint 或 Flux-Fill，标准提示词 |
| 替换天空 | SD-Inpaint + "blue sky at sunset" |
| 扩展画布 | SDXL 外扩模式（8px 羽化）或带外扩遮罩的 Flux-Fill |
| 重新生成手 / 脸 | SD-Inpaint，提示词重新描述主体 + ControlNet-Openpose |
| 改变某一区域的风格 | 对遮罩区域用 `t/T=0.5` 的 SDEdit |
| "改成日落" | InstructPix2Pix 或 Flux-Kontext |
| 背景替换 | SAM 遮罩 → SD-Inpaint |
| 超高保真 | 最难的场景用 Flux-Fill 或 GPT-Image（托管服务） |

SAM（Meta 的 Segment Anything，2023）+ 扩散修复是 2026 年的背景去除标准流水线。SAM 2（2024）支持视频。

## 交付产物

保存 `outputs/skill-editing-pipeline.md`。该技能的输入是原图 + 编辑描述 + 可选的遮罩（或 SAM 提示），输出包括：遮罩生成方案、基础模型、CFG 系数（图像 + 文本）、SDEdit 的 t 值或修复模式，以及 QA 检查清单。

## 练习

1. **简单。** 在 `code/main.py` 中，把被遮罩维度的比例从 0.2 变到 0.8。在哪个比例时修复质量（遮罩维度上的残差）与无条件生成持平？
2. **中等。** 实现 RePaint：每 10 个反向步骤就回跳 5 步（加噪）再重新去噪。测量它是否减小了遮罩边缘的边界残差。
3. **困难。** 用 Hugging Face diffusers 对比：SD 1.5 Inpaint + ControlNet-Openpose 与 Flux.1-Fill 在 20 个人脸重生成任务上的表现。分别评估姿态贴合度和身份保持度。

## 关键术语

| 术语 | 大家怎么说 | 实际含义 |
|------|-----------------|-----------------------|
| 图像修复（Inpainting） | "把洞补上" | 在遮罩内部重新生成；保留外部像素。 |
| 图像外扩（Outpainting） | "把画布扩出去" | 在画布之外重新生成；保留内部。 |
| 9 通道 U-Net | "真正的修复模型" | 以 `noisy \| encoded-source \| mask` 为输入的 U-Net。 |
| SDEdit | "带噪声强度的 img2img" | 加噪到时刻 `t`，再用新提示词去噪。 |
| InstructPix2Pix | "纯文字编辑" | 在（图像、指令、输出）三元组上微调的扩散模型。 |
| RePaint | "不用重新训练" | 反向过程中周期性重新加噪，以减少接缝。 |
| SAM | "Segment Anything" | 通过点击或框选生成遮罩的工具；与修复配套使用。 |
| Flux-Kontext | "带上下文的编辑" | 接受参考图像 + 指令进行编辑的 Flux 变体。 |

## 生产笔记：编辑流水线对延迟很敏感

正在编辑图像的用户期望 5 秒以内的往返时间。1024² 分辨率下 30 步的 SDXL-Inpaint 在 L4 上需要 3-4 秒，再加上 SAM 遮罩生成（约 200 毫秒）和 VAE 编解码（合计约 500 毫秒）。按生产视角来看，这是 TTFT 受限而非吞吐受限的场景——批大小为 1、低并发，需要压缩每一个阶段：

- **SAM-H 是慢的那个。** 1024² 下 SAM-H 约 200 毫秒；SAM-ViT-B 约 40 毫秒，质量损失很小。SAM 2（视频版）会增加时序开销；不要在单图编辑中使用它。
- **能跳过编码就跳过。** `pipe.image_processor.preprocess(img)` 会编码到潜空间。如果你手头有上一次生成的潜变量（迭代编辑 UI 中很常见），通过 `latents=...` 直接传入即可省掉一次 VAE 编码。
- **遮罩膨胀对吞吐量也有影响。** 遮罩很小意味着 U-Net 前向计算的大部分都被浪费了（未遮罩的像素反正会被钳制回原值）。`diffusers` 的 `StableDiffusionInpaintPipeline` 无论如何都会跑完整的 U-Net；只有 9 通道的真正修复变体才能利用遮罩压缩计算。
- **Flux-Kontext 是 2025 年的答案。** 对 `(source_image, instruction)` 做单次前向计算——不需要单独的遮罩，也不需要 SDEdit 的噪声扫参。在 H100 上约 1.5 秒就能完成一次编辑。架构层面的启示：把多个阶段合并成一个。

## 延伸阅读

- [Lugmayr et al. (2022). RePaint: Inpainting using Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2201.09865) —— 免训练的图像修复。
- [Meng et al. (2022). SDEdit: Guided Image Synthesis and Editing with Stochastic Differential Equations](https://arxiv.org/abs/2108.01073) —— SDEdit。
- [Brooks, Holynski, Efros (2023). InstructPix2Pix](https://arxiv.org/abs/2211.09800) —— 文字指令编辑。
- [Kirillov et al. (2023). Segment Anything](https://arxiv.org/abs/2304.02643) —— SAM，遮罩的来源。
- [Ravi et al. (2024). SAM 2: Segment Anything in Images and Videos](https://arxiv.org/abs/2408.00714) —— 视频版 SAM。
- [Hertz et al. (2022). Prompt-to-Prompt Image Editing with Cross-Attention Control](https://arxiv.org/abs/2208.01626) —— 注意力层面的编辑。
- [Black Forest Labs (2024). Flux.1-Fill and Flux.1-Kontext](https://blackforestlabs.ai/flux-1-tools/) —— 2024 年的工具链。
