# 潜在扩散与 Stable Diffusion

> 在 512×512 图像上做像素空间扩散，简直是对算力的暴行。Rombach 等人（2022）注意到：生成一张图像并不需要全部 786k 个维度——只需要足以刻画语义结构的维度，剩下的交给一个独立的解码器即可。把扩散过程放进 VAE 的潜在空间里跑。这一个想法，就是 Stable Diffusion。

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 8 · 02 (VAE), Phase 8 · 06 (DDPM), Phase 7 · 09 (ViT)
**Time:** ~75 minutes

## 问题背景

在 512² 分辨率上做像素空间扩散，意味着 U-Net 要在形状为 `[B, 3, 512, 512]` 的张量上运行。对一个 5 亿参数的 U-Net 来说，每个采样步约需 100 GFLOPS。五十步就是每张图 5 TFLOPS。在十亿张图像上训练，算力账单将高得离谱。

这些 FLOPs 大部分都耗在让感知上无关紧要的细节穿过网络——那些高频纹理本可以被一个有损 VAE 压缩掉。Rombach 的想法是：先训练一次 VAE（即*第一阶段*），冻结它，然后让扩散完全在 4 通道 64×64 的潜在空间中进行（即*第二阶段*）。同样的 U-Net。像素数只有 1/16。FLOPs 约减少 64 倍，质量却相当。

这就是 Stable Diffusion 的配方。SD 1.x / 2.x 在 `64×64×4` 潜在表示上使用 8.6 亿参数的 U-Net，SDXL 在 `128×128×4` 上使用 26 亿参数的 U-Net，SD3 则把 U-Net 换成了搭配流匹配（flow matching）的扩散 Transformer（Diffusion Transformer, DiT）。Flux.1-dev（Black Forest Labs，2024）搭载了一个 120 亿参数的 DiT-MMDiT。它们全部运行在同一套两阶段基座之上。

## 核心概念

![Latent diffusion: VAE compression + diffusion in latent space](../assets/latent-diffusion.svg)

**两个阶段，分开训练。**

1. **阶段 1——VAE。** 编码器 `E(x) → z`，解码器 `D(z) → x`。目标压缩率：每个空间轴上 8 倍下采样，并调整通道数，使潜在表示的总大小约为像素数的 1/16。损失 = 重建损失（L1 + LPIPS 感知损失）+ KL 损失（权重很小，避免把 `z` 强行压成过于高斯的分布，因为我们并不需要从 `z` 精确采样）。通常还会加上对抗损失，让解码出的图像更锐利。

2. **阶段 2——在 `z` 上做扩散。** 把 `z = E(x_real)` 当作数据。训练一个 U-Net（或 DiT）对 `z_t` 去噪。推理时：通过扩散采样得到 `z_0`，再计算 `x = D(z_0)`。

**文本条件化。** 需要两个额外组件。一个冻结的文本编码器（SD 1.x 用 CLIP-L，SD 2/XL 用 CLIP-L+OpenCLIP-G，SD3 和 Flux 用 T5-XXL）。一个交叉注意力注入机制：U-Net 的每个块都接收 `[Q = 图像特征, K = V = 文本 token]` 并把它们混合进来。这些 token 是文本影响图像的唯一通道。

**损失函数与第 06 课完全相同。** 同样是 DDPM / 流匹配的噪声 MSE。你只是换了数据域而已。

## 架构

| 模型 | 年份 | 主干网络 | 潜在表示形状 | 文本编码器 | 参数量 |
|-------|------|----------|--------------|--------------|--------|
| SD 1.5 | 2022 | U-Net | 64×64×4 | CLIP-L（77 token） | 860M |
| SD 2.1 | 2022 | U-Net | 64×64×4 | OpenCLIP-H | 865M |
| SDXL | 2023 | U-Net + refiner | 128×128×4 | CLIP-L + OpenCLIP-G | 2.6B + 6.6B |
| SDXL-Turbo | 2023 | 蒸馏版 | 128×128×4 | 同上 | 1-4 步采样 |
| SD3 | 2024 | MMDiT（多模态 DiT） | 128×128×16 | T5-XXL + CLIP-L + CLIP-G | 2B / 8B |
| Flux.1-dev | 2024 | MMDiT | 128×128×16 | T5-XXL + CLIP-L | 12B |
| Flux.1-schnell | 2024 | MMDiT 蒸馏版 | 128×128×16 | T5-XXL + CLIP-L | 12B，1-4 步 |

整体趋势：用 DiT 取代 U-Net（在潜在 patch 上运行的 Transformer），扩大文本编码器规模（在提示词遵循度上 T5 胜过 CLIP），增加潜在通道数（4 → 16 为细节留出更多空间）。

```figure
noise-schedule
```

## 从零实现

`code/main.py` 在第 06 课的 DDPM 之上叠加了一个玩具版一维「VAE」（恒等编码器 + 解码器，仅作演示；真正的 VAE 会是卷积网络），并加入了带无分类器引导（classifier-free guidance）的类别条件化。它表明：无论是在原始一维数值上还是在编码后的数值上，同一个扩散损失都同样有效——这正是关键洞察。

### 第 1 步：编码器/解码器

```python
def encode(x):    return x * 0.5          # toy "compression" to smaller scale
def decode(z):    return z * 2.0
```

真正的 VAE 拥有训练出的权重。但出于教学目的，这个线性映射足以说明：扩散在 `z` 上运行，完全不关心原始数据空间是什么。

### 第 2 步：在 `z` 空间做扩散

与第 06 课相同的 DDPM。网络看到的数据是 `z = E(x)`。采样得到 `z_0` 后，用 `D(z_0)` 解码。

### 第 3 步：无分类器引导

训练时，有 10% 的概率丢弃类别标签（替换为空 token）。推理时，同时计算 `ε_cond` 和 `ε_uncond`，然后：

```python
eps_cfg = (1 + w) * eps_cond - w * eps_uncond
```

`w = 0` = 无引导（多样性最大），`w = 3` = 默认值，`w = 7+` = 饱和 / 过度锐化。

### 第 4 步：文本条件化（仅讲概念，不写代码）

把类别标签换成冻结文本编码器的输出。通过交叉注意力把文本嵌入送入 U-Net：

```python
h = h + CrossAttention(Q=h, K=text_embed, V=text_embed)
```

这就是类别条件扩散模型与 Stable Diffusion 之间唯一的实质性区别。

## 常见陷阱

- **VAE 缩放不匹配。** SD 1.x 的 VAE 在编码后会施加一个缩放常数（`scaling_factor ≈ 0.18215`）。忘记这一步，会让 U-Net 在方差严重失常的潜在表示上训练。每个 checkpoint 都附带这个常数。
- **文本编码器悄悄出错。** SD3 需要 T5-XXL 且 token 数 >=128，退回纯 CLIP 会丢失信息。务必确认 `use_t5=True`，否则提示词保真度会一落千丈。
- **混用潜在空间。** SDXL、SD3、Flux 各自使用不同的 VAE。在 SDXL 潜在空间上训练的 LoRA 在 SD3 上不会生效。Hugging Face diffusers 0.30+ 会拒绝加载不匹配的 checkpoint。
- **CFG 过高。** `w > 10` 会产生饱和、油腻的图像，并以牺牲多样性为代价过拟合提示词。最佳区间是 `w = 3-7`。
- **负向提示词泄漏。** 空的负向提示词会变成空 token；填了内容的负向提示词则成为 `ε_uncond`。两者并不相同；有些 pipeline 会悄悄默认使用空 token。

## 生产实践

2026 年的生产技术栈：

| 目标 | 推荐主干 |
|--------|----------------------|
| 窄领域、配对数据、从头训练模型 | SDXL 微调（LoRA / 全量）——最快上线 |
| 开放领域文生图、开放权重 | Flux.1-dev（12B，Apache / 非商用）或 SD3.5-Large |
| 推理速度最快、开放权重 | Flux.1-schnell（1-4 步，Apache）或 SDXL-Lightning |
| 提示词遵循度最佳、托管服务 | GPT-Image / DALL-E 3（仍然在列）、Midjourney v7、Imagen 4 |
| 编辑工作流 | Flux.1-Kontext（2024 年 12 月）——原生支持图像 + 文本输入 |
| 研究、基线 | SD 1.5——虽然古老但研究透彻 |

## 交付产物

保存 `outputs/skill-sd-prompter.md`。该 skill 接收一个文本提示词 + 目标风格，输出：模型 + checkpoint、CFG scale、采样器、负向提示词、分辨率、可选的 ControlNet/IP-Adapter 组合，以及一份逐步骤的 QA 检查清单。

## 练习

1. **简单。** 用引导系数 `w ∈ {0, 1, 3, 7, 15}` 运行 `code/main.py`。记录每个类别的样本均值。在哪个 `w` 值时，类别均值开始偏离真实数据均值？
2. **中等。** 把玩具线性编码器换成一对带重建损失的 tanh-MLP 编码器/解码器。在新的潜在表示上重新训练扩散模型。样本质量有变化吗？
3. **困难。** 用 diffusers 搭建一次真正的 Stable Diffusion 推理：加载 `sdxl-base`，以 CFG=7 跑 30 步 Euler 采样并计时。然后切换到 `sdxl-turbo`，用 4 步、CFG=0。同一主题、不同质量——描述发生了什么变化以及原因。

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|-----------------|-----------------------|
| 第一阶段 | 「那个 VAE」 | 训练好的编码器/解码器对；把 512² 压缩到 64²。 |
| 第二阶段 | 「那个 U-Net」 | 运行在潜在空间上的扩散模型。 |
| CFG | 「引导系数」 | `(1+w)·ε_cond - w·ε_uncond`；调节条件化强度。 |
| 空 token | 「空提示词嵌入」 | 用于计算 `ε_uncond` 的无条件嵌入。 |
| 交叉注意力 | 「文本进来的通道」 | U-Net 的每个块把文本 token 作为 K 和 V 进行注意力计算。 |
| DiT | 「扩散 Transformer」 | 用在潜在 patch 上运行的 Transformer 取代 U-Net；扩展性更好。 |
| MMDiT | 「多模态 DiT」 | SD3 的架构：文本流和图像流做联合注意力。 |
| VAE 缩放因子 | 「魔法数字」 | 把潜在表示除以约 5.4，让扩散在单位方差空间中进行。 |

## 生产笔记：在 8GB 消费级 GPU 上运行 Flux-12B

参考的 Flux 集成方案，就是「我只有消费级 GPU，能上线吗？」这个问题的标准答案。诀窍就是生产推理文献中列出的那套三旋钮配方，应用到扩散 DiT 上：

1. **分段加载。** Flux 包含三组永远不需要同时驻留 VRAM 的网络：T5-XXL 文本编码器（fp32 下约 10 GB）、CLIP-L（很小）、12B 的 MMDiT，以及 VAE。先编码提示词，*删除*编码器，加载 DiT，去噪，*删除* DiT，加载 VAE，解码。8GB 消费级 GPU 一次只装得下一个阶段。
2. **用 bitsandbytes 做 4-bit 量化。** 对 T5 编码器和 DiT 都使用 `BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16)`。内存削减 8 倍，根据 Aritra 的基准测试（notebook 中有链接），文生图场景下的质量下降几乎不可察觉。
3. **CPU 卸载。** `pipe.enable_model_cpu_offload()` 会随着每次前向传播的推进，自动在 CPU 和 GPU 之间交换模块。延迟增加 10-20%，但让整个 pipeline 得以运行。

内存账目如下：`10 GB T5 / 8 = 1.25 GB`（量化后），`12 B 参数 × 0.5 字节 = 约 6 GB`（量化后的 DiT），再加上激活值。用 stas00 的话说，这是 TP=1 推理的极端形态——没有模型并行、最大化量化。生产环境你会在 H100 上跑 TP=2 或 TP=4；而对一台开发笔记本来说，这就是那份配方。

## 延伸阅读

- [Rombach et al. (2022). High-Resolution Image Synthesis with Latent Diffusion Models](https://arxiv.org/abs/2112.10752) —— Stable Diffusion。
- [Podell et al. (2023). SDXL: Improving Latent Diffusion Models for High-Resolution Image Synthesis](https://arxiv.org/abs/2307.01952) —— SDXL。
- [Peebles & Xie (2023). Scalable Diffusion Models with Transformers (DiT)](https://arxiv.org/abs/2212.09748) —— DiT。
- [Esser et al. (2024). Scaling Rectified Flow Transformers for High-Resolution Image Synthesis](https://arxiv.org/abs/2403.03206) —— SD3、MMDiT。
- [Ho & Salimans (2022). Classifier-Free Diffusion Guidance](https://arxiv.org/abs/2207.12598) —— CFG。
- [Labs (2024). Flux.1 — Black Forest Labs announcement](https://blackforestlabs.ai/announcing-black-forest-labs/) —— Flux.1 系列。
- [Hugging Face Diffusers docs](https://huggingface.co/docs/diffusers/index) —— 上述所有 checkpoint 的参考实现。
