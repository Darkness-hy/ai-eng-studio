# ControlNet、LoRA 与条件控制

> 仅靠文本是一种笨拙的控制信号。ControlNet 让你克隆一个预训练扩散模型，用深度图、姿态骨架、涂鸦或边缘图来引导它。LoRA 让你只训练 1000 万个参数，就能微调一个 20 亿参数的模型。两者合力，把 Stable Diffusion 从玩具变成了 2026 年每家创意机构都在交付的图像生产管线。

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 8 · 07 (Latent Diffusion), Phase 10 (LLMs from Scratch — for LoRA foundation)
**Time:** ~75 minutes

## 问题背景

像"一位穿红裙的女人在繁忙街道上遛狗"这样的提示词，并没有告诉模型狗在*哪里*、女人是*什么姿势*、街道的*视角*如何。文本只能锁定指定一张图像所需信息的大约 10%。其余部分都是视觉性的，无法用语言高效描述。

为每种信号（姿态、深度、Canny 边缘、分割）从零训练一个新的条件模型代价高得难以承受。你想要的是：保持 26 亿参数的 SDXL 主干冻结不动，挂上一个读取条件信号的小型旁路网络，让它微调主干的中间特征。这就是 ControlNet。

你还想教模型学会新概念（你的脸、你的产品、你的风格），而不必重新训练整个模型。你想要一个小 100 倍的增量。这就是 LoRA——插入现有注意力权重的低秩适配器（low-rank adapter）。

ControlNet + LoRA + 文本 = 2026 年从业者的工具箱。大多数生产级图像管线都会在 SDXL / SD3 / Flux 基座之上叠加 2-5 个 LoRA、1-3 个 ControlNet，外加一个 IP-Adapter。

## 核心概念

![ControlNet clones the encoder; LoRA adds low-rank deltas](../assets/controlnet-lora.svg)

### ControlNet（Zhang et al., 2023）

取一个预训练的 SD。*克隆* U-Net 的编码器一半。冻结原模型。训练克隆体接受一个额外的条件输入（边缘、深度、姿态）。再通过*零卷积（zero-convolution）*跳跃连接（初始化为零的 1×1 卷积——一开始是空操作，逐渐学出一个增量）把克隆体接回原模型的解码器一半。

```
SD U-Net decoder:   ... ← orig_enc_features + zero_conv(controlnet_enc(condition))
```

零卷积初始化意味着 ControlNet 一开始就是恒等映射——即使还没训练也不会造成破坏。在 100 万条（提示词、条件、图像）三元组上用标准扩散损失训练即可。

各模态的 ControlNet 以小型旁路模型的形式发布（SDXL 约 360M，SD 1.5 约 70M）。推理时可以组合使用：

```
features += weight_a * control_a(depth) + weight_b * control_b(pose)
```

### LoRA（Hu et al., 2021）

对模型中任意一个线性层 `W ∈ R^{d×d}`，冻结 `W`，加上一个低秩增量：

```
W' = W + ΔW,  ΔW = B @ A,  A ∈ R^{r×d},  B ∈ R^{d×r}
```

其中 `r << d`。注意力层通常用秩 4-16，重度微调用秩 64-128。新增参数量是 `2 · d · r` 而不是 `d²`。以 SDXL 注意力的 `d=640`、`r=16` 为例：每个适配器只需 2 万参数而不是 41 万——缩小 20 倍。放眼整个模型：一个 LoRA 通常只有 20-200MB，而基座模型是 5GB。

推理时可以对 LoRA 进行缩放：`W' = W + α · B @ A`。`α = 0.5-1.5` 是常规范围。多个 LoRA 可以加性叠加（但要注意它们之间会以非线性方式相互影响）。

### IP-Adapter（Ye et al., 2023）

一个极小的适配器，接受*图像*作为条件（与文本并行）。它用 CLIP 图像编码器生成图像 token，将其与文本 token 一起注入交叉注意力。每个基座模型约 20MB。让你无需 LoRA 就能实现"按这张参考图的风格生成图像"。

## 可组合性矩阵

| 工具 | 控制什么 | 体积 | 何时使用 |
|------|------------------|------|-------------|
| ControlNet | 空间结构（姿态、深度、边缘） | 70-360MB | 精确布局、构图 |
| LoRA | 风格、主体、概念 | 20-200MB | 个性化、风格化 |
| IP-Adapter | 来自参考图像的风格或主体 | 20MB | 文本无法描述的视觉效果 |
| Textual Inversion | 用一个新 token 表示单个概念 | 10KB | 遗留方案，基本被 LoRA 取代 |
| DreamBooth | 针对某个主体的全量微调 | 2-5GB | 强身份一致性，算力开销大 |
| T2I-Adapter | 更轻量的 ControlNet 替代品 | 70MB | 边缘设备、推理预算受限 |

ControlNet ≈ 空间控制。LoRA ≈ 语义控制。两者并用。

## 从零实现

`code/main.py` 在一维上模拟这两种机制：

1. **LoRA。** 一个预训练线性层 `W`。冻结它。训练一个低秩的 `B @ A`，使 `W + BA` 匹配一个目标线性层。证明 `r = 1` 就足以完美学到一个秩 1 的修正量。

2. **ControlNet-lite。** 一个"冻结基座"预测器，加一个读取额外信号的"旁路网络"。旁路网络的输出由一个初始化为零的可学习标量门控（这是我们版本的零卷积）。训练并观察门控值逐步爬升。

### 第 1 步：LoRA 数学

```python
def lora(W, A, B, x, alpha=1.0):
    # W is frozen; A, B are the trainable low-rank factors.
    return [W[i][j] * x[j] for i, j in ...] + alpha * (B @ (A @ x))
```

### 第 2 步：零初始化旁路网络

```python
side_out = control_net(x, condition)
gated = gate * side_out  # gate initialized to 0
h = base(x) + gated
```

第 0 步时输出与基座完全相同。训练早期 `gate` 更新得很缓慢——不会出现灾难性漂移。

## 常见陷阱

- **过度放大 LoRA。** `α = 2` 或 `α = 3` 是常见的"再强一点"的偷懒做法，结果是过度风格化甚至崩坏的输出。保持 `α ≤ 1.5`。
- **ControlNet 权重冲突。** 姿态 ControlNet 用权重 1.0、深度 ControlNet 也用权重 1.0，通常会用力过猛。权重之和 ≈ 1.0 是安全的默认值。
- **LoRA 用错基座。** SDXL 的 LoRA 在 SD 1.5 上会静默失效，因为注意力维度不匹配。Diffusers 0.30+ 会给出警告。
- **Textual Inversion 漂移。** 在一个 checkpoint 上训练的 token 换到另一个 checkpoint 上会严重漂移。LoRA 的可移植性更好。
- **LoRA 权重合并与存储。** 你可以把 LoRA 烘焙进基座模型权重以加快推理（省去运行时加法），但会失去运行时调节 `α` 的能力。两个版本都保留。

## 生产实践

| 目标 | 2026 年管线 |
|------|---------------|
| 复现某品牌的美术风格 | 在约 30 张精选图像上以秩 32 训练 LoRA |
| 把我的脸放进生成图像 | DreamBooth 或 LoRA + IP-Adapter-FaceID |
| 指定姿态 + 提示词 | ControlNet-Openpose + SDXL + 文本 |
| 深度感知构图 | ControlNet-Depth + SD3 |
| 参考图 + 提示词 | IP-Adapter + 文本 |
| 精确布局 | ControlNet-Scribble 或 ControlNet-Canny |
| 背景替换 | ControlNet-Seg + 图像修复（第 09 课） |
| 快速单步风格化 | LCM-LoRA on SDXL-Turbo |

## 交付产物

保存 `outputs/skill-sd-toolkit-composer.md`。该技能接收一个任务（输入素材：提示词、可选参考图、可选姿态、可选深度图、可选涂鸦），输出工具栈、各项权重，以及一套可复现的随机种子协议。

## 练习

1. **简单。** 在 `code/main.py` 中，把 LoRA 的秩 `r` 从 1 调到 4。在哪个秩上 LoRA 能精确匹配一个秩 2 的目标增量？
2. **中等。** 针对两个目标变换分别训练两个 LoRA。同时加载它们，展示其加性交互。这种交互在什么情况下会破坏线性？
3. **困难。** 用 diffusers 叠加：SDXL-base + Canny-ControlNet（权重 0.8）+ 一个风格 LoRA（α 0.8）+ IP-Adapter（权重 0.6）。在调整各层权重时，测量 FID 与提示词遵循度之间的权衡。

## 关键术语

| 术语 | 大家怎么说 | 实际含义 |
|------|-----------------|-----------------------|
| ControlNet | "空间控制" | 克隆编码器 + 零卷积跳连；读取一张条件图像。 |
| 零卷积（Zero convolution） | "一开始是恒等" | 初始化为零的 1×1 卷积；ControlNet 起步时是空操作。 |
| LoRA | "低秩适配器" | `W + B @ A`，`r << d`；参数比全量微调少 100 倍。 |
| 秩 r | "那个旋钮" | LoRA 的压缩程度；常用 4-16，重度个性化用 64+。 |
| α | "LoRA 强度" | 对 LoRA 增量的运行时缩放。 |
| IP-Adapter | "参考图像" | 通过 CLIP 图像 token 实现的小型图像条件适配器。 |
| DreamBooth | "主体全量微调" | 在某主体的约 30 张图像上训练整个模型。 |
| Textual Inversion | "新 token" | 只学习一个新的词嵌入；遗留方案，基本已被取代。 |

## 生产笔记：LoRA 热切换、ControlNet 通道、多租户服务

一个真实的文生图 SaaS 要在同一个基座 checkpoint 上服务数百个 LoRA 和十几个 ControlNet。这个服务问题和 LLM 多租户非常相似（生产文献中 LLM 的对应方案是 continuous batching 与 LoRAX / S-LoRA）：

- **热切换 LoRA，不要合并。** 把 `W' = W + α·B·A` 合并进基座可以让每步推理快约 3-5%，但会把 `α` 和基座都锁死。把 LoRA 以秩 r 增量的形式常驻 VRAM；diffusers 提供 `pipe.load_lora_weights()` + `pipe.set_adapters([...], adapter_weights=[...])` 实现按请求激活。切换成本就是那 `2 · d · r · num_layers` 个权重——MB 量级，亚秒级完成。
- **把 ControlNet 看作第二条注意力通道。** 克隆的编码器与基座并行运行。两个权重各为 1.0 的 ControlNet = 每步多两次前向传播，而不是一次合并的传播。批大小余量会按二次方下降。每个激活的 ControlNet 按约 1.5 倍单步开销做预算。
- **LoRA 同样可以量化。** 如果你量化了基座（见第 07 课，8GB 上跑 Flux），LoRA 增量也能干净地量化到 8-bit 或 4-bit。QLoRA 式加载让你在 4-bit Flux 基座上叠加 5-10 个 LoRA 而不会撑爆显存。

Flux 专属提示：Niels 的 Flux-on-8GB notebook 把基座量化到 4-bit；在这个量化基座上以 `weight_name="pytorch_lora_weights.safetensors"` 叠加一个风格 LoRA（`pipe.load_lora_weights("user/style-lora")`）依然可行。这就是 2026 年大多数 SaaS 创意机构交付的配方。

## 延伸阅读

- [Zhang, Rao, Agrawala (2023). Adding Conditional Control to Text-to-Image Diffusion Models](https://arxiv.org/abs/2302.05543) — ControlNet。
- [Hu et al. (2021). LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685) — LoRA（最初面向 LLM；可移植到扩散模型）。
- [Ye et al. (2023). IP-Adapter: Text Compatible Image Prompt Adapter](https://arxiv.org/abs/2308.06721) — IP-Adapter。
- [Mou et al. (2023). T2I-Adapter: Learning Adapters to Dig Out More Controllable Ability](https://arxiv.org/abs/2302.08453) — ControlNet 的更轻量替代品。
- [Ruiz et al. (2023). DreamBooth: Fine Tuning Text-to-Image Diffusion Models for Subject-Driven Generation](https://arxiv.org/abs/2208.12242) — DreamBooth。
- [HuggingFace Diffusers — ControlNet / LoRA / IP-Adapter docs](https://huggingface.co/docs/diffusers/training/controlnet) — 参考管线。
