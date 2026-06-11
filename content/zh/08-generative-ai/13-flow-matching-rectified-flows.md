# 流匹配与整流流（Flow Matching & Rectified Flows）

> 扩散模型需要 20-50 步采样，因为它们沿着一条从噪声到数据的弯曲路径行走。流匹配（Flow Matching，Lipman et al., 2023）和整流流（Rectified Flow，Liu et al., 2022）训练的是笔直的路径。路径越直，步数越少，推理就越快。Stable Diffusion 3、Flux.1 和 AudioCraft 2 都在 2024 年切换到了流匹配。

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 8 · 06 (DDPM), Phase 1 · Calculus
**Time:** ~45 minutes

## 问题背景

DDPM 的逆向过程是一场 1000 步的随机游走，从 `N(0, I)` 走回数据分布。DDIM 把它压缩到 20-50 步确定性采样。你想要更少的步数——最好只要一步。瓶颈在于：求解逆向过程的 ODE 是刚性的，路径是弯曲的。

如果能把模型训练成从噪声到数据的路径是一条*直线*，那么从 `t=1` 到 `t=0` 只需一步 Euler 积分就够了。流匹配正是直接构造了这一点：定义从 `x_1 ∼ N(0, I)` 到 `x_0 ∼ data` 的直线插值，训练一个向量场 `v_θ(x, t)` 去拟合它的时间导数，推理时做数值积分。

整流流（Liu 2022）更进一步：用 reflow 过程迭代地把路径拉直，得到一个越来越接近线性的 ODE。经过两轮 reflow 迭代后，2 步采样器就能达到 50 步 DDPM 的质量。

## 核心概念

![Flow matching: straight-line interpolation between noise and data](../assets/flow-matching.svg)

### 直线流

定义：

```
x_t = t · x_1 + (1 - t) · x_0,   t ∈ [0, 1]
```

其中 `x_0 ~ data`，`x_1 ~ N(0, I)`。沿这条直线的时间导数是常数：

```
dx_t / dt = x_1 - x_0
```

定义一个神经网络向量场 `v_θ(x_t, t)`，训练它去拟合这个导数：

```
L = E_{x_0, x_1, t} || v_θ(x_t, t) - (x_1 - x_0) ||²
```

这就是**条件流匹配（conditional flow matching）**损失（Lipman 2023）。训练是免模拟（simulation-free）的：你完全不需要展开 ODE，只需采样 `(x_0, x_1, t)` 然后做回归。

### 采样

推理时，沿时间*反向*积分学到的向量场：

```
x_{t-Δt} = x_t - Δt · v_θ(x_t, t)
```

从 `x_1 ~ N(0, I)` 出发，用 Euler 步一路走到 `t=0`。

### 整流流（Liu 2022）

直线流是可行的，但学到的路径*并非真正笔直*——它们会弯曲，因为多个 `x_0` 可能映射到同一个 `x_1`。整流流的 reflow 步骤如下：

1. 用随机配对训练流模型 v_1。
2. 从 `x_1` 出发积分 v_1 直到落点 `x_0`，由此采样 N 个配对 `(x_1, x_0)`。
3. 在这些配对样本上训练 v_2。由于这些配对现在是"经 ODE 匹配的"，它们之间的直线插值确实更平直了。
4. 重复以上步骤。

实践中 2 轮 reflow 迭代就能接近线性，从而支持 2-4 步推理。SDXL-Turbo、SD3-Turbo、LCM 都是从流匹配蒸馏出来的模型。

### 为什么它在 2024 年的图像生成中胜出

三个原因：

1. **免模拟训练**——训练期间无需展开 ODE，实现起来非常简单。
2. **更好的损失几何**——直线路径有一致的信噪比，而 DDPM 的 ε 损失在调度两端的 SNR 很差。
3. **更快的推理**——4-8 步即可达到 SDXL-Turbo 的质量；配合一致性蒸馏只需 1 步。

## 流匹配 vs DDPM——精确的对应关系

采用高斯条件路径的流匹配，本质上就是*带特定噪声调度*的扩散。选取 `x_t = α(t) x_0 + σ(t) x_1` 这样的调度，流匹配就还原成 Stratonovich 形式重写的扩散，其中 `v = α'·x_0 - σ'·x_1`。对于高斯路径，二者在代数上等价。

流匹配新增的贡献在于：目标的*清晰性*（一个朴素的速度量）、更干净的损失函数，以及尝试非高斯插值的自由度。

## 从零实现

`code/main.py` 在一个双峰高斯混合分布上实现 1 维流匹配。向量场 `v_θ(x, t)` 是一个用直线目标训练的小型 MLP。推理时分别用 1、2、4、20 步 Euler 积分，并对比样本质量。

### 第 1 步：训练损失

```python
def train_step(x0, net, rng, lr):
    x1 = rng.gauss(0, 1)
    t = rng.random()
    x_t = t * x1 + (1 - t) * x0
    target = x1 - x0
    pred = net_forward(x_t, t)
    loss = (pred - target) ** 2
    # backprop + update
```

### 第 2 步：多步推理

```python
def sample(net, num_steps):
    x = rng.gauss(0, 1)
    for i in range(num_steps):
        t = 1.0 - i / num_steps
        dt = 1.0 / num_steps
        x -= dt * net_forward(x, t)
    return x
```

### 第 3 步：对比不同步数

预期 4 步采样器就已经能追平 20 步的质量——这对延迟来说意义重大。

## 常见陷阱

- **时间参数化。**流匹配使用 `t ∈ [0, 1]`，`t=0` 对应数据、`t=1` 对应噪声。DDPM 使用 `t ∈ [0, T]`，`t=0` 对应数据、`t=T` 对应噪声。方向相同，尺度不同。论文里这一点经常写错。
- **调度选择。**整流流的直线是"标准的"流匹配调度，但你也可以用余弦或 logit-normal 的 t 采样（SD3 就是这么做的）来获得更好的尺度覆盖。
- **Reflow 的代价。**为 reflow 生成配对数据集，每个样本都要跑一次完整推理。只有当你确实需要 1-2 步推理时才做 reflow。
- **无分类器引导（classifier-free guidance）依然适用。**只需在线性组合中把 ε 换成 v：`v_cfg = (1+w) v_cond - w v_uncond`。

## 生产实践

| 使用场景 | 2026 技术栈 |
|----------|-----------|
| 文生图，追求最佳质量 | 流匹配：SD3、Flux.1-dev |
| 文生图，1-4 步 | 蒸馏后的流匹配：Flux.1-schnell、SD3-Turbo、SDXL-Turbo |
| 实时推理 | 基于流匹配底模的一致性蒸馏（LCM、PCM） |
| 音频生成 | 流匹配：Stable Audio 2.5、AudioCraft 2 |
| 视频生成 | 流匹配与扩散混合（Sora、Veo、Stable Video） |
| 科学/物理（粒子轨迹、分子） | 流匹配 + 等变向量场 |

在 2025-2026 年，只要论文里说"比扩散更快"，几乎都是流匹配 + 蒸馏。

## 交付产物

保存 `outputs/skill-fm-tuner.md`。这个 Skill 接收一个扩散风格的模型规格，并将其转换为流匹配训练配置：调度选择、时间采样分布（uniform / logit-normal）、优化器、reflow 计划、目标步数、评估方案。

## 练习

1. **简单。**运行 `code/main.py`，对比 1 步与 20 步采样相对真实数据分布的 MSE。
2. **中等。**把均匀的 `t` 采样换成 logit-normal（把采样集中在中间的 t 区域）。模型质量有提升吗？
3. **困难。**实现一轮 reflow 迭代：通过积分第一个模型生成配对的 (x_0, x_1)，在配对数据上训练第二个模型，并对比 1 步采样质量。

## 关键术语

| 术语 | 大家怎么说 | 实际含义 |
|------|-----------------|-----------------------|
| 流匹配 | "直线扩散" | 训练 `v_θ(x, t)` 沿插值路径拟合 `x_1 - x_0`。 |
| 整流流 | "Reflow" | 迭代地把学到的流拉直的过程。 |
| 速度场 | "v_θ" | 模型的输出——`x_t` 应当移动的方向。 |
| 直线插值 | "路径" | `x_t = (1-t)·x_0 + t·x_1`；目标导数极其简单。 |
| Euler 采样器 | "一阶 ODE 求解器" | 最简单的积分器；路径笔直时效果很好。 |
| Logit-normal t | "SD3 采样" | 把 `t` 的采样集中在梯度最强的中间值附近。 |
| 一致性蒸馏 | "1 步采样器" | 训练学生模型把任意 `x_t` 直接映射到 `x_0`。 |
| 速度版 CFG | "v-CFG" | `v_cfg = (1+w) v_cond - w v_uncond`；同一个技巧，换了个变量。 |

## 生产笔记：Flux.1-schnell 是流匹配速度的极致

流匹配在生产中的标志性胜利是 Flux.1-schnell——一个经过流匹配训练的 DiT，被蒸馏到 1-4 步推理，同时保持 Flux-dev 级别的质量。Niels 的 "Run Flux on an 8GB machine" notebook 是参考部署方案：T5 + CLIP 编码，量化后的 MMDiT 去噪（schnell 用 4 步，dev 用 50 步），VAE 解码。成本账目如下：

| 变体 | 步数 | L4 上 1024² 的延迟 | 总 FLOPs（相对值） |
|---------|-------|------------------------|------------------------|
| Flux.1-dev（原始） | 50 | ~15 s | 1.0× |
| Flux.1-schnell | 4 | ~1.2 s | 0.08×（快 12 倍） |
| SDXL-base | 30 | ~4 s | 0.25× |
| SDXL-Lightning 2 步 | 2 | ~0.3 s | 0.03× |

生产法则：**流匹配底模 + 蒸馏 = 2026 年快速文生图的默认方案。**每家主要厂商都在出货这个组合：SD3-Turbo（SD3 + 流匹配 + 蒸馏）、Flux-schnell（Flux-dev + 整流流拉直）、CogView-4-Flash。纯扩散底模只剩遗留 checkpoint 还在用。

## 延伸阅读

- [Liu, Gong, Liu (2022). Flow Straight and Fast: Learning to Generate and Transfer Data with Rectified Flow](https://arxiv.org/abs/2209.03003) — 整流流。
- [Lipman et al. (2023). Flow Matching for Generative Modeling](https://arxiv.org/abs/2210.02747) — 流匹配。
- [Esser et al. (2024). Scaling Rectified Flow Transformers for High-Resolution Image Synthesis](https://arxiv.org/abs/2403.03206) — SD3，大规模整流流。
- [Albergo, Vanden-Eijnden (2023). Stochastic Interpolants](https://arxiv.org/abs/2303.08797) — 同时涵盖流匹配与扩散的一般性框架。
- [Song et al. (2023). Consistency Models](https://arxiv.org/abs/2303.01469) — 扩散/流模型的 1 步蒸馏。
- [Sauer et al. (2023). Adversarial Diffusion Distillation (SDXL-Turbo)](https://arxiv.org/abs/2311.17042) — turbo 变体。
- [Black Forest Labs (2024). Flux.1 models](https://blackforestlabs.ai/announcing-black-forest-labs/) — 生产环境中的流匹配。
