# 扩散模型 —— 从零实现 DDPM

> Ho、Jain、Abbeel（2020）给了这个领域一个再也放不下的配方：用一千个小步骤逐步以噪声摧毁数据，训练一个神经网络去预测噪声，推理时把过程逆转回来。如今所有主流的图像、视频、3D 和音乐模型都跑在这个循环上，顶多再叠加流匹配（flow matching）或一致性模型（consistency）这类技巧。

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 3 · 02 (Backprop), Phase 8 · 02 (VAE)
**Time:** ~75 minutes

## 问题背景

你想要一个能从 `p_data(x)` 采样的采样器。GAN 玩的是经常发散的极小极大（minimax）博弈；VAE 的高斯解码器生成的样本则很模糊。你真正想要的训练目标是：(a) 单一且稳定的损失（没有鞍点、没有 minimax），(b) 是 `log p(x)` 的下界（这样你能得到似然），(c) 样本质量能匹敌 SOTA。

Sohl-Dickstein 等人（2015）给出了理论上的答案：定义一条逐步添加高斯噪声的马尔可夫链 `q(x_t | x_{t-1})`，再训练一条逆向链 `p_θ(x_{t-1} | x_t)` 来去噪。Ho、Jain、Abbeel（2020）证明这个损失可以简化成一行——预测噪声——并把数学推导收拾干净。2020 年它还只是个新奇玩意；2021 年它生成了最先进的样本；2022 年它变成了 Stable Diffusion；2026 年它已是整个领域的基座。

## 核心概念

![DDPM：前向加噪，逆向去噪](../assets/ddpm.svg)

**前向过程 `q`。** 用 `T` 个小步骤添加高斯噪声。其闭式解——也是数学上之所以可解的原因——在于累积之后的一步仍是高斯分布：

```
q(x_t | x_0) = N( sqrt(α̅_t) · x_0,  (1 - α̅_t) · I )
```

其中 `α̅_t = ∏_{s=1..t} (1 - β_s)`，由 `β_t` 调度表决定。把 `β_t` 设为在 T=1000 步上从 1e-4 线性增加到 0.02，那么 `x_T` 就近似服从 `N(0, I)`。

**逆向过程 `p_θ`。** 学习一个神经网络 `ε_θ(x_t, t)`，预测当初加入的噪声。给定 `x_t`，按下式去噪：

```
x_{t-1} = (1 / sqrt(α_t)) · ( x_t - (β_t / sqrt(1 - α̅_t)) · ε_θ(x_t, t) )  +  σ_t · z
```

其中 `σ_t` 取 `sqrt(β_t)` 或一个学习得到的方差。这个表达式看着丑，但只是代数运算——基于后验分布 `q(x_{t-1} | x_t, x_0)` 解出 `x_{t-1}`，再把 `x_0` 替换成由噪声预测得到的估计值。

**训练损失。**

```
L_simple = E_{x_0, t, ε} [ || ε - ε_θ( sqrt(α̅_t) · x_0 + sqrt(1 - α̅_t) · ε,  t ) ||² ]
```

从数据中采样 `x_0`，随机选一个 `t`，采样 `ε ~ N(0, I)`，通过闭式解一步算出带噪的 `x_t`，再对噪声做回归。只有一个损失，没有 minimax，没有 KL，没有重参数化技巧。

**采样。** 从 `x_T ~ N(0, I)` 出发，从 `t = T` 到 `1` 迭代执行逆向步骤。完事。

## 为什么有效

三个直觉：

1. **去噪容易，生成困难。** 在 `t=T` 时数据是纯噪声——网络面对的是一个平凡问题。在 `t=0` 时网络只需修整几个像素。在中间的 `t` 上问题确实难，但来自所有噪声水平的梯度都流经同一组权重。

2. **变相的得分匹配（score matching）。** Vincent（2011）证明了预测噪声等价于估计 `∇_x log q(x_t | x_0)`，即*得分（score）*。逆向 SDE 利用这个得分沿密度梯度向上爬——一场被引导着走向高概率区域的随机游走。

3. **ELBO 退化为简单的 MSE。** 完整的变分下界对每个时间步都有一个 KL 项。在 DDPM 的参数化下，这些 KL 项简化为带特定系数的噪声预测 MSE；Ho 干脆把系数扔掉（称之为 "simple" 损失），质量反而*提升*了。

```figure
diffusion-denoise
```

## 从零实现

`code/main.py` 实现了一个一维 DDPM。数据是双峰混合分布。"网络"是一个接收 `(x_t, t)` 并输出预测噪声的小型 MLP。训练就是那一行损失。采样则迭代逆向链。

### 第 1 步：前向调度表（闭式解）

```python
betas = [1e-4 + (0.02 - 1e-4) * t / (T - 1) for t in range(T)]
alphas = [1 - b for b in betas]
alpha_bars = []
cum = 1.0
for a in alphas:
    cum *= a
    alpha_bars.append(cum)
```

### 第 2 步：一步采出 `x_t`

```python
def forward_sample(x0, t, alpha_bars, rng):
    a_bar = alpha_bars[t]
    eps = rng.gauss(0, 1)
    x_t = math.sqrt(a_bar) * x0 + math.sqrt(1 - a_bar) * eps
    return x_t, eps
```

### 第 3 步：单次训练步

```python
def train_step(x0, model, alpha_bars, rng):
    t = rng.randrange(T)
    x_t, eps = forward_sample(x0, t, alpha_bars, rng)
    eps_hat = model_forward(model, x_t, t)
    loss = (eps - eps_hat) ** 2
    return loss, gradient_step(model, ...)
```

### 第 4 步：逆向采样

```python
def sample(model, alpha_bars, T, rng):
    x = rng.gauss(0, 1)
    for t in range(T - 1, -1, -1):
        eps_hat = model_forward(model, x, t)
        beta_t = 1 - alphas[t]
        x = (x - beta_t / math.sqrt(1 - alpha_bars[t]) * eps_hat) / math.sqrt(alphas[t])
        if t > 0:
            x += math.sqrt(beta_t) * rng.gauss(0, 1)
    return x
```

对于一个使用 40 个时间步、24 个隐藏单元 MLP 的一维问题，约 200 个 epoch 就能学会这个双峰混合分布。

## 时间步条件注入

网络需要知道自己正在为哪个时间步去噪。两种标准做法：

- **正弦嵌入（sinusoidal embedding）。** 类似 Transformer 的位置编码。`embed(t) = [sin(t/ω_0), cos(t/ω_0), sin(t/ω_1), ...]`。过一个 MLP，再广播注入到网络中。
- **FiLM / group-norm 条件注入。** 在每个 block 把嵌入投影为逐通道的缩放/偏置（FiLM）。

我们的玩具代码用正弦嵌入 → 拼接（concat）。生产级 U-Net 用 FiLM。

## 常见陷阱

- **调度表影响巨大。** 线性 `β` 是 DDPM 的默认设置，但余弦调度（Nichol & Dhariwal, 2021）在相同算力下能拿到更好的 FID。质量遇到瓶颈时不妨换调度表。
- **时间步嵌入很脆弱。** 把原始 `t` 当浮点数直接传入，在玩具级一维问题上可行，在图像上会失效；务必使用正经的嵌入。
- **V-prediction 与 ε-prediction 之争。** 在极端区间（t 非常小或非常大）`ε` 的信噪比很差。V-prediction（`v = α·ε - σ·x`）更稳定；SDXL、SD3 和 Flux 都在用它。
- **无分类器引导（classifier-free guidance）。** 推理时同时计算条件和无条件的 `ε`，然后 `ε_cfg = (1 + w) · ε_cond - w · ε_uncond`，`w ≈ 3-7`。第 08 课会讲。
- **1000 步太多了。** 生产环境用 DDIM（20-50 步）、DPM-Solver（10-20 步）或蒸馏（1-4 步）。见第 12 课。

## 生产实践

| 角色 | 2026 年的典型技术栈 |
|------|-----------------------|
| 图像像素空间扩散（小型、玩具级） | DDPM + U-Net |
| 图像潜空间扩散 | VAE 编码器 + U-Net 或 DiT（第 07 课） |
| 视频潜空间扩散 | 时空 DiT（Sora、Veo、WAN） |
| 音频潜空间扩散 | Encodec + 扩散 Transformer |
| 科学领域（分子、蛋白质、物理） | 等变扩散（EDM、RFdiffusion、AlphaFold3） |

扩散是通用的生成式骨干。流匹配（第 13 课）是 2024-2026 年的竞争者，在相同质量下通常以推理速度取胜。

## 交付产物

保存 `outputs/skill-diffusion-trainer.md`。该 Skill 接收一个数据集和算力预算，输出：调度表（线性/余弦/sigmoid）、预测目标（ε/v/x）、步数、引导系数、采样器家族，以及一套评估方案。

## 练习

1. **简单。** 在 `code/main.py` 中把 T 从 40 改成 10。样本质量（输出的可视化直方图）如何退化？T 降到多少时双峰结构会崩塌？
2. **中等。** 从 ε-prediction 切换到 v-prediction。重新推导逆向步骤。比较最终样本质量。
3. **困难。** 加入无分类器引导。以类别标签 `c ∈ {0, 1}` 为条件，训练时 10% 的概率丢弃它，采样时使用 `ε = (1+w)·ε_cond - w·ε_uncond`。测量在 `w = 0, 1, 3, 7` 下命中条件模式的比例。

## 关键术语

| 术语 | 大家怎么说 | 实际含义 |
|------|-----------------|-----------------------|
| 前向过程 | "加噪" | 摧毁数据的固定马尔可夫链 `q(x_t \| x_{t-1})`。 |
| 逆向过程 | "去噪" | 重建数据的学习链 `p_θ(x_{t-1} \| x_t)`。 |
| β 调度表 | "噪声阶梯" | 每步的方差；线性、余弦或 sigmoid。 |
| α̅ | "Alpha bar" | 累积乘积 `∏(1 - β)`；提供从 `x_0` 到 `x_t` 的闭式解。 |
| Simple 损失 | "对噪声做 MSE" | `\|\|ε - ε_θ(x_t, t)\|\|²`；所有变分推导最终都坍缩到这个式子。 |
| ε-prediction | "预测噪声" | 输出是当初加入的噪声；标准 DDPM 做法。 |
| V-prediction | "预测速度" | 输出是 `α·ε - σ·x`；在各个 t 上数值条件更好。 |
| DDPM | "那篇论文" | Ho et al. 2020；线性 β，1000 步，U-Net。 |
| DDIM | "确定性采样器" | 非马尔可夫采样器，20-50 步，训练目标不变。 |
| 无分类器引导 | "CFG" | 混合条件与无条件噪声预测，以放大条件信号。 |

## 生产备注：扩散推理是一个步数问题

DDPM 论文跑的是 T=1000 步逆向采样。没人会把这个原样部署上线。每个真实推理栈都会从三种策略中选择——每一种都能清晰对应到生产视角下"延迟从哪来"的拆解：

1. **更快的采样器，模型不变。** DDIM（20-50 步）、DPM-Solver++（10-20 步）、UniPC（8-16 步）。直接替换逆向循环即可；训练好的 `ε_θ` 权重原封不动。延迟降低 20-50 倍。
2. **蒸馏。** 训练学生模型用更少的步数匹配教师模型：Progressive Distillation（2 → 1）、Consistency Models（任意 → 1-4）、LCM、SDXL-Turbo、SD3-Turbo。延迟再降 5-10 倍，但需要重新训练。
3. **缓存与编译。** `torch.compile(unet, mode="reduce-overhead")`、TensorRT-LLM 的扩散后端、`xformers`/SDPA 注意力、bf16 权重。每步延迟约降 2 倍，且可以与 (1)、(2) 叠加。

对生产级扩散服务器来说，预算讨论和生产文献中对 LLM 的描述是一样的：延迟是 `num_steps × step_cost + VAE_decode`，吞吐量是 `batch_size × (num_steps × step_cost)^-1`。TTFT 很小（只需一步）；TPOT 的对应量则是整个响应时间，因为在用户视角图像生成是"一次性"完成的。

## 延伸阅读

- [Sohl-Dickstein et al. (2015). Deep Unsupervised Learning using Nonequilibrium Thermodynamics](https://arxiv.org/abs/1503.03585) —— 扩散模型的开山之作，超前于时代。
- [Ho, Jain, Abbeel (2020). Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2006.11239) —— DDPM。
- [Song, Meng, Ermon (2021). Denoising Diffusion Implicit Models](https://arxiv.org/abs/2010.02502) —— DDIM，更少的步数。
- [Nichol & Dhariwal (2021). Improved DDPM](https://arxiv.org/abs/2102.09672) —— 余弦调度、学习方差。
- [Dhariwal & Nichol (2021). Diffusion Models Beat GANs on Image Synthesis](https://arxiv.org/abs/2105.05233) —— 分类器引导。
- [Ho & Salimans (2022). Classifier-Free Diffusion Guidance](https://arxiv.org/abs/2207.12598) —— CFG。
- [Karras et al. (2022). Elucidating the Design Space of Diffusion-Based Generative Models (EDM)](https://arxiv.org/abs/2206.00364) —— 统一记号、最干净的配方。
