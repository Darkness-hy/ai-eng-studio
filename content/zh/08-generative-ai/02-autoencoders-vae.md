# 自编码器与变分自编码器（VAE）

> 普通自编码器只会先压缩再重建。它在记忆，而不会生成。加上一个技巧——强迫编码看起来像高斯分布——你就得到了一个采样器。这个技巧就是 `z = μ + σ·ε` 的重参数化，也正是 2026 年你用到的每一个潜在扩散和流匹配图像模型，输入端都带着一个 VAE 的原因。

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 3 · 02 (Backprop), Phase 3 · 07 (CNNs), Phase 8 · 01 (Taxonomy)
**Time:** ~75 minutes

## 问题背景

把一张 784 像素的 MNIST 数字图压缩成 16 个数的编码，再重建出来。普通自编码器能在重建 MSE 上拿满分，但编码空间是一团坑坑洼洼的烂泥。在编码空间里随便选一个点去解码，得到的只是噪声。它没有采样器，只是一个打扮成生成模型的压缩模型。

你真正想要的是：(a) 编码空间是一个干净、平滑、可以采样的分布——比如各向同性高斯 `N(0, I)`；(b) 解码任意一个样本都能得到一个像样的数字；(c) 编码器和解码器仍然有良好的压缩能力。三个目标，一个架构，一个损失。

Kingma 2013 年的 VAE 是这样解决的：训练编码器输出一个*分布* `q(z|x) = N(μ(x), σ(x)²)`，用一个 KL 惩罚项把这个分布拉向先验 `N(0, I)`，然后在解码前从 `q(z|x)` 中采样 `z`。推理时，扔掉编码器，采样 `z ~ N(0, I)`，直接解码。正是 KL 惩罚迫使编码空间变得有结构。

到了 2026 年，VAE 已经很少单独上线——在原始图像质量上它早被扩散模型甩开——但它是每一个潜在扩散模型（SD 1/2/XL/3、Flux、AudioCraft）的首选编码器。学会 VAE，你就学会了你所用的每一条图像流水线中那看不见的第一层。

## 核心概念

![Autoencoder vs VAE: the reparameterization trick](../assets/vae.svg)

**自编码器（Autoencoder）。** `z = encoder(x)`，`x̂ = decoder(z)`，损失 = `||x - x̂||²`。编码空间没有结构。

**VAE 编码器。** 输出两个向量：`μ(x)` 和 `log σ²(x)`。它们定义了 `q(z|x) = N(μ, diag(σ²))`。

**重参数化技巧（Reparameterization trick）。** 从 `q(z|x)` 中采样是不可微的。把采样改写为 `z = μ + σ·ε`，其中 `ε ~ N(0, I)`。现在 `z` 是 `(μ, σ)` 的确定性函数加上一份与参数无关的噪声——梯度可以流过 `μ` 和 `σ`。

**损失。** 证据下界（Evidence Lower BOund，ELBO），由两项组成：

```
loss = reconstruction + β · KL[q(z|x) || N(0, I)]
     = ||x - x̂||²  + β · Σ_i ( σ_i² + μ_i² - log σ_i² - 1 ) / 2
```

重建项把 `x̂` 推向 `x`。KL 项把 `q(z|x)` 推向先验。两者相互制衡。β 小（<1）= 样本更锐利，编码空间偏离高斯。β 大（>1）= 编码空间更干净，样本更模糊。β-VAE（Higgins 2017）让这个旋钮声名大噪，也开启了解耦表示（disentanglement）研究。

**采样。** 推理时：抽取 `z ~ N(0, I)`，过一遍解码器。一次前向传播——不像扩散模型那样需要迭代采样。

```figure
vae-latent-grid
```

## 从零实现

`code/main.py` 实现了一个不依赖 numpy 和 torch 的微型 VAE。输入是 8 维合成数据，从一个 8 维双分量高斯混合分布中抽取。编码器和解码器都是单隐藏层 MLP。我们手写了 tanh 激活、前向传播、损失以及反向传播。不是生产代码——纯为教学。

### 第 1 步：编码器前向

```python
def encode(x, enc):
    h = tanh(add(matmul(enc["W1"], x), enc["b1"]))
    mu = add(matmul(enc["W_mu"], h), enc["b_mu"])
    log_sigma2 = add(matmul(enc["W_sig"], h), enc["b_sig"])
    return mu, log_sigma2
```

输出 `log σ²` 而非 `σ`，这样网络输出不受约束（对 σ 用 softplus 是个坑——σ ≈ 0 时梯度会消失）。

### 第 2 步：重参数化并解码

```python
def reparameterize(mu, log_sigma2, rng):
    eps = [rng.gauss(0, 1) for _ in mu]
    sigma = [math.exp(0.5 * lv) for lv in log_sigma2]
    return [m + s * e for m, s, e in zip(mu, sigma, eps)]

def decode(z, dec):
    h = tanh(add(matmul(dec["W1"], z), dec["b1"]))
    return add(matmul(dec["W_out"], h), dec["b_out"])
```

### 第 3 步：ELBO

```python
def elbo(x, x_hat, mu, log_sigma2, beta=1.0):
    recon = sum((a - b) ** 2 for a, b in zip(x, x_hat))
    kl = 0.5 * sum(math.exp(lv) + m * m - lv - 1 for m, lv in zip(mu, log_sigma2))
    return recon + beta * kl, recon, kl
```

因为两边都是高斯分布，KL 有精确的闭式解。不要做数值积分。2026 年了还有人在上线的代码里用蒙特卡洛估计 KL——平白慢 3 倍，毫无理由。

### 第 4 步：生成

```python
def sample(dec, z_dim, rng):
    z = [rng.gauss(0, 1) for _ in range(z_dim)]
    return decode(z, dec)
```

这就是生成模型本身。五行代码。

## 常见陷阱

- **后验坍缩（Posterior collapse）。** KL 项过于强势地把 `q(z|x) → N(0, I)`，导致 `z` 完全不携带关于 `x` 的信息。解法：β 退火（β 从 0 开始，逐步升到 1）、free bits，或对不活跃维度跳过 KL。
- **样本模糊。** 高斯解码器似然意味着 MSE 重建，而 MSE 在 L2 意义下的贝叶斯最优解是均值——一堆合理数字图像的均值就是一张糊掉的数字。解法：离散解码器（VQ-VAE、NVAE），或者只把 VAE 当编码器用，在潜变量上叠扩散模型（Stable Diffusion 就是这么做的）。
- **β 太大、加得太早。** 参见后验坍缩。从 β≈0.01 起步，逐步上调。
- **潜在维度太小。** MNIST 用 16 维够了，ImageNet 256² 要 256 维，ImageNet 1024² 要 2048 维。Stable Diffusion 的 VAE 把 512×512×3 压缩到 64×64×4（空间面积上 32 倍下采样，通道上也压了 32 倍）。

## 生产实践

2026 年的 VAE 技术选型：

| 场景 | 选择 |
|-----------|------|
| 扩散模型的图像潜变量编码器 | Stable Diffusion VAE（`sd-vae-ft-ema`）或 Flux VAE |
| 音频潜变量编码器 | Encodec（Meta）、SoundStream 或 DAC（Descript） |
| 视频潜变量 | Sora 的时空 patch、Latte VAE、WAN VAE |
| 解耦表示学习 | β-VAE、FactorVAE、TCVAE |
| 离散潜变量（供 Transformer 建模） | VQ-VAE、RVQ（ResidualVQ） |
| 用于生成的连续潜变量 | 普通 VAE，再在该潜空间上条件化一个流/扩散模型 |

潜在扩散模型就是一个 VAE，只是在编码器和解码器之间住着一个扩散模型。VAE 做粗压缩，扩散模型干重活。视频（VAE + 视频扩散 DiT）和音频（Encodec + MusicGen transformer）用的也是同一套模式。

## 交付产物

保存 `outputs/skill-vae-trainer.md`。

该 Skill 的输入：数据集画像 + 目标潜在维度 + 下游用途（重建、采样，或作为潜在扩散的输入）；输出：架构选择（plain/β/VQ/RVQ）、β 调度方案、潜在维度、解码器似然（高斯 vs 类别分布），以及评估方案（重建 MSE、逐维 KL、`q(z|x)` 与 `N(0, I)` 之间的 Fréchet 距离）。

## 练习

1. **简单。** 把 `code/main.py` 中的 `β` 依次改成 `0.01`、`0.1`、`1.0`、`5.0`。记录最终的重建 MSE 和 KL。对你的合成数据来说，哪个 β 是 Pareto 最优？
2. **中等。** 把高斯解码器似然换成 Bernoulli 似然（交叉熵损失）。在同一份合成数据的二值化版本上比较样本质量。
3. **困难。** 把 `code/main.py` 扩展成一个迷你 VQ-VAE：用 K=32 大小码本中的最近邻查找替换连续的 `z`。比较重建 MSE，并报告实际被用到的码本条目数量（码本坍缩是真实存在的）。

## 关键术语

| 术语 | 大家怎么说 | 实际含义 |
|------|-----------------|-----------------------|
| Autoencoder | 编码-解码网络 | `x → z → x̂`，学 MSE。不具备生成能力。 |
| VAE | 带采样器的 AE | 编码器输出一个分布，KL 惩罚塑造编码空间。 |
| ELBO | 证据下界 | `log p(x) ≥ recon - KL[q(z\|x) \|\| p(z)]`；当 `q = p(z\|x)` 时取等。 |
| 重参数化 | `z = μ + σ·ε` | 把随机节点改写为确定性函数加纯噪声。让梯度能反向传播穿过采样。 |
| 先验 | `p(z)` | 潜变量的目标分布，通常是 `N(0, I)`。 |
| 后验坍缩 | “KL 项赢了” | 编码器无视 `x`，直接输出先验；解码器只能靠幻想重建。 |
| β-VAE | 可调的 KL 权重 | `loss = recon + β·KL`。β 越大解耦越好，但样本越模糊。 |
| VQ-VAE | 离散潜变量 | 用最近的码本向量替换连续的 `z`；使 Transformer 建模成为可能。 |

## 生产笔记：VAE 是扩散推理服务里最烫的路径

在 Stable Diffusion / Flux / SD3 流水线里，VAE 每个请求会被调用两次——编码一次（如果做 img2img / inpainting）、解码一次。在 1024² 分辨率下，解码这一遍往往是整条流水线中单次激活显存峰值最大的环节，因为它要把 `128×128×16` 的潜变量上采样回 `1024×1024×3`。两个实际后果：

- **解码时做切片或分块。** `diffusers` 提供 `pipe.vae.enable_slicing()` 和 `pipe.vae.enable_tiling()`。分块以一点轻微的接缝伪影为代价，把显存从 `O(H·W)` 降到 `O(tile²)`。在消费级 GPU 上跑 1024² 及以上分辨率时必不可少。
- **解码器用 bf16，最终 resize 用 fp32 数值。** SD 1.x 的 VAE 发布时是 fp32 的，转成 fp16 后在 1024² 及以上分辨率会*悄无声息地产生 NaN*。SDXL 提供了 `madebyollin/sdxl-vae-fp16-fix`——始终优先用 fp16-fix 版本，或者直接用 bf16。

## 延伸阅读

- [Kingma & Welling (2013). Auto-Encoding Variational Bayes](https://arxiv.org/abs/1312.6114) — VAE 开山论文。
- [Higgins et al. (2017). β-VAE: Learning Basic Visual Concepts with a Constrained Variational Framework](https://openreview.net/forum?id=Sy2fzU9gl) — 解耦表示的 β-VAE。
- [van den Oord et al. (2017). Neural Discrete Representation Learning](https://arxiv.org/abs/1711.00937) — VQ-VAE。
- [Vahdat & Kautz (2021). NVAE: A Deep Hierarchical Variational Autoencoder](https://arxiv.org/abs/2007.03898) — 最先进的图像 VAE。
- [Rombach et al. (2022). High-Resolution Image Synthesis with Latent Diffusion Models](https://arxiv.org/abs/2112.10752) — Stable Diffusion；VAE 作为编码器。
- [Défossez et al. (2022). High Fidelity Neural Audio Compression](https://arxiv.org/abs/2210.13438) — Encodec，音频 VAE 的事实标准。
