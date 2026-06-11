# 缩放定律

> 2020 年的 Kaplan 论文说：模型越大，损失越低。2022 年的 Hoffmann 论文说：你们训练得不够。算力要花在两个口袋里——参数和 token——而怎么分配并不显而易见。

**Type:** Learn
**Languages:** Python
**Prerequisites:** Phase 7 · 05 (Full Transformer), Phase 7 · 07 (GPT)
**Time:** ~45 minutes

## 问题背景

当你手握 C FLOPs 的训练算力、想训出最好的模型时，你面前有两个旋钮：

1. **用多少参数（N）？** 模型越大，容量越高。
2. **用多少训练 token（D）？** 数据越多，容量被利用得越充分。

FLOPs 约等于 `6 × N × D`。你可以把 N 调大、D 调小，也可以反过来。哪种更好？

2022 年之前，答案是「使劲推大 N」。GPT-3（2020）有 1750 亿参数，却只在约 3000 亿 token 上训练，每个参数大约只对应 1.7 个 token。Kaplan 缩放定律为这种做法背书。

Hoffmann 等人（2022）训练了一族名为 Chinchilla 的小模型，得出了不同的结论：最优比例接近**每个参数 20 个 token**。GPT-3 训练不足了 10 倍。Chinchilla（700 亿参数、1.4 万亿 token）在所有基准上都击败了 GPT-3（1750 亿参数、3000 亿 token），推理成本还低 2.5 倍。

2026 年是 Chinchilla 的世界——但有一个重要的转折。Llama 3 8B 在 15 万亿 token 上训练，每个参数对应 1875 个 token，是 Chinchilla 最优比例的 94 倍。对于将被大规模使用的模型，推理成本比训练成本更重要，所以为了换取更小的可部署体积而过度训练（超出 Chinchilla 比例）已成为 2026 年的默认做法。

## 核心概念

![Chinchilla curves: loss vs compute at various N/D ratios](../assets/scaling-laws.svg)

### Hoffmann 定律

根据 Chinchilla 论文，损失服从：

```
L(N, D) = A / N^α + B / D^β + E
```

- `N` = 参数量（不含嵌入层）。
- `D` = 训练 token 数。
- `α ≈ 0.34`、`β ≈ 0.28`（大致对称）。
- `E ≈ 1.69`，即不可约损失（irreducible loss）下限。
- `A ≈ 406`、`B ≈ 411`。

随着规模扩大，两项相互制衡。在固定算力（C = 6ND）下对 `N` 求导并求解：

```
N_opt ≈ 0.6 × (C/6)^0.5
D_opt ≈ 0.6 × (C/6)^0.5
D_opt / N_opt ≈ 20
```

算力最优解：每个参数 20 个 token。

### 为什么还是要过度训练

Chinchilla 最优比例最小化的是每训练 FLOP 的训练损失。但训练成本只付一次，推理成本却要永远付下去。

对于每月要服务一万亿 token 的聊天机器人，推理在总成本中占主导。Llama 的思路是：把模型训得更小、更久。8B 模型配 15T token，是深度面向推理优化的：

- 能装进消费级 GPU。
- 延迟只有 Chinchilla 最优 70B 模型的零头。
- 质量对大多数任务来说已经足够接近。

DeepMind 2024 年的论文（《Over-training is the new optimal》）将这一思路形式化。对于推理主导的工作负载，合适的比例更接近每个参数 100–500 个 token，具体取决于服务量。

### 涌现 vs 平滑

有一种说法：某些能力（算术、多步推理、跟随思维链）会在某个规模上突然「涌现」。

Schaeffer 等人（2023）论证这是测量假象：涌现指标使用不连续的打分方式（精确匹配、阈值化准确率），掩盖了底层 logits 的平滑改进。换用连续指标（交叉熵）后，曲线是平滑的。

2026 年的共识是：基于连续损失的预测是可靠的，基准分数的跳变往往是打分器的假象。预算规划应当对照连续指标进行。

### 2026 年的全景

缩放定律仍然有效，但是：

| 因素 | 改变了什么 |
|--------|-------------|
| 数据质量 | 精选「优质」token（Phi 风格）可使曲线移动超过 2 倍有效算力 |
| MoE | 总参数量与激活 FLOPs 解耦；缩放定律按每激活 FLOP 计算 |
| 后训练 | 某些能力（指令跟随、代码）受 SFT+RLHF 的影响超过预训练 |
| 多模态 | 图像 + 文本 token 一起扩展；每种模态有各自的曲线 |
| 合成数据 | 模型自己生成训练数据；有效算力可以复利式增长 |

Muon 优化器（Kimi Moonlight，2024）在相同数据量下展现出相对 AdamW 约 2 倍的有效算力增益。2026 年的一些训练默认采用 Muon。它改变的是缩放定律中的绝对常数，而非定律的形状。

```figure
scaling-laws
```

## 从零实现

参见 `code/main.py`。我们实现 Chinchilla 损失方程，并在多个算力预算下分别求解算力最优的 `(N, D)`。

### 第 1 步：Chinchilla 损失

```python
def chinchilla_loss(N, D, A=406.4, B=410.7, alpha=0.34, beta=0.28, E=1.69):
    return A / N ** alpha + B / D ** beta + E
```

在固定 `C = 6ND` 的条件下，把 `L` 画成 `(N, D)` 平面上的等高线图，找出最小值。

### 第 2 步：算力最优前沿

对于从 `1e17` 到 `1e25` FLOPs 的算力预算，在约束 `6ND = C` 下找出使损失最小的 `(N, D)`。验证比例 `D/N ≈ 20`。

### 第 3 步：过度训练的代价

计算把模型缩小 10 倍（最优 N 的 1/10、最优 D 的 10 倍）需要额外付出的损失。同时报告换来的推理 FLOP 节省（与 N 成正比）。

### 第 4 步：与真实模型对比

代入 GPT-3、Chinchilla、Llama 3 8B、DeepSeek-V3（激活参数量）的已知 `(N, D)` 组合，比较预测损失与公开报告的损失。

## 生产实践

你大概率不会亲自训练一个前沿模型。但缩放定律能告诉你：

1. **你的微调数据够不够。** 如果任务数据少于基座模型每参数 20 个 token，损失会在某个下限处饱和。
2. **要不要换更大的基座模型。** 如果预算全花在推理上，应优先选择更小、训练时间更长的模型。
3. **收益在哪里递减。** 超出 Chinchilla 最优比例 1000 倍之后，对数损失的变化就成了噪声。

**2026 年的研究走向：**

- **数据受限阶段。** 互联网上的高质量 token 是有限的（过滤后的英文约 5–10 万亿）。前沿预训练正在逼近这个天花板。合成数据、多语言、多模态以及规模化的 RLHF 微调是接下来的杠杆。
- **算力倍增技巧。** Muon 优化器、MoE、更好的数据筛选——每一项都只移动绝对常数，不改变渐近线。
- **强化学习的缩放定律。** 仍是开放问题。早期证据表明 RL 样本数也服从幂律，但指数与预训练截然不同。

## 交付产物

参见 `outputs/skill-training-budget-estimator.md`。该技能根据算力预算、部署约束和目标损失，为新的训练任务选择 `(N, D, hours, GPU)`。

## 练习

1. **简单。** 运行 `code/main.py`。打印算力预算为 `1e20`、`1e22`、`1e24` 时的 Chinchilla 最优 `(N, D)`，并与真实模型表对比。
2. **中等。** 实现 Hoffmann 的损失-算力曲线。沿算力最优前沿绘制损失 vs `log10(C)`。找出定律预测在何时需要 `>10^28` FLOPs 才能让交叉熵再降 0.1。
3. **困难。** 在同一数据集上训练 5 个微型模型（10 万到 1000 万参数），拟合你自己的缩放定律。估计 `α` 和 `E`。你的指数与已发表的结果吻合程度如何？

## 关键术语

| 术语 | 大家怎么说 | 实际含义 |
|------|-----------------|-----------------------|
| 参数量（N） | 「模型大小」 | 不含嵌入层的权重数量；决定容量。 |
| Token 数（D） | 「训练数据」 | 见过的训练 token 数量；决定参数被利用得多充分。 |
| 算力（C） | 「花掉的 FLOPs」 | 对标准 Transformer 约为 `6 × N × D`。 |
| Chinchilla 最优 | 「D/N ≈ 20」 | 使每预训练 FLOP 的损失最小的比例。 |
| 过度训练 | 「超出 Chinchilla」 | 多花训练 FLOPs 来省推理 FLOPs；D/N >> 20。 |
| 不可约损失 | 「下限」 | 缩放定律中的 `E` 项；数据本身的熵。 |
| 涌现能力 | 「规模到了就突然跳变」 | 往往是打分器假象；连续损失是平滑的。 |
| 有效算力 | 「训练效率乘数」 | 更好的数据 / 优化器 / 架构会成倍放大每个 FLOP 的作用。 |

## 延伸阅读

- [Kaplan et al. (2020). Scaling Laws for Neural Language Models](https://arxiv.org/abs/2001.08361) —— 第一篇缩放定律论文；结论是训练不足。
- [Hoffmann et al. (2022). Training Compute-Optimal Large Language Models](https://arxiv.org/abs/2203.15556) —— Chinchilla。
- [Schaeffer et al. (2023). Are Emergent Abilities of Large Language Models a Mirage?](https://arxiv.org/abs/2304.15004) —— 涌现是测量假象。
- [Sardana, Frankle (2024). Beyond Chinchilla-Optimal: Accounting for Inference in Language Model Scaling Laws](https://arxiv.org/abs/2401.00448) —— 为什么 Llama 的过度训练对它的工作负载是正确的。
- [Jordan et al. (2024). Muon: An optimizer for hidden layers in neural networks](https://kellerjordan.github.io/posts/muon/) —— 2 倍算力乘数。
