# 位置编码 — Sinusoidal、RoPE、ALiBi

> 注意力对排列顺序是不变的。在没有位置信号的情况下，"The cat sat on the mat" 和 "mat the on sat cat the" 会产生相同的输出。三种算法解决了这个问题——每种算法对"位置"的含义都下了不同的赌注。

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 7 · 02 (Self-Attention), Phase 7 · 03 (Multi-Head Attention)
**Time:** ~45 minutes

## 问题背景

缩放点积注意力对顺序是"失明"的。注意力矩阵 `softmax(Q K^T / √d) V` 由成对相似度计算得出。把 `X` 的行打乱，输出的行也会以同样的方式被打乱。注意力内部没有任何东西关心位置。

对于词袋模型来说这不算缺陷。但对于语言、代码、音频、视频——任何顺序承载意义的场景——这是致命的。

解决办法是设法把位置信息注入嵌入（embedding）中。三个时代给出了三种答案：

1. **绝对正弦位置编码（Absolute sinusoidal）**（Vaswani 2017）。把位置的 `sin/cos` 加到嵌入上。简单、无需学习参数，但在超出训练长度后外推能力很差。
2. **RoPE — 旋转位置编码（Rotary Position Embeddings）**（Su 2021）。将 Q 和 K 向量旋转一个与位置成正比的角度。直接在点积中编码*相对*位置。是 2026 年的主流方案。
3. **ALiBi — 线性偏置注意力（Attention with Linear Biases）**（Press 2022）。完全跳过嵌入；基于距离给注意力分数加上每个头独立的线性惩罚。长度外推能力出色。

截至 2026 年，几乎所有前沿开源模型都使用 RoPE：Llama 2/3/4、Qwen 2/3、Mistral、Mixtral、DeepSeek-V3、Kimi。少数长上下文模型使用 ALiBi 或其现代变体。绝对正弦位置编码已成为历史。

## 核心概念

![Sinusoidal absolute vs RoPE rotations vs ALiBi distance bias](../assets/positional-encoding.svg)

### 绝对正弦位置编码

预先计算一个形状为 `(max_len, d_model)` 的固定矩阵 `PE`：

```
PE[pos, 2i]   = sin(pos / 10000^(2i / d_model))
PE[pos, 2i+1] = cos(pos / 10000^(2i / d_model))
```

然后在注意力之前执行 `X' = X + PE[:N]`。每个维度都是一条不同频率的正弦曲线。模型学会从相位模式中读取位置。超出 `max_len` 就会失效：如果模型只见过位置 0–2047，没人告诉它位置 2048 会发生什么。

### RoPE

旋转 Q 和 K 向量（而不是嵌入）。对于一对维度 `(2i, 2i+1)`：

```
[q'_2i    ]   [ cos(pos·θ_i)  -sin(pos·θ_i) ] [q_2i   ]
[q'_2i+1  ] = [ sin(pos·θ_i)   cos(pos·θ_i) ] [q_2i+1 ]

θ_i = base^(-2i / d_head),  base = 10000 by default
```

对 key 用位置 `pos_k` 施加同样的旋转。点积 `q'_m · k'_n` 就变成了只依赖 `(m - n)` 的函数。也就是说：**注意力分数只取决于相对距离**，尽管旋转是基于绝对位置进行的。漂亮的技巧。

扩展 RoPE：可以对 `base` 进行缩放（NTK-aware、YaRN、LongRoPE），无需重新训练就能外推到更长的上下文。Llama 3 正是用这种方式把上下文从 8K 扩展到了 128K。

### ALiBi

跳过嵌入技巧，直接给注意力分数加偏置：

```
attn_score[i, j] = (q_i · k_j) / √d  -  m_h · |i - j|
```

其中 `m_h` 是每个头特有的斜率（例如 `1 / 2^(8·h/H)`）。距离近的 token 得到加强；距离远的 token 受到惩罚。没有训练时开销。论文显示其长度外推能力优于正弦位置编码，并在原始训练长度上与 RoPE 持平。

### 2026 年该怎么选

| 变体 | 外推能力 | 训练成本 | 使用者 |
|---------|---------------|---------------|---------|
| 绝对正弦位置编码 | 差 | 免费 | 原始 Transformer、早期 BERT |
| 可学习绝对位置编码 | 无 | 极小 | GPT-2、GPT-3 |
| RoPE | 配合缩放后良好 | 免费 | Llama 2/3/4、Qwen 2/3、Mistral、DeepSeek-V3、Kimi |
| RoPE + YaRN | 出色 | 需微调阶段 | Qwen2-1M、Llama 3.1 128K |
| ALiBi | 出色 | 免费 | BLOOM、MPT、Baichuan |

RoPE 胜出的原因在于：它能直接嵌入注意力机制而无需改变架构、编码的是相对位置，并且它的 `base` 超参数为长上下文微调提供了一个干净的调节旋钮。

```figure
rope-explorer
```

## 从零实现

### 第 1 步：正弦位置编码

参见 `code/main.py`。核心计算只有 4 行：

```python
def sinusoidal(N, d):
    pe = [[0.0] * d for _ in range(N)]
    for pos in range(N):
        for i in range(d // 2):
            theta = pos / (10000 ** (2 * i / d))
            pe[pos][2 * i]     = math.sin(theta)
            pe[pos][2 * i + 1] = math.cos(theta)
    return pe
```

在第一个注意力层之前，把它加到嵌入矩阵上。

### 第 2 步：把 RoPE 应用到 Q、K

RoPE 直接在 Q 和 K 上原地操作。对每一对维度：

```python
def apply_rope(x, pos, base=10000):
    d = len(x)
    out = list(x)
    for i in range(d // 2):
        theta = pos / (base ** (2 * i / d))
        c, s = math.cos(theta), math.sin(theta)
        a, b = x[2 * i], x[2 * i + 1]
        out[2 * i]     = a * c - b * s
        out[2 * i + 1] = a * s + b * c
    return out
```

关键点：对位置 `m` 的 Q 和位置 `n` 的 K 应用同一个函数。它们的点积会在每一对坐标上获得一个 `cos((m-n)·θ_i)` 因子。注意力不费吹灰之力就学到了相对位置。

### 第 3 步：ALiBi 斜率与偏置

```python
def alibi_bias(n_heads, seq_len):
    # slope_h = 2 ** (-8 * h / n_heads) for h = 1..n_heads
    slopes = [2 ** (-8 * (h + 1) / n_heads) for h in range(n_heads)]
    bias = []
    for m in slopes:
        row = [[-m * abs(i - j) for j in range(seq_len)] for i in range(seq_len)]
        bias.append(row)
    return bias  # add to attention scores before softmax
```

把 `bias[h]` 加到第 `h` 个头的 `(seq_len, seq_len)` 注意力分数矩阵上，然后做 softmax。

### 第 4 步：验证 RoPE 的相对距离性质

随机取两个向量 `a, b`。先按 `(pos_a, pos_b)` 旋转，再按 `(pos_a + k, pos_b + k)` 旋转。两次得到的点积必须在浮点误差范围内一致。这个性质正是 RoPE 的全部意义所在——它对绝对偏移量不敏感，只有相对间隔才重要。

## 生产实践

PyTorch 2.5+ 在 `torch.nn.functional` 中提供了 RoPE 工具函数。大多数生产代码使用 `flash_attn` 或 `xformers`，RoPE 在注意力内核（kernel）内部完成应用。

```python
from transformers import AutoModel
model = AutoModel.from_pretrained("meta-llama/Llama-3.2-3B")
# model.config.rope_scaling → {"type": "yarn", "factor": 32.0, "original_max_position_embeddings": 8192}
```

**2026 年的长上下文技巧：**

- **NTK-aware 插值。** 从 4K 扩展到 16K+ 时，将 `base` 重新缩放为 `base * (scale_factor)^(d/(d-2))`。
- **YaRN。** 更聪明的插值方式，在长上下文上保持注意力熵不变。Llama 3.1 128K 使用了它。
- **LongRoPE。** Microsoft 2024 年的方法，用进化搜索为每个维度挑选缩放因子。Phi-3-Long 使用了它。
- **位置插值 + 微调。** 直接按扩展倍数缩小位置，然后用 1–5B token 进行微调。效果出奇地好。

## 交付产物

参见 `outputs/skill-positional-encoding-picker.md`。该 skill 会根据目标上下文长度、外推需求和训练预算，为新模型选择一种位置编码策略。

## 练习

1. **简单。** 以热力图形式绘制 `max_len=512, d=128` 的正弦 `PE` 矩阵。确认"维度索引越大，条纹越宽"的模式。
2. **中等。** 实现 NTK-aware 的 RoPE 缩放。在长度为 256 的序列上训练一个微型语言模型，然后分别在有缩放和无缩放的情况下测试长度 1024 的序列。测量困惑度（perplexity）。
3. **困难。** 在同一个注意力模块中同时实现 ALiBi 和 RoPE。在长度为 512 的序列上用复制任务训练一个 4 层 Transformer。测试时外推到 2048。比较两者的性能退化情况。

## 关键术语

| 术语 | 人们的说法 | 实际含义 |
|------|-----------------|-----------------------|
| 位置编码（Positional encoding） | "告诉注意力顺序信息" | 任何加入嵌入或注意力中、用来编码位置的信号。 |
| 正弦位置编码（Sinusoidal） | "最原始的那个" | 把按几何级数频率排列的 `sin/cos` 加到嵌入上；无法外推。 |
| RoPE | "旋转嵌入" | 按位置相关的角度旋转 Q、K；点积编码相对距离。 |
| ALiBi | "线性偏置技巧" | 给注意力分数加上 `-m·\|i-j\|`；无需嵌入，外推能力极佳。 |
| base | "RoPE 的旋钮" | RoPE 中的频率缩放因子；调大它可以在推理时扩展上下文。 |
| NTK-aware | "一种 RoPE 缩放技巧" | 重新缩放 `base`，使高频维度在上下文扩展时不被过度压缩。 |
| YaRN | "高级版的那个" | 逐维度的插值+外推方法，保持注意力熵不变。 |
| 外推（Extrapolation） | "在超出训练长度后仍能工作" | 位置方案能否在超过训练时见过的 `max_len` 后仍输出正确结果？ |

## 延伸阅读

- [Vaswani et al. (2017). Attention Is All You Need §3.5](https://arxiv.org/abs/1706.03762) — 原始正弦位置编码。
- [Su et al. (2021). RoFormer: Enhanced Transformer with Rotary Position Embedding](https://arxiv.org/abs/2104.09864) — RoPE 论文。
- [Press, Smith, Lewis (2021). Train Short, Test Long: Attention with Linear Biases Enables Input Length Extrapolation](https://arxiv.org/abs/2108.12409) — ALiBi。
- [Peng et al. (2023). YaRN: Efficient Context Window Extension of Large Language Models](https://arxiv.org/abs/2309.00071) — 当前最先进的 RoPE 缩放方法。
- [Chen et al. (2023). Extending Context Window of Large Language Models via Positional Interpolation](https://arxiv.org/abs/2306.15595) — Meta 的 Llama 2 长上下文论文。
- [Ding et al. (2024). LongRoPE: Extending LLM Context Window Beyond 2 Million Tokens](https://arxiv.org/abs/2402.13753) — Phi-3-Long 使用的 Microsoft 方法，在"生产实践"一节中提及。
- [HuggingFace Transformers — `modeling_rope_utils.py`](https://github.com/huggingface/transformers/blob/main/src/transformers/modeling_rope_utils.py) — 各种 RoPE 缩放方案（default、linear、dynamic、YaRN、LongRoPE、Llama-3）的生产级实现。
