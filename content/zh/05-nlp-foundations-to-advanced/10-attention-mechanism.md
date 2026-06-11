# 注意力机制 —— 关键突破

> 解码器不再眯着眼盯一份压缩摘要，而是直接审视整个源序列。此后的一切，都是注意力加上工程化。

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 5 · 09 (Sequence-to-Sequence Models)
**Time:** ~45 minutes

## 问题背景

第 09 课以一次量化出来的失败收场。一个在玩具复制任务上训练的 GRU 编码器-解码器，在长度为 5 时准确率有 89%，到长度 80 时几乎降到随机水平。原因是结构性的，不是训练上的 bug：编码器获取的全部信息都必须塞进一个固定大小的隐藏状态里，而解码器除此之外什么也看不到。

Bahdanau、Cho 和 Bengio 在 2014 年发表了一个三行就能讲清的修复方案。不要只把编码器的最终状态交给解码器，而是保留每一个编码器状态。在每个解码步，计算编码器状态的加权平均，权重表达的是"此刻解码器需要看编码器位置 `i` 多少眼？"这个加权平均就是上下文，并且它在每个解码步都会变化。

这就是全部思想。Transformer 把它扩展开来。自注意力（self-attention）把它用在单个序列上。多头注意力（multi-head attention）让它并行运行。但 2014 年的版本就已经打破了瓶颈，一旦掌握了它，转向 Transformer 就只是工程问题，而非概念问题。

## 核心概念

![Bahdanau attention: decoder queries all encoder states](../assets/attention.svg)

在每个解码步 `t`：

1. 把上一步的解码器隐藏状态 `s_{t-1}` 用作**查询（query）**。
2. 用它与每个编码器隐藏状态 `h_1, ..., h_T` 计算得分。每个编码器位置一个标量。
3. 对得分做 softmax，得到总和为 1 的注意力权重 `α_{t,1}, ..., α_{t,T}`。
4. 上下文向量 `c_t = Σ α_{t,i} * h_i`。即编码器状态的加权平均。
5. 解码器接收 `c_t` 加上前一个输出 token，生成下一个 token。

加权平均正是关键所在。当解码器需要把 "Je" 翻译成 "I" 时，它会给覆盖 "Je" 的那个编码器状态很高的权重，其他的权重很低。当它需要生成 "not" 时，就给 "pas" 高权重。上下文向量在每一步都会重新塑形。

## 形状（坑过所有人的地方）

每个人第一次实现注意力时，都会在这里栽跟头。请慢慢读。

| 对象 | 形状 | 说明 |
|-------|-------|-------|
| 编码器隐藏状态 `H` | `(T_enc, d_h)` | 如果是 BiLSTM，则 `d_h = 2 * d_hidden` |
| 解码器隐藏状态 `s_{t-1}` | `(d_s,)` | 一个向量 |
| 注意力得分 `e_{t,i}` | 标量 | 每个编码器位置一个 |
| 注意力权重 `α_{t,i}` | 标量 | 对所有 `i` 做 softmax 之后 |
| 上下文向量 `c_t` | `(d_h,)` | 与单个编码器状态形状相同 |

**Bahdanau（加性）得分。** `e_{t,i} = v_α^T * tanh(W_a * s_{t-1} + U_a * h_i)`。

- `s_{t-1}` 形状为 `(d_s,)`，`h_i` 形状为 `(d_h,)`。
- `W_a` 形状为 `(d_attn, d_s)`。`U_a` 形状为 `(d_attn, d_h)`。
- 两者在 tanh 内部相加后形状为 `(d_attn,)`。
- `v_α` 形状为 `(d_attn,)`。与 `v_α` 做内积后坍缩成一个标量。**这就是 `v_α` 的作用。**它不是魔法，而是把注意力维度的向量投影成标量得分的那个投影。

**Luong（乘性）得分。** 共三种变体：

- `dot`：`e_{t,i} = s_t^T * h_i`。要求 `d_s == d_h`，这是硬性约束。如果你的编码器是双向的，直接跳过。
- `general`：`e_{t,i} = s_t^T * W * h_i`，其中 `W` 形状为 `(d_s, d_h)`。去掉了维度相等的约束。
- `concat`：本质上就是 Bahdanau 的形式。由于前两种更便宜，很少使用。

**一个值得点名的 Bahdanau / Luong 易错点。** Bahdanau 使用 `s_{t-1}`（生成当前词*之前*的解码器状态）。Luong 使用 `s_t`（生成*之后*的状态）。把两者搞混会产生微妙错误的梯度，极难调试。选定一篇论文，严格遵守它的约定。

```figure
attention-heatmap
```

## 从零实现

### 第 1 步：加性（Bahdanau）注意力

```python
import numpy as np


def additive_attention(decoder_state, encoder_states, W_a, U_a, v_a):
    projected_dec = W_a @ decoder_state
    projected_enc = encoder_states @ U_a.T
    combined = np.tanh(projected_enc + projected_dec)
    scores = combined @ v_a
    weights = softmax(scores)
    context = weights @ encoder_states
    return context, weights


def softmax(x):
    x = x - np.max(x)
    e = np.exp(x)
    return e / e.sum()
```

对照上面的表格检查形状。`encoder_states` 形状为 `(T_enc, d_h)`。`projected_enc` 形状为 `(T_enc, d_attn)`。`projected_dec` 形状为 `(d_attn,)`，会自动广播。`combined` 形状为 `(T_enc, d_attn)`。`scores` 形状为 `(T_enc,)`。`weights` 形状为 `(T_enc,)`。`context` 形状为 `(d_h,)`。可以交付了。

### 第 2 步：Luong 的 dot 和 general

```python
def dot_attention(decoder_state, encoder_states):
    scores = encoder_states @ decoder_state
    weights = softmax(scores)
    return weights @ encoder_states, weights


def general_attention(decoder_state, encoder_states, W):
    projected = W.T @ decoder_state
    scores = encoder_states @ projected
    weights = softmax(scores)
    return weights @ encoder_states, weights
```

每个只要三行。这就是 Luong 那篇论文成功的原因：在大多数任务上准确率相同，代码却少得多。

### 第 3 步：一个手算数值示例

给定三个编码器状态（大致对应 "cat"、"sat"、"mat"），以及一个与第一个状态最对齐的解码器状态，注意力分布会集中在位置 0。如果解码器状态转而与最后一个对齐，注意力就会移到位置 2。上下文向量随之跟踪变化。

```python
H = np.array([
    [1.0, 0.0, 0.2],
    [0.5, 0.5, 0.1],
    [0.1, 0.9, 0.3],
])

s_close_to_cat = np.array([0.9, 0.1, 0.2])
ctx, w = dot_attention(s_close_to_cat, H)
print("weights:", w.round(3))
```

```
weights: [0.464 0.305 0.231]
```

第一行胜出。然后把解码器状态移近第三个编码器状态，观察权重如何转移。就是这样。注意力就是显式的对齐。

### 第 4 步：为什么这是通往 Transformer 的桥梁

把上面的语言翻译成 Q/K/V：

- **Query（查询）** = 解码器状态 `s_{t-1}`
- **Key（键）** = 编码器状态（用来计算得分的对象）
- **Value（值）** = 编码器状态（用来加权求和的对象）

在经典注意力中，键和值是同一个东西。自注意力把它们分开了：你可以让一个序列对自身做查询，K 和 V 各用不同的可学习投影。多头注意力用不同的可学习投影并行运行。Transformer 把整个模块堆叠很多层，并彻底抛弃了 RNN。

数学是一样的。形状是一样的。从 Bahdanau 注意力跳到缩放点积注意力（scaled dot-product attention），在教学上基本只是换了记号。

## 生产实践

PyTorch 和 TensorFlow 都直接内置了注意力。

```python
import torch
import torch.nn as nn

mha = nn.MultiheadAttention(embed_dim=128, num_heads=8, batch_first=True)
query = torch.randn(2, 5, 128)
key = torch.randn(2, 10, 128)
value = torch.randn(2, 10, 128)

output, weights = mha(query, key, value)
print(output.shape, weights.shape)
```

```
torch.Size([2, 5, 128]) torch.Size([2, 5, 10])
```

这就是一个 Transformer 注意力层。查询批次有 5 个位置，键/值批次有 10 个位置，每个 128 维，8 个头。`output` 是融合了上下文的新查询表示。`weights` 是那个 5x10 的对齐矩阵，可以拿来可视化。

### 经典注意力仍然重要的场景

- 教学。单头、单层、基于 RNN 的版本让每个概念都清晰可见。
- Transformer 放不下的端侧序列任务。
- 任何 2014-2017 年的论文。不了解 Bahdanau 的约定，你会读错。
- 机器翻译中的细粒度对齐分析。即使在 Transformer 模型上，原始注意力权重也是一种可解释性工具，而读懂它们的前提是知道它们到底是什么。

### "注意力权重即解释"的陷阱

注意力权重看起来很可解释。它们是在各位置上总和为 1 的权重；可以画出来；数值高就意味着"看了这里"。审稿人很喜欢它们。

但它们并不像看起来那么可解释。Jain 和 Wallace（2019）的研究表明，在某些任务上，注意力分布可以被打乱或替换成任意的其他分布，而模型预测不变。在没有消融实验或反事实检验的情况下，绝不要把注意力权重当作模型推理过程的证据来报告。

## 交付产物

保存为 `outputs/prompt-attention-shapes.md`：

```markdown
---
name: attention-shapes
description: Debug shape bugs in attention implementations.
phase: 5
lesson: 10
---

Given a broken attention implementation, you identify the shape mismatch. Output:

1. Which matrix has the wrong shape. Name the tensor.
2. What its shape should be, derived from (d_s, d_h, d_attn, T_enc, T_dec, batch_size).
3. One-line fix. Transpose, reshape, or project.
4. A test to catch regressions. Typically: assert `output.shape == (batch, T_dec, d_h)` and `weights.shape == (batch, T_dec, T_enc)` and `weights.sum(dim=-1) close to 1`.

Refuse to recommend fixes that silently broadcast. Broadcast-hiding bugs surface later as silent accuracy degradation, the worst kind of attention bug.

For Bahdanau confusion, insist the decoder input is `s_{t-1}` (pre-step state). For Luong, `s_t` (post-step state). For dot-product, flag dimension mismatch between query and key as the most common first-time error.
```

## 练习

1. **简单。**实现带掩码（masking）的 `softmax`，使编码器中的填充（padding）token 注意力权重为零。在一个包含变长序列的批次上测试。
2. **中等。**给 Luong 的 `general` 形式加上多头注意力。把 `d_h` 切分成 `n_heads` 组，逐头计算注意力，再拼接。验证单头情形与你之前的实现结果一致。
3. **困难。**在第 09 课的玩具复制任务上训练一个带 Bahdanau 注意力的 GRU 编码器-解码器。绘制准确率随序列长度变化的曲线，并与无注意力基线对比。你应该看到差距随长度增长而拉大，证实注意力解除了瓶颈。

## 关键术语

| 术语 | 大家怎么说 | 实际含义 |
|------|-----------------|-----------------------|
| 注意力（Attention） | 看东西 | 对值序列的加权平均，权重由查询-键相似度计算得到。 |
| Query、Key、Value | QKV | 三种投影：Q 负责提问，K 是被匹配的对象，V 是被返回的内容。 |
| 加性注意力 | Bahdanau | 前馈式得分：`v^T tanh(W q + U k)`。 |
| 乘性注意力 | Luong dot / general | 得分为 `q^T k` 或 `q^T W k`。更便宜，多数任务上准确率相同。 |
| 对齐矩阵 | 那张好看的图 | 排成 `(T_dec, T_enc)` 网格的注意力权重。读它就能看到模型关注了什么。 |

## 延伸阅读

- [Bahdanau, Cho, Bengio (2014). Neural Machine Translation by Jointly Learning to Align and Translate](https://arxiv.org/abs/1409.0473) —— 原始论文。
- [Luong, Pham, Manning (2015). Effective Approaches to Attention-based Neural Machine Translation](https://arxiv.org/abs/1508.04025) —— 三种得分变体及其对比。
- [Jain and Wallace (2019). Attention is not Explanation](https://arxiv.org/abs/1902.10186) —— 可解释性方面的警示。
- [Dive into Deep Learning — Bahdanau Attention](https://d2l.ai/chapter_attention-mechanisms-and-transformers/bahdanau-attention.html) —— 可运行的 PyTorch 实战讲解。
