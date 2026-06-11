# 多头注意力

> 一个注意力头一次只能学一种关系。八个头就能学八种。多加头几乎不花成本，那就多要几个。

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 7 · 02 (Self-Attention from Scratch)
**Time:** ~75 minutes

## 问题背景

单个自注意力（self-attention）头只计算一个注意力矩阵。这个矩阵只能捕捉一种关系——通常是在当前训练信号下最能降低损失的那一种。如果你的数据里同时纠缠着主谓一致、共指消解、长距离语篇衔接和句法成分切分，单个头会把它们全部糅进同一个 softmax 分布里，丢掉一半信号。

2017 年 Vaswani 那篇论文给出的解法是：并行运行多个注意力函数，每个都有自己的 Q、K、V 投影，最后把输出拼接起来。每个头在一个维度为 `d_model / n_heads` 的更小子空间里工作。总参数量不变，表达能力却上去了。

多头注意力是 2026 年所有 Transformer 出厂自带的默认配置。剩下的争论只是用*多少个*头，以及键和值是否共享投影（分组查询注意力 Grouped-Query Attention、多查询注意力 Multi-Query Attention、多头潜在注意力 Multi-head Latent Attention）。

## 核心概念

![Multi-head attention splits, attends, concatenates](../assets/multi-head-attention.svg)

**切分。** 取形状为 `(N, d_model)` 的 `X`，投影得到 Q、K、V，各自形状为 `(N, d_model)`。reshape 成 `(N, n_heads, d_head)`，其中 `d_head = d_model / n_heads`。再转置成 `(n_heads, N, d_head)`。

**并行计算注意力。** 在每个头内部执行缩放点积注意力。每个头产生 `(N, d_head)` 的输出。各个头分别作用于嵌入的不同子空间，在注意力计算过程中彼此完全不交流。

**拼接并投影。** 把各头的输出叠回 `(N, d_model)`，再乘上一个可学习的输出矩阵 `W_o`，形状为 `(d_model, d_model)`。`W_o` 就是各个头互相混合的地方。

**为什么有效。** 每个头可以专精于自己的任务，不必和其他头争抢表示容量。2019–2024 年间的探针研究表明，不同的头确实分化出了不同角色：位置头、专门关注前一个 token 的头、复制头、命名实体头，以及归纳头（induction head，上下文学习能力背后的机制）。

**截至 2026 年的变体谱系：**

| 变体 | Q 头数 | K/V 头数 | 使用者 |
|---------|---------|-----------|---------|
| 多头注意力（MHA） | N | N | GPT-2、BERT、T5 |
| 多查询注意力（MQA） | N | 1 | PaLM、Falcon |
| 分组查询注意力（GQA） | N | G（如 N/8） | Llama 2 70B、Llama 3+、Qwen 2+、Mistral |
| 多头潜在注意力（MLA） | N | 压缩为低秩 | DeepSeek-V2、V3 |

GQA 是现代的默认选择，因为它把 KV 缓存的内存占用缩小为原来的 `G/N`，质量却几乎不打折。MLA 更进一步，把 K/V 压缩到一个潜在空间，计算时再投影回来——多花一点 FLOPs，省下多得多的内存。

```figure
multihead-split
```

## 从零实现

### 第 1 步：在已有的单头注意力基础上切分出多个头

拿出第 02 课的 `SelfAttention`，在外面套上一对 split/concat 操作。NumPy 实现见 `code/main.py`，核心逻辑是：

```python
def split_heads(X, n_heads):
    n, d = X.shape
    d_head = d // n_heads
    return X.reshape(n, n_heads, d_head).transpose(1, 0, 2)  # (heads, n, d_head)

def combine_heads(H):
    h, n, d_head = H.shape
    return H.transpose(1, 0, 2).reshape(n, h * d_head)
```

一次 reshape 加一次 transpose，没有循环。PyTorch 的 `nn.MultiheadAttention` 底层做的正是这件事。

### 第 2 步：在每个头内执行缩放点积注意力

每个头各自取走 Q、K、V 的一个切片。注意力变成一次批量矩阵乘法：

```python
def mha_forward(X, W_q, W_k, W_v, W_o, n_heads):
    Q = X @ W_q
    K = X @ W_k
    V = X @ W_v
    Qh = split_heads(Q, n_heads)         # (heads, n, d_head)
    Kh = split_heads(K, n_heads)
    Vh = split_heads(V, n_heads)
    scores = Qh @ Kh.transpose(0, 2, 1) / np.sqrt(Qh.shape[-1])
    weights = softmax(scores, axis=-1)
    out = weights @ Vh                    # (heads, n, d_head)
    concat = combine_heads(out)
    return concat @ W_o, weights
```

在真实硬件上，`Qh @ Kh.transpose(...)` 就是一次 `bmm`。GPU 看到的是一个形状为 `(heads, N, d_head) × (heads, d_head, N) -> (heads, N, N)` 的批量矩阵乘法。增加头数几乎不增加开销。

### 第 3 步：分组查询注意力（GQA）变体

只有键和值的投影发生变化。Q 仍有 `n_heads` 个组；K 和 V 只有 `n_kv_heads < n_heads` 个组，并通过重复来对齐：

```python
def gqa_project(X, W, n_kv_heads, n_heads):
    kv = split_heads(X @ W, n_kv_heads)       # (kv_heads, n, d_head)
    repeat = n_heads // n_kv_heads
    return np.repeat(kv, repeat, axis=0)      # (n_heads, n, d_head)
```

推理时这样能省内存，因为 KV 缓存里只需要存 `n_kv_heads` 份副本，而不是 `n_heads` 份。Llama 3 70B 用了 64 个查询头配 8 个 KV 头——缓存直接缩小 8 倍。

### 第 4 步：探查每个头学到了什么

在一个短句子上跑 4 个头的 MHA。对每个头打印它的 `(N, N)` 注意力矩阵。你会看到即便是随机初始化，不同的头也会捕捉到不同的结构——其中一部分是真实信号，另一部分是子空间的旋转对称性。

## 生产实践

在 PyTorch 里，一行版本：

```python
import torch.nn as nn

mha = nn.MultiheadAttention(embed_dim=512, num_heads=8, batch_first=True)
```

PyTorch 2.5+ 中的 GQA：

```python
from torch.nn.functional import scaled_dot_product_attention

# scaled_dot_product_attention auto-dispatches Flash Attention on CUDA.
# For GQA, pass Q of shape (B, n_heads, N, d_head) and K,V of shape
# (B, n_kv_heads, N, d_head). PyTorch handles the repeat.
out = scaled_dot_product_attention(q, k, v, is_causal=True, enable_gqa=True)
```

**用多少个头？** 2026 年生产级模型的经验值：

| 模型规模 | d_model | n_heads | d_head |
|------------|---------|---------|--------|
| 小型（~125M） | 768 | 12 | 64 |
| 基础（~350M） | 1024 | 16 | 64 |
| 大型（~1B） | 2048 | 16 | 128 |
| 前沿（~70B） | 8192 | 64 | 128 |

`d_head` 几乎总是落在 64 或 128。它决定了单个头能"看到"多少信息。低于 32，各个头就会开始和缩放因子 `sqrt(d_head)` 较劲；高于 256，就失去了"许多小而专的专家"带来的好处。

## 交付产物

见 `outputs/skill-mha-configurator.md`。该技能根据参数预算、序列长度和部署目标，为新 Transformer 推荐头数、KV 头数和投影策略。

## 练习

1. **简单。** 取 `code/main.py` 里的 MHA，固定 `d_model=64`，把 `n_heads` 从 1 调到 16。在一个合成复制任务上画出微型单层模型的损失曲线。更多的头是有帮助、进入平台期，还是反而有害？
2. **中等。** 实现 MQA（所有查询头共享一个 KV 头）。测量参数量相比完整 MHA 下降了多少。计算 N=2048 时推理阶段 KV 缓存缩小了多少。
3. **困难。** 实现一个微型版的多头潜在注意力：把 K、V 压缩到秩为 `r` 的潜在表示，KV 缓存中只存潜在表示，注意力计算时再解压。`r` 取多少时缓存内存降到完整 MHA 的 1/8 以下，同时验证集困惑度（ppl）损失保持在 1 bit 以内？

## 关键术语

| 术语 | 大家怎么说 | 实际含义 |
|------|-----------------|-----------------------|
| 头（Head） | "一条独立的注意力回路" | 一组维度为 `d_head = d_model / n_heads` 的 Q/K/V 投影，拥有自己的注意力矩阵。 |
| d_head | "头维度" | 每个头的隐藏宽度；生产环境几乎总是 64 或 128。 |
| 切分 / 合并 | "reshape 小技巧" | 注意力前后的 `(N, d_model) ↔ (n_heads, N, d_head)` reshape+transpose。 |
| W_o | "输出投影" | 拼接各头之后施加的 `(d_model, d_model)` 矩阵；各头在这里混合。 |
| MQA | "只有一个 KV 头" | 多查询注意力（Multi-Query Attention）：单份共享的 K/V 投影。KV 缓存最小，质量略有损失。 |
| GQA | "Llama 2 以来的默认配置" | 分组查询注意力（Grouped-Query Attention），`n_kv_heads < n_heads`；通过重复对齐 Q。 |
| MLA | "DeepSeek 的妙招" | 多头潜在注意力（Multi-head Latent Attention）：K、V 压缩成低秩潜在表示，注意力计算时再解压。 |
| 归纳头（Induction head） | "上下文学习背后的回路" | 一对协作的头：检测某个模式之前出现的位置，并复制其后面跟着的内容。 |

## 延伸阅读

- [Vaswani et al. (2017). Attention Is All You Need §3.2.2](https://arxiv.org/abs/1706.03762) — 多头注意力的原始定义。
- [Shazeer (2019). Fast Transformer Decoding: One Write-Head is All You Need](https://arxiv.org/abs/1911.02150) — MQA 论文。
- [Ainslie et al. (2023). GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints](https://arxiv.org/abs/2305.13245) — 如何在训练完成后把 MHA 转换为 GQA。
- [DeepSeek-AI (2024). DeepSeek-V2 Technical Report](https://arxiv.org/abs/2405.04434) — MLA 及其在缓存内存上胜过 MHA/GQA 的原因。
- [Olsson et al. (2022). In-context Learning and Induction Heads](https://transformer-circuits.pub/2022/in-context-learning-and-induction-heads/index.html) — 从机制可解释性视角看各个头到底在做什么。
