# 注意力变体 —— 滑动窗口、稀疏与差分注意力

> 全量注意力是一个圆：每个 token 都看见每个 token，而内存为此买单。四种变体改变了这个圆的形状，省下了一半的开销。

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 7 · 02 (Self-Attention), Phase 7 · 03 (Multi-Head), Phase 7 · 12 (KV Cache / Flash Attention)
**Time:** ~60 minutes

## 问题背景

全量注意力的内存和计算开销均为序列长度的 `O(N²)`。对于 128K 上下文的 Llama 3 70B，这意味着每层 160 亿个注意力条目，再乘以 80 层。Flash Attention（第 12 课）隐藏了 `O(N²)` 的激活内存，但没有改变算术开销——每个 token 仍然要关注其他所有 token。

有三类变体直接改变注意力矩阵本身的拓扑结构：

1. **滑动窗口注意力（Sliding Window Attention，SWA）。** 每个 token 只关注固定窗口内的邻近 token，而不是完整前缀。内存和计算降至 `O(N · W)`，其中 `W` 是窗口大小。Gemma 2/3、Mistral 7B 的前几层、Phi-3-Long 都在用。
2. **稀疏 / 分块注意力。** 只对选定的 `(i, j)` 对计算分数，其余位置的权重被强制置零。Longformer、BigBird、OpenAI sparse transformer。
3. **差分注意力（Differential Attention）。** 用两组独立的 Q/K 投影计算两张注意力图，再将其中一张减去另一张。这消灭了把权重泄漏到开头几个 token 的「注意力沉降（attention sink）」现象。Microsoft 的 DIFF Transformer（2024）。

这些变体可以共存。一个 2026 年的前沿模型常常混用它们：大多数层是 SWA-1024，每第五层是全局全量注意力，再加少量差分注意力头来提升检索质量。Gemma 3 的 5:1 SWA 与全局层比例是当前的教科书级默认配置。

## 核心概念

### 滑动窗口注意力（SWA）

位置 `i` 处的每个查询只关注 `[i - W, i]` 范围内的位置（因果 SWA），或 `[i - W/2, i + W/2]`（双向）。窗口之外的 token 在分数矩阵中被置为 `-inf`。

```
full causal:           sliding window (W=4):
positions 0-7          positions 0-7, W=4
    0 1 2 3 4 5 6 7        0 1 2 3 4 5 6 7
0 | x                0 |  x
1 | x x              1 |  x x
2 | x x x            2 |  x x x
3 | x x x x          3 |  x x x x
4 | x x x x x        4 |    x x x x
5 | x x x x x x      5 |      x x x x
6 | x x x x x x x    6 |        x x x x
7 | x x x x x x x x  7 |          x x x x
```

当 `N = 8192`、`W = 1024` 时，分数矩阵中非零条目期望规模是 1024 × 8192——缩减了 8 倍。

**SWA 会让 KV 缓存变小。** 每层只需保留最后 `W` 个 token 的 K 和 V。对于类 Gemma-3 的配置（窗口 1024、上下文 128K），KV 缓存缩小 128 倍。

**质量代价。** 只用 SWA 的 Transformer 在长程检索上表现吃力。解决办法：将 SWA 层与全量注意力层交错排布。Gemma 3 用的是 5:1 的 SWA 与全局层比例。Mistral 7B 用的是因果 SWA 堆叠，信息通过相互重叠的窗口「向前流动」——每一层把有效感受野扩展 `W`，经过 `L` 层后模型可以回看 `L × W` 个 token。

### 稀疏 / 分块注意力

预先选定一个 `N × N` 的稀疏模式。三种经典形状：

- **局部 + 跨步（OpenAI sparse transformer）。** 关注最近 `W` 个 token，外加更早位置中每隔 `stride` 个取一个的 token。以 `O(N · sqrt(N))` 的计算量同时捕获局部与长程信息。
- **Longformer / BigBird。** 局部窗口 + 少量全局 token（如 `[CLS]`，它们关注所有人、也被所有人关注）+ 随机稀疏连接。实验表明在同等质量下可支撑 2 倍上下文。
- **Native Sparse Attention（DeepSeek，2025）。** 学习哪些 `(Q, K)` 块重要，在内核层面直接跳过零块。与 FlashAttention 兼容。

稀疏注意力本质上是一个内核工程的故事。数学很简单（给分数矩阵加掩码）；收益来自从不把零条目加载进 SRAM。FlashAttention-3 和 2026 年的 FlexAttention API 让自定义稀疏模式在 PyTorch 中成为一等公民。

### 差分注意力（DIFF Transformer，2024）

常规注意力存在「注意力沉降」问题：softmax 强制每行权重之和为 1，于是那些没有特定关注对象的 token 会把权重倾倒在第一个 token（或开头几个 token）上。这窃取了本该分配给真实内容的容量。

差分注意力的修复方式是计算**两张**注意力图并做减法：

```
A1 = softmax(Q1 K1^T / √d)
A2 = softmax(Q2 K2^T / √d)
DiffAttn = (A1 - λ · A2) V
```

其中 `λ` 是一个可学习的标量（通常为 0.5–0.8）。A1 捕获真实的内容权重；A2 捕获沉降部分。相减抵消掉沉降，把权重重新分配给真正相关的 token。

报告的结果（Microsoft 2024）：困惑度降低 5–10%，在相同训练长度下有效上下文延长 1.5–2 倍，「大海捞针」式检索更精准。

### 变体对比

| 变体 | 计算量 | KV 缓存 | 相对全量注意力的质量 | 生产应用 |
|---------|---------|----------|-----------------|----------------|
| 全量注意力 | O(N²) | 每层 O(N) | 基线 | 每个模型的默认层 |
| SWA（窗口 1024） | O(N·W) | 每层 O(W) | -0.1 ppl，搭配全局层效果好 | Gemma 2/3、Phi-3-Long |
| 局部 + 跨步稀疏 | O(N·√N) | 混合 | 与 SWA 相近 | OpenAI sparse transformer、Longformer |
| BigBird（局部 + 全局 + 随机） | 近似 O(N) | 混合 | 2 倍上下文下与全量持平 | 早期长上下文 BERT |
| Native Sparse（DeepSeek-V3.2） | O(N · 活跃比例) | O(N) | 差距在 0.05 ppl 以内 | DeepSeek-V3.2，2025 |
| 差分注意力 | O(2·N²) | O(2N) | ppl 降低 5% 至 10% | DIFF Transformer、2026 年初的模型 |

```figure
gqa-kv-sharing
```

## 从零实现

参见 `code/main.py`。我们实现一个因果掩码对比器，在一个玩具序列上并排展示全量、SWA、局部+跨步、差分四种注意力。

### 第 1 步：全量因果掩码（基线）

```python
def causal_mask(n):
    return [[0.0 if j <= i else float("-inf") for j in range(n)] for i in range(n)]
```

来自第 07 课的基线。下三角矩阵；对角线以上权重为零。

### 第 2 步：滑动窗口因果掩码

```python
def swa_mask(n, window):
    M = [[float("-inf")] * n for _ in range(n)]
    for i in range(n):
        lo = max(0, i - window + 1)
        for j in range(lo, i + 1):
            M[i][j] = 0.0
    return M
```

只有一个参数——`window`。当 `window >= n` 时退化为全量因果注意力；当 `window = 1` 时每个 token 只关注自己。

### 第 3 步：局部 + 跨步稀疏掩码

```python
def strided_mask(n, window, stride):
    M = [[float("-inf")] * n for _ in range(n)]
    for i in range(n):
        lo = max(0, i - window + 1)
        for j in range(lo, i + 1):
            M[i][j] = 0.0
        for j in range(0, i + 1, stride):
            M[i][j] = 0.0
    return M
```

稠密的局部窗口，外加从序列开头起每隔 `stride` 个取一个的 token。感受野随层数增加按对数步长增长。

### 第 4 步：差分注意力

```python
def diff_attention(Q1, K1, Q2, K2, V, lam):
    A1 = softmax_causal(Q1 @ K1.T / sqrt_d)
    A2 = softmax_causal(Q2 @ K2.T / sqrt_d)
    return (A1 - lam * A2) @ V
```

两次注意力计算，用一个可学习的混合系数做减法。代码中我们对比单注意力与差分注意力的沉降热力图，观察沉降现象的消失。

### 第 5 步：KV 缓存大小

打印 `N = 131072` 时各变体每层的缓存大小。SWA 和稀疏变体缩小 10–100 倍，差分注意力则翻倍。要清醒地为内存账单付费。

## 生产实践

2026 年的生产模式：

```python
from transformers import AutoModelForCausalLM
# Gemma 3 mixes SWA (window=1024) and global layers at 5:1.
model = AutoModelForCausalLM.from_pretrained("google/gemma-3-27b-it")
# print(model.config.sliding_window, model.config.layer_types)
```

PyTorch 2.5+ 中的 FlexAttention 接受一个掩码函数：

```python
from torch.nn.attention.flex_attention import flex_attention, create_block_mask

def swa_pattern(b, h, q_idx, kv_idx):
    return (q_idx - kv_idx < 1024) & (q_idx >= kv_idx)

mask = create_block_mask(swa_pattern, B=batch, H=heads, Q_LEN=n, KV_LEN=n)
out = flex_attention(q, k, v, block_mask=mask)
```

它会编译成定制的 Triton 内核。对于常见模式，速度与 FlashAttention-3 相差不到 10%，而且掩码函数就是一个 Python 可调用对象。

**各方案的选择时机：**

- **纯全量注意力** —— 上下文不超过约 16K 时全部用它，或检索质量是头等大事的场景。
- **SWA + 全局混合** —— 长上下文（>32K），训练和推理受内存限制。超过 32K 时的 2026 年默认方案。
- **稀疏分块注意力** —— 定制内核、定制模式。留给专门的工作负载（检索、音频）。
- **差分注意力** —— 任何受注意力沉降污染影响的负载（长上下文 RAG、大海捞针检索）。

## 交付产物

参见 `outputs/skill-attention-variant-picker.md`。该技能根据目标上下文长度、检索需求以及训练/推理算力画像，为新模型挑选注意力拓扑。

## 练习

1. **简单。** 运行 `code/main.py`。验证 `window=4` 的 SWA 把每行最近 4 个 token 之外的位置全部置零。验证 `window=n` 与全量因果注意力逐比特一致。
2. **中等。** 在第 07 课的 capstone 之上实现 `window=1024` 的因果 SWA。在 tinyshakespeare 上训练 1,000 步。验证损失相比全量注意力退化了多少？峰值内存下降了多少？
3. **困难。** 在 capstone 模型中实现 Gemma-3 风格的 5:1 层混合（5 层 SWA、1 层全局）。在参数量相同的条件下，与纯 SWA 和纯全局基线对比损失、内存和生成质量。
4. **困难。** 实现每个注意力头拥有独立可学习 `λ` 的差分注意力。在一个合成检索任务上训练（1 个目标、2,000 个干扰项）。在参数量相同的条件下，对比其与单注意力基线的检索准确率。

## 关键术语

| 术语 | 通俗说法 | 实际含义 |
|------|-----------------|-----------------------|
| 滑动窗口注意力（SWA） | 「局部注意力」 | 每个查询只关注最近 `W` 个 token；KV 缓存缩小到 `O(W)`。 |
| 有效感受野 | 「模型能回看多远」 | 在窗口为 `W` 的 `L` 层 SWA 堆叠中，最远可达 `L × W` 个 token。 |
| Longformer / BigBird | 「局部 + 全局 + 随机」 | 带少量始终参与注意力的全局 token 的稀疏模式；早期的长上下文方案。 |
| Native Sparse Attention | 「DeepSeek 的内核技巧」 | 学习块级稀疏性；在内核层面跳过零块，同时保持质量。 |
| 差分注意力 | 「两张图，相减一张」 | DIFF Transformer：从第一张注意力图中减去可学习的 `λ` 乘以第二张图，以抵消注意力沉降。 |
| 注意力沉降（attention sink） | 「权重泄漏到 token 0」 | softmax 归一化强制每行权重和为 1；没有明确关注目标的查询会把权重倾倒在位置 0 上。 |
| FlexAttention | 「掩码即 Python」 | PyTorch 2.5+ 的 API，把任意掩码函数编译成 FlashAttention 形态的内核。 |
| 层类型混合 | 「5:1 SWA 比全局」 | 在堆叠中交错排布稀疏层与全量注意力层，以更低内存保住质量。 |

## 延伸阅读

- [Beltagy, Peters, Cohan (2020). Longformer: The Long-Document Transformer](https://arxiv.org/abs/2004.05150) —— 滑动窗口 + 全局 token 的开山之作。
- [Zaheer et al. (2020). Big Bird: Transformers for Longer Sequences](https://arxiv.org/abs/2007.14062) —— 局部 + 全局 + 随机。
- [Child et al. (2019). Generating Long Sequences with Sparse Transformers](https://arxiv.org/abs/1904.10509) —— OpenAI 的局部+跨步模式。
- [Gemma Team (2024). Gemma 2: Improving Open Language Models at a Practical Size](https://arxiv.org/abs/2408.00118) —— 1:1 的 SWA 与全局层混合。
- [Gemma Team (2025). Gemma 3 technical report](https://arxiv.org/abs/2503.19786) —— 窗口 1024、比例 5:1 的混合方案，如今的教科书默认配置。
- [Ye et al. (2024). Differential Transformer](https://arxiv.org/abs/2410.05258) —— DIFF Transformer 论文。
- [Yuan et al. (2025). Native Sparse Attention](https://arxiv.org/abs/2502.11089) —— DeepSeek-V3.2 的可学习稀疏注意力。
- [PyTorch — FlexAttention blog and docs](https://pytorch.org/blog/flexattention/) —— 「生产实践」一节中掩码即可调用对象模式的 API 参考。
