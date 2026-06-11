# 差分注意力（V2）

> Softmax 注意力会把一小部分概率分摊到每一个不匹配的 token 上。在 10 万 token 的上下文里，这些噪声累积起来足以淹没信号。Differential Transformer（Ye et al., ICLR 2025）的解决方案是把注意力计算成两个 softmax 之差，从而减掉两者共享的噪声底。DIFF V2（Microsoft，2026 年 1 月）是面向生产栈的重写版本：解码延迟与基线 Transformer 持平、无需自定义内核、兼容 FlashAttention。本课从 V1 到 V2 完整走一遍，并提供一个可以直接用标准库 Python 运行的差分运算玩具实现。

**Type:** Build
**Languages:** Python (stdlib)
**Prerequisites:** Phase 7 · 02 (self-attention), Phase 7 · 15 (attention variants), Phase 10 · 14 (architecture walkthrough)
**Time:** ~60 minutes

## 学习目标

- 精确说明 softmax 注意力为什么存在噪声底（noise floor），以及它为什么会随上下文长度增长。
- 推导差分注意力公式，并解释为什么相减能消除共享的噪声分量、同时保留信号。
- 走读 V1 到 V2 的差异：哪里变快了、哪里变简单了、哪里变稳定了，以及为什么每项改动对生产级预训练都是必要的。
- 用纯 Python 从零实现差分注意力，并在一条合成的"信号+噪声"查询上实验验证噪声消除特性。

## 问题背景

标准 softmax 注意力有一个数学性质，在大规模场景下会变成实际运维难题。对一个查询 `q`，注意力权重是 `softmax(qK^T / sqrt(d))`。Softmax 永远不会产生精确的零——每个不匹配的 token 都会分到一些正的概率质量。这些残余质量就是噪声，并且它随上下文长度增长。在 128k token 时，即使每个不匹配的 token 只分到 0.001% 的概率，127,999 个加起来也贡献了大约 12% 的总量。模型不得不学会绕开一个随上下文增长的噪声底。

实证上，这表现为注意力头之间的干扰：长上下文 RAG 中出现幻觉引用、10 万 token 检索任务上的"迷失在中间"（lost-in-the-middle）失败，以及超过 32k 后大海捞针（needle-in-haystack）基准上微妙的精度退化。Differential Transformer 论文（arXiv:2410.05258, ICLR 2025）量化了这一差距：相比同尺寸基线，DIFF Transformer 的困惑度更低、长上下文准确率更高、幻觉更少。

DIFF V1 有三个问题，使它一直无法进入前沿预训练流水线：每个解码步要把 value 缓存加载两次；需要自定义 CUDA 内核，破坏了 FlashAttention 兼容性；其逐头 RMSNorm 会在 70B 以上规模的长时间训练中导致不稳定。DIFF V2（Microsoft unilm 博客，2026 年 1 月 20 日）解决了全部三个问题。本课走读两个版本，构建差分算子，并在一条玩具查询上测量噪声消除效果。

## 核心概念

### softmax 的噪声底

对一个查询 `q` 和键 `K = [k_1, ..., k_N]`，注意力权重为：

```
w_i = exp(q . k_i / sqrt(d)) / sum_j exp(q . k_j / sqrt(d))
```

没有任何一个 `w_i` 会等于零。如果 `k_i` 与 `q` 完全无关，分数 `q . k_i` 并不是 0——它在零附近波动，方差为 `||q||^2 / d`。经过 softmax 归一化后，每个无关 token 仍然向加权和贡献 `O(1/N)`。所有无关 token 的总贡献是 `O((N-1)/N) = O(1)`——这不是一个小量。

模型真正想要的是类似硬性 top-k 的效果：匹配的 token 拿到高权重，其余位置权重接近零。Softmax 太平滑，无法直接做到这一点。

### 差分思想

把每个头的 Q、K 投影各拆成两份：Q = (Q_1, Q_2)，K = (K_1, K_2)。计算两张注意力图：

```
A_1 = softmax(Q_1 K_1^T / sqrt(d))
A_2 = softmax(Q_2 K_2^T / sqrt(d))
```

输出：

```
DiffAttn = (A_1 - lambda * A_2) V
```

相减会消除两张图共享的噪声分布。如果两张图在 12.7 万个无关 token 上的权重都大致均匀（随机初始化时确实如此），这部分就被抵消了。而信号——少数真正相关 token 上的尖峰权重——只有在两张图中以相同幅度出现时才会被抵消，而模型一旦经过训练就不会出现这种情况。

`lambda` 是每个头一个的可学习标量，参数化为 `lambda = exp(lambda_q1 dot lambda_k1) - exp(lambda_q2 dot lambda_k2) + lambda_init`。它可以为负。`lambda_init` 默认是一个较小的正数，例如 0.8。

### 为什么这对应耳机式的主动降噪

想象两个有噪声的麦克风录制同一个人的声音。两者都录到了说话人加上相关的背景噪声。把一路从另一路中减掉，共享噪声就消失了。人声之所以保留下来，是因为两路信号在相位或幅度上差异足够大，不会被完全抵消。每个头的 `lambda` 学的正是这个平衡。

### V1 与 V2 的差异

V1 保持参数量与基线 Transformer 相等。为了让每个头有两个查询，它把头维度（head dimension）减半。这牺牲了头的表达能力，而且——更痛苦的是——把每个头的 value 缓存也减半了。解码时每一步必须加载 value 缓存两次（每个 softmax 分支一次）。结果：尽管参数量持平，解码却比基线更慢。

V2 把查询头数翻倍，KV 头数保持不变（参数从上投影那里借来）。头维度与基线保持一致。相减之后，多出的维度被投影回与基线 Transformer 的 O_W 投影匹配的尺寸。三件事同时发生：

1. 解码速度与基线持平（KV 缓存只加载一次）。
2. FlashAttention 原样可用（无需自定义内核）。
3. 解码时的算术强度（arithmetic intensity）提高了（每从 HBM 加载一字节，能做更多计算）。

V2 还移除了 V1 用来稳定相减操作的逐头 RMSNorm。在 70B 量级的预训练规模下，这个 RMSNorm 会在训练后期造成不稳定。V2 用一个更简单的初始化方案替代了它，无需额外模块也能保持训练稳定。

### 什么时候该用它

| 工作负载 | 收益 |
|----------|---------|
| 长上下文 RAG（64k+） | 注意力图更干净，幻觉引用更少 |
| 大海捞针基准 | 超过 32k 后准确率显著提升 |
| 多文档问答 | 跨文档干扰更少 |
| 8k 上下文的代码补全 | 收益边际，不值得改架构 |
| 短对话（< 4k） | 与基线基本无差别 |

收益随上下文长度增长。在 4k token 时噪声底足够小，标准注意力没问题。到了 128k，它就在拖你后腿了。

### 它如何与 2026 年的其他技术组合

| 特性 | 与 DIFF V2 兼容吗？ |
|---------|------------------------|
| GQA | 兼容（V2 增加的是 Q 头，不是 KV 头） |
| MLA（DeepSeek） | 原理上兼容，尚无两者结合的公开论文 |
| MoE | 兼容（注意力与 MLP 块相互独立） |
| RoPE | 兼容（无需改动） |
| YaRN / 长上下文扩展 | 兼容（这正是 DIFF 帮助最大的场景） |
| FlashAttention | V2 兼容（V1 不兼容） |
| 投机解码 | 兼容（注意力的改动对投机解码循环不可见） |

```figure
differential-attention
```

## 从零实现

`code/main.py` 用纯 Python 实现差分注意力。一条具有已知"信号+噪声"结构的玩具查询，让你可以直接测量噪声消除比。

### 第 1 步：标准 softmax 注意力

标准库矩阵运算：列表的列表、手写矩阵乘法、为数值稳定先减去最大值的 softmax。

```python
def softmax(row):
    m = max(row)
    exps = [math.exp(x - m) for x in row]
    s = sum(exps)
    return [e / s for e in exps]
```

### 第 2 步：把 Q、K 拆成两半

V1 风格：头维度减半。V2 风格：保持头维度不变，头数翻倍。玩具实现为了讲解清晰采用 V1 方式——数学完全相同，区别只在记账方式。

### 第 3 步：两个 softmax 分支 + 相减

```python
A1 = [softmax([dot(q1, k) / scale for k in K1]) for q1 in Q1]
A2 = [softmax([dot(q2, k) / scale for k in K2]) for q2 in Q2]
diff_weights = [[a1 - lam * a2 for a1, a2 in zip(r1, r2)] for r1, r2 in zip(A1, A2)]
out = [[sum(w * v[j] for w, v in zip(row, V)) for j in range(d_v)] for row in diff_weights]
```

注意：输出权重可以为负。这没有问题——value 缓存照常处理带符号的贡献，后续的 V 投影会吸收这个符号。

### 第 4 步：噪声消除测量

构建一条长度 1024 的合成序列。把信号 token 放在已知位置，其余填充噪声。分别计算（a）标准 softmax 注意力在信号位置上的权重和（b）差分注意力的权重。测量两者各自的信噪比。DIFF 注意力可靠地产生更高的信噪比，倍数在 3 倍到 10 倍之间，取决于两个分支经过训练后差异有多大。

### 第 5 步：V1 与 V2 的参数核算

给定配置（hidden=4096, heads=32, d_head=128），打印：

- 基线 Transformer：Q、K、V 各为 `hidden * hidden`，MLP 为 4 * hidden。
- DIFF V1：Q、K 各为 `hidden * hidden`，V 为 `hidden * hidden`（不变），内部头维度减半。新增每个头的 `lambda` 参数（O(heads * d_head)）。
- DIFF V2：Q 为 `2 * hidden * hidden`，K 为 `hidden * hidden`，V 为 `hidden * hidden`。多出的维度在 O_W 之前投影回原尺寸。新增同样的 `lambda` 参数。

玩具实现会测量 V2 的额外参数开销（每个注意力块大约多出 `hidden * hidden`）并打印出来。

## 生产实践

截至 2026 年 4 月，DIFF V2 尚未在所有生产推理服务器中落地，但 vLLM 和 SGLang 的集成工作正在进行中。与此同时，这一模式已经出现在：

- Microsoft 内部的长上下文生产模型。
- 多个面向 256k 以上上下文的开源模型训练中的研究复现。
- 将 DIFF 注意力与滑动窗口注意力按层交替组合的混合架构。

2026 年你会在什么情况下用它：

- 从零训练一个目标有效上下文在 64k 以上的新模型。从一开始就加入差分注意力；事后重训代价高昂。
- 微调一个长上下文模型，且"迷失在中间"失败主导了你的评测结果。在 Q 投影上加一个 LoRA 可以近似 DIFF 结构。

什么情况下不用：

- 你在服务一个长上下文性能稳定的预训练稠密模型。在现有权重上重训的成本很少能收回。
- 你的上下文始终在 16k 以下。噪声底可以忽略不计。

## 交付产物

本课产出 `outputs/skill-diff-attention-integrator.md`。给定模型架构、目标上下文长度、幻觉表现和训练预算，它会生成一份在新的预训练或 LoRA 微调中加入差分注意力的集成方案。

## 练习

1. 运行 `code/main.py`。验证在合成查询上，差分注意力报告的信噪比高于标准 softmax 注意力。改变噪声幅度，找出标准注意力变得不可用的交叉点。

2. 对一个 7B 量级模型（hidden=4096, heads=32, d_head=128, 32 层），计算从基线到 DIFF V1、以及从基线到 DIFF V2 的参数量增量。指出哪些组件增加了参数、哪些保持不变。

3. 阅读 DIFF V1 论文（arXiv:2410.05258）的第 3 节和 DIFF V2 Hugging Face 博客的第 2 节。用两句话解释：为什么 V1 需要逐头 RMSNorm，以及为什么 V2 移除它而不会导致训练发散。

4. 实现一个消融实验：分别用 `lambda = 0`（纯第一个 softmax）和 `lambda = 1`（完全相减）计算差分注意力。在合成查询上测量信噪比在整个扫描区间内的变化，找出使信噪比最大的 `lambda`。

5. 把玩具实现扩展到 GQA + DIFF V2。选用 8 个 KV 头和 32 个 Q 头。证明 KV 缓存大小与同样 (8, 32) 配置的基线 GQA 模型一致。

## 关键术语

| 术语 | 大家怎么说 | 实际含义 |
|------|----------------|------------------------|
| 差分注意力 | "两个 softmax 互减" | 把 Q、K 各拆成两半，计算两张 softmax 图，从第一张中减去乘以 lambda 的第二张，再乘以 V |
| 噪声底 | "softmax 的非零尾巴" | softmax 分给每个无关 token 的 O(1/N) 权重，在长上下文中累加为 O(1) |
| lambda | "相减的缩放系数" | 每个头一个的可学习标量，参数化为 `exp(lq1.lk1) - exp(lq2.lk2) + lambda_init`；可以为负 |
| DIFF V1 | "ICLR 2025 版本" | 原始 Differential Transformer；为保持参数量将头维度减半，需要自定义内核，解码更慢 |
| DIFF V2 | "2026 年 1 月的修复版" | Q 头翻倍而 KV 头不变；解码速度与基线持平，可用 FlashAttention |
| 逐头 RMSNorm | "V1 的稳定器" | V1 在差分之后额外加的归一化；V2 移除了它，以避免训练后期不稳定 |
| 信噪比 | "有多少注意力被浪费了" | 真实信号位置上的权重与无关位置平均权重之比 |
| 迷失在中间 | "长上下文失败模式" | 经验现象：长上下文中间部分文档的检索准确率会下降——DIFF 注意力可缓解这一点 |
| 算术强度 | "每加载一字节做多少 FLOPs" | V2 通过每次 KV 加载执行双倍查询而提升的比值；对受内存带宽限制的解码很重要 |

## 延伸阅读

- [Ye et al. — Differential Transformer (arXiv:2410.05258, ICLR 2025)](https://arxiv.org/abs/2410.05258) —— 原始论文，包含噪声消除理论和长上下文消融实验
- [Microsoft unilm — Differential Transformer V2 (Hugging Face blog, January 2026)](https://huggingface.co/blog/microsoft/diff-attn-v2) —— 面向生产栈的重写版本，解码速度与基线持平，兼容 FlashAttention
- [Understanding Differential Transformer Unchains Pretrained Self-Attentions (arXiv:2505.16333)](https://arxiv.org/abs/2505.16333) —— 关于相减为何能恢复预训练注意力结构的理论分析
- [Shared DIFF Transformer (arXiv:2501.17900)](https://arxiv.org/html/2501.17900) —— 参数共享变体
- [Vaswani et al. — Attention Is All You Need (arXiv:1706.03762)](https://arxiv.org/abs/1706.03762) —— DIFF 所减去的基线 Transformer
- [Liu et al. — Lost in the Middle (arXiv:2307.03172)](https://arxiv.org/abs/2307.03172) —— DIFF 注意力所针对的长上下文基准
