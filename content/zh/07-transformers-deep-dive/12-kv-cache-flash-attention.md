# KV 缓存、Flash Attention 与推理优化

> 训练是并行的、受算力（FLOP）限制；推理是串行的、受显存带宽限制。瓶颈不同，优化手段也不同。

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 7 · 02 (Self-Attention), Phase 7 · 05 (Full Transformer), Phase 7 · 07 (GPT)
**Time:** ~75 minutes

## 问题背景

朴素的自回归解码器生成 `N` 个 token 需要 `O(N²)` 的计算量：每一步都要对整个前缀重新计算注意力。对于一段 4K token 的回复，这意味着 1600 万次注意力运算，其中绝大部分是冗余的。前缀中每个 token 的隐藏状态一旦算出就是确定的——你只需要拿新 token 的查询（query）去和之前所有 token 缓存好的键（key）和值（value）做计算。

更糟的是，注意力本身就要搬运大量数据。标准注意力会显式生成 N×N 的分数矩阵、N×d 的 softmax 输出、N×d 的最终输出——对 HBM 的读写次数太多了。当 N≥2K 时，注意力在受算力限制之前就先变成了受内存带宽限制。经典的注意力核函数（kernel）让现代 GPU 的利用率低了 4–10 倍。

两项优化（都出自 Dao 等人）把前沿模型的推理从"慢"推向了"快"：

1. **KV 缓存（KV cache）。** 存储每个前缀 token 的 K 和 V 向量。每个新 token 的注意力只需用一个查询去匹配缓存的键。推理的每步生成开销从 `O(N²)` 降到 `O(N)`。
2. **Flash Attention。** 对注意力计算做分块（tiling），让完整的 N×N 矩阵永远不落到 HBM。softmax 和矩阵乘法全部在 SRAM 中完成。在 A100 上有 2–4 倍的实际加速；在 H100 上配合 FP8 可达 5–10 倍。

到 2026 年，这两项优化已经无处不在。每一个生产级推理框架（vLLM、TensorRT-LLM、SGLang、llama.cpp）都默认依赖它们。每一个前沿模型出厂时都开启了 Flash Attention。

## 核心概念

![KV cache growth and Flash Attention tiling](../assets/kv-cache-flash-attn.svg)

### KV 缓存的计算

每个解码器层、每个 token、每个注意力头：

```
bytes_per_token_per_layer = 2 * d_head * dtype_size
                          ^
                          K and V
```

以一个 7B 模型为例（32 层、32 个头、d_head=128、fp16）：

```
per token per layer = 2 * 128 * 2 = 512 bytes
per token (32 layers) = 16 KB
per 32K context = 512 MB
```

对于 Llama 3 70B（80 层、d_head=128、GQA 共 8 个 KV 头）：

```
per token per layer = 2 * 8 * 128 * 2 = 4096 bytes (4 KB)
per 32K context = 10.4 GB
```

正是这 10 GB 解释了为什么 Llama 3 70B 在 128K 上下文下，即便批大小为 1，光是 KV 缓存就要吃掉一张 40 GB A100 的大部分显存。

**GQA 是 KV 缓存层面的最大赢家。** 如果用 64 个头的 MHA，将需要 32 GB。MLA 则压缩得更狠。

拖动各个维度，观察缓存大小如何变化。把序列长度或批大小往上推，看看它多快就超出单张 GPU 的容量：

```figure
kv-cache-sizer
```

### Flash Attention——分块的技巧

标准注意力：

```
S = Q @ K^T          (HBM read, N×N, HBM write)
P = softmax(S)       (HBM read, HBM write)
O = P @ V            (HBM read, HBM write)
```

三次 HBM 往返。在 H100 上，HBM 带宽是 3 TB/s，SRAM 是 30 TB/s。相比把数据全部留在芯片上，每一次 HBM 往返都意味着 10 倍的减速。

Flash Attention：

```
for each block of Q (tile size ~128 × 128):
    load Q_tile into SRAM
    for each block of K, V:
        load K_tile, V_tile into SRAM
        compute S_tile = Q_tile @ K_tile^T     (SRAM)
        running softmax aggregation             (SRAM)
        accumulate into O_tile                  (SRAM)
    write O_tile to HBM
```

每个分块只需一次 HBM 往返。总内存占用从 `O(N²)` 降到 `O(N)`。反向传播时重新计算前向过程中的部分中间值，而不是把它们存下来——又省了一笔内存。

**数值技巧。** 滚动式（running）softmax 在分块之间维护 `(max, sum)`，使最终的归一化结果精确无误。这不是近似算法——Flash Attention 的输出与标准注意力逐比特一致（除去 fp16 运算不满足结合律带来的差异）。

**版本演进：**

| 版本 | 年份 | 关键改动 | 在参考硬件上的加速 |
|---------|------|-----------|-------------------------------|
| Flash 1 | 2022 | 分块 SRAM 核函数 | A100 上 2 倍 |
| Flash 2 | 2023 | 更好的并行度、因果优先的计算顺序 | A100 上 3 倍 |
| Flash 3 | 2024 | Hopper 异步特性、FP8 | H100 上 1.5–2 倍（FP16 约 740 TFLOPs） |
| Flash 4 | 2026 | Blackwell 五级流水线、软件 exp2 | 面向推理优先（初期仅支持前向） |

Flash 4 发布时只支持前向传播。训练仍然使用 Flash 3。Flash 4 对 GQA 和变长序列（varlen）的支持尚未落地（预计 2026 年中）。

### 投机解码（speculative decoding）——另一项延迟优化

廉价的小模型提议 N 个 token，大模型并行验证这 N 个。如果验证通过了 k 个 token，你就用 1 次大模型前向传播的代价换来了 k 个 token 的生成。在代码和散文场景下，典型的 k 为 3–5。

2026 年的默认选择：
- **EAGLE 2 / Medusa。** 集成式草稿头，与验证模型共享隐藏状态。2–3 倍加速且无质量损失。
- **基于草稿模型的投机解码。** 在消费级硬件上有 2–4 倍加速。
- **Lookahead 解码。** 基于 Jacobi 迭代，无需草稿模型。小众但零成本。

### 连续批处理（continuous batching）

经典的批量推理：等批内最慢的序列结束，再开始新的一批。短回复提前结束时，GPU 就被白白浪费。

连续批处理（最早由 Orca 实现，现已进入 vLLM、TensorRT-LLM、SGLang）：旧请求一结束就立即把新请求换入批中。对典型的对话负载有 5–10 倍的吞吐提升。

### PagedAttention——把 KV 缓存当作虚拟内存

这是 vLLM 的招牌特性。KV 缓存以 16 个 token 为一块进行分配；一张页表把逻辑位置映射到物理块。这样就能在并行采样（束搜索、并行采样）间共享 KV、为提示词缓存热切换前缀，并能整理内存碎片。相比朴素的连续分配，吞吐提升 4 倍。

```figure
flash-attention-memory
```

## 从零实现

参见 `code/main.py`。我们将实现：

1. 一个朴素的 `O(N²)` 增量解码器。
2. 一个 `O(N)` 的 KV 缓存解码器。
3. 一个模拟 Flash Attention 滚动最大值算法的分块 softmax。

### 第 1 步：KV 缓存

```python
class KVCache:
    def __init__(self, n_layers, n_heads, d_head):
        self.K = [[[] for _ in range(n_heads)] for _ in range(n_layers)]
        self.V = [[[] for _ in range(n_heads)] for _ in range(n_layers)]

    def append(self, layer, head, k, v):
        self.K[layer][head].append(k)
        self.V[layer][head].append(v)

    def read(self, layer, head):
        return self.K[layer][head], self.V[layer][head]
```

很简单：在按层、按头组织的列表里持续追加每个 token 的 K、V 向量。

### 第 2 步：分块 softmax

```python
def tiled_softmax_dot(q, K, V, tile=4):
    """Flash-attention-style softmax(qK^T)V with running max/sum."""
    m = float("-inf")
    s = 0.0
    out = [0.0] * len(V[0])
    for start in range(0, len(K), tile):
        k_block = K[start:start + tile]
        v_block = V[start:start + tile]
        scores = [sum(qi * ki for qi, ki in zip(q, k)) for k in k_block]
        new_m = max(m, *scores)
        exp_old = math.exp(m - new_m) if m != float("-inf") else 0.0
        exp_new = [math.exp(sc - new_m) for sc in scores]
        s = s * exp_old + sum(exp_new)
        for j in range(len(out)):
            out[j] = out[j] * exp_old + sum(e * v[j] for e, v in zip(exp_new, v_block))
        m = new_m
    return [o / s for o in out]
```

输出与一次性计算 `softmax(qK) V` 逐比特一致，但任意时刻的工作集只是一个 `tile × d_head` 的块，而不是完整的 `N × d_head`。

### 第 3 步：在生成 100 个 token 时对比朴素解码与缓存解码

统计注意力运算次数。朴素版：`O(N²)` = 5050。缓存版：`O(N)` = 100。代码会把两个数字都打印出来。

## 生产实践

```python
# HuggingFace transformers auto-enables KV cache on decoder-only generate().
from transformers import AutoModelForCausalLM
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3.2-3B",
    attn_implementation="flash_attention_2",  # use FA3 if Hopper
    torch_dtype="bfloat16",
)
# generate() uses KV cache automatically
```

vLLM 生产部署：

```bash
pip install vllm
vllm serve meta-llama/Llama-3.1-70B-Instruct \
    --tensor-parallel-size 4 \
    --max-model-len 32768 \
    --enable-prefix-caching \
    --kv-cache-dtype fp8
```

跨请求的前缀缓存（prefix caching）是 2026 年的一大收益——相同的系统提示词、少样本示例或长上下文文档可以在多次调用间复用 KV。对于工具提示词反复出现的智能体（agent）负载，前缀缓存通常能带来 5 倍的吞吐提升。

## 交付产物

参见 `outputs/skill-inference-optimizer.md`。该技能会为新的推理部署选择注意力实现、KV 缓存策略、量化方案和投机解码方案。

## 练习

1. **简单。** 运行 `code/main.py`。确认朴素解码器和缓存解码器输出一致；记录两者运算次数的差异。
2. **中等。** 实现前缀缓存：给定一个提示词 P 和若干续写，先对 P 做一次前向传播填充 KV 缓存，再按续写分支计算。测量相比为每个续写重新编码 P 的加速比。
3. **困难。** 实现一个玩具版 PagedAttention：KV 缓存按固定的 16-token 块分配，并维护一个空闲块列表（free-list）。序列结束时把它的块归还到池中。模拟 1000 次长度各异的对话补全。对比内存碎片化程度与连续分配方案的差异。

## 关键术语

| 术语 | 大家怎么说 | 实际含义 |
|------|-----------------|-----------------------|
| KV 缓存 | "让解码变快的那个技巧" | 存储所有前缀 token 的 K 和 V；新的查询直接对它们做注意力，而不是重新计算。 |
| HBM | "GPU 主显存" | 高带宽内存（High Bandwidth Memory）；H100 上 80 GB，B200 上 192 GB。带宽约 3 TB/s。 |
| SRAM | "片上内存" | 每个 SM 的高速内存，H100 上每 SM 约 256 KB。带宽约 30 TB/s。 |
| Flash Attention | "分块注意力核函数" | 计算注意力时不在 HBM 中生成 N×N 矩阵。 |
| 连续批处理 | "不等待的批处理" | 把完成的序列换出、新序列换入，无需清空整个批次。 |
| PagedAttention | "vLLM 的招牌" | KV 缓存按固定块分配并配以页表；消除碎片化。 |
| 前缀缓存 | "复用长提示词" | 跨请求缓存共享前缀的 KV；对智能体场景是重大的成本削减。 |
| 投机解码 | "草稿 + 验证" | 廉价的草稿模型提议 token；大模型一次前向验证 k 个。 |

## 延伸阅读

- [Dao et al. (2022). FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness](https://arxiv.org/abs/2205.14135) — Flash 1。
- [Dao (2023). FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning](https://arxiv.org/abs/2307.08691) — Flash 2。
- [Shah et al. (2024). FlashAttention-3: Fast and Accurate Attention with Asynchrony and Low-precision](https://arxiv.org/abs/2407.08608) — Flash 3。
- [FlashAttention-4 release notes (Dao-AILab, 2026)](https://github.com/Dao-AILab/flash-attention) — Blackwell 五级流水线与软件 exp2 技巧；本课提到的"仅支持前向"发布限制详见仓库 README。
- [Kwon et al. (2023). Efficient Memory Management for Large Language Model Serving with PagedAttention](https://arxiv.org/abs/2309.06180) — vLLM 论文。
- [Leviathan et al. (2023). Fast Inference from Transformers via Speculative Decoding](https://arxiv.org/abs/2211.17192) — 投机解码。
- [Li et al. (2024). EAGLE: Speculative Sampling Requires Rethinking Feature Uncertainty](https://arxiv.org/abs/2401.15077) — EAGLE-1/2 论文，即本课引用的集成式草稿方案。
- [Cai et al. (2024). Medusa: Simple LLM Inference Acceleration Framework with Multiple Decoding Heads](https://arxiv.org/abs/2401.10774) — 与 EAGLE 并列提及的 Medusa 方案。
- [vLLM docs — PagedAttention](https://docs.vllm.ai/en/latest/design/kernel/paged_attention.html) — 关于 16-token 块和页表设计的权威深度解读。
