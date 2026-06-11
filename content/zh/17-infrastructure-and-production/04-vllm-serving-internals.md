# vLLM 服务内部机制：PagedAttention、连续批处理与分块预填充

> vLLM 在 2026 年的统治地位来自三个相互叠加的默认机制，而不是某个单一技巧。PagedAttention 始终开启。连续批处理（continuous batching）在每次解码迭代之间把新请求注入正在运行的批次。分块预填充（chunked prefill）把长提示词切片，避免解码 token 被饿死。三者全开后，单张 H100 SXM5 上的 Llama 3.3 70B FP8 在 128 并发下可以跑到 2,200-2,400 tok/s——比 vLLM 自身的默认配置高约 25%，是朴素 PyTorch 循环的 3-4 倍。本课将带你把调度器和注意力内核读到能画出架构图的程度，最后在 `code/main.py` 中实现一个玩具版连续批处理器，按照 vLLM 的方式调度预填充和解码。

**Type:** Learn
**Languages:** Python (stdlib, toy continuous batching scheduler)
**Prerequisites:** Phase 17 · 01 (Model Serving), Phase 11 (LLM Engineering)
**Time:** ~75 minutes

## 学习目标

- 把 PagedAttention 解释为一个 KV 缓存分配器：块（block）、块表（block table），以及为什么生产负载下碎片率能保持在 4% 以下。
- 在迭代层面画出连续批处理的流程图：已完成的序列如何离开批次、新序列如何在不清空批次的情况下加入。
- 用一句话描述分块预填充，并说出它保护的是哪个延迟指标（提示：是 TTFT 尾部延迟，不是平均吞吐量）。
- 说出 2026 年 vLLM v0.18.0 中那个会坑到"把所有优化一次性全开"的团队的陷阱。

## 问题背景

朴素的 PyTorch 服务循环一次只处理一个请求：分词、预填充、解码直到 EOS、返回。一个用户时这没问题。一百个用户时，这就是一条耐心排队的长龙。显而易见的修复方案——静态批处理——会把每个请求填充到窗口内最长提示词的长度，把每次解码填充到最长预期输出的长度，并让整个批次卡在最慢的序列上。你为从未用上的填充付费，快请求还要等慢请求。

vLLM 一次性解决了三个问题。PagedAttention 阻止 KV 缓存碎片像经典连续分配那样吃掉 60-80% 的 GPU 显存。连续批处理让请求可以在每次解码迭代之间加入和离开批次，使批次始终装满真实的工作。分块预填充把一个 32k token 的提示词切成约 512 token 的切片，与解码交错执行，这样一个长提示词就不会冻结 GPU 上所有其他的解码 token。

2026 年的生产默认配置是三者全开。你需要理解每个机制各自做了什么，因为失败模式全都出在调度器上，而不是模型上。

## 核心概念

### 把 PagedAttention 看作虚拟内存系统

每个序列的 KV 缓存大小是 `num_layers × 2 × num_heads × head_dim × seq_len × bytes_per_element`。对于 8192 token 长度的 Llama 3.3 70B，BF16 下每个序列约 1.25 GB。如果你为每个请求预留 8192 个槽位，但平均请求只用 1500 个 token，那么预留的 HBM 大约浪费 82%。经典批处理就要付出这种浪费。

PagedAttention 借鉴了操作系统虚拟内存的思想。KV 缓存不再按序列连续存放，而是按固定大小的块分配（默认 16 个 token）。每个序列都有一张块表，把它的逻辑 token 位置映射到物理块 ID。当序列长度超出已分配的块时，再添加一个块。序列结束时，它的块归还到池中。

碎片率从 60-80%（经典分配）降到 4% 以下（PagedAttention）。你不需要用任何标志来启用 PagedAttention——它是 vLLM 唯一内置的分配器。可调的旋钮是 `--gpu-memory-utilization`（默认 0.9），它告诉 vLLM 在加载权重和激活之后，要为 KV 块预留多少比例的 HBM。

### 迭代层面的连续批处理

旧式的"动态批处理"会等待一个时间窗口（比如 10 ms）来凑满一个批次，然后执行预填充 + 解码 + 解码 + 解码，直到所有序列结束。快序列提前完成后只能闲置，等 GPU 跑完慢序列。

连续批处理在每个解码步之间运行。把正在运行的序列集合称为 `RUNNING` 列表。在每次迭代中：

1. `RUNNING` 中刚命中 EOS 或 max_tokens 的序列被移除。
2. 调度器查看等待队列。如果有空闲的 KV 块，就接纳新序列（预填充或恢复执行）。
3. 前向传播在当前 `RUNNING` 中的所有序列上运行，每个序列产出一个新 token。

批次大小从不填充到固定数量。处于输出不同位置的序列共享一次融合的前向计算。在 2026 年的 vLLM 中，这被称为 `V1 scheduler`。关键不变量是：调度器每个解码迭代运行一次，而不是每个请求运行一次。

### 分块预填充保护 TTFT 尾部延迟

预填充是计算密集型的。一个 32k token 的提示词在单张 H100 上跑 Llama 3.3 70B 的纯预填充需要约 800 ms。预填充运行期间，批次中所有其他序列的解码 token 都在等待。在服务循环里，一个长提示词的首 token 延迟（TTFT）就变成了几十个其他用户的 token 间延迟（ITL）尖刺。

分块预填充把预填充切成固定大小的块（默认 512 个 token），并把每个块作为一个调度单元。在块与块之间，调度器可以让解码序列各前进一个 token。你用一点点绝对预填充延迟的代价（每块几毫秒），换来解码时抖动的大幅降低。在公开基准测试中，混合负载下的 P99 ITL 从约 50 ms 降到约 15 ms。

### 三个默认机制相互依存

这三个特性彼此互为前提。PagedAttention 给调度器提供了一种细粒度的 KV 资源用于权衡取舍。连续批处理需要这种细粒度资源，这样接纳一个新序列才不会引发全局重排。分块预填充则是调度器在同一个 `RUNNING` 列表上做出的决策——它只是多了一条调度策略，而不是一个独立的系统。

你不需要记住每一个标志。你需要知道调度器在优化什么：在 KV 块预算约束下、受分块预填充切片约束的有效吞吐量（goodput）。

### 2026 年 v0.18.0 的陷阱

在 vLLM v0.18.0 中，你不能把 `--enable-chunked-prefill` 与草稿模型投机解码（`--speculative-model`）组合使用。文档中注明的例外是 V1 调度器中的 N-gram GPU 投机解码。不读发布说明就把所有标志全开的团队会在启动时收到运行时错误，而不是悄无声息的性能退化。如果你的投机解码收益值得你为之启用分块预填充，请重新审视这个选择——2026 年的正确答案往往是 EAGLE-3 不配分块预填充，而不是草稿模型加一个根本跑不起来的分块预填充。

### 你应该记住的数字

- Llama 3.3 70B FP8，H100 SXM5，128 并发，三者全开：2,200-2,400 tok/s。
- 同样的模型，vLLM 默认配置（无分块预填充）：约 1,800 tok/s。
- 同样的模型，朴素 PyTorch 前向循环：约 600 tok/s。
- 生产负载下 PagedAttention 的 KV 碎片浪费：<4%。
- 混合负载下的 P99 ITL：开分块预填充约 15 ms，不开约 50 ms。

### 调度器长什么样

```
while True:
    finished = [s for s in RUNNING if s.is_done()]
    for s in finished: release_blocks(s); RUNNING.remove(s)

    while WAITING and have_free_blocks_for(WAITING[0]):
        s = WAITING.pop(0)
        allocate_initial_blocks(s)
        RUNNING.append(s)

    # schedule prefill chunks + decode in one batch
    batch = []
    for s in RUNNING:
        if s.in_prefill:
            batch.append(next_prefill_chunk(s))   # e.g. 512 tokens
        else:
            batch.append(decode_one_token(s))     # 1 token

    run_forward(batch)                            # one fused GPU call
```

`code/main.py` 就是这个循环的 stdlib Python 实现，使用模拟的 token 计数和模拟的前向延迟。运行它可以看到分块预填充如何在一个长预填充期间让解码序列保持活跃。

```figure
tensor-parallel
```

## 生产实践

`code/main.py` 模拟了一个 vLLM 风格的调度器，各特性可以独立开关。运行它可以观察：

- `NAIVE` 模式：一次一个请求，没有批处理。
- `STATIC` 模式：填充并等待，经典批处理。
- `CONTINUOUS` 模式：迭代级的接纳与释放。
- `CONTINUOUS + CHUNKED` 模式：预填充切片与解码交错执行。

输出会显示总吞吐量（每虚拟秒的 token 数）、TTFT 均值和 P99 ITL。在混合流量下，`CONTINUOUS + CHUNKED` 这一行应该全面占优。

## 交付产物

本课产出 `outputs/skill-vllm-scheduler-reader.md`。给定一份服务配置（批次大小、KV 显存利用率、分块预填充大小、投机解码配置），它会生成一份调度器诊断报告，指出三个默认机制中哪一个是瓶颈，以及应该调什么参数。

## 练习

1. 运行 `code/main.py`。在长短请求混合的工作负载上比较 `STATIC` 和 `CONTINUOUS`。吞吐量差距来自哪里——预填充效率、解码效率，还是尾部延迟？
2. 修改玩具调度器，加入 `--max-num-batched-tokens`。对于运行 Llama 3.3 70B FP8 的 H100，合适的取值是多少？（提示：它是 KV 块大小和空闲块数量的函数，而不是裸 HBM 容量的函数。）
3. 重读 vLLM v0.18.0 的发布说明。哪些标志组合是互斥的？把它们列出来。
4. 对一条包含 1,000 个请求的轨迹（输出 token 均值 1,500、标准差 600），分别计算两种情况下的 KV 缓存碎片浪费：(a) 按 8192 上限的逐请求连续分配，(b) 使用 16 token 块的 PagedAttention。
5. 用一段话解释为什么分块预填充单独使用时能改善 P99 ITL 但不能提升吞吐量。实践中的吞吐量收益又来自哪里？

## 关键术语

| 术语 | 大家怎么叫 | 实际含义 |
|------|----------------|------------------------|
| PagedAttention | "那个 KV 技巧" | KV 缓存的固定大小块分配器；碎片率 <4% |
| Block table | "页表" | 每序列一张，从逻辑 token 位置到物理 KV 块的映射 |
| Continuous batching | "做对了的动态批处理" | 每个解码迭代都做接纳/释放决策 |
| Chunked prefill | "预填充切分" | 把长预填充切成 512 token 的切片，与解码交错执行 |
| TTFT | "首 token 时间" | 预填充 + 排队 + 网络；长提示词下由预填充主导 |
| ITL | "token 间延迟" | 相邻解码 token 之间的时间；由批次大小主导 |
| Goodput | "满足 SLO 的吞吐量" | 每个请求仍命中 TTFT 和 ITL 目标前提下的 tokens/sec |
| V1 scheduler | "新调度器" | vLLM 的 2026 调度器；N-gram 投机解码是与分块预填充兼容的路径 |
| `--gpu-memory-utilization` | "显存旋钮" | 加载权重和激活之后为 KV 块预留的 HBM 比例 |

## 延伸阅读

- [vLLM documentation — Speculative Decoding](https://docs.vllm.ai/en/latest/features/spec_decode/) — 关于分块预填充与投机解码兼容性的官方来源。
- [vLLM Release Notes (NVIDIA)](https://docs.nvidia.com/deeplearning/frameworks/vllm-release-notes/index.html) — 2026 年的发布节奏和各版本特有行为。
- [vLLM Blog — PagedAttention](https://blog.vllm.ai/2023/06/20/vllm.html) — 至今仍定义着如何理解这个分配器的原始博文。
- [PagedAttention paper (arXiv:2309.06180)](https://arxiv.org/abs/2309.06180) — 碎片分析与调度器设计。
- [Aleksa Gordic — Inside vLLM](https://www.aleksagordic.com/blog/vllm) — 带火焰图的 V1 调度器详细解读。
