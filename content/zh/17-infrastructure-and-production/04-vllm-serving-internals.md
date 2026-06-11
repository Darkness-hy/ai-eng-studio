# vLLM 服务内幕：PagedAttention、连续批处理与分块预填充

> vLLM 在 2026 年的统治地位建立在三个相互叠加的默认机制之上，而非某个单一技巧。PagedAttention 始终开启。连续批处理（continuous batching）在解码迭代之间把新请求注入活跃批次。分块预填充（chunked prefill）把长提示词切片，让解码 token 永远不会被饿死。三者全开后，一张 H100 SXM5 上的 Llama 3.3 70B FP8 在 128 并发下能跑出 2,200-2,400 tok/s——比 vLLM 自身默认配置高约 25%，是朴素 PyTorch 循环的 3-4 倍。本课会把调度器和注意力内核读到你能画出示意图的程度，最后在 `code/main.py` 中实现一个玩具级连续批处理器，按 vLLM 的方式调度预填充和解码。

**Type:** Learn
**Languages:** Python (stdlib, toy continuous batching scheduler)
**Prerequisites:** Phase 17 · 01 (Model Serving), Phase 11 (LLM Engineering)
**Time:** ~75 minutes

## 学习目标

- 把 PagedAttention 解释为一个 KV 缓存分配器：块（block）、块表（block table），以及为什么生产负载下碎片率能保持在 4% 以下。
- 在迭代级别画出连续批处理的示意图：已完成的序列如何离开批次、新序列如何加入，且全程无需排空批次。
- 用一句话描述分块预填充，并说出它保护的是哪个延迟指标（提示：是 TTFT 尾部延迟，不是平均吞吐量）。
- 说出 2026 年 vLLM v0.18.0 的那个坑——它专咬那些把所有优化一次性全开的团队。

## 问题背景

朴素的 PyTorch 服务循环一次只处理一个请求：分词、预填充、解码到 EOS、返回。一个用户时这没问题。一百个用户时，这就是一条排满耐心等待者的队列。显而易见的修复方案——静态批处理——会把每个请求填充到窗口内最长的提示词长度，把每次解码填充到最长的预期输出长度，并让整个批次卡在最慢的序列上。你为永远用不上的填充付费，快请求还得等慢请求。

vLLM 一次解决三个问题。PagedAttention 阻止 KV 缓存碎片像经典连续分配那样吃掉 60-80% 的 GPU 显存。连续批处理让请求可以在每次解码迭代之间加入和离开批次，使批次始终装满真实的工作。分块预填充把 32k token 的提示词拆成约 512 token 的切片，与解码交错执行，于是一条长提示词不会冻结 GPU 上的所有解码 token。

2026 年的生产默认配置是三者全开。你需要理解每一项各自做了什么，因为所有的故障模式都出在调度器上，而不是模型上。

## 核心概念

### 把 PagedAttention 看作虚拟内存系统

每条序列的 KV 缓存大小是 `num_layers × 2 × num_heads × head_dim × seq_len × bytes_per_element`。对于 8192 token 的 Llama 3.3 70B，BF16 下每条序列约占 1.25 GB。如果你为每个请求预留 8192 个槽位，而平均请求只用 1500 个 token，那么你预留的 HBM 中约有 82% 被浪费。经典批处理就在支付这笔浪费。

PagedAttention 借鉴了操作系统虚拟内存的思想。KV 缓存不再按序列连续分配，而是以固定大小的块（默认 16 个 token）分配。每条序列有一张块表，把它的逻辑 token 位置映射到物理块 ID。当序列增长超过已分配的块时，就再加一个块。序列结束时，它的块归还回池中。

碎片率从 60-80%（经典分配）降到 4% 以下（PagedAttention）。你不需要用某个开关来启用 PagedAttention——它是 vLLM 唯一内置的分配器。可调的旋钮是 `--gpu-memory-utilization`（默认 0.9），它告诉 vLLM 在加载权重和激活之后，为 KV 块保留多少 HBM。

### 迭代级别的连续批处理

旧式的"动态批处理"会等待一个时间窗口（比如 10 ms）来攒满一个批次，然后跑预填充 + 解码 + 解码 + 解码，直到批内所有序列结束。快序列提前完成后只能闲置，等 GPU 处理完慢序列。

连续批处理在每个解码步之间运作。把正在运行的序列集合称为 `RUNNING` 列表。每次迭代：

1. `RUNNING` 中任何刚到达 EOS 或 max_tokens 的序列被移除。
2. 调度器查看等待队列。如果有空闲 KV 块，就接纳新序列（预填充或恢复执行）。
3. 前向传播在当前 `RUNNING` 中的所有序列上运行，每条序列产出一个新 token。

批次大小从不填充到固定数值。处于各自输出不同位置的序列共享同一次融合前向计算。在 2026 年的 vLLM 中，这套机制叫 `V1 scheduler`。关键不变式：调度器每个解码迭代运行一次，而不是每个请求运行一次。

### 分块预填充保护的是 TTFT 尾部延迟

预填充是计算受限的。一条 32k token 的提示词在 Llama 3.3 70B 上需要约 800 ms 的纯预填充时间（单张 H100）。预填充运行期间，批内所有其他序列的解码 token 都在等待。在服务循环中，一条长提示词的首 token 延迟（TTFT）会变成几十个其他用户的 token 间延迟（ITL）尖刺。

分块预填充把预填充拆成固定大小的块（默认 512 个 token），并把每个块作为一个调度单元。在块与块之间，调度器可以让解码序列前进一个 token。你用一点点绝对预填充延迟（每块几毫秒）换来低得多的解码时抖动。在已发表的基准测试中，混合负载下的 P99 ITL 从约 50 ms 降到约 15 ms。

### 三个默认机制相互依赖

这三个特性彼此假设对方存在。PagedAttention 给调度器提供了一种可供权衡取舍的细粒度 KV 资源。连续批处理需要这种细粒度资源，这样接纳一条新序列才不会强制全局重排。分块预填充是调度器在同一个 `RUNNING` 列表上做出的决策——它只是多了一条调度策略，而不是一个独立的系统。

你不需要记住每个开关。你需要知道调度器优化的是什么：在 KV 块预算约束下、受分块预填充切片影响的有效吞吐量（goodput）。

### 2026 年 v0.18.0 的坑

在 vLLM v0.18.0 中，你不能把 `--enable-chunked-prefill` 与草稿模型投机解码（`--speculative-model`）组合使用。文档中标明的例外是 V1 调度器里的 N-gram GPU 投机解码。那些不读发布说明就把所有开关全打开的团队，得到的是启动时的运行时报错，而不是悄无声息的性能回退。如果你的投机解码收益值得为它启用分块预填充，请重新审视这个选择——2026 年的正确答案往往是不带分块预填充的 EAGLE-3，而不是"草稿模型 + 分块预填充"这个根本跑不起来的组合。

### 你应该记住的数字

- Llama 3.3 70B FP8，H100 SXM5，128 并发，三者全开：2,200-2,400 tok/s。
- 同一模型，vLLM 默认配置（无分块预填充）：约 1,800 tok/s。
- 同一模型，朴素 PyTorch 前向循环：约 600 tok/s。
- 生产负载下 PagedAttention 的 KV 碎片浪费：<4%。
- 混合负载下的 P99 ITL：开启分块预填充约 15 ms，不开约 50 ms。

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

`code/main.py` 就是这个循环的 stdlib Python 实现，用假的 token 计数和假的前向延迟。运行它能看到分块预填充如何在长预填充期间让解码序列保持活跃。

```figure
tensor-parallel
```

## 生产实践

`code/main.py` 模拟了一个带可切换特性的 vLLM 式调度器。运行它可以看到：

- `NAIVE` 模式：一次一个请求，无批处理。
- `STATIC` 模式：填充并等待，经典批处理。
- `CONTINUOUS` 模式：迭代级别的接纳与释放。
- `CONTINUOUS + CHUNKED` 模式：预填充切片与解码交错执行。

输出显示总吞吐量（每虚拟秒的 token 数）、TTFT 均值和 P99 ITL。在混合流量下，`CONTINUOUS + CHUNKED` 这一行应当全面占优。

## 交付产物

本课产出 `outputs/skill-vllm-scheduler-reader.md`。给定一份服务配置（批次大小、KV 显存利用率、分块预填充大小、投机解码配置），它会产出一份调度器诊断报告，指出三个默认机制中哪一个是瓶颈以及该调什么。

## 练习

1. 运行 `code/main.py`。在一个长短请求混合的工作负载上比较 `STATIC` 与 `CONTINUOUS`。吞吐量差距来自哪里——预填充效率、解码效率，还是尾部延迟？
2. 修改玩具调度器，加入 `--max-num-batched-tokens`。对于运行 Llama 3.3 70B FP8 的 H100，合理的值是多少？（提示：它是 KV 块大小和空闲块数量的函数，不是裸 HBM 容量的函数。）
3. 重读 vLLM v0.18.0 的发布说明。哪些开关组合是互斥的？把它们列出来。
4. 计算一条包含 1,000 个请求的轨迹的 KV 缓存碎片浪费，输出 token 均值 1,500、标准差 600，分别在 (a) 按请求连续分配且最大 8192，(b) PagedAttention 配 16 token 块两种方案下。
5. 用一段话解释为什么分块预填充单独使用能改善 P99 ITL 却不能提升吞吐量。实践中的吞吐量收益从何而来？

## 关键术语

| 术语 | 大家怎么说 | 实际含义 |
|------|----------------|------------------------|
| PagedAttention | "那个 KV 技巧" | KV 缓存的固定大小块分配器；碎片率 <4% |
| 块表（Block table） | "页表" | 每序列一张，从逻辑 token 位置到物理 KV 块的映射 |
| 连续批处理（Continuous batching） | "动态批处理，但做对了" | 接纳/释放决策在每个解码迭代都执行一次 |
| 分块预填充（Chunked prefill） | "预填充切分" | 把长预填充拆成 512 token 的切片，与解码交错执行 |
| TTFT | "首 token 时间" | 预填充 + 排队 + 网络；长提示词下由预填充主导 |
| ITL | "token 间延迟" | 相邻解码 token 之间的时间；由批次大小主导 |
| 有效吞吐量（Goodput） | "满足 SLO 的吞吐量" | 每个请求都仍达到 TTFT 和 ITL 目标前提下的 tokens/sec |
| V1 调度器 | "新调度器" | vLLM 的 2026 调度器；N-gram 投机解码是与分块预填充兼容的路径 |
| `--gpu-memory-utilization` | "那个显存旋钮" | 权重和激活之外，为 KV 块保留的 HBM 比例 |

## 延伸阅读

- [vLLM documentation — Speculative Decoding](https://docs.vllm.ai/en/latest/features/spec_decode/) — 关于分块预填充与投机解码兼容性的官方信息源。
- [vLLM Release Notes (NVIDIA)](https://docs.nvidia.com/deeplearning/frameworks/vllm-release-notes/index.html) — 2026 年的发布节奏与各版本特有行为。
- [vLLM Blog — PagedAttention](https://blog.vllm.ai/2023/06/20/vllm.html) — 最初的官方文章，至今仍定义着理解这个分配器的方式。
- [PagedAttention paper (arXiv:2309.06180)](https://arxiv.org/abs/2309.06180) — 碎片分析与调度器设计。
- [Aleksa Gordic — Inside vLLM](https://www.aleksagordic.com/blog/vllm) — 带火焰图的 V1 调度器详细解读。
