# SGLang 与 RadixAttention：应对前缀密集型工作负载

> SGLang 把 KV 缓存当作一等公民的可复用资源，存储在一棵基数树（radix tree）中。vLLM 按 FCFS（先来先服务）调度请求，而 SGLang 的缓存感知调度器会优先处理共享前缀更长的请求——实际上相当于对基数树做深度优先遍历，让热点分支始终驻留在 HBM 中。在 Llama 3.1 8B 上运行类 ShareGPT 的 1K prompt 工作负载时，SGLang 达到约 16,200 tok/s，而 vLLM 约为 12,500，领先约 29%。在前缀密集的 RAG 工作负载上，优势可达 6.4 倍。在语音克隆类工作负载上，缓存命中率超过 86%。2026 年已部署在 xAI、LinkedIn、Cursor、Oracle、GCP、Azure、AWS 等超过 40 万张 GPU 上。需要注意的陷阱是：一旦前缀顺序不一致，6.4 倍的数字就会化为乌有——前缀顺序正是工程师手里的杠杆。

**Type:** Learn
**Languages:** Python (stdlib, toy radix-tree cache + cache-aware scheduler)
**Prerequisites:** Phase 17 · 04 (vLLM Serving Internals), Phase 14 (Agentic RAG)
**Time:** ~75 minutes

## 学习目标

- 画出 RadixAttention 的工作原理图：前缀如何存储在基数树中，根植于同一分支的多条序列如何共享 KV 块。
- 解释缓存感知调度（cache-aware scheduling），以及为什么 FCFS 不适合前缀密集型流量。
- 在给定前缀缓存命中率和 prompt 长度分布的情况下，计算某个工作负载的预期加速比。
- 说出让 6.4 倍数字真正落地（而不是白白错失）所需的 prompt 顺序纪律。

## 问题背景

传统的推理服务把每个请求的 prompt 当作不透明的整体。即使 5,000 个 RAG 请求都以同一段 2,000 token 的系统提示加同一段检索前导开头，vLLM 也会把这段 2,000 token 的前缀预填充（prefill）5,000 次。GPU 在一遍又一遍地做同样的工作。

关键观察是：智能体和 RAG 工作负载中的 prompt 几乎总是共享很长的前缀。系统提示、工具 schema、少样本示例、检索头部、对话历史——这些都会在请求之间重复出现。如果把这段前缀的 KV 缓存存储一次并复用，就不需要再为它做预填充。

RadixAttention 做的正是这件事。token 被索引在一棵基数树中；每个节点拥有从根节点到该节点路径上的 token 序列所对应的 KV 块。新请求到来时沿树向下遍历：任何 token 匹配的节点，其 KV 块都可以直接复用。预填充的开销变成与"新增"后缀成正比，而不是与完整 prompt 成正比。

挑战在于调度。如果两个请求共享一段 2,000 token 的前缀，第三个请求只共享其中 200 个 token，你会希望把那两个长共享请求放在一起处理，让长前缀留在 HBM 中。FCFS 恰恰相反——谁先到先服务谁，结果可能在下一个长前缀请求到达之前，热点分支已经被驱逐了。

## 核心概念

### 基数树作为 KV 索引

基数树（radix tree，即压缩前缀树 compact trie）存储 token 序列。每个节点拥有一段 token 范围以及为该范围计算出的 KV 块。子节点在序列上延伸一个或多个 token。

```
root
 |- "You are a helpful assistant..."  (2,000 tokens, 124 KV blocks)
      |- "Context: <doc A>..."        (500 tokens, 31 blocks)
           |- "Question: Alice..."    (80 tokens, 5 blocks)
           |- "Question: Bob..."      (95 tokens, 6 blocks)
      |- "Context: <doc B>..."        (520 tokens, 33 blocks)
```

一个新请求带着 系统提示 + "Context: <doc A>" + "Question: Carol" 到来。调度器沿树遍历：系统前缀匹配（复用 124 个块），doc-A 分支匹配（复用 31 个块），然后只为 "Question: Carol" 分配新块（4 个块）。预填充开销：4 个块的新 token。没有这棵树的话：160 个块。预填充节省约 40 倍。

### 缓存感知调度

如果缓存频繁颠簸（churn），基数树支撑的复用就毫无意义。两条关键策略：

1. **深度优先派发**。从队列中挑选下一个请求时，优先选择与当前运行集合根植于同一分支的请求。这样可以把热点分支固定住。
2. **在分支级别而非块级别做 LRU**。驱逐整条分支（从最久未使用的叶子开始），而不是逐个驱逐块，让缓存的形状与基数树的形状保持一致。

FCFS 把这两条都破坏了。一个共享 2,000 token 的请求排在一个只共享 50 token 的请求后面，然后那条 2,000 token 的分支被驱逐，给 50 token 的请求腾位置。

### 你应该记住的基准数字

- Llama 3.1 8B，H100，ShareGPT 1K prompt：SGLang 约 16,200 tok/s，vLLM 约 12,500（领先约 29%）。
- 前缀密集的 RAG（相同系统提示 + 相同文档，问题不同）：SGLang 最高可达 6.4 倍。
- 语音克隆工作负载：86.4% 的前缀缓存命中率。
- SGLang 客户的生产环境命中率：50-99%，取决于 prompt 纪律。
- 2026 年已部署在超过 40 万张 GPU 上。

### 顺序陷阱

6.4 倍这个数字依赖于一致的 prompt 模板顺序。如果你的客户端在一些请求中按 `[system, tools, context, history, question]` 构造 prompt，在另一些请求中按 `[system, context, tools, history, question]` 构造，基数树就找不到共享前缀。在人类看来是共享的前缀，在基数树看来是两条截然不同的序列。

工程师的杠杆：你的 prompt 模板就是缓存键。固定顺序。把所有不可变的内容（系统提示、工具、schema）放在最前面。检索上下文放在其后。用户问题放在最后。不要把动态内容穿插进前缀。

研究中的真实案例：某个部署只是把动态内容移出可缓存前缀，缓存命中率就从 7% 提升到了 74%——只改了这一处。

### RadixAttention 在哪里赢、在哪里输

赢的场景：
- RAG（相同的检索前导，问题不同）。
- 智能体（相同的工具 schema，查询不同）。
- 带有长系统提示的对话。
- 带有重复前导的语音/视觉工作负载。

输的场景（吞吐退回到 vLLM 的水平）：
- prompt 各不相同的单次生成（代码补全、没有系统提示的开放式对话）。
- 每个请求都把独特内容穿插进前缀的动态 prompt。

### 为什么这是调度器问题，而不只是 kernel 问题

KV 复用可以作为一个 kernel 技巧来实现。SGLang 的洞见是：只有调度器让热点分支保持驻留，复用才能真正带来收益。一个朴素的"有就复用"策略在混合负载下会让缓存颠簸不止。正是这个以基数树为索引的调度器，把 kernel 技巧变成了生产环境中 29% 的优势。

### 与 vLLM 的关系

这两个系统并不是严格意义上的竞争对手。2026 年，vLLM 加入了前缀缓存（`--enable-prefix-caching`）和缓存感知路由器（用 Rust 编写的 vLLM Router）。差距缩小了，但没有完全消失——SGLang 的整个技术栈是以基数树为先的设计；vLLM 则是后期嫁接上去的。对于以前缀复用为主导的工作负载，SGLang 仍然是默认选择。对于没有明显前缀模式的通用推理服务，vLLM 仍然不相上下甚至更好。

```figure
roofline
```

## 生产实践

`code/main.py` 实现了一个玩具版的基数树 KV 缓存，外加一个支持两种策略的调度器：FCFS 和缓存感知。它把同一份工作负载分别跑过两种策略，报告前缀缓存命中率和吞吐差异。然后再跑一份"顺序打乱"的工作负载，展示 6.4 倍优势如何崩塌。

## 交付产物

本课产出 `outputs/skill-radix-scheduler-advisor.md`。给定一份工作负载描述（prompt 模板形态、检索模式、并发租户数量），它会输出一份 prompt 顺序处方，以及是否采用 SGLang 的 go/no-go 结论。

## 练习

1. 运行 `code/main.py`。在同一份工作负载上对比 FCFS 和缓存感知策略。差异来自哪里——预填充节省、解码节省，还是排队延迟？
2. 修改工作负载，让 prompt 随机打乱 `[system, tools, context]` 的顺序。重新运行。命中率发生了什么变化？为什么？
3. 计算在 Llama 3.1 8B 上把一段 2,000 token 的系统提示作为一条基数树分支常驻 HBM 的开销。与不做前缀复用的 16 序列批次的开销做对比。
4. 阅读 SGLang 的 RadixAttention 论文。用三句话解释为什么在前缀密集负载下，树形 LRU 驱逐优于块形 LRU。
5. 某客户报告缓存命中率只有 8%。说出三个最可能的原因，以及针对每个原因你会执行的诊断手段。

## 关键术语

| 术语 | 大家怎么说 | 实际含义 |
|------|----------------|------------------------|
| RadixAttention | "SGLang 那个东西" | 以基数树索引的 KV 缓存，让共享前缀复用 KV 块 |
| 基数树（Radix tree） | "压缩前缀树" | 每个节点拥有一段 token 范围及其 KV 块的树结构 |
| 缓存感知调度器 | "热点分支优先" | 优先处理与驻留分支共享前缀的请求的调度器 |
| 前缀缓存命中率 | "你的 prompt 有多少是白送的" | prompt 中由复用 KV 块服务的 token 占比 |
| FCFS | "先来先服务" | 破坏前缀局部性的默认调度方式 |
| 分支级 LRU | "驱逐叶子" | 与基数树形状匹配的驱逐策略 |
| Prompt 模板顺序 | "缓存键" | prompt 各组成部分的顺序决定了树能共享什么 |
| 系统提示固定 | "驻留前缀" | 把不可变的系统部分固定住，避免驱逐颠簸 |

## 延伸阅读

- [SGLang GitHub](https://github.com/sgl-project/sglang) —— 源码与文档。
- [SGLang documentation](https://sgl-project.github.io/) —— RadixAttention 与调度细节。
- [SGLang paper — Efficiently Programming Large Language Models (arXiv:2312.07104)](https://arxiv.org/abs/2312.07104) —— 设计参考。
- [LMSYS blog — SGLang with RadixAttention](https://www.lmsys.org/blog/2024-01-17-sglang/) —— 基准数字与调度器设计依据。
- [vLLM — Prefix Caching](https://docs.vllm.ai/en/latest/features/prefix_caching.html) —— vLLM 自己的类基数树实现，可作对比。
