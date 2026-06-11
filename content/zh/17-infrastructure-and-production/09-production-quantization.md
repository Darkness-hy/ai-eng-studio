# 生产环境量化 — AWQ、GPTQ、GGUF K-quants、FP8、MXFP4/NVFP4

> 量化格式没有放之四海而皆准的选择——它取决于硬件、推理引擎和工作负载。GGUF Q4_K_M 或 Q5_K_M 通过 llama.cpp 和 Ollama 交付，统治 CPU 和边缘场景。当你需要在同一个基座模型上挂多个 LoRA 时，GPTQ 在 vLLM 中胜出。AWQ 配合 Marlin-AWQ 内核在 7B 级别模型上达到约 741 tok/s，且在 INT4 格式中 Pass@1 最高——是 2026 年数据中心生产环境的默认选择。FP8 在 Hopper、Ada 和 Blackwell 上保持中间地带——接近无损且支持广泛。NVFP4 和 MXFP4（Blackwell 微缩放格式）较为激进，需要逐块验证。有两个陷阱常坑团队：校准数据集必须匹配部署领域；KV 缓存与权重量化是两回事——「我的模型现在只有 4 GB 了」这种 AWQ 式的想法忘了生产批量大小下 10-30 GB 的 KV 缓存。

**Type:** Learn
**Languages:** Python (stdlib, toy memory and throughput comparison across formats)
**Prerequisites:** Phase 10 · 13 (Quantization foundations), Phase 17 · 04 (vLLM Serving Internals)
**Time:** ~75 minutes

## 学习目标

- 说出 2026 年六种生产级量化格式及各自的最佳适用场景。
- 根据硬件（CPU 还是 GPU、Hopper 还是 Blackwell）、引擎（vLLM、TRT-LLM、llama.cpp）和工作负载（日常聊天、推理任务、多 LoRA）选择格式。
- 计算选定格式节省的权重显存，以及不受影响的 KV 缓存。
- 说出会让量化模型在领域流量上掉点的校准数据集陷阱。

## 问题背景

量化（quantization）降低显存占用和 HBM 带宽消耗，而这正是解码阶段最需要的。一个 FP16 的 70B 模型权重有 140 GB。把权重量化到 INT4（AWQ 或 GPTQ），模型就缩到 35 GB——可以放进一张 H100，还能留出 KV 缓存的空间。这一点很关键，因为在 128 路并发、2k 上下文时，仅 KV 缓存就要 20-30 GB。

但量化不是免费的。激进的量化会损害质量，尤其是在推理密集型任务上。不同格式适配不同引擎，不同硬件原生支持的精度也不同。2026 年的格式大观园是真实存在的，你不能照抄别人的选择——必须基于自己的技术栈来挑。

## 核心概念

### 六种格式

| 格式 | 位数 | 最佳场景 | 引擎 |
|--------|------|-----------|---------|
| GGUF Q4_K_M / Q5_K_M | 4-5 | CPU、边缘设备、笔记本 | llama.cpp, Ollama |
| GPTQ | 4-8 | vLLM 上的多 LoRA | vLLM, TGI |
| AWQ | 4 | 数据中心 GPU 生产环境 | vLLM (Marlin-AWQ), TGI |
| FP8 | 8 | Hopper/Ada/Blackwell 数据中心 | vLLM, TRT-LLM, SGLang |
| MXFP4 | 4 | Blackwell 多用户场景 | TRT-LLM |
| NVFP4 | 4 | Blackwell 多用户场景 | TRT-LLM |

### GGUF —— CPU/边缘场景的默认选择

GGUF 严格来说是一种文件格式，而非量化方案——它把多个 K-quant 变体（Q2_K、Q3_K_M、Q4_K_M、Q5_K_M、Q6_K、Q8_0）打包在一个容器里。Q4_K_M 和 Q5_K_M 是生产默认配置——以 4-5 比特实现接近 BF16 的质量。它是 CPU 或边缘服务的最佳选择，因为 llama.cpp 是目前遥遥领先的最快 CPU 推理引擎。

在 vLLM 中存在吞吐惩罚：7B 模型约 93 tok/s——该格式并未针对 GPU 内核做优化。部署目标是 CPU/边缘时用 GGUF，其他情况不用。

### GPTQ —— vLLM 中的多 LoRA 方案

GPTQ 是一种带校准步骤的训练后量化（post-training quantization）算法。Marlin 内核让它在 GPU 上跑得很快（比非 Marlin 的 GPTQ 提速 2.6 倍）。7B 模型约 712 tok/s。

它的独特优势：GPTQ-Int4 在 vLLM 中支持 LoRA 适配器。如果你要服务一个基座模型加 10-50 个微调变体（每个都是一个 LoRA），GPTQ 就是你的路径。截至 2026 年初，NVFP4 还不支持 LoRA。

### AWQ —— 数据中心 GPU 的默认选择

激活感知权重量化（Activation-aware Weight Quantization）。在量化过程中保护约 1% 最显著的权重。Marlin-AWQ 内核：比朴素实现提速 10.9 倍。7B 模型约 741 tok/s，在 INT4 格式中 Pass@1 最高。

新的 GPU 服务部署优先选 AWQ，除非你需要多 LoRA（选 GPTQ）或激进的 Blackwell FP4（选 NVFP4）。

### FP8 —— 可靠的中间方案

8 位浮点。接近无损。支持广泛。Hopper Tensor Core 原生加速 FP8，Blackwell 继承了这一能力。当质量不容妥协时（推理、医疗、代码生成），FP8 是 2026 年的安全默认选择。显存节省只有 INT4 的一半，但质量风险低得多。

### MXFP4 / NVFP4 —— Blackwell 上的激进方案

微缩放（microscaling）FP4。每个权重块拥有独立的缩放因子。激进，但在 Blackwell Tensor Core 上有硬件加速。相比 FP8，每 token 的字节数再减半——这正是 Phase 17 · 07 中讲的经济账。

注意事项：
- 暂不支持 LoRA（2026 年初）。
- 在推理密集型工作负载上质量下降明显。
- 每个模型都要在你自己的评估集上验证。

### 校准陷阱

AWQ 和 GPTQ 需要校准数据集——通常是 C4 或 WikiText。对领域模型（代码、医疗、法律）来说，用通用网络文本做校准会让算法对该保护哪些权重做出错误判断。HumanEval 上的 Pass@1 可能掉好几个点。

解决办法：用领域内数据做校准。几百条领域样本通常就够了。上线前在评估集上测试。

### KV 缓存陷阱

AWQ 把权重压缩到 4 比特。KV 缓存是另一回事，仍然保持 FP16/FP8。以一个用 AWQ 的 70B 模型为例：

- 权重：约 35 GB（从 140 GB 压到 INT4）。
- 128 路并发 × 2k 上下文的 KV 缓存：约 20 GB。
- 激活值：约 5 GB。
- 合计：约 60 GB——能放进 H100 80GB。

天真地说「我把模型量化到 4 GB 了」会忘掉另外 30-50 GB。要对 HBM 做整体预算。

另外，KV 缓存量化（FP8 KV 或 INT8 KV）是一个有自己权衡的独立选择——它直接影响注意力精度，并非白赚的收益。

### AWQ INT4 对推理任务有风险

思维链、数学、长上下文代码生成——这些任务在激进量化下会有明显损失。AWQ INT4 在 MATH 上约掉 3-5 个点。对推理密集型工作负载，上 FP8 或 BF16，接受显存代价。

### 2026 选型指南

- CPU/边缘服务：GGUF Q4_K_M。搞定。
- GPU 服务、日常聊天、无 LoRA：AWQ。
- GPU 服务、多 LoRA：GPTQ 配 Marlin。
- 推理工作负载：FP8。
- Blackwell 数据中心、质量已验证：NVFP4 + FP8 KV。
- 拿不准：在每个候选格式上跑 1,000 条样本的评估。

```figure
gpu-memory-breakdown
```

## 生产实践

`code/main.py` 针对一系列模型规模，计算六种格式的显存占用（权重 + KV + 激活值）和相对吞吐。它展示了 KV 缓存在哪些场景占主导、权重压缩在哪些场景划算，以及哪些场景 FP8 是安全之选。

## 交付产物

本课产出 `outputs/skill-quantization-picker.md`。给定硬件、模型规模、工作负载类型和质量容忍度，它会选出一种格式，并生成一份校准/验证计划。

## 练习

1. 运行 `code/main.py`。对一个 128 路并发、2k 上下文的 70B 模型，计算每种格式的总 HBM 占用。哪种格式能让你装进一张 H100 80GB？
2. 你有一个 7B 编程模型。选一种格式并给出理由。如果你对质量容忍度判断错了，恢复路径是什么？
3. 计算为医疗领域模型校准 AWQ 所需的校准数据集规模。为什么数据不是越多越好？
4. 阅读 Marlin-AWQ 内核的论文或发布说明。用三句话解释为什么 AWQ 在 7B 上能达到 741 tok/s，而原始 GPTQ 只有约 712。
5. 什么时候把 AWQ 权重和 FP8 KV 缓存组合使用是合理的，什么时候应该让 KV 保持 BF16？

## 关键术语

| 术语 | 大家怎么说 | 实际含义 |
|------|----------------|------------------------|
| GGUF | 「llama.cpp 格式」 | 打包多个 K-quant 变体的文件格式；CPU/边缘场景默认选择 |
| Q4_K_M | 「Q4 K M」 | 4 位 K-quant 中等档位；GGUF 的生产默认配置 |
| GPTQ | 「gee pee tee q」 | 带校准的训练后 INT4 量化；在 vLLM 中支持 LoRA |
| AWQ | 「a w q」 | 激活感知 INT4；Marlin 内核；INT4 中 Pass@1 最高 |
| Marlin 内核 | 「快速 INT4 内核」 | Hopper 上的 INT4 定制 CUDA 内核；提速 10 倍 |
| FP8 | 「8 位浮点」 | Hopper/Ada/Blackwell 上的安全精度默认选择 |
| MXFP4 / NVFP4 | 「微缩放 4 位」 | Blackwell 的 4 位浮点，带逐块缩放因子 |
| 校准数据集 | 「cal data」 | 用于确定量化参数的输入文本；必须匹配领域 |
| KV 缓存量化 | 「KV INT8」 | 与权重量化相互独立的选择；直接影响注意力精度 |

## 延伸阅读

- [VRLA Tech — LLM Quantization 2026](https://vrlatech.com/llm-quantization-explained-int4-int8-fp8-awq-and-gptq-in-2026/) — 各格式对比基准。
- [Jarvis Labs — vLLM Quantization Complete Guide](https://jarvislabs.ai/blog/vllm-quantization-complete-guide-benchmarks) — 按格式列出的吞吐数据。
- [PremAI — GGUF vs AWQ vs GPTQ vs bitsandbytes 2026](https://blog.premai.io/llm-quantization-guide-gguf-vs-awq-vs-gptq-vs-bitsandbytes-compared-2026/) — 逐格式的选型指南。
- [vLLM docs — Quantization](https://docs.vllm.ai/en/latest/features/quantization/index.html) — 支持的格式与参数标志。
- [AWQ paper (arXiv:2306.00978)](https://arxiv.org/abs/2306.00978) — AWQ 的原始论文。
- [GPTQ paper (arXiv:2210.17323)](https://arxiv.org/abs/2210.17323) — GPTQ 的原始论文。
