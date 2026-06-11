# Blackwell 上的 TensorRT-LLM：FP8 与 NVFP4

> TensorRT-LLM 只支持 NVIDIA 硬件，但它在 Blackwell 上赢麻了。在 GB200 NVL72 上配合 Dynamo 编排，SemiAnalysis 的 InferenceX 在 2026 年 Q1-Q2 测得 120B 模型每百万 token 仅 $0.012，而 H100 + vLLM 是 $0.09/M——经济性差了 7 倍。这套技术栈是三种浮点精度方案的叠加：FP8 对 KV 缓存和注意力内核仍然不可或缺，因为它们需要 FP8 的动态范围；NVFP4（4 位微缩放格式）负责权重和激活值；多 token 预测（MTP）和预填充/解码分离再叠加 2-3 倍。Day-0 模型支持让 FP4 权重可以直接加载，无需训练后转换。对 2026 年的工程团队来说，要注意的是：TRT-LLM 是 NVIDIA 的封闭技术栈，采用它意味着用可移植性换吞吐量。在做出承诺之前，先针对你的模型和硬件组合算清这笔账。

**Type:** Learn
**Languages:** Python (stdlib, toy FP8/NVFP4 memory and cost calculator)
**Prerequisites:** Phase 17 · 04 (vLLM Serving Internals), Phase 10 · 13 (Quantization)
**Time:** ~75 minutes

## 学习目标

- 解释为什么即使权重已经用 NVFP4，FP8 对 KV 缓存和注意力仍然不可或缺。
- 计算前沿模型在 BF16、FP8 和 NVFP4 下的 HBM 占用，并分析节省来自哪里。
- 说出 TRT-LLM 利用的 Blackwell 专属特性（Day-0 FP4、MTP、分离式服务、all-to-all 原语）。
- 判断什么时候 TRT-LLM 的 NVIDIA 锁定值得换取相对 Hopper 上 vLLM 的 7 倍成本差距。

## 问题背景

2026 年推理经济学的前沿问题是"每美元能产出多少 token"。答案取决于四个层层叠加的选择：硬件代际（Hopper H100/H200 还是 Blackwell B200/GB200）、精度（BF16 → FP8 → NVFP4）、服务引擎（vLLM、SGLang 还是 TRT-LLM）、编排方式（普通部署、分离式部署还是 Dynamo）。

在 Hopper 上用 vLLM，一个 120B MoE 模型的成本约为每百万 token $0.09。在 Blackwell 上用 TRT-LLM + Dynamo，同一个模型约为 $0.012——便宜 7 倍。这个差距一部分来自硬件（Blackwell 的单 GPU LLM 吞吐量是 Hopper 的 11-15 倍），一部分来自技术栈：FP4 权重、MTP 草稿、预填充/解码分离，以及用于 MoE 专家通信的 NVLink 5 all-to-all。

在 NVIDIA 技术栈之外你无法复现这一切。这就是代价——用可移植性换经济性。理解每项技术栈选择各自贡献了多少差距，正是本课的目的。

## 核心概念

### 为什么 FP8 仍是 KV 缓存的底线

2026 年一个常见的错误：以为 NVFP4 可以用在所有地方。并非如此。KV 缓存需要 FP8（8 位浮点），因为它存储的注意力键和值跨越很宽的动态范围。把 KV 量化到 FP4 会造成灾难性的精度损失——分布的尾部被截断，注意力分数随之崩塌。FP8 的指数位为 KV 缓存提供了所需的范围。

NVFP4（2025-2026）用于权重和激活值。微缩放（microscaling）：每个权重块都有自己的缩放因子，因此小块之间可以覆盖不同的动态范围，而不会有逐张量缩放带来的损失。对激活值而言，FP4 也能撑住，因为单层内的激活值范围较小。

典型的 Blackwell 配置：

- 权重：NVFP4（4 位微缩放）。
- 激活值：NVFP4。
- KV 缓存：FP8。
- 注意力累加器：FP32（保证 softmax 稳定性）。

### TRT-LLM 使用的 Blackwell 专属原语

- **Day-0 FP4 权重**：模型提供方直接发布 FP4 权重；TRT-LLM 无需训练后转换即可加载。FP4 不再需要 AWQ / GPTQ 步骤。
- **多 token 预测（MTP）**：与 EAGLE（Phase 17 · 05）思路相同，但直接集成进 TRT-LLM 构建中。
- **分离式服务（Disaggregated serving）**：预填充和解码运行在不同的 GPU 池上，KV 缓存通过 NVLink 或 InfiniBand 传输。与 Dynamo（Phase 17 · 20）思路相同。
- **All-to-all 通信原语**：NVLink 5 把 MoE 专家通信延迟相比 Hopper 降低了 3 倍。TRT-LLM 的 MoE 内核针对此做了调优。
- **NVFP4 + MXFP8 微缩放**：Blackwell Tensor Core 对缩放因子处理提供硬件加速。

### 应该记住的几个数字

- HGX B200 通过 TRT-LLM 跑 GPT-OSS-120B，成本 $0.02/M token。
- GB200 NVL72 通过 Dynamo（编排 TRT-LLM）做到 $0.012/M token。
- H100 + vLLM 在可比工作负载上约 $0.09/M token。
- TRT-LLM 三个月内的更新带来 2.8 倍吞吐提升（2026 年）。
- Blackwell 对 Hopper 的单 GPU LLM 吞吐量是 11-15 倍。
- MLPerf Inference v6.0（2026 年 4 月）：Blackwell 在所有提交的任务上全面领先。

### FP4 在质量上的真实代价

NVFP4 很激进。在重推理的工作负载上（思维链、数学、长上下文代码生成），FP4 权重会出现明显退化。逐块校准能缓解但无法消除。交付推理模型的团队通常折中采用 FP8 权重 + FP4 激活值，或者干脆留在 H200 上全程使用 FP8。

规则：在决定采用 NVFP4 权重之前，务必在你自己的评测集上验证任务质量。

### 为什么这是一个 NVIDIA 锁定的决策

TRT-LLM 是 C++ + CUDA + 闭源内核。模型必须针对特定 GPU SKU 编译。不支持 AMD、不支持 Intel、不支持 ARM。如果你的基础设施策略是多供应商，TRT-LLM 在其服务层级上就无从谈起——你仍然可以在混合硬件上用 vLLM 提供服务。如果你本来就只用 NVIDIA，那 7 倍的差距足以支付锁定的代价。

### 2026 年实用配方

对于年推理账单超过 1 亿美元的团队，继续跑 Hopper + vLLM 等于把 7-10 倍的收益留在桌上。把成本占大头的工作负载迁移到 Blackwell + TRT-LLM + Dynamo。把实验层保留在 H100 + vLLM 上以保证模型迭代速度。每个转换成 NVFP4 的模型在上生产前都要验证质量。

### 分离式部署的额外红利

TRT-LLM 的分离式服务（独立的预填充和解码池）将在 Phase 17 · 20 深入讲解。在 Blackwell 上，倍数效应是相乘的：FP4 权重 × MTP 加速 × 分离式部署 × 缓存感知路由。7 倍这个数字假设的是这套完整技术栈。

```figure
pipeline-parallel
```

## 生产实践

`code/main.py` 计算一个模型在三种技术栈下的 HBM 占用、解码吞吐量（内存受限场景）和每百万 token 成本：H100 + BF16 + vLLM、H100 + FP8 + vLLM、B200 + NVFP4/FP8 + TRT-LLM。运行它，观察叠加效应以及每项变更各自贡献的差距份额。

## 交付产物

本课产出 `outputs/skill-trtllm-blackwell-advisor.md`。给定工作负载、模型规模和年 token 量，它会判断 Blackwell + TRT-LLM 技术栈是否值得接受 NVIDIA 锁定。

## 练习

1. 运行 `code/main.py`。对一个激活参数占 30% 的 120B MoE 模型，计算 H100 BF16、H100 FP8 和 B200 NVFP4/FP8 下受内存带宽限制的解码吞吐量。最大的跃升来自哪里？
2. 一个客户每年在 H100 + vLLM 上花费 200 万美元。在 7 倍经济差距下，他们需要购买多少块 Blackwell GPU 才能在 12 个月内摊平迁移到 TRT-LLM 的成本？
3. NVFP4 权重转换后，你发现 MATH 上精度掉了 3 个点。说出两条恢复路径：一条质量优先（保留 FP8 权重），一条成本优先（用领域内数据校准）。
4. 阅读 MLPerf v6.0 推理结果。哪个任务上 Blackwell 对 Hopper 的差距最小？为什么？
5. 计算 405B 模型在 NVFP4 权重 + FP8 KV 缓存、128k 上下文下所需的 HBM。它能放进单个 GB200 NVL72 节点吗？

## 关键术语

| 术语 | 大家怎么说 | 实际含义 |
|------|----------------|------------------------|
| FP8 | "8 位浮点" | 8 位浮点格式；因动态范围优势用于 KV 缓存和注意力 |
| NVFP4 | "4 位微缩放" | NVIDIA 的 4 位微缩放浮点格式；Blackwell 上用于权重和激活值 |
| MXFP8 | "MX 8 位" | 微缩放 FP8 变体；在 Blackwell Tensor Core 上有硬件加速 |
| Day-0 FP4 | "直接发 FP4 权重" | 模型提供方发布时权重就是 FP4；没有训练后转换步骤 |
| MTP | "多 token 预测" | TRT-LLM 内置的投机解码草稿机制（Phase 17 · 05） |
| 分离式服务 | "拆分预填充/解码" | 预填充和解码运行在不同 GPU 池上；KV 通过 NVLink/IB 传输 |
| All-to-all | "MoE 专家通信" | 把 token 路由到专家 GPU 的通信模式；NVLink 5 降低 3 倍延迟 |
| InferenceX | "SemiAnalysis 推理基准" | 2026 年业界公认的每 token 成本基准 |

## 延伸阅读

- [NVIDIA — Blackwell Ultra MLPerf Inference v6.0](https://developer.nvidia.com/blog/nvidia-blackwell-ultra-sets-new-inference-records-in-mlperf-debut/) — 2026 年 4 月 MLPerf 结果。
- [NVIDIA — MoE Inference on Blackwell](https://developer.nvidia.com/blog/delivering-massive-performance-leaps-for-mixture-of-experts-inference-on-nvidia-blackwell/) — NVLink 5 all-to-all 与 MoE 内核。
- [TensorRT-LLM Overview](https://nvidia.github.io/TensorRT-LLM/overview.html) — 官方引擎文档。
- [NVIDIA — Introducing Dynamo](https://developer.nvidia.com/blog/introducing-nvidia-dynamo-a-low-latency-distributed-inference-framework-for-scaling-reasoning-ai-models/) — 构建在 TRT-LLM 之上的分离式编排。
- [MLPerf Inference](https://mlcommons.org/benchmarks/inference-datacenter/) — 发布 Blackwell 数据的基准测试套件。
