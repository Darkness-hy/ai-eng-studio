# 毕业项目 14 — 投机解码推理服务器

> vLLM 0.7 中的 EAGLE-3 在真实流量上带来 2.5-3 倍的吞吐提升。P-EAGLE（AWS 2026）把并行投机推得更远。SGLang 的 SpecForge 实现了大规模草稿头训练。Red Hat 的 Speculators 仓库为常见开源模型发布了对齐好的草稿模型。TensorRT-LLM 让投机解码在 NVIDIA 平台上成为一等公民。2026 年的生产级推理服务栈就是 vLLM 或 SGLang 配 EAGLE 系草稿模型、FP8 或 INT4 量化，外加基于队列等待时间的 HPA。这个毕业项目的目标：以基线 2.5 倍以上的吞吐量服务两个开源模型，并交出一份完整的尾延迟报告。

**Type:** Capstone
**Languages:** Python (serving), C++ / CUDA (kernel inspection), YAML (configs)
**Prerequisites:** Phase 3 (deep learning), Phase 7 (transformers), Phase 10 (LLMs from scratch), Phase 17 (infrastructure)
**涉及阶段：** P3 · P7 · P10 · P17
**Time:** 30 hours

## 问题背景

到 2026 年，投机解码（speculative decoding）已经成为标配。EAGLE-3 草稿头在目标模型的隐藏状态上训练，向前预测 N 个 token；目标模型在单次前向中完成验证。60-80% 的接受率可以换来 2-3 倍的端到端吞吐量。vLLM 0.7 已原生集成。SGLang + SpecForge 提供了训练流水线。Red Hat 的 Speculators 为 Llama 3.3 70B、Qwen3-Coder-30B MoE、GPT-OSS-120B 发布了对齐好的草稿模型。

真正的功夫在服务运维，而不在模型本身。接受率会随流量分布漂移（ShareGPT、代码、领域数据各不相同）。发生拒绝时的尾延迟比不开投机时更差——你必须报告多个批大小下的 p99，而不能只看稳态的 tokens/sec。和 Anthropic / OpenAI API 对比的每 100 万 token 成本，才是说服力的关键杠杆。

## 核心概念

投机解码分两层。**草稿（draft）**模型（EAGLE-3 头、ngram，或与目标对齐的较小模型）每步提议 k 个候选 token。**目标（target）**模型在一次前向中验证全部 k 个；被接受的任意前缀直接替代贪心路径。接受率取决于草稿与目标的对齐程度，以及输入分布。

在大多数流量上，EAGLE-3 优于 ngram 草稿。P-EAGLE 用并行投机构建更深的草稿树。代价是：发生拒绝时 p99 延迟更高，因为验证前向更大。服务配置必须按批大小分桶报告延迟，才能暴露这个问题。

部署在 Kubernetes 上。vLLM 0.7 每个 GPU 或张量并行分片跑一个副本。HPA 基于队列等待时间而非 CPU 自动扩缩容。FP8（Marlin）和 INT4（AWQ）量化把 GPU 显存控制在 H100 / H200 的容量范围内。端到端报告包括吞吐量、接受率、批大小 1/8/32 下的 p50/p99，以及每 100 万 token 的美元成本。

## 架构

```
request ingress
    |
    v
vLLM server (0.7) or SGLang (0.4)
    |
    +-- draft: EAGLE-3 heads | P-EAGLE parallel | ngram fallback
    +-- target: Llama 3.3 70B | Qwen3-Coder-30B | GPT-OSS-120B
    |     quantized FP8-Marlin or INT4-AWQ
    |
    v
verify pass: batch k draft tokens through target
    |
    v (accept prefix; resample for rejected suffix)
    v
token stream back to client
    |
    v
Prometheus metrics: throughput, acceptance rate, queue wait, latency p50/p99
    |
    v
HPA on queue-wait metric
```

## 技术栈

- 推理服务：vLLM 0.7 或 SGLang 0.4
- 投机方法：EAGLE-3 草稿头、P-EAGLE 并行投机、ngram 兜底
- 草稿训练：SpecForge（SGLang）或 Red Hat Speculators
- 目标模型：Llama 3.3 70B、Qwen3-Coder-30B MoE、GPT-OSS-120B
- 量化：FP8（Marlin）、INT4 AWQ
- 部署：Kubernetes + NVIDIA device plugin；基于队列等待指标的 HPA
- 评测：ShareGPT、MT-Bench-v2、GSM8K、HumanEval，用于测量跨领域的接受率分布
- 参考：TensorRT-LLM 投机解码，作为厂商基线

## 从零实现

1. **目标模型准备。** 选 Llama 3.3 70B。用 Marlin 量化到 FP8。部署在 vLLM 0.7 上，单卡 H100（或 2 卡张量并行）。

2. **草稿来源。** 从 Red Hat Speculators 拉取对齐好的 EAGLE-3 草稿头（或用 SpecForge 自己训练一个）。加载进 vLLM 的投机解码配置。

3. **基线数据。** 开启投机之前：批大小 1/8/32 下的 tokens/s、p50/p99 延迟、GPU 利用率。发布出来。

4. **启用 EAGLE-3。** 切换配置；重跑同一套基准测试。报告加速比、接受率、p99 尾延迟变化量。

5. **P-EAGLE。** 启用并行投机；测量更深的草稿树相对串行 EAGLE-3 的表现。报告 P-EAGLE 由利转弊的拐点。

6. **领域流量。** 让 ShareGPT、HumanEval 和领域专属流量分别打到同一台服务器。测量每种分布下的接受率。识别草稿模型何时发生漂移。

7. **第二个目标模型。** 在 Qwen3-Coder-30B MoE 上跑同一套流水线。草稿更难做（MoE 路由噪声）。给出报告。

8. **K8s HPA。** 部署到 K8s，HPA 跟踪 `queue_wait_ms`。演示负载翻三倍时的横向扩容。

9. **成本对比。** 在同一套评测上，计算每 100 万 token 的美元成本，与 Anthropic Claude Sonnet 4.7 和 OpenAI GPT-5.4 对比。发布出来。

## 生产实践

```
$ curl https://infer.example.com/v1/chat/completions -d '{"messages":[...]}'
[serve]     vLLM 0.7, Llama 3.3 70B FP8, EAGLE-3 active
[decode]    bs=8, accepted_tokens_per_step=3.2, acceptance_rate=0.76
[latency]   first-token 42ms, full-response 980ms (620 tokens)
[cost]      $0.34 per 1M output tokens at sustained throughput
```

## 交付产物

`outputs/skill-inference-server.md` 描述了交付物。一套带投机解码的、经过实测的推理服务栈，一份完整的基准测试报告，以及一套 K8s 部署。

| 权重 | 评分标准 | 测量方式 |
|:-:|---|---|
| 25 | 相对基线的实测加速比 | 在两个模型上、质量持平的前提下达到 2.5 倍以上吞吐 |
| 20 | 真实流量上的接受率 | 按分布分别报告接受率 |
| 20 | p99 尾延迟纪律 | 批大小 1/8/32 下开 / 不开投机的 p99 |
| 20 | 运维 | K8s 部署、基于队列等待的 HPA、平滑发布 |
| 15 | 报告撰写与方法论 | 清楚解释改了什么、为什么改 |
| **100** | | |

## 练习

1. 测量草稿模型落后目标模型一个版本时的接受率退化（例如 Llama 3.3 -> 3.4 漂移）。构建一个监控告警。

2. 实现 ngram 兜底：当 EAGLE-3 接受率跌破阈值时切换到 ngram 草稿。报告可靠性提升。

3. 做一个受控的 MoE 实验：同一个 Qwen3-Coder-30B，对比注入路由噪声与不注入两种情况。测量草稿接受率的敏感度。

4. 扩展到 H200（141 GB）。报告每副本可容纳模型大小的余量提升，以及能否服务未量化的 Llama 3.3 70B。

5. 在同样的 H100 硬件上基准测试 TensorRT-LLM 的投机解码。报告它在哪些场景下优于 vLLM。

## 关键术语

| 术语 | 大家怎么说 | 实际含义 |
|------|-----------------|------------------------|
| 草稿模型 | "Speculator" | 提议 N 个 token、交给目标模型验证的小模型 |
| EAGLE-3 | "2026 年的草稿架构" | 在目标模型隐藏状态上训练的草稿头；接受率约 75% |
| P-EAGLE | "并行投机" | 草稿分支构成的树，在目标模型的一次前向中完成验证 |
| 接受率 | "命中率" | 草稿 token 中无需重采样即被接受的比例 |
| 量化 | "FP8 / INT4" | 用更低精度的权重在 GPU 显存中装下更大的模型 |
| 队列等待 | "HPA 指标" | 请求在待处理队列中等待、直到推理开始的时间 |
| Speculators 仓库 | "对齐好的草稿模型" | Red Hat Neural Magic 维护的、面向常见开源模型的 EAGLE 草稿仓库 |

## 延伸阅读

- [vLLM EAGLE and P-EAGLE documentation](https://docs.vllm.ai) — 参考推理服务栈
- [P-EAGLE (AWS 2026)](https://aws.amazon.com/blogs/machine-learning/p-eagle-faster-llm-inference-with-parallel-speculative-decoding-in-vllm/) — 并行投机解码论文 + 集成
- [SGLang SpecForge](https://github.com/sgl-project/SpecForge) — 草稿头训练流水线
- [Red Hat Speculators](https://github.com/neuralmagic/speculators) — 对齐草稿模型仓库
- [TensorRT-LLM speculative decoding](https://nvidia.github.io/TensorRT-LLM/) — 厂商替代方案
- [Fireworks.ai serving architecture](https://fireworks.ai/blog) — 商业参考
- [EAGLE-3 paper (arXiv:2503.01840)](https://arxiv.org/abs/2503.01840) — 方法论文
- [vLLM repository](https://github.com/vllm-project/vllm) — 代码与基准测试
