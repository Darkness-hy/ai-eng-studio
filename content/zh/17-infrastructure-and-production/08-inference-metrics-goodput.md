# 推理指标 — TTFT、TPOT、ITL、Goodput、P99

> 四个指标决定一个推理部署是否真正可用。TTFT 等于预填充（prefill）加排队加网络时间。TPOT（等价于 ITL）是受显存带宽限制的逐 token 解码成本。端到端延迟等于 TTFT 加上 TPOT 乘以输出长度。吞吐量是整个集群聚合的每秒 token 数。但对产品真正重要的是有效吞吐量（goodput）——同时满足所有 SLO 的请求所占的比例。高吞吐量配低 goodput，意味着你处理的 token 根本没有按时送到用户手上。2026 年 Llama-3.1-8B-Instruct 在 TRT-LLM 上的参考数字：平均 TTFT 162 ms，平均 TPOT 7.33 ms，平均 E2E 1,093 ms。永远报告 P50、P90、P99——绝不要只报均值。还要小心测量陷阱：GenAI-Perf 在计算 ITL 时排除 TTFT，LLMPerf 则包含它；同一次运行，两个工具给出的 TPOT 并不一致。

**Type:** Learn
**Languages:** Python (stdlib, toy percentile calculator and goodput reporter)
**Prerequisites:** Phase 17 · 04 (vLLM Serving Internals)
**Time:** ~60 minutes

## 学习目标

- 精确定义 TTFT、TPOT、ITL、E2E、吞吐量和 goodput，并说出每个指标度量的是哪个环节。
- 解释为什么均值是 LLM 服务中错误的统计量，以及如何解读 P50/P90/P99。
- 构造一个多约束 SLO（例如 TTFT<500 ms 且 TPOT<15 ms 且 E2E<2 s），并据此计算 goodput。
- 说出两个对同一次运行给出不同 TPOT 的基准测试工具，并解释原因。

## 问题背景

「我们的吞吐量是每秒 15,000 个 token。」那又如何？如果 40% 的请求端到端超过了 2 秒，用户早就放弃会话了。光看吞吐量，无法判断产品是否可用。

推理有多个延迟维度，每个维度的失效方式各不相同。预填充是算力受限的，随提示词长度增长。解码是显存带宽受限的，随批大小增长。排队延迟是运维问题。网络是物理距离问题。你需要为每个环节准备独立的指标，需要百分位数，还需要一个综合指标来回答「用户是否得到了预期的体验」——这就是 goodput。

## 核心概念

### TTFT — 首个 token 时间

`TTFT = queue_time + network_request + prefill_time`

提示词较长时，预填充占主导。Llama-3.3-70B FP8 在 H100 上，32k 提示词的纯预填充耗时约 800 ms。排队时间取决于负载下的调度器行为。网络请求时间是包含 TLS 在内的线路时间。TTFT 是用户在任何内容流式返回之前感受到的延迟。

### TPOT / ITL — token 间延迟

同一个量有很多名字。`TPOT`（time per output token）、`ITL`（inter-token latency）、`decode latency per token`——都是一回事。它指的是首个 token 之后，相邻两个流式 token 之间的时间间隔。

`TPOT = (decode_forward_time + scheduler_overhead) / tokens_produced`

同样的 Llama-3.3-70B H100 配置下，启用分块预填充（chunked prefill）时，TPOT 均值约 7 ms。不启用分块预填充时，当相邻序列正在执行长预填充，TPOT 可能飙升到 50 ms。盯住 P99，不要盯均值。

### E2E 延迟

`E2E = TTFT + TPOT * output_tokens + network_response`

对于长输出（>500 个 token），E2E 由 TPOT 主导。对于长提示词配短输出，E2E 由 TTFT 主导。报告 E2E 时要按输出长度分组。

### 吞吐量

`throughput = total_output_tokens / elapsed_time`

这是一个聚合指标。它告诉你集群效率，但不告诉你单个请求的健康状况。

### Goodput — 你真正该关心的指标

`goodput = fraction of requests meeting (TTFT <= a) AND (TPOT <= b) AND (E2E <= c)`

SLO 是一个多约束条件。只有当每个约束都满足时，一个请求才算「好」。goodput 就是这个比例。高吞吐量配 60% goodput 是失败，低一些的吞吐量配 99% goodput 才是目标。

到 2026 年，goodput 已是 MLPerf Inference v6.0 提交结果中使用的指标，也是各 AI 平台提供商内部 SLA 跟踪所用的指标。

### 为什么均值是错误的统计量

LLM 的延迟分布是右偏的。一个解码批次中只要有一个长预填充的邻居，就可能出现 500 个 token 的 TPOT 约 7 ms，而 20 个 token 的 TPOT 约 60 ms 的情况。TPOT 均值是 9 ms，P99 TPOT 却是 65 ms。用户经常撞上 P99——这正是他们流失的原因。

永远报告三元组（P50、P90、P99）。从用户体验的角度，P99 才是你要优化的那个数字。

### 参考数字 — Llama-3.1-8B-Instruct on TRT-LLM, 2026

- 平均 TTFT：162 ms
- 平均 TPOT：7.33 ms
- 平均 E2E：1,093 ms
- P99 TPOT：取决于分块预填充配置，在 10-25 ms 之间波动。

这些是 NVIDIA 公布的参考数据点。它们会随模型规模（70B 会有 3-5 倍变化）、硬件（H100 对比 B200 约 3 倍）和负载而变化。

### 测量陷阱

2026 年最常用的两个基准测试工具，对同一次运行给出的 TPOT 并不一致：

- **NVIDIA GenAI-Perf**：在 ITL 计算中排除 TTFT。ITL 从第 2 个 token 开始计。
- **LLMPerf**：包含 TTFT。ITL 从第 1 个 token 开始计。

对于一个 TTFT 500 ms、总解码时间 700 ms、输出 100 个 token 的请求，GenAI-Perf 报告 `ITL = 700/99 = 7.07 ms`，LLMPerf 报告 `ITL = 1200/100 = 12.00 ms`。选哪个工具，数字就不一样。

永远注明使用了哪个工具。永远公布指标定义。

### 构造一个 SLO

2026 年面向消费者的 70B 聊天模型的一个合理 SLO：

- TTFT P99 <= 800 ms。
- TPOT P99 <= 25 ms。
- 对 <300 token 的输出，E2E P99 <= 3 s。
- Goodput 目标 >= 99%。

企业级 SLO 会收紧 TTFT（200-400 ms）并放宽 E2E。关键是把它们写下来，三项都测量，并把 goodput 作为单一综合指标来跟踪。

### 如何测量

- 跑真实流量或贴近真实的合成流量（LLMPerf 配合 `--mean-input-tokens 800 --stddev-input-tokens 300 --mean-output-tokens 150`）。
- 基准测试以峰值并发的 2 倍为目标。
- 运行 30-50 轮迭代，对合并样本取百分位数。
- 发布结果时附上工具名称、工具版本、模型、硬件、并发数、提示词分布。

```figure
throughput-latency
```

## 生产实践

`code/main.py` 是一个玩具版 goodput 计算器。生成一个合成延迟分布，套用一个 SLO，计算 goodput。它还在同一条 trace 上演示了 GenAI-Perf 与 LLMPerf 的 TPOT 差异。

## 交付产物

本课产出 `outputs/skill-slo-goodput-gate.md`。给定工作负载和 SLO，它生成一份可直接用于 CI/CD 的基准测试方案，以 goodput 而非吞吐量作为部署门禁。

## 练习

1. 运行 `code/main.py`。生成一个带 1% 尾部尖刺的分布。当你把 P99 TPOT 约束从 30 ms 收紧到 15 ms 时，goodput 如何变化？
2. 某供应商宣称「Llama 3.3 70B 在 H100 上达到 15,000 tok/s」。在相信这个数字之前，列出三个该问的问题。
3. 为什么分块预填充能保护 P99 TPOT，却保护不了 TPOT 均值？
4. 为一个语音助手构造一个消费者 SLO（首个 token 是被听到的，不是被读到的）。哪个指标对用户最显眼？
5. 阅读 LLMPerf 的 README 和 GenAI-Perf 的文档。找出另外三个两个工具定义不一致的指标。

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|----------------|------------------------|
| TTFT | 「首个 token 时间」 | 排队 + 网络 + 预填充；长提示词时由预填充主导 |
| TPOT | 「每个输出 token 的时间」 | 首个 token 之后，受显存带宽限制的逐 token 解码成本 |
| ITL | 「token 间延迟」 | 在大多数工具中等同于 TPOT（并非全部——见 GenAI-Perf） |
| E2E | 「端到端」 | TTFT + TPOT * output_len；再叠加响应侧网络时间 |
| 吞吐量 | 「tok/s」 | 集群效率；没有延迟百分位数就毫无意义 |
| Goodput | 「SLO 达标率」 | 同时满足所有 SLO 约束的请求所占比例 |
| P99 | 「尾部」 | 百分之一概率的最差延迟；衡量用户体验的指标 |
| SLO 多约束 | 「联合约束」 | 三个延迟上限的 AND；任一约束被违反，请求即判失败 |
| GenAI-Perf vs LLMPerf | 「工具陷阱」 | 两个工具对 ITL 是否包含 TTFT 的定义不一致 |

## 延伸阅读

- [NVIDIA NIM — LLM Benchmarking Metrics](https://docs.nvidia.com/nim/benchmarking/llm/latest/metrics.html) — TTFT、ITL、TPOT 的权威定义。
- [Anyscale — LLM Serving Benchmarking Metrics](https://docs.anyscale.com/llm/serving/benchmarking/metrics) — 另一套定义和测量方案。
- [BentoML — LLM Inference Metrics](https://bentoml.com/llm/inference-optimization/llm-inference-metrics) — 真实部署上的实用测量方法。
- [LLMPerf](https://github.com/ray-project/llmperf) — 基于 Ray 的开源基准测试工具。
- [GenAI-Perf](https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/client/src/c++/perf_analyzer/genai-perf/README.html) — NVIDIA 的基准测试工具。
- [MLPerf Inference](https://mlcommons.org/benchmarks/inference-datacenter/) — 业界公认的基于 goodput 的基准测试。
