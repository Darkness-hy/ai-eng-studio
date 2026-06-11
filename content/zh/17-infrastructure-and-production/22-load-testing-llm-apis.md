# LLM API 负载测试——为什么 k6 和 Locust 会"撒谎"

> 传统负载测试工具的设计并未考虑流式响应、可变输出长度、token 级指标以及 GPU 饱和。大多数团队会踩中两个陷阱。GIL 陷阱：Locust 的 token 级测量在 Python GIL 下执行分词，高并发时与请求生成相互争抢；分词积压随之抬高报告出来的 token 间延迟——瓶颈在你的客户端，而不是服务器。提示词单一化陷阱：循环发送相同的提示词只测到 token 分布上的一个点；真实流量的长度各异、前缀匹配情况多样。LLMPerf 用 `--mean-input-tokens` + `--stddev-input-tokens` 解决了这个问题。2026 年工具版图：LLM 专用工具（GenAI-Perf、LLMPerf、LLM-Locust、guidellm）提供 token 级精度；**k6 v2026.1.0** + **k6 Operator 1.0 GA（2025 年 9 月）**——支持流式感知、通过 TestRun/PrivateLoadZone CRD 实现 Kubernetes 原生分布式测试，最适合 CI/CD 门禁；Vegeta 适合 Go 实现的恒定速率打满测试；Locust 2.43.3 只有配合 LLM-Locust 扩展才能用于流式场景。负载模式：稳态、爬坡、突发（测试自动扩缩容）、浸泡（查内存泄漏）。

**Type:** Build
**Languages:** Python (stdlib, toy realistic-prompt generator + latency collector)
**Prerequisites:** Phase 17 · 08 (Inference Metrics), Phase 17 · 03 (GPU Autoscaling)
**Time:** ~75 minutes

## 学习目标

- 解释让通用负载测试工具在 LLM API 上"撒谎"的两个反模式（GIL 陷阱、提示词单一化陷阱）。
- 按用途选对工具：LLMPerf（基准测试运行）、k6 + 流式扩展（CI 门禁）、guidellm（大规模合成负载）、GenAI-Perf（NVIDIA 参考实现）。
- 设计四种负载模式（稳态、爬坡、突发、浸泡），并说出每种模式能捕获的故障类型。
- 用输入 token 的均值 + 标准差构建真实的提示词分布，而不是用固定长度。

## 问题背景

你用 k6 在 500 并发用户下测了你的 LLM 端点。它扛住了。你上线了。结果生产环境只有 200 个真实用户时服务就崩了——P99 TTFT 暴涨，GPU 被打满。

发生了两件事。第一，k6 发送的是 500 条完全相同的提示词——请求合并和前缀缓存让它看起来像在处理 500 路并发解码，实际上只处理了一路。第二，k6 不会按人眼实际感受的方式跟踪流式响应的 token 间延迟；它看到的是一条 HTTP 连接，而不是以不同间隔陆续到达的 500 个 token。

LLM 的负载测试是一门独立的学问。

## 核心概念

### GIL 陷阱（Locust）

Locust 用 Python 实现，并在客户端的 GIL 下执行分词。高并发时分词器排在请求生成后面排队。报告出来的 token 间延迟包含了客户端的分词积压。你以为是服务器慢；其实是测试工具本身慢。

解决办法：LLM-Locust 扩展把分词挪到独立进程，或者改用编译型语言的测试工具（k6、使用 tokenizers.rs 的 LLMPerf）。

### 提示词单一化陷阱

所有已知的负载测试工具都允许你只配置一条提示词。在 10,000 次迭代的循环测试中，每次发送的都是同一条提示词。服务器每次看到的都是相同前缀——前缀缓存命中率逼近 100%，吞吐量看起来好得不像话。

解决办法：从提示词分布中采样。LLMPerf 使用 `--mean-input-tokens 500 --stddev-input-tokens 150`——长度多样、内容多样。

### 四种负载模式

1. **稳态（Steady-state）**——恒定 RPS 持续 30-60 分钟。能捕获：基线性能回退。
2. **爬坡（Ramp）**——在 15 分钟内将 RPS 从 0 线性升至目标值。能捕获：容量拐点、预热异常。
3. **突发（Spike）**——RPS 骤增 3-10 倍持续 2 分钟后回落。能捕获：自动扩缩容延迟、队列饱和、冷启动影响。
4. **浸泡（Soak）**——稳态持续 4-8 小时。能捕获：内存泄漏、连接池漂移、可观测性数据溢出。

### 2026 年工具版图

**LLMPerf**（Anyscale）——Python 实现但分词由 Rust 支撑。支持均值/标准差提示词。流式感知。性能测试的最佳默认选择。

**NVIDIA GenAI-Perf**——NVIDIA 的参考实现。使用 Triton 客户端；指标覆盖全面。注意它的 ITL 不含 TTFT，而 LLMPerf 的包含。同一台服务器，两个工具会得出不同的 TPOT。

**LLM-Locust**（TrueFoundry）——修复了 GIL 陷阱的 Locust 扩展。熟悉的 Locust DSL + 流式指标。

**guidellm**——大规模合成基准测试。

**k6 v2026.1.0** + **k6 Operator 1.0 GA（2025 年 9 月）**：
- k6 本体（Go 编写、编译型、无 GIL）新增了流式感知指标。
- k6 Operator 通过 TestRun / PrivateLoadZone CRD 实现 Kubernetes 原生分布式测试。
- 最适合 CI/CD 门禁与 SLA 测试。

**Vegeta**——Go 实现，比 k6 更简单。恒定速率的 HTTP 打满测试。不感知 LLM，但适合网关 / 限流测试。

**Locust 2.43.3 原版**——在 LLM 场景下存在 GIL 陷阱。只能配合 LLM-Locust 扩展使用。

### CI 中的 SLA 门禁

在 PR 上运行 k6：

- 在基线 RPS 下各跑 30-50 次迭代。
- 门禁条件：P50/P95 TTFT、5xx < 5%、TPOT 低于阈值。
- 任一指标越界即让构建失败。

### 真实的提示词分布

从真实流量样本构建（如果有的话），或者使用公开发布的分布（例如聊天场景用 ShareGPT 提示词，代码场景用 HumanEval）。把均值 + 标准差喂给 LLMPerf。无论如何都要避免"单条提示词循环发送"。

### 应该记住的数字

- k6 Operator 1.0 GA：2025 年 9 月。
- k6 v2026.1.0：流式感知指标。
- 典型的 LLMPerf 运行：并发度为 X 时发送 100-1000 个请求。
- 典型的 CI 门禁：每个 PR 跑 30-50 次迭代。
- 四种模式：稳态、爬坡、突发、浸泡。

## 生产实践

`code/main.py` 用真实的提示词分布模拟一次负载测试，测量有效 TPOT，并演示单一提示词陷阱。

## 交付产物

本课产出 `outputs/skill-load-test-plan.md`。给定工作负载和 SLA，选择工具并设计四种负载模式。

## 练习

1. 运行 `code/main.py`。对比单一分布与真实分布——差距在哪里？
2. 为 CI 门禁编写 k6 脚本：100 并发下 TTFT P95 < 800 ms，运行时长 5 分钟。
3. 你的浸泡测试显示内存以 50 MB/小时的速度增长。说出三个可能原因，以及用什么监测手段区分它们。
4. 突发测试从 10 RPS 升到 100 RPS。如果已部署 Karpenter + vLLM production-stack（Phase 17 · 03 + 18），预期恢复时间是多少？
5. 同一台服务器上，GenAI-Perf 报告 TPOT=6ms，LLMPerf 报告 TPOT=11ms。解释原因。

## 关键术语

| 术语 | 大家怎么说 | 实际含义 |
|------|----------------|------------------------|
| LLMPerf | "那个 LLM 压测工具" | Anyscale 出品的基准测试工具，流式感知 |
| GenAI-Perf | "NVIDIA 的工具" | NVIDIA 的参考测试工具 |
| LLM-Locust | "LLM 版 Locust" | 修复 GIL 陷阱的 Locust 扩展 |
| guidellm | "合成基准测试" | 大规模合成负载工具 |
| k6 Operator | "K8s 版 k6" | 基于 CRD 的分布式 k6 |
| GIL 陷阱 | "Python 客户端开销" | 分词积压抬高报告出来的延迟 |
| 提示词单一化陷阱 | "单提示词谎言" | 循环发同一条提示词命中缓存，虚高吞吐量 |
| 稳态 | "恒定负载" | 平稳 RPS 持续 N 分钟 |
| 爬坡 | "线性上升" | 在指定时长内从 0 升到目标值 |
| 突发 | "突发测试" | 骤增数倍后回落 |
| 浸泡 | "长时间测试" | 持续数小时以检测泄漏 |

## 延伸阅读

- [TianPan — Load Testing LLM Applications](https://tianpan.co/blog/2026-03-19-load-testing-llm-applications)
- [PremAI — Load Testing LLMs 2026](https://blog.premai.io/load-testing-llms-tools-metrics-realistic-traffic-simulation-2026/)
- [NVIDIA NIM — Introduction to LLM Inference Benchmarking](https://docs.nvidia.com/nim/large-language-models/1.0.0/benchmarking.html)
- [TrueFoundry — LLM-Locust](https://www.truefoundry.com/blog/llm-locust-a-tool-for-benchmarking-llm-performance)
- [LLMPerf](https://github.com/ray-project/llmperf)
- [k6 Operator](https://github.com/grafana/k6-operator)
