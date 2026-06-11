# LLM 可观测性技术栈选型

> 2026 年的可观测性市场分裂为两大类。开发平台（LangSmith、Langfuse、Comet Opik）把监控与评估、提示词管理、会话回放打包在一起。网关/插桩类工具（Helicone、SigNoz、OpenLLMetry、Phoenix）则专注于遥测。Langfuse 核心采用 MIT 许可，开源平衡性强（云端免费额度为每月 5 万事件）。Phoenix 原生支持 OpenTelemetry，采用 Elastic License 2.0——非常适合漂移/RAG 可视化，但不是持久化的生产后端。Arize AX 采用基于 Iceberg/Parquet 的零拷贝集成，宣称比单体式可观测性便宜 100 倍。LangSmith 在 LangChain/LangGraph 场景下领先，每用户每月 39 美元，仅 Enterprise 版支持自托管。Helicone 基于代理，15-30 分钟即可接入，每月 10 万请求免费，但在智能体链路追踪上深度不足。常见的生产模式：网关（Helicone/Portkey）+ 评估平台（Phoenix/TruLens），用 OpenTelemetry 粘合。

**Type:** Learn
**Languages:** Python (stdlib, toy trace-sampling simulator)
**Prerequisites:** Phase 17 · 08 (Inference Metrics), Phase 14 (Agent Engineering)
**Time:** ~60 minutes

## 学习目标

- 区分开发平台（打包提供：评估 + 提示词 + 会话）与网关/遥测类工具（仅链路追踪 + 指标）。
- 把六个主流工具（Langfuse、LangSmith、Phoenix、Arize AX、Helicone、Opik）对应到各自的许可协议、定价与最适用场景。
- 解释 OpenTelemetry 粘合模式，它让你能把网关工具和独立的评估平台组合使用。
- 说出 2026 年的成本差异化因素（Arize AX 的零拷贝方案对比单体式数据摄入），并给出大致的 100 倍差距。

## 问题背景

你上线了一个 LLM 功能。它能跑。但你对提示词失败、工具循环、延迟退化、成本激增、提示词缓存命中率毫无可见性。你搜索"LLM observability"，得到八个工具，它们都声称解决同一个问题，却给出三档不同的价格。

它们解决的并不是同一个问题。LangSmith 回答的是"这次 LangGraph 运行为什么失败？"Phoenix 回答的是"我的 RAG 流水线是否在漂移？"Helicone 回答的是"哪个应用在烧 token？"Langfuse 回答的是"我能不能整套自托管？"不同的工具，面向不同的受众。

选型涉及四个维度：技术栈（LangChain？原生 SDK？多厂商？）、许可协议容忍度（只接受 MIT？Elastic 也行？商业授权没问题？）、预算（免费额度？每月 100 美元？每月 1000 美元？）、自托管（必须？最好有？永远不会？）。

## 核心概念

### 两大类别

**开发平台**把可观测性与评估、提示词管理、数据集版本管理、会话回放打包在一起。你可以跑实验、看哪个提示词有效、用数据集回归测试把新提示词和历史最优版本对比。代表：LangSmith、Langfuse、Comet Opik。

**网关/遥测类工具**对推理调用插桩——提示词、响应、token 数、延迟、模型、成本。代表：Helicone、SigNoz、OpenLLMetry、Phoenix。走极简路线。可以通过 OpenTelemetry 与独立的评估工具组合使用。

### Langfuse——开源平衡之选

- 核心采用 Apache / MIT 许可；可通过 Docker 自托管。
- 云端免费额度：每月 5 万事件。付费版：团队版每月 29 美元。
- 提供评估、提示词管理、链路追踪、数据集。对开发平台四大功能的覆盖都比较合理。
- 最适用场景：你想要 LangSmith 级别的功能，但必须自托管或坚持开源许可。

### Phoenix（Arize）——遥测优先、原生 OpenTelemetry

- Elastic License 2.0；自托管非常简单。
- 在 RAG 和漂移可视化上表现出色。嵌入空间散点图作为一等公民功能内置。
- 并非为持久化生产后端设计——主要面向开发期可观测性。
- 最适用场景：RAG 流水线开发、漂移调试，生产环境可与独立网关搭配使用。

### Arize AX——规模化打法

- 商业产品。通过 Iceberg/Parquet 实现零拷贝数据湖集成。
- 宣称在大规模场景下比单体式可观测性（Datadog 级别）便宜约 100 倍。背后逻辑：你把链路追踪数据以 Parquet 格式存在自己的 S3 上，Arize 直接读取。
- 最适用场景：每天超过 1000 万条 trace、已有数据湖、想要 LLM 专属仪表盘又不想付 Datadog 的价格。

### LangSmith——LangChain/LangGraph 优先

- 商业产品，每用户每月 39 美元。仅 Enterprise 版支持自托管。
- 在 LangChain 和 LangGraph 技术栈上是同类最佳。如果你不用这两者，吸引力就小很多。
- 最适用场景：团队深度绑定 LangChain，且愿意付费。

### Helicone——基于代理的最小可行方案

- 把你的 `OPENAI_API_BASE` 换成 Helicone 代理即可，15-30 分钟接入。
- MIT 许可；每月 10 万请求免费，付费版每月 20 美元起。
- 内置故障转移、缓存、限流——同时也充当网关。
- 在智能体/多步链路追踪上深度不足。
- 最适用场景：快速起步、单一技术栈的应用、需要网关与可观测性二合一。

### Opik（Comet）——开源开发平台

- Apache 2.0，完全开源。
- 功能集与 Langfuse 类似，带有 Comet 的血统。
- 最适用场景：已在使用 Comet 的 ML 团队，想在同一个界面里获得 LLM 可观测性。

### SigNoz——OpenTelemetry 优先的全栈 APM

- Apache 2.0。通过 OpenTelemetry 同时支持通用 APM 和 LLM。
- 最适用场景：跨服务与 LLM 调用的统一可观测性。

### 粘合层：OpenTelemetry + GenAI 语义约定

OpenTelemetry 在 2025 年末发布了 GenAI 语义约定（`gen_ai.system`、`gen_ai.request.model`、`gen_ai.usage.input_tokens`）。能消费 OTel 数据的工具之间可以互操作。正在形成的生产模式是：

1. 从每次 LLM 调用发出带 GenAI 约定的 OTel 数据。
2. 路由到网关（Helicone / Portkey）用于日常运维。
3. 双写到评估平台（Phoenix / Langfuse）用于回归检测。
4. 归档到数据湖（Iceberg），通过 Arize AX 或 DuckDB 做长期分析。

### 陷阱：在错误的层插桩

在智能体框架内部插桩（例如添加 LangSmith trace）会把你和那个框架耦合在一起。在 HTTP/OpenAI-SDK 层插桩（通过 OpenLLMetry 或你的网关）才具有可移植性。

### 采样——你不可能保留所有数据

当每天请求超过 100 万时，全量链路保留的成本比 LLM 调用本身还高。按规则采样：错误 100% 保留、高成本 100% 保留、成功请求保留 5%。聚合指标始终保留；原始数据为长尾分析保留。

### 你应该记住的数字

- Langfuse 云端免费额度：每月 5 万事件。
- LangSmith：每用户每月 39 美元。
- Helicone 免费额度：每月 10 万请求。
- Arize AX 的宣称：大规模场景下比单体式方案便宜约 100 倍。
- OpenTelemetry GenAI 约定：2025 年发布，2026 年被广泛采用。

## 生产实践

`code/main.py` 模拟一天 100 万条 trace 在不同保留策略下的表现（100% 摄入、采样、采样 + 错误全保留）。报告每种策略的存储成本以及丢失了什么。

## 交付产物

本课产出 `outputs/skill-observability-stack.md`。给定技术栈、规模、预算、许可协议立场，选出合适的工具组合。

## 练习

1. 你的团队使用 LangChain，想要开源自托管的可观测性方案。在 Langfuse 和 Opik 之间选一个，并给出理由。
2. 每天 500 万条 trace，Datadog 报价每月 15 万美元，计算 Arize AX 的盈亏平衡点。
3. 设计一套 OpenTelemetry GenAI 属性集，作为你所在组织规范中每次 LLM 调用必须携带的属性。
4. 论证只用 Phoenix 是否足以支撑生产环境。什么情况下它不够用？
5. Helicone 带来 20ms 的代理开销。在 P99 TTFT 为 300 ms 时，这可以接受吗？如果 SLA 是 100 ms 呢？

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|----------------|------------------------|
| OpenLLMetry | "LLM 版的 OTel" | 面向 LLM 的开源 OpenTelemetry 插桩库 |
| GenAI 约定 | "OTel 属性" | LLM 调用的标准 OTel 属性命名 |
| LangSmith | "LangChain 可观测性" | 与 LangChain 生态捆绑的商业平台 |
| Langfuse | "开源版 LangSmith" | 功能集类似的 MIT 开源方案 |
| Phoenix | "Arize 的开发工具" | 原生 OpenTelemetry 的开发/评估平台 |
| Arize AX | "规模化可观测性" | 基于 Iceberg/Parquet 零拷贝的商业可观测性产品 |
| Helicone | "代理式可观测性" | 收集 LLM 遥测数据的 HTTP 代理 + 网关功能 |
| Opik | "Comet LLM" | Comet 出品的 Apache 2.0 开源开发平台 |
| 会话回放（Session replay） | "trace 重跑" | 完整回放一次带工具调用的智能体会话 |
| 评估（Eval） | "离线测试" | 在带标注的数据集上运行候选模型/提示词 |

## 延伸阅读

- [SigNoz — Top LLM Observability Tools 2026](https://signoz.io/comparisons/llm-observability-tools/)
- [Langfuse — Arize AX Alternative analysis](https://langfuse.com/faq/all/best-phoenix-arize-alternatives)
- [PremAI — Setting Up Langfuse, LangSmith, Helicone, Phoenix](https://blog.premai.io/llm-observability-setting-up-langfuse-langsmith-helicone-phoenix/)
- [OpenTelemetry GenAI Semantic Conventions](https://opentelemetry.io/docs/specs/semconv/gen-ai/)
- [Arize Phoenix docs](https://docs.arize.com/phoenix)
- [Helicone docs](https://docs.helicone.ai/)
