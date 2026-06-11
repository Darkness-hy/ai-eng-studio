# OpenTelemetry GenAI — 端到端追踪工具调用

> 一个智能体调用了五个工具、三个 MCP 服务器和两个子智能体，你需要一条贯穿全部环节的链路追踪。OpenTelemetry GenAI 语义约定（v1.37 及以上版本中的稳定属性）是 2026 年的行业标准，Datadog、Langfuse、Arize Phoenix、OpenLLMetry 和 AgentOps 都原生支持。本课会列出必需的属性，讲解 span 层级结构（agent → LLM → tool），并交付一个可以接入任意 OTel exporter 的标准库 span 发射器。

**Type:** Build
**Languages:** Python (stdlib, OTel span emitter)
**Prerequisites:** Phase 13 · 07 (MCP server), Phase 13 · 08 (MCP client)
**Time:** ~75 minutes

## 学习目标

- 说出 LLM span 和工具执行 span 所需的 OTel GenAI 必备属性。
- 构建一条覆盖智能体循环、LLM 调用、工具调用和 MCP 客户端分发的追踪层级。
- 判断哪些内容应该捕获（显式开启）、哪些应该脱敏（默认行为）。
- 在不改写工具代码的前提下，把 span 发送到本地收集器（Jaeger、Langfuse）。

## 问题背景

2026 年 2 月的一次排障：用户反馈"我的智能体有时要 30 秒才响应，有时只要 3 秒"。没有链路追踪。日志里能看到 LLM 调用，但看不到工具分发，看不到 MCP 服务器的往返耗时，也看不到子智能体。你只能靠猜。最后才发现：某个 MCP 服务器偶尔会在冷启动时卡住。

没有端到端追踪，这种问题根本查不出来。OTel GenAI 解决的就是这个问题。

这套约定在 2025-2026 年间由 OpenTelemetry semantic-conventions 工作组定稿。它定义了稳定的属性名，因此 Datadog、Langfuse、Phoenix、OpenLLMetry 和 AgentOps 解析的都是同一种 span。埋点一次，就能对接任意后端。

## 核心概念

### Span 层级结构

```
agent.invoke_agent  (top, INTERNAL span)
 ├── llm.chat       (CLIENT span)
 ├── tool.execute   (INTERNAL)
 │    └── mcp.call  (CLIENT span)
 ├── llm.chat       (CLIENT span)
 └── subagent.invoke (INTERNAL)
```

整棵树嵌套在同一个 trace id 之下。span id 把父子关系串联起来。

### 必需属性

按 2025-2026 版语义约定（semconv）：

- `gen_ai.operation.name` — `"chat"`、`"text_completion"`、`"embeddings"`、`"execute_tool"`、`"invoke_agent"`。
- `gen_ai.provider.name` — `"openai"`、`"anthropic"`、`"google"`、`"azure_openai"`。
- `gen_ai.request.model` — 请求的模型字符串（例如 `"gpt-4o-2024-08-06"`）。
- `gen_ai.response.model` — 实际提供服务的模型。
- `gen_ai.usage.input_tokens` / `gen_ai.usage.output_tokens`。
- `gen_ai.response.id` — 服务商返回的响应 id，用于关联。

工具 span 的属性：

- `gen_ai.tool.name` — 工具标识符。
- `gen_ai.tool.call.id` — 本次调用的具体 id。
- `gen_ai.tool.description` — 工具描述（可选）。

智能体 span 的属性：

- `gen_ai.agent.name` / `gen_ai.agent.id` / `gen_ai.agent.description`。

### Span 类型（SpanKind）

- `SpanKind.CLIENT` 用于跨进程边界的调用（LLM 服务商、MCP 服务器）。
- `SpanKind.INTERNAL` 用于智能体自身的循环步骤和工具执行。

### 显式开启的内容捕获

默认情况下，span 只携带指标和耗时信息——不包含提示词和补全内容。大体积负载和个人敏感信息（PII）默认关闭。设置 `OTEL_SEMCONV_STABILITY_OPT_IN=gen_ai_latest_experimental` 以及相应的内容捕获环境变量后才会记录内容。在生产环境启用前请仔细评估。

### Span 上的事件

token 级别的事件可以作为 span event 附加：

- `gen_ai.content.prompt` — 输入消息。
- `gen_ai.content.completion` — 输出消息。
- `gen_ai.content.tool_call` — 记录下来的工具调用。

事件在 span 内按时间排序，便于精细回放。

### Exporter

OTel span 可以导出到：

- **Jaeger / Tempo。** 开源，可私有化部署。
- **Langfuse。** 专为 LLM 可观测性而生；可视化 token 用量。
- **Arize Phoenix。** 评测 + 追踪一体。
- **Datadog。** 商业产品；原生解析 `gen_ai.*` 属性。
- **Honeycomb。** 列式存储；查询体验好。

它们都讲 OTLP 这一传输格式。你的代码完全无感。

### 跨 MCP 的链路传播

当 MCP 客户端调用服务器时，把 W3C traceparent 头注入请求。Streamable HTTP 支持标准 HTTP 头。stdio 原生不携带 HTTP 头；规范的 2026 路线图正在讨论给 JSON-RPC 调用加一个 `_meta.traceparent` 字段。

在该特性落地之前：手动把 traceparent 放进每个请求的 `_meta` 里。服务器端记录 trace id。

### 指标

除 span 之外，GenAI 语义约定还定义了指标：

- `gen_ai.client.token.usage` — 直方图。
- `gen_ai.client.operation.duration` — 直方图。
- `gen_ai.tool.execution.duration` — 直方图。

不需要逐次调用细节的仪表盘可以直接用这些指标。

### AgentOps 这一层

AgentOps（2024 年创立）专注于 GenAI 可观测性。它封装了主流框架（LangGraph、Pydantic AI、CrewAI），自动发射 OTel span。如果你的技术栈用的是受支持的框架，它很有用；否则就用手动埋点。

## 生产实践

`code/main.py` 为一个调用一次 LLM、分发两个工具、做一次 MCP 往返的智能体，把 OTel 形状的 span 输出到 stdout（类 OTLP-JSON 格式）。没有接真实 exporter——本课聚焦的是 span 的形状和属性集。你可以把输出粘进兼容 OTLP 的查看器，或者直接阅读。

重点看这几处：

- 所有 span 共享同一个 trace id。
- 父子关系通过 `parentSpanId` 编码。
- 必需的 `gen_ai.*` 属性已填充。
- 内容捕获默认关闭；其中一个场景通过环境变量开启了它。

## 交付产物

本课产出 `outputs/skill-otel-genai-instrumentation.md`。给定一个智能体代码库，该 skill 会生成一份埋点方案：在哪里加 span、填充哪些属性、对接哪些 exporter。

## 练习

1. 运行 `code/main.py`。数一数 span 的数量，并区分哪些是 CLIENT、哪些是 INTERNAL。

2. 打开内容捕获（环境变量），确认 `gen_ai.content.prompt` 和 `gen_ai.content.completion` 事件出现了。思考这对 PII 意味着什么。

3. 添加工具执行指标 `gen_ai.tool.execution.duration`，并为每次调用发射一个直方图样本。

4. 把父级智能体 span 的 traceparent 传播到 MCP 请求的 `_meta.traceparent` 字段。验证 MCP 服务器看到的会是同一个 trace id。

5. 阅读 OTel GenAI 语义约定规范。找出一个规范中列出、但本课代码没有发射的属性，把它补上。

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|----------------|------------------------|
| OTel | "OpenTelemetry" | 追踪、指标、日志的开放标准 |
| GenAI semconv | "GenAI 语义约定" | LLM / 工具 / 智能体 span 的稳定属性名 |
| `gen_ai.*` | "属性命名空间" | 所有 GenAI 属性共享这个前缀 |
| Span | "带计时的操作" | 有起点、终点和属性的工作单元 |
| Trace | "跨 span 的谱系" | 共享同一个 trace id 的 span 树 |
| SpanKind | "CLIENT / SERVER / INTERNAL" | 标示 span 方向的提示 |
| OTLP | "OpenTelemetry Line Protocol" | exporter 使用的传输格式 |
| Opt-in content | "提示词 / 补全捕获" | 默认关闭；通过环境变量启用 |
| traceparent | "W3C 头" | 跨服务传播追踪上下文 |
| Exporter | "面向特定后端的发送器" | 把 span 发往 Jaeger / Datadog 等的组件 |

## 延伸阅读

- [OpenTelemetry — GenAI semconv](https://opentelemetry.io/docs/specs/semconv/gen-ai/) — GenAI span、指标和事件的权威约定
- [OpenTelemetry — GenAI spans](https://opentelemetry.io/docs/specs/semconv/gen-ai/gen-ai-spans/) — LLM 和工具执行 span 的属性列表
- [OpenTelemetry — GenAI agent spans](https://opentelemetry.io/docs/specs/semconv/gen-ai/gen-ai-agent-spans/) — 智能体级别的 `invoke_agent` span
- [open-telemetry/semantic-conventions — GenAI spans](https://github.com/open-telemetry/semantic-conventions/blob/main/docs/gen-ai/gen-ai-spans.md) — GitHub 上的权威源文档
- [Datadog — LLM OTel semantic convention](https://www.datadoghq.com/blog/llm-otel-semantic-convention/) — 生产环境集成实战
