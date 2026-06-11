# OpenTelemetry GenAI 语义约定

> OpenTelemetry 的 GenAI SIG（2024 年 4 月成立）定义了智能体遥测数据的标准 schema。Span 名称、属性和内容捕获规则在各厂商之间趋于统一，使智能体追踪数据在 Datadog、Grafana、Jaeger 和 Honeycomb 中具有相同的含义。

**Type:** Learn + Build
**Languages:** Python (stdlib)
**Prerequisites:** Phase 14 · 13 (LangGraph), Phase 14 · 24 (Observability Platforms)
**Time:** ~60 minutes

## 学习目标

- 说出 GenAI 的几类 span：模型/客户端、智能体、工具。
- 区分 `invoke_agent` 的 CLIENT 与 INTERNAL 两种 span 类型，以及各自的适用场景。
- 列出 GenAI 的顶层属性：提供方名称、请求模型、数据源 ID。
- 解释内容捕获契约：默认关闭需显式开启（opt-in）、`OTEL_SEMCONV_STABILITY_OPT_IN`、外部引用存储的推荐做法。

## 问题背景

每家厂商都在发明自己的 span 名称，运维团队不得不为每个框架单独搭建仪表盘。OpenTelemetry 的 GenAI SIG 通过定义一套整个生态系统共同遵循的标准来解决这个问题。

## 核心概念

### Span 分类

1. **模型 / 客户端 span。** 覆盖原始 LLM 调用。由提供方 SDK（Anthropic、OpenAI、Bedrock）和框架的模型适配器发出。
2. **智能体 span。** `create_agent`（智能体构建时）和 `invoke_agent`（智能体运行时）。
3. **工具 span。** 每次工具调用对应一个 span，通过父子关系与智能体 span 关联。

### 智能体 span 命名

- Span 名称：智能体有名称时为 `invoke_agent {gen_ai.agent.name}`；否则回退为 `invoke_agent`。
- Span 类型（kind）：
  - **CLIENT** —— 用于远程智能体服务（OpenAI Assistants API、Bedrock Agents）。
  - **INTERNAL** —— 用于进程内智能体框架（LangChain、CrewAI、本地 ReAct）。

### 关键属性

- `gen_ai.provider.name` —— `anthropic`、`openai`、`aws.bedrock`、`google.vertex`。
- `gen_ai.request.model` —— 模型 ID。
- `gen_ai.response.model` —— 实际解析出的模型（由于路由的原因，可能与请求的模型不同）。
- `gen_ai.agent.name` —— 智能体标识符。
- `gen_ai.operation.name` —— `chat`、`completion`、`invoke_agent`、`tool_call`。
- `gen_ai.data_source.id` —— 用于 RAG：本次检索查询了哪个语料库或存储。

针对 Anthropic、Azure AI Inference、AWS Bedrock、OpenAI 还有各自的技术专属约定。

### 内容捕获

默认规则：instrumentation 默认不应（SHOULD NOT）捕获输入/输出。捕获需通过以下属性显式开启：

- `gen_ai.system_instructions`
- `gen_ai.input.messages`
- `gen_ai.output.messages`

生产环境推荐模式：将内容存储在外部（S3、你的日志存储），在 span 上只记录引用（指针 ID，而非正文）。这正是第 27 课的内容投毒防御在可观测性中的落地。

### 稳定性

截至 2026 年 3 月，大多数约定仍处于实验阶段。通过以下方式选择启用稳定预览版：

```
OTEL_SEMCONV_STABILITY_OPT_IN=gen_ai_latest_experimental
```

Datadog v1.37+ 会将 GenAI 属性原生映射到其 LLM Observability schema。其他后端（Grafana、Honeycomb、Jaeger）支持原始属性。

### 这个模式的常见翻车点

- **在 span 中捕获完整提示词。** PII、密钥、客户数据出现在运维人员可读的追踪数据中。应外部存储。
- **缺少 `gen_ai.provider.name`。** 缺少归属信息时，多提供方仪表盘会失效。
- **Span 没有父级关联。** 产生孤儿工具 span。务必始终传播上下文。
- **没有设置稳定性 opt-in。** 后端升级时你的属性可能被重命名。

## 从零实现

`code/main.py` 用标准库实现了一个符合 GenAI 约定的 span 发射器：

- 带有 GenAI 属性 schema 的 `Span`。
- 带有 `start_span` 和嵌套上下文的 `Tracer`。
- 一段脚本化的智能体运行，发出：`create_agent`、`invoke_agent`（INTERNAL）、每个工具的 span、LLM 调用的 `chat` span。
- 一种内容捕获模式：将提示词存储在外部，在 span 上只记录 ID。

运行：

```
python3 code/main.py
```

输出：一棵包含所有必需 GenAI 属性的 span 树，以及一个展示 opt-in 内容引用的「外部存储」。

## 生产实践

- **Datadog LLM Observability**（v1.37+）—— 原生映射这些属性。
- **Langfuse / Phoenix / Opik**（第 24 课）—— 对整个生态自动插桩。
- **Jaeger / Honeycomb / Grafana Tempo** —— 原始 OTel 追踪；基于 GenAI 属性搭建仪表盘。
- **自托管** —— 运行带 GenAI 处理器的 OTel Collector。

## 交付产物

`outputs/skill-otel-genai.md` 将 OTel GenAI span 接入一个现有智能体，附带内容捕获默认配置和外部引用存储。

## 练习

1. 用 `invoke_agent`（INTERNAL）加上每个工具的 span 为你第 01 课的 ReAct 循环插桩，发送到一个 Jaeger 实例。
2. 添加「仅引用」模式的内容捕获：提示词存入 SQLite，span 属性只携带行 ID。
3. 阅读 `gen_ai.data_source.id` 的规范，将它接入你第 09 课的 Mem0 检索。
4. 设置 `OTEL_SEMCONV_STABILITY_OPT_IN=gen_ai_latest_experimental`，验证你的属性不会被 collector 重命名。
5. 搭建一个仪表盘：仅凭 GenAI 属性回答「哪些工具错误与哪些模型相关」。

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|----------------|------------------------|
| GenAI SIG | 「OpenTelemetry GenAI 小组」 | 定义该 schema 的 OTel 工作组 |
| invoke_agent | 「智能体 span」 | 表示一次智能体运行的 span 名称 |
| CLIENT span | 「远程调用」 | 调用远程智能体服务的 span |
| INTERNAL span | 「进程内」 | 进程内智能体运行的 span |
| gen_ai.provider.name | 「提供方」 | anthropic / openai / aws.bedrock / google.vertex |
| gen_ai.data_source.id | 「RAG 来源」 | 检索命中了哪个语料库/存储 |
| 内容捕获 | 「提示词日志」 | 消息的 opt-in 捕获；生产环境应外部存储 |
| 稳定性 opt-in | 「预览模式」 | 用于固定实验性约定的环境变量 |

## 延伸阅读

- [OpenTelemetry GenAI semantic conventions](https://opentelemetry.io/docs/specs/semconv/gen-ai/) —— 规范本身
- [OpenAI Agents SDK](https://openai.github.io/openai-agents-python/) —— 默认发出 GenAI span
- [AutoGen v0.4 (Microsoft Research)](https://www.microsoft.com/en-us/research/articles/autogen-v0-4-reimagining-the-foundation-of-agentic-ai-for-scale-extensibility-and-robustness/) —— 内置 OTel span
- [Claude Agent SDK](https://platform.claude.com/docs/en/agent-sdk/overview) —— W3C 追踪上下文传播
