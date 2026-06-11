# 函数调用深入解析 — OpenAI、Anthropic、Gemini

> 2024 年，三家前沿模型提供商在工具调用循环上殊途同归，却在其余所有细节上各走各路。OpenAI 用 `tools` 和 `tool_calls`；Anthropic 用 `tool_use` 和 `tool_result` 内容块；Gemini 用 `functionDeclarations` 和唯一 id 关联机制。本节课将三者并排对比，确保在一家提供商上跑通的代码移植到另一家时不会出问题。

**Type:** Build
**Languages:** Python (stdlib, schema translators)
**Prerequisites:** Phase 13 · 01 (the tool interface)
**Time:** ~75 minutes

## 学习目标

- 说出 OpenAI、Anthropic、Gemini 三家函数调用载荷在声明、调用、结果三个环节上的结构差异。
- 把同一个工具声明翻译成三种提供商格式，并预判严格模式（strict mode）约束在哪些地方会不一致。
- 在每家提供商中使用 `tool_choice` 来强制、禁止或自动选择工具调用。
- 了解各提供商的硬性限制（工具数量、schema 深度、参数长度），以及超出限制时各自抛出的错误特征。

## 问题背景

函数调用请求的结构因提供商而异。以下是 2026 年生产环境中的三个具体例子：

**OpenAI Chat Completions / Responses API。** 你传入 `tools: [{type: "function", function: {name, description, parameters, strict}}]`。模型的响应包含 `choices[0].message.tool_calls: [{id, type: "function", function: {name, arguments}}]`，其中 `arguments` 是一个需要你自行解析的 JSON 字符串。严格模式（`strict: true`）通过约束解码（constrained decoding）来强制 schema 合规。

**Anthropic Messages API。** 你传入 `tools: [{name, description, input_schema}]`。响应以 `content: [{type: "text"}, {type: "tool_use", id, name, input}]` 的形式返回。`input` 已经是解析好的对象（不是字符串）。你需要用一条新的 `user` 消息回复，其中包含一个 `{type: "tool_result", tool_use_id, content}` 块。

**Google Gemini API。** 你传入 `tools: [{functionDeclarations: [{name, description, parameters}]}]`（嵌套在 `functionDeclarations` 之下）。响应以 `candidates[0].content.parts: [{functionCall: {name, args, id}}]` 的形式到达，其中 `id` 在 Gemini 3 及之后版本中是唯一的，用于并行调用的关联。你用 `{functionResponse: {name, id, response}}` 来回复。

同样的循环，却是不同的字段名、不同的嵌套层级、不同的字符串与对象约定、不同的关联机制。一个团队在 OpenAI 上写好天气智能体之后，仅仅为了这些管道差异，移植到 Anthropic 要花两天，再移植到 Gemini 又要一天。

本节课将构建一个翻译器，把三种格式统一成一份规范化的工具声明，只在边界处做路由。Phase 13 · 17 会把同样的模式推广成一个 LLM 网关。

## 核心概念

### 共同结构

每家提供商都需要五样东西：

1. **工具列表。** 每个工具的名称、描述和输入 schema。
2. **工具选择。** 强制使用某个工具、禁止使用工具，或让模型自行决定。
3. **调用发出。** 给出工具名称和参数的结构化输出。
4. **调用 id。** 把响应关联到正确的调用（在并行场景下很重要）。
5. **结果注入。** 一条消息或一个内容块，把结果与原调用绑定起来。

### 逐字段对比结构差异

| 维度 | OpenAI | Anthropic | Gemini |
|--------|--------|-----------|--------|
| 声明外层结构 | `{type: "function", function: {...}}` | `{name, description, input_schema}` | `{functionDeclarations: [{...}]}` |
| Schema 字段 | `parameters` | `input_schema` | `parameters` |
| 响应容器 | assistant 消息上的 `tool_calls[]` | 类型为 `tool_use` 的 `content[]` | 类型为 `functionCall` 的 `parts[]` |
| 参数类型 | 字符串化的 JSON | 已解析的对象 | 已解析的对象 |
| Id 格式 | `call_...`（OpenAI 生成） | `toolu_...`（Anthropic） | UUID（Gemini 3+） |
| 结果块 | role 为 `tool`，带 `tool_call_id` | 带 `tool_result` 和 `tool_use_id` 的 `user` 消息 | 带匹配 `id` 的 `functionResponse` |
| 强制使用某个工具 | `tool_choice: {type: "function", function: {name}}` | `tool_choice: {type: "tool", name}` | `tool_config: {function_calling_config: {mode: "ANY"}}` |
| 禁止使用工具 | `tool_choice: "none"` | `tool_choice: {type: "none"}` | `mode: "NONE"` |
| 严格 schema | `strict: true` | schema 即契约（始终生效） | 请求级别的 `responseSchema` |

### 实际会撞上的限制

- **OpenAI。** 每个请求最多 128 个工具。Schema 深度上限 5。参数字符串不超过 8192 字节。严格模式要求：不能有 `$ref`，不能有存在重叠的 `oneOf`/`anyOf`/`allOf`，所有属性都必须列在 `required` 里。
- **Anthropic。** 每个请求最多 64 个工具。Schema 深度实际上不设限，但实践中以 10 为限。没有严格模式开关；schema 是一份契约，模型通常会遵守。
- **Gemini。** 每个请求最多 64 个函数。Schema 类型采用 OpenAPI 3.0 子集（与 JSON Schema 2020-12 有细微差异）。自 Gemini 3 起并行调用带唯一 id。

### `tool_choice` 的行为

三种模式各家都支持，只是名字不同。

- **Auto。** 模型自行选择调用工具还是输出文本。默认值。
- **Required / Any。** 模型必须至少调用一个工具。
- **None。** 模型不得调用工具。

此外，每家提供商各有一个独有模式：

- **OpenAI。** 按名称强制使用某个特定工具。
- **Anthropic。** 按名称强制使用某个特定工具；`disable_parallel_tool_use` 标志区分单次调用和多次调用。
- **Gemini。** `mode: "VALIDATED"` 会让每个响应都经过 schema 校验器，无论模型意图如何。

### 并行调用

OpenAI 的 `parallel_tool_calls: true`（默认开启）会在一条 assistant 消息中发出多个调用。你把它们全部执行，再用一条批量的 tool 角色消息回复，每个 `tool_call_id` 对应一条结果。Anthropic 早期只支持单次调用；`disable_parallel_tool_use: false`（自 Claude 3.5 起为默认值）启用了多调用。Gemini 2 允许并行调用但没有稳定的 id；Gemini 3 加入了 UUID，使乱序返回的结果也能干净地关联。

### 流式传输

三家都支持流式工具调用，但线上格式各不相同：

- **OpenAI。** `tool_calls[i].function.arguments` 的增量分块逐步到达。你持续累积，直到收到 `finish_reason: "tool_calls"`。
- **Anthropic。** block-start / block-delta / block-stop 事件流。`input_json_delta` 分块携带部分参数。
- **Gemini。** `streamFunctionCallArguments`（Gemini 3 新增）发出的分块带有 `functionCallId`，因此多个并行调用可以交错传输。

Phase 13 · 03 会深入讲解并行与流式重组。本节课专注于声明和单次调用的结构。

### 错误与修复

参数非法时的错误表现也各不相同。

- **OpenAI（非严格模式）。** 模型返回 `arguments: "{bad json}"`，你的 JSON 解析失败，于是注入一条错误消息并重新调用。
- **OpenAI（严格模式）。** 校验在解码过程中完成；非法 JSON 不可能出现，但可能出现 `refusal`。
- **Anthropic。** `input` 中可能出现预期之外的字段；schema 只是建议性的。需要在服务端自行校验。
- **Gemini。** OpenAPI 3.0 的怪癖：对象字段上的 `enum` 会被静默忽略；需要自行校验。

### 翻译器模式

代码中的规范化工具声明长这样（具体形状由你决定）：

```python
Tool(
    name="get_weather",
    description="Use when ...",
    input_schema={"type": "object", "properties": {...}, "required": [...]},
    strict=True,
)
```

三个小函数把它翻译成三种提供商格式。`code/main.py` 中的测试框架正是这样做的，随后再把一个伪造的工具调用按各家提供商的响应结构走一遍往返流程。全程无需网络 — 本节课教的是结构，不是 HTTP。

生产团队会把这个翻译器封装进 `AbstractToolset`（Pydantic AI）、`UniversalToolNode`（LangGraph）或 `BaseTool`（LlamaIndex）。Phase 13 · 17 会交付一个网关，在三家提供商之上对外暴露一个 OpenAI 风格的 API。

## 生产实践

`code/main.py` 定义了一个规范化的 `Tool` 数据类，以及三个分别生成 OpenAI、Anthropic、Gemini 声明 JSON 的翻译器。接着，它把各家结构的手工构造的提供商响应解析成同一个规范化调用对象，证明在表象之下三者语义完全一致。运行它，把三份声明并排对比一番。

值得关注的点：

- 三个声明块只在外层结构和字段名上有差异。
- 三个响应块的差异在于调用所处的位置（顶层 `tool_calls`、`content[]` 块、`parts[]` 条目）。
- 一个 `canonical_call()` 函数就能从三种响应结构中提取出 `{id, name, args}`。

## 交付产物

本节课产出 `outputs/skill-provider-portability-audit.md`。给定一个针对某家提供商的函数调用集成，该技能会生成一份可移植性审计报告：它依赖了哪些提供商限制、哪些字段需要改名，以及移植到另外两家时各会出什么问题。

## 练习

1. 运行 `code/main.py`，验证三份提供商声明 JSON 序列化的都是同一个底层 `Tool` 对象。修改规范化工具，加入一个枚举参数，确认只有 Gemini 翻译器需要处理那个 OpenAPI 怪癖。

2. 为每家提供商添加一个 `ListToolsResponse` 解析器，用于提取模型在 `list_tools` 或发现调用之后返回的工具列表。OpenAI 原生没有这个机制；请记录这种不对称。

3. 实现 `tool_choice` 转换：把规范化的 `ToolChoice(mode="force", tool_name="x")` 映射成全部三种提供商格式。再映射 `mode="any"` 和 `mode="none"`。对照本课的差异对比表检查。

4. 从三家提供商中挑一家，把它的函数调用指南从头读到尾。在其 schema 规范中找出一个另外两家不支持的字段。候选项：OpenAI 的 `strict`、Anthropic 的 `disable_parallel_tool_use`、Gemini 的 `function_calling_config.allowed_function_names`。

5. 编写一个测试向量：一个参数违反所声明 schema 的工具调用。让它依次通过各家提供商的校验器（第 01 课中的标准库校验器可以作为替身），记录各自触发了哪些错误。写明在追求严格性时你会在生产中选用哪家提供商。

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|----------------|------------------------|
| 函数调用（Function calling） | 「工具使用」 | 提供商级别的 API，用于发出结构化工具调用 |
| 工具声明 | 「工具规格」 | 名称 + 描述 + JSON Schema 输入载荷 |
| `tool_choice` | 「强制 / 禁止」 | auto / required / none / 指定名称四种模式 |
| 严格模式 | 「schema 强制」 | OpenAI 的标志位，把解码约束在 schema 范围内 |
| `tool_use` 块 | 「Anthropic 的调用结构」 | 内联内容块，包含 id、name、input |
| `functionCall` 部件 | 「Gemini 的调用结构」 | 一个 `parts[]` 条目，包含 name、args 和 id |
| 参数即字符串 | 「字符串化的 JSON」 | OpenAI 把参数作为 JSON 字符串而非对象返回 |
| 并行工具调用 | 「单轮扇出」 | 一条 assistant 消息中包含多个工具调用 |
| 拒绝（Refusal） | 「模型拒绝」 | 仅在严格模式下出现的拒绝块，代替工具调用 |
| OpenAPI 3.0 子集 | 「Gemini schema 怪癖」 | Gemini 使用一种类 JSON Schema 方言，存在细微差异 |

## 延伸阅读

- [OpenAI — Function calling guide](https://platform.openai.com/docs/guides/function-calling) — 权威参考，涵盖严格模式与并行调用
- [Anthropic — Tool use overview](https://docs.anthropic.com/en/docs/agents-and-tools/tool-use/overview) — `tool_use` 与 `tool_result` 块的语义
- [Google — Gemini function calling](https://ai.google.dev/gemini-api/docs/function-calling) — 并行调用、唯一 id 与 OpenAPI 子集
- [Vertex AI — Function calling reference](https://docs.cloud.google.com/vertex-ai/generative-ai/docs/multimodal/function-calling) — Gemini 的企业级入口
- [OpenAI — Structured outputs](https://platform.openai.com/docs/guides/structured-outputs) — 严格模式 schema 强制的细节
