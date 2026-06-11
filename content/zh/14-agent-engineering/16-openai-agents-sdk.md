# OpenAI Agents SDK：交接、护栏与追踪

> OpenAI Agents SDK 是构建在 Responses API 之上的轻量级多智能体框架。五个原语：Agent、Handoff、Guardrail、Session、Tracing。交接（handoff）是名为 `transfer_to_<agent>` 的工具。护栏（guardrail）在输入或输出端触发。追踪默认开启。

**Type:** Learn + Build
**Languages:** Python (stdlib)
**Prerequisites:** Phase 14 · 01 (Agent Loop), Phase 14 · 06 (Tool Use)
**Time:** ~75 minutes

## 学习目标

- 说出 OpenAI Agents SDK 的五个原语。
- 解释交接机制：为什么将其建模为工具、模型看到的名称形态是什么、上下文如何传递。
- 区分输入护栏、输出护栏和工具护栏；解释 `run_in_parallel` 与阻塞模式的差别。
- 用 Python 标准库实现一个带交接、护栏和 span 风格追踪的运行时。

## 问题背景

无法干净利落地委派任务的智能体，最终会把所有内容都塞进一个提示词里。没有护栏的智能体会泄露 PII、输出违反策略的内容，或者陷入无限循环。OpenAI 的 SDK 把让多智能体工作变得可控的三个原语固化了下来。

## 核心概念

### 五个原语

1. **Agent。** LLM + 指令 + 工具 + 交接。
2. **Handoff（交接）。** 把任务委派给另一个智能体。在模型眼中表现为一个名为 `transfer_to_<agent_name>` 的工具。
3. **Guardrail（护栏）。** 对输入（仅第一个智能体）、输出（仅最后一个智能体）或工具调用（针对每个函数工具）进行校验。
4. **Session（会话）。** 跨轮次自动维护对话历史。
5. **Tracing（追踪）。** 为 LLM 生成、工具调用、交接、护栏内置 span。

### 作为工具的交接

模型在工具列表中看到 `transfer_to_billing_agent`。调用它会通知运行时执行以下操作：

1. 复制对话上下文（或通过 `nest_handoff_history` beta 功能折叠上下文）。
2. 用目标智能体的指令初始化它。
3. 由目标智能体继续本次运行。

这就是监督者模式（第 13 课 / 第 28 课）的产品化版本。

### 护栏

三种类型：

- **输入护栏。** 作用于第一个智能体的输入。在任何 LLM 调用之前拒绝不安全或超出范围的请求。
- **输出护栏。** 作用于最后一个智能体的输出。拦截 PII 泄露、策略违规和格式错误的响应。
- **工具护栏。** 针对每个函数工具执行。校验参数、检查权限、审计执行过程。

运行模式：

- **并行**（默认）。护栏 LLM 与主 LLM 同时运行。尾延迟更低。一旦触发，主 LLM 的工作会被丢弃（浪费 token）。
- **阻塞**（`run_in_parallel=False`）。护栏 LLM 先运行。一旦触发，主调用不会浪费任何 token。

绊线（tripwire）触发时抛出 `InputGuardrailTripwireTriggered` / `OutputGuardrailTripwireTriggered`。

### 追踪

默认开启。每一次 LLM 生成、工具调用、交接和护栏检查都会发出一个 span。设置 `OPENAI_AGENTS_DISABLE_TRACING=1` 可以退出。`add_trace_processor(processor)` 可以把 span 同时分发到你自己的后端和 OpenAI 的后端。

### 会话

`Session` 把对话历史存储在某个后端（SQLite、Redis、自定义）。`Runner.run(agent, input, session=session)` 会自动加载并追加历史。

### 这个模式容易出错的地方

- **交接漂移。** 智能体 A 交接给智能体 B，B 又交接回 A。需要加一个跳数计数器。
- **护栏绕过。** 工具护栏只对函数工具生效；内置工具（文件读取器、网页抓取）需要单独的策略。
- **过度追踪。** 敏感内容进入 span。配合 OTel GenAI 内容捕获规则（第 23 课）——内容存到外部，span 中只保留 ID 引用。

## 从零实现

`code/main.py` 用标准库实现了 SDK 的核心形态：

- `Agent`、`FunctionTool`、`Handoff`（作为带有转移语义的函数工具）。
- 带输入/输出/工具护栏、交接分发和跳数计数器的 `Runner`。
- 一个简单的 span 发射器，用来展示追踪的形态。
- 一个分诊智能体，根据用户查询把任务交接给账单智能体或支持智能体；其中一条输入会触发护栏。

运行：

```
python3 code/main.py
```

追踪结果显示两次成功的交接、一次输入护栏触发，以及一棵与真实 SDK 输出形态一致的 span 树。

## 生产实践

- **OpenAI Agents SDK** 适用于以 OpenAI 为主的产品。
- **Claude Agent SDK**（第 17 课）适用于以 Claude 为主的产品。
- **LangGraph**（第 13 课）适用于需要显式状态和持久化恢复的场景。
- **自定义实现** 适用于需要精确控制的场景（语音、多供应商、联邦化部署）。

## 交付产物

`outputs/skill-agents-sdk-scaffold.md` 提供了一个 Agents SDK 应用脚手架，包含分诊智能体、交接、输入/输出/工具护栏、会话存储和一个追踪处理器。

## 练习

1. 加一个交接跳数计数器：超过 N 次转移后拒绝。追踪其行为。
2. 把 `nest_handoff_history` 实现为一个选项——在转移前把之前的消息折叠成一条摘要。
3. 写一个阻塞式输出护栏。对比会触发护栏的提示词与能通过的提示词之间的延迟差异。
4. 把 `add_trace_processor` 接到一个 JSON 日志记录器上。它为每个 span 输出的是什么形态？
5. 阅读 SDK 文档。把你的标准库玩具实现移植到 `openai-agents-python`。你哪里建模错了？

## 关键术语

| 术语 | 大家怎么说 | 实际含义 |
|------|----------------|------------------------|
| Agent | “LLM + 指令” | SDK 中的 Agent 类型；持有工具和交接 |
| Handoff | “转移” | 模型调用的工具，用于把任务委派给另一个智能体 |
| Guardrail | “策略检查” | 对输入 / 输出 / 工具调用的校验 |
| Tripwire | “护栏触发” | 护栏拒绝时抛出的异常 |
| Session | “历史存储” | 在多次运行之间持久化的对话记忆 |
| Tracing | “Spans” | 覆盖 LLM + 工具 + 交接 + 护栏的内置可观测性 |
| 阻塞护栏 | “顺序检查” | 护栏先运行；触发时不浪费 token |
| 并行护栏 | “并发检查” | 护栏与主调用同时运行；延迟更低，但触发时浪费 token |

## 延伸阅读

- [OpenAI Agents SDK docs](https://openai.github.io/openai-agents-python/) — 原语、交接、护栏、追踪
- [Claude Agent SDK overview](https://platform.claude.com/docs/en/agent-sdk/overview) — Claude 风格的对应方案
- [Anthropic, Building Effective Agents](https://www.anthropic.com/research/building-effective-agents) — 什么时候才需要用交接
- [OpenTelemetry GenAI semantic conventions](https://opentelemetry.io/docs/specs/semconv/gen-ai/) — Agents SDK span 所映射的标准
