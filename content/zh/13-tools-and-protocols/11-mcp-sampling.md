# MCP Sampling —— 服务器发起的 LLM 补全与 Agent 循环

> 大多数 MCP 服务器只是「无脑执行器」：接收参数、运行代码、返回内容。采样（Sampling）让服务器反转方向：由它请求客户端的 LLM 来做决策。这使得服务器可以承载 Agent 循环，而无需持有任何模型凭证。2025-11-25 合并的 SEP-1577 允许在采样请求中携带工具，让循环可以包含更深层的推理。漂移风险提示：SEP-1577 的「采样中带工具」形态在 2026 年第一季度仍处于实验阶段，其 SDK API 尚未完全稳定。

**Type:** Build
**Languages:** Python (stdlib, sampling harness)
**Prerequisites:** Phase 13 · 07 (MCP server), Phase 13 · 10 (resources and prompts)
**Time:** ~75 minutes

## 学习目标

- 解释 `sampling/createMessage` 解决了什么问题（服务器承载循环，但不需要服务器端 API key）。
- 实现一个服务器，让客户端基于多轮提示词进行采样，并返回补全结果。
- 使用 `modelPreferences`（成本 / 速度 / 智能优先级）来引导客户端的模型选择。
- 构建一个 `summarize_repo` 工具，让它在内部通过采样迭代，而不是把行为硬编码。

## 问题背景

一个用于代码摘要工作流的 MCP 服务器需要：遍历文件树、挑选要读取的文件、综合生成摘要、然后返回。那么 LLM 推理应该发生在哪里？

方案 A：服务器调用自己的 LLM。需要 API key，费用记在服务器端，按用户计算成本高昂。

方案 B：服务器返回原始内容，由客户端的 Agent 来做推理。可行，但把服务器逻辑挪进了客户端提示词中，非常脆弱。

方案 C：服务器通过 `sampling/createMessage` 请求客户端的 LLM。服务器保留算法（读哪些文件、做几轮处理），而客户端保留计费和模型选择权。服务器完全不持有任何凭证。

采样就是方案 C。它是让一个受信任的服务器能够承载 Agent 循环、却又不必自己成为完整 LLM 宿主的机制。

## 核心概念

### `sampling/createMessage` 请求

服务器发送：

```json
{
  "jsonrpc": "2.0",
  "id": 42,
  "method": "sampling/createMessage",
  "params": {
    "messages": [{"role": "user", "content": {"type": "text", "text": "..."}}],
    "systemPrompt": "...",
    "includeContext": "none",
    "modelPreferences": {
      "costPriority": 0.3,
      "speedPriority": 0.2,
      "intelligencePriority": 0.5,
      "hints": [{"name": "claude-3-5-sonnet"}]
    },
    "maxTokens": 1024
  }
}
```

客户端运行自己的 LLM，返回：

```json
{"jsonrpc": "2.0", "id": 42, "result": {
  "role": "assistant",
  "content": {"type": "text", "text": "..."},
  "model": "claude-3-5-sonnet-20251022",
  "stopReason": "endTurn"
}}
```

### `modelPreferences`

三个总和为 1.0 的浮点数：

- `costPriority`：偏好更便宜的模型。
- `speedPriority`：偏好更快的模型。
- `intelligencePriority`：偏好能力更强的模型。

再加上 `hints`：服务器偏好的具名模型列表。客户端可以遵循也可以忽略这些提示；客户端的用户配置永远优先。

### `includeContext`

三个取值：

- `"none"` —— 只包含服务器提供的消息。默认值。
- `"thisServer"` —— 包含本服务器会话中的历史消息。
- `"allServers"` —— 包含整个会话的全部上下文。

`includeContext` 自 2025-11-25 起被软弃用（soft-deprecated），因为它会泄漏跨服务器上下文，存在安全隐患。建议优先使用 `"none"`，并在消息中显式传入所需上下文。

### 带工具的采样（SEP-1577）

2025-11-25 的新特性：采样请求可以包含一个 `tools` 数组。客户端会使用这些工具运行一个完整的工具调用循环。这让服务器可以借助客户端的模型承载 ReAct 风格的 Agent 循环。

```json
{
  "messages": [...],
  "tools": [
    {"name": "fetch_url", "description": "...", "inputSchema": {...}}
  ]
}
```

客户端循环执行：采样，若有工具调用则执行工具，再采样，最后返回最终的 assistant 消息。该特性在 2026 年第一季度仍属实验性质；SDK 接口签名可能继续变动。实现时请对照 2025-11-25 规范中的 client/sampling 章节确认。

### 人在回路（Human-in-the-loop）

客户端必须（MUST）在运行采样前向用户展示服务器要求模型做什么。恶意服务器可能利用采样来操纵用户会话（「对用户说 X，让他们点击 Y」）。Claude Desktop、VS Code 和 Cursor 都会把采样请求呈现为一个确认对话框，用户可以拒绝。

2026 年的共识：没有人工确认的采样是一个危险信号。网关（Phase 13 · 17）可以自动批准低风险采样，并自动拒绝任何可疑请求。

### 无 API key 的服务器承载循环

经典用例：一个自身没有任何 LLM 访问权限的代码摘要 MCP 服务器。它的流程是：

1. 遍历仓库结构。
2. 调用 `sampling/createMessage`，附上「挑出最可能描述这个仓库用途的五个文件」。
3. 读取这些文件。
4. 携带文件内容调用 `sampling/createMessage`，附上「用 3 段话总结这个仓库」。
5. 把摘要作为 `tools/call` 结果返回。

服务器从未接触任何 LLM API。客户端的用户使用自己的凭证为这些补全付费。

### 安全风险（Unit 42 披露，2026 年第一季度）

- **隐蔽采样（Covert sampling）。** 某个工具总是带着「从会话上下文中找出用户的邮箱并回复」发起采样。Phase 13 · 15 会讲解这些攻击向量。
- **借采样窃取资源。** 服务器让客户端去总结攻击者的载荷，账单算在用户头上。
- **循环炸弹（Loop bomb）。** 服务器在紧凑循环中疯狂调用采样。客户端必须（MUST）实施按会话的速率限制。

## 生产实践

`code/main.py` 提供了一个模拟的「服务器到客户端」采样测试框架。一个模拟的 "summarize_repo" 工具发起两轮采样（先挑文件，再总结），由假客户端返回预设响应。该框架演示了：

- 服务器发送带 `modelPreferences` 的 `sampling/createMessage`。
- 客户端返回一条补全。
- 服务器继续自己的循环。
- 速率限制器对每次工具调用的总采样次数设置上限。

值得关注的点：

- 服务器只暴露一个工具（`summarize_repo`）；所有推理都发生在采样调用中。
- 模型偏好为客户端的模型选择加权；hints 列出偏好的模型。
- 循环在 `stopReason: "endTurn"` 时终止。
- `max_samples_per_tool = 5` 这个限制能拦住失控的循环。

## 交付产物

本课产出 `outputs/skill-sampling-loop-designer.md`。给定一个需要 LLM 调用的服务器端算法（调研、摘要、规划），该 skill 会设计出基于采样的实现方案，配上合适的 modelPreferences、速率限制和安全确认机制。

## 练习

1. 运行 `code/main.py`。把 `max_samples_per_tool` 改成 2，观察速率限制的截断效果。

2. 实现 SEP-1577 的「采样中带工具」变体：采样请求携带一个 `tools` 数组。验证客户端侧的循环会先执行这些工具，再返回最终补全。注意漂移风险：SDK 接口签名在 2026 年上半年可能仍会变化。

3. 加入人在回路确认：在服务器发出第一个 `sampling/createMessage` 之前暂停，等待用户批准。被拒绝的调用返回一个类型化的拒绝响应。

4. 增加一个以客户端会话为 key 的按用户速率限制器。同一用户在同一服务器上的多个循环应共享同一份预算。

5. 设计一个 `summarize_pdf` 工具，用采样来挑选要纳入的内容块。画出要发送的消息草图。当 `modelPreferences.intelligencePriority` 取 0.1 和 0.9 时，行为会有什么不同？

## 关键术语

| 术语 | 大家怎么说 | 实际含义 |
|------|----------------|------------------------|
| Sampling | 「服务器到客户端的 LLM 调用」 | 服务器请求客户端的模型生成一条补全 |
| `sampling/createMessage` | 「那个方法」 | 用于采样请求的 JSON-RPC 方法 |
| `modelPreferences` | 「模型优先级」 | 成本 / 速度 / 智能权重，外加具名模型提示 |
| `includeContext` | 「跨会话泄漏」 | 已被软弃用的上下文包含模式 |
| SEP-1577 | 「采样里的工具」 | 允许在采样中携带工具，实现服务器承载的 ReAct |
| Human-in-the-loop | 「用户确认」 | 客户端在运行前把采样请求呈现给用户 |
| Loop bomb | 「失控采样」 | 服务器端的无限采样循环；客户端必须做速率限制 |
| Covert sampling | 「隐藏的推理」 | 恶意服务器把真实意图藏在采样提示词里 |
| Resource theft | 「花用户的 LLM 预算」 | 服务器迫使客户端为它不想要的采样买单 |
| `stopReason` | 「生成为何停止」 | `endTurn`、`stopSequence` 或 `maxTokens` |

## 延伸阅读

- [MCP — Concepts: Sampling](https://modelcontextprotocol.io/docs/concepts/sampling) —— 采样的高层概览
- [MCP — Client sampling spec 2025-11-25](https://modelcontextprotocol.io/specification/2025-11-25/client/sampling) —— `sampling/createMessage` 的权威定义
- [MCP — GitHub SEP-1577](https://github.com/modelcontextprotocol/modelcontextprotocol) —— 关于采样中带工具的规范演进提案（实验性）
- [Unit 42 — MCP attack vectors](https://unit42.paloaltonetworks.com/model-context-protocol-attack-vectors/) —— 隐蔽采样与资源窃取攻击模式
- [Speakeasy — MCP sampling core concept](https://www.speakeasy.com/mcp/core-concepts/sampling) —— 附带客户端代码示例的逐步讲解
