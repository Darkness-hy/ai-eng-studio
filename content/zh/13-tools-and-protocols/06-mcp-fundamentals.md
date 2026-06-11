# MCP 基础 —— 原语、生命周期与 JSON-RPC 基座

> 在 MCP 出现之前，每一次集成都是一次性的。模型上下文协议（Model Context Protocol，MCP）由 Anthropic 于 2024 年 11 月首次发布，如今由 Linux Foundation 旗下的 Agentic AI Foundation 负责维护，它把工具的发现与调用标准化，使任何客户端都能与任何服务器通信。2025-11-25 版规范定义了六个原语（三个服务器侧、三个客户端侧）、一个三阶段生命周期，以及基于 JSON-RPC 2.0 的线上消息格式。掌握这些之后，本阶段 MCP 章节的其余内容就只是顺势阅读了。

**Type:** Learn
**Languages:** Python (stdlib, JSON-RPC parser)
**Prerequisites:** Phase 13 · 01 through 05 (the tool interface and function calling)
**Time:** ~45 minutes

## 学习目标

- 说出 MCP 全部六个原语（服务器侧的 tools、resources、prompts；客户端侧的 roots、sampling、elicitation），并为每个原语给出一个使用场景。
- 走通三阶段生命周期（初始化、运行、关闭），并说明每个阶段由哪一方发送哪些消息。
- 解析并生成 JSON-RPC 2.0 的请求、响应和通知信封。
- 解释 `initialize` 阶段的能力协商（capability negotiation）是什么，以及缺少它会出什么问题。

## 问题背景

在 MCP 之前，每个使用工具的智能体都有自己的一套协议。Cursor 有一个形似 MCP 却互不兼容的工具系统，Claude Desktop 自带另一套，VS Code 的 Copilot 扩展又是第三套。一个团队做了一个「Postgres 查询」工具，就得把同一个工具写三遍，分别对接不同宿主的 API。想复用，只能复制代码。

结果是一场一次性集成的「寒武纪大爆发」，生态发展速度被牢牢卡住。

MCP 通过标准化线上消息格式解决了这个问题。一个 MCP 服务器可以在所有 MCP 客户端中工作：Claude Desktop、ChatGPT、Cursor、VS Code、Gemini、Goose、Zed、Windsurf——到 2026 年 4 月已有 300 多个客户端。SDK 月下载量达 1.1 亿次，公开服务器超过 10,000 个。2025 年 12 月，Linux Foundation 在新成立的 Agentic AI Foundation 下接管了该协议的治理。

本阶段采用的规范版本是 **2025-11-25**。该版本新增了异步 Tasks（SEP-1686）、URL 模式 elicitation（SEP-1036）、带工具的 sampling（SEP-1577）、增量范围授权（SEP-835），以及 OAuth 2.1 resource-indicator 语义。Phase 13 · 09 到 16 会讲解这些扩展。本课只覆盖基础部分。

## 核心概念

### 三个服务器原语

1. **Tools（工具）。** 可调用的动作。与 Phase 13 · 01 中相同的四步循环。
2. **Resources（资源）。** 对外暴露的数据。只读内容，通过 URI 寻址：`file:///path`、`db://query/...`，以及自定义 scheme。
3. **Prompts（提示模板）。** 可复用的模板。在宿主 UI 中表现为斜杠命令；服务器提供模板，客户端填入参数。

### 三个客户端原语

4. **Roots（根范围）。** 服务器被允许访问的 URI 集合。由客户端声明，服务器必须遵守。
5. **Sampling（采样）。** 服务器请求客户端的模型执行一次补全。这使得服务器端可以运行智能体循环，而无需在服务器侧持有 API 密钥。
6. **Elicitation（信息征询）。** 服务器在执行过程中向客户端的用户索取结构化输入。形式为表单或 URL（SEP-1036）。

MCP 中的每一项能力都恰好归属于这六者之一。Phase 13 · 10 到 14 会逐一深入讲解。

### 线上格式：JSON-RPC 2.0

每条消息都是一个包含以下字段的 JSON 对象：

- 请求：`{jsonrpc: "2.0", id, method, params}`。
- 响应：`{jsonrpc: "2.0", id, result | error}`。
- 通知：`{jsonrpc: "2.0", method, params}` —— 没有 `id`，不期待响应。

基础规范约有 15 个方法，按原语分组。其中重要的有：

- `initialize` / `initialized`（握手）
- `tools/list`、`tools/call`
- `resources/list`、`resources/read`、`resources/subscribe`
- `prompts/list`、`prompts/get`
- `sampling/createMessage`（服务器发往客户端）
- `notifications/tools/list_changed`、`notifications/resources/updated`、`notifications/progress`

### 三阶段生命周期

**阶段 1：初始化。**

客户端发送 `initialize`，携带自己的 `capabilities` 和 `clientInfo`。服务器响应自己的 `capabilities`、`serverInfo`，以及它所支持的规范版本。客户端消化完响应后，发送 `notifications/initialized`。此后，双方都可以按照协商好的能力发送请求。

**阶段 2：运行。**

双向通信。客户端调用 `tools/list` 进行发现，再用 `tools/call` 进行调用。如果服务器声明了相应能力，它可以发送 `sampling/createMessage`；当工具集发生变化时，可以发送 `notifications/tools/list_changed`。当用户改变根范围时，客户端可以发送 `notifications/roots/list_changed`。

**阶段 3：关闭。**

任意一方关闭传输层即可。MCP 没有结构化的关闭方法；连接终止信号由传输层（stdio 或 Streamable HTTP，见 Phase 13 · 09）承载。

### 能力协商

`initialize` 握手中的 `capabilities` 就是契约。服务器侧的一个示例：

```json
{
  "tools": {"listChanged": true},
  "resources": {"subscribe": true, "listChanged": true},
  "prompts": {"listChanged": true}
}
```

服务器声明它可以发出 `tools/list_changed` 通知，并支持 `resources/subscribe`。客户端则声明自己的能力作为回应：

```json
{
  "roots": {"listChanged": true},
  "sampling": {},
  "elicitation": {}
}
```

如果客户端没有声明 `sampling`，服务器就绝不能调用 `sampling/createMessage`。这是对称的：如果服务器没有声明 `resources.subscribe`，客户端就不能尝试订阅。

正是这一机制防止了生态分裂。不支持 sampling 的客户端仍然是合法的 MCP 客户端；不调用 `sampling` 的服务器仍然是合法的 MCP 服务器。两者只是不会一起使用这项功能而已。

### 结构化内容与错误形态

`tools/call` 返回一个由类型化块组成的 `content` 数组：`text`、`image`、`resource`。Phase 13 · 14 会把 MCP Apps（`ui://` 交互式 UI）加入这个列表。

错误采用 JSON-RPC 错误码。规范额外定义了：`-32002`「Resource not found」、`-32603`「Internal error」，以及通过 `error.data` 携带的 MCP 专有错误数据。

### 客户端能力 vs 工具调用细节

一个常见的混淆点：`capabilities.tools` 表示的是客户端是否支持工具列表变更通知。至于客户端「会不会」调用某个具体工具，是由其模型驱动的运行时决策，而不是能力标志。能力标志是规范层面的契约，模型的选择与之正交。

### 为什么用 JSON-RPC 而不是 REST？

JSON-RPC 2.0（2010 年）是一个轻量的双向协议，而 REST 只能由客户端发起。MCP 需要服务器主动发起的消息（sampling、通知），因此具有对称请求/响应结构的 JSON-RPC 是自然之选。JSON-RPC 还能干净地运行在 stdio 和 WebSocket/Streamable HTTP 之上，无需重新发明 HTTP 的请求结构。

```figure
mcp-tool-call
```

## 生产实践

`code/main.py` 提供了一个最小化的 JSON-RPC 2.0 解析器和生成器，然后手工走一遍 `initialize` → `tools/list` → `tools/call` → `shutdown` 序列，打印每一条消息。没有真实的传输层，只演示消息的形态。可对照「延伸阅读」中链接的规范，逐一核对每个信封。

值得关注的点：

- `initialize` 双向声明能力；响应中包含 `serverInfo` 和 `protocolVersion: "2025-11-25"`。
- `tools/list` 返回一个 `tools` 数组；每个条目都有 `name`、`description`、`inputSchema`。
- `tools/call` 使用 `params.name` 和 `params.arguments`。
- 响应中的 `content` 是由 `{type, text}` 块组成的数组。

## 交付产物

本课产出 `outputs/skill-mcp-handshake-tracer.md`。给定一段 pcap 风格的 MCP 客户端-服务器交互记录，该 skill 会为每条消息标注它属于哪个原语、处于哪个生命周期阶段，以及依赖哪项能力。

## 练习

1. 运行 `code/main.py`。找出发生能力协商的那一行，并描述如果服务器没有声明 `tools.listChanged` 会发生什么变化。

2. 扩展解析器以处理 `notifications/progress`。消息形态为：`{method: "notifications/progress", params: {progressToken, progress, total}}`。在一个长时间运行的 `tools/call` 进行期间发出该通知，并确认客户端处理器能够据此显示进度条。

3. 从头到尾通读 MCP 2025-11-25 规范——全文约 80 页。找出大多数服务器并不需要的那个能力标志。提示：它与资源订阅有关。

4. 在纸上推演：假设要加一个「cron 定时任务」功能，它应该归属于哪个原语？（提示：服务器希望客户端在预定时间调用它。现有六个原语都不适用。）MCP 的 2026 路线图中有一份针对此功能的 SEP 草案。

5. 解析 GitHub 上某个开源 MCP 服务器的一份会话日志。统计请求、响应、通知消息各有多少条，并计算生命周期消息与运行阶段消息各占流量的比例。

## 关键术语

| 术语 | 通俗说法 | 实际含义 |
|------|----------------|------------------------|
| MCP | 「Model Context Protocol」 | 用于模型到工具的发现与调用的开放协议 |
| 服务器原语 | 「服务器对外暴露的东西」 | tools（动作）、resources（数据）、prompts（模板） |
| 客户端原语 | 「客户端允许服务器使用的东西」 | roots（范围）、sampling（LLM 回调）、elicitation（用户输入） |
| JSON-RPC 2.0 | 「线上消息格式」 | 对称的请求/响应/通知信封 |
| `initialize` 握手 | 「能力协商」 | 第一对消息；服务器与客户端各自声明支持的功能 |
| `tools/list` | 「发现」 | 客户端向服务器请求其当前工具集 |
| `tools/call` | 「调用」 | 客户端请求服务器带参数执行某个工具 |
| `notifications/*_changed` | 「变更事件」 | 服务器告知客户端其原语列表已发生变化 |
| 内容块 | 「类型化结果」 | 工具结果中的 `{type: "text" \| "image" \| "resource" \| "ui_resource"}` |
| SEP | 「Spec Evolution Proposal」 | 具名的规范演进提案草案（例如异步 Tasks 的 SEP-1686） |

## 延伸阅读

- [Model Context Protocol — Specification 2025-11-25](https://modelcontextprotocol.io/specification/2025-11-25) —— 权威规范文档
- [Model Context Protocol — Architecture concepts](https://modelcontextprotocol.io/docs/concepts/architecture) —— 六原语心智模型
- [Anthropic — Introducing the Model Context Protocol](https://www.anthropic.com/news/model-context-protocol) —— 2024 年 11 月的发布公告
- [MCP blog — First MCP anniversary](https://blog.modelcontextprotocol.io/posts/2025-11-25-first-mcp-anniversary/) —— 一周年回顾及 2025-11-25 规范变更
- [WorkOS — MCP 2025-11-25 spec update](https://workos.com/blog/mcp-2025-11-25-spec-update) —— SEP-1686、1036、1577、835 与 1724 的摘要
