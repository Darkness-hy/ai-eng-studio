# MCP 资源与提示模板 —— 工具之外的上下文暴露方式

> 工具（tools）占据了 MCP 关注度的 90%。但另外两个服务器原语解决的是不同的问题。资源（resources）暴露可供读取的数据；提示模板（prompts）将可复用模板以斜杠命令的形式暴露出来。很多服务器本应使用资源，而不是把读取操作包装成工具；本应使用提示模板，而不是把工作流硬编码进客户端提示中。本课讲清这条决策规则，并逐一过一遍 `resources/*` 和 `prompts/*` 消息。

**Type:** Build
**Languages:** Python (stdlib, resource + prompt handler)
**Prerequisites:** Phase 13 · 07 (MCP server)
**Time:** ~45 minutes

## 学习目标

- 针对给定领域，判断一个能力应暴露为工具、资源还是提示模板。
- 实现 `resources/list`、`resources/read`、`resources/subscribe`，并处理 `notifications/resources/updated`。
- 实现带参数模板的 `prompts/list` 和 `prompts/get`。
- 识别宿主何时把提示模板呈现为斜杠命令，何时作为自动注入的上下文。

## 问题背景

一个朴素的笔记应用 MCP 服务器把所有能力都暴露成工具：`notes_read`、`notes_list`、`notes_search`。这样每次数据访问都被包装成一次由模型驱动的工具调用。后果是：

- 对于每个可能受益于上下文的查询，模型都得自己判断是否调用 `notes_read`。
- 只读内容无法被订阅，也无法流式推送到宿主的侧边面板。
- 客户端 UI（Claude Desktop 的资源附加面板、Cursor 的「Include file」选择器）无法呈现这些数据。

正确的拆分方式是：数据暴露为资源，有副作用或需要计算的操作暴露为工具，可复用的多步工作流暴露为提示模板。每个原语都有自己的 UX 形态和访问模式。

## 核心概念

### 工具 vs 资源 vs 提示模板 —— 决策规则

| 能力 | 原语 |
|------------|-----------|
| 用户想搜索、过滤或转换数据 | 工具 |
| 用户想让宿主把这份数据作为上下文引入 | 资源 |
| 用户想要一个可以反复运行的模板化工作流 | 提示模板 |

判断准则：如果模型在每个相关查询中调用它都有好处，那它是工具。如果用户把它附加到对话中会有好处，那它是资源。如果用户想复用的单元是一整个多步工作流，那它是提示模板。

### 资源

`resources/list` 返回 `{resources: [{uri, name, mimeType, description?}]}`。`resources/read` 接收 `{uri}`，返回 `{contents: [{uri, mimeType, text | blob}]}`。

URI 可以是任何可寻址的标识：

- `file:///Users/alice/notes/mcp.md`
- `postgres://my-db/query/SELECT ...`
- `notes://note-14`（自定义 scheme）
- `memory://session-2026-04-22/recent`（服务器自定义）

`contents[]` 同时支持文本和二进制内容。二进制用 `blob` 字段承载 base64 编码字符串，外加一个 `mimeType`。

### 资源订阅

在能力声明中加入 `{resources: {subscribe: true}}`。客户端调用 `resources/subscribe {uri}`。资源发生变化时，服务器发送 `notifications/resources/updated {uri}`。客户端重新读取。

使用场景：一个笔记服务器的资源是磁盘上的文件；文件监视器触发更新通知；当文件在宿主之外被编辑时，Claude Desktop 重新拉取文件进入上下文。

### 资源模板（2025-11-25 新增）

`resourceTemplates` 让你暴露参数化的 URI 模式：`notes://{id}`，其中 `id` 是一个补全目标。客户端可以在资源选择器中自动补全 id。

### 提示模板

`prompts/list` 返回 `{prompts: [{name, description, arguments?}]}`。`prompts/get` 接收 `{name, arguments}`，返回 `{description, messages: [{role, content}]}`。

提示模板是一个填充后生成消息列表的模板，宿主把这些消息喂给自己的模型。例如，一个 `code_review` 提示模板接收 `file_path` 参数，返回一个三消息序列：一条系统消息、一条包含文件内容的用户消息，以及一条带推理模板的助手起手消息。

### 宿主与提示模板

Claude Desktop、VS Code 和 Cursor 在聊天 UI 中把提示模板呈现为斜杠命令。用户输入 `/code_review`，然后在表单中选择参数。服务器的提示模板就是「用户快捷方式」与「发送给模型的完整提示」之间的契约。

并非所有客户端都已支持提示模板 —— 要检查能力协商结果。如果服务器声明了 prompt 能力，而客户端不支持 prompt，那用户只是看不到这些斜杠命令而已。

### 「list changed」通知

资源和提示模板的集合发生变化时，都会发出 `notifications/list_changed`。一个刚导入 20 条新笔记的笔记服务器会发出 `notifications/resources/list_changed`；客户端重新调用 `resources/list` 来获取新增内容。

### 内容类型约定

文本：`mimeType: "text/plain"`、`text/markdown`、`application/json`。
二进制：`image/png`、`application/pdf`，外加 `blob` 字段。
MCP Apps（第 14 课）：`ui://` URI 中的 `text/html;profile=mcp-app`。

### 动态资源

资源 URI 不必对应静态文件。`notes://recent` 可以在每次读取时返回最新的五条笔记。`db://query/users/active` 可以执行一条参数化查询。服务器可以自由地动态计算内容。

规则：如果客户端可以按 URI 缓存，URI 就必须稳定。如果计算是一次性的，URI 应包含时间戳或随机数（nonce），以免客户端缓存失效不及时。

### 订阅 vs 轮询

支持订阅的客户端通过 `notifications/resources/updated` 获得服务器推送。订阅机制出现之前的客户端，或不支持订阅的宿主，则通过重复读取来轮询。两者都符合规范。服务器的能力声明会告诉客户端它支持哪种方式。

订阅的代价：服务器要维护按会话的状态（谁订阅了什么）。要让订阅集合保持有界；断开连接的客户端应当超时清理。

### 提示模板 vs 系统提示

MCP 中的提示模板不是系统提示。宿主自己的系统提示（它的运行指令）和 MCP 提示模板（由用户触发的服务器提供的模板）是并存的。行为良好的客户端绝不会让服务器的提示模板覆盖自己的系统提示，而是将两者分层叠加。

## 生产实践

`code/main.py` 在第 07 课的笔记服务器基础上扩展了：

- 按笔记划分的资源（`notes://note-1` 等），支持 `resources/subscribe`。
- 一个 `review_note` 提示模板，渲染成三消息模板。
- 一个文件监视器模拟，在笔记被修改时发出 `notifications/resources/updated`。
- 一个 `notes://recent` 动态资源，始终返回最新的五条笔记。

运行 demo 即可看到完整流程。

## 交付产物

本课产出 `outputs/skill-primitive-splitter.md`。给定一个待设计的 MCP 服务器，该技能将每个能力归类为工具 / 资源 / 提示模板，并给出理由。

## 练习

1. 运行 `code/main.py`。观察初始资源列表，然后触发一次笔记编辑，验证 `notifications/resources/updated` 事件被触发。

2. 添加一个 `resources/list_changed` 发射器：创建新笔记时发送该通知，让客户端重新发现资源。

3. 为一个 GitHub MCP 服务器设计三个提示模板：`summarize_pr`、`triage_issue`、`release_notes`。每个都带参数 schema。提示模板正文应当无需进一步修改即可直接运行。

4. 选取第 07 课服务器中的一个现有工具，判断它应该保留为工具，还是拆分为一个资源加一个工具。用一句话说明理由。

5. 阅读规范的 `server/resources` 和 `server/prompts` 章节。找出 `resources/read` 中那个很少被填充但规范支持的字段。提示：看资源内容上的 `_meta`。

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|----------------|------------------------|
| 资源（Resource） | 「暴露的数据」 | 宿主可读取的、以 URI 寻址的内容 |
| 资源 URI | 「指向数据的指针」 | 带 scheme 前缀的标识符（`file://`、`notes://` 等） |
| `resources/subscribe` | 「监听变化」 | 客户端主动选择加入的、针对特定 URI 的服务器推送更新 |
| `notifications/resources/updated` | 「资源变了」 | 通知客户端某个已订阅资源有新内容的信号 |
| 资源模板（Resource template） | 「参数化 URI」 | 带补全提示的 URI 模式，供宿主选择器使用 |
| 提示模板（Prompt） | 「斜杠命令模板」 | 带参数槽位的命名多消息模板 |
| 提示模板参数 | 「模板输入」 | 宿主在渲染前收集的带类型参数 |
| `prompts/get` | 「渲染模板」 | 服务器返回填充完成的消息列表 |
| 内容块（Content block） | 「带类型的数据块」 | `{type: text \| image \| resource \| ui_resource}` |
| 斜杠命令 UX | 「用户快捷方式」 | 宿主把提示模板呈现为以 `/` 开头的命令 |

## 延伸阅读

- [MCP — Concepts: Resources](https://modelcontextprotocol.io/docs/concepts/resources) —— 资源 URI、订阅与模板
- [MCP — Concepts: Prompts](https://modelcontextprotocol.io/docs/concepts/prompts) —— 提示模板与斜杠命令集成
- [MCP — Server resources spec 2025-11-25](https://modelcontextprotocol.io/specification/2025-11-25/server/resources) —— 完整的 `resources/*` 消息参考
- [MCP — Server prompts spec 2025-11-25](https://modelcontextprotocol.io/specification/2025-11-25/server/prompts) —— 完整的 `prompts/*` 消息参考
- [MCP — Protocol info site: resources](https://modelcontextprotocol.info/docs/concepts/resources/) —— 在官方文档基础上展开的社区指南
