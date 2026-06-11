# 构建 MCP 客户端 —— 发现、调用与会话管理

> 大多数 MCP 教程只讲服务器，对客户端一笔带过。但真正复杂的编排逻辑都在客户端代码里：进程启动、能力协商、跨多个服务器的工具列表合并、采样回调、重连，以及命名空间冲突的解决。本课将构建一个多服务器客户端，把三个不同的 MCP 服务器汇聚成一个供模型使用的扁平工具命名空间。

**Type:** Build
**Languages:** Python (stdlib, multi-server MCP client)
**Prerequisites:** Phase 13 · 07 (building an MCP server)
**Time:** ~75 minutes

## 学习目标

- 以子进程方式启动 MCP 服务器，完成 `initialize`，并发送 `notifications/initialized`。
- 维护每个服务器各自的会话状态（能力、工具列表、最近收到的通知 id）。
- 将多个服务器的工具列表合并为一个命名空间，并处理命名冲突。
- 将工具调用路由到拥有该工具的服务器，并组装返回结果。

## 问题背景

真实的智能体宿主（Claude Desktop、Cursor、Goose、Gemini CLI）会同时加载多个 MCP 服务器。用户可能同时运行着文件系统服务器、Postgres 服务器和 GitHub 服务器。客户端要做的事情是：

1. 启动每个服务器。
2. 与每个服务器独立完成握手。
3. 对每个服务器调用 `tools/list`，并将结果拍平。
4. 当模型发出 `notes_search` 调用时，在合并后的命名空间中查找它，并路由到正确的服务器。
5. 处理来自任意服务器的通知（`tools/list_changed`），且不能阻塞。
6. 在传输层故障时重连。

亲手实现这一整套，正是「玩具」和「可用」的分界线。官方 SDK 封装了这些细节，但心智模型必须是你自己的。

## 核心概念

### 子进程启动

使用 `subprocess.Popen`，并设置 `stdin=PIPE, stdout=PIPE, stderr=PIPE`。设置 `bufsize=1` 并使用文本模式，以便逐行读取。每个服务器对应一个进程；客户端为每个服务器持有一个 `Popen` 句柄。

### 每个服务器各自的会话状态

每个服务器对应一个 `Session` 对象，包含：

- `process` —— Popen 句柄。
- `capabilities` —— 服务器在 `initialize` 时声明的能力。
- `tools` —— 最近一次 `tools/list` 的结果。
- `pending` —— 请求 id 到等待响应的 promise/future 的映射。

请求本质上是异步的；在服务器 B 还在处理调用时，向服务器 A 发送的 `tools/call` 不能被阻塞。要么用线程加队列，要么用 asyncio。

### 合并命名空间

当客户端汇总所有工具列表时，名称可能冲突。两个服务器可能都暴露了 `search`。客户端有三种选择：

1. **按服务器名加前缀。** `notes/search`、`files/search`。清晰但不美观。
2. **静默先到先得。** 后加载的服务器的 `search` 覆盖先前的。有风险，会掩盖冲突。
3. **拒绝冲突。** 拒绝加载第二个服务器并通知用户。对安全敏感的宿主来说最安全。

Claude Desktop 使用按服务器加前缀的方式。Cursor 使用冲突拒绝并给出明确报错。VS Code MCP 同样采用按服务器加前缀。

### 路由

合并完成后，用一张分发表（dispatch table）建立 `tool_name -> session` 的映射。模型按名称发出调用；客户端找到对应会话，向该服务器的 stdin 写入一条 `tools/call` 消息，然后等待响应。

### 采样回调

如果服务器在 `initialize` 时声明了 `sampling` 能力，它可能会发送 `sampling/createMessage`，请求客户端运行自己的 LLM。客户端必须：

1. 在采样完成前阻塞对该服务器的后续请求，或者在其实现支持并发时采用流水线方式处理。
2. 调用自己的 LLM 提供方。
3. 把响应发回给服务器。

第 11 课会完整讲解采样。本课为完整性起见只做桩实现（stub）。

### 通知处理

`notifications/tools/list_changed` 意味着需要重新调用 `tools/list`。`notifications/resources/updated` 意味着如果资源正在使用，需要重新读取它。通知不能产生响应 —— 不要试图对它们回 ack。

一个常见的客户端 bug：在 `tools/call` 上阻塞读循环，而一条通知正卡在流里。应当使用后台读取线程，把每条消息推入队列；主线程出队并分发。

### 重连

传输层可能失败：服务器崩溃、操作系统杀掉了进程、stdio 管道断裂。客户端在 stdout 上检测到 EOF，就把该会话视为已死亡。可选方案：

- 静默重启服务器并重新握手。适用于纯只读服务器。
- 把故障呈现给用户。适用于具有用户可见会话的有状态服务器。

Phase 13 · 09 会讲 Streamable HTTP 的重连语义；stdio 要简单得多。

### 保活与会话 id

Streamable HTTP 使用 `Mcp-Session-Id` 请求头。stdio 没有会话 id —— 进程本身就是会话。保活 ping 是可选的；stdio 管道不会因空闲而断开。

## 生产实践

`code/main.py` 以子进程方式启动三个模拟的 MCP 服务器，与每个服务器握手，合并它们的工具列表，并把工具调用路由到正确的服务器。这些「服务器」实际上是运行玩具应答器的其他 Python 进程（没有真实 LLM）。运行它，你会看到：

- 三次初始化，每个服务器都有自己的能力集合。
- 三个 `tools/list` 结果合并成一个含 7 个工具的命名空间。
- 一次基于工具名称的路由决策。
- 一次通过命名空间前缀避免的冲突。

值得关注的地方：

- `Session` dataclass 干净地保存了每个服务器的状态。
- 后台读取线程在不阻塞主线程的情况下读取 stdout 的每一行。
- 分发表只是一个简单的 `dict[str, Session]`。
- 冲突处理是显式的：当两个服务器声明了相同的名称时，后者会被加上前缀重命名。

## 交付产物

本课产出 `outputs/skill-mcp-client-harness.md`。给定一份 MCP 服务器的声明式清单（名称、命令、参数），该 skill 会生成一个 harness：启动这些服务器、合并工具列表，并提供一个带冲突解决的路由函数。

## 练习

1. 运行 `code/main.py`，观察服务器启动日志。用 SIGTERM 杀掉其中一个模拟服务器进程，观察客户端如何检测到 EOF 并把该会话标记为死亡。

2. 实现命名空间前缀。当两个服务器都暴露 `search` 时，把第二个重命名为 `<server>/search`。更新分发表，并验证工具调用能正确路由。

3. 为服务器重启添加类似连接池风格的退避策略：连续失败时指数退避，上限 30 秒，连续失败三次后向用户发出通知。

4. 设计一个支持 100 个并发 MCP 服务器的客户端草图。什么数据结构可以替代简单的分发字典？（提示：用 trie 做前缀命名空间，再加一个按服务器统计工具数的指标。）

5. 把客户端移植到官方的 MCP Python SDK。SDK 封装了 `stdio_client` 和 `ClientSession`。代码应当从约 200 行缩减到约 40 行，同时保留多服务器路由能力。

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|----------------|------------------------|
| MCP 客户端 | 「智能体宿主」 | 启动服务器并编排工具调用的进程 |
| 会话（Session） | 「每个服务器的状态」 | 能力、工具列表，以及待处理请求的记账信息 |
| 合并命名空间 | 「一份工具列表」 | 跨所有活跃服务器的扁平工具名称集合 |
| 命名空间冲突 | 「两个服务器同名工具」 | 客户端必须对重复名称加前缀、拒绝或先到先得 |
| 路由 | 「这个调用归谁？」 | 从工具名称到所属服务器的分发 |
| 后台读取器 | 「非阻塞 stdout」 | 把服务器 stdout 排空到队列中的线程或任务 |
| 采样回调 | 「LLM 即服务」 | 客户端处理来自服务器的 `sampling/createMessage` |
| `notifications/*_changed` | 「原语变了」 | 客户端必须重新发现或重新读取的信号 |
| 重连策略 | 「服务器死了怎么办」 | 传输层故障时的重启语义 |
| stdio 会话 | 「进程 = 会话」 | 没有会话 id；子进程的生命周期就是会话 |

## 延伸阅读

- [Model Context Protocol — Client spec](https://modelcontextprotocol.io/specification/2025-11-25/client) —— 客户端行为的权威规范
- [MCP — Quickstart client guide](https://modelcontextprotocol.io/quickstart/client) —— 基于 Python SDK 的 hello-world 客户端教程
- [MCP Python SDK — client module](https://github.com/modelcontextprotocol/python-sdk) —— `ClientSession` 与 `stdio_client` 的参考实现
- [MCP TypeScript SDK — Client](https://github.com/modelcontextprotocol/typescript-sdk) —— TypeScript 对应实现
- [VS Code — MCP in extensions](https://code.visualstudio.com/api/extension-guides/ai/mcp) —— VS Code 如何在单个编辑器宿主中复用多个 MCP 服务器
