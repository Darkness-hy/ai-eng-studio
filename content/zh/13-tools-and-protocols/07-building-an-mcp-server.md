# 构建一个 MCP 服务器 — Python + TypeScript SDK

> 大多数 MCP 教程只演示 stdio 的 hello-world。一个真正的服务器需要同时暴露工具（tools）、资源（resources）和提示词（prompts），处理能力协商（capability negotiation），输出结构化错误，并且在不同 SDK 之间保持一致的行为。本课从零到一构建一个笔记服务器：纯标准库实现的 stdio 传输、JSON-RPC 分发、三大服务器原语，以及一种纯函数风格的写法——当你准备升级时，可以直接套进 Python SDK 的 FastMCP 或 TypeScript SDK。

**Type:** Build
**Languages:** Python (stdlib, stdio MCP server)
**Prerequisites:** Phase 13 · 06 (MCP fundamentals)
**Time:** ~75 minutes

## 学习目标

- 实现 `initialize`、`tools/list`、`tools/call`、`resources/list`、`resources/read`、`prompts/list` 和 `prompts/get` 方法。
- 编写一个分发循环：从 stdin 读取 JSON-RPC 消息，向 stdout 写入响应。
- 按照 JSON-RPC 2.0 规范及 MCP 的扩展错误码输出结构化错误响应。
- 将标准库实现平滑迁移到 FastMCP（Python SDK）或 TypeScript SDK，而无需重写工具逻辑。

## 问题背景

在使用远程传输（Phase 13 · 09）或认证层（Phase 13 · 16）之前，你需要先有一个干净的本地服务器。本地意味着 stdio：服务器由客户端作为子进程启动，消息以换行分隔的形式在 stdin/stdout 上流动。

2025-11-25 版规范规定，stdio 消息编码为 JSON 对象，并以显式的 `\n` 分隔。这里没有 SSE；SSE 是旧的远程模式，将在 2026 年年中被移除（Atlassian 的 Rovo MCP 服务器于 2026 年 6 月 30 日弃用它；Keboola 则是 2026 年 4 月 1 日）。对于 stdio，每行一个 JSON 对象就是全部的线上格式（wire format）。

笔记服务器是一个很合适的题材，因为它能完整覆盖三大服务器原语。工具负责变更操作（`notes_create`），资源负责暴露数据（`notes://{id}`），提示词负责提供模板（`review_note`）。本课的结构可以推广到任何领域。

## 核心概念

### 分发循环

```
loop:
  line = stdin.readline()
  msg = json.loads(line)
  if has id:
    handle request -> write response
  else:
    handle notification -> no response
```

三条规则：

- 不要向 stdout 打印任何非 JSON-RPC 封包的内容。调试日志一律写到 stderr。
- 每个请求必须（MUST）有一个携带相同 `id` 的响应与之对应。
- 通知（notification）绝不能（MUST NOT）被响应。

### 实现 `initialize`

```python
def initialize(params):
    return {
        "protocolVersion": "2025-11-25",
        "capabilities": {
            "tools": {"listChanged": True},
            "resources": {"listChanged": True, "subscribe": False},
            "prompts": {"listChanged": False},
        },
        "serverInfo": {"name": "notes", "version": "1.0.0"},
    }
```

只声明你实际支持的能力。客户端会依据这份能力集合来决定开启哪些功能。

### 实现 `tools/list` 和 `tools/call`

`tools/list` 返回 `{tools: [...]}`，其中每一项包含 `name`、`description`、`inputSchema`。`tools/call` 接收 `{name, arguments}`，返回 `{content: [blocks], isError: bool}`。

内容块（content block）是带类型的。最常见的几种：

```json
{"type": "text", "text": "Found 2 notes"}
{"type": "resource", "resource": {"uri": "notes://14", "text": "..."}}
{"type": "image", "data": "<base64>", "mimeType": "image/png"}
```

工具错误有两种形态。协议级错误（未知方法、参数错误）是 JSON-RPC 错误；工具级错误（调用合法但工具执行失败）则以 `{content: [...], isError: true}` 的形式返回。这样模型就能在自己的上下文中看到这次失败。

### 实现资源

资源在设计上是只读的。`resources/list` 返回一份清单；`resources/read` 返回内容。URI 可以是 `file://...`、`http://...`，也可以是 `notes://` 这样的自定义 scheme。

把数据作为资源而非工具来暴露时：

- 模型不会去「调用」它；客户端可以在用户要求时把它注入上下文。
- 订阅机制让服务器在资源变化时主动推送更新（Phase 13 · 10）。
- Phase 13 · 14 用 `ui://` 把这一机制扩展到交互式资源。

### 实现提示词

提示词是带命名参数的模板。宿主（host）会把它们展示为斜杠命令。比如一个 `review_note` 提示词可以接收 `note_id` 参数，生成一个多消息的提示词模板，由客户端交给它的模型。

### stdio 传输的细节

- 换行分隔的 JSON。没有长度前缀帧（length-prefixed framing）。
- 不要缓冲。每次写入后调用 `sys.stdout.flush()`。
- 生命周期由客户端控制。stdin 关闭（EOF）时，干净地退出。
- 不要悄悄吞掉 SIGPIPE；记录日志后退出。

### 注解（Annotations）

每个工具都可以携带 `annotations`，描述其安全属性：

- `readOnlyHint: true` — 纯读取，重试安全。
- `destructiveHint: true` — 不可逆的副作用；客户端应进行确认。
- `idempotentHint: true` — 相同输入产生相同输出。
- `openWorldHint: true` — 与外部系统交互。

客户端依据这些注解来决定用户体验（确认对话框、状态指示器）和路由策略（Phase 13 · 17）。

### 升级路径

`code/main.py` 中的标准库实现约 180 行。FastMCP（Python）能把同样的逻辑压缩成装饰器风格：

```python
from fastmcp import FastMCP
app = FastMCP("notes")

@app.tool()
def notes_search(query: str, limit: int = 10) -> list[dict]:
    ...
```

TypeScript SDK 的形态与之等价。准备好之后即可无缝迁移；底层概念（能力声明、分发、内容块）完全相同。

## 生产实践

`code/main.py` 是一个完整的、纯标准库实现的 stdio 笔记 MCP 服务器。它处理 `initialize`、针对三个工具（`notes_list`、`notes_search`、`notes_create`）的 `tools/list` 和 `tools/call`、针对每条笔记的 `resources/list` 和 `resources/read`，以及一个 `review_note` 提示词。你可以通过管道发送 JSON-RPC 消息来驱动它：

```
echo '{"jsonrpc":"2.0","id":1,"method":"initialize","params":{}}' | python main.py
```

值得关注的点：

- 分发器是一个以方法名为键的 `dict[str, Callable]`。
- 每个工具执行器返回的是内容块列表，而不是裸字符串。
- 执行器抛出异常时会设置 `isError: true`。

## 交付产物

本课产出 `outputs/skill-mcp-server-scaffolder.md`。给定一个领域（笔记、工单、文件、数据库），该技能会脚手架出一个 MCP 服务器，包含合理的工具 / 资源 / 提示词划分以及 SDK 升级路径。

## 练习

1. 运行 `code/main.py`，用手写的 JSON-RPC 消息驱动它。先调用 `notes_create`，再用 `resources/read` 取回新建的笔记。

2. 添加一个带 `annotations: {destructiveHint: true}` 的 `notes_delete` 工具。验证客户端会弹出确认对话框（这需要一个真实的宿主；Claude Desktop 可以）。

3. 实现 `resources/subscribe`，让服务器在笔记被修改时推送 `notifications/resources/updated`。再加一个保活（keepalive）任务。

4. 把服务器移植到 FastMCP。Python 文件应缩减到 80 行以内。线上行为必须完全一致；用同一套 JSON-RPC 测试脚本验证。

5. 阅读规范的 `server/tools` 一节，找出工具定义中一个本课服务器没有实现的字段。（提示：有好几个；挑一个加上。）

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|----------------|------------------------|
| MCP server | 「暴露工具的那个东西」 | 通过 stdio 或 HTTP 讲 MCP JSON-RPC 的进程 |
| stdio transport | 「子进程模式」 | 服务器由客户端启动；通过 stdin/stdout 通信 |
| Dispatcher | 「方法路由器」 | JSON-RPC 方法名到处理函数的映射 |
| Content block | 「工具结果分块」 | 工具响应 `content` 数组中带类型的元素 |
| `isError` | 「工具级失败」 | 表示工具执行失败；与 JSON-RPC 错误区分开 |
| Annotations | 「安全提示」 | readOnly / destructive / idempotent / openWorld 标志 |
| FastMCP | 「Python SDK」 | 构建在 MCP 协议之上的装饰器风格高层框架 |
| Resource URI | 「可寻址数据」 | 标识资源的 `file://`、`db://` 或自定义 scheme |
| Prompt template | 「斜杠命令简报」 | 服务器提供的、带参数槽位的模板，供宿主 UI 使用 |
| Capability declaration | 「功能开关」 | 在 `initialize` 中按原语逐项声明的标志 |

## 延伸阅读

- [Model Context Protocol — Python SDK](https://github.com/modelcontextprotocol/python-sdk) — Python 参考实现
- [Model Context Protocol — TypeScript SDK](https://github.com/modelcontextprotocol/typescript-sdk) — 平行的 TS 实现
- [FastMCP — server framework](https://gofastmcp.com/) — 装饰器风格的 MCP 服务器 Python API
- [MCP — Quickstart server guide](https://modelcontextprotocol.io/quickstart/server) — 任选一种 SDK 的端到端教程
- [MCP — Server tools spec](https://modelcontextprotocol.io/specification/2025-11-25/server/tools) — tools/* 消息的完整参考
