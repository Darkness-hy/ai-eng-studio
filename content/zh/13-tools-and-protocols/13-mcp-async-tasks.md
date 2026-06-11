# 异步任务（SEP-1686）—— 长耗时工作的"即时调用、稍后取结果"模式

> 真实的智能体工作往往要花几分钟到几小时：CI 运行、深度研究的综合分析、批量导出。同步工具调用会掉线、超时，或者阻塞 UI。SEP-1686 于 2025-11-25 合并，引入了任务（Tasks）原语：任何请求都可以被增强为一个任务，结果可以稍后获取，或通过状态通知以流式方式推送。漂移风险提示：在 2026 年上半年，Tasks 仍处于实验阶段；SDK 接口仍在围绕规范进行设计。

**Type:** Build
**Languages:** Python (stdlib, async task state machine)
**Prerequisites:** Phase 13 · 07 (MCP server), Phase 13 · 09 (transports)
**Time:** ~75 minutes

## 学习目标

- 判断何时应把工具从同步调用提升为任务增强模式（服务端工作超过 30 秒）。
- 走通任务生命周期：`working` → `input_required` → `completed` / `failed` / `cancelled`。
- 持久化任务状态，确保崩溃不会丢失进行中的工作。
- 正确地轮询 `tasks/status` 并获取 `tasks/result`。

## 问题背景

一个 `generate_report` 工具要运行一条耗时数分钟的抽取流水线。在同步模型下有以下几种选择：

1. 把连接保持打开三分钟。远程传输会断开连接；客户端超时；UI 卡死。
2. 立即返回一个占位符，要求客户端轮询自定义端点。这破坏了 MCP 的统一性。
3. 发出请求后不管结果（fire-and-forget），拿不到任何返回。

没有一个是好选择。SEP-1686 增加了第四种：任务增强（task augmentation）。任何请求（通常是 `tools/call`）都可以被标记为任务。服务器立即返回一个任务 id。客户端轮询 `tasks/status`，并在完成后获取 `tasks/result`。服务端状态能在重启后保留。

## 核心概念

### 任务增强

通过设置 `params._meta.task.required: true`（或 `optional: true`，由服务器决定），一个请求就变成了任务。服务器立即返回：

```json
{
  "jsonrpc": "2.0", "id": 1,
  "result": {
    "_meta": {
      "task": {
        "id": "tsk_9f7b...",
        "state": "working",
        "ttl": 900000
      }
    }
  }
}
```

`ttl` 是服务器对保留状态的承诺；超过 ttl 后任务结果会被丢弃。

### 按工具粒度的选择性启用

工具注解可以声明任务支持级别：

- `taskSupport: "forbidden"` —— 该工具始终同步运行。适合执行很快的工具。
- `taskSupport: "optional"` —— 客户端可以请求任务增强。
- `taskSupport: "required"` —— 客户端必须使用任务增强。

`generate_report` 工具应设为 `required`，`notes_search` 工具应设为 `forbidden`。

### 状态

```
working  -> input_required -> working  (loop via elicitation)
working  -> completed
working  -> failed
working  -> cancelled
```

状态机是只增不改的（append-only）：一旦进入 `completed`、`failed` 或 `cancelled`，任务即为终态。

### 方法

- `tasks/status {taskId}` —— 返回当前状态和进度提示。
- `tasks/result {taskId}` —— 阻塞等待，或在尚未完成时返回 404。
- `tasks/cancel {taskId}` —— 幂等操作；终态任务会忽略该请求。
- `tasks/list` —— 可选；枚举活跃任务及最近完成的任务。

### 流式状态变更

当服务器支持时，客户端可以订阅状态通知：

```
server -> notifications/tasks/updated {taskId, state, progress?}
```

采用流式订阅而非轮询的客户端能获得更好的用户体验。轮询作为最小化接口始终受支持。

### 持久化状态

规范要求声明支持任务的服务器必须持久化状态。在 ttl 之内，崩溃不应丢失已完成的结果。存储方案从 SQLite 到 Redis 再到文件系统都可以。第 13 课的练习框架使用文件系统。

### 取消语义

`tasks/cancel` 是幂等的。如果任务正在执行，服务器会尝试停止它（依赖执行器的协作式取消）。如果任务已处于终态，该请求就是空操作（no-op）。

### 崩溃恢复

当服务器进程重启时：

1. 加载所有已持久化的任务状态。
2. 把进程死亡时仍处于 `working` 状态的任务标记为 `failed`，错误码为 `CRASH_RECOVERY`。
3. 在各自的 ttl 内保留 `completed` / `failed` / `cancelled` 状态。

### 异步任务与采样的组合

任务本身也可以调用 `sampling/createMessage`。长耗时的研究类任务就是这样运作的：服务器的任务线程按需对客户端的模型进行采样，而客户端的 UI 把任务显示为 `working` 并定期更新进度。

### 为什么这仍是实验性的

SEP-1686 于 2025-11-25 发布，但更大的路线图指出了三个未解决的问题：持久化订阅原语、子任务（父子任务关系）以及结果 TTL 的标准化。预计该规范在整个 2026 年都会持续演进。生产代码应只在常见场景下把 Tasks 当作稳定特性，并对子任务相关的未来 SDK 变更做好防护。

## 生产实践

`code/main.py` 实现了一个持久化任务存储（基于文件系统）和一个在后台线程中运行的 `generate_report` 工具。客户端调用该工具后立即获得任务 id，在工作线程更新进度的同时轮询 `tasks/status`，完成后获取 `tasks/result`。取消功能可正常工作；崩溃恢复通过杀掉工作线程并重新加载状态来模拟。

值得关注的点：

- 任务状态以 JSON 形式持久化到 `/tmp/lesson-13-tasks/<id>.json`。
- 工作线程更新 `progress` 字段；轮询时能看到进度在推进。
- 客户端发起取消时会设置一个事件；工作线程检测到后提前退出。
- 在"崩溃"后重新加载状态时，会把进行中的任务标记为 `failed`，错误码为 `CRASH_RECOVERY`。

## 交付产物

本课产出 `outputs/skill-task-store-designer.md`。给定一个长耗时工具（研究、构建、导出），该技能负责设计任务存储（状态结构、ttl、持久性），选择合适的 taskSupport 标志，并勾勒出进度通知的方案。

## 练习

1. 运行 `code/main.py`。发起一个 `generate_report` 任务，轮询状态，然后获取结果。

2. 在运行中途调用 `tasks/cancel`。验证工作线程响应了取消，且状态变为 `cancelled`。

3. 模拟崩溃恢复：杀掉工作线程，重启加载器，观察 `CRASH_RECOVERY` 故障模式。

4. 把存储扩展到 SQLite。持久性收益不变；但查询能力会更丰富（例如列出会话 X 的所有任务）。

5. 阅读 MCP 的 2026 路线图博文。找出未来一年最可能影响 SDK API 设计的那一个与 Tasks 相关的未决问题。

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|----------------|------------------------|
| Task | "长耗时工具调用" | 通过 `_meta.task` 增强、以异步方式执行的请求 |
| SEP-1686 | "Tasks 规范" | 于 2025-11-25 引入 Tasks 的规范演进提案（Spec Evolution Proposal） |
| `_meta.task` | "任务信封" | 包含 id、state、ttl 的逐请求元数据 |
| taskSupport | "工具标志" | 每个工具的 `forbidden` / `optional` / `required` 设置 |
| `tasks/status` | "轮询方法" | 获取当前状态和可选的进度提示 |
| `tasks/result` | "取结果" | 返回已完成的载荷，未完成时返回 404 |
| `tasks/cancel` | "停掉它" | 幂等的取消请求 |
| ttl | "保留预算" | 服务器承诺保留任务状态的毫秒数 |
| `notifications/tasks/updated` | "状态推送" | 服务器主动发起的状态变更事件 |
| Durable store | "崩溃安全的状态" | 文件系统 / SQLite / Redis 持久化层 |

## 延伸阅读

- [MCP — GitHub SEP-1686 issue](https://github.com/modelcontextprotocol/modelcontextprotocol/issues/1686) —— 原始提案及完整讨论
- [WorkOS — MCP async tasks for AI agent workflows](https://workos.com/blog/mcp-async-tasks-ai-agent-workflows) —— 带设计依据的设计讲解
- [DeepWiki — MCP task system and async operations](https://deepwiki.com/modelcontextprotocol/modelcontextprotocol/2.7-task-system-and-async-operations) —— 运行机制与状态机
- [FastMCP — Tasks](https://gofastmcp.com/servers/tasks) —— SDK 层面的任务实现模式
- [MCP blog — 2026 roadmap](https://blog.modelcontextprotocol.io/posts/2026-mcp-roadmap/) —— 未决问题与 2026 年优先事项（包括子任务）
