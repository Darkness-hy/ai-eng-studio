# Claude Agent SDK：子智能体与会话存储

> Claude Agent SDK 是 Claude Code 执行框架（harness）的库化形态。内置工具、用于上下文隔离的子智能体、钩子、W3C 链路追踪传播、与 TypeScript 对等的会话存储。Claude Managed Agents 则是面向长时异步任务的托管替代方案。

**Type:** Learn + Build
**Languages:** Python (stdlib)
**Prerequisites:** Phase 14 · 01 (Agent Loop), Phase 14 · 10 (Skill Libraries)
**Time:** ~75 minutes

## 学习目标

- 说明 Anthropic Client SDK（原始 API）与 Claude Agent SDK（执行框架形态）的区别。
- 描述子智能体（subagent）——并行化与上下文隔离——以及何时该使用它们。
- 列出 Python SDK 的会话存储接口（`append`、`load`、`list_sessions`、`delete`、`list_subkeys`）以及 `--session-mirror` 的作用。
- 用标准库实现一个具备内置工具、上下文隔离的子智能体派生、生命周期钩子和会话存储的执行框架。

## 问题背景

原始的 LLM API 只能完成一次往返调用。而一个生产级智能体需要工具执行、MCP 服务器、生命周期钩子、子智能体派生、会话持久化、链路追踪传播。Claude Agent SDK 把这套形态以库的形式提供出来——与 Claude Code 使用的是同一套执行框架，开放给你构建自定义智能体。

## 核心概念

### Client SDK 与 Agent SDK

- **Client SDK（`anthropic`）。** 原始的 Messages API。循环、工具、状态都由你自己负责。
- **Agent SDK（`claude-agent-sdk`）。** 内置工具执行、MCP 连接、钩子、子智能体派生、会话存储。相当于把 Claude Code 的循环做成了一个库。

### 内置工具

SDK 开箱即带 10 余种工具：文件读写、shell、grep、glob、网页抓取等等。自定义工具通过标准的工具 schema 接口注册。

### 子智能体

Anthropic 文档中给出了两个用途：

1. **并行化。** 并发执行相互独立的工作。「为这 20 个模块各找出对应的测试文件」就是 20 个并行的子智能体任务。
2. **上下文隔离。** 子智能体使用自己的上下文窗口；只有结果会返回给编排者（orchestrator）。编排者的上下文预算得以保留。

Python SDK 近期新增：`list_subagents()`、`get_subagent_messages()`，用于读取子智能体的对话记录。

### 会话存储

与 TypeScript 版本协议对等：

- `append(session_id, message)` —— 追加一轮对话。
- `load(session_id)` —— 恢复会话。
- `list_sessions()` —— 枚举所有会话。
- `delete(session_id)` —— 删除，并级联删除子智能体会话。
- `list_subkeys(session_id)` —— 列出子智能体键。

`--session-mirror`（CLI 标志）会在流式输出的同时把对话记录镜像写入外部文件，便于调试。

### 钩子

可以注册的生命周期钩子：

- `PreToolUse`、`PostToolUse` —— 拦截或审计工具调用。
- `SessionStart`、`SessionEnd` —— 初始化与清理。
- `UserPromptSubmit` —— 在模型看到用户输入之前先做处理。
- `PreCompact` —— 在上下文压缩之前执行。
- `Stop` —— 智能体退出时清理。
- `Notification` —— 旁路告警。

pro-workflow（Phase 14 课程引用）等系统正是通过钩子来添加横切行为的。

### W3C 链路追踪上下文

调用方上活跃的 OTel span 会通过 W3C trace context 头传播到 CLI 子进程中。整个多进程调用链在你的后端中呈现为同一条 trace。

### Claude Managed Agents

托管替代方案（beta 头 `managed-agents-2026-04-01`）。面向长时异步任务，内置提示词缓存、内置上下文压缩。以放弃部分控制权换取托管基础设施。

### 这一模式的常见误区

- **子智能体滥发。** 为 100 个微小任务派生 100 个子智能体，开销反而占了大头。应该改为批处理。
- **钩子膨胀。** 每个团队都加钩子，启动时间不断膨胀。应每季度审查一次钩子。
- **会话堆积。** 会话不断累积、体积持续增长。应使用 `list_sessions` 配合过期策略。

## 从零实现

`code/main.py` 用标准库实现了 SDK 的形态：

- `Tool`、`ToolRegistry`，内置 `read_file`、`write_file`、`list_dir`。
- `Subagent` —— 私有上下文、隔离运行、只返回结果。
- `SessionStore` —— append、load、list、delete、list_subkeys。
- `Hooks` —— `pre_tool_use`、`post_tool_use`、`session_start`、`session_end`。
- 一个演示：主智能体并行派生 3 个子智能体（各自隔离），聚合结果，持久化会话。

运行：

```
python3 code/main.py
```

运行轨迹展示了子智能体的上下文隔离（编排者的上下文规模保持有界）、钩子执行以及会话持久化。

## 生产实践

- **Claude Agent SDK**：适合以 Claude 为核心、想要 Claude Code 执行框架形态的产品。
- **Claude Managed Agents**：适合托管的长时异步任务。
- **OpenAI Agents SDK**（第 16 课）：OpenAI 阵营的对应方案。
- **LangGraph + 自定义工具**：如果你想要的是图状的状态机。

## 交付产物

`outputs/skill-claude-agent-scaffold.md` 提供了一个 Claude Agent SDK 应用脚手架，包含子智能体、钩子、会话存储、MCP 服务器挂载和 W3C 链路追踪传播。

## 练习

1. 添加一个子智能体派生器，把 20 个任务分批为每组 5 个并行子智能体。对比编排者上下文规模与「一任务一子智能体」方案的差异。
2. 实现一个 `PreToolUse` 钩子，对 `write_file` 调用限流（每个会话每分钟 5 次）。追踪其行为。
3. 接入 `list_subkeys` 渲染一棵子智能体树。深层嵌套会是什么样子？
4. 把这个玩具实现移植到真实的 `claude-agent-sdk` Python 包上。工具注册方式发生了什么变化？
5. 阅读 Claude Managed Agents 文档。什么情况下你会从自托管切换到托管方案？

## 关键术语

| 术语 | 人们常说的 | 实际含义 |
|------|----------------|------------------------|
| Agent SDK | 「库版 Claude Code」 | 执行框架形态：工具、MCP、钩子、子智能体、会话存储 |
| 子智能体 | 「子代理」 | 独立上下文、独立预算；结果向上汇报 |
| 会话存储 | 「对话数据库」 | 持久化、加载、枚举、删除对话轮次，并级联处理子智能体会话 |
| 钩子 | 「生命周期回调」 | 工具调用前后、会话、提示词提交、压缩、停止 |
| W3C trace context | 「跨进程追踪」 | 父 span 传播到 CLI 子进程 |
| Managed Agents | 「托管执行框架」 | Anthropic 托管的长时异步任务 |
| `--session-mirror` | 「对话记录镜像」 | 在流式输出的同时把会话轮次写入外部文件 |
| MCP 服务器 | 「工具面」 | 挂载到智能体上的外部工具/资源来源 |

## 延伸阅读

- [Claude Agent SDK overview](https://platform.claude.com/docs/en/agent-sdk/overview) —— Claude Code 的库化形态
- [Anthropic, Building agents with the Claude Agent SDK](https://www.anthropic.com/engineering/building-agents-with-the-claude-agent-sdk) —— 生产模式
- [Claude Managed Agents overview](https://platform.claude.com/docs/en/managed-agents/overview) —— 托管替代方案
- [OpenAI Agents SDK](https://openai.github.io/openai-agents-python/) —— 对应方案
