# AutoGen v0.4：Actor 模型与智能体框架

> AutoGen v0.4（Microsoft Research，2025 年 1 月）围绕 Actor 模型重新设计了智能体编排：异步消息交换、事件驱动的智能体、故障隔离、天然的并发能力。该框架目前已进入维护模式，Microsoft Agent Framework（2025 年 10 月公开预览）将作为其继任者。

**Type:** Learn + Build
**Languages:** Python (stdlib)
**Prerequisites:** Phase 14 · 01 (Agent Loop), Phase 14 · 12 (Workflow Patterns)
**Time:** ~75 minutes

## 学习目标

- 描述 Actor 模型：智能体即 Actor，消息是唯一的进程间通信（IPC）方式，每个 Actor 的故障相互隔离。
- 说出 AutoGen v0.4 的三个 API 层——Core、AgentChat、Extensions——以及各自的用途。
- 解释为什么把消息投递与消息处理解耦能带来故障隔离和天然的并发能力。
- 用 Python 标准库实现一个 Actor 运行时，并把一个双智能体代码评审流程迁移到它上面。

## 问题背景

大多数智能体框架是同步的：一个智能体生产、一个智能体消费，全都串在一条调用栈上。一处失败就会让整个调用栈崩溃。并发是事后补丁，分布式部署则需要重写。

AutoGen v0.4 的答案是 Actor 模型。每个智能体都是一个拥有私有收件箱的 Actor，消息是唯一的交互方式。运行时把投递与处理解耦，故障被限制在单个 Actor 内部，并发是原生能力，分布式部署只不过是换一种传输方式。

## 核心概念

### Actor

一个 Actor 拥有：

- 私有状态（外部绝不能直接触碰）。
- 一个收件箱（消息队列）。
- 一个处理器：`receive(message) -> effects`，其中 effects 可以是「回复」「发送给其他 Actor」「派生新 Actor」「更新状态」「停止自身」。

两个 Actor 不能共享内存，只能互相发送消息。

### AutoGen v0.4 的三个 API 层

1. **Core。** 底层 Actor 框架。`AgentRuntime`、`Agent`、`Message`、`Topic`。异步消息交换，事件驱动。
2. **AgentChat。** 面向任务的高层 API（替代 v0.2 的 ConversableAgent）。`AssistantAgent`、`UserProxyAgent`、`RoundRobinGroupChat`、`SelectorGroupChat`。
3. **Extensions。** 各类集成——OpenAI、Anthropic、Azure、工具、记忆。

### 为什么解耦很重要

在 v0.2 的模型中，调用 `agent_a.chat(agent_b)` 会同步阻塞 agent_a，直到 agent_b 返回。在 v0.4 中，`send(agent_b, msg)` 只是把消息放进 agent_b 的收件箱就立即返回，由运行时稍后投递。由此产生三个结果：

- **故障隔离。** Agent B 崩溃不会让 Agent A 崩溃——运行时会在 B 的处理器中捕获故障，并决定如何处置（记录日志、重试、转入死信队列）。
- **天然并发。** 大量消息可以同时在途；各 Actor 并发地处理自己的收件箱。
- **天生面向分布式。** 收件箱 + 传输层是同一套抽象，不论 Actor 在同一进程内还是在另一台主机上。

### 拓扑结构

- **RoundRobinGroupChat。** 智能体按固定轮换顺序依次发言。
- **SelectorGroupChat。** 由一个选择器智能体根据对话上下文决定下一个发言者。
- **Magentic-One。** 用于网页浏览、代码执行、文件处理的参考多智能体团队，构建在 AgentChat 之上。

### 可观测性

内置 OpenTelemetry 支持。每条消息都会产生一个 span；工具调用按照 2026 年 OTel GenAI 语义约定携带 `gen_ai.*` 属性（见第 23 课）。

### 现状：维护模式

2026 年初：AutoGen v0.7.x 对研究和原型开发而言是稳定的。Microsoft 已把主要开发力量转向 Microsoft Agent Framework（2025 年 10 月 1 日公开预览；1.0 GA 目标定在 2026 年第一季度末）。AutoGen 的模式可以平滑迁移过去——Actor 模型才是经久不衰的核心思想。

## 从零实现

`code/main.py` 用标准库实现了一个 Actor 运行时：

- `Message` —— 带类型的载荷，包含 `sender`、`recipient`、`topic`、`body`。
- `Actor` —— 抽象基类，定义 `receive(message, runtime)`。
- `Runtime` —— 带共享队列的事件循环，负责投递与故障隔离。
- 一个双 Actor 演示：`ReviewerAgent` 评审代码，`ChecklistAgent` 执行检查清单；两者交换消息直到达成共识。

运行：

```
python3 code/main.py
```

运行轨迹会展示消息投递、其中一个 Actor 的模拟故障不会让另一个崩溃，以及双方最终收敛到一个共同结论。

## 生产实践

- **AutoGen v0.4/v0.7**（维护模式）——对研究、原型开发、多智能体模式而言是稳定的。
- **Microsoft Agent Framework**（公开预览）——前进方向；同样的 Actor 模型思想，换上了更新的 API。
- **LangGraph swarm 拓扑**（第 13 课）——通过共享工具交接实现的类似模式。
- **自建 Actor 运行时**——当你需要特定的传输层（NATS、RabbitMQ、gRPC）时。

## 交付产物

`outputs/skill-actor-runtime.md` 会针对给定的多智能体任务，生成一个最小化的 Actor 运行时外加一个团队模板（RoundRobin 或 Selector）。

## 练习

1. 添加死信队列（dead-letter queue）：当处理器抛出异常时，把失败的消息暂存起来供人工检查。在你的玩具实现里 DLQ 被触发的频率有多高？
2. 实现 `SelectorGroupChat`：由一个选择器 Actor 根据对话状态决定谁来处理下一条消息。
3. 添加分布式传输：把进程内队列换成 JSON-over-HTTP 服务器，让各 Actor 可以运行在不同进程中。
4. 为每条消息接入一个 OTel span（或用一个空操作替身）。按第 23 课的要求输出 `gen_ai.agent.name`、`gen_ai.operation.name`。
5. 阅读 AutoGen v0.4 的架构发布文章。把你的玩具实现迁移到真实的 `autogen_core` API 上。你略过了哪些在生产环境中至关重要的东西？

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|----------------|------------------------|
| Actor | 「智能体」 | 私有状态 + 收件箱 + 处理器；无共享内存 |
| Message | 「事件」 | 带类型的载荷；Actor 之间唯一的交互方式 |
| Inbox | 「邮箱」 | 每个 Actor 专属的待处理消息队列 |
| Runtime | 「智能体宿主」 | 负责路由消息并隔离故障的事件循环 |
| Topic | 「频道」 | Actor 之间命名的发布-订阅路由 |
| Fault isolation | 「让它崩溃（Let it crash）」 | 一个 Actor 失败不会让其他 Actor 崩溃 |
| RoundRobinGroupChat | 「固定轮换团队」 | 智能体按顺序轮流发言 |
| SelectorGroupChat | 「上下文路由团队」 | 由选择器决定下一个发言者 |
| Magentic-One | 「参考团队」 | 处理网页 + 代码 + 文件的多智能体小队 |

## 延伸阅读

- [AutoGen v0.4, Microsoft Research](https://www.microsoft.com/en-us/research/articles/autogen-v0-4-reimagining-the-foundation-of-agentic-ai-for-scale-extensibility-and-robustness/) —— 重新设计的官方发布文章
- [LangGraph overview](https://docs.langchain.com/oss/python/langgraph/overview) —— 图结构的替代方案
- [OpenTelemetry GenAI semantic conventions](https://opentelemetry.io/docs/specs/semconv/gen-ai/) —— AutoGen 默认输出的 span 规范
