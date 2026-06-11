# 交接与例程 —— 无状态编排

> OpenAI 的 Swarm（2024 年 10 月）把多智能体编排提炼为两个原语：**例程（routine）**（指令 + 工具构成的系统提示词）和**交接（handoff）**（一个返回另一个 Agent 的工具）。没有状态机，没有分支 DSL —— LLM 通过调用正确的交接工具来完成路由。OpenAI Agents SDK（2025 年 3 月）是其生产级继任者。Swarm 本身仍是最简洁的概念参考 —— 它的全部源码只有几百行。这个模式之所以广为流传，是因为其 API 表面大致就是"agent = 提示词 + 工具；handoff = 返回 agent 的函数"。局限：无状态，因此记忆是调用方的问题。

**Type:** Learn + Build
**Languages:** Python (stdlib)
**Prerequisites:** Phase 16 · 04 (Primitive Model)
**Time:** ~60 minutes

## 问题背景

每个多智能体框架都想让你学习它的 DSL：LangGraph 的节点和边，CrewAI 的 crew 和 task，AutoGen 的 GroupChat 和 manager。这些 DSL 是真实存在的抽象，但它们让这件事显得比实际需要的更沉重。

Swarm 反其道而行：直接利用模型已经具备的工具调用能力。交接变成工具调用。编排器就是当前掌握对话的那个智能体。状态机隐含在各智能体的系统提示词中。

## 核心概念

### 两个原语

**例程（Routine）。** 定义智能体角色和可用工具的系统提示词。可以把它看作一组限定范围的指令："你是分诊智能体；如果用户询问退款，就交接给退款智能体。"

**交接（Handoff）。** 智能体可以调用的一个工具，它返回一个新的 Agent 对象。Swarm 运行时检测到返回值是 Agent，就在下一轮切换活跃智能体。

整个抽象就这么多。

```
def transfer_to_refunds():
    return refund_agent  # Swarm sees Agent return → switch active agent

triage_agent = Agent(
    name="triage",
    instructions="Route the user to the right specialist.",
    functions=[transfer_to_refunds, transfer_to_sales, transfer_to_support],
)
```

分诊智能体的系统提示词让它根据用户消息选择正确的交接。LLM 的工具调用完成了路由。

### 为什么它广为流传

- **API 小。** 只需学习两个概念。
- **利用模型已有的能力。** 工具调用在各家提供商那里都已达到生产级水准。
- **没有状态机负担。** 你不需要描述图；智能体的提示词描述了它们交接给谁。

### 无状态的代价

Swarm 在运行之间是明确无状态的。框架在单次运行期间会保留消息历史，但不会持久化任何东西。记忆、连续性、长期运行的任务 —— 全是调用方的问题。

在生产环境中（OpenAI Agents SDK，2025 年 3 月），这正是主要改变之一：SDK 在保留交接原语的同时，增加了内置的会话管理、护栏（guardrails）和追踪（tracing）。

### Swarm/交接适用的场景

- **分诊模式。** 一线智能体把用户路由到专家。
- **基于技能的交接。** "如果任务需要写代码，调用编码者；如果需要做研究，调用研究者。"
- **简短、有界的对话。** 客户支持、FAQ 转工单、简单工作流。

### Swarm 力不从心的场景

- **需要共享记忆的长会话。** 交接会把对话状态重置为新智能体的提示词加历史记录。没有调用方管理的记忆，就没有跨智能体的持久状态。
- **并行执行。** 交接是一次一个的 —— 活跃智能体发生切换。并行需要调用方编排多个 Swarm 运行。
- **审计与回放。** 无状态运行很难精确回放；LLM 的交接选择不是确定性的。

### OpenAI Agents SDK（2025 年 3 月）

这个生产级继任者增加了：

- **会话状态。** 跨运行的持久线程。
- **护栏。** 输入/输出验证钩子。
- **追踪。** 每次工具调用和交接都被记录。
- **交接过滤器。** 控制交接时传递哪些上下文。

交接原语得以保留；围绕它增加了生产级的工程便利。

### Swarm 对比 GroupChat

两者都使用 LLM 驱动的路由，但在**由谁选择下一个**这一点上不同：

- GroupChat：由外部的选择器（函数或 LLM）挑选下一个发言者。
- Swarm：当前智能体通过调用交接工具来挑选自己的继任者。

Swarm 是"智能体决定下一步"；GroupChat 是"管理者决定下一步"。Swarm 的决策体现在活跃智能体的工具调用中；GroupChat 的决策体现在 `GroupChatManager` 中。

## 从零实现

`code/main.py` 从零实现了 Swarm：一个 Agent dataclass、一个交接机制（工具返回 Agent）、以及一个能检测智能体切换的运行循环。

演示：一个分诊智能体把请求路由到退款、销售或支持专家。每个专家有自己的工具。运行循环打印每次交接。

运行：

```
python3 code/main.py
```

## 生产实践

`outputs/skill-handoff-designer.md` 为给定任务设计交接拓扑：存在哪些智能体、它们可以调用哪些交接、传递什么上下文。

## 交付产物

检查清单：

- **交接日志。** 每次交接都写入一条追踪事件，包含来源智能体、目标智能体、上下文快照。
- **上下文传递规则。** 决定交接时传递什么：完整历史（昂贵）、最近 N 条消息、还是摘要。
- **交接护栏。** 交接给拥有不同工具权限的专家时必须进行鉴权 —— 否则提示词注入可以强制触发非预期的交接。
- **循环检测。** 两个智能体来回互相交接是常见故障；用简单的最近 K 次环检测即可发现。
- **兜底智能体。** 如果交接目标不存在，回退到一个安全的默认智能体。

## 练习

1. 运行 `code/main.py`，分诊到退款智能体。确认第二轮的活跃智能体是退款智能体。
2. 添加一条循环检测规则：如果同样两个智能体连续交接 3 次，强制退出。设计兜底方案。
3. 阅读 OpenAI Agents SDK 关于交接过滤器的文档。实现一个"交接时摘要"版本：移交方智能体在接收方智能体接手之前，把上下文压缩成要点摘要。
4. 比较 Swarm 的交接与 GroupChatManager 的选择器。哪种模式会让提示词注入问题更严重，为什么？
5. 阅读 Swarm cookbook（https://developers.openai.com/cookbook/examples/orchestrating_agents）。找出 Swarm 做出的一个明确设计决策，看 OpenAI Agents SDK 是改变了它还是保留了它。

## 关键术语

| 术语 | 大家怎么说 | 实际含义 |
|------|----------------|------------------------|
| 例程（Routine） | "智能体提示词" | 系统提示词 + 工具列表。定义角色和可用的交接。 |
| 交接（Handoff） | "转给另一个智能体" | 活跃智能体可以调用的一个工具，它返回一个新的 Agent。运行时随之切换活跃智能体。 |
| 无状态 | "运行之间没有记忆" | Swarm 不持久化任何东西；记忆是调用方的责任。 |
| 活跃智能体 | "现在谁在发言" | 当前掌握对话的智能体。交接会改变它。 |
| 上下文传递 | "交接时传什么" | 决定接收方智能体能看到哪些历史的策略：完整、最近 N 条、或摘要。 |
| 交接循环 | "智能体打乒乓" | 两个智能体不断互相交接的故障模式。 |
| OpenAI Agents SDK | "生产版 Swarm" | 2025 年 3 月的继任者；在交接原语之上增加了会话、护栏、追踪。 |
| 交接过滤器 | "传递时的关卡" | SDK 特性，用于在交接边界检查和修改上下文。 |

## 延伸阅读

- [OpenAI cookbook — Orchestrating Agents: Routines and Handoffs](https://developers.openai.com/cookbook/examples/orchestrating_agents) —— 参考性的阐述
- [OpenAI Swarm repo](https://github.com/openai/swarm) —— 原始实现，作为概念参考保留
- [OpenAI Agents SDK docs](https://openai.github.io/openai-agents-python/) —— 带会话和追踪的生产级继任者
- [Anthropic handoff-in-Claude notes](https://docs.anthropic.com/en/docs/claude-code) —— Claude Code 子智能体如何通过 `Task` 使用类交接模式
