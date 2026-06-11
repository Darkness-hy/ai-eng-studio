# Agent 框架取舍 — LangGraph vs CrewAI vs AutoGen vs Agno

> 每个框架都在卖同一个演示（研究 Agent 生成一份报告），也都藏着同一个坑（状态 schema 和编排层打架）。选那个核心抽象与你问题形状相匹配的框架；其余一切都是你要写两遍的胶水代码。

**Type:** Learn
**Languages:** Python
**Prerequisites:** Phase 11 · 09 (Function Calling), Phase 11 · 16 (LangGraph)
**Time:** ~45 minutes

## 问题背景

你有一个需要多次 LLM 调用的任务。可能是一个研究工作流（规划、搜索、总结、引用），可能是一条代码审查流水线（解析 diff、批评、打补丁、验证），也可能是一个多轮助手，要订机票、写邮件、报销费用。于是你选了一个框架。

三天之后，你发现框架的抽象会泄漏。CrewAI 给了你角色，但当"研究员"需要把一份结构化计划交给"写作者"时，它就开始跟你较劲。AutoGen 给了你 Agent 之间的对话，但没有一等公民的状态，于是你的检查点变成了一份对话日志的 pickle。LangGraph 给了你状态图，却强迫你在还不知道 Agent 会做什么之前就给每条转移命名。Agno 给了你单 Agent 抽象，可一旦你想扇出到三个并发 worker，它就开始抗议。

解法不是"选出最好的框架"，而是让框架的核心抽象匹配你问题的形状。这节课就是要画出这张地图。

## 核心概念

![Agent framework matrix: core abstraction vs problem shape](../assets/framework-matrix.svg)

四个框架主导着 2026 年的格局。它们的核心抽象并不相同。

| 框架 | 核心抽象 | 最适合 | 最不适合 |
|-----------|------------------|----------|-----------|
| **LangGraph** | `StateGraph` —— 类型化状态、节点、条件边、检查点器（checkpointer）。 | 需要显式状态和人在回路（human-in-the-loop）中断的工作流；需要时间旅行调试的生产级 Agent。 | 拓扑未知、松散的角色驱动头脑风暴。 |
| **CrewAI** | `Crew` —— 角色（目标、背景设定）、任务、流程（顺序或层级）。 | 角色扮演或人设驱动、计划短小且线性/层级化的工作流。 | 任何超出 crew 轮次历史的状态需求；复杂分支。 |
| **AutoGen** | `ConversableAgent` 对 —— 两个或更多 Agent 轮流发言，直到满足退出条件。 | 多 Agent *对话*（师生、提议者-批评者、执行者-审阅者），思考从聊天中涌现。 | DAG 已知的确定性工作流；任何需要跨重启持久化状态的场景。 |
| **Agno** | `Agent` —— 单个 LLM + 工具 + 记忆，可组合成团队。 | 快速搭建的单 Agent 和轻量团队；多模态能力强，自带存储驱动。 | 深层、显式分支、带自定义 reducer 的图。 |

### "抽象"到底是什么意思

框架的核心抽象，就是你在白板上推销架构方案时画出来的那个东西。

- **LangGraph** → 你画一张图。节点是步骤，边是转移，每个时刻的状态对象都有类型。心智模型是状态机。
- **CrewAI** → 你画一张组织架构图。每个角色有职位描述，由一个经理来分派任务。心智模型是一支小型专家团队。
- **AutoGen** → 你画一段 Slack 私聊。两个 Agent 互发消息；需要主持人时第三个加入。心智模型是聊天。
- **Agno** → 你画一个挂着工具的盒子。把多个盒子并排放就是团队。心智模型是"开箱即用的 Agent"。

### 状态问题

状态是大多数框架选择在生产环境中崩盘的地方。

- **LangGraph。** 类型化状态（`TypedDict` 或 Pydantic 模型）、按字段的 reducer、一等公民的检查点器（SQLite/Postgres/Redis）。恢复、中断、时间旅行都是白送的。*（见 Phase 11 · 16。）*
- **CrewAI。** 状态以字符串形式通过 `context` 字段在任务间流动，或通过 `output_pydantic` 结构化传递。开箱没有按 crew 的持久存储；如果 crew 必须在重启后存活，你得自己拼装。
- **AutoGen。** 状态就是聊天历史加上用户自定义的 `context`。对话记录可以持久化；任意工作流状态则不行，除非你自己写适配器。
- **Agno。** 内置存储驱动（SQLite、Postgres、Mongo、Redis、DynamoDB），通过 `storage=` 挂到 `Agent` 上——会话和用户记忆自动持久化。但它不是完整的图检查点器，只是一个会话存储。

### 分支问题

每个非平凡的 Agent 都会分支。由谁来决定分支，这一点很关键。

- **LangGraph** —— 你来决定，通过条件边。路由是一个带命名分支的 Python 函数。分支在编译后的图中是一等公民；检查点器会记录走了哪条分支。
- **CrewAI** —— 层级模式下由经理决定；顺序模式下你在构建时决定。路由隐含在任务列表里；除了经理的提示词之外，没有一等公民的 "if"。
- **AutoGen** —— Agent 们通过聊天决定。分支从"下一个谁发言"中涌现。`GroupChatManager` 选择下一个发言者；你可以手写一个 `speaker_selection_method`，但默认是 LLM 驱动的。
- **Agno** —— Agent 通过下一次调用哪个工具来决定。团队有协调者/路由器/协作者三种模式；超出这些的分支是开发者自己的责任。

### 可观测性问题

- **LangGraph** —— 通过 LangSmith 或任意 OTel exporter 接入 OpenTelemetry。每次节点转移都是一个 trace span；检查点同时充当可回放的 trace。LangSmith 是第一方选项；Langfuse/Phoenix 也有适配器。
- **CrewAI** —— 自 2025 年底起一等公民支持 OpenTelemetry；集成了 Langfuse、Phoenix、Opik、AgentOps。
- **AutoGen** —— 通过 `autogen-core` 集成 OpenTelemetry；AgentOps 和 Opik 有连接器。追踪粒度是按 Agent 消息，而不是按节点。
- **Agno** —— 内置 `monitoring=True` 开关，外加 OpenTelemetry exporter；与 Langfuse 的会话追踪深度集成。

### 成本与延迟

四个框架都会引入每次调用的开销（框架逻辑、校验、序列化）。开销从低到高的大致排序：Agno ≈ LangGraph < CrewAI ≈ AutoGen。差距主要取决于框架额外做了多少 LLM 路由。CrewAI 的层级经理要花 token 决定下一个轮到谁；AutoGen 的 `GroupChatManager` 同理。LangGraph 只在你写 `llm.invoke` 的地方花 token。Agno 的单 Agent 路径很薄。

当单次运行的成本很重要时，优先选显式路由（LangGraph 的边、AutoGen 的 `speaker_selection_method`），而不是 LLM 选择的路由。

### 互操作性

- **LangGraph** ↔ **LangChain** 的工具、检索器、LLM。一等公民的 MCP 适配器（工具以 MCP server 的形式导入）。
- **CrewAI** ↔ 工具继承自 `BaseTool`；LangChain 工具、LlamaIndex 工具和 MCP 工具都能适配进来。crew 之间的委派通过 `allow_delegation=True`。
- **AutoGen** → `FunctionTool` 可以包装任意 Python 可调用对象；有 MCP 适配器。Agent 间协作模式与 AG2 生态深度耦合。
- **Agno** → `@tool` 装饰器或 BaseTool 子类；有 MCP 适配器；工具可以在 Agent 和团队之间共享。

## 核心技能

> 你能用一句话解释，为什么某个框架适合某个特定的 Agent 问题。

动手之前的检查清单：

1. **画出形状。** 这是一张图（类型化状态、命名转移）？一场角色扮演（专家之间交接工作）？一段聊天（Agent 聊到完成为止）？还是一个带工具的单 Agent？
2. **决定由谁分支。** 开发者决定分支 → LangGraph。经理 Agent 决定 → CrewAI 层级模式。聊天涌现 → AutoGen。工具调用决定 → Agno。
3. **核算状态预算。** 你需要从检查点恢复吗？时间旅行？运行中途的人工中断？需要的话，LangGraph 是默认选项；Agno 的会话能覆盖对话级别的状态。
4. **核算成本预算。** LLM 选择路由每轮都要多花 token。如果这个 Agent 一天要跑几千次，优先用显式路由。
5. **算上框架本身的开销。** 每个框架都是一个额外的依赖。如果任务只是两次 LLM 调用加一个工具，那就写 30 行纯 Python；没有什么框架比不用框架更便宜。

在你能画出那张图、那张组织架构图、那段聊天或那个 Agent 盒子之前，拒绝伸手去拿框架。拒绝选一个会逼你为真正需要的东西去对抗它状态模型的框架。

## 决策矩阵

| 问题形状 | 首选框架 | 原因 |
|---------------|---------------------|-----|
| 带类型化状态、人工审批、长时间运行的工作流 DAG | LangGraph | 一等公民的状态、检查点器、中断、时间旅行。 |
| 角色分明的研究/写作流水线 | CrewAI（顺序模式）或 LangGraph 子图 | 按任务分角色在 CrewAI 里表达成本很低；分支变复杂时升级到 LangGraph。 |
| 提议者-批评者或师生对话 | AutoGen | 双 Agent 聊天是它的原生形状。 |
| 带工具、会话、记忆的单 Agent | Agno | 配置最薄，内置存储和记忆。 |
| 数千路并行扇出加 reducer | LangGraph + `Send` | 唯一拥有一等公民并行分发 API 的框架。 |
| 快速原型，不想绑定框架 | 纯 Python + 厂商 SDK | 不用框架就是最快的框架。 |

## 练习

1. **简单。** 拿同一个任务——"研究 Anthropic 的总部所在地，写一份 200 词的简报，并注明出处"——分别用 LangGraph（四个节点：plan、search、write、cite）和 CrewAI（三个角色：researcher、writer、editor）实现。报告每次运行的 token 成本和代码行数。
2. **中等。** 用 AutoGen（researcher ↔ writer 聊天，editor 通过 `GroupChat` 加入）和 Agno（一个带 `search_tools` 和 `write_tools` 的单 Agent，外加一个会话存储）实现同一个任务。从三个维度给四种实现排名：(a) 单次运行成本，(b) 崩溃后恢复的能力，(c) 在写作步骤前注入人工审批的能力。
3. **困难。** 写一个决策树脚本 `pick_framework.py`，接收一段简短的问题描述（JSON：`{has_typed_state, has_roles, has_dialogue, has_parallel_fanout, needs_resume}`），返回一个推荐结果和一句话理由。用你自己设计的六个用例验证它。

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|-----------------|-----------------------|
| 编排（Orchestration） | "Agent 怎么协调" | 决定下一个运行哪个节点/角色/Agent 的那一层。 |
| 持久状态（Durable state） | "重启后能恢复" | 能在进程死亡后存活的状态，挂在检查点或会话存储上。 |
| LLM 选择路由 | "让模型来决定" | 一个规划 LLM 每轮挑选下一步；灵活，但每次决策都要付 token。 |
| 显式路由 | "开发者决定" | 由一个 Python 函数或静态边挑选下一步；便宜且可审计。 |
| Crew | "一个 CrewAI 团队" | 角色 + 任务 + 流程（顺序或层级）绑定成一个可运行单元。 |
| GroupChat | "AutoGen 的多 Agent 聊天" | N 个 Agent 之间由发言者选择器管理的对话。 |
| Team（Agno） | "多 Agent 版 Agno" | 在一组 Agent 之上的路由/协调/协作模式。 |
| StateGraph | "LangGraph 的图" | 类型化状态、节点、条件边、检查点器的抽象。 |

## 延伸阅读

- [LangGraph documentation](https://langchain-ai.github.io/langgraph/) —— StateGraph、检查点器、中断、时间旅行。
- [CrewAI documentation](https://docs.crewai.com/) —— Crews、Flows、Agents、Tasks、Processes。
- [AutoGen documentation](https://microsoft.github.io/autogen/) —— ConversableAgent、GroupChat、团队、工具。
- [Agno documentation](https://docs.agno.com/) —— Agent、Team、Workflow、存储、记忆。
- [Anthropic — Building effective agents (Dec 2024)](https://www.anthropic.com/research/building-effective-agents) —— 框架无关的模式库（提示链、路由、并行化、orchestrator-workers、evaluator-optimizer）。
- [Yao et al., "ReAct: Synergizing Reasoning and Acting" (ICLR 2023)](https://arxiv.org/abs/2210.03629) —— 每个框架都在包装的那个循环。
- [Wu et al., "AutoGen: Enabling Next-Gen LLM Applications via Multi-Agent Conversation" (2023)](https://arxiv.org/abs/2308.08155) —— AutoGen 的设计论文。
- [Park et al., "Generative Agents: Interactive Simulacra of Human Behavior" (UIST 2023)](https://arxiv.org/abs/2304.03442) —— CrewAI 式人设栈所依托的角色扮演奠基工作。
- Phase 11 · 16 (LangGraph) —— 本课用来对标的框架。
- Phase 11 · 19 (Reflexion) —— 一个能干净地映射到 LangGraph、但在 CrewAI 里很别扭的模式。
- Phase 11 · 22（生产可观测性）—— 无论选了哪个框架，如何给它装上观测仪表。
