# LangGraph：有状态图与持久化执行

> LangGraph 是 2026 年底层有状态编排的参考实现。Agent 是一台状态机；节点是函数；边是状态转移；状态不可变，且每一步之后都会写入检查点。无论在哪里失败，都能从失败处精确恢复。

**Type:** Learn + Build
**Languages:** Python (stdlib)
**Prerequisites:** Phase 14 · 01 (Agent Loop), Phase 14 · 12 (Workflow Patterns)
**Time:** ~75 minutes

## 学习目标

- 描述 LangGraph 的核心模型：由不可变状态、函数节点、条件边和步后检查点组成的状态机。
- 说出官方文档强调的四项能力：持久化执行、流式输出、人在回路、完备的记忆机制。
- 解释 LangGraph 支持的三种编排拓扑：监督者（supervisor）、点对点（swarm）、层级式（嵌套子图）。
- 用标准库实现一个状态图，包含不可变状态、条件边以及一个检查点/恢复周期。

## 问题背景

Agent 和工作流面临同一个问题：一个 40 步的运行在第 38 步失败时，你希望从第 38 步恢复，而不是从头再来。把状态当二等公民的设计，会迫使运维人员在一个默认每次都全新运行的库外面拼凑重试逻辑。

LangGraph 的设计答案是：状态是一等的类型化对象，状态变更必须显式声明，每个节点之后都会持久化检查点。恢复只需要一次 `load_state(session_id)` 调用。

## 核心概念

### 图

一张图由以下要素定义：

- **状态类型。** 一个类型化字典（或 Pydantic 模型），每个节点都读取并修改它。
- **节点。** 纯函数 `(state) -> state_update`。函数返回后，更新被合并进状态。
- **边。** 节点之间的条件转移或直接转移。
- **入口与出口。** `START` 和 `END` 两个哨兵节点标记图的边界。

例子：一个包含 `classify`、`refund`、`bug`、`sales`、`done` 节点的 agent——本质是把路由工作流表达成一张图。

### 持久化执行

每个节点返回后，运行时会把状态序列化并写入检查点存储（checkpointer，可以是 SQLite、Postgres、Redis 或自定义后端）。当第 N 步失败时，运行时可以 `resume(session_id)`，带着精确的状态从第 N+1 步继续。

LangGraph 文档明确列出了在意这一点的生产用户：Klarna、Uber、J.P. Morgan。卖点不在图这个形态本身，而在于图形态加上检查点机制让故障恢复变得廉价。

### 流式输出

每个节点都可以产出部分输出。图会把按节点增量的事件流式推送给调用方，UI 因此能在图运行时实时更新。

### 人在回路

在节点之间检查并修改状态。实现方式：在关键节点前暂停，把状态呈现给人类，接受修改，然后恢复执行。检查点机制让这件事变得容易，因为状态本来就已经序列化了。

### 记忆

短期记忆（单次运行内——存放在状态中的对话历史）和长期记忆（跨运行——通过检查点存储加一个独立的长期存储来持久化）。LangGraph 通过工具与外部记忆系统（Mem0、自定义）集成。

### 三种拓扑

1. **监督者（Supervisor）。** 由一个中心路由 LLM 把任务分派给各个专家子 agent。对应 `langgraph-supervisor` 中的 `create_supervisor()`（不过 LangChain 团队在 2026 年建议直接通过工具调用实现，以获得更强的上下文控制）。
2. **Swarm / 点对点。** 各 agent 通过共享的工具界面直接交接任务，没有中心路由器。
3. **层级式。** 监督者管理下级监督者，以嵌套子图的方式实现。

### 这个模式哪里会出错

- **检查点范围太小。** 只对对话轮次做检查点，会让工具状态和记忆写入无法恢复。完整状态必须可序列化。
- **非确定性节点。** 恢复机制假设同样的节点输入会产生同样的状态更新。随机种子、墙上时钟、外部 API 都必须被捕获记录。
- **条件边滥用。** 每条边都是条件边的图，是一台无法推理的状态机。优先使用线性链，偶尔分支。

## 从零实现

`code/main.py` 用标准库实现了一个有状态图：

- `State` —— 一个类型化字典，包含 `messages`、`step`、`route`、`output`、`human_approval`。
- `Node` —— 接受状态、返回更新字典的可调用对象。
- `StateGraph` —— 节点 + 边 + 条件边 + 运行 + 恢复。
- `SQLiteCheckpointer`（内存版模拟实现）—— 在每个节点之后序列化状态；`load(session_id)` 负责恢复。
- 一个演示图：classify -> branch(refund / bug / sales) -> human gate -> send。

运行：

```
python3 code/main.py
```

运行轨迹会展示第一次运行在人工审批门处失败、状态被持久化、随后恢复执行并产出最终结果的全过程。

## 生产实践

- **LangGraph** —— 参考实现，生产可用。使用 `create_react_agent`、`create_supervisor`，或自己构建图。
- **AutoGen v0.4**（第 14 课）—— 面向高并发场景的 actor 模型替代方案。
- **Claude Agent SDK**（第 17 课）—— 托管式运行框架，内置会话存储。
- **自研** —— 当你需要精确控制状态结构或检查点后端时。

## 交付产物

`outputs/skill-state-graph.md` 可以在任意目标运行时中生成一个 LangGraph 风格的状态图，并接好检查点与恢复机制。

## 练习

1. 添加一条从 `classify` 到 `end` 的条件边：当分类置信度低于阈值时走这条边。在人类手动设置 `route` 之后恢复运行。
2. 把类 SQLite 的模拟实现换成真正的 SQLite 检查点存储。测量每一步的序列化开销。
3. 实现并行边：两个节点并发运行，由自定义 reducer 合并结果。不可变状态在这里带来了什么好处？
4. 阅读 `langgraph-supervisor` 参考文档。把这个玩具示例移植到 `create_supervisor`。比较两种实现的运行轨迹形态。
5. 添加流式输出：每个节点在运行时产出部分状态。增量数据到达时即时打印。

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|----------------|------------------------|
| 状态图 | “Agent 即状态机” | 类型化状态 + 节点 + 边 + reducer |
| 检查点存储（checkpointer） | “持久化后端” | 在每个节点之后序列化状态；支撑恢复机制 |
| Reducer | “状态合并器” | 把当前状态与节点的更新合并起来的函数 |
| 条件边 | “分支” | 由一个以状态为输入的函数选择的边 |
| 子图 | “嵌套图” | 被当作另一张图中一个节点使用的图 |
| 持久化执行 | “从失败处恢复” | 带着精确状态，从最后一个成功的节点重新开始 |
| 监督者（Supervisor） | “路由 LLM” | 面向专家子 agent 的中心分发器 |
| Swarm | “P2P agents” | Agent 通过共享工具相互交接；没有中心路由器 |

## 延伸阅读

- [LangGraph overview](https://docs.langchain.com/oss/python/langgraph/overview) —— 官方参考文档
- [langgraph-supervisor reference](https://reference.langchain.com/python/langgraph/supervisor/) —— supervisor 模式 API
- [AutoGen v0.4, Microsoft Research](https://www.microsoft.com/en-us/research/articles/autogen-v0-4-reimagining-the-foundation-of-agentic-ai-for-scale-extensibility-and-robustness/) —— actor 模型替代方案
- [Claude Agent SDK overview](https://platform.claude.com/docs/en/agent-sdk/overview) —— 会话存储与子 agent
