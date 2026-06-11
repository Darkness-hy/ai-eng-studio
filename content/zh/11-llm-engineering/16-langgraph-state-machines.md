# LangGraph —— 智能体的状态机

> 手写的 ReAct 循环就是一个 `while True`。用 LangGraph 写的 ReAct 循环则是一张图，你可以对它做检查点、中断、分支，甚至时间旅行。智能体本身没变，变的是包裹它的运行框架（harness）。

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 11 · 09 (Function Calling), Phase 11 · 14 (Model Context Protocol)
**Time:** ~75 minutes

## 问题背景

你上线了一个函数调用智能体。它顺利跑了三轮，然后出问题了：模型调用的工具返回 500，用户中途改了主意，或者智能体决定在没有人工签批的情况下给订单退款。`while True:` 循环没有任何钩子。你既不能暂停它，也不能回退它，更不能分支出去看"如果模型当时选了另一个工具会怎样"。只要这个东西离开演示环境真正上线，智能体就变成了一个黑盒——要么成功，要么失败，别无所知。

一旦看清这一点，下一步就显而易见了。智能体本来就是一个状态机——系统提示加消息历史加待执行的工具调用加下一步动作。把这个状态机显式化：用节点表示"模型思考"、"工具运行"、"人工审批"，用边表示它们之间的条件转移。一旦图被显式化，运行框架就免费获得四样东西：检查点（checkpointing，在步骤之间保存状态）、中断（interrupts，为人工介入而暂停）、流式输出（streaming，流式输出 token 和中间事件）、时间旅行（time-travel，回退到先前状态并尝试不同分支）。

LangGraph 就是把这套抽象封装好交付的库。它不是 LangChain 意义上的智能体框架（"给你一个 AgentExecutor，祝你好运"）。它是一个图运行时，原生支持状态、原生支持持久化、原生支持中断。智能体循环是你画出来的，不是你手写出来的。

## 核心概念

![LangGraph StateGraph: nodes, edges, and the checkpointer](../assets/langgraph-stategraph.svg)

一个 `StateGraph` 由三部分组成。

1. **状态（State）。** 一个带类型的字典（TypedDict 或 Pydantic 模型），在图中流动。每个节点接收完整状态并返回部分更新，LangGraph 按字段用*归约器*（reducer）进行合并——需要累加的列表用 `operator.add`，默认则是覆盖。
2. **节点（Nodes）。** 形如 `state -> partial_state` 的 Python 函数。每个节点是一个独立步骤："调用模型"、"运行工具"、"做摘要"。
3. **边（Edges）。** 节点之间的转移。静态边只指向一个目标。条件边接受一个路由函数 `state -> next_node_name`，让图可以根据模型输出进行分支。

然后你编译这张图。编译会固定拓扑结构、挂载检查点存储器（checkpointer，可选但对生产环境至关重要），并返回一个可运行对象。你用初始状态和一个 `thread_id` 来调用它。执行的每一步都会以 `(thread_id, checkpoint_id)` 为键持久化一个检查点。

### 四项超能力

**检查点。** 每次节点转移都会把新状态写入存储（测试用内存存储，生产用 Postgres/Redis/SQLite）。用同一个 `thread_id` 再次调用图即可恢复，图会从暂停的位置继续执行。

**中断。** 用 `interrupt_before=["human_review"]` 标记一个节点，执行就会在该节点运行前停下。状态被持久化。你的 API 向用户返回"等待审批"。之后对同一 `thread_id` 发起带 `Command(resume=...)` 的请求即可恢复执行。

**流式输出。** `graph.stream(state, mode="updates")` 在状态增量产生时逐条产出。`mode="messages"` 流式输出模型节点内部的 LLM token。`mode="values"` 产出完整快照。展示什么给 UI 由你自己选择。

**时间旅行。** `graph.get_state_history(thread_id)` 返回完整的检查点日志。把任意一个先前的 `checkpoint_id` 传给 `graph.invoke`，就能从那个点分叉出去。这对调试（"如果模型当时选了工具 B 会怎样？"）和回放生产轨迹的回归测试都极其好用。

### 归约器才是重点

每个状态字段都有一个归约器。大多数情况默认行为就够了——新值覆盖旧值。但消息列表需要 `operator.add`，让新消息追加而不是替换。并行边的更新也通过归约器合并。如果两个节点同时更新 `messages` 而你忘了写 `Annotated[list, add_messages]`，后到的更新会悄无声息地覆盖前者，你就丢掉了半轮对话。归约器是这个库里唯一微妙的地方；把它做对，其余一切自然组合起来。

### 四个节点的 ReAct 图

一个生产级 ReAct 智能体就是四个节点加两条边：

1. `agent` —— 用当前消息历史调用 LLM。返回助手消息（其中可能包含 tool_calls）。
2. `tools` —— 执行最后一条助手消息中的所有 tool_calls，把工具结果作为工具消息追加进去。
3. 一条从 `agent` 出发的条件边：如果最后一条消息含有 tool_calls 则路由到 `tools`，否则到 `END`。
4. 一条从 `tools` 回到 `agent` 的静态边。

就这么多。完整的 ReAct 循环（思考 → 行动 → 观察 → 思考 → ……），带检查点、中断和流式输出，大约 40 行代码搞定。

### StateGraph 与 Send（扇出）

`Send(node_name, state)` 让一个节点可以派发并行子图。例如：智能体决定同时查询三个检索器。每个 `Send` 都会启动目标节点的一次并行执行，它们的输出通过状态归约器合并。这就是 LangGraph 表达"编排器-工作者"（orchestrator-workers）模式的方式，不需要任何线程原语。

### 子图

一个已编译的图可以作为另一张图中的一个节点。外层图只看到一个节点；内层图拥有自己的状态和自己的检查点。团队正是这样构建"监督者-工作者"（supervisor-worker）智能体的：监督者图把用户意图路由到各个领域专属的工作者子图。

## 从零实现

### 第 1 步：状态与节点

```python
from typing import Annotated, TypedDict
from langchain_core.messages import AnyMessage, HumanMessage, AIMessage
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver

class State(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]

def agent_node(state: State) -> dict:
    response = llm.invoke(state["messages"])
    return {"messages": [response]}

def should_continue(state: State) -> str:
    last = state["messages"][-1]
    return "tools" if getattr(last, "tool_calls", None) else END

tool_node = ToolNode(tools=[search_web, read_file])

graph = StateGraph(State)
graph.add_node("agent", agent_node)
graph.add_node("tools", tool_node)
graph.set_entry_point("agent")
graph.add_conditional_edges("agent", should_continue, {"tools": "tools", END: END})
graph.add_edge("tools", "agent")

app = graph.compile(checkpointer=MemorySaver())
```

`add_messages` 就是让消息列表累加而非覆盖的那个归约器。忘记写它是最常见的 LangGraph bug。

### 第 2 步：在一个线程中运行

```python
config = {"configurable": {"thread_id": "user-42"}}
for event in app.stream(
    {"messages": [HumanMessage("find the Anthropic headquarters address")]},
    config,
    stream_mode="updates",
):
    print(event)
```

每个更新都是一个 `{node_name: state_delta}` 形式的字典。你的前端可以把它们流式推送到 UI，让用户看到"智能体正在思考……调用 search_web……拿到结果……正在回答"。

### 第 3 步：加入人工介入（human-in-the-loop）中断

标记一个节点，让执行在它运行前暂停。

```python
app = graph.compile(
    checkpointer=MemorySaver(),
    interrupt_before=["tools"],  # pause before every tool call
)

state = app.invoke({"messages": [HumanMessage("delete the production database")]}, config)
# state["__interrupt__"] is set. Inspect proposed tool calls.
# If approved:
from langgraph.types import Command
app.invoke(Command(resume=True), config)
# If denied: write a rejection message and resume
app.update_state(config, {"messages": [AIMessage("Blocked by human reviewer.")]})
```

状态、检查点和线程在中断期间全部持久保留。除了执行过程中，没有任何东西只存在于内存里。

### 第 4 步：用时间旅行调试

```python
history = list(app.get_state_history(config))
for snapshot in history:
    print(snapshot.values["messages"][-1].content[:80], snapshot.config)

# Fork from a prior checkpoint
target = history[3].config  # three steps back
for event in app.stream(None, target, stream_mode="values"):
    pass  # replay from that point forward
```

输入传 `None` 表示从给定检查点重放；传一个值则会先把它作为更新追加到该检查点的状态上，再继续执行。这样你就能复现一次糟糕的智能体运行，而无需把整段对话重新跑一遍。

### 第 5 步：换上生产级检查点存储器

```python
from langgraph.checkpoint.postgres import PostgresSaver

with PostgresSaver.from_conn_string("postgresql://...") as checkpointer:
    checkpointer.setup()
    app = graph.compile(checkpointer=checkpointer)
```

SQLite、Redis 和 Postgres 都是官方自带的。`MemorySaver` 只用于测试。任何需要跨重启持久化的场景都应该用真正的存储。

## 核心技能

> 你用图来构建智能体，而不是用 `while True` 循环。

在动手用 LangGraph 之前，先做一次 60 秒的设计：

1. **列出节点。** 每个独立的决策或带副作用的动作都是一个节点。"智能体思考"、"工具运行"、"审核人批准"、"流式返回响应"。如果你列不出来，说明这个任务还不具备智能体的形态。
2. **声明状态。** 用最小化的 TypedDict，每个列表字段都配归约器。不要把所有东西都塞进 `messages`；把任务专属的字段（进行中的 `plan`、`budget` 计数器、`retrieved_docs` 列表）提升到顶层。
3. **画出边。** 除非下一步取决于模型输出，否则用静态边。每条条件边都需要一个带命名分支的路由函数。
4. **预先选定检查点存储器。** 测试用 `MemorySaver`，其余一律用 Postgres/Redis/SQLite。没有检查点存储器就不要上线——没有它就没有恢复、没有中断、没有时间旅行。
5. **中断要放在工具运行之前，而不是之后。** 审批应放在进入带副作用节点的边上，这样可以在造成损害之前取消；校验应放在模型输出之后的边上，这样可以低成本地拒绝错误调用。
6. **默认开启流式输出。** UI 用 `mode="updates"`，模型节点内部的 token 级流式输出用 `mode="messages"`，评估时用 `mode="values"` 看完整快照。

拒绝上线没有检查点存储器的 LangGraph 智能体。拒绝上线在副作用*之后*才中断的智能体。拒绝上线 `messages` 字段没有配 `add_messages` 归约器的智能体。

## 练习

1. **简单。** 用一个计算器工具和一个网页搜索工具实现上面的四节点 ReAct 图。验证两轮对话后 `list(app.get_state_history(config))` 至少返回四个检查点。
2. **中等。** 添加一个在 `agent` 之前运行的 `planner` 节点，向状态中写入结构化的 `plan: list[str]`。让 `agent` 把计划步骤标记为已完成。如果 `plan` 在检查点恢复后丢失（归约器写错了），测试应当失败。
3. **困难。** 构建一个监督者图，用 `Send` 在三个子图（`researcher`、`writer`、`reviewer`）之间路由。每个子图有自己的状态和检查点存储器。在外层图上加 `interrupt_before=["writer"]`，让人工可以审批研究简报。确认从先前检查点做时间旅行时，只会重新运行分叉出去的那条分支。

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|-----------------|-----------------------|
| StateGraph | "LangGraph 的那张图" | 编译前用来添加节点和边的构建器对象。 |
| 归约器（Reducer） | "字段怎么合并" | 当节点返回某字段的更新时应用的函数 `(old, new) -> merged`；默认是覆盖，`add_messages` 是追加。 |
| 线程（Thread） | "一个会话 ID" | 一个 `thread_id` 字符串，限定一次会话的所有检查点的作用域。 |
| 检查点（Checkpoint） | "暂停的状态" | 节点转移后完整图状态的持久化快照，以 `(thread_id, checkpoint_id)` 为键。 |
| 中断（Interrupt） | "暂停等人" | `interrupt_before` / `interrupt_after` 在节点边界处停止执行；用 `Command(resume=...)` 恢复。 |
| 时间旅行（Time-travel） | "从之前的某一步分叉" | `graph.invoke(None, config_with_old_checkpoint_id)` 从该检查点开始向前重放。 |
| Send | "并行子图派发" | 节点可返回的一种构造器，用来启动目标节点的 N 次并行执行。 |
| 子图（Subgraph） | "把编译好的图当节点" | 一个已编译的 StateGraph 作为另一张图中的节点；保留自己独立的状态作用域。 |

## 延伸阅读

- [LangGraph documentation](https://langchain-ai.github.io/langgraph/) —— StateGraph、归约器、检查点存储器和中断的权威参考。
- [LangGraph concepts: state, reducers, checkpointers](https://langchain-ai.github.io/langgraph/concepts/low_level/) —— 本课所用的心智模型，来自官方一手资料。
- [LangGraph Persistence and Checkpoints](https://langchain-ai.github.io/langgraph/concepts/persistence/) —— 关于 Postgres/SQLite/Redis 存储、检查点命名空间和线程 ID 的细节。
- [LangGraph Human-in-the-loop](https://langchain-ai.github.io/langgraph/concepts/human_in_the_loop/) —— `interrupt_before`、`interrupt_after`、`Command(resume=...)` 以及编辑状态的模式。
- [Yao et al., "ReAct: Synergizing Reasoning and Acting in Language Models" (ICLR 2023)](https://arxiv.org/abs/2210.03629) —— 每个 LangGraph 智能体都在实现的模式；读它能理解推理轨迹背后的设计动机。
- [Anthropic — Building effective agents (Dec 2024)](https://www.anthropic.com/research/building-effective-agents) —— 该优先选择哪些图形态（链式、路由器、编排器-工作者、评估器-优化器），以及何时选择。
- Phase 11 · 09 (Function Calling) —— 每个 LangGraph 智能体节点都在复用的工具调用原语。
- Phase 11 · 14 (Model Context Protocol) —— 通过 MCP 适配器接入 LangGraph `ToolNode` 的外部工具发现机制。
- Phase 11 · 17 (Agent framework tradeoffs) —— 什么时候该选 LangGraph 而不是 CrewAI、AutoGen 或 Agno。
