# CrewAI：基于角色的 Crew 与 Flow

> CrewAI 是 2026 年基于角色的多智能体框架。四个基本原语：Agent、Task、Crew、Process。两种顶层形态：Crew（自主的、基于角色的协作）和 Flow（事件驱动、确定性执行）。官方文档说得很直白："任何生产级应用，都应该从 Flow 开始。"

**Type:** Learn + Build
**Languages:** Python (stdlib)
**Prerequisites:** Phase 14 · 12 (Workflow Patterns), Phase 14 · 14 (Actor Model)
**Time:** ~75 minutes

## 学习目标

- 说出 CrewAI 的四个基本原语（Agent、Task、Crew、Process）以及各自负责什么。
- 区分 Sequential、Hierarchical 和规划中的 Consensus 流程；针对不同工作负载做出选择。
- 区分 Crew（自主的、基于角色）和 Flow（事件驱动、确定性），并解释官方文档的生产环境建议。
- 用 `@tool` 装饰器和 `BaseTool` 子类接入工具；理解结构化输出与自由文本的权衡。
- 说出 CrewAI 的四种记忆类型以及各自的适用场景。
- 用标准库实现一个三智能体 crew（researcher、writer、editor），产出一份简报。
- 识别 CrewAI 的三种失败模式：提示词膨胀、manager LLM 开销、脆弱的任务交接。

## 问题背景

采用多智能体框架的团队都会撞上同一堵墙。"自主协作"在演示里听起来很美。然后客户提了一个 bug，你需要确定性重放。或者财务部门问一个 LLM 路由的 crew 每次运行花多少钱。或者值班同事凌晨三点需要知道哪个智能体卡住了。

自由形态、由 LLM 路由的 crew 对这些问题一个都答不清楚。纯 DAG 全都能回答，却丢掉了头脑风暴智能体所需的探索性形态。

CrewAI 的拆分对这个权衡很坦诚。Crew 用于协作式、基于角色、探索性的工作。Flow 用于事件驱动、代码掌控、可审计的生产环境。同一个框架，两种形态，按场景选择。

## 核心概念

### 四个基本原语

CrewAI 的 API 表面很小。记住这些，剩下的都是配置。

- **Agent。** `role + goal + backstory + tools + (optional) llm`。backstory（背景设定）是承重墙。它塑造语气、判断力，以及智能体何时停止。tools 是智能体可以调用的函数（下文详述）。
- **Task。** `description + expected_output + agent + (optional) context + (optional) output_pydantic`。可复用的工作单元。`expected_output` 是契约。`context` 列出上游任务，其输出会被传入。`output_pydantic` 强制要求结构化的输出形状。
- **Crew。** 容器。持有 `agents` 列表、`tasks` 列表、`process`，以及可选的 `memory`、`verbose`、`manager_llm` 设置。
- **Process。** 执行策略。Sequential、Hierarchical、Consensus（规划中）。决定整个运行的形态。

智能体之间互相不可见。Task 引用 Agent。Crew 对 Task 排序。Process 决定谁来挑选下一个任务。这就是完整的心智模型。

> **验证版本：** CrewAI 0.86（2026-05）。新版本可能重命名或合并流程类型；在依赖某个具体形态之前，先查阅 [CrewAI Processes docs](https://docs.crewai.com/concepts/processes)。

### Sequential vs Hierarchical vs Consensus

- **Sequential。** 任务按声明顺序运行。任务 N 的输出作为 `context` 提供给任务 N+1。成本最低。最可预测。顺序固定时使用。
- **Hierarchical。** 一个 manager Agent（单独的 LLM 调用）在专家之间路由。CrewAI 从你的 `manager_llm` 配置或默认配置生成这个 manager。manager 每一轮挑选下一个任务，可以拒绝或重新路由。当你有四个及以上专家、且执行顺序真正取决于前序输出时使用。
- **Consensus。** 规划中，目前未在公开 API 中实现。文档为未来基于投票的流程保留了这个名字。今天不要依赖它。

Hierarchical 在每次专家调用之上额外增加一轮 LLM 调用（manager）。一个五步的运行里，token 成本可能翻三倍。只有真正需要这种路由时才为它付费。

### Crew vs Flow

这是 2026 年官方文档开篇就给出的框架。

- **Crew。** LLM 驱动的自主性。框架在运行时决定形态。适合：调研、头脑风暴、初稿，以及任何"路径本身就是答案一部分"的场景。难以重放。难以测试。原型成本低。
- **Flow。** 由你掌控的事件驱动图。`@start` 标记入口。`@listen(topic)` 标记一个步骤，当另一步骤发出该 topic 时触发。每个步骤都是普通 Python（内部可以调用 Crew）。适合：生产环境。可观测。可测试。确定性。

官方文档 2026 年的生产环境建议：从 Flow 开始。当自主性值回成本时，再以 `Crew.kickoff()` 调用的形式把 Crew 折叠进 Flow 的步骤中。Flow 给你审计轨迹，Crew 给你探索能力。组合使用，不要二选一。

### 工具集成

给 Agent 接入工具有三种方式。选最简单够用的那种。

1. **`@tool` 装饰器。** 纯函数直接变成工具。函数签名就是 schema；docstring 是 LLM 看到的描述。最适合一次性的辅助函数。

   ```python
   from crewai.tools import tool

   @tool("Search the web")
   def search(query: str) -> str:
       """Return top results for the query."""
       return run_search(query)
   ```

2. **`BaseTool` 子类。** 基于类的工具，带显式参数 schema、异步支持、重试。当工具有状态（一个客户端、一个缓存）或需要结构化参数时使用。

   ```python
   from crewai.tools import BaseTool
   from pydantic import BaseModel

   class SearchArgs(BaseModel):
       query: str
       limit: int = 10

   class SearchTool(BaseTool):
       name = "web_search"
       description = "Search the web and return top results."
       args_schema = SearchArgs

       def _run(self, query: str, limit: int = 10) -> str:
           return self.client.search(query, limit=limit)
   ```

3. **内置工具集。** CrewAI 自带第一方适配器：`SerperDevTool`、`FileReadTool`、`DirectoryReadTool`、`CodeInterpreterTool`、`RagTool`、`WebsiteSearchTool`。一个 import 即可接入。

结构化输出使用 Pydantic。在 Task 上传 `output_pydantic=MyModel`。CrewAI 会用该模型校验 LLM 的响应，要么强制转换、要么重试。配合一个收紧的 `expected_output` 字符串使用。自由文本输出对初稿没问题；结构化输出才是下游 Flow 能消费的东西。

### 记忆钩子

CrewAI 开箱即用提供四种记忆类型。它们可以组合：一个 Crew 可以同时启用全部四种。

> **验证版本：** CrewAI 0.86（2026-05）。近期版本把所有记忆都路由到一个统一的 `Memory` 系统，它封装了这四种存储。下面的概念模型依然成立，但公开的类接口在新版本中可能收敛为单一的 `Memory` 入口；查阅 [CrewAI memory docs](https://docs.crewai.com/concepts/memory) 了解当前 API。

- **短期记忆（Short-term）。** 单次运行内的对话缓冲区。运行结束即清空。
- **长期记忆（Long-term）。** 跨运行持久化。存储在向量数据库中（默认 Chroma，可替换）。按与当前任务的相似度检索。
- **实体记忆（Entity）。** 按实体维度记录的事实。"客户 X 在企业版套餐上。"按实体而非相似度索引。跨运行保留。
- **上下文记忆（Contextual）。** 组装时检索。在 Agent 需要的那一刻拉取相关记忆，而非预先加载。

在 Crew 上用 `memory=True` 启用，或按类型分别配置。底层由你配置的嵌入提供方支撑（默认 OpenAI，可换成本地模型）。记忆是 CrewAI 相对更薄的框架真正体现价值的地方之一；纯 LangGraph 需要你自己把这些全部接好。

### CrewAI 适用的场景

- 三到六个有明确角色名称的智能体，配合协作式工作流。起草、评审、规划、头脑风暴。
- LLM 对下一步的判断本身就是价值一部分的路由场景（Hierarchical）。
- 团队更愿意读 `role + goal + backstory` 而不是读图定义的任何地方。

### CrewAI 不适用的场景

- 顺序严格的确定性 DAG。用 LangGraph（第 13 课）。图的形态才是正确的抽象；CrewAI 的角色框架反而是阻力。
- 亚秒级延迟预算。Hierarchical 增加往返次数。即使是 Sequential，也要序列化包含 backstory 和前序输出的提示词。
- 单智能体循环。跳过框架；一个智能体循环（第 1 课）加一个工具注册表更短。

第 17 课（智能体框架权衡）用一张矩阵把这些列了出来。简短结论：CrewAI 位于"协作式、基于角色"那个角落。

### 依赖形态

独立于 LangChain。Python 3.10 到 3.13。使用 `uv`。Star 数：见 [crewAIInc/crewAI](https://github.com/crewAIInc/crewAI)（2026-05 快照）。AWS Bedrock 集成有文档记载；厂商基准测试声称在 QA 工作负载上比 LangGraph 有大幅加速，但方法论（数据集、硬件、评估指标）并未公开，所以框架厂商给出的数字只当方向性参考。

### 这个模式哪里会出问题

- **backstory 导致提示词膨胀。** 每个智能体 2000 字的 backstory，加上一个五智能体的 crew，第一次工具调用之前就烧光了上下文预算。把 backstory 控制在 200 字以内。在智能体之间复用措辞；不要把团队风格重复写五遍。
- **manager LLM 的 token 开销。** Hierarchical 流程在每次专家调用之前都加一次 manager LLM 调用。一个五任务的 crew 就是六次而不是五次 LLM 调用，而且 manager 调用携带完整任务列表加前序输出。除非路由真正依赖输出，否则换成 Sequential。
- **脆弱的任务交接。** 任务 N 的 `expected_output` 是"一份大纲"。任务 N+1 把它当 `context` 读取并尝试解析三个章节。LLM 生成了四个。下游 Agent 开始即兴发挥。修复方法是在任务 N 上加 `output_pydantic`，让任务 N+1 读到的是类型化对象，而不是自由文本。
- **把 Crew 直接当生产环境。** 自由形态的 Crew 没有 Flow 包装就上了生产。输出波动大；重放不可能；值班同事没法把一次坏的运行和一次好的运行做对比。用 Flow 包起来。

## 从零实现

`code/main.py` 用标准库实现了两种形态，外加一个三智能体 crew。

结构：

- `Agent`、`Task` 数据类，对应 CrewAI 的接口形态。
- `SequentialCrew.kickoff(inputs)` 按声明顺序运行任务，把输出作为 `context` 串联传递。
- `HierarchicalCrew.kickoff(topic)` 增加一个 manager Agent，每轮挑选下一个专家，遇到 "done" 停止。
- `Flow`，带 `@start` 和 `@listen(topic)` 装饰器、一个微型事件循环和一份 trace。
- `tool(name)` 装饰器，对应 CrewAI 的 `@tool` 形态。
- `Memory`，带 `short_term`、`long_term`、`entity` 存储；模拟的相似度计算使用 numpy。
- 模拟的 LLM 响应是硬编码字符串，按角色加输入前缀索引。无网络。确定性。

具体演示：researcher、writer、editor 组成的 crew，产出一份关于 "agent engineering 2026" 的简报。Researcher 拉取（模拟的）资料。Writer 起草。Editor 收紧文字。同一个 crew 再通过一个 Flow 运行一遍，展示确定性的形态。

运行：

```bash
python3 code/main.py
```

Trace 覆盖：sequential crew 通过 `context` 串联输出、hierarchical crew 的 manager 选择过程（researcher、writer、editor，然后 "done"）、flow 用显式 topic（`researched`、`drafted`、`edited`）运行同样三个步骤、通过 `@tool` 路由的工具调用，以及跨两次 kickoff 留存的长期记忆。

Crew 的 trace 是流动的；manager 原则上可以重新排序。Flow 的 trace 是固定的。这个差异就是本课的要点。

## 生产实践

- **CrewAI Flow** 用于生产环境。即使整个 Flow 只有一步、内容就是调用 `Crew.kickoff()`。Flow 提供审计边界。
- **CrewAI Crew（Sequential）** 用于顺序清晰的协作工作，尤其是初稿和评审循环。
- **CrewAI Crew（Hierarchical）** 用于路由依赖输出、且有四个及以上专家的场景。
- **LangGraph**（第 13 课）用于显式状态机、持久化恢复、严格顺序。
- **AutoGen v0.4**（第 14 课）用于 Actor 模型并发和故障隔离。
- **OpenAI Agents SDK**（第 16 课）用于以 OpenAI 为主、带 handoff 和 guardrail 的产品。
- **Claude Agent SDK**（第 17 课）用于以 Claude 为主、带子智能体和会话存储的产品。

## 交付产物

`outputs/skill-crew-or-flow.md` 针对一个任务做出 Crew 还是 Flow 的选择，并搭出最小实现的脚手架。硬性拒绝：没有 backstory 的 Crew、没有显式 topic 的 Flow、专家少于三个的 Hierarchical。

## 常见陷阱

- **把 backstory 当装饰。** 它实际塑造输出。每个智能体测三个变体；差异是真实存在的。选定一个，冻结它。
- **跳过 `expected_output`。** 没有每个任务的契约，下游任务只能接住 LLM 随手产出的东西。Crew 能跑；审计过不了。
- **记忆全程常开。** 长期记忆每次运行都写入。向量数据库不断膨胀。检索越来越多噪声。把写入限定在事实确实需要持久化的任务上。
- **manager 提示词漂移。** Hierarchical 的 manager 提示词是隐式的。如果路由行为变得诡异，用 verbose 模式把它打印出来读一读。
- **在 Crew 里执行带副作用的工具。** Crew 调用工具的次数可能超出预期。POST、DELETE、支付操作属于 Flow 的步骤，绝不该是 Crew 的工具。

## 练习

1. 把 Sequential crew 改写成 Flow。数一数有多少个接触点的波动性下降了。记录哪些地方可读性下降了。
2. 给 crew 加上实体记忆：关于某个客户的事实跨 kickoff 持久化。验证检索能拉到正确的实体。
3. 实现一个 Hierarchical 流程：在 writer 的输出至少有三个段落之前，manager 拒绝路由到 editor。追踪重试过程。
4. 为一个（模拟的）网页搜索接一个 `BaseTool` 子类。对比它和 `@tool` 装饰器版本的 trace 形态。
5. 给 editor 任务加 `output_pydantic=Brief`，其中 `Brief` 有 `title`、`summary`、`sections`。让 writer 任务输出一次格式错误的 JSON；在 trace 中验证 CrewAI 的重试行为。
6. 阅读 CrewAI 文档的导言。把这个玩具版移植到真实的 `crewai` API。标准库版本省略了哪些保证？
7. 给一次真实运行接上 AgentOps 或 Langfuse（第 24 课）。标准库版本里你漏掉了哪些 trace？

## 关键术语

| 术语 | 通俗说法 | 实际含义 |
|------|----------------|------------------------|
| Agent | "人设" | 角色 + 目标 + backstory + 工具 |
| Task | "工作单元" | 描述 + 期望输出 + 负责的智能体 + 可选的结构化输出 |
| Crew | "智能体团队" | Agent + Task + Process 的容器 |
| Process | "执行策略" | Sequential / Hierarchical / Consensus（规划中） |
| Flow | "确定性工作流" | 事件驱动、代码掌控、可测试 |
| Backstory | "人设提示词" | 塑造 Agent 语气与判断力的设定 |
| `@tool` | "函数工具" | 把函数变成 Agent 可调用工具的装饰器 |
| `BaseTool` | "类工具" | 基于类的工具，带参数 schema、重试、异步支持 |
| 实体记忆 | "按实体的事实" | 限定在某个客户 / 账户 / 工单范围内的记忆 |
| 长期记忆 | "跨运行记忆" | 由向量库支撑、在 kickoff 之间留存的记忆 |
| 上下文记忆 | "即时检索" | 在 Agent 需要的那一刻拉取的记忆 |
| Manager LLM | "路由智能体" | Hierarchical 流程中负责挑选下一个任务的额外 LLM |
| `expected_output` | "任务契约" | 告诉 Agent（和审计）应返回什么形状的字符串 |

## 延伸阅读

- [CrewAI docs introduction](https://docs.crewai.com/en/introduction)：核心概念与推荐的生产路径
- [CrewAI Flows guide](https://docs.crewai.com/en/concepts/flows)：事件驱动形态、`@start`、`@listen`
- [CrewAI tools reference](https://docs.crewai.com/en/concepts/tools)：`@tool`、`BaseTool`、内置工具集
- [CrewAI memory](https://docs.crewai.com/en/concepts/memory)：短期、长期、实体、上下文记忆
- [Anthropic, Building Effective Agents](https://www.anthropic.com/research/building-effective-agents)：多智能体何时有用、何时没用
- [LangGraph overview](https://docs.langchain.com/oss/python/langgraph/overview)：状态机路线的替代方案
