# 多智能体原语模型

> 2026 年面世的每一个多智能体框架——AutoGen、LangGraph、CrewAI、OpenAI Agents SDK、Microsoft Agent Framework——都是一个四维设计空间中的一个点。四个原语，仅此而已：智能体（agent）、交接（handoff）、共享状态（shared state）、编排器（orchestrator）。本课从零构建这四个原语，在一个玩具系统上把它们全部跑一遍，然后把每个主流框架映射到同一组坐标轴上，让你今后只用一段话就能读懂任何新发布的框架。

**Type:** Learn
**Languages:** Python (stdlib)
**Prerequisites:** Phase 14 (Agent Engineering), Phase 16 · 01 (Why Multi-Agent)
**Time:** ~60 minutes

## 问题背景

每隔六个月就有一个新的多智能体框架问世。2023 年的 AutoGen。2024 年的 CrewAI。同样在 2024 年的 LangGraph 和 OpenAI Swarm。2025 年 4 月的 Google ADK。2026 年 2 月的 Microsoft Agent Framework RC。每份新闻稿都宣称自己是"正确的抽象"。

如果你试图一个一个地学，迟早会被耗尽。它们的 API 看起来各不相同。文档对"智能体"是什么各执一词。一个框架把共享内存叫"黑板（blackboard）"，另一个叫"消息池（message pool）"，第三个叫"StateGraph"。你开始怀疑这个领域只是在原地打转。

并非如此。剥开营销外衣，四个原语是稳定的。学一次，以后读任何新框架都只需要一段话。

## 核心概念

### 四个原语

1. **智能体（Agent）**——一个系统提示词加一份工具列表。无状态；每次运行都从它的系统提示词和当前消息历史开始。
2. **交接（Handoff）**——控制权从一个智能体到另一个智能体的结构化转移。从机制上看，要么是一个返回新智能体的工具调用，要么是一条按条件流转的图边。
3. **共享状态（Shared state）**——任何能被多个智能体读取（有时也能写入）的数据结构。消息池、黑板、键值存储、向量记忆。
4. **编排器（Orchestrator）**——决定下一个谁发言的角色。可选方案：显式的图（确定性）、由 LLM 充当的发言者选择器（软性）、上一个发言者的交接调用（OpenAI Swarm），或基于队列的调度器（swarm 架构）。

这就是整个设计空间。每个框架只是在每条轴上选了一组默认值；其余的都是表层语法。

### 2026 年的各个框架如何映射到这个模型

| 框架 | 智能体 | 交接 | 共享状态 | 编排器 |
|-----------|-------|---------|--------------|--------------|
| OpenAI Swarm / Agents SDK | `Agent(instructions, tools)` | 工具返回 Agent | 留给调用方自理 | LLM 的下一次交接调用 |
| AutoGen v0.4 / AG2 | `ConversableAgent` | GroupChat 上的发言者选择器 | 消息池 | 选择器函数（LLM 或轮询） |
| CrewAI | `Agent(role, goal, backstory)` | `Process.Sequential / Hierarchical` | Task 输出串联 | 管理者 LLM 或静态顺序 |
| LangGraph | 节点函数 | 图边 + 条件 | `StateGraph` reducer | 由图决定，确定性 |
| Microsoft Agent Framework | 智能体 + 编排模式 | 取决于具体模式 | 线程 / 上下文 | 取决于具体模式 |
| Google ADK | 智能体 + A2A card | A2A 任务 | A2A artifacts | 由宿主决定 |

表面差异看起来巨大。底层：同样的四个旋钮。

### 为什么这很重要

一旦看清了这些原语，框架比较就变成一张简短的检查清单：

- 编排器是把路由交给 LLM 信任（Swarm），还是把路由固化在代码里（LangGraph）？
- 共享状态是全量历史（GroupChat）还是投影视图（StateGraph reducer）？
- 智能体能否修改彼此的提示词（CrewAI 的管理者），还是只能交接（Swarm）？

这三个问题就能回答 80% 的"哪个框架适合某个问题"。你不再到处选购"最好的多智能体框架"，而是开始围绕你真正在意的那条轴来做设计。

### 无状态洞察

除共享状态之外的每个原语都是无状态的。智能体是 (prompt, tools) 的函数。交接是一次函数调用。编排器是一个调度器。**系统中唯一有状态的东西就是共享状态。**所有有意思的 bug 都住在那里：记忆投毒（第 15 课）、消息排序、版本管理、写入争用。

隐藏共享状态的框架（Swarm）把问题推给调用方。集中管理共享状态的框架（LangGraph 的 checkpoint、AutoGen 的消息池）让它变得可检视，但把协调成本转移到了共享状态的实现上。

### 单个原语的解剖

#### 智能体

```
Agent = (system_prompt, tools, model, optional_name)
```

没有记忆。没有状态。两个系统提示词和工具都相同的智能体是可以互换的。一切看起来像智能体私有状态的东西，实际上都在共享状态或交接协议里。

#### 交接

```
Handoff = (from_agent, to_agent, reason, payload)
```

三种主流实现：

- **函数返回**——工具返回下一个智能体。这是 OpenAI Swarm 的模式。路由信息内嵌在智能体的工具 schema 中。
- **图边**——LangGraph。边是声明式的。LLM 产出一个值，由条件来选择下一个节点。
- **发言者选择**——AutoGen GroupChat。一个选择器函数（有时本身就是一次 LLM 调用）读取消息池并挑选下一个发言者。

#### 共享状态

```
SharedState = { messages: [], artifacts: {}, context: {} }
```

至少是一份消息列表。通常还有更多：结构化产物（CrewAI 的 Task 输出）、类型化上下文（LangGraph 的 reducer）、外部记忆（MCP、向量数据库）。

两种拓扑：**全量池**（每个智能体看到每条消息）和**投影视图**（智能体只看到按角色限定的视图）。全量池简单但扩展性差。投影视图可扩展，但需要预先做好 schema 设计。

#### 编排器

```
Orchestrator = ({state, last_speaker}) -> next_agent
```

四种风格：

- **静态**——图在构建时就固定（LangGraph 确定性模式、CrewAI Sequential）。
- **LLM 选择**——由一个 LLM 读取消息池并挑选下一个发言者（AutoGen、CrewAI Hierarchical）。
- **交接驱动**——当前智能体通过调用交接工具自行决定（Swarm）。
- **队列驱动**——工作者从共享队列里拉取任务；没有显式的"下一个发言者"（swarm 架构、Matrix）。

### 框架之间真正变化的是什么

一旦原语固定下来，剩下的设计决策就是：

- **记忆策略**——临时记忆还是持久化检查点（LangGraph 的 checkpointer）。
- **安全边界**——谁有权批准一次交接（人在回路）。
- **成本核算**——按智能体设定的 token 预算。
- **可观测性**——追踪交接、持久化状态以便回放。

所有这些都可以在四个原语之上实现。它们没有一个是新原语。

## 从零实现

`code/main.py` 用约 150 行标准库 Python 实现了这四个原语。没有真实 LLM——每个智能体都是一个脚本化策略，这样注意力可以集中在协调结构上。

该文件导出：

- `Agent`——由名称、系统提示词、工具、策略函数组成的 dataclass。
- `Handoff`——一个返回新智能体的函数。
- `SharedState`——线程安全的消息池。
- `Orchestrator`——三个变体：`StaticOrchestrator`、`HandoffOrchestrator`、`LLMSelectorOrchestrator`（模拟实现）。

演示程序把同一条三智能体流水线（研究 → 写作 → 评审）分别在三种编排器类型上跑一遍，最后打印消息池。你会看到输出的差异只在于*谁来挑选下一个发言者*；各次运行中的智能体和共享状态完全相同。

运行方式：

```
python3 code/main.py
```

预期输出：三次编排器运行，每种模式一次。每次都打印最终的消息池。如果研究员智能体提前判定任务完成，交接驱动的那一轮会触达更少的智能体——这正是 LLM 路由权衡的微缩版。

## 生产实践

`outputs/skill-primitive-mapper.md` 是一个技能：它读取任何多智能体代码库或框架文档，返回四原语映射。在新框架发布时先跑一遍它，用一段话建立理解，再去深读文档。

## 交付产物

在采用一个新框架之前，先为它写出原语映射。如果写不出来，要么文档不完整，要么这个框架发明了第五个原语（很罕见——检查一下是不是某种你没见过的共享状态变体）。

把这份映射固定在你的架构文档里。新成员加入时，先发给他们这份映射，再发 API 文档。框架版本变更时，对比映射的差异，而不是 changelog。

## 练习

1. 用不同的智能体策略运行 `code/main.py` 三次。观察编排器的选择如何改变哪些智能体会被执行。
2. 实现第四种编排器类型：队列驱动型，让智能体轮询共享状态来领取工作。会出现什么死锁？如何检测？
3. 拿 LangGraph 快速入门（https://docs.langchain.com/oss/python/langgraph/workflows-agents）用四原语重写一遍。LangGraph 的哪些抽象能一一对应，哪些只是便利性的封装？
4. 阅读 OpenAI Swarm cookbook（https://developers.openai.com/cookbook/examples/orchestrating_agents）。指出四个原语中 Swarm 把哪一个做得最顺手，又把哪一个推给了调用方。
5. 在上表中找出一个完全隐藏共享状态的框架。解释当智能体需要跨交接协调、却无法重读历史时，会出什么问题。

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|----------------|------------------------|
| 智能体（Agent） | "带工具的 LLM" | 一个 `(system_prompt, tools, model)` 三元组。无状态。 |
| 交接（Handoff） | "控制权转移" | 一次结构化调用，指明下一个智能体和可选载荷。三种实现：函数返回、图边、发言者选择。 |
| 共享状态（Shared state） | "记忆" / "上下文" | 多智能体系统中唯一有状态的部分。消息池或黑板。 |
| 编排器（Orchestrator） | "协调者" | 决定下一个谁运行的角色。静态图、LLM 选择器、交接驱动或队列驱动。 |
| 原语（Primitive） | "抽象" | 每个框架都要参数化的四条轴之一。不是某个框架的特性。 |
| 消息池（Message pool） | "共享聊天历史" | 全量历史的共享状态。易于推理，扩展性差。 |
| 投影状态（Projected state） | "受限视图" | 共享状态的按角色定制视图。可扩展，但需要 schema 设计。 |
| 发言者选择（Speaker selection） | "下一个谁说话" | 一种编排器模式：由一个函数（通常是 LLM）从一组智能体中挑选下一个。 |

## 延伸阅读

- [OpenAI cookbook: Orchestrating Agents — Routines and Handoffs](https://developers.openai.com/cookbook/examples/orchestrating_agents)——对交接驱动编排最清晰的阐述
- [AutoGen stable docs](https://microsoft.github.io/autogen/stable/)——GroupChat + 发言者选择是 LLM 选择型编排的参考实现
- [LangGraph workflows and agents](https://docs.langchain.com/oss/python/langgraph/workflows-agents)——图边编排与基于 reducer 的共享状态
- [CrewAI introduction](https://docs.crewai.com/en/introduction)——role-goal-backstory 智能体，Sequential / Hierarchical 流程
- [AG2 (community AutoGen continuation)](https://github.com/ag2ai/ag2)——Microsoft 将 v0.4 转入维护模式后仍在活跃的 AutoGen v0.2 分支
