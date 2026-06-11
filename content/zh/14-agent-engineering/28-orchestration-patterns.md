# 编排模式：监督者、蜂群与层级结构

> 2026 年的各类框架中反复出现四种编排模式：监督者-工作者（supervisor-worker）、蜂群/点对点（swarm / peer-to-peer）、层级式（hierarchical）、辩论（debate）。Anthropic 的建议是："关键在于为你的需求构建合适的系统。"从简单做起；只有当单个智能体加上五种工作流模式不够用时，才引入拓扑结构。

**Type:** Learn + Build
**Languages:** Python (stdlib)
**Prerequisites:** Phase 14 · 12 (Workflow Patterns), Phase 14 · 25 (Multi-Agent Debate)
**Time:** ~60 minutes

## 学习目标

- 说出四种反复出现的编排模式，以及各自适用的场景。
- 描述 2026 年 LangChain 的建议：基于工具调用的监督 vs 监督者库。
- 解释 Anthropic 的"构建合适的系统"原则，以及它如何决定拓扑选择。
- 仅用标准库、基于同一个脚本化 LLM 实现全部四种模式。

## 问题背景

很多团队还没真正需要"多智能体"，就急着上手。四种模式在各类框架中反复出现；一旦你能叫出它们的名字，就能选出合适的那一个——或者干脆完全跳过拓扑设计。

## 核心概念

### 监督者-工作者（Supervisor-Worker）

- 一个中心化的路由 LLM 将任务分派给各个专家智能体。
- 由它决定：回到自身继续循环、移交给某个专家，还是终止。
- 专家之间互不通信；所有路由都经过监督者。

相关框架：LangGraph 的 `create_supervisor`、Anthropic 的 orchestrator-workers、CrewAI 的 Hierarchical Process。

**2026 年 LangChain 的建议：** 通过直接工具调用来实现监督，而不是使用 `create_supervisor`。这能带来更精细的上下文工程控制——由你精确决定每个专家看到什么。

### 蜂群 / 点对点（Swarm / Peer-to-Peer）

- 智能体通过共享的工具界面直接相互移交。
- 没有中心路由器。
- 延迟低于监督者模式（跳数更少）。
- 更难推理和分析（没有单一控制点）。

相关框架：LangGraph 的 swarm 拓扑、OpenAI Agents SDK 的 handoffs（当所有智能体都能移交给所有其他智能体时）。

### 层级式（Hierarchical）

- 监督者管理子监督者，子监督者再管理工作者。
- 在 LangGraph 中以嵌套子图实现；在 CrewAI 中以嵌套 crew 实现。
- 能扩展到大规模智能体群体，代价是运维复杂度上升。

什么时候需要它：当单个监督者的上下文预算装不下所有专家的描述时。

### 辩论（Debate）

- 并行提议者 + 迭代式交叉评审（第 25 课）。
- 严格说不算编排——更接近验证——但在各框架中常作为一种拓扑选项出现。

### CrewAI 的 Crew vs Flow

CrewAI 正式定义了两种部署模式：

- **Flow** 用于确定性的事件驱动自动化（推荐的生产环境起点）。
- **Crew** 用于自主的、基于角色的协作。

这与上述四种模式是正交的，但可以映射到拓扑：Flow 通常对应监督者或层级式；Crew 通常对应带 LLM 路由器的监督者模式。

### Anthropic 的指导原则

"在 LLM 领域，成功不在于构建最精巧复杂的系统，而在于为你的需求构建合适的系统。"

决策顺序：

1. 单智能体 + 工作流模式（第 12 课）——从这里开始。
2. 监督者-工作者——当你有 2-4 个专家时。
3. 蜂群——当延迟比推理清晰度更重要时。
4. 层级式——仅当监督者的上下文预算撑不住时。
5. 辩论——当准确性比成本更重要时。

### 这一模式容易出错的地方

- **拓扑优先思维。** 还没搞清多智能体到底解决什么问题，就先喊"我们需要多智能体"。
- **蜂群中的来回弹跳移交。** A -> B -> A -> B。请使用跳数计数器。
- **虚假层级。** 因为要"企业级"就搭三层，实际只有两个团队。把它压扁。

## 从零实现

`code/main.py` 仅用标准库、基于一个脚本化 LLM 实现了全部四种模式：

- `Supervisor`——中心路由器。
- `Swarm`——带直接移交的点对点。
- `Hierarchical`——监督者之上的监督者。
- `Debate`——并行提议者 + 评审。

每种模式处理同一个包含三种意图的任务（退款 / bug / 销售）。各模式的 trace 形状各不相同。

运行：

```
python3 code/main.py
```

输出：每个模式的 trace + 操作计数。监督者最干净；蜂群最短；层级最深；辩论最贵。

## 生产实践

- **LangGraph** 用于监督者和层级式（嵌套子图）。
- **OpenAI Agents SDK** 用于"移交即工具"（监督者形态）。
- **CrewAI Flow** 用于生产环境的确定性流程。
- **自研实现** 用于辩论模式，或当你需要完全精确的控制时。

## 交付产物

`outputs/skill-orchestration-picker.md` 负责选定一种拓扑并实现它。

## 练习

1. 通过移除路由器，把一个监督者-工作者系统改造成蜂群。什么坏了？什么变好了？
2. 给蜂群加一个跳数计数器：3 次移交后拒绝。它能捕获 A->B->A 的弹跳吗？
3. 为一个拥有 12 个专家的领域构建两层层级系统。如果不做嵌套，上下文预算会在哪里失效？
4. 在贴近生产形态的负载上对四种模式做性能剖析。哪种模式在哪个指标（延迟、成本、准确性、可调试性）上胜出？
5. 阅读 Anthropic 的 "Building Effective Agents" 一文。把你的每个生产流程映射到四种模式之一。有没有映射不上的？

## 关键术语

| 术语 | 人们常说 | 实际含义 |
|------|----------------|------------------------|
| 监督者-工作者（Supervisor-Worker） | "路由器 + 专家" | 中心 LLM 分派给各专家；专家之间互不通信 |
| 蜂群（Swarm） | "点对点" | 通过共享工具直接移交；没有中心路由器 |
| 层级式（Hierarchical） | "监督者之上的监督者" | 用嵌套子图支撑大规模智能体群体 |
| 辩论（Debate） | "提议者 + 评审" | 并行提议者，交叉评审（第 25 课） |
| 基于工具调用的监督 | "不依赖库的监督者" | 把监督者实现为直接工具调用，以便控制上下文 |
| Crew | "自主团队" | CrewAI 基于角色的协作模式 |
| Flow | "确定性工作流" | CrewAI 面向生产的事件驱动模式 |

## 延伸阅读

- [Anthropic, Building Effective Agents](https://www.anthropic.com/research/building-effective-agents) —— 五种模式 + 智能体 vs 工作流
- [LangGraph overview](https://docs.langchain.com/oss/python/langgraph/overview) —— 监督者、蜂群、层级式
- [CrewAI docs](https://docs.crewai.com/en/introduction) —— Crew vs Flow
- [Du et al., Society of Minds (arXiv:2305.14325)](https://arxiv.org/abs/2305.14325) —— 辩论模式
