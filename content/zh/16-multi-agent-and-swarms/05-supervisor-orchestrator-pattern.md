# Supervisor / Orchestrator-Worker 模式（监督者/编排者-工作者模式）

> 一个主导智能体负责规划和分派任务；多个专门化的工作者在并行上下文中执行并汇报结果。这正是 Anthropic Research 系统背后的模式（Claude Opus 4 作为主导，Sonnet 4 作为子智能体），在内部研究评测中比单智能体 Opus 4 高出 90.2%。Anthropic 的工程博文指出，BrowseComp 上 80% 的方差仅由 token 用量就能解释——多智能体之所以胜出，很大程度上是因为每个子智能体都拥有全新的上下文窗口。本课将从基础原语出发构建 supervisor 模式，并讲解 2026 年来自生产部署的工程经验。

**Type:** Learn + Build
**Languages:** Python (stdlib, `threading`)
**Prerequisites:** Phase 16 · 04 (Primitive Model)
**Time:** ~75 minutes

## 问题背景

研究类任务是单智能体系统失效的典型场景。你问"2023 到 2026 年间多智能体系统发生了哪些变化？"单个智能体只能顺序读完五篇论文，把一半上下文塞满论文原文，然后还要对它们整体进行推理。等读到第五篇时，它已经忘了第一篇。它也无法并行化。

supervisor 模式解决了这个问题：一个主导智能体规划检索方案，把每个子问题分派给一个工作者，最后做综合。每个工作者针对一个窄问题独享自己的 200k token 窗口。主导智能体从不接触论文原文——只看工作者的摘要。

Anthropic 的生产级 Research 系统报告显示，在内部研究评测上比单个 Opus 4 高出 90.2%。同一篇博文还指出，BrowseComp 上 80% 的方差*仅由 token 用量*就能解释。每个子智能体拥有全新上下文，是其中的主要机制。

## 核心概念

### 模式本身

```
                 ┌──────────────┐
                 │   Lead       │  plans, decomposes,
                 │  (Opus 4)    │  synthesizes
                 └──┬────┬───┬──┘
                    │    │   │
            ┌───────┘    │   └───────┐
            ▼            ▼           ▼
      ┌─────────┐  ┌─────────┐  ┌─────────┐
      │ Worker1 │  │ Worker2 │  │ Worker3 │
      │(Sonnet) │  │(Sonnet) │  │(Sonnet) │
      └─────────┘  └─────────┘  └─────────┘
         fresh       fresh        fresh
         context     context      context
```

主导智能体从不阅读原始材料。各工作者在主导智能体做综合之前互相看不到彼此的工作。每个箭头都是一次只传递窄产物的交接。

### 为什么它能赢

三个机制：

1. **每个子智能体拥有全新上下文。** 一个探究"FIPA-ACL 渊源"的工作者，不必背负主导智能体规划阶段消耗的 40k token。它针对一个问题独享 200k 窗口。
2. **通过提示词实现专门化。** 主导智能体的提示词是"分解和综合"，而不是"做研究"。每个工作者的提示词都很窄："找出 X 中发生了哪些变化"。聚焦的提示词产生聚焦的输出。
3. **并行性。** 工作者并发运行。墙钟时间大致是 `max(worker_times) + plan + synthesis`，而不是 `sum(worker_times)`。

### 工程经验（Anthropic 2025）

Anthropic 的博文列出了若干在 2026 年仍然适用的生产经验：

- **按查询复杂度调配投入。** 简单查询：一个智能体，3-10 次工具调用。复杂查询：10 个以上智能体。这个估算必须由主导智能体来做，而不是调用方。
- **先广后深。** 先分解成较宽泛的子问题，如果答案值得深挖，再针对单个子问题派出更多工作者。
- **彩虹部署（rainbow deployment）。** 智能体是长时运行且有状态的，传统蓝绿部署行不通。Anthropic 采用彩虹部署：新版本逐步上线，旧版本逐渐排空。
- **token 用量是主导因素。** 多智能体的 token 消耗约为单智能体的 15 倍。只有当任务价值能覆盖成本时才使用它。

### LangGraph 的转向

LangGraph 最初发布了 `langgraph-supervisor` 库，提供高层封装的 `create_supervisor` 辅助函数。2025 年 LangChain 把推荐方式改为直接通过工具调用实现 supervisor 模式，因为工具调用能更好地控制*supervisor 看到什么*（即上下文工程）。该库仍然可用；但文档现在推荐工具调用形式。

### 失败模式

- **主导智能体把计划"幻觉"出来。** 如果主导智能体生成的子问题并没有真正分解原始问题，工作者就会在错误的目标上做精准研究。
- **工作者过度探索。** 没有明确的范围边界时，工作者会偏离分派给它的子问题，污染综合环节。
- **综合冲突。** 两个工作者返回相互矛盾的事实。主导智能体要么重新提问（增加一轮），要么显式标注分歧。最糟糕的失败是悄悄选边站：用户永远不知道曾经存在分歧。

### 什么时候不该用 supervisor

- **顺序型任务。** 如果第 2 步确实需要第 1 步的输出，并行就毫无收益。请用流水线（CrewAI Sequential、LangGraph 线性图）。
- **简单查询。** 单智能体处理得更快更便宜。在派出工作者之前先用主导智能体的"投入调配"检查。
- **严格确定性。** supervisor 依赖 LLM 自主选择分派目标。当审计/重放比适应性更重要时，静态图是更好的选择。

```figure
supervisor-hierarchy
```

## 从零实现

`code/main.py` 用 `threading` 实现了一个带三个并行工作者的 supervisor。主导智能体把一个查询分解为子问题，工作者并发处理各自的子问题，最后由主导智能体综合。没有真实 LLM——工作者用脚本模拟"抓取并摘要"的过程。

关键结构：

- `Lead.plan(query)` 把一个查询拆成 3 个子问题。
- `Worker.run(sub_q)` 返回一段假摘要（在生产环境中可以换成任何会用工具的智能体）。
- `Lead.run(query)` 在线程中启动工作者、join 等待，然后综合。

运行：

```
python3 code/main.py
```

输出会展示计划、带起止时间戳的并行工作者轨迹，以及最终的综合结果。你能看到墙钟时间的收益：三个各耗时 0.3 秒的工作者总共只用约 0.35 秒，而不是 0.9 秒。

## 生产实践

`outputs/skill-supervisor-designer.md` 接收一个用户查询，产出一份 supervisor 模式设计方案：主导智能体的系统提示词、工作者角色、子问题分解规则，以及综合模板。在构建新的研究型智能体系统之前先用它。

## 交付产物

部署 supervisor 模式前的检查清单：

- **模型搭配。** 主导智能体用推理级模型（Opus 级别、`o3` 级别）。工作者用更快更便宜的模型（Sonnet、`o4-mini`）。
- **工作者超时。** 任何超过中位运行时长 2 倍的工作者直接终止；主导智能体要么用更窄的范围重新派出，要么不带它继续。
- **每个工作者的 token 上限。** 硬性限制（比如预期综合输入的 10 倍）可以防止失控的工作者烧光预算。
- **可观测性。** 追踪主导智能体的计划、每个工作者的工具调用，以及综合过程。这是任何事后调试的基础。
- **彩虹式发布。** 有状态的长时运行智能体需要渐进式版本切换，而不是热替换。

## 练习

1. 运行 `code/main.py`，然后把主导智能体改成派出 5 个工作者而非 3 个。观察墙钟时间的变化。在这个演示中，工作者数量达到多少时，启动开销会超过并行带来的节省？
2. 实现工作者超时机制：终止任何运行超过 0.5 秒的工作者，让主导智能体只综合剩余结果。要知道某个工作者被砍掉，你需要哪些可观测性手段？
3. 给主导智能体的综合环节加一步冲突检测：如果两个工作者返回相互矛盾的答案，主导智能体应标注分歧而不是择一采用。在不调用 LLM 的情况下如何检测矛盾？
4. 阅读 Anthropic 的 Research 系统工程博文。列出这个玩具演示要在生产环境运行所需采纳的三项实践。
5. 比较 LangGraph 的 `create_supervisor`（遗留方案）和新的工具调用推荐方案。哪一种能让你更好地控制 supervisor 看到的内容？为什么 Anthropic 明确只把子答案而非工作者的原始上下文传入综合环节？

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|----------------|------------------------|
| Supervisor | "主导智能体" | 负责规划、分派和综合的编排智能体。本身不执行具体工作。 |
| Worker | "子智能体" | 由 supervisor 调用、范围窄且拥有独立上下文窗口的专注智能体。 |
| Orchestrator-worker | "supervisor 模式" | 同一概念的不同叫法。2026 年的文献两个名字都在用。 |
| 全新上下文（fresh context） | "干净窗口" | 工作者的上下文从其系统提示词和被分派的问题开始，不包含主导智能体的历史。 |
| 彩虹部署（rainbow deployment） | "渐进式发布" | 长时运行的有状态智能体需要按版本逐步排空替换，而不是蓝绿部署。 |
| token 主导性 | "上下文才是变量" | 据 Anthropic，研究评测中 80% 的方差来自总 token 用量，而不是模型选择。 |
| 投入调配（scale effort） | "智能体数量匹配复杂度" | 主导智能体估算查询难度，据此派出 1 个或 10 个以上工作者。 |
| 综合冲突 | "工作者意见不一" | 两个工作者返回相互矛盾的事实；主导智能体必须把分歧摆到明面上，而不是悄悄择一。 |

## 延伸阅读

- [Anthropic engineering — How we built our multi-agent research system](https://www.anthropic.com/engineering/multi-agent-research-system) — supervisor 模式的生产级参考
- [LangGraph workflows and agents](https://docs.langchain.com/oss/python/langgraph/workflows-agents) — 工具调用式 supervisor 现已是推荐形式
- [LangGraph supervisor reference](https://reference.langchain.com/python/langgraph-supervisor) — 遗留辅助库，2026 年生产环境仍在使用
- [OpenAI cookbook — Orchestrating Agents: Routines and Handoffs](https://developers.openai.com/cookbook/examples/orchestrating_agents) — 基于 handoff 的 supervisor 变体
