# ReWOO 与 Plan-and-Execute：解耦式规划

> ReAct 把思考和行动交织在同一条流水线里。ReWOO 把两者分开：先一次性制定完整计划，再执行。token 用量减少 5 倍，在 HotpotQA 上准确率提升 4%，而且规划器还能蒸馏成 7B 模型。Plan-and-Execute 把它泛化为通用模式；Plan-and-Act 把它扩展到了网页导航任务。

**Type:** Build
**Languages:** Python (stdlib)
**Prerequisites:** Phase 14 · 01 (Agent Loop)
**Time:** ~60 minutes

## 学习目标

- 解释为什么 ReWOO 的 Planner / Worker / Solver 三角色拆分比 ReAct 的交织循环更省 token、也更健壮。
- 用纯标准库实现一个计划 DAG、一个按依赖顺序调度的执行器，以及一个组合各 worker 输出的求解器。
- 借助 2026 年「五种工作流模式」框架（Anthropic），判断一个任务应该采用先规划后执行，还是交织式 ReAct。
- 识别什么时候长程网页或移动端任务需要用到 Plan-and-Act 的合成计划数据。

## 问题背景

ReAct 的「思考-行动-观察」交织循环简单灵活，但每次工具调用都必须携带全部历史上下文——包括之前的每一条思考。token 用量随深度呈平方增长。更糟的是：当某个工具在循环中途失败时，模型必须从错误观察出发重新推导整个计划。

ReWOO（Xu et al., arXiv:2305.18323, 2023 年 5 月）注意到了这一点并下了一个赌注：先一次性规划好全部步骤，并行获取证据，最后再组合出答案。一次 LLM 调用做规划，N 次工具调用取证据（可以并行），一次 LLM 调用做求解。代价是灵活性变差（计划是静态的），换来的是 token 效率大幅提升和更清晰的失败模式。

## 核心概念

### 三个角色

```
Planner:  user_question -> [plan_dag]
Workers:  [plan_dag]     -> [evidence]        (tool calls, possibly parallel)
Solver:   user_question, plan_dag, evidence -> final_answer
```

Planner 产出一个 DAG。每个节点指明一个工具、它的参数，以及它依赖哪些前置节点（形如 `#E1`、`#E2` 的引用）。Worker 按拓扑顺序执行节点。Solver 把所有结果缝合成最终答案。

### 为什么能省 5 倍 token

ReAct 的 prompt 长度随步数线性增长。到第 10 步时，prompt 已经包含了思考 1、行动 1、观察 1、思考 2、行动 2、观察 2……依此类推。每个中间步骤还会冗余地重复原始 prompt。

ReWOO 只需付出一个规划器 prompt（较大）、N 个小型 worker prompt（每个只含工具调用本身，没有思考链），以及一个求解器 prompt。论文在 HotpotQA 上测得 token 用量约减少 5 倍，同时绝对准确率提升 4 个百分点。

### 为什么更健壮

在 ReAct 中，如果 worker 3 失败，循环必须在流程中途从错误里推理出脱困办法。在 ReWOO 中，worker 3 只是返回一个错误字符串；求解器会在上下文中连同原始计划一起看到它，从而可以优雅降级。失败的定位粒度是按节点，而不是按步骤。

### 规划器蒸馏

论文的第二个结论：由于规划器看不到观察结果，你可以用 175B 教师模型的规划器输出去微调一个 7B 模型。小模型负责规划；推理时不再需要大模型。这如今已是标准做法——2026 年的许多生产级智能体都采用小规划器加大执行器（或反过来）的组合。

### Plan-and-Execute（LangChain, 2023）

LangChain 团队 2023 年 8 月的博文把 ReWOO 泛化为一个模式名称：Plan-and-Execute。前置规划器产出步骤列表，执行器逐步执行，可选的重规划器（replanner）可以在观察到结果后修订计划。这比 ReWOO 更接近 ReAct（重规划器把观察结果重新引入规划环节），但保留了省 token 的优势。

### Plan-and-Act（Erdogan et al., arXiv:2503.09572, ICML 2025）

Plan-and-Act 把这一模式扩展到长程网页和移动端智能体。其核心贡献是合成计划数据：一个带标注的轨迹生成器产出计划被显式标注的训练数据。用这些数据微调出的规划器模型，在类 WebArena 任务上超过 30–50 步后仍能正常工作，而单条 ReAct 轨迹在这种任务上早已失去连贯性。

### 什么时候选哪个

| 模式 | 适用场景 |
|---------|------|
| ReAct | 短任务、环境未知、需要响应式异常处理 |
| ReWOO | 工具已知的结构化任务、对 token 敏感、证据可并行获取 |
| Plan-and-Execute | 类似 ReWOO，但部分执行后可重新规划 |
| Plan-and-Act | 长程任务（>30 步）、网页/移动端/计算机操作 |
| Tree of Thoughts | 搜索的代价值得付出（第 04 课） |

Anthropic 2024 年 12 月的指导原则：从最简单的方案开始。如果任务只是一次工具调用加一段总结，不要去搭 ReWOO。如果任务是 40 步的研究作业，也不要只用 ReAct。

## 从零实现

`code/main.py` 实现了一个玩具版 ReWOO：

- `Planner` —— 一个脚本化策略，根据 prompt 产出计划 DAG。
- `Worker` —— 通过注册表分发每个节点的工具调用。
- `Solver` —— 脚本化的组合逻辑，读取证据并产出最终答案。
- 依赖解析 —— 形如 `#E1` 的引用会被替换为前置 worker 的输出。

演示回答的问题是「法国首都的人口是多少（四舍五入到百万）？」，使用一个两步计划：(1) 查首都，(2) 查人口，然后求解。

运行：

```
python3 code/main.py
```

运行轨迹会先展示完整计划，然后是各 worker 的结果，最后是求解器的组合过程。把 token 数（我们打印了一个粗略的字符计数）和 ReAct 式交织运行对比一下——在这类结构化任务上 ReWOO 完胜。

## 生产实践

LangGraph 把 Plan-and-Execute 作为现成配方提供（ReAct 用 `create_react_agent`，plan-execute 用自定义图）。CrewAI 的 Flows 直接编码了这一模式：你预先定义好任务，Flow DAG 负责执行。Plan-and-Act 的合成数据方法目前仍主要停留在研究阶段；其运行时模式（显式计划 DAG）则通过 LangGraph 和 CrewAI Flows 进入了生产环境。

## 交付产物

`outputs/skill-rewoo-planner.md` 在给定工具目录的前提下，根据用户请求生成 ReWOO 计划 DAG。它会在交给执行器之前校验计划（无环、每个引用都能解析、每个工具都存在）。

## 练习

1. 把相互独立的计划节点的 worker 执行并行化。在一个含 2 个并行组的 6 节点 DAG 上，这能带来什么收益？
2. 添加一个重规划节点，在任意 worker 返回错误时触发。把 ReWOO 变成 Plan-and-Execute 的最小改动是什么？
3. 用一个小模型（7B 量级）替换 `Planner`，`Solver` 保留前沿模型。比较端到端质量——这种拆分会在哪里失效？
4. 阅读 ReWOO 论文第 4 节关于规划器蒸馏的内容。从概念上复现 175B -> 7B 的结果：你需要什么训练数据，又如何评估计划质量？
5. 把这个玩具实现移植成 Plan-and-Act 的轨迹形态：计划是一个序列，而不是 DAG。哪些权衡会发生变化？

## 关键术语

| 术语 | 大家怎么说 | 实际含义 |
|------|----------------|------------------------|
| ReWOO | "Reasoning without observations"（无观察推理） | 先规划，再并行取证据，最后求解——规划 prompt 中不含观察结果 |
| Plan-and-Execute | "LangChain 的 plan-execute 模式" | ReWOO 加上一个执行后可选的重规划节点 |
| Plan-and-Act | "规模化的 plan-execute" | 显式的规划器/执行器拆分，配合面向长程任务的合成计划训练数据 |
| 证据引用（Evidence reference） | "#E1, #E2, ..." | 计划节点中的占位符，调度时被替换为前置 worker 的输出 |
| 规划器蒸馏（Planner distillation） | "小规划器，大执行器" | 用大教师模型的规划轨迹微调一个小模型 |
| token 效率 | "更少的往返次数" | 论文中在 HotpotQA 上比 ReAct 少用 5 倍 token |
| DAG 执行器 | "拓扑调度器" | 按依赖顺序运行计划节点；每一层内可并行 |

## 延伸阅读

- [Xu et al., ReWOO: Decoupling Reasoning from Observations (arXiv:2305.18323)](https://arxiv.org/abs/2305.18323) —— 原始论文
- [Erdogan et al., Plan-and-Act (arXiv:2503.09572)](https://arxiv.org/abs/2503.09572) —— 基于合成计划的规模化规划器-执行器
- [LangGraph Plan-and-Execute tutorial](https://docs.langchain.com/oss/python/langgraph/overview) —— 框架配方
- [Anthropic, Building Effective Agents](https://www.anthropic.com/research/building-effective-agents) —— 选用能解决问题的最简单模式
