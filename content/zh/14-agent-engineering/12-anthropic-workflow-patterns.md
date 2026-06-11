# Anthropic 的工作流模式：简单优先于复杂

> Schluntz 与 Zhang（Anthropic，2024 年 12 月）将工作流（预定义路径）与智能体（动态工具调用）区分开来。五种工作流模式覆盖了大多数场景。从直接调用 API 开始，只有当步骤无法预先确定时才引入智能体。

**Type:** Learn + Build
**Languages:** Python (stdlib)
**Prerequisites:** Phase 14 · 01 (Agent Loop)
**Time:** ~60 minutes

## 学习目标

- 说出 Anthropic 的五种工作流模式：提示链（prompt chaining）、路由（routing）、并行化（parallelization）、编排器-工作器（orchestrator-workers）、评估器-优化器（evaluator-optimizer）。
- 解释智能体与工作流的区别，以及各自的工程成本。
- 判断何时该选工作流而不是智能体（以及反过来何时该选智能体）。
- 只用标准库，基于一个脚本化的 LLM 实现全部五种模式。

## 问题背景

不少团队把多智能体框架用在一个函数调用就能解决的问题上。这种代价是实实在在的：框架增加的层层抽象会遮蔽提示词、隐藏控制流，并诱发过早的复杂化。Schluntz 与 Zhang 在 2024 年 12 月发表的那篇文章是业界引用最多的反思之声：从简单做起，只有当复杂性能挣回它的成本时才引入。

## 核心概念

### 工作流 vs 智能体

- **工作流（Workflow）。** LLM 与工具按预定义的代码路径编排。图由工程师掌控。
- **智能体（Agent）。** LLM 动态指挥自己的工具、自主决定执行步骤。图由模型掌控。

两者各有用武之地。工作流更便宜、更快、也更容易调试。智能体能解锁开放式问题，但其失败模式更难推断。

### 增强型 LLM

这是全部五种模式的基础：一个接入了三种能力的 LLM——搜索（检索）、工具（行动）、记忆（持久化）。任何一次 API 调用都可以用上这些能力。

### 五种模式

1. **提示链（Prompt chaining）。** 第 1 次调用的输出作为第 2 次调用的输入。当任务可以干净地做线性分解时使用。各步骤之间可以加入可选的程序化校验关卡。

2. **路由（Routing）。** 由一个分类器 LLM 决定调用哪个下游 LLM 或工具。当类别截然不同的输入需要不同处理方式时使用（一线客服 vs 退款 vs 缺陷 vs 销售）。

3. **并行化（Parallelization）。** 并发运行 N 次 LLM 调用，再聚合结果。有两种形态：分段（sectioning，处理不同的块）和投票（voting，同一提示词运行 N 次，多数表决或综合）。

4. **编排器-工作器（Orchestrator-workers）。** 一个编排器 LLM 动态决定运行哪些工作器（同样是 LLM），并综合它们的输出。与智能体循环类似，但编排器不会无限循环下去。

5. **评估器-优化器（Evaluator-optimizer）。** 一个 LLM 提出答案，另一个 LLM 对其评估。迭代直到评估器通过。这是 Self-Refine（第 05 课）的泛化形式。

### 工作流胜过智能体的场景

- **可预测的任务。** 如果你能把步骤一一列举出来，那就应该这么做。
- **成本受限的任务。** 工作流的步骤数有上界；智能体可能失控膨胀。
- **合规受限的任务。** 审计人员想直接读到那张图，而不是从执行轨迹里去推断它。

### 智能体胜过工作流的场景

- **开放式研究。** 当下一步取决于上一步返回了什么。
- **长度不定的任务。** 几分钟到几小时的工作量，步骤数未知。
- **全新领域。** 当你还不知道正确的工作流长什么样——先探索，再固化。

### 与之配套的上下文工程

《Effective context engineering for AI agents》（Anthropic 2025）将相邻的这门学科形式化：200k 的上下文窗口是预算，而不是容器。包含什么、何时压缩、何时放任上下文增长。详见 Phase 14 关于上下文压缩的课程（在重新编号之前，是本课程体系中 Phase 14 较早的第 06 课）。

## 从零实现

`code/main.py` 基于一个 `ScriptedLLM` 实现了全部五种工作流模式：

- `prompt_chain(input, steps)` —— 顺序执行。
- `route(input, classifier, handlers)` —— 分类 + 分发。
- `parallel_vote(prompt, n, aggregator)` —— 运行 N 次，聚合。
- `orchestrator_workers(task, workers)` —— 编排器挑选工作器。
- `evaluator_optimizer(task, proposer, evaluator, max_iter)` —— 循环直到通过。

运行：

```
python3 code/main.py
```

每种模式都会打印自己的执行轨迹。每种模式的代码总量约 10-15 行；而引入一个框架的成本要以数千行计。

## 生产实践

- 大多数任务直接调用 API。
- 只有当模式确实需要持久化状态（LangGraph）、actor 模型并发（AutoGen v0.4）或角色模板化（CrewAI）时才使用框架。
- 当你想要 Claude Code 那种执行框架的形态而又不想重新造轮子时，选 Claude Agent SDK。

## 交付产物

`outputs/skill-workflow-picker.md` 能为给定的任务描述选出合适的模式，包括决策理由，以及当工作流不够用时重构为智能体的路径。

## 练习

1. 实现带置信度阈值的路由。低于阈值 -> 升级给人工处理。对于一线客服场景，这个阈值应该定在哪里？
2. 给 `parallel_vote` 加上超时。当某次调用挂起时会发生什么？投票缺失时该如何聚合？
3. 把 `evaluator_optimizer` 改造成赌博机（bandit）：跨迭代保留前 2 名的输出，这样后期出现的好结果不会被后期出现的坏结果覆盖。
4. 把提示链与路由组合起来：由一个路由器从三条链中选一条。对比单个大提示词方案，测量 token 成本。
5. 挑一个你生产环境里的功能。画出工作流图。数一数步骤。换成智能体真的会更好吗？

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|----------------|------------------------|
| 工作流（Workflow） | “预定义流程” | 由工程师掌控的 LLM 与工具调用图 |
| 智能体（Agent） | “自主 AI” | 由模型掌控的图；动态指挥工具 |
| 增强型 LLM（Augmented LLM） | “带工具的 LLM” | LLM + 搜索 + 工具 + 记忆；原子单元 |
| 提示链（Prompt chaining） | “顺序调用” | 第 N 次调用的输出作为第 N+1 次调用的输入 |
| 路由（Routing） | “分类器分发” | 选择由哪条链/哪个模型处理输入 |
| 并行化（Parallelization） | “扇出” | N 次并发调用；按分段或投票方式聚合 |
| 编排器-工作器（Orchestrator-workers） | “调度型智能体” | 编排器 LLM 动态挑选专家 LLM |
| 评估器-优化器（Evaluator-optimizer） | “提议者 + 评审者” | 迭代直到评估器通过；Self-Refine 的泛化 |

## 延伸阅读

- [Anthropic, Building Effective Agents (Dec 2024)](https://www.anthropic.com/research/building-effective-agents) —— 五种工作流模式
- [Anthropic, Effective context engineering for AI agents](https://www.anthropic.com/engineering/effective-context-engineering-for-ai-agents) —— 与之配套的学科
- [LangGraph overview](https://docs.langchain.com/oss/python/langgraph/overview) —— 有状态图何时值回成本
- [OpenAI Agents SDK](https://openai.github.io/openai-agents-python/) —— 编排器-工作器模式的产品化形态
