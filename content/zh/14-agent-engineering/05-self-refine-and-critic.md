# Self-Refine 与 CRITIC：迭代式输出改进

> Self-Refine（Madaan et al., 2023）让同一个 LLM 在循环中扮演三种角色——生成、反馈、改进，在 7 项任务上平均带来 +20 个绝对百分点的提升。CRITIC（Gou et al., 2023）则强化了反馈环节：把验证步骤交给外部工具来完成。到 2026 年，这一模式已被各大框架内置，Anthropic 称之为「评估器-优化器」（evaluator-optimizer），OpenAI Agents SDK 则以护栏循环（guardrail loop）的形式提供。

**Type:** Build
**Languages:** Python (stdlib)
**Prerequisites:** Phase 14 · 01 (Agent Loop), Phase 14 · 03 (Reflexion)
**Time:** ~60 minutes

## 学习目标

- 说出 Self-Refine 的三个提示词（生成、反馈、改进），并解释为什么历史记录对改进提示词至关重要。
- 解释 CRITIC 的核心洞见：缺少外部依据时，LLM 的自我验证并不可靠。
- 仅用标准库实现一个带历史记录、可选外部验证器的 Self-Refine 循环。
- 将该模式对应到 Anthropic 的「评估器-优化器」工作流和 OpenAI Agents SDK 的输出护栏。

## 问题背景

智能体给出的答案差一点就对了。也许某行代码有语法错误，也许摘要写得太长，也许计划漏掉了一个边界情况。你想要的是：智能体先批评自己的输出，然后修正它。

Self-Refine 证明了单个模型就能做到这一点，不需要训练数据，也不需要强化学习。但有一个隐患：在硬性事实上，LLM 的自我验证能力很差。CRITIC 点明了解法——把验证步骤交给外部工具（搜索引擎、代码解释器、计算器、测试运行器）。

这两篇论文共同定义了 2026 年迭代改进的默认范式：生成、验证（尽可能借助外部工具）、改进、验证器通过后停止。

## 核心概念

### Self-Refine（Madaan et al., NeurIPS 2023）

一个 LLM，三种角色：

```
generate(task)            -> output_0
feedback(task, output_0)  -> critique_0
refine(task, output_0, critique_0, history) -> output_1
feedback(task, output_1)  -> critique_1
refine(task, output_1, critique_1, history) -> output_2
...
stop when feedback says "no issues" or budget exhausted.
```

关键细节：`refine` 能看到完整历史——之前所有的输出和批评——因此不会重复犯错。论文做了消融实验：去掉历史记录后，质量大幅下降。

核心结论：在 7 项任务（数学、代码、缩写、对话）上平均提升 +20 个绝对百分点，GPT-4 也包含在内。无需训练，无需外部工具，只用单个模型。

### CRITIC（Gou et al., arXiv:2305.11738, v4 Feb 2024）

Self-Refine 的弱点在于：反馈环节是 LLM 给自己打分。对于事实性陈述，这并不可靠（幻觉在产生它的模型眼里往往看起来很有说服力）。CRITIC 把 `feedback(task, output)` 替换为 `verify(task, output, tools)`，其中 `tools` 包括：

- 用搜索引擎核查事实性陈述。
- 用代码解释器验证代码正确性。
- 用计算器检查算术运算。
- 领域专用验证器（单元测试、类型检查器、linter）。

验证器基于工具结果生成结构化的批评意见，改进器再以这份批评为条件进行修订。

核心结论：在事实性任务上，CRITIC 优于 Self-Refine，因为它的批评是有依据的。在没有外部验证器的任务上（创意写作、格式整理），CRITIC 退化为 Self-Refine。

### 停止条件

两种常见形式：

1. **验证器通过。** 外部测试返回成功。在可用时优先采用（单元测试、类型检查器、护栏断言）。
2. **无反馈意见。** 模型说「输出没问题」。更省成本但不可靠，需配合最大迭代次数上限使用。

2026 年的默认做法是把两者结合起来：「验证器通过，或者模型说没问题且迭代次数 >= 2，或者迭代次数 >= max_iterations 时停止。」

### 评估器-优化器（Anthropic, 2024）

Anthropic 在 2024 年 12 月的文章中将其列为五大工作流模式之一。两种角色：

- 评估器（Evaluator）：给输出打分并生成批评意见。
- 优化器（Optimizer）：根据批评意见修订输出。

循环直到评估器通过。这就是 Anthropic 语境下的 Self-Refine/CRITIC。Anthropic 补充的关键工程细节是：评估器和优化器的提示词应当有显著差异，否则模型会流于形式、直接盖章通过。

### OpenAI Agents SDK 的输出护栏

OpenAI Agents SDK 以「输出护栏」（output guardrails）的形式提供这一模式。护栏是一个在智能体最终输出上运行的验证器。如果护栏被触发（抛出 `OutputGuardrailTripwireTriggered`），输出会被拒绝，智能体可以重试。护栏既可以调用工具（CRITIC 式），也可以是纯函数（Self-Refine 式）。

### 2026 年的常见陷阱

- **盖章式循环。** 同一个模型用同一种提示词风格做生成和批评，最终会收敛到「我觉得没问题」。应使用结构上不同的提示词，或用一个更小、更便宜的模型来做批评。
- **过度改进。** 每一轮改进都会增加延迟和 token 消耗。预算控制在 1-3 轮；超出后升级到人工审核。
- **在琐碎任务上用 CRITIC。** 如果没有外部验证器，CRITIC 就退化为 Self-Refine；不要为一个空壳验证器白白付出延迟。

## 从零实现

`code/main.py` 在一个玩具任务上实现了 Self-Refine 和 CRITIC：给定主题生成一个简短的要点列表。验证器检查格式（3 个要点，每条不超过 60 个字符）。CRITIC 额外加了一个外部「事实验证器」，用来惩罚已知的幻觉。

组件构成：

- `generate` —— 脚本化的生成器。
- `feedback` —— LLM 风格的自我批评。
- `verify_external` —— CRITIC 风格的有依据验证器。
- `refine` —— 基于历史记录重写输出。
- 停止条件 —— 验证器通过，或最多 4 次迭代。

运行方式：

```
python3 code/main.py
```

对比 Self-Refine 和 CRITIC 的运行结果。CRITIC 抓住了一个 Self-Refine 漏掉的事实错误，因为外部验证器拥有自我批评所不具备的事实依据。

## 生产实践

Anthropic 的评估器-优化器就是这一模式的 Claude 版表述。OpenAI Agents SDK 的输出护栏是 CRITIC 形态的（护栏可以调用工具）。LangGraph 内置了一个 reflection 节点，读起来就像 Self-Refine。Google 的 Gemini 2.5 Computer Use 增加了逐步安全评估器，这是 CRITIC 的一种变体：每个动作在提交前都会被验证。

## 交付产物

`outputs/skill-refine-loop.md` 根据任务形态、验证器可用性和迭代预算来配置一个评估器-优化器循环，输出生成器、评估器/验证器、优化器的提示词，外加一份停止策略。

## 练习

1. 用 max_iterations=1 运行这个玩具示例。CRITIC 还有帮助吗？
2. 把外部验证器换成一个有噪声的版本（随机 30% 假阳性）。循环会出现什么行为？这就是 2026 年大多数护栏方案的真实处境。
3. 实现一个「生成器与批评器用不同模型」的变体：大模型生成，小模型批评。它能胜过同模型方案吗？
4. 阅读 CRITIC 第 3 节（arXiv:2305.11738 v4）。说出三类验证工具，并各举一例。
5. 把 OpenAI Agents SDK 的 `output_guardrails` 对应到 CRITIC 的验证器角色。这个 SDK 哪里做错了，哪里做对了？

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|----------------|------------------------|
| Self-Refine | 「会自我修正的 LLM」 | 单模型内的生成 -> 反馈 -> 改进循环，带历史记录 |
| CRITIC | 「基于工具的验证」 | 用外部验证器（搜索、代码、计算器、测试）替换自我反馈 |
| 评估器-优化器（Evaluator-Optimizer） | 「Anthropic 工作流模式」 | 两种角色——评估器打分，优化器修订——循环直至收敛 |
| 输出护栏（Output guardrail） | 「事后检查」 | OpenAI Agents SDK 的验证器，在智能体产出输出后运行 |
| 验证步骤 | 「批评阶段」 | 起决定性作用的环节：有外部依据，还是自我打分 |
| 改进历史（Refine history） | 「模型已经试过什么」 | 之前的输出 + 批评意见前置到改进提示词中；去掉后质量崩塌 |
| 盖章式循环（Rubber-stamp loop） | 「自我认同失效」 | 同一提示词的批评只会返回「看起来不错」；用结构上不同的提示词来修复 |
| 停止条件 | 「收敛测试」 | 验证器通过，或无反馈意见且达到迭代上限；切勿只用单一条件 |

## 延伸阅读

- [Madaan et al., Self-Refine (arXiv:2303.17651)](https://arxiv.org/abs/2303.17651) —— 奠基论文
- [Gou et al., CRITIC (arXiv:2305.11738)](https://arxiv.org/abs/2305.11738) —— 基于工具的验证
- [Anthropic, Building Effective Agents](https://www.anthropic.com/research/building-effective-agents) —— 评估器-优化器工作流模式
- [OpenAI Agents SDK docs](https://openai.github.io/openai-agents-python/) —— 作为 CRITIC 形态验证器的输出护栏
