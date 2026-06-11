# 心智理论与涌现式协调

> Li 等人（arXiv:2310.10701）的研究表明，LLM 智能体在合作型文字游戏中会表现出**涌现的高阶心智理论**（Theory of Mind, ToM）——即推理另一个智能体对第三个智能体信念的看法——但由于上下文管理问题和幻觉，它们在长程规划上会失败。Riedl（arXiv:2510.05174）测量了整个智能体群体的高阶协同效应，发现**只有**在 ToM 提示词条件下才会出现与身份绑定的角色分化和目标导向的互补性；能力较弱的 LLM 只表现出虚假的涌现。也就是说，协调的涌现取决于提示词、依赖于模型，并非免费的午餐。本课实现一个最小化的 ToM 感知智能体，分别在有无 ToM 提示词的条件下运行同一个合作任务，并按照 Riedl 2025 的实验协议测量协调效果的差异。

**Type:** Learn + Build
**Languages:** Python (stdlib)
**Prerequisites:** Phase 16 · 07 (Society of Mind and Debate), Phase 16 · 17 (Generative Agents)
**Time:** ~75 minutes

## 问题背景

多智能体协调常常看起来很神奇：智能体分工合作、相互预判、避免重复劳动。但通常这种"涌现"只是提示词工程的产物——有人在提示词里告诉智能体要"协调配合"。删掉这句提示词，协调也就随之消失。

Riedl 2025 年的发现更为严格：在受控条件下，只有当智能体被提示去推理**其他智能体的心智**（ToM）时，协调才会涌现。没有 ToM 提示词时，即使是强模型表现出的协调模式也经不起统计控制的检验。这对生产环境很重要：很多团队上线的"多智能体协调"功能其实依赖提示词、十分脆弱。

本课将 ToM 视为一种具体能力（对信念的信念进行推理），构建一个最小化的 ToM 感知智能体，并测量真实的协调与提示词包装出来的协调分别是什么样子。

## 核心概念

### ToM 是什么意思

发展心理学告诉我们：3 岁的孩子认为所有人的内心世界都和自己一样；5 岁的孩子能理解他人持有不同的信念；7 岁的孩子能对"信念的信念"进行推理（"她认为我认为球在杯子下面"）。这分别对应零阶、一阶和二阶 ToM。

对于 LLM 智能体，各阶 ToM 的对应关系如下：

- **零阶（Zeroth-order）：** 不对他人建模。智能体只根据自己的观察行动。
- **一阶（First-order）：** 智能体对每个其他智能体的信念建模。"Alice 相信 X。"
- **二阶（Second-order）：** 智能体对递归信念建模。"Alice 相信 Bob 相信 X。"

Li 等人在 2023 年发现，一阶和二阶 ToM 会在合作游戏中的 LLM 智能体身上涌现，但在长任务周期和不可靠通信的条件下会退化。

### Sally-Anne 测试简介

这是 1985 年的一个错误信念测试：Sally 把一颗弹珠放进篮子 A，然后离开。Anne 把弹珠移到篮子 B。Sally 回来后会去哪里找弹珠？具备一阶 ToM 的孩子会回答篮子 A（Sally 的信念与现实不一致）；不具备的孩子会回答篮子 B。

GPT-4 时代的 LLM 在题目表述直白时能通过 Sally-Anne 类测试。但当叙述很长、场景多次变化或问题以间接方式提出时，它们会失败。这就是 2026 年生产环境中 LLM 的 ToM 能力的实际状态。

### Riedl 的协调测量方法

Riedl（arXiv:2510.05174）构建了一个群体规模的测试：N 个智能体、一个合作目标、可变的提示词条件。测量指标包括：

1. **身份绑定的角色分化（Identity-linked differentiation）。** 智能体是否随时间形成稳定的角色区分？
2. **目标导向的互补性（Goal-directed complementarity）。** 智能体的行动是否相互补充（承担不同子任务）而非重复？
3. **高阶协同（Higher-order synergy）。** 一种统计度量，衡量群体是否达成了任何子集都无法达成的成果。

结果：只有在 ToM 提示词条件下，这三项指标才全部产生高于基线的信号。没有 ToM 提示词时，中等能力模型的各项指标徘徊在随机水平附近。大模型在没有显式 ToM 提示词时也表现出一定的协调，但效应小于有显式提示词的情况。

### 协调的幻象

在缺乏统计控制时，演示中的"涌现式协调"往往源于：

- 把协调直接写进提示词的提示词工程（系统提示词里写着"一起合作"）。
- 观察者偏差（我们看到的是自己期望看到的模式）。
- 对成功运行结果的事后筛选。

那些宣传"涌现式协调"却拿不出可测量信号的生产系统，应当被视为营销话术。先测量，再下结论。

### 一个最小化的 ToM 感知智能体

结构如下：

```
agent state:
  own_beliefs:    {facts the agent believes}
  other_models:   {other_agent_id -> {beliefs_the_agent_attributes_to_them}}
  actions_last_N: [history of others' actions]

observation update:
  - update own_beliefs from direct observation
  - update other_models[agent_id] from their action + prior beliefs

action selection:
  - enumerate candidate actions
  - for each, predict what each other agent will do next given their modeled beliefs
  - pick action that maximizes joint outcome under those predictions
```

`other_models` 属性就是 ToM 状态。一阶 ToM 只保留一层。二阶则增加 `other_models[i][other_models_of_j]`——我认为智能体 i 认为智能体 j 相信什么。

### 为什么长任务周期会出问题

Li 等人的论文记录了这一点：上下文限制导致智能体忘记某条信念属于谁。幻觉会向其他智能体的模型中添加错误信念。两者都会产生"我以为他以为 X"这类错误，并随时间不断累积。

论文及 2024-2026 年的后续工作中记录的缓解措施包括：

- **在提示词中显式维护 ToM 状态。** 采用结构化格式：`{agent_id: belief_list}`。强制检索时保持"身份-信念"的绑定关系。
- **缩短推理链。** 每轮减少 ToM 更新次数，可以减少幻觉的累积。
- **外部 ToM 存储。** 在 LLM 上下文之外维护模型；每轮只注入相关部分。

### ToM 在生产环境中的失效场景

- **对抗性场景。** ToM 能力强的智能体更容易被操纵（对手可以建模"它对你的建模"，然后加以利用）。
- **异构团队。** 当各智能体使用不同模型时，对某一个对手有效的 ToM 模型无法泛化到其他对手。
- **依赖客观事实的任务。** ToM 关注的是信念；如果正确性取决于事实，ToM 反而可能是干扰。

### 真正可以测量的协调

判断一个团队的协调是真实的而非提示词包装的，有三个实用信号：

1. **随时间保持的互补性。** 在多轮任务中，智能体的行动是否覆盖了互不重叠的子任务？
2. **预判。** 智能体 A 在第 T+1 轮的行动，是否基于对 B 在第 T+2 轮行动的预测，而且该预测最终被证明是正确的？
3. **纠错。** 当 A 在第 T 轮误读了 B 的信念时，A 是否能在第 T+2 轮之前完成纠正？

这些信号在带日志的多智能体系统中都是可测量的。它们才是"协调"叙事中真正有实质内容的版本。

## 从零实现

`code/main.py` 实现了：

- `ToMAgent`——跟踪自身信念以及对每个其他智能体的信念模型。
- 一个合作任务：三个智能体需要从三个盒子中收集三枚代币；每个盒子只能容纳一枚代币。智能体之间不能通信，只能从彼此的行动中推断意图。
- 两种配置：`zeroth_order`（无 ToM）和 `first_order`（带一层信念模型的 ToM）。
- 在 200 次随机试验上的测量：完成率、重复率（两个智能体瞄准同一个盒子）、平均完成轮数。

运行：

```
python3 code/main.py
```

预期输出：零阶智能体的重复劳动率约为 35%，在 10 轮内完成约 60% 的试验。一阶 ToM 智能体的重复率约为 5%，完成率约为 95%。这个差值就是可测量的协调效应。

## 生产实践

`outputs/skill-tom-auditor.md` 是一个用于审计多智能体系统"涌现式协调"声明的技能。它检查是否存在提示词包装、相对于对照组的统计显著性，以及实测的互补性。

## 交付产物

协调声明检查清单：

- **对照条件。** 准备一个去掉协调提示词的系统版本。两者都要测量。
- **统计检验。** 在你的指标上，系统与对照组之间的差异是否在 `p < 0.05` 水平上显著？
- **互补性度量。** 测量随时间的行动不重叠程度，而不只看最终是否成功。
- **失败案例日志。** 当智能体协调失败时，ToM 状态是什么样子？
- **模型能力披露。** 如果效应在更小的模型上消失，要如实说明。

## 练习

1. 运行 `code/main.py`。确认一阶 ToM 把重复率降低了约 7 倍。当扩展到 5 个智能体和 5 个盒子时，这个差距还存在吗？
2. 实现二阶 ToM（智能体 A 建模 B 对 C 的看法）。它比一阶有提升吗？在哪些任务上有提升？
3. 向 ToM 状态注入**幻觉**：每轮随机翻转一条信念。这会让一阶 ToM 的性能下降多少？
4. 阅读 Li 等人的论文（arXiv:2310.10701）。复现"长程退化"这一发现：当轮数从 10 增加到 30 时，你的一阶 ToM 性能如何变化？
5. 阅读 Riedl 2025（arXiv:2510.05174）。在你的模拟日志上实现高阶协同统计量。在没有 ToM 提示词的条件下，该效应还存在吗？

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|----------------|------------------------|
| 心智理论（Theory of Mind） | "理解他人的心智" | 对另一个智能体的信念进行建模的能力。按阶数分级（0 阶、1 阶、2 阶及以上）。 |
| Sally-Anne 测试 | "错误信念测试" | 1985 年的发展心理学实验；LLM 能通过直白版本，复杂版本则会失败。 |
| 一阶 ToM | "A 相信 X" | 对另一个体关于事实的信念进行建模。 |
| 二阶 ToM | "A 相信 B 相信 X" | 再深一层的递归建模。 |
| 身份绑定的角色分化 | "随时间保持稳定的角色" | Riedl 的指标：角色持续存在，而非随机分布。 |
| 目标导向的互补性 | "互不重叠的行动" | 智能体瞄准不同的子任务，而非同一个。 |
| 高阶协同 | "群体超越任何子集" | Riedl 用于衡量真实协调的统计度量。 |
| 协调的幻象 | "看起来很协调" | 由提示词包装出来的协调表象，缺乏可测量的信号。 |

## 延伸阅读

- [Li et al. — Theory of Mind for Multi-Agent Collaboration via Large Language Models](https://arxiv.org/abs/2310.10701) — 合作游戏中涌现的 ToM；长程失效模式
- [Riedl — Emergent Coordination in Multi-Agent Language Models](https://arxiv.org/abs/2510.05174) — 群体规模的测量；ToM 提示词是起决定作用的条件
- [Premack & Woodruff — Does the chimpanzee have a theory of mind?](https://www.cambridge.org/core/journals/behavioral-and-brain-sciences/article/does-the-chimpanzee-have-a-theory-of-mind/1E96B02CD9850E69AF20F81FA7EB3595) — 1978 年 ToM 概念的起源
- [Baron-Cohen, Leslie, Frith — Does the autistic child have a theory of mind?](https://www.cambridge.org/core/journals/behavioral-and-brain-sciences/article/does-the-autistic-child-have-a-theory-of-mind/) — Sally-Anne 论文（1985）
