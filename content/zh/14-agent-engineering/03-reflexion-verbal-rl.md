# Reflexion：语言强化学习

> 基于梯度的强化学习需要数千次试验和一个 GPU 集群才能修复一种失败模式。Reflexion（Shinn et al., NeurIPS 2023）用自然语言就做到了：每次试验失败后，智能体写下一条反思，存入情景记忆（episodic memory），并让下一次试验以这段记忆为条件。这正是 Letta 的 sleep-time compute、Claude Code 的 CLAUDE.md 经验沉淀、以及 pro-workflow 的 learn-rule 背后的模式。

**Type:** Build
**Languages:** Python (stdlib)
**Prerequisites:** Phase 14 · 01 (Agent Loop), Phase 14 · 02 (ReWOO)
**Time:** ~60 minutes

## 学习目标

- 说出 Reflexion 的三个组件（Actor、Evaluator、Self-Reflector）以及情景记忆的作用。
- 用标准库实现一个 Reflexion 循环，包含二元评估器、反思缓冲区和全新的重试。
- 针对给定任务，在标量、启发式和自评估三种反馈来源之间做出选择。
- 解释为什么语言强化能捕捉到基于梯度的强化学习需要数千次试验才能修复的错误。

## 问题背景

智能体在一个任务上失败了。在标准强化学习里，你需要再跑数千次试验、计算梯度、更新权重。又贵又慢，而且大多数生产环境中的智能体并没有为每次失败准备训练预算。

Reflexion（Shinn et al., arXiv:2303.11366）换了一个问法：如果智能体只是思考一下自己为什么失败，然后把这个想法放进提示词里再试一次，会怎样？不更新权重，不算梯度，只是在试验之间存储自然语言。

结果是：在 ALFWorld 上它击败了 ReAct 和其他未经微调的基线。在 HotpotQA 上它优于 ReAct。在代码生成（HumanEval/MBPP）上它创下了当时的最高水平。这一切都没有执行过一次梯度更新。

## 核心概念

### 三个组件

```
Actor         : generates a trajectory (ReAct-style loop)
Evaluator     : scores the trajectory — binary, heuristic, or self-eval
Self-Reflector: writes a natural-language reflection on the failure
```

外加一个数据结构：

```
Episodic memory: list of prior reflections, prepended to the next trial's prompt
```

一次试验由 Actor 执行，Evaluator 打分。如果分数低，Self-Reflector 会生成一条反思（"我选错了工具，因为我把问题误读成了在问 X，而它实际在问 Y"）。这条反思被存入情景记忆。下一次试验从头开始，但能看到这条反思。

### 三种评估器类型

1. **标量（Scalar）**——外部二元信号。ALFWorld 要么成功要么失败，HumanEval 的测试要么通过要么不通过。最简单，信号最强。
2. **启发式（Heuristic）**——预定义的失败特征。"如果智能体连续两次产生相同的动作，标记为卡住。""如果轨迹超过 50 步，标记为低效。"
3. **自评估（Self-evaluated）**——由 LLM 给自己的轨迹打分。在没有真值（ground truth）时不得不用。信号较弱；适合与工具锚定的验证配合使用（第 05 课——CRITIC）。

2026 年的默认做法是混合使用：有标量信号时用标量，没有时用自评估，启发式作为安全护栏。

### 为什么这个模式具有普适性

Reflexion 与其说是一个新算法，不如说是一个被命名的模式。几乎所有生产环境中的"自愈"智能体都在运行它的某个变体：

- Letta 的 sleep-time compute（第 08 课）：一个独立的智能体反思过往对话并写入记忆块。
- Claude Code 的 `CLAUDE.md` / "save memory" 模式：把反思沉淀为经验，前置到未来的会话中。
- pro-workflow 的 `/learn-rule` 命令：把纠正意见沉淀为显式规则。
- LangGraph 的反思节点：一个对输出打分、必要时路由到改进环节的节点。

它们都源自同一个洞察：自然语言是一种足够丰富的媒介，能够在多次运行之间承载"我从失败中学到了什么"。

### 什么时候有效，什么时候无效

Reflexion 在以下情况有效：

- 存在清晰的失败信号（测试失败、工具报错、答案错误）。
- 任务类别是可复现的（同类问题可以被再次提出）。
- 反思有改进轨迹的空间（有足够的动作预算）。

Reflexion 在以下情况帮不上忙：

- 智能体第一次尝试就成功了。
- 失败来自外部（网络断了、工具坏了）——针对"网络断了"的反思对未来的运行没有帮助。
- 反思变成了迷信——把一次偶发的不稳定运行写成了叙事并存了下来。

2026 年的常见坑：记忆腐烂（memory rot）。反思不断累积，其中一些已经过时或本身就是错的；随着情景缓冲区增长，重跑会越来越慢。缓解手段：定期压缩（第 06 课）、给反思设置 TTL，或用一个独立的 sleep-time 清理智能体（Letta）。

```figure
react-trace
```

## 从零实现

`code/main.py` 在一个玩具谜题上实现了 Reflexion：生成一个三元素列表，使其和等于目标值。Actor 给出候选列表；Evaluator 检查总和；Self-Reflector 写一行关于哪里出错的诊断。这条反思进入情景记忆，供下一次试验使用。

组件：

- `Actor`——一个脚本化策略，看到反思后会改进。
- `Evaluator.binary()`——按目标和判定通过/失败。
- `SelfReflector`——生成一行失败诊断。
- `EpisodicMemory`——一个带 TTL 语义的有界列表。

运行它：

```
python3 code/main.py
```

运行轨迹显示三次试验。第 1 次试验失败，存入一条反思；第 2 次试验看到反思后有所改进但仍然失败；第 3 次试验成功。对比基线运行（无反思）——它一直停留在第 1 次试验的答案上。

## 生产实践

LangGraph 把反思作为一种节点模式内置发布。Claude Code 的 `/memory` 命令和 pro-workflow 的 `/learn-rule` 把情景缓冲区外化为一个 markdown 文件。Letta 的 sleep-time compute 在空闲时间运行 Self-Reflector，让主智能体保持低延迟。OpenAI Agents SDK 没有直接提供 Reflexion；你需要自己搭建：用一个自定义 Guardrail 按分数拒绝轨迹，再用一个跨运行持久化的记忆 `Session`。

## 交付产物

`outputs/skill-reflexion-buffer.md` 创建并维护一个情景缓冲区，支持反思采集、TTL 和去重。给定一个任务类别和一次失败，它会生成一条对下一次试验真正有帮助的反思（而不是一句泛泛的"下次小心点"）。

## 练习

1. 把二元评估器换成返回距离度量（离目标差多少）的标量评估器。收敛会更快吗？
2. 给反思加上 10 次试验的 TTL。超过这个期限后，旧反思是有害还是有益？
3. 实现启发式评估器：如果同一动作重复出现，标记该次试验为卡住。它与 Self-Reflector 会如何相互作用？
4. 用一个无视反思的对抗性 Actor 运行 Reflexion。要迫使 Actor 注意到反思，最少需要做哪些反思提示词工程？
5. 阅读 Reflexion 论文中关于 ALFWorld 的第 4 节。从概念上复现 130% 的成功率提升：相对于普通 ReAct，关键差异是什么？

## 关键术语

| 术语 | 人们常说的 | 实际含义 |
|------|----------------|------------------------|
| Reflexion | "自我纠错" | Shinn et al. 2023——Actor、Evaluator、Self-Reflector 加情景记忆 |
| 语言强化（Verbal reinforcement） | "无梯度学习" | 前置到下一次试验提示词中的自然语言反思 |
| 情景记忆（Episodic memory） | "按任务的反思" | 针对一个任务类别的历史反思有界缓冲区 |
| 标量评估器 | "二元成功信号" | 来自真值的通过/失败或数值分数 |
| 启发式评估器 | "基于模式的检测器" | 预定义的失败特征（如卡住循环、步数过多） |
| 自评估器 | "对自己轨迹做 LLM-as-judge" | 没有真值时的弱信号兜底——应与工具锚定的验证配合使用 |
| 记忆腐烂（Memory rot） | "过期的反思" | 情景缓冲区被过时条目填满；用压缩/TTL 修复 |
| 睡眠时反思（Sleep-time reflection） | "异步自我反思" | 把 Self-Reflector 移出热路径运行，让主智能体保持快速 |

## 延伸阅读

- [Shinn et al., Reflexion: Language Agents with Verbal Reinforcement Learning (arXiv:2303.11366)](https://arxiv.org/abs/2303.11366)——原始论文
- [Letta, Sleep-time Compute](https://www.letta.com/blog/sleep-time-compute)——生产环境中的异步反思
- [Anthropic, Effective context engineering for AI agents](https://www.anthropic.com/engineering/effective-context-engineering-for-ai-agents)——把情景缓冲区当作上下文的一部分来管理
- [LangGraph overview](https://docs.langchain.com/oss/python/langgraph/overview)——反思节点模式
