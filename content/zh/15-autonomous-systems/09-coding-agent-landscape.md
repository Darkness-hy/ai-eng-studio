# 自主编码智能体全景（2026）

> SWE-bench Verified 在不到三年里从 4% 涨到 80.9%。同一个 Claude Sonnet 4.5，在 SWE-agent v1 上得 43.2%，在 Cline 自主模式下得 59.8%——模型外围的脚手架（scaffolding）如今和模型本身同等重要。OpenHands（前身为 OpenDevin）是最活跃的 MIT 许可平台，它的 CodeAct 循环直接在沙箱里执行 Python 动作，而不是 JSON 工具调用。这些亮眼数字背后藏着一个方法论问题：SWE-bench Verified 的 500 个任务中有 161 个只需改 1–2 行代码，而在 SWE-bench Pro（10 行以上改动的任务）上，同样的前沿模型只有 23–59%。

**Type:** Learn
**Languages:** Python (stdlib, CodeAct vs JSON tool-call comparison)
**Prerequisites:** Phase 14 · 07 (Tool use), Phase 15 · 01 (Long-horizon agents)
**Time:** ~45 minutes

## 问题背景

"哪个编码智能体最好"是个错误的问题。正确的问题是：在与我的工作相匹配的任务分布上，使用我将在生产环境中运行的脚手架，我能获得怎样的端到端可靠性？

2022 到 2026 年间，这个领域认识到：脚手架——检索层、规划器、沙箱、编辑-验证循环、反馈格式——是承重结构。Claude Sonnet 4.5 在 SWE-agent v1 上的 SWE-bench Verified 得分是 43.2%；同一个模型放进 Cline 的自主脚手架里得 59.8%。16.6 个绝对百分点的差距，权重完全相同。基础模型只是一个组件；循环才是产品。

与之相伴的问题是，基准饱和会掩盖回退。SWE-bench Verified 已接近饱和，而简单任务的长尾（500 个任务中有 161 个只需改 ≤2 行）抬高了榜首分数。真实世界的质量更适合用 SWE-bench Pro（10 行以上的改动）这样的分布来衡量——在那里，同样的领先者仍然停留在 23–59%。

## 核心概念

### 一段话讲清 SWE-bench

SWE-bench（Jimenez 等人）取真实的 GitHub issue 及其标准答案补丁，要求智能体生成一个能让测试套件通过的补丁。SWE-bench Verified（OpenAI，2024）是经人工筛选的 500 任务子集，剔除了模糊和损坏的任务。SWE-bench Pro 是更难的后继者——需要 10 行以上改动的任务，目前的前沿智能体停留在 23–59%。

### 2022 → 2026 这条曲线到底说明了什么

- **2022**：研究模型在原始 SWE-bench 上约 4%。
- **2024**：GPT-4 加上 Devin 式脚手架约 14%；SWE-agent 约 12%。
- **2025**：Claude 3.5/3.7 Sonnet 在 Aider 和 SWE-agent 中推进到 40–55% 区间。
- **2026**：Claude Sonnet 4.5 与前沿竞争者在 SWE-bench Verified 上达到 70–80% 以上。Epoch AI 的榜单实时跟踪这一进展。

这条斜率来自三个相互叠加的因素：更好的基础模型、更好的脚手架（CodeAct、反思、验证器循环）、更好的基准（Verified 剔除了噪声）。

### CodeAct vs JSON 工具调用

OpenHands（All-Hands-AI，arXiv:2407.16741，前身为 OpenDevin）押注了一个特定的架构方案：不让模型输出由宿主解码执行的 JSON 工具调用，而是让模型输出 Python 代码，由一个 Jupyter 风格的内核在沙箱中运行。智能体可以在一个动作内遍历文件、串联工具、捕获自己抛出的异常。

权衡如下：

- **JSON 工具调用**：每个动作占一轮；易于审计；组合能力有限；默认安全，因为每次调用都经过显式验证器。
- **CodeAct**：一个动作可以是一整段程序；可组合；需要加固的沙箱（OpenHands 使用 Docker 隔离）；失败模式包括沙箱运行时允许的任何行为。

两种架构都已在生产环境中使用。CodeAct 在开放平台中占主导（OpenHands、smolagents）。JSON 工具调用在托管服务中仍占主导（Anthropic Managed Agents、OpenAI Assistants），因为执行器由提供方控制。

### 2026 年格局中的各个脚手架

| 脚手架 | 许可证 | 执行模型 | 显著特性 |
|---|---|---|---|
| OpenHands (OpenDevin) | MIT | Docker 中的 CodeAct | 最活跃的开放平台；事件流可重放 |
| SWE-agent | MIT | 智能体-计算机接口（ACI） | 首个端到端 SWE-bench 脚手架 |
| Aider | Apache-2 | 在本地仓库中以 diff 方式编辑 | 极简脚手架，回归稳定性强 |
| Cline | Apache-2 | 带工具策略的 VS Code 智能体 | 在 Sonnet 4.5 上得分最高的开放脚手架 |
| Devin (Cognition) | 专有 | 托管虚拟机 + 规划器 | 首创"AI 软件工程师"产品品类 |
| Claude Code | 专有 | 权限模式 + 例程 | 第 10 课详细讲解其智能体循环 |

### 为什么脚手架起决定作用

一次编码运行是一条长程轨迹（第 1 课）。可靠性在多个步骤间复合累积。脚手架在三个地方能挣到分数：

1. **检索**：找到该读的文件是无声的瓶颈。SWE-agent 的 ACI、OpenHands 的文件索引、Aider 的 repo-map 都在攻克这一点。
2. **验证器循环**：运行测试、阅读堆栈跟踪、再次尝试，在 SWE-bench 上能带来 10 个百分点以上的差距。
3. **故障隔离**：出错时回滚的沙箱可以阻止损害复合扩散。同一个模型，有无验证器循环，看起来像两个不同的产品。

### 基准饱和与真实分布

OpenHands 的作者和 Epoch AI 都指出，SWE-bench Verified 有一条简单任务长尾：500 个任务中有 161 个只需改 1–2 行。高分部分由这条长尾驱动。SWE-bench Pro 限定为 10 行以上的改动后，即便是前沿系统，得分也回落到 23–59% 区间。你的生产任务分布几乎肯定更接近 Pro 而不是 Verified。

对选择智能体的启示：在你自己的 bug 积压中跑一个类 Pro 的子集。真正重要的分数，是在能代表你实际交付内容的任务上的分数。

## 生产实践

`code/main.py` 在一个固定的迷你任务分布上比较两个玩具级智能体脚手架：

1. **JSON 工具调用**脚手架，每轮执行一个动作。
2. **CodeAct** 脚手架，每个动作可以输出一小段 Python 代码。

两者都使用桩"模型"（确定性规则），以便把脚手架的影响与模型质量隔离开来。输出显示，CodeAct 脚手架以更大的单动作爆炸半径为代价，用更少的轮次解决了更多任务。

## 交付产物

`outputs/skill-scaffold-audit.md` 帮你在采用某个编码智能体脚手架之前进行审计：检索质量、验证器是否存在、沙箱隔离程度，以及基准与任务分布的匹配度。

## 练习

1. 运行 `code/main.py`。在同一任务集上，两个脚手架各用了多少轮？各自的单动作爆炸半径是多少？

2. 阅读 OpenHands 论文（arXiv:2407.16741）。论文认为 CodeAct 在复杂任务上胜过 JSON 工具调用。找出论文承认的一种失败模式，并用一句话说明这种失败模式在生产环境中何时会占主导。

3. 从你的 bug 积压中挑选一个需要跨两个文件改动 10 行以上的任务。估计前沿模型在 (a) JSON 工具调用和 (b) CodeAct 下的端到端成功概率，并论证两者之间的差距。

4. SWE-bench Verified 有 161 个单文件、1–2 行改动的任务。构造一个排除这些任务的得分。榜单排名会如何洗牌？

5. 阅读 "Introducing SWE-bench Verified"（OpenAI）。解释剔除模糊任务所用的具体方法论，并指出该筛选会遗漏的一类任务。

## 关键术语

| 术语 | 通俗说法 | 实际含义 |
|---|---|---|
| SWE-bench | "编码基准" | 真实 GitHub issue，配有标准答案补丁和测试套件 |
| SWE-bench Verified | "清洗后的子集" | 500 个人工筛选任务，仍存在简单任务长尾 |
| SWE-bench Pro | "更难的子集" | 10 行以上改动；前沿模型停留在 23–59% |
| CodeAct | "代码即动作" | 智能体输出 Python；Jupyter 风格内核在沙箱中执行 |
| JSON 工具调用 | "函数调用" | 每个动作是一个结构化 JSON 载荷，执行前经过验证 |
| 脚手架（Scaffold） | "智能体框架" | 围绕基础模型的检索 + 规划器 + 执行器 + 验证器循环 |
| ACI（智能体-计算机接口） | "SWE-agent 的格式" | 为 LLM 使用习惯而非人类 shell 设计的命令集 |
| 验证器循环 | "测试再重试" | 运行测试、阅读输出、修订补丁；模型之外最大的可靠性来源 |

## 延伸阅读

- [Jimenez et al. — SWE-bench](https://www.swebench.com/) — 原始基准及其方法论。
- [OpenAI — Introducing SWE-bench Verified](https://openai.com/index/introducing-swe-bench-verified/) — 这个人工筛选子集是如何构建的。
- [Wang et al. — OpenHands: An Open Platform for AI Software Developers](https://arxiv.org/abs/2407.16741) — CodeAct 架构与事件流设计。
- [Epoch AI — SWE-bench leaderboard](https://epoch.ai/benchmarks) — 实时跟踪的得分。
- [Anthropic — Measuring agent autonomy](https://www.anthropic.com/research/measuring-agent-autonomy) — 关于长程编码智能体可靠性的分析框架。
