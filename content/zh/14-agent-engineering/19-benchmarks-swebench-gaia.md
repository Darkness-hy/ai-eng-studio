# 基准测试：SWE-bench、GAIA、AgentBench

> 2026 年的智能体评估以三个基准为锚点。SWE-bench 测试代码补丁能力，GAIA 测试通用工具使用能力，AgentBench 测试多环境推理能力。你需要了解它们的构成、各自的数据污染情况，以及它们没有衡量到的东西。

**Type:** Learn
**Languages:** Python (stdlib)
**Prerequisites:** Phase 14 · 06 (Tool Use)
**Time:** ~60 minutes

## 学习目标

- 说出 SWE-bench 的测试评估机制（FAIL_TO_PASS），并解释它为什么以单元测试作为通过门槛。
- 解释 SWE-bench Verified（OpenAI，500 个任务）为何存在，以及它剔除了什么。
- 描述 GAIA 的设计理念：对人类简单、对 AI 困难；分为三个难度级别。
- 说出 AgentBench 的八个环境，以及它揭示的开源 LLM 的主要瓶颈。
- 概括 SWE-bench+ 的数据污染（contamination）发现及其影响。

## 问题背景

排行榜只告诉你哪个模型在某一个基准上赢了，但不会告诉你：

- 这个基准是否被污染（解法混入训练数据、测试集泄漏）。
- 这个基准衡量的是否是你关心的能力（代码 vs 网页浏览 vs 通用能力）。
- 评估器是否足够稳健（AST 匹配、状态检查、人工审查）。

在引用任何分数之前，先了解这三个锚点基准及其失效模式。

## 核心概念

### SWE-bench（Jimenez et al., ICLR 2024 oral）

- 来自 12 个流行 Python 仓库的 2,294 个真实 GitHub issue。
- 智能体获得：修复前 commit 的完整代码库 + 自然语言的 issue 描述。
- 智能体产出：一个补丁。
- 评估器：应用补丁，运行该仓库的测试套件。补丁必须让 FAIL_TO_PASS 测试翻转（之前失败、现在通过），同时不破坏 PASS_TO_PASS 测试。

SWE-agent（Yang et al., 2024）发布时达到 12.5%，靠的是强化智能体-计算机接口（文件编辑器命令、模型能理解的搜索语法）。

### SWE-bench Verified

OpenAI，2024 年 8 月。人工筛选的 500 个任务子集。剔除了表述含糊的 issue、不可靠的测试，以及修复方案不明确的任务。它是回答「你的智能体能不能交付真实补丁？」的首选基准。

### 数据污染

- 超过 94% 的 SWE-bench issue 早于大多数模型的训练数据截止时间。
- **SWE-bench+** 发现 32.67% 的成功补丁存在解法泄漏（issue 文本中包含修复方案，模型在描述里直接看到了答案），另有 31.08% 因测试覆盖薄弱而存疑。
- Verified 更干净，但并非完全没有污染。

实际影响：一个在 SWE-bench 上拿到 50% 的模型，在 SWE-bench+ 上可能只有 35%。如果你声称某个 SWE-bench 成绩，请始终同时报告两者。

### GAIA（Mialon et al., 2023 年 11 月）

- 466 个问题；其中 300 个保留用于 huggingface.co/gaia-benchmark 的私有排行榜。
- 设计理念：「在概念上对人类简单（92%），但对 AI 困难（带插件的 GPT-4：15%）。」
- 测试推理、多模态、网页和工具使用能力。
- 分为三个难度级别；Level 3 需要跨模态的长工具链。

GAIA 是用来衡量「通用能力」的基准，不要和面向代码的基准混为一谈。

### AgentBench（Liu et al., ICLR 2024）

- 8 个环境，覆盖代码（Bash、DB、KG）、游戏（Alfworld、LTP）、网页（WebShop、Mind2Web）和开放式生成。
- 多轮交互，每个数据切分约 4k-13k 轮。
- 主要发现：长程推理、决策和指令遵循是开源 LLM 追赶商业模型的瓶颈所在。

### 这些基准没有衡量的东西

- 真实世界的运行成本（token 数、墙钟时间）。
- 对抗条件下的安全行为。
- 在你自己领域上的表现（用你自己的评估集，见第 30 课）。
- 尾部失败（基准看平均值，而生产环境的运维者关心最差的 1%）。

### 基准测试容易出错的地方

- **执着于单一数字。** SWE-bench 上的 50% 提供的信息，远不如 P50/P75/P95 成本 + 步数分布来得多。
- **被污染的成绩声明。** 报告 SWE-bench 成绩却不提 Verified 或 SWE-bench+，是有误导性的。
- **把基准当成开发目标。** 针对基准做优化，会偏离生产环境中的实际价值。

## 从零实现

`code/main.py` 实现了一个类 SWE-bench 的玩具评估框架：

- 合成的修 bug 任务（3 个任务）。
- 一个按脚本运行、负责提出补丁的「智能体」。
- 一个测试执行器，检查 FAIL_TO_PASS（bug 已修复）和 PASS_TO_PASS（没有破坏其他功能）。
- 一个 GAIA 风格的难度分类器，基于问题分解的深度。

运行方式：

```
python3 code/main.py
```

输出展示按任务和按难度统计的解决率，把评估器规则变得具体可见。

## 生产实践

- 代码智能体用 **SWE-bench Verified**。始终报告 Verified 分数。
- 通用智能体用 **GAIA**。使用私有排行榜的数据切分。
- 多环境对比用 **AgentBench**。
- 针对你产品的实际形态，使用**自定义评估集**（第 30 课）。

## 交付产物

`outputs/skill-benchmark-harness.md` 为任意「代码库-任务」组合构建一个 SWE-bench 风格的评估框架，带 FAIL_TO_PASS / PASS_TO_PASS 门控。

## 练习

1. 把玩具评估框架移植到一个真实仓库上（挑一个你自己的）。为已知 bug 写 3 个 FAIL_TO_PASS 测试。
2. 增加一个步数指标。在你的 3 个任务上，每次成功解决需要多少个智能体步骤？
3. 阅读 SWE-bench+ 论文。实现一个解法泄漏检查（把 issue 文本与 diff 做模式匹配）。
4. 从 GAIA 公开切分中下载一个问题。推演一个 GPT-4 级别的智能体会怎么做，它需要哪些工具？
5. 阅读 AgentBench 按环境的拆解结果。哪个环境最接近你的产品场景？在那个环境中「SOTA」是什么水平？

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|----------------|------------------------|
| SWE-bench | 「代码智能体基准」 | 2,294 个 GitHub issue；补丁必须让 FAIL_TO_PASS 测试翻转 |
| SWE-bench Verified | 「干净版 SWE-bench」 | 500 个人工筛选的任务，由 OpenAI 完成 |
| FAIL_TO_PASS | 「修复门槛」 | 之前失败、打补丁后必须通过的测试 |
| PASS_TO_PASS | 「无回归门槛」 | 之前通过、之后也必须继续通过的测试 |
| GAIA | 「通用能力基准」 | 466 个对人类容易、对 AI 困难的多工具问题 |
| AgentBench | 「多环境基准」 | 8 个环境；长程多轮交互 |
| 数据污染（Contamination） | 「训练集泄漏」 | 基准任务出现在模型训练数据中 |
| SWE-bench+ | 「污染审计」 | 在成功的 SWE-bench 补丁中发现 32.67% 存在解法泄漏 |

## 延伸阅读

- [Jimenez et al., SWE-bench (arXiv:2310.06770)](https://arxiv.org/abs/2310.06770) — 原始基准
- [OpenAI, SWE-bench Verified](https://openai.com/index/introducing-swe-bench-verified/) — 人工筛选的子集
- [Mialon et al., GAIA (arXiv:2311.12983)](https://arxiv.org/abs/2311.12983) — 通用能力基准
- [Liu et al., AgentBench (arXiv:2308.03688)](https://arxiv.org/abs/2308.03688) — 多环境基准套件
