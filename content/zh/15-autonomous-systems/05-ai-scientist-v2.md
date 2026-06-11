# AI Scientist v2 —— Workshop 级别的自主科研

> Sakana 的 AI Scientist v2（Yamada et al., arXiv:2504.08066）跑通了完整的科研闭环：提出假设、写代码、做实验、出图、撰写论文、投稿。它是第一个让生成的论文通过 ICLR 2025 workshop 同行评审的系统。独立评估（Beel et al.）发现 42% 的实验因代码错误而失败，文献综述还经常把已有的成熟概念误标为新颖。Sakana 自己的文档也警告该代码库会执行 LLM 编写的代码，并建议使用 Docker 隔离。这幅图景的两面，正是本课的重点。

**Type:** Learn
**Languages:** Python (stdlib, research-loop state-machine toy)
**Prerequisites:** Phase 15 · 03 (AlphaEvolve), Phase 15 · 04 (DGM)
**Time:** ~60 minutes

## 问题背景

科研是一项开放式任务。不同于 AlphaEvolve 的算法搜索或 DGM 受基准约束的自我修改，科研成果没有可机器验证的正确性标准。论文由评审人评判，而不是单元测试。这让闭环更难实现——可一旦闭环成功，价值也更大，因为科研正是复利式进步的所在。

AI Scientist v1（Sakana, 2024）靠人工编写的模板实现了闭环。LLM 在固定的脚手架内填充实验。AI Scientist v2（Yamada et al., 2025）通过智能体树搜索（agentic tree search）加上视觉语言模型批评循环，去掉了对模板的依赖。系统会生成想法、实现实验、产出图表、撰写论文，并根据评审反馈迭代。

同行评审的判决：一篇 v2 生成的论文在 ICLR 2025 workshop 被接收（并已披露其来源）。独立评估的判决：这个系统离可靠还差得很远。两者都是事实。

## 核心概念

### 架构

1. **想法生成。** LLM 在给定主题和已有文献的条件下提出研究想法。v1 使用模板；v2 在假设空间上做智能体搜索。
2. **新颖性检查。** 一个文献检索步骤检查该想法是否已被发表。Beel et al. 的评估正是在这一步发现了误标——成熟方法经常被判定为新颖。
3. **实验计划。** 智能体起草实验方案并编写代码。
4. **执行。** 代码在沙箱中运行。失败结果被反馈进重试循环。按 Beel et al. 的测量，42% 的实验在这一阶段因代码错误失败。
5. **图表生成。** 视觉语言模型读取生成的图表并为提升清晰度而重写。这是 v2 的关键技术新增项。
6. **撰写论文。** LLM 起草论文，与内部评审人迭代打磨。
7. **可选：投稿。** 论文被提交到某个会议或 workshop。

### Workshop 论文被接收意味着什么

一篇 v2 生成的论文通过了 ICLR 2025 workshop 的同行评审。作者向程序委员会披露了论文的来源。这次接收是一个数据点，并不意味着可以宣称这个系统"会做科研"。

重要的背景：workshop 论文的门槛低于主会论文。同行评审本身是有噪声的；任何时候都有一小部分投稿能被接收。一次成功是概念验证，不是可靠性声明。Nature 2026 那篇论文记录了端到端的闭环，但论文本身有人类研究者共同署名；这不等于"系统写出了一篇 Nature 论文"。

### 独立评估发现了什么

Beel et al.（arXiv:2502.14297）做了一次外部评估。主要发现：

- **实验失败。** 42% 的实验因代码错误失败（错误的导入、张量形状不匹配、未定义变量）。重试循环能捕获一部分，但不是全部。
- **新颖性误标。** 文献检索步骤经常把成熟概念标记为新颖。这相当于科研领域的幻觉。
- **呈现质量与实质的落差。** 视觉语言模型的图表批评产出了出版级别的可视化效果，反而掩盖了底层实验的薄弱。

最后一条发现对本阶段最重要。一个能产出令人信服的输出却没有做出令人信服的研究的系统，比一个会明显失败的系统更危险，而不是更安全。评估必须触及底层的论断本身，不能止步于图表。

### 沙箱逃逸的隐患

Sakana 自己的仓库 README 警告道：

> Due to the nature of this software, which executes LLM-generated code, we cannot guarantee safety. There are risks of dangerous packages, uncontrolled web access, and spawning of unintended processes. Use at your own risk and consider Docker isolation.

这就是在无法验证的领域里自主性的实际运作形态。LLM 写代码；代码运行；代码可以做进程权限允许的任何事。如果没有一个对文件系统、网络和进程操作做硬性限制的沙箱，任何自主导向的科研智能体都可能外泄数据、烧光算力，或者改写自身。

AlphaEvolve 的沙箱问题更简单，因为它的评估器很严密。AI Scientist v2 的循环则以开放式目标运行开放式代码。这就是它需要更强隔离（Docker 是底线；首选 seccomp / gVisor），并且每一份投稿离开系统前都要人工审查的原因。

### v2 在前沿系统栈中的位置

| 系统 | 目标 | 输出类型 | 评估器 | 已知失败模式 |
|---|---|---|---|---|
| AlphaEvolve | 算法 | 代码 | 单元测试 + 基准 | 受限于评估器的严密程度 |
| DGM | 智能体脚手架 | 代码 | SWE-bench | 奖励黑客 |
| AI Scientist v2 | 研究论文 | 文本 + 代码 + 图表 | 同行评审（弱） | 实验失败、新颖性误标、包装掩盖缺陷 |

三者之中，v2 的自动评估器最弱、输出面最宽、通往公开产物的路径最短。安全保障主要靠运营层面的控制手段（沙箱、人工审查、来源披露）在支撑。

## 生产实践

`code/main.py` 把 v2 的循环模拟成一个状态机：想法 → 新颖性检查 → 实验 → 图表 → 撰写 → 评审 → 接收或继续迭代。每个状态都有一个可配置的失败概率，取值来自 Beel et al. 的发现。运行模拟器 N 轮并统计：

- 多少个想法最终走到投稿。
- 多少份投稿存在被包装精美的论文掩盖的关键实验缺陷。
- 重试预算如何在质量与产量之间权衡。

## 交付产物

`outputs/skill-ai-scientist-sandbox-review.md` 是一份双关卡审查清单，用于在科研循环智能体的任何产物离开沙箱之前进行把关。

## 练习

1. 用默认参数运行 `code/main.py`。多大比例的循环产出"干净"的论文？多大比例的论文带有被图表批评环节包装掉的实验失败缺陷？

2. 默认参数已经采用了 Beel et al. 的 42% / 25%。先用 `--experiment-failure 0.20 --novelty-mislabel 0.10` 重跑，再用 `--experiment-failure 0.60 --novelty-mislabel 0.40` 重跑。两次运行之间，"包装精美但有缺陷"的论文占比如何变化？

3. 阅读 Sakana 的 AI Scientist v2 仓库 README 中关于沙箱要求的部分。针对一次多日的自主运行，说出你会在 Docker 之外额外施加的两条限制。

4. 阅读 Beel et al. 第 4 节关于呈现质量落差的内容。设计一个额外的评估器，能识别出看起来精美但实验有缺陷的论文。

5. 提出一个比"博士逐篇阅读每份论文"更具扩展性的科研智能体产物人工审查协议。找出其中的瓶颈并围绕它做设计。

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|---|---|---|
| AI Scientist v1 | "Sakana 的模板化科研智能体" | 在固定脚手架内填充实验 |
| AI Scientist v2 | "无模板科研智能体" | 智能体树搜索加 VLM 图表批评 |
| 智能体树搜索（Agentic tree search） | "分支式科研智能体" | 并行扩展多个实验计划；由内部批评者剪枝 |
| 视觉语言批评（Vision-language critique） | "用 VLM 打磨图表" | 多模态模型读取图表并为提升清晰度而重写 |
| 文献检索（Literature retrieval） | "新颖性检查" | 搜索已有工作以确认想法新颖性——已被记录会误标 |
| 包装掩盖（Polish masking） | "论文漂亮，研究垮掉" | 呈现质量超过实验质量；掩盖了缺陷 |
| 沙箱逃逸（Sandbox escape） | "LLM 代码越狱" | 智能体执行的代码做了循环设计者意料之外的事 |

## 延伸阅读

- [Yamada et al. (2025). The AI Scientist-v2](https://arxiv.org/abs/2504.08066) —— 原论文。
- [Sakana blog on the Nature 2026 publication](https://sakana.ai/ai-scientist-nature/) —— 厂商总结，附同行评审背景。
- [Beel et al. (2025). Independent evaluation of The AI Scientist](https://arxiv.org/abs/2502.14297) —— 外部评估数据。
- [Sakana AI Scientist v1 paper](https://arxiv.org/abs/2408.06292) —— 模板化的前代系统。
- [Anthropic — Measuring AI agent autonomy](https://www.anthropic.com/research/measuring-agent-autonomy) —— 对开放式科研智能体的更宏观讨论。
