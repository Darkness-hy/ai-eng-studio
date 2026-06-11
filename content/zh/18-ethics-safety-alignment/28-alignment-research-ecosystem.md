# 对齐研究生态 —— MATS、Redwood、Apollo、METR

> 五家机构构成了 2026 年实验室之外的对齐研究层。MATS（ML Alignment & Theory Scholars）：自 2021 年底以来培养 527+ 名研究者，发表 180+ 篇论文，引用量超 1 万次，h-index 47；2024 年夏季届注册为 501(c)(3) 非营利组织，约 90 名学者与 40 名导师；2025 年之前的校友中 80% 从事安全/安保工作，其中 200+ 人任职于 Anthropic、DeepMind、OpenAI、UK AISI、RAND、Redwood、METR、Apollo。Redwood Research：由 Buck Shlegeris 创立的应用对齐实验室；提出了 AI 控制（AI Control，第 10 课）；与 UK AISI 在控制安全论证（control safety cases）上合作。Apollo Research：为前沿实验室提供部署前的图谋（scheming）评估；撰写了 In-Context Scheming（第 8 课）与 Towards Safety Cases for AI Scheming。METR（Model Evaluation and Threat Research）：基于任务的能力评估、自主任务时间跨度（time-horizon）研究；其 "Common Elements of Frontier AI Safety Policies" 对各实验室的框架做了比较。Eleos AI Research：模型福祉（model welfare）部署前评估（第 19 课）；执行了 Claude Opus 4 的福祉评估。

**Type:** Learn
**Languages:** none
**Prerequisites:** Phase 18 · 01-27 (prior Phase 18 lessons)
**Time:** ~45 minutes

## 学习目标

- 识别实验室之外对齐研究生态中的五家机构及其核心产出。
- 描述 MATS 的规模（学者数、论文数、h-index）及其作为人才管道的角色。
- 描述 Redwood 的 AI 控制研究议程及其与 UK AISI 的合作。
- 描述 METR 基于任务的评估方法论。

## 问题背景

前沿实验室（第 18 课）在内部开展安全评估，并选择性地发布部分结果。实验室之外的生态系统才是评估得到验证、新型失效模式首先被发现、人才得到培养的地方。理解这个生态，有助于判断各项研究发现分别被谁所信任。

## 核心概念

### MATS（ML Alignment & Theory Scholars）

始于 2021 年底。研究导师制项目；学者与一位资深研究者共同投入 10-12 周，专攻一个具体的对齐问题。

规模（2026 年）：
- 自成立以来培养 527+ 名研究者。
- 发表 180+ 篇论文。
- 引用量超 1 万次。
- h-index 47。
- 2024 年夏季届：90 名学者 + 40 名导师；注册为 501(c)(3) 非营利组织。

职业去向：2025 年之前的校友中约 80% 从事安全/安保工作。200+ 人任职于 Anthropic、DeepMind、OpenAI、UK AISI、RAND、Redwood、METR、Apollo。

### Redwood Research

应用对齐实验室。由 Buck Shlegeris 创立。提出了 AI 控制研究议程（第 10 课）。与 UK AISI 在控制安全论证上合作。为 DeepMind 和 Anthropic 的评估设计提供咨询。

代表性论文：Greenblatt, Shlegeris et al., "AI Control"（arXiv:2312.06942, ICML 2024）；Alignment Faking（Greenblatt, Denison, Wright et al., arXiv:2412.14093，与 Anthropic 合著）。

研究风格：具体的威胁模型、最坏情况下的对手、可以接受压力测试的具体协议。

### Apollo Research

为前沿实验室提供部署前的图谋评估。撰写了 In-Context Scheming（第 8 课，arXiv:2412.04984）。是 2025 年 OpenAI 反图谋（anti-scheming）训练合作项目的合作方。产出了 Towards Safety Cases for AI Scheming（2024）。

研究风格：在可能涌现欺骗行为的智能体（agentic）场景中做评估；三支柱分解框架（未对齐、目标导向性、情境感知）。

### METR（Model Evaluation and Threat Research）

基于任务的能力评估。自主任务完成的时间跨度研究。"Common Elements of Frontier AI Safety Policies"（metr.org/common-elements, 2025）对各实验室的框架做了比较。

与 Apollo 合著了 AI Scheming 安全论证草案（safety-case sketch）。

研究风格：长程任务评估、实证的能力测量、框架综述。

### Eleos AI Research

模型福祉部署前评估。执行了 Claude Opus 4 的福祉评估，记录于系统卡片（system card）第 5.3 节。为第 19 课中与福祉相关的论断提供外部方法论核验。

### 生态流动

MATS 培养研究者。毕业生进入 Anthropic、DeepMind、OpenAI（实验室安全团队），或进入 Redwood、Apollo、METR、Eleos（外部评估机构）。外部评估机构与实验室以及 UK AISI / CAISI 合作。研究成果又回流到 MATS，反哺下一届学者。

### 为什么这一层很重要

单一来源的评估并不可靠：实验室评估自己的模型存在结构性的利益冲突。外部评估者可以提出并验证实验室可能少报的失效模式。2024 年的 Sleeper Agents 论文（第 7 课）由 Anthropic + Redwood 合作完成；Alignment Faking 是 Anthropic + Redwood；In-Context Scheming 是 Apollo；Anti-Scheming 是 Apollo + OpenAI。多机构结构本身就是质量控制机制。

### 在 Phase 18 中的位置

第 7-11 课引用了 Redwood 和 Apollo 的工作；第 18 课引用了 METR 的框架比较；第 19 课引用了 Eleos。第 28 课则是这张组织地图本身——本阶段其余课程所依赖的生态全貌。

## 生产实践

本课没有代码。请阅读 METR 的 "Common Elements of Frontier AI Safety Policies"，作为外部综述如何为实验室内部政策工作增加价值的范例。

## 交付产物

本课产出 `outputs/skill-ecosystem-map.md`。给定一项对齐论断或评估，它能识别出所属机构、发表渠道和方法论风格，并与已知的对应机构进行交叉核验。

## 练习

1. 从第 7-15 课中挑选一篇论文，识别其中涉及的机构。将作者与 MATS 校友及当前生态中的任职情况做交叉核对。

2. 阅读 METR 的 "Common Elements of Frontier AI Safety Policies"。找出他们强调的三处跨实验室共识，以及两处最大的分歧。

3. MATS 的职业去向约 80% 集中在安全/安保领域。请论证这种筛选压力究竟是适应性的（为领域培养人才）还是有偏的（过滤掉了非主流立场）。

4. Redwood 和 Apollo 都做控制/图谋相关工作，但风格不同。挑选一种失效模式，描述两家机构各自会如何研究它。

5. Eleos AI 是唯一专注模型福祉的机构。设计一家假想的第二机构，聚焦另一个与福祉相邻的问题（认知自由、机器人具身等），并阐述其方法论。

## 关键术语

| 术语 | 大家怎么说 | 实际含义 |
|------|-----------------|------------------------|
| MATS | "那个导师制项目" | ML Alignment & Theory Scholars；自 2021 年以来培养 527+ 名研究者 |
| Redwood Research | "做控制的实验室" | 应用对齐；AI Control 的作者；UK AISI 的合作方 |
| Apollo Research | "做图谋评估的" | 为前沿实验室提供部署前图谋评估 |
| METR | "做任务时间跨度评估的" | 基于任务的能力评估；框架综述 |
| Eleos AI | "做福祉的实验室" | 模型福祉部署前评估 |
| 人才管道 | "MATS -> 实验室" | MATS 毕业生流向 Anthropic、DM、OpenAI、Redwood、Apollo、METR |
| 外部评估 | "非实验室的核验" | 不由模型生产方自己执行的评估；增加可信度 |

## 延伸阅读

- [MATS (ML Alignment & Theory Scholars)](https://www.matsprogram.org/) — 导师制项目
- [Redwood Research](https://www.redwoodresearch.org/) — AI Control 系列论文
- [Apollo Research](https://www.apolloresearch.ai/) — 图谋评估
- [METR — Common Elements of Frontier AI Safety Policies](https://metr.org/blog/2025-03-26-common-elements-of-frontier-ai-safety-policies/) — 框架比较
- [Eleos AI Research](https://www.eleosai.org/research) — 模型福祉方法论
