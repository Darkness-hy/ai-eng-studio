# 前沿安全框架 — RSP、PF、FSF

> 三大实验室的框架定义了 2026 年前沿能力的行业治理格局。Anthropic 的 Responsible Scaling Policy（负责任扩展政策）v3.0（2026 年 2 月）引入了分级的 AI 安全等级（ASL-1 至 ASL-5+），仿照生物安全等级设计，其中 ASL-3 已于 2025 年 5 月针对 CBRN 相关模型启动。OpenAI 的 Preparedness Framework（准备度框架）v2（2025 年 4 月）为被追踪能力定义了五条标准，并将能力报告（Capabilities Reports）与防护措施报告（Safeguards Reports）分开。DeepMind 的 Frontier Safety Framework（前沿安全框架）v3.0（2025 年 9 月）引入了关键能力等级（Critical Capability Levels），其中包括新增的「有害操纵」CCL。三者如今都包含竞争者调整条款，允许在同行实验室未配备同等防护措施就发布时推迟相应要求。跨实验室的对齐仍是结构性而非术语性的：「Capability Thresholds」「High Capability thresholds」和「Critical Capability Levels」指代的是同类构造。

**Type:** Learn
**Languages:** none
**Prerequisites:** Phase 18 · 17 (WMDP), Phase 18 · 07-09 (deception failures)
**Time:** ~75 minutes

## 学习目标

- 描述 Anthropic 的 ASL 分级结构，以及是什么触发了 ASL-3。
- 说出 OpenAI Preparedness Framework v2 中被追踪能力的五条标准。
- 描述 DeepMind 的关键能力等级结构以及「有害操纵」CCL。
- 解释竞争者调整条款，以及它们为何对竞赛动态至关重要。
- 给出安全论证（safety case）的定义，并描述其三支柱结构（监测、不可谋划、能力欠缺）。

## 问题背景

第 7-17 课已经确认：欺骗是可能的、双重用途能力确实存在、评估有其局限。一个拥有前沿能力模型的实验室需要一套内部治理结构，它要：
- 定义何时必须新增防护措施的阈值。
- 定义扩展规模之前必须完成的评估。
- 描述一份安全论证应当是什么样子。
- 处理竞赛动态问题（如果竞争对手不带防护措施就发布，你该怎么办？）。

这三份 2025-2026 年的框架代表了当前的最高水平——不完美、仍在演进，但各实验室之间已足够对齐，以至于治理问题如今变成了这些框架是否够用，而不是它们是否存在。

## 核心概念

### Anthropic Responsible Scaling Policy v3.0（2026 年 2 月）

ASL 结构：
- ASL-1：非前沿模型（归入弱于前沿的基线）。
- ASL-2：当前的前沿基线；按常规防护措施部署。
- ASL-3：灾难性滥用风险显著更高；具备 CBRN 相关能力。2025 年 5 月启动。
- ASL-4：跨越 AI R&D-2 阈值；能够自动化入门级 AI 研究的模型。
- ASL-5+：高阶 AI 研发；能显著加速有效扩展的模型。

v3.0 的新内容：
- 前沿安全路线图（以删节形式公开）。
- 风险报告（每季度一次，部分接受外部评审）。
- AI 研发被拆分为 AI R&D-2 与 AI R&D-4。
- 一旦跨越 AI R&D-4，就必须提交正面安全论证（affirmative safety case），识别模型追求未对齐目标所带来的未对齐风险。

### OpenAI Preparedness Framework v2（2025 年 4 月 15 日）

被追踪能力的五条标准：
- **可信（Plausible）。** 存在合理的威胁模型。
- **可测（Measurable）。** 可以进行实证评估。
- **严重（Severe）。** 危害规模巨大。
- **全新（Net-new）。** 不是已有风险的简单放大。
- **即时或不可挽回（Instantaneous-or-irremediable）。** 危害发生迅速，或无法撤销。

满足全部五条的能力会被追踪，其余的不会。

PF v2 的其他结构：
- 将能力报告（模型能做什么）与防护措施报告（现有哪些控制手段）分开。
- 由安全咨询小组（Safety Advisory Group）评审。
- 领导层批准；董事会的安全与保障委员会（Safety & Security Committee）负责监督。
- 「调整条款」：如果另一家实验室未配备同等防护措施就发布，OpenAI 可以降低要求。

### DeepMind Frontier Safety Framework v3.0（2025 年 9 月）

按领域划分的关键能力等级（CCL）：
- 生物武器助力（Bioweapon Uplift）
- 网络攻击助力（Cyber Uplift）
- ML 研发加速（ML R&D Acceleration）
- 有害操纵（Harmful Manipulation，v3.0 新增）：能够在高风险情境中显著改变他人信念或行为的模型。

v2.0（2025 年 2 月）增加了欺骗性对齐（Deceptive Alignment）章节，并为 ML 研发 CCL 设置了更高的安全等级。

### 跨实验室对齐

- Anthropic：「Capability Thresholds」（能力阈值）。
- DeepMind：「Critical Capability Levels」（关键能力等级）。
- OpenAI：「High Capability thresholds」（高能力阈值）。

尚无行业标准术语。但结构上是对齐的：三级前沿能力分层，并配有公开的评估标准。三者都包含竞争者调整条款（2025 年起）。

### 安全论证

安全论证（safety case）是一份书面论证，说明某次部署在最坏情况假设下是可接受地安全的。标准结构围绕三个支柱：

- **监测（Monitoring）。** 如果不良行为发生，我们能否检测到？
- **不可谋划（Illegibility）。** 模型是否缺乏执行一套连贯致害计划的能力？
- **能力欠缺（Incapability）。** 模型是否缺乏造成相应危害的能力？

不同的安全论证侧重不同的支柱。对于 ASL-3 的 CBRN 论证，能力欠缺（通过遗忘（unlearning）实现）是首要目标。对于欺骗性对齐，监测与不可谋划是目标。对于网络攻击助力，三者都相关。

### 竞赛动态问题

竞争者调整条款备受争议。批评者认为它们会引发逐底竞争：如果三家实验室都会在竞争对手「背叛」时降低要求，均衡就会向背叛一侧偏移。辩护者认为，当背叛的实验室安全意识更差时，替代方案（单方面坚持防护措施）会带来更糟的结果。

英国 AISI、美国 CAISI 和欧盟 AI Office（第 24 课）是外部治理层面的对应机构。实验室框架是自愿性的；监管框架正在形成之中。

### 在第 18 阶段中的位置

第 17-18 课是叠加在欺骗与红队分析之上的测量与治理层。第 19-24 课涵盖福祉、偏见、隐私、水印与监管结构。第 28 课梳理将这些评估落地实施的研究生态（MATS、Redwood、Apollo、METR）。

## 生产实践

本课没有代码。请阅读三份一手资料：RSP v3.0、PF v2、FSF v3.0。把每家实验室的分级结构与其他两家逐一对应，并找出每家实验室定义了而其他两家没有定义的一个阈值。

## 交付产物

本课产出 `outputs/skill-framework-diff.md`。给定一份安全框架或发布说明，它会将该框架的阈值定义、所需评估和安全论证结构与 RSP v3.0、PF v2、FSF v3.0 进行对比，并标记跨实验室的差距。

## 练习

1. 阅读 RSP v3.0、PF v2 和 FSF v3.0。整理一张表格，列出每家实验室的 CBRN 阈值、各自的 AI 研发阈值，以及各自要求的部署前评估。

2. 竞争者调整条款存在于全部三份框架中（2025 年起）。写一段支持它的论证，再写一段反对它的论证。指出每种立场所依赖的假设。

3. 为一个跨越 Anthropic AI R&D-4 阈值的模型设计一份安全论证。说明三个支柱（监测、不可谋划、能力欠缺）各自需要什么证据。

4. DeepMind 的 FSF v3.0 引入了「有害操纵」CCL。提出三种能够表明模型已跨越该阈值的实证测量方法。

5. 阅读 METR 的「Common Elements of Frontier AI Safety Policies」（2025）。指出三处最强的跨实验室趋同点和两处最大的分歧点。

## 关键术语

| 术语 | 人们常说 | 实际含义 |
|------|-----------------|------------------------|
| RSP | 「Anthropic 的框架」 | Responsible Scaling Policy（负责任扩展政策）；ASL 分级；v3.0，2026 年 2 月 |
| PF | 「OpenAI 的框架」 | Preparedness Framework（准备度框架）；五条标准；v2，2025 年 4 月 |
| FSF | 「DeepMind 的框架」 | Frontier Safety Framework（前沿安全框架）；CCL；v3.0，2025 年 9 月 |
| ASL-3 | 「类比生物安全三级」 | Anthropic 针对 CBRN 相关能力的等级；2025 年 5 月启动 |
| CCL | 「关键能力等级」 | DeepMind 的阈值构造；按领域划分 |
| Safety case | 「正式论证」 | 书面论证，说明部署在最坏情况假设下是可接受地安全的 |
| Adjustment clause | 「竞争者背叛许可」 | 框架条款，允许在竞争对手未配备同等防护措施就发布时降低要求 |

## 延伸阅读

- [Anthropic — Responsible Scaling Policy v3.0 (February 2026)](https://www.anthropic.com/responsible-scaling-policy) — ASL 分级、路线图、AI 研发阈值拆分
- [OpenAI — Updating the Preparedness Framework (April 15, 2025)](https://openai.com/index/updating-our-preparedness-framework/) — 五条标准、调整条款
- [DeepMind — Strengthening our Frontier Safety Framework (September 2025)](https://deepmind.google/blog/strengthening-our-frontier-safety-framework/) — CCL v3.0、有害操纵
- [METR — Common Elements of Frontier AI Safety Policies (2025)](https://metr.org/blog/2025-03-26-common-elements-of-frontier-ai-safety-policies/) — 跨实验室对比
