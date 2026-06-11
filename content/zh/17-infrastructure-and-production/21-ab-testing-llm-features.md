# LLM 功能的 A/B 测试 —— GrowthBook、Statsig 与「凭感觉上线」的问题

> 传统 A/B 测试并不是为非确定性的 LLM 设计的。关键区别在于：评测（evals）回答的是「模型能不能完成任务？」，A/B 测试回答的是「用户在不在乎？」两者缺一不可；靠感觉验收（vibe check）就上线的时代已经结束。2026 年值得测试的内容：提示工程（措辞）、模型选型（GPT-4 vs GPT-3.5 vs 开源模型；准确率 vs 成本 vs 延迟）、生成参数（temperature、top-p）。真实案例：某聊天机器人的奖励模型变体带来对话长度 +70%、留存 +30%；Nextdoor 的 AI 邮件主题行实验在精炼奖励函数后带来 CTR +1%；Khan Academy 的 Khanmigo 在「延迟 vs 数学准确率」这条轴上持续迭代。平台格局：**Statsig**（2025 年 9 月被 OpenAI 以 11 亿美元收购）——序贯检验、CUPED、一体化平台。**GrowthBook**——开源、数仓原生，提供贝叶斯 + 频率学派 + 序贯三套统计引擎，支持 CUPED、SRM 检查、Benjamini-Hochberg + Bonferroni 校正。选哪个，取决于你是否偏好数仓 SQL 工作流，以及「被 OpenAI 收购」对你的组织是否重要。

**Type:** Learn
**Languages:** Python (stdlib, toy sequential test simulator)
**Prerequisites:** Phase 17 · 13 (Observability), Phase 17 · 20 (Progressive Deployment)
**Time:** ~60 minutes

## 学习目标

- 区分评测（「模型能不能完成任务」）与 A/B 测试（「用户在不在乎」）。
- 列举三条可测试的轴（提示、模型、参数），并为每条轴选定指标。
- 解释 CUPED、序贯检验（sequential testing）以及 Benjamini-Hochberg 多重比较校正。
- 根据数仓 SQL 取向和对收购方的态度，在 Statsig 与 GrowthBook 之间做出选择。

## 问题背景

你手工调好了一个系统提示。感觉更好了。你把它上线。转化率的变化只是噪声。你怪指标不行。又或者你换了一个新模型，转化率没动——是模型退化了，还是改动太小测不出来？你不知道，因为你上线时没有做 A/B。

评测回答的是模型能否在一个带标注的数据集上完成任务，并不回答用户是否更喜欢这些输出。只有受控的在线实验才能回答后者——而且前提是实验有足够的统计功效、控制了非确定性，并对多重比较做了校正。

## 核心概念

### 评测 vs A/B 测试

**评测（Evals）**——离线、带标注数据集、有评判者（评分细则、LLM-as-judge 或人工）。回答：「在这个固定分布上，输出是否正确 / 有帮助 / 安全？」

**A/B 测试**——在线、真实用户、随机分组。回答：「新变体是否真的撬动了那个重要的用户级指标？」

两者都不可少。评测在曝光前拦截回归；A/B 在上线后确认产品影响。

### 测什么

1. **提示工程**——措辞、系统提示结构、示例。指标：任务成功率、用户留存、单请求成本。
2. **模型选型**——GPT-4 vs GPT-3.5-Turbo vs Llama 开源模型。指标：准确率（任务层面）+ 单请求成本 + P99 延迟。多目标优化。
3. **生成参数**——temperature、top-p、max_tokens。指标：随任务而定（输出多样性 vs 确定性）。

### CUPED——方差缩减

全称 Controlled-experiments Using Pre-Experiment Data（利用实验前数据的受控实验）。在比较实验期数据之前，先用回归消去实验前阶段的方差。典型的方差缩减幅度：30-70%。有效样本量等于白白增加。

实现情况：Statsig 和 GrowthBook 都已实现。

### 序贯检验

经典 A/B 假设样本量固定。序贯检验（「边看边决策」，peek-and-decide）在反复查看结果的情况下仍能控制假阳性率。永远有效（always-valid）的序贯方法（mSPRT、Howard 的置信序列）允许你在赢家明显时提前停止实验。

### 多重比较校正

在 95% 置信水平下跑 20 个 A/B 测试，光凭运气就会出一个假阳性。Bonferroni 校正收紧每个检验的 α；Benjamini-Hochberg 控制错误发现率（false-discovery rate）。GrowthBook 两者都实现了。

### SRM——样本比例失配

分流哈希把用户随机分到各个变体。如果设定的 50/50 分流实际跑出 47/53，说明有东西坏了——SRM 检查会把它标出来。两个平台都实现了该检查。

### Statsig vs GrowthBook

**Statsig**：
- 被 OpenAI 以 11 亿美元收购（2025 年 9 月）。托管式 SaaS。
- 序贯检验、CUPED、保留对照人群（held-out populations）。
- 一体化：功能开关 + 实验 + 可观测性。
- 最适合：团队本来就想要打包好的产品，且不在意 OpenAI 的所有权。

**GrowthBook**：
- 开源（MIT 协议）；数仓原生（直接读取 Snowflake/BigQuery/Redshift）。
- 多套引擎：贝叶斯、频率学派、序贯。
- CUPED、SRM、Bonferroni、BH 校正。
- 可自托管，也有托管云服务。
- 最适合：以数仓 SQL 为核心的团队，数据团队掌控指标层，并希望使用开源方案。

### 非确定性让功效计算变复杂

同一个提示会产生不同的输出。传统的功效计算假设观测值独立同分布（IID）。在 LLM 非确定性下，有效样本量低于名义样本量。把所需样本量乘以约 1.3-1.5 倍作为安全余量。

### 真实案例结果

- 聊天机器人奖励模型变体：对话长度 +70%，留存 +30%。
- Nextdoor 邮件主题行：精炼奖励函数后 CTR +1%。
- Khan Academy 的 Khanmigo：在「延迟 vs 数学准确率」之间迭代权衡。

### 反模式：凭感觉上线

每个资深工程师都能说出一个因为「感觉更好」就没做 A/B 直接上线的功能。其中大多数让产品指标出现了回归，而团队几个月都没察觉。A/B 测试是强制纠偏机制。

### 应该记住的数字

- Statsig 被 OpenAI 收购：11 亿美元，2025 年 9 月。
- GrowthBook：开源 MIT 协议；贝叶斯 + 频率学派 + 序贯。
- CUPED 方差缩减：30-70%。
- LLM 非确定性 → 样本量加 30-50% 缓冲。

## 生产实践

`code/main.py` 模拟一个同时带固定边界和序贯边界的序贯 A/B 测试，展示序贯检验如何让你提前停止实验。

## 交付产物

本课产出 `outputs/skill-ab-plan.md`。给定功能改动、负载情况和基线，它会选出平台、设定门控标准并计算样本量。

## 练习

1. 运行 `code/main.py`。在基线转化率 3%、预期提升 5% 的情况下，达到 80% 功效需要多大样本量？
2. 为一个受医疗合规监管、要求本地部署的客户选择 Statsig 还是 GrowthBook。
3. 设计一个对比 GPT-4 与 GPT-3.5「单张已解决工单成本」的 A/B 实验。主指标、护栏指标、次要指标分别是什么？
4. 金丝雀发布通过了，但 A/B 显示转化率 -1.2%。要上线吗？写出升级处理（escalation）标准。
5. 对一个方差为实验期 60% 的实验前阶段应用 CUPED，计算有效样本量的提升幅度。

## 关键术语

| 术语 | 大家怎么说 | 实际含义 |
|------|----------------|------------------------|
| Eval | 「离线测试」 | 在带标注数据集上评估模型能力 |
| A/B 测试 | 「实验」 | 在真实用户上做的随机化对照比较 |
| CUPED | 「方差缩减」 | 用实验前阶段做回归以降低方差 |
| 序贯检验 | 「可以随时偷看的检验」 | 允许提前停止的永远有效的检验过程 |
| 多重比较 | 「族错误」 | 跑很多检验会推高假阳性率 |
| Bonferroni | 「严格校正」 | 把 α 除以检验数量 |
| Benjamini-Hochberg | 「BH FDR」 | 控制错误发现率，相对没那么保守 |
| SRM | 「分流坏了」 | 样本比例失配；分流逻辑有 bug |
| Statsig | 「OpenAI 家的」 | 商业一体化平台，2025 年被收购 |
| GrowthBook | 「开源那个」 | MIT 协议、数仓原生的实验平台 |
| mSPRT | 「序贯概率比检验」 | 经典的序贯检验方法 |

## 延伸阅读

- [GrowthBook — How to A/B Test AI](https://blog.growthbook.io/how-to-a-b-test-ai-a-practical-guide/)
- [Statsig — Beyond Prompts: Data-Driven LLM Optimization](https://www.statsig.com/blog/llm-optimization-online-experimentation)
- [Statsig vs GrowthBook comparison](https://www.statsig.com/perspectives/ab-testing-feature-flags-comparison-tools)
- [Deng et al. — CUPED](https://www.exp-platform.com/Documents/2013-02-CUPED-ImprovingSensitivityOfControlledExperiments.pdf)
- [Howard — Confidence Sequences](https://arxiv.org/abs/1810.08240)
