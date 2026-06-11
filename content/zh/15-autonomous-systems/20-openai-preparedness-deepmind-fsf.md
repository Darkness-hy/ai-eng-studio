# OpenAI Preparedness Framework 与 DeepMind Frontier Safety Framework

> OpenAI Preparedness Framework v2（2025 年 4 月）引入了研究类别（Research Categories）——长程自主性（Long-range Autonomy）、藏拙（Sandbagging）、自主复制与适应（Autonomous Replication and Adaptation）、破坏安全防护（Undermining Safeguards）——与追踪类别（Tracked Categories）相区别。追踪类别会触发能力报告（Capabilities Reports）和安全防护报告（Safeguards Reports），由安全顾问小组（Safety Advisory Group）审查。DeepMind 的 FSF v3（2025 年 9 月，并于 2026 年 4 月 17 日新增追踪能力等级 Tracked Capability Levels）将自主性并入 ML 研发和网络安全两个领域（ML 研发自主性 1 级 = 以相对于"人类 + AI 工具"具有竞争力的成本完全自动化 AI 研发流水线）。FSF v3 通过对工具性推理滥用的自动化监控，明确应对欺骗性对齐（deceptive alignment）。坦诚地说：PF v2 中的研究类别（包括长程自主性）不会自动触发缓解措施；政策措辞是"潜在的"缓解措施。DeepMind 自己也承认，如果工具性推理能力增强，自动化监控"长期来看将不再足够"。

**Type:** Learn
**Languages:** Python (stdlib, three-framework decision-table diff tool)
**Prerequisites:** Phase 15 · 19 (Anthropic RSP)
**Time:** ~45 minutes

## 问题背景

第 19 课细读了 Anthropic 的扩展政策。本课通过阅读 OpenAI 和 DeepMind 的政策来补全全貌。这三份文件是回答同一个问题的同源产物——前沿实验室应该在什么时候暂停或限制一个模型——它们在一小组类别上趋同，又在一些关键的具体之处分歧。

趋同之处：三家都把长程自主性列为值得追踪的能力类别。三家都承认欺骗性行为（对齐伪装、藏拙）是一类特定风险。三家都设有内部审查机构。分歧之处：OpenAI 将类别分为"追踪"（强制缓解）和"研究"（无自动触发）两类。DeepMind 将自主性并入两个领域，而不是单独命名。各实验室的命名方式各异——追踪 vs 研究、关键 vs 中等、一级 vs 二级；但一项能力被归入哪个类别，其操作性后果在各实验室之间是不同的。

把它们放在一起读才是有用的练习。同一项能力可以在 Anthropic 是"强制缓解"，在 OpenAI 是"监控但不触发"，在 DeepMind 是"在特定领域内追踪"。政策立场至关重要。

## 核心概念

### OpenAI Preparedness Framework v2（2025 年 4 月）

结构：

- **追踪类别（Tracked Categories）**：触发能力报告（模型能做什么）和安全防护报告（已部署了哪些缓解措施）。部署前由安全顾问小组审查。
- **研究类别（Research Categories）**：实验室正在追踪、但尚未承诺采取具体缓解措施的模型能力。包括长程自主性、藏拙、自主复制与适应、破坏安全防护。

v2 的关键定调：研究类别不会自动触发缓解措施。政策措辞是"潜在的"缓解措施。这正是需要紧盯的差异点：一个类别位于追踪还是研究，是"被限制"与"被观察"之间的枢纽。

### DeepMind Frontier Safety Framework v3（2025 年 9 月；2026 年 4 月 17 日新增追踪能力等级）

结构：

- **关键能力等级（Critical Capability Levels, CCLs）**：跨五个领域的能力阈值：网络安全、生物、ML 研发、CBRN、自主性（并入 ML 研发和网络安全）。
- **追踪能力等级（Tracked Capability Levels）**：2026 年 4 月新增的更细粒度划分。具体例子：ML 研发自主性 1 级 = 以相对于"人类 + AI 工具"具有竞争力的成本完全自动化 AI 研发流水线。
- **欺骗性对齐监控**：明确承诺对工具性推理滥用进行自动化监控。

其对自主性的定调与 OpenAI 不同。DeepMind 没有把"自主性"保留为顶级领域；它被并入了自主性可能造成危害的领域（ML 研发和网络安全）。支持的论点是：脱离领域的自主性是没有风险的能力；反方论点是：跨领域的自主性是一种元风险，框架应该为它单独命名。

### 三家趋同的地方

- 内部安全顾问小组（Anthropic 称 SAG，OpenAI 称 SAG，DeepMind 为内部委员会）。高能力模型部署前进行审查。
- 明确提及欺骗性对齐 / 对齐伪装是一类风险。
- 按公开声明的节奏产出常态化文档（Anthropic：Frontier Safety Roadmap、Risk Report；OpenAI：Capabilities and Safeguards Reports；DeepMind：FSF 更新周期）。
- 承认仅靠监控的防御存在上限。DeepMind 说得很直白："自动化监控长期来看将不再足够。"

### 它们分歧的地方

- **Anthropic**：v3.0 中移除了暂停承诺；AI R&D-4 阈值是点名的下一道关口。
- **OpenAI**：追踪 vs 研究的二分；研究类别（包括长程自主性）不会自动设限。
- **DeepMind**：自主性并入其他领域；2026 年 4 月的追踪能力等级增加了粒度。

### 藏拙：一项让三份框架都变复杂的具体能力

藏拙（模型在评估中策略性地表现不佳）位列 OpenAI 的研究类别。Anthropic 的 RSP v3.0 通过评估情境差距来应对它（第 1 课）。DeepMind 通过 FSF v3 中的欺骗性对齐监控来应对。

如果模型在评估中藏拙，每个框架的能力阈值都会被低估。框架只有在测量可靠时才有效。这就是为什么除实验室自评之外，还需要外部测量（第 21 课，METR）和对抗性评估。

### 政策阅读技能

- 定位：你关心的每项能力都应当能在政策中找到。找不到，就说明政策没有覆盖它。
- 归类：它是追踪类（触发缓解）还是研究类（追踪但不触发）？OpenAI 明确命名了这一区分；Anthropic 和 DeepMind 有各自的对应物。
- 节奏：政策是按公开声明的时间表更新，还是只在特定事件之后更新？公开声明的节奏更有约束力。
- 独立性：外部审查是强制的还是可选的？Anthropic 与 Apollo 和美国 AI 安全研究所（US AI Safety Institute）合作；OpenAI 与 METR 合作；DeepMind 主要依靠内部 SAG。

## 生产实践

`code/main.py` 实现了一个小型决策表对比工具。给定一项能力（自主性、欺骗性对齐、研发自动化、网络攻击增益等），它输出三份政策各自如何归类该能力，以及会触发哪些缓解措施。它是一个阅读辅助工具，不是政策工具。

## 交付产物

`outputs/skill-cross-policy-diff.md` 以三份框架为参照，针对某项具体能力生成跨政策对比。

## 练习

1. 运行 `code/main.py`。至少挑两项你能对照原始文件核实的能力，确认对比工具的输出与政策一致。

2. 完整阅读 OpenAI Preparedness Framework v2。找出每个研究类别。针对每一个，用一句话说明它为什么被归入研究类而不是追踪类。

3. 完整阅读 DeepMind FSF v3，外加 2026 年 4 月的追踪能力等级更新。找出 ML 研发自主性 1 级的具体评估标准。你会如何从外部测量它？

4. 藏拙位列 OpenAI 的研究类别。设计一个能迫使藏拙模型暴露其真实能力的评估。参考第 1 课关于评估情境博弈的讨论。

5. 就一项具体能力（自选）对比三份政策。指出你认为哪份政策的归类最严谨、哪份最不严谨。用原文佐证。

## 关键术语

| 术语 | 通俗说法 | 实际含义 |
|---|---|---|
| Preparedness Framework | "OpenAI 的扩展政策" | PF v2（2025 年 4 月）；追踪 vs 研究类别 |
| 追踪类别（Tracked Category） | "强制缓解" | 触发能力报告 + 安全防护报告；SAG 审查 |
| 研究类别（Research Category） | "仅监控" | 追踪但无自动缓解；包括长程自主性 |
| Frontier Safety Framework | "DeepMind 的扩展政策" | FSF v3（2025 年 9 月）+ 追踪能力等级（2026 年 4 月） |
| CCL | "关键能力等级" | DeepMind 按领域设定的阈值（网络安全、生物、ML 研发、CBRN） |
| ML 研发自主性 1 级 | "研发自动化" | 以有竞争力的成本完全自动化 AI 研发流水线 |
| 藏拙（Sandbagging） | "策略性表现不佳" | 模型在评估中故意表现不佳；位列 OpenAI 研究类别 |
| 工具性推理（Instrumental reasoning） | "手段—目的推理" | 关于如何达成目标的推理；DeepMind 监控的对象 |

## 延伸阅读

- [OpenAI — Updating our Preparedness Framework](https://openai.com/index/updating-our-preparedness-framework/) — v2 发布公告。
- [OpenAI — Preparedness Framework v2 PDF](https://cdn.openai.com/pdf/18a02b5d-6b67-4cec-ab64-68cdfbddebcd/preparedness-framework-v2.pdf) — 完整文档。
- [DeepMind — Strengthening our Frontier Safety Framework](https://deepmind.google/blog/strengthening-our-frontier-safety-framework/) — FSF v3 发布公告。
- [DeepMind — Updating the Frontier Safety Framework (April 2026)](https://deepmind.google/blog/updating-the-frontier-safety-framework/) — 追踪能力等级的新增说明。
- [Gemini 3 Pro FSF Report](https://storage.googleapis.com/deepmind-media/gemini/gemini_3_pro_fsf_report.pdf) — FSF 格式风险报告的示例。
