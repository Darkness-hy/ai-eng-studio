# Anthropic 负责任扩展政策 v3.0

> RSP v3.0 于 2026 年 2 月 24 日生效，取代 2023 年版政策。采用两级缓解措施：Anthropic 单方面承诺执行的部分，与定位为全行业建议的部分（包括 RAND SL-4 安全标准）。新增前沿安全路线图（Frontier Safety Roadmap）和风险报告（Risk Report），并将其设为常设文件而非一次性交付物。删除了 2023 年的暂停承诺。引入 AI R&D-4 阈值：一旦越过，Anthropic 必须发布一份肯定性论证（affirmative case），指出失准（misalignment）风险及缓解措施。Claude Opus 4.6 未越过该阈值。Anthropic 在 v3.0 公告中表示「要有信心地排除这一点正变得越来越困难」。SaferAI 给 2023 年 RSP 的评分是 2.2；v3.0 被下调至 1.9，使 Anthropic 落入「弱」RSP 类别，与 OpenAI 和 DeepMind 同列。定性阈值取代了 2023 年的定量承诺；删除暂停条款是最显著的倒退。

**Type:** Learn
**Languages:** Python (stdlib, RSP threshold decision engine)
**Prerequisites:** Phase 15 · 06 (AAR), Phase 15 · 07 (RSI)
**Time:** ~45 minutes

## 问题背景

前沿实验室发布的扩展政策，一部分是技术文件，一部分是治理文件，还有一部分是向监管机构释放的信号。RSP v3.0 是 Anthropic 当前生效的文件。细读它之所以重要，不在于遵守它有约束力（它没有），而在于这种表述框架塑造了实验室如何理解灾难性风险，以及他们如何向公众传达取舍。

v3.0 与 v2.0 的差异（diff）才是有用的分析单元。新增的内容：前沿安全路线图、风险报告、AI R&D-4 阈值。删除的内容：2023 年的暂停承诺。重新表述的内容：把缓解措施拆分为 Anthropic 单方面承诺与行业建议两级。外部评审机构 SaferAI 将评分从 2.2（v2）下调至 1.9（v3.0）。这就是一份扩展政策如何在外观更精致的同时变得不那么严格。

## 核心概念

### 两级缓解措施

- **Anthropic 单方面行动**：无论其他实验室怎么做，Anthropic 都会执行的部分。超过阈值即停止训练、特定的安全措施、特定的部署门槛。
- **全行业建议**：Anthropic 认为整个行业应当共同采取的措施。包括 RAND SL-4 安全标准。这些并非 Anthropic 一方的承诺，而是政策倡导。

这种两级结构在 v2 中并不存在。这意味着读者需要看清每条承诺位于哪一栏。放在「全行业建议」一栏的安全措施不是 Anthropic 的承诺，而是 Anthropic 的期望。

### AI R&D-4 阈值

这是 RSP v3.0 点名的下一个重要能力阈值。具体而言：模型能够以有竞争力的成本自动化相当大一部分 AI 研究。一旦 Anthropic 认为某个模型越过该阈值，他们必须在继续扩展之前发布一份肯定性论证，指出失准风险及缓解措施。

按 v3.0 公告，Claude Opus 4.6 未越过该阈值。文件补充道：「要有信心地排除这一点正变得越来越困难。」这一措辞很关键；它承认这个阈值已经近到成为现实关切，而非一个推测性的远期界限。

第 6 课（自动化对齐研究）和第 7 课（递归自我改进）直接关联这一阈值。自动化对齐研究员达到研究质量门槛，就是 AI R&D-4 阈值正在逼近的证据。

### 前沿安全路线图与风险报告

v3.0 将两类产出物提升为常设文件：

- **前沿安全路线图（Frontier Safety Roadmap）**：前瞻性文件，描述计划中的安全工作、能力预期和缓解研究。
- **风险报告（Risk Report）**：针对具体模型在发布后的回顾性文件，描述观测到的能力和残余风险。

两者均公开。两者均按声明的节奏更新。其用处在于：读者可以追踪 Anthropic 在路线图中声称要做的事，与他们在风险报告中实际汇报的内容相比对照如何。

### 删除暂停条款

2023 年的 RSP 包含一条明确的暂停承诺：如果模型越过特定能力阈值，训练将暂停，直到缓解措施到位。v3.0 用更软的表述取而代之（发布一份肯定性论证，若缓解措施充分则继续推进）。SaferAI 和其他分析者直接指出，这是新文件中最严重的倒退。

支持这一改动的政策论点是：2023 年的定量阈值到 2026 年已无法用当时的能力基准衡量，因为基准本身被重新标定了。反方论点是：扩展政策中的暂停条款是一种承诺装置（commitment device）；删掉它就删掉了这份政策的可信度。

### SaferAI 的降级

SaferAI 是一家对 RSP 类文件进行评级的独立机构。其公开评分：2023 年 Anthropic RSP 得分 2.2（该量表中 4.0 代表当前最佳 RSP，1.0 代表名义上存在）。v3.0 得分 1.9。这使 Anthropic 从「中等」降至「弱」，与 OpenAI 和 DeepMind 同处弱类别。

SaferAI 列出的降级因素：
- 定性阈值取代了定量阈值。
- 暂停承诺被删除。
- AI R&D-4 阈值的缓解措施被描述为「肯定性论证」，而非具体措施。
- 审查机制依赖 Anthropic 自己的安全顾问小组（Safety Advisory Group），独立监督有限。

### 这节课不是什么

这不是一节合规课。RSP v3.0 不是法规；没有任何东西强制 Anthropic 遵守它。这节课教的是以它应得的具体性和怀疑态度去阅读这份文件。扩展政策是前沿实验室就灾难性风险姿态向公众发出的主要信号。读懂它们，对任何工作依赖前沿能力的人来说都是一项实用技能。

## 生产实践

`code/main.py` 实现了一个小型决策引擎，复刻 RSP 阈值评估的形态：给定一个候选模型和一组能力测量值，返回是否越过 AI R&D-4 阈值、所需的肯定性论证章节，以及部署能否继续推进。它刻意保持简单；重点是把文件中的逻辑显式化。

## 交付产物

`outputs/skill-scaling-policy-review.md` 对照 v3.0 基准评审一份扩展政策（Anthropic、OpenAI、DeepMind 或内部政策）：两级结构、阈值、暂停承诺、独立审查。

## 练习

1. 运行 `code/main.py`。输入三个处于不同能力水平的合成模型。确认阈值评估器的行为符合预期，并生成正确的肯定性论证模板。

2. 完整阅读 RSP v3.0（32 页）。找出所有位于「全行业建议」一级的承诺。其中哪些承诺在 v2 中本属于「Anthropic 单方面行动」？

3. 阅读 SaferAI 的 RSP 评级方法。将其评分细则应用于 v3.0 文件，复现 1.9 的得分。哪一条细则对降级影响最大？

4. 2023 年的暂停承诺被删除了。提出一条替代承诺，既能保住这份政策的可信度，又能正视 2026 年基准重标定的问题。

5. 将 RSP v3.0 与 OpenAI Preparedness Framework v2（第 20 课）对比。挑出一个 v3.0 更强的方面，再挑出一个 Preparedness Framework 更强的方面。

## 关键术语

| 术语 | 通俗说法 | 实际含义 |
|---|---|---|
| RSP | 「Anthropic 的扩展政策」 | 负责任扩展政策（Responsible Scaling Policy）；v3.0 于 2026 年 2 月 24 日生效 |
| AI R&D-4 | 「研究自动化阈值」 | 以有竞争力的成本自动化大量 AI 研究的能力 |
| 肯定性论证（Affirmative case） | 「安全论证」 | 公开发布的论证，说明风险已被识别且缓解措施充分 |
| 前沿安全路线图 | 「前瞻计划」 | 关于计划中安全工作和预期能力的常设文件 |
| 风险报告 | 「对某个模型的回顾」 | 关于发布后观测到的能力和残余风险的常设文件 |
| 两级缓解措施 | 「单方面 vs 行业」 | Anthropic 承诺与行业建议，分开列出 |
| 暂停承诺 | 「2023 条款」 | 明确承诺暂停训练；在 v3.0 中被删除 |
| SaferAI 评分 | 「独立 RSP 评级」 | 第三方评分细则；v3.0 得分 1.9（v2 为 2.2） |

## 延伸阅读

- [Anthropic — Responsible Scaling Policy v3.0](https://anthropic.com/responsible-scaling-policy/rsp-v3-0) — 完整的 32 页政策文件。
- [Anthropic — RSP v3.0 announcement](https://www.anthropic.com/news/responsible-scaling-policy-v3) — 相对 v2 变更的摘要。
- [Anthropic — Frontier Safety Roadmap](https://www.anthropic.com/research/frontier-safety) — RSP v3.0 链接的常设文件。
- [Anthropic — Risk Report: Claude Opus 4.6](https://www.anthropic.com/research/risk-report-claude-opus-4-6) — 对当前前沿模型的回顾。
- [Anthropic — Measuring agent autonomy in practice](https://www.anthropic.com/research/measuring-agent-autonomy) — 将 AI R&D-4 与可测量的自主性关联起来。
