# 合规 — SOC 2、HIPAA、GDPR、PCI-DSS、EU AI Act、ISO 42001

> 多框架合规覆盖是 2026 年企业级订单的入场门槛。**EU AI Act**：自 2024 年 8 月 1 日起生效。大多数高风险要求于 2026 年 8 月 2 日开始执行。针对高风险系统义务的罚款最高可达 1500 万欧元或全球年营业额的 3%（第 99(4) 条）；针对被禁止的 AI 行为最高可达 3500 万欧元或 7%（第 99(3) 条）。只要服务欧盟用户即全球适用。**Colorado AI Act**：2026 年 6 月 30 日生效（因 SB25B-004 从 2026 年 2 月推迟）——要求对高风险系统进行影响评估，并赋予对 AI 决策提出申诉的权利。Virginia 在信贷/就业/住房/教育领域有类似规定。**SOC 2 Type II**：事实上的 B2B AI 准入要求（金融科技领域要求 Type II 而非 Type I）。**GDPR**：迄今有记录的最大 AI 专项罚款是荷兰 DPA 于 2024 年 9 月对 Clearview AI 开出的 3050 万欧元；意大利 Garante 于 2024 年 12 月对 OpenAI 开出 1500 万欧元罚款（2026 年 3 月上诉后被推翻）。推理时的实时 PII 脱敏是站得住脚的标准做法；事后清理远远不够。**HIPAA**：医疗行业受其约束——没有 BAA 不得将 PHI 发送给外部 AI 服务。**PCI-DSS**：AI 交互层的合规覆盖需要配置加合同约定，并非自动获得。**ISO 42001**：新兴的 AI 治理标准，正与 ISO 27001 一道成为越来越普遍的采购要求。参考画像：OpenAI 持有 SOC 2 Type 2、ISO/IEC 27001:2022、ISO/IEC 27701:2019、GDPR/CCPA/HIPAA（BAA）/FERPA，以及面向 ChatGPT 支付组件的 PCI-DSS。跨框架映射可以降低审计疲劳：访问控制可同时映射到 ISO 27001 A.5.15-5.18、GDPR 第 32 条、HIPAA §164.312(a)。

**Type:** Learn
**Languages:** (Python optional — compliance is policy + process, not code)
**Prerequisites:** Phase 17 · 25 (Security), Phase 17 · 13 (Observability)
**Time:** ~60 minutes

## 学习目标

- 列举 2026 年与 LLM 产品相关的七个合规框架，并将每个框架对应到相应的客户群体。
- 准确说出 EU AI Act 的执行时间线（2024 年 8 月生效；高风险条款 2026 年 8 月开始执行）以及两档罚款上限（高风险义务：1500 万欧元 / 3%；被禁止行为：3500 万欧元 / 7%）。
- 解释为什么事后 PII 清理在 GDPR 下不够，并指出实时推理层脱敏才是站得住脚的标准。
- 描述跨框架控制映射（例如访问控制可映射到 ISO 27001 A.5.15-5.18 + GDPR 第 32 条 + HIPAA §164.312(a)）。

## 问题背景

某企业客户的采购方要求提供 SOC 2 Type II、GDPR、HIPAA BAA、ISO 27001 以及"EU AI Act 合规声明"。你的团队只有 SOC 2 Type I，距离拿到 Type II 还有六个月，而 GDPR 第 30 条记录还没开始做。

多框架合规覆盖不是一个 LLM 问题——它是一个企业级 SaaS 问题，只是叠加了 LLM 特有的要求。2026 年的采购团队想要的是一张矩阵——每个框架一行、每项控制一列——而不是一份 PDF。

## 核心概念

### 七个框架

| 框架 | 适用范围 | LLM 特有要求 |
|-----------|-------|--------------------------|
| SOC 2 Type II | B2B SaaS 基线 | 流程控制需经过 6-12 个月的审计 |
| HIPAA | 美国医疗 | 必须签署 BAA；没有签署协议，PHI 不得离开自有基础设施 |
| GDPR | 欧盟用户 | 实时 PII 脱敏；数据主体权利；第 30 条记录 |
| PCI-DSS | 支付数据 | 涉及支付的 AI 需要配置加合同约定 |
| EU AI Act | 服务欧盟用户 | 风险等级分类；高风险系统需符合性评估、文档、日志记录 |
| Colorado AI Act | 服务科罗拉多州居民 | 影响评估；申诉权 |
| ISO 42001 | AI 治理 | 新兴标准；与 ISO 27001 配套 |

### EU AI Act 时间线

- 2024 年 8 月 1 日：生效。
- 2025 年 2 月 2 日：被禁止的 AI 行为开始执行。
- 2026 年 8 月 2 日：高风险系统条款开始执行（符合性评估、文档、日志记录）。
- 2027 年 8 月：纳入统一立法监管的产品中的高风险系统开始执行。

风险等级：不可接受（禁止）、高风险（符合性评估 + 日志记录）、有限风险（透明度要求）、最低风险（无约束）。大多数 B2B LLM SaaS 属于有限风险；涉及就业、信贷、教育、执法、移民、基本公共服务时触发高风险。

罚款（第 99 条）：违反高风险系统义务最高 1500 万欧元或全球年营业额的 3%（第 99(4) 条）；从事被禁止的 AI 行为最高 3500 万欧元或 7%（第 99(3) 条）；以较高者为准。

### GDPR — 实时脱敏才是标准

事后清理（让 LLM 看到 PII 之后再脱敏）不是站得住脚的姿态——模型已经看过数据了。实时推理层脱敏才是 2026 年的标准做法：

- 在调用 LLM 之前做实体识别。
- 一致性化名替换（Mesh 方案）保留语义。
- 只存储脱敏后的提示词，原始数据仅在用户明确同意后保留。

近期执法案例：荷兰 DPA 于 2024 年 9 月对 Clearview AI 开出的 3050 万欧元罚款是迄今有记录的最大 AI 专项 GDPR 罚款；意大利 Garante 于 2024 年 12 月对 OpenAI 开出的 1500 万欧元是最大的 LLM 专项罚款，尽管该处罚在 2026 年 3 月上诉后被推翻，相关裁定仍在进一步审理中。"事后清理"的合规主张已在审计中败下阵来。

### HIPAA — BAA 不是可选项

没有签署商业伙伴协议（Business Associate Agreement，BAA），不得将 PHI 发送给外部 AI 服务。三大云厂商的 LLM 平台（Bedrock、Azure OpenAI、Vertex）都提供 BAA。OpenAI 直连 API 提供 BAA。Anthropic 直连 API 提供 BAA。发送 PHI 之前务必先确认。

### SOC 2 Type II

Type I：控制已设计并形成文档。
Type II：控制在 6-12 个月内有效运行。

2026 年的 B2B 采购默认要求 Type II。Type I 是起步，Type II 才是门槛。

常见审计关注点：访问日志（谁看了什么）、变更管理（如何部署的）、风险评估（按季度）、事件响应（演练过吗？）。Phase 17 · 25 中的审计日志可以直接复用。

### 跨框架映射

一份访问控制策略可以同时满足多个框架的控制要求：

| 控制项 | 框架 |
|---------|-----------|
| 访问日志 | ISO 27001 A.5.15-5.18、GDPR 第 32 条、HIPAA §164.312(a) |
| 变更管理 | ISO 27001 A.8.32、PCI DSS 要求 6、HIPAA 违规通知范围 |
| 传输加密 | ISO 27001 A.8.24、GDPR 第 32 条、HIPAA §164.312(e) |
| 密钥管理 | ISO 27001 A.8.19、PCI DSS 要求 8、SOC 2 CC6.1 |

合规工具（Drata、Vanta、Secureframe）可以自动完成这种映射。规模上去之后物有所值。

### ISO 42001 — 新兴标准

2023 年底发布。正与 ISO 27001 一道成为越来越普遍的采购要求。这是一个 AI 治理框架，涵盖风险管理、数据质量、透明度、人工监督。

### OpenAI 的参考画像

OpenAI 持有 SOC 2 Type 2、ISO/IEC 27001:2022、ISO/IEC 27701:2019、GDPR/CCPA/HIPAA（BAA）/FERPA，以及面向 ChatGPT 支付组件的 PCI-DSS。这大致就是 2026 年企业级市场的入场门槛。

### 应该记住的数字

- EU AI Act 罚款：最高 1500 万欧元 / 3%（高风险义务，第 99(4) 条）；最高 3500 万欧元 / 7%（被禁止行为，第 99(3) 条）。
- EU AI Act 高风险条款执行日期：2026 年 8 月 2 日。
- 有记录的最大 AI 专项 GDPR 罚款：3050 万欧元，Clearview AI（荷兰 DPA，2024 年 9 月）。
- 最大的 LLM 专项 GDPR 罚款：1500 万欧元，OpenAI（意大利 Garante，2024 年 12 月；2026 年 3 月上诉后被推翻）。
- SOC 2 Type II 窗口期：控制有效运行 6-12 个月。
- Colorado AI Act 生效日期：2026 年 6 月 30 日（因 SB25B-004 从 2026 年 2 月推迟）。

## 生产实践

`code/main.py` 是一个用 Python 写的合规映射表——给定一项控制，列出它能满足的所有框架。

## 交付产物

本课产出 `outputs/skill-compliance-matrix.md`。给定客户群体和地域，给出所需的框架与控制项清单。

## 练习

1. 你的第一个企业客户要求 SOC 2 Type II、HIPAA BAA、EU AI Act 声明。赢下这单的最小可行合规姿态是什么？
2. 按 EU AI Act 风险等级对三个假想的 LLM 产品进行分类。落入高风险后会有什么变化？
3. 你不小心把 PHI 发给了一个没有签 BAA 的供应商。走一遍事件响应流程。
4. 论证 ISO 42001 对一家中型 AI 厂商而言在 2026 年是否"必需"。
5. 把你的 LLM 审计日志字段（Phase 17 · 25）映射到至少三个框架的控制项。

## 关键术语

| 术语 | 大家怎么说 | 实际含义 |
|------|----------------|------------------------|
| SOC 2 Type II | "审计过的控制" | 控制在 6-12 个月内持续运行，并经独立鉴证 |
| HIPAA BAA | "医疗合同" | 商业伙伴协议（Business Associate Agreement）；处理 PHI 时必需 |
| GDPR | "欧盟隐私" | 实时 PII 脱敏才是 2026 年站得住脚的标准 |
| EU AI Act | "欧盟 AI 法规" | 高风险条款 2026 年 8 月执行；1500 万欧元 / 3%（高风险义务）— 3500 万欧元 / 7%（被禁止行为） |
| Colorado AI Act | "美国州级 AI 法" | 2026 年 6 月 30 日生效（因 SB25B-004 推迟）；要求影响评估 |
| ISO 42001 | "AI 治理" | 面向 AI 风险与透明度的新兴框架 |
| ISO 27001 | "安全 ISMS" | 信息安全管理体系基线 |
| 符合性评估 | "欧盟 AI 文档包" | 高风险系统要求：文档、测试、日志记录 |
| 跨框架映射 | "一项控制，多个框架" | 单一策略同时满足多个框架的控制要求 |

## 延伸阅读

- [OpenAI Security and Privacy](https://openai.com/security-and-privacy/) — 参考合规画像。
- [GuardionAI — LLM Compliance 2026: ISO 42001, EU AI Act, SOC 2, GDPR](https://guardion.ai/blog/llm-compliance-guide-iso-42001-eu-ai-act-soc2-gdpr-2026)
- [Dsalta — SOC 2 Type 2 Audit Guide 2026: 10 AI Controls](https://www.dsalta.com/resources/ai-compliance/soc-2-type-2-audit-guide-2026-10-ai-powered-controls-every-saas-team-needs)
- [EU AI Act official text](https://eur-lex.europa.eu/eli/reg/2024/1689/oj) — 一手来源。
- [Colorado AI Act](https://leg.colorado.gov/bills/sb24-205) — 一手来源。
- [ISO/IEC 42001:2023](https://www.iso.org/standard/81230.html) — AI 管理体系标准。
