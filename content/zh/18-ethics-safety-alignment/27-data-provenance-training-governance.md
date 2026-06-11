# 数据溯源与训练数据治理

> EU AI Act 要求 GPAI 在 2025 年 8 月前支持机器可读的退出（opt-out）标准（依据 EU Copyright Directive 的 TDM 例外条款）。加州 AB 2013 法案（2024 年签署）——生成式 AI 训练数据透明度法，要求开发者发布包含 12 个法定字段的数据集摘要。2025 年各数据保护机构（DPA）在合法利益问题上趋于一致：爱尔兰 DPC（2025 年 5 月 21 日）在 EDPB 意见出台后，接受 Meta 在配套保障措施下使用第一方公开的 EU/EEA 成年用户内容训练 LLM；科隆高等地区法院（2025 年 5 月 23 日）驳回禁令申请；汉堡 DPA 撤销紧急程序；英国 ICO（2025 年 9 月 23 日）对 LinkedIn 的 AI 训练保障措施（透明度、简化的退出机制、延长的异议窗口）给出积极的监管回应并持续监督——这并非正式批准。巴西 ANPD（2024 年 7 月 2 日）以信息透明度不足为由暂停了 Meta 的数据处理；在 Meta 提交合规计划后，该预防性措施于 2024 年 8 月 30 日解除。关键的不可逆性问题：cookie 同意框架是为实时、可逆的追踪设计的；数据一旦进入模型权重，就无法做外科手术式的删除——对于已训练的神经网络，GDPR 的被遗忘权（right to erasure）没有实际可行的对应手段。合规窗口在数据采集时。Data Provenance Initiative（dataprovenance.org，Longpre、Mahari、Lee 等人，"Consent in Crisis"，2024 年 7 月）：大规模审计显示，随着出版方不断添加 robots.txt 限制，AI 数据公地正在快速萎缩。

**Type:** Learn
**Languages:** Python (stdlib, 12-field California AB 2013 scaffolding generator)
**Prerequisites:** Phase 18 · 24 (regulatory), Phase 18 · 26 (cards)
**Time:** ~60 minutes

## 学习目标

- 描述加州 AB 2013 法案为生成式 AI 训练数据透明度规定的 12 个法定字段。
- 陈述 2025 年各数据保护机构（爱尔兰 DPC、英国 ICO、汉堡、科隆）对基于合法利益的 LLM 训练的立场。
- 描述不可逆性问题：为什么 GDPR 的被遗忘权对已训练的神经网络没有实际可行的对应手段。
- 陈述 Data Provenance Initiative 的"Consent in Crisis"研究发现。

## 问题背景

训练数据治理是每一份模型卡（第 26 课）和每一项监管义务（第 24 课）的上游环节。2024-2025 年间，监管格局收敛到三项原则：退出（opt-out）基础设施、逐数据集披露，以及对公开可得数据的合法利益豁免安排。在数据采集时不合规的提供方，事后无法在下游补救。

## 核心概念

### 加州 AB 2013 法案

2024 年签署。对于 2022 年 1 月 1 日或之后发布的系统，相关文档必须在 2026 年 1 月 1 日或之前公布。第 3111(a) 条要求开发者发布训练所用数据集的高层级摘要，包含 12 个法定条目：
1. 数据集的来源或所有者。
2. 说明数据集如何服务于 AI 系统的预期用途。
3. 数据集中数据点的数量（可使用大致范围；动态数据集可使用估计值）。
4. 数据点类型的描述（有标注数据集说明标签类型；无标注数据集说明总体特征）。
5. 数据集是否包含受版权、商标或专利保护的数据，或是否完全属于公有领域。
6. 数据集是购买的还是经许可获得的。
7. 数据集是否包含个人信息（依据 Cal. Civ. Code §1798.140(v)）。
8. 数据集是否包含聚合消费者信息（依据 Cal. Civ. Code §1798.140(b)）。
9. 开发者所做的清洗、处理或其他修改，及其预期目的。
10. 数据采集的时间段，若采集仍在持续须予以说明。
11. 数据集在开发过程中首次使用的日期。
12. 系统是否使用或持续使用合成数据生成。

相对于 Gebru 等人 2018 年的 datasheets，第 12 项（合成数据）是新增的。第 7 项（个人信息）会触发《加州隐私权法案》（CPRA）下的义务。该法规豁免了安全/完整性系统、航空器运行系统，以及仅供联邦使用的国家安全系统（第 3111(b) 条）。

### EU AI Act（第 24 课）与 TDM 退出机制

EU Copyright Directive 的文本与数据挖掘（text-and-data-mining，TDM）例外条款允许在公开可得内容上进行训练，除非权利人选择退出。EU AI Act 的 GPAI Code of Practice 版权章节要求 GPAI 提供方遵守机器可读的退出信号（robots.txt、C2PA "No AI Training" 声明等）。

### 2025 年各 DPA 在合法利益上的趋同

爱尔兰 DPC（2025 年 5 月 21 日）：在 EDPB 意见出台后，接受 Meta 在配套保障措施下使用第一方公开的 EU/EEA 成年用户内容进行训练的方案。科隆高等地区法院（2025 年 5 月 23 日）驳回针对 Meta 的禁令申请：退出机制已经足够。汉堡 DPA 为保持欧盟范围内的一致性，撤销紧急程序。英国 ICO（2025 年 9 月 23 日）对 LinkedIn 在类似保障措施下恢复 AI 训练给出积极的监管回应——并非正式批准——并持续监督。

趋同的原则是：合法利益可以为在公开可得的第一方内容上训练提供正当性，前提是提供退出机制。无需获得同意。

### 巴西 ANPD（2024 年 6 月）

以信息透明度不足为由，暂停了 Meta 将巴西用户数据用于 AI 训练的处理活动。结果与欧盟各 DPA 不同——ANPD 将透明度置于合法利益的可受理性之上。

### 不可逆性问题

cookie 同意机制是为实时、可逆的追踪设计的。训练数据则不同：数据一旦进入模型权重，就无法做外科手术式的删除。从头重新训练是唯一彻底的补救手段，而其成本高得令人望而却步。

部分补救手段：
- **遗忘学习（Unlearning）。** 近似删除；通过成员推理攻击（MIA，第 22 课）来度量。
- **基于影响函数的定位。** 识别受该数据影响最大的权重，进行选择性更新。
- **微调抑制。** 训练模型拒绝输出源自该数据的内容。

没有任何一种能完全解决问题。合规窗口在数据采集时。

### Data Provenance Initiative

dataprovenance.org。Longpre、Mahari、Lee 等人的"Consent in Crisis"（2024 年 7 月）：对 AI 训练数据公地的大规模审计。发现：出版方正在以加速的节奏添加 robots.txt 限制。可公开用于训练的数据公地正在快速萎缩。2023 至 2024 年间，约 25% 的头部训练数据来源添加了某种形式的限制。其含义是：未来训练数据的可得性取决于新的获取范式（授权许可、合成数据生成、激励式参与）。

### 在 Phase 18 中的位置

第 26 课是模型层面的文档化。第 27 课是数据集层面的治理。二者共同构成透明度层。第 28 课则梳理研究这些问题的学术生态。

## 生产实践

`code/main.py` 为一个玩具数据集生成符合加州 AB 2013 法案的 12 字段数据集摘要脚手架。你可以填写各字段，并观察哪些字段会触发后续的隐私或版权义务。

## 交付产物

本课产出 `outputs/skill-provenance-check.md`。给定一个用于训练的数据集，它会检查 AB 2013 的 12 字段覆盖情况、退出基础设施的合规性、与各 DPA 立场的一致性，以及不可逆性风险评估。

## 练习

1. 运行 `code/main.py`。为一个玩具数据集生成 12 字段摘要，并找出哪些字段说明不够充分。

2. EU Copyright Directive 的 TDM 退出机制是机器可读的。为该退出信号提出一种标准格式，并将其与 robots.txt 和 C2PA "No AI Training" 进行比较。

3. 阅读 Data Provenance Initiative 的"Consent in Crisis"（2024 年 7 月）。描述限制增长最快的三个内容类别，并论证其中一项经济后果。

4. 2025 年各 DPA 的趋同立场接受以合法利益为基础进行公开内容训练。构造一个合法利益不足以成立的场景，并指出提供方在该场景下需要改用的法律依据。

5. 草拟一份训练数据溯源清单（manifest），使其既能与 AB 2013 的各字段组合，又能为每个数据集提供 C2PA 签名的溯源链。指出一个技术障碍和一个法律障碍。

## 关键术语

| 术语 | 通俗说法 | 实际含义 |
|------|-----------------|------------------------|
| AB 2013 | "加州那部法律" | 生成式 AI 训练数据透明度法；12 个法定字段 |
| TDM 例外 | "文本与数据挖掘" | EU Copyright Directive 中带退出机制的训练数据例外条款 |
| 合法利益 | "欧盟那个依据" | GDPR 第 6 条规定的法律基础，可为公开内容训练提供正当性 |
| 退出信号 | "机器可读的禁止训练标记" | robots.txt、C2PA "No AI Training"、TDM.Reservation |
| 不可逆性 | "无法反向训练" | 进入模型权重的数据无法做外科手术式删除 |
| 遗忘学习 | "近似删除" | 训练后的干预手段，用于降低模型对特定数据的依赖 |
| Consent in Crisis | "DPI 那份审计" | 2024 年 7 月发现 robots.txt 限制正在加速增长 |

## 延伸阅读

- [California AB 2013](https://leginfo.legislature.ca.gov/faces/billNavClient.xhtml?bill_id=202320240AB2013) — 生成式 AI 训练数据透明度法
- [EU AI Act + GPAI Code of Practice (Lesson 24)](https://digital-strategy.ec.europa.eu/en/policies/regulatory-framework-ai) — 版权章节
- [Longpre, Mahari, Lee et al. — Consent in Crisis (dataprovenance.org, July 2024)](https://www.dataprovenance.org/consent-in-crisis-paper) — DPI 审计报告
- [IAPP — EU Digital Omnibus GDPR amendments (2025)](https://iapp.org/news/a/eu-digital-omnibus-amendments-to-gdpr-to-facilitate-ai-training-miss-the-mark) — 监管背景
