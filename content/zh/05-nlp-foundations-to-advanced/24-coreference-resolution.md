# 共指消解

> “她给他打了电话。他没有接。那位医生正在吃午饭。”三个指代、两个人物，却没有一个名字。共指消解（Coreference Resolution）就是要弄清楚谁是谁。

**Type:** Learn
**Languages:** Python
**Prerequisites:** Phase 5 · 06 (NER), Phase 5 · 07 (POS & Parsing)
**Time:** ~60 minutes

## 问题背景

从一篇 300 词的文章中抽取所有提及 Apple Inc. 的地方。文章直接写 “Apple” 时很容易；但写成 “the company”、“they”、“Cupertino's technology giant” 或 “Jobs's firm” 时就难了。如果不把这些指称归并到同一个实体，你的 NER 流水线会漏掉 60-80% 的提及。

共指消解把所有指向同一个真实世界实体的表达式连接成一个簇。它是表层 NLP（NER、句法分析）与下游语义任务（信息抽取、问答、摘要、知识图谱）之间的粘合剂。

为什么它在 2026 年依然重要：

- 摘要：“The CEO announced...” 与 “Tim Cook announced...” —— 摘要应当直接给出这位 CEO 的名字。
- 问答：回答 “Who did she call?” 需要先消解 “she”。
- 信息抽取：知识图谱中如果把 “PER1 founded Apple” 和 “Jobs founded Apple” 当作两条独立记录，那就是错的。
- 多文档信息抽取：把多篇文章中关于同一事件的提及合并起来，就是跨文档共指。

## 核心概念

![共指聚类：提及 → 实体](../assets/coref.svg)

**任务定义。** 输入：一篇文档。输出：对提及（文本片段，span）的聚类，每个簇对应一个实体。

**提及类型。**

- **命名实体（Named entity）。** “Tim Cook”
- **名词性提及（Nominal）。** “the CEO”、“the company”
- **代词性提及（Pronominal）。** “he”、“she”、“they”、“it”
- **同位语（Appositive）。** “Tim Cook, Apple's CEO,”

**架构。**

1. **基于规则（Hobbs，1978）。** 利用语法规则在句法树上做代词消解。一个不错的基线，在代词消解上出人意料地难以超越。
2. **提及对分类器（Mention-pair classifier）。** 对每一对提及 (m_i, m_j) 预测它们是否共指，再通过传递闭包聚类。2016 年之前的标准方法。
3. **提及排序（Mention-ranking）。** 对每个提及，给候选先行词（包括“无先行词”）排序，取最高分。
4. **基于片段的端到端方法（Lee et al.，2017）。** Transformer 编码器。枚举所有不超过长度上限的候选片段，预测提及分数，再为每个片段预测先行词概率，最后贪心聚类。现代默认方案。
5. **生成式（2024+）。** 直接提示 LLM：“列出文中每个代词及其先行词。”在简单场景表现不错，但在长文档和罕见指称对象上表现挣扎。

**评估指标。** 共有五个标准指标（MUC、B³、CEAF、BLANC、LEA），因为没有任何单一指标能完全刻画聚类质量。前三个指标的平均值即 CoNLL F1。2026 年在 CoNLL-2012 上的最先进水平约为 83 F1。

**已知难点。**

- 有定描述指向数页之前引入的实体。
- 桥接回指（bridging anaphora）：“the wheels” → 前文提到的一辆车。
- 中文、日文等语言中的零形回指（zero anaphora）。
- 后指（cataphora，代词出现在指称对象之前）：“When **she** walked in, Mary smiled.”

## 从零实现

### 第 1 步：预训练神经共指模型（AllenNLP / spaCy-experimental）

```python
import spacy
nlp = spacy.load("en_coreference_web_trf")   # experimental model
doc = nlp("Apple announced new products. The company said they would ship soon.")
for cluster in doc._.coref_clusters:
    print(cluster, "->", [m.text for m in cluster])
```

在更长的文档上，你会得到类似这样的结果：
- 簇 1：[Apple, The company, they]
- 簇 2：[new products]

### 第 2 步：基于规则的代词消解器（教学用途）

参见 `code/main.py` 中仅用标准库的实现：

1. 抽取提及：命名实体（首字母大写的片段）、代词（词典查找）、有定描述（“the X”）。
2. 对每个代词，回看前 K 个提及，并按以下因素打分：
   - 性别/单复数一致性（启发式）
   - 邻近度（越近越优先）
   - 句法角色（主语优先）
3. 将代词链接到得分最高的先行词。

它无法与神经模型竞争，但能展示搜索空间，以及端到端模型必须做出的那些决策。

### 第 3 步：用 LLM 做共指消解

```python
prompt = f"""Text: {text}

List every pronoun and noun phrase that refers to a person or company.
Cluster them by what they refer to. Output JSON:
[{{"entity": "Apple", "mentions": ["Apple", "the company", "it"]}}, ...]
"""
```

要注意两种失败模式。第一，LLM 会过度合并（把指向两个不同人的 “him” 和 “her” 合到一起）。第二，LLM 在长文档中会悄悄丢掉一些提及。务必通过片段偏移（span-offset）检查来验证。

### 第 4 步：评估

标准的 conll-2012 脚本会计算 MUC、B³、CEAF-φ4 并报告平均值。如果要做内部评估，先在标注好的测试集上计算片段级精确率和召回率，再加上提及链接 F1。

## 常见陷阱

- **单例爆炸。** 有些系统把每个提及都报告成独立的簇。B³ 对此宽容，MUC 会重罚。务必同时检查三个指标。
- **长上下文中的代词。** 文档超过 2,000 个 token 时，性能下降约 15 F1。分块时要小心。
- **性别假设。** 硬编码的性别规则在非二元性别指称、组织机构、动物上会失效。改用学习得到的模型或中性打分。
- **LLM 在长文档上漂移。** 单次 API 调用无法可靠地对跨越 50+ 段落的提及做聚类。使用滑动窗口 + 合并。

## 生产实践

2026 年的技术栈：

| 场景 | 选择 |
|-----------|------|
| 英文、单文档 | `en_coreference_web_trf`（spaCy-experimental）或 AllenNLP 神经共指模型 |
| 多语言 | 在 OntoNotes 或 Multilingual CoNLL 上训练的 SpanBERT / XLM-R |
| 跨文档事件共指 | 专用端到端模型（2025–26 SOTA） |
| 快速 LLM 基线 | GPT-4o / Claude 配合结构化输出的共指提示词 |
| 生产级对话系统 | 规则兜底 + 神经模型为主 + 关键槽位人工复核 |

2026 年真正能上线的集成模式：先跑 NER，再跑共指，然后把共指簇合并进 NER 实体。下游任务看到的是每个簇一个实体，而不是每个提及一个实体。

## 交付产物

保存为 `outputs/skill-coref-picker.md`：

```markdown
---
name: coref-picker
description: Pick a coreference approach, evaluation plan, and integration strategy.
version: 1.0.0
phase: 5
lesson: 24
tags: [nlp, coref, information-extraction]
---

Given a use case (single-doc / multi-doc, domain, language), output:

1. Approach. Rule-based / neural span-based / LLM-prompted / hybrid. One-sentence reason.
2. Model. Named checkpoint if neural.
3. Integration. Order of operations: tokenize → NER → coref → downstream task.
4. Evaluation. CoNLL F1 (MUC + B³ + CEAF-φ4 average) on held-out set + manual cluster review on 20 documents.

Refuse LLM-only coref for documents over 2,000 tokens without sliding-window merge. Refuse any pipeline that runs coref without a mention-level precision-recall report. Flag gender-heuristic systems deployed in demographically diverse text.
```

## 练习

1. **简单。** 在 5 段手工编写的文字上运行 `code/main.py` 中的规则消解器，对照人工标注的真值测量提及链接准确率。
2. **中等。** 在一篇新闻文章上使用预训练神经共指模型，将其输出的簇与你自己的人工标注对比。它在哪些地方失败了？
3. **困难。** 构建一条共指增强的 NER 流水线：先做 NER，再通过共指簇合并。在 100 篇文章上测量相对于纯 NER 的实体覆盖率提升。

## 关键术语

| 术语 | 人们常说的 | 实际含义 |
|------|-----------------|-----------------------|
| 提及（Mention） | 一个指代 | 指向某实体的文本片段（名字、代词、名词短语）。 |
| 先行词（Antecedent） | “it” 指的东西 | 后面的提及所共指的、出现在前面的那个提及。 |
| 簇（Cluster） | 该实体的所有提及 | 全部指向同一个真实世界实体的提及集合。 |
| 回指（Anaphora） | 向前回看的指代 | 后面的提及指向前面的（“he” → “John”）。 |
| 后指（Cataphora） | 向后预指的指代 | 前面的提及指向后面的（“When he arrived, John...”）。 |
| 桥接（Bridging） | 隐式指代 | “I bought a car. The wheels were bad.”（那辆车的轮子。） |
| CoNLL F1 | 排行榜上的那个数字 | MUC、B³、CEAF-φ4 三个 F1 分数的平均值。 |

## 延伸阅读

- [Jurafsky & Martin, SLP3 Ch. 26 — Coreference Resolution and Entity Linking](https://web.stanford.edu/~jurafsky/slp3/26.pdf) —— 权威教材章节。
- [Lee et al. (2017). End-to-end Neural Coreference Resolution](https://arxiv.org/abs/1707.07045) —— 基于片段的端到端方法。
- [Joshi et al. (2020). SpanBERT](https://arxiv.org/abs/1907.10529) —— 提升共指效果的预训练方法。
- [Pradhan et al. (2012). CoNLL-2012 Shared Task](https://aclanthology.org/W12-4501/) —— 基准评测。
- [Hobbs (1978). Resolving Pronoun References](https://www.sciencedirect.com/science/article/pii/0024384178900064) —— 基于规则的经典之作。
