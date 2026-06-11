# 关系抽取与知识图谱构建

> NER 找到了实体，实体链接为它们锚定了身份，关系抽取则找出实体之间的边。知识图谱就是节点、边与其溯源信息的总和。

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 5 · 06 (NER), Phase 5 · 25 (Entity Linking)
**Time:** ~60 minutes

## 问题背景

一位分析师读到："Tim Cook became CEO of Apple in 2011."（Tim Cook 于 2011 年成为 Apple 的 CEO。）这句话包含四条事实：

- `(Tim Cook, role, CEO)`
- `(Tim Cook, employer, Apple)`
- `(Tim Cook, start_date, 2011)`
- `(Apple, type, Organization)`

关系抽取（Relation Extraction，RE）把自由文本转换成结构化三元组 `(subject, relation, object)`。在整个语料库上聚合，你就得到一个知识图谱；再加上查询能力，你就拥有了一个可供 RAG、分析或合规审计使用的推理基座。

2026 年的问题是：LLM 抽取关系时太"热情"了。它们会幻觉出源文本根本不支持的三元组。没有溯源信息，你无法区分真实三元组和貌似合理的虚构内容。2026 年的答案是 AEVS 风格的"锚定—验证"流水线。

## 核心概念

![Text → triples → knowledge graph](../assets/relation-extraction.svg)

**三元组形式。** `(subject_entity, relation_type, object_entity)`。关系来自一个封闭本体（Wikidata 属性、FIBO、UMLS）或开放集合（OpenIE 风格，什么都行）。

**三种抽取方法。**

1. **规则 / 模式式。** Hearst 模式："X such as Y" → `(Y, isA, X)`，再加手工编写的正则表达式。脆弱、精确、可解释。
2. **监督分类器。** 给定句子中的两个实体提及，从固定集合中预测关系类型。在 TACRED、ACE、KBP 上训练。这是 2015–2022 年的标准做法。
3. **生成式 LLM。** 提示模型直接输出三元组。开箱即用，但需要溯源，否则会幻觉出看似合理的垃圾。

**AEVS（Anchor-Extraction-Verification-Supplement，锚定—抽取—验证—补充，2026）。** 当前的幻觉缓解框架：

- **锚定（Anchor）。** 识别每个实体片段和关系短语片段，并记录精确位置。
- **抽取（Extract）。** 生成与锚定片段关联的三元组。
- **验证（Verify）。** 把每个三元组元素匹配回源文本；拒绝任何无依据的内容。
- **补充（Supplement）。** 用一遍覆盖检查确保没有任何锚定片段被遗漏。

幻觉率大幅下降。需要更多算力，但结果可审计。

**开放与封闭的权衡。**

- **封闭本体。** 固定属性列表（例如 Wikidata 的 11,000+ 个属性）。可预测、可查询、不会凭空发明关系。
- **开放信息抽取（Open IE）。** 任何动词短语都能成为关系。召回率高，精确率低，难以查询。

生产级知识图谱通常混合使用：先用开放信息抽取做发现，再把关系规范化到封闭本体上，然后才合并进主图谱。

## 从零实现

### 第 1 步：基于模式的抽取

```python
PATTERNS = [
    (r"(?P<s>[A-Z]\w+) (?:is|was) (?:a|an|the) (?P<o>[A-Z]?\w+)", "isA"),
    (r"(?P<s>[A-Z]\w+) (?:is|was) born in (?P<o>\w+)", "bornIn"),
    (r"(?P<s>[A-Z]\w+) works? (?:at|for) (?P<o>[A-Z]\w+)", "worksAt"),
    (r"(?P<s>[A-Z]\w+) founded (?P<o>[A-Z]\w+)", "founded"),
]
```

完整的玩具抽取器见 `code/main.py`。Hearst 模式至今仍在领域专用流水线中使用，因为它们易于调试。

### 第 2 步：监督式关系分类

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification

tok = AutoTokenizer.from_pretrained("Babelscape/rebel-large")
model = AutoModelForSequenceClassification.from_pretrained("Babelscape/rebel-large")

text = "Tim Cook was born in Alabama. He later became CEO of Apple."
encoded = tok(text, return_tensors="pt", truncation=True)
output = model.generate(**encoded, max_length=200)
triples = tok.batch_decode(output, skip_special_tokens=False)
```

REBEL 是一个 seq2seq 关系抽取器：文本进，三元组出，且输出直接就是 Wikidata 属性 id。它在远程监督（distant supervision）数据上微调而成，是标准的开放权重基线。

### 第 3 步：带锚定的 LLM 提示式抽取

```python
prompt = f"""Extract (subject, relation, object) triples from the text.
For each triple, include the exact character span in the source text.

Text: {text}

Output JSON:
[{{"subject": {{"text": "...", "span": [start, end]}},
   "relation": "...",
   "object": {{"text": "...", "span": [start, end]}}}}, ...]

Only include triples fully supported by the text. No inference beyond what is stated.
"""
```

把每个返回的片段位置都和源文本核对。凡是 `text[start:end] != triple_entity` 的一律拒绝。这就是 AEVS"验证"步骤的最小形式。

### 第 4 步：规范化到封闭本体

```python
RELATION_MAP = {
    "is the CEO of": "P169",       # "chief executive officer"
    "was born in":   "P19",         # "place of birth"
    "founded":        "P112",       # "founded by" (inverted subject/object)
    "works at":       "P108",       # "employer"
}


def canonicalize(relation):
    rel_low = relation.lower().strip()
    if rel_low in RELATION_MAP:
        return RELATION_MAP[rel_low]
    return None   # drop unmapped open relations or route to manual review
```

规范化（canonicalization）往往占整个工程工作量的 60-80%。请提前为它做好预算。

### 第 5 步：构建一个小图谱并查询

```python
triples = extract(text)
graph = {}
for s, r, o in triples:
    graph.setdefault(s, []).append((r, o))


def neighbors(node, relation=None):
    return [(r, o) for r, o in graph.get(node, []) if relation is None or r == relation]


print(neighbors("Tim Cook", relation="P108"))    # -> [(P108, Apple)]
```

这就是所有基于知识图谱的 RAG 系统的最小原子。要扩展规模，可以使用 RDF 三元组存储（Blazegraph、Virtuoso）、属性图（Neo4j），或带向量增强的图存储。

## 常见陷阱

- **先做指代消解再做关系抽取。** "He founded Apple"——RE 需要知道"he"是谁。先运行指代消解（第 24 课）。
- **实体规范化。** "Apple Inc" 和 "Apple" 必须解析到同一个节点。先做实体链接（第 25 课）。
- **幻觉三元组。** LLM 会输出文本不支持的三元组。必须强制执行片段验证。
- **关系规范化漂移。** 开放信息抽取得到的关系不一致（"was born in"、"came from"、"is a native of"）。必须折叠成规范 id，否则图谱无法查询。
- **时间错误。** "Tim Cook is CEO of Apple"——现在为真，2005 年为假。许多关系有时间边界。使用限定符（Wikidata 中的 `P580` 开始时间、`P582` 结束时间）。
- **领域不匹配。** REBEL 在 Wikipedia 上训练。法律、医学和科学文本通常需要经过领域微调的 RE 模型。

## 生产实践

2026 年的技术栈：

| 场景 | 选择 |
|-----------|------|
| 快速上生产、通用领域 | REBEL 或 LlamaPred + Wikidata 规范化 |
| 特定领域（生物医学、法律） | SciREX 风格的领域微调 + 自定义本体 |
| LLM 提示式、需可审计输出 | AEVS 流水线：锚定 → 抽取 → 验证 → 补充 |
| 大批量新闻信息抽取 | 模式式 + 监督式混合 |
| 从零构建知识图谱 | 开放信息抽取 + 人工规范化环节 |
| 时态知识图谱 | 抽取时附带限定符（开始/结束时间、时间点） |

集成模式：NER → 指代消解 → 实体链接 → 关系抽取 → 本体映射 → 图谱加载。每一个阶段都是潜在的质量关卡。

## 交付产物

保存为 `outputs/skill-re-designer.md`：

```markdown
---
name: re-designer
description: Design a relation extraction pipeline with provenance and canonicalization.
version: 1.0.0
phase: 5
lesson: 26
tags: [nlp, relation-extraction, knowledge-graph]
---

Given a corpus (domain, language, volume) and downstream use (KG-RAG, analytics, compliance), output:

1. Extractor. Pattern-based / supervised / LLM / AEVS hybrid. Reason tied to precision vs recall target.
2. Ontology. Closed property list (Wikidata / domain) or open IE with canonicalization pass.
3. Provenance. Every triple carries source char-span + doc id. Non-negotiable for audit.
4. Merge strategy. Canonical entity id + relation id + temporal qualifiers; dedup policy.
5. Evaluation. Precision / recall on 200 hand-labelled triples + hallucination-rate on LLM-extracted sample.

Refuse any LLM-based RE pipeline without span verification (source provenance). Refuse open-IE output flowing into a production graph without canonicalization. Flag pipelines with no temporal qualifier on time-bounded relations (employer, spouse, position).
```

## 练习

1. **简单。** 在 5 个新闻句子上运行 `code/main.py` 中的模式抽取器，人工核查精确率。
2. **中等。** 在同样的句子上使用 REBEL（或一个小型 LLM），比较两者的三元组。哪个抽取器精确率更高？哪个召回率更高？
3. **困难。** 构建 AEVS 流水线：用 LLM 抽取 + 对照源文本验证片段位置。在 50 个 Wikipedia 风格的句子上，测量验证步骤前后的幻觉率。

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|-----------------|-----------------------|
| 三元组（Triple） | 主语-关系-宾语 | `(s, r, o)` 元组，知识图谱的原子单元。 |
| 开放信息抽取（Open IE） | 什么都能抽 | 开放词表的关系短语；召回率高，精确率低。 |
| 封闭本体（Closed ontology） | 固定模式 | 有界的关系类型集合（Wikidata、UMLS、FIBO）。 |
| 规范化（Canonicalization） | 把一切归一化 | 把表面名称 / 关系映射到规范 id。 |
| AEVS | 有据可依的抽取 | 锚定-抽取-验证-补充流水线（2026）。 |
| 溯源（Provenance） | 事实来源链接 | 每个三元组都带有指向来源的文档 id + 字符位置。 |
| 远程监督（Distant supervision） | 廉价标注 | 将文本与已有知识图谱对齐以生成训练数据。 |

## 延伸阅读

- [Mintz et al. (2009). Distant supervision for relation extraction without labeled data](https://www.aclweb.org/anthology/P09-1113.pdf) —— 远程监督的开山之作。
- [Huguet Cabot, Navigli (2021). REBEL: Relation Extraction By End-to-end Language generation](https://aclanthology.org/2021.findings-emnlp.204.pdf) —— seq2seq 关系抽取的主力模型。
- [Wadden et al. (2019). Entity, Relation, and Event Extraction with Contextualized Span Representations (DyGIE++)](https://arxiv.org/abs/1909.03546) —— 联合信息抽取。
- [AEVS — Anchor-Extraction-Verification-Supplement framework](https://www.mdpi.com/2073-431X/15/3/178) —— 2026 年的幻觉缓解设计。
- [Wikidata SPARQL tutorial](https://www.wikidata.org/wiki/Wikidata:SPARQL_tutorial) —— 标准的图谱查询教程。
