# 实体链接与消歧

> NER 找到了 "Paris"。实体链接要决定：是法国巴黎？Paris Hilton？德州 Paris？还是特洛伊王子 Paris？没有链接这一步，你的知识图谱就始终是模糊的。

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 5 · 06 (NER), Phase 5 · 24 (Coreference Resolution)
**Time:** ~60 minutes

## 问题背景

有这样一句话："Jordan beat the press."（Jordan 突破了紧逼防守。）你的 NER 把 "Jordan" 标成了 PERSON。很好。但到底是*哪个* Jordan？

- Michael Jordan（篮球运动员）？
- Michael B. Jordan（演员）？
- Michael I. Jordan（伯克利的机器学习教授——没错，这个混淆在 ML 论文里真实存在）？
- Jordan（约旦这个国家）？
- Jordan（希伯来语人名）？

实体链接（Entity Linking，EL）把每个提及（mention）解析到知识库中的唯一条目：Wikidata、Wikipedia、DBpedia，或你自己的领域知识库。它包含两个子任务：

1. **候选生成。** 给定 "Jordan"，哪些知识库条目是可能的？
2. **消歧。** 给定上下文，哪个候选才是正确的那个？

这两步都可以学习，也都有基准测试。这套组合流水线十年来一直很稳定——变化的只是消歧器的质量。

## 核心概念

![Entity linking pipeline: mention → candidates → disambiguated entity](../assets/entity-linking.svg)

**候选生成。** 给定提及的表层形式（"Jordan"），在别名索引中查找候选。Wikipedia 的别名词典覆盖了绝大多数命名实体："JFK" → John F. Kennedy、Jacqueline Kennedy、JFK 机场、电影《JFK》。典型的索引每个提及返回 10-30 个候选。

**消歧：三种方法。**

1. **先验 + 上下文（Milne & Witten, 2008）。** `P(entity | mention) × context-similarity(entity, text)`。效果不错、速度快、无需训练。
2. **基于嵌入（ESS / REL / Blink）。** 编码提及 + 上下文，再编码每个候选的描述，取余弦相似度最大者。这是 2020-2024 年的默认方案。
3. **生成式（GENRE, 2021；基于 LLM, 2023+）。** 逐 token 解码实体的规范名称。解码被约束在一棵由合法实体名构成的字典树（trie）上，保证输出一定是合法的知识库 id。

**端到端 vs 流水线。** 现代模型（ELQ、BLINK、ExtEnD、GENRE）在一次前向中完成 NER + 候选生成 + 消歧。但流水线系统在生产中仍占主导，因为各个组件可以单独替换。

### 两个关键指标

- **提及召回率（候选生成阶段）。** 正确知识库条目出现在候选列表中的标注提及所占的比例。它是整条流水线的上限地板。
- **消歧准确率 / F1。** 在候选正确的前提下，top-1 命中的频率。

两个指标都要报告。一个在 80% 候选召回率上达到 99% 消歧准确率的系统，整体只是一条 80% 的流水线。

## 从零实现

### 第 1 步：用 Wikipedia 重定向构建别名索引

```python
alias_to_entities = {
    "jordan": ["Q41421 (Michael Jordan)", "Q810 (Jordan, country)", "Q254110 (Michael B. Jordan)"],
    "paris":  ["Q90 (Paris, France)", "Q663094 (Paris, Texas)", "Q55411 (Paris Hilton)"],
    "apple":  ["Q312 (Apple Inc.)", "Q89 (apple, fruit)"],
}
```

Wikipedia 别名数据约有 1800 万个（别名，实体）对。可从 Wikidata dumps 下载，以倒排索引形式存储。

### 第 2 步：基于上下文的消歧

```python
def disambiguate(mention, context, alias_index, entity_desc):
    candidates = alias_index.get(mention.lower(), [])
    if not candidates:
        return None, 0.0
    context_words = set(tokenize(context))
    best, best_score = None, -1
    for entity_id in candidates:
        desc_words = set(tokenize(entity_desc[entity_id]))
        union = len(context_words | desc_words)
        score = len(context_words & desc_words) / union if union else 0.0
        if score > best_score:
            best, best_score = entity_id, score
    return best, best_score
```

这里的 Jaccard 重叠只是个玩具实现。应换成基于嵌入的余弦相似度（transformer 版本见 `code/main.py` 的 step-2）。

### 第 3 步：基于嵌入的方法（BLINK 风格）

```python
from sentence_transformers import SentenceTransformer
encoder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

def embed_mention(text, mention_span):
    start, end = mention_span
    marked = f"{text[:start]} [MENTION] {text[start:end]} [/MENTION] {text[end:]}"
    return encoder.encode([marked], normalize_embeddings=True)[0]

def embed_entity(entity_id, description):
    return encoder.encode([f"{entity_id}: {description}"], normalize_embeddings=True)[0]
```

建索引时，对知识库中每个实体只编码一次。查询时，对提及 + 上下文编码一次，与候选池做点积，取最大值。

### 第 4 步：生成式实体链接（概念）

GENRE 逐字符解码实体的 Wikipedia 标题。约束解码（见第 20 课）确保只能输出合法标题，并与知识库支撑的字典树紧密集成。它的现代后继是 REL-GEN 以及带结构化输出的 LLM 提示式 EL。

```python
prompt = f"""Text: {text}
Mention: {mention}
List the best Wikipedia title for this mention.
Respond with JSON: {{"title": "..."}}"""
```

配合白名单（Outlines 的 `choice`），这是 2026 年最容易上线的 EL 流水线。

### 第 5 步：在 AIDA-CoNLL 上评估

AIDA-CoNLL 是标准的 EL 基准：1,393 篇路透社文章、3.4 万个提及、Wikipedia 实体。需报告知识库内准确率（`P@1`）和知识库外的 NIL 检测率。

## 常见陷阱

- **NIL 处理。** 有些提及不在知识库中（新兴实体、冷门人物）。系统必须预测 NIL，而不是硬猜一个错误实体。这要单独度量。
- **提及边界错误。** 上游 NER 漏掉部分跨度（"Bank of America" 只被标成 "Bank"），EL 召回率随之下降。
- **流行度偏差。** 训练出来的系统会过度预测高频实体。ML 论文里提到的 "Michael I. Jordan" 经常被链接到打篮球的 Jordan。
- **跨语言 EL。** 把中文文本中的提及映射到英文 Wikipedia 实体，需要多语言编码器或翻译步骤。
- **知识库过期。** 新公司、新事件、新人物不在去年的 Wikipedia dump 里。生产流水线需要一个定期刷新机制。

## 生产实践

2026 年的技术选型：

| 场景 | 选择 |
|-----------|------|
| 通用英文 + Wikipedia | BLINK 或 REL |
| 跨语言，知识库为 Wikipedia | mGENRE |
| LLM 友好、每天提及量少 | 用候选列表 + 约束 JSON 提示 Claude/GPT-4 |
| 领域知识库（医疗、法律） | 定制 BERT + 知识库感知检索，并在领域 AIDA 风格数据集上微调 |
| 极低延迟 | 仅用精确匹配先验（Milne-Witten 基线） |
| 研究 SOTA | GENRE / ExtEnD / 生成式 LLM-EL |

2026 年能上线的生产模式：NER → 共指消解 → 对每个提及做 EL → 把共指簇收敛为每簇一个规范实体。输出：文档中每个实体一个知识库 id，而不是每个提及一个。

## 交付产物

保存为 `outputs/skill-entity-linker.md`：

```markdown
---
name: entity-linker
description: Design an entity linking pipeline — KB, candidate generator, disambiguator, evaluation.
version: 1.0.0
phase: 5
lesson: 25
tags: [nlp, entity-linking, knowledge-graph]
---

Given a use case (domain KB, language, volume, latency budget), output:

1. Knowledge base. Wikidata / Wikipedia / custom KB. Version date. Refresh cadence.
2. Candidate generator. Alias-index, embedding, or hybrid. Target mention recall @ K.
3. Disambiguator. Prior + context, embedding-based, generative, or LLM-prompted.
4. NIL strategy. Threshold on top score, classifier, or explicit NIL candidate.
5. Evaluation. Mention recall @ 30, top-1 accuracy, NIL-detection F1 on held-out set.

Refuse any EL pipeline without a mention-recall baseline (you cannot evaluate a disambiguator without knowing candidate gen surfaced the right entity). Refuse any pipeline using LLM-prompted EL without constrained output to valid KB ids. Flag systems where popularity bias affects minority entities (e.g. name-clashes) without domain fine-tuning.
```

## 练习

1. **简单。** 在 `code/main.py` 中实现先验+上下文消歧器，应用于 10 个歧义提及（Paris、Jordan、Apple）。手工标注正确实体，测量准确率。
2. **中等。** 用 sentence transformer 编码 50 个歧义提及，并嵌入每个候选的描述。比较基于嵌入的消歧与 Jaccard 上下文重叠的效果。
3. **困难。** 构建一个含 1000 个实体的领域知识库（例如公司里的员工 + 产品）。端到端实现 NER + EL，在 100 条留出句子上测量精确率和召回率。

## 关键术语

| 术语 | 大家怎么说 | 实际含义 |
|------|-----------------|-----------------------|
| 实体链接（EL） | 链接到 Wikipedia | 把一个提及映射到知识库中的唯一条目。 |
| 候选生成 | 可能是谁？ | 为一个提及返回一份可能的知识库条目候选名单。 |
| 消歧 | 选对的那个 | 用上下文给候选打分，选出胜者。 |
| 别名索引 | 查找表 | 从表层形式映射到候选实体。 |
| NIL | 不在知识库里 | 显式预测没有任何知识库条目匹配。 |
| KB | 知识库 | Wikidata、Wikipedia、DBpedia，或你的领域知识库。 |
| AIDA-CoNLL | 那个基准 | 1,393 篇带标准实体链接标注的路透社文章。 |

## 延伸阅读

- [Milne, Witten (2008). Learning to Link with Wikipedia](https://www.cs.waikato.ac.nz/~ihw/papers/08-DM-IHW-LearningToLinkWithWikipedia.pdf) —— 奠基性的先验+上下文方法。
- [Wu et al. (2020). Zero-shot Entity Linking with Dense Entity Retrieval (BLINK)](https://arxiv.org/abs/1911.03814) —— 基于嵌入的主力方案。
- [De Cao et al. (2021). Autoregressive Entity Retrieval (GENRE)](https://arxiv.org/abs/2010.00904) —— 带约束解码的生成式 EL。
- [Hoffart et al. (2011). Robust Disambiguation of Named Entities in Text (AIDA)](https://www.aclweb.org/anthology/D11-1072.pdf) —— 基准数据集论文。
- [REL: An Entity Linker Standing on the Shoulders of Giants (2020)](https://arxiv.org/abs/2006.01969) —— 开源的生产级技术栈。
