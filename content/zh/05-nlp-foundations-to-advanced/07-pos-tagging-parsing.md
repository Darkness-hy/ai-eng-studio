# 词性标注与句法分析

> 语法研究曾一度无人问津。后来每条 LLM 流水线都需要校验结构化抽取的结果，它又回到了舞台中央。

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 5 · 01 (Text Processing), Phase 2 · 14 (Naive Bayes)
**Time:** ~45 minutes

## 问题背景

第 01 课说过，词形还原（lemmatization）需要词性标签。不知道 `running` 是动词，词形还原器就无法把它还原成 `run`；不知道 `better` 是形容词，就无法还原成 `good`。

这句承诺背后藏着一整个子领域。词性标注（POS tagging）负责给词分配语法类别；句法分析（syntactic parsing）则恢复句子的树状结构：哪个词修饰哪个词，哪个动词支配哪些论元。经典 NLP 花了二十年打磨这两项技术。后来深度学习把它们压缩成了预训练 Transformer 之上的一个 token 分类任务，研究界便转向了别处。

但应用界没有。每条结构化抽取流水线的底层仍在使用 POS 标签和依存树。LLM 生成的 JSON 要用语法约束来校验；问答系统借助依存分析来分解查询；机器翻译质量评估器会检查分析树的对齐情况。

值得一学。本课介绍标签集、基线方法，以及你该在哪个节点停止从零实现、转而调用 spaCy。

## 核心概念

**词性标注（POS tagging）**给每个 token 打上语法类别标签。**Penn Treebank（PTB）**标签集是英语的默认选择。它有 36 个标签，区分粒度细到普通读者会觉得繁琐：`NN` 单数名词、`NNS` 复数名词、`NNP` 单数专有名词、`VBD` 动词过去式、`VBZ` 动词第三人称单数现在式，等等。**Universal Dependencies（UD）**标签集更粗（17 个标签）且与语言无关，已成为跨语言工作的默认标准。

```
The/DET cats/NOUN were/AUX running/VERB at/ADP 3pm/NOUN ./PUNCT
```

**句法分析（syntactic parsing）**产出一棵树。主要有两种风格：

- **成分句法分析（constituency parsing）。**名词短语、动词短语、介词短语层层嵌套。输出是一棵由非终结符类别（NP、VP、PP）构成的树，词是叶子节点。
- **依存句法分析（dependency parsing）。**每个词都依附于唯一的中心词（head），边上标注语法关系。输出是一棵树，每条边都是一个（中心词，依存词，关系）三元组。

依存句法分析在 2010 年代胜出，因为它能干净地泛化到各种语言，尤其是语序自由的语言。

```
running is ROOT
cats is nsubj of running
were is aux of running
at is prep of running
3pm is pobj of at
```

## 从零实现

### 第 1 步：最高频标签基线

能跑起来的最笨的 POS 标注器：对每个词，预测它在训练数据里出现最多的那个标签。

```python
from collections import Counter, defaultdict


def train_mft(train_examples):
    word_tag_counts = defaultdict(Counter)
    all_tags = Counter()
    for tokens, tags in train_examples:
        for token, tag in zip(tokens, tags):
            word_tag_counts[token.lower()][tag] += 1
            all_tags[tag] += 1
    word_best = {w: c.most_common(1)[0][0] for w, c in word_tag_counts.items()}
    default_tag = all_tags.most_common(1)[0][0]
    return word_best, default_tag


def predict_mft(tokens, word_best, default_tag):
    return [word_best.get(t.lower(), default_tag) for t in tokens]
```

在 Brown 语料上，这个基线能达到约 85% 的准确率。不算好，但这是底线——任何正经模型都不应低于它。

### 第 2 步：二元 HMM 标注器

对序列的联合概率建模：

```
P(tags, words) = prod P(tag_i | tag_{i-1}) * P(word_i | tag_i)
```

两张表：转移概率（给定前一个标签的当前标签概率）、发射概率（给定标签的词概率）。两者都用计数加 Laplace 平滑来估计。解码用 Viterbi 算法（在标签格栅上做动态规划）。

```python
import math


def train_hmm(train_examples, alpha=0.01):
    transitions = defaultdict(Counter)
    emissions = defaultdict(Counter)
    tags = set()
    vocab = set()

    for tokens, ts in train_examples:
        prev = "<BOS>"
        for token, tag in zip(tokens, ts):
            transitions[prev][tag] += 1
            emissions[tag][token.lower()] += 1
            tags.add(tag)
            vocab.add(token.lower())
            prev = tag
        transitions[prev]["<EOS>"] += 1

    return transitions, emissions, tags, vocab


def log_prob(table, given, key, smooth_denom, alpha):
    return math.log((table[given].get(key, 0) + alpha) / smooth_denom)


def viterbi(tokens, transitions, emissions, tags, vocab, alpha=0.01):
    tags_list = list(tags)
    n = len(tokens)
    V = [[0.0] * len(tags_list) for _ in range(n)]
    back = [[0] * len(tags_list) for _ in range(n)]

    for j, tag in enumerate(tags_list):
        em_denom = sum(emissions[tag].values()) + alpha * (len(vocab) + 1)
        tr_denom = sum(transitions["<BOS>"].values()) + alpha * (len(tags_list) + 1)
        tr = log_prob(transitions, "<BOS>", tag, tr_denom, alpha)
        em = log_prob(emissions, tag, tokens[0].lower(), em_denom, alpha)
        V[0][j] = tr + em
        back[0][j] = 0

    for i in range(1, n):
        for j, tag in enumerate(tags_list):
            em_denom = sum(emissions[tag].values()) + alpha * (len(vocab) + 1)
            em = log_prob(emissions, tag, tokens[i].lower(), em_denom, alpha)
            best_prev = 0
            best_score = -1e30
            for k, prev_tag in enumerate(tags_list):
                tr_denom = sum(transitions[prev_tag].values()) + alpha * (len(tags_list) + 1)
                tr = log_prob(transitions, prev_tag, tag, tr_denom, alpha)
                score = V[i - 1][k] + tr + em
                if score > best_score:
                    best_score = score
                    best_prev = k
            V[i][j] = best_score
            back[i][j] = best_prev

    last_best = max(range(len(tags_list)), key=lambda j: V[n - 1][j])
    path = [last_best]
    for i in range(n - 1, 0, -1):
        path.append(back[i][path[-1]])
    return [tags_list[j] for j in reversed(path)]
```

二元 HMM 在 Brown 语料上能达到约 93% 的准确率。从 85% 到 93% 的提升主要来自转移概率——模型学到了 `DET NOUN` 很常见而 `NOUN DET` 很罕见。

### 第 3 步：为什么现代标注器更胜一筹

转移概率加发射概率都是局部的。它们无法捕捉这样的现象：`saw` 在 "I bought a saw" 里是名词，在 "I saw the movie" 里却是动词。带任意特征（后缀、词形、前后词、词本身）的 CRF 能达到约 97%；BiLSTM-CRF 或 Transformer 能达到 98% 以上。

这个任务的上限由标注者分歧决定。人类标注者在 Penn Treebank 上的一致率约为 97%。超过 98% 的模型很可能是在过拟合测试集。

### 第 4 步：依存句法分析速览

从零实现完整的依存句法分析超出了本课范围；权威的教科书讲解见 Jurafsky 和 Martin。需要了解两个经典流派：

- **基于转移（transition-based）**的分析器（arc-eager、arc-standard）像移进-归约（shift-reduce）分析器一样工作：读入 token，移进栈中，再执行创建依存弧的归约动作。贪心解码非常快。经典实现是 MaltParser；现代神经版本是 Chen 和 Manning 的基于转移的分析器。
- **基于图（graph-based）**的分析器（Eisner 算法、Dozat-Manning biaffine）为每条可能的中心词-依存词边打分，然后取最大生成树。更慢但更准。

对于绝大多数应用工作，直接调用 spaCy：

```python
import spacy

nlp = spacy.load("en_core_web_sm")
doc = nlp("The cats were running at 3pm.")
for token in doc:
    print(f"{token.text:10s} tag={token.tag_:5s} pos={token.pos_:6s} dep={token.dep_:10s} head={token.head.text}")
```

```
The        tag=DT    pos=DET    dep=det        head=cats
cats       tag=NNS   pos=NOUN   dep=nsubj      head=running
were       tag=VBD   pos=AUX    dep=aux        head=running
running    tag=VBG   pos=VERB   dep=ROOT       head=running
at         tag=IN    pos=ADP    dep=prep       head=running
3pm        tag=NN    pos=NOUN   dep=pobj       head=at
.          tag=.     pos=PUNCT  dep=punct      head=running
```

从下往上读 `dep` 一列，句子的语法结构便一目了然。

## 生产实践

每个生产级 NLP 库都把 POS 标注器和依存分析器作为标准流水线的一部分发布。

- **spaCy**（`en_core_web_sm` / `md` / `lg` / `trf`）。快速、准确，与分词 + NER + 词形还原一体集成。`token.tag_`（Penn）、`token.pos_`（UD）、`token.dep_`（依存关系）。
- **Stanford NLP (stanza)**。Stanford 对 CoreNLP 的继任者。在 60 多种语言上达到业界领先水平。
- **trankit**。基于 Transformer，UD 准确率出色。
- **NLTK**。`pos_tag`。能用，但慢且偏旧。教学用没问题。

### 2026 年这些技术依然重要的场景

- **词形还原。**第 01 课的正确词形还原离不开 POS。永远如此。
- **LLM 输出的结构化抽取。**校验生成的句子是否满足语法约束（如主谓一致、必需的修饰语）。
- **基于方面的情感分析。**依存分析告诉你哪个形容词修饰哪个名词。
- **查询理解。**"movies directed by Wes Anderson starring Bill Murray" 通过句法分析分解为结构化约束。
- **跨语言迁移。**UD 标签和依存关系与语言无关，可以对新语言做零样本结构化分析。
- **低算力流水线。**如果你部署不了 Transformer，POS + 依存分析 + 词表（gazetteer）能让你走得出乎意料地远。

## 交付产物

保存为 `outputs/skill-grammar-pipeline.md`：

```markdown
---
name: grammar-pipeline
description: Design a classical POS + dependency pipeline for a downstream NLP task.
version: 1.0.0
phase: 5
lesson: 07
tags: [nlp, pos, parsing]
---

Given a downstream task (information extraction, rewrite validation, query decomposition, lemmatization), you output:

1. Tagset to use. Penn Treebank for English-only legacy pipelines, Universal Dependencies for multilingual or cross-lingual.
2. Library. spaCy for most production, stanza for academic-grade multilingual, trankit for highest UD accuracy. Name the specific model ID.
3. Integration pattern. Show the 3-5 lines that call the library and consume the needed attributes (`.pos_`, `.dep_`, `.head`).
4. Failure mode to test. Noun-verb ambiguity (`saw`, `book`, `can`) and PP-attachment ambiguity are the classical traps. Sample 20 outputs and eyeball.

Refuse to recommend rolling your own parser. Building parsers from scratch is a research project, not an application task. Flag any pipeline that consumes POS tags without handling lowercase/uppercase variants as fragile.
```

## 练习

1. **简单。**在一个小型已标注语料（如 NLTK 的 Brown 子集）上运行最高频标签基线，在留出句子上测量准确率，验证约 85% 的结果。
2. **中等。**训练上面的二元 HMM，报告每个标签的精确率/召回率。HMM 最容易混淆哪些标签？
3. **困难。**用 spaCy 的依存分析从 1000 句样本中抽取主-谓-宾三元组，在 50 条人工标注的三元组上评估。记录抽取失败的场景（常见于被动句、并列结构和省略主语）。

## 关键术语

| 术语 | 大家怎么说 | 实际含义 |
|------|-----------------|-----------------------|
| POS 标签 | 词的类型 | 语法类别。PTB 有 36 个；UD 有 17 个。 |
| Penn Treebank | 标准标签集 | 英语专用。细粒度区分动词时态和名词单复数。 |
| Universal Dependencies | 多语言标签集 | 比 PTB 更粗；语言中立；跨语言工作的默认选择。 |
| 依存分析 | 句子的树 | 每个词有唯一中心词，每条边带一个语法关系。 |
| Viterbi | 动态规划 | 在给定发射和转移概率下找出概率最高的标签序列。 |

## 延伸阅读

- [Jurafsky and Martin — Speech and Language Processing, chapters 8 and 18](https://web.stanford.edu/~jurafsky/slp3/) —— POS 与句法分析的权威教科书讲解。
- [Universal Dependencies project](https://universaldependencies.org/) —— 所有多语言分析器都在使用的跨语言标签集与树库集合。
- [spaCy linguistic features guide](https://spacy.io/usage/linguistic-features) —— `Token` 上每个属性的实用参考。
- [Chen and Manning (2014). A Fast and Accurate Dependency Parser using Neural Networks](https://nlp.stanford.edu/pubs/emnlp2014-depparser.pdf) —— 把神经网络分析器带入主流的论文。
