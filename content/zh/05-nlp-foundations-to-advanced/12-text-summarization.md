# 文本摘要

> 抽取式系统告诉你文档说了什么，生成式系统告诉你作者想表达什么。任务不同，陷阱也不同。

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 5 · 02 (BoW + TF-IDF), Phase 5 · 11 (Machine Translation)
**Time:** ~75 minutes

## 问题背景

一篇 2,000 词的新闻文章出现在你的信息流里，你需要 120 个词来概括它。你可以从文章中挑出三个最重要的句子（抽取式），也可以用自己的话改写内容（生成式）。两者都叫摘要，但它们是完全不同的问题。

抽取式摘要（extractive summarization）是一个排序问题：给每个句子打分，返回得分最高的 `k` 个。输出永远合乎语法，因为句子是原封不动搬过来的。风险在于会漏掉分散在全文各处的内容。

生成式摘要（abstractive summarization）是一个生成问题：Transformer 以输入为条件产生新文本。输出流畅且压缩度高，但可能凭空捏造原文中不存在的事实。风险在于一本正经地编造。

这节课两种都要构建，并逐一剖析各自特有的失效模式。

## 核心概念

![Extractive TextRank vs abstractive transformer](../assets/summarization.svg)

**抽取式。** 把文章看作一张图：节点是句子，边是句子间的相似度。在图上运行 PageRank（或类似算法），按每个句子与其他句子的连接紧密程度打分。得分最高的句子就是摘要。经典实现是 **TextRank**（Mihalcea and Tarau, 2004）。

**生成式。** 在「文档-摘要」对上微调一个 Transformer 编码器-解码器（BART、T5、Pegasus）。推理时，模型读入文档，通过交叉注意力逐 token 生成摘要。其中 Pegasus 使用了「缺口句子」（gap-sentence）预训练目标，使它几乎不需要微调就能在摘要任务上表现出色。

评估使用 **ROUGE**（Recall-Oriented Understudy for Gisting Evaluation）。ROUGE-1 和 ROUGE-2 衡量一元和二元 n-gram 的重叠度，ROUGE-L 衡量最长公共子序列。分数越高越好，但 40 的 ROUGE-L 算「不错」，50 就是「出类拔萃」了。每篇论文都会报告这三项。使用 `rouge-score` 包即可。

## 从零实现

### Step 1: TextRank（抽取式）

```python
import math
import re
from collections import Counter


def sentence_split(text):
    return re.split(r"(?<=[.!?])\s+", text.strip())


def similarity(s1, s2):
    w1 = Counter(s1.lower().split())
    w2 = Counter(s2.lower().split())
    intersection = sum((w1 & w2).values())
    denom = math.log(len(w1) + 1) + math.log(len(w2) + 1)
    if denom == 0:
        return 0.0
    return intersection / denom


def textrank(text, top_k=3, damping=0.85, iterations=50, epsilon=1e-4):
    sentences = sentence_split(text)
    n = len(sentences)
    if n <= top_k:
        return sentences

    sim = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if i != j:
                sim[i][j] = similarity(sentences[i], sentences[j])

    scores = [1.0] * n
    for _ in range(iterations):
        new_scores = [1 - damping] * n
        for i in range(n):
            total_out = sum(sim[i]) or 1e-9
            for j in range(n):
                if sim[i][j] > 0:
                    new_scores[j] += damping * sim[i][j] / total_out * scores[i]
        if max(abs(s - ns) for s, ns in zip(scores, new_scores)) < epsilon:
            scores = new_scores
            break
        scores = new_scores

    ranked = sorted(range(n), key=lambda k: scores[k], reverse=True)[:top_k]
    ranked.sort()
    return [sentences[i] for i in ranked]
```

有两点值得点明。相似度函数使用对数归一化的词重叠，这是 TextRank 原始论文中的变体；用 TF-IDF 向量的余弦相似度同样可行。阻尼系数 0.85 和迭代次数都是 PageRank 的默认值。

### Step 2: 用 BART 做生成式摘要

```python
from transformers import pipeline

summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

article = """(long news article text)"""

summary = summarizer(article, max_length=120, min_length=60, do_sample=False)
print(summary[0]["summary_text"])
```

BART-large-CNN 是在 CNN/DailyMail 语料上微调的，开箱即可产出新闻风格的摘要。对于其他领域（科学论文、对话、法律），换用对应的 Pegasus 检查点，或在你的目标数据上微调。

### Step 3: ROUGE 评估

```python
from rouge_score import rouge_scorer

scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
scores = scorer.score(reference_summary, generated_summary)
print({k: round(v.fmeasure, 3) for k, v in scores.items()})
```

务必启用词干提取（stemming）。否则 "running" 和 "run" 会被算作不同的词，导致 ROUGE 低估重叠度。

### 超越 ROUGE（2026 年的摘要评估）

ROUGE 统治摘要评估二十年了，但到 2026 年，仅靠它已经不够。一项针对 NLG 论文的大规模元分析显示：

- **BERTScore**（基于上下文嵌入的相似度）在 2023 年前后逐渐普及，如今大多数摘要论文都会与 ROUGE 一并报告。
- **BARTScore** 把评估当作生成问题：用预训练 BART 在给定原文条件下对摘要赋予的似然来打分。
- **MoverScore**（在上下文嵌入上计算 Earth Mover's Distance）在 2025 年的摘要基准中登顶，因为它比 ROUGE 更能捕捉语义层面的重叠。
- **FactCC** 和**基于 QA 的忠实度评估**在 2021-2023 年很常见，如今往往被 **G-Eval** 取代（一条 GPT-4 提示链，借助思维链推理对连贯性、一致性、流畅度和相关性打分）。
- **G-Eval** 及类似的 LLM 评审方法在评分量规设计良好时，与人类判断的一致率约为 80%。

生产环境的建议：报告 ROUGE-L 以便与历史结果对比，用 BERTScore 衡量语义重叠，用 G-Eval 评估连贯性和事实性。并用 50-100 条人工标注的摘要做校准。

### Step 4: 事实性问题

生成式摘要容易产生幻觉（hallucination）。抽取式摘要的幻觉风险低得多，因为输出是从原文原封不动搬过来的——不过如果原文句子被脱离上下文、信息过时或被打乱顺序引用，它仍可能误导读者。这是生产系统在合规相关内容上仍然偏爱抽取式方法的最大原因。

需要点名的幻觉类型：

- **实体替换。** 原文说 "John Smith"，摘要写成 "John Brown"。
- **数字漂移。** 原文说 "25,000"，摘要写成 "25 million"。
- **极性翻转。** 原文说 "rejected the offer"，摘要写成 "accepted the offer"。
- **事实捏造。** 原文根本没提 CEO，摘要却说 CEO 批准了。

行之有效的评估方法：

- **FactCC。** 一个在原文句子与摘要句子的蕴含关系上训练的二分类器，预测「符合事实/不符合事实」。
- **基于 QA 的事实性。** 向 QA 模型提出答案在原文中的问题；如果摘要支持的是不同的答案，就标记出来。
- **实体级 F1。** 对比原文与摘要中的命名实体，只出现在摘要中的实体值得怀疑。

凡是面向用户且事实性至关重要的场景（新闻、医疗、法律、金融），抽取式都是更安全的默认选择。生成式则需要在流程中加入事实性检查。

## 生产实践

2026 年的技术栈：

| 使用场景 | 推荐方案 |
|---------|-------------|
| 新闻，3-5 句摘要，英文 | `facebook/bart-large-cnn` |
| 科学论文 | `google/pegasus-pubmed` 或微调过的 T5 |
| 多文档、长文本 | 任何 32k+ 上下文的 LLM，配合提示词 |
| 对话摘要 | `philschmid/bart-large-cnn-samsum` |
| 抽取式，结构上天然低幻觉风险 | TextRank 或 `sumy` 的 LSA / LexRank |

2026 年，在算力不受限时，长上下文 LLM 常常胜过专用模型。代价是成本和可复现性；专用模型的输出更稳定一致。

## 交付产物

保存为 `outputs/skill-summary-picker.md`：

```markdown
---
name: summary-picker
description: Pick extractive or abstractive, named library, factuality check.
version: 1.0.0
phase: 5
lesson: 12
tags: [nlp, summarization]
---

Given a task (document type, compliance requirement, length, compute budget), output:

1. Approach. Extractive or abstractive. Explain in one sentence why.
2. Starting model / library. Name it. `sumy.TextRankSummarizer`, `facebook/bart-large-cnn`, `google/pegasus-pubmed`, or an LLM prompt.
3. Evaluation plan. ROUGE-1, ROUGE-2, ROUGE-L (use rouge-score with stemming). Plus factuality check if abstractive.
4. One failure mode to probe. Entity swap is the most common in abstractive news summarization; flag samples where source entities do not appear in summary.

Refuse abstractive summarization for medical, legal, financial, or regulated content without a factuality gate. Flag input over the model's context window as needing chunked map-reduce summarization (not just truncation).
```

## 练习

1. **简单。** 在 5 篇新闻文章上运行 TextRank，把得分前 3 的句子与参考摘要对比，测量 ROUGE-L。在 CNN/DailyMail 风格的文章上，你应该能得到 30-45 的 ROUGE-L。
2. **中等。** 实现实体级事实性检查：用 spaCy 从原文和摘要中抽取命名实体，计算原文实体在摘要中的召回率，以及摘要实体相对原文的精确率。高精确率、低召回率意味着安全但过于简略；低精确率则意味着出现了幻觉实体。
3. **困难。** 在 50 篇 CNN/DailyMail 文章上对比 BART-large-CNN 与一个 LLM（Claude 或 GPT-4）。报告 ROUGE-L、事实性（用实体 F1 衡量）和每条摘要的成本，记录各自在哪些方面胜出。

## 关键术语

| 术语 | 人们常说的 | 实际含义 |
|------|-----------------|-----------------------|
| 抽取式 | 挑句子 | 从原文中原样返回句子，永远不会产生幻觉。 |
| 生成式 | 改写 | 以原文为条件生成新文本，可能产生幻觉。 |
| ROUGE | 摘要指标 | 系统输出与参考摘要之间的 n-gram / LCS 重叠度。 |
| TextRank | 基于图的抽取式方法 | 在句子相似度图上运行 PageRank。 |
| 事实性 | 写得对不对 | 摘要中的论断是否有原文支撑。 |
| 幻觉 | 编出来的内容 | 摘要中没有原文支撑的内容。 |

## 延伸阅读

- [Mihalcea and Tarau (2004). TextRank: Bringing Order into Texts](https://aclanthology.org/W04-3252/) — 抽取式摘要的经典论文。
- [Lewis et al. (2019). BART: Denoising Sequence-to-Sequence Pre-training](https://arxiv.org/abs/1910.13461) — BART 论文。
- [Zhang et al. (2019). PEGASUS: Pre-training with Extracted Gap-sentences](https://arxiv.org/abs/1912.08777) — Pegasus 与缺口句子预训练目标。
- [Lin (2004). ROUGE: A Package for Automatic Evaluation of Summaries](https://aclanthology.org/W04-1013/) — ROUGE 论文。
- [Maynez et al. (2020). On Faithfulness and Factuality in Abstractive Summarization](https://arxiv.org/abs/2005.00661) — 全面梳理事实性问题的论文。
