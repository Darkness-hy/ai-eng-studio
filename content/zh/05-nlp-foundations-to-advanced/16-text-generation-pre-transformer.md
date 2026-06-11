# Transformer 之前的文本生成 —— N-gram 语言模型

> 如果一个词让模型感到意外，说明模型不够好。困惑度把"意外程度"变成一个数字，而平滑让它保持有限。

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 5 · 01 (Text Processing), Phase 2 · 14 (Naive Bayes)
**Time:** ~45 minutes

## 问题背景

在 Transformer 之前、在 RNN 之前、在词嵌入之前，语言模型靠统计一个词跟在前 `n-1` 个词后面出现的次数来预测下一个词。统计发现 "the cat" 后面接 "sat" 出现了 47 次，接 "jumped" 出现了 12 次，接 "refrigerator" 出现了 0 次。归一化之后就得到一个概率分布。

这就是 n-gram 语言模型。从 1980 年到 2015 年，每一个语音识别器、每一个拼写检查器、每一个基于短语的机器翻译系统都靠它驱动。今天当你需要低成本的端侧语言建模时，它仍然在用。

真正有意思的问题是如何处理没见过的 n-gram。纯基于计数的模型会给任何没见过的序列赋零概率，这是灾难性的，因为句子很长，几乎每个长句子都至少包含一个未见过的序列。五十年的平滑（smoothing）研究解决了这个问题。Kneser-Ney 平滑是这条路线的最终成果，而现代深度学习继承了它的经验主义传统。

## 核心概念

![N-gram model: count, smooth, generate](../assets/ngram.svg)

**N-gram 概率：** `P(w_i | w_{i-n+1}, ..., w_{i-1})`。固定 `n`（通常三元组取 3，四元组取 4），由计数得到：

```text
P(w | context) = count(context, w) / count(context)
```

**零计数问题。** 任何训练中没出现过的 n-gram 概率都是零。2007 年一项基于 Brown 语料库的研究发现，即使是 4-gram 模型，留出集中也有 30% 的 4-gram 在训练集中从未出现。不做平滑，你根本无法在任何真实文本上做评估。

**平滑方法，按精巧程度排序：**

1. **Laplace（加一平滑）。** 给每个计数加 1。简单，但在稀有事件上表现很差。
2. **Good-Turing。** 基于"频率的频率"，把高频事件的概率质量重新分配给未见事件。
3. **插值（Interpolation）。** 用可调权重组合 n-gram、(n-1)-gram 等各阶估计。
4. **回退（Backoff）。** 如果 n-gram 计数为零，就回退到 (n-1)-gram。Katz backoff 对此做了归一化。
5. **绝对折扣（Absolute discounting）。** 从所有计数中减去一个固定折扣 `D`，再把省下的质量分给未见事件。
6. **Kneser-Ney。** 绝对折扣，外加对低阶模型的一个巧妙选择：用*延续概率*（continuation probability，即一个词出现在多少种上下文中）替代原始词频。

Kneser-Ney 的洞察非常深刻。"San Francisco" 是常见的二元组。一元组 "Francisco" 几乎只出现在 "San" 之后。朴素的绝对折扣会给 "Francisco" 很高的一元组概率（因为它的计数很高）。Kneser-Ney 注意到 "Francisco" 只出现在一种上下文中，于是相应地降低它的延续概率。结果是：任何以 "Francisco" 结尾的新二元组都会得到恰当的低概率。

**评估：困惑度（perplexity）。** 在留出测试集上，每个词平均负对数似然的指数。越低越好。困惑度为 100 意味着模型的困惑程度，相当于在 100 个词中均匀随机选择。

```text
perplexity = exp(- (1/N) * Σ log P(w_i | context_i))
```

```figure
ngram-backoff
```

## 从零实现

### 第 1 步：三元组计数

```python
from collections import Counter, defaultdict


def train_ngram(corpus_tokens, n=3):
    ngrams = Counter()
    contexts = Counter()
    for sentence in corpus_tokens:
        padded = ["<s>"] * (n - 1) + sentence + ["</s>"]
        for i in range(len(padded) - n + 1):
            ctx = tuple(padded[i:i + n - 1])
            word = padded[i + n - 1]
            ngrams[ctx + (word,)] += 1
            contexts[ctx] += 1
    return ngrams, contexts


def raw_probability(ngrams, contexts, context, word):
    ctx = tuple(context)
    if contexts.get(ctx, 0) == 0:
        return 0.0
    return ngrams.get(ctx + (word,), 0) / contexts[ctx]
```

输入是分好词的句子列表。输出是 n-gram 计数和上下文计数。`<s>` 和 `</s>` 是句子边界标记。

### 第 2 步：Laplace 平滑

```python
def laplace_probability(ngrams, contexts, vocab_size, context, word):
    ctx = tuple(context)
    numerator = ngrams.get(ctx + (word,), 0) + 1
    denominator = contexts.get(ctx, 0) + vocab_size
    return numerator / denominator
```

给每个计数加 1。能平滑，但分给未见事件的质量过多，连带损害了那些罕见但确实出现过的事件。

### 第 3 步：Kneser-Ney（二元组，插值版）

```python
def kneser_ney_bigram_model(corpus_tokens, discount=0.75):
    unigrams = Counter()
    bigrams = Counter()
    unigram_contexts = defaultdict(set)

    for sentence in corpus_tokens:
        padded = ["<s>"] + sentence + ["</s>"]
        for i, w in enumerate(padded):
            unigrams[w] += 1
            if i > 0:
                prev = padded[i - 1]
                bigrams[(prev, w)] += 1
                unigram_contexts[w].add(prev)

    total_unique_bigrams = sum(len(ctx_set) for ctx_set in unigram_contexts.values())
    continuation_prob = {
        w: len(ctx_set) / total_unique_bigrams for w, ctx_set in unigram_contexts.items()
    }

    context_totals = Counter()
    for (prev, w), count in bigrams.items():
        context_totals[prev] += count

    unique_follow = defaultdict(set)
    for (prev, w) in bigrams:
        unique_follow[prev].add(w)

    def prob(prev, w):
        count = bigrams.get((prev, w), 0)
        denom = context_totals.get(prev, 0)
        if denom == 0:
            return continuation_prob.get(w, 1e-9)
        first_term = max(count - discount, 0) / denom
        lambda_prev = discount * len(unique_follow[prev]) / denom
        return first_term + lambda_prev * continuation_prob.get(w, 1e-9)

    return prob
```

三个关键部件。`continuation_prob` 刻画"这个词出现在多少种不同的上下文中？"（这正是 Kneser-Ney 的创新点）。`lambda_prev` 是折扣腾出来的概率质量，用来给回退项加权。最终概率等于打了折扣的主项加上加权后的延续项。

### 第 4 步：用采样生成文本

```python
import random


def generate(prob_fn, vocab, prefix, max_len=30, seed=0):
    rng = random.Random(seed)
    tokens = list(prefix)
    for _ in range(max_len):
        candidates = [(w, prob_fn(tokens[-1], w)) for w in vocab]
        total = sum(p for _, p in candidates)
        r = rng.random() * total
        acc = 0.0
        for w, p in candidates:
            acc += p
            if r <= acc:
                tokens.append(w)
                break
        if tokens[-1] == "</s>":
            break
    return tokens
```

按概率比例采样。不同的随机种子会得到不同的输出。如果想要类似束搜索（beam search）的输出，可以在每一步取 argmax（贪心），再加上一个小的随机性旋钮（温度）。

### 第 5 步：困惑度

```python
import math


def perplexity(prob_fn, sentences):
    total_log_prob = 0.0
    total_tokens = 0
    for sentence in sentences:
        padded = ["<s>"] + sentence + ["</s>"]
        for i in range(1, len(padded)):
            p = prob_fn(padded[i - 1], padded[i])
            total_log_prob += math.log(max(p, 1e-12))
            total_tokens += 1
    return math.exp(-total_log_prob / total_tokens)
```

越低越好。在 Brown 语料库上，一个调优良好的 4-gram KN 模型困惑度约为 140。Transformer 语言模型在同一测试集上能达到 15-30。差距大约 10 倍。这个差距正是整个领域转向新方法的原因。

## 生产实践

- **经典 NLP 教学。** 学习平滑、最大似然估计（MLE）和困惑度的最清晰途径。
- **KenLM。** 生产级 n-gram 库。在对延迟敏感的语音和机器翻译系统中用作重打分器（rescorer）。
- **端侧自动补全。** 键盘输入法里的三元组模型。直到今天依然在用。
- **基线。** 在宣称你的神经语言模型很好之前，永远先算一个 n-gram LM 的困惑度。如果你的 Transformer 没有大幅超过 KN，那一定哪里出了问题。

## 交付产物

保存为 `outputs/prompt-lm-baseline.md`：

```markdown
---
name: lm-baseline
description: Build a reproducible n-gram language model baseline before training a neural LM.
phase: 5
lesson: 16
---

Given a corpus and target use (next-word prediction, rescoring, perplexity baseline), output:

1. N-gram order. Trigram for general English, 4-gram if corpus is large, 5-gram for speech rescoring.
2. Smoothing. Modified Kneser-Ney is the default; Laplace only for teaching.
3. Library. `kenlm` for production, `nltk.lm` for teaching, roll your own only to learn.
4. Evaluation. Held-out perplexity with consistent tokenization between train and test sets.

Refuse to report perplexity computed with different tokenization between systems being compared — perplexity numbers are comparable only under identical tokenization. Flag OOV rate in test set; KN handles OOV poorly unless you reserve a special <UNK> token during training.
```

## 练习

1. **简单。** 在一个 1,000 句的莎士比亚语料上训练三元组语言模型，生成 20 个句子。它们会局部通顺但全局不连贯。这是最经典的演示。
2. **中等。** 在莎士比亚留出集上为你的 KN 模型实现困惑度计算，并与 Laplace 平滑对比。你应该能看到 KN 把困惑度降低 30-50%。
3. **困难。** 构建一个三元组拼写纠错器：给定拼错的词及其上下文，生成候选纠正词，并按语言模型下的上下文概率排序。在 Birkbeck 拼写语料库（公开数据集）上评估。

## 关键术语

| 术语 | 人们常说的 | 实际含义 |
|------|-----------------|-----------------------|
| N-gram | 词序列 | `n` 个连续 token 组成的序列。 |
| 平滑（Smoothing） | 避免零概率 | 重新分配概率质量，让未见事件获得非零概率。 |
| 困惑度（Perplexity） | 语言模型质量指标 | 留出数据上的 `exp(-average log-prob)`。越低越好。 |
| 回退（Backoff） | 回退到更短的上下文 | 如果三元组计数为零，就用二元组。Katz backoff 将其形式化。 |
| Kneser-Ney | 最好的 n-gram 平滑方法 | 绝对折扣 + 用延续概率构建低阶模型。 |
| 延续概率（Continuation probability） | KN 专属概念 | 按 `w` 出现的上下文数量加权的 `P(w)`，而不是按原始计数。 |

## 延伸阅读

- [Jurafsky and Martin — Speech and Language Processing, Chapter 3 (2026 draft)](https://web.stanford.edu/~jurafsky/slp3/3.pdf) —— n-gram 语言模型与平滑的权威教材章节。
- [Chen and Goodman (1998). An Empirical Study of Smoothing Techniques for Language Modeling](https://dash.harvard.edu/handle/1/25104739) —— 一锤定音地确立 Kneser-Ney 为最佳 n-gram 平滑方法的论文。
- [Kneser and Ney (1995). Improved Backing-off for M-gram Language Modeling](https://ieeexplore.ieee.org/document/479394) —— KN 的原始论文。
- [KenLM](https://kheafield.com/code/kenlm/) —— 快速的生产级 n-gram 语言模型，2026 年仍在延迟敏感的场景中使用。
