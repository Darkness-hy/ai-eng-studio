# GloVe、FastText 与子词嵌入

> Word2Vec 给每个词训练一个嵌入。GloVe 直接分解共现矩阵。FastText 给词的碎片做嵌入。BPE 则架起了通往 Transformer 的桥梁。

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 5 · 03 (Word2Vec from Scratch)
**Time:** ~45 minutes

## 问题背景

Word2Vec 留下了两个悬而未决的问题。

第一，当时存在一条并行的研究路线：直接分解共现矩阵（LSA、HAL），而不是做在线的 skip-gram 更新。Word2Vec 的迭代式方法是否在本质上更优？还是说两者的差距只是计数处理方式不同造成的假象？**GloVe** 回答了这个问题：只要损失函数设计得当，矩阵分解可以追平甚至超越 Word2Vec，而且训练开销更低。

第二，这两种方法都无法处理从未见过的词。`Zoomer-approved`、`dogecoin`、上周刚造出来的任何专有名词、每个罕见词根的各种屈折变形。**FastText** 通过对字符 n-gram 做嵌入解决了这个问题：一个词等于其各部分（包括词素）之和，因此即使是词表外（out-of-vocabulary）的词也能得到一个合理的向量。

第三，Transformer 出现之后，问题再次发生了转移。词级词表的规模上限大约在一百万条；而真实语言的开放程度远超于此。**字节对编码（Byte-pair encoding，BPE）**及其变体的解法是：学习一个由高频子词单元组成的词表，覆盖一切输入。如今所有现代 LLM 的分词器都是子词分词器。

本节课依次讲解这三种方法，然后说明在什么场景下该选哪一个。

## 核心概念

**GloVe（Global Vectors）。**构建词-词共现矩阵 `X`，其中 `X[i][j]` 表示词 `j` 在词 `i` 的上下文中出现的频次。训练向量，使得 `v_i · v_j + b_i + b_j ≈ log(X[i][j])`。对损失加权，避免高频词对主导训练。就这么简单。

**FastText。**一个词等于其字符 n-gram 加上词本身的总和。`where` 变成 `<wh, whe, her, ere, re>, <where>`。词向量是这些组成部分向量之和。训练方式与 Word2Vec 相同。好处是：未见过的词（`whereupon`）可以由已知的 n-gram 组合出来。

**BPE（Byte-Pair Encoding，字节对编码）。**从由单个字节（或字符）构成的词表出发。统计语料中所有相邻的符号对。把出现频次最高的一对合并成一个新 token。重复 `k` 轮。结果是一个含 `k + 256` 个 token 的词表：高频序列（`ing`、`tion`、`the`）成为单个 token，罕见词则被拆成熟悉的片段。任何句子都能被分词成有效结果。

## 从零实现

### GloVe：分解共现矩阵

```python
import numpy as np
from collections import Counter


def build_cooccurrence(docs, window=5):
    pair_counts = Counter()
    vocab = {}
    for doc in docs:
        for token in doc:
            if token not in vocab:
                vocab[token] = len(vocab)
    for doc in docs:
        indexed = [vocab[t] for t in doc]
        for i, center in enumerate(indexed):
            for j in range(max(0, i - window), min(len(indexed), i + window + 1)):
                if i != j:
                    distance = abs(i - j)
                    pair_counts[(center, indexed[j])] += 1.0 / distance
    return vocab, pair_counts


def glove_train(vocab, pair_counts, dim=16, epochs=100, lr=0.05, x_max=100, alpha=0.75, seed=0):
    n = len(vocab)
    rng = np.random.default_rng(seed)
    W = rng.normal(0, 0.1, size=(n, dim))
    W_tilde = rng.normal(0, 0.1, size=(n, dim))
    b = np.zeros(n)
    b_tilde = np.zeros(n)

    for epoch in range(epochs):
        for (i, j), x_ij in pair_counts.items():
            weight = (x_ij / x_max) ** alpha if x_ij < x_max else 1.0
            diff = W[i] @ W_tilde[j] + b[i] + b_tilde[j] - np.log(x_ij)
            coef = weight * diff

            grad_W_i = coef * W_tilde[j]
            grad_W_tilde_j = coef * W[i]
            W[i] -= lr * grad_W_i
            W_tilde[j] -= lr * grad_W_tilde_j
            b[i] -= lr * coef
            b_tilde[j] -= lr * coef

    return W + W_tilde
```

有两个值得点名的设计。加权函数 `f(x) = (x/x_max)^alpha` 会降低超高频词对（比如 `(the, and)`）的权重，避免它们主导损失。最终的嵌入是 `W`（中心词）和 `W_tilde`（上下文词）两张表之和。把两者相加是论文中提出的技巧，效果通常优于只用其中一张表。

### FastText：感知子词的嵌入

```python
def char_ngrams(word, n_min=3, n_max=6):
    wrapped = f"<{word}>"
    grams = {wrapped}
    for n in range(n_min, n_max + 1):
        for i in range(len(wrapped) - n + 1):
            grams.add(wrapped[i:i + n])
    return grams
```

```python
>>> char_ngrams("where")
{'<where>', '<wh', 'whe', 'her', 'ere', 're>', '<whe', 'wher', 'here', 'ere>', '<wher', 'where', 'here>'}
```

每个词由其 n-gram 集合表示（通常取 3 到 6 个字符）。词嵌入是其各 n-gram 嵌入之和。做 skip-gram 训练时，把它替换到 Word2Vec 原本使用单一向量的位置即可。

```python
def fasttext_vector(word, ngram_table):
    grams = char_ngrams(word)
    vecs = [ngram_table[g] for g in grams if g in ngram_table]
    if not vecs:
        return None
    return np.sum(vecs, axis=0)
```

对于未见过的词，只要它的部分 n-gram 是已知的，你仍然能得到一个向量。`whereupon` 与 `where` 共享 `<wh`、`her`、`ere` 和 `<where`，所以这两个词的向量会落在彼此附近。

### BPE：学习得到的子词词表

```python
def learn_bpe(corpus, k_merges):
    vocab = Counter()
    for word, freq in corpus.items():
        tokens = tuple(word) + ("</w>",)
        vocab[tokens] = freq

    merges = []
    for _ in range(k_merges):
        pair_freq = Counter()
        for tokens, freq in vocab.items():
            for a, b in zip(tokens, tokens[1:]):
                pair_freq[(a, b)] += freq
        if not pair_freq:
            break
        best = pair_freq.most_common(1)[0][0]
        merges.append(best)

        new_vocab = Counter()
        for tokens, freq in vocab.items():
            new_tokens = []
            i = 0
            while i < len(tokens):
                if i + 1 < len(tokens) and (tokens[i], tokens[i + 1]) == best:
                    new_tokens.append(tokens[i] + tokens[i + 1])
                    i += 2
                else:
                    new_tokens.append(tokens[i])
                    i += 1
            new_vocab[tuple(new_tokens)] = freq
        vocab = new_vocab
    return merges


def apply_bpe(word, merges):
    tokens = list(word) + ["</w>"]
    for a, b in merges:
        new_tokens = []
        i = 0
        while i < len(tokens):
            if i + 1 < len(tokens) and tokens[i] == a and tokens[i + 1] == b:
                new_tokens.append(a + b)
                i += 2
            else:
                new_tokens.append(tokens[i])
                i += 1
        tokens = new_tokens
    return tokens
```

```python
>>> corpus = Counter({"low": 5, "lower": 2, "newest": 6, "widest": 3})
>>> merges = learn_bpe(corpus, k_merges=10)
>>> apply_bpe("lowest", merges)
['low', 'est</w>']
```

第一轮迭代合并最常见的相邻符号对。迭代足够多轮之后，高频子串（`low`、`est`、`tion`）成为单个 token，罕见词则被干净利落地拆开。

真实的 GPT / BERT / T5 分词器学习 3 万到 10 万次合并。结果是：任何文本都能分词成长度有界、ID 全部已知的序列，永远不会出现 OOV。

## 生产实践

实践中你几乎不会自己训练这些模型，而是直接加载预训练的检查点。

```python
import fasttext.util
fasttext.util.download_model("en", if_exists="ignore")
ft = fasttext.load_model("cc.en.300.bin")
print(ft.get_word_vector("whereupon").shape)
print(ft.get_word_vector("zoomerapproved").shape)
```

在 Transformer 时代使用 BPE 风格的子词分词：

```python
from transformers import AutoTokenizer

tok = AutoTokenizer.from_pretrained("gpt2")
print(tok.tokenize("unbelievably tokenized"))
```

```
['un', 'bel', 'iev', 'ably', 'Ġtoken', 'ized']
```

前缀 `Ġ` 标记词的边界（GPT-2 的约定）。所有现代分词器都是 BPE 的某种变体：WordPiece（BERT）或 SentencePiece（T5、LLaMA）。

### 什么时候选哪个

| 场景 | 选择 |
|-----------|------|
| 需要预训练的通用词向量，且无需容忍 OOV | GloVe 300d |
| 需要预训练的通用词向量，且必须处理拼写错误 / 新造词 / 形态丰富的语言 | FastText |
| 任何要送入 Transformer 的内容（无论训练还是推理） | 模型自带的分词器。绝对不要换。 |
| 从零训练自己的语言模型 | 先在你的语料上训练一个 BPE 或 SentencePiece 分词器 |
| 用线性模型做生产环境的文本分类 | 还是 TF-IDF。见第 02 课。 |

## 交付产物

保存为 `outputs/skill-embeddings-picker.md`：

```markdown
---
name: tokenizer-picker
description: Pick a tokenization approach for a new language model or text pipeline.
version: 1.0.0
phase: 5
lesson: 04
tags: [nlp, tokenization, embeddings]
---

Given a task and dataset description, you output:

1. Tokenization strategy (word-level, BPE, WordPiece, SentencePiece, byte-level). One-sentence reason.
2. Vocabulary size target (e.g., 32k for an English-only LM, 64k-100k for multilingual).
3. Library call with the exact training command. Name the library. Quote the arguments.
4. One reproducibility pitfall. Tokenizer-model mismatch is the single most common silent production bug; call out which pair must be used together.

Refuse to recommend training a custom tokenizer when the user is fine-tuning a pretrained LLM. Refuse to recommend word-level tokenization for any model targeting production inference. Flag non-English / multi-script corpora as needing SentencePiece with byte fallback.
```

## 练习

1. **简单。**运行 `char_ngrams("playing")` 和 `char_ngrams("played")`。计算两个 n-gram 集合的 Jaccard 重叠度。你会看到大量共享片段（`pla`、`lay`、`play`），这正是 FastText 能在形态变体之间良好迁移的原因。
2. **中等。**扩展 `learn_bpe`，跟踪词表的增长过程。绘制「每个语料字符对应的 token 数」随合并次数变化的曲线。你会看到初期压缩很快，随后渐近收敛到每个 token 约 2-3 个字符。
3. **困难。**在莎士比亚全集上训练一个 1k 次合并的 BPE。比较常见词与罕见专有名词的分词结果。测量训练前后的平均每词 token 数。写下让你意外的发现。

## 关键术语

| 术语 | 大家口中的说法 | 实际含义 |
|------|-----------------|-----------------------|
| 共现矩阵 | 词-词频次表 | `X[i][j]` = 词 `j` 在词 `i` 的窗口范围内出现的频次。 |
| 子词 | 词的片段 | 一个字符 n-gram（FastText）或学习得到的 token（BPE/WordPiece/SentencePiece）。 |
| BPE | 字节对编码 | 迭代合并最高频的相邻符号对，直到词表达到目标规模。 |
| OOV | 词表外（Out of vocabulary） | 模型从未见过的词。Word2Vec/GloVe 会失效，FastText 和 BPE 能处理。 |
| 字节级 BPE | 对原始字节做 BPE | GPT-2 的方案。词表从 256 个字节起步，因此永远不会出现 OOV。 |

## 延伸阅读

- [Pennington, Socher, Manning (2014). GloVe: Global Vectors for Word Representation](https://nlp.stanford.edu/pubs/glove.pdf) —— GloVe 原始论文，仅七页，至今仍是该损失函数最好的推导。
- [Bojanowski et al. (2017). Enriching Word Vectors with Subword Information](https://arxiv.org/abs/1607.04606) —— FastText。
- [Sennrich, Haddow, Birch (2016). Neural Machine Translation of Rare Words with Subword Units](https://arxiv.org/abs/1508.07909) —— 把 BPE 引入现代 NLP 的那篇论文。
- [Hugging Face tokenizer summary](https://huggingface.co/docs/transformers/tokenizer_summary) —— BPE、WordPiece 和 SentencePiece 在实践中的真实差异。
