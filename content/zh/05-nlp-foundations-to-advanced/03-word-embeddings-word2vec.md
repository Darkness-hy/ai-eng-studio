# 词嵌入 —— 从零实现 Word2Vec

> 观其伴侣，知其词义。用一个浅层网络去训练这个想法，几何结构便自然浮现。

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 5 · 02 (BoW + TF-IDF), Phase 3 · 03 (Backpropagation from Scratch)
**Time:** ~75 minutes

## 问题背景

TF-IDF 知道 `dog` 和 `puppy` 是两个不同的词，却不知道它们的意思几乎一样。在 `dog` 上训练出来的分类器，没法泛化到一条关于 `puppy` 的评论。你可以靠罗列同义词来勉强应付，但一遇到罕见词、领域行话，以及任何你没预料到的语言，这条路就走不通了。

你想要的是这样一种表示：`dog` 和 `puppy` 在空间中彼此靠近；`king - man + woman` 落在 `queen` 附近；在 `dog` 上训练的模型能免费把一部分信号迁移给 `puppy`。

Word2Vec 给了我们这样的空间。一个两层神经网络，在万亿级 token 上训练，2013 年发表。架构简单得近乎令人难为情，结果却重塑了之后十年的 NLP。

## 核心概念

**分布假设（distributional hypothesis）**（Firth, 1957）："You shall know a word by the company it keeps."（观其伴侣，知其词义。）如果两个词出现在相似的上下文中，它们的含义很可能也相似。

Word2Vec 有两种形式，都建立在这个想法之上。

- **Skip-gram。** 给定中心词，预测周围的词。窗口大小为 2 时，`cat -> (the, sat, on)`。
- **CBOW（连续词袋，continuous bag of words）。** 给定周围的词，预测中心词。`(the, sat, on) -> cat`。

Skip-gram 训练更慢，但对罕见词的效果更好，因此成了默认选择。

这个网络只有一个隐藏层，且没有非线性激活。输入是词表上的 one-hot 向量，输出是词表上的 softmax。训练完成后，输出层直接丢掉，隐藏层的权重就是嵌入。

```
one-hot(center) ── W ──▶ hidden (d-dim) ── W' ──▶ softmax(vocab)
                          ^
                          this is the embedding
```

关键技巧在于：对 10 万个词做 softmax 的开销大到无法承受。Word2Vec 用**负采样（negative sampling）**把它变成一个二分类任务：预测「这个上下文词是否出现在这个中心词附近，是还是否」。每个训练对只采样少量负例（不共现的词），而不是对整个词表计算 softmax。

```figure
word-vector-arithmetic
```

## 从零实现

### 第 1 步：从语料生成训练对

```python
def skipgram_pairs(docs, window=2):
    pairs = []
    for doc in docs:
        for i, center in enumerate(doc):
            for j in range(max(0, i - window), min(len(doc), i + window + 1)):
                if i == j:
                    continue
                pairs.append((center, doc[j]))
    return pairs
```

```python
>>> skipgram_pairs([["the", "cat", "sat", "on", "mat"]], window=2)
[('the', 'cat'), ('the', 'sat'),
 ('cat', 'the'), ('cat', 'sat'), ('cat', 'on'),
 ('sat', 'the'), ('sat', 'cat'), ('sat', 'on'), ('sat', 'mat'),
 ...]
```

窗口内的每个（中心词, 上下文词）对都是一个正训练样本。

### 第 2 步：嵌入表

两个矩阵。`W` 是中心词嵌入表（最终保留的那个）。`W'` 是上下文词表（通常丢弃，有时与 `W` 取平均）。

```python
import numpy as np


def init_embeddings(vocab_size, dim, seed=0):
    rng = np.random.default_rng(seed)
    W = rng.normal(0, 0.1, size=(vocab_size, dim))
    W_prime = rng.normal(0, 0.1, size=(vocab_size, dim))
    return W, W_prime
```

用小的随机值初始化。词表 1 万、维度 100 是比较现实的设置；教学场景下，50 个词、16 维就足以看出几何结构。

### 第 3 步：负采样目标

对每个正样本对 `(center, context)`，从词表中随机采样 `k` 个词作为负例。训练目标是让点积 `W[center] · W'[context]` 对正例尽量大、对负例尽量小。

```python
def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -20, 20)))


def train_pair(W, W_prime, center_idx, context_idx, negative_indices, lr):
    v_c = W[center_idx]
    u_pos = W_prime[context_idx]
    u_negs = W_prime[negative_indices]

    pos_score = sigmoid(v_c @ u_pos)
    neg_scores = sigmoid(u_negs @ v_c)

    grad_center = (pos_score - 1) * u_pos
    for i, u in enumerate(u_negs):
        grad_center += neg_scores[i] * u

    W[context_idx] = W[context_idx]
    W_prime[context_idx] -= lr * (pos_score - 1) * v_c
    for i, neg_idx in enumerate(negative_indices):
        W_prime[neg_idx] -= lr * neg_scores[i] * v_c
    W[center_idx] -= lr * grad_center
```

核心公式：正例对上的 logistic 损失（希望 sigmoid 接近 1），加上负例对上的 logistic 损失（希望 sigmoid 接近 0）。梯度同时流向两张表。完整推导见原论文；想真正记住的话，找时间用纸笔走一遍。

### 第 4 步：在玩具语料上训练

```python
def train(docs, dim=16, window=2, k_neg=5, epochs=100, lr=0.05, seed=0):
    vocab = build_vocab(docs)
    vocab_size = len(vocab)
    rng = np.random.default_rng(seed)
    W, W_prime = init_embeddings(vocab_size, dim, seed=seed)
    pairs = skipgram_pairs(docs, window=window)

    for epoch in range(epochs):
        rng.shuffle(pairs)
        for center, context in pairs:
            c_idx = vocab[center]
            ctx_idx = vocab[context]
            negs = rng.integers(0, vocab_size, size=k_neg)
            negs = [n for n in negs if n != ctx_idx and n != c_idx]
            train_pair(W, W_prime, c_idx, ctx_idx, negs, lr)
    return vocab, W
```

在大语料上训练足够多的轮数后，共享上下文的词会得到相似的中心词嵌入。在玩具语料上，这个效应只能隐约看到；在数十亿 token 上，效果非常显著。

### 第 5 步：类比技巧

```python
def nearest(vocab, W, target_vec, topk=5, exclude=None):
    exclude = exclude or set()
    inv_vocab = {i: w for w, i in vocab.items()}
    norms = np.linalg.norm(W, axis=1, keepdims=True) + 1e-9
    W_norm = W / norms
    target = target_vec / (np.linalg.norm(target_vec) + 1e-9)
    sims = W_norm @ target
    order = np.argsort(-sims)
    out = []
    for i in order:
        if i in exclude:
            continue
        out.append((inv_vocab[i], float(sims[i])))
        if len(out) == topk:
            break
    return out


def analogy(vocab, W, a, b, c, topk=5):
    v = W[vocab[b]] - W[vocab[a]] + W[vocab[c]]
    return nearest(vocab, W, v, topk=topk, exclude={vocab[a], vocab[b], vocab[c]})
```

在预训练的 300 维 Google News 向量上：

```python
>>> analogy(vocab, W, "man", "king", "woman")
[('queen', 0.71), ('monarch', 0.62), ('princess', 0.59), ...]
```

`king - man + woman = queen`。这不是因为模型懂得什么是王室，而是因为向量 `(king - man)` 捕捉到了类似「royal（王室）」的方向，把它加到 `woman` 上，正好落在「王室女性」那片区域附近。

## 生产实践

从零手写 Word2Vec 是为了教学。生产环境的 NLP 用 `gensim`。

```python
from gensim.models import Word2Vec

sentences = [
    ["the", "cat", "sat", "on", "the", "mat"],
    ["the", "dog", "ran", "across", "the", "room"],
]

model = Word2Vec(
    sentences,
    vector_size=100,
    window=5,
    min_count=1,
    sg=1,
    negative=5,
    workers=4,
    epochs=30,
)

print(model.wv["cat"])
print(model.wv.most_similar("cat", topn=3))
```

真正干活时，你几乎从不自己训练 Word2Vec，而是直接下载预训练向量。

- **GloVe** —— Stanford 基于共现矩阵分解的方法。提供 50、100、200、300 维的 checkpoint。通用覆盖面好。第 04 课会专门讲 GloVe。
- **fastText** —— Facebook 对 Word2Vec 的扩展，为字符 n-gram 学习嵌入，通过组合子词来处理词表外的词。见第 04 课。
- **基于 Google News 预训练的 Word2Vec** —— 300 维，词表 300 万词，2013 年发布，至今每天都有人下载。

### 2026 年 Word2Vec 仍然占优的场景

- 轻量级的领域专用检索。在笔记本电脑上花一小时在医学摘要上训练，就能得到通用模型捕捉不到的专业向量。
- 类比式特征工程。`gender_vector = mean(man - woman pairs)`，把它从其他词向量中减去，就得到一条性别中立的轴。公平性研究中仍在使用。
- 可解释性。100 维小到可以用 PCA 或 t-SNE 画出来，真的能看到聚类成形。
- 任何必须在无 GPU 的设备端做推理的场合。Word2Vec 的查询就是取一行向量。

### Word2Vec 失效的地方

一词多义（polysemy）这堵墙。`bank` 只有一个向量，`river bank`（河岸）和 `financial bank`（银行）共用它；`table`（电子表格 vs. 家具）也共用一个向量。下游的分类器无法从这个向量中区分词义。

上下文相关嵌入（ELMo、BERT，以及之后的所有 Transformer）解决了这个问题：根据周围上下文，为词的每次出现生成不同的向量。这就是从 Word2Vec 到 BERT 的跨越：从静态到上下文相关。Phase 7 会讲 Transformer 那一半。

另一个失效点是词表外（OOV，out-of-vocabulary）问题。如果训练数据里没有 `Zoomer-approved`，Word2Vec 就从未见过它，也没有任何兜底手段。fastText 用子词组合修复了这一点（第 04 课）。

## 交付产物

保存为 `outputs/skill-embedding-probe.md`：

```markdown
---
name: embedding-probe
description: Inspect a word2vec model. Run analogies, find neighbors, diagnose quality.
version: 1.0.0
phase: 5
lesson: 03
tags: [nlp, embeddings, debugging]
---

You probe trained word embeddings to verify they are working. Given a `gensim.models.KeyedVectors` object and a vocabulary, you run:

1. Three canonical analogy tests. `king : man :: queen : woman`. `paris : france :: tokyo : japan`. `walking : walked :: swimming : ?`. Report the top-1 result and its cosine.
2. Five nearest-neighbor tests on domain-specific words the user supplies. Print top-5 neighbors with cosines.
3. One symmetry check. `similarity(a, b) == similarity(b, a)` to within float precision.
4. One degenerate check. If any embedding has a norm below 0.01 or above 100, the model has a training bug. Flag it.

Refuse to declare a model good on analogy accuracy alone. Analogy benchmarks are gameable and do not transfer to downstream tasks. Recommend intrinsic + downstream evaluation together.
```

## 练习

1. **简单。** 在一个很小的语料（20 句关于猫和狗的句子）上运行训练循环。训练 200 轮后，验证 `nearest(vocab, W, W[vocab["cat"]])` 的前 3 名里有 `dog`。如果没有，就增加训练轮数或扩大词表。
2. **中等。** 加入高频词的下采样（subsampling）：频率高于 `10^-5` 的词，以与其频率成正比的概率从训练对中剔除。测量这对罕见词相似度的影响。
3. **困难。** 在 20 Newsgroups 语料上训练一个模型。计算两条偏见轴：`he - she` 和 `doctor - nurse`。把职业词投影到这两条轴上，报告哪些职业的偏见差距最大。这正是公平性研究者使用的那类探针。

## 关键术语

| 术语 | 大家怎么说 | 实际含义 |
|------|-----------------|-----------------------|
| 词嵌入（word embedding） | 词的向量表示 | 从上下文中学到的稠密低维表示（通常 100-300 维）。 |
| Skip-gram | Word2Vec 的招数 | 用中心词预测上下文词。比 CBOW 慢，但对罕见词效果更好。 |
| 负采样（negative sampling） | 训练捷径 | 用针对 `k` 个随机词的二分类，替代整个词表上的 softmax。 |
| 静态嵌入（static embedding） | 一个词一个向量 | 不管上下文如何，向量都相同。在一词多义上失效。 |
| 上下文相关嵌入（contextual embedding） | 随上下文变化的向量 | 根据周围词，为每次出现生成不同的向量。Transformer 产出的就是这种。 |
| OOV | 词表外（out of vocabulary） | 训练时没见过的词。Word2Vec 无法为它们生成向量。 |

## 延伸阅读

- [Mikolov et al. (2013). Distributed Representations of Words and Phrases and their Compositionality](https://arxiv.org/abs/1310.4546) —— 负采样那篇论文。篇幅短，易读。
- [Rong, X. (2014). word2vec Parameter Learning Explained](https://arxiv.org/abs/1411.2738) —— 最清晰的梯度推导。如果原论文的数学读起来费劲，看这篇。
- [gensim Word2Vec tutorial](https://radimrehurek.com/gensim/models/word2vec.html) —— 真正可用的生产级训练配置。
