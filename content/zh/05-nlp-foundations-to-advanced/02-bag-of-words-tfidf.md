# 词袋模型、TF-IDF 与文本表示

> 先数数，再思考。到了 2026 年，在定义清晰的任务上 TF-IDF 依然能打败嵌入模型。

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 5 · 01 (Text Processing), Phase 2 · 02 (Linear Regression from Scratch)
**Time:** ~75 minutes

## 问题背景

模型需要数字，而你手里只有字符串。

每一条 NLP 流水线都必须回答同一个问题：如何把一段变长的 token 流变成分类器能消费的定长向量？这个领域最早落地的答案，是最笨却管用的那个——数词频，拼成向量。

这种向量支撑过的生产级 NLP 系统比任何嵌入（embedding）模型都多：垃圾邮件过滤、主题分类、日志异常检测、搜索排序（在 BM25 出现之前）、第一波情感分析，以及学术界 NLP 基准测试的头十年。2026 年的从业者在面对窄域分类任务时，仍然会先伸手拿它。它快、可解释，而且在「词是否出现」就是关键信号的任务上，效果常常与一个 4 亿参数的嵌入模型难分高下。

本课先从零实现词袋模型，再实现 TF-IDF；然后展示 scikit-learn 如何用三行代码完成同样的事；最后点明那个会迫使你转向嵌入模型的失效模式。

## 核心概念

**词袋模型（Bag of Words, BoW）** 抛弃了词序。对每篇文档，统计词表中每个词出现的次数。向量长度等于词表大小，第 `i` 个位置就是第 `i` 个词的计数。

**TF-IDF** 对 BoW 重新加权。一个在每篇文档里都出现的词没有信息量，所以降低它的权重；一个在整个语料中罕见、却在某篇文档里频繁出现的词是信号，所以提升它的权重。

```
TF-IDF(w, d) = TF(w, d) * IDF(w)
             = count(w in d) / |d| * log(N / df(w))
```

其中 `TF` 是词在文档中的词频，`df` 是文档频率（包含该词的文档数），`N` 是文档总数。`log` 让无处不在的词的权重保持有界。

关键性质：两者产出的都是坐标轴可解释的稀疏向量。你可以查看训练好的分类器权重，直接读出哪些词把文档推向哪个类别。而面对一个 768 维的 BERT 嵌入，你做不到这一点。

```figure
bow-tfidf
```

## 从零实现

### 第 1 步：构建词表

```python
def build_vocab(docs):
    vocab = {}
    for doc in docs:
        for token in doc:
            if token not in vocab:
                vocab[token] = len(vocab)
    return vocab
```

输入：分词后的文档列表（任何词级分词器都行；本课的 `code/main.py` 使用一个简化的小写化版本）。输出：`{word: index}` 字典。由于字典保持插入顺序，索引 0 对应第一篇文档中第一个出现的词。各实现的约定不同；scikit-learn 按字母序排序。

### 第 2 步：词袋

```python
def bag_of_words(docs, vocab):
    matrix = [[0] * len(vocab) for _ in docs]
    for i, doc in enumerate(docs):
        for token in doc:
            if token in vocab:
                matrix[i][vocab[token]] += 1
    return matrix
```

```python
>>> docs = [["cat", "sat", "on", "mat"], ["cat", "cat", "ran"]]
>>> vocab = build_vocab(docs)
>>> bag_of_words(docs, vocab)
[[1, 1, 1, 1, 0], [2, 0, 0, 0, 1]]
```

行是文档，列是词表索引。第 `[i][j]` 项表示「词 `j` 在文档 `i` 中出现了多少次」。文档 1 中 `cat` 的计数是 2，因为它确实出现了两次；文档 0 中 `ran` 的计数是 0，因为它没出现。

### 第 3 步：词频与文档频率

```python
import math


def term_frequency(doc_bow, doc_length):
    return [c / doc_length if doc_length else 0 for c in doc_bow]


def document_frequency(bow_matrix):
    df = [0] * len(bow_matrix[0])
    for row in bow_matrix:
        for j, count in enumerate(row):
            if count > 0:
                df[j] += 1
    return df


def inverse_document_frequency(df, n_docs):
    return [math.log((n_docs + 1) / (d + 1)) + 1 for d in df]
```

有两个值得点名的平滑技巧。`(n+1)/(d+1)` 避免了 `log(x/0)`；末尾的 `+1` 保证出现在所有文档中的词 IDF 仍为 1（而不是 0），与 scikit-learn 的默认行为一致。也有实现直接用原始的 `log(N/df)`。两种都能用，平滑版本更友好。

### 第 4 步：TF-IDF

```python
def tfidf(bow_matrix):
    n_docs = len(bow_matrix)
    df = document_frequency(bow_matrix)
    idf = inverse_document_frequency(df, n_docs)
    out = []
    for row in bow_matrix:
        length = sum(row)
        tf = term_frequency(row, length)
        out.append([tf_j * idf_j for tf_j, idf_j in zip(tf, idf)])
    return out
```

```python
>>> docs = [
...     ["the", "cat", "sat"],
...     ["the", "dog", "sat"],
...     ["the", "cat", "ran"],
... ]
>>> vocab = build_vocab(docs)
>>> bow = bag_of_words(docs, vocab)
>>> tfidf(bow)
```

三篇文档，五个词表词（`the`、`cat`、`sat`、`dog`、`ran`）。`the` 在三篇里都出现，所以 IDF 低；`dog` 只出现在一篇里，所以 IDF 高。得到的向量是稀疏的（大多数项很小），而有区分度的词会凸显出来。

### 第 5 步：按行做 L2 归一化

```python
def l2_normalize(matrix):
    out = []
    for row in matrix:
        norm = math.sqrt(sum(x * x for x in row))
        out.append([x / norm if norm else 0 for x in row])
    return out
```

如果不做归一化，较长的文档会得到更大的向量，从而主导相似度得分。L2 归一化把每篇文档都放到单位超球面上。此后行与行之间的余弦相似度就只是一个点积。

## 生产实践

scikit-learn 自带生产级实现。

```python
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

docs = ["the cat sat on the mat", "the dog sat on the mat", "the cat ran"]

bow_vectorizer = CountVectorizer()
bow = bow_vectorizer.fit_transform(docs)
print(bow_vectorizer.get_feature_names_out())
print(bow.toarray())

tfidf_vectorizer = TfidfVectorizer()
tfidf = tfidf_vectorizer.fit_transform(docs)
print(tfidf.toarray().round(3))
```

`CountVectorizer` 一次调用就完成分词、建词表和 BoW。`TfidfVectorizer` 在此之上加了 IDF 加权和 L2 归一化。两者返回的都是稀疏矩阵。对于 10 万篇文档，稠密版本根本放不进内存；在分类器要求稠密输入之前，请一直保持稀疏。

几个能彻底改变结果的参数：

| 参数 | 效果 |
|-----|--------|
| `ngram_range=(1, 2)` | 加入二元组（bigram）。通常能提升分类效果。 |
| `min_df=2` | 丢弃出现在少于 2 篇文档中的词。在嘈杂数据上能精简词表。 |
| `max_df=0.95` | 丢弃出现在超过 95% 文档中的词。近似于停用词去除，又不依赖硬编码列表。 |
| `stop_words="english"` | scikit-learn 内置的停用词表。是否使用取决于任务——情感分析*不*应该丢掉否定词。 |
| `sublinear_tf=True` | 用 `1 + log(tf)` 代替原始 `tf`。当某个词在一篇文档中重复多次时有帮助。 |

### TF-IDF 仍然占优的场景（截至 2026 年）

- 垃圾邮件检测、主题标注、日志异常标记。这里关键信号是词是否出现，语义细微差别无关紧要。
- 低数据量场景（只有几百条标注样本）。TF-IDF 加逻辑回归没有任何预训练成本。
- 任何对延迟敏感的地方。TF-IDF 加线性模型在微秒级给出答案；让一篇文档过一遍 Transformer 做嵌入需要 10-100 毫秒。
- 必须解释自身预测的系统。检查分类器的系数，权重最高的正向词就是理由。

### TF-IDF 失效的场景

语义盲区式失效。看这两篇文档：

- "The movie was not good at all."
- "The movie was excellent."

一篇是负面影评，一篇是正面影评。它们的 TF-IDF 重叠恰好是 `{the, movie, was}`。词袋分类器只能死记硬背「`not` 出现在 `good` 附近会翻转标签」。数据够多时它能学会，但永远不如一个理解句法的模型来得优雅。

另一种失效：推理时遇到词表外（out-of-vocabulary）的词。一个在 IMDb 影评上训练的 BoW 模型，面对训练中从未出现过的 `Zoomer-approved` 完全无能为力。子词嵌入（第 04 课）能处理这种情况，TF-IDF 不能。

### 混合方案：TF-IDF 加权嵌入

2026 年中等数据量分类任务的务实默认做法：把 TF-IDF 权重当作施加在词嵌入上的注意力。

```python
def tfidf_weighted_embedding(doc, tfidf_scores, embedding_table, dim):
    vec = [0.0] * dim
    total_weight = 0.0
    for token in doc:
        if token not in embedding_table or token not in tfidf_scores:
            continue
        weight = tfidf_scores[token]
        emb = embedding_table[token]
        for i in range(dim):
            vec[i] += weight * emb[i]
        total_weight += weight
    if total_weight == 0:
        return vec
    return [v / total_weight for v in vec]
```

你从嵌入中获得语义容量，从 TF-IDF 中获得对罕见词的强调。分类器在池化后的向量上训练。在标注样本不超过约 5 万条的情感、主题和意图分类任务上，这种混合方案胜过两者各自单独使用。

## 交付产物

保存为 `outputs/prompt-vectorization-picker.md`：

```markdown
---
name: vectorization-picker
description: Given a text-classification task, recommend BoW, TF-IDF, embeddings, or a hybrid.
phase: 5
lesson: 02
---

You recommend a text-vectorization strategy. Given a task description, output:

1. Representation (BoW, TF-IDF, transformer embeddings, or a hybrid). Explain why in one sentence.
2. Specific vectorizer configuration. Name the library. Quote the arguments (`ngram_range`, `min_df`, `max_df`, `sublinear_tf`, `stop_words`).
3. One failure mode to test before shipping.

Refuse to recommend embeddings when the user has under 500 labeled examples unless they show evidence of semantic failure in a TF-IDF baseline. Refuse to remove stopwords for sentiment analysis (negations carry signal). Flag class imbalance as needing more than a vectorizer change.

Example input: "Classifying 30k customer support tickets into 12 categories. Most tickets are 2-3 sentences. English only. Need explainability for audit logs."

Example output:

- Representation: TF-IDF. 30k examples is not small; explainability requirement rules out dense embeddings.
- Config: `TfidfVectorizer(ngram_range=(1, 2), min_df=3, max_df=0.95, sublinear_tf=True, stop_words=None)`. Keep stopwords because category keywords sometimes are stopwords ("not working" vs "working").
- Failure to test: verify `min_df=3` does not drop rare category keywords. Run `get_feature_names_out` filtered by class and eyeball.
```

## 练习

1. **简单。** 在 L2 归一化后的 TF-IDF 输出上实现 `cosine_similarity(doc_vec_a, doc_vec_b)`。验证完全相同的文档得分为 1.0，词表完全不相交的文档得分为 0.0。
2. **中等。** 给 `bag_of_words` 加上 `n-gram` 支持。参数 `n` 控制对 `n` 元组计数。测试 `n=2` 时，`["the", "cat", "sat"]` 应产生 `["the cat", "cat sat"]` 的二元组计数。
3. **困难。** 用 GloVe 100 维向量（下载一次后缓存）实现上文的 TF-IDF 加权嵌入混合方案。在 20 Newsgroups 数据集上对比它与纯 TF-IDF、纯均值池化嵌入的分类准确率。报告各自在哪些情况下胜出。

## 关键术语

| 术语 | 人们常说的 | 实际含义 |
|------|-----------------|-----------------------|
| BoW | 词频向量 | 词表中各词在一篇文档中的计数。抛弃词序。 |
| TF | 词频 | 词在文档中的出现次数，可选地按文档长度归一化。 |
| DF | 文档频率 | 至少包含该词一次的文档数。 |
| IDF | 逆文档频率 | 平滑后的 `log(N / df)`。降低无处不在的词的权重。 |
| 稀疏向量 | 大部分是零 | 词表通常有 1 万到 10 万个词；对任何一篇文档来说，绝大多数词都不出现。 |
| 余弦相似度 | 向量夹角 | L2 归一化向量的点积。1 表示完全相同，0 表示正交。 |

## 延伸阅读

- [scikit-learn — feature extraction from text](https://scikit-learn.org/stable/modules/feature_extraction.html#text-feature-extraction) —— 权威 API 参考，附带每个参数的说明。
- [Salton, G., & Buckley, C. (1988). Term-weighting approaches in automatic text retrieval](https://www.sciencedirect.com/science/article/pii/0306457388900210) —— 让 TF-IDF 成为此后十年默认方案的论文。
- ["Why TF-IDF Still Beats Embeddings" — Ashfaque Thonikkadavan (Medium)](https://medium.com/@cmtwskb/why-tf-idf-still-beats-embeddings-ad85c123e1b2) —— 2026 年视角：老方法何时取胜、为何取胜。
