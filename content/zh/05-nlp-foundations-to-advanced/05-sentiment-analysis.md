# 情感分析

> NLP 的经典任务。经典文本分类需要掌握的知识，大部分都会在这里出现。

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 5 · 02 (BoW + TF-IDF), Phase 2 · 14 (Naive Bayes)
**Time:** ~75 minutes

## 问题背景

"The food was not great."（这家的菜不怎么样。）是正面还是负面？

情感分析听起来很简单。评论者表达了喜欢或不喜欢，给句子打个标签就行。它之所以成为 NLP 的经典任务，是因为每一个看似简单的案例背后都藏着一个棘手的案例。否定会翻转语义，讽刺会反转语义。"Not bad at all" 包含两个带负面色彩的词，但整体却是正面的。表情符号携带的信号往往比周围的文字更强。领域词汇也很重要（音乐评论里的 `tight` 和时尚评论里的 `tight` 含义不同）。

情感分析是经典 NLP 的实战练兵场。如果你理解了每个朴素基线为什么会有特定的失效模式，你就理解了每个更复杂的模型为什么会被发明出来。本课从零实现一个 Naive Bayes 基线，再加上逻辑回归，并指出那些让生产环境的情感分析变成合规级难题的陷阱。

## 核心概念

经典情感分析是一个两步配方。

1. **表示（Represent）。** 把文本转成特征向量。BoW、TF-IDF 或 n-gram。
2. **分类（Classify）。** 在标注样本上拟合一个线性模型（Naive Bayes、逻辑回归、SVM）。

Naive Bayes 是"最笨却有效"的模型。它假设在给定标签的条件下所有特征相互独立，通过计数估计 `P(word | positive)` 和 `P(word | negative)`，推理时把这些概率连乘起来。这个"朴素"的独立性假设错得离谱，结果却出奇地好。原因是：在文本特征稀疏、数据量中等的情况下，分类器更关心每个词倾向于哪一边，而不是倾向的程度有多大。

逻辑回归修正了独立性假设。它为每个特征学习一个权重，包括负权重。把 `not good` 作为 bigram 特征，它会得到一个负权重。而对于从未见过标注的 bigram，Naive Bayes 做不到这一点。

```figure
sentiment-logits
```

## 从零实现

### 第 1 步：一个真实的迷你数据集

```python
POSITIVE = [
    "absolutely loved this movie",
    "beautiful cinematography and a great story",
    "one of the best films of the year",
    "brilliant acting from the lead",
    "heartwarming and funny",
]

NEGATIVE = [
    "boring and far too long",
    "not worth your time",
    "the plot made no sense",
    "terrible acting, awful script",
    "i want my two hours back",
]
```

故意做得很小。实际工作中会用到数万条样本（IMDb、SST-2、Yelp polarity）。数学原理完全相同。

### 第 2 步：从零实现多项式 Naive Bayes

```python
import math
from collections import Counter


def train_nb(docs_by_class, vocab, alpha=1.0):
    class_priors = {}
    class_word_probs = {}
    total_docs = sum(len(d) for d in docs_by_class.values())

    for cls, docs in docs_by_class.items():
        class_priors[cls] = len(docs) / total_docs
        counts = Counter()
        for doc in docs:
            for token in doc:
                counts[token] += 1
        total = sum(counts.values()) + alpha * len(vocab)
        class_word_probs[cls] = {
            w: (counts[w] + alpha) / total for w in vocab
        }
    return class_priors, class_word_probs


def predict_nb(doc, class_priors, class_word_probs):
    scores = {}
    for cls in class_priors:
        s = math.log(class_priors[cls])
        for token in doc:
            if token in class_word_probs[cls]:
                s += math.log(class_word_probs[cls][token])
        scores[cls] = s
    return max(scores, key=scores.get)
```

加性平滑（alpha=1.0）就是拉普拉斯平滑（Laplace smoothing）。没有它，某个类别中未出现过的词概率为零，取对数时就会爆掉。实践中常用 `alpha=0.01`，教学中默认用 `alpha=1.0`。

### 第 3 步：从零实现逻辑回归

```python
import numpy as np


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -20, 20)))


def train_lr(X, y, epochs=500, lr=0.05, l2=0.01):
    n_features = X.shape[1]
    w = np.zeros(n_features)
    b = 0.0
    for _ in range(epochs):
        logits = X @ w + b
        preds = sigmoid(logits)
        err = preds - y
        grad_w = X.T @ err / len(y) + l2 * w
        grad_b = err.mean()
        w -= lr * grad_w
        b -= lr * grad_b
    return w, b


def predict_lr(X, w, b):
    return (sigmoid(X @ w + b) >= 0.5).astype(int)
```

L2 正则化在这里很关键。文本特征是稀疏的，没有 L2 正则，模型会把训练样本背下来。从 `0.01` 开始，再做调优。

### 第 4 步：处理否定（失效模式所在）

考虑 "not good" 和 "not bad"。BoW 分类器看到的是 `{not, good}` 和 `{not, bad}`，学到的结果取决于训练数据里哪种组合出现得更多。bigram 分类器看到的是 `not_good` 和 `not_bad`，把它们当作不同的特征来学习。这通常就够了。

在没有 bigram 的情况下，还有一个更粗糙但有效的办法：**否定作用域标注（negation scoping）**。把否定词之后直到下一个标点为止的所有词元都加上 `NOT_` 前缀。

```python
NEGATION_WORDS = {"not", "no", "never", "nor", "none", "nothing", "neither"}
NEGATION_TERMINATORS = {".", "!", "?", ",", ";"}


def apply_negation(tokens):
    out = []
    negate = False
    for token in tokens:
        if token in NEGATION_TERMINATORS:
            negate = False
            out.append(token)
            continue
        if token in NEGATION_WORDS:
            negate = True
            out.append(token)
            continue
        out.append(f"NOT_{token}" if negate else token)
    return out
```

```python
>>> apply_negation(["not", "good", "at", "all", ".", "but", "funny"])
['not', 'NOT_good', 'NOT_at', 'NOT_all', '.', 'but', 'funny']
```

现在 `good` 和 `NOT_good` 是两个不同的特征，分类器可以给它们赋予方向相反的权重。三行预处理代码，就能在情感分析基准上换来可测量的准确率提升。

### 第 5 步：真正重要的评估指标

在类别不平衡时，只看准确率会产生误导。真实情感语料通常是 70-80% 正面或 70-80% 负面；一个永远预测多数类的分类器就能拿到 80% 的准确率，但毫无价值。以下指标每一项都要报告：

- **逐类精确率（precision）和召回率（recall）。** 每个类别一对指标。对它们做宏平均（macro-average），得到一个尊重类别平衡的单一数字。
- **Macro-F1（不平衡数据的首要指标）。** 各类别 F1 分数的等权平均。类别不平衡时用它代替准确率。
- **Weighted-F1（备选）。** 和 macro 相同，但按类别频率加权。当不平衡本身具有业务意义时，与 macro-F1 一起报告。
- **混淆矩阵（confusion matrix）。** 原始计数。在相信任何标量指标之前务必先检查它；它能揭示模型混淆的是哪一对类别。
- **逐类错误样本。** 每个类别抽 5 条错误预测，逐条读一遍。没有什么能替代亲自阅读真实的错误。

对于严重不平衡的数据（> 95-5 的比例），用 **AUROC** 和 **AUPRC** 代替准确率。AUPRC 对少数类更敏感，而少数类通常正是你关心的（垃圾邮件、欺诈、罕见情感）。

**要避免的常见错误。** 在不平衡数据上报告 micro-F1 而不是 macro-F1，数字会因为被多数类主导而显得很高。macro-F1 会逼你直面少数类的表现。

```python
def evaluate(y_true, y_pred):
    tp = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 1)
    fp = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 1)
    fn = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 0)
    tn = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 0)
    precision = tp / (tp + fp) if tp + fp else 0
    recall = tp / (tp + fn) if tp + fn else 0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0
    return {"tp": tp, "fp": fp, "tn": tn, "fn": fn, "precision": precision, "recall": recall, "f1": f1}
```

## 生产实践

scikit-learn 用六行代码就能做对这件事。

```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

pipe = Pipeline([
    ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True, stop_words=None)),
    ("clf", LogisticRegression(C=1.0, max_iter=1000)),
])
pipe.fit(X_train, y_train)
print(pipe.score(X_test, y_test))
```

注意三件事。`stop_words=None` 保留了否定词。`ngram_range=(1, 2)` 加入了 bigram，让 `not_good` 成为一个特征。`sublinear_tf=True` 抑制了重复出现的词。在 SST-2 上，这三个参数就是 75% 准确率基线和 85% 准确率基线之间的差距。

### 什么时候该上 Transformer

- 讽刺检测。经典模型在这里必败，没有例外。
- 情感在文中途反转的长篇评论。
- 基于方面的情感分析（aspect-based sentiment）。"Camera was great but battery was terrible."（相机很棒，但电池很糟。）你需要把情感归属到具体方面。只有 Transformer 或结构化输出模型能做到。
- 非英语、低资源语言。多语言 BERT 免费提供一个零样本基线。

如果你需要以上任何一项，直接跳到第 7 阶段（Transformer 深入解析）。否则，TF-IDF 加 bigram 加否定处理之上的 Naive Bayes 或逻辑回归，就是你 2026 年的生产基线。

### 可复现性陷阱（再次出现）

重新训练情感模型是家常便饭，重新评估它们却不是。论文里报告的准确率数字依赖于特定的数据划分、特定的预处理、特定的分词器。如果你在对比新模型和基线时没有使用完全相同的流水线，得到的差值就是误导性的。永远在你自己的流水线上重新生成基线，而不是直接引用论文里的数字。

## 交付产物

保存为 `outputs/prompt-sentiment-baseline.md`：

```markdown
---
name: sentiment-baseline
description: Design a sentiment analysis baseline for a new dataset.
phase: 5
lesson: 05
---

Given a dataset description (domain, language, size, label granularity, latency budget), you output:

1. Feature extraction recipe. Specify tokenizer, n-gram range, stopword policy (usually keep), negation handling (scoped prefix or bigrams).
2. Classifier. Naive Bayes for baseline, logistic regression for production, transformer only if the domain needs sarcasm / aspects / cross-lingual.
3. Evaluation plan. Report precision, recall, F1, confusion matrix, and per-class error samples (not just scalars).
4. One failure mode to monitor post-deployment. Domain drift and sarcasm are the top two.

Refuse to recommend dropping stopwords for sentiment tasks. Refuse to report accuracy as the sole metric when classes are imbalanced (e.g., 90% positive). Flag subword-rich languages as needing FastText or transformer embeddings over word-level TF-IDF.
```

## 练习

1. **简单。** 在 scikit-learn 流水线中加入 `apply_negation` 作为预处理步骤，在一个小型情感数据集上测量 F1 的变化。
2. **中等。** 实现带类别权重的逻辑回归（向 scikit-learn 传入 `class_weight="balanced"`，或自己推导梯度）。在一个 90-10 类别不平衡的合成数据集上测量效果。
3. **困难。** 在情感模型的残差上训练第二个分类器，构建一个讽刺检测器。记录你的实验设置。当准确率低于随机水平时要向读者说明（二分类讽刺检测的随机水平约为 50%，大多数初次尝试都落在这附近）。

## 关键术语

| 术语 | 通俗说法 | 实际含义 |
|------|-----------------|-----------------------|
| 极性（Polarity） | 正面或负面 | 二元标签；有时扩展为中性或细粒度（五星制）。 |
| 基于方面的情感分析（Aspect-based sentiment） | 逐方面的极性 | 把情感归属到文本中提及的具体实体或属性。 |
| 否定作用域标注（Negation scoping） | 翻转附近的词元 | 给 "not" 之后直到标点为止的词元加上 `NOT_` 前缀。 |
| 拉普拉斯平滑（Laplace smoothing） | 计数加 1 | 防止 Naive Bayes 中出现零概率特征。 |
| L2 正则化（L2 regularization） | 收缩权重 | 在损失中加入 `lambda * sum(w^2)`。对稀疏文本特征必不可少。 |

## 延伸阅读

- [Pang and Lee (2008). Opinion Mining and Sentiment Analysis](https://www.cs.cornell.edu/home/llee/opinion-mining-sentiment-analysis-survey.html) — 奠基性综述。篇幅很长，但前四节涵盖了经典方法的全部内容。
- [Wang and Manning (2012). Baselines and Bigrams: Simple, Good Sentiment and Topic Classification](https://aclanthology.org/P12-2018/) — 这篇论文证明了 bigram + Naive Bayes 在短文本上很难被超越。
- [scikit-learn text feature extraction docs](https://scikit-learn.org/stable/modules/feature_extraction.html#text-feature-extraction) — `CountVectorizer`、`TfidfVectorizer` 以及所有可调参数的参考文档。
