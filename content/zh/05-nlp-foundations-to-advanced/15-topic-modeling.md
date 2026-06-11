# 主题建模 —— LDA 与 BERTopic

> LDA：文档是主题的混合，主题是词的分布。BERTopic：文档在嵌入空间中聚成簇，簇就是主题。目标相同，分解方式不同。

**Type:** Learn
**Languages:** Python
**Prerequisites:** Phase 5 · 02 (BoW + TF-IDF), Phase 5 · 03 (Word2Vec)
**Time:** ~45 minutes

## 问题背景

你手上有 10,000 条客服工单、50,000 篇新闻报道，或者 200,000 条推文。你需要在不通读全文的情况下知道这批文本在讲什么。你没有标注好的类别，甚至不知道一共有多少个类别。

主题建模（Topic Modeling）在无监督的条件下回答这个问题。给它一个语料库，它返回一小组语义连贯的主题，并为每篇文档给出在这些主题上的分布。

两大算法家族占据主导地位。LDA（2003）把每篇文档看作若干潜在主题的混合，把每个主题看作词上的概率分布，推断过程是贝叶斯式的。在需要混合隶属（mixed-membership）主题分配和可解释的词级概率分布的场景里，它至今仍在生产环境中使用。

BERTopic（2020）用 BERT 编码文档，用 UMAP 降维，用 HDBSCAN 聚类，再通过基于类别的 TF-IDF（class-based TF-IDF）提取主题词。在短文本、社交媒体，以及任何语义相似度比词面重叠更重要的场景中，它表现更好。它的局限是一篇文档只分配一个主题，对长文内容来说不够灵活。

这节课为两种方法都建立直觉，并明确针对给定语料该选哪一个。

## 核心概念

![LDA 混合模型与 BERTopic 聚类对比](../assets/topic-modeling.svg)

**LDA 的生成式故事。** 每个主题是词上的一个分布，每篇文档是主题的一个混合。要在某篇文档中生成一个词，先从该文档的主题混合中采样一个主题，再从该主题的词分布中采样一个词。推断则是反过来：给定观测到的词，反推每篇文档的主题分布和每个主题的词分布。具体计算由坍缩吉布斯采样（collapsed Gibbs sampling）或变分贝叶斯（variational Bayes）完成。

LDA 的关键输出：

- `doc_topic`：形状为 `(n_docs, n_topics)` 的矩阵，每行求和为 1（文档的主题混合）。
- `topic_word`：形状为 `(n_topics, vocab_size)` 的矩阵，每行求和为 1（主题的词分布）。

**BERTopic 流水线。**

1. 用句子级 Transformer（例如 `all-MiniLM-L6-v2`）编码每篇文档，得到 384 维向量。
2. 用 UMAP 把维度降到约 5 维。BERT 嵌入维度太高，不适合直接聚类。
3. 用 HDBSCAN 聚类。它基于密度，能产生大小不一的簇，并带有一个"离群点"标签。
4. 对每个簇，在该簇的文档上计算基于类别的 TF-IDF，提取最具代表性的词。

输出是每篇文档一个主题（外加一个 -1 离群点标签）。可选地，可以通过 HDBSCAN 的概率向量得到软隶属度。

## 从零实现

### 第 1 步：用 scikit-learn 实现 LDA

```python
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import numpy as np


def fit_lda(documents, n_topics=5, max_features=1000):
    cv = CountVectorizer(
        max_features=max_features,
        stop_words="english",
        min_df=2,
        max_df=0.9,
    )
    X = cv.fit_transform(documents)
    lda = LatentDirichletAllocation(
        n_components=n_topics,
        random_state=42,
        max_iter=50,
        learning_method="online",
    )
    doc_topic = lda.fit_transform(X)
    feature_names = cv.get_feature_names_out()
    return lda, cv, doc_topic, feature_names


def print_top_words(lda, feature_names, n_top=10):
    for idx, topic in enumerate(lda.components_):
        top_idx = np.argsort(-topic)[:n_top]
        words = [feature_names[i] for i in top_idx]
        print(f"topic {idx}: {' '.join(words)}")
```

注意：去除了停用词，min_df 和 max_df 分别过滤掉过于稀有和过于普遍的词，并且使用 CountVectorizer（而不是 TfidfVectorizer），因为 LDA 需要的是原始词频计数。

### 第 2 步：BERTopic（生产级）

```python
from bertopic import BERTopic

topic_model = BERTopic(
    embedding_model="sentence-transformers/all-MiniLM-L6-v2",
    min_topic_size=15,
    verbose=True,
)

topics, probs = topic_model.fit_transform(documents)
info = topic_model.get_topic_info()
print(info.head(20))
valid_topics = info[info["Topic"] != -1]["Topic"].tolist()
for topic_id in valid_topics[:5]:
    print(f"topic {topic_id}: {topic_model.get_topic(topic_id)[:10]}")
```

`Topic != -1` 这个过滤条件去掉了 BERTopic 的离群点桶（即 HDBSCAN 无法聚类的文档）。`min_topic_size` 控制 HDBSCAN 的最小簇大小；BERTopic 库的默认值是 10，本例为了匹配课程的数据规模显式设为 15。对于超过 10,000 篇文档的语料，建议增大到 50 或 100。

### 第 3 步：评估

两种方法都会输出主题词。问题在于这些词是否语义连贯。

- **主题连贯性（c_v）。** 在滑动窗口上下文中计算高频主题词两两之间的 NPMI（归一化点互信息），把得分聚合成主题向量，再用余弦相似度比较这些向量。数值越高越好。使用 `gensim.models.CoherenceModel` 并设置 `coherence="c_v"`。
- **主题多样性。** 所有主题的高频词中唯一词所占的比例。越高越好（说明主题之间不重叠）。
- **人工定性检查。** 读一遍每个主题的高频词，它们是否指向一个真实存在的东西？人的判断仍是最后一道防线。

## 如何选择

| 场景 | 选择 |
|-----------|------|
| 短文本（推文、评论、标题） | BERTopic |
| 包含主题混合的长文档 | LDA |
| 没有 GPU / 算力有限 | LDA 或 NMF |
| 需要文档级的多主题分布 | LDA |
| 集成 LLM 做主题命名 | BERTopic（直接支持） |
| 资源受限的边缘部署 | LDA |
| 追求最高语义连贯性 | BERTopic |

实践中最重要的考量是文档长度。BERT 嵌入会截断输入；LDA 的计数则对任何长度都适用。对于超出嵌入模型上下文长度的文档，要么分块后聚合，要么直接用 LDA。

## 生产实践

2026 年的技术栈：

- **BERTopic。** 短文本以及任何看重语义的场景的默认选择。
- **`gensim.models.LdaModel`。** 生产环境中的经典 LDA，成熟且久经考验。
- **`sklearn.decomposition.LatentDirichletAllocation`。** 适合做实验的简易 LDA。
- **NMF。** 非负矩阵分解（Non-negative matrix factorization）。LDA 的快速替代方案，在短文本上质量相当。
- **Top2Vec。** 设计思路与 BERTopic 类似。社区较小，但在一些基准上表现不错。
- **FASTopic。** 更新的方法，在超大语料上比 BERTopic 更快。
- **基于 LLM 的命名。** 先跑任意聚类算法，再用提示词让模型给每个簇起名字。

## 交付产物

保存为 `outputs/skill-topic-picker.md`：

```markdown
---
name: topic-picker
description: Pick LDA or BERTopic for a corpus. Specify library, knobs, evaluation.
version: 1.0.0
phase: 5
lesson: 15
tags: [nlp, topic-modeling]
---

Given a corpus description (document count, avg length, domain, language, compute budget), output:

1. Algorithm. LDA / NMF / BERTopic / Top2Vec / FASTopic. One-sentence reason.
2. Configuration. Number of topics: `recommended = max(5, round(sqrt(n_docs)))`, clamped to 200 for corpora under 40,000 docs; permit >200 only when the corpus is genuinely large (>40k) and note the increased compute cost. `min_df` / `max_df` filters and embedding model for neural approaches also belong here.
3. Evaluation. Topic coherence (c_v) via `gensim.models.CoherenceModel`, topic diversity, and a 20-sample human read.
4. Failure mode to probe. For LDA, "junk topics" absorbing stopwords and frequent terms. For BERTopic, the -1 outlier cluster swallowing ambiguous documents.

Refuse BERTopic on documents longer than the embedding model's context window without a chunking strategy. Refuse LDA on very short text (tweets, reviews under 10 tokens) as coherence collapses. Flag any n_topics choice below 5 as likely wrong; flag >200 on corpora under 40k docs as likely over-splitting.
```

## 练习

1. **简单。** 在 20 Newsgroups 数据集上用 5 个主题拟合 LDA。打印每个主题的前 10 个词，并人工给每个主题命名。算法找到真实类别了吗？
2. **中等。** 在同一份 20 Newsgroups 子集上拟合 BERTopic。对比它与 LDA 在主题数量、高频词和定性连贯性上的差异。哪种方法更干净地还原了真实类别？
3. **困难。** 在你自己的语料上分别为 LDA 和 BERTopic 计算 c_v 连贯性。每种方法分别用 5、10、20、50 个主题各跑一次，绘制连贯性随主题数量变化的曲线，并报告哪种方法在不同主题数量下更稳定。

## 关键术语

| 术语 | 大家通常怎么说 | 实际含义 |
|------|-----------------|-----------------------|
| 主题（Topic） | 语料库在讲的某件事 | 词上的概率分布（LDA），或一簇相似的文档（BERTopic）。 |
| 混合隶属（Mixed membership） | 一篇文档同时属于多个主题 | LDA 给每篇文档分配一个覆盖所有主题的分布。 |
| UMAP | 降维 | 保留局部结构的流形学习；BERTopic 中使用。 |
| HDBSCAN | 密度聚类 | 找出大小不一的簇；对离群点输出"噪声"标签（-1）。 |
| c_v 连贯性 | 主题质量指标 | 主题高频词在滑动窗口内的平均点互信息。 |

## 延伸阅读

- [Blei, Ng, Jordan (2003). Latent Dirichlet Allocation](https://www.jmlr.org/papers/volume3/blei03a/blei03a.pdf) —— LDA 原始论文。
- [Grootendorst (2022). BERTopic: Neural topic modeling with a class-based TF-IDF procedure](https://arxiv.org/abs/2203.05794) —— BERTopic 原始论文。
- [Röder, Both, Hinneburg (2015). Exploring the Space of Topic Coherence Measures](https://svn.aksw.org/papers/2015/WSDM_Topic_Evaluation/public.pdf) —— 提出 c_v 及相关指标的论文。
- [BERTopic documentation](https://maartengr.github.io/BERTopic/) —— 生产级参考文档，示例非常出色。
