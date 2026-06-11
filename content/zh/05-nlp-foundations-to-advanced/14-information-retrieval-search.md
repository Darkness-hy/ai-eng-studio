# 信息检索与搜索

> BM25 精确但脆弱。稠密检索撒网很广却会漏掉关键词。混合检索是 2026 年的默认方案，其余一切都只是调优。

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 5 · 02 (BoW + TF-IDF), Phase 5 · 04 (GloVe, FastText, Subword)
**Time:** ~75 minutes

## 问题背景

用户输入"骗钱的人会怎么样"，期望找到真正覆盖这种情况的法条："Section 420 IPC"。关键词搜索完全找不到它（没有共同词汇）。如果嵌入模型没有在法律文本上训练过，语义搜索也会漏掉它。真实的搜索系统必须同时应对这两种情况。

信息检索（Information Retrieval, IR）是每个 RAG 系统、每个搜索框、每个文档站点模糊查找功能背后的流水线。2026 年在生产环境中真正有效的架构不是单一方法，而是一条由互补方法组成的链路，每一环都能捕获前一环的失败案例。

本课将逐一构建每个组件，并指出每个组件各自能捕获哪些失败。

## 核心概念

![Hybrid retrieval: BM25 + dense + RRF + cross-encoder rerank](../assets/retrieval.svg)

四个层次，按需选用。

1. **稀疏检索（BM25）。** 速度快，精确匹配上很准，但语义上很糟。基于倒排索引运行，在百万级文档上每次查询不到 10ms。法条引用、产品编码、错误信息、命名实体都能命中。
2. **稠密检索。** 把查询和文档编码成向量，做最近邻搜索。能捕捉同义改写和语义相似性，但会漏掉只差一个字符的精确关键词匹配。配合 FAISS 或向量数据库，每次查询 50-200ms。
3. **融合。** 合并稀疏和稠密的排序列表。倒数排名融合（Reciprocal Rank Fusion, RRF）是省心的默认选择，因为它忽略原始分数（两者尺度不同），只使用排名位置。当你明确知道某个信号在你的领域占主导时，加权融合是一个可选项。
4. **交叉编码器重排序。** 取融合后的 top-30，用交叉编码器（把查询和文档拼在一起，为每个配对打分）重新评分，保留 top-5。交叉编码器对每个配对的处理比双编码器慢，但准确得多。只在 top-30 上运行就能摊薄成本。

三路检索（BM25 + 稠密 + SPLADE 这类学习型稀疏检索）在 2026 年的基准测试中优于两路检索，但需要支持学习型稀疏索引的基础设施。对大多数团队来说，两路检索加交叉编码器重排序是最佳平衡点。

## 从零实现

### 第 1 步：从零实现 BM25

```python
import math
import re
from collections import Counter

TOKEN_RE = re.compile(r"[a-z0-9]+")


def tokenize(text):
    return TOKEN_RE.findall(text.lower())


class BM25:
    def __init__(self, corpus, k1=1.5, b=0.75):
        if not corpus:
            raise ValueError("corpus must not be empty")
        self.corpus = [tokenize(d) for d in corpus]
        self.k1 = k1
        self.b = b
        self.n_docs = len(self.corpus)
        self.avg_dl = sum(len(d) for d in self.corpus) / self.n_docs
        self.df = Counter()
        for doc in self.corpus:
            for term in set(doc):
                self.df[term] += 1

    def idf(self, term):
        n = self.df.get(term, 0)
        return math.log(1 + (self.n_docs - n + 0.5) / (n + 0.5))

    def score(self, query, doc_idx):
        q_tokens = tokenize(query)
        doc = self.corpus[doc_idx]
        dl = len(doc)
        freq = Counter(doc)
        score = 0.0
        for term in q_tokens:
            f = freq.get(term, 0)
            if f == 0:
                continue
            numerator = f * (self.k1 + 1)
            denominator = f + self.k1 * (1 - self.b + self.b * dl / self.avg_dl)
            score += self.idf(term) * numerator / denominator
        return score

    def rank(self, query, top_k=10):
        scored = [(self.score(query, i), i) for i in range(self.n_docs)]
        scored.sort(reverse=True)
        return scored[:top_k]
```

有两个参数值得了解。`k1=1.5` 控制词频饱和度，值越高，词项重复的权重越大。`b=0.75` 控制长度归一化，0 表示完全忽略文档长度，1 表示完全归一化。这两个默认值是 Robertson 在原始论文中给出的建议值，很少需要调整。

### 第 2 步：用双编码器做稠密检索

```python
from sentence_transformers import SentenceTransformer
import numpy as np


def build_dense_index(corpus, model_id="sentence-transformers/all-MiniLM-L6-v2"):
    encoder = SentenceTransformer(model_id)
    embeddings = encoder.encode(corpus, normalize_embeddings=True)
    return encoder, embeddings


def dense_search(encoder, embeddings, query, top_k=10):
    q_emb = encoder.encode([query], normalize_embeddings=True)
    sims = (embeddings @ q_emb.T).flatten()
    order = np.argsort(-sims)[:top_k]
    return [(float(sims[i]), int(i)) for i in order]
```

对嵌入做 L2 归一化，这样点积就等于余弦相似度。`all-MiniLM-L6-v2` 是 384 维的，速度快，对大多数英文检索来说足够强。多语言场景用 `paraphrase-multilingual-MiniLM-L12-v2`。追求最高准确率则用 `bge-large-en-v1.5` 或 `e5-large-v2`。

### 第 3 步：倒数排名融合

```python
def reciprocal_rank_fusion(rankings, k=60):
    scores = {}
    for ranking in rankings:
        for rank, (_, doc_idx) in enumerate(ranking):
            scores[doc_idx] = scores.get(doc_idx, 0.0) + 1.0 / (k + rank + 1)
    fused = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [(score, doc_idx) for doc_idx, score in fused]
```

常数 `k=60` 来自 RRF 的原始论文。`k` 越大，排名差异的贡献越被拉平；`k` 越小，靠前的排名越占主导。60 是论文发表时给出的默认值，很少需要调整。

### 第 4 步：混合搜索 + 重排序

```python
from sentence_transformers import CrossEncoder

reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")


def hybrid_search(query, bm25, encoder, dense_embeddings, corpus, top_k=5, pool_size=30, reranker=reranker):
    sparse_ranking = bm25.rank(query, top_k=pool_size)
    dense_ranking = dense_search(encoder, dense_embeddings, query, top_k=pool_size)
    fused = reciprocal_rank_fusion([sparse_ranking, dense_ranking])[:pool_size]

    pairs = [(query, corpus[doc_idx]) for _, doc_idx in fused]
    scores = reranker.predict(pairs)
    reranked = sorted(zip(scores, [doc_idx for _, doc_idx in fused]), reverse=True)
    return reranked[:top_k]
```

三个阶段组合在一起。BM25 找词面匹配，稠密检索找语义匹配，RRF 在无需校准分数的情况下合并两个排序。交叉编码器把查询和文档拼在一起对 top-30 重新打分，捕捉到双编码器漏掉的细粒度相关性。最后保留 top-5。

### 第 5 步：评估

| 指标 | 含义 |
|--------|---------|
| Recall@k | 在正确文档确实存在的查询中，有多大比例它出现在 top-k 里？ |
| MRR（平均倒数排名） | 首个相关文档排名的倒数（1/rank）的平均值。 |
| nDCG@k | 考虑相关性的等级差异，而不只是二元的相关/不相关。 |

具体到 RAG，检索器的 **Recall@k** 是最重要的数字。如果正确的段落不在检索结果集中，阅读器（reader）根本无法回答。

调试技巧：对失败的查询，对比稀疏和稠密两边的排序结果。如果一边找到了正确文档而另一边没有，那要么是词汇不匹配（修复方法：补上缺失的那一路），要么是语义歧义（修复方法：换更好的嵌入模型或加重排序器）。

## 生产实践

2026 年的技术栈：

| 规模 | 技术栈 |
|-------|-------|
| 1k-100k 文档 | 内存中的 BM25 + `all-MiniLM-L6-v2` 嵌入 + RRF。不需要单独的数据库。 |
| 100k-10M 文档 | 稠密侧用 FAISS 或 pgvector + BM25 侧用 Elasticsearch / OpenSearch。并行运行。 |
| 10M+ 文档 | 支持混合检索的 Qdrant / Weaviate / Vespa / Milvus。在 top-30 上做交叉编码器重排序。 |
| 质量最优的前沿方案 | 三路检索（BM25 + 稠密 + SPLADE）+ ColBERT 晚期交互重排序 |

无论选哪种，都要为评估预留预算。先做检索召回率的基准测试，再做端到端 RAG 准确率的基准测试。检索器漏掉的东西，阅读器是补救不了的。

### 2026 年生产 RAG 的血泪教训

- **80% 的 RAG 失败可以追溯到数据摄入和分块，而不是模型。** 团队花数周时间换 LLM、调提示词，而检索却悄悄地每三次查询就返回一次错误的上下文。先修分块。
- **分块策略比块大小更重要。** 固定大小切分会切坏表格、代码和嵌套标题。句子感知切分是默认选择；对技术文档和产品手册，语义分块或基于 LLM 的分块物有所值。
- **父文档模式。** 检索小的"子"块以保证精确性。当同一父级章节下的多个子块同时出现时，换入整个父级块以保留上下文。这能稳定提升回答质量，且无需重新训练。
- **k_rerank=3 通常是最优的。** 超过这个数之后，每多一个块只会增加 token 成本和生成延迟，却不会提升回答质量。如果对你来说 k=8 仍然好于 k=3，说明重排序器表现不佳。
- **HyDE / 查询扩展。** 从查询生成一个假设性答案，对它做嵌入，再去检索。这能弥合简短问题与长文档之间的措辞鸿沟。无需训练就能白拿的精度提升。
- **上下文预算控制在 8K token 以内。** 如果经常触顶这个上限，说明重排序器的阈值太松。
- **给一切做版本管理。** 提示词、分块规则、嵌入模型、重排序器。任何漂移都会悄无声息地破坏回答质量。在 CI 中对忠实度（faithfulness）、上下文精度和未回答问题率设门禁，在用户看到之前拦截回归。
- **三路检索（BM25 + 稠密 + SPLADE 这类学习型稀疏检索）优于两路**，2026 年基准测试已证实，对专有名词与语义混合的查询尤其如此。等基础设施支持 SPLADE 索引后就上线它。

根据 2026 年的行业测量数据，恰当的检索设计能把幻觉减少 70-90%。RAG 性能提升大多来自更好的检索，而不是模型微调。

## 交付产物

保存为 `outputs/skill-retrieval-picker.md`：

```markdown
---
name: retrieval-picker
description: Pick a retrieval stack for a given corpus and query pattern.
version: 1.0.0
phase: 5
lesson: 14
tags: [nlp, retrieval, rag, search]
---

Given requirements (corpus size, query pattern, latency budget, quality bar, infra constraints), output:

1. Stack. BM25 only, dense only, hybrid (BM25 + dense + RRF), hybrid + cross-encoder rerank, or three-way (BM25 + dense + learned-sparse).
2. Dense encoder. Name the specific model. Match to language(s), domain, and context length.
3. Reranker. Name the specific cross-encoder model if used. Flag that rerank adds 30-100ms latency on top-30.
4. Evaluation plan. Recall@10 is the primary retriever metric. MRR for multi-answer. Baseline first, incremental improvements measured against it.

Refuse to recommend dense-only for corpora with named entities, error codes, or product SKUs unless the user has evidence dense handles exact matches. Refuse to skip reranking for high-stakes retrieval (legal, medical) where the final top-5 decides the user's answer.
```

## 练习

1. **简单。** 在一个 500 篇文档的语料上实现上面的 `hybrid_search`。测试 20 个查询，比较纯 BM25、纯稠密和混合检索三者的 recall@5。
2. **中等。** 加入 MRR 计算。对每个已知正确文档的测试查询，找出正确文档在 BM25、稠密和混合三种排序中的排名，并分别报告 MRR。
3. **困难。** 用 MultipleNegativesRankingLoss（Sentence Transformers）在你的领域上微调一个稠密编码器。用 500 个查询-文档对构建训练集，对比微调前后的召回率。

## 关键术语

| 术语 | 大家怎么叫 | 实际含义 |
|------|-----------------|-----------------------|
| BM25 | 关键词搜索 | Okapi BM25。根据词频、IDF 和长度为文档打分。 |
| 稠密检索 | 向量搜索 | 把查询和文档编码成向量，寻找最近邻。 |
| 双编码器 | 嵌入模型 | 独立编码查询和文档。查询时速度快。 |
| 交叉编码器 | 重排序模型 | 把查询和文档拼在一起编码。慢但准确。 |
| RRF | 排名融合 | 通过累加 `1/(k + rank)` 来合并两个排序。 |
| Recall@k | 检索指标 | 相关文档出现在 top-k 中的查询所占比例。 |

## 延伸阅读

- [Robertson and Zaragoza (2009). The Probabilistic Relevance Framework: BM25 and Beyond](https://www.staff.city.ac.uk/~sbrp622/papers/foundations_bm25_review.pdf) — BM25 的权威论述。
- [Karpukhin et al. (2020). Dense Passage Retrieval for Open-Domain QA](https://arxiv.org/abs/2004.04906) — DPR，双编码器的经典之作。
- [Formal et al. (2021). SPLADE: Sparse Lexical and Expansion Model](https://arxiv.org/abs/2107.05720) — 缩小与稠密检索差距的学习型稀疏检索器。
- [Cormack, Clarke, Büttcher (2009). Reciprocal Rank Fusion outperforms Condorcet and individual Rank Learning Methods](https://plg.uwaterloo.ca/~gvcormac/cormacksigir09-rrf.pdf) — RRF 论文。
- [Khattab and Zaharia (2020). ColBERT: Efficient and Effective Passage Search](https://arxiv.org/abs/2004.12832) — 晚期交互检索。
