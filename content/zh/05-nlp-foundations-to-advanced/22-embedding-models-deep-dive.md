# 嵌入模型——2026 深度解析

> Word2Vec 给每个词一个向量。现代嵌入模型给每段文本一个向量，支持跨语言，提供稀疏、稠密、多向量等多种视图，并且维度可以按索引需求裁剪。选错了模型，你的 RAG 就会检索到错误的内容。

**Type:** Learn
**Languages:** Python
**Prerequisites:** Phase 5 · 03 (Word2Vec), Phase 5 · 14 (Information Retrieval)
**Time:** ~60 minutes

## 问题背景

你的 RAG 系统有 40% 的概率检索到错误的段落。罪魁祸首很少是向量数据库或提示词，而是嵌入模型。

在 2026 年选择嵌入模型，意味着要在五个维度上做取舍：

1. **稠密 vs 稀疏 vs 多向量。** 每段文本一个向量，还是每个 token 一个向量，还是一个带权重的稀疏词袋。
2. **语言覆盖。** 在纯英文任务上，单语英文模型仍然占优。语料混合多种语言时，多语言模型胜出。
3. **上下文长度。** 512 token vs 8,192 vs 32,768——而实际有效容量往往只有标称最大值的 60-70%。
4. **维度预算。** 3,072 个全精度浮点数 = 每个向量 12 KB。1 亿个向量时，存储成本为每月 1,300 美元。Matryoshka 截断可将其削减 4 倍。
5. **开源权重 vs 托管服务。** 开源权重意味着你掌控整个技术栈和数据。托管服务意味着用控制权换取永远最新的模型。

这一课会把这些取舍讲清楚，让你基于证据做选择，而不是看上个季度什么模型火。

## 核心概念

![Dense, sparse, and multi-vector embeddings](../assets/embedding-modes.svg)

**稠密嵌入（Dense embeddings）。** 每段文本一个向量（通常 384-3,072 维）。用余弦相似度按语义接近程度对段落排序。代表：OpenAI `text-embedding-3-large`、BGE-M3 稠密模式、Voyage-3。默认选择。

**稀疏嵌入（Sparse embeddings）。** SPLADE 风格。Transformer 为词表中的每个 token 预测一个权重，然后把其中大部分置零。结果是一个大小为 |vocab| 的稀疏向量。它捕捉词面匹配（类似 BM25），但词项权重是学出来的。在关键词密集的查询上表现强劲。

**多向量（晚期交互，late interaction）。** ColBERTv2、Jina-ColBERT。每个 token 一个向量。用 MaxSim 打分：对每个查询 token，找到最相似的文档 token，再把分数求和。存储和打分都更昂贵，但在长查询和领域特定语料上胜出。

**BGE-M3：三者合一。** 单个模型同时输出稠密、稀疏和多向量三种表示。每种表示都可以独立查询；分数通过加权求和融合。当你想用一个 checkpoint 获得最大灵活性时，这是 2026 年的默认选择。

**Matryoshka 表示学习（Matryoshka Representation Learning）。** 训练时让向量的前 N 维本身就构成一个可用的独立嵌入。把 1,536 维向量截断到 256 维，只损失约 1% 的准确率，换来 6 倍的存储节省。OpenAI text-3、Cohere v4、Voyage-4、Jina v5、Gemini Embedding 2、Nomic v1.5+ 都支持。

### MTEB 排行榜只讲了一半的故事

Massive Text Embedding Benchmark——发布时（2022 年）涵盖 8 类任务共 56 个任务，MTEB v2 已扩展到 100 多个任务。2026 年初，Gemini Embedding 2 领跑检索榜（67.71 MTEB-R），Cohere embed-v4 领跑综合榜（65.2 MTEB），BGE-M3 领跑开源权重多语言榜（63.0）。排行榜是必要的，但不充分——务必在你自己的领域上做基准测试。

### 三层模式

| 使用场景 | 模式 |
|----------|---------|
| 快速初筛 | 稠密双编码器（BGE-M3、text-3-small） |
| 提升召回 | 稀疏（SPLADE、BGE-M3 稀疏模式）+ RRF 融合 |
| top-50 上的精排 | 多向量（ColBERTv2）或交叉编码器重排序器 |

大多数生产系统三层全用。

## 从零实现

### 第 1 步：基线——用 Sentence-BERT 做稠密嵌入

```python
from sentence_transformers import SentenceTransformer
import numpy as np

encoder = SentenceTransformer("BAAI/bge-small-en-v1.5")
corpus = [
    "The first iPhone launched in 2007.",
    "Apple released the iPod in 2001.",
    "Android is an operating system from Google.",
]
emb = encoder.encode(corpus, normalize_embeddings=True)

query = "When was the iPhone released?"
q_emb = encoder.encode([query], normalize_embeddings=True)[0]
scores = emb @ q_emb
print(sorted(enumerate(scores), key=lambda x: -x[1]))
```

`normalize_embeddings=True` 让点积等于余弦相似度。一定要设置它。

### 第 2 步：Matryoshka 截断

```python
def truncate(vectors, dim):
    out = vectors[:, :dim]
    return out / np.linalg.norm(out, axis=1, keepdims=True)

emb_256 = truncate(emb, 256)
emb_128 = truncate(emb, 128)
```

截断后要重新归一化。Nomic v1.5、OpenAI text-3 和 Voyage-4 经过专门训练，前几个截断档位几乎无损。非 Matryoshka 模型（原版 Sentence-BERT）截断后性能会急剧下降。

### 第 3 步：BGE-M3 的多功能输出

```python
from FlagEmbedding import BGEM3FlagModel

model = BGEM3FlagModel("BAAI/bge-m3", use_fp16=True)

output = model.encode(
    corpus,
    return_dense=True,
    return_sparse=True,
    return_colbert_vecs=True,
)
# output["dense_vecs"]:    (n_docs, 1024)
# output["lexical_weights"]: list of dict {token_id: weight}
# output["colbert_vecs"]:  list of (n_tokens, 1024) arrays
```

一次推理调用，三种索引。分数融合：

```python
dense_score = ... # cosine over dense_vecs
sparse_score = model.compute_lexical_matching_score(q_lex, d_lex)
colbert_score = model.colbert_score(q_col, d_col)
final = 0.4 * dense_score + 0.2 * sparse_score + 0.4 * colbert_score
```

权重要在你自己的领域上调优。

### 第 4 步：在自定义任务上跑 MTEB 评测

```python
from mteb import MTEB

tasks = ["ArguAna", "SciFact", "NFCorpus"]
evaluation = MTEB(tasks=tasks)
results = evaluation.run(encoder, output_folder="./mteb-results")
```

在*有代表性*的任务子集上运行候选模型。不要只信排行榜排名——你的领域才是关键。

### 第 5 步：从零手写余弦相似度

参见 `code/main.py`。基于平均哈希技巧（Hashing Trick）的嵌入（仅用标准库）。性能比不上 Transformer 嵌入，但展示了完整流程：分词 → 向量 → 归一化 → 点积。

## 常见陷阱

- **查询和文档用同一套编码。** 有些模型（Voyage、Jina-ColBERT）使用非对称编码——查询和文档走不同的路径。一定要查模型卡。
- **漏掉前缀。** `bge-*` 系列模型需要在查询前拼接 `"Represent this sentence for searching relevant passages: "`。忘了加会损失 3-5 个百分点的召回率。
- **Matryoshka 截得过狠。** 1,536 → 256 通常安全，1,536 → 64 则不然。要在你的评测集上验证。
- **上下文被截断。** 大多数模型会静默截断超过最大长度的输入。长文档需要分块（见第 23 课）。
- **忽视延迟长尾。** MTEB 分数掩盖了 p99 延迟。一个 600M 模型可能比 335M 模型高 2 分，但每次查询的成本是后者的 3 倍。

## 生产实践

2026 年的技术栈：

| 场景 | 选型 |
|-----------|------|
| 纯英文、追求速度、走 API | `text-embedding-3-large` 或 `voyage-3-large` |
| 开源权重、英文 | `BAAI/bge-large-en-v1.5` |
| 开源权重、多语言 | `BAAI/bge-m3` 或 `Qwen3-Embedding-8B` |
| 长上下文（32k+） | Voyage-3-large、Cohere embed-v4、Qwen3-Embedding-8B |
| 仅 CPU 部署 | Nomic Embed v2（137M 参数，MoE） |
| 存储受限 | Matryoshka 截断 + int8 量化 |
| 关键词密集的查询 | 加上 SPLADE 稀疏检索，与稠密检索做 RRF 融合 |

2026 年的常见做法：先用 BGE-M3 或 text-3-large，用 MTEB 在自己的领域上评测，如果某个领域特定模型领先超过 3 分就换。

## 交付产物

保存为 `outputs/skill-embedding-picker.md`：

```markdown
---
name: embedding-picker
description: Pick embedding model, dimension, and retrieval mode for a given corpus and deployment.
version: 1.0.0
phase: 5
lesson: 22
tags: [nlp, embeddings, retrieval]
---

Given a corpus (size, languages, domain, avg length), deployment target (cloud / edge / on-prem), latency budget, and storage budget, output:

1. Model. Named checkpoint or API. One-sentence reason.
2. Dimension. Full / Matryoshka-truncated / int8-quantized. Reason tied to storage budget.
3. Mode. Dense / sparse / multi-vector / hybrid. Reason.
4. Query prefix / template if required by the model card.
5. Evaluation plan. MTEB tasks relevant to domain + held-out domain eval with nDCG@10.

Refuse recommendations that truncate Matryoshka to <64 dims without domain validation. Refuse ColBERTv2 for corpora under 10k passages (overhead not justified). Flag long-document corpora (>8k tokens) routed to models with 512-token windows.
```

## 练习

1. **简单。** 用 `bge-small-en-v1.5` 对 100 个句子分别在全维度（384）和 Matryoshka 128 维下编码。在 10 个查询上测量 MRR 的下降幅度。
2. **中等。** 在你所在领域的 500 个段落上比较 BGE-M3 的稠密、稀疏和 colbert 模式。哪种在 recall@10 上胜出？RRF 融合能否超过最好的单一模式？
3. **困难。** 在你最重要的 2 个领域任务上对三个候选模型跑 MTEB。报告 MTEB 分数、100 条查询批次的 p99 延迟，以及每百万次查询的美元成本。选出 Pareto 最优的那个。

## 关键术语

| 术语 | 大家怎么说 | 实际含义 |
|------|-----------------|-----------------------|
| 稠密嵌入 | 「那个向量」 | 每段文本一个固定大小的向量，用余弦相似度排序。 |
| 稀疏嵌入 | 学习版 BM25 | 词表中每个 token 一个权重，大部分为零，端到端训练。 |
| 多向量 | ColBERT 风格 | 每个 token 一个向量，MaxSim 打分；索引更大，召回更好。 |
| Matryoshka | 俄罗斯套娃技巧 | 前 N 维本身就是一个有效的小尺寸嵌入。 |
| MTEB | 那个基准 | Massive Text Embedding Benchmark——发布时 56 个任务，v2 超过 100 个。 |
| BEIR | 那个检索基准 | 18 个零样本检索任务，常被引用来衡量跨领域鲁棒性。 |
| 非对称编码 | 查询 ≠ 文档路径 | 模型对查询和文档使用不同的投影。 |

## 延伸阅读

- [Reimers, Gurevych (2019). Sentence-BERT](https://arxiv.org/abs/1908.10084) —— 双编码器的开山之作。
- [Muennighoff et al. (2022). MTEB: Massive Text Embedding Benchmark](https://arxiv.org/abs/2210.07316) —— 排行榜论文。
- [Chen et al. (2024). BGE-M3: Multi-lingual, Multi-functionality, Multi-granularity](https://arxiv.org/abs/2402.03216) —— 三模式统一模型。
- [Kusupati et al. (2022). Matryoshka Representation Learning](https://arxiv.org/abs/2205.13147) —— 维度阶梯训练目标。
- [Santhanam et al. (2022). ColBERTv2: Effective and Efficient Retrieval via Lightweight Late Interaction](https://arxiv.org/abs/2112.01488) —— 晚期交互的生产实践。
- [MTEB leaderboard on Hugging Face](https://huggingface.co/spaces/mteb/leaderboard) —— 实时排名。
