# 高级 RAG（分块、重排序、混合检索）

> 基础 RAG 只是检索与查询最相似的前 k 个文本块。这对简单问题够用，但一旦遇到多跳推理、模糊查询和大规模语料就会失效。高级 RAG 正是「能在 10 个文档上跑通的演示」与「能在 1000 万文档上可用的系统」之间的分水岭。

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 11, Lesson 06 (RAG)
**Time:** ~90 minutes
**相关课程：** Phase 5 · 23（Chunking Strategies for RAG）完整讲解了六种分块算法——递归分块、语义分块、句子分块、父文档分块、后期分块（late chunking）、上下文检索（contextual retrieval）——并附 Vectara/Anthropic 的基准测试。本课在此基础上继续深入：混合检索、重排序、查询变换。

## 学习目标

- 实现高级分块策略（语义分块、递归分块、父子分块），保留文档结构与上下文
- 构建混合检索流水线，将 BM25 关键词匹配、语义向量检索与交叉编码器重排序器结合起来
- 应用查询变换技术（HyDE、多查询、step-back），改善模糊或复杂问题的检索效果
- 诊断并修复常见的 RAG 故障：检索到错误的文本块、答案不在上下文中、多跳推理失效

## 问题背景

你在第 06 课构建了一个基础 RAG 流水线。它在小语料上回答直接的问题没有问题。现在试试这些：

**模糊查询**："上个季度的营收是多少？"语义检索返回的是关于营收战略、营收预测以及 CFO 对营收增长看法的文本块。它们都与"营收"这个词在语义上相似，但没有一个包含真正的数字。正确的文本块写的是"2025 年 Q3 为 $47.2M"，但用的词是"earnings"（盈利）而不是"revenue"（营收）。嵌入模型认为"营收战略"比"Q3 earnings were $47.2M"更接近查询。

**多跳问题**："哪个团队的客户满意度得分提升最大？"这需要先找到每个团队的满意度得分，再进行比较，最后找出最大值。没有任何单个文本块包含答案——信息分散在各个团队的报告中。

**大语料问题**：你有 200 万个文本块，正确答案在第 1,847,293 号块里。而你的 top-5 检索返回了第 14、89,201、1,200,000、44 和 901,333 号块。它们在嵌入空间中很接近，却都不包含答案。在这种规模下，近似最近邻检索引入的误差足以把真正相关的结果挤出 top-k。

基础 RAG 失效的根源在于：向量相似度不等于相关性。一个文本块可以在语义上与查询相似，却对回答问题毫无用处。高级 RAG 用四种技术来解决这个问题：混合检索（加入关键词匹配）、重排序（更精细地为候选打分）、查询变换（在检索前修正查询）、更好的分块（在合适的粒度上检索）。

## 核心概念

### 混合检索：语义 + 关键词

语义检索（向量相似度）擅长理解含义。"How do I cancel my subscription?" 能匹配到 "Steps to terminate your plan"，尽管两者没有任何共同词汇。但它会漏掉精确匹配。"Error code E-4021" 可能匹配不到包含 "E-4021" 的文本块，因为嵌入模型把它当成了噪声。

关键词检索（BM25）正好相反。它擅长精确匹配，"E-4021" 能完美命中。但如果文档里写的是 "terminate your plan"，那么搜 "cancel my subscription" 就会一无所获。

混合检索同时运行两者，然后合并结果。

**BM25**（Best Matching 25）是标准的关键词检索算法，自 1990 年代以来一直是搜索引擎的支柱。公式如下：

```
BM25(q, d) = sum over terms t in q:
    IDF(t) * (tf(t,d) * (k1 + 1)) / (tf(t,d) + k1 * (1 - b + b * |d| / avgdl))
```

其中 tf(t,d) 是词项 t 在文档 d 中的词频，IDF(t) 是逆文档频率，|d| 是文档长度，avgdl 是平均文档长度，k1 控制词频饱和度（默认 1.2），b 控制长度归一化（默认 0.75）。

直白地说：文档包含查询词项（尤其是罕见词）时 BM25 给出更高的分数，但词项重复出现的收益是递减的。一个出现 50 次"营收"的文档，并不会比只出现 1 次的文档相关 50 倍。

### 倒数排名融合（Reciprocal Rank Fusion, RRF）

你现在有两个排序列表：一个来自向量检索，一个来自 BM25。怎么合并？倒数排名融合是标准做法。

```
RRF_score(d) = sum over rankings R:
    1 / (k + rank_R(d))
```

其中 k 是一个常数（通常取 60），用于防止排名第一的结果占据压倒性优势。

一个在向量检索中排第 1、在 BM25 中排第 5 的文档得到：1/(60+1) + 1/(60+5) = 0.0164 + 0.0154 = 0.0318

一个在向量检索中排第 3、在 BM25 中排第 2 的文档得到：1/(60+3) + 1/(60+2) = 0.0159 + 0.0161 = 0.0320

RRF 天然地平衡了两路信号。在两个列表中都排名靠前的文档得分最高；在一个列表中排第 1 但在另一个列表中缺席的文档只能拿到中等分数。这种方法之所以稳健，是因为它用的是排名而不是原始分数，所以两个系统之间分数分布的差异不会造成影响。

### 重排序

检索（无论是向量、关键词还是混合）速度快但精度有限。它使用双编码器（bi-encoder）：查询和每个文档各自独立地做嵌入，再进行比较。嵌入只需计算一次并缓存，因此可以扩展到数百万文档。

重排序使用交叉编码器（cross-encoder）：查询和候选文档一起输入模型，由模型输出相关性分数。模型同时看到两段文本，能够捕捉它们之间细粒度的交互。交叉编码器能理解 "What were Q3 earnings?" 与包含 "$47.2M in Q3" 的文本块高度相关——即使双编码器漏掉了这层联系。

代价是：交叉编码器比双编码器慢 100-1000 倍，因为它要对查询-文档对进行联合处理。你不可能为一百万个文档预先计算交叉编码器分数。解决方案：先检索一个较大的候选集（混合检索取 top-50），再用交叉编码器重排序得到最终的 top-5。

```mermaid
graph LR
    Q["Query"] --> H["Hybrid Search"]
    H --> C50["Top 50 candidates"]
    C50 --> RR["Cross-Encoder Reranker"]
    RR --> C5["Top 5 final results"]
    C5 --> P["Build prompt"]
    P --> LLM["Generate answer"]
```

常见的重排序模型（2026 年阵容）：
- Cohere Rerank 3.5：托管 API，多语言，在混合语料上召回率提升最佳
- Voyage rerank-2.5：托管 API，托管方案中延迟最低
- Jina-Reranker-v2 Multilingual：开放权重，支持 100 多种语言
- bge-reranker-v2-m3：开放权重，强力基线
- cross-encoder/ms-marco-MiniLM-L-6-v2：开放权重，可在 CPU 上运行，适合原型验证
- ColBERTv2 / Jina-ColBERT-v2：后期交互（late-interaction）多向量重排序器——打分时间复杂度是 O(tokens) 而非 O(docs)

### 查询变换

有时问题不在检索，而在查询本身。"上次那个新政策变更是怎么回事？"是一个糟糕的搜索查询：没有任何具体词项，嵌入向量含混不清，任何检索系统都无法据此找到正确的文档。

**查询重写**：把用户的查询改写成更好的搜索查询。可以由 LLM 来完成：

```
User: "What was that thing about the new policy change?"
Rewritten: "Recent policy changes and updates"
```

**HyDE（Hypothetical Document Embeddings，假设性文档嵌入）**：不直接用查询去搜索，而是先生成一个假设性答案，对它做嵌入，再去检索与之相似的真实文档。

```
Query: "What is the refund policy for enterprise?"
Hypothetical answer: "Enterprise customers are eligible for a full refund
within 60 days of purchase. Refunds are pro-rated based on the remaining
subscription period and processed within 5-7 business days."
```

对这个假设性答案做嵌入，然后检索与之相似的真实文档。其直觉在于：假设性答案在嵌入空间中比原始问题更接近真实答案。问题和答案的语言结构是不同的——通过生成假设性答案，你在嵌入空间里架起了"问题空间"与"答案空间"之间的桥梁。

HyDE 在检索前增加了一次 LLM 调用，会增加 500-2000ms 的延迟。当原始查询的检索质量很差时，这笔开销是值得的。

### 父子分块

标准分块迫使你做权衡：小块利于精确检索，大块利于提供充足上下文。父子分块消除了这一权衡。

为检索建立小块（128 个 token）的索引。当一个小块被检索到时，返回它的父块（512 个 token）放进提示词。小块负责精确匹配查询，父块负责给 LLM 提供足够的上下文来生成好的答案。

```mermaid
graph TD
    P["Parent chunk (512 tokens)<br/>Full section about refund policy"]
    C1["Child chunk (128 tokens)<br/>Standard plan: 30-day refund"]
    C2["Child chunk (128 tokens)<br/>Enterprise: 60-day pro-rated"]
    C3["Child chunk (128 tokens)<br/>Processing time: 5-7 days"]
    C4["Child chunk (128 tokens)<br/>How to submit a request"]

    P --> C1
    P --> C2
    P --> C3
    P --> C4

    Q["Query: enterprise refund?"] -.->|"matches child"| C2
    C2 -.->|"return parent"| P
```

查询 "enterprise refund?" 精确匹配到子块 C2，但提示词收到的是完整的父块 P，其中包含了处理时长和提交流程等周边上下文。

### 元数据过滤

在运行向量检索之前，先按元数据过滤语料：日期、来源、类别、作者、语言。这能缩小搜索空间，避免不相关的结果。

"上个月安全策略有什么变更？"应该只检索最近 30 天内、安全类别下的文档。没有元数据过滤的话，你会搜索整个语料库，可能检索到一篇两年前的、碰巧语义相似的安全文档。

生产级 RAG 系统会在每个文本块旁存储元数据：来源文档、创建日期、类别、作者、版本。向量数据库支持在相似度检索之前先按元数据预过滤，这对大规模场景下的性能至关重要。

### 评估

你构建了一个 RAG 系统。怎么知道它是否有效？三个指标：

**检索相关性（Recall@k）**：对一组已知相关文档的测试问题，相关文档出现在 top-k 结果中的比例是多少？如果某个问题的答案在第 47 号块里，第 47 号块有没有出现在 top-5 中？

**忠实度（Faithfulness）**：生成的答案是否有检索文档作为依据？如果检索到的文本块说"60 天退款窗口"，而模型说"90 天退款窗口"，这就是一次忠实度失败——模型在拥有正确上下文的情况下仍然产生了幻觉。

**答案正确性**：生成的答案是否与预期答案一致？这是端到端指标，综合了检索质量和生成质量。

一个简单的忠实度检查：取出生成答案中的每条断言，验证它（在实质内容上）是否出现在检索到的文本块中。如果答案包含任何检索块里没有的事实，那它很可能是幻觉。

```mermaid
graph TD
    subgraph "Evaluation Framework"
        Q["Test questions<br/>+ expected answers<br/>+ relevant doc IDs"]
        Q --> Ret["Retrieval evaluation<br/>Recall@k: are right<br/>docs retrieved?"]
        Q --> Faith["Faithfulness evaluation<br/>Is answer grounded<br/>in retrieved docs?"]
        Q --> Correct["Correctness evaluation<br/>Does answer match<br/>expected answer?"]
    end
```

## 从零实现

### 第 1 步：实现 BM25

```python
import math
from collections import Counter

class BM25:
    def __init__(self, k1=1.2, b=0.75):
        self.k1 = k1
        self.b = b
        self.docs = []
        self.doc_lengths = []
        self.avg_dl = 0
        self.doc_freqs = {}
        self.n_docs = 0

    def index(self, documents):
        self.docs = documents
        self.n_docs = len(documents)
        self.doc_lengths = []
        self.doc_freqs = {}

        for doc in documents:
            words = doc.lower().split()
            self.doc_lengths.append(len(words))
            unique_words = set(words)
            for word in unique_words:
                self.doc_freqs[word] = self.doc_freqs.get(word, 0) + 1

        self.avg_dl = sum(self.doc_lengths) / self.n_docs if self.n_docs else 1

    def score(self, query, doc_idx):
        query_words = query.lower().split()
        doc_words = self.docs[doc_idx].lower().split()
        doc_len = self.doc_lengths[doc_idx]
        word_counts = Counter(doc_words)
        score = 0.0

        for term in query_words:
            if term not in word_counts:
                continue
            tf = word_counts[term]
            df = self.doc_freqs.get(term, 0)
            idf = math.log((self.n_docs - df + 0.5) / (df + 0.5) + 1)
            numerator = tf * (self.k1 + 1)
            denominator = tf + self.k1 * (1 - self.b + self.b * doc_len / self.avg_dl)
            score += idf * numerator / denominator

        return score

    def search(self, query, top_k=10):
        scores = [(i, self.score(query, i)) for i in range(self.n_docs)]
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]
```

### 第 2 步：倒数排名融合

```python
def reciprocal_rank_fusion(ranked_lists, k=60):
    scores = {}
    for ranked_list in ranked_lists:
        for rank, (doc_id, _) in enumerate(ranked_list):
            if doc_id not in scores:
                scores[doc_id] = 0.0
            scores[doc_id] += 1.0 / (k + rank + 1)
    fused = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return fused
```

### 第 3 步：混合检索流水线

```python
def hybrid_search(query, chunks, vector_embeddings, vocab, idf, bm25_index, top_k=5, fusion_k=60):
    query_emb = tfidf_embed(query, vocab, idf)
    vector_results = search(query_emb, vector_embeddings, top_k=top_k * 3)
    bm25_results = bm25_index.search(query, top_k=top_k * 3)
    fused = reciprocal_rank_fusion([vector_results, bm25_results], k=fusion_k)
    return fused[:top_k]
```

### 第 4 步：简易重排序器

在生产环境中你会使用交叉编码器模型。这里我们构建一个利用词汇重叠、词项重要性和短语匹配来评估查询-文档相关性的重排序器。

```python
def rerank(query, candidates, chunks):
    query_words = set(query.lower().split())
    stop_words = {"the", "a", "an", "is", "are", "was", "were", "what", "how",
                  "why", "when", "where", "do", "does", "for", "of", "in", "to",
                  "and", "or", "on", "at", "by", "it", "its", "this", "that",
                  "with", "from", "be", "has", "have", "had", "not", "but"}
    query_terms = query_words - stop_words

    scored = []
    for doc_id, initial_score in candidates:
        chunk = chunks[doc_id].lower()
        chunk_words = set(chunk.split())

        term_overlap = len(query_terms & chunk_words)

        query_bigrams = set()
        q_list = [w for w in query.lower().split() if w not in stop_words]
        for i in range(len(q_list) - 1):
            query_bigrams.add(q_list[i] + " " + q_list[i + 1])
        bigram_matches = sum(1 for bg in query_bigrams if bg in chunk)

        position_boost = 0
        for term in query_terms:
            pos = chunk.find(term)
            if pos != -1 and pos < len(chunk) // 3:
                position_boost += 0.5

        rerank_score = (
            term_overlap * 1.0
            + bigram_matches * 2.0
            + position_boost
            + initial_score * 5.0
        )
        scored.append((doc_id, rerank_score))

    scored.sort(key=lambda x: x[1], reverse=True)
    return scored
```

### 第 5 步：HyDE（假设性文档嵌入）

```python
def hyde_generate_hypothesis(query):
    templates = {
        "what": "The answer to '{query}' is as follows: Based on our documentation, {topic} involves specific policies and procedures that define how the process works.",
        "how": "To address '{query}': The process involves several steps. First, you need to initiate the request. Then, the system processes it according to the defined rules.",
        "default": "Regarding '{query}': Our records indicate specific details and policies related to this topic that provide a comprehensive answer."
    }
    query_lower = query.lower()
    if query_lower.startswith("what"):
        template = templates["what"]
    elif query_lower.startswith("how"):
        template = templates["how"]
    else:
        template = templates["default"]

    topic_words = [w for w in query.lower().split()
                   if w not in {"what", "is", "the", "how", "do", "does", "a", "an",
                                "for", "of", "to", "in", "on", "at", "by", "and", "or"}]
    topic = " ".join(topic_words) if topic_words else "this topic"

    return template.format(query=query, topic=topic)


def hyde_search(query, chunks, vector_embeddings, vocab, idf, top_k=5):
    hypothesis = hyde_generate_hypothesis(query)
    hypothesis_emb = tfidf_embed(hypothesis, vocab, idf)
    results = search(hypothesis_emb, vector_embeddings, top_k)
    return results, hypothesis
```

### 第 6 步：父子分块

```python
def create_parent_child_chunks(text, parent_size=200, child_size=50):
    words = text.split()
    parents = []
    children = []
    child_to_parent = {}

    parent_idx = 0
    start = 0
    while start < len(words):
        parent_end = min(start + parent_size, len(words))
        parent_text = " ".join(words[start:parent_end])
        parents.append(parent_text)

        child_start = start
        while child_start < parent_end:
            child_end = min(child_start + child_size, parent_end)
            child_text = " ".join(words[child_start:child_end])
            child_idx = len(children)
            children.append(child_text)
            child_to_parent[child_idx] = parent_idx
            child_start += child_size

        parent_idx += 1
        start += parent_size

    return parents, children, child_to_parent
```

### 第 7 步：忠实度评估

```python
def evaluate_faithfulness(answer, retrieved_chunks):
    answer_sentences = [s.strip() for s in answer.split(".") if len(s.strip()) > 10]
    if not answer_sentences:
        return 1.0, []

    grounded = 0
    ungrounded = []
    context = " ".join(retrieved_chunks).lower()

    for sentence in answer_sentences:
        words = set(sentence.lower().split())
        stop_words = {"the", "a", "an", "is", "are", "was", "were", "and", "or",
                      "to", "of", "in", "for", "on", "at", "by", "it", "this", "that"}
        content_words = words - stop_words
        if not content_words:
            grounded += 1
            continue

        matched = sum(1 for w in content_words if w in context)
        ratio = matched / len(content_words) if content_words else 0

        if ratio >= 0.5:
            grounded += 1
        else:
            ungrounded.append(sentence)

    score = grounded / len(answer_sentences) if answer_sentences else 1.0
    return score, ungrounded


def evaluate_retrieval_recall(queries_with_relevant, retrieval_fn, k=5):
    total_recall = 0.0
    results = []

    for query, relevant_indices in queries_with_relevant:
        retrieved = retrieval_fn(query, k)
        retrieved_indices = set(idx for idx, _ in retrieved)
        relevant_set = set(relevant_indices)
        hits = len(retrieved_indices & relevant_set)
        recall = hits / len(relevant_set) if relevant_set else 1.0
        total_recall += recall
        results.append({
            "query": query,
            "recall": recall,
            "hits": hits,
            "total_relevant": len(relevant_set)
        })

    avg_recall = total_recall / len(queries_with_relevant) if queries_with_relevant else 0
    return avg_recall, results
```

## 生产实践

使用真实的交叉编码器做重排序：

```python
from sentence_transformers import CrossEncoder

reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

def rerank_with_cross_encoder(query, candidates, chunks, top_k=5):
    pairs = [(query, chunks[doc_id]) for doc_id, _ in candidates]
    scores = reranker.predict(pairs)
    scored = list(zip([doc_id for doc_id, _ in candidates], scores))
    scored.sort(key=lambda x: x[1], reverse=True)
    return scored[:top_k]
```

使用 Cohere 的托管重排序器：

```python
import cohere

co = cohere.Client()

def rerank_with_cohere(query, candidates, chunks, top_k=5):
    docs = [chunks[doc_id] for doc_id, _ in candidates]
    response = co.rerank(
        model="rerank-english-v3.0",
        query=query,
        documents=docs,
        top_n=top_k
    )
    return [(candidates[r.index][0], r.relevance_score) for r in response.results]
```

用真实 LLM 实现 HyDE：

```python
import anthropic

client = anthropic.Anthropic()

def hyde_with_llm(query):
    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=256,
        messages=[{
            "role": "user",
            "content": f"Write a short paragraph that would be a good answer to this question. Do not say you don't know. Just write what the answer would look like.\n\nQuestion: {query}"
        }]
    )
    return response.content[0].text
```

用 Weaviate 实现生产级混合检索：

```python
import weaviate

client = weaviate.connect_to_local()

collection = client.collections.get("Documents")
response = collection.query.hybrid(
    query="enterprise refund policy",
    alpha=0.5,
    limit=10
)
```

alpha 参数控制两路的权重平衡：0.0 = 纯关键词（BM25），1.0 = 纯向量，0.5 = 等权重。大多数生产系统将 alpha 设在 0.3 到 0.7 之间。

## 交付产物

本课产出：
- `outputs/prompt-advanced-rag-debugger.md` —— 一个用于诊断和修复 RAG 质量问题的提示词
- `outputs/skill-advanced-rag.md` —— 一个用于构建带混合检索和重排序的生产级 RAG 的技能

## 练习

1. 在示例文档上比较 BM25、向量检索和混合检索。对 5 个测试查询中的每一个，记录哪种方法把最相关的文本块排在第 1 位。混合检索应该在 5 个查询中至少胜出 3 个。

2. 实现一个元数据过滤器。为每个文档添加一个 "category" 字段（security、billing、api、product）。在运行向量检索之前，先把文本块过滤到相关类别。用 "What encryption is used?" 测试，验证它只检索 security 类别的文本块。

3. 用第 06 课的简易生成函数构建完整的 HyDE 流水线。在全部 5 个测试查询上，比较直接查询检索与 HyDE 检索的检索质量（top-3 相关性）。对于模糊查询，HyDE 应能改善结果。

4. 在示例文档上实现父子分块策略。使用 child_size=30 和 parent_size=100。用子块检索，但在提示词中返回父块。把生成的答案与 chunk_size=50 的标准分块对比。

5. 创建一个评估数据集：10 个已知答案文本块的问题。分别测量 (a) 仅向量检索、(b) 仅 BM25、(c) 混合检索、(d) 混合 + 重排序 的 Recall@3、Recall@5 和 Recall@10。绘制结果图表，找出重排序帮助最大的场景。

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|----------------|----------------------|
| BM25 | "关键词检索" | 一种概率排序算法，根据词频、逆文档频率和文档长度归一化为文档打分 |
| 混合检索 | "两全其美" | 并行运行语义（向量）检索和关键词（BM25）检索，再用排名融合合并结果 |
| 倒数排名融合（RRF） | "合并排序列表" | 把每个文档在所有排序列表中的 1/(k + rank) 求和，以合并多个排序列表 |
| 重排序 | "二次打分" | 使用计算开销更大的交叉编码器模型，对初步检索得到的候选集重新打分 |
| 交叉编码器（Cross-encoder） | "查询-文档联合模型" | 将查询和文档作为单一输入、输出相关性分数的模型；比双编码器更准确，但用于全语料检索太慢 |
| 双编码器（Bi-encoder） | "独立嵌入模型" | 对查询和文档分别独立做嵌入的模型；因嵌入可预计算而速度快，但准确性不如交叉编码器 |
| HyDE | "用假答案去搜索" | 为查询生成一个假设性答案，对其做嵌入，再检索与之相似的真实文档 |
| 父子分块 | "小块检索，大块上下文" | 索引小块以实现精确检索，但返回更大的父块以提供充足上下文 |
| 元数据过滤 | "先收窄再检索" | 在运行向量检索之前按属性（日期、来源、类别）过滤文档，缩小搜索空间 |
| 忠实度（Faithfulness） | "答案有没有依据" | 生成的答案是否有检索文档支撑，而非来自模型训练数据的幻觉 |

## 延伸阅读

- Robertson & Zaragoza, "The Probabilistic Relevance Framework: BM25 and Beyond" (2009) —— BM25 的权威参考文献，阐释了公式背后的概率论基础
- Cormack et al., "Reciprocal Rank Fusion Outperforms Condorcet and Individual Rank Learning Methods" (2009) —— RRF 的原始论文，证明它击败了更复杂的融合方法
- Gao et al., "Precise Zero-Shot Dense Retrieval without Relevance Labels" (2022) —— HyDE 论文，证明假设性文档嵌入无需任何训练数据即可改善检索
- Nogueira & Cho, "Passage Re-ranking with BERT" (2019) —— 证明了在 BM25 之上叠加交叉编码器重排序能显著提升检索质量
- [Khattab et al., "DSPy: Compiling Declarative Language Model Calls into Self-Improving Pipelines" (2023)](https://arxiv.org/abs/2310.03714) —— 把提示词构造和权重选择视为检索流水线上的优化问题；想了解如何"编程 LLM"而不是"提示 LLM"，读这篇。
- [Edge et al., "From Local to Global: A Graph RAG Approach to Query-Focused Summarization" (Microsoft Research 2024)](https://arxiv.org/abs/2404.16130) —— GraphRAG 论文：实体-关系抽取 + Leiden 社区检测，用于面向查询的摘要；提出了全局检索与局部检索的区分。
- [Asai et al., "Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection" (ICLR 2024)](https://arxiv.org/abs/2310.11511) —— 带反思 token 的自评估 RAG；超越静态"先检索后生成"范式的智能体前沿。
- [LangChain Query Construction blog](https://blog.langchain.dev/query-construction/) —— 如何在检索前将自然语言查询转换为结构化数据库查询（Text-to-SQL、Cypher）。
