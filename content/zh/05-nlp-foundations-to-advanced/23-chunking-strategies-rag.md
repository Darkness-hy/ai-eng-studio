# RAG 的分块策略

> 分块配置对检索质量的影响不亚于嵌入模型的选择（Vectara NAACL 2025）。分块做错了，再多的重排序也救不回来。

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 5 · 14 (Information Retrieval), Phase 5 · 22 (Embedding Models)
**Time:** ~60 minutes

## 问题背景

你把一份 50 页的合同放进 RAG 系统。用户问："终止条款是什么？"检索器却返回了封面页。为什么？因为模型是在 512 token 的分块上训练的，而终止条款在第 20 页，恰好被分页符切开，周围又没有能和查询关联起来的局部关键词。

解决办法不是"换个更好的嵌入模型"，而是分块（chunking）。块该多大？要不要重叠？在哪里切分？要不要附带上下文？

2026 年 2 月的基准测试给出了一些出人意料的结果：

- Vectara 2026 年的研究：递归式 512 token 分块以 69% 对 54% 的准确率击败了语义分块。
- SPLADE + Mistral-8B 在 Natural Questions 上：重叠没有带来任何可测量的收益。
- 上下文悬崖（context cliff）：上下文达到约 2,500 token 时，回答质量急剧下降。

那个"显而易见"的答案（语义分块、20% 重叠、1000 token）往往是错的。这一课将带你建立对六种策略的直觉，并告诉你什么时候该用哪一种。

## 核心概念

![Six chunking strategies visualized on one passage](../assets/chunking.svg)

**固定分块（Fixed chunking）。** 每 N 个字符或 token 切一刀。最简单的基线。会从句子中间截断。压缩效果好，连贯性差。

**递归分块（Recursive）。** LangChain 的 `RecursiveCharacterTextSplitter`。先尝试按 `\n\n` 切分，再依次退到 `\n`、`.`、空格。回退逻辑干净利落。2026 年的默认选择。

**语义分块（Semantic）。** 对每个句子做嵌入，计算相邻句子的余弦相似度，在相似度跌破阈值的位置切分。能保持主题连贯。速度较慢；有时会产出 40 token 左右的细碎片段，反而损害检索。

**句子分块（Sentence）。** 按句子边界切分。每块一个句子，或者一个 N 句的窗口。在约 5k token 以内的文档上，效果与语义分块相当，成本却低得多。

**父文档分块（Parent-document）。** 同时存储用于检索的小型子块*和*用于提供上下文的大型父块。用子块检索，返回父块。退化表现也很平滑：即使子块质量差，返回的父块仍然合理。

**后期分块（Late chunking，2024）。** 先在 token 级别对整篇文档做嵌入，再把 token 嵌入池化成块嵌入。能保留跨块上下文。需要配合长上下文嵌入模型（BGE-M3、Jina v3）。计算开销更高。

**上下文检索（Contextual retrieval，Anthropic，2024）。** 在每个块前面拼接一段由 LLM 生成的、描述该块在文档中位置的摘要（"本块是终止条款的 3.2 节……"）。在 Anthropic 自己的基准上检索效果提升 35-50%。索引成本高昂。

### 一条胜过所有默认配置的规则

让块大小匹配查询类型：

| 查询类型 | 块大小 |
|------------|-----------|
| 事实型（"CEO 叫什么名字？"） | 256-512 token |
| 分析型 / 多跳 | 512-1024 token |
| 整节理解 | 1024-2048 token |

这是 NVIDIA 2026 年的基准结论。块要大到足以容纳答案及其局部上下文，又要小到让检索器的 top-K 结果聚焦在答案本身，而不是被上下文噪声淹没。

## 从零实现

### 第 1 步：固定分块与递归分块

```python
def chunk_fixed(text, size=512, overlap=0):
    step = size - overlap
    return [text[i:i + size] for i in range(0, len(text), step)]


def chunk_recursive(text, size=512, seps=("\n\n", "\n", ". ", " ")):
    if len(text) <= size:
        return [text]
    for sep in seps:
        if sep not in text:
            continue
        parts = text.split(sep)
        chunks = []
        buf = ""
        for p in parts:
            if len(p) > size:
                if buf:
                    chunks.append(buf)
                    buf = ""
                chunks.extend(chunk_recursive(p, size=size, seps=seps[1:] or (" ",)))
                continue
            candidate = buf + sep + p if buf else p
            if len(candidate) <= size:
                buf = candidate
            else:
                if buf:
                    chunks.append(buf)
                buf = p
        if buf:
            chunks.append(buf)
        return [c for c in chunks if c.strip()]
    return chunk_fixed(text, size)
```

### 第 2 步：语义分块

```python
def chunk_semantic(text, encoder, threshold=0.6, min_chars=200, max_chars=2048):
    sentences = split_sentences(text)
    if not sentences:
        return []
    embs = encoder.encode(sentences, normalize_embeddings=True)
    chunks = [[sentences[0]]]
    for i in range(1, len(sentences)):
        sim = float(embs[i] @ embs[i - 1])
        current_len = sum(len(s) for s in chunks[-1])
        if sim < threshold and current_len >= min_chars:
            chunks.append([sentences[i]])
        else:
            chunks[-1].append(sentences[i])

    result = []
    for group in chunks:
        text_group = " ".join(group)
        if len(text_group) > max_chars:
            result.extend(chunk_recursive(text_group, size=max_chars))
        else:
            result.append(text_group)
    return result
```

`threshold` 要在你自己的领域数据上调。阈值太高 → 碎片化；太低 → 整篇变成一个巨型块。

### 第 3 步：父文档分块

```python
def chunk_parent_child(text, parent_size=2048, child_size=256):
    parents = chunk_recursive(text, size=parent_size)
    mapping = []
    for p_idx, parent in enumerate(parents):
        children = chunk_recursive(parent, size=child_size)
        for child in children:
            mapping.append({"child": child, "parent_idx": p_idx, "parent": parent})
    return mapping


def retrieve_parent(child_query, mapping, encoder, top_k=3):
    child_embs = encoder.encode([m["child"] for m in mapping], normalize_embeddings=True)
    q_emb = encoder.encode([child_query], normalize_embeddings=True)[0]
    scores = child_embs @ q_emb
    top = np.argsort(-scores)[:top_k]
    seen, parents = set(), []
    for i in top:
        if mapping[i]["parent_idx"] not in seen:
            parents.append(mapping[i]["parent"])
            seen.add(mapping[i]["parent_idx"])
    return parents
```

关键点：对父块去重。多个子块可能映射到同一个父块，如果全部返回会白白浪费上下文。

### 第 4 步：上下文检索（Anthropic 模式）

```python
def contextualize_chunks(document, chunks, llm):
    context_prompts = [
        f"""<document>{document}</document>
Here is the chunk to situate: <chunk>{c}</chunk>
Write 50-100 words placing this chunk in the document's context."""
        for c in chunks
    ]
    contexts = llm.batch(context_prompts)
    return [f"{ctx}\n\n{c}" for ctx, c in zip(contexts, chunks)]
```

索引的是加了上下文的块。查询时，这些额外的环境信号会让检索受益。

### 第 5 步：评估

```python
def recall_at_k(queries, corpus_chunks, encoder, k=5):
    chunk_embs = encoder.encode(corpus_chunks, normalize_embeddings=True)
    hits = 0
    for q_text, gold_idxs in queries:
        q_emb = encoder.encode([q_text], normalize_embeddings=True)[0]
        top = np.argsort(-(chunk_embs @ q_emb))[:k]
        if any(i in gold_idxs for i in top):
            hits += 1
    return hits / len(queries)
```

一定要做基准测试。对你的语料来说"最好"的策略，可能和任何博客文章说的都不一样。

## 常见陷阱

- **只用事实型查询评估分块。** 多跳查询会暴露出截然不同的赢家。要用按查询类型分层的评估集。
- **语义分块不设最小尺寸。** 会产生 40 token 的碎片，损害检索效果。务必强制 `min_tokens`。
- **把重叠当成教条（cargo cult）。** 2026 年的研究发现重叠经常零收益，索引成本却翻倍。要靠测量，不要靠假设。
- **不做最小/最大限制。** 5 token 和 5000 token 的块都会让检索失效。要做截断约束。
- **跨文档分块。** 永远不要让一个块横跨两篇文档。先按文档分块，再合并。

## 生产实践

2026 年的技术栈：

| 场景 | 策略 |
|-----------|----------|
| 首次构建、语料未知 | 递归分块，512 token，无重叠 |
| 事实型问答 | 递归分块，256-512 token |
| 分析型 / 多跳 | 递归分块，512-1024 token + 父文档 |
| 大量交叉引用（合同、论文） | 后期分块或上下文检索 |
| 对话 / 会话语料 | 按轮次分块 + 说话人元数据 |
| 短文本（推文、评论） | 一篇文档 = 一个块 |

从递归 512 起步。在一个 50 条查询的评估集上测 recall@5，再据此调优。

## 交付产物

保存为 `outputs/skill-chunker.md`：

```markdown
---
name: chunker
description: Pick a chunking strategy, size, and overlap for a given corpus and query distribution.
version: 1.0.0
phase: 5
lesson: 23
tags: [nlp, rag, chunking]
---

Given a corpus (document types, avg length, domain) and query distribution (factoid / analytical / multi-hop), output:

1. Strategy. Recursive / sentence / semantic / parent-document / late / contextual. Reason.
2. Chunk size. Token count. Reason tied to query type.
3. Overlap. Default 0; justify if >0.
4. Min/max enforcement. `min_tokens`, `max_tokens` guards.
5. Evaluation plan. Recall@5 on 50-query stratified eval set (factoid, analytical, multi-hop).

Refuse any chunking strategy without min/max chunk size enforcement. Refuse overlap above 20% without an ablation showing it helps. Flag semantic chunking recommendations without a min-token floor.
```

## 练习

1. **简单。** 用 fixed(512, 0)、recursive(512, 0) 和 recursive(512, 100) 分别对一篇 20 页的文档分块。比较块数量和切分边界质量。
2. **中等。** 基于 5 篇文档构建一个 30 条查询的评估集。测量递归、语义和父文档三种策略的 recall@5。哪个赢了？和博客文章的结论一致吗？
3. **困难。** 实现上下文检索。测量相对递归基线的 MRR 提升。报告索引成本（LLM 调用次数）与准确率收益的对比。

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|-----------------|-----------------------|
| 块（Chunk） | 文档的一段 | 被嵌入、索引和检索的子文档单元。 |
| 重叠（Overlap） | 安全余量 | 相邻块共享的 N 个 token；在 2026 年的基准中常常没用。 |
| 语义分块 | 聪明的分块 | 在相邻句子嵌入相似度下降处切分。 |
| 父文档 | 两级检索 | 检索小的子块，返回更大的父块。 |
| 后期分块 | 先嵌入再分块 | 在 token 级别嵌入整篇文档，再池化成块向量。 |
| 上下文检索 | Anthropic 的技巧 | 索引前在每个块前拼接一段 LLM 生成的摘要。 |
| 上下文悬崖 | 2500 token 之墙 | RAG 中上下文约 2.5k token 时观察到的质量骤降（2026 年 1 月）。 |

## 延伸阅读

- [Yepes et al. / LangChain — Recursive Character Splitting docs](https://python.langchain.com/docs/how_to/recursive_text_splitter/) — 生产环境的默认选择。
- [Vectara (2024, NAACL 2025). Chunking configurations analysis](https://arxiv.org/abs/2410.13070) — 分块和嵌入模型的选择同等重要。
- [Jina AI — Late Chunking in Long-Context Embedding Models (2024)](https://jina.ai/news/late-chunking-in-long-context-embedding-models/) — 后期分块的原始论文。
- [Anthropic — Contextual Retrieval](https://www.anthropic.com/news/contextual-retrieval) — 用 LLM 生成的上下文前缀带来 35-50% 的检索提升。
- [NVIDIA 2026 chunk-size benchmark — Premai summary](https://blog.premai.io/rag-chunking-strategies-the-2026-benchmark-guide/) — 按查询类型选择块大小。
