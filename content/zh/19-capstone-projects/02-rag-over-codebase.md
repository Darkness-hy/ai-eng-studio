# 毕业项目 02 — 基于代码库的 RAG（跨仓库语义搜索）

> 到了 2026 年，每一家认真做工程的组织都在运行一套理解语义、而非只会匹配字符串的内部代码搜索。Sourcegraph Amp、Cursor 的代码库问答、Augment 的企业级图谱、Aider 的 repomap、Pinterest 的内部 MCP——形态如出一辙：摄取多个仓库，用 tree-sitter 解析，对函数级和类级代码块做嵌入，混合搜索，重排序，最后给出带引用的回答。这个毕业项目要求你构建一个这样的系统：覆盖 10 个仓库、200 万行代码，并能在每次 git push 时扛住增量重建索引。

**Type:** Capstone
**Languages:** Python (ingestion), TypeScript (API + UI)
**Prerequisites:** Phase 5 (NLP foundations), Phase 7 (transformers), Phase 11 (LLM engineering), Phase 13 (tools), Phase 17 (infrastructure)
**涉及阶段：** P5 · P7 · P11 · P13 · P17
**Time:** 30 hours

## 问题背景

到 2026 年，每一个前沿编码智能体都自带代码库检索层，因为仅靠上下文窗口解决不了跨仓库问题。Claude 的 100 万 token 上下文有帮助，但它并不能消除对带排序检索的需求。在原始代码块上做朴素的余弦搜索，会在生成代码、monorepo 重复代码以及很少被导入的长尾符号上污染结果。生产级的答案是：在 AST 感知的代码块上做混合（稠密 + BM25）搜索，配合重排序器，并以符号引用图作为支撑。

你将通过索引一个真实的仓库群——而不是单个教程仓库——并测量 MRR@10、引用忠实度和增量新鲜度来学习这些。失败模式都出在基础设施层面：一个 10 万文件的 monorepo、一次改动了一半文件的 push、一个需要跨越四个仓库才能正确回答的查询。

## 核心概念

一条 AST 感知的摄取流水线用 tree-sitter 解析每个文件，提取函数和类节点，并在节点边界处分块，而不是按固定 token 窗口切分。每个代码块获得三种表示：稠密嵌入（Voyage-code-3 或 nomic-embed-code）、稀疏 BM25 词项，以及一段简短的自然语言摘要。摘要带来了第三种可检索的模态——用户问「X 是如何授权的」时，摘要里会出现 "authz"，即便代码里只有 `check_permission`。

检索是混合式的。一条查询同时触发稠密和 BM25 搜索，合并 top-k，再把并集交给交叉编码器（cross-encoder）重排序器（Cohere rerank-3 或 bge-reranker-v2-gemma-2b）。重排序后的列表送入长上下文合成器（启用提示词缓存的 Claude Sonnet 4.7，或自托管的 Llama 3.3 70B），并要求每条论断都按文件和行号范围给出引用。没有引用的回答会被后置过滤器拒绝。

增量新鲜度才是真正的基础设施难题。git push 触发一次 diff：哪些文件变了，哪些符号变了。只有受影响的代码块需要重新嵌入。受影响的跨文件符号边（导入、方法调用）会被重新计算。索引始终保持一致，而无需在每次提交时重新处理 200 万行代码。

## 架构

```
git push --> webhook --> ingest worker (LlamaIndex Workflow)
                           |
                           v
             tree-sitter parse + AST chunk
                           |
            +--------------+----------------+
            v              v                v
          dense        BM25 index       summary (LLM)
        (Voyage / bge)  (Tantivy)        (Haiku 4.5)
            |              |                |
            +------> Qdrant / pgvector <----+
                            |
                            v
                      symbol graph (Neo4j / kuzu)
                            |
  query --> LangGraph agent (retrieve -> rerank -> synth)
                            |
                            v
                 Claude Sonnet 4.7 1M context
                            |
                            v
                 answer + file:line citations
```

## 技术栈

- 解析：tree-sitter，支持 17 种语言语法（Python、TS、Rust、Go、Java、C++ 等）
- 稠密嵌入：Voyage-code-3（托管）或 nomic-embed-code-v1.5（自托管），bge-code-v1 作为后备
- 稀疏索引：Tantivy（Rust），使用 BM25F，按符号名与函数体做字段加权
- 向量数据库：Qdrant 1.12（带混合搜索），或 pgvector + pgvectorscale（适合 5000 万向量以下的团队）
- 代码块摘要模型：Claude Haiku 4.5 或 Gemini 2.5 Flash，启用提示词缓存
- 重排序器：Cohere rerank-3 或自托管的 bge-reranker-v2-gemma-2b
- 编排：摄取用 LlamaIndex Workflows，查询智能体用 LangGraph
- 合成器：Claude Sonnet 4.7（100 万上下文），启用提示词缓存
- 符号图：Neo4j（托管）或 kuzu（嵌入式），存储导入边和调用边
- 可观测性：Langfuse 为每个检索 + 合成步骤记录 span

## 从零实现

1. **摄取遍历器。** 在每次 push 钩子触发时迭代 git 历史，收集变更文件。对每个文件用 tree-sitter 解析，提取函数和类节点及其完整源码范围。输出代码块记录 `{repo, path, start_line, end_line, symbol, body}`。

2. **代码块摘要器。** 将代码块分批送入 Haiku 4.5，对系统前导部分启用提示词缓存。提示词："Summarize this function in one sentence, naming its public contract and side effects."（用一句话总结这个函数，说明它的公开契约和副作用。）摘要与代码块一并存储。

3. **嵌入池。** 两条并行队列：稠密嵌入（Voyage-code-3，批大小 128）和摘要嵌入（同一模型，但作用于摘要字符串）。将向量写入 Qdrant，payload 为 `{repo, path, start_line, end_line, symbol, kind}`。

4. **BM25 索引。** 字段加权的 Tantivy 索引：符号名权重 4，符号体权重 1，摘要权重 2。这样既支持「找名为 X 的函数」，也支持「找做 X 这件事的函数」。

5. **符号图。** 为每个代码块记录边：导入（此文件使用了仓库 Z 中的符号 Y）、调用（此函数调用了类 C 上的方法 M）、继承。存入 kuzu。查询时用它把检索扩展到仓库边界之外。

6. **查询智能体。** 由三个节点组成的 LangGraph。`retrieve` 并行触发稠密 + BM25 检索，按 (repo, path, symbol) 去重。`rerank` 在 top-50 上运行交叉编码器，保留 top-10。`synth` 把重排序后的代码块放进上下文调用 Claude Sonnet 4.7，缓存系统提示词，并要求给出 file:line 引用。

7. **引用强制校验。** 解析模型输出；任何缺少 `(repo/path:start-end)` 锚点的论断会被标记为重新询问或直接丢弃。只把带引用的答案返回给用户。

8. **增量重建索引。** 每次 webhook 触发时计算符号级 diff。只重新嵌入文本发生变化的代码块；对导入发生变化的代码块重新计算符号边。指标：在 200 万行代码的仓库群上，一次 50 文件的 push 在 60 秒内完成重建索引。

9. **评估。** 标注 100 个跨仓库问题，附带 file:line 黄金答案。测量 MRR@10、nDCG@10、引用忠实度（带可验证锚点的论断占比）以及 p50/p99 延迟。

## 生产实践

```
$ code-rag ask "how is S3 multipart abort wired into our retry budget?"
[retrieve]  12 chunks dense + 7 chunks bm25, 16 unique after dedup
[rerank]    top-5 kept (cohere rerank-3)
[synth]     claude-sonnet-4.7, cache hit rate 68%, 2.1s
answer:
  Multipart aborts are triggered by `AbortMultipartOnFail` in
  services/uploader/retry.go:122-148, which decrements the per-bucket
  retry budget defined in config/budgets.yaml:34-51 ...
  citations: [services/uploader/retry.go:122-148, config/budgets.yaml:34-51,
              libs/s3client/multipart.ts:44-61]
```

## 交付产物

交付一份技能文档 `outputs/skill-codebase-rag.md`。给定一组仓库语料，它能搭建起摄取流水线、混合索引和查询智能体，并对任意跨仓库问题返回带引用的答案。评分标准：

| 权重 | 标准 | 测量方式 |
|:-:|---|---|
| 25 | 检索质量 | 在 100 题留出集上的 MRR@10 与 nDCG@10 |
| 20 | 引用忠实度 | 答案中带可验证 file:line 锚点的论断占比 |
| 20 | 延迟与规模 | 在已索引语料规模下、10k QPS 时的 p95 查询延迟 |
| 20 | 增量索引正确性 | 50 文件提交从 git push 到可被搜索的时间 |
| 15 | UX 与答案格式 | 引用可点击性、代码片段预览、追问支持 |
| **100** | | |

## 练习

1. 把 Voyage-code-3 换成自托管的 nomic-embed-code。测量 MRR@10 的差值。报告启用重排序后差距是否缩小。

2. 向语料中注入 20% 的生成代码（LLM 产出的样板代码）并重新评估。观察检索污染现象。在 payload 中加入 "generated" 标记，并降低这些命中结果的权重。

3. 在你的语料规模下对 Qdrant 混合搜索与 pgvector + pgvectorscale 做基准测试。报告批大小为 1 时的 p99。

4. 增加基于采样的漂移检查：每周重跑 100 题评估，MRR@10 下降超过 5% 时告警。

5. 扩展到跨语言符号解析：一个通过 gRPC 调用 Go 服务的 Python 函数。用符号图把它们关联起来。

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|-----------------|------------------------|
| AST 感知分块（AST-aware chunking） | 「函数级切分」 | 在 tree-sitter 节点边界处切分代码，而不是按固定 token 窗口 |
| 混合搜索（hybrid search） | 「稠密 + 稀疏」 | 并行运行 BM25 和向量搜索，合并 top-k，再重排序 |
| 交叉编码器重排序（cross-encoder rerank） | 「第二阶段排序」 | 把每个（查询，候选）对放在一起打分的模型，比余弦相似度更准确 |
| 提示词缓存（prompt caching） | 「缓存的系统提示词」 | 2026 年 Claude / OpenAI 的特性，对重复出现的前缀 token 最高给出 90% 的折扣 |
| 符号图（symbol graph） | 「代码图」 | 跨文件、跨仓库的导入、调用、继承关系边 |
| 引用忠实度（citation faithfulness） | 「有据回答率」 | 用户能通过点击锚点并阅读所引代码段来验证的论断占比 |
| 增量重建索引（incremental re-index） | 「push 到可搜索的时间」 | 从 git push 到变更符号可被查询的真实耗时 |

## 延伸阅读

- [Sourcegraph Amp](https://ampcode.com) — 生产级跨仓库代码智能
- [Sourcegraph Cody RAG architecture](https://sourcegraph.com/blog/how-cody-understands-your-codebase) — 本毕业项目的参考深度解读
- [Aider repo-map](https://aider.chat/docs/repomap.html) — 基于 tree-sitter 的排序式仓库视图
- [Augment Code enterprise graph](https://www.augmentcode.com) — 商业化的符号图 RAG
- [Qdrant hybrid search docs](https://qdrant.tech/documentation/concepts/hybrid-queries/) — 参考实现
- [Voyage AI code embeddings](https://docs.voyageai.com/docs/embeddings) — Voyage-code-3 详情
- [Cohere rerank-3](https://docs.cohere.com/reference/rerank) — 交叉编码器参考
- [Pinterest MCP internal search](https://medium.com/pinterest-engineering) — 内部平台参考
