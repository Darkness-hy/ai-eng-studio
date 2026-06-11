# 分块策略对比

> 分块（chunking）决定了检索器究竟能召回什么。边界切错了，再好的嵌入模型、重排器或 LLM 都无法在下游挽回损失。

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 11 lessons 04 (embeddings), 06 (RAG), 07 (advanced RAG); Phase 19 Track B foundations (lessons 20-29)
**Time:** ~90 minutes

## 学习目标
- 从零实现五种分块策略：固定窗口、句子分块、递归切分、语义聚类、基于 markdown 标题的结构化分块。
- 在带有黄金标注答案区间（gold-labeled answer spans）的测试语料上测量 recall@k，并解释为什么某个策略在散文文本上胜出，而另一个策略在技术文档上胜出。
- 读懂分块长度分布图，识别每种策略引入的失败模式：孤儿句、符号中间截断、仅含标题的分块、语义漂移。
- 不跑基准测试，仅通过检查三个属性——文档类型、平均段落长度、格式是否带有显式结构——就能为新语料选出默认策略。

## 问题背景

每个 RAG 流水线都始于把源文档切成小块：小到嵌入模型能容纳，大到每一块都承载一个自洽的观点。在哪里下刀不是一个普通超参数，它是检索器所能返回内容的上界。

一个询问"预算中止阈值是什么样的"的查询，只有在包含该中止阈值的分块可达时才可能成功。如果固定窗口切分器把阈值数值从其上下文中切了出去，嵌入就会移动到另一个聚类，BM25 分数下降，重排器看到的全是噪声，LLM 生成的答案也就是错的。2024 年的论文 "LongRAG: Enhancing Retrieval-Augmented Generation with Long-context LLMs" 测得仅因分块选择不同，检索召回率就有 35 个百分点的绝对差距。2025 年关于上下文分块标题（contextual chunk headers）的后续工作缩小了这一差距，但没有完全消除。

本课并排构建五种策略，在带有黄金标注答案区间的测试语料上运行它们，让你亲自解读召回率数字。

## 核心概念

```mermaid
flowchart LR
  Doc[Source Document] --> S1[Fixed Window]
  Doc --> S2[Sentence]
  Doc --> S3[Recursive Split]
  Doc --> S4[Semantic Cluster]
  Doc --> S5[Structural Markdown]
  S1 --> Chunks1[Chunks]
  S2 --> Chunks2[Chunks]
  S3 --> Chunks3[Chunks]
  S4 --> Chunks4[Chunks]
  S5 --> Chunks5[Chunks]
  Chunks1 --> Index[Embedding Index]
  Chunks2 --> Index
  Chunks3 --> Index
  Chunks4 --> Index
  Chunks5 --> Index
  Index --> Eval[Recall@k vs Gold Spans]
```

### 固定窗口

最朴素的基线。每隔 N 个字符切一刀。可以选择加入重叠，这样在位置 N 被切断的句子，会完整出现在从位置 N - overlap 开始的下一个分块里。快速、确定性强，但边界处理极差。把它当作对照组，而不是默认选项。

### 句子分块

用正则表达式或简单状态机按句子边界切分。把一个或多个句子打包进一个分块，直到达到目标字符预算。不会从单词中间切断，但仍会从段落和章节中间切断。这是许多早期 RAG 流水线的默认方案，对于没有其他结构的散文文本是一个合理选择。

### 递归切分

由 2023 年前后的各类库带火的层级策略。先尝试用最强的分隔符切分（双换行符、段落），失败则退到次一级（单换行符），再退到句子，最后退到字符。当分块满足预算时递归终止。对结构不一致的文档表现强劲，因为它能按区域自适应。

### 语义聚类

对每个句子做嵌入。把共享同一主题中心（centroid）的连续句子聚成一类。每当与中心的滑动相似度跌破阈值时就切一刀。边界反映的是语义而不是字符数。构建更慢，且依赖嵌入模型，但对在段落内部切换话题的文档有更强的韧性。

### 基于 markdown 标题的结构化分块

对于带有显式结构的文档（markdown、reStructuredText、RFC 风格的编号章节），在标题边界处切分。每个分块由标题及其下方直到下一个同级或更高级标题之前的所有内容组成。每个主题的分块最小，但只有在语料格式良好时才可用。

### recall@k 如何衡量边界选择

每个带黄金标注的查询都携带答案区间在源文档中的精确字符偏移量。分块之后，你问一个问题：检索器返回的 top-k 分块里，有没有任何一个与黄金区间重叠？有则该查询的 recall@k 为 1，没有则为 0。在整个查询集上取平均。对每种策略跑同一套评测，分数差距就能告诉你哪种边界策略经得起你手头语料的考验。

## 从零实现

`code/main.py` 实现了：

- `fixed_window(text, size, overlap)` —— 基线。
- `sentence_chunks(text, target)` —— 简单的句子打包器。
- `recursive_split(text, separators, target)` —— 层级递归切分。
- `semantic_chunks(text, similarity_threshold)` —— 基于确定性模拟嵌入的中心聚类。
- `structural_markdown(text)` —— 标题感知的切分器。
- `mock_embed(text, dim)` —— 基于哈希的嵌入，让整个流程可以离线运行。
- `DenseIndex` —— 与 Phase 19 Track B 混合检索课中相同的结构。
- `eval_recall(strategy, corpus, queries, k)` —— 对比评测循环。
- 一个 `main()`，在测试语料上运行每种策略并打印 recall@k 表格。

运行：

```bash
python3 code/main.py
```

输出是一张小表格，每行一个策略，每列一个 k 值。句子分块在结构化测试语料上落败。结构化 markdown 分块在 markdown 测试语料上获胜。递归切分在混合语料上站得住脚，因为递归会自适应。语义聚类在没有可用结构线索的散文语料上获胜。

## 表格掩盖不了的失败模式

**孤儿句。** 句子打包会产生缺少主题句的分块，嵌入随之指向错误的聚类。

**符号中间截断。** 固定窗口切到代码或 YAML 内部时，会把一个标识符切成两半，两半各自嵌入成噪声。

**仅含标题的分块。** 结构化 markdown 分块会产出只包含 `## Title` 的分块。要么过滤掉它们，要么把下一个分块的第一段附加进去。

**语义漂移。** 当语料整体主题高度一致时，语义聚类会切得不够。一个 5000 字符的分块把许多具体答案压进了一个模糊的嵌入里。应将语义分块与硬性字符上限组合使用。

**过期嵌入。** 语义聚类依赖嵌入模型。换了模型，分块也就跟着变。要么把分块模型与检索模型分开固定版本，要么一起重建索引。

## 不跑基准测试也能选出默认策略

三个属性决定了新语料的默认分块器。

| 属性 | 取值 | 默认策略 |
|----------|-------|---------|
| 文档类型 | 无结构的散文 | 递归切分，目标 800 |
| 文档类型 | Markdown / RFC / API 文档 | 结构化 markdown 分块 |
| 文档类型 | 代码 | AST 感知（超出本课范围；见 Phase 19 lesson 02） |
| 段落长度 | 长、单一主题 | 句子分块，目标 500 |
| 段落长度 | 短、主题混杂 | 语义分块，阈值 0.6 |

拿不准时，选递归切分。它是最强的单策略基线。

## 生产实践

生产环境的实践模式：

- 上线新流水线之前先跑评测；不要盲信你所用库的默认策略。
- 每次更换嵌入模型或语料构成时重新跑评测；胜出者取决于语料。
- 把策略名称持久化到每个分块的元数据里，方便日后归因回归问题。

## 交付产物

Track F 在 lesson 69 的端到端 RAG 系统把这里选出的分块器作为第一阶段。lesson 68 的评测框架读取的 recall@k 数据结构，与本课 `eval_recall` 返回的结构相同。选出在你的语料上获胜的策略，并将它向后传递。

## 练习

1. 增加第六种策略：使用 `tiktoken` 按 token 数而非字符数的 token 窗口分块。在同一测试语料上与固定窗口对比。
2. 向散文测试语料中注入 30% 比例的代码块。重新生成表格。解释为什么除结构化 markdown 之外的所有策略召回率都下降。
3. 把确定性模拟嵌入替换成你项目实际供应商的嵌入。测量语义聚类的召回率变化。报告各策略之间的差距是扩大了还是缩小了。
4. 给每个分块增加一个 `summary` 字段：一句话的中心描述。把摘要附加到分块正文后重新跑评测。测量召回率的提升。

## 关键术语

| 术语 | 大家怎么说 | 实际含义 |
|------|-----------------|------------------------|
| Recall@k | "我们拿到正确的分块了吗？" | top-k 分块中任意一个与黄金答案区间重叠的查询所占比例 |
| 分块重叠 | "滑动窗口" | 把上一个分块的最后 N 个字符重新包含进下一个分块 |
| 结构化切分器 | "标题感知分块" | 在 H1/H2/H3 边界处切分；标题文本属于分块的一部分 |
| 语义分块器 | "主题感知分块" | 对句子做嵌入，按中心相似度聚类，在漂移处切分 |
| 中心漂移 | "主题切换" | 滑动均值向量与下一个句子之间的余弦相似度跌破阈值 |

## 延伸阅读

- [LongRAG: Enhancing Retrieval-Augmented Generation with Long-context LLMs (arXiv 2406.15319)](https://arxiv.org/abs/2406.15319)
- [Anthropic, Contextual Retrieval](https://www.anthropic.com/news/contextual-retrieval)
- [LlamaIndex, Chunking strategies for production RAG](https://docs.llamaindex.ai/en/stable/optimizing/production_rag/)
- Phase 11 lesson 06 —— RAG 基础
- Phase 11 lesson 07 —— 进阶 RAG
- Phase 19 lesson 65 —— 对本课产出的分块进行排序的混合检索
- Phase 19 lesson 68 —— 在生产中为策略选择打分的评测框架
