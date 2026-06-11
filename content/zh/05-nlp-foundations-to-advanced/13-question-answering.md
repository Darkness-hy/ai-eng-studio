# 问答系统

> 三类系统塑造了现代问答（QA）。抽取式负责定位答案片段，检索增强式让答案有文档依据，生成式直接产出答案。如今的每一个 AI 助手都是这三者的混合体。

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 5 · 11 (Machine Translation), Phase 5 · 10 (Attention Mechanism)
**Time:** ~75 minutes

## 问题背景

用户输入"第一代 iPhone 是什么时候发布的？"，期待得到的是"2007 年 6 月 29 日"。不是"Apple 的历史悠久而丰富"，也不是孤零零一个没有上下文的"2007"。用户要的是一个直接、有依据、正确的答案。

过去十年，三种架构主导了问答领域。

- **抽取式问答（Extractive QA）。** 给定一个问题和一段已知包含答案的文本，找出答案片段在文本中的起止索引。SQuAD 是这一任务的经典基准。
- **开放域问答（Open-domain QA）。** 不提供文本段落。先检索出相关段落，再抽取或生成答案。这是当今所有 RAG 流水线的基石。
- **生成式 / 闭卷问答（Generative / Closed-book QA）。** 大语言模型直接从参数化记忆中作答。没有检索环节。推理速度最快，但在事实准确性上最不可靠。

2026 年的趋势是混合式：先检索出最相关的几个段落，再提示生成式模型基于这些段落作答。这就是 RAG——第 14 课会深入讲检索那一半，本课负责构建问答这一半。

## 核心概念

![QA architectures: extractive, retrieval-augmented, generative](../assets/qa.svg)

**抽取式。** 用 Transformer（BERT 系列）把问题和段落编码在一起，训练两个预测头，分别预测答案的起始和结束 token 索引。损失是对合法位置的交叉熵。输出是段落中的一个片段。它从构造上就不会产生幻觉，但也从构造上无法处理段落中没有答案的问题。

**检索增强式（RAG）。** 分两个阶段。第一阶段，检索器（retriever）从语料库中找出 top-`k` 个段落；第二阶段，阅读器（reader，可以是抽取式或生成式）基于这些段落产出答案。检索器与阅读器分离，使二者可以独立训练和评估。现代 RAG 通常还会在二者之间加一个重排器（reranker）。

**生成式。** 一个 decoder-only 的大语言模型（GPT、Claude、Llama）从学到的权重中直接作答。没有检索步骤。在常识性知识上表现优异，在罕见或近期事实上会灾难性翻车。幻觉率与该事实在预训练数据中的出现频率成反比。

## 从零实现

### 第 1 步：用预训练模型做抽取式问答

```python
from transformers import pipeline

qa = pipeline("question-answering", model="deepset/roberta-base-squad2")

passage = (
    "Apple Inc. released the first iPhone on June 29, 2007. "
    "The device was announced by Steve Jobs at Macworld in January 2007."
)
question = "When was the first iPhone released?"

answer = qa(question=question, context=passage)
print(answer)
```

```python
{'score': 0.98, 'start': 57, 'end': 70, 'answer': 'June 29, 2007'}
```

`deepset/roberta-base-squad2` 在 SQuAD 2.0 上训练，该数据集包含无法回答的问题。默认情况下，`question-answering` pipeline 即使在模型的空答案（null）得分胜出时也会返回得分最高的片段——它*不会*自动返回空答案。要获得显式的"无答案"行为，需要在调用 pipeline 时传入 `handle_impossible_answer=True`：此时只有当空答案得分超过所有片段得分时，pipeline 才返回空答案。无论哪种方式，都要检查 `score` 字段。

### 第 2 步：检索增强流水线（示意版）

```python
from sentence_transformers import SentenceTransformer
import numpy as np

encoder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

corpus = [
    "Apple Inc. released the first iPhone on June 29, 2007.",
    "Macworld 2007 featured the iPhone announcement by Steve Jobs.",
    "Android launched in 2008 as Google's mobile operating system.",
    "The first iPod was released in 2001.",
]
corpus_embeddings = encoder.encode(corpus, normalize_embeddings=True)


def retrieve(question, top_k=2):
    q_emb = encoder.encode([question], normalize_embeddings=True)
    sims = (corpus_embeddings @ q_emb.T).squeeze()
    order = np.argsort(-sims)[:top_k]
    return [corpus[i] for i in order]


def answer(question):
    passages = retrieve(question, top_k=2)
    combined = " ".join(passages)
    return qa(question=question, context=combined)


print(answer("When was the first iPhone released?"))
```

两阶段流水线。稠密检索器（Sentence-BERT）通过语义相似度找到相关段落，抽取式阅读器（RoBERTa-SQuAD）从拼接后的最优段落中抽出答案片段。这套方案适用于小语料。对于百万级文档的语料库，应使用 FAISS 或向量数据库。

### 第 3 步：用 RAG 做生成式问答

```python
def rag_generate(question, llm):
    passages = retrieve(question, top_k=3)
    prompt = f"""Context:
{chr(10).join('- ' + p for p in passages)}

Question: {question}

Answer using only the context above. If the context does not contain the answer, say "I don't know."
"""
    return llm(prompt)
```

提示词的写法很关键。明确要求模型只依据上下文作答、并在上下文不足时回答 "I don't know"，相比朴素提示能把幻觉率降低 40-60%。更精细的模式还会加入引用标注、置信度分数和结构化抽取。

### 第 4 步：贴近真实世界的评估

SQuAD 使用**精确匹配（Exact Match, EM）**和 **token 级 F1**。EM 是归一化（转小写、去标点、去冠词）之后的严格匹配——预测要么完全匹配，要么得 0 分。F1 基于预测与参考答案之间的 token 重叠计算，会给部分匹配打分。这两个指标都低估了同义改写："June 29, 2007" 与 "June 29th, 2007" 通常 EM 得 0（序数词后缀破坏了归一化），但凭借重叠的 token 仍能拿到可观的 F1。

对于生产环境的问答系统：

- **答案准确率**（由 LLM 评判或人工评判，因为自动指标无法捕捉语义等价性）。
- **引用准确率。** 被引用的段落是否真正支撑了答案？通过生成的引用与检索段落之间的字符串匹配即可轻松自动检查。
- **拒答校准（Refusal calibration）。** 当答案不在检索到的段落中时，系统能否正确地说 "I don't know"？衡量虚假自信率。
- **检索召回率。** 在评估阅读器之前，先测量检索器是否把正确段落送进了 top-`k`。段落缺失，阅读器无力回天。

### RAGAS：2026 年的生产级评估框架

`RAGAS` 是专为 RAG 系统打造的评估框架，也是 2026 年的默认上线选择。它在不需要标准参考答案的情况下对四个维度打分：

- **忠实度（Faithfulness）。** 答案中的每条论断是否都来自检索到的上下文？通过基于 NLI 的蕴含关系来衡量。这是你的首要幻觉指标。
- **答案相关性（Answer relevance）。** 答案是否回应了问题？方法是从答案反向生成假设性问题，再与真实问题比较。
- **上下文精确率（Context precision）。** 在检索到的片段中，真正相关的占多少比例？精确率低意味着提示词里混入了噪声。
- **上下文召回率（Context recall）。** 检索结果是否覆盖了所有必要信息？召回率低意味着阅读器不可能成功。

无参考答案的打分方式让你可以直接在线上生产流量上做评估，无需精心标注的标准答案。对于精确匹配指标完全失效的开放式问题，再叠加一层 LLM-as-judge。

`pip install ragas`。接入你的检索器和阅读器，每个查询得到四个标量分数，对回归报警。

## 生产实践

2026 年的技术栈。

| 使用场景 | 推荐方案 |
|---------|-------------|
| 给定段落，找答案片段 | `deepset/roberta-base-squad2` |
| 在固定语料上作答，闭卷不可接受 | RAG：稠密检索器 + LLM 阅读器 |
| 在文档库上实时问答 | RAG，混合检索器（BM25 + 稠密）+ 重排器（第 14 课） |
| 对话式问答（带追问） | LLM 携带对话历史 + 每轮做 RAG |
| 高度事实性、受监管的领域 | 在权威语料上做抽取式；绝不单独使用生成式 |

抽取式问答在 2026 年已不时髦，因为 RAG 加 LLM 能覆盖更多场景。但在要求逐字引用的场合它仍在上线使用：法律检索、合规审查、审计工具。

## 交付产物

保存为 `outputs/skill-qa-architect.md`：

```markdown
---
name: qa-architect
description: Choose QA architecture, retrieval strategy, and evaluation plan.
version: 1.0.0
phase: 5
lesson: 13
tags: [nlp, qa, rag]
---

Given requirements (corpus size, question type, factuality constraint, latency budget), output:

1. Architecture. Extractive, RAG with extractive reader, RAG with generative reader, or closed-book LLM. One-sentence reason.
2. Retriever. None, BM25, dense (name the encoder), or hybrid.
3. Reader. SQuAD-tuned model, LLM by name, or "domain-fine-tuned DistilBERT."
4. Evaluation. EM + F1 for extractive benchmarks; answer accuracy + citation accuracy + refusal calibration for production. Name what you are measuring and how you are measuring it.

Refuse closed-book LLM answers for regulatory or compliance-sensitive questions. Refuse any QA system without a retrieval-recall baseline (you cannot evaluate the reader without knowing the retriever surfaced the right passage). Flag questions that require multi-hop reasoning as needing specialized multi-hop retrievers like HotpotQA-trained systems.
```

## 练习

1. **简单。** 在 10 段 Wikipedia 文本上搭建上面的 SQuAD 抽取式流水线。手写 10 个问题，统计答案正确的频率。如果段落和问题都干净，正确数应在 7-9 个之间。
2. **中等。** 加一个拒答分类器。当检索最高分低于某个阈值（比如余弦相似度 0.3）时，直接返回 "I don't know"，不再调用阅读器。在留出集上调阈值。
3. **困难。** 在你自选的 10,000 篇文档语料上构建 RAG 流水线。实现混合检索（BM25 + 稠密）并用 RRF 融合（见第 14 课）。分别测量有无混合检索时的答案准确率，记录哪些类型的问题受益最大。

## 关键术语

| 术语 | 通俗说法 | 实际含义 |
|------|-----------------|-----------------------|
| 抽取式问答 | 找出答案片段 | 在给定段落内预测答案的起始和结束索引。 |
| 开放域问答 | 在语料库上做问答 | 不提供段落；必须先检索再作答。 |
| RAG | 先检索再生成 | 检索增强生成。检索器 + 阅读器流水线。 |
| SQuAD | 经典基准 | Stanford Question Answering Dataset。使用 EM + F1 指标。 |
| 幻觉 | 编造的答案 | 阅读器输出的内容没有检索上下文的支撑。 |
| 拒答校准 | 知道何时闭嘴 | 系统在无法作答时正确地说 "I don't know"。 |

## 延伸阅读

- [Rajpurkar et al. (2016). SQuAD: 100,000+ Questions for Machine Comprehension of Text](https://arxiv.org/abs/1606.05250) — 基准数据集论文。
- [Karpukhin et al. (2020). Dense Passage Retrieval for Open-Domain QA](https://arxiv.org/abs/2004.04906) — DPR，问答领域经典的稠密检索器。
- [Lewis et al. (2020). Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks](https://arxiv.org/abs/2005.11401) — 为 RAG 命名的论文。
- [Gao et al. (2023). Retrieval-Augmented Generation for Large Language Models: A Survey](https://arxiv.org/abs/2312.10997) — 全面的 RAG 综述。
