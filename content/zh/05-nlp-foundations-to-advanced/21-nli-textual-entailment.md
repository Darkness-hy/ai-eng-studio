# 自然语言推理 —— 文本蕴含

> "t 蕴含 h" 的意思是：人类读完 t 之后会断定 h 为真。NLI 的任务就是预测蕴含 / 矛盾 / 中立。表面上看很无聊，但在生产环境中是承重墙。

**Type:** Learn
**Languages:** Python
**Prerequisites:** Phase 5 · 05 (Sentiment Analysis), Phase 5 · 13 (Question Answering)
**Time:** ~60 minutes

## 问题背景

你做了一个摘要器，它生成了一段摘要。你怎么知道这段摘要里没有幻觉？

你做了一个聊天机器人，它回答了"是"。你怎么知道这个答案确实有检索到的段落作支撑？

你需要把 10,000 篇新闻按主题分类，却没有任何训练标签。能不能复用一个现成模型？

这三个问题都可以归约为自然语言推理（Natural Language Inference, NLI）。NLI 问的是：给定前提（premise）`t` 和假设（hypothesis）`h`，`h` 是被 `t` 蕴含、被它矛盾，还是中立（无关）？

- **幻觉检测：** `t` = 源文档，`h` = 摘要中的断言。非蕴含 = 幻觉。
- **有依据的问答（Grounded QA）：** `t` = 检索到的段落，`h` = 生成的答案。非蕴含 = 凭空捏造。
- **零样本分类：** `t` = 文档，`h` = 文字化的标签（"This is about sports"）。蕴含 = 预测出的标签。

一个任务，三种生产用途。这就是为什么每个 RAG 评估框架的底层都内置了一个 NLI 模型。

## 核心概念

![NLI：三分类，前提 vs 假设](../assets/nli.svg)

**三种标签。**

- **蕴含（Entailment）。** `t` → `h`。"The cat is on the mat" 蕴含 "There is a cat"。
- **矛盾（Contradiction）。** `t` → ¬`h`。"The cat is on the mat" 与 "There is no cat" 矛盾。
- **中立（Neutral）。** 双向都推不出来。"The cat is on the mat" 对 "The cat is hungry" 是中立的。

**不是逻辑蕴含。** NLI 是*自然*语言推理 —— 看的是普通读者会得出什么推断，而不是严格的逻辑。在 NLI 中，"John walked his dog" 蕴含 "John has a dog"；但在严格的一阶逻辑里，只有先把"拥有"公理化才能承认这一步。

**数据集。**

- **SNLI**（2015）。57 万条人工标注的句对，前提来自图像描述。领域偏窄。
- **MultiNLI**（2017）。覆盖 10 种体裁的 43.3 万条句对。是 2026 年的标准训练语料。
- **ANLI**（2019）。对抗式 NLI（Adversarial NLI）。标注者专门编写能击溃现有模型的样本。难度更高。
- **DocNLI、ConTRoL**（2020–21）。前提是文档级长度。考验多跳与长程推理。

**架构。** 一个 Transformer 编码器（BERT、RoBERTa、DeBERTa）读入 `[CLS] premise [SEP] hypothesis [SEP]`。`[CLS]` 的表示接一个三分类 softmax。在 MNLI 上训练，在留出基准上评估，分布内句对的准确率可以达到 90% 以上。

**借助 NLI 做零样本分类。** 给定一篇文档和若干候选标签，把每个标签改写成一条假设（"This text is about sports"），分别计算蕴含概率，取最大者。这就是 Hugging Face `zero-shot-classification` pipeline 背后的机制。

## 从零实现

### 第 1 步：运行一个预训练 NLI 模型

```python
from transformers import pipeline

nli = pipeline("text-classification",
               model="facebook/bart-large-mnli",
               top_k=None)  # return all labels; replaces deprecated return_all_scores=True

premise = "The cat is sleeping on the couch."
hypothesis = "There is a cat in the room."

result = nli({"text": premise, "text_pair": hypothesis})[0]
print(result)
# [{'label': 'entailment', 'score': 0.97},
#  {'label': 'neutral', 'score': 0.02},
#  {'label': 'contradiction', 'score': 0.01}]
```

在生产环境做 NLI，开源默认选项是 `facebook/bart-large-mnli` 和 `microsoft/deberta-v3-large-mnli`。DeBERTa-v3 雄踞各大排行榜。

### 第 2 步：零样本分类

```python
zs = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

text = "The stock market rallied after the central bank cut interest rates."
labels = ["finance", "sports", "politics", "technology"]

result = zs(text, candidate_labels=labels)
print(result)
# {'labels': ['finance', 'politics', 'technology', 'sports'],
#  'scores': [0.92, 0.05, 0.02, 0.01]}
```

默认模板是 "This example is about {label}."，可以通过 `hypothesis_template` 自定义。不需要训练数据，不需要微调，开箱即用。

### 第 3 步：为 RAG 做忠实度检查

```python
def is_faithful(answer, context, threshold=0.5):
    result = nli({"text": context, "text_pair": answer})[0]
    entail = next(s for s in result if s["label"] == "entailment")
    return entail["score"] > threshold
```

这就是 RAGAS 忠实度指标的核心。把生成的答案拆成原子断言，逐条对照检索到的上下文做检查，再报告被蕴含的断言占比。

### 第 4 步：手写一个 NLI 分类器（概念演示）

参见 `code/main.py`，那是一个只用标准库的玩具实现：通过词汇重叠 + 否定词检测来比较前提和假设。它不可能与 Transformer 模型竞争 —— 但它展示了这个任务的基本形态：输入两段文本，输出三分类标签，损失 = 基于 `{entail, contradict, neutral}` 的交叉熵。

## 常见陷阱

- **仅靠假设的捷径。** 模型只看假设、不看前提，在 SNLI 上就能达到约 60% 的预测准确率，因为 "not"、"nobody"、"never" 与矛盾标签高度相关。这是检测标签泄漏的有力基线。
- **词汇重叠启发式。** 子序列启发式（"每个子序列都被蕴含"）在 SNLI 上能过关，但在 HANS/ANLI 上会失效。要用对抗性基准。
- **文档级性能退化。** 单句级 NLI 模型在文档级前提上 F1 会掉 20 分以上。处理长上下文要用在 DocNLI 上训练的模型。
- **零样本模板敏感性。** "This example is about {label}"、"{label}"、"The topic is {label}" 之间的差异能让准确率波动 10 分以上。务必调模板。
- **领域不匹配。** MNLI 的训练语料是通用英语。法律、医学和科学文本需要领域专用的 NLI 模型（例如 SciNLI、MedNLI）。

## 生产实践

2026 年的技术栈：

| 使用场景 | 模型 |
|---------|-------|
| 通用 NLI | `microsoft/deberta-v3-large-mnli` |
| 高速 / 边缘端 | `cross-encoder/nli-deberta-v3-base` |
| 零样本分类（轻量级） | `facebook/bart-large-mnli` |
| 文档级 NLI | `MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli` |
| 多语言 | `MoritzLaurer/multilingual-MiniLMv2-L6-mnli-xnli` |
| RAG 中的幻觉检测 | RAGAS / DeepEval 内部的 NLI 层 |

2026 年的元模式：NLI 是文本理解领域的万能胶带。只要你需要判断"A 是否支持 B？"或"A 是否与 B 矛盾？"—— 先伸手去拿 NLI，再考虑多打一次 LLM 调用。

## 交付产物

保存为 `outputs/skill-nli-picker.md`：

```markdown
---
name: nli-picker
description: Pick an NLI model, label template, and evaluation setup for a classification / faithfulness / zero-shot task.
version: 1.0.0
phase: 5
lesson: 21
tags: [nlp, nli, zero-shot]
---

Given a use case (faithfulness check, zero-shot classification, document-level inference), output:

1. Model. Named NLI checkpoint. Reason tied to domain, length, language.
2. Template (if zero-shot). Verbalization pattern. Example.
3. Threshold. Entailment cutoff for the decision rule. Reason based on calibration.
4. Evaluation. Accuracy on held-out labeled set, hypothesis-only baseline, adversarial subset.

Refuse to ship zero-shot classification without a 100-example labeled sanity check. Refuse to use a sentence-level NLI model on document-length premises. Flag any claim that NLI solves hallucination — it reduces it; it does not eliminate it.
```

## 练习

1. **简单。** 在 20 条手工构造、覆盖全部三个类别的（前提、假设、标签）三元组上运行 `facebook/bart-large-mnli`，测量准确率。再加入针对"子序列启发式"的对抗陷阱（"I did not eat the cake" vs "I ate the cake"），看看模型会不会被击溃。
2. **中等。** 在 100 条 AG News 标题上比较零样本模板 `"This text is about {label}"`、`"The topic is {label}"` 和 `"{label}"`，报告准确率的波动幅度。
3. **困难。** 构建一个 RAG 忠实度检查器：原子断言分解 + 逐条断言做 NLI。在 50 条带标准上下文的 RAG 生成答案上评估，对照人工标注测量假阳性率和假阴性率。

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|-----------------|-----------------------|
| NLI | 自然语言推理 | 对前提-假设关系做三分类。 |
| RTE | 文本蕴含识别（Recognizing Textual Entailment） | NLI 的旧称；同一个任务。 |
| 蕴含 | "t 推出 h" | 普通读者读了 t 之后会断定 h 为真。 |
| 矛盾 | "t 排除了 h" | 普通读者读了 t 之后会断定 h 为假。 |
| 中立 | "无法判定" | 从 t 到 h 双向都推不出结论。 |
| 零样本分类 | 把 NLI 当分类器 | 把标签文字化为假设，取蕴含分最高者。 |
| 忠实度 | 答案有支撑吗？ | 在（检索上下文，生成答案）上做 NLI。 |

## 延伸阅读

- [Bowman et al. (2015). A large annotated corpus for learning natural language inference](https://arxiv.org/abs/1508.05326) —— SNLI。
- [Williams, Nangia, Bowman (2017). A Broad-Coverage Challenge Corpus for Sentence Understanding through Inference](https://arxiv.org/abs/1704.05426) —— MultiNLI。
- [Nie et al. (2019). Adversarial NLI](https://arxiv.org/abs/1910.14599) —— ANLI 基准。
- [Yin, Hay, Roth (2019). Benchmarking Zero-shot Text Classification](https://arxiv.org/abs/1909.00161) —— 把 NLI 当分类器。
- [He et al. (2021). DeBERTa: Decoding-enhanced BERT with Disentangled Attention](https://arxiv.org/abs/2006.03654) —— 2026 年 NLI 的主力模型。
