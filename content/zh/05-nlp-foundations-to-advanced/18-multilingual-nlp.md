# 多语言 NLP

> 一个模型，支持 100 多种语言，其中大多数语言连训练数据都没有。跨语言迁移（cross-lingual transfer）是 2020 年代的一个实用奇迹。

**Type:** Learn
**Languages:** Python
**Prerequisites:** Phase 5 · 04 (GloVe, FastText, Subword), Phase 5 · 11 (Machine Translation)
**Time:** ~45 minutes

## 问题背景

英语有数十亿条带标注的样本，乌尔都语只有几千条，迈蒂利语几乎一条都没有。任何面向全球用户的实用 NLP 系统，都必须在那些缺乏任务专用训练数据的长尾语言上正常工作。

多语言模型的解法是：用多种语言同时训练同一个模型。共享表示让模型把在高资源语言上学到的能力迁移到低资源语言上。在英语情感分析数据上微调这个模型，它对乌尔都语的情感预测开箱即用、效果出人意料地好。这就是零样本跨语言迁移（zero-shot cross-lingual transfer），它彻底改变了 NLP 走向全世界的方式。

这节课会讲清楚其中的权衡、几个经典模型，以及一个最容易让多语言新手团队栽跟头的决策：为迁移挑选源语言。

## 核心概念

![Cross-lingual transfer via shared multilingual embedding space](../assets/multilingual.svg)

**共享词表。** 多语言模型使用在所有目标语言文本上训练的 SentencePiece 或 WordPiece 分词器。词表是共享的：同一个子词单元在相关语言中表示同一个语素。英语和意大利语中的 `anti-` 会得到同一个 token。

**共享表示。** 一个用掩码语言建模在多种语言上预训练的 Transformer 会学到：不同语言中语义相近的句子产生相近的隐藏状态。mBERT、XLM-R 和 NLLB 都表现出这种性质。英语 "cat" 的嵌入会聚在法语 "chat" 和西班牙语 "gato" 附近，整句的嵌入也是如此。

**零样本迁移。** 用某一种语言（通常是英语）的标注数据微调模型。推理时，直接在模型支持的任何其他语言上运行。完全不需要目标语言的标注。对于类型学上相近的语言效果很强，对差异较大的语言效果较弱。

**少样本微调。** 加入 100-500 条目标语言的标注样本，分类任务的准确率会跃升到英语基线的 95-98%。这是多语言 NLP 中性价比最高的一个杠杆。

## 主流模型

| 模型 | 年份 | 覆盖范围 | 备注 |
|-------|------|----------|-------|
| mBERT | 2018 | 104 种语言 | 在 Wikipedia 上训练。第一个实用的多语言语言模型。低资源语言上较弱。 |
| XLM-R | 2019 | 100 种语言 | 在 CommonCrawl 上训练（规模远大于 Wikipedia）。确立了跨语言基线。Base 270M，Large 550M。 |
| XLM-V | 2023 | 100 种语言 | XLM-R 的 100 万 token 词表版本（对比原来的 25 万）。在低资源语言上更好。 |
| mT5 | 2020 | 101 种语言 | 面向多语言生成的 T5 架构。 |
| NLLB-200 | 2022 | 200 种语言 | Meta 的翻译模型；包含 55 种低资源语言。 |
| BLOOM | 2022 | 46 种语言 + 13 种编程语言 | 开放的 176B 多语言训练 LLM。 |
| Aya-23 | 2024 | 23 种语言 | Cohere 的多语言 LLM。在阿拉伯语、印地语、斯瓦希里语上表现强。 |

按用途选型。分类任务用 XLM-R-base 作为稳妥的默认选择就很好。生成任务根据是翻译还是开放生成，选 mT5 或 NLLB。LLM 风格的工作则搭配 Aya-23 或 Claude，配合显式的多语言提示。

## 源语言决策（2026 年研究）

大多数团队默认用英语作为微调的源语言。最近的研究（2026 年）表明这往往是错的。

语言相似度比原始语料规模更能预测迁移质量。对于斯拉夫语目标语言，德语或俄语常常胜过英语；对于印度语系目标语言，印地语常常胜过英语。**qWALS** 相似度指标（2026 年，基于 World Atlas of Language Structures 特征）量化了这一点。**LANGRANK**（Lin et al., ACL 2019）是另一个更早的方法，它综合语言学相似性、语料规模和谱系亲缘关系来对候选源语言排序。

实用法则：如果你的目标语言有一个类型学上相近的高资源亲属语言，先试着在它上面微调，再和英语微调做对比。

## 从零实现

### 第 1 步：零样本跨语言分类

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

tok = AutoTokenizer.from_pretrained("joeddav/xlm-roberta-large-xnli")
model = AutoModelForSequenceClassification.from_pretrained("joeddav/xlm-roberta-large-xnli")


def classify(text, candidate_labels, hypothesis_template="This text is about {}."):
    scores = {}
    for label in candidate_labels:
        hypothesis = hypothesis_template.format(label)
        inputs = tok(text, hypothesis, return_tensors="pt", truncation=True)
        with torch.no_grad():
            logits = model(**inputs).logits[0]
        entail_score = torch.softmax(logits, dim=-1)[2].item()
        scores[label] = entail_score
    return dict(sorted(scores.items(), key=lambda x: -x[1]))


print(classify("I love this product!", ["positive", "negative", "neutral"]))
print(classify("मुझे यह उत्पाद पसंद है!", ["positive", "negative", "neutral"]))
print(classify("J'adore ce produit !", ["positive", "negative", "neutral"]))
```

一个模型、三种语言、同一套 API。在 NLI 数据上训练的 XLM-R 借助蕴含（entailment）技巧很好地迁移到了分类任务。

### 第 2 步：多语言嵌入空间

```python
from sentence_transformers import SentenceTransformer
import numpy as np

model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

pairs = [
    ("The cat is sleeping.", "Le chat dort."),
    ("The cat is sleeping.", "El gato está durmiendo."),
    ("The cat is sleeping.", "Die Katze schläft."),
    ("The cat is sleeping.", "The dog is barking."),
]

for eng, other in pairs:
    emb_eng = model.encode([eng], normalize_embeddings=True)[0]
    emb_other = model.encode([other], normalize_embeddings=True)[0]
    sim = float(np.dot(emb_eng, emb_other))
    print(f"  {eng!r} <-> {other!r}: cos={sim:.3f}")
```

互为翻译的句子在嵌入空间中彼此靠近，而另一句不同含义的英语句子则落在更远处。跨语言检索、聚类和相似度计算之所以可行，靠的就是这一点。

### 第 3 步：少样本微调策略

```python
from transformers import TrainingArguments, Trainer
from datasets import Dataset


def few_shot_finetune(base_model, base_tokenizer, examples):
    ds = Dataset.from_list(examples)

    def tokenize_fn(ex):
        out = base_tokenizer(ex["text"], truncation=True, max_length=128)
        out["labels"] = ex["label"]
        return out

    ds = ds.map(tokenize_fn)
    args = TrainingArguments(
        output_dir="out",
        per_device_train_batch_size=8,
        num_train_epochs=5,
        learning_rate=2e-5,
        save_strategy="no",
    )
    trainer = Trainer(model=base_model, args=args, train_dataset=ds)
    trainer.train()
    return base_model
```

对于 100-500 条目标语言样本，`num_train_epochs=5` 和 `learning_rate=2e-5` 是安全的默认值。学习率过高会导致多语言对齐崩塌，你得到的就只剩一个纯英语模型了。

## 真正有效的评估

- **在各语言的留出集上分别测准确率。** 不要做聚合。聚合指标会掩盖长尾。
- **和单语基线对比。** 对于数据足够多的语言，从头训练的单语模型有时会胜过多语言模型。要实测。
- **实体级测试。** 用目标语言中的命名实体来测。多语言模型对远离拉丁字母的文字系统往往分词能力很弱。
- **跨语言一致性。** 两种语言表达同一含义时应得到相同的预测。把这个差距量出来。

## 生产实践

2026 年的技术栈：

| 任务 | 推荐方案 |
|-----|-------------|
| 分类，100 种语言 | 微调后的 XLM-R-base（约 270M） |
| 零样本文本分类 | `joeddav/xlm-roberta-large-xnli` |
| 多语言句子嵌入 | `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` |
| 翻译，200 种语言 | `facebook/nllb-200-distilled-600M`（见第 11 课） |
| 多语言生成 | Claude、GPT-4、Aya-23、mT5-XXL |
| 低资源语言 NLP | XLM-V，或在相近高资源语言上做领域微调 |

如果性能重要，一定要为目标语言的微调留出预算。零样本只是起点，不是最终答案。

### 分词税（低资源语言会出什么问题）

多语言模型让所有语言共享同一个分词器。这个词表是在被英语、法语、西班牙语、中文、德语主导的语料上训练的。对于主导集合之外的任何语言，三种「税」会悄无声息地叠加：

- **繁殖率税（fertility tax）。** 低资源语言的文本被切成的 token 数远多于英语：每个词对应的子词更多。一句印地语可能需要等价英语句子 3-5 倍的 token。这 3-5 倍会吞掉你的上下文窗口、训练效率和延迟。
- **变体恢复税。** 每一个拼写错误、变音符号变体、Unicode 规范化不一致或大小写变化，在嵌入空间里都变成一段毫无关联、从零起步的序列。母语者一眼就能看出的正字法对应关系，模型却学不到。
- **容量挤占税。** 前两种税消耗了上下文位置、层深和嵌入维度。留给真正推理的容量，系统性地少于同一个模型分配给高资源语言的容量。

实际症状是：模型在印地语上训练一切正常，损失曲线没问题，评估困惑度也合理，但生产环境的输出就是有微妙的错误。词形变化在句中崩坏，罕见的屈折形式永远恢复不出来。**分词器坏了，靠堆数据是救不回来的。**

缓解手段：选一个对目标语言覆盖良好的分词器（XLM-V 的 100 万 token 词表就是直接的解法）；在训练前用留出的目标语言文本核实分词繁殖率；对真正长尾的文字系统使用字节级回退（SentencePiece 的 `byte_fallback=True`，或 GPT-2 风格的字节级 BPE），保证任何字符都不会变成 OOV。

## 交付产物

保存为 `outputs/skill-multilingual-picker.md`：

```markdown
---
name: multilingual-picker
description: Pick source language, target model, and evaluation plan for a multilingual NLP task.
version: 1.0.0
phase: 5
lesson: 18
tags: [nlp, multilingual, cross-lingual]
---

Given requirements (target languages, task type, available labeled data per language), output:

1. Source language for fine-tuning. Default English; check LANGRANK or qWALS if target language has a typologically close high-resource language.
2. Base model. XLM-R (classification), mT5 (generation), NLLB (translation), Aya-23 (generative LLM).
3. Few-shot budget. Start with 100-500 target-language examples if available. Zero-shot only if labeling is infeasible.
4. Evaluation plan. Per-language accuracy (not aggregate), cross-lingual consistency, entity-level F1 on non-Latin scripts.

Refuse to ship a multilingual model without per-language evaluation — aggregate metrics hide long-tail failures. Flag scripts with low tokenization coverage (Amharic, Tigrinya, many African languages) as needing a model with byte-fallback (SentencePiece with byte_fallback=True, or byte-level tokenizer like GPT-2).
```

## 练习

1. **简单。** 在英语、法语、印地语、阿拉伯语上各取 10 个句子，运行零样本分类流水线，分别报告每种语言的准确率。你应该会看到：法语很强，印地语尚可，阿拉伯语波动较大。
2. **中等。** 用 `paraphrase-multilingual-MiniLM-L12-v2` 在一个小型混合语言语料上搭建跨语言检索器。用英语查询，检索任意语言的文档，测量 recall@5。
3. **困难。** 针对一个印地语分类任务，对比以英语为源和以印地语为源的微调。两种方案都用 500 条目标语言样本做少样本微调。报告哪个源语言得到的印地语准确率更高、高多少。这就是 LANGRANK 论点的微缩版实验。

## 关键术语

| 术语 | 人们常说的 | 实际含义 |
|------|-----------------|-----------------------|
| 多语言模型 | 一个模型，多种语言 | 跨语言共享词表和参数。 |
| 跨语言迁移 | 在一种语言上训练，在另一种上运行 | 在源语言上微调，在目标语言上评估，无需目标语言标注。 |
| 零样本 | 没有目标语言标注 | 不在目标语言上微调就完成迁移。 |
| 少样本 | 少量目标语言标注 | 用 100-500 条目标语言样本做微调。 |
| mBERT | 第一个多语言语言模型 | 在 Wikipedia 上预训练的 104 语言 BERT。 |
| XLM-R | 标准的跨语言基线 | 在 CommonCrawl 上预训练的 100 语言 RoBERTa。 |
| NLLB | Meta 的 200 语言机器翻译 | No Language Left Behind。包含 55 种低资源语言。 |

## 延伸阅读

- [Conneau et al. (2019). Unsupervised Cross-lingual Representation Learning at Scale](https://arxiv.org/abs/1911.02116) — XLM-R 论文。
- [Pires, Schlinger, Garrette (2019). How Multilingual is Multilingual BERT?](https://arxiv.org/abs/1906.01502) — 开启跨语言迁移研究路线的分析论文。
- [Costa-jussà et al. (2022). No Language Left Behind](https://arxiv.org/abs/2207.04672) — NLLB-200 论文。
- [Üstün et al. (2024). Aya Model: An Instruction Finetuned Open-Access Multilingual Language Model](https://arxiv.org/abs/2402.07827) — Aya，Cohere 的多语言 LLM。
- [Language Similarity Predicts Cross-Lingual Transfer Learning Performance (2026)](https://www.mdpi.com/2504-4990/8/3/65) — qWALS / LANGRANK 源语言选择论文。
