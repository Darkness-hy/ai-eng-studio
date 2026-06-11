# 命名实体识别

> 把名字抽出来。听起来简单，直到你遇上模糊的边界、嵌套实体和领域行话。

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 5 · 02 (BoW + TF-IDF), Phase 5 · 03 (Word Embeddings)
**Time:** ~75 minutes

## 问题背景

"Apple sued Google over its iPhone search deal in the US."（苹果就 iPhone 搜索协议在美国起诉谷歌。）这句话里有五个实体：Apple（ORG）、Google（ORG）、iPhone（PRODUCT）、search deal（也许算）、US（GPE）。一个好的 NER 系统能把它们全部抽出来并标对类型。差的系统会漏掉 iPhone，把水果 Apple 和公司 Apple 搞混，还会把 "US" 标成 PERSON。

NER 是每一条结构化抽取流水线背后的主力。简历解析、合规日志扫描、医疗记录脱敏、搜索查询理解、聊天机器人回复的事实锚定、法律合同抽取。你几乎看不见它，但你时时刻刻依赖它。

这节课沿着经典路线（基于规则、HMM、CRF）走向现代路线（BiLSTM-CRF，再到 Transformer）。每一步都解决了前一步的某个具体局限。这个演进模式本身就是这节课的要点。

## 核心概念

**BIO 标注**（或 BILOU）把实体抽取转化为一个序列标注问题。给每个 token 打上 `B-TYPE`（实体开头）、`I-TYPE`（实体内部）或 `O`（不属于任何实体）的标签。

```
Apple    B-ORG
sued     O
Google   B-ORG
over     O
its      O
iPhone   B-PRODUCT
search   O
deal     O
in       O
the      O
US       B-GPE
.        O
```

多 token 实体首尾相接：`New B-GPE`、`York I-GPE`、`City I-GPE`。理解 BIO 的模型可以抽取任意长度的片段（span）。

架构的演进路线：

- **基于规则。** 正则表达式 + 词典（gazetteer）查找。对已知实体精确率高，对新实体覆盖率为零。
- **HMM。** 隐马尔可夫模型（Hidden Markov Model）。给定标签下 token 的发射概率，加上标签到标签的转移概率。用 Viterbi 解码。在标注数据上训练。
- **CRF。** 条件随机场（Conditional Random Field）。类似 HMM 但是判别式模型，因此可以混合任意特征（词形、大小写、邻近词）。到 2026 年仍是低资源部署场景下经典的生产主力。
- **BiLSTM-CRF。** 用神经网络特征替代手工特征。LSTM 从两个方向读句子，顶部的 CRF 层保证标签序列的一致性。
- **基于 Transformer。** 用 token 分类头微调 BERT。准确率最高，算力开销也最大。

```figure
ner-bio-tagging
```

## 从零实现

### Step 1: BIO tagging helpers

```python
def spans_to_bio(tokens, spans):
    labels = ["O"] * len(tokens)
    for start, end, label in spans:
        labels[start] = f"B-{label}"
        for i in range(start + 1, end):
            labels[i] = f"I-{label}"
    return labels


def bio_to_spans(tokens, labels):
    spans = []
    current = None
    for i, label in enumerate(labels):
        if label.startswith("B-"):
            if current:
                spans.append(current)
            current = (i, i + 1, label[2:])
        elif label.startswith("I-") and current and current[2] == label[2:]:
            current = (current[0], i + 1, current[2])
        else:
            if current:
                spans.append(current)
                current = None
    if current:
        spans.append(current)
    return spans
```

```python
>>> tokens = ["Apple", "sued", "Google", "over", "iPhone", "sales", "."]
>>> labels = ["B-ORG", "O", "B-ORG", "O", "B-PRODUCT", "O", "O"]
>>> bio_to_spans(tokens, labels)
[(0, 1, 'ORG'), (2, 3, 'ORG'), (4, 5, 'PRODUCT')]
```

### Step 2: hand-crafted features

对经典（非神经网络）NER 来说，特征决定一切。常用的有：

```python
def token_features(token, prev_token, next_token):
    return {
        "lower": token.lower(),
        "is_upper": token.isupper(),
        "is_title": token.istitle(),
        "has_digit": any(c.isdigit() for c in token),
        "suffix_3": token[-3:].lower(),
        "shape": word_shape(token),
        "prev_lower": prev_token.lower() if prev_token else "<BOS>",
        "next_lower": next_token.lower() if next_token else "<EOS>",
    }


def word_shape(word):
    out = []
    for c in word:
        if c.isupper():
            out.append("X")
        elif c.islower():
            out.append("x")
        elif c.isdigit():
            out.append("d")
        else:
            out.append(c)
    return "".join(out)
```

`word_shape("iPhone")` 返回 `xXxxxx`，`word_shape("USA-2024")` 返回 `XXX-dddd`。大小写模式对识别专有名词是非常强的信号。

### Step 3: a simple rule-based + dictionary baseline

```python
ORG_GAZETTEER = {"Apple", "Google", "Microsoft", "OpenAI", "Meta", "Amazon", "Netflix"}
GPE_GAZETTEER = {"US", "USA", "UK", "India", "Germany", "France"}
PRODUCT_GAZETTEER = {"iPhone", "Android", "Windows", "ChatGPT", "Claude"}


def rule_based_ner(tokens):
    labels = []
    for token in tokens:
        if token in ORG_GAZETTEER:
            labels.append("B-ORG")
        elif token in GPE_GAZETTEER:
            labels.append("B-GPE")
        elif token in PRODUCT_GAZETTEER:
            labels.append("B-PRODUCT")
        else:
            labels.append("O")
    return labels
```

生产级词典通常有数百万条目，从 Wikipedia 和 DBpedia 抓取而来。覆盖率不错，但消歧能力（公司 `Apple` 还是水果 apple）非常糟糕。这正是统计模型胜出的原因。

### Step 4: the CRF step (sketch, not full impl)

不先打好概率论基础，用 50 行从零写一个完整 CRF 并没有什么启发性。直接用 `sklearn-crfsuite`：

```python
import sklearn_crfsuite

def to_features(tokens):
    out = []
    for i, tok in enumerate(tokens):
        prev = tokens[i - 1] if i > 0 else ""
        nxt = tokens[i + 1] if i + 1 < len(tokens) else ""
        out.append({
            "word.lower()": tok.lower(),
            "word.isupper()": tok.isupper(),
            "word.istitle()": tok.istitle(),
            "word.isdigit()": tok.isdigit(),
            "word.suffix3": tok[-3:].lower(),
            "word.shape": word_shape(tok),
            "prev.word.lower()": prev.lower(),
            "next.word.lower()": nxt.lower(),
            "BOS": i == 0,
            "EOS": i == len(tokens) - 1,
        })
    return out


crf = sklearn_crfsuite.CRF(algorithm="lbfgs", c1=0.1, c2=0.1, max_iterations=100, all_possible_transitions=True)
X_train = [to_features(s) for s in sentences_tokenized]
crf.fit(X_train, bio_labels_train)
```

`c1` 和 `c2` 分别是 L1 和 L2 正则化系数。`all_possible_transitions=True` 让模型学到非法序列（比如 `O` 之后紧跟 `I-ORG`）出现的概率很低——CRF 就是这样在不需要你手写约束的情况下保证 BIO 一致性的。

### Step 5: what a BiLSTM-CRF adds

特征变成了学出来的。输入是 token 嵌入（GloVe 或 fastText）。LSTM 从左到右、从右到左各读一遍，拼接后的隐藏状态送入 CRF 输出层。CRF 仍然负责保证标签序列的一致性；LSTM 则用学到的特征替代了手工特征。

```python
import torch
import torch.nn as nn


class BiLSTM_CRF_Head(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, n_labels):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, bidirectional=True, batch_first=True)
        self.fc = nn.Linear(hidden_dim * 2, n_labels)

    def forward(self, token_ids):
        e = self.embed(token_ids)
        h, _ = self.lstm(e)
        emissions = self.fc(h)
        return emissions
```

CRF 层用 `torchcrf.CRF`（pip install pytorch-crf）。相对手工特征 CRF 的提升是可测量的，但除非你有几万句标注数据，否则提升会比你预期的小。

## 生产实践

spaCy 开箱即带生产级 NER。

```python
import spacy

nlp = spacy.load("en_core_web_sm")
doc = nlp("Apple sued Google over its iPhone search deal in the US.")
for ent in doc.ents:
    print(f"{ent.text:20s} {ent.label_}")
```

```
Apple                ORG
Google               ORG
iPhone               ORG
US                   GPE
```

注意 `iPhone` 被标成了 `ORG` 而不是 `PRODUCT`——spaCy 的小模型对产品类实体的覆盖较弱。大模型（`en_core_web_lg`）表现更好，Transformer 模型（`en_core_web_trf`）更上一层。

用 Hugging Face 做基于 BERT 的 NER：

```python
from transformers import pipeline

ner = pipeline("ner", model="dslim/bert-base-NER", aggregation_strategy="simple")
print(ner("Apple sued Google over its iPhone in the US."))
```

```
[{'entity_group': 'ORG', 'word': 'Apple', ...},
 {'entity_group': 'ORG', 'word': 'Google', ...},
 {'entity_group': 'MISC', 'word': 'iPhone', ...},
 {'entity_group': 'LOC', 'word': 'US', ...}]
```

`aggregation_strategy="simple"` 会把连续的 B-X、I-X token 合并为一个片段。不加它你只会得到 token 级标签，得自己动手合并。

### 基于 LLM 的 NER（2026 年的选项）

零样本（zero-shot）和少样本（few-shot）的 LLM NER 在许多领域已能与微调模型相抗衡，在标注数据稀缺时则有压倒性优势。

- **零样本提示。** 给 LLM 一份实体类型列表和一个示例 schema，要求输出 JSON。开箱即用；在新领域上准确率中等。
- **ZeroTuneBio 风格提示。** 把任务拆解为候选抽取 → 含义解释 → 判定 → 复核。多阶段提示（而非一次性提示）能在生物医学 NER 上显著提升准确率。同样的模式也适用于法律、金融和科学领域。
- **结合 RAG 的动态提示。** 每次推理时，从一个小规模标注种子集中检索最相似的标注样例，动态构建少样本提示。在 2026 年的基准测试中，这使 GPT-4 在生物医学 NER 上的 F1 比静态提示高出 11-12%。
- **按实体类型分解。** 对长文档，一次调用同时抽取所有实体类型时，召回率会随长度增长而下降。改为每种实体类型跑一遍抽取。推理成本更高，准确率显著更高。这是临床记录和法律合同处理的标准模式。

截至 2026 年的生产建议：在收集训练数据之前，先用 LLM 零样本跑一个基线。F1 往往已经够用，根本不需要微调。

### 经典 NER 仍然胜出的场景

即便有 LLM 可用，经典 NER 在以下情况下仍然占优：

- 延迟预算在 50ms 以内。
- 你有数千条标注样本，且需要 98% 以上的 F1。
- 领域有稳定的本体（ontology），预训练的 CRF 或 BiLSTM 迁移效果良好。
- 监管约束要求本地部署的非生成式模型。

### 失效的场景

- **领域偏移。** 在 CoNLL 上训练的 NER 用到法律合同上，表现还不如词典。要在你的领域上微调。
- **嵌套实体。** "Bank of America Tower" 同时是 ORG 和 FACILITY。标准 BIO 无法表示重叠片段。你需要嵌套 NER（多遍抽取或基于 span 的模型）。
- **长实体。** "United States Federal Deposit Insurance Corporation."（美国联邦存款保险公司）。token 级模型有时会把它切断。用 `aggregation_strategy` 或后处理来解决。
- **稀疏类型。** 医疗 NER 标签如 DRUG_BRAND、ADVERSE_EVENT、DOSE。通用模型对它们一无所知。这种场景下的起点是 Scispacy 和 BioBERT。

## 交付产物

保存为 `outputs/skill-ner-picker.md`：

```markdown
---
name: ner-picker
description: Pick the right NER approach for a given extraction task.
version: 1.0.0
phase: 5
lesson: 06
tags: [nlp, ner, extraction]
---

Given a task description (domain, label set, language, latency, data volume), output:

1. Approach. Rule-based + gazetteer, CRF, BiLSTM-CRF, or transformer fine-tune.
2. Starting model. Name it (spaCy model ID, Hugging Face checkpoint ID, or "custom, trained from scratch").
3. Labeling strategy. BIO, BILOU, or span-based. Justify in one sentence.
4. Evaluation. Use `seqeval`. Always report entity-level F1 (not token-level).

Refuse to recommend fine-tuning a transformer for under 500 labeled examples unless the user already has a pretrained domain model. Flag nested entities as needing span-based or multi-pass models. Require a gazetteer audit if the user mentions "production scale" and labels are unchanged from CoNLL-2003.
```

## 练习

1. **简单。** 实现 `bio_to_spans`（`spans_to_bio` 的逆操作），并在 10 个句子上验证往返转换的一致性。
2. **中等。** 在 CoNLL-2003 英文 NER 数据集上训练上面的 sklearn-crfsuite CRF。用 `seqeval` 报告各实体类型的 F1。典型结果：约 84 F1。
3. **困难。** 在一个特定领域的 NER 数据集（医疗、法律或金融）上微调 `distilbert-base-cased`，与 spaCy 小模型对比。记录数据泄漏检查过程，并写下让你意外的发现。

## 关键术语

| 术语 | 大家怎么说 | 实际含义 |
|------|-----------------|-----------------------|
| NER | 抽取名字 | 给 token 片段打上类型标签（PERSON、ORG、GPE、DATE 等）。 |
| BIO | 标注方案 | `B-X` 开头，`I-X` 延续，`O` 在实体外。 |
| BILOU | 更好的 BIO | 增加 `L-X`（末尾）和 `U-X`（单 token 实体），边界更清晰。 |
| CRF | 结构化分类器 | 不仅建模发射概率，还建模标签间的转移。保证序列合法。 |
| 嵌套 NER | 重叠实体 | 一个片段与其子片段分属不同实体。BIO 无法表达这种情况。 |
| 实体级 F1 | 正确的 NER 指标 | 预测片段必须与真实片段完全匹配。token 级 F1 会高估准确率。 |

## 延伸阅读

- [Lample et al. (2016). Neural Architectures for Named Entity Recognition](https://arxiv.org/abs/1603.01360) — BiLSTM-CRF 的原始论文，经典必读。
- [Devlin et al. (2018). BERT: Pre-training of Deep Bidirectional Transformers](https://arxiv.org/abs/1810.04805) — 提出了后来成为标准做法的 token 分类范式。
- [spaCy linguistic features — named entities](https://spacy.io/usage/linguistic-features#named-entities) — `Doc.ents` 和 `Span` 上所有属性的实用参考。
- [seqeval](https://github.com/chakki-works/seqeval) — 正确的指标库，任何时候都用它。
