# 文本处理 —— 分词、词干提取与词形还原

> 语言是连续的，模型是离散的，预处理就是连接两者的桥梁。

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 2 · 14 (Naive Bayes)
**Time:** ~45 minutes

## 问题背景

模型读不懂 "The cats were running."，它读的是整数。

每个 NLP 系统的开场都是同样的三个问题：一个词从哪里开始？这个词的词根是什么？什么时候应该把 "run"、"running"、"ran" 当作同一个东西，什么时候又该区别对待？

分词做错了，模型就在学垃圾。如果你的分词器把 `don't` 切成一个 token，却把 `do n't` 切成两个，训练分布就会分裂。如果你的词干提取器把 `organization` 和 `organ` 归并到同一个词干，主题建模就完了。如果你的词形还原器需要词性上下文而你没有传入，动词就会被当成名词处理。

这节课从零实现这三个预处理步骤，然后展示 NLTK 和 spaCy 是如何完成同样工作的，让你看清其中的取舍。

## 核心概念

三个操作，各有职责，也各有失效模式。

**分词（Tokenization）** 把字符串切分成 token。"token" 这个词刻意保持模糊，因为合适的粒度取决于任务：经典 NLP 用词级，Transformer 用子词级，没有空格分隔的语言用字符级。

**词干提取（Stemming）** 用规则砍掉后缀。快速、激进、不动脑子。`running -> run`，`organization -> organ`。第二个例子就是它的失效模式。

**词形还原（Lemmatization）** 利用语法知识把词归约为词典形式。更慢、更准确，需要查找表或形态分析器。`ran -> run`（需要知道 "ran" 是 "run" 的过去式），`better -> good`（需要知道比较级形式）。

经验法则：当速度优先、且能容忍噪声时用词干提取（搜索索引、粗略分类）；当语义重要时用词形还原（问答、语义搜索、任何用户会看到的场景）。

```figure
edit-distance
```

## 从零实现

### 第一步：基于正则的词级分词器

最简单又实用的分词器按非字母数字字符切分，同时把标点保留为独立 token。不完美，也不是最终版，但一行就能跑。

```python
import re

def tokenize(text):
    return re.findall(r"[A-Za-z]+(?:'[A-Za-z]+)?|[0-9]+|[^\sA-Za-z0-9]", text)
```

三个模式按优先级排列：可带内部撇号的单词（`don't`、`it's`）；纯数字；任意单个非空白、非字母数字的字符作为独立 token（标点）。

```python
>>> tokenize("The cats weren't running at 3pm.")
['The', 'cats', "weren't", 'running', 'at', '3', 'pm', '.']
```

注意几个失效模式。`3pm` 被切成 `['3', 'pm']`，因为我们在字母串和数字串之间交替匹配——对大多数任务来说够用了。URL、邮箱、话题标签全都会被切碎。生产环境中，要在通用模式之前加上专门的匹配模式。

### 第二步：Porter 词干提取器（仅 step 1a）

完整的 Porter 算法有五个阶段的规则。仅 step 1a 就覆盖了英语中最高频的后缀，也足以教会你这种模式。

```python
def stem_step_1a(word):
    if word.endswith("sses"):
        return word[:-2]
    if word.endswith("ies"):
        return word[:-2]
    if word.endswith("ss"):
        return word
    if word.endswith("s") and len(word) > 1:
        return word[:-1]
    return word
```

```python
>>> [stem_step_1a(w) for w in ["caresses", "ponies", "caress", "cats"]]
['caress', 'poni', 'caress', 'cat']
```

自上而下读这些规则。`ies -> i` 这条规则就是 `ponies -> poni` 而不是 `pony` 的原因。完整的 Porter 算法有 step 1b 可以修正它。规则之间相互竞争，靠前的规则胜出。规则的顺序比任何单条规则都更重要。

### 第三步：基于查找表的词形还原器

真正的词形还原需要形态学知识。一个易于上手的教学版本可以用一张小型词元表加一个兜底逻辑。

```python
LEMMA_TABLE = {
    ("running", "VERB"): "run",
    ("ran", "VERB"): "run",
    ("runs", "VERB"): "run",
    ("better", "ADJ"): "good",
    ("best", "ADJ"): "good",
    ("cats", "NOUN"): "cat",
    ("cat", "NOUN"): "cat",
    ("were", "VERB"): "be",
    ("was", "VERB"): "be",
    ("is", "VERB"): "be",
}

def lemmatize(word, pos):
    key = (word.lower(), pos)
    if key in LEMMA_TABLE:
        return LEMMA_TABLE[key]
    if pos == "VERB" and word.endswith("ing"):
        return word[:-3]
    if pos == "NOUN" and word.endswith("s"):
        return word[:-1]
    return word.lower()
```

```python
>>> lemmatize("running", "VERB")
'run'
>>> lemmatize("cats", "NOUN")
'cat'
>>> lemmatize("better", "ADJ")
'good'
>>> lemmatize("watched", "VERB")
'watched'
```

最后一个例子是关键的教学点。`watched` 不在我们的表里，而兜底逻辑只处理 `ing`。真正的词形还原要覆盖 `ed`、不规则动词、形容词比较级、发生音变的复数（`children -> child`）。这就是为什么生产系统会使用 WordNet、spaCy 的形态分析组件，或一个完整的形态分析器。

### 第四步：串成流水线

```python
def preprocess(text, pos_tagger=None):
    tokens = tokenize(text)
    stems = [stem_step_1a(t.lower()) for t in tokens]
    tags = pos_tagger(tokens) if pos_tagger else [(t, "NOUN") for t in tokens]
    lemmas = [lemmatize(word, pos) for word, pos in tags]
    return {"tokens": tokens, "stems": stems, "lemmas": lemmas}
```

缺失的环节是词性标注器。Phase 5 · 07（POS Tagging）会从零实现一个。眼下先把所有词默认标为 `NOUN`，并明确承认这一局限。

## 生产实践

NLTK 和 spaCy 提供了生产级实现，各自只需几行代码。

### NLTK

```python
import nltk
nltk.download("punkt_tab")
nltk.download("wordnet")
nltk.download("averaged_perceptron_tagger_eng")

from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk import pos_tag

text = "The cats were running."
tokens = word_tokenize(text)
stems = [PorterStemmer().stem(t) for t in tokens]
lemmatizer = WordNetLemmatizer()
tagged = pos_tag(tokens)


def nltk_pos_to_wordnet(tag):
    if tag.startswith("V"):
        return "v"
    if tag.startswith("J"):
        return "a"
    if tag.startswith("R"):
        return "r"
    return "n"


lemmas = [lemmatizer.lemmatize(t, nltk_pos_to_wordnet(tag)) for t, tag in tagged]
```

`word_tokenize` 能处理缩写、Unicode 和各种你的正则漏掉的边界情况。`PorterStemmer` 会跑完全部五个阶段。`WordNetLemmatizer` 需要把词性标签从 NLTK 的 Penn Treebank 体系转换成 WordNet 的缩写集。上面这段转换胶水代码正是大多数教程跳过的部分。

### spaCy

```python
import spacy

nlp = spacy.load("en_core_web_sm")
doc = nlp("The cats were running.")

for token in doc:
    print(token.text, token.lemma_, token.pos_)
```

```
The      the     DET
cats     cat     NOUN
were     be      AUX
running  run     VERB
.        .       PUNCT
```

spaCy 把整条流水线藏在 `nlp(text)` 背后：分词、词性标注、词形还原全部一起运行。大规模场景下比 NLTK 更快，开箱即用的准确率也更高。代价是你很难轻易替换其中的单个组件。

### 如何选择

| 场景 | 选择 |
|-----------|------|
| 教学、研究、需要替换组件 | NLTK |
| 生产环境、多语言、速度优先 | spaCy |
| Transformer 流水线（反正你会用模型自带的分词器） | 使用 `tokenizers` / `transformers`，跳过经典预处理 |

### 没人提醒你的两个失效模式

大多数教程讲完算法就结束了。但有两件事会在真实的预处理流水线里咬你一口，而它们几乎从不被提及。

**可复现性漂移。** NLTK 和 spaCy 的分词和词形还原行为会在版本之间发生变化。在 spaCy 2.x 中产出 `['do', "n't"]` 的输入，到了 3.x 可能产出 `["don't"]`。你的模型是在一种分布上训练的，推理却跑在另一种分布上。准确率悄悄下滑，没人知道原因。在 `requirements.txt` 中锁定库的版本。写一个预处理回归测试，固化 20 个样例句子的预期分词结果，每次升级都跑一遍。

**训练 / 推理不一致。** 训练时用激进的预处理（小写化、去停用词、词干提取），部署时却直接吃原始用户输入，性能就会崩盘。这是生产环境 NLP 最常见的单一故障来源。如果训练时做了预处理，推理时就必须运行完全相同的函数。把预处理作为函数打包进模型包里，而不是留成一个让服务团队自己重写的 notebook 单元格。

## 交付产物

一个可复用的提示词，帮助工程师在不啃三本教科书的情况下选定预处理策略。

保存为 `outputs/prompt-preprocessing-advisor.md`：

```markdown
---
name: preprocessing-advisor
description: Recommends a tokenization, stemming, and lemmatization setup for an NLP task.
phase: 5
lesson: 01
---

You advise on classical NLP preprocessing. Given a task description, you output:

1. Tokenization choice (regex, NLTK word_tokenize, spaCy, or transformer tokenizer). Explain why.
2. Whether to stem, lemmatize, both, or neither. Explain why.
3. Specific library calls. Name the functions. Quote the POS-tag translation if NLTK is involved.
4. One failure mode the user should test for.

Refuse to recommend stemming for user-visible text. Refuse to recommend lemmatization without POS tags. Flag non-English input as needing a different pipeline.
```

## 练习

1. **简单。** 扩展 `tokenize`，让 URL 保持为单个 token。测试：`tokenize("Visit https://example.com today.")` 应该产出一个完整的 URL token。
2. **中等。** 实现 Porter step 1b：如果单词包含元音且以 `ed` 或 `ing` 结尾，则移除该后缀。处理双辅音规则（`hopping -> hop`，而不是 `hopp`）。
3. **困难。** 构建一个词形还原器：以 WordNet 作为查找表，当 WordNet 没有词条时回退到你的 Porter 词干提取器。在一个带标注的语料上测量准确率，与纯 WordNet 和纯 Porter 方案对比。

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|-----------------|-----------------------|
| Token | 一个词 | 模型实际消费的单元，可以是词、子词、字符或字节。 |
| 词干（Stem） | 词的词根 | 基于规则的后缀剥离结果，不一定是真实存在的词。 |
| 词元（Lemma） | 词典形式 | 你在词典里查的那个形式。需要语法上下文才能正确计算。 |
| 词性标签（POS tag） | 词性 | 像 NOUN、VERB、ADJ 这样的类别。准确的词形还原离不开它。 |
| 形态学（Morphology） | 词形变化规则 | 词如何随时态、单复数、格而变形。词形还原依赖于它。 |

## 延伸阅读

- [Porter, M. F. (1980). An algorithm for suffix stripping](https://tartarus.org/martin/PorterStemmer/def.txt) —— 原始论文，只有五页，至今仍是最清晰的讲解。
- [spaCy 101 — linguistic features](https://spacy.io/usage/linguistic-features) —— 一条真实流水线是如何接线的。
- [NLTK book, chapter 3](https://www.nltk.org/book/ch03.html) —— 那些你还没想到的分词边界情况。
