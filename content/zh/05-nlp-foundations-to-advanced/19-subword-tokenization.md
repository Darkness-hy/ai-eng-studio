# 子词分词 —— BPE、WordPiece、Unigram、SentencePiece

> 词级分词器遇到没见过的词就卡壳，字符级分词器又会让序列长度爆炸。子词分词器在两者之间取得平衡——如今每一个现代 LLM 都构建在它之上。

**Type:** Learn
**Languages:** Python
**Prerequisites:** Phase 5 · 01 (Text Processing), Phase 5 · 04 (GloVe / FastText / Subword)
**Time:** ~60 minutes

## 问题背景

你的词表有 50,000 个词。用户输入了 "untokenizable"，你的分词器返回 `[UNK]`，模型对这个词完全失去了信号。更糟的是：语料中第 90 百分位的文档包含 40 个罕见词，相当于每篇文档丢掉 40 比特的信息。

子词分词（subword tokenization）解决了这个问题。常见词保持为单个 token，罕见词则分解成有意义的片段：`untokenizable` → `un`、`token`、`izable`。由于任何字符串归根结底都是字节序列，训练数据可以覆盖一切。

2026 年的每一个前沿 LLM 都构建在三种算法之一（BPE、Unigram、WordPiece）之上，并由三个库之一（tiktoken、SentencePiece、HF Tokenizers）封装。不选定其中一种，你就无法发布语言模型。

## 核心概念

![BPE vs Unigram vs WordPiece, character-by-character](../assets/subword-tokenization.svg)

**BPE（Byte-Pair Encoding，字节对编码）。** 从字符级词表出发，统计所有相邻字符对，把出现最频繁的一对合并成新 token，反复迭代直到达到目标词表大小。这是当前的主流算法：GPT-2/3/4、Llama、Gemma、Qwen2、Mistral 都在用。

**字节级 BPE（Byte-level BPE）。** 算法相同，但作用在原始字节（256 个基础 token）而不是 Unicode 字符上。保证绝不出现 `[UNK]` token——任何字节序列都能编码。GPT-2 使用 50,257 个 token（256 个字节 + 50,000 次合并 + 1 个特殊 token）。

**Unigram。** 从一个巨大的词表出发，给每个 token 赋一个一元（unigram）概率，然后迭代剪枝：每次删掉对语料对数似然影响最小的 token。推理时是概率式的：可以对分词结果采样（对通过子词正则化做数据增强很有用）。T5、mBART、ALBERT、XLNet、Gemma 都在用。

**WordPiece。** 合并时选择能最大化训练语料似然的字符对，而不是单纯按频率。BERT、DistilBERT、ELECTRA 在用。

**SentencePiece vs tiktoken。** SentencePiece 是直接在原始 Unicode 文本上*训练*词表（BPE 或 Unigram）的库，它把空白字符编码为 `▁`。tiktoken 是 OpenAI 的高速*编码器*，只针对预构建词表做编码，不做训练。

经验法则：

- **训练新词表：** 用 SentencePiece（多语言、无需预分词）或 HF Tokenizers。
- **针对 GPT 词表的快速推理：** 用 tiktoken（cl100k_base、o200k_base）。
- **两者都要：** 用 HF Tokenizers——一个库同时覆盖训练与上线。

```figure
bpe-merge
```

## 从零实现

### 第 1 步：从零写一个 BPE

见 `code/main.py`。核心循环：

```python
def train_bpe(corpus, num_merges):
    vocab = {tuple(word) + ("</w>",): count for word, count in corpus.items()}
    merges = []
    for _ in range(num_merges):
        pairs = Counter()
        for symbols, freq in vocab.items():
            for a, b in zip(symbols, symbols[1:]):
                pairs[(a, b)] += freq
        if not pairs:
            break
        best = pairs.most_common(1)[0][0]
        merges.append(best)
        vocab = apply_merge(vocab, best)
    return merges
```

这段算法编码了三个事实：`</w>` 标记词尾，让 "low"（作为后缀）和 "lower"（作为前缀）保持区分；频率加权使高频字符对率先胜出；合并列表是有序的——推理时必须按训练顺序应用合并。

### 第 2 步：用学到的合并规则做编码

```python
def encode_bpe(word, merges):
    symbols = list(word) + ["</w>"]
    for a, b in merges:
        i = 0
        while i < len(symbols) - 1:
            if symbols[i] == a and symbols[i + 1] == b:
                symbols = symbols[:i] + [a + b] + symbols[i + 2:]
            else:
                i += 1
    return symbols
```

朴素实现的复杂度是 O(n·|merges|)。生产级实现（tiktoken、HF Tokenizers）用合并优先级查找加优先队列，能跑到近似线性时间。

### 第 3 步：实战 SentencePiece

```python
import sentencepiece as spm

spm.SentencePieceTrainer.train(
    input="corpus.txt",
    model_prefix="my_tokenizer",
    vocab_size=8000,
    model_type="bpe",          # or "unigram"
    character_coverage=0.9995, # lower for CJK (e.g. 0.9995 for English, 0.995 for Japanese)
    normalization_rule_name="nmt_nfkc",
)

sp = spm.SentencePieceProcessor(model_file="my_tokenizer.model")
print(sp.encode("untokenizable", out_type=str))
# ['▁un', 'token', 'izable']
```

注意：不需要预分词，空格被编码为 `▁`，`character_coverage` 控制罕见字符是被保留还是被映射到 `<unk>` 的激进程度。

### 第 4 步：用 tiktoken 对接 OpenAI 兼容词表

```python
import tiktoken
enc = tiktoken.get_encoding("o200k_base")
print(enc.encode("untokenizable"))        # [127340, 101028]
print(len(enc.encode("Hello, world!")))   # 4
```

只做编码，速度快（Rust 后端）。与 GPT-4/5 的分词完全一致，适合字节计数、成本估算、上下文窗口预算。

## 2026 年仍在上线的常见陷阱

- **分词器漂移。** 用词表 A 训练，却用词表 B 部署。token ID 不一致，模型输出垃圾。在 CI 中校验 `tokenizer.json` 的哈希。
- **空白歧义。** BPE 中 "hello" 和 " hello" 产生不同的 token。永远显式指定 `add_special_tokens` 和 `add_prefix_space`。
- **多语言训练不足。** 以英文为主的语料训练出的词表，会把非拉丁文字切成 5-10 倍数量的 token。同样的提示词，在 GPT-3.5 上用日语或阿拉伯语要贵 5-10 倍。o200k_base 部分缓解了这个问题。
- **表情符号被切碎。** 一个 emoji 可能占 5 个 token。做上下文预算时记得检查 emoji 的处理方式。

## 生产实践

2026 年的技术选型：

| 场景 | 选择 |
|-----------|------|
| 从零训练单语言模型 | HF Tokenizers（BPE） |
| 训练多语言模型 | SentencePiece（Unigram，`character_coverage=0.9995`） |
| 提供 OpenAI 兼容 API | tiktoken（GPT-4+ 用 `o200k_base`） |
| 领域专用词表（代码、数学、蛋白质） | 在领域语料上训练自定义 BPE，再与基础词表合并 |
| 端侧推理、小模型 | Unigram（更小的词表效果更好） |

词表大小是一个随规模变化的决策，不是常数。粗略经验：参数量小于 1B 用 32k，1-10B 用 50-100k，多语言或前沿模型用 200k 以上。

## 交付产物

保存为 `outputs/skill-bpe-vs-wordpiece.md`：

```markdown
---
name: tokenizer-picker
description: Pick tokenizer algorithm, vocab size, library for a given corpus and deployment target.
version: 1.0.0
phase: 5
lesson: 19
tags: [nlp, tokenization]
---

Given a corpus (size, languages, domain) and deployment target (training from scratch / fine-tuning / API-compatible inference), output:

1. Algorithm. BPE, Unigram, or WordPiece. One-sentence reason.
2. Library. SentencePiece, HF Tokenizers, or tiktoken. Reason.
3. Vocab size. Rounded to nearest 1k. Reason tied to model size and language coverage.
4. Coverage settings. `character_coverage`, `byte_fallback`, special-token list.
5. Validation plan. Average tokens-per-word on held-out set, OOV rate, compression ratio, round-trip decode equality.

Refuse to train a character-coverage <0.995 tokenizer on corpora with rare-script content. Refuse to ship a vocab without a frozen `tokenizer.json` hash check in CI. Flag any monolingual tokenizer under 16k vocab as likely under-spec.
```

## 练习

1. **简单。** 在 `code/main.py` 的小语料上训练一个 500 次合并的 BPE，编码三个训练时没见过的词。有多少个词恰好产生 1 个 token，多少个产生超过 1 个？
2. **中等。** 在 100 句英文 Wikipedia 句子上，比较 `cl100k_base`、`o200k_base` 和你自己用 vocab=32k 训练的 SentencePiece BPE 的 token 数量。报告各自的压缩比。
3. **困难。** 用 BPE、Unigram、WordPiece 在同一语料上分别训练分词器，在一个小型情感分类器上测量各自的下游准确率。分词器的选择能让 F1 变动超过 1 个点吗？

## 关键术语

| 术语 | 大家怎么说 | 实际含义 |
|------|-----------------|-----------------------|
| BPE | 字节对编码 | 贪心合并最高频字符对，直到达到目标词表大小。 |
| 字节级 BPE | 永远没有未知 token | 在原始 256 个字节上做 BPE；GPT-2 / Llama 在用。 |
| Unigram | 概率式分词器 | 从大候选集出发，按对数似然剪枝；T5、Gemma 在用。 |
| SentencePiece | 处理空格的那个 | 在原始文本上训练 BPE/Unigram 的库；空格编码为 `▁`。 |
| tiktoken | 速度快的那个 | OpenAI 基于 Rust 的 BPE 编码器，针对预构建词表，不做训练。 |
| 合并列表 | 那串魔法数字 | `(a, b) → ab` 形式的有序合并列表；推理时按顺序应用。 |
| 字符覆盖率 | 多罕见算太罕见？ | 分词器必须覆盖的训练语料字符比例；典型值约 0.9995。 |

## 延伸阅读

- [Sennrich, Haddow, Birch (2015). Neural Machine Translation of Rare Words with Subword Units](https://arxiv.org/abs/1508.07909) —— BPE 原始论文。
- [Kudo (2018). Subword Regularization with Unigram Language Model](https://arxiv.org/abs/1804.10959) —— Unigram 原始论文。
- [Kudo, Richardson (2018). SentencePiece: A simple and language independent subword tokenizer](https://arxiv.org/abs/1808.06226) —— 库本身的论文。
- [Hugging Face — Summary of the tokenizers](https://huggingface.co/docs/transformers/tokenizer_summary) —— 简明参考。
- [OpenAI tiktoken repo](https://github.com/openai/tiktoken) —— cookbook 与编码列表。
