# BERT — 掩码语言建模

> GPT 预测下一个词，BERT 预测缺失的词。一句话的差别——却催生了之后五年所有与嵌入相关的一切。

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 7 · 05 (Full Transformer), Phase 5 · 02 (Text Representation)
**Time:** ~45 minutes

## 问题背景

2018 年，每个 NLP 任务——情感分析、命名实体识别（NER）、问答、文本蕴含——都要在各自的标注数据上从零训练自己的模型。当时没有一个预训练好的「理解英语」的检查点可以拿来微调。ELMo（2018）证明了可以用双向 LSTM 预训练上下文嵌入；它有帮助，但泛化能力不足。

BERT（Devlin et al. 2018）提出了一个问题：如果我们拿一个 Transformer 编码器，用互联网上的所有句子来训练它，并强迫它根据两侧的上下文预测缺失的词，会怎样？之后只需在下游任务上微调一个头部即可。这种参数效率堪称一场革命。

结果是：在 18 个月内，BERT 及其变体（RoBERTa、ALBERT、ELECTRA）统治了当时存在的所有 NLP 排行榜。到 2020 年，地球上每一个搜索引擎、内容审核流水线和语义搜索系统里都装着一个 BERT。

到了 2026 年，仅编码器（encoder-only）模型仍然是分类、检索和结构化抽取的正确工具——它们每 token 的运行速度比解码器快 5–10 倍，其嵌入是每一套现代检索栈的支柱。ModernBERT（2024 年 12 月）用 Flash Attention + RoPE + GeGLU 把这一架构推进到了 8K 上下文。

## 核心概念

![Masked language modeling: pick tokens, mask them, predict originals](../assets/bert-mlm.svg)

### 训练信号

取一个句子：`the quick brown fox jumps over the lazy dog`。

随机掩码 15% 的 token：

```
input:  the [MASK] brown fox jumps [MASK] the lazy dog
target: the  quick brown fox jumps  over  the lazy dog
```

训练模型在被掩码的位置预测原始 token。由于编码器是双向的，在位置 1 预测 `[MASK]` 时可以利用位置 2 之后的 `brown fox jumps`。这正是 GPT 做不到的事。

### BERT 的掩码规则

在被选中用于预测的 15% 的 token 中：

- 80% 被替换为 `[MASK]`。
- 10% 被替换为一个随机 token。
- 10% 保持不变。

为什么不全部替换成 `[MASK]`？因为 `[MASK]` 在推理时永远不会出现。如果训练时让模型在 100% 的掩码位置都看到 `[MASK]`，就会在预训练和微调之间制造分布偏移。10% 随机 + 10% 不变的设计让模型保持「诚实」。

### 下一句预测（NSP）——以及它为何被弃用

最初的 BERT 还训练了 NSP 任务：给定两个句子 A 和 B，预测 B 是否紧跟在 A 之后。RoBERTa（2019）通过消融实验证明 NSP 有害无益。现代编码器都跳过了它。

### 2026 年的变化：ModernBERT

2024 年的 ModernBERT 论文用 2026 年的基础组件重建了这一模块：

| 组件 | 原始 BERT（2018） | ModernBERT（2024） |
|-----------|----------------------|-------------------|
| 位置编码 | 可学习的绝对位置 | RoPE |
| 激活函数 | GELU | GeGLU |
| 归一化 | LayerNorm | Pre-norm RMSNorm |
| 注意力 | 全稠密 | 局部（128）+ 全局交替 |
| 上下文长度 | 512 | 8192 |
| 分词器 | WordPiece | BPE |

而且与 2018 年的技术栈不同，它原生支持 Flash Attention。在序列长度为 8K 时，推理速度比 DeBERTa-v3 快 2–3 倍，GLUE 分数还更高。

### 2026 年仍然选择编码器的应用场景

| 任务 | 编码器为何胜过解码器 |
|------|---------------------------|
| 检索 / 语义搜索嵌入 | 双向上下文 = 每 token 更高的嵌入质量 |
| 分类（情感、意图、毒性检测） | 一次前向传播；没有生成开销 |
| NER / token 级标注 | 逐位置输出，天然双向 |
| 零样本蕴含（NLI） | 在编码器之上加一个分类头 |
| RAG 的重排序器 | 交叉编码器打分，比 LLM 重排序器快 10 倍 |

```figure
transformer-residual
```

## 从零实现

### 第 1 步：掩码逻辑

见 `code/main.py`。函数 `create_mlm_batch` 接收一个 token ID 列表、词表大小和掩码概率，返回输入 ID（已应用掩码）和标签（只在掩码位置有值，其余位置为 -100——PyTorch 的 ignore index 约定）。

```python
def create_mlm_batch(tokens, vocab_size, mask_prob=0.15, rng=None):
    input_ids = list(tokens)
    labels = [-100] * len(tokens)
    for i, t in enumerate(tokens):
        if rng.random() < mask_prob:
            labels[i] = t
            r = rng.random()
            if r < 0.8:
                input_ids[i] = MASK_ID
            elif r < 0.9:
                input_ids[i] = rng.randrange(vocab_size)
            # else: keep original
    return input_ids, labels
```

### 第 2 步：在一个微型语料上运行 MLM 预测

在一个 20 词词表、200 个句子的语料上训练一个 2 层编码器 + MLM 头。不计算梯度——我们只做前向传播的健全性检查。完整训练需要 PyTorch。

### 第 3 步：比较不同掩码类型

展示三分掩码规则如何让模型在没有 `[MASK]` 的情况下依然可用。分别在一个未掩码的句子和一个掩码后的句子上做预测。两者都应该产生合理的 token 分布，因为模型在训练中两种模式都见过。

### 第 4 步：微调头部

在一个玩具情感数据集上，把 MLM 头替换成分类头。只训练头部，编码器保持冻结。这就是所有 BERT 应用都遵循的模式。

## 生产实践

```python
from transformers import AutoModel, AutoTokenizer

tok = AutoTokenizer.from_pretrained("answerdotai/ModernBERT-base")
model = AutoModel.from_pretrained("answerdotai/ModernBERT-base")

text = "Attention is all you need."
inputs = tok(text, return_tensors="pt")
out = model(**inputs).last_hidden_state   # (1, N, 768)
```

**嵌入模型就是微调过的 BERT。** `sentence-transformers` 中诸如 `all-MiniLM-L6-v2` 的模型，就是用对比损失训练的 BERT。编码器没变，变的是损失函数。

**交叉编码器重排序器也是微调过的 BERT。** 在 `[CLS] query [SEP] doc [SEP]` 上做成对分类。查询与文档之间的双向注意力，正是交叉编码器在质量上压过双编码器（biencoder）的原因。

**2026 年什么时候不该选 BERT。** 任何生成式任务。编码器没有合理的方式自回归地生成 token。另外：在 1B 参数以下的场景，小型解码器（Phi-3-Mini、Qwen2-1.5B）能以更高的灵活性达到同等质量。

## 交付产物

见 `outputs/skill-bert-finetuner.md`。该 skill 用于为新的分类或抽取任务规划一次 BERT 微调（骨干网络选择、头部设计、数据、评估、停止条件）。

## 练习

1. **简单。** 运行 `code/main.py`，统计并打印 10,000 个 token 上的掩码分布。确认约 15% 被选中，其中约 80% 变成了 `[MASK]`。
2. **中等。** 实现全词掩码（whole-word masking）：如果一个词被切分成多个子词，要么把所有子词一起掩码，要么都不掩码。在一个 500 句的语料上测量这是否提升了 MLM 准确率。
3. **困难。** 在一个公开数据集的 10,000 个句子上训练一个微型（2 层，d=64）BERT。针对 SST-2 情感任务微调 `[CLS]` token。与参数量相同的仅解码器基线对比——谁更胜一筹？

## 关键术语

| 术语 | 大家怎么说 | 实际含义 |
|------|-----------------|-----------------------|
| MLM | 「掩码语言建模」 | 训练信号：随机把 15% 的 token 替换成 `[MASK]`，然后预测原始 token。 |
| 双向（Bidirectional） | 「两个方向都能看」 | 编码器的注意力没有因果掩码——每个位置都能看到其他所有位置。 |
| `[CLS]` | 「池化 token」 | 拼接在每个序列开头的特殊 token；它的最终嵌入被用作句子级表示。 |
| `[SEP]` | 「分段分隔符」 | 用于分隔成对的序列（例如查询/文档、句子 A/B）。 |
| NSP | 「下一句预测」 | BERT 的第二个预训练任务；RoBERTa 证明它没有用，2019 年之后被弃用。 |
| 微调（Fine-tuning） | 「适配到某个任务」 | 编码器大体冻结；在其上训练一个小的头部来完成下游任务。 |
| 交叉编码器（Cross-encoder） | 「重排序器」 | 同时接收查询和文档作为输入、输出相关性分数的 BERT。 |
| ModernBERT | 「2024 年的翻新版」 | 用 RoPE、RMSNorm、GeGLU、局部/全局交替注意力和 8K 上下文重建的编码器。 |

## 延伸阅读

- [Devlin et al. (2018). BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/abs/1810.04805) — 原始论文。
- [Liu et al. (2019). RoBERTa: A Robustly Optimized BERT Pretraining Approach](https://arxiv.org/abs/1907.11692) — 如何正确地训练 BERT；终结了 NSP。
- [Clark et al. (2020). ELECTRA: Pre-training Text Encoders as Discriminators Rather Than Generators](https://arxiv.org/abs/2003.10555) — 在算力相同的条件下，替换 token 检测胜过 MLM。
- [Warner et al. (2024). Smarter, Better, Faster, Longer: A Modern Bidirectional Encoder](https://arxiv.org/abs/2412.13663) — ModernBERT 论文。
- [HuggingFace `modeling_bert.py`](https://github.com/huggingface/transformers/blob/main/src/transformers/models/bert/modeling_bert.py) — 编码器的权威参考实现。
