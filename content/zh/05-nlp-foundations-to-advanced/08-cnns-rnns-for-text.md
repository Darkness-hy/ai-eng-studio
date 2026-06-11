# 用于文本的 CNN 与 RNN

> 卷积学习 n-gram，循环网络负责记忆。两者都已被注意力取代，但在受限硬件上仍有用武之地。

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 3 · 11 (PyTorch Intro), Phase 5 · 03 (Word Embeddings), Phase 4 · 02 (Convolutions from Scratch)
**Time:** ~75 minutes

## 问题背景

TF-IDF 和 Word2Vec 产生的是忽略词序的扁平向量。基于它们构建的分类器无法区分 `dog bites man` 和 `man bites dog`。而词序有时恰恰承载着关键信号。

在 Transformer 出现之前，有两类架构填补了这个空缺。

**面向文本的卷积网络（TextCNN）。** 在词嵌入序列上做 1D 卷积。一个宽度为 3 的滤波器就是一个可学习的三元组（trigram）检测器：它覆盖三个词并输出一个分数。叠加不同宽度（2、3、4、5）的滤波器即可检测多尺度模式。再用最大池化得到固定大小的表示。扁平、并行、快速。

**循环网络（RNN、LSTM、GRU）。** 逐个处理 token，维护一个向前传递信息的隐藏状态。串行、带记忆、可处理变长输入。从 2014 年到 2017 年统治了序列建模，直到注意力出现。

本课将两者都实现一遍，然后点明那个催生注意力机制的失败之处。

## 核心概念

**TextCNN**（Kim, 2014）。先把 token 转成嵌入。一个宽度为 `k` 的 1D 卷积把滤波器滑过嵌入序列上连续的 `k`-gram，产生一张特征图。在该特征图上做全局最大池化，选出最强的激活值。把多个滤波器宽度的最大池化输出拼接起来，送入分类头。

为什么有效。一个滤波器就是一个可学习的 n-gram。最大池化具有位置不变性，所以 "not good" 无论出现在评论的开头还是中间，触发的都是同一个特征。三种滤波器宽度、每种 100 个滤波器，就得到 300 个学到的 n-gram 检测器。训练是并行的，没有串行依赖。

**RNN。** 在每个时间步 `t`，隐藏状态为 `h_t = f(W * x_t + U * h_{t-1} + b)`。`W`、`U`、`b` 在所有时间步之间共享。时刻 `T` 的隐藏状态是整个前缀的摘要。做分类时，在 `h_1 ... h_T` 上做池化（最大、平均或取最后一个）。

普通 RNN 受困于梯度消失。**LSTM** 增加了门控机制，决定遗忘什么、存储什么、输出什么，从而在长序列中稳定梯度。**GRU** 把 LSTM 简化为两个门，参数更少而表现相近。

**双向 RNN** 同时运行一个前向 RNN 和一个后向 RNN，并把隐藏状态拼接起来。这样每个 token 的表示都能同时看到左右两侧的上下文。对标注类任务必不可少。

```figure
rnn-unroll
```

## 从零实现

### Step 1: 用 PyTorch 实现 TextCNN

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


class TextCNN(nn.Module):
    def __init__(self, vocab_size, embed_dim, n_classes, filter_widths=(2, 3, 4), n_filters=64, dropout=0.3):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.convs = nn.ModuleList([
            nn.Conv1d(embed_dim, n_filters, kernel_size=k)
            for k in filter_widths
        ])
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(n_filters * len(filter_widths), n_classes)

    def forward(self, token_ids):
        x = self.embed(token_ids).transpose(1, 2)
        pooled = []
        for conv in self.convs:
            c = F.relu(conv(x))
            p = F.max_pool1d(c, c.size(2)).squeeze(2)
            pooled.append(p)
        h = torch.cat(pooled, dim=1)
        return self.fc(self.dropout(h))
```

`transpose(1, 2)` 把 `[batch, seq_len, embed_dim]` 变形为 `[batch, embed_dim, seq_len]`，因为 `nn.Conv1d` 把中间那个维度当作通道。池化后的输出大小固定，与输入长度无关。

### Step 2: LSTM 分类器

```python
class LSTMClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, n_classes, bidirectional=True, dropout=0.3):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True, bidirectional=bidirectional)
        factor = 2 if bidirectional else 1
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim * factor, n_classes)

    def forward(self, token_ids):
        x = self.embed(token_ids)
        out, _ = self.lstm(x)
        pooled = out.max(dim=1).values
        return self.fc(self.dropout(pooled))
```

在整个序列上做最大池化，而不是只取最后状态。做分类时，最大池化通常优于取最后一个隐藏状态，因为长序列末尾的信息往往会主导最后状态。

### Step 3: 梯度消失演示（直觉）

没有门控的普通 RNN 学不会长程依赖。考虑一个玩具任务：预测 token `A` 是否在序列中的任意位置出现过。如果 `A` 在第 1 个位置而序列长 100 个 token，那么来自损失的梯度必须反向流过 99 次循环权重的乘法。权重小于 1，梯度就消失；大于 1，梯度就爆炸。

```python
def vanishing_gradient_sim(seq_len, recurrent_weight=0.9):
    import math
    return math.pow(recurrent_weight, seq_len)


# At weight=0.9 over 100 steps:
#   0.9 ^ 100 ≈ 2.7e-5
# The gradient from step 100 to step 1 is effectively zero.
```

LSTM 用一个**细胞状态（cell state）**解决了这个问题：它贯穿整个网络，几乎只有加性交互（遗忘门会对它做乘法缩放，但梯度仍能沿这条"高速公路"流动）。GRU 用更少的参数做了类似的事。两者都能让训练在 100 步以上的序列中保持稳定。

### Step 4: 为什么这仍然不够

即便有了 LSTM，三个问题依然存在。

1. **串行瓶颈。** 在长度为 1000 的序列上训练 RNN，需要 1000 个串行的前向/反向步骤。无法在时间维度上并行。
2. **编码器-解码器结构中固定大小的上下文向量。** 解码器只能看到编码器的最终隐藏状态，整个输入都被压缩进去。长输入会丢失细节。第 09 课会直接讨论这个问题。
3. **远距离依赖的准确率天花板。** LSTM 比普通 RNN 强，但在跨越 200 步以上传递特定信息时依然吃力。

注意力把这三个问题全部解决了。Transformer 则彻底抛弃了循环结构。第 10 课就是那个转折点。

## 生产实践

PyTorch 的 `nn.LSTM`、`nn.GRU` 和 `nn.Conv1d` 都是生产可用的。训练代码也是标准写法。

Hugging Face 提供的预训练嵌入可以直接接入作为输入层：

```python
from transformers import AutoModel

encoder = AutoModel.from_pretrained("bert-base-uncased")
for param in encoder.parameters():
    param.requires_grad = False


class BertCNN(nn.Module):
    def __init__(self, n_classes, filter_widths=(2, 3, 4), n_filters=64):
        super().__init__()
        self.encoder = encoder
        self.convs = nn.ModuleList([nn.Conv1d(768, n_filters, kernel_size=k) for k in filter_widths])
        self.fc = nn.Linear(n_filters * len(filter_widths), n_classes)

    def forward(self, input_ids, attention_mask):
        with torch.no_grad():
            out = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        x = out.transpose(1, 2)
        pooled = [F.max_pool1d(F.relu(conv(x)), kernel_size=conv(x).size(2)).squeeze(2) for conv in self.convs]
        return self.fc(torch.cat(pooled, dim=1))
```

按约束选型的检查清单。

- **边缘 / 端侧推理。** TextCNN 加 GloVe 嵌入比 Transformer 小 10 到 100 倍。如果部署目标是手机，就用这套方案。
- **流式 / 在线分类。** RNN 逐个处理 token；Transformer 需要完整序列。对实时到达的文本，LSTM 仍占优势。
- **作为基线的小模型。** 在新任务上快速迭代。一个 TextCNN 在 CPU 上 5 分钟就能训完。
- **数据有限的序列标注。** BiLSTM-CRF（第 06 课）在 1k 到 10k 条标注句子的规模下，仍是生产级的 NER 架构。

其余情况一律交给 Transformer。

## 交付产物

保存为 `outputs/prompt-text-encoder-picker.md`：

```markdown
---
name: text-encoder-picker
description: Pick a text encoder architecture for a given constraint set.
phase: 5
lesson: 08
---

Given constraints (task, data volume, latency budget, deploy target, compute budget), output:

1. Encoder architecture: TextCNN, BiLSTM, BiLSTM-CRF, transformer fine-tune, or "use a pretrained transformer as a frozen encoder + small head".
2. Embedding input: random init, GloVe / fastText frozen, or contextualized transformer embeddings.
3. Training recipe in 5 lines: optimizer, learning rate, batch size, epochs, regularization.
4. One monitoring signal. For RNN/CNN models: attention mechanism absence means they miss long-range deps; check per-length accuracy. For transformers: fine-tuning collapse if LR too high; check train loss.

Refuse to recommend fine-tuning a transformer when data is under ~500 labeled examples without showing that a TextCNN / BiLSTM baseline has plateaued. Flag edge deployment as needing architecture-before-everything.
```

## 练习

1. **简单。** 在一个三分类玩具数据集（数据由你自己构造）上训练 TextCNN。验证滤波器宽度组合（2, 3, 4）在平均 F1 上优于单一宽度（3）。
2. **中等。** 为 LSTM 分类器实现最大池化、平均池化和最后状态池化三种方式。在一个小数据集上对比；记录哪种池化胜出，并提出你的解释假设。
3. **困难。** 构建一个 BiLSTM-CRF NER 标注器（结合第 06 课和本课）。在 CoNLL-2003 上训练。与第 06 课的纯 CRF 基线以及 BERT 微调进行对比。报告训练时间、显存占用和 F1。

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|-----------------|-----------------------|
| TextCNN | 文本卷积网络 | 在词嵌入上叠加 1D 卷积并做全局最大池化。Kim（2014）。 |
| RNN | 循环网络 | 隐藏状态在每个时间步更新：`h_t = f(W x_t + U h_{t-1})`。 |
| LSTM | 带门控的 RNN | 增加输入门 / 遗忘门 / 输出门和一个细胞状态。能在长序列中稳定训练。 |
| GRU | 简化版 LSTM | 两个门而不是三个。准确率相近，参数更少。 |
| 双向（Bidirectional） | 两个方向 | 前向 + 后向 RNN 拼接。每个 token 都能看到上下文的两侧。 |
| 梯度消失 | 训练信号消亡 | 普通 RNN 中反复乘以小于 1 的权重，使早期时间步的梯度趋近于零。 |

## 延伸阅读

- [Kim, Y. (2014). Convolutional Neural Networks for Sentence Classification](https://arxiv.org/abs/1408.5882) — TextCNN 原始论文。八页，易读。
- [Hochreiter, S. and Schmidhuber, J. (1997). Long Short-Term Memory](https://www.bioinf.jku.at/publications/older/2604.pdf) — LSTM 原始论文。出人意料地清晰。
- [Olah, C. (2015). Understanding LSTM Networks](https://colah.github.io/posts/2015-08-Understanding-LSTMs/) — 让所有人都能看懂 LSTM 的那组图解。
