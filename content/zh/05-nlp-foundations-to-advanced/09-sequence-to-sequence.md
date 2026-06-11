# 序列到序列模型

> 两个 RNN 假扮成一个翻译器。它们撞上的瓶颈，正是注意力机制存在的理由。

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 5 · 08 (CNNs + RNNs for Text), Phase 3 · 11 (PyTorch Intro)
**Time:** ~75 minutes

## 问题背景

分类任务把一个变长序列映射到一个标签。翻译任务则把一个变长序列映射到另一个变长序列。输入和输出来自不同的词表，可能是不同的语言，长度也不保证对应。

seq2seq 架构（Sutskever, Vinyals, Le, 2014）用一个刻意简单的方案解决了这个问题。两个 RNN。一个读取源句子，产出一个固定大小的上下文向量。另一个读取这个向量，逐 token 生成目标句子。和你在第 08 课写的代码一样，只是拼接方式不同。

这个架构值得学习有两个原因。第一，上下文向量瓶颈是 NLP 中最有教学价值的失败案例，它解释了注意力和 Transformer 所擅长的一切。第二，它的训练方法（教师强制、计划采样、推理时的束搜索）至今仍适用于包括 LLM 在内的所有现代生成系统。

## 核心概念

**编码器（Encoder）。** 一个读取源句子的 RNN。它的最后一个隐藏状态就是**上下文向量（context vector）**——对整个输入的固定大小的摘要。理论上，除了源句子本身，什么都不丢。

**解码器（Decoder）。** 另一个 RNN，用上下文向量初始化。每一步它接收上一步生成的 token 作为输入，输出一个覆盖目标词表的概率分布。通过采样或 argmax 选出下一个 token，再喂回去。如此循环，直到生成 `<EOS>` token 或达到最大长度。

**训练：** 在解码器的每一步计算交叉熵损失，对整个序列求和。然后对两个网络做标准的沿时间反向传播。

**教师强制（Teacher forcing）。** 训练时，解码器在第 `t` 步的输入是位置 `t-1` 处的*真实* token，而不是解码器自己上一步的预测。这能稳定训练；没有它，早期的错误会层层累积，模型永远学不会。而推理时只能用模型自己的预测，所以训练和推理之间始终存在分布差异。这个差异叫做**曝光偏差（exposure bias）**。

**瓶颈。** 编码器对源句子学到的一切都必须挤进那一个上下文向量里。长句子会丢失细节。罕见词会变得模糊。词序重排（chat noir 与 black cat）只能靠死记硬背，无法通过计算得出。

注意力机制（第 10 课）的解法是让解码器看到*每一个*编码器隐藏状态，而不只是最后一个。这就是它的全部卖点。

```figure
lstm-gates
```

## 从零实现

### 第 1 步：编码器

```python
import torch
import torch.nn as nn


class Encoder(nn.Module):
    def __init__(self, src_vocab_size, embed_dim, hidden_dim):
        super().__init__()
        self.embed = nn.Embedding(src_vocab_size, embed_dim, padding_idx=0)
        self.gru = nn.GRU(embed_dim, hidden_dim, batch_first=True)

    def forward(self, src):
        e = self.embed(src)
        outputs, hidden = self.gru(e)
        return outputs, hidden
```

`outputs` 的形状是 `[batch, seq_len, hidden_dim]`——每个输入位置一个隐藏状态。`hidden` 的形状是 `[1, batch, hidden_dim]`——最后一步的状态。第 08 课说过"分类任务要对 outputs 做池化"。这里我们保留最后的隐藏状态作为上下文向量，忽略每一步的输出。

### 第 2 步：解码器

```python
class Decoder(nn.Module):
    def __init__(self, tgt_vocab_size, embed_dim, hidden_dim):
        super().__init__()
        self.embed = nn.Embedding(tgt_vocab_size, embed_dim, padding_idx=0)
        self.gru = nn.GRU(embed_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, tgt_vocab_size)

    def forward(self, token, hidden):
        e = self.embed(token)
        out, hidden = self.gru(e, hidden)
        logits = self.fc(out)
        return logits, hidden
```

解码器每次只前进一步。输入：一个批次的单个 token 和当前隐藏状态。输出：下一个 token 的词表 logits 和更新后的隐藏状态。

### 第 3 步：带教师强制的训练循环

```python
def train_batch(encoder, decoder, src, tgt, bos_id, optimizer, teacher_forcing_ratio=0.9):
    optimizer.zero_grad()
    _, hidden = encoder(src)
    batch_size, tgt_len = tgt.shape
    input_token = torch.full((batch_size, 1), bos_id, dtype=torch.long)
    loss = 0.0
    loss_fn = nn.CrossEntropyLoss(ignore_index=0)

    for t in range(tgt_len):
        logits, hidden = decoder(input_token, hidden)
        step_loss = loss_fn(logits.squeeze(1), tgt[:, t])
        loss += step_loss
        use_teacher = torch.rand(1).item() < teacher_forcing_ratio
        if use_teacher:
            input_token = tgt[:, t].unsqueeze(1)
        else:
            input_token = logits.argmax(dim=-1)

    loss.backward()
    optimizer.step()
    return loss.item() / tgt_len
```

有两个旋钮值得点名。`ignore_index=0` 让损失跳过填充 token。`teacher_forcing_ratio` 是每一步使用真实 token 而非模型预测的概率。从 1.0（完全教师强制）开始，在训练过程中逐渐退火到约 0.5，以缩小曝光偏差带来的差距。

### 第 4 步：推理循环（贪心）

```python
@torch.no_grad()
def greedy_decode(encoder, decoder, src, bos_id, eos_id, max_len=50):
    _, hidden = encoder(src)
    batch_size = src.shape[0]
    input_token = torch.full((batch_size, 1), bos_id, dtype=torch.long)
    output_ids = []
    for _ in range(max_len):
        logits, hidden = decoder(input_token, hidden)
        next_token = logits.argmax(dim=-1)
        output_ids.append(next_token)
        input_token = next_token
        if (next_token == eos_id).all():
            break
    return torch.cat(output_ids, dim=1)
```

贪心解码在每一步都选概率最高的 token。它可能越走越偏：一旦选定一个 token，就无法收回。**束搜索（beam search）**则同时保留得分最高的 `k` 个部分序列，最后从完整序列中选出得分最高的那个。束宽 3-5 是标准配置。

### 第 5 步：实测瓶颈

在一个玩具复制任务上训练模型：源序列 `[a, b, c, d, e]`，目标序列 `[a, b, c, d, e]`。逐步增加序列长度，观察准确率。

```
seq_len=5   copy accuracy: 98%
seq_len=10  copy accuracy: 91%
seq_len=20  copy accuracy: 62%
seq_len=40  copy accuracy: 23%
```

单个 GRU 隐藏状态无法无损地记住 40 个 token 的输入。信息明明存在于编码器的每一步里，但解码器只看得到最后一个状态。注意力机制正是直接解决这个问题。

## 生产实践

PyTorch 提供了 `nn.Transformer` 和基于 `nn.LSTM` 的 seq2seq 模板。Hugging Face 的 `transformers` 库则提供了在数十亿 token 上训练好的完整编码器-解码器模型（BART、T5、mBART、NLLB）。

```python
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

tok = AutoTokenizer.from_pretrained("facebook/bart-base")
model = AutoModelForSeq2SeqLM.from_pretrained("facebook/bart-base")

src = tok("Translate this to French: Hello, how are you?", return_tensors="pt")
out = model.generate(**src, max_new_tokens=50, num_beams=4)
print(tok.decode(out[0], skip_special_tokens=True))
```

现代编码器-解码器模型用 Transformer 取代了 RNN。但顶层结构（编码器、解码器、逐 token 生成）与 2014 年的 seq2seq 论文完全一致，不同的只是每个模块内部的机制。

### 什么时候还会用到基于 RNN 的 seq2seq

新项目几乎永远不会。少数例外：

- 流式翻译：需要逐 token 消费输入，且内存占用有上限。
- 端侧文本生成：Transformer 的内存开销难以承受。
- 教学。理解编码器-解码器瓶颈，是理解 Transformer 为什么胜出的最快路径。

### 曝光偏差及其缓解手段

- **计划采样（Scheduled sampling）。** 训练时逐渐降低教师强制比例，让模型学会从自己的错误中恢复。
- **最小风险训练（Minimum risk training）。** 用句子级 BLEU 分数代替 token 级交叉熵作为训练目标。更接近你真正想要的东西。
- **强化学习微调。** 用某个指标作为奖励来训练序列生成器。现代 LLM 的 RLHF 用的就是这一思路。

这三种方法在基于 Transformer 的生成中同样适用。

## 交付产物

保存为 `outputs/prompt-seq2seq-design.md`：

```markdown
---
name: seq2seq-design
description: Design a sequence-to-sequence pipeline for a given task.
phase: 5
lesson: 09
---

Given a task (translation, summarization, paraphrase, question rewrite), output:

1. Architecture. Pretrained transformer encoder-decoder (BART, T5, mBART, NLLB) is the default. RNN-based seq2seq only for specific constraints.
2. Starting checkpoint. Name it (`facebook/bart-base`, `google/flan-t5-base`, `facebook/nllb-200-distilled-600M`). Match the checkpoint to task and language coverage.
3. Decoding strategy. Greedy for deterministic output, beam search (width 4-5) for quality, sampling with temperature for diversity. One sentence justification.
4. One failure mode to verify before shipping. Exposure bias manifests as generation drift on longer outputs; sample 20 outputs at the 90th-percentile length and eyeball.

Refuse to recommend training a seq2seq from scratch for under a million parallel examples. Flag any pipeline that uses greedy decoding for user-facing content as fragile (greedy repeats and loops).
```

## 练习

1. **简单。** 实现玩具复制任务。在目标等于源的输入-输出对上训练一个 GRU seq2seq。测量长度为 5、10、20 时的准确率，复现瓶颈现象。
2. **中等。** 添加束宽为 3 的束搜索解码。在一个小型平行语料上对比束搜索和贪心解码的 BLEU。记录束搜索在哪些地方获胜（通常是末尾的 token），在哪些地方没有差别。
3. **困难。** 在一个 1 万对的复述数据集上微调 `facebook/bart-base`。在留出集输入上，对比微调模型和基础模型的束宽 4 输出。报告 BLEU，并挑选 10 个定性示例。

## 关键术语

| 术语 | 大家怎么说 | 实际含义 |
|------|-----------------|-----------------------|
| 编码器 | 输入 RNN | 读取源序列。产出每一步的隐藏状态和最终的上下文向量。 |
| 解码器 | 输出 RNN | 由上下文向量初始化。逐个生成目标 token。 |
| 上下文向量 | 那个摘要 | 编码器最后的隐藏状态。大小固定。是注意力要解决的瓶颈。 |
| 教师强制 | 用真实 token | 训练时把上一位置的真实 token 喂给解码器。稳定学习过程。 |
| 曝光偏差 | 训练/测试差距 | 模型在真实 token 上训练，从未练习过如何从自己的错误中恢复。 |
| 束搜索 | 更好的解码 | 每一步保留 top-k 个部分序列，而不是贪心地一步定死。 |

## 延伸阅读

- [Sutskever, Vinyals, Le (2014). Sequence to Sequence Learning with Neural Networks](https://arxiv.org/abs/1409.3215) —— seq2seq 原始论文。只有四页。
- [Cho et al. (2014). Learning Phrase Representations using RNN Encoder-Decoder for Statistical Machine Translation](https://arxiv.org/abs/1406.1078) —— 提出了 GRU 和编码器-解码器框架。
- [Bahdanau, Cho, Bengio (2014). Neural Machine Translation by Jointly Learning to Align and Translate](https://arxiv.org/abs/1409.0473) —— 注意力论文。学完本课后立刻阅读。
- [PyTorch NLP from Scratch tutorial](https://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html) —— 可动手实现的 seq2seq + 注意力代码。
