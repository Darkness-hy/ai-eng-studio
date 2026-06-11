# T5 与 BART —— 编码器-解码器模型

> 编码器负责理解，解码器负责生成。把两者重新组合起来，就得到了一个专为「输入 → 输出」任务而生的模型：翻译、摘要、改写、转录。

**Type:** Learn
**Languages:** Python
**Prerequisites:** Phase 7 · 05 (Full Transformer), Phase 7 · 06 (BERT), Phase 7 · 07 (GPT)
**Time:** ~45 minutes

## 问题背景

仅解码器的 GPT 和仅编码器的 BERT 各自针对不同目标对 2017 年的原始架构做了删减。但许多任务天然就是输入-输出形式的：

- 翻译：英语 → 法语。
- 摘要：5,000 个 token 的文章 → 200 个 token 的摘要。
- 语音识别：音频 token → 文本 token。
- 结构化抽取：自然语言文本 → JSON。

对这类任务而言，编码器-解码器（encoder-decoder）是最贴合的架构。编码器为源序列生成一个稠密表示；解码器在生成输出的每一步都对这个表示做交叉注意力（cross-attention）。训练在输出侧采用错位一格（shift-by-one）的方式，损失函数与 GPT 相同，只是额外以编码器输出为条件。

两篇论文奠定了现代的标准做法：

1. **T5**（Raffel et al. 2019）。"Text-to-Text Transfer Transformer"。把所有 NLP 任务都重新表述为「文本进、文本出」。统一的架构、统一的词表、统一的损失。预训练任务是掩码片段预测（在输入中破坏若干片段，在输出中将其解码出来）。
2. **BART**（Lewis et al. 2019）。"Bidirectional and Auto-Regressive Transformer"。去噪自编码器：用多种方式破坏输入（打乱、掩码、删除、旋转），让解码器重建原文。

到了 2026 年，编码器-解码器架构在输入结构重要的场景里依然活跃：

- Whisper（语音 → 文本）。
- Google 的翻译技术栈。
- 一些具有明确「上下文 + 编辑」结构的代码补全 / 代码修复模型。
- 面向结构化推理任务的 Flan-T5 及其变体。

仅解码器架构抢走了聚光灯，但编码器-解码器从未离场。

## 核心概念

![Encoder-decoder with cross-attention](../assets/encoder-decoder.svg)

### 前向计算流程

```
source tokens ─▶ encoder ─▶ (N_src, d_model)  ──┐
                                                 │
target tokens ─▶ decoder block                   │
                 ├─▶ masked self-attention       │
                 ├─▶ cross-attention ◀───────────┘
                 └─▶ FFN
                ↓
              next-token logits
```

关键在于：编码器对每个输入只运行一次。解码器自回归地运行，但每一步都对*同一份*编码器输出做交叉注意力。对长输入而言，缓存编码器输出是一项零成本的加速手段。

### T5 预训练 —— 片段破坏

随机挑选输入中的若干片段（平均长度 3 个 token，总量占 15%），将每个片段替换为一个唯一的哨兵 token（sentinel token）：`<extra_id_0>`、`<extra_id_1>` 等等。解码器只需输出被破坏的片段及其哨兵前缀：

```
source: The quick <extra_id_0> fox jumps <extra_id_1> dog
target: <extra_id_0> brown <extra_id_1> over the lazy
```

比预测整个序列的训练信号更便宜。在 T5 论文的消融实验中，它与 MLM（BERT）和 prefix-LM（UniLM）效果相当。

### BART 预训练 —— 多噪声去噪

BART 尝试了五种加噪函数：

1. Token 掩码。
2. Token 删除。
3. 文本填充（text infilling，掩盖一个片段，由解码器推断出正确长度并补全）。
4. 句子重排。
5. 文档旋转。

组合「文本填充 + 句子重排」得到了最好的下游任务结果。解码器始终重建完整原文。BART 的输出是整个序列，而不只是被破坏的片段——因此预训练的计算开销高于 T5。

### 推理

与 GPT 相同的自回归生成。贪心解码 / 束搜索 / top-p 采样均适用。翻译和摘要任务的标准做法是束搜索（beam search，宽度 4–5），因为这类任务的输出分布比对话更窄。

### 2026 年如何选型

| 任务 | 用编码器-解码器吗？ | 原因 |
|------|------------------|-----|
| 翻译 | 通常用 | 源序列清晰；输出分布固定；束搜索效果好 |
| 语音转文本 | 用（Whisper） | 输入与输出的模态不同；编码器负责塑造音频特征 |
| 对话 / 推理 | 不用，选仅解码器 | 没有持久的「输入」——对话本身就是序列 |
| 代码补全 | 通常不用 | 长上下文的仅解码器模型占优；Qwen 2.5 Coder 等代码模型都是仅解码器 |
| 摘要 | 两者皆可 | BART、PEGASUS 曾击败早期的仅解码器基线；现代仅解码器 LLM 已追平 |
| 结构化抽取 | 两者皆可 | T5 很干净，因为「文本 → 文本」能涵盖任意输出格式 |

约 2022 年以来的趋势：仅解码器架构接管了编码器-解码器曾经主导的任务，原因有三：(a) 指令微调后的仅解码器 LLM 可以靠提示泛化到任何任务；(b) 一种架构比两种更容易扩展规模；(c) RLHF 默认面向解码器。编码器-解码器在输入模态不同（语音、图像）或束搜索质量重要的场景中守住了阵地。

## 从零实现

参见 `code/main.py`。我们在一个玩具语料上实现 T5 风格的片段破坏——这是本课最有价值的一个环节，因为此后所有编码器-解码器预训练方案中都会出现它。

### 第 1 步：片段破坏

```python
def corrupt_spans(tokens, mask_rate=0.15, mean_span=3.0, rng=None):
    """Pick spans summing to ~mask_rate of tokens. Return (corrupted_input, target)."""
    n = len(tokens)
    n_mask = max(1, int(n * mask_rate))
    n_spans = max(1, int(round(n_mask / mean_span)))
    ...
```

目标格式遵循 T5 约定：`<sent0> span0 <sent1> span1 ...`。被破坏的输入则把未变动的 token 与片段位置处的哨兵 token 交错排列。

### 第 2 步：验证可逆性

给定破坏后的输入和目标序列，重建原始句子。如果你的破坏过程可逆，前向计算就是良定义的。这是一道健全性检查——真实训练从不会做这一步，但这个测试成本极低，而且能抓出片段索引记录里的差一（off-by-one）错误。

### 第 3 步：BART 加噪

五个函数：`token_mask`、`token_delete`、`text_infill`、`sentence_permute`、`document_rotate`。组合其中两个并展示结果。

## 生产实践

HuggingFace 参考代码：

```python
from transformers import T5ForConditionalGeneration, T5Tokenizer
tok = T5Tokenizer.from_pretrained("google/flan-t5-base")
model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-base")

inputs = tok("translate English to French: Attention is all you need.", return_tensors="pt")
out = model.generate(**inputs, max_new_tokens=32)
print(tok.decode(out[0], skip_special_tokens=True))
```

T5 的诀窍在于：任务名直接写进输入文本。同一个模型能处理几十种任务，因为每个任务都是「文本进、文本出」。到 2026 年，这一模式已被指令微调的仅解码器模型推广开来，但最先将其系统化的是 T5。

## 交付产物

参见 `outputs/skill-seq2seq-picker.md`。该技能根据输入-输出结构、延迟和质量目标，为新任务在编码器-解码器与仅解码器之间做出选择。

## 练习

1. **简单。** 运行 `code/main.py`，对一个 30 token 的句子做片段破坏，验证把源序列中的非哨兵 token 与解码出的目标片段拼接后能还原原句。
2. **中等。** 实现 BART 的 `text_infill` 噪声：把随机片段替换为单个 `<mask>` token，解码器必须推断出正确的片段长度和内容。展示一个例子。
3. **困难。** 在一个小型「英语 → 儿童黑话（pig-Latin）」语料（200 对）上微调 `flan-t5-small`，在留出的 50 对测试集上计算 BLEU。再用相同数据、相同算力微调 `Llama-3.2-1B`，对比结果。

## 关键术语

| 术语 | 大家怎么说 | 实际含义 |
|------|-----------------|-----------------------|
| 编码器-解码器 | 「seq2seq Transformer」 | 两个堆叠：处理输入的双向编码器，加上带交叉注意力的因果解码器负责输出。 |
| 交叉注意力 | 「源序列和目标序列对话的地方」 | 解码器的 Q × 编码器的 K/V。这是编码器信息进入解码器的唯一通道。 |
| 片段破坏 | 「T5 的预训练招数」 | 把随机片段替换成哨兵 token；解码器输出这些片段。 |
| 去噪目标 | 「BART 的玩法」 | 对输入施加噪声函数，训练解码器重建干净序列。 |
| 哨兵 token | 「`<extra_id_N>` 占位符」 | 特殊 token，在源序列中标记被破坏的片段，并在目标序列中再次标记它们。 |
| Flan | 「指令微调版 T5」 | 在 1,800 多个任务上微调的 T5；让编码器-解码器在指令跟随上具备了竞争力。 |
| 束搜索 | 「一种解码策略」 | 每一步保留 top-k 个部分序列；翻译/摘要任务的标准做法。 |
| 教师强制 | 「训练时的输入方式」 | 训练时把真实的上一个输出 token（而非采样得到的）喂给解码器。 |

## 延伸阅读

- [Raffel et al. (2019). Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer](https://arxiv.org/abs/1910.10683) —— T5。
- [Lewis et al. (2019). BART: Denoising Sequence-to-Sequence Pre-training for Natural Language Generation, Translation, and Comprehension](https://arxiv.org/abs/1910.13461) —— BART。
- [Chung et al. (2022). Scaling Instruction-Finetuned Language Models](https://arxiv.org/abs/2210.11416) —— Flan-T5。
- [Radford et al. (2022). Robust Speech Recognition via Large-Scale Weak Supervision](https://arxiv.org/abs/2212.04356) —— Whisper，2026 年最具代表性的编码器-解码器模型。
- [HuggingFace `modeling_t5.py`](https://github.com/huggingface/transformers/blob/main/src/transformers/models/t5/modeling_t5.py) —— 参考实现。
