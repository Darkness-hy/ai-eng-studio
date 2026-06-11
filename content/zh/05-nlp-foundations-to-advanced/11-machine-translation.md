# 机器翻译

> 翻译这个任务为 NLP 研究买了三十年的单，而且至今还在持续付账。

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 5 · 10 (Attention Mechanism), Phase 5 · 04 (GloVe, FastText, Subword)
**Time:** ~75 minutes

## 问题背景

模型读入一种语言的句子，输出另一种语言的句子。长度会变，语序会变。有的源语言词对应多个目标语言词，反之亦然。习语更是拒绝一一对应："I miss you" 在法语里是 "tu me manques"——字面意思是"你对我来说是缺失的"。任何词级别的对齐都扛不住这种情况。

机器翻译（Machine Translation）正是逼着 NLP 发明出编码器-解码器、注意力、Transformer，乃至整个 LLM 范式的那个任务。每一次进步的出现，都是因为翻译质量可以量化，而人与机器之间的差距又始终顽固存在。

这节课跳过历史课，直接教 2026 年的实用流水线：预训练多语言编码器-解码器（NLLB-200 或 mBART）、子词分词、束搜索（beam search）、BLEU 与 chrF 评估，以及那几种至今仍会悄无声息混进生产环境的失败模式。

## 核心概念

![MT pipeline: tokenize → encode → decode with attention → detokenize](../assets/mt-pipeline.svg)

现代机器翻译是在平行语料上训练的 Transformer 编码器-解码器。编码器按照源语言的分词方式读入源文本。解码器通过交叉注意力（第 10 课）利用编码器的输出，逐个子词地生成目标文本。解码使用束搜索来避开贪心解码的陷阱。输出经过反分词（detokenize）、还原大小写（detruecase），再与参考译文对照打分。

三个工程选择决定了真实场景下的翻译质量。

- **分词器。** 在混合语言语料上训练的 SentencePiece BPE。跨语言共享词表正是 NLLB 实现零样本语言对的关键。
- **模型规模。** NLLB-200 蒸馏版 600M 笔记本电脑就能跑。NLLB-200 3.3B 是官方发布的生产默认配置。54.5B 则是研究天花板。
- **解码。** 通用内容用束宽 4-5。用长度惩罚（length penalty）避免输出过短。需要术语一致性时用受限解码（constrained decoding）。

```figure
seq2seq-alignment
```

## 从零实现

### 第 1 步：调用一个预训练 MT 模型

```python
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

model_id = "facebook/nllb-200-distilled-600M"
tok = AutoTokenizer.from_pretrained(model_id, src_lang="eng_Latn")
model = AutoModelForSeq2SeqLM.from_pretrained(model_id)

src = "The cats are running."
inputs = tok(src, return_tensors="pt")

out = model.generate(
    **inputs,
    forced_bos_token_id=tok.convert_tokens_to_ids("fra_Latn"),
    num_beams=5,
    length_penalty=1.0,
    max_new_tokens=64,
)
print(tok.batch_decode(out, skip_special_tokens=True)[0])
```

```text
Les chats courent.
```

这里有三个关键点。`src_lang` 告诉分词器该使用哪种文字和切分方式。`forced_bos_token_id` 告诉解码器要生成哪种语言。这两个都是 NLLB 特有的技巧；mBART 和 M2M-100 各有自己的约定，不能互换使用。

### 第 2 步：BLEU 与 chrF

BLEU 衡量输出与参考译文之间的 n-gram 重叠度。取 1-4 共四种 n-gram 长度，对各精确率取几何平均，并对过短输出施加简短惩罚（brevity penalty）。分数落在 [0, 100] 区间。它使用广泛，但解读起来很折磨人：30 BLEU 算"可用"，40 算"良好"，50 算"出色"，差距小于 1 BLEU 的就是噪声。

chrF 衡量字符级 F 分数。对形态丰富的语言更敏感——这类语言里 BLEU 会漏算很多匹配。通常与 BLEU 一起报告。

```python
import sacrebleu

hypotheses = ["Les chats courent."]
references = [["Les chats courent."]]

bleu = sacrebleu.corpus_bleu(hypotheses, references)
chrf = sacrebleu.corpus_chrf(hypotheses, references)
print(f"BLEU: {bleu.score:.1f}  chrF: {chrf.score:.1f}")
```

永远使用 `sacrebleu`。它统一了分词规范，使分数在不同论文之间可比。自己手搓 BLEU 计算，正是误导性基准结果的来源。

### 三层评估体系（2026）

现代机器翻译评估使用三个互补的指标家族。上线时至少带上其中两个。

- **启发式指标**（BLEU、chrF）。快、基于参考译文、可解释，但对同义改写不敏感。用于与历史结果对比和回归检测。
- **学习型指标**（COMET、BLEURT、BERTScore）。在人工评判数据上训练的神经模型；比较译文与源文本和参考译文的语义相似度。COMET 自 2023 年以来与机器翻译研究的关联度最高，是 2026 年对质量有要求场景的生产默认选择。
- **LLM 评审（LLM-as-judge）**（无需参考译文）。用提示词让大模型从流畅度、忠实度、语气、文化适配等维度给译文打分。当评分细则设计得当时，GPT-4 作为评审与人工判断的一致率约 80%。适用于不存在参考译文的开放式内容。

2026 年的实用组合：`sacrebleu` 算 BLEU 和 chrF，`unbabel-comet` 算 COMET，再用一个带提示词的 LLM 输出最终面向人的质量信号。每个指标在用于生产数据之前，都要先用 50-100 条人工标注样本做校准。

无参考指标（COMET-QE、BLEURT-QE、LLM-as-judge）让你在没有参考译文的情况下也能评估翻译质量，这对那些根本不存在参考译文的长尾语言对至关重要。

### 第 3 步：生产环境里会坏掉的地方

上面这条可用的流水线会在 80% 的情况下流畅地完成翻译，而在剩下 20% 的情况下悄无声息地失败。已有命名的失败模式如下：

- **幻觉（Hallucination）。** 模型编造源文本中不存在的内容。在陌生领域词汇上很常见。症状：输出很流畅，却声称了源文本没有陈述的事实。缓解手段：对领域术语使用受限解码，对受监管内容做人工审核，监控输出长度远超输入的情况。
- **目标语言跑偏（Off-target generation）。** 模型翻译成了错误的语言。NLLB 在罕见语言对上出人意料地容易犯这个错。缓解手段：核对 `forced_bos_token_id`，并且解码后必须用语言识别模型检查输出。
- **术语漂移。** "Sign up" 在文档 1 里译成 "s'inscrire"，到文档 2 里变成 "créer un compte"。对 UI 文案和面向用户的字符串来说，一致性比绝对质量更重要。缓解手段：基于术语表的受限解码，或译后字典替换。
- **正式程度错配。** 法语的 "tu" 与 "vous"、日语的敬语层级。模型会选训练数据里更常见的那种形式。对面向客户的内容，这通常是错的。缓解手段：如果模型支持，在提示前缀里加正式程度标记，或者在纯正式语料上微调一个小模型。
- **短输入引发的长度爆炸。** 非常短的输入句子常常产出过长的译文，因为长度惩罚在源端少于约 5 个 token 时会断崖式失效。缓解手段：设置与源长度成比例的硬性最大长度上限。

### 第 4 步：面向领域的微调

预训练模型是通才。法律、医疗或游戏对白翻译，在领域平行语料上微调能带来可量化的提升。配方并不花哨：

```python
from transformers import Trainer, TrainingArguments
from datasets import Dataset

pairs = [
    {"src": "The defendant pleaded guilty.", "tgt": "L'accusé a plaidé coupable."},
]

ds = Dataset.from_list(pairs)


def preprocess(ex):
    return tok(
        ex["src"],
        text_target=ex["tgt"],
        truncation=True,
        max_length=128,
        padding="max_length",
    )


ds = ds.map(preprocess, remove_columns=["src", "tgt"])

args = TrainingArguments(output_dir="out", per_device_train_batch_size=4, num_train_epochs=3, learning_rate=3e-5)
Trainer(model=model, args=args, train_dataset=ds).train()
```

几千条高质量平行样本胜过几十万条嘈杂的网络爬取数据。训练数据质量是生产环境中最大的单一杠杆。

## 生产实践

2026 年机器翻译的生产技术选型：

| 使用场景 | 推荐起点 |
|---------|---------------------------|
| 任意语言互译，200 种语言 | `facebook/nllb-200-distilled-600M`（笔记本）或 `nllb-200-3.3B`（生产） |
| 以英语为中心、高质量、50 种语言 | `facebook/mbart-large-50-many-to-many-mmt` |
| 短任务、低成本推理、英语-法/德/西 | Helsinki-NLP / Marian 系列模型 |
| 延迟敏感的浏览器端 | ONNX 量化的 Marian（约 50 MB） |
| 追求最高质量、愿意付费 | GPT-4 / Claude / Gemini 配合翻译提示词 |

截至 2026 年，LLM 已在多个语言对上超越了专用机器翻译模型，尤其是习语内容和长上下文场景。代价是按 token 计费的成本和延迟。当上下文长度、风格一致性，或通过提示词做领域适配比吞吐量更重要时，选 LLM。

## 交付产物

保存为 `outputs/skill-mt-evaluator.md`：

```markdown
---
name: mt-evaluator
description: Evaluate a machine translation output for shipping.
version: 1.0.0
phase: 5
lesson: 11
tags: [nlp, translation, evaluation]
---

Given a source text and a candidate translation, output:

1. Automatic score estimate. BLEU and chrF ranges you would expect. State whether a reference is available.
2. Five-point human-verifiable check list: (a) content preservation (no hallucinations), (b) correct language, (c) register / formality match, (d) terminology consistency with glossary if provided, (e) no truncation or length explosion.
3. One domain-specific issue to probe. E.g., for legal: named entities and statute citations. For medical: drug names and dosages. For UI: placeholder variables `{name}`.
4. Confidence flag. "Ship" / "Ship with review" / "Do not ship". Tie to the severity of issues found in step 2.

Refuse to ship a translation without a language-ID check on output. Refuse to evaluate without a reference unless the user explicitly opts in to reference-free scoring (COMET-QE, BLEURT-QE). Flag any content over 1000 tokens as likely needing chunked translation.
```

## 练习

1. **简单。** 用 `nllb-200-distilled-600M` 把一段 5 句话的英文翻译成法语，再译回英文。衡量往返翻译与原文的接近程度。你应该会看到语义得到保留，但用词出现漂移。
2. **中等。** 用 `fasttext lid.176` 或 `langdetect` 对翻译输出实现语言识别检查。将其集成进 MT 调用流程，让目标语言跑偏的输出在返回前就被拦截。
3. **困难。** 在自选的 5,000 对领域语料上微调 `nllb-200-distilled-600M`。在留出集上对比微调前后的 BLEU。报告哪类句子有提升、哪类出现了退化。

## 关键术语

| 术语 | 大家怎么说 | 实际含义 |
|------|-----------------|-----------------------|
| BLEU | 翻译评分 | 带简短惩罚的 n-gram 精确率。范围 [0, 100]。 |
| chrF | 字符 F 分数 | 字符级 F 分数。对形态丰富的语言更敏感。 |
| NMT | 神经机器翻译 | 在平行语料上训练的 Transformer 编码器-解码器。2017 年之后的默认方案。 |
| NLLB | No Language Left Behind | Meta 的 200 语言机器翻译模型家族。 |
| 受限解码 | 可控输出 | 强制特定 token 或 n-gram 在输出中出现/不出现。 |
| 幻觉 | 编造内容 | 模型输出中没有源文本依据的部分。 |

## 延伸阅读

- [Costa-jussà et al. (2022). No Language Left Behind: Scaling Human-Centered Machine Translation](https://arxiv.org/abs/2207.04672) —— NLLB 论文。
- [Post (2018). A Call for Clarity in Reporting BLEU Scores](https://aclanthology.org/W18-6319/) —— 为什么 `sacrebleu` 是报告 BLEU 的唯一正确方式。
- [Popović (2015). chrF: character n-gram F-score for automatic MT evaluation](https://aclanthology.org/W15-3049/) —— chrF 论文。
- [Hugging Face MT guide](https://huggingface.co/docs/transformers/tasks/translation) —— 实操微调教程。
