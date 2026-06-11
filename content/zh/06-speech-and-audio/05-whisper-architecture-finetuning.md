# Whisper——架构与微调

> Whisper 是一个以 30 秒为窗口的 Transformer 编码器-解码器模型，在 68 万小时的多语言弱监督音频-文本对上训练而成。一套架构、多种任务，在 99 种语言上都很稳健。它是 2026 年的参考级 ASR。

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 6 · 04 (ASR), Phase 5 · 10 (Attention), Phase 7 · 05 (Full Transformer)
**Time:** ~75 minutes

## 问题背景

Whisper 由 OpenAI 于 2022 年 9 月发布，是第一个真正「开箱即用」的 ASR 模型：丢进音频，得到文本，支持 99 种语言，抗噪声，还能在笔记本电脑上运行。到 2024 年，OpenAI 又发布了 Large-v3 和 Turbo 变体；到 2026 年，从播客转写、语音助手到 YouTube 字幕，Whisper 已是几乎一切场景的默认基线。

但你不能永远把 Whisper 当黑盒用。领域偏移（domain shift）会让它崩溃——技术行话、说话人口音、专有名词、短音频、静音段。你需要弄清楚：

1. 它内部到底是什么。
2. 如何正确地给它喂入分块、流式或长音频。
3. 什么时候需要微调，以及怎么微调。

## 核心概念

![Whisper encoder-decoder, tasks, chunked inference, fine-tune](../assets/whisper.svg)

**架构。** 标准的 Transformer 编码器-解码器。

- 输入：30 秒的对数梅尔频谱图（log-mel spectrogram），80 个梅尔频带，10 ms 帧移 → 3000 帧。更短的音频补零填充，更长的音频切块处理。
- 编码器：卷积下采样（步长 2）+ `N` 个 Transformer 块。Large-v3 为：32 层、1280 维、20 个注意力头。
- 解码器：`N` 个 Transformer 块，包含因果自注意力 + 对编码器输出的交叉注意力。规模与编码器相同。
- 输出：基于 51,865 个 token 词表的 BPE token。

Large-v3 有 15.5 亿参数。Turbo 将解码器从 32 层砍到 4 层，延迟降低 8 倍，WER 损失不到 1%。

**提示词格式。** Whisper 是一个多任务模型，通过解码器提示中的特殊 token 来控制行为：

```
<|startoftranscript|><|en|><|transcribe|><|notimestamps|> Hello world.<|endoftext|>
```

- `<|en|>` —— 语言标签；决定走翻译还是转写的行为路径。
- `<|transcribe|>` 或 `<|translate|>` —— 将任意语言的输入翻译为英文输出，或逐字转写。
- `<|notimestamps|>` —— 跳过词级时间戳（更快）。

正是这套提示机制让一个模型能做多种任务。把 `<|en|>` 换成 `<|fr|>`，它就转写法语。

**30 秒窗口。** 一切都钉死在 30 秒上。更长的音频需要切块，更短的音频要填充。窗口原生不支持流式处理——这正是 WhisperX、Whisper-Streaming 和 faster-whisper 存在的原因。

**对数梅尔归一化。** `(log_mel - mean) / std`，其中统计量来自 Whisper 自己的训练语料。你*必须*使用 Whisper 自带的预处理（`whisper.audio.log_mel_spectrogram`），而不是 `librosa.feature.melspectrogram`。

### 2026 年的各个变体

| 变体 | 参数量 | 延迟（A100） | WER（LibriSpeech-clean） |
|---------|--------|----------------|------------------------|
| Tiny | 39M | 1 倍实时 | 5.4% |
| Base | 74M | 1× | 4.1% |
| Small | 244M | 1× | 3.0% |
| Medium | 769M | 1× | 2.7% |
| Large-v3 | 1.55B | 2× | 1.8% |
| Large-v3-turbo | 809M | 8× | 1.58% |
| Whisper-Streaming（2024） | 1.55B | 流式 | 2.0% |

### 微调

2026 年的标准工作流：

1. 收集 10–100 小时带对齐转写文本的目标领域音频。
2. 运行 `transformers.Seq2SeqTrainer`，配合 `generate_with_loss` 回调。
3. 参数高效方案：在注意力层的 `q_proj`、`k_proj`、`v_proj` 上挂 LoRA，可将 GPU 显存占用降低 4 倍，WER 代价小于 0.3。
4. 如果数据不足 10 小时，冻结编码器，只调解码器。
5. 使用 Whisper 自带的分词器和提示词格式；绝不要换分词器。

社区结果：在 20 小时医疗口述数据上微调 Medium，可将医学词汇上的 WER 从 12% 降到 4.5%。在 4 小时冰岛语数据上微调 Turbo，可将 WER 从 18% 降到 6%。

## 从零实现

### 第 1 步：开箱直接运行 Whisper

```python
import whisper
model = whisper.load_model("large-v3-turbo")
result = model.transcribe(
    "clip.wav",
    language="en",
    task="transcribe",
    temperature=0.0,
    condition_on_previous_text=False,  # prevents runaway repetition
)
print(result["text"])
for seg in result["segments"]:
    print(f"[{seg['start']:.2f}–{seg['end']:.2f}] {seg['text']}")
```

几个你应当始终显式覆盖的默认参数：`temperature=0.0`（采样默认走 0.0 → 0.2 → 0.4 …的回退链）、`condition_on_previous_text=False`（防止级联幻觉问题）、以及 `no_speech_threshold=0.6`（静音检测）。

### 第 2 步：分块处理长音频

```python
# whisperx is the 2026 reference for long-form with word-level timestamps
import whisperx
model = whisperx.load_model("large-v3-turbo", device="cuda", compute_type="float16")
segments = model.transcribe("1hour.mp3", batch_size=16, chunk_size=30)
```

WhisperX 在 Whisper 之上加了：(1) Silero VAD 门控，(2) 基于 wav2vec 2.0 的词级对齐，(3) 基于 `pyannote.audio` 的说话人分离（diarization）。它是 2026 年生产级转写的主力工具。

### 第 3 步：用 LoRA 微调

```python
from transformers import WhisperForConditionalGeneration, WhisperProcessor
from peft import LoraConfig, get_peft_model

model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-large-v3-turbo")
lora = LoraConfig(
    r=16, lora_alpha=32, target_modules=["q_proj", "v_proj"],
    lora_dropout=0.1, bias="none", task_type="SEQ_2_SEQ_LM",
)
model = get_peft_model(model, lora)
# model.print_trainable_parameters()  -> ~3M trainable / 809M total
```

之后就是标准的 Trainer 训练循环。每 1000 步保存一次检查点。在留出集上用 WER 评估。

### 第 4 步：查看每一层学到了什么

```python
# Grab cross-attention weights during decode to see what the decoder attends to.
with torch.inference_mode():
    out = model.generate(
        input_features=features,
        return_dict_in_generate=True,
        output_attentions=True,
    )
# out.cross_attentions: layer × head × step × src_len
```

用热力图可视化——你会看到随着解码步扫过编码器帧而呈现的对角线对齐模式。这条对角线就是 Whisper 词级时间戳的来源。

## 生产实践

2026 年的技术选型：

| 场景 | 选择 |
|-----------|------|
| 通用英语、离线 | 经 `whisperx` 跑 Large-v3-turbo |
| 移动 / 边缘端 | 量化（int8）的 Whisper-Tiny 或 Moonshine |
| 多语言长音频 | 经 `whisperx` 跑 Large-v3 + 说话人分离 |
| 低资源语言 | 用 LoRA 微调 Medium 或 Turbo |
| 流式（2 秒延迟） | Whisper-Streaming 或 Parakeet-TDT |
| 词级时间戳 | WhisperX（基于 wav2vec 2.0 的强制对齐） |

`faster-whisper`（CTranslate2 后端）是 2026 年最快的 CPU+GPU 推理运行时——比原版快 4 倍，输出完全一致。

## 2026 年仍在线上出没的常见陷阱

- **静音段上的幻觉文本。** Whisper 的训练数据来自字幕，包含「Thanks for watching!」「Subscribe!」、歌词等。调用前务必用 VAD 做门控。
- **`condition_on_previous_text` 级联。** 一次幻觉会污染后续所有窗口。除非你需要跨块的流畅性，否则设为 `False`。
- **短音频填充。** 一段 2 秒的音频被填充到 30 秒后，可能在尾部静音中产生幻觉。使用 `pad=False` 或 VAD 门控。
- **错误的梅尔统计量。** 用 librosa 的梅尔频谱代替 Whisper 自带的，输出会接近随机。请使用 `whisper.audio.log_mel_spectrogram`。

## 交付产物

保存为 `outputs/skill-whisper-tuner.md`。为给定领域设计一套 Whisper 微调或推理流水线。

## 练习

1. **简单。** 运行 `code/main.py`。它会对一个 Whisper 风格的提示词做分词，计算解码后的形状预算，并打印一段 10 分钟音频的分块调度方案。
2. **中等。** 安装 `faster-whisper`，转写一段 10 分钟的播客，与人工转写文本对比 WER。比较 `language="auto"` 与强制 `language="en"` 的差异。
3. **困难。** 使用 HF `datasets`，挑一种 Whisper 表现不佳的语言（如乌尔都语），在 2 小时数据上用 LoRA 微调 Medium 训练 2 个 epoch，并报告 WER 变化。

## 关键术语

| 术语 | 人们常说 | 实际含义 |
|------|-----------------|-----------------------|
| 30 秒窗口 | Whisper 的限制 | 硬性输入上限；更长的音频要切块。 |
| SOT | 转写起始符 | `<\|startoftranscript\|>` 启动解码器提示。 |
| 时间戳 token | 时间对齐 | 每 0.02 秒一个偏移量，都是 51k 词表中的特殊 token。 |
| Turbo | 快速变体 | 4 层解码器，快 8 倍，WER 退化小于 1%。 |
| WhisperX | 长音频封装层 | VAD + Whisper + wav2vec 对齐 + 说话人分离。 |
| LoRA 微调 | 高效调优 | 在注意力层上加低秩适配器；只训练约 0.3% 的参数。 |
| 幻觉 | 静默失败 | Whisper 会从噪声/静音中生成流畅的英文。 |

## 延伸阅读

- [Radford et al. (2022). Whisper paper](https://arxiv.org/abs/2212.04356) —— 原始架构与训练方案。
- [OpenAI (2024). Whisper Large-v3-turbo release](https://github.com/openai/whisper/discussions/2363) —— 4 层解码器，8 倍加速。
- [Bain et al. (2023). WhisperX](https://arxiv.org/abs/2303.00747) —— 长音频、词级对齐、说话人分离。
- [Systran — faster-whisper repo](https://github.com/SYSTRAN/faster-whisper) —— CTranslate2 后端，快 4 倍。
- [HuggingFace — Whisper fine-tune tutorial](https://huggingface.co/blog/fine-tune-whisper) —— LoRA / 全参数微调的权威教程。
