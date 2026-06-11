# 音频 Transformer —— Whisper 架构

> 音频就是一张「频率随时间变化」的图像。Whisper 就是一个吃进梅尔频谱图、吐出文字的 ViT。

**Type:** Learn
**Languages:** Python
**Prerequisites:** Phase 7 · 05 (Full Transformer), Phase 7 · 08 (Encoder-Decoder), Phase 7 · 09 (ViT)
**Time:** ~45 minutes

## 问题背景

在 Whisper（OpenAI，Radford 等人，2022）出现之前，最先进的自动语音识别（ASR）方案是 wav2vec 2.0 和 HuBERT —— 自监督特征提取器加上一个微调的输出头。质量很高，但数据管线昂贵，且对领域变化十分脆弱。多语言语音识别需要按语系分别训练模型。

Whisper 押了三个注：

1. **什么都拿来训练。** 从互联网上抓取的 68 万小时弱标注音频，覆盖 97 种语言。没有干净的学术语料库，没有音素标注。
2. **多任务单模型。** 一个解码器通过任务 token 联合训练转写、翻译、语音活动检测、语种识别和时间戳预测。
3. **标准的编码器-解码器 Transformer。** 编码器输入对数梅尔频谱图，解码器自回归地生成文本 token。没有声码器，没有 CTC，没有 HMM。

结果是：Whisper large-v3 在各种口音、噪声环境，以及完全没有干净标注数据的语言上都表现稳健。到 2026 年，它是所有开源语音助手以及绝大多数商业语音助手的默认语音前端。

## 核心概念

![Whisper pipeline: audio → mel → encoder → decoder → text](../assets/whisper.svg)

### 第 1 步 —— 重采样 + 加窗

音频以 16 kHz 采样。裁剪或填充到 30 秒。计算对数梅尔频谱图（log-mel spectrogram）：80 个梅尔频带，10 ms 步长 → 约 3,000 帧 × 80 个特征。这就是 Whisper 看到的「输入图像」。

### 第 2 步 —— 卷积干（convolutional stem）

两层卷积核为 3、步幅为 2 的 Conv1D 层把 3,000 帧降到 1,500 帧。在几乎不增加参数量的情况下把序列长度减半。

### 第 3 步 —— 编码器

一个 24 层（large 版）Transformer 编码器，处理 1,500 个时间步。正弦位置编码、自注意力、GELU 前馈网络。输出 1,500 × 1,280 的隐藏状态。

### 第 4 步 —— 解码器

一个 24 层的 Transformer 解码器。它自回归地从一个 BPE 词表中生成 token，该词表是 GPT-2 词表的超集，外加少量音频专用的特殊 token。

### 第 5 步 —— 任务 token

解码器的提示词以控制 token 开头，告诉模型该做什么：

```
<|startoftranscript|>  <|en|>  <|transcribe|>  <|0.00|>
```

或

```
<|startoftranscript|>  <|fr|>  <|translate|>   <|0.00|>
```

模型就是按这套约定训练的。你通过前缀来控制任务。这相当于 2026 年的指令微调，只不过应用在语音上。

### 第 6 步 —— 输出

带对数概率阈值的束搜索（beam search，宽度为 5）。当 `<|notimestamps|>` token 缺席时，模型每 0.02 秒音频预测一次时间戳。

### Whisper 各规格

| 模型 | 参数量 | 层数 | d_model | 注意力头数 | 显存（fp16） |
|-------|--------|--------|---------|-------|-------------|
| Tiny | 39M | 4 | 384 | 6 | ~1 GB |
| Base | 74M | 6 | 512 | 8 | ~1 GB |
| Small | 244M | 12 | 768 | 12 | ~2 GB |
| Medium | 769M | 24 | 1024 | 16 | ~5 GB |
| Large | 1550M | 32 | 1280 | 20 | ~10 GB |
| Large-v3 | 1550M | 32 | 1280 | 20 | ~10 GB |
| Large-v3-turbo | 809M | 32 | 1280 | 20 | ~6 GB（4 层解码器） |

Large-v3-turbo（2024）把解码器从 32 层砍到 4 层。解码速度提升 8 倍，而 WER（词错误率）退化不到 1 个点。正是这一解码速度的突破，让 Whisper-turbo 成为 2026 年实时语音智能体的默认选择。

### Whisper 不做什么

- 不做说话人分离（diarization，即分辨谁在说话）。需要的话搭配 pyannote。
- 原生不支持实时流式处理 —— 30 秒窗口是固定的。现代封装库（`faster-whisper`、`WhisperX`）通过 VAD + 重叠窗口在外层补上流式能力。
- 没有超出 30 秒的长上下文，必须依赖外部分块。实践中效果依然不错，因为人类语音的转写很少需要长程上下文。

### 2026 年的格局

| 任务 | 模型 | 备注 |
|------|-------|-------|
| 英语 ASR | Whisper-turbo、Moonshine | Moonshine 在端侧快 4 倍 |
| 多语言 ASR | Whisper-large-v3 | 97 种语言 |
| 流式 ASR | faster-whisper + VAD | 可以做到 150 ms 延迟目标 |
| TTS | Piper、XTTS-v2、Kokoro | 编码器-解码器模式，结构和 Whisper 同款 |
| 音频 + 语言 | AudioLM、SeamlessM4T | 文本 token + 音频 token 共用一个 Transformer |

## 从零实现

参见 `code/main.py`。我们不训练 Whisper —— 我们构建对数梅尔频谱图管线 + 任务 token 提示词格式化器。这两部分才是你在生产中真正会碰到的。

### 第 1 步：合成音频

生成一段 1 秒、440 Hz 的正弦波，采样率 16 kHz。共 16,000 个采样点。

### 第 2 步：对数梅尔频谱图（简化版）

完整的梅尔频谱图需要 FFT。我们做一个简化版的分帧 + 逐帧能量计算，在不依赖 `librosa` 的前提下展示整条管线：

```python
def frame_signal(x, frame_size=400, hop=160):
    frames = []
    for start in range(0, len(x) - frame_size + 1, hop):
        frames.append(x[start:start + frame_size])
    return frames
```

帧长 = 25 ms，帧移 = 10 ms。与 Whisper 的加窗方式一致。出于教学目的，用逐帧能量代替梅尔频带。

### 第 3 步：填充到 30 秒

Whisper 永远按 30 秒分块处理。把频谱图填充（或裁剪）到 3,000 帧。

### 第 4 步：构建提示词 token

```python
def whisper_prompt(lang="en", task="transcribe", timestamps=True):
    tokens = ["<|startoftranscript|>", f"<|{lang}|>", f"<|{task}|>"]
    if not timestamps:
        tokens.append("<|notimestamps|>")
    return tokens
```

这就是任务控制的全部接口。一个 4 个 token 的前缀。

## 生产实践

```python
import whisper
model = whisper.load_model("large-v3-turbo")
result = model.transcribe("meeting.wav", language="en", task="transcribe")
print(result["text"])
print(result["segments"][0]["start"], result["segments"][0]["end"])
```

更快、兼容 OpenAI 接口的版本：

```python
from faster_whisper import WhisperModel
model = WhisperModel("large-v3-turbo", compute_type="int8_float16")
segments, info = model.transcribe("meeting.wav", vad_filter=True)
for s in segments:
    print(f"{s.start:.2f} - {s.end:.2f}: {s.text}")
```

**2026 年什么时候选 Whisper：**

- 用单个模型做多语言 ASR。
- 对嘈杂、多样化音频做稳健转写。
- 研究 / 原型阶段的 ASR —— 上手最快的起点。

**什么时候选别的：**

- 端侧超低延迟流式识别 —— 同等质量下 Moonshine 胜过 Whisper。
- 需要 <200 ms 的实时对话式 AI —— 用专门的流式 ASR。
- 说话人分离 —— Whisper 不做这件事；外挂 pyannote。

## 交付产物

参见 `outputs/skill-asr-configurator.md`。该 skill 为新的语音应用选择 ASR 模型、解码参数和预处理管线。

## 练习

1. **简单。** 运行 `code/main.py`。确认 16 kHz 采样、10 ms 帧移下，1 秒信号的帧数约为 100 帧；30 秒约为 3,000 帧。
2. **中等。** 用 `numpy.fft` 构建完整的对数梅尔频谱图。验证 80 个梅尔频带与 `librosa.feature.melspectrogram(n_mels=80)` 的结果在数值误差范围内一致。
3. **困难。** 实现流式推理：把音频切成 10 秒窗口、2 秒重叠的分块，对每个分块运行 Whisper，再合并转写结果。在一段 5 分钟的播客样本上，对比流式与单次整段处理的词错误率。

## 关键术语

| 术语 | 大家怎么说 | 实际含义 |
|------|-----------------|-----------------------|
| 梅尔频谱图 | 「音频图像」 | 二维表示：一个轴是频率频带，另一个轴是时间帧；每个单元格是对数刻度的能量。 |
| 对数梅尔（log-mel） | 「Whisper 看到的东西」 | 取对数后的梅尔频谱图；近似人耳对响度的感知。 |
| 帧 | 「一个时间切片」 | 25 ms 的采样窗口；以 10 ms 步长相互重叠。 |
| 任务 token | 「语音版的提示词前缀」 | 解码器提示词中的特殊 token，如 `<\|transcribe\|>` / `<\|translate\|>`。 |
| 语音活动检测（VAD） | 「找出有人说话的部分」 | 在 ASR 之前剔除静音的门控；能大幅削减成本。 |
| CTC | 「Connectionist Temporal Classification」 | 经典的免对齐 ASR 训练损失；Whisper 并不使用它。 |
| Whisper-turbo | 「小解码器，完整编码器」 | large-v3 编码器 + 4 层解码器；解码快 8 倍。 |
| Faster-whisper | 「生产级封装」 | CTranslate2 重新实现；int8 量化；比 OpenAI 参考实现快 4 倍。 |

## 延伸阅读

- [Radford et al. (2022). Robust Speech Recognition via Large-Scale Weak Supervision](https://arxiv.org/abs/2212.04356) —— Whisper 论文。
- [OpenAI Whisper repo](https://github.com/openai/whisper) —— 参考代码 + 模型权重。读一读 `whisper/model.py`，约 400 行从头到尾看完 Conv1D 干 + 编码器 + 解码器。
- [OpenAI Whisper — `whisper/decoding.py`](https://github.com/openai/whisper/blob/main/whisper/decoding.py) —— 第 5–6 步描述的束搜索 + 任务 token 逻辑就在这里；500 行，完全读得下来。
- [Baevski et al. (2020). wav2vec 2.0: A Framework for Self-Supervised Learning of Speech Representations](https://arxiv.org/abs/2006.11477) —— 前驱工作；其特征在某些场景下至今仍是 SOTA。
- [SYSTRAN/faster-whisper](https://github.com/SYSTRAN/faster-whisper) —— 生产级封装，比参考实现快 4 倍。
- [Jia et al. (2024). Moonshine: Speech Recognition for Live Transcription and Voice Commands](https://arxiv.org/abs/2410.15608) —— 2024 年面向端侧的 ASR，结构与 Whisper 同款但更小。
- [HuggingFace blog — "Fine-Tune Whisper For Multilingual ASR with 🤗 Transformers"](https://huggingface.co/blog/fine-tune-whisper) —— 权威的微调教程，包含梅尔频谱图预处理器和 token 时间戳的处理。
- [HuggingFace `modeling_whisper.py`](https://github.com/huggingface/transformers/blob/main/src/transformers/models/whisper/modeling_whisper.py) —— 完整实现（编码器、解码器、交叉注意力、生成），与本课的架构图一一对应。
