# 文本转语音（TTS）——从 Tacotron 到 F5 与 Kokoro

> ASR 把语音还原为文本；TTS 则把文本还原为语音。2026 年的技术栈分为三部分：文本 → token、token → 梅尔频谱、梅尔频谱 → 波形。每一部分都有一个能在笔记本电脑上运行的默认模型。

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 6 · 02 (Spectrograms & Mel), Phase 5 · 09 (Seq2Seq), Phase 7 · 05 (Full Transformer)
**Time:** ~75 minutes

## 问题背景

你手里有一个字符串："Please remind me to water the plants at 6 pm."。你需要生成一段 3 秒的音频：听起来自然、韵律（prosody）正确（停顿、重音）、"plants" 的元音发音准确，并且要在 CPU 上以低于 300 ms 的延迟运行，以支撑一个实时语音助手。你还需要能切换音色、处理语码混用的输入（"remind me at 6 pm, daijoubu?"），并且别在人名发音上闹笑话。

现代 TTS 流水线长这样：

1. **文本前端。** 对文本做规范化（日期、数字、邮箱），转换为音素或子词 token，并预测韵律特征。
2. **声学模型。** 文本 → 梅尔频谱图。Tacotron 2（2017）、FastSpeech 2（2020）、VITS（2021）、F5-TTS（2024）、Kokoro（2024）。
3. **声码器（vocoder）。** 梅尔频谱 → 波形。WaveNet（2016）、WaveRNN、HiFi-GAN（2020）、BigVGAN（2022），以及 2024 年之后的神经编解码器声码器。

到了 2026 年，端到端的扩散模型和流匹配（flow matching）模型让声学模型与声码器的界限变得模糊。但这个「三段式」心智模型在调试时依然好用。

## 核心概念

![Tacotron, FastSpeech, VITS, F5/Kokoro side-by-side](../assets/tts.svg)

**Tacotron 2（2017）。** Seq2seq 架构：字符嵌入 → BiLSTM 编码器 → 位置敏感注意力 → 自回归 LSTM 解码器逐帧输出梅尔频谱。速度慢（自回归），长文本上容易不稳定。如今仍被作为基线引用。

**FastSpeech 2（2020）。** 非自回归。时长预测器（duration predictor）输出每个音素对应多少个梅尔帧。单次前向，比 Tacotron 快 10 倍。自然度略有损失（单调对齐），但部署极其广泛。

**VITS（2021）。** 通过变分推断把编码器、基于流的时长模型和 HiFi-GAN 声码器联合端到端训练。质量高，单一模型。2022–2024 年开源 TTS 的主流。变体：YourTTS（多说话人零样本）、XTTS v2（2024，Coqui）。

**F5-TTS（2024）。** 基于流匹配的扩散 Transformer。韵律自然，仅需 5 秒参考音频即可零样本克隆音色。位居 2026 年开源 TTS 排行榜榜首。参数量 335M。

**Kokoro（2024）。** 小巧（82M）、可在 CPU 上运行，是实时场景下同级别中最好的英语 TTS。封闭词表、仅支持英语，采用 apache-2.0 许可。

**OpenAI TTS-1-HD、ElevenLabs v2.5、Google Chirp-3。** 商业领域的最高水平。ElevenLabs v2.5 的情感标签（"[whispered]"、"[laughing]"）和角色音色在 2026 年主导了有声书制作。

### 声码器演进

| 时代 | 声码器 | 延迟 | 质量 |
|-----|---------|---------|---------|
| 2016 | WaveNet | 仅限离线 | 发布时的 SOTA |
| 2018 | WaveRNN | 约实时 | 不错 |
| 2020 | HiFi-GAN | 100 倍实时 | 接近人声 |
| 2022 | BigVGAN | 50 倍实时 | 可泛化到不同说话人/语言 |
| 2024 | SNAC、DAC（神经编解码器） | 与自回归模型集成 | 离散 token，比特效率高 |

到 2026 年，大多数「TTS」模型已是从文本到波形的端到端模型；梅尔频谱图只是一种内部表示。

### 评估

- **MOS（平均意见分，Mean Opinion Score）。** 1–5 分制，众包打分。至今仍是金标准；但慢得令人痛苦。
- **CMOS（对比 MOS）。** A 对 B 的偏好评测。单次标注的置信区间更紧。
- **UTMOS、DNSMOS。** 无参考的神经 MOS 预测器。用于排行榜。
- **借助 ASR 计算 CER（字符错误率）。** 把 TTS 输出送进 Whisper，与输入文本计算 CER。作为可懂度的代理指标。
- **SECS（说话人嵌入余弦相似度）。** 衡量音色克隆质量。

2026 年在 LibriTTS test-clean 上的数据：

| 模型 | UTMOS | CER（经 Whisper） | 大小 |
|-------|-------|-------------------|------|
| 真实录音 | 4.08 | 1.2% | — |
| F5-TTS | 3.95 | 2.1% | 335M |
| XTTS v2 | 3.81 | 3.5% | 470M |
| VITS | 3.62 | 3.1% | 25M |
| Kokoro v0.19 | 3.87 | 1.8% | 82M |
| Parler-TTS Large | 3.76 | 2.8% | 2.3B |

## 从零实现

### 第 1 步：对输入做音素化

```python
from phonemizer import phonemize
ph = phonemize("Hello world", language="en-us", backend="espeak")
# 'həloʊ wɜːld'
```

音素是通用的桥梁。凡是质量达不到 VITS 水平的模型，都不要直接喂原始文本。

### 第 2 步：运行 Kokoro（2026 年的 CPU 默认选项）

```python
from kokoro import KPipeline
tts = KPipeline(lang_code="a")  # "a" = American English
audio, sr = tts("Please remind me to water the plants at 6 pm.", voice="af_bella")
# audio: float32 tensor, sr=24000
```

可离线运行，单文件，82M 参数。

### 第 3 步：用 F5-TTS 做音色克隆

```python
from f5_tts.api import F5TTS
tts = F5TTS()
wav = tts.infer(
    ref_file="my_voice_5s.wav",
    ref_text="The quick brown fox jumps over the lazy dog.",
    gen_text="Please remind me to water the plants.",
)
```

传入一段 5 秒的参考音频及其文字稿；F5 会克隆其韵律和音色。

### 第 4 步：从零实现 HiFi-GAN 声码器

完整代码放不进一篇教程脚本，但骨架是这样的：

```python
class HiFiGAN(nn.Module):
    def __init__(self, mel_channels=80, upsample_rates=[8, 8, 2, 2]):
        super().__init__()
        # 4 upsample blocks, total 256x to go from mel-rate to audio-rate
        ...
    def forward(self, mel):
        return self.blocks(mel)  # -> waveform
```

训练方式：对抗损失（判别器作用于短窗口）+ 梅尔频谱重建损失 + 特征匹配损失。这部分已经完全商品化——直接使用 `hifi-gan` 仓库或 nvidia-NeMo 的预训练检查点即可。

### 第 5 步：完整流水线（伪代码）

```python
text = "Please remind me at 6 pm."
phones = phonemize(text)
mel = acoustic_model(phones, speaker=alice)      # [T, 80]
wav = vocoder(mel)                                # [T * 256]
soundfile.write("out.wav", wav, 24000)
```

## 生产实践

2026 年的技术栈：

| 场景 | 选型 |
|-----------|------|
| 实时英语语音助手 | Kokoro（CPU）或 XTTS v2（GPU） |
| 基于 5 秒参考音频的音色克隆 | F5-TTS |
| 商业角色音色 | ElevenLabs v2.5 |
| 有声书旁白 | ElevenLabs v2.5 或 XTTS v2 + 微调 |
| 低资源语言 | 用 5–20 小时目标语言数据训练 VITS |
| 富表现力 / 情感标签 | ElevenLabs v2.5 或 StyleTTS 2 微调 |

2026 年的开源领跑者：**质量看 F5-TTS，效率看 Kokoro**。除非你是做考古，否则别再用 Tacotron。

## 常见陷阱

- **没做文本规范化。** "Dr. Smith" 读成 "Doctor" 还是 "Drive"？"2026" 读成 "twenty twenty six" 还是 "two zero two six"？务必在音素化器之前完成规范化。
- **词表外的专有名词。** "Ghumare" 被读成 "ghyu-mair"？为未知 token 准备一个备用的字素到音素（grapheme-to-phoneme）模型。
- **削波（clipping）。** 声码器输出本身很少削波，但推理时梅尔缩放不匹配可能让幅值超出 ±1.0。永远加上 `np.clip(wav, -1, 1)`。
- **采样率不匹配。** Kokoro 输出 24 kHz；下游流水线如果预期 16 kHz → 要么重采样，要么得到混叠失真。

## 交付产物

保存为 `outputs/skill-tts-designer.md`。针对给定的音色、延迟和语言目标，设计一条 TTS 流水线。

## 练习

1. **简单。** 运行 `code/main.py`。它会从一个玩具词表构建音素词典，估算每个音素的时长，并打印一份模拟的「梅尔」时间表。
2. **中等。** 安装 Kokoro，分别用 `af_bella` 和 `am_adam` 两个音色合成同一句话。比较音频时长和主观质量。
3. **困难。** 录一段 5 秒的自己的参考音频。用 F5-TTS 进行克隆。报告参考音频与克隆输出之间的 SECS。

## 关键术语

| 术语 | 通俗说法 | 实际含义 |
|------|-----------------|-----------------------|
| 音素（Phoneme） | 声音单元 | 抽象的音类；英语有 39 个（ARPABet）。 |
| 时长预测器 | 每个音素持续多久 | 非自回归模型的输出；每个音素对应的整数帧数。 |
| 声码器（Vocoder） | 梅尔频谱 → 波形 | 把梅尔频谱映射为原始采样点的神经网络。 |
| HiFi-GAN | 标准声码器 | 基于 GAN；2020–2024 年的主流。 |
| MOS | 主观质量 | 人类评分者给出的 1–5 平均意见分。 |
| SECS | 音色克隆指标 | 目标与输出说话人嵌入之间的余弦相似度。 |
| F5-TTS | 2024 年开源 SOTA | 流匹配扩散模型；零样本克隆。 |
| Kokoro | CPU 英语第一梯队 | 82M 参数模型，Apache 2.0 许可。 |

## 延伸阅读

- [Shen et al. (2017). Tacotron 2](https://arxiv.org/abs/1712.05884) —— seq2seq 基线。
- [Kim, Kong, Son (2021). VITS](https://arxiv.org/abs/2106.06103) —— 端到端、基于流的模型。
- [Chen et al. (2024). F5-TTS](https://arxiv.org/abs/2410.06885) —— 当前的开源 SOTA。
- [Kong, Kim, Bae (2020). HiFi-GAN](https://arxiv.org/abs/2010.05646) —— 到 2026 年仍在大规模部署的声码器。
- [Kokoro-82M on HuggingFace](https://huggingface.co/hexgrad/Kokoro-82M) —— 2024 年对 CPU 友好的英语 TTS。
