# 神经音频编解码器 —— EnCodec、SNAC、Mimi、DAC 与语义-声学分离

> 2026 年的音频生成几乎全部基于 token。EnCodec、SNAC、Mimi 和 DAC 把连续波形转换为 Transformer 可以预测的离散序列。语义 token 与声学 token 的分离——第一个码本承载语义、其余码本承载声学——是音频领域自 Transformer 以来最重要的架构变革。

**Type:** Learn
**Languages:** Python
**Prerequisites:** Phase 6 · 02 (Spectrograms), Phase 10 · 11 (Quantization), Phase 5 · 19 (Subword Tokenization)
**Time:** ~60 minutes

## 问题背景

语言模型处理的是离散 token，而音频是连续的。如果你想为语音 / 音乐构建一个 LLM 风格的模型——MusicGen、Moshi、Sesame CSM、VibeVoice、Orpheus——首先需要一个**神经音频编解码器（neural audio codec）**：一个经过学习的编码器，把音频离散化为一个小词表的 token，再配上一个对应的解码器来重建波形。

目前已经形成了两大流派：

1. **重建优先的编解码器**——EnCodec、DAC。以感知音频质量为优化目标。token 是"声学的"——它们捕捉一切信息，包括说话人身份、音色、背景噪声。
2. **语义优先的编解码器**——Mimi（Kyutai）、SpeechTokenizer。强制第一个码本编码语言学 / 音素内容（通常通过从 WavLM 蒸馏实现）。后续码本承载声学细节。

2024-2026 年的关键洞见是：**纯重建型编解码器在从文本生成语音时会产出含混不清的结果。** 基于编解码器 token 的 LLM 不得不在同一组码本中同时学习语言结构和声学结构，这种方式无法扩展。把两者分开——码本 0 负责语义，码本 1-N 负责声学——正是 Moshi 和 Sesame CSM 能够成功的原因。

## 核心概念

![Four codec landscape: EnCodec, DAC, SNAC (multi-scale), Mimi (semantic+acoustic)](../assets/codec-comparison.svg)

### 核心技巧：残差向量量化（Residual Vector Quantization, RVQ）

与其使用一个巨大的码本（要达到良好质量需要数百万个编码），所有现代音频编解码器都采用 **RVQ**：一组级联的小码本。第一个码本对编码器输出进行量化；第二个码本对残差进行量化；依此类推。每个码本包含 1024 个编码。8 个码本 = 等效词表大小 1024^8 = 10^24。

推理时，解码器把每帧选中的所有编码相加来完成重建。

### 2026 年最重要的四个编解码器

**EnCodec（Meta，2022）。** 基线模型。波形上的编码器-解码器结构，RVQ 瓶颈层。24 kHz，最多可用 32 个码本，默认 4 个码本 @ 1.5 kbps。采用 `1D conv + transformer + 1D conv` 架构。被 MusicGen 使用。

**DAC（Descript，2023）。** 采用 L2 归一化码本、周期激活函数和改进损失函数的 RVQ。在所有开源编解码器中重建保真度最高——使用 12 个码本时有时与原始语音难以区分。44.1 kHz 全频带。

**SNAC（Hubert Siuzdak，2024）。** 多尺度 RVQ——粗粒度码本的帧率低于细粒度码本。实际上以层次化方式建模音频：约 12 Hz 的粗粒度"草图"加上 50 Hz 的细节。Orpheus-3B 采用它，因为这种层次结构与基于语言模型的生成方式契合得很好。

**Mimi（Kyutai，2024）。** 2026 年的游戏规则改变者。12.5 Hz 帧率（极低），8 个码本 @ 4.4 kbps。码本 0 **从 WavLM 蒸馏而来**——训练目标是预测 WavLM 的语音内容特征。码本 1-7 是声学残差。这种分离支撑了 Moshi（第 15 课）和 Sesame CSM。

### 帧率对语言建模至关重要

帧率越低 = 序列越短 = 语言模型越快。

| 编解码器 | 帧率 | 1 秒 = N 帧 | 适用场景 |
|-------|-----------|----------------|---------|
| EnCodec-24k | 75 Hz | 75 | 音乐、通用音频 |
| DAC-44.1k | 86 Hz | 86 | 高保真音乐 |
| SNAC-24k（粗粒度） | ~12 Hz | 12 | 高效自回归语言模型 |
| Mimi | 12.5 Hz | 12.5 | 流式语音 |

在 12.5 Hz 下，一段 10 秒的语音只有 125 个编解码帧——Transformer 可以轻松预测它们。

### 语义 token 与声学 token

```
frame_t → [semantic_token_t, acoustic_token_0_t, acoustic_token_1_t, ..., acoustic_token_6_t]
```

- **语义 token（Mimi 中的码本 0）。** 编码说了什么——音素、词、内容。通过一个辅助预测损失从 WavLM 蒸馏而来。
- **声学 token（码本 1-7）。** 编码音色、说话人身份、韵律、背景噪声、细节。

自回归语言模型先预测语义 token（以文本为条件），再预测声学 token（以语义 token + 说话人参考为条件）。这种因子分解正是现代 TTS 能够零样本克隆声音的原因：语义模型负责内容，声学模型负责音色。

### 2026 年重建质量（每秒比特数，码率越低越好）

| 编解码器 | 码率 | PESQ | ViSQOL |
|-------|---------|------|--------|
| Opus-20kbps | 20 kbps | 4.0 | 4.3 |
| EnCodec-6kbps | 6 kbps | 3.2 | 3.8 |
| DAC-6kbps | 6 kbps | 3.5 | 4.0 |
| SNAC-3kbps | 3 kbps | 3.3 | 3.8 |
| Mimi-4.4kbps | 4.4 kbps | 3.1 | 3.7 |

在感知质量的单位比特效率上，Opus 等传统编解码器仍然胜出。神经编解码器的优势在于**离散 token**（Opus 不产生 token）和**生成模型质量**（语言模型能用这些 token 做什么）。

## 从零实现

### 第 1 步：用 EnCodec 编码

```python
from encodec import EncodecModel
import torch

model = EncodecModel.encodec_model_24khz()
model.set_target_bandwidth(6.0)  # kbps

wav = torch.randn(1, 1, 24000)
with torch.no_grad():
    encoded = model.encode(wav)
codes, scale = encoded[0]
# codes: (1, n_codebooks, n_frames), dtype=int64
```

6 kbps 下 `n_codebooks=8`。每个编码取值 0-1023（10 比特）。

### 第 2 步：解码并测量重建效果

```python
with torch.no_grad():
    wav_recon = model.decode([(codes, scale)])

from torchaudio.functional import compute_deltas
import torch.nn.functional as F

mse = F.mse_loss(wav_recon[:, :, :wav.shape[-1]], wav).item()
```

### 第 3 步：语义-声学分离（Mimi 风格）

```python
from moshi.models import loaders
mimi = loaders.get_mimi()

with torch.no_grad():
    codes = mimi.encode(wav)  # shape (1, 8, frames@12.5Hz)

semantic = codes[:, 0]
acoustic = codes[:, 1:]
```

语义码本 0 与 WavLM 对齐。你可以训练一个文本到语义的 Transformer——词表比直接生成音频小得多。然后再用一个独立的声学到波形解码器，以说话人参考为条件进行解码。

### 第 4 步：为什么基于编解码 token 的自回归语言模型可行

对于一段 10 秒的语音，按 Mimi 的 12.5 Hz × 8 个码本计算：

```
N_tokens = 10 * 12.5 * 8 = 1000 tokens
```

1000 个 token 对 Transformer 来说是微不足道的上下文长度。一个 2.56 亿参数的 Transformer 在现代 GPU 上可以在毫秒级时间内生成 10 秒的语音。

## 生产实践

按问题选编解码器：

| 任务 | 编解码器 |
|------|-------|
| 通用音乐生成 | EnCodec-24k |
| 最高保真度重建 | DAC-44.1k |
| 基于语音的自回归语言模型（TTS） | SNAC 或 Mimi |
| 流式全双工语音 | Mimi（12.5 Hz） |
| 带文本条件的音效库 | EnCodec + T5 条件 |
| 细粒度音频编辑 | DAC + 修补（inpainting） |

经验法则：**如果你在构建生成模型，从 Mimi 或 SNAC 入手。如果你在构建压缩管线，用 Opus。**

## 常见陷阱

- **码本太多。** 增加码本会让保真度线性提升，但语言模型的序列长度也会线性增长。在 8-12 个时停手。
- **帧率不匹配。** 先在 12.5 Hz 的 Mimi 上训练语言模型，再到 50 Hz 的 EnCodec 上微调，会静默失败。
- **假设所有码本同等重要。** 在 Mimi 中，码本 0 承载内容；丢失它会摧毁可懂度。丢失码本 7 则几乎察觉不到。
- **只用重建质量这一个指标。** 一个编解码器可以重建质量极好，但如果语义结构很差，对基于语言模型的生成毫无用处。

## 交付产物

保存为 `outputs/skill-codec-picker.md`。为给定的生成或压缩任务选择一个编解码器。

## 练习

1. **简单。** 运行 `code/main.py`。它实现了一个玩具级的标量 + 残差量化器，并测量随着码本增加时重建误差的变化。
2. **中等。** 安装 `encodec`，在一段留出的语音片段上比较 1、4、8、32 个码本。绘制 PESQ 或 MSE 随码率变化的曲线。
3. **困难。** 加载 Mimi。编码一段音频。把码本 0 替换为随机整数后解码。再对码本 7 做同样的替换。对比两种破坏的效果——破坏码本 0 应该会摧毁可懂度；破坏码本 7 应该几乎没有变化。

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|-----------------|-----------------------|
| RVQ | 残差量化 | 级联的小码本；每个码本量化前一级的残差。 |
| 帧率 | 编解码器速度 | 每秒多少个 token 帧。越低 = 语言模型越快。 |
| 语义码本 | 码本 0（Mimi） | 从 SSL 特征蒸馏而来的码本；编码内容。 |
| 声学码本 | 其余所有码本 | 音色、韵律、噪声、细节。 |
| PESQ / ViSQOL | 感知质量 | 与 MOS 相关的客观指标。 |
| EnCodec | Meta 的编解码器 | RVQ 基线；被 MusicGen 使用。 |
| Mimi | Kyutai 的编解码器 | 12.5 Hz 帧率；语义-声学分离；支撑 Moshi。 |

## 延伸阅读

- [Défossez et al. (2023). EnCodec](https://arxiv.org/abs/2210.13438) —— RVQ 基线。
- [Kumar et al. (2023). Descript Audio Codec (DAC)](https://arxiv.org/abs/2306.06546) —— 保真度最高的开源编解码器。
- [Siuzdak (2024). SNAC](https://arxiv.org/abs/2410.14411) —— 多尺度 RVQ。
- [Kyutai (2024). Mimi codec](https://kyutai.org/codec-explainer) —— 语义-声学分离，WavLM 蒸馏。
- [Borsos et al. (2023). AudioLM](https://arxiv.org/abs/2209.03143) —— 两阶段语义/声学范式的开创者。
- [Zeghidour et al. (2021). SoundStream](https://arxiv.org/abs/2107.03312) —— 最早的可流式 RVQ 编解码器。
