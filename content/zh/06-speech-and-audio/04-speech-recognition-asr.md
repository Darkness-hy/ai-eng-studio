# 语音识别（ASR）— CTC、RNN-T 与注意力

> 语音识别本质上是在每个时间步做音频分类，再由一个懂英语、也懂静音的序列模型把结果串起来。CTC、RNN-T 和注意力是三种实现路径。选定一种，并弄清楚为什么选它。

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 6 · 02 (Spectrograms & Mel), Phase 5 · 08 (CNNs & RNNs for Text), Phase 5 · 10 (Attention)
**Time:** ~45 minutes

## 问题背景

你手上有一段 10 秒、16 kHz 的音频，想得到一个字符串："turn on the kitchen lights"。难点是结构性的：音频帧与字符之间没有一一对应的对齐关系。单词 "okay" 可能只占 200 ms，也可能拖到 1200 ms。静音穿插在整句话之间。有些音素比其他音素更长。输出 token 的数量事先并不可知。

有三种建模方式可以解决这个问题：

1. **CTC（Connectionist Temporal Classification，连接时序分类）。** 逐帧输出 token 概率，其中包含一个特殊的*空白符（blank）*。解码时合并重复字符并删除空白符。非自回归，速度快。被 wav2vec 2.0、MMS 采用。
2. **RNN-T（Recurrent Neural Network Transducer，循环神经网络转录器）。** 联合网络根据编码器帧和历史 token 预测下一个 token。可流式处理。被 Google 的端侧 ASR、NVIDIA Parakeet 采用。
3. **注意力编码器-解码器。** 编码器把音频压缩成隐藏状态，解码器通过交叉注意力自回归地生成 token。被 Whisper、SeamlessM4T 采用。

到 2026 年，LibriSpeech test-clean 上的 SOTA WER 是 1.4%（Parakeet-TDT-1.1B，NVIDIA）和 1.58%（Whisper-Large-v3-turbo）。指标差距微乎其微，部署层面的差异却非常大。

## 核心概念

![Three ASR formulations: CTC, RNN-T, attention-encoder-decoder](../assets/asr-formulations.svg)

**CTC 的直觉。** 让编码器输出 `T` 个帧级分布，覆盖 `V+1` 个 token（V 个字符加 blank）。对于一个长度为 `U < T` 的目标字符串 `y`，任何能折叠回 `y` 的帧级对齐都算有效。CTC 损失对所有这些对齐求和。推理时：逐帧取 argmax，合并重复，删除 blank。

优点：非自回归、可流式、零前瞻。缺点：*条件独立假设*——每一帧的预测彼此独立，因此模型内部没有语言模型。补救办法是通过束搜索（beam search）或浅融合（shallow fusion）接入外部 LM。

**RNN-T 的直觉。** 增加一个对 token 历史做嵌入的*预测器（predictor）*网络，以及一个把预测器状态与编码器帧组合成 `V+1` 维联合分布的*连接器（joiner）*（这里的 `+1` 是空输出 / 不发射）。它显式建模了 CTC 忽略的条件依赖。由于每一步只依赖过去的帧和过去的 token，因此可以流式处理。

优点：可流式 + 自带内部 LM。缺点：训练更复杂、更吃显存（3D 损失格点）；RNN-T 损失的算子内核自成一个库类目。

**注意力编码器-解码器。** 编码器（6-32 层 Transformer）作用于 log-mel 帧。解码器（6-32 层 Transformer）对编码器输出做交叉注意力，自回归地生成 token。没有对齐约束——注意力可以看到音频的任何位置。除非限制注意力范围（分块的 Whisper-Streaming，2024），否则无法流式处理。

优点：离线 ASR 质量最高，可以用标准 seq2seq 工具链轻松训练。缺点：自回归延迟与输出长度成正比；不做额外工程就无法流式处理。

### WER：唯一的核心指标

**词错误率（Word Error Rate）** = `(S + D + I) / N`，其中 S=替换数、D=删除数、I=插入数、N=参考文本词数。等价于词级别的 Levenshtein 编辑距离。越低越好。WER 超过 20% 基本不可用；低于 5% 在朗读语音上已达到人类水平。2026 年标准基准上的数字：

| 模型 | LibriSpeech test-clean | LibriSpeech test-other | 规模 |
|-------|------------------------|------------------------|------|
| Parakeet-TDT-1.1B | 1.40% | 2.78% | 1.1B 参数 |
| Whisper-Large-v3-turbo | 1.58% | 3.03% | 809M |
| Canary-1B Flash | 1.48% | 2.87% | 1B |
| Seamless M4T v2 | 1.7% | 3.5% | 2.3B |

以上全部基于编码器-解码器或 RNN-T。纯 CTC 系统（wav2vec 2.0）在 test-clean 上大约在 1.8–2.1%。

## 从零实现

### 第 1 步：贪心 CTC 解码

```python
def ctc_greedy(frame_logits, blank=0, vocab=None):
    # frame_logits: list of per-frame probability vectors
    preds = [max(range(len(p)), key=lambda i: p[i]) for p in frame_logits]
    out = []
    prev = -1
    for p in preds:
        if p != prev and p != blank:
            out.append(p)
        prev = p
    return "".join(vocab[i] for i in out) if vocab else out
```

两条规则：合并连续重复，删除 blank。例如：`a a _ _ a b b _ c` → `a a b c`。

### 第 2 步：束搜索 CTC

```python
def ctc_beam(frame_logits, beam=8, blank=0):
    import math
    beams = [([], 0.0)]  # (tokens, log_prob)
    for p in frame_logits:
        log_p = [math.log(max(pi, 1e-10)) for pi in p]
        candidates = []
        for seq, lp in beams:
            for t, lpt in enumerate(log_p):
                new = seq[:] if t == blank else (seq + [t] if not seq or seq[-1] != t else seq)
                candidates.append((new, lp + lpt))
        candidates.sort(key=lambda x: -x[1])
        beams = candidates[:beam]
    return beams[0][0]
```

生产环境用的是带 LM 融合的前缀树束搜索；这里给出的是概念骨架。

### 第 3 步：WER

```python
def wer(ref, hyp):
    r, h = ref.split(), hyp.split()
    dp = [[0] * (len(h) + 1) for _ in range(len(r) + 1)]
    for i in range(len(r) + 1):
        dp[i][0] = i
    for j in range(len(h) + 1):
        dp[0][j] = j
    for i in range(1, len(r) + 1):
        for j in range(1, len(h) + 1):
            cost = 0 if r[i - 1] == h[j - 1] else 1
            dp[i][j] = min(
                dp[i - 1][j] + 1,
                dp[i][j - 1] + 1,
                dp[i - 1][j - 1] + cost,
            )
    return dp[len(r)][len(h)] / max(1, len(r))
```

### 第 4 步：用 Whisper 跑推理

```python
import whisper
model = whisper.load_model("large-v3-turbo")
result = model.transcribe("clip.wav")
print(result["text"])
```

一行代码就能用上 2026 年最强的通用 ASR。在 24 GB GPU 上能以约 20 倍实时速度运行。

### 第 5 步：用 Parakeet 或 wav2vec 2.0 做流式识别

```python
from transformers import pipeline
asr = pipeline("automatic-speech-recognition", model="nvidia/parakeet-tdt-1.1b")
for chunk in streaming_audio():
    print(asr(chunk, return_timestamps=True))
```

流式 ASR 需要分块的编码器注意力和跨块的状态传递；请使用支持这些特性的库（Parakeet 用 NeMo，或用 `transformers` pipeline 配合 `chunk_length_s`）。

## 生产实践

2026 年的技术选型：

| 场景 | 选择 |
|-----------|------|
| 英语、离线、追求最高质量 | Whisper-large-v3-turbo |
| 多语言、鲁棒性优先 | SeamlessM4T v2 |
| 流式、低延迟 | Parakeet-TDT-1.1B 或 Riva |
| 边缘端、移动端、延迟 <500 ms | 量化后的 Whisper-Tiny 或 Moonshine（2024） |
| 长音频 | Whisper 配合基于 VAD 的分块（WhisperX） |
| 垂直领域（医疗、法律） | 微调 wav2vec 2.0 + 领域 LM 融合 |

## 2026 年仍在线上出现的常见陷阱

- **不加 VAD。** 让 Whisper 处理静音会产生幻觉（输出 "Thanks for watching!"）。务必用 VAD 做前置过滤。
- **字符级、词级、子词级 WER 混用。** 应在归一化（小写化、去标点）*之后*报告词级 WER。
- **语言识别漂移。** Whisper 的自动语言识别（LID）会把嘈杂音频错判成日语或威尔士语；已知语言时请强制指定 `language="en"`。
- **长音频不分块。** Whisper 的窗口只有 30 秒。超过这个长度请用 `chunk_length_s=30, stride=5`。

## 交付产物

保存为 `outputs/skill-asr-picker.md`。针对给定的部署目标，选定模型、解码策略、分块方案和 LM 融合方式。

## 练习

1. **简单。** 运行 `code/main.py`。它会对一段手工构造的 CTC 输出做贪心解码，并对照参考文本计算 WER。
2. **中等。** 正式实现第 2 步中的前缀树束搜索（处理好 blank 合并规则）。在一个 10 条样本的合成数据集上与贪心解码做对比。
3. **困难。** 在 [LibriSpeech test-clean](https://www.openslr.org/12) 上运行 `whisper-large-v3-turbo`。计算前 100 条语音的 WER，并与公开发表的数字对比。

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|-----------------|-----------------------|
| CTC | 带空白符的损失 | 对所有帧到 token 对齐求边缘分布；非自回归。 |
| RNN-T | 流式损失 | CTC + 下一 token 预测器；能处理词序。 |
| 注意力编码器-解码器 | Whisper 路线 | 编码器 + 交叉注意力解码器；离线质量最佳。 |
| WER | 你汇报的那个数字 | 词级别的 `(S+D+I)/N`。 |
| Blank | 空白占位 | CTC 中表示"本帧不输出任何 token"的特殊符号。 |
| LM 融合 | 外部语言模型 | 束搜索时叠加加权的 LM 对数概率。 |
| VAD | 静音门控 | 语音活动检测器；裁掉非语音片段。 |

## 延伸阅读

- [Graves et al. (2006). Connectionist Temporal Classification](https://www.cs.toronto.edu/~graves/icml_2006.pdf) — CTC 原始论文。
- [Graves (2012). Sequence Transduction with RNNs](https://arxiv.org/abs/1211.3711) — RNN-T 原始论文。
- [Radford et al. / OpenAI (2022). Whisper: Robust Speech Recognition via Large-Scale Weak Supervision](https://arxiv.org/abs/2212.04356) — 2022 年的奠基性论文；v3-turbo 扩展发布于 2024 年。
- [NVIDIA NeMo — Parakeet-TDT card](https://huggingface.co/nvidia/parakeet-tdt-1.1b) — 2026 年 Open ASR Leaderboard 榜首。
- [Hugging Face — Open ASR Leaderboard](https://huggingface.co/spaces/hf-audio/open_asr_leaderboard) — 覆盖 25+ 模型的实时基准。
