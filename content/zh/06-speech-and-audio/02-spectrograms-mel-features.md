# 频谱图、梅尔刻度与音频特征

> 神经网络并不擅长直接消化原始波形，它们更适合处理频谱图（spectrogram），而梅尔频谱图（mel spectrogram）效果更好。2026 年的每一个 ASR、TTS 和音频分类器，成败都系于这一个预处理选择。

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 6 · 01 (Audio Fundamentals)
**Time:** ~45 minutes

## 问题背景

拿一段 10 秒、16 kHz 的音频来看：那是 160,000 个浮点数，全部落在 `[-1, 1]` 区间内，与「狗叫」或「单词 cat」这样的标签几乎完全不相关。原始波形包含信息，但模型很难直接从这种形式中提取。相隔 100 毫秒说出的两个完全相同的音素，其原始采样值截然不同。

频谱图解决了这个问题。它把人类感知会忽略的时间细节（微秒级抖动）压缩掉，同时保留感知所关注的结构（在约 10–25 毫秒的时间窗口内，哪些频率带有能量）。

梅尔频谱图更进一步。人类对音高的感知是对数式的：100 Hz 与 200 Hz 之间的「距离感」和 1000 Hz 与 2000 Hz 之间相同。梅尔刻度（mel scale）对频率轴做了相应的扭曲来匹配这种感知。从 2010 年到 2026 年，梅尔刻度频谱图始终是语音机器学习中最重要的单一特征。

## 核心概念

![Waveform to STFT to mel spectrogram to MFCC ladder](../assets/mel-features.svg)

**STFT（短时傅里叶变换，Short-Time Fourier Transform）。** 把波形切成相互重叠的帧（典型设置：25 毫秒窗长、10 毫秒帧移，在 16 kHz 下即 400 个采样点 / 160 个采样点）。每一帧乘以一个窗函数（Hann 是默认选择；Hamming 的取舍略有不同）。对每帧做 FFT。把幅度谱堆叠成形状为 `(n_frames, n_freq_bins)` 的矩阵——这就是你的频谱图。

**对数幅度。** 原始幅度的跨度有 5–6 个数量级。取 `log(|X| + 1e-6)` 或 `20 * log10(|X|)` 来压缩动态范围。所有生产管线用的都是对数幅度，而不是原始幅度。

**梅尔刻度。** 频率 `f`（Hz）通过 `m = 2595 * log10(1 + f / 700)` 映射到梅尔值 `m`。这个映射在 1 kHz 以下大致是线性的，以上则大致是对数的。覆盖 0–8 kHz 的 80 个梅尔频带是 ASR 的标准输入。

**梅尔滤波器组（mel filterbank）。** 一组在梅尔刻度上等间距排列的三角滤波器。每个滤波器是相邻 FFT 频带的加权和。用滤波器组矩阵乘以 STFT 幅度，一次矩阵乘法就得到梅尔频谱图。

**对数梅尔频谱图（log-mel spectrogram）。** `log(mel_spec + 1e-10)`。Whisper 的输入、Parakeet 的输入、SeamlessM4T 的输入——2026 年通用的音频前端。

**MFCC。** 取对数梅尔频谱图，做一次 DCT（type II），保留前 13 个系数。它对特征做了去相关并进一步压缩。在 2015 年左右 CNN/Transformer 直接处理原始 log-mel 的能力追上来之前，MFCC 是主导特征。如今在说话人识别（x-vectors、ECAPA）中仍在使用。

**分辨率取舍。** FFT 越大，频率分辨率越好，但时间分辨率越差。25 毫秒 / 10 毫秒是音频机器学习的默认配置；音乐用 50 毫秒 / 12.5 毫秒；瞬态检测（鼓点、爆破音）用 5 毫秒 / 2 毫秒。

```figure
spectrogram-window
```

## 从零实现

### 第 1 步：对波形分帧

```python
def frame(signal, frame_len, hop):
    n = 1 + (len(signal) - frame_len) // hop
    return [signal[i * hop : i * hop + frame_len] for i in range(n)]
```

一段 10 秒、16 kHz 的音频，在 `frame_len=400, hop=160` 下会产生 998 帧。

### 第 2 步：Hann 窗

```python
import math

def hann(N):
    return [0.5 * (1 - math.cos(2 * math.pi * n / (N - 1))) for n in range(N)]
```

在 FFT 之前逐元素相乘。它消除了在非零端点处截断所导致的频谱泄漏。

### 第 3 步：STFT 幅度

```python
def stft_magnitude(signal, frame_len=400, hop=160):
    win = hann(frame_len)
    frames = frame(signal, frame_len, hop)
    return [magnitudes(dft([w * s for w, s in zip(win, f)])) for f in frames]
```

生产环境使用 `torch.stft` 或 `librosa.stft`（基于 FFT、向量化）。这里的循环写法只为教学目的；在 `code/main.py` 中处理短音频时可以正常运行。

### 第 4 步：梅尔滤波器组

```python
def hz_to_mel(f):
    return 2595.0 * math.log10(1.0 + f / 700.0)

def mel_to_hz(m):
    return 700.0 * (10 ** (m / 2595.0) - 1)

def mel_filterbank(n_mels, n_fft, sr, fmin=0, fmax=None):
    fmax = fmax or sr / 2
    mels = [hz_to_mel(fmin) + (hz_to_mel(fmax) - hz_to_mel(fmin)) * i / (n_mels + 1)
            for i in range(n_mels + 2)]
    hzs = [mel_to_hz(m) for m in mels]
    bins = [int(h * n_fft / sr) for h in hzs]
    fb = [[0.0] * (n_fft // 2 + 1) for _ in range(n_mels)]
    for m in range(n_mels):
        for k in range(bins[m], bins[m + 1]):
            fb[m][k] = (k - bins[m]) / max(1, bins[m + 1] - bins[m])
        for k in range(bins[m + 1], bins[m + 2]):
            fb[m][k] = (bins[m + 2] - k) / max(1, bins[m + 2] - bins[m + 1])
    return fb
```

覆盖 0–8 kHz 的 80 个梅尔频带，配合 `n_fft=400`，得到一个 `(80, 201)` 的矩阵。用 `(n_frames, 201)` 的 STFT 幅度乘以它的转置，就得到 `(n_frames, 80)` 的梅尔频谱图。

### 第 5 步：对数梅尔

```python
def log_mel(mel_spec, eps=1e-10):
    return [[math.log(max(v, eps)) for v in frame] for frame in mel_spec]
```

常见的替代方案：`librosa.power_to_db`（按参考值归一化的 dB）、`10 * log10(power + eps)`。Whisper 使用了一套更复杂的裁剪加归一化流程（参见 Whisper 的 `log_mel_spectrogram`）。

### 第 6 步：MFCC

```python
def dct_ii(x, n_coeffs):
    N = len(x)
    return [
        sum(x[n] * math.cos(math.pi * k * (2 * n + 1) / (2 * N)) for n in range(N))
        for k in range(n_coeffs)
    ]
```

对每个对数梅尔帧做 DCT，保留前 13 个系数，得到的就是 MFCC 矩阵。第一个系数通常会丢弃（它编码的是整体能量）。

## 生产实践

2026 年的技术栈：

| 任务 | 特征 |
|------|----------|
| ASR（Whisper、Parakeet、SeamlessM4T） | 80 个 log-mel，10 毫秒帧移，25 毫秒窗长 |
| TTS 声学模型（VITS、F5-TTS、Kokoro） | 80 个 mel，5–12 毫秒帧移以实现精细的时间控制 |
| 音频分类（AST、PANNs、BEATs） | 128 个 log-mel，10 毫秒帧移 |
| 说话人嵌入（ECAPA-TDNN、WavLM） | 80 个 log-mel 或基于原始波形的自监督学习 |
| 音乐（MusicGen、Stable Audio 2） | EnCodec 离散 token（不用 mel） |
| 关键词检测 | 微型设备用 40 个 MFCC |

经验法则：**只要不是做音乐，就从 80 个 log-mel 开始。** 任何偏离这个默认值的选择都需要给出充分理由。

## 2026 年仍会上线的陷阱

- **梅尔数量不匹配。** 训练用 80 个 mel，推理用 128 个 mel。静默失败。在两端都记录特征的形状。
- **上游采样率不匹配。** 22.05 kHz 下计算的 mel 和 16 kHz 下的看起来不一样。在特征提取*之前*先统一采样率。
- **dB 与 log 混淆。** Whisper 期望的是 log-mel，不是 dB-mel。某些 HF 管线会自动检测，但你的自定义代码不会。
- **归一化漂移。** 训练时用逐句归一化，推理时用全局归一化。这是会让 WER 翻倍的生产 bug。
- **填充导致的泄漏。** 对音频末尾做零填充会在尾部帧产生平坦的频谱。改用对称填充或复制填充。

## 交付产物

保存为 `outputs/skill-feature-extractor.md`。这份技能文档负责为给定的目标模型选择特征类型、梅尔数量、帧长/帧移和归一化方式。

## 练习

1. **简单。** 运行 `code/main.py`。它会合成一段 chirp 信号（频率从 200 扫到 4000 Hz），并打印每帧能量最大的梅尔频带索引。绘图（可选）并确认它与扫频轨迹一致。
2. **中等。** 分别用 `{40, 80, 128}` 中的 `n_mels` 和 `{200, 400, 800}` 中的 `frame_len` 重新运行。沿时间轴测量尖峰的带宽。哪种组合对 chirp 的分辨效果最好？
3. **困难。** 实现 `power_to_db`，并在 AudioMNIST 上用一个小型 CNN 分类器比较三种特征的 ASR 准确率：(a) 原始 log-mel，(b) `ref=max` 的 dB-mel，(c) MFCC-13 + delta + delta-delta。报告 top-1 准确率。

## 关键术语

| 术语 | 大家口中的说法 | 实际含义 |
|------|-----------------|-----------------------|
| 帧（Frame） | 一个切片 | 送入一次 FFT 的 25 毫秒波形片段。 |
| 帧移（Hop） | 步长 | 相邻帧之间间隔的采样数；ASR 默认 10 毫秒。 |
| 窗（Window） | Hann/Hamming 那个东西 | 逐点相乘的系数，使帧的边缘渐变到零。 |
| STFT | 频谱图生成器 | 分帧加窗后的 FFT；产出时间 × 频率矩阵。 |
| 梅尔（Mel） | 扭曲后的频率 | 对数感知刻度；`m = 2595·log10(1 + f/700)`。 |
| 滤波器组（Filterbank） | 那个矩阵 | 把 STFT 投影到梅尔频带的三角滤波器。 |
| Log-mel | Whisper 的输入 | `log(mel_spec + eps)`；2026 年的标准做法。 |
| MFCC | 老派特征 | 对数梅尔的 DCT；13 个系数，已去相关。 |

## 延伸阅读

- [Davis, Mermelstein (1980). Comparison of parametric representations for monosyllabic word recognition](https://ieeexplore.ieee.org/document/1163420) —— MFCC 的开山论文。
- [Stevens, Volkmann, Newman (1937). A Scale for the Measurement of the Psychological Magnitude Pitch](https://pubs.aip.org/asa/jasa/article-abstract/8/3/185/735757/) —— 梅尔刻度的原始文献。
- [OpenAI — Whisper source, log_mel_spectrogram](https://github.com/openai/whisper/blob/main/whisper/audio.py) —— 阅读参考实现。
- [librosa feature extraction docs](https://librosa.org/doc/main/feature.html) —— `mfcc`、`melspectrogram` 以及帧移/窗长的参考文档。
- [NVIDIA NeMo — audio preprocessing](https://docs.nvidia.com/deeplearning/nemo/user-guide/docs/en/main/asr/asr_all.html#featurizers) —— Parakeet 与 Canary 模型的生产级处理管线。
