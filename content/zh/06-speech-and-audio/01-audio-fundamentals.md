# 音频基础 — 波形、采样与傅里叶变换

> 波形是原始信号，频谱图是表示形式，梅尔特征是适合机器学习的形态。每一条现代 ASR 和 TTS 流水线都要走过这条阶梯，而第一级台阶就是理解采样和傅里叶变换。

**Type:** Learn
**Languages:** Python
**Prerequisites:** Phase 1 · 06 (Vectors & Matrices), Phase 1 · 14 (Probability Distributions)
**Time:** ~45 minutes

## 问题背景

麦克风产生的是「声压-时间」信号，而你的神经网络消费的是张量。两者之间隔着一整套约定，一旦违反就会产生静默的 bug：模型训练一切正常但 WER 翻倍，或者 TTS 上线后带着嘶嘶声，或者声音克隆系统记住的是麦克风而不是说话人。

语音系统里的每一个 bug 最终都能追溯到这三个问题之一：

1. 数据是以什么采样率录制的，模型又期望什么采样率？
2. 信号有没有发生混叠？
3. 你操作的是原始采样点，还是频域表示？

把这三点搞对，Phase 6 剩下的内容就不难处理；搞错了，就算是 Whisper-Large-v4 也只会输出垃圾。

## 核心概念

![Waveform, sampling, DFT, and frequency bins visualized](../assets/audio-fundamentals.svg)

**波形（Waveform）。** 一个取值在 `[-1.0, 1.0]` 区间的一维浮点数组，以采样点编号为索引。换算成秒数时除以采样率即可：`t = n / sr`。一段 16 kHz 的 10 秒音频就是一个包含 160,000 个浮点数的数组。

**采样率（Sampling rate, sr）。** 每秒采集多少个样本。2026 年的常见采样率：

| 采样率 | 用途 |
|------|-----|
| 8 kHz | 电话、传统 VOIP。奈奎斯特频率只有 4 kHz，辅音信息被砍掉。做 ASR 时要避开。 |
| 16 kHz | ASR 标准。Whisper、Parakeet、SeamlessM4T v2 都消费 16 kHz 音频。 |
| 22.05 kHz | 旧模型的 TTS 声码器训练。 |
| 24 kHz | 现代 TTS（Kokoro、F5-TTS、xTTS v2）。 |
| 44.1 kHz | CD 音频、音乐。 |
| 48 kHz | 电影、专业音频、高保真 TTS（VALL-E 2、NaturalSpeech 3）。 |

**奈奎斯特-香农采样定理（Nyquist-Shannon）。** 采样率为 `sr` 时，能够无歧义表示的最高频率是 `sr/2`。这条 `sr/2` 边界就是*奈奎斯特频率（Nyquist frequency）*。高于奈奎斯特频率的能量会发生*混叠（aliasing）*——被折叠到更低的频率上——从而污染信号。降采样之前务必先做低通滤波。

**位深（Bit depth）。** 16 位 PCM（有符号 int16，范围 ±32,767）是通用交换格式；音乐用 24 位，内部 DSP 用 32 位浮点。`soundfile` 之类的库读取 int16 后会暴露为 `[-1, 1]` 区间的 float32 数组。

**傅里叶变换（Fourier Transform）。** 任何有限信号都可以分解为不同频率正弦波的和。离散傅里叶变换（DFT）对 `N` 个采样点计算出 `N` 个复数系数——每个频率 bin 一个。`bin k` 对应的频率是 `k · sr / N` Hz。模长是该频率上的幅度，辐角是相位。

**FFT。** 快速傅里叶变换（Fast Fourier Transform）：当 `N` 是 2 的幂时，一种 `O(N log N)` 复杂度的 DFT 算法。所有音频库底层都用 FFT。在 16 kHz 下做 1024 点 FFT，能得到 512 个可用频率 bin，覆盖 0–8 kHz，分辨率为 15.6 Hz。

**分帧 + 加窗（Framing + window）。** 我们不会对整段音频做 FFT，而是把它切成相互重叠的*帧*（通常帧长 25 ms、帧移 10 ms），将每帧乘以一个窗函数（Hann、Hamming）以消除边缘的不连续，然后对每帧做 FFT。这就是短时傅里叶变换（STFT）。第 02 课会从这里继续。

```figure
mel-scale
```

## 从零实现

### 第 1 步：读取音频并绘制波形

`code/main.py` 只使用标准库的 `wave` 模块，以保持示例零依赖。生产环境中你会用 `soundfile` 或 `torchaudio.load`（两者都返回 `(waveform, sr)` 元组）：

```python
import soundfile as sf
waveform, sr = sf.read("clip.wav", dtype="float32")  # shape (T,), sr=int
```

### 第 2 步：从第一性原理合成正弦波

```python
import math

def sine(freq_hz, sr, seconds, amp=0.5):
    n = int(sr * seconds)
    return [amp * math.sin(2 * math.pi * freq_hz * i / sr) for i in range(n)]
```

在 16 kHz 下合成 1 秒的 440 Hz 正弦波（音乐会标准音 A），就是 16,000 个浮点数。用 `wave.open(..., "wb")` 以 16 位 PCM 编码写出。

### 第 3 步：手写 DFT

```python
def dft(x):
    N = len(x)
    out = []
    for k in range(N):
        re = sum(x[n] * math.cos(-2 * math.pi * k * n / N) for n in range(N))
        im = sum(x[n] * math.sin(-2 * math.pi * k * n / N) for n in range(N))
        out.append((re, im))
    return out
```

复杂度 `O(N²)`——在 `N=256` 时用来验证正确性没问题，但对真实音频毫无用处。真实代码会调用 `numpy.fft.rfft` 或 `torch.fft.rfft`。

### 第 4 步：找出主导频率

幅度峰值的索引 `k_star` 对应频率 `k_star * sr / N`。对 440 Hz 正弦波运行这一步，峰值应该出现在 bin `440 * N / sr` 处。

### 第 5 步：演示混叠

用 10 kHz 采样率（奈奎斯特频率 = 5 kHz）去采样一个 7 kHz 的正弦波。7 kHz 音调高于奈奎斯特频率，会被折叠到 `10 − 7 = 3 kHz`，FFT 峰值出现在 3 kHz 处。这是经典的混叠演示，也是每个 DAC/ADC 都内置陡峭低通滤波器（brick-wall low-pass filter）的原因。

## 生产实践

2026 年你真正会上线的技术栈：

| 任务 | 库 | 原因 |
|------|---------|-----|
| 读写 WAV/FLAC/OGG | `soundfile`（libsndfile 封装） | 最快、稳定、返回 float32。 |
| 重采样 | `torchaudio.transforms.Resample` 或 `librosa.resample` | 内置正确的抗混叠处理。 |
| STFT / 梅尔特征 | `torchaudio` 或 `librosa` | 对 GPU 友好；属于 PyTorch 生态。 |
| 实时流式处理 | `sounddevice` 或 `pyaudio` | 跨平台的 PortAudio 绑定。 |
| 检查文件信息 | `ffprobe` 或 `soxi` | 命令行工具，速度快，能报告采样率/声道/编解码器。 |

决策准则：**先对齐采样率，再考虑其他一切**。Whisper 期望 16 kHz 单声道 float32。给它喂 44.1 kHz 立体声，你得到的垃圾输出看起来就像模型本身出了 bug。

## 交付产物

保存为 `outputs/skill-audio-loader.md`。这个 skill 帮你检查音频输入是否符合下游模型的预期，并在不符合时正确地重采样。

## 练习

1. **简单。** 在 16 kHz 下合成 1 秒的 220 Hz + 440 Hz + 880 Hz 混合信号。运行 DFT，确认三个峰值出现在预期的 bin 上。
2. **中等。** 以 48 kHz 录制一段 3 秒的人声 WAV。分别用 `torchaudio.transforms.Resample`（带抗混叠）降采样到 16 kHz，以及用朴素抽取法（每隔三个采样点取一个）降到 16 kHz。对两者做 FFT。混叠出现在哪里？
3. **困难。** 只用 `math` 和第 3 步的 DFT，从零实现 STFT。帧长 400，帧移 160，Hann 窗。用 `matplotlib.pyplot.imshow` 绘制幅度。这就是第 02 课要讲的频谱图。

## 关键术语

| 术语 | 大家口中的说法 | 实际含义 |
|------|-----------------|-----------------------|
| 采样率（Sample rate） | 每秒多少个样本 | ADC 测量信号的频率，单位 Hz。 |
| 奈奎斯特频率（Nyquist） | 能表示的最高频率 | `sr/2`；高于它的能量会混叠折返。 |
| 位深（Bit depth） | 每个采样点的分辨率 | `int16` = 65,536 个量化级；`float32` = `[-1, 1]` 区间内 24 位精度。 |
| DFT | 序列版的傅里叶变换 | `N` 个采样点 → `N` 个复数频率系数。 |
| FFT | 快速版 DFT | `O(N log N)` 算法，要求 `N` 为 2 的幂。 |
| 频率 bin（Bin） | 频率列 | `k · sr / N` Hz；分辨率 = `sr / N`。 |
| STFT | 频谱图的底层机制 | 在时间维上做分帧 + 加窗的 FFT。 |
| 混叠（Aliasing） | 诡异的频率幽灵 | 高于奈奎斯特频率的能量镜像折返到更低的 bin。 |

## 延伸阅读

- [Shannon (1949). Communication in the Presence of Noise](https://people.math.harvard.edu/~ctm/home/text/others/shannon/entropy/entropy.pdf) — 采样定理背后的那篇论文。
- [Smith — The Scientist and Engineer's Guide to Digital Signal Processing](https://www.dspguide.com/ch8.htm) — 免费的经典 DSP 教科书。
- [librosa docs — audio primer](https://librosa.org/doc/latest/tutorial.html) — 带代码的实战入门。
- [Heinrich Kuttruff — Room Acoustics (6th ed.)](https://www.routledge.com/Room-Acoustics/Kuttruff/p/book/9781482260434) — 解释真实世界的音频为何不是干净正弦波的参考书。
- [Steve Eddins — FFT Interpretation notebook](https://blogs.mathworks.com/steve/2020/03/30/fft-spectrum-and-spectral-densities/) — 10 分钟讲透频率 bin 的直觉。
