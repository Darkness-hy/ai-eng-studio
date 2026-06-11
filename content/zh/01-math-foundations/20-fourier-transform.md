# 傅里叶变换

> 任何信号都是正弦波的叠加。傅里叶变换告诉你叠加的是哪些正弦波。

**Type:** Build
**Language:** Python
**Prerequisites:** Phase 1, Lessons 01-04, 19 (complex numbers)
**Time:** ~90 minutes

## 学习目标

- 从零实现 DFT，并用 O(N log N) 的 Cooley-Tukey FFT 验证其正确性
- 解读频率系数：从信号中提取幅度、相位和功率谱
- 运用卷积定理，通过 FFT 乘法完成卷积运算
- 把傅里叶频率分解与 Transformer 的位置编码、CNN 的卷积层联系起来

## 问题背景

一段音频录音是随时间变化的气压测量序列。一支股票的价格是按天排列的数值序列。一张图像是空间上的像素强度网格。这些都是时域（time domain，或空间域）数据：你看到的是数值随某个索引变化。

但许多模式在时域里是看不见的。这段音频是单音还是和弦？这支股票有没有周线周期？这张图像是否有重复纹理？这些问题都关乎频率成分（frequency content），而时域把它们藏了起来。

傅里叶变换把数据从时域转换到频域（frequency domain）。它把一个信号分解成不同频率的正弦波。每个正弦波有一个幅度（amplitude，表示强度）和一个相位（phase，表示起始位置）。傅里叶变换会同时给出这两者。

这对机器学习很重要，因为频域思维无处不在。卷积神经网络做的是卷积，而卷积在频域中就是乘法。Transformer 的位置编码用频率分解来表示位置。音频模型（语音识别、音乐生成）以声谱图——声音的频率表示——为输入。时间序列模型寻找的是周期模式。理解傅里叶变换，你就掌握了处理所有这些问题的通用语言。

## 核心概念

### DFT 的定义

给定 N 个采样点 x[0], x[1], ..., x[N-1]，离散傅里叶变换（Discrete Fourier Transform, DFT）产生 N 个频率系数 X[0], X[1], ..., X[N-1]：

```
X[k] = sum_{n=0}^{N-1} x[n] * e^(-2*pi*i*k*n/N)

for k = 0, 1, ..., N-1
```

每个 X[k] 都是一个复数。它的模 |X[k]| 表示频率 k 的幅度，它的辐角 angle(X[k]) 表示该频率的相位偏移。

关键洞察：`e^(-2*pi*i*k*n/N)` 是一个以频率 k 旋转的相量（phasor）。DFT 计算的是信号与 N 个等间隔频率各自的相关性。如果信号在频率 k 上有能量，相关值就大；如果没有，相关值就接近零。

### 每个系数的含义

**X[0]：直流分量（DC component）。** 它是所有采样点之和——与均值成正比，代表信号的常数（零频率）偏移。

```
X[0] = sum_{n=0}^{N-1} x[n] * e^0 = sum of all samples
```

**1 <= k <= N/2 时的 X[k]：正频率。** X[k] 表示每 N 个采样点内振荡 k 个周期的频率。k 越大，频率越高（振荡越快）。

**X[N/2]：奈奎斯特频率（Nyquist frequency）。** 这是 N 个采样点所能表示的最高频率。超过它就会出现混叠（aliasing）——高频伪装成低频。

**N/2 < k < N 时的 X[k]：负频率。** 对于实值信号，X[N-k] = conj(X[k])。负频率是正频率的镜像。这就是为什么有用信息都集中在前 N/2 + 1 个系数中。

### 逆 DFT

逆 DFT 从频率系数重建原始信号：

```
x[n] = (1/N) * sum_{k=0}^{N-1} X[k] * e^(2*pi*i*k*n/N)

for n = 0, 1, ..., N-1
```

与正向 DFT 的唯一区别是：指数中的符号为正（不是负），并且多了一个 1/N 的归一化因子。

逆 DFT 是完美重建——不丢失任何信息。你可以从时域到频域再返回，没有任何误差。DFT 本质上是一次基变换（change of basis）：它把同样的信息在另一个坐标系下重新表达。

### FFT：让它变快

按上述定义计算 DFT 的复杂度是 O(N^2)：N 个输出系数，每个都要对 N 个输入采样点求和。当 N = 100 万时，需要 10^12 次运算。

快速傅里叶变换（Fast Fourier Transform, FFT）用 O(N log N) 计算出同样的结果。当 N = 100 万时，只需约 2000 万次运算，而不是一万亿次。正是它让频率分析变得切实可行。

Cooley-Tukey 算法（最常用的 FFT）采用分治法：

1. 把信号拆分成偶数索引和奇数索引两组采样点。
2. 递归计算两半各自的 DFT。
3. 用「旋转因子」（twiddle factor）e^(-2*pi*i*k/N) 把两个半长 DFT 合并起来。

```
X[k] = E[k] + e^(-2*pi*i*k/N) * O[k]          for k = 0, ..., N/2 - 1
X[k + N/2] = E[k] - e^(-2*pi*i*k/N) * O[k]    for k = 0, ..., N/2 - 1

where E = DFT of even-indexed samples
      O = DFT of odd-indexed samples
```

由于这种对称性，每层递归只做 O(N) 的工作，而递归深度为 log2(N) 层。总复杂度：O(N log N)。

```mermaid
graph TD
    subgraph "8-point FFT (Cooley-Tukey)"
        X["x[0..7]<br/>8 samples"] -->|"split even/odd"| E["Even: x[0,2,4,6]"]
        X -->|"split even/odd"| O["Odd: x[1,3,5,7]"]
        E -->|"4-pt FFT"| EK["E[0..3]"]
        O -->|"4-pt FFT"| OK["O[0..3]"]
        EK -->|"combine with twiddle factors"| XK["X[0..7]"]
        OK -->|"combine with twiddle factors"| XK
    end
    subgraph "Complexity"
        C1["DFT: O(N^2) = 64 multiplications"]
        C2["FFT: O(N log N) = 24 multiplications"]
    end
```

FFT 要求信号长度为 2 的幂。实践中通常把信号补零到下一个 2 的幂长度。

### 频谱分析

**功率谱（power spectrum）**是 |X[k]|^2——每个频率系数模的平方。它显示各频率上有多少能量。

**相位谱（phase spectrum）**是 angle(X[k])——每个频率的相位偏移。在大多数分析任务中，你只关心功率谱而忽略相位。

```
Power at frequency k:  P[k] = |X[k]|^2 = X[k].real^2 + X[k].imag^2
Phase at frequency k:  phi[k] = atan2(X[k].imag, X[k].real)
```

### 频率分辨率

DFT 的频率分辨率取决于采样点数 N 和采样率 fs。

```
Frequency of bin k:      f_k = k * fs / N
Frequency resolution:    delta_f = fs / N
Maximum frequency:       f_max = fs / 2  (Nyquist)
```

要分辨两个相距很近的频率，需要更多采样点；要捕捉高频，需要更高的采样率。

### 卷积定理

这是信号处理中最重要的结论之一，并且与 CNN 直接相关。

**时域中的卷积等于频域中的逐点乘法。**

```
x * h = IFFT(FFT(x) . FFT(h))

where * is convolution and . is element-wise multiplication
```

为什么它重要：

- 直接对长度为 N 和 M 的两个信号做卷积需要 O(N*M) 次运算。
- 基于 FFT 的卷积只需 O(N log N)：先变换两者，再相乘，最后逆变换回来。
- 对于大核（kernel），FFT 卷积快得多。
- 这正是大感受野卷积层中发生的事情。

注意：DFT 计算的是循环卷积（circular convolution，信号会环绕回卷）。要得到线性卷积（无环绕），需先把两个信号都补零到长度 N + M - 1 再计算。

```mermaid
graph LR
    subgraph "Time Domain"
        TA["Signal x[n]"] -->|"convolve (slow: O(NM))"| TC["Output y[n]"]
        TB["Filter h[n]"] -->|"convolve"| TC
    end
    subgraph "Frequency Domain"
        FA["FFT(x)"] -->|"multiply (fast: O(N))"| FC["FFT(x) * FFT(h)"]
        FB["FFT(h)"] -->|"multiply"| FC
        FC -->|"IFFT"| FD["y[n]"]
    end
    TA -.->|"FFT"| FA
    TB -.->|"FFT"| FB
    FD -.->|"same result"| TC
```

### 加窗

DFT 假设信号是周期性的——它把 N 个采样点视为一个无限重复信号的单个周期。如果信号的起点和终点取值不同，边界处就会出现不连续，表现为虚假的高频成分。这种现象称为频谱泄漏（spectral leakage）。

加窗（windowing）通过在计算 DFT 之前把信号两端逐渐压低到零来减少泄漏。

常用窗函数：

| 窗函数 | 形状 | 主瓣宽度 | 旁瓣电平 | 适用场景 |
|--------|-------|----------------|-----------------|----------|
| 矩形窗 | 平坦（不加窗） | 最窄 | 最高（-13 dB） | 信号恰好在 N 个采样点内完整周期时 |
| Hann | 升余弦 | 中等 | 低（-31 dB） | 通用频谱分析 |
| Hamming | 修正余弦 | 中等 | 更低（-42 dB） | 音频处理、语音分析 |
| Blackman | 三重余弦 | 宽 | 极低（-58 dB） | 旁瓣抑制至关重要时 |

```
Hann window:    w[n] = 0.5 * (1 - cos(2*pi*n / (N-1)))
Hamming window: w[n] = 0.54 - 0.46 * cos(2*pi*n / (N-1))
```

加窗的方法是在 DFT 之前把窗函数与信号逐元素相乘：`X = DFT(x * w)`。

### DFT 的性质

| 性质 | 时域 | 频域 |
|----------|-------------|-----------------|
| 线性 | a*x + b*y | a*X + b*Y |
| 时移 | x[n - k] | X[f] * e^(-2*pi*i*f*k/N) |
| 频移 | x[n] * e^(2*pi*i*f0*n/N) | X[f - f0] |
| 卷积 | x * h | X * H（逐点） |
| 乘法 | x * h（逐点） | X * H（循环卷积，乘以 1/N） |
| Parseval 定理 | sum \|x[n]\|^2 | (1/N) * sum \|X[k]\|^2 |
| 共轭对称（实数输入） | x[n] 为实数 | X[k] = conj(X[N-k]) |

Parseval 定理表明两个域中的总能量相同。能量在变换过程中守恒。

### 与位置编码的联系

原始 Transformer 使用正弦位置编码：

```
PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
```

每个维度对 (2i, 2i+1) 以不同频率振荡。这些频率从高（维度 0,1）到低（最后几个维度）按几何级数分布。这让每个位置在所有频段上都拥有独一无二的模式——就像傅里叶系数能唯一标识一个信号。

它提供的关键性质：

- **唯一性：** 没有两个位置拥有相同的编码。
- **有界取值：** sin 和 cos 始终在 [-1, 1] 范围内。
- **相对位置：** 位置 p+k 的编码可以表示为位置 p 的编码的线性函数。模型可以学会关注相对位置。

### 与 CNN 的联系

卷积层把一个学到的滤波器（卷积核）在信号或图像上滑动来作用于输入。在数学上，这就是卷积运算。

根据卷积定理，这等价于：
1. 对输入做 FFT
2. 对卷积核做 FFT
3. 在频域相乘
4. 对结果做 IFFT

标准的 CNN 实现使用直接卷积（对 3x3 这样的小核更快）。但对于大核或全局卷积，基于 FFT 的方法明显更快。一些架构（如 FNet）干脆用 FFT 完全取代注意力，以 O(N log N) 而非 O(N^2) 的复杂度达到了有竞争力的精度。

### 声谱图与短时傅里叶变换

单次 FFT 给出整个信号的频率成分，但完全不告诉你这些频率出现在何时。一段啁啾信号（chirp，频率随时间升高的信号）和一个和弦（所有频率同时存在）可能有完全相同的幅度谱。

短时傅里叶变换（Short-Time Fourier Transform, STFT）通过在信号的重叠窗口上分别计算 FFT 来解决这个问题。结果就是声谱图（spectrogram）：一个二维表示，一个轴是时间，另一个轴是频率。每个点的强度表示该时刻在该频率上的能量。

```
STFT procedure:
1. Choose a window size (e.g., 1024 samples)
2. Choose a hop size (e.g., 256 samples -- 75% overlap)
3. For each window position:
   a. Extract the windowed segment
   b. Apply a Hann/Hamming window
   c. Compute FFT
   d. Store the magnitude spectrum as one column of the spectrogram
```

声谱图是音频机器学习模型的标准输入表示。语音识别模型（Whisper、DeepSpeech）以梅尔声谱图（mel-spectrogram）为输入——即把频率映射到梅尔刻度的声谱图，这种刻度更符合人耳对音高的感知。

### 混叠

如果信号包含高于 fs/2（奈奎斯特频率）的频率，以采样率 fs 采样就会产生混叠副本。一个 90 Hz 的信号以 100 Hz 采样后，看起来与 10 Hz 的信号完全一样。仅凭采样点无法区分二者。

```
Example:
  True signal: 90 Hz sine wave
  Sampling rate: 100 Hz
  Apparent frequency: 100 - 90 = 10 Hz

  The samples from the 90 Hz signal at 100 Hz sampling rate
  are identical to the samples from a 10 Hz signal.
  No amount of math can recover the original 90 Hz.
```

这就是为什么模数转换器（ADC）会内置抗混叠滤波器，在采样前去除高于奈奎斯特频率的成分。在机器学习中，混叠出现在没有恰当低通滤波就对特征图做下采样的场景——一些架构用抗混叠池化层来解决这个问题。

### 补零不会提高分辨率

一个常见误解：在 FFT 之前对信号补零可以提高频率分辨率。事实并非如此。补零只是在现有的频率 bin 之间做插值，让频谱看起来更平滑，但它无法揭示原始采样中本就不存在的频率细节。

真正的频率分辨率只取决于观测时长 T = N / fs。要分辨相距 delta_f 的两个频率，至少需要 T = 1 / delta_f 秒的数据。再多的补零也改变不了这个根本限制。

```figure
fourier-synthesis
```

## 从零实现

### 第 1 步：从零实现 DFT

O(N^2) 的 DFT 直接照搬定义即可。

```python
import math

class Complex:
    ...

def dft(x):
    N = len(x)
    result = []
    for k in range(N):
        total = Complex(0, 0)
        for n in range(N):
            angle = -2 * math.pi * k * n / N
            w = Complex(math.cos(angle), math.sin(angle))
            xn = x[n] if isinstance(x[n], Complex) else Complex(x[n])
            total = total + xn * w
        result.append(total)
    return result
```

### 第 2 步：逆 DFT

结构相同，指数取正号，再除以 N。

```python
def idft(X):
    N = len(X)
    result = []
    for n in range(N):
        total = Complex(0, 0)
        for k in range(N):
            angle = 2 * math.pi * k * n / N
            w = Complex(math.cos(angle), math.sin(angle))
            total = total + X[k] * w
        result.append(Complex(total.real / N, total.imag / N))
    return result
```

### 第 3 步：FFT（Cooley-Tukey）

递归 FFT 要求长度为 2 的幂。拆分成偶、奇两组，递归计算，再用旋转因子合并。

```python
def fft(x):
    N = len(x)
    if N <= 1:
        return [x[0] if isinstance(x[0], Complex) else Complex(x[0])]
    if N % 2 != 0:
        return dft(x)

    even = fft([x[i] for i in range(0, N, 2)])
    odd = fft([x[i] for i in range(1, N, 2)])

    result = [Complex(0)] * N
    for k in range(N // 2):
        angle = -2 * math.pi * k / N
        twiddle = Complex(math.cos(angle), math.sin(angle))
        t = twiddle * odd[k]
        result[k] = even[k] + t
        result[k + N // 2] = even[k] - t
    return result
```

### 第 4 步：频谱分析辅助函数

```python
def power_spectrum(X):
    return [xk.real ** 2 + xk.imag ** 2 for xk in X]

def convolve_fft(x, h):
    N = len(x) + len(h) - 1
    padded_N = 1
    while padded_N < N:
        padded_N *= 2

    x_padded = x + [0.0] * (padded_N - len(x))
    h_padded = h + [0.0] * (padded_N - len(h))

    X = fft(x_padded)
    H = fft(h_padded)

    Y = [xk * hk for xk, hk in zip(X, H)]

    y = idft(Y)
    return [y[n].real for n in range(N)]
```

## 生产实践

实际工作中应使用 numpy 的 FFT，其底层是高度优化的 C 库。

```python
import numpy as np

signal = np.sin(2 * np.pi * 5 * np.arange(256) / 256)
spectrum = np.fft.fft(signal)
freqs = np.fft.fftfreq(256, d=1/256)

power = np.abs(spectrum) ** 2

positive_freqs = freqs[:len(freqs)//2]
positive_power = power[:len(power)//2]
```

加窗和更高级的频谱分析：

```python
from scipy.signal import windows, stft

window = windows.hann(256)
windowed = signal * window
spectrum = np.fft.fft(windowed)
```

卷积：

```python
from scipy.signal import fftconvolve

result = fftconvolve(signal, kernel, mode='full')
```

声谱图：

```python
from scipy.signal import stft

frequencies, times, Zxx = stft(signal, fs=sample_rate, nperseg=256)
spectrogram = np.abs(Zxx) ** 2
```

声谱图矩阵的形状为 (n_frequencies, n_time_frames)。每一列是一个时间窗口的功率谱。这就是音频机器学习模型实际接收的输入。

## 交付产物

运行 `code/fourier.py`，生成 `outputs/prompt-spectral-analyzer.md`。

## 练习

1. **纯音识别。** 创建一个只含单一正弦波的信号，频率未知（在 1 到 50 Hz 之间），以 128 Hz 采样 1 秒。用你的 DFT 识别出该频率，并验证答案正确。然后加入标准差为 0.5 的高斯噪声并重复实验。噪声对频谱有什么影响？

2. **FFT 与 DFT 对照验证。** 生成一个长度为 64 的随机信号。同时计算 DFT（O(N^2)）和 FFT。验证所有系数的误差都在 1e-10 以内。再在长度为 256、512、1024 和 2048 的信号上分别给两个函数计时。绘制 DFT 耗时与 FFT 耗时的比值曲线。

3. **用实例验证卷积定理。** 创建信号 x = [1, 2, 3, 4, 0, 0, 0, 0] 和滤波器 h = [1, 1, 1, 0, 0, 0, 0, 0]。先直接计算它们的循环卷积（嵌套循环），再通过 FFT 计算（变换、相乘、逆变换）。验证结果一致。然后通过适当补零完成线性卷积。

4. **加窗效果。** 创建一个由 10 Hz 和 12 Hz（非常接近）两个正弦波叠加而成的信号，以 128 Hz 采样 1 秒。分别在不加窗、加 Hann 窗、加 Hamming 窗的情况下计算功率谱。哪种窗最容易分辨两个峰？为什么？

5. **位置编码分析。** 生成 d_model = 128、max_pos = 512 的正弦位置编码。对每一对位置 (p1, p2)，计算二者编码的点积。证明点积只取决于 |p1 - p2|，与绝对位置无关。随着距离增大，点积会发生什么变化？

## 关键术语

| 术语 | 含义 |
|------|---------------|
| DFT（离散傅里叶变换） | 把 N 个时域采样点转换为 N 个频域系数。每个系数是信号与该频率复正弦的相关值 |
| FFT（快速傅里叶变换） | 计算 DFT 的 O(N log N) 算法。Cooley-Tukey 算法递归地按偶/奇索引拆分 |
| 逆 DFT | 从频率系数重建时域信号。公式与 DFT 相同，但指数符号取反并乘以 1/N |
| 频率 bin | DFT 输出中的每个索引 k 代表频率 k*fs/N Hz。「bin」即离散频率槽位 |
| 直流分量 | X[0]，零频率系数，与信号均值成正比 |
| 奈奎斯特频率 | fs/2，采样率为 fs 时可表示的最高频率。高于此频率会发生混叠 |
| 功率谱 | \|X[k]\|^2，每个频率系数模的平方。显示能量在各频率上的分布 |
| 相位谱 | angle(X[k])，每个频率分量的相位偏移。分析中通常被忽略 |
| 频谱泄漏 | 把非周期信号当作周期信号处理而产生的虚假频率成分。可通过加窗减少 |
| 窗函数 | 在 DFT 之前应用的渐变收窄函数（Hann、Hamming、Blackman），用于减少频谱泄漏 |
| 旋转因子 | FFT 蝶形运算中用来合并子 DFT 的复指数 e^(-2*pi*i*k/N) |
| 卷积定理 | 时域卷积等于频域逐点乘法。信号处理和 CNN 的基础 |
| 循环卷积 | 信号首尾环绕的卷积。这是 DFT 天然计算的卷积形式 |
| 线性卷积 | 无环绕的标准卷积。通过在 DFT 之前补零实现 |
| Parseval 定理 | 总能量在傅里叶变换中守恒。sum \|x[n]\|^2 = (1/N) sum \|X[k]\|^2 |
| 混叠 | 采样率不足时，高于奈奎斯特频率的成分表现为更低的频率 |

## 延伸阅读

- [Cooley & Tukey: An Algorithm for the Machine Calculation of Complex Fourier Series (1965)](https://www.ams.org/journals/mcom/1965-19-090/S0025-5718-1965-0178586-1/) - 改变了整个计算领域的 FFT 原始论文
- [3Blue1Brown: But what is the Fourier Transform?](https://www.youtube.com/watch?v=spUNpyF58BY) - 最好的傅里叶变换可视化入门
- [Lee-Thorp et al.: FNet: Mixing Tokens with Fourier Transforms (2021)](https://arxiv.org/abs/2105.03824) - 在 Transformer 中用 FFT 取代自注意力
- [Smith: The Scientist and Engineer's Guide to Digital Signal Processing](http://www.dspguide.com/) - 免费在线教材，深入讲解 FFT、加窗和频谱分析
- [Vaswani et al.: Attention Is All You Need (2017)](https://arxiv.org/abs/1706.03762) - 源自傅里叶频率分解的正弦位置编码
- [Radford et al.: Whisper (2022)](https://arxiv.org/abs/2212.04356) - 以梅尔声谱图为输入表示的语音识别
