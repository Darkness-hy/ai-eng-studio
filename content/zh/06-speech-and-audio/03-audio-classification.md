# 音频分类 —— 从 MFCC 上的 k-NN 到 AST 与 BEATs

> 从"狗叫还是警笛"到"这是哪种语言"，这些都是音频分类。特征是梅尔谱，架构每十年换一轮，而评估始终是 AUC、F1 和逐类召回率。

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 6 · 02 (Spectrograms & Mel), Phase 3 · 06 (CNNs), Phase 5 · 08 (CNNs & RNNs for Text)
**Time:** ~75 minutes

## 问题背景

给你一段 10 秒的音频，你想知道："这是什么？"城市声音（警笛、电钻、狗叫）、语音指令（yes/no/stop）、语种识别（en/es/ar）、说话人情绪（愤怒/中性），或环境声音（室内/室外、人声嘈杂）。这些全都属于*音频分类*。到 2026 年，基线架构已经成熟：log-mel → CNN 或 Transformer → softmax。

核心难点不在网络本身，而在数据。音频数据集存在严重的类别不平衡、强烈的域偏移（干净 vs 嘈杂）以及标签噪声（谁来界定"城市人声嘈杂"和"餐厅噪声"的区别？）。这个问题 80% 的工作量在数据整理、增强和评估上，而不是把 CNN 换成 Transformer。

## 核心概念

![Audio classification ladder: k-NN on MFCCs to AST to BEATs](../assets/audio-classification.svg)

**MFCC 上的 k-NN（1990 年代的基线）。** 把每段音频的 MFCC 展平，与一个带标签的样本库计算余弦相似度，对前 K 个近邻做多数投票。在干净的小数据集（Speech Commands、ESC-50）上效果出奇地好，而且不需要 GPU。

**log-mel 上的 2D CNN（2015-2019）。** 把 `(T, n_mels)` 的 log-mel 当作图像处理，套用 ResNet-18 或 VGG 风格的网络，对时间轴做全局平均池化，再对类别做 softmax。在 2026 年的多数 kaggle 比赛中，它仍是基线。

**音频频谱 Transformer，即 AST（2021-2024）。** 把 log-mel 切成图块（patch，例如 16×16），加上位置嵌入，输入 ViT。在监督学习设定下是 AudioSet 上的最先进水平（mAP 0.485）。

**BEATs 与 WavLM-base（2024-2026）。** 在数百万小时的音频上做自监督预训练，再在你的任务上微调，所需的监督数据只有原来的 1-10%。到 2026 年，这是非语音音频任务的默认起点。BEATs-iter3 在 AudioSet 上比 AST 高 1-2 个 mAP，而算力只用 1/4。

**把 Whisper 编码器当作冻结骨干网络（2024）。** 取 Whisper 的编码器，丢掉解码器，接一个线性分类器。在语种识别和简单事件分类上接近 SOTA，且完全不需要音频增强。这是"免费午餐"式的基线。

### 类别不平衡才是真正的挑战

ESC-50：50 个类别，每类 40 段音频——均衡，简单。UrbanSound8K：10 个类别，不平衡度 10:1。AudioSet：632 个类别，长尾比例达 100,000:1。有效的技术包括：

- 训练时做均衡采样（评估时不要做）。
- Mixup：把两段音频（及其标签）线性插值，作为数据增强。
- SpecAugment：随机遮盖时间带和频率带。简单，但至关重要。

### 评估

- 多类互斥（Speech Commands）：top-1 准确率、top-5 准确率。
- 多类多标签（AudioSet、UrbanSound 类型）：平均精度均值（mAP）。
- 严重不平衡：逐类召回率 + 宏平均 F1。

你应该记住的 2026 年数字：

| 基准 | 基线 | 2026 年 SOTA | 来源 |
|-----------|----------|-----------|--------|
| ESC-50 | 82%（AST） | 97.0%（BEATs-iter3） | BEATs 论文（2024） |
| AudioSet mAP | 0.485（AST） | 0.548（BEATs-iter3） | HEAR 排行榜 2026 |
| Speech Commands v2 | 98%（CNN） | 99.0%（Audio-MAE） | HEAR v2 结果 |

## 从零实现

### 第 1 步：特征提取

```python
def featurize_mfcc(signal, sr, n_mfcc=13, n_mels=40, frame_len=400, hop=160):
    mag = stft_magnitude(signal, frame_len, hop)
    fb = mel_filterbank(n_mels, frame_len, sr)
    mels = apply_filterbank(mag, fb)
    log = log_transform(mels)
    return [dct_ii(frame, n_mfcc) for frame in log]
```

### 第 2 步：定长摘要

```python
def summarize(mfcc_frames):
    n = len(mfcc_frames[0])
    mean = [sum(f[i] for f in mfcc_frames) / len(mfcc_frames) for i in range(n)]
    var = [
        sum((f[i] - mean[i]) ** 2 for f in mfcc_frames) / len(mfcc_frames) for i in range(n)
    ]
    return mean + var
```

简单但有效：对时间维取均值 + 方差，13 系数的 MFCC 就得到一个 26 维的定长嵌入。计算瞬间完成。直到 2017 年，这种方法在 ESC-50 上还能击败当时最先进的神经网络基线。

### 第 3 步：k-NN

```python
def cosine(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a)) or 1e-12
    nb = math.sqrt(sum(x * x for x in b)) or 1e-12
    return dot / (na * nb)

def knn_classify(q, bank, labels, k=5):
    sims = sorted(range(len(bank)), key=lambda i: -cosine(q, bank[i]))[:k]
    votes = Counter(labels[i] for i in sims)
    return votes.most_common(1)[0][0]
```

### 第 4 步：升级为 log-mel 上的 CNN

用 PyTorch 实现：

```python
import torch.nn as nn

class AudioCNN(nn.Module):
    def __init__(self, n_mels=80, n_classes=50):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.head = nn.Linear(128, n_classes)

    def forward(self, x):  # x: (B, 1, T, n_mels)
        return self.head(self.body(x).flatten(1))
```

300 万参数。单张 RTX 4090 上约 10 分钟即可在 ESC-50 上训完，准确率 80% 以上。

### 第 5 步：2026 年的默认做法 —— 微调 BEATs

```python
from transformers import ASTFeatureExtractor, ASTForAudioClassification

ext = ASTFeatureExtractor.from_pretrained("MIT/ast-finetuned-audioset-10-10-0.4593")
model = ASTForAudioClassification.from_pretrained(
    "MIT/ast-finetuned-audioset-10-10-0.4593",
    num_labels=50,
    ignore_mismatched_sizes=True,
)

inputs = ext(audio, sampling_rate=16000, return_tensors="pt")
logits = model(**inputs).logits
```

至于 BEATs，可通过 `beats` 库使用 `microsoft/BEATs-base`；transformers 的 API 形态完全一样。

## 生产实践

2026 年的技术选型：

| 场景 | 起步方案 |
|-----------|-----------|
| 极小数据集（<1000 段） | MFCC 均值上的 k-NN（你的基线）+ 音频增强 |
| 中等数据集（1K–100K） | 微调 BEATs 或 AST |
| 大数据集（>100K） | 从头训练，或微调 Whisper 编码器 |
| 实时、边缘端 | 40-MFCC CNN，量化到 int8（KWS 风格） |
| 多标签（AudioSet） | BEATs-iter3 + BCE 损失 + mixup + SpecAugment |
| 语种识别 | MMS-LID，SpeechBrain VoxLingua107 基线 |

决策准则：**从冻结的骨干网络起步，而不是从头训练新模型**。微调一个 BEATs 分类头，只需几小时（而不是几周）就能达到 SOTA 的 95%。

## 交付产物

保存为 `outputs/skill-classifier-designer.md`。针对给定的音频分类任务，选定架构、增强方法、类别均衡策略和评估指标。

## 练习

1. **简单。** 运行 `code/main.py`。它在一个 4 类合成数据集（不同音高的纯音）上训练 k-NN MFCC 基线。报告混淆矩阵。
2. **中等。** 把 `summarize` 替换为 [mean, var, skew, kurtosis]。在同一个合成数据集上，四阶矩池化能否胜过均值+方差？
3. **困难。** 使用 `torchaudio`，在 ESC-50 的 fold 1 上训练一个 2D CNN。报告 5 折交叉验证准确率。加入 SpecAugment（时间掩码 = 20，频率掩码 = 10）并报告前后差异。

## 关键术语

| 术语 | 大家怎么说 | 实际含义 |
|------|-----------------|-----------------------|
| AudioSet | 音频界的 ImageNet | Google 的弱标注 YouTube 数据集，200 万段音频、632 个类别。 |
| ESC-50 | 小型分类基准 | 环境声音，50 个类别 × 每类 40 段音频。 |
| AST | 音频频谱 Transformer | 在 log-mel 图块上运行的 ViT；2021 年的 SOTA。 |
| BEATs | 自监督音频 | Microsoft 的模型，截至 2026 年 iter3 在 AudioSet 上领先。 |
| Mixup | 成对增强 | `x = λ·x1 + (1-λ)·x2; y = λ·y1 + (1-λ)·y2`。 |
| SpecAugment | 基于掩码的增强 | 把频谱图中随机的时间带和频率带置零。 |
| mAP | 多标签任务的主要指标 | 跨类别和阈值的平均精度均值。 |

## 延伸阅读

- [Gong, Chung, Glass (2021). AST: Audio Spectrogram Transformer](https://arxiv.org/abs/2104.01778) —— 2021–2024 年间的标志性架构。
- [Chen et al. (2022, rev. 2024). BEATs: Audio Pre-Training with Acoustic Tokenizers](https://arxiv.org/abs/2212.09058) —— 2024 年以后的默认选择。
- [Park et al. (2019). SpecAugment](https://arxiv.org/abs/1904.08779) —— 占主导地位的音频增强方法。
- [Piczak (2015). ESC-50 dataset](https://github.com/karolpiczak/ESC-50) —— 经久不衰的 50 类基准。
- [Gemmeke et al. (2017). AudioSet](https://research.google.com/audioset/) —— 632 类的 YouTube 分类体系；仍是黄金标准。
