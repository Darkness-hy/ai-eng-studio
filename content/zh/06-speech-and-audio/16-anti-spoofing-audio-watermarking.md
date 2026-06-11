# 语音反欺骗与音频水印 — ASVspoof 5、AudioSeal、WaveVerify

> 语音克隆的落地速度超过了防御手段。2026 年的生产级语音系统需要两样东西：一个区分真实语音与伪造语音的检测器（AASIST、RawNet2），以及一个能在压缩和编辑后依然存活的水印（AudioSeal）。两者都要交付，否则就不要上线语音克隆。

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 6 · 06 (Speaker Recognition), Phase 6 · 08 (Voice Cloning)
**Time:** ~75 minutes

## 问题背景

三种相关的防御手段：

1. **反欺骗 / 深度伪造检测（Anti-spoofing / deepfake detection）。** 给定一段音频，判断它是合成的还是真实的。ASVspoof 基准（ASVspoof 2019 → 2021 → 5）是这一领域的黄金标准。
2. **音频水印（Audio watermarking）。** 在生成的音频中嵌入一个不可感知的信号，供检测器日后提取。AudioSeal（Meta）和 WavMark 是开源选项。
3. **可认证的来源溯源（Authenticated provenance）。** 对音频文件和元数据进行密码学签名。C2PA / Content Authenticity Initiative。

检测应对的是不配合的对手，水印应对的是合规要求——AI 生成的音频应当可以被识别为 AI 生成。在 2026 年，两者缺一不可。

## 核心概念

![Anti-spoofing vs watermarking vs provenance — three defense layers](../assets/spoofing-watermark.svg)

### ASVspoof 5 — 2024-2025 基准

相比前几届最大的变化：

- **众包数据**（而非录音棚级干净数据）——更贴近真实条件。
- **约 2000 名说话人**（之前约 100 名）。
- **32 种攻击算法。** TTS + 语音转换 + 对抗扰动。
- **两条赛道。** 对策（Countermeasure，CM）赛道做独立检测；抗欺骗说话人验证（Spoofing-robust ASV，SASV）赛道面向生物识别系统。

ASVspoof 5 上的最先进水平：约 7.23% EER。在更早的 ASVspoof 2019 LA 上：0.42% EER。真实世界部署中，对野外采集的音频片段预期 5-10% 的 EER。

### AASIST 与 RawNet2 — 检测模型家族

**AASIST**（2021 年提出，持续更新至 2026 年）。在频谱特征上使用图注意力。是目前 ASVspoof 5 对策任务的 SOTA。

**RawNet2。** 在原始波形上使用卷积前端 + TDNN 主干。更简单的基线；经过微调后仍有竞争力。

**NeXt-TDNN + SSL 特征。** 2025 年的变体：ECAPA 风格结构 + WavLM 特征 + focal loss。在 ASVspoof 2019 LA 上达到 0.42% 的 EER。

### AudioSeal — 2024 年的水印默认选择

Meta 的 **AudioSeal**（2024 年 1 月发布，v0.2 于 2024 年 12 月发布）。关键设计：

- **局部化检测。** 在 16 kHz 采样分辨率（1/16000 秒）下逐帧检测水印。
- **生成器与检测器联合训练。** 生成器学习嵌入不可闻的信号；检测器学习在各种增强变换之后仍能找到它。
- **鲁棒。** 能在 MP3 / AAC 压缩、EQ、±10% 变速、+10 dB SNR 噪声混入后存活。
- **快速。** 检测器运行速度达实时的 485 倍，比 WavMark 快 1000 倍。
- **容量。** 每段语音可嵌入 16 比特载荷（可编码模型 ID、生成时间戳、用户 ID）。

### WavMark

AudioSeal 之前的开源基线。基于可逆神经网络，32 比特/秒。存在的问题：

- 同步需要暴力搜索，速度慢。
- 可被高斯噪声或 MP3 压缩去除。
- 不适合实时场景。

### WaveVerify（2025 年 7 月）

针对 AudioSeal 的弱点——尤其是时间维度的篡改（倒放、变速）。使用基于 FiLM 的生成器 + 混合专家（Mixture-of-Experts）检测器。在标准攻击上与 AudioSeal 相当；能应对时间维度的编辑。

### 对手会利用的缺口

来自 AudioMarkBench 的结论："在变调（pitch shift）攻击下，所有水印的比特恢复准确率（Bit Recovery Accuracy）都低于 0.6，意味着水印几乎被完全去除。" **变调是通用攻击。** 2026 年没有任何水印能完全抵抗激进的变调操作。这正是除了水印之外还需要检测（AASIST）的原因。

### C2PA / Content Authenticity Initiative

这不是一种 ML 技术，而是一种清单（manifest）格式。音频文件携带经过密码学签名的元数据，记录创建工具、作者、日期。Audobox / Seamless 在使用它。适合做来源溯源；但如果恶意行为者重新编码并剥离元数据，它就毫无作用。

## 从零实现

### 第 1 步：一个简单的频谱特征检测器（玩具版）

```python
def spectral_rolloff(spec, percentile=0.85):
    cum = 0
    total = sum(spec)
    if total == 0:
        return 0
    threshold = total * percentile
    for k, v in enumerate(spec):
        cum += v
        if cum >= threshold:
            return k
    return len(spec) - 1

def is_suspicious(audio):
    spec = magnitude_spectrum(audio)
    rolloff = spectral_rolloff(spec)
    return rolloff / len(spec) > 0.92
```

合成语音的高频能量往往异常平坦。生产环境的检测器用的是 AASIST，而不是这个玩具版，但直觉是一致的。

### 第 2 步：AudioSeal 嵌入与检测

```python
from audioseal import AudioSeal
import torch

generator = AudioSeal.load_generator("audioseal_wm_16bits")
detector = AudioSeal.load_detector("audioseal_detector_16bits")

audio = load_wav("generated.wav", sr=16000)[None, None, :]
payload = torch.tensor([[1, 0, 1, 1, 0, 1, 0, 0, 1, 1, 0, 1, 0, 1, 1, 0]])
watermark = generator.get_watermark(audio, sample_rate=16000, message=payload)
watermarked = audio + watermark

result, decoded_payload = detector.detect_watermark(watermarked, sample_rate=16000)
# result: float in [0, 1] — probability of watermark presence
# decoded_payload: 16 bits; match against embedded payload
```

### 第 3 步：评估 — EER

```python
def eer(real_scores, fake_scores):
    thresholds = sorted(set(real_scores + fake_scores))
    best = (1.0, 0.0)
    for t in thresholds:
        far = sum(1 for s in fake_scores if s >= t) / len(fake_scores)
        frr = sum(1 for s in real_scores if s < t) / len(real_scores)
        if abs(far - frr) < best[0]:
            best = (abs(far - frr), (far + frr) / 2)
    return best[1]
```

### 第 4 步：生产环境集成

```python
def safe_tts(text, voice, clone_reference=None):
    if clone_reference is not None:
        verify_consent(user_id, clone_reference)
    audio = tts_model.synthesize(text, voice)
    audio_with_wm = audioseal_embed(audio, payload=build_payload(user_id, model_id))
    manifest = c2pa_sign(audio_with_wm, user_id, timestamp=now())
    return audio_with_wm, manifest
```

每次生成都要同时交付：(1) 水印，(2) 签名清单，(3) 符合保留策略的审计日志。

## 生产实践

| 使用场景 | 防御手段 |
|----------|---------|
| 上线 TTS / 语音克隆 | 每个输出都嵌入 AudioSeal 水印（没有商量余地） |
| 声纹生物识别解锁 | AASIST + ECAPA 集成；活体挑战 |
| 呼叫中心欺诈检测 | 对 20% 的来电抽样运行 AASIST |
| 播客真实性验证 | 上传时做 C2PA 签名，AI 生成内容加 AudioSeal |
| 研究 / 训练检测器 | ASVspoof 5 的 train/dev/eval 数据集 |

## 常见陷阱

- **只打水印却从不运行检测器。** 毫无意义。把检测器放进你的 CI。
- **检测不做校准。** 在 ASVspoof LA 上训练的 AASIST 会过拟合，真实场景准确率会下降。要在你自己的领域数据上校准。
- **变调缺口。** 激进的变调能去除大多数水印。要准备检测作为兜底手段。
- **元数据剥离后重新发布。** C2PA 只需重新编码就能轻易绕过。务必同时部署密码学防御和感知层（水印）防御。
- **把活体检测当成伪造检测。** 让用户念一段随机短语。这能防住重放攻击，但防不住实时克隆。

## 交付产物

保存为 `outputs/skill-spoof-defender.md`。为一次语音生成系统的部署选定检测模型、水印方案、来源溯源清单和运维操作手册。

## 练习

1. **简单。** 运行 `code/main.py`。在合成音频上跑玩具检测器和玩具水印的嵌入/检测。
2. **中等。** 安装 `audioseal`，在一段 TTS 输出中嵌入 16 比特载荷并重新解码。给音频加入噪声破坏，测量比特恢复准确率。
3. **困难。** 在 ASVspoof 2019 LA 上微调一个 RawNet2 或 AASIST。测量 EER。再在一组留出的 F5-TTS 生成片段上测试——观察分布外（OOD）检测性能如何退化。

## 关键术语

| 术语 | 通俗说法 | 实际含义 |
|------|-----------------|-----------------------|
| ASVspoof | 那个基准 | 两年一届的挑战赛；2024 年这届即 ASVspoof 5。 |
| CM（对策） | 检测器 | 分类器：真实语音 vs 合成/转换语音。 |
| SASV | 说话人验证 + CM | 集成了生物识别与欺骗检测。 |
| AudioSeal | Meta 的水印 | 局部化检测、16 比特载荷、比 WavMark 快 485 倍。 |
| 比特恢复准确率（Bit Recovery Accuracy） | 水印存活率 | 攻击之后被成功恢复的载荷比特占比。 |
| C2PA | 来源溯源清单 | 关于创建过程/作者身份的密码学元数据。 |
| AASIST | 检测器家族 | 基于图注意力的反欺骗 SOTA。 |

## 延伸阅读

- [Todisco et al. (2024). ASVspoof 5](https://dl.acm.org/doi/10.1016/j.csl.2025.101825) — 当前的基准。
- [Defossez et al. (2024). AudioSeal](https://arxiv.org/abs/2401.17264) — 水印的默认选择。
- [Chen et al. (2025). WaveVerify](https://arxiv.org/abs/2507.21150) — 应对时间维度攻击的 MoE 检测器。
- [Jung et al. (2022). AASIST](https://arxiv.org/abs/2110.01200) — SOTA 检测主干。
- [AudioMarkBench (2024)](https://proceedings.neurips.cc/paper_files/paper/2024/file/5d9b7775296a641a1913ab6b4425d5e8-Paper-Datasets_and_Benchmarks_Track.pdf) — 鲁棒性评估。
- [C2PA specification](https://c2pa.org/specifications/specifications/) — 来源溯源清单格式。
