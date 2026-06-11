# 音频评估 — WER、MOS、UTMOS、MMAU、FAD 与公开排行榜

> 无法度量的东西就无法上线。本课梳理 2026 年各类音频任务的标准指标：ASR（WER、CER、RTFx）、TTS（MOS、UTMOS、SECS、ASR 回环 WER）、音频语言模型（MMAU、LongAudioBench）、音乐（FAD、CLAP）以及说话人（EER），并介绍可供横向比较的排行榜。

**Type:** Learn
**Languages:** Python
**Prerequisites:** Phase 6 · 04, 06, 07, 09, 10; Phase 2 · 09 (Model Evaluation)
**Time:** ~60 minutes

## 问题背景

每类音频任务都有多个指标，各自衡量不同的维度。用错指标，就会出现模型在你的仪表盘上光鲜亮丽、上线后却一塌糊涂的情况。2026 年的标准指标清单：

| 任务 | 主要指标 | 次要指标 |
|------|---------|-----------|
| ASR | WER | CER · RTFx · 首 token 延迟 |
| TTS | MOS / UTMOS | SECS · ASR 回环 WER · CER · TTFA |
| 声音克隆 | SECS（ECAPA 余弦相似度） | MOS · CER |
| 说话人验证 | EER | minDCF · 工作点上的 FAR / FRR |
| 说话人日志 | DER | JER · 说话人混淆 |
| 音频分类 | top-1 · mAP | 宏平均 F1 · 各类别召回率 |
| 音乐生成 | FAD | CLAP · 听评小组 MOS |
| 音频语言模型 | MMAU-Pro | LongAudioBench · AudioCaps FENSE |
| 流式语音到语音 | 延迟 P50/P95 | WER · MOS |

## 核心概念

![Audio evaluation matrix — metrics vs tasks vs 2026 leaderboards](../assets/eval-landscape.svg)

### ASR 指标

**WER（词错误率，Word Error Rate）。** `(S + D + I) / N`。评分前先转小写、去除标点、归一化数字。可使用 `jiwer` 或 OpenAI 的 `whisper_normalizer`。&lt; 5% 即达到朗读语音的人类水平。

**CER（字符错误率，Character Error Rate）。** 公式相同，但在字符级别计算。用于分词存在歧义的声调语言（普通话、粤语）。

**RTFx（实时率倒数，inverse real-time factor）。** 每个真实时钟秒能处理多少秒音频。越高越好。Parakeet-TDT 达到 3380×，Whisper-large-v3 约 30×。

**首 token 延迟。** 从音频输入到产出第一个转写 token 的真实时钟时间。对流式场景至关重要。Deepgram Nova-3 约 150 ms。

### TTS 指标

**MOS（平均意见分，Mean Opinion Score）。** 1-5 分人工评分。黄金标准但速度慢。每条样本需 20 名以上听评者，每个模型需 100 条以上样本。

**UTMOS（2022-2026）。** 学习得到的 MOS 预测器。在标准基准上与人工 MOS 的相关性约 0.9。F5-TTS：UTMOS 3.95；真实语音：4.08。

**SECS（说话人编码器余弦相似度，Speaker Encoder Cosine Similarity）。** 用于声音克隆。计算参考音频与克隆输出的 ECAPA 嵌入余弦相似度。&gt; 0.75 即克隆可辨认。

**ASR 回环 WER（WER-on-ASR-round-trip）。** 用 Whisper 转写 TTS 输出，再与输入文本计算 WER。能捕捉可懂度的退化。2026 年 SOTA：CER &lt; 2%。

**TTFA（首音频时间，time-to-first-audio）。** 真实时钟延迟。Kokoro-82M 约 100 ms；F5-TTS 约 1 s。

### 声音克隆专用

将 **SECS + MOS + CER** 作为三元组使用。SECS 高但 MOS 低，说明音色对了但听感不自然；反之则是声音自然但说话人不对。

### 说话人验证

**EER（等错误率，Equal Error Rate）。** 误接受率（False Accept Rate）与误拒绝率（False Reject Rate）相等时的阈值。ECAPA 在 VoxCeleb1-O 上为 0.87%。

**minDCF（最小检测代价，min Detection Cost）。** 在选定工作点（通常 FAR=0.01）上的加权代价。比 EER 更贴近生产需求。

### 说话人日志

**DER（日志错误率，Diarization Error Rate）。** `(FA + Miss + Confusion) / total_speaker_time`。漏检语音 + 误报语音 + 说话人混淆，各自占比相加。AMI 会议数据上 DER 约 10-20% 属于现实水平。pyannote 3.1 与 Precision-2 商业方案在录音质量良好的音频上可达 &lt;10% DER。

**JER（Jaccard 错误率，Jaccard Error Rate）。** DER 的替代方案，对短片段偏差更稳健。

### 音频分类

多标签任务：在所有类别上计算 **mAP（平均精度均值，mean Average Precision）**。AudioSet 上 BEATs-iter3 为 0.548 mAP。

互斥多分类任务：**top-1、top-5 准确率**。Speech Commands v2 上 99.0% top-1（Audio-MAE）。

类别不均衡：**宏平均 F1** + **各类别召回率**。要按类别分别报告——总体准确率会掩盖哪些类别在失效。

### 音乐生成

**FAD（Fréchet 音频距离，Fréchet Audio Distance）。** 真实音频与生成音频的 VGGish 嵌入分布之间的距离。MusicGen-small 在 MusicCaps 上为 4.5，MusicLM 为 4.0。越低越好。

**CLAP Score。** 基于 CLAP 嵌入的文本-音频对齐分数。&gt; 0.3 即对齐尚可。

**听评小组 MOS。** 仍是消费级音乐质量的最终裁决。Suno v5 在 TTS Arena 上 ELO 1293（来自成对人工偏好）。

### 音频语言模型基准

**MMAU（大规模多音频理解，Massive Multi-Audio Understanding）。** 1 万条音频问答对。

**MMAU-Pro。** 1800 道高难度题目，分四类：语音 / 声音 / 音乐 / 多音频。四选一的随机猜测基线为 25%。Gemini 2.5 Pro 总体约 60%；所有模型在多音频类别上都只有约 22%。

**LongAudioBench。** 数分钟长的音频片段配语义查询。Audio Flamingo Next 优于 Gemini 2.5 Pro。

**AudioCaps / Clotho。** 音频描述基准。指标为 SPICE、CIDEr、FENSE。

### 流式语音到语音

**延迟 P50 / P95 / P99。** 从用户说话结束到第一段可听响应的真实时钟时间。Moshi：200 ms；GPT-4o Realtime：300 ms。

**输出的 WER / MOS。**

**插话响应速度（barge-in responsiveness）。** 从用户打断到助手静音的时间。目标 &lt; 150 ms。

### 2026 年的排行榜

| 排行榜 | 赛道 | URL |
|------------|--------|-----|
| Open ASR Leaderboard (HF) | 英语 + 多语种 + 长音频 | `huggingface.co/spaces/hf-audio/open_asr_leaderboard` |
| TTS Arena (HF) | 英语 TTS | `huggingface.co/spaces/TTS-AGI/TTS-Arena` |
| Artificial Analysis Speech | TTS + STT，基于成对投票的 ELO | `artificialanalysis.ai/speech` |
| MMAU-Pro | LALM 推理 | `mmaubenchmark.github.io` |
| SpeakerBench / VoxSRC | 说话人识别 | `voxsrc.github.io` |
| MMAU 音乐子集 | 音乐 LALM | （包含在 MMAU 中） |
| HEAR benchmark | 自监督音频 | `hearbenchmark.com` |

## 从零实现

### 第 1 步：带归一化的 WER

```python
from jiwer import wer, Compose, ToLowerCase, RemovePunctuation, Strip

transform = Compose([ToLowerCase(), RemovePunctuation(), Strip()])
score = wer(
    truth="Please turn on the lights.",
    hypothesis="please turn on the light",
    truth_transform=transform,
    hypothesis_transform=transform,
)
# ~0.17
```

### 第 2 步：TTS 回环 WER

```python
def ttr_wer(tts_model, asr_model, texts):
    errors = []
    for txt in texts:
        audio = tts_model.synthesize(txt)
        recog = asr_model.transcribe(audio)
        errors.append(wer(truth=txt, hypothesis=recog))
    return sum(errors) / len(errors)
```

### 第 3 步：声音克隆的 SECS

```python
from speechbrain.inference.speaker import EncoderClassifier
sv = EncoderClassifier.from_hparams("speechbrain/spkrec-ecapa-voxceleb")

emb_ref = sv.encode_batch(load_wav("reference.wav"))
emb_clone = sv.encode_batch(load_wav("cloned.wav"))
secs = torch.nn.functional.cosine_similarity(emb_ref, emb_clone, dim=-1).item()
```

### 第 4 步：音乐生成的 FAD

```python
from frechet_audio_distance import FrechetAudioDistance
fad = FrechetAudioDistance()
score = fad.get_fad_score("generated_folder/", "reference_folder/")
```

### 第 5 步：说话人验证的 EER（与第 6 课代码相同）

```python
def eer(same_scores, diff_scores):
    thresholds = sorted(set(same_scores + diff_scores))
    best = (1.0, 0.0)
    for t in thresholds:
        far = sum(1 for s in diff_scores if s >= t) / len(diff_scores)
        frr = sum(1 for s in same_scores if s < t) / len(same_scores)
        if abs(far - frr) < best[0]:
            best = (abs(far - frr), (far + frr) / 2)
    return best[1]
```

## 生产实践

每次部署都要搭配一套固定的评估流水线，在每次模型更新时运行。三条铁律：

1. **评分前先归一化。** 转小写、去标点、展开数字，并报告所用的归一化规则。
2. **报告分布，而不是平均值。** 延迟报 P50/P95/P99；分类报各类别召回率；MMAU 按类别报告。
3. **跑一个权威公开基准。** 即使你的生产数据与之不同，在 Open ASR / TTS Arena / MMAU 上报告结果能让评审者进行公平比较。

## 常见陷阱

- **UTMOS 外推失效。** 它在 VCTK 风格的干净语音上训练，对带噪、克隆或情感化音频打分很不可靠。
- **MOS 评审小组偏差。** 20 名 Amazon Mechanical Turk 众包工人不等于 20 名目标用户。事关重大时请付费组建领域内的听评小组。
- **FAD 依赖参考集。** 跨模型比较时必须使用同一参考分布。
- **聚合 WER。** 总体 5% 的 WER 可能掩盖带口音语音上 30% 的 WER。按人群切片分别报告。
- **公开基准饱和。** 大多数前沿模型在标准基准上已接近天花板。请构建一个能反映你真实流量的内部留出集。

## 交付产物

保存为 `outputs/skill-audio-evaluator.md`。为任意音频模型发布选定指标、基准与报告格式。

## 练习

1. **简单。** 运行 `code/main.py`。在玩具输入上计算 WER / CER / EER / SECS / 类 FAD / 类 MMAU 指标。
2. **中等。** 搭建一套 TTS 回环 WER 流水线。把你的 Kokoro 或 F5-TTS 输出送入 Whisper，在 50 条提示词上计算 WER，标记出 WER &gt; 10% 的提示词。
3. **困难。** 用第 10 课选定的 LALM 在 MMAU-Pro 的语音与多音频子集（各 50 题）上评测。报告各类别准确率，并与公开发布的数字对比。

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|-----------------|-----------------------|
| WER | ASR 分数 | 归一化后在词级别计算的 `(S+D+I)/N`。 |
| CER | 字符级 WER | 用于声调语言或字符级系统。 |
| MOS | 人工意见分 | 1-5 评分；20 名以上听评者 × 100 条样本。 |
| UTMOS | 机器学习 MOS 预测器 | 学习得到的模型；与人工 MOS 相关性约 0.9。 |
| SECS | 声音克隆相似度 | 参考音频与克隆音频的 ECAPA 余弦相似度。 |
| EER | 说话人验证分数 | FAR = FRR 时的阈值。 |
| DER | 说话人日志分数 | (FA + Miss + Confusion) / total。 |
| FAD | 音乐生成质量 | VGGish 嵌入上的 Fréchet 距离。 |
| RTFx | 吞吐量 | 每个真实时钟秒处理的音频秒数。 |

## 延伸阅读

- [jiwer](https://github.com/jitsi/jiwer) — 带归一化工具的 WER/CER 库。
- [UTMOS (Saeki et al. 2022)](https://arxiv.org/abs/2204.02152) — 学习得到的 MOS 预测器。
- [Fréchet Audio Distance (Kilgour et al. 2019)](https://arxiv.org/abs/1812.08466) — 音乐生成的标准指标。
- [Open ASR Leaderboard](https://huggingface.co/spaces/hf-audio/open_asr_leaderboard) — 2026 年实时排名。
- [TTS Arena](https://huggingface.co/spaces/TTS-AGI/TTS-Arena) — 基于人工投票的 TTS 排行榜。
- [MMAU-Pro benchmark](https://mmaubenchmark.github.io/) — LALM 推理排行榜。
- [HEAR benchmark](https://hearbenchmark.com/) — 音频自监督学习基准。
