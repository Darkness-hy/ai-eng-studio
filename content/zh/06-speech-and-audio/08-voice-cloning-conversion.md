# 声音克隆与语音转换

> 声音克隆（voice cloning）用别人的声音朗读你的文字；语音转换（voice conversion）把你的声音改写成别人的声音，同时保留你说的内容。两者依赖同一个分解思路：把说话人身份与语音内容分离开。

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 6 · 06 (Speaker Recognition), Phase 6 · 07 (TTS)
**Time:** ~75 minutes

## 问题背景

在 2026 年，只需一段 5 秒的音频，用一块消费级 GPU 就能高质量地克隆任何人的声音。ElevenLabs、F5-TTS、OpenVoice v2、VoiceBox 都已提供零样本（zero-shot）或少样本（few-shot）克隆能力。这项技术既是福音（无障碍 TTS、配音、辅助发声），也是武器（诈骗电话、政治深度伪造、知识产权盗用）。

两个密切相关的任务：

- **声音克隆（TTS 侧）：** 文本 + 5 秒参考语音 → 用该声音朗读的音频。
- **语音转换（语音侧）：** 源音频（A 说了内容 X）+ B 的参考语音 → B 说内容 X 的音频。

两者都把波形分解为（内容、说话人、韵律），再把一个来源的内容与另一个来源的说话人重新组合。

你在 2026 年交付产品时必须遵守的关键约束：**水印与同意门控在欧盟（AI Act，2026 年 8 月起强制执行）和加州（AB 2905，2025 年生效）已成为法律要求**。你的流水线必须嵌入不可闻的水印，并拒绝未经同意的克隆请求。

## 核心概念

![Voice cloning vs conversion: factorize, swap speaker, recombine](../assets/voice-cloning.svg)

**零样本克隆。** 把一段 5 秒的音频交给一个在数千名说话人数据上训练过的模型。说话人编码器把音频映射为说话人嵌入；TTS 解码器以该嵌入加文本为条件生成语音。

采用者：F5-TTS（2024）、YourTTS（2022）、XTTS v2（2024）、OpenVoice v2（2024）。

**少样本微调。** 录制 5-30 分钟的目标声音，用 LoRA 对基座模型微调一小时。质量从"还行"跃升到"难以分辨"。Coqui 和 ElevenLabs 都支持这种模式；社区也常用它配合 F5-TTS。

**语音转换（VC）。** 两大流派：

- **识别-合成（recognition-synthesis）。** 先用类 ASR 模型提取内容表示（如软音素后验、PPG），再结合目标说话人嵌入重新合成。对语言和口音都很鲁棒。采用者：KNN-VC（2023）、Diff-HierVC（2023）。
- **解耦（disentanglement）。** 训练一个自编码器，在瓶颈处的潜空间中把内容、说话人、韵律分离开，推理时替换说话人嵌入。质量略低但速度更快。采用者：AutoVC（2019）、各类 VITS-VC 变体。

**基于神经编解码器的克隆（2024+）。** VALL-E、VALL-E 2、NaturalSpeech 3、VoiceBox——把音频视为来自 SoundStream / EnCodec 的离散 token，在编解码器 token 上训练大型自回归或流匹配（flow-matching）模型。在短提示音条件下质量可与 ElevenLabs 比肩。

### 伦理是核心环节，不是事后补丁

**水印。** PerTh（Perth）和 SilentCipher（2024）能在音频中以不可察觉的方式嵌入约 16-32 比特的 ID，可经受重编码、流式传输和常见编辑。已有生产级开源实现。

**同意门控。** 每一段克隆输出都必须配有可验证的同意记录。"我，Rohit，于 2026-04-22 授权将此声音用于 X 用途。"存入防篡改日志。

**检测。** AASIST、RawNet2 和 Wav2Vec2-AASIST 已作为检测器发布。ASVspoof 2025 挑战赛公布的数据显示，最先进的检测器面对 ElevenLabs、VALL-E 2 和 Bark 的输出时，EER 为 0.8–2.3%。

### 关键数据（2026）

| 模型 | 零样本？ | SECS（目标相似度） | WER（可懂度） | 参数量 |
|-------|-----------|--------------------|--------------|--------|
| F5-TTS | 是 | 0.72 | 2.1% | 335M |
| XTTS v2 | 是 | 0.65 | 3.5% | 470M |
| OpenVoice v2 | 是 | 0.70 | 2.8% | 220M |
| VALL-E 2 | 是 | 0.77 | 2.4% | 370M |
| VoiceBox | 是 | 0.78 | 2.1% | 330M |

对大多数听众而言，SECS > 0.70 时克隆声音通常与目标声音难以区分。

## 从零实现

### 第 1 步：用识别-合成思路做分解（main.py 中的纯代码演示）

```python
def clone_pipeline(ref_audio, text, target_embedder, tts_model):
    speaker_emb = target_embedder.encode(ref_audio)
    mel = tts_model(text, speaker=speaker_emb)
    return vocoder(mel)
```

概念上很简单；实现的重头都在 `tts_model` 和说话人编码器里。

### 第 2 步：用 F5-TTS 做零样本克隆

```python
from f5_tts.api import F5TTS
tts = F5TTS()
wav = tts.infer(
    ref_file="rohit_5s.wav",
    ref_text="The quick brown fox jumps over the lazy dog.",
    gen_text="Please add milk and bread to my list.",
)
```

参考文本必须与参考音频完全一致；不匹配会破坏对齐。

### 第 3 步：用 KNN-VC 做语音转换

```python
import torch
from knnvc import KNNVC  # 2023 model, https://github.com/bshall/knn-vc
vc = KNNVC.load("wavlm-base-plus")
out_wav = vc.convert(source="my_voice.wav", target_pool=["alice_1.wav", "alice_2.wav"])
```

KNN-VC 用 WavLM 为源音频和目标音频池提取逐帧嵌入，然后把每一帧源音频替换为池中的最近邻帧。非参数化方法，只需一分钟的目标语音即可工作。

### 第 4 步：嵌入水印

```python
from silentcipher import SilentCipher
sc = SilentCipher(model="2024-06-01")
payload = b"consent_id:abc123;ts:1745353200"
watermarked = sc.embed(wav, sr=24000, message=payload)
detected = sc.detect(watermarked, sr=24000)   # returns payload bytes
```

约 32 比特的载荷，经 MP3 重编码和轻度噪声后仍可检出。

### 第 5 步：同意门控

```python
def cloned_inference(text, ref_audio, consent_record):
    assert verify_signature(consent_record), "Signed consent required"
    assert consent_record["speaker_id"] == hash_speaker(ref_audio)
    wav = tts.infer(ref_file=ref_audio, gen_text=text)
    wav = watermark(wav, payload=consent_record["id"])
    return wav
```

## 生产实践

2026 年的技术选型：

| 场景 | 选型 |
|-----------|------|
| 5 秒零样本克隆，开源 | F5-TTS 或 OpenVoice v2 |
| 商业生产级克隆 | ElevenLabs Instant Voice Clone v2.5 |
| 语音转换（改写声音） | KNN-VC 或 Diff-HierVC |
| 多说话人微调 | StyleTTS 2 + 说话人适配器 |
| 跨语言克隆 | XTTS v2 或 VALL-E X |
| 深度伪造检测 | Wav2Vec2-AASIST |

## 常见陷阱

- **参考文本与音频不对齐。** F5-TTS 及同类模型要求参考文本与参考音频完全一致，标点也不能差。
- **参考音频有混响。** 回声会毁掉克隆效果。请在干燥声学环境下近距离录音。
- **情绪不匹配。** 用"欢快"的参考音频训练，克隆出来的所有内容都是欢快的。参考音频的情绪要与目标用途匹配。
- **语言泄漏。** 克隆一位英语说话人后让模型说法语，往往仍带英语口音；请使用跨语言模型（XTTS、VALL-E X）。
- **没有水印。** 2026 年 8 月起在欧盟将无法合法上线。

## 交付产物

保存为 `outputs/skill-voice-cloner.md`。设计一条克隆或转换流水线，包含同意门控 + 水印 + 质量目标。

## 练习

1. **简单。** 运行 `code/main.py`。它通过计算替换前后两个"说话人"嵌入之间的余弦相似度，演示说话人嵌入替换的效果。
2. **中等。** 用 OpenVoice v2 克隆你自己的声音。测量参考音频与克隆音频之间的 SECS，并用 Whisper 测量 CER。
3. **困难。** 对 20 段克隆音频施加 SilentCipher 水印，经过 128 kbps MP3 编码再解码，然后检测载荷。报告比特准确率。

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|-----------------|-----------------------|
| 零样本克隆 | 5 秒就够了 | 预训练模型 + 说话人嵌入；无需训练。 |
| PPG | 音素后验图 | 逐帧 ASR 后验，用作与语言无关的内容表示。 |
| KNN-VC | 最近邻转换 | 把每一帧源音频替换为目标池中的最近邻帧。 |
| 神经编解码器 TTS | VALL-E 风格 | 在 EnCodec/SoundStream token 上的自回归模型。 |
| 水印 | 不可闻的签名 | 嵌入音频中的比特，可经受重编码。 |
| SECS | 克隆保真度 | 目标与克隆说话人嵌入之间的余弦相似度。 |
| AASIST | 深度伪造检测器 | 反欺骗模型；检测合成语音。 |

## 延伸阅读

- [Chen et al. (2024). F5-TTS](https://arxiv.org/abs/2410.06885) —— 开源 SOTA 零样本克隆。
- [Baevski et al. / Microsoft (2023). VALL-E](https://arxiv.org/abs/2301.02111) 与 [VALL-E 2 (2024)](https://arxiv.org/abs/2406.05370) —— 神经编解码器 TTS。
- [Qian et al. (2019). AutoVC](https://arxiv.org/abs/1905.05879) —— 基于解耦的语音转换。
- [Baas, Waubert de Puiseau, Kamper (2023). KNN-VC](https://arxiv.org/abs/2305.18975) —— 基于检索的 VC。
- [SilentCipher (2024) — Audio Watermarking](https://github.com/sony/silentcipher) —— 生产级 32 比特音频水印。
- [ASVspoof 2025 results](https://www.asvspoof.org/) —— 检测器与合成器的军备竞赛，2026 年更新。
