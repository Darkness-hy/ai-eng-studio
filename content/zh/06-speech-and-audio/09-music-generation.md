# 音乐生成——MusicGen、Stable Audio、Suno 与版权大地震

> 2026 年的音乐生成：商业领域由 Suno v5 和 Udio v4 主导；开源领域由 MusicGen、Stable Audio Open 和 ACE-Step 领跑。技术问题已基本解决，而法律问题（Warner Music 5 亿美元和解、UMG 和解）在 2025-2026 年重塑了整个领域。

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 6 · 02 (Spectrograms), Phase 4 · 10 (Diffusion Models)
**Time:** ~75 minutes

## 问题背景

文本 → 一段 30 秒到 4 分钟、带歌词、人声和曲式结构的音乐。这可以拆成三个子问题：

1. **纯器乐生成。** 输入 "lo-fi hip-hop drums with warm keys" 这样的文本 → 输出音频。代表：MusicGen、Stable Audio、AudioLDM。
2. **歌曲生成（含人声 + 歌词）。** "Country song about rainy Texas nights" → 一首完整的歌。代表：Suno、Udio、YuE、ACE-Step。
3. **条件 / 可控生成。** 续写已有片段、重新生成一段桥段、切换曲风、分离音轨（stem），或做局部重绘（inpaint）。Udio 的 inpainting + 音轨分离是 2026 年大家都在对标的功能。

## 核心概念

![Music generation: token-LM vs diffusion, the 2026 model map](../assets/music-generation.svg)

### 基于神经编解码器 token 的 token 语言模型

Meta 的 **MusicGen**（2023，MIT 协议）及众多衍生模型：以文本/旋律嵌入（embedding）为条件，自回归地预测 EnCodec token（32 kHz，4 个码本），再用 EnCodec 解码。参数量 300M - 3.3B。是很强的基线，但超过 30 秒就力不从心。

**ACE-Step**（开源，4B XL 于 2026 年 4 月发布）将这一路线扩展到以歌词为条件的整曲生成，是开源社区最接近 Suno 的方案。

### 基于梅尔谱或潜空间的扩散模型

**Stable Audio（2023）** 和 **Stable Audio Open（2024）**：在压缩音频上做潜空间扩散。擅长循环乐段（loop）、声音设计、氛围音色，但不擅长结构完整的整首歌曲。

**AudioLDM / AudioLDM2**：用类似文生图（T2I）的潜空间扩散实现文本到音频，并推广到音乐、音效、语音。

### 混合架构（生产级）——Suno、Udio、Lyria

权重闭源。大概率是自回归编解码器语言模型 + 基于扩散的声码器，并配有专门的人声 / 鼓 / 旋律头。Suno v5（2026）是 ELO 1293 的质量领跑者；Udio v4 新增 inpainting + 音轨分离（贝斯、鼓、人声可分别下载）。

### 评估

- **FAD（Fréchet Audio Distance，弗雷歇音频距离）。** 用 VGGish 或 PANNs 特征衡量生成音频与真实音频分布之间的嵌入级距离，越低越好。MusicGen small 在 MusicCaps 上为 4.5 FAD；SOTA 约 3.0。
- **音乐性（主观）。** 人类偏好。Suno v5 以 ELO 1293 领先。
- **文本-音频对齐。** 用提示词与输出之间的 CLAP 分数衡量。
- **音乐性瑕疵。** 不在拍点上的过渡、人声乐句漂移、超过 30 秒后曲式结构丢失。

## 2026 年模型版图

| 模型 | 参数量 | 时长 | 人声 | 许可证 |
|-------|--------|--------|--------|---------|
| MusicGen-large | 3.3B | 30 秒 | 无 | MIT |
| Stable Audio Open | 1.2B | 47 秒 | 无 | Stability 非商业 |
| ACE-Step XL（2026 年 4 月） | 4B | &gt; 2 分钟 | 有 | Apache-2.0 |
| YuE | 7B | &gt; 2 分钟 | 有，多语言 | Apache-2.0 |
| Suno v5（闭源） | ? | 4 分钟 | 有，ELO 1293 | 商业 |
| Udio v4（闭源） | ? | 4 分钟 | 有 + 音轨分离 | 商业 |
| Google Lyria 3（闭源） | ? | 实时 | 有 | 商业 |
| MiniMax Music 2.5 | ? | 4 分钟 | 有 | 商业 API |

## 法律格局（2025-2026）

- **Warner Music 诉 Suno 和解。** 金额 5 亿美元。WMG 由此获得对 Suno 上 AI 仿声形象、音乐版权及用户生成曲目的监督权。Udio 与 UMG 也达成了类似和解。
- **EU AI Act** + **加州 SB 942** 法案：AI 生成的音乐必须明确披露。
- **Riffusion / MusicGen** 采用 MIT 协议，没有合规包袱，但也没有可商用的人声。

可以放心上线的几种模式：

1. 只生成纯器乐（MusicGen、Stable Audio Open，MIT/CC0 输出）。
2. 使用商业 API（Suno、Udio、ElevenLabs Music），按次获得生成许可。
3. 在自有或已获授权的曲库上训练（大多数企业最终走这条路）。
4. 给生成结果打上水印 + 元数据标签。

## 从零实现

### 第 1 步：用 MusicGen 生成

```python
from audiocraft.models import MusicGen
import torchaudio

model = MusicGen.get_pretrained("facebook/musicgen-small")
model.set_generation_params(duration=10)
wav = model.generate(["upbeat synthwave with driving drums, 128 BPM"])
torchaudio.save("out.wav", wav[0].cpu(), 32000)
```

三个尺寸：`small`（300M，快）、`medium`（1.5B）、`large`（3.3B）。验证"这个想法是否成立"用 small 就够了。

### 第 2 步：旋律条件生成

```python
melody, sr = torchaudio.load("humming.wav")
wav = model.generate_with_chroma(
    ["jazz piano cover"],
    melody.squeeze(),
    sr,
)
```

MusicGen-melody 接收一个色度图（chromagram），在替换音色的同时保留旋律。适合"把这段旋律改成弦乐四重奏"这类需求。

### 第 3 步：FAD 评估

```python
from frechet_audio_distance import FrechetAudioDistance
fad = FrechetAudioDistance()

fad.get_fad_score("generated_folder/", "reference_folder/")
```

计算 VGGish 嵌入距离。适合做曲风层面的回归测试，但不能替代真人试听。

### 第 4 步：接入 LLM-音乐工作流

结合第 7-8 课的思路：

```python
prompt = "Write a 30-second jazz loop. Describe the drums, bass, and piano voicing."
description = llm.complete(prompt)
music = musicgen.generate([description], duration=30)
```

## 生产实践

| 目标 | 技术栈 |
|------|-------|
| 器乐声音设计 | Stable Audio Open |
| 游戏 / 自适应音乐 | Google Lyria RealTime（闭源） |
| 带人声的完整歌曲（商用） | Suno v5 或 Udio v4，并取得明确许可 |
| 带人声的完整歌曲（开源） | ACE-Step XL 或 YuE |
| 短广告配乐 | MusicGen 以哼唱参考做旋律条件生成 |
| 音乐视频背景 | MusicGen + Stable Video Diffusion |

## 2026 年仍会上线的坑

- **洗版权式提示词。** "Song in the style of Taylor Swift"——商业版 Suno/Udio 现在会过滤这类提示词，开源模型不会。请自行维护一份过滤列表。
- **超过 30 秒的重复 / 漂移。** 自回归模型会陷入循环。可以把多次生成结果交叉淡化拼接，或改用结构连贯性更好的 ACE-Step。
- **速度漂移。** 模型会偏离 BPM。在提示词里加 BPM 标签，并用 librosa 的 `beat_track` 做后置过滤。
- **人声清晰度。** Suno 表现出色，开源模型的咬字常常含糊。如果歌词很重要，请用商业 API 或自己微调。
- **单声道输出。** 开源模型生成的是单声道或伪立体声。用真正的立体声重建方案升级（ezst、Cartesia 的立体声扩散）。

## 交付产物

保存为 `outputs/skill-music-designer.md`。为一次音乐生成部署选定模型、许可证策略、时长 / 曲式结构方案以及披露元数据。

## 练习

1. **简单。** 运行 `code/main.py`。它会以 ASCII 符号输出一段"生成式"的和弦进行 + 鼓点节奏——一个音乐生成的卡通版。愿意的话可以用任意 MIDI 渲染器回放。
2. **中等。** 安装 `audiocraft`，用 MusicGen-small 在 4 种曲风提示词下各生成 10 秒片段，并对照一个参考曲风集合测量 FAD。
3. **困难。** 用 ACE-Step（或 MusicGen-melody），以不同音色提示词为同一段旋律生成三个变体，再计算与提示词的 CLAP 相似度以验证对齐程度。

## 关键术语

| 术语 | 大家怎么说 | 实际含义 |
|------|-----------------|-----------------------|
| FAD | 音频版 FID | 真实音频与生成音频嵌入分布之间的弗雷歇距离。 |
| Chromagram | 用音高表示的旋律 | 每帧 12 维的向量；旋律条件生成的输入。 |
| Stems | 乐器分轨 | 分离出的贝斯 / 鼓 / 人声 / 旋律，以 WAV 形式存放。 |
| Inpainting | 重生成某一段 | 遮住一个时间窗口，模型只重新生成这一段。 |
| CLAP | 文本-音频版 CLIP | 对比式音频-文本嵌入；用于评估文本-音频对齐。 |
| EnCodec | 音乐编解码器 | MusicGen 使用的 Meta 神经编解码器；32 kHz，4 个码本。 |

## 延伸阅读

- [Copet et al. (2023). MusicGen](https://arxiv.org/abs/2306.05284) ——开源自回归方案的标杆。
- [Evans et al. (2024). Stable Audio Open](https://arxiv.org/abs/2407.14358) ——声音设计的默认选择。
- [ACE-Step](https://github.com/ace-step/ACE-Step) ——开源 4B 整曲生成器，2026 年 4 月发布。
- [Suno v5 platform docs](https://suno.com) ——商业领域的质量领跑者。
- [AudioLDM2](https://arxiv.org/abs/2308.05734) ——面向音乐 + 音效的潜空间扩散。
- [WMG-Suno settlement coverage](https://www.musicbusinessworldwide.com/suno-warner-music-settlement/) ——2025 年 11 月的判例先声。
