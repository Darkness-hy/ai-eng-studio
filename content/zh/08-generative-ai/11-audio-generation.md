# 音频生成

> 音频是采样率 16-48 kHz 的一维信号。一段五秒的音频就有 8 万到 24 万个采样点，没有任何 Transformer 能直接对这么长的序列做注意力计算。2026 年所有生产级音频模型的解决方案都一样：用神经编解码器（Encodec、SoundStream、DAC）把音频压缩成 50-75 Hz 的离散 token，再由 Transformer 或扩散模型来生成 token。

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 6 · 02 (Audio Features), Phase 6 · 04 (ASR), Phase 8 · 06 (DDPM)
**Time:** ~45 minutes

## 问题背景

音频生成有三类任务：

1. **文本转语音（Text-to-speech）。** 给定文本，生成语音。干净的语音是窄带信号，且具有很强的音素结构——基于 token 的 Transformer 已经很好地解决了这个问题。代表系统：VALL-E（Microsoft）、NaturalSpeech 3、ElevenLabs、OpenAI TTS。
2. **音乐生成。** 给定提示（文本、旋律、和弦进行、风格），生成音乐。数据分布宽得多。代表系统：MusicGen（Meta）、Stable Audio 2.5、Suno v4、Udio、Riffusion。
3. **音效 / 声音设计。** 给定提示，生成环境音或拟音（Foley）。代表系统：AudioGen、AudioLDM 2、Stable Audio Open。

这三类任务运行在同一套底座上：神经音频编解码器 + token 自回归或扩散生成器。

## 核心概念

![Audio generation: codec tokens + transformer or diffusion](../assets/audio-generation.svg)

### 神经音频编解码器

Encodec（Meta，2022）、SoundStream（Google，2021）、Descript Audio Codec（DAC，2023）。卷积编码器把波形压缩成每个时间步一个向量；残差向量量化（RVQ）把每个向量转换成 K 个级联的码本索引。解码器执行逆过程。24 kHz 音频在 2 kbps 码率下，使用 8 个 RVQ 码本、75 Hz 帧率 = 每秒 600 个 token。

```
waveform (16000 samples/sec)
    └─ encoder conv ─┐
                     ├─ RVQ layer 1 → indices at 75 Hz
                     ├─ RVQ layer 2 → indices at 75 Hz
                     ├─ ...
                     └─ RVQ layer 8
```

### 编解码器之上的两种生成范式

**Token 自回归。** 把 RVQ token 展平成一个序列，用 decoder-only Transformer 建模。MusicGen 使用「延迟并行（delayed parallel）」方案，让 K 路码本流以各自的偏移并行输出。VALL-E 则从文本提示 + 3 秒语音样本生成语音 token。

**潜空间扩散。** 把编解码器 token 打包成连续潜变量，或用类别扩散直接建模离散 token。Stable Audio 2.5 在连续音频潜变量上使用流匹配（flow matching）。AudioLDM 2 使用文本到梅尔频谱再到音频的扩散流程。

2024-2026 的趋势：流匹配正在音乐领域胜出（推理更快、样本更干净），而 token 自回归仍然主导语音领域，因为它天然是因果的，非常适合流式输出。

## 生产格局

| 系统 | 任务 | 主干架构 | 延迟 |
|--------|------|----------|---------|
| ElevenLabs V3 | TTS | Token 自回归 + 神经声码器 | 首 token 约 300ms |
| OpenAI GPT-4o audio | 全双工语音 | 端到端多模态自回归 | 约 200ms |
| NaturalSpeech 3 | TTS | 潜空间流匹配 | 非流式 |
| Stable Audio 2.5 | 音乐 / 音效 | DiT + 音频潜变量上的流匹配 | 1 分钟片段约 10s |
| Suno v4 | 完整歌曲 | 未公开；疑似 token 自回归 | 每首约 30s |
| Udio v1.5 | 完整歌曲 | 未公开 | 每首约 30s |
| MusicGen 3.3B | 音乐 | 基于 Encodec 32kHz 的 token 自回归 | 实时 |
| AudioCraft 2 | 音乐 + 音效 | 流匹配 | 5s 片段约 5s |
| Riffusion v2 | 音乐 | 频谱图扩散 | 约 10s |

## 从零实现

`code/main.py` 模拟核心思想：在合成的「音频 token」序列上训练一个微型的下一 token 预测 Transformer。这些序列由两种不同的「风格」生成（风格 A 是高低 token 交替，风格 B 是单调递增的斜坡）。以风格为条件进行采样。

### 第 1 步：合成音频 token

```python
def make_tokens(style, length, vocab_size, rng):
    if style == 0:  # "speech-like": alternating
        return [i % vocab_size for i in range(length)]
    # "music-like": ramp
    return [(i * 3) % vocab_size for i in range(length)]
```

### 第 2 步：训练一个微型 token 预测器

一个以风格为条件的 bigram 式预测器。重点在于这个模式本身：编解码器 token → 交叉熵训练 → 自回归采样。

### 第 3 步：条件采样

给定风格 token 和起始 token，从预测分布中采样下一个 token。持续生成 20-40 个 token。

## 常见陷阱

- **编解码器质量决定输出上限。** 如果编解码器无法忠实表示某种声音，生成器再好也无济于事。DAC 是当前最好的开源编解码器。
- **RVQ 误差累积。** 每一层 RVQ 建模的是上一层的残差，第 1 层的误差会向后传播。对高层使用温度为 0 的采样有帮助。
- **音乐结构。** 75 Hz 帧率下，30 秒音乐就是 2 万多个 token，对 Transformer 来说很困难。MusicGen 用滑动窗口 + 提示续写；Stable Audio 用更短的片段 + 交叉淡化拼接。
- **拼接边界处的伪影。** 在生成片段之间做交叉淡化需要精心设计的重叠相加（overlap-add）。
- **对干净数据的胃口。** 音乐生成器需要数万小时的授权音乐。2024 年 RIAA 起诉 Suno / Udio 的案件把这个问题摆上了台面。
- **声音克隆的伦理问题。** 一段 3 秒样本加一句文本提示，就足以让 VALL-E / XTTS / ElevenLabs 克隆一个人的声音。所有生产级模型都需要滥用检测 + 退出名单（opt-out list）。

## 生产实践

| 任务 | 2026 年技术栈 |
|------|------------|
| 商用 TTS | ElevenLabs、OpenAI TTS 或 Azure Neural |
| 声音克隆（需验证授权同意） | XTTS v2（开源）或 ElevenLabs Pro |
| 快速生成背景音乐 | Stable Audio 2.5 API、Suno 或 Udio |
| 带歌词的音乐 | Suno v4 或 Udio v1.5 |
| 音效 / 拟音 | AudioCraft 2、ElevenLabs SFX 或 Stable Audio Open |
| 实时语音智能体 | GPT-4o realtime 或 Gemini Live |
| 开放权重音乐研究 | MusicGen 3.3B、Stable Audio Open 1.0、AudioLDM 2 |
| 配音 / 翻译 | HeyGen、ElevenLabs Dubbing |

## 交付产物

保存 `outputs/skill-audio-brief.md`。该 skill 接收一份音频需求简报（任务、时长、风格、声音、授权），输出：模型 + 托管方案、提示格式（风格标签、风格描述词、结构标记）、编解码器 + 生成器 + 声码器链路、随机种子协议，以及评估方案（MOS / CLAP 分数 / TTS 的 CER / 用户 A/B 测试）。

## 练习

1. **简单。** 运行 `code/main.py` 并显式设置风格。验证生成的序列符合该风格的模式。
2. **中等。** 加入延迟并行解码：模拟 2 路必须保持 1 步偏移的 token 流，训练一个联合预测器。
3. **困难。** 用 HuggingFace transformers 在本地运行 MusicGen-small。用三个不同的提示各生成一段 10 秒的片段，对风格遵循度做 A/B 比较。

## 关键术语

| 术语 | 通俗说法 | 实际含义 |
|------|-----------------|-----------------------|
| 编解码器（Codec） | 「神经压缩」 | 音频的编码器 / 解码器；典型输出是 50-75 Hz 的 token。 |
| RVQ | 「残差 VQ」 | K 个量化器的级联；每一个建模上一个的残差。 |
| Token | 「一个编解码符号」 | 指向码本的离散索引；典型码本大小为 1024 或 2048。 |
| 延迟并行（Delayed parallel） | 「错位码本」 | 以交错偏移并行输出 K 路 token 流，从而缩短序列长度。 |
| 流匹配（Flow matching） | 「2024 年音频领域的赢家」 | 扩散的「直线路径」替代方案；采样更快。 |
| 声音提示（Voice prompt） | 「3 秒样本」 | 引导克隆声音的说话人嵌入或 token 前缀。 |
| 梅尔频谱图（Mel spectrogram） | 「可视化图」 | 对数幅度的感知频谱图；许多 TTS 系统使用。 |
| 声码器（Vocoder） | 「梅尔转波形」 | 把梅尔频谱图还原为音频的神经组件。 |

## 生产备注：音频本质上是流式问题

音频是唯一一种用户期望*边生成边到达*、而不是一次性交付的输出模态。在生产层面这意味着 TPOT（Time Per Output Token，单个输出 token 的耗时）很关键，因为目标吞吐量是用户的听觉速度——而不是阅读速度。对于以约 75 token/秒（Encodec）分词的 16kHz 音频，服务器必须为每个用户生成 ≥75 token/秒才能保证播放流畅。

由此带来两个架构后果：

- **流匹配音频模型无法简单地流式输出。** Stable Audio 2.5 和 AudioCraft 2 都是一次性渲染固定长度的片段。要做流式输出，就得把片段切块并在边界处重叠——类似滑动窗口扩散——相比编解码器自回归模型会增加 100-300ms 的延迟开销。

如果产品是「实时语音聊天」或「实时音乐续写」，选编解码器自回归路线。如果是「提交后渲染一段 30 秒的片段」，流匹配在质量和总延迟上都更胜一筹。

## 延伸阅读

- [Défossez et al. (2022). Encodec: High Fidelity Neural Audio Compression](https://arxiv.org/abs/2210.13438) —— 编解码器的事实标准。
- [Zeghidour et al. (2021). SoundStream](https://arxiv.org/abs/2107.03312) —— 第一个被广泛使用的神经音频编解码器。
- [Kumar et al. (2023). High-Fidelity Audio Compression with Improved RVQGAN (DAC)](https://arxiv.org/abs/2306.06546) —— DAC。
- [Wang et al. (2023). Neural Codec Language Models are Zero-Shot Text to Speech Synthesizers (VALL-E)](https://arxiv.org/abs/2301.02111) —— VALL-E。
- [Copet et al. (2023). Simple and Controllable Music Generation (MusicGen)](https://arxiv.org/abs/2306.05284) —— MusicGen。
- [Liu et al. (2023). AudioLDM 2: Learning Holistic Audio Generation with Self-supervised Pretraining](https://arxiv.org/abs/2308.05734) —— AudioLDM 2。
- [Stability AI (2024). Stable Audio 2.5](https://stability.ai/news/introducing-stable-audio-2-5) —— 2025 年基于流匹配的文本生成音乐。
