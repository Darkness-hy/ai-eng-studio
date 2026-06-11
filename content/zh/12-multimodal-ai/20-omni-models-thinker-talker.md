# 全模态模型：Qwen2.5-Omni 与 Thinker-Talker 分离架构

> GPT-4o 在 2024 年 5 月的产品演示之所以具有颠覆性，不在于底层模型本身，而在于产品形态——一个语音交互界面：你开口说话，模型能看到摄像头画面，并在 250ms 内用语音回应你。开源生态在 2024 年剩余时间和整个 2025 年都在追赶这一产品形态。Qwen2.5-Omni（2025 年 3 月）是这一方向的参考性开源设计：一个 Thinker（生成文本的大型 Transformer）加一个 Talker（并行生成语音的 Transformer），二者通过流式语音 token 衔接。Mini-Omni 简化了这一设计，Moshi 在延迟上与之持平，GLM-4-Voice 将其扩展到了中文。本课将剖析 Thinker-Talker 架构，以及让流式实时对话得以成立的延迟预算。

**Type:** Build
**Languages:** Python (stdlib, streaming pipeline latency simulator + VAD loop)
**Prerequisites:** Phase 12 · 19 (audio-LLMs), Phase 12 · 16 (any-to-any)
**Time:** ~180 minutes

## 学习目标

- 将推理管线拆分为 Thinker（文本推理）和 Talker（语音合成），并解释为什么并行流式处理是可行的。
- 逐组件计算一次对话交互的首音频字节时间（time-to-first-audio-byte，TTFAB）预算。
- 描述 TMRoPE 如何在 Thinker 内部对视觉、音频和文本进行时间对齐的位置编码。
- 说出三种实时对话模式：半双工、轮流发言、全双工。

## 问题背景

一个实时语音助手需要在极短时间内完成很多事：

1. 听清用户。实时语音 token 化，并通过语音活动检测（voice activity detection，VAD）判断用户何时说完。
2. 可选地看见。摄像头输入以 2-4 FPS 流入 Thinker，与音频并行。
3. 思考。基于对话历史组织回应。
4. 说话。合成音频 token，解码为波形，流式传输到用户的扬声器。

每一步都会增加延迟。要有对话感，总往返时间必须 < 500ms——低于这个值，用户就不再察觉到延迟。GPT-4o 宣称约 250ms。Moshi 约 160ms。Qwen2.5-Omni 约 350-500ms。

每个组件都必须支持流式处理。任何环节都不能是「先批量处理完再解码」。

## 核心概念

### Thinker 与 Talker

Qwen2.5-Omni 的拆分方式：

- Thinker：一个 7B-80B 的文本生成 Transformer。输入交错排列的文本 + 图像 + 音频 token，输出表示「该说什么」的文本 token。
- Talker：一个较小的语音生成 Transformer（200M-1B）。输入 Thinker 输出的文本 token 加上近期的语音上下文 token，输出离散语音 token（残差 VQ 索引）。
- 语音解码器：一个流式波形解码器（SNAC、MoVQGAN 系列），实时将语音 token 转换为音频采样。

这种分离至关重要。Thinker 必须足够大才能保证推理质量。Talker 可以很小，因为它的任务是局部的——把文本转换成语音 token。更大的 Talker 并不会更有表现力，只会更慢。

二者并行运行：

1. Thinker 输出文本 token t_i。
2. Talker（通过流式方式）消费 t_i，输出语音 token s_i, s_{i+1}, ..., s_{i+k}。
3. 语音解码器边接收语音 token 边输出音频采样。
4. 当 Thinker 进行到文本 token t_{i+3} 时，Talker 已经把 t_0..t_{i+2} 对应的音频流式输出完毕。

### TMRoPE——时间对齐的多模态位置编码

Thinker 需要整合图像帧（比如以 4 FPS 到达）、音频帧（以每秒 50 帧到达）和对话历史中的文本。朴素的序列排序（先所有图像，再所有音频，最后文本）会丢失时间对齐信息。

TMRoPE 为每个 token 分配绝对时间戳。视觉 token 在 t=2.3s。音频 token 在 t=2.32s。用户说「stop」对应的文本 token 在 t=2.35s。RoPE 按时间戳旋转注意力；模型因此把它们视为时间上同时发生的事件。

这正是让「他一边说你好一边挥手」能够被理解的基础设施——模型在同一个概念时刻同时看到视频帧和音频。

### 流式语音合成

语音 token 必须流式生成。Mini-Omni（Xie & Wu, 2024）提出了「语言模型可以边听、边想、边以流式方式说话」：Thinker 的输出 token 和 Talker 的输出 token 在同一序列中交错排列。Thinker 一确定下一个文本 token，Talker 就立即开始工作，没有批处理边界。

Moshi（Défossez et al., 2024 年 10 月）是目前最快的开源实现。单张 A100 上 TTFAB 为 160ms。架构：一个 7B 的单一 Transformer，在交替的位置上输出文本 token 和语音 token，并用「内心独白（inner monologue）」机制将思考流与说话流分开。这实际上是把 Thinker + Talker 通过精心训练融合进了一个模型。

### VAD 与轮流发言

语音活动检测运行在输入侧。有两种模式：

- 半双工（half-duplex）：用户说话时模型倾听，模型说话时用户倾听。通过 VAD 静音检测（约 200ms）实现清晰的话轮交接。
- 全双工（full-duplex）：双方可以同时说话。模型可以发出附和声（「嗯哼」）或打断对方。难度大得多。Moshi 支持这种模式。

Qwen2.5-Omni 默认支持半双工，通过静音阈值实现轮流发言。全双工需要在应用层额外处理。

### Qwen3-Omni（2025 年 11 月）

后继版本。采用 Qwen3-80B 作为 Thinker，更大的 Talker，改进版 TMRoPE-v2。延迟接近 GPT-4o 的 250ms。开放权重。在 OmniBench 基准上的表现与 Gemini 2.0 Live 不相上下。

### 生产环境延迟预算

一次典型的流式交互：

- 麦克风 -> 音频 token：40-80ms。
- Prefill（提示词 + 历史）：7B 模型 100-200ms，70B 模型显著更长。
- Thinker 的第一个文本 token：40ms。
- Talker 处理第一个文本 token：20ms。
- 第一批语音 token 确定：40ms。
- 残差 VQ 解码：30ms。
- 语音波形解码：50-80ms。

总 TTFAB：7B 模型 320-510ms，70B 模型 600-900ms。前沿质量通常意味着 70B 以上，这就是前沿模型的延迟差距来源。

### Token 速率计算

在 16kHz 语音、基础语音 token 速率为 50 Hz 的条件下，每秒输出需要 50 个语音 token。Talker 必须以 ≥50 tok/s 的速度输出才跟得上。按 H100 上典型的 LLM 吞吐量 30-80 tok/s 计算，一个小型（200-300M）Talker 足够快，而一个 7B 的 Talker 会跟不上节奏。

这就是为什么会存在专用的小型 Talker 模型，而不是「直接用主模型」。

## 生产实践

`code/main.py`：

- 用模拟的 token 输出速率仿真一条 Thinker-Talker 管线。
- 针对可配置的模型规模和麦克风采样率计算 TTFAB。
- 演示基于 VAD 静音阈值的半双工轮流发言。

## 交付产物

本课产出 `outputs/skill-omni-streaming-budget.md`。给定一个实时语音产品的目标 TTFAB 和功能集（视觉输入、双语、全双工），从 Qwen2.5-Omni、Qwen3-Omni、Moshi 或 Mini-Omni 中选型，并确定 Thinker/Talker 的规模。

## 练习

1. 你的目标 TTFAB 是 300ms。在 7B Thinker 和 300M Talker 的配置下，写出每个组件的延迟。

2. Qwen2.5-Omni 使用 TMRoPE。描述这样一个场景下模型看到了什么：用户在 t=1s 开始说话，摄像头在 t=1.2s 捕捉到一个手势。

3. 全双工支持要求模型在倾听的同时输出音频。提出一种能教会模型这种能力的训练数据格式。

4. 阅读 Moshi 论文的第 4 节。描述「内心独白」的分离机制，以及它为什么可以避免 Thinker-Talker 拆分。

5. 计算吞吐量预算：要跟上 16kHz 语音、每秒 50 个基础层 token 的速度，Talker 必须以多快的速度输出 token？

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|-----------------|------------------------|
| Thinker | 「推理大脑」 | 生成「该说什么」的大型文本生成 Transformer |
| Talker | 「发声的嘴」 | 根据 Thinker 的文本生成离散语音 token 的小型 Transformer |
| TTFAB | 「延迟预算」 | 首音频字节时间：从用户说完话到输出第一个音频采样的时间 |
| TMRoPE | 「时间对齐的 RoPE」 | 基于绝对时间戳、跨视觉/音频/文本的位置编码 |
| 半双工 | 「轮流发言」 | 用户和模型交替发言；VAD 静音检测判断用户说完 |
| 全双工 | 「同时进行」 | 模型可以边说边听；能够发出附和声 |
| 内心独白 | 「Moshi 式分离」 | 单模型设计，思考流与说话流交错排列 |

## 延伸阅读

- [Xu et al. — Qwen2.5-Omni (arXiv:2503.20215)](https://arxiv.org/abs/2503.20215)
- [Qwen Team — Qwen3-Omni (arXiv:2509.17765)](https://arxiv.org/html/2509.17765v1)
- [Xie & Wu — Mini-Omni (arXiv:2408.16725)](https://arxiv.org/abs/2408.16725)
- [Défossez et al. — Moshi (arXiv:2410.00037)](https://arxiv.org/abs/2410.00037)
- [Zeng et al. — GLM-4-Voice (arXiv:2412.02612)](https://arxiv.org/abs/2412.02612)
