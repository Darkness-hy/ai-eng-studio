# 流式语音到语音 —— Moshi、Hibiki 与全双工对话

> 2024-2026 年重新定义了语音 AI。Moshi 用单个模型实现边听边说，延迟仅 200 ms。Hibiki 逐块完成语音到语音翻译。两者都抛弃了 ASR → LLM → TTS 流水线，转向基于 Mimi 编解码器 token 的统一全双工架构。这是新的参考设计。

**Type:** Learn
**Languages:** Python
**Prerequisites:** Phase 6 · 13 (Neural Audio Codecs), Phase 6 · 11 (Real-Time Audio), Phase 7 · 05 (Full Transformer)
**Time:** ~75 minutes

## 问题背景

基于第 11 + 12 课构建的所有语音智能体都有一个约 300-500 ms 的延迟下限：VAD 触发、STT 处理、LLM 推理、TTS 生成。每个阶段都有自己的最低延迟。你可以调优和并行化，但流水线的形态本身就限定了上限。

Moshi（Kyutai，2024-2026）提出了一个不同的问题：如果根本没有流水线呢？如果一个模型直接接收音频、连续输出音频，文本只作为中间的「内心独白」而非必经阶段呢？

答案就是**全双工语音到语音（full-duplex speech-to-speech）**。理论延迟 160 ms（80 ms Mimi 帧 + 80 ms 声学延迟）。在单张 L4 GPU 上的实际延迟为 200 ms。这只有顶级流水线式语音智能体延迟的一半。

## 核心概念

![Moshi architecture: two parallel Mimi streams + inner-monologue text](../assets/moshi-hibiki.svg)

### Moshi 架构

**输入。** 两条 Mimi 编解码器流，均为 12.5 Hz × 8 个码本：

- 流 1：用户音频（经 Mimi 编码，持续到达）
- 流 2：Moshi 自身的音频（由 Moshi 生成）

**Transformer。** 一个 70 亿参数的时序 Transformer（Temporal Transformer）同时处理这两条流和一条文本「内心独白」流。在每个 80 ms 的时间步，它会：

1. 消费最新的用户 Mimi token（8 个码本）。
2. 消费最近生成的 Moshi Mimi token（8 个码本，边生成边消费）。
3. 生成下一个 Moshi 文本 token（内心独白）。
4. 生成下一组 Moshi Mimi token（8 个码本，由一个小型 Depth Transformer 生成）。

三条流——用户音频、Moshi 音频、Moshi 文本——并行运行。Moshi 可以边说边听用户讲话；用户打断时可以自我中断；还能在不打断主要话语的情况下做出附和回应（"嗯哼"）。

**Depth Transformer。** 在一帧内，8 个码本并非并行预测——它们之间存在码本间依赖。一个小型的 2 层「depth transformer」在 80 ms 内顺序预测它们。这是自回归编解码器语言模型的标准分解方式（VALL-E、VibeVoice 也采用了这种方式）。

### 为什么内心独白文本有用

如果没有显式文本，模型就必须在声学流中隐式地建模语言。Moshi 的洞见在于：强制模型在输出音频的同时输出文本 token。这条文本流本质上就是 Moshi 所说内容的转录文本。这提高了语义连贯性，让替换语言模型头变得更容易，还免费提供了转录文本。

### Hibiki：流式语音到语音翻译

架构相同，但在翻译数据对上训练。源语言音频输入，目标语言音频连续输出。Hibiki-Zero（2026 年 2 月）消除了对词级对齐训练数据的依赖——使用句子级数据 + GRPO 强化学习来优化延迟。

最初支持四个语言对；只需约 1000 小时数据即可适配到新语言。

### 更广阔的 Kyutai 技术体系（2026）

- **Moshi** —— 全双工对话（法语优先，英语支持良好）
- **Hibiki / Hibiki-Zero** —— 同声语音翻译
- **Kyutai STT** —— 流式 ASR（500 ms 或 2.5 s 前瞻）
- **Kyutai Pocket TTS** —— 1 亿参数 TTS，可在 CPU 上运行（2026 年 1 月）
- **Unmute** —— 在公共服务器上组合上述组件的完整流水线

在一张 L40S GPU 上的吞吐量：64 路并发会话，3 倍实时速度。

### Sesame CSM —— 近亲

Sesame CSM（2025）采用了类似的思路——Llama-3 主干网络配上 Mimi 编解码器头。但 CSM 是单向的（接收上下文 + 文本，生成语音），而非全双工。它是市面上「语音临场感」最佳的 TTS；但与 Moshi 的全双工能力并不完全相同。

### 2026 年性能数据

| 模型 | 延迟 | 用途 | 许可证 |
|-------|---------|----------|---------|
| Moshi | 200 ms (L4) | 全双工英语 / 法语对话 | CC-BY 4.0 |
| Hibiki | 12.5 Hz 帧率 | 法语 ↔ 英语流式翻译 | CC-BY 4.0 |
| Hibiki-Zero | 同上 | 5 个语言对，无需对齐数据 | CC-BY 4.0 |
| Sesame CSM-1B | 200 ms TTFA | 上下文条件化 TTS | Apache-2.0 |
| GPT-4o Realtime | ~300 ms | 闭源，OpenAI API | 商业 |
| Gemini 2.5 Live | ~350 ms | 闭源，Google API | 商业 |

## 从零实现

### 第 1 步：接口

Moshi 提供一个 WebSocket 服务器，接收 80 ms 一块的 Mimi 编码音频，并返回 80 ms 一块的 Mimi 编码音频。双向。持续不断。

```python
import asyncio
import websockets
from moshi.client_utils import encode_audio_mimi, decode_audio_mimi

async def moshi_chat():
    async with websockets.connect("ws://localhost:8998/api/chat") as ws:
        mic_task = asyncio.create_task(stream_mic_to(ws))
        spk_task = asyncio.create_task(stream_from_to_speaker(ws))
        await asyncio.gather(mic_task, spk_task)
```

### 第 2 步：全双工循环

```python
async def stream_mic_to(ws):
    async for chunk_80ms in mic_stream_at_12_5_hz():
        mimi_tokens = encode_audio_mimi(chunk_80ms)
        await ws.send(serialize(mimi_tokens))

async def stream_from_to_speaker(ws):
    async for msg in ws:
        mimi_tokens, text_token = deserialize(msg)
        audio = decode_audio_mimi(mimi_tokens)
        await play(audio)
```

两个方向同时运行。Python asyncio 或 Rust futures 是标准的传输方式。

### 第 3 步：训练目标（概念层面）

对于每个 80 ms 的帧 `t`：

- 输入：`user_mimi[0..t]`、`moshi_mimi[0..t-1]`、`moshi_text[0..t-1]`
- 预测：`moshi_text[t]`，然后是 `moshi_mimi[t, codebook_0..7]`

文本在音频之前预测（内心独白）；音频在 depth transformer 内部按码本顺序逐个预测。

### 第 4 步：Moshi 的优势与不足

Moshi 的优势：

- 在廉价硬件上实现低于 250 ms 的端到端延迟。
- 自然的附和回应和打断处理。
- 不需要流水线的胶水代码。

Moshi 的不足：

- 工具调用（未经此类训练；需要单独的 LLM 路径）。
- 长链推理（Moshi 是一个约 80 亿参数的对话模型，不是 Claude/GPT-4）。
- 小众主题上的事实准确性。
- 大多数生产级企业用例（2026 年仍在使用流水线）。

## 生产实践

| 场景 | 选择 |
|-----------|------|
| 延迟最低的语音陪伴应用 | Moshi |
| 实时翻译通话 | Hibiki |
| 语音演示 / 研究 | Moshi、CSM |
| 带工具的企业智能体 | 流水线（第 12 课），而非 Moshi |
| 上下文中的定制语音 TTS | Sesame CSM |
| 语音到语音，任意语言 | GPT-4o Realtime 或 Gemini 2.5 Live（商业） |

## 常见陷阱

- **工具调用能力有限。** Moshi 是对话模型，不是智能体框架。需要工具时请与流水线组合使用。
- **特定音色的条件化。** Moshi 使用单一训练好的人设音色；克隆音色需要单独的训练。
- **语言覆盖。** 法语 + 英语支持极佳；其他语言有限。Hibiki-Zero 有所帮助，但你仍然需要训练数据。
- **资源成本。** 一个完整的 Moshi 会话会占用一个 GPU 槽位；不适合廉价的多租户共享部署模式。

## 交付产物

保存为 `outputs/skill-duplex-pipeline.md`。针对一个语音智能体工作负载，在流水线与全双工架构之间做出选择，并给出理由。

## 练习

1. **简单。** 运行 `code/main.py`。它以符号化方式模拟双流 + 内心独白架构。
2. **中等。** 从 HuggingFace 拉取 Moshi，运行服务器，测试一次对话。测量从用户说话结束到 Moshi 开始响应的真实耗时。
3. **困难。** 拿你第 12 课的流水线智能体，在 20 条匹配的测试语句上与 Moshi 对比 P50 延迟。写一篇分析，说明在哪些情况下流水线在架构上反而占优。

## 关键术语

| 术语 | 通俗说法 | 实际含义 |
|------|-----------------|-----------------------|
| 全双工（Full-duplex） | 边听边说 | 同一模型上两条音频流同时活跃。 |
| 内心独白（Inner monologue） | 模型的文本流 | Moshi 在输出音频的同时输出文本 token。 |
| Depth transformer | 码本间预测器 | 在一个 80 ms 帧内预测 8 个码本的小型 transformer。 |
| Mimi | Kyutai 的编解码器 | 12.5 Hz × 8 个码本；语义 + 声学；驱动 Moshi。 |
| 流式 S2S（Streaming S2S） | 音频 → 音频实时转换 | 逐块进行的翻译 / 对话，没有流水线阶段。 |
| 附和回应（Back-channeling） | "嗯哼"式反应 | Moshi 可以发出简短的认可回应而不中断自己的话轮。 |

## 延伸阅读

- [Défossez et al. (2024). Moshi — speech-text foundation model](https://arxiv.org/html/2410.00037v2) —— 原始论文。
- [Kyutai Labs (2026). Hibiki-Zero](https://arxiv.org/abs/2602.12345) —— 无需对齐数据的流式翻译。
- [Sesame (2025). Crossing the uncanny valley of voice](https://www.sesame.com/research/crossing_the_uncanny_valley_of_voice) —— CSM 规格说明。
- [Kyutai — Moshi repo](https://github.com/kyutai-labs/moshi) —— 安装 + 服务器。
- [OpenAI — Realtime API](https://platform.openai.com/docs/guides/realtime) —— 闭源商业对标产品。
- [Kyutai — Delayed Streams Modeling](https://github.com/kyutai-labs/delayed-streams-modeling) —— 底层的 STT/TTS 框架。
