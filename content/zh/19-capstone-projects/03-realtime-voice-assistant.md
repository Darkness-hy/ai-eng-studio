# Capstone 03 — 实时语音助手（ASR 到 LLM 到 TTS）

> 一个体验自然的语音智能体，端到端延迟要低于 800ms，能判断你什么时候说完了话，能处理打断（barge-in），还能在不卡顿音频的前提下调用工具。Retell、Vapi、LiveKit Agents 和 Pipecat 在 2026 年都达到了这个水准。它们的实现形态是一样的：流式 ASR、轮次检测器、流式 LLM、流式 TTS，全部通过 WebRTC 串联，并在每一跳都施加严格的延迟预算。亲手构建一个，测量 WER、MOS 和误切断率，并让它在丢包环境下运行。

**Type:** Capstone
**Languages:** Python (agent + pipeline), TypeScript (web client)
**Prerequisites:** Phase 6 (speech and audio), Phase 7 (transformers), Phase 11 (LLM engineering), Phase 13 (tools), Phase 14 (agents), Phase 17 (infrastructure)
**Phases exercised:** P6 · P7 · P11 · P13 · P14 · P17
**Time:** 30 hours

## 问题背景

语音是 2025-2026 年间演进最快的 AI 交互形态。技术天花板每个季度都在下降。OpenAI Realtime API、Gemini 2.5 Live、Cartesia Sonic-2、ElevenLabs Flash v3、LiveKit Agents 1.0 和 Pipecat 0.0.70 都让「首音频输出低于 800ms」变得触手可及。但门槛不只是延迟，而是交互手感：不抢用户的话，不被误打断，能从句中被打断的状态恢复，能在对话中途调用工具而不让音频卡顿，还能扛住抖动严重的移动网络。

把三个 REST 调用串起来是做不到这些的。这套架构必须是端到端的流水线式流式处理。亲手构建之后，各种失效模式才会显形：为电话音频调优的 VAD 被背景电视声误触发、轮次检测器苦等一个永远不会出现的标点、TTS 先缓冲 400ms 才开始输出。这个 Capstone 的任务就是在负载下逐一修复这些问题，并发布一份延迟与质量报告。

## 核心概念

整条流水线包含五个流式阶段：**音频输入**（来自浏览器或 PSTN 的 WebRTC）、**ASR**（Deepgram Nova-3 或 faster-whisper 输出流式部分转写）、**轮次检测**（VAD 加上一个小型轮次检测模型，从部分转写中读取话语完成的线索）、**LLM**（一旦判定轮次结束就开始流式输出 token）、**TTS**（在收到第一个 LLM token 后约 200ms 内开始流式输出音频）。

还有三个横切关注点。**打断（barge-in）**：当用户在智能体说话时开口，TTS 立即取消，ASR 立刻接管。**工具调用**：对话中途的函数调用（天气、日历）必须走旁路通道，不能卡住音频；如果延迟超过 300ms，智能体会预先填一句应答语（「稍等……」）。**背压（backpressure）**：在丢包时，部分转写会被暂存，VAD 提高语音门限阈值，智能体也会避免在消息未确认时抢着说话。

衡量标准是量化的。在 15 dB 信噪比的 Hamming VAD 基准上 WER 低于 8%。100 通实测通话的首音频输出 p50 低于 800ms。误切断率低于 3%。TTS 的 MOS 高于 4.2。单台 g5.xlarge 支撑 50 路并发通话。这些数字就是交付物。

## 架构

```
browser / Twilio PSTN
        |
        v
   WebRTC / SIP edge
        |
        v
  LiveKit Agents 1.0  (or Pipecat 0.0.70)
        |
   +----+--------------+--------------+-----------------+
   |                   |              |                 |
   v                   v              v                 v
  ASR              VAD v5         turn-detector     side-channel
(Deepgram         (Silero)          (LiveKit)        tools
 Nova-3 /         speech-gate    completion score    (weather,
 Whisper-v3)      per 20ms        on partials        calendar)
   |                   |              |
   +--------+----------+--------------+
            v
        LLM (streaming)
     GPT-4o-realtime / Gemini 2.5 Flash /
     cascaded Claude Haiku 4.5
            |
            v
        TTS streaming
     Cartesia Sonic-2 / ElevenLabs Flash v3
            |
            v
     audio back to caller
            |
            v
   OpenTelemetry voice traces -> Langfuse
```

## 技术栈

- 传输层：LiveKit Agents 1.0（WebRTC）加 Twilio PSTN 网关；Pipecat 0.0.70 作为备选框架
- ASR：Deepgram Nova-3（流式，首个部分转写低于 300ms）或自托管的 faster-whisper Whisper-v3-turbo
- VAD：Silero VAD v5 加 LiveKit 轮次检测器（读取部分转写的小型 Transformer）
- LLM：OpenAI GPT-4o-realtime（集成最紧密）、Gemini 2.5 Flash Live，或级联式 Claude Haiku 4.5（流式补全，音频走独立路径）
- TTS：Cartesia Sonic-2（首字节最快）、ElevenLabs Flash v3，或自托管可选开源的 Orpheus
- 工具：FastMCP 旁路通道处理天气/日历/预订；工具耗时超过 300ms 时智能体预先发出填充语
- 可观测性：OpenTelemetry 语音 span，Langfuse 语音追踪并支持音频回放
- 部署：单台 g5.xlarge（24GB 显存）跑自托管的 Whisper + Orpheus；追求最低延迟则用托管 API

## 从零实现

1. **WebRTC 会话。** 搭建一个 LiveKit 房间和一个推送麦克风音频流的 Web 客户端。在服务端挂载一个加入该房间的 agent worker。

2. **ASR 流式转写。** 把 20ms 的 PCM 帧送入 Deepgram Nova-3（或 GPU 上的 faster-whisper）。订阅部分转写和最终转写。记录每条部分转写的延迟。

3. **VAD 与轮次检测器。** 在帧流上运行 Silero VAD v5。在语音结束事件触发时，用最新的部分转写调用 LiveKit 轮次检测器。只有当 VAD 判定静音持续 500ms 且轮次检测器的完成度评分 > 0.6 时，才确认「轮次结束」。

4. **LLM 流式输出。** 轮次结束后，带上当前对话和最终转写发起 LLM 调用。流式输出 token。收到第一个 token 时立即移交给 TTS。

5. **TTS 流式输出。** Cartesia Sonic-2 流式返回音频块。第一个音频块必须在第一个 LLM token 后 200ms 内离开服务器。把音频块发到 LiveKit 房间；客户端通过 WebRTC 抖动缓冲区播放。

6. **打断处理。** 当 TTS 正在播放而 VAD 检测到新的用户语音时，立即取消 TTS 流，丢弃剩余的 LLM 输出，并重新激活 ASR。发布一个 `tts_canceled` span。

7. **工具旁路通道。** 把天气和日历注册为函数调用工具。被调用时并发执行；若 300ms 内未返回，让 LLM 先说一句「稍等，我查一下」作为填充语；工具返回后继续。

8. **评测框架。** 录制 100 通通话。计算 WER（对照留出的参考转写）、误切断率（用户话说到一半 TTS 就被取消）、首音频输出 p50、TTS 的 MOS（人工评分或 NISQA），并做一次抖动丢包测试（丢弃 3% 的数据包）。

9. **负载测试。** 用合成呼叫方在单台 g5.xlarge 上驱动 50 路并发通话。测量持续负载下的首音频输出 p95。

## 生产实践

```
caller: "what is the weather in tokyo tomorrow"
[asr  ] partial @280ms: "what is the"
[asr  ] partial @540ms: "what is the weather"
[turn ] completion score 0.82 at @820ms; commit
[llm  ] first token @960ms
[tool ] weather.tokyo tomorrow -> 68/52 partly cloudy @1140ms
[tts  ] first audio-out @1040ms: "Tokyo tomorrow will be partly cloudy..."
turn latency: 1040ms user-stop -> audio-out
```

## 交付产物

交付物是 `outputs/skill-voice-agent.md`。给定一个领域（客服、日程安排或自助终端），它能搭起一个 LiveKit 智能体，其 ASR/VAD/LLM/TTS 流水线调优到满足衡量标准。评分标准：

| 权重 | 评分项 | 测量方式 |
|:-:|---|---|
| 25 | 端到端延迟 | 100 通录制通话中首音频输出 p50 低于 800ms |
| 20 | 轮次交替质量 | Hamming VAD 基准上误切断率低于 3% |
| 20 | 工具调用正确性 | 对话中途的工具调用返回正确数据且不卡顿音频 |
| 20 | 丢包下的可靠性 | 注入 3% 丢包时 WER 与轮次交替的稳定性 |
| 15 | 评测框架完整度 | 配置公开、可复现的测量结果 |
| **100** | | |

## 练习

1. 在 g5.xlarge 上把 Deepgram Nova-3 换成 faster-whisper v3 turbo。测量延迟和 WER 的差距。找出哪些环节的 CPU 与 GPU 取舍真正重要。

2. 加入一套打断仲裁策略：用户在工具调用过程中插话时，智能体该怎么办？比较三种策略（立即硬取消、跑完工具再停、把下一轮排队）。

3. 做一次对抗性轮次检测测试：让用户在句子中间长时间停顿。调节 VAD 静音阈值和轮次检测器评分阈值，在不突破 900ms 的前提下把误切断率压到最低。

4. 通过 Twilio 把同一个智能体部署到 PSTN。比较 PSTN 与 WebRTC 的首音频输出。解释抖动缓冲区和编解码器上的差异。

5. 为非英语语言（日语、西班牙语）加入语音活动检测。测量 Silero VAD v5 的误触发率，并与针对特定语言微调的版本对比。

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|-----------------|------------------------|
| 轮次检测（turn detection） | 「话语结束」 | 一个分类器，基于 VAD 静音和部分转写，判断用户是否已经说完 |
| 打断（barge-in） | 「打断处理」 | 当 VAD 检测到新的用户语音时，取消正在播放的 TTS |
| 首音频输出（first-audio-out） | 「延迟」 | 从用户停止说话到第一个音频包离开服务器的时间 |
| VAD | 「语音门限」 | 把音频帧分类为语音或静音的模型；Silero VAD v5 是 2026 年的默认选择 |
| 抖动缓冲区（jitter buffer） | 「音频平滑」 | 客户端缓冲区，短暂保存数据包以吸收网络波动 |
| 填充语（filler） | 「应答 token」 | 工具响应较慢时智能体说的一句短语，用来避免冷场 |
| MOS | 「平均意见分」 | 语音质量的主观感知评分；NISQA 是其自动化替代指标 |

## 延伸阅读

- [LiveKit Agents 1.0](https://github.com/livekit/agents) — WebRTC 智能体参考框架
- [Pipecat](https://github.com/pipecat-ai/pipecat) — 备选的 Python 优先流式智能体框架
- [OpenAI Realtime API](https://platform.openai.com/docs/guides/realtime) — 一体化语音模型的参考
- [Deepgram Nova-3 documentation](https://developers.deepgram.com/docs) — 流式 ASR 参考
- [Silero VAD v5](https://github.com/snakers4/silero-vad) — VAD 参考模型
- [Cartesia Sonic-2](https://docs.cartesia.ai) — 低延迟 TTS 参考
- [Retell AI architecture](https://docs.retellai.com) — 生产级语音智能体架构
- [Vapi.ai production stack](https://docs.vapi.ai) — 另一套生产级参考实现
