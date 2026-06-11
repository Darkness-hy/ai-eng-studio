# 实时音频处理

> 批处理管线处理的是一个文件，实时管线则必须在下一个 20 毫秒到来之前处理完当前的 20 毫秒。每一个对话式 AI、广播演播室和电话机器人的生死，都系于这个延迟预算。

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 6 · 02 (Spectrograms), Phase 6 · 04 (ASR), Phase 6 · 07 (TTS)
**Time:** ~75 minutes

## 问题背景

你想要一个有「活人感」的语音助手。人类对话中轮换发言的延迟约为 230 毫秒（从静默到回应）。超过 500 毫秒就显得呆板；超过 1500 毫秒则感觉已经坏掉了。2026 年，一个完整的**听到 → 理解 → 回应 → 说出**循环的延迟预算是：

| 阶段 | 预算 |
|-------|--------|
| 麦克风 → 缓冲区 | 20 ms |
| VAD | 10 ms |
| ASR（流式） | 150 ms |
| LLM（首 token） | 100 ms |
| TTS（首块音频） | 100 ms |
| 渲染 → 扬声器 | 20 ms |
| **总计** | **~400 ms** |

Moshi（Kyutai，2024）实现了 200 毫秒的全双工延迟。GPT-4o-realtime（2024）约为 320 毫秒。而 2022 年的级联管线上线时还是 2500 毫秒。这 10 倍的提升来自三项技术：（1）全链路流式化，（2）带部分结果的异步流水线，（3）可中断的生成。

## 核心概念

![Streaming audio pipeline with ring buffer, VAD gate, interruption](../assets/real-time.svg)

**帧 / 块 / 窗口（frame / chunk / window）。** 实时音频以固定大小的数据块流动。常见选择是 20 毫秒（16 kHz 下为 320 个采样点）。下游的所有环节都必须跟上这个节奏。

**环形缓冲区（ring buffer）。** 固定大小的循环缓冲区。生产者线程写入新帧，消费者线程读取，避免在热路径上进行内存分配。容量 ≈ 最大延迟 × 采样率；一个 2 秒、16 kHz 的环形缓冲区 = 32,000 个采样点。

**VAD（语音活动检测，Voice Activity Detection）。** 在没人说话时拦住下游工作。Silero VAD 4.0（2024）在 CPU 上处理每个 30 毫秒的帧耗时不到 1 毫秒。`webrtcvad` 是较老的替代方案。

**流式 ASR。** 随着音频到达即输出部分转写结果的模型。Parakeet-CTC-0.6B 的流式模式（NeMo，2024）能在 320 毫秒延迟下达到 2–5% 的词错误率（WER）。Whisper-Streaming（Macháček et al., 2023）将 Whisper 分块处理，实现约 2 秒延迟的准流式转写。

**打断（interruption）。** 当用户在助手说话时开口，你必须（a）检测到插话（barge-in），（b）停止 TTS，（c）丢弃 LLM 剩余的输出。这一切要在 100 毫秒内完成，否则用户会觉得助手「装聋」。

**WebRTC Opus 传输。** 20 毫秒帧，48 kHz，自适应码率 8–128 kbps。浏览器和移动端的标准方案。LiveKit、Daily.co、Pion 是 2026 年构建语音应用的主流技术栈。

**抖动缓冲区（jitter buffer）。** 网络数据包会乱序或迟到。抖动缓冲区负责重排和平滑；太小 → 出现可听见的断点，太大 → 延迟增加。典型值为 60–80 毫秒。

### 常见坑点

- **线程争用。** Python 的 GIL 加上重型模型可能让音频线程「饿死」。使用基于 C 回调的音频库（sounddevice、PortAudio），让 Python 远离热路径。
- **采样率转换延迟。** 在管线内部重采样会增加 5–20 毫秒。要么在入口处一次性重采样，要么使用零延迟重采样器（PolyPhase、`soxr_hq`）。
- **TTS 预热。** 即便是 Kokoro 这样的快速 TTS，首次请求也有 100–200 毫秒的预热开销。在第一轮真实对话之前缓存模型并用一次空跑预热它。
- **回声消除。** 没有 AEC 时，TTS 的输出会回灌进麦克风，让 ASR 转写机器人自己的声音。WebRTC AEC3 是开源的默认选择。

```figure
nyquist-aliasing
```

## 从零实现

### 第 1 步：环形缓冲区

```python
import collections

class RingBuffer:
    def __init__(self, capacity):
        self.buf = collections.deque(maxlen=capacity)
    def write(self, frame):
        self.buf.extend(frame)
    def read(self, n):
        return [self.buf.popleft() for _ in range(min(n, len(self.buf)))]
    def level(self):
        return len(self.buf)
```

容量决定了最大缓冲延迟。16 kHz 下 32,000 个采样点 = 2 秒。

### 第 2 步：VAD 门控

```python
def simple_energy_vad(frame, threshold=0.01):
    return sum(x * x for x in frame) / len(frame) > threshold ** 2
```

生产环境请换成 Silero VAD：

```python
import torch
vad, _ = torch.hub.load("snakers4/silero-vad", "silero_vad")
is_speech = vad(torch.tensor(frame), 16000).item() > 0.5
```

### 第 3 步：流式 ASR

```python
# Parakeet-CTC-0.6B streaming via NeMo
from nemo.collections.asr.models import EncDecCTCModelBPE
asr = EncDecCTCModelBPE.from_pretrained("nvidia/parakeet-ctc-0.6b")
# chunk_ms=320 ms, look_ahead_ms=80 ms
for chunk in audio_stream():
    partial_text = asr.transcribe_streaming(chunk)
    print(partial_text, end="\r")
```

### 第 4 步：打断处理器

```python
class Dialog:
    def __init__(self):
        self.tts_task = None

    def on_user_speech(self, frame):
        if self.tts_task and not self.tts_task.done():
            self.tts_task.cancel()   # barge-in
        # then feed to streaming ASR

    def on_final_user_utterance(self, text):
        self.tts_task = asyncio.create_task(self.reply(text))

    async def reply(self, text):
        async for tts_chunk in llm_then_tts(text):
            speaker.write(tts_chunk)
```

关键在于异步 I/O 和可取消的 TTS 流式输出。在音频轨道上调用 WebRTC 的 peerconnection.stop() 是规范做法。

## 生产实践

2026 年的技术栈：

| 层级 | 选型 |
|-------|------|
| 传输 | LiveKit（WebRTC）或 Pion（Go） |
| VAD | Silero VAD 4.0 |
| 流式 ASR | Parakeet-CTC-0.6B 或 Whisper-Streaming |
| LLM 首 token | Groq、Cerebras、vLLM-streaming |
| 流式 TTS | Kokoro 或 ElevenLabs Turbo v2.5 |
| 回声消除 | WebRTC AEC3 |
| 端到端原生方案 | OpenAI Realtime API 或 Moshi |

## 常见陷阱

- **为了保险缓冲 500 毫秒。** 缓冲区*就是*你的延迟下限。把它压小。
- **不固定线程优先级。** 音频回调跑在比 UI 优先级还低的线程上 = 高负载时出现卡顿杂音。
- **TTS 分块太小。** 小于 200 毫秒的分块会让声码器（vocoder）伪影变得可听。320 毫秒是最佳分块大小。
- **没有抖动缓冲区。** 真实网络充满抖动；不做平滑就会出现爆音。
- **一次性的错误处理。** 音频管线必须防崩溃。一个异常就会杀死整个会话。

## 交付产物

保存为 `outputs/skill-realtime-designer.md`。设计一个实时音频管线，为每个阶段给出具体的延迟预算。

## 练习

1. **简单。** 运行 `code/main.py`。它模拟一个环形缓冲区 + 能量 VAD，并为一段伪造的 10 秒音频流打印各阶段延迟。
2. **中等。** 用 `sounddevice` 搭建一个直通回路，以 20 毫秒帧处理你的麦克风输入，并在每一帧打印 VAD 状态。
3. **困难。** 用 `aiortc` 搭建一个全双工回声测试：浏览器 → WebRTC → Python → WebRTC → 浏览器。用 1 kHz 脉冲测量端到端（glass-to-glass）延迟。

## 关键术语

| 术语 | 大家怎么说 | 实际含义 |
|------|-----------------|-----------------------|
| 环形缓冲区 | 循环队列 | 固定大小、无锁（或 SPSC 加锁）的音频帧 FIFO。 |
| VAD | 静默门控 | 区分语音与非语音的模型或启发式方法。 |
| 流式 ASR | 实时 STT | 随音频到达输出部分文本；前瞻窗口有上界。 |
| 抖动缓冲区 | 网络平滑器 | 重排乱序数据包的队列；典型值 60–80 毫秒。 |
| AEC | 回声消除 | 减去扬声器到麦克风的反馈路径。 |
| 插话（barge-in） | 用户打断 | 系统在 TTS 播放途中检测到用户说话；必须取消播放。 |
| 全双工 | 双向同时通话 | 用户和机器人可以同时说话；Moshi 是全双工的。 |

## 延伸阅读

- [Macháček et al. (2023). Whisper-Streaming](https://arxiv.org/abs/2307.14743) — 分块实现的准流式 Whisper。
- [Kyutai (2024). Moshi](https://kyutai.org/Moshi.pdf) — 200 毫秒延迟的全双工模型。
- [LiveKit Agents framework (2024)](https://docs.livekit.io/agents/) — 生产级音频智能体编排框架。
- [Silero VAD repo](https://github.com/snakers4/silero-vad) — 低于 1 毫秒的 VAD，Apache 2.0 协议。
- [WebRTC AEC3 paper](https://webrtc.googlesource.com/src/+/main/modules/audio_processing/aec3/) — 开源的回声消除实现。
