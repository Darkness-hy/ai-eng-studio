# 构建语音助手流水线 — 第六阶段毕业项目

> 把第 01-11 课的全部内容串联起来，构建一个能听、能思考、能回答的语音助手。在 2026 年，这已经是一个被解决的工程问题，而不是研究问题——但集成细节决定了它能否真正落地。

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 6 · 04, 05, 06, 07, 11; Phase 11 · 09 (Function Calling); Phase 14 · 01 (Agent Loop)
**Time:** ~120 minutes

## 问题背景

构建一个端到端的助手：

1. 采集麦克风输入（16 kHz 单声道）。
2. 检测用户语音的起止点。
3. 流式转写语音。
4. 将转写文本传给可以调用工具（计时器、天气、日历）的 LLM。
5. 将 LLM 输出的文本流式送入 TTS。
6. 把音频播放给用户。
7. 如果用户在回复中途打断，立即停止。

延迟目标：在笔记本电脑 CPU 上，从用户说完话到第一个 TTS 音频字节产生不超过 800 ms。质量目标：不丢词、静音时不产生幻觉字幕、无声音克隆泄漏、无提示词注入得逞。

## 核心概念

![Voice assistant pipeline: mic → VAD → STT → LLM+tools → TTS → speaker](../assets/voice-assistant.svg)

### 七个组件

1. **音频采集。** 麦克风 → 16 kHz 单声道 → 20 ms 音频块。Python 中通常用 `sounddevice`，生产环境用原生的 AudioUnit/ALSA/WASAPI。
2. **VAD（第 11 课）。** Silero VAD，阈值 0.5，最短语音 250 ms，静音保持（hang-over）500 ms。发出"开始"和"结束"信号。
3. **流式 STT（第 4-5 课）。** Whisper-streaming、Parakeet-TDT 或 Deepgram Nova-3（API）。输出部分转写和最终转写。
4. **支持工具调用的 LLM。** GPT-4o / Claude 3.5 / Gemini 2.5 Flash。工具用 JSON schema 定义。流式输出 token。
5. **流式 TTS（第 7 课）。** Kokoro-82M（最快的开源方案）或 Cartesia Sonic（商业方案）。LLM 输出 20 个 token 后即启动 TTS。
6. **播放。** 输出到扬声器；低带宽网络下使用 opus 编码。
7. **打断处理器。** 如果 TTS 播放期间 VAD 触发，则停止播放、取消 LLM、重启 STT。

### 你一定会遇到的三种故障模式

1. **首词截断。** VAD 启动慢了一拍，用户的"嘿"丢了。启动阈值设为 0.3，而不是 0.5。
2. **回复中途打断的混乱。** 用户打断后 LLM 仍在继续生成，助手和用户抢着说话。把 VAD 接到取消 LLM 的逻辑上。
3. **静音幻觉。** Whisper 在静音的预热帧上输出"Thanks for watching"。务必用 VAD 做门控。

### 2026 年生产参考技术栈

| 技术栈 | 延迟 | 许可证 | 备注 |
|-------|---------|---------|-------|
| LiveKit + Deepgram + GPT-4o + Cartesia | 350-500 ms | 商业 API | 2026 年行业默认方案 |
| Pipecat + Whisper-streaming + GPT-4o + Kokoro | 500-800 ms | 大部分开源 | 适合自建 |
| Moshi（全双工） | 200-300 ms | CC-BY 4.0 | 单模型；不同架构，见第 15 课 |
| Vapi / Retell（托管） | 300-500 ms | 商业 | 上线最快；可定制性有限 |
| Whisper.cpp + llama.cpp + Kokoro-ONNX | 离线 | 开源 | 隐私 / 边缘场景 |

## 从零实现

### 第 1 步：带分块的麦克风采集（伪代码）

```python
import sounddevice as sd

def mic_stream(chunk_ms=20, sr=16000):
    q = queue.Queue()
    def cb(indata, frames, time, status):
        q.put(indata.copy().flatten())
    with sd.InputStream(channels=1, samplerate=sr, blocksize=int(sr * chunk_ms/1000), callback=cb):
        while True:
            yield q.get()
```

### 第 2 步：VAD 门控的轮次采集

```python
def capture_turn(stream, vad, pre_roll_ms=300, silence_ms=500):
    buf, pre, triggered = [], collections.deque(maxlen=pre_roll_ms // 20), False
    silent = 0
    for chunk in stream:
        pre.append(chunk)
        if vad(chunk):
            if not triggered:
                buf = list(pre)
                triggered = True
            buf.append(chunk)
            silent = 0
        elif triggered:
            silent += 20
            buf.append(chunk)
            if silent >= silence_ms:
                return b"".join(buf)
```

### 第 3 步：流式 STT → LLM → TTS

```python
async def turn(audio_bytes):
    transcript = await stt.transcribe(audio_bytes)
    async for token in llm.stream(transcript):
        async for audio in tts.stream(token):
            await speaker.play(audio)
```

### 第 4 步：LLM 循环内的工具调用

```python
tools = [
    {"name": "get_weather", "parameters": {"location": "string"}},
    {"name": "set_timer", "parameters": {"seconds": "int"}},
]

async for chunk in llm.stream(user_text, tools=tools):
    if chunk.type == "tool_call":
        result = dispatch(chunk.name, chunk.args)
        continue_streaming(result)
    if chunk.type == "text":
        await tts.stream(chunk.text)
```

### 第 5 步：打断处理

```python
tts_task = asyncio.create_task(tts_loop())
while True:
    chunk = await mic.get()
    if vad(chunk):
        tts_task.cancel()
        await speaker.stop()
        await new_turn()
        break
```

## 生产实践

参见 `code/main.py`，这是一个可运行的模拟程序，用桩（stub）模型把全部七个组件串联起来，让你在没有硬件的情况下也能看清流水线的结构。要做真实实现，把桩替换为：

- `silero-vad`（`pip install silero-vad`）
- `deepgram-sdk` 或 `openai-whisper`
- `openai`（`gpt-4o`）或 `anthropic`
- `kokoro` 或 `cartesia`
- `sounddevice` 负责音频 I/O

## 常见陷阱

- **永久记录 PII。** 在大多数司法辖区，完整轮次的音频属于个人身份信息（PII）。保留 30 天，静态加密存储。
- **不支持插话打断（barge-in）。** 用户一定会打断，你的助手必须能闭嘴。
- **阻塞的 TTS。** 同步 TTS 会阻塞事件循环。改用异步或独立线程。
- **没有工具调用的错误处理。** 工具会失败。必须把错误返回给 LLM 并重试一次，之后优雅降级。
- **过度激进的幻觉过滤器。** 过滤太狠，助手会反复说"我帮不了你"；过滤太松，它会胡说八道。在留出集上做校准。
- **没有唤醒词选项。** 持续监听是隐私隐患。加一个唤醒词门控（Porcupine 或 openWakeWord）。

## 交付产物

保存为 `outputs/skill-voice-assistant-architect.md`。给定预算、规模、语言和合规约束，产出一份完整的技术栈规格说明。

## 练习

1. **简单。** 运行 `code/main.py`。它用桩模块端到端模拟一个完整轮次，并打印各阶段延迟。
2. **中等。** 把 STT 桩替换为真实的 Whisper 模型，处理一段预先录制的 `.wav`。测量词错误率（WER）和端到端延迟。
3. **困难。** 加入工具调用：实现 `get_weather`（任选一个 API）和 `set_timer`。让 LLM 通过工具完成请求，并验证当用户说"设一个 5 分钟的计时器"时，正确的函数被触发，且语音回复确认了这一操作。

## 关键术语

| 术语 | 大家口中的说法 | 实际含义 |
|------|-----------------|-----------------------|
| 轮次（Turn） | 一次用户与助手的往返 | 一段由 VAD 界定的用户语音 + 一次 LLM-TTS 回复。 |
| 插话打断（Barge-in） | 打断 | 助手说话时用户开口；助手立即停止。 |
| 唤醒词（Wake word） | "嘿，助手" | 短关键词检测器；Porcupine、Snowboy、openWakeWord。 |
| 端点检测（End-pointing） | 轮次结束 | 由 VAD + 最短静音时长判定用户已说完。 |
| 预滚动（Pre-roll） | 语音前缓冲 | 在 VAD 触发前保留 200-400 ms 的音频，避免首词截断。 |
| 工具调用（Tool call） | 函数调用 | LLM 输出 JSON；运行时分发执行；结果在循环内回传。 |

## 延伸阅读

- [LiveKit — voice agent quickstart](https://docs.livekit.io/agents/) — 生产级参考。
- [Pipecat — voice agent examples](https://github.com/pipecat-ai/pipecat) — 适合自建的框架。
- [OpenAI Realtime API](https://platform.openai.com/docs/guides/realtime) — 托管的语音原生路线。
- [Kyutai Moshi](https://github.com/kyutai-labs/moshi) — 全双工参考实现（第 15 课）。
- [Porcupine wake-word](https://picovoice.ai/products/porcupine/) — 唤醒词门控。
- [Anthropic — tool use guide](https://docs.anthropic.com/en/docs/build-with-claude/tool-use) — LLM 函数调用。
