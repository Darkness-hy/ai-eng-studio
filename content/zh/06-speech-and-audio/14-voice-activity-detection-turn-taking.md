# 语音活动检测与轮次切换 —— Silero、Cobra 与 flush 技巧

> 每个语音智能体的成败都系于两个判断：用户现在在说话吗？他说完了吗？VAD 回答第一个问题；轮次检测（VAD + 静音延迟 + 语义端点模型）回答第二个。任何一个判断出错，你的助手要么打断用户，要么喋喋不休。

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 6 · 11 (Real-Time Audio), Phase 6 · 12 (Voice Assistant)
**Time:** ~45 minutes

## 问题背景

语音智能体在每个 20 ms 音频块上都要做三个不同的判断：

1. **这一帧是语音吗？** —— VAD，逐帧的二分类。
2. **用户是否开始了新的发言？** —— 起始检测（onset detection）。
3. **用户说完了吗？** —— 端点检测（end-pointing，即轮次结束）。

朴素的做法（能量阈值）在任何噪声面前都会失效——车流声、键盘声、人群嘈杂声。2026 年的答案是：Silero VAD（开源、深度学习方案）+ 轮次检测模型（语义端点检测）+ 基于 VAD 校准的静音延迟。

## 核心概念

![VAD cascade: energy → Silero → turn-detector → flush trick](../assets/vad-turn-taking.svg)

### 三级 VAD 级联

**第一级：能量门限。** 最廉价的方案。对 RMS 设 -40 dBFS 的阈值。能过滤明显的静音，但任何超过阈值的噪声都会触发。

**第二级：Silero VAD**（2020-2026，MIT 协议）。100 万参数，在 6000 多种语言上训练。单 CPU 线程处理每个 30 ms 音频块仅需约 1 ms。在 5% FPR 下 TPR 达 87.7%。开源方案的默认之选。

**第三级：语义轮次检测器。** LiveKit 的轮次检测模型（2024-2026）或你自己训练的小型分类器。能区分「句中停顿」和「说完了」。利用语言学上下文（语调 + 最近的词），而不只是静音。

### 关键参数及其默认值

- **阈值。** Silero 输出一个概率；&gt; 0.5（默认）判为语音，或 &gt; 0.3（高灵敏度）。阈值越低 = 首词被截断越少，但误报越多。
- **最短语音时长。** 拒绝短于 250 ms 的语音——通常是咳嗽或椅子的噪声。
- **静音延迟（端点检测）。** VAD 归零后，等待 500-800 ms 再宣告轮次结束。太短 → 打断用户。太长 → 反应迟钝。
- **预滚缓冲（pre-roll buffer）。** 保留 VAD 触发前 300-500 ms 的音频，防止「hey」被截掉。

### flush 技巧（Kyutai 2025）

流式 STT 模型存在前瞻延迟（Kyutai STT-1B 为 500 ms，STT-2.6B 为 2.5 s）。通常你得在语音结束后等这么久才能拿到转写文本。flush 技巧的做法是：当 VAD 检测到语音结束时，**向 STT 发送一个 flush 信号**，强制其立即输出。STT 以约 4 倍实时速度处理，因此 500 ms 的缓冲只需约 125 ms 就能处理完。

端到端：125 ms VAD + flush STT = 对话级延迟。

### 2026 年 VAD 对比

| VAD | TPR @ 5% FPR | 延迟 | 协议 |
|-----|--------------|---------|---------|
| WebRTC VAD（Google，2013） | 50.0% | 30 ms | BSD |
| Silero VAD（2020-2026） | 87.7% | ~1 ms | MIT |
| Cobra VAD（Picovoice） | 98.9% | ~1 ms | 商业授权 |
| pyannote segmentation | 95% | ~10 ms | 类 MIT |

Silero 是正确的默认选择。Cobra 是面向合规 / 精度要求的升级方案。纯能量 VAD 在 2026 年的生产环境中已无立足之地。

## 从零实现

### 第 1 步：能量门限

```python
def energy_vad(chunk, threshold_dbfs=-40.0):
    rms = (sum(x * x for x in chunk) / len(chunk)) ** 0.5
    dbfs = 20.0 * math.log10(max(rms, 1e-10))
    return dbfs > threshold_dbfs
```

### 第 2 步：在 Python 中使用 Silero VAD

```python
from silero_vad import load_silero_vad, get_speech_timestamps

vad = load_silero_vad()
audio = torch.tensor(waveform_16k, dtype=torch.float32)
segments = get_speech_timestamps(
    audio, vad, sampling_rate=16000,
    threshold=0.5,
    min_speech_duration_ms=250,
    min_silence_duration_ms=500,
    speech_pad_ms=300,
)
for s in segments:
    print(f"{s['start']/16000:.2f}s - {s['end']/16000:.2f}s")
```

### 第 3 步：轮次结束状态机

```python
class TurnDetector:
    def __init__(self, silence_hangover_ms=500, min_speech_ms=250):
        self.state = "idle"
        self.speech_ms = 0
        self.silence_ms = 0
        self.silence_hangover_ms = silence_hangover_ms
        self.min_speech_ms = min_speech_ms

    def update(self, is_speech, chunk_ms=20):
        if is_speech:
            self.speech_ms += chunk_ms
            self.silence_ms = 0
            if self.state == "idle" and self.speech_ms >= self.min_speech_ms:
                self.state = "speaking"
                return "START"
        else:
            self.silence_ms += chunk_ms
            if self.state == "speaking" and self.silence_ms >= self.silence_hangover_ms:
                self.state = "idle"
                self.speech_ms = 0
                return "END"
        return None
```

### 第 4 步：flush 技巧骨架

```python
def flush_on_end(stt_client, audio_buffer):
    stt_client.send_audio(audio_buffer)
    stt_client.send_flush()
    return stt_client.recv_transcript(timeout_ms=150)
```

这要求 STT（Kyutai、Deepgram、AssemblyAI）必须支持 flush 才能生效。Whisper 流式方案不支持——它基于分块处理，总是要等满一个块。

## 生产实践

| 场景 | VAD 选择 |
|-----------|-----------|
| 开源、快速、通用 | Silero VAD |
| 商业呼叫中心 | Cobra VAD |
| 端侧（手机） | Silero VAD ONNX |
| 研究 / 说话人分离 | pyannote segmentation |
| 零依赖兜底 | WebRTC VAD（遗留方案） |
| 需要高质量轮次结束判断 | Silero + LiveKit 轮次检测器叠加 |

经验法则：除非真的别无选择，否则永远不要把纯能量 VAD 上线。

## 常见陷阱

- **固定阈值。** 安静环境管用，嘈杂环境失效。要么在设备端做校准，要么换用 Silero。
- **静音延迟太短。** 智能体会在用户句中插话。对话语音的最佳区间是 500-800 ms。
- **延迟太长。** 体验迟钝。和目标用户做 A/B 测试。
- **没有预滚缓冲。** 用户音频的前 200-300 ms 会丢失。务必保留一段滚动的预滚缓冲。
- **忽视语义端点检测。** 「嗯，让我想想……」中包含长停顿。用户最恨思考时被打断。使用 LiveKit 的轮次检测器或类似方案。

## 交付产物

保存为 `outputs/skill-vad-tuner.md`。针对一个具体工作负载，选定 VAD 模型、阈值、静音延迟、预滚缓冲和轮次检测策略。

## 练习

1. **简单。** 运行 `code/main.py`。它会模拟「语音 + 静音 + 语音 + 咳嗽」序列，并测试三级 VAD。
2. **中等。** 安装 `silero-vad`，处理一段 5 分钟的录音，调节阈值以同时最小化首词截断和误触发。报告精确率/召回率。
3. **困难。** 构建一个迷你轮次检测器：Silero VAD + 基于最近 10 个词嵌入的 3 层 MLP（使用 sentence-transformers）。在手工标注的轮次结束数据集上训练。F1 比纯 Silero 方案高出 10%。

## 关键术语

| 术语 | 通俗说法 | 实际含义 |
|------|-----------------|-----------------------|
| VAD | 语音检测器 | 逐帧二分类：这一帧是语音吗？ |
| 轮次检测 | 端点检测 | VAD + 静音延迟 + 语义端点。 |
| 静音延迟 | 语音后的等待 | 宣告轮次结束前的等待时间；500-800 ms。 |
| 预滚缓冲 | 语音前缓冲 | 保留 VAD 触发前 300-500 ms 的音频。 |
| flush 技巧 | Kyutai 的妙招 | VAD → flush-STT → 延迟从 500 ms 降到 125 ms。 |
| 语义端点 | 「他们是真想停下吗？」 | 看词语而不只是静音的 ML 分类器。 |
| TPR @ FPR 5% | ROC 工作点 | VAD 的标准基准；Silero 为 87.7%，WebRTC 为 50%。 |

## 延伸阅读

- [Silero VAD](https://github.com/snakers4/silero-vad) —— 开源 VAD 的参考实现。
- [Picovoice Cobra VAD](https://picovoice.ai/products/cobra/) —— 商业方案中的精度领先者。
- [Kyutai — Unmute + flush trick](https://kyutai.org/stt) —— 实现 200 ms 以内延迟的工程技巧。
- [LiveKit — turn detection](https://docs.livekit.io/agents/logic/turns/) —— 生产环境中的语义端点检测。
- [WebRTC VAD](https://webrtc.googlesource.com/src/) —— 遗留基线方案。
- [pyannote segmentation](https://github.com/pyannote/pyannote-audio) —— 说话人分离级别的语音分段。
