# 语音智能体：Pipecat 与 LiveKit

> 到 2026 年，语音智能体已经是一个一线的生产级品类。Pipecat 提供基于帧（frame）的 Python 流水线（VAD → STT → LLM → TTS → 传输层）。LiveKit Agents 通过 WebRTC 把 AI 模型与用户连接起来。顶级技术栈的生产级端到端延迟目标落在 450–600ms。

**Type:** Learn
**Languages:** Python (stdlib)
**Prerequisites:** Phase 14 · 01 (Agent Loop), Phase 14 · 12 (Workflow Patterns)
**Time:** ~60 minutes

## 学习目标

- 描述 Pipecat 基于帧的流水线：DOWNSTREAM（源→汇）与 UPSTREAM（控制）两个方向。
- 说出语音流水线的标准阶段，以及 Pipecat 支持哪些传输层。
- 解释 LiveKit Agents 的两种语音智能体类（MultimodalAgent、VoicePipelineAgent）以及各自适用的场景。
- 概括 2026 年生产环境的延迟预期，以及它们如何左右架构选型。

## 问题背景

语音智能体不是在文本循环上简单加一层 TTS。延迟预算极其苛刻（约 600ms），部分音频（partial audio）是常态，话轮检测本身就是一个模型，传输层从电话 SIP 到 WebRTC 不一而足。你要么自己搭一条基于帧的流水线（Pipecat），要么依托一个平台（LiveKit）。

## 核心概念

### Pipecat（pipecat-ai/pipecat）

- 基于帧的 Python 流水线框架。
- `Frame` → `FrameProcessor` 处理链。
- 两个流动方向：
  - **DOWNSTREAM** —— 源 → 汇（音频输入，TTS 输出）。
  - **UPSTREAM** —— 反馈与控制（取消、指标、抢话打断）。
- `PipelineTask` 通过事件（`on_pipeline_started`、`on_pipeline_finished`、`on_idle_timeout`）管理生命周期，并通过观察者（observer）提供指标、追踪和 RTVI 支持。

典型流水线：

```
VAD (Silero) → STT → LLM (context alternates user/assistant) → TTS → transport
```

传输层：Daily、LiveKit、SmallWebRTCTransport、FastAPI WebSocket、WhatsApp。

Pipecat Flows 在此之上增加结构化对话（状态机）。Pipecat Cloud 是托管运行时。

### LiveKit Agents（livekit/agents）

- 通过 WebRTC 把 AI 模型与用户连接起来。
- 关键概念：`Agent`、`AgentSession`、`entrypoint`、`AgentServer`。
- 两种语音智能体类：
  - **MultimodalAgent** —— 通过 OpenAI Realtime 或同类接口直接处理音频。
  - **VoicePipelineAgent** —— STT → LLM → TTS 级联，提供文本层面的控制能力。
- 基于 Transformer 模型的语义话轮检测（semantic turn detection）。
- 原生 MCP 集成。
- 通过 SIP 支持电话接入。
- 通过 LiveKit Inference 免 API key 使用 50+ 模型，再通过插件接入 200+ 模型。

### 商业平台

Vapi（优化后的高端技术栈约 450–600ms）和 Retell（180 通测试通话端到端约 600ms）都构建在这些基础之上。如果你想要托管的语音技术栈、又不想养一支 WebRTC 团队，就选平台。

### 这个模式哪里容易出错

- **没有处理抢话打断（barge-in）。** 用户打断了，智能体还在自顾自说话。在 Pipecat 中需要 UPSTREAM 取消帧，LiveKit 中有对应机制。
- **忽略 STT 置信度。** 低置信度的转写结果被当成真理喂给 LLM。要按置信度做门控，或请用户确认。
- **TTS 半句被掐断。** 流水线在话说到一半时取消，TTS 需要被告知，或者直接切断音频。
- **无视延迟预算。** 每个组件都会增加 50–200ms。上线前先把整条链路的延迟加总。

### 2026 年的典型延迟

- VAD：20–60ms
- STT 部分结果：100–250ms
- LLM 首 token：150–400ms
- TTS 首段音频：100–200ms
- 传输层 RTT：30–80ms

端到端 450–600ms 算顶级，800–1200ms 是常态，超过 1500ms 就让人觉得坏了。

## 从零实现

`code/main.py` 是一个基于帧的玩具流水线，包含：

- `Frame` 类型（audio、transcript、text、tts_audio、control）。
- 带 `process(frame)` 方法的 `Processor` 接口。
- 由脚本化处理器构成的五阶段流水线（VAD → STT → LLM → TTS → transport）。
- 一个 UPSTREAM 取消帧，用来演示抢话打断。

运行方式：

```
python3 code/main.py
```

运行轨迹会展示正常流程，以及一次抢话打断取消如何让 TTS 半句即停。

## 生产实践

- **Pipecat**：需要完全掌控时选它——自定义处理器、Python 优先、可插拔的服务提供商。
- **LiveKit Agents**：面向 WebRTC 优先的部署和电话接入。
- **Vapi / Retell**：不想组建 WebRTC 团队时的托管语音智能体方案。
- **OpenAI Realtime / Gemini Live**：音频直进直出（MultimodalAgent）。

## 交付产物

`outputs/skill-voice-pipeline.md` 提供了一个 Pipecat 风格的语音流水线脚手架，包含 VAD + STT + LLM + TTS + 传输层，外加抢话打断处理。

## 练习

1. 给你的玩具流水线加一个指标观察者：统计每个阶段每秒的帧数。延迟在哪里累积？
2. 实现按置信度门控的 STT：低于阈值时回复"能再说一遍吗？"
3. 加一个语义话轮检测：简单规则——转写结果以"?"结尾即视为话轮结束。
4. 阅读 Pipecat 的传输层文档。把标准库实现的传输层替换为 SmallWebRTCTransport 配置（桩实现即可）。
5. 用同一条查询对比 OpenAI Realtime 与 STT+LLM+TTS 级联。文本层面的控制要付出多少延迟代价？

## 关键术语

| 术语 | 大家怎么叫 | 实际含义 |
|------|----------------|------------------------|
| Frame | "事件" | 流水线中的类型化数据单元（音频、转写、文本、控制） |
| Processor | "流水线阶段" | 带 process(frame) 的处理器 |
| DOWNSTREAM | "正向流" | 从源到汇：音频进，语音出 |
| UPSTREAM | "反馈流" | 控制信号：取消、指标、抢话打断 |
| VAD | "语音活动检测" | 检测用户何时在说话 |
| 语义话轮检测 | "聪明的话轮结束判断" | 由模型判断用户是否说完了 |
| MultimodalAgent | "直接音频智能体" | 音频进、音频出，中间没有文本 |
| VoicePipelineAgent | "级联智能体" | STT + LLM + TTS，提供文本层面控制 |

## 延伸阅读

- [Pipecat docs](https://docs.pipecat.ai/getting-started/introduction) —— 基于帧的流水线、处理器、传输层
- [LiveKit Agents docs](https://docs.livekit.io/agents/) —— WebRTC + 语音原语
- [Vapi](https://vapi.ai/) —— 托管语音平台
- [Retell AI](https://www.retellai.com/) —— 托管语音，附延迟基准测试
