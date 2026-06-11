# 音频语言模型 — Qwen2.5-Omni、Audio Flamingo、GPT-4o Audio

> 2026 年的音频语言模型可以对语音、环境声音和音乐进行联合推理。Qwen2.5-Omni-7B 在 MMAU-Pro 上与 GPT-4o Audio 打平。Audio Flamingo Next 在 LongAudioBench 上超过了 Gemini 2.5 Pro。开源与闭源之间的差距基本已被抹平——唯独多音频任务例外，在这类任务上所有模型都接近随机水平。

**Type:** Learn
**Languages:** Python
**Prerequisites:** Phase 6 · 04 (ASR), Phase 12 · 03 (Vision-Language Models), Phase 7 · 10 (Audio Transformers)
**Time:** ~45 minutes

## 问题背景

你手上有 5 秒钟的音频：狗叫、有人大喊"stop!"、然后是一片寂静。有价值的问题横跨多个维度：

- **转写。**"说了什么？"——这是 ASR 的领地。
- **语义推理。**"这个人是否处于危险中？"——需要对狗叫 + 喊叫 + 寂静做联合理解。
- **音乐推理。**"主旋律由哪些乐器演奏？"
- **长音频检索。**"在这段 90 分钟的讲座里，讲师在哪里讲解了梯度下降？"

能用一条提示词回答以上所有问题的单一模型，就是**音频语言模型**（audio-language model，LALM / ALM）。它与纯 ASR 的区别在于：LALM 输出的是自由形式的自然语言回答，而不只是转写文本。

## 核心概念

![Audio-language model: audio encoder + projector + LLM decoder](../assets/alm-architecture.svg)

### 三组件模板

2026 年的每一个 LALM 都共享同一副骨架：

1. **音频编码器。**Whisper 编码器 · BEATs · CLAP · WavLM · 或各模型自研的编码器。
2. **投影器（projector）。**用线性层或 MLP 把音频编码器特征桥接到 LLM 的词元嵌入空间。
3. **LLM。**基于 Llama / Qwen / Gemma 的解码器。接收文本与音频词元交错的输入，生成文本。

训练流程：

- **第一阶段。**冻结编码器和 LLM，仅在 ASR / 音频描述数据上训练投影器。
- **第二阶段。**在指令跟随类音频任务（问答、推理、音乐理解）上做全量或 LoRA 微调。
- **第三阶段（可选）。**语音输入 / 语音输出能力需要额外加一个语音解码器。Qwen2.5-Omni 和 AF3-Chat 就是这样做的。

### 2026 年模型版图

| 模型 | 主干 | 音频编码器 | 输出模态 | 获取方式 |
|-------|----------|---------------|-----------------|--------|
| Qwen2.5-Omni-7B | Qwen2.5-7B | 自研 + Whisper | 文本 + 语音 | Apache-2.0 |
| Qwen3-Omni | Qwen3 | 自研 | 文本 + 语音 | Apache-2.0 |
| Audio Flamingo 3 | Qwen2 | AF-CLAP | 文本 | NVIDIA 非商业许可 |
| Audio Flamingo Next | Qwen2 | AF-CLAP v2 | 文本 | NVIDIA 非商业许可 |
| SALMONN | Vicuna | Whisper + BEATs | 文本 | Apache-2.0 |
| LTU / LTU-AS | Llama | CAV-MAE | 文本 | Apache-2.0 |
| GAMA | Llama | AST + Q-Former | 文本 | Apache-2.0 |
| Gemini 2.5 Flash/Pro（闭源） | Gemini | 专有 | 文本 + 语音 | API |
| GPT-4o Audio（闭源） | GPT-4o | 专有 | 文本 + 语音 | API |

### 基准测试现实核查（2026）

**MMAU-Pro。**1800 个问答对，覆盖语音 / 声音 / 音乐 / 混合类别，并包含多音频子集。

| 模型 | 总体 | 语音 | 声音 | 音乐 | 多音频 |
|-------|---------|--------|-------|-------|-------------|
| Gemini 2.5 Pro | ~60% | 73.4% | 51.9% | 64.9% | ~22% |
| Gemini 2.5 Flash | ~57% | 73.4% | 50.5% | 64.9% | 21.2% |
| GPT-4o Audio | 52.5% | — | — | — | 26.5% |
| Qwen2.5-Omni-7B | 52.2% | 57.4% | 47.6% | 61.5% | ~20% |
| Audio Flamingo 3 | ~54% | — | — | — | — |
| Audio Flamingo Next | LongAudioBench 上的 SOTA | — | — | — | — |

**多音频这一列对所有模型都是致命打击。**四选一选择题的随机猜测正确率是 25%，而大多数模型的得分就在这附近。LALM 在比较两段音频片段这件事上仍然举步维艰。

### 2026 年 LALM 的实用场景

- **呼叫中心录音的合规审计。**"客服是否提到了必要的免责声明？"
- **无障碍辅助。**为听障用户描述声音事件（而不只是转写）。
- **内容审核。**检测暴力言语 + 威胁性语气 + 背景环境信息。
- **播客 / 会议章节划分。**做语义摘要，而不只是切分说话人轮次。
- **音乐曲库分析。**"找出所有 B 段发生转调的曲目。"

### （目前）还不实用的场景

- 细粒度乐理分析（和弦层级以下）。
- 长对话中带说话人归属的推理（超过 10 分钟后性能退化）。
- 多音频比较（22-26% 仅勉强高于随机水平）。
- 实时流式推理（大多数模型只支持离线批量推理）。

## 从零实现

### 第 1 步：调用 Qwen2.5-Omni

```python
from transformers import AutoModelForCausalLM, AutoProcessor

processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-Omni-7B")
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-Omni-7B", torch_dtype="auto")

audio, sr = load_wav("clip.wav", sr=16000)
messages = [{
    "role": "user",
    "content": [
        {"type": "audio", "audio": audio},
        {"type": "text", "text": "What sounds do you hear, and what's happening?"},
    ],
}]
inputs = processor.apply_chat_template(messages, tokenize=True, return_tensors="pt")
output = model.generate(**inputs, max_new_tokens=200)
print(processor.decode(output[0], skip_special_tokens=True))
```

### 第 2 步：投影器模式

```python
import torch.nn as nn

class AudioProjector(nn.Module):
    def __init__(self, audio_dim=1280, llm_dim=4096):
        super().__init__()
        self.down = nn.Linear(audio_dim, llm_dim)
        self.act = nn.GELU()
        self.up = nn.Linear(llm_dim, llm_dim)

    def forward(self, audio_features):
        return self.up(self.act(self.down(audio_features)))
```

就这么简单。投影器通常只有 1-3 个线性层。在 ASR 配对数据（音频 → 转写文本）上训练它，就是第一阶段的预训练任务。

### 第 3 步：在 MMAU / LongAudioBench 上做基准测试

```python
from datasets import load_dataset
mmau = load_dataset("MMAU/MMAU-Pro")

correct = 0
for item in mmau["test"]:
    answer = call_model(item["audio"], item["question"], item["choices"])
    if answer == item["correct_choice"]:
        correct += 1
print(f"Accuracy: {correct / len(mmau['test']):.3f}")
```

按类别（语音 / 声音 / 音乐 / 多音频）分别报告结果。聚合指标会掩盖模型真正失败的地方。

## 生产实践

| 任务 | 2026 年首选 |
|------|-----------|
| 自由形式音频问答（开源） | Qwen2.5-Omni-7B |
| 长音频上最强的开源模型 | Audio Flamingo Next |
| 最强闭源模型 | Gemini 2.5 Pro |
| 语音输入 / 语音输出智能体 | Qwen2.5-Omni 或 GPT-4o Audio |
| 音乐推理 | Audio Flamingo 3 或 2（针对音乐特化的 AF-CLAP） |
| 呼叫中心审计 | 通过 API 调用 Gemini 2.5 Pro，并对你的合规政策文档做 RAG |

## 常见陷阱

- **对多音频任务过度信任。**如果你的任务需要回答"哪段音频里出现了 X"，随机水平的性能是真实存在的。
- **长音频性能退化。**超过 10 分钟后，大多数模型的说话人归属能力会崩溃。先做说话人分离（第 6 课），再做摘要。
- **静音段幻觉。**使用 Whisper 编码器的 LALM 继承了同样的 Whisper 式问题。用 VAD 做门控。
- **基准测试摘樱桃。**厂商博客只展示表现最好的类别。请自己跑一遍 MMAU-Pro 的多音频子集。

## 交付产物

保存为 `outputs/skill-alm-picker.md`。针对给定的音频理解任务，选定 LALM + 基准测试子集 + 输出模态（文本 vs 语音）。

## 练习

1. **简单。**运行 `code/main.py`，观察一个玩具级投影器模式 + 模拟 LALM 如何把（音频嵌入, 文本词元）路由为输出词元。
2. **中等。**用 Qwen2.5-Omni-7B 在 100 道 MMAU-Pro 语音题上评分，并与论文报告的数字对比。
3. **困难。**搭建一个最小化的音频描述基线：BEATs 编码器 + 2 层投影器 + 冻结的 Llama-3.2-1B。仅在 AudioCaps 上微调投影器，再在 Clotho-AQA 上与 SALMONN 对比。

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|-----------------|-----------------------|
| LALM | 音频版 ChatGPT | 音频编码器 + 投影器 + LLM 解码器。 |
| 投影器（projector） | 适配器 | 把音频特征映射到 LLM 嵌入空间的小型 MLP。 |
| MMAU | 那个基准测试 | 覆盖语音、声音、音乐的 1 万个音频问答对。 |
| MMAU-Pro | 更难的 MMAU | 1800 道多音频 / 重推理的题目。 |
| LongAudioBench | 长音频评测 | 数分钟长的片段配上语义查询。 |
| 语音输入 / 语音输出 | 语音原生 | 模型直接吃进语音、吐出语音，不绕道文本。 |

## 延伸阅读

- [Chu et al. (2024). Qwen2-Audio](https://arxiv.org/abs/2407.10759) — 参考架构。
- [Alibaba (2025). Qwen2.5-Omni](https://huggingface.co/Qwen/Qwen2.5-Omni-7B) — 语音进、语音出。
- [NVIDIA (2025). Audio Flamingo 3](https://arxiv.org/abs/2507.08128) — 开源长音频领跑者。
- [NVIDIA (2026). Audio Flamingo Next](https://arxiv.org/abs/2604.10905) — LongAudioBench SOTA。
- [Tang et al. (2023). SALMONN](https://arxiv.org/abs/2310.13289) — 双编码器先驱。
- [MMAU-Pro leaderboard](https://mmaubenchmark.github.io/) — 2026 年实时排行榜。
