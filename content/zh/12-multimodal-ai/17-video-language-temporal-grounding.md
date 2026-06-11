# 视频-语言模型：时序 Token 与时间定位

> 视频不是照片的堆叠。一段 5 秒的片段包含因果顺序、动作动词和事件时间，这些是图像模型无法表示的。Video-LLaMA（Zhang et al., 2023 年 6 月）发布了首个具备音画定位能力的开源视频 LLM。VideoChat 和 Video-LLaVA 将这一模式规模化。到 2025 年，Qwen2.5-VL 的 TMRoPE 缩小了与前沿闭源模型的差距。每个系统对时序 token 的处理方式各不相同——按片段使用 Q-former、按帧拼接池化、按 token 应用 TMRoPE。本课将解读这些模式，构建一个均匀采样与动态采样对比的帧采样器，并在时间定位任务上进行评估。

**Type:** Build
**Languages:** Python (stdlib, frame sampler + temporal-grounding evaluator)
**Prerequisites:** Phase 12 · 08 (LLaVA-OneVision)
**Time:** ~180 minutes

## 学习目标

- 解释为什么时序位置编码会独立于视觉编码器影响视频 VLM 的性能。
- 在每秒 token 数与定位精度两个维度上，比较均匀采样、动态 FPS 采样和事件驱动帧采样。
- 描述按片段 Q-former（Video-LLaMA）、按帧池化（Video-LLaVA）与按 token M-RoPE（Qwen2.5-VL）三种设计。
- 说出四个视频基准：VideoMME、TempCompass、EgoSchema、Video-MMMU。

## 问题背景

一段 30 FPS 的 1 分钟视频有 1800 帧。按每帧 196 个视觉 token 计算（ViT-B，224 分辨率），总计 352k 个 token——超过 2024 年任何 LLM 的上下文长度。

存在三种压缩策略：

1. 帧下采样（根据内容取 1-8 FPS）。
2. 对每帧的 patch token 进行激进池化（3x3 或 4x4 双线性池化）。
3. 通过 Q-former 压缩：输入一个 16 帧片段，输出 64 个 token。

各自的取舍不同。下采样丢失时序细节。池化丢失空间细节。Q-former 两者都略有损失，但节省 token。

时序位置编码是另一个维度：模型如何知道第 5 帧在第 6 帧之前？可选方案包括简单的一维时序 RoPE（Video-LLaMA）、可学习的时序嵌入（Video-LLaVA），以及 TMRoPE（Qwen2.5-VL，完整 3D）。

## 核心概念

### Video-LLaMA：按片段 Q-former + 音频分支

Video-LLaMA（2023）是首个开源视频 LLM。架构如下：

- 以 2 FPS 取 16 帧片段（即 8 秒）。
- 逐帧 ViT 特征 -> Video Q-former 对全部 16 帧做交叉注意力 -> 32 个可学习查询 -> LLM。
- 并行音频分支：波形 -> ImageBind 音频编码器 -> Audio Q-former -> 32 个查询 -> LLM。

优势：音画联合推理。劣势：固定片段长度，无法做任意时间点的定位。

### VideoChat 与 Video-LLaVA

VideoChat 保留了 Video-LLaMA 的思路，但去掉了音频并做了简化。Video-LLaVA（Lin et al., 2023）用同一个视觉编码器同时在图像和视频帧上训练（"投影前对齐"，alignment before projection），得到统一的表示。两者都是冻结的 CLIP 编码器 + MLP + LLM 结构。

两者都无法处理长视频，均为 8-16 帧的系统。

### Qwen2.5-VL 与 TMRoPE

Qwen2.5-VL 引入了 TMRoPE——时序-模态旋转位置编码（Temporal-Modality Rotary Position Embedding）。每个 patch token 携带一个 (t, h, w) 位置，其中 t 是实际时间戳（而非帧索引）。

与简单时序嵌入的关键区别：

- 绝对时间，而非索引。模型看到的是"在 4.2 秒处"，而不是"在第 15 帧"。
- 按 token 旋转，而非按片段。每个视觉 token 根据自己的时间戳独立旋转。
- 兼容动态 FPS。如果这一段以 2 FPS 采样、那一段以 4 FPS 采样，TMRoPE 原生支持这种不均匀间隔。

TMRoPE 使"猫在第几秒跳起来？"这类查询成为可能。模型可以输出"在 4.2 秒处"。Video-LLaMA 只能回答"在片段的早期"。

### 帧采样策略

均匀采样：在整段时长上均匀取 N 帧。简单，但会错过运动高峰。

动态 FPS：根据运动强度自适应采样。用光流或帧间差分找出高运动片段并加密采样。Qwen2.5-VL 在这种数据上训练。

事件驱动：运行一个轻量级检测器，在动作发生处加密采样。VideoAgent 采用此方案。

关键帧 + 上下文：在镜头切换边界采样，外加少量相邻帧。用于影视类内容。

### 按帧池化

在 1 FPS、每帧 576 个 token 的设定下，一段 5 分钟的视频是 172,800 个 token。Qwen2.5-VL-72B 的 128k 上下文勉强能装下，但代价高昂。

3x3 双线性池化将其降至每帧 64 个 token -> 5 分钟共 19,200 个 token。这是多数任务的最佳平衡点。

对空间细节要求不高的智能体工作流，可以更激进地池化（6x6 -> 每帧 16 个 token）。

### 四个视频基准

- VideoMME：综合视频理解，覆盖短、中、长视频。
- TempCompass：细粒度时序推理，"之前"/"之后"类问题。
- EgoSchema：长时程第一人称视频。
- Video-MMMU：多模态多学科视频问答。

完整的视频 VLM 评估应覆盖全部四个基准。它们考察的维度各不相同——TempCompass 专注于事件顺序，EgoSchema 考察 3 分钟以上的推理，VideoMME 覆盖多种时长。

### 定位输出格式

时间定位的输出格式：

- 自由文本："The cat jumps around the 4-second mark."（猫大约在 4 秒处跳起。）易于解析但不精确。
- 结构化 JSON：`{"event": "jump", "start": 4.1, "end": 4.3}`。Qwen2.5-VL 在这种格式上训练。
- 基于 token：在答案中穿插特殊的 `<time>4.1</time>` token。这是 Qwen2.5-VL 的内部格式。

基于 token 的格式对下游使用最精确。Qwen2.5-VL 的 JSON 输出格式可直接解析。

### 2026 年最佳实践

2026 年的视频 VLM：

- 编码器：SigLIP 2 搭配 M-RoPE 或 TMRoPE（Qwen2.5-VL）。
- 帧采样：动态 FPS（根据运动强度取 1-4），并设最大帧数上限。
- 按帧池化：3x3 双线性。
- 输出：包含 time 与 event 字段的结构化 JSON。
- 基准：通用评估用 VideoMME + TempCompass；长时程评估用 EgoSchema。

## 生产实践

`code/main.py` 包含：

- 均匀和动态 FPS 帧采样器。
- 一个玩具级时间定位评估器：给定时间 T 处的"真值"事件和模型输出，按容差打分。
- 一个横向对比：Video-LLaMA（16 帧，Q-former）、Video-LLaVA（8 帧，MLP）、Qwen2.5-VL（动态 FPS + TMRoPE）。

## 交付产物

本课产出 `outputs/skill-video-vlm-frame-planner.md`。给定一个视频任务（监控、动作识别、时间定位、摘要），它会选择帧采样器、池化系数、输出格式以及预期精度档位。

## 练习

1. 针对一段 3 分钟的烹饪演示视频，在均匀采样和动态 FPS 之间做选择。用 token 数量来论证。

2. TMRoPE 具体增加了哪些简单时序嵌入表无法做到的能力？

3. 编写一个 VLM 可以学会输出的时间定位 JSON schema。包含错误情形。

4. 阅读 Video-LLaVA 论文第 3 节关于 "Alignment Before Projection" 的内容。为什么这比分别训练图像和视频编码器更好？

5. 根据 VideoMME 排行榜，截至 2026 年，最强开源模型与最强闭源模型之间的差距是多少？其中有多少可以归因于时序编码，又有多少归因于基座 LLM 的规模？

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|-----------------|------------------------|
| 时间定位（Temporal grounding） | "时间定位的回答" | VLM 针对事件发生的时刻输出具体的时间戳区间 |
| TMRoPE | "时间-多模态 RoPE" | 带绝对时间戳的 3D 旋转位置编码，Qwen2.5-VL 采用 |
| 动态 FPS | "运动感知采样" | 高运动片段多采帧，静态片段少采帧 |
| 帧池化 | "逐帧空间压缩" | 在送入 LLM 之前用双线性插值减少每帧的 patch 数量 |
| Video Q-former | "片段压缩器" | 用交叉注意力瓶颈把 N 帧映射为 K 个可学习查询 |
| VideoMME | "视频基准" | 覆盖短/中/长视频的综合基准，2500+ 样本 |

## 延伸阅读

- [Zhang et al. — Video-LLaMA (arXiv:2306.02858)](https://arxiv.org/abs/2306.02858)
- [Li et al. — VideoChat (arXiv:2305.06355)](https://arxiv.org/abs/2305.06355)
- [Lin et al. — Video-LLaVA (arXiv:2311.10122)](https://arxiv.org/abs/2311.10122)
- [Qwen Team — Qwen2.5-VL (arXiv:2502.13923)](https://arxiv.org/abs/2502.13923)
- [Lin et al. — VILA-1.5 (arXiv:2312.07533)](https://arxiv.org/abs/2312.07533)
