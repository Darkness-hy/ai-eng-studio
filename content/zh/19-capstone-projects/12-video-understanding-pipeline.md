# Capstone 12 — 视频理解流水线（场景、问答、搜索）

> Twelve Labs 把 Marengo + Pegasus 做成了产品。VideoDB 推出了"视频版 CRUD"API。AI2 的 Molmo 2 发布了开源 VLM 检查点。Gemini 凭借长上下文能力可以原生处理数小时的视频。TimeLens-100K 定义了大规模时序定位（temporal grounding）。2026 年的流水线形态已经定型：场景分割、逐场景字幕 + 嵌入、转录文本对齐、多向量索引，以及返回 (start, end) 时间戳和帧预览的查询。本毕业项目的目标是摄取 100 小时视频、跑通公开基准测试，并测量计数类与动作类问题上的幻觉率。

**Type:** Capstone
**Languages:** Python (pipeline), TypeScript (UI)
**Prerequisites:** Phase 4 (CV), Phase 6 (speech), Phase 7 (transformers), Phase 11 (LLM engineering), Phase 12 (multimodal), Phase 17 (infrastructure)
**Phases exercised:** P4 · P6 · P7 · P11 · P12 · P17
**Time:** 30 hours

## 问题背景

长视频问答是 2026 年规模下最消耗带宽的多模态问题。Gemini 2.5 Pro 可以原生读取一段 2 小时的视频，但要把 100 小时视频摄取成一个可查询的语料库，仍然需要场景级索引。生产环境的标准形态组合了场景分割（TransNetV2 或 PySceneDetect）、用 VLM 做逐场景字幕生成（Gemini 2.5、Qwen3-VL-Max 或 Molmo 2）、转录文本对齐（带词级时间戳的 Whisper-v3-turbo），以及一个把字幕、帧嵌入和转录文本并排存储的多向量索引。查询流水线返回 (start, end) 时间戳和帧预览。

基准测试使用公开数据集（ActivityNet-QA、NeXT-GQA），外加你自建的 100 条查询的自定义集合。计数类与动作类问题上的幻觉是公认的难点失败类别；本毕业项目会显式地测量它。

## 核心概念

摄取阶段有三条流水线并行运行。**场景分割**把视频切成场景。**VLM 字幕生成**为每个场景生成一条字幕，并从关键帧生成一个帧嵌入。**ASR 对齐**产出词级时间戳。三条流通过 (scene_id, 时间区间) 连接。每个场景在多向量索引（Qdrant）中获得三种向量类型：字幕嵌入、关键帧嵌入、转录文本嵌入。

查询时，自然语言问题同时打向三种向量；结果用 RRF 合并；一个时序定位适配器（TimeLens 风格）在排名最高的场景内细化 (start, end) 窗口。VLM 合成器（Gemini 2.5 Pro 或 Qwen3-VL-Max）接收查询 + 高分场景 + 裁剪后的帧，回答时附带引用的时间戳和帧预览。

幻觉测量很重要。计数类（"有几个人进了房间？"）和动作类（"厨师是先倒料再搅拌吗？"）问题出了名地不可靠。要把它们的准确率与描述类问题分开报告。

## 架构

```
video file / URL
      |
      v
PySceneDetect / TransNetV2  (scene segmentation)
      |
      +--- per-scene keyframe --- VLM caption + frame embedding
      |                            (Gemini 2.5 Pro / Qwen3-VL-Max / Molmo 2)
      |
      +--- audio channel --- Whisper-v3-turbo ASR + word timestamps
      |
      v
multi-vector Qdrant: {caption_emb, keyframe_emb, transcript_emb}
      |
query:
  dense queries against all three -> RRF merge -> top-k scenes
      |
      v
TimeLens / VideoITG temporal grounding (refine start/end within scene)
      |
      v
VLM synth: query + top scenes + frame previews
      |
      v
answer + (start, end) timestamps + frame thumbs + citations
```

## 技术栈

- 场景分割：TransNetV2（2024-26 年的最先进方案）或 PySceneDetect
- ASR：通过 faster-whisper 运行带词级时间戳的 Whisper-v3-turbo
- VLM 字幕生成与问答模型：Gemini 2.5 Pro、Qwen3-VL-Max 或 Molmo 2
- 时序定位：基于 TimeLens-100K 训练的适配器，或 VideoITG
- 索引：支持多向量的 Qdrant（字幕 / 帧 / 转录文本）
- UI：Next.js 15，带 HTML5 视频播放器和场景缩略图
- 评测：ActivityNet-QA、NeXT-GQA，以及人工标注的 100 题自定义集合
- 幻觉基准：带人工标注的计数类与动作类子集

## 从零实现

1. **摄取入口（ingest walker）。** 接受 YouTube URL 或本地 MP4。必要时降采样到 720p。持久化 `{video_id, file_path}`。

2. **场景分割。** 运行 TransNetV2 或 PySceneDetect，产出 `[{scene_id, start_ms, end_ms, keyframe_path}]`。目标 100 小时：约 6k-8k 个场景。

3. **ASR 处理。** 在音频上运行 Whisper-v3-turbo；导出词级时间戳；按场景切分出逐场景的转录片段。

4. **VLM 字幕生成。** 对每个场景，用关键帧和一个简短的字幕模板调用 Gemini 2.5 Pro（或 Qwen3-VL-Max）。产出字幕 + 帧嵌入。

5. **多向量索引。** 创建带三个命名向量的 Qdrant 集合。Payload：`{video_id, scene_id, start_ms, end_ms, keyframe_url}`。

6. **查询。** 自然语言问题触发三路稠密查询；用倒数排名融合（reciprocal rank fusion）合并；取 top-k=5 个场景。

7. **时序定位。** 在排名最高的场景上运行 TimeLens 风格的适配器，细化场景内部的 (start, end) 窗口。

8. **VLM 合成。** 用查询 + 前 3 个场景片段（以图片或短视频形式）+ 转录文本调用 Gemini 2.5 Pro。要求输出 `(video_id, start_ms, end_ms)` 引用。

9. **评测。** 运行 ActivityNet-QA 和 NeXT-GQA。构建 100 条查询的自定义集合。报告整体准确率 + 按类别细分（计数、动作、描述）。

## 生产实践

```
$ video-qa ask --url=https://youtube.com/watch?v=X "how many cars pass the intersection in the first minute?"
[scene]    23 scenes detected
[asr]      transcript complete, 4m12s
[index]    69 vectors written (23 scenes x 3)
[query]    top scene: scene 3 [01:32-01:54], confidence 0.84
[ground]   refined window: [00:12-00:58]
[synth]    gemini 2.5 pro, 1.4s
answer:    5 cars pass the intersection between 00:12 and 00:58.
citations: [scene 3: 00:12-00:58]
          [frame preview at 00:14, 00:27, 00:44, 00:51, 00:57]
```

## 交付产物

交付物是 `outputs/skill-video-qa.md`。给定一个 YouTube URL 或上传的视频，流水线为场景建立索引，并用带时间戳引用的方式回答问题。

| 权重 | 评分标准 | 测量方式 |
|:-:|---|---|
| 25 | 时序定位 IoU | 在留出的定位集合上计算交并比 |
| 20 | 问答准确率 | NeXT-GQA 与自定义 100 条查询 |
| 20 | 摄取吞吐量 | 每美元能处理多少小时视频 |
| 20 | UI 与引用体验 | 时间戳链接、缩略图条、跳转到指定帧 |
| 15 | 幻觉率 | 分别统计计数类与动作类的准确率 |
| **100** | | |

## 练习

1. 在字幕生成环节把 Gemini 2.5 Pro 换成 Qwen3-VL-Max。在一个由人工评分的 50 场景样本上报告字幕质量差异。

2. 把逐场景帧嵌入压缩为单个池化向量，替代多向量方案。测量检索性能的退化幅度。

3. 构建一个"严格计数"模式：合成器把每个被计数的实例连同时间戳一起提取出来，由用户点击逐一核对。测量用户核对是否能降低幻觉。

4. 对摄取成本做基准测试：比较三种 VLM 选择下每美元能处理多少小时视频。选出性价比最优点。

5. 加入说话人分离（speaker diarization）的转录文本：在音频上运行 pyannote 说话人分离，并按说话人嵌入转录文本。演示"Alice 关于 X 说了什么？"这类查询。

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|-----------------|------------------------|
| 场景分割 | "镜头检测" | 在镜头边界处把视频切分为场景 |
| 多向量索引 | "字幕 + 帧 + 转录文本" | 为每种表示设置命名向量的 Qdrant 集合 |
| 时序定位 | "到底是什么时候发生的" | 为查询答案细化 (start, end) 时间窗口 |
| 帧嵌入 | "视觉表示" | 关键帧的向量嵌入；用于场景间的视觉相似度 |
| RRF 融合 | "倒数排名融合" | 跨多个排序列表的合并策略；混合检索的经典技巧 |
| 计数幻觉 | "数错了" | VLM 在"有多少个 X"类问题上的已知失败模式 |
| ActivityNet-QA | "视频问答基准" | 长视频问答准确率基准测试 |

## 延伸阅读

- [AI2 Molmo 2](https://allenai.org/blog/molmo2) — 开源 VLM 检查点
- [TimeLens (CVPR 2026)](https://github.com/TencentARC/TimeLens) — 大规模时序定位
- [Gemini Video long-context](https://deepmind.google/technologies/gemini) — 托管服务的参考实现
- [VideoDB](https://videodb.io) — "视频版 CRUD"API 参考
- [Twelve Labs Marengo + Pegasus](https://www.twelvelabs.io) — 商业产品参考
- [TransNetV2](https://github.com/soCzech/TransNetV2) — 场景分割模型
- [PySceneDetect](https://github.com/Breakthrough/PySceneDetect) — 经典的开源替代方案
- [ActivityNet-QA](https://arxiv.org/abs/1906.02467) — 参考评测基准
