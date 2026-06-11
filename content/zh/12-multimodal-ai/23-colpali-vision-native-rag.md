# ColPali 与视觉原生文档 RAG

> 传统 RAG 把 PDF 解析成文本、切分成块、对块做嵌入、存储向量。每一步都在丢失信号：OCR 丢掉图表数据，分块切断表格行，文本嵌入忽略图片。ColPali（Faysse 等，2024 年 7 月）提出了一个更简单的问题：为什么要提取文本？直接用 PaliGemma 嵌入页面图像，用 ColBERT 式的后期交互（late interaction）做检索，把文档自带的版式、图表、字体、格式信号全部保留下来。公开基准测试显示：在视觉信息丰富的文档上，端到端准确率比文本 RAG 高 20-40%。ColQwen2、ColSmol 和 VisRAG 延续了这一模式。本课解读视觉原生 RAG 的核心论点，并构建一个微型的类 ColPali 索引器。

**Type:** Build
**Languages:** Python (stdlib, multi-vector indexer + MaxSim scorer)
**Prerequisites:** Phase 11 (LLM Engineering — RAG basics), Phase 12 · 05 (LLaVA)
**Time:** ~180 minutes

## 学习目标

- 解释双编码器检索（bi-encoder，每篇文档一个向量）与后期交互检索（每篇文档多个向量）的区别。
- 描述 ColBERT 的 MaxSim 运算，以及 ColPali 如何把它从文本 token 推广到图像块（patch）。
- 构建一个微型的类 ColPali 索引器：页面 → patch 嵌入 → 与查询词嵌入做 MaxSim → 返回 top-k 页面。
- 在发票 / 财务报告的使用场景下，对比 ColPali + Qwen2.5-VL 生成器与文本 RAG + GPT-4。

## 问题背景

对 PDF 做文本 RAG 会丢掉文档的大部分内容。财务报告的 Q3 营收增长通常画在图表里；医疗报告的检查结论写在带标注的影像上；法律合同的签名栏是一个版式事实，而不是文本事实。

文本 RAG 的流水线：

1. PDF → 通过 OCR / pdftotext 转成文本。
2. 文本 → 300-500 token 的分块。
3. 分块 → 双编码器嵌入（一个向量）。
4. 用户查询 → 嵌入 → 余弦相似度 → top-k 分块。
5. 分块 + 查询 → LLM。

五个有损步骤。图表没有被捕获，表格被切断在不同分块里，多栏排版被压平，图片标注消失了。

ColPali 的解法：跳过 OCR，直接嵌入页面图像。检索使用 ColBERT 式的后期交互，让模型在查询时能关注到细粒度的图像块。

## 核心概念

### ColBERT（2020）

ColBERT（Khattab & Zaharia，arXiv:2004.12832）是一种文本检索方法。它不是为每篇文档生成一个向量，而是为每个 token 生成一个向量。在查询时：

- 查询的各个 token 拥有自己的嵌入（N_q 个向量）。
- 文档的各个 token 也有嵌入（N_d 个向量，通常预先缓存）。
- 得分 = 对每个查询 token 取它与所有文档 token 的最大余弦相似度，再求和：Σ_i max_j cos(q_i, d_j)。

这就是 MaxSim 运算。每个查询 token「挑选」与它最匹配的文档 token，最终得分是这些最大值之和。

优点：召回率高，能处理词级语义。缺点：每篇文档要存 N_d 个向量，存储开销大。

### ColPali

ColPali（Faysse 等，arXiv:2407.01449）把 ColBERT 的模式应用到图像上。

- 每个页面由 PaliGemma（ViT + 语言模型）编码成 patch 嵌入：每页 N_p 个向量。
- 每条用户查询（文本）被编码成查询 token 嵌入：N_q 个向量。
- 得分 = Σ_i max_j cos(q_i, p_j)，即在查询文本 token 与页面图像 patch 之间做 MaxSim。
- 按总分检索 top-k 页面。

在文档入库时：用 PaliGemma 嵌入每一页，存储全部 patch 嵌入。在查询时：嵌入查询 token，对所有已索引的页面嵌入计算 MaxSim，返回 top-k 页面。

优点：在视觉信息丰富的文档上，端到端效果比文本 RAG 高 20-40%。每个 patch 向量都捕获了局部的版式和内容。

缺点：每页 N_p 个 patch × 4 字节浮点数 × D 维向量，存储量增长很快。可以用 PQ / OPQ 量化来缓解。

### ColQwen2 与 ColSmol

ColQwen2（illuin-tech，2024-2025）把 PaliGemma 换成 Qwen2-VL。基础编码器更强，检索效果更好。

ColSmol 是面向本地 / 边缘场景的小型变体。一个约 1B 参数的 ColSmol 检索器可以在消费级 GPU 上运行。

### VisRAG

VisRAG（Yu 等，arXiv:2410.10594）是另一种变体：它不在 patch 上做 MaxSim，而是用 VLM 把每页池化成单个向量，再用双编码器方式检索。索引更快、存储更小，但召回率较弱。

质量与成本的取舍：追求质量选 ColPali，追求规模选 VisRAG。

### M3DocRAG

M3DocRAG（Cho 等，arXiv:2411.04952）把多模态检索扩展到多页、多文档推理。它跨文档检索页面，为 VLM 组装一个多页上下文。

### ViDoRe —— 基准测试

ColPali 的配套基准，全称 Visual Document Retrieval Evaluation（视觉文档检索评测）。任务涵盖财务报告、科研论文、行政文件、医疗记录、操作手册。指标：nDCG@5。

ColPali-v1 在 ViDoRe 上的 nDCG@5 约为 80%；文本 RAG 在同样的文档上约为 50-60%。

### 端到端 RAG 流水线

视觉原生 RAG 的完整流程：

1. 入库：PDF → 页面图像 → PaliGemma 编码 → 存储全部 patch 嵌入。
2. 查询：用户文本 → 查询 token 嵌入 → 对所有已索引页面做 MaxSim → top-k 页面。
3. 生成：top-k 页面图像 + 查询 → VLM（Qwen2.5-VL 或 Claude）→ 答案。

全程没有 OCR。图片、图表、字体、版式全部流入最终答案。

### 存储计算

一份 50 页的财务报告，每页 729 个 patch，嵌入维度 128：

- ColPali：50 * 729 * 128 * 4 字节 = 原始约 18 MB，经 PQ 压缩后约 4 MB。
- 文本 RAG：50 个分块 * 768 维 * 4 字节 = 约 150 kB。

ColPali 每篇文档的存储量约是文本 RAG 的 30 倍。在大规模场景下，OPQ / PQ 可以把这一比例降到约 5-10 倍，通常可以接受。

### 文本 RAG 仍然占优的场景

- 没有版式信号的纯文本文档（wiki 文章、聊天记录）。文本 RAG 更简单、存储更便宜。
- 存储成本占主导的数百万页级别归档。
- 严格的监管要求，规定检索之外必须同时提供可提取的 OCR 文本。

除此之外，到 2026 年的其他场景——财务报告、科研论文、法律合同、医疗记录、UX 文档——视觉原生 RAG 全面胜出。

## 生产实践

`code/main.py`：

- 玩具版 patch 编码器：把一个「页面」（小型特征向量网格）映射为一组 patch 嵌入。
- MaxSim 打分器：计算查询 token 嵌入集合与页面 patch 集合之间的 ColBERT 式得分。
- 索引 5 个玩具页面，运行 3 条查询，返回带分数的 top-k 结果。

## 交付产物

本课产出 `outputs/skill-vision-rag-designer.md`。给定一个文档 RAG 项目，它会在 ColPali / ColQwen2 / VisRAG / 文本 RAG 中做出选择，并估算存储规模。

## 练习

1. 一份 200 页的年报，每页 729 个 patch，嵌入维度 128，4 字节浮点数。计算原始存储量和经 PQ 压缩（8 倍）后的存储量。

2. MaxSim 是 Σ_i max_j cos(q_i, p_j)。这个求和捕获了哪些简单平均相似度无法捕获的信息？

3. ColPali 以 patch 集合为单位索引页面。如果改为在词级别索引（像 ColBERT 那样），会发生什么变化？有哪些取舍？

4. 为一个 100 万页的语料库设计端到端流水线，单次查询的延迟预算为 500ms。在 ColQwen2 / VisRAG 中做出选择并说明理由。

5. 阅读 M3DocRAG（arXiv:2411.04952）。描述其多页注意力模式，以及它与单页 ColPali 检索的区别。

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|-----------------|------------------------|
| 后期交互（Late interaction） | 「ColBERT 式」 | 使用逐 token 或逐 patch 嵌入 + MaxSim 做检索，而非单个文档向量 |
| MaxSim | 「对 patch 取最大」 | 为每个查询 token 选取相似度最高的文档 token，再对所有查询 token 求和 |
| 双编码器（Bi-encoder） | 「单向量」 | 每篇文档一个向量；速度更快但损失粒度 |
| 多向量（Multi-vector） | 「每篇文档多向量」 | 每篇文档 / 每页存储 N_p 个向量；存储成本上升但召回率提升 |
| Patch 嵌入 | 「页面特征」 | VLM 编码器为每个图像块输出一个向量，按页缓存 |
| ViDoRe | 「视觉文档基准」 | ColPali 的视觉文档检索基准测试套件 |
| PQ 量化 | 「乘积量化」 | 在保持向量相似度的同时将存储压缩约 8 倍的压缩方法 |

## 延伸阅读

- [Faysse et al. — ColPali (arXiv:2407.01449)](https://arxiv.org/abs/2407.01449)
- [Khattab & Zaharia — ColBERT (arXiv:2004.12832)](https://arxiv.org/abs/2004.12832)
- [Yu et al. — VisRAG (arXiv:2410.10594)](https://arxiv.org/abs/2410.10594)
- [Cho et al. — M3DocRAG (arXiv:2411.04952)](https://arxiv.org/abs/2411.04952)
- [illuin-tech/colpali GitHub](https://github.com/illuin-tech/colpali)
