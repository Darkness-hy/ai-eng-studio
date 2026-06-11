# 毕业项目 04 —— 多模态文档问答（视觉优先的 PDF、表格与图表）

> 2026 年的文档问答前沿已经从「先 OCR 再做文本」转向视觉优先的后期交互（late interaction）。ColPali、ColQwen2.5 和 ColQwen3-omni 把每一页 PDF 当作图像处理，用多向量后期交互对其进行嵌入，让查询直接对图像 patch 做注意力匹配。在财务 10-K 报告、科学论文和手写笔记上，这种模式大幅领先 OCR 优先的方案。你将在 1 万页文档上端到端搭建这条流水线，并发布与 OCR-then-text 方案的并排对比结果。

**Type:** Capstone
**Languages:** Python (pipeline), TypeScript (viewer UI)
**Prerequisites:** Phase 4 (computer vision), Phase 5 (NLP), Phase 7 (transformers), Phase 11 (LLM engineering), Phase 12 (multimodal), Phase 17 (infrastructure)
**涉及阶段：** P4 · P5 · P7 · P11 · P12 · P17
**Time:** 30 hours

## 问题背景

企业手里堆着大量被 OCR 流水线糟蹋的 PDF：表格旋转过的扫描版 10-K、公式密集的科学论文、只有以图像形式才有意义的图表、手写批注。把这些内容当作纯文本处理，意味着丢掉一半的信息。2026 年的答案是直接在原始页面图像上做后期交互式多向量检索。ColPali（Illuin Tech）首创了这一方法；ColQwen2.5-v0.2 和 ColQwen3-omni 进一步推高了精度。在 ViDoRe v3 上，视觉优先检索以明显的优势超过 OCR-then-text——而且在图表、表格和手写内容上差距还会进一步拉大。

代价是存储和延迟。一个 ColQwen 嵌入是每页约 2048 个 patch 向量，而不是单个 1024 维向量。原始存储量会急剧膨胀。DocPruner（2026）能在没有可测量精度损失的前提下剪掉 50% 的向量。你将索引 1 万页文档，测量 ViDoRe v3 的 nDCG@5，在 2 秒内返回答案，并与 OCR-then-text 基线直接对比。

## 核心概念

后期交互的含义是：每个查询 token 与每个 patch token 逐一打分，然后对每个查询 token 取最大分数并求和。这样无需把整页压缩成单个池化向量，就能获得细粒度匹配。多向量索引（Vespa、Qdrant 多向量或 AstraDB）存储逐 patch 的嵌入，并在检索时执行 MaxSim。

答案生成器是一个视觉-语言模型（VLM），它接收查询和 top-k 检索到的页面图像，输出带证据区域（边界框或页码引用）的答案。Qwen3-VL-30B、Gemini 2.5 Pro 和 InternVL3 是 2026 年的前沿选择。对于公式和科学记号，可以接入 OCR 兜底方案（Nougat、dots.ocr），作为可选的文本通道。

评测是一个二维矩阵。一个维度是内容类型（纯文本段落、密集表格、柱状/折线图、手写笔记、公式）；另一个维度是检索方法（视觉优先后期交互 vs OCR-then-text vs 混合方案）。每个单元格记录 nDCG@5 和答案准确率。这份评测报告就是交付物。

## 架构

```
PDFs -> page renderer (PyMuPDF, 180 DPI)
           |
           v
  ColQwen2.5-v0.2 embed (multi-vector per page, ~2048 patches)
           |
           +------> DocPruner 50% compression
           |
           v
   multi-vector index (Vespa or Qdrant multi-vector)
           |
query ----+----> retrieve top-k pages (MaxSim)
           |
           v
  VLM answerer: Qwen3-VL-30B | Gemini 2.5 Pro | InternVL3
    inputs: query + top-k page images + optional OCR text
           |
           v
  answer with cited page numbers + evidence regions
           |
           v
  Streamlit / Next.js viewer: highlighted boxes on source page
```

## 技术栈

- 页面渲染：PyMuPDF（fitz），180 DPI，统一为纵向
- 后期交互模型：ColQwen2.5-v0.2 或 ColQwen3-omni（Hugging Face 上的 vidore 团队）
- 索引：带多向量字段的 Vespa，或 Qdrant 多向量，或支持 MaxSim 的 AstraDB
- 剪枝：DocPruner 2026 策略（保留高方差 patch，50% 压缩率下精度损失 < 0.5%）
- OCR 兜底（公式 / 密集表格）：dots.ocr 或 Nougat
- VLM 答案生成器：自托管 Qwen3-VL-30B 或托管版 Gemini 2.5 Pro；InternVL3 作为备选
- 评测：ViDoRe v3 基准，多页推理用 M3DocVQA
- 查看器 UI：Next.js 15，用 canvas 叠加层展示证据区域

## 从零实现

1. **数据接入。**遍历一个包含 1 万页 PDF 的语料库，涵盖 10-K 报告、科学论文和扫描文档。把每页渲染成 1536x2048 的 PNG。持久化 `{doc_id, page_num, image_path}`。

2. **嵌入。**对每张页面图像运行 ColQwen2.5-v0.2。输出形状约为 2048 个 128 维的 patch 嵌入。用 DocPruner 保留信号最强的一半。写入 Vespa 多向量字段或 Qdrant 多向量索引。

3. **查询。**对每个传入的查询，用查询塔生成嵌入（token 级嵌入）。对索引执行 MaxSim：对每个查询 token，取它与页面 patch 嵌入点积的最大值，再求和。返回 top-k 页面。

4. **答案合成。**用查询和 top-5 页面图像调用 Qwen3-VL-30B。提示词："Answer using only the supplied pages. Cite each claim by (doc_id, page) and name the region (figure, table, paragraph)."

5. **证据区域。**对答案做后处理，提取引用的区域。如果 VLM 输出边界框（Qwen3-VL 支持），就在查看器中渲染为叠加层。

6. **OCR 兜底。**对识别为公式密集的页面（基于图像方差的启发式规则），运行 Nougat 或 dots.ocr，把 OCR 文本作为图像之外的额外通道一并传入。

7. **评测。**运行 ViDoRe v3（检索 nDCG@5）和 M3DocVQA（多页问答准确率）。同时在同一语料库上用同一个答案合成器运行 OCR-then-text 流水线。产出「内容类型 × 方法」矩阵。

8. **UI。**先做 Streamlit 原型；再做 Next.js 15 生产级查看器，支持逐页的证据区域叠加显示。

## 生产实践

```
$ doc-qa ask "what was the 2024 operating margin change for segment EMEA?"
[retrieve]   top-5 pages in 320ms (ColQwen2.5, MaxSim, Vespa)
[synth]      qwen3-vl-30b, 1.4s, cited (form-10k-2024, p. 88) + (..., p. 92)
answer:
  EMEA operating margin moved from 18.2% to 16.8%, a 140bp decline.
  cited: 10-K-2024.pdf p.88 (Table 4, Segment Operating Margin)
         10-K-2024.pdf p.92 (MD&A, Operating Performance)
[viewer]     open with highlighted bounding boxes overlaid on p.88 Table 4
```

## 交付产物

`outputs/skill-doc-qa.md` 描述交付物：一个针对特定语料库调优的视觉优先多模态文档问答系统，并在 ViDoRe v3 上与 OCR-then-text 基线做了对比评测。

| 权重 | 评分项 | 衡量方式 |
|:-:|---|---|
| 25 | ViDoRe v3 / M3DocVQA 准确率 | 基准数字 vs OCR-text 基线及公开排行榜 |
| 20 | 证据区域定位 | 引用区域中确实包含答案片段的比例 |
| 20 | 存储与延迟工程 | DocPruner 压缩比、索引 p95、答案 p95 |
| 20 | 多页推理 | 在人工标注的 100 题多页问题集上的准确率 |
| 15 | 溯源检查体验 | 查看器清晰度、叠加层精确度、并排对比工具 |
| **100** | | |

## 练习

1. 在同一语料库上对比 ColQwen2.5-v0.2 与 ColQwen3-omni。哪些页面一个能答对而另一个会漏掉？给索引加一个「内容类别」标签，按类型做路由。

2. 更激进地剪枝嵌入（75%、90%）。找到压缩悬崖：ViDoRe nDCG@5 跌破 OCR 基线的临界点。

3. 构建混合方案：并行运行 OCR-then-text 和 ColQwen，用 RRF 融合，再用交叉编码器重排序。混合方案能否同时打败两者？它在哪些场景帮助最大？

4. 把 Qwen3-VL-30B 换成更小的 VLM（Qwen2.5-VL-7B）。测量每美元准确率曲线。

5. 增加手写笔记支持。渲染手写语料，用 ColQwen 嵌入，测量检索效果。与手写 OCR 流水线做对比。

## 关键术语

| 术语 | 大家怎么说 | 实际含义 |
|------|-----------------|------------------------|
| 后期交互（Late interaction） | "ColPali 式检索" | 查询 token 与页面 patch 独立打分；由 MaxSim 聚合 |
| 多向量（Multi-vector） | "逐 patch 嵌入" | 每个文档对应许多向量，而不是单个池化向量 |
| MaxSim | "后期交互打分" | 对每个查询 token，取它与文档向量相似度的最大值，再求和 |
| DocPruner | "patch 压缩" | 2026 年的剪枝方法，保留 50% 的 patch，精度损失可忽略 |
| ViDoRe v3 | "文档检索基准" | 2026 年衡量视觉文档检索的标准基准 |
| 证据区域（Evidence region） | "引用边界框" | 源页面上定位答案片段的边界框 |
| OCR 兜底（OCR fallback） | "公式通道" | 在公式或表格密集的页面上与视觉通道并行使用的文本流水线 |

## 延伸阅读

- [ColPali (Illuin Tech) repository](https://github.com/illuin-tech/colpali) —— 后期交互文档检索的参考实现
- [ColPali paper (arXiv:2407.01449)](https://arxiv.org/abs/2407.01449) —— 奠基性的方法论文
- [ColQwen family on Hugging Face](https://huggingface.co/vidore) —— 可直接用于生产的模型权重
- [M3DocRAG (Adobe)](https://arxiv.org/abs/2411.04952) —— 多页多模态 RAG 基线
- [Vespa multi-vector tutorial](https://docs.vespa.ai/en/colpali.html) —— 参考服务架构
- [Qdrant multi-vector support](https://qdrant.tech/documentation/concepts/vectors/#multivectors) —— 备选索引
- [AstraDB multi-vector](https://docs.datastax.com/en/astra-db-serverless/databases/vector-search.html) —— 备选托管索引
- [Nougat OCR](https://github.com/facebookresearch/nougat) —— 支持公式的 OCR 兜底方案
