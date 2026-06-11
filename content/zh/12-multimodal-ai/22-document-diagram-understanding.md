# 文档与图表理解

> 文档不是照片。一份 PDF、科学论文、发票或手写表单包含版面布局、表格、图表、脚注、页眉和语义结构，这些都是单纯的图像理解无法捕捉的。VLM 出现之前的技术栈是一条流水线：Tesseract OCR + LayoutLMv3 + 表格抽取启发式规则。VLM 浪潮用免 OCR（OCR-free）模型取代了这套方案——Donut（2022）、Nougat（2023）、DocLLM（2023）——它们直接输出结构化标记。到了 2026 年，前沿做法就是"把页面图像以 2576px 原生分辨率喂给 Claude Opus 4.7"，结构化标记输出顺带就有了。本节课梳理文档 AI 的三个时代演进脉络。

**Type:** Build
**Languages:** Python (stdlib, layout-aware document parser skeleton)
**Prerequisites:** Phase 12 · 05 (LLaVA), Phase 5 (NLP)
**Time:** ~180 minutes

## 学习目标

- 解释文档 AI 的三个时代：OCR 流水线、免 OCR、VLM 原生。
- 描述 LayoutLMv3 的三路输入流：文本、版面布局（bbox）、图像块（image patches），以及统一掩码训练。
- 比较 Donut（免 OCR，图像 → 标记）、Nougat（科学论文 → LaTeX）、DocLLM（版面感知的生成式模型）、PaliGemma 2（VLM 原生）。
- 为新任务选择文档模型（发票、科学论文、手写表单、中文票据）。

## 问题背景

"理解这份 PDF"看似简单，实则很难。信息分布在：

- 文本内容（占信号的 90%）。
- 版面布局（页眉、脚注、侧边栏、双栏排版）。
- 表格（行、列、合并单元格）。
- 图形与图表。
- 手写批注。
- 字体与排版（标题与正文的区别）。

原始 OCR 只导出文本，其余全部丢失。一个关注发票的系统需要知道"Total: $1,245"来自右下角，而不是来自某个脚注。

## 核心概念

### 第一时代——OCR 流水线（2021 年前）

经典技术栈：

1. PDF → 每页转成图像。
2. Tesseract（或商业 OCR）提取文本，并给出每个词的边界框。
3. 版面分析器识别区块（页眉、表格、段落）。
4. 表格结构识别器解析表格。
5. 领域规则 + 正则表达式抽取字段。

对干净的印刷文本有效。但在手写体、倾斜扫描件、复杂表格、非英文文字上会失效。每一种失败模式都需要一条定制的异常处理路径。

### TrOCR（2021）

TrOCR（Li et al., arXiv:2109.10282）用一个在合成 + 真实文本图像上训练的 Transformer 编码器-解码器，取代了 Tesseract 的经典 CNN-CTC 方案。在手写和多语言文本上取得了明显胜利。它依然是流水线（先检测，再 TrOCR，再版面分析），但 OCR 这一步有了巨大提升。

### 第二时代——免 OCR（2022-2023）

第一批免 OCR 模型的思路是：彻底跳过检测，直接把图像像素映射为结构化输出。

Donut（Kim et al., arXiv:2111.15664）：
- 编码器-解码器 Transformer，编码器为 Swin-B。
- 输出可以是表单理解的 JSON、摘要任务的 markdown，或任意任务特定的 schema。
- 没有 OCR，没有版面分析，没有检测。

Nougat（Blecher et al., arXiv:2308.13418）：
- 专门在科学论文上训练。
- 输出是 LaTeX / markdown。
- 能处理公式、多栏排版、图表。
- 所有 arXiv 解析器都在调用的那个模型。

它们是专才而非通才。把科学论文喂给 Donut 会失败；把发票喂给 Nougat 也会失败。

### LayoutLMv3（2022）

另一条路线。LayoutLMv3（Huang et al., arXiv:2204.08387）保留 OCR，但加入了版面理解：

- 三路输入流：OCR 文本 token、每个 token 的二维边界框、图像块。
- 跨三种模态的掩码训练目标（掩码文本、掩码图像块、掩码版面）。
- 下游任务：分类、实体抽取、表格问答。

LayoutLMv3 是基于 OCR 的文档理解的巅峰。在表单和发票上表现强劲。需要上游提供 OCR 结果。在标准化文档基准上拥有 VLM 之前最好的精度。

### DocLLM（2023）

DocLLM（Wang et al., arXiv:2401.00908）是 LayoutLM 的生成式同门。它以版面 token 为条件生成自由形式的回答。更适合文档问答；但仍然依赖 OCR 输入。

### 第三时代——VLM 原生（2024 起）

2024 年，VLM 已经强到可以完全取代流水线。把整页图像以高分辨率喂给 VLM，提问，得到答案。

- LLaVA-NeXT 的 336 分块 AnyRes 适用于小型文档。
- Qwen2.5-VL 的动态分辨率可以原生处理 2048+ 像素。
- Claude Opus 4.7 支持 2576px 的文档。
- PaliGemma 2（2025 年 4 月）专门针对文档 + 手写体进行训练。

VLM 原生方案与 OCR 流水线之间的差距迅速缩小。到 2026 年，VLM 原生方案在以下场景胜出：

- 场景文字（手写 + 印刷、混合文字体系）。
- 含合并单元格的复杂表格。
- 嵌在正文中的数学公式。
- 带文字标注的图表。

OCR 流水线仍然在以下场景胜出：

- 海量纯扫描件工作负载，且单页延迟很关键。
- 流水线可靠性（确定性失败 vs VLM 幻觉）。
- 要求 OCR 输出可审计的受监管环境。

### Claude 4.7 / GPT-5 前沿

在 2576 像素原生输入下，前沿 VLM 的文档理解已接近人类精度。2026 年初的基准数据：

- DocVQA：Claude 4.7 约 95.1，PaliGemma 2 约 88.4，Nougat 约 77.3，流水线版 LayoutLMv3 约 83。
- ChartQA：Claude 4.7 约 92.2，GPT-4V 约 78。
- VisualMRC：Claude 4.7 约 94。

闭源模型的领先主要来自分辨率和基座 LLM 的规模。7B 量级的开源模型落后几个点，但正在追赶。

### 数学公式与 LaTeX 输出

科学论文需要公式的精确 LaTeX 输出。Nougat 就是为此训练的。带 LaTeX 训练目标的 VLM（Qwen2.5-VL-Math、Nougat 衍生模型）能产出可用的 LaTeX。没有显式 LaTeX 训练的 VLM 产出的转写可读但不精确。

2026 年的科学论文流水线做法：先用 Nougat 处理 PDF，再用 VLM 处理棘手的页面。

### 手写体

依然是最难的子任务。印刷 + 手写混排（医生病历、填写好的表单）是 OCR 流水线在成本上仍胜过 VLM 的场景。纯手写场景的 VLM 正在进步（Claude 4.7、PaliGemma 2）。

### 2026 年选型配方

对一个新的文档 AI 项目：

- 大规模纯印刷发票：LayoutLMv3 + 规则，成本效率高。
- 混合文档（科学论文 + 手写 + 表单）：VLM 原生（PaliGemma 2 或 Qwen2.5-VL）。
- 完整 arXiv 摄取：Nougat 处理数学公式，VLM 处理图表。
- 监管场景：OCR 流水线 + VLM 校验器做交叉核对。

## 生产实践

`code/main.py`：

- 一个玩具级版面感知分词器：给定 (text, bbox) 对，生成 LayoutLMv3 风格的输入。
- 一个 Donut 风格的任务 schema 生成器：表单的 JSON 模板。
- 比较 OCR 流水线、Donut、Nougat 和 VLM 原生方案的每页 token 预算。

## 交付产物

本节课产出 `outputs/skill-document-ai-stack-picker.md`。给定一个文档 AI 项目（领域、规模、质量、监管要求），在 OCR 流水线、免 OCR 专用模型和 VLM 原生方案之间做出选择。

## 练习

1. 你的项目每天要处理 1000 万张发票。哪种技术栈能在不损失精度的前提下最小化每页成本？

2. 为什么 LayoutLMv3 在表单问答上优于纯 CLIP 路线的 VLM，却在场景文字上表现更差？bbox 输入流牺牲了什么？

3. Nougat 生成 LaTeX。提出一个 VLM 原生输出在 LaTeX 保真度上胜过 Nougat 的测试用例，以及一个 Nougat 胜出的用例。

4. 阅读 PaliGemma 2 论文（Google, 2024）。相比 PaliGemma 1，提升文档精度的关键训练数据新增项是什么？

5. 设计一个满足监管要求的混合方案：OCR 流水线为主，VLM 为辅做交叉核对。出现分歧时如何裁决？

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|-----------------|------------------------|
| OCR 流水线 | "Tesseract 式" | 分阶段技术栈：检测 -> OCR -> 版面 -> 规则；确定性强但脆弱 |
| 免 OCR | "Donut 式" | 跳过显式 OCR 的图像到输出 Transformer；单一模型 |
| 版面感知 | "LayoutLM" | 输入包含每个 token 的 bbox 坐标；跨模态统一掩码 |
| VLM 原生 | "前沿 VLM" | 把页面图像以高分辨率直接喂给 Claude/GPT/Qwen VLM；无流水线 |
| DocVQA | "文档基准" | 文档视觉问答标准基准；引用最多的指标 |
| 标记输出 | "LaTeX / MD" | 用结构化输出格式取代自由文本；为下游自动化提供基础 |

## 延伸阅读

- [Li et al. — TrOCR (arXiv:2109.10282)](https://arxiv.org/abs/2109.10282)
- [Blecher et al. — Nougat (arXiv:2308.13418)](https://arxiv.org/abs/2308.13418)
- [Huang et al. — LayoutLMv3 (arXiv:2204.08387)](https://arxiv.org/abs/2204.08387)
- [Kim et al. — Donut (arXiv:2111.15664)](https://arxiv.org/abs/2111.15664)
- [Wang et al. — DocLLM (arXiv:2401.00908)](https://arxiv.org/abs/2401.00908)
