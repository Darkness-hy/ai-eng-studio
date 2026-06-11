# 开放权重 VLM 配方：什么才真正重要

> 2024-2026 年的开放权重 VLM 文献是一片消融实验表格的森林。Apple 的 MM1 测试了图像编码器、连接器和数据配比的 13 种组合。Allen AI 的 Molmo 证明了详细人工标注字幕优于 GPT-4V 蒸馏。Cambrian-1 跑了 20 多种编码器的对比。Idefics2 形式化了五轴设计空间。Prismatic VLMs 在受控基准上比较了 27 种训练配方。在所有这些噪声中，有一小批结论在各篇论文间始终成立：图像编码器比连接器架构更重要，数据配比比这两者都更重要，详细人工标注字幕优于蒸馏出的合成数据。这节课替你读完了那些表格。

**Type:** Learn + lab
**Languages:** Python (stdlib, ablation table parser + recipe picker)
**Prerequisites:** Phase 12 · 05 (LLaVA baseline)
**Time:** ~180 minutes

## 学习目标

- 说出 VLM 五轴设计空间：图像编码器、连接器、LLM、数据配比、分辨率调度。
- 读懂 MM1 / Idefics2 / Cambrian-1 的消融实验表，并预测哪个旋钮会影响给定基准的分数。
- 在给定算力预算和任务组合的前提下，为一个新 VLM 挑选配方（编码器、连接器、数据、分辨率）。
- 解释为什么在相同 token 数下，详细人工标注字幕优于 GPT-4V 蒸馏。

## 问题背景

开放权重的 VLM 有数百个。「好」与「最先进」之间的差距大部分不在架构，而在数据、分辨率调度和编码器选择。当模型表现不佳时，知道应该先拧哪个旋钮，能帮你避免一个 500 万 GPU 小时的错误。

2023 年那一波（LLaVA-1.5、InstructBLIP、MiniGPT-4）靠的是字幕对预训练 + LLaVA-Instruct-150k。是不错的基线，但 MMMU 大约停在 35%。

2024 年那一波（MM1、Idefics2、Molmo、Cambrian-1、Prismatic VLMs）做了穷举式的消融实验。结果既出人意料又很有实用价值。

## 核心概念

### 五轴设计空间

Idefics2（Laurençon et al., 2024）命名了这些轴：

1. 图像编码器。CLIP ViT-L/14、SigLIP SO400m/14、DINOv2 ViT-g/14、InternViT-6B。各编码器在 patch 大小、分辨率和预训练目标上各不相同。
2. 连接器。MLP（2-4 层）、Q-Former（32 个查询 + 交叉注意力）、Perceiver Resampler（64 个查询）、C-Abstractor（卷积 + 双线性池化）。
3. 语言模型。Llama-3 8B / 70B、Mistral 7B、Phi-3、Gemma-2、Qwen2.5。LLM 的规模是参数开销的大头。
4. 训练数据。字幕对（CC3M、LAION）、图文交错（OBELICS、MMC4）、指令数据（LLaVA-Instruct、ShareGPT4V、PixMo、Cauldron）。
5. 分辨率调度。固定 224/336/448、AnyRes、原生动态分辨率。训练中递增或保持恒定。

每个生产级 VLM 都要在每条轴上做出选择。MMMU 分数的大部分方差由轴 1、4、5 解释——而不是你选了哪种连接器。

### 轴 1：编码器 > 连接器

MM1 第 3.2 节的结论：把 CLIP ViT-L/14 换成 SigLIP SO400m/14，MMMU 提升 3 分以上；把连接器从 MLP 换成 Perceiver Resampler，提升不到 1 分。Idefics2 复现了这一点：SigLIP > CLIP，而在相同 token 数下 Q-Former ≈ MLP ≈ Perceiver。

Cambrian-1 的「Cambrian Vision Encoders Match-Up」（Tong et al., 2024）在一个以视觉为中心的基准（CV-Bench）上跑了 20 多种编码器。排行榜顶部由 DINOv2 和 SigLIP 混合占据；CLIP 处于中游；ImageBind 和 ViT-MAE 排名更低。从 CLIP ViT-L 到 DINOv2 ViT-g/14，CV-Bench 上的差距约为 5-7 分。

2026 年开放 VLM 的默认编码器是 SigLIP 2 SO400m/14，用于语义 + 稠密特征，有时会与 DINOv2 ViT-g/14 的特征拼接（Cambrian 的「Spatial Vision Aggregator」就是这么做的）。

### 轴 2：连接器设计差别不大

MM1、Idefics2、Prismatic 和 MM-Interleaved 全都得出了同一个结论：在固定视觉 token 数的前提下，连接器架构几乎无关紧要。在相同 token 预算下，对均值池化后的 patch 接一个 2 层 MLP，与 32 查询的 Q-Former 表现相差不到 1 分。

真正重要的是 token 数。更多视觉 token = 更多 LLM 计算 = 更好的性能，但到一定程度后收益递减。每张图 64 个 token 对 OCR 来说太少；576-1024 个 token 是大多数开放 VLM 的最佳区间；2048+ 只对文档和图表有帮助。

Q-Former 与 MLP 之争是成本问题，不是质量问题：Q-Former 无论图像分辨率多高都把 token 数限制在 32-64；MLP 则输出全部 patch token。对高分辨率输入，Q-Former 能节省 LLM 上下文；对低分辨率，两者差异只是噪声。

### 轴 3：LLM 规模决定上限

把 LLM 从 7B 翻倍到 13B，在每篇 VLM 论文里都能稳定地在 MMMU 上多拿 2-4 分。到 70B 时大多数基准趋于饱和。VLM 的多模态推理上限就是 LLM 的文本推理上限——视觉编码器只能给它喂数据，不能替它推理。

这就是为什么 Qwen2.5-VL-72B 和 Claude Opus 4.7 能碾压 MMMU-Pro 和 ScreenSpot-Pro：语言大脑足够大。一个 7B 的 VLM 无法靠巧妙的连接器设计顶替 70B 的 VLM。

### 轴 4：数据——详细人工标注字幕优于蒸馏

Molmo + PixMo（Deitke et al., 2024）是 2024 年人人都该读的成果。Allen AI 让人工标注员用 1-3 分钟的密集语音转文字方式描述图像，得到 71.2 万张密集标注的图像。训练数据里没有任何 GPT-4V 蒸馏。

Molmo-72B 在 11 个基准中的 11 个上全部击败 Llama-3.2-90B-Vision。差距不在架构——在字幕质量。详细的人工标注字幕每张图包含的信息量是简短网络字幕的 5-10 倍，而且在 GPT-4V 蒸馏会产生幻觉的地方仍能保持事实准确。

ShareGPT4V（Chen et al., 2023）和 Cauldron（Idefics2）用人工 + GPT-4V 混合字幕走了同样的路线。趋势很清晰：对于 2026 年的前沿模型，字幕密度 > 字幕数量 > 蒸馏的便利性。

### 轴 5：分辨率及其调度

Idefics2 的消融实验：384 -> 448 提升 1-2 分；448 -> 980 配合图像切分（AnyRes）在 OCR 基准上再加 3-5 分。固定分辨率训练会停滞在中等精度；分辨率递增（从 224 开始，到 448 或原生分辨率结束）训练更快，最终效果更好。

Cambrian-1 做了分辨率与 token 数的权衡实验：在固定算力下，你可以选择低分辨率配更多 token，或高分辨率配更少 token。OCR 任务上高分辨率获胜；通用场景理解上低分辨率多 token 获胜。

2026 年的生产配方：Stage 1 固定 384 训练，Stage 2 对 OCR 密集型任务用最高 1280 的动态分辨率。

### Prismatic 的受控比较

Prismatic VLMs（Karamcheti et al., 2024）是把所有轴都控制住的那篇论文。相同的 13B LLM、相同的指令数据、相同的评测——每次只变动一条轴。结果：

- 每张图的视觉 token 数解释约 60% 的方差。
- 编码器选择解释约 20%。
- 连接器架构解释约 5%。
- 其余一切（数据配比、调度器、学习率）解释剩下的约 15%。

这是一个粗略的分解，但它是文献中对「我应该先消融什么」最干净的回答。

### 一个面向 2026 年的选型器

基于这些证据，2026 年新项目的默认开放 VLM 配方：

- 编码器：SigLIP 2 SO400m/14，配合 NaFlex 使用原生分辨率；如果需要分割/视觉定位，再拼接 DINOv2 ViT-g/14 的稠密特征。
- 连接器：作用于 patch token 的 2 层 MLP。除非 token 受限，否则跳过 Q-Former。
- LLM：Qwen2.5 / Llama-3.1 / Gemma 2，追求成本选 7B，追求质量选 70B，按目标延迟决定。
- 数据：PixMo + ShareGPT4V + Cauldron，再补充任务专用的指令数据。
- 分辨率：动态（长边最小 256、最大 1280 像素）。
- 调度：Stage 1 对齐（仅训练投影器），Stage 2 全量微调，Stage 3 任务专用微调。

这里的每一项默认值都能追溯到本课末尾引用论文中的一项实测消融实验。

## 生产实践

`code/main.py` 是一个消融实验表解析器加配方选型器。它编码了 MM1 和 Idefics2 的消融实验表（精简版），并支持以下查询：

- 「给定预算 X 和任务 Y，哪个配方胜出？」
- 「在 7B Llama 上把 SigLIP 换成 CLIP，预期的 MMMU 变化是多少？」
- 「想要 80% 置信度的答案，我应该先消融哪条轴？」

输出是一份带预期基准分数变化的配方排序列表，外加一条「优先消融」建议。

## 交付产物

本课产出 `outputs/skill-vlm-recipe-picker.md`。给定目标任务组合、算力预算和延迟目标，它会输出一份完整配方（编码器、连接器、LLM、数据配比、分辨率调度），并附上证明每项选择合理性的消融实验出处。它能避免工程师在每个新 VLM 项目启动时重新发明一遍 Idefics2 的消融实验表。

## 练习

1. 阅读 MM1 第 3.2 节。在固定 2B LLM、预算 5000 万张图像的条件下，哪个编码器胜出？换成 13B LLM 答案会反转吗？为什么？

2. Cambrian-1 发现，拼接 DINOv2 + SigLIP 在以视觉为中心的基准上优于单独使用任一编码器，但在 MMMU 上没有增益。预测哪些基准会受益，哪些保持不变。

3. 你的目标是在 2B LLM 上做一个移动端 UI 智能体。选出编码器、连接器、分辨率和数据配比，并用具体的消融实验表论证每项选择。

4. Molmo 发布了 4B 和 72B 两个模型。4B 能与闭源 7B VLM 竞争；72B 在 11/11 个基准上击败 Llama-3.2-90B-Vision。这对「LLM 规模平台期」假说意味着什么？

5. 设计一张消融实验表，在 7B VLM 上把数据配比质量与编码器质量分离开来。最少需要多少次训练？给出四种轴设置方案。

## 关键术语

| 术语 | 大家怎么说 | 实际含义 |
|------|-----------------|------------------------|
| 消融实验（Ablation） | 「只拧一个旋钮」 | 训练多个仅在设计空间的某一条轴上不同、其余全部保持不变的模型 |
| 连接器（Connector） | 「桥」/「投影器」 | 把视觉编码器输出映射到 LLM token 空间的可训练模块（MLP、Q-Former、Perceiver） |
| 详细人工标注字幕 | 「密集字幕」 | 人工撰写的多句描述（通常 80-300 个 token），比网页 alt 文本信息更丰富 |
| 蒸馏（Distillation） | 「GPT-4V 字幕」 | 由更强的专有 VLM 生成的训练数据；方便，但容易继承幻觉 |
| AnyRes / 动态分辨率 | 「高分辨率路径」 | 通过切片或 M-RoPE，把超过编码器原生分辨率的图像喂进去的策略 |
| 分辨率递增 | 「课程学习」 | 从低分辨率开始、逐步提高的训练调度，可加速对齐学习 |
| 以视觉为中心的基准 | 「CV-Bench / BLINK」 | 侧重细粒度视觉感知而非语言主导推理的评测 |
| PixMo | 「Molmo 的数据」 | Allen AI 的 71.2 万张密集标注图像数据集；由人工语音转写成密集字幕 |

## 延伸阅读

- [McKinzie et al. — MM1 (arXiv:2403.09611)](https://arxiv.org/abs/2403.09611)
- [Laurençon et al. — Idefics2 / What matters building VLMs (arXiv:2405.02246)](https://arxiv.org/abs/2405.02246)
- [Deitke et al. — Molmo and PixMo (arXiv:2409.17146)](https://arxiv.org/abs/2409.17146)
- [Tong et al. — Cambrian-1 (arXiv:2406.16860)](https://arxiv.org/abs/2406.16860)
- [Karamcheti et al. — Prismatic VLMs (arXiv:2402.07865)](https://arxiv.org/abs/2402.07865)
