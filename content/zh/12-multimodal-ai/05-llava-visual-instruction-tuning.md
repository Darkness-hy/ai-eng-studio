# LLaVA 与视觉指令微调

> LLaVA（2023 年 4 月）是这个星球上被复刻次数最多的多模态架构。它用一个两层 MLP 取代了 BLIP-2 的 Q-Former，用朴素的 token 拼接取代了 Flamingo 的门控交叉注意力，并在 GPT-4 仅凭纯文本描述生成的 15.8 万条视觉指令对话上完成训练。2023 到 2026 年间构建过 VLM 的从业者，几乎都在构建 LLaVA 的某种变体。LLaVA-1.5 加入了 AnyRes。LLaVA-NeXT 提升了分辨率。LLaVA-OneVision 用一套训练方案统一了单图、多图与视频。本课将解读这套训练方案，亲手实现投影器，并解释为什么"更简单的方案赢了"。

**Type:** Build
**Languages:** Python (stdlib, projector + instruction-template builder)
**Prerequisites:** Phase 12 · 02 (CLIP), Phase 11 (LLM Engineering — instruction tuning)
**Time:** ~180 minutes

## 学习目标

- 构建一个两层 MLP 投影器（projector），把 ViT 的图块嵌入（维度 1024）映射到 LLM 的嵌入维度（维度 4096）。
- 走通 LLaVA 的两阶段训练方案：（1）在 55.8 万条图文描述对上做投影器对齐，（2）在 15.8 万条 GPT-4 生成的对话上做视觉指令微调。
- 构造 LLaVA 格式的提示词，包含图像 token 占位符、系统提示，以及 user/assistant 对话轮次。
- 解释尽管 Q-Former 在 token 预算上占优，社区为什么还是从 Q-Former 转向了 MLP。

## 问题背景

BLIP-2 的 Q-Former（第 12.03 课）把一张图像压缩成 32 个 token。干净、高效、刷榜好用。但它有两个问题。

第一，Q-Former 虽然可训练，但它的损失并不是最终任务。第一阶段训练 ITC+ITM+ITG，第二阶段训练 LM 损失。这些查询向量学到的是某种中间表示，LLM 还得再去解码它。信息在瓶颈处丢失了。

第二，Q-Former 占用 1.88 亿参数，而在 LLaVA 诞生的 2023 年的算力规模下，你必须针对目标 LLM 协同设计它。换 LLM，就要重训 Q-Former；换视觉编码器，也要重训。每种组合都是一个独立的研发项目。

LLaVA 的答案简单到令人尴尬：拿 ViT 的 576 个图块 token，每个都过一个两层 MLP（`1024 → 4096 → 4096`），然后把这 576 个全部塞进 LLM 的输入序列。没有瓶颈。没有用古怪目标做第一阶段预训练。只用直接的 LM 损失训练这个 MLP。

数据从哪来？LLaVA 的第二个洞见：用（纯文本的）GPT-4 来生成指令数据。把图像对应的 COCO 描述和边界框数据喂给 GPT-4，让它生成对话、描述和复杂推理问题。15.8 万条指令-回复对话，零成本到手。不需要人工标注。

结果是：一个在 8 张 A100 上跑一天就能训完的 VLM，在 MMMU 上击败了 Flamingo，并开源了一个社区可以扩展的检查点。到 2023 年底，它已经催生了 50 多个分支项目。

## 核心概念

### 架构

13B 规模的 LLaVA-1.5：
- 视觉编码器：CLIP ViT-L/14 @ 336（第一阶段冻结，第二阶段可选解冻）。
- 投影器：带 GELU 激活的两层 MLP，`1024 → 4096 → 4096`。
- LLM：Vicuna-13B（后来换成 Llama-3.1-8B）。

对一张图像 + 文本提示的前向传播：

```
img -> ViT -> 576 patches of dim 1024
patches -> MLP -> 576 tokens of dim 4096
prompt: system + "<image>" placeholder + user question
replace <image> token with the 576 projected tokens
feed the full sequence to the LLM
decode response
```

这张图像占用 LLM 上下文中的 576 个 token。在 2048 上下文长度下，留给文本的还有 1472 个 token；在 32k 上下文下，这点占用就是个零头。

### 第一阶段：投影器对齐

冻结 ViT。冻结 LLM。只训练那个两层 MLP。数据集：55.8 万对图文描述（LAION-CC-SBU）。损失：以投影后的图像 token 为条件，对描述文本做语言建模。

以 batch size 128 跑一个 epoch，几小时就能完成。投影器学会把 ViT 空间映射到 LLM 空间。不需要任何任务相关的监督信号。

### 第二阶段：视觉指令微调

投影器继续保持可训练。解冻 LLM（通常全量解冻，有时用 LoRA）。在 15.8 万条视觉指令对话上训练。

指令数据是关键诀窍。Liu 等人是这样生成数据的：
1. 取一张 COCO 图像。
2. 提取其文本描述（5 条人工标注的描述 + 边界框列表）。
3. 用三种提示模板发给 GPT-4：
   - 对话：「围绕这张图像，生成一段用户与助手之间的多轮对话。」
   - 详细描述：「为这张图像给出一段丰富、详尽的描述。」
   - 复杂推理：「提出一个需要对图像进行推理才能回答的问题，然后回答它。」
4. 把 GPT-4 的输出解析成（指令，回复）对。

整个过程完全没有接触图像本身——只用了文本描述。GPT-4 会幻觉出貌似合理的图像内容。有一些噪声，但确实奏效：15.8 万条对话就足以解锁对话能力。

### 社区为什么复刻这套方案

- 没有需要专门调试的第一阶段损失。全程只用 LM 损失。
- 投影器几小时就能训完，而不是几天。
- 只需重训投影器就能换 LLM（LLaVA-Llama2、LLaVA-Mistral、LLaVA-Llama3）。
- 视觉指令数据流水线基于 GPT-4，为新领域重新生成数据的成本很低。

### LLaVA-1.5 与 LLaVA-NeXT

LLaVA-1.5（2023 年 10 月）新增：
- 把学术任务数据（VQA、OKVQA、RefCOCO）混入指令微调。
- 更好的系统提示。
- 上下文从 2048 → 32k。

LLaVA-NeXT（2024 年 1 月）新增：
- AnyRes：把高分辨率图像切成 2x2 或 1x3 的 336x336 网格切片，再加一张全局低分辨率缩略图。每个切片对应 576 个 token；每张图像总计约 2880 个视觉 token。OCR 和图表任务的表现大幅跃升。
- 用 ShareGPT4V（高质量的 GPT-4V 描述）改进了指令数据配比。
- 更强的基座 LLM（Mistral-7B、Yi-34B）。

### LLaVA-OneVision

第 12.08 课会深入讲解 OneVision。简短版本：投影器不变，但用一套课程式训练（curriculum）在一个模型中覆盖单图、多图和视频，共享同一份视觉 token 预算。

### 与 Q-Former 的对比

| | Q-Former (BLIP-2) | MLP (LLaVA) |
|---|---|---|
| 每张图像的视觉 token 数 | 32 | 576（基础）或 2880（AnyRes） |
| 可训练参数 | 188M + LM | 40M + LM |
| 第一阶段损失 | ITC+ITM+ITG | 仅 LM |
| 更换 LLM | 需要重训 | 极少量重训即可替换 |
| 多图 | 别扭 | 自然（直接拼接） |
| 视频 | 别扭 | 自然（逐帧拼接） |
| Token 预算 | 小 | 大 |

MLP 在简单性和 token 灵活性上胜出。Q-Former 在 token 预算上胜出。到 2023 年底，token 预算不再是约束瓶颈（LLM 上下文增长到 32k-128k 以上），简单性成了主导因素。

### 提示词格式

```
A chat between a curious human and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the human's questions. USER: <image> Describe this image in detail. ASSISTANT: The image shows ...
```

`<image>` 是一个占位 token。在分词之前，它会被替换为那 576 个视觉 token（AnyRes 下为 2880 个）。分词器看到的序列比它训练时见过的略长，但 LLM 能处理这种新型输入，因为第一阶段已经教会了它。

### 参数经济学

LLaVA-1.5-7B 的参数拆解：
- CLIP ViT-L/14 @ 336：303M（第一阶段冻结，第二阶段常解冻）。
- 投影器（2 个线性层）：约 22M 可训练参数。
- Llama-7B：7B。
- 总计：73 亿参数。第二阶段可训练部分：完整的 7B + 22M 投影器。

第二阶段的训练成本：8 张 A100 上约 20 小时。这是关键数字——一天、一个节点、可复现。这就是 LLaVA 得以传播的原因。

## 生产实践

`code/main.py` 实现了：

1. 两层 MLP 投影器（玩具规模为维度 16 → 32 → 32），用纯 Python 编写。
2. 提示词构建流水线：系统提示 + 被 N 个投影 token 替换的 `<image>` + 用户轮次 + 助手生成占位符。
3. 一个可视化工具，展示 576 个视觉 token 在 LLM 上下文中的占比（占 2k / 32k / 128k 上下文的百分比）。

## 交付产物

本课产出 `outputs/skill-llava-vibes-eval.md`。给定一个 LLaVA 系列的检查点，它会运行一套包含 10 条提示的体感评测（3 条图像描述、3 条 VQA、2 条推理、2 条拒答），并输出一份人类可读的评分卡。它不是基准测试，而是一个冒烟测试，用来确认投影器和 LLM 衔接良好。

## 练习

1. 计算 `1024 → 4096 → 4096` 的两层 MLP 投影器的可训练参数量。算上 GELU 和偏置项，它占 LLaVA-13B 的多大比例？

2. 构造一个「拒答」场景的 LLaVA 提示词——图像中包含一个私人个体。写出期望的助手回复。为什么 LLaVA 应该在零样本（zero-shot）情况下就拒绝回答？需要什么样的训练数据来强化这种拒答行为？

3. 阅读 LLaVA-NeXT 博客中关于 AnyRes 的章节。计算一张 1344x672 图像在 AnyRes 下的视觉 token 数量，并与 336x336 基础分辨率下的 576 个 token 做对比。

4. LLaVA 第一阶段的投影器是用描述文本上的 LM 损失训练的。如果跳过第一阶段，直接进行第二阶段（视觉指令微调），会发生什么？请引用 Prismatic VLMs 的消融实验（arXiv:2402.07865）作答。

5. LLaVA-Instruct-150k 用 GPT-4 加 COCO 描述来生成指令。针对一个新领域（医学 X 光片、卫星影像），描述生成领域指令数据的四步流水线。每一步可能出什么问题？

## 关键术语

| 术语 | 大家怎么说 | 实际含义 |
|------|----------------|------------------------|
| 投影器（Projector） | 「MLP 桥」 | 带 GELU 的两层 MLP，把 ViT 维度映射到 LLM 维度 |
| 图像 token | 「<image> 占位符」 | 提示词中的标记，在推理前被替换为 N 个投影后的视觉 token |
| 视觉指令微调 | 「LLaVA 第二阶段」 | 在 GPT-4 生成的（图像，指令，回复）三元组上训练 |
| 第一阶段对齐 | 「投影器预训练」 | 冻结 ViT 和 LLM，用描述文本上的 LM 损失训练投影器 |
| AnyRes | 「多切片拼贴」 | 把高分辨率图像切成网格切片，并拼接每个切片的视觉 token |
| LLaVA-Instruct | 「GPT-4 生成的」 | 由 COCO 描述 + GPT-4 合成的 15.8 万条指令-回复对 |
| 视觉编码器冻结 | 「主干锁定」 | CLIP 权重在第一阶段不更新，有时第二阶段也不更新 |
| ShareGPT4V | 「更好的描述」 | 由 GPT-4V 生成的 100 万条稠密描述，用于更高质量的对齐 |
| VQA | 「视觉问答」 | 针对图像回答自由形式问题的任务 |
| Prismatic VLMs | 「设计空间论文」 | Karamcheti 2024 年的消融研究，系统性地测试投影器与数据的各种选择 |

## 延伸阅读

- [Liu et al. — Visual Instruction Tuning (arXiv:2304.08485)](https://arxiv.org/abs/2304.08485) — LLaVA 论文。
- [Liu et al. — Improved Baselines with Visual Instruction Tuning (arXiv:2310.03744)](https://arxiv.org/abs/2310.03744) — LLaVA-1.5。
- [Chen et al. — ShareGPT4V (arXiv:2311.12793)](https://arxiv.org/abs/2311.12793) — 稠密描述数据集。
- [Karamcheti et al. — Prismatic VLMs (arXiv:2402.07865)](https://arxiv.org/abs/2402.07865) — 设计空间消融研究。
- [Li et al. — LLaVA-OneVision (arXiv:2408.03326)](https://arxiv.org/abs/2408.03326) — 统一单图、多图与视频。
