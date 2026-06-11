# Show-o 与离散扩散统一模型

> Transfusion 混合使用连续和离散表示。Show-o（Xie 等人，2024 年 8 月）则走了另一条路：文本 token 用因果式的下一 token 预测，图像 token 沿用 MaskGIT 思路的掩码离散扩散（masked discrete diffusion）。二者共存于同一个 Transformer 中，通过混合注意力掩码区分。其结果是在一个主干网络上统一了 VQA、文生图、图像修复（inpainting）和混合模态生成——每个模态一个分词器，一种损失形式（下一 token 预测扩展为掩码预测）。本课将完整讲解 Show-o 的设计——为什么掩码离散扩散是一种并行、少步数的图像生成器——并与 Transfusion、Emu3 进行对比。

**Type:** Learn
**Languages:** Python (stdlib, masked-discrete-diffusion sampler)
**Prerequisites:** Phase 12 · 13 (Transfusion)
**Time:** ~120 minutes

## 学习目标

- 解释掩码离散扩散：先按调度均匀地掩盖 token，再让 Transformer 把它们恢复出来。
- 在速度和质量两个维度上，比较并行图像解码（Show-o、MaskGIT）与自回归图像解码（Chameleon、Emu3）。
- 说出 Show-o 在单个 checkpoint 中支持的三类任务：T2I、VQA、图像修复。
- 选择一种掩码调度（余弦、线性、截断式），并分析它对采样质量的影响。

## 问题背景

Transfusion 的双损失训练可行，但训练动态更难处理——连续扩散损失与离散 NTP 损失处在不同的数值量级上，平衡损失权重本身就是一场超参数搜索。这套架构有效，但偏复杂。

Show-o 的答案是：让两种模态都保持离散（与 Chameleon 相同），但通过掩码离散扩散并行生成图像，而不是逐个 token 顺序生成。训练目标因此简化为单一的掩码 token 预测，它能自然地推广下一 token 预测。

## 核心概念

### 掩码离散扩散（MaskGIT）

Chang 等人（2022）提出的 MaskGIT 技巧非常优雅。从一张完全被掩盖的图像出发（每个 token 都是特殊的 `<MASK>` id）。每一步并行预测所有被掩盖的 token，然后保留置信度最高的前 K 个预测，其余重新掩盖。经过约 8-16 次迭代，所有 token 都被填满。每步揭开多少 token 的调度需要调优——余弦调度效果不错。

训练很简单：从 [0, 1] 中均匀采样一个掩码比例，应用到图像的 VQ token 上，训练 Transformer 恢复被掩盖的部分。和 BERT 对文本做的事完全一样，只是规模放大到了图像生成。

### Show-o：一个 Transformer，混合掩码

Show-o 把 MaskGIT 装进了一个因果语言模型 Transformer。其注意力掩码为：

- 文本 token：因果式（标准 LLM）。
- 图像 token：图像块内部完全双向（这样被掩盖的 token 在预测时能看到图像内的所有其他 token）。
- 文本与图像之间：文本可关注先前的图像，图像可关注先前的文本。

训练在以下任务间交替：
1. 文本序列上的标准 NTP。
2. T2I 样本：文本 → 图像，掩盖图像 token，使用掩码 token 预测损失。
3. VQA 样本：图像 → 文本，掩盖文本 token（实际上就是 NTP）。

统一损失是 `<MASK>` token 上的交叉熵，它同时覆盖文本 NTP（只有最后一个 token 被"掩盖"）和图像掩码扩散（随机子集被掩盖）。

### 并行采样

Show-o 生成一张图像约需 16 步，而不是约 1000 步（逐 token 自回归）或约 20 步（扩散模型）。每一步并行预测所有被掩盖的 token；确认置信度最高的前 K 个；重复此过程。

对比：
- Chameleon / Emu3（逐 token 自回归）：N_tokens 次前向计算，每张图像通常需要 1024-4096 次。
- Transfusion（连续扩散）：约 20 步，每步一次完整的 Transformer 前向。
- Show-o（掩码离散扩散）：约 16 步，每步一次完整的 Transformer 前向。

在相近规模的模型上，Show-o 比 Chameleon 快，步数与 Transfusion 大致持平，但单步开销更低（离散词表 logits 对比连续 MSE 损失）。

### 单个 checkpoint 中的多任务

Show-o 在推理时支持四类任务，通过提示词格式选择：

- 文本生成：标准的自回归文本输出。
- VQA：输入图像，输出文本。
- T2I：输入文本，通过掩码离散扩散输出图像。
- 图像修复：输入部分 token 被掩盖的图像，填补缺失部分。

图像修复能力是掩码预测训练免费附赠的。掩盖 VQ token 网格的某个区域，把其余部分连同文本提示一起输入，预测被掩盖的 token 即可。

### 掩码调度

每步揭开多少 token 的调度决定了生成质量。Show-o 推荐余弦调度：

```
mask_ratio(t) = cos(pi * t / (2 * T))   # t = 0..T
```

第 0 步时，所有 token 都被掩盖（比例 1.0）。第 T 步时，无掩盖。余弦调度把权重集中在中间区段的比例上，那里的预测信息量最大。线性调度也可行，但会更早进入平台期。

### Show-o2

Show-o2（2025 年的后续工作，arXiv 2506.15564）对 Show-o 进行了扩展：更大的 LLM 基座、更好的分词器、改进的掩码调度。架构模式保持不变。

### Show-o 的定位

在 2026 年的分类版图中：

- 离散 token + NTP：Chameleon、Emu3。简单但推理慢。
- 离散 token + 掩码扩散：Show-o、MaskGIT、LlamaGen、Muse。支持并行采样，但仍受分词器带来的信息损失。
- 连续表示 + 扩散：Transfusion、MMDiT、DiT。质量最高，训练更复杂。
- 连续表示 + VLM 中的流匹配（flow matching）：JanusFlow、InternVL-U。最新方向。

按任务选型：如果想用一个开源模型以合理的速度同时获得 T2I + 图像修复 + VQA，选 Show-o；如果质量至上且能承担双损失的工程开销，选 Transfusion。

## 生产实践

`code/main.py` 模拟了 Show-o 的采样过程：

- 一个由 16 个 VQ token 组成的玩具网格。
- 一个模拟的"Transformer"，根据提示词和当前已揭开的 token 预测 logits。
- 使用余弦调度、共 8 步的并行掩码采样。
- 打印中间状态（掩码模式的演化过程）和最终 token。

运行它，观察掩码如何一步步消解。

## 交付产物

本课产出 `outputs/skill-unified-gen-model-picker.md`。给定一个既需要理解能力（VQA、图像描述）又需要生成能力（T2I、图像修复）、且要求开源权重的产品，在 Show-o 系、Transfusion/MMDiT 系和 Emu3 / Chameleon 系之间做出选择，并给出具体的权衡分析。

## 练习

1. 掩码离散扩散用约 16 步完成采样。为什么不能 1 步？如果在第 0 步就揭开所有 token，会出什么问题？

2. 图像修复是掩码扩散免费附赠的能力。提出一个产品用例（真实或假想均可），说明 Show-o 的图像修复在何种场景下优于专用模型。

3. 余弦调度对比线性调度：在 T=8 时逐步推算每步揭开的 token 数量。哪种更均衡？

4. 一张 512x512 的 Show-o 图像对应 1024 个 token。在词表 K=16384 下，模型输出 1024 * log2(16384) = 14,336 比特（约 1.75 KiB）的数据。Stable Diffusion 输出 512*512*24 比特 = 6,291,456 比特（约 768 KiB）的原始像素。压缩比是多少？这种压缩换来了怎样的质量？

5. 阅读 LlamaGen（arXiv:2406.06525）。LlamaGen 的类条件自回归图像模型与 Show-o 的掩码方法有何不同？

## 关键术语

| 术语 | 人们常说 | 实际含义 |
|------|-----------------|------------------------|
| 掩码离散扩散 | "MaskGIT 风格" | 训练时预测被掩盖的 token；推理时迭代地揭开置信度最高的预测 |
| 余弦调度 | "揭开调度" | 掩码比例随推理步数衰减；把置信度增长集中在中间区段 |
| 并行解码 | "一次性出所有 token" | 每步用一次前向计算预测全部被掩盖的 token，然后确认前 K 个 |
| 混合注意力 | "因果 + 双向" | 对文本 token 因果、在图像块内部双向的注意力掩码 |
| 图像修复 | "填空式生成" | 以部分 token 被掩盖的图像为条件，预测缺失部分；由训练目标免费获得 |
| 确认率 | "每步前 K 个" | 每次迭代有多少 token 被宣告"完成"；控制推理速度与质量的权衡 |

## 延伸阅读

- [Xie et al. — Show-o (arXiv:2408.12528)](https://arxiv.org/abs/2408.12528)
- [Show-o2 (arXiv:2506.15564)](https://arxiv.org/abs/2506.15564)
- [Chang et al. — MaskGIT (arXiv:2202.04200)](https://arxiv.org/abs/2202.04200)
- [Sun et al. — LlamaGen (arXiv:2406.06525)](https://arxiv.org/abs/2406.06525)
- [Chang et al. — Muse (arXiv:2301.00704)](https://arxiv.org/abs/2301.00704)
