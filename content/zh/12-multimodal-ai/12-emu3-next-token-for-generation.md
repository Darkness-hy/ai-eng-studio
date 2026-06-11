# Emu3：用下一 token 预测做图像与视频生成

> BAAI 的 Emu3（Wang et al., 2024 年 9 月）是 2024 年那个本应终结"扩散 vs 自回归"之争的成果。一个 Llama 风格的 decoder-only Transformer，只用下一 token 预测（next-token prediction）目标训练，作用在文本 + VQ 图像 token + 3D VQ 视频 token 的统一词表上，在图像生成上击败 SDXL，在感知任务上击败 LLaVA-1.6。没有 CLIP 损失，没有扩散调度。推理时用了无分类器引导（classifier-free guidance）来提升质量，但核心训练目标就是带教师强制（teacher forcing）的下一 token 预测。论文发表在 Nature 上。本课解读 Emu3 的核心论点——为什么"更好的分词器 + 规模"就足够了——并与扩散方法进行对比。

**Type:** Learn
**Languages:** Python (stdlib, 3D video tokenizer math + autoregressive sampler skeleton)
**Prerequisites:** Phase 12 · 11 (Chameleon)
**Time:** ~120 minutes

## 学习目标

- 解释为什么 Emu3 的单一损失、下一 token 目标行得通——尽管长期以来人们认为图像质量必须依赖扩散。
- 描述 3D 视频分词器：时空 VQ 码本长什么样，patch 为什么要跨越时间维度。
- 在（训练算力、推理成本、质量上限）三个维度上对比 Emu3 与 Stable Diffusion XL。
- 说出同一个 Emu3 模型扮演的三种角色：Emu3-Gen（图像生成）、Emu3-Chat（感知）、Emu3-Stage2（视频生成）。

## 问题背景

直到 2024 年的主流共识是：图像生成需要扩散模型。理由是：离散图像 token 损失太多信息，无法重建细节；而自回归采样在数千个 token 上会累积误差。Stable Diffusion、DALL-E 3、Imagen、Midjourney 全都使用某种形式的扩散。Chameleon（第 12.11 课）在小规模上部分推翻了这一观点，但质量没有追上 SDXL。

Emu3 正面迎击了这个论点。它的主张是：更好的视觉分词器 + 足够的规模 + 下一 token 损失 = 在同一个模型里既能做出超越扩散的图像生成，又能做感知。

发表时这个押注颇有争议。两年过去，开源统一生成家族（Emu3、Show-o、Janus-Pro、Transfusion）已成为研究的默认路线；前沿生产模型看起来也在使用某种变体。

## 核心概念

### Emu3 分词器

关键成分是视觉分词器。Emu3 训练了一个定制的 IBQ 类分词器（Inverse Bottleneck Quantizer，属于 SBER-MoVQGAN 家族），每个 token 对应 8x8 的分辨率压缩。一张 512x512 的图像变成 64x64 = 4096 个 token，码本大小为 32768。

这比 Chameleon 在 K=8192 下每张 512x512 图像 1024 个 token 的方案 token 数更多，但单 token 成本更低（更小的码本查找、更简单的编解码器）。关键指标：重建 PSNR 达到 30.5 dB，与 Stable Diffusion 连续潜空间的 32 dB 相当。

视频方面：3D VQ 分词器把一个时空 patch（4x4x4 像素）编码为一个整数。一段 8 FPS、4 秒的片段有 32 帧；在 256x256 分辨率、4 倍空间压缩和 4 倍时间压缩下，token 数为 (256/4) * (256/4) * (32/4) = 64 * 64 * 8 = 32,768 个 token。

分词器质量决定了上限。Emu3 的贡献有一部分就是"我们训练了一个非常好的分词器"。

### 单一损失训练

Emu3 只用一个目标：在文本 token、2D 图像 token 和 3D 视频 token 共享的词表上做下一 token 预测。训练时各模态的损失会乘上模态特定的权重系数以平衡贡献，但损失函数本身完全相同。

训练数据混合了：
- 图像生成：`<text caption> <image> image_tokens </image>`
- 图像感知：`<image> image_tokens </image> <question> text_tokens`
- 视频生成：`<text caption> <video> video_tokens </video>`
- 视频感知：与上类似。
- 纯文本：标准 NTP。

模型从数据分布中学会何时输出图像 token、何时输出文本 token。生成能力就来自模型在 `<image>` 标签之后预测图像 token 这一行为本身。

### 无分类器引导与温度

自回归图像生成在推理时配合无分类器引导（CFG）会好得多。Emu3 用了它：生成两次，一次带完整描述，一次带空描述，再用引导权重混合 logits（典型值 3.0-7.0）。这正是扩散模型使用的 CFG 技巧，被借用到了自回归场景。

温度很重要：太高会出伪影，太低会模式坍缩。Emu3 推荐的温度是：感知任务 1.0，图像生成 0.8。

### 一个模型，三种角色

Emu3 以三个功能上不同的 API 形式发布，但底层是同一套权重：

- Emu3-Gen。图像生成。输入文本，输出图像 token。
- Emu3-Chat。VQA 与图像描述。输入图像（token），输出文本。
- Emu3-Stage2。视频生成与视频 VQA。输入文本或视频，输出文本或视频。

没有任务特定的头。只是不同的提示模板。同一个检查点。

### 基准测试

来自 Emu3 论文（2024 年 9 月）：

- 图像生成：在 MJHQ-30K FID 上击败 SDXL（5.4 vs 5.6），GenEval 总分（0.54 vs 0.55——统计上打平），Deep-Eval 综合指标持平。
- 图像感知：在 VQAv2 上击败 LLaVA-1.6（75.1 vs 72.4），在 MMMU 上大致持平。
- 视频生成：4 秒片段的质量在 FVD 上与 Sora 时代有公开基准结果的模型相当。

这些数字并非全面领先——Emu3 这里赢一分、那里输一分——但"下一 token 预测就是你所需要的一切"这一主张在各模态上都站得住脚。

### 算力成本

Emu3 用一个 7B 参数模型在约 3000 亿多模态 token 上训练。GPU 小时数大致与 Llama-2-7B 预训练相当（A100 级硬件上 2k-4k GPU 年）。Stable Diffusion 3 之类的扩散模型训练预算相近，但需要独立的文本编码器和更复杂的流水线。

推理时，Emu3 每张图比 SDXL 慢：4096 个图像 token 以 30 tok/s 的速度生成，每张 512x512 图像约需 2 分钟，而 SDXL 只要 2-5 秒。投机解码（speculative decoding）和 KV 缓存优化能缩小差距，但无法消除。自回归图像生成的计算开销很大；这是目前仍然存在的权衡。

### 为什么重要

Emu3 的深层贡献在于概念层面。如果下一 token 预测能通过扩大规模在图像生成上追平扩散，那么统一模型路线（一个损失、一个主干、任意模态）就是可行的。未来的模型不再需要独立的文本编码器、独立的扩散调度器、独立的 VAE。一个 Transformer，每个模态一个分词器，然后扩大规模。

Show-o、Janus-Pro 和 InternVL-U 都在这一论点上构建或对其发起挑战。直到 2025 年，中国实验室（BAAI、DeepSeek）在这个方向上的发表比美国实验室更为积极。

## 生产实践

`code/main.py` 构建了两个玩具组件：

- 一个 2D vs 3D VQ 分词器 token 数计算器：给定（分辨率、patch、片段时长、FPS），计算图像与视频的 token 数。
- 一个带温度的、支持无分类器引导的自回归图像 token 采样器。

CFG 的实现与 Emu3 的配方一致——用引导权重混合条件与无条件 logits。

## 交付产物

本课产出 `outputs/skill-token-gen-cost-analyzer.md`。给定一个生成类产品规格（图像或视频、目标分辨率、质量档位、延迟预算），它会计算 token 数、推理成本，并在 Emu3 系与扩散方案之间做出选择。

## 练习

1. Emu3 在 8x8 压缩下每张 512x512 图像产生 4096 个 token。计算 1024x1024 和 2048x2048 对应的 token 数。推理延迟会发生什么变化？

2. 阅读 Emu3 论文 3.3 节关于视频分词器的内容。描述 3D VQ patch 的形状，并解释为什么是 4x4x4 而不是 8x8x1。

3. 无分类器引导权重 5.0 vs 3.0：视觉效果上有什么差别？在 `code/main.py` 中追踪相关数学计算。

4. 计算 Emu3-7B 在 3000 亿 token 上的训练 FLOPs，并与 Stable Diffusion 3 对比。哪个训练更贵？

5. Emu3 在 FID 上击败 SDXL，但在 VQAv2 上不敌专门的 VLM。解释为什么统一损失方法在不同基准上相对专门模型表现出不同的强项。

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|-----------------|------------------------|
| 下一 token 预测 | "NTP" | 标准自回归损失：给定 token[0..i] 预测 token[i+1]；任何模态只要完成 token 化就适用 |
| IBQ 分词器 | "Inverse bottleneck quantizer" | 一类 VQ-VAE，码本更大（32768+），重建质量优于 Chameleon 的方案 |
| 3D VQ | "时空量化器" | 以 (时间, 行, 列) 索引的码本；一个 token 覆盖一个 4x4x4 像素立方体 |
| 无分类器引导 | "CFG" | 用权重 gamma 混合条件与无条件 logits；在推理时提升图像质量 |
| 统一词表 | "共享 token" | 文本 + 图像 + 视频全部取自同一个整数空间；模型预测接下来出现的任意模态 |
| MJHQ-30K | "图像生成基准" | 包含 3 万条提示的 Midjourney 级质量基准；Emu3 在此报告 FID |

## 延伸阅读

- [Wang et al. — Emu3: Next-Token Prediction is All You Need (arXiv:2409.18869)](https://arxiv.org/abs/2409.18869)
- [Sun et al. — Emu: Generative Pretraining in Multimodality (arXiv:2307.05222)](https://arxiv.org/abs/2307.05222)
- [Liu et al. — LWM (arXiv:2402.08268)](https://arxiv.org/abs/2402.08268)
- [Yu et al. — MAGVIT-v2 (arXiv:2310.05737)](https://arxiv.org/abs/2310.05737)
- [Tian et al. — VAR (arXiv:2404.02905)](https://arxiv.org/abs/2404.02905)
