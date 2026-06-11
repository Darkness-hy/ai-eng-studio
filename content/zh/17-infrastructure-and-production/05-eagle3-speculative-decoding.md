# EAGLE-3 投机解码的生产实践

> 投机解码（speculative decoding）让一个快速的草稿模型与目标模型配对工作。草稿模型提议 K 个 token，目标模型在一次前向传播中完成全部验证，被接受的 token 等于零成本。到了 2026 年，EAGLE-3 是生产级的主流变体——它在目标模型的隐藏状态（而非原始 token）上训练草稿头，把通用对话场景的接受率 alpha 推到了 0.6-0.8 区间。正确的问题不是"草稿模型有多快"，而是"在我的流量上 alpha 是多少？"如果 alpha 跌破约 0.55，在高并发下投机解码反而是净负收益，因为每次被拒绝的草稿都要付出第二次目标模型前向传播的代价。这节课教你先测 alpha，再开开关。

**Type:** Learn
**Languages:** Python (stdlib, toy acceptance-rate simulator)
**Prerequisites:** Phase 17 · 04 (vLLM Serving Internals), Phase 10 · 18 (Multi-Token Prediction)
**Time:** ~60 minutes

## 学习目标

- 说出投机解码的三代演进，并解释 EAGLE-3 相对于 EAGLE-2 和经典草稿模型分别改变了什么。
- 定义接受率 alpha，根据 alpha 和 K（草稿长度）计算期望加速比，并找出你目标并发下的盈亏平衡 alpha。
- 解释为什么投机解码在 vLLM 2026 中是显式开启（而非默认开启），以及为什么不测 alpha 就打开开关是一种生产环境反模式。
- 写出一份测量方案：用哪个基准测试、哪种提示词分布、哪个并发点、以哪个指标作为门控。

## 问题背景

解码（decode）阶段是内存带宽受限的。在一块运行 Llama 3.3 70B FP8 的 H100 上，每解码一个 token 要读取约 140 GB/s 的权重数据，却只产出一个 token。解码期间 GPU 的算力几乎处于闲置状态——瓶颈是 HBM 带宽，而不是矩阵乘法吞吐量。

投机解码正是利用了这个差距。先用一个廉价的草稿模型生成 K 个候选 token，然后让目标模型在一次前向传播中验证全部 K 个。每个通过验证的 token 实际上是免费的（摊销进了一次目标模型本来就要做的 batch-of-K 前向计算中）。

经典的草稿模型方案使用同系列的小模型（用 Llama 3.2 1B 给 Llama 3.3 70B 打草稿）。这能用，但接受率平平——小模型的分布与目标模型有偏差。EAGLE、EAGLE-2 再到 EAGLE-3，则直接在目标模型的内部状态上训练一个轻量草稿头，使草稿的分布与目标模型贴合得紧密得多。这就是为什么 alpha 能从草稿模型方案的 0.4 提升到 EAGLE-3 的 0.6-0.8。

要注意的是：在 vLLM 2026 中 EAGLE-3 是显式开启的。必须明确设置 `speculative_config`。不设标志，就没有加速。那些不在真实流量上测 alpha 就直接打开开关的团队，常常看到尾部延迟变差而不是变好。

## 核心概念

### 投机解码到底买到了什么

不开投机解码时，每个 token 的成本是一次目标模型前向。开启后，在草稿长度 K、接受率 alpha 下，每次目标前向的期望产出 token 数是 `1 + K * alpha`。加速比为 `(1 + K * alpha) / (1 + epsilon)`，其中 epsilon 是草稿加验证的开销。当 K=5、alpha=0.7 时：`(1 + 5*0.7) / (1 + 0.1) = 4.5 / 1.1 = 4.1x`。现实数字大多落在 2-3 倍左右，因为生产流量上 alpha 很少有那么高，而且 epsilon 在大 batch 下会上涨。

### 为什么 alpha 是唯一重要的指标

被拒绝的 token 不会凭空消失——首个被拒 token 会逼迫目标模型再做一次前向。在 alpha 跌到 0.4 的负载上，你要付出草稿开销、验证开销，外加重算（re-roll）的代价。在高并发下（比如 256 并发），解码 batch 已经足够大，"只跑目标模型"和"目标模型加验证"之间的内存带宽差距会缩小。在 2026 年的大多数硬件上，alpha 低于 0.55 时投机解码是净负收益。

alpha 因负载而异。在 ShareGPT 风格的通用对话上，用 ShareGPT 训练的 EAGLE-3 能达到 0.6-0.8。而在领域特定流量（代码、医疗、法律）上，用通用数据训练的草稿头会掉到 0.4-0.6。训练一个领域特定的草稿头可以把 alpha 找回来——相比目标模型微调，这是个轻量、快速的训练任务。

### EAGLE 各代速览

- **经典草稿模型**：同系列小模型。Alpha 0.3-0.5。基础设施简单——加载两个模型，每次目标前向草稿模型跑 K 次前向。
- **EAGLE-1（2024）**：在目标模型隐藏状态（最后一层）上训练的单个草稿头。Alpha 约 0.5-0.6。相对目标模型的参数开销很小。
- **EAGLE-2（2025）**：自适应草稿长度加树状草稿（在一次目标前向中验证多个分支）。Alpha 约 0.6-0.7。草稿调度器更复杂。
- **EAGLE-3（2025-2026）**：草稿头在目标模型的多个层（不只最后一层）上训练，对齐更好。通用对话上 alpha 约 0.6-0.8。

### 2026 年的生产配方

1. 先裸跑目标模型上线。在目标并发下测出基线 TTFT、ITL 和吞吐量。
2. 通过 vLLM 的 `speculative_config` 启用 EAGLE-3 草稿。重跑基准测试。
3. 记录接受率 alpha。vLLM V1 以 `spec_decode_metrics.accepted_tokens_per_request` 报告该指标。除以请求的草稿长度即得 alpha。
4. 如果生产流量分布上 alpha < 0.55，要么关掉投机解码，要么训练一个领域特定的 EAGLE-3 草稿头。
5. 在生产并发下重跑。确认 P99 ITL 没有变差。

### 生产陷阱：P99 尾部

开启投机解码后平均 ITL 会下降。但如果不做调优，P99 可能变差。被拒绝的草稿会触发两遍计算的序列（草稿 + 验证失败 + 重算）。在满 batch 状态下，这两遍计算会串行执行。盯住 P99 ITL，而不是 P50。

### EAGLE-3 已经部署在哪里

Google 在 2025 年将投机解码部署到了 AI Overviews（质量不变，响应更快）。vLLM V1 把 `speculative_config` 作为文档化的接口；V1 中的 N-gram GPU 投机解码是与 chunked prefill 兼容的变体。SGLang 支持 EAGLE-3，并将其作为前缀密集型负载的推荐草稿路径。

### 一行盈亏平衡数学

期望加速比：`S(alpha, K) = (1 + K*alpha) / (1 + verify_overhead)`。令 `S = 1` 解出 alpha：`alpha_breakeven = verify_overhead / K`。在典型的 verify_overhead 约 0.15、K=5 时：`alpha_breakeven = 0.03`。但这只是纯解码层面的算术。在高并发下验证开销上升，而且解码 batch 本身已经把内存读取摊销到多条序列上，所以实践中有效的 alpha_breakeven 会爬升到约 0.45-0.55。

### 什么时候不该用投机解码

- 不在乎延迟的 batch-1 离线生成。直接用目标模型。
- 输出很短（低于 50 个 token）。草稿开销和验证成本占主导。
- 没有领域训练草稿头的专业领域。Alpha 太低。
- vLLM v0.18.0 加草稿模型投机解码再加 `--enable-chunked-prefill`。这个组合无法编译。文档中的例外是 V1 中的 N-gram GPU 投机解码。

## 生产实践

`code/main.py` 在一系列 alpha 值和草稿长度 K 上模拟开启与不开启投机解码的解码循环。它会打印盈亏平衡 alpha、实测加速比和尾部行为。在多组 (alpha, K) 组合上运行它，精确观察投机解码从哪里开始不再划算。

## 交付产物

这节课产出 `outputs/skill-eagle3-rollout.md`。给定目标模型、流量分布描述和并发目标，它会生成一份分阶段的 EAGLE-3 上线方案——跑基线基准、启用配置、测量 alpha、以 alpha >= 0.55 作为门控、监控 P99 ITL。

## 练习

1. 运行 `code/main.py`。在 K=5 时，要达到 2 倍加速需要多大的 alpha？3 倍呢？这个结果对 verify_overhead 有多敏感？
2. 假设生产流量由 70% 通用对话和 30% 代码组成。用 ShareGPT 训练的 EAGLE-3 在通用对话上 alpha 达到 0.7，在代码上是 0.4。混合后的 alpha 是多少？投机解码是净正收益吗？
3. 阅读 vLLM 的 `speculative_config` 文档。说出三种模式（草稿模型、EAGLE、N-gram），以及其中哪一种与 chunked prefill 兼容。
4. 启用 EAGLE-3 后你看到平均 ITL 下降了 25%，但 P99 ITL 上升了 15%。诊断原因并提出缓解方案。
5. 计算 Llama 3.3 70B 的 EAGLE-3 草稿头的内存成本。与用 Llama 3.2 1B 作经典草稿模型相比如何？

## 关键术语

| 术语 | 大家怎么说 | 实际含义 |
|------|----------------|------------------------|
| 投机解码（Speculative decoding） | "草稿加验证" | 用廉价模型提议 K 个 token，在一次目标前向中验证全部 K 个 |
| 接受率 alpha | "spec accept rate" | 草稿 token 被目标模型接受的比例；唯一真正重要的指标 |
| 草稿长度 K | "spec k" | 每次目标前向草稿模型提议的 token 数；典型值 4-8 |
| 验证开销 epsilon | "spec overhead" | 相比单纯目标前向，验证加重算的额外成本；随 batch 增长 |
| EAGLE-3 | "最新的 EAGLE" | 2025-2026 年的变体；在目标模型多个层上训练草稿头；通用对话上 alpha 0.6-0.8 |
| `speculative_config` | "vLLM spec config" | vLLM V1 中的显式开关；不设置就意味着没有加速 |
| N-gram 投机解码 | "N-gram draft" | GPU 侧草稿方案，在提示词中做 N-gram 查找；与 chunked prefill 兼容 |
| 盈亏平衡 alpha | "no-op alpha" | 投机解码加速比归零时的 alpha；在生产并发下要盯住它 |
| 拒稿两遍计算（Rejected-draft two-pass） | "重算成本" | 草稿被拒时要做两次目标前向；是 P99 尾部恶化的主因 |

## 延伸阅读

- [vLLM — Speculative Decoding docs](https://docs.vllm.ai/en/latest/features/spec_decode/) —— 关于 `speculative_config` 以及 V1 中 chunked prefill 兼容性的权威来源。
- [vLLM Speculative Config API](https://docs.vllm.ai/en/latest/api/vllm/config/speculative/) —— 精确的字段列表。
- [EAGLE paper (arXiv:2401.15077)](https://arxiv.org/abs/2401.15077) —— 最初的 EAGLE 草稿头形式化定义。
- [EAGLE-2 paper (arXiv:2406.16858)](https://arxiv.org/abs/2406.16858) —— 自适应草稿与树状结构。
- [UC Berkeley EECS-2025-224](https://www2.eecs.berkeley.edu/Pubs/TechRpts/2025/EECS-2025-224.html) —— 带投机解码的高效 LLM 系统。
- [BentoML — Speculative Decoding](https://bentoml.com/llm/inference-optimization/speculative-decoding) —— 生产上线检查清单。
