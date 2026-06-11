# EAGLE-3 投机解码的生产实践

> 投机解码（speculative decoding）把一个快速的草稿模型与目标模型配对使用。草稿模型提议 K 个 token，目标模型用一次前向传播完成验证，被接受的 token 几乎是免费的。到 2026 年，EAGLE-3 已是生产级方案——它在目标模型的隐藏状态（而非原始 token）上训练草稿头，把通用聊天场景的接受率 alpha 推到 0.6-0.8 区间。正确的问题不是"草稿模型有多快"，而是"在我的真实流量上 alpha 是多少？"如果 alpha 低于约 0.55，在高并发下投机解码就是净亏损，因为每次被拒绝的草稿都要付出第二次目标模型前向传播的代价。这节课教你先测 alpha，再开开关。

**Type:** Learn
**Languages:** Python (stdlib, toy acceptance-rate simulator)
**Prerequisites:** Phase 17 · 04 (vLLM Serving Internals), Phase 10 · 18 (Multi-Token Prediction)
**Time:** ~60 minutes

## 学习目标

- 说出投机解码的三代演进，并解释 EAGLE-3 相对 EAGLE-2 和经典草稿模型分别改变了什么。
- 定义接受率 alpha，根据 alpha 和 K（草稿长度）计算期望加速比，并找出目标并发量下的盈亏平衡 alpha。
- 解释为什么在 2026 年的 vLLM 中投机解码是显式开启（而非默认开启）的功能，以及为什么不测 alpha 就直接打开是一种生产环境反模式。
- 写出一份测量方案：用哪个基准、哪种 prompt 分布、哪个并发点、以哪个指标作为门控。

## 问题背景

解码（decode）阶段是受内存带宽限制的。在一块 H100 上运行 Llama 3.3 70B FP8，每解码一个 token 要以约 140 GB/s 的速度读取权重，却只产出一个 token。解码期间 GPU 的算力几乎闲置——瓶颈在 HBM 带宽，而不是矩阵乘法吞吐。

投机解码正是利用了这个差距。先用一个廉价的草稿模型生成 K 个候选 token，再让目标模型用一次前向传播验证全部 K 个。每个通过验证的 token 实际上是免费的（其成本摊销进了目标模型反正都要做的一次 batch-of-K 前向传播中）。

经典的草稿模型方案是用同系列的小模型（用 Llama 3.2 1B 为 Llama 3.3 70B 打草稿）。它可行，但接受率平平——小模型的分布与目标模型偏差较大。EAGLE、EAGLE-2 到 EAGLE-3 则直接在目标模型的内部状态上训练一个轻量草稿头，使草稿分布更紧密地贴合目标模型。这就是为什么 alpha 能从草稿模型方案的 0.4 提升到 EAGLE-3 的 0.6-0.8。

需要注意的是：在 2026 年的 vLLM 中，EAGLE-3 是显式开启的功能。必须明确设置 `speculative_config`。不加这个配置，就没有加速。那些没在真实流量上测过 alpha 就把开关打开的团队，往往会看到尾部延迟变得更糟，而不是更好。

## 核心概念

### 投机解码到底买到了什么

不开投机解码时，每个 token 的成本是一次目标模型前向传播。开启后，在草稿长度 K 和接受率 alpha 下，每次目标前向传播的期望产出 token 数是 `1 + K * alpha`。加速比为 `(1 + K * alpha) / (1 + epsilon)`，其中 epsilon 是草稿加验证的额外开销。取 K=5、alpha=0.7：`(1 + 5*0.7) / (1 + 0.1) = 4.5 / 1.1 = 4.1x`。现实中的数字大多集中在 2-3 倍，因为生产流量上 alpha 很少有那么高，而且 epsilon 在大 batch 下会增长。

### 为什么 alpha 是唯一重要的指标

被拒绝的 token 不会凭空消失——它们会迫使目标模型为第一个被拒 token 再做一次前向传播。在一个 alpha 跌到 0.4 的工作负载上，你要同时支付草稿开销、验证开销和重采样（re-roll）开销。在高并发下（比如 256 路并发），解码 batch 已经足够大，"只跑目标模型"与"目标模型加验证"之间的内存带宽差距会缩小。在大多数 2026 年的硬件上，alpha 低于 0.55 时投机解码就是净亏损。

alpha 随工作负载而变化。在 ShareGPT 风格的通用聊天上，用 ShareGPT 训练的 EAGLE-3 能达到 0.6-0.8。在领域特定流量（代码、医疗、法律）上，用通用数据训练的草稿头会跌到 0.4-0.6。训练一个领域专用的草稿头可以把 alpha 找回来——相比目标模型微调，这是一项轻量、快速的训练任务。

### EAGLE 各代速览

- **经典草稿模型**：同系列小模型。alpha 0.3-0.5。基础设施简单——加载两个模型，草稿模型在每次目标前向传播前跑 K 次前向。
- **EAGLE-1（2024）**：在目标模型隐藏状态（最后一层）上训练单个草稿头。alpha 约 0.5-0.6。相对目标模型只增加少量参数。
- **EAGLE-2（2025）**：自适应草稿长度与树状草稿（一次目标前向传播验证多个分支）。alpha 约 0.6-0.7。草稿调度器更复杂。
- **EAGLE-3（2025-2026）**：草稿头在目标模型的多个层（而不只是最后一层）上训练，对齐更好。通用聊天上 alpha 约 0.6-0.8。

### 2026 年的生产配方

1. 先裸跑目标模型。在目标并发量下测量基线 TTFT、ITL 和吞吐。
2. 通过 vLLM 的 `speculative_config` 启用 EAGLE-3 草稿。重新跑基准。
3. 记录接受率 alpha。vLLM V1 通过 `spec_decode_metrics.accepted_tokens_per_request` 上报该值。除以请求的草稿长度即得 alpha。
4. 如果生产流量分布上 alpha < 0.55，关闭投机解码，或训练一个领域专用的 EAGLE-3 草稿头。
5. 在生产并发量下重新跑一遍。确认 P99 ITL 没有变差。

### 生产陷阱：P99 尾部

开启投机解码后平均 ITL 会下降。但如果不做调优，P99 可能变差。被拒绝的草稿会触发两段式序列（草稿 + 验证失败 + 重采样）。在 batch 打满时，这两次前向传播会串行执行。盯住 P99 ITL，而不是 P50。

### EAGLE-3 已经部署在哪里

Google 在 2025 年把投机解码部署到了 AI Overviews（同等质量，更快响应）。vLLM V1 把 `speculative_config` 作为官方文档化的接口；V1 中的 N-gram GPU 投机解码是与 chunked prefill 兼容的变体。SGLang 支持 EAGLE-3，并将其作为前缀密集型工作负载的推荐草稿路径。

### 一行算清盈亏平衡

期望加速比：`S(alpha, K) = (1 + K*alpha) / (1 + verify_overhead)`。令 `S = 1` 解出 alpha：`alpha_breakeven = verify_overhead / K`。取典型的 verify_overhead 约 0.15 和 K=5：`alpha_breakeven = 0.03`。但这只是纯解码的数学。在高并发下，验证开销上升，且解码 batch 本身已经把权重读取摊销到多个序列上，所以实际的有效 alpha_breakeven 会爬升到约 0.45-0.55。

### 什么时候不该用投机解码

- 不在乎延迟的 batch-1 离线生成。直接用目标模型。
- 输出非常短（不到 50 个 token）。草稿开销和验证成本占主导。
- 没有领域专训草稿头的专业领域。alpha 太低。
- vLLM v0.18.0 同时开草稿模型投机解码与 `--enable-chunked-prefill`。这个组合无法构建运行。文档中标明的例外是 V1 中的 N-gram GPU 投机解码。

## 生产实践

`code/main.py` 在一系列 alpha 取值和草稿长度 K 上模拟开启与不开启投机解码的解码循环。它会打印盈亏平衡 alpha、实测加速比和尾部行为。在多组 (alpha, K) 组合上运行它，精确观察投机解码在哪里开始不再划算。

## 交付产物

这节课产出 `outputs/skill-eagle3-rollout.md`。给定目标模型、流量分布描述和并发目标，它会生成一份分阶段的 EAGLE-3 上线方案——基线基准测试、启用配置、测量 alpha、以 alpha >= 0.55 为门控、监控 P99 ITL。

## 练习

1. 运行 `code/main.py`。在 K=5 时，达到 2 倍加速需要多大的 alpha？3 倍加速呢？该结果对 verify_overhead 有多敏感？
2. 假设生产流量按 70% 通用聊天、30% 代码划分。用 ShareGPT 训练的 EAGLE-3 在通用聊天上 alpha 达 0.7，在代码上为 0.4。混合后的 alpha 是多少？投机解码是否还是净收益？
3. 阅读 vLLM 的 `speculative_config` 文档。说出三种模式（草稿模型、EAGLE、N-gram），以及哪一种与 chunked prefill 兼容。
4. 启用 EAGLE-3 后你看到平均 ITL 下降了 25%，但 P99 ITL 上升了 15%。诊断原因并提出缓解措施。
5. 计算 Llama 3.3 70B 的 EAGLE-3 草稿头的显存成本。与把 Llama 3.2 1B 当经典草稿模型来跑相比如何？

## 关键术语

| 术语 | 大家怎么说 | 实际含义 |
|------|----------------|------------------------|
| 投机解码（speculative decoding） | "草稿加验证" | 用廉价模型提议 K 个 token，目标模型用一次前向传播验证全部 K 个 |
| 接受率 alpha | "spec accept rate" | 草稿 token 被目标模型接受的比例；唯一重要的指标 |
| 草稿长度 K | "spec k" | 每次目标前向传播中草稿模型提议的 token 数；典型值 4-8 |
| 验证开销 epsilon | "spec overhead" | 相对一次普通目标前向传播，验证加重采样的额外成本；随 batch 增长 |
| EAGLE-3 | "最新的 EAGLE" | 2025-2026 年的变体；在目标模型多个层上训练草稿头；通用聊天上 alpha 0.6-0.8 |
| `speculative_config` | "vLLM spec config" | vLLM V1 中的显式开关；没有配置就没有加速 |
| N-gram 投机解码 | "N-gram 草稿" | 在 GPU 端用 prompt 中的 N-gram 查找生成草稿；与 chunked prefill 兼容 |
| 盈亏平衡 alpha | "no-op alpha" | 投机解码加速比为零时的 alpha；在生产并发量下盯住它 |
| 拒绝草稿两段式 | "重采样成本" | 草稿被拒时的两次目标前向传播；推高 P99 尾部 |

## 延伸阅读

- [vLLM — Speculative Decoding docs](https://docs.vllm.ai/en/latest/features/spec_decode/) — 关于 V1 中 `speculative_config` 与 chunked prefill 兼容性的权威来源。
- [vLLM Speculative Config API](https://docs.vllm.ai/en/latest/api/vllm/config/speculative/) — 精确的字段列表。
- [EAGLE paper (arXiv:2401.15077)](https://arxiv.org/abs/2401.15077) — EAGLE 草稿头的原始公式化表述。
- [EAGLE-2 paper (arXiv:2406.16858)](https://arxiv.org/abs/2406.16858) — 自适应草稿与树状结构。
- [UC Berkeley EECS-2025-224](https://www2.eecs.berkeley.edu/Pubs/TechRpts/2025/EECS-2025-224.html) — 带投机解码的高效 LLM 系统。
- [BentoML — Speculative Decoding](https://bentoml.com/llm/inference-optimization/speculative-decoding) — 生产上线检查清单。
