# 多区域 LLM 服务与 KV 缓存局部性

> 对于带缓存的 LLM 推理，轮询（round-robin）负载均衡不仅没用，反而有害。如果请求没有落在持有其前缀缓存的节点上，就要支付完整的预填充（prefill）成本——长提示词下 P50 约为 800 ms，而缓存命中时只需约 80 ms。2026 年的生产模式是缓存感知路由器（用 Rust 编写的 vLLM Router、llm-d router）：它消费 KV 缓存事件，按前缀哈希匹配进行路由。近期研究（GORGO）则把跨区域网络延迟作为显式项纳入路由目标。商用的"跨区域推理"产品（Bedrock cross-region inference、GKE 多集群网关）把推理当作黑盒——它们解决的是可用性，而不是 TTFT。JPMorgan 和 Mayo Clinic 在 2024 年 11 月执行的 us-east-1 故障切换耗时约 22 分钟。灾备（DR）的现实是：32% 的 LLM 灾备失败，原因是团队备份了模型权重，却忘了分词器文件或量化配置。

**Type:** Learn
**Languages:** Python (stdlib, toy prefix-cache-aware router simulator)
**Prerequisites:** Phase 17 · 04 (vLLM Serving), Phase 17 · 06 (SGLang RadixAttention)
**Time:** ~60 minutes

## 学习目标

- 解释为什么轮询负载均衡会破坏带缓存的推理，并量化由此产生的 TTFT 损失。
- 画出缓存感知路由器的结构图：输入（KV 缓存事件）、算法（前缀哈希匹配）、平局裁决（GPU 利用率）。
- 说出导致 32% LLM 灾备失败的原因（缺失分词器文件 / 量化配置），并给出一份"三文件"灾备清单。
- 区分商用跨区域产品（Bedrock CRI、GKE Multi-Cluster Gateway）与 KV 感知路由。

## 问题背景

你的服务运行在 us-east-1、us-west-2 和 eu-west-1 三个区域。你在前面挂了一个 ALB，采用轮询策略。生产环境的前缀缓存命中率掉到 8%，TTFT P50 涨了三倍。vLLM 日志显示每个请求都在支付完整的预填充成本。

轮询对无状态服务是最优的。但 LLM 推理天生有状态——KV 缓存编码了模型见过的所有内容。盲目路由就是把请求路由到错误的缓存上。

另一方面，你的团队有一份灾备计划：把模型权重跨区域备份到 S3。某天一个区域宕机，你尝试故障切换，副本却拒绝启动。原来 tokenizer.json、量化配置和 RoPE 缩放配置放在另一个没有同步的 bucket 里。

多区域 LLM 服务是一个缓存问题、一个路由问题、一个灾备卫生问题——而不是一个负载均衡器问题。

## 核心概念

### 缓存感知路由

请求携带提示词到达。路由器对前缀做哈希（比如前 512 个 token），然后询问每个副本"你缓存了这个前缀吗？"。副本在分配和驱逐缓存块时，会通过发布/订阅通道发布 KV 缓存事件。路由器选择有匹配的副本；若无人匹配，则回退到基于 GPU 利用率的平局裁决。

**vLLM Router**（Rust 实现，2026 年 production-stack）：订阅 `kv.cache.block_added` 事件，维护一个前缀哈希 → 副本的索引，以 O(1) 查找完成路由。无匹配时回退到最短队列深度策略。

**llm-d router**：同样的模式，Kubernetes 原生。通过 ControlPlane API 发布事件。

**SGLang RadixAttention**（Phase 17 · 06）是副本内部的对应机制。跨副本路由严格位于其上游。

### 关键数字

2K-token 提示词、Llama 3.3 70B FP8、H100 上的 TTFT P50：
- 缓存命中（同一副本，前缀驻留）：约 80 ms。
- 缓存未命中（冷预填充）：约 800 ms。

10 倍差距。如果路由器在多副本间达到 60-80% 的前缀缓存命中率，你就能以 N 副本的容量逼近单副本的性能。如果只有 10%，那就只是朴素的水平扩展。

### 跨区域引入新约束——网络延迟

跨区域 RTT：
- us-east-1 ↔ us-west-2：约 65 ms。
- us-east-1 ↔ eu-west-1：约 75 ms。
- us-east-1 ↔ ap-southeast-1：约 220 ms。

如果路由把一个 us-east-1 的请求送到 ap-southeast-1 上的热前缀，省下的预填充时间（800 → 80 ms）会被 440 ms 的往返延迟完全吞掉。GORGO（2026 年研究）把这一点显式化——联合最小化 `prefill_time + network_latency`，而不是只优化预填充。通常的答案是：路由保持区域内，只有在预填充占绝对主导的超大（数 MB）前缀上才考虑跨区域。

### 商用"跨区域推理"在这里帮不上忙

AWS Bedrock cross-region inference 在容量吃紧时自动把请求路由到其他区域。它优化的是可用性而不是 TTFT，并且把推理当作黑盒。GKE Multi-Cluster Gateway 也一样——服务级别的故障切换，对 KV 缓存毫无感知。

即使使用这些产品，你仍然需要应用层的缓存感知路由器。它们处理的是"us-east-1 着火了"的场景，缓存感知路由处理的是 TTFT 的场景。

### 灾备卫生——32% 的文件缺失问题

2026 年被广泛引用的统计：32% 的 LLM 灾备失败是因为团队备份了权重，却忘了：

- `tokenizer.json` 或 `tokenizer.model`
- 量化配置（`quantize_config.json`、AWQ scales、GPTQ zero-points）
- 模型特定配置（RoPE 缩放、注意力掩码、对话模板）
- 引擎配置（`vllm_config.yaml`、采样默认值、LoRA 适配器清单）

解决办法是一份最小化的"三文件"灾备清单（DR manifest）：

1. HF 模型仓库下的所有文件（权重 + 配置 + 分词器）。
2. 引擎特定的服务配置。
3. 部署清单（K8s YAML、Dockerfile、依赖锁文件）。

此外：每季度演练一次灾备。JPMorgan 在 2024 年 11 月的 us-east-1 演练能做到 22 分钟恢复，靠的就是反复演练过的预案。

### 数据驻留是另一个独立维度

欧盟客户的 PHI 数据不能离开欧盟。如果你的缓存感知路由器为了前缀匹配把一个源自巴黎的请求发到 us-east-1，那么无论 TTFT 收益多大，你都已经违反了 GDPR。先按数据驻留边界对路由器分区，再去优化缓存。

### 应该记住的数字

- 缓存命中 vs 未命中的 TTFT 差距：约 10 倍（2K 提示词下 80 ms vs 800 ms）。
- 美欧之间跨区域 RTT：约 75 ms。
- 灾备失败：32% 因缺失分词器 / 量化配置。
- JPMorgan us-east-1 故障切换（2024 年 11 月）：22 分钟（SLA 为 30 分钟）。

## 生产实践

`code/main.py` 在一个多区域工作负载上模拟三种路由策略（轮询、区域内缓存感知、全局缓存感知），并报告缓存命中率、TTFT P50/P99 和跨区域流量账单。

## 交付产物

本课产出 `outputs/skill-multi-region-router.md`。给定区域、数据驻留约束和 SLA，设计一套路由方案。

## 练习

1. 运行 `code/main.py`。在 75 ms RTT 的前提下，提示词长度达到多少时跨区域路由开始优于仅本地路由？
2. 你的缓存命中率从 70% 跌到 12%。诊断三个可能原因，并给出能确认每个原因的可观测指标。
3. 为一个用 vLLM 服务、带 5 个 LoRA 适配器的 70B AWQ 量化模型设计一份灾备清单。列出每一个文件和配置。
4. 论证 Bedrock cross-region inference 对一家有严格 TTFT SLO 的金融科技公司是否"够用"。引用具体行为作为依据。
5. 一个源自巴黎的请求在 us-east-1 匹配到了前缀。要不要路由过去？写出这条策略。

## 关键术语

| 术语 | 大家怎么说 | 实际含义 |
|------|----------------|------------------------|
| 缓存感知路由（Cache-aware routing） | "智能 LB" | 按前缀哈希匹配，把请求路由到持有 KV 缓存的副本 |
| KV 缓存事件 | "缓存 pub-sub" | 副本发布缓存块的添加/驱逐事件；路由器据此建立索引 |
| 前缀哈希（Prefix hash） | "缓存键" | 对前 N 个 token 做哈希，用作路由器的查找键 |
| GORGO | "跨区域路由研究" | arXiv 2602.11688；把网络延迟作为显式优化项 |
| 跨区域推理（Cross-region inference） | "Bedrock CRI" | AWS 产品；可用性故障切换，不感知 TTFT |
| 灾备清单（DR manifest） | "备份列表" | 恢复服务所需的每一个文件——不只是权重 |
| 数据驻留（Data residency） | "GDPR 边界" | 关于哪个区域可以接触用户数据的法律约束 |
| RTT | "往返时延" | 网络延迟；美欧约 75 ms，美国-亚太约 220 ms |
| LLM 感知 LB | "缓存命中 LB" | 作为产品品类的缓存感知路由器 |

## 延伸阅读

- [BentoML — Multi-cloud and cross-region inference](https://bentoml.com/llm/infrastructure-and-operations/multi-cloud-and-cross-region-inference)
- [arXiv — GORGO (2602.11688)](https://arxiv.org/html/2602.11688v1) —— 带网络延迟项的跨区域 KV 缓存复用。
- [TianPan — Multi-Region LLM Serving Cache Locality](https://tianpan.co/blog/2026-04-17-multi-region-llm-serving-data-residency-routing)
- [AWS Bedrock Cross-Region Inference](https://docs.aws.amazon.com/bedrock/latest/userguide/cross-region-inference.html) —— 可用性故障切换文档。
- [vLLM Production Stack Router](https://github.com/vllm-project/production-stack) —— 缓存感知路由器源码。
