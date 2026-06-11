# 预填充/解码分离（Disaggregated Prefill/Decode）—— NVIDIA Dynamo 与 llm-d

> 预填充（prefill）是计算受限的，解码（decode）是显存带宽受限的。把两者跑在同一块 GPU 上必然浪费其中一种资源。分离式架构将它们拆到独立的资源池中，并通过 NIXL（RDMA/InfiniBand，或回退到 TCP）在池间传输 KV 缓存。NVIDIA Dynamo（GTC 2025 发布，已 1.0 GA）位于 vLLM/SGLang/TRT-LLM 之上 —— 其 Planner Profiler 与 SLA Planner 能自动匹配预填充与解码的配比以满足 SLO。NVIDIA 公布的吞吐提升大致在这个量级 —— developer.nvidia.com（2025-06）显示，在中等延迟区间下，GB200 NVL72 + Dynamo 上的 DeepSeek-R1 MoE 约有 6 倍提升；Dynamo 产品页（developer.nvidia.com，未注明日期）则宣称 GB300 NVL72 + Dynamo 相比 Hopper 的 MoE 吞吐最高可达 50 倍。"30 倍"这个数字是社区对全栈 Blackwell + Dynamo + DeepSeek-R1 各类报告的汇总；我们没有找到任何一个明确给出 30 倍的原始来源，所以请将它视为方向性结论。llm-d（Red Hat + AWS）是 Kubernetes 原生方案：预填充 / 解码 / 路由器作为独立 Service 运行，按角色分别配置 HPA。llm-d 0.5 新增了分层 KV 卸载、缓存感知的 LoRA 路由、UCCL 网络层以及缩容到零（scale-to-zero）。经济账：对多家客户披露数据的内部汇总表明，在 SLA 不变的前提下，把 200 万美元量级的推理开支从同址部署（colocated）切换到 Dynamo 分离式部署，可以节省 30–40%（即每年 60–80 万美元）；这个 200 万美元 → 60–80 万美元的数字是内部综合估计，并非来自单一公开案例 —— 请把它当作数量级参考，而不是可引用的出处。短提示词（<512 个 token，且输出也短）不值得承担传输开销。

**Type:** Learn
**Languages:** Python (stdlib, toy disaggregated-vs-colocated simulator)
**Prerequisites:** Phase 17 · 04 (vLLM Serving Internals), Phase 17 · 08 (Inference Metrics)
**Time:** ~75 minutes

## 学习目标

- 解释为什么预填充和解码的最优 GPU 配置不同，并量化同址部署下的资源浪费。
- 画出分离式架构图：预填充池、解码池、经 NIXL 的 KV 传输、路由器。
- 说出分离式架构「不划算」的条件（短提示词、短输出）。
- 区分 NVIDIA Dynamo（栈上层编排）与 llm-d（Kubernetes 原生），并把它们对应到各自适合的运维场景。

## 问题背景

你在 8 块 H100 上运行 Llama 3.3 70B。在混合负载下（长提示词 + 短输出），由于大部分算力都花在预填充上，GPU 在解码阶段处于空闲状态。换一种负载（短提示词 + 长输出），情况则恰好相反。预填充与解码同址部署，意味着两边都要超额配置。

预算影响：20-40% 的 GPU 时间浪费在错配的资源上。你要么在用 H100 的算力跑显存带宽受限的解码，要么在用 H100 的 HBM 带宽跑计算受限的预填充。两种浪费都很昂贵。

分离式架构把预填充和解码拆到各自按瓶颈定容的独立资源池中。KV 缓存通过高带宽互连从预填充池传输到解码池。

## 核心概念

### 为什么瓶颈不同

**预填充** —— 对完整输入提示词做一次前向计算。矩阵乘法占主导；计算受限。H100 FP8 可提供约 2000 TFLOPS 的有效吞吐。批处理效率很好 —— 一次前向就能处理大量 token。

**解码** —— 一次生成一个 token，每次迭代都要读取全部权重。显存带宽受限。HBM3 提供约 3 TB/s。只有在高并发下批处理效率才好 —— 权重读取的开销会在整个批次上摊薄。

把两者同址部署：你买的是对两种工作都要好的 GPU。H100 两边都擅长，但价格不会因此变便宜。规模上去之后，你会希望预填充池用 H100 / 偏算力的机型；解码池用 H200 / 偏显存的机型，或者配合激进的量化。

### 架构

```
            ┌──────────────┐
  Request → │    Router    │ ───────────────────────┐
            └──────┬───────┘                        │
                   │                                │
                   ▼ (prompt only)                  │
            ┌──────────────┐    KV cache    ┌───────▼──────┐
            │ Prefill pool │ ─── NIXL ────► │ Decode pool  │
            │  (compute)   │                │  (memory)    │
            └──────────────┘                └──────┬───────┘
                                                   │ tokens
                                                   ▼
                                                 Client
```

NIXL 是 NVIDIA 的跨节点传输层。可用时走 RDMA/InfiniBand，否则回退到 TCP。传输延迟是实打实的 —— 在 70B FP8 模型上，一个 4K token 提示词的 KV 缓存通常需要 20-80 ms。这正是短提示词不值得分离的原因：传输税超过了收益。

### Dynamo 与 llm-d 对比

**NVIDIA Dynamo**（GTC 2025 发布，已 1.0 GA）：
- 作为编排器位于 vLLM、SGLang、TRT-LLM 之上。
- Planner Profiler 测量负载，SLA Planner 自动配置预填充与解码的配比。
- Rust 核心，Python 扩展能力。
- 吞吐提升：NVIDIA 报告在中等延迟区间下，GB200 NVL72 + Dynamo 上的 DeepSeek-R1 MoE 有 6 倍提升（developer.nvidia.com，2025-06）；社区关于全栈 Blackwell + Dynamo + DeepSeek-R1 上「最高 30 倍」的说法缺少单一原始来源，应视为方向性结论。
- GB300 NVL72 + Dynamo：据 Dynamo 产品页（developer.nvidia.com，未注明日期），相比 Hopper 的 MoE 吞吐最高可达 50 倍。

**llm-d**（Red Hat + AWS，Kubernetes 原生）：
- 预填充 / 解码 / 路由器作为独立的 Kubernetes Service 运行。
- 按角色分别配置 HPA，信号分别为队列深度（预填充）/ KV 利用率（解码）。
- `topologyConstraint packDomain: rack` 将预填充 + 解码组合打包到同一机架，以获得高带宽的 KV 传输。
- llm-d 0.5（2026）：分层 KV 卸载、缓存感知的 LoRA 路由、UCCL 网络层、缩容到零。

如果你想要一个托管式的栈上层编排器，用 Dynamo。如果你想要 Kubernetes 原生原语并且已经深度投入 CNCF 生态，用 llm-d。

### 经济账

内部综合估计（并非单一公开案例 —— 仅作数量级参考）：

- 同址部署方案下每年 200 万美元的推理开支。
- 切换为 Dynamo 分离式部署。
- 请求量不变，P99 延迟 SLA 不变。
- 报告的节省：每年 60-80 万美元（降低 30–40%）。
- 无需新硬件。

这个数字是我们从多家客户披露中综合得出的，而非来自某个可引用的单一案例；最接近的公开数据点是 Baseten 用 Dynamo KV 路由实现 TTFT 加快 2 倍 / 吞吐提升 61%（baseten.co，2025-10），以及 VAST + CoreWeave 在 40–60% KV 命中率下每美元 token 数提升 60–130% 的预测（vastdata.com，2025-12）。节省来自对每个池的精准定容；预填充密集的负载（带 8K+ 前缀的 RAG）比均衡负载受益更大。

### 什么时候不该分离

- 提示词 < 512 个 token 且输出 < 200 个 token：传输税压过收益。
- 小集群（< 4 块 GPU）：资源池没有足够的差异化空间。
- 团队无法运维两个带按角色扩缩容的 GPU 池：Dynamo 能帮忙，但并非零成本。
- 没有 RDMA 网络：TCP 的传输税更重。

### 路由器与 Phase 17 · 11 的衔接

分离式架构的路由器是 KV 缓存感知的（Phase 17 · 11）。请求会落到持有其前缀的解码池上 —— 如果没有命中，则走预填充 → 解码的路径。命中率和分离式架构是相乘的关系 —— 缓存感知路由器决定了是否还需要做一次新的预填充。

### Blackwell 上的 MoE 才是真正出数字的地方

GB300 NVL72 + Dynamo 相比 Hopper 基线展示出 50 倍的 MoE 吞吐。MoE 专家路由在预填充阶段是计算密集的，在解码阶段则是显存密集的（专家缓存），所以分离式架构是双重收益。2026 年前沿模型的推理服务以 MoE 为主（DeepSeek-V3、未来的 GPT-5 变体）。

### 应该记住的数字

基准数字会漂移 —— NVIDIA 和整个推理栈每个季度都会发布新结果。引用前请重新核实。

- GB200 NVL72 + Dynamo 上的 DeepSeek-R1：中等延迟区间下相比基线约 6 倍吞吐（developer.nvidia.com，2025-06）；社区关于全栈 Blackwell + Dynamo 上「最高 30 倍」的说法属于方向性汇总，没有单一原始来源。
- GB300 NVL72 + Dynamo：相比 Hopper 的 MoE 吞吐最高可达 50 倍（developer.nvidia.com，未注明日期）。
- 节省锚点（内部综合估计，并非单一案例）：在 SLA 不变的前提下，200 万美元年度开支中节省 60-80 万美元/年。
- 分离阈值：提示词 >512 个 token + 输出 >200 个 token。
- 经 NIXL 的 KV 传输：70B FP8 上 4K 提示词的 KV 需要 20-80 ms。

## 生产实践

`code/main.py` 模拟同址部署与分离式部署。报告吞吐、每请求成本，以及提示词长度的交叉点。

## 交付产物

本课产出 `outputs/skill-disaggregation-decider.md`。给定负载和集群，判断是否应该分离。

## 练习

1. 运行 `code/main.py`。提示词长度达到多少时分离式开始胜过同址部署？
2. 为一个 P99 前缀长度 8K、输出 300 的 RAG 服务设计预填充池和解码池。
3. Dynamo 与 llm-d：为一个纯 Kubernetes、对 Python 运行时无偏好的团队选一个。
4. 计算 KV 传输成本：70B FP8 上 4K 预填充 ≈ 500 MB KV。RDMA 100 GB/s 下传输 = 5 ms。TCP 10 GB/s 下 = 50 ms。哪个对你的 SLA 有影响？
5. MoE 专家路由会改变 KV 访问模式。对于每个 token 激活不同专家的 MoE，分离式架构会表现如何？

## 关键术语

| 术语 | 大家怎么说 | 实际含义 |
|------|----------------|------------------------|
| 分离式推理服务（Disaggregated serving） | "拆分预填充/解码" | 为两个阶段分别设立独立的 GPU 池 |
| NIXL | "NVIDIA 传输层" | Dynamo 的跨节点 KV 传输（RDMA/TCP） |
| NVIDIA Dynamo | "那个编排器" | 位于 vLLM/SGLang/TRT-LLM 之上的协调层 |
| llm-d | "Kubernetes 原生" | Red Hat + AWS 的 K8s 分离式技术栈 |
| Planner Profiler | "Dynamo 自动配置" | 测量负载，配置资源池配比 |
| SLA Planner | "Dynamo 策略" | 自动匹配预填充:解码配比以满足 SLO |
| `packDomain: rack` | "llm-d 拓扑" | 把预填充+解码打包到同一机架以加快 KV 传输 |
| UCCL | "统一集合通信" | llm-d 0.5 中支持缩容到零的网络层 |
| MoE 专家路由 | "每个 token 一个专家" | DeepSeek-V3 的模式；分离式架构对其有利 |

## 延伸阅读

- [NVIDIA — Introducing Dynamo](https://developer.nvidia.com/blog/introducing-nvidia-dynamo-a-low-latency-distributed-inference-framework-for-scaling-reasoning-ai-models/)
- [NVIDIA — Disaggregated LLM Inference on Kubernetes](https://developer.nvidia.com/blog/deploying-disaggregated-llm-inference-workloads-on-kubernetes/)
- [TensorRT-LLM Disaggregated Serving blog](https://nvidia.github.io/TensorRT-LLM/blogs/tech_blog/blog5_Disaggregated_Serving_in_TensorRT-LLM.html)
- [llm-d GitHub](https://github.com/llm-d/llm-d)
- [llm-d 0.5 release notes](https://github.com/llm-d/llm-d/releases)
