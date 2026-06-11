# Kubernetes 上的 GPU 自动扩缩容 — Karpenter、KAI Scheduler 与 Gang Scheduling

> 三层架构，而不是一层。Karpenter 动态供给节点（不到一分钟，比 Cluster Autoscaler 快 40%）。KAI Scheduler 负责 gang scheduling（成组调度）、拓扑感知和层级队列——它能避免「8 缺 1」的部分分配陷阱：七个节点空等空烧，只为等那一块缺失的 GPU。应用层自动扩缩器（NVIDIA Dynamo Planner、llm-d Workload Variant Autoscaler）基于推理特有的信号扩缩——队列深度、KV cache 利用率——而不是 CPU 或 DCGM 占空比。经典的 HPA 陷阱在于 `DCGM_FI_DEV_GPU_UTIL` 是一个占空比指标：100% 可能是 10 个请求，也可能是 100 个。vLLM 会预分配 KV cache 显存，所以显存占用永远不会触发缩容。本课教你如何组合这三层，并避开 Karpenter 默认的 `WhenEmptyOrUnderutilized` 策略——它会在推理进行到一半时终止正在运行的 GPU 任务。

**Type:** Learn
**Languages:** Python (stdlib, toy queue-depth autoscaler simulator)
**Prerequisites:** Phase 17 · 02 (Inference Platform Economics), Phase 17 · 04 (vLLM Serving Internals)
**Time:** ~75 minutes

## 学习目标

- 画出自动扩缩容的三层结构（节点供给、gang scheduling、应用层），并说出每一层使用的工具。
- 解释为什么 `DCGM_FI_DEV_GPU_UTIL` 对 vLLM 来说是错误的 HPA 信号，并说出两个替代信号（队列深度、KV cache 利用率）。
- 描述 gang scheduling，以及 KAI Scheduler 所防止的部分分配失败模式（8 块 GPU 中 7 块空闲）。
- 说出会终止正在运行的 GPU 任务的 Karpenter 整合策略（`WhenEmptyOrUnderutilized`），并给出 2026 年的安全替代方案。

## 问题背景

你的团队在 Kubernetes 上部署了一个 LLM 服务。你配置了 HPA，用 `DCGM_FI_DEV_GPU_UTIL` 作为信号。工作时间段服务的利用率钉在 100%。HPA 从不扩容——它认为你已经满载了。你手动加了一个副本，TTFT 立刻下降。HPA 仍然不扩容。这个信号在骗你。

另一个场景：你用 Cluster Autoscaler 管理节点。凌晨 2 点来了一个 100 万 token 的 prompt；集群花了 3 分钟供给节点，请求超时了。

再一个场景：你部署一个 70B 模型，需要跨 2 个节点的 8 块 GPU。集群有 7 块空闲 GPU，剩下 1 块分散在 3 个节点上。Cluster Autoscaler 为缺失的那 1 块 GPU 供给新节点。在 Kubernetes 把最后一块 GPU 拉起来的 4 分钟里，七个节点空等着烧钱。

三层，三种不同的失败模式。2026 年的 GPU 感知自动扩缩容不是「打开 HPA」就完事,而是把节点供给、gang scheduling 和应用信号扩缩组合起来。

## 核心概念

### 第 1 层 — 节点供给（Karpenter）

Karpenter 监视处于 pending 状态的 pod，并在约 45-60 秒内供给节点（Cluster Autoscaler 供给 GPU 节点通常需要 90-120 秒）。它根据 `NodePool` 约束动态选择实例类型——如果你的 pod 需要 8 块 H100 而集群中没有匹配的节点，Karpenter 会直接供给一个，而不是去扩容某个已有的节点组。

**整合陷阱**：Karpenter 默认的 `consolidationPolicy: WhenEmptyOrUnderutilized` 对 GPU 池来说很危险。它会终止一个正在运行的 GPU 节点，把 pod 迁移到更便宜、规格更匹配的实例上。对推理工作负载来说，这意味着驱逐正在处理的请求，并在新节点上重新加载一个 70B 模型。损失是几分钟的容量外加请求失败。

GPU 池的安全配置：

```yaml
disruption:
  consolidationPolicy: WhenEmpty
  consolidateAfter: 1h
```

允许 Karpenter 在一小时后整合真正空闲的节点，但绝不驱逐正在运行的任务。

### 第 2 层 — gang scheduling（KAI Scheduler）

KAI Scheduler（项目原名「Karp」，后更名）处理默认 kube-scheduler 做不到的事情：

**Gang scheduling（成组调度）** — 全有或全无的调度。一个需要 8 块 GPU 的分布式推理 pod，要么 8 块一起启动，要么一块都不启动。没有它，你就会掉进部分分配陷阱：8 个 pod 启动了 7 个，无限期等待，持续烧钱。

**拓扑感知** — 知道哪些 GPU 共享 NVLink，哪些在同一机架上，哪些之间有 InfiniBand 互连，并据此放置 pod。一个 DeepSeek-V3 67B 张量并行工作负载必须留在同一个 NVLink 域内；KAI Scheduler 会遵守这一点。

**层级队列** — 多个团队带着优先级和配额竞争同一个 GPU 池。只有在优先级规则允许的情况下，团队 B 的训练任务才能抢占团队 A 的生产高峰资源。

KAI 作为二级调度器与 kube-scheduler 并行部署；你通过给工作负载加注解来使用它。Ray 和 vLLM production-stack 都提供了集成。

### 第 3 层 — 应用层信号

**HPA 陷阱**：`DCGM_FI_DEV_GPU_UTIL` 是占空比（duty-cycle）指标——它衡量的是 GPU 在每个采样间隔内是否在干活。100% 的利用率可能意味着 10 个并发请求，也可能是 100 个；两种情况下 GPU 都是忙的。基于占空比扩缩就是盲目扩缩。

更糟的是，vLLM 及类似引擎会预分配 KV cache 显存（最高到 `--gpu-memory-utilization`）。哪怕只有一个请求，显存占用也维持在 90% 左右。基于显存的 HPA 永远不会缩容。

**2026 年的替代信号**：

- 队列深度（等待 prefill 的请求数量）。
- KV cache 利用率（分配给活跃序列的 block 占比）。
- 单副本 P99 TTFT（你的 SLA 信号）。
- Goodput（每秒满足全部 SLO 的请求数）。

NVIDIA Dynamo Planner 和 llm-d Workload Variant Autoscaler 消费这些信号并扩缩副本数。对 LLM 服务来说，它们可以完全取代 HPA。

### 何时用什么

| 扩缩决策 | 工具 |
|----------------|------|
| 增删节点 | Karpenter |
| 调度多 GPU 任务 | KAI Scheduler |
| 增删副本 | Dynamo Planner / llm-d WVA（或基于队列深度的自定义 HPA） |
| 选择 GPU 类型 | Karpenter NodePool |
| 抢占低优先级任务 | KAI Scheduler 队列 |

### 分离式 prefill/decode 让一切变复杂

如果你运行分离式 prefill/decode（Phase 17 · 17），你就有两类扩缩触发条件不同的 pod：prefill pod 基于队列深度扩缩，decode pod 基于 KV cache 压力扩缩。llm-d 把它们暴露为独立的 `Services`，各自配置按角色划分的 HPA。不要试图用一个 HPA 同时管这两类 pod。

### 冷启动在这里同样重要

冷启动缓解（Phase 17 · 10）正是节点供给时间变得用户可见的地方。Karpenter 45-60 秒的预热，加上 20GB 的模型加载，再加上引擎初始化，意味着一个从零开始的请求要花 2-5 分钟。对 SLO 关键路径保持一个热池（`min_workers=1`），或者在应用层使用 Modal 风格的检查点（checkpointing）。

### 应该记住的数字

- Karpenter 节点供给：约 45-60 秒，对比 Cluster Autoscaler 约 90-120 秒（GPU 节点）。
- KAI Scheduler 防止部分分配浪费——「8 缺 1」陷阱。
- 把 `DCGM_FI_DEV_GPU_UTIL` 当 HPA 信号：不可用；改用队列深度或 KV 利用率。
- Karpenter 的 `WhenEmptyOrUnderutilized`：会终止正在运行的 GPU 任务。推理场景用 `WhenEmpty + consolidateAfter: 1h`。

```figure
autoscaling
```

## 生产实践

`code/main.py` 在一个突发性 GPU 工作负载上模拟三层自动扩缩器，对比朴素 HPA（占空比）、队列深度 HPA 和 KAI gang 调度扩缩三种方案，报告未满足的请求数、GPU 空闲分钟数和一个综合得分。

## 交付产物

本课产出 `outputs/skill-gpu-autoscaler-plan.md`。给定集群拓扑、工作负载形态和 SLO，它会设计一套三层自动扩缩容方案。

## 练习

1. 运行 `code/main.py`。在突发性工作负载下，朴素占空比 HPA 丢掉了多少个队列深度 HPA 能接住的请求？差距从哪里来？
2. 为一个在 H100 SXM5 上以 FP8 服务 Llama 3.3 70B 的集群设计一个 Karpenter NodePool。指定 `capacity-type`、`disruption.consolidationPolicy`、`consolidateAfter`，以及一个把非 GPU 工作负载挡在这些节点之外的 taint。
3. 你的团队报告部署卡在 Pending 状态，原因是「有可用 GPU 但 pod 调度不上去」。诊断一下——问题出在 Karpenter、kube-scheduler 还是 KAI Scheduler？哪些指标能确认？
4. 为分离式 prefill pod 选一个自动扩缩信号，再为 decode pod 选一个不同的信号。分别给出理由。
5. 计算 `WhenEmptyOrUnderutilized` 整合陷阱在一个 24x7 生产服务上的成本，该服务平均每天发生 60 次 P99 TTFT > 10 秒导致的请求丢弃事件。

## 关键术语

| 术语 | 通俗说法 | 实际含义 |
|------|----------------|------------------------|
| Karpenter | 「节点供给器」 | Kubernetes 节点自动扩缩器；分钟级以内供给 |
| Cluster Autoscaler | 「老一代扩缩器」 | Kubernetes 节点自动扩缩器的前身；更慢，基于节点组 |
| KAI Scheduler | 「GPU 调度器」 | 提供 gang 调度 + 拓扑感知 + 队列的二级调度器 |
| Gang scheduling | 「全有或全无」 | 原子化调度 N 个 pod，要么全部调度，要么全部推迟 |
| 拓扑感知 | 「机架感知」 | 基于 NVLink/IB/机架位置放置 pod |
| `DCGM_FI_DEV_GPU_UTIL` | 「GPU 利用率」 | 占空比指标；不是 LLM 的扩缩信号 |
| 队列深度 | 「等待中的请求」 | prefill 受限场景下正确的 HPA 信号 |
| KV cache 利用率 | 「显存压力」 | decode 受限场景下正确的 HPA 信号 |
| 整合（Consolidation） | 「Karpenter 整合」 | 终止节点并换到更便宜的实例类型 |
| `WhenEmpty + 1h` | 「安全整合」 | 不会驱逐正在运行的 GPU 任务的策略 |

## 延伸阅读

- [KAI Scheduler GitHub](https://github.com/kai-scheduler/KAI-Scheduler) — 设计文档与配置示例。
- [Karpenter Disruption Controls](https://karpenter.sh/docs/concepts/disruption/) — 整合策略语义与 GPU 安全默认值。
- [NVIDIA — Disaggregated LLM Inference on Kubernetes](https://developer.nvidia.com/blog/deploying-disaggregated-llm-inference-workloads-on-kubernetes/) — Dynamo Planner 扩缩信号。
- [Ray docs — KAI Scheduler for RayClusters](https://docs.ray.io/en/latest/cluster/kubernetes/k8s-ecosystem/kai-scheduler.html) — Ray 集成模式。
- [AWS EKS Compute and Autoscaling Best Practices](https://docs.aws.amazon.com/eks/latest/best-practices/aiml-compute.html) — 托管 Kubernetes 的专项指导。
- [llm-d GitHub](https://github.com/llm-d/llm-d) — Workload Variant Autoscaler 设计。
