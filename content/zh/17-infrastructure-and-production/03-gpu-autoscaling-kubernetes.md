# Kubernetes 上的 GPU 自动扩缩容 — Karpenter、KAI Scheduler 与 Gang Scheduling

> 是三层架构，不是一层。Karpenter 动态供给节点（不到一分钟，比 Cluster Autoscaler 快 40%）。KAI Scheduler 负责成组调度（gang scheduling）、拓扑感知和层级队列——它能避免「8 缺 1」的部分分配陷阱：七个节点干等着烧钱，只为等那一块缺失的 GPU。应用层自动扩缩器（NVIDIA Dynamo Planner、llm-d Workload Variant Autoscaler）基于推理特有的信号扩缩——队列深度、KV 缓存利用率——而不是 CPU 或 DCGM 占空比。经典的 HPA 陷阱在于 `DCGM_FI_DEV_GPU_UTIL` 是一个占空比指标：100% 可能是 10 个请求，也可能是 100 个。vLLM 会预分配 KV 缓存显存，所以显存指标永远不会触发缩容。本课教你组合这三层架构，并避开 Karpenter 默认的 `WhenEmptyOrUnderutilized` 策略——它会在推理进行到一半时终止正在运行的 GPU 作业。

**Type:** Learn
**Languages:** Python (stdlib, toy queue-depth autoscaler simulator)
**Prerequisites:** Phase 17 · 02 (Inference Platform Economics), Phase 17 · 04 (vLLM Serving Internals)
**Time:** ~75 minutes

## 学习目标

- 画出自动扩缩容的三层架构图（节点供给、成组调度、应用层），并说出每一层使用的工具。
- 解释为什么 `DCGM_FI_DEV_GPU_UTIL` 对 vLLM 来说是错误的 HPA 信号，并说出两个替代信号（队列深度、KV 缓存利用率）。
- 描述成组调度，以及 KAI Scheduler 所避免的部分分配失败模式（8 块 GPU 中 7 块闲置）。
- 说出会终止正在运行的 GPU 作业的 Karpenter 整合策略（`WhenEmptyOrUnderutilized`），并给出 2026 年的安全替代方案。

## 问题背景

你的团队在 Kubernetes 上部署了一个 LLM 服务。你配置了 HPA，用 `DCGM_FI_DEV_GPU_UTIL` 作为信号。工作时间内服务的利用率一直钉在 100%。HPA 从不扩容——它认为你已经满载了。你手动加了一个副本，TTFT 立刻下降。HPA 还是不动。这个信号在骗你。

另一边，你用 Cluster Autoscaler 管理节点。凌晨 2 点来了一个 100 万 token 的提示词；集群花了 3 分钟供给节点，请求超时了。

再另一边，你要部署一个需要跨 2 个节点、共 8 块 GPU 的 70B 模型。集群里有 7 块空闲 GPU，还有 1 块分散在 3 个节点上。Cluster Autoscaler 为缺的那 1 块 GPU 供给了一个新节点。在 Kubernetes 把最后一块 GPU 拉起来的 4 分钟里，七个节点干等着烧钱。

三层架构，三种不同的失败模式。2026 年的 GPU 感知自动扩缩容不是「打开 HPA」就完事，而是把节点供给、成组调度和应用信号扩缩组合起来。

## 核心概念

### 第 1 层 — 节点供给（Karpenter）

Karpenter 监视处于 Pending 状态的 Pod，并在约 45-60 秒内供给节点（Cluster Autoscaler 供给 GPU 节点通常需要 90-120 秒）。它根据 `NodePool` 约束动态选择实例类型——如果你的 Pod 需要 8 块 H100 而集群中没有匹配的节点，Karpenter 会直接供给一个，而不是扩展某个现有的节点组。

**整合陷阱**：Karpenter 默认的 `consolidationPolicy: WhenEmptyOrUnderutilized` 对 GPU 池来说很危险。它会终止一个正在运行的 GPU 节点，把 Pod 迁移到更便宜、规格更合适的实例上。对推理工作负载来说，这意味着驱逐正在处理的请求，并在新节点上重新加载一个 70B 模型。损失是数分钟的算力外加请求失败。

GPU 池的安全配置：

```yaml
disruption:
  consolidationPolicy: WhenEmpty
  consolidateAfter: 1h
```

让 Karpenter 在一小时后整合真正空闲的节点，但永远不会驱逐正在运行的作业。

### 第 2 层 — 成组调度（KAI Scheduler）

KAI Scheduler（项目原名 "Karp"，后改名）处理默认 kube-scheduler 做不到的事：

**成组调度（gang scheduling）** — 全有或全无地调度。一个需要 8 块 GPU 的分布式推理 Pod，要么 8 个一起启动，要么一个都不启动。没有它，你就会掉进部分分配陷阱：8 个 Pod 启动了 7 个，无限期等待，白白烧钱。

**拓扑感知（topology awareness）** — 知道哪些 GPU 共享 NVLink、哪些在同一机架上、哪些之间有 InfiniBand 连接，并据此放置 Pod。一个 DeepSeek-V3 67B 张量并行工作负载必须留在同一个 NVLink 域内；KAI Scheduler 会遵守这一点。

**层级队列（hierarchical queues）** — 多个团队带着优先级和配额竞争同一个 GPU 池。只有在优先级规则允许的情况下，团队 B 的训练作业才能抢占团队 A 的生产应急容量。

KAI 与 kube-scheduler 并存，作为第二调度器部署；你通过给工作负载加注解来使用它。Ray 和 vLLM production-stack 都已集成。

### 第 3 层 — 应用层信号

**HPA 陷阱**：`DCGM_FI_DEV_GPU_UTIL` 是一个占空比指标——它衡量的是 GPU 在每个采样间隔内是否在干活。100% 利用率可能意味着 10 个并发请求，也可能是 100 个；两种情况下 GPU 都是忙的。基于占空比扩缩就是盲目扩缩。

更糟的是，vLLM 及类似引擎会预分配 KV 缓存显存（最高至 `--gpu-memory-utilization`）。哪怕只有一个请求，显存占用也维持在 90% 附近。基于显存的 HPA 永远不会缩容。

**2026 年的替代信号**：

- 队列深度（等待 prefill 的请求数量）。
- KV 缓存利用率（分配给活跃序列的 block 占比）。
- 单副本 P99 TTFT（你的 SLA 信号）。
- Goodput（每秒满足全部 SLO 的请求数）。

NVIDIA Dynamo Planner 和 llm-d Workload Variant Autoscaler 消费这些信号并扩缩副本。在 LLM 服务场景下，它们完全取代了 HPA。

### 什么场景用什么工具

| 扩缩决策 | 工具 |
|----------------|------|
| 增减节点 | Karpenter |
| 调度多 GPU 作业 | KAI Scheduler |
| 增减副本 | Dynamo Planner / llm-d WVA（或基于队列深度的自定义 HPA） |
| 选择 GPU 类型 | Karpenter NodePool |
| 抢占低优先级作业 | KAI Scheduler 队列 |

### 分离式 prefill/decode 让一切更复杂

如果你运行分离式 prefill/decode（Phase 17 · 17），你就有两类扩缩触发条件不同的 Pod：prefill Pod 基于队列深度扩缩，decode Pod 基于 KV 缓存压力扩缩。llm-d 将它们暴露为独立的 `Services`，每个角色配各自的 HPA。不要试图用一个 HPA 同时管两者。

### 冷启动在这里同样重要

冷启动缓解（Phase 17 · 10）正是节点供给时间变得用户可感知的地方。Karpenter 的 45-60 秒预热，加上 20GB 模型加载，再加上引擎初始化，意味着一个从零开始的请求需要 2-5 分钟。为 SLO 关键路径保留一个温池（`min_workers=1`），或者在应用层使用 Modal 式的检查点机制。

### 你应该记住的数字

- Karpenter 节点供给：约 45-60 秒，对比 Cluster Autoscaler 约 90-120 秒（GPU 节点）。
- KAI Scheduler 避免部分分配浪费——「8 缺 1」陷阱。
- 用 `DCGM_FI_DEV_GPU_UTIL` 作 HPA 信号：不可靠；改用队列深度或 KV 利用率。
- Karpenter `WhenEmptyOrUnderutilized`：会终止正在运行的 GPU 作业。推理场景用 `WhenEmpty + consolidateAfter: 1h`。

```figure
autoscaling
```

## 生产实践

`code/main.py` 在一个突发型 GPU 工作负载上模拟一个三层自动扩缩器。对比朴素 HPA（占空比）、队列深度 HPA 和 KAI 成组调度扩缩三种方案，报告未满足的请求数、GPU 空闲分钟数和一个综合得分。

## 交付产物

本课产出 `outputs/skill-gpu-autoscaler-plan.md`。给定集群拓扑、工作负载形态和 SLO，它设计一份三层自动扩缩容方案。

## 练习

1. 运行 `code/main.py`。在突发型工作负载下，朴素占空比 HPA 比队列深度 HPA 多丢弃了多少请求？差异从何而来？
2. 为一个在 H100 SXM5 上以 FP8 提供 Llama 3.3 70B 服务的集群设计一个 Karpenter NodePool。指定 `capacity-type`、`disruption.consolidationPolicy`、`consolidateAfter`，以及一个把非 GPU 工作负载挡在这些节点之外的污点（taint）。
3. 你的团队报告部署卡在 Pending 状态，原因是「GPU 明明可用但 Pod 调度不上去」。诊断一下——问题出在 Karpenter、kube-scheduler 还是 KAI Scheduler？哪些指标能确认？
4. 为分离式 prefill Pod 选一个自动扩缩信号，再为 decode Pod 选一个不同的信号。分别说明理由。
5. 计算 `WhenEmptyOrUnderutilized` 整合陷阱在一个 7x24 生产服务上的成本：该服务平均每天发生 60 次 P99 TTFT > 10 秒的请求丢弃事件。

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|----------------|------------------------|
| Karpenter | 「节点供给器」 | Kubernetes 节点自动扩缩器；分钟级以内供给 |
| Cluster Autoscaler | 「老一代扩缩器」 | Kubernetes 节点自动扩缩器的前辈；更慢，基于节点组 |
| KAI Scheduler | 「GPU 调度器」 | 提供成组调度 + 拓扑感知 + 队列的第二调度器 |
| Gang scheduling | 「全有或全无」 | 原子地调度 N 个 Pod，否则全部推迟 |
| Topology awareness | 「机架感知」 | 基于 NVLink/IB/机架位置放置 Pod |
| `DCGM_FI_DEV_GPU_UTIL` | 「GPU 利用率」 | 占空比指标；不是 LLM 的扩缩信号 |
| Queue depth | 「等待中的请求」 | prefill 受限场景下正确的 HPA 信号 |
| KV cache utilization | 「显存压力」 | decode 受限场景下正确的 HPA 信号 |
| Consolidation | 「Karpenter 整合」 | 终止节点以换成更便宜的实例类型 |
| `WhenEmpty + 1h` | 「安全整合」 | 不会驱逐正在运行的 GPU 作业的策略 |

## 延伸阅读

- [KAI Scheduler GitHub](https://github.com/kai-scheduler/KAI-Scheduler) — 设计文档与配置示例。
- [Karpenter Disruption Controls](https://karpenter.sh/docs/concepts/disruption/) — 整合策略语义与 GPU 安全默认值。
- [NVIDIA — Disaggregated LLM Inference on Kubernetes](https://developer.nvidia.com/blog/deploying-disaggregated-llm-inference-workloads-on-kubernetes/) — Dynamo Planner 扩缩信号。
- [Ray docs — KAI Scheduler for RayClusters](https://docs.ray.io/en/latest/cluster/kubernetes/k8s-ecosystem/kai-scheduler.html) — Ray 集成模式。
- [AWS EKS Compute and Autoscaling Best Practices](https://docs.aws.amazon.com/eks/latest/best-practices/aiml-compute.html) — 托管 Kubernetes 专属指南。
- [llm-d GitHub](https://github.com/llm-d/llm-d) — Workload Variant Autoscaler 设计。
