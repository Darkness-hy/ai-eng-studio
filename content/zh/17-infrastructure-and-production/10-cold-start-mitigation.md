# Serverless LLM 的冷启动缓解

> 一个 20 GB 的模型镜像从冷态到可服务，7B 模型需要 5-10 分钟，70B 模型需要 20 分钟以上。在真正的 serverless 世界里，这不是预热——这是一次故障。缓解手段分布在五个层面：预置节点镜像（AWS 上的 Bottlerocket、双卷架构）、模型流式加载（NVIDIA Run:ai Model Streamer，已原生集成进 vLLM）、GPU 内存快照（Modal 的 checkpoint，重启速度最高提升 10 倍）、温池（`min_workers=1`）、分层加载（ServerlessLLM 的 NVMe→DRAM→HBM 流水线，延迟降低 10-200 倍），以及只迁移输入 token（KB 级）而非 KV 缓存（GB 级）的实时迁移。Modal 公布的冷启动下限为 2-4 秒；Baseten 默认 5-10 秒，配合预热可达亚秒级。本课教你测量、做预算，并叠加这五个层面的手段。

**Type:** Learn
**Languages:** Python (stdlib, toy cold-start path simulator)
**Prerequisites:** Phase 17 · 02 (Inference Platform Economics), Phase 17 · 03 (GPU Autoscaling)
**Time:** ~60 minutes

## 学习目标

- 列举冷启动缓解的五个层面，并为每个层面说出一个工具或模式。
- 计算 70B 模型的总冷启动时间，即（节点供给）+（权重下载）+（权重加载进 HBM）+（引擎初始化）之和。
- 解释实时迁移为什么传输输入 token（KB 级）而非 KV 缓存（GB 级），以及代价是什么（重新计算）。
- 说出温池的权衡（为空闲 GPU 付费，还是接受冷启动长尾），以及 `min_workers > 0` 成为必选项的 SLA 阈值。

## 问题背景

你的 serverless LLM 端点在夜间缩容到零。早上 8 点流量陡增。第一个请求要等待以下过程：

1. Karpenter 供给一个 GPU 节点：45-60 秒。
2. 容器拉取一个含权重的 30 GB 镜像：120-300 秒。
3. 引擎把权重加载进 HBM：45-120 秒，取决于模型大小和存储速度。
4. vLLM 或 TRT-LLM 初始化 CUDA 图、KV 缓存池、分词器：10-30 秒。

合计：220-510 秒（约 3-8 分钟）才能返回第一个 token。而你的 SLA 是 2 秒。你上线了温池（`min_workers=1`），问题似乎消失了——但现在你要为一块空闲 GPU 全天候付费。如果你的服务有 5 个产品，每个产品各保留一个温副本，那就是 5 × 24 × 30 = 3,600 GPU 小时/月，不管有没有一个用户来调用。

冷启动缓解，就是在保住 serverless 经济性的同时，让延迟逼近常驻服务的水平。

## 核心概念

### 第 1 层——预置节点镜像（Bottlerocket）

在 AWS 上，Bottlerocket 的双卷架构把操作系统与数据分离。对预先拉好容器镜像的数据卷做快照，并在 `EC2NodeClass` 中引用该快照 ID。新节点启动时权重已经在本地 NVMe 上——步骤 2 以及步骤 3 的一部分直接消失。它与 Karpenter 原生兼容。典型收益：大模型每次冷启动节省 2-4 分钟。

GCP 上的等价做法：预烘焙容器层的自定义 VM 镜像。Azure 上：采用相同模式的托管磁盘快照。

### 第 2 层——模型流式加载（Run:ai Model Streamer）

不必在回答第一个请求前加载完整文件，而是逐层把权重流式传入 GPU 内存，只要第一个 Transformer 块就位就开始处理。NVIDIA Run:ai Model Streamer 已在 vLLM 2026 中原生集成。支持 S3、GCS 和本地 NVMe。通过让 I/O 与计算初始化重叠，大模型的权重加载时间大约减半。

### 第 3 层——GPU 内存快照（Modal）

Modal 在首次加载后对 GPU 状态（权重、CUDA 图、KV 缓存区域）做 checkpoint。后续重启直接反序列化进 HBM——比重新初始化快 10 倍。这是最接近"2 秒内启动一块温 GPU"的方案。权衡：快照与 GPU 拓扑绑定，如果 Karpenter 把你迁移到不同的 SKU，就要重新做 checkpoint。

### 第 4 层——温池（min_workers=1）

最简单的缓解：始终保持一个副本就绪。代价是一块 GPU 全天候的小时费率。这笔账对小模型很残酷（花 $0.85-$1.50/小时只为避免 30 秒冷启动），对大模型则划算（花 $4/小时避免 5 分钟冷启动）。温池成为必选项的 SLA 阈值：通常是 70B 以上模型要求 TTFT P99 < 60 秒。

### 第 5 层——分层加载（ServerlessLLM）

ServerlessLLM 把存储视为一个层级体系：NVMe（快且容量大）、DRAM（居中且分层）、HBM（极小但即时可用）。权重预加载到 DRAM；按需加载进 HBM。论文报告冷加载延迟相比朴素的磁盘到 HBM 路径降低 10-200 倍。生产环境采用尚处早期，但已有与 vLLM 的集成。

### 第 6 层——实时迁移（加餐模式）

当节点变得不可用（spot 实例驱逐、节点排空），传统模式是冷启动另一个副本并清空请求队列。实时迁移把输入 token（千字节级）转移到已加载模型的目标节点，并在目标上重新计算 KV 缓存。重新计算比在网络上传输 GB 级 KV 缓存更便宜。适用于分离式（disaggregated）部署。

### 温池的算术题

对一个 P99 TTFT SLA 为 2 秒的服务，问题不是"要不要温池"，而是"要多少个温副本，以及哪些路径配置温副本"。

- 高价值交互路径（实时聊天、语音 Agent）：`min_workers=1-2`。
- 后台批处理路径（夜间分类任务）：可以接受缩容到零，5-10 分钟的冷启动也可容忍。
- 高级套餐：按租户配置 `min_workers`，提供专属容量。

### 先测量，再优化

70B 模型在全新节点上的冷启动解剖（示意性数据）：

| 阶段 | 时间 | 缓解手段 |
|-------|------|-----------|
| 节点供给 | 50s | Bottlerocket + 预置镜像、温池 |
| 镜像拉取 | 180s | 预置数据卷（消除） |
| 权重进入 HBM | 75s | 模型流式加载（减半）；GPU 快照（消除） |
| 引擎初始化 | 20s | 持久化 CUDA 图缓存 |
| 首次前向计算 | 3s | 最小固有延迟 |
| **冷启动总计** | **328s** | |
| **叠加缓解后总计** | **~15s** | 降低 22 倍 |

### 你应该记住的数字

- Modal 冷启动：2-4 秒（使用 GPU 快照）。
- Baseten 默认冷启动：5-10 秒；配合预热可达亚秒级。
- 裸 70B 冷启动：3-8 分钟。
- Run:ai Model Streamer：权重加载速度约 2 倍提升。
- ServerlessLLM 分层加载：延迟降低 10-200 倍（论文数据）。

## 生产实践

`code/main.py` 模拟了一条冷启动路径，对比启用与不启用各项缓解手段的效果。报告冷启动总时间、温池成本，以及温池开始划算的盈亏平衡请求速率。

## 交付产物

本课产出 `outputs/skill-cold-start-planner.md`。给定 SLA、模型大小和流量形态，它会选出应该叠加哪些缓解手段。

## 练习

1. 运行 `code/main.py`。计算盈亏平衡请求速率：超过该速率后，保留一个温副本比按 SLO 计算的额外请求丢弃所付出的冷启动代价更便宜。
2. 你部署了一个 13B 模型，P99 TTFT SLA 为 3 秒。挑出能达成目标的最小缓解组合（层数最少）。
3. Bottlerocket 预置消除了镜像拉取，但权重仍需从快照加载到 HBM。假设快照支撑的 NVMe 读取速度为 7 GB/s，计算 70B 模型的实际耗时。
4. 你的 serverless 服务商提供 GPU 快照（Modal），而你的团队以"快照会泄露 PII"为由拒绝。请论证双方观点——现实风险是什么，缓解措施是什么（临时快照、加密、命名空间隔离）？
5. 设计一套分层温池策略：付费用户、试用用户和批处理工作负载各配多少个温副本？给出计算过程。

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|----------------|------------------------|
| 冷启动（Cold start） | "那个大停顿" | 在全新副本上从收到请求到产出第一个 token 的时间 |
| 温池（Warm pool） | "常驻最小值" | `min_workers >= 1`，保持至少一个副本就绪 |
| 预置镜像 | "烘焙好的 AMI" | 容器与权重预先驻留的节点镜像 |
| Bottlerocket | "AWS 节点操作系统" | AWS 容器优化操作系统，支持双卷快照 |
| 模型流式加载 | "流式加载" | 权重 I/O 与计算初始化重叠进行 |
| GPU 快照 | "checkpoint 到 HBM" | 序列化加载完成后的 GPU 状态；重启时反序列化 |
| 分层加载 | "NVMe + DRAM + HBM" | 存储分层体系；按需加载 |
| 实时迁移（Live migration） | "搬 token" | 传输输入（KB 级），在目标节点重算 KV |
| `min_workers` | "温副本" | serverless 的最小保活数量 |
| 缩容到零（Scale-to-zero） | "完全 serverless" | 空闲时零成本；接受完整的冷启动代价 |

## 延伸阅读

- [Modal — Cold start performance](https://modal.com/docs/guide/cold-start) — Modal 公布的基准测试与 checkpoint 架构。
- [AWS Bottlerocket](https://github.com/bottlerocket-os/bottlerocket) — 预置数据卷快照模式。
- [NVIDIA Run:ai Model Streamer](https://github.com/run-ai/runai-model-streamer) — 权重加载与计算初始化重叠。
- [Baseten — Cold-start mitigation](https://www.baseten.co/blog/cold-start-mitigation/) — 预热实战手册。
- [ServerlessLLM paper (USENIX OSDI'24)](https://www.usenix.org/conference/osdi24/presentation/fu) — 分层加载设计。
- [NVIDIA — Disaggregated LLM Inference on Kubernetes](https://developer.nvidia.com/blog/deploying-disaggregated-llm-inference-workloads-on-kubernetes/) — 分离式部署的实时迁移。
