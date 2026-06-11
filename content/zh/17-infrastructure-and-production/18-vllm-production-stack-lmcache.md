# vLLM Production Stack 与 LMCache KV 卸载

> vLLM 的 production-stack 是参考级的 Kubernetes 部署方案——把路由器、引擎和可观测性组件串联在一起。LMCache 是 KV 卸载（KV offloading）层，它把 KV 缓存从 GPU 显存中抽取出来，在不同查询和不同引擎之间复用（先存 CPU DRAM，再下沉到磁盘/Ceph）。vLLM 0.11.0 的 KV Offloading Connector（2026 年 1 月）让这一过程变为异步，并可通过 Connector API（v0.9.0+）插拔。卸载延迟对用户不可见。即使没有共享前缀，LMCache 也有价值——当 GPU 耗尽 KV 槽位时，被抢占的请求可以从 CPU 恢复，而不必重新计算预填充（prefill）。在 4 台 a3-highgpu-4g 上共 16x H100（80GB HBM）的公开基准测试显示：当 KV 缓存超出 HBM 时，原生 CPU 卸载和 LMCache 都能大幅提升吞吐量；当 KV 占用较低时，所有配置与基线持平，仅有少量额外开销。

**Type:** Learn
**Languages:** Python (stdlib, toy KV-spill simulator)
**Prerequisites:** Phase 17 · 04 (vLLM Serving Internals), Phase 17 · 06 (SGLang/RadixAttention)
**Time:** ~60 minutes

## 学习目标

- 画出 vLLM production-stack 的分层架构：路由器、引擎、KV 卸载、可观测性。
- 解释 KV Offloading Connector API（v0.9.0+），以及 0.11.0 的异步路径如何隐藏卸载延迟。
- 量化 LMCache CPU-DRAM 何时有收益（KV > HBM）、何时只增加开销（KV 小到足以放进 HBM）。
- 根据部署约束，在 vLLM 原生 CPU 卸载与 LMCache connector 之间做出选择。

## 问题背景

你的 vLLM 服务显示 GPU 的 HBM 占用 100%，并发量一上升就出现抢占（preemption）事件。请求被驱逐、重新排队，同一个 2K token 的提示词一分钟内被重新预填充四次。GPU 算力浪费在冗余的预填充上；有效吞吐（goodput）远低于原始吞吐量。

增加 GPU 的成本是线性的。增加 HBM 则根本做不到。但 CPU DRAM 很便宜——单个插槽就有 512 GB 以上，虽然延迟比 HBM 差几个数量级，但用来存放「暂时温热」的 KV 缓存完全够用。

LMCache 把 KV 缓存抽取到 CPU DRAM，让被抢占的请求快速恢复；跨引擎重复出现的前缀也能共享缓存，不需要每个引擎各自重新预填充。

## 核心概念

### vLLM production-stack

`github.com/vllm-project/production-stack` 是参考级的 Kubernetes 部署方案：

- **路由器（Router）**——缓存感知（Phase 17 · 11）。消费 KV 事件。
- **引擎（Engines）**——vLLM worker。每个 GPU 或每个 TP/PP 组一个。
- **KV 缓存卸载**——LMCache 部署或原生 connector。
- **可观测性**——Prometheus 抓取、Grafana 看板、OTel 链路追踪。
- **控制平面**——服务发现、配置、滚动更新。

以 Helm chart + operator 的形式发布。

### KV Offloading Connector API（v0.9.0+）

vLLM 0.9.0 引入了 Connector API，用于可插拔的 KV 缓存后端。引擎把 KV 块卸载给 connector；connector 负责存储（RAM、磁盘、对象存储、LMCache）。当请求需要某个块时，connector 再把它加载回来。

vLLM 0.11.0（2026 年 1 月）新增了异步卸载路径——卸载可以在后台进行，常见情况下引擎不会被它阻塞。端到端延迟和吞吐量仍取决于负载形态、KV 缓存命中率和系统压力；vLLM 自己的说明也指出，自定义内核（custom-kernel）卸载在低命中率下可能降低吞吐量，且异步调度与投机解码（speculative decoding）存在已知的交互问题。

### 原生 CPU 卸载 vs LMCache

**vLLM 原生 CPU 卸载**：引擎本地。把 KV 块存到主机 RAM。实现快，零网络跳数。但不跨引擎。

**LMCache connector**：集群级。把块存到共享的 LMCache 服务器（CPU DRAM + Ceph/S3 层级）。任意引擎都能访问这些块。有 16x H100 的公开基准测试。

单个引擎有 HBM 压力时选原生方案。多个引擎共享前缀时选 LMCache（共用系统提示词的 RAG、共享模板的多租户场景）。

### 基准测试表现

分布在 4 台 a3-highgpu-4g 上的 16x H100（80 GB HBM）测试：

- 低 KV 占用（短提示词、低并发）：所有配置与基线持平，LMCache 增加约 3-5% 开销。
- 中等占用：LMCache 开始在跨引擎前缀复用上产生收益。
- KV 超出 HBM：原生 CPU 卸载和 LMCache 都大幅提升吞吐量；LMCache 收益更大，因为有跨引擎共享。

### LMCache 起决定性作用的场景

- 多租户服务，系统提示词在租户间共享。
- RAG 场景，文档块在多次查询中重复出现。
- 同一基础模型上的多个微调变体（LoRA），基础模型的 KV 复用能省去冗余计算。
- 抢占频繁的负载：从 CPU 恢复比重新预填充便宜。

### 不该启用的场景

- HBM 压力很小——只付出开销，没有收益。
- 短上下文（<1K token）——传输时间 > 重新预填充。
- 单租户、单提示词负载——没有可捕获的复用。

### 与分离式服务的集成

Phase 17 · 17 的分离式服务（disaggregated serving）与 LMCache 叠加生效：从预填充池传到解码池的 KV 如果没被用上，会落到 LMCache；后续查询可以从 LMCache 拉取。Phase 17 · 11 的缓存感知路由器可以把请求路由到本地缓存或 LMCache 共享缓存匹配的引擎。

### 你应该记住的数字

- vLLM 0.9.0：Connector API 发布。
- vLLM 0.11.0（2026 年 1 月）：异步卸载路径；端到端延迟影响取决于负载、KV 命中率和系统压力（不是绝对保证）。
- 16x H100 基准测试：KV 占用超出 HBM 时 LMCache 有收益。
- HBM 压力小时：3-5% 开销，没有收益。

```figure
zero-sharding
```

## 生产实践

`code/main.py` 模拟一个抢占频繁的负载，对比启用和不启用 LMCache 的情况。报告避免的重新预填充次数、吞吐量增益，以及收支平衡点对应的 HBM 利用率。

## 交付产物

本课产出 `outputs/skill-vllm-stack-decider.md`。给定负载形态和 vLLM 部署情况，判定该用原生方案、LMCache，还是都不用。

## 练习

1. 运行 `code/main.py`。HBM 利用率达到多少时 LMCache 开始有回报？
2. 某租户的 200 次查询/小时共享一个 6K token 的系统提示词。计算该租户预期的 LMCache 节省量。
3. LMCache 服务器是单点故障。设计高可用（HA）策略（副本、回退到原生方案）。
4. LMCache 存储到机械磁盘上的 Ceph。对于 70B FP8 模型的 4K token KV（500 MB），读取时间和重新预填充相比如何？
5. 论证 vLLM 0.11.0 的异步路径是否「免费」——开销藏在哪里？

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|----------------|------------------------|
| Production-stack | 「参考部署方案」 | vLLM 的 Kubernetes Helm chart + operator |
| Connector API | 「KV 后端接口」 | vLLM 0.9.0+ 可插拔的 KV 存储接口 |
| 原生 CPU 卸载 | 「引擎本地溢出」 | 把 KV 存到同一引擎的主机 RAM |
| LMCache | 「集群 KV 缓存」 | 基于 CPU DRAM + 磁盘的跨引擎 KV 缓存服务器 |
| 0.11.0 异步 | 「非阻塞卸载」 | 卸载隐藏在引擎流之后 |
| 抢占（Preemption） | 「驱逐腾位置」 | HBM 满时的 KV 缓存腾挪 |
| 前缀复用 | 「相同的系统提示词」 | 多个查询共享开头部分；缓存命中 |
| Ceph 层 | 「磁盘层」 | 缓存层次结构中 DRAM 之下的持久存储 |

## 延伸阅读

- [vLLM Blog — KV Offloading Connector (Jan 2026)](https://blog.vllm.ai/2026/01/08/kv-offloading-connector.html)
- [vLLM Production Stack GitHub](https://github.com/vllm-project/production-stack) — Helm chart + operator。
- [LMCache for Enterprise-Scale LLM Inference (arXiv:2510.09665)](https://arxiv.org/html/2510.09665v2)
- [LMCache GitHub](https://github.com/LMCache/LMCache) — Connector 实现。
- [vLLM 0.11.0 release notes](https://github.com/vllm-project/vllm/releases) — 异步路径细节。
