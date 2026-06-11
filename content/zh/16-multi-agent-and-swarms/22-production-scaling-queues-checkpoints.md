# 生产级扩展——队列、检查点与持久性

> 要把多智能体系统扩展到数千个并发运行，需要**持久化执行（durable execution）**。LangGraph 的运行时在每个超步（super-step）之后以 `thread_id` 为键写入一个检查点（默认存储在 Postgres）；worker 崩溃后释放租约，由另一个 worker 接续执行。智能体可以无限期休眠，等待人工输入。**MegaAgent**（arXiv:2408.09955）为每个智能体运行一个生产者-消费者队列，包含三种状态（Idle / Processing / Response），并采用两层协调机制（组内聊天 + 组间管理员聊天）。在 LLM 流式输出场景下，**纤程/异步（fiber/async）**优于每任务一线程的模式：线程在等待 token 时 99% 的时间处于空闲，而纤程会在 I/O 时协作式让出。反方观点：Ashpreet Bedi 的《Scaling Agentic Software》主张在负载证明有必要之前，只用 **FastAPI + Postgres，别的什么都不加**——简单架构能走得比预期更远。本课将构建一个持久化检查点日志、一个带状态转换的按智能体工作队列、一个异步与线程对比的演示，并落实务实的"从简单开始"原则。

**Type:** Learn + Build
**Languages:** Python (stdlib, `asyncio`, `sqlite3`)
**Prerequisites:** Phase 16 · 09 (Parallel Swarm Networks), Phase 16 · 13 (Shared Memory)
**Time:** ~75 minutes

## 问题背景

一个多智能体系统原型在一台笔记本上运行良好：三个智能体跑在一个内存事件循环中。当你迁移到生产环境时：

- 智能体有时要运行数小时（长时间研究、等待人工介入）。
- worker 进程会崩溃。重启意味着丢失状态。
- 峰值负载是平均值的 10 倍；你需要水平扩展。
- 用户按智能体运行次数付费；计费需要精确一次（exactly-once）语义。

内存事件循环不满足以上任何一条。你需要在底层加一个持久化执行层。2026 年的标准选项有：

1. 带检查点的工作流引擎（Temporal、LangGraph 运行时）。
2. 消息队列加状态存储（Postgres + SQS/RabbitMQ）。
3. Actor 模型框架（MegaAgent 的按智能体生产者-消费者模式）。
4. 手写 FastAPI + Postgres（Bedi 的主张）。

本课为每一种方案构建一个微缩版本。

## 核心概念

### 持久化执行这一模式

持久化执行引擎在每个"步骤"（LangGraph 的术语是超步）之后持久化完整的程序状态。崩溃时：

```
worker crashes mid-step
  -> lease timeout
  -> another worker picks up the thread_id
  -> resumes from last checkpoint
  -> no duplicate side effects
```

要让这个机制成立，需满足以下要求：

- **状态可序列化。** 所有智能体状态都必须可以持久化。持有活跃数据库连接的函数闭包无法存活。
- **可确定性恢复。** 给定相同的状态和相同的输入，智能体产生相同的动作（或者把 LLM 调用委托给外部的确定性预言机）。
- **副作用幂等。** 外部调用（工具调用、支付）必须是幂等的，或使用去重键。

LangGraph 在每个超步后写检查点；Temporal 在每个 activity 后写；Restate 使用事件溯源日志。三者实现的是同一个模式。

### LangGraph 的运行时

每个智能体有一个 `thread_id`；状态是一个带类型的字典；每个超步向检查点表写入一行。恢复时，运行时从最后一个检查点重放，而不是从头开始。智能体可以 `interrupt()` 等待人工输入；运行时持久化状态并释放 worker。当输入到达时，任意 worker 都可以接续执行。

这是 2026 年 4 月的参考性生产设计。

### MegaAgent 的按智能体队列

arXiv:2408.09955 描述了一个规模化实验：单个集群中运行数千个并发智能体。架构如下：

```
agent i:
  state ∈ {Idle, Processing, Response}
  in_queue   <- messages addressed to agent i
  out_queue  -> replies + side effects

coordinators:
  intra-group chat  (agents in the same group)
  inter-group admin chat  (high-level routing)
```

两层协调机制让组内对话可以高密度进行，而组间通信保持稀疏——这正是在数千个智能体规模下保持成本线性增长的模式。

### 异步 vs 每任务一线程

LLM 调用是 I/O 密集型的。等待下一个 token 的线程 99% 的时间处于空闲。每个线程约占 1MB 内存；1 万个并发调用，仅栈空间就要 10GB。

纤程（Python 的 `asyncio`、Go 的 goroutine、Rust 的 `tokio`）会在 I/O 时协作式让出。同样的 1 万个调用可以轻松装进一个进程。在 LLM 智能体的规模上，异步不是一种优化——它就是架构本身。

例外：CPU 密集型的后处理（嵌入计算、分词器技巧）仍然适合线程或进程。把 I/O 层和 CPU 层分开。

### Bedi 的反方观点

《Scaling Agentic Software》（Ashpreet Bedi，2026）认为大多数团队在测量负载之前就过度设计。务实的默认方案是：

- FastAPI + Postgres。
- 每次智能体运行对应一行记录；用乐观并发就地更新状态。
- 后台任务通过 `pg_notify` 或一个简单的 Celery worker 实现。
- 重试策略写在应用代码里。

对于可控任务、并发智能体运行数低于约 100 的负载，这往往就够了。等你实际测量到它撑不住时再升级。

原则是：当你遇到简单架构无法解决的具体问题时，再采用持久化执行框架。过早采用会把时间烧在不产生回报的繁文缛节上。

### 精确一次语义

对于付费的智能体运行，你需要"实际效果精确一次"（至少一次投递 + 幂等消费者）。具体的工程手段：

- **每次运行一个去重键。** 在每个副作用调用中都带上它。
- **发件箱模式（outbox pattern）。** 副作用先写入一张表，再由独立进程执行。两个步骤都要幂等。
- **补偿事务。** 当副作用成功但其跟踪记录写入失败时，调度一次补偿操作。

这些是数据库工程模式，与 LLM 无关。LLM 带来的额外代价仅仅是 LLM 调用很慢；其余一切都是标准的分布式系统问题。

### 彩虹部署

Anthropic 的多智能体研究系统使用"彩虹部署（rainbow deployment）"：多个版本的智能体运行时并发运行，这样长时间运行的智能体不必在每次代码部署时被杀掉。新版本先在一部分流量上做金丝雀验证；旧版本等其上的智能体全部完成后再下线。

这是长时间运行的有状态系统的标准做法；2026 年的变化在于智能体可以存活数小时，部署周期必须与之适配。

### 标准生产检查清单

- 持久化状态（检查点、快照，或发件箱 + 可重放日志）。
- 幂等的副作用。
- LLM 调用使用异步 I/O 层。
- 至少一次投递 + 去重。
- 有状态工作负载使用彩虹/金丝雀部署。
- 可观测性：按智能体的链路追踪、超步审计、重试计数器。

## 从零实现

`code/main.py` 实现了：

- `CheckpointStore`——基于 SQLite 的检查点日志，以 thread-id 为键。每个超步追加一行。
- `run_with_checkpoint(agent, thread_id)`——模拟运行中途崩溃；第二个 worker 从最后一个检查点接续执行。
- `AgentQueue`——按智能体的 Idle / Processing / Response 状态机，带一个小型工作队列。
- `demo_async_vs_threads()`——分别用 asyncio 和线程运行 500 个并发的模拟"LLM 调用"；报告墙钟时间和峰值内存（近似值）。

运行：

```
python3 code/main.py
```

预期输出：模拟崩溃后检查点恢复成功；异步版本在 1 秒内处理完 500 个并发调用；线程版本耗时数秒，且每个并发单元的内存开销高出几个数量级。

## 生产实践

`outputs/skill-scaling-advisor.md` 就持久化执行方案的选型给出建议：FastAPI + Postgres、LangGraph 运行时、Temporal，还是自研。依据负载、状态保留需求和部署频率来校准。

## 交付产物

标准的生产加固措施：

- **从简单开始（Bedi 原则）。** 用 FastAPI + Postgres，直到实测发现它撑不住。
- **先做全面埋点，再做优化。** 按运行的延迟直方图、按步骤的耗时、重试次数、失败分类。
- **副作用走发件箱模式。** 尤其是支付和外部 API 调用。
- **彩虹部署。** 部署期间绝不杀掉进行中的智能体运行。
- **在遇到具体问题时再采用持久化执行引擎（Temporal / LangGraph / Restate）：** 长达数小时的人工介入等待、跨区域协调、复杂的重试/补偿策略。
- **I/O 层用异步。** 线程只用于 CPU 密集型的后处理。

## 练习

1. 运行 `code/main.py`。确认检查点恢复正常工作；测量异步与线程的并发性能差异。
2. 实现一张**发件箱（outbox）**表：每次工具调用先写入 outbox，再由独立的 goroutine/任务执行。把同一工具调用运行两次，验证幂等性。
3. 模拟一次**彩虹部署**：两个运行时版本并发运行；把新的 thread_id 各路由一半给两个版本；确认旧版本上进行中的线程不会被打断。
4. 阅读 LangGraph 的运行时文档（链接见下）。找出运行时的哪些特性在手写 FastAPI + Postgres 方案中复刻起来最费时间。这是采用它的理由，还是可以暂缓？
5. 阅读 MegaAgent（arXiv:2408.09955）第 3 节。其中明确描述了两层协调机制（组内聊天 + 组间管理员聊天）。画一个草图，说明如何用包含两类队列族的消息队列来实现它。

## 关键术语

| 术语 | 通俗说法 | 实际含义 |
|------|----------------|------------------------|
| 持久化执行（Durable execution） | "持久化程序状态" | 引擎在每个超步后写入状态；崩溃恢复是确定性的。 |
| 超步（Super-step） | "事务边界" | 两次检查点之间的工作单元。LangGraph 术语。 |
| thread_id | "智能体运行标识符" | 绑定检查点与恢复逻辑的键。 |
| 幂等性（Idempotency） | "可以安全重试" | 重复执行一个副作用，结果与执行一次相同。 |
| 发件箱模式（Outbox pattern） | "解耦副作用" | 先把意图写入一张表；由独立执行器执行并标记完成。 |
| 至少一次投递（At-least-once delivery） | "可能有重复" | 消息队列语义；去重键让消费者达到实际效果一次。 |
| 彩虹部署（Rainbow deploy） | "版本重叠" | 长时间运行的工作负载期间，多个运行时版本并发存在。 |
| 异步纤程（Async fiber） | "协作式让出" | 用户态并发；对于 I/O 密集型负载，比线程便宜得多。 |
| 检查点（Checkpoint） | "状态快照" | 超步边界处序列化的状态；恢复执行的关键。 |

## 延伸阅读

- [LangChain — The runtime behind production deep agents](https://www.langchain.com/conceptual-guides/runtime-behind-production-deep-agents) —— LangGraph 运行时设计
- [MegaAgent](https://arxiv.org/abs/2408.09955) —— 按智能体的生产者-消费者队列；数千并发智能体规模下的两层协调机制
- [Matrix](https://arxiv.org/abs/2511.21686) —— 以消息队列为协调基底的去中心化框架
- [Temporal docs](https://docs.temporal.io/) —— 持久化执行领域的参考性工作流引擎
- [Anthropic — Multi-agent research system](https://www.anthropic.com/engineering/multi-agent-research-system) —— 包括彩虹部署在内的生产经验
