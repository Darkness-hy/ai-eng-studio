# 长时间运行的后台智能体：持久化执行

> 生产环境中的长程智能体并不是跑在 `while True` 里。每一次 LLM 调用都会变成一个带有检查点、重试和重放能力的活动（activity）。Temporal 与 OpenAI Agents SDK 的集成于 2026 年 3 月正式发布（GA）。Claude Code Routines（Anthropic）能够按计划调度运行 Claude Code，无需常驻的本地进程。会话在等待人工输入时暂停，能挺过部署更新，并从以 `thread_id` 为键的最新检查点恢复。这些新颖的使用体验背后，是一个古老的模式——工作流编排（workflow orchestration）——加上一个新的输入：LLM 调用作为非确定性活动，必须在恢复时被确定性地重放。

**Type:** Learn
**Languages:** Python (stdlib, minimal durable-execution state machine)
**Prerequisites:** Phase 15 · 10 (Permission modes), Phase 15 · 01 (Long-horizon agents)
**Time:** ~60 minutes

## 问题背景

设想一个运行四小时的智能体。它调用了三个工具，向用户发起了两次询问，并执行了四十次 LLM 调用。在中途，承载它的主机重启了。会发生什么？

- 在朴素的 `while True` 循环里：一切全部丢失。运行从头重来。那三次工具调用（带有真实副作用）会再次执行。用户会被再次询问那些早已批准过的事项。四十次 LLM 调用要重新付费。
- 在持久化执行下：运行从最近的检查点恢复。已完成的活动不会被重新执行；它们的结果从持久化日志中重放。用户不必重新批准已经批准过的事项。已经发出的 LLM 调用不会被重复计费。

这与工作流引擎已经交付了十年的模式如出一辙（Temporal、Cadence、Uber 的 Cherami）。新颖之处在于，LLM 调用如今也成为一种活动——非确定性、昂贵、带副作用——而它恰好与这一模式严丝合缝。

本课贯穿始终的主题是：长程可靠性会衰减（METR 观察到「35 分钟衰减」——成功率随任务时长大致呈二次方下降）。持久化执行让运行时长可以超出可靠性曲线所能支撑的范围，这是一种新的可能性：设计正确时可以安全地失败，设计错误时则会不安全地失败。

## 核心概念

### 活动、工作流与重放

- **工作流（Workflow）**：确定性的编排代码。定义活动的顺序、分支和等待。必须保持确定性，才能从事件日志重放而不产生意外分歧。
- **活动（Activity）**：非确定性的、可能失败的工作单元。LLM 调用、工具调用、文件写入、HTTP 请求。每个活动都会连同其输入（以及完成后的输出）被记录。
- **事件日志（Event log）**：持久化的底层存储。每个活动的开始、完成、失败、重试，以及每个工作流决策都会被记录。
- **重放（Replay）**：恢复时，工作流代码从头重新运行；已经完成的活动直接返回日志中记录的结果而不重新执行。只有尚未完成的活动才会真正运行。

这与 React 基于虚拟 DOM 的重新渲染，或 Git 从提交记录重建工作树的形态相同。编排器的确定性正是让持久化变得廉价的关键。

### 为什么 LLM 调用契合这一模式

LLM 调用具有以下特性：
- 非确定性（temperature > 0；即使 temperature 为 0，输出也会随模型版本漂移）。
- 昂贵（花钱且耗时）。
- 可能失败（速率限制、超时）。
- 带副作用（如果它们调用了工具）。

这正是活动的典型画像。把每一次 LLM 调用包装成活动，你就获得了带指数退避的重试、跨重启的检查点，以及可重放的调试追踪。

### 以 `thread_id` 为键的检查点

LangGraph、Microsoft Agent Framework、Cloudflare Durable Objects 和 Claude Code Routines 不约而同地收敛到同一种 API 形态：一个 `thread_id`（或等价物）标识会话；每次状态转移都持久化到某个后端（默认 PostgreSQL，开发用 SQLite，缓存用 Redis）；恢复时读取最新检查点。

后端的选择很重要：

- **PostgreSQL**：持久、可查询、能挺过部署更新。LangGraph 的默认选项。
- **SQLite**：仅限本地开发；跨主机会丢数据。
- **Redis**：速度快，但除非配置了 AOF/快照，否则数据易失。
- **Cloudflare Durable Objects**：透明的分布式存储；以唯一键划定作用域；可存活数小时到数周。

### 人工输入作为一等状态

「先提议后提交」（propose-then-commit，第 15 课）要求一个持久化的「等待人工」状态。工作流暂停，外部队列保存待处理请求，一次批准就能从那个精确的位置恢复。没有持久化，这只能尽力而为；有了它，凌晨送达的一次批准会让工作流在早上接着干下去。

### 35 分钟衰减

METR 观察到，所有被测的智能体类别在持续运行约 35 分钟后都出现可靠性衰减。任务时长翻倍，失败率大约翻四倍。持久化执行并不能修复这一点；它只是让你能运行得比可靠性曲线所能支撑的更久。安全的做法是把持久化与「重新进入时必须重新走 HITL」的检查点结合起来，再配合预算熔断开关（第 13 课），无论墙钟时间多长都对总计算量设限。

### 什么时候持久化执行是错误答案

- 几分钟以内、不涉及人工输入的运行。开销大于收益。
- 严格只读的信息检索。
- 正确性要求在单个上下文窗口内端到端完成的任务（某些推理任务；某些一次性生成任务）。

```figure
memory-consolidation
```

## 生产实践

`code/main.py` 用 Python 标准库实现了一个最小化的持久化执行引擎。它支持：

- `@activity` 装饰器，把输入和输出记录到一个 JSON 事件日志。
- 一个串联各活动的工作流函数。
- 一个 `run_or_replay(workflow, event_log)` 函数，重放已完成的活动而不重新执行它们。

驱动程序模拟了一个三活动的工作流，在中途崩溃，然后展示 (a) 朴素重试把所有活动重新执行一遍，与 (b) 重放只运行缺失的那个活动之间的差异。

## 交付产物

`outputs/skill-durable-execution-review.md` 对一个拟议的长时运行智能体部署方案做评审，检查其持久化执行的形态是否正确：活动划分、确定性、检查点后端、人工输入状态，以及恢复时的 HITL 策略。

## 练习

1. 运行 `code/main.py`。观察朴素重试与重放在活动执行次数上的差异。改变崩溃点，验证重放次数随之变化。

2. 把这个玩具引擎改成显式使用 `thread_id`。模拟两个并发会话共用该引擎，并确认它们的事件日志互不冲突。

3. 取玩具引擎中的一个活动。引入一处非确定性（在工作流决策中使用墙钟时间戳）。演示重放时产生的分歧。解释真实引擎如何处理这种情况（副作用注册、`Workflow.now()` 这类 API）。

4. 阅读 LangChain 的 "Runtime behind production deep agents" 一文。列出该运行时持久化的每一种状态，并指出每种状态对应覆盖的失败模式。

5. 为一个 6 小时的自主编码任务设计检查点策略。在哪里设检查点？崩溃后的恢复流程是什么样的？哪些环节需要重新走 HITL？

## 关键术语

| 术语 | 大家口中的说法 | 实际含义 |
|---|---|---|
| 工作流（Workflow） | 「智能体的脚本」 | 确定性的编排代码；可从事件日志重放 |
| 活动（Activity） | 「一个步骤」 | 非确定性单元（LLM 调用、工具调用）；执行前后均记录日志 |
| 事件日志（Event log） | 「底层存储」 | 每次状态转移的持久化记录 |
| 重放（Replay） | 「恢复」 | 重新运行工作流；已完成活动返回日志结果而不重新执行 |
| 检查点（Checkpoint） | 「存档点」 | 以 thread_id 为键的持久化状态；恢复时取最新者 |
| thread_id | 「会话键」 | 划定持久化状态作用域的标识符 |
| 35 分钟衰减 | 「可靠性衰减」 | METR：成功率随任务时长大致呈二次方下降 |
| 非确定性（Non-determinism） | 「重放时漂移」 | 墙钟时间、随机数、LLM 输出；必须注册为副作用 |

## 延伸阅读

- [Anthropic — Claude Code Agent SDK: agent loop](https://code.claude.com/docs/en/agent-sdk/agent-loop) — 预算、轮次与恢复语义。
- [Microsoft — Agent Framework: human-in-the-loop and checkpointing](https://learn.microsoft.com/en-us/agent-framework/workflows/human-in-the-loop) — RequestInfoEvent 的形态。
- [LangChain — The Runtime Behind Production Deep Agents](https://www.langchain.com/conceptual-guides/runtime-behind-production-deep-agents) — 具体的运行时需求。
- [OpenAI Agents SDK + Temporal integration (Trigger.dev announcement)](https://trigger.dev) — LLM 调用的活动形态。
- [Anthropic — Measuring agent autonomy in practice](https://www.anthropic.com/research/measuring-agent-autonomy) — 35 分钟衰减的参考来源。
