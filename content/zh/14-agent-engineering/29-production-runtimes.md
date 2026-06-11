# 生产运行时：队列、事件、定时任务

> 生产环境中的智能体运行在六种运行时形态上：请求-响应、流式、持久化执行、基于队列的后台任务、事件驱动和定时调度。先选形态，再选框架。在每一种形态下，可观测性都是承重结构。

**Type:** Learn
**Languages:** Python (stdlib)
**Prerequisites:** Phase 14 · 13 (LangGraph), Phase 14 · 22 (Voice)
**Time:** ~60 minutes

## 学习目标

- 说出六种生产运行时形态，并将每种形态对应到一种框架/产品模式。
- 解释为什么持久化执行（LangGraph）对长程任务至关重要。
- 描述事件驱动运行时，以及 Claude Managed Agents 在什么场景下适用。
- 解释为什么对于多步智能体来说，「可观测性是承重结构」这一论断成立。

## 问题背景

生产环境中的智能体会以 Jupyter notebook 暴露不出来的方式失败：第 37 步遇到网络超时、用户在语音通话中途挂断、机器重启导致 cron 任务挂掉、后台 worker 内存耗尽。运行时形态决定了哪些故障是可以挺过去的。

## 核心概念

### 请求-响应

- 同步 HTTP。用户等待任务完成。
- 只适用于短任务（<30 秒）。
- 技术栈：Agno（Python + FastAPI）、Mastra（TypeScript + Express/Hono/Fastify/Koa）。
- 可观测性：标准 HTTP 访问日志 + OTel span。

### 流式

- 通过 SSE 或 WebSocket 渐进式输出。
- LiveKit 将其扩展到 WebRTC，用于语音/视频（第 22 课）。
- 技术栈：任何支持流式输出的框架 + 能处理 SSE/WS 的前端。
- 可观测性：逐 chunk 计时、首 token 延迟、尾部延迟。

### 持久化执行

- 每一步之后都对状态做检查点（checkpoint）；失败后自动恢复。
- AutoGen v0.4 的 actor 模型将故障隔离到单个智能体（第 14 课）。
- 这是 LangGraph 的核心差异化能力（第 13 课）。
- 当步数未知且恢复成本很高时，这是必备能力。

### 基于队列 / 后台任务

- 任务进入队列，worker 取出执行，结果通过 webhook 或 pub/sub 回传。
- 对长程智能体来说必不可少（按 Anthropic 的 computer use 发布公告，每个任务通常有几十到几百步）。
- 技术栈：Celery（Python）、BullMQ（Node）、SQS + Lambda（AWS）、自研。
- 可观测性：队列深度、单任务延迟分布、DLQ 大小。

### 事件驱动

- 智能体订阅触发器：新邮件、PR 被打开、cron 触发。
- Claude Managed Agents 开箱即用地覆盖了这一形态（第 17 课）。
- CrewAI Flows（第 15 课）用于组织事件驱动的确定性工作流。
- 可观测性：触发来源、事件到启动的延迟、智能体延迟。

### 定时调度

- cron 形态的智能体，周期性运行。
- 与持久化执行组合，这样失败的夜间任务会在下一次触发时恢复。
- 技术栈：Kubernetes CronJob + 持久化框架；托管方案（Render cron、Vercel cron）。

### 2026 年的部署模式

- **CrewAI Flows** 用于事件驱动的生产环境。
- **Agno** 无状态 FastAPI，用于 Python 微服务。
- **Mastra** 服务器适配器（Express、Hono、Fastify、Koa），用于嵌入现有服务。
- **Pipecat Cloud / LiveKit Cloud** 用于托管语音（第 22 课）。
- **Claude Managed Agents** 用于托管的长时间运行异步任务。

### 可观测性是承重结构

没有 OpenTelemetry GenAI span（第 23 课）加上 Langfuse/Phoenix/Opik 后端（第 24 课），你就无法调试一个在第 40 步失败的多步智能体。对生产环境来说这不是可选项。它决定了你是「快速调试」，还是「加更多日志后从头重放」。

### 生产运行时在哪里失败

- **选错形态。** 给一个 5 分钟的任务选了请求-响应。用户挂断；worker 堆积；重试雪上加霜。
- **没有 DLQ。** 队列 worker 没有死信队列。失败的任务直接消失。
- **不透明的后台工作。** 后台智能体运行时没有导出 trace。故障在用户上报之前完全不可见。
- **跳过持久化状态。** 任何超过 30 秒、且你承受不起重启代价的运行，都需要持久化执行。

## 从零实现

`code/main.py` 是一个仅用标准库实现的多形态演示：

- 请求-响应端点（普通函数）。
- 流式处理器（生成器）。
- 带 DLQ 的队列 worker。
- 事件触发器注册表。
- cron 形态的调度器。

运行：

```bash
python3 code/main.py
```

输出：五条 trace，展示每种形态在同一任务上的行为。相同的智能体逻辑，不同的外壳。持久化执行（第六种形态）有意放在第 13 课，配合 LangGraph 检查点机制讲解。

## 生产实践

- **请求-响应** 用于聊天式 UX。
- **流式** 用于渐进式响应。
- **持久化** 用于长程任务。
- **队列** 用于批处理 / 异步 / 长时间运行。
- **事件** 用于智能体的响应式触发。
- **Cron** 用于日常维护（记忆整合、评测、成本报告）。

## 交付产物

`outputs/skill-runtime-shape.md` 为任务挑选运行时形态，并接入相应的可观测性要求。

## 练习

1. 把你在第 01 课实现的 ReAct 循环移植到你技术栈中的全部六种形态。哪种形态适合哪种产品形态（product surface）？
2. 给基于队列的演示加上 DLQ。模拟 10% 的任务失败率；把 DLQ 大小暴露出来。
3. 写一个由 cron 触发的评测智能体，每晚针对当天 top 20 的 trace 运行。
4. 实现带背压（backpressure）的流式输出：如果客户端很慢，就暂停智能体。这与轮次预算（turn budget）如何相互影响？
5. 阅读 Claude Managed Agents 文档。什么情况下你会把自托管的长程智能体迁移到托管服务？

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|----------------|------------------------|
| 请求-响应 | 「同步」 | 用户等待；仅适用于短任务 |
| 流式 | 「SSE / WS」 | 渐进式输出；更好的 UX；延迟可按 chunk 观测 |
| 持久化执行 | 「从失败中恢复」 | 检查点化的状态；从最后一步重启 |
| 基于队列 | 「后台任务」 | 生产者 / worker 池 / DLQ |
| 事件驱动 | 「基于触发器」 | 智能体响应外部事件 |
| DLQ | 「死信队列」 | 失败任务的停放场 |
| Claude Managed Agents | 「托管 harness」 | Anthropic 托管的长时间运行异步服务，带缓存与压缩 |

## 延伸阅读

- [LangGraph overview](https://docs.langchain.com/oss/python/langgraph/overview) — 持久化执行细节
- [Claude Managed Agents overview](https://platform.claude.com/docs/en/managed-agents/overview) — 托管的长时间运行异步服务
- [Anthropic, Introducing computer use](https://www.anthropic.com/news/3-5-models-and-computer-use) — 「每个任务几十到几百步」
- [AutoGen v0.4 (Microsoft Research)](https://www.microsoft.com/en-us/research/articles/autogen-v0-4-reimagining-the-foundation-of-agentic-ai-for-scale-extensibility-and-robustness/) — actor 模型的故障隔离
