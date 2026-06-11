# Agno 与 Mastra：生产级运行时

> Agno（Python）和 Mastra（TypeScript）是 2026 年生产级运行时的代表组合。Agno 主打微秒级的智能体实例化和无状态的 FastAPI 后端。Mastra 则基于 Vercel AI SDK 底座，提供智能体、工具、工作流、统一模型路由和组合式存储。

**Type:** Learn
**Languages:** Python, TypeScript
**Prerequisites:** Phase 14 · 01 (Agent Loop), Phase 14 · 13 (LangGraph)
**Time:** ~45 minutes

## 学习目标

- 说出 Agno 的性能指标，以及这些指标在什么场景下才真正重要。
- 列举 Mastra 的三大原语——Agents、Tools、Workflows——以及它支持的服务器适配器。
- 解释为什么无状态、会话级（session-scoped）的 FastAPI 后端是 Agno 推荐的生产部署路径。
- 根据给定技术栈（Python 优先 vs TypeScript 优先）在 Agno 和 Mastra 之间做出选择。

## 问题背景

LangGraph、AutoGen、CrewAI 都是重框架。如果团队想要的只是"一个跑得快、贴合我现有运行时的智能体循环"，他们会选择 Agno（Python）或 Mastra（TypeScript）。两者都用一部分框架自带的原语换取了原始速度，以及与周边技术栈更紧密的契合。

## 核心概念

### Agno

- Python 运行时，前身是 Phi-data。
- "没有图、没有链、没有绕来绕去的模式——只有纯粹的 Python。"
- 官方文档给出的性能指标：智能体实例化约 2μs，每个智能体约占 3.75 KiB 内存，支持约 23 个模型提供商。
- 生产路径：无状态、会话级的 FastAPI 后端。每个请求都启动一个全新的智能体；会话状态存放在数据库中。
- 原生支持多模态（文本、图像、音频、视频、文件）和智能体式 RAG（agentic RAG）。

当你每秒要处理成千上万个短生命周期的智能体（聊天请求汇聚、评估流水线）时，这些速度指标才有意义。如果一个智能体一跑就是 10 分钟，它们就没那么重要了。

### Mastra

- TypeScript 实现，构建在 Vercel AI SDK 之上。
- 三大原语：**Agents**（智能体）、**Tools**（工具，用 Zod 定义类型）、**Workflows**（工作流）。
- 统一模型路由（Unified Model Router）——覆盖 94 个提供商的 3,300 多个模型（2026 年 3 月数据）。
- 组合式存储：记忆、工作流、可观测性数据可以分别写入不同后端；大规模可观测性场景推荐使用 ClickHouse。
- 采用 Apache 2.0 协议，但 `ee/` 目录适用源码可见（source-available）的企业许可证。
- 提供 Express、Hono、Fastify、Koa 的服务器适配器；对 Next.js 和 Astro 有一等公民级集成。
- 自带用于调试的 Mastra Studio（localhost:4111）。
- 1.0 发布时（2026 年 1 月）已有 22k+ GitHub star、每周 300k+ npm 下载量。

### 定位

两者都没有想成为 LangGraph。它们的竞争点在于：

- **语言契合度。** Agno 面向 Python 优先的团队；Mastra 面向 TypeScript 优先的团队。
- **运行时人体工学。** Agno = 接近零开销；Mastra = 与 Vercel 生态深度集成。
- **可观测性。** 两者都能接入 Langfuse/Phoenix/Opik（第 24 课），但 Mastra Studio 是官方第一方工具。

### 如何选择

- **Agno**——Python 后端、大量短生命周期智能体、对性能要求高、团队本来就用 FastAPI。
- **Mastra**——TypeScript 后端、部署在 Next.js / Vercel 上、需要统一的多提供商模型路由、需要 Zod 类型化的工具。
- **LangGraph**（第 13 课）——当持久化状态和显式的图结构推理比原始速度更重要时。
- **OpenAI / Claude Agent SDK**——当你想要提供商打包好的产品化形态时（第 16–17 课）。

### 这种模式容易翻车的地方

- **为性能而性能。** 工作负载明明是每个请求只有一次慢速智能体调用，却因为"2μs"听起来很厉害而选了 Agno。框架开销根本不是瓶颈。
- **生态锁定。** Mastra 偏向 Vercel 的集成方式在 Vercel 上是加分项，在其他地方就是减分项。
- **企业许可证的误解。** Mastra 的 `ee/` 目录是源码可见许可证，不是 Apache 2.0。如果你打算 fork，先把许可证读清楚。

## 从零实现

这节课以对比为主——没有哪一份代码能同时公允地展示两个框架。请看 `code/main.py` 中的并排示例：同一个最小化的"运行智能体、流式输出、持久化会话"流程实现了两遍（一遍按 Agno 的风格，一遍按 Mastra 的风格）。

运行方式：

```
python3 code/main.py
```

你会看到两条结构不同但功能等价的执行轨迹。

## 生产实践

- **Agno**——需要速度且采用 FastAPI 形态的 Python 后端。
- **Mastra**——需要多提供商支持和工作流原语的 TypeScript 后端。
- 两者都自带第一方可观测性钩子，也都能接入 Langfuse。

## 交付产物

`outputs/skill-runtime-picker.md` 会根据技术栈、延迟预算和运维形态，在 Agno、Mastra、LangGraph 或提供商 SDK 之间做出选择。

## 练习

1. 阅读 Agno 的文档。把标准库版的 ReAct 循环（第 01 课）移植到 Agno。哪些东西消失了？哪些保留了下来？
2. 阅读 Mastra 的文档。把同一个循环移植到 Mastra。工具类型定义发生了什么变化（Zod vs 没有类型）？
3. 做基准测试：在你的技术栈上测量智能体实例化延迟。Agno 的 2μs 对你的工作负载重要吗？
4. 设计一次迁移：如果你一直在 Python 上跑 CrewAI，迁移到 Agno 会有哪些东西坏掉？
5. 阅读 Mastra 的 `ee/` 许可证条款。哪些限制会影响一个开源 fork？

## 关键术语

| 术语 | 大家口中的说法 | 实际含义 |
|------|----------------|------------------------|
| Agno | "快速的 Python 智能体" | 无状态、会话级的智能体运行时 |
| Mastra | "基于 Vercel AI SDK 的 TypeScript 智能体" | Agents + Tools + Workflows + 模型路由 |
| 统一模型路由（Unified Model Router） | "多提供商接入" | 一个客户端覆盖 94 个提供商的 3,300 多个模型 |
| 组合式存储 | "多后端" | 记忆/工作流/可观测性各自写入不同的存储 |
| Mastra Studio | "本地调试器" | 用于检视智能体内部状态的 localhost:4111 界面 |
| 源码可见（Source-available） | "不算开源" | 许可证允许阅读源码，但限制商业使用 |

## 延伸阅读

- [Agno Agent Framework docs](https://www.agno.com/agent-framework) —— 性能指标、FastAPI 集成
- [Mastra docs](https://mastra.ai/docs) —— 原语、服务器适配器、模型路由
- [LangGraph overview](https://docs.langchain.com/oss/python/langgraph/overview) —— 有状态图方案的替代选项
- [Comet Opik](https://www.comet.com/site/products/opik/) —— Mastra 集成中引用的可观测性对比
