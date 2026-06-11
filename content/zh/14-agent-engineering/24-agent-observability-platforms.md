# Agent 可观测性：Langfuse、Phoenix、Opik

> 2026 年，三大开源 Agent 可观测性平台占据主导地位。Langfuse（MIT）——每月 600 万+ 安装量，提供链路追踪 + 提示词管理 + 评估 + 会话回放。Arize Phoenix（Elastic 2.0）——深度的 Agent 专项评估、RAG 相关性、OpenInference 自动插桩。Comet Opik（Apache 2.0）——自动化提示词优化、护栏、LLM 评审幻觉检测。

**Type:** Learn
**Languages:** Python (stdlib)
**Prerequisites:** Phase 14 · 23 (OTel GenAI)
**Time:** ~45 minutes

## 学习目标

- 说出三大开源 Agent 可观测性平台及其许可证。
- 区分各平台最擅长的方向：Langfuse（提示词管理 + 会话）、Phoenix（RAG + 自动插桩）、Opik（优化 + 护栏）。
- 解释为什么到 2026 年有 89% 的组织表示已经部署了 Agent 可观测性。
- 用标准库实现一条从追踪到看板（trace-to-dashboard）的流水线，并带有 LLM 评审评估。

## 问题背景

OTel GenAI（第 23 课）给了你 schema。你仍然需要一个平台来摄取 span、运行评估、存储提示词版本，并把回归问题暴露出来。三个竞争者各自侧重生命周期的不同环节。

## 核心概念

### Langfuse (MIT)

- 每月 600 万+ SDK 安装量，19k+ GitHub star。
- 功能：链路追踪、带版本管理与 Playground 的提示词管理、评估（LLM-as-judge、用户反馈、自定义）、会话回放。
- 2025 年 6 月：原先的商业模块（LLM-as-a-judge、标注队列、提示词实验、Playground）以 MIT 许可证开源。
- 最擅长：端到端可观测性，且与提示词管理形成紧密闭环。

### Arize Phoenix (Elastic License 2.0)

- 更深入的 Agent 专项评估：trace 聚类、异常检测、面向 RAG 的检索相关性。
- 原生 OpenInference 自动插桩。
- 可与托管版 Arize AX 搭配用于生产环境。
- 不提供提示词版本管理——定位为与更全面的平台并行使用的漂移/行为回归工具。
- 最擅长：RAG 相关性、行为漂移、异常检测。

### Comet Opik (Apache 2.0)

- 通过 A/B 实验实现自动化提示词优化。
- 护栏（PII 脱敏、话题约束）。
- LLM 评审幻觉检测。
- 来自 Comet 自家测量的基准：Opik 完成日志记录 + 评估用时 23.44 秒，Langfuse 为 327.15 秒（约 14 倍差距）——厂商基准仅作方向性参考。
- 最擅长：优化闭环、自动化实验、护栏强制执行。

### 行业数据

据 Maxim（2026 年实地分析）：89% 的组织已经部署了 Agent 可观测性；质量问题是生产环境的首要障碍（32% 的受访者提及）。

### 如何选择

| 需求 | 选择 |
|------|------|
| 带提示词管理的一体化方案 | Langfuse |
| 深度 RAG 评估 + 漂移检测 | Phoenix |
| 自动化优化 + 护栏 | Opik |
| 开放许可证，不要 ELv2 | Langfuse（MIT）或 Opik（Apache 2.0） |
| Datadog / New Relic 集成 | 任选其一——它们都能导出 OTel |

### 这种模式容易出错的地方

- **没有评估策略。** 没有评估的链路追踪只是昂贵的日志记录。
- **自研 LLM 评审却缺乏事实依据。** CRITIC 模式（第 05 课）同样适用——评审需要外部工具来做事实核验。
- **提示词版本没有关联到 trace。** 当生产环境出现回归时，你无法二分定位到引发问题的那个提示词。

## 从零实现

`code/main.py` 实现了一个标准库版的 trace 收集器 + LLM 评审评估器：

- 摄取符合 GenAI 形态的 span。
- 按会话分组，给失败的运行打标签（护栏触发、低置信度评估）。
- 一个脚本化的 LLM 评审，按评分细则（rubric）给 Agent 响应打分。
- 一份类似看板的汇总：失败率、主要失败原因、评估分数分布。

运行：

```
python3 code/main.py
```

输出：每个会话的评估分数和失败分类，与 Langfuse/Phoenix/Opik 中会展示的内容一致。

## 生产实践

- **Langfuse** 自托管或云端；通过 OTel 或其 SDK 接入。
- **Arize Phoenix** 自托管；用 OpenInference 自动插桩。
- **Comet Opik** 自托管或云端；自动化优化闭环。
- **Datadog LLM Observability** 适合已经在用 Datadog 的运维 + ML 混合团队。

## 交付产物

`outputs/skill-obs-platform-wiring.md` 选定一个平台，并把 trace + 评估 + 提示词版本接入一个现有的 Agent。

## 练习

1. 把一周的 OTel trace 导出到 Langfuse 云端（免费层）。哪些会话失败了？为什么？
2. 为你的领域编写一份 LLM 评审评分细则（事实正确性、语气、范围遵循）。在 50 条 trace 上测试。
3. 对比 Langfuse 的提示词版本管理与 Phoenix 的 trace 聚类。哪个能更快告诉你哪里出了问题？
4. 阅读 Opik 的护栏文档。给你的某次 Agent 运行接入一个 PII 脱敏护栏。
5. 在你自己的语料上对三个平台做基准测试。忽略厂商公布的数字；自己测量。

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|----------------|------------------------|
| 链路追踪（Tracing） | “span 收集器” | 摄取 OTel / SDK span；按会话建立索引 |
| 提示词管理 | “提示词 CMS” | 与 trace 关联的版本化提示词 |
| LLM-as-judge | “自动化评估” | 由另一个 LLM 按评分细则给 Agent 输出打分 |
| 会话回放 | “trace 回放” | 逐步回看过去的运行以便调试 |
| RAG 相关性 | “检索质量” | 检索到的上下文是否匹配查询 |
| trace 聚类 | “行为分组” | 把相似的运行聚类以检测漂移 |
| 护栏强制执行 | “记录时的策略检查” | 对记录的内容做 PII/毒性/范围检查 |

## 延伸阅读

- [Langfuse 文档](https://langfuse.com/) —— 链路追踪、评估、提示词管理
- [Arize Phoenix 文档](https://docs.arize.com/phoenix) —— 自动插桩、漂移检测
- [Comet Opik](https://www.comet.com/site/products/opik/) —— 优化 + 护栏
- [OpenTelemetry GenAI 语义约定](https://opentelemetry.io/docs/specs/semconv/gen-ai/) —— 三家平台共同消费的 schema
