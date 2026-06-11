# 毕业项目 11 — LLM 可观测性与评估看板

> Langfuse 走向了开放内核（open-core）模式。Arize Phoenix 发布了 2026 版 GenAI 语义约定（semconv）映射。Helicone 和 Braintrust 都在按用户的成本归因上加倍投入。Traceloop 的 OpenLLMetry 成为事实上的 SDK 插桩标准。生产环境的典型形态是：ClickHouse 存追踪数据，Postgres 存元数据，Next.js 做 UI，外加一批评估任务（DeepEval、RAGAS、LLM-judge）跑在采样后的追踪数据上。你要自托管搭建一套这样的系统，接入至少四个 SDK 家族，并演示在五分钟内捕获一次人为注入的回归。

**Type:** Capstone
**Languages:** TypeScript (UI), Python / TypeScript (ingest + evals), SQL (ClickHouse)
**Prerequisites:** Phase 11 (LLM engineering), Phase 13 (tools), Phase 17 (infrastructure), Phase 18 (safety)
**Phases exercised:** P11 · P13 · P17 · P18
**Time:** 25 hours

## 问题背景

2026 年，每一个跑生产流量的 AI 团队都会在模型旁边维护一个可观测性平面。成本归因。幻觉检测。漂移监控。越狱信号。SLO 看板。PII 泄漏告警。开源参考实现——Langfuse、Phoenix、OpenLLMetry——已经收敛到 OpenTelemetry GenAI 语义约定作为统一的摄取 schema。现在你可以用一套 SDK 对 OpenAI、Anthropic、Google、LangChain、LlamaIndex 和 vLLM 进行插桩，并产出互相兼容的 span。

你将构建一个自托管看板：从至少四个 SDK 家族摄取数据，对采样后的追踪运行一小组评估任务，检测漂移并触发告警。衡量标准是：面对一次刻意注入的回归（某个提示词开始产出 PII），看板要在五分钟内发现并发出告警。

## 核心概念

摄取采用 OTLP HTTP。SDK 产出符合 GenAI 语义约定的 span：`gen_ai.system`、`gen_ai.request.model`、`gen_ai.usage.input_tokens`、`gen_ai.response.id`、`llm.prompts`、`llm.completions`。span 落入 ClickHouse 做列式分析；元数据（用户、会话、应用）落入 Postgres。

评估以批处理任务的形式跑在采样后的追踪上。DeepEval 评估忠实度（faithfulness）、毒性和回答相关性。当追踪携带检索上下文时，RAGAS 评估检索指标。自定义 LLM-judge 执行领域特定的检查（PII 泄漏、违反策略的回复）。评估结果以评估 span 的形式写回同一个 ClickHouse，并链接到父追踪。

漂移检测关注嵌入空间分布随时间的变化（对提示词嵌入计算 PSI 或 KL 散度），外加评估分数的趋势。告警送入 Prometheus Alertmanager，再转发到 Slack / PagerDuty。UI 使用 Next.js 15 配合 Recharts。

## 架构

```
production apps:
  OpenAI SDK  +  Anthropic SDK  +  Google GenAI SDK
  LangChain + LlamaIndex + vLLM
       |
       v
  OpenTelemetry SDK with GenAI semconv
       |
       v  OTLP HTTP
  collector (ingest, sample, fan-out)
       |
       +-------------+-----------+
       v             v           v
   ClickHouse    Postgres    S3 archive
   (spans)       (metadata)  (raw events)
       |
       +---> eval jobs (DeepEval, RAGAS, LLM-judge)
       |     sampled or all-trace
       |     write eval spans back
       |
       +---> drift detector (PSI / KL on prompt embeddings)
       |
       +---> Prometheus metrics -> Alertmanager -> Slack / PagerDuty
       |
       v
   Next.js 15 dashboard (Recharts)
```

## 技术栈

- 摄取：OpenTelemetry SDK + GenAI 语义约定；OTLP HTTP 传输
- 收集器：带尾部采样（tail-sampling）处理器的 OpenTelemetry Collector（用于控制成本）
- 存储：ClickHouse 存 span，Postgres 存元数据，S3 归档原始事件
- 评估：DeepEval、RAGAS 0.2、Arize Phoenix 评估器套件、自定义 LLM-judge
- 漂移：每周对汇集的提示词嵌入（sentence-transformers）计算 PSI / KL
- 告警：Prometheus Alertmanager -> Slack / PagerDuty
- UI：Next.js 15 App Router + Recharts + server actions
- 开箱即用支持的 SDK：OpenAI、Anthropic、Google GenAI、LangChain、LlamaIndex、vLLM

## 从零实现

1. **收集器配置。** 配置 OpenTelemetry Collector：启用 OTLP HTTP 接收器，配置尾部采样器（保留 100% 的出错追踪和 10% 的成功追踪），并设置到 ClickHouse 和 S3 的导出器。

2. **ClickHouse schema。** 建立 `spans` 表，列与 GenAI 语义约定一一对应：`gen_ai_system`、`gen_ai_request_model`、`input_tokens`、`output_tokens`、`latency_ms`、`prompt_hash`、`trace_id`、`parent_span_id`，外加一个 JSON 字段存放长载荷。按 user_id 和 app_id 添加二级索引。

3. **SDK 覆盖测试。** 用每个 SDK（OpenAI、Anthropic、Google、LangChain、LlamaIndex、vLLM）各写一个小客户端应用，通过 OpenLLMetry 自动插桩。验证每个 SDK 都产出符合规范的 GenAI span 并落入 ClickHouse。

4. **评估任务。** 一个定时任务读取最近 15 分钟的采样追踪，运行 DeepEval 的忠实度、毒性和回答相关性评估。输出是链接到父追踪的评估 span。

5. **自定义 LLM-judge。** 一个 PII 泄漏裁判：给定一条回复，调用守卫 LLM 评估 PII 泄漏的可能性。高分回复进入分诊队列。

6. **漂移检测。** 每周任务计算本周汇集的提示词嵌入与过去 4 周基线之间的 PSI。PSI 超过阈值则告警。

7. **看板。** Next.js 15，包含以下页面：总览（spans/sec、每用户成本、p95 延迟）、追踪（搜索 + 瀑布图）、评估（忠实度趋势、毒性）、漂移（PSI 随时间变化）、告警。

8. **告警链路。** Prometheus exporter 读取评估分数聚合值和延迟分位数；Alertmanager 将警告级路由到 Slack，将严重级路由到 PagerDuty。

9. **回归探针。** 注入一个 bug：被评估的聊天机器人开始以 1% 的概率泄漏伪造的 SSN。测量 MTTR：从 bug 部署到 Slack 告警的时间。

## 生产实践

```
$ curl -X POST https://my-otel-collector/v1/traces -d @trace.json
[collector]  accepted 1 trace, 3 spans
[clickhouse] inserted 3 spans (app=chat, user=u_42)
[eval]       DeepEval faithfulness 0.82, toxicity 0.03
[drift]      weekly PSI 0.08 (below 0.2 threshold)
[ui]         live at https://obs.example.com
```

## 交付产物

交付物是 `outputs/skill-llm-observability.md`。给定一个 LLM 应用，看板能摄取它的追踪、运行评估、对漂移告警，并在 Next.js 中呈现按用户的成本拆解。

| 权重 | 标准 | 衡量方式 |
|:-:|---|---|
| 25 | 追踪 schema 覆盖度 | 产出规范 GenAI span 的 SDK 家族数量（目标：6 个以上） |
| 20 | 评估正确性 | DeepEval / RAGAS 分数与人工标注集对比 |
| 20 | 看板用户体验 | 注入回归的 MTTR（目标 5 分钟以内） |
| 20 | 成本 / 规模 | 持续摄取 1k spans/sec 且无积压 |
| 15 | 告警 + 漂移检测 | Prometheus/Alertmanager 链路端到端验证 |
| **100** | | |

## 练习

1. 为 Haystack 框架添加自定义插桩。验证规范 span 落入 ClickHouse，且 `gen_ai.*` 属性准确无误。

2. 在同一批追踪上把 DeepEval 换成 Phoenix 评估器。测量两个评估引擎之间的分数偏差。

3. 加强漂移检测器：按 app-id 而不是全局计算 PSI。展示每个应用的漂移轨迹。

4. 添加一个「用户影响」页面：每用户成本和每用户失败率，配上迷你趋势图（sparkline）。

5. 构建一个尾部采样策略：保留 100% 毒性 > 0.5 的追踪，外加对其余追踪做 10% 的分层采样。测量由此引入的采样偏差。

## 关键术语

| 术语 | 大家怎么说 | 实际含义 |
|------|-----------------|------------------------|
| GenAI semconv | 「OTel 的 LLM 属性」 | 2025 年 OpenTelemetry 规范中针对 LLM span 属性的定义（系统、模型、token 数） |
| 尾部采样（Tail sampling） | 「追踪结束后再采样」 | 收集器在追踪完成后才决定保留或丢弃（可以先看到错误） |
| PSI | 「群体稳定性指数」 | 比较两个分布的漂移指标；> 0.2 通常意味着出现了显著漂移 |
| LLM-judge | 「用模型做评估」 | 用一个 LLM 按评分标准给另一个 LLM 的输出打分（忠实度、毒性、PII） |
| 尾部采样策略 | 「保留规则」 | 决定哪些追踪持久化、哪些丢弃的规则；出错的全留 + 按比例采样 |
| 评估 span | 「链接的评估追踪」 | 携带评估分数的子 span，链接到原始 LLM 调用 span |
| 每用户成本 | 「单位经济学」 | 一个时间窗口内归因到某个 user_id 的美元成本；关键产品指标 |

## 延伸阅读

- [Langfuse](https://github.com/langfuse/langfuse) — 开放内核可观测性平台的参考实现
- [Arize Phoenix](https://github.com/Arize-ai/phoenix) — 另一个参考实现，漂移支持很强
- [OpenLLMetry (Traceloop)](https://github.com/traceloop/openllmetry) — 自动插桩 SDK 家族
- [OpenTelemetry GenAI semantic conventions](https://opentelemetry.io/docs/specs/semconv/gen-ai/) — 摄取 schema
- [Helicone](https://www.helicone.ai) — 另一个托管型可观测性方案
- [Braintrust](https://www.braintrust.dev) — 另一个以评估为先的平台
- [ClickHouse documentation](https://clickhouse.com/docs) — 列式 span 存储
- [DeepEval](https://github.com/confident-ai/deepeval) — 评估器库
