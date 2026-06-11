# LLM 路由层 — LiteLLM、OpenRouter、Portkey

> 被供应商锁定的代价很高。不同的工具调用工作负载适合不同的模型。路由网关提供统一的 API 接口、重试、故障转移（failover）、成本追踪和护栏（guardrails）。2026 年有三种主流形态：LiteLLM（开源自托管）、OpenRouter（托管 SaaS）、Portkey（生产级，2026 年 3 月开源）。本课讲清楚选型标准，并带你走一遍纯标准库实现的路由网关。

**Type:** Learn
**Languages:** Python (stdlib, routing + failover + cost tracker)
**Prerequisites:** Phase 13 · 02 (function calling), Phase 13 · 17 (gateways)
**Time:** ~45 minutes

## 学习目标

- 区分自托管、托管和生产级三类路由方案。
- 实现一条回退链（fallback chain），在供应商故障时按既定优先级顺序重试。
- 跨供应商追踪每次请求的成本和 token 用量。
- 针对给定的生产约束，在 LiteLLM、OpenRouter 和 Portkey 之间做出选择。

## 问题背景

需要做供应商路由的场景：

1. **成本。** Claude Sonnet 的价格是 Haiku 的 3 倍。分诊类任务用 Haiku 就够了；综合归纳类任务才值得用 Sonnet。按请求路由。

2. **故障转移。** OpenAI 出了一小时故障，所有请求都失败。你希望自动回退到 Anthropic，而不需要重新部署。

3. **延迟。** 实时聊天界面需要很快的首个 token 响应时间（time-to-first-token），批量摘要任务则不需要。按延迟 SLA 路由。

4. **合规。** 欧盟用户的数据必须留在欧盟区域。按区域路由。

5. **实验。** 在同一工作负载上对两个模型做 A/B 测试。按测试分桶路由。

每接入一个供应商就手写一遍这些逻辑，全是重复劳动。路由网关提供一个 OpenAI 兼容的 API，其余的它来处理。

## 核心概念

### OpenAI 兼容的代理形态

大家都说 OpenAI 这套"方言"。路由网关暴露 `/v1/chat/completions` 端点，接受 OpenAI 格式的请求，内部再代理到 Anthropic / Gemini / Cohere / Ollama / 任何后端。客户端对此毫无感知。

### 模型别名

你的代码里写的不是 `claude-3-5-sonnet-20251022`，而是 `our_smart_model`。网关负责把别名映射到真实模型。当 Anthropic 发布 Claude 4 时，你只需在服务端改一下别名映射，代码一行都不用动。

### 回退链

```
primary: openai/gpt-4o
on 5xx: anthropic/claude-3-5-sonnet
on 5xx: google/gemini-1.5-pro
on 5xx: refuse
```

网关在配置里定义这条链。重试会计入预算，避免回退级联导致成本失控。

### 语义缓存

完全相同或近似相同的提示词会命中缓存，而不是打到供应商。在重复的智能体循环中，节省可达 30% 到 60%。缓存键基于嵌入（embedding），近似相同的提示词共享同一个缓存槽位。

### 护栏

网关层面的护栏包括：

- **PII 脱敏。** 在发送提示词之前，先做一轮基于正则或机器学习的检查。
- **策略违规。** 拒绝包含违禁内容的提示词。
- **输出过滤。** 清洗模型输出，防止信息泄露。

Portkey 和 Kong 都自带一套有立场的护栏方案。LiteLLM 则把护栏留作可选项。

### 按 API key 限流

一个 API key 对应一个团队。按 key 设置预算，防止某个团队耗尽共享配额。大多数网关都支持这一点。

### 自托管与托管的取舍

| 因素 | LiteLLM（自托管） | OpenRouter（托管） | Portkey（生产级） |
|--------|----------------------|----------------------|----------------------|
| 代码 | 开源，Python | 托管 SaaS | 开源（2026 年 3 月）+ 托管 |
| 部署 | 自己部署代理 | 注册即用 | 两者皆可 |
| 供应商数量 | 100+ | 300+ | 100+ |
| 计费 | 用自己的 key | OpenRouter 积分 | 用自己的 key |
| 可观测性 | OpenTelemetry | 控制台仪表盘 | 完整 OTel + PII 脱敏 |
| 适合场景 | 想要完全掌控的团队 | 快速原型验证 | 有合规要求的生产环境 |

如果你有 SRE 团队并且需要数据主权，选 LiteLLM。如果你想要单一订阅、零基础设施，选 OpenRouter。如果你需要开箱即用的护栏和合规能力，选 Portkey。

### 成本追踪

每个请求都携带 `provider`、`model`、`input_tokens`、`output_tokens`。乘以每个模型的单 token 价格（来自网关维护的价格表），就能按用户 / 团队 / 项目聚合统计。

### MCP 与路由结合

一个网关可以同时路由 LLM 调用和 MCP 采样（sampling）请求。当采样请求的 modelPreferences 偏好某个特定模型时，网关会将其翻译到对应的后端。正因如此，Phase 13 · 17 的 MCP 网关和本课的路由网关有时会合并成同一个服务。

### 路由策略

- **静态优先级。** 取列表中的第一个；出错时回退。
- **负载均衡。** 轮询或加权分配。
- **成本感知。** 在满足延迟 / 质量要求的前提下选最便宜的模型。
- **延迟感知。** 选过去 N 分钟内最快的模型。
- **任务感知。** 用提示词分类器把编码任务路由到一个模型，摘要任务路由到另一个。

## 生产实践

`code/main.py` 用约 150 行代码实现了一个路由网关：接受 OpenAI 格式的请求，翻译成各供应商的桩（stub）调用，执行优先级回退链，追踪每次请求的成本，并对输入做一轮 PII 脱敏。用三个场景运行它：正常请求、主供应商故障触发回退、PII 泄露被脱敏拦截。

重点看这几处：

- `ROUTES` 字典：别名 -> 按优先级排序的具体供应商列表。
- 回退循环在 5xx 错误时重试。
- 成本追踪器用 token 用量乘以各模型的单价。
- PII 脱敏器在转发前清洗形如 SSN 的模式。

## 交付产物

本课产出 `outputs/skill-routing-config-designer.md`。给定一份工作负载画像（延迟、成本、合规），该 skill 会在 LiteLLM / OpenRouter / Portkey 中做出选择，并生成一份路由配置。

## 练习

1. 运行 `code/main.py`。触发故障场景，确认回退落在第二个供应商上，且成本被正确归属。

2. 加入语义缓存：用提示词的 SHA256 作为查找键，缓存命中时立即返回。测量重复调用时节省了多少成本。

3. 加一个提示词分类器，把 "code ..." 开头的提示词路由到偏向智能的别名，把 "summarize ..." 开头的提示词路由到偏向速度的别名。

4. 设计按团队的预算：每个团队有月度支出上限，达到上限后网关拒绝请求。选定一种执行粒度（按请求或按时间窗口）。

5. 并排阅读 LiteLLM、OpenRouter 和 Portkey 的文档。各找出一个该产品独有、其余两家没有的功能。

## 关键术语

| 术语 | 大家怎么说 | 实际含义 |
|------|----------------|------------------------|
| 路由网关 | "LLM 代理" | 位于多个供应商之前的统一 API 接口层 |
| OpenAI 兼容 | "说 OpenAI 这套 schema" | 接受 `/v1/chat/completions` 格式，翻译到任意后端 |
| 模型别名 | "our_smart_model" | 代码里使用的名字，由网关映射到具体模型 |
| 回退链 | "重试列表" | 失败时按顺序尝试的供应商列表 |
| 语义缓存 | "提示词嵌入缓存" | 以提示词的嵌入为键，近似重复的提示词共享缓存命中 |
| 护栏 | "输入/输出过滤器" | 脱敏 PII，拒绝违反策略的内容 |
| 按 key 限流 | "团队预算" | 以 API key 为作用域的配额 |
| 成本追踪 | "按请求计费" | 按模型聚合 token 用量 x 单价 |
| LiteLLM | "那个开源代理" | 可自托管的开源路由网关 |
| OpenRouter | "那个托管 SaaS" | 基于积分计费的托管网关 |
| Portkey | "生产级选项" | 开源 + 托管，内置护栏 |

## 延伸阅读

- [LiteLLM — docs](https://docs.litellm.ai/) — 自托管路由网关
- [OpenRouter — quickstart](https://openrouter.ai/docs/quickstart) — 托管路由 SaaS
- [Portkey — docs](https://portkey.ai/docs) — 带护栏的生产级路由
- [TrueFoundry — LiteLLM vs OpenRouter](https://www.truefoundry.com/blog/litellm-vs-openrouter) — 选型指南
- [Relayplane — LLM gateway comparison 2026](https://relayplane.com/blog/llm-gateway-comparison-2026) — 厂商综述
