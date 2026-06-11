# AI 网关 — LiteLLM、Portkey、Kong AI Gateway、Bifrost

> 网关（gateway）位于应用与模型提供商之间，核心功能包括提供商路由、故障切换（fallback）、重试、速率限制、密钥引用、可观测性和护栏（guardrails）。2026 年的市场格局：**LiteLLM** 是 MIT 协议的开源项目，支持 100+ 提供商、兼容 OpenAI API，但在约 2000 RPS 时会崩溃（8 GB 内存占用，公开基准测试中出现级联故障）；最适合 Python 技术栈、<500 RPS、开发与原型场景。**Portkey** 定位于控制平面（护栏、PII 脱敏、越狱检测、审计日志），2026 年 3 月以 Apache 2.0 协议开源，每请求 20-40 ms 延迟开销，生产档位 $49/月。**Kong AI Gateway** 构建在 Kong Gateway 之上——Kong 在同样 12 核 CPU 上的自家基准测试：比 Portkey 快 228%，比 LiteLLM 快 859%；定价 $100/模型/月（Plus 档最多 5 个模型）；如果你已在使用 Kong，是企业级的合适选择。**Bifrost**（Maxim AI）——支持可配置退避策略的自动重试，OpenAI 返回 429 时切换到 Anthropic。**Cloudflare / Vercel AI Gateway**——托管、零运维、基础重试。数据驻留（data residency）是决定是否自托管的关键因素；Portkey 和 Kong 处于中间地带，既有开源版也有可选托管版。

**Type:** Learn
**Languages:** Python (stdlib, toy gateway-routing simulator)
**Prerequisites:** Phase 17 · 01 (Managed LLM Platforms), Phase 17 · 16 (Model Routing)
**Time:** ~60 minutes

## 学习目标

- 列举网关的六大核心功能（路由、故障切换、重试、速率限制、密钥管理、可观测性、护栏）。
- 将 2026 年的四款网关（LiteLLM、Portkey、Kong AI、Bifrost）与各自的规模上限和适用场景对应起来。
- 引用 Kong 的基准测试数据（比 Portkey 快 228%、比 LiteLLM 快 859%），并解释为什么它对 >500 RPS 的场景至关重要。
- 根据数据驻留要求和运维预算，在自托管与托管方案之间做出选择。

## 问题背景

你的产品同时调用 OpenAI、Anthropic 和一个自托管的 Llama。每个提供商的 SDK、错误模型、速率限制和认证方式都不一样。你需要故障切换（OpenAI 返回 429 时改试 Anthropic）、统一的凭证存储、统一的可观测性，以及按租户划分的速率限制。

在应用层重复造这套轮子，会让每个服务都与每个提供商耦合在一起。网关层将这一切收拢到一个进程中，对外暴露一个 API（通常兼容 OpenAI），再向各提供商分发请求。

## 核心概念

### 六大核心功能

1. **提供商路由** —— 把 OpenAI、Anthropic、Gemini、自托管模型等收拢到一个 API 后面。
2. **故障切换** —— 遇到 429、5xx 或质量问题时，换一个提供商重试。
3. **重试** —— 指数退避，限制尝试次数。
4. **速率限制** —— 按租户、按密钥、按模型。
5. **密钥引用** —— 运行时从密钥库（vault）拉取凭证（绝不放在应用里）。
6. **可观测性** —— OTel + GenAI 属性（Phase 17 · 13）+ 成本归因。
7. **护栏** —— PII 脱敏、越狱检测、允许话题过滤。

### LiteLLM —— MIT 开源，Python

- 100+ 提供商，兼容 OpenAI，支持路由配置、故障切换和基础可观测性。
- 在 Kong 的基准测试中约 2000 RPS 时崩溃；8 GB 内存占用，持续负载下出现级联故障。
- 最适合：Python 应用、<500 RPS、开发/预发环境网关、实验性路由。
- 成本：开源版免费；云端有免费档。

### Portkey —— 控制平面定位

- 2026 年 3 月起以 Apache 2.0 协议开源。提供护栏、PII 脱敏、越狱检测、审计日志。
- 每请求 20-40 ms 延迟开销。
- 生产档位 $49/月，包含数据保留与 SLA。
- 最适合：需要护栏与可观测性打包方案的强监管行业。

### Kong AI Gateway —— 规模化方案

- 构建在 Kong Gateway 之上（成熟的 API 网关产品，lua+OpenResty）。
- Kong 在 12 核 CPU 等效配置上的自家基准测试：比 Portkey 快 228%，比 LiteLLM 快 859%。
- 定价：$100/模型/月，Plus 档最多 5 个模型。
- 最适合：已在使用 Kong；>1000 RPS；愿意付费获取许可。

### Bifrost（Maxim AI）

- 支持可配置退避策略的自动重试。
- OpenAI 返回 429 时切换到 Anthropic 是其经典用法。
- 较新的入局者；商业产品。

### Cloudflare AI Gateway / Vercel AI Gateway

- 托管、零运维。提供基础重试与可观测性。
- 最适合：部署在 Cloudflare/Vercel 上的边缘 JavaScript 应用。
- 在护栏和速率限制方面不如 Kong/Portkey。

### 自托管 vs 托管

数据驻留是决定性因素。医疗和金融行业默认自托管（LiteLLM、Portkey 开源版或 Kong）。消费级产品默认托管方案（Cloudflare AI Gateway）或中间档（Portkey 托管版）。混合模式：受监管租户用自托管，其余租户用托管。

### 延迟预算

- LiteLLM：典型开销 5-15 ms。
- Portkey：开销 20-40 ms。
- Kong：开销 3-8 ms。
- Cloudflare/Vercel：开销 1-3 ms（边缘优势）。

网关延迟会直接叠加到 TTFT 上。如果 SLA 要求 TTFT P99 < 100 ms，选 Kong 或 Cloudflare。如果 P99 < 500 ms，任意一个都行。

### 速率限制的语义很重要

简单的令牌桶（token-bucket）够用到中等规模。多租户场景需要滑动窗口（sliding-window）+ 突发额度 + 按租户分级。LiteLLM 内置令牌桶；Kong 内置滑动窗口；Portkey 内置分级限流。

### 网关 + 可观测性 + 路由是一体的

Phase 17 · 13（可观测性）+ 16（模型路由）+ 19（网关）在生产环境中属于同一层。要么选一个覆盖全部三项的工具，要么仔细地把它们拼接起来：2026 年的多数部署采用职责分工，将 Helicone（可观测性）或 Portkey（护栏）与 Kong（规模化）组合使用。

### 必须记住的数字

- LiteLLM：约 2000 RPS 时崩溃，8 GB 内存。
- Portkey：20-40 ms 开销；2026 年 3 月起采用 Apache 2.0。
- Kong：比 Portkey 快 228%，比 LiteLLM 快 859%。
- Kong 定价：$100/模型/月，Plus 档最多 5 个。
- Cloudflare/Vercel：边缘部署，1-3 ms 开销。

## 生产实践

`code/main.py` 模拟带故障切换的网关路由，在注入 429/5xx 错误的情况下跨 3 个提供商分发请求。报告延迟、重试率和故障切换命中率。

## 交付产物

本课产出 `outputs/skill-gateway-picker.md`。给定规模、运维姿态、合规要求和延迟预算，它会帮你选出一款网关。

## 练习

1. 运行 `code/main.py`。配置 OpenAI→Anthropic→自托管 的故障切换链。在 5% 提供商错误率下，预期的命中率是多少？
2. 你的 SLA 要求在 300 ms 基线上 TTFT P99 < 200 ms。哪些网关能留在预算之内？
3. 一个医疗客户要求自托管 + PII 脱敏 + 审计。在 Portkey 开源版和 Kong 之间做出选择。
4. 对比 LiteLLM 和 Kong：团队应该在什么 RPS 上限时迁移？
5. 为一个多租户 SaaS 设计速率限制策略：免费档、试用档、付费档。用令牌桶还是滑动窗口？

## 关键术语

| 术语 | 大家怎么说 | 实际含义 |
|------|----------------|------------------------|
| 网关（Gateway） | "API 中介" | 位于应用与提供商之间的进程 |
| LiteLLM | "那个 MIT 协议的" | Python 开源项目，100+ 提供商，2K RPS 时崩溃 |
| Portkey | "护栏网关" | 控制平面 + 可观测性，Apache 2.0 |
| Kong AI Gateway | "规模化那个" | 构建在 Kong Gateway 之上，基准测试领先者 |
| Bifrost | "Maxim 家的网关" | 重试 + 切换 Anthropic 的经典方案 |
| Cloudflare AI Gateway | "边缘托管" | 部署在边缘的托管网关，零运维 |
| PII 脱敏 | "数据清洗" | 在发送给模型前用正则 + NER 进行掩码 |
| 越狱检测 | "提示注入防护" | 作用于用户输入的分类器 |
| 审计日志 | "合规日志" | 每次 LLM 调用的不可篡改记录 |
| 令牌桶 | "简单限流" | 基于补充令牌的速率限制器 |
| 滑动窗口 | "精确限流" | 基于时间窗口的速率限制器；公平性更好 |

## 延伸阅读

- [Kong AI Gateway Benchmark](https://konghq.com/blog/engineering/ai-gateway-benchmark-kong-ai-gateway-portkey-litellm)
- [TrueFoundry — AI Gateways 2026 Comparison](https://www.truefoundry.com/blog/a-definitive-guide-to-ai-gateways-in-2026-competitive-landscape-comparison)
- [Techsy — Top LLM Gateway Tools 2026](https://techsy.io/en/blog/best-llm-gateway-tools)
- [LiteLLM GitHub](https://github.com/BerriAI/litellm)
- [Portkey GitHub](https://github.com/Portkey-AI/gateway)
- [Kong AI Gateway docs](https://docs.konghq.com/gateway/latest/ai-gateway/)
