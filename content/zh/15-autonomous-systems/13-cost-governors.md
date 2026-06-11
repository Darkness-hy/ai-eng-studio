# 行动预算、迭代上限与成本管控器

> 一家中型电商的智能体（agent）团队启用了「订单跟踪」技能后，月度 LLM 费用从 1,200 美元跳到 4,800 美元。这不是定价出了问题，而是智能体找到了一个新循环，并在循环里持续花钱。Microsoft 的 Agent Governance Toolkit（2026 年 4 月 2 日）把针对这类问题的防御手段固化成了规范：单次请求的 `max_tokens`、单任务的 token 与美元预算、按天/按月的上限、迭代上限、分层模型路由、提示词缓存、上下文窗口管理、昂贵操作前的 HITL 检查点、预算超限时的熔断开关。Anthropic 的 Claude Code Agent SDK 提供了同样的原语，只是名称不同。财务流速限制——例如「10 分钟内花费超过 50 美元就切断访问」——比月度上限更快地捕获循环。

**Type:** Learn
**Languages:** Python (stdlib, layered cost-governor simulator)
**Prerequisites:** Phase 15 · 10 (Permission modes), Phase 15 · 12 (Durable execution)
**Time:** ~60 minutes

## 问题背景

自主智能体每一轮都在花真金白银。聊天机器人输出错了，代价是一条糟糕的回复；智能体陷入坏循环，代价是一张账单。业界对这种失效模式有一个明确的术语——「钱包拒绝服务」（Denial of Wallet）：智能体不停推理、不停调用工具、不停计费，没有任何东西能让它停下来，因为根本没人设计过让它停下来的机制。

解决方案不是设一个数字，而是在不同时间尺度和粒度上叠加一组限制：单次请求、单个任务、每小时、每天、每月。一套设计良好的限制栈能在几分钟内捕获失控循环，在几小时内捕获缓慢泄漏，在一天内捕获糟糕的版本发布。当智能体是长程且自主的时候，也正是这套限制栈让预算还能称之为预算。

这是一节工程课：数学部分微不足道，团队真正栽跟头的地方在于纪律。下面列出的每一项限制，都能在 Microsoft Agent Governance Toolkit 或 Anthropic Claude Code Agent SDK 文档中找到对应的名字。

## 核心概念

### 成本管控器栈

1. **单次请求的 `max_tokens`。** 最简单的一层。防止任何一次调用产生无上限的补全输出。
2. **单任务 token 预算。** 整个运行过程累计不超过 N 个 token。触顶即硬停。
3. **单任务美元预算。** 和 token 预算一样，只是以货币计。在 Claude Code 中是 `max_budget_usd`。
4. **单工具调用上限。** `WebFetch` 调用不超过 N 次、`shell_exec` 调用不超过 N 次，依此类推。
5. **迭代上限（`max_turns`）。** 智能体循环的总迭代次数；防止无限推理循环。
6. **每分钟 / 每小时 / 每天 / 每月上限。** 滚动窗口。在不同时间尺度上捕获泄漏。
7. **财务流速限制。** 例如「10 分钟内花费超过 50 美元就切断访问」。在月度上限触发之前就捕获循环型烧钱。
8. **分层模型路由。** 默认使用较小的模型；只有当分类器判定任务确实需要时，才升级到更大的模型。
9. **提示词缓存。** 系统提示词和稳定上下文存储在服务商的缓存中；重复发送的 token 成本接近于零。
10. **上下文窗口管理。** 通过压缩 / 摘要把活跃上下文保持在阈值以下；直接降低 token 成本。
11. **昂贵操作前的 HITL 检查点。** 在已知昂贵的操作（耗时的工具调用、大文件下载、升级到昂贵模型）之前，要求人工确认一下。
12. **预算超限时的熔断开关。** 任何一个上限触发，会话立即中止。触发记录在案；重新启用需要走单独的流程。

### 为什么要用限制栈，而不是单个上限

单个月度上限只有在钱包见底之后才能发现失控的智能体；单个请求级上限在会话层面什么也拦不住。不同的失效模式需要不同的时间尺度：

- **失控循环**（智能体卡在 5 秒一次的重试里）：由流速限制捕获。
- **缓慢泄漏**（智能体每个任务做了约 2 倍于预期的工作）：由日上限捕获。
- **糟糕的发布**（新版本消耗 5 倍 token）：由周 / 月上限捕获。
- **正常的需求激增**（真实需求，不是 bug）：由小时 / 日上限配合清晰日志捕获。

### Claude Code 的预算接口

Claude Code Agent SDK 公开文档中提供：

- `max_turns` —— 迭代上限。
- `max_budget_usd` —— 美元上限；超限即中止会话。
- `allowed_tools` / `disallowed_tools` —— 工具白名单与黑名单。
- 工具调用前的钩子（hook）位点，用于自定义成本核算。

把这些与权限模式阶梯（第 10 课）结合使用。一个没有设置 `max_budget_usd` 的 `autoMode` 会话就是不受管控的自主性。Anthropic 明确指出 Auto Mode 必须配合预算控制；分类器与成本控制是两个相互独立的维度。

### EU AI Act 与 OWASP Agentic Top 10

Microsoft 的 Agent Governance Toolkit 覆盖了 OWASP Agentic Top 10 以及 EU AI Act 第 14 条（人类监督）的要求。在欧盟的生产环境中，日志记录和上限强制执行不是可选项。

### 真实发生的 1,200 美元 → 4,800 美元案例

Microsoft 文档中记录的真实案例：一个电商智能体在新增一个工具后，月度成本翻了三倍。这个工具允许智能体在每次会话中轮询订单状态。没有循环检测，没有单工具上限，没有针对周环比增长的告警。最终的修复方案是单工具上限加上日增长告警。这是一个可复用的模板：每一个新的工具面都是一个新的潜在循环；每一个新工具都需要自己的上限和自己的告警。

## 生产实践

`code/main.py` 模拟了一次智能体运行，分别对比有无分层成本管控栈的情况。模拟的智能体在若干轮之后漂移进入轮询循环；分层管控栈在流速窗口内就捕获了它，而单一的月度上限要到几天后才会触发。

## 交付产物

`outputs/skill-agent-budget-audit.md` 审计一个待部署智能体方案的成本管控栈，并标出缺失的层。

## 练习

1. 运行 `code/main.py`。确认在轮询循环的运行轨迹上，流速限制先于迭代上限触发。然后禁用流速限制，测量在迭代上限拦住智能体之前，它「花掉」了多少。

2. 为浏览器智能体（第 11 课）设计一套单工具上限。哪个工具需要最严格的上限？哪个工具可以不设限地运行而没有风险？

3. 阅读 Microsoft Agent Governance Toolkit 文档。列出该工具包命名的所有上限类型。把每一种映射到一种失效模式（失控循环、缓慢泄漏、糟糕的发布、需求激增）。

4. 为一个现实任务（例如「在一个仓库里分诊 50 个 issue」）的夜间无人值守运行估算成本。把 `max_budget_usd` 设为你点估计的 2 倍。说明为什么是 2 倍。

5. Claude Code 的 `max_budget_usd` 在会话累计成本上触发。设计一个由外部强制执行的、与之互补的流速限制。什么条件触发切断？重新启用的流程是什么样的？

## 关键术语

| 术语 | 人们的叫法 | 实际含义 |
|---|---|---|
| Denial of Wallet | 「失控账单」 | 智能体循环不断产生花费，且没有任何上限能让它停下 |
| max_tokens | 「单请求上限」 | 单次补全输出大小的上限 |
| max_turns | 「迭代上限」 | 一次会话中智能体循环迭代次数的上限 |
| max_budget_usd | 「美元熔断开关」 | 会话成本上限；超限即中止 |
| Velocity limit | 「速率上限」 | 短时间窗口内花费的限制（例如 50 美元 / 10 分钟） |
| Tiered routing | 「先用小模型」 | 默认使用廉价模型；仅当分类器判定有必要时才升级 |
| Prompt caching | 「缓存系统提示词」 | 服务商侧缓存把重复发送的 token 成本降到接近零 |
| HITL checkpoint | 「人工审批门」 | 昂贵操作之前必须经过人工确认 |

## 延伸阅读

- [Anthropic Claude Code Agent SDK — agent loop and budgets](https://code.claude.com/docs/en/agent-sdk/agent-loop) —— `max_turns`、`max_budget_usd`、工具白名单。
- [Microsoft Agent Framework — human-in-the-loop and governance](https://learn.microsoft.com/en-us/agent-framework/workflows/human-in-the-loop) —— 成本管控检查点。
- [Anthropic — Claude Managed Agents overview](https://platform.claude.com/docs/en/managed-agents/overview) —— 服务商侧成本控制。
- [Anthropic — Prompt caching (Claude API docs)](https://platform.claude.com/docs/en/prompt-caching) —— 缓存机制。
- [Anthropic — Measuring agent autonomy in practice](https://www.anthropic.com/research/measuring-agent-autonomy) —— 长程智能体的成本特征。
