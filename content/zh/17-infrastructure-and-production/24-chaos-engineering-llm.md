# LLM 生产环境的混沌工程

> 到 2026 年，面向 LLM 的混沌工程已经发展为一门独立学科。在生产环境运行实验前必须具备：明确定义的 SLI/SLO、覆盖 trace+指标+日志的可观测性、自动回滚、运维手册（runbook）、值班机制。架构包含四个平面：控制（实验调度器）、目标（服务、基础设施、数据存储）、安全（守护机制 + 中止开关 + 流量过滤）、可观测性（指标 + trace + 日志），外加反馈（回流到 SLO 调整）。护栏机制是强制要求：当每日错误预算消耗超过预期 2 倍时，燃烧率告警会暂停实验；告警抑制窗口 + trace-ID 关联用于去重告警噪声。节奏安排：每周小规模金丝雀实验 + SLO 评审；每月演练日（game day）+ 事后复盘；每季度跨团队韧性审计 + 依赖关系梳理。LLM 特有实验：内存过载、网络故障、提供商宕机、畸形 prompt、KV 缓存驱逐风暴。工具链：Harness Chaos Engineering（LLM 驱动的实验推荐、爆炸半径缩减、MCP 工具集成）；LitmusChaos（CNCF）；Chaos Mesh（CNCF，Kubernetes 原生）。

**Type:** Learn
**Languages:** Python (stdlib, toy chaos experiment runner)
**Prerequisites:** Phase 17 · 23 (SRE for AI), Phase 17 · 13 (Observability)
**Time:** ~60 minutes

## 学习目标

- 说出混沌工程的五项前置条件（SLI/SLO、可观测性、回滚、运维手册、值班），并解释缺少任意一项为何会让整个实践失效。
- 画出四个平面（控制、目标、安全、可观测性）以及回流到 SLO 的反馈环路。
- 列举五种 LLM 特有实验（内存过载、网络故障、提供商宕机、畸形 prompt、KV 驱逐风暴）。
- 根据技术栈选择工具——Harness、LitmusChaos 或 Chaos Mesh。

## 问题背景

传统技术栈中的混沌测试已是成熟实践。但 LLM 技术栈引入了新的故障模式。一个含有毒字符的 4K token prompt 会让分词器卡住 12 秒。上游提供商返回 429，你的网关开始重试，你的服务在重试放大的并发压力下 OOM。突发负载下的 KV 缓存驱逐风暴引发重新预填充（re-prefill）级联，把计算资源打满。

这些问题在单元测试里一个都暴露不出来。混沌工程就是让你在用户之前发现它们的手段。

## 核心概念

### 前置条件

不满足以下条件时，不要在生产环境运行混沌实验：

1. **SLI/SLO** —— 明确定义的服务级指标和目标。
2. **可观测性** —— trace、指标、日志，并接入仪表盘。
3. **自动回滚** —— Phase 17 · 20 的策略开关回滚。
4. **运维手册** —— 结构化的 runbook，见 Phase 17 · 23。
5. **值班机制** —— 有人负责响应。

缺少任意一项，混沌实验就会演变成真实事故。

### 四个平面 + 反馈

**控制平面** —— 实验调度器（Litmus 工作流、Chaos Mesh 计划任务、Harness UI）。

**目标平面** —— 服务、Pod、节点、负载均衡器、数据存储。

**安全平面** —— 紧急停止开关、告警抑制窗口、爆炸半径限制、错误预算门控。

**可观测性平面** —— 常规指标 + trace-ID 关联，用于区分混沌注入的故障与自然发生的故障。

**反馈环路** —— 实验发现回流到 SLO 调整、runbook 更新和代码修复。

### 护栏机制是强制要求

- **燃烧率告警**：当每日错误预算消耗超过预期 2 倍时，暂停实验。
- **告警抑制窗口**：实验期间静默爆炸半径内与实验无关的告警。
- **Trace-ID 关联**：所有由实验引发的错误都携带标记，方便值班人员去重。

### 五种 LLM 特有实验

1. **内存过载** —— 以高并发发送长上下文请求，强制触发 KV 缓存抢占风暴。观察：服务是优雅地卸载负载，还是直接崩溃？

2. **网络故障** —— 切断推理网关与提供商之间的连接。观察：fallback 是否在 SLA 内生效？（Phase 17 · 19）

3. **提供商宕机模拟** —— 让 OpenAI 100% 返回 429。观察：路由是否会故障转移到 Anthropic？（Phase 17 · 16, 19）

4. **畸形 prompt** —— 注入会卡死分词器的载荷（例如深度嵌套的 unicode、超大 UTF-8 码点）。观察：单个请求是否会锁死一个 worker？

5. **KV 驱逐风暴** —— 打满 vLLM 的 block 预算，强制触发驱逐。观察：LMCache 能否恢复，还是服务持续劣化？

### 节奏安排

- **每周** —— 在 staging 环境做小规模金丝雀实验，或许覆盖 5% 的生产流量。
- **每月** —— 针对特定场景的计划性演练日（game day）；跨团队参与；事后复盘。
- **每季度** —— 跨团队韧性审计；更新依赖关系图。

### 工具链

- **Harness Chaos Engineering** —— 商业产品；AI 生成的实验推荐；爆炸半径缩减；MCP 工具集成。
- **LitmusChaos** —— CNCF 毕业项目；基于 Kubernetes 工作流。
- **Chaos Mesh** —— CNCF 沙箱项目；Kubernetes 原生 CRD 风格。
- **Gremlin** —— 商业产品；支持范围广。
- **AWS FIS** / **Azure Chaos Studio** —— 云厂商托管服务。

### 从小处着手

第一个实验：在稳定流量下杀掉一个 decode 副本的 Pod。观察流量重路由和恢复情况。如果运行正常且看起来安全，再升级到网络混沌实验。

第一个 LLM 特有实验：注入单个提供商持续 5 分钟的 429。观察 fallback 表现。大多数团队会发现自己的 fallback 从未被充分测试过。

### 需要记住的数字

- 四个平面：控制、目标、安全、可观测性。
- 燃烧率暂停阈值：每日预算消耗达到预期的 2 倍。
- 节奏：每周金丝雀、每月演练日、每季度审计。
- 五种 LLM 实验：内存、网络、提供商、畸形 prompt、KV 风暴。

## 生产实践

`code/main.py` 模拟三个带安全平面门控的混沌实验，并报告哪些实验会触发燃烧率中止。

## 交付产物

本课产出 `outputs/skill-chaos-plan.md`。根据给定的技术栈和成熟度，选出前三个实验及配套工具。

## 练习

1. 运行 `code/main.py`。哪个实验触发了燃烧率门控？为什么？
2. 为一个基于 vLLM 的 RAG 服务设计前五个混沌实验，并给出成功标准。
3. 燃烧率告警暂停了你的实验。你如何判定根因——是混沌注入还是自然故障？
4. 论证混沌实验应该在生产环境运行还是只在 staging 运行。什么情况下生产环境才是正确答案？
5. 说出三种通用网络混沌无法复现的 LLM 特有故障模式。

## 关键术语

| 术语 | 人们的说法 | 实际含义 |
|------|----------------|------------------------|
| SLI / SLO | “服务目标” | 指标 + 目标；必备前置条件 |
| 爆炸半径（Blast radius） | “影响范围” | 受实验影响的服务 / 用户集合 |
| 燃烧率告警（Burn-rate alert） | “预算门控” | 当错误预算消耗速率超过预期 2 倍时触发 |
| 演练日（Game day） | “月度演习” | 计划性的跨团队混沌演练 |
| LitmusChaos | “CNCF 工作流” | CNCF 毕业的 Kubernetes 混沌工具 |
| Chaos Mesh | “CNCF CRD” | CNCF 沙箱的 Kubernetes 原生混沌工具 |
| Harness CE | “商业 AI 辅助” | 带 AI 推荐的 Harness 混沌产品 |
| 畸形 prompt（Malformed prompt） | “分词器炸弹” | 会卡死分词过程的输入 |
| KV 驱逐风暴（KV eviction storm） | “抢占级联” | 大规模驱逐触发重新预填充 |

## 延伸阅读

- [DevSecOps School — Chaos Engineering 2026 Guide](https://devsecopsschool.com/blog/chaos-engineering/)
- [Ankush Sharma — Observability for LLMs (book)](https://www.amazon.com/Observability-Large-Language-Models-Engineering-ebook/dp/B0DJSR65TR)
- [LitmusChaos (CNCF)](https://litmuschaos.io/)
- [Chaos Mesh (CNCF)](https://chaos-mesh.org/)
- [Harness Chaos Engineering](https://www.harness.io/products/chaos-engineering)
- [AWS FIS](https://aws.amazon.com/fis/)
