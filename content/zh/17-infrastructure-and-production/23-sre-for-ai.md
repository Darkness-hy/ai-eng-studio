# AI 时代的 SRE — 多智能体事故响应、运行手册与预测式检测

> AI SRE 通过 RAG 让 LLM 扎根于基础设施数据（日志、运行手册、服务拓扑），自动化事故处理中的调查、记录与协调环节。2026 年的主流架构模式是多智能体编排（multi-agent orchestration）——由一个监督者（supervisor）协调多个专职智能体（日志、指标、运行手册）；AI 提出假设和查询，人类负责判断性决策。Datadog Bits AI 和 Azure SRE Agent 已将其作为托管产品交付。运行手册（runbook）也在演进：NeuBird Hawkeye 采用对抗式评估（两个模型分析同一事故；一致 = 高置信，分歧 = 不确定）；运维记忆（operational memory）可跨团队人员更替持续保留。自动修复仍保持谨慎：AI 给建议，人类做审批。完全自主的动作范围很窄（重启 Pod、回滚特定部署），且有严格护栏——任何兜售"一劳永逸"的人都是在夸大其词。新兴前沿是事故前预测：MIT 的研究报告称，基于历史日志 + GPU 温度 + API 错误模式训练的 LLM 能提前 10-15 分钟预测 89% 的宕机。预测：到 2026 年底，95% 的企业级 LLM 将具备自动故障转移能力。

**Type:** Learn
**Languages:** Python (stdlib, toy multi-agent incident triage simulator)
**Prerequisites:** Phase 17 · 13 (Observability), Phase 17 · 24 (Chaos Engineering)
**Time:** ~60 minutes

## 学习目标

- 画出多智能体 AI SRE 架构图：监督者 + 专职智能体（日志、指标、运行手册）+ 人工审批关卡。
- 解释为什么自动修复的范围是窄的（重启 Pod、回滚部署）而非宽的（重构服务架构）。
- 说出对抗式评估模式（NeuBird Hawkeye）：两个模型一致 = 高置信；分歧 = 升级人工处理。
- 引用 MIT 89% 提前检测的结果，并说明其运维约束：没有执行机制的预测只是仪表盘。

## 问题背景

一位值班工程师凌晨 3 点被电话叫醒："结账服务错误率飙升。"他查看 Datadog、Loki、三份运行手册和部署日志。30 分钟后才发现根因是 KV 缓存突增导致 vLLM OOM。重启 Pod，错误消失。

到 2026 年，这次调查的前 20 分钟是可以自动化的。按服务分组日志、关联近期部署、匹配运行手册——这些都是 RAG + 工具调用。一个受监督的智能体可以完成第一轮分诊，在人类打开 Datadog 之前就给出假设。

完全自主的修复则是另一回事。重启 Pod：安全。扩容 GPU 池：策略允许的话也安全。重构服务架构：绝对不行。这门学科的关键在于划清这条窄边界。

## 核心概念

### 多智能体架构

```
          Incident
             │
             ▼
        Supervisor
        /    |    \
       ▼     ▼     ▼
  Log agent  Metric agent  Runbook agent
       │     │     │
       └─────┴─────┘
             │
             ▼
        Hypothesis + evidence
             │
             ▼
        Human approval
             │
             ▼
        Action (narrow set)
```

监督者把事故拆解成子查询。专职智能体各自拥有工具访问权限（日志搜索、PromQL、文档检索）。监督者汇总结果，向人类呈现假设和证据。人类批准或重新引导。

### 自动修复的范围

**安全（窄范围）**：重启 Pod、回滚特定部署、在预先批准的范围内扩缩容、启用预先批准的特性开关。

**不安全（宽范围）**：改变服务拓扑、修改资源限制、部署新代码、变更 IAM、改动数据库。

任何兜售"一劳永逸"的人都是在夸大其词。随着 AI SRE 走向成熟，安全集合会逐步扩大，但这条边界是真实存在的。

### 对抗式评估（NeuBird Hawkeye）

两个模型独立分析同一事故。如果它们对根因结论一致，置信度高。如果产生分歧，就升级给人类处理，并同时展示两个假设。模式简单，却能有效过滤掉幻觉出来的根因。

### 运维记忆

团队人员流失是传统 SRE 的隐形杀手——部落知识（tribal knowledge）会随人离开。AI SRE 把运行手册和事后复盘（post-mortem）存入向量数据库；智能体在每次新事故中检索它们。新工程师加入时，AI 已掌握完整历史。

### 事故前预测

MIT 2025 年的研究：基于历史日志、GPU 温度、API 错误模式训练的 LLM，在测试集上能提前 10-15 分钟预测 89% 的宕机。

现实检验：没有执行机制的预测只是仪表盘。真正的运维问题是"预测到了之后，我们做什么？"提前排空节点？呼叫值班？自动扩容？答案取决于具体策略。

### 2026 年的产品

- **Datadog Bits AI** — Datadog 内置的托管 SRE 副驾驶。
- **Azure SRE Agent** — Azure 原生。
- **NeuBird Hawkeye** — 对抗式评估 + 运维记忆。
- **PagerDuty AIOps** — 分诊 + 去重。
- **Incident.io Autopilot** — 事故指挥 + 协调。

### 运行手册即代码

运行手册正从 Confluence 页面演进为带结构化章节（症状、假设、验证、行动）的版本化 markdown。结构化的运行手册能让 RAG 检索效果更好。任何 AI SRE 落地都应从把非结构化运行手册改造成结构化开始。

### 你应该记住的数字

- MIT 提前检测：89% 的宕机，提前 10-15 分钟。
- 多智能体分诊：监督者 + （日志、指标、运行手册）+ 人类。
- 安全的自动修复集合：重启 Pod、回滚部署、在边界内扩缩容。
- 对抗式评估：两个模型独立分析；一致 = 高置信。

## 生产实践

`code/main.py` 模拟一次多智能体分诊：日志智能体发现错误，指标智能体发现 CPU 突增，运行手册智能体匹配到已知问题。监督者对假设进行排序。

## 交付产物

本课产出 `outputs/skill-ai-sre-plan.md`。基于当前的值班安排、事故量和团队成熟度，设计一套 AI SRE 落地方案。

## 练习

1. 运行 `code/main.py`。如果日志智能体和指标智能体产生分歧会怎样？监督者如何裁决？
2. 为你的服务定义三个"安全"的自动修复动作，并逐一说明理由。
3. 编写一份结构化运行手册模板：章节、必填字段、验证命令。
4. 预测式检测在提前 12 分钟时触发。你的策略是什么——呼叫值班、提前排空，还是两者都做？
5. 论证一个 3 人团队应该在 2026 年采用 AI SRE 还是再等等。考虑成熟度、事故量和风险。

## 关键术语

| 术语 | 大家怎么说 | 实际含义 |
|------|----------------|------------------------|
| AI SRE | "值班智能体" | 由 LLM 支撑的事故调查 + 协调 |
| 监督者智能体 | "编排者" | 把事故拆解成子查询的顶层智能体 |
| 专职智能体 | "领域智能体" | 拥有工具访问权限的子智能体（日志、指标、运行手册） |
| 自动修复 | "AI 自己搞定" | 预先批准的窄范围动作；不是大范围的架构重构 |
| 运维记忆 | "向量化运行手册" | 存入向量数据库供 RAG 使用的事后复盘 + 运行手册 |
| 对抗式评估 | "双模型校验" | 独立分析；一致 = 高置信 |
| NeuBird Hawkeye | "搞对抗那家" | 采用对抗式评估 + 记忆模式的产品 |
| Bits AI | "Datadog 的 SRE 智能体" | Datadog 托管的 AI SRE |
| 事故前预测 | "提前检测" | 提前 10-15 分钟预测宕机 |

## 延伸阅读

- [incident.io — AI SRE Complete Guide 2026](https://incident.io/blog/what-is-ai-sre-complete-guide-2026)
- [InfoQ — Human-Centred AI for SRE](https://www.infoq.com/news/2026/01/opsworker-ai-sre/)
- [DZone — AI in SRE 2026](https://dzone.com/articles/ai-in-sre-whats-actually-coming-in-2026)
- [Datadog Bits AI](https://www.datadoghq.com/product/bits-ai/)
- [NeuBird Hawkeye](https://www.neubird.ai/)
- [awesome-ai-sre](https://github.com/agamm/awesome-ai-sre)
