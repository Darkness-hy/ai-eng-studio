# 失败模式：智能体为什么会崩坏

> MASFT（Berkeley，2025）将多智能体失败模式归纳为 3 大类共 14 种。Microsoft 的 Taxonomy 记录了已有 AI 失败如何在智能体场景中被放大。业界一线数据收敛于五种反复出现的模式：幻觉动作、范围蔓延、级联错误、上下文丢失、工具误用。

**Type:** Learn + Build
**Languages:** Python (stdlib)
**Prerequisites:** Phase 14 · 05 (Self-Refine and CRITIC), Phase 14 · 24 (Observability)
**Time:** ~60 minutes

## 学习目标

- 说出 MASFT 的三大失败类别，以及每类中至少四种具体模式。
- 解释为什么智能体化的失败会放大已有的 AI 失败模式（偏见、幻觉）。
- 描述业界反复出现的五种模式及其缓解手段。
- 用标准库实现一个检测器，为智能体执行轨迹打上失败模式标签。

## 问题背景

团队上线的智能体在 90% 的轨迹上工作正常。剩下 10% 的失败并不是随机噪声——它们落在少数几个反复出现的类别里。一旦你能叫出它们的名字，就能对它们做监控，并着手修复。

## 核心概念

### MASFT（Berkeley，arXiv:2503.13657）

多智能体系统失败分类法（Multi-Agent System Failure Taxonomy）。14 种失败模式聚类为 3 大类。标注者之间的 Cohen's Kappa 达到 0.88——说明这些类别可以被可靠地区分。

核心论断：失败是多智能体系统中根本性的设计缺陷，而不是靠更好的基座模型就能修复的 LLM 局限。

### Microsoft《Taxonomy of Failure Mode in Agentic AI Systems》

- 已有的 AI 失败（偏见、幻觉、数据泄露）在智能体场景中被放大。
- 自主性催生新的失败：规模化的意外动作、工具误用、任务漂移（mission drift）。
- 这份白皮书就是智能体产品的风险登记册。

### Characterizing Faults in Agentic AI（arXiv:2603.06847）

- 失败源自编排、内部状态演化以及与环境的交互。
- 不只是"代码写坏了"或"模型输出不好"。

### LLM Agent Hallucinations Survey（arXiv:2509.18970）

两种主要表现：

1. **指令遵循偏离（Instruction-following Deviation）**——智能体不遵守系统提示词。
2. **长程上下文误用（Long-range Contextual Misuse）**——智能体遗忘或错误使用早先轮次的上下文。

子意图错误：遗漏（Omission，漏掉步骤）、冗余（Redundancy，重复步骤）、乱序（Disorder，步骤顺序错乱）。

### 业界反复出现的五种模式

Arize、Galileo、NimbleBrain 在 2024-2026 年的一线分析收敛于：

1. **幻觉动作（Hallucinated actions）。** 智能体调用一个不存在的工具，或者凭空捏造参数。
2. **范围蔓延（Scope creep）。** 智能体把任务扩展到用户要求之外（多建 PR、多发邮件）。
3. **级联错误（Cascading errors）。** 一次错误调用触发下游连锁反应。一个凭空幻觉出来的 SKU 触发四次 API 调用——演变成跨系统事故。
4. **上下文丢失（Context loss）。** 长程任务忘记早期轮次的约束。
5. **工具误用（Tool misuse）。** 调用了正确的工具但参数错误，或者干脆调错了工具。

级联是最致命的。智能体无法区分"我失败了"和"任务本身不可能完成"，遇到 400 错误时常常幻觉出一条成功消息来草草收尾。

### 缓解手段：每一步都设关卡

在推理链的每一步都设置自动验证关卡，对照环境状态检查事实依据。具体做法：

- 每步运行安全分类器（第 21 课）。
- 工具调用参数校验（第 06 课）。
- 将检索到的内容与已知事实交叉验证（第 05 课，CRITIC）。
- 通过重新探测状态来发现成功幻觉（文件真的被创建了吗？）。

### 失败监控容易在哪里出错

- **只给崩溃打标签。** 大多数智能体失败产生的输出看起来是合法的。需要内容层面的检查。
- **没有基线。** 漂移检测需要一个"最近一次正常"的基线；没有它你说不出"情况正在变糟"。
- **过度告警。** 每个失败都触发一次值班呼叫。应当聚类并做限流。

## 从零实现

`code/main.py` 用标准库实现了一个失败模式打标器：

- 覆盖五种模式的合成轨迹数据集。
- 每种模式对应的检测函数（基于工具调用、输出、重复动作的特征模式）。
- 一个打标器，为每条轨迹打标签并报告模式分布。

运行：

```
python3 code/main.py
```

输出：每条轨迹的标签 + 总体分布，这是对 Phoenix 轨迹聚类所揭示内容的一个低成本复刻。

## 生产实践

- **Phoenix**：生产环境的漂移聚类（第 24 课）。
- **Langfuse**：会话回放 + 标注。
- **自研方案**：用于你的可观测性平台检测不到的领域特定特征模式。

## 交付产物

`outputs/skill-failure-detector.md` 可生成针对你所在领域定制、接入轨迹存储的失败模式检测器。

## 练习

1. 添加一个"成功幻觉"检测器：智能体返回成功，但目标状态没有变化。
2. 给你做过的某个产品的 100 条真实轨迹打标签。哪种模式占主导？修复它的成本是多少？
3. 实现一个"级联半径"指标：给定第 N 步发生的失败，它影响了多少个下游步骤？
4. 阅读 MASFT 的 14 种失败模式。挑出适用于你产品的三种，为它们编写检测器。
5. 把一个检测器接入 CI 任务：当 >=5% 的轨迹被打上某种模式标签时，让构建失败。

## 关键术语

| 术语 | 大家怎么说 | 实际含义 |
|------|----------------|------------------------|
| MASFT | "多智能体失败分类法" | Berkeley 提出的 14 种模式分类 |
| 级联错误 | "连锁失败" | 一个早期错误传播贯穿 N 个步骤 |
| 上下文丢失 | "忘了约束" | 长程任务的后期轮次丢掉了早期轮次的事实 |
| 工具误用 | "工具不对 / 参数不对" | 调用本身合法，但调用方式错误 |
| 成功幻觉 | "假装完成" | 智能体在 400 错误上声称成功；状态没有变化 |
| 范围蔓延 | "越权扩张" | 智能体做了超出要求的事 |
| 指令遵循偏离 | "不听话" | 无视系统提示词或用户约束 |
| 子意图错误 | "计划中的 bug" | 计划执行中的遗漏、冗余、乱序 |

## 延伸阅读

- [Cemri et al., MASFT (arXiv:2503.13657)](https://arxiv.org/abs/2503.13657) —— 14 种失败模式，3 大类别
- [Microsoft, Taxonomy of Failure Mode in Agentic AI Systems](https://cdn-dynmedia-1.microsoft.com/is/content/microsoftcorp/microsoft/final/en-us/microsoft-brand/documents/Taxonomy-of-Failure-Mode-in-Agentic-AI-Systems-Whitepaper.pdf) —— 风险登记册
- [Arize Phoenix](https://docs.arize.com/phoenix) —— 漂移聚类的实战
- [Anthropic, Building Effective Agents](https://www.anthropic.com/research/building-effective-agents) —— 何时用更简单的模式从根上避开这些失败
