# 内容审核系统 —— OpenAI、Perspective、Llama Guard

> 生产环境中的内容审核（moderation）系统把第 12-16 课定义的安全策略落到实处。OpenAI Moderation API：`omni-moderation-latest`（2024）基于 GPT-4o 构建，一次调用即可同时分类文本和图像；在多语言测试集上比上一代版本提升 42%；响应 schema 返回 13 个类别布尔值 —— harassment、harassment/threatening、hate、hate/threatening、illicit、illicit/violent、self-harm、self-harm/intent、self-harm/instructions、sexual、sexual/minors、violence、violence/graphic；对大多数开发者免费。分层模式：输入审核（生成前）、输出审核（生成后）、自定义审核（领域规则）。异步并行调用可以掩盖延迟；命中标记时返回占位响应。Llama Guard 3/4（第 16 课）：14 个 MLCommons 危害类别、Code Interpreter Abuse、8 种语言（v3）、多图像（v4）。Perspective API（Google Jigsaw）：早于「LLM 当审核员」浪潮的毒性评分系统；主要是单维度毒性评分，附带 severe-toxicity/insult/profanity 等变体；是内容审核研究的基线。弃用情况：Azure Content Moderator 于 2024 年 2 月弃用、2027 年 2 月退役，由 Azure AI Content Safety 取代。

**Type:** Build
**Languages:** Python (stdlib, three-layer moderation harness)
**Prerequisites:** Phase 18 · 16 (Llama Guard / Garak / PyRIT)
**Time:** ~60 minutes

## 学习目标

- 描述 OpenAI Moderation API 的类别体系，以及它与 Llama Guard 3 的 MLCommons 类别集有何不同。
- 描述三层审核模式（输入、输出、自定义），并说出每一层的一种失效模式。
- 描述 Perspective API 作为前 LLM 时代基线的定位，以及它为何仍在研究中被使用。
- 说出 Azure 的弃用时间表。

## 问题背景

第 12-16 课讲的是攻击与防御工具。第 29 课讲的是已部署的内容审核系统，它们在用户直接接触产品的界面上把这些防御落地。三层模式是 2026 年的默认配置。

## 核心概念

### OpenAI Moderation API

`omni-moderation-latest`（2024）。基于 GPT-4o 构建。一次调用即可分类文本和图像。对大多数开发者免费。

类别（响应 schema 中的 13 个布尔值）：
- harassment、harassment/threatening
- hate、hate/threatening
- self-harm、self-harm/intent、self-harm/instructions
- sexual、sexual/minors
- violence、violence/graphic
- illicit、illicit/violent

多模态支持适用于 `violence`、`self-harm` 和 `sexual`，但不适用于 `sexual/minors`；其余类别仅支持文本。

在 `code/main.py` 的代码框架中，出于教学上的简化，我们把 `/threatening`、`/intent`、`/instructions` 和 `/graphic` 这些子类别折叠进各自的顶级父类别。生产代码应使用完整的 13 类 schema。

在多语言测试集上比上一代审核端点提升 42%。每个类别都给出分数；阈值由应用方自行设定。

### Llama Guard 3/4

已在第 16 课介绍。14 个 MLCommons 危害类别（组织方式与 OpenAI 响应 schema 中的 13 个布尔值不同）。支持 8 种语言（v3）。Llama Guard 4（2025 年 4 月）原生多模态，12B 参数。

OpenAI 和 Llama Guard 的类别体系有重叠但并不一致。OpenAI 用一个宽泛的「illicit」类别；Llama Guard 则把「violent crimes」和「non-violent crimes」分开。部署方根据自身策略与类别体系的契合度来选择。

### Perspective API（Google Jigsaw）

早于「LLM 当审核员」浪潮（2020 年之前）的毒性评分系统。类别：TOXICITY、SEVERE_TOXICITY、INSULT、PROFANITY、THREAT、IDENTITY_ATTACK。以单维度主分数（TOXICITY）为核心，附带子维度变体。

它被广泛用作内容审核研究的基线，因为这个 API 稳定、文档完善，且积累了多年的校准数据。对于现代 LLM 相关的使用场景，Llama Guard 或 OpenAI Moderation 通常是更合适的选择。

### 三层模式

1. **输入审核。** 在生成之前对用户的提示词进行分类。命中标记则拒绝。延迟：一次分类器调用。
2. **输出审核。** 在交付之前对模型输出进行分类。命中标记则替换为拒答。延迟：生成之后再加一次分类器调用。
3. **自定义审核。** 领域专属规则（正则表达式、白名单、业务策略）。可在输入端或输出端运行。

这三层在设计上是串行的：输入审核必须在生成之前完成，输出审核在生成之后运行。并行只发生在层内 —— 对同一段文本并发运行多个分类器（例如 OpenAI Moderation + Llama Guard + Perspective），可以掩盖单个分类器的延迟。作为可选优化，可以在输入审核完成期间展示一条占位响应（「请稍候，正在检查……」），并推迟首个 token 的流式输出。命中标记后的行为是可配置的：拒绝、净化、或升级到人工审核。

### 失效模式

- **仅输入审核。** 无法捕获输出端的幻觉问题（第 12-14 课的编码攻击可以绕过输入分类器）。
- **仅输出审核。** 允许任意输入到达模型；增加成本；向攻击者暴露内部推理过程。
- **仅自定义审核。** 跨类别的鲁棒性不足；正则表达式很脆弱。

分层是默认做法。多重保险（belt-and-suspenders）。

### Azure 弃用

Azure Content Moderator：2024 年 2 月弃用，2027 年 2 月退役。由 Azure AI Content Safety 取代，后者基于 LLM，并与 Azure OpenAI 集成。对于 Azure 部署来说，这次迁移是一个横跨 2024-2027 年的全行业工程。

### 本课在 Phase 18 中的位置

第 16 课在红队语境下讲审核工具。第 29 课讲已部署的审核系统。第 30 课以当前的双重用途能力证据收尾。

## 生产实践

`code/main.py` 构建了一个三层审核框架：输入审核器（关键词 + 类别分数）、输出审核器（对输出运行同一个分类器）、自定义审核器（领域规则）。你可以把各种输入跑一遍，观察每一层各自拦截了什么。

## 交付产物

本课产出 `outputs/skill-moderation-stack.md`。给定一个部署场景，它会推荐一套审核栈配置：输入端用哪个分类器、输出端用哪个、配哪些自定义规则，以及边界情况用什么评判器。

## 练习

1. 运行 `code/main.py`。把一条无害输入、一条边界输入和一条有害输入分别跑过全部三层。报告每条输入触发了哪一层。

2. 扩展该框架，针对某个具体类别加入 Perspective API 风格的毒性评分。对比它的阈值行为与类别分数的差异。

3. 阅读 OpenAI Moderation API 文档和 Llama Guard 3 的类别列表。把每个 OpenAI 类别映射到最接近的 Llama Guard 类别。找出三个无法干净映射的类别。

4. 为一个代码助手部署场景（例如 GitHub Copilot）设计一套审核栈。找出最相关和最不相关的类别，并提出自定义规则。

5. Azure Content Moderator 将于 2027 年 2 月退役。规划一次向 Azure AI Content Safety 的迁移。找出这次迁移中风险最高的环节。

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|-----------------|------------------------|
| OpenAI Moderation | "omni-moderation-latest" | 基于 GPT-4o 的 13 类（文本）分类器，部分支持多模态 |
| Perspective API | "Google Jigsaw toxicity" | 前 LLM 时代的毒性评分基线 |
| Llama Guard | "MLCommons 14-category" | Meta 的危害分类器（v3：8B 文本、8 种语言；v4：12B 多模态） |
| 输入审核 | "pre-generation filter" | 在调用模型前对用户提示词运行的分类器 |
| 输出审核 | "post-generation filter" | 在交付前对模型输出运行的分类器 |
| 自定义审核 | "domain rules" | 部署专属规则（正则、白名单、策略） |
| 分层审核 | "all three layers" | 标准的生产部署模式 |

## 延伸阅读

- [OpenAI Moderation API docs](https://platform.openai.com/docs/api-reference/moderations) — omni-moderation 端点
- [Meta PurpleLlama + Llama Guard](https://github.com/meta-llama/PurpleLlama) — Llama Guard 代码仓库
- [Google Jigsaw Perspective API](https://perspectiveapi.com/) — 毒性评分
- [Azure AI Content Safety](https://learn.microsoft.com/en-us/azure/ai-services/content-safety/) — Azure 的替代产品
