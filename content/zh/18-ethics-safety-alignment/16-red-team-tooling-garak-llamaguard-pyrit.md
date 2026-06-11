# 红队工具链 — Garak、Llama Guard、PyRIT

> 三款生产级工具构成了 2026 年红队技术栈的骨架。Llama Guard（Meta）——基于 Llama-3.1-8B、针对 14 个 MLCommons 危害类别微调的分类器；2025 年的 Llama Guard 4 是从 Llama 4 Scout 剪枝而来的 12B 原生多模态分类器。Garak（NVIDIA）——开源 LLM 漏洞扫描器，提供针对幻觉、数据泄露、提示注入、毒性内容和越狱的静态、动态与自适应探针。PyRIT（Microsoft）——支持 Crescendo、TAP 和自定义转换器链的多轮红队攻防战役，用于深度利用。Llama Guard 3 记载于 Meta 的 "Llama 3 Herd of Models" 论文（arXiv:2407.21783）；Llama Guard 3-1B-INT4 见 arXiv:2411.17713；Garak 的探针架构见 github.com/NVIDIA/garak。这些工具是 2026 年连接红队研究（第 12-15 课）与部署（第 17 课及之后）的生产级接口。

**Type:** Build
**Languages:** Python (stdlib, tool-architecture simulator and Llama Guard-style classifier mock)
**Prerequisites:** Phase 18 · 12-15 (jailbreaks and IPI)
**Time:** ~75 minutes

## 学习目标

- 说明 Llama Guard 3/4 在安全栈中的位置：输入分类器、输出分类器，还是两者兼备。
- 列出 14 个 MLCommons 危害类别，并指出其中一个不那么显而易见的类别（代码解释器滥用）。
- 描述 Garak 的探针架构：探针（probes）、检测器（detectors）、测试框架（harnesses）。
- 描述 PyRIT 的多轮战役结构，以及它如何与 Garak 探针组合使用。

## 问题背景

第 12-15 课呈现了攻击面。生产部署需要可重复、可扩展的评估手段。2026 年有三款工具占据主导地位：Llama Guard（防御分类器）、Garak（扫描器）、PyRIT（战役编排器）。它们各自针对红队生命周期的不同层面。

## 核心概念

### Llama Guard（Meta）

Llama Guard 3 是一个基于 Llama-3.1-8B 微调的模型，用于按照 MLCommons AILuminate 的 14 个类别对输入/输出进行分类：
- 暴力犯罪、非暴力犯罪、性相关犯罪、CSAM、诽谤
- 专业领域建议、隐私、知识产权、无差别杀伤性武器、仇恨
- 自杀/自残、色情内容、选举、代码解释器滥用

支持 8 种语言。使用方式：放在 LLM 之前（输入审核）、之后（输出审核），或两者兼用。这两种用法对应不同的训练数据分布——Llama Guard 3 以单一模型同时处理两种场景。

Llama Guard 3-1B-INT4（arXiv:2411.17713，440MB，在移动端 CPU 上约 30 tokens/s）是其量化的边缘端变体。

Llama Guard 4（2025 年 4 月）为 12B 参数、原生多模态，从 Llama 4 Scout 剪枝而来。它用一个能同时处理文本和图像的分类器，取代了此前的 8B 文本版和 11B 视觉版两个前代模型。

### Garak（NVIDIA）

开源漏洞扫描器。架构如下：
- **探针（Probes）。** 攻击生成器，覆盖幻觉、数据泄露、提示注入、毒性内容、越狱。分为静态（固定提示词）、动态（生成式提示词）、自适应（根据目标输出做出响应）三类。
- **检测器（Detectors）。** 对照预期的失败模式给输出打分——是否有毒、是否泄露、是否被越狱。
- **测试框架（Harnesses）。** 管理探针-检测器配对，运行战役，生成报告。

TrustyAI 将 Garak 与 Llama-Stack 防护盾（Prompt-Guard-86M 输入分类器、Llama-Guard-3-8B 输出分类器）集成，实现对带防护目标的端到端评估。基于层级的评分（TBSA）取代了二元的通过/失败——同一探针下，模型可能在严重性层级 3 通过，却在严重性层级 5 失败。

### PyRIT（Microsoft）

Python Risk Identification Toolkit（Python 风险识别工具包）。用于多轮红队攻防战役。其核心构件包括：
- **转换器（Converters）。** 对种子提示词进行变换——改写、编码、翻译、角色扮演。
- **编排器（Orchestrators）。** 执行战役：Crescendo（逐步升级）、TAP（分支搜索）、RedTeaming（自定义循环）。
- **评分（Scoring）。** LLM 充当裁判，或分类器充当裁判。

PyRIT 是 Garak 的重型表亲。Garak 运行数千个单轮探针；PyRIT 运行专为击破特定失败模式而设计的深度多轮战役。

### 技术栈全景

在模型两侧都部署 Llama Guard。每晚运行 Garak 做回归测试。发布前运行 PyRIT 战役。这是 2026 年大多数生产部署的默认配置。

### 评估陷阱

- **裁判身份。** 三款工具都可以使用 LLM 裁判；裁判的校准程度决定了报告中的 ASR（第 12 课）。报告工具的同时必须说明所用裁判。
- **探针老化。** 随着模型针对性地打补丁，Garak 探针会逐渐失效。自适应探针（PAIR 形态）比静态探针老化得更慢。
- **Llama Guard 对良性内容的误报率。** 早期的 Llama Guard 版本对政治和 LGBTQ+ 内容存在过度标记；Llama Guard 3/4 的校准有所改进，但并未针对每个部署场景单独校准。

### 在 Phase 18 中的位置

第 12-15 课讲攻击家族。第 16 课讲生产工具链。第 17 课（WMDP）讲双重用途能力的评估。第 18 课讲将这些工具纳入政策框架的前沿安全框架。

## 生产实践

`code/main.py` 构建了一个玩具版 Llama Guard 风格分类器（基于 14 个类别的关键词 + 语义特征）、一个玩具版 Garak 测试框架（探针-检测器循环），以及一条 PyRIT 风格的多轮转换器链。你可以用这三款工具对一个模拟目标发起测试，观察它们各自不同的覆盖特征。

## 交付产物

本课产出 `outputs/skill-red-team-stack.md`。给定一段部署描述，它能指出三款工具中哪些适用、各自需要配置什么，以及应当采用怎样的回归测试节奏。

## 练习

1. 运行 `code/main.py`。比较 Llama Guard 风格分类器在单轮攻击与多轮攻击上的检测率。

2. 实现一个新的 Garak 探针：base64 编码的有害请求。测量 Llama Guard 风格分类器对它的检测效果。

3. 为 PyRIT 风格的转换器链增加一个"翻译成法语，再改写"的转换器。重新测量攻击成功率。

4. 阅读 Llama Guard 3 的危害类别列表。找出两个类别，其训练数据在现实中很可能对正当的开发者内容产生高误报率。

5. 比较 Garak 和 PyRIT 的设计理念。各举一个部署场景，论证它们分别是该场景下的正确工具。

## 关键术语

| 术语 | 人们口中的说法 | 实际含义 |
|------|-----------------|------------------------|
| Llama Guard | "那个分类器" | 基于 Llama-3.1-8B/4-12B 微调的安全分类器，覆盖 14 个危害类别 |
| Garak | "那个扫描器" | NVIDIA 开源漏洞扫描器；探针、检测器、测试框架 |
| PyRIT | "那个战役工具" | Microsoft 多轮红队编排器；转换器、编排器、评分 |
| Prompt-Guard | "那个小分类器" | Meta 的 86M 提示注入分类器，与 Llama Guard 配对使用 |
| TBSA | "基于层级的评分" | Garak 用层级化的通过/失败取代二元结果 |
| 转换器链 | "改写 + 编码 + ……" | PyRIT 用于构建多步攻击的组合原语 |
| MLCommons 危害类别 | "那 14 个分类" | Llama Guard 所对标的行业标准分类体系 |

## 延伸阅读

- [Meta — Llama Guard 3 (in Llama 3 Herd paper, arXiv:2407.21783)](https://arxiv.org/abs/2407.21783) — 8B 分类器
- [Meta — Llama Guard 3-1B-INT4 (arXiv:2411.17713)](https://arxiv.org/abs/2411.17713) — 量化的移动端分类器
- [NVIDIA Garak — GitHub](https://github.com/NVIDIA/garak) — 扫描器代码仓库与文档
- [Microsoft PyRIT — GitHub](https://github.com/Azure/PyRIT) — 战役工具包
