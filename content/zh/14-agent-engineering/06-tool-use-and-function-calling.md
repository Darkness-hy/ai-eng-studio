# 工具使用与函数调用

> Toolformer（Schick et al., 2023）开创了自监督工具标注。Berkeley Function Calling Leaderboard V4（Patil et al., 2025）设定了 2026 年的标杆：40% 智能体（agentic）、30% 多轮、10% 真实（live）、10% 合成（non-live）、10% 幻觉。单轮调用已基本解决，而记忆、动态决策和长程工具链尚未解决。

**Type:** Build
**Languages:** Python (stdlib)
**Prerequisites:** Phase 14 · 01 (Agent Loop), Phase 13 · 01 (Function Calling Deep Dive)
**Time:** ~60 minutes

## 学习目标

- 解释 Toolformer 的自监督训练信号：只有当工具执行结果能降低下一个 token 的损失时，才保留该工具标注。
- 说出 BFCL V4 的五个评测类别，以及各自衡量的内容。
- 用标准库实现一个工具注册表，包含 schema 校验、参数强制转换和执行沙箱化。
- 诊断 2026 年的三大开放问题：长程工具链、动态决策和记忆。

## 问题背景

早期的工具使用问的是：模型能否预测出一次正确的函数调用？现代的工具使用问的是：模型能否跨 40 步串联工具，带着记忆、在部分可观测的条件下，从工具失败中恢复，并且不会幻觉出不存在的工具？

Toolformer 确立了基线：模型可以通过自监督学会何时调用工具。BFCL V4 定义了 2026 年的评测目标。两者之间的差距，正是生产环境智能体所处的空间。

## 核心概念

### Toolformer（Schick et al., NeurIPS 2023）

核心想法：让模型自己为预训练语料标注候选 API 调用。对每个候选调用执行一次，只有当把工具结果纳入上下文能降低下一个 token 的损失时才保留该标注，最后在过滤后的语料上微调。

覆盖的工具：计算器、问答系统、搜索引擎、翻译器、日历。这个自监督信号只关心工具是否有助于预测文本——不需要任何人工标注。

规模效应结论：工具使用能力随规模涌现。较小的模型会被工具标注拖累；较大的模型则从中受益。这就是为什么 2026 年的前沿模型天生具备很强的工具使用能力，而大多数 7B 模型需要专门的工具使用微调才能可靠工作。

### Berkeley Function Calling Leaderboard V4（Patil et al., ICML 2025）

BFCL 是 2026 年事实上的标准评测。V4 的构成：

- **智能体（40%）**——完整的智能体轨迹：记忆、多轮、动态决策。
- **多轮（30%）**——带工具链的交互式对话。
- **真实（10%）**——用户提交的真实提示词（分布更难）。
- **合成（10%）**——人工构造的测试用例。
- **幻觉（10%）**——检测什么时候不应该调用任何工具。

V3 引入了基于状态的评测：在一串工具调用之后，检查 API 的实际状态（例如「文件是否真的被创建了？」），而不是匹配工具调用的 AST。V4 新增了网页搜索、记忆和格式敏感性等类别。

2026 年的关键发现：单轮函数调用已接近解决。失败集中在记忆（跨轮携带上下文）、动态决策（根据先前结果选择工具）、长程工具链（20 步以上后出现漂移）和幻觉检测（没有合适工具时拒绝调用）。

### 工具 schema

每家提供商都有自己的 schema。细节各异，但形状相同：

```
name: string
description: string (what it does, when to use it)
input_schema: JSON Schema (properties, required, types, enums)
```

Anthropic 直接使用 `input_schema`，OpenAI 使用 `function.parameters`，两者都接受 JSON Schema。描述是承重结构——模型靠读描述来选对工具。糟糕的工具描述是「选错工具」类失败的头号根因。

### 参数校验

不要信任任何工具调用。必须校验：

1. **类型强制转换。** 模型可能在 schema 要求 int 的地方返回字符串 "5"。语义明确时就转换；有歧义时就拒绝。
2. **枚举校验。** 如果 schema 规定 `status in {"open", "closed"}` 而模型给出 `"in_progress"`，应拒绝并返回描述性错误。
3. **必填字段。** 缺少必填字段 -> 立即把错误观察结果返回给模型，而不是让程序崩溃。
4. **格式校验。** 日期、邮箱、URL——用真正的解析器校验，不要用正则。

每次校验失败都应返回结构化的观察结果，让模型能按正确的形状重试。

### 并行工具调用

现代提供商支持在一个助手回合内并行调用多个工具。流程如下：

1. 模型发出 3 个工具调用，各自带有不同的 `tool_use_id`。
2. 运行时执行它们（若相互独立则并行执行）。
3. 每个结果以 `tool_result` 块返回，通过 `tool_use_id` 关联。

工程准则：把关联 ID 当作承重结构。一旦弄混，就会把错误的结果路由给错误的工具调用。

### 沙箱化

工具执行就是沙箱边界。详见第 09 课。简短版：每个工具都应明确读写范围、网络访问、超时和内存上限。通用的 `run_shell(cmd)` 是危险信号；具体的 `git_status()` 更安全。

```figure
tool-routing
```

## 从零实现

`code/main.py` 实现了一个生产形态的工具注册表：

- JSON Schema 子集校验器（仅用标准库）。
- 工具注册，包含描述、输入 schema、超时和执行器。
- 参数强制转换与枚举校验。
- 带关联 ID 的并行工具分发。
- 以结构化字符串形式返回错误观察结果。

运行方式：

```
python3 code/main.py
```

运行轨迹展示了一个迷你智能体在一个回合内调用三个工具，其中一个调用被故意构造成格式错误，并被拒绝，同时返回一条模型可以据此行动的描述性错误。

## 生产实践

每家提供商都有自己的工具 schema——Anthropic、OpenAI、Gemini、Bedrock。如果需要支持多提供商，请使用翻译层（OpenAI Agents SDK、Vercel AI SDK、LangChain 工具适配器）。BFCL 是参考基准——如果工具使用是产品的核心，上线前应先用它评测你的智能体。

## 交付产物

`outputs/skill-tool-registry.md` 可针对给定任务领域生成工具目录、schema 和注册表。其中包含描述质量检查（每个工具的描述是否告诉了模型何时该使用它？）。

## 练习

1. 添加一个「no-op」工具，让模型可以显式拒绝使用任何其他工具。在类 BFCL 的幻觉测试上测量效果。
2. 实现 int-as-string 和 float-as-string 的参数强制转换。从哪里开始，强制转换会掩盖真正的 bug？
3. 添加按工具的超时和熔断器（连续失败 3 次后，60 秒内拒绝该工具）。这会如何改变模型的恢复方式？
4. 阅读 BFCL V4 的说明。挑一个类别（例如「多轮」），用你的智能体跑 10 个示例提示词，报告通过率。
5. 把标准库版校验器移植到 Pydantic 或 Zod。Pydantic/Zod 抓到了哪些玩具版漏掉的问题？

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|----------------|------------------------|
| 函数调用（Function calling） | 「工具使用」 | 经过 schema 校验的结构化输出工具调用 |
| Toolformer | 「自监督工具标注」 | Schick 2023——保留那些结果能降低下一个 token 损失的工具调用 |
| BFCL | 「Berkeley Function Calling Leaderboard」 | 2026 年基准：40% 智能体、30% 多轮、10% 真实、10% 合成、10% 幻觉 |
| 工具 schema | 「给模型看的函数签名」 | name、description、参数的 JSON Schema |
| tool_use_id | 「关联 ID」 | 把工具调用与其结果绑定；并行分发的关键 |
| 幻觉检测 | 「知道何时不该调用」 | V4 类别：没有合适工具时拒绝调用 |
| 参数强制转换 | 「字符串转整数修复」 | 针对可预测 schema 不匹配的窄范围修复；有歧义时拒绝 |
| 沙箱化 | 「工具执行边界」 | 按工具划定读写范围、网络、超时、内存上限 |

## 延伸阅读

- [Schick et al., Toolformer (arXiv:2302.04761)](https://arxiv.org/abs/2302.04761) —— 自监督工具标注
- [Berkeley Function Calling Leaderboard (V4)](https://gorilla.cs.berkeley.edu/leaderboard.html) —— 2026 年评测基准
- [Anthropic, Tool use documentation](https://platform.claude.com/docs/en/agent-sdk/overview) —— Claude Agent SDK 中的生产级工具 schema
- [OpenAI Agents SDK docs](https://openai.github.io/openai-agents-python/) —— function tool 类型与 Guardrails
