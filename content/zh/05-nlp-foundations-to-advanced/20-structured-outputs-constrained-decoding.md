# 结构化输出与约束解码

> 让 LLM 返回 JSON，大多数时候它确实会返回 JSON。但在生产环境里，"大多数"恰恰是问题所在。约束解码（constrained decoding）通过在采样之前修改 logits，把"大多数"变成"始终"。

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 5 · 17 (Chatbots), Phase 5 · 19 (Subword Tokenization)
**Time:** ~60 minutes

## 问题背景

一个分类器向 LLM 发出提示："从 {positive, negative, neutral} 中返回一个。"模型却返回："The sentiment is positive — this review is overwhelmingly favorable because the customer explicitly states that they ..."。你的解析器崩溃了，分类器的 F1 直接变成 0.0。

自由形式的生成不是契约，只是一种建议。而生产系统需要的是契约。

到 2026 年，存在三个层次的方案。

1. **提示词（Prompting）。** 礼貌地请求："只返回 JSON 对象。"在前沿模型上约 80% 有效，小模型上更低。
2. **原生结构化输出 API。** OpenAI 的 `response_format`、Anthropic 的工具调用、Gemini 的 JSON 模式。在受支持的 schema 上很可靠，但被供应商锁定。
3. **约束解码。** 在每个生成步骤修改 logits，让模型*无法*输出无效 token。从构造上保证 100% 有效，且适用于任何本地模型。

本课为这三种方案建立直觉，并说明何时该选用哪一种。

## 核心概念

![Constrained decoding masking invalid tokens at each step](../assets/constrained-decoding.svg)

**约束解码的工作原理。** 在每个生成步骤，LLM 会在整个词表（约 10 万个 token）上产生一个 logit 向量。一个 *logit 处理器（logit processor）*位于模型和采样器之间。它根据当前在目标语法——JSON Schema、正则表达式、上下文无关文法——中的位置，计算哪些 token 是有效的，并把所有无效 token 的 logit 设为负无穷。对剩余 logits 做 softmax 后，概率质量只会落在有效的续写上。

2026 年的主要实现：

- **Outlines。** 将 JSON Schema 或正则表达式编译成有限状态机。每个 token 的有效下一 token 查询是 O(1)。基于 FSM，因此递归 schema 需要展平。
- **XGrammar / llguidance。** 上下文无关文法引擎。能处理递归的 JSON Schema，解码开销接近为零。OpenAI 在其 2025 年的结构化输出实现中致谢了 llguidance。
- **vLLM guided decoding。** 内置 `guided_json`、`guided_regex`、`guided_choice`、`guided_grammar`，后端可选 Outlines、XGrammar 或 lm-format-enforcer。
- **Instructor。** 基于 Pydantic 的封装，可套在任何 LLM 上。校验失败时自动重试。跨供应商，但不修改 logits——它依赖重试加上对结构化输出友好的提示词。

### 反直觉的结论

约束解码往往比无约束生成*更快*。原因有二。第一，它缩小了下一 token 的搜索空间。第二，巧妙的实现会对强制 token 完全跳过生成步骤（像 `{"name": "` 这样的脚手架——每个字节都是确定的）。

### 让你付出代价的陷阱

字段顺序很重要。把 `answer` 放在 `reasoning` 前面，模型会在思考之前就锁定答案。JSON 是有效的，答案却是错的。没有任何校验能捕捉到这种问题。

```json
// BAD
{"answer": "yes", "reasoning": "because ..."}

// GOOD
{"reasoning": "... therefore ...", "answer": "yes"}
```

Schema 的字段顺序属于逻辑，而不是格式。

## 从零实现

### 第 1 步：从零实现正则约束生成

完整的独立 FSM 实现见 `code/main.py`。核心思想 30 行代码就能表达：

```python
def mask_logits(logits, valid_token_ids):
    mask = [float("-inf")] * len(logits)
    for tid in valid_token_ids:
        mask[tid] = logits[tid]
    return mask


def generate_constrained(model, tokenizer, prompt, fsm):
    ids = tokenizer.encode(prompt)
    state = fsm.initial_state
    while not fsm.is_accept(state):
        logits = model.next_token_logits(ids)
        valid = fsm.valid_tokens(state, tokenizer)
        logits = mask_logits(logits, valid)
        tok = sample(logits)
        ids.append(tok)
        state = fsm.transition(state, tok)
    return tokenizer.decode(ids)
```

FSM 记录我们到目前为止已经满足了语法的哪些部分。`valid_tokens(state, tokenizer)` 计算哪些词表 token 能在不偏离接受路径的前提下推进 FSM。

### 第 2 步：用 Outlines 实现 JSON Schema 约束

```python
from pydantic import BaseModel
from typing import Literal
import outlines


class Review(BaseModel):
    sentiment: Literal["positive", "negative", "neutral"]
    confidence: float
    evidence_span: str


model = outlines.models.transformers("meta-llama/Llama-3.2-3B-Instruct")
generator = outlines.generate.json(model, Review)

result = generator("Classify: 'The wait staff was attentive and the food arrived hot.'")
print(result)
# Review(sentiment='positive', confidence=0.93, evidence_span='attentive ... hot')
```

零校验错误。永远如此。FSM 让无效输出根本无法到达。

### 第 3 步：用 Instructor 实现与供应商无关的 Pydantic 方案

```python
import instructor
from anthropic import Anthropic
from pydantic import BaseModel, Field


class Invoice(BaseModel):
    vendor: str
    total_usd: float = Field(ge=0)
    line_items: list[str]


client = instructor.from_anthropic(Anthropic())
invoice = client.messages.create(
    model="claude-opus-4-7",
    max_tokens=1024,
    response_model=Invoice,
    messages=[{"role": "user", "content": "Extract from: 'Acme Corp $420. Widget, Gizmo.'"}],
)
```

机制完全不同。Instructor 不触碰 logits。它把 schema 格式化进提示词，解析输出，并在校验失败时重试（默认 3 次）。适用于任何供应商。重试会增加延迟和成本。跨供应商可移植性是它的卖点。

### 第 4 步：原生供应商 API

```python
from openai import OpenAI

client = OpenAI()
response = client.responses.create(
    model="gpt-5",
    input=[{"role": "user", "content": "Classify: 'The food was cold.'"}],
    text={"format": {"type": "json_schema", "name": "sentiment",
          "schema": {"type": "object", "required": ["sentiment"],
                     "properties": {"sentiment": {"type": "string",
                                                  "enum": ["positive", "negative", "neutral"]}}}}},
)
print(response.output_parsed)
```

服务端的约束解码。在受支持的 schema 上，可靠性与 Outlines 持平。无需管理本地模型，但会被供应商锁定。

## 常见陷阱

- **递归 schema。** Outlines 会把递归展平到固定深度。树状结构的输出（嵌套评论、AST）需要 XGrammar 或 llguidance（基于 CFG）。
- **超大枚举。** 含 10,000 个选项的枚举要么编译缓慢，要么超时。改用检索器：先预测 top-k 候选，再约束到这些候选上。
- **语法过于严格。** 强制 `date: "YYYY-MM-DD"` 正则后，模型在日期缺失时无法输出 `"unknown"`，于是只能编造一个日期。要允许 `null` 或哨兵值。
- **过早锁定。** 见上面的字段顺序陷阱。永远把 reasoning 放在前面。
- **不带 schema 的供应商 JSON 模式。** 纯 JSON 模式只保证 JSON 语法有效，不保证*对你的用例*有效。一定要提供完整的 schema。

## 生产实践

2026 年的技术选型：

| 场景 | 选择 |
|-----------|------|
| OpenAI/Anthropic/Google 模型，简单 schema | 原生供应商结构化输出 |
| 任意供应商，Pydantic 工作流，可容忍重试 | Instructor |
| 本地模型，需要 100% 有效性，扁平 schema | Outlines（FSM） |
| 本地模型，递归 schema | XGrammar 或 llguidance |
| 自托管推理服务器 | vLLM guided decoding |
| 可接受重试的批处理 | Instructor + 最便宜的模型 |

## 交付产物

保存为 `outputs/skill-structured-output-picker.md`：

```markdown
---
name: structured-output-picker
description: Choose a structured output approach, schema design, and validation plan.
version: 1.0.0
phase: 5
lesson: 20
tags: [nlp, llm, structured-output]
---

Given a use case (provider, latency budget, schema complexity, failure tolerance), output:

1. Mechanism. Native vendor structured output, Instructor retries, Outlines FSM, or XGrammar CFG. One-sentence reason.
2. Schema design. Field order (reasoning first, answer last), nullable fields for "unknown", enum vs regex, required fields.
3. Failure strategy. Max retries, fallback model, graceful `null` handling, out-of-distribution refusal.
4. Validation plan. Schema compliance rate (target 100%), semantic validity (LLM-judge), field-coverage rate, latency p50/p99.

Refuse any design that puts `answer` or `decision` before reasoning fields. Refuse to use bare JSON mode without a schema. Flag recursive schemas behind an FSM-only library.
```

## 练习

1. **简单。** 在不使用约束解码的情况下，用提示词让一个小型开源权重模型（如 Llama-3.2-3B）输出 `Review(sentiment, confidence, evidence_span)`。在 100 条评论上统计能解析为有效 JSON 的比例。
2. **中等。** 在同一语料上使用 Outlines 的 JSON 模式。比较合规率、延迟和语义准确率。
3. **困难。** 从零实现一个面向电话号码（`\d{3}-\d{3}-\d{4}`）的正则约束解码器。验证在 1000 个样本上无效输出为 0。

## 关键术语

| 术语 | 大家怎么说 | 实际含义 |
|------|-----------------|-----------------------|
| 约束解码（Constrained decoding） | 强制输出有效结果 | 在每个生成步骤屏蔽无效 token 的 logits。 |
| Logit 处理器 | 负责施加约束的那个东西 | 一个函数：`(logits, state) -> masked_logits`。 |
| FSM | 有限状态机 | 编译后的语法表示；有效下一 token 查询为 O(1)。 |
| CFG | 上下文无关文法 | 能处理递归的文法；比 FSM 慢但表达力更强。 |
| Schema 字段顺序 | 这重要吗？ | 重要——第一个字段就会锁定结论；永远把 reasoning 放在 answer 之前。 |
| Guided decoding | vLLM 对它的叫法 | 同一个概念，集成进了推理服务器。 |
| JSON 模式 | OpenAI 的早期版本 | 只保证 JSON 语法正确；不保证符合 schema。 |

## 延伸阅读

- [Willard, Louf (2023). Efficient Guided Generation for LLMs](https://arxiv.org/abs/2307.09702) —— Outlines 论文。
- [XGrammar paper (2024)](https://arxiv.org/abs/2411.15100) —— 快速的基于 CFG 的约束解码。
- [vLLM — Structured Outputs](https://docs.vllm.ai/en/latest/features/structured_outputs.html) —— 推理服务器集成。
- [OpenAI — Structured Outputs guide](https://platform.openai.com/docs/guides/structured-outputs) —— API 参考及注意事项。
- [Instructor library](https://python.useinstructor.com/) —— 跨供应商的 Pydantic + 重试方案。
- [JSONSchemaBench (2025)](https://arxiv.org/abs/2501.10868) —— 对 6 个约束解码框架的基准评测。
