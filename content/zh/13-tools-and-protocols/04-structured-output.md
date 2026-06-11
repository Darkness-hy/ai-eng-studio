# 结构化输出 — JSON Schema、Pydantic、Zod 与约束解码

> "好言好语地请模型返回 JSON"，即使在前沿模型上也有 5% 到 15% 的失败率。结构化输出（structured outputs）用约束解码（constrained decoding）补上了这个缺口：模型在物理上就无法生成会违反 schema 的 token。OpenAI 的 strict 模式、Anthropic 的 schema 类型化工具调用、Gemini 的 `responseSchema`、Pydantic AI 的 `output_type`、以及 Zod 的 `.parse`，是同一思想的五种表现形式。本课将构建 schema 校验器和 strict 模式契约，学习者在每一条生产级信息抽取流水线中都会用到它们。

**Type:** Build
**Languages:** Python (stdlib, JSON Schema 2020-12 subset)
**Prerequisites:** Phase 13 · 02 (function calling deep dive)
**Time:** ~75 minutes

## 学习目标

- 为一个抽取目标编写 JSON Schema 2020-12，并使用恰当的约束（enum、min/max、required、pattern）。
- 解释为什么 strict 模式和约束解码提供的保证与"生成后再校验"不同。
- 区分三种失败模式：解析错误、schema 违例、模型拒答。
- 交付一条带有类型化修复和类型化拒答处理的抽取流水线。

## 问题背景

一个负责阅读采购订单邮件的智能体，需要把自由文本转成 `{customer, line_items, total_usd}`。有三种做法。

**做法一：用提示词要 JSON。**"请用 JSON 回复，字段为 customer、line_items、total_usd。"在前沿模型上有 85% 到 95% 的成功率。失败有六种形式：缺少花括号、多余的尾逗号、类型错误、幻觉出的字段、在 token 上限处被截断、夹带"Here is your JSON:"之类的多余文字。

**做法二：生成后校验。**自由生成，解析，按 schema 校验，失败则重试。可靠但昂贵——每次重试都要付费，而截断类 bug 每出现一次就多花一轮。

**做法三：约束解码。**由服务商在解码时强制执行 schema。无效 token 直接从采样分布中被屏蔽掉。输出保证可解析、保证通过 schema 校验。失败坍缩为唯一一种模式：拒答（模型判定输入无法套进这个 schema）。

2026 年的每一家前沿服务商都提供了某种形式的做法三。

- **OpenAI。**`response_format: {type: "json_schema", strict: true}`，模型拒绝时响应中带有 `refusal` 字段。
- **Anthropic。**对 `tool_use` 输入强制执行 schema；不存在 `stop_reason: "refusal"` 这种东西，但 `end_turn` 且没有工具调用就是拒答的信号。
- **Gemini。**在请求层面使用 `responseSchema`；2026 年 Gemini 已为部分类型提供 token 级语法约束。
- **Pydantic AI。**`output_type=InvoiceModel` 产出类型化为 `InvoiceModel` 的结构化 `RunResult`。
- **Zod（TypeScript）。**用 Zod schema 校验服务商输出的运行时解析器；与 OpenAI 的 `beta.chat.completions.parse` 搭配使用。

共同的主线是：schema 只声明一次，端到端强制执行。

## 核心概念

### JSON Schema 2020-12 —— 通用语言

每家服务商都接受 JSON Schema 2020-12。最常用的构件有：

- `type`：取值为 `object`、`array`、`string`、`number`、`integer`、`boolean`、`null` 之一。
- `properties`：字段名到子 schema 的映射。
- `required`：必须出现的字段名列表。
- `enum`：允许值的封闭集合。
- `minimum` / `maximum`（数字），`minLength` / `maxLength` / `pattern`（字符串）。
- `items`：应用于每个数组元素的子 schema。
- `additionalProperties`：设为 `false` 则禁止额外字段（默认值因模式而异）。

OpenAI strict 模式额外增加三项要求：每个属性都必须列在 `required` 里、所有地方都要 `additionalProperties: false`、不允许未解析的 `$ref`。违反任意一条，API 会在请求时直接返回 400。

### Pydantic：Python 侧的绑定

Pydantic v2 通过 `model_json_schema()` 从 dataclass 风格的模型生成 JSON Schema。Pydantic AI 在其之上做了封装，你只需要写：

```python
class Invoice(BaseModel):
    customer: str
    line_items: list[LineItem]
    total_usd: Decimal
```

智能体框架会在边界处把这个 schema 翻译成 OpenAI strict 模式、Anthropic `input_schema` 或 Gemini `responseSchema`。模型的输出以类型化的 `Invoice` 实例返回。校验失败会抛出带有类型化错误路径的 `ValidationError`。

### Zod：TypeScript 侧的绑定

Zod（`z.object({customer: z.string(), ...})`）是 TS 侧的对应物。OpenAI 的 Node SDK 提供 `zodResponseFormat(Invoice)`，它会翻译成 API 所需的 JSON Schema 载荷。

### 拒答

strict 模式无法强迫模型作答。如果输入根本套不进 schema（"这封邮件是一首诗，不是发票"），模型会输出一个包含原因的 `refusal` 字段。你的代码必须把拒答当作一等公民的结果来处理，而不是当作故障。拒答还可以充当安全信号：当模型被要求从受保护内容的邮件中抽取信用卡号时，会返回附带安全原因的拒答。

### 开源世界中的约束解码

开放权重模型的实现采用三种技术。

1. **基于语法的解码**（`outlines`、`guidance`、`lm-format-enforcer`）：从 schema 构建确定性有限自动机（DFA）；每一步都屏蔽掉会违反该有限状态机（FSM）的 token 的 logits。
2. **配合 JSON 解析器的 logit 屏蔽**：让流式 JSON 解析器与模型同步运行；每一步计算合法的下一 token 集合。
3. **带验证器的投机解码**：廉价的草稿模型提议 token，验证器强制执行 schema。

商业服务商在幕后会选用其中一种。2026 年的最新技术水平是：对短结构化输出比普通生成更快，对长输出则速度大致相当。

### 三种失败模式

1. **解析错误。**输出不是合法 JSON。strict 模式下不可能发生。非 strict 服务商上仍可能发生。
2. **schema 违例。**输出能解析但违反 schema。strict 模式下不可能发生。在其之外很常见。
3. **拒答。**模型拒绝作答。必须作为类型化的结果来处理。

### 重试策略

当你处在 strict 模式之外时（Anthropic 工具调用、非 strict 的 OpenAI、较旧的 Gemini），恢复模式是：

```
generate -> parse -> validate -> if fail, inject error and retry, max 3x
```

一次重试通常就够了。三次重试能兜住弱模型的偶发抖动。超过三次说明 schema 本身有问题：对某些输入模型根本无法满足它，需要修改提示词或 schema。

### 小模型支持

约束解码在小模型上同样有效。一个带语法强制的 3B 参数开源模型，在结构化任务上胜过裸提示词驱动的 70B 参数模型。这正是结构化输出对生产环境至关重要的主要原因：它把可靠性与模型规模解耦了。

## 生产实践

`code/main.py` 提供了一个仅用标准库实现的最小化 JSON Schema 2020-12 校验器（支持 types、required、enum、min/max、pattern、items、additionalProperties）。它封装了一个 `Invoice` schema，并把一段伪造的 LLM 输出送入校验器，演示解析错误、schema 违例和拒答三条路径。在生产环境中，把伪造输出换成任意服务商的真实响应即可。

值得关注的点：

- 校验器返回带路径和消息的类型化 `[ValidationError]` 列表。这正是你希望在重试提示词中呈现的形态。
- 拒答分支不会重试。它记录日志并返回类型化的拒答。Phase 14 · 09 会把拒答用作安全信号。
- `additionalProperties: false` 检查会在对抗性测试输入上触发，展示了 strict 模式为什么能把幻觉字段拒之门外。

## 交付产物

本课产出 `outputs/skill-structured-output-designer.md`。给定一个自由文本抽取目标（发票、工单、简历等），该 skill 会生成一份兼容 strict 模式的 JSON Schema 2020-12，以及一个与之镜像的 Pydantic 模型，并预置好类型化的拒答和重试处理桩代码。

## 练习

1. 运行 `code/main.py`。添加第四个测试用例，其 `total_usd` 为负数。确认校验器以 `minimum` 约束路径拒绝它。

2. 扩展校验器以支持带判别字段（discriminator）的 `oneOf`。常见场景：`line_item` 要么是商品要么是服务，由 `kind` 字段标记。strict 模式在这里有一些微妙的规则；请查阅 OpenAI 的结构化输出指南。

3. 把同一个 Invoice schema 写成 Pydantic BaseModel，对比 `model_json_schema()` 的输出与你手写的 schema。找出 Pydantic 默认设置、而手写版本遗漏的那一个字段。

4. 测量拒答率。构造十条不应被抽取的输入（一段歌词、一个数学证明、一封空白邮件），用真实服务商在 strict 模式下运行。统计拒答与幻觉输出的数量。这就是你做拒答感知重试的基准事实。

5. 从头到尾读完 OpenAI 的结构化输出指南。找出它在 strict 模式下明确禁止、而普通 JSON Schema 允许的那一个构件。然后设计一个非必要地使用了该被禁构件的 schema，并将其重构为 strict 兼容的形式。

## 关键术语

| 术语 | 大家怎么说 | 实际含义 |
|------|----------------|------------------------|
| JSON Schema 2020-12 | "schema 规范" | 所有现代服务商都支持的 IETF 草案 schema 方言 |
| strict 模式 | "保证符合 schema" | OpenAI 的标志位，通过约束解码强制执行 schema |
| 约束解码 | "logit 屏蔽" | 解码时强制执行，屏蔽无效的下一 token |
| 拒答 | "模型拒绝" | 输入无法套进 schema 时的类型化结果 |
| 解析错误 | "非法 JSON" | 输出无法解析为 JSON；strict 模式下不可能发生 |
| schema 违例 | "形状不对" | 能解析但违反了类型 / required / enum / 取值范围 |
| `additionalProperties: false` | "不许多余字段" | 禁止未知字段；OpenAI strict 模式的必备项 |
| Pydantic BaseModel | "类型化输出" | 能生成并校验 JSON Schema 的 Python 类 |
| Zod schema | "TypeScript 输出类型" | 用于校验服务商输出的 TS 运行时 schema |
| 语法强制 | "开放权重的约束解码" | 基于 FSM 的 logit 屏蔽，如 outlines / guidance |

## 延伸阅读

- [OpenAI — Structured outputs](https://platform.openai.com/docs/guides/structured-outputs) — strict 模式、拒答与 schema 要求
- [OpenAI — Introducing structured outputs](https://openai.com/index/introducing-structured-outputs-in-the-api/) — 2024 年 8 月的发布文章，解释了解码层面的保证
- [Pydantic AI — Output](https://ai.pydantic.dev/output/) — 可序列化到各服务商的类型化 output_type 绑定
- [JSON Schema — 2020-12 release notes](https://json-schema.org/draft/2020-12/release-notes) — 规范正典
- [Microsoft — Structured outputs in Azure OpenAI](https://learn.microsoft.com/en-us/azure/foundry/openai/how-to/structured-outputs) — 企业部署说明与 strict 模式注意事项
