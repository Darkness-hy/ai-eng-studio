# 结构化输出：JSON、Schema 校验与约束解码

> LLM 返回的是字符串，而你的应用需要的是 JSON。这道鸿沟搞垮的生产系统比任何模型幻觉都多。结构化输出就是连接自然语言与类型化数据的桥梁。做对了，你的 LLM 就是一个可靠的 API；做错了，你就得在凌晨三点用正则表达式解析自由文本。

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 10, Lessons 01-05 (LLMs from Scratch)
**Time:** ~90 minutes
**Related:** Phase 5 · 20（Structured Outputs & Constrained Decoding）讲解了解码器层面的理论（FSM/CFG logit 处理器、Outlines、XGrammar）。本课聚焦于生产环境的 SDK 接口（OpenAI 的 `response_format`、Anthropic 的工具使用、Instructor）——如果你想理解 API 之下到底发生了什么，请先阅读 Phase 5 · 20。

## 学习目标

- 使用 OpenAI 和 Anthropic 的 API 参数实现 JSON 模式与 schema 约束输出
- 构建一个 Pydantic 校验层，拒绝格式错误的 LLM 输出，并带着错误反馈进行重试
- 解释约束解码（constrained decoding）如何在 token 层面强制生成合法 JSON，无需任何后处理
- 设计健壮的抽取提示词，可靠地把非结构化文本转换为类型化的数据结构

## 问题背景

你问 LLM："从这段文本中提取产品名称、价格和库存状态。"它回答：

```
The product is the Sony WH-1000XM5 headphones, which cost $348.00 and are currently in stock.
```

这是一个完全正确的回答，但对你的应用来说毫无用处。你的库存系统需要的是 `{"product": "Sony WH-1000XM5", "price": 348.00, "in_stock": true}`。你需要的是一个带有特定键名、特定类型、特定取值约束的 JSON 对象，而不是一句话。

最朴素的方案：在提示词里加一句"请以 JSON 格式回答"。这在 90% 的情况下有效。剩下的 10% 里，模型会把 JSON 包在 markdown 代码围栏里，或者加上一句"Here's the JSON:"之类的开场白，或者因为提前闭合了某个括号而产生语法非法的 JSON。你的 JSON 解析器崩溃，流水线中断。于是你加上 try/except 和重试循环。重试有时又会产生不同的数据。现在你在解析问题之上又多了一个一致性问题。

这不是提示词工程问题，而是解码问题。模型从左到右生成 token。在每个位置上，它从 10 万以上的词表中选出最可能的下一个 token。在任何给定位置，词表中的绝大多数选项都会产生非法 JSON。如果模型刚输出了 `{"price":`，那么下一个 token 必须是数字、引号（表示字符串）、`null`、`true`、`false` 或负号——其它任何选择都会让 JSON 非法。没有约束时，模型可能选出一个语义上完全合理、但在语法上是灾难的英文单词。

## 核心概念

### 结构化输出的控制谱系

结构化输出控制分为四个层级，可靠性逐级递增。

```mermaid
graph LR
    subgraph Spectrum["Structured Output Spectrum"]
        direction LR
        A["Prompt-based\n'Return JSON'\n~90% valid"] --> B["JSON Mode\nGuaranteed valid JSON\nNo schema guarantee"]
        B --> C["Schema Mode\nJSON + matches schema\nGuaranteed compliance"]
        C --> D["Constrained Decoding\nToken-level enforcement\n100% compliance"]
    end

    style A fill:#1a1a2e,stroke:#ff6b6b,color:#fff
    style B fill:#1a1a2e,stroke:#ffa500,color:#fff
    style C fill:#1a1a2e,stroke:#51cf66,color:#fff
    style D fill:#1a1a2e,stroke:#0f3460,color:#fff
```

**基于提示词**（"请返回合法 JSON"）：没有任何强制机制。模型通常会遵从，但有时不会。可靠性约 90%。失败形态：markdown 围栏、开场白文本、输出被截断、结构错误。

**JSON 模式**：API 保证输出是语法合法的 JSON。OpenAI 的 `response_format: { type: "json_object" }` 即开启此模式。输出一定能成功解析，但不保证符合你期望的 schema——可能有多余的键、错误的类型、缺失的字段。

**Schema 模式**：API 接收一个 JSON Schema，并保证输出与之匹配。到 2026 年，所有主流厂商都原生支持：OpenAI 的 `response_format: { type: "json_schema", json_schema: {...} }`（也可以用 `tool_choice="required"`）、Anthropic 带 `input_schema` 的工具使用，以及 Gemini 的 `response_schema` + `response_mime_type: "application/json"`。输出会精确具备你指定的键名、类型和约束。

**约束解码**：在生成过程中的每个 token 位置上，解码器把所有会导致非法输出的 token 屏蔽掉。如果 schema 要求一个数字而模型即将输出一个字母，该 token 的概率会被置为零。模型只能生成通向合法输出的 token。这正是 OpenAI 的结构化输出模式以及 Outlines、Guidance 等库在底层实现的机制。

### JSON Schema：契约语言

JSON Schema 是你告诉模型（或校验层）输出必须长成什么形状的方式。所有主流结构化输出系统都使用它。

```json
{
  "type": "object",
  "properties": {
    "product": { "type": "string" },
    "price": { "type": "number", "minimum": 0 },
    "in_stock": { "type": "boolean" },
    "categories": {
      "type": "array",
      "items": { "type": "string" }
    }
  },
  "required": ["product", "price", "in_stock"]
}
```

这个 schema 的含义是：输出必须是一个对象，包含字符串类型的 `product`、非负数值类型的 `price`、布尔类型的 `in_stock`，以及一个可选的字符串数组 `categories`。任何不匹配的输出都会被拒绝。

Schema 能处理各种棘手情况：嵌套对象、带类型元素的数组、枚举（把字符串限制在特定取值集合内）、模式匹配（对字符串做正则约束），以及组合器（oneOf、anyOf、allOf，用于多态输出）。

### Pydantic 模式

在 Python 中，你不需要手写 JSON Schema。定义一个 Pydantic 模型，它会自动为你生成 schema。

```python
from pydantic import BaseModel

class Product(BaseModel):
    product: str
    price: float
    in_stock: bool
    categories: list[str] = []
```

这段代码生成的 JSON Schema 与上面那个完全相同。Instructor 库（以及 OpenAI 的 SDK）直接接受 Pydantic 模型：传入模型类，拿回一个经过校验的实例。如果 LLM 输出不匹配，Instructor 会自动重试。

### 函数调用（Function Calling）/ 工具使用（Tool Use）

这是解决同一问题的另一种接口。不是让模型直接生成 JSON，而是定义带类型参数的"工具"（函数），模型输出一个带结构化参数的函数调用。OpenAI 称之为"function calling"，Anthropic 称之为"tool use"。结果相同：结构化数据。

```mermaid
graph TD
    subgraph ToolUse["Tool Use Flow"]
        U["User: Extract product info\nfrom this review text"] --> M["Model processes input"]
        M --> TC["Tool Call:\nextract_product(\n  product='Sony WH-1000XM5',\n  price=348.00,\n  in_stock=true\n)"]
        TC --> V["Validate against\nfunction schema"]
        V --> R["Structured Result:\n{product, price, in_stock}"]
    end

    style U fill:#1a1a2e,stroke:#0f3460,color:#fff
    style TC fill:#1a1a2e,stroke:#e94560,color:#fff
    style V fill:#1a1a2e,stroke:#ffa500,color:#fff
    style R fill:#1a1a2e,stroke:#51cf66,color:#fff
```

当模型需要自己选择调用哪个函数、而不仅仅是填充参数时，工具使用是更好的选择。如果你有 10 个不同的抽取 schema，模型必须根据输入选出正确的那个，工具使用能同时给你 schema 选择和结构化输出。

### 常见失败模式

即使有 schema 强制约束，结构化输出仍可能以一些隐蔽的方式失败。

**幻觉值**：输出符合 schema，但内容是编造的。文本里写的是 $348，模型却输出 `{"price": 299.99}`。Schema 校验抓不住这种错误——类型正确，值是错的。

**枚举混淆**：你把某字段限制为 `["in_stock", "out_of_stock", "preorder"]`，模型输出 `"available"`——语义上正确，但不在允许的集合内。良好的约束解码可以杜绝这种情况，基于提示词的方法做不到。

**嵌套对象深度**：嵌套很深的 schema（4 层以上）会产生更多错误。每多一层嵌套，模型就多一个可能跟丢结构的地方。

**数组长度**：模型生成的数组元素可能过多或过少。Schema 支持 `minItems` 和 `maxItems`，但并非所有厂商都在解码层面强制执行它们。

**可选字段缺失**：模型省略了那些技术上可选、但对你的用例语义上很重要的字段。即使数据有时确实缺失，也应在 schema 中把它们设为 required——强制模型显式输出 `null`。

## 从零实现

### 第 1 步：JSON Schema 校验器

从零构建一个校验器，检查一个 Python 对象是否匹配某个 JSON Schema。这是在输出端运行、用来验证合规性的部分。

```python
import json

def validate_schema(data, schema):
    errors = []
    _validate(data, schema, "", errors)
    return errors

def _validate(data, schema, path, errors):
    schema_type = schema.get("type")

    if schema_type == "object":
        if not isinstance(data, dict):
            errors.append(f"{path}: expected object, got {type(data).__name__}")
            return
        for key in schema.get("required", []):
            if key not in data:
                errors.append(f"{path}.{key}: required field missing")
        properties = schema.get("properties", {})
        for key, value in data.items():
            if key in properties:
                _validate(value, properties[key], f"{path}.{key}", errors)

    elif schema_type == "array":
        if not isinstance(data, list):
            errors.append(f"{path}: expected array, got {type(data).__name__}")
            return
        min_items = schema.get("minItems", 0)
        max_items = schema.get("maxItems", float("inf"))
        if len(data) < min_items:
            errors.append(f"{path}: array has {len(data)} items, minimum is {min_items}")
        if len(data) > max_items:
            errors.append(f"{path}: array has {len(data)} items, maximum is {max_items}")
        items_schema = schema.get("items", {})
        for i, item in enumerate(data):
            _validate(item, items_schema, f"{path}[{i}]", errors)

    elif schema_type == "string":
        if not isinstance(data, str):
            errors.append(f"{path}: expected string, got {type(data).__name__}")
            return
        enum_values = schema.get("enum")
        if enum_values and data not in enum_values:
            errors.append(f"{path}: '{data}' not in allowed values {enum_values}")

    elif schema_type == "number":
        if not isinstance(data, (int, float)):
            errors.append(f"{path}: expected number, got {type(data).__name__}")
            return
        minimum = schema.get("minimum")
        maximum = schema.get("maximum")
        if minimum is not None and data < minimum:
            errors.append(f"{path}: {data} is less than minimum {minimum}")
        if maximum is not None and data > maximum:
            errors.append(f"{path}: {data} is greater than maximum {maximum}")

    elif schema_type == "boolean":
        if not isinstance(data, bool):
            errors.append(f"{path}: expected boolean, got {type(data).__name__}")

    elif schema_type == "integer":
        if not isinstance(data, int) or isinstance(data, bool):
            errors.append(f"{path}: expected integer, got {type(data).__name__}")
```

### 第 2 步：Pydantic 风格的模型转 Schema

构建一个最小化的"类转 schema"转换器：定义一个 Python 类，自动生成对应的 JSON Schema。

```python
class SchemaField:
    def __init__(self, field_type, required=True, default=None, enum=None, minimum=None, maximum=None):
        self.field_type = field_type
        self.required = required
        self.default = default
        self.enum = enum
        self.minimum = minimum
        self.maximum = maximum

def python_type_to_schema(field):
    type_map = {
        str: "string",
        int: "integer",
        float: "number",
        bool: "boolean",
    }

    schema = {}

    if field.field_type in type_map:
        schema["type"] = type_map[field.field_type]
    elif field.field_type == list:
        schema["type"] = "array"
        schema["items"] = {"type": "string"}
    elif isinstance(field.field_type, dict):
        schema = field.field_type

    if field.enum:
        schema["enum"] = field.enum
    if field.minimum is not None:
        schema["minimum"] = field.minimum
    if field.maximum is not None:
        schema["maximum"] = field.maximum

    return schema

def model_to_schema(name, fields):
    properties = {}
    required = []

    for field_name, field in fields.items():
        properties[field_name] = python_type_to_schema(field)
        if field.required:
            required.append(field_name)

    return {
        "type": "object",
        "properties": properties,
        "required": required,
    }
```

### 第 3 步：约束 Token 过滤器

模拟约束解码。给定一个不完整的 JSON 字符串和一个 schema，判断当前位置上哪些 token 类别是合法的。

```python
def next_valid_tokens(partial_json, schema):
    stripped = partial_json.strip()

    if not stripped:
        return ["{"]

    try:
        json.loads(stripped)
        return ["<EOS>"]
    except json.JSONDecodeError:
        pass

    last_char = stripped[-1] if stripped else ""

    if last_char == "{":
        return ['"', "}"]
    elif last_char == '"':
        if stripped.endswith('":'):
            return ['"', "0-9", "true", "false", "null", "[", "{"]
        return ["a-z", '"']
    elif last_char == ":":
        return [" ", '"', "0-9", "true", "false", "null", "[", "{"]
    elif last_char == ",":
        return [" ", '"', "{", "["]
    elif last_char in "0123456789":
        return ["0-9", ".", ",", "}", "]"]
    elif last_char == "}":
        return [",", "}", "]", "<EOS>"]
    elif last_char == "]":
        return [",", "}", "<EOS>"]
    elif last_char == "[":
        return ['"', "0-9", "true", "false", "null", "{", "[", "]"]
    else:
        return ["any"]

def demonstrate_constrained_decoding():
    partial_states = [
        '',
        '{',
        '{"product"',
        '{"product":',
        '{"product": "Sony"',
        '{"product": "Sony",',
        '{"product": "Sony", "price":',
        '{"product": "Sony", "price": 348',
        '{"product": "Sony", "price": 348}',
    ]

    print(f"{'Partial JSON':<45} {'Valid Next Tokens'}")
    print("-" * 80)
    for state in partial_states:
        valid = next_valid_tokens(state, {})
        display = state if state else "(empty)"
        print(f"{display:<45} {valid}")
```

### 第 4 步：抽取流水线

把上面所有部分组合成一条抽取流水线：定义 schema，模拟 LLM 生成结构化输出，校验输出，并处理重试。

```python
def simulate_llm_extraction(text, schema, attempt=0):
    if "headphones" in text.lower() or "sony" in text.lower():
        if attempt == 0:
            return '{"product": "Sony WH-1000XM5", "price": 348.00, "in_stock": true, "categories": ["audio", "headphones"]}'
        return '{"product": "Sony WH-1000XM5", "price": 348.00, "in_stock": true}'

    if "laptop" in text.lower():
        return '{"product": "MacBook Pro 16", "price": 2499.00, "in_stock": false, "categories": ["computers"]}'

    return '{"product": "Unknown", "price": 0, "in_stock": false}'

def extract_with_retry(text, schema, max_retries=3):
    for attempt in range(max_retries):
        raw = simulate_llm_extraction(text, schema, attempt)

        try:
            data = json.loads(raw)
        except json.JSONDecodeError as e:
            print(f"  Attempt {attempt + 1}: JSON parse error -- {e}")
            continue

        errors = validate_schema(data, schema)
        if not errors:
            return data

        print(f"  Attempt {attempt + 1}: Schema validation errors -- {errors}")

    return None

product_schema = {
    "type": "object",
    "properties": {
        "product": {"type": "string"},
        "price": {"type": "number", "minimum": 0},
        "in_stock": {"type": "boolean"},
        "categories": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["product", "price", "in_stock"],
}
```

### 第 5 步：运行完整流水线

```python
def run_demo():
    print("=" * 60)
    print("  Structured Output Pipeline Demo")
    print("=" * 60)

    print("\n--- Schema Definition ---")
    product_fields = {
        "product": SchemaField(str),
        "price": SchemaField(float, minimum=0),
        "in_stock": SchemaField(bool),
        "categories": SchemaField(list, required=False),
    }
    generated_schema = model_to_schema("Product", product_fields)
    print(json.dumps(generated_schema, indent=2))

    print("\n--- Schema Validation ---")
    test_cases = [
        ({"product": "Test", "price": 10.0, "in_stock": True}, "Valid object"),
        ({"product": "Test", "price": -5.0, "in_stock": True}, "Negative price"),
        ({"product": "Test", "in_stock": True}, "Missing price"),
        ({"product": "Test", "price": "ten", "in_stock": True}, "String as price"),
        ("not an object", "String instead of object"),
    ]

    for data, label in test_cases:
        errors = validate_schema(data, product_schema)
        status = "PASS" if not errors else f"FAIL: {errors}"
        print(f"  {label}: {status}")

    print("\n--- Constrained Decoding Simulation ---")
    demonstrate_constrained_decoding()

    print("\n--- Extraction Pipeline ---")
    texts = [
        "The Sony WH-1000XM5 headphones are priced at $348 and currently available.",
        "The new MacBook Pro 16-inch laptop costs $2499 but is sold out.",
        "This is a random sentence with no product info.",
    ]

    for text in texts:
        print(f"\n  Input: {text[:60]}...")
        result = extract_with_retry(text, product_schema)
        if result:
            print(f"  Output: {json.dumps(result)}")
        else:
            print(f"  Output: FAILED after retries")
```

## 生产实践

### OpenAI Structured Outputs

```python
# from openai import OpenAI
# from pydantic import BaseModel
#
# client = OpenAI()
#
# class Product(BaseModel):
#     product: str
#     price: float
#     in_stock: bool
#
# response = client.beta.chat.completions.parse(
#     model="gpt-5-mini",
#     messages=[
#         {"role": "system", "content": "Extract product information."},
#         {"role": "user", "content": "Sony WH-1000XM5, $348, in stock"},
#     ],
#     response_format=Product,
# )
#
# product = response.choices[0].message.parsed
# print(product.product, product.price, product.in_stock)
```

OpenAI 的结构化输出模式在内部使用约束解码。模型生成的每一个 token 都保证产出符合 Pydantic schema 的输出。不需要重试，也不需要校验——约束直接内嵌在解码过程中。

### Anthropic Tool Use

```python
# import anthropic
#
# client = anthropic.Anthropic()
#
# response = client.messages.create(
#     model="claude-opus-4-7",
#     max_tokens=1024,
#     tools=[{
#         "name": "extract_product",
#         "description": "Extract product information from text",
#         "input_schema": {
#             "type": "object",
#             "properties": {
#                 "product": {"type": "string"},
#                 "price": {"type": "number"},
#                 "in_stock": {"type": "boolean"},
#             },
#             "required": ["product", "price", "in_stock"],
#         },
#     }],
#     messages=[{"role": "user", "content": "Extract: Sony WH-1000XM5, $348, in stock"}],
# )
```

Anthropic 通过工具使用实现结构化输出：模型发出一个工具调用，其结构化参数与 input_schema 匹配。结果相同，只是 API 接口形式不同。

### Instructor 库

```python
# pip install instructor
# import instructor
# from openai import OpenAI
# from pydantic import BaseModel
#
# client = instructor.from_openai(OpenAI())
#
# class Product(BaseModel):
#     product: str
#     price: float
#     in_stock: bool
#
# product = client.chat.completions.create(
#     model="gpt-5-mini",
#     response_model=Product,
#     messages=[{"role": "user", "content": "Sony WH-1000XM5, $348, in stock"}],
# )
```

Instructor 可以包装任意 LLM 客户端，并加上带校验的自动重试。如果第一次尝试没有通过校验，它会把错误信息作为上下文发回给模型，要求模型修正输出。这适用于任何厂商，不仅限于 OpenAI。

## 交付产物

本课产出 `outputs/prompt-structured-extractor.md`——一个可复用的提示词模板，给定 schema 定义后可以从任意文本中抽取结构化数据。喂给它一个 JSON Schema 和非结构化文本，它就返回校验通过的 JSON。

本课还产出 `outputs/skill-structured-outputs.md`——一个决策框架，帮你根据所用厂商、可靠性要求和 schema 复杂度，选择合适的结构化输出策略。

## 练习

1. 扩展 schema 校验器以支持 `oneOf`（数据必须恰好匹配多个 schema 中的一个）。这能处理多态输出——例如，某个字段既可能是 `Product` 对象也可能是形状不同的 `Service` 对象。

2. 构建一个"schema diff"工具，比较两个 schema 并区分破坏性变更（删除必填字段、修改类型）与非破坏性变更（新增可选字段、放宽约束）。这是在生产环境中对抽取 schema 做版本管理的必备能力。

3. 实现一个更接近真实的约束解码模拟器。给定一个 JSON Schema 和一个包含 100 个 token 的词表（字母、数字、标点、关键字），逐步走一遍生成过程，在每个位置屏蔽非法 token。统计每一步中词表里合法 token 所占的百分比。

4. 构建一个抽取评测套件。创建 50 条产品描述，并人工标注对应的 JSON 输出。在这 50 条上运行你的抽取流水线，测量完全匹配率、字段级准确率和类型合规率。找出哪些字段最难正确抽取。

5. 为你的抽取流水线加上"置信度分数"。对每个抽取出的字段，估计模型的置信程度（基于 token 概率，或者运行 3 次抽取并测量一致性）。把低置信度字段标记出来，交给人工复核。

## 关键术语

| 术语 | 通俗说法 | 实际含义 |
|------|----------------|----------------------|
| JSON 模式（JSON mode） | "返回 JSON" | API 开关，保证输出是语法合法的 JSON，但不强制任何特定 schema |
| 结构化输出（structured output） | "带类型的 JSON" | 匹配特定 JSON Schema 的输出，键名、类型和约束全部正确 |
| 约束解码（constrained decoding） | "引导式生成" | 在每个 token 位置屏蔽掉会产生非法输出的 token——保证 100% 符合 schema |
| JSON Schema | "JSON 模板" | 一种声明式语言，用于描述 JSON 数据的结构、类型和约束（OpenAPI、JSON Forms 等都在使用） |
| Pydantic | "Python dataclasses 加强版" | 定义带类型校验数据模型的 Python 库，被 FastAPI 和 Instructor 用于生成 JSON Schema |
| 函数调用（function calling） | "工具使用" | LLM 输出一个结构化的函数调用（名称 + 类型化参数）而非自由文本——OpenAI 和 Anthropic 都支持 |
| Instructor | "面向 LLM 的 Pydantic" | 包装 LLM 客户端、返回校验通过的 Pydantic 实例的 Python 库，校验失败时自动重试 |
| Token 屏蔽（token masking） | "过滤词表" | 在生成过程中把特定 token 的概率置为零，使模型无法生成它们 |
| Schema 合规（schema compliance） | "形状匹配" | 输出包含所有必填字段、类型正确、取值在约束范围内、没有多余的非法字段 |
| 重试循环（retry loop） | "重试到成功为止" | 把校验错误发回给模型并要求它修正输出——Instructor 自动完成这一过程，重试次数上限可配置 |

## 延伸阅读

- [OpenAI Structured Outputs Guide](https://platform.openai.com/docs/guides/structured-outputs)——OpenAI API 中基于 JSON Schema 的约束解码官方文档
- [Willard & Louf, 2023 -- "Efficient Guided Generation for Large Language Models"](https://arxiv.org/abs/2307.09702)——Outlines 论文，描述如何把 JSON Schema 编译成有限状态机以实现 token 级约束
- [Instructor documentation](https://python.useinstructor.com/)——从任意 LLM 获取结构化输出、带 Pydantic 校验和重试的标准库
- [Anthropic Tool Use Guide](https://docs.anthropic.com/en/docs/tool-use)——Claude 如何通过带 JSON Schema input_schema 的工具使用实现结构化输出
- [JSON Schema specification](https://json-schema.org/)——所有主流结构化输出系统使用的 schema 语言完整规范
- [Outlines library](https://github.com/outlines-dev/outlines)——开源约束生成库，把正则表达式和 JSON Schema 编译为有限状态机
- [Dong et al., "XGrammar: Flexible and Efficient Structured Generation Engine for Large Language Models" (MLSys 2025)](https://arxiv.org/abs/2411.15100)——当前最先进的语法引擎；基于下推自动机的编译方案，token 屏蔽开销约 100 ns / token。
- [Beurer-Kellner et al., "Prompting Is Programming: A Query Language for Large Language Models" (LMQL)](https://arxiv.org/abs/2212.06094)——LMQL 论文，把约束解码表述为一种带类型和取值约束的查询语言。
- [Microsoft Guidance (framework docs)](https://github.com/guidance-ai/guidance)——模板驱动的约束生成框架；与厂商无关，可作为 Outlines 和 XGrammar 的补充。
