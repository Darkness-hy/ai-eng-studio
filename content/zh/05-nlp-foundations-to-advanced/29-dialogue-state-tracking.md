# 对话状态跟踪

> 「我想在北区找一家便宜的餐厅……还是改成中等价位吧……再加上意大利菜。」三轮对话，三次状态更新。对话状态跟踪（DST）负责让槽位-值字典始终保持同步，预订才能成功。

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 5 · 17 (Chatbots), Phase 5 · 20 (Structured Outputs)
**Time:** ~75 minutes

## 问题背景

在任务型对话系统中，用户的目标被编码为一组槽位-值对（slot-value pairs）：`{cuisine: italian, area: north, price: moderate}`。用户的每一轮发言都可能新增、修改或删除某个槽位。系统必须读取完整对话，并正确输出当前状态。

只要弄错一个槽位，系统就会订错餐厅、排错航班、扣错银行卡。DST 是连接「用户说了什么」与「后端执行什么」之间的枢纽。

为什么在 2026 年、有了 LLM 之后它依然重要：

- 合规敏感领域（银行、医疗、机票预订）要求确定性的槽位值，而不是自由生成的文本。
- 工具调用智能体在调用 API 之前，仍然需要先完成槽位解析。
- 多轮纠正比看起来更难：「不对，还是改到周四吧。」

现代流水线是：经典 DST 概念 + LLM 抽取器 + 结构化输出护栏。

## 核心概念

![DST: dialog history → slot-value state](../assets/dst.svg)

**任务结构。** 由一个模式（schema）定义领域（餐厅、酒店、出租车）及各自的槽位（菜系、区域、价位、人数）。每个槽位可以为空、取自封闭集合中的值（price: {cheap, moderate, expensive}），或者是自由文本值（name: "The Copper Kettle"）。

**DST 的两种形式化方式。**

- **分类式。** 对每个（slot, candidate_value）对预测是/否。适用于封闭词表槽位。2020 年之前的标准做法。
- **生成式。** 给定对话，以自由文本形式生成槽位值。适用于开放词表槽位。当下的默认方案。

**评估指标。** 联合目标准确率（Joint Goal Accuracy, JGA）——*所有*槽位都正确的轮次占比。全对或全错。2026 年 MultiWOZ 2.4 榜单的最高水平在 83% 左右。

**架构演进。**

1. **基于规则（槽位正则 + 关键词）。** 在窄领域中是强基线。便于调试。
2. **TripPy / BERT-DST。** 基于 BERT 编码的拷贝式生成。LLM 之前的标准方案。
3. **LDST（LLaMA + LoRA）。** 经指令微调的 LLM，配合领域-槽位提示。在 MultiWOZ 2.4 上达到 ChatGPT 级别的质量。
4. **无本体方案（2024–26）。** 跳过模式定义，直接生成槽位名和槽位值。可处理开放领域。
5. **提示词 + 结构化输出（2024–26）。** LLM 配合 Pydantic 模式 + 约束解码。5 行代码，可直接上生产。

### 经典失败模式

- **跨轮指代。** 「我们还是选第一个吧。」需要解析「第一个」指的是哪个选项。
- **覆盖还是追加。** 用户说「加上意大利菜」。你应该替换 cuisine 还是追加？
- **隐式确认。** 「行，挺好」——这算是接受了刚才提供的预订吗？
- **纠正。** 「还是改成晚上 7 点吧。」必须更新时间，同时不能清空其他槽位。
- **对系统上一句话的指代。** 「对，就那个。」「那个」是哪个？

## 从零实现

### 第 1 步：基于规则的槽位抽取器

参见 `code/main.py`。正则 + 同义词词典就能覆盖窄领域中 70% 的标准表述：

```python
CUISINE_SYNONYMS = {
    "italian": ["italian", "pasta", "pizza", "italy"],
    "chinese": ["chinese", "chow mein", "noodles"],
}


def extract_cuisine(utterance):
    for canonical, synonyms in CUISINE_SYNONYMS.items():
        if any(syn in utterance.lower() for syn in synonyms):
            return canonical
    return None
```

一旦超出标准词表就很脆弱。但用于确定性的槽位确认场景是可行的。

### 第 2 步：状态更新循环

```python
def update_state(state, utterance):
    new_state = dict(state)
    for slot, extractor in SLOT_EXTRACTORS.items():
        value = extractor(utterance)
        if value is not None:
            new_state[slot] = value
    for slot in NEGATION_CLEARS:
        if is_negated(utterance, slot):
            new_state[slot] = None
    return new_state
```

三条不变式：

- 用户没有提到的槽位，绝不重置。
- 显式否定（「菜系不用管了」）必须清空对应槽位。
- 用户纠正（「还是……吧」）必须覆盖，而不是追加。

### 第 3 步：LLM 驱动的 DST 与结构化输出

```python
from pydantic import BaseModel
from typing import Literal, Optional
import instructor

class RestaurantState(BaseModel):
    cuisine: Optional[Literal["italian", "chinese", "indian", "thai", "any"]] = None
    area: Optional[Literal["north", "south", "east", "west", "center"]] = None
    price: Optional[Literal["cheap", "moderate", "expensive"]] = None
    people: Optional[int] = None
    day: Optional[str] = None


def llm_dst(history, llm):
    prompt = f"""You track the slot values of a restaurant booking across turns.
Dialogue so far:
{render(history)}

Update the state based on the latest user turn. Output only the JSON state."""
    return llm(prompt, response_model=RestaurantState)
```

Instructor + Pydantic 保证输出的状态对象一定合法。不需要正则，不会出现模式不匹配，也不会幻觉出不存在的槽位。

### 第 4 步：JGA 评估

```python
def joint_goal_accuracy(predicted_states, gold_states):
    correct = sum(1 for p, g in zip(predicted_states, gold_states) if p == g)
    return correct / len(predicted_states)
```

校准基准：系统在多大比例的轮次中把所有槽位全部答对？在 MultiWOZ 2.4 上，2026 年顶尖系统是 80-83%。你的领域内系统在自己的窄词表上应该超过这个水平，否则直接用 LLM 基线就能赢过你。

### 第 5 步：处理纠正

```python
CORRECTION_CUES = {"actually", "no wait", "on second thought", "change that to"}


def is_correction(utterance):
    return any(cue in utterance.lower() for cue in CORRECTION_CUES)
```

检测到纠正时，应覆盖最近更新的槽位，而不是追加。没有 LLM 帮助时这一点很难做对。现代做法是：始终让 LLM 根据完整历史重新生成整个状态，而不是增量更新——这样纠正问题就被自然消化了。

## 常见陷阱

- **全历史重生成的成本。** 让 LLM 每轮重新生成状态，总 token 开销是 O(n²)。要限制历史长度，或对较早的轮次做摘要。
- **模式漂移。** 事后新增槽位会让旧训练数据失效。要给模式打版本号。
- **大小写敏感。** "Italian"、"italian"、"ITALIAN"——所有环节都要做归一化。
- **隐式继承。** 如果用户之前说过「4 个人」，之后改时间的新请求不应清空人数槽位。永远传入完整历史。
- **自由文本与封闭集合。** 名称、时间、地址需要自由文本槽位；菜系和区域是封闭集合。模式中两者要混合使用。

## 生产实践

2026 年的技术选型：

| 场景 | 方案 |
|-----------|----------|
| 窄领域（一两个意图） | 基于规则 + 正则 |
| 宽领域、有标注数据 | LDST（在 MultiWOZ 风格数据上做 LLaMA + LoRA） |
| 宽领域、无标注、需要直接上生产 | LLM + Instructor + Pydantic 模式 |
| 语音 / 口语 | ASR + 归一化器 + LLM-DST |
| 多领域预订流程 | 模式引导的 LLM，每个领域一个 Pydantic 模型 |
| 合规敏感 | 规则为主，LLM 兜底并附带确认流程 |

## 交付产物

保存为 `outputs/skill-dst-designer.md`：

```markdown
---
name: dst-designer
description: Design a dialogue state tracker — schema, extractor, update policy, evaluation.
version: 1.0.0
phase: 5
lesson: 29
tags: [nlp, dialogue, task-oriented]
---

Given a use case (domain, languages, vocab openness, compliance needs), output:

1. Schema. Domain list, slots per domain, open vs closed vocabulary per slot.
2. Extractor. Rule-based / seq2seq / LLM-with-Pydantic. Reason.
3. Update policy. Regenerate-whole-state / incremental; correction handling; negation handling.
4. Evaluation. Joint Goal Accuracy on a held-out dialogue set, slot-level precision/recall, confusion on the hardest slot.
5. Confirmation flow. When to explicitly ask the user to confirm (destructive actions, low-confidence extractions).

Refuse LLM-only DST for compliance-sensitive slots without a rule-based secondary check. Refuse any DST that cannot roll back a slot on user correction. Flag schemas without version tags.
```

## 练习

1. **简单。** 在 `code/main.py` 中实现基于规则的状态跟踪器，支持 3 个槽位（cuisine、area、price）。在 10 段手工编写的对话上测试。测量 JGA。
2. **中等。** 用 Instructor + Pydantic + 一个小型 LLM 跑同一数据集。对比 JGA。逐一检查最难的轮次。
3. **困难。** 两者都实现并做路由：规则为主，当规则抽取出的高置信度槽位少于 2 个时回退到 LLM。测量组合后的 JGA 和每轮推理成本。

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|-----------------|-----------------------|
| DST | 对话状态跟踪 | 跨对话轮次维护槽位-值字典。 |
| Slot（槽位） | 用户意图的单元 | 后端所需的命名参数（菜系、日期）。 |
| Domain（领域） | 任务范围 | 餐厅、酒店、出租车——各自对应一组槽位。 |
| JGA | 联合目标准确率 | 所有槽位都正确的轮次占比。全对或全错。 |
| MultiWOZ | 那个基准 | 多领域 WOZ 数据集；DST 评估的标准。 |
| 无本体 DST | 不要模式 | 直接生成槽位名和槽位值，没有固定列表。 |
| Correction（纠正） | 「还是……吧」 | 覆盖此前已填槽位的对话轮次。 |

## 延伸阅读

- [Budzianowski et al. (2018). MultiWOZ — A Large-Scale Multi-Domain Wizard-of-Oz](https://arxiv.org/abs/1810.00278) —— 公认的标准基准。
- [Feng et al. (2023). Towards LLM-driven Dialogue State Tracking (LDST)](https://arxiv.org/abs/2310.14970) —— 用 LLaMA + LoRA 指令微调做 DST。
- [Heck et al. (2020). TripPy — A Triple Copy Strategy for Value Independent Neural Dialog State Tracking](https://arxiv.org/abs/2005.02877) —— 拷贝式 DST 的主力方案。
- [King, Flanigan (2024). Unsupervised End-to-End Task-Oriented Dialogue with LLMs](https://arxiv.org/abs/2404.10753) —— 基于 EM 的无监督任务型对话。
- [MultiWOZ leaderboard](https://github.com/budzianowski/multiwoz) —— 权威的 DST 结果汇总。
