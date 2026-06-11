# FIPA-ACL 与言语行为理论的遗产

> 在 MCP 之前、在 A2A 之前，先有 FIPA-ACL。2000 年，IEEE 智能物理代理基金会（Foundation for Intelligent Physical Agents）批准了一种智能体通信语言，包含二十个施为语（performative）、两种内容语言和一组交互协议——合同网（contract net）、订阅/通知（subscribe/notify）、条件请求（request-when）。它最终从工业界淡出，因为本体（ontology）的负担对 Web 来说太重了。但 LLM 驱动的多智能体复兴正在悄悄重新实现同样的思想，只是丢掉了形式语义：JSON 契约充当施为语，自然语言充当本体。这节课认真研读 FIPA-ACL，让你能看清 2026 年的哪些协议决策是重新发明、哪些是真正的新东西，以及当前这波浪潮会在哪里重新遭遇 2000 年代已经解决过的问题。

**Type:** Learn
**Languages:** Python (stdlib)
**Prerequisites:** Phase 16 · 01 (Why Multi-Agent)
**Time:** ~60 minutes

## 问题背景

2026 年的智能体协议生态非常热闹：MCP 负责工具，A2A 负责智能体，ACP 负责企业审计，ANP 负责去中心化信任，NLIP 负责自然语言内容，外加 CA-MCP 和二十多个研究提案。每个规范都宣称自己是奠基性的。

诚实的解读是：它们中的大多数都在重新发现一棵非常具体的、二十年前的决策树。Austin（1962）和 Searle（1969）的言语行为理论（speech-act theory）告诉我们"话语即行动"。KQML（1993）把它变成了线上协议。FIPA-ACL（2000 年批准）则产出了参考标准化成果：二十个施为语、内容语言 SL0/SL1、面向合同网和订阅-通知的交互协议。JADE 和 JACK 是当时的 Java 参考平台。这场运动在 2010 年前后衰落，因为本体开销太重，而 Web 正在胜出。

当你看 MCP 的 `tools/call`、A2A 的任务生命周期或 CA-MCP 的共享上下文存储时，你看到的就是 FIPA 决策的一个更松散、JSON 原生的翻版。了解这段历史能告诉你两件事：哪些新"创新"其实是重新发明，以及新规范将会重新踩到哪些旧的失败模式。

## 核心概念

### 一段话讲清言语行为

Austin 注意到，有些句子并不是在描述世界——而是在改变世界。"我承诺。""我请求。""我宣布。"他称之为施为话语（performative utterance）。Searle 将其形式化为五类：断言类（assertive）、指令类（directive）、承诺类（commissive）、表达类（expressive）、宣告类（declarative）。KQML（Finin 等人，1993）让这套理论在软件智能体上可操作：一条消息 = 一个施为语（动作本身）加上内容（动作所针对的对象）。FIPA-ACL 补上了 KQML 的缺口，并标准化为二十个施为语。

### FIPA 的二十个施为语（部分列表）

| 施为语 | 意图 |
|---|---|
| `inform` | "我告诉你 P 为真" |
| `request` | "我请求你去做 X" |
| `query-if` | "P 是真的吗？" |
| `query-ref` | "X 的值是什么？" |
| `propose` | "我提议我们做 X" |
| `accept-proposal` | "我接受这个提案" |
| `reject-proposal` | "我拒绝这个提案" |
| `agree` | "我同意去做 X" |
| `refuse` | "我拒绝去做 X" |
| `confirm` | "我确认 P 为真" |
| `disconfirm` | "我否认 P" |
| `not-understood` | "你的消息无法解析" |
| `cfp` | "就 X 征集提案" |
| `subscribe` | "X 发生变化时通知我" |
| `cancel` | "取消正在进行的 X" |
| `failure` | "我尝试了 X 但失败了" |

完整列表见 `fipa00037.pdf`（FIPA ACL Message Structure）。重点不在于背下来——重点是其中每一个都对应着某个 LLM 协议最终会重新加回来的原语。

### 典型的 FIPA-ACL 消息

```
(inform
  :sender       agent1@platform
  :receiver     agent2@platform
  :content      "((price IBM 83))"
  :language     SL0
  :ontology     finance
  :protocol     fipa-request
  :conversation-id   conv-42
  :reply-with   msg-17
)
```

七个字段承载协议信封；一个字段（`content`）承载载荷。其余字段正是你每次往 JSON 协议上补加重试、会话串联和本体时都要重新发明的那些东西。

### 两个传世平台

**JADE**（Java Agent DEvelopment framework，1999–2020 年代）是使用最广泛的 FIPA 兼容运行时。智能体继承一个基类，交换 ACL 消息，运行在容器内，并用"行为（behaviors）"进行协调。其交互协议库内置了合同网、订阅-通知、条件请求和提议-接受。

**JACK**（Agent Oriented Software 出品，商业软件）强调在 FIPA 消息之上做 BDI（Belief-Desire-Intention，信念-愿望-意图）推理。更形式化，但采用更少。

随着 Web 技术栈吞噬了多智能体的应用场景，两者都衰落了。MCP 和 A2A 就是 2026 年的运行时"容器"。

### FIPA 为什么衰落

- **本体开销。** FIPA 要求共享本体才能解析 `content`。就本体达成一致是一个耗时数年的标准化过程。而 Web 直接用 HTTP + JSON。
- **没人使用的形式语义。** SL（Semantic Language，语义语言）提供了严格的真值条件，但大多数生产系统使用自由格式的内容，无视这套形式化体系。
- **工具链锁定。** JADE 只支持 Java；JACK 是商业产品。多语言团队绕开了两者。
- **互联网赢得了技术栈之争。** REST、然后是 JSON-RPC、再然后是 gRPC，取代了 ACL 的传输层。

### LLM 复兴就是轻量版 FIPA

把 FIPA 的 `request` 和 MCP 的 `tools/call` 放在一起比较：

```
(request                                {
  :sender  agent1                         "jsonrpc": "2.0",
  :receiver tool-server                   "method":  "tools/call",
  :content "(lookup stock IBM)"           "params":  {"name":"lookup_stock",
  :ontology finance                                   "arguments":{"symbol":"IBM"}},
  :conversation-id c42                    "id": 42
)                                        }
```

同样的信封，不同的语法。两者携带的都是：谁、发给谁、意图、载荷、关联 ID。彼此之间谁也不是革命性突破——它们只是同一个设计上的不同取舍。

Liu 等人 2025 年的综述（"A Survey of Agent Interoperability Protocols: MCP, ACP, A2A, ANP"，arXiv:2505.02279）把这条传承脉络说得很明白：MCP 对应工具使用类言语行为，A2A 对应智能体对等类言语行为，ACP 对应审计追踪类言语行为，ANP 对应去中心化身份扩展。这些新规范就是采用 JSON 语法、语义更松散的 ACL 后裔。

### 把取舍说透

**FIPA 给了你、而现代规范丢掉的东西：**

- 形式语义——你可以证明 `inform` 蕴含发送方相信其内容为真。
- 一份标准的施为语目录——你不必再重新争论"我们要不要加一个 `cancel`？"。
- 数十年积累的交互协议模式——合同网、订阅-通知、提议-接受——并且都有已知的正确性属性。

**现代规范给了你、而 FIPA 没有的东西：**

- 与所有现代工具兼容的 JSON 原生载荷。
- LLM 无需手工编码本体就能理解的自然语言内容。
- Web 技术栈传输（HTTP、SSE、WebSocket）。
- 通过自描述文档实现的能力发现（MCP 的 `listTools`、A2A 的 Agent Card）。

用更松散的意图语义换取更容易的实现。这就是这笔交易的全部内容。

### 值得移植的交互协议

FIPA 提供了约 15 个交互协议。其中三个值得带进 LLM 多智能体系统：

1. **合同网协议（Contract Net Protocol，CNP）。** 管理者发出 `cfp`（征集提案）；投标者以 `propose` 回应；管理者接受/拒绝。这是经典的任务市场模式（Phase 16 · 16 Negotiation）。
2. **订阅/通知（Subscribe/Notify）。** 订阅者发送 `subscribe`；主题一旦变化，发布者就发送 `inform`。这就是 2026 年的每一条事件总线。
3. **条件请求（Request-When）。** "当条件 Y 成立时执行 X。"带前置条件的延迟动作。2026 年的对应物是持久化工作流引擎中的延迟任务（Phase 16 · 22 Production Scaling）。

每一个都能干净地映射到现代消息队列、HTTP + 轮询或 SSE 流式传输上。

### 丢掉本体后会坏掉什么

没有共享本体，智能体就要从自然语言内容中推断含义。2026 年有记录的失败模式是**语义漂移（semantic drift）**：两个智能体用同一个词（`"customer"`）指代有微妙差异的概念，接收方的智能体基于错误的理解采取行动，而任何 schema 校验器都抓不住这个问题。FIPA 的本体要求本可以在解析阶段就拒绝这条消息。

在不引入完整本体的前提下的缓解手段：

- 对 `content` 施加 JSON Schema——在线上协议层就拒绝结构性错误。
- 类型化产物（A2A）——拒绝错误的模态。
- 信封中显式的施为语——即使内容是自然语言，也让意图毫无歧义。

### 2026 年的规范与言语行为传承的对照

| 现代规范 | FIPA 对应物 | 保留了什么 | 丢掉了什么 |
|---|---|---|---|
| MCP `tools/call` | `request` | 显式意图、关联 ID | 形式语义、本体 |
| MCP `resources/read` | `query-ref` | 显式意图、关联 ID | 形式语义 |
| A2A 任务生命周期 | 合同网 + 条件请求 | 异步生命周期、状态转换 | 形式化的完备性保证 |
| A2A 流式事件 | 订阅/通知 | 异步推送 | 类型化谓词订阅 |
| CA-MCP 共享上下文 | 黑板模型（Hayes-Roth 1985） | 多写者共享内存 | 逻辑一致性模型 |
| NLIP | 自然语言内容 | LLM 原生 | schema |

从上到下读这张表，模式是一致的：保留结构性原语，丢掉形式化体系，让 LLM 来掩盖歧义。

## 从零实现

`code/main.py` 实现了一个纯标准库的 FIPA-ACL 转换器。它对典型的 ACL 信封做编码和解码，并展示每一种 MCP / A2A 消息形态如何都可以归约为同样的七个字段。演示内容：

- 把五条 MCP 风格和 A2A 风格的消息编码为 FIPA-ACL。
- 把 FIPA-ACL 解码回现代等价形式。
- 在一个管理者和三个投标者之间，用 `cfp`、`propose`、`accept-proposal`、`reject-proposal` 运行一场玩具版的合同网协商。

运行：

```
python3 code/main.py
```

输出是一份并排对照的轨迹，展示每条现代消息的 2026 年 JSON 形式和 FIPA-ACL 形式，然后是一次合同网投标的往返转换。同样的协议原语在往返转换中完好无损；变的只有语法。

## 生产实践

`outputs/skill-fipa-mapper.md` 是一个技能（skill），它读取任意智能体协议规范并产出与 FIPA-ACL 的映射。在采纳新协议之前先用它来回答："这是真正的新东西，还是套了 JSON 语法的 `inform`？"

## 交付产物

不要把 FIPA-ACL 本身带回来。带回它的检查清单：

- 每条消息的意图原语（施为语）是什么？
- 是否有用于请求-响应和取消操作的关联 ID？
- 是否有显式的内容语言（JSON-RPC、纯文本、结构化的类型化产物）？
- 交互协议是一等公民，还是你正在从头重新实现合同网？
- 当两个智能体对内容含义产生分歧（语义漂移）时会发生什么？

把任何新协议交付到生产环境之前，先把这五个问题记录下来。

## 练习

1. 运行 `code/main.py`。观察往返编码。指出 `tools/call`、`resources/read` 和 A2A 任务创建分别对应哪个 FIPA 施为语。
2. 给合同网演示扩展一个 `cancel` 施为语，让管理者能在投标进行中撤回任务。`cancel` 解决了哪种仅靠重试无法解决的失败场景？
3. 阅读 FIPA ACL Message Structure（http://www.fipa.org/specs/fipa00037/）4.1–4.3 节。挑选一个本课未覆盖的施为语，描述它在现代 JSON-RPC 中的对应物。
4. 阅读 Liu 等人的 arXiv:2505.02279。针对 MCP、A2A、ACP、ANP，分别列出它们保留和丢弃了哪些 FIPA 施为语族。
5. 为你自己系统中 `request` 施为语的 `content` 字段设计一份最小化 JSON Schema。这份 schema 给了你什么纯自然语言给不了的东西？它的代价又是什么？

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|----------------|------------------------|
| 言语行为（Speech act） | "一句会做事的话" | Austin/Searle：话语即行动。ACL 的理论源头。 |
| FIPA | "那个老掉牙的 XML 东西" | IEEE 智能物理代理基金会。2000 年标准化了 ACL。 |
| ACL | "Agent Communication Language" | FIPA 的信封格式：施为语 + 内容 + 元数据。 |
| 施为语（Performative） | "动词" | 消息的意图类别：`inform`、`request`、`propose`、`cfp` 等。 |
| KQML | "FIPA 的前身" | Knowledge Query and Manipulation Language（1993）。更简单，范围更窄。 |
| 本体（Ontology） | "共享词汇表" | 对内容语言所谈论概念的形式化定义。 |
| SL0 / SL1 | "FIPA 的内容语言" | Semantic Language 第 0 级和第 1 级——形式化内容语言家族。 |
| 合同网（Contract Net） | "任务市场" | 管理者发出 cfp；投标者提议；管理者接受。最经典的交互协议。 |
| 交互协议（Interaction protocol） | "消息的模式" | 一组具有已知正确性的施为语序列：条件请求、订阅-通知等。 |

## 延伸阅读

- [Liu et al. — A Survey of Agent Interoperability Protocols: MCP, ACP, A2A, ANP](https://arxiv.org/html/2505.02279v1) —— 2025 年的权威综述，把现代规范与 FIPA 传承联系起来
- [FIPA ACL Message Structure Specification (fipa00037)](http://www.fipa.org/specs/fipa00037/) —— 2000 年批准的信封格式
- [FIPA Communicative Act Library Specification (fipa00037)](http://www.fipa.org/specs/fipa00037/) —— 完整的施为语目录
- [MCP specification 2025-11-25](https://modelcontextprotocol.io/specification/2025-11-25) —— `request`/`query-ref` 的现代工具使用等价物
- [A2A specification](https://a2a-protocol.org/latest/specification/) —— 合同网和订阅-通知的现代智能体对等等价物
