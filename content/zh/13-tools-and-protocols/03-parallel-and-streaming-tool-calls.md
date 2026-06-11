# 并行工具调用与工具流式输出

> 三次相互独立的天气查询如果串行执行，就是三个完整的往返。改成并行后，总耗时会坍缩到最慢的那一次调用。如今所有前沿模型提供商都支持在单轮中发出多个工具调用。收益是实打实的，但底层管线很微妙。本课讲透这两个部分：并行扇出（fan-out）与流式参数的重组，并重点剖析 id 关联这个陷阱。

**Type:** Build
**Languages:** Python (stdlib, thread pool + streaming harness)
**Prerequisites:** Phase 13 · 02 (function calling deep dive)
**Time:** ~75 minutes

## 学习目标

- 解释 `parallel_tool_calls: true` 为何存在，以及何时应该禁用它。
- 在并行扇出过程中，把流式到达的参数分片正确关联到对应的工具调用 id。
- 把不完整的 `arguments` 字符串重组成完整 JSON，而不过早解析。
- 运行一个三城市天气基准测试，直观展示串行与并行的延迟差异。

## 问题背景

没有并行调用时，一个智能体回答「班加罗尔、东京、苏黎世的天气如何」的流程是这样的：

```
user -> LLM
LLM -> call get_weather(Bengaluru)
host -> run executor, reply with result
LLM -> call get_weather(Tokyo)
host -> run executor, reply with result
LLM -> call get_weather(Zurich)
host -> run executor, reply with result
LLM -> final text answer
```

三次 LLM 往返，每次还要额外付出执行器的延迟。墙钟时间大约是理想情况的 4 倍。

启用并行调用后：

```
user -> LLM
LLM -> call get_weather(Bengaluru); call get_weather(Tokyo); call get_weather(Zurich)
host -> run all three executors concurrently, reply with three results
LLM -> final text answer
```

只有一次 LLM 往返。执行器耗时取三者的最大值，而不是总和。在 OpenAI、Anthropic 和 Gemini 上的生产环境基准测试显示，扇出类工作负载的墙钟时间可减少 60% 到 70%。

代价是关联的复杂性。当三个调用以乱序完成时，你返回的结果必须携带匹配的 `tool_call_id`，模型才能把它们对上号。当结果以流式到达时，你必须先把不完整的参数片段重组成完整 JSON，才能执行。Gemini 3 引入唯一 id，部分原因就是为了解决一个真实问题：对同一工具的两个并行调用此前无法区分。

## 核心概念

### 启用并行

- **OpenAI。** `parallel_tool_calls: true` 默认开启。设为 `false` 可强制串行。
- **Anthropic。** 通过 `disable_parallel_tool_use: false` 启用并行（Claude 3.5 及更高版本默认开启）。设为 `true` 则串行。
- **Gemini。** 始终具备并行能力；`tool_config.function_calling_config.mode = "AUTO"` 让模型自行决定。

以下情况应禁用并行：工具之间存在顺序依赖（先 `create_file` 再 `write_file`）、某个调用的输出是另一个调用的输入、或者限流器扛不住扇出。

### Id 关联

模型发出的每个调用都带有一个 `id`。宿主返回的每个结果都必须包含同一个 id。否则，结果之间就无法区分。

- **OpenAI。** 每条 tool 角色消息上的 `tool_call_id`。
- **Anthropic。** 每个 `tool_result` 块上的 `tool_use_id`。
- **Gemini。** 每个 `functionResponse` 上的 `id`（Gemini 3 及更高版本；Gemini 2 按名称匹配，遇到同名并行调用就会出错）。

### 并发执行调用

宿主把每个调用的执行器放在独立的线程、协程或远程 worker 上运行。最简单的实现用线程池；生产环境用 asyncio 配合 `asyncio.gather` 或结构化并发。完成顺序不可预测——id 才是唯一的标识。

一个常见 bug：按调用列表顺序而非完成顺序回复结果。这通常也能正常工作，因为模型只关心 `tool_call_id`，但一旦有结果被丢弃或重复，乱序提交会让调试变得更困难。建议按完成顺序回复，并显式带上 id。

### 流式工具调用

当模型以流式输出时，`arguments` 是分片到达的。三个并行调用对应的三条分片流会在线路上交错。你需要为每个 id 维护一个累积器（accumulator）。

各提供商的格式：

- **OpenAI。** 每个分片是 `choices[0].delta.tool_calls[i].function.arguments`（不完整字符串）。分片携带 `index`（在调用列表中的位置）。按 index 累积，在 `id` 首次出现时读取它，当 `finish_reason = "tool_calls"` 时再解析 JSON。
- **Anthropic。** 流事件依次为 `message_start`，然后每个 `tool_use` 类型的块对应一个 `content_block_start`（包含 id、name 和空的 input）。`content_block_delta` 事件携带 `input_json_delta` 分片。`content_block_stop` 关闭对应的块。
- **Gemini。** `streamFunctionCallArguments`（Gemini 3 及更高版本）发出的分片带有 `functionCallId`，因此多个调用可以干净地交错。Gemini 3 之前，流式输出一次只返回一个完整调用。

### 不完整 JSON 与过早解析陷阱

在 `arguments` 完整之前不能解析。像 `{"city": "Beng` 这样的不完整 JSON 是非法的，解析会直接抛异常。正确的判断依据是提供商的调用结束信号：OpenAI 的 `finish_reason = "tool_calls"`、Anthropic 的 `content_block_stop`、或 Gemini 的流结束事件。只有此时才尝试 `json.loads`。更健壮的做法是使用增量 JSON 解析器，它在结构逐步完整时持续产出事件；OpenAI 的流式指南推荐用这种方式实现实时「思考中」指示器的 UX。数大括号不是可靠的完整性检测手段（引号字符串内的大括号或转义内容会造成误判），只能当作非正式的调试启发式。

### 乱序完成

```
call_A: fast API, returns first
call_B: slow API, returns second
call_C: median API, returns third
```

宿主的回复仍必须引用这些 id：

```
[{role: "tool", tool_call_id: "call_A", content: ...},
 {role: "tool", tool_call_id: "call_B", content: ...},
 {role: "tool", tool_call_id: "call_C", content: ...}]
```

在 OpenAI 或 Anthropic 上，回复中的顺序不影响正确性。Gemini 也接受任意顺序，只要 id 匹配。

### 基准测试：串行 vs 并行

`code/main.py` 中的测试框架模拟了三个延迟分别为 400、600、800 毫秒的执行器。串行总共耗时 1800 毫秒。并行耗时 max(400, 600, 800) = 800 毫秒。差值是常数级的，不是按比例缩放的，因此工具数量越多，节省越可观。

现实中的注意事项：并行调用会给下游 API 带来压力。对一个被限流的服务做 10 路扇出必然失败。Phase 13 · 17 讲网关层背压；重试语义计划在后续阶段中展开。

### 流式扇出的墙钟时间

如果模型本身在流式输出，那么只要某个调用的参数一完整，就可以立刻开始执行，而不必等所有调用都最终确定。这是 OpenAI 文档记载的一项优化，但并非所有 SDK 都暴露了它。本课的测试框架实现了这一点：模拟流一产出完整的参数对象，宿主就启动对应的调用。

## 生产实践

`code/main.py` 分为两部分。第一部分用 `concurrent.futures.ThreadPoolExecutor` 分别以串行和并行方式运行三个模拟天气调用，并打印墙钟时间。第二部分回放一个伪造的流式响应——三个并行调用的 `arguments` 分片在同一条流上交错——并用 `StreamAccumulator` 按 id 重组。没有 LLM，没有网络，只有重组逻辑本身。

值得关注的点：

- 串行计时器停在 1.8 秒。同样的模拟延迟下，并行计时器停在 0.8 秒。
- 累积器通过按 id 缓冲来处理乱序到达的分片，只在每个调用的 JSON 完整后才解析。
- 某个 id 的参数一旦定稿，执行器立即启动，而不是等所有流结束。

## 交付产物

本课产出 `outputs/skill-parallel-call-safety-check.md`。给定一个工具注册表，该 skill 会审计哪些工具可以安全并行、哪些存在顺序依赖、哪些会压垮下游限流——并返回一份带有逐工具 `parallel_safe` 标志的修订版注册表。

## 练习

1. 运行 `code/main.py` 并调整模拟延迟。确认并行与串行的耗时比近似为 `max/sum`（由于线程调度、序列化和框架开销，真实运行会与理想值略有偏差）。在什么样的延迟分布下，并行就不再有意义了？

2. 扩展累积器，处理「调用在流中途被取消」的情况：丢弃其缓冲并发出一个 `cancelled` 事件。哪家提供商对这种情况有明确的文档说明？查阅 Anthropic 的 `content_block_stop` 语义和 OpenAI 的 `finish_reason: "length"` 行为。

3. 用 `asyncio.gather` 替换线程池。对两者做基准测试。由于上下文切换成本更低，异步版本应该会有小幅优势，但前提是执行器在做真实 I/O。

4. 挑选两个不应该并行的工具（例如先 `create_file` 再 `write_file`）。给注册表添加一个 `ordering_dependency` 依赖图，并基于该图对并行扇出加以约束。这是依赖感知调度的最小机制，后续的智能体工程阶段会将其形式化。

5. 阅读 OpenAI 的 parallel-function-calling 章节和 Anthropic 的 `disable_parallel_tool_use` 文档。找出 Anthropic 建议禁用并行的那一类真实工具。（提示：对同一资源的有后果的变更操作。）

## 关键术语

| 术语 | 通俗说法 | 实际含义 |
|------|----------------|------------------------|
| 并行工具调用 | 「单轮扇出」 | 模型在单条 assistant 消息中发出多个工具调用 |
| `parallel_tool_calls` | 「OpenAI 的开关」 | 启用或禁用多调用输出 |
| `disable_parallel_tool_use` | 「Anthropic 的反向开关」 | 选择性关闭的标志；默认启用并行 |
| 工具调用 id | 「关联句柄」 | 每个调用的标识符，结果消息必须原样回传 |
| 累积器 | 「流缓冲区」 | 按 id 缓冲不完整 `arguments` 分片的字符串缓冲区 |
| 乱序完成 | 「快的先到」 | 并行调用以不可预测的顺序完成；id 是粘合剂 |
| 依赖图 | 「顺序约束」 | 输出会作为其他工具输入的工具；不能并行 |
| 过早解析陷阱 | 「JSON.parse 炸了」 | 试图解析一个不完整的 `arguments` 字符串 |
| `streamFunctionCallArguments` | 「Gemini 3 的特性」 | 带有每调用唯一 id 的流式参数分片 |
| 按完成顺序回复 | 「别等所有都完成」 | 结果一到就按 id 回复，无需等待全部完成 |

## 延伸阅读

- [OpenAI — Parallel function calling](https://platform.openai.com/docs/guides/function-calling#parallel-function-calling) — 默认行为与关闭开关
- [Anthropic — Tool use: implementing tool use](https://docs.anthropic.com/en/docs/agents-and-tools/tool-use/implementing-tool-use) — `disable_parallel_tool_use` 与结果批量返回
- [Google — Gemini function calling parallel section](https://ai.google.dev/gemini-api/docs/function-calling) — Gemini 3 起支持的 id 关联并行调用
- [OpenAI — Streaming responses with tools](https://platform.openai.com/docs/api-reference/responses-streaming) — OpenAI 流式输出的参数分片重组
- [Anthropic — Streaming messages](https://docs.anthropic.com/en/api/messages-streaming) — `content_block_delta` 与 `input_json_delta`
