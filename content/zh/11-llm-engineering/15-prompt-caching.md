# 提示词缓存与上下文缓存

> 你的系统提示词有 4,000 个 token，RAG 上下文有 20,000 个 token。每次请求你都要把两者一起发送——也每次都要为它们付费。提示词缓存（prompt caching）让服务商在他们那一侧把这段前缀保持「热」状态，复用时只按正常费率的 10% 计费。用得好，它能把推理成本降低 50–90%，把首 token 延迟降低 40–85%。

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 11 · 01 (Prompt Engineering), Phase 11 · 05 (Context Engineering), Phase 11 · 11 (Caching and Cost)
**Time:** ~60 minutes

## 问题背景

一个编码智能体在对话的每一轮都向 Claude 发送同一段 15,000 token 的系统提示词。按 $3/百万输入 token 计算，二十轮对话仅输入成本就是 $0.90——这还没算用户的实际消息。乘以每天 10,000 个对话，为这段从不改变的文本付出的账单就高达 $9,000/天。

你不能压缩提示词，否则会损害质量。你也不能不发送它——模型每一轮都需要它。唯一的出路，是不再为服务商已经见过的前缀支付全价。

这个出路就是提示词缓存。Anthropic 在 2024 年 8 月推出了这一功能（2025 年又增加了 1 小时扩展 TTL 的变体），OpenAI 在同年晚些时候将其自动化，Google 则随 Gemini 1.5 推出了显式的上下文缓存（context caching），如今三家都在其前沿模型上将它作为一等公民特性提供。

## 核心概念

![Prompt caching: write once, read cheap](../assets/prompt-caching.svg)

**工作机制。** 当一个请求的前缀与近期某个请求的前缀匹配时，服务商会直接复用上一次运行的 KV 缓存，而不是重新编码这些 token。第一次你支付少量的写入溢价，之后每一次都享受大幅的读取折扣。

**2026 年三家服务商的三种风格。**

| 服务商 | API 风格 | 命中折扣 | 写入溢价 | 默认 TTL | 最小可缓存长度 |
|---------|-----------|--------------|---------------|-------------|---------------|
| Anthropic | 在内容块上显式标注 `cache_control` | 输入价 1 折 | 加收 25% | 5 分钟（可扩展至 1 小时） | 1,024 token（Sonnet/Opus），2,048（Haiku） |
| OpenAI | 自动前缀检测 | 输入价 5 折 | 无 | 最长 1 小时（尽力而为） | 1,024 token |
| Google (Gemini) | 显式的 `CachedContent` API | 按存储计费；读取约为正常价的 25% | 按 token·小时收存储费 | 用户自定（默认 1 小时） | 4,096 token（Flash），32,768（Pro） |

**不变式。** 三家都只缓存前缀。只要两次请求之间有任何一个 token 不同，从第一个不同的 token 开始往后全部失效。把*稳定*的部分放在顶部，把*可变*的部分放在底部。

### 对缓存友好的布局

```
[system prompt]          <-- cache this
[tool definitions]       <-- cache this
[few-shot examples]      <-- cache this
[retrieved documents]    <-- cache if reused, else don't
[conversation history]   <-- cache up to last turn
[current user message]   <-- never cache (different every time)
```

一旦违反这个顺序——比如把用户消息放在系统提示词之上，或在 few-shot 示例之间穿插动态检索结果——缓存就永远不会命中。

### 盈亏平衡计算

Anthropic 的 25% 写入溢价意味着一个缓存块至少要被读取两次才能净省钱。1 次写入 + 1 次读取，平均每请求成本为 0.675 倍（节省 32%）；1 次写入 + 10 次读取，平均为 0.205 倍（节省 80%）。经验法则：凡是预期在 TTL 内复用至少 3 次的内容，都值得缓存。

## 从零实现

### 第 1 步：用显式标记实现 Anthropic 提示词缓存

```python
import anthropic

client = anthropic.Anthropic()

SYSTEM = [
    {
        "type": "text",
        "text": "You are a senior Python reviewer. Follow the rubric exactly.\n\n" + RUBRIC_15K_TOKENS,
        "cache_control": {"type": "ephemeral"},
    }
]

def review(code: str):
    return client.messages.create(
        model="claude-opus-4-7",
        max_tokens=1024,
        system=SYSTEM,
        messages=[{"role": "user", "content": code}],
    )
```

`cache_control` 标记告诉 Anthropic 把这个块存储 5 分钟。在窗口期内复用即命中；过期后复用则重新写入。

**响应中的 usage 字段：**

```python
response = review(code_a)
response.usage
# InputTokensUsage(
#     input_tokens=120,
#     cache_creation_input_tokens=15023,   # paid at 1.25x
#     cache_read_input_tokens=0,
#     output_tokens=340,
# )

response_b = review(code_b)
response_b.usage
# cache_creation_input_tokens=0
# cache_read_input_tokens=15023           # paid at 0.1x
```

在 CI 中同时检查这两个字段——如果跨请求 `cache_read_input_tokens` 始终为零，说明你的缓存键在漂移。

### 第 2 步：1 小时扩展 TTL

对于长时间运行的批处理任务，5 分钟的默认 TTL 会在任务间隔中过期。设置 `ttl`：

```python
{"type": "text", "text": RUBRIC, "cache_control": {"type": "ephemeral", "ttl": "1h"}}
```

1 小时 TTL 的写入溢价是默认的 2 倍（在基准价上加收 50% 而非 25%），但只要批处理任务对该前缀的复用超过 5 次，就能很快回本。

### 第 3 步：OpenAI 自动缓存

OpenAI 不需要任何配置。任何超过 1,024 token、且与近期请求匹配的前缀会自动获得 50% 折扣。

```python
from openai import OpenAI
client = OpenAI()

resp = client.chat.completions.create(
    model="gpt-5",
    messages=[
        {"role": "system", "content": SYSTEM_PROMPT},   # long and stable
        {"role": "user", "content": user_msg},
    ],
)
resp.usage.prompt_tokens_details.cached_tokens  # the discounted portion
```

缓存友好布局规则同样适用。有两件事会破坏 OpenAI 的缓存而不影响 Anthropic：修改 `user` 字段（它被用作缓存键的组成部分），以及调整工具顺序。

### 第 4 步：Gemini 显式上下文缓存

Gemini 把缓存当作一个由你创建并命名的一等对象：

```python
from google import genai
from google.genai import types

client = genai.Client()

cache = client.caches.create(
    model="gemini-3-pro",
    config=types.CreateCachedContentConfig(
        display_name="rubric-v3",
        system_instruction=RUBRIC,
        contents=[FEW_SHOT_EXAMPLES],
        ttl="3600s",
    ),
)

resp = client.models.generate_content(
    model="gemini-3-pro",
    contents=["Review this code:\n" + code],
    config=types.GenerateContentConfig(cached_content=cache.name),
)
```

Gemini 在缓存存活期间按 token·小时收取存储费，读取价约为正常输入费率的 25%。当你需要跨多个会话、连续数天复用同一个巨型提示词时，这种形态最合适。

### 第 5 步：在生产环境中测量命中率

参见 `code/main.py`，它实现了一个模拟的三服务商成本核算器，跟踪写入/读取/未命中次数，并计算每 1K 请求的混合成本。用目标命中率作为部署的闸门——大多数生产环境的 Anthropic 配置在预热后读取占比应超过 80%。

## 2026 年仍在上线的常见陷阱

- **顶部的动态时间戳。** 把 `"Current time: 2026-04-22 15:30:02"` 放在系统提示词顶部，每个请求都会未命中。把时间戳移到缓存断点之下。
- **工具顺序变动。** 用稳定的顺序序列化工具——部署之间字典顺序的一次重排就会让所有命中失效。
- **自由文本的近似重复。** "You are helpful." 和 "You are a helpful assistant."——一个字节的差异 = 完全未命中。
- **缓存块太小。** Anthropic 强制 1,024 token 的下限（Haiku 为 2,048）。更小的块会静默地不被缓存。
- **粗放的成本看板。** 把「输入 token」拆分为已缓存和未缓存两类。否则一次流量下降看起来会像是缓存的功劳。

## 生产实践

2026 年的缓存技术选型：

| 场景 | 选择 |
|-----------|------|
| 拥有稳定的 10k+ 系统提示词、多轮交互的智能体 | Anthropic `cache_control`，5 分钟 TTL |
| 复用同一前缀超过 30 分钟的批处理任务 | Anthropic，`ttl: "1h"` |
| 跑在 GPT-5 上的 Serverless 端点，没有自建基础设施 | OpenAI 自动缓存（只需让前缀稳定且足够长） |
| 跨多天复用巨型代码/文档语料 | Gemini 显式 `CachedContent` |
| 跨服务商容灾切换 | 在各服务商间保持可缓存前缀布局完全一致，任意一边都能命中 |

将它与语义缓存（Phase 11 · 11）结合用于用户消息层：提示词缓存处理 *token 完全相同*的复用，语义缓存处理*语义相同*的复用。

## 交付产物

保存 `outputs/skill-prompt-caching-planner.md`：

```markdown
---
name: prompt-caching-planner
description: Design a cache-friendly prompt layout and pick the right provider caching mode.
version: 1.0.0
phase: 11
lesson: 15
tags: [llm-engineering, caching, cost]
---

Given a prompt (system + tools + few-shot + retrieval + history + user) and a usage profile (requests per hour, TTL needed, provider), output:

1. Layout. Reordered sections with a single cache breakpoint marked; explain which sections are stable, which are volatile.
2. Provider mode. Anthropic cache_control, OpenAI automatic, or Gemini CachedContent. Justify from TTL and reuse pattern.
3. Break-even. Expected reads per write within TTL; net cost vs no-cache with math.
4. Verification plan. CI assertion that cache_read_input_tokens > 0 on the second identical request; dashboard split by cached vs uncached tokens.
5. Failure modes. List the three most likely reasons the cache will miss in this setup (dynamic timestamp, tool reorder, near-duplicate text) and how you will prevent each.

Refuse to ship a cache plan that places a dynamic field above the breakpoint. Refuse to enable 1h TTL without a reuse count that makes the 2x write premium pay back.
```

## 练习

1. **简单。** 用一段 5,000 token 的系统提示词对 Claude 跑一个 10 轮对话，分别在不带 `cache_control` 和带 `cache_control` 的情况下运行，报告两种情况下的输入 token 账单。
2. **中等。** 编写一个测试工具：给定一个提示词模板和一份请求日志，计算各服务商（Anthropic 5 分钟、Anthropic 1 小时、OpenAI 自动、Gemini 显式）的预期命中率和美元节省额。
3. **困难。** 构建一个布局优化器：给定一个提示词和一组标注了 `stable=True/False` 的字段，在不丢失信息的前提下重写提示词，把单一缓存断点放到对缓存最友好的位置。在真实的 Anthropic 端点上验证。

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|-----------------|-----------------------|
| 提示词缓存（Prompt caching） | 「让长提示词变便宜」 | 对匹配的前缀复用服务商侧的 KV 缓存；重复输入 token 享受 50-90% 折扣。 |
| `cache_control` | 「Anthropic 的那个标记」 | 内容块上的属性，声明「到此为止的内容都可缓存」；`{"type": "ephemeral"}`。 |
| 缓存写入（Cache write） | 「付溢价」 | 填充缓存的第一个请求；Anthropic 按约 1.25 倍输入费率计费，OpenAI 免费。 |
| 缓存读取（Cache read） | 「拿折扣」 | 后续匹配前缀的请求；按 10%（Anthropic）、50%（OpenAI）、约 25%（Gemini）计费。 |
| TTL | 「缓存能活多久」 | 缓存保持热状态的秒数；Anthropic 默认 5 分钟（可扩展至 1 小时），OpenAI 尽力而为最长 1 小时，Gemini 用户自定。 |
| 扩展 TTL（Extended TTL） | 「Anthropic 的 1 小时缓存」 | `{"type": "ephemeral", "ttl": "1h"}`；写入溢价翻倍，但对批处理复用很划算。 |
| 前缀匹配（Prefix match） | 「我的缓存为什么没命中」 | 只有从开头到断点的每一个 token 都逐字节相同时缓存才会命中。 |
| 上下文缓存（Context caching，Gemini） | 「显式的那种」 | Google 的命名式、按存储计费的缓存对象；最适合跨多天复用大型语料。 |

## 延伸阅读

- [Anthropic — Prompt caching](https://docs.anthropic.com/en/docs/build-with-claude/prompt-caching) — `cache_control`、1 小时 TTL、盈亏平衡表。
- [OpenAI — Prompt caching](https://platform.openai.com/docs/guides/prompt-caching) — 自动前缀匹配。
- [Google — Context caching](https://ai.google.dev/gemini-api/docs/caching) — `CachedContent` API 与存储定价。
- [Anthropic engineering — Prompt caching for long-context workloads](https://www.anthropic.com/news/prompt-caching) — 包含延迟数据的最初发布文章。
- Phase 11 · 05（Context Engineering）— 如何切分提示词，让缓存有落点。
- Phase 11 · 11（Caching and Cost）— 把提示词缓存与作用于用户消息的语义缓存搭配使用。
- [Pope et al., "Efficiently Scaling Transformer Inference" (2022)](https://arxiv.org/abs/2211.05102) — 提示词缓存向用户暴露的 KV 缓存内存模型；解释了为什么重读一段已缓存的前缀比重新计算便宜约 10 倍。
- [Agrawal et al., "SARATHI: Efficient LLM Inference by Piggybacking Decodes with Chunked Prefills" (2023)](https://arxiv.org/abs/2308.16369) — 预填充（prefill）正是提示词缓存所跳过的阶段；这篇论文解释了为什么缓存命中时 TTFT 大幅下降而 TPOT 不受影响。
- [Leviathan et al., "Fast Inference from Transformers via Speculative Decoding" (2023)](https://arxiv.org/abs/2211.17192) — 提示词缓存与推测解码、Flash Attention、MQA/GQA 同属压低推理成本曲线的几大杠杆；想了解其余三个，读这篇。
