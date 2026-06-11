# 提示词缓存与语义缓存的经济学

> **定价快照截至 2026-04。** 下文中的数字均来自本课发布时采集的厂商价目表；在向下游引用之前，请对照所附文档链接核实最新数据。

> 缓存发生在两个层面。L2（提供商层）提示词/前缀缓存会对重复前缀复用注意力 KV——Anthropic 的 prompt-caching 文档宣称在长提示词上最高可降低 90% 成本、85% 延迟；以 Claude 3.5 Sonnet 为例，缓存读取价格为 $0.30/M，而全新输入为 $3.00/M，默认 5 分钟 TTL，选择 1 小时 TTL 则写入溢价为 2 倍（docs.anthropic.com，2026-04）。OpenAI 的提示词缓存对 ≥1024 token 的提示词自动生效，缓存输入的价格相比全新输入约打 1 折（platform.openai.com，2026-04）；具体每个模型的缓存费率以实时价目表为准。L1（应用层）语义缓存则在嵌入相似度命中时完全跳过 LLM。厂商所说的「95% 准确率」指的是匹配的正确性，而非命中率——生产环境实际报告的命中率从 10%（开放式聊天）到 70%（结构化 FAQ）不等；两家提供商都没有发布官方基线，所以应将这些数字视为社区遥测数据而非保证。生产环境的两大陷阱：并行化会摧毁缓存（在第一次缓存写入完成之前发出的 N 个并行请求可能让开销膨胀数倍）；前缀中的动态内容会让缓存完全无法命中。ProjectDiscovery 报告称，通过把动态文本移出可缓存前缀，命中率从 7% 提升到 74%（2025-11）。

**Type:** Learn
**Languages:** Python (stdlib, toy two-layer cache simulator)
**Prerequisites:** Phase 17 · 04 (vLLM Serving Internals), Phase 17 · 06 (SGLang RadixAttention)
**Time:** ~60 minutes

## 学习目标

- 区分 L2 提示词/前缀缓存（提供商侧的 KV 复用）与 L1 语义缓存（相似提示词直接绕过 LLM）。
- 解释 Anthropic 的 `cache_control` 显式标记机制，以及两种 TTL 选项（5 分钟 vs 1 小时）各自的价格乘数。
- 根据命中率、提示词/响应配比和 token 价格，计算预期的每月节省额。
- 说出会让账单膨胀 5-10 倍的并行化反模式，以及会让命中率崩盘的动态内容反模式。

## 问题背景

你给 RAG 服务加上了提示词缓存，账单却纹丝不动。你测了一下命中率：7%。你的提示词看起来是静态的，实际上不是——系统提示词里包含精确到分钟的当前时间、一个请求 ID，还有为了多样性而随机重排的示例。每个请求都写入一条新缓存，读取为零。

另一边，你的智能体对每个用户问题发起十个并行工具调用。这十个请求在第一次缓存写入完成之前就全部到达了提供商。十次写入，零次读取。你的账单是「开启缓存后」理论成本的 5-10 倍。

缓存是一套协议，不是一个开关。两个层面，两种截然不同的失败模式。

## 核心概念

### L2——提供商提示词/前缀缓存

提供商存储可缓存前缀对应的注意力 KV，并在下一个匹配该前缀的请求上复用。你只付一次写入成本，后续读取几乎免费。

**Anthropic（Claude 3.5 / 3.7 / 4 系列）**：在请求中显式添加 `cache_control` 标记，由你指定哪些块可缓存。TTL：5 分钟（写入成本为基础价的 1.25 倍）或 1 小时（写入成本为基础价的 2 倍）。缓存读取：Claude 3.5 Sonnet 上为 $0.30/M，全新输入为 $3.00/M——便宜 10 倍（docs.anthropic.com，截至 2026-04）。各模型费率不同（Opus/Haiku 单独公布）；务必对照实时定价页核对。

**OpenAI**：对 ≥1024 token 的提示词自动缓存（platform.openai.com，2026-04），无需显式标记。在当前 gpt-4o/gpt-5 价目表上，缓存输入比全新输入约便宜 10 倍。官方文档和发布说明都没有公布命中率基线；社区报告显示，在精心设计提示词的前提下命中率集中在 30–60%。请监控 `usage.cached_tokens` 来测量你自己的命中情况。

**Google（Gemini）**：通过显式 API 实现上下文缓存；100 万 token 的上下文意味着缓存的收益更大。

**自托管（vLLM、SGLang）**：Phase 17 · 06 讲解了 RadixAttention——在你自己的算力上实现同样的模式。

### L1——应用层语义缓存

在调用 LLM 之前，先对提示词做哈希、做嵌入，然后查找相似的已缓存请求（余弦相似度高于阈值，通常为 0.95+）。命中则直接返回缓存的响应；未命中则调用 LLM 并缓存结果。

开源方案：Redis Vector Similarity、GPTCache、Qdrant。商业方案：Portkey Cache、Helicone Cache。

厂商的准确率宣称指的是返回的缓存响应在语义上恰当的频率——而不是你的命中频率。生产环境命中率：

- 开放式聊天：10-15%。
- 结构化 FAQ / 客服：40-70%。
- 代码问题：20-30%（细微变体会让命中失效）。
- 重复提示词的语音智能体：50-80%（语音归一化后形成固定集合）。

### 并行化反模式

你的智能体并行发起 10 个工具调用，10 个请求带着相同的 4K token 系统提示词。Anthropic 的缓存写入按请求进行；提供商看到提示词后约 300 毫秒才能完成第一次缓存写入。第 2-10 个请求在同一毫秒窗口内到达，每个都遭遇缓存未命中。你支付了 10 份写入溢价，拿到 0 份读取折扣。

修复方法：先串后并——先单独发出请求 1，等它的缓存写入完成后，再齐发请求 2-10。第一个工具调用多花 300 毫秒，账单节省 5-10 倍。

### 动态内容反模式

你的系统提示词长这样：

```
You are a helpful assistant. The current time is 14:32:17.
User ID: abc123. Today is Tuesday...
```

每个请求都是唯一的。每个请求都在写入。命中为零。

修复方法：把所有真正静态的内容移到可缓存前缀中，动态内容追加到缓存边界之后：

```
[cacheable]
You are a helpful assistant. [rules, examples, instructions]
[/cacheable]
[dynamic, not cached]
Current time: 14:32:17. User: abc123.
```

ProjectDiscovery 正是用这种方式把缓存命中率从 7% 提升到 74%，并公开了完整复盘。

### 批处理 + 缓存叠加，应对过夜工作负载

批处理 API（Phase 17 · 15）以 24 小时交付为代价提供 50% 折扣。在此基础上叠加缓存输入，又能拿到约 10 倍优惠。过夜的分类、标注和报告生成类工作负载，通过两者叠加可以降到同步无缓存成本的约 10%。

### 你应该记住的数字

定价数据采集于 2026-04，来自所附厂商文档，每隔几个月就会变动——使用前请重新核实。

- Anthropic 缓存读取：Claude 3.5 Sonnet 上 $0.30/M，比全新输入约便宜 10 倍（docs.anthropic.com）。
- Anthropic 缓存写入溢价：1.25 倍（5 分钟 TTL）或 2 倍（1 小时 TTL）。
- OpenAI 自动缓存：对 ≥1024 token 的提示词生效；当前价目表上缓存输入约为全新输入价格的 10%（platform.openai.com）。
- 语义缓存命中率（社区报告）：开放式聊天约 10%；结构化 FAQ 最高约 70%。并非厂商文档给出的基线。
- ProjectDiscovery：把动态内容移出前缀后，命中率从 7% 提升至 74%（项目博客，2025-11）。
- 并行化反模式：N 个并行请求错过第一次缓存写入时，账单膨胀 5–10 倍是常见报告值。

## 生产实践

`code/main.py` 在混合工作负载上模拟 L1 + L2 缓存，报告命中率与账单，并展示并行化带来的代价。

## 交付产物

本课产出 `outputs/skill-cache-auditor.md`。给定提示词模板和流量特征，它会审计可缓存性并给出重构建议。

## 练习

1. 运行 `code/main.py`。切换并行化开关，账单变化了多少？
2. 你的系统提示词里有日期。把它移出去，给出前后命中率的对比计算。
3. 根据你的请求到达率，计算 1 小时 TTL（2 倍写入）与 5 分钟 TTL（1.25 倍写入）的盈亏平衡点。
4. 语义缓存阈值取 0.95 时命中 20%；取 0.85 时命中 50%，但出现了错误的缓存响应。选出合适的阈值并给出理由。
5. 你为每个用户问题批量发起 10 个并行子查询。在不增加端到端延迟的前提下，把它改写成缓存友好的形式。

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|----------------|------------------------|
| L2 提示词缓存 | 「前缀缓存」 | 提供商为重复前缀存储 KV |
| `cache_control` | 「Anthropic 缓存标记」 | 显式标注可缓存块的属性 |
| 缓存写入溢价 | 「写入税」 | 首次未命中转入缓存的额外成本（1.25 倍或 2 倍） |
| L1 语义缓存 | 「嵌入缓存」 | 应用层在调用 LLM 前先哈希加嵌入 |
| GPTCache | 「LLM 缓存库」 | 流行的开源 L1 缓存库 |
| 缓存命中率 | 「命中数 / 总数」 | 由缓存提供服务的请求占比 |
| 并行化反模式 | 「N 次写入陷阱」 | N 个并行请求未命中缓存 N 次 |
| 动态内容陷阱 | 「提示词带时间陷阱」 | 前缀中的动态字节摧毁命中率 |
| RadixAttention | 「副本内缓存」 | SGLang 的前缀缓存实现 |

## 延伸阅读

- [Anthropic Prompt Caching](https://docs.anthropic.com/en/docs/build-with-claude/prompt-caching) —— 官方 `cache_control` 语义与 TTL 说明。
- [OpenAI Prompt Caching](https://platform.openai.com/docs/guides/prompt-caching) —— 自动缓存行为与生效条件。
- [TianPan — Semantic Caching for LLMs Production](https://tianpan.co/blog/2026-04-10-semantic-caching-llm-production)
- [ProjectDiscovery — Cut LLM Costs 59% With Prompt Caching](https://projectdiscovery.io/blog/how-we-cut-llm-cost-with-prompt-caching)
- [DigitalOcean / Anthropic — Prompt Caching](https://www.digitalocean.com/blog/prompt-caching-with-digital-ocean)
