# 毕业项目 08 — 面向受监管垂直领域的生产级 RAG 聊天机器人

> Harvey、Glean、Mendable 和 LlamaCloud 在 2026 年跑的都是同一套生产架构。用 docling 或 Unstructured 做摄取，视觉内容交给 ColPali。混合检索。用 bge-reranker-v2-gemma 重排序。用 Claude Sonnet 4.7 做合成，配合提示词缓存（prompt caching）达到 60-80% 的命中率。用 Llama Guard 4 和 NeMo Guardrails 做防护。用 Langfuse 和 Phoenix 做监控。用 RAGAS 在 200 题黄金测试集上打分。在一个受监管领域（法律、临床、保险）里把它搭出来——通过黄金测试集、红队测试和漂移看板，就是这个毕业项目的过关标准。

**Type:** Capstone
**Languages:** Python (pipeline + API), TypeScript (chat UI)
**Prerequisites:** Phase 5 (NLP), Phase 7 (transformers), Phase 11 (LLM engineering), Phase 12 (multimodal), Phase 17 (infrastructure), Phase 18 (safety)
**Phases exercised:** P5 · P7 · P11 · P12 · P17 · P18
**Time:** 30 hours

## 问题背景

受监管领域的 RAG（法律合同、临床试验方案、保险条款）是 2026 年落地最多的生产形态，因为投资回报显而易见、风险后果具体可见。Harvey（Allen & Overy）为法律行业做了这件事。Mendable 做的是开发者文档版本。Glean 覆盖企业搜索。这套模式是：高保真摄取，混合检索加重排序，带引用约束和提示词缓存的合成，多层安全防护，以及持续的漂移监控。

难点不在模型本身。难的是感知司法辖区的合规要求（HIPAA、GDPR、SOC2）、精确到引用级别的可审计性、成本控制（提示词缓存在命中率高时能省下 60-90% 的费用）、基于 RAGAS 忠实度（faithfulness）的幻觉检测，以及当源文档更新而索引没跟上时的漂移检测。这个毕业项目要求你在一个 200 题的黄金测试集上把这一整套都交付出来，并附带一套红队测试。

## 核心概念

整条流水线分为两侧。**摄取侧**：docling 或 Unstructured 解析结构化文档；ColPali 处理视觉信息丰富的文档；每个分块（chunk）都附带摘要、标签和基于角色的访问权限标记。向量进入 pgvector + pgvectorscale（5000 万向量以内）或 Qdrant Cloud；稀疏 BM25 检索并行运行。**对话侧**：LangGraph 负责记忆与多轮对话；每个查询先做混合检索，用 bge-reranker-v2-gemma-2b 重排序，用 Claude Sonnet 4.7（启用提示词缓存）合成答案，输出再经过 Llama Guard 4 和 NeMo Guardrails，最后给出带引用锚点的回答。

评估体系有四层。**黄金测试集**（200 条带引用的人工标注问答对）衡量正确性。**红队测试**（越狱、PII 提取尝试、域外问题）衡量安全性。**RAGAS** 逐轮自动评估忠实度 / 答案相关性 / 上下文精确度。**漂移看板**（Arize Phoenix）每周监控检索质量和幻觉分数。

提示词缓存是成本杠杆。Claude 4.5+ 和 GPT-5+ 都支持缓存系统提示词加检索到的上下文。在 60-80% 的命中率下，单次查询成本下降 3-5 倍。流水线必须围绕稳定前缀来设计（系统提示词 + 重排序后的上下文放在前面），才能达到高缓存命中率。

## 架构

```
documents (contracts, protocols, policies)
      |
      v
docling / Unstructured parse + ColPali for visuals
      |
      v
chunks + summaries + role-labels + jurisdiction tags
      |
      v
pgvector + pgvectorscale  +  BM25 (Tantivy)
      |
query + role + jurisdiction
      |
      v
LangGraph conversational agent
   +--- retrieve (hybrid)
   +--- filter by role + jurisdiction
   +--- rerank (bge-reranker-v2-gemma-2b or Voyage rerank-2)
   +--- synthesize (Claude Sonnet 4.7, prompt cached)
   +--- guard (Llama Guard 4 + NeMo Guardrails + Presidio output PII scrub)
   +--- cite + return
      |
      v
eval:
  RAGAS faithfulness / answer_relevance / context_precision (online)
  Langfuse annotation queue (sampled)
  Arize Phoenix drift (weekly)
  red team suite (pre-release)
```

## 技术栈

- 摄取：Unstructured.io 或 docling 处理结构化文档；ColPali 处理视觉信息丰富的 PDF
- 向量数据库：5000 万向量以内用 pgvector + pgvectorscale；超出则用 Qdrant Cloud
- 稀疏检索：带字段权重的 Tantivy BM25
- 编排：LlamaIndex Workflows（摄取）+ LangGraph（对话）
- 重排序器：自托管 bge-reranker-v2-gemma-2b，或托管版 Voyage rerank-2
- LLM：Claude Sonnet 4.7 配合提示词缓存；备选方案为自托管 Llama 3.3 70B
- 评估：RAGAS 0.2 在线评估，DeepEval 负责幻觉和越狱测试套件
- 可观测性：自托管 Langfuse 加标注队列；Arize Phoenix 负责漂移监控
- 防护栏：Llama Guard 4 输入/输出分类器，NeMo Guardrails v0.12 策略，Presidio PII 清洗
- 合规：分块上的基于角色的访问标签；针对 GDPR/HIPAA 的司法辖区标签

```figure
canary-rollout
```

## 从零实现

1. **摄取。** 用 Unstructured 或 docling 解析你的语料库（认真做的话需要 1000-10000 份文档）。扫描件 / 视觉内容多的页面走 ColPali。生成附带摘要、角色标签、司法辖区标签的分块。

2. **建索引。** 稠密嵌入（Voyage-3 或 Nomic-embed-v2）存入 pgvector + pgvectorscale。通过 Tantivy 建 BM25 旁路索引。角色和司法辖区过滤条件存为 payload。

3. **混合检索。** 先按角色+司法辖区过滤；然后并行执行稠密检索 + BM25；用倒数排名融合（reciprocal rank fusion）合并；top-20 送入重排序器；top-5 送入合成阶段。

4. **带提示词缓存的合成。** 系统提示词 + 静态策略放在缓存头部；重排序后的上下文作为缓存扩展；用户问题作为不缓存的后缀。稳态下目标缓存命中率 60-80%。

5. **防护栏。** Llama Guard 4 检查输入；NeMo Guardrails 的策略规则拦截域外问题或策略禁止的话题；Presidio 清洗输出中意外出现的 PII；最后做引用强制校验的后置过滤。

6. **黄金测试集。** 由领域专家标注的 200 条问答对，每条带（答案、引用）。从精确引用匹配、答案正确性、忠实度（RAGAS）三个维度给智能体打分。

7. **红队测试。** 50 条对抗性提示：越狱（PAIR、TAP）、PII 窃取尝试、域外问题、跨司法辖区泄露。以通过/失败加严重程度打分。

8. **漂移看板。** Arize Phoenix 每周跟踪检索质量（nDCG、引用忠实度）。下降 5% 即告警。

9. **成本报告。** Langfuse：提示词缓存命中率、每查询 token 数、按阶段拆分的每查询美元成本。

## 生产实践

```
$ chat --role=analyst --jurisdiction=GDPR
> what is the data-retention obligation for EU user profiles under our contract?
[retrieve]  hybrid top-20 filtered to GDPR + analyst-role
[rerank]    top-5 kept
[synth]     claude-sonnet-4.7, cache hit 74%, 0.8s
answer:
  The contract (Section 12.4, Master Services Agreement dated 2024-03-11)
  obligates EU user profile deletion within 30 days of termination per GDPR
  Article 17. The DPA amendment (DPA-v2.1, Section 5) extends this to 14 days
  for "restricted" category data.
  citations: [MSA-2024-03-11 s12.4, DPA-v2.1 s5]
```

## 交付产物

`outputs/skill-production-rag.md` 描述了交付物。一个部署完成、带合规标签的受监管领域聊天机器人，通过了评分细则，并配有实时漂移监控。

| 权重 | 评分项 | 测量方式 |
|:-:|---|---|
| 25 | RAGAS 忠实度 + 答案相关性 | 黄金测试集（200 条问答）上的在线分数 |
| 20 | 引用正确性 | 答案中带可验证来源锚点的比例 |
| 20 | 防护栏覆盖度 | Llama Guard 4 通过率 + 越狱测试套件结果 |
| 20 | 成本 / 延迟工程 | 提示词缓存命中率、p95 延迟、每查询美元成本 |
| 15 | 漂移监控看板 | Phoenix 实时看板及每周检索质量趋势 |
| **100** | | |

## 练习

1. 在不同司法辖区下构建第二份语料库切片（例如 GDPR 之外再加 HIPAA）。用一组 20 题的跨辖区探测问题，证明角色+司法辖区过滤能防止跨域泄露。

2. 测量一周生产流量下的提示词缓存命中率。找出哪些查询破坏了缓存前缀。重构它们。

3. 加入多轮记忆，使用 10k token 的摘要缓冲区。测量随对话变长，忠实度是否下降。

4. 把 Claude Sonnet 4.7 换成自托管的 Llama 3.3 70B。测量每查询成本和忠实度的变化量。

5. 加入「不确定」模式：当重排序后的最高分低于阈值时，智能体回答「我没有可信的引用来源」而不是强行作答。测量虚假自信的下降幅度。

## 关键术语

| 术语 | 大家怎么说 | 实际含义 |
|------|-----------------|------------------------|
| 提示词缓存（Prompt caching） | 「缓存系统提示词 + 上下文」 | Claude/OpenAI 的功能：命中时缓存的前缀 token 折扣 60-90% |
| RAGAS | 「RAG 评估器」 | 自动评估忠实度、答案相关性、上下文精确度 |
| 黄金测试集（Golden set） | 「标注评估集」 | 200 条以上专家标注的带引用问答对；即真值标准 |
| 司法辖区标签（Jurisdiction tag） | 「合规标签」 | 附加在分块上的 GDPR/HIPAA/SOC2 适用范围；由检索过滤器强制执行 |
| 引用忠实度（Citation faithfulness） | 「有据可依的回答率」 | 有可检索来源片段支撑的论断所占比例 |
| 漂移（Drift） | 「检索质量衰减」 | nDCG 或引用分数的周度变化；告警阈值 5% |
| 红队测试（Red team） | 「对抗性评估」 | 发布前的越狱、PII 提取、域外问题探测 |

## 延伸阅读

- [Harvey AI](https://www.harvey.ai) — 法律领域生产架构参考
- [Glean enterprise search](https://www.glean.com) — 企业级 RAG 参考
- [Mendable documentation](https://mendable.ai) — 开发者文档 RAG 参考
- [LlamaCloud Parse + Index](https://docs.llamaindex.ai/en/stable/examples/llama_cloud/llama_parse/) — 托管式摄取
- [Anthropic prompt caching](https://docs.anthropic.com/en/docs/build-with-claude/prompt-caching) — 成本杠杆的权威参考
- [RAGAS 0.2 documentation](https://docs.ragas.io/) — RAG 评估的标准框架
- [Arize Phoenix](https://github.com/Arize-ai/phoenix) — 漂移可观测性参考
- [Llama Guard 4](https://ai.meta.com/research/publications/llama-guard-4/) — 2026 年的安全分类器
- [NeMo Guardrails v0.12](https://docs.nvidia.com/nemo-guardrails/) — 策略防护栏框架
