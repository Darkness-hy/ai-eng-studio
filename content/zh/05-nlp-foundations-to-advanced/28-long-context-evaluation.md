# 长上下文评估 — NIAH、RULER、LongBench、MRCR

> Gemini 3 Pro 宣称支持 1000 万 token 的上下文。但在 100 万 token 处，8-needle MRCR 得分跌到 26.3%。宣称的 ≠ 可用的。长上下文评估告诉你：你实际部署的模型到底有多少真实容量。

**Type:** Learn
**Languages:** Python
**Prerequisites:** Phase 5 · 13 (Question Answering), Phase 5 · 23 (Chunking Strategies)
**Time:** ~60 minutes

## 问题背景

你手上有一份 200 页的合同。模型宣称支持 100 万 token 的上下文。你把合同整篇贴进去，问："终止条款是什么？"模型给出了回答——但答的是封面页的内容，因为终止条款埋在 12 万 token 的深处，超出了模型实际能关注到的范围。

这就是 2026 年的上下文容量鸿沟。规格表上写着 100 万或 1000 万，现实是只有 60-70% 可用，而且"可用"还取决于具体任务。

- **检索（大海捞针式单针）：** 前沿模型在宣称的最大长度内几乎完美。
- **多跳 / 聚合：** 大多数模型在超过约 128k 后急剧退化。
- **对分散事实的推理：** 最先失效的任务。

长上下文评估衡量的就是这几个维度。本节课会逐一介绍这些基准、各自真正测的是什么，以及如何为你的领域构建一个定制化的 needle 测试。

## 核心概念

![NIAH baseline, RULER multi-task, LongBench holistic](../assets/long-context-eval.svg)

**大海捞针（Needle-in-a-Haystack，NIAH，2023）。** 把一个事实（"魔法词是 pineapple"）放在长上下文中受控的深度位置，让模型把它找出来。对深度 × 长度做网格扫描。这是最早的长上下文基准。前沿模型如今已在它上面饱和；它是必要但不充分的基线。

**RULER（Nvidia，2024）。** 4 大类共 13 种任务：检索（单键 / 多键 / 多值）、多跳追踪（变量追踪）、聚合（高频词统计）、问答。上下文长度可配置（4k 到 128k+）。它能揭示那些在 NIAH 上饱和却在多跳任务上失败的模型。在 2024 年发布的结果中，17 个宣称支持 32k+ 上下文的模型里，只有一半在 32k 处保住了质量。

**LongBench v2（2024）。** 503 道多项选择题，上下文长度 8k 到 200 万词，覆盖六类任务：单文档问答、多文档问答、长上下文学习、长对话、代码仓库、长结构化数据。它是衡量真实世界长上下文表现的生产级基准。

**MRCR（Multi-Round Coreference Resolution，多轮指代消解）。** 大规模多轮指代任务，有 8-needle、24-needle、100-needle 等变体。它能暴露一个模型在注意力退化之前到底能同时跟踪多少个事实。

**NoLiMa。** "非词面 needle"。needle 与查询之间没有任何字面重叠，检索需要一步语义推理。比 NIAH 更难。

**HELMET。** 把大量文档拼接在一起，从其中任意一篇提问。测试选择性注意力。

**BABILong。** 把 bAbI 推理链嵌入无关的干扰文本中。测的是"在干草堆里推理"，而不只是检索。

### 真正应该报告什么

- **宣称的上下文窗口。** 规格表上的数字。
- **有效检索长度。** NIAH 在某个阈值（如 90%）下的通过长度。
- **有效推理长度。** 多跳或聚合任务在该阈值下的通过长度。
- **退化曲线。** 按任务类型分别绘制的准确率随上下文长度变化曲线。

你的规格表应该写两个数字：检索有效长度和推理有效长度。通常推理有效长度只有宣称窗口的 25-50%。

## 从零实现

### 第 1 步：为你的领域定制一个 NIAH

参见 `code/main.py`。骨架如下：

```python
def build_haystack(filler_text, needle, depth_ratio, total_tokens):
    if not (0.0 <= depth_ratio <= 1.0):
        raise ValueError(f"depth_ratio must be in [0, 1], got {depth_ratio}")
    if total_tokens <= 0:
        raise ValueError(f"total_tokens must be positive, got {total_tokens}")

    filler_tokens = tokenize(filler_text)
    needle_tokens = tokenize(needle)
    if not filler_tokens:
        raise ValueError("filler_text produced no tokens")

    # Repeat filler until long enough to fill the haystack body.
    body_len = max(total_tokens - len(needle_tokens), 0)
    while len(filler_tokens) < body_len:
        filler_tokens = filler_tokens + filler_tokens
    filler_tokens = filler_tokens[:body_len]

    insert_at = min(int(body_len * depth_ratio), body_len)
    haystack = filler_tokens[:insert_at] + needle_tokens + filler_tokens[insert_at:]
    return " ".join(haystack)


def score_niah(model, haystack, question, expected):
    answer = model.complete(f"Context: {haystack}\nQ: {question}\nA:", max_tokens=50)
    return 1 if expected.lower() in answer.lower() else 0
```

对 `depth_ratio` ∈ {0, 0.25, 0.5, 0.75, 1.0} × `total_tokens` ∈ {1k, 4k, 16k, 64k} 做网格扫描，画出热力图。这就是你目标模型的 NIAH 成绩卡。

### 第 2 步：多 needle 变体

```python
def build_multi_needle(filler, needles, total_tokens):
    depths = [0.1, 0.4, 0.7]
    chunks = [filler[:int(total_tokens * 0.1)]]
    for depth, needle in zip(depths, needles):
        chunks.append(needle)
        next_chunk = filler[int(total_tokens * depth): int(total_tokens * (depth + 0.3))]
        chunks.append(next_chunk)
    return " ".join(chunks)
```

像"三个魔法词分别是什么？"这样的问题要求把三个 needle 全部找回来。单 needle 的成功率并不能预测多 needle 的成功率。

### 第 3 步：多跳变量追踪（RULER 风格）

```python
haystack = """X1 = 42. ... (filler) ... X2 = X1 + 10. ... (filler) ... X3 = X2 * 2."""
question = "What is X3?"
```

答案需要串起三个赋值语句。前沿模型在 128k 长度下做这种任务时，准确率经常掉到 50-70%。

### 第 4 步：在你的技术栈上跑 LongBench v2

```python
from datasets import load_dataset
longbench = load_dataset("THUDM/LongBench-v2")

def eval_model_on_longbench(model, subset="single-doc-qa"):
    tasks = [x for x in longbench["test"] if x["task"] == subset]
    correct = 0
    for x in tasks:
        answer = model.complete(x["context"] + "\n\nQ: " + x["question"], max_tokens=20)
        if normalize(answer) == normalize(x["answer"]):
            correct += 1
    return correct / len(tasks)
```

按类别分别报告准确率。聚合分数会掩盖任务层面的巨大差异。

## 常见陷阱

- **只做 NIAH 评估。** 在 100 万 token 上通过 NIAH 并不能说明任何多跳能力。一定要跑 RULER 或自建的多跳测试。
- **深度采样单一。** 很多实现只测 depth=0.5。要测 depth=0、0.25、0.5、0.75、1.0——"迷失在中间（lost in the middle）"效应是真实存在的。
- **needle 与填充文本有词面重叠。** 如果 needle 与填充文本共享关键词，检索就变得毫无难度。要用 NoLiMa 风格、无词面重叠的 needle。
- **忽略延迟。** 100 万 token 的提示词需要 30-120 秒才能完成预填充（prefill）。在测准确率的同时也要测首 token 延迟（time-to-first-token）。
- **轻信厂商自报数字。** OpenAI、Google、Anthropic 都发布自己的分数。一定要在你自己的用例上独立复跑。

## 生产实践

2026 年的技术栈：

| 场景 | 基准 |
|-----------|-----------|
| 快速健全性检查 | 定制 NIAH，3 个深度 × 3 个长度 |
| 生产环境选型 | 在你的目标长度上跑 RULER（13 项任务） |
| 真实世界问答质量 | LongBench v2 的 single-doc-QA 子集 |
| 多跳推理 | BABILong 或自建变量追踪任务 |
| 会话 / 对话场景 | 在你的目标长度上跑 MRCR 8-needle |
| 模型升级回归测试 | 固定的内部 NIAH + RULER 评测框架，每个新模型都跑一遍 |

生产环境的经验法则：在你的目标长度上跑过 NIAH + 至少 1 个推理任务之前，永远不要相信任何上下文窗口数字。

## 交付产物

保存为 `outputs/skill-long-context-eval.md`：

```markdown
---
name: long-context-eval
description: Design a long-context evaluation battery for a given model and use case.
version: 1.0.0
phase: 5
lesson: 28
tags: [nlp, long-context, evaluation]
---

Given a target model, target context length, and use case, output:

1. Tests. NIAH depth × length grid; RULER multi-hop; custom domain task.
2. Sampling. Depths 0, 0.25, 0.5, 0.75, 1.0 at each length.
3. Metrics. Retrieval pass rate; reasoning pass rate; time-to-first-token; cost-per-query.
4. Cutoff. Effective retrieval length (90% pass) and effective reasoning length (70% pass). Report both.
5. Regression. Fixed harness, rerun on every model upgrade, surface deltas.

Refuse to trust a context window from the model card alone. Refuse NIAH-only evaluation for any multi-hop workload. Refuse vendor self-reported long-context scores as independent evidence.
```

## 练习

1. **简单。** 构建一个 3 个深度（0.25、0.5、0.75）× 3 个长度（1k、4k、16k）的 NIAH，在任意模型上运行，把通过率画成 3×3 热力图。
2. **中等。** 增加一个 3-needle 变体。在每个长度上测量全部 3 个 needle 的检索成功率，并与同长度下的单 needle 通过率对比。
3. **困难。** 构造一个嵌入在 64k 填充文本中的变量追踪任务（X1 → X2 → X3，3 跳）。在 3 个前沿模型上测量准确率，分别报告每个模型的有效推理长度。

## 关键术语

| 术语 | 大家怎么说 | 实际含义 |
|------|-----------------|-----------------------|
| NIAH | 大海捞针 | 把一个事实埋进填充文本，让模型把它找出来。 |
| RULER | 加强版 NIAH | 覆盖检索 / 多跳 / 聚合 / 问答的 13 种任务类型。 |
| 有效上下文 | 真实容量 | 准确率仍保持在阈值之上的最大长度。 |
| 迷失在中间 | 深度偏差 | 模型对长输入中间位置内容的关注不足。 |
| 多 needle | 同时多个事实 | 埋入多个事实；测的是注意力的并行跟踪能力，而不只是检索。 |
| MRCR | 多轮指代 | 8、24 或 100 个 needle 的指代消解；暴露注意力饱和点。 |
| NoLiMa | 非词面 needle | needle 与查询没有任何字面 token 重叠；必须依靠推理。 |

## 延伸阅读

- [Kamradt (2023). Needle in a Haystack analysis](https://github.com/gkamradt/LLMTest_NeedleInAHaystack) — 最初的 NIAH 仓库。
- [Hsieh et al. (2024). RULER: What's the Real Context Size of Your Long-Context LMs?](https://arxiv.org/abs/2404.06654) — 多任务基准。
- [Bai et al. (2024). LongBench v2](https://arxiv.org/abs/2412.15204) — 真实世界长上下文评估。
- [Modarressi et al. (2024). NoLiMa: Non-lexical needles](https://arxiv.org/abs/2404.06666) — 更难的 needle。
- [Kuratov et al. (2024). BABILong](https://arxiv.org/abs/2406.10149) — 在干草堆中推理。
- [Liu et al. (2024). Lost in the Middle: How Language Models Use Long Contexts](https://arxiv.org/abs/2307.03172) — 深度偏差的开山论文。
