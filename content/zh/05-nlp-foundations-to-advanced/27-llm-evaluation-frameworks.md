# LLM 评估 — RAGAS、DeepEval、G-Eval

> 精确匹配和 F1 无法识别语义等价，人工评审又难以规模化。LLM-as-judge 是生产环境的答案——前提是做足校准，让这个分数值得信任。

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 5 · 13 (Question Answering), Phase 5 · 14 (Information Retrieval)
**Time:** ~75 minutes

## 问题背景

你的 RAG 系统回答："June 29th, 2007."
标准参考答案是："June 29, 2007."
精确匹配（Exact Match）得 0 分，F1 约 75%，而人类会打 100 分。

现在把这个问题乘以 10,000 个测试用例，再乘以检索器、分块、提示词或模型的每一次变更。你需要一个评估器：它理解语义、能低成本大规模运行、不会在回归问题上撒谎，还能暴露出真正重要的失败模式。

2026 年，有三个框架主导着这个领域。

- **RAGAS。** Retrieval-Augmented Generation ASsessment。四个 RAG 指标（忠实度、答案相关性、上下文精确率、上下文召回率），后端基于 NLI + LLM 评审。有研究支撑，轻量级。
- **DeepEval。** LLM 界的 Pytest。提供 G-Eval、任务完成度、幻觉、偏见等指标。原生支持 CI/CD。
- **G-Eval。** 一种方法（同时也是 DeepEval 的一个指标）：带思维链的 LLM-as-judge，支持自定义标准，输出 0-1 分数。

三者都依赖 LLM-as-judge。本课将帮你建立对这一方法的直觉，以及围绕它的信任层。

## 核心概念

![Four evaluation dimensions, LLM-as-judge architecture](../assets/llm-evaluation.svg)

**LLM 评审（LLM-as-judge）。** 用一个根据评分标准给输出打分的 LLM 取代静态指标。给定 `(query, context, answer)`，向评审 LLM 发出提示："按忠实度打 0-1 分。"返回分数。

它为什么有效：LLM 能以极低的成本近似人类判断。GPT-4o-mini 每个评分用例约 $0.003，1000 个样本的回归评估一轮跑下来不到 $5。

它为什么会悄无声息地失效：

1. **评审偏见。** 评审模型偏爱更长的答案、来自同一模型家族的答案、与提示风格一致的答案。
2. **JSON 解析失败。** 坏的 JSON → NaN 分数 → 被静默排除在聚合结果之外。RAGAS 用户都懂这种痛。用 try/except + 显式失败模式做防护。
3. **模型版本漂移。** 升级评审模型会改变所有指标。冻结评审模型及其版本。

**RAG 四大指标。**

| 指标 | 回答的问题 | 后端 |
|--------|----------|---------|
| 忠实度（Faithfulness） | 答案中的每条声明是否都来自检索到的上下文？ | 基于 NLI 的蕴含判断 |
| 答案相关性（Answer relevance） | 答案是否回应了问题？ | 从答案生成假设性问题，与真实问题比对 |
| 上下文精确率（Context precision） | 检索到的分块中，有多少比例是相关的？ | LLM 评审 |
| 上下文召回率（Context recall） | 检索是否返回了所需的全部信息？ | LLM 评审，对照标准答案 |

**G-Eval。** 定义一条自定义标准："答案是否引用了正确的来源？"框架会自动将其展开为思维链评估步骤，然后给出 0-1 分数。适合 RAGAS 未覆盖的领域特定质量维度。

**校准（Calibration）。** 在与人工标注做相关性验证之前，永远不要相信原始评审分数。跑 100 个人工标注样本，画出评审分 vs 人工分的散点图，计算 Spearman rho。如果 rho < 0.7，你的评审标准需要返工。

## 从零实现

### Step 1: 用 NLI 实现忠实度（RAGAS 风格）

```python
from typing import Callable
from transformers import pipeline

nli = pipeline("text-classification",
               model="MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli",
               top_k=None)

# `llm` is any callable: prompt str -> generated str.
# Example: llm = lambda p: client.messages.create(model="claude-haiku-4-5", ...).content[0].text
LLM = Callable[[str], str]


def atomic_claims(answer: str, llm: LLM) -> list[str]:
    prompt = f"""Break this answer into simple factual claims (one per line):
{answer}
"""
    return llm(prompt).splitlines()


def faithfulness(answer: str, context: str, llm: LLM) -> float:
    claims = atomic_claims(answer, llm)
    if not claims:
        return 0.0
    supported = 0
    for claim in claims:
        result = nli({"text": context, "text_pair": claim})[0]
        entail = next((s for s in result if s["label"] == "entailment"), None)
        if entail and entail["score"] > 0.5:
            supported += 1
    return supported / len(claims)
```

将答案分解为原子声明，逐条用 NLI 对照检索到的上下文做校验。忠实度 = 被支持声明的比例。

### Step 2: 答案相关性

```python
import numpy as np
from sentence_transformers import SentenceTransformer

# encoder: any model implementing .encode(texts, normalize_embeddings=True) -> ndarray
# e.g., encoder = SentenceTransformer("BAAI/bge-small-en-v1.5")

def answer_relevance(question: str, answer: str, encoder, llm: LLM, n: int = 3) -> float:
    prompt = f"Write {n} questions this answer could be the answer to:\n{answer}"
    generated = [line for line in llm(prompt).splitlines() if line.strip()][:n]
    if not generated:
        return 0.0
    q_emb = np.asarray(encoder.encode([question], normalize_embeddings=True)[0])
    g_embs = np.asarray(encoder.encode(generated, normalize_embeddings=True))
    sims = [float(q_emb @ g_emb) for g_emb in g_embs]
    return sum(sims) / len(sims)
```

如果答案暗示的问题与实际提出的问题不同，相关性就会下降。

### Step 3: G-Eval 自定义指标

```python
from deepeval.metrics import GEval
from deepeval.test_case import LLMTestCaseParams, LLMTestCase

metric = GEval(
    name="Correctness",
    criteria="The answer should be factually accurate and match the expected output.",
    evaluation_steps=[
        "Read the expected output.",
        "Read the actual output.",
        "List factual claims in the actual output.",
        "For each claim, mark supported or unsupported by the expected output.",
        "Return score = fraction supported.",
    ],
    evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT, LLMTestCaseParams.EXPECTED_OUTPUT],
)

test = LLMTestCase(input="When was the first iPhone released?",
                   actual_output="June 29th, 2007.",
                   expected_output="June 29, 2007.")
metric.measure(test)
print(metric.score, metric.reason)
```

评估步骤就是评分标准。显式的步骤比隐式的"打 0-1 分"提示更稳定。

### Step 4: CI 门禁

```python
import deepeval
from deepeval.metrics import FaithfulnessMetric, ContextualRelevancyMetric


def test_rag_system():
    cases = load_regression_cases()
    faith = FaithfulnessMetric(threshold=0.85)
    rel = ContextualRelevancyMetric(threshold=0.7)
    for case in cases:
        faith.measure(case)
        assert faith.score >= 0.85, f"faithfulness regression on {case.id}"
        rel.measure(case)
        assert rel.score >= 0.7, f"relevancy regression on {case.id}"
```

以 pytest 文件形式交付。每个 PR 都跑一遍，出现回归就阻止合并。

### Step 5: 从零写一个玩具版评估

见 `code/main.py`。仅用标准库近似实现忠实度（答案声明与上下文的重叠）和相关性（答案 token 与问题 token 的重叠）。不能用于生产，但能看出整体形态。

## 常见陷阱

- **不做校准。** 与人工标注相关性只有 0.3 的评审就是噪声。上线前必须先跑一轮校准。
- **自我评估。** 用同一个 LLM 既生成又评审，会让分数虚高 10-20%。评审要用不同的模型家族。
- **成对评审中的位置偏见。** 评审模型偏爱先呈现的选项。务必随机化顺序并双向各跑一次。
- **原始聚合值掩盖失败。** 平均分 0.85 往往掩盖了 5% 的灾难性失败。一定要检查最低分位段。
- **黄金数据集腐化。** 不做版本管理、随时间漂移的评估集会破坏纵向比较。每次变更都要给数据集打标签。
- **LLM 成本。** 规模上去之后，评审调用主导总成本。选用能达到校准阈值的最便宜模型：GPT-4o-mini、Claude Haiku、Mistral-small。

## 生产实践

2026 年的技术栈：

| 用例 | 框架 |
|---------|-----------|
| RAG 质量监控 | RAGAS（4 个指标） |
| CI/CD 回归门禁 | DeepEval + pytest |
| 自定义领域标准 | DeepEval 中的 G-Eval |
| 在线真实流量监控 | RAGAS 的无参考（reference-free）模式 |
| 人工抽查（human-in-the-loop） | 带标注界面的 LangSmith 或 Phoenix |
| 红队测试 / 安全评估 | Promptfoo + DeepEval |

典型组合：RAGAS 做监控，DeepEval 做 CI，G-Eval 处理新维度。三个一起跑——它们的分歧本身就有价值。

## 交付产物

保存为 `outputs/skill-eval-architect.md`：

```markdown
---
name: eval-architect
description: Design an LLM evaluation plan with calibrated judge and CI gates.
version: 1.0.0
phase: 5
lesson: 27
tags: [nlp, evaluation, rag]
---

Given a use case (RAG / agent / generative task), output:

1. Metrics. Faithfulness / relevance / context-precision / context-recall + any custom G-Eval metrics with criteria.
2. Judge model. Named model + version, rationale for cost vs accuracy.
3. Calibration. Hand-labeled set size, target Spearman rho vs human > 0.7.
4. Dataset versioning. Tag strategy, change log, stratification.
5. CI gate. Thresholds per metric, regression-window logic, bottom-quantile alert.

Refuse to rely on a judge untested against ≥50 human-labeled examples. Refuse self-evaluation (same model generates + judges). Refuse aggregate-only reporting without bottom-10% surfacing. Flag any pipeline where judge upgrade lands without parallel baseline eval.
```

## 练习

1. **简单。** 在 10 个含已知幻觉的 RAG 样例上运行 RAGAS。验证忠实度指标能逐一捕获这些幻觉。
2. **中等。** 人工给 50 条 QA 答案按正确性打 0-1 分，再用 G-Eval 打分，计算评审分与人工分之间的 Spearman rho。
3. **困难。** 用 DeepEval 搭一个 pytest CI 门禁。故意让检索器性能回退，验证门禁会失败。再通过对最低 10% 分数做阈值检查，加上最低分位段告警。

## 关键术语

| 术语 | 大家怎么说 | 实际含义 |
|------|-----------------|-----------------------|
| LLM-as-judge | 用 LLM 打分 | 给评审模型一份评分标准，让它对输出打 0-1 分。 |
| RAGAS | 那个 RAG 指标库 | 开源评估框架，含 4 个无参考的 RAG 指标。 |
| 忠实度（Faithfulness） | 答案有依据吗？ | 答案声明中能被检索上下文蕴含的比例。 |
| 上下文精确率（Context precision） | 检索到的分块相关吗？ | top-K 分块中真正起作用的比例。 |
| 上下文召回率（Context recall） | 检索找全了吗？ | 标准答案声明中被检索分块支持的比例。 |
| G-Eval | 自定义 LLM 评审 | 评分标准 + 思维链评估步骤 + 0-1 分数。 |
| 校准（Calibration） | 信任但要验证 | 评审分数与人工分数之间的 Spearman 相关性。 |

## 延伸阅读

- [Es et al. (2023). RAGAS: Automated Evaluation of Retrieval Augmented Generation](https://arxiv.org/abs/2309.15217) — RAGAS 论文。
- [Liu et al. (2023). G-Eval: NLG Evaluation using GPT-4 with Better Human Alignment](https://arxiv.org/abs/2303.16634) — G-Eval 论文。
- [DeepEval docs](https://deepeval.com/docs/metrics-introduction) — 开源的生产级评估栈。
- [Zheng et al. (2023). Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena](https://arxiv.org/abs/2306.05685) — 偏见、校准与局限。
- [MLflow GenAI Scorer](https://mlflow.org/blog/third-party-scorers) — 集成 RAGAS、DeepEval、Phoenix 的统一框架。
