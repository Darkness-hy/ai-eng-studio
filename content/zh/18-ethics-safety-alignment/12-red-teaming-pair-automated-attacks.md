# 红队测试：PAIR 与自动化攻击

> Chao、Robey、Dobriban、Hassani、Pappas、Wong（NeurIPS 2023，arXiv:2310.08419）。PAIR——提示自动迭代精炼（Prompt Automatic Iterative Refinement）——是最经典的自动化黑盒越狱方法。一个带有红队系统提示的攻击者 LLM 不断迭代地为目标 LLM 构造越狱提示，并将历次尝试与目标回复累积在自己的对话历史中，作为上下文内反馈。PAIR 通常在 20 次查询内即可成功，比 GCG（Zou 等人的 token 级梯度搜索）高效几个数量级，且无需白盒访问权限。PAIR 如今已是 JailbreakBench（arXiv:2404.01318）和 HarmBench 中的标准基线，与 GCG、AutoDAN、TAP 和 Persuasive Adversarial Prompt 并列。

**Type:** Build
**Languages:** Python (stdlib, mock PAIR loop against a toy target)
**Prerequisites:** Phase 18 · 01 (instruction-following), Phase 14 (agent engineering)
**Time:** ~75 minutes

## 学习目标

- 描述 PAIR 算法：攻击者系统提示、迭代精炼、上下文内反馈。
- 解释为什么在目标为黑盒时，PAIR 严格优于 GCG 的效率。
- 列举另外四个自动化攻击基线（GCG、AutoDAN、TAP、PAP），并说出每个方法的一个显著特征。
- 描述 JailbreakBench 和 HarmBench 的评估协议，以及"攻击成功率"在各自语境下的含义。

## 问题背景

红队测试（red-teaming）过去是一项人工活动：少数专家测试者手工构造对抗性提示，并记录哪些奏效。这种方式无法扩展——攻击成功率需要有统计意义的样本量，而且每次模型发布后目标都在变化。PAIR 把红队测试形式化为一个针对黑盒目标的优化问题。

## 核心概念

### PAIR 算法

输入：
- 目标 LLM T（我们要攻击的模型）。
- 评判 LLM J（判定某个回复是否构成越狱）。
- 攻击者 LLM A（红队优化器）。
- 目标字符串 G："respond with [harmful instruction]."
- 预算 K（通常为 20 次查询）。

循环，对 k 从 1 到 K：
1. 用目标 G 和迄今为止的（提示，回复）对历史来提示 A。
2. A 生成一个新提示 p_k。
3. 将 p_k 提交给 T，得到回复 r_k。
4. J 针对目标对 (p_k, r_k) 打分。
5. 如果分数 >= 阈值，停止——找到了越狱提示。
6. 否则，将 (p_k, r_k) 追加到 A 的历史中，继续循环。

实证结果（NeurIPS 2023）：对 GPT-3.5-turbo 和 Llama-2-7B-chat 的攻击成功率超过 50%；成功所需的平均查询次数在 10-20 次之间。

### 为什么 PAIR 效率高

GCG（Zou et al. 2023）通过梯度搜索对抗性 token 后缀，需要对模型的白盒访问，且生成的后缀不可读。PAIR 是黑盒方法，生成的自然语言攻击可以在模型间迁移。PAIR 的上下文内反馈让攻击者能从每次被拒绝的尝试中学习；GCG 没有对应机制（每次新的 token 更新都必须重新发现之前的进展）。

### 相关的自动化攻击

- **GCG（Zou et al. 2023，arXiv:2307.15043）。** token 级梯度搜索对抗性后缀。白盒、可迁移，生成不可读的字符串。
- **AutoDAN（Liu et al. 2023）。** 在提示空间上做演化搜索，由分层目标函数引导。
- **TAP（Mehrotra et al. 2024）。** 带剪枝的攻击树（tree-of-attacks with pruning）——并行展开多条 PAIR 式攻击路径并分支。
- **PAP（Zeng et al. 2024）。** 劝说式对抗提示（Persuasive Adversarial Prompts）——将人类劝说技巧编码为提示模板。

### JailbreakBench 与 HarmBench

两者（均发布于 2024 年）都对评估做了标准化：

- JailbreakBench（arXiv:2404.01318）。覆盖 10 个 OpenAI 政策类别的 100 种有害行为。以攻击成功率（ASR）作为主要指标。需要一个评判模型（GPT-4-turbo、Llama Guard 或 StrongREJECT）。
- HarmBench（Mazeika et al. 2024）。覆盖 7 个类别的 510 种行为，包含语义性和功能性危害测试。在 33 个模型上比较了 18 种攻击。

ASR 通常在固定查询预算下报告。比较攻击方法时必须使用相同预算；200 次查询下 90% 的 ASR 与 20 次查询下 85% 的 ASR 没有可比性。

### 为什么这对 2026 年的部署很重要

如今每个前沿实验室在发布前都会用 PAIR 和 TAP 攻击生产模型。ASR 变化曲线会出现在模型卡（第 26 课）和安全论证附录（第 18 课）中。这种攻击并不罕见——它已是标准基础设施。

### 本课在 Phase 18 中的位置

第 12 课是自动化攻击的基础。第 13 课（多样本越狱，Many-Shot Jailbreaking）是与之互补的长度利用攻击。第 14 课（ASCII Art / 视觉攻击）是编码类攻击。第 15 课（间接提示注入）是 2026 年的生产环境攻击面。第 16 课讲对应的防御工具（Llama Guard、Garak、PyRIT）。

## 生产实践

`code/main.py` 构建了一个玩具版 PAIR 循环。目标是一个会拒绝"明显"有害提示的模拟分类器（基于关键词过滤）。攻击者是一个基于规则的精炼器，会尝试改写、角色扮演包装和编码变换。评判器对回复打分。你会看到攻击者在大约 5-15 次迭代内攻破关键词过滤器，但在语义过滤器面前失败。

## 交付产物

本课产出 `outputs/skill-attack-audit.md`。给定一份红队评估报告，它会审计：运行了哪些攻击（PAIR、GCG、TAP、AutoDAN、PAP）、各自的查询预算、使用了哪个评判模型、基于哪个有害行为集（JailbreakBench、HarmBench 或内部数据集）。

## 练习

1. 运行 `code/main.py`。测量三种内置攻击者策略各自的平均成功查询次数。解释每种策略利用了目标防御的哪条假设。

2. 实现第四种攻击者策略（例如翻译成另一种语言、base64 编码）。报告新策略针对关键词过滤目标和语义过滤目标的平均成功查询次数。

3. 阅读 Chao et al. 2023 的 Figure 5（PAIR 与 GCG 的对比）。描述两种尽管 PAIR 效率更高、却仍应优先选择 GCG 的场景。

4. JailbreakBench 在固定的目标集上报告 ASR。设计一个额外的指标来衡量攻击多样性（成功提示之间的差异程度）。解释为什么多样性对防御评估很重要。

5. TAP（Mehrotra 2024）用分支加剪枝扩展了 PAIR。为 `code/main.py` 勾画一个 TAP 式的扩展方案，并描述计算成本与成功率之间的权衡。

## 关键术语

| 术语 | 大家怎么说 | 实际含义 |
|------|-----------------|------------------------|
| PAIR | "自动化越狱" | 提示自动迭代精炼（Prompt Automatic Iterative Refinement）；攻击者 LLM + 评判 LLM 的循环 |
| GCG | "梯度越狱" | 白盒 token 级梯度搜索对抗性后缀 |
| 攻击成功率（ASR） | "k 次查询下的越狱百分比" | 主要指标；报告时必须附带查询预算和评判模型身份 |
| 评判 LLM | "打分器" | 判定某个回复是否满足有害目标的 LLM |
| JailbreakBench | "那个评估基准" | 标准化的有害行为集，带类别标签 |
| HarmBench | "更广的基准" | 510 种行为，包含功能性 + 语义性危害测试 |
| TAP | "攻击树" | 带分支 + 剪枝的 PAIR；以更高算力换取更高 ASR |

## 延伸阅读

- [Chao et al. — Jailbreaking Black Box LLMs in Twenty Queries (arXiv:2310.08419)](https://arxiv.org/abs/2310.08419) — PAIR 论文，NeurIPS 2023
- [Zou et al. — Universal and Transferable Adversarial Attacks on Aligned LLMs (arXiv:2307.15043)](https://arxiv.org/abs/2307.15043) — GCG 论文
- [Chao et al. — JailbreakBench (arXiv:2404.01318)](https://arxiv.org/abs/2404.01318) — 标准化评估
- [Mazeika et al. — HarmBench (ICML 2024)](https://arxiv.org/abs/2402.04249) — 覆盖更广的评估
