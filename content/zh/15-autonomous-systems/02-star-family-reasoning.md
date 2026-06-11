# STaR、V-STaR、Quiet-STaR —— 自学推理

> 最小的自我改进循环就藏在推理过程（rationale）里。模型生成一条思维链，保留那些最终得到正确答案的，再用它们做微调。这就是 STaR。V-STaR 加入了一个验证器，让推理时的筛选更准。Quiet-STaR 则把推理下沉到每一个 token。三种方法都有效，但都不是魔法——这个循环会保留任何碰巧得出正确答案的捷径。

**Type:** Learn
**Languages:** Python (stdlib, bootstrap-loop simulator)
**Prerequisites:** Phase 13 · 01-03 (Reasoning and CoT), Phase 15 · 01 (long-horizon framing)
**Time:** ~60 minutes

## 问题背景

教模型推理最直接的办法是收集人类编写的推理轨迹。但这样做成本高、速度慢，而且受限于人类愿意写多少高质量的思维链。

STaR（Self-Taught Reasoner，Zelikman et al., 2022）提出：如果让模型自己写推理过程，再用已知答案给它打分会怎样？循环如下：

1. 采样一条推理轨迹及其答案。
2. 如果最终答案正确，保留这条轨迹。
3. 在保留的轨迹上做微调。
4. 重复以上步骤。

这套方法是有效的。GSM8K 和 CommonsenseQA 上的成绩都在没有新增人工标注的情况下得到了提升。但这个循环有一个内在偏差：只要推理过程产生了正确答案就会被保留，不管推理本身是否合理。V-STaR（Hosseini et al., 2024）用一个习得的验证器来修补这一点；Quiet-STaR（Zelikman et al., 2024）则把这个思路推广为逐 token 的内部推理。

## 核心概念

### STaR：在有效的样本上自举

从一个具备一定弱推理能力的基础模型开始。对每个训练问题，采样一条推理过程及答案。如果答案与标签匹配，就保留这个（问题、推理过程、答案）三元组。在保留的集合上微调模型，然后重复。

有一个关键细节。如果模型对某个问题永远做不对，循环就无法从中学习。STaR 为此加入了**合理化（rationalization）**：对模型答错的问题，把正确答案作为提示注入，再让模型生成一条通向该答案的推理过程。这些合理化得到的推理过程也会被加入训练集。

原始论文（Zelikman et al., 2022）的结果：GPT-J 基础模型经过多轮带合理化的 STaR 迭代，在 GSM8K 上从 5.8% 提升到 10.7%——绝对提升约 5 个百分点。在 CommonsenseQA 上，经 STaR 训练的 GPT-J 6B 达到 72.5%，与在人工标注推理数据上微调的 GPT-3 175B（约 73%）相当——后者是一个大约 30 倍规模的模型。

### V-STaR：用 DPO 训练验证器

STaR 会丢弃错误的推理过程。Hosseini et al. (2024) 注意到这些也是数据：每一对（推理过程，「它是否正确」）都可以用来训练一个验证器。他们用直接偏好优化（Direct Preference Optimization，DPO）在正确和错误的解答上训练出一个排序器。推理时采样 N 条推理过程，选出验证器评分最高的那条。

报告的提升幅度：在 GSM8K 和 MATH 上比此前的自我改进基线高出 4 到 17 个百分点，其中大部分增益来自用验证器做推理时筛选，而不是对生成器做额外微调。

### Quiet-STaR：逐 token 的内部推理

Zelikman et al. (2024) 提出：如果模型在每个 token 位置都学会生成一段简短的内部推理，而不只是在问题和答案之间生成，会怎样？Quiet-STaR 训练模型在预测每个 token 之前先生成一段隐藏的「思考」，然后通过一个可学习的权重把带思考的预测与基线预测混合起来。

结果：Mistral 7B 在没有任务特定微调的情况下，零样本成绩在 GSM8K 上从 5.9% 提升到 10.9%，在 CommonsenseQA 上从 36.3% 提升到 47.2%。模型学会了「何时该思考」——困难的 token 得到更长的内部推理；简单的 token 几乎不分配。

### 为什么三者共享同一个安全隐患

三种方法都用最终答案作为梯度信号。一条通过有缺陷的推理得到正确答案的推理过程——利用捷径、靠猜、或依赖不可泛化的模式——会被正向强化。在分布内的问题上，捷径管用；在分布外的问题上，它会悄无声息地失效。

V-STaR 的验证器通过学习对推理过程排序来缓解这个问题，但验证器是在同一套标签上训练的。它可能学会偏爱格式漂亮的错误推理，而非诚实的不确定性表达。更安全的设计是把 STaR 式数据与以下两者结合：(a) 过程监督奖励模型（奖励中间步骤，而不只是答案），(b) 能够击破简单捷径的留出分布外（OOD）评测。

### 对比

| 方法 | 训练信号 | 推理成本 | 数据浪费 | 已知失效模式 |
|---|---|---|---|---|
| STaR | 答案正确则保留（推理过程，答案） | 1x | 丢弃所有错误推理过程 | 捷径推理 |
| STaR + 合理化 | 上一行 + 注入正确答案的提示重试 | 1x | 较少 | 合理化得到的推理可能不可信 |
| V-STaR | STaR + 用正负两类样本 DPO 训练验证器 | Nx（best-of-N） | 极少 | 验证器可能强化自信的错误 |
| Quiet-STaR | 逐 token 推理 + 混合权重 | 1.5-3x | 极少 | 梯度仍以答案为条件 |

### 在 2026 年技术栈中的位置

STaR 是个老方法。但这个模式在 2025-2026 年随处可见。在可验证数学问题上做强化学习（DeepSeek-R1、Kimi-k1.5、o1）正是 STaR 那种以答案为条件的梯度信号的规模化版本。过程奖励模型（Lightman et al., 2023；OpenAI 的「Let's verify step by step」）是过程监督的替代路线。AlphaEvolve（第 3 课）是把 STaR 用于代码，用程序评估器代替标签。Darwin Godel Machine（第 4 课）是把 STaR 用于智能体脚手架本身。

理解了 STaR，这些工作就都能融会贯通。它是最小可行的自我改进循环。

```figure
reflection-loop
```

## 生产实践

`code/main.py` 在一个玩具算术任务上运行模拟的 STaR 循环。你可以观察：

- 准确率如何随自举轮次攀升。
- 捷径如何悄悄混进来：模拟器包含一类「偷懒」推理，它有 40% 的概率蒙对答案但泛化很差。观察 STaR 是否会保留它们。
- 验证器（V-STaR 风格）如何在推理时起作用，但无法完全剪除训练阶段引入的捷径。

## 交付产物

`outputs/skill-star-loop-reviewer.md` 帮助你在训练之前审查一条候选的自学推理流水线。

## 练习

1. 运行模拟器。先把捷径频率设为零，再设为 0.4。两次运行在训练分布上都达到了 >90%，但最终准确率相差多少？

2. 给模拟器加一个留出的 OOD 测试。从另一个分布中抽取问题，在分布内和 OOD 两个集合上评估自举后的模型。量化两者的差距。

3. 阅读 Quiet-STaR 论文（arXiv:2403.09629）第 3 节。分别用三句话解释「end-of-thought」token 和混合权重头（mixing-weight head）。

4. 把 STaR 的「正确才保留」过滤器与一种独立奖励每个推理步骤的过程监督方案做比较。指出两者在标注成本上的差异，以及质量上可能的差异。

5. 设计一个能在已部署模型中捕捉捷径推理的评测。它不必完美——只需要能击破 STaR 循环最容易强化的那些捷径。

## 关键术语

| 术语 | 通俗说法 | 实际含义 |
|---|---|---|
| STaR | 「Self-Taught Reasoner（自学推理器）」 | 在模型自己生成且答案正确的推理过程上微调；反复迭代 |
| 合理化（Rationalization） | 「带提示的重试」 | 对基础模型答错的问题，注入正确答案后重新提示生成推理过程 |
| V-STaR | 「带验证器的 STaR」 | 用正确和错误两类推理过程 DPO 训练验证器，用于推理时筛选 |
| Quiet-STaR | 「逐 token 推理」 | 在每个 token 位置生成隐藏的思考；与基线预测混合 |
| 答案条件梯度（Answer-conditioned gradient） | 「基于结果的信号」 | 训练循环奖励的是最终答案，而非推理步骤 |
| 过程奖励模型（Process reward model） | 「步骤级验证器」 | 在逐步正确性而非结果上训练的奖励模型——与 STaR 形成对照 |
| 捷径推理（Shortcut rationale） | 「答案对、推理错」 | 通过不可泛化的模式碰到标签的推理过程；STaR 会保留它们 |

## 延伸阅读

- [Zelikman et al. (2022). STaR: Bootstrapping Reasoning With Reasoning](https://arxiv.org/abs/2203.14465) —— 原始论文。
- [Hosseini et al. (2024). V-STaR: Training Verifiers for Self-Taught Reasoners](https://arxiv.org/abs/2402.06457) —— 加入 DPO 验证器用于推理时筛选。
- [Zelikman et al. (2024). Quiet-STaR: Language Models Can Teach Themselves to Think Before Speaking](https://arxiv.org/abs/2403.09629) —— 逐 token 的内部推理。
- [Lightman et al. (2023). Let's Verify Step by Step](https://arxiv.org/abs/2305.20050) —— 过程奖励模型，另一种梯度信号。
- [DeepSeek-R1 paper (arXiv:2501.12948)](https://arxiv.org/abs/2501.12948) —— 在可验证任务上做强化学习，STaR 在前沿训练中的规模化。
