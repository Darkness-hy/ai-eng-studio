# 奖励建模与 RLHF

> 人类无法为"好的助手回复"写出一个奖励函数，但他们可以比较两条回复并选出更好的那条。把奖励模型拟合到这些比较数据上，再用 RL 让语言模型对着它优化。Christiano 2017。InstructGPT 2022。这就是把 GPT-3 变成 ChatGPT 的配方。到 2026 年它大多已被 DPO 取代——但这套思维模型依然成立。

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 5 · 05 (Sentiment), Phase 9 · 08 (PPO)
**Time:** ~45 minutes

## 问题背景

你用下一个 token 预测的目标训练了一个语言模型。它能写出语法正确的英文。但它也会撒谎、东拉西扯，并且该拒绝时不拒绝。靠更多预训练解决不了这个问题——网络文本本身就是病因，而不是解药。

你想要一个*标量奖励*，能表达"对于指令 X，回复 A 比回复 B 更好"。手写这个奖励函数是不可能的。"有帮助性"不是一个能用 token 写出闭式表达式的东西。但人类可以比较两条输出并标注偏好。这种数据可以低成本地大规模收集。

RLHF（Christiano et al. 2017；Ouyang et al. 2022）把偏好数据转化为一个奖励模型，再用 PPO 对着这个奖励优化语言模型。三步走：SFT → RM → PPO。这就是 2023–2025 年间催生 ChatGPT、Claude、Gemini 以及所有其他对齐 LLM 的配方。

到 2026 年，PPO 这一步大多已被 DPO（Phase 10 · 08）取代，因为后者更便宜，而且在对齐微调上效果几乎相当。但*奖励模型*这一块仍然是每个 Best-of-N 采样器、每条可验证奖励 RL 流水线、以及每个使用过程奖励模型的推理模型的基石。理解了 RLHF，你就理解了整个对齐技术栈。

## 核心概念

![Three-stage RLHF: SFT, RM training on pairwise prefs, PPO with KL penalty](../assets/rlhf.svg)

**阶段 1：监督微调（Supervised Fine-Tuning，SFT）。** 从预训练的基座模型出发。在人类撰写的目标行为示范数据上微调（遵循指令的回复、有帮助的回答等）。结果：得到一个*偏向良好行为*的模型 `π_SFT`，但它的动作空间仍然不受约束。

**阶段 2：奖励模型训练。**

- 针对提示词 `x` 收集成对回复 `(y_+, y_-)`，由人类标注为"y_+ 优于 y_-"。
- 训练一个奖励模型 `R_φ(x, y)`，使其给 `y_+` 打更高的分。
- 损失函数：**Bradley-Terry 成对逻辑损失**：

  `L(φ) = -E[ log σ(R_φ(x, y_+) - R_φ(x, y_-)) ]`

  σ 是 sigmoid 函数。奖励之差对应偏好的对数几率。BT 自 1952 年（Bradley-Terry）以来一直是标准做法，也是现代 RLHF 中的主流选择。

- `R_φ` 通常从 SFT 模型初始化，并在其上加一个标量头。同一个 Transformer 主干；由一个单独的线性层输出奖励。

**阶段 3：带 KL 惩罚地对 RM 跑 PPO。**

- 从 `π_SFT` 初始化可训练策略 `π_θ`。保留一个冻结的*参考模型* `π_ref = π_SFT`。
- 一条回复 `y` 结束时的奖励：

  `r_total(x, y) = R_φ(x, y) - β · KL(π_θ(·|x) || π_ref(·|x))`

  KL 惩罚阻止 `π_θ` 任意偏离 `π_SFT`——它是一个*正则化项*，而不是硬性信任域。`β` 通常取 `0.01`-`0.05`。
- 用这个奖励跑 PPO（第 08 课）。优势值在 token 级轨迹上计算，但 RM 只对完整回复打分。

**为什么要 KL？** 没有它，PPO 会毫不客气地找到奖励黑客（reward hacking）策略——RM 只在分布内的补全上训练过。某个分布外的回复可能比任何人类写的回复得分都高。KL 把 `π_θ` 约束在 RM 训练所在的流形附近。它是 RLHF 中最重要的一个旋钮。

**2026 年现状：**

- **DPO**（Rafailov 2023）：通过闭式代数推导把阶段 2+3 坍缩成偏好数据上的单一监督损失。不需要 RM，不需要 PPO。在对齐基准上质量相当，算力只需一小部分。在 Phase 10 · 08 讲解。
- **GRPO**（DeepSeek 2024–2025）：用组内相对基线代替 critic 的 PPO，奖励来自*验证器*（代码能跑通 / 数学答案匹配）而不是人类训练的 RM。是推理模型的主流方法。在 Phase 9 · 12 讲解。
- **过程奖励模型（Process reward models，PRM）：** 对部分解（每个推理步骤）打分，在 RLHF 和 GRPO 的推理变体中都有使用。
- **Constitutional AI / RLAIF：** 用已对齐的 LLM 代替人类生成偏好数据。让偏好预算得以扩展。

## 从零实现

本课用很小的合成"提示词"和"回复"，都以字符串表示。RM 是基于词袋（bag-of-tokens）表示的线性打分器。没有真实 LLM——重要的是流水线的*形态*，而不是规模。见 `code/main.py`。

### 第 1 步：合成偏好数据

```python
PROMPTS = ["help me", "answer me", "explain this"]
GOOD_WORDS = {"clear", "specific", "kind", "thorough"}
BAD_WORDS = {"vague", "rude", "wrong", "short"}

def make_pair(rng):
    x = rng.choice(PROMPTS)
    y_good = rng.choice(list(GOOD_WORDS)) + " " + rng.choice(list(GOOD_WORDS))
    y_bad = rng.choice(list(BAD_WORDS)) + " " + rng.choice(list(BAD_WORDS))
    return (x, y_good, y_bad)
```

在真实 RLHF 中，这一步由人类标注员完成。数据形态——`(prompt, preferred_response, rejected_response)`——完全相同。

### 第 2 步：Bradley-Terry 奖励模型

线性打分：`R(x, y) = w · bag(y)`。训练目标是最小化 BT 成对对数损失：

```python
def rm_train_step(w, x, y_pos, y_neg, lr):
    r_pos = dot(w, bag(y_pos))
    r_neg = dot(w, bag(y_neg))
    p = sigmoid(r_pos - r_neg)
    for tok, cnt in bag(y_pos).items():
        w[tok] += lr * (1 - p) * cnt
    for tok, cnt in bag(y_neg).items():
        w[tok] -= lr * (1 - p) * cnt
```

几百次更新之后，`w` 会给好词 token 分配正权重，给坏词 token 分配负权重。

### 第 3 步：在 RM 之上跑类 PPO 策略

我们的玩具策略从词表中生成单个 token。我们用 RM 给这个 token 打分，计算 `log π_θ(token | prompt)`，加上对参考模型的 KL 惩罚，再应用带裁剪的 PPO 替代目标。

```python
def rlhf_step(theta, ref, w, prompt, rng, eps=0.2, beta=0.1, lr=0.05):
    logits_theta = policy_logits(theta, prompt)
    probs = softmax(logits_theta)
    token = sample(probs, rng)
    logits_ref = policy_logits(ref, prompt)
    probs_ref = softmax(logits_ref)
    reward = dot(w, bag([token])) - beta * kl(probs, probs_ref)
    # ppo-style update on theta, treating reward as the return
    ...
```

### 第 4 步：监控 KL

每次更新都追踪平均 `KL(π_θ || π_ref)`。如果它悄悄超过 `~5-10`，说明策略已经大幅偏离 `π_SFT`——要么是有效 `β` 太低，要么奖励黑客已经开始。这是真实 RLHF 中排第一的诊断指标。

### 第 5 步：用 TRL 实现生产配方

理解了玩具流水线之后，下面是真实库用户写出的同一套循环。Hugging Face 的 [TRL](https://huggingface.co/docs/trl) 是参考实现——阶段 2 用 `RewardTrainer`，阶段 3 用 `PPOTrainer`（内置对参考模型的 KL 惩罚）。

```python
# Stage 2: reward model from pairwise preferences
from trl import RewardTrainer, RewardConfig
from transformers import AutoModelForSequenceClassification, AutoTokenizer

tok = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")
rm = AutoModelForSequenceClassification.from_pretrained(
    "meta-llama/Llama-3.1-8B-Instruct", num_labels=1
)

# dataset rows: {"prompt", "chosen", "rejected"} — Bradley-Terry format
trainer = RewardTrainer(
    model=rm,
    tokenizer=tok,
    train_dataset=preference_data,
    args=RewardConfig(output_dir="./rm", num_train_epochs=1, learning_rate=1e-5),
)
trainer.train()
```

```python
# Stage 3: PPO against the RM with KL penalty to the SFT reference
from trl import PPOTrainer, PPOConfig, AutoModelForCausalLMWithValueHead

policy = AutoModelForCausalLMWithValueHead.from_pretrained("./sft-checkpoint")
ref    = AutoModelForCausalLMWithValueHead.from_pretrained("./sft-checkpoint")  # frozen

ppo = PPOTrainer(
    config=PPOConfig(learning_rate=1.41e-5, batch_size=64, init_kl_coef=0.05,
                     target_kl=6.0, adap_kl_ctrl=True),
    model=policy, ref_model=ref, tokenizer=tok,
)

for batch in dataloader:
    responses = ppo.generate(batch["query_ids"], max_new_tokens=128)
    rewards   = rm(torch.cat([batch["query_ids"], responses], dim=-1)).logits[:, 0]
    stats     = ppo.step(batch["query_ids"], responses, rewards)
    # stats includes: mean_kl, clip_frac, value_loss — the three PPO diagnostics
```

库替你做了三件事。`adap_kl_ctrl=True` 实现了自适应 β 调度：观测到的 KL 超过 `target_kl` 时 β 翻倍；低于一半时 β 减半。参考模型按惯例是冻结的——你绝不能不小心让它和 `policy` 共享参数。价值头与策略共用同一个主干（`AutoModelForCausalLMWithValueHead` 会附加一个标量 MLP 头），这也是 TRL 把 `policy/kl` 和 `value/loss` 分开汇报的原因。

## 常见陷阱

- **过度优化 / 奖励黑客。** RM 是不完美的；`π_θ` 会找到得分高但实际很差的对抗性补全。症状：奖励一路上涨，而人工评估分数停滞或下降。解法：提前停止、调高 `β`、扩大 RM 训练数据的覆盖面。
- **长度黑客。** 在有帮助回复上训练的 RM 往往隐式地奖励长度。策略学会给回复灌水。补救：长度归一化的奖励，或使用带长度感知 RM 的 RLAIF。
- **RM 太小。** RM 至少要和策略一样大。一个太小的 RM 无法忠实地给策略的输出打分。
- **KL 调参。** β 太低 → 漂移和奖励黑客。β 太高 → 策略几乎不变。标准技巧是用*自适应* β，使每步的 KL 维持在固定目标值。
- **偏好数据噪声。** 约 30% 的人类标注是有噪声或模棱两可的。可以通过在标注一致性过滤后的数据上训练 RM，或在 BT 上加温度系数来校准。
- **离策略问题。** 第一个 epoch 之后，PPO 的数据就略微离策略了。按第 08 课的方法监控裁剪比例（clip fraction）。

## 生产实践

2026 年的 RLHF 是分层的：

| 层 | 目标 | 方法 |
|-------|--------|--------|
| 指令遵循、有帮助性、无害性 | 对齐 | DPO（Phase 10 · 08）优先于 RLHF-PPO。 |
| 推理正确性（数学、代码） | 能力 | 带验证器奖励的 GRPO（Phase 9 · 12）。 |
| 长程多步任务 | 智能体 | 基于步骤级过程奖励模型的 PPO / GRPO。 |
| 安全 / 拒绝行为 | 安全 | 配独立安全 RM 的 RLHF-PPO，或 Constitutional AI。 |
| 推理时 Best-of-N | 快速对齐 | 在解码时使用 RM；无需训练策略。 |
| 奖励蒸馏 | 推理算力 | 在冻结的 LM 上训练一个小的"奖励头"。 |

RLHF 在 2022–2024 年是*唯一的*主流方法。到 2026 年，生产对齐流水线以 DPO 为先，PPO 只用于重度依赖 RM 或安全攸关的环节。

## 交付产物

保存为 `outputs/skill-rlhf-architect.md`：

```markdown
---
name: rlhf-architect
description: Design an RLHF / DPO / GRPO alignment pipeline for a language model, including RM, KL, and data strategy.
version: 1.0.0
phase: 9
lesson: 9
tags: [rl, rlhf, alignment, llm]
---

Given a base LM, a target behavior (alignment / reasoning / refusal / agent), and a preference or verifier budget, output:

1. Stage. SFT? RM? DPO? GRPO? With justification.
2. Preference or verifier source. Humans, AI feedback, rule-based, unit-test-pass, or reward distillation.
3. KL strategy. Fixed β, adaptive β, or DPO (implicit KL).
4. Diagnostics. Mean KL, reward stability, over-optimization guard (holdout human eval).
5. Safety gate. Red-team set, refusal rate, safety RM separate from helpfulness RM.

Refuse to ship RLHF-PPO without a KL monitor. Refuse to use an RM smaller than the target policy. Refuse length-only rewards. Flag any pipeline that does not hold back a blind human-eval set as lacking over-optimization protection.
```

## 练习

1. **简单。** 用 500 对合成偏好数据训练 `code/main.py` 中的 Bradley-Terry 奖励模型。在留出的 100 对数据上测量成对准确率。应该超过 90%。
2. **中等。** 用 `β ∈ {0.0, 0.1, 1.0}` 跑玩具 PPO-RLHF 循环。对每个取值，画出 RM 分数与相对参考模型 KL 随更新次数的变化曲线。哪些运行出现了奖励黑客？
3. **困难。** 在同样的偏好数据上实现 DPO（闭式偏好似然损失），并与 RLHF-PPO 流水线比较所用算力和最终达到的 RM 分数。

## 关键术语

| 术语 | 大家怎么说 | 实际含义 |
|------|-----------------|-----------------------|
| RLHF | "对齐 RL" | SFT + RM + PPO 三阶段流水线（Christiano 2017, Ouyang 2022）。 |
| 奖励模型（RM） | "打分网络" | 通过 Bradley-Terry 拟合成对偏好的可学习标量函数。 |
| Bradley-Terry | "成对逻辑损失" | `P(y_+ ≻ y_-) = σ(R(y_+) - R(y_-))`；标准 RM 目标函数。 |
| KL 惩罚 | "别离参考模型太远" | 奖励中的 `β · KL(π_θ \|\| π_ref)`；防奖励黑客的正则化项。 |
| 奖励黑客 | "古德哈特定律" | 策略利用 RM 的缺陷；症状：奖励上涨，人工评估持平。 |
| RLAIF | "AI 标注的偏好" | 标签来自另一个 LM 而非人类的 RLHF。 |
| PRM | "过程奖励模型" | 对部分推理步骤打分；用于推理流水线。 |
| Constitutional AI | "Anthropic 的方法" | 由显式规则引导的 AI 生成偏好。 |

## 延伸阅读

- [Christiano et al. (2017). Deep Reinforcement Learning from Human Preferences](https://arxiv.org/abs/1706.03741) —— 开创 RLHF 的论文。
- [Ouyang et al. (2022). InstructGPT — Training language models to follow instructions with human feedback](https://arxiv.org/abs/2203.02155) —— ChatGPT 背后的配方。
- [Stiennon et al. (2020). Learning to summarize with human feedback](https://arxiv.org/abs/2009.01325) —— 更早用于摘要任务的 RLHF。
- [Rafailov et al. (2023). Direct Preference Optimization](https://arxiv.org/abs/2305.18290) —— DPO；2026 年后 RLHF 时代的默认方法。
- [Bai et al. (2022). Constitutional AI: Harmlessness from AI Feedback](https://arxiv.org/abs/2212.08073) —— RLAIF 与自我批评循环。
- [Anthropic RLHF paper (Bai et al. 2022). Training a Helpful and Harmless Assistant](https://arxiv.org/abs/2204.05862) —— HH 论文。
- [Hugging Face TRL library](https://huggingface.co/docs/trl) —— 生产级 `RewardTrainer` 和 `PPOTrainer`。读 trainer 源码可了解自适应 KL 和价值头的细节。
- [Hugging Face — Illustrating Reinforcement Learning from Human Feedback](https://huggingface.co/blog/rlhf)，作者 Lambert、Castricato、von Werra、Havrilla —— 三阶段流水线的权威图解讲解。
- [von Werra et al. (2020). TRL: Transformer Reinforcement Learning](https://github.com/huggingface/trl) —— 库本身；`examples/` 中有面向 Llama、Mistral 和 Qwen 的端到端 RLHF 脚本。
- [Sutton & Barto (2018). Ch. 17.4 — Designing Reward Signals](http://incompleteideas.net/book/RLbook2020.pdf) —— 奖励假说视角；思考奖励黑客问题的必备前置读物。
