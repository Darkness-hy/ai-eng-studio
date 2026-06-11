# 近端策略优化（Proximal Policy Optimization, PPO）

> A2C 每次更新后就把整条 rollout 丢弃。PPO 用一个带裁剪的重要性比率包装策略梯度，让你可以在同一批数据上跑 10 个以上的 epoch，而策略不会爆炸。Schulman et al. (2017)。到 2026 年仍是默认的策略梯度算法。

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 9 · 06 (REINFORCE), Phase 9 · 07 (Actor-Critic)
**Time:** ~75 minutes

## 问题背景

A2C（第 07 课）是 on-policy 的：梯度 `E_{π_θ}[A · ∇ log π_θ]` 要求数据必须采样自*当前的* `π_θ`。做一次更新后，`π_θ` 就变了；你刚用过的数据已经变成 off-policy。再拿来用，梯度就有偏。

而 rollout 的代价很高。在 Atari 上，一次 rollout 跨 8 个环境 × 128 步 = 1024 条 transition，加上十几秒的环境运行时间。只做一步梯度更新就把它扔掉，太浪费了。

信赖域策略优化（Trust Region Policy Optimization，TRPO，Schulman 2015）是第一个解决方案：约束每次更新，使新旧策略之间的 KL 散度保持在 `δ` 以下。理论上很干净，但每次更新都要做一次共轭梯度求解。2026 年已经没人跑 TRPO 了。

PPO（Schulman et al. 2017）用一个简单的裁剪目标函数取代了硬性的信赖域约束。多写一行代码。每条 rollout 跑十个 epoch。不需要共轭梯度。理论保证足够好。九年过去了，从 MuJoCo 到 RLHF，它仍然是默认的策略梯度算法。

## 核心概念

![PPO clipped surrogate objective: ratio clipping at 1 ± ε](../assets/ppo.svg)

**重要性比率（importance ratio）。**

`r_t(θ) = π_θ(a_t | s_t) / π_{θ_old}(a_t | s_t)`

这是新策略相对于采集数据时的旧策略的似然比。`r_t = 1` 表示没有变化。`r_t = 2` 表示新策略采取 `a_t` 的概率是旧策略的两倍。

**裁剪代理目标（clipped surrogate）。**

`L^{CLIP}(θ) = E_t [ min( r_t(θ) A_t, clip(r_t(θ), 1-ε, 1+ε) A_t ) ]`

两个分项：

- 如果优势 `A_t > 0` 且比率试图涨过 `1 + ε`，裁剪会把梯度压平——不要把一个好动作的概率推到比旧概率高出 `+ε` 以上。
- 如果优势 `A_t < 0` 且比率试图越过 `1 - ε`（意味着相比裁剪后的降幅，我们会让一个坏动作变得更可能），裁剪会封住梯度——不要把一个坏动作压到 `-ε` 以下。

`min` 处理另一个方向：如果比率朝着*有利的*方向移动了，你仍然能拿到梯度（在会让你受损的那一侧不做裁剪）。

通常取 `ε = 0.2`。把目标函数画成 `r_t` 的函数：一条分段线性曲线，在"好的一侧"有平顶，在"坏的一侧"有平底。

**完整的 PPO 损失。**

`L(θ, φ) = L^{CLIP}(θ) - c_v · (V_φ(s_t) - V_t^{target})² + c_e · H(π_θ(·|s_t))`

与 A2C 相同的 actor-critic 结构。三个系数，通常取 `c_v = 0.5`、`c_e = 0.01`、`ε = 0.2`。

**训练循环。**

1. 在 `N` 个并行环境中各采集 `T` 步，共 `N × T` 条 transition。
2. 计算优势（GAE），将其冻结为常量。
3. 把当前 `π_θ` 的快照冻结为 `π_{θ_old}`。
4. 跑 `K` 个 epoch，对每个 `(s, a, A, V_target, log π_old(a|s))` 小批量：
   - 计算 `r_t(θ) = exp(log π_θ(a|s) - log π_old(a|s))`。
   - 应用 `L^{CLIP}` + 价值损失 + 熵项。
   - 做一步梯度更新。
5. 丢弃这条 rollout，回到第 1 步。

`K = 10`、小批量大小 64 是一组标准超参数。PPO 很鲁棒：具体数值在 ±50% 范围内变动通常无关紧要。

**KL 惩罚变体。**原论文还提出了一种使用自适应 KL 惩罚的替代方案：`L = L^{PG} - β · KL(π_θ || π_old)`，其中 `β` 根据观测到的 KL 动态调整。后来裁剪版本成为主流；KL 变体则在 RLHF 中存活下来（在 RLHF 中，对参考策略的 KL 本来就是你始终需要的一个独立约束）。

## 从零实现

### 第 1 步：在 rollout 时记录 `log π_old(a | s)`

```python
for step in range(T):
    probs = softmax(logits(theta, state_features(s)))
    a = sample(probs, rng)
    s_next, r, done = env.step(s, a)
    buffer.append({
        "s": s, "a": a, "r": r, "done": done,
        "v_old": value(w, state_features(s)),
        "log_pi_old": log(probs[a] + 1e-12),
    })
    s = s_next
```

快照只在 rollout 时取一次，在更新的各个 epoch 中保持不变。

### 第 2 步：计算 GAE 优势（第 07 课）

与 A2C 相同。在整个批次上做归一化。

### 第 3 步：裁剪代理目标更新

```python
for _ in range(K_EPOCHS):
    for mb in minibatches(buffer, size=64):
        for rec in mb:
            x = state_features(rec["s"])
            probs = softmax(logits(theta, x))
            logp = log(probs[rec["a"]] + 1e-12)
            ratio = exp(logp - rec["log_pi_old"])
            adv = rec["advantage"]
            surrogate = min(
                ratio * adv,
                clamp(ratio, 1 - EPS, 1 + EPS) * adv,
            )
            # backprop -surrogate, add value loss, subtract entropy
            grad_logpi = onehot(rec["a"]) - probs
            if (adv > 0 and ratio >= 1 + EPS) or (adv < 0 and ratio <= 1 - EPS):
                pg_grad = 0.0  # clipped
            else:
                pg_grad = ratio * adv
            for i in range(N_ACTIONS):
                for j in range(N_FEAT):
                    theta[i][j] += LR * pg_grad * grad_logpi[i] * x[j]
```

"被裁剪 → 梯度归零"这一模式是 PPO 的核心。如果新策略已经在有利方向上漂移得太远，更新就停止。

### 第 4 步：价值与熵

给 critic 目标加上标准 MSE，给 actor 加上熵奖励，做法与 A2C 相同。

### 第 5 步：诊断指标

每次更新要盯住三样东西：

- **平均 KL** `E[log π_old - log π_θ]`。应保持在 `[0, 0.02]` 之间。如果冲过 `0.1`，就减小 `K_EPOCHS` 或 `LR`。
- **裁剪比例（clip fraction）**——比率落在 `[1-ε, 1+ε]` 之外的样本所占比例。应该在 `~0.1-0.3`。如果接近 `~0`，说明裁剪从未触发 → 提高 `LR` 或 `K_EPOCHS`。如果达到 `~0.5+`，说明你在过拟合这条 rollout → 把它们调低。
- **解释方差（explained variance）** `1 - Var(V_target - V_pred) / Var(V_target)`。衡量 critic 质量的指标。随着 critic 学习，应当向 1 攀升。

## 常见陷阱

- **裁剪系数调错。**`ε = 0.2` 是事实上的标准。降到 `0.1` 会让更新过于胆怯；`0.3+` 则招致不稳定。
- **epoch 太多。**`K > 20` 经常导致不稳定，因为策略会漂离 `π_old` 太远。要给 epoch 设上限，对大网络尤其如此。
- **没做奖励归一化。**奖励尺度过大会侵占裁剪区间。在计算优势之前先对奖励做归一化（滑动标准差）。
- **忘记优势归一化。**逐批次的零均值/单位标准差归一化是标准做法。跳过它会在大多数基准上毁掉 PPO。
- **学习率不衰减。**PPO 受益于线性衰减到零的学习率。恒定学习率往往效果更差。
- **重要性比率的数值计算错误。**为了数值稳定性，永远用 `exp(log_new - log_old)`，而不是 `new / old`。
- **梯度符号写反。**最大化代理目标 = *最小化* `-L^{CLIP}`。符号写反是最常见的 PPO bug。

## 生产实践

PPO 是 2026 年覆盖领域多到惊人的默认 RL 算法：

| 用例 | PPO 变体 |
|----------|-------------|
| MuJoCo / 机器人控制 | 高斯策略的 PPO，GAE(0.95) |
| Atari / 离散动作游戏 | 类别分布策略的 PPO，滚动式 128 步 rollout |
| LLM 的 RLHF | 对参考模型加 KL 惩罚的 PPO，奖励由 RM 在回复末尾给出 |
| 大规模游戏智能体 | IMPALA + PPO（AlphaStar、OpenAI Five） |
| 推理型 LLM | GRPO（第 12 课）——无 critic 的 PPO 变体 |
| 仅有偏好数据 | DPO——把 PPO+KL 闭式坍缩，无需在线采样 |

PPO 的*损失结构*——裁剪代理目标 + 价值 + 熵——是 DPO、GRPO 以及几乎所有 RLHF 流水线的脚手架。

## 交付产物

保存为 `outputs/skill-ppo-trainer.md`：

```markdown
---
name: ppo-trainer
description: Produce a PPO training config and a diagnostic plan for a given environment.
version: 1.0.0
phase: 9
lesson: 8
tags: [rl, ppo, policy-gradient]
---

Given an environment and training budget, output:

1. Rollout size. `N` envs × `T` steps.
2. Update schedule. `K` epochs, minibatch size, LR schedule.
3. Surrogate params. `ε` (clip), `c_v`, `c_e`, advantage normalization on.
4. Advantage. GAE(`λ`) with explicit `γ` and `λ`.
5. Diagnostics plan. KL, clip fraction, explained variance thresholds with alerts.

Refuse `K > 30` or `ε > 0.3` (unsafe trust region). Refuse any PPO run without advantage normalization or KL/clip monitoring. Flag clip fraction sustained above 0.4 as drift.
```

## 练习

1. **简单。**在 4×4 GridWorld 上用 `ε=0.2, K=4` 跑 PPO。在相同环境步数下，与 A2C（每条 rollout 只跑一个 epoch）比较样本效率。
2. **中等。**扫描 `K ∈ {1, 4, 10, 30}`。绘制回报 vs 环境步数的曲线，并跟踪每次更新的平均 KL。在这个任务上，KL 在哪个 `K` 时开始爆炸？
3. **困难。**把裁剪代理目标换成自适应 KL 惩罚（`KL > 2·target` 时 `β` 翻倍，`KL < target/2` 时减半）。比较最终回报、稳定性，以及不依赖裁剪的程度。

## 关键术语

| 术语 | 人们怎么说 | 它实际指什么 |
|------|-----------------|-----------------------|
| 重要性比率 | "r_t(θ)" | `π_θ(a\|s) / π_old(a\|s)`；衡量相对采集数据的策略的偏离程度。 |
| 裁剪代理目标 | "PPO 的核心技巧" | `min(r·A, clip(r, 1-ε, 1+ε)·A)`；在有利一侧超出裁剪范围时梯度为零。 |
| 信赖域 | "TRPO / PPO 的本意" | 限制每次更新的 KL，以保证单调改进。 |
| KL 惩罚 | "软信赖域" | PPO 的另一种形式：`L - β · KL(π_θ \|\| π_old)`。自适应 `β`。 |
| 裁剪比例 | "裁剪触发的频率" | 诊断指标——应在 0.1-0.3 之间；超出说明超参数调错了。 |
| 多 epoch 训练 | "数据复用" | 每条 rollout 跑 K 个 epoch；用方差代价换取样本效率。 |
| 准 on-policy | "基本算 on-policy" | PPO 名义上是 on-policy，但 K>1 的 epoch 安全地使用了轻微 off-policy 的数据。 |
| PPO-KL | "另一个 PPO" | KL 惩罚变体；用于 RLHF，那里对参考模型的 KL 本来就是约束。 |

## 延伸阅读

- [Schulman et al. (2017). Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347) —— 原论文。
- [Schulman et al. (2015). Trust Region Policy Optimization](https://arxiv.org/abs/1502.05477) —— TRPO，PPO 的前身。
- [Andrychowicz et al. (2021). What Matters In On-Policy RL? A Large-Scale Empirical Study](https://arxiv.org/abs/2006.05990) —— 对每个 PPO 超参数都做了消融。
- [Ouyang et al. (2022). Training language models to follow instructions with human feedback](https://arxiv.org/abs/2203.02155) —— InstructGPT；RLHF 中使用 PPO 的配方。
- [OpenAI Spinning Up — PPO](https://spinningup.openai.com/en/latest/algorithms/ppo.html) —— 清晰的现代讲解，附 PyTorch 实现。
- [CleanRL PPO implementation](https://github.com/vwxyzjn/cleanrl) —— 被许多论文引用的单文件 PPO 参考实现。
- [Hugging Face TRL — PPOTrainer](https://huggingface.co/docs/trl/main/en/ppo_trainer) —— 在语言模型上跑 PPO 的生产级配方；建议与第 09 课（RLHF）一起阅读。
- [Engstrom et al. (2020). Implementation Matters in Deep Policy Gradients](https://arxiv.org/abs/2005.12729) —— 那篇"37 个代码层面优化"的论文；告诉你哪些 PPO 技巧是真正起作用的，哪些只是以讹传讹。
