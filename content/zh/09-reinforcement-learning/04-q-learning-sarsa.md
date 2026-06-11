# 时序差分 — Q-Learning 与 SARSA

> 蒙特卡洛要等到回合结束才更新，TD 则通过自举（bootstrapping）下一个价值估计，在每一步之后就更新。Q-learning 是离策略的、乐观的；SARSA 是同策略的、谨慎的。两者都只有一行代码，也都是本阶段所有深度强化学习方法的基石。

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 9 · 01 (MDPs), Phase 9 · 02 (Dynamic Programming), Phase 9 · 03 (Monte Carlo)
**Time:** ~75 minutes

## 问题背景

蒙特卡洛方法可行，但它有两个昂贵的要求：回合必须终止，而且只有在最终回报到手后才能更新。如果一个回合有 1,000 步，MC 就要等 1,000 步才能更新任何东西。它是高方差、低偏差的，在实践中很慢。

动态规划的特性正好相反——零方差的自举回溯——但需要已知的环境模型。

时序差分（Temporal Difference, TD）学习取两者之间的折中。从单个转移 `(s, a, r, s')` 出发，构造一个单步目标 `r + γ V(s')`，并把 `V(s)` 朝它推一小步。不需要模型，不需要完整回合。由于右侧使用了近似的 `V` 会引入偏差，但方差远低于 MC，并且从第一步起就能在线更新。

这是整个现代强化学习——DQN、A2C、PPO、SAC——赖以运转的支点。Phase 9 余下的内容，都是在本课你将亲手写出的单步 TD 更新之上，层层叠加函数近似和各种技巧。

## 核心概念

![Q-learning vs SARSA: off-policy max vs on-policy Q(s', a')](../assets/td.svg)

**针对 V 的 TD(0) 更新：**

`V(s) ← V(s) + α [r + γ V(s') - V(s)]`

方括号里的量就是 TD 误差 `δ = r + γ V(s') - V(s)`。它是 MC 中 `G_t - V(s_t)` 的在线版本。收敛性要求 `α` 满足 Robbins-Monro 条件（`Σ α = ∞`，`Σ α² < ∞`），且所有状态被无限次访问。

**Q-learning。** 一种用于控制问题的离策略（off-policy）TD 方法：

`Q(s, a) ← Q(s, a) + α [r + γ max_{a'} Q(s', a') - Q(s, a)]`

这个 `max` 假设从 `s'` 起将一直遵循*贪心*策略，而不管智能体实际采取什么动作。正是这种解耦让 Q-learning 在智能体通过 ε-greedy 探索的同时学到 `Q*`。Mnih 等人（2015）将它扩展成了 Atari 上的深度 Q-learning（第 05 课）。

**SARSA。** 一种同策略（on-policy）TD 方法：

`Q(s, a) ← Q(s, a) + α [r + γ Q(s', a') - Q(s, a)]`

名字来自五元组 `(s, a, r, s', a')`。SARSA 使用智能体下一步*实际*采取的动作 `a'`，而不是贪心的 `argmax`。它收敛到当前运行的那个 ε-greedy 策略 `π` 对应的 `Q^π`，在 `ε → 0` 的极限下就变成 `Q*`。

**悬崖行走的差异。** 在经典的悬崖行走任务中（掉下悬崖 = 奖励 -100），Q-learning 学到沿崖边的最优路径，但探索时偶尔会吃到惩罚。SARSA 学到一条离悬崖一步之遥的更安全路径，因为它把探索噪声计入了自己的 Q 值。随着训练进行，在 `ε → 0` 时两者都能达到最优。实践中这个差异很重要：当部署时确实存在探索行为，SARSA 的表现更保守。

**Expected SARSA。** 把 `Q(s', a')` 替换为它在 `π` 下的期望值：

`Q(s, a) ← Q(s, a) + α [r + γ Σ_{a'} π(a'|s') Q(s', a') - Q(s, a)]`

方差比 SARSA 更低（不需要采样 `a'`），同策略目标不变。在现代教科书中往往是默认选择。

**n 步 TD 与 TD(λ)。** 通过等待 `n` 步再自举，在 TD(0) 和 MC 之间插值。`n=1` 是 TD，`n=∞` 是 MC。TD(λ) 用几何权重 `(1-λ)λ^{n-1}` 对所有 `n` 取平均。大多数深度强化学习使用 3 到 20 之间的 `n`。

```figure
qlearning-gridworld
```

## 从零实现

### 第 1 步：基于 ε-greedy 策略的 SARSA

```python
def sarsa(env, episodes, alpha=0.1, gamma=0.99, epsilon=0.1):
    Q = defaultdict(lambda: {a: 0.0 for a in ACTIONS})

    def choose(s):
        if random() < epsilon:
            return choice(ACTIONS)
        return max(Q[s], key=Q[s].get)

    for _ in range(episodes):
        s = env.reset()
        a = choose(s)
        while True:
            s_next, r, done = env.step(s, a)
            a_next = choose(s_next) if not done else None
            target = r + (gamma * Q[s_next][a_next] if not done else 0.0)
            Q[s][a] += alpha * (target - Q[s][a])
            if done:
                break
            s, a = s_next, a_next
    return Q
```

八行代码。与 Q-learning 的*唯一*区别就在 target 那一行。

### 第 2 步：Q-learning

```python
def q_learning(env, episodes, alpha=0.1, gamma=0.99, epsilon=0.1):
    Q = defaultdict(lambda: {a: 0.0 for a in ACTIONS})
    for _ in range(episodes):
        s = env.reset()
        while True:
            a = choose(s, Q, epsilon)
            s_next, r, done = env.step(s, a)
            target = r + (gamma * max(Q[s_next].values()) if not done else 0.0)
            Q[s][a] += alpha * (target - Q[s][a])
            if done:
                break
            s = s_next
    return Q
```

这个 `max` 把目标和行为解耦。这一个符号就是同策略与离策略的全部区别。

### 第 3 步：学习曲线

跟踪每 100 个回合的平均回报。在简单的确定性 GridWorld 上 Q-learning 收敛更快；在悬崖行走上 SARSA 更保守。在 `code/main.py` 的 4×4 GridWorld 上，使用 `α=0.1, ε=0.1`，两者在约 2,000 个回合后都接近最优。

### 第 4 步：与 DP 真值对比

运行价值迭代（第 02 课）得到 `Q*`。检查 `max_{s,a} |Q_learned(s,a) - Q*(s,a)|`。一个健康的表格型 TD 智能体在 4×4 GridWorld 上训练 10,000 个回合后，误差应落在 `~0.5` 以内。

## 常见陷阱

- **Q 值初始化很重要。** 乐观初始化（对负奖励任务设 `Q = 0`）会鼓励探索。悲观初始化可能让贪心策略永远陷入局部。
- **α 调度。** 对非平稳问题，常数 `α` 就够用。衰减式 `α_n = 1/n` 理论上保证收敛，但实践中太慢——把 `α` 固定在 `[0.05, 0.3]` 并盯着学习曲线。
- **ε 调度。** 从高起点开始（`ε=1.0`），衰减到 `ε=0.05`。"GLIE"（极限贪心且无限探索）是收敛条件。
- **Q-learning 的最大化偏差。** 当 `Q` 含噪时，`max` 算子有向上的偏差，导致高估——Hasselt 的 Double Q-learning（第 05 课的 DDQN 所采用）用两张 Q 表修复了这个问题。
- **不终止的回合。** TD 可以在没有终止状态的情况下学习，但你需要限制步数上限，或在到达上限时正确处理自举。标准做法：把步数上限视为非终止状态，继续自举。
- **状态哈希。** 如果状态是元组/张量，要使用可哈希的键（用 tuple 而非 list；用取整后的浮点数元组而非原始值）。

## 生产实践

2026 年的 TD 全景图：

| 任务 | 方法 | 原因 |
|------|--------|--------|
| 小型表格环境 | Q-learning | 直接学习最优策略。 |
| 同策略、安全攸关 | SARSA / Expected SARSA | 探索期间更保守。 |
| 高维状态 | DQN（Phase 9 · 05） | 带经验回放和目标网络的神经网络 Q 函数。 |
| 连续动作 | SAC / TD3（Phase 9 · 07） | 在 Q 网络上做 TD 更新；策略网络输出动作。 |
| LLM 强化学习（基于奖励模型） | PPO / GRPO（Phase 9 · 08, 12） | Actor-critic，通过 GAE 计算 TD 式优势。 |
| 离线强化学习 | CQL / IQL（Phase 9 · 08） | 带保守正则化的 Q-learning。 |

2026 年论文里你读到的"RL"，九成都是 Q-learning 或 SARSA 的某种延伸。在深入阅读之前，先把表格型更新练到熟稔于心。

## 交付产物

保存为 `outputs/skill-td-agent.md`：

```markdown
---
name: td-agent
description: Pick between Q-learning, SARSA, Expected SARSA for a tabular or small-feature RL task.
version: 1.0.0
phase: 9
lesson: 4
tags: [rl, td-learning, q-learning, sarsa]
---

Given a tabular or small-feature environment, output:

1. Algorithm. Q-learning / SARSA / Expected SARSA / n-step variant. One-sentence reason tied to on-policy vs off-policy and variance.
2. Hyperparameters. α, γ, ε, decay schedule.
3. Initialization. Q_0 value (optimistic vs zero) and justification.
4. Convergence diagnostic. Target learning curve, `|Q - Q*|` check if DP is possible.
5. Deployment caveat. How will exploration behave at inference? Is SARSA's conservatism needed?

Refuse to apply tabular TD to state spaces > 10⁶. Refuse to ship a Q-learning agent without a max-bias caveat. Flag any agent trained with ε held at 1.0 throughout (no exploitation phase).
```

## 练习

1. **简单。** 在 4×4 GridWorld 上实现 Q-learning 和 SARSA。绘制 2,000 个回合的学习曲线（每 100 个回合的平均回报）。谁收敛更快？
2. **中等。** 构建一个悬崖行走环境（4×12，最后一行是悬崖，奖励 -100 并重置回起点）。比较 Q-learning 和 SARSA 的最终策略。截图记录各自走的路径。哪个离悬崖更近？
3. **困难。** 实现 Double Q-learning。在一个带噪声奖励的 GridWorld 上（每步奖励叠加 σ=5 的高斯噪声），证明 Q-learning 会明显高估 `V*(0,0)`，而 Double Q-learning 不会。

## 关键术语

| 术语 | 通常的说法 | 实际含义 |
|------|-----------------|-----------------------|
| TD 误差 | "更新信号" | `δ = r + γ V(s') - V(s)`，自举残差。 |
| TD(0) | "单步 TD" | 每次转移后只用下一状态的估计值进行更新。 |
| Q-learning | "离策略 RL 入门第一课" | 对下一状态动作取 `max` 的 TD 更新；无论行为策略如何都能学到 `Q*`。 |
| SARSA | "同策略版 Q-learning" | 使用实际下一动作的 TD 更新；学到当前 ε-greedy 策略 π 对应的 `Q^π`。 |
| Expected SARSA | "低方差版 SARSA" | 把采样的 `a'` 替换为它在 π 下的期望。 |
| GLIE | "正确的探索调度" | 极限贪心且无限探索（Greedy in the Limit with Infinite Exploration）；Q-learning 收敛的必要条件。 |
| 自举（Bootstrapping） | "在目标中使用当前估计" | TD 与 MC 的本质区别。偏差的来源，但带来巨大的方差缩减。 |
| 最大化偏差 | "Q-learning 会高估" | 对含噪估计取 `max` 有向上偏差；由 Double Q-learning 修复。 |

## 延伸阅读

- [Watkins & Dayan (1992). Q-learning](https://link.springer.com/article/10.1007/BF00992698) — 原始论文与收敛性证明。
- [Sutton & Barto (2018). Ch. 6 — Temporal-Difference Learning](http://incompleteideas.net/book/RLbook2020.pdf) — TD(0)、SARSA、Q-learning、Expected SARSA。
- [Hasselt (2010). Double Q-learning](https://papers.nips.cc/paper_files/paper/2010/hash/091d584fced301b442654dd8c23b3fc9-Abstract.html) — 最大化偏差的修复方案。
- [Seijen, Hasselt, Whiteson, Wiering (2009). A Theoretical and Empirical Analysis of Expected SARSA](https://ieeexplore.ieee.org/document/4927542) — Expected SARSA 的动机。
- [Rummery & Niranjan (1994). On-line Q-learning using connectionist systems](https://www.researchgate.net/publication/2500611_On-Line_Q-Learning_Using_Connectionist_Systems) — 提出 SARSA 的论文（当时称为 "modified connectionist Q-learning"）。
- [Sutton & Barto (2018). Ch. 7 — n-step Bootstrapping](http://incompleteideas.net/book/RLbook2020.pdf) — 把 TD(0) 推广到 TD(n)，也是从 Q-learning 通向资格迹、再到 PPO 中 GAE 的路径。
