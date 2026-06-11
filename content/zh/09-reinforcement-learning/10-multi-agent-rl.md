# 多智能体强化学习

> 单智能体强化学习假设环境是平稳的。把两个正在学习的智能体放进同一个世界，这个假设就被打破了：每个智能体都是对方环境的一部分，而双方都在不断变化。多智能体强化学习（Multi-Agent RL）就是一整套技巧，用来在 Markov 假设不再成立时让学习依然收敛。

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 9 · 04 (Q-learning), Phase 9 · 06 (REINFORCE), Phase 9 · 07 (Actor-Critic)
**Time:** ~45 minutes

## 问题背景

一个机器人学习在房间里导航，这是单智能体强化学习问题。一支足球队就不是了。AlphaStar 对战 StarCraft 对手也不是。一个由竞价智能体组成的市场不是。两辆车在十字路口协商先后通行不是。现实世界里大量多对多的问题都不是。

在任何多智能体场景中，从任意一个智能体的视角看，其他智能体*就是*环境的一部分。随着它们不断学习、改变行为，环境就变得非平稳。Markov 性质——"下一个状态只取决于当前状态和我的动作"——被违反了，因为下一个状态还取决于*其他*智能体的选择，而它们的策略是不断移动的靶子。

这破坏了表格型方法的收敛性证明（Q-learning 的收敛保证以平稳环境为前提）。它也会让朴素的深度强化学习失效：智能体彼此追逐、陷入循环，永远收敛不到稳定策略。你需要多智能体专属的技术：中心化训练/去中心化执行、反事实基线、联赛训练、自我博弈。

2026 年的应用场景：机器人集群、交通路由、自动驾驶车队、市场模拟器、多智能体 LLM 系统（Phase 16），以及任何有不止一个智能玩家的游戏。

## 核心概念

![Four MARL regimes: indep, centralized critic, self-play, league](../assets/marl.svg)

**形式化定义：Markov 博弈（Markov Game）。** 它是 MDP 的推广：状态 `S`、联合动作 `a = (a_1, …, a_n)`、转移 `P(s' | s, a)`，以及每个智能体各自的奖励 `R_i(s, a, s')`。每个智能体 `i` 在自己的策略 `π_i` 下最大化自己的回报。如果所有智能体的奖励完全相同，就是**完全合作**；如果是零和，就是**对抗**；如果两者混合，就是**一般和（general-sum）**。

**核心挑战：**

- **非平稳性。** 从智能体 `i` 的视角看，`P(s' | s, a_i)` 依赖于 `π_{-i}`，而后者一直在变。
- **信用分配。** 奖励是共享的，到底是哪个智能体的功劳？
- **探索协调。** 智能体需要探索互补的策略，而不是重复地探索同一个状态。
- **可扩展性。** 联合动作空间随 `n` 呈指数增长。
- **部分可观测。** 每个智能体只能看到自己的观测，全局状态是隐藏的。

**四种主流范式：**

**1. 独立 Q-learning / 独立 PPO（IQL、IPPO）。** 每个智能体学习自己的 Q 或策略，把其他智能体当作环境的一部分。简单，有时也确实管用（尤其是经验回放在一定程度上起到了平滑对手建模的作用）。理论收敛性：没有。实践中：松耦合任务表现不错，紧耦合任务表现糟糕。

**2. 中心化训练、去中心化执行（CTDE）。** 当今最常见的范式。每个智能体有自己的*策略* `π_i`，只依赖局部观测 `o_i`——部署时是标准的去中心化执行。而在*训练*阶段，一个中心化的 critic `Q(s, a_1, …, a_n)` 以完整的全局状态和联合动作为条件。例子：
- **MADDPG**（Lowe et al. 2017）：DDPG 加上每个智能体一个中心化 critic。
- **COMA**（Foerster et al. 2017）：反事实基线——问"如果我当时改选动作 `a'`，我的奖励会是多少？"——以此分离出我的贡献。
- **MAPPO** / 共享 critic 的 **IPPO**（Yu et al. 2022）：带中心化价值函数的 PPO。2026 年合作型 MARL 的主流选择。
- **QMIX**（Rashid et al. 2018）：价值分解——`Q_tot(s, a) = f(Q_1(s, a_1), …, Q_n(s, a_n))`，混合函数满足单调性。

**3. 自我博弈（Self-play）。** 同一个智能体的两份副本互相对战。对手的策略*就是*我自己过去某个快照的策略。AlphaGo / AlphaZero / MuZero。OpenAI Five。最适合零和博弈，因为训练信号是对称的。

**4. 联赛训练（League play）。** 自我博弈在一般和/对抗环境下的扩展：维护一个由过去和当前策略组成的种群，从联赛中采样对手并与之训练。还会加入 exploiter（专门针对当前最强策略）和 main exploiter（专门针对 exploiter）。AlphaStar（StarCraft II）就是这么做的。当博弈存在"石头剪刀布"式的策略循环时，就需要这种方案。

**通信。** 允许智能体彼此发送可学习的消息 `m_i`。在合作场景中有效。Foerster et al.（2016）证明了可微的智能体间通信可以端到端训练。如今基于 LLM 的多智能体系统（Phase 16）本质上是在用自然语言通信。

## 从零实现

本课使用一个 6×6 的 GridWorld，里面有两个合作的智能体。它们从对角出发，必须到达一个共享目标。共享奖励：只要任一智能体还在移动，每步 `-1`；两者都到达时 `+10`。参见 `code/main.py`。

### 第 1 步：多智能体环境

```python
class CoopGridWorld:
    def __init__(self):
        self.size = 6
        self.goal = (5, 5)

    def reset(self):
        return ((0, 0), (5, 0))  # two agents

    def step(self, state, actions):
        a1, a2 = state
        new1 = move(a1, actions[0])
        new2 = move(a2, actions[1])
        done = (new1 == self.goal) and (new2 == self.goal)
        reward = 10.0 if done else -1.0
        return (new1, new2), reward, done
```

*联合*动作空间是 `|A|² = 16`。全局状态由两个位置构成。

### 第 2 步：独立 Q-learning

每个智能体维护自己的 Q 表，键是联合状态。每一步：双方各自用 ε-贪心选动作，收集联合转移，然后各自用共享奖励更新自己的 Q。

```python
def independent_q(env, episodes, alpha, gamma, epsilon):
    Q1, Q2 = defaultdict(default_q), defaultdict(default_q)
    for _ in range(episodes):
        s = env.reset()
        while not done:
            a1 = epsilon_greedy(Q1, s, epsilon)
            a2 = epsilon_greedy(Q2, s, epsilon)
            s_next, r, done = env.step(s, (a1, a2))
            target1 = r + gamma * max(Q1[s_next].values())
            target2 = r + gamma * max(Q2[s_next].values())
            Q1[s][a1] += alpha * (target1 - Q1[s][a1])
            Q2[s][a2] += alpha * (target2 - Q2[s][a2])
            s = s_next
```

它在这个任务上能跑通，因为奖励既稠密又方向一致。但在紧耦合任务上会失败（比如一个智能体必须*等待*另一个的场景）。

### 第 3 步：中心化 Q 与价值分解式更新

使用一个定义在联合动作上的 Q：`Q(s, a_1, a_2)`，用共享奖励更新。执行时通过边际化来去中心化：`π_i(s) = argmax_{a_i} max_{a_{-i}} Q(s, a_1, a_2)`。代价是指数级的联合动作空间，换来的是一个*正确*的全局视角。

### 第 4 步：简单自我博弈（对抗式双智能体）

同一个智能体，两个角色。让智能体 A 与智能体 B 对战训练；每隔 `K` 个回合，把 A 的权重复制给 B。训练对称、进步稳定。这就是 AlphaZero 配方的微缩版。

## 常见陷阱

- **非平稳的经验回放。** 独立智能体配经验回放比单智能体更糟，因为旧的转移是由如今已过时的对手生成的。对策：重新标注，或按时间新近度加权。
- **信用分配模糊。** 一个长回合结束后才拿到共享奖励，没法说清哪个智能体做了贡献。对策：反事实基线（COMA），或按智能体做奖励塑形。
- **策略漂移/互相追逐。** 每个智能体的最优应对都随对方的更新而变。对策：中心化 critic、放慢学习率，或一次只更新一个、冻结其余。
- **借助协作的奖励投机。** 智能体找到设计者没有预料到的协同漏洞。竞价智能体会集体收敛到出价为零。对策：仔细设计奖励、加行为约束。
- **探索冗余。** 两个智能体探索同样的状态-动作对。对策：按智能体加熵奖励，或用角色条件化。
- **联赛循环。** 纯自我博弈可能困在支配循环里。对策：用多样化对手的联赛训练。
- **样本爆炸。** `n` 个智能体 × 状态空间 × 联合动作。用函数逼近来近似；用分解的动作空间（每个智能体一个策略输出头）。

## 生产实践

2026 年的 MARL 应用地图：

| 领域 | 方法 | 说明 |
|--------|--------|-------|
| 合作导航/操作 | MAPPO / QMIX | CTDE；共享 critic + 去中心化 actor。 |
| 双人博弈（国际象棋、围棋、扑克） | 自我博弈 + MCTS（AlphaZero） | 零和；对称训练。 |
| 复杂多人游戏（Dota、StarCraft） | 联赛训练 + 模仿学习预训练 | OpenAI Five、AlphaStar。 |
| 自动驾驶车队 | CTDE MAPPO / 带注意力的 PPO | 部分可观测；队伍规模可变。 |
| 拍卖市场 | 博弈论均衡 + RL | 当 `n` → ∞ 时用平均场 RL。 |
| LLM 多智能体系统（Phase 16） | 自然语言通信 + 角色条件化 | RL 循环作用于智能体规划层。 |

到 2026 年，MARL 增长最快的领域是基于 LLM 的应用：由语言模型智能体组成的群体进行谈判、辩论、协作开发软件。这里的 RL 表现为对*轨迹级*输出做偏好优化，而不是 token 级（Phase 16 · 03）。

## 交付产物

保存为 `outputs/skill-marl-architect.md`：

```markdown
---
name: marl-architect
description: Pick the right multi-agent RL regime (IPPO, CTDE, self-play, league) for a given task.
version: 1.0.0
phase: 9
lesson: 10
tags: [rl, multi-agent, marl, self-play]
---

Given a task with `n` agents, output:

1. Regime classification. Cooperative / adversarial / general-sum. Justify.
2. Algorithm. IPPO / MAPPO / QMIX / self-play / league. Reason tied to coupling tightness and reward structure.
3. Information access. Centralized training (what global info goes to the critic)? Decentralized execution?
4. Credit assignment. Counterfactual baseline, value decomposition, or reward shaping.
5. Exploration plan. Per-agent entropy, population-based training, or league.

Refuse independent Q-learning on tightly-coupled cooperative tasks. Refuse to recommend self-play for general-sum with cycle risks. Flag any MARL pipeline without a fixed-opponent eval (cherry-picked self-play numbers are common).
```

## 练习

1. **简单。** 在双智能体合作 GridWorld 上训练独立 Q-learning。需要多少个回合平均回报才能 > 0？画出联合学习曲线。
2. **中等。** 增加一个"协调"任务：只有两个智能体在同一回合一起踏上目标格才算到达。独立 Q-learning 还能收敛吗？是什么环节失效了？
3. **困难。** 实现一个 MAPPO 式训练用的中心化 critic，在协调任务上与独立 PPO 比较收敛速度。

## 关键术语

| 术语 | 大家怎么说 | 实际含义 |
|------|-----------------|-----------------------|
| Markov 博弈 | "多智能体 MDP" | `(S, A_1, …, A_n, P, R_1, …, R_n)`；每个智能体有自己的奖励。 |
| CTDE | "中心化训练、去中心化执行" | 训练时用联合 critic；每个智能体的策略只用局部观测。 |
| IPPO | "独立 PPO" | 每个智能体各自跑 PPO。简单的基线，常被低估。 |
| MAPPO | "多智能体 PPO" | 带以全局状态为条件的中心化价值函数的 PPO。 |
| QMIX | "单调价值分解" | `Q_tot = f_monotone(Q_1, …, Q_n)`，使去中心化的 argmax 成为可能。 |
| COMA | "反事实多智能体" | 优势 = 我的 Q 减去对我的动作做边际化后的期望 Q。 |
| 自我博弈 | "智能体对战过去的自己" | 单个智能体扮演两个角色；零和博弈的标准做法。 |
| 联赛训练 | "种群训练" | 缓存历史策略，从池子里采样对手；解决策略循环问题。 |

## 延伸阅读

- [Lowe et al. (2017). Multi-Agent Actor-Critic for Mixed Cooperative-Competitive Environments (MADDPG)](https://arxiv.org/abs/1706.02275) — 带中心化 critic 的 CTDE。
- [Foerster et al. (2017). Counterfactual Multi-Agent Policy Gradients (COMA)](https://arxiv.org/abs/1705.08926) — 用反事实基线解决信用分配。
- [Rashid et al. (2018). QMIX: Monotonic Value Function Factorisation](https://arxiv.org/abs/1803.11485) — 带单调性约束的价值分解。
- [Yu et al. (2022). The Surprising Effectiveness of PPO in Cooperative Multi-Agent Games (MAPPO)](https://arxiv.org/abs/2103.01955) — PPO 在 MARL 上意外地强。
- [Vinyals et al. (2019). Grandmaster level in StarCraft II using multi-agent reinforcement learning (AlphaStar)](https://www.nature.com/articles/s41586-019-1724-z) — 大规模联赛训练。
- [Silver et al. (2017). Mastering the game of Go without human knowledge (AlphaGo Zero)](https://www.nature.com/articles/nature24270) — 零和博弈中的纯自我博弈。
- [Sutton & Barto (2018). Ch. 15 — Neuroscience & Ch. 17 — Frontiers](http://incompleteideas.net/book/RLbook2020.pdf) — 包含教材对多智能体场景的简短论述，以及 CTDE 旨在解决的非平稳性问题。
- [Zhang, Yang & Başar (2021). Multi-Agent Reinforcement Learning: A Selective Overview](https://arxiv.org/abs/1911.10635) — 覆盖合作、竞争与混合 MARL 及其收敛性结果的综述。
