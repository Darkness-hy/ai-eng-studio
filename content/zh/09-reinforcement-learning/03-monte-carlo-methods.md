# 蒙特卡洛方法 — 从完整回合中学习

> 动态规划需要模型，蒙特卡洛只需要回合。运行策略，观察回报，取平均。这是强化学习中最简单的思想——也是解锁后续一切的那一个。

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 9 · 01 (MDPs), Phase 9 · 02 (Dynamic Programming)
**Time:** ~75 minutes

## 问题背景

动态规划很优雅，但它假设你可以对每个状态和动作查询 `P(s' | s, a)`。现实世界几乎没有任何东西是这样运作的。机器人无法在施加关节力矩后解析地算出摄像头像素的分布；定价算法无法对每一种可能的客户反应做积分；LLM 无法枚举一个 token 之后的所有可能续写。

你需要一种只依赖从环境中*采样*的方法。运行策略，得到一条轨迹 `s_0, a_0, r_1, s_1, a_1, r_2, …, s_T`，用它来估计价值。这就是蒙特卡洛（Monte Carlo）。

从 DP 到 MC 的转变在思想上非常重要：我们从*已知模型 + 精确备份*转向*采样回合 + 平均回报*。方差随之上升，但适用范围爆炸式扩大。本课之后的每一个强化学习算法——TD、Q-learning、REINFORCE、PPO、GRPO——本质上都是蒙特卡洛估计器，有时只是在上面叠加了自举（bootstrapping）。

## 核心概念

![Monte Carlo: rollout, compute returns, average; first-visit vs every-visit](../assets/monte-carlo.svg)

**核心思想，一行写完：** `V^π(s) = E_π[G_t | s_t = s] ≈ (1/N) Σ_i G^{(i)}(s)`，其中 `G^{(i)}(s)` 是在策略 `π` 下访问状态 `s` 之后观测到的回报。

**首次访问 vs 每次访问 MC。** 如果一个回合多次访问状态 `s`，首次访问（first-visit）MC 只统计第一次访问之后的回报；每次访问（every-visit）MC 统计所有访问。两者在极限意义下都是无偏的。首次访问更容易分析（样本独立同分布）；每次访问对每个回合利用了更多数据，实践中通常收敛更快。

**增量均值。** 不必存储所有回报，改为更新运行平均值：

`V_n(s) = V_{n-1}(s) + (1/n) [G_n - V_{n-1}(s)]`

整理一下：`V_new = V_old + α · (target - V_old)`，其中 `α = 1/n`。把 `1/n` 换成常数步长 `α ∈ (0, 1)`，你就得到了一个能跟踪 `π` 变化的非平稳 MC 估计器。这一步就是从 MC 到 TD、再到所有现代强化学习算法的全部跳跃。

**探索成了一个问题。** DP 通过枚举触及每一个状态，而 MC 只能看到策略实际访问的状态。如果 `π` 是确定性的，状态空间的整片区域永远不会被采样到，它们的价值估计将永远停留在零。三种修复方式，按历史顺序：

1. **探索性起点（exploring starts）。** 每个回合从随机的 (s, a) 对开始。保证覆盖；但在实践中不现实（你不能把机器人"重置"到任意状态）。
2. **ε-greedy。** 相对于当前 Q 贪心地行动，但以概率 `ε` 选一个随机动作。所有状态-动作对在渐近意义上都会被采样到。
3. **离策略（off-policy）MC。** 在行为策略 `μ` 下收集数据，通过重要性采样（importance sampling）学习目标策略 `π`。方差高，但它是通向 DQN 这类经验回放（replay buffer）方法的桥梁。

**蒙特卡洛控制。** 评估 → 改进 → 评估，和策略迭代一样，但评估基于采样：

1. 运行 `π`，得到一个回合。
2. 用观测到的回报更新 `Q(s, a)`。
3. 让 `π` 相对于 `Q` 变为 ε-greedy。
4. 重复。

在温和条件下（每个状态-动作对被访问无穷多次，`α` 满足 Robbins-Monro 条件），以概率 1 收敛到 `Q*` 和 `π*`。

```figure
epsilon-greedy
```

## 从零实现

### 第 1 步：rollout → (s, a, r) 列表

```python
def rollout(env, policy, max_steps=200):
    trajectory = []
    s = env.reset()
    for _ in range(max_steps):
        a = policy(s)
        s_next, r, done = env.step(s, a)
        trajectory.append((s, a, r))
        s = s_next
        if done:
            break
    return trajectory
```

不需要模型，只需要 `env.reset()` 和 `env.step(s, a)`。和 gym 环境的接口相同，只是做了精简。

### 第 2 步：计算回报（反向扫描）

```python
def returns_from(trajectory, gamma):
    returns = []
    G = 0.0
    for _, _, r in reversed(trajectory):
        G = r + gamma * G
        returns.append(G)
    return list(reversed(returns))
```

单次遍历，`O(T)`。反向递推式 `G_t = r_{t+1} + γ G_{t+1}` 避免了重复求和。

### 第 3 步：首次访问 MC 评估

```python
def mc_policy_evaluation(env, policy, episodes, gamma=0.99):
    V = defaultdict(float)
    counts = defaultdict(int)
    for _ in range(episodes):
        trajectory = rollout(env, policy)
        returns = returns_from(trajectory, gamma)
        seen = set()
        for t, ((s, _, _), G) in enumerate(zip(trajectory, returns)):
            if s in seen:
                continue
            seen.add(s)
            counts[s] += 1
            V[s] += (G - V[s]) / counts[s]
    return V
```

核心工作只有三行：首次访问时把状态标记为已见、递增计数、更新运行均值。

### 第 4 步：ε-greedy MC 控制（同策略）

```python
def mc_control(env, episodes, gamma=0.99, epsilon=0.1):
    Q = defaultdict(lambda: {a: 0.0 for a in ACTIONS})
    counts = defaultdict(lambda: {a: 0 for a in ACTIONS})

    def policy(s):
        if random() < epsilon:
            return choice(ACTIONS)
        return max(Q[s], key=Q[s].get)

    for _ in range(episodes):
        trajectory = rollout(env, policy)
        returns = returns_from(trajectory, gamma)
        seen = set()
        for (s, a, _), G in zip(trajectory, returns):
            if (s, a) in seen:
                continue
            seen.add((s, a))
            counts[s][a] += 1
            Q[s][a] += (G - Q[s][a]) / counts[s][a]
    return Q, policy
```

### 第 5 步：与 DP 黄金标准对比

随着回合数 → ∞，你对 `V^π` 的 MC 估计应当与第 02 课的 DP 结果一致。实践中：在 4×4 GridWorld 上跑 50,000 个回合，结果与 DP 答案的差距在 `~0.1` 以内。

## 常见陷阱

- **无限回合。** MC 要求回合必须*终止*。如果你的策略可能永远循环，设置 `max_steps` 上限，并把触顶视为隐式失败。GridWorld 配上随机策略经常超时——这很正常，只要确保你正确地统计了它。
- **方差。** MC 使用完整回报。在长回合上方差巨大——结尾处一个倒霉的奖励会让 `V(s_0)` 偏移同样的量。TD 方法（第 04 课）通过自举来削减这一点。
- **状态覆盖。** 在全新的 Q 上做贪心 MC，遇到并列值时永远只会尝试一个动作。你*必须*探索（ε-greedy、探索性起点、UCB）。
- **非平稳策略。** 如果 `π` 在变化（如 MC 控制中那样），旧的回报来自不同的策略。常数 α 的 MC 能处理这一点；样本平均 MC 则不能。
- **离策略重要性采样。** 权重 `π(a|s)/μ(a|s)` 沿轨迹连乘，方差随时间跨度爆炸。用逐决策（per-decision）加权 IS 控制它，或者改用 TD。

## 生产实践

蒙特卡洛方法在 2026 年的角色：

| 应用场景 | 为什么用 MC |
|----------|--------|
| 短回合博弈（21 点、扑克） | 回合自然终止；回报干净。 |
| 已记录策略的离线评估 | 对存储的轨迹求折扣回报的平均。 |
| 蒙特卡洛树搜索（AlphaZero） | 从树叶节点出发的 MC 模拟引导节点选择。 |
| LLM 强化学习评估 | 对给定策略采样的补全计算平均奖励。 |
| PPO 中的基线估计 | 优势目标 `A_t = G_t - V(s_t)` 使用 MC 的 `G_t`。 |
| 强化学习教学 | 最简单且真正可用的算法——剥掉自举，看清核心。 |

现代深度强化学习算法（PPO、SAC）通过 `n` 步回报或 GAE 在纯 MC（完整回报）和纯 TD（单步自举）之间插值。两个端点是同一个估计器的不同实例。

## 交付产物

保存为 `outputs/skill-mc-evaluator.md`：

```markdown
---
name: mc-evaluator
description: Evaluate a policy via Monte Carlo rollouts and produce a convergence report with DP-comparison if available.
version: 1.0.0
phase: 9
lesson: 3
tags: [rl, monte-carlo, evaluation]
---

Given an environment (episodic, with reset+step API) and a policy, output:

1. Method. First-visit vs every-visit MC. Reason.
2. Episode budget. Target number, variance diagnostic, expected standard error.
3. Exploration plan. ε schedule (if needed) or exploring starts.
4. Gold-standard comparison. DP-optimal V* if tabular; otherwise a bound from a Q-learning / PPO baseline.
5. Termination check. Max-step cap, timeouts, handling of non-terminating trajectories.

Refuse to run MC on non-episodic tasks without a finite horizon cap. Refuse to report V^π estimates from fewer than 100 episodes per state for tabular tasks. Flag any policy with zero-variance actions as an exploration risk.
```

## 练习

1. **简单。** 实现对 4×4 GridWorld 上均匀随机策略的首次访问 MC 评估。运行 10,000 个回合。绘制 `V(0,0)` 随回合数变化的曲线，并与 DP 答案对比。
2. **中等。** 实现 ε-greedy MC 控制，取 `ε ∈ {0.01, 0.1, 0.3}`。比较 20,000 个回合后的平均回报。曲线长什么样？偏差-方差权衡体现在哪里？
3. **困难。** 实现带重要性采样的*离策略* MC：在均匀随机策略 `μ` 下收集数据，估计确定性最优策略 `π` 的 `V^π`。比较普通 IS、逐决策 IS 和加权 IS。哪个方差最低？

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|-----------------|-----------------------|
| 蒙特卡洛（Monte Carlo） | "随机采样" | 通过对来自该分布的独立同分布样本取平均来估计期望。 |
| 回报 `G_t` | "未来奖励" | 从第 `t` 步到回合结束的折扣奖励之和：`Σ_{k≥0} γ^k r_{t+k+1}`。 |
| 首次访问 MC | "每个状态只数一次" | 一个回合中只有第一次访问对价值估计有贡献。 |
| 每次访问 MC | "用上所有访问" | 每次访问都有贡献；略有偏差，但样本效率更高。 |
| ε-greedy | "探索噪声" | 以概率 `1-ε` 选贪心动作；以概率 `ε` 选随机动作。 |
| 重要性采样（importance sampling） | "纠正从错误分布采样的偏差" | 用 `π(a\|s)/μ(a\|s)` 的连乘积对回报重新加权，从 `μ` 的数据估计 `V^π`。 |
| 同策略（on-policy） | "从自己的数据中学习" | 目标策略 = 行为策略。原始 MC、PPO、SARSA。 |
| 离策略（off-policy） | "从别人的数据中学习" | 目标策略 ≠ 行为策略。重要性采样 MC、Q-learning、DQN。 |

## 延伸阅读

- [Sutton & Barto (2018). Ch. 5 — Monte Carlo Methods](http://incompleteideas.net/book/RLbook2020.pdf) — 经典的权威论述。
- [Singh & Sutton (1996). Reinforcement Learning with Replacing Eligibility Traces](https://link.springer.com/article/10.1007/BF00114726) — 首次访问 vs 每次访问的分析。
- [Precup, Sutton, Singh (2000). Eligibility Traces for Off-Policy Policy Evaluation](http://incompleteideas.net/papers/PSS-00.pdf) — 离策略 MC 与方差控制。
- [Mahmood et al. (2014). Weighted Importance Sampling for Off-Policy Learning](https://arxiv.org/abs/1404.6362) — 现代低方差 IS 估计器。
- [Tesauro (1995). TD-Gammon, A Self-Teaching Backgammon Program](https://dl.acm.org/doi/10.1145/203330.203343) — MC/TD 自我对弈收敛到超人水平的首个大规模实证演示；本阶段后半部分每一课的思想先驱。
