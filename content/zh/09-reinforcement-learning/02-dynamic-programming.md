# 动态规划 —— 策略迭代与价值迭代

> 动态规划是「开了挂」的强化学习。你已经知道转移函数和奖励函数，只需反复迭代 Bellman 方程，直到 `V` 或 `π` 不再变化。它是所有基于采样的方法都试图逼近的基准。

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 9 · 01 (MDPs)
**Time:** ~75 minutes

## 问题背景

你手里有一个模型已知的 MDP：对任意状态-动作对，都可以直接查询 `P(s' | s, a)` 和 `R(s, a, s')`。库存管理员知道需求的分布。棋类游戏的转移是确定性的。一个 gridworld 用四行 Python 就能写完。你拥有的是一个*模型*。

无模型强化学习（Q-learning、PPO、REINFORCE）是为没有模型的场景发明的——那时你只能从环境中采样。但当你确实有模型时，存在更快、更好的方法：动态规划（dynamic programming）。Bellman 在 1957 年就设计了这些算法。它们至今仍是正确性的定义：当人们说「这个 MDP 的最优策略」时，指的就是 DP 会返回的那个策略。

到了 2026 年，你仍然需要它们，原因有三。第一，强化学习研究中的每个表格型环境（GridWorld、FrozenLake、CliffWalking）都用 DP 求解，以产生黄金标准策略。第二，精确的价值让你能够*调试*采样方法：如果 Q-learning 对 `V*(s_0)` 的估计与 DP 的答案相差 30%，那你的 Q-learning 有 bug。第三，现代离线强化学习和规划方法（MCTS、AlphaZero 的搜索、Phase 9 · 10 的基于模型的强化学习）都在对一个学到的或给定的模型迭代 Bellman 备份。

## 核心概念

![Policy iteration and value iteration, side by side](../assets/dp.svg)

**两种算法，都是对 Bellman 方程的不动点迭代。**

**策略迭代（policy iteration）。** 交替执行两个步骤，直到策略不再变化。

1. *评估（evaluation）：* 给定策略 `π`，通过反复应用 `V(s) ← Σ_a π(a|s) Σ_{s',r} P(s',r|s,a) [r + γ V(s')]` 直至收敛，计算 `V^π`。
2. *改进（improvement）：* 给定 `V^π`，让 `π` 对 `V^π` 取贪心：`π(s) ← argmax_a Σ_{s',r} P(s',r|s,a) [r + γ V(s')]`。

收敛性有保证，因为（a）每次改进步骤要么保持 `π` 不变，要么使某个状态的 `V^π` 严格增大，（b）确定性策略的空间是有限的。即使状态空间很大，通常也只需约 5–20 次外层迭代即可收敛。

**价值迭代（value iteration）。** 把评估和改进合并成一次扫描。应用 Bellman *最优性*方程：

`V(s) ← max_a Σ_{s',r} P(s',r|s,a) [r + γ V(s')]`

重复执行，直到 `max_s |V_{new}(s) - V(s)| < ε`。最后取贪心动作来提取策略。每次迭代严格更快——没有内层评估循环——但通常需要更多次迭代才能收敛。

**广义策略迭代（Generalized policy iteration，GPI）。** 统一的视角。价值函数和策略被锁在一个双向改进的循环中；任何驱动二者趋向相互一致的方法（异步价值迭代、修正策略迭代、Q-learning、actor-critic、PPO）都是 GPI 的一个实例。

**为什么 `γ < 1` 很重要。** Bellman 算子在上确界范数（sup-norm）下是一个 `γ`-压缩映射：`||T V - T V'||_∞ ≤ γ ||V - V'||_∞`。压缩性意味着唯一不动点和几何级收敛。去掉 `γ < 1`，保证就没了——这时你需要有限的时间范围（horizon）或一个吸收终止状态。

```figure
value-iteration-gamma
```

## 从零实现

### 第 1 步：构建 GridWorld 的 MDP 模型

沿用第 01 课的 4×4 GridWorld。我们加一个随机变体：智能体以 `0.1` 的概率滑向某个随机的垂直方向。

```python
SLIP = 0.1

def transitions(state, action):
    if state == TERMINAL:
        return [(state, 0.0, 1.0)]
    outcomes = []
    for direction, prob in action_probs(action):
        outcomes.append((apply_move(state, direction), -1.0, prob))
    return outcomes
```

`transitions(s, a)` 返回一个由 `(s', r, p)` 组成的列表。这就是完整的模型。

### 第 2 步：策略评估

给定策略 `π(s) = {action: prob}`，迭代 Bellman 方程直到 `V` 不再变化：

```python
def policy_evaluation(policy, gamma=0.99, tol=1e-6):
    V = {s: 0.0 for s in states()}
    while True:
        delta = 0.0
        for s in states():
            v = sum(pi_a * sum(p * (r + gamma * V[s_prime])
                              for s_prime, r, p in transitions(s, a))
                   for a, pi_a in policy(s).items())
            delta = max(delta, abs(v - V[s]))
            V[s] = v
        if delta < tol:
            return V
```

### 第 3 步：策略改进

把 `π` 替换为对 `V` 贪心的策略。如果 `π` 没有变化，则返回——我们已到达最优。

```python
def policy_improvement(V, gamma=0.99):
    new_policy = {}
    for s in states():
        best_a = max(
            ACTIONS,
            key=lambda a: sum(p * (r + gamma * V[s_prime])
                              for s_prime, r, p in transitions(s, a)),
        )
        new_policy[s] = best_a
    return new_policy
```

### 第 4 步：把它们拼起来

```python
def policy_iteration(gamma=0.99):
    policy = {s: "up" for s in states()}   # arbitrary start
    for _ in range(100):
        V = policy_evaluation(lambda s: {policy[s]: 1.0}, gamma)
        new_policy = policy_improvement(V, gamma)
        if new_policy == policy:
            return V, policy
        policy = new_policy
```

在 4×4 上的典型收敛：4–6 次外层迭代。输出 `V*(0,0) ≈ -6`，以及一个使步数严格递减的策略。

### 第 5 步：价值迭代（单循环版本）

```python
def value_iteration(gamma=0.99, tol=1e-6):
    V = {s: 0.0 for s in states()}
    while True:
        delta = 0.0
        for s in states():
            v = max(sum(p * (r + gamma * V[s_prime])
                       for s_prime, r, p in transitions(s, a))
                   for a in ACTIONS)
            delta = max(delta, abs(v - V[s]))
            V[s] = v
        if delta < tol:
            break
    policy = policy_improvement(V, gamma)
    return V, policy
```

同样的不动点，更少的代码行数。

## 常见陷阱

- **忘记处理终止状态。** 如果对吸收状态应用 Bellman 方程，它仍会挑出一个什么都改变不了的「最佳动作」。用 `if s == terminal: V[s] = 0` 加以防护。
- **上确界范数 vs L2 收敛。** 用 `max |V_new - V|`，而不是平均值。理论保证建立在上确界范数上。
- **就地更新 vs 同步更新。** 就地更新 `V[s]`（Gauss-Seidel 式）比用单独的 `V_new` 字典（Jacobi 式）收敛更快。生产代码用就地更新。
- **策略并列。** 如果两个动作的 Q 值相等，`argmax` 在每次迭代中可能以不同方式打破并列，导致「策略稳定」的检查来回振荡。使用稳定的并列打破规则（按固定顺序取第一个动作）。
- **状态空间爆炸。** DP 每次扫描的开销是 `O(|S| · |A|)`。最多能处理约 10⁷ 个状态。超过这个规模，你需要函数逼近（Phase 9 · 05 起）。

## 生产实践

在 2026 年，DP 既是正确性基线，也是规划器的内层循环：

| 用例 | 方法 |
|----------|--------|
| 精确求解小型表格型 MDP | 价值迭代（更简单）或策略迭代（外层步数更少） |
| 验证 Q-learning / PPO 实现 | 在玩具环境上与 DP 最优的 V* 对比 |
| 基于模型的强化学习（Phase 9 · 10） | 在学到的转移模型上做 Bellman 备份 |
| AlphaZero / MuZero 中的规划 | 蒙特卡洛树搜索 = 异步 Bellman 备份 |
| 离线强化学习（CQL、IQL） | 保守 Q 迭代——对 OOD 动作施加惩罚的 DP |

每当有人说「最优价值函数」，他们指的就是「DP 的不动点」。当你在论文中看到 `V*` 或 `Q*`，脑中浮现的就应该是这个循环。

## 交付产物

保存为 `outputs/skill-dp-solver.md`：

```markdown
---
name: dp-solver
description: Solve a small tabular MDP exactly via policy iteration or value iteration. Report convergence behavior.
version: 1.0.0
phase: 9
lesson: 2
tags: [rl, dynamic-programming, bellman]
---

Given an MDP with a known model, output:

1. Choice. Policy iteration vs value iteration. Reason tied to |S|, |A|, γ.
2. Initialization. V_0, starting policy. Convergence sensitivity.
3. Stopping. Sup-norm tolerance ε. Expected number of sweeps.
4. Verification. V*(s_0) computed exactly. Greedy policy extracted.
5. Use. How this baseline will be used to debug/evaluate sampling-based methods.

Refuse to run DP on state spaces > 10⁷. Refuse to claim convergence without a sup-norm check. Flag any γ ≥ 1 on an infinite-horizon task as a guarantee violation.
```

## 练习

1. **简单。** 在 4×4 GridWorld 上以 `γ ∈ {0.9, 0.99}` 运行价值迭代。需要多少次扫描才能达到 `max |ΔV| < 1e-6`？把 `V*` 打印成 4×4 网格。
2. **中等。** 在*随机*版 GridWorld（滑动概率 `0.1`）上对比策略迭代与价值迭代。统计：扫描次数、墙钟时间、最终的 `V*(0,0)`。哪个按迭代次数收敛更快？按墙钟时间呢？
3. **困难。** 构建修正策略迭代（modified policy iteration）：在评估步骤中只跑 `k` 次扫描，而不是跑到收敛。对 `k ∈ {1, 2, 5, 10, 50}` 绘制 `V*(0,0)` 误差随 `k` 变化的曲线。这条曲线告诉了你评估/改进之间怎样的权衡？

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|-----------------|-----------------------|
| 策略迭代 | 「DP 算法」 | 交替进行评估（`V^π`）和改进（对 `V^π` 贪心的 `π`），直到策略不再变化。 |
| 价值迭代 | 「更快的 DP」 | 在一次扫描中应用 Bellman 最优性备份；以几何速率收敛到 `V*`。 |
| Bellman 算子 | 「那个递推式」 | `(T V)(s) = max_a Σ P (r + γ V(s'))`；在上确界范数下是 `γ`-压缩映射。 |
| 压缩映射 | 「DP 为什么收敛」 | 任何满足 `\|\|T x - T y\|\| ≤ γ \|\|x - y\|\|` 的算子 `T` 都有唯一不动点。 |
| GPI | 「一切都是 DP」 | 广义策略迭代（Generalized Policy Iteration）：任何驱动 `V` 和 `π` 趋向相互一致的方法。 |
| 同步更新 | 「Jacobi 式」 | 整个扫描过程使用旧的 `V`；便于分析，但更慢。 |
| 就地更新 | 「Gauss-Seidel 式」 | 边更新边使用 `V`；实践中收敛更快。 |

## 延伸阅读

- [Sutton & Barto (2018). Ch. 4 — Dynamic Programming](http://incompleteideas.net/book/RLbook2020.pdf) —— 策略迭代与价值迭代的经典论述。
- [Bertsekas (2019). Reinforcement Learning and Optimal Control](http://www.athenasc.com/rlbook.html) —— 对压缩映射论证的严格处理。
- [Puterman (2005). Markov Decision Processes](https://onlinelibrary.wiley.com/doi/book/10.1002/9780470316887) —— 修正策略迭代及其收敛性分析。
- [Howard (1960). Dynamic Programming and Markov Processes](https://mitpress.mit.edu/9780262582300/dynamic-programming-and-markov-processes/) —— 策略迭代的原始论文。
- [Bertsekas & Tsitsiklis (1996). Neuro-Dynamic Programming](http://www.athenasc.com/ndpbook.html) —— 从 DP 到近似 DP / 深度强化学习的桥梁，后续每一课都会用到。
