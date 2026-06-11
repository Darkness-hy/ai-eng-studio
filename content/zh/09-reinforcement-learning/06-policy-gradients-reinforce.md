# 策略梯度 —— 从零实现 REINFORCE

> 别再估计价值了。直接参数化策略，计算期望回报的梯度，沿梯度方向往上走。Williams（1992）用一条定理就写清了这件事。PPO、GRPO 以及所有 LLM 强化学习训练循环的存在，都源于它。

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 3 · 03 (Backpropagation), Phase 9 · 03 (Monte Carlo), Phase 9 · 04 (TD Learning)
**Time:** ~75 minutes

## 问题背景

Q-learning 和 DQN 参数化的是*价值*函数。你通过 `argmax Q` 来选择动作。这对离散动作和离散状态没问题。但当动作是连续的（对一个 10 维力矩怎么做 `argmax`？），或者你想要一个随机策略（`argmax` 天生就是确定性的）时，它就失效了。

策略梯度（policy gradient）转而参数化*策略*本身。`π_θ(a | s)` 是一个输出动作分布的神经网络。从中采样来执行动作。计算期望回报对 `θ` 的梯度。沿梯度上升。不需要 `argmax`，不需要 Bellman 递归，只需对 `J(θ) = E_{π_θ}[G]` 做梯度上升。

REINFORCE 定理（Williams 1992）告诉你这个梯度是可计算的：`∇J(θ) = E_π[ G · ∇_θ log π_θ(a | s) ]`。跑一个回合，计算回报，在每一步乘上 `∇ log π_θ(a | s)`，求平均，做梯度上升。完事。

2026 年的每一个 LLM 强化学习算法——PPO、DPO、GRPO——都是 REINFORCE 的改良版。把它练到指尖纯熟，是学完本阶段其余内容的前提，也是 Phase 10 · 07（RLHF 实现）和 Phase 10 · 08（DPO）的前提。

## 核心概念

![Policy gradient: softmax policy, log-π gradient, return-weighted update](../assets/policy-gradient.svg)

**策略梯度定理。** 对任意由 `θ` 参数化的策略 `π_θ`：

`∇J(θ) = E_{τ ~ π_θ}[ Σ_{t=0}^{T} G_t · ∇_θ log π_θ(a_t | s_t) ]`

其中 `G_t = Σ_{k=t}^{T} γ^{k-t} r_{k+1}` 是从第 `t` 步开始的折扣回报。期望是对从 `π_θ` 采样的完整轨迹 `τ` 取的。

**证明很短。** 在期望下对 `J(θ) = Σ_τ P(τ; θ) G(τ)` 求导。利用 `∇P(τ; θ) = P(τ; θ) ∇ log P(τ; θ)`（对数导数技巧，log-derivative trick）。把 `log P(τ; θ)` 分解为 `Σ log π_θ(a_t | s_t) + 与 θ 无关的环境项`。环境项消失。两行代数就得到了这条定理。

**方差缩减技巧。** 原始 REINFORCE 的方差大得惊人——回报有噪声，`∇ log π` 有噪声，二者的乘积噪声更大。两种标准修法：

1. **基线减除（baseline subtraction）。** 把 `G_t` 替换成 `G_t - b(s_t)`，其中基线 `b(s_t)` 可以是任何不依赖于 `a_t` 的函数。由于 `E[b(s_t) · ∇ log π(a_t | s_t)] = 0`，估计仍是无偏的。典型选择：用一个评论家（critic）学到的 `b(s_t) = V̂(s_t)` → 演员-评论家方法（第 07 课）。
2. **未来回报（reward-to-go）。** 把 `Σ_t G_t · ∇ log π_θ(a_t | s_t)` 替换成 `Σ_t G_t^{from t} · ∇ log π_θ(a_t | s_t)`。对某个动作而言，只有未来的回报才有意义——过去的奖励只贡献零均值噪声。

两者结合，得到：

`∇J ≈ (1/N) Σ_{i=1}^{N} Σ_{t=0}^{T_i} [ G_t^{(i)} - V̂(s_t^{(i)}) ] · ∇_θ log π_θ(a_t^{(i)} | s_t^{(i)})`

这就是带基线的 REINFORCE——A2C（第 07 课）和 PPO（第 08 课）的直系祖先。

**Softmax 策略参数化。** 对离散动作，标准做法是：

`π_θ(a | s) = exp(f_θ(s, a)) / Σ_{a'} exp(f_θ(s, a'))`

其中 `f_θ` 可以是任何为每个动作输出一个分数的神经网络。其梯度有简洁的形式：

`∇_θ log π_θ(a | s) = ∇_θ f_θ(s, a) - Σ_{a'} π_θ(a' | s) ∇_θ f_θ(s, a')`

即：所选动作的得分梯度，减去其在策略下的期望值。

**面向连续动作的高斯策略。** `π_θ(a | s) = N(μ_θ(s), σ_θ(s))`。`∇ log N(a; μ, σ)` 有闭式解。Phase 9 · 07 的 SAC 需要的就只有这些。

```figure
policy-gradient-landscape
```

## 从零实现

### 第 1 步：softmax 策略网络

```python
def policy_logits(theta, state_features):
    return [dot(theta[a], state_features) for a in range(N_ACTIONS)]

def softmax(logits):
    m = max(logits)
    exps = [exp(l - m) for l in logits]
    Z = sum(exps)
    return [e / Z for e in exps]
```

在表格型环境中使用线性策略（每个动作一个权重向量）即可。换到 Atari 时，把它换成 CNN，但保留 softmax 输出头。

### 第 2 步：采样与对数概率

```python
def sample_action(probs, rng):
    x = rng.random()
    cum = 0
    for a, p in enumerate(probs):
        cum += p
        if x <= cum:
            return a
    return len(probs) - 1

def log_prob(probs, a):
    return log(probs[a] + 1e-12)
```

### 第 3 步：采集轨迹并记录对数概率

```python
def rollout(theta, env, rng, gamma):
    trajectory = []
    s = env.reset()
    while not done:
        logits = policy_logits(theta, s)
        probs = softmax(logits)
        a = sample_action(probs, rng)
        s_next, r, done = env.step(s, a)
        trajectory.append((s, a, r, probs))
        s = s_next
    return trajectory
```

### 第 4 步：REINFORCE 更新

```python
def reinforce_step(theta, trajectory, gamma, lr, baseline=0.0):
    returns = compute_returns(trajectory, gamma)
    for (s, a, _, probs), G in zip(trajectory, returns):
        advantage = G - baseline
        grad_log_pi_a = [-p for p in probs]
        grad_log_pi_a[a] += 1.0
        for i in range(N_ACTIONS):
            for j in range(len(s)):
                theta[i][j] += lr * advantage * grad_log_pi_a[i] * s[j]
```

梯度 `∇ log π(a|s) = e_a - π(·|s)`（动作 `a` 的 one-hot 向量减去概率向量）是 softmax 策略梯度的核心。把它刻进肌肉记忆里。

### 第 5 步：基线

用近期若干回合 `G` 的滑动平均作基线，方差缩减的效果就足以让一个 4×4 GridWorld 跑起来；大约 500 个回合即可收敛。把基线升级为学习得到的 `V̂(s)`，你就得到了演员-评论家方法。

## 常见陷阱

- **梯度爆炸。** 回报可能非常大。在与 `∇ log π` 相乘之前，务必把整个批次的 `G` 归一化到 `~N(0, 1)`。
- **熵坍缩。** 策略过早收敛到一个近乎确定性的动作，停止探索，陷入局部解。修法：在目标函数中加入熵奖励项 `β · H(π(·|s))`。
- **高方差。** 原始 REINFORCE 需要数千个回合。标准修法是引入评论家基线（第 07 课）或 TRPO/PPO 的信赖域（第 08 课）。
- **样本效率低。** 同策略（on-policy）意味着每次更新后所有转移数据都得扔掉。通过重要性采样做异策略（off-policy）修正可以把数据找回来，代价是方差增大（PPO 的比率本质上是一个裁剪过的重要性采样权重）。
- **梯度非平稳。** 100 个回合前的同一份梯度用的是旧的 `π`。同策略方法每隔几次轨迹采集就更新一次，原因就在这里。
- **信用分配。** 不用 reward-to-go 的话，过去的奖励只会贡献噪声。永远使用 reward-to-go。

## 生产实践

到了 2026 年，REINFORCE 已很少被直接使用，但它的梯度公式无处不在：

| 应用场景 | 衍生方法 |
|----------|---------------|
| 连续控制 | 配高斯策略的 PPO / SAC |
| LLM RLHF | 带 KL 惩罚的 PPO，运行在 token 级策略上 |
| LLM 推理（DeepSeek） | GRPO——带组相对基线的 REINFORCE，无需评论家 |
| 多智能体 | 中心化评论家的 REINFORCE（MADDPG、COMA） |
| 离散动作机器人 | A2C、A3C、PPO |
| 仅有偏好信号的场景 | DPO——把 REINFORCE 改写为偏好似然损失，无需采样 |

当你在 2026 年的某个训练脚本里读到 `loss = -advantage * log_prob` 时，那就是带基线的 REINFORCE。整篇整篇的论文（DPO、GRPO、RLOO）都是在这一行代码之上做方差缩减。

## 交付产物

保存为 `outputs/skill-policy-gradient-trainer.md`：

```markdown
---
name: policy-gradient-trainer
description: Produce a REINFORCE / actor-critic / PPO training config for a given task and diagnose variance issues.
version: 1.0.0
phase: 9
lesson: 6
tags: [rl, policy-gradient, reinforce]
---

Given an environment (discrete / continuous actions, horizon, reward stats), output:

1. Policy head. Softmax (discrete) or Gaussian (continuous) with parameter counts.
2. Baseline. None (vanilla), running mean, learned `V̂(s)`, or A2C critic.
3. Variance controls. Reward-to-go on by default, return normalization, gradient clip value.
4. Entropy bonus. Coefficient β and decay schedule.
5. Batch size. Episodes per update; on-policy data freshness contract.

Refuse REINFORCE-no-baseline on horizons > 500 steps. Refuse continuous-action control with a softmax head. Flag any run with `β = 0` and observed policy entropy < 0.1 as entropy-collapsed.
```

## 练习

1. **简单。** 在 4×4 GridWorld 上用线性 softmax 策略实现 REINFORCE。不加基线训练 1,000 个回合。绘制学习曲线；测量方差（回报的标准差）。
2. **中等。** 加入滑动平均基线，重新训练。与不加基线的版本对比样本效率和方差。基线让收敛所需的步数减少了多少？
3. **困难。** 加入熵奖励项 `β · H(π)`。在 `β ∈ {0, 0.01, 0.1, 1.0}` 上做扫描。绘制最终回报与策略熵。在这个任务上，最佳取值落在哪里？

## 关键术语

| 术语 | 人们常说的 | 实际含义 |
|------|-----------------|-----------------------|
| 策略梯度（Policy gradient） | "直接训练策略" | `∇J(θ) = E[G · ∇ log π_θ(a\|s)]`；由对数导数技巧推导而来。 |
| REINFORCE | "最初的策略梯度算法" | Williams（1992）；蒙特卡洛回报乘以对数策略梯度。 |
| 对数导数技巧（Log-derivative trick） | "得分函数估计器" | `∇P(τ;θ) = P(τ;θ) · ∇ log P(τ;θ)`；让期望的梯度变得可计算。 |
| 基线（Baseline） | "方差缩减" | 从 `G` 中减去的任意 `b(s)`；由于 `E[b · ∇ log π] = 0`，估计仍无偏。 |
| 未来回报（Reward-to-go） | "只有未来回报才算数" | 用 `G_t^{from t}` 代替完整的 `G_0`；既正确又方差更低。 |
| 熵奖励（Entropy bonus） | "鼓励探索" | `+β · H(π(·\|s))` 项防止策略坍缩。 |
| 同策略（On-policy） | "用刚刚见到的数据训练" | 梯度期望是相对当前策略取的——不能直接复用旧数据。 |
| 优势（Advantage） | "比平均好多少" | `A(s, a) = G(s, a) - V(s)`；带基线的 REINFORCE 所乘的那个带符号的量。 |

## 延伸阅读

- [Williams (1992). Simple Statistical Gradient-Following Algorithms for Connectionist Reinforcement Learning](https://link.springer.com/article/10.1007/BF00992696) —— REINFORCE 的原始论文。
- [Sutton et al. (2000). Policy Gradient Methods for Reinforcement Learning with Function Approximation](https://papers.nips.cc/paper_files/paper/1999/hash/464d828b85b0bed98e80ade0a5c43b0f-Abstract.html) —— 带函数逼近的现代策略梯度定理。
- [Sutton & Barto (2018). Ch. 13 — Policy Gradient Methods](http://incompleteideas.net/book/RLbook2020.pdf) —— 教科书式的讲解。
- [OpenAI Spinning Up — VPG / REINFORCE](https://spinningup.openai.com/en/latest/algorithms/vpg.html) —— 清晰的教学讲解，附 PyTorch 代码。
- [Peters & Schaal (2008). Reinforcement Learning of Motor Skills with Policy Gradients](https://homes.cs.washington.edu/~todorov/courses/amath579/reading/PolicyGradient.pdf) —— 方差缩减与自然梯度视角，把 REINFORCE 与信赖域家族（TRPO、PPO）联系起来。
