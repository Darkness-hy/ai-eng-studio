# Actor-Critic——A2C 与 A3C

> REINFORCE 噪声太大。加一个学习 `V̂(s)` 的 critic，从回报中减去它，就得到了期望不变但方差大幅降低的优势函数。这就是 actor-critic。A2C 以同步方式运行它；A3C 则跨线程异步运行。两者是理解所有现代深度强化学习方法的思维模型。

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 9 · 04 (TD Learning), Phase 9 · 06 (REINFORCE)
**Time:** ~75 minutes

## 问题背景

原始的 REINFORCE 能用，但方差糟糕透顶。蒙特卡洛回报 `G_t` 在不同回合之间可能相差 10 倍以上。把这种噪声乘以 `∇ log π` 再取平均，得到的梯度估计器需要数千个回合才能把策略推进同样的距离——而 DQN 用少得多的更新就能做到。

方差来自直接使用原始回报。如果减去一个基线 `b(s_t)`——任何只依赖状态的函数，包括一个学到的价值函数——期望保持不变，方差却会下降。最优且可行的基线是 `V̂(s_t)`。此时与 `∇ log π` 相乘的量就是*优势*（advantage）：

`A(s, a) = G - V̂(s)`

一个动作如果产生了高于平均的回报，它就是好动作；低于平均则是坏动作。带学习型 critic 的 REINFORCE 就是 *actor-critic*。critic 为 actor 提供了一个低方差的老师。2015 年之后所有深度策略方法（A2C、A3C、PPO、SAC、IMPALA）都是这个结构。

## 核心概念

![Actor-critic: policy net plus value net, TD residual as advantage](../assets/actor-critic.svg)

**两个网络，一个联合损失：**

- **Actor** `π_θ(a | s)`：策略本身。通过采样来行动，用策略梯度训练。
- **Critic** `V_φ(s)`：估计从某状态出发的期望回报。通过最小化 `(V_φ(s) - target)²` 来训练。

**优势函数。** 两种标准形式：

- *MC 优势：* `A_t = G_t - V_φ(s_t)`。无偏，但方差较高。
- *TD 优势：* `A_t = r_{t+1} + γ V_φ(s_{t+1}) - V_φ(s_t)`。有偏（因为用了 `V_φ`），但方差低得多。也叫 *TD 残差* `δ_t`。

**n 步优势。** 在两者之间插值：

`A_t^{(n)} = r_{t+1} + γ r_{t+2} + … + γ^{n-1} r_{t+n} + γ^n V_φ(s_{t+n}) - V_φ(s_t)`

`n = 1` 是纯 TD，`n = ∞` 是 MC。大多数实现中，Atari 用 `n = 5`，MuJoCo 上的 PPO 用 `n = 2048`。

**广义优势估计（Generalized Advantage Estimation, GAE）。** Schulman 等人（2016）提出对所有 n 步优势做指数加权平均：

`A_t^{GAE} = Σ_{l=0}^{∞} (γλ)^l δ_{t+l}`

其中 `λ ∈ [0, 1]`。`λ = 0` 是 TD（低方差、高偏差），`λ = 1` 是 MC（高方差、无偏）。`λ = 0.95` 是 2026 年的默认值——调这个旋钮，直到偏差/方差的平衡点落在你想要的位置。

**A2C：同步优势 actor-critic。** 在 `N` 个并行环境中各收集 `T` 步，为每一步计算优势，然后在合并的批次上更新 actor 和 critic，循环往复。它是 A3C 更简单、更易扩展的同胞兄弟。

**A3C：异步优势 actor-critic。** Mnih 等人（2016）。启动 `N` 个工作线程，每个线程跑一个环境。每个 worker 在自己的 rollout 上本地计算梯度，再异步地把梯度应用到共享的参数服务器上。不需要回放缓冲区——各 worker 通过运行不同轨迹来去除数据相关性。A3C 证明了可以在 CPU 上规模化训练。到了 2026 年，基于 GPU 的 A2C（批量并行环境）占据主导，因为 GPU 喜欢大批次。

**联合损失。**

`L(θ, φ) = -E[ A_t · log π_θ(a_t | s_t) ]  +  c_v · E[(V_φ(s_t) - G_t)²]  -  c_e · E[H(π_θ(·|s_t))]`

三个部分：策略梯度损失、价值回归、熵奖励项。`c_v ~ 0.5`、`c_e ~ 0.01` 是约定俗成的起始值。

## 从零实现

### 第 1 步：实现一个 critic

用 MSE 更新的线性 critic `V_φ(s) = w · features(s)`：

```python
def critic_update(w, x, target, lr):
    v_hat = dot(w, x)
    err = target - v_hat
    for j in range(len(w)):
        w[j] += lr * err * x[j]
    return v_hat
```

在表格型环境中，critic 几百个回合就能收敛。在 Atari 上，把线性 critic 换成共享 CNN 主干加价值头。

### 第 2 步：n 步优势

给定长度为 `T` 的 rollout 和自举得到的末端价值 `V(s_T)`：

```python
def compute_advantages(rewards, values, gamma=0.99, lam=0.95, last_value=0.0):
    advantages = [0.0] * len(rewards)
    gae = 0.0
    for t in reversed(range(len(rewards))):
        next_v = values[t + 1] if t + 1 < len(values) else last_value
        delta = rewards[t] + gamma * next_v - values[t]
        gae = delta + gamma * lam * gae
        advantages[t] = gae
    returns = [a + v for a, v in zip(advantages, values)]
    return advantages, returns
```

`returns` 是 critic 的回归目标，`advantages` 是与 `∇ log π` 相乘的量。

### 第 3 步：联合更新

```python
for step_i, (x, a, _r, probs) in enumerate(traj):
    adv = advantages[step_i]
    target_v = returns[step_i]

    # critic
    critic_update(w, x, target_v, lr_v)

    # actor
    for i in range(N_ACTIONS):
        grad_logpi = (1.0 if i == a else 0.0) - probs[i]
        for j in range(N_FEAT):
            theta[i][j] += lr_a * adv * grad_logpi * x[j]
```

同策略（on-policy），每次更新用一个 rollout，actor 和 critic 各用独立的学习率。

### 第 4 步：并行化（A3C vs A2C）

- **A3C：** 启动 `N` 个线程。每个线程跑自己的环境、做自己的前向计算，周期性地把梯度推送到共享的主参数上。主参数不加锁——竞态没关系，只是多了一点噪声。
- **A2C：** 在单个进程中跑 `N` 个环境实例，把观测堆叠成 `[N, obs_dim]` 的批次，批量前向、批量反向。GPU 利用率更高、结果确定、更容易推理分析。2026 年的默认选择。

我们的玩具代码为了清晰起见是单线程的；改写成批量 A2C 只需三行 numpy。

## 常见陷阱

- **actor 梯度之前的 critic 偏差。** 如果 critic 还是随机的，它给出的基线毫无信息量，你等于在纯噪声上训练。先用几百步预热 critic 再开启策略梯度，或者给 actor 用一个很慢的学习率。
- **优势归一化。** 每个批次内把优势归一化为零均值、单位标准差。几乎零成本，却能大幅稳定训练。
- **共享主干。** 对图像输入，actor 和 critic 共用一个特征提取器，各接独立的头。共享特征能同时从两个损失中受益。
- **同策略契约。** A2C 的数据只能复用恰好一次更新。再多用，梯度就有偏了（PPO 加的正是重要性采样修正）。
- **熵坍缩。** 没有 `c_e > 0`，策略在几百次更新内就会变得近乎确定性，停止探索。
- **奖励尺度。** 优势的量级取决于奖励尺度。对奖励做归一化（例如除以滑动标准差），让梯度量级在不同任务间保持一致。

## 生产实践

到了 2026 年，A2C/A3C 很少是最终选择，但后续的一切方法都是在这个架构上做改良：

| 方法 | 与 A2C 的关系 |
|--------|----------------|
| PPO | A2C + 截断的重要性比率，支持多 epoch 更新 |
| IMPALA | A3C + V-trace 异策略修正 |
| SAC (Phase 9 · 07) | 带软价值 critic 的异策略 A2C（下一课） |
| GRPO (Phase 9 · 12) | 去掉 critic 的 A2C——组内相对优势 |
| DPO | 把 A2C 折叠成偏好排序损失，无需采样 |
| AlphaStar / OpenAI Five | A2C + 联赛训练 + 模仿学习预训练 |

如果你在 2026 年的论文里看到「advantage」，就往 actor-critic 上想。

## 交付产物

保存为 `outputs/skill-actor-critic-trainer.md`：

```markdown
---
name: actor-critic-trainer
description: Produce an A2C / A3C / GAE configuration for a given environment, with advantage estimation and loss weights specified.
version: 1.0.0
phase: 9
lesson: 7
tags: [rl, actor-critic, gae]
---

Given an environment and compute budget, output:

1. Parallelism. A2C (GPU batched) vs A3C (CPU async) and the number of workers.
2. Rollout length T. Steps per env per update.
3. Advantage estimator. n-step or GAE(λ); specify λ.
4. Loss weights. `c_v` (value), `c_e` (entropy), gradient clip.
5. Learning rates. Actor and critic (separate if using).

Refuse single-worker A2C on environments with horizon > 1000 (too on-policy, too slow). Refuse to ship without advantage normalization. Flag any run with `c_e = 0` and observed entropy < 0.1 as entropy-collapsed.
```

## 练习

1. **简单。** 在 4×4 GridWorld 上用 MC 优势（`G_t - V(s_t)`）训练 actor-critic。与第 06 课的「REINFORCE + 滑动均值基线」对比样本效率。
2. **中等。** 切换到 TD 残差优势（`r + γ V(s') - V(s)`）。测量各批次优势的方差，看看下降了多少。
3. **困难。** 实现 GAE(λ)。扫描 `λ ∈ {0, 0.5, 0.9, 0.95, 1.0}`，绘制最终回报与样本效率的关系图。对这个任务来说，偏差/方差的最佳平衡点在哪里？

## 关键术语

| 术语 | 大家怎么说 | 实际含义 |
|------|-----------------|-----------------------|
| Actor | 「策略网络」 | `π_θ(a\|s)`，由策略梯度更新。 |
| Critic | 「价值网络」 | `V_φ(s)`，通过对回报 / TD 目标做 MSE 回归来更新。 |
| 优势（Advantage） | 「比平均好多少」 | `A(s, a) = Q(s, a) - V(s)` 或其各种估计器。与 `∇ log π` 相乘的系数。 |
| TD 残差 | 「δ」 | `δ_t = r + γ V(s') - V(s)`；单步优势估计。 |
| GAE | 「插值旋钮」 | n 步优势的指数加权和，由 `λ` 参数化。 |
| A2C | 「同步 actor-critic」 | 跨环境批量计算；每个 rollout 做一次梯度更新。 |
| A3C | 「异步 actor-critic」 | 工作线程把梯度推送到共享参数服务器。原始论文的方案；2026 年已不常用。 |
| 自举（Bootstrap） | 「在视界处用 V」 | 截断 rollout，加上 `γ^n V(s_{t+n})` 来补全求和。 |

## 延伸阅读

- [Mnih et al. (2016). Asynchronous Methods for Deep Reinforcement Learning](https://arxiv.org/abs/1602.01783)——A3C，最早的异步 actor-critic 论文。
- [Schulman et al. (2016). High-Dimensional Continuous Control Using Generalized Advantage Estimation](https://arxiv.org/abs/1506.02438)——GAE。
- [Sutton & Barto (2018). Ch. 13 — Actor-Critic Methods](http://incompleteideas.net/book/RLbook2020.pdf)——基础理论；当 critic 是神经网络时，请搭配第 9 章函数逼近一起阅读。
- [Espeholt et al. (2018). IMPALA](https://arxiv.org/abs/1802.01561)——可扩展的分布式 actor-critic，带 V-trace 异策略修正。
- [OpenAI Baselines / Stable-Baselines3](https://stable-baselines3.readthedocs.io/)——值得一读的生产级 A2C/PPO 实现。
- [Konda & Tsitsiklis (2000). Actor-Critic Algorithms](https://papers.nips.cc/paper/1786-actor-critic-algorithms)——双时间尺度 actor-critic 分解的奠基性收敛结果。
