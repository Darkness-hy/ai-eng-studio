# 深度 Q 网络（DQN）

> 2013 年：Mnih 用一个 Q-learning 网络直接从原始像素学习，在七款 Atari 游戏上击败了所有经典强化学习智能体。2015 年：扩展到 49 款游戏，论文发表在 Nature 上，引爆了深度强化学习时代。DQN 就是 Q-learning 加上三个让函数近似保持稳定的技巧。

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 3 · 03 (Backpropagation), Phase 9 · 04 (Q-learning, SARSA)
**Time:** ~75 minutes

## 问题背景

表格型 Q-learning 需要为每个（状态，动作）对单独维护一个 Q 值。一个国际象棋棋盘约有 10⁴³ 个状态，一帧 Atari 画面有 210×160×3 = 100,800 个特征。表格型 RL 在几千个状态时就撑不住了，更别提几十亿。

事后看，解决方案显而易见：用神经网络 `Q(s, a; θ)` 替换 Q 表。但这个"事后显而易见"耗费了几十年。把朴素的函数近似直接套上 Q-learning 会在"致命三要素"（deadly triad）——函数近似 + 自举（bootstrapping）+ 离策略（off-policy）学习——的作用下发散。Mnih 等人（2013、2015）找出了三个稳定训练的工程技巧：

1. **经验回放（experience replay）** 打破转移样本之间的相关性。
2. **目标网络（target network）** 冻结自举目标。
3. **奖励裁剪（reward clipping）** 归一化梯度幅度。

DQN 在 Atari 上的成功，是历史上第一次用单一架构、单一超参数组合，直接从原始像素解决了几十个控制问题。此后所有"深度 RL"的成果——DDQN、Rainbow、Dueling、Distributional、R2D2、Agent57——全都建立在这个三技巧基础之上。

## 核心概念

![DQN training loop: env, replay buffer, online net, target net, Bellman TD loss](../assets/dqn.svg)

**目标函数。** DQN 在神经 Q 函数上最小化单步 TD 损失：

`L(θ) = E_{(s,a,r,s')~D} [ (r + γ max_{a'} Q(s', a'; θ^-) - Q(s, a; θ))² ]`

`θ` 是在线网络（online network），每一步都通过梯度下降更新。`θ^-` 是目标网络，定期从 `θ` 复制而来（约每 10,000 步一次）。`D` 是存放历史转移样本的回放缓冲区。

**三个技巧，按重要性排序：**

**经验回放。** 一个容量约 `~10⁶` 的环形缓冲区存放转移样本。每个训练步从中均匀随机采样一个小批量。这打破了时间相关性（相邻帧几乎完全相同），让网络能反复学习稀有的高奖励转移，并使连续的梯度更新彼此去相关。没有它，神经网络上的同策略 TD 在 Atari 上会发散。

**目标网络。** 如果在 Bellman 方程两侧使用同一个网络 `Q(·; θ)`，目标会随着每次更新而移动——相当于"追着自己的尾巴跑"。解决办法：维护第二个权重冻结的网络 `Q(·; θ^-)`，每隔 `C` 步执行一次复制 `θ → θ^-`。这样回归目标可以在数千个梯度步内保持稳定。软更新 `θ^- ← τ θ + (1-τ) θ^-`（DDPG、SAC 采用）是一种更平滑的变体。

**奖励裁剪。** Atari 游戏的奖励幅度从 1 到 1000+ 不等。裁剪到 `{-1, 0, +1}` 可以防止任何一款游戏主导梯度。当奖励幅度本身有意义时这样做是错的；但对只看奖励符号的 Atari 来说没有问题。

**Double DQN。** Hasselt（2016）修复了最大化偏差：用在线网络*选择*动作，用目标网络*评估*它。

`target = r + γ Q(s', argmax_{a'} Q(s', a'; θ); θ^-)`

可以直接替换原有目标，效果稳定更好。默认就用它。

**其他改进（Rainbow，2017）：** 优先经验回放（更多地采样高 TD 误差的转移）、对偶架构（dueling，把 `V(s)` 和优势分成两个头）、噪声网络（可学习的探索）、n 步回报、分布式 Q（C51/QR-DQN）、多步自举。每项改进带来几个百分点的提升，且增益大致可叠加。

## 从零实现

这里的代码只用标准库、不依赖 numpy——我们手写一个单隐藏层 MLP，运行在一个微型连续 GridWorld 上，每个训练步只需微秒级时间。算法本身与大规模 Atari DQN 完全一致。

### 第 1 步：回放缓冲区

```python
class ReplayBuffer:
    def __init__(self, capacity):
        self.buf = []
        self.capacity = capacity
    def push(self, s, a, r, s_next, done):
        if len(self.buf) == self.capacity:
            self.buf.pop(0)
        self.buf.append((s, a, r, s_next, done))
    def sample(self, batch, rng):
        return rng.sample(self.buf, batch)
```

Atari 上容量约 50,000；我们的玩具环境用 5,000 就够了。

### 第 2 步：一个微型 Q 网络（手写 MLP）

```python
class QNet:
    def __init__(self, n_in, n_hidden, n_actions, rng):
        self.W1 = [[rng.gauss(0, 0.3) for _ in range(n_in)] for _ in range(n_hidden)]
        self.b1 = [0.0] * n_hidden
        self.W2 = [[rng.gauss(0, 0.3) for _ in range(n_hidden)] for _ in range(n_actions)]
        self.b2 = [0.0] * n_actions
    def forward(self, x):
        h = [max(0.0, sum(w * xi for w, xi in zip(row, x)) + b) for row, b in zip(self.W1, self.b1)]
        q = [sum(w * hi for w, hi in zip(row, h)) + b for row, b in zip(self.W2, self.b2)]
        return q, h
```

前向传播：线性层 → ReLU → 线性层。整个网络就这么多。

### 第 3 步：DQN 更新

```python
def train_step(online, target, batch, gamma, lr):
    grads = zeros_like(online)
    for s, a, r, s_next, done in batch:
        q, h = online.forward(s)
        if done:
            y = r
        else:
            q_next, _ = target.forward(s_next)
            y = r + gamma * max(q_next)
        td_error = q[a] - y
        accumulate_grads(grads, online, s, h, a, td_error)
    apply_sgd(online, grads, lr / len(batch))
```

整体结构就是第 04 课的 Q-learning，只有两处不同：(a) 我们对可微的 `Q(·; θ)` 做反向传播，而不是去索引一张表；(b) 目标用的是 `Q(·; θ^-)`。

### 第 4 步：外层循环

每个回合中，按 ε-贪心策略基于 `Q(·; θ)` 行动，把转移样本压入缓冲区，采样一个小批量，执行一步梯度更新，并定期同步 `θ^- ← θ`。模式如下：

```python
for episode in range(N):
    s = env.reset()
    while not done:
        a = epsilon_greedy(online, s, epsilon)
        s_next, r, done = env.step(s, a)
        buffer.push(s, a, r, s_next, done)
        if len(buffer) >= batch:
            train_step(online, target, buffer.sample(batch), gamma, lr)
        if steps % sync_every == 0:
            target = copy(online)
        s = s_next
```

在我们这个使用 16 维 one-hot 状态的微型 GridWorld 上，智能体大约 500 个回合就能学到接近最优的策略。换到 Atari，只需把规模扩到 2 亿帧并加一个 CNN 特征提取器。

## 常见陷阱

- **致命三要素。** 函数近似 + 离策略 + 自举可能导致发散。DQN 靠目标网络 + 经验回放来缓解；两者都不能去掉。
- **探索。** ε 必须衰减，典型做法是在训练前约 10% 的时间内从 1.0 衰减到 0.01。早期探索不足，Q 网络会收敛到一个局部盆地。
- **过估计。** 对带噪声的 Q 取 `max` 会产生向上偏差。生产环境中务必使用 Double DQN。
- **奖励尺度。** 对奖励做裁剪或归一化；梯度幅度与奖励幅度成正比。
- **回放缓冲区冷启动。** 缓冲区积累到几千个转移之前不要开始训练。在约 20 个样本上算出的早期梯度会过拟合。
- **目标同步频率。** 同步太频繁 ≈ 没有目标网络；太稀疏 ≈ 目标过时。Atari DQN 用 10,000 个环境步。经验法则：大约每 1/100 的训练总长同步一次。
- **观测预处理。** Atari DQN 堆叠 4 帧来让状态满足马尔可夫性。任何包含速度信息的环境都需要帧堆叠或循环状态。

## 生产实践

到了 2026 年，DQN 已经很少是最先进方案，但仍然是离策略算法的参照基准：

| 任务 | 首选方法 | 为什么不用 DQN？ |
|------|------------------|--------------|
| 离散动作、类 Atari 任务 | Rainbow DQN 或 Muesli | 同一框架，技巧更多。 |
| 连续控制 | SAC / TD3（Phase 9 · 07） | DQN 没有策略网络。 |
| 同策略 / 高吞吐 | PPO（Phase 9 · 08） | 无回放缓冲区，更易扩展。 |
| 离线 RL | CQL / IQL / Decision Transformer | 保守的 Q 目标，避免自举爆炸。 |
| 大规模离散动作空间（推荐系统） | 带动作嵌入的 DQN，或 IMPALA | 可用；关键在细节装配。 |
| LLM 的强化学习 | PPO / GRPO | 序列级而非步级，损失函数不同。 |

但这些经验依然在流传。回放缓冲区和目标网络出现在 SAC、TD3、DDPG、SAC-X、AlphaZero 的自博弈缓冲区以及所有离线 RL 方法中。奖励裁剪则以优势归一化的形式活在 PPO 里。这套架构就是后续一切的蓝图。

## 交付产物

保存为 `outputs/skill-dqn-trainer.md`：

```markdown
---
name: dqn-trainer
description: Produce a DQN training config (buffer, target sync, ε schedule, reward clipping) for a discrete-action RL task.
version: 1.0.0
phase: 9
lesson: 5
tags: [rl, dqn, deep-rl]
---

Given a discrete-action environment (observation shape, action count, horizon, reward scale), output:

1. Network. Architecture (MLP / CNN / Transformer), feature dim, depth.
2. Replay buffer. Capacity, minibatch size, warmup size.
3. Target network. Sync strategy (hard every C steps or soft τ).
4. Exploration. ε start / end / schedule length.
5. Loss. Huber vs MSE, gradient clip value, reward clipping rule.
6. Double DQN. On by default unless explicit reason to disable.

Refuse to ship a DQN with no target network, no replay buffer, or ε held at 1. Refuse continuous-action tasks (route to SAC / TD3). Flag any reward range > 10× per-step mean as needing clipping or scale normalization.
```

## 练习

1. **简单。** 运行 `code/main.py`，绘制每回合回报曲线。需要多少回合，回报的滑动平均才能超过 -10？
2. **中等。** 禁用目标网络（Bellman 目标两侧都用在线网络）。观察训练的不稳定性——回报会振荡还是发散？
3. **困难。** 加入 Double DQN：用在线网络选出 `argmax a'`，用目标网络评估。在一个带噪声奖励的 GridWorld 上训练 1,000 回合，比较有无 Double DQN 时 `Q(s_0, best_a)` 相对真实 `V*(s_0)` 的偏差。

## 关键术语

| 术语 | 人们常说 | 实际含义 |
|------|-----------------|-----------------------|
| DQN | "深度 Q-learning" | 带神经 Q 函数、回放缓冲区和目标网络的 Q-learning。 |
| 经验回放 | "打乱的转移样本" | 环形缓冲区，每个梯度步均匀采样；用于数据去相关。 |
| 目标网络 | "冻结的自举" | Bellman 目标中使用的 Q 网络定期副本；稳定训练。 |
| 致命三要素 | "RL 为什么发散" | 函数近似 + 自举 + 离策略 = 没有收敛保证。 |
| Double DQN | "最大化偏差的修复" | 在线网络选动作，目标网络评估它。 |
| Dueling DQN | "V 头和 A 头" | 分解 Q = V + A - mean(A)；输出相同，梯度流更好。 |
| Rainbow | "所有技巧打包" | DDQN + PER + dueling + n 步 + noisy + 分布式 Q 的合体。 |
| PER | "优先经验回放" | 按 TD 误差幅度成比例地采样转移。 |

## 延伸阅读

- [Mnih et al. (2013). Playing Atari with Deep Reinforcement Learning](https://arxiv.org/abs/1312.5602) —— 开启深度 RL 时代的 2013 年 NeurIPS workshop 论文。
- [Mnih et al. (2015). Human-level control through deep reinforcement learning](https://www.nature.com/articles/nature14236) —— Nature 论文，49 款游戏的 DQN。
- [Hasselt, Guez, Silver (2016). Deep Reinforcement Learning with Double Q-learning](https://arxiv.org/abs/1509.06461) —— DDQN。
- [Wang et al. (2016). Dueling Network Architectures](https://arxiv.org/abs/1511.06581) —— dueling DQN。
- [Hessel et al. (2018). Rainbow: Combining Improvements in Deep RL](https://arxiv.org/abs/1710.02298) —— 技巧大合集论文。
- [OpenAI Spinning Up — DQN](https://spinningup.openai.com/en/latest/algorithms/dqn.html) —— 清晰的现代讲解。
- [Sutton & Barto (2018). Ch. 9 — On-policy Prediction with Approximation](http://incompleteideas.net/book/RLbook2020.pdf) —— 教材中对"致命三要素"（函数近似 + 自举 + 离策略）的系统论述，DQN 的目标网络和回放缓冲区正是为驯服它而设计的。
- [CleanRL DQN implementation](https://docs.cleanrl.dev/rl-algorithms/dqn/) —— 消融研究中常用的单文件参考实现；适合与本课的从零实现对照阅读。
