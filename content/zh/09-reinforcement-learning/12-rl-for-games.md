# 游戏中的强化学习 — AlphaZero、MuZero 与 LLM 推理时代

> 1992 年：TD-Gammon 仅靠时序差分（TD）就在西洋双陆棋上击败人类冠军。2016 年：AlphaGo 战胜李世石。2017 年：AlphaZero 从零开始统治国际象棋、将棋和围棋。2024 年：DeepSeek-R1 证明了同样的配方——用 GRPO 替换 PPO——同样适用于推理任务。游戏是驱动本阶段每一次突破的基准测试。

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 9 · 05 (DQN), Phase 9 · 08 (PPO), Phase 9 · 09 (RLHF), Phase 9 · 10 (MARL)
**Time:** ~120 minutes

## 问题背景

游戏拥有强化学习想要的一切。干净的奖励信号（赢/输）。无限的回合（自我博弈可以随时重置）。完美的仿真（游戏本身*就是*模拟器）。离散或较小的连续动作空间。迫使智能体具备对抗鲁棒性的多智能体结构。

而且游戏正是每一次重大 RL 突破的试验场。TD-Gammon（西洋双陆棋，1992）。Atari-DQN（2013）。AlphaGo（2016）。AlphaZero（2017）。OpenAI Five（Dota 2，2019）。AlphaStar（星际争霸 II，2019）。MuZero（学习到的模型，2019）。AlphaTensor（矩阵乘法，2022）。AlphaDev（排序算法，2023）。DeepSeek-R1（数学推理，2025）——这是游戏 RL 技术同样适用于文本的最新例证。

这节压轴课通过一个统一视角——**自我博弈 + 搜索 + 策略改进**——梳理三个里程碑式的架构：AlphaZero、MuZero 和 GRPO。每一个都是前一个的泛化；尤其是 GRPO，它就是 AlphaZero 的配方应用到 LLM 推理上：token 是动作，数学验证就是胜负信号。

## 核心概念

![AlphaZero ↔ MuZero ↔ GRPO: same loop, different environments](../assets/rl-games.svg)

**统一的循环。**

```
while True:
    trajectory = self_play(current_policy, search)     # play game against self
    policy_target = search.improved_policy(trajectory) # search improves raw policy
    policy_net.update(policy_target, value_target)     # supervised on search output
```

**AlphaZero（2017）。** Silver 等人。给定一个规则已知的游戏（国际象棋、将棋、围棋）：

- 策略-价值网络：单一主干 `f_θ(s) → (p, v)`。`p` 是合法走法上的先验分布。`v` 是对局结果的期望值。
- 蒙特卡洛树搜索（Monte Carlo Tree Search，MCTS）：每走一步，展开一棵后续可能局面的搜索树。用 `(p, v)` 作为先验 + 自举值。按 UCB（PUCT）选择节点：`a* = argmax Q(s, a) + c · p(a|s) · √N(s) / (1 + N(s, a))`。
- 自我博弈：让智能体与自己对弈。在第 `t` 步，MCTS 的访问分布 `π_t` 成为策略的训练目标。
- 损失：`L = (v - z)² - π · log p + c · ||θ||²`。`z` 是对局结果（+1 / 0 / -1）。

零人类知识。零手工启发式规则。同一份配方，在各自经过几千万局自我博弈后，分别精通了国际象棋、将棋和围棋。

**MuZero（2019）。** Schrittwieser 等人。去掉了"规则必须已知"这一要求。

- 不再依赖固定环境，而是学习一个*潜在动力学模型*（latent dynamics model）`(h, g, f)`：
  - `h(s)`：将观测编码为潜在状态。
  - `g(s_latent, a)`：预测下一个潜在状态 + 奖励。
  - `f(s_latent)`：预测策略先验 + 价值。
- MCTS 在*学到的潜在空间*中运行。搜索方式不变，训练循环不变。
- 适用于围棋、国际象棋、将棋，*还有* Atari——一个算法，不需要任何规则知识。

**Stochastic MuZero（2022）。** 加入随机动力学和机会节点；扩展到西洋双陆棋这一类游戏。

**Muesli、Gumbel MuZero（2022-2024）。** 在样本效率和确定性搜索上的改进。

**GRPO（2024-2025）。** DeepSeek-R1 的配方。同样是 AlphaZero 形态的循环，应用到语言模型推理上：

- "游戏"：回答一道数学 / 编程 / 推理题。"赢" = 验证器（测试用例通过、数值答案匹配）返回 1。
- 策略：LLM。动作：token。状态：提示词 + 目前已生成的回复。
- 没有 critic（PPO 式的 V_φ）。取而代之的是：对每个提示词，从策略中采样 `G` 个补全，逐一计算奖励，然后用**组相对优势**（group-relative advantage）`A_i = (r_i - mean_r) / std_r` 作为 REINFORCE 式更新的信号。
- 对参考策略施加 KL 惩罚以防止漂移（与 RLHF 相同）。
- 完整损失：

  `L_GRPO(θ) = -E_{q, {o_i}} [ (1/G) Σ_i A_i · log π_θ(o_i | q) ] + β · KL(π_θ || π_ref)`

不需要奖励模型，不需要 critic，不需要 MCTS。组相对基线一并取代了这三者。在推理基准上达到甚至超过 PPO-RLHF 的质量，而算力只需要一小部分。

**完整的 R1 配方。** DeepSeek-R1（DeepSeek 2025）是一篇论文里的两个模型：

- **R1-Zero。** 从 DeepSeek-V3 基座模型出发。不做 SFT。直接施加 GRPO，奖励由两部分组成：*准确性奖励*（基于规则——最终答案是否解析为正确数字 / 代码是否通过单元测试）和*格式奖励*（补全是否把思维链包裹在 `<think>…</think>` 标签里）。经过数千步训练，平均回复长度从约 100 个 token 增长到约 10,000 个 token，数学基准分数攀升到接近 o1-preview 的水平。模型从零学会了推理。缺点是：它的思维链常常难以阅读、中英夹杂，且缺乏文风上的打磨。
- **R1。** 用一个四阶段流水线修复 R1-Zero 的可读性问题：
  1. **冷启动 SFT。** 收集几千条格式整洁的长思维链（CoT）示范，用它们对基座模型做有监督微调。这提供了一个可读的起点。
  2. **面向推理的 GRPO。** 施加 GRPO，奖励为准确性+格式奖励，再加一个*语言一致性*奖励以防止语言混杂。
  3. **拒绝采样 + 第二轮 SFT。** 从 RL 检查点采样约 60 万条推理轨迹，只保留最终答案正确且 CoT 可读的样本，再与约 20 万条非推理 SFT 样本（写作、问答、自我认知）合并，重新微调基座模型。
  4. **全谱 GRPO。** 再做一轮 RL，同时覆盖推理（基于规则的奖励）和通用对齐（基于偏好的有用性/无害性奖励）。

最终结果在 AIME 和 MATH-500 上达到 o1 水平，权重开源，并且小到足以蒸馏。同一篇论文还发布了六个蒸馏后的稠密模型（从 Qwen-1.5B 到 Llama-70B），方法是用 R1 的推理轨迹对学生模型做 SFT——学生侧完全不做 RL。在学生模型的规模上，蒸馏一个强大的 RL 教师始终胜过从零开始做 RL。

**为什么推理任务用 GRPO 而不是 PPO。** DeepSeekMath 论文（2024 年 2 月）给出三个原因：（1）不用训练价值网络，显存减半；（2）组基线天然适应推理任务产生的稀疏轨迹末端奖励；（3）按提示词归一化使得难度天差地别的题目之间的优势值可以相互比较，而 PPO 的单一 critic 做不到这一点。

**免搜索 vs 基于搜索。** 游戏领域已经分化：

- *长视野的完美信息游戏*（围棋、国际象棋）：仍然基于搜索。AlphaZero / MuZero 占据统治地位。
- *LLM 推理*：生产环境中尚无 MCTS；GRPO 在完整 rollout 上运行，推理时算力用 best-of-N。过程奖励模型（Process Reward Model，PRM）暗示步级搜索将被重新引入。

## 从零实现

`code/main.py` 中的代码实现了一个**微缩版 GRPO**——一个带有多组采样的多臂老虎机。算法与 LLM 上的完全相同，只是策略和环境更简单。它教会你*损失函数*和*组相对优势*——这正是 2025 年的创新所在。

### 第 1 步：一个极小的验证器环境

```python
QUESTIONS = [
    {"prompt": "q1", "correct": 3},
    {"prompt": "q2", "correct": 1},
]

def verify(prompt_idx, answer_token):
    return 1.0 if answer_token == QUESTIONS[prompt_idx]["correct"] else 0.0
```

在真实的 GRPO 里，验证器会运行单元测试或检查数学等式。

### 第 2 步：策略——每个提示词上对 K 个答案 token 做 softmax

```python
def policy_probs(theta, p_idx):
    return softmax(theta[p_idx])
```

等价于 LLM 在给定提示词条件下最后一层的输出。

### 第 3 步：组采样与组相对优势

```python
def grpo_step(theta, p_idx, G=8, beta=0.01, lr=0.1, rng=None):
    probs = policy_probs(theta, p_idx)
    samples = [sample(probs, rng) for _ in range(G)]
    rewards = [verify(p_idx, s) for s in samples]
    mean_r = sum(rewards) / G
    std_r = stddev(rewards) + 1e-8
    advs = [(r - mean_r) / std_r for r in rewards]

    for a, A in zip(samples, advs):
        grad = onehot(a) - probs
        for i in range(len(probs)):
            theta[p_idx][i] += lr * A * grad[i]
    # KL penalty: pull theta toward reference
    for i in range(len(probs)):
        theta[p_idx][i] -= beta * (theta[p_idx][i] - reference[p_idx][i])
```

组相对优势就是 2024 年 DeepSeek 的技巧。不需要 critic。"基线"是组内均值，归一化用组内标准差。

### 第 4 步：与 REINFORCE 基线对比（不依赖价值函数）

同样的设置，同样的算力，运行普通的 REINFORCE。GRPO 收敛更快、更稳定。

### 第 5 步：观察熵与 KL

与 RLHF 相同的诊断指标：对参考策略的平均 KL、策略熵、奖励随时间的变化。这些指标一旦稳定，训练就完成了。

## 常见陷阱

- **通过钻验证器空子进行奖励作弊。** GRPO 继承了 RLHF 的风险：如果验证器有错或可被利用，LLM 一定会找到这个漏洞。鲁棒的验证器（多组测试用例、形式化证明）非常重要。
- **组大小太小。** 组基线的方差按 `1/√G` 缩放。`G = 4` 以下时优势信号噪声很大；标准选择是 `G = 8` 到 `64`。
- **长度偏差。** 不同长度的 LLM 补全有不同的对数概率。按 token 数归一化，或使用序列级对数概率，或截断到最大长度。
- **纯自我博弈陷入循环。** 在一般和博弈上，AlphaZero 式训练可能卡在压制循环里。缓解方法是多样化的对手池（联赛训练，见第 10 课）。
- **搜索与策略不匹配。** AlphaZero 训练策略去模仿搜索输出。如果策略网络太小，无法表示搜索的分布，训练就会停滞。
- **算力门槛。** MuZero / AlphaZero 需要海量算力。一次消融实验往往就是数百 GPU 小时。学习用途有微缩版演示（例如四子棋上的 AlphaZero）。
- **验证器覆盖不足。** 让有 bug 的解法也能通过的单元测试，会强化这个 bug。要设计能捕捉边界情况的验证器。

## 生产实践

2026 年游戏 RL 的格局，按领域划分：

| 领域 | 主导方法 |
|--------|-----------------|
| 双人零和棋类游戏（围棋、国际象棋、将棋） | AlphaZero / MuZero / KataGo |
| 不完美信息卡牌游戏（扑克） | CFR + 深度学习（DeepStack、Libratus、Pluribus） |
| Atari / 像素游戏 | Muesli / MuZero / IMPALA-PPO |
| 大型多人策略游戏（Dota、星际争霸） | PPO + 自我博弈 + 联赛（OpenAI Five、AlphaStar） |
| LLM 数学/代码推理 | GRPO（DeepSeek-R1、Qwen-RL、各开源复现） |
| LLM 对齐 | DPO / RLHF-PPO（不用 GRPO；验证器是偏好而非可验证信号） |
| 机器人 | PPO + 域随机化（不属于游戏 RL，但用同一套策略梯度工具） |
| 组合优化问题 | AlphaZero 变体（AlphaTensor、AlphaDev） |

这套*配方*——自我博弈、搜索增强的改进、策略蒸馏——横跨文本、像素和物理控制。GRPO 是其中最年轻的实例；更多实例还在路上。

## 交付产物

保存为 `outputs/skill-game-rl-designer.md`：

```markdown
---
name: game-rl-designer
description: Design a game-RL or reasoning-RL training pipeline (AlphaZero / MuZero / GRPO) for a given domain.
version: 1.0.0
phase: 9
lesson: 12
tags: [rl, alphazero, muzero, grpo, self-play]
---

Given a target (perfect-info game / imperfect-info / Atari / LLM reasoning / combinatorial), output:

1. Environment fit. Known rules? Markov? Stochastic? Multi-agent? Informs AlphaZero vs MuZero vs GRPO.
2. Search strategy. MCTS (PUCT with learned prior), Gumbel-sampled, best-of-N, or none.
3. Self-play plan. Symmetric self-play / league / offline data / verifier-generated.
4. Target signal. Game outcome / verifier reward / preference / learned model. Include robustness plan.
5. Diagnostics. Win rate vs baseline, ELO curve, verifier pass rate, KL to reference.

Refuse AlphaZero on imperfect-info games (route to CFR). Refuse GRPO without a trusted verifier. Refuse any game-RL pipeline without a fixed baseline opponent set (self-play ELO is uncalibrated otherwise).
```

## 练习

1. **简单。** 在 `code/main.py` 中实现 GRPO 老虎机。在 2 个提示词 × 各 4 个答案 token 上训练。要求在 `G=8` 下少于 1,000 次更新内收敛。
2. **中等。** 接入 PPO（带裁剪）和原版 REINFORCE。在同一个老虎机上对比它们与 GRPO 的样本效率和奖励方差。
3. **困难。** 扩展为长度为 2 的"推理链"：智能体输出两个 token，验证器对这一对 token 给出奖励。测量 GRPO 在两步序列上如何处理信用分配。（提示：按*完整序列*计算组优势，再传播到两个 token 位置。）

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|-----------------|-----------------------|
| MCTS | "用学习到的网络做树搜索" | 蒙特卡洛树搜索；用学到的 `(p, v)` 先验做 UCB1/PUCT 选择。 |
| AlphaZero | "自我博弈 + MCTS" | 训练策略-价值网络去匹配 MCTS 访问分布和对局结果。 |
| MuZero | "学习模型版 AlphaZero" | 同一个循环，但通过学到的动力学在潜在空间中运行。 |
| GRPO | "不带 critic 的 PPO" | 组相对策略优化（Group Relative Policy Optimization）；带组均值基线 + KL 的 REINFORCE。 |
| PUCT | "AlphaZero 的 UCB" | `Q + c · p · √N / (1 + N_a)` —— 在价值估计与先验之间取得平衡。 |
| 自我博弈 | "智能体对战过去的自己" | 零和博弈的标准做法；提供对称的训练信号。 |
| 联赛训练 | "基于种群的自我博弈" | 从过去版本 + 当前版本 + 针对性剥削者中采样对手。 |
| 验证器奖励 | "可验证 RL" | 奖励来自确定性的检查器（测试通过、答案匹配）。 |
| 过程奖励 | "PRM" | 给每个推理步骤打分，而不只看最终答案。 |

## 延伸阅读

- [Silver et al. (2017). Mastering the game of Go without human knowledge (AlphaGo Zero)](https://www.nature.com/articles/nature24270)。
- [Silver et al. (2018). A general reinforcement learning algorithm that masters chess, shogi, and Go through self-play (AlphaZero)](https://www.science.org/doi/10.1126/science.aar6404)。
- [Schrittwieser et al. (2020). Mastering Atari, Go, chess and shogi by planning with a learned model (MuZero)](https://www.nature.com/articles/s41586-020-03051-4)。
- [Vinyals et al. (2019). Grandmaster level in StarCraft II (AlphaStar)](https://www.nature.com/articles/s41586-019-1724-z)。
- [DeepSeek-AI (2024). DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models (GRPO)](https://arxiv.org/abs/2402.03300) —— 提出 GRPO 和组相对基线的论文。
- [DeepSeek-AI (2025). DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning](https://arxiv.org/abs/2501.12948) —— 完整的四阶段 R1 配方以及 R1-Zero 消融实验。
- [Brown et al. (2019). Superhuman AI for multiplayer poker (Pluribus)](https://www.science.org/doi/10.1126/science.aay2400) —— 大规模的 CFR + 深度学习。
- [Tesauro (1995). Temporal Difference Learning and TD-Gammon](https://dl.acm.org/doi/10.1145/203330.203343) —— 开创这一切的论文。
- [Hugging Face TRL — GRPOTrainer](https://huggingface.co/docs/trl/main/en/grpo_trainer) —— 使用自定义奖励函数应用 GRPO 的生产级参考实现。
- [Qwen Team (2024). Qwen2.5-Math — GRPO replication](https://github.com/QwenLM/Qwen2.5-Math) —— 在多个规模上对 R1 配方的开源复现。
- [Sutton & Barto (2018). Ch. 17 — Frontiers of Reinforcement Learning](http://incompleteideas.net/book/RLbook2020.pdf) —— 关于自我博弈、搜索与"设计的奖励"的教科书框架，R1 在 LLM 规模上将其变为现实。
