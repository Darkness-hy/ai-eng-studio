# 多智能体强化学习（MARL）— MADDPG、QMIX、MAPPO

> 多智能体协同的强化学习传统，至今仍在影响 2026 年的 LLM 智能体系统。**MADDPG**（Lowe et al., NeurIPS 2017, arXiv:1706.02275）提出了集中式训练、分散式执行（Centralized Training, Decentralized Execution, CTDE）：训练时每个评论家（critic）可以看到所有智能体的状态和动作；测试时只运行各自的本地行动者（actor）。适用于合作、竞争和混合场景。**QMIX**（Rashid et al., ICML 2018, arXiv:1803.11485）是带单调混合网络的值分解方法；各智能体的 Q 值组合成联合 Q，使 `argmax` 可以干净地分解到每个智能体——在 StarCraft Multi-Agent Challenge（SMAC）上长期占优。**MAPPO**（Yu et al., NeurIPS 2022, arXiv:2103.01955）是带集中式价值函数的 PPO；在 particle-world、SMAC、Google Research Football、Hanabi 上只需极少调参就「出人意料地有效」。这些方法支撑着「必须分散行动的智能体团队」的策略训练。MAPPO 是 **2026 年合作型 MARL 的默认基线**。本课通过一个小型网格世界玩具示例从零构建这三种思想，在接触 LLM 智能体训练之前先把它们刻进肌肉记忆。

**Type:** Learn
**Languages:** Python (stdlib, small NumPy-free implementations)
**Prerequisites:** Phase 09 (Reinforcement Learning), Phase 16 · 09 (Parallel Swarm Networks)
**Time:** ~90 minutes

## 问题背景

LLM 智能体系统越来越多地需要为智能体间协同训练策略：什么时候让步、什么时候行动、该调用哪个同伴。告诉你如何训练这类策略的文献就是多智能体强化学习（Multi-Agent Reinforcement Learning, MARL）——它早于 LLM 浪潮出现，并且有一小批占主导地位的算法。

不掌握这套模式词汇就去读 MARL 论文会非常痛苦。集中式训练分散式执行（CTDE）、值分解、集中式评论家都不是空泛的流行词——它们是针对具体问题的具体答案：

- 独立强化学习（每个智能体各学各的）从每个智能体的视角看环境是非平稳的。糟糕。
- 集中式强化学习（一个智能体控制所有人）无法扩展，且违反执行约束。
- CTDE 兼得两者之长：用全局信息训练，用本地策略部署。

## 核心概念

### 论文常用的三类环境

- **Particle World（多智能体粒子环境）。** 简单的二维物理环境，包含合作/竞争任务。MADDPG 的原始测试平台。
- **StarCraft Multi-Agent Challenge（SMAC）。** 合作型微操任务，部分可观测。QMIX 的测试平台。离散动作，连续状态。
- **Google Research Football、Hanabi、MPE。** MAPPO 的基准环境。

不同环境有不同的动作/观测类型。各算法据此选择适用场景。

### MADDPG（2017）— CTDE 模式

每个智能体 `i` 有一个行动者 `mu_i(o_i)`，把自己的观测映射为动作。每个智能体还有一个评论家 `Q_i(x, a_1, ..., a_n)`，在训练时能看到所有观测和所有动作。行动者根据评论家的评估通过策略梯度更新。

```
actor update:    grad_theta_i J = E[grad_theta mu_i(o_i) * grad_a_i Q_i(x, a_1..n) at a_i=mu_i(o_i)]
critic update:   TD on Q_i(x, a_1..n) given next-state joint estimate
```

为什么用 CTDE：训练时我们知道所有人的动作，可以利用它来降低每个评论家的方差。部署时，每个智能体只看到 `o_i`，只调用 `mu_i(o_i)`。

失效模式：评论家的规模随智能体数量 N 增长（输入包含所有动作）。不做近似的话，超过约 10 个智能体就难以扩展。

### QMIX（2018）— 值分解

仅适用于合作场景。全局奖励是各智能体 Q 值经单调函数组合的结果：

```
Q_tot(tau, a) = f(Q_1(tau_1, a_1), ..., Q_n(tau_n, a_n)),   df/dQ_i >= 0
```

单调性保证了 `argmax_a Q_tot` 可以由每个智能体独立选取 `argmax_{a_i} Q_i` 来计算。这**正是你需要的分散式执行性质**。训练时，由一个混合网络（mixing network）从各智能体的 Q 值生成 `Q_tot`。

为什么 QMIX 能在 SMAC 上获胜：合作型 StarCraft 微操任务具有同质智能体、局部观测、全局奖励——和值分解完美契合。

失效模式：单调性约束有局限；有些任务的奖励结构无法单调分解（比如一个智能体为团队牺牲自己）。后续扩展（QTRAN、QPLEX）放宽了这一约束。

### MAPPO（2022）— 被忽视的默认选项

多智能体 PPO（Multi-Agent PPO）：带集中式价值函数的 PPO。每个智能体有自己的策略；所有智能体共享（或各自拥有）能看到全局状态的价值函数。Yu et al. 2022 在五个基准上把 MAPPO 与 MADDPG、QMIX 及其扩展进行了对比，发现：

- MAPPO 在 particle-world、SMAC、Google Research Football、Hanabi、MPE 上追平或超越离策略（off-policy）MARL 方法。
- 几乎不需要超参数调优。
- 训练稳定，跨随机种子可复现。

在这篇论文之前，社区一直低估在策略（on-policy）MARL。到 2026 年，MAPPO 已是合作型 MARL 的默认基线；任何新方法都必须先打败它。

### LLM 智能体工程师为什么要关心这些

三个直接用途：

1. **路由器训练。** 一个元智能体（meta-agent）决定由哪个子智能体处理任务。这本质上是一个 MARL 问题：N 个分散的子智能体加一个集中式路由器。MAPPO 正好适用。
2. **角色涌现。** 在生成式智能体（generative-agent）模拟中，训练智能体随时间形成互补角色，本质上就是伪装成别的问题的 MARL。QMIX 式的值分解从结构上强制了互补性。
3. **多智能体工具使用。** 当多个智能体共享工具并争夺预算时，用 CTDE 训练它们能得到可部署的、遵守资源约束的本地策略。

实践提醒：在 2026 年，大多数生产环境的 LLM 智能体系统是通过提示词来设定策略，而不是训练策略。MARL 的用武之地在于：（a）有大量交互数据，（b）有清晰的奖励信号，（c）愿意投入训练基础设施。

### CTDE 作为超越 RL 的设计模式

即使不做训练，CTDE 也是一个有用的架构模式：

- 在*设计阶段*，假设可以看到整个团队的信息。
- 在*运行时*，强制分散式执行：每个智能体只看到 `o_i`。

这个模式迫使你把每个智能体的状态显式化，并提前考虑部分可观测性。许多生产多智能体系统在各处默默假设状态共享——CTDE 纪律可以防止这种情况。

### 非平稳性问题

当多个智能体同时学习时，每个智能体的环境（包含其他智能体的策略）都是非平稳的。经典单智能体 RL 的证明在此失效。本课中的 MARL 算法都在应对这个问题：

- MADDPG：全局评论家看到所有动作，因此它的价值估计是平稳的。
- QMIX：值分解把学习转移到联合 Q 空间，在那里最优性有明确定义。
- MAPPO：集中式价值函数抑制了其他智能体策略变化带来的方差。

在 LLM 智能体系统中，非平稳性的表现是：「我的智能体上个月还好好的，上游另一个智能体一改，我的就出问题了。」用 CTDE 训练 MARL 是有原则的修复方式；提示词层面的修补更快，但不够持久。

### 本课不涵盖的内容

训练真实神经网络是 Phase 09 的主题。本课构建的是脚本化策略版本，在不做梯度更新的前提下演示 CTDE、值分解和集中式价值这三种模式。目标是先内化这些模式，再去用完整的 MARL 库（PyMARL、MARLlib、RLlib multi-agent）。

## 从零实现

`code/main.py` 实现了三种模式演示，全部跑在一个小型的双智能体合作网格世界上：

- 环境：4x4 网格上有 2 个智能体和一个奖励豆子。任一智能体到达豆子位置奖励为 1，任务结束。
- `IndependentAgents` —— 每个智能体把其他智能体当作环境的一部分。基线。
- `MADDPGStyle` —— 集中式评论家计算联合价值；行动者策略据此更新。脚本化的策略改进。
- `QMIXStyle` —— 带单调混合器的值分解。
- `MAPPOStyle` —— 集中式价值函数；策略相对于共享基线进行更新。

四者运行相同的回合（episode）并报告到达目标的平均步数。CTDE 变体收敛到的路径比独立基线更短。

运行：

```
python3 code/main.py
```

预期输出：独立智能体平均约需 6 步；CTDE 变体收敛到约 3.5 步（4x4 网格的最优解是 3 步）。即便策略是脚本化的，模式之间的差异依然显现。

## 生产实践

`outputs/skill-marl-picker.md` 是一个技能，用于针对给定的多智能体任务选择 MARL 算法：合作还是竞争、同质还是异质、动作空间类型、规模、奖励信号。

## 交付产物

MARL 在生产环境中很少见。当你确实要用时：

- **从 MAPPO 开始。** 2022 年那篇论文已确立它为基线；先复现它，可以省下数周追逐更花哨方法的时间。
- **记录每个智能体的观测流和动作流。** 没有逐智能体的轨迹，调试 MARL 毫无希望。
- **训练代码与执行代码分离。** CTDE 是一种纪律；要让执行路径真正只看到 `o_i`。
- **奖励塑形（reward shaping）警告。** MARL 对奖励设计极其敏感。塑形里出一个协同 bug，智能体就会学会钻空子。要做对抗性测试。
- **对于 LLM 智能体**，先考虑提示词层面的策略。只有当交互数据、奖励信号和基础设施三者齐备时，才投入 MARL 训练。

## 练习

1. 运行 `code/main.py`。测量独立智能体与 MAPPO 式智能体之间的到达目标步数差距。在 6x6 网格上，这个差距是变大还是变小？
2. 实现一个竞争变体：两个智能体、一个豆子，只有先到的才获得奖励。哪种模式能干净地处理竞争？从历史上看是 MADDPG。
3. 阅读 MADDPG（arXiv:1706.02275）第 3 节。用你自己的话，以伪代码符号化地实现确切的评论家更新规则。
4. 阅读 MAPPO（arXiv:2103.01955）。作者为什么认为集中式价值 + PPO 在他们的基准上优于离策略 MARL？列出三个最有力的论点。
5. 把 CTDE 作为设计模式应用于一个假想的 LLM 智能体系统（例如研究智能体 + 摘要智能体 + 编码智能体）。设计阶段可用、但运行时不可用的联合信息是什么？

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|----------------|------------------------|
| MARL | 「多智能体 RL」 | 面向多智能体系统的强化学习。 |
| CTDE | 「集中式训练、分散式执行」 | 用全局信息训练；用本地策略部署。 |
| MADDPG | 「多智能体 DDPG」 | CTDE，每个智能体的评论家能看到所有观测和动作。 |
| QMIX | 「值分解」 | 对各智能体 Q 值做单调混合。仅限合作场景。 |
| MAPPO | 「多智能体 PPO」 | 带集中式价值函数的 PPO。2026 年默认基线。 |
| 值分解 | 「单个 Q 的求和」 | 联合 Q 表示为各智能体 Q 的单调函数。 |
| 非平稳性 | 「移动靶」 | 随着其他智能体学习，每个智能体的环境在变化。MARL 的核心难题。 |
| 在策略 / 离策略 | 「从当前数据学 / 从回放学」 | PPO 是在策略的（MAPPO）；DDPG 和 Q-learning 是离策略的。 |
| SMAC | 「StarCraft Multi-Agent Challenge」 | 合作型微操基准；QMIX 的发源地。 |

## 延伸阅读

- [Lowe et al. — Multi-Agent Actor-Critic for Mixed Cooperative-Competitive Environments](https://arxiv.org/abs/1706.02275) — MADDPG；NeurIPS 2017
- [Rashid et al. — QMIX: Monotonic Value Function Factorisation for Deep Multi-Agent Reinforcement Learning](https://arxiv.org/abs/1803.11485) — QMIX；ICML 2018
- [Yu et al. — The Surprising Effectiveness of PPO in Cooperative Multi-Agent Games](https://arxiv.org/abs/2103.01955) — MAPPO；NeurIPS 2022
- [BAIR blog post on MAPPO](https://bair.berkeley.edu/blog/2021/07/14/mappo/) — 对 MAPPO 结果的易读解读
- [SMAC repository](https://github.com/oxwhirl/smac) — StarCraft Multi-Agent Challenge
