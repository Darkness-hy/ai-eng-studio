# 仿真到现实的迁移（Sim-to-Real Transfer）

> 在模拟器里训练好、上了硬件却失效的策略，本质上只是记住了模拟器。域随机化、域自适应和系统辨识，是让学习到的控制器跨越「现实鸿沟」的三件工具。

**Type:** Learn
**Languages:** Python
**Prerequisites:** Phase 9 · 08 (PPO), Phase 2 · 10 (Bias/Variance)
**Time:** ~45 minutes

## 问题背景

训练一台真实机器人既慢、又危险、还昂贵。一台双足机器人要花几百万个训练回合才能学会走路；而真实的双足机器人哪怕摔倒一次都可能损坏硬件。仿真则提供了无限次重置、确定性的可复现性、并行环境，并且不会造成任何物理损伤。

但模拟器是不准的。轴承的摩擦力比 MuJoCo 模型里的更大。相机有镜头畸变，模拟器并未建模。电机存在延迟、齿隙和饱和，99% 的仿真模型都把这些省略了。风、灰尘和多变的光照会让在「无菌渲染」下训练出来的策略失灵。**现实鸿沟（reality gap）**——仿真分布与真实分布之间的系统性差异——是机器人强化学习落地部署的核心问题。

你需要的是一个*对 sim-to-real 分布偏移具备鲁棒性*的策略。历史上有三种思路：随机化模拟器（域随机化）、用少量真实数据调整策略（域自适应 / 微调），或者辨识真实系统的参数并让仿真与之匹配（系统辨识）。到了 2026 年，主流方案是把这三者与大规模并行仿真（Isaac Sim、Isaac Lab、GPU 上的 Mujoco MJX）结合起来。

## 核心概念

![Three sim-to-real regimes: domain randomization, adaptation, system identification](../assets/sim-to-real.svg)

**域随机化（Domain Randomization, DR）。** Tobin et al. 2017、Peng et al. 2018。在训练期间，随机化所有可能与真实机器人存在差异的仿真参数：质量、摩擦系数、电机 PD 增益、传感器噪声、相机位置、光照、纹理、接触模型。策略学到的是关于「今天身处哪个模拟器」的条件分布，从而在整个参数范围内泛化。只要真实机器人落在训练包络之内，策略就能工作。

- **优点：** 不需要真实数据。一套配方，适用多种机器人。
- **缺点：** 随机化过度会得到一个「万能」却过分保守的策略。噪声太多 ≈ 正则化太强。

**系统辨识（System Identification, SI）。** 在训练之前，用真实世界的数据来拟合模拟器的参数。如果能在真实机器人上测出机械臂关节的摩擦力，就把它填进仿真里，然后训练一个以这些数值为前提的策略。需要能接触到真实系统，但能直接缩小现实鸿沟。

- **优点：** 训练目标精确、低噪声。
- **缺点：** 残余的模型误差对策略不可见；微小的未辨识效应（例如电机死区）仍会让部署失败。

**域自适应（Domain Adaptation）。** 先在仿真中训练，再用少量真实数据微调。有两种形式：

- **Real2Sim2Real：** 利用真实轨迹学习一个残差模拟器 `f(s, a, z) - f_sim(s, a)`，再在修正后的仿真中训练。无需太多真实数据就能缩小差距。
- **观测自适应：** 训练一个通过学习到的特征提取器（例如逐像素的 GAN）把真实观测映射为类仿真观测的策略。控制器本身始终留在仿真里。

**特权学习 / 师生范式（Privileged learning / teacher-student）。** Miki et al. 2022（ANYmal 四足机器人）。在仿真中训练一个能访问特权信息（真实摩擦力、地形高度、IMU 漂移）的*教师*策略，再蒸馏出一个只能看到真实传感器观测的*学生*策略。学生学会从历史观测中推断特权特征，从而对各种物理参数具备鲁棒性。

**大规模并行仿真。** 2024–2026。Isaac Lab、Mujoco MJX、Brax 都能在单块 GPU 上跑数千个并行机器人。用 4,096 个并行人形机器人跑 PPO，几小时就能收集相当于数年的经验。训练分布越宽，「现实鸿沟」就越小；当这 4,096 个环境各自带有不同的随机化参数时，DR 几乎是免费的。

**2026 年的真实世界配方（以四足行走为例）：**

1. 大规模并行仿真，对重力、摩擦力、电机增益、负载做域随机化。
2. 用特权信息（地形图、机身速度真值）训练教师策略。
3. 从教师蒸馏出只用本体感受（腿部关节编码器）的学生策略。
4. 可选：用自编码器对真实 IMU 做观测自适应。
5. 部署。在 10 个以上环境中零样本（zero-shot）运行。如果失败，用带安全约束的 PPO 做几分钟的真实世界微调。

## 从零实现

本课的代码是在带*噪声*转移的 GridWorld 上做域随机化的小型演示。我们训练一个在「仿真」中经历随机化打滑概率的策略，然后在「真实」环境中用训练时从未见过的打滑水平做评估。这个结构可以直接映射到从 MuJoCo 到硬件的迁移。

### 第 1 步：参数化的模拟器

```python
def step(state, action, slip):
    if rng.random() < slip:
        action = random_perpendicular(action)
    ...
```

`slip` 是模拟器暴露出来的一个参数。在真实机器人场景中，它可以是摩擦力、质量、电机增益——任何在仿真与真实之间会发生偏移的量。

### 第 2 步：用 DR 训练

在每个回合开始时，采样 `slip ~ Uniform[0.0, 0.4]`。用 PPO / Q-learning / 任意算法训练，重复许多回合。

### 第 3 步：在「真实」打滑水平上做零样本评估

在 `slip ∈ {0.0, 0.1, 0.2, 0.3, 0.5, 0.7}` 上评估。前四个在训练支撑集内；`0.5` 和 `0.7` 在支撑集外。经过 DR 训练的策略应该在支撑集内保持接近最优，在支撑集外平缓退化。而在固定打滑值上训练的策略，一旦超出其训练打滑值就会变得脆弱。

### 第 4 步：与窄分布训练对比

再训练第二个策略，只用 `slip = 0.0`。在同一组 `slip` 上扫描评估。你会看到：只要真实打滑值 > 0，性能就会灾难性下跌。

## 常见陷阱

- **随机化过度。** 在 `slip ∈ [0, 0.9]` 上训练，策略会规避风险到从不尝试最优路径的地步。要匹配*预期的*真实世界分布，而不是「什么都可能发生」。
- **随机化不足。** 只在一个窄区间上训练，策略完全无法泛化。可以使用自适应课程（自动域随机化，Automatic Domain Randomization），随着策略进步逐步加宽分布。
- **参数空间选错。** 随机化了错误的东西（真实差距在电机延迟，你却随机化相机色调），DR 就帮不上忙。先对真实机器人做剖析。
- **特权信息泄漏。** 如果教师的动作依赖全局状态而不只是观测，蒸馏出的学生可能永远追不上。要确保在给定观测历史的条件下，教师的策略对学生是可实现的。
- **sim-to-sim 迁移失败。** 如果你的策略对一个更难的仿真变体都不鲁棒，那它对真实世界也不会鲁棒。部署前务必在留出的仿真变体上测试。
- **缺少真实世界安全包络。** 一个在仿真中工作、在真实中「也工作」但没有底层安全护盾的策略，仍然可能损坏硬件。要在非学习的控制器里加上速率限制、力矩限制和关节限位。

## 生产实践

2026 年的 sim-to-real 技术栈：

| 领域 | 技术栈 |
|--------|-------|
| 足式运动（ANYmal、Spot、人形机器人） | Isaac Lab + DR + 特权教师 / 学生 |
| 操作任务（灵巧手、抓取放置） | Isaac Lab + DR + 用于视觉的 DR-GAN |
| 自动驾驶 | CARLA / NVIDIA DRIVE Sim + DR + 真实数据微调 |
| 无人机竞速 | RotorS / Flightmare + DR + 在线自适应 |
| 手指 / 手内操作 | OpenAI Dactyl（空前规模的 DR） |
| 工业机械臂 | MuJoCo-Warp + SI + 少量真实数据微调 |

对各种规模的控制任务，工作流是一致的：先尽力拟合模拟器，拟合不了的就随机化，训练巨大的策略，蒸馏，再带着安全护盾部署。

## 交付产物

保存为 `outputs/skill-sim2real-planner.md`：

```markdown
---
name: sim2real-planner
description: Plan a sim-to-real transfer pipeline for a given robot + task, covering DR, SI, and safety.
version: 1.0.0
phase: 9
lesson: 11
tags: [rl, sim2real, robotics, domain-randomization]
---

Given a robot platform, a task, and access to real hardware time, output:

1. Reality gap inventory. Suspected sources ranked by expected impact (contact, sensing, actuation delay, vision).
2. DR parameters. Exact list, ranges, distribution. Justify each range against real measurements.
3. SI steps. Which parameters to measure; measurement method.
4. Teacher/student split. What privileged info the teacher uses; what obs the student uses.
5. Safety envelope. Low-level limits, emergency stops, backup controller.

Refuse to deploy without (a) a zero-shot sim-variant test, (b) a safety shield, (c) a rollback plan. Flag any DR range wider than 3× measured real variability as likely over-randomized.
```

## 练习

1. **简单。** 在固定打滑值的 GridWorld（slip=0.0）上训练一个 Q-learning 智能体。在 slip ∈ {0.0, 0.1, 0.3, 0.5} 上评估。绘制回报与打滑值的关系曲线。
2. **中等。** 训练一个 DR Q-learning 智能体，采样 `slip ~ Uniform[0, 0.3]`。在同样的扫描区间上评估。在 slip=0.5（分布外）时，DR 带来了多少收益？
3. **困难。** 实现一个课程：从 slip=0.0 开始，每当策略达到最优值的 90% 就加宽 DR 范围。测量达到 slip=0.3 零样本性能所需的总环境步数，并与固定 DR 基线对比。

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|-----------------|-----------------------|
| 现实鸿沟（reality gap） | 「仿真与现实的差异」 | 训练与部署在物理 / 感知上的分布偏移。 |
| 域随机化（DR） | 「在随机模拟器上训练」 | 训练期间随机化仿真参数，使策略具备泛化能力。 |
| 系统辨识（SI） | 「测量真实系统并拟合仿真」 | 估计真实物理参数；让仿真与之匹配。 |
| 域自适应 | 「在真实数据上微调」 | 仿真训练后用少量真实数据微调；可以自适应观测或动力学。 |
| 特权信息 | 「给教师的真值」 | 只有仿真才有的信息；学生必须从观测历史中推断它。 |
| 师生范式 | 「把特权信息蒸馏成可观测的」 | 教师借助捷径训练；学生学会在没有捷径的情况下模仿。 |
| ADR | 「自动域随机化」 | 随着策略进步而加宽 DR 范围的课程机制。 |
| Real2Sim | 「用真实数据缩小差距」 | 学习一个残差，让仿真模仿真实轨迹。 |

## 延伸阅读

- [Tobin et al. (2017). Domain Randomization for Transferring Deep Neural Networks from Simulation to the Real World](https://arxiv.org/abs/1703.06907) —— 最早的 DR 论文（面向机器人的视觉）。
- [Peng et al. (2018). Sim-to-Real Transfer of Robotic Control with Dynamics Randomization](https://arxiv.org/abs/1710.06537) —— 针对动力学的 DR，四足运动。
- [OpenAI et al. (2019). Solving Rubik's Cube with a Robot Hand](https://arxiv.org/abs/1910.07113) —— Dactyl，规模化的 ADR。
- [Miki et al. (2022). Learning robust perceptive locomotion for quadrupedal robots in the wild](https://www.science.org/doi/10.1126/scirobotics.abk2822) —— ANYmal 的师生范式。
- [Makoviychuk et al. (2021). Isaac Gym: High Performance GPU Based Physics Simulation for Robot Learning](https://arxiv.org/abs/2108.10470) —— 驱动 2025–2026 年各类部署的大规模并行模拟器。
- [Akkaya et al. (2019). Automatic Domain Randomization](https://arxiv.org/abs/1910.07113) —— ADR 课程方法。
- [Sutton & Barto (2018). Ch. 8 — Planning and Learning with Tabular Methods](http://incompleteideas.net/book/RLbook2020.pdf) —— Dyna 框架（用模型做规划和轨迹推演），是现代 sim-to-real 流水线的理论基础。
- [Zhao, Queralta & Westerlund (2020). Sim-to-Real Transfer in Deep Reinforcement Learning for Robotics: a Survey](https://arxiv.org/abs/2009.13303) —— sim-to-real 方法分类综述，附基准测试结果。
