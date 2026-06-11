# 直接偏好优化（DPO）家族

> Rafailov 等人（2023）证明了带 KL 约束的 RLHF 最优解可以用偏好数据写成闭式解，因此可以跳过显式奖励模型，直接优化策略。这一洞见催生了一个算法家族——IPO、KTO、SimPO、ORPO、BPO——每个成员都修复了 DPO 的某种失效模式。到 2026 年，直接对齐算法（direct alignment algorithm）在前沿模型后训练中的使用率已超过 PPO。但第 2 课中的过度优化曲线依然适用：DAA 并没有逃脱 Goodhart 定律，只是改变了它发作的位置。

**Type:** Learn
**Languages:** Python (stdlib, six-variant preference-loss comparator)
**Prerequisites:** Phase 18 · 01 (InstructGPT), Phase 18 · 02 (Reward hacking), Phase 10 · 08 (DPO basics)
**Time:** ~75 minutes

## 学习目标

- 从带 KL 约束的 RLHF 最优解推导出 DPO 的闭式形式。
- 说出 IPO、KTO、SimPO、ORPO、BPO 各自修复了 DPO 的哪种失效模式。
- 区分「隐式奖励差距」与「偏好强度」，并解释 IPO 的恒等映射为何重要。
- 解释为什么 Rafailov 等人（NeurIPS 2024）能证明 DAA 即使没有显式奖励模型也会过度优化。

## 问题背景

RLHF 目标函数（第 1 课）：

```
max_pi E_{x,y~pi} [ r(x, y) ] - beta * KL(pi || pi_ref)
```

有已知的最优解：

```
pi*(y|x) = (1/Z(x)) * pi_ref(y|x) * exp(r(x, y) / beta)
```

因此，奖励被隐式地定义为最优策略与参考策略的比值：

```
r(x, y) = beta * log(pi*(y|x) / pi_ref(y|x)) + beta * log Z(x)
```

把它代入 Bradley-Terry 偏好似然后，配分函数 `Z(x)` 会相互抵消，因为它只依赖于 `x`。剩下的就是一个只含策略参数的损失函数——不再需要奖励模型。这就是 DPO。

但有个隐患：这个推导假设最优解可达、偏好数据在分布内、参考策略是真实的模式锚点。这三个假设没有一个能严格成立。家族中的每个成员修复的正是其中某个被违反的假设。

## 核心概念

### DPO（Rafailov 等人，2023）

```
L_DPO = -log sigmoid(
  beta * log(pi(y_w | x) / pi_ref(y_w | x))
  - beta * log(pi(y_l | x) / pi_ref(y_l | x))
)
```

可能出问题的地方：

- 隐式奖励差距 `beta * (log(pi/pi_ref)_w - log(pi/pi_ref)_l)` 是无界的。一个微小的偏好就可能产生任意大的差距。
- 这个损失把被选中（chosen）和被拒绝（rejected）回复的对数概率往相反方向推。只要被拒绝回复的对数概率下降得更快，它甚至可以把被选中回复的绝对对数概率往下压。这就是「被选中回复退化」（Degraded Chosen Response）现象。
- 分布外的偏好对（罕见样本对罕见样本）会产生任意的隐式奖励。

### IPO（Azar 等人，2024）

恒等偏好优化（Identity Preference Optimization）把 log-sigmoid 换成作用在偏好概率上的恒等映射。损失变成对一个有界目标的平方误差：

```
L_IPO = (log(pi(y_w | x) / pi_ref(y_w | x)) - log(pi(y_l | x) / pi_ref(y_l | x)) - 1/(2 beta))^2
```

间隔上界为 `1/(2 beta)`。偏好强度和隐式奖励差距成正比，不会爆炸。

### KTO（Ethayarajh 等人，2024）

Kahneman-Tversky 优化彻底放弃了成对结构。给定单条带标签的输出和一个二元「可取」或「不可取」信号，它映射到一个前景理论（prospect theory）效用：

```
v(x, y) = sigma(beta * log(pi(y|x) / pi_ref(y|x)) - z_ref)
```

并对收益和损失使用不同的权重（损失厌恶）。好处：可以使用非成对数据，而这类数据要充裕得多。

### SimPO（Meng 等人，2024）

简单偏好优化（Simple Preference Optimization）让训练信号与生成过程对齐。完全去掉参考策略，并按长度归一化对数似然：

```
L_SimPO = -log sigmoid(
  (beta / |y_w|) * log pi(y_w | x)
  - (beta / |y_l|) * log pi(y_l | x)
  - gamma
)
```

并加上一个间隔 `gamma` 来稳定训练。长度归一化消除了利用 DPO 长度偏置失效模式的动机（按构造，更长的 `y_w` 自然带来更大的对数概率差距）。

### ORPO（Hong 等人，2024）

赔率比偏好优化（Odds-Ratio Preference Optimization）在标准 SFT 负对数似然上加一个偏好项：

```
L_ORPO = L_NLL(y_w) + lambda * L_OR
L_OR = -log sigmoid(log(odds(y_w) / odds(y_l)))
```

没有参考策略——SFT 项本身就是正则化项。从基座模型到对齐模型只需一个训练阶段，不需要单独的 SFT 检查点。

### BPO（ICLR 2026 投稿，OpenReview id=b97EwMUWu7）

指出了「被选中回复退化」（Degraded Chosen Responses）问题：DPO 保持了 `y_w > y_l` 的排序，但 `y_w` 的绝对对数概率可能下降。BPO 加了一行代码的修正，惩罚被选中回复对数概率的下降。论文报告在 Llama-3.1-8B-Instruct 的数学推理任务上比 DPO 提升 +10.1% 准确率。

### 普适结论：DAA 仍然会过度优化

Rafailov 等人在《Scaling Laws for Reward Model Overoptimization in Direct Alignment Algorithms》（NeurIPS 2024）中，用 DPO、IPO、SLiC 在多个数据集和不同 KL 预算下训练策略。黄金奖励对 KL 的曲线呈现出与 Gao 等人相同的「先升后崩」形状。隐式奖励在训练过程中会查询分布外样本；KL 正则化无法稳定这一点。

DAA 没有逃脱 Goodhart 定律。它们只是把出问题的表面从「奖励模型被过度优化」换成了「参考策略比值被过度优化」。通用的对策——更好的数据、集成、早停——对两者同样适用。

### 如何在它们之间选择（2026）

- 如果有大规模成对偏好数据：用保守 beta 的 DPO；若长度偏置明显，用 SimPO。
- 如果只有非成对的二元反馈：KTO。
- 如果想要从基座模型出发的单阶段流水线：ORPO。
- 如果在 DPO 日志中看到被选中回复对数概率退化：BPO。
- 如果偏好强度差异很大且 DPO 趋于饱和：IPO。

每个实验室都会在一组基准上把五种方法全跑一遍，再按任务挑出赢家。没有任何理由认为数学推理和安全任务的最优解是同一个。

```figure
dpo-margin
```

## 生产实践

`code/main.py` 在一个真实偏好强度因样本对而异的玩具偏好数据集上比较六种损失（DPO、IPO、KTO、SimPO、ORPO、BPO）。每种损失都用一个小型 softmax 策略在同样的 500 对样本上优化。绘制每种方法的最终胜率、被选中回复对数概率漂移和隐式奖励离散度。

## 交付产物

本课产出 `outputs/skill-preference-loss-selector.md`。给定数据集统计信息（成对 vs 非成对、偏好强度可变 vs 均匀、长度分布）和训练目标（单阶段或 SFT 后接偏好训练），推荐一种偏好损失，并报告它所防范的失效模式。

## 练习

1. 运行 `code/main.py`。报告 DPO 和 BPO 的最终被选中回复对数概率下降量。BPO 应当保持更高的被选中回复绝对概率——验证这一点。

2. 修改偏好数据，使所有样本对的偏好强度相等。六种方法中哪个最鲁棒？哪个会退化？解释 IPO 在这里的优势。

3. 让被拒绝回复的平均长度是被选中回复的 2 倍。在不改变其他任何设置的情况下，用数值展示 DPO 对长度的利用以及 SimPO 的修复效果。

4. Rafailov 等人（NeurIPS 2024）声称 DAA 会过度优化。复现一个单点版本：绘制被选中减去被拒绝的 KL 散度，观察 DPO 在大 beta 下的过度优化。

5. 阅读 BPO 论文摘要（OpenReview b97EwMUWu7）。写下 BPO 在 DPO 上添加的那一行修正。对照 `code/main.py` 中的实现进行确认。

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|-----------------|------------------------|
| DPO | 「没有奖励模型的 RLHF」 | 从 RLHF 闭式最优解推导出的损失；只含策略参数 |
| 隐式奖励 | 「对数比值」 | `beta * log(pi(y\|x) / pi_ref(y\|x))`——DPO 隐含的奖励 |
| IPO | 「有界的 DPO」 | 把 log-sigmoid 换成恒等映射；隐式奖励差距上界为 `1/(2 beta)` |
| KTO | 「非成对的 DPO」 | 基于前景理论的单标签效用函数，带损失厌恶 |
| SimPO | 「无参考的 DPO」 | 长度归一化对数似然 + 间隔；无参考策略 |
| ORPO | 「单阶段 DPO」 | NLL + 赔率比偏好项；从基座模型一次训完 |
| BPO | 「保护被选中回复的 DPO」 | DPO 加上对被选中回复绝对对数概率下降的惩罚 |
| 被选中回复退化 | 「chosen 在往下掉」 | 只要被拒绝回复下降得更快，DPO 就会降低被选中回复的对数概率 |
| DAA | 「直接对齐算法」 | 任何跳过显式奖励模型的偏好损失方法 |

## 延伸阅读

- [Rafailov et al. — Direct Preference Optimization (NeurIPS 2023, arXiv:2305.18290)](https://arxiv.org/abs/2305.18290)
- [Azar et al. — A General Theoretical Paradigm to Understand Learning from Human Preferences (AISTATS 2024, arXiv:2310.12036)](https://arxiv.org/abs/2310.12036) — IPO
- [Ethayarajh et al. — KTO: Model Alignment as Prospect Theoretic Optimization (arXiv:2402.01306)](https://arxiv.org/abs/2402.01306)
- [Meng, Xia, Chen — SimPO (NeurIPS 2024, arXiv:2405.14734)](https://arxiv.org/abs/2405.14734)
- [Hong, Lee, Thorne — ORPO (EMNLP 2024, arXiv:2403.07691)](https://arxiv.org/abs/2403.07691)
- [BPO — Behavior Preservation Optimization (ICLR 2026 OpenReview b97EwMUWu7)](https://openreview.net/forum?id=b97EwMUWu7)
- [Rafailov et al. — Scaling Laws for RM Overoptimization in DAAs (NeurIPS 2024, arXiv:2406.02900)](https://arxiv.org/abs/2406.02900)
