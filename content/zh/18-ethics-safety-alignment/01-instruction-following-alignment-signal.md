# 指令遵循作为对齐信号

> 后续对 RLHF 的每一条批评，针对的都是这条流水线。在研究优化压力如何扭曲代理目标之前，你得先看清这个代理目标本身。InstructGPT（Ouyang et al., 2022）定义了参考架构：在指令-回复对上做监督微调，用成对偏好排序训练奖励模型，再以 PPO 对奖励模型做优化，并对 SFT 策略施加 KL 惩罚。一个 1.3B 的 InstructGPT 在人类偏好上胜过了 175B 的 GPT-3。仅这一个结果，就是 2026 年所有前沿实验室仍在交付 RLHF 形态后训练流水线的原因。

**Type:** Learn
**Languages:** Python (stdlib, toy three-stage pipeline)
**Prerequisites:** Phase 10 · 06 (SFT), Phase 10 · 07 (RLHF), Phase 10 · 08 (DPO)
**Time:** ~45 minutes

## 学习目标

- 说出 InstructGPT 流水线的三个阶段，以及每个阶段使用的损失函数。
- 解释为什么一个 1.3B 的指令微调模型能在人类偏好评估中击败原始的 175B GPT-3。
- 说明第三阶段的 KL 惩罚在防范什么，以及为什么去掉它会坍缩成模式寻求（mode-seeking）行为。
- 描述对齐税（alignment tax），以及 Ouyang et al. 用来缓解它的 PPO-ptx 方案。

## 问题背景

预训练语言模型做的是文本续写，而不是回答问题。向 GPT-3 提问"write a Python function that reverses a list"，你常常得到的是另一个提示语，因为训练分布的主体是网页文本，而网页文本的后面接的还是网页文本。模型尽职地完成了它的工作——只是这份工作本身定义错了。

所有严肃的实验室用来修正这一点的代理目标，都是人类偏好。把两个补全结果交给标注员，标注员选出更好的那个，再训练一个奖励模型去学习标注员的判断。然后用一个 RL 循环把策略推向奖励模型打高分的输出。三句话就讲完了 InstructGPT 的全部论点，论文剩下的部分都是工程实现。

## 核心概念

### 阶段一：监督微调（SFT）

收集提示-回复对，回复内容是一位善意的人类会写出的回答。Ouyang et al. 使用了来自标注员和 OpenAI API 的 13k 条提示。用标准交叉熵损失在这批数据上微调基座模型。

SFT 给你的是：模型从此回答问题，而不是续写问题。SFT 不会给你的是：当多个回答都说得通时，标注员更偏好哪一个的任何信号。

### 阶段二：奖励模型（RM）

对每条提示，从 SFT 模型采样 K 个补全，由标注员排序。训练一个奖励模型为任意提示-回复对打分，使得对于 `y_w` 被偏好于 `y_l` 的样本对：

```
L_RM = -log sigmoid(r(x, y_w) - r(x, y_l))
```

这就是 Bradley-Terry 成对偏好损失。RM 通常从 SFT 模型初始化，并把语言模型头替换为标量输出头。

奖励模型可以很小：对 175B 的 InstructGPT 来说，6B 就够用了。它们也很脆弱——论文第 5 节大部分篇幅都在讲小规模实验中就已出现的奖励劫持（reward hacking）行为。

### 阶段三：带 KL 惩罚的 PPO

定义目标函数：

```
J(pi) = E_{x~D, y~pi(.|x)} [ r(x, y) ] - beta * KL(pi(.|x) || pi_SFT(.|x))
```

用 PPO 最大化它。KL 项约束 `pi` 不要漂离 SFT 策略太远。没有这一项，优化器会找到对抗样本——那些在 RM 下得高分的字符串，得高分只是因为 RM 从未见过它们，而不是因为人类真的偏好它们。

KL 系数 `beta` 是 RLHF 中最重要的单个超参数。太低：奖励劫持。太高：相比 SFT 毫无提升。

### 对齐税

经过 RLHF 后，模型更受人类偏好，但在标准基准（SQuAD、HellaSwag、DROP）上出现退步。Ouyang et al. 把这称为对齐税，并用 PPO-ptx 来修正：把预训练梯度混入 RL 目标，让模型不会忘记那些它从未因之获得奖励的下游任务能力。

```
J_ptx(pi) = J(pi) + gamma * E_{x~D_pretrain} [ log pi(x) ]
```

PPO-ptx 后来成了标准做法。Anthropic、DeepMind 和 Meta 都在使用某种变体。

### 结果

一个 1.3B 的 InstructGPT（SFT + RM + PPO-ptx）在大约 70% 的情况下被标注员偏好于 175B 的基座 GPT-3。在来自生产流量的隐藏测试提示上，差距还会进一步拉大。从这个数字可以读出两件事：

1. 对齐与能力是两条不同的轴。175B 模型能力更强，1.3B 模型对齐更好，标注员选了对齐更好的那个。
2. 能力的下限由基座模型决定。你无法靠 RLHF 让一个基座模型知道它从未见过的事实。

### 为什么这是 Phase 18 的参照点

后续课程中的每一条批评——奖励劫持（第 2 课）、DPO（第 3 课）、谄媚（sycophancy，第 4 课）、CAI（第 5 课）、潜伏特工（sleeper agents，第 7 课）、对齐伪装（alignment faking，第 9 课）——针对的都是这条流水线的某个环节。奖励劫持攻击阶段二。DPO 把阶段二和阶段三合并。CAI 替换掉人类标注员。谄媚表明标注员是一个有偏的信号。对齐伪装表明策略可以完全绕过阶段三。如果脑子里没有这条流水线，你就无法跟上其中任何一条批评。

## 生产实践

`code/main.py` 在玩具偏好数据上模拟这三个阶段。基座"策略"是一枚在动作集 {A, B, C} 上有偏的硬币。阶段一的 SFT 在 200 条提示上模仿标注员的动作。阶段二用 500 条成对排序拟合一个 Bradley-Terry 奖励模型。阶段三运行一个简化的 PPO 更新，并对 SFT 策略施加 KL 惩罚。你可以观察奖励攀升、KL 散度增长、策略漂移——还可以关闭 KL 项，在 50 个更新步以内亲眼看到奖励劫持的出现。

观察要点：

- `beta = 0.1` 与 `beta = 0.0` 下的奖励轨迹。
- KL(pi || pi_SFT) 随训练步数的变化。
- 最终动作分布与标注员偏好的对比。

## 交付产物

本课产出 `outputs/skill-instructgpt-explainer.md`。给定一段 RLHF 流水线描述或一篇论文摘要，它能识别出被修改的是三个阶段中的哪一个、每个阶段使用了什么损失，以及是否存在 KL 惩罚或等价的正则项。

## 练习

1. 运行 `code/main.py`。设 `beta = 0.0`，报告 200 个 PPO 步之后的动作分布。用一段话解释其中的模式寻求行为。

2. 修改奖励模型，给动作 B 加上 +0.5 的偏置（模拟一个奖励 bug）。以 `beta = 0.1` 运行 PPO。KL 惩罚能阻止策略利用这个偏置吗？`beta` 降到多少时，利用行为开始显现？

3. 阅读 Ouyang et al.（arXiv:2203.02155）的 Figure 1。分别运行 1、5、20、100 个 PPO 步，测量相对 SFT 模型的偏好率，复现标注员偏好曲线。

4. 论文 4.3 节报告 1.3B InstructGPT 在约 70% 的情况下击败 175B GPT-3。为什么在隐藏的生产提示上这个比例会比在标注员自己的提示上更高？

5. 在同一份偏好数据上，把 PPO 损失替换为 DPO（Phase 10 · 08）。对比最终的策略漂移（相对 SFT 的 KL）和最终奖励。在奖励相当的条件下，哪种方法漂移得更远？

## 关键术语

| 术语 | 人们常说的 | 实际含义 |
|------|-----------------|------------------------|
| SFT | "指令微调" | 阶段一：在提示-回复对上做交叉熵微调 |
| 奖励模型 | "RM" | 对 (提示, 回复) 打分的标量回归器，用 Bradley-Terry 在成对标签上训练 |
| Bradley-Terry | "成对偏好损失" | -log sigmoid(r_w - r_l)；把成对排序归约为二分类 |
| KL 惩罚 | "正则项" | `beta * KL(pi \|\| pi_SFT)` —— 把 RL 策略锚定在 SFT 策略附近 |
| PPO-ptx | "混入预训练的 PPO" | 在 PPO 目标中加入一定比例的预训练对数似然，以抵消对齐税 |
| 对齐税 | "RLHF 退步" | RLHF 之后在其未针对的标准基准上的性能下降 |
| 标注员偏好 | "真值（ground truth）" | 人类排序的一个样本；RM 是对它的统计代理，而不是对"人类价值观"的代理 |

## 延伸阅读

- [Ouyang et al. — Training language models to follow instructions with human feedback (arXiv:2203.02155)](https://arxiv.org/abs/2203.02155) —— InstructGPT 论文，此后所有 RLHF 流水线的基石
- [Stiennon et al. — Learning to summarize from human feedback (arXiv:2009.01325)](https://arxiv.org/abs/2009.01325) —— 面向摘要任务的 RLHF 先驱工作
- [Christiano et al. — Deep reinforcement learning from human preferences (arXiv:1706.03741)](https://arxiv.org/abs/1706.03741) —— 基于偏好的 RL 的原始提出
- [Bai et al. — Training a Helpful and Harmless Assistant with RLHF (arXiv:2204.05862)](https://arxiv.org/abs/2204.05862) —— Anthropic 在 InstructGPT 流水线上的 HH 扩展
