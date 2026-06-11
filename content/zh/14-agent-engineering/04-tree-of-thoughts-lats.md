# 思维树与 LATS：深思熟虑的搜索

> 单条思维链（chain-of-thought）轨迹没有任何回溯的余地。ToT（Yao et al., 2023）把推理变成一棵树，并在每个节点上做自我评估。LATS（Zhou et al., 2024）则用蒙特卡洛树搜索（Monte Carlo Tree Search）把 ToT、ReAct 和 Reflexion 统一起来。Game of 24 的准确率从 4%（CoT）跃升至 74%（ToT）；LATS 在 HumanEval 上达到 92.7% 的 pass@1。

**Type:** Build
**Languages:** Python (stdlib)
**Prerequisites:** Phase 14 · 01 (Agent Loop), Phase 14 · 03 (Reflexion)
**Time:** ~75 minutes

## 学习目标

- 把推理建模为搜索：节点是「思维」，边是「扩展」，价值是「这条路有多大希望」。
- 用 Python 标准库实现一个带自我评估打分的 ToT 风格 BFS 树搜索。
- 扩展为一个玩具版 LATS MCTS 循环，包含选择 / 扩展 / 模拟 / 反向传播四个阶段。
- 判断什么时候值得为搜索付出成倍的 token 开销（Game of 24、代码生成），什么时候单条轨迹就够了（简单问答）。

## 问题背景

思维链是一条线性路径。如果第一步就走错了，后面的每一步都建立在错误前提之上。在 Game of 24（用四个数字和 + − × ÷ 凑出 24）上，GPT-4 的 CoT 准确率只有 4%。模型很早就选错了子表达式，之后再也无法挽回。

推理真正需要的是：提出多个候选、逐一评估、挑出有希望的继续走，遇到死胡同就回溯。这就是搜索。思维树（Tree of Thoughts）和 LATS 是两种经典的形式化方案。

## 核心概念

### 思维树（Yao et al., NeurIPS 2023）

每个节点是一个连贯的中间步骤（「一个思维」）。每个节点可以扩展出 K 个子思维。LLM 通过打分提示词对每个节点做自我评估。搜索过程遍历这棵树——可以用 BFS、DFS 或束搜索（beam search）。

```
                     (root: "find 24 from 4 6 4 1")
                    /               |            \
           ("6 - 4 = 2")    ("4 + 1 = 5")    ("4 * 6 = 24")  <- Score: HIGH
              /   \              |                  |
          ...    ...          ...                finish
```

自我评估是整个方法的承重墙。论文给出了三种变体：`sure / likely / impossible` 三档分类、`1..10` 数值打分，以及候选之间互相投票。这三种方式在 Game of 24 上都大幅超过 CoT（GPT-4 上从 4% 提升到 74%）。

### LATS（Zhou et al., ICML 2024）

LATS 用 MCTS 把 ToT、ReAct 和 Reflexion 统一起来。LLM 同时扮演三个角色：

- **策略（Policy）**：提出候选的下一步动作（ReAct 风格）。
- **价值函数（Value function）**：为部分轨迹打分（ToT 风格的自我评估）。
- **自我反思器（Self-reflector）**：失败时写一段自然语言反思（Reflexion 风格），并用它为后续 rollout 提供种子。

环境反馈（观测结果）会混入价值函数，这样搜索就建立在真实的工具结果之上，而不只是模型的主观判断。论文发表时的结果：GPT-4 在 HumanEval 上 pass@1 达到 92.7%（SOTA），GPT-3.5 在 WebShop 上平均得分 75.9（接近基于梯度的微调）。

### MCTS 最小化版本

每轮迭代四个阶段：

1. **选择（Select）**——用 UCT（树的置信上界，upper confidence bound for trees）从根节点走到一个叶子节点。
2. **扩展（Expand）**——通过策略生成 K 个子节点。
3. **模拟（Simulate）**——从某个子节点出发用策略做 rollout，用价值函数（或环境奖励）给叶子打分。
4. **反向传播（Backpropagate）**——沿路径向上更新访问次数和价值估计。

UCT 公式：`Q(s, a) + c * sqrt(ln N(s) / N(s, a))`。第一项是利用（exploitation），第二项是探索（exploration）。`c` 需要按任务调参。

### 成本的现实

搜索会让 token 用量爆炸。ToT 在 Game of 24 上消耗的 token 是 CoT 的 100–1000 倍。LATS 也差不多。这不是免费的；搜索应当留给：

- 单条轨迹明显不够用的任务（Game of 24、复杂代码）。
- 正确性比耗时更重要的任务。
- 拥有廉价且可靠的价值函数的任务（代码有单元测试，数学有明确目标值）。

如果你的任务只有一个正确答案，而评估器又有噪声，搜索往往会让事情更糟——它会找到一个「得分很高」的错误答案。

### 2026 年的定位

大多数生产环境的智能体并不运行 LATS。它们运行的是带工具落地验证的 ReAct（CRITIC，第 05 课）。搜索出现在一些专门的细分场景：

- 用测试作为价值函数的编码智能体（HumanEval 风格）。
- 探索多条查询路径的深度研究智能体。
- LangGraph 子图内部以规划为主的工作流。

AlphaEvolve（第 11 课）是 2025 年的极端案例：对代码做进化搜索，配合机器可校验的适应度函数，取得了前沿突破（56 年来首次改进 4x4 矩阵乘法）。

## 从零实现

`code/main.py` 实现了：

- 在一个风格化的「选算术运算」任务上的迷你 ToT BFS。
- 同一任务上的玩具版 LATS MCTS 循环（Select / Expand / Simulate / Backpropagate），使用 UCT 选择。
- 一个由符号化得分加自我评估得分组合而成的价值函数。

运行：

```
python3 code/main.py
```

跟踪输出展示了 ToT 用 BFS 在每个节点扩展三个候选，与 LATS 通过 MCTS 收敛到最佳 rollout 的对比。两者的 token 用量都会打印出来。

## 生产实践

LangGraph 以子图模式的形式提供 ToT 风格的探索；LangChain 团队关于 LATS 的博客（2024 年 5 月）是参考教程。LlamaIndex 提供了一个 `TreeOfThoughts` 智能体。对 2026 年大多数生产环境的智能体来说，这个模式藏在一个 `if task_complexity > threshold: use_search()` 的门控后面——参见第 05 课的评估器-优化器（evaluator-optimizer）模式。

## 交付产物

`outputs/skill-search-policy.md` 根据任务形态、预算和评估器可信度，在线性 ReAct、ToT、LATS 和进化搜索之间做出选择。

## 练习

1. 分别用 UCT c=0.1 和 c=2.0 运行玩具版 LATS。跟踪输出中有什么变化？
2. 把价值函数换成噪声更大的打分器（加入随机抖动）。MCTS 还能找到最佳叶子节点吗？它能容忍的最低信噪比是多少？
3. 实现束搜索版 ToT（每层只保留 top-k），并与 BFS 对比。在 token 预算紧张时哪个更好？
4. 阅读 LATS 论文第 5.1 节。复现 HumanEval 的轨迹数量：需要多少次 rollout 才能达到论文报告的 pass@1？
5. 阅读 LATS 论文中关于「LATS 何时帮助较小」的讨论。写一段决策规则，把任务形态映射到搜索策略。

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|----------------|------------------------|
| Tree of Thoughts | 「分叉的 CoT」 | Yao et al.——由思维节点组成的树，配合自我评估 |
| LATS | 「LLM 的 MCTS」 | Zhou et al.——用 MCTS 统一 ToT + ReAct + Reflexion |
| UCT | 「置信上界」 | 在利用（Q）和探索（ln N / n）之间取得平衡的选择公式 |
| 价值函数 | 「这个状态有多好」 | 提示词驱动的 LLM 打分或环境奖励；为反向传播提供输入 |
| 策略 | 「动作提议器」 | ReAct 风格的生成器；产出候选的下一步思维/动作 |
| Rollout | 「模拟轨迹」 | 用策略从某个节点走到叶子，再用价值函数打分 |
| 反向传播 | 「更新祖先节点」 | 把叶子的奖励沿路径向上推送，更新访问次数和 Q 值 |
| 搜索成本 | 「token 爆炸」 | Game of 24 上是 CoT 的 100-1000 倍；采用前先做预算 |

## 延伸阅读

- [Yao et al., Tree of Thoughts (arXiv:2305.10601)](https://arxiv.org/abs/2305.10601) —— 经典原始论文
- [Zhou et al., LATS (arXiv:2310.04406)](https://arxiv.org/abs/2310.04406) —— 带 Reflexion 反馈的 MCTS
- [LangGraph overview](https://docs.langchain.com/oss/python/langgraph/overview) —— 用于搜索的子图模式
- [AlphaEvolve (arXiv:2506.13131)](https://arxiv.org/abs/2506.13131) —— 配合程序化评估器的进化搜索
