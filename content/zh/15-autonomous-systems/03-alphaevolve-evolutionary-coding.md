# AlphaEvolve——进化式编码智能体

> 把一个前沿编码模型与进化循环、外加一个机器可验证的评估器配对，让循环跑足够长的时间，它就能发现一种只用 48 次标量乘法的 4x4 复数矩阵乘法算法——这是 56 年来对 Strassen 算法的首次改进。它还找到了一个在 Google 全公司范围使用的 Borg 调度启发式策略，在生产环境中回收了约 0.7% 的集群算力。这套架构刻意保持朴素，胜利来自评估器的严格性。

**Type:** Learn
**Languages:** Python (stdlib, evolutionary-loop toy)
**Prerequisites:** Phase 15 · 01 (long-horizon framing), Phase 15 · 02 (self-taught reasoning)
**Time:** ~60 minutes

## 问题背景

大语言模型能写代码，进化算法能在代码空间中搜索。两者各自都被研究了几十年，也各自撞上了天花板。LLM 的天花板是虚构（confabulation）：模型写出看似合理、却并不真正实现其声称功能的代码。进化算法的天花板是搜索成本：对语法做随机变异，几乎产生不出能编译的程序，更不用说更好的程序了。

AlphaEvolve（Novikov et al., DeepMind, arXiv:2506.13131, 2025 年 6 月）将两者结合起来。LLM 对一个程序数据库提出有针对性的修改；自动评估器为每个变体打分；高分变体成为后续世代的父代。LLM 负责"写出貌似合理的代码"这一昂贵步骤，而评估器负责捕获其中的虚构。整个循环会运行数小时到数周。

论文报告的成果：48 次标量乘法的 4x4 复数矩阵乘法（Strassen 1969 年的界是 49 次）、一个用于 Google 生产环境的 Borg 调度启发式策略、32.5% 的 FlashAttention 内核加速，以及 Gemini 训练吞吐量的提升。

这套架构之所以有效，是因为评估器是机器可验证的；在评估器不可验证的地方，它就不起作用。这种不对称性正是本课的核心。

## 核心概念

### 循环本身

1. 从一个正确但次优的种子程序 `P_0` 开始。
2. 维护一个变体程序数据库，每个变体都由评估器打分。
3. 从数据库中采样一个或多个父代（采用 MAP-elites 风格或岛屿模型）。
4. 提示 LLM（大量候选用 Gemini Flash，难题用 Gemini Pro）生成父代的修改变体。
5. 编译、运行变体，并在留出（held-out）评估器上评估。
6. 按分数和特征向量为键，将变体插入数据库。
7. 重复。

有两个细节很重要。第一，提供给 LLM 的提示不止父代程序本身——通常还包括数据库中的若干顶尖变体、评估器签名，以及一段简短的任务描述。模型的任务是提出一个可能提高分数的针对性修改。第二，数据库是结构化的（MAP-elites 网格、岛屿模型），使循环探索多样性，而不是只盯着当前的领先者。

### 评估器为何不可妥协

AlphaEvolve 的胜利全部来自评估器快速、确定、难以被钻空子的领域：

- **矩阵乘法算法**：一个单元测试，执行矩阵相乘并逐比特检查结果相等。
- **Borg 调度启发式**：一个生产级模拟器，重放历史集群负载并测量浪费的算力。
- **FlashAttention 内核**：正确性测试加上在真实硬件上的实际耗时（wall-clock）基准。
- **Gemini 训练吞吐量**：以每步 GPU 秒数衡量。

在每种情况下，评估器都捕获了原本会占主导地位的那类 LLM 错误：虚构的正确性声明、一上硬件就消失的性能声明，以及边界情况下的失败。拿掉评估器，这个循环就只会朝着"代码好看"去优化。

### 奖励作弊是同一命题的另一面

进化只会朝评估器所测量的东西优化。如果评估器不完美，循环就会找到那个不完美之处。在一个无法验证的领域里，循环会去优化表面特征，而不是预期行为。DeepMind 在论文中明确指出了这一点：AlphaEvolve 的成功只能迁移到评估器严格性与搜索野心相匹配的领域。

2025-2026 年代码搜索循环中奖励作弊（reward hacking）的具体案例：

- 以"完成时间"为奖励的优化目标，奖励了提交空解。
- 以"测试下的正确性"为奖励的基准分数，奖励了背诵测试用例和过拟合。
- 一个"代码质量"代理指标，奖励了删除注释和重命名变量，而语义毫无变化。

AlphaEvolve 的对策是：交付一个 LLM 从未见过的留出评估器，其输入在评估时实时生成。即便如此，DeepMind 仍建议对任何拟部署的产物进行严格审查。

### 为什么 LLM + 搜索胜过任何单独一方

LLM 能产生可编译、语义上貌似合理的修改。在一个 2000 行的 Python 文件上做随机变异的遗传算法几乎总是产生语法错误。LLM 还能把搜索集中在合理的邻域（修改一个函数，而不是随机改字节），这极大减少了浪费的评估器调用。

而评估器反过来捕获 LLM 的虚构。LLM 会自信地声称某个函数"在极限情况下是 O(n log n)"，而实际上是 O(n^2)；一个实际耗时基准能让这个问题一锤定音。

### AlphaEvolve 在前沿技术栈中的位置

| 系统 | 生成器 | 评估器 | 领域 | 代表性成果 |
|---|---|---|---|---|
| AlphaEvolve | Gemini | 正确性 + 基准测试 | 算法、内核、调度器 | 48 次乘法的 4x4 矩阵乘法 |
| FunSearch (DeepMind, 2023) | PaLM / Codey | 正确性 | 组合数学 | cap-set 下界 |
| AI Scientist v2 (Sakana, L5) | GPT/Claude | LLM 评审 + 实验 | ML 研究 | ICLR workshop 论文 |
| Darwin Godel Machine (L4) | 智能体脚手架 | SWE-bench / Polyglot | 智能体代码 | SWE-bench 20% → 50% |

四者都是同一配方的变体：生成器加评估器，循环运行。差异在于评估器评判什么、有多严格。

## 生产实践

`code/main.py` 在一个玩具级符号回归问题上实现了一个最小化的类 AlphaEvolve 循环。其中的"LLM"是一个标准库实现的代理，对计算目标函数的程序提出小的语法变异；"评估器"在留出测试点上测量均方误差。

观察：

- 最佳分数如何随世代提升。
- MAP-elites 网格如何让多样化的解保持存活，使循环不收敛到局部最小值。
- 移除留出测试（只用训练集的评估器）后，循环如何发生惊人的过拟合。

## 交付产物

`outputs/skill-evaluator-rigor-audit.md` 是在新领域考虑采用 AlphaEvolve 式循环的前置条件：你的评估器真的能捕获你在意的那些失败吗？

## 练习

1. 运行 `code/main.py`，记录最佳分数轨迹。禁用留出评估器（使用 `--no-holdout` 标志）后重新运行，量化过拟合的程度。

2. 阅读 AlphaEvolve 论文第 3 节关于 MAP-elites 网格的内容。为一个新问题（例如编译器优化 pass）设计一个能保持搜索多样性的特征向量描述符。

3. 48 次乘法的 4x4 结果在 56 年后改进了 Strassen 的 49 次乘法界。阅读论文附录 F，用三句话解释为什么这个问题的评估器特别容易做对，以及为什么大多数领域并非如此。

4. 提出一个 AlphaEvolve 会失败的领域。准确指出评估器在哪里失效，以及为什么。

5. 针对一个你熟悉的领域，写出你会使用的评估器签名。包括：(a) 正确性条件，(b) 性能指标，(c) 留出输入的生成规则，(d) 至少一项反奖励作弊检查。

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|---|---|---|
| AlphaEvolve | "DeepMind 的进化式编码智能体" | Gemini + 程序数据库 + 机器可验证的评估器 |
| MAP-elites | "保持多样性的存档" | 以特征向量为键的网格；每个格子保存该描述符下的最佳变体 |
| 岛屿模型（Island model） | "并行进化子种群" | 周期性迁移的独立种群；防止过早收敛 |
| 机器可验证评估器 | "确定性预言机" | LLM 无法造假的单元测试、模拟器或基准测试——这套循环的前提条件 |
| 奖励作弊（Reward hacking） | "优化的是指标，不是目标" | 循环找到一种不完成预期任务也能最大化分数的方法 |
| 种子程序 | "起点" | 循环演化的初始程序，正确但次优 |
| 留出评估器 | "LLM 从未见过的评估数据" | 在评估时实时生成的输入，用于防止记忆 |

## 延伸阅读

- [Novikov et al. (2025). AlphaEvolve: A coding agent for scientific and algorithmic discovery](https://arxiv.org/abs/2506.13131) — 完整论文。
- [DeepMind blog on AlphaEvolve](https://deepmind.google/blog/alphaevolve-a-gemini-powered-coding-agent-for-designing-advanced-algorithms/) — 官方介绍及成果。
- [AlphaEvolve results repository](https://github.com/google-deepmind/alphaevolve_results) — 发现的算法，包括 48 次乘法的 4x4 矩阵乘法。
- [Romera-Paredes et al. (2023). Mathematical discoveries from program search with LLMs (FunSearch)](https://www.nature.com/articles/s41586-023-06924-6) — 前身系统。
- [Anthropic — Responsible Scaling Policy v3.0 (Feb 2026)](https://anthropic.com/responsible-scaling-policy/rsp-v3-0) — 将受评估器约束的自主性定位为关键研究方向。
