# 投机解码与 EAGLE

> 前沿 LLM 每生成一个 token，都要对数十亿参数做一次完整的前向传播。这次前向传播其实严重「供过于求」：大多数时候，一个小得多的模型就能正确猜出接下来的 3-5 个 token，大模型只需要*验证*这个猜测。猜对了，你就用一次前向传播的代价拿到了 5 个 token。投机解码（speculative decoding，Leviathan et al. 2023）把这个想法做到了数学上严格等价，而 EAGLE-3（2025）把接受率推到了每次验证约 4.5 个 token——在输出分布完全一致的前提下实现 4-5 倍加速。

**Type:** Build
**Languages:** Python (with numpy)
**Prerequisites:** Phase 10 Lesson 12 (Inference Optimization), Phase 10 Lesson 04 (Pre-training Mini-GPT)
**Time:** ~75 minutes

## 问题背景

70B 量级模型在 H100 上的解码吞吐通常是每秒 40-80 个 token。每个 token 都需要一次完整的前向传播，把所有模型权重从 HBM 读一遍。你不能把模型变小，否则输出就变了；你也不能把批大小加到超出显存。看起来无路可走——除非你能让模型每次前向传播输出不止一个 token。

自回归生成看上去是天然串行的：`x_{t+1} = sample(p(· | x_{1:t}))`。但这里存在并行的机会。如果有一个廉价的预测器说「接下来的 4 个 token 大概率是 [a, b, c, d]」，你就可以用**大模型的一次前向传播**同时验证全部 5 个位置，然后接受其中最长的匹配前缀。

Leviathan、Kalai、Matias（2023，"Fast Inference from Transformers via Speculative Decoding"）通过一条巧妙的接受/拒绝规则把这件事做到了严格等价——保持目标模型的采样分布不变。同样的输出分布，2-4 倍提速。

## 核心概念

### 双模型架构

- **目标模型（target model）** `M_p`：又大又慢、质量高，是你真正想从中采样的模型。分布记为 `p(x)`。
- **草稿模型（draft model）** `M_q`：又小又快、质量较低的模型。分布记为 `q(x)`。比目标模型小 5-30 倍。

每一步：

1. 草稿模型自回归地提议 `K` 个 token：`x_1, x_2, ..., x_K ~ q`。
2. 目标模型对全部 `K+1` 个位置并行地跑一次前向传播，为每个提议的 token 算出 `p(x_k)`。
3. 按下文的修正版拒绝采样规则，从左到右逐个 token 决定接受/拒绝。接受最长的匹配前缀。
4. 一旦某个 token 被拒绝，就从校正后的分布中采样一个替换 token 并停止。否则从 `p(· | x_1...x_K)` 中额外采样一个奖励 token。

如果草稿与目标完全一致，每次目标前向传播能拿到 K+1 个 token。如果草稿在第 1 个位置就猜错，你只拿到 1 个 token。

### 精确性规则

投机解码**可证明在分布上等价于直接从 p 采样**。拒绝规则如下：

```
For each drafted token x_t:
    r ~ Uniform(0, 1)
    if r < p(x_t) / q(x_t):
        accept x_t
    else:
        sample replacement from residual: (p - q)+ / ||(p - q)+||_1
        stop
```

其中 `(p - q)+` 表示逐点差值的正部。当草稿与目标一致（`p ≈ q`）时，接受率接近 1。当二者不一致时，残差分布的构造方式保证整体采样结果仍然严格服从 `p`。

**贪心情形。** 对 temperature=0 的采样，只需检查 `argmax(p) == x_t`。相等就接受；否则输出 `argmax(p)` 并停止。

### 期望加速比

设草稿模型在 token 层面的接受率为 `α`，则每次目标前向传播产出的期望 token 数为：

```
E[tokens] = (1 - α^{K+1}) / (1 - α)        # K = draft length, α in [0, 1]
```

在 `α = 0.8, K = 4` 时：`(1 - 0.8^5)/(1 - 0.8) = 3.36` 个 token / 前向传播。一轮的总开销大约是 `cost_q * K + cost_p`（K 步草稿外加一次目标验证）。如果 `cost_p >> cost_q * K`，吞吐上的加速比就是 `3.36× / 1 = 3.36×`。

真正起作用的参数只有 `α`，它完全取决于草稿与目标的对齐程度。一个好的草稿模型就是一切。

### 训练草稿模型：蒸馏

随机初始化的小模型当草稿效果很差。标准做法是从目标模型蒸馏：

1. 选一个小架构（70B 目标配约 1B，7B 目标配约 500M）。
2. 用目标模型跑一遍大规模文本语料，存下它的下一 token 分布。
3. 用 KL 散度对齐目标的分布来训练草稿模型（而不是对齐真实标签 token）。

结果：`α` 在代码任务上通常为 0.6-0.8，在自然语言对话上为 0.7-0.85。生产环境加速 2-3 倍。

### EAGLE：树状草稿 + 特征复用

Li、Wei、Zhang、Zhang（2024，"EAGLE: Speculative Sampling Requires Rethinking Feature Uncertainty"）发现标准投机解码存在两处低效：

1. 草稿模型要做 K 步串行计算，每步都要跑完整模型。但草稿其实可以复用目标模型在最近一次验证中算出的特征（隐藏状态）——目标模型已经计算出了丰富的表示，草稿却在从零重新推导。
2. 草稿输出的是一条线性链。如果草稿能输出一棵候选*树*（每个节点多个猜测），目标模型的单次前向传播就能借助树注意力掩码并行验证多条候选路径，并选出被接受的最长分支。

EAGLE-1 的改动：
- 草稿的输入 = 目标模型在位置 t 的最终隐藏状态，而非原始 token。
- 草稿的架构 = 1 层 Transformer 解码器层（而非一个独立小模型）。
- 输出 = 每层深度 K = 4-8 个候选的树，深度 4-6。

EAGLE-2（2024）加入了动态树拓扑：草稿不确定的地方树长得更宽，自信的地方保持窄。在不增加验证开销的情况下提高了 `α_effective`。

EAGLE-3（Li et al. 2025，"EAGLE-3: Scaling up Inference Acceleration of Large Language Models via Training-Time Test"）去掉了对固定顶层特征的依赖，并用一种新的「测试时模拟」损失来训练草稿——草稿的训练数据是与目标模型测试时分布一致的输出，而不是教师强制（teacher forcing）的训练分布。接受率从 0.75（EAGLE-2）提升到 0.82（EAGLE-3），每次验证的平均 token 数从 3.0 提升到 4.5。

### 树注意力验证

当草稿输出一棵树时，目标模型用**树注意力掩码（tree attention mask）**在单次前向传播中完成验证——这是一种编码树拓扑而非纯线性结构的因果掩码。每个 token 只关注它在树中的祖先节点。验证仍然是一次前向、一次矩阵乘法；拓扑掩码只多花几条 KV 条目的代价。

```
        root
       /    \
      a      b
     / \    / \
    c  d   e   f
```

如果 `a, b` 是相互竞争的第一个 token 候选，`c, d, e, f` 是第二个 token 候选，那么全部六个位置在一次前向传播中完成验证。输出是所有被接受路径中的最长前缀。

### 什么时候赚，什么时候不赚

**赚的场景：**
- 文本可预测的对话/补全任务（代码、常见英文、结构化输出）。`α` 高。
- 解码期间 GPU 算力有闲置的场景（访存受限阶段）。树状草稿正好把空闲 FLOPs 用起来。

**亏或不赚的场景：**
- 高度随机的输出（高温度下的创意写作）。`α` 跌向 `1/|vocab|`。
- 并发非常高的批量服务——批处理已经把 FLOPs 填满了，没什么空间留给树验证。
- 目标模型本身很小，草稿没法小太多的情况。

生产团队的典型报告：对话任务实际加速 2-3 倍，代码生成 3-5 倍，创意写作几乎为零。

```figure
speculative-decoding
```

## 从零实现

`code/main.py`:

- 一个参考实现 `speculative_decode(target, draft, prompt, K, temperature)`，实现精确拒绝规则，并验证它保持目标模型的分布不变（与纯目标采样相比，经验 KL < 0.01）。
- 一个 EAGLE 风格的树状草稿器，用 top-p 分支构建深度为 K 的树。
- 一个树注意力掩码构建器，为验证器生成正确的因果模式。
- 一个接受率测试框架，在小型 LM 上跑通整套流程（从 GPT-2-medium 目标蒸馏出一个 GPT-2-small 草稿）。

```python
def speculative_step(p_target, q_draft, K, temperature=1.0):
    """One round of speculative decoding. Returns list of accepted tokens."""
    # 1. Draft K tokens
    draft_tokens = []
    q_probs = []
    state = draft_state_init()
    for _ in range(K):
        probs = softmax(q_draft(state) / temperature)
        t = np.random.choice(len(probs), p=probs)
        draft_tokens.append(t)
        q_probs.append(probs[t])
        state = draft_step(state, t)

    # 2. Target computes p at every drafted position + 1 extra
    p_probs_all = target_forward_batched(p_target, draft_tokens, temperature)

    # 3. Accept/reject left-to-right
    accepted = []
    for k, tok in enumerate(draft_tokens):
        r = np.random.uniform()
        if r < p_probs_all[k][tok] / q_probs[k]:
            accepted.append(tok)
        else:
            residual = np.maximum(p_probs_all[k] - q_probs[k], 0)
            residual /= residual.sum()
            accepted.append(np.random.choice(len(residual), p=residual))
            return accepted
    # 4. All K accepted → sample bonus token from target
    accepted.append(np.random.choice(len(p_probs_all[-1]), p=p_probs_all[-1]))
    return accepted
```

## 生产实践

- **vLLM** 和 **SGLang** 都内置一等公民级的投机解码支持。参数：`--speculative_model`、`--num_speculative_tokens`。通过 `--spec_decoding_algorithm eagle` 参数支持 EAGLE-2/3。
- **NVIDIA TensorRT-LLM** 原生支持 Medusa 和 EAGLE 树。
- **参考草稿模型**：`Qwen/Qwen3-0.6B-spec`（为 Qwen3-32B 做草稿）、`meta-llama/Llama-3.2-1B-Instruct-spec`（为 70B 做草稿）。
- **Medusa 头**（Cai et al. 2024，"Medusa: Simple LLM Inference Acceleration Framework with Multiple Decoding Heads"）：不用独立的草稿模型，而是直接给目标模型加 K 个并行预测头。部署更简单，接受率略低于 EAGLE。

## 交付产物

本课产出 `outputs/skill-speculative-tuning.md`——一个对目标模型工作负载做画像并据此选择以下配置的技能：草稿模型、K（草稿长度）、树宽度、温度，以及何时回退到普通解码。

## 练习

1. 实现精确拒绝规则并做经验验证。分别用 `speculative_decode` 和纯目标采样各跑 1 万个样本；计算两个输出分布之间的 TV 距离。应小于 0.01。

2. 计算加速比公式。给定固定的 `α` 和 `K`，画出每次目标前向传播的期望 token 数曲线。求出 α ∈ {0.5, 0.7, 0.9} 时的最优 K。

3. 训练一个微型草稿模型。以 124M 的 GPT-2 为目标，用 KL 损失在 1 亿 token 上蒸馏出一个 30M 的 GPT-2 草稿。在保留集文本上测量 `α`。预期：0.6-0.7。

4. 实现 EAGLE 风格的树状草稿。把链式输出改成让草稿在每个深度输出 top-3 分支。构建树注意力掩码。验证目标模型能接受最长的正确分支。

5. 测量失败模式。在 temperature=1.5（高随机性）下跑投机解码。展示 α 崩塌，且由于草稿开销，算法比普通解码更慢。

## 关键术语

| 术语 | 大家怎么说 | 实际含义 |
|------|-----------------|------------------------|
| 目标模型 | 「那个大模型」 | 又慢、质量又高、你真正想从中采样的模型（p 分布） |
| 草稿模型 | 「投机器」 | 又小又快的预测器（q 分布）；小 5-30 倍 |
| K / 草稿长度 | 「前瞻量」 | 每次验证投机猜测的 token 数 |
| α / 接受率 | 「命中率」 | 草稿提议在单个 token 层面被接受的概率 |
| 精确拒绝规则 | 「接受测试」 | r < p/q 的比较，保持目标模型的分布不变 |
| 残差分布 | 「校正后的 p-q」 | (p - q)+ / \|\|(p - q)+\|\|_1，拒绝时用来采样的分布 |
| 树状草稿 | 「分支投机」 | 草稿输出一棵候选树，用树结构注意力掩码在一次前向传播中完成验证 |
| 树注意力掩码 | 「拓扑掩码」 | 编码树拓扑的因果掩码，每个节点只关注其祖先 |
| Medusa 头 | 「并行头」 | 直接加在目标模型上的 K 个额外预测头；不需要独立草稿模型 |
| EAGLE 特征复用 | 「隐藏状态草稿」 | 草稿的输入是目标模型的最后一层隐藏状态而非原始 token，从而缩小草稿规模 |
| 测试时模拟损失 | 「EAGLE-3 训练法」 | 用与目标模型测试时分布一致的输出训练草稿，而非教师强制 |

## 延伸阅读

- [Leviathan, Kalai, Matias, 2023 — "Fast Inference from Transformers via Speculative Decoding"](https://arxiv.org/abs/2211.17192) — 精确拒绝规则与理论加速比分析
- [Chen, Borgeaud, Irving et al., 2023 — "Accelerating Large Language Model Decoding with Speculative Sampling"](https://arxiv.org/abs/2302.01318) — DeepMind 同期的投机采样论文
- [Cai, Li, Geng, Wang, Wang, Zhu, Dao, 2024 — "Medusa: Simple LLM Inference Acceleration Framework with Multiple Decoding Heads"](https://arxiv.org/abs/2401.10774) — 用并行头替代草稿模型的方案
- [Li, Wei, Zhang, Zhang, 2024 — "EAGLE: Speculative Sampling Requires Rethinking Feature Uncertainty"](https://arxiv.org/abs/2401.15077) — 特征复用与树状草稿
- [Li et al., 2024 — "EAGLE-2: Faster Inference of Language Models with Dynamic Draft Trees"](https://arxiv.org/abs/2406.16858) — 动态树拓扑
- [Li et al., 2025 — "EAGLE-3: Scaling up Inference Acceleration of Large Language Models via Training-Time Test"](https://arxiv.org/abs/2503.01840) — 训练时与测试时分布匹配
- [Fu, Haotian, Peng et al., 2024 — "Break the Sequential Dependency of LLM Inference Using Lookahead Decoding"](https://arxiv.org/abs/2402.02057) — Jacobi/前瞻解码，一种无需投机器的替代方案
