# 推测解码 —— 起草、验证、循环往复

> 自回归解码是串行的：每个 token 都要等前一个生成完。推测解码（speculative decoding）打破了这条链：用一个廉价模型先起草 N 个 token，再让昂贵的大模型在一次前向传播中验证全部 N 个。如果草稿全对，你只花了一次大模型前向的代价就拿到了 N 个 token。

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 7 · 07 (GPT Causal LM), Phase 7 · 12 (KV Cache & Flash Attention)
**Time:** ~60 minutes

## 问题背景

一个 70B 的 LLM 在 H100 上采样一个 token 大约需要 30 ms，而一个 3B 的草稿模型只需约 3 ms。如果让 3B 模型先往前起草 5 个 token，再让 70B 模型*跑一次*来验证这 5 个，总耗时是 `5×3 + 30 = 45 ms`，最多可以接受 5 个 token —— 而逐个直接生成需要 `5×30 = 150 ms`。这就是推测解码的全部卖点：用少量额外的 GPU 显存（草稿模型）换取 2–4 倍的解码延迟降低。

这个技巧必须保持分布不变。推测采样（speculative sampling）由 Leviathan 等人（2023）提出，Chen 等人同期也独立提出，它保证输出序列与大模型独立生成的结果**同分布**。没有任何质量损失，只是更快。

2026 年的推理领域由四类「草稿-验证器」组合主导：

1. **原版推测解码（Leviathan 2023）。** 独立的草稿模型（如 Llama 3 1B）+ 验证器（如 Llama 3 70B）。
2. **Medusa（Cai 2024）。** 在验证器上加多个解码头，并行预测位置 `t+1..t+k`。不需要单独的草稿模型。
3. **EAGLE 系列（Li 2024, 2025）。** 轻量草稿模型，复用验证器的隐藏状态；接受率比原版更高；典型加速 3–4 倍。
4. **前瞻解码（Lookahead decoding，Fu 2024）。** 基于 Jacobi 迭代；完全不需要草稿模型，属于自推测（self-speculation）。小众但零依赖。

2026 年的每个生产级推理栈都默认内置推测解码。vLLM、TensorRT-LLM、SGLang 和 llama.cpp 至少都支持原版 + EAGLE-2。

## 核心概念

### 核心算法

给定验证器 `M_q` 和更廉价的草稿模型 `M_p`：

1. 设 `x_1..x_k` 为已解码的前缀。
2. **起草**：用 `M_p` 自回归地提出 `d_{k+1}, d_{k+2}, ..., d_{k+N}`，对应的草稿概率为 `p_1..p_N`。
3. **并行验证**：将 `x_1..x_k, d_{k+1}, ..., d_{k+N}` 一次性输入 `M_q`，得到位置 `k+1..k+N+1` 上的验证器概率 `q_1..q_{N+1}`。
4. **从左到右逐个接受/拒绝草稿 token**：对每个 `i`，以概率 `min(1, q_i(d_i) / p_i(d_i))` 接受。
5. 在位置 `j` 首次拒绝时：从归一化后的「残差」分布 `(q_j - p_j)_+` 中采样 `t_j`。`j` 之后的所有草稿全部丢弃。
6. 若全部 `N` 个都被接受：从 `q_{N+1}` 中额外采样一个 token `t_{N+1}`（免费的奖励 token）。

残差分布这一招正是数学上的关键洞察：它保证输出的分布与 `M_q` 从头采样完全一致。

### 加速比由什么决定

设 `α` = 每个草稿 token 的期望接受率，`c` = 草稿与验证器的成本比。每一步：

- 朴素生成每个 token 需要 1 次大模型调用。
- 推测解码每 `(1 - α^{N+1}) / (1 - α) ≈ 1/(1-α)` 个 token 才需要 1 次大模型调用（当 `α` 较高时）。

经验法则：在 `α = 0.75`、`N = 5` 时，大模型调用减少 3 倍。草稿成本只有大模型的五分之一。总墙钟时间下降约 2.5 倍。

**α 取决于：**

- 草稿模型对验证器的逼近程度。同一模型家族 / 同样的训练数据能显著提升 α。
- 解码策略。贪心草稿对贪心验证器：α 很高。温度采样：更难匹配，接受率下降。
- 任务类型。代码和结构化输出接受率更高（可预测性强）；自由发挥的创意写作接受率更低。

### Medusa —— 不用草稿模型的起草

Medusa 用验证器上的额外输出头取代草稿模型。在位置 `t`：

```
shared trunk → hidden h_t
    ├── head_0: predict token at t+1  (standard LM head)
    ├── head_1: predict token at t+2
    ├── head_2: predict token at t+3
    ├── head_3: predict token at t+4
```

每个头输出自己的 logits。推理时从每个头采样得到候选序列，然后用一次前向传播完成验证，借助树注意力（tree attention）机制同时考察所有候选续写。

优点：不需要第二个模型。缺点：增加可训练参数；需要一个监督微调阶段（约 1B token）；接受率比配了好草稿模型的原版推测解码略低。

### EAGLE —— 复用隐藏状态得到更好的草稿

EAGLE-1/2/3（Li 等，2024–2025）把草稿模型做成一个微型 Transformer（通常只有 1 层），其输入是验证器最后一层的隐藏状态。由于草稿模型看到的是验证器的特征表示，它的预测与验证器的输出分布高度相关。接受率从原版的约 0.6 提升到 0.85 以上。

EAGLE-3（2025）增加了对候选续写的树搜索。vLLM 和 SGLang 已将 EAGLE-2/3 作为 Llama 3/4 和 Qwen 3 的默认推测解码路径。

### KV 缓存的腾挪

验证阶段把 `N` 个草稿 token 一次性送进验证器前向传播，这会让验证器的 KV 缓存增加 `N` 个条目。如果部分草稿被拒绝，就必须把缓存回滚到已接受前缀的长度。

生产级实现（vLLM 的 `--speculative-model`、TensorRT-LLM 的 LookaheadDecoder）用临时 KV 缓冲区处理这件事：先写入，接受后再提交。概念上不难，但实现起来很琐碎。

## 从零实现

见 `code/main.py`。我们实现核心的推测采样算法（拒绝步骤 + 残差分布），包括：

- 一个「大模型」：对手工设定的分布做确定性 softmax（这样可以解析地验证接受率的数学）。
- 一个「草稿模型」：在大模型基础上加扰动。
- 一个接受/拒绝循环：其产生的边缘分布与直接采样完全相同。

### 第 1 步：拒绝步骤

```python
def accept_or_reject(q_prob, p_prob, draft_token, u):
    ratio = q_prob / p_prob if p_prob > 0 else float("inf")
    return u < min(1.0, ratio)
```

`u` 是一个均匀随机数。`q_prob` 是验证器对该草稿 token 给出的概率，`p_prob` 是草稿模型的概率。Leviathan 定理指出：这个 Bernoulli 决策，加上拒绝时从残差分布采样，能精确保持验证器的分布。

### 第 2 步：残差分布

```python
def residual_dist(q, p):
    raw = [max(0.0, qi - pi) for qi, pi in zip(q, p)]
    s = sum(raw)
    return [r / s for r in raw]
```

将 `q` 逐元素减去 `p`，把负值截断为零，再重新归一化。任何一次拒绝都从这个分布中采样。

### 第 3 步：一次完整的推测步骤

```python
def spec_step(prefix, q_model, p_model, N, rng):
    drafts = []
    p_probs = []
    ctx = list(prefix)
    for _ in range(N):
        p_dist = p_model(ctx)
        d = sample(p_dist, rng)
        drafts.append(d)
        p_probs.append(p_dist[d])
        ctx.append(d)

    q_dists = [q_model(prefix + drafts[:i]) for i in range(N + 1)]

    for i, d in enumerate(drafts):
        u = rng.random()
        q_prob = q_dists[i][d]
        p_prob = p_probs[i]
        if u < min(1.0, q_prob / p_prob if p_prob > 0 else float("inf")):
            prefix = prefix + [d]
        else:
            res = residual_dist(q_dists[i], p_model(prefix))
            prefix = prefix + [sample(res, rng)]
            return prefix
    prefix = prefix + [sample(q_dists[N], rng)]
    return prefix
```

接受 5 个 → 奖励 1 个 → 一次验证器前向产出 6 个 token。

### 第 4 步：测量接受率

在不同草稿质量水平下运行 10,000 次推测步骤。绘制接受率随草稿与验证器分布之间 KL 散度变化的曲线。你应该看到一条干净的单调关系。

### 第 5 步：验证分布等价性

经验验证：推测循环产出的 token 直方图应当与直接从验证器采样得到的直方图一致。这就是 Leviathan 定理的实践体现。卡方检验确认两者在采样误差范围内一致。

## 生产实践

生产环境：

```bash
# vLLM with EAGLE
vllm serve meta-llama/Llama-3.1-70B-Instruct \
    --speculative-model /models/llama-3.1-eagle-70b \
    --speculative-draft-tensor-parallel-size 1 \
    --num-speculative-tokens 5

# vLLM with vanilla draft model
vllm serve meta-llama/Llama-3.1-70B-Instruct \
    --speculative-model meta-llama/Llama-3.2-1B-Instruct \
    --num-speculative-tokens 5
```

截至 2026 年中，TensorRT-LLM 拥有最快的 Medusa 路径。`faster-whisper` 用小草稿模型为 Whisper-large 封装了推测解码。

**如何选择草稿策略：**

| 策略 | 适用场景 | 加速比 |
|----------|--------------|---------|
| 原版草稿模型（1B/3B Llama 家族） | 快速原型，无需训练 | 1.8–2.3× |
| Medusa 多头 | 你能对验证器做微调 | 2–3× |
| EAGLE-2 / 3 | 生产环境，追求极致速度 | 3–4× |
| Lookahead | 无草稿、无训练、无额外参数 | 1.3–1.6× |

**什么时候不该用推测解码：**

- 单序列只生成 1–5 个 token。开销占主导。
- 极度发散 / 高温度采样（α 会下降）。
- 显存受限的部署（草稿模型会额外占用 VRAM）。

## 交付产物

见 `outputs/skill-spec-decode-picker.md`。该 skill 为新的推理工作负载选择推测解码策略（原版 / Medusa / EAGLE / lookahead）及调优参数（N、草稿温度）。

## 练习

1. **简单。** 运行 `code/main.py`。在 50,000 个 token 上确认推测解码的 token 分布与验证器直接采样的分布一致，卡方检验 p > 0.05。
2. **中等。** 在 `α = 0.5, 0.7, 0.85` 下，绘制加速比（每次大模型前向产出的 token 数）随 `N` 变化的曲线。找出每个 α 对应的最优 `N`。（提示：每次验证调用的期望 token 数 = `(1 - α^{N+1}) / (1 - α)`。）
3. **困难。** 实现一个迷你 Medusa：取第 14 课的 capstone GPT，加 3 个额外的 LM 头分别预测位置 t+2、t+3、t+4。在 tinyshakespeare 上用多头联合损失训练。与截断同一模型得到的原版草稿对比接受率。
4. **困难。** 实现回滚：从一个 10 token 前缀的 KV 缓存开始，输入 5 个草稿 token，模拟在位置 3 发生拒绝。验证下一轮迭代时缓存读取的内容恰好等于「前缀 + 前 2 个已接受的草稿」。

## 关键术语

| 术语 | 大家怎么叫 | 实际含义 |
|------|-----------------|-----------------------|
| 草稿模型（draft model） | 「便宜的那个」 | 提出候选 token 的小模型；通常比验证器便宜 10–50 倍。 |
| 验证器（verifier） | 「大的那个」 | 我们要保持其分布的目标模型；每个推测步骤只运行一次。 |
| 接受率（α） | 「草稿猜对的频率」 | 验证器接受草稿 token 的逐 token 概率。典型值 0.7–0.9。 |
| 残差分布（residual distribution） | 「拒绝后的兜底」 | 归一化后的 `(q - p)_+`；拒绝时从中采样可保持验证器的分布。 |
| 奖励 token（bonus token） | 「白送的那个」 | 当全部 N 个草稿都被接受时，从验证器的下一步分布中再采样一个。 |
| Medusa | 「无草稿模型的推测解码」 | 验证器上的多个 LM 头并行预测位置 t+1..t+k。 |
| EAGLE | 「隐藏状态草稿」 | 以验证器最后一层隐藏状态为条件的微型 Transformer 草稿模型。 |
| 前瞻解码（lookahead decoding） | 「Jacobi 迭代」 | 用不动点迭代实现自推测；不需要草稿模型。 |
| 树注意力（tree attention） | 「一次验证多个候选」 | 分支式验证，同时考察多条草稿续写。 |
| KV 回滚（KV rollback） | 「撤销被拒的草稿」 | 临时 KV 缓冲区；接受则提交，拒绝则丢弃。 |

## 延伸阅读

- [Leviathan, Kalman, Matias (2023). Fast Inference from Transformers via Speculative Decoding](https://arxiv.org/abs/2211.17192) —— 核心算法与分布等价性定理。
- [Chen et al. (2023). Accelerating Large Language Model Decoding with Speculative Sampling](https://arxiv.org/abs/2302.01318) —— 同期独立提出；干净的 Bernoulli 拒绝证明。
- [Cai et al. (2024). Medusa: Simple LLM Inference Acceleration Framework with Multiple Decoding Heads](https://arxiv.org/abs/2401.10774) —— Medusa 论文；树注意力验证。
- [Li et al. (2024). EAGLE: Speculative Sampling Requires Rethinking Feature Uncertainty](https://arxiv.org/abs/2401.15077) —— EAGLE-1；以隐藏状态为条件的草稿。
- [Li et al. (2024). EAGLE-2: Faster Inference of Language Models with Dynamic Draft Trees](https://arxiv.org/abs/2406.16858) —— EAGLE-2；动态树深度。
- [Li et al. (2025). EAGLE-3: Scaling up Inference Acceleration of Large Language Models via Training-Time Test](https://arxiv.org/abs/2503.01840) —— EAGLE-3。
- [Fu et al. (2024). Break the Sequential Dependency of LLM Inference Using Lookahead Decoding](https://arxiv.org/abs/2402.02057) —— 前瞻解码，无草稿方案。
- [vLLM docs — Speculative Decoding](https://docs.vllm.ai/en/latest/features/spec_decode.html) —— 权威生产参考，四种策略全部就绪。
- [SafeAILab / EAGLE reference implementation](https://github.com/SafeAILab/EAGLE) —— EAGLE-1/2/3 的参考代码。
