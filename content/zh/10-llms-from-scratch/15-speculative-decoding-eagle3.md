# 投机解码与 EAGLE-3

> Phase 7 · Lesson 16 已经证明了数学结论：Leviathan 拒绝规则可以精确保持验证模型的输出分布。本课从训练栈的视角审视 2026 年生产环境中的投机解码。EAGLE-3 把草稿模型从一个廉价的近似品，变成了一个直接在验证模型自身隐藏状态上训练的专用小网络，并加入了训练时测试（training-time test）循环来对齐其训练分布与推理分布。结果：端到端加速 3× 到 6.5×，聊天任务上单 token 接受率超过 0.9，且没有任何分布上的折损。2026 年所有生产推理栈都默认搭载它。

**Type:** Build
**Languages:** Python (stdlib)
**Prerequisites:** Phase 7 · 16 (speculative decoding math), Phase 10 · 12 (inference optimization)
**Time:** ~75 minutes

## 学习目标

- 用一句话陈述 Leviathan 定理，并证明投机解码循环产生的样本与验证模型的输出分布完全相同。
- 梳理从原始投机解码（Leviathan 2023）到 EAGLE、EAGLE-2、EAGLE-3 的两年演进史，并准确指出每一步消除了什么限制。
- 根据接受率 `α` 和草稿与验证模型的成本比 `c` 计算期望加速比，并为不同场景选择最优草稿长度 `N`。
- 从零实现完整的投机解码循环：起草、验证、从残差分布做拒绝采样、拒绝时回滚 KV 缓存、全部接受时输出附赠 token。

## 问题背景

70B 模型在 H100 上做自回归解码，速度大概只有每秒 35 个 token。GPU 远没有被打满。瓶颈在内存带宽：每生成一个 token 都要从 HBM 加载 70B 的权重，做一步算术运算，产出一个浮点数。计算单元大部分时间在空转。

投机解码（speculative decoding）把这变成一个真正可以解决的吞吐量问题。一个廉价的草稿模型用 `N` 次小型前向传播提议 `N` 个 token。验证模型在前缀加上全部 `N` 个草稿 token 上只跑一次。如果验证模型在位置 `i` 的分布与草稿一致（一致的含义在统计意义上，我们稍后会精确定义），就接受；否则拒绝，并从残差分布中采样一个修正 token。一次大模型前向最多能产出 `N+1` 个被接受的 token，而不是一个。

关键定理来自 Leviathan, Kalman, Matias（ICML 2023）：输出分布与直接从验证模型采样得到的分布完全相同。不是近似相同，是完全相同。这正是投机解码能被生产环境接受的全部理由——它是一个纯粹的延迟优化，没有任何质量折损。

Phase 7 · Lesson 16 给你的是数学。本课给你的是训练栈。一个好草稿带来的加速比廉价草稿多 2×。EAGLE、EAGLE-2 和 EAGLE-3（Li et al., 2024–2025）把「草稿 = 同系列的更小模型」这件事变成了一门精确的工程学科。2026 年的生产推理服务器默认使用 EAGLE-3。

## 核心概念

### 不变量：Leviathan 拒绝采样

设 `p(t)` 为草稿模型在给定前缀下对下一个 token 的分布，`q(t)` 为验证模型的分布。采样一个草稿 token `d ~ p`。以概率 `min(1, q(d) / p(d))` 接受。拒绝时，从残差分布 `(q - p)_+ / ||(q - p)_+||_1` 中采样。最终得到的样本服从分布 `q`。无论 `p` 有多差，这一结论都成立——`p` 越差，拒绝越频繁，但输出始终精确。

把 `N` 次这样的调用串起来，只用一次验证模型前向，输入是 `prefix + d_1 + ... + d_N`。验证模型同时返回 `q_1, q_2, ..., q_{N+1}`。从左到右扫描。在位置 `j` 第一次发生拒绝时，从 `residual(q_j, p_j)` 中采样并停止。全部接受时，从 `q_{N+1}` 中额外采样一个附赠 token（bonus token）。

### 什么决定加速比

设 `α` 为每个草稿 token 的期望接受率，`c = cost(draft) / cost(verifier)` 为成本比。每次验证模型前向的期望接受 token 数为：

```
E[accepted] = (1 - α^(N+1)) / (1 - α)
```

每个被接受 token 的期望总耗时为 `(N * c + 1) / E[accepted]`。对 `N` 求最小值，就得到最佳取值。当 `α = 0.8, c = 0.05` 时：最优 `N` 大约在 5–7 之间，加速比 3.2×。当 `α = 0.95, c = 0.02` 时：最优 `N` 大约在 8–10 之间，加速比逼近 5×。

最大的单一杠杆是 `α`。在固定 `N = 5` 的情况下，从 `α = 0.6`（原始草稿）提升到 `α = 0.9`（EAGLE-3），每次验证模型前向的期望接受 token 数从 2.2 提升到 4.1。同一个验证模型，吞吐量提升近 2×。

### 两年演进史

**原始投机解码（Leviathan, 2023）。** 草稿模型是同系列中独立训练的较小 LLM。容易接入，`α ≈ 0.6`，加速最多 2× 左右。

**EAGLE-1（Li et al., 2024）。** 草稿是一个极小的 Transformer——通常只有一两层——以验证模型最后一层的隐藏状态为输入，直接预测下一个 token。由于草稿能看到验证模型的特征表示，其分布与验证模型接近得多。`α` 攀升至 0.7–0.8。

**EAGLE-2（Li et al., 2024）。** 加入动态草稿树（dynamic draft tree）：不再提议单条 `N` 个 token 的序列，而是提议一棵小型候选树，用验证模型在一次前向（树注意力）中给每个候选打分，然后走概率最高的路径。草稿长度变为每步自适应。按接受路径计算的单 token `α` 超过 0.85。

**EAGLE-3（Li et al., 2025, NeurIPS）。** 又有两处改动。第一，彻底去掉特征预测损失——EAGLE-1/2 训练草稿去拟合验证模型的隐藏状态，这限制了数据规模带来的收益。EAGLE-3 直接在 token 预测上训练。第二，训练时测试（training-time test, TTT）：在草稿训练过程中，将草稿自身之前的预测结果多步回灌为输入，与它在推理时的运行方式完全一致。这对齐了训练与测试分布，遏制了误差累积。实测加速：聊天任务最高 6.5×，在 H100 上 SGLang 批大小 64 时吞吐量提升 38%。

### KV 缓存回滚

验证过程会在一次前向中把验证模型的 KV 缓存扩展 `N` 个条目。如果在位置 `j` 发生拒绝，位置 `j-1` 之后的缓存内容就是错的。两种常见实现：先写入暂存缓冲区、接受时才提交（vLLM、TensorRT-LLM）；或维护物理 KV 缓存加一个逻辑长度，拒绝时截断逻辑长度。无论哪种方式，回滚的开销是每层每个注意力头的字节量，相比前向传播的成本可以忽略不计。

对于 EAGLE-2 的树搜索，验证模型用一个符合树拓扑的非因果掩码来计算注意力。工程上有些琐碎，但计算本身就是一次带自定义掩码的标准 flash-attention 调用。

### 2026 年的草稿架构格局

| 策略 | 草稿类型 | `α` | 加速比 | 训练成本 |
|----------|-----------|-----|---------|---------------|
| 原始投机解码 | 独立的小型 LLM | 0.55-0.70 | 1.8-2.3× | 无（复用现有小模型） |
| Medusa | 验证模型上的额外 LM 头 | 0.65-0.75 | 2-3× | ~1B SFT token |
| EAGLE-1 | 基于隐藏状态的 1 层 Transformer | 0.70-0.80 | 2.5-3× | ~60B token |
| EAGLE-2 | EAGLE-1 + 动态草稿树 | 0.80-0.88 | 3-4× | ~60B token |
| EAGLE-3 | 多层特征融合 + TTT | 0.88-0.92 | 3.5-6.5× | ~60-200B token |
| Lookahead | 无草稿（Jacobi 迭代） | N/A | 1.3-1.6× | 无 |

2026 年的生产现状：vLLM 和 SGLang 在可用时默认使用 EAGLE-3，否则使用 EAGLE-2。TensorRT-LLM 为 Meta 和 NVIDIA 公开模型提供最快的 Medusa 路径。llama.cpp 为 CPU 部署提供原始草稿方案。

## 从零实现

参见 `code/main.py`。这是包含全部组件的完整 Leviathan 投机解码循环：起草 `N` 个 token、验证模型并行前向、逐位置拒绝判定、残差采样、附赠 token、KV 回滚，以及验证输出分布与直接从 `q` 采样一致的经验性检验。

### 第 1 步：拒绝规则

```python
def accept(q_prob, p_prob, u):
    if p_prob <= 0:
        return True
    return u < min(1.0, q_prob / p_prob)
```

### 第 2 步：残差分布

```python
def residual(q, p):
    raw = [max(0.0, qi - pi) for qi, pi in zip(q, p)]
    s = sum(raw)
    if s == 0:
        return list(q)
    return [r / s for r in raw]
```

### 第 3 步：完整的投机解码步骤

`spec_step` 函数先从 `p` 起草 `N` 个 token，再用一次并行的 `q` 计算验证全部 token。对每个草稿 token 应用拒绝规则，在第一次拒绝时从残差分布中采样修正 token。如果全部接受，则从 `q_{N+1}` 中输出一个附赠 token。

### 第 4 步：KV 回滚记账

模拟器为每个工作单元维护一个逻辑 `kv_length`。接受 `k` 个草稿时，`kv_length += k`。在位置 `j` 发生拒绝时，缓存已经写到了 `j` 之后，但逻辑长度被设为 `prefix_length + j + 1`——即修正 token 之后一位。后续读取按逻辑长度截断。

### 第 5 步：Leviathan 检验

运行 50,000 次投机解码步骤。统计被接受 token 的经验分布。与直接从 `q` 采样 50,000 次的结果对比。卡方统计量应该远低于临界值。定理在实践中成立。

### 第 6 步：加速比与 α 的关系

通过以不同幅度扰动 `p` 使其偏离 `q`，扫描草稿质量。测量 `α`，然后绘制每次验证模型调用的期望 token 数关于 `α` 和 `N` 的曲线。代码会打印一张表格，展示 EAGLE-3 级别的草稿质量（`α ≈ 0.9`）如何解锁每次验证调用 4–5 个 token。

## 生产实践

带 EAGLE-3 的生产级 `vllm serve`：

```bash
vllm serve meta-llama/Llama-3.3-70B-Instruct \
  --speculative-config '{
    "model": "yuhuili/EAGLE3-LLaMA3.3-Instruct-70B",
    "num_speculative_tokens": 5,
    "method": "eagle3"
  }'
```

SGLang 在 H100 上以批大小 64 运行 EAGLE-3：根据 EAGLE-3 论文，吞吐量比批大小 64 的原始解码高约 1.38×。

适合使用投机解码的场景：

- 任何 p50 延迟比峰值吞吐量更重要的交互式聊天负载。
- 代码生成和结构化输出（JSON、SQL）。由于目标分布高度可预测，`α` 超过 0.9。
- 长文本生成（数千 token）。摊销后的加速持续生效。

不适合的场景：

- 非常小的模型（< 3B）。草稿模型并不比验证模型便宜多少。
- 极小的批大小为 1 的 CPU 部署。草稿模型的内存开销可能不划算。
- 温度极高的创意采样，此时 `α` 会崩塌。

## 交付产物

本课产出 `outputs/skill-eagle3-tuner.md`。给定一个推理负载（模型、批大小、目标延迟、任务画像），它会推荐一套投机解码策略和调优参数（草稿系列、`N`、树深度、温度感知切换）。

## 练习

1. 运行 `code/main.py`。确认 Leviathan 分布检验的卡方统计量在 50,000 个样本下保持在 95% 临界值以下。

2. 固定 `α` 为 0.9、`c` 为 0.04，将 `N` 从 1 扫描到 10。绘制每次验证调用的期望 token 数和每个 token 的实际耗时。找出使耗时最小的 `N`，并解释曲线的形状。

3. 修改代码来模拟 EAGLE-2 树搜索：每一步草稿模型提议一棵形状为 `[2, 2, 2]` 的树（八条候选路径）。验证模型只跑一次，概率最高的被接受路径胜出。计算每个叶节点的 `α` 和每次验证调用的总 token 数。与等量计算下的线性链投机解码做对比。

4. 实现一个支持两个并发序列的批量 KV 回滚模拟器。序列 A 的所有草稿都被接受；序列 B 在位置 2 被拒绝。证明每个序列的 `kv_length` 都被正确更新，且没有浪费任何计算。

5. 阅读 EAGLE-3 论文第 4 节（Training-Time Test）。用两句话解释为什么没有 TTT 的朴素草稿训练会遭受暴露偏差（exposure bias），以及为什么在训练中把草稿自己的预测回灌给它能解决这个问题。将其与 seq2seq 中的 scheduled-sampling 文献联系起来。

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|----------------|------------------------|
| Leviathan 规则 | "min(1, q over p)" | 以概率 `min(1, q(d)/p(d))` 做伯努利接受/拒绝判定；只要拒绝时从残差分布采样，就能精确保持验证模型的分布 |
| 残差分布 | "(q minus p) plus, normalized" | `(q - p)_+` 截断为零后重新归一化——拒绝时应当采样的正确分布 |
| 接受率 α | "草稿猜对的频率" | 在拒绝规则下每 token 伯努利成功的期望概率；支配所有加速比的计算 |
| EAGLE-1 | "hidden-state draft" | 以验证模型最后一层隐藏状态为条件的极小 Transformer 草稿（Li et al., 2024） |
| EAGLE-2 | "dynamic draft tree" | EAGLE-1 加上一棵候选续写树，用树注意力在一次验证前向中打分 |
| EAGLE-3 | "training-time test" | 去掉特征预测损失，直接在 token 预测上训练，并在训练中把草稿自己的输出回灌给它 |
| 训练时测试（TTT） | "exposure bias fix" | 训练时以自回归方式运行草稿模型，使训练与测试的输入分布一致——scheduled sampling 的直接对应物 |
| KV 回滚 | "撤销被拒绝的草稿" | 拒绝发生后把验证模型的 KV 缓存重置到已接受前缀长度的记账操作 |
| 附赠 token | "白送的那个" | 当全部 `N` 个草稿都被接受时，从 `q_{N+1}` 中额外采样一个 token，不增加任何验证成本 |
| 树注意力 | "一次验证多个候选" | 用符合草稿树拓扑的非因果掩码做注意力计算；一次前向算出树中每个节点的 `q_i` |

## 延伸阅读

- [Leviathan, Kalman, Matias — Fast Inference from Transformers via Speculative Decoding (arXiv:2211.17192, ICML 2023)](https://arxiv.org/abs/2211.17192) — 奠基性论文及等价性定理
- [Chen et al. — Accelerating Large Language Model Decoding with Speculative Sampling (arXiv:2302.01318)](https://arxiv.org/abs/2302.01318) — 同期独立提出的工作，证明简洁清晰
- [Li et al. — EAGLE: Speculative Sampling Requires Rethinking Feature Uncertainty (arXiv:2401.15077)](https://arxiv.org/abs/2401.15077) — EAGLE-1，以隐藏状态为条件的草稿
- [Li et al. — EAGLE-2: Faster Inference of Language Models with Dynamic Draft Trees (arXiv:2406.16858)](https://arxiv.org/abs/2406.16858) — 动态树搜索
- [Li et al. — EAGLE-3: Scaling up Inference Acceleration via Training-Time Test (arXiv:2503.01840, NeurIPS 2025)](https://arxiv.org/abs/2503.01840) — 2026 年的生产默认方案
- [Cai et al. — Medusa: Multiple Decoding Heads (arXiv:2401.10774)](https://arxiv.org/abs/2401.10774) — 不依赖草稿模型的替代方案
- [vLLM Speculative Decoding documentation](https://docs.vllm.ai/en/latest/features/spec_decode.html) — 集成了所有策略的权威生产参考
