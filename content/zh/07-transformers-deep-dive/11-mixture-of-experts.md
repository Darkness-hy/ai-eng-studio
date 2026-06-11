# 混合专家模型（Mixture of Experts，MoE）

> 一个 70B 的稠密 Transformer 对每个 token 都要激活全部参数。而一个 671B 的 MoE 每个 token 只激活 37B 参数，却在所有基准测试中都胜出。稀疏性是这十年最重要的扩展思想。

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 7 · 05 (Full Transformer), Phase 7 · 07 (GPT)
**Time:** ~45 minutes

## 问题背景

稠密 Transformer 在推理时的 FLOPs 等于其参数量（前向传播再乘以 2）。把稠密模型规模做大，每个 token 都得付全额账单。到 2024 年，前沿模型已经撞上了算力墙：要想显著更聪明，每个 token 需要的 FLOPs 得指数级增长。

混合专家模型打破了这个绑定。把每个 FFN 替换成 `E` 个独立的专家，外加一个为每个 token 挑选 `k` 个专家的路由器（router）。总参数量 = `E × FFN_size`。每个 token 的激活参数量 = `k × FFN_size`。2026 年的典型配置：`E=256`，`k=8`。存储随 `E` 增长，计算随 `k` 增长。

2026 年的前沿模型几乎全是 MoE：DeepSeek-V3（总参数 671B / 激活 37B）、Mixtral 8×22B、Qwen2.5-MoE、Llama 4、Kimi K2、gpt-oss。在 Artificial Analysis 的独立排行榜上，开源模型前 10 名全部是 MoE。

## 核心概念

![MoE layer: router selects k of E experts per token](../assets/moe.svg)

### 替换 FFN

稠密 Transformer 块：

```
h = x + attn(norm(x))
h = h + FFN(norm(h))
```

MoE 块：

```
h = x + attn(norm(x))
scores = router(norm(h))              # (N_tokens, E)
top_k = argmax_k(scores)              # pick k of E per token
h = h + sum_{e in top_k}(
        gate(scores[e]) * Expert_e(norm(h))
    )
```

每个专家都是一个独立的 FFN（通常是 SwiGLU）。路由器只是一个线性层。每个 token 各自挑选自己的 `k` 个专家，得到这些专家输出的门控加权混合。

### 负载均衡问题

如果路由器把 90% 的 token 都送进专家 3，其他专家就会"饿死"。业界尝试过三种解决方案：

1. **辅助负载均衡损失**（Switch Transformer、Mixtral）。加一项与专家使用率方差成正比的惩罚。有效，但引入了一个超参数和第二个梯度信号。
2. **专家容量 + 丢弃 token**（早期 Switch）。每个专家最多处理 `C × N/E` 个 token；溢出的 token 直接跳过该层。损害质量。
3. **无辅助损失的均衡**（DeepSeek-V3）。给每个专家加一个可学习的偏置，用来调整路由器的 top-k 选择。偏置在训练损失之外更新，对主目标没有任何惩罚。这是 2024 年的重大突破。

DeepSeek-V3 的做法：每个训练步结束后，逐个检查每个专家的使用率是高于还是低于目标值，把偏置微调 `±γ`。选择专家时用 `scores + bias`，而用于门控的专家概率仍然是原始的 `scores`，不做改动。这把路由与表达解耦了。

### 共享专家

DeepSeek-V2/V3 还把专家分为*共享*专家和*路由*专家。每个 token 都会经过所有共享专家；路由专家则通过 top-k 选出。共享专家捕获通用知识，路由专家负责专精。V3 使用 1 个共享专家，外加从 256 个路由专家中选 top-8。

### 细粒度专家

经典 MoE（GShard、Switch）：每个专家与完整 FFN 一样宽。`E` 很小（8–64），`k` 很小（1–2）。

现代细粒度 MoE（DeepSeek-V3、Qwen-MoE）：每个专家更窄（FFN 大小的 1/8）。`E` 很大（256+），`k` 更大（8+）。总参数量相同，但组合数增长得快得多。每个 token 有 `C(256, 8) = 400 trillion` 种可能的"专家组合"。质量上升，延迟不变。

### 成本画像

每个 token、每一层：

| 配置 | 每 token 激活参数 | 总参数 |
|--------|-----------------------|--------------|
| Mixtral 8×22B | ~39B | 141B |
| Llama 3 70B（稠密） | 70B | 70B |
| DeepSeek-V3 | 37B | 671B |
| Kimi K2（MoE） | ~32B | 1T |

DeepSeek-V3 在几乎所有基准上都胜过 Llama 3 70B（稠密），而且**每个 token 的激活 FLOPs 还更少**。参数越多 = 知识越多；激活 FLOPs 越多 = 每个 token 的计算量越大。MoE 把这两者解耦了。

### 代价：内存

无论哪些专家被激活，所有专家都得驻留在 GPU 上。一个 671B 的模型，fp16 权重需要约 1.3 TB 显存。前沿 MoE 部署必须使用专家并行（expert parallelism）——把专家分片到多块 GPU 上，让 token 在网络间路由。延迟的主要瓶颈是 all-to-all 通信，而不是矩阵乘法。

## 从零实现

参见 `code/main.py`。一个纯标准库实现的紧凑 MoE 层，包含：

- `n_experts=8` 个类 SwiGLU 专家（为演示方便，每个只有一个线性层）
- top-k=2 路由
- softmax 归一化的门控权重
- 基于逐专家偏置的无辅助损失均衡

### 第 1 步：路由器

```python
def route(hidden, W_router, top_k, bias):
    scores = [sum(h * w for h, w in zip(hidden, W_router[e])) for e in range(len(W_router))]
    biased = [s + b for s, b in zip(scores, bias)]
    top_idx = sorted(range(len(biased)), key=lambda i: -biased[i])[:top_k]
    # softmax over ORIGINAL scores of the chosen experts
    chosen = [scores[i] for i in top_idx]
    m = max(chosen)
    exps = [math.exp(c - m) for c in chosen]
    s = sum(exps)
    gates = [e / s for e in exps]
    return top_idx, gates
```

偏置只影响选择，不影响门控权重。这正是 DeepSeek-V3 的诀窍——偏置纠正负载不均，却不会干扰模型的预测。

### 第 2 步：让 100 个 token 通过路由器

记录每个专家被激活的频率。不加偏置时，使用率是偏斜的。加上偏置更新循环（过载专家 `-γ`，欠载专家 `+γ`）后，几次迭代内使用率就收敛到均匀分布。

### 第 3 步：参数量对比

打印一个 MoE 配置的"稠密等价物"。按 DeepSeek-V3 的形状：256 个路由专家 + 1 个共享专家，激活 8 个，d_model=7168。总参数量大得惊人，而激活参数量只有稠密 Llama 3 70B 的七分之一。

## 生产实践

HuggingFace 加载：

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
model = AutoModelForCausalLM.from_pretrained("mistralai/Mixtral-8x22B-v0.1")
```

2026 年的生产推理：vLLM 原生支持 MoE 路由，SGLang 拥有最快的专家并行路径。两者都自动处理 top-k 选择和专家并行。

**何时选择 MoE：**
- 你想以更低的单 token 推理成本获得前沿质量。
- 你有足够的显存 / 专家并行基础设施。
- 你的工作负载是 token 密集型（聊天、代码），而不是上下文密集型（长文档）。

**何时不该选 MoE：**
- 边缘部署——任何激活 FLOP 都要付出全额存储成本。
- 延迟敏感的单用户服务——专家路由会增加开销。
- 小模型（<7B）——MoE 的质量优势只在超过某个算力阈值（约 6B 激活参数）后才会显现。

## 交付产物

参见 `outputs/skill-moe-configurator.md`。该技能根据参数预算、训练 token 数和部署目标，为新的 MoE 模型选定 E、k 以及共享专家布局。

## 练习

1. **简单。** 运行 `code/main.py`。观察无辅助损失的偏置更新如何在 50 次迭代内把专家使用率拉平。
2. **中等。** 把可学习路由器换成基于哈希的路由器（确定性、无需学习）。比较质量和均衡度。为什么可学习路由器更好？
3. **困难。** 实现 GRPO 风格的"rollout 匹配路由"（DeepSeek-V3.2 的技巧）：记录推理时激活了哪些专家，在梯度计算时强制使用相同的路由。在一个玩具策略梯度实验中测量其效果。

## 关键术语

| 术语 | 人们常说的 | 实际含义 |
|------|-----------------|-----------------------|
| 专家（Expert） | "众多 FFN 之一" | 一个独立的前馈网络；其参数专用于 FFN 计算中的一个稀疏切片。 |
| 路由器（Router） | "门控" | 一个很小的线性层，为每个 token 对每个专家打分；做 top-k 选择。 |
| Top-k 路由 | "每个 token 激活 k 个专家" | 每个 token 的 FFN 计算恰好经过 k 个专家，按门控权重加权。 |
| 辅助损失 | "负载均衡惩罚" | 额外的损失项，惩罚偏斜的专家使用分布。 |
| 无辅助损失 | "DeepSeek-V3 的技巧" | 仅通过逐专家偏置影响路由器的选择来实现均衡；不引入额外梯度。 |
| 共享专家 | "始终激活" | 每个 token 都会经过的额外专家；捕获通用知识。 |
| 专家并行 | "按专家分片" | 把不同专家分布到不同 GPU 上；让 token 在网络间路由。 |
| 稀疏性 | "激活参数 < 总参数" | 比值 `k × expert_size / (E × expert_size)`；DeepSeek-V3 为 37/671 ≈ 5.5%。 |

## 延伸阅读

- [Shazeer et al. (2017). Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer](https://arxiv.org/abs/1701.06538) —— 思想的起源。
- [Fedus, Zoph, Shazeer (2022). Switch Transformer: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity](https://arxiv.org/abs/2101.03961) —— Switch，经典的 MoE。
- [Jiang et al. (2024). Mixtral of Experts](https://arxiv.org/abs/2401.04088) —— Mixtral 8×7B。
- [DeepSeek-AI (2024). DeepSeek-V3 Technical Report](https://arxiv.org/abs/2412.19437) —— MLA + 无辅助损失 MoE + MTP。
- [Wang et al. (2024). Auxiliary-Loss-Free Load Balancing Strategy for Mixture-of-Experts](https://arxiv.org/abs/2408.15664) —— 基于偏置的均衡方法论文。
- [Dai et al. (2024). DeepSeekMoE: Towards Ultimate Expert Specialization in Mixture-of-Experts Language Models](https://arxiv.org/abs/2401.06066) —— 本课路由器所采用的细粒度 + 共享专家拆分方案。
- [Kim et al. (2022). DeepSpeed-MoE: Advancing Mixture-of-Experts Inference and Training](https://arxiv.org/abs/2201.05596) —— 最早提出共享专家的论文。
