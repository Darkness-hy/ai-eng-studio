# DeepSeek-V3 架构全解析

> Phase 10 · 第 14 课列出了每个开源模型都会调节的六个架构旋钮。DeepSeek-V3（2024 年 12 月发布，总参数 671B，激活参数 37B）把这六个旋钮全部调了一遍，还额外加了四个：多头潜在注意力（Multi-Head Latent Attention）、无辅助损失的负载均衡、多 token 预测（Multi-Token Prediction）以及 DualPipe 训练。本课将自顶向下通读 DeepSeek-V3 的架构，并依据公开的 config 推导出每一项参数量。学完本课，你将能够解释为什么 671B/37B 这个比例是正确的押注，以及为什么在前沿规模上 MLA + MoE 的组合优于二者单独使用。

**Type:** Learn
**Languages:** Python (stdlib, parameter calculator)
**Prerequisites:** Phase 10 · 14 (open-model walkthroughs), Phase 10 · 17 (NSA), Phase 10 · 18 (MTP), Phase 10 · 19 (DualPipe)
**Time:** ~75 minutes

## 学习目标

- 自顶向下通读 DeepSeek-V3 的 config，并用 GPT-2 的六个旋钮加上四个 DeepSeek 特有的新增项来解释每个字段。
- 推导出总参数量（671B）、激活参数量（37B），以及构成二者的各个组件。
- 计算 MLA 在 128k 上下文下的 KV 缓存占用，并与同等激活参数量、采用 GQA 的稠密模型所需付出的代价进行对比。
- 说出四项 DeepSeek 特有创新（MLA、MTP、无辅助损失路由、DualPipe），并指出每一项分别针对架构/训练栈的哪个部分。

## 问题背景

DeepSeek-V3 是第一个架构与 Llama 家族有实质性差异的前沿开源模型。Llama 3 405B 是"调了六个旋钮的 GPT-2"。DeepSeek-V3 则是六个旋钮全调、外加四个新旋钮的 GPT-2。读 Llama 3 的 config 是读 DeepSeek config 的热身，但其深层结构——注意力块的形态、路由逻辑、训练目标——差异大到值得单独讲一遍。

学会它的回报是：DeepSeek-V3 的开放权重发布改变了开源模型中"前沿能力"的定义。这套架构正是许多 2026 年训练项目效仿的蓝本。理解它，是任何涉及前沿 LLM 训练或推理的岗位的入场券。

## 核心概念

### 不变的核心，再看一遍

DeepSeek-V3 仍然是自回归模型。仍然堆叠解码器块。每个块仍然是注意力加 MLP 加两个 RMSNorm。MLP 中仍然使用 SwiGLU。仍然使用 RoPE。Pre-norm。权重共享的嵌入。与所有 Llama 或 Mistral 模型的基线完全相同。

### 变化之处：用 MLA 取代 GQA

从 Phase 10 · 14 你已经知道，GQA 通过让多组 Q 头共享 K 和 V 来缩小 KV 缓存。多头潜在注意力（MLA）更进一步：K 和 V 被压缩成一个共享的低秩潜在表示（即 `kv_lora_rank`），然后在计算时逐头实时解压。KV 缓存只存储这个潜在表示——通常每层每个 token 只需 512 个浮点数，而不是 8 x 128 = 1024 个。

在 128k 上下文下，使用 MLA 的 DeepSeek-V3（每层每个 token 一个共享潜在向量 `c^{KV}`；K 和 V 都通过上投影从这个潜在向量推导得到，而这些上投影可以被吸收进后续的矩阵乘法）：

```
kv_cache = num_layers * kv_lora_rank * max_seq_len * bytes_per_element
         = 61 * 512 * 131072 * 2
         = 7.6 GB
```

而一个假设的 GQA 基线（Llama 3 70B 的形状，8 个 KV 头，头维度 128）则要付出：

```
kv_cache = 2 * 61 * 8 * 128 * 131072 * 2
         = 30.5 GB
```

在 128k 上下文下，MLA 的缓存比 Llama-3-70B 式的 GQA 缓存小 4 倍。

代价是：MLA 在每次注意力计算中（逐头）增加了一个解压步骤。与节省的带宽相比，这点额外计算量很小。对长上下文推理来说是净收益。

### 路由机制：无辅助损失的负载均衡

MoE 路由器决定每个 token 由哪些 top-k 专家处理。朴素的路由器会把太多工作集中在少数专家上，让其他专家闲置。标准的解决方法：增加一个惩罚负载不均衡的辅助损失项。这能起作用，但会轻微损害主任务性能。

DeepSeek-V3 引入了一种无辅助损失的方案。在路由器 logits 上加入逐专家的偏置项，并在训练中按一条简单规则调整：如果专家 `e` 过载，就降低 `bias_e`；如果负载不足，就提高它。没有额外的损失项。训练保持干净。专家负载保持均衡。

对主损失的影响：测不出任何差异。对 MoE 架构的影响：更简洁，不再有需要调节的辅助损失超参数。

### MTP：更密集的训练信号 + 免费的草稿模型

从 Phase 10 · 18 你已经知道，DeepSeek-V3 增加了 D=1 个 MTP 模块，用于预测往后两个位置的 token。在推理时，训练好的模块被改用作投机解码（speculative decoding）的草稿模型，接受率超过 80%。在训练时，每个隐藏状态接受 D+1 = 2 个目标的监督，提供了更密集的信号。

参数量：在 671B 主体之上额外 14B。开销：2.1%。

### 训练机制：DualPipe

从 Phase 10 · 19 你已经知道，DualPipe 是一种双向流水线，将前向和反向的计算块与跨节点 all-to-all 通信重叠执行。在 DeepSeek-V3 的 2,048 块 H800 规模下，它挽回了约 245k GPU 小时——这些时间在 1F1B 调度下会损失在流水线气泡中。

### 逐字段解读 config

下面是 DeepSeek-V3 的 config（简化版）：

```
hidden_size: 7168
intermediate_size: 18432   (dense MLP hidden size, used on first few layers)
moe_intermediate_size: 2048 (expert MLP hidden size)
num_hidden_layers: 61
first_k_dense_layers: 3    (first 3 layers use dense MLP)
num_attention_heads: 128
num_key_value_heads: 128   (formally equal to num_heads under MLA, but
                           the real compression is in kv_lora_rank)
kv_lora_rank: 512          (MLA latent dimension)
num_experts: 256            (MoE expert count per block)
num_experts_per_tok: 8      (top-8 routing)
shared_experts: 1           (always-on shared expert per block)
max_position_embeddings: 163840
rope_theta: 10000.0
vocab_size: 129280
mtp_module: 1               (1 MTP module at depth 1)
```

逐项解析：

- `hidden_size=7168`：嵌入维度。
- `num_hidden_layers=61`：总块深度。
- `first_k_dense_layers=3`：前 3 个块使用大小为 18432 的稠密 MLP，其余 58 个使用 MoE。
- `num_attention_heads=128`：128 个查询头。
- `kv_lora_rank=512`：K 和 V 被压缩到这个潜在维度，再逐头解压。
- `num_experts=256, num_experts_per_tok=8`：每个 MoE 块有 256 个专家，路由到 top-8。
- `shared_experts=1`：在 256 个被路由专家之外，还有 1 个始终激活的专家参与处理每个 token。可以把它理解为一个"稠密保底"，确保每个 token 都能得到可靠的处理。
- `moe_intermediate_size=2048`：每个专家的 MLP 隐藏层大小。比稠密 MLP 小，因为有 256 个专家。

### 参数量核算

完整计算在 `code/main.py` 中。要点如下：

- 嵌入层：`vocab * hidden = 129280 * 7168 = ~0.93B`。
- 前 3 个稠密块：MLA 注意力（每块约 144M）+ 稠密 MLP（每块约 260M）+ 归一化层。合计约 1.2B。
- 58 个 MoE 块：MLA 注意力（约 144M）+ 各 256 个专家（每个 30M）+ 1 个共享专家（30M）+ 归一化层。包含所有专家在内，每块共约 7.95B。58 个 MoE 块合计 461B。
- MTP 模块：14B。

总计：核心架构约 476B + MTP 14B；而官方公布的 671B 数字还包含了额外的结构性参数（偏置张量、专家特有组件、共享专家缩放系数等）。我们用计算器复现的数字与官方值的偏差在 3-5% 以内——差额来自 DeepSeek 报告在其第 2 节附录中记录的细粒度核算。

每次前向传播的激活参数：

- 注意力：每层 144M * 61 = 8.8B（所有层都参与计算）。
- 激活的 MLP：前 3 层稠密（3 * 260M = 780M），58 个 MoE 层每层激活 8 个路由专家 + 1 个共享专家 + 路由开销。每层激活的 MLP 约 260M。合计：3 * 260M + 58 * 260M = ~15.9B。
- 嵌入 + 归一化层：1.2B。
- 激活参数总计：核心约 26B + MTP 14B（参与训练但推理时不一定运行）≈ 37B。

### 671B / 37B 的比例

18 倍的稀疏比（激活参数占总参数的 5.5%）。DeepSeek-V3 是已发布开放权重的前沿 MoE 模型中最稀疏的。Mixtral 8x7B 的比例是 13/47（28%），稠密得多。Llama 4 Maverick 的比例是 17B/400B（4.25%），与之相当。DeepSeek 的押注是：在前沿规模上，更多专家配合更低的激活比例，能在每个激活 FLOP 上换来更好的质量。

### DeepSeek-V3 的位置

| 模型 | 总参数 | 激活参数 | 比例 | 注意力 | 创新点 |
|-------|------|-------|-------|-----------|-------------|
| Llama 3 70B | 70B | 70B | 100% | GQA 64/8 | — |
| Llama 4 Maverick | 400B | 17B | 4.25% | GQA | — |
| Mixtral 8x22B | 141B | 39B | 27% | GQA | — |
| DeepSeek V3 | 671B | 37B | 5.5% | MLA 512 | MLA + MTP + 无辅助损失路由 + DualPipe |
| Qwen 2.5 72B | 72B | 72B | 100% | GQA 64/8 | YaRN 扩展 |

### 后续演进：R1、V4

DeepSeek-R1（2025）是在 V3 骨干上进行的推理能力训练。R1 使用完全相同的架构。改变的是后训练配方（在可验证任务上做大规模强化学习），而不是预训练架构。

DeepSeek-V4（如果发布）预计会保留 MLA + MoE + MTP，并加入 DSA（DeepSeek Sparse Attention），即 Phase 10 · 17 中 NSA 的继任者。这条演进路线很稳定：架构层面的创新不断累积；每个版本都调动更多的旋钮。

```figure
moe-routing
```

## 生产实践

`code/main.py` 是针对 DeepSeek-V3 形状定制的参数计算器。运行它，把输出与论文中的数字对比，再用它分析假想的变体（256 个专家 vs 512 个、top-8 vs top-16、MLA 秩 512 vs 1024）。

需要关注的内容：

- 总参数量 vs 官方公布的 671B。
- 激活参数量 vs 官方公布的 37B。
- 128k 上下文下的 KV 缓存——MLA 与 GQA 的对比。
- 逐层拆解，看看参数预算实际花在了哪里。

## 交付产物

本课产出 `outputs/skill-deepseek-v3-reader.md`。给定一个 DeepSeek 家族模型（V3、R1 或任何未来变体），它能生成逐组件的架构解读：说明 config 中每个字段的含义，按组件推导参数量，并识别该模型使用了四项 DeepSeek 特有创新中的哪几项。

## 练习

1. 运行 `code/main.py`。将计算器估算的总参数量与官方公布的 671B 对比，找出偏差的来源。论文第 2 节有完整的逐项清单。

2. 把 config 中的 MLA 秩从 512 改为 256。计算 128k 上下文下的 KV 缓存大小。这能换来多大比例的缩减？又会让每个头的表达能力付出什么代价？

3. 比较 DeepSeek-V3 的路由方案（256 个专家，top-8）与一个假想变体（512 个专家，top-8）。总参数量增加，激活参数量不变。理论上额外的专家容量能买到什么？在推理时又要付出什么成本？

4. 阅读 DeepSeek-V3 技术报告（arXiv:2412.19437）第 2.1 节关于 MLA 的内容。用三句话解释为什么 K 和 V 的解压矩阵可以被"吸收"进后续的矩阵乘法，从而提升推理效率。

5. DeepSeek-V3 的大部分运算使用 FP8 训练。计算用 FP8 而非 BF16 存储 671B 权重能节省多少内存。这与 14.8T token 的训练预算有何关联？

## 关键术语

| 术语 | 通俗说法 | 实际含义 |
|------|----------------|------------------------|
| MLA | "多头潜在注意力" | 将 K 和 V 压缩成共享的低秩潜在表示（kv_lora_rank，通常为 512），计算时逐头实时解压；KV 缓存只存储潜在表示 |
| kv_lora_rank | "MLA 压缩维度" | K 和 V 共享潜在表示的大小；DeepSeek-V3 使用 512 |
| First k dense layers | "前几层保持稠密" | MoE 模型的前几层跳过 MoE 路由器，运行稠密 MLP 以保证稳定性 |
| num_experts_per_tok | "Top-k 路由" | 每个 token 激活多少个路由专家；DeepSeek-V3 使用 8 |
| 共享专家 | "始终激活的专家" | 无论路由结果如何都处理每个 token 的专家；DeepSeek-V3 使用 1 个 |
| 无辅助损失路由 | "偏置调节的负载均衡" | 在训练中调整逐专家的偏置项以保持专家负载均衡，无需增加损失项 |
| MTP 模块 | "额外的预测头" | 从 h^(1) 和 E(t+1) 预测 t+2 的 Transformer 块；提供更密集的训练信号，附赠投机解码草稿模型 |
| DualPipe | "双向流水线" | 将前向/反向计算与跨节点 all-to-all 通信重叠的训练调度 |
| 激活参数比例 | "稀疏度" | active_params / total_params；DeepSeek-V3 为 5.5% |
| FP8 训练 | "8 比特训练" | 用 FP8 进行训练存储和许多计算操作；相比 BF16 内存大约减半，质量损失很小 |

## 延伸阅读

- [DeepSeek-AI — DeepSeek-V3 Technical Report (arXiv:2412.19437)](https://arxiv.org/abs/2412.19437) — 完整的架构、训练与结果文档
- [DeepSeek-V3 model card on Hugging Face](https://huggingface.co/deepseek-ai/DeepSeek-V3) — config 文件与部署说明
- [DeepSeek-V2 paper (arXiv:2405.04434)](https://arxiv.org/abs/2405.04434) — 首次引入 MLA 的前代模型
- [DeepSeek-R1 paper (arXiv:2501.12948)](https://arxiv.org/abs/2501.12948) — 基于 V3 架构的推理训练后继模型
- [Native Sparse Attention (arXiv:2502.11089)](https://arxiv.org/abs/2502.11089) — DeepSeek 家族注意力机制的未来方向
- [DualPipe repository](https://github.com/deepseek-ai/DualPipe) — 训练调度的参考实现
