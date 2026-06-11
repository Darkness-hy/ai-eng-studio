# GPT —— 因果语言建模

> BERT 能看到两侧，GPT 只能看到过去。那个三角形掩码，是现代 AI 中影响最深远的一行代码。

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 7 · 02 (Self-Attention), Phase 7 · 05 (Full Transformer), Phase 7 · 06 (BERT)
**Time:** ~75 minutes

## 问题背景

语言模型回答的只有一个问题：给定前 `t-1` 个 token，第 `t` 个 token 的概率分布是什么？用这个信号——下一个 token 预测（next-token prediction）——来训练，你就得到了一个能逐 token 生成任意文本的模型。

要在整条序列上并行地端到端训练，每个位置的预测必须只依赖于更早的位置。否则模型只要偷看答案就能轻松作弊。

因果掩码（causal mask）正是为此而生。它就是一个上三角的 `-inf` 矩阵，在 softmax 之前加到注意力分数上。经过 softmax 后，这些位置的权重变为 0。每个位置只能关注自己和更早的位置。而且由于它一次性作用于整条序列，一次前向传播就能得到 N 个并行的下一个 token 预测。

GPT-1（2018）、GPT-2（2019）、GPT-3（2020）、GPT-4（2023）、GPT-5（2024）、Claude、Llama、Qwen、Mistral、DeepSeek、Kimi——它们全都是 decoder-only 的因果 Transformer，核心循环完全相同。区别只在于更大的规模、更好的数据和更好的 RLHF。

## 核心概念

![Causal mask creates a triangular attention matrix](../assets/causal-attention.svg)

### 掩码

给定长度为 `N` 的序列，构造一个 `N × N` 矩阵：

```
M[i, j] = 0       if j <= i
M[i, j] = -inf    if j > i
```

在 softmax 之前把 `M` 加到原始注意力分数上。由于 `exp(-inf) = 0`，被掩码的位置贡献的权重为零。注意力矩阵的每一行都是一个只覆盖之前位置的概率分布。

实现成本：一次 `torch.tril()` 调用。计算耗时：纳秒级。对整个领域的影响：决定了一切。

### 并行训练，串行推理

训练：把整条 `(N, d_model)` 序列一次性前向传播，计算 N 个交叉熵损失（每个位置一个），求和，反向传播。沿序列维度完全并行。这就是 GPT 训练能扩展的原因——一个 batch 的 100 万个 token 只需一次 GPU 前向。

推理：你只能逐 token 生成。输入 `[t1, t2, t3]`，得到 `t4`。输入 `[t1, t2, t3, t4]`，得到 `t5`。输入 `[t1, t2, t3, t4, t5]`，得到 `t6`。KV 缓存（第 12 课）会保存 `t1…tn` 的隐藏状态，避免每一步重复计算。但推理时的串行深度等于输出长度。这就是自回归税（autoregressive tax），也是解码成为所有 LLM 延迟瓶颈的原因。

### 损失函数 —— 错位一格

给定 token 序列 `[t1, t2, t3, t4]`：

- 输入：`[t1, t2, t3]`
- 目标：`[t2, t3, t4]`

对每个位置 `i`，计算 `-log P(target_i | inputs[:i+1])`，然后求和。这就是整条序列的交叉熵。

你听说过的每一个 Transformer 语言模型都用这个损失训练。预训练、微调、SFT——损失相同，只是数据不同。

### 解码策略

训练完成后，采样方式的选择比人们想象的更重要。

| 方法 | 做什么 | 何时使用 |
|--------|--------------|-------------|
| 贪心（Greedy） | 每一步取 argmax | 确定性任务、代码补全 |
| 温度（Temperature） | logits 除以 T 后采样 | 创意任务，T 越高多样性越强 |
| Top-k | 只从前 k 个 token 中采样 | 砍掉低概率长尾 |
| Top-p（核采样） | 从累积概率 ≥ p 的最小集合中采样 | 2020 年后的默认选择；能自适应分布形状 |
| Min-p | 保留满足 `p > min_p * max_p` 的 token | 2024 年后出现；比 top-p 更善于剔除长尾 |
| 投机解码（Speculative decoding） | 草稿模型提议 N 个 token，大模型验证 | 在同等质量下降低 2–3 倍延迟 |

到 2026 年，min-p 加温度 0.7 是开源权重模型的合理默认配置。投机解码则是任何生产推理栈的标配。

### 「GPT 配方」为什么成功

1. **Decoder-only。** 没有编码器开销。每层只需一次注意力 + FFN。
2. **规模扩展。** 124M → 1.5B → 175B → 数万亿参数。Chinchilla 扩展律（第 13 课）告诉你算力该怎么花。
3. **上下文学习（in-context learning）。** 大约在 6B–13B 规模时涌现。模型无需微调即可跟随 few-shot 示例。
4. **RLHF。** 基于人类偏好的后训练，把原始的预训练文本模型变成了聊天助手。
5. **Pre-norm + RoPE + SwiGLU。** 保证大规模训练的稳定性。

核心架构自 GPT-2 以来几乎没变。所有有趣的进展都发生在数据、规模和后训练上。

```figure
causal-mask
```

## 从零实现

### 第 1 步：因果掩码

见 `code/main.py`。一行就够：

```python
def causal_mask(n):
    return [[0.0 if j <= i else float("-inf") for j in range(n)] for i in range(n)]
```

在 softmax 之前把它加到注意力分数上。整个机制就这么多。

### 第 2 步：一个 2 层的类 GPT 模型

堆叠两个解码器块（带掩码的自注意力 + FFN，没有交叉注意力）。再加上 token 嵌入、位置编码和反嵌入层（unembedding，与 token 嵌入矩阵共享权重——这是 GPT-2 以来的标准技巧）。

### 第 3 步：端到端的下一个 token 预测

在一个 20 词的玩具词表上，为每个位置生成 logits。用错位一格的目标计算交叉熵损失。不求梯度——这只是一次前向传播的健全性检查。

### 第 4 步：采样

实现贪心、温度、top-k、top-p、min-p。对同一个固定 prompt 分别运行并比较输出。一个采样函数也就 10 行代码。

## 生产实践

PyTorch，2026 年惯用写法：

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-3B-Instruct")
tok = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-3B-Instruct")

prompt = "Attention is all you need because"
inputs = tok(prompt, return_tensors="pt")
out = model.generate(
    **inputs,
    max_new_tokens=64,
    temperature=0.7,
    top_p=0.9,
    do_sample=True,
)
print(tok.decode(out[0]))
```

在底层，`generate()` 执行前向传播，取出最后一个位置的 logits，采样下一个 token，把它追加到序列末尾，然后重复。所有生产级 LLM 推理栈（vLLM、TensorRT-LLM、llama.cpp、Ollama、MLX）实现的都是同一个循环，只是做了大量优化——批量 prefill、连续批处理（continuous batching）、KV 缓存分页、投机解码。

**GPT vs BERT，各用一句话概括：** GPT 预测 `P(x_t | x_{<t})`，BERT 预测 `P(x_masked | x_unmasked)`。损失函数决定了模型能否生成。

## 交付产物

见 `outputs/skill-sampling-tuner.md`。该 skill 为新的生成任务选择采样参数，并在需要确定性解码时给出提示。

## 练习

1. **简单。** 运行 `code/main.py`，验证 softmax 后的因果注意力矩阵是下三角的。抽查：第 3 行的权重应只出现在第 0–3 列。
2. **中等。** 实现宽度为 4 的束搜索（beam search）。在 10 个短 prompt 上比较 beam-4 与贪心的困惑度。束搜索总是赢吗？（提示：在翻译任务上通常是，在开放式对话上则不然。）
3. **困难。** 实现投机解码：用一个 2 层小模型做草稿，6 层模型做验证器。在 100 条长度为 64 的补全上测量实际耗时加速比。确认输出与验证器的贪心解码完全一致。

## 关键术语

| 术语 | 大家怎么说 | 实际含义 |
|------|-----------------|-----------------------|
| 因果掩码 | 「那个三角形」 | 加到注意力分数上的上三角 `-inf` 矩阵，使位置 `i` 只能看到位置 `≤ i`。 |
| 下一个 token 预测 | 「那个损失」 | 模型分布与每个位置真实下一个 token 之间的交叉熵。 |
| 自回归 | 「一次生成一个」 | 把输出回喂作输入；并行只存在于训练阶段，生成阶段没有。 |
| Logits | 「softmax 前的分数」 | LM 头在 softmax 之前的原始输出；采样就在它上面进行。 |
| 温度 | 「创造力旋钮」 | logits 除以 T；T→0 等于贪心，T→∞ 等于均匀分布。 |
| Top-p | 「核采样」 | 把分布截断到累积概率 ≥ p 的最小集合，再从剩余部分采样。 |
| Min-p | 「比 top-p 更好」 | 保留满足 `p ≥ min_p × max_p` 的 token；截断阈值随分布尖锐程度自适应。 |
| 投机解码 | 「草稿 + 验证」 | 便宜的模型提议 N 个 token，大模型并行验证。 |
| Teacher forcing | 「训练技巧」 | 训练时输入真实的前一个 token，而不是模型自己的预测。所有 seq2seq 语言模型的标准做法。 |

## 延伸阅读

- [Radford et al. (2018). Improving Language Understanding by Generative Pre-Training](https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf) —— GPT-1。
- [Radford et al. (2019). Language Models are Unsupervised Multitask Learners](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf) —— GPT-2。
- [Brown et al. (2020). Language Models are Few-Shot Learners](https://arxiv.org/abs/2005.14165) —— GPT-3 与上下文学习。
- [Leviathan, Kalman, Matias (2023). Fast Inference from Transformers via Speculative Decoding](https://arxiv.org/abs/2211.17192) —— 投机解码论文。
- [HuggingFace `modeling_llama.py`](https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py) —— 因果语言模型的权威参考实现。
