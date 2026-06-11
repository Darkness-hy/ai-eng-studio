# 从零构建 Transformer — 毕业项目

> 十三节课。一个模型。没有捷径。

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 7 · 01 through 13. Don't skip.
**Time:** ~120 minutes

## 问题背景

每篇论文你都读过了。注意力、多头拆分、位置编码、编码器与解码器块、BERT 和 GPT 的损失函数、MoE、KV 缓存，你都亲手实现过了。现在，让它们在一个真实任务上协同工作。

毕业项目：在字符级语言建模任务上，端到端训练一个小型 decoder-only Transformer。它读莎士比亚，再生成新的莎士比亚。它小到能在笔记本电脑上 10 分钟内训完；又足够正确——换上更大的数据集、训练更久，你就能得到一个真正的语言模型。

这是本课程的「nanoGPT」。它并非原创——Karpathy 2023 年的 nanoGPT 教程是每个学习者至少要写一遍的参考实现。我们沿用它的整体形态，并围绕我们已学过的内容重新打磨。

## 核心概念

![Transformer-from-scratch block diagram](../assets/capstone.svg)

带注解的架构图：

```
input tokens (B, N)
   │
   ▼
token embedding + positional embedding  ◀── Lesson 04 (RoPE option)
   │
   ▼
┌──── block × L ────────────────────┐
│  RMSNorm                          │  ◀── Lesson 05
│  MultiHeadAttention (causal)      │  ◀── Lesson 03 + 07 (causal mask)
│  residual                         │
│  RMSNorm                          │
│  SwiGLU FFN                       │  ◀── Lesson 05
│  residual                         │
└────────────────────────────────── ┘
   │
   ▼
final RMSNorm
   │
   ▼
lm_head (tied to token embedding)
   │
   ▼
logits (B, N, V)
   │
   ▼
shift-by-one cross-entropy            ◀── Lesson 07
```

### 我们要交付的

- `GPTConfig` —— 在一处集中配置所有超参数。
- `MultiHeadAttention` —— 因果、批量化，可选 Flash 风格的计算路径（PyTorch 的 `scaled_dot_product_attention`）。
- `SwiGLUFFN` —— 现代 FFN。
- `Block` —— pre-norm、残差包裹的注意力 + FFN。
- `GPT` —— 嵌入层、堆叠的 block、LM head、generate()。
- 训练循环，含 AdamW、余弦学习率、梯度裁剪。
- 基于莎士比亚文本的字符级分词器。

### 我们不交付的

- RoPE —— 第 04 课已从概念上实现过。这里为简单起见使用可学习的位置嵌入。练习会要求你换成 RoPE。
- 生成阶段的 KV 缓存（KV cache）—— 每一步生成都会对完整前缀重新计算注意力。更慢，但更简单。练习会要求你加上 KV 缓存。
- Flash Attention —— PyTorch 2.0+ 在输入符合条件时会自动调度；我们使用 `F.scaled_dot_product_attention`。
- MoE —— 每个 block 只用单个 FFN。第 11 课你已经见过 MoE。

### 目标指标

在一台 Mac M2 笔记本上，一个 4 层、4 头、d_model=128 的 GPT，在 `tinyshakespeare.txt` 上训练 2,000 步：

- 训练损失约 6 分钟内从 ~4.2（随机初始化）收敛到 ~1.5。
- 采样输出看起来「像莎士比亚」：古体词、换行、诸如 "ROMEO:" 这样的专名开始出现。
- 验证损失（留出文本末尾 10%）与训练损失贴近；在这个规模/预算下没有过拟合。

## 从零实现

本课使用 PyTorch。安装 `torch`（CPU 版即可）。参见 `code/main.py`。脚本负责：

- 缺失时下载 `tinyshakespeare.txt`（或读取本地副本）。
- 字节级字符分词器。
- 按 90/10 划分训练/验证集。
- 在支持的硬件上使用 bf16 autocast 的训练循环。
- 训练完成后进行采样。

### 第 1 步：数据

```python
text = open("tinyshakespeare.txt").read()
chars = sorted(set(text))
stoi = {c: i for i, c in enumerate(chars)}
itos = {i: c for c, i in stoi.items()}
encode = lambda s: [stoi[c] for c in s]
decode = lambda xs: "".join(itos[x] for x in xs)
```

65 个不同字符。极小的词表。一个 4 字节的 vocab_size 就装得下。不需要 BPE，没有分词器的各种麻烦。

### 第 2 步：模型

参见 `code/main.py`。block 完全照搬第 05 课的标准写法——pre-norm、RMSNorm、SwiGLU、因果 MHA。4/4/128 配置下的参数量约 80 万。

### 第 3 步：训练循环

随机取一批长度为 256 的 token 窗口。前向计算。错位一格的交叉熵。反向传播。AdamW 更新。记录日志。重复。

```python
for step in range(max_steps):
    x, y = get_batch("train")
    logits = model(x)
    loss = F.cross_entropy(logits.view(-1, vocab_size), y.view(-1))
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()
    opt.zero_grad()
```

### 第 4 步：采样

给定一个提示词，反复前向计算，从 top-p 的 logits 中采样，把采到的 token 追加到序列后继续。生成 500 个 token 后停止。

### 第 5 步：阅读输出

训练 2,000 步之后：

```
ROMEO:
Away and mild will not thy friend, that thou shalt wit:
The chief that well shame and hath been his friends,
...
```

不是莎士比亚，但像莎士比亚。对于约 80 万参数、笔记本上 6 分钟的训练来说，这是明明白白的胜利。

## 生产实践

这个毕业项目是一份参考架构。想把它推向真正可用，有三个扩展方向：

1. **换分词器。** 使用 BPE（例如 `tiktoken.get_encoding("cl100k_base")`）。词表规模从 65 跃升到约 50,000。模型容量需要相应扩大来匹配。
2. **在更大的语料上训练。** 使用 `OpenWebText` 或 `fineweb-edu`（HuggingFace）。在单张 A100 上训 100 亿 token，一个 1.25 亿参数的 GPT 大约需要 24 小时。
3. **加上 RoPE + KV 缓存 + Flash Attention。** 下面的练习会带你逐一完成。

最终你会得到一个能生成流畅英文的 1.25 亿参数 GPT。不是前沿模型，但同一条代码路径——只是规模更大——正是 Karpathy、EleutherAI 和 Allen Institute 在 2026 年训练研究用 checkpoint 所走的路。

## 交付产物

参见 `outputs/skill-transformer-review.md`。该 skill 用于审查一个从零实现的 Transformer，对照前 13 课的全部内容检查其正确性。

## 练习

1. **简单。** 运行 `code/main.py`。确认训练后模型在最后一步的验证损失低于 2.0。把 `max_steps` 从 2,000 改为 5,000——验证损失还在继续下降吗？
2. **中等。** 把可学习的位置嵌入换成 RoPE。在 `MultiHeadAttention` 内部对 Q 和 K 施加旋转。训练并验证验证损失至少不变差。
3. **中等。** 在采样循环中实现 KV 缓存。分别在有缓存和无缓存的情况下生成 500 个 token。在笔记本上，实际耗时应该缩短 5–20 倍。
4. **困难。** 给模型加第二个 head，预测「下下个」token（MTP——来自 DeepSeek-V3 的 Multi-Token Prediction，多 token 预测）。联合训练。有帮助吗？
5. **困难。** 把每个 block 中的单个 FFN 换成 4 专家的 MoE。路由器 + top-2 路由。在激活参数量相同的条件下，看验证损失如何变化。

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|-----------------|-----------------------|
| nanoGPT | 「Karpathy 的教程仓库」 | 极简的 decoder-only Transformer 训练代码，约 300 行；公认的参考实现。 |
| tinyshakespeare | 「标准玩具语料」 | 约 1.1 MB 的文本；2015 年以来每个字符级语言模型教程都用它。 |
| 权重绑定嵌入（Tied embeddings） | 「共享输入/输出矩阵」 | LM head 的权重 = token 嵌入矩阵的转置；省参数，还能提升质量。 |
| bf16 autocast | 「训练精度技巧」 | 前向/反向用 bf16 计算，优化器状态保持 fp32；2021 年起的标准做法。 |
| 梯度裁剪（Gradient clipping） | 「防止尖峰」 | 把全局梯度范数上限设为 1.0；防止训练崩掉。 |
| 余弦学习率调度（Cosine LR schedule） | 「2020 年后的默认选择」 | 学习率先线性升高（warmup），再按余弦曲线衰减到峰值的 10%。 |
| MFU | 「Model FLOP Utilization（模型 FLOP 利用率）」 | 实际达到的 FLOPs / 理论峰值；2026 年稠密模型 40%、MoE 30% 已算很强。 |
| 验证损失（Val loss） | 「留出集损失」 | 在模型从未见过的数据上的交叉熵；过拟合探测器。 |

## 延伸阅读

- [The Annotated Transformer (Harvard NLP)](https://nlp.seas.harvard.edu/annotated-transformer/) —— 经典的逐行注解实现。
