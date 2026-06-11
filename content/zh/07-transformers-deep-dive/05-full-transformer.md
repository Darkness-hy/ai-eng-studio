# 完整的 Transformer —— 编码器 + 解码器

> 注意力是主角。其余的一切——残差、归一化、前馈网络、交叉注意力——都是让你能把它堆深的脚手架。

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 7 · 02 (Self-Attention), Phase 7 · 03 (Multi-Head Attention), Phase 7 · 04 (Positional Encoding)
**Time:** ~75 minutes

## 问题背景

单个注意力层只是一个特征提取器，不是一个模型。每层一次矩阵乘法的容量对语言来说远远不够。你需要深度——而没有正确的"管道"，深度就会崩塌。

2017 年 Vaswani 的论文把六个设计决策打包在一起，把单个注意力层变成了可堆叠的模块（block）。此后的每一个 Transformer——仅编码器（BERT）、仅解码器（GPT）、编码器-解码器（T5）——都继承了同一副骨架。到了 2026 年，模块本身经过了改良（RMSNorm、SwiGLU、pre-norm、RoPE），但骨架完全相同。

这一课讲的就是这副骨架。后续课程会对它做专门化——06 讲编码器，07 讲解码器，08 讲编码器-解码器。

## 核心概念

![Encoder and decoder block internals, wired](../assets/full-transformer.svg)

### 六个组成部分

1. **嵌入 + 位置信号。** token → 向量。位置通过 RoPE（现代做法）或正弦编码（经典做法）注入。
2. **自注意力。** 每个位置关注所有其他位置。在解码器中带掩码。
3. **前馈网络（FFN）。** 逐位置的两层 MLP：`W_2 · activation(W_1 · x)`。默认扩展比为 4 倍。
4. **残差连接。** `x + sublayer(x)`。没有它，梯度在大约 6 层之后就会消失。
5. **层归一化。** `LayerNorm` 或 `RMSNorm`（现代做法）。稳定残差流。
6. **交叉注意力（仅解码器）。** 查询（Q）来自解码器，键（K）和值（V）来自编码器输出。

观察一个向量如何流过一个模块：注意力在各位置之间混合信息，残差把它向前传递，FFN 对它做变换，归一化让整个数据流保持稳定。

```figure
transformer-block
```

### 编码器模块（BERT、T5 编码器使用）

```
x → LN → MHA(self) → + → LN → FFN → + → out
                     ^              ^
                     |              |
                     └── residual ──┘
```

编码器是双向的。没有掩码。所有位置都能看到所有位置。

### 解码器模块（GPT、T5 解码器使用）

```
x → LN → MHA(masked self) → + → LN → MHA(cross to encoder) → + → LN → FFN → + → out
```

解码器每个模块有三个子层。中间那个——交叉注意力——是信息从编码器流向解码器的唯一通道。在纯解码器架构（GPT）中，交叉注意力被省略，只剩下带掩码的自注意力 + FFN。

### Pre-norm 与 post-norm

原论文的写法：`x + sublayer(LN(x))` 对比 `LN(x + sublayer(x))`。Post-norm 在 2019 年前后失宠——不做精细的学习率预热（warmup）就很难训练得深。Pre-norm（`LN` 放在子层*之前*）是 2026 年的默认选择：Llama、Qwen、GPT-3+、Mistral 都在用。

### 2026 年的现代化模块

Vaswani 2017 用的是 LayerNorm + ReLU。现代技术栈把两者都替换掉了。生产环境中的模块实际长这样：

| 组件 | 2017 | 2026 |
|-----------|------|------|
| 归一化 | LayerNorm | RMSNorm |
| FFN 激活函数 | ReLU | SwiGLU |
| FFN 扩展比 | 4 倍 | 2.6 倍（SwiGLU 用三个矩阵，总参数量持平） |
| 位置编码 | 正弦绝对位置 | RoPE |
| 注意力 | 完整 MHA | GQA（或 MLA） |
| 偏置项 | 有 | 无 |

RMSNorm 去掉了 LayerNorm 的均值中心化（少一次减法），既省计算，经验上也至少同样稳定。SwiGLU（`Swish(W1 x) ⊙ W3 x`）在 Llama、PaLM 和 Qwen 的论文中稳定地比 ReLU/GELU FFN 好约 0.5 个困惑度（ppl）点。

### 参数量计算

对于一个 `d_model = d`、FFN 扩展比为 `r` 的模块：

- MHA：`4 · d²`（Q、K、V、O 投影）
- FFN（SwiGLU）：`3 · d · (r · d)` ≈ `3rd²`
- 归一化层：可忽略

在 `d = 4096, r = 2.6, layers = 32`（大致对应 Llama 3 8B）时，总计：`32 · (4·4096² + 3·2.6·4096²) ≈ 32 · (16 + 32) M = ~1.5B parameters per layer × 32 ≈ 7B`（外加嵌入和输出头）。与公开发布的参数量吻合。

## 从零实现

### 第 1 步：基础构件

使用第 03 课的微型 `Matrix` 类（为保持独立性已复制到本文件中）：

- `layer_norm(x, eps=1e-5)` —— 减去均值，除以标准差。
- `rms_norm(x, eps=1e-6)` —— 除以 RMS。不减均值。
- `gelu(x)` 和 `silu(x) * W3 x`（SwiGLU）。
- `ffn_swiglu(x, W1, W2, W3)`。
- `encoder_block(x, params)` 和 `decoder_block(x, enc_out, params)`。

完整的接线见 `code/main.py`。

### 第 2 步：搭建一个 2 层编码器和一个 2 层解码器

把它们堆起来。把编码器输出传入每一层解码器的交叉注意力。在输出投影之前加一个最终的 LN。

```python
def encode(tokens, params):
    x = embed(tokens, params.emb) + sinusoidal(len(tokens), params.d)
    for block in params.encoder_blocks:
        x = encoder_block(x, block)
    return x

def decode(target_tokens, encoder_out, params):
    x = embed(target_tokens, params.emb) + sinusoidal(len(target_tokens), params.d)
    for block in params.decoder_blocks:
        x = decoder_block(x, encoder_out, block)
    return x
```

### 第 3 步：在玩具示例上跑一次前向传播

输入一个 6 个 token 的源序列和一个 5 个 token 的目标序列。验证输出形状为 `(5, vocab)`。不做训练——这一课讲的是架构，不是损失函数。

### 第 4 步：换上 RMSNorm + SwiGLU

把 LayerNorm 和 ReLU-FFN 替换成 RMSNorm 和 SwiGLU。确认形状仍然匹配。只需替换一个函数，就完成了 2026 年的现代化改造。

## 生产实践

PyTorch/TF 的参考实现是 `nn.TransformerEncoderLayer` 和 `nn.TransformerDecoderLayer`。但 2026 年的大多数生产代码都自己实现模块，原因是：

- Flash Attention 是在注意力内部直接调用的，而不是通过 `nn.MultiheadAttention`。
- GQA / MLA 不在标准库参考实现里。
- RoPE、RMSNorm、SwiGLU 都不是 PyTorch 的默认选项。

HF `transformers` 里有值得一读的干净参考实现：`modeling_llama.py` 是 2026 年仅解码器模块的标准范本。大约 500 行，值得完整走读一遍。

**编码器、解码器、编码器-解码器——如何选择：**

| 需求 | 选择 | 示例 |
|------|------|---------|
| 分类、嵌入、文本问答 | 仅编码器 | BERT, DeBERTa, ModernBERT |
| 文本生成、对话、代码、推理 | 仅解码器 | GPT, Llama, Claude, Qwen |
| 结构化输入 → 结构化输出（翻译、摘要） | 编码器-解码器 | T5, BART, Whisper |

仅解码器架构赢得了语言领域，因为它扩展最干净，并且同时胜任理解和生成。当输入有明确的"源序列"身份时（翻译、语音识别、结构化任务），编码器-解码器仍然是最佳选择。

## 交付产物

见 `outputs/skill-transformer-block-reviewer.md`。该技能依据 2026 年的默认标准审查一个新的 Transformer 模块实现，并标记缺失的部分（pre-norm、RoPE、RMSNorm、GQA、FFN 扩展比）。

## 练习

1. **简单。** 计算你的 encoder_block 在 `d_model=512, n_heads=8, ffn_expansion=4, swiglu=True` 下的参数量。通过实现该模块并使用 `sum(p.numel() for p in block.parameters())` 来验证。
2. **中等。** 从 post-norm 切换到 pre-norm。两种都初始化，在随机输入上测量堆叠 12 层之后的激活范数。Post-norm 的激活应该会爆炸；pre-norm 的应该保持有界。
3. **困难。** 在一个玩具复制任务（把 `x` 反转后复制）上实现一个 4 层编码器-解码器。训练 100 步。报告损失。换上 RMSNorm + SwiGLU + RoPE——损失会下降吗？

## 关键术语

| 术语 | 大家怎么说 | 实际含义 |
|------|-----------------|-----------------------|
| Block（模块） | "一个 Transformer 层" | 归一化 + 注意力 + 归一化 + FFN 的堆叠，外面包着残差连接。 |
| 残差 | "跳跃连接" | 输出为 `x + f(x)`；让梯度能流过很深的堆叠。 |
| Pre-norm | "先归一化，而不是后归一化" | 现代写法：`x + sublayer(LN(x))`。不需要繁琐的 warmup 技巧就能训练更深。 |
| RMSNorm | "去掉均值的 LayerNorm" | 除以 RMS；少一个操作，经验上同样稳定。 |
| SwiGLU | "大家都换上的那个 FFN" | `Swish(W1 x) ⊙ W3 x → W2`。在语言模型困惑度上优于 ReLU/GELU。 |
| 交叉注意力 | "解码器看到编码器的方式" | Q 来自解码器、K/V 来自编码器输出的 MHA。 |
| FFN 扩展比 | "中间那个 MLP 有多宽" | 隐藏层大小与 d_model 的比值，通常为 4（LayerNorm）或 2.6（SwiGLU）。 |
| 无偏置 | "去掉 +b 项" | 现代技术栈在线性层中省略偏置；困惑度略有提升，模型更小。 |

## 延伸阅读

- [Vaswani et al. (2017). Attention Is All You Need](https://arxiv.org/abs/1706.03762) —— 原始模块规格。
- [Xiong et al. (2020). On Layer Normalization in the Transformer Architecture](https://arxiv.org/abs/2002.04745) —— 为什么在深层网络中 pre-norm 优于 post-norm。
- [Zhang, Sennrich (2019). Root Mean Square Layer Normalization](https://arxiv.org/abs/1910.07467) —— RMSNorm。
- [Shazeer (2020). GLU Variants Improve Transformer](https://arxiv.org/abs/2002.05202) —— SwiGLU 论文。
- [HuggingFace `modeling_llama.py`](https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py) —— 2026 年仅解码器模块的标准范本。
