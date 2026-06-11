# 为什么是 Transformer —— RNN 的问题所在

> RNN 一次只处理一个 token，Transformer 一次处理所有 token。这一个架构上的押注，改写了 2017 年之后深度学习的每一条扩展曲线。

**Type:** Learn
**Languages:** Python
**Prerequisites:** Phase 3 (Deep Learning Core), Phase 5 · 09 (Sequence-to-Sequence), Phase 5 · 10 (Attention Mechanism)
**Time:** ~45 minutes

## 问题背景

2017 年之前，全世界所有最先进的序列模型——无论语言、翻译还是语音——都是循环神经网络（RNN）。LSTM 和 GRU 在翻译领域称霸了相当于 ImageNet 级别的基准测试长达五年。当时人们手里只有这一种工具。

它们有三个致命弱点。首先是顺序计算：你无法沿时间轴并行化，token `t+1` 需要 token `t` 的隐藏状态。一条 1,024 个 token 的序列意味着 1,024 个串行步骤，而 GPU 每个周期能做 1,000,000 次浮点运算。在为并行而生的硬件上，训练的实际耗时却随序列长度线性增长。

其次是梯度消失：50 个 token 之前的信息，已经被 50 层非线性变换压扁了。门控循环单元（LSTM、GRU）缓解了这种挤压，但从未根除它。长距离依赖——"我去年夏天在飞往京都的飞机上读的那本书是……"——经常失效。

最后是固定宽度的隐藏状态：编码器必须把整个源序列压进一个向量，解码器才能开始工作。源序列是 5 个 token 还是 500 个 token 都无所谓，瓶颈的形状不变。

2017 年的论文 "Attention Is All You Need" 提出了一个激进的方案：彻底抛弃循环结构。让每个位置并行地关注所有其他位置。用一次大矩阵乘法完成训练，而不是 1,024 次串行计算。

结果是，到 2026 年 Transformer 统治了所有模态。语言（GPT-5、Claude 4、Llama 4）、视觉（ViT、DINOv2、SAM 3）、音频（Whisper）、生物（AlphaFold 3）、机器人（RT-2）。同一个模块，不同的输入。

## 核心概念

![RNN sequential compute vs Transformer parallel attention](../assets/rnn-vs-transformer.svg)

**循环结构是瓶颈。** RNN 的计算是 `h_t = f(h_{t-1}, x_t)`。每一步都依赖前一步。算不出 `h_4` 就算不出 `h_5`。在拥有 10,000 个以上并行核心的现代 GPU 上，处理长序列时 99% 的芯片算力都被浪费了。

**注意力是广播。** 自注意力（self-attention）对每一对 `(i, j)` 同时计算 `output_i = sum_j(a_ij * v_j)`。整个 N×N 注意力矩阵在一次批量矩阵乘法中填满。任何一步都不依赖另一步。GPU 最喜欢这种计算。

**这种加速不是常数倍。** 它是 `O(N)` 串行深度与 `O(1)` 串行深度的区别。实践中，在相同硬件、N=512 的条件下，Transformer 每个 epoch 的训练速度快 5–10 倍，而且差距随序列长度继续拉大，直到撞上注意力的 `O(N²)` 内存墙（后来 Flash Attention 解决了这个问题——见第 12 课）。

**Transformer 的代价。** 注意力的内存开销按 `O(N²)` 增长。2K 上下文没问题，但到了 128K 上下文，就需要滑动窗口、RoPE 外推、Flash Attention 分块或线性注意力变体。循环结构在时间和内存上都是 `O(N)`；Transformer 用内存换时间，再通过并行把时间赢回来。

**归纳偏置（inductive bias）的转变。** RNN 假设局部性和近因性，Transformer 什么都不假设——任意一对位置都是注意力的候选。这就是为什么 Transformer 需要更多数据才能训练好，但一旦数据充足就能扩展得更远。Chinchilla（2022）把这一点形式化了：只要 token 足够多，相同参数量的 Transformer 总能胜过 RNN。

## 从零实现

这里不涉及神经网络——我们用数值方式模拟核心瓶颈，让你在自己的笔记本电脑上感受这个差距。

### 第一步：测量串行深度

见 `code/main.py`。我们写两个函数。一个把序列编码为一连串加法（串行，像 RNN）；另一个编码为并行归约（广播，像注意力）。数学相同，依赖图不同。

```python
def rnn_style(xs):
    h = 0.0
    for x in xs:
        h = 0.9 * h + x   # can't parallelize: h depends on previous h
    return h

def attention_style(xs):
    return sum(xs) / len(xs)  # every x is independent
```

我们在长度最高 100,000 个元素的序列上对两者计时。RNN 版本是 O(N) 的，只能占用单条 CPU 流水线。即使在纯 Python 里，注意力风格的归约在长度 ≥ 1,000 时也会胜出，因为 Python 的 `sum()` 是 C 实现的，每一步迭代没有解释器开销。

### 第二步：统计理论运算量

两个算法都做 N 次加法。区别在于*依赖深度*：有多少运算必须顺序完成，下一个才能开始。RNN 的深度 = N。注意力的深度用树形归约是 log(N)，用并行扫描（parallel scan）则是 1。决定 GPU 耗时的是深度，不是运算次数。

### 第三步：长序列上的实证扩展性

我们打印一张计时表，把 O(N) 的差距直观呈现出来。在一台 2026 年的 Mac 笔记本上，1,000 个元素以内的序列快到无法测量；100,000 个元素的序列则呈现出干净的线性扫描曲线。把它放大到一个 16,384 token 的 Transformer 对比等价的 12 层 LSTM，你就明白为什么训练耗时在 2016 年是个拦路虎。

## 生产实践

2026 年仍然该选 RNN 的场景：

| 场景 | 选择 |
|-----------|------|
| 流式推理、一次一个 token、常数内存 | RNN 或状态空间模型（Mamba、RWKV） |
| 极长序列（>1M token），注意力内存爆炸 | 线性注意力、Mamba 2、Hyena |
| 没有矩阵乘法加速器的边缘设备 | 深度可分离 RNN 在 FLOPs/瓦特上仍然占优 |
| 其他一切场景（训练、批量推理、128K 以内的上下文） | Transformer |

状态空间模型（State-space model, SSM），比如 Mamba，本质上是带结构化参数化的 RNN，兼得两家之长：`O(N)` 的扫描内存，并通过选择性扫描（selective scan）实现并行训练。它们能达到 Transformer 90% 的质量，且长上下文扩展性更好。2026 年大多数前沿实验室训练的是 SSM 与 Transformer 的混合模型（如 Jamba、Samba）——循环结构没有死，它变成了一个组件。

## 交付产物

见 `outputs/skill-architecture-picker.md`。这个 skill 根据序列长度、吞吐量和训练预算约束，为新的序列问题挑选架构。对于训练数据超过 10 亿 token 的训练任务，它必须拒绝推荐纯 RNN，除非明确说明其中的权衡。

## 练习

1. **简单。** 把 `code/main.py` 里的 `rnn_style` 的标量隐藏状态换成长度为 64 的隐藏状态向量，重新计时。串行开销随隐藏状态维度增长了多少？
2. **中等。** 用纯 Python 实现并行前缀和（Hillis-Steele 扫描）。验证它在长度 1024 上与串行扫描的数值输出一致。数一数它的深度。
3. **困难。** 把注意力风格的归约移植到 GPU 上的 PyTorch。把序列长度从 64 扫到 65,536，对两者计时。画出曲线并解释其形状。

## 关键术语

| 术语 | 大家怎么说 | 实际含义 |
|------|-----------------|-----------------------|
| 循环（Recurrence） | "RNN 是顺序的" | 第 `t` 步依赖第 `t-1` 步的计算，迫使沿时间轴串行执行。 |
| 串行深度（Serial depth） | "计算图有多深" | 最长的依赖运算链；即使硬件无限，它也决定实际耗时的下限。 |
| 注意力（Attention） | "让 token 互相看见" | 加权和 `sum_j a_ij v_j`，其中 `a_ij` 来自位置 i 和 j 之间的相似度分数。 |
| 上下文窗口（Context window） | "模型能看到多少" | 注意力层能接收的位置数量；二次方的内存开销在此处累积。 |
| 归纳偏置（Inductive bias） | "刻进架构里的假设" | 对数据形态的先验；CNN 假设平移不变性，RNN 假设近因性。 |
| 状态空间模型（State-space model） | "有代数支撑的 RNN" | 通过结构化状态空间矩阵参数化的循环结构，可实现并行训练。 |
| 二次方瓶颈（Quadratic bottleneck） | "为什么上下文这么贵" | 注意力内存 = 序列长度的 `O(N²)`；Flash Attention 隐藏的是常数，不是增长阶。 |

## 延伸阅读

- [Vaswani et al. (2017). Attention Is All You Need](https://arxiv.org/abs/1706.03762) —— 在主流 NLP 中终结循环结构的那篇论文。
- [Bahdanau, Cho, Bengio (2014). Neural MT by Jointly Learning to Align and Translate](https://arxiv.org/abs/1409.0473) —— 注意力的诞生之地，当时还附着在 RNN 上。
- [Hochreiter, Schmidhuber (1997). Long Short-Term Memory](https://www.bioinf.jku.at/publications/older/2604.pdf) —— LSTM 的原始论文，立此存照。
- [Gu, Dao (2023). Mamba: Linear-Time Sequence Modeling with Selective State Spaces](https://arxiv.org/abs/2312.00752) —— 对 Transformer 的现代循环式回应。
