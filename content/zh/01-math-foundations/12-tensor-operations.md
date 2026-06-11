# 张量运算

> 张量是数据与深度学习之间的通用语言。每一张图像、每一个句子、每一个梯度都在其中流动。

**Type:** Build
**Language:** Python
**Prerequisites:** Phase 1, Lessons 01 (Linear Algebra Intuition), 02 (Vectors, Matrices & Operations)
**Time:** ~90 minutes

## 学习目标

- 从零实现一个张量类，包含形状（shape）、步长（strides）、reshape、转置和逐元素运算
- 运用广播（broadcasting）规则，在不复制数据的前提下对不同形状的张量进行运算
- 编写 einsum 表达式，实现点积、矩阵乘法、外积和批量运算
- 逐步追踪多头注意力中每一步的精确张量形状

## 问题背景

你搭建了一个 Transformer，前向传播看起来很干净。一运行就报错：`RuntimeError: mat1 and mat2 shapes cannot be multiplied (32x768 and 512x768)`。你盯着这些形状，试了一次转置，结果变成 `Expected 4D input (got 3D input)`。你又加了一个 unsqueeze，别的地方又崩了。

形状错误是深度学习代码中最常见的 bug。它们在概念上并不难——每个运算都有自己的形状约定——但错误会快速叠加。一个 Transformer 里串联着几十个 reshape、转置和广播操作，只要一个轴搞错，错误就会层层传导。更糟的是，有些形状错误根本不会抛出异常：它们沿着错误的维度广播，或者在错误的轴上求和，悄无声息地产出垃圾结果。

矩阵处理的是两组对象之间的两两关系。真实数据装不进二维。一批 32 张 224x224 的 RGB 图像是一个 4D 张量：`(32, 3, 224, 224)`。12 个头的自注意力同样是 4D：`(batch, heads, seq_len, head_dim)`。你需要一种能推广到任意维数的数据结构，其运算能在所有维度上干净地组合。这个结构就是张量。掌握了张量运算，形状错误的调试就变得轻而易举。

## 核心概念

### 什么是张量

张量是一个数据类型统一的多维数组。维度的数量称为**秩（rank，又称 order）**。每一个维度是一个**轴（axis）**。**形状（shape）**是一个元组，列出每个轴上的大小。

```mermaid
graph LR
    S["Scalar<br/>rank 0<br/>shape: ()"] --> V["Vector<br/>rank 1<br/>shape: (3,)"]
    V --> M["Matrix<br/>rank 2<br/>shape: (2,3)"]
    M --> T3["3D Tensor<br/>rank 3<br/>shape: (2,2,2)"]
    T3 --> T4["4D Tensor<br/>rank 4<br/>shape: (B,C,H,W)"]
```

元素总数 = 各轴大小的乘积。形状为 `(2, 3, 4)` 的张量包含 `2 * 3 * 4 = 24` 个元素。

### 深度学习中的张量形状

不同类型的数据按照约定对应特定的张量形状。

```mermaid
graph TD
    subgraph Vision
        V1["(B, C, H, W)<br/>32, 3, 224, 224"]
    end
    subgraph NLP
        N1["(B, T, D)<br/>16, 128, 768"]
    end
    subgraph Attention
        A1["(B, H, T, D)<br/>16, 12, 128, 64"]
    end
    subgraph Weights
        W1["Linear: (out, in)<br/>Conv2D: (out_c, in_c, kH, kW)<br/>Embedding: (vocab, dim)"]
    end
```

PyTorch 使用 NCHW（通道在前）。TensorFlow 默认使用 NHWC（通道在后）。布局不匹配会导致悄无声息的性能下降或报错。

### 内存布局的工作原理

二维数组在内存中是一段一维的字节序列。**步长（strides）**告诉你沿每个轴前进一步需要跳过多少个元素。

```mermaid
graph LR
    subgraph "Row-major (C order)"
        R["a b c d e f<br/>strides: (3, 1)"]
    end
    subgraph "Column-major (F order)"
        C["a d b e c f<br/>strides: (1, 2)"]
    end
```

转置并不移动数据。它只是交换步长，使张量变为**非连续（non-contiguous）**——同一行的元素在内存中不再相邻。

### 广播规则

广播让你在不复制数据的情况下对不同形状的张量进行运算。形状从右向左对齐。两个维度兼容的条件是相等或其中一个为 1。维数较少的形状在左侧补 1。

```
Tensor A:     (8, 1, 6, 1)
Tensor B:        (7, 1, 5)
Padded B:     (1, 7, 1, 5)
Result:       (8, 7, 6, 5)
```

### Einsum：通用的张量运算

爱因斯坦求和（Einstein summation）给每个轴标上一个字母。只出现在输入中、不出现在输出中的轴会被求和；两边都出现的轴则被保留。

```mermaid
graph LR
    subgraph "matmul: ik,kj -> ij"
        A["A(I,K)"] --> |"sum over k"| C["C(I,J)"]
        B["B(K,J)"] --> |"sum over k"| C
    end
```

关键模式：`i,i->`（点积）、`i,j->ij`（外积）、`ii->`（迹）、`ij->ji`（转置）、`bij,bjk->bik`（批量矩阵乘法）、`bhtd,bhsd->bhts`（注意力分数）。

```figure
tensor-broadcast
```

## 从零实现

代码位于 `code/tensors.py`。每一步都对应其中的实现。

### 第 1 步：张量存储与步长

张量存储一个扁平的数字列表，外加形状元数据。步长告诉索引逻辑如何把多维索引映射到扁平位置。

```python
class Tensor:
    def __init__(self, data, shape=None):
        if isinstance(data, (list, tuple)):
            self._data, self._shape = self._flatten_nested(data)
        elif isinstance(data, np.ndarray):
            self._data = data.flatten().tolist()
            self._shape = tuple(data.shape)
        else:
            self._data = [data]
            self._shape = ()

        if shape is not None:
            total = reduce(lambda a, b: a * b, shape, 1)
            if total != len(self._data):
                raise ValueError(
                    f"Cannot reshape {len(self._data)} elements into shape {shape}"
                )
            self._shape = tuple(shape)

        self._strides = self._compute_strides(self._shape)

    @staticmethod
    def _compute_strides(shape):
        if len(shape) == 0:
            return ()
        strides = [1] * len(shape)
        for i in range(len(shape) - 2, -1, -1):
            strides[i] = strides[i + 1] * shape[i + 1]
        return tuple(strides)
```

对于形状 `(3, 4)`，步长是 `(4, 1)`——前进一行跳过 4 个元素，前进一列跳过 1 个元素。

### 第 2 步：Reshape、squeeze、unsqueeze

Reshape 在不改变元素顺序的前提下改变形状，元素总数必须保持不变。可以在某一个维度上用 `-1` 让其大小自动推断。

```python
t = Tensor(list(range(12)), shape=(2, 6))
r = t.reshape((3, 4))
r = t.reshape((-1, 3))
```

Squeeze 删除大小为 1 的轴，unsqueeze 插入一个这样的轴。Unsqueeze 对广播至关重要——把一个偏置向量 `(D,)` 加到一个批次 `(B, T, D)` 上，需要先 unsqueeze 成 `(1, 1, D)`。

```python
t = Tensor(list(range(6)), shape=(1, 3, 1, 2))
s = t.squeeze()
v = Tensor([1, 2, 3])
u = v.unsqueeze(0)
```

### 第 3 步：转置与 permute

转置交换两个轴，permute 重新排列所有轴。这就是 NCHW 和 NHWC 之间相互转换的方式。

```python
mat = Tensor(list(range(6)), shape=(2, 3))
tr = mat.transpose(0, 1)

t4d = Tensor(list(range(24)), shape=(1, 2, 3, 4))
perm = t4d.permute((0, 2, 3, 1))
```

转置或 permute 之后，张量在内存中变为非连续。在 PyTorch 中，`view` 在非连续张量上会失败——请改用 `reshape`，或先调用 `.contiguous()`。

### 第 4 步：逐元素运算与归约

逐元素运算（加、乘、减）独立作用于每个元素，并保持形状不变。归约（sum、mean、max）会折叠一个或多个轴。

```python
a = Tensor([[1, 2], [3, 4]])
b = Tensor([[10, 20], [30, 40]])
c = a + b
d = a * 2
s = a.sum(axis=0)
```

CNN 中的全局平均池化：`(B, C, H, W).mean(axis=[2, 3])` 得到 `(B, C)`。NLP 中的序列平均池化：`(B, T, D).mean(axis=1)` 得到 `(B, D)`。

### 第 5 步：用 NumPy 做广播

`tensors.py` 中的 `demo_broadcasting_numpy()` 函数展示了核心模式。

```python
activations = np.random.randn(4, 3)
bias = np.array([0.1, 0.2, 0.3])
result = activations + bias

images = np.random.randn(2, 3, 4, 4)
scale = np.array([0.5, 1.0, 1.5]).reshape(1, 3, 1, 1)
result = images * scale

a = np.array([1, 2, 3]).reshape(-1, 1)
b = np.array([10, 20, 30, 40]).reshape(1, -1)
outer = a * b
```

用广播计算两两距离：把 `(M, 2)` reshape 成 `(M, 1, 2)`，把 `(N, 2)` reshape 成 `(1, N, 2)`，相减、平方、沿最后一个轴求和、再开平方。结果是 `(M, N)`。

### 第 6 步：Einsum 运算

`demo_einsum()` 和 `demo_einsum_gallery()` 函数逐一演示所有常见模式。

```python
a = np.array([1.0, 2.0, 3.0])
b = np.array([4.0, 5.0, 6.0])
dot = np.einsum("i,i->", a, b)

A = np.array([[1, 2], [3, 4], [5, 6]], dtype=float)
B = np.array([[7, 8, 9], [10, 11, 12]], dtype=float)
matmul = np.einsum("ik,kj->ij", A, B)

batch_A = np.random.randn(4, 3, 5)
batch_B = np.random.randn(4, 5, 2)
batch_mm = np.einsum("bij,bjk->bik", batch_A, batch_B)
```

一次张量缩并（contraction）的计算成本等于所有索引大小（保留的和被求和的）的乘积。对 `bij,bjk->bik`，取 B=32、I=128、J=64、K=128：`32 * 128 * 64 * 128 = 33,554,432` 次乘加运算。

### 第 7 步：用 einsum 实现注意力机制

`demo_attention_einsum()` 函数端到端地实现了多头注意力。

```python
B, H, T, D = 2, 4, 8, 16
E = H * D

X = np.random.randn(B, T, E)
W_q = np.random.randn(E, E) * 0.02

Q = np.einsum("bte,ek->btk", X, W_q)
Q = Q.reshape(B, T, H, D).transpose(0, 2, 1, 3)

scores = np.einsum("bhtd,bhsd->bhts", Q, K) / np.sqrt(D)
weights = softmax(scores, axis=-1)
attn_output = np.einsum("bhts,bhsd->bhtd", weights, V)

concat = attn_output.transpose(0, 2, 1, 3).reshape(B, T, E)
output = np.einsum("bte,ek->btk", concat, W_o)
```

每一步都是张量运算：投影（用 einsum 做矩阵乘法）、分头（reshape + 转置）、注意力分数（用 einsum 做批量矩阵乘法）、加权求和（用 einsum 做批量矩阵乘法）、合并多头（转置 + reshape）、输出投影（用 einsum 做矩阵乘法）。

## 生产实践

### 从零实现 vs NumPy

| 运算 | 从零实现（Tensor 类） | NumPy |
|---|---|---|
| 创建 | `Tensor([[1,2],[3,4]])` | `np.array([[1,2],[3,4]])` |
| Reshape | `t.reshape((3,4))` | `a.reshape(3,4)` |
| 转置 | `t.transpose(0,1)` | `a.T` 或 `a.transpose(0,1)` |
| Squeeze | `t.squeeze(0)` | `np.squeeze(a, 0)` |
| 求和 | `t.sum(axis=0)` | `a.sum(axis=0)` |
| Einsum | 无 | `np.einsum("ij,jk->ik", a, b)` |

### 从零实现 vs PyTorch

```python
import torch

t = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.float32)
t.shape
t.stride()
t.is_contiguous()

t.reshape(3, 2)
t.unsqueeze(0)
t.transpose(0, 1)
t.transpose(0, 1).contiguous()

torch.einsum("ik,kj->ij", A, B)
```

PyTorch 增加了自动求导（autograd）、GPU 支持和优化过的 BLAS 内核，但形状语义完全相同。理解了从零实现的版本，PyTorch 的形状错误就能一眼看懂。

### 每一种神经网络层都是张量运算

| 运算 | 张量形式 | Einsum |
|---|---|---|
| 线性层 | `Y = X @ W.T + b` | `"bd,od->bo"` + 偏置 |
| 注意力 QKV | `Q = X @ W_q` | `"btd,dh->bth"` |
| 注意力分数 | `Q @ K.T / sqrt(d)` | `"bhtd,bhsd->bhts"` |
| 注意力输出 | `softmax(scores) @ V` | `"bhts,bhsd->bhtd"` |
| 批归一化 | `(X - mu) / sigma * gamma` | 逐元素 + 广播 |
| Softmax | `exp(x) / sum(exp(x))` | 逐元素 + 归约 |

## 交付产物

本课产出两个可复用的提示词：

1. **`outputs/prompt-tensor-shapes.md`**——一个系统化的张量形状不匹配调试提示词。包含每种常见运算（matmul、广播、cat、Linear、Conv2d、BatchNorm、softmax）的决策表和一张修复对照表。

2. **`outputs/prompt-tensor-debugger.md`**——一个分步调试提示词，当形状错误卡住你时，粘贴到任意 AI 助手即可使用。把错误信息和张量形状喂给它，就能拿到准确的修复方案。

## 练习

1. **简单——Reshape 往返。** 取一个形状为 `(2, 3, 4)` 的张量。把它 reshape 成 `(6, 4)`，再 reshape 成 `(24,)`，最后变回 `(2, 3, 4)`。每一步打印扁平数据，验证元素顺序始终保持不变。

2. **中等——实现广播。** 给 `Tensor` 类扩展一个 `broadcast_to(shape)` 方法，把大小为 1 的维度扩展到目标形状。然后修改 `_elementwise_op`，使其在运算前自动广播。用形状 `(3, 1)` 和 `(1, 4)` 测试，结果应为 `(3, 4)`。

3. **困难——从零构建 einsum。** 实现一个基础的 `einsum(subscripts, *tensors)` 函数，至少支持：点积（`i,i->`）、矩阵乘法（`ij,jk->ik`）、外积（`i,j->ij`）和转置（`ij->ji`）。解析下标字符串，识别被缩并的索引，并遍历所有索引组合。将你的结果与 `np.einsum` 进行对比。

4. **困难——注意力形状追踪器。** 编写一个函数，输入 `batch_size`、`seq_len`、`embed_dim` 和 `num_heads`，打印多头注意力每一步的精确形状：输入、Q/K/V 投影、分头、注意力分数、softmax 权重、加权求和、合并多头、输出投影。与 `demo_attention_einsum()` 的输出进行核对。

## 关键术语

| 术语 | 人们常说的 | 实际含义 |
|---|---|---|
| 张量（Tensor） | "维度更多的矩阵" | 数据类型统一、具有明确形状、步长和运算的多维数组 |
| 秩（Rank） | "维度的数量" | 轴的数量。矩阵的秩是 2，而不是它的矩阵秩 |
| 形状（Shape） | "张量的大小" | 列出每个轴上大小的元组。`(2, 3)` 表示 2 行 3 列 |
| 步长（Stride） | "内存怎么排布" | 沿每个轴前进一个位置需要跳过的元素数 |
| 广播（Broadcasting） | "形状不同也能直接算" | 一套严格的规则：从右对齐，维度必须相等或其中一个为 1 |
| 连续（Contiguous） | "正常的张量" | 元素按逻辑布局在内存中顺序存储，没有间隙或重排 |
| Einsum | "matmul 的花哨写法" | 一种通用记法，能用一行表达任意张量缩并、外积、迹或转置 |
| 视图（View） | "和 reshape 一样" | 共享同一块内存缓冲区、但形状/步长元数据不同的张量。在非连续数据上会失败 |
| 缩并（Contraction） | "对某个索引求和" | 一种通用运算：张量之间的共享索引相乘并求和，产生更低秩的结果 |
| NCHW / NHWC | "PyTorch 和 TensorFlow 的格式" | 图像张量的内存布局约定。NCHW 把通道放在空间维度之前，NHWC 放在之后 |

## 延伸阅读

- [NumPy Broadcasting](https://numpy.org/doc/stable/user/basics.broadcasting.html)——权威的广播规则与可视化示例
- [PyTorch Tensor Views](https://pytorch.org/docs/stable/tensor_view.html)——视图何时生效、何时会复制数据
- [einops](https://github.com/arogozhnikov/einops)——让张量重排既可读又安全的库
- [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/)——可视化注意力中流动的张量形状
- [Einstein Summation in NumPy](https://numpy.org/doc/stable/reference/generated/numpy.einsum.html)——完整的 einsum 文档与示例
