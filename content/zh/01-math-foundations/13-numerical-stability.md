# 数值稳定性

> 浮点数是一个会泄漏的抽象。它会在训练时咬你一口，而你完全看不到它的到来。

**Type:** Build
**Language:** Python
**Prerequisites:** Phase 1, Lessons 01-04
**Time:** ~120 minutes

## 学习目标

- 使用减最大值技巧实现数值稳定的 softmax 和 log-sum-exp
- 识别浮点计算中的上溢、下溢和灾难性抵消
- 使用中心差分法对照数值梯度验证解析梯度
- 解释为什么训练时 bfloat16 比 float16 更受青睐，以及损失缩放如何防止梯度下溢

## 问题背景

你的模型训练了三个小时，然后损失变成了 NaN。你加了一个 print 语句。第 9,000 步时 logits 还正常，第 9,001 步就变成了 `inf`，到第 9,002 步所有梯度都是 `nan`，训练彻底报废。

或者：你的模型顺利训练完成，但准确率比论文宣称的低 2%。你检查了一切：架构一致，超参数一致，数据一致。问题在于论文用的是 float32，而你用了 float16 却没有做正确的缩放。三十二位累积的舍入误差悄无声息地吃掉了你的准确率。

又或者：你从零实现交叉熵损失。在小 logits 上它运行良好，可一旦 logits 超过 100，它就返回 `inf`。softmax 上溢了，因为 `exp(100)` 超出了 float32 能表示的范围。每个 ML 框架都用一个两行代码的技巧处理这个问题，而你根本不知道这个技巧的存在。

数值稳定性不是理论问题。它是训练成功与悄然失败之间的分水岭。你将来调试的每一个严重的 ML bug，最终都会归结到浮点数上。

## 核心概念

### IEEE 754：计算机如何存储实数

计算机按照 IEEE 754 标准以浮点值的形式存储实数。一个浮点数由三部分组成：符号位、指数和尾数（significand）。

```
Float32 layout (32 bits total):
[1 sign] [8 exponent] [23 mantissa]

Value = (-1)^sign * 2^(exponent - 127) * 1.mantissa
```

尾数决定精度（有多少位有效数字），指数决定范围（数值能有多大或多小）。

```
Format     Bits   Exponent  Mantissa  Decimal digits  Range (approx)
float64    64     11        52        ~15-16          +/- 1.8e308
float32    32     8         23        ~7-8            +/- 3.4e38
float16    16     5         10        ~3-4            +/- 65,504
bfloat16   16     8         7         ~2-3            +/- 3.4e38
```

float32 大约有 7 位十进制有效数字。这意味着它能区分 1.0000001 和 1.0000002，但区分不了 1.00000001 和 1.00000002。超过 7 位之后，一切都是舍入噪声。

float16 大约只有 3 位有效数字。它能表示的最大数是 65,504。对于 logits、梯度和激活值动辄超过这个数的 ML 来说，这个上限小得令人不安。

bfloat16 是 Google 针对 float16 范围问题给出的答案。它拥有和 float32 一样的 8 位指数（范围相同，最高可达 3.4e38），但只有 7 位尾数（精度低于 float16）。训练神经网络时，范围比精度更重要，所以 bfloat16 通常胜出。

### 为什么 0.1 + 0.2 != 0.3

0.1 这个数无法在二进制浮点中被精确表示。在二进制下，它是一个无限循环小数：

```
0.1 in binary = 0.0001100110011001100110011... (repeating forever)
```

Float32 将其截断为 23 位尾数。存储的值约为 0.100000001490116。类似地，0.2 被存储为约 0.200000002980232。两者之和是 0.300000004470348，而不是 0.3。

```
In Python:
>>> 0.1 + 0.2
0.30000000000000004

>>> 0.1 + 0.2 == 0.3
False
```

这对 ML 很重要，因为：

1. 像 `if loss < threshold` 这样的损失比较可能给出错误答案
2. 累加大量小数值（数千步的梯度更新）会逐渐偏离真实总和
3. 如果用 `==` 比较浮点数，校验和与可复现性测试都会失败

解决办法：永远不要用 `==` 比较浮点数。改用 `abs(a - b) < epsilon` 或 `math.isclose()`。

### 灾难性抵消（Catastrophic Cancellation）

当你对两个几乎相等的浮点数做减法时，有效数字相互抵消，剩下的只是被提升为高位的舍入噪声。

```
a = 1.0000001    (stored as 1.00000011920929 in float32)
b = 1.0000000    (stored as 1.00000000000000 in float32)

True difference:  0.0000001
Computed:         0.00000011920929

Relative error: 19.2%
```

一次减法就产生了 19% 的相对误差。在 ML 中，以下场景都会出现这种情况：

- 对均值很大的数据计算方差：当 E[x] 很大时用 `E[x^2] - E[x]^2`
- 对几乎相等的对数概率做减法
- 用过小的 epsilon 计算有限差分梯度

解决办法：重新排列公式，避免对两个很大且几乎相等的数做减法。对于方差，使用 Welford 算法或先对数据做中心化。对于对数概率，全程在对数空间中计算。

### 上溢与下溢

上溢（overflow）发生在结果太大无法表示时。下溢（underflow）发生在结果太小时（比最小可表示的正数更接近零）。

```
Float32 boundaries:
  Maximum:  3.4028235e+38
  Minimum positive (normal): 1.175e-38
  Minimum positive (denorm): 1.401e-45
  Overflow:  anything > 3.4e38 becomes inf
  Underflow: anything < 1.4e-45 becomes 0.0
```

`exp()` 函数是 ML 中上溢的头号来源：

```
exp(88.7)  = 3.40e+38   (barely fits in float32)
exp(89.0)  = inf         (overflow)
exp(-87.3) = 1.18e-38   (barely above underflow)
exp(-104)  = 0.0         (underflow to zero)
```

`log()` 函数则在另一个方向出问题：

```
log(0.0)   = -inf
log(-1.0)  = nan
log(1e-45) = -103.3      (fine)
log(1e-46) = -inf        (input underflowed to 0, then log(0) = -inf)
```

在 ML 中，`exp()` 出现在 softmax、sigmoid 和概率计算中；`log()` 出现在交叉熵、对数似然和 KL 散度中。如果没有正确的技巧，`log(exp(x))` 这种组合就是雷区。

### Log-Sum-Exp 技巧

直接计算 `log(sum(exp(x_i)))` 在数值上很危险。如果任何一个 `x_i` 很大，`exp(x_i)` 就会上溢。如果所有 `x_i` 都非常小（很大的负数），每个 `exp(x_i)` 都会下溢为零，而 `log(0)` 是 `-inf`。

技巧是：在取指数之前先减去最大值。

```
log(sum(exp(x_i))) = max(x) + log(sum(exp(x_i - max(x))))
```

为什么有效：减去 `max(x)` 之后，最大的指数项是 `exp(0) = 1`，不可能上溢。求和中至少有一项是 1，所以总和至少为 1，而 `log(1) = 0`，也不可能下溢成 `-inf`。

证明：

```
log(sum(exp(x_i)))
= log(sum(exp(x_i - c + c)))                    (add and subtract c)
= log(sum(exp(x_i - c) * exp(c)))               (exp(a+b) = exp(a)*exp(b))
= log(exp(c) * sum(exp(x_i - c)))               (factor out exp(c))
= c + log(sum(exp(x_i - c)))                    (log(a*b) = log(a) + log(b))
```

令 `c = max(x)`，上溢就被消除了。

这个技巧在 ML 中随处可见：
- Softmax 归一化
- 交叉熵损失计算
- 序列模型中的对数概率求和
- 高斯混合模型
- 变分推断

### 为什么 Softmax 需要减最大值技巧

Softmax 将 logits 转换为概率：

```
softmax(x_i) = exp(x_i) / sum(exp(x_j))
```

不用这个技巧时，[100, 101, 102] 这样的 logits 会导致上溢：

```
exp(100) = 2.69e43
exp(101) = 7.31e43
exp(102) = 1.99e44
sum      = 2.99e44

These overflow float32 (max ~3.4e38)? No, 2.69e43 < 3.4e38? Actually:
exp(88.7) is already at the float32 limit.
exp(100) = inf in float32.
```

使用技巧后，减去 max(x) = 102：

```
exp(100 - 102) = exp(-2) = 0.135
exp(101 - 102) = exp(-1) = 0.368
exp(102 - 102) = exp(0)  = 1.000
sum = 1.503

softmax = [0.090, 0.245, 0.665]
```

得到的概率完全相同，但计算是安全的。这不是一种优化，而是正确性的必要条件。

### NaN 与 Inf：检测与预防

`nan`（Not a Number）和 `inf`（无穷大）会像病毒一样在计算中传播。梯度更新中出现一个 `nan` 就会让权重变成 `nan`，进而让之后的所有输出都变成 `nan`。一步之内训练就死了。

`inf` 的来源：
- 对很大的正数取 `exp()`
- 除以零：`1.0 / 0.0`
- 累加过程中的 `float32` 上溢

`nan` 的来源：
- `0.0 / 0.0`
- `inf - inf`
- `inf * 0`
- 对负数取 `sqrt()`
- 对负数取 `log()`
- 任何涉及已有 `nan` 的算术运算

检测：

```python
import math

math.isnan(x)       # True if x is nan
math.isinf(x)       # True if x is +inf or -inf
math.isfinite(x)    # True if x is neither nan nor inf
```

预防策略：

1. 对 `exp()` 的输入做截断：`exp(clamp(x, -80, 80))`
2. 在分母上加 epsilon：`x / (y + 1e-8)`
3. 在 `log()` 内部加 epsilon：`log(x + 1e-8)`
4. 使用稳定实现（log-sum-exp、稳定 softmax）
5. 用梯度裁剪防止权重爆炸
6. 调试期间在每次前向传播后检查 `nan`/`inf`

### 数值梯度检验

解析梯度（来自反向传播）可能有 bug。数值梯度检验通过有限差分计算梯度来验证它们。

中心差分公式：

```
df/dx ~= (f(x + h) - f(x - h)) / (2h)
```

它的精度是 O(h^2)，远好于前向差分 `(f(x+h) - f(x)) / h` 的 O(h)。

如何选择 h：太大则近似不准，太小则灾难性抵消会毁掉结果。典型取值是 `h = 1e-5` 到 `1e-7`。

检验方法：计算解析梯度与数值梯度之间的相对差异。

```
relative_error = |grad_analytical - grad_numerical| / max(|grad_analytical|, |grad_numerical|, 1e-8)
```

经验法则：
- relative_error < 1e-7：完美，梯度正确
- relative_error < 1e-5：可接受，大概率正确
- relative_error > 1e-3：有问题
- relative_error > 1：梯度完全错误

实现新的层或损失函数时，一定要做梯度检验。PyTorch 为此提供了 `torch.autograd.gradcheck()`。

### 混合精度训练

现代 GPU 拥有专用硬件（Tensor Core），计算 float16 矩阵乘法的速度比 float32 快 2-8 倍。混合精度（mixed precision）训练就是利用这一点：

```
1. Maintain float32 master copy of weights
2. Forward pass in float16 (fast)
3. Compute loss in float32 (prevents overflow)
4. Backward pass in float16 (fast)
5. Scale gradients to float32
6. Update float32 master weights
```

纯 float16 训练的问题是：梯度通常非常小（1e-8 或更小）。Float16 会把低于约 6e-8 的值全部下溢为零。所有梯度更新都成了零，模型也就停止了学习。

解决办法是损失缩放（loss scaling）：

```
1. Multiply loss by a large scale factor (e.g., 1024)
2. Backward pass computes gradients of (loss * 1024)
3. All gradients are 1024x larger (pushed above float16 underflow)
4. Divide gradients by 1024 before updating weights
5. Net effect: same update, but no underflow
```

动态损失缩放会自动调整缩放因子：从一个大值（65536）开始，如果梯度上溢为 `inf` 就减半；如果连续 N 步没有上溢就翻倍。

### bfloat16 vs float16：为什么训练时 bfloat16 胜出

```
float16:   [1 sign] [5 exponent]  [10 mantissa]
bfloat16:  [1 sign] [8 exponent]  [7 mantissa]
```

float16 精度更高（10 位尾数 vs 7 位），但范围有限（最大约 65,504）。bfloat16 精度较低，但范围与 float32 相同（最大约 3.4e38）。

对于训练神经网络：

- 训练中出现尖峰时，激活值和 logits 经常超过 65,504。float16 会上溢，而 bfloat16 能扛住。
- float16 必须配合损失缩放，而 bfloat16 通常不需要，因为它的范围覆盖了梯度量级的整个谱系。
- bfloat16 就是 float32 的简单截断：去掉尾数的低 16 位。转换非常简单，且指数部分无损。

float16 更适合推理，因为推理时数值有界，精度更重要。bfloat16 更适合训练，因为训练时范围更重要。这就是 TPU 和现代 NVIDIA GPU（A100、H100）原生支持 bfloat16 的原因。

### 梯度裁剪

当梯度在多层之间呈指数增长时（在 RNN、深层网络和 Transformer 中很常见），就会发生梯度爆炸。一个巨大的梯度可以在一步之内毁掉所有权重。

两种裁剪方式：

**按值裁剪：** 对每个梯度元素独立截断。

```
grad = clamp(grad, -max_val, max_val)
```

简单，但可能改变梯度向量的方向。

**按范数裁剪：** 缩放整个梯度向量，使其范数不超过阈值。

```
if ||grad|| > max_norm:
    grad = grad * (max_norm / ||grad||)
```

保留梯度的方向。`torch.nn.utils.clip_grad_norm_()` 做的就是这件事，这也是标准选择。

典型取值：Transformer 用 `max_norm=1.0`，RL 用 `max_norm=0.5`，较简单的网络用 `max_norm=5.0`。

梯度裁剪不是临时补丁，而是一种安全机制。没有它，一个异常 batch 产生的巨大梯度就足以毁掉数周的训练成果。

### 归一化层是数值稳定器

批归一化、层归一化和 RMS 归一化通常被介绍为帮助训练收敛的正则化手段。但它们同时也是数值稳定器。

没有归一化时，激活值可能在层与层之间指数式增长或缩小：

```
Layer 1: values in [0, 1]
Layer 5: values in [0, 100]
Layer 10: values in [0, 10,000]
Layer 50: values in [0, inf]
```

归一化在每一层都对激活值重新居中并重新缩放：

```
LayerNorm(x) = (x - mean(x)) / (std(x) + epsilon) * gamma + beta
```

其中的 `epsilon`（通常为 1e-5）防止所有激活值相同时出现除零。可学习参数 `gamma` 和 `beta` 让网络能够恢复它需要的任何尺度。

这使得数值在整个网络中始终处于数值安全的范围内，既防止前向传播中的上溢，也防止反向传播中的梯度爆炸。

### 常见的 ML 数值 Bug

**Bug：训练几个 epoch 后损失变成 NaN。**
原因：logits 变得太大，softmax 上溢；或者学习率太高，权重发散。
修复：使用稳定 softmax（减最大值）、降低学习率、加梯度裁剪。

**Bug：损失卡在 log(num_classes) 不动。**
原因：模型输出接近均匀分布的概率。通常意味着梯度消失，或者模型完全没在学习。
修复：检查数据标签是否正确、验证损失函数、检查死 ReLU。

**Bug：验证准确率比预期低 1-3%。**
原因：混合精度训练没有配合正确的损失缩放。梯度下溢悄悄把小的更新清零了。
修复：启用动态损失缩放，或改用 bfloat16。

**Bug：某些层的梯度范数为 0.0。**
原因：死 ReLU 神经元（所有输入都为负），或 float16 下溢。
修复：使用 LeakyReLU 或 GELU、使用梯度缩放、检查权重初始化。

**Bug：模型在一块 GPU 上正常，在另一块上结果不同。**
原因：浮点累加顺序的非确定性。GPU 并行归约在不同硬件上的求和顺序不同，而浮点加法不满足结合律。
修复：接受微小差异（1e-6），或设置 `torch.use_deterministic_algorithms(True)` 并接受速度损失。

**Bug：损失计算中 `exp()` 返回 `inf`。**
原因：原始 logits 没有经过减最大值技巧就直接传给了 `exp()`。
修复：使用 `torch.nn.functional.log_softmax()`，它内部实现了 log-sum-exp。

**Bug：从 float32 切换到 float16 后训练发散。**
原因：float16 无法表示低于 6e-8 的梯度量级，也无法表示高于 65,504 的激活值。
修复：使用带损失缩放的混合精度（AMP），或改用 bfloat16。

```figure
logsumexp-stability
```

## 从零实现

### 第 1 步：演示浮点精度极限

```python
print("=== Floating Point Precision ===")
print(f"0.1 + 0.2 = {0.1 + 0.2}")
print(f"0.1 + 0.2 == 0.3? {0.1 + 0.2 == 0.3}")
print(f"Difference: {(0.1 + 0.2) - 0.3:.2e}")
```

### 第 2 步：实现朴素版与稳定版 softmax

```python
import math

def softmax_naive(logits):
    exps = [math.exp(z) for z in logits]
    total = sum(exps)
    return [e / total for e in exps]

def softmax_stable(logits):
    max_logit = max(logits)
    exps = [math.exp(z - max_logit) for z in logits]
    total = sum(exps)
    return [e / total for e in exps]

safe_logits = [2.0, 1.0, 0.1]
print(f"Naive:  {softmax_naive(safe_logits)}")
print(f"Stable: {softmax_stable(safe_logits)}")

dangerous_logits = [100.0, 101.0, 102.0]
print(f"Stable: {softmax_stable(dangerous_logits)}")
# softmax_naive(dangerous_logits) would return [nan, nan, nan]
```

### 第 3 步：实现稳定的 log-sum-exp

```python
def logsumexp_naive(values):
    return math.log(sum(math.exp(v) for v in values))

def logsumexp_stable(values):
    c = max(values)
    return c + math.log(sum(math.exp(v - c) for v in values))

safe = [1.0, 2.0, 3.0]
print(f"Naive:  {logsumexp_naive(safe):.6f}")
print(f"Stable: {logsumexp_stable(safe):.6f}")

large = [500.0, 501.0, 502.0]
print(f"Stable: {logsumexp_stable(large):.6f}")
# logsumexp_naive(large) returns inf
```

### 第 4 步：实现稳定的交叉熵

```python
def cross_entropy_naive(true_class, logits):
    probs = softmax_naive(logits)
    return -math.log(probs[true_class])

def cross_entropy_stable(true_class, logits):
    max_logit = max(logits)
    shifted = [z - max_logit for z in logits]
    log_sum_exp = math.log(sum(math.exp(s) for s in shifted))
    log_prob = shifted[true_class] - log_sum_exp
    return -log_prob

logits = [2.0, 5.0, 1.0]
true_class = 1
print(f"Naive:  {cross_entropy_naive(true_class, logits):.6f}")
print(f"Stable: {cross_entropy_stable(true_class, logits):.6f}")
```

### 第 5 步：梯度检验

```python
def numerical_gradient(f, x, h=1e-5):
    grad = []
    for i in range(len(x)):
        x_plus = x[:]
        x_minus = x[:]
        x_plus[i] += h
        x_minus[i] -= h
        grad.append((f(x_plus) - f(x_minus)) / (2 * h))
    return grad

def check_gradient(analytical, numerical, tolerance=1e-5):
    for i, (a, n) in enumerate(zip(analytical, numerical)):
        denom = max(abs(a), abs(n), 1e-8)
        rel_error = abs(a - n) / denom
        status = "OK" if rel_error < tolerance else "FAIL"
        print(f"  param {i}: analytical={a:.8f} numerical={n:.8f} "
              f"rel_error={rel_error:.2e} [{status}]")

def f(params):
    x, y = params
    return x**2 + 3*x*y + y**3

def f_grad(params):
    x, y = params
    return [2*x + 3*y, 3*x + 3*y**2]

point = [2.0, 1.0]
analytical = f_grad(point)
numerical = numerical_gradient(f, point)
check_gradient(analytical, numerical)
```

## 生产实践

### 混合精度模拟

```python
import struct

def float32_to_float16_round(x):
    packed = struct.pack('f', x)
    f32 = struct.unpack('f', packed)[0]
    packed16 = struct.pack('e', f32)
    return struct.unpack('e', packed16)[0]

def simulate_bfloat16(x):
    packed = struct.pack('f', x)
    as_int = int.from_bytes(packed, 'little')
    truncated = as_int & 0xFFFF0000
    repacked = truncated.to_bytes(4, 'little')
    return struct.unpack('f', repacked)[0]
```

### 梯度裁剪

```python
def clip_by_norm(gradients, max_norm):
    total_norm = math.sqrt(sum(g**2 for g in gradients))
    if total_norm > max_norm:
        scale = max_norm / total_norm
        return [g * scale for g in gradients]
    return gradients

grads = [10.0, 20.0, 30.0]
clipped = clip_by_norm(grads, max_norm=5.0)
print(f"Original norm: {math.sqrt(sum(g**2 for g in grads)):.2f}")
print(f"Clipped norm:  {math.sqrt(sum(g**2 for g in clipped)):.2f}")
print(f"Direction preserved: {[c/clipped[0] for c in clipped]} == {[g/grads[0] for g in grads]}")
```

### NaN/Inf 检测

```python
def check_tensor(name, values):
    has_nan = any(math.isnan(v) for v in values)
    has_inf = any(math.isinf(v) for v in values)
    if has_nan or has_inf:
        print(f"WARNING {name}: nan={has_nan} inf={has_inf}")
        return False
    return True

check_tensor("good", [1.0, 2.0, 3.0])
check_tensor("bad",  [1.0, float('nan'), 3.0])
check_tensor("ugly", [1.0, float('inf'), 3.0])
```

完整实现及全部边界情况演示见 `code/numerical.py`。

## 交付产物

本节课产出：
- `code/numerical.py`：包含稳定 softmax、log-sum-exp、交叉熵、梯度检验和混合精度模拟
- `outputs/prompt-numerical-debugger.md`：用于诊断训练中的 NaN/Inf 及其他数值问题

这些稳定实现会在 Phase 3 构建训练循环时和 Phase 4 实现注意力机制时再次登场。

## 练习

1. **灾难性抵消。** 在 float32 下用朴素公式 `E[x^2] - E[x]^2` 计算 [1000000.0, 1000001.0, 1000002.0] 的方差，再用 Welford 在线算法计算一次。对比两者相对于真实方差（0.6667）的误差。

2. **精度寻踪。** 在 Python 中找出最小的正 float32 值 `x`，使得 `1.0 + x == 1.0`。这就是机器精度（machine epsilon）。验证它与 `numpy.finfo(numpy.float32).eps` 一致。

3. **Log-sum-exp 边界情况。** 用以下输入测试你的 `logsumexp_stable` 函数：(a) 所有值都相等；(b) 一个值远大于其余值；(c) 所有值都是很大的负数（-1000）。验证它在朴素版本失效的地方仍能给出正确结果。

4. **对神经网络层做梯度检验。** 实现一个线性层 `y = Wx + b` 及其解析反向传播。使用 `numerical_gradient` 对一个 3x2 权重矩阵验证其正确性。

5. **损失缩放实验。** 模拟 float16 训练：生成范围在 [1e-9, 1e-3] 的随机梯度，转换为 float16，测量有多少比例变成了零。然后应用损失缩放（乘以 1024），转换为 float16，再缩放回去，重新测量归零比例。

## 关键术语

| 术语 | 人们常说的 | 实际含义 |
|------|----------------|----------------------|
| IEEE 754 | "浮点数标准" | 定义二进制浮点格式、舍入规则和特殊值（inf、nan）的国际标准。每个现代 CPU 和 GPU 都实现了它。 |
| 机器精度（Machine epsilon） | "精度极限" | 在给定浮点格式中使 1.0 + e != 1.0 成立的最小值 e。对 float32 而言约为 1.19e-7。 |
| 灾难性抵消 | "减法导致的精度丢失" | 对两个几乎相等的浮点数做减法时，有效数字相互抵消，舍入噪声主导了结果。 |
| 上溢 | "数太大了" | 结果超出最大可表示值，变成 inf。exp(89) 会让 float32 上溢。 |
| 下溢 | "数太小了" | 结果比最小可表示正数更接近零，变成 0.0。exp(-104) 会让 float32 下溢。 |
| Log-sum-exp 技巧 | "先减最大值" | 通过提取 exp(max(x)) 因子来计算 log(sum(exp(x)))，防止上溢和下溢。用于 softmax、交叉熵和对数概率运算。 |
| 稳定 softmax | "不会爆炸的 softmax" | 在取指数前减去 max(logits)。结果在数值上完全相同，且不可能上溢。 |
| 梯度检验 | "验证你的反向传播" | 将反向传播得到的解析梯度与有限差分得到的数值梯度做对比，以捕获实现 bug。 |
| 混合精度 | "前向用 float16，反向用 float32" | 对速度敏感的操作使用低精度浮点，对数值敏感的操作使用高精度浮点。典型加速比为 2-3 倍。 |
| 损失缩放 | "防止梯度下溢" | 在反向传播前将损失乘以一个大常数，使梯度保持在 float16 的可表示范围内，权重更新前再除以同一常数。 |
| bfloat16 | "Brain floating point" | Google 的 16 位格式，拥有 8 位指数（与 float32 范围相同）和 7 位尾数（精度低于 float16）。训练首选。 |
| 梯度裁剪 | "给梯度范数封顶" | 缩放梯度向量使其范数不超过阈值。防止梯度爆炸毁掉权重。 |
| NaN | "Not a Number" | 由未定义运算（0/0、inf-inf、sqrt(-1)）产生的特殊浮点值。会传播到后续所有算术运算中。 |
| Inf | "无穷大" | 由上溢或除零产生的特殊浮点值。组合运算可能产生 NaN（inf - inf、inf * 0）。 |
| 数值梯度 | "暴力求导" | 通过计算 f(x+h) 和 f(x-h) 并除以 2h 来近似导数。慢，但用于验证非常可靠。 |

## 延伸阅读

- [What Every Computer Scientist Should Know About Floating-Point Arithmetic (Goldberg 1991)](https://docs.oracle.com/cd/E19957-01/806-3568/ncg_goldberg.html) —— 这一领域的权威参考，内容密集但完备
- [Mixed Precision Training (Micikevicius et al., 2018)](https://arxiv.org/abs/1710.03740) —— NVIDIA 提出 float16 训练损失缩放的那篇论文
- [AMP: Automatic Mixed Precision (PyTorch docs)](https://pytorch.org/docs/stable/amp.html) —— PyTorch 混合精度的实用指南
- [bfloat16 format (Google Cloud TPU docs)](https://cloud.google.com/tpu/docs/bfloat16) —— Google 为什么为 TPU 选择这种格式
- [Kahan Summation (Wikipedia)](https://en.wikipedia.org/wiki/Kahan_summation_algorithm) —— 减少浮点求和舍入误差的算法
