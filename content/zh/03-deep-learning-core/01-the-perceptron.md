# 感知机（Perceptron）

> 感知机是神经网络的原子。把它剖开，里面是权重、偏置和一个决策。

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 1 (Linear Algebra Intuition)
**Time:** ~60 minutes

## 学习目标

- 用 Python 从零实现一个感知机，包括权重更新规则和阶跃激活函数
- 解释为什么单个感知机只能解决线性可分问题，并演示 XOR 失败案例
- 通过组合 OR、NAND、AND 门构建多层感知机来解决 XOR
- 使用 sigmoid 激活和反向传播训练一个两层网络，让它自动学会 XOR

## 问题背景

你已经懂向量和点积，也知道矩阵能把输入变换成输出。但机器是如何*学会*该用哪种变换的？

感知机回答了这个问题。它是最简单的学习机器：接收若干输入，乘以权重，加上偏置，做出一个二元决策。然后调整。就这么简单。有史以来的每一个神经网络，都是这个想法一层层堆叠出来的。

理解感知机，就是理解「学习」在代码层面到底意味着什么：不断调整数字，直到输出与现实相符。

## 核心概念

### 一个神经元，一次决策

感知机接收 n 个输入，每个输入乘以一个权重，求和后加上偏置，再把结果送入激活函数。

```mermaid
graph LR
    x1["x1"] -- "w1" --> sum["Σ(wi*xi) + b"]
    x2["x2"] -- "w2" --> sum
    x3["x3"] -- "w3" --> sum
    bias["bias"] --> sum
    sum --> step["step(z)"]
    step --> out["output (0 or 1)"]
```

阶跃函数（step function）非常粗暴：加权和加偏置 >= 0 就输出 1，否则输出 0。

```
step(z) = 1  if z >= 0
           0  if z < 0
```

这是一个线性分类器。权重和偏置定义了一条直线（在更高维空间中则是超平面），把输入空间切成两个区域。

### 决策边界

对于两个输入，感知机在二维空间中画出一条直线：

```
  x2
  ┤
  │  Class 1        /
  │    (0)          /
  │                /
  │               / w1·x1 + w2·x2 + b = 0
  │              /
  │             /     Class 2
  │            /        (1)
  ┼───────────/──────────── x1
```

线一侧的所有点输出 0，另一侧的所有点输出 1。训练就是不断移动这条线，直到它能正确分开两个类别。

### 学习规则

感知机的学习规则很简单：

```
For each training example (x, y_true):
    y_pred = predict(x)
    error = y_true - y_pred

    For each weight:
        w_i = w_i + learning_rate * error * x_i
    bias = bias + learning_rate * error
```

预测正确时 error = 0，什么都不变。预测为 0 但应该是 1 时，权重增大；预测为 1 但应该是 0 时，权重减小。学习率控制每次调整的幅度。

### XOR 问题

感知机就在这里碰壁。看看这几个逻辑门：

```
AND gate:           OR gate:            XOR gate:
x1  x2  out         x1  x2  out         x1  x2  out
0   0   0           0   0   0           0   0   0
0   1   0           0   1   1           0   1   1
1   0   0           1   0   1           1   0   1
1   1   1           1   1   1           1   1   0
```

AND 和 OR 都是线性可分的：你可以画一条线把 0 和 1 分开。XOR 不行。没有任何一条直线能把 [0,1] 和 [1,0] 与 [0,0] 和 [1,1] 分开。

```
AND (separable):        XOR (not separable):

  x2                      x2
  1 ┤  0     1            1 ┤  1     0
    │     /                 │
  0 ┤  0 / 0              0 ┤  0     1
    ┼──/──────── x1         ┼──────────── x1
       line works!          no single line works!
```

这是一个根本性的限制。单个感知机只能解决线性可分问题。Minsky 和 Papert 在 1969 年证明了这一点，几乎让神经网络研究停滞了十年。

解法：把感知机堆叠成层。多层感知机可以把两个线性决策组合成一个非线性决策，从而解决 XOR。

```figure
perceptron-boundary
```

## 从零实现

### 步骤 1：Perceptron 类

```python
class Perceptron:
    def __init__(self, n_inputs, learning_rate=0.1):
        self.weights = [0.0] * n_inputs
        self.bias = 0.0
        self.lr = learning_rate

    def predict(self, inputs):
        total = sum(w * x for w, x in zip(self.weights, inputs))
        total += self.bias
        return 1 if total >= 0 else 0

    def train(self, training_data, epochs=100):
        for epoch in range(epochs):
            errors = 0
            for inputs, target in training_data:
                prediction = self.predict(inputs)
                error = target - prediction
                if error != 0:
                    errors += 1
                    for i in range(len(self.weights)):
                        self.weights[i] += self.lr * error * inputs[i]
                    self.bias += self.lr * error
            if errors == 0:
                print(f"Converged at epoch {epoch + 1}")
                return
        print(f"Did not converge after {epochs} epochs")
```

### 步骤 2：在逻辑门上训练

```python
and_data = [
    ([0, 0], 0),
    ([0, 1], 0),
    ([1, 0], 0),
    ([1, 1], 1),
]

or_data = [
    ([0, 0], 0),
    ([0, 1], 1),
    ([1, 0], 1),
    ([1, 1], 1),
]

not_data = [
    ([0], 1),
    ([1], 0),
]

print("=== AND Gate ===")
p_and = Perceptron(2)
p_and.train(and_data)
for inputs, _ in and_data:
    print(f"  {inputs} -> {p_and.predict(inputs)}")

print("\n=== OR Gate ===")
p_or = Perceptron(2)
p_or.train(or_data)
for inputs, _ in or_data:
    print(f"  {inputs} -> {p_or.predict(inputs)}")

print("\n=== NOT Gate ===")
p_not = Perceptron(1)
p_not.train(not_data)
for inputs, _ in not_data:
    print(f"  {inputs} -> {p_not.predict(inputs)}")
```

### 步骤 3：见证 XOR 失败

```python
xor_data = [
    ([0, 0], 0),
    ([0, 1], 1),
    ([1, 0], 1),
    ([1, 1], 0),
]

print("\n=== XOR Gate (single perceptron) ===")
p_xor = Perceptron(2)
p_xor.train(xor_data, epochs=1000)
for inputs, expected in xor_data:
    result = p_xor.predict(inputs)
    status = "OK" if result == expected else "WRONG"
    print(f"  {inputs} -> {result} (expected {expected}) {status}")
```

它永远不会收敛。这就是单个感知机无法学会 XOR 的铁证。

### 步骤 4：用两层网络解决 XOR

诀窍在于：XOR = (x1 OR x2) AND NOT (x1 AND x2)。把三个感知机组合起来：

```mermaid
graph LR
    x1["x1"] --> OR["OR neuron"]
    x1 --> NAND["NAND neuron"]
    x2["x2"] --> OR
    x2 --> NAND
    OR --> AND["AND neuron"]
    NAND --> AND
    AND --> out["output"]
```

```python
def xor_network(x1, x2):
    or_neuron = Perceptron(2)
    or_neuron.weights = [1.0, 1.0]
    or_neuron.bias = -0.5

    nand_neuron = Perceptron(2)
    nand_neuron.weights = [-1.0, -1.0]
    nand_neuron.bias = 1.5

    and_neuron = Perceptron(2)
    and_neuron.weights = [1.0, 1.0]
    and_neuron.bias = -1.5

    hidden1 = or_neuron.predict([x1, x2])
    hidden2 = nand_neuron.predict([x1, x2])
    output = and_neuron.predict([hidden1, hidden2])
    return output


print("\n=== XOR Gate (multi-layer network) ===")
for inputs, expected in xor_data:
    result = xor_network(inputs[0], inputs[1])
    print(f"  {inputs} -> {result} (expected {expected})")
```

四种情况全部正确。把感知机堆叠成层，就能产生任何单个感知机都画不出来的决策边界。

### 步骤 5：训练一个两层网络

步骤 4 的权重是手工设定的。这对 XOR 行得通，但真实问题中你事先并不知道正确的权重。解法：用 sigmoid 替换阶跃函数，并通过反向传播（backpropagation）自动学习权重。

```python
class TwoLayerNetwork:
    def __init__(self, learning_rate=0.5):
        import random
        random.seed(0)
        self.w_hidden = [[random.uniform(-1, 1), random.uniform(-1, 1)] for _ in range(2)]
        self.b_hidden = [random.uniform(-1, 1), random.uniform(-1, 1)]
        self.w_output = [random.uniform(-1, 1), random.uniform(-1, 1)]
        self.b_output = random.uniform(-1, 1)
        self.lr = learning_rate

    def sigmoid(self, x):
        import math
        x = max(-500, min(500, x))
        return 1.0 / (1.0 + math.exp(-x))

    def forward(self, inputs):
        self.inputs = inputs
        self.hidden_outputs = []
        for i in range(2):
            z = sum(w * x for w, x in zip(self.w_hidden[i], inputs)) + self.b_hidden[i]
            self.hidden_outputs.append(self.sigmoid(z))
        z_out = sum(w * h for w, h in zip(self.w_output, self.hidden_outputs)) + self.b_output
        self.output = self.sigmoid(z_out)
        return self.output

    def train(self, training_data, epochs=10000):
        for epoch in range(epochs):
            total_error = 0
            for inputs, target in training_data:
                output = self.forward(inputs)
                error = target - output
                total_error += error ** 2

                d_output = error * output * (1 - output)

                saved_w_output = self.w_output[:]
                hidden_deltas = []
                for i in range(2):
                    h = self.hidden_outputs[i]
                    hd = d_output * saved_w_output[i] * h * (1 - h)
                    hidden_deltas.append(hd)

                for i in range(2):
                    self.w_output[i] += self.lr * d_output * self.hidden_outputs[i]
                self.b_output += self.lr * d_output

                for i in range(2):
                    for j in range(len(inputs)):
                        self.w_hidden[i][j] += self.lr * hidden_deltas[i] * inputs[j]
                    self.b_hidden[i] += self.lr * hidden_deltas[i]
```

```python
net = TwoLayerNetwork(learning_rate=2.0)
net.train(xor_data, epochs=10000)
for inputs, expected in xor_data:
    result = net.forward(inputs)
    predicted = 1 if result >= 0.5 else 0
    print(f"  {inputs} -> {result:.4f} (rounded: {predicted}, expected {expected})")
```

与步骤 4 相比有两个关键区别。第一，sigmoid 替换了阶跃函数——它是平滑的，所以梯度存在。第二，`train` 方法把误差从输出层反向传播到隐藏层，按照每个权重对误差的贡献比例进行调整。这就是反向传播，只用了 20 行代码。

这是通向第 03 课的桥梁。`d_output` 和 `hidden_deltas` 背后的数学，就是把链式法则应用到网络计算图上。我们会在那一课正式推导。

## 生产实践

你刚刚从零构建的一切，一行 import 就有了：

```python
from sklearn.linear_model import Perceptron as SkPerceptron
import numpy as np

X = np.array([[0,0],[0,1],[1,0],[1,1]])
y = np.array([0, 0, 0, 1])

clf = SkPerceptron(max_iter=100, tol=1e-3)
clf.fit(X, y)
print([clf.predict([x])[0] for x in X])
```

五行代码。你那个 30 行的 `Perceptron` 类做的是同一件事。sklearn 版本多了收敛检查、多种损失函数和稀疏输入支持——但核心循环完全相同：加权求和、阶跃函数、出错时更新权重。

真正的差距出现在规模化时。生产级网络的变化包括：

- 阶跃函数换成 sigmoid、ReLU 或其他平滑激活函数
- 权重通过反向传播自动学习（第 03 课）
- 层数越来越深：3 层、10 层、100+ 层
- 但同一个原理始终成立：每一层都从上一层的输出中创造新的特征

单个感知机只能画直线。把它们堆叠起来，你就能画出任何形状。

## 交付产物

本课产出：
- `outputs/skill-perceptron.md` - 一份技能文档，说明何时需要单层架构、何时需要多层架构

## 练习

1. 在 NAND 门上训练一个感知机（NAND 是通用门——任何逻辑电路都可以用 NAND 搭建）。验证它的权重和偏置构成一个有效的决策边界。
2. 修改 Perceptron 类，在每个 epoch 记录决策边界（w1*x1 + w2*x2 + b = 0）。打印在 AND 门上训练时这条线是如何移动的。
3. 构建一个 3 输入感知机，仅当 3 个输入中至少有 2 个为 1 时输出 1（多数表决函数）。它是线性可分的吗？为什么？

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|----------------|----------------------|
| 感知机（Perceptron） | 「一个假神经元」 | 一个线性分类器：输入与权重的点积加上偏置，再经过阶跃函数 |
| 权重（Weight） | 「输入有多重要」 | 一个乘数，用来缩放每个输入对决策的贡献 |
| 偏置（Bias） | 「阈值」 | 一个常数，用来平移决策边界，让感知机即使在输入全为零时也能激活 |
| 激活函数（Activation function） | 「把数值压扁的东西」 | 加权求和之后应用的函数——感知机用阶跃函数，现代网络用 sigmoid/ReLU |
| 线性可分（Linearly separable） | 「能在它们之间画一条线」 | 一个单一超平面就能把类别完美分开的数据集 |
| XOR 问题 | 「感知机做不到的事」 | 证明单层网络无法学习非线性可分函数 |
| 决策边界（Decision boundary） | 「分类器切换的位置」 | 把输入空间分成两个类别的超平面 w*x + b = 0 |
| 多层感知机（Multi-layer perceptron） | 「真正的神经网络」 | 按层堆叠的感知机，每一层的输出作为下一层的输入 |

## 延伸阅读

- Frank Rosenblatt, "The Perceptron: A Probabilistic Model for Information Storage and Organization in the Brain" (1958)——开创这一切的原始论文
- Minsky & Papert, "Perceptrons" (1969)——证明单层网络无法解决 XOR、让感知机研究停滞十年的那本书
- Michael Nielsen, "Neural Networks and Deep Learning", Chapter 1 (http://neuralnetworksanddeeplearning.com/)——免费在线阅读，对感知机如何组合成网络的最佳可视化讲解
