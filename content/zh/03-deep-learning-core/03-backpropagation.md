# 从零实现反向传播

> 反向传播是让学习成为可能的算法。没有它，神经网络只是昂贵的随机数生成器。

**Type:** Build
**Languages:** Python
**Prerequisites:** Lesson 03.02 (Multi-Layer Networks)
**Time:** ~120 minutes

## 学习目标

- 实现一个基于 Value 的自动求导（autograd）引擎，构建计算图并通过拓扑排序计算梯度
- 用链式法则推导加法、乘法和 sigmoid 的反向传播公式
- 仅用自己从零实现的反向传播引擎，在 XOR 和圆形分类任务上训练多层网络
- 识别深层 sigmoid 网络中的梯度消失问题，并解释梯度为何会指数级缩小

## 问题背景

你的网络只有一个隐藏层，输入维度 768，输出维度 3072，共有 2,359,296 个权重。它做出了一次错误的预测。是哪些权重导致了这个错误？逐个测试每个权重意味着 230 万次前向传播。而反向传播只需一次反向传播就能算出全部 230 万个梯度。这不是一种优化，而是"可训练"与"不可能训练"之间的分界线。

朴素的做法是：取一个权重，微调一点点，重新跑一遍前向传播，看损失是上升还是下降。这就得到了该权重的梯度。然后对网络中的每个权重重复这个过程，再乘以成千上万的训练步数和数以百万计的数据点。要训练出任何有用的东西，你需要的是地质年代级别的时间。

反向传播解决了这个问题。一次前向传播，一次反向传播，所有梯度全部算出。诀窍是微积分中的链式法则，被系统性地应用到一张计算图上。正是这个算法让深度学习变得实用。没有它，我们至今还会困在玩具问题上。

## 核心概念

### 链式法则在网络中的应用

你在 Phase 01 第 05 课见过链式法则。快速回顾：如果 y = f(g(x))，那么 dy/dx = f'(g(x)) * g'(x)。沿着链条把导数相乘。

在神经网络中，这条"链"就是从输入到损失的一系列操作。每一层应用权重、加上偏置、通过激活函数。损失函数比较最终输出与目标值。反向传播沿着这条链反向回溯，计算每个操作对误差的贡献。

### 计算图

每次前向传播都会构建一张图。每个节点是一个操作（乘法、加法、sigmoid）。每条边在前向时传递数值，在反向时传递梯度。

```mermaid
graph LR
    x["x"] --> mul["*"]
    w["w"] --> mul
    mul -- "z1 = w*x" --> add["+"]
    b["b"] --> add
    add -- "z2 = z1 + b" --> sig["sigmoid"]
    sig -- "a = sigmoid(z2)" --> loss["Loss"]
    y["target"] --> loss
```

前向传播：数值从左向右流动。x 和 w 得到 z1 = w*x。加上 b 得到 z2。经过 sigmoid 得到激活值 a。用损失函数比较 a 和目标值 y。

反向传播：梯度从右向左流动。从 dL/da 开始（损失对激活值的变化率），乘以 da/dz2（sigmoid 的导数），得到 dL/dz2。再拆分为 dL/db（等于 dL/dz2，因为 z2 = z1 + b）和 dL/dz1。然后 dL/dw = dL/dz1 * x，dL/dx = dL/dz1 * w。

在反向传播过程中，图中每个节点只做一件事：接收上游传来的梯度，乘以自己的局部导数，再传给下游。

### 前向与反向

```mermaid
graph TB
    subgraph Forward["Forward Pass"]
        direction LR
        f1["Input x"] --> f2["z = Wx + b"]
        f2 --> f3["a = sigmoid(z)"]
        f3 --> f4["Loss = (a - y)^2"]
    end
    subgraph Backward["Backward Pass"]
        direction RL
        b4["dL/dL = 1"] --> b3["dL/da = 2(a-y)"]
        b3 --> b2["dL/dz = dL/da * a(1-a)"]
        b2 --> b1["dL/dW = dL/dz * x\ndL/db = dL/dz"]
    end
    Forward --> Backward
```

前向传播会保存所有中间值：z、a，以及每一层的输入。反向传播需要这些保存下来的值才能计算梯度。这就是反向传播核心的"内存换计算"权衡：用内存（存储激活值）换取速度（一次传播代替几百万次）。

### 梯度在网络中的流动

对于一个 3 层网络，梯度要链式穿过每一层：

```mermaid
graph RL
    L["Loss"] -- "dL/da3" --> L3["Layer 3\na3 = sigmoid(z3)"]
    L3 -- "dL/dz3 = dL/da3 * sigmoid'(z3)" --> L2["Layer 2\na2 = sigmoid(z2)"]
    L2 -- "dL/dz2 = dL/da2 * sigmoid'(z2)" --> L1["Layer 1\na1 = sigmoid(z1)"]
    L1 -- "dL/dz1 = dL/da1 * sigmoid'(z1)" --> I["Input"]
```

在每一层，梯度都要乘以 sigmoid 的导数。sigmoid 的导数是 a * (1 - a)，最大值为 0.25（当 a = 0.5 时取得）。穿过三层后，梯度最多被乘以 0.25^3 = 0.0156。穿过十层后：0.25^10 = 0.000001。

### 梯度消失

这就是梯度消失（vanishing gradient）问题。sigmoid 把输出压缩到 0 和 1 之间，它的导数永远小于 0.25。堆叠足够多的 sigmoid 层，梯度就会缩小到几乎为零。前面的层几乎学不到任何东西，因为它们收到的梯度接近于零。

```
sigmoid(z):     Output range [0, 1]
sigmoid'(z):    Max value 0.25 (at z = 0)

After 5 layers:   gradient * 0.25^5 = 0.001x original
After 10 layers:  gradient * 0.25^10 = 0.000001x original
```

这就是深层 sigmoid 网络几乎无法训练的原因。解决方案——ReLU 及其变体——是第 04 课的主题。目前你只需要理解：反向传播本身工作得很完美，问题出在它要穿过的东西上。

### 推导 2 层网络的梯度

下面是具体的数学推导：网络包含输入 x、带 sigmoid 的隐藏层、带 sigmoid 的输出层，损失函数为 MSE。

前向传播：
```
z1 = W1 * x + b1
a1 = sigmoid(z1)
z2 = W2 * a1 + b2
a2 = sigmoid(z2)
L = (a2 - y)^2
```

反向传播（逐步应用链式法则）：
```
dL/da2 = 2(a2 - y)
da2/dz2 = a2 * (1 - a2)
dL/dz2 = dL/da2 * da2/dz2 = 2(a2 - y) * a2 * (1 - a2)

dL/dW2 = dL/dz2 * a1
dL/db2 = dL/dz2

dL/da1 = dL/dz2 * W2
da1/dz1 = a1 * (1 - a1)
dL/dz1 = dL/da1 * da1/dz1

dL/dW1 = dL/dz1 * x
dL/db1 = dL/dz1
```

每个梯度都是从损失出发回溯得到的一串局部导数的乘积。反向传播的全部内容就是这些。

```figure
backprop-vanishing
```

## 从零实现

### 第 1 步：Value 节点

计算中的每个数字都变成一个 Value。它存储自己的数据、梯度，以及它是如何被创建的（这样它才知道如何反向计算梯度）。

```python
class Value:
    def __init__(self, data, children=(), op=''):
        self.data = data
        self.grad = 0.0
        self._backward = lambda: None
        self._children = set(children)
        self._op = op

    def __repr__(self):
        return f"Value(data={self.data:.4f}, grad={self.grad:.4f})"
```

此时还没有梯度（0.0），也还没有反向函数（空操作）。`_children` 记录哪些 Value 生成了当前这个 Value，以便之后对图做拓扑排序。

### 第 2 步：带反向函数的运算

每个运算都会创建一个新的 Value，并定义梯度如何反向流过它。

```python
def __add__(self, other):
    other = other if isinstance(other, Value) else Value(other)
    out = Value(self.data + other.data, (self, other), '+')

    def _backward():
        self.grad += out.grad
        other.grad += out.grad

    out._backward = _backward
    return out

def __mul__(self, other):
    other = other if isinstance(other, Value) else Value(other)
    out = Value(self.data * other.data, (self, other), '*')

    def _backward():
        self.grad += other.data * out.grad
        other.grad += self.data * out.grad

    out._backward = _backward
    return out
```

加法：d(a+b)/da = 1，d(a+b)/db = 1。所以两个输入直接接收输出的梯度。

乘法：d(a*b)/da = b，d(a*b)/db = a。每个输入接收的是另一个输入的值乘以输出梯度。

`+=` 至关重要。一个 Value 可能被多个运算使用，它的梯度是来自所有路径的梯度之和。

### 第 3 步：Sigmoid 与损失

```python
import math

def sigmoid(self):
    x = self.data
    x = max(-500, min(500, x))
    s = 1.0 / (1.0 + math.exp(-x))
    out = Value(s, (self,), 'sigmoid')

    def _backward():
        self.grad += (s * (1 - s)) * out.grad

    out._backward = _backward
    return out
```

sigmoid 的导数：sigmoid(x) * (1 - sigmoid(x))。我们在前向传播时已经算出了 sigmoid(x) = s，直接复用即可，不需要额外计算。

```python
def mse_loss(predicted, target):
    diff = predicted + Value(-target)
    return diff * diff
```

单个输出的 MSE：(predicted - target)^2。我们把减法表示为加上一个取负的 Value。

### 第 4 步：反向传播

拓扑排序确保我们按正确顺序处理节点——一个节点的梯度必须完全累积好之后，才能继续向下传播。

```python
def backward(self):
    topo = []
    visited = set()

    def build_topo(v):
        if v not in visited:
            visited.add(v)
            for child in v._children:
                build_topo(child)
            topo.append(v)

    build_topo(self)
    self.grad = 1.0
    for v in reversed(topo):
        v._backward()
```

从损失节点开始（梯度 = 1.0，因为 dL/dL = 1），沿着排好序的图反向遍历。每个节点的 `_backward` 把梯度推送给它的子节点。

### 第 5 步：层与网络

```python
import random

class Neuron:
    def __init__(self, n_inputs):
        scale = (2.0 / n_inputs) ** 0.5
        self.weights = [Value(random.uniform(-scale, scale)) for _ in range(n_inputs)]
        self.bias = Value(0.0)

    def __call__(self, x):
        act = sum((wi * xi for wi, xi in zip(self.weights, x)), self.bias)
        return act.sigmoid()

    def parameters(self):
        return self.weights + [self.bias]


class Layer:
    def __init__(self, n_inputs, n_outputs):
        self.neurons = [Neuron(n_inputs) for _ in range(n_outputs)]

    def __call__(self, x):
        out = [n(x) for n in self.neurons]
        return out[0] if len(out) == 1 else out

    def parameters(self):
        params = []
        for n in self.neurons:
            params.extend(n.parameters())
        return params


class Network:
    def __init__(self, sizes):
        self.layers = []
        for i in range(len(sizes) - 1):
            self.layers.append(Layer(sizes[i], sizes[i + 1]))

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
            if not isinstance(x, list):
                x = [x]
        return x[0] if len(x) == 1 else x

    def parameters(self):
        params = []
        for layer in self.layers:
            params.extend(layer.parameters())
        return params

    def zero_grad(self):
        for p in self.parameters():
            p.grad = 0.0
```

一个 Neuron 接收输入，计算加权和加上偏置，再应用 sigmoid。权重初始化按 sqrt(2/n_inputs) 缩放，以防止较深网络中 sigmoid 饱和。Layer 是 Neuron 的列表，Network 是 Layer 的列表。`parameters()` 方法收集所有可学习的 Value，方便我们更新它们。

### 第 6 步：训练 XOR

```python
random.seed(42)
net = Network([2, 4, 1])

xor_data = [
    ([0.0, 0.0], 0.0),
    ([0.0, 1.0], 1.0),
    ([1.0, 0.0], 1.0),
    ([1.0, 1.0], 0.0),
]

learning_rate = 1.0

for epoch in range(1000):
    total_loss = Value(0.0)
    for inputs, target in xor_data:
        x = [Value(i) for i in inputs]
        pred = net(x)
        loss = mse_loss(pred, target)
        total_loss = total_loss + loss

    net.zero_grad()
    total_loss.backward()

    for p in net.parameters():
        p.data -= learning_rate * p.grad

    if epoch % 100 == 0:
        print(f"Epoch {epoch:4d} | Loss: {total_loss.data:.6f}")

print("\nXOR Results:")
for inputs, target in xor_data:
    x = [Value(i) for i in inputs]
    pred = net(x)
    print(f"  {inputs} -> {pred.data:.4f} (expected {target})")
```

观察损失逐渐下降。从随机预测到正确的 XOR 输出，完全靠反向传播计算梯度、把权重朝正确方向微调来驱动。

### 第 7 步：圆形分类

在第 02 课，你为圆形分类手工调过权重。现在让网络自己学习这些权重。

```python
random.seed(7)

def generate_circle_data(n=100):
    data = []
    for _ in range(n):
        x1 = random.uniform(-1.5, 1.5)
        x2 = random.uniform(-1.5, 1.5)
        label = 1.0 if x1 * x1 + x2 * x2 < 1.0 else 0.0
        data.append(([x1, x2], label))
    return data

circle_data = generate_circle_data(80)

circle_net = Network([2, 8, 1])
learning_rate = 0.5

for epoch in range(2000):
    random.shuffle(circle_data)
    total_loss_val = 0.0
    for inputs, target in circle_data:
        x = [Value(i) for i in inputs]
        pred = circle_net(x)
        loss = mse_loss(pred, target)
        circle_net.zero_grad()
        loss.backward()
        for p in circle_net.parameters():
            p.data -= learning_rate * p.grad
        total_loss_val += loss.data

    if epoch % 200 == 0:
        correct = 0
        for inputs, target in circle_data:
            x = [Value(i) for i in inputs]
            pred = circle_net(x)
            predicted_class = 1.0 if pred.data > 0.5 else 0.0
            if predicted_class == target:
                correct += 1
        accuracy = correct / len(circle_data) * 100
        print(f"Epoch {epoch:4d} | Loss: {total_loss_val:.4f} | Accuracy: {accuracy:.1f}%")
```

这里我们使用在线 SGD——每个样本之后立即更新权重，而不是累积完整批次。这样能更快打破对称性，并避免在完整损失面上发生 sigmoid 饱和。每个 epoch 都打乱数据顺序，防止网络记住样本的排列。

不需要任何手工调参。网络自己发现了圆形决策边界。这就是反向传播的力量：你定义架构、损失函数和数据，算法自己找出权重。

## 生产实践

PyTorch 用几行代码就能完成上面的一切。核心思想完全相同——自动求导在前向传播时构建计算图，再反向遍历它来计算梯度。

```python
import torch
import torch.nn as nn

model = nn.Sequential(
    nn.Linear(2, 4),
    nn.Sigmoid(),
    nn.Linear(4, 1),
    nn.Sigmoid(),
)
optimizer = torch.optim.SGD(model.parameters(), lr=1.0)
criterion = nn.MSELoss()

X = torch.tensor([[0,0],[0,1],[1,0],[1,1]], dtype=torch.float32)
y = torch.tensor([[0],[1],[1],[0]], dtype=torch.float32)

for epoch in range(1000):
    pred = model(X)
    loss = criterion(pred, y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

print("PyTorch XOR Results:")
with torch.no_grad():
    for i in range(4):
        pred = model(X[i])
        print(f"  {X[i].tolist()} -> {pred.item():.4f} (expected {y[i].item()})")
```

`loss.backward()` 对应你写的 `total_loss.backward()`。`optimizer.step()` 对应你手动的 `p.data -= lr * p.grad`。`optimizer.zero_grad()` 对应你的 `net.zero_grad()`。同一个算法，工业级的实现。PyTorch 还处理了 GPU 加速、混合精度、梯度检查点和数百种层类型。但反向传播本身仍是同一条链式法则，应用在同一张计算图上。

训练包括前向传播、反向传播，然后更新权重。推理只运行前向传播——没有梯度，没有更新。这个区别很重要，因为生产环境中运行的就是推理。当你调用 Claude 或 GPT 这样的 API 时，你在执行推理——你的提示词向前流过网络，token 从另一端输出，没有任何权重被改变。理解反向传播之所以重要，是因为那个网络中的每一个权重都是由它塑造的。

## 交付产物

本课产出：
- `outputs/prompt-gradient-debugger.md` —— 一个可复用的提示词，用于诊断任意神经网络中的梯度问题（消失、爆炸、NaN）

## 练习

1. 为 Value 类添加 `__sub__` 方法（a - b = a + (-1 * b)），然后实现 `__neg__` 方法。用 (a - b)^2 这样的简单表达式手工推导梯度，验证实现是否正确。

2. 为 Value 添加 `relu` 方法（输出 max(0, x)，导数在 x > 0 时为 1，否则为 0）。把隐藏层的 sigmoid 替换为 relu，重新训练 XOR，比较收敛速度。你应该会看到训练更快——这是第 04 课的预告。

3. 为 Value 实现支持整数幂的 `__pow__` 方法。用它把 `mse_loss` 替换为更规范的 `(predicted - target) ** 2` 表达式。验证梯度与原实现一致。

4. 在训练循环中加入梯度裁剪：调用 `backward()` 之后，把所有梯度裁剪到 [-1, 1]。训练一个更深的网络（4 层以上、使用 sigmoid），比较有无裁剪时的损失曲线。这是你对抗梯度爆炸的第一道防线。

5. 做一个可视化：在 XOR 上训练完成后，打印网络中每个参数的梯度。找出梯度最小的那一层。这正好印证了你在核心概念部分读到的梯度消失问题。

## 关键术语

| 术语 | 人们常说 | 实际含义 |
|------|----------------|----------------------|
| 反向传播（Backpropagation） | "网络在学习" | 通过沿计算图反向应用链式法则，为每个权重计算 dL/dw 的算法 |
| 计算图（Computational graph） | "网络结构" | 一个有向无环图，节点是操作，边在前向时传递数值、在反向时传递梯度 |
| 链式法则（Chain rule） | "把导数乘起来" | 如果 y = f(g(x))，那么 dy/dx = f'(g(x)) * g'(x)——反向传播的数学基础 |
| 梯度（Gradient） | "最陡上升的方向" | 损失对某个参数的偏导数——告诉你该如何调整这个参数来降低损失 |
| 梯度消失（Vanishing gradient） | "深层网络学不动" | 梯度穿过带饱和激活函数（如 sigmoid）的层时呈指数级缩小 |
| 前向传播（Forward pass） | "运行网络" | 依次应用每一层的操作、从输入计算输出，并保存中间值 |
| 反向传播过程（Backward pass） | "计算梯度" | 反向遍历计算图，用链式法则在每个节点累积梯度 |
| 学习率（Learning rate） | "学得有多快" | 控制权重更新步长的标量：w_new = w_old - lr * gradient |
| 拓扑排序（Topological sort） | "正确的顺序" | 一种图节点排序，每个节点出现在它所依赖的所有节点之后——确保梯度先完全累积再传播 |
| 自动求导（Autograd） | "自动微分" | 在前向计算时构建计算图并自动计算梯度的系统——PyTorch 引擎做的就是这件事 |

## 延伸阅读

- Rumelhart、Hinton 与 Williams，"Learning representations by back-propagating errors"（1986）——让反向传播成为主流、解锁多层网络训练的论文
- 3Blue1Brown，"Neural Networks" 系列（https://www.youtube.com/playlist?list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi）——关于反向传播和梯度在网络中流动的最佳可视化讲解
