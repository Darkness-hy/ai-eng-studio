# 优化器

> 梯度下降告诉你该往哪个方向走，却不告诉你该走多远、走多快。SGD 是指南针，Adam 则是带实时路况的 GPS。

**Type:** Build
**Languages:** Python
**Prerequisites:** Lesson 03.05 (Loss Functions)
**Time:** ~75 minutes

## 学习目标

- 用 Python 从零实现 SGD、带动量的 SGD、Adam 和 AdamW 优化器
- 解释 Adam 的偏差修正（bias correction）如何补偿训练初期零初始化的矩估计
- 演示为什么在同一任务上 AdamW 的泛化效果优于 Adam 加 L2 正则化
- 为 Transformer、CNN、GAN 和微调任务选择合适的优化器及默认超参数

## 问题背景

你已经算出了梯度。你知道第 4,721 号权重应该减小 0.003 才能降低损失。但 0.003 是什么单位？按什么比例缩放？第 1 步和第 1,000 步的更新量应该一样吗？

朴素的梯度下降在每一步对每个参数都使用相同的学习率：w = w - lr * gradient。这在实践中带来了三个让神经网络训练痛苦不堪的问题。

第一，震荡。损失地形很少是一个光滑的碗，更像一条又长又窄的山谷。梯度指向山谷的横向（陡峭方向），而不是沿着山谷的纵向（平缓方向）。梯度下降在狭窄的横向上来回弹跳，而在真正有用的方向上进展甚微。你一定见过这种现象：损失先快速下降然后停滞——不是因为模型收敛了，而是因为它在震荡。

第二，所有参数共用一个学习率是错误的。有些权重需要大幅更新（它们还处于早期欠拟合阶段），有些权重只需要微小更新（它们已经接近最优值）。适合前者的学习率会毁掉后者，反之亦然。

第三，鞍点。在高维空间中，损失地形存在大片梯度接近零的平坦区域。朴素 SGD 以梯度的速度在这些区域里爬行，而这个速度实际上等于零。模型看起来卡住了。其实它没有卡住——它只是处在一个平坦区域，另一边就有可以继续下降的路。但 SGD 没有任何机制能推动它穿越过去。

Adam 同时解决了这三个问题。它为每个参数维护两个滑动平均——梯度的均值（动量，解决震荡问题）和梯度平方的均值（自适应学习率，解决尺度不一的问题）。再加上对前几步的偏差修正，你就得到了一个用默认超参数就能搞定 80% 问题的优化器。这节课会从零构建它，让你确切理解它在另外 20% 的情况下何时失效、为何失效。

## 核心概念

### 随机梯度下降（SGD）

最简单的优化器。在一个小批量上计算梯度，然后朝相反方向走一步。

```
w = w - lr * gradient
```

"随机"指的是你用数据的随机子集（小批量）来估计梯度，而不是整个数据集。这种噪声其实是有用的——它有助于逃离尖锐的局部极小值。但噪声同时也会引起震荡。

学习率是唯一的旋钮。太高：损失发散。太低：训练慢到没完没了。最优值取决于网络结构、数据、批量大小以及训练所处的阶段。对于现代网络上的朴素 SGD，典型取值在 0.01 到 0.1 之间。但即使在同一次训练运行中，理想的学习率也在不断变化。

### 动量（Momentum）

小球滚下山的类比虽被用滥，却很准确。你不再只按梯度走一步，而是维护一个累积历史梯度的速度量。

```
m_t = beta * m_{t-1} + gradient
w = w - lr * m_t
```

Beta（通常取 0.9）控制保留多少历史。当 beta = 0.9 时，动量大致等于最近 10 个梯度的平均值（1 / (1 - 0.9) = 10）。

为什么这能解决震荡：指向同一方向的梯度会累积，方向翻转的梯度会相互抵消。在那条狭窄的山谷里，"横向"分量每一步都在变号，于是被抑制；"纵向"分量保持一致，于是被放大。结果就是在有用的方向上平滑加速。

实际数字：在条件数很差的损失地形上，单独的 SGD 可能需要 10,000 步，而带动量的 SGD（beta=0.9）在同样的问题上通常只需 3,000 到 5,000 步。这个加速可不是边际收益。

### RMSProp

第一个真正可用的逐参数自适应学习率方法。由 Hinton 在一节 Coursera 课程中提出（从未正式发表）。

```
s_t = beta * s_{t-1} + (1 - beta) * gradient^2
w = w - lr * gradient / (sqrt(s_t) + epsilon)
```

s_t 跟踪梯度平方的滑动平均。梯度持续偏大的参数会被一个大数除（有效学习率变小），梯度偏小的参数会被一个小数除（有效学习率变大）。

这解决了"所有参数共用一个学习率"的问题。一个一直在大幅更新的权重很可能已经接近目标——让它慢下来；一个一直只有微小更新的权重可能训练不足——让它快起来。

Epsilon（通常取 1e-8）防止参数尚未更新时出现除零。

### Adam：动量 + RMSProp

Adam 把两个思路合二为一。它为每个参数维护两个指数滑动平均：

```
m_t = beta1 * m_{t-1} + (1 - beta1) * gradient        (first moment: mean)
v_t = beta2 * v_{t-1} + (1 - beta2) * gradient^2       (second moment: variance)
```

**偏差修正**是大多数讲解都略过的关键细节。在第 1 步时，m_1 = (1 - beta1) * gradient。当 beta1 = 0.9 时，这等于 0.1 * gradient——比实际值小了十倍。滑动平均还没有"热身"完成。偏差修正对此进行补偿：

```
m_hat = m_t / (1 - beta1^t)
v_hat = v_t / (1 - beta2^t)
```

第 1 步时，beta1 = 0.9：m_hat = m_1 / (1 - 0.9) = m_1 / 0.1 = 真实的梯度。第 100 步时，(1 - 0.9^100) 约等于 1.0，修正项基本消失。偏差修正在前约 10 步至关重要，约 50 步之后就无关紧要了。

更新公式：

```
w = w - lr * m_hat / (sqrt(v_hat) + epsilon)
```

Adam 的默认值：lr = 0.001、beta1 = 0.9、beta2 = 0.999、epsilon = 1e-8。这套默认值能应对 80% 的问题。当它们不起作用时，先调 lr，再调 beta2。几乎永远不需要动 beta1 和 epsilon。

### AdamW：把权重衰减做对

L2 正则化在损失上加一项 lambda * w^2。在朴素 SGD 中，这等价于权重衰减（weight decay，即每一步从权重中减去 lambda * w）。但在 Adam 中，这种等价性不成立。

Loshchilov 与 Hutter 的洞见在于：当你把 L2 加到损失里、再让 Adam 处理梯度时，自适应学习率也会缩放正则化项。梯度方差大的参数得到的正则化更少，方差小的参数得到的正则化更多。这不是你想要的——你希望正则化是均匀的，与梯度统计量无关。

AdamW 的解决办法是在 Adam 更新之后，把权重衰减直接施加在权重上：

```
w = w - lr * m_hat / (sqrt(v_hat) + epsilon) - lr * lambda * w
```

权重衰减项（lr * lambda * w）不会被 Adam 的自适应因子缩放。每个参数都按相同比例收缩。

这看起来像个小细节，但并不是。在几乎所有任务上，AdamW 都能收敛到比 Adam + L2 正则化更好的解。它是 PyTorch 中训练 Transformer、扩散模型以及大多数现代架构的默认优化器。BERT、GPT、LLaMA、Stable Diffusion——全都用 AdamW 训练。

### 学习率：最重要的超参数

```mermaid
graph TD
    LR["Learning Rate"] --> TooHigh["Too high (lr > 0.01)"]
    LR --> JustRight["Just right"]
    LR --> TooLow["Too low (lr < 0.00001)"]

    TooHigh --> Diverge["Loss explodes<br/>NaN weights<br/>Training crashes"]
    JustRight --> Converge["Loss decreases steadily<br/>Reaches good minimum<br/>Generalizes well"]
    TooLow --> Stall["Loss decreases slowly<br/>Gets stuck in suboptimal minimum<br/>Wastes compute"]

    JustRight --> Schedule["Usually needs scheduling"]
    Schedule --> Warmup["Warmup: ramp from 0 to max<br/>First 1-10% of training"]
    Schedule --> Decay["Decay: reduce over time<br/>Cosine or linear"]
```

如果只能调一个超参数，那就调学习率。学习率改变 10 倍，比你将做出的任何架构决策影响都大。常用默认值：

- SGD：lr = 0.01 到 0.1
- Adam/AdamW：lr = 1e-4 到 3e-4
- 微调预训练模型：lr = 1e-5 到 5e-5
- 学习率预热（warmup）：在前 1-10% 的训练步数内线性爬升

### 优化器对比

```mermaid
flowchart LR
    subgraph "Optimization Path"
        SGD_P["SGD<br/>Oscillates across valley<br/>Slow but finds flat minima"]
        Mom_P["SGD + Momentum<br/>Smoother path<br/>3x faster than SGD"]
        Adam_P["Adam<br/>Adapts per-parameter<br/>Fast convergence"]
        AdamW_P["AdamW<br/>Adam + proper decay<br/>Best generalization"]
    end
    SGD_P --> Mom_P --> Adam_P --> AdamW_P
```

### 各优化器的适用场景

```mermaid
flowchart TD
    Task["What are you training?"] --> Type{"Model type?"}

    Type -->|"Transformer / LLM"| AdamW["AdamW<br/>lr=1e-4, wd=0.01-0.1"]
    Type -->|"CNN / ResNet"| SGD_M["SGD + Momentum<br/>lr=0.1, momentum=0.9"]
    Type -->|"GAN"| Adam2["Adam<br/>lr=2e-4, beta1=0.5"]
    Type -->|"Fine-tuning"| AdamW2["AdamW<br/>lr=2e-5, wd=0.01"]
    Type -->|"Don't know yet"| Default["Start with AdamW<br/>lr=3e-4, wd=0.01"]
```

```figure
optimizer-trajectory
```

## 从零实现

### 第 1 步：朴素 SGD

```python
class SGD:
    def __init__(self, lr=0.01):
        self.lr = lr

    def step(self, params, grads):
        for i in range(len(params)):
            params[i] -= self.lr * grads[i]
```

### 第 2 步：带动量的 SGD

```python
class SGDMomentum:
    def __init__(self, lr=0.01, beta=0.9):
        self.lr = lr
        self.beta = beta
        self.velocities = None

    def step(self, params, grads):
        if self.velocities is None:
            self.velocities = [0.0] * len(params)
        for i in range(len(params)):
            self.velocities[i] = self.beta * self.velocities[i] + grads[i]
            params[i] -= self.lr * self.velocities[i]
```

### 第 3 步：Adam

```python
import math

class Adam:
    def __init__(self, lr=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m = None
        self.v = None
        self.t = 0

    def step(self, params, grads):
        if self.m is None:
            self.m = [0.0] * len(params)
            self.v = [0.0] * len(params)

        self.t += 1

        for i in range(len(params)):
            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * grads[i]
            self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * grads[i] ** 2

            m_hat = self.m[i] / (1 - self.beta1 ** self.t)
            v_hat = self.v[i] / (1 - self.beta2 ** self.t)

            params[i] -= self.lr * m_hat / (math.sqrt(v_hat) + self.epsilon)
```

### 第 4 步：AdamW

```python
class AdamW:
    def __init__(self, lr=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8, weight_decay=0.01):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.weight_decay = weight_decay
        self.m = None
        self.v = None
        self.t = 0

    def step(self, params, grads):
        if self.m is None:
            self.m = [0.0] * len(params)
            self.v = [0.0] * len(params)

        self.t += 1

        for i in range(len(params)):
            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * grads[i]
            self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * grads[i] ** 2

            m_hat = self.m[i] / (1 - self.beta1 ** self.t)
            v_hat = self.v[i] / (1 - self.beta2 ** self.t)

            params[i] -= self.lr * m_hat / (math.sqrt(v_hat) + self.epsilon)
            params[i] -= self.lr * self.weight_decay * params[i]
```

### 第 5 步：训练对比

用第 05 课的圆形数据集训练同一个两层网络，分别使用四种优化器，比较收敛情况。

```python
import random

def sigmoid(x):
    x = max(-500, min(500, x))
    return 1.0 / (1.0 + math.exp(-x))

def make_circle_data(n=200, seed=42):
    random.seed(seed)
    data = []
    for _ in range(n):
        x = random.uniform(-2, 2)
        y = random.uniform(-2, 2)
        label = 1.0 if x * x + y * y < 1.5 else 0.0
        data.append(([x, y], label))
    return data


class OptimizerTestNetwork:
    def __init__(self, optimizer, hidden_size=8):
        random.seed(0)
        self.hidden_size = hidden_size
        self.optimizer = optimizer

        self.w1 = [[random.gauss(0, 0.5) for _ in range(2)] for _ in range(hidden_size)]
        self.b1 = [0.0] * hidden_size
        self.w2 = [random.gauss(0, 0.5) for _ in range(hidden_size)]
        self.b2 = 0.0

    def get_params(self):
        params = []
        for row in self.w1:
            params.extend(row)
        params.extend(self.b1)
        params.extend(self.w2)
        params.append(self.b2)
        return params

    def set_params(self, params):
        idx = 0
        for i in range(self.hidden_size):
            for j in range(2):
                self.w1[i][j] = params[idx]
                idx += 1
        for i in range(self.hidden_size):
            self.b1[i] = params[idx]
            idx += 1
        for i in range(self.hidden_size):
            self.w2[i] = params[idx]
            idx += 1
        self.b2 = params[idx]

    def forward(self, x):
        self.x = x
        self.z1 = []
        self.h = []
        for i in range(self.hidden_size):
            z = self.w1[i][0] * x[0] + self.w1[i][1] * x[1] + self.b1[i]
            self.z1.append(z)
            self.h.append(max(0.0, z))

        self.z2 = sum(self.w2[i] * self.h[i] for i in range(self.hidden_size)) + self.b2
        self.out = sigmoid(self.z2)
        return self.out

    def compute_grads(self, target):
        eps = 1e-15
        p = max(eps, min(1 - eps, self.out))
        d_loss = -(target / p) + (1 - target) / (1 - p)
        d_sigmoid = self.out * (1 - self.out)
        d_out = d_loss * d_sigmoid

        grads = [0.0] * (self.hidden_size * 2 + self.hidden_size + self.hidden_size + 1)
        idx = 0
        for i in range(self.hidden_size):
            d_relu = 1.0 if self.z1[i] > 0 else 0.0
            d_h = d_out * self.w2[i] * d_relu
            grads[idx] = d_h * self.x[0]
            grads[idx + 1] = d_h * self.x[1]
            idx += 2

        for i in range(self.hidden_size):
            d_relu = 1.0 if self.z1[i] > 0 else 0.0
            grads[idx] = d_out * self.w2[i] * d_relu
            idx += 1

        for i in range(self.hidden_size):
            grads[idx] = d_out * self.h[i]
            idx += 1

        grads[idx] = d_out
        return grads

    def train(self, data, epochs=300):
        losses = []
        for epoch in range(epochs):
            total_loss = 0.0
            correct = 0
            for x, y in data:
                pred = self.forward(x)
                grads = self.compute_grads(y)
                params = self.get_params()
                self.optimizer.step(params, grads)
                self.set_params(params)

                eps = 1e-15
                p = max(eps, min(1 - eps, pred))
                total_loss += -(y * math.log(p) + (1 - y) * math.log(1 - p))
                if (pred >= 0.5) == (y >= 0.5):
                    correct += 1
            avg_loss = total_loss / len(data)
            accuracy = correct / len(data) * 100
            losses.append((avg_loss, accuracy))
            if epoch % 75 == 0 or epoch == epochs - 1:
                print(f"    Epoch {epoch:3d}: loss={avg_loss:.4f}, accuracy={accuracy:.1f}%")
        return losses
```

## 生产实践

PyTorch 的优化器内置了参数组、梯度裁剪和学习率调度：

```python
import torch
import torch.optim as optim

model = torch.nn.Sequential(
    torch.nn.Linear(784, 256),
    torch.nn.ReLU(),
    torch.nn.Linear(256, 10),
)

optimizer = optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)

scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)

for epoch in range(100):
    optimizer.zero_grad()
    output = model(torch.randn(32, 784))
    loss = torch.nn.functional.cross_entropy(output, torch.randint(0, 10, (32,)))
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()
    scheduler.step()
```

固定的模式永远是：zero_grad、前向、损失、反向、（裁剪）、step、（调度）。把这个顺序背下来。顺序弄错（例如在 optimizer.step() 之前调用 scheduler.step()）是隐蔽 bug 的常见来源。

训练 CNN 时，许多从业者仍然偏好 SGD + 动量（lr=0.1、momentum=0.9、weight_decay=1e-4），配合阶梯式或余弦调度。SGD 倾向于找到更平坦的极小值，泛化往往更好。训练 Transformer 和 LLM 时，AdamW 配合预热 + 余弦衰减是普遍默认方案。没有实测依据，就别和这个共识对着干。

## 交付产物

本课产出：
- `outputs/prompt-optimizer-selector.md` —— 一份决策提示词，用于为任意架构选择合适的优化器和学习率

## 练习

1. 实现 Nesterov 动量：在"前瞻"位置（w - lr * beta * v）而不是当前位置计算梯度。在圆形数据集上比较其与标准动量的收敛速度。

2. 实现一个学习率预热调度：在前 10% 的训练步数内从 0 线性爬升到 max_lr，之后余弦衰减到 0。分别用 Adam + 预热和不带预热的 Adam 训练，测量在圆形数据集上达到 90% 准确率各需要多少个 epoch。

3. 在 Adam 训练过程中跟踪每个参数的有效学习率。有效学习率为 lr * m_hat / (sqrt(v_hat) + eps)。绘制第 10、50、200 步之后有效学习率的分布。所有参数的更新速度都相同吗？

4. 实现梯度裁剪（按全局范数裁剪）。把最大梯度范数设为 1.0。用较高的学习率（Adam 取 lr=0.01）分别在有裁剪和无裁剪的情况下训练。在 10 个随机种子上统计有无裁剪时各有多少次运行发散（损失变成 NaN）。

5. 在一个权重较大的网络上比较 Adam 与 AdamW。把所有权重初始化为 [-5, 5] 内的随机值（远大于正常初始化）。用 weight_decay=0.1 训练 200 个 epoch。绘制两种优化器训练过程中权重 L2 范数的变化曲线。AdamW 应表现出更快的权重收缩。

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|----------------|----------------------|
| 学习率 | "步长" | 梯度更新上的标量乘数；训练中影响最大的单个超参数 |
| SGD | "基础梯度下降" | 随机梯度下降：在小批量上计算梯度，用权重减去 lr * gradient 来更新 |
| 动量 | "滚球类比" | 历史梯度的指数滑动平均；抑制震荡并加速方向一致的更新 |
| RMSProp | "自适应学习率" | 用每个参数近期梯度的滑动 RMS 去除该参数的梯度；起到均衡学习率的作用 |
| Adam | "默认优化器" | 结合动量（一阶矩）与 RMSProp（二阶矩），并对初始步数做偏差修正 |
| AdamW | "做对了的 Adam" | 带解耦权重衰减的 Adam；把正则化直接施加在权重上，而不是通过梯度 |
| 偏差修正 | "滑动平均的预热" | 除以 (1 - beta^t)，补偿 Adam 矩估计零初始化带来的偏差 |
| 权重衰减 | "收缩权重" | 每一步从权重中减去其一定比例；一种惩罚大权重的正则化手段 |
| 学习率调度 | "随时间改变 lr" | 训练过程中调整学习率的函数；预热 + 余弦衰减是现代默认方案 |
| 梯度裁剪 | "给梯度范数封顶" | 当梯度向量的范数超过阈值时按比例缩小；防止梯度爆炸式更新 |

## 延伸阅读

- Kingma & Ba, "Adam: A Method for Stochastic Optimization" (2014) —— Adam 原始论文，包含收敛性分析和偏差修正的推导
- Loshchilov & Hutter, "Decoupled Weight Decay Regularization" (2017) —— 证明了在 Adam 中 L2 正则化与权重衰减并不等价，并提出 AdamW
- Smith, "Cyclical Learning Rates for Training Neural Networks" (2017) —— 提出 LR range test 与循环学习率调度，免去了调固定学习率的麻烦
- Ruder, "An Overview of Gradient Descent Optimization Algorithms" (2016) —— 关于各类优化器变体最好的单篇综述，对比清晰、直觉到位
