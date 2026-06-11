# JAX 入门

> PyTorch 原地修改张量，TensorFlow 构建计算图，而 JAX 编译纯函数。最后这一种方式会改变你思考深度学习的方式。

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 03 Lessons 01-10, basic NumPy
**Time:** ~90 minutes

## 学习目标

- 使用 JAX 的函数式 API（jax.numpy、jax.grad、jax.jit、jax.vmap）编写纯函数风格的神经网络代码
- 解释 PyTorch 的即时执行加原地修改模式与 JAX 的函数式编译模型之间的关键设计差异
- 应用 jit 编译和 vmap 向量化来加速训练循环，并与朴素 Python 实现对比
- 在 JAX 中训练一个简单网络，并将其显式状态管理与 PyTorch 的面向对象方式进行对比

## 问题背景

你已经会用 PyTorch 构建神经网络了：定义一个 `nn.Module`，调用 `.backward()`，再让优化器走一步。它能用，而且有数百万人在用。

但 PyTorch 的 DNA 里刻着一个约束：它在 Python 中即时地、逐个地追踪操作。每一次 `tensor + tensor` 都是一次独立的核函数（kernel）启动。每个训练步都要重新解释同样的 Python 代码。在一般场景下这没问题——直到你需要在 2,048 块 TPU 上训练一个 5400 亿参数的模型，这些开销就会要了你的命。

Google DeepMind 用 JAX 训练 Gemini，Anthropic 用 JAX 训练了 Claude。这些可不是小打小闹——它们是地球上规模最大的神经网络训练任务。他们选择 JAX，是因为它把你的训练循环当作一个可编译的程序，而不是一串 Python 调用。

JAX 就是带有三种超能力的 NumPy：自动微分、面向 XLA 的 JIT 编译，以及自动向量化。你只需写一个处理单个样本的函数，JAX 就能给你一个能处理整个批次、计算梯度、编译为机器码并在多设备上运行的函数——而原始函数一个字都不用改。

## 核心概念

### JAX 的哲学

JAX 是一个函数式框架。没有类，没有可变状态，没有 `.backward()` 方法。取而代之的是：

| PyTorch | JAX |
|---------|-----|
| 带状态的 `nn.Module` 类 | 纯函数：`f(params, x) -> y` |
| `loss.backward()` | `jax.grad(loss_fn)(params, x, y)` |
| 即时执行 | 通过 XLA 进行 JIT 编译 |
| `for x in batch:` 手动循环 | `jax.vmap(f)` 自动向量化 |
| `DataParallel` / `FSDP` | `jax.pmap(f)` 自动并行 |
| 可变的 `model.parameters()` | 由数组构成的不可变 pytree |

这不是代码风格上的偏好，而是编译器的硬性约束。JIT 编译要求纯函数（pure function）——相同输入永远产生相同输出，没有副作用。正是这个限制让 100 倍的加速成为可能。

### jax.numpy：熟悉的外表

JAX 在加速器上重新实现了 NumPy 的 API：

```python
import jax.numpy as jnp

a = jnp.array([1.0, 2.0, 3.0])
b = jnp.array([4.0, 5.0, 6.0])
c = jnp.dot(a, b)
```

同样的函数名，同样的广播规则，同样的切片语义。但这些数组驻留在 GPU/TPU 上，而且每个操作都可以被编译器追踪。

一个关键区别：JAX 数组是不可变的。不能写 `a[0] = 5`，要写 `a = a.at[0].set(5)`。开头一周你会觉得别扭，然后某天突然就想通了——不可变性正是让 `grad`、`jit`、`vmap` 这些变换可以任意组合的根基。

### jax.grad：函数式自动微分

PyTorch 把梯度挂在张量上（`.grad`），JAX 把梯度挂在函数上。

```python
import jax

def f(x):
    return x ** 2

df = jax.grad(f)
df(3.0)
```

`jax.grad` 接收一个函数，返回一个计算梯度的新函数。不需要调用 `.backward()`，也没有存储在张量上的计算图。梯度本身就是另一个函数，你可以直接调用它、组合它，或者对它做 JIT 编译。

这种组合可以任意嵌套：

```python
d2f = jax.grad(jax.grad(f))
d2f(3.0)
```

二阶导数、三阶导数、Jacobian 矩阵、Hessian 矩阵——全都靠组合 `grad` 得到。PyTorch 也能做到（`torch.autograd.functional.hessian`），但那是后期补上的功能；在 JAX 里，这就是地基。

约束在于：`grad` 只对纯函数有效。函数内部不能有 print 语句（它们会在追踪阶段执行，而不是真正运行时）；不能修改外部状态；不能在没有显式 key 管理的情况下生成随机数。

### jit：编译到 XLA

```python
@jax.jit
def train_step(params, x, y):
    loss = loss_fn(params, x, y)
    return loss

fast_step = jax.jit(train_step)
```

第一次调用时，JAX 会追踪（trace）这个函数——记录下发生了哪些操作，但并不真正执行。然后它把这份追踪记录交给 XLA（Accelerated Linear Algebra），即 Google 为 TPU 和 GPU 打造的编译器。XLA 会融合操作、消除冗余的内存拷贝，并生成优化过的机器码。

后续调用会完全绕开 Python，编译后的代码以 C++ 级别的速度直接在加速器上运行。

JIT 有帮助的场景：
- 训练步（同样的计算重复执行成千上万次）
- 推理（同一个模型，不同的输入）
- 任何会以相似形状的输入被多次调用的函数

JIT 有害的场景：
- 含有依赖具体数值的 Python 控制流的函数（比如 `if x > 0`，而 x 是一个被追踪的数组）
- 一次性计算（编译开销超过运行时间）
- 调试（追踪机制会掩盖实际的执行过程）

控制流限制是实打实存在的：`jax.lax.cond` 取代 `if/else`，`jax.lax.scan` 取代 `for` 循环。这些不是可选项——它们是编译必须付出的代价。

### vmap：自动向量化

你写一个处理单个样本的函数：

```python
def predict(params, x):
    return jnp.dot(params['w'], x) + params['b']
```

`vmap` 把它提升为可以处理整个批次的函数：

```python
batch_predict = jax.vmap(predict, in_axes=(None, 0))
```

`in_axes=(None, 0)` 的意思是：`params` 不参与批处理（所有样本共享），而沿 `x` 的第 0 轴做批处理。不用手写 `for` 循环，不用 reshape，也不用在每一层手动传递批次维度。JAX 会自己识别批次维度并把整个计算向量化。

这不是语法糖。`vmap` 生成的是融合后的向量化代码，比 Python 循环快 10 到 100 倍。而且它还能和 `jit`、`grad` 自由组合：

```python
per_example_grads = jax.vmap(jax.grad(loss_fn), in_axes=(None, 0, 0))
```

逐样本梯度（per-example gradients），一行搞定。在 PyTorch 里，不靠各种 hack 几乎不可能做到这一点。

### pmap：跨设备数据并行

```python
parallel_step = jax.pmap(train_step, axis_name='devices')
```

`pmap` 会把函数复制到所有可用设备（GPU/TPU）上并拆分批次。在函数内部，`jax.lax.pmean` 和 `jax.lax.psum` 负责跨设备同步梯度。

Google 用 `pmap`（及其后继者 `shard_map`）在数千块 TPU v5e 芯片上训练 Gemini。编程模型很简单：写好单设备版本，用 `pmap` 包一层，完事。

### Pytree：通用数据结构

JAX 的操作对象是「pytree」——由列表、元组、字典和数组任意嵌套组合而成的结构。你的模型参数就是一个 pytree：

```python
params = {
    'layer1': {'w': jnp.zeros((784, 256)), 'b': jnp.zeros(256)},
    'layer2': {'w': jnp.zeros((256, 128)), 'b': jnp.zeros(128)},
    'layer3': {'w': jnp.zeros((128, 10)),  'b': jnp.zeros(10)},
}
```

JAX 的每一种变换——`grad`、`jit`、`vmap`——都知道如何遍历 pytree。`jax.tree.map(f, tree)` 会把 `f` 应用到每一个叶子节点。优化器就是这样一次性更新全部参数的：

```python
params = jax.tree.map(lambda p, g: p - lr * g, params, grads)
```

没有 `.parameters()` 方法，也不需要参数注册。树结构本身就是模型。

### 函数式 vs 面向对象

PyTorch 把状态存在对象内部：

```python
class Model(nn.Module):
    def __init__(self):
        self.linear = nn.Linear(784, 10)

    def forward(self, x):
        return self.linear(x)
```

JAX 使用纯函数加显式状态：

```python
def predict(params, x):
    return jnp.dot(x, params['w']) + params['b']
```

参数通过传参进入函数，没有任何东西被存储，也没有任何东西被修改。这让每个函数都可测试、可组合、可编译。代价是参数得由你自己管理——或者交给 Flax、Equinox 这类库。

### JAX 生态

JAX 提供原语，生态库提供易用性：

| 库 | 角色 | 风格 |
|---------|------|-------|
| **Flax**（Google） | 神经网络层 | 带显式状态的 `nn.Module` |
| **Equinox**（Patrick Kidger） | 神经网络层 | 基于 pytree，更 Pythonic |
| **Optax**（DeepMind） | 优化器 + 学习率调度 | 可组合的梯度变换 |
| **Orbax**（Google） | 检查点管理 | 保存/恢复 pytree |
| **CLU**（Google） | 指标 + 日志 | 训练循环工具集 |

Optax 是标准优化器库。它把梯度变换（Adam、SGD、裁剪）与参数更新解耦，使得组合变得轻而易举：

```python
optimizer = optax.chain(
    optax.clip_by_global_norm(1.0),
    optax.adam(learning_rate=1e-3),
)
```

### 何时用 JAX、何时用 PyTorch

| 因素 | JAX | PyTorch |
|--------|-----|---------|
| TPU 支持 | 一等公民（两者都是 Google 出品） | 社区维护（torch_xla） |
| GPU 支持 | 良好（通过 XLA 走 CUDA） | 业界最佳（原生 CUDA） |
| 调试 | 困难（追踪 + 编译） | 容易（即时执行，可逐行排查） |
| 生态 | 偏研究（Flax、Equinox） | 庞大（HuggingFace、torchvision 等） |
| 招聘市场 | 小众（Google/DeepMind/Anthropic） | 主流（无处不在） |
| 大规模训练 | 更优（XLA、pmap、mesh） | 良好（FSDP、DeepSpeed） |
| 原型开发速度 | 较慢（函数式的额外负担） | 较快（改了就跑） |
| 生产推理 | TensorFlow Serving、Vertex AI | TorchServe、Triton、ONNX |
| 谁在用 | DeepMind（Gemini）、Anthropic（Claude） | Meta（Llama）、OpenAI（GPT）、Stability AI |

实话实说：除非有明确理由，否则就用 PyTorch。这些理由包括——能用上 TPU、需要逐样本梯度、要做超大规模多设备训练，或者你在 Google/DeepMind/Anthropic 工作。

### JAX 中的随机数

JAX 没有全局随机状态。每一个随机操作都需要显式传入 PRNG key：

```python
key = jax.random.PRNGKey(42)
key1, key2 = jax.random.split(key)
w = jax.random.normal(key1, shape=(784, 256))
```

一开始确实烦人。但它保证了跨设备、跨编译的可复现性——这是 PyTorch 的 `torch.manual_seed` 在多 GPU 场景下无法保证的性质。

```figure
batchnorm-effect
```

## 从零实现

### 第 1 步：环境与数据

我们将用 JAX 和 Optax 在 MNIST 上训练一个 3 层 MLP：784 个输入，两个分别为 256 和 128 个神经元的隐藏层，10 个输出类别。

```python
import jax
import jax.numpy as jnp
from jax import random
import optax

def get_mnist_data():
    from sklearn.datasets import fetch_openml
    mnist = fetch_openml('mnist_784', version=1, as_frame=False, parser='auto')
    X = mnist.data.astype('float32') / 255.0
    y = mnist.target.astype('int')
    X_train, X_test = X[:60000], X[60000:]
    y_train, y_test = y[:60000], y[60000:]
    return X_train, y_train, X_test, y_test
```

### 第 2 步：初始化参数

不用类，只需要一个返回 pytree 的函数：

```python
def init_params(key):
    k1, k2, k3 = random.split(key, 3)
    scale1 = jnp.sqrt(2.0 / 784)
    scale2 = jnp.sqrt(2.0 / 256)
    scale3 = jnp.sqrt(2.0 / 128)
    params = {
        'layer1': {
            'w': scale1 * random.normal(k1, (784, 256)),
            'b': jnp.zeros(256),
        },
        'layer2': {
            'w': scale2 * random.normal(k2, (256, 128)),
            'b': jnp.zeros(128),
        },
        'layer3': {
            'w': scale3 * random.normal(k3, (128, 10)),
            'b': jnp.zeros(10),
        },
    }
    return params
```

手工实现的 He 初始化。从一个种子拆分出三个 PRNG key。每个权重都是嵌套字典中一个不可变的数组。

### 第 3 步：前向传播

```python
def forward(params, x):
    x = jnp.dot(x, params['layer1']['w']) + params['layer1']['b']
    x = jax.nn.relu(x)
    x = jnp.dot(x, params['layer2']['w']) + params['layer2']['b']
    x = jax.nn.relu(x)
    x = jnp.dot(x, params['layer3']['w']) + params['layer3']['b']
    return x

def loss_fn(params, x, y):
    logits = forward(params, x)
    one_hot = jax.nn.one_hot(y, 10)
    return -jnp.mean(jnp.sum(jax.nn.log_softmax(logits) * one_hot, axis=-1))
```

纯函数。参数进，预测出。没有 `self`，没有内部存储的状态。`loss_fn` 从零计算交叉熵——softmax、取对数、取负均值。

### 第 4 步：JIT 编译的训练步

```python
@jax.jit
def train_step(params, opt_state, x, y):
    loss, grads = jax.value_and_grad(loss_fn)(params, x, y)
    updates, opt_state = optimizer.update(grads, opt_state, params)
    params = optax.apply_updates(params, updates)
    return params, opt_state, loss

@jax.jit
def accuracy(params, x, y):
    logits = forward(params, x)
    preds = jnp.argmax(logits, axis=-1)
    return jnp.mean(preds == y)
```

`jax.value_and_grad` 在一次计算中同时返回损失值和梯度。`@jax.jit` 装饰器把这两个函数都编译到 XLA。首次调用之后，每个训练步的执行都不再经过 Python。

### 第 5 步：训练循环

```python
optimizer = optax.adam(learning_rate=1e-3)

X_train, y_train, X_test, y_test = get_mnist_data()
X_train, X_test = jnp.array(X_train), jnp.array(X_test)
y_train, y_test = jnp.array(y_train), jnp.array(y_test)

key = random.PRNGKey(0)
params = init_params(key)
opt_state = optimizer.init(params)

batch_size = 128
n_epochs = 10

for epoch in range(n_epochs):
    key, subkey = random.split(key)
    perm = random.permutation(subkey, len(X_train))
    X_shuffled = X_train[perm]
    y_shuffled = y_train[perm]

    epoch_loss = 0.0
    n_batches = len(X_train) // batch_size
    for i in range(n_batches):
        start = i * batch_size
        xb = X_shuffled[start:start + batch_size]
        yb = y_shuffled[start:start + batch_size]
        params, opt_state, loss = train_step(params, opt_state, xb, yb)
        epoch_loss += loss

    train_acc = accuracy(params, X_train[:5000], y_train[:5000])
    test_acc = accuracy(params, X_test, y_test)
    print(f"Epoch {epoch + 1:2d} | Loss: {epoch_loss / n_batches:.4f} | "
          f"Train Acc: {train_acc:.4f} | Test Acc: {test_acc:.4f}")
```

10 个 epoch，测试准确率约 97%。第一个 epoch 较慢（JIT 编译），第 2 到第 10 个 epoch 很快。

注意少了什么：没有 `.zero_grad()`，没有 `.backward()`，没有 `.step()`。整个更新就是一次组合好的函数调用。梯度的计算、Adam 的变换、参数的更新——全都发生在 `train_step` 内部。

## 生产实践

### Flax：Google 的标准库

Flax 是最常用的 JAX 神经网络库。它把 `nn.Module` 加了回来，但状态管理是显式的：

```python
import flax.linen as nn

class MLP(nn.Module):
    @nn.compact
    def __call__(self, x):
        x = nn.Dense(256)(x)
        x = nn.relu(x)
        x = nn.Dense(128)(x)
        x = nn.relu(x)
        x = nn.Dense(10)(x)
        return x

model = MLP()
params = model.init(jax.random.PRNGKey(0), jnp.ones((1, 784)))
logits = model.apply(params, x_batch)
```

结构和 PyTorch 一样，但 `params` 与模型分离。`model.init()` 创建参数，`model.apply(params, x)` 执行前向传播。模型对象本身不持有任何状态。

### Equinox：更 Pythonic 的选择

Equinox（Patrick Kidger 开发）把模型本身表示成 pytree：

```python
import equinox as eqx

model = eqx.nn.MLP(
    in_size=784, out_size=10, width_size=256, depth=2,
    activation=jax.nn.relu, key=jax.random.PRNGKey(0)
)
logits = model(x)
```

模型自己就是一个 pytree，不需要 `.apply()`。参数就是模型这棵树的叶子节点。这更贴近 JAX 本身的思维方式。

### Optax：可组合的优化器

Optax 把梯度变换与参数更新解耦：

```python
schedule = optax.warmup_cosine_decay_schedule(
    init_value=0.0, peak_value=1e-3,
    warmup_steps=1000, decay_steps=50000
)

optimizer = optax.chain(
    optax.clip_by_global_norm(1.0),
    optax.adamw(learning_rate=schedule, weight_decay=0.01),
)
```

梯度裁剪、学习率预热、权重衰减——全部组合成一条变换链。每个变换接收梯度、修改梯度，再传给下一个。没有大而全的优化器类。

## 交付产物

**安装：**

```bash
pip install jax jaxlib optax flax
```

GPU 支持：

```bash
pip install jax[cuda12]
```

TPU（Google Cloud）：

```bash
pip install jax[tpu] -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
```

**性能方面的注意事项：**

- 首次 JIT 调用很慢（在编译）。做基准测试前先预热。
- 避免在 JIT 内部用 Python 循环遍历 JAX 数组，改用 `jax.lax.scan` 或 `jax.lax.fori_loop`。
- `jax.debug.print()` 在 JIT 内部能用，普通的 `print()` 不能。
- 用 `jax.profiler` 或 TensorBoard 做性能分析。XLA 编译可能掩盖瓶颈。
- JAX 默认预分配 75% 的 GPU 显存。设置 `XLA_PYTHON_CLIENT_PREALLOCATE=false` 可以关闭这一行为。

**检查点：**

```python
import orbax.checkpoint as ocp
checkpointer = ocp.PyTreeCheckpointer()
checkpointer.save('/tmp/model', params)
restored = checkpointer.restore('/tmp/model')
```

**本课产出：**
- `outputs/prompt-jax-optimizer.md` —— 一个用于选择合适 JAX 优化器配置的提示词
- `outputs/skill-jax-patterns.md` —— 一个涵盖 JAX 函数式模式的技能文档

## 练习

1. 给 MLP 加上 dropout。在 JAX 中，dropout 需要一个 PRNG key——把 key 贯穿前向传播，并为每个 dropout 层拆分一个子 key。对比加与不加 dropout 时的测试准确率。

2. 用 `jax.vmap` 为一个包含 32 张 MNIST 图像的批次计算逐样本梯度。计算每个样本的梯度范数。哪些样本的梯度最大？为什么？

3. 把手写的前向函数替换为一个适用于任意层数的通用函数 `mlp_forward(params, x)`。用 `jax.tree.leaves` 自动确定网络深度。

4. 对比有无 `@jax.jit` 时训练步的性能。各计时 100 步。在你的硬件上加速比有多大？首次调用的编译开销是多少？

5. 通过组合 `optax.chain(optax.clip_by_global_norm(1.0), optax.adam(1e-3))` 实现梯度裁剪。分别在有无裁剪的情况下训练，并绘制训练过程中梯度范数的变化曲线来观察效果。

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|----------------|----------------------|
| XLA | "让 JAX 变快的那个东西" | Accelerated Linear Algebra——一个能融合操作、从计算图生成优化后 GPU/TPU 核函数的编译器 |
| JIT | "即时编译" | JAX 在首次调用时追踪函数并编译到 XLA，后续调用直接运行编译好的版本 |
| 纯函数 | "没有副作用" | 输出只取决于输入的函数——没有全局状态，没有原地修改，没有不带显式 key 的随机性 |
| vmap | "自动批处理" | 把处理单个样本的函数变换成处理整个批次的函数，无需重写代码 |
| pmap | "自动并行" | 把函数复制到多个设备上并拆分输入批次 |
| Pytree | "嵌套的数组字典" | 任何由列表、元组、字典和数组嵌套而成、JAX 能遍历和变换的结构 |
| 追踪（Tracing） | "记录计算过程" | JAX 用抽象值执行函数以构建计算图，但不计算真实结果 |
| 函数式自动微分 | "对函数求 grad" | 通过变换函数来计算导数，而不是在张量上附加梯度存储 |
| Optax | "JAX 的优化器库" | 一个由可组合梯度变换构成的库——Adam、SGD、裁剪、调度——可以串成链 |
| Flax | "JAX 版的 nn.Module" | Google 为 JAX 打造的神经网络库，提供层抽象的同时保持状态显式 |

## 延伸阅读

- JAX 文档：https://jax.readthedocs.io/ —— 官方文档，包含关于 grad、jit 和 vmap 的出色教程
- "JAX: composable transformations of Python+NumPy programs"（Bradbury et al., 2018）—— 阐述设计哲学的原始论文
- Flax 文档：https://flax.readthedocs.io/ —— Google 为 JAX 打造的神经网络库
- Patrick Kidger, "Equinox: neural networks in JAX via callable PyTrees and filtered transformations"（2021）—— Flax 之外更 Pythonic 的选择
- DeepMind, "Optax: composable gradient transformation and optimisation" —— 标准优化器库
- "You Don't Know JAX"（Colin Raffel, 2020）—— 一份关于 JAX 各种坑和模式的实用指南，作者是 T5 论文作者之一
