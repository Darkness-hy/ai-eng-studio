# 支持向量机

> 在两个类别之间找出最宽的那条「街道」。这就是全部思想。

**Type:** Build
**Language:** Python
**Prerequisites:** Phase 1 (Lessons 08 Optimization, 14 Norms and Distances, 18 Convex Optimization)
**Time:** ~90 minutes

## 学习目标

- 基于原始（primal）形式，用合页损失（hinge loss）和梯度下降从零实现一个线性 SVM
- 解释最大间隔原则，并从训练好的模型中找出支持向量
- 比较线性核、多项式核和 RBF 核，并解释核技巧如何避免显式的高维映射
- 评估 C 参数在间隔宽度与分类错误之间所控制的权衡

## 问题背景

你有两类数据点，需要画一条线（或超平面）把它们分开。可行的线有无穷多条，该选哪一条？

选间隔最大的那条。间隔（margin）是决策边界与两侧最近数据点之间的距离。间隔越宽，分类器的判断越有把握，在未见数据上的泛化能力也越好。

这一直觉引出了支持向量机（Support Vector Machine, SVM），它是机器学习中数学上最优雅的算法之一。在深度学习兴起之前，SVM 一直是主流的分类方法；如今在小数据集、高维数据，以及需要原理清晰、有理论保证的模型的场景下，它仍然是最佳选择。

SVM 与 Phase 1 的内容直接相连：它的优化问题是凸的（第 18 课），间隔用范数度量（第 14 课），而核技巧利用点积来处理非线性边界，全程无需在高维空间中实际计算。

## 核心概念

### 最大间隔分类器

给定线性可分的数据，标签 y_i 取值于 {-1, +1}，特征向量为 x_i，我们要找一个能分开两类的超平面 w^T x + b = 0。

点 x_i 到超平面的距离为：

```
distance = |w^T x_i + b| / ||w||
```

对一个被正确分类的点有：y_i * (w^T x_i + b) > 0。间隔等于超平面到任一侧最近点距离的两倍。

```mermaid
graph LR
    subgraph Margin
        direction TB
        A["w^T x + b = +1"] ~~~ B["w^T x + b = 0"] ~~~ C["w^T x + b = -1"]
    end
    D["+ class points"] --> A
    E["- class points"] --> C
    B --- F["Decision boundary"]
```

优化问题为：

```
maximize    2 / ||w||     (the margin width)
subject to  y_i * (w^T x_i + b) >= 1  for all i
```

等价地（最小化 ||w||^2 更便于优化）：

```
minimize    (1/2) ||w||^2
subject to  y_i * (w^T x_i + b) >= 1  for all i
```

这是一个凸二次规划问题，存在唯一的全局解。恰好落在间隔边界上（即满足 y_i * (w^T x_i + b) = 1）的数据点就是支持向量（support vectors）。只有它们决定决策边界。移动或删除任何非支持向量的点，边界都不会改变。

### 支持向量：关键的少数

```mermaid
graph TD
    subgraph Classification
        SV1["Support Vector (+ class)<br>y(w'x+b) = 1"] --- DB["Decision Boundary<br>w'x+b = 0"]
        DB --- SV2["Support Vector (- class)<br>y(w'x+b) = 1"]
    end
    O1["Other + points<br>(do not affect boundary)"] -.-> SV1
    O2["Other - points<br>(do not affect boundary)"] -.-> SV2
```

绝大多数训练点都无关紧要，只有支持向量起作用。这正是 SVM 在预测时省内存的原因：你只需存储支持向量，而不是整个训练集。

支持向量的数量还给出了泛化误差的一个上界。支持向量相对数据集规模越少，泛化能力越好。

### 软间隔：用 C 参数处理噪声

真实数据很少能完美分开。有些点可能落在边界的错误一侧，或落在间隔内部。软间隔（soft margin）形式通过引入松弛变量来允许违例。

```
minimize    (1/2) ||w||^2 + C * sum(xi_i)
subject to  y_i * (w^T x_i + b) >= 1 - xi_i
            xi_i >= 0  for all i
```

松弛变量 xi_i 度量第 i 个点违反间隔约束的程度。C 控制其中的权衡：

| C 值 | 行为 |
|---------|----------|
| 大 C | 重罚违例。间隔窄，误分类少。容易过拟合 |
| 小 C | 允许更多违例。间隔宽，误分类多。容易欠拟合 |

C 相当于正则化强度的倒数。C 大 = 正则化弱，C 小 = 正则化强。

### 合页损失：SVM 的损失函数

软间隔 SVM 可以改写为一个无约束优化问题：

```
minimize    (1/2) ||w||^2 + C * sum(max(0, 1 - y_i * (w^T x_i + b)))
```

其中 max(0, 1 - y_i * f(x_i)) 就是合页损失（hinge loss）。当点被正确分类且位于间隔之外时，损失为零；当点位于间隔之内或被误分类时，损失呈线性增长。

```
Hinge loss for a single point:

loss
  |
  | \
  |  \
  |   \
  |    \
  |     \_______________
  |
  +-----|-----|-------->  y * f(x)
       0     1

Zero loss when y*f(x) >= 1 (correctly classified, outside margin).
Linear penalty when y*f(x) < 1.
```

对比逻辑损失（逻辑回归）：

```
Hinge:     max(0, 1 - y*f(x))          Hard cutoff at margin
Logistic:  log(1 + exp(-y*f(x)))        Smooth, never exactly zero
```

合页损失产生稀疏解（只有支持向量贡献非零项），而逻辑损失会用到所有数据点。这使 SVM 在预测时更省内存。

### 用梯度下降训练线性 SVM

你可以直接对「合页损失 + L2 正则化」做梯度下降来训练线性 SVM，无需求解带约束的二次规划：

```
L(w, b) = (lambda/2) * ||w||^2 + (1/n) * sum(max(0, 1 - y_i * (w^T x_i + b)))

Gradient with respect to w:
  If y_i * (w^T x_i + b) >= 1:  dL/dw = lambda * w
  If y_i * (w^T x_i + b) < 1:   dL/dw = lambda * w - y_i * x_i

Gradient with respect to b:
  If y_i * (w^T x_i + b) >= 1:  dL/db = 0
  If y_i * (w^T x_i + b) < 1:   dL/db = -y_i
```

这被称为原始形式（primal formulation）。它每个 epoch 的开销为 O(n * d)，其中 n 是样本数，d 是特征数。对大规模、稀疏、高维的数据（如文本分类），这种方法非常快。

### 对偶形式与核技巧

SVM 问题的拉格朗日对偶（见 Phase 1 第 18 课，KKT 条件）为：

```
maximize    sum(alpha_i) - (1/2) * sum_ij(alpha_i * alpha_j * y_i * y_j * (x_i . x_j))
subject to  0 <= alpha_i <= C
            sum(alpha_i * y_i) = 0
```

对偶形式只涉及数据点之间的点积 x_i . x_j。这就是关键洞察：把每个点积换成核函数 K(x_i, x_j)，SVM 就能学习非线性边界，而无需显式计算任何变换。

```
Linear kernel:      K(x, z) = x . z
Polynomial kernel:  K(x, z) = (x . z + c)^d
RBF (Gaussian):     K(x, z) = exp(-gamma * ||x - z||^2)
```

RBF 核把数据映射到一个无穷维空间。在输入空间里相近的点，核函数值接近 1；相距遥远的点，核函数值接近 0。它可以学习任意光滑的决策边界。

```mermaid
graph LR
    subgraph "Input Space (not separable)"
        A["Data points in 2D<br>circular boundary"]
    end
    subgraph "Feature Space (separable)"
        B["Data points in higher dim<br>linear boundary"]
    end
    A -->|"Kernel trick<br>K(x,z) = phi(x).phi(z)"| B
```

核技巧在不实际进入高维空间的情况下，计算出高维空间中的点积。以 D 维输入上的 d 次多项式核为例，显式特征空间有 O(D^d) 维，但 K(x, z) 只需 O(D) 时间即可算出。

### 用于回归的 SVM（SVR）

支持向量回归（Support Vector Regression, SVR）在数据周围拟合一条宽度为 epsilon 的「管道」。管内的点损失为零，管外的点受到线性惩罚。

```
minimize    (1/2) ||w||^2 + C * sum(xi_i + xi_i*)
subject to  y_i - (w^T x_i + b) <= epsilon + xi_i
            (w^T x_i + b) - y_i <= epsilon + xi_i*
            xi_i, xi_i* >= 0
```

epsilon 参数控制管道宽度。管道越宽 = 支持向量越少 = 拟合越平滑；管道越窄 = 支持向量越多 = 拟合越紧贴数据。

### SVM 为何输给深度学习（以及它仍然胜出的场景）

从 1990 年代末到 2010 年代初，SVM 一直主导机器学习领域。深度学习超越它有以下几个原因：

| 因素 | SVM | 深度学习 |
|--------|------|---------------|
| 特征工程 | 必须手工做 | 自动学习特征 |
| 可扩展性 | 核方法为 O(n^2) 到 O(n^3) | 用 SGD 每个 epoch 为 O(n) |
| 图像/文本/音频 | 需要手工特征 | 直接从原始数据学习 |
| 大数据集（>100k） | 慢 | 扩展性好 |
| GPU 加速 | 收益有限 | 大幅加速 |

SVM 在以下场景仍然胜出：
- 小数据集（几百到几千个样本）
- 高维稀疏数据（带 TF-IDF 特征的文本）
- 需要数学保证（间隔界）的时候
- 训练时间必须极短的时候（线性 SVM 非常快）
- 间隔结构清晰的二分类问题
- 异常检测（one-class SVM）

```figure
svm-margin
```

## 从零实现

### 第 1 步：合页损失及其梯度

这是基础。计算一个批次的合页损失及其梯度。

```python
def hinge_loss(X, y, w, b):
    n = len(X)
    total_loss = 0.0
    for i in range(n):
        margin = y[i] * (dot(w, X[i]) + b)
        total_loss += max(0.0, 1.0 - margin)
    return total_loss / n
```

### 第 2 步：用梯度下降实现线性 SVM

通过最小化带正则项的合页损失来训练，不需要 QP 求解器。

```python
class LinearSVM:
    def __init__(self, lr=0.001, lambda_param=0.01, n_epochs=1000):
        self.lr = lr
        self.lambda_param = lambda_param
        self.n_epochs = n_epochs
        self.w = None
        self.b = 0.0

    def fit(self, X, y):
        n_features = len(X[0])
        self.w = [0.0] * n_features
        self.b = 0.0

        for epoch in range(self.n_epochs):
            for i in range(len(X)):
                margin = y[i] * (dot(self.w, X[i]) + self.b)
                if margin >= 1:
                    self.w = [wj - self.lr * self.lambda_param * wj
                              for wj in self.w]
                else:
                    self.w = [wj - self.lr * (self.lambda_param * wj - y[i] * X[i][j])
                              for j, wj in enumerate(self.w)]
                    self.b -= self.lr * (-y[i])

    def predict(self, X):
        return [1 if dot(self.w, x) + self.b >= 0 else -1 for x in X]
```

### 第 3 步：核函数

实现线性核、多项式核和 RBF 核。

```python
def linear_kernel(x, z):
    return dot(x, z)

def polynomial_kernel(x, z, degree=3, c=1.0):
    return (dot(x, z) + c) ** degree

def rbf_kernel(x, z, gamma=0.5):
    diff = [xi - zi for xi, zi in zip(x, z)]
    return math.exp(-gamma * dot(diff, diff))
```

### 第 4 步：间隔与支持向量识别

训练结束后，找出哪些点是支持向量，并计算间隔宽度。

```python
def find_support_vectors(X, y, w, b, tol=1e-3):
    support_vectors = []
    for i in range(len(X)):
        margin = y[i] * (dot(w, X[i]) + b)
        if abs(margin - 1.0) < tol:
            support_vectors.append(i)
    return support_vectors
```

完整实现及全部演示见 `code/svm.py`。

## 生产实践

使用 scikit-learn：

```python
from sklearn.svm import SVC, LinearSVC, SVR
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

clf = Pipeline([
    ("scaler", StandardScaler()),
    ("svm", SVC(kernel="rbf", C=1.0, gamma="scale")),
])
clf.fit(X_train, y_train)
print(f"Accuracy: {clf.score(X_test, y_test):.4f}")
print(f"Support vectors: {clf['svm'].n_support_}")
```

重要提示：训练 SVM 前务必先对特征做缩放。SVM 对特征的数值量级很敏感，因为间隔依赖 ||w||，未缩放的特征会扭曲几何结构。

对大数据集，应使用 `LinearSVC`（原始形式，每个 epoch 为 O(n)）而不是 `SVC`（对偶形式，O(n^2) 到 O(n^3)）：

```python
from sklearn.svm import LinearSVC

clf = Pipeline([
    ("scaler", StandardScaler()),
    ("svm", LinearSVC(C=1.0, max_iter=10000)),
])
```

## 练习

1. 生成一个二维线性可分数据集。训练你的 LinearSVM 并找出支持向量。验证支持向量正是离决策边界最近的那些点。

2. 在一个含噪声的数据集上把 C 从 0.001 变到 1000。为每个 C 值画出决策边界。观察从宽间隔（欠拟合）到窄间隔（过拟合）的过渡。

3. 构造一个类别边界为圆形（非线性）的数据集。展示线性 SVM 在其上失效。计算 RBF 核矩阵，并展示两类数据在核诱导的特征空间中变得可分。

4. 在同一数据集上比较合页损失与逻辑损失。分别训练线性 SVM 和逻辑回归，统计有多少训练点参与决定各模型的决策边界（支持向量 vs 全部点）。

5. 实现 SVR（epsilon 不敏感损失）。将其拟合到 y = sin(x) + noise。画出预测周围的 epsilon 管道，并标出支持向量（管道外的点）。

## 关键术语

| 术语 | 实际含义 |
|------|----------------------|
| 支持向量（support vectors） | 离决策边界最近的训练点。决定超平面的仅有这些点 |
| 间隔（margin） | 决策边界与最近支持向量之间的距离。SVM 要最大化它 |
| 合页损失（hinge loss） | max(0, 1 - y*f(x))。正确分类且在间隔之外时为零，否则为线性惩罚 |
| C 参数 | 间隔宽度与分类错误之间的权衡。C 大 = 间隔窄，C 小 = 间隔宽 |
| 软间隔（soft margin） | 通过松弛变量允许间隔违例的 SVM 形式。可处理不可分数据 |
| 核技巧（kernel trick） | 在高维特征空间中计算点积，而无需显式映射到该空间 |
| 线性核 | K(x, z) = x . z。等价于普通点积。适用于线性可分数据 |
| RBF 核 | K(x, z) = exp(-gamma * \|\|x-z\|\|^2)。映射到无穷维。可学习任意光滑边界 |
| 多项式核 | K(x, z) = (x . z + c)^d。映射到由多项式组合构成的特征空间 |
| 对偶形式（dual formulation） | SVM 问题的一种重写，只依赖数据点之间的点积。使核方法成为可能 |
| SVR | 支持向量回归。在数据周围拟合一条 epsilon 管道。管内的点损失为零 |
| 松弛变量（slack variables） | xi_i：度量一个点违反间隔约束的程度。对间隔之外被正确分类的点为零 |
| 最大间隔（maximum margin） | 选择能最大化到两类最近点距离的超平面这一原则 |

## 延伸阅读

- [Vapnik: The Nature of Statistical Learning Theory (1995)](https://link.springer.com/book/10.1007/978-1-4757-3264-1) - SVM 与统计学习理论的奠基之作
- [Cortes & Vapnik: Support-vector networks (1995)](https://link.springer.com/article/10.1007/BF00994018) - SVM 的原始论文
- [Platt: Sequential Minimal Optimization (1998)](https://www.microsoft.com/en-us/research/publication/sequential-minimal-optimization-a-fast-algorithm-for-training-support-vector-machines/) - 让 SVM 训练变得实用的 SMO 算法
- [scikit-learn SVM documentation](https://scikit-learn.org/stable/modules/svm.html) - 包含实现细节的实用指南
- [LIBSVM: A Library for Support Vector Machines](https://www.csie.ntu.edu.tw/~cjlin/libsvm/) - 支撑大多数 SVM 实现的 C++ 库
