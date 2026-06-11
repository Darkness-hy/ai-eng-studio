# 范数与距离

> 你的距离函数定义了「相似」的含义。选错了，下游的一切都会崩塌。

**Type:** Build
**Language:** Python
**Prerequisites:** Phase 1, Lessons 01 (Linear Algebra Intuition), 02 (Vectors, Matrices & Operations)
**Time:** ~90 minutes

## 学习目标

- 从零实现 L1、L2、余弦、马氏（Mahalanobis）、Jaccard 和编辑距离函数
- 为给定的机器学习任务选择合适的距离度量，并解释其他选项为何失效
- 将 L1 和 L2 范数与 LASSO 和 Ridge 正则化及其几何约束区域联系起来
- 演示同一份数据集在不同度量下会产生不同的最近邻

## 问题背景

你手上有两个向量。可能是词嵌入，可能是用户画像，也可能是像素数组。你需要知道：它们有多接近？

答案完全取决于你选择哪个距离函数。两个数据点在一种度量下可能互为最近邻，而在另一种度量下却相距甚远。你的 KNN 分类器、推荐引擎、向量数据库、聚类算法、损失函数——它们全都依赖这个选择。选错了，你的模型就会朝着错误的目标优化。

不存在普适的最佳距离。L2 适合空间数据。余弦相似度在 NLP 中占主导地位。Jaccard 处理集合。编辑距离处理字符串。马氏距离考虑特征间的相关性。Wasserstein 搬运概率质量。每一种都编码了一种对「相似」含义的不同假设。

本课将从零构建所有主要的距离函数，告诉你每一种在什么场景下是正确的工具，并演示同一份数据如何因度量不同而产生完全不同的最近邻。

## 核心概念

### 范数：度量向量的大小

范数（norm）度量一个向量的「大小」。任意两个向量之间的距离函数都可以写成它们之差的范数：d(a, b) = ||a - b||。所以理解范数就是理解距离。

### L1 范数（曼哈顿距离）

L1 范数是所有分量绝对值之和。

```
||x||_1 = |x_1| + |x_2| + ... + |x_n|
```

它被称为曼哈顿距离，因为它度量的是在城市街区网格上行走的距离——你只能沿着坐标轴移动，不能走对角线。

```
Point A = (1, 1)
Point B = (4, 5)

L1 distance = |4-1| + |5-1| = 3 + 4 = 7

On a grid, you walk 3 blocks east and 4 blocks north.
```

何时使用 L1：

- 高维稀疏数据（文本特征、独热编码）
- 需要对离群值保持稳健时（单个巨大的差异不会主导结果）
- 特征选择问题（L1 正则化促进稀疏性）

与 L1 正则化（Lasso）的联系：在损失函数中加入 ||w||_1，会惩罚权重绝对值之和。这会把较小的权重直接压到零，自动完成特征选择。L1 惩罚在权重空间中形成菱形约束区域，而菱形的尖角正好落在坐标轴上——那里某些权重恰好为零。

与损失函数的联系：平均绝对误差（MAE）就是预测值与目标值之间 L1 距离的平均。它对所有误差线性惩罚，因此相比 MSE 对离群值更加稳健。

### L2 范数（欧氏距离）

L2 范数是直线距离：各分量平方和的平方根。

```
||x||_2 = sqrt(x_1^2 + x_2^2 + ... + x_n^2)
```

这就是你在几何课上学到的距离——n 维空间中的勾股定理。

```
Point A = (1, 1)
Point B = (4, 5)

L2 distance = sqrt((4-1)^2 + (5-1)^2) = sqrt(9 + 16) = sqrt(25) = 5.0

The straight line, cutting diagonally through the grid.
```

何时使用 L2：

- 中低维度的连续数据
- 特征量纲可比较时
- 物理距离（空间数据、传感器读数）
- 像素级别的图像相似度

与 L2 正则化（Ridge）的联系：在损失函数中加入 ||w||_2^2，会惩罚较大的权重。与 L1 不同，它不会把权重压到零，而是让所有权重按比例向零收缩。L2 惩罚形成圆形约束区域，没有落在坐标轴上的尖角。权重会变小，但很少恰好为零。

与损失函数的联系：均方误差（MSE）是 L2 距离平方的平均。平方运算让大误差受到的惩罚远重于小误差。

```
MAE (L1 loss):  |y - y_hat|         Linear penalty. Robust to outliers.
MSE (L2 loss):  (y - y_hat)^2       Quadratic penalty. Sensitive to outliers.
```

### Lp 范数：一般化的家族

L1 和 L2 都是 Lp 范数的特例：

```
||x||_p = (|x_1|^p + |x_2|^p + ... + |x_n|^p)^(1/p)
```

不同的 p 值会产生形状不同的「单位球」（即到原点距离为 1 的所有点的集合）：

```
p=1:    Diamond shape      (corners on axes)
p=2:    Circle/sphere      (the usual round ball)
p=3:    Superellipse       (rounded square)
p=inf:  Square/hypercube   (flat sides along axes)
```

### L-infinity 范数（切比雪夫距离）

当 p 趋向无穷大时，Lp 范数收敛为分量绝对值的最大值。

```
||x||_inf = max(|x_1|, |x_2|, ..., |x_n|)
```

两点之间的距离由差异最大的那一个维度决定，其他维度全部被忽略。

```
Point A = (1, 1)
Point B = (4, 5)

L-inf distance = max(|4-1|, |5-1|) = max(3, 4) = 4
```

何时使用 L-infinity：

- 当任意单个维度的最坏偏差才是关键时
- 棋盘类游戏（国际象棋中的王按 L-infinity 移动：朝任意方向走一步代价都是 1）
- 制造公差（每个尺寸都必须在规格范围内）

### 余弦相似度与余弦距离

余弦相似度度量两个向量之间的夹角，忽略它们的大小。

```
cos_sim(a, b) = (a . b) / (||a||_2 * ||b||_2)
```

它的取值范围从 -1（方向相反）到 +1（方向相同）。相互垂直的向量余弦相似度为 0。

余弦距离把它转换为一种距离：cosine_distance = 1 - cosine_similarity。取值范围从 0（方向完全相同）到 2（方向完全相反）。

```
a = (1, 0)    b = (1, 1)

cos_sim = (1*1 + 0*1) / (1 * sqrt(2)) = 1/sqrt(2) = 0.707
cos_dist = 1 - 0.707 = 0.293
```

为什么余弦在 NLP 和嵌入领域占主导地位：在文本场景中，文档长度不应影响相似度。一篇讲猫的文档即使比另一篇讲猫的文档长一倍，两者也应当依然「相似」。余弦相似度忽略大小（长度），只关心方向。两篇词频分布相同但长度不同的文档指向同一方向，余弦相似度为 1.0。

何时使用余弦相似度：

- 文本相似度（TF-IDF 向量、词嵌入、句子嵌入）
- 任何「大小是噪声、方向是信号」的领域
- 推荐系统（用户偏好向量）
- 嵌入检索（向量数据库几乎总是使用余弦或点积）

### 点积相似度 vs 余弦相似度

两个向量的点积是：

```
a . b = a_1*b_1 + a_2*b_2 + ... + a_n*b_n
      = ||a|| * ||b|| * cos(angle)
```

余弦相似度就是用两个向量的模长归一化后的点积。当两个向量都已做单位归一化（模长 = 1）时，点积和余弦相似度完全相同。

```
If ||a|| = 1 and ||b|| = 1:
    a . b = cos(angle between a and b)
```

二者何时不同：点积包含大小信息。模长更大的向量会得到更高的点积分数。在某些检索系统中这一点很重要——你可能希望「热门」条目排名更靠前，此时模长充当了隐式的质量或重要性信号。

```
a = (3, 0)    b = (1, 0)    c = (0, 1)

dot(a, b) = 3     dot(a, c) = 0
cos(a, b) = 1.0   cos(a, c) = 0.0

Both agree on direction, but dot product also reflects magnitude.
```

实践建议：

- 需要纯粹的方向相似度时，使用余弦相似度
- 当模长携带有意义的信息时，使用点积
- 许多向量数据库（Pinecone、Weaviate、Qdrant）允许你在两者间选择
- 如果你的嵌入已做 L2 归一化，选哪个都一样

### 马氏距离

欧氏距离平等对待所有维度。但如果你的特征之间存在相关性或量纲不同，L2 会给出误导性的结果。

马氏距离（Mahalanobis distance）会考虑数据的协方差结构。

```
d_M(x, y) = sqrt((x - y)^T * S^(-1) * (x - y))
```

其中 S 是数据的协方差矩阵。

直观理解：马氏距离先对数据去相关并归一化（白化），然后在变换后的空间中计算 L2 距离。如果 S 是单位矩阵（特征不相关且方差为 1），马氏距离就退化为欧氏距离。

```
Example: height and weight are correlated.
Someone 6'2" and 180 lbs is not unusual.
Someone 5'0" and 180 lbs is unusual.

Euclidean distance might say they are equally far from the mean.
Mahalanobis distance correctly identifies the second as an outlier
because it accounts for the height-weight correlation.
```

何时使用马氏距离：

- 离群值检测（与均值的马氏距离很大的点就是离群点）
- 特征量纲不同且相关时的分类任务
- 当数据量足以可靠估计协方差矩阵时
- 制造业质量控制（多变量过程监控）

### Jaccard 相似度（针对集合）

Jaccard 相似度度量两个集合之间的重叠程度。

```
J(A, B) = |A intersect B| / |A union B|
```

取值范围从 0（无重叠）到 1（集合完全相同）。Jaccard 距离 = 1 - Jaccard 相似度。

```
A = {cat, dog, fish}
B = {cat, bird, fish, snake}

Intersection = {cat, fish}         size = 2
Union = {cat, dog, fish, bird, snake}  size = 5

Jaccard similarity = 2/5 = 0.4
Jaccard distance = 0.6
```

何时使用 Jaccard：

- 比较标签、类别或特征的集合
- 基于词语是否出现（而非频次）的文档相似度
- 近似重复检测（用 MinHash 近似 Jaccard）
- 比较二值特征向量（存在/缺失类数据）
- 评估分割模型（交并比 IoU 就是 Jaccard）

### 编辑距离（Levenshtein 距离）

编辑距离统计把一个字符串变成另一个字符串所需的最少单字符操作数。操作包括：插入、删除或替换。

```
"kitten" -> "sitting"

kitten -> sitten  (substitute k -> s)
sitten -> sittin  (substitute e -> i)
sittin -> sitting (insert g)

Edit distance = 3
```

用动态规划计算：填充一个矩阵，其中第 (i, j) 个元素是字符串 A 的前 i 个字符与字符串 B 的前 j 个字符之间的编辑距离。

```
        ""  s  i  t  t  i  n  g
    ""   0  1  2  3  4  5  6  7
    k    1  1  2  3  4  5  6  7
    i    2  2  1  2  3  4  5  6
    t    3  3  2  1  2  3  4  5
    t    4  4  3  2  1  2  3  4
    e    5  5  4  3  2  2  3  4
    n    6  6  5  4  3  3  2  3
```

何时使用编辑距离：

- 拼写检查与纠错
- DNA 序列比对（使用带权重的操作）
- 模糊字符串匹配
- 杂乱文本数据的去重

### KL 散度（不是距离，却常被当作距离用）

KL 散度度量一个概率分布与另一个概率分布的差异程度。第 09 课已有讲解，但它属于本课的讨论范围，因为人们常把它当作「距离」使用——尽管它并不是。

```
D_KL(P || Q) = sum(p(x) * log(p(x) / q(x)))
```

关键性质：KL 散度不对称。

```
D_KL(P || Q) != D_KL(Q || P)
```

这意味着它不满足距离度量的基本要求，也不满足三角不等式。它是一种散度（divergence），不是距离。

前向 KL（D_KL(P || Q)）是「均值寻找型」（mean-seeking）：Q 试图覆盖 P 的所有峰。
反向 KL（D_KL(Q || P)）是「峰值寻找型」（mode-seeking）：Q 聚焦在 P 的单个峰上。

你会在哪些地方见到 KL 散度：

- VAE（ELBO 中的 KL 项把隐变量分布推向先验分布）
- 知识蒸馏（学生模型试图匹配教师模型的分布）
- RLHF（KL 惩罚让微调后的模型贴近基础模型）
- 策略梯度方法（约束策略更新的幅度）

### Wasserstein 距离（推土机距离）

Wasserstein 距离度量把一个概率分布变换为另一个分布所需的最小「功」。可以这样想象：如果一个分布是一堆土，另一个分布是一个坑，你需要搬多少土、搬多远？

```
W(P, Q) = inf over all transport plans gamma of E[d(x, y)]
```

对于一维分布，它可以简化为两个累积分布函数之差的绝对值的积分：

```
W_1(P, Q) = integral |CDF_P(x) - CDF_Q(x)| dx
```

Wasserstein 为什么重要：

- 它是真正的度量（对称，满足三角不等式）
- 即使两个分布不重叠，它也能提供梯度（此时 KL 散度会趋于无穷大）
- 这一性质让它成为 Wasserstein GAN（WGAN）的核心，解决了原始 GAN 训练不稳定的问题

```
Distributions with no overlap:

P: [1, 0, 0, 0, 0]    Q: [0, 0, 0, 0, 1]

KL divergence: infinity (log of zero)
Wasserstein: 4 (move all mass 4 bins)

Wasserstein gives a meaningful gradient. KL does not.
```

何时使用 Wasserstein：

- GAN 训练（WGAN、WGAN-GP）
- 比较可能不重叠的分布
- 最优传输问题
- 图像检索（比较颜色直方图）

### 为什么不同任务需要不同的距离

| 任务 | 最佳距离 | 原因 |
|------|--------------|-----|
| 文本相似度 | 余弦 | 大小是噪声，方向才是语义 |
| 图像像素比较 | L2 | 空间关系重要，特征量纲可比 |
| 稀疏高维特征 | L1 | 稳健，不会放大罕见的大差异 |
| 集合重叠（标签、类别） | Jaccard | 数据天然是集合形式，不是向量 |
| 字符串匹配 | 编辑距离 | 操作对应人类编辑的直觉 |
| 离群值检测 | 马氏距离 | 考虑特征相关性和量纲差异 |
| 比较概率分布 | KL 散度 | 度量用 Q 代替 P 所损失的信息 |
| GAN 训练 | Wasserstein | 分布不重叠时仍能提供梯度 |
| 嵌入（向量数据库） | 余弦或点积 | 嵌入经过训练，语义编码在方向中 |
| 推荐系统 | 点积 | 模长可以编码热门度或置信度 |
| DNA 序列 | 带权编辑距离 | 替换代价因核苷酸对而异 |
| 制造业质检 | L-infinity | 任意维度的最坏偏差才是关键 |

### 与损失函数的联系

损失函数就是作用在预测值与目标值之间的距离函数。

```
Loss function       Distance it uses       Behavior
MSE                 L2 squared             Penalizes large errors heavily
MAE                 L1                     Penalizes all errors equally
Huber loss          L1 for large errors,   Best of both: robust to outliers,
                    L2 for small errors    smooth gradient near zero
Cross-entropy       KL divergence          Measures distribution mismatch
Hinge loss          max(0, margin - d)     Only penalizes below margin
Triplet loss        L2 (typically)         Pulls positives close, pushes
                                           negatives away
Contrastive loss    L2                     Similar pairs close, dissimilar
                                           pairs beyond margin
```

### 与正则化的联系

正则化就是在损失函数中加入对权重的范数惩罚。

```
L1 regularization (Lasso):   loss + lambda * ||w||_1
  -> Sparse weights. Some weights become exactly zero.
  -> Automatic feature selection.
  -> Solution has corners (non-differentiable at zero).

L2 regularization (Ridge):   loss + lambda * ||w||_2^2
  -> Small weights. All weights shrink toward zero.
  -> No feature selection (nothing goes to exactly zero).
  -> Smooth solution everywhere.

Elastic Net:                  loss + lambda_1 * ||w||_1 + lambda_2 * ||w||_2^2
  -> Combines sparsity of L1 with stability of L2.
  -> Groups of correlated features are kept or dropped together.
```

为什么 L1 产生稀疏性而 L2 不会：想象二维权重空间中的约束区域。L1 是菱形，L2 是圆形。损失函数的等高线（椭圆）最有可能在菱形的尖角处相切——而尖角处恰好有一个权重为零。等高线与圆相切的位置是光滑点，两个权重都不为零。

### 最近邻搜索

每个距离函数都对应一个最近邻搜索问题：给定一个查询点，在数据集中找到离它最近的点。

精确最近邻搜索在含 n 个点、d 维的数据集中每次查询的复杂度是 O(n * d)。对于大规模数据集，这太慢了。

近似最近邻（Approximate Nearest Neighbor，ANN）算法用少量精度损失换取巨大的速度提升：

```
Algorithm         Approach                      Used by
KD-trees          Axis-aligned space partition   scikit-learn (low-dim)
Ball trees        Nested hyperspheres            scikit-learn (medium-dim)
LSH               Random hash projections        Near-duplicate detection
HNSW              Hierarchical navigable         FAISS, Qdrant, Weaviate
                  small-world graph
IVF               Inverted file index with       FAISS (billion-scale)
                  cluster-based search
Product quant.    Compress vectors, search       FAISS (memory-constrained)
                  in compressed space
```

HNSW（Hierarchical Navigable Small World，分层可导航小世界图）是现代向量数据库中占主导地位的算法。它构建一个多层图，每个节点连接到其近似最近邻。搜索从顶层开始（稀疏，长距离跳跃），逐层下降到底层（稠密，短距离跳跃）。

```figure
norm-unit-balls
```

## 从零实现

### 第 1 步：实现所有范数和距离函数

完整实现见 `code/distances.py`。每个函数都只用基础 Python 数学运算从零构建。

### 第 2 步：同一份数据，不同距离，不同最近邻

`distances.py` 中的演示会创建一个数据集，选取一个查询点，然后展示最近邻如何随距离度量的变化而变化。在 L1 下「最近」的点，在 L2 或余弦下未必最近。

### 第 3 步：嵌入相似度检索

代码包含一个模拟的嵌入相似度检索：分别用余弦相似度和 L2 距离查找与查询最相似的「文档」，展示两种排名可能不同。

## 生产实践

最常见的实际应用：在向量数据库中查找相似条目。

```python
import numpy as np

def cosine_similarity_matrix(X):
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1, norms)
    X_normalized = X / norms
    return X_normalized @ X_normalized.T

embeddings = np.random.randn(1000, 768)

sim_matrix = cosine_similarity_matrix(embeddings)

query_idx = 0
similarities = sim_matrix[query_idx]
top_k = np.argsort(similarities)[::-1][1:6]
print(f"Top 5 most similar to item 0: {top_k}")
print(f"Similarities: {similarities[top_k]}")
```

当你调用 `model.encode(text)` 然后检索向量数据库时，底层发生的就是这件事。嵌入模型把文本映射为向量，向量数据库计算你的查询向量与每个已存储向量之间的余弦相似度（或点积），并用 ANN 算法避免逐一比对所有向量。

## 练习

1. 计算 (1, 2, 3) 和 (4, 0, 6) 之间的 L1、L2 和 L-infinity 距离。验证对任意一对点都有 L-inf <= L2 <= L1。证明这个排序为什么总是成立。

2. 构造两个向量，使其余弦相似度很高（> 0.9）但 L2 距离很大（> 10）。从几何角度解释发生了什么。然后再构造两个向量，使其余弦相似度很低（< 0.3）但 L2 距离很小（< 0.5）。

3. 实现一个函数，输入一个数据集和一个查询点，分别返回 L1、L2、余弦和马氏距离下的最近邻。找出一个让这四种度量对「哪个点最近」全部意见不一的数据集。

4. 用 CDF 方法手工计算 [0.5, 0.5, 0, 0] 与 [0, 0, 0.5, 0.5] 之间的 Wasserstein 距离。再计算 [0.25, 0.25, 0.25, 0.25] 与 [0, 0, 0.5, 0.5] 之间的距离。哪个更大？为什么？

5. 实现用于近似 Jaccard 相似度的 MinHash。生成 100 个随机集合，计算所有点对的精确 Jaccard，并与使用 50、100、200 个哈希函数的 MinHash 近似值进行比较。绘制近似误差曲线。

## 关键术语

| 术语 | 通俗说法 | 实际含义 |
|------|----------------|----------------------|
| 范数（Norm） | 「向量的大小」 | 把向量映射为非负标量的函数，满足三角不等式、绝对齐次性，且仅在零向量处取零 |
| L1 范数 | 「曼哈顿距离」 | 各分量绝对值之和。在优化中产生稀疏性，对离群值稳健 |
| L2 范数 | 「欧氏距离」 | 各分量平方和的平方根。欧氏空间中的直线距离 |
| Lp 范数 | 「广义范数」 | 各分量绝对值 p 次幂之和的 p 次方根。L1 和 L2 是其特例 |
| L-infinity 范数 | 「最大范数」或「切比雪夫距离」 | 分量绝对值的最大值。Lp 在 p 趋于无穷时的极限 |
| 余弦相似度 | 「向量间的夹角」 | 用两个模长归一化后的点积。取值范围 -1 到 +1，忽略向量长度 |
| 余弦距离 | 「1 减去余弦相似度」 | 把余弦相似度转换为距离。取值范围 0 到 2 |
| 点积 | 「未归一化的余弦」 | 各分量逐项相乘求和。等于余弦相似度乘以两个向量的模长 |
| 马氏距离 | 「相关性感知的距离」 | 在用数据协方差矩阵白化（去相关并归一化）后的空间中计算的 L2 距离 |
| Jaccard 相似度 | 「集合重叠度」 | 交集大小除以并集大小。针对集合而非向量 |
| 编辑距离 | 「Levenshtein 距离」 | 把一个字符串变成另一个所需的最少插入、删除和替换次数 |
| KL 散度 | 「分布之间的距离」 | 并非真正的距离（不对称）。度量用 Q 编码 P 所需的额外比特数 |
| Wasserstein 距离 | 「推土机距离」 | 把质量从一个分布搬运到另一个分布所需的最小功。是真正的度量 |
| 近似最近邻 | 「ANN 检索」 | 一类算法（HNSW、LSH、IVF），比精确搜索快得多地找到近似最近的点 |
| HNSW | 「向量数据库的算法」 | Hierarchical Navigable Small World 图。用于快速近似最近邻搜索的多层图结构 |
| L1 正则化 | 「Lasso」 | 在损失中加入权重的 L1 范数。把权重压到零（稀疏性） |
| L2 正则化 | 「Ridge」或「权重衰减」 | 在损失中加入权重 L2 范数的平方。让权重向零收缩但不产生稀疏性 |
| Elastic Net | 「L1 + L2」 | 结合 L1 和 L2 正则化。比单独使用任何一种都更好地处理相关特征组 |

## 延伸阅读

- [FAISS: A Library for Efficient Similarity Search](https://github.com/facebookresearch/faiss) - Meta 推出的十亿级 ANN 检索库
- [Wasserstein GAN (Arjovsky et al., 2017)](https://arxiv.org/abs/1701.07875) - 把推土机距离引入 GAN 的开创性论文
- [Locality-Sensitive Hashing (Indyk & Motwani, 1998)](https://dl.acm.org/doi/10.1145/276698.276876) - 奠基性的 ANN 算法
- [Efficient Estimation of Word Representations (Mikolov et al., 2013)](https://arxiv.org/abs/1301.3781) - Word2Vec，余弦相似度自此成为嵌入领域的默认选择
- [sklearn.neighbors documentation](https://scikit-learn.org/stable/modules/neighbors.html) - scikit-learn 中距离度量与近邻算法的实用指南
