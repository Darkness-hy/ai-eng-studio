# 朴素贝叶斯

> 那个「朴素」的假设是错的，但它照样管用。这正是它的美妙之处。

**Type:** Build
**Language:** Python
**Prerequisites:** Phase 2, Lessons 01-07 (classification, Bayes' theorem)
**Time:** ~75 minutes

## 学习目标

- 从零实现带拉普拉斯平滑（Laplace smoothing）的多项式朴素贝叶斯（Multinomial Naive Bayes），用于文本分类
- 解释为什么朴素独立性假设在数学上是错的，但在实践中仍能给出正确的类别排序
- 比较 Multinomial、Bernoulli 和 Gaussian 三种朴素贝叶斯变体，并能根据特征类型选出合适的那一个
- 在高维稀疏数据上对比朴素贝叶斯与逻辑回归的表现，并解释其中起作用的偏差-方差权衡

## 问题背景

你需要给文本分类。把邮件分成垃圾邮件和正常邮件，把用户评论分成好评和差评，把客服工单分到各个类别。你有上千个特征（每个词一个），训练数据却很有限。

大多数分类器在这种场景下会卡壳。逻辑回归需要足够多的样本才能可靠地估计上千个权重；决策树每次只按一个词来划分，会严重过拟合；KNN 在 10,000 维空间里毫无意义，因为每个点到其他所有点的距离都差不多。

朴素贝叶斯能搞定这种场景。它做了一个数学上错误的假设（在给定类别的条件下，每个特征都与其他特征相互独立），却依然能在文本分类上胜过那些「更聪明」的模型，尤其是在训练集很小的时候。它只需对数据扫一遍就能完成训练，可以扩展到上百万个特征，还能输出概率估计（不过由于独立性假设，这些概率往往校准得很差）。

理解为什么一个错误的假设能带来好的预测，会让你领悟到机器学习的一个根本道理：最好的模型不是最正确的那个，而是对你的数据来说偏差-方差权衡最优的那个。

## 核心概念

### 贝叶斯定理（快速回顾）

贝叶斯定理把条件概率反转过来：

```
P(class | features) = P(features | class) * P(class) / P(features)
```

我们想求的是 `P(class | features)`——给定文档中的词，这篇文档属于某个类别的概率。它可以由以下几项计算出来：
- `P(features | class)`——在该类别的文档中看到这些词的似然
- `P(class)`——类别的先验概率（垃圾邮件总体上有多常见？）
- `P(features)`——证据项，对所有类别都相同，所以在比较时可以忽略

`P(class | features)` 最高的类别胜出。

### 朴素独立性假设

要精确计算 `P(features | class)`，需要估计所有特征的联合概率。词表有 10,000 个词时，你需要在 2^10,000 种可能组合上估计一个分布。不可能做到。

朴素假设：在给定类别的条件下，每个特征相互条件独立。

```
P(w1, w2, ..., wn | class) = P(w1 | class) * P(w2 | class) * ... * P(wn | class)
```

这样就不必估计一个不可能完成的联合分布，而是估计 n 个简单的单特征分布。每一个只需要一次计数。

这个假设显然是错的。在任何文档里，「machine」和「learning」这两个词都不是独立的。但分类器并不需要正确的概率估计，它需要的是正确的排序——哪个类别的概率最高。独立性假设会引入系统性误差，但这些误差对所有类别的影响相似，所以排序仍然正确。

### 为什么它依然有效

三个原因：

1. **排序重于校准。** 分类只要求排名第一的类别是对的。即使真实概率是 0.7 而模型给出 P(spam) = 0.99999，分类器依然能正确选出垃圾邮件。我们不需要正确的概率，只需要正确的赢家。

2. **高偏差，低方差。** 独立性假设是一个很强的先验，它对模型施加了强约束，从而防止过拟合。在训练数据有限时，一个略有偏差但稳定的模型胜过一个理论上正确却极不稳定的模型。这正是偏差-方差权衡的实际体现。

3. **特征冗余相互抵消。** 相关的特征提供的是冗余证据。分类器会把这些证据重复计算，但它对正确的类别同样重复计算。如果「machine」和「learning」总是一起出现，它们都为「科技」类提供证据。朴素贝叶斯把它们算了两次，但算两次算在了正确的类别上。

还有第四个实用层面的原因：朴素贝叶斯极其快。训练就是对数据扫一遍、统计频次；预测就是一次矩阵乘法。一百万篇文档几秒钟就能训完。这种速度意味着你可以更快地迭代、尝试更多特征集、跑更多实验，远超那些更慢的模型。

### 数学推导一步步来

我们用一个具体例子来推演。假设有两个类别：垃圾邮件（spam）和正常邮件（not-spam）。词表只有三个词：「free」「money」「meeting」。

训练数据：
- 垃圾邮件中「free」出现 80 次，「money」60 次，「meeting」10 次（共 150 个词）
- 正常邮件中「free」出现 5 次，「money」10 次，「meeting」100 次（共 115 个词）
- 40% 的邮件是垃圾邮件，60% 是正常邮件

使用拉普拉斯平滑（alpha=1）：

```
P(free | spam)    = (80 + 1) / (150 + 3) = 81/153 = 0.529
P(money | spam)   = (60 + 1) / (150 + 3) = 61/153 = 0.399
P(meeting | spam) = (10 + 1) / (150 + 3) = 11/153 = 0.072

P(free | not-spam)    = (5 + 1) / (115 + 3) = 6/118 = 0.051
P(money | not-spam)   = (10 + 1) / (115 + 3) = 11/118 = 0.093
P(meeting | not-spam) = (100 + 1) / (115 + 3) = 101/118 = 0.856
```

新邮件包含：「free」（2 次）、「money」（1 次）、「meeting」（0 次）。

```
log P(spam | email) = log(0.4) + 2*log(0.529) + 1*log(0.399) + 0*log(0.072)
                    = -0.916 + 2*(-0.637) + (-0.919) + 0
                    = -3.109

log P(not-spam | email) = log(0.6) + 2*log(0.051) + 1*log(0.093) + 0*log(0.856)
                        = -0.511 + 2*(-2.976) + (-2.375) + 0
                        = -8.838
```

垃圾邮件以巨大优势胜出。「free」出现两次是垃圾邮件的强证据。注意「meeting」没有出现，它对两个对数和的贡献都是零（0 * log(P)）——在 Multinomial NB 中，未出现的词没有任何影响。显式地为词的缺失建模的是 Bernoulli NB。

### 三种变体

朴素贝叶斯有三种口味，每一种对 `P(feature | class)` 的建模方式不同。

#### 多项式朴素贝叶斯（Multinomial Naive Bayes）

把每个特征建模为计数。最适合特征是词频或 TF-IDF 值的文本数据。

```
P(word_i | class) = (count of word_i in class + alpha) / (total words in class + alpha * vocab_size)
```

其中 `alpha` 是拉普拉斯平滑（下文解释）。这个变体是文本分类的主力。

#### 高斯朴素贝叶斯（Gaussian Naive Bayes）

把每个特征建模为正态分布。最适合连续特征。

```
P(x_i | class) = (1 / sqrt(2 * pi * var)) * exp(-(x_i - mean)^2 / (2 * var))
```

每个类别对每个特征都有自己的均值和方差。当特征在每个类别内部确实近似呈钟形曲线分布时，效果很好。

#### 伯努利朴素贝叶斯（Bernoulli Naive Bayes）

把每个特征建模为二值（出现或不出现）。最适合短文本或二值特征向量。

```
P(word_i | class) = (docs in class containing word_i + alpha) / (total docs in class + 2 * alpha)
```

与 Multinomial 不同，Bernoulli 会显式地惩罚词的缺失。如果「free」通常出现在垃圾邮件里，而这封邮件里没有它，Bernoulli 会把这一点视为不利于垃圾邮件的证据。

### 各变体的适用场景

| 变体 | 特征类型 | 最适合 | 示例 |
|---------|-------------|----------|---------|
| Multinomial | 计数或频率 | 文本分类、词袋模型 | 邮件垃圾过滤、主题分类 |
| Gaussian | 连续值 | 特征近似正态的表格数据 | Iris 分类、传感器数据 |
| Bernoulli | 二值（0/1） | 短文本、二值特征向量 | 短信垃圾过滤、出现/缺失特征 |

### 拉普拉斯平滑

如果某个词出现在测试数据里，却从未在某个类别的训练数据中出现过，会发生什么？

不做平滑时：`P(word | class) = 0/N = 0`。一个零乘进整个连乘式，就会让 `P(class | features) = 0`，其他所有证据都无济于事。一个没见过的词就能毁掉整个预测，不管有多少其他证据支持它。

拉普拉斯平滑给每个特征计数加上一个小量 `alpha`（通常是 1）：

```
P(word_i | class) = (count(word_i, class) + alpha) / (total_words_in_class + alpha * vocab_size)
```

当 alpha=1 时，每个词至少获得一点微小的概率。测试邮件里出现「discombobulate」也不会再把垃圾邮件的概率打成零。这种平滑有贝叶斯解释：它等价于在词分布上放一个均匀的 Dirichlet 先验。

alpha 越大，平滑越强（分布越趋于均匀）；alpha 越小，模型越信任数据。alpha 是一个需要调的超参数。

alpha 的影响：

| Alpha | 影响 | 适用时机 |
|-------|--------|-------------|
| 0.001 | 几乎不平滑，信任数据 | 训练集非常大，预计没有未见过的特征 |
| 0.1 | 轻度平滑 | 大训练集 |
| 1.0 | 标准拉普拉斯平滑 | 默认起点 |
| 10.0 | 重度平滑，拉平分布 | 训练集非常小，预计有大量未见过的特征 |

### 对数空间计算

把成百上千个概率（每个都小于 1）相乘会导致浮点下溢。即使真实值是一个非常小的正数，乘积在浮点数里也会变成零。

解决办法：在对数空间里计算。不再相乘概率，而是相加它们的对数：

```
log P(class | x1, x2, ..., xn) = log P(class) + sum_i log P(xi | class)
```

这把预测变成了一个点积：

```
log_scores = X @ log_feature_probs.T + log_class_priors
prediction = argmax(log_scores)
```

矩阵乘法。这就是朴素贝叶斯预测如此之快的原因——它和单层线性模型是同一种运算。

### 朴素贝叶斯 vs 逻辑回归

两者都是用于文本的线性分类器，区别在于建模对象不同。

| 维度 | 朴素贝叶斯 | 逻辑回归 |
|--------|------------|-------------------|
| 类型 | 生成式（建模 P(X\|Y)） | 判别式（建模 P(Y\|X)） |
| 训练 | 统计频次 | 优化损失函数 |
| 小数据 | 更好（强先验有帮助） | 更差（数据不够估计权重） |
| 大数据 | 更差（错误假设拖后腿） | 更好（边界更灵活） |
| 特征 | 假设独立 | 能处理相关性 |
| 速度 | 单次扫描，非常快 | 迭代优化 |
| 校准 | 概率较差 | 概率更好 |

经验法则：先用朴素贝叶斯。如果数据足够多且 NB 性能进入平台期，再换成逻辑回归。

### 分类流水线

```mermaid
flowchart LR
    A[Raw Text] --> B[Tokenize]
    B --> C[Build Vocabulary]
    C --> D[Count Word Frequencies]
    D --> E[Apply Smoothing]
    E --> F[Compute Log Probabilities]
    F --> G[Predict: argmax P class given words]

    style A fill:#f9f,stroke:#333
    style G fill:#9f9,stroke:#333
```

实践中我们在对数空间里计算，以避免浮点下溢。不是把许多小概率相乘，而是把它们的对数相加：

```
log P(class | features) = log P(class) + sum_i log P(feature_i | class)
```

```figure
naive-bayes
```

## 从零实现

`code/naive_bayes.py` 中的代码从零实现了 MultinomialNB 和 GaussianNB。

### MultinomialNB

从零实现的步骤：

1. **fit(X, y)**：对每个类别，统计每个特征的频次，加上拉普拉斯平滑，计算对数概率，并存储类别先验（类别频率的对数）。

2. **predict_log_proba(X)**：对每个样本，针对所有类别计算 log P(class) + sum of log P(feature_i | class)。这是一次矩阵乘法：X @ log_probs.T + log_priors。

3. **predict(X)**：返回对数概率最高的类别。

```python
class MultinomialNB:
    def __init__(self, alpha=1.0):
        self.alpha = alpha

    def fit(self, X, y):
        classes = np.unique(y)
        n_classes = len(classes)
        n_features = X.shape[1]

        self.classes_ = classes
        self.class_log_prior_ = np.zeros(n_classes)
        self.feature_log_prob_ = np.zeros((n_classes, n_features))

        for i, c in enumerate(classes):
            X_c = X[y == c]
            self.class_log_prior_[i] = np.log(X_c.shape[0] / X.shape[0])
            counts = X_c.sum(axis=0) + self.alpha
            self.feature_log_prob_[i] = np.log(counts / counts.sum())

        return self
```

关键洞察：拟合之后，预测就只是矩阵乘法加一个偏置。这就是朴素贝叶斯如此之快的原因。

### GaussianNB

对于连续特征，我们按类别、按特征估计均值和方差：

```python
class GaussianNB:
    def __init__(self):
        pass

    def fit(self, X, y):
        classes = np.unique(y)
        self.classes_ = classes
        self.means_ = np.zeros((len(classes), X.shape[1]))
        self.vars_ = np.zeros((len(classes), X.shape[1]))
        self.priors_ = np.zeros(len(classes))

        for i, c in enumerate(classes):
            X_c = X[y == c]
            self.means_[i] = X_c.mean(axis=0)
            self.vars_[i] = X_c.var(axis=0) + 1e-9
            self.priors_[i] = X_c.shape[0] / X.shape[0]

        return self
```

预测时对每个特征使用高斯概率密度函数（PDF），再在特征间相乘（在对数空间里相加）。

### 演示：文本分类

代码生成模拟两个类别（科技文章 vs 体育文章）的合成词袋数据。每个类别有不同的词频分布。MultinomialNB 用词计数对它们进行分类。

合成数据是这样构造的：我们创建 200 个「词」（特征列）。第 0-39 号词在科技文章中高频出现，在体育文章中低频；第 80-119 号词在体育文章中高频，在科技文章中低频；第 40-79 号词在两类中都是中等频率。这构造出一个贴近现实的场景：有些词是强类别指示词，另一些则是噪声。

### 演示：连续特征

代码生成类似 Iris 的数据（3 个类别、4 个特征、高斯簇）。GaussianNB 用每个类别的均值和方差进行分类。每个类别有不同的中心（均值向量）和不同的散布（方差），模拟真实世界中各类别间测量值存在系统性差异的数据。

代码还演示了：
- **平滑对比：** 用不同的 alpha 值训练 MultinomialNB，展示平滑强度对准确率的影响。
- **训练集规模实验：** 训练数据从 20 个样本增长到 1600 个样本时，NB 的准确率如何提升。NB 在样本极少时就能达到不错的准确率——这是它的主要优势。
- **混淆矩阵：** 按类别给出精确率、召回率和 F1 分数，展示 NB 在哪些地方出错。

### 预测速度

朴素贝叶斯的预测就是一次矩阵乘法。对于 n 个样本、d 个特征、k 个类别：
- MultinomialNB：一次矩阵乘法 (n x d) @ (d x k) = O(n * d * k)
- GaussianNB：n * k 次高斯 PDF 计算，每次涉及 d 个特征 = O(n * d * k)

两者在每个维度上都是线性的。对比 KNN（需要计算到所有训练点的距离）或带 RBF 核的 SVM（需要对所有支持向量做核计算），NB 在预测时快几个数量级。

## 生产实践

用 sklearn，两种变体都是一行代码：

```python
from sklearn.naive_bayes import GaussianNB, MultinomialNB

gnb = GaussianNB()
gnb.fit(X_train, y_train)
print(f"GaussianNB accuracy: {gnb.score(X_test, y_test):.3f}")

mnb = MultinomialNB(alpha=1.0)
mnb.fit(X_train_counts, y_train)
print(f"MultinomialNB accuracy: {mnb.score(X_test_counts, y_test):.3f}")
```

用 sklearn 做文本分类：

```python
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

text_clf = Pipeline([
    ("vectorizer", CountVectorizer()),
    ("classifier", MultinomialNB(alpha=1.0)),
])

text_clf.fit(train_texts, train_labels)
accuracy = text_clf.score(test_texts, test_labels)
```

`naive_bayes.py` 中的代码在同一份数据上把从零实现与 sklearn 进行对比，以验证正确性。

### TF-IDF 与朴素贝叶斯

原始词计数让每个词的每次出现获得相同权重。但像「the」和「is」这种常见词在每个类别中都高频出现——它们不携带任何信息。TF-IDF（词频-逆文档频率，Term Frequency - Inverse Document Frequency）降低常见词的权重，提升稀有且有区分度的词的权重。

```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

text_clf = Pipeline([
    ("tfidf", TfidfVectorizer()),
    ("classifier", MultinomialNB(alpha=0.1)),
])
```

TF-IDF 值是非负的，所以可以和 MultinomialNB 配合使用。TF-IDF + MultinomialNB 的组合是文本分类最强的基线之一。在训练样本少于 10,000 个的数据集上，它经常打败更复杂的模型。

### 用 BernoulliNB 处理短文本

对于短文本（推文、短信、聊天消息），BernoulliNB 可能胜过 MultinomialNB。短文本的词计数很低，MultinomialNB 依赖的频率信息因此噪声很大。BernoulliNB 只关心词是否出现，这在短文本上更可靠。

```python
from sklearn.naive_bayes import BernoulliNB
from sklearn.feature_extraction.text import CountVectorizer

text_clf = Pipeline([
    ("vectorizer", CountVectorizer(binary=True)),
    ("classifier", BernoulliNB(alpha=1.0)),
])
```

CountVectorizer 的 `binary=True` 参数把所有计数转换为 0/1。不加这个参数 BernoulliNB 也能运行，但它看到的是不符合其设计假设的计数数据。

### 校准 NB 的概率

NB 的概率校准很差。当 NB 说 P(spam) = 0.95 时，真实概率可能只有 0.7。如果你需要可靠的概率估计（例如用来设定阈值，或与其他模型组合），可以使用 sklearn 的 CalibratedClassifierCV：

```python
from sklearn.calibration import CalibratedClassifierCV

calibrated_nb = CalibratedClassifierCV(MultinomialNB(), cv=5, method="sigmoid")
calibrated_nb.fit(X_train, y_train)
proba = calibrated_nb.predict_proba(X_test)
```

它通过交叉验证在 NB 的原始得分之上拟合一个逻辑回归。得到的概率与真实类别频率接近得多。

### 常见坑

1. **负的特征值。** MultinomialNB 要求特征非负。如果有负值（比如某些设置下的 TF-IDF，或标准化后的特征），请改用 GaussianNB，或把特征平移到正值区间。

2. **零方差特征。** GaussianNB 要除以方差。如果某个特征在某个类别内方差为零（所有值相同），概率计算就会崩坏。代码给所有方差加了一个微小的平滑项（1e-9）来避免这个问题。

3. **类别不平衡。** 如果 99% 的邮件是正常邮件，先验 P(not-spam) = 0.99 强到能压倒似然证据。你可以手动设定类别先验，或使用 sklearn 的 class_prior 参数。

4. **特征缩放。** MultinomialNB 不需要缩放（它处理的是计数），GaussianNB 也不需要（它逐特征估计统计量）。这一点优于对特征尺度敏感的逻辑回归和 SVM。

## 交付产物

本课产出：
- `outputs/skill-naive-bayes-chooser.md`——一份用于挑选合适 NB 变体的决策技能文档
- `code/naive_bayes.py`——从零实现的 MultinomialNB 和 GaussianNB，并与 sklearn 对比

### 朴素贝叶斯什么时候会失败

当独立性假设导致排序错误（而不只是概率错误）时，NB 就会失败。这种情况发生在：

1. **强特征交互。** 如果类别取决于两个特征的组合，而单看任何一个都没用（类似 XOR 的模式），NB 会完全错过它。每个特征单独看都不提供证据，而 NB 无法对它们做非线性组合。

2. **高度相关且证据相反的特征。** 如果特征 A 说「垃圾邮件」、特征 B 说「正常邮件」，但 A 和 B 完全相关（现实中它们总是一致），NB 会在并不存在冲突的地方看到冲突的证据。

3. **非常大的训练集。** 数据足够多时，逻辑回归这类判别式模型能学到真正的决策边界，从而超过 NB。曾在小数据上帮过忙的独立性假设，这时反而成了模型的束缚。

实践中，这些失败模式在文本分类里很少见。文本特征数量多、单个特征弱，而且独立性假设的误差往往会相互抵消。对于特征不多且强相关的表格数据，应优先考虑逻辑回归或树模型。

## 练习

1. **平滑实验。** 用 alpha 为 0.01、0.1、1.0、10.0 和 100.0 在文本数据上训练 MultinomialNB。绘制准确率与 alpha 的关系图。性能在哪里达到峰值？为什么 alpha 太大会有损性能？

2. **特征独立性检验。** 拿一个真实文本数据集，挑两个明显相关的词（「machine」和「learning」）。计算 P(word1 | class) * P(word2 | class)，并与 P(word1 AND word2 | class) 比较。独立性假设错得有多离谱？它影响分类准确率吗？

3. **Bernoulli 实现。** 给代码扩展一个 BernoulliNB 类。把词袋转换为二值（出现/缺失），在文本数据上与 MultinomialNB 比较准确率。Bernoulli 什么时候会赢？

4. **NB vs 逻辑回归。** 在文本数据上同时训练两者。从 100 个训练样本开始，增加到 10,000 个。绘制两者准确率随训练集规模变化的曲线。逻辑回归在哪个点超过朴素贝叶斯？

5. **垃圾邮件过滤器。** 构建一个完整的垃圾邮件分类器：对原始邮件文本分词、构建词表、生成词袋特征、训练 MultinomialNB，并用精确率和召回率评估（不只是准确率——为什么？）。

## 关键术语

| 术语 | 人们常说 | 实际含义 |
|------|----------------|----------------------|
| 朴素贝叶斯（Naive Bayes） | 「简单的概率分类器」 | 在「给定类别下特征条件独立」的假设下应用贝叶斯定理的分类器 |
| 条件独立（Conditional independence） | 「特征互不影响」 | P(A, B \| C) = P(A \| C) * P(B \| C)——一旦知道 C，知道 B 就不会再带来任何关于 A 的新信息 |
| 拉普拉斯平滑（Laplace smoothing） | 「加一平滑」 | 给每个特征加一个小计数，防止零概率主导预测 |
| 先验（Prior） | 「看到数据之前的信念」 | P(class)——观察任何特征之前每个类别的概率 |
| 似然（Likelihood） | 「数据有多吻合」 | P(features \| class)——在已知类别的前提下观察到这些特征的概率 |
| 后验（Posterior） | 「看到数据之后的信念」 | P(class \| features)——观察到特征之后，对类别概率的更新结果 |
| 生成式模型（Generative model） | 「建模数据如何生成」 | 学习 P(X \| Y) 和 P(Y)，再用贝叶斯定理得到 P(Y \| X) 的模型 |
| 判别式模型（Discriminative model） | 「建模决策边界」 | 直接学习 P(Y \| X)、不对 X 的生成过程建模的模型 |
| 对数概率（Log probability） | 「避免下溢」 | 用 log P 代替 P 来计算，防止许多小数的乘积在浮点数中变成零 |

## 延伸阅读

- [scikit-learn Naive Bayes docs](https://scikit-learn.org/stable/modules/naive_bayes.html)——三种变体及其数学细节
- [McCallum and Nigam, A Comparison of Event Models for Naive Bayes Text Classification (1998)](https://www.cs.cmu.edu/~knigam/papers/multinomial-aaaiws98.pdf)——文本场景下 Multinomial 与 Bernoulli 对比的经典之作
- [Rennie et al., Tackling the Poor Assumptions of Naive Bayes Text Classifiers (2003)](https://people.csail.mit.edu/jrennie/papers/icml03-nb.pdf)——针对文本的 NB 改进方法
- [Ng and Jordan, On Discriminative vs. Generative Classifiers (2001)](https://ai.stanford.edu/~ang/papers/nips01-discriminativegenerative.pdf)——证明 NB 在数据更少时比 LR 收敛更快
