/** Placement quiz ("find your level") — based on the upstream /find-your-level
 *  agent skill, scaled up: 50 questions (10 per area across 5 areas), graduate
 *  difficulty, score 0-50 → entry phase. */
import { useSyncExternalStore } from 'react';
import { cloudEnabled, getSupabase } from './supabase';

/** Each knowledge area is probed by this many questions. */
export const QUESTIONS_PER_AREA = 10;
/** An area scored below this (out of QUESTIONS_PER_AREA) is "partial" → Review. */
const SOLID_THRESHOLD = 8;

export interface AreaDef {
  key: string;
  zh: string;
  en: string;
  /** Phases marked "Review" (not "Skip") when this area is only partially mastered. */
  reviewPhases: number[];
}

export const AREAS: AreaDef[] = [
  { key: 'math', zh: '数学与统计', en: 'Math & Statistics', reviewPhases: [1] },
  { key: 'ml', zh: '经典机器学习', en: 'Classical ML', reviewPhases: [2] },
  { key: 'dl', zh: '深度学习', en: 'Deep Learning', reviewPhases: [3] },
  { key: 'nlp', zh: 'NLP 与 Transformer', en: 'NLP & Transformers', reviewPhases: [5, 7] },
  { key: 'applied', zh: '应用 AI', en: 'Applied AI', reviewPhases: [14] },
];

export interface PlacementQuestion {
  area: string;
  zh: string;
  en: string;
  optionsZh: string[];
  optionsEn: string[];
  correct: number;
}

export const QUESTIONS: PlacementQuestion[] = [
  {
    area: 'math',
    zh: "设实矩阵 A 的奇异值分解为 A = UΣVᵀ。关于 A 的奇异值与 AᵀA 的特征值，下列说法哪个正确？",
    en: "Let a real matrix A have SVD A = UΣVᵀ. Which statement about the relationship between A's singular values and the eigenvalues of AᵀA is correct?",
    optionsZh: ["A 的奇异值等于 AᵀA 的特征值", "A 的奇异值是 AᵀA 非负特征值的平方根", "A 的奇异值是 AᵀA 特征值的平方", "A 的奇异值与 AᵀA 的特征值之间没有固定关系"],
    optionsEn: ["A's singular values equal the eigenvalues of AᵀA", "A's singular values are the square roots of the nonnegative eigenvalues of AᵀA", "A's singular values are the squares of the eigenvalues of AᵀA", "There is no fixed relationship between A's singular values and the eigenvalues of AᵀA"],
    correct: 1,
  },
  {
    area: 'math',
    zh: "对 n 个独立同分布高斯样本，方差的最大似然估计 σ̂²_MLE = (1/n)Σ(xᵢ−x̄)²。关于它的偏差，下列哪个正确？",
    en: "For n i.i.d. Gaussian samples, the MLE of variance is σ̂²_MLE = (1/n)Σ(xᵢ−x̄)². Which statement about its bias is correct?",
    optionsZh: ["它是无偏的，因为 MLE 总是无偏估计", "它系统性高估真实方差，偏差因子为 (n+1)/n", "它系统性低估真实方差，期望为 (n−1)/n·σ²", "它的偏差随 n 增大而增大"],
    optionsEn: ["It is unbiased, since the MLE is always unbiased", "It systematically overestimates the true variance by a factor (n+1)/n", "It systematically underestimates the true variance, with expectation (n−1)/n·σ²", "Its bias grows larger as n increases"],
    correct: 2,
  },
  {
    area: 'math',
    zh: "对两个同方差高斯分布 p=N(0,1)、q=N(1,1)，KL 散度 D_KL(p‖q) 等于多少？并且关于 KL 对称性的判断正确的是？",
    en: "For two equal-variance Gaussians p=N(0,1), q=N(1,1), what is the KL divergence D_KL(p‖q), and which claim about KL symmetry is correct?",
    optionsZh: ["D_KL(p‖q)=0，因为两分布方差相同", "D_KL(p‖q)=1/2，且 KL 一般不对称：D_KL(p‖q)≠D_KL(q‖p)", "D_KL(p‖q)=1，且 KL 始终对称", "D_KL(p‖q)=1/2，但因均值差为对称量，故此处 KL 对称"],
    optionsEn: ["D_KL(p‖q)=0, because the two distributions have the same variance", "D_KL(p‖q)=1/2, and KL is generally asymmetric: D_KL(p‖q)≠D_KL(q‖p)", "D_KL(p‖q)=1, and KL is always symmetric", "D_KL(p‖q)=1/2, but since the mean gap is symmetric, KL is symmetric here"],
    correct: 1,
  },
  {
    area: 'math',
    zh: "采用 Beta(2,2) 作为伯努利参数 θ 的先验，观测到 7 次成功、3 次失败。关于后验均值与 MAP 估计，下列哪个正确？",
    en: "Using Beta(2,2) as the prior for a Bernoulli parameter θ, you observe 7 successes and 3 failures. Which statement about the posterior mean and the MAP estimate is correct?",
    optionsZh: ["后验为 Beta(9,5)，后验均值 = 9/14 ≈ 0.643，MAP = 8/12 ≈ 0.667", "后验为 Beta(7,3)，后验均值 = 7/10 = 0.7，MAP 也是 0.7", "后验均值与 MAP 必然相等，因为 Beta 分布对称", "后验为 Beta(9,5)，但 MAP = 后验均值 = 9/14"],
    optionsEn: ["Posterior is Beta(9,5); posterior mean = 9/14 ≈ 0.643, MAP = 8/12 ≈ 0.667", "Posterior is Beta(7,3); posterior mean = 7/10 = 0.7, and MAP is also 0.7", "The posterior mean and MAP must coincide, because the Beta distribution is symmetric", "Posterior is Beta(9,5), but MAP = posterior mean = 9/14"],
    correct: 0,
  },
  {
    area: 'math',
    zh: "最小化 f(x)=(x−2)² 受约束 x ≤ 1（一维不等式约束）。关于其 KKT 条件，下列哪个正确？",
    en: "Minimize f(x)=(x−2)² subject to x ≤ 1 (a one-dimensional inequality constraint). Which statement about its KKT conditions is correct?",
    optionsZh: ["最优点 x*=2，乘子 μ=0，约束未激活", "最优点 x*=1，乘子 μ=−2<0，满足 KKT", "无解，因为无约束最小点 x=2 落在可行域外", "最优点 x*=1，乘子 μ=2>0，约束激活且互补松弛成立"],
    optionsEn: ["Optimum x*=2, multiplier μ=0, constraint inactive", "Optimum x*=1, multiplier μ=−2<0, satisfying KKT", "No solution, since the unconstrained minimizer x=2 lies outside the feasible region", "Optimum x*=1, multiplier μ=2>0, constraint active and complementary slackness holds"],
    correct: 3,
  },
  {
    area: 'math',
    zh: "对联合高斯随机变量 (X,Y)，相关系数 ρ=0.6。它们之间的互信息 I(X;Y) 等于多少？关于 ρ=0 时的情况判断正确的是？",
    en: "For jointly Gaussian random variables (X,Y) with correlation coefficient ρ=0.6, what is the mutual information I(X;Y), and which claim about the ρ=0 case is correct?",
    optionsZh: ["I = −½ln(1−ρ²) ≈ 0.223 nats；当 ρ=0 时 I=0，二者独立", "I = ½ln(1−ρ²) < 0；互信息可以为负", "I = ρ² = 0.36 nats；ρ=0 时 I 仍可能大于 0", "I = −½ln(1−ρ²) ≈ 0.223 nats；但即便 ρ=0，联合高斯也可能不独立"],
    optionsEn: ["I = −½ln(1−ρ²) ≈ 0.223 nats; at ρ=0, I=0 and the two are independent", "I = ½ln(1−ρ²) < 0; mutual information can be negative", "I = ρ² = 0.36 nats; even at ρ=0, I can be positive", "I = −½ln(1−ρ²) ≈ 0.223 nats; but even at ρ=0 jointly Gaussian variables may be dependent"],
    correct: 0,
  },
  {
    area: 'math',
    zh: "三对角矩阵 M = [[2,−1,0],[−1,2,−1],[0,−1,2]]。关于它的正定性，下列哪个判断正确？",
    en: "Consider the tridiagonal matrix M = [[2,−1,0],[−1,2,−1],[0,−1,2]]. Which statement about its positive definiteness is correct?",
    optionsZh: ["非正定，因为含有负的非对角元素", "半正定但非正定，因为它是退化的（行列式为 0）", "正定：三个顺序主子式分别为 2、3、4，全部为正", "无法判断，需要计算全部特征值才能确定"],
    optionsEn: ["Not positive definite, because it contains negative off-diagonal entries", "Positive semidefinite but not positive definite, since it is degenerate (determinant 0)", "Positive definite: the three leading principal minors are 2, 3, 4, all positive", "Cannot be determined without computing all eigenvalues"],
    correct: 2,
  },
  {
    area: 'math',
    zh: "二次型 f(x) = xᵀAx，其中 A 不一定对称。关于其梯度 ∇f，下列哪个正确？",
    en: "For the quadratic form f(x) = xᵀAx, where A is not necessarily symmetric, which expression for the gradient ∇f is correct?",
    optionsZh: ["∇f = 2Ax，无论 A 是否对称", "∇f = Aᵀx", "∇f = (A + Aᵀ)x；仅当 A 对称时才简化为 2Ax", "∇f = (A − Aᵀ)x，因为只有反对称部分贡献梯度"],
    optionsEn: ["∇f = 2Ax, regardless of whether A is symmetric", "∇f = Aᵀx", "∇f = (A + Aᵀ)x; only when A is symmetric does it simplify to 2Ax", "∇f = (A − Aᵀ)x, since only the antisymmetric part contributes to the gradient"],
    correct: 2,
  },
  {
    area: 'math',
    zh: "下列哪个 2×2 矩阵是合法的协方差矩阵（对应某个二维随机向量）？",
    en: "Which of the following 2×2 matrices is a valid covariance matrix (for some 2-dimensional random vector)?",
    optionsZh: ["[[1, 1.2],[1.2, 1]]", "[[−1, 0],[0, 2]]", "[[1, 0.8],[0.8, 1]]", "[[1, 0.5],[0.6, 1]]（非对称）"],
    optionsEn: ["[[1, 1.2],[1.2, 1]]", "[[−1, 0],[0, 2]]", "[[1, 0.8],[0.8, 1]]", "[[1, 0.5],[0.6, 1]] (non-symmetric)"],
    correct: 2,
  },
  {
    area: 'math',
    zh: "对任意离散分布 p 和 q（以 bit 为单位、用 log₂），交叉熵 H(p,q) 与 p 的熵 H(p)、KL 散度 D_KL(p‖q) 之间的关系，下列哪个正确？",
    en: "For arbitrary discrete distributions p and q (in bits, using log₂), which relationship among the cross-entropy H(p,q), the entropy H(p), and the KL divergence D_KL(p‖q) is correct?",
    optionsZh: ["H(p,q) = H(p) − D_KL(p‖q)，所以 H(p,q) ≤ H(p)", "H(p,q) = D_KL(p‖q)，与 H(p) 无关", "H(p,q) 总是等于 H(p)，因为交叉熵只依赖真实分布 p", "H(p,q) = H(p) + D_KL(p‖q)，所以 H(p,q) ≥ H(p)，当且仅当 p=q 时取等"],
    optionsEn: ["H(p,q) = H(p) − D_KL(p‖q), so H(p,q) ≤ H(p)", "H(p,q) = D_KL(p‖q), independent of H(p)", "H(p,q) always equals H(p), since cross-entropy depends only on the true distribution p", "H(p,q) = H(p) + D_KL(p‖q), so H(p,q) ≥ H(p), with equality iff p=q"],
    correct: 3,
  },
  {
    area: 'ml',
    zh: "在偏差-方差分解中，对一个回归模型在测试点的期望平方误差为 E[(y - f̂(x))²] = Bias² + Variance + 不可约噪声。现已知某模型的 Bias² = 0.04、Variance = 0.09，且数据本身的噪声方差为 0.01。若我们通过加强正则化使模型复杂度下降，从而把 Variance 减半到 0.045，但 Bias² 因此上升到 0.07，下列关于总期望测试误差变化的判断哪个正确？",
    en: "In the bias-variance decomposition, the expected squared error at a test point is E[(y - f̂(x))²] = Bias² + Variance + irreducible noise. A model has Bias² = 0.04, Variance = 0.09, and the data noise variance is 0.01. If stronger regularization halves the Variance to 0.045 but raises Bias² to 0.07, which statement about the change in total expected test error is correct?",
    optionsZh: ["总误差从 0.13 升到 0.115，正则化在此点恶化了泛化", "总误差不变，因为不可约噪声主导了误差", "总误差从 0.14 降到 0.125，正则化在此点改善了泛化", "总误差从 0.14 升到 0.16，因为偏差上升幅度大于方差下降"],
    optionsEn: ["Total error rises from 0.13 to 0.115, so regularization hurts generalization at this point", "Total error is unchanged because irreducible noise dominates", "Total error drops from 0.14 to 0.125, so regularization improves generalization at this point", "Total error rises from 0.14 to 0.16 because the bias increase exceeds the variance decrease"],
    correct: 2,
  },
  {
    area: 'ml',
    zh: "关于 L1 与 L2 正则化为何 L1 倾向产生稀疏解，下列从优化几何/次梯度角度的解释哪个最准确？",
    en: "Regarding why L1 (but not L2) regularization tends to produce sparse solutions, which explanation from the optimization-geometry / subgradient perspective is most accurate?",
    optionsZh: ["L1 罚项 |w| 在 w=0 处不可导，其次梯度区间为 [-λ, λ]，只要无正则损失在 0 处的梯度绝对值不超过 λ，该权重的最优值就严格为 0", "L1 罚项处处可导且导数恒为常数，因此把权重平滑地推向零并最终精确归零", "L2 罚项的等高线是菱形，其顶点更易与损失等高线相切于坐标轴，故 L2 更稀疏", "L1 与 L2 都能产生精确为零的解，区别只在 L1 收敛更快"],
    optionsEn: ["The L1 penalty |w| is non-differentiable at w=0, with subgradient interval [-λ, λ]; whenever the unregularized loss's gradient magnitude at 0 does not exceed λ, the optimal weight is exactly 0", "The L1 penalty is differentiable everywhere with a constant derivative, so it smoothly pushes weights to and exactly at zero", "The L2 penalty's contours are a diamond whose corners more easily meet loss contours on the axes, so L2 is sparser", "Both L1 and L2 produce exactly-zero solutions; the only difference is that L1 converges faster"],
    correct: 0,
  },
  {
    area: 'ml',
    zh: "对软间隔（soft-margin）SVM，下列关于支持向量与 KKT 条件的描述哪个正确？",
    en: "For a soft-margin SVM, which statement about support vectors and the KKT conditions is correct?",
    optionsZh: ["仅恰好落在间隔边界上（ξ=0 且 0<αᵢ<C）的点才是支持向量，被错分的点不算支持向量", "支持向量必为训练集中离决策面最远的点", "松弛变量 ξᵢ>0 的点对应 αᵢ=0，因此不影响决策边界", "所有满足 αᵢ>0 的点都是支持向量，包括落在间隔内或被错分的点（此时 αᵢ=C）；落在边界上的点 0<αᵢ<C"],
    optionsEn: ["Only points exactly on the margin boundary (ξ=0 and 0<αᵢ<C) are support vectors; misclassified points are not support vectors", "Support vectors are the training points farthest from the decision surface", "Points with slack ξᵢ>0 correspond to αᵢ=0 and thus do not affect the decision boundary", "All points with αᵢ>0 are support vectors, including those inside the margin or misclassified (where αᵢ=C); points on the boundary have 0<αᵢ<C"],
    correct: 3,
  },
  {
    area: 'ml',
    zh: "关于 Bagging（如随机森林）与 Boosting（如梯度提升）在偏差-方差上的作用机制，下列哪个判断正确？",
    en: "Regarding how Bagging (e.g., random forests) and Boosting (e.g., gradient boosting) act on bias and variance, which statement is correct?",
    optionsZh: ["Bagging 主要降低偏差，Boosting 主要降低方差", "Bagging 主要降低方差且对单棵树的偏差几乎不增；Boosting 通过逐步拟合残差主要降低偏差，但可能增大方差并对噪声更敏感", "两者都主要降低偏差，方差不变", "Bagging 的基学习器必须是高偏差弱学习器，Boosting 的基学习器必须是高方差强学习器"],
    optionsEn: ["Bagging mainly reduces bias, while Boosting mainly reduces variance", "Bagging mainly reduces variance with little increase in each tree's bias; Boosting mainly reduces bias by sequentially fitting residuals, but can increase variance and be more sensitive to noise", "Both mainly reduce bias and leave variance unchanged", "Bagging's base learners must be high-bias weak learners, while Boosting's must be high-variance strong learners"],
    correct: 1,
  },
  {
    area: 'ml',
    zh: "在类别极不平衡（正例占比 1%）的二分类问题中，下列关于评估指标选择的判断哪个最正确？",
    en: "In a highly imbalanced binary classification problem (positives are 1% of data), which statement about choosing evaluation metrics is most correct?",
    optionsZh: ["ROC-AUC 在极端不平衡下可能给出过于乐观的印象，因为 FPR 的分母（真负例）很大，少量假正例对 FPR 影响小；此时 Precision-Recall 曲线/PR-AUC 通常更能反映正类检测质量", "准确率（accuracy）是此场景最可靠的指标，因为它直接衡量整体正确比例", "PR-AUC 与 ROC-AUC 完全等价，二者在任何分布下都给出相同排序结论", "只要 ROC-AUC 高于 0.5，模型在正类上的精确率必然较高"],
    optionsEn: ["ROC-AUC can look overly optimistic under extreme imbalance because FPR's denominator (true negatives) is large, so a few false positives barely move FPR; the Precision-Recall curve / PR-AUC usually better reflects positive-class detection quality", "Accuracy is the most reliable metric here because it directly measures overall correctness", "PR-AUC and ROC-AUC are fully equivalent and give the same ranking conclusions under any distribution", "As long as ROC-AUC exceeds 0.5, precision on the positive class must be high"],
    correct: 0,
  },
  {
    area: 'ml',
    zh: "下列哪种做法会导致交叉验证产生过度乐观的性能估计（数据泄漏）？",
    en: "Which practice causes cross-validation to yield over-optimistic performance estimates (data leakage)?",
    optionsZh: ["在每个训练折内部独立拟合预处理，并仅用其参数变换对应验证折", "对时间序列采用按时间顺序的前向链式（forward-chaining）划分", "在每折内独立做缺失值插补，统计量只从该折训练部分估计", "在划分训练/验证折之前，先用全体数据拟合标准化（均值/方差）或特征选择，再进行交叉验证"],
    optionsEn: ["Fitting preprocessing independently within each training fold and using only its parameters to transform the corresponding validation fold", "Using forward-chaining (time-ordered) splits for time-series data", "Doing missing-value imputation independently within each fold, estimating statistics only from that fold's training portion", "Fitting standardization (mean/variance) or feature selection on the entire dataset before splitting into train/validation folds, then running cross-validation"],
    correct: 3,
  },
  {
    area: 'ml',
    zh: "对一个标准化后的数据集做 PCA，协方差矩阵的特征值（降序）为 [4.0, 1.0, 0.5, 0.5]。下列关于主成分与降维的判断哪个正确？",
    en: "Running PCA on a standardized dataset yields covariance eigenvalues (descending) [4.0, 1.0, 0.5, 0.5]. Which statement about the principal components / dimensionality reduction is correct?",
    optionsZh: ["前两个主成分保留约 50% 的方差，因为它们是 4 个成分中的 2 个", "前两个主成分保留了总方差的约 83.3%，且各主成分方向相互正交、对应的得分（投影）互不相关", "主成分的得分（投影坐标）一般彼此高度相关，因为它们来自同一数据", "由于后两个特征值相等（0.5），PCA 无法定义任何主成分方向，必须放弃 PCA"],
    optionsEn: ["The top two PCs retain about 50% of variance because they are 2 of the 4 components", "The top two PCs retain about 83.3% of total variance, and the PC directions are mutually orthogonal with corresponding scores (projections) being uncorrelated", "PC scores (projection coordinates) are generally highly correlated since they come from the same data", "Because the last two eigenvalues are equal (0.5), PCA cannot define any PC direction and must be abandoned"],
    correct: 1,
  },
  {
    area: 'ml',
    zh: "关于高斯混合模型（GMM）的 EM 算法，下列哪个描述正确？",
    en: "Regarding the EM algorithm for Gaussian Mixture Models (GMM), which statement is correct?",
    optionsZh: ["EM 保证收敛到对数似然的全局最优解，与初始化无关", "M 步直接对边际似然求导置零，因此不需要 E 步的后验责任", "E 步用当前参数计算每个样本对各成分的后验责任（responsibility），M 步在这些责任下最大化期望完全数据对数似然；每次迭代保证观测数据对数似然不下降，但可能收敛到局部最优", "硬分配的 K-means 与 GMM-EM 完全等价，二者对所有协方差结构给出相同结果"],
    optionsEn: ["EM is guaranteed to reach the global optimum of the log-likelihood, independent of initialization", "The M-step directly differentiates the marginal likelihood and sets it to zero, so the E-step's posterior responsibilities are unnecessary", "The E-step computes each sample's posterior responsibilities over components under current parameters; the M-step maximizes the expected complete-data log-likelihood under those responsibilities; each iteration guarantees the observed-data log-likelihood does not decrease, but may converge to a local optimum", "Hard-assignment K-means and GMM-EM are exactly equivalent and give identical results for all covariance structures"],
    correct: 2,
  },
  {
    area: 'ml',
    zh: "在有向概率图模型（贝叶斯网络）中，考虑 V 形结构（collider）A → C ← B，其中 A 与 B 边缘独立。下列关于 d-分离/条件独立的判断哪个正确？",
    en: "In a directed probabilistic graphical model (Bayesian network), consider a collider (v-structure) A → C ← B where A and B are marginally independent. Which statement about d-separation / conditional independence is correct?",
    optionsZh: ["不观测 C 时 A 与 B 独立；一旦以 C（或 C 的任一后代）为条件，A 与 B 一般变为条件相关（解释消除 / explaining away）", "以 C 为条件会使 A 与 B 变得条件独立，因为 C 阻断了路径", "无论是否观测 C，A 与 B 始终独立，因为它们没有共同父节点", "观测 C 的后代不影响 A 与 B 的独立性，只有直接观测 C 才有影响"],
    optionsEn: ["When C is unobserved, A and B are independent; once we condition on C (or any descendant of C), A and B generally become dependent (explaining away)", "Conditioning on C makes A and B conditionally independent because C blocks the path", "A and B are always independent whether or not C is observed, since they share no common parent", "Observing a descendant of C does not affect A–B independence; only observing C directly does"],
    correct: 0,
  },
  {
    area: 'ml',
    zh: "对决策树用于分类的不纯度度量，下列关于 Gini 不纯度与信息增益（基于熵）的判断哪个正确？",
    en: "For impurity measures in classification trees, which statement about Gini impurity versus information gain (entropy-based) is correct?",
    optionsZh: ["Gini 不纯度在纯节点处取最大值，在均匀分布处取最小值，与熵的方向相反", "二者都在节点纯（单一类别）时取最小值 0、在类别均匀分布时取最大值，且都偏好能降低子节点加权不纯度的划分；它们常给出相近的树，差异通常不大", "信息增益总是偏好把样本均匀分到各子节点的划分", "只有熵能保证选出的划分使加权子节点不纯度严格为 0，Gini 无法做到"],
    optionsEn: ["Gini impurity is maximized at a pure node and minimized at a uniform distribution, opposite to entropy", "Both attain their minimum 0 at a pure node and maximum at a uniform class distribution, and both favor splits that reduce the weighted child impurity; they usually yield similar trees with minor differences", "Information gain always prefers splits that distribute samples uniformly across children", "Only entropy can guarantee selecting a split whose weighted child impurity is exactly 0; Gini cannot"],
    correct: 1,
  },
  {
    area: 'dl',
    zh: "对一个标量损失 L 和某层线性变换 z = Wx（W 为矩阵，x 为列向量），反向传播时已知上游梯度 ∂L/∂z。关于 ∂L/∂W 和 ∂L/∂x 的正确表达式是哪一个？",
    en: "For a scalar loss L and a linear layer z = Wx (W a matrix, x a column vector), during backpropagation the upstream gradient ∂L/∂z is known. Which expression for ∂L/∂W and ∂L/∂x is correct?",
    optionsZh: ["∂L/∂W = (∂L/∂z) xᵀ，∂L/∂x = Wᵀ (∂L/∂z)", "∂L/∂W = xᵀ (∂L/∂z)，∂L/∂x = W (∂L/∂z)", "∂L/∂W = (∂L/∂z)ᵀ x，∂L/∂x = (∂L/∂z) Wᵀ", "∂L/∂W = W (∂L/∂z)，∂L/∂x = (∂L/∂z) xᵀ"],
    optionsEn: ["∂L/∂W = (∂L/∂z) xᵀ, ∂L/∂x = Wᵀ (∂L/∂z)", "∂L/∂W = xᵀ (∂L/∂z), ∂L/∂x = W (∂L/∂z)", "∂L/∂W = (∂L/∂z)ᵀ x, ∂L/∂x = (∂L/∂z) Wᵀ", "∂L/∂W = W (∂L/∂z), ∂L/∂x = (∂L/∂z) xᵀ"],
    correct: 0,
  },
  {
    area: 'dl',
    zh: "He（Kaiming）初始化相对 Xavier（Glorot）初始化，核心区别在于前向方差守恒时所用的方差系数。对一个 fan_in 个输入、使用 ReLU 激活的线性层，He 初始化建议权重方差为下列哪个？",
    en: "Compared to Xavier (Glorot) initialization, He (Kaiming) initialization differs in the variance factor used to preserve forward variance. For a linear layer with fan_in inputs followed by ReLU, He initialization sets the weight variance to which of the following?",
    optionsZh: ["1 / fan_in", "1 / (fan_in + fan_out)", "2 / fan_in", "2 / (fan_in + fan_out)"],
    optionsEn: ["1 / fan_in", "1 / (fan_in + fan_out)", "2 / fan_in", "2 / (fan_in + fan_out)"],
    correct: 2,
  },
  {
    area: 'dl',
    zh: "关于 BatchNorm 与 LayerNorm 在训练与推理阶段的行为，下列哪项陈述是正确的？",
    en: "Regarding BatchNorm and LayerNorm behavior at training versus inference, which statement is correct?",
    optionsZh: ["LayerNorm 在推理时也要切换到用训练期累积的移动平均统计量", "BatchNorm 推理时使用训练期累积的移动平均均值/方差，而 LayerNorm 训练和推理都用当前样本自身的统计量", "BatchNorm 和 LayerNorm 的可学习仿射参数 γ、β 都会随 batch 大小自动改变形状", "LayerNorm 沿 batch 维归一化，因此对小 batch 比 BatchNorm 更敏感"],
    optionsEn: ["LayerNorm must also switch to running-average statistics accumulated during training at inference time", "BatchNorm uses running-average mean/variance accumulated during training at inference, whereas LayerNorm uses the current sample's own statistics in both training and inference", "Both BatchNorm and LayerNorm have learnable affine parameters γ, β whose shape changes automatically with batch size", "LayerNorm normalizes along the batch dimension, making it more sensitive to small batches than BatchNorm"],
    correct: 1,
  },
  {
    area: 'dl',
    zh: "Adam 优化器在第 t 步对一阶矩 m_t 和二阶矩 v_t 做偏差校正。关于偏差校正的作用，下列哪项最准确？",
    en: "Adam performs bias correction on the first moment m_t and second moment v_t at step t. Which statement most accurately describes the purpose of this bias correction?",
    optionsZh: ["偏差校正用来抵消 m、v 因零初始化在早期被低估（偏向 0）的问题，t→∞ 时校正因子 1/(1−βᵗ)→1 影响消失", "偏差校正等价于把学习率随步数线性衰减", "偏差校正修正的是梯度本身的方差，使其变为单位方差", "偏差校正只对二阶矩 v_t 必要，对一阶矩 m_t 没有作用"],
    optionsEn: ["Bias correction counteracts the early-step underestimation (bias toward 0) of m and v caused by zero initialization; as t→∞ the factor 1/(1−βᵗ)→1 and its effect vanishes", "Bias correction is equivalent to a linear decay of the learning rate with step count", "Bias correction normalizes the variance of the gradient itself to unit variance", "Bias correction is only needed for the second moment v_t and has no effect on the first moment m_t"],
    correct: 0,
  },
  {
    area: 'dl',
    zh: "标准（inverted）Dropout 在训练时以保留概率 p 随机保留激活，并对保留的激活乘以 1/p。这样做的核心原因是什么？",
    en: "Standard (inverted) Dropout keeps activations with probability p during training and scales kept activations by 1/p. What is the core reason for this scaling?",
    optionsZh: ["为了让被丢弃的神经元梯度也能反传", "为了增大训练时的有效学习率", "为了使每层输出严格服从标准正态分布", "为了在推理时无需任何改动即可直接关闭 Dropout 并使期望激活值与训练时一致"],
    optionsEn: ["So that gradients can flow back through the dropped neurons", "To increase the effective learning rate during training", "To force each layer's output to follow a standard normal distribution", "So that at inference Dropout can simply be turned off with no other change while keeping the expected activation magnitude consistent with training"],
    correct: 3,
  },
  {
    area: 'dl',
    zh: "输入为 224×224 的特征图，经过一个 7×7 卷积（stride=2, padding=3），再经过一个 3×3 最大池化（stride=2, padding=1）。最终输出的空间尺寸（高=宽）是多少？（用 floor((W−K+2P)/S)+1）",
    en: "An input feature map of 224×224 passes through a 7×7 convolution (stride=2, padding=3), then a 3×3 max pooling (stride=2, padding=1). What is the final spatial size (height = width)? (Use floor((W−K+2P)/S)+1)",
    optionsZh: ["55", "112", "28", "56"],
    optionsEn: ["55", "112", "28", "56"],
    correct: 3,
  },
  {
    area: 'dl',
    zh: "残差连接 y = x + F(x) 缓解深层网络梯度消失的核心机制，从反向传播角度看是哪一项？",
    en: "From the backpropagation perspective, what is the core mechanism by which a residual connection y = x + F(x) mitigates vanishing gradients in deep networks?",
    optionsZh: ["它把激活函数从 ReLU 换成了恒等映射，消除了非线性", "∂y/∂x = I + ∂F/∂x 中的恒等项使梯度有一条不被反复缩小的直通路径", "它通过对 x 做归一化把梯度范数固定为 1", "它减少了网络的总参数量从而避免过拟合导致的梯度消失"],
    optionsEn: ["It replaces ReLU with an identity map, removing all nonlinearity", "The identity term in ∂y/∂x = I + ∂F/∂x gives the gradient a direct path that is not repeatedly shrunk", "It normalizes x so the gradient norm is fixed to 1", "It reduces total parameters, thereby avoiding overfitting-induced vanishing gradients"],
    correct: 1,
  },
  {
    area: 'dl',
    zh: "关于 AdamW 相对于带 L2 正则的 Adam，下列哪项准确描述了其关键改进？",
    en: "Regarding AdamW versus Adam with L2 regularization, which statement accurately describes its key improvement?",
    optionsZh: ["AdamW 取消了二阶矩 v_t，仅保留一阶矩", "AdamW 用更大的 β2 使二阶矩更新更平滑", "AdamW 把权重衰减从梯度中解耦，直接作用于权重更新，避免衰减项被自适应学习率（除以 √v_t）缩放而失真", "AdamW 把 L2 正则换成了 L1 正则以获得稀疏解"],
    optionsEn: ["AdamW removes the second moment v_t, keeping only the first moment", "AdamW uses a larger β2 to smooth the second-moment update", "AdamW decouples weight decay from the gradient, applying it directly to the weight update, so the decay is not scaled (distorted) by the adaptive per-parameter learning rate (division by √v_t)", "AdamW replaces L2 regularization with L1 to obtain sparse solutions"],
    correct: 2,
  },
  {
    area: 'dl',
    zh: "堆叠三个 3×3 卷积（均 stride=1，无空洞），相对单个卷积，其等效感受野与所需参数（忽略通道数与偏置，仅比空间核）的对比，下列哪项正确？",
    en: "Stacking three 3×3 convolutions (each stride=1, no dilation), compared to a single convolution: which statement about the equivalent receptive field and the number of spatial-kernel parameters (ignoring channels and bias, comparing only spatial kernels) is correct?",
    optionsZh: ["等效感受野为 5×5，参数比单个 5×5 卷积少", "等效感受野为 9×9，参数与单个 9×9 卷积相同", "等效感受野为 7×7，但参数（27）多于单个 7×7 卷积（49）", "等效感受野为 7×7，参数（3×9=27）少于单个 7×7 卷积（49），且引入更多非线性"],
    optionsEn: ["The equivalent receptive field is 5×5, with fewer parameters than a single 5×5 conv", "The equivalent receptive field is 9×9, with the same parameters as a single 9×9 conv", "The equivalent receptive field is 7×7, but more parameters (27) than a single 7×7 conv (49)", "The equivalent receptive field is 7×7, with fewer parameters (3×9=27) than a single 7×7 conv (49), and it introduces more nonlinearity"],
    correct: 3,
  },
  {
    area: 'dl',
    zh: "关于 ReLU、GELU、SiLU(Swish) 三个激活函数的性质，下列哪项陈述是正确的？",
    en: "Regarding the properties of the ReLU, GELU, and SiLU (Swish) activations, which statement is correct?",
    optionsZh: ["三者在 x<0 区域的输出恒为 0，因此都存在“死亡神经元”问题", "GELU 和 SiLU 都是处处可导的光滑函数，且在 x<0 的某区间允许小幅负输出（非单调），而 ReLU 在 x=0 不可导且负区恒为 0", "SiLU 定义为 x·tanh(x)，是 GELU 的精确等价形式", "ReLU 在正区间会饱和，导致梯度消失，而 GELU 不会"],
    optionsEn: ["All three output exactly 0 for x<0, so all suffer from the dying-neuron problem", "Both GELU and SiLU are smooth and differentiable everywhere and allow a small negative output (non-monotonic) over some interval of x<0, whereas ReLU is non-differentiable at x=0 and is exactly 0 for negative inputs", "SiLU is defined as x·tanh(x) and is an exact equivalent of GELU", "ReLU saturates in the positive region, causing vanishing gradients, while GELU does not"],
    correct: 1,
  },
  {
    area: 'nlp',
    zh: "在 scaled dot-product attention 中，假设 query 和 key 的各维分量独立同分布、均值 0、方差 1，点积 q·k 的方差为 d_k。除以 √d_k 的根本目的是什么？",
    en: "In scaled dot-product attention, assume query and key components are i.i.d. with mean 0 and variance 1, so the dot product q·k has variance d_k. What is the fundamental purpose of dividing by √d_k?",
    optionsZh: ["把 logits 的方差归一化到约 1，避免 softmax 进入饱和区导致梯度消失", "保证注意力权重之和精确等于 1，否则 softmax 不归一化", "补偿多头拼接后维度变大带来的数值溢出", "使点积结果恒为非负，从而满足概率分布的要求"],
    optionsEn: ["Normalize the variance of logits to about 1, preventing softmax from entering its saturated region and causing vanishing gradients", "Ensure attention weights sum exactly to 1, which softmax would otherwise fail to enforce", "Compensate for numerical overflow caused by the larger dimension after multi-head concatenation", "Make dot products always non-negative so they satisfy the requirements of a probability distribution"],
    correct: 0,
  },
  {
    area: 'nlp',
    zh: "关于多头注意力（h 个头，每头维度 d_k=d_model/h），以下哪个说法最准确？",
    en: "Regarding multi-head attention (h heads, each of dimension d_k=d_model/h), which statement is most accurate?",
    optionsZh: ["多头的本质是对同一注意力矩阵做 h 次独立 dropout 以提升鲁棒性", "头数越多，每个头能捕获的上下文长度越长", "多头的总计算量与参数量远大于等维度的单头，因此用计算换表达力", "由于每头维度被切分，多头允许模型在不同子空间并行关注不同表示，总计算量与等维单头相近"],
    optionsEn: ["Multi-head essentially applies h independent dropouts to the same attention matrix to improve robustness", "The more heads, the longer the context length each head can capture", "Total compute and parameters of multi-head greatly exceed a single full-dimension head, trading compute for expressiveness", "Because each head's dimension is split, multi-head lets the model attend to different representation subspaces in parallel, with total compute close to a single full-dimension head"],
    correct: 3,
  },
  {
    area: 'nlp',
    zh: "关于 RoPE（旋转位置编码）与 ALiBi，以下对比哪一项正确？",
    en: "Comparing RoPE (Rotary Position Embedding) and ALiBi, which statement is correct?",
    optionsZh: ["RoPE 在 softmax 之后给注意力权重乘以位置相关系数；ALiBi 把正弦向量加到 value 上", "两者都把可学习的绝对位置向量加到 token embedding，再送入第一层", "RoPE 通过对 Q、K 施加随位置变化的旋转使点积只依赖相对位置；ALiBi 在注意力 logits 上加一个随相对距离线性增长的负偏置", "ALiBi 需要为每个相对距离学习一个独立的标量偏置，因此参数随序列长度增长"],
    optionsEn: ["RoPE multiplies attention weights by a position-dependent factor after softmax; ALiBi adds a sinusoidal vector to the values", "Both add learnable absolute position vectors to token embeddings before the first layer", "RoPE applies a position-dependent rotation to Q and K so that their dot product depends only on relative position; ALiBi adds a negative bias to attention logits that grows linearly with relative distance", "ALiBi learns an independent scalar bias for each relative distance, so its parameters grow with sequence length"],
    correct: 2,
  },
  {
    area: 'nlp',
    zh: "对长度为 n 的序列，标准自注意力的时间与显存复杂度主要瓶颈是什么？",
    en: "For a sequence of length n, what is the dominant bottleneck in time and memory complexity of standard self-attention?",
    optionsZh: ["时间 O(n log n)，因为注意力可用 FFT 加速", "时间和存储注意力矩阵的显存均为 O(n²)（与每步特征维 d 相乘），随序列长度二次增长", "时间 O(n·d²)，显存 O(n)，与序列长度仅线性相关", "瓶颈在前馈层 O(n·d²)，注意力部分可忽略"],
    optionsEn: ["Time O(n log n), because attention can be accelerated with the FFT", "Both the time and the memory for storing the attention matrix are O(n²) (multiplied by per-step feature dim d), growing quadratically with sequence length", "Time O(n·d²), memory O(n), only linear in sequence length", "The bottleneck is the feed-forward layer O(n·d²); the attention part is negligible"],
    correct: 1,
  },
  {
    area: 'nlp',
    zh: "把 LoRA 应用到一个 d×d 的权重矩阵 W，用 W+BA 近似更新，其中 A∈ℝ^{r×d}、B∈ℝ^{d×r}，r≪d。关于其原理，哪项正确？",
    en: "Applying LoRA to a d×d weight matrix W with update W+BA, where A∈ℝ^{r×d}, B∈ℝ^{d×r}, and r≪d. Which statement about its principle is correct?",
    optionsZh: ["LoRA 直接对 W 做 SVD 截断并只微调最大的 r 个奇异值，因此推理时必须保留 SVD", "BA 的秩可达 d，因此能表达对 W 的任意更新而不损失表达力", "为保证收敛，A 与 B 都必须用相同的非零随机值初始化，否则梯度为零", "训练参数从 d² 降到 2rd；初始化时通常令 B=0（或 A=0），使初始增量为零，不破坏预训练权重"],
    optionsEn: ["LoRA performs SVD truncation on W and fine-tunes only the top r singular values, so SVD must be retained at inference", "BA can have rank up to d, so it can express any arbitrary update to W without loss of expressiveness", "For convergence, A and B must both be initialized with the same nonzero random values, or the gradients will be zero", "Trainable parameters drop from d² to 2rd; B (or A) is typically initialized to zero so the initial increment is zero and does not disturb the pretrained weights"],
    correct: 3,
  },
  {
    area: 'nlp',
    zh: "关于自回归生成中的 KV cache，以下哪项描述最准确？",
    en: "Regarding the KV cache in autoregressive generation, which description is most accurate?",
    optionsZh: ["缓存的是每步的 softmax 注意力权重，省去重复的 softmax 计算", "KV cache 缓存 Q、K、V 三者，因此显存与缓存的 Q 成正比", "缓存已生成 token 的 K 和 V，使每步只需为新 token 计算 Q 并与历史 K、V 做注意力，把单步从 O(n²) 降到 O(n)", "它通过缓存 logits 使模型在 prefill 阶段无需重新前向，从而支持双向注意力"],
    optionsEn: ["It caches the softmax attention weights of each step, saving redundant softmax computation", "The KV cache stores Q, K, and V, so memory is proportional to the cached Q", "It caches K and V of already-generated tokens, so each step only computes Q for the new token and attends to historical K, V, reducing per-step cost from O(n²) to O(n)", "It caches logits so the model needs no re-forward during prefill, thereby enabling bidirectional attention"],
    correct: 2,
  },
  {
    area: 'nlp',
    zh: "对比 top-k 采样、top-p（nucleus）采样和温度 T，以下哪项正确？",
    en: "Comparing top-k sampling, top-p (nucleus) sampling, and temperature T, which statement is correct?",
    optionsZh: ["温度 T>1 使分布更尖锐、更确定；T<1 使分布更平坦", "top-k 固定保留 k 个候选；top-p 保留累积概率刚好达到 p 的最小候选集，候选数随分布形状动态变化", "top-p 在每步保留的候选数恒定，而 top-k 的候选数随上下文变化", "温度只缩放最终采样到的 token，对未被采样的候选没有影响"],
    optionsEn: ["Temperature T>1 makes the distribution sharper and more deterministic; T<1 makes it flatter", "top-k keeps a fixed k candidates; top-p keeps the smallest candidate set whose cumulative probability just reaches p, so the number of candidates varies dynamically with the distribution shape", "top-p keeps a constant number of candidates per step, whereas top-k's count varies with context", "Temperature only rescales the finally sampled token and has no effect on the non-sampled candidates"],
    correct: 1,
  },
  {
    area: 'nlp',
    zh: "关于 decoder 中的因果遮罩（causal mask），以下哪项正确？",
    en: "Regarding the causal mask in a decoder, which statement is correct?",
    optionsZh: ["它在 softmax 之前把未来位置的注意力 logits 设为 −∞，使这些位置 softmax 后权重为 0，从而防止信息泄漏", "它在 softmax 之后把未来位置的权重直接置 0 并重新归一化，效果与 −∞ 完全等价且更稳定", "它遮蔽的是 padding token，与位置先后无关", "因果遮罩使训练时无法并行，必须逐 token 前向"],
    optionsEn: ["Before softmax it sets the attention logits of future positions to −∞, so those positions get zero weight after softmax, preventing information leakage", "After softmax it directly zeroes the weights of future positions and renormalizes, which is exactly equivalent to −∞ and more stable", "It masks padding tokens and is unrelated to position ordering", "The causal mask makes training non-parallelizable, requiring per-token forward passes"],
    correct: 0,
  },
  {
    area: 'nlp',
    zh: "关于 Pre-LN 与 Post-LN Transformer，以下哪项描述最准确？",
    en: "Regarding Pre-LN versus Post-LN Transformers, which description is most accurate?",
    optionsZh: ["Post-LN 把 LayerNorm 放在残差相加之前，因此残差是恒等映射，深层训练天然稳定", "两者数学上完全等价，区别仅是实现习惯，对训练稳定性无影响", "Pre-LN 因为在子层后做归一化，会放大梯度，必须配合更长的 warmup 才能训练", "Pre-LN 把 LayerNorm 放在子层之前、残差路径保持恒等，使深层梯度更稳定，通常可去掉或缩短 warmup；Post-LN 训练更难但收敛后表达常更强"],
    optionsEn: ["Post-LN places LayerNorm before the residual addition, so the residual is an identity mapping, making deep training inherently stable", "The two are mathematically equivalent, differing only in implementation convention with no effect on training stability", "Pre-LN normalizes after the sublayer, which amplifies gradients and requires a longer warmup to train", "Pre-LN places LayerNorm before each sublayer with an identity residual path, giving more stable deep gradients and often allowing warmup to be removed or shortened; Post-LN is harder to train but, once converged, often yields stronger representations"],
    correct: 3,
  },
  {
    area: 'nlp',
    zh: "对比 GPT 式因果（单向）语言模型与 BERT 式双向编码器，以下哪项正确？",
    en: "Comparing a GPT-style causal (unidirectional) language model with a BERT-style bidirectional encoder, which statement is correct?",
    optionsZh: ["BERT 用因果遮罩做自回归预测，因此可直接用于逐 token 文本生成", "两者都用 next-token 预测预训练，唯一区别是参数规模", "GPT 每个位置只能看左侧上下文，适合自回归生成；BERT 双向可见、用掩码语言建模(MLM)预训练，更适合需要全局上下文的理解类任务，但不能直接做自回归生成", "BERT 的双向注意力意味着它在预测被掩码 token 时也用到了该 token 自身的输入表示，从而避免信息泄漏"],
    optionsEn: ["BERT uses a causal mask for autoregressive prediction and can therefore be used directly for token-by-token text generation", "Both are pretrained with next-token prediction; the only difference is parameter scale", "GPT sees only left context at each position, suiting autoregressive generation; BERT is bidirectional, pretrained with masked language modeling (MLM), better for understanding tasks needing global context, but cannot directly do autoregressive generation", "BERT's bidirectional attention means that when predicting a masked token it also uses that token's own input representation, thereby avoiding information leakage"],
    correct: 2,
  },
  {
    area: 'applied',
    zh: "在标准的投机解码（speculative decoding, Leviathan et al.）中，草稿模型每步提议 γ=4 个 token，目标模型逐位置验证。假设每个位置的接受概率独立同分布为 α=0.8，那么每个验证步骤平均能产出多少个 token（含被接受后追加的那个由目标模型采样的“奖励”token）？提示：期望 token 数 = (1 − α^(γ+1)) / (1 − α)。",
    en: "In standard speculative decoding (Leviathan et al.), the draft model proposes γ=4 tokens per step and the target model verifies each position. Assuming i.i.d. per-position acceptance probability α=0.8, how many tokens are produced on average per verification step (including the bonus token sampled from the target model after the accepted prefix)? Hint: expected tokens = (1 − α^(γ+1)) / (1 − α).",
    optionsZh: ["恰好 4 个 token（草稿全部被接受）", "约 2.95 个 token", "约 3.36 个 token", "约 5.00 个 token"],
    optionsEn: ["Exactly 4 tokens (the whole draft is always accepted)", "About 2.95 tokens", "About 3.36 tokens", "About 5.00 tokens"],
    correct: 2,
  },
  {
    area: 'applied',
    zh: "关于 DPO（Direct Preference Optimization）相对经典 RLHF（奖励模型 + PPO）的简化，下列哪一项描述最准确？",
    en: "Regarding how DPO (Direct Preference Optimization) simplifies the classic RLHF pipeline (reward model + PPO), which statement is the most accurate?",
    optionsZh: ["DPO 用一个分类损失直接在偏好数据上优化策略，无需显式训练独立奖励模型，也无需在线采样做 RL，但仍隐式假设 Bradley-Terry 偏好模型且依赖一个参考策略做 KL 正则", "DPO 完全去掉了 KL 正则项，因此不再需要参考模型，训练只依赖正负样本对", "DPO 仍然需要先训练奖励模型，只是把 PPO 换成了更稳定的离线策略梯度", "DPO 通过在线 rollout 生成新样本来估计优势函数，比 PPO 采样效率更高"],
    optionsEn: ["DPO directly optimizes the policy on preference data with a classification-style loss, requiring neither a separately trained reward model nor on-policy RL sampling, while still implicitly assuming a Bradley-Terry preference model and relying on a reference policy for KL regularization", "DPO removes the KL term entirely, so it no longer needs a reference model and trains only on positive/negative pairs", "DPO still trains a reward model first and only replaces PPO with a more stable offline policy gradient", "DPO generates fresh samples via online rollouts to estimate the advantage function, making it more sample-efficient than PPO"],
    correct: 0,
  },
  {
    area: 'applied',
    zh: "RAG 中的 HyDE（Hypothetical Document Embeddings）方法，其核心机制是什么？",
    en: "In RAG, what is the core mechanism of the HyDE (Hypothetical Document Embeddings) method?",
    optionsZh: ["对检索回来的文档做假设性改写后再喂给生成模型，以提升答案忠实度", "用稀疏向量（如 BM25）生成假设文档，再与稠密向量做混合检索", "在嵌入前对 query 做多次 dropout 扰动以生成多个假设嵌入并平均", "先让 LLM 根据 query 生成一段“假设性答案文档”，再用该文档的嵌入去检索，从而缓解 query 与文档在嵌入空间中的措辞鸿沟"],
    optionsEn: ["It rewrites retrieved documents hypothetically before feeding them to the generator to improve answer faithfulness", "It uses sparse vectors (e.g., BM25) to generate hypothetical documents, then does hybrid retrieval with dense vectors", "It applies multiple dropout perturbations to the query before embedding to produce and average several hypothetical embeddings", "It first has the LLM generate a hypothetical answer document for the query, then uses that document's embedding to retrieve, narrowing the lexical/semantic gap between query and documents in embedding space"],
    correct: 3,
  },
  {
    area: 'applied',
    zh: "关于稠密检索（dense retrieval, 如双塔 DPR）与稀疏检索（sparse, 如 BM25）的对比，下列哪项判断最准确？",
    en: "Comparing dense retrieval (e.g., dual-encoder DPR) with sparse retrieval (e.g., BM25), which judgment is most accurate?",
    optionsZh: ["稠密检索在所有任务上都严格优于 BM25，因为它捕获语义而非词面", "稠密检索擅长语义/同义匹配但在域外稀有实体、精确关键词（如型号、ID）上常逊于 BM25；二者互补，混合检索通常优于单一方法", "BM25 因为是稠密向量内积，所以对长尾术语泛化更好", "稠密检索不需要训练数据，是无监督的，而 BM25 需要标注的相关性对"],
    optionsEn: ["Dense retrieval is strictly better than BM25 on all tasks because it captures semantics rather than surface form", "Dense retrieval excels at semantic/synonym matching but often underperforms BM25 on out-of-domain rare entities and exact keywords (e.g., model numbers, IDs); they are complementary and hybrid retrieval usually beats either alone", "BM25 generalizes better to long-tail terms because it uses dense vector inner products", "Dense retrieval needs no training data and is unsupervised, whereas BM25 requires labeled relevance pairs"],
    correct: 1,
  },
  {
    area: 'applied',
    zh: "在评估 RAG 检索质量时，给定单个 query 的相关文档排序，下列指标中哪一个对“相关文档排得越靠前奖励越高、且对排名位置敏感”刻画得最直接？",
    en: "When evaluating RAG retrieval quality for a single query's ranked list, which metric most directly captures 'rewarding relevant documents that appear earlier, with sensitivity to rank position'?",
    optionsZh: ["Precision@k，只看前 k 个中相关比例，对 k 内部的排序顺序不敏感", "Recall@k，衡量前 k 个覆盖了多少相关文档，与具体排名位置无关", "nDCG（归一化折损累积增益），通过对数位置折损让靠前命中权重更大，并用理想排序归一化", "F1 分数，是 Precision 与 Recall 的调和平均，天然编码排名位置"],
    optionsEn: ["Precision@k, which only measures the fraction of relevant items in the top-k and is insensitive to the ordering within k", "Recall@k, which measures how many relevant documents are covered in the top-k, independent of exact rank position", "nDCG (normalized Discounted Cumulative Gain), which applies a logarithmic position discount so earlier hits weigh more, normalized by the ideal ordering", "F1 score, the harmonic mean of Precision and Recall, which inherently encodes rank position"],
    correct: 2,
  },
  {
    area: 'applied',
    zh: "关于 ReAct 范式与单纯的 Chain-of-Thought（CoT），以及“反思（Reflexion）”机制，下列哪项理解最准确？",
    en: "Regarding the ReAct paradigm versus plain Chain-of-Thought (CoT), and the 'Reflexion' mechanism, which understanding is most accurate?",
    optionsZh: ["ReAct 通过梯度反向传播把工具调用结果回传以更新参数，本质是一种在线微调", "Reflexion 与 ReAct 等价，都是在单次前向中完成，不涉及多轮重试或外部记忆", "CoT 因为引入了外部工具调用，所以比 ReAct 更不容易产生幻觉", "ReAct 将推理（thought）与动作（action，如调用工具/检索）交替进行，用外部观测（observation）反馈来约束推理、减少幻觉；Reflexion 则在任务失败后生成自然语言反思并写入记忆，用于后续重试，而非更新模型权重"],
    optionsEn: ["ReAct backpropagates tool-call results to update parameters and is essentially a form of online fine-tuning", "Reflexion is equivalent to ReAct: both complete in a single forward pass with no multi-attempt retries or external memory", "CoT, because it introduces external tool calls, is less prone to hallucination than ReAct", "ReAct interleaves reasoning (thought) with actions (e.g., tool calls/retrieval), using external observations to ground reasoning and reduce hallucination; Reflexion, after a failed task, generates a natural-language self-reflection stored in memory to guide later retries, rather than updating model weights"],
    correct: 3,
  },
  {
    area: 'applied',
    zh: "关于 GPTQ 与 AWQ 这两种后训练权重量化（PTQ）方法的差异，下列哪项最准确？",
    en: "Regarding the difference between GPTQ and AWQ post-training weight quantization (PTQ) methods, which statement is most accurate?",
    optionsZh: ["GPTQ 基于近似二阶（Hessian）信息逐列量化并补偿误差；AWQ 观察到少量“显著权重”由激活幅度决定，通过按通道缩放保护这些权重，二者都属于仅权重量化", "AWQ 量化激活而 GPTQ 量化权重，因此 AWQ 是激活量化方法", "GPTQ 需要全量重训练，而 AWQ 完全不需要任何校准数据", "两者都要求量化感知训练（QAT），无法在已训练好的模型上离线进行"],
    optionsEn: ["GPTQ quantizes column-by-column using approximate second-order (Hessian) information with error compensation; AWQ observes that a small set of 'salient weights' is determined by activation magnitudes and protects them via per-channel scaling—both are weight-only methods", "AWQ quantizes activations while GPTQ quantizes weights, so AWQ is an activation-quantization method", "GPTQ requires full retraining, whereas AWQ needs no calibration data at all", "Both require quantization-aware training (QAT) and cannot be applied offline to an already-trained model"],
    correct: 0,
  },
  {
    area: 'applied',
    zh: "vLLM 的 PagedAttention 与连续批处理（continuous batching）共同解决了 LLM 服务中的什么核心瓶颈？",
    en: "What core bottleneck in LLM serving do vLLM's PagedAttention and continuous batching jointly address?",
    optionsZh: ["通过把注意力矩阵分块计算来降低注意力的 O(n²) 计算复杂度到 O(n log n)", "KV cache 的内存碎片与利用率：PagedAttention 用类似虚拟内存分页的方式按块分配 KV，消除内部/外部碎片并支持前缀共享；连续批处理在序列完成后立即换入新请求，提升 GPU 吞吐", "降低模型权重的显存占用，使 70B 模型能放进单张 24GB 显卡", "通过分页机制把 KV cache 卸载到磁盘，从而支持无限长上下文"],
    optionsEn: ["It reduces attention's O(n²) compute complexity to O(n log n) by computing the attention matrix in blocks", "KV cache memory fragmentation and utilization: PagedAttention allocates KV in blocks like virtual-memory paging, eliminating internal/external fragmentation and enabling prefix sharing, while continuous batching swaps in new requests the moment sequences finish, raising GPU throughput", "It reduces model weight memory so a 70B model fits on a single 24GB GPU", "Paging offloads the KV cache to disk, enabling unlimited context length"],
    correct: 1,
  },
  {
    area: 'applied',
    zh: "在 RAG 流水线中加入“重排序器（reranker，如基于 cross-encoder 的模型）”，相对于仅用双塔（bi-encoder）稠密检索，主要权衡是什么？",
    en: "Adding a reranker (e.g., a cross-encoder model) to a RAG pipeline, compared to bi-encoder dense retrieval alone, involves which primary trade-off?",
    optionsZh: ["reranker 用预计算的文档向量做内积，所以比 bi-encoder 更快且更准", "reranker 取代了向量数据库，使得检索阶段不再需要任何近似最近邻索引", "cross-encoder 与 bi-encoder 精度相同，加 reranker 唯一好处是降低延迟", "cross-encoder 让 query 与每个候选文档拼接后联合编码，能建模细粒度交互、显著提升相关性精度，但因无法预计算文档向量、需对每个 query-doc 对在线前向，计算成本随候选数线性增长，故通常只对 bi-encoder 召回的 top-k 候选重排"],
    optionsEn: ["A reranker uses precomputed document vectors with inner products, so it is both faster and more accurate than a bi-encoder", "A reranker replaces the vector database, so the retrieval stage no longer needs any approximate-nearest-neighbor index", "A cross-encoder has the same accuracy as a bi-encoder; the only benefit of adding a reranker is lower latency", "A cross-encoder jointly encodes the query concatenated with each candidate document, modeling fine-grained interactions and substantially boosting relevance precision, but since document vectors cannot be precomputed it must run an online forward pass per query-doc pair—cost scales linearly with candidates—so it is typically applied only to rerank the bi-encoder's top-k"],
    correct: 3,
  },
  {
    area: 'applied',
    zh: "关于 Anthropic 提出的 MCP（Model Context Protocol）与传统的“function calling / 工具调用”，下列哪项区分最准确？",
    en: "Regarding Anthropic's MCP (Model Context Protocol) versus traditional 'function calling / tool calling', which distinction is most accurate?",
    optionsZh: ["MCP 取代了 function calling：启用 MCP 后模型不再输出结构化的工具调用参数", "function calling 是一个跨厂商的网络传输协议，MCP 只是某个厂商的私有函数签名格式", "function calling 是单个模型 API 层面“模型决定调用哪个已声明函数并返回结构化参数”的能力；MCP 是一个开放的客户端-服务器协议，标准化外部工具、数据源和提示的发现与接入，使任意 MCP 客户端无需为每个应用定制集成即可复用同一套服务器", "MCP 在模型权重内部硬编码了可用工具，因此添加新工具必须重新训练模型"],
    optionsEn: ["MCP replaces function calling: once MCP is enabled the model no longer emits structured tool-call arguments", "Function calling is a cross-vendor network transport protocol, while MCP is one vendor's proprietary function-signature format", "Function calling is a model-API capability where the model decides which declared function to invoke and returns structured arguments; MCP is an open client-server protocol that standardizes discovery and connection of external tools, data sources, and prompts, so any MCP client can reuse the same servers without bespoke per-app integration", "MCP hardcodes available tools inside the model weights, so adding a new tool requires retraining the model"],
    correct: 2,
  },
];

/** Score (0-50) → entry-phase mapping. Boundaries scaled from the upstream
 *  0-10 mapping (×5): the harder 50-question quiz keeps the same proportions. */
export function entryPhase(total: number): number {
  if (total <= 15) return 1;
  if (total <= 25) return 3;
  if (total <= 35) return 7;
  if (total <= 45) return 11;
  return 14;
}

export type PathStatus = 'skip' | 'review' | 'do';

/** Per-phase status (Phase 0 always skip). A phase below the entry point is
 *  "Skip" if its area was solidly mastered, else "Review". */
export function phaseStatus(phaseNum: number, entry: number, areaScores: Record<string, number>): PathStatus {
  if (phaseNum === 0) return 'skip';
  if (phaseNum >= entry) return 'do';
  const partialArea = AREAS.find(
    (a) => a.reviewPhases.includes(phaseNum) && (areaScores[a.key] ?? 0) < SOLID_THRESHOLD,
  );
  return partialArea ? 'review' : 'skip';
}

export interface PlacementResult {
  v: 1;
  answers: number[];
  areaScores: Record<string, number>;
  total: number;
  entry: number;
  date: string;
}

const KEY = 'aes:placement:v1';
const listeners = new Set<() => void>();

function read(): PlacementResult | null {
  try {
    const raw = localStorage.getItem(KEY);
    if (!raw) return null;
    const parsed = JSON.parse(raw) as PlacementResult;
    return parsed?.v === 1 ? parsed : null;
  } catch {
    return null;
  }
}

let current: PlacementResult | null = read();

function commit(next: PlacementResult | null) {
  current = next;
  if (next) localStorage.setItem(KEY, JSON.stringify(next));
  else localStorage.removeItem(KEY);
  listeners.forEach((fn) => fn());
}

export function savePlacement(result: PlacementResult) {
  commit(result);
}

export function loadPlacement(): PlacementResult | null {
  return current;
}

export function clearPlacement() {
  commit(null);
}

export function usePlacement(): PlacementResult | null {
  return useSyncExternalStore(
    (cb) => {
      listeners.add(cb);
      return () => listeners.delete(cb);
    },
    () => current,
  );
}

// ── cloud sync (one row per user) ───────────────────────────────────
interface PlacementRow {
  user_id: string;
  answers: number[];
  area_scores: Record<string, number>;
  total: number;
  entry: number;
  taken_at: string;
}

export async function pushPlacementCloud(userId: string): Promise<void> {
  if (!cloudEnabled || !current) return;
  const row: PlacementRow = {
    user_id: userId,
    answers: current.answers,
    area_scores: current.areaScores,
    total: current.total,
    entry: current.entry,
    taken_at: current.date,
  };
  const { error } = await getSupabase().from('placement').upsert(row);
  if (error) console.warn('[placement] push failed', error);
}

export async function deletePlacementCloud(userId: string): Promise<void> {
  if (!cloudEnabled) return;
  const { error } = await getSupabase().from('placement').delete().eq('user_id', userId);
  if (error) console.warn('[placement] delete failed', error);
}

/** Pull on login: the newer of local/remote wins on both sides. */
export async function syncPlacementCloud(userId: string): Promise<void> {
  if (!cloudEnabled) return;
  const { data, error } = await getSupabase()
    .from('placement')
    .select('*')
    .eq('user_id', userId)
    .maybeSingle();
  if (error) {
    console.warn('[placement] pull failed', error);
    return;
  }
  const remote = data as PlacementRow | null;
  if (remote && (!current || remote.taken_at > current.date)) {
    commit({
      v: 1,
      answers: remote.answers,
      areaScores: remote.area_scores,
      total: remote.total,
      entry: remote.entry,
      date: remote.taken_at,
    });
  } else if (current && (!remote || current.date > remote.taken_at)) {
    await pushPlacementCloud(userId);
  }
}
