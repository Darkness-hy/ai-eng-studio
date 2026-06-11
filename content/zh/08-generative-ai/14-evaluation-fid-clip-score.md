# 评估 — FID、CLIP Score 与人类偏好

> 每个生成模型排行榜都会引用 FID、CLIP score 和来自人类偏好竞技场的胜率。这三个数字各有一种失效模式，存心的研究者都能钻空子。如果你不了解这些失效模式，就分不清真正的改进和刷分。

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 8 · 01 (Taxonomy), Phase 2 · 04 (Evaluation Metrics)
**Time:** ~45 minutes

## 问题背景

生成模型的优劣取决于*样本质量*和*条件遵循度*。两者都没有闭式的度量方法。你的模型要渲染 10,000 张图像；必须有某种机制为它们打分；而且这些分数要在不同模型家族、不同分辨率、不同架构之间都值得信任。经过 2014-2026 年的层层筛选，有三类指标存活了下来：

- **FID（Fréchet Inception Distance，弗雷歇 Inception 距离）。** 在 Inception 网络的特征空间中衡量真实分布与生成分布之间的距离。越低越好。
- **CLIP score。** 生成图像的 CLIP 图像嵌入与提示词的 CLIP 文本嵌入之间的余弦相似度。越高越好。衡量提示词遵循度。
- **人类偏好。** 让两个模型在同一提示词上正面对决，由人类（或 GPT-4 级别的模型）选出更好的一个，再聚合为 Elo 分数。

你还会见到：IS（inception score，基本已退役）、KID、CMMD、ImageReward、PickScore、HPSv2、MJHQ-30k。每一个都是为了修正前一个的某种缺陷。

## 核心概念

![FID, CLIP, and preference: three axes, different failure modes](../assets/evaluation.svg)

### FID — 样本质量

Heusel 等人（2017）。步骤：

1. 对 N 张真实图像和 N 张生成图像分别提取 Inception-v3 特征（2048 维）。
2. 对每个特征池拟合一个高斯分布：计算均值 `μ_r, μ_g` 和协方差 `Σ_r, Σ_g`。
3. FID = `||μ_r - μ_g||² + Tr(Σ_r + Σ_g - 2 · (Σ_r · Σ_g)^0.5)`。

解释：它是特征空间中两个多元高斯分布之间的弗雷歇距离。越低 = 两个分布越相似。

失效模式：
- **小 N 下有偏。** FID 是对特征分布的均方度量——N 太小会低估协方差，导致 FID 虚低。务必使用 N ≥ 10,000。
- **依赖 Inception。** Inception-v3 是在 ImageNet 上训练的。与 ImageNet 差距很大的领域（人脸、艺术作品、文字图像）算出的 FID 没有意义。应改用领域专用的特征提取器。
- **可被刷分。** 对 Inception 先验过拟合可以在视觉质量没有提升的情况下得到低 FID。用 CMMD（见下文）来对付它。

### CLIP score — 提示词遵循度

Radford 等人（2021）。对一张生成图像和它的提示词：

```
clip_score = cos_sim( CLIP_image(x_gen), CLIP_text(prompt) )
```

在 3 万张生成图像上取平均 → 得到一个可在模型之间比较的标量。

失效模式：
- **CLIP 自身的盲区。** CLIP 的组合推理能力很弱（"a red cube on a blue sphere" 经常失败）。模型可以在没有真正遵循复杂提示词的情况下，在 CLIP score 上排名靠前。
- **短提示词偏差。** 短提示词在真实数据中有更多匹配的 CLIP 图像，长提示词的 CLIP score 在机制上就会更低。
- **提示词刷分。** 在提示词里加上 "high quality, 4k, masterpiece" 会抬高 CLIP score，但并没有改善图文对应关系。

CMMD（Jayasumana 等人，2024）修复了其中部分问题：用 CLIP 特征替代 Inception 特征，用最大均值差异（maximum-mean discrepancy）替代弗雷歇距离。它更擅长察觉细微的质量差异。

### 人类偏好 — 真正的标尺

选定一组提示词。用模型 A 和模型 B 分别生成。把成对结果展示给人类（或一个强大的 LLM 评审）。将胜负聚合为 Elo 或 Bradley-Terry 分数。常见基准：

- **PartiPrompts（Google）**：1,600 条多样化提示词，覆盖 12 个类别。
- **HPSv2**：10.7 万条人类标注，被广泛用作自动化代理指标。
- **ImageReward**：13.7 万对提示词-图像偏好数据，MIT 许可。
- **PickScore**：在 Pick-a-Pic 的 260 万条偏好数据上训练。
- **Chatbot-Arena 式的图像竞技场**：https://imagearena.ai/ 等。

失效模式：
- **评审差异。** 非专家与专家的偏好不同。两者都要用。
- **提示词分布。** 精心挑选的提示词会偏向某一类模型。务必记录在案。
- **LLM 评审的奖励欺骗（reward hacking）。** GPT-4 评审会被"好看但不对"的输出骗到。需要与人类评估交叉验证。

## 组合使用

一份生产级评估报告应包含：

1. 在 1 万到 3 万个样本上、对照留出的真实分布计算 FID（样本质量）。
2. 在同一批样本上、对照各自提示词计算 CLIP score / CMMD（遵循度）。
3. 在盲测竞技场中与上一版模型对比的胜率（整体偏好）。
4. 失效模式分析：随机抽取 50 个输出，针对已知问题逐一标记（手部解剖结构、文字渲染、物体数量一致性）。

任何单一指标都是谎言。三个互相印证的指标加上人工定性审查，才算得上一个结论。

## 从零实现

`code/main.py` 在合成的"特征向量"上实现了 FID、类 CLIP score 和 Elo 聚合（我们用 4 维向量充当 Inception 特征的替身）。你会看到：

- 在小 N 和大 N 上分别计算 FID——直观感受偏差。
- 用特征池之间的余弦相似度充当 "CLIP score"。
- 基于合成偏好数据流的 Elo 更新规则。

### 第 1 步：四行代码实现 FID

```python
def fid(real_features, gen_features):
    mu_r, cov_r = mean_and_cov(real_features)
    mu_g, cov_g = mean_and_cov(gen_features)
    mean_diff = sum((a - b) ** 2 for a, b in zip(mu_r, mu_g))
    trace_term = trace(cov_r) + trace(cov_g) - 2 * sqrt_cov_product(cov_r, cov_g)
    return mean_diff + trace_term
```

### 第 2 步：CLIP 式余弦相似度

```python
def clip_like(image_feat, text_feat):
    dot = sum(a * b for a, b in zip(image_feat, text_feat))
    norm = math.sqrt(dot_self(image_feat) * dot_self(text_feat))
    return dot / max(norm, 1e-8)
```

### 第 3 步：Elo 聚合

```python
def elo_update(r_a, r_b, winner, k=32):
    expected_a = 1 / (1 + 10 ** ((r_b - r_a) / 400))
    actual_a = 1.0 if winner == "a" else 0.0
    r_a_new = r_a + k * (actual_a - expected_a)
    r_b_new = r_b - k * (actual_a - expected_a)
    return r_a_new, r_b_new
```

## 常见陷阱

- **在 N=1000 下计算 FID。** 经验上 N 小于 1 万时不可靠。报告小 N FID 的论文是在刷分。
- **跨分辨率比较 FID。** Inception 的 299×299 缩放会改变特征分布。只能在匹配的分辨率下比较。
- **只报告一个随机种子。** 至少跑 3 个种子，并报告标准差。
- **用负向提示词抬高 CLIP score。** 某些流水线靠对提示词过拟合来刷 CLIP 分数。检查图像是否出现视觉饱和。
- **提示词重叠导致的 Elo 偏差。** 如果两个模型在训练中都见过基准提示词，Elo 就毫无意义。使用留出的提示词集。
- **付费众包人群的偏斜。** Prolific、MTurk 上的标注者偏年轻、偏技术圈。要混入招募来的艺术/设计专家。

## 生产实践

2026 年的生产级评估协议：

| 支柱 | 最低要求 | 推荐做法 |
|--------|---------|-------------|
| 样本质量 | 1 万样本对照留出真实集的 FID | + 5 千样本的 CMMD + 按类别子集分别计算 FID |
| 提示词遵循度 | 3 万样本的 CLIP score | + HPSv2 + ImageReward + VQA 式问答验证 |
| 偏好 | 与基线对比的 200 组盲测对 | + 2000 组人类成对评估 + LLM 评审 + Chatbot Arena |
| 失效分析 | 50 个人工标记样本 | 500 个人工标记样本 + 自动化安全分类器 |

四个支柱齐全的报告 = 结论。任何单独一项 = 营销。

## 交付产物

保存 `outputs/skill-eval-report.md`。该 Skill 接收一个新模型 checkpoint 和一个基线模型，输出完整的评估方案：样本量、指标、失效模式探查项、签收标准。

## 练习

1. **简单。** 运行 `code/main.py`。在同一组合成分布上比较 N=100 与 N=1000 时的 FID，报告偏差的大小。
2. **中等。** 基于合成的 CLIP 式特征实现 CMMD（公式见 Jayasumana 等人，2024）。与 FID 对比两者对质量差异的敏感度。
3. **困难。** 复现 HPSv2 的设置：从 Pick-a-Pic 的一个子集中取 1000 对图像-提示词，在偏好数据上微调一个小型的基于 CLIP 的打分器，并在留出集上衡量它与人类偏好的一致性。

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|-----------------|-----------------------|
| FID | "Fréchet Inception Distance" | 对真实与生成图像的 Inception 特征分别做高斯拟合后的弗雷歇距离。 |
| CLIP score | "图文相似度" | CLIP 图像嵌入与文本嵌入之间的余弦相似度。 |
| CMMD | "FID 的替代者" | 基于 CLIP 特征的 MMD；偏差更小，无高斯假设。 |
| IS | "Inception score" | Exp KL(p(y\|x) \|\| p(y))；在现代模型上相关性很差，已退役。 |
| HPSv2 / ImageReward / PickScore | "习得的偏好代理" | 在人类偏好数据上训练的小模型；用作自动评审。 |
| Elo | "国际象棋等级分" | 对成对胜负的 Bradley-Terry 聚合。 |
| PartiPrompts | "那个基准提示词集" | Google 整理的 1,600 条提示词，覆盖 12 个类别。 |
| FD-DINO | "自监督替代方案" | 使用 DINOv2 特征的 FD；在 ImageNet 之外的领域表现更好。 |

## 生产笔记：评估本身也是一种推理负载

在 1 万个样本上跑 FID，意味着要先生成 1 万张图像。以 50 步的 SDXL base、1024² 分辨率、单张 L4 计算，单请求推理大约要 11 个小时。评估预算是真实存在的，而其框架正是离线推理场景（最大化吞吐量，忽略 TTFT）：

- **狠批量、忘掉延迟。** 离线评估 = 在显存允许的最大批量下做静态批处理。在 80GB H100 上用 `num_images_per_prompt=8` 调用 `pipe(...).images`，墙钟时间比单请求快 4-6 倍。
- **缓存真实集特征。** 对真实参考集的 Inception（FID）或 CLIP（CLIP score、CMMD）特征提取只需运行*一次*，存为 `.npz` 文件。不要每次评估都重算。

对于 CI / 回归门禁：每个 PR 在 500 样本子集上跑 FID + CLIP score（约 30 分钟）；每晚跑完整的 1 万样本 FID + HPSv2 + Elo。

## 延伸阅读

- [Heusel et al. (2017). GANs Trained by a Two Time-Scale Update Rule Converge to a Local Nash Equilibrium (FID)](https://arxiv.org/abs/1706.08500) — FID 论文。
- [Jayasumana et al. (2024). Rethinking FID: Towards a Better Evaluation Metric for Image Generation (CMMD)](https://arxiv.org/abs/2401.09603) — CMMD。
- [Radford et al. (2021). Learning Transferable Visual Models from Natural Language Supervision (CLIP)](https://arxiv.org/abs/2103.00020) — CLIP。
- [Wu et al. (2023). HPSv2: A Comprehensive Human Preference Score](https://arxiv.org/abs/2306.09341) — HPSv2。
- [Xu et al. (2023). ImageReward: Learning and Evaluating Human Preferences for Text-to-Image Generation](https://arxiv.org/abs/2304.05977) — ImageReward。
- [Yu et al. (2023). Scaling Autoregressive Models for Content-Rich Text-to-Image Generation (Parti + PartiPrompts)](https://arxiv.org/abs/2206.10789) — PartiPrompts。
- [Stein et al. (2023). Exposing flaws of generative model evaluation metrics](https://arxiv.org/abs/2306.04675) — 失效模式综述。
