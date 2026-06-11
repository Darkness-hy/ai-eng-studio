# 3D 生成

> 3D 是「2D 能力撬动 3D」杠杆效应最强的模态。2023 年的突破是 3D 高斯泼溅（3D Gaussian Splatting）。2024-2026 年的生成式浪潮则在其之上叠加了多视角扩散 + 3D 重建，从一句提示词或一张照片直接生成物体和场景。

**Type:** Learn
**Languages:** Python
**Prerequisites:** Phase 4 (Vision), Phase 8 · 07 (Latent Diffusion)
**Time:** ~45 minutes

## 问题背景

3D 内容很棘手：

- **表示方式。** 网格（mesh）、点云、体素网格、有向距离场（SDF）、神经辐射场（NeRF）、3D 高斯。每种都有取舍。
- **数据稀缺。** ImageNet 有 1400 万张图像。而最大的清洗后 3D 数据集（Objaverse-XL，2023）只有约 1000 万个物体，且大多质量不高。
- **显存。** 一个 512³ 的体素网格有 1.28 亿个体素；一个可用的场景 NeRF 每条光线需要采样 100 万次。生成比重建更难。
- **监督信号。** 2D 图像有现成的像素可用。3D 通常只有寥寥几张 2D 视图，必须自己把它们「抬升」到 3D。

2026 年的技术栈把这两个问题拆开处理。第一步，用扩散模型生成*多视角 2D 图像*。第二步，把一个 *3D 表示*（通常是高斯泼溅）拟合到这些图像上。

## 核心概念

![3D generation: multi-view diffusion + 3D reconstruction](../assets/3d-generation.svg)

### 表示方式：3D 高斯泼溅（Kerbl et al., 2023）

把场景表示为约 100 万个 3D 高斯组成的点云。每个高斯有 59 个参数：位置（3）、协方差（6，或四元数 4 + 缩放 3）、不透明度（1）、球谐函数颜色（3 阶为 48 个，0 阶为 3 个）。

渲染 = 投影 + alpha 合成。速度快（在 4090 上 1080p 约 100 fps）。可微分。通过梯度下降对照真实照片拟合。一个场景在消费级 GPU 上 5-30 分钟就能拟合完成。

在此之上有两项 2023-2024 年的创新：

- **生成式高斯泼溅。** LGM、LRM、InstantMesh 等模型可以从一张或几张图像直接预测出高斯点云。
- **4D 高斯泼溅。** 为高斯加上逐帧偏移量，用于动态场景。

### 多视角扩散

对预训练的图像扩散模型做微调，让它根据文本提示词或单张图像生成同一物体多个一致的视角。代表工作有 Zero123（Liu et al., 2023）、MVDream（Shi et al., 2023）、SV3D（Stability, 2024）、CAT3D（Google, 2024）。通常输出物体周围的 4-16 个视角，再通过高斯泼溅或 NeRF 抬升到 3D。

### 文本到 3D 流水线

| 模型 | 输入 | 输出 | 耗时 |
|-------|-------|--------|------|
| DreamFusion (2022) | 文本 | 经 SDS 得到 NeRF | 每个资产约 1 小时 |
| Magic3D | 文本 | 网格 + 纹理 | 约 40 分钟 |
| Shap-E (OpenAI, 2023) | 文本 | 隐式 3D | 约 1 分钟 |
| SJC / ProlificDreamer | 文本 | NeRF / 网格 | 约 30 分钟 |
| LRM (Meta, 2023) | 图像 | 三平面（triplane） | 约 5 秒 |
| InstantMesh (2024) | 图像 | 网格 | 约 10 秒 |
| SV3D (Stability, 2024) | 图像 | 新视角图像 | 约 2 分钟 |
| CAT3D (Google, 2024) | 1-64 张图像 | 3D NeRF | 约 1 分钟 |
| TripoSR (2024) | 图像 | 网格 | 约 1 秒 |
| Meshy 4 (2025) | 文本 + 图像 | PBR 网格 | 约 30 秒 |
| Rodin Gen-1.5 (2025) | 文本 + 图像 | PBR 网格 | 约 60 秒 |
| Tencent Hunyuan3D 2.0 (2025) | 图像 | 网格 | 约 30 秒 |

2025-2026 年的方向：直接文本到网格的模型，输出带 PBR 材质、可直接用于游戏引擎的资产。但对于通用物体，多视角扩散作为中间步骤仍是效果最好的方案。

### NeRF（背景知识）

神经辐射场（Neural Radiance Field，Mildenhall et al., 2020）。一个小型 MLP 接收 `(x, y, z, view direction)`，输出 `(color, density)`。沿光线积分来渲染。在新视角合成的质量上胜过基于网格的方法，但渲染速度慢 100-1000 倍。在大多数实时场景中已被高斯泼溅取代，但在研究领域仍占主导地位。

## 从零实现

`code/main.py` 实现了一个玩具版的 2D「高斯泼溅」拟合：把一张合成目标图像（一个平滑渐变）表示为若干 2D 高斯斑点之和。通过梯度下降优化位置、颜色和协方差来匹配目标。你会看到两个核心操作：前向渲染（泼溅 + alpha 合成）和梯度下降拟合。

### 第 1 步：2D 高斯斑点

```python
def gaussian_at(x, y, gaussian):
    px, py = gaussian["pos"]
    sigma = gaussian["sigma"]
    d2 = (x - px) ** 2 + (y - py) ** 2
    return math.exp(-d2 / (2 * sigma * sigma))
```

### 第 2 步：累加斑点完成渲染

```python
def render(image_size, gaussians):
    img = [[0.0] * image_size for _ in range(image_size)]
    for g in gaussians:
        for y in range(image_size):
            for x in range(image_size):
                img[y][x] += g["color"] * gaussian_at(x, y, g)
    return img
```

真实的 3D 高斯泼溅会按深度对高斯排序，再依次做 alpha 合成。我们的 2D 玩具版只做简单累加。

### 第 3 步：梯度下降拟合

```python
for step in range(steps):
    pred = render(size, gaussians)
    loss = mse(pred, target)
    gradients = compute_grads(pred, target, gaussians)
    update(gaussians, gradients, lr)
```

## 常见陷阱

- **视角不一致。** 如果独立生成 4 个视角，而它们对物体结构的描述互相矛盾，3D 拟合结果就会发糊。解决办法：使用带共享注意力的多视角扩散。
- **背面幻觉。** 单图到 3D 必须凭空「编造」看不见的那一面，质量波动极大。
- **高斯数量爆炸。** 不加约束的训练会膨胀到 1000 万个高斯并过拟合。致密化（densification）+ 剪枝启发式策略（来自 3D-GS 原始论文）必不可少。
- **拓扑问题。** 从隐式场（SDF）提取的网格经常有破洞或自相交。交付前先跑一遍重网格化工具（如 blender 的 voxel remesh）。
- **训练数据许可。** Objaverse 的许可证混杂，能否商用因模型而异。

## 生产实践

| 任务 | 2026 年首选 |
|------|-----------|
| 从照片重建场景 | 高斯泼溅（3DGS、Gsplat、Scaniverse） |
| 游戏用文本到 3D 物体 | Meshy 4 或 Rodin Gen-1.5（PBR 输出） |
| 图像到 3D | Hunyuan3D 2.0、TripoSR、InstantMesh |
| 少量图像的新视角合成 | CAT3D、SV3D |
| 动态场景重建 | 4D 高斯泼溅 |
| 数字人 / 带服装的人体 | Gaussian Avatar、HUGS |
| 研究 / SOTA | 上周刚发布的最新成果 |

如果要在游戏或电商流水线中交付生产级 3D：Meshy 4 或 Rodin Gen-1.5 输出的 PBR 网格可以直接导入 Unity / Unreal。

## 交付产物

保存 `outputs/skill-3d-pipeline.md`。该技能接收一份 3D 需求简报（输入：文本 / 单张图像 / 少量图像；输出：网格 / 高斯 / NeRF；用途：渲染 / 游戏 / VR），并输出：流水线方案（多视角扩散 + 拟合，或直接网格模型）、基础模型、迭代预算、拓扑后处理、所需材质通道。

## 练习

1. **简单。** 用 4、16、64 个高斯分别运行 `code/main.py`，汇报与目标图像的最终 MSE。
2. **中等。** 扩展为彩色高斯（RGB）。确认重建结果与目标的颜色图案一致。
3. **困难。** 使用 gsplat 或 Nerfstudio，从 50 张照片采集中重建一个真实物体。汇报拟合时间，以及在留出视角上的最终 SSIM。

## 关键术语

| 术语 | 大家怎么说 | 实际含义 |
|------|-----------------|-----------------------|
| 3D 高斯泼溅 | "3DGS" | 把场景表示为 3D 高斯点云；可微的 alpha 合成渲染。 |
| NeRF | 「神经辐射场」 | 在 3D 空间点上输出颜色 + 密度的 MLP；沿光线积分来渲染。 |
| 三平面（Triplane） | 「三个 2D 平面」 | 把 3D 分解为三个轴对齐的 2D 特征网格；比体积表示更省。 |
| SDS | 「分数蒸馏采样」 | 用 2D 扩散模型的分数作为伪梯度来训练 3D 模型。 |
| 多视角扩散 | 「一次出多个视角」 | 能输出一批相机视角且彼此一致的扩散模型。 |
| PBR | 「基于物理的渲染」 | 包含反照率、粗糙度、金属度、法线通道的材质。 |
| 致密化（Densification） | 「增殖高斯」 | 3DGS 训练启发式策略：在高梯度区域分裂 / 克隆高斯。 |

## 生产笔记：3D 尚无统一基座

不同于图像（潜在扩散 + DiT）和视频（时空 DiT），到 2026 年 3D 仍没有单一占主导的运行时。生产决策树在表示方式上分叉：

- **NeRF / 三平面。** 推理是光线步进 + 每个采样点一次 MLP 前向。渲染一张 512² 图像需要数百万次 MLP 前向。要对光线采样点做激进的批处理；SDPA/xformers 同样适用。
- **多视角扩散 + LRM 重建。** 两阶段流水线。第一阶段（多视角 DiT）是一个扩散服务，和第 07 课完全一样。第二阶段（LRM Transformer）是对各视图做一次性前向。整体延迟特征是「扩散 + 一次前向」——按阶段分别选取服务原语。
- **SDS / DreamFusion。** 是逐资产优化，而不是推理。应该构建为批处理作业，而非请求处理器。

对 2026 年的大多数产品来说，正确答案是「按请求运行多视角扩散模型，异步重建为 3DGS，再用 3DGS 提供实时浏览」。这把工作负载干净地拆分为 GPU 推理服务器（快）和离线优化器（慢）。

## 延伸阅读

- [Mildenhall et al. (2020). NeRF: Representing Scenes as Neural Radiance Fields](https://arxiv.org/abs/2003.08934) — NeRF。
- [Kerbl et al. (2023). 3D Gaussian Splatting for Real-Time Radiance Field Rendering](https://arxiv.org/abs/2308.04079) — 3DGS。
- [Poole et al. (2022). DreamFusion: Text-to-3D using 2D Diffusion](https://arxiv.org/abs/2209.14988) — SDS。
- [Liu et al. (2023). Zero-1-to-3: Zero-shot One Image to 3D Object](https://arxiv.org/abs/2303.11328) — Zero123。
- [Shi et al. (2023). MVDream](https://arxiv.org/abs/2308.16512) — 多视角扩散。
- [Hong et al. (2023). LRM: Large Reconstruction Model for Single Image to 3D](https://arxiv.org/abs/2311.04400) — LRM。
- [Gao et al. (2024). CAT3D: Create Anything in 3D with Multi-View Diffusion Models](https://arxiv.org/abs/2405.10314) — CAT3D。
- [Stability AI (2024). Stable Video 3D (SV3D)](https://stability.ai/research/sv3d) — SV3D。
