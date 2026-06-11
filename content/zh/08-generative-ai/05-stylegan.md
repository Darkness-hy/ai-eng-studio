# StyleGAN

> 大多数生成器把 `z` 一次性搅进每一层。StyleGAN 把这件事拆开了：先把 `z` 映射成中间表示 `w`，再通过 AdaIN 把 `w` *注入*到每个分辨率层级。就这一处改动，理清了潜空间的纠缠，让照片级人脸生成在此后七年里一直是个已解决的问题。

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 8 · 03 (GANs), Phase 4 · 08 (Normalization), Phase 3 · 07 (CNNs)
**Time:** ~45 minutes

## 问题背景

DCGAN 通过一叠转置卷积把 `z` 映射成图像。问题在于：`z` 控制着一切——姿态、光照、身份、背景——全都纠缠在一起。沿 `z` 的某一个轴移动，这四样东西会同时变化。你没法要求模型"同一个人、换个姿态"，因为这种表示根本不是按这个方式分解的。

Karras 等人（2019，NVIDIA）提出：不要再把 `z` 直接喂给卷积层。改为把一个常量 `4×4×512` 张量作为网络输入。学习一个 8 层 MLP，把 `z ∈ Z → w ∈ W`。通过*自适应实例归一化*（adaptive instance normalization，AdaIN）在每个分辨率上注入 `w`：先对每个卷积特征图做归一化，再用 `w` 的仿射投影做缩放和平移。另外在每层加入噪声，提供随机细节（毛孔、发丝）。

结果是：`W` 拥有大致正交的轴，分别对应"高层风格"（姿态、身份）和"精细风格"（光照、颜色）。你可以在两张图像之间交换风格：低分辨率层级用图像 A 的 `w`，高分辨率层级用图像 B 的 `w`。这开启了图像编辑、跨域风格化，以及整条"StyleGAN 反演（inversion）"研究路线。

## 核心概念

![StyleGAN: mapping network + AdaIN + per-layer noise](../assets/stylegan.svg)

**映射网络（mapping network）。** `f: Z → W`，一个 8 层 MLP。`Z = N(0, I)^512`。`W` 不被强制服从高斯分布——它会学出一个适配数据的形状。

**合成网络（synthesis network）。** 从一个可学习的常量 `4×4×512` 开始。每个分辨率块为：`upsample → conv → AdaIN(w_i) → noise → conv → AdaIN(w_i) → noise`。分辨率逐级翻倍：4、8、16、32、64、128、256、512、1024。

**AdaIN。**

```
AdaIN(x, y) = y_scale · (x - mean(x)) / std(x) + y_bias
```

其中 `y_scale` 和 `y_bias` 来自 `w` 的仿射投影。先逐特征图归一化，再重新赋予风格。这里的"风格"指的就是特征图的一阶和二阶统计量。

**逐层噪声（per-layer noise）。** 在每个特征图上加单通道高斯噪声，并按一个可学习的逐通道系数缩放。它控制随机细节，而不影响全局结构。

**截断技巧（truncation trick）。** 推理时，采样 `z`，计算 `w = mapping(z)`，再取 `w' = ŵ + ψ·(w - ŵ)`，其中 `ŵ` 是大量样本上的 `w` 均值。`ψ < 1` 以多样性换质量。几乎所有 StyleGAN 演示都用 `ψ ≈ 0.7`。

## StyleGAN 1 → 2 → 3

| 版本 | 年份 | 创新点 |
|---------|------|------------|
| StyleGAN | 2019 | 映射网络 + AdaIN + 噪声 + 渐进式增长（progressive growing）。 |
| StyleGAN2 | 2020 | 用权重解调（weight demodulation）取代 AdaIN（修复水滴伪影）；skip/残差架构；路径长度正则化。 |
| StyleGAN3 | 2021 | 无混叠卷积（alias-free convolution）+ 等变卷积核；消除纹理粘连在像素网格上的问题。 |
| StyleGAN-XL | 2022 | 类别条件生成，1024²，ImageNet。 |
| R3GAN | 2024 | 以更强的正则化重新登场；在 FFHQ-1024 上以少 20 倍的参数追平扩散模型。 |

到 2026 年，StyleGAN3 仍是以下场景的默认选择：(a) 窄域、高帧率的照片级真实感生成；(b) 少样本域适应（在 100 张图的新数据集上训练，冻结映射网络）；(c) 基于反演的编辑（先找到能重建真实照片的 `w`，再编辑这个 `w`）。至于开放域的文本生成图像，它不是合适的工具——该用扩散模型。

## 从零实现

`code/main.py` 在一维上实现了一个玩具版"style-GAN lite"：一个映射 MLP、一个合成函数（接收可学习的常量向量，并用由 `w` 推导的 scale/bias 对其调制），以及逐层噪声。它展示了通过仿射调制注入 `w`，效果不输甚至优于把 `z` 拼接进生成器输入。

### Step 1: mapping network

```python
def mapping(z, M):
    h = z
    for i in range(num_layers):
        h = leaky_relu(add(matmul(M[f"W{i}"], h), M[f"b{i}"]))
    return h
```

### Step 2: adaptive instance normalization

```python
def adain(x, w_scale, w_bias):
    mu = mean(x)
    sd = std(x)
    x_norm = [(xi - mu) / (sd + 1e-8) for xi in x]
    return [w_scale * xi + w_bias for xi in x_norm]
```

逐特征图的 scale 和 bias 由 `w` 经线性投影得到。

### Step 3: per-layer noise

```python
def add_noise(x, sigma, rng):
    return [xi + sigma * rng.gauss(0, 1) for xi in x]
```

每个通道的 sigma 是可学习的。

## 常见陷阱

- **水滴伪影（droplet artifacts）。** StyleGAN 1 会在特征图里产生一团水滴状斑块，因为 AdaIN 把均值归零了。StyleGAN 2 的权重解调通过改为缩放卷积权重修复了这个问题。
- **纹理粘连（texture sticking）。** StyleGAN 1 和 2 的纹理跟随像素坐标而非物体坐标（在插值时肉眼可见）。StyleGAN 3 用加窗 sinc 滤波器实现的无混叠卷积修复了它。
- **模式覆盖。** 截断 `ψ < 0.7` 看起来干净，但只从一个很窄的锥体里采样；如果需要多样性，请用 `ψ = 1.0`。
- **反演是有损的。** 把真实照片反演到 `W` 通常依靠优化或编码器（e4e、ReStyle、HyperStyle）完成。迭代次数一多，结果会逐渐漂移。

## 生产实践

| 使用场景 | 方案 |
|----------|----------|
| 照片级人脸（动漫、产品、窄域） | StyleGAN3 FFHQ / 自定义微调 |
| 基于照片的人脸编辑 | e4e 反演 + StyleSpace / InterFaceGAN 编辑方向 |
| 换脸 / 表情重演 | StyleGAN + 编码器 + 混合 |
| 头像（avatar）流水线 | StyleGAN3 配合 ADA 做低数据量微调 |
| 用少量图像做域适应 | 冻结映射网络，微调合成网络 |
| 多模态或文本条件生成 | 别用它——用扩散模型 |

对于答案是"一张人脸照片"的产品级演示，在同等质量标准下，StyleGAN 在推理成本（单次前向传播，4090 上 <10ms）和清晰度上都胜过扩散模型。

## 交付产物

保存 `outputs/skill-stylegan-inversion.md`。该 skill 接收一张真实照片，输出：反演方法（e4e / ReStyle / HyperStyle）、预期潜变量损失、编辑预算（在 `W` 中能移动多远才会出现伪影），以及一份已知可靠的编辑方向清单（年龄、表情、姿态）。

## 练习

1. **简单。** 分别用 `adain_on=True` 和 `adain_on=False` 运行 `code/main.py`。比较固定潜变量与扰动潜变量下输出的离散程度。
2. **中等。** 实现混合正则化（mixing regularization）：对一个训练批次，计算 `w_a`、`w_b`，在合成的前半段应用 `w_a`，后半段应用 `w_b`。解码器学到解耦的风格了吗？
3. **困难。** 取一个预训练的 StyleGAN3 FFHQ 模型（ffhq-1024.pkl）。在带标签的样本上训练 SVM，找到控制"微笑"的 `w` 方向；报告在身份发生漂移之前能把它推多远。

## 关键术语

| 术语 | 大家怎么说 | 实际含义 |
|------|-----------------|-----------------------|
| 映射网络 | "那个 MLP" | `f: Z → W`，8 层，把潜空间几何与数据统计解耦。 |
| W 空间 | "风格空间" | 映射网络的输出；大致解耦。 |
| AdaIN | "自适应实例归一化" | 先归一化特征图，再用 `w` 的投影做缩放和平移。 |
| 截断技巧 | "Psi" | `w = mean + ψ·(w - mean)`，ψ<1 以多样性换质量。 |
| 路径长度正则化 | "PL reg" | 惩罚 `w` 单位变化引起的图像剧烈变化；让 `W` 更平滑。 |
| 权重解调 | "StyleGAN2 的修复" | 改为归一化卷积权重而非激活值；消灭水滴伪影。 |
| 无混叠（alias-free） | "StyleGAN3 的招数" | 加窗 sinc 滤波器；消除纹理粘连在像素网格上的问题。 |
| 反演（inversion） | "给真实图像找 w" | 通过优化或编码实现 `x → w`，使 `G(w) ≈ x`。 |

## 生产笔记：为什么 StyleGAN 在 2026 年仍在线上跑

StyleGAN3 在 4090 上生成一张 1024² 的 FFHQ 人脸不到 10 毫秒——`num_steps = 1`，没有 VAE 解码，没有交叉注意力计算。从生产角度看，这是任何图像生成器的延迟下限。同分辨率下，50 步的 SDXL + VAE 解码流水线约需 3 秒。这是 **300 倍的差距**，对窄域产品（头像服务、证件照流水线、素材人脸生成）而言，它在总拥有成本（TCO）上赢了。

两个运维上的推论：

- **不需要调度器，也不需要批处理器。** 在目标占用率下使用静态批处理就是最优解。连续批处理（对 LLM 和扩散模型必不可少）在这里毫无收益，因为每个请求消耗的 FLOPs 完全相同。
- **截断 `ψ` 是安全旋钮。** `ψ < 0.7` 只从映射网络值域中一个很窄的锥体里采样。这是服务层对样本方差唯一可用的杠杆。高峰负载时调低 `ψ`，对付费用户调高它。

## 延伸阅读

- [Karras et al. (2019). A Style-Based Generator Architecture for GANs](https://arxiv.org/abs/1812.04948) — StyleGAN。
- [Karras et al. (2020). Analyzing and Improving the Image Quality of StyleGAN](https://arxiv.org/abs/1912.04958) — StyleGAN2。
- [Karras et al. (2021). Alias-Free Generative Adversarial Networks](https://arxiv.org/abs/2106.12423) — StyleGAN3。
- [Tov et al. (2021). Designing an Encoder for StyleGAN Image Manipulation](https://arxiv.org/abs/2102.02766) — e4e 反演。
- [Sauer et al. (2022). StyleGAN-XL: Scaling StyleGAN to Large Diverse Datasets](https://arxiv.org/abs/2202.00273) — StyleGAN-XL。
- [Huang et al. (2024). R3GAN: The GAN is dead; long live the GAN!](https://arxiv.org/abs/2501.05441) — 现代极简 GAN 配方。
