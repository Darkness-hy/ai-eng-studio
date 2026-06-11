# 世界模型与视频扩散

> 一个能预测场景未来几秒画面的视频模型，本质上就是一个世界模拟器。再让这种预测以动作为条件，你就得到了一个学习出来的游戏引擎。

**Type:** Learn + Build
**Languages:** Python
**Prerequisites:** Phase 4 Lesson 10 (Diffusion), Phase 4 Lesson 12 (Video Understanding), Phase 4 Lesson 23 (DiT + Rectified Flow)
**Time:** ~75 minutes

## 学习目标

- 说明纯视频生成模型（Sora 2）与动作条件世界模型（Genie 3、DreamerV3）之间的区别
- 描述视频 DiT 的结构：时空 patch、3D 位置编码、跨 (T, H, W) token 的联合注意力
- 梳理世界模型如何接入机器人系统：VLM 做规划 → 视频模型做仿真 → 逆动力学模型输出动作
- 针对给定场景（创意视频、交互式仿真、自动驾驶数据合成），在 Sora 2、Genie 3、Runway GWM-1 Worlds、Wan-Video 和 HunyuanVideo 之间做出选型

## 问题背景

视频生成与世界建模在 2026 年走向了融合。一个能生成连贯一分钟视频的模型，某种意义上已经学会了世界如何运转：物体恒存性、重力、因果关系、风格。如果让这种预测以动作（向左走、开门）为条件，视频模型就变成了一个可学习的模拟器，可以替代游戏引擎、驾驶模拟器或机器人环境。

这件事的意义非常具体。Genie 3 能从一张图片生成可游玩的环境。Runway GWM-1 Worlds 能合成可无限探索的场景。Sora 2 能生成带同步音频、建模了物理规律的分钟级视频。NVIDIA Cosmos-Drive、Wayve Gaia-2 和 Tesla DrivingWorld 为自动驾驶训练数据生成逼真的驾驶视频。世界模型范式正在悄然接管机器人领域的 sim-to-real。

这节课是 Phase 4 的"全局视野"课。它把图像生成、视频理解和智能体推理串联成一个架构模式——主流研究正在朝这个方向汇聚。

## 核心概念

### 世界建模的三个家族

```mermaid
flowchart LR
    subgraph GEN["Pure video generation"]
        G1["Text / image prompt"] --> G2["Video DiT"] --> G3["Video frames"]
    end
    subgraph ACTION["Action-conditioned world model"]
        A1["Past frames + action"] --> A2["Latent-action video DiT"] --> A3["Next frames"]
        A3 --> A1
    end
    subgraph RL["World models for RL (DreamerV3)"]
        R1["State + action"] --> R2["Latent transition model"] --> R3["Next latent + reward"]
        R3 --> R1
    end

    style GEN fill:#dbeafe,stroke:#2563eb
    style ACTION fill:#fef3c7,stroke:#d97706
    style RL fill:#dcfce7,stroke:#16a34a
```

- **Sora 2** 是纯视频生成，以提示词为条件。没有动作接口。你无法在生成过程中"操控"它。
- **Genie 3**、**GWM-1 Worlds**、**Mirage / Magica** 是动作条件世界模型。先从观测到的视频中推断潜在动作（latent action），再让未来帧的预测以动作为条件。它们是交互式的——你按下按键或移动镜头，场景会做出响应。
- **DreamerV3** 及经典的 RL 世界模型家族在潜空间中做预测，带显式动作条件，由奖励信号训练。视觉效果较弱，但对高样本效率的强化学习更有用。

### 视频 DiT 架构

```
Video latent:          (C, T, H, W)
Patchify (spatial):    grid of P_h x P_w patches per frame
Patchify (temporal):   group P_t frames into a temporal patch
Resulting tokens:      (T / P_t) * (H / P_h) * (W / P_w) tokens
```

位置编码是 3D 的：每个 (t, h, w) 坐标对应一个旋转式或可学习的嵌入。注意力有几种形式：

- **完全联合（Full joint）**——所有 token 互相注意。N 个 token 时复杂度为 O(N^2)，对长视频来说代价过高。
- **分离式（Divided）**——交替执行时间注意力（同一空间位置、跨时间：`(H*W) * T^2`）和空间注意力（同一时间步、跨空间：`T * (H*W)^2`）。TimeSformer 和大多数视频 DiT 采用这种方式。
- **窗口式（Window）**——在 (t, h, w) 上使用局部窗口。Video Swin 采用这种方式。

2026 年的每一个视频扩散模型都使用这三种模式之一，再加上 AdaLN 条件机制（第 23 课）和整流流（rectified flow）。

### 以动作为条件：潜在动作模型

Genie 通过判别式地预测一对连续帧之间发生的动作，为每一帧学习一个**潜在动作（latent action）**。模型的解码器随后以推断出的潜在动作为条件——而不是以显式的键盘按键为条件。推理时，用户可以指定一个潜在动作（或从一个新的先验中采样一个），模型生成与该动作一致的下一帧。

Sora 完全跳过了动作接口。它的解码器根据过去的时空 token 预测下一批时空 token。提示词只决定开头；生成过程中没有任何东西可以操控它。

### 物理合理性

Sora 2 在 2026 年发布时明确宣传了**物理合理性（physical plausibility）**：重量、平衡、物体恒存性、因果关系。团队通过人工评分的合理性指标来衡量；相比 Sora 1，模型在掉落物体、角色碰撞以及"故意失败"（比如跳跃失误）的场景上有了肉眼可见的提升。

合理性仍然是主要的失败模式。2024-2025 年那些人吃意大利面、用玻璃杯喝水的视频暴露了模型缺乏持久的物体表示。2026 年的模型（Sora 2、Runway Gen-5、HunyuanVideo）减少了但没有消除这些问题。

### 自动驾驶世界模型

驾驶世界模型以轨迹、边界框或导航地图为条件，生成逼真的道路场景。用途如下：

- **Cosmos-Drive-Dreams**（NVIDIA）——为 RL 训练生成分钟级驾驶视频。
- **Gaia-2**（Wayve）——以轨迹为条件的场景合成，用于策略评估。
- **DrivingWorld**（Tesla）——模拟多样的天气、时段和交通状况。
- **Vista**（ByteDance）——可响应的驾驶场景合成。

它们替代了针对长尾场景的昂贵真实数据采集——夜间行人横穿马路、结冰的路口、罕见车型——这些场景如果靠实际驾驶采集，需要数百万英里的里程。

### 机器人技术栈：VLM + 视频模型 + 逆动力学

正在兴起的三组件机器人闭环：

1. **VLM** 解析目标（"拿起红色杯子"），规划高层动作序列。
2. **视频生成模型** 模拟执行每个动作会是什么样子——预测 N 帧之后的观测。
3. **逆动力学模型（inverse dynamics model）** 提取能产生这些观测的具体电机指令。

这套结构取代了奖励塑形和样本密集的 RL。世界模型负责想象；逆动力学模型在执行端闭合回路。Genie Envisioner 是其中一种实现；许多研究团队正在朝这个结构汇聚。

### 评估

- **视觉质量**——FVD（Fréchet Video Distance）、用户研究。
- **提示词对齐**——逐帧 CLIPScore、VQA 式评估。
- **物理合理性**——在基准套件上人工评分（Sora 2 的内部基准、VBench）。
- **可控性**（针对交互式世界模型）——动作 → 观测的一致性；能否回到之前的状态？

### 2026 年模型版图

| 模型 | 用途 | 参数量 | 输出 | 许可 |
|-------|-----|------------|--------|---------|
| Sora 2 | 文生视频、音频 | — | 1 分钟 1080p + 音频 | 仅 API |
| Runway Gen-5 | 文/图生视频 | — | 10 秒片段 | API |
| Runway GWM-1 Worlds | 交互式世界 | — | 无限 3D 推演 | API |
| Genie 3 | 从图像生成交互式世界 | 11B+ | 可游玩的帧序列 | 研究预览 |
| Wan-Video 2.1 | 开源文生视频 | 14B | 高质量片段 | 非商用 |
| HunyuanVideo | 开源文生视频 | 13B | 10 秒片段 | 宽松许可 |
| Cosmos / Cosmos-Drive | 自动驾驶仿真 | 7-14B | 驾驶场景 | NVIDIA 开源 |
| Magica / Mirage 2 | AI 原生游戏引擎 | — | 可修改的世界 | 产品 |

## 从零实现

### 第 1 步：视频的 3D patchify

```python
import torch
import torch.nn as nn


class VideoPatch3D(nn.Module):
    def __init__(self, in_channels=4, dim=64, patch_t=2, patch_h=2, patch_w=2):
        super().__init__()
        self.proj = nn.Conv3d(
            in_channels, dim,
            kernel_size=(patch_t, patch_h, patch_w),
            stride=(patch_t, patch_h, patch_w),
        )
        self.patch_t = patch_t
        self.patch_h = patch_h
        self.patch_w = patch_w

    def forward(self, x):
        # x: (N, C, T, H, W)
        x = self.proj(x)
        n, c, t, h, w = x.shape
        tokens = x.reshape(n, c, t * h * w).transpose(1, 2)
        return tokens, (t, h, w)
```

一个步长等于卷积核大小的 3D 卷积，就是时空 patchifier。`(T, H, W) -> (T/2, H/2, W/2)` 的 token 网格。

### 第 2 步：3D 旋转位置编码

旋转位置嵌入（RoPE）沿 `t`、`h`、`w` 三个轴分别施加：

```python
def rope_3d(tokens, t_dim, h_dim, w_dim, grid):
    """
    tokens: (N, T*H*W, D)
    grid: (T, H, W) sizes
    t_dim + h_dim + w_dim == D
    """
    T, H, W = grid
    n, seq, d = tokens.shape
    if t_dim + h_dim + w_dim != d:
        raise ValueError(f"t_dim+h_dim+w_dim ({t_dim}+{h_dim}+{w_dim}) must equal D={d}")
    assert seq == T * H * W
    t_idx = torch.arange(T, device=tokens.device).repeat_interleave(H * W)
    h_idx = torch.arange(H, device=tokens.device).repeat_interleave(W).repeat(T)
    w_idx = torch.arange(W, device=tokens.device).repeat(T * H)
    # Simplified: just scale channels by frequencies. Real RoPE rotates pairs.
    freqs_t = torch.exp(-torch.log(torch.tensor(10000.0)) * torch.arange(t_dim // 2, device=tokens.device) / (t_dim // 2))
    freqs_h = torch.exp(-torch.log(torch.tensor(10000.0)) * torch.arange(h_dim // 2, device=tokens.device) / (h_dim // 2))
    freqs_w = torch.exp(-torch.log(torch.tensor(10000.0)) * torch.arange(w_dim // 2, device=tokens.device) / (w_dim // 2))
    emb_t = torch.cat([torch.sin(t_idx[:, None] * freqs_t), torch.cos(t_idx[:, None] * freqs_t)], dim=-1)
    emb_h = torch.cat([torch.sin(h_idx[:, None] * freqs_h), torch.cos(h_idx[:, None] * freqs_h)], dim=-1)
    emb_w = torch.cat([torch.sin(w_idx[:, None] * freqs_w), torch.cos(w_idx[:, None] * freqs_w)], dim=-1)
    return tokens + torch.cat([emb_t, emb_h, emb_w], dim=-1)
```

这是简化的加性形式。真正的 RoPE 按频率旋转成对的通道；但携带的位置信息是一样的。

### 第 3 步：分离式注意力块

```python
class DividedAttentionBlock(nn.Module):
    def __init__(self, dim=64, heads=2):
        super().__init__()
        self.time_attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.space_attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.ln1 = nn.LayerNorm(dim)
        self.ln2 = nn.LayerNorm(dim)
        self.ln3 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(nn.Linear(dim, 4 * dim), nn.GELU(), nn.Linear(4 * dim, dim))

    def forward(self, x, grid):
        T, H, W = grid
        n, seq, d = x.shape
        # time attention: same (h, w), across t
        xt = x.view(n, T, H * W, d).permute(0, 2, 1, 3).reshape(n * H * W, T, d)
        a, _ = self.time_attn(self.ln1(xt), self.ln1(xt), self.ln1(xt), need_weights=False)
        xt = (xt + a).reshape(n, H * W, T, d).permute(0, 2, 1, 3).reshape(n, seq, d)
        # space attention: same t, across (h, w)
        xs = xt.view(n, T, H * W, d).reshape(n * T, H * W, d)
        a, _ = self.space_attn(self.ln2(xs), self.ln2(xs), self.ln2(xs), need_weights=False)
        xs = (xs + a).reshape(n, T, H * W, d).reshape(n, seq, d)
        xs = xs + self.mlp(self.ln3(xs))
        return xs
```

时间注意力在每个空间位置内沿时间方向计算注意力；空间注意力在每一帧内沿空间位置计算注意力。两次 O(T^2 + (HW)^2) 操作，取代一次 O((THW)^2) 操作。这是 TimeSformer 以及所有现代视频 DiT 的核心。

### 第 4 步：组装一个微型视频 DiT

```python
class TinyVideoDiT(nn.Module):
    def __init__(self, in_channels=4, dim=64, depth=2, heads=2):
        super().__init__()
        self.patch = VideoPatch3D(in_channels=in_channels, dim=dim, patch_t=2, patch_h=2, patch_w=2)
        self.blocks = nn.ModuleList([DividedAttentionBlock(dim, heads) for _ in range(depth)])
        self.out = nn.Linear(dim, in_channels * 2 * 2 * 2)

    def forward(self, x):
        tokens, grid = self.patch(x)
        for blk in self.blocks:
            tokens = blk(tokens, grid)
        return self.out(tokens), grid
```

这不是一个可用的视频生成器，而是一个结构演示，用来验证每个组件的形状都正确。

### 第 5 步：检查形状

```python
vid = torch.randn(1, 4, 8, 16, 16)  # (N, C, T, H, W)
model = TinyVideoDiT()
out, grid = model(vid)
print(f"input  {tuple(vid.shape)}")
print(f"tokens grid {grid}")
print(f"output {tuple(out.shape)}")
```

patch 之后预期 `grid = (4, 8, 8)`、`out = (1, 256, 32)`；输出头随后投影为逐 token 的时空 patch，可以反 patchify 回视频。

## 生产实践

2026 年的生产环境接入方式：

- **Sora 2 API**（OpenAI）——文生视频、同步音频。高端定价。
- **Runway Gen-5 / GWM-1**（Runway）——图生视频、交互式世界。
- **Wan-Video 2.1 / HunyuanVideo**——开源自托管。
- **Cosmos / Cosmos-Drive**（NVIDIA）——驾驶仿真开放权重。
- **Genie 3**——研究预览，需申请访问。

如果要搭建交互式世界模型演示：从 Wan-Video 入手保证质量，再叠加一个潜在动作适配器实现交互性。如果做自动驾驶仿真：Cosmos-Drive 是 2026 年的开源参考实现。

机器人领域实际在用的技术栈：

1. 语言目标 -> VLM（Qwen3-VL）-> 高层规划。
2. 规划 -> 潜在动作视频模型 -> 想象中的推演。
3. 推演 -> 逆动力学模型 -> 底层动作。
4. 动作执行 -> 观测回流到第 1 步。

## 交付产物

这节课产出：

- `outputs/prompt-video-model-picker.md`——根据任务、许可和延迟，在 Sora 2 / Runway / Wan / HunyuanVideo / Cosmos 之间做选型。
- `outputs/skill-physical-plausibility-checks.md`——一个 skill，定义自动化检查项（物体恒存性、重力、连续性），在任何生成视频上线前运行。

## 练习

1. **（简单）** 计算一段 5 秒 360p 视频在 patch-t=2、patch-h=8、patch-w=8 下的 token 数量。推算这一规模下注意力的显存需求。
2. **（中等）** 把上面的分离式注意力块换成完全联合注意力块，测量形状和参数量。解释为什么真实视频模型必须使用分离式注意力。
3. **（困难）** 构建一个最小的潜在动作视频模型：取一个 (frame_t, action_t, frame_{t+1}) 三元组数据集（任何简单的 2D 游戏都行），训练一个以动作嵌入为条件的微型视频 DiT，并证明不同动作会产生不同的下一帧。

## 关键术语

| 术语 | 大家怎么说 | 实际含义 |
|------|----------------|----------------------|
| 世界模型 | "学习出来的模拟器" | 给定状态和动作，预测未来观测的模型 |
| 视频 DiT | "时空 Transformer" | 带 3D patchify 和分离式注意力的扩散 Transformer |
| 潜在动作 | "推断出的控制信号" | 从帧对中推断出的离散或连续动作潜变量；用于条件化下一帧生成 |
| 分离式注意力 | "先时间后空间" | 每个块做两次注意力——先跨时间再跨空间——以控制 O(N^2) 的开销 |
| 物体恒存性 | "东西不能凭空消失" | 视频模型必须学会的场景属性；在食物、玻璃器皿上是经典失败模式 |
| FVD | "Fréchet Video Distance" | FID 的视频版本；首要的视觉质量指标 |
| 逆动力学模型 | "从观测到动作" | 给定（状态，下一状态），输出连接二者的动作；闭合机器人回路 |
| Cosmos-Drive | "NVIDIA 驾驶仿真" | 开放权重的自动驾驶世界模型，用于 RL 和评估 |

## 延伸阅读

- [Sora technical report (OpenAI)](https://openai.com/index/video-generation-models-as-world-simulators/)
- [Genie: Generative Interactive Environments (Bruce et al., 2024)](https://arxiv.org/abs/2402.15391) —— 潜在动作世界模型
- [TimeSformer (Bertasius et al., 2021)](https://arxiv.org/abs/2102.05095) —— 视频 Transformer 的分离式注意力
- [DreamerV3 (Hafner et al., 2023)](https://arxiv.org/abs/2301.04104) —— 用于 RL 的世界模型
- [Cosmos-Drive-Dreams (NVIDIA, 2025)](https://research.nvidia.com/labs/toronto-ai/cosmos-drive-dreams/) —— 驾驶世界模型
- [Top 10 Video Generation Models 2026 (DataCamp)](https://www.datacamp.com/blog/top-video-generation-models)
- [From Video Generation to World Model — survey repo](https://github.com/ziqihuangg/Awesome-From-Video-Generation-to-World-Model/)
