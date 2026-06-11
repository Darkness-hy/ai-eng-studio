# 基准测试：WebArena 与 OSWorld

> WebArena 在四个自托管应用上测试 Web 智能体的能力。OSWorld 在 Ubuntu、Windows、macOS 上测试桌面智能体的能力。两者在发布时（2023–2024 年）都显示出顶尖智能体与人类之间的巨大差距。如今差距在缩小，但失败模式没有变。

**Type:** Learn
**Languages:** Python (stdlib)
**Prerequisites:** Phase 14 · 19 (SWE-bench, GAIA)
**Time:** ~60 minutes

## 学习目标

- 描述 WebArena 的四个自托管应用，以及基于执行的评估为何重要。
- 解释 OSWorld 为何使用真实操作系统截图而非无障碍 API。
- 说出 OSWorld 揭示的两大主要失败模式：GUI 定位与操作知识。
- 概括 OSWorld-G 和 OSWorld-Human 在基础基准之上增加了什么。

## 问题背景

通用智能体能调用工具。但它们能驾驭浏览器、连续点击 20 次完成一次购物结账吗？能只用键盘和鼠标配置一台 Linux 机器吗？WebArena 和 OSWorld 回答的正是这些问题。

## 核心概念

### WebArena（Zhou et al., ICLR 2024）

- 812 个长程任务，分布在四个自托管 Web 应用上：一个购物网站、一个论坛、一个类 GitLab 的开发工具、一个企业 CMS。
- 外加辅助工具：地图、计算器、草稿板。
- 评估通过 gym API 基于执行结果进行——订单是否下达、issue 是否关闭、CMS 页面是否更新？
- 发布时：最佳 GPT-4 智能体成功率为 14.41%，而人类为 78.24%。

自托管这一设定很关键——由于目标应用版本固定且可复现，基准测试不会出现不稳定的结果。

### 扩展

- **VisualWebArena**——视觉定位任务，成功与否取决于对图像的理解（截图作为一等公民的观测输入）。
- **TheAgentCompany**（2024 年 12 月）——增加终端与编程能力，更接近真实的远程办公环境。

### OSWorld（Xie et al., NeurIPS 2024）

- 369 个真实计算机任务，覆盖 Ubuntu、Windows、macOS。
- 对真实应用进行自由形式的键盘和鼠标控制。
- 以 1920×1080 截图作为观测输入。
- 发布时：最佳模型成功率 12.24%，而人类为 72.36%。

### 主要失败模式

1. **GUI 定位（GUI grounding）。** 像素到界面元素的映射。模型很难在 1920×1080 分辨率下可靠地定位 UI 元素。
2. **操作知识（operational knowledge）。** 哪个菜单里有这个设置、用哪个快捷键、在哪个偏好设置面板。这是人类经年累月积累的长尾知识。

### 后续工作

- **OSWorld-G**——564 个样本的定位测试套件 + Jedi 训练集。将定位能力与规划能力解耦，使二者可以分开度量。
- **OSWorld-Human**——人工整理的黄金动作轨迹。表明顶尖智能体使用的步数是必要步数的 1.4-2.7 倍（即轨迹效率差距）。

### 为什么这很重要

Claude computer use、OpenAI CUA、Gemini 2.5 Computer Use（第 21 课）的训练负载都由 WebArena 和 OSWorld 塑造。基准测试是靶标；生产模型是交付的答案。

### 基准测试容易出错的地方

- **只用截图做评估。** OSWorld 是截图驱动的；如果在 OSWorld 上评估一个使用 DOM 或无障碍 API 的智能体，就绕过了定位这一挑战。
- **忽视轨迹长度。** 只看成功率会漏掉 OSWorld-Human 揭示的 1.4-2.7 倍步数低效问题。
- **过时的自托管应用。** WebArena 的应用固定在特定版本；不经重新整理就升级会破坏结果的可比性。

## 从零实现

`code/main.py` 实现了一个玩具级 Web 智能体测试框架：

- 一个最小化的"购物应用"状态机：list_items、add_to_cart、checkout。
- 3 个任务的黄金轨迹。
- 一个按脚本执行每个任务的智能体。
- 基于执行的评估器（状态检查）和轨迹效率指标（实际步数对比黄金步数）。

运行方式：

```
python3 code/main.py
```

输出：每个任务的成功率与轨迹效率，复现 OSWorld-Human 的方法论。

## 生产实践

- **WebArena Verified** 自托管在内部集群上，用于持续评估。
- **OSWorld** 部署在虚拟机集群中，用于评估桌面智能体。
- **计算机操作智能体**（第 21 课）——Claude、OpenAI CUA、Gemini——都在这类工作负载上训练。
- **你自己的产品流程**——为你最重要的 20 个任务采集黄金轨迹；每周用智能体跑一遍。

## 交付产物

`outputs/skill-web-desktop-harness.md` 构建一个带有基于执行的评估和轨迹效率指标的 Web/桌面智能体测试框架。

## 练习

1. 给玩具框架扩展第二个应用（一个论坛）。编写 3 个任务及对应的黄金轨迹。
2. 增加按任务汇报轨迹效率的功能。在你的玩具框架上，智能体的步数是黄金轨迹的 1 倍、2 倍还是 3 倍？
3. 实现一个"干扰"工具——黄金轨迹从不使用的工具。脚本化智能体会被它诱惑吗？
4. 阅读 OSWorld-G。在你自己的评估中，你会如何把定位失败与规划失败区分开？
5. 阅读 WebArena 各应用的 README。当你升级其中一个固定版本的应用时，会有什么东西被破坏？

## 关键术语

| 术语 | 通俗说法 | 实际含义 |
|------|----------------|------------------------|
| WebArena | "Web 智能体基准" | 4 个自托管应用上的 812 个任务；gym 风格评估 |
| VisualWebArena | "视觉版 WebArena" | 视觉定位版 WebArena；截图作为观测输入 |
| OSWorld | "桌面智能体基准" | 真实 Ubuntu/Windows/macOS 上的 369 个任务 |
| GUI 定位 | "像素到元素的映射" | 模型在 1920x1080 分辨率下定位 UI 元素 |
| 操作知识 | "操作系统使用经验" | 哪个菜单、哪个快捷键、哪个偏好设置面板 |
| OSWorld-G | "定位测试套件" | 564 个纯定位样本 + 训练集 |
| OSWorld-Human | "黄金轨迹" | 人工专家动作序列，用于度量效率 |
| 轨迹效率 | "相对黄金轨迹的步数倍率" | 智能体步数除以人类最少步数 |

## 延伸阅读

- [Zhou et al., WebArena (arXiv:2307.13854)](https://arxiv.org/abs/2307.13854) — 四应用 Web 基准
- [Xie et al., OSWorld (arXiv:2404.07972)](https://arxiv.org/abs/2404.07972) — 跨操作系统桌面基准
- [Anthropic, Introducing computer use](https://www.anthropic.com/news/3-5-models-and-computer-use) — Claude 由基准塑造的能力
- [OpenAI, Computer-Using Agent](https://openai.com/index/computer-using-agent/) — OSWorld 与 WebArena 上的成绩
