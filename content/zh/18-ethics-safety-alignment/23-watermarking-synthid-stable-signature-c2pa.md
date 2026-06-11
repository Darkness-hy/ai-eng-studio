# 水印技术 — SynthID、Stable Signature 与 C2PA

> 2026 年 AI 生成内容溯源（provenance）格局由三项技术构成。SynthID（Google DeepMind）——图像水印于 2023 年 8 月推出，文本与视频水印于 2024 年 5 月上线（Gemini + Veo），文本水印于 2024 年 10 月通过 Responsible GenAI Toolkit 开源，2025 年 11 月随 Gemini 3 Pro 发布统一的多模态检测器。文本水印通过对下一个 token 的采样概率做不可察觉的调整来嵌入信号；图像/视频水印能在压缩、裁剪、滤镜、帧率变化后存活。Stable Signature（Fernandez et al., ICCV 2023, arXiv:2303.15435）——通过微调潜变量扩散模型的解码器，使每个输出都携带一条固定消息；即使生成图像被裁剪到仅剩 10% 的内容，检测率仍 >90%，FPR<1e-6。后续工作《Stable Signature is Unstable》（arXiv:2405.07145，2024 年 5 月）表明——微调可以在保持图像质量的同时移除水印。C2PA——经密码学签名、可检测篡改的元数据标准（C2PA 2.2 Explainer，2025）。水印与 C2PA 互为补充：元数据可被剥离但承载更丰富的溯源信息；水印能在转码后留存但携带的信息量更少。

**Type:** Build
**Languages:** Python (stdlib, token-watermark embed + detect)
**Prerequisites:** Phase 10 · 04 (sampling), Phase 01 · 09 (information theory)
**Time:** ~75 minutes

## 学习目标

- 描述 token 级水印（SynthID-text 风格）及其可被检测的机制。
- 描述 Stable Signature 以及 2024 年攻破它的水印移除攻击。
- 说明 C2PA 的作用，以及它为何与水印技术互补。
- 描述关键局限：信号依赖特定模型、改写（paraphrase）下的鲁棒性、以及保持语义的攻击（arXiv:2508.20228）。

## 问题背景

2023-2024 年，深度伪造（deepfake）和 AI 生成内容大规模进入政治与消费场景。水印是被提出的技术性溯源信号：在生成时打上标记，事后再检测。2025 年的证据表明：没有任何水印是无条件鲁棒的，但与 C2PA 元数据分层叠加后，组合方案能提供一套可用的溯源体系。

## 核心概念

### 文本水印（SynthID-text 风格）

Kirchenbauer et al. 2023 提出的机制，由 Google 实现了产品化：

1. 在每个解码步，对前 K 个 token 做哈希，生成一个伪随机划分，把词表分成「绿色」和「红色」两个集合。
2. 给绿色集合的 logits 加上 δ，使采样偏向绿色集合。
3. 生成的文本中绿色 token 的占比会高于随机水平。

检测：对每个前缀重新做哈希，统计生成文本中的绿色 token 数量，计算 z 分数（z-score）。带水印的文本 z 分数 >0，人类撰写的文本 z 分数约为 0。

性质：

- 对读者不可察觉（δ 足够小，质量损失轻微）。
- 只要能访问词表划分函数即可检测。
- 对改写不鲁棒——重写文本会摧毁信号。

SynthID-text 于 2024 年 10 月通过 Google 的 Responsible GenAI Toolkit 开源。

### Stable Signature（图像）

Fernandez et al., ICCV 2023。微调潜变量扩散模型的解码器，使每张生成图像都在潜表示中嵌入一条固定的二进制消息。检测时用一个神经解码器从潜表示中解码消息。图像被裁剪到仅剩 10% 的内容时，检测率仍 >90%，FPR<1e-6。

2024 年 5 月的《Stable Signature is Unstable》（arXiv:2405.07145）：微调解码器可以在保持图像质量的同时移除水印。生成后的对抗性微调成本很低；该水印的对抗鲁棒性有限。

### SynthID 统一检测器（2025 年 11 月）

随 Gemini 3 Pro 一同发布：一个多模态检测器，在单个 API 中读取文本、图像、音频、视频中的 SynthID 信号。统一了 Google 的溯源技术栈。

### C2PA

Coalition for Content Provenance and Authenticity（内容来源与真实性联盟）。经密码学签名、可检测篡改的元数据标准。C2PA 2.2 Explainer（2025）。一份 C2PA 清单（manifest）记录溯源声明（谁创建、何时创建、做过哪些变换），并由创建者的密钥签名。

与水印互补：

- 元数据可以被剥离；水印则没那么容易。
- 元数据信息丰富（完整溯源链）；水印只携带少量比特。
- C2PA 依赖平台采纳；水印自动嵌入。

Google 在 Search、Ads 和「About this image」中同时集成了二者。

### 局限

- **依赖特定模型。** SynthID 只给启用了 SynthID 的模型的生成内容打水印。来自未启用 SynthID 的模型的内容没有水印，因此「没有 SynthID 信号」不能证明内容出自人类之手。
- **改写。** 文本水印无法在保持语义的改写后存活。
- **变换攻击。** arXiv:2508.20228（2025）展示了能同时摧毁文本水印和许多图像水印的保义攻击。
- **微调移除。** 根据《Stable Signature is Unstable》，生成后的微调可以移除嵌入的水印。

### EU AI Act 第 50 条

针对 AI 生成内容标注的透明度行为准则（Transparency Code）（首稿 2025 年 12 月，第二稿 2026 年 3 月，按[欧盟委员会状态页面](https://digital-strategy.ec.europa.eu/en/policies/code-practice-ai-generated-content)预计 2026 年 6 月定稿）。截至 2026 年 4 月，该准则仍处于草案阶段，时间表可能变动。它是要求上述技术层落地的监管层。深度伪造内容必须被标注。

### 本课在 Phase 18 中的位置

第 22-23 课关注模型输出了什么（隐私数据、溯源信号）。第 27 课讨论训练数据治理。第 24 课是要求这些技术措施落地的监管框架。

## 生产实践

`code/main.py` 构建了一个玩具级文本水印。token 是整数 0..N-1；带水印的采样会偏向哈希定义的绿色集合。检测器计算绿色 token 的 z 分数。你可以观察 1000 个 token 长度生成的检测效果，看到改写如何摧毁信号，并在人类文本上测量误报率。

## 交付产物

本课产出 `outputs/skill-provenance-audit.md`。给定一个带溯源声明的内容部署，它会审计：水印机制（如有）、C2PA 签名链（如有）、二者各自的对抗鲁棒性，以及各模态的覆盖情况。

## 练习

1. 运行 `code/main.py`。报告 1000 token 带水印生成与人类撰写文本的 z 分数。找出 95% 置信阈值下的误报率。

2. 实现一个改写攻击：用同义词替换 30% 的 token。重新测量 z 分数。

3. 阅读 Kirchenbauer et al. 2023 第 6 节关于鲁棒性的讨论。为什么文本水印在改写下失效，而图像水印却能在裁剪后存活？

4. 设计一个同时使用 SynthID-text + C2PA 元数据的部署方案。描述消费者看到的溯源链。指出每个组件的一种失效模式。

5. 2024 年《Stable Signature is Unstable》的结果表明微调能移除图像水印。设计一项限制此类攻击的部署控制措施——例如，要求微调后的检查点必须经过签名才能发布。

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|-----------------|------------------------|
| SynthID | 「Google 的水印」 | 跨模态溯源信号；覆盖文本、图像、音频、视频 |
| Token 水印 | 「Kirchenbauer 风格」 | 基于有偏采样的文本水印，通过绿色 token 的 z 分数检测 |
| Stable Signature | 「图像水印」 | 通过微调解码器实现的水印；ICCV 2023 |
| C2PA | 「那个元数据标准」 | 经密码学签名、可检测篡改的溯源元数据 |
| 改写鲁棒性 | 「换种说法能不能破掉它」 | 文本水印的性质；目前很有限 |
| 微调移除 | 「对抗性去水印」 | 通过微调解码器移除图像水印的攻击 |
| 跨模态检测器 | 「统一版 SynthID」 | 2025 年 11 月发布的跨模态统一 API |

## 延伸阅读

- [Kirchenbauer et al. — A Watermark for Large Language Models (ICML 2023, arXiv:2301.10226)](https://arxiv.org/abs/2301.10226) — token 水印机制
- [Fernandez et al. — Stable Signature (ICCV 2023, arXiv:2303.15435)](https://arxiv.org/abs/2303.15435) — 图像水印论文
- ["Stable Signature is Unstable" (arXiv:2405.07145)](https://arxiv.org/abs/2405.07145) — 水印移除攻击
- [Google DeepMind — SynthID](https://deepmind.google/models/synthid/) — 跨模态水印
- [C2PA 2.2 Explainer (2025)](https://c2pa.org/specifications/specifications/2.2/explainer/Explainer.html) — 元数据标准
