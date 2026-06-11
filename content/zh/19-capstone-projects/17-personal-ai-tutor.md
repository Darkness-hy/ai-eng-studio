# 毕业项目 17 —— 个人 AI 导师（自适应、多模态、带记忆）

> Khanmigo（Khan Academy）、Duolingo Max、Google LearnLM / Gemini for Education、Quizlet Q-Chat 和 Synthesis Tutor 在 2026 年都已规模化交付自适应多模态辅导产品。它们的共同形态是：苏格拉底式策略（绝不直接抛出答案）、每次交互后都会更新的学习者模型（贝叶斯知识追踪风格）、语音 + 文本 + 拍照数学题输入、课程图谱检索、间隔重复调度，以及针对年龄适宜内容的硬性安全过滤。本毕业项目的目标是交付一个特定学科的导师（K-12 代数或 Python 入门），对 10 名学习者开展为期两周的成效研究，并通过内容安全审计。

**Type:** Capstone
**Languages:** Python (backend, learner model), TypeScript (web app), SQL (curriculum graph via Postgres + Neo4j)
**Prerequisites:** Phase 5 (NLP), Phase 6 (speech), Phase 11 (LLM engineering), Phase 12 (multimodal), Phase 14 (agents), Phase 17 (infrastructure), Phase 18 (safety)
**涉及阶段：** P5 · P6 · P11 · P12 · P14 · P17 · P18
**Time:** 30 hours

## 问题背景

自适应辅导曾经只是教育科技领域的研究小众方向。到 2026 年，它已经成为消费级产品。Khanmigo 已部署到美国大多数学区。Duolingo Max 的月活用户达到数千万。Google 的 LearnLM / Gemini for Education 为 Google Classroom 中的辅导功能提供支持。Quizlet Q-Chat 与抽认卡功能并列上线。Synthesis Tutor 凭借"给好奇孩子的导师"实现了病毒式传播。它们的共同要素是：多模态输入（打字、说话、拍摄方程）、苏格拉底式教学法（先提问，后讲解）、每次交互后更新的学习者模型，以及严格的年龄适宜安全机制。

你将为一个特定人群构建这样一款产品。衡量标准是一次真实的成效研究：10 名学习者，历时两周，前测与后测成绩对比。语音环路必须自然流畅（复用毕业项目 03 的子技术栈）。记忆系统必须尊重隐私。安全过滤器必须通过面向 K-12、符合 COPPA 要求的红队测试。

## 核心概念

四个组件。**导师策略（Tutor policy）**是一个苏格拉底式循环：当学习者直接索要答案时，策略会抛出一个引导性问题；当学习者答对时，转向下一个概念；当学习者卡住时，提供脚手架式提示。**学习者模型（Learner model）**是贝叶斯知识追踪（Bayesian knowledge tracing，或其简化变体），在每次交互后更新每个课程节点的掌握概率。**课程图谱（Curriculum graph）**是一个存放概念及其先修关系边的 Neo4j 图数据库；策略通过遍历图谱来挑选下一个概念。**记忆（Memory）**是情景记忆 + 语义记忆存储（agentmemory 风格），保存历史交互、错误和偏好。

用户体验是多模态的。文本输入用于键入答案。语音输入通过 LiveKit + Whisper（复用毕业项目 03）。拍照输入用于数学题，通过 dots.ocr 或 PaliGemma 2 识别。语音输出通过 Cartesia Sonic-2。安全方面使用 Llama Guard 4，外加年龄适宜过滤器（拦截成人内容、暴力、自残），以及符合 COPPA 要求的记忆保留策略。

成效研究就是交付物。10 名学习者，前测与后测，为期两周。报告学习增益差值及置信区间。与非自适应基线（同样的内容按线性顺序交付、不使用导师策略）进行对比。

## 架构

```
learner device
  |
  +-- text         -> web app
  +-- voice        -> LiveKit Agents (ASR + TTS)
  +-- photo math   -> dots.ocr / PaliGemma 2
       |
       v
  tutor policy (LangGraph)
       - Socratic decision head
       - next-concept chooser (curriculum graph walk)
       - hint scaffolder
       - mastery update
       |
       v
  learner model (BKT / item-response theory)
       - per-concept mastery probability
       - spaced-repetition scheduler (SM-2 or FSRS)
       |
       v
  memory (agentmemory-style)
       - episodic: every interaction
       - semantic: learned mistakes, preferences
       - retention policy: COPPA / GDPR aware
       |
       v
  curriculum graph (Neo4j)
       - prerequisite edges
       - OER content attached
       |
       v
  safety:
    Llama Guard 4 + age-appropriate filter
    memory access guarded by learner ID scope
```

## 技术栈

- 学科选择：K-12 代数或 Python 入门（选定一个深入做）
- 导师策略：基于 Claude Sonnet 4.7 的 LangGraph（启用提示词缓存）
- 学习者模型：贝叶斯知识追踪（经典版）或用 FSRS 做间隔调度
- 课程图谱：Neo4j，包含概念 + 先修关系边 + OER 内容
- 记忆：agentmemory 风格的持久化向量 + 情景 + 语义存储
- 语音：LiveKit Agents 1.0 + Cartesia Sonic-2（复用毕业项目 03 的子技术栈）
- 拍照数学题：dots.ocr 或 PaliGemma 2 做方程识别
- 安全：Llama Guard 4 + 自定义年龄适宜过滤器
- 评估：Bloom 层级问题生成、前测/后测框架、成效研究工具链

## 从零实现

1. **课程图谱。** 构建一个包含 50-150 个概念节点的 Neo4j 图谱（例如 K-12 代数，从"数轴"到"求根公式"），节点之间用先修关系边连接。为每个节点挂接 OER 内容（Open Textbook、OpenStax）。

2. **学习者模型。** 用先验参数初始化贝叶斯知识追踪：猜对率（guess）、失误率（slip）、学习率（learn-rate）。每次交互后更新各概念的掌握度。按学习者持久化存储。

3. **导师策略。** 用 LangGraph 构建，节点包括：`read_signal`（学习者的回答是正确 / 部分正确 / 卡住了？）、`select_concept`（遍历课程图谱挑选优先级最高的概念）、`scaffold`（苏格拉底式提示）、`update_mastery`。

4. **记忆。** 每次交互都写入情景存储。错误和偏好提升到语义记忆。符合 COPPA 要求的保留策略：1 年后自动删除，家长可访问。

5. **语音链路。** 将 LiveKit Agents worker 接入导师策略。ASR 用 Whisper-v3-turbo。TTS 用 Cartesia Sonic-2。支持打断（barge-in，复用毕业项目 03 的机制）。

6. **拍照数学题链路。** 上传或拍摄图片；运行 dots.ocr 或 PaliGemma 2 识别方程；以结构化输入的形式喂给导师。

7. **安全。** 每条模型输出都要经过 Llama Guard 4 + 年龄适宜过滤器（拦截自残、成人内容、暴力）。记忆访问按学习者 ID 限定范围；提供供家长执行删除操作的访问入口。

8. **成效研究。** 10 名学习者，前测（标准化的 30 题基线测试），两周的导师交互（每周 3 次），后测。与使用相同内容的 10 名非自适应基线组学习者进行对比。

9. **每周进度报告。** 为每位学习者自动生成 PDF 摘要，涵盖已探索的主题、掌握度轨迹和推荐的后续步骤。

## 生产实践

```
learner: "I don't understand why 3x + 6 = 12 means x = 2"
[signal]   stuck
[concept]  'isolating variables' (prerequisite: addition-subtraction-equality)
[scaffold] "what number would you subtract from both sides to start?"
learner: "6"
[signal]   correct
[mastery]  addition-subtraction-equality: 0.62 -> 0.77
[concept]  continue 'isolating variables'
[scaffold] "great. now what is 3x / 3 equal to?"
```

## 交付产物

交付物是 `outputs/skill-ai-tutor.md`。一个面向特定学科的自适应导师，具备多模态输入、学习者模型、记忆、安全机制和经过实测的成效。

| 权重 | 评分项 | 测量方式 |
|:-:|---|---|
| 25 | 学习增益差值 | 10 名学习者两周研究中的前测/后测差值 |
| 20 | 苏格拉底式忠实度 | 对话记录样本的量规评分 |
| 20 | 多模态用户体验 | 语音 + 拍照 + 文本的端到端连贯性 |
| 20 | 安全与隐私态势 | Llama Guard 4 通过率 + 符合 COPPA 的数据保留 |
| 15 | 课程广度与图谱质量 | 概念覆盖度 + 先修关系图谱一致性 |
| **100** | | |

## 练习

1. 分别在启用和禁用自适应学习者模型（概念随机排序）的情况下运行成效研究。报告两者的差值。预期自适应版本会胜出，但差距的大小才是真正有意思的数字。

2. 增加一个多模态探针：同一个概念问题分别以文本、语音和图片形式交付。测量学习者在使用其偏好的模态时是否收敛更快。

3. 构建家长仪表盘：练习过的主题、掌握度轨迹、即将学习的概念、安全事件（任何护栏触发记录）。符合 COPPA 要求。

4. 增加语言切换模式：导师接受西班牙语输入并用西班牙语授课。测量 X-Guard 的覆盖率。

5. 对记忆隐私做压力测试：验证学习者 A 即使通过语音片段重新摄取攻击也无法看到学习者 B 的数据。记录访问尝试并发出告警。

## 关键术语

| 术语 | 大家怎么说 | 实际含义 |
|------|-----------------|------------------------|
| 苏格拉底式策略 | "提问，别灌输" | 导师抛出引导性问题，而不是直接给出答案 |
| 贝叶斯知识追踪 | "BKT" | 经典的学习者模型方程组，计算每个概念的掌握概率 |
| FSRS | "Free Spaced Repetition Scheduler" | 2024 年的间隔重复调度算法，优于 SM-2 |
| 课程图谱 | "概念 DAG" | 存放概念及先修关系边的 Neo4j 图 |
| 情景记忆 | "逐次交互日志" | 每次交互都被存储，供日后检索 |
| 语义记忆 | "习得模式存储" | 从情景记忆中提炼压缩出的错误和偏好 |
| COPPA | "儿童隐私法" | 美国法律，限制对 13 岁以下儿童的数据收集 |

## 延伸阅读

- [Khanmigo (Khan Academy)](https://www.khanmigo.ai) —— 消费级 K-12 导师的参考产品
- [Duolingo Max](https://blog.duolingo.com/duolingo-max/) —— 语言学习导师的参考产品
- [Google LearnLM / Gemini for Education](https://blog.google/technology/google-deepmind/learnlm) —— 托管参考模型
- [Quizlet Q-Chat](https://quizlet.com) —— 备选参考产品
- [Synthesis Tutor](https://www.synthesis.com) —— 创业公司参考产品
- [FSRS algorithm](https://github.com/open-spaced-repetition/fsrs4anki) —— 间隔重复调度器
- [Bayesian Knowledge Tracing](https://en.wikipedia.org/wiki/Bayesian_knowledge_tracing) —— 学习者模型经典方法
- [LiveKit Agents](https://github.com/livekit/agents) —— 语音技术栈
