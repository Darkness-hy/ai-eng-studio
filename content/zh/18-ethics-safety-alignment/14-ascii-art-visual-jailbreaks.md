# ASCII 艺术与视觉越狱

> Jiang, Xu, Niu, Xiang, Ramasubramanian, Li, Poovendran, "ArtPrompt: ASCII Art-based Jailbreak Attacks against Aligned LLMs" (ACL 2024, arXiv:2402.11753)。把有害请求中与安全相关的 token 遮蔽掉，换成同样字母的 ASCII 艺术渲染，再把这条伪装后的提示词发出去。GPT-3.5、GPT-4、Gemini、Claude、Llama-2 全都无法稳健地识别 ASCII 艺术 token。这种攻击能绕过 PPL（困惑度过滤器）、改写（Paraphrase）防御和重分词（Retokenization）。相关工作：ViTC 基准测量模型对非语义视觉提示的识别能力；StructuralSleight 将其推广到非常见文本编码结构（Uncommon Text-Encoded Structures，如树、图、嵌套 JSON），构成一整类编码攻击。

**Type:** Build
**Languages:** Python (stdlib, ArtPrompt token-masking harness)
**Prerequisites:** Phase 18 · 12 (PAIR), Phase 18 · 13 (MSJ)
**Time:** ~60 minutes

## 学习目标

- 描述 ArtPrompt 攻击的流程：词语识别步骤、ASCII 艺术替换、最终的伪装提示词。
- 解释为什么标准防御（PPL、改写、重分词）对 ArtPrompt 失效。
- 给出 ViTC 的定义并说明它测量什么。
- 说明 StructuralSleight 如何将该攻击推广到任意非常见文本编码结构。

## 问题背景

通过改写和角色扮演的攻击（第 12 课）以及通过长上下文的攻击（第 13 课）都作用在文本层面的模式上。ArtPrompt 则作用在识别层面：模型并没有解析那个被禁止的 token，它解析的是一幅由字符渲染出来的图像。安全过滤器看到的是无害的标点符号，而模型看到的是一个单词。

## 核心概念

### ArtPrompt 的两个步骤

第 1 步：词语识别（Word Identification）。给定一条有害请求，攻击者用一个 LLM 找出其中与安全相关的词（例如 "how to make a bomb" 中的 "bomb"）。

第 2 步：伪装提示词生成（Cloaked Prompt Generation）。把每个识别出的词替换成它的 ASCII 艺术渲染（用 7x5 或 7x7 的字符块拼出字母形状）。模型收到的是一片由标点和空格组成的网格，能力足够强的模型能把它识别成那个单词，而安全过滤器只看到一片网格。

结果：GPT-4、Gemini、Claude、Llama-2、GPT-3.5 全部失守。在他们的基准测试子集上，攻击成功率超过 75%。

### 为什么标准防御会失效

- **PPL（困惑度过滤器）。** ASCII 艺术的困惑度很高——但所有新颖输入的困惑度都很高。能拦住 ArtPrompt 的阈值，同样会拦住合法的结构化输入。
- **改写（Paraphrase）。** 改写提示词会破坏 ASCII 艺术。但在实践中，负责改写的 LLM 往往会保留甚至重建这些艺术字。
- **重分词（Retokenization）。** 换一种方式切分 token，改变不了模型是在"看"字母形状这一事实。

根本问题在于：安全过滤器工作在 token 层或语义层，而 ArtPrompt 工作在视觉识别层。

### ViTC 基准

测量对非语义视觉提示的识别能力。它衡量模型读取 ASCII 艺术、wingdings 字体以及其他非文本语义视觉内容的能力。ArtPrompt 的攻击效果与 ViTC 准确率正相关：模型读视觉文本的能力越强，ArtPrompt 对它就越有效。这是一种能力与安全的权衡（capability-safety tradeoff）。

### StructuralSleight

将 ArtPrompt 推广为非常见文本编码结构（Uncommon Text-Encoded Structures，UTES）：树、图、嵌套 JSON、JSON 内嵌 CSV、diff 风格的代码块。只要某种结构在训练安全数据中罕见、但模型又能解析它，它就能用来藏匿有害内容。

对防御的启示：安全能力必须能泛化到模型可解析的所有结构化表示上。而这个集合很大，并且还在增长。

### 图像模态的对应攻击

视觉 LLM（GPT-5.2、Gemini 3 Pro、Claude Opus 4.5、Grok 4.1）扩大了攻击面。使用真实图像的 ArtPrompt 式攻击比 ASCII 艺术版本更强，因为图像编码器产生的信号更丰富。

### 在 Phase 18 中的位置

第 12-14 课描述了三个相互正交的攻击向量：迭代式改进（PAIR）、上下文长度（MSJ）和编码（ArtPrompt/StructuralSleight）。第 15 课从以模型为中心的攻击转向系统边界攻击（间接提示注入）。第 16 课介绍防御性工具链的应对方案。

## 生产实践

`code/main.py` 构建了一个简易版 ArtPrompt。你可以用 ASCII 艺术字形遮蔽有害查询中的特定词语，验证伪装后的字符串能通过关键词过滤器，还可以（可选地）用一个简单的识别器把伪装字符串解码回原词。

## 交付产物

本课产出 `outputs/skill-encoding-audit.md`。给定一份越狱防御报告，它会列举报告覆盖的各编码攻击家族（ASCII 艺术、base64、leet-speak、UTF-8 同形字、UTES），以及拦截每种攻击的防御层。

## 练习

1. 运行 `code/main.py`。验证伪装后的字符串能通过一个简单的关键词过滤器。报告所需的字符级改动量。

2. 实现第二种编码：对同一个目标词使用 base64。比较它与 ArtPrompt 的过滤器绕过率和恢复难度。

3. 阅读 Jiang et al. 2024 第 4.3 节（五个模型的结果）。提出一个原因，解释为什么在同一基准上 Claude 对 ArtPrompt 的抵抗力高于 Gemini。

4. 设计一种生成前防御，检测提示词中形似 ASCII 艺术的区域。在合法的代码、表格和数学符号上测量误报率。

5. StructuralSleight 列出了 10 种编码结构。勾画一个能覆盖全部 10 种的通用防御方案，并估算每条受防护提示词的计算开销。

## 关键术语

| 术语 | 大家怎么说 | 实际含义 |
|------|-----------------|------------------------|
| ArtPrompt | "那个 ASCII 艺术攻击" | 一种两步越狱：用 ASCII 艺术渲染遮蔽安全相关词 |
| 伪装（Cloaking） | "把那个词藏起来" | 把被禁止的 token 换成一种模型读得懂、过滤器读不懂的视觉表示 |
| UTES | "非常见结构" | 非常见文本编码结构（Uncommon Text-Encoded Structure）——树、图、嵌套 JSON 等，用于偷运内容 |
| ViTC | "视觉文本能力" | 衡量模型读取非语义视觉编码能力的基准 |
| 困惑度过滤器 | "PPL 防御" | 拒绝高困惑度的提示词；失效原因是合法的结构化输入分数同样很高 |
| 重分词（Retokenization） | "分词器偏移防御" | 用另一种分词器预处理提示词；失效原因是识别发生在视觉层面 |
| 同形字（Homoglyph） | "长得一样的字符" | 看起来与拉丁字母完全相同的 Unicode 字符；可绕过子串检查 |

## 延伸阅读

- [Jiang et al. — ArtPrompt (ACL 2024, arXiv:2402.11753)](https://arxiv.org/abs/2402.11753) — ASCII 艺术越狱论文
- [Li et al. — StructuralSleight (arXiv:2406.08754)](https://arxiv.org/abs/2406.08754) — UTES 推广
- [Chao et al. — PAIR (Lesson 12, arXiv:2310.08419)](https://arxiv.org/abs/2310.08419) — 互补的迭代式攻击
- [Anil et al. — Many-shot Jailbreaking (Lesson 13)](https://www.anthropic.com/research/many-shot-jailbreaking) — 互补的长上下文攻击
