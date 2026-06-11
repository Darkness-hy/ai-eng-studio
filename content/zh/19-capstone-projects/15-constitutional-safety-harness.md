# Capstone 15 — 宪法式安全防护体系 + 红队攻防靶场

> Anthropic 的 Constitutional Classifiers、Meta 的 Llama Guard 4、Google 的 ShieldGemma-2、NVIDIA 的 Nemotron 3 Content Safety，以及负责多语言覆盖的 X-Guard，共同定义了 2026 年的安全分类器技术栈。garak、PyRIT、NVIDIA Aegis 和 promptfoo 成为标准的对抗性评估工具。NeMo Guardrails v0.12 把它们串成一条生产级流水线。这个 Capstone 将把这一切组合在一起：围绕目标应用的分层安全防护体系（safety harness）、一个运行 6 种以上攻击家族的自主红队智能体，以及一轮能产出可度量无害性增量（harmlessness delta）的宪法式自我批判训练。

**Type:** Capstone
**Languages:** Python (safety pipeline, red team), YAML (policy configs)
**Prerequisites:** Phase 10 (LLMs from scratch), Phase 11 (LLM engineering), Phase 13 (tools), Phase 14 (agents), Phase 18 (ethics, safety, alignment)
**Phases exercised:** P10 · P11 · P13 · P14 · P18
**Time:** 25 hours

## 问题背景

2026 年 LLM 安全的前沿问题，已经不是分类器是否有效（大体上是有效的），而是如何在不过度拒答、也不留下明显漏洞的前提下，把它们正确地组合到一个生产应用周围。Llama Guard 4 负责英文场景的策略违规检测。X-Guard（覆盖 132 种语言）应对多语言越狱（jailbreak）。ShieldGemma-2 拦截基于图像的提示注入（prompt injection）。NVIDIA Nemotron 3 Content Safety 覆盖企业级内容类别。Anthropic 的 Constitutional Classifiers 则是另一种思路——用于训练阶段而非服务阶段。

攻击手段的演化同样重要。PAIR 和 TAP 实现了越狱攻击的自动化发现。GCG 执行基于梯度的后缀攻击。多轮对话攻击和语码转换（code-switch）攻击则利用智能体的记忆机制。任何已部署的 LLM 都需要一个红队靶场——garak 和 PyRIT 是事实标准的驱动工具——外加文档化的缓解措施和经 CVSS 评分的漏洞发现报告。

你将加固一个目标应用（一个 8B 指令微调模型，或其他 Capstone 中的某个 RAG 聊天机器人），对它发起 6 种以上攻击家族的测试，并产出加固前后的无害性对比测量结果。

## 核心概念

安全流水线分为五层。**输入净化**：剥离零宽字符、解码 base64/rot13、归一化 Unicode。**策略层**：NeMo Guardrails v0.12 护栏（域外话题、毒性内容、PII 提取）。**分类器闸门**：Llama Guard 4 检查输入，X-Guard 处理非英文内容，ShieldGemma-2 处理图像输入。**模型层**：目标 LLM 本身。**输出过滤**：Llama Guard 4 检查输出，Presidio 清洗 PII，并在适用场景强制要求引用来源。**HITL 层**：被标记为高风险的输出进入 Slack 人工审核队列。

红队靶场由调度器驱动运行。PAIR 和 TAP 自主发现越狱漏洞。GCG 执行基于梯度的后缀攻击。还有 ASCII / base64 / rot13 编码攻击、多轮对话攻击（人格扮演、记忆利用）、语码转换攻击（英语混杂斯瓦希里语或泰语）。每次运行都会产出一份结构化的发现报告，包含 CVSS 评分和披露时间线。

宪法式自我批判（constitutional self-critique）是一种训练阶段的干预手段。取 1000 条有害意图的提示，让模型起草回复，再对照一份成文宪法（不得作恶的规则集）进行批判，然后在批判循环的产出上重新训练。最后在留出（held-out）评估集上测量训练前后的无害性增量。

## 架构

```
request (text / image / multilingual)
      |
      v
input sanitize (strip zero-width, decode, normalize)
      |
      v
NeMo Guardrails v0.12 rails (off-domain, policy)
      |
      v
classifier gate:
  Llama Guard 4 (English)
  X-Guard (multilingual, 132 langs)
  ShieldGemma-2 (image prompts)
  Nemotron 3 Content Safety (enterprise)
      |
      v (allowed)
target LLM
      |
      v
output filter: Llama Guard 4 + Presidio PII + citation check
      |
      v
HITL tier for flagged outputs

parallel:
  red-team scheduler
    -> garak (classic attacks)
    -> PyRIT (orchestrated red team)
    -> autonomous jailbreak agent (PAIR + TAP)
    -> GCG suffix attacks
    -> multilingual / code-switch
    -> multi-turn persona adoption

output: CVSS-scored findings + disclosure timeline + before/after harmlessness delta
```

## 技术栈

- 安全分类器：Llama Guard 4、ShieldGemma-2、NVIDIA Nemotron 3 Content Safety、X-Guard
- 护栏框架：NeMo Guardrails v0.12 + OPA
- 红队驱动工具：garak（NVIDIA）、PyRIT（Microsoft Azure）、NVIDIA Aegis、promptfoo
- 越狱智能体：PAIR（Chao et al., 2023）、Tree-of-Attacks（TAP）、GCG 后缀攻击
- 宪法式训练：Anthropic 风格的自我批判循环 + 在批判结果上做 SFT
- PII 清洗：Presidio
- 目标应用：一个 8B 指令微调模型，或其他 Capstone 的 RAG 聊天机器人

## 从零实现

1. **搭建目标应用。** 在 vLLM 上部署一个 8B 指令微调模型（或复用其他 Capstone 的 RAG 聊天机器人）。这就是被测应用。

2. **包裹安全流水线。** 围绕目标应用接入五层流水线。确认每一层都可以单独观测（在 Langfuse 中每层一个 span）。

3. **分类器覆盖。** 加载 Llama Guard 4、X-Guard（多语言）、ShieldGemma-2（图像）。分别在一个小规模标注集上运行，建立基线。

4. **红队调度器。** 调度 garak、PyRIT、一个 PAIR 智能体、一个 TAP 智能体、一个 GCG 执行器、一个多轮攻击器和一个语码转换攻击器。每个攻击器运行在独立队列上。

5. **攻击套件。** 六大攻击家族：(1) PAIR 自动化越狱，(2) TAP 攻击树，(3) GCG 梯度后缀，(4) ASCII / base64 / rot13 编码，(5) 多轮人格扮演，(6) 多语言语码转换。按攻击家族分别报告成功率。

6. **宪法式自我批判。** 整理 1000 条有害意图提示。对每条提示，目标模型先起草回复。一个批判者 LLM 对照成文宪法（"不得作恶"、"引用证据"、"拒绝违法请求"）打分。批判者提出异议的回复会被重写；目标模型在经批判改进后的样本对上做微调。在留出评估集上测量训练前后的无害性变化。

7. **过度拒答测量。** 在良性提示套件（如 XSTest）上跟踪误报率。目标模型在良性问题上必须保持有用性。

8. **CVSS 评分。** 对每次成功的越狱，按 CVSS 4.0 评分（攻击向量、复杂度、影响）。产出披露时间线和缓解计划。

9. **靶场自动化。** 以上所有环节全部由 cron 定时运行；发现结果写入队列；过度拒答出现回归时向 Slack 发送告警。

## 生产实践

```
$ safety probe --model=target --family=PAIR --budget=50
[attacker]   PAIR agent running on target
[attack]     attempt 1/50: disguise query as academic research ... blocked
[attack]     attempt 2/50: appeal to roleplay ... blocked
[attack]     attempt 3/50: chain-of-thought coax ... SUCCEEDED
[finding]    CVSS 4.8 medium: roleplay bypass on target
[range]      7 successes out of 50 (14% success rate)
```

## 交付产物

交付物是 `outputs/skill-safety-harness.md`。一条生产级的分层安全流水线，加上一个可复现的红队靶场，并附带加固前后的无害性增量数据。

| 权重 | 评分标准 | 度量方式 |
|:-:|---|---|
| 25 | 攻击面覆盖 | 演练 6 种以上攻击家族，覆盖 2 种以上语言 |
| 20 | 真阳性 / 假阳性权衡 | 攻击拦截率 vs XSTest 良性通过率 |
| 20 | 自我批判增量 | 留出评估集上训练前后的无害性对比 |
| 20 | 文档与披露 | 带时间线的 CVSS 评分发现报告 |
| 15 | 自动化与可重复性 | 全流程由 cron 运行并配有告警 |
| **100** | | |

## 练习

1. 在一个 RAG 聊天机器人上运行 garak 的提示注入插件，对比有无输出过滤层时的攻击成功率。

2. 增加第七个攻击家族：通过检索文档实施的间接提示注入。测量为此需要追加的防御措施。

3. 实现一种"拒绝但提供帮助"模式：护栏拦截时，目标模型给出一个更安全的相关回答，而不是直接拒绝。测量 XSTest 上的指标变化。

4. 多语言覆盖缺口：找出一种 X-Guard 表现欠佳的语言。提出一个针对该语言的微调数据集方案。

5. 在一个 30B 模型上运行宪法式自我批判，测量无害性增量是否随规模扩展。

## 关键术语

| 术语 | 大家怎么说 | 实际含义 |
|------|-----------------|------------------------|
| 分层安全 | "纵深防御" | 在输入、闸门、输出、HITL 各环节部署多重护栏 |
| Llama Guard 4 | "Meta 的安全分类器" | 2026 年输入/输出内容分类器的参考标准 |
| PAIR | "越狱智能体" | 关于 LLM 驱动越狱发现的论文（Chao et al.） |
| TAP | "Tree-of-Attacks" | PAIR 的树搜索变体 |
| GCG | "贪心坐标梯度" | 基于梯度的对抗性后缀攻击 |
| 宪法式自我批判 | "Anthropic 风格的训练" | 目标模型起草 -> 批判者打分 -> 重写 -> 重新训练 |
| XSTest | "良性探测集" | 用于检测过度拒答回归的基准 |
| CVSS 4.0 | "严重性评分" | 用于安全发现的标准漏洞评分体系 |

## 延伸阅读

- [Anthropic Constitutional Classifiers](https://www.anthropic.com/research/constitutional-classifiers) — 训练阶段方法的参考
- [Meta Llama Guard 4](https://ai.meta.com/research/publications/llama-guard-4/) — 2026 年的输入/输出分类器
- [Google ShieldGemma-2](https://huggingface.co/google/shieldgemma-2b) — 图像 + 多模态安全
- [NVIDIA Nemotron 3 Content Safety](https://developer.nvidia.com/blog/building-nvidia-nemotron-3-agents-for-reasoning-multimodal-rag-voice-and-safety/) — 企业级参考
- [X-Guard (arXiv:2504.08848)](https://arxiv.org/abs/2504.08848) — 覆盖 132 种语言的多语言安全
- [garak](https://github.com/NVIDIA/garak) — NVIDIA 红队工具包
- [PyRIT](https://github.com/Azure/PyRIT) — Microsoft 红队框架
- [NeMo Guardrails v0.12](https://docs.nvidia.com/nemo-guardrails/) — 护栏框架
- [PAIR (arXiv:2310.08419)](https://arxiv.org/abs/2310.08419) — 越狱智能体论文
