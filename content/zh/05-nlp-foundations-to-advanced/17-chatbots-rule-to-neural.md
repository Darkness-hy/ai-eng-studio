# 聊天机器人——从规则到神经网络再到 LLM 智能体

> ELIZA 用模式匹配来回复。DialogFlow 把输入映射成意图。GPT 从模型权重里给出答案。Claude 会调用工具并验证结果。每一代都解决了上一代最显眼的失败。

**Type:** Learn
**Languages:** Python
**Prerequisites:** Phase 5 · 13 (Question Answering), Phase 5 · 14 (Information Retrieval)
**Time:** ~75 minutes

## 问题背景

用户说"我想改签航班"。系统必须弄清楚用户想要什么、缺哪些信息、怎么获取这些信息、如何完成这个操作。接着用户又说"等等，要是我直接退票呢？"——系统必须记住上下文、切换任务，并保留状态。

对话对机器学习系统来说很难。输入是开放式的，输出要在多轮对话中保持连贯，系统可能还要对真实世界采取行动（改签航班、扣款）。每一步出错，用户都看得见。

聊天机器人架构经历了四代范式的更替，每一代的出现都是因为上一代的失败太过显眼。本课按顺序逐一讲解。2026 年的生产格局是后两代范式的混合体。

## 核心概念

![Chatbot evolution: rule-based → retrieval → neural → agent](../assets/chatbot.svg)

**基于规则（Rule-based，如 ELIZA、AIML、DialogFlow）。** 手工编写的模式匹配用户输入并产出回复。意图分类器把请求路由到预定义的流程。槽位填充（slot-filling）状态机收集必需的信息。在设计好的狭窄范围内表现出色，一旦超出范围立刻失效。在不容许幻觉的安全关键领域（银行身份验证、航班预订）至今仍在线上运行。

**基于检索（Retrieval-based）。** 一种 FAQ 式系统。把每一对（用户话语，回复）编码成向量。运行时对用户消息编码，检索最近邻的已存回复。可以参考 Zendesk 经典的"相似文章"功能。比规则更能处理同义改写。不做生成，所以没有幻觉。

**神经网络（seq2seq）。** 在对话日志上训练的编码器-解码器，从零生成回复。流畅，但容易输出泛泛之词（"我不知道"）并出现事实漂移，从来无法稳定地切题。这正是 2016-2019 年间 Google、Facebook、Microsoft 的聊天机器人全都令人失望的原因。

**LLM 智能体（LLM agents）。** 把语言模型包在一个会规划、调用工具、验证结果的循环里。它不是套了一段长提示词的聊天机器人，而是一个智能体循环：规划 → 调用工具 → 观察结果 → 决定下一步。检索优先的事实锚定（RAG）防止它产生幻觉，工具调用让它能真正做事。这就是 2026 年的架构。

这四代范式并不是顺序替代的关系。一个 2026 年的生产级聊天机器人会同时路由到这四种范式：规则负责身份验证和破坏性操作，检索负责 FAQ，神经生成负责自然的措辞，LLM 智能体负责模糊的开放式查询。

## 从零实现

### 第 1 步：基于规则的模式匹配

```python
import re


class RulePattern:
    def __init__(self, pattern, response_template):
        self.regex = re.compile(pattern, re.IGNORECASE)
        self.template = response_template


PATTERNS = [
    RulePattern(r"my name is (\w+)", "Nice to meet you, {0}."),
    RulePattern(r"i (need|want) (.+)", "Why do you {0} {1}?"),
    RulePattern(r"i feel (.+)", "Why do you feel {0}?"),
    RulePattern(r"(.*)", "Tell me more about that."),
]


def rule_based_respond(user_input):
    for pattern in PATTERNS:
        m = pattern.regex.match(user_input.strip())
        if m:
            return pattern.template.format(*m.groups())
    return "I don't understand."
```

20 行代码实现 ELIZA。反射技巧（"I feel sad" → "Why do you feel sad"）就是 Weizenbaum 1966 年那个经典的心理治疗师演示，至今仍有教学价值。

### 第 2 步：基于检索（FAQ）

下面这段演示代码需要 `pip install sentence-transformers`（会连带安装 torch）。本课可运行的 `code/main.py` 改用标准库实现的 Jaccard 相似度，因此整节课无需外部依赖即可运行。

```python
from sentence_transformers import SentenceTransformer
import numpy as np


FAQ = [
    ("how do i reset my password", "Go to Settings > Security > Reset Password."),
    ("how do i cancel my order", "Go to Orders, find the order, click Cancel."),
    ("what is your return policy", "30-day returns on unused items, original packaging."),
]


encoder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
faq_questions = [q for q, _ in FAQ]
faq_embeddings = encoder.encode(faq_questions, normalize_embeddings=True)


def faq_respond(user_input, threshold=0.5):
    q_emb = encoder.encode([user_input], normalize_embeddings=True)[0]
    sims = faq_embeddings @ q_emb
    best = int(np.argmax(sims))
    if sims[best] < threshold:
        return None
    return FAQ[best][1]
```

基于阈值的拒答是这里的关键设计决策。如果最佳匹配不够接近，就返回 `None`，让系统升级处理。

### 第 3 步：神经生成（基线）

使用一个小型指令微调过的编码器-解码器（FLAN-T5），或一个微调过的对话模型。单独使用的话在 2026 年完全达不到生产可用（自相矛盾、偏题漂移、事实性胡说），但作为混合系统的一部分，它负责输出自然的措辞。DialoGPT 这类仅解码器（decoder-only）模型需要显式的轮次分隔符和 EOS 处理才能产出连贯的回复；作为教学示例，FLAN-T5 的 text2text 流水线开箱即用。

```python
from transformers import pipeline

chatbot = pipeline("text2text-generation", model="google/flan-t5-small")

response = chatbot("Respond politely to: Hi there!", max_new_tokens=40)
print(response[0]["generated_text"])
```

### 第 4 步：LLM 智能体循环

2026 年的生产形态：

```python
def agent_loop(user_message, tools, llm, max_steps=5):
    history = [{"role": "user", "content": user_message}]
    for _ in range(max_steps):
        response = llm(history, tools=tools)
        tool_call = response.get("tool_call")
        if tool_call:
            tool_name = tool_call.get("name")
            args = tool_call.get("arguments")
            if not isinstance(tool_name, str) or tool_name not in tools:
                history.append({"role": "assistant", "tool_call": tool_call})
                history.append({"role": "tool", "name": str(tool_name), "content": f"error: unknown tool {tool_name!r}"})
                continue
            if not isinstance(args, dict):
                history.append({"role": "assistant", "tool_call": tool_call})
                history.append({"role": "tool", "name": tool_name, "content": f"error: arguments must be a dict, got {type(args).__name__}"})
                continue
            fn = tools[tool_name]
            result = fn(**args)
            history.append({"role": "assistant", "tool_call": tool_call})
            history.append({"role": "tool", "name": tool_name, "content": result})
        else:
            return response["content"]
    return "I could not complete the task in the step budget."
```

有三个要点需要明确命名。工具（tools）是 LLM 可以调用的函数。当 LLM 返回最终答案而不是工具调用时，循环终止。步数预算（step budget）防止在模糊任务上陷入无限循环。

真实的生产系统还会加上：检索优先的事实锚定（每次调用 LLM 前注入相关文档）、护栏（未经确认就拒绝执行破坏性操作）、可观测性（记录每一步日志），以及评估（自动检查智能体行为是否符合规范）。

### 第 5 步：混合路由

```python
def hybrid_chat(user_input):
    if is_destructive_action(user_input):
        return structured_flow(user_input)

    faq_answer = faq_respond(user_input, threshold=0.6)
    if faq_answer:
        return faq_answer

    return agent_loop(user_input, tools, llm)


def is_destructive_action(text):
    danger_words = ["delete", "cancel", "charge", "refund", "transfer"]
    return any(w in text.lower() for w in danger_words)
```

这个模式是：所有破坏性操作走确定性规则，固定 FAQ 走检索，其余一切交给 LLM 智能体。2026 年线上运行的客服系统就是这么做的。

## 生产实践

2026 年的技术栈：

| 用例 | 架构 |
|---------|---------------|
| 预订、支付、身份验证 | 基于规则的状态机 + 槽位填充 |
| 客服 FAQ | 在精选答案上做检索 |
| 开放式帮助对话 | 带 RAG 和工具调用的 LLM 智能体 |
| 内部工具 / IDE 助手 | 带工具调用（搜索、读、写）的 LLM 智能体 |
| 陪伴 / 角色聊天机器人 | 微调过的 LLM，配人设系统提示词，在知识上做检索 |

生产环境务必使用混合路由。没有任何单一架构能把所有请求都处理好。路由层本身通常是一个小型意图分类器。

## 至今仍在线上发生的失败模式

- **自信地编造。** LLM 智能体声称完成了它并未执行的操作。缓解措施：验证结果、记录工具调用日志，绝不允许 LLM 在没有成功的工具返回的情况下声称做过某件事。
- **提示词注入（Prompt injection）。** 用户插入覆盖系统提示词的文本。在 OWASP Top 10 for LLM Applications 2025 中被列为 LLM01。有两种形式：直接注入（粘贴进聊天框）和间接注入（藏在智能体读取的文档、邮件或工具输出里）。

  攻击成功率因场景而异。在通用工具使用和编程基准测试中，对前沿模型的实测成功率约为 0.5-8.5%。特定的高风险场景（针对 AI 编程智能体的自适应攻击、存在漏洞的编排层）成功率曾高达约 84%。生产环境的 CVE 包括 EchoLeak（CVE-2025-32711，CVSS 9.3）——Microsoft 365 Copilot 中由攻击者控制的邮件触发的零点击数据外泄漏洞。

  缓解措施：在整个循环中把用户输入视为不可信；调用工具前先做净化；把工具输出与主提示词隔离；使用规划-验证-执行（Plan-Verify-Execute，PVE）模式——智能体先规划，再对照计划验证每个动作后才执行（这能阻止工具结果注入计划外的新动作）；破坏性操作要求用户确认；对工具权限范围实行最小权限原则。

  再多的提示词工程也无法完全消除这一风险。外部运行时防御层（LLM Guard、白名单校验、语义异常检测）是必需的。
- **任务范围蔓延。** 某次工具调用返回了沾边但无关的信息，智能体因此偏离任务。缓解措施：收窄工具契约；保持系统提示词聚焦；为偏题率添加评估。
- **无限循环。** 智能体反复调用同一个工具。缓解措施：步数预算、工具调用去重、用 LLM 评判"我们是否在取得进展"。
- **上下文窗口耗尽。** 长对话把最早的轮次挤出上下文。缓解措施：对较早的轮次做摘要、按相似度检索相关的历史轮次，或改用长上下文模型。

## 交付产物

保存为 `outputs/skill-chatbot-architect.md`：

```markdown
---
name: chatbot-architect
description: Design a chatbot stack for a given use case.
version: 1.0.0
phase: 5
lesson: 17
tags: [nlp, agents, chatbot]
---

Given a product context (user need, compliance constraints, available tools, data volume), output:

1. Architecture. Rule-based, retrieval, neural, LLM agent, or hybrid (specify which paths go where).
2. LLM choice if applicable. Name the model family (Claude, GPT-4, Llama-3.1, Mixtral). Match to tool-use quality and cost.
3. Grounding strategy. RAG sources, retrieval method (see lesson 14), tool contracts.
4. Evaluation plan. Task success rate, tool-call correctness, off-task rate, hallucination rate on held-out dialogs.

Refuse to recommend a pure-LLM agent for any destructive action (payments, account deletion, data modification) without a structured confirmation flow. Refuse to skip the prompt-injection audit if the agent has write access to anything.
```

## 练习

1. **简单。** 用上面的规则式回复实现一个咖啡店点单机器人，写 10 条模式。测试边界情况：重复下单、修改订单、取消订单、意图不明。
2. **中等。** 构建一个 FAQ + LLM 兜底的混合系统。为某个 SaaS 产品准备 50 条固定 FAQ，LLM 兜底时在文档站点上做检索。在 100 个真实客服问题上测量拒答率和准确率。
3. **困难。** 实现上面的智能体循环，配三个工具（搜索、读取用户数据、发送邮件）。用 50 个测试场景（含提示词注入尝试）跑一次评估。报告偏题率、任务失败率以及任何注入成功的案例。

## 关键术语

| 术语 | 大家口中的说法 | 实际含义 |
|------|-----------------|-----------------------|
| 意图（Intent） | 用户想要什么 | 类别标签（book_flight、reset_password）。路由到对应的处理器。 |
| 槽位（Slot） | 一条信息 | 机器人需要的参数（日期、目的地）。槽位填充就是逐项追问的过程。 |
| RAG | 检索加生成 | 先检索相关文档，再用它们为 LLM 的回复提供事实锚定。 |
| 工具调用（Tool call） | 函数调用 | LLM 输出一个带名称和参数的结构化调用，运行时执行并返回结果。 |
| 智能体循环（Agent loop） | 规划、行动、验证 | 一个控制器，交替执行 LLM 调用和工具调用，直到任务完成。 |
| 提示词注入（Prompt injection） | 用户攻击提示词 | 试图覆盖系统提示词的恶意输入。 |

## 延伸阅读

- [Weizenbaum (1966). ELIZA — A Computer Program For the Study of Natural Language Communication](https://web.stanford.edu/class/cs124/p36-weizenabaum.pdf) —— 基于规则的聊天机器人的开山论文。
- [Thoppilan et al. (2022). LaMDA: Language Models for Dialog Applications](https://arxiv.org/abs/2201.08239) —— Google 的神经聊天机器人末期论文，发表于 LLM 智能体接管之前。
- [Yao et al. (2022). ReAct: Synergizing Reasoning and Acting in Language Models](https://arxiv.org/abs/2210.03629) —— 为智能体循环模式命名的论文。
- [Anthropic's guide on building effective agents](https://www.anthropic.com/research/building-effective-agents) —— 2024 年的生产实践指南，到 2026 年依然成立。
- [Greshake et al. (2023). Not what you've signed up for: Compromising Real-World LLM-Integrated Applications with Indirect Prompt Injection](https://arxiv.org/abs/2302.12173) —— 提示词注入的开创性论文。
- [OWASP Top 10 for LLM Applications 2025 — LLM01 Prompt Injection](https://genai.owasp.org/llmrisk/llm01-prompt-injection/) —— 让提示词注入成为头号安全问题的那份排名。
- [AWS — Securing Amazon Bedrock Agents against Indirect Prompt Injections](https://aws.amazon.com/blogs/machine-learning/securing-amazon-bedrock-agents-a-guide-to-safeguarding-against-indirect-prompt-injections/) —— 编排层防御的实战指南，包括规划-验证-执行和用户确认流程。
- [EchoLeak (CVE-2025-32711)](https://www.vectra.ai/topics/prompt-injection) —— 间接提示词注入导致零点击数据外泄的标志性 CVE。这是说明为什么有写权限的智能体需要运行时防御的参考案例。
