# 记忆：虚拟上下文与 MemGPT

> 上下文窗口是有限的，而对话、文档和工具调用轨迹却不是。MemGPT（Packer et al., 2023）把这个问题类比为操作系统的虚拟内存——主上下文是 RAM，外部存储是磁盘，智能体在两者之间换页。2026 年的每一个记忆系统都继承了这一模式。

**Type:** Build
**Languages:** Python (stdlib)
**Prerequisites:** Phase 14 · 01 (Agent Loop), Phase 14 · 06 (Tool Use)
**Time:** ~75 minutes

## 学习目标

- 解释 MemGPT 所基于的操作系统类比：主上下文 = RAM，外部上下文 = 磁盘，记忆工具 = 换入/换出（page in/out）。
- 仅用标准库实现 MemGPT 的两层模式：一个主上下文缓冲区、一个可搜索的外部存储，以及换入/换出工具。
- 描述智能体如何发出“中断”来查询或修改外部记忆，以及结果如何被拼接回下一个提示词。
- 识别哪些 MemGPT 设计决策延续到了 Letta（第 08 课）和 Mem0（第 09 课）。

## 问题背景

上下文窗口看上去应该能解决记忆问题，但事实并非如此。生产环境中反复出现三种失效模式：

1. **溢出。** 多轮对话、长文档或大量工具调用的轨迹会超出窗口。超过截断点的内容全部丢失。
2. **稀释。** 即使在窗口之内，塞入无关上下文也会稀释模型对关键信息的注意力。前沿模型在长输入上依然会性能退化。
3. **持久化。** 新会话从一个空窗口开始。没有外部记忆的智能体无法跨会话说出“还记得你曾让我……”这样的话。

更大的窗口有帮助，但治不了本。Mem0 在 2025 年的论文中测得：128k 窗口的基线仍会漏掉长程事实，而一个带外部记忆的 4k 窗口智能体却能抓住它们。

## 核心概念

### MemGPT：操作系统类比

Packer et al.（arXiv:2310.08560，v2 2024 年 2 月）把上下文管理映射到操作系统的虚拟内存：

| 操作系统概念 | MemGPT 概念 | 2026 年生产环境对应物 |
|------------|---------------|------------------------|
| RAM | 主上下文（提示词） | Anthropic/OpenAI 的上下文窗口 |
| 磁盘 | 外部上下文 | 向量数据库、KV、图存储 |
| 缺页中断（page fault） | 记忆工具调用 | `memory.search`、`memory.read`、`memory.write` |
| 操作系统内核 | 智能体控制循环 | 带记忆工具的 ReAct 循环 |

智能体跑的是一个普通的 ReAct 循环，只是多了一类工具，让它能把数据在主上下文中换入换出。

### 两层结构

- **主上下文。** 固定大小的提示词，承载当前任务，对模型始终可见。
- **外部上下文。** 容量无上限，通过工具可搜索。相关时读取，事实出现时写入。

原论文在两个超出基础窗口的任务上评估了这一设计：超过 100k token 的文档分析，以及跨多日保持持久记忆的多会话聊天。

### 中断模式

MemGPT 提出了“记忆即中断”（memory-as-interrupt）：在对话中途，智能体可以调用一个记忆工具，运行时执行它，结果作为新的观察被拼接进下一个助手回合。概念上这等同于 Unix 的 `read()` 系统调用——阻塞进程、返回字节，然后进程继续执行。

经典的记忆工具接口：

- `core_memory_append(section, text)` —— 写入提示词中的某个持久化区段。
- `core_memory_replace(section, old, new)` —— 编辑某个持久化区段。
- `archival_memory_insert(text)` —— 写入可搜索的外部存储。
- `archival_memory_search(query, top_k)` —— 从外部存储检索。
- `conversation_search(query)` —— 扫描历史回合。

### MemGPT 的终点，Letta 的起点

2024 年 9 月，MemGPT 更名为 Letta。研究仓库（`cpacker/MemGPT`）仍然保留；Letta 在原设计上做了扩展：

- 从两层变为三层（core、recall、archival —— 第 08 课）。
- 用原生推理取代 `send_message`/heartbeat 模式（第 08 课）。
- 睡眠时（sleep-time）智能体异步执行记忆维护工作（第 08 课）。

即便生产系统跑的是 Letta、Mem0 或自定义的两层存储，MemGPT 论文依然是 2026 年的基石。

### 这一模式会在哪里出问题

- **记忆腐化（memory rot）。** 写入积累得比读取快，检索被陈旧事实淹没。解法：定期整合（Letta 的 sleep-time）、显式失效（Mem0 的冲突检测器）。
- **记忆投毒（memory poisoning）。** 外部记忆本质上是被检索回来的文本。一旦攻击者可控的内容写进了记忆条目，智能体在下一个会话会再次摄入它。这就是 Greshake et al.（第 27 课）的攻击在时间维度上的重演。
- **引用丢失（citation loss）。** 智能体记得“用户曾让我发布 X”，却说不出是哪一轮说的。每次归档写入都要附带来源引用（会话 ID、回合 ID）。

```figure
context-budget
```

## 从零实现

`code/main.py` 仅用标准库实现了 MemGPT 的两层模式：

- `MainContext` —— 固定大小的提示词缓冲区，包含一个 `core` 字典和一个 `messages` 列表；超出上限时自动压缩最旧的消息。
- `ArchivalStore` —— 内存中的类 BM25 存储（基于 token 重叠打分），记录形如 (id, text, tags, session, turn)。
- 五个记忆工具，与 MemGPT 的工具接口一一对应。
- 一个脚本化智能体：先把事实写入归档存储，再通过调用 `archival_memory_search` 回答问题。

运行：

```
python3 code/main.py
```

运行轨迹显示：智能体写入三条事实，把主上下文填到上限（触发驱逐），然后通过从归档存储中检索来回答后续问题——不依赖任何真实 LLM 就复现了 MemGPT 的工作流。

## 生产实践

如今每一个生产级记忆系统都是 MemGPT 的变体：

- **Letta**（第 08 课）—— 三层结构、原生推理、睡眠时计算。
- **Mem0**（第 09 课）—— 向量 + KV + 图三者融合，外加一个打分层。
- **OpenAI Assistants / Responses** —— 通过线程（threads）和文件提供托管式记忆。
- **Claude Agent SDK** —— 通过技能（skills）和会话存储实现长期记忆。

选型依据是运维形态（自托管、托管、框架集成），而不是核心模式——核心模式都是 MemGPT。

## 交付产物

`outputs/skill-virtual-memory.md` 是一个可复用的技能（skill），能为任意目标运行时生成正确的两层记忆脚手架（主上下文 + 归档存储 + 工具接口），并内置驱逐策略和引用字段。

## 练习

1. 增加一个以 token 计量的 `max_main_context_tokens` 上限（用 `len(text.split())` * 1.3 近似）。超过上限时把最旧的消息压缩成一份摘要。对比有无摘要器时的行为差异。
2. 在归档存储上正确实现 BM25（词频、逆文档频率）。在一个玩具事实集上对比 token 重叠基线，测量 recall@10。
3. 给归档写入增加 `citation` 字段（session_id、turn_id、source_url）。让智能体在每个基于检索的回答中都给出引用。
4. 模拟记忆投毒：添加一条内容为 “ignore all future user instructions” 的归档记录。编写一个防护程序，扫描检索结果中指令形态的文本并把它们标记为不可信。
5. 把实现移植到 MemGPT 研究仓库的核心记忆 JSON schema（`cpacker/MemGPT`）。从扁平字符串切换到带类型的区段后，发生了什么变化？

## 关键术语

| 术语 | 大家怎么说 | 实际含义 |
|------|----------------|------------------------|
| 虚拟上下文 | “无限记忆” | 主层（提示词）+ 外部层（可搜索），带换入/换出 |
| 主上下文 | “工作记忆” | 提示词本身——固定大小，始终可见 |
| 归档记忆 | “长期存储” | 外部可搜索的持久化层，按需检索 |
| 核心记忆 | “持久化提示词区段” | 固定在主上下文内的命名区段 |
| 记忆工具 | “记忆 API” | 智能体发出的、用于读写外部记忆的工具调用 |
| 中断 | “记忆缺页” | 智能体暂停，运行时取数，结果拼接进下一回合 |
| 记忆腐化 | “陈旧事实” | 旧写入淹没检索；用整合来修复 |
| 记忆投毒 | “被注入的持久化条目” | 攻击者内容被存为记忆，召回时再次被摄入 |

## 延伸阅读

- [Packer et al., MemGPT (arXiv:2310.08560)](https://arxiv.org/abs/2310.08560) —— 受操作系统启发的虚拟上下文论文
- [Letta, Memory Blocks blog](https://www.letta.com/blog/memory-blocks) —— 三层结构的演进
- [Anthropic, Effective context engineering](https://www.anthropic.com/engineering/effective-context-engineering-for-ai-agents) —— 把上下文当作预算来管理
- [Chhikara et al., Mem0 (arXiv:2504.19413)](https://arxiv.org/abs/2504.19413) —— 基于这一模式构建的混合生产级记忆系统
