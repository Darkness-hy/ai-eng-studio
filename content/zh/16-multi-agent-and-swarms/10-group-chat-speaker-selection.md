# 群聊与发言者选择

> AutoGen GroupChat 和 AG2 GroupChat 让 N 个智能体共享同一段对话；由一个选择器函数（LLM、轮询或自定义）决定下一个发言者。这是涌现式多智能体对话的原型——智能体并不知道自己在某个静态图中的角色，它们只是对共享消息池做出反应。AutoGen v0.2 的 GroupChat 语义在 AG2 分支中得以保留；AutoGen v0.4 则将其重写为事件驱动的 actor 模型。Microsoft 于 2026 年 2 月将 AutoGen 转入维护模式，并将其与 Semantic Kernel 合并为 Microsoft Agent Framework（2026 年 2 月发布 RC 版）。GroupChat 这一原语在 AG2 和 Microsoft Agent Framework 中都得以延续——学一次，处处可用。

**Type:** Learn + Build
**Languages:** Python (stdlib)
**Prerequisites:** Phase 16 · 04 (Primitive Model)
**Time:** ~60 minutes

## 问题背景

当工作流已知时，静态图（LangGraph）非常好用。但真实的对话并不是静态的：有时编码者要问审查者，有时问研究员，有时问写作者。把每一种可能的交接都硬编码出来，会导致边的数量爆炸。你想要的是*让智能体对一个共享消息池做出反应*，再由某个函数决定下一个发言者。

这正是 AutoGen GroupChat 所做的事。

## 核心概念

### 整体形态

```
              ┌─── shared pool ────┐
              │   m1  m2  m3  ...  │
              └─────────┬──────────┘
                        │ (everyone reads all)
      ┌───────┬─────────┼─────────┬───────┐
      ▼       ▼         ▼         ▼       ▼
    Agent A  Agent B  Agent C  Agent D  Selector
                                           │
                                           ▼
                                  "next speaker = C"
```

每个智能体都能看到每条消息。每一轮都会调用一个选择器函数来挑选下一个发言者。

### 三种选择器风格

**轮询（Round-robin）。** 固定循环，确定性强。随 N 线性扩展，但完全不考虑上下文——即使话题是法律审查，也会轮到编码者发言。

**LLM 选择。** 调用一次 LLM，让它读取最近的消息池并返回最合适的下一个发言者。具备上下文感知能力，但速度慢：每一轮都会多一次 LLM 调用。这是 AutoGen 的默认方式。

**自定义。** 一个 Python 函数，逻辑随你定义。典型做法：LLM 选择加兜底规则（例如"编码者发言之后，总是把发言权交给验证者"）。

### ConversableAgent API

```
agent = ConversableAgent(
    name="coder",
    system_message="You write Python.",
    llm_config={...},
)
chat = GroupChat(agents=[coder, reviewer, tester], messages=[])
manager = GroupChatManager(groupchat=chat, llm_config={...})
```

`GroupChatManager` 持有选择器。当一个智能体完成发言后，管理器调用选择器，由它返回下一个智能体。循环持续进行，直到满足终止条件。

### 终止条件

三种常见模式：

- **最大轮数。** 对总轮数设置硬性上限。
- **"TERMINATE" 令牌。** 智能体可以发出一条哨兵消息；管理器一旦发现就停止对话。
- **目标达成检查。** 一个轻量级验证器在每轮运行，任务完成时停止聊天。

### AutoGen → AG2 的分裂与 Microsoft Agent Framework 的合并

2025 年初，Microsoft 开始围绕事件驱动的 actor 模型对 AutoGen 进行大规模重写（v0.4）。社区将 AutoGen v0.2 的 GroupChat 语义分叉为 AG2，保留了早期用户已经集成的 API。

2026 年 2 月，Microsoft 宣布 AutoGen 进入维护模式，事件驱动的 actor 模型并入 **Microsoft Agent Framework**（2026 年 2 月发布 RC 版，现已与 Semantic Kernel 合并）。GroupChat 概念在两条路线中都得以延续，但实现细节有所不同。对于兼容 v0.2 的代码，AG2 是首选的上游项目。

### GroupChat 适用的场景

- **涌现式对话。** 你不想预先连好每一种可能的"下一个发言者"。
- **角色混合任务。** 编码者问研究员，研究员问档案员，档案员又反过来问编码者。流程不是一个 DAG。
- **探索式问题求解。** 想象"头脑风暴会议"，而不是"流水线"。

### 失效场景

- **严格确定性。** LLM 选择器可能不一致。相同提示词、不同运行，选出的下一个发言者也不同。
- **谄媚级联（sycophancy cascades）。** 智能体会附和说话最自信的那一方。需要在提示词中明确加以抵消。
- **上下文膨胀。** 每个智能体读取每条消息；10 轮之后上下文就会非常庞大。使用投影（第 15 课）来限定视图范围。
- **热点发言者。** 某个智能体主导整场对话，因为选择器偏爱它的专长。可以把发言者均衡机制加进选择器。

### 群聊 vs 监督者

原语相同，默认行为不同：

- 监督者：一个智能体负责规划，其他智能体负责执行。选择器就是"问规划者下一步做什么"。
- 群聊：所有智能体都是平等的；选择器是作用在共享消息池上的一个函数。

二者都使用第 04 课的四个原语。群聊默认采用 LLM 选择的编排方式和全池共享状态。

## 从零实现

`code/main.py` 仅用标准库从零实现了一个 GroupChat。包含三个智能体（编码者、审查者、管理者）、轮询和 LLM 选择两种变体，以及基于 `TERMINATE` 令牌的终止机制。

演示会打印两种变体的对话记录，以及选择器的决策轨迹。

运行：

```
python3 code/main.py
```

## 生产实践

`outputs/skill-groupchat-selector.md` 为给定任务配置 GroupChat 选择器——轮询 vs LLM 选择 vs 自定义，以及该使用哪些选择器输入（最近消息、智能体专长、轮次计数）。

## 交付产物

检查清单：

- **最大轮数上限。** 必须设置。典型任务取 10-20。
- **发言者均衡指标。** 跟踪每个智能体的发言轮次；失衡超过阈值时告警。
- **终止令牌。** `TERMINATE` 或一个专门的验证者智能体。
- **投影或限定范围的记忆。** 约 10 条消息之后，考虑只给每个智能体一个限定范围的视图，防止上下文膨胀。
- **选择器日志。** 对于 LLM 选择的变体，同时记录选择器的输入和它的选择。否则无法调试。

## 练习

1. 运行 `code/main.py`。对比轮询与 LLM 选择两种方式下的对话。每种方式下哪个智能体占主导？
2. 在选择器中加一条"每个智能体最大发言次数"规则。它对对话记录有什么影响？
3. 实现一个目标达成式终止：当审查者返回 "approved" 时停止。它在轮数上限触发之前生效的频率有多高？
4. 阅读 AutoGen 稳定版文档中关于 GroupChat 的部分（https://microsoft.github.io/autogen/stable/user-guide/core-user-guide/design-patterns/group-chat.html）。找出 `GroupChatManager` 使用的默认选择器。
5. 阅读 AG2 仓库（https://github.com/ag2ai/ag2），对比其 v0.2 GroupChat 与 v0.4 事件驱动版本。v0.4 具体增加了哪种性质（吞吐量、容错性、可组合性）？

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|----------------|------------------------|
| GroupChat | "智能体在同一个聊天室里" | 共享消息池 + 选择器函数。AutoGen / AG2 的原语。 |
| 发言者选择 | "下一个谁说话" | 挑选下一个智能体的函数。轮询、LLM 选择或自定义。 |
| GroupChatManager | "会议主持人" | 持有选择器并循环推进轮次的 AutoGen 组件。 |
| ConversableAgent | "基础智能体" | AutoGen 基类；一个能收发消息的智能体。 |
| 终止令牌 | "那个'停止'词" | 结束聊天的哨兵字符串（通常是 `TERMINATE`）。 |
| 热点发言者 | "一个智能体占主导" | 选择器反复挑选同一个智能体的失效模式。 |
| 上下文膨胀 | "消息池无限增长" | 每个智能体读取之前的每条消息；上下文随轮次增长。 |
| 投影 | "限定范围的视图" | 针对角色的共享消息池视图，用于防止上下文膨胀。 |

## 延伸阅读

- [AutoGen group chat docs](https://microsoft.github.io/autogen/stable/user-guide/core-user-guide/design-patterns/group-chat.html) — 参考实现
- [AG2 repo](https://github.com/ag2ai/ag2) — 社区维护的 AutoGen v0.2 延续项目
- [Microsoft Agent Framework docs](https://microsoft.github.io/agent-framework/) — 合并后的后继框架，2026 年 2 月发布 RC 版
- [AutoGen v0.4 release notes](https://microsoft.github.io/autogen/stable/) — 事件驱动 actor 模型重写的细节
