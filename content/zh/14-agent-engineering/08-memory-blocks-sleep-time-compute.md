# 记忆块与睡眠时计算（Letta）

> MemGPT 在 2024 年更名为 Letta。2026 年的演进新增了两个想法：模型可以直接编辑的离散功能性记忆块（memory block），以及在主代理空闲时异步整理记忆的睡眠时代理（sleep-time agent）。这就是把记忆扩展到单次对话之外的方法。

**Type:** Build
**Languages:** Python (stdlib)
**Prerequisites:** Phase 14 · 07 (MemGPT)
**Time:** ~75 minutes

## 学习目标

- 说出 Letta 使用的三个记忆层级（core、recall、archival）及各自的作用。
- 解释记忆块模式：Human 块、Persona 块，以及作为一等类型化对象的用户自定义块。
- 描述什么是睡眠时计算（sleep-time compute）、它为何位于关键路径之外、为何可以使用比主代理更强的模型。
- 实现一个脚本化的双代理循环：主代理负责响应，睡眠时代理在轮次之间整理记忆块。

## 问题背景

MemGPT（第 07 课）解决了虚拟内存式的控制流。但随之暴露出三个生产环境问题：

1. **延迟。** 每次记忆操作都位于关键路径上。如果代理必须在用户等待时进行裁剪、摘要或调和，尾延迟就会暴涨。
2. **记忆腐化。** 写入不断累积。被推翻的事实仍然留存。检索被陈旧内容淹没。
3. **结构丢失。** 扁平的归档存储无法表达"Human 块始终在提示词中；Persona 块始终在提示词中；Task 块按会话切换"。

Letta（letta.com）是 2026 年的重写版本。记忆块让结构显式化；睡眠时计算把整理工作移出关键路径。

## 核心概念

### 三个层级

| 层级 | 范围 | 所在位置 | 写入方 |
|------|-------|----------------|------------|
| Core | 始终可见 | 主提示词内部 | 代理工具调用 + 睡眠时重写 |
| Recall | 对话历史 | 可检索 | 自动轮次日志 |
| Archival | 任意事实 | 向量 + KV + 图 | 代理工具调用 + 睡眠时摄入 |

Core 就是 MemGPT 的核心记忆。Recall 是对话缓冲区及其被淘汰的尾部。Archival 是外部存储。这种划分理清了 MemGPT 两层结构的职责混杂问题。

### 记忆块

块是 core 层中一个类型化、持久化、可编辑的区段。最初的 MemGPT 论文定义了两个：

- **Human 块** —— 关于用户的事实（姓名、角色、偏好、目标）。
- **Persona 块** —— 代理的自我认知（身份、语气、约束）。

Letta 将其泛化为任意用户自定义块：表示当前目标的 `Task` 块、记录代码库事实的 `Project` 块、承载硬性约束的 `Safety` 块。每个块都有 `id`、`label`、`value`、`limit`（字符上限）、`description`（让模型知道何时该编辑它）。

块可以通过工具接口编辑：

- `block_append(label, text)`
- `block_replace(label, old, new)`
- `block_read(label)`
- `block_summarize(label)` —— 压缩一个接近上限的块。

### 睡眠时计算

这是 2025 年 Letta 新增的特性：在后台运行第二个代理，脱离关键路径。睡眠时代理处理对话记录和代码库上下文，把 `learned_context` 写入共享块，并对归档记录做整理或失效处理。

由此自然获得的特性：

- **零延迟开销。** 主代理的响应不需要等待记忆操作。
- **允许更强的模型。** 睡眠时代理不受延迟约束，因此可以使用更昂贵、更慢的模型。
- **天然的整理窗口。** 在用户没有等待时去重、摘要、使被推翻的事实失效。

这个形态与人类的工作方式一致：你完成任务，睡一觉，长期记忆在夜间自然沉淀。

### Letta V1 与原生推理

Letta V1（`letta_v1_agent`，2026）废弃了 `send_message`/heartbeat 和内联的 `Thought:` 标记，改用原生推理（native reasoning）。Responses API（OpenAI）和支持扩展思考的 Messages API（Anthropic）在独立通道上输出推理内容，并跨轮次传递（生产环境中跨供应商加密传输）。控制循环仍然是 ReAct。思考轨迹是结构化的，而不是嵌入提示词的。

### 这个模式的常见出错点

- **块膨胀。** 无限 `block_append` 会很快撞上上限。在写入即将超限之前，接入块摘要器。
- **静默漂移。** 睡眠时代理重写了某个块，主代理却毫无察觉。给块加版本，并在轨迹中展示差异（diff）。
- **整理投毒。** 睡眠时代理把攻击者可触达的内容整理进 core。第 27 课的内容同样适用于睡眠时这个攻击面。

## 从零实现

`code/main.py` 实现了：

- `Block` —— id、label、value、limit、description。
- `BlockStore` —— CRUD + `near_limit(label)` 辅助方法。
- 两个脚本化代理 —— `PrimaryAgent` 处理一个轮次，`SleepTimeAgent` 在轮次之间整理。
- 一份轨迹，展示三轮带块写入的对话，外加一次睡眠时处理：摘要一个块并使一条陈旧事实失效。

运行：

```
python3 code/main.py
```

输出记录展示了这种分工：主代理的轮次很快，产生原始写入；睡眠时处理负责压缩和清理。

## 生产实践

- **Letta**（letta.com）作为参考实现。可自托管或使用托管云服务。
- **Claude Agent SDK 的技能（skills）** 即块形态的知识 —— 一个 skill 就是一个有名称、有版本、可检索的指令块，代理按需加载。
- **自研方案** 适合想完全掌控存储后端的团队。遵循 Letta 的 API 契约，以便日后迁移。

## 交付产物

`outputs/skill-memory-blocks.md` 为任意运行时生成一套 Letta 形态的块系统，带睡眠时钩子，包含安全规则和引用接线。

## 练习

1. 添加一个 `block_summarize` 工具：当 `near_limit` 返回 true 时，用模型生成的摘要替换块的值。哪个触发阈值能同时最小化摘要调用次数和块溢出？
2. 对归档存储实现睡眠时去重：两条记录的文本 token 重叠超过 90% 时合并为一条。只在睡眠时处理中执行，绝不在关键路径上执行。
3. 给块加版本。每次写入都记录旧值和差异。暴露 `block_history(label)`，让运维人员能排查"代理为什么忘了 X"。
4. 把睡眠时代理当作不可信的写入方。当它们改动 Persona 或 Safety 块时，要求第二个代理审核后才能提交。
5. 把示例移植到 Letta API（`letta_v1_agent`）。块的 schema 会发生什么变化？原生推理又如何改变轨迹的形态？

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|----------------|------------------------|
| 记忆块（Memory block） | "可编辑的提示词区段" | core 记忆中类型化、持久化、可由 LLM 编辑的片段 |
| Human 块 | "用户记忆" | 关于用户的事实，固定在 core 中 |
| Persona 块 | "代理身份" | 自我认知、语气、约束，固定在 core 中 |
| 睡眠时计算（Sleep-time compute） | "异步记忆工作" | 由第二个代理在关键路径之外执行整理 |
| Core / Recall / Archival | "层级" | 三层记忆划分：始终可见 / 对话 / 外部 |
| 块上限（Block limit） | "上限" | 每个块的字符限制；倒逼摘要 |
| 原生推理（Native reasoning） | "思考通道" | 供应商层面的推理输出，而非提示词层面的 `Thought:` |
| 习得上下文（Learned context） | "睡眠产出" | 睡眠时代理写入共享块的事实 |

## 延伸阅读

- [Letta, Memory Blocks blog](https://www.letta.com/blog/memory-blocks) —— 记忆块模式
- [Letta, Sleep-time Compute blog](https://www.letta.com/blog/sleep-time-compute) —— 异步整理
- [Letta, Rearchitecting the Agent Loop](https://www.letta.com/blog/letta-v1-agent) —— 原生推理重写
- [Packer et al., MemGPT (arXiv:2310.08560)](https://arxiv.org/abs/2310.08560) —— 起源
