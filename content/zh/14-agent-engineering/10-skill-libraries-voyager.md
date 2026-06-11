# 技能库与终身学习（Voyager）

> Voyager（Wang et al., TMLR 2024）将可执行代码视为技能。技能有名称、可检索、可组合，并通过环境反馈不断打磨。它是 Claude Agent SDK skills、skillkit 以及 2026 年技能库模式的参考架构。

**Type:** Build
**Languages:** Python (stdlib)
**Prerequisites:** Phase 14 · 07 (MemGPT), Phase 14 · 08 (Letta Blocks)
**Time:** ~75 minutes

## 学习目标

- 说出 Voyager 的三个组件——自动课程、技能库、迭代提示——以及各自的作用。
- 解释为什么 Voyager 把动作空间定义为代码，而不是原子命令。
- 用 Python 标准库实现一个技能库，支持注册、检索、组合，以及由失败驱动的改进。
- 把 Voyager 模式映射到 2026 年的 Claude Agent SDK skills 和 skillkit 生态。

## 问题背景

每个会话都从零重建全部能力的智能体，会犯三个错误：

1. **浪费 token。** 每个任务都要重新引出同样的推理。
2. **丢失进展。** 会话 A 中学到的修正无法迁移到会话 B。
3. **搞不定长程组合任务。** 复杂任务需要能力层级；一次性提示词表达不了这种层级。

Voyager 的答案：把每个可复用能力当作一段有名称的代码存进库里，按相似度检索，可与其他技能组合，并通过执行反馈不断改进。

## 核心概念

### 三个组件

Voyager（arXiv:2305.16291）围绕以下三部分构建智能体：

1. **自动课程（automatic curriculum）。** 一个由好奇心驱动的任务提议器，根据智能体当前的技能集合和环境状态选出下一个任务。探索是自底向上的。
2. **技能库（skill library）。** 每个技能都是可执行代码。任务成功时新增技能。检索方式是计算查询与技能描述之间的相似度。
3. **迭代提示机制（iterative prompting mechanism）。** 失败时，智能体收到执行错误、环境反馈和自我验证输出，然后据此改进技能。

Minecraft 评测结果（Wang et al., 2024）：相对基线，独特物品数多 3.3 倍，获得石质工具快 8.5 倍，铁质工具快 6.4 倍，地图探索距离长 2.3 倍。这些数字只针对 Minecraft，但模式是可迁移的。

### 动作空间 = 代码

大多数智能体输出的是原子命令，Voyager 输出的是 JavaScript 函数。一个技能形如：

```
async function craftIronPickaxe(bot) {
  await mineIron(bot, 3);
  await mineStick(bot, 2);
  await placeCraftingTable(bot);
  await craft(bot, 'iron_pickaxe');
}
```

由子技能组合而成。以描述和嵌入（embedding）为键存储。检索出来的是程序，不是提示词。

这正是 2026 年的 Claude Agent SDK skill：一段有名称、可检索的代码加上一份指令，由智能体按需加载。

### 技能检索

来了个新任务"制作一把钻石镐"。智能体会：

1. 对任务描述做嵌入。
2. 在技能库中查询 top-k 相似技能。
3. 检索出 `craftIronPickaxe`、`mineDiamond`、`placeCraftingTable` 等。
4. 用检索到的原语加上新逻辑组合出新技能。

这就是 MCP resources（Phase 13）和 Agent SDK skills 所实现的模式：在一个知识/代码层面上做检索，范围限定在当前任务。

### 迭代改进

Voyager 的反馈循环：

1. 智能体编写一个技能。
2. 技能在环境中运行。
3. 返回三种信号之一：`success`、`error`（带堆栈跟踪）、`self-verification failure`。
4. 智能体以该信号为上下文重写技能。
5. 循环直到成功或达到最大轮数。

这就是 Self-Refine（第 05 课）应用于代码生成，并以环境为依据做验证。CRITIC（第 05 课）是同一模式，只不过用外部工具作为验证器。

### 课程与探索

Voyager 的课程模块会根据智能体已有什么、还没做过什么，提出诸如"在湖边建一个庇护所"这样的任务。提议器利用环境状态加技能清单，挑选刚好略高于当前能力的任务——这正是探索的最佳区间。

对生产环境的智能体来说，这对应一个"还缺什么"操作：给定当前技能库和一个领域，我们还有哪些技能没有覆盖？团队通常以人工课程评审的方式来实现。

### 这种模式会在哪里出错

- **技能库腐化。** 同一个技能以略有差异的描述被添加了 10 次。在写入时做去重；检索只返回一个。
- **组合技能漂移。** 父技能依赖的子技能被改进了。给技能加版本号；固定在 v1 的父技能不会神奇地用上 v3。
- **检索质量。** 库的规模超过几百个技能后，基于技能描述的向量检索会退化。用标签过滤和硬约束来补充（"只要带 `category=tooling` 的技能"）。

## 从零实现

`code/main.py` 用标准库实现了一个技能库：

- `Skill` —— 名称、描述、代码（字符串形式）、版本、标签、依赖。
- `SkillLibrary` —— 注册、搜索（token 重叠度）、组合（对依赖做拓扑排序）、改进（更新时版本号递增）。
- 一个脚本化智能体：注册三个原语技能，组合出第四个，遇到一次失败，然后完成改进。

运行：

```
python3 code/main.py
```

运行轨迹会展示库写入、检索、组合、一次失败的执行和一次 v2 改进——端到端跑通 Voyager 的循环。

## 生产实践

- **Claude Agent SDK skills**（Anthropic）—— 2026 年的参考实现：每个技能包含描述、代码和指令；在智能体会话中按需加载。
- **skillkit**（npm: skillkit）—— 面向 32+ 个 AI 编码智能体的跨智能体技能管理。
- **自建技能库** —— 面向特定领域（数据智能体的 SQL 技能、基础设施智能体的 Terraform 技能）。Voyager 模式可以按需缩小规模。
- **OpenAI Agents SDK `tools`** —— 处于最简一端；每个工具就是一个轻量级技能。

## 交付产物

`outputs/skill-skill-library.md` 可为任意目标运行时生成一个 Voyager 形态的技能库，内置注册、检索、版本管理和改进机制。

## 练习

1. 给 `compose()` 加一个依赖环检测器。当技能 A 依赖 B、B 又依赖 A 时会发生什么？该报错还是警告？
2. 实现逐技能的版本固定（version pinning）。当父技能组合了子技能 `crafting@1` 时，把它改进到 `crafting@2` 不能悄悄升级父技能。
3. 用 sentence-transformers 嵌入（或一个纯标准库实现的 BM25）替换 token 重叠检索。在一个 50 个技能的玩具库上测量 retrieval@5。
4. 加一个"课程"智能体：给定当前库和一份领域描述，提出 5 个缺失的技能。每周调用一次。
5. 阅读 Anthropic 的 Claude Agent SDK skill 文档。把玩具库移植到 SDK 的技能 schema。可发现性会发生什么变化？

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|----------------|------------------------|
| 技能（Skill） | "可复用能力" | 有名称的代码块加描述，可按相似度检索 |
| 技能库 | "智能体的'怎么做'记忆" | 技能的持久化存储，可搜索、可组合 |
| 课程 | "任务提议器" | 由当前能力缺口驱动的自底向上目标生成器 |
| 组合 | "技能 DAG" | 技能调用技能；执行时做拓扑排序 |
| 迭代改进 | "自我修正循环" | 环境反馈 + 错误 + 自我验证回馈到下一个版本 |
| 动作空间即代码 | "程序化动作" | 输出函数而非原子命令，以实现时间上延展的行为 |
| 写入去重 | "技能归并" | 近似重复的描述被归并为一个规范技能 |

## 延伸阅读

- [Wang et al., Voyager (arXiv:2305.16291)](https://arxiv.org/abs/2305.16291) —— 技能库的开山论文
- [Claude Agent SDK overview](https://platform.claude.com/docs/en/agent-sdk/overview) —— skills 作为 2026 年的产品化形态
- [Anthropic, Building agents with the Claude Agent SDK](https://www.anthropic.com/engineering/building-agents-with-the-claude-agent-sdk) —— skills 与子智能体的实战
- [Madaan et al., Self-Refine (arXiv:2303.17651)](https://arxiv.org/abs/2303.17651) —— Voyager 底层的改进循环
